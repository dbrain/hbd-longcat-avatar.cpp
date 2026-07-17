// ultrashape_refine.hpp — INLINE native UltraShape refine, factored from the validated ultrashape_e2e.cpp
// so the image_to_rig driver can call it in-process (no docker, no python, warm weights). Turns a coarse
// pixal3d mesh (in RAM) + the raw matte PNG into a clean/watertight densified mesh:
//
//   coarse mesh + matte.png
//     -> native conditioner (ImageProcessorV2 + DINOv2 + fg-mask token gather)  -> cond_main [1024,Tc]
//     -> in-mem voxelizer (surface-sample -> res-128 occupancy -> K latent voxels) -> voxel_cond [N,3]
//     -> flow-matching Euler sampling (steps, CFG) over us_refine_dit               -> refined latents
//     -> 1/scale -> us_vae_transformer (ONCE) -> us_geo_decoder per query-chunk     -> dense grid logits
//     -> marching cubes (bounds ±1.0)                                               -> svae::Mesh (±1 frame)
//
// The returned mesh is in the UltraShape ±1.0 box (us_mc::scale_to_bbox bounds=1.0). The caller must
// re-canonicalize it onto the pixal3d coarse-mesh bbox before baking the PBR volume (mirrors
// run_pipeline.sh's RP_CANON_TO_DENSE=1). VALIDATED PARITY (ultrashape_e2e.cpp, 2026-06-18/07-16):
// final_latents cos 0.9999, grid cos 0.99995, Chamfer 0.03% bbox-diag; lossless decode-hoist fix banked.
#pragma once
#include "ultrashape_dit.hpp"
#include "ultrashape_vae.hpp"
#include "ultrashape_mc.hpp"
#include "dinov2_graph.hpp"
#include "ultrashape_cond.hpp"
#include "us_image_proc.hpp"
#include "us_voxelize.hpp"
#include "torch_randn.hpp"
#include "sparse_vae.hpp"                 // svae::Mesh
#include "m1_ggml.hpp"                    // M1Harness, new_graph
#include "../../sparse_spike/npy.hpp"     // npy_load (meta.npy: scale_factor, vox_res)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <functional>
#include <unordered_map>

namespace usr {

static inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Drop connected components whose face count is < min_frac of the largest component's. UltraShape's dense
// marching cubes leaves a handful of tiny detached floaters (e.g. the ~44 MC specks on toy2) around the
// main watertight body; this removes them WITHOUT touching the main mesh. It cannot remove limb-tip bits
// that UltraShape has WELDED into the body (those are one component) — for those the fix is an A-pose
// input, not this filter. Vertices are compacted; faces reindexed. min_frac<=0 → no-op.
static inline void drop_small_components(svae::Mesh& m, float min_frac) {
    if (min_frac <= 0.f || m.F == 0) return;
    const int V = m.N;
    std::vector<int> parent(V);
    for (int i = 0; i < V; i++) parent[i] = i;
    std::function<int(int)> find = [&](int x){ while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    auto uni = [&](int a, int b){ int ra = find(a), rb = find(b); if (ra != rb) parent[ra] = rb; };
    for (size_t f = 0; f < m.faces.size(); f += 3) { uni((int)m.faces[f], (int)m.faces[f+1]); uni((int)m.faces[f+1], (int)m.faces[f+2]); }
    std::unordered_map<int,int> fcount;
    for (size_t f = 0; f < m.faces.size(); f += 3) fcount[find((int)m.faces[f])]++;
    int maxc = 0; for (auto& kv : fcount) maxc = std::max(maxc, kv.second);
    const int thresh = (int)std::ceil((double)min_frac * maxc);
    std::vector<int64_t> nf; nf.reserve(m.faces.size());
    for (size_t f = 0; f < m.faces.size(); f += 3)
        if (fcount[find((int)m.faces[f])] >= thresh) { nf.push_back(m.faces[f]); nf.push_back(m.faces[f+1]); nf.push_back(m.faces[f+2]); }
    std::vector<int> remap(V, -1); std::vector<float> nv; int n = 0;
    for (auto& idx : nf) { int ii = (int)idx; if (remap[ii] < 0) { remap[ii] = n++; nv.push_back(m.verts[ii*3]); nv.push_back(m.verts[ii*3+1]); nv.push_back(m.verts[ii*3+2]); } idx = remap[ii]; }
    m.verts = std::move(nv); m.faces = std::move(nf); m.N = n; m.F = (int)(m.faces.size()/3);
}

struct RefineCfg {
    int     octree      = 512;      // dense decode grid (octree_resolution); OCT=256 = the toy default
    int     num_latents = 8192;     // token_num / latent voxels (prod)
    int     steps       = 50;       // flow-matching Euler steps
    float   guidance    = 7.5f;     // CFG scale (pipelines.py single-CFG default)
    int64_t chunk       = 2048;     // dense-decode query chunk (VRAM lever)
    float   drop_frac   = 0.01f;    // drop detached components < this fraction of the largest (MC floaters)
    uint64_t seed       = 0;
    bool    verbose     = true;
    // weight dirs (npy); the DiT F16 GGUF (dit.gguf) is loaded from gguf_dir via PIXAL3D_GGUF_DIR.
    std::string dit_w   = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/dit";
    std::string vae_w   = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/vae";
    std::string cnd_w   = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/conditioner";
    std::string gguf_dir= "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/gguf";       // has dit.gguf (F16)
    std::string meta    = "/mnt/hdd/3d/avatar-shootout/e2e_goldens/meta.npy";           // scale_factor, vox_res
};

// Native conditioner: raw matte PNG -> ImageProcessorV2 -> DINOv2 encode -> fg-mask token gather ->
// cond_main [1024, Tc] (HID fastest). Harness scoped so its GPU weights free before the DiT loads.
static inline std::vector<float> run_conditioner(const std::string& cnd_w, const std::string& img_path,
                                                 bool use_cuda, int& Tc_out) {
    const int OUT = 1022, PATCH = 14, rows = 73, cols = 73, HID = dinov2::HID, SEQ = 1 + rows*cols;
    us_imgproc::Processed pp = us_imgproc::process(img_path.c_str(), 1024, 0.15);   // raw PNG -> proc_image/mask
    const int PS = pp.size;
    std::vector<float> inputs = us_cond::preprocess(pp.image.data(), 3, PS, PS, OUT,
                                                    us_cond::IMAGENET_MEAN, us_cond::IMAGENET_STD);
    M1Harness H(cnd_w, 2048, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t img_ne[4] = {(int64_t)cols*PATCH, (int64_t)rows*PATCH, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);
    NpyArray pe = npy_load(cnd_w + "/main_image_encoder.model.embeddings.position_embeddings.npy");
    std::vector<float> pos = dinov2::interp_pos_embed(pe.f32(), rows, cols, 0.1);
    int64_t pos_ne[4] = {HID, (int64_t)SEQ, 1, 1};
    ggml_tensor* pos_const = H.const_tensor("pos_embed_interp", 2, pos_ne, pos);
    ggml_tensor* out = dinov2::encode(ctx, H, img, pos_const, rows, cols, dinov2::HF, "main_image_encoder.model.");
    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, out);
    H.alloc_and_upload(gf);
    H.upload_input_raw(img, inputs);
    H.compute(gf);
    std::vector<float> hidden((size_t)HID*SEQ);
    ggml_backend_tensor_get(out, hidden.data(), 0, hidden.size()*sizeof(float));
    std::vector<int> valid = us_cond::mask_valid_tokens(pp.mask.data(), PS, PS, OUT, PATCH, true);
    Tc_out = (int)valid.size();
    if (Tc_out <= 0) throw std::runtime_error("UltraShape conditioner produced 0 valid tokens (bad matte?)");
    return us_cond::gather_tokens(hidden.data(), HID, SEQ, valid);
}

// Full refine. Returns the refined mesh in the UltraShape ±1.0 frame. Throws on hard failure.
// PIXAL3D_GGUF_DIR is temporarily pointed at cfg.gguf_dir (so the DiT loads dit.gguf F16; cond/vae have
// no gguf there and fall back to npy) and RESTORED to its prior value on exit.
static inline svae::Mesh refine(const std::vector<float>& coarse_verts,
                                const std::vector<int64_t>& coarse_faces,
                                const std::string& matte_png,
                                const RefineCfg& cfg, bool use_cuda) {
    // ---- run-config constants (scale_factor + occupancy voxel res) from meta.npy ----
    float scale_factor = 1.0f; int vox_res = 128;
    { NpyArray meta = npy_load(cfg.meta); const double* mp = meta.f64();
      scale_factor = (float)mp[6]; vox_res = (int)mp[7]; }
    const int   N     = cfg.num_latents;
    const int   steps = cfg.steps;
    const float GUID  = cfg.guidance;
    const int   OCT   = cfg.octree;
    const int64_t CHUNK = cfg.chunk;
    const float BNDS  = 1.0f;
    if (cfg.verbose)
        printf("  [US] refine: N=%d steps=%d guidance=%.2f octree=%d chunk=%lld scale=%.6f vox_res=%d\n",
               N, steps, GUID, OCT, (long long)CHUNK, scale_factor, vox_res);

    // ---- point PIXAL3D_GGUF_DIR at the UltraShape gguf (dit.gguf F16), restore on scope exit ----
    std::string prev_gguf; bool had_prev = false;
    if (const char* p = std::getenv("PIXAL3D_GGUF_DIR")) { prev_gguf = p; had_prev = true; }
    setenv("PIXAL3D_GGUF_DIR", cfg.gguf_dir.c_str(), 1);
    struct EnvRestore {
        bool had; std::string prev;
        ~EnvRestore(){ if (had) setenv("PIXAL3D_GGUF_DIR", prev.c_str(), 1); else unsetenv("PIXAL3D_GGUF_DIR"); }
    } _restore{had_prev, prev_gguf};

    // ---- conditioner (scoped: GPU weights free before DiT) ----
    double t_c0 = now_s();
    int Tc_i = 0;
    std::vector<float> cond_main = run_conditioner(cfg.cnd_w, matte_png, use_cuda, Tc_i);
    const int64_t Tc = Tc_i;
    std::vector<float> uncond_main((size_t)Tc * dinov2::HID, 0.0f);   // unconditional = zero tokens
    if (cfg.verbose) printf("  [US] conditioner: %lld valid tokens  (%.1fs)\n", (long long)Tc, now_s()-t_c0);

    // ---- voxelizer (in-mem coarse mesh -> voxel_cond [N,3]) ----
    us_vox::VoxelCond vc;
    if (!us_vox::voxelize_mesh_inmem(coarse_verts, coarse_faces, N, vox_res, 409600, cfg.seed, vc))
        throw std::runtime_error("UltraShape in-mem voxelizer failed");
    std::vector<float> voxf((size_t)N*3);
    for (size_t i = 0; i < voxf.size(); i++) voxf[i] = (float)vc.coords[i];
    if (cfg.verbose) printf("  [US] voxelizer: %d occupied -> %d latent voxels\n", vc.n_occupied, N);

    // ---- init latents + sigma schedule (flow-matching, shift=1) ----
    trandn::Generator g(cfg.seed);
    std::vector<float> x = g.randn((int64_t)N * 64);                  // init_latents [k*64+c]
    std::vector<float> sigmas(steps + 1);
    for (int i = 0; i < steps; i++) sigmas[i] = (float)i / (steps - 1);
    sigmas[steps] = 1.0f;

    // ---- DiT sampling (scope frees the 6.1GB F16 weights before the VAE harness) ----
    {
        UsDitCfg dcfg;
        const int hidden = dcfg.hidden, ich = dcfg.in_channels;
        M1Harness Hd(cfg.dit_w, 8192, use_cuda);
        ggml_context* cd = Hd.ctx;
        std::vector<float> rcos, rsin;
        us_rope_3d(voxf.data(), N, dcfg, rcos, rsin);
        int dim = dcfg.head_dim(); int64_t S = 1 + N;
        int64_t rope_ne[4] = {dim, 1, S, 1};
        ggml_tensor* rcos_t = Hd.const_tensor("rope_cos", 3, rope_ne, rcos);
        ggml_tensor* rsin_t = Hd.const_tensor("rope_sin", 3, rope_ne, rsin);
        int64_t ts_ne[4] = {hidden, 1, 1, 1};
        ggml_tensor* ts_t = Hd.input("ts_embed", 2, ts_ne);
        int64_t x_ne[4] = {ich, N, 1, 1};
        ggml_tensor* x_lat = Hd.input("x_lat", 2, x_ne);
        int64_t c_ne[4] = {dcfg.context_dim, Tc, 1, 1};
        ggml_tensor* cond = Hd.input("cond", 2, c_ne);
        ggml_tensor* dout = us_refine_dit(Hd, cd, dcfg, x_lat, ts_t, cond, rcos_t, rsin_t);  // [64,N]
        ggml_set_output(dout);
        ggml_cgraph* gd = new_graph(cd, 65536);
        ggml_build_forward_expand(gd, dout);
        Hd.alloc_and_upload(gd);

        std::vector<float> pred_c((size_t)N*ich), pred_u((size_t)N*ich);
        double t_s0 = now_s();
        for (int i = 0; i < steps; i++) {
            float tnorm = sigmas[i];   // t/num_train == sigmas[i] (shift=1)
            std::vector<float> ts = us_timesteps_embed(tnorm, hidden);
            // cond pass — re-upload ALL inputs before each compute (gallocr input-clobber, PERF note).
            Hd.upload_input_raw(ts_t, ts);
            Hd.upload_input_raw(x_lat, x);
            Hd.upload_input_raw(cond, cond_main);
            Hd.compute(gd);
            ggml_backend_tensor_get(dout, pred_c.data(), 0, pred_c.size()*sizeof(float));
            // uncond pass (zero cond tokens)
            Hd.upload_input_raw(ts_t, ts);
            Hd.upload_input_raw(x_lat, x);
            Hd.upload_input_raw(cond, uncond_main);
            Hd.compute(gd);
            ggml_backend_tensor_get(dout, pred_u.data(), 0, pred_u.size()*sizeof(float));
            float dsig = sigmas[i+1] - sigmas[i];
            for (size_t j = 0; j < x.size(); j++) {
                float pred = pred_u[j] + GUID*(pred_c[j]-pred_u[j]);
                x[j] += dsig * pred;
            }
            if (cfg.verbose && (i%10==0 || i==steps-1)) {
                double m=0; for (float v:x) m+=std::fabs(v); m/=x.size();
                printf("  [US] step %2d/%d t=%.4f |x|=%.4f\n", i, steps, tnorm, m);
            }
        }
        if (cfg.verbose)
            printf("  [US] DiT sampling: %d steps x2(CFG) over N=%d = %.1fs (%.2fs/step)\n",
                   steps, N, now_s()-t_s0, (now_s()-t_s0)/std::max(1,steps));
    }   // Hd freed here

    // ---- export: 1/scale -> vae transformer (once) -> geo_decoder per chunk ----
    std::vector<float> lat_scaled(x.size());
    for (size_t j = 0; j < x.size(); j++) lat_scaled[j] = x[j] / scale_factor;

    UsVaeCfg vcfg;
    std::vector<float> tr_host((size_t)vcfg.width * N);
    {
        M1Harness Ht(cfg.vae_w, 4096, use_cuda);
        int64_t vlat_ne[4] = {vcfg.embed_dim, N, 1, 1};
        ggml_tensor* vlat = Ht.input("vlatents", 2, vlat_ne);
        ggml_tensor* tr = us_vae_transformer(Ht, Ht.ctx, vcfg, vlat);        // [width, N]
        ggml_set_output(tr);
        ggml_cgraph* gt = new_graph(Ht.ctx, 32768);
        ggml_build_forward_expand(gt, tr);
        Ht.alloc_and_upload(gt);
        Ht.upload_input_raw(vlat, lat_scaled);
        double t_tr0 = now_s();
        Ht.compute(gt);
        ggml_backend_tensor_get(tr, tr_host.data(), 0, tr_host.size()*sizeof(float));
        if (cfg.verbose) printf("  [US] VAE transformer (16 layers over N=%d, once): %.1fs\n", N, now_s()-t_tr0);
    }   // Ht freed

    M1Harness Hv(cfg.vae_w, 4096, use_cuda);
    ggml_context* cv = Hv.ctx;
    int64_t tr_ne[4] = {vcfg.width, N, 1, 1};
    ggml_tensor* tr_in = Hv.const_tensor("tr_cached", 2, tr_ne, std::move(tr_host));
    int64_t qe_ne[4] = {vcfg.fourier_out(), CHUNK, 1, 1};
    ggml_tensor* query_embed = Hv.input("query_embed", 2, qe_ne);
    ggml_tensor* occ = us_geo_decoder(Hv, cv, vcfg, query_embed, tr_in);     // [1,CHUNK]
    ggml_set_output(occ);
    ggml_cgraph* gv = new_graph(cv, 32768);
    ggml_build_forward_expand(gv, occ);
    Hv.alloc_and_upload(gv);

    int G;
    std::vector<float> grid = us_dense_grid_queries(OCT, BNDS, G);
    int64_t Ngrid = (int64_t)G*G*G;
    std::vector<float> logits(Ngrid, 0.0f);
    double t_d0 = now_s();
    if (cfg.verbose)
        printf("  [US] VAE dense decode: G=%d -> %lld queries / %lld-chunk = %lld computes ...\n",
               G, (long long)Ngrid, (long long)CHUNK, (long long)((Ngrid+CHUNK-1)/CHUNK));
    for (int64_t s = 0; s < Ngrid; s += CHUNK) {
        int64_t n = std::min(CHUNK, Ngrid-s);
        std::vector<float> qchunk((size_t)CHUNK*3, 0.0f);
        std::copy(grid.begin()+s*3, grid.begin()+(s+n)*3, qchunk.begin());
        std::vector<float> qe = us_fourier_embed(qchunk.data(), CHUNK, vcfg);
        Hv.upload_input_raw(query_embed, qe);
        Hv.compute(gv);
        std::vector<float> got(CHUNK);
        ggml_backend_tensor_get(occ, got.data(), 0, CHUNK*sizeof(float));
        std::copy(got.begin(), got.begin()+n, logits.begin()+s);
    }
    if (cfg.verbose) printf("  [US] VAE dense decode: %.1fs\n", now_s()-t_d0);

    // ---- marching cubes (bounds ±1.0) -> svae::Mesh (UltraShape frame) ----
    // `logits` are OCCUPANCY logits: inside = POSITIVE (sigmoid>0.5 <=> logit>0). us_mc defaults to
    // that convention (INSIDE_ABOVE_ISO) and therefore emits OUTWARD normals. Feeding an inside-
    // positive field to the old inside-negative predicate is what inverted the winding of every mesh
    // downstream of here — see the WINDING block in ultrashape_mc.hpp. USR_FIX_WINDING=0 to A/B.
    us_mc::Mesh m = us_mc::marching_cubes(logits.data(), G, 0.0f);
    us_mc::scale_to_bbox(m.verts, G, BNDS);
    svae::Mesh out;
    out.verts = std::move(m.verts);
    out.faces = std::move(m.faces);
    out.N = (int)(out.verts.size()/3);
    out.F = (int)(out.faces.size()/3);

    // ---- QC: THE check that was missing. A closed, outward-wound mesh has POSITIVE signed volume.
    // Nothing else in this pipeline could see the flip: the cubvh golden test compares faces as
    // SORTED position triples (winding-blind), Chamfer/cos parity are orientation-blind, and every
    // material is doubleSided:true so it even LOOKS right. This one number sees it. O(F), free.
    {
        double cx=0, cy=0, cz=0;
        for (int i = 0; i < out.N; i++) { cx += out.verts[i*3]; cy += out.verts[i*3+1]; cz += out.verts[i*3+2]; }
        if (out.N) { cx /= out.N; cy /= out.N; cz /= out.N; }
        double vol = 0.0;
        for (size_t f = 0; f < out.faces.size(); f += 3) {
            const float *p0=&out.verts[out.faces[f]*3], *p1=&out.verts[out.faces[f+1]*3], *p2=&out.verts[out.faces[f+2]*3];
            const double a[3]={p0[0]-cx,p0[1]-cy,p0[2]-cz}, b[3]={p1[0]-cx,p1[1]-cy,p1[2]-cz}, c[3]={p2[0]-cx,p2[1]-cy,p2[2]-cz};
            const double cr[3]={b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]};
            vol += a[0]*cr[0] + a[1]*cr[1] + a[2]*cr[2];
        }
        vol /= 6.0;
        if (vol < 0.0)
            fprintf(stderr,
                "  [US] *** QC: refined mesh signed volume = %.6g < 0 -> INWARD/INVERTED WINDING ***\n"
                "  [US]     Normals point INTO the model. Masked by doubleSided:true materials, but it\n"
                "  [US]     breaks texture projection, backface culling, normal-map bakes, physics and\n"
                "  [US]     3D printing. Expected cause: the grid_logits sign convention no longer matches\n"
                "  [US]     us_mc's inside predicate (see ultrashape_mc.hpp WINDING block; USR_FIX_WINDING).\n",
                vol);
        else if (cfg.verbose)
            printf("  [US] QC: signed volume = %+.6g (>0 = OUTWARD winding, correct)\n", vol);
    }
    // drop tiny detached MC floaters (env USR_DROP_FRAC overrides; 0 = keep all, for A/B).
    float dfrac = cfg.drop_frac;
    if (const char* e = std::getenv("USR_DROP_FRAC")) dfrac = (float)atof(e);
    if (dfrac > 0.f) {
        int f0 = out.F, v0 = out.N;
        drop_small_components(out, dfrac);
        if (cfg.verbose && out.F != f0)
            printf("  [US] drop floaters (<%.1f%% of largest): verts %d->%d, faces %d->%d\n",
                   dfrac*100.f, v0, out.N, f0, out.F);
    }
    if (cfg.verbose) printf("  [US] refined mesh: verts=%d faces=%d (G=%d)\n", out.N, out.F, G);
    return out;
}

}  // namespace usr
