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
    std::string dit_w   = "/home/dbrain/models/3d/refine/npy/dit";
    std::string vae_w   = "/home/dbrain/models/3d/refine/npy/vae";
    std::string cnd_w   = "/home/dbrain/models/3d/refine/npy/conditioner";
    std::string gguf_dir= "/home/dbrain/models/3d/refine/gguf";    // dit.gguf = Q8_0 (default; −44% time/−40% VRAM, near-lossless). gguf/ = F16 fallback (--us-gguf)
    std::string meta    = "/home/dbrain/models/3d/refine/meta.npy";           // scale_factor, vox_res
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

    // ------------------------------------------------------------------------------------------
    // USR_SPARSE_DECODE (default ON; =0 restores the dense sweep for A/B) -- close the port gap
    // ultrashape_e2e.cpp:84 admits: "DENSE grid here; Python uses a hierarchical octree".
    //
    // The dense sweep evaluates all G^3 corners to find an isosurface that MEASURED occupies 0.341%
    // of them (voxelising refined.glb: 457,147 of 134M; surface-area cross-check 0.366%). So >99% of
    // the most expensive stage in the pipeline decodes empty space or solid interior.
    //
    // Construction (the reference's, NOT a coarse band -- a band alone is NOT lossless: a +-4 band
    // still misses 0.40% of isosurface cells => holes):
    //   (1) decode a stride-S coarse lattice in full;
    //   (2) any coarse CELL whose 8 corners straddle the iso is subdivided -- every fine corner in it
    //       becomes active. Dilated by one coarse cell so a straddle never sits on the active border;
    //   (3) UNION a +-BAND fine-cell shell around the coarse mesh's own vertices, which catches
    //       features thinner than a coarse cell (fingers, straps) that (2) cannot see;
    //   (4) undecoded corners inherit their coarse cell's uniform sign as +-BIG. Marching cubes is
    //       UNCHANGED: an all-same-sign cell emits no triangle, so filled regions are inert.
    // Verified by the QC the pipeline already prints (signed volume / boundary) + a mesh A/B vs the
    // dense path at the same octree -- NOT by timing.
    // ------------------------------------------------------------------------------------------
    const bool  sparse   = [](){ const char* e = getenv("USR_SPARSE_DECODE"); return !e || atoi(e) != 0; }();
    const int   S        = [](){ const char* e = getenv("USR_SPARSE_STRIDE"); return e ? std::max(2, atoi(e)) : 4; }();
    const int   BAND     = [](){ const char* e = getenv("USR_SPARSE_BAND");   return e ? std::max(0, atoi(e)) : 2; }();
    const float BIG      = 1e4f;

    std::vector<int64_t> active_idx;      // fine-grid linear indices still to decode
    std::vector<uint8_t> decoded;         // 1 = logits[p] holds a real decoded value
    if (sparse) {
        double t_a0 = now_s();
        decoded.assign((size_t)Ngrid, 0);
        std::vector<uint8_t> act((size_t)Ngrid, 0);
        const int GC = (G - 1) / S + 1;                  // coarse lattice points per axis
        auto FIDX = [&](int64_t i, int64_t j, int64_t k) { return (i*G + j)*G + k; };

        // (1) coarse lattice -> active
        for (int ci = 0; ci < GC; ++ci)
          for (int cj = 0; cj < GC; ++cj)
            for (int ck = 0; ck < GC; ++ck)
              act[(size_t)FIDX(std::min(ci*S, G-1), std::min(cj*S, G-1), std::min(ck*S, G-1))] = 1;
        for (int64_t p = 0; p < Ngrid; ++p) if (act[(size_t)p]) active_idx.push_back(p);
        if (cfg.verbose)
            printf("  [US] sparse decode: L0 stride=%d -> %zu coarse queries (%.3f%% of %lld)\n",
                   S, active_idx.size(), 100.0*(double)active_idx.size()/(double)Ngrid, (long long)Ngrid);
        (void)t_a0;
    }
    if (cfg.verbose)
        printf("  [US] VAE %s decode: G=%d -> %lld queries / %lld-chunk = %lld computes ...\n",
               sparse ? "sparse" : "dense", G, (long long)Ngrid, (long long)CHUNK,
               (long long)((Ngrid+CHUNK-1)/CHUNK));
    // Per-phase accounting for the decode (the pipeline's single most expensive stage). A prod run
    // measured 38.37 ms/chunk while a prod-shaped micro-bench of the SAME fixed-shape graph measured
    // 27.4 ms/chunk -- a 1.4x gap that host-overhead, clock throttling and attention tiling were each
    // measured and eliminated as explanations. These four counters split the loop so the gap is
    // attributed instead of guessed. Cost: 4 doubles + one clock_gettime per phase.
    double th_prep = 0, th_up = 0, th_gpu = 0, th_dl = 0;
    int64_t n_queries = 0;
    // Decode an arbitrary list of fine-grid linear indices, chunked. The tail of the last chunk is
    // zero-padded and its results are ignored (identical to the dense loop's tail handling).
    std::vector<float> qchunk((size_t)CHUNK*3, 0.0f);   // hoisted: the dense loop re-allocated these
    std::vector<float> got((size_t)CHUNK);              // ~198k heap allocs over a prod run
    auto decode_indices = [&](const std::vector<int64_t>& idx) {
        for (size_t s = 0; s < idx.size(); s += (size_t)CHUNK) {
            const size_t n = std::min((size_t)CHUNK, idx.size() - s);
            double a = now_s();
            std::fill(qchunk.begin(), qchunk.end(), 0.0f);
            for (size_t t = 0; t < n; ++t) {
                const int64_t p = idx[s+t];
                qchunk[t*3+0] = grid[(size_t)p*3+0];
                qchunk[t*3+1] = grid[(size_t)p*3+1];
                qchunk[t*3+2] = grid[(size_t)p*3+2];
            }
            std::vector<float> qe = us_fourier_embed(qchunk.data(), CHUNK, vcfg);
            double b = now_s(); th_prep += b - a;
            Hv.upload_input_raw(query_embed, qe);
            double c = now_s(); th_up += c - b;
            Hv.compute(gv);
            double d = now_s(); th_gpu += d - c;
            ggml_backend_tensor_get(occ, got.data(), 0, CHUNK*sizeof(float));
            for (size_t t = 0; t < n; ++t) {
                logits[(size_t)idx[s+t]] = got[t];
                if (!decoded.empty()) decoded[(size_t)idx[s+t]] = 1;
            }
            th_dl += now_s() - d;
            n_queries += (int64_t)n;
        }
    };

    if (!sparse) {
        for (int64_t s = 0; s < Ngrid; s += CHUNK) {
            int64_t n = std::min(CHUNK, Ngrid-s);
            double a = now_s();
            std::fill(qchunk.begin(), qchunk.end(), 0.0f);
            std::copy(grid.begin()+s*3, grid.begin()+(s+n)*3, qchunk.begin());
            std::vector<float> qe = us_fourier_embed(qchunk.data(), CHUNK, vcfg);
            double b = now_s(); th_prep += b - a;
            Hv.upload_input_raw(query_embed, qe);
            double c = now_s(); th_up += c - b;
            Hv.compute(gv);
            double d = now_s(); th_gpu += d - c;
            ggml_backend_tensor_get(occ, got.data(), 0, CHUNK*sizeof(float));
            std::copy(got.begin(), got.begin()+n, logits.begin()+s);
            th_dl += now_s() - d;
            n_queries += n;
        }
    } else {
        const int GC = (G - 1) / S + 1, NC = GC - 1;
        auto FIDX = [&](int64_t i, int64_t j, int64_t k) { return (i*G + j)*G + k; };
        auto CLI  = [&](int v) { return std::min(v, G-1); };

        decode_indices(active_idx);                       // (1) L0 coarse lattice
        const size_t n_l0 = active_idx.size();

        std::vector<uint8_t> act((size_t)Ngrid, 0);
        auto mark_box = [&](int i0, int i1, int j0, int j1, int k0, int k1) {
            for (int i = std::max(0,i0); i <= std::min(G-1,i1); ++i)
              for (int j = std::max(0,j0); j <= std::min(G-1,j1); ++j)
                for (int k = std::max(0,k0); k <= std::min(G-1,k1); ++k)
                  act[(size_t)FIDX(i,j,k)] = 1;
        };

        // (2) subdivide sign-straddling coarse cells, dilated by one coarse cell so a straddle can
        //     never sit exactly on the active-set border.
        size_t n_mixed = 0;
        for (int ci = 0; ci < NC; ++ci)
          for (int cj = 0; cj < NC; ++cj)
            for (int ck = 0; ck < NC; ++ck) {
                bool pos = false, neg = false;
                for (int dx = 0; dx < 2; ++dx)
                  for (int dy = 0; dy < 2; ++dy)
                    for (int dz = 0; dz < 2; ++dz) {
                        const float v = logits[(size_t)FIDX(CLI((ci+dx)*S), CLI((cj+dy)*S), CLI((ck+dz)*S))];
                        (v >= 0.0f ? pos : neg) = true;
                    }
                if (pos && neg) {
                    ++n_mixed;
                    mark_box(ci*S - S, ci*S + 2*S, cj*S - S, cj*S + 2*S, ck*S - S, ck*S + 2*S);
                }
            }

        // (3) UNION a +-BAND fine shell around the coarse mesh itself -- catches features thinner
        //     than a coarse cell, which (2) is blind to. FRAME: us_vox::voxelize_mesh_inmem
        //     normalises ITS OWN COPY (longest axis -> [-1,1], then *0.99); the caller hands us the
        //     pixal [-0.5,0.5] frame, NOT the grid's +-1. Replicate exactly or the band lands in the
        //     wrong place and silently makes holes.
        size_t n_band = 0;
        if (BAND > 0) {
            std::vector<float> bv = coarse_verts;
            rig::normalize_mesh(bv);
            for (auto& v : bv) v *= 0.99f;
            const float inv = (float)(G - 1) / (2.0f * BNDS);
            for (size_t v = 0; v + 2 < bv.size(); v += 3) {
                const int gi = (int)std::lround((bv[v+0] + BNDS) * inv);
                const int gj = (int)std::lround((bv[v+1] + BNDS) * inv);
                const int gk = (int)std::lround((bv[v+2] + BNDS) * inv);
                mark_box(gi-BAND, gi+BAND, gj-BAND, gj+BAND, gk-BAND, gk+BAND);
                ++n_band;
            }
        }

        // (3b) Dilate the active set by ONE fine cell. MC does not just test signs, it INTERPOLATES
        //      between corners: t = (iso - v0)/(v1 - v0). A straddling cell with one decoded corner
        //      and one FILLED corner would interpolate against the +-BIG sentinel and displace the
        //      vertex. The +-S straddle margin is wide enough, but the vertex band has a hard edge --
        //      MEASURED: without this, 2 of 630,300 verts moved (max 0.023 of a cell) at octree 512.
        //      Dilating guarantees every cell touching the active set has all 8 corners decoded, so
        //      no interpolation ever reads a sentinel.
        {
            std::vector<uint8_t> act2 = act;
            for (int64_t i = 0; i < G; ++i)
              for (int64_t j = 0; j < G; ++j)
                for (int64_t k = 0; k < G; ++k) {
                    if (!act[(size_t)FIDX(i,j,k)]) continue;
                    for (int di = -1; di <= 1; ++di)
                      for (int dj = -1; dj <= 1; ++dj)
                        for (int dk = -1; dk <= 1; ++dk) {
                            const int64_t ni = i+di, nj = j+dj, nk = k+dk;
                            if (ni < 0 || nj < 0 || nk < 0 || ni >= G || nj >= G || nk >= G) continue;
                            act2[(size_t)FIDX(ni,nj,nk)] = 1;
                        }
                }
            act.swap(act2);
        }

        std::vector<int64_t> refine_idx;
        for (int64_t p = 0; p < Ngrid; ++p) if (act[(size_t)p] && !decoded[(size_t)p]) refine_idx.push_back(p);
        if (cfg.verbose)
            printf("  [US] sparse decode: %zu straddling cells + band over %zu verts -> %zu refine queries\n",
                   n_mixed, n_band, refine_idx.size());
        decode_indices(refine_idx);                       // (2)+(3)

        // (4) undecoded corners inherit their coarse cell's uniform sign. MC is unchanged: an
        //     all-same-sign cell emits nothing, so these regions are inert.
        int64_t n_fill = 0;
        for (int64_t i = 0; i < G; ++i)
          for (int64_t j = 0; j < G; ++j)
            for (int64_t k = 0; k < G; ++k) {
                const int64_t p = FIDX(i,j,k);
                if (decoded[(size_t)p]) continue;
                const float pv = logits[(size_t)FIDX(CLI((int)(i/S)*S), CLI((int)(j/S)*S), CLI((int)(k/S)*S))];
                logits[(size_t)p] = (pv >= 0.0f) ? BIG : -BIG;
                ++n_fill;
            }
        if (cfg.verbose)
            printf("  [US] sparse decode: L0 %zu + refine %zu = %lld decoded (%.3f%% of %lld), %lld filled\n",
                   n_l0, refine_idx.size(), (long long)n_queries,
                   100.0*(double)n_queries/(double)Ngrid, (long long)Ngrid, (long long)n_fill);
    }
    if (cfg.verbose) {
        const double tot = now_s()-t_d0;
        const long long nch = std::max<long long>(1, (n_queries+CHUNK-1)/CHUNK);   // ACTUAL chunks run
        printf("  [US] VAE %s decode: %.1fs\n", sparse ? "sparse" : "dense", tot);
        printf("  [US]   decode split over %lld chunks (%.3f ms/chunk): prep %.1fs (%.1f%%) "
               "upload %.1fs (%.1f%%) compute %.1fs (%.1f%%) readback %.1fs (%.1f%%)\n",
               nch, 1000.0*tot/(double)nch,
               th_prep, 100*th_prep/tot, th_up, 100*th_up/tot,
               th_gpu, 100*th_gpu/tot, th_dl, 100*th_dl/tot);
    }

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
