// Shared geometry chain: image (preprocessed, [0,1] CHW @512 + @1024) -> untextured mesh.
// Extracted from the validated geometry_e2e.cpp so the CLI (pixal3d) and the validation
// harness call the SAME code. Weights load from per-tensor .npy or (PIXAL3D_GGUF_DIR set)
// the per-model GGUFs — bit-identical (A2). See geometry_e2e.cpp for the per-stage validation.
#pragma once
#include "dinov3_graph.hpp"
#include "naf_graph.hpp"
#include "ss_dit_graph.hpp"
#include "ss_vae_graph.hpp"
#include "slat_dit_graph.hpp"
#include "tex_dit_cross.hpp"     // STAGE-2 texture: TRELLIS-2 cross-mode tex DiT (full-token DINOv3 cond)
#include "geometry_e2e.hpp"
#include "sparse_vae_pipeline.hpp"
#include "svp_gpu.hpp"            // GPU-resident decode (Phase C #1): linear→cuBLAS, feats resident
#include <cuda_profiler_api.h>    // PIXAL3D_NSYS_{M3B,TEX}: nsys/ncu --capture-range=cudaProfilerApi
#include "remesh.hpp"            // A1: marching-tetrahedra manifold watertight remesh
#include "qem.hpp"               // lap-17: feature-preserving QEM decimation of the dual-grid mesh
#include "tex_grid_sample.hpp"   // lap-17: per-vertex PBR grid_sample for the remeshed mesh
#include "torch_randn.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

namespace pix {

// model subdir basenames (the GGUF lookup uses the basename; "weights_npy/" is the .npy fallback)
static const char* DINO_W = "weights_npy/dinov3";
static const char* NAF_W  = "weights_npy/naf";
static const char* SSF_W  = "weights_npy/ss_flow";
static const char* SSD_W  = "weights_npy/ss_dec";
static const char* SL512_W = "weights_npy/slat_flow_512";
static const char* SL1024_W = "weights_npy/slat_flow_1024";
static const char* SHAPEDEC_W = "weights_npy/shape_dec";
static const char* TEXFLOW_W = "weights_npy/trellis2_tex_1024";             // M6 tex DiT (CROSS-mode, in_ch 64, full-token DINOv3 cond)
// PROJ-MODE tex DiT — pixal3d's OWN `slat_flow_imgshape2tex_1024`, i.e. the model that Python's
// Pixal3DImageTo3DPipeline.run() actually runs (Trellis2TexturingPipeline is dead code in that repo,
// so the cross-mode TEXFLOW_W above is a DIFFERENT model that only the tex_goldens came from).
// Arch = the geometry slat DiT VERBATIM (cross_attn.cross_attn_block.* over global[5,1024] +
// proj_linear over proj_cond[2048,M]) with input_layer 64ch instead of 32 — so it needs NO new graph:
// slatdit::build_slat_dit_forward runs it as-is (verified: tensor-name sets of slat_flow_imgshape2tex_1024
// and slat_flow_1024 are identical bar input_layer.weight [1536,64] vs [1536,32]). Its cond is EXACTLY
// the stage3b cond this chain already computes (golden stage3b_cond == stage4_cond byte-for-byte).
// tex_slat latent space + normalization + tex decoder are SHARED with cross-mode (refs/tex_slat_norm_*
// == the Trellis-2 texturing json constants, and m6_tex_sampler_test samples THIS model with them).
static const char* TEXFLOW_PROJ_W = "weights_npy/slat_flow_imgshape2tex_1024";
static const char* TEXDEC_W = "weights_npy/tex_dec";                        // M6 tex decoder (out 6) — shared by both modes

static inline double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// DINOv3: normalized image CHW [3,IMG,IMG] -> global[5*1024] + patchmap[NP*1024] BHWC
static inline void run_dinov3(const std::vector<float>& img_norm, const dino::Cfg& cfg, bool use_cuda,
                              std::vector<float>& global_out, std::vector<float>& patchmap_out) {
    M1Harness H(DINO_W, 1024, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t img_ne[4] = {cfg.IMG, cfg.IMG, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);
    int64_t cs_ne[4] = {dino::HEAD_DIM, 1, cfg.NP(), 1};
    ggml_tensor* cosT = H.input("rope_cos", 3, cs_ne), *sinT = H.input("rope_sin", 3, cs_ne);
    ggml_tensor *global = nullptr, *patchmap = nullptr;
    dino::build_dinov3(ctx, H, img, cosT, sinT, &global, &patchmap, cfg);
    ggml_cgraph* gf = new_graph(ctx, 16384);
    ggml_build_forward_expand(gf, patchmap);
    ggml_build_forward_expand(gf, global);
    H.alloc_and_upload(gf);
    H.upload_input_raw(img, img_norm);
    std::vector<float> cosb, sinb; dino::rope_cos_sin(cosb, sinb, cfg);
    H.upload_input_raw(cosT, cosb); H.upload_input_raw(sinT, sinb);
    H.compute(gf);
    global_out.resize(ggml_nelements(global));
    patchmap_out.resize(ggml_nelements(patchmap));
    ggml_backend_tensor_get(global, global_out.data(), 0, global_out.size()*4);
    ggml_backend_tensor_get(patchmap, patchmap_out.data(), 0, patchmap_out.size()*4);
}

// NAF: guide CHW [3,IMG_IN,IMG_IN] [0,1] + patchmap[SRC*SRC*1024] BHWC -> hr[1024*512*512] BCHW
static inline std::vector<float> run_naf(const std::vector<float>& guide_raw, const std::vector<float>& patchmap,
                                         const naf::Cfg& cfg, bool use_cuda) {
    M1Harness H(NAF_W, 2048, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t g_ne[4] = {cfg.IMG_IN, cfg.IMG_IN, 3, 1};  ggml_tensor* guide = H.input("guide", 4, g_ne);
    int64_t v_ne[4] = {cfg.SRC, cfg.SRC, naf::VC, 1};  ggml_tensor* vin = H.input("vin", 4, v_ne);
    ggml_tensor* hr = naf::build_naf_forward(ctx, H, guide, vin, nullptr, cfg);
    ggml_set_output(hr);
    ggml_cgraph* gf = new_graph(ctx, 16384);
    ggml_build_forward_expand(gf, hr);
    H.alloc_and_upload(gf);
    H.upload_input_raw(guide, guide_raw);
    const int SRC = cfg.SRC, VC = naf::VC;
    std::vector<float> vd((size_t)SRC*SRC*VC);
    for (int hh=0; hh<SRC; hh++) for (int ww=0; ww<SRC; ww++) for (int c=0; c<VC; c++)
        vd[(size_t)c*SRC*SRC + (size_t)hh*SRC + ww] = patchmap[((size_t)hh*SRC+ww)*VC + c];
    H.upload_input_raw(vin, vd);
    H.compute(gf);
    std::vector<float> out(ggml_nelements(hr));
    ggml_backend_tensor_get(hr, out.data(), 0, out.size()*4);
    return out;
}

static inline std::vector<float> imagenet_norm(const std::vector<float>& raw, int IMG) {
    std::vector<float> n = raw;
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    const size_t HW = (size_t)IMG*IMG;
    for (int c=0;c<3;c++) for (size_t i=0;i<HW;i++) n[(size_t)c*HW+i] = (n[(size_t)c*HW+i]-mean[c])/sd[c];
    return n;
}

// Per-stage Trellis.2/Pixal3D sampler config (the knobs Python's run_inference exposes; see
// inference.py). guidance = "how close to the image" (CFG strength: higher = more faithful to the
// input, less invention); rescale = guidance_rescale (std re-normalization, 0..1); rescale_t = the
// Möbius time-warp; [lo,hi] = the guidance interval; steps = sampler iterations (detail/speed).
struct StageSampler { float guidance, rescale, rescale_t; double lo, hi; int steps; };

struct ChainInput {
    std::vector<float> img512_raw, img1024_raw;   // [0,1] CHW (DINOv3 input is normalized internally)
    float cam, dist, ms;                          // camera scalars (radians / units); ms = mesh_scale
    std::vector<float> norm_mean, norm_std;       // shape_slat denorm [32]
    std::vector<float> tex_mean, tex_std;         // tex_slat denorm [32] (only if textured)
    bool use_cuda = true;
    bool verbose = true;
    bool textured = false;                        // M6: also run the tex branch -> per-vertex color
    // Which tex DiT the texture branch runs. false (DEFAULT) = today's behaviour exactly: the CROSS-mode
    // trellis2_tex_1024 (full 4101-token DINOv3 cross-attn cond). true = PROJ-mode slat_flow_imgshape2tex_1024,
    // the model pixal3d's own Python pipeline runs (5 global tokens + proj_linear over the stage3b proj cond).
    // Everything downstream (denorm, tex decoder, bake) is identical — this swaps ONLY the generative model,
    // so it is a clean A/B. tex_flow_w overrides the weight dir/basename for whichever mode is selected.
    // DEFAULT = true (PROJ-mode) since 2026-07-17. Owner-judged on the eye-test A/B (proj vs cross,
    // matched F16 weights, seed 42, identical mesh both sides): "they all look about the same with
    // slightly different lighting... the proj-mode (gilly's model) is the very very mildly best... so
    // yeah default should be proj". It is ALSO what committed HEAD always shipped -- cross-mode was an
    // uncommitted working-tree swap (`git log -S'trellis2_tex_1024'` is empty) -- and it is ~12% faster
    // (tex DiT 157.7s vs 178.7s; 5 KV tokens vs 4101).
    // The A/B settled a bigger question: the model swap is NOT the cause of the texture quality gap.
    // That was the bake fraying material boundaries (fixed by --tex-volume-direct: 5.13% -> 0.74% bad
    // texels) plus the subject (gilly is anime cel-shaded, ~95% flat colour fields, so the volume's
    // band-limiting costs her almost nothing). `--tex-dit cross` A/Bs back.
    bool tex_proj = true;
    std::string tex_flow_w;                       // "" = the mode's default (TEXFLOW_W / TEXFLOW_PROJ_W)
    bool watertight = true;                       // A1: close the dual-grid holes (interim hack path)
    bool remesh = false;                          // A1: QEM-decimate the dual-grid mesh (non-manifold-tolerant)
    // MC manifold remesh (A1, the de-spike fix): re-mesh the grid-`resolution` occupancy with the
    // flood-fill-solid marching-cubes (remesh.hpp marching_cubes_solid) -> WATERTIGHT MANIFOLD surface
    // (boundary==0, nonmanifold==0), then Taubin-smooth. This replaces the raw O-Voxel dual-grid mesh
    // (~162k non-manifold edges + ~58k boundary) that was strangling the xatlas atlas (>400s, tens of
    // thousands of charts) and rendering spikey. The texture is sampled from the per-voxel PBR volume
    // positionally (texatlas::bake grid_sample), so the new manifold verts are fine. Validated
    // standalone (remesh_test): stride 2 / blur 1 / smooth 1 = manifold in ~17s; precluster bake = ~18s.
    bool mc_remesh = false;
    int  mc_stride = 2;                           // fine voxels per coarse marching cell (grid = resolution/stride)
    int  mc_blur   = 1;                           // box-blur radius on the binary solid field (smoother iso surface)
    int  mc_post_smooth = 1;                      // Taubin low-pass iterations on the manifold mesh
    // sampler conditioning (defaults == inference.py run_inference defaults). seed = noise variation.
    int seed = 42;
    StageSampler ss    {7.5f, 0.7f, 5.0f, 0.6, 1.0, 12};   // sparse-structure (SS) DiT
    StageSampler shape {7.5f, 0.5f, 3.0f, 0.6, 1.0, 12};   // shape_slat DiT (M2 lr + M3b hr share this)
    StageSampler tex   {1.0f, 0.0f, 3.0f, 0.6, 0.9, 12};   // tex_slat DiT (CFG off: guidance 1.0)
    // image_resolution (512) is structurally baked into the cond config (CFG512/CFG1024 graphs);
    // max_num_tokens (Python's complex-asset OOM guard) is superseded by query-tiled attention (A2).
    int max_num_tokens = 49152;
    // HR cascade target resolution (Python pipeline_type "1024_cascade" / "1536_cascade").
    // Drives the HR sparse lattice (grid = resolution/16 → 64 @1024, 96 @1536) and the dense
    // dual-grid marching resolution. 1536 = finer voxels = thinner separable features (fingers).
    int resolution = 1024;
};
struct ChainStats { int N1=0, M=0; double secs=0; };

// MC MANIFOLD REMESH (the de-spike fix) — reusable so both run_geometry and the cached re-entry path
// (image_to_rig --from-geo) build the manifold the same way. grid-`resolution` occupancy coords [N,4]
// -> flood-fill-solid marching cubes -> WATERTIGHT MANIFOLD (boundary==0, nonmanifold==0) + Taubin.
static inline svae::Mesh build_mc_remesh(const std::vector<int32_t>& coords1024, int resolution,
                                         int stride, int blur, int post_smooth, bool verbose) {
    double tw = now_s();
    const int n = (int)coords1024.size()/4;
    svae::Mesh mm = svae::marching_cubes_solid(coords1024.data(), n, resolution, stride, blur, 0.5f);
    if (post_smooth > 0) svae::taubin_smooth(mm, post_smooth);
    int64_t b,nm; svae::mesh_topology_stats(mm, b, nm);
    if (verbose) printf("[7b] MC-REMESH (solid stride=%d blur=%d smooth=%d): grid=%d verts=%d faces=%d boundary=%lld nonmanifold=%lld (%.1fs)\n",
                        stride, blur, post_smooth, resolution/stride, mm.N, mm.F, (long long)b, (long long)nm, now_s()-tw);
    return mm;
}

// image -> mesh (+ optional texture outputs if in.textured). seed-42 noise reproduced.
//   out_vcolors    : interim per-vertex RGB [V*3] (base_color), aligned 1:1 with mesh.verts.
//   out_pbr_feats  : the full per-voxel 6-ch PBR volume [Nvox*6] (base_color3/metallic/roughness/alpha)
//   out_pbr_coords : the matching voxel coords [Nvox*4] (b,x,y,z) on the grid-1024 lattice.
// The (feats,coords) volume feeds the UV-atlas bake (texatlas::bake); vcolors is the interim COLOR_0.
static inline svae::Mesh run_geometry(const ChainInput& in, ChainStats* stats = nullptr,
                                      std::vector<float>* out_vcolors = nullptr,
                                      std::vector<float>* out_pbr_feats = nullptr,
                                      std::vector<int32_t>* out_pbr_coords = nullptr,
                                      std::vector<int32_t>* out_coords1024 = nullptr) {
    const bool V = in.verbose; const bool cuda = in.use_cuda;
    auto LOG = [&](const char* fmt, double a){ if (V) printf(fmt, a); fflush(stdout); };
    double t0 = now_s();
    trandn::Generator gen(in.seed);
    std::vector<float> img512_norm = imagenet_norm(in.img512_raw, 512);
    std::vector<float> img1024_norm = imagenet_norm(in.img1024_raw, 1024);
    geo::ProjCam cam(in.cam, in.dist, in.ms);

    // (1) DINOv3@512 (shared by stage1 + stage2)
    std::vector<float> global512, patchmap512;
    { double t=now_s(); run_dinov3(img512_norm, dino::CFG512, cuda, global512, patchmap512); LOG("[1] DINOv3@512 (%.1fs)\n", now_s()-t); }

    // (2) stage1: proj16 -> SS DiT -> SS VAE -> coords1
    std::vector<int32_t> coords1;
    {
        std::vector<float> proj_ss = geo::proj_cond_ss(patchmap512.data(), 32, 32, dino::HID, 16, 512, cam);
        const int NEL = ssdit::SEQ * ssdit::INCH;
        std::vector<float> noise1 = gen.randn(NEL);
        std::vector<float> z_s(NEL);
        {
            M1Harness Hf(SSF_W, 512, cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={ssdit::SEQ,ssdit::INCH,1,1}; ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                    ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t g_ne[4]={1024,5,1,1};                 ggml_tensor* gin=Hf.input("global",2,g_ne);
            int64_t p_ne[4]={1024,ssdit::SEQ,1,1};        ggml_tensor* pin=Hf.input("proj",2,p_ne);
            ggml_tensor* vout = ssdit::build_ss_dit_forward(cf, Hf, xin, tin, gin, pin);
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            std::vector<float> zg(global512.size(),0.f), zp(proj_ss.size(),0.f);
            auto fwd=[&](const std::vector<float>& xx, float ts, bool c){
                Hf.upload_input_raw(xin,xx); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(gin, c?global512:zg); Hf.upload_input_raw(pin, c?proj_ss:zp);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };
            double t=now_s();
            const bool nsys_ss = std::getenv("PIXAL3D_NSYS_SS") != nullptr;
            if (nsys_ss) cudaProfilerStart();
            z_s = geo::flow_sampler(NEL, noise1, 1e-5f, in.ss.guidance, in.ss.rescale, in.ss.rescale_t, in.ss.lo, in.ss.hi, in.ss.steps, fwd, V?"ss-dit":nullptr);
            if (nsys_ss) cudaProfilerStop();
            LOG("[2] SS DiT (%.1fs)\n", now_s()-t);
        }
        {
            M1Harness Hd(SSD_W, 256, cuda);
            ggml_context* cd = Hd.ctx;
            int64_t z_ne[4]={16,16,16,8}; ggml_tensor* zin=Hd.input("z_s",4,z_ne);
            ggml_tensor* logits = ssvae::build_ss_vae_decode(cd, Hd, zin);
            ggml_set_output(logits);
            ggml_cgraph* gfd = new_graph(cd, 8192); ggml_build_forward_expand(gfd, logits);
            Hd.alloc_and_upload(gfd); Hd.upload_input_raw(zin, z_s); Hd.compute(gfd);
            std::vector<float> L((size_t)64*64*64); ggml_backend_tensor_get(logits,L.data(),0,L.size()*4);
            auto cs = ssvae::logits_to_coords(L);
            coords1.reserve(cs.size()*4);
            for (auto& c : cs) { coords1.push_back(0); coords1.push_back(c[0]); coords1.push_back(c[1]); coords1.push_back(c[2]); }
        }
    }
    int N1 = (int)coords1.size()/4;
    if (V) printf("[2] stage1 coords N1=%d\n", N1);

    // (3) stage2 cond + M2 DiT -> lr_slat (denorm)
    std::vector<float> lr_denorm;
    {
        double tn=now_s();
        std::vector<float> naf_hr = run_naf(in.img512_raw, patchmap512, naf::CFG512, cuda);
        LOG("[3] NAF@512 (%.1fs)\n", now_s()-tn);
        std::vector<float> cond2 = geo::proj_cond_shape(coords1.data(), N1, patchmap512.data(), 32, 32,
                                                        naf_hr.data(), 512, 512, dino::HID, 32, 512, cam);
        std::vector<int32_t> cxyz((size_t)N1*3);
        for (int n=0;n<N1;n++) for (int j=0;j<3;j++) cxyz[n*3+j] = coords1[n*4+1+j];
        const int NEL = N1 * slatdit::INCH;
        std::vector<float> noise2 = gen.randn((int64_t)NEL);
        {
            M1Harness Hf(SL512_W, 1024, cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={slatdit::INCH,N1,1,1};      ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                   ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t g_ne[4]={1024,5,1,1};                ggml_tensor* gin=Hf.input("global",2,g_ne);
            int64_t p_ne[4]={slatdit::PROJ_IN,N1,1,1};   ggml_tensor* pin=Hf.input("proj",2,p_ne);
            ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, N1, xin, tin, gin, pin, cxyz.data());
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            std::vector<float> zg(global512.size(),0.f), zp(cond2.size(),0.f);
            auto fwd=[&](const std::vector<float>& xx, float ts, bool c){
                Hf.upload_input_raw(xin,xx); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(gin, c?global512:zg); Hf.upload_input_raw(pin, c?cond2:zp);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };
            double t=now_s();
            std::vector<float> lr_raw = geo::flow_sampler(NEL, noise2, 1e-5f, in.shape.guidance, in.shape.rescale, in.shape.rescale_t, in.shape.lo, in.shape.hi, in.shape.steps, fwd, V?"m2-dit":nullptr);
            LOG("[3] M2 DiT (%.1fs)\n", now_s()-t);
            lr_denorm = lr_raw;
            geo::denorm_inplace(lr_denorm, in.norm_mean.data(), in.norm_std.data(), slatdit::INCH);
        }
    }

    // (4) M3a upsample + quantize grid64
    std::vector<int32_t> coordsM;
    {
        double t=now_s();
#ifdef M3A_USE_CUDA
        std::vector<int32_t> hr_coords = cuda ? svpg::m3a_upsample(coords1, lr_denorm, SHAPEDEC_W)
                                              : svp::m3a_upsample(coords1, lr_denorm, SHAPEDEC_W, cuda);
#else
        std::vector<int32_t> hr_coords = svp::m3a_upsample(coords1, lr_denorm, SHAPEDEC_W, cuda);
#endif
        int Nh = (int)hr_coords.size()/4;
        // HR sparse lattice = resolution/16 (64 @1024, 96 @1536). Python cascade quantizes the
        // upsampled coords onto this grid; the decoder's fixed 16x sparse upsample then lands the
        // dense dual-grid at exactly `resolution`. (Python also steps resolution down by 128 if the
        // token count exceeds max_num_tokens; the C++ query-tiled attention removes that OOM guard,
        // so we keep the full requested grid — watch VRAM at 1536 on the 3060.)
        const int hr_grid = in.resolution / 16;
        coordsM = geo::quantize_grid_unique(hr_coords.data(), Nh, 512, hr_grid);
        if (V) printf("[4] M3a Nh=%d -> M=%d (grid%d, res%d) (%.1fs)\n", Nh, (int)coordsM.size()/4, hr_grid, in.resolution, now_s()-t);
    }
    int M = (int)coordsM.size()/4;

    // (5) stage3b cond (== stage4/tex cond) — kept for the tex branch. cxyz for sparse rope.
    //     patchmap1024 is hoisted out of the block: the cross-mode tex DiT (stage 8) consumes the
    //     FULL DINOv3 token stream (global[5,1024] ++ patchmap[4096,1024]) as its cross-attn context.
    std::vector<float> global1024, cond3b, patchmap1024;
    std::vector<int32_t> cxyz((size_t)M*3);
    for (int n=0;n<M;n++) for (int j=0;j<3;j++) cxyz[n*3+j] = coordsM[n*4+1+j];
    {
        { double t=now_s(); run_dinov3(img1024_norm, dino::CFG1024, cuda, global1024, patchmap1024); LOG("[5] DINOv3@1024 (%.1fs)\n", now_s()-t); }
        double tn=now_s();
        std::vector<float> naf_hr = run_naf(in.img1024_raw, patchmap1024, naf::CFG1024, cuda);
        LOG("[5] NAF@1024 (%.1fs)\n", now_s()-tn);
        // R (coord lattice resolution for cam.project) = resolution/16; patchmap stays 64x64
        // (DINOv3@1024 token grid) and img_res stays 1024 (DINOv3 input), independent of resolution.
        cond3b = geo::proj_cond_shape(coordsM.data(), M, patchmap1024.data(), 64, 64,
                                      naf_hr.data(), 512, 512, dino::HID, in.resolution/16, 1024, cam);
    }

    // (6) M3b DiT -> shape_slat (denorm)
    std::vector<float> shape_denorm;
    {
        const int NEL = M * slatdit::INCH;
        std::vector<float> noise3 = gen.randn((int64_t)NEL);
        M1Harness Hf(SL1024_W, 2048, cuda);
        ggml_context* cf = Hf.ctx;
        int64_t x_ne[4]={slatdit::INCH,M,1,1};       ggml_tensor* xin=Hf.input("x",2,x_ne);
        int64_t t_ne[4]={1,1,1,1};                   ggml_tensor* tin=Hf.input("t",1,t_ne);
        int64_t g_ne[4]={1024,5,1,1};                ggml_tensor* gin=Hf.input("global",2,g_ne);
        int64_t p_ne[4]={slatdit::PROJ_IN,M,1,1};    ggml_tensor* pin=Hf.input("proj",2,p_ne);
        ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, M, xin, tin, gin, pin, cxyz.data());
        ggml_set_output(vout);
        ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
        Hf.alloc_and_upload(gff);
        std::vector<float> zg(global1024.size(),0.f), zp(cond3b.size(),0.f);
        auto fwd=[&](const std::vector<float>& xx, float ts, bool c){
            Hf.upload_input_raw(xin,xx); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
            Hf.upload_input_raw(gin, c?global1024:zg); Hf.upload_input_raw(pin, c?cond3b:zp);
            Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };
        double t=now_s();
        const bool nsys_m3b = std::getenv("PIXAL3D_NSYS_M3B") != nullptr;
        if (nsys_m3b) cudaProfilerStart();
        std::vector<float> hr_raw = geo::flow_sampler(NEL, noise3, 1e-5f, in.shape.guidance, in.shape.rescale, in.shape.rescale_t, in.shape.lo, in.shape.hi, in.shape.steps, fwd, V?"m3b-dit":nullptr);
        if (nsys_m3b) cudaProfilerStop();
        LOG("[6] M3b DiT (%.1fs)\n", now_s()-t);
        shape_denorm = hr_raw;
        geo::denorm_inplace(shape_denorm, in.norm_mean.data(), in.norm_std.data(), slatdit::INCH);
    }

    // (7) M4 shape decoder + mesh (capture per-level subs for the tex branch; grid-1024 occupancy
    //     for the remesh). close_surface (the interim hole-fill hack) is skipped when remeshing.
    double t=now_s();
    std::vector<std::vector<uint8_t>> subs;
    std::vector<int32_t> coords1024;
    const bool cs = in.watertight && !in.remesh && !in.mc_remesh; // interim close_surface only when NOT remeshing
    // MC manifold remesh needs the grid-`resolution` occupancy; QEM remesh decimates the dual mesh
    // directly (no occupancy needed). Request coords1024 for the MC path OR when the caller wants it
    // dumped (out_coords1024, e.g. --dump-geo for cached remesh re-entry).
    std::vector<int32_t>* pc1024 = (in.mc_remesh || out_coords1024) ? &coords1024 : nullptr;
#ifdef M3A_USE_CUDA
    svae::Mesh mesh = cuda ? svpg::m4_decode_mesh(coordsM, shape_denorm, SHAPEDEC_W, in.textured ? &subs : nullptr, cs, pc1024, in.resolution)
                           : svp::m4_decode_mesh(coordsM, shape_denorm, SHAPEDEC_W, cuda, in.textured ? &subs : nullptr, cs, pc1024, in.resolution);
#else
    svae::Mesh mesh = svp::m4_decode_mesh(coordsM, shape_denorm, SHAPEDEC_W, cuda, in.textured ? &subs : nullptr, cs, pc1024, in.resolution);
#endif
    if (V) printf("[7] M4 mesh: verts=%d faces=%d (%.1fs)\n", mesh.N, mesh.F, now_s()-t);
    if (out_coords1024) *out_coords1024 = coords1024;   // cache for --from-geo remesh re-entry

    // (7b) A1 remesh — FEATURE-PRESERVING QEM DECIMATION of the crisp dual-grid mesh (lap-17).
    //   The M4 dual-grid mesh above (`mesh`) IS the sharp "PS4-quality" source: its vertices are the
    //   O-Voxel QEF positions, so heel spikes / fingers / chin gap are reconstructed exactly — it's
    //   just massive (~3.25M faces) and non-manifold (~162k nm edges + ~58k boundary). Earlier laps
    //   re-marched a COARSE/blurred grid (rounded everything to nubs — the "N64 putty blob") OR fell
    //   to meshopt simplifySloppy (noisy mottled normals), because meshopt QUALITY LOCKS on the
    //   non-manifold skeleton (~1.08M floor). Fix = our own Garland-Heckbert QEM (`qem.hpp`) which
    //   accumulates a per-vertex quadric from all incident faces and collapses each edge
    //   independently — non-manifold is no obstacle — flattening flat regions while DEFERRING
    //   collapses across sharp edges (high quadric cost) + rejecting normal flips → keeps the shape,
    //   fewer polys. Decimates 3.25M→~150-200k in ~3s, render ≈ the golden full mesh.
    //   PIXAL3D_REMESH_FACES = target triangle budget (default 200000; **0 = keep the raw max-detail
    //   mesh** for the "give me the massive mesh" option). PIXAL3D_REMESH_AGGR = QEM aggressiveness.
    // REMESH timing: when TEXTURED, the QEM is DEFERRED to the texture branch so we can colour the
    // DENSE mesh first (exact, smooth, hole-free) and carry the colour through the collapses (avoids
    // the noisy "zombie" patchwork from re-sampling the sparse PBR volume at the moved output verts).
    // Geometry-only remesh decimates here.
    const bool remesh_deferred = in.remesh && in.textured;
    const int   remesh_target = std::getenv("PIXAL3D_REMESH_FACES") ? atoi(std::getenv("PIXAL3D_REMESH_FACES")) : 200000;
    const double remesh_aggr  = std::getenv("PIXAL3D_REMESH_AGGR")  ? atof(std::getenv("PIXAL3D_REMESH_AGGR"))  : 7.0;
    if (in.mc_remesh) {
        // MC MANIFOLD REMESH (the de-spike fix): the raw dual-grid `mesh` above is the crisp but
        // non-manifold O-Voxel surface. Replace it with the flood-fill-solid marching-cubes manifold
        // (→ tight precluster atlas downstream). Texture is sampled positionally from the PBR volume,
        // so the new verts carry colour fine.
        mesh = build_mc_remesh(coords1024, in.resolution, in.mc_stride, in.mc_blur, in.mc_post_smooth, V);
    } else if (in.remesh && !remesh_deferred) {
        double tw = now_s();
        if (remesh_target > 0 && mesh.F > remesh_target) {
            int f0 = mesh.F;
            mesh = qem::qem_simplify(mesh, remesh_target, remesh_aggr);
            svae::orient_consistent(mesh);
            int64_t b,nm; svae::mesh_topology_stats(mesh, b, nm);
            if (V) printf("[7b] REMESH (QEM %d→%d faces, target %d, aggr %.1f): verts=%d boundary=%lld nonmanifold=%lld (%.1fs)\n",
                          f0, mesh.F, remesh_target, remesh_aggr, mesh.N, (long long)b, (long long)nm, now_s()-tw);
        } else if (V) {
            printf("[7b] REMESH: raw max-detail mesh kept (faces=%d, target=%d) (%.1fs)\n", mesh.F, remesh_target, now_s()-tw);
        }
    } else if (in.watertight) {
        double tw = now_s();
        int64_t b0 = svae::boundary_edge_count(mesh);
        int added = (b0 > 0) ? svae::fill_holes(mesh) : 0;
        int64_t b1 = svae::boundary_edge_count(mesh);
        if (V) printf("[7b] watertight: boundary %lld->%lld (+%d ear tris), verts=%d faces=%d (%.1fs)\n",
                      (long long)b0, (long long)b1, added, mesh.N, mesh.F, now_s()-tw);
    }

    // (8) M6 TEXTURE branch: tex DiT (in_ch64 = noise || re-normed shape_slat, CFG off, 4th
    //     seed-42 draw) -> tex_slat -> tex decoder (out 6 PBR, guide_subs=subs) -> per-vertex color.
    if (in.textured && (out_vcolors || out_pbr_feats)) {
        // re-normalize shape_slat for the concat: (x - shape_mean)/shape_std
        std::vector<float> shape_renorm = shape_denorm;
        for (size_t i=0;i<shape_renorm.size();i++){ int c=(int)(i%slatdit::INCH); shape_renorm[i]=(shape_renorm[i]-in.norm_mean[c])/in.norm_std[c]; }
        const int INCH=slatdit::INCH, IN64=64, NEL=M*INCH;
        std::vector<float> tex_noise = gen.randn((int64_t)NEL);   // 4th seed-42 draw
        std::vector<float> tex_raw;
        if (in.tex_proj) {
            // ---- PROJ-MODE tex DiT (pixal3d's OWN model — what Python's Pixal3DImageTo3DPipeline runs) ----
            // Identical call shape to the M3b geometry DiT above, because it IS the same graph: cond is
            // global1024[5,1024] + cond3b[2048,M] (the stage3b proj cond, already computed at step (5) —
            // golden stage3b_cond == stage4_cond byte-for-byte, so no new cond path exists or is needed).
            // Only input_layer differs (64ch), which is a WEIGHT property, so build_slat_dit_forward is
            // reused verbatim. x pre-concat [x32 || shape_renorm32]; CFG off by default (guidance 1.0).
            // Cheaper than cross-mode: the cross-attn KV context is 5 tokens, not 4101, so the per-block
            // to_kv + attention over the image cond shrinks ~820x (it becomes negligible next to self-attn
            // over M voxels); proj_linear adds one [1536,2048]xM matmul per block. Net: strictly less work
            // and less VRAM than cross-mode at equal M. NOTE: this was the COMMITTED HEAD behaviour —
            // cross-mode is an uncommitted working-tree change that swapped production off this model.
            const char* pw = in.tex_flow_w.empty() ? TEXFLOW_PROJ_W : in.tex_flow_w.c_str();
            M1Harness Hf(pw, 2048, cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={IN64,M,1,1};                ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                   ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t g_ne[4]={1024,5,1,1};                ggml_tensor* gin=Hf.input("global",2,g_ne);
            int64_t p_ne[4]={slatdit::PROJ_IN,M,1,1};    ggml_tensor* pin=Hf.input("proj",2,p_ne);
            ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, M, xin, tin, gin, pin, cxyz.data());
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            // Zero-cond (the CFG negative branch) is only ever requested when guidance != 1.0; the tex
            // stage ships CFG OFF, so don't pay ~M*2048*4 bytes of zeros for a branch that never runs.
            std::vector<float> zg, zp;
            if (in.tex.guidance != 1.0f) { zg.assign(global1024.size(), 0.f); zp.assign(cond3b.size(), 0.f); }
            std::vector<float> x64((size_t)M*IN64);
            auto fwd=[&](const std::vector<float>& x32, float ts, bool c){
                for (int n=0;n<M;n++){ for (int cc=0;cc<INCH;cc++) x64[(size_t)n*IN64+cc]=x32[(size_t)n*INCH+cc];
                                       for (int cc=0;cc<INCH;cc++) x64[(size_t)n*IN64+INCH+cc]=shape_renorm[(size_t)n*INCH+cc]; }
                Hf.upload_input_raw(xin,x64); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(gin, c?global1024:zg); Hf.upload_input_raw(pin, c?cond3b:zp);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };
            double tt=now_s();
            const bool nsys_tex = std::getenv("PIXAL3D_NSYS_TEX") != nullptr;
            if (nsys_tex) cudaProfilerStart();
            tex_raw = geo::flow_sampler(NEL, tex_noise, 1e-5f, in.tex.guidance, in.tex.rescale, in.tex.rescale_t, in.tex.lo, in.tex.hi, in.tex.steps, fwd, V?"tex-dit":nullptr);
            if (nsys_tex) cudaProfilerStop();
            if (V) { printf("[8] tex DiT (PROJ-mode, 5 global tokens + proj_linear, w=%s) (%.1fs)\n", pw, now_s()-tt); fflush(stdout); }
        } else {
        // CROSS-MODE tex DiT (TRELLIS-2 oracle's tex_slat_flow_model = image_attn_mode="cross"):
        // full-token DINOv3 image cond `cin[1024,Ntok]` = global[5,1024] (cls+4 reg) ++ patchmap
        // [4096,1024]. NOT the geometry path's proj-mode (global[5]+proj_cond). build_tex_dit_cross_forward
        // reuses the slat-DiT self-attn skeleton + 3D-RoPE; cross-attn is standard full-token (to_kv over
        // cin). x is fed pre-concat [x32 || shape_renorm32]; CFG off (guidance 1.0). Validated vs the
        // tex_goldens at cosine 0.9998 (tex_dit_test) with the golden cond.
        const int Ntok = 5 + (int)(patchmap1024.size()/1024);    // 5 special + 4096 patch tokens = 4101
        std::vector<float> cin = global1024;                     // [5,1024] row-major (token-major)
        cin.insert(cin.end(), patchmap1024.begin(), patchmap1024.end());  // ++ [4096,1024] -> [Ntok,1024]
        {
            M1Harness Hf(in.tex_flow_w.empty() ? TEXFLOW_W : in.tex_flow_w.c_str(), 4096, cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={IN64,M,1,1};               ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                  ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t c_ne[4]={1024,Ntok,1,1};            ggml_tensor* cinT=Hf.input("cond",2,c_ne);
            ggml_tensor* vout = texdit::build_tex_dit_cross_forward(cf, Hf, M, Ntok, xin, tin, cinT, cxyz.data());
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 65536); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            std::vector<float> x64((size_t)M*IN64);
            auto fwd=[&](const std::vector<float>& x32, float ts, bool /*CFG off*/){
                for (int n=0;n<M;n++){ for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+c]=x32[(size_t)n*INCH+c];
                                       for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+INCH+c]=shape_renorm[(size_t)n*INCH+c]; }
                Hf.upload_input_raw(xin,x64); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(cinT,cin);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };
            double tt=now_s();
            tex_raw = geo::flow_sampler(NEL, tex_noise, 1e-5f, in.tex.guidance, in.tex.rescale, in.tex.rescale_t, in.tex.lo, in.tex.hi, in.tex.steps, fwd, V?"tex-dit":nullptr);
            if (V) { printf("[8] tex DiT (cross-mode, Ntok=%d) (%.1fs)\n", Ntok, now_s()-tt); fflush(stdout); }
        }
        }
        std::vector<float> tex_slat = tex_raw;
        geo::denorm_inplace(tex_slat, in.tex_mean.data(), in.tex_std.data(), INCH);
        double tt=now_s();
        std::vector<int32_t> pbr_coords;
#ifdef M3A_USE_CUDA
        std::vector<float> pbr = cuda ? svpg::m6_tex_decode(coordsM, tex_slat, subs, TEXDEC_W, &pbr_coords)
                                      : svp::m6_tex_decode(coordsM, tex_slat, subs, TEXDEC_W, cuda, &pbr_coords);  // [V,6]
#else
        std::vector<float> pbr = svp::m6_tex_decode(coordsM, tex_slat, subs, TEXDEC_W, cuda, &pbr_coords);  // [V,6]
#endif
        if (V) printf("[8] tex decoder: %d PBR voxels (%.1fs)\n", (int)pbr.size()/6, now_s()-tt);
        int V6 = (int)pbr.size()/6;
        // full PBR volume (feeds the UV-atlas bake): feats [V6*6] + coords [V6*4]
        if (out_pbr_feats)  *out_pbr_feats  = pbr;
        if (out_pbr_coords) *out_pbr_coords = pbr_coords;
        // PIXAL3D_DUMP_BAKE: also bank the DENSE (pre-QEM) outer-shell mesh here, while it still
        // exists — the lap-18 BVH-reproject needs the smooth dense surface (closest-point-on-triangle)
        // as the colour source-of-truth, not just the decimated QEM mesh. `mesh` is the dense dual-grid
        // mesh at this point (the deferred QEM decimate runs below). The QEM mesh + the PBR volume are
        // dumped separately from pixal3d.cpp once run_geometry returns. (out_pbr_feats != null ⇒ the
        // uvatlas dump path; gate on the same env so plain runs pay nothing.)
        if (out_pbr_feats && remesh_deferred && std::getenv("PIXAL3D_DUMP_BAKE")) {
            auto sv=[](const char* p, const void* d, size_t n){ FILE* f=fopen(p,"wb"); if(f){fwrite(d,1,n,f);fclose(f);} };
            sv("dump_dense_v.bin", mesh.verts.data(), mesh.verts.size()*4);
            sv("dump_dense_f.bin", mesh.faces.data(), mesh.faces.size()*8);
            FILE* m=fopen("dump_dense.txt","w");
            if (m){ fprintf(m,"%zu %zu\n", mesh.verts.size()/3, mesh.faces.size()/3); fclose(m); }
            if (V) printf("  [dump] dense mesh -> dump_dense_*.bin (%zu v / %zu f)\n", mesh.verts.size()/3, mesh.faces.size()/3);
        }
        // per-vertex base_color. NON-remesh: mesh.verts[i] <-> pbr voxel i 1:1 zip (the dual-grid mesh
        // vertex IS voxel i). REMESH: the QEM mesh verts are NEW, so grid_sample (trilinear) the PBR
        // volume at each vertex position. This is the CLEAN colour path for the remeshed mesh — the
        // UV-atlas bake's per-texel positions interpolate across folded charts into the TEAL INTERIOR
        // of the model (→ the "teal splattered everywhere" glitch); sampling at the actual surface
        // VERTICES never reads the interior. Small fallback (off-shell QEM verts are sub-voxel away).
        if (out_vcolors) {
            if (in.mc_remesh) {
                // MC manifold verts are NEW (not voxel-aligned) → trilinear grid_sample the PBR volume
                // at each manifold vertex position (the clean, interior-free colour path).
                texgs::VolIndex vol(pbr_coords.data(), V6, 4, 1);
                const int fbr = std::getenv("PIXAL3D_VCOLOR_FB") ? atoi(std::getenv("PIXAL3D_VCOLOR_FB")) : 12;
                out_vcolors->resize((size_t)mesh.N*3);
                #pragma omp parallel for schedule(dynamic,2048)
                for (int i=0;i<mesh.N;i++){ float s[6]={0};
                    float q0=(mesh.verts[(size_t)i*3]+0.5f)*1024.f, q1=(mesh.verts[(size_t)i*3+1]+0.5f)*1024.f, q2=(mesh.verts[(size_t)i*3+2]+0.5f)*1024.f;
                    texgs::sample_one(vol, pbr.data(), 6, q0,q1,q2, s, fbr);
                    for (int c=0;c<3;c++){ float v=s[c]; out_vcolors->at((size_t)i*3+c)=v<0?0:(v>1?1:v); } }
            } else if (remesh_deferred) {
                // colour the DENSE dual-grid mesh per-vertex (verts sit on the shell → trilinear hits
                // the exact voxel → smooth, hole-free), THEN QEM-decimate carrying the colour through
                // the collapses. This is what kills the "zombie" patchwork (re-sampling the sparse
                // volume at the sparse, slightly-off output verts was the noise source).
                texgs::VolIndex vol(pbr_coords.data(), V6, 4, 1);
                const int fbr = std::getenv("PIXAL3D_VCOLOR_FB") ? atoi(std::getenv("PIXAL3D_VCOLOR_FB")) : 12;
                std::vector<float> dcol((size_t)mesh.N*3);
                #pragma omp parallel for schedule(dynamic,2048)
                for (int i=0;i<mesh.N;i++){ float s[6]={0};
                    float q0=(mesh.verts[(size_t)i*3]+0.5f)*1024.f, q1=(mesh.verts[(size_t)i*3+1]+0.5f)*1024.f, q2=(mesh.verts[(size_t)i*3+2]+0.5f)*1024.f;
                    texgs::sample_one(vol, pbr.data(), 6, q0,q1,q2, s, fbr);
                    for (int c=0;c<3;c++){ float v=s[c]; dcol[(size_t)i*3+c]=v<0?0:(v>1?1:v); } }
                double tw=now_s();
                if (remesh_target > 0 && mesh.F > remesh_target) {
                    int f0=mesh.F;
                    svae::Mesh dec = qem::qem_simplify(mesh, remesh_target, remesh_aggr, &dcol, out_vcolors);
                    svae::orient_consistent(dec); mesh = dec;
                    int64_t b,nm; svae::mesh_topology_stats(mesh,b,nm);
                    if (V) printf("[7b] REMESH (QEM+colour %d→%d faces, target %d): verts=%d boundary=%lld nonmanifold=%lld (%.1fs)\n",
                                  f0, mesh.F, remesh_target, mesh.N, (long long)b, (long long)nm, now_s()-tw);
                } else { *out_vcolors = dcol; if (V) printf("[7b] REMESH: raw max-detail mesh + per-vertex colour (faces=%d)\n", mesh.F); }
            } else {
                out_vcolors->resize((size_t)mesh.N*3);
                for (int i=0;i<mesh.N && i<V6;i++) for (int c=0;c<3;c++){ float v=pbr[(size_t)i*6+c]; out_vcolors->at((size_t)i*3+c)=v<0?0:(v>1?1:v); }
            }
        }
    }

    // Catch-all: if the QEM was deferred (textured remesh) but the colour path above didn't run it
    // (e.g. forced UV-atlas, no out_vcolors), decimate now so the returned/baked mesh is web-sized.
    if (remesh_deferred && remesh_target > 0 && mesh.F > remesh_target) {
        double tw=now_s(); int f0=mesh.F;
        mesh = qem::qem_simplify(mesh, remesh_target, remesh_aggr);
        svae::orient_consistent(mesh);
        if (V) printf("[7b] REMESH (QEM deferred %d→%d faces) (%.1fs)\n", f0, mesh.F, now_s()-tw);
    }

    if (stats) { stats->N1=N1; stats->M=M; stats->secs=now_s()-t0; }
    return mesh;
}

}  // namespace pix
