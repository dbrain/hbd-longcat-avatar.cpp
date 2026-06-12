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
#include "geometry_e2e.hpp"
#include "sparse_vae_pipeline.hpp"
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
static const char* TEXFLOW_W = "weights_npy/slat_flow_imgshape2tex_1024";   // M6 tex DiT (in_ch 64)
static const char* TEXDEC_W = "weights_npy/tex_dec";                        // M6 tex decoder (out 6)

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

struct ChainInput {
    std::vector<float> img512_raw, img1024_raw;   // [0,1] CHW (DINOv3 input is normalized internally)
    float cam, dist, ms;                          // camera scalars (radians / units)
    std::vector<float> norm_mean, norm_std;       // shape_slat denorm [32]
    std::vector<float> tex_mean, tex_std;         // tex_slat denorm [32] (only if textured)
    bool use_cuda = true;
    bool verbose = true;
    bool textured = false;                        // M6: also run the tex branch -> per-vertex color
};
struct ChainStats { int N1=0, M=0; double secs=0; };

// image -> mesh (+ optional per-vertex base_color if in.textured). seed-42 noise reproduced.
// out_vcolors (if textured): per-vertex RGB [V*3] (base_color), aligned 1:1 with mesh.verts.
static inline svae::Mesh run_geometry(const ChainInput& in, ChainStats* stats = nullptr,
                                      std::vector<float>* out_vcolors = nullptr) {
    const bool V = in.verbose; const bool cuda = in.use_cuda;
    auto LOG = [&](const char* fmt, double a){ if (V) printf(fmt, a); fflush(stdout); };
    double t0 = now_s();
    trandn::Generator gen(42);
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
            z_s = geo::flow_sampler(NEL, noise1, 1e-5f, 7.5f, 0.7f, 5.0, 0.6, 1.0, 12, fwd, V?"ss-dit":nullptr);
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
            std::vector<float> lr_raw = geo::flow_sampler(NEL, noise2, 1e-5f, 7.5f, 0.5f, 3.0, 0.6, 1.0, 12, fwd, V?"m2-dit":nullptr);
            LOG("[3] M2 DiT (%.1fs)\n", now_s()-t);
            lr_denorm = lr_raw;
            geo::denorm_inplace(lr_denorm, in.norm_mean.data(), in.norm_std.data(), slatdit::INCH);
        }
    }

    // (4) M3a upsample + quantize grid64
    std::vector<int32_t> coordsM;
    {
        double t=now_s();
        std::vector<int32_t> hr_coords = svp::m3a_upsample(coords1, lr_denorm, SHAPEDEC_W, cuda);
        int Nh = (int)hr_coords.size()/4;
        coordsM = geo::quantize_grid_unique(hr_coords.data(), Nh, 512, 64);
        if (V) printf("[4] M3a Nh=%d -> M=%d (%.1fs)\n", Nh, (int)coordsM.size()/4, now_s()-t);
    }
    int M = (int)coordsM.size()/4;

    // (5) stage3b cond (== stage4/tex cond) — kept for the tex branch. cxyz for sparse rope.
    std::vector<float> global1024, cond3b;
    std::vector<int32_t> cxyz((size_t)M*3);
    for (int n=0;n<M;n++) for (int j=0;j<3;j++) cxyz[n*3+j] = coordsM[n*4+1+j];
    {
        std::vector<float> patchmap1024;
        { double t=now_s(); run_dinov3(img1024_norm, dino::CFG1024, cuda, global1024, patchmap1024); LOG("[5] DINOv3@1024 (%.1fs)\n", now_s()-t); }
        double tn=now_s();
        std::vector<float> naf_hr = run_naf(in.img1024_raw, patchmap1024, naf::CFG1024, cuda);
        LOG("[5] NAF@1024 (%.1fs)\n", now_s()-tn);
        cond3b = geo::proj_cond_shape(coordsM.data(), M, patchmap1024.data(), 64, 64,
                                      naf_hr.data(), 512, 512, dino::HID, 64, 1024, cam);
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
        std::vector<float> hr_raw = geo::flow_sampler(NEL, noise3, 1e-5f, 7.5f, 0.5f, 3.0, 0.6, 1.0, 12, fwd, V?"m3b-dit":nullptr);
        LOG("[6] M3b DiT (%.1fs)\n", now_s()-t);
        shape_denorm = hr_raw;
        geo::denorm_inplace(shape_denorm, in.norm_mean.data(), in.norm_std.data(), slatdit::INCH);
    }

    // (7) M4 shape decoder + mesh (capture per-level subs for the tex branch)
    double t=now_s();
    std::vector<std::vector<uint8_t>> subs;
    svae::Mesh mesh = svp::m4_decode_mesh(coordsM, shape_denorm, SHAPEDEC_W, cuda, in.textured ? &subs : nullptr);
    if (V) printf("[7] M4 mesh: verts=%d faces=%d (%.1fs)\n", mesh.N, mesh.F, now_s()-t);

    // (8) M6 TEXTURE branch: tex DiT (in_ch64 = noise || re-normed shape_slat, CFG off, 4th
    //     seed-42 draw) -> tex_slat -> tex decoder (out 6 PBR, guide_subs=subs) -> per-vertex color.
    if (in.textured && out_vcolors) {
        // re-normalize shape_slat for the concat: (x - shape_mean)/shape_std
        std::vector<float> shape_renorm = shape_denorm;
        for (size_t i=0;i<shape_renorm.size();i++){ int c=(int)(i%slatdit::INCH); shape_renorm[i]=(shape_renorm[i]-in.norm_mean[c])/in.norm_std[c]; }
        const int INCH=slatdit::INCH, IN64=64, NEL=M*INCH;
        std::vector<float> tex_noise = gen.randn((int64_t)NEL);   // 4th seed-42 draw
        std::vector<float> tex_raw;
        {
            M1Harness Hf(TEXFLOW_W, 2048, cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={IN64,M,1,1};               ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                  ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t g_ne[4]={1024,5,1,1};               ggml_tensor* gin=Hf.input("global",2,g_ne);
            int64_t p_ne[4]={slatdit::PROJ_IN,M,1,1};   ggml_tensor* pin=Hf.input("proj",2,p_ne);
            ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, M, xin, tin, gin, pin, cxyz.data());
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            std::vector<float> x64((size_t)M*IN64);
            auto fwd=[&](const std::vector<float>& x32, float ts, bool /*CFG off*/){
                for (int n=0;n<M;n++){ for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+c]=x32[(size_t)n*INCH+c];
                                       for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+INCH+c]=shape_renorm[(size_t)n*INCH+c]; }
                Hf.upload_input_raw(xin,x64); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(gin,global1024); Hf.upload_input_raw(pin,cond3b);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };
            double tt=now_s();
            tex_raw = geo::flow_sampler(NEL, tex_noise, 1e-5f, 1.0f, 0.0f, 3.0, 0.6, 0.9, 12, fwd, V?"tex-dit":nullptr);
            LOG("[8] tex DiT (%.1fs)\n", now_s()-tt);
        }
        std::vector<float> tex_slat = tex_raw;
        geo::denorm_inplace(tex_slat, in.tex_mean.data(), in.tex_std.data(), INCH);
        double tt=now_s();
        std::vector<float> pbr = svp::m6_tex_decode(coordsM, tex_slat, subs, TEXDEC_W, cuda);  // [V,6]
        if (V) printf("[8] tex decoder: %d PBR voxels (%.1fs)\n", (int)pbr.size()/6, now_s()-tt);
        // per-vertex base_color (mesh.verts[i] <-> pbr voxel i)
        int V6 = (int)pbr.size()/6;
        out_vcolors->resize((size_t)mesh.N*3);
        for (int i=0;i<mesh.N && i<V6;i++) for (int c=0;c<3;c++){ float v=pbr[(size_t)i*6+c]; out_vcolors->at((size_t)i*3+c)=v<0?0:(v>1?1:v); }
    }

    if (stats) { stats->N1=N1; stats->M=M; stats->secs=now_s()-t0; }
    return mesh;
}

}  // namespace pix
