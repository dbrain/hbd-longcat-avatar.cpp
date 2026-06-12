// ============================================================================
// A1 — Pixal3D GEOMETRY E2E driver: image -> untextured mesh (PLY + GLB), entirely in C++.
//
// Chains the per-stage-VALIDATED programs into ONE image->mesh pipeline, reproducing the
// python pipeline's seed-42 output (cross-stage torch CPU randn ported in torch_randn.hpp):
//   1. DINOv3@512 (shared) -> global512 + patchmap512
//   2. stage1: proj_grid16(patchmap512) -> SS dense DiT (12-step FlowEuler, GS7.5/GR0.7/RT5/[0.6,1])
//              -> SS VAE decode -> coords1[N1,4]@grid32                 (noise1 = randn(1,8,16^3))
//   3. stage2: NAF@512(guide512,patchmap512) + proj_grid32 @coords1 -> cond2[N1,2048];
//              M2 DiT (slat_flow_512, GS7.5/GR0.5/RT3/[0.6,1]) -> denorm -> lr_slat[N1,32]
//                                                                          (noise2 = randn(N1,32))
//   4. M3a: decoder.upsample x4 -> hr_coords -> quantize grid64 -> coordsM[M,4]
//   5. DINOv3@1024 + NAF@1024 + proj_grid64 @coordsM -> cond3b[M,2048]
//   6. M3b DiT (slat_flow_1024) -> denorm -> shape_slat[M,32]            (noise3 = randn(M,32))
//   7. M4: decoder.forward(grid64->1024) + FDG head + O-Voxel mesh -> verts/faces -> PLY + GLB
//
// Validation: token-set IoU @grid64 vs golden stage3b coords + mesh V/F vs golden stage5_mesh
// (fp32 chain vs fp16 golden => IoU ~0.97-0.99 = boundary noise, the M1 lesson).
// Build: ./build.sh geometry_e2e cuda     Run: NVIDIA_TF32_OVERRIDE=0 ./geometry_e2e cuda
// ============================================================================
#include "dinov3_graph.hpp"
#include "naf_graph.hpp"
#include "ss_dit_graph.hpp"
#include "ss_vae_graph.hpp"
#include "slat_dit_graph.hpp"
#include "geometry_e2e.hpp"
#include "sparse_vae_pipeline.hpp"
#include "torch_randn.hpp"
#include "glb_writer.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <array>
#include <chrono>

static const char* DINO_W = "weights_npy/dinov3";
static const char* NAF_W  = "weights_npy/naf";
static const char* SSF_W  = "weights_npy/ss_flow";
static const char* SSD_W  = "weights_npy/ss_dec";
static const char* SL512_W = "weights_npy/slat_flow_512";
static const char* SL1024_W = "weights_npy/slat_flow_1024";
static const char* SHAPEDEC_W = "weights_npy/shape_dec";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";

// camera scalars (cam.json — miku decode, from MoGe)
static const float CAM = 0.7332379387484828f, DIST = 1.3021559715270996f, MS = 1.0f;

static double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// ---- DINOv3 runner: normalized image CHW [3,IMG,IMG] -> global[5*1024] + patchmap[NP*1024] BHWC
static void run_dinov3(const std::vector<float>& img_norm, const dino::Cfg& cfg, bool use_cuda,
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

// ---- NAF runner: guide CHW [3,IMG_IN,IMG_IN] [0,1] + patchmap[SRC*SRC*1024] BHWC -> hr[1024*512*512] BCHW
static std::vector<float> run_naf(const std::vector<float>& guide_raw, const std::vector<float>& patchmap,
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
    // vin = patchmap BHWC [SRC*SRC,1024] -> [C,H,W]
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

// imagenet-normalize a raw [0,1] CHW image (in place copy)
static std::vector<float> imagenet_norm(const std::vector<float>& raw, int IMG) {
    std::vector<float> n = raw;
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    const size_t HW = (size_t)IMG*IMG;
    for (int c=0;c<3;c++) for (size_t i=0;i<HW;i++) n[(size_t)c*HW+i] = (n[(size_t)c*HW+i]-mean[c])/sd[c];
    return n;
}

typedef std::array<int32_t,4> C4;
static std::set<C4> cset4(const int32_t* c, int N){ std::set<C4> s; for(int i=0;i<N;i++) s.insert({c[i*4],c[i*4+1],c[i*4+2],c[i*4+3]}); return s; }

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    printf("==== geometry_e2e (image -> mesh GLB) backend=%s ====\n", use_cuda ? "cuda" : "cpu");
    double t0 = now_s();

    trandn::Generator gen(42);  // ONE continuous CPU RNG stream (== torch.manual_seed(42))

    // ====================== input images ======================
    // @512: raw [0,1] guide (naf_guide.npy) -> imagenet-normalize for DINOv3; @1024: image_1024_chw raw.
    NpyArray g512N = npy_load(std::string(REFS)+"/naf_guide.npy");            // [3,512,512] [0,1]
    std::vector<float> img512_raw(g512N.f32(), g512N.f32()+g512N.numel());
    std::vector<float> img512_norm = imagenet_norm(img512_raw, 512);
    NpyArray g1024N = npy_load(std::string(REFS)+"/stage3b/image_1024_chw.npy"); // [3,1024,1024] [0,1]
    std::vector<float> img1024_raw(g1024N.f32(), g1024N.f32()+g1024N.numel());
    std::vector<float> img1024_norm = imagenet_norm(img1024_raw, 1024);

    // shape_slat normalization (denorm: slat*std+mean)
    NpyArray mN = npy_load(std::string(REFS)+"/shape_slat_norm_mean.npy");
    NpyArray sN = npy_load(std::string(REFS)+"/shape_slat_norm_std.npy");
    std::vector<float> norm_mean(mN.f32(), mN.f32()+32), norm_std(sN.f32(), sN.f32()+32);

    // ====================== (1) DINOv3@512 (shared by stage1 + stage2) ======================
    std::vector<float> global512, patchmap512;
    { double t=now_s(); run_dinov3(img512_norm, dino::CFG512, use_cuda, global512, patchmap512);
      printf("[1] DINOv3@512: global=%zu patchmap=%zu  (%.1fs)\n", global512.size(), patchmap512.size(), now_s()-t); }

    geo::ProjCam cam(CAM, DIST, MS);

    // ====================== (2) stage1: proj16 -> SS DiT -> SS VAE -> coords1 ======================
    std::vector<int32_t> coords1;  // [N1,4] @grid32, sorted (b,x,y,z)
    {
        // proj cond (DINO-only, dense grid16) [4096,1024]
        std::vector<float> proj_ss = geo::proj_cond_ss(patchmap512.data(), 32, 32, dino::HID, 16, 512, cam);
        // SS DiT 12-step sampler
        const int NEL = ssdit::SEQ * ssdit::INCH;  // 4096*8
        std::vector<float> noise1 = gen.randn(NEL);   // randn(1,8,16,16,16)
        std::vector<float> z_s(NEL);
        {
            M1Harness Hf(SSF_W, 512, use_cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={ssdit::SEQ,ssdit::INCH,1,1}; ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                    ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t g_ne[4]={1024,5,1,1};                 ggml_tensor* gin=Hf.input("global",2,g_ne);
            int64_t p_ne[4]={1024,ssdit::SEQ,1,1};        ggml_tensor* pin=Hf.input("proj",2,p_ne);
            ggml_tensor* vout = ssdit::build_ss_dit_forward(cf, Hf, xin, tin, gin, pin);
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            std::vector<float> zero_g(global512.size(),0.f), zero_p(proj_ss.size(),0.f);
            auto forward=[&](const std::vector<float>& xx, float ts, bool cond){
                Hf.upload_input_raw(xin,xx); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(gin, cond?global512:zero_g); Hf.upload_input_raw(pin, cond?proj_ss:zero_p);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v;
            };
            double t=now_s();
            z_s = geo::flow_sampler(NEL, noise1, 1e-5f, 7.5f, 0.7f, 5.0, 0.6, 1.0, 12, forward, "ss-dit");
            printf("[2] SS DiT 12 steps  (%.1fs)\n", now_s()-t);
        }
        // SS VAE decode -> coords
        {
            M1Harness Hd(SSD_W, 256, use_cuda);
            ggml_context* cd = Hd.ctx;
            int64_t z_ne[4]={16,16,16,8}; ggml_tensor* zin=Hd.input("z_s",4,z_ne);
            ggml_tensor* logits = ssvae::build_ss_vae_decode(cd, Hd, zin);
            ggml_set_output(logits);
            ggml_cgraph* gfd = new_graph(cd, 8192); ggml_build_forward_expand(gfd, logits);
            Hd.alloc_and_upload(gfd); Hd.upload_input_raw(zin, z_s); Hd.compute(gfd);
            std::vector<float> L((size_t)64*64*64); ggml_backend_tensor_get(logits,L.data(),0,L.size()*4);
            auto coordset = ssvae::logits_to_coords(L);   // set<array<int,3>> sorted (x,y,z)
            coords1.reserve(coordset.size()*4);
            for (auto& c : coordset) { coords1.push_back(0); coords1.push_back(c[0]); coords1.push_back(c[1]); coords1.push_back(c[2]); }
        }
    }
    int N1 = (int)coords1.size()/4;
    printf("[2] stage1 coords N1=%d @grid32\n", N1);

    // ====================== (3) stage2 cond + M2 DiT -> lr_slat ======================
    std::vector<float> lr_denorm;  // [N1,32]
    {
        // NAF@512 (guide=raw img512, vin=patchmap512) -> hr[1024,512,512]
        double tn=now_s();
        std::vector<float> naf_hr = run_naf(img512_raw, patchmap512, naf::CFG512, use_cuda);
        printf("[3] NAF@512  (%.1fs)\n", now_s()-tn);
        // proj cond grid32 @coords1: concat[proj_lr(patchmap512 32x32), proj_hr(naf 512x512)] -> [N1,2048]
        std::vector<float> cond2 = geo::proj_cond_shape(coords1.data(), N1, patchmap512.data(), 32, 32,
                                                        naf_hr.data(), 512, 512, dino::HID, 32, 512, cam);
        // coords_xyz for sparse rope
        std::vector<int32_t> cxyz((size_t)N1*3);
        for (int n=0;n<N1;n++) for (int j=0;j<3;j++) cxyz[n*3+j] = coords1[n*4+1+j];
        const int NEL = N1 * slatdit::INCH;
        std::vector<float> noise2 = gen.randn((int64_t)NEL);  // randn(N1,32)
        {
            M1Harness Hf(SL512_W, 1024, use_cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={slatdit::INCH,N1,1,1};      ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                   ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t g_ne[4]={1024,5,1,1};                ggml_tensor* gin=Hf.input("global",2,g_ne);
            int64_t p_ne[4]={slatdit::PROJ_IN,N1,1,1};   ggml_tensor* pin=Hf.input("proj",2,p_ne);
            ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, N1, xin, tin, gin, pin, cxyz.data());
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            std::vector<float> zero_g(global512.size(),0.f), zero_p(cond2.size(),0.f);
            auto forward=[&](const std::vector<float>& xx, float ts, bool cond){
                Hf.upload_input_raw(xin,xx); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(gin, cond?global512:zero_g); Hf.upload_input_raw(pin, cond?cond2:zero_p);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v;
            };
            double t=now_s();
            std::vector<float> lr_raw = geo::flow_sampler(NEL, noise2, 1e-5f, 7.5f, 0.5f, 3.0, 0.6, 1.0, 12, forward, "m2-dit");
            printf("[3] M2 DiT 12 steps  (%.1fs)\n", now_s()-t);
            lr_denorm = lr_raw;
            geo::denorm_inplace(lr_denorm, norm_mean.data(), norm_std.data(), slatdit::INCH);
        }
    }

    // ====================== (4) M3a upsample + quantize grid64 ======================
    std::vector<int32_t> coordsM;
    {
        double t=now_s();
        std::vector<int32_t> hr_coords = svp::m3a_upsample(coords1, lr_denorm, SHAPEDEC_W, use_cuda);
        int Nh = (int)hr_coords.size()/4;
        coordsM = geo::quantize_grid_unique(hr_coords.data(), Nh, 512, 64);
        printf("[4] M3a upsample Nh=%d -> quantize grid64 -> M=%d  (%.1fs)\n", Nh, (int)coordsM.size()/4, now_s()-t);
    }
    int M = (int)coordsM.size()/4;

    // ====================== (5)+(6) stage3b cond + M3b DiT -> shape_slat ======================
    std::vector<float> shape_denorm;  // [M,32]
    {
        std::vector<float> global1024, patchmap1024;
        { double t=now_s(); run_dinov3(img1024_norm, dino::CFG1024, use_cuda, global1024, patchmap1024);
          printf("[5] DINOv3@1024  (%.1fs)\n", now_s()-t); }
        double tn=now_s();
        std::vector<float> naf_hr = run_naf(img1024_raw, patchmap1024, naf::CFG1024, use_cuda);
        printf("[5] NAF@1024  (%.1fs)\n", now_s()-tn);
        std::vector<float> cond3b = geo::proj_cond_shape(coordsM.data(), M, patchmap1024.data(), 64, 64,
                                                         naf_hr.data(), 512, 512, dino::HID, 64, 1024, cam);
        std::vector<int32_t> cxyz((size_t)M*3);
        for (int n=0;n<M;n++) for (int j=0;j<3;j++) cxyz[n*3+j] = coordsM[n*4+1+j];
        const int NEL = M * slatdit::INCH;
        std::vector<float> noise3 = gen.randn((int64_t)NEL);  // randn(M,32)
        {
            M1Harness Hf(SL1024_W, 2048, use_cuda);
            ggml_context* cf = Hf.ctx;
            int64_t x_ne[4]={slatdit::INCH,M,1,1};       ggml_tensor* xin=Hf.input("x",2,x_ne);
            int64_t t_ne[4]={1,1,1,1};                   ggml_tensor* tin=Hf.input("t",1,t_ne);
            int64_t g_ne[4]={1024,5,1,1};                ggml_tensor* gin=Hf.input("global",2,g_ne);
            int64_t p_ne[4]={slatdit::PROJ_IN,M,1,1};    ggml_tensor* pin=Hf.input("proj",2,p_ne);
            ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, M, xin, tin, gin, pin, cxyz.data());
            ggml_set_output(vout);
            ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
            Hf.alloc_and_upload(gff);
            std::vector<float> zero_g(global1024.size(),0.f), zero_p(cond3b.size(),0.f);
            auto forward=[&](const std::vector<float>& xx, float ts, bool cond){
                Hf.upload_input_raw(xin,xx); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
                Hf.upload_input_raw(gin, cond?global1024:zero_g); Hf.upload_input_raw(pin, cond?cond3b:zero_p);
                Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v;
            };
            double t=now_s();
            std::vector<float> hr_raw = geo::flow_sampler(NEL, noise3, 1e-5f, 7.5f, 0.5f, 3.0, 0.6, 1.0, 12, forward, "m3b-dit");
            printf("[6] M3b DiT 12 steps  (%.1fs)\n", now_s()-t);
            shape_denorm = hr_raw;
            geo::denorm_inplace(shape_denorm, norm_mean.data(), norm_std.data(), slatdit::INCH);
        }
    }

    // ====================== (7) M4 decoder + mesh ======================
    double t=now_s();
    svae::Mesh mesh = svp::m4_decode_mesh(coordsM, shape_denorm, SHAPEDEC_W, use_cuda);
    printf("[7] M4 decoder+mesh: verts=%d faces=%d  (%.1fs)\n", mesh.N, mesh.F, now_s()-t);

    // write artifacts
    svae::write_ply("miku_geometry_e2e.ply", mesh.verts, mesh.faces);
    glb::write_glb("miku_geometry_e2e.glb", mesh.verts, mesh.faces);
    printf("[7] wrote miku_geometry_e2e.{ply,glb}\n");

    // ====================== validation vs golden (sanity) ======================
    printf("\n==== validation (fp32 chain vs fp16 golden — expect IoU ~0.97-0.99) ====\n");
    // grid64 token-set IoU vs golden stage3b coords
    { NpyArray gN = npy_load(std::string(GOLD)+"/stage3b_out/shape_slat_coords.npy");
      auto sg = cset4(gN.i32(), (int)gN.shape[0]);
      auto sm = cset4(coordsM.data(), M);
      int inter=0; for (auto&c:sm) if (sg.count(c)) inter++;
      int uni=(int)sm.size()+(int)sg.size()-inter;
      printf("  grid64 tokens: mine M=%d golden=%d  IoU=%.4f\n", M, (int)sg.size(), (double)inter/uni); }
    // mesh counts vs golden stage5_mesh
    { NpyArray gvN=npy_load(std::string(GOLD)+"/stage5_mesh/vertices.npy");
      NpyArray gfN=npy_load(std::string(GOLD)+"/stage5_mesh/faces.npy");
      printf("  mesh verts: mine=%d golden=%d (%.2f%%)  faces: mine=%d golden=%d (%.2f%%)\n",
             mesh.N, (int)gvN.shape[0], 100.0*mesh.N/gvN.shape[0],
             mesh.F, (int)gfN.shape[0], 100.0*mesh.F/gfN.shape[0]); }

    printf("\n==== geometry_e2e DONE  N1=%d M=%d verts=%d faces=%d  total %.1fs ====\n",
           N1, M, mesh.N, mesh.F, now_s()-t0);
    bool ok = (N1 > 900 && N1 < 1400 && M > 4000 && M < 6000 && mesh.N > 1000000 && mesh.F > 2000000);
    printf("[geometry_e2e] %s\n", ok ? "PASS (sensible geometry)" : "CHECK");
    return ok ? 0 : 1;
}
