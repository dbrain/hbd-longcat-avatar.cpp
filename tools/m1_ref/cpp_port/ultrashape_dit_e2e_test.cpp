// Single-forward DiT parity in the E2E REGIME (real cond Tc=3970, N=512, t=0 / t=0.5) — isolates the
// e2e divergence. dit_test passes cosine 1.0 on synthetic K=256/Tc=32/t=0.7; this runs us_refine_dit
// on the SAME inputs the e2e uses and compares to the banked single-forward golden (capture_dit_e2e.py).
//   ./build.sh ultrashape_dit_e2e_test [cuda]   (cuda = F16 GGUF via PIXAL3D_GGUF_DIR; F16==fp32 here)
#include "ultrashape_dit.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* GD  = "/mnt/hdd/3d/avatar-shootout/e2e_goldens";
static const char* DGD = "/mnt/hdd/3d/avatar-shootout/dit_e2e_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/dit";

static double cos_vec(const float* a, const float* b, int64_t n) {
    double dot=0,na=0,nb=0; for (int64_t i=0;i<n;i++){dot+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];}
    return dot/(std::sqrt(na)*std::sqrt(nb)+1e-30);
}
static double meanabs(const float* a, const float* b, int64_t n){ double s=0; for(int64_t i=0;i<n;i++) s+=std::fabs((double)a[i]-b[i]); return s/n; }

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    UsDitCfg cfg;
    M1Harness H(WDIR, 8192, use_cuda);
    ggml_context* ctx = H.ctx;

    NpyArray xin = npy_load(std::string(GD) + "/init_latents.npy");   // [1,512,64]
    int64_t N = xin.shape[1];
    NpyArray vox = npy_load(std::string(GD) + "/voxel_cond.npy");     // [1,512,3] f32 (capture save()s float)
    std::vector<float> voxf(vox.f32(), vox.f32() + vox.numel());
    NpyArray cnd = npy_load(std::string(GD) + "/cond_main.npy");      // [1,3970,1024]
    NpyArray unc = npy_load(std::string(GD) + "/uncond_main.npy");
    int64_t Tc = cnd.shape[1];
    std::vector<float> cond_main(cnd.f32(), cnd.f32()+cnd.numel());
    std::vector<float> uncond_main(unc.f32(), unc.f32()+unc.numel());
    std::vector<float> xv(xin.f32(), xin.f32()+xin.numel());

    std::vector<float> rcos, rsin; us_rope_3d(voxf.data(), N, cfg, rcos, rsin);
    int dim = cfg.head_dim(); int64_t S = 1 + N;
    int64_t rope_ne[4] = {dim,1,S,1};
    ggml_tensor* rcos_t = H.const_tensor("rope_cos", 3, rope_ne, rcos);
    ggml_tensor* rsin_t = H.const_tensor("rope_sin", 3, rope_ne, rsin);
    int64_t ts_ne[4] = {cfg.hidden,1,1,1};
    ggml_tensor* ts_t = H.input("ts_embed", 2, ts_ne);
    int64_t x_ne[4] = {cfg.in_channels, N, 1, 1};
    ggml_tensor* x_lat = H.input("x_lat", 2, x_ne);
    int64_t c_ne[4] = {cfg.context_dim, Tc, 1, 1};
    ggml_tensor* cond = H.input("cond", 2, c_ne);
    ggml_tensor* out = us_refine_dit(H, ctx, cfg, x_lat, ts_t, cond, rcos_t, rsin_t);
    ggml_set_output(out);
    ggml_cgraph* gf = new_graph(ctx, 65536);
    ggml_build_forward_expand(gf, out);
    H.alloc_and_upload(gf);

    std::vector<float> got((size_t)N*cfg.in_channels);
    auto run = [&](float t, const std::vector<float>& c, const char* gname){
        std::vector<float> ts = us_timesteps_embed(t, cfg.hidden);
        H.upload_input_raw(ts_t, ts);
        H.upload_input_raw(x_lat, xv);
        H.upload_input_raw(cond, c);
        H.compute(gf);
        ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
        NpyArray g = npy_load(std::string(DGD) + "/" + gname + ".npy");
        double cc = cos_vec(got.data(), g.f32(), (int64_t)got.size());
        double ma = meanabs(got.data(), g.f32(), (int64_t)got.size());
        double mm=0; for(float v:got) mm+=std::fabs(v); mm/=got.size();
        double mg=0; for(int64_t i=0;i<(int64_t)got.size();i++) mg+=std::fabs(g.f32()[i]); mg/=got.size();
        printf("  [%-16s] cosine=%.6f meanabs=%.3e  |mine|=%.5f |gold|=%.5f\n", gname, cc, ma, mm, mg);
        return cc;
    };
    printf("[dit_e2e] backend=%s N=%lld Tc=%lld\n", use_cuda?"cuda":"cpu", (long long)N, (long long)Tc);
    double c1 = run(0.0f, cond_main,   "pred_cond_t0");
    double c2 = run(0.0f, uncond_main, "pred_uncond_t0");
    double c3 = run(0.5f, cond_main,   "pred_cond_t05");
    double c4 = run(0.5f, uncond_main, "pred_uncond_t05");
    bool ok = c1>0.999 && c2>0.999 && c3>0.999 && c4>0.999;
    printf("[dit_e2e] %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
