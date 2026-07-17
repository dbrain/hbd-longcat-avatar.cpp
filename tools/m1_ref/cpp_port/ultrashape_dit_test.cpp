// Validate the UltraShape RefineDiT port (ultrashape_dit.hpp) against banked goldens.
//   x_in[1,K,64] + t + cond[1,Tc,1024] + voxel_cond[1,K,3]  ->  noise_pred[1,K,64]
// Also host-checks the 3D-RoPE precompute against rope_cos_vox/sin goldens.
//   ./build.sh ultrashape_dit_test [cuda]      (CPU = fp32 oracle; 12GB fp32 weights need CPU/host RAM)
#include "ultrashape_dit.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/dit_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/dit";

static double cos_vec(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return dot / (std::sqrt(na)*std::sqrt(nb) + 1e-30);
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    UsDitCfg cfg;
    M1Harness H(WDIR, 8192, use_cuda);
    ggml_context* ctx = H.ctx;

    // --- load inputs ---
    NpyArray xin = npy_load(std::string(GDIR) + "/x_in.npy");        // [1,K,64]
    int64_t K = xin.shape[1];
    NpyArray vox = npy_load(std::string(GDIR) + "/voxel_cond.npy");  // [1,K,3]
    NpyArray cnd = npy_load(std::string(GDIR) + "/cond.npy");        // [1,Tc,1024]
    int64_t Tc = cnd.shape[1];
    NpyArray tv  = npy_load(std::string(GDIR) + "/t_val.npy");       // [1]
    float tval = tv.f32()[0];

    // --- host RoPE precompute + self-check vs goldens (voxel tokens only) ---
    std::vector<float> rcos, rsin;
    us_rope_3d(vox.f32(), K, cfg, rcos, rsin);                       // [(1+K)*128] each
    {
        NpyArray gc = npy_load(std::string(GDIR) + "/rope_cos_vox.npy");  // [1,K,128]
        NpyArray gs = npy_load(std::string(GDIR) + "/rope_sin_vox.npy");
        int dim = cfg.head_dim();
        double cc = cos_vec(rcos.data() + dim, gc.f32(), (int64_t)K*dim);  // skip token-0 identity
        double cs = cos_vec(rsin.data() + dim, gs.f32(), (int64_t)K*dim);
        printf("[us_dit] RoPE host-check  cos cosine=%.7f  sin cosine=%.7f\n", cc, cs);
    }

    int dim = cfg.head_dim();
    int64_t S = 1 + K;
    // RoPE const tensors [head_dim, 1, S] (broadcast over heads)
    int64_t rope_ne[4] = {dim, 1, S, 1};
    ggml_tensor* rcos_t = H.const_tensor("rope_cos", 3, rope_ne, rcos);
    ggml_tensor* rsin_t = H.const_tensor("rope_sin", 3, rope_ne, rsin);

    // timesteps embedding (host) -> [hidden, 1]
    std::vector<float> ts = us_timesteps_embed(tval, cfg.hidden);
    int64_t ts_ne[4] = {cfg.hidden, 1, 1, 1};
    ggml_tensor* ts_t = H.const_tensor("ts_embed", 2, ts_ne, ts);

    int64_t x_ne[4] = {cfg.in_channels, K, 1, 1};
    ggml_tensor* x_lat = H.input("x_lat", 2, x_ne);
    int64_t c_ne[4] = {cfg.context_dim, Tc, 1, 1};
    ggml_tensor* cond = H.input("cond", 2, c_ne);

    ggml_tensor* out = us_refine_dit(H, ctx, cfg, x_lat, ts_t, cond, rcos_t, rsin_t);  // [64, K]
    ggml_set_output(out);

    ggml_cgraph* gf = new_graph(ctx, 65536);
    ggml_build_forward_expand(gf, out);
    H.alloc_and_upload(gf);
    H.upload_input_npy(x_lat, std::string(GDIR) + "/x_in.npy");
    H.upload_input_npy(cond, std::string(GDIR) + "/cond.npy");
    H.compute(gf);

    printf("[us_dit] backend=%s K=%lld Tc=%lld t=%.3f\n", use_cuda ? "cuda" : "cpu", (long long)K, (long long)Tc, tval);
    CmpStats s = compare_to_npy(H, out, std::string(GDIR) + "/noise_pred.npy", true, "noise_pred");
    bool ok = s.meanabs < 5e-3;
    printf("[us_dit] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
