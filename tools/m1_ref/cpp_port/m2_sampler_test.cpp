// M2 Stage-2 shape-SLat LR sampler + denorm E2E from GOLDEN cond.
//   seed-42 stage2 noise (refs/stage2_noise.npy) + golden (global, proj) over the 1126
//   golden coords -> FlowEulerGuidanceIntervalSampler(12 steps, shape params:
//   GS 7.5 / GR 0.5 / RT 3.0 / interval [0.6,1.0]) -> lr_slat[N,32] -> denorm.
//
// Mirrors sampler_test.cpp (stage-1) but with the sparse DiT (slat_dit_graph), N voxel
// tokens, in/out 32, proj 2048, and sparse-coord RoPE. The sampler scalar arithmetic
// runs on the host (identical to stage-1; sparse .std == population std over all N*32).
// Targets (== torch_stage2_ref fp32 oracle):
//   lr_slat(pre-denorm) vs torch_lr_slat_fp32   tight (~1e-3 CPU);
//   lr_slat(denorm)     vs golden(bf16)         ~1e-1 abs / cosine ~0.998 (bf16 torso).
#include "slat_dit_graph.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>

static const char* FLOW_W = "weights_npy/slat_flow_512";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";

static double vstd(const std::vector<float>& x) {  // population std over all elems (ratio-invariant)
    double m = 0; for (float v : x) m += v; m /= x.size();
    double s = 0; for (float v : x) s += (v - m) * (v - m);
    return std::sqrt(s / x.size());
}
static void stat(const char* tag, const std::vector<float>& x) {
    int nan = 0; double mn = 1e30, mx = -1e30;
    for (float v : x) { if (std::isnan(v) || std::isinf(v)) nan++; else { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); } }
    printf("    %-6s nan/inf=%d min=%.4g max=%.4g\n", tag, nan, mn, mx);
}
static void cmp(const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
    double ma = 0, sum = 0, dot = 0, na = 0, nb = 0;
    int n = (int)a.size();
    for (int i = 0; i < n; i++) {
        double d = std::fabs((double)a[i] - b[i]); ma = std::max(ma, d); sum += d;
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    printf("  [%s] maxabs=%.3e meanabs=%.3e cosine=%.6f\n", tag, ma, sum / n, dot / (std::sqrt(na * nb) + 1e-12));
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const float SM = 1e-5f, GS = 7.5f, GR = 0.5f;
    // t_seq / guidance-interval decisions MUST be double: torch builds t_seq via numpy
    // float64 and stage-2's Mobius warp (RT=3) lands a step EXACTLY at t=0.6, the interval
    // boundary. In float32 that t rounds to 0.5999999 -> CFG is skipped where torch applies
    // it (guidance 7.5 vs 1) -> the flow ODE amplifies the flip to ~1.3 maxabs. (Stage-1
    // RT=5 never hit the boundary, so float32 was fine there.)
    const double RT = 3.0, IV0 = 0.6, IV1 = 1.0;
    const int STEPS = 12;

    // coords [N,4] -> coords_xyz [N,3] (golden 1126 coords; cond computed over these)
    NpyArray cN = npy_load(std::string(GOLD) + "/stage2_cond/proj_coords.npy");
    int N = (int)cN.shape[0];
    const int32_t* c4 = cN.i32();
    std::vector<int32_t> coords_xyz((size_t)N * 3);
    for (int n = 0; n < N; n++) for (int j = 0; j < 3; j++) coords_xyz[n * 3 + j] = c4[n * 4 + 1 + j];
    const int NEL = N * slatdit::INCH;  // N*32

    // ===== DiT forward graph (slat_flow_512) =====
    M1Harness Hf(FLOW_W, 1024, use_cuda);
    ggml_context* cf = Hf.ctx;
    int64_t x_ne[4] = {slatdit::INCH, N, 1, 1};   ggml_tensor* xin = Hf.input("x", 2, x_ne);
    int64_t t_ne[4] = {1, 1, 1, 1};               ggml_tensor* tin = Hf.input("t", 1, t_ne);
    int64_t g_ne[4] = {1024, 5, 1, 1};            ggml_tensor* gin = Hf.input("global", 2, g_ne);
    int64_t p_ne[4] = {slatdit::PROJ_IN, N, 1, 1}; ggml_tensor* pin = Hf.input("proj", 2, p_ne);
    ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, N, xin, tin, gin, pin, coords_xyz.data());
    ggml_set_output(vout);
    ggml_cgraph* gff = new_graph(cf, 32768);
    ggml_build_forward_expand(gff, vout);
    Hf.alloc_and_upload(gff);

    NpyArray gN = npy_load(std::string(GOLD) + "/stage2_cond/global.npy");      // [1,5,1024]
    NpyArray pN = npy_load(std::string(GOLD) + "/stage2_cond/proj_feats.npy");  // [N,2048]
    std::vector<float> cond_g(gN.f32(), gN.f32() + gN.numel());
    std::vector<float> cond_p(pN.f32(), pN.f32() + pN.numel());
    std::vector<float> zero_g(cond_g.size(), 0.f), zero_p(cond_p.size(), 0.f);

    auto forward = [&](const std::vector<float>& x, float t_scaled, bool cond) {
        Hf.upload_input_raw(xin, x);
        std::vector<float> tv{t_scaled}; Hf.upload_input_raw(tin, tv);
        Hf.upload_input_raw(gin, cond ? cond_g : zero_g);
        Hf.upload_input_raw(pin, cond ? cond_p : zero_p);
        Hf.compute(gff);
        std::vector<float> v(NEL);
        ggml_backend_tensor_get(vout, v.data(), 0, NEL * sizeof(float));
        return v;
    };
    auto pred_to_x0 = [&](const std::vector<float>& xt, float t, const std::vector<float>& pr) {
        std::vector<float> o(NEL); float a = (1 - SM), b = (SM + (1 - SM) * t);
        for (int i = 0; i < NEL; i++) o[i] = a * xt[i] - b * pr[i];
        return o;
    };
    auto x0_to_pred = [&](const std::vector<float>& xt, float t, const std::vector<float>& x0) {
        std::vector<float> o(NEL); float a = (1 - SM), b = (SM + (1 - SM) * t);
        for (int i = 0; i < NEL; i++) o[i] = (a * xt[i] - x0[i]) / b;
        return o;
    };

    // ===== sampler loop =====
    NpyArray noiseN = npy_load(std::string(REFS) + "/stage2_noise.npy");  // [N,32]
    std::vector<float> x(noiseN.f32(), noiseN.f32() + NEL);
    std::vector<double> tseq(STEPS + 1);
    for (int i = 0; i <= STEPS; i++) { double lt = 1.0 - (double)i / STEPS; tseq[i] = RT * lt / (1 + (RT - 1) * lt); }
    for (int s = 0; s < STEPS; s++) {
        double t = tseq[s], tp = tseq[s + 1];
        bool in_iv = (t >= IV0 && t <= IV1);
        printf("  step %2d/%d t=%.6f %s\n", s + 1, STEPS, t, in_iv ? "[cfg]" : "[cond]");
        std::vector<float> v;
        if (in_iv) {
            std::vector<float> vp = forward(x, (float)(1000.0 * t), true);
            std::vector<float> vn = forward(x, (float)(1000.0 * t), false);
            std::vector<float> pred(NEL);
            for (int i = 0; i < NEL; i++) pred[i] = GS * vp[i] + (1 - GS) * vn[i];
            std::vector<float> x0p = pred_to_x0(x, t, vp), x0c = pred_to_x0(x, t, pred);
            double sp = vstd(x0p), sc = vstd(x0c), r = sp / sc;
            printf("    std_pos=%.4g std_cfg=%.4g r=%.4g\n", sp, sc, r);
            std::vector<float> x0(NEL);
            for (int i = 0; i < NEL; i++) { float resc = x0c[i] * (float)r; x0[i] = GR * resc + (1 - GR) * x0c[i]; }
            v = x0_to_pred(x, (float)t, x0);
        } else {
            v = forward(x, (float)(1000.0 * t), true);
        }
        for (int i = 0; i < NEL; i++) x[i] -= (float)(t - tp) * v[i];
        stat("x", x); fflush(stdout);
    }
    std::vector<float> lr_raw = x;  // pre-denorm [N,32] token-major

    // ===== denorm: slat = slat*std + mean (32-dim, per channel c = i%32) =====
    NpyArray mN = npy_load(std::string(REFS) + "/shape_slat_norm_mean.npy");
    NpyArray sN = npy_load(std::string(REFS) + "/shape_slat_norm_std.npy");
    const float* mean = mN.f32(); const float* sd = sN.f32();
    std::vector<float> lr_denorm(NEL);
    for (int i = 0; i < NEL; i++) { int c = i % slatdit::INCH; lr_denorm[i] = lr_raw[i] * sd[c] + mean[c]; }

    // ===== compare =====
    printf("[m2_sampler] backend=%s  N=%d\n", use_cuda ? "cuda" : "cpu", N);
    NpyArray orefN = npy_load(std::string(REFS) + "/torch_lr_slat_fp32.npy");        // pre-denorm fp32
    NpyArray odenN = npy_load(std::string(REFS) + "/torch_lr_slat_denorm_fp32.npy"); // post-denorm fp32
    NpyArray goldN = npy_load(std::string(GOLD) + "/stage2_out/lr_slat_feats.npy");  // denorm bf16
    std::vector<float> oref(orefN.f32(), orefN.f32() + NEL);
    std::vector<float> oden(odenN.f32(), odenN.f32() + NEL);
    std::vector<float> gold(goldN.f32(), goldN.f32() + NEL);
    cmp("pre-denorm vs torch_fp32 (TIGHT)", lr_raw, oref);
    cmp("denorm    vs torch_fp32 (TIGHT)", lr_denorm, oden);
    cmp("denorm    vs golden bf16     ", lr_denorm, gold);

    // gate: tight vs fp32 oracle (CPU ~1e-3, CUDA looser); cosine vs golden ~0.998
    double ma = 0; for (int i = 0; i < NEL; i++) ma = std::max(ma, (double)std::fabs(lr_raw[i] - oref[i]));
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < NEL; i++) { dot += (double)lr_denorm[i] * gold[i]; na += (double)lr_denorm[i] * lr_denorm[i]; nb += (double)gold[i] * gold[i]; }
    double cosg = dot / (std::sqrt(na * nb) + 1e-12);
    bool ok = (ma < (use_cuda ? 2e-1 : 5e-3)) && (cosg > 0.995);
    printf("[m2_sampler] %s (pre-denorm maxabs vs fp32=%.3e, cosine vs golden=%.6f)\n",
           ok ? "PASS" : "FAIL", ma, cosg);
    return ok ? 0 : 1;
}
