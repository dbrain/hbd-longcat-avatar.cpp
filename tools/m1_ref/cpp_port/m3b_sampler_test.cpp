// M3b Stage-3b shape-SLat HR sampler + denorm from GOLDEN cond (grid 64).
// Identical arch to M2 (build_slat_dit_forward) — only the weights (slat_flow_1024), the
// HR coords (M=4734 golden) and the cond (stage3b_cond) differ. Validates the cascade at HR.
//   stage3b noise (refs/stage3b_noise.npy, 3rd seed-42 draw) + golden cond -> 12-step FlowEuler
//   (GS7.5/GR0.5/RT3.0/[0.6,1.0]) -> denorm -> shape_slat[M,32].
// Target == torch_stage3b_ref fp32 oracle (tight) + bf16 golden (cosine ~0.99).
#include "slat_dit_graph.hpp"
#include <cmath>
#include <cstdio>

static const char* FLOW_W = "weights_npy/slat_flow_1024";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";

static double vstd(const std::vector<float>& x) {
    double m = 0; for (float v : x) m += v; m /= x.size();
    double s = 0; for (float v : x) s += (v - m) * (v - m);
    return std::sqrt(s / x.size());
}
static void cmp(const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
    double ma = 0, sum = 0, dot = 0, na = 0, nb = 0; int n = (int)a.size();
    for (int i = 0; i < n; i++) { double d = std::fabs((double)a[i] - b[i]); ma = std::max(ma, d); sum += d;
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    printf("  [%s] maxabs=%.3e meanabs=%.3e cosine=%.6f\n", tag, ma, sum / n, dot / (std::sqrt(na * nb) + 1e-12));
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const float SM = 1e-5f, GS = 7.5f, GR = 0.5f;
    const double RT = 3.0, IV0 = 0.6, IV1 = 1.0;  // double t_seq (interval-boundary safe; see M2)
    const int STEPS = 12;

    NpyArray cN = npy_load(std::string(GOLD) + "/stage3b_cond/proj_coords.npy");
    int N = (int)cN.shape[0];
    const int32_t* c4 = cN.i32();
    std::vector<int32_t> coords_xyz((size_t)N * 3);
    for (int n = 0; n < N; n++) for (int j = 0; j < 3; j++) coords_xyz[n * 3 + j] = c4[n * 4 + 1 + j];
    const int NEL = N * slatdit::INCH;

    M1Harness Hf(FLOW_W, 2048, use_cuda);
    ggml_context* cf = Hf.ctx;
    int64_t x_ne[4] = {slatdit::INCH, N, 1, 1};    ggml_tensor* xin = Hf.input("x", 2, x_ne);
    int64_t t_ne[4] = {1, 1, 1, 1};                ggml_tensor* tin = Hf.input("t", 1, t_ne);
    int64_t g_ne[4] = {1024, 5, 1, 1};             ggml_tensor* gin = Hf.input("global", 2, g_ne);
    int64_t p_ne[4] = {slatdit::PROJ_IN, N, 1, 1}; ggml_tensor* pin = Hf.input("proj", 2, p_ne);
    ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, N, xin, tin, gin, pin, coords_xyz.data());
    ggml_set_output(vout);
    ggml_cgraph* gff = new_graph(cf, 32768);
    ggml_build_forward_expand(gff, vout);
    Hf.alloc_and_upload(gff);

    NpyArray gN = npy_load(std::string(GOLD) + "/stage3b_cond/global.npy");
    NpyArray pN = npy_load(std::string(GOLD) + "/stage3b_cond/proj_feats.npy");
    std::vector<float> cond_g(gN.f32(), gN.f32() + gN.numel());
    std::vector<float> cond_p(pN.f32(), pN.f32() + pN.numel());
    std::vector<float> zero_g(cond_g.size(), 0.f), zero_p(cond_p.size(), 0.f);

    auto forward = [&](const std::vector<float>& x, float t_scaled, bool cond) {
        Hf.upload_input_raw(xin, x);
        std::vector<float> tv{t_scaled}; Hf.upload_input_raw(tin, tv);
        Hf.upload_input_raw(gin, cond ? cond_g : zero_g);
        Hf.upload_input_raw(pin, cond ? cond_p : zero_p);
        Hf.compute(gff);
        // DIAG: dump captured block-0 q/k/v (PIXAL3D_FA_CAPTURE) once → cap_{q,k,v}.bin for replay.
        if (std::getenv("PIXAL3D_FA_CAPTURE")) { static int fc=0; int capfwd=getenv("CAPFWD")?atoi(getenv("CAPFWD")):0;
          bool fire = (fc==capfwd); fc++; if (fire) {
            // find the FIRST block whose flash output is NaN (cap_out_<n>)
            for (int n=0;n<40;n++){ char nm[24]; snprintf(nm,sizeof(nm),"cap_out_%d",n);
                ggml_tensor* t=ggml_get_tensor(cf,nm); if(!t) break;
                std::vector<float> buf(ggml_nelements(t)); ggml_backend_tensor_get(t,buf.data(),0,ggml_nbytes(t));
                size_t nan=0; float amax=0; for(float x:buf){ if(std::isnan(x)||std::isinf(x))nan++; else amax=std::max(amax,std::fabs(x)); }
                printf("[cap] block %2d flash out: nan=%zu absmax=%.1f\n", n, nan, amax);
                if (nan>0) { printf("[cap] >>> FIRST NaN at block %d\n", n); break; }
            }
            for (const char* nm : {"cap_q","cap_k","cap_v"}) {
                ggml_tensor* t = ggml_get_tensor(cf, nm); if (!t) continue;
                std::vector<float> buf(ggml_nelements(t)); ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
                FILE* fp = fopen((std::string(nm)+".bin").c_str(), "wb"); fwrite(buf.data(), 4, buf.size(), fp); fclose(fp);
            } fflush(stdout); }
        }
        std::vector<float> v(NEL);
        ggml_backend_tensor_get(vout, v.data(), 0, NEL * sizeof(float));
        { static int fwd=0; size_t nan=0; float amax=0; for(float x:v){ if(std::isnan(x)||std::isinf(x))nan++; else amax=std::max(amax,std::fabs(x)); }
          if (std::getenv("FWD_DBG")) printf("    [fwd %d] vout nan=%zu absmax=%.2f\n", fwd, nan, amax);
          fwd++; }
        return v;
    };
    auto pred_to_x0 = [&](const std::vector<float>& xt, float t, const std::vector<float>& pr) {
        std::vector<float> o(NEL); float a = (1 - SM), b = (SM + (1 - SM) * t);
        for (int i = 0; i < NEL; i++) o[i] = a * xt[i] - b * pr[i]; return o; };
    auto x0_to_pred = [&](const std::vector<float>& xt, float t, const std::vector<float>& x0) {
        std::vector<float> o(NEL); float a = (1 - SM), b = (SM + (1 - SM) * t);
        for (int i = 0; i < NEL; i++) o[i] = (a * xt[i] - x0[i]) / b; return o; };

    NpyArray noiseN = npy_load(std::string(REFS) + "/stage3b_noise.npy");
    std::vector<float> x(noiseN.f32(), noiseN.f32() + NEL);
    std::vector<double> tseq(STEPS + 1);
    for (int i = 0; i <= STEPS; i++) { double lt = 1.0 - (double)i / STEPS; tseq[i] = RT * lt / (1 + (RT - 1) * lt); }
    for (int s = 0; s < STEPS; s++) {
        double t = tseq[s], tp = tseq[s + 1];
        bool in_iv = (t >= IV0 && t <= IV1);
        printf("  step %2d/%d t=%.6f %s\n", s + 1, STEPS, t, in_iv ? "[cfg]" : "[cond]"); fflush(stdout);
        std::vector<float> v;
        if (in_iv) {
            std::vector<float> vp = forward(x, (float)(1000.0 * t), true);
            std::vector<float> vn = forward(x, (float)(1000.0 * t), false);
            std::vector<float> pred(NEL);
            for (int i = 0; i < NEL; i++) pred[i] = GS * vp[i] + (1 - GS) * vn[i];
            std::vector<float> x0p = pred_to_x0(x, (float)t, vp), x0c = pred_to_x0(x, (float)t, pred);
            double r = vstd(x0p) / vstd(x0c);
            std::vector<float> x0(NEL);
            for (int i = 0; i < NEL; i++) { float resc = x0c[i] * (float)r; x0[i] = GR * resc + (1 - GR) * x0c[i]; }
            v = x0_to_pred(x, (float)t, x0);
        } else v = forward(x, (float)(1000.0 * t), true);
        for (int i = 0; i < NEL; i++) x[i] -= (float)(t - tp) * v[i];
    }
    std::vector<float> lr_raw = x;

    NpyArray mN = npy_load(std::string(REFS) + "/shape_slat_norm_mean.npy");
    NpyArray sN = npy_load(std::string(REFS) + "/shape_slat_norm_std.npy");
    const float* mean = mN.f32(); const float* sd = sN.f32();
    std::vector<float> denorm(NEL);
    for (int i = 0; i < NEL; i++) { int c = i % slatdit::INCH; denorm[i] = lr_raw[i] * sd[c] + mean[c]; }

    printf("[m3b] backend=%s N=%d\n", use_cuda ? "cuda" : "cpu", N);
    NpyArray orefN = npy_load(std::string(REFS) + "/torch_shape_slat_fp32.npy");
    NpyArray odenN = npy_load(std::string(REFS) + "/torch_shape_slat_denorm_fp32.npy");
    NpyArray goldN = npy_load(std::string(GOLD) + "/stage3b_out/shape_slat_feats.npy");
    std::vector<float> oref(orefN.f32(), orefN.f32() + NEL);
    std::vector<float> oden(odenN.f32(), odenN.f32() + NEL);
    std::vector<float> gold(goldN.f32(), goldN.f32() + NEL);
    cmp("pre-denorm vs torch_fp32 (TIGHT)", lr_raw, oref);
    cmp("denorm    vs torch_fp32 (TIGHT)", denorm, oden);
    cmp("denorm    vs golden bf16     ", denorm, gold);

    double ma = 0; for (int i = 0; i < NEL; i++) ma = std::max(ma, (double)std::fabs(lr_raw[i] - oref[i]));
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < NEL; i++) { dot += (double)denorm[i] * gold[i]; na += (double)denorm[i] * denorm[i]; nb += (double)gold[i] * gold[i]; }
    double cosg = dot / (std::sqrt(na * nb) + 1e-12);
    bool ok = (ma < (use_cuda ? 3e-1 : 5e-3)) && (cosg > 0.99);
    printf("[m3b] %s (pre-denorm maxabs vs fp32=%.3e, cosine vs golden=%.6f)\n", ok ? "PASS" : "FAIL", ma, cosg);
    return ok ? 0 : 1;
}
