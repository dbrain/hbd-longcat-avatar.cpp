// Stage-1 sampler + VAE E2E from GOLDEN cond (isolates sampler/DiT/VAE from DINOv3).
//   seed-42 noise + golden (global,proj) -> FlowEulerGuidanceIntervalSampler(12 steps)
//   -> z_s -> SS VAE decode -> coords.
//
// Mirrors tools/m1_ref/ss_dit.py FlowSampler (CFG + guidance-interval + std-rescale +
// Mobius t-warp) exactly. The DiT forward is the validated ss_dit_graph; the sampler
// scalar arithmetic runs on the host. Targets (== torch_stage1_ref fp32 closure):
//   coords N=1120, IoU 0.9859 vs golden(1126);  z_s vs torch_z_s_fp32 tight (~1e-3).
#include "ss_dit_graph.hpp"
#include "ss_vae_graph.hpp"
#include <cmath>
#include <cstdio>

static const char* FLOW_W = "weights_npy/ss_flow";
static const char* DEC_W = "weights_npy/ss_dec";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";
static const int NEL = ssdit::SEQ * ssdit::INCH;  // 32768

static double vstd(const std::vector<float>& x) {
    double m = 0; for (float v : x) m += v; m /= x.size();
    double s = 0; for (float v : x) s += (v - m) * (v - m);
    return std::sqrt(s / x.size());
}
static void stat(const char* tag, const std::vector<float>& x) {
    int nan = 0; double mn = 1e30, mx = -1e30;
    for (float v : x) { if (std::isnan(v) || std::isinf(v)) nan++; else { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); } }
    printf("    %-6s nan/inf=%d min=%.4g max=%.4g\n", tag, nan, mn, mx);
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const float SM = 1e-5f, GS = 7.5f, GR = 0.7f, RT = 5.0f, IV0 = 0.6f, IV1 = 1.0f;
    const int STEPS = 12;

    // ===== DiT forward graph (ss_flow) =====
    M1Harness Hf(FLOW_W, 512, use_cuda);
    ggml_context* cf = Hf.ctx;
    int64_t x_ne[4] = {ssdit::SEQ, ssdit::INCH, 1, 1};
    ggml_tensor* xin = Hf.input("x", 2, x_ne);
    int64_t t_ne[4] = {1, 1, 1, 1};
    ggml_tensor* tin = Hf.input("t", 1, t_ne);
    int64_t g_ne[4] = {1024, 5, 1, 1};
    ggml_tensor* gin = Hf.input("global", 2, g_ne);
    int64_t p_ne[4] = {1024, ssdit::SEQ, 1, 1};
    ggml_tensor* pin = Hf.input("proj", 2, p_ne);
    ggml_tensor* vout = ssdit::build_ss_dit_forward(cf, Hf, xin, tin, gin, pin);
    ggml_set_output(vout);
    ggml_cgraph* gff = new_graph(cf, 32768);
    ggml_build_forward_expand(gff, vout);
    Hf.alloc_and_upload(gff);
    NpyArray gN = npy_load(std::string(REFS) + "/dino_global.npy");   // golden cond global
    NpyArray pN = npy_load(std::string(REFS) + "/proj.npy");          // golden cond proj
    std::vector<float> cond_g(gN.f32(), gN.f32() + gN.numel());
    std::vector<float> cond_p(pN.f32(), pN.f32() + pN.numel());
    std::vector<float> zero_g(cond_g.size(), 0.f), zero_p(cond_p.size(), 0.f);

    // forward(x,t_scaled,use_cond) -> v_host[NEL]
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
        std::vector<float> o(NEL);
        float a = (1 - SM), b = (SM + (1 - SM) * t);
        for (int i = 0; i < NEL; i++) o[i] = a * xt[i] - b * pr[i];
        return o;
    };
    auto x0_to_pred = [&](const std::vector<float>& xt, float t, const std::vector<float>& x0) {
        std::vector<float> o(NEL);
        float a = (1 - SM), b = (SM + (1 - SM) * t);
        for (int i = 0; i < NEL; i++) o[i] = (a * xt[i] - x0[i]) / b;
        return o;
    };

    // ===== sampler loop =====
    NpyArray noiseN = npy_load(std::string(REFS) + "/noise_seed42.npy");
    std::vector<float> x(noiseN.f32(), noiseN.f32() + NEL);
    std::vector<float> tseq(STEPS + 1);
    for (int i = 0; i <= STEPS; i++) {
        float lt = 1.0f - (float)i / STEPS;          // linspace(1,0,13)
        tseq[i] = RT * lt / (1 + (RT - 1) * lt);     // Mobius warp
    }
    for (int s = 0; s < STEPS; s++) {
        float t = tseq[s], tp = tseq[s + 1];
        std::vector<float> v;
        printf("  step %2d/%d t=%.4f %s\n", s + 1, STEPS, t, (t >= IV0 && t <= IV1) ? "[cfg]" : "[cond]");
        if (t >= IV0 && t <= IV1) {
            std::vector<float> vp = forward(x, 1000.f * t, true);
            std::vector<float> vn = forward(x, 1000.f * t, false);
            stat("vp", vp); stat("vn", vn);
            std::vector<float> pred(NEL);
            for (int i = 0; i < NEL; i++) pred[i] = GS * vp[i] + (1 - GS) * vn[i];
            // std-rescale
            std::vector<float> x0p = pred_to_x0(x, t, vp), x0c = pred_to_x0(x, t, pred);
            double sp = vstd(x0p), sc = vstd(x0c), r = sp / sc;
            printf("    std_pos=%.4g std_cfg=%.4g r=%.4g\n", sp, sc, r);
            std::vector<float> x0(NEL);
            for (int i = 0; i < NEL; i++) { float resc = x0c[i] * (float)r; x0[i] = GR * resc + (1 - GR) * x0c[i]; }
            v = x0_to_pred(x, t, x0);
        } else {
            v = forward(x, 1000.f * t, true);
            stat("v", v);
        }
        for (int i = 0; i < NEL; i++) x[i] -= (t - tp) * v[i];
        stat("x", x);
        fflush(stdout);
    }
    std::vector<float>& z_s = x;  // [NEL] ch-major == VAE z input layout

    // ===== compare z_s =====
    printf("[sampler] backend=%s\n", use_cuda ? "cuda" : "cpu");
    {
        NpyArray gold = npy_load(std::string(GOLD) + "/stage1_ssdec/z_s.npy");  // bf16 torso
        double ma = 0, sum = 0; const float* g = gold.f32();
        for (int i = 0; i < NEL; i++) { double d = std::fabs(z_s[i] - g[i]); ma = std::max(ma, d); sum += d; }
        printf("  z_s vs golden(bf16): maxabs=%.3e median(mean)=%.3e (expect ~1.2 / ~3.6e-3)\n", ma, sum / NEL);
    }
    {
        std::string tp = std::string(REFS) + "/torch_z_s_fp32.npy";
        std::ifstream f(tp);
        if (f.good()) {
            NpyArray tz = npy_load(tp);
            double ma = 0, sum = 0; const float* g = tz.f32();
            for (int i = 0; i < NEL; i++) { double d = std::fabs(z_s[i] - g[i]); ma = std::max(ma, d); sum += d; }
            printf("  z_s vs torch_fp32: maxabs=%.3e meanabs=%.3e (tight fp32 cross-check)\n", ma, sum / NEL);
        } else {
            printf("  (torch_z_s_fp32.npy not ready yet — skipping tight cross-check)\n");
        }
    }

    // ===== VAE decode -> coords =====
    M1Harness Hd(DEC_W, 256, use_cuda);
    ggml_context* cd = Hd.ctx;
    int64_t z_ne[4] = {16, 16, 16, 8};
    ggml_tensor* zin = Hd.input("z_s", 4, z_ne);
    ggml_tensor* logits = ssvae::build_ss_vae_decode(cd, Hd, zin);
    ggml_set_output(logits);
    ggml_cgraph* gfd = new_graph(cd, 8192);
    ggml_build_forward_expand(gfd, logits);
    Hd.alloc_and_upload(gfd);
    Hd.upload_input_raw(zin, z_s);
    Hd.compute(gfd);
    std::vector<float> L((size_t)64 * 64 * 64);
    ggml_backend_tensor_get(logits, L.data(), 0, L.size() * sizeof(float));
    auto mine = ssvae::logits_to_coords(L);
    auto gold = ssvae::load_golden_coords(std::string(GOLD) + "/stage1_out/coords.npy");
    int inter = 0; for (auto& c : mine) if (gold.count(c)) inter++;
    int uni = (int)mine.size() + (int)gold.size() - inter;
    double iou = (double)inter / uni;
    printf("  coords mine N=%zu  golden N=%zu  inter=%d  IoU=%.4f\n", mine.size(), gold.size(), inter, iou);
    bool ok = (mine.size() >= 1100 && mine.size() <= 1135 && iou > 0.97);
    printf("[sampler] %s (expect N~1120, IoU~0.986)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
