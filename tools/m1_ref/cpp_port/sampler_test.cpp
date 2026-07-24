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
    // Optional exact stage-boundary oracle.  This is diagnostic-only: native
    // production never reads Python artifacts, but a pinned Python capture is
    // the only valid way to measure whether a native precision experiment has
    // actually repaired the first occupancy divergence.
    const char* oracle_env = std::getenv("PIXAL3D_STAGE1_ORACLE_DIR");
    const std::string oracle = oracle_env ? oracle_env : GOLD;
    const bool has_oracle = oracle_env != nullptr;
    const char* raw_inputs_env = std::getenv("PIXAL3D_STAGE1_NATIVE_INPUT_DIR");
    printf("[sampler] reference=%s%s\n", oracle.c_str(), has_oracle ? " (stage oracle)" : " (legacy golden)");
    if (raw_inputs_env) printf("[sampler] inputs=%s (captured native F32 boundary)\n", raw_inputs_env);

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
    auto load_raw_f32 = [](const std::string& path, size_t n) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        std::vector<float> out(n);
        f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n * sizeof(float)));
        if ((size_t)f.gcount() != n * sizeof(float)) throw std::runtime_error("short read " + path);
        return out;
    };
    std::vector<float> cond_g, cond_p;
    if (raw_inputs_env) {
        const std::string raw = raw_inputs_env;
        cond_g = load_raw_f32(raw + "/native_stage1_global_f32.bin", (size_t)5 * 1024);
        cond_p = load_raw_f32(raw + "/native_stage1_proj_f32.bin", (size_t)ssdit::SEQ * 1024);
    } else {
        NpyArray gN = npy_load(oracle + "/stage1_cond/global.npy");
        NpyArray pN = npy_load(oracle + "/stage1_cond/proj.npy");
        cond_g.assign(gN.f32(), gN.f32() + gN.numel());
        cond_p.assign(pN.f32(), pN.f32() + pN.numel());
    }
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
    std::vector<float> x;
    if (raw_inputs_env) x = load_raw_f32(std::string(raw_inputs_env) + "/native_stage1_noise_f32.bin", NEL);
    else {
        NpyArray noiseN = npy_load(oracle + "/stage1_noise/noise.npy");
        x.assign(noiseN.f32(), noiseN.f32() + NEL);
    }
    // Match geo::flow_sampler and Python's np.linspace/Möbius construction:
    // t pairs are formed in double, then individual tensor operations receive
    // their F32 scalar.  A float-built sequence is enough to move a handful
    // of occupancy-threshold voxels and is not a valid production comparison.
    std::vector<double> tseq(STEPS + 1);
    for (int i = 0; i <= STEPS; i++) {
        double lt = 1.0 - (double)i / STEPS;          // np.linspace(1,0,13)
        tseq[i] = RT * lt / (1 + (RT - 1) * lt);     // Mobius warp
    }
    for (int s = 0; s < STEPS; s++) {
        double t = tseq[s], tp = tseq[s + 1];
        std::vector<float> v;
        printf("  step %2d/%d t=%.4f %s\n", s + 1, STEPS, t, (t >= IV0 && t <= IV1) ? "[cfg]" : "[cond]");
        if (t >= IV0 && t <= IV1) {
            std::vector<float> vp = forward(x, (float)(1000.0 * t), true);
            std::vector<float> vn = forward(x, (float)(1000.0 * t), false);
            stat("vp", vp); stat("vn", vn);
            std::vector<float> pred(NEL);
            for (int i = 0; i < NEL; i++) pred[i] = GS * vp[i] + (1 - GS) * vn[i];
            // std-rescale
            std::vector<float> x0p = pred_to_x0(x, (float)t, vp), x0c = pred_to_x0(x, (float)t, pred);
            double sp = vstd(x0p), sc = vstd(x0c), r = sp / sc;
            printf("    std_pos=%.4g std_cfg=%.4g r=%.4g\n", sp, sc, r);
            std::vector<float> x0(NEL);
            for (int i = 0; i < NEL; i++) { float resc = x0c[i] * (float)r; x0[i] = GR * resc + (1 - GR) * x0c[i]; }
            v = x0_to_pred(x, (float)t, x0);
        } else {
            v = forward(x, (float)(1000.0 * t), true);
            stat("v", v);
        }
        for (int i = 0; i < NEL; i++) x[i] -= (float)(t - tp) * v[i];
        stat("x", x);
        fflush(stdout);
    }
    std::vector<float>& z_s = x;  // [NEL] ch-major == VAE z input layout

    // ===== compare z_s =====
    printf("[sampler] backend=%s\n", use_cuda ? "cuda" : "cpu");
    {
        NpyArray gold = npy_load(oracle + "/stage1_ssdec/z_s.npy");
        double ma = 0, sum = 0; const float* g = gold.f32();
        for (int i = 0; i < NEL; i++) { double d = std::fabs(z_s[i] - g[i]); ma = std::max(ma, d); sum += d; }
        printf("  z_s vs %s: maxabs=%.3e meanabs=%.3e%s\n",
               has_oracle ? "stage reference" : "golden(bf16)", ma, sum / NEL,
               has_oracle ? "" : " (expect ~1.2 / ~3.6e-3)");
    }
    {
        std::string tp = std::string(REFS) + "/torch_z_s_fp32.npy";
        std::ifstream f(tp);
        if (!has_oracle && f.good()) {
            NpyArray tz = npy_load(tp);
            double ma = 0, sum = 0; const float* g = tz.f32();
            for (int i = 0; i < NEL; i++) { double d = std::fabs(z_s[i] - g[i]); ma = std::max(ma, d); sum += d; }
            printf("  z_s vs torch_fp32: maxabs=%.3e meanabs=%.3e (tight fp32 cross-check)\n", ma, sum / NEL);
        } else if (!has_oracle) {
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
    auto gold = ssvae::load_golden_coords(oracle + "/stage1_out/coords.npy");
    int inter = 0; for (auto& c : mine) if (gold.count(c)) inter++;
    int uni = (int)mine.size() + (int)gold.size() - inter;
    double iou = (double)inter / uni;
    printf("  coords mine N=%zu  golden N=%zu  inter=%d  IoU=%.4f\n", mine.size(), gold.size(), inter, iou);
    const bool ok = has_oracle
        ? (mine.size() == gold.size() && inter == (int)gold.size())
        : (mine.size() >= 1100 && mine.size() <= 1135 && iou > 0.97);
    printf("[sampler] %s %s\n", ok ? "PASS" : "FAIL",
           has_oracle ? "(exact stage oracle requires coordinate-set equality)"
                      : "(expect N~1120, IoU~0.986)");
    return ok ? 0 : 1;
}
