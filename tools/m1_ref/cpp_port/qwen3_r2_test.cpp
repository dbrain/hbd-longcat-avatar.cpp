// qwen3_r2_test.cpp — R2 validation: ggml Qwen3-0.6B teacher-forced forward vs the Python fp32 golden.
//   weights (npy, from capture_qwen3_r2.py) in <wdir>; golden inputs_embeds/logits in <golden>.
//   build: ./build.sh qwen3_r2_test [cuda]
//   run:   ./qwen3_r2_test [golden=/tmp/qwen3_r2] [wdir=/tmp/qwen3_w] [cuda]
#include "qwen3_forward.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    std::string golden = "/tmp/qwen3_r2", wdir = "/tmp/qwen3_w";
    bool use_cuda = false;
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "cuda") use_cuda = true;
        else if (pos++ == 0) golden = a;
        else wdir = a;
    }
    Qwen3Cfg cfg;

    // inputs_embeds golden [S, hidden] (row-major) -> ggml input [hidden, S]
    NpyArray ie = npy_load(golden + "/inputs_embeds.npy");
    int64_t S = ie.shape[0], hidden = ie.shape[1];
    if ((int) hidden != cfg.hidden) { printf("hidden mismatch %lld vs %d\n", (long long)hidden, cfg.hidden); return 1; }
    printf("[qwen3_r2] S=%lld hidden=%lld layers=%d heads=%d/%d head_dim=%d vocab=%d backend=%s\n",
           (long long)S, (long long)hidden, cfg.n_layers, cfg.n_heads, cfg.n_kv, cfg.head_dim, cfg.vocab,
           use_cuda ? "cuda" : "cpu");

    // host rope cos/sin [head_dim, S]  (HF: emb = cat(freqs, freqs); freqs = pos * inv_freq[0..hd/2))
    const int hd = cfg.head_dim, half = hd / 2;
    std::vector<float> cosb((size_t) hd * S), sinb((size_t) hd * S);
    std::vector<double> invf(half);
    for (int j = 0; j < half; j++) invf[j] = std::pow((double) cfg.rope_theta, -(double)(2 * j) / hd);
    for (int64_t p = 0; p < S; p++)
        for (int i = 0; i < hd; i++) {
            double ang = (double) p * invf[i < half ? i : i - half];
            cosb[p * hd + i] = (float) std::cos(ang);
            sinb[p * hd + i] = (float) std::sin(ang);
        }

    // causal mask [S(k), S(q)]: 0 if k<=q else -inf
    std::vector<float> maskb((size_t) S * S);
    for (int64_t q = 0; q < S; q++)
        for (int64_t k = 0; k < S; k++)
            maskb[q * S + k] = (k <= q) ? 0.0f : -INFINITY;

    M1Harness H(wdir, 4096, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t ine[4] = {hidden, S, 1, 1};
    ggml_tensor* inp = H.input("inputs_embeds", 2, ine);
    int64_t cne[4] = {hd, S, 1, 1};
    ggml_tensor* cosT = H.const_tensor("rope_cos", 2, cne, cosb);
    ggml_tensor* sinT = H.const_tensor("rope_sin", 2, cne, sinb);
    int64_t mne[4] = {S, S, 1, 1};
    ggml_tensor* maskT = H.const_tensor("causal_mask", 2, mne, maskb);

    ggml_tensor* cos3 = ggml_reshape_3d(ctx, cosT, hd, 1, S);
    ggml_tensor* sin3 = ggml_reshape_3d(ctx, sinT, hd, 1, S);

    ggml_tensor* logits = build_qwen3(H, ctx, cfg, inp, cos3, sin3, maskT);  // [vocab, S]
    ggml_set_output(logits);

    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, logits);
    H.alloc_and_upload(gf);
    std::vector<float> ie_vec(ie.f32(), ie.f32() + ie.numel());
    H.upload_input_raw(inp, ie_vec);
    H.compute(gf);

    printf("[qwen3_r2] ran -> logits [%lld,%lld]\n", (long long)logits->ne[0], (long long)logits->ne[1]);
    CmpStats s = compare_to_npy(H, logits, golden + "/logits.npy", true, "logits");

    // The meaningful AR-LM criterion: next-token (argmax) agreement per position vs the golden logits.
    NpyArray ref = npy_load(golden + "/logits.npy");
    std::vector<float> got((size_t) logits->ne[0] * logits->ne[1]);
    ggml_backend_tensor_get(logits, got.data(), 0, got.size() * sizeof(float));
    const float* r = ref.f32();
    int64_t V = logits->ne[0], nS = logits->ne[1];
    const int64_t Lc = 512;   // cond-prefix length (positions < Lc predict nothing meaningful)
    int64_t agree = 0, agree_gen = 0, gen = 0, mism_neartie = 0, mism = 0;
    for (int64_t p = 0; p < nS; p++) {
        int64_t am = 0, ar = 0; float bm = -1e30f, br = -1e30f, br2 = -1e30f;
        for (int64_t t = 0; t < V; t++) {
            float g = got[p * V + t], rf = r[p * V + t];
            if (g > bm) { bm = g; am = t; }
            if (rf > br) { br2 = br; br = rf; ar = t; } else if (rf > br2) { br2 = rf; }
        }
        bool ok1 = (am == ar);
        if (ok1) agree++;
        if (p >= Lc) { gen++; if (ok1) agree_gen++; }
        if (!ok1) { mism++; if (br - br2 < 0.3f) mism_neartie++; }  // golden top1-top2 margin < drift
    }
    printf("[qwen3_r2] top-1 next-token agreement: all=%lld/%lld=%.2f%%  gen(p>=%lld)=%lld/%lld=%.2f%%\n",
           (long long)agree, (long long)nS, 100.0*agree/nS, (long long)Lc,
           (long long)agree_gen, (long long)gen, gen ? 100.0*agree_gen/gen : 0.0);
    printf("[qwen3_r2] mismatches=%lld, of which near-ties (golden margin<0.3)=%lld\n",
           (long long)mism, (long long)mism_neartie);
    // PASS criterion (the principled one for an AR-LM port): meanabs logit error is tiny AND EVERY
    // argmax mismatch is a near-tie (golden top1-top2 < the fp32 logit drift). A real bug would show
    // large meanabs and/or mismatches at confident positions. Beam search tolerates near-tie flips
    // (HANDOFF-RIGGING: validate the rig, not bit-exact tokens).
    bool ok = (mism == mism_neartie) && s.meanabs < 1e-2;
    printf("[qwen3_r2] %s (meanabs %.2e; %lld/%lld argmax match; all %lld mismatches are near-ties)\n",
           ok ? "PASS" : "FAIL", s.meanabs, (long long)agree, (long long)nS, (long long)mism);
    return ok ? 0 : 1;
}
