// qwen3_decode_test.cpp — VALIDATE the KV-cache incremental decode (qwen3_decode.hpp) against the
//   teacher-forced fp32 golden (capture_qwen3_r2.py). PREFILL the first P positions, then DECODE_STEP
//   the remaining positions one at a time feeding the GOLDEN embeds (teacher-forced), collecting the
//   per-position logits, and compare to logits.npy at the decoded positions.
//
//   build: ./build.sh qwen3_decode_test cuda
//   run:   ./qwen3_decode_test [golden=/tmp/qwen3_r2] [wdir=/tmp/qwen3_w] [cuda] [prefill=514]
//
// DESIGN: the persistent KV cache (Qwen3KvCache) lives in its own backend buffer. Weights are
// allocated ONCE (ctx_w) on the first graph. Each chunk (the prefill, then each 1-token step) builds
// a FRESH compute ctx + graph + gallocr (intermediates reused; cache + weights persist). The cache
// is written in-graph via ggml_cpy(new_kv -> cache_view); we ggml_build_forward_expand over BOTH the
// logits output AND every cache-write node so the writes actually execute.
#include "qwen3_decode.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

// Build RoPE cos/sin [hd, L] for absolute positions [p0, p0+L).  HF: emb = cat(freqs,freqs),
// freqs[j] = pos * theta^(-2j/hd).  (Identical construction to qwen3_r2_test / rig_generate.)
static void rope_tables(const Qwen3Cfg& cfg, int64_t p0, int64_t L,
                        std::vector<float>& cosb, std::vector<float>& sinb) {
    const int hd = cfg.head_dim, half = hd / 2;
    cosb.assign((size_t)hd * L, 0.f);
    sinb.assign((size_t)hd * L, 0.f);
    std::vector<double> invf(half);
    for (int j = 0; j < half; j++) invf[j] = std::pow((double)cfg.rope_theta, -(double)(2 * j) / hd);
    for (int64_t t = 0; t < L; t++) {
        int64_t p = p0 + t;
        for (int i = 0; i < hd; i++) {
            double ang = (double)p * invf[i < half ? i : i - half];
            cosb[t * hd + i] = (float)std::cos(ang);
            sinb[t * hd + i] = (float)std::sin(ang);
        }
    }
}

int main(int argc, char** argv) {
    std::string golden = "/tmp/qwen3_r2", wdir = "/tmp/qwen3_w";
    bool use_cuda = false;
    int prefill = 514;
    int posarg = 0;
    ggml_type prec = GGML_TYPE_F32;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "cuda") use_cuda = true;
        else if (a.rfind("prefill=", 0) == 0) prefill = std::atoi(a.c_str() + 8);
        else if (a.rfind("prec=", 0) == 0) {
            std::string p = a.c_str() + 5;
            if (p == "f16") prec = GGML_TYPE_F16; else if (p == "bf16") prec = GGML_TYPE_BF16;
            else prec = GGML_TYPE_F32;
        }
        else if (posarg++ == 0) golden = a;
        else wdir = a;
    }
    Qwen3Cfg cfg;
    cfg.compute_type = prec;
    printf("[decode] precision=%s\n", prec==GGML_TYPE_F16?"f16":prec==GGML_TYPE_BF16?"bf16":"f32");
    const int hidden = cfg.hidden, hd = cfg.head_dim, V = cfg.vocab;

    // golden inputs_embeds [S, hidden] (row-major) + golden logits [S, vocab].
    NpyArray ie = npy_load(golden + "/inputs_embeds.npy");
    int64_t S = ie.shape[0];
    if ((int)ie.shape[1] != hidden) { printf("hidden mismatch\n"); return 1; }
    NpyArray ref = npy_load(golden + "/logits.npy");
    if (ref.shape[0] != S || (int)ref.shape[1] != V) { printf("logits shape mismatch\n"); return 1; }
    const float* ie_all = ie.f32();      // [S, hidden]
    const float* ref_l  = ref.f32();     // [S, vocab]

    if (prefill < 1 || prefill > S) prefill = (int)std::min<int64_t>(514, S);
    const int max_seq = (int)S + 64;     // >= S (cache headroom)
    printf("[decode] S=%lld hidden=%d layers=%d heads=%d/%d hd=%d vocab=%d prefill=%d max_seq=%d backend=%s\n",
           (long long)S, hidden, cfg.n_layers, cfg.n_heads, cfg.n_kv, hd, V, prefill, max_seq,
           use_cuda ? "cuda" : "cpu");

    M1Harness H(wdir, 4096, use_cuda);
    Qwen3KvCache cache;
    cache.init(H, cfg, max_seq);

    bool weights_ready = false;
    std::vector<float> got_logits((size_t)S * V, 0.f);   // collected incremental logits per position

    // ---- helper: build + run a chunk graph; copies the chunk's logits into got_logits[write0..]. ----
    auto run_chunk = [&](int write0, int L, bool causal_mask) {
        std::vector<float> cosb, sinb;
        rope_tables(cfg, write0, L, cosb, sinb);

        // additive mask [klen, L]: position q (col) sees key k (row) iff (write0+q) >= k.
        const int klen = write0 + L;
        std::vector<float> maskb;
        if (causal_mask) {
            maskb.assign((size_t)klen * L, 0.f);
            for (int q = 0; q < L; q++) {
                int qabs = write0 + q;
                for (int k = 0; k < klen; k++)
                    maskb[(size_t)q * klen + k] = (k <= qabs) ? 0.0f : -INFINITY;
            }
        }

        ggml_init_params ip{ (size_t)512 * 1024 * 1024, nullptr, true /*no_alloc*/ };
        ggml_context* cctx = ggml_init(ip);

        int64_t ine[4] = { hidden, L, 1, 1 };
        ggml_tensor* inp = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, ine);
        ggml_set_name(inp, "chunk_embeds"); ggml_set_input(inp);
        int64_t cne[4] = { hd, L, 1, 1 };
        ggml_tensor* cosT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(cosT);
        ggml_tensor* sinT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(sinT);
        ggml_tensor* cos3 = ggml_reshape_3d(cctx, cosT, hd, 1, L);
        ggml_tensor* sin3 = ggml_reshape_3d(cctx, sinT, hd, 1, L);

        ggml_tensor* maskT = nullptr;
        if (causal_mask) {
            int64_t mne[4] = { klen, L, 1, 1 };
            maskT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, mne); ggml_set_input(maskT);
        }

        std::vector<ggml_tensor*> writes;
        ggml_tensor* logits = build_qwen3_cached(H, cctx, cfg, cache, inp, cos3, sin3,
                                                 write0, L, maskT, writes);  // [vocab, L]
        ggml_set_output(logits);

        ggml_cgraph* gf = new_graph(cctx, 32768);
        ggml_build_forward_expand(gf, logits);
        for (ggml_tensor* w : writes) ggml_build_forward_expand(gf, w);  // ensure cache writes run

        if (!weights_ready) {
            H.wbuf = ggml_backend_alloc_ctx_tensors(H.ctx_w, H.backend);
            for (auto& gu : H.gguf_uploads)
                ggml_backend_tensor_set(gu.first, gu.second, 0, ggml_nbytes(gu.first));
            for (auto& u : H.uploads) {
                NpyArray a = npy_load(u.second);
                ggml_backend_tensor_set(u.first, a.raw.data(), 0, (size_t)a.numel() * 4);
            }
            weights_ready = true;
        }

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(H.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            ggml_gallocr_free(ga); ggml_free(cctx);
            throw std::runtime_error("decode: gallocr_alloc_graph failed");
        }

        // upload chunk inputs: rows [write0, write0+L) of the golden embeds.
        ggml_backend_tensor_set(inp, ie_all + (size_t)write0 * hidden, 0, (size_t)L * hidden * sizeof(float));
        ggml_backend_tensor_set(cosT, cosb.data(), 0, cosb.size() * sizeof(float));
        ggml_backend_tensor_set(sinT, sinb.data(), 0, sinb.size() * sizeof(float));
        if (causal_mask) ggml_backend_tensor_set(maskT, maskb.data(), 0, maskb.size() * sizeof(float));

        ggml_backend_graph_compute(H.backend, gf);

        // pull the chunk's L columns of logits -> got_logits[write0 .. write0+L)
        ggml_backend_tensor_get(logits, got_logits.data() + (size_t)write0 * V, 0,
                                (size_t)L * V * sizeof(float));

        ggml_gallocr_free(ga);
        ggml_free(cctx);
        cache.pos = write0 + L;
    };

    // ---- PREFILL the first `prefill` positions (causal), then DECODE the rest one token at a time. ----
    printf("[decode] prefill %d positions ..\n", prefill);
    run_chunk(0, prefill, /*causal_mask=*/true);
    printf("[decode] prefill done, cache.pos=%d\n", cache.pos);

    for (int p = prefill; p < (int)S; p++) {
        run_chunk(p, 1, /*causal_mask=*/false);   // single query: all past visible, no mask needed
        if (p == prefill || p == (int)S - 1 || (p - prefill) % 128 == 0)
            printf("[decode] step pos=%d cache.pos=%d\n", p, cache.pos);
    }

    // ---- compare incremental logits to the golden at the DECODED positions [prefill, S). ----
    // (positions [0, prefill) were prefilled; we validate the decoded region — the whole point.)
    auto report = [&](int lo, int hi, const char* tag) {
        double maxabs = 0, sum = 0; int64_t n = 0, argmatch = 0, total = 0, mism = 0, neartie = 0;
        for (int p = lo; p < hi; p++) {
            int gm = 0, rm = 0; float gb = -1e30f, rb = -1e30f, rb2 = -1e30f;
            for (int t = 0; t < V; t++) {
                float g = got_logits[(size_t)p * V + t], rf = ref_l[(size_t)p * V + t];
                double d = std::fabs((double)g - (double)rf);
                if (d > maxabs) maxabs = d;
                sum += d; n++;
                if (g > gb) { gb = g; gm = t; }
                if (rf > rb) { rb2 = rb; rb = rf; rm = t; } else if (rf > rb2) rb2 = rf;
            }
            total++;
            if (gm == rm) argmatch++; else { mism++; if (rb - rb2 < 0.3f) neartie++; }
        }
        double meanabs = n ? sum / n : 0.0;
        printf("[decode] %s: positions=%lld  maxabs=%.3e meanabs=%.3e  argmax=%lld/%lld=%.2f%%  "
               "mismatch=%lld (near-tie=%lld)\n", tag, (long long)total, maxabs, meanabs,
               (long long)argmatch, (long long)total, total ? 100.0 * argmatch / total : 0.0,
               (long long)mism, (long long)neartie);
        return std::make_pair(meanabs, mism == neartie);
    };

    // also report prefill region for sanity (should match the full-forward exactly there too).
    report(0, prefill, "prefill-region");
    auto [meanabs, all_neartie] = report(prefill, (int)S, "decoded-region");

    bool ok = (meanabs < 1e-2) && all_neartie;
    printf("[decode] %s (decoded meanabs %.2e; all argmax mismatches are near-ties=%s)\n",
           ok ? "PASS" : "FAIL", meanabs, all_neartie ? "yes" : "no");
    return ok ? 0 : 1;
}
