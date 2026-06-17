// rig_sample.hpp — HF-style SAMPLING decode for the TokenRig auto-rigger ("R3"), built ON TOP of the
// VALIDATED KV-cache decode (qwen3_decode.hpp) + the VALIDATED grammar (rig_grammar.hpp).
//
// WHY SAMPLING (not greedy, not beam): the model's golden output was produced by HF .generate() with
//   max_length=2048, top_k=10, top_p=0.95, temperature=1.5, repetition_penalty=2.0, num_beams=10,
//   do_sample=True (see capture_skintokens_e2e.py). Greedy argmax (rig_generate) and plain beam
//   (rig_beam_generate) omit the warpers and fall into a degenerate loop (coordinate ~128 / branch
//   forever, never choosing the skeleton-end token) so generation never terminates. The LOOP-BREAKER
//   is repetition_penalty=2.0; temperature+top_k+top_p+sampling add the diversity needed to escape.
//
// This is a SINGLE-SEQUENCE KV-cache sampling decode: ONE Qwen3KvCache, NO beam, NO cache-fork — much
// simpler than rig_beam_generate (no per-beam caches, no fork host-bounce, no snapshot/restore). It
// REUSES that file's proven host plumbing pattern (beam_rope_tables + the run_chunk lambda) verbatim,
// driving the SAME build_qwen3_cached prefill/step graph.
//
// PROCESSING ORDER (matches HF generation: logits_processors[rep_penalty, grammar] THEN
//   logits_warpers[temperature, top_k, top_p] THEN multinomial):
//     (a) repetition_penalty (HF/CTRL formula) over tokens already in the grammar sequence
//     (b) grammar mask (disallowed -> -inf)
//     (c) temperature divide (finite logits only)
//     (d) top_k  (keep k largest finite, rest -inf)
//     (e) top_p  (nucleus: smallest prefix whose cumprob >= top_p, always keep >=1)
//     (f) softmax survivors -> multinomial draw (mt19937_64)
//   All softmax / cumulative sums are kept fp32/double.
#pragma once
#include "qwen3_decode.hpp"
#include "rig_grammar.hpp"
#include "rig_generate.hpp"   // reuse rig::apply_grammar_mask
#include "m1_ggml.hpp"
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <stdexcept>

namespace rig {

// ---- RoPE cos/sin [hd, L] for absolute positions [p0, p0+L) — identical to beam_rope_tables. ----
// (kept local so this header is usable standalone; bit-identical math to rig_beam_generate.)
static inline void sample_rope_tables(const Qwen3Cfg& cfg, int64_t p0, int64_t L,
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

// ----------------------------------------------------------------------------
// THE GPU SINGLE-SEQUENCE SAMPLING LOOP.
//
//   H            : M1Harness with the qwen3 weights registered. Weights are allocated + uploaded ONCE
//                  here (first graph). Must be a FRESH harness used only for this loop.
//   cfg          : Qwen3Cfg.
//   embed_table  : host row-major [vocab, hidden] embed_tokens.weight (host token-embed gather).
//   mesh_cond    : host row-major [n_cond, hidden] R1 mesh-condition.
//   start_tokens : grammar init, == {bos, cls} (model-vocab ids).
//   spec         : grammar spec (model_eos = 33035).
//   repetition_penalty / temperature / top_k / top_p : HF warper params (golden defaults).
//   max_new_tokens : hard cap on generated tokens.
//   seed         : mt19937_64 seed.
//
// Returns the grammar sequence (start_tokens ++ generated), ending at model_eos if it terminated.
// ----------------------------------------------------------------------------
inline std::vector<int> rig_generate_sample(M1Harness& H, const Qwen3Cfg& cfg,
                                            const float* embed_table, /* [vocab, hidden] */
                                            const float* mesh_cond, int n_cond, /* [n_cond, hidden] */
                                            const std::vector<int>& start_tokens,
                                            const GrammarSpec& spec = GrammarSpec(),
                                            float repetition_penalty = 2.0f,
                                            float temperature = 1.5f,
                                            int top_k = 10,
                                            float top_p = 0.95f,
                                            int max_new_tokens = 800,
                                            uint64_t seed = 0,
                                            bool verbose = true) {
    const int hidden = cfg.hidden, hd = cfg.head_dim, V = cfg.vocab;
    const int P = n_cond + (int)start_tokens.size();   // prefix length (cond rows + start tokens)
    const int max_seq = n_cond + max_new_tokens + (int)start_tokens.size() + 8;

    // ---- embed gather helper (host row from embed_table). ----
    auto embed_row = [&](int tok) -> const float* { return embed_table + (size_t)tok * hidden; };

    // ---- prefix embeds [hidden, P] (row-major): mesh_cond rows ++ embed(start_tokens) rows. ----
    std::vector<float> prefix_embeds;
    prefix_embeds.reserve((size_t)P * hidden);
    prefix_embeds.insert(prefix_embeds.end(), mesh_cond, mesh_cond + (size_t)n_cond * hidden);
    for (int t : start_tokens) {
        const float* r = embed_row(t);
        prefix_embeds.insert(prefix_embeds.end(), r, r + hidden);
    }

    // ---- ONE cache (single sequence, NO fork). ----
    Qwen3KvCache cache;
    cache.init(H, cfg, max_seq);

    bool weights_ready = false;

    // ---- shared chunk runner: build a cache-backed graph for `cache`, decode L tokens at write0 from
    //      host embeds `chunk_embeds` [hidden, L], return the chunk's logits [V, L] (host). Mirrors the
    //      run_chunk lambda in rig_beam_generate EXACTLY (single cache, no fork). ----
    auto run_chunk = [&](Qwen3KvCache& c, int write0, int L,
                         const float* chunk_embeds, bool causal_mask) -> std::vector<float> {
        std::vector<float> cosb, sinb;
        sample_rope_tables(cfg, write0, L, cosb, sinb);

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
        ggml_tensor* logits = build_qwen3_cached(H, cctx, cfg, c, inp, cos3, sin3,
                                                 write0, L, maskT, writes);   // [vocab, L]
        ggml_set_output(logits);

        ggml_cgraph* gf = new_graph(cctx, 32768);
        ggml_build_forward_expand(gf, logits);
        for (ggml_tensor* w : writes) ggml_build_forward_expand(gf, w);   // ensure cache writes run

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
            throw std::runtime_error("rig_generate_sample: gallocr_alloc_graph failed");
        }

        ggml_backend_tensor_set(inp,  chunk_embeds, 0, (size_t)L * hidden * sizeof(float));
        ggml_backend_tensor_set(cosT, cosb.data(),  0, cosb.size() * sizeof(float));
        ggml_backend_tensor_set(sinT, sinb.data(),  0, sinb.size() * sizeof(float));
        if (causal_mask) ggml_backend_tensor_set(maskT, maskb.data(), 0, maskb.size() * sizeof(float));

        ggml_backend_graph_compute(H.backend, gf);

        std::vector<float> out((size_t)L * V);
        ggml_backend_tensor_get(logits, out.data(), 0, (size_t)L * V * sizeof(float));

        ggml_gallocr_free(ga);
        ggml_free(cctx);
        c.pos = write0 + L;
        return out;
    };

    // ============================ PREFILL ============================
    cache.reset();
    std::vector<float> prefix_logits_all = run_chunk(cache, 0, P, prefix_embeds.data(), /*causal=*/true);
    // last-position logits (column P-1) of the prefill seed the first sampling step.
    std::vector<float> logits(prefix_logits_all.begin() + (size_t)(P - 1) * V,
                              prefix_logits_all.begin() + (size_t)P * V);
    if (verbose)
        printf("[rig_sample] prefill P=%d (n_cond=%d + |start|=%zu), max_seq=%d, "
               "reppen=%.2f temp=%.2f top_k=%d top_p=%.2f seed=%llu max_new=%d\n",
               P, n_cond, start_tokens.size(), max_seq,
               repetition_penalty, temperature, top_k, top_p,
               (unsigned long long)seed, max_new_tokens);

    // grammar sequence (model-vocab ids); starts as a COPY of start_tokens (python `init`).
    std::vector<int> sequence = start_tokens;
    std::mt19937_64 rng(seed);

    bool terminated = false;
    int generated = 0;

    for (int step = 0; step < max_new_tokens; ++step) {
        // working copy of the current-position logits over the FULL model vocab.
        std::vector<float> lg = logits;

        // (a) REPETITION PENALTY (HF/CTRL): for each UNIQUE token already in `sequence`,
        //     logit = logit > 0 ? logit/penalty : logit*penalty. (apply once per unique id.)
        if (repetition_penalty != 1.0f) {
            std::unordered_set<int> seen(sequence.begin(), sequence.end());
            for (int t : seen) {
                if (t < 0 || t >= V) continue;
                float& l = lg[t];
                l = (l > 0.0f) ? (l / repetition_penalty) : (l * repetition_penalty);
            }
        }

        // (b) GRAMMAR MASK: disallowed -> -inf.
        std::vector<int> allowed = allowed_next_tokens(sequence, spec);
        apply_grammar_mask(lg, allowed);

        // (c) TEMPERATURE: divide finite logits.
        if (temperature != 1.0f && temperature > 0.0f) {
            for (float& v : lg) if (std::isfinite(v)) v /= temperature;
        }

        // (d) TOP_K: keep the k largest FINITE logits, rest -inf. (If fewer than k allowed, keep all.)
        if (top_k > 0) {
            std::vector<std::pair<float,int>> fin;
            fin.reserve(allowed.size());
            for (size_t i = 0; i < lg.size(); ++i) if (std::isfinite(lg[i])) fin.push_back({ lg[i], (int)i });
            if ((int)fin.size() > top_k) {
                std::partial_sort(fin.begin(), fin.begin() + top_k, fin.end(),
                                  [](const auto& a, const auto& b){ return a.first > b.first; });
                std::vector<char> keep(lg.size(), 0);
                for (int i = 0; i < top_k; ++i) keep[fin[i].second] = 1;
                const float ninf = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < lg.size(); ++i) if (!keep[i]) lg[i] = ninf;
            }
        }

        // (e) TOP_P (nucleus): softmax survivors, sort desc, keep the smallest prefix whose cumulative
        //     prob >= top_p (always keep at least the top-1); rest -inf.
        if (top_p < 1.0f) {
            std::vector<std::pair<float,int>> fin;
            for (size_t i = 0; i < lg.size(); ++i) if (std::isfinite(lg[i])) fin.push_back({ lg[i], (int)i });
            if (fin.size() > 1) {
                std::sort(fin.begin(), fin.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
                double mx = fin[0].first;
                double sum = 0.0;
                for (auto& pr : fin) sum += std::exp((double)pr.first - mx);
                // cumulative; keep while cum < top_p, always keep index 0.
                std::vector<char> keep(lg.size(), 0);
                double cum = 0.0;
                for (size_t r = 0; r < fin.size(); ++r) {
                    double p = std::exp((double)fin[r].first - mx) / sum;
                    keep[fin[r].second] = 1;
                    cum += p;
                    if (cum >= (double)top_p) break;   // this entry is the last to keep
                }
                const float ninf = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < lg.size(); ++i) if (!keep[i]) lg[i] = ninf;
            }
        }

        // (f) SAMPLE: softmax survivors -> multinomial draw.
        int next_id = -1;
        {
            double mx = -std::numeric_limits<double>::infinity();
            for (float v : lg) if (std::isfinite(v) && (double)v > mx) mx = (double)v;
            if (!std::isfinite(mx)) throw std::runtime_error("rig_generate_sample: all logits masked");
            double sum = 0.0;
            for (float v : lg) if (std::isfinite(v)) sum += std::exp((double)v - mx);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            double u = uni(rng) * sum;   // draw in [0, sum)
            double acc = 0.0;
            for (size_t i = 0; i < lg.size(); ++i) {
                if (!std::isfinite(lg[i])) continue;
                acc += std::exp((double)lg[i] - mx);
                if (u < acc) { next_id = (int)i; break; }
            }
            if (next_id < 0)   // fp guard: fall to the last finite survivor
                for (size_t i = lg.size(); i-- > 0; ) if (std::isfinite(lg[i])) { next_id = (int)i; break; }
        }

        sequence.push_back(next_id);
        generated++;

        if (verbose && (step < 4 || next_id == spec.model_eos || step % 64 == 0))
            printf("[rig_sample] step %d: pos=%d next=%d (allowed=%zu)\n",
                   step, cache.pos, next_id, allowed.size());

        if (next_id == spec.model_eos) { terminated = true; break; }

        // decode the chosen token at the current position -> next-position logits.
        const float* erow = embed_row(next_id);
        std::vector<float> step_embed(erow, erow + hidden);
        std::vector<float> lg_next = run_chunk(cache, cache.pos, 1, step_embed.data(), /*causal=*/false);
        logits = std::move(lg_next);   // [V] at the new position
    }

    if (verbose) {
        int approx_bones = bones_in_sequence(sequence, spec);
        printf("[rig_sample] DONE: %s after %d generated tokens (total seq=%zu, last=%d). "
               "bones_in_sequence~=%d\n",
               terminated ? "TERMINATED (model_eos)" : "NO-TERMINATE (hit max_new_tokens)",
               generated, sequence.size(),
               sequence.empty() ? -1 : sequence.back(), approx_bones);
    }
    return sequence;
}

// ----------------------------------------------------------------------------
// TEACHER-FORCED forward (diagnostic): prefix = [mesh_cond ++ embed(seq_in)], ONE causal forward.
// Returns host logits [L*vocab] column-major (out[col*V + v]) for all L = n_cond + |seq_in| positions.
// Column c predicts the token AFTER input position c. Used by the e2e tfprobe mode to read the rank of
// the switch token (258) at the golden's switch context — isolates forward fidelity from search.
// ----------------------------------------------------------------------------
inline std::vector<float> rig_teacher_force_logits(M1Harness& H, const Qwen3Cfg& cfg,
                                                   const float* embed_table, const float* mesh_cond,
                                                   int n_cond, const std::vector<int>& seq_in) {
    const int hidden = cfg.hidden, hd = cfg.head_dim, V = cfg.vocab;
    const int L = n_cond + (int)seq_in.size();
    std::vector<float> prefix; prefix.reserve((size_t)L * hidden);
    prefix.insert(prefix.end(), mesh_cond, mesh_cond + (size_t)n_cond * hidden);
    for (int t : seq_in) { const float* r = embed_table + (size_t)t * hidden; prefix.insert(prefix.end(), r, r + hidden); }

    Qwen3KvCache cache; cache.init(H, cfg, L + 8); cache.reset();
    std::vector<float> cosb, sinb; sample_rope_tables(cfg, 0, L, cosb, sinb);
    std::vector<float> maskb((size_t)L * L, 0.f);
    for (int q = 0; q < L; q++) for (int k = 0; k < L; k++) maskb[(size_t)q * L + k] = (k <= q) ? 0.0f : -INFINITY;

    ggml_init_params ip{ (size_t)1024 * 1024 * 1024, nullptr, true };
    ggml_context* cctx = ggml_init(ip);
    int64_t ine[4] = { hidden, L, 1, 1 };
    ggml_tensor* inp = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, ine); ggml_set_input(inp);
    int64_t cne[4] = { hd, L, 1, 1 };
    ggml_tensor* cosT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(cosT);
    ggml_tensor* sinT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(sinT);
    ggml_tensor* cos3 = ggml_reshape_3d(cctx, cosT, hd, 1, L);
    ggml_tensor* sin3 = ggml_reshape_3d(cctx, sinT, hd, 1, L);
    int64_t mne[4] = { L, L, 1, 1 };
    ggml_tensor* maskT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, mne); ggml_set_input(maskT);
    std::vector<ggml_tensor*> writes;
    ggml_tensor* logits = build_qwen3_cached(H, cctx, cfg, cache, inp, cos3, sin3, 0, L, maskT, writes);
    ggml_set_output(logits);
    ggml_cgraph* gf = new_graph(cctx, 32768);
    ggml_build_forward_expand(gf, logits);

    H.wbuf = ggml_backend_alloc_ctx_tensors(H.ctx_w, H.backend);
    for (auto& gu : H.gguf_uploads) ggml_backend_tensor_set(gu.first, gu.second, 0, ggml_nbytes(gu.first));
    for (auto& u : H.uploads) { NpyArray a = npy_load(u.second); ggml_backend_tensor_set(u.first, a.raw.data(), 0, (size_t)a.numel() * 4); }

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(H.backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(cctx); throw std::runtime_error("tf: alloc failed"); }
    ggml_backend_tensor_set(inp,  prefix.data(), 0, (size_t)L * hidden * sizeof(float));
    ggml_backend_tensor_set(cosT, cosb.data(),  0, cosb.size() * sizeof(float));
    ggml_backend_tensor_set(sinT, sinb.data(),  0, sinb.size() * sizeof(float));
    ggml_backend_tensor_set(maskT, maskb.data(), 0, maskb.size() * sizeof(float));
    ggml_backend_graph_compute(H.backend, gf);
    std::vector<float> out((size_t)L * V);
    ggml_backend_tensor_get(logits, out.data(), 0, (size_t)L * V * sizeof(float));
    ggml_gallocr_free(ga); ggml_free(cctx);
    return out;   // out[col*V + v]
}

} // namespace rig
