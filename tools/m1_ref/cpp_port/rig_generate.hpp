// rig_generate.hpp — the REAL autoregressive sampling loop for the TokenRig auto-rigger ("R3").
//
// Builds the AR decode ON TOP of qwen3_forward.hpp (the VALIDATED teacher-forced Qwen3-0.6B core)
// and rig_grammar.hpp (the VALIDATED two-phase constrained-decoding grammar). It does NOT re-derive
// either; it drives them.
//
// Reference: tokenrig.py TokenRig.generate() + VocabSwitchingLogitsProcessor.
//
// Dataflow (greedy / argmax — see note below):
//   inputs_embeds = [ mesh_cond (n_cond tokens, [hidden,n_cond]) ]              (R1 output_proj)
//                ++ [ embed(bos), embed(cls) ]                                  (start_tokens)
//   loop:
//       logits[:,S-1] = qwen3_forward(inputs_embeds[:, :S])      // [vocab] at the LAST position
//       allowed       = rig::allowed_next_tokens(generated)      // generated = the grammar sequence
//       mask logits to `allowed` (everything else -> -inf)
//       next          = argmax(masked logits)
//       generated.push_back(next); inputs_embeds += embed(next)
//       if next == model_eos (33035) break
//   return generated   (starts at bos, ends at model_eos)
//
// GRAMMAR SEQUENCE vs PROMPT: the mesh_cond tokens occupy LLM positions but are NOT part of the
// grammar `sequence`. The grammar sequence == start_tokens (bos, cls) ++ generated ids only. This
// mirrors VocabSwitchingLogitsProcessor, whose `init` = start_tokens is prepended each step.
//
// GREEDY vs BEAM: python uses HF .generate (num_beams=10, nondeterministic ordering). The rig is
// validated by GEOMETRY (joints/skin), not by exact token match, so the C++ MVP uses GREEDY argmax
// over the grammar-masked logits — the deterministic, reproducible path (HANDOFF-RIGGING).
//
// PERF (TODO): this MVP RE-RUNS THE FULL qwen3 forward every step (no KV cache). With n_cond=512 +
// up to ~567 generated => S up to ~1080, ~567 steps. That is O(S^2) graph builds — fine for a
// correctness-first MVP on the validated core, but the obvious perf follow-up is an incremental
// KV-cache decode (cache K/V per layer, feed only the new token's embed each step). qwen3_forward.hpp
// currently builds a full causal graph from scratch; KV-cache decode is a separate lap.
//
// WEIGHTS: pulled from the M1Harness (ctx_w-resident, allocated ONCE here on the first step). The
// caller registers the qwen3 weights (the harness `weight()` cache is populated by build_qwen3) and
// passes the host embed_tokens table for the per-token host gather.
#pragma once
#include "qwen3_forward.hpp"
#include "rig_grammar.hpp"
#include <vector>
#include <limits>
#include <cstddef>
#include <cmath>
#include <cstdio>

namespace rig {

// Apply the grammar mask in place: set every logit NOT in `allowed` to -inf.
// `allowed` ids index directly into model-vocab logits ([0, model_vocab)).
inline void apply_grammar_mask(std::vector<float>& logits, const std::vector<int>& allowed) {
    std::vector<char> keep(logits.size(), 0);
    for (int id : allowed) {
        if (id >= 0 && id < (int)logits.size()) keep[id] = 1;
    }
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!keep[i]) logits[i] = neg_inf;
    }
}

inline int argmax(const std::vector<float>& logits) {
    int best = 0;
    float bv = logits.empty() ? 0.f : logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
    }
    return best;
}

// ----------------------------------------------------------------------------
// Generic, GPU-free driver (kept for unit testing the grammar/argmax path against
// a canned `forward_step` — see rig_generate_test if/when added). `forward_step(seq)`
// returns the model-vocab logits for the NEXT token given the current grammar sequence.
// ----------------------------------------------------------------------------
template <typename ForwardFn>
inline std::vector<int> generate(const std::vector<int>& start_tokens,
                                 ForwardFn&& forward_step,
                                 const GrammarSpec& g = GrammarSpec(),
                                 int max_new_tokens = 4096) {
    std::vector<int> sequence = start_tokens; // == python `init`
    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<float> logits = forward_step(sequence);
        std::vector<int> allowed = allowed_next_tokens(sequence, g);
        apply_grammar_mask(logits, allowed);
        int next_id = argmax(logits);
        sequence.push_back(next_id);
        if (next_id == g.model_eos) break;
    }
    return sequence;
}

// ----------------------------------------------------------------------------
// THE REAL GPU AR LOOP.
//
//   H          : M1Harness with the qwen3 weights registered (npy or gguf). Weights are allocated
//                + uploaded ONCE here (first step). Must be a FRESH harness used only for this loop.
//   cfg        : Qwen3Cfg (hidden 896, vocab 33036, ...).
//   embed_table: host row-major [vocab, hidden] embed_tokens.weight (for the host token-embed gather).
//   mesh_cond  : host row-major [n_cond, hidden] R1 mesh-condition (output_proj(mesh_encoder(...))).
//   start_tokens: the grammar init, == {bos, cls} (model-vocab ids).
//   spec       : grammar spec.
//   max_new_tokens: hard cap on generated length (incl. the trailing model_eos).
//
// Returns the GENERATED grammar sequence (== start_tokens ++ generated body, ending at model_eos),
// i.e. directly comparable to the python output_ids (which prepends start_tokens to results).
// ----------------------------------------------------------------------------
inline std::vector<int> rig_generate(M1Harness& H, const Qwen3Cfg& cfg,
                                     const float* embed_table, /* [vocab, hidden] */
                                     const float* mesh_cond, int n_cond, /* [n_cond, hidden] */
                                     const std::vector<int>& start_tokens,
                                     const GrammarSpec& spec = GrammarSpec(),
                                     int max_new_tokens = 1024,
                                     bool verbose = true) {
    const int hidden = cfg.hidden;
    const int hd = cfg.head_dim, half = hd / 2;

    // grammar sequence (model-vocab ids): starts as a COPY of start_tokens (the python `init`).
    std::vector<int> sequence = start_tokens;

    // inputs_embeds, host, row-major [S, hidden] (S = n_cond + |sequence|). We append rows in place.
    // Row order: [mesh_cond rows ...][embed(start_tokens) rows ...][embed(generated) rows ...].
    std::vector<float> embeds;
    embeds.reserve((size_t)(n_cond + max_new_tokens + (int)start_tokens.size()) * hidden);
    embeds.insert(embeds.end(), mesh_cond, mesh_cond + (size_t)n_cond * hidden);
    auto append_embed = [&](int tok) {
        const float* row = embed_table + (size_t)tok * hidden;
        embeds.insert(embeds.end(), row, row + hidden);
    };
    for (int t : start_tokens) append_embed(t);

    // The harness allocates weights once (ctx_w). build_qwen3 populates the weight cache the first
    // time; we alloc+upload them after that first build. Per step we use a FRESH compute ctx +
    // gallocr (the MVP full-recompute path).
    bool weights_ready = false;

    for (int step = 0; step < max_new_tokens; ++step) {
        const int64_t S = (int64_t)embeds.size() / hidden;   // current sequence length (incl. cond)

        // --- per-step rope cos/sin [hd, S] (HF: emb = cat(freqs,freqs); freqs = pos*inv_freq) ---
        std::vector<float> cosb((size_t)hd * S), sinb((size_t)hd * S);
        std::vector<double> invf(half);
        for (int j = 0; j < half; j++) invf[j] = std::pow((double)cfg.rope_theta, -(double)(2 * j) / hd);
        for (int64_t p = 0; p < S; p++)
            for (int i = 0; i < hd; i++) {
                double ang = (double)p * invf[i < half ? i : i - half];
                cosb[p * hd + i] = (float)std::cos(ang);
                sinb[p * hd + i] = (float)std::sin(ang);
            }
        // --- causal mask [S(k), S(q)] : 0 if k<=q else -inf ---
        std::vector<float> maskb((size_t)S * S);
        for (int64_t q = 0; q < S; q++)
            for (int64_t k = 0; k < S; k++)
                maskb[q * S + k] = (k <= q) ? 0.0f : -INFINITY;

        // --- fresh compute context for this step's graph (weights stay in ctx_w) ---
        ggml_init_params ip{ (size_t)512 * 1024 * 1024, nullptr, true /*no_alloc*/ };
        ggml_context* cctx = ggml_init(ip);

        int64_t ine[4] = { hidden, S, 1, 1 };
        ggml_tensor* inp = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, ine);
        ggml_set_name(inp, "inputs_embeds"); ggml_set_input(inp);
        int64_t cne[4] = { hd, S, 1, 1 };
        ggml_tensor* cosT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(cosT);
        ggml_tensor* sinT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(sinT);
        int64_t mne[4] = { S, S, 1, 1 };
        ggml_tensor* maskT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, mne); ggml_set_input(maskT);

        ggml_tensor* cos3 = ggml_reshape_3d(cctx, cosT, hd, 1, S);
        ggml_tensor* sin3 = ggml_reshape_3d(cctx, sinT, hd, 1, S);

        ggml_tensor* logits = build_qwen3(H, cctx, cfg, inp, cos3, sin3, maskT);  // [vocab, S]
        ggml_set_output(logits);

        ggml_cgraph* gf = new_graph(cctx, 32768);
        ggml_build_forward_expand(gf, logits);

        // Allocate + upload the qwen3 weights ONCE (after the first graph references them all).
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

        // per-step graph allocation (intermediates only; weights already resident).
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(H.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(cctx); throw std::runtime_error("rig_generate: gallocr_alloc_graph failed"); }

        ggml_backend_tensor_set(inp,   embeds.data(), 0, embeds.size() * sizeof(float));
        ggml_backend_tensor_set(cosT,  cosb.data(),   0, cosb.size()   * sizeof(float));
        ggml_backend_tensor_set(sinT,  sinb.data(),   0, sinb.size()   * sizeof(float));
        ggml_backend_tensor_set(maskT, maskb.data(),  0, maskb.size()  * sizeof(float));

        ggml_backend_graph_compute(H.backend, gf);

        // read logits at the LAST position (column S-1): [vocab]
        std::vector<float> last(cfg.vocab);
        const size_t col_bytes = (size_t)cfg.vocab * sizeof(float);
        ggml_backend_tensor_get(logits, last.data(), (size_t)(S - 1) * col_bytes, col_bytes);

        ggml_gallocr_free(ga);
        ggml_free(cctx);

        // mask to grammar-allowed, argmax, append.
        std::vector<int> allowed = allowed_next_tokens(sequence, spec);
        apply_grammar_mask(last, allowed);
        int next_id = argmax(last);
        sequence.push_back(next_id);
        append_embed(next_id);

        if (verbose && (step < 4 || next_id == spec.model_eos || step % 64 == 0))
            printf("[rig_generate] step %d: S=%lld next=%d (allowed=%zu)\n",
                   step, (long long)S, next_id, allowed.size());

        if (next_id == spec.model_eos) break;
    }
    return sequence;
}

} // namespace rig
