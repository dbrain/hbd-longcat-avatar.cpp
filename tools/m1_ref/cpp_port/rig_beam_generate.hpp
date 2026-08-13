// rig_beam_generate.hpp — HF-style BEAM SEARCH for the TokenRig auto-rigger ("R3"), built ON TOP of
// the VALIDATED KV-cache decode (qwen3_decode.hpp) + the VALIDATED grammar (rig_grammar.hpp).
//
// WHY BEAM: greedy argmax (rig_generate) under-generates (47 joints vs the golden 60). tokenrig.py's
// HF .generate() uses num_beams=10; that is the proven config. This is a faithful C++ port of HF
// beam_search over the grammar-masked, log_softmax'd logits, driving the cached decode.
//
// CONTRAST WITH rig_generate (greedy): rig_generate keeps the FULL-recompute path (no cache) and
// argmaxes. This file uses the KV-cache (one cache PER BEAM) and tracks num_beams hypotheses with
// cumulative logprob, length-normalized scoring, eos->finished bookkeeping. rig_generate is untouched.
//
// DATAFLOW (mirrors HF generation/beam_search + the rig grammar):
//   PREFILL (template cache): inputs_embeds = [ mesh_cond (n_cond rows) ++ embed(start_tokens) ];
//     prefill those P = n_cond + |start_tokens| positions once -> last-position logits.
//   STEP 0 (expand prefix -> num_beams beams): grammar-mask the prefix logits to
//     allowed_next_tokens(start_tokens), log_softmax over the FULL model vocab (masked entries are
//     -inf -> 0 prob), take the top num_beams tokens. For each: FORK the template cache (copy region
//     [0,P)), DECODE the chosen token at position P, score = logp[token].
//   STEPS 1..: for each ACTIVE beam, take its last decoded logits, grammar-mask
//     allowed_next_tokens(beam.sequence), log_softmax, and for each allowed token t emit candidate
//     (parent, t, parent.score + logp[t]). Pool ALL candidates across active beams; consider the top
//     2*num_beams (HF trick: keeps active beams from being starved when several beams want eos), then
//     fill num_beams output slots in descending candidate score:
//        * token == model_eos -> FINISHED hypothesis {seq++eos, normscore = score / (gen_len^lp)}.
//        * else -> a NEW active beam: fork parent cache (copy [0,parent.pos)), decode t at parent.pos.
//     Keep exactly num_beams active beams (fewer as they finish). Finished beams compete by normscore.
//   STOP: when num_beams hypotheses finished (early_stopping=True MVP), or all beams finished, or
//     max_new_tokens reached. Return the finished hypothesis with the highest normscore (fall back to
//     the best active beam, length-normalized, if none finished).
//
// CACHE FORK (device-to-device, ~100x faster than the old host bounce): per layer, copy the
//   [0,pos) cache region with a single cudaMemcpyDeviceToDevice on the tensors' DEVICE data pointers
//   (ggml-cuda stores the device ptr in tensor->data for the default backend buffer):
//     cudaMemcpy(dst->k_cache[l]->data, src->k_cache[l]->data, pos*nb2, cudaMemcpyDeviceToDevice)
//   (nb2 = stride of one position = head_dim*n_kv*4 bytes.) The CPU fallback build (no M1_USE_CUDA)
//   keeps the host bounce via ggml_backend_tensor_get/set. This is PURELY a copy-mechanism change:
//   the SAME [0,pos) bytes are copied, so same-seed sampling produces identical tokens.
//
// ORDER-SAFETY via DOUBLE-BUFFERING (gen A / gen B). The old recycle reused the just-freed parent
//   caches as children, so a destination could alias a parent still needed by another survivor; it
//   worked around this by snapshotting every parent region to HOST first (read/write decoupled) and
//   restoring after — itself a host bounce. We remove BOTH bounces by keeping TWO disjoint pools:
//   gen A holds the current beams' parent caches; gen B holds fresh child caches. Each step forks
//   parents (gen A, never written during the copy) device-to-device into gen B children, decodes into
//   B, then SWAPS A<->B for the next step. Because a child is ALWAYS a distinct gen-B cache, no
//   destination can ever alias a live parent — order-safety holds with no snapshot/restore.
//
// CACHE COUNT / VRAM: we allocate 2*num_beams + 1 caches (gen A: num_beams, gen B: num_beams, plus 1
//   template that primes gen A through step 0). Each cache is per-layer K and V of
//   [head_dim, n_kv, max_seq] f32:  bytes = 2 * n_layers * head_dim * n_kv * max_seq * 4.
//   For the golden config (hd=128, n_kv=8, n_layers=28) and max_seq ~= n_cond(512)+600+2+8 = 1122:
//     per cache = 2*28*128*8*1122*4 ~= 257 MB; (2*10+1)=21 caches ~= 5.4 GB of cache VRAM. Plus the
//   resident qwen3 weights (~2.4 GB fp32) + per-step intermediates ~= 7.8 GB. Fits the 12 GB 3060,
//   but tighter than before — VRAM RISK if max_seq grows. (If tight, trim max_seq or num_beams.)
#pragma once
#include "qwen3_decode.hpp"
#include "rig_grammar.hpp"
#include "rig_generate.hpp"   // reuse rig::apply_grammar_mask (+ keeps the greedy path available)
#include "m1_ggml.hpp"
#ifdef M1_USE_CUDA
#include <cuda_runtime.h>     // cudaMemcpy / cudaMemcpyDeviceToDevice for the device-to-device fork
#endif
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <random>
#include <unordered_set>
#include <stdexcept>

namespace rig {

// ---- VRAM peak instrumentation (CUDA only). cudaMemGetInfo's (total-free) is the true device
//      high-watermark at the call instant — it sees the cache pool + weights + live activations.
//      The e2e's own vram_sample() runs AFTER rig_beam_generate returns (caches already freed), so
//      it MISSES the beam-phase peak entirely; this samples WHILE the 2*num_beams+1 caches are live.
static inline double rig_vram_used_mib() {
#ifdef M1_USE_CUDA
    size_t fb = 0, tb = 0;
    if (cudaMemGetInfo(&fb, &tb) == cudaSuccess && tb > 0) return (double)(tb - fb) / (1024.0 * 1024.0);
#endif
    return -1.0;
}

// ============================================================================
// BEAM-SAMPLE warper helpers — copied EXACTLY from rig_sample.hpp (same HF math/order).
// Used only for choosing WHICH tokens a beam proposes (do_sample=true). The cumulative-logprob
// SCORE still comes from beam_log_softmax_inplace over the (rep_penalty + grammar-masked) logits.
//
// Order (matches HF logits_warpers): temperature -> top_k -> top_p -> softmax -> multinomial.
// Input `lg` is ALREADY rep-penalized + grammar-masked (disallowed == -inf), as in rig_sample.
// ============================================================================

// (c) TEMPERATURE: divide finite logits.
static inline void beam_warp_temperature(std::vector<float>& lg, float temperature) {
    if (temperature != 1.0f && temperature > 0.0f)
        for (float& v : lg) if (std::isfinite(v)) v /= temperature;
}

// (d) TOP_K: keep the k largest FINITE logits, rest -inf. (If fewer than k allowed, keep all.)
static inline void beam_warp_top_k(std::vector<float>& lg, int top_k, int min_tokens_to_keep = 1) {
    if (top_k <= 0) return;
    top_k = std::max(top_k, min_tokens_to_keep);
    std::vector<std::pair<float,int>> fin;
    fin.reserve(lg.size());
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
static inline void beam_warp_top_p(std::vector<float>& lg, float top_p, int min_tokens_to_keep = 1) {
    if (top_p >= 1.0f) return;
    std::vector<std::pair<float,int>> fin;
    for (size_t i = 0; i < lg.size(); ++i) if (std::isfinite(lg[i])) fin.push_back({ lg[i], (int)i });
    if (fin.size() <= 1) return;
    std::sort(fin.begin(), fin.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });
    double mx = fin[0].first;
    double sum = 0.0;
    for (auto& pr : fin) sum += std::exp((double)pr.first - mx);
    std::vector<char> keep(lg.size(), 0);
    double cum = 0.0;
    for (size_t r = 0; r < fin.size(); ++r) {
        double p = std::exp((double)fin[r].first - mx) / sum;
        keep[fin[r].second] = 1;
        cum += p;
        if (cum >= (double)top_p && (int)r + 1 >= min_tokens_to_keep) break;
    }
    const float ninf = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < lg.size(); ++i) if (!keep[i]) lg[i] = ninf;
}

// (f) MULTINOMIAL DRAW over the warped survivor distribution `lg` (softmax + CDF draw).
//     Returns the sampled token id, or -1 if everything is masked. Identical to rig_sample (f).
static inline int beam_multinomial_draw(const std::vector<float>& lg, std::mt19937_64& rng) {
    double mx = -std::numeric_limits<double>::infinity();
    for (float v : lg) if (std::isfinite(v) && (double)v > mx) mx = (double)v;
    if (!std::isfinite(mx)) return -1;
    double sum = 0.0;
    for (float v : lg) if (std::isfinite(v)) sum += std::exp((double)v - mx);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double u = uni(rng) * sum;
    double acc = 0.0;
    for (size_t i = 0; i < lg.size(); ++i) {
        if (!std::isfinite(lg[i])) continue;
        acc += std::exp((double)lg[i] - mx);
        if (u < acc) return (int)i;
    }
    for (size_t i = lg.size(); i-- > 0; ) if (std::isfinite(lg[i])) return (int)i;
    return -1;
}

// Sample up to `want` DISTINCT tokens from the warped distribution `warped` (HF beam_sample draws
// 2 per beam without replacement). Each drawn token is removed (set -inf) before the next draw so the
// draws are distinct. Returns the list of sampled token ids (<= want; fewer if survivors run out).
static inline std::vector<int> beam_sample_distinct(std::vector<float> warped, int want,
                                                    std::mt19937_64& rng) {
    std::vector<int> out;
    out.reserve(want);
    for (int n = 0; n < want; ++n) {
        int t = beam_multinomial_draw(warped, rng);
        if (t < 0) break;                 // no finite survivors left
        out.push_back(t);
        warped[t] = -std::numeric_limits<float>::infinity();   // sample-without-replacement
    }
    return out;
}

// ---- RoPE cos/sin [hd, L] for absolute positions [p0, p0+L) (identical to qwen3_decode_test). ----
static inline void beam_rope_tables(const Qwen3Cfg& cfg, int64_t p0, int64_t L,
                                    std::vector<float>& cosb, std::vector<float>& sinb) {
    const int hd = cfg.head_dim, half = hd / 2;
    cosb.assign((size_t)hd * L, 0.f);
    sinb.assign((size_t)hd * L, 0.f);
    // Upstream Qwen3 is loaded in BF16, so RotaryEmbedding's persistent
    // inv_freq buffer is BF16 too.  Its forward promotes those rounded values
    // to F32 for position multiplication and sin/cos; computing an unrounded
    // host double here moves the RoPE phase enough to perturb attention.
    std::vector<float> invf(half);
    for (int j = 0; j < half; j++) {
        const float inv = std::pow((float)cfg.rope_theta, -(float)(2 * j) / hd);
        invf[j] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(inv));
    }
    for (int64_t t = 0; t < L; t++) {
        int64_t p = p0 + t;
        for (int i = 0; i < hd; i++) {
            const float ang = (float)p * invf[i < half ? i : i - half];
            cosb[t * hd + i] = std::cos(ang);
            sinb[t * hd + i] = std::sin(ang);
        }
    }
}

// ---- repetition penalty (HF/CTRL): for every token id already in `seq`, logit/=pen if >0 else *pen.
// Applied to the RAW logits BEFORE the grammar mask (mirrors HF: RepetitionPenaltyLogitsProcessor runs
// before the custom grammar processor). Dedups the seen-set so each id is penalized once.
static inline void beam_repetition_penalty(std::vector<float>& logits, const std::vector<int>& seq,
                                           float penalty) {
    if (penalty == 1.0f) return;
    std::vector<char> seen(logits.size(), 0);
    for (int t : seq) if (t >= 0 && t < (int)logits.size()) seen[t] = 1;
    for (size_t i = 0; i < logits.size(); ++i)
        if (seen[i]) logits[i] = logits[i] > 0 ? logits[i] / penalty : logits[i] * penalty;
}

// ---- log_softmax over a vocab vector, computed only over FINITE (non -inf, grammar-allowed) entries.
// Masked (-inf) entries stay -inf (logprob -inf == 0 probability). Numerically stable (subtract max).
static inline void beam_log_softmax_inplace(std::vector<float>& logits) {
    float mx = -std::numeric_limits<float>::infinity();
    for (float v : logits) if (std::isfinite(v) && v > mx) mx = v;
    if (!std::isfinite(mx)) return;   // everything masked (shouldn't happen with a valid grammar)
    double sum = 0.0;
    for (float v : logits) if (std::isfinite(v)) sum += std::exp((double)(v - mx));
    float logZ = mx + (float)std::log(sum);
    for (float& v : logits) {
        if (std::isfinite(v)) v -= logZ;          // log p = logit - logsumexp
        // else: leave -inf
    }
}

// ----------------------------------------------------------------------------
// Per-beam state.
//   cache : owns its Qwen3KvCache (the beam's KV through cache.pos positions).
//   sequence : grammar sequence (model-vocab ids), starts == start_tokens.
//   last_logits : the model-vocab logits at the beam's most recently decoded position (cache.pos-1
//                 absolute); seeds the next step's candidate expansion.
//   score : cumulative sum of selected-token logprobs (HF beam_scores).
//   finished : true once model_eos has been emitted (then it's a completed hypothesis).
// ----------------------------------------------------------------------------
struct RigBeam {
    Qwen3KvCache*      cache = nullptr;   // non-owning: lifetime held by `owner`; freed via free_pool
    std::vector<int>   sequence;
    std::vector<float> last_logits;   // [vocab] at the last decoded position (for active beams)
    double score = 0.0;               // cumulative logprob
    bool   finished = false;
};

// A completed hypothesis. `eos` distinguishes a GENUINE termination (emitted model_eos -> a clean
// rig) from a max-length TRUNCATION (budget exhausted mid-skeleton). HF returns only finished hyps;
// we additionally PREFER eos-terminated ones so a clean rig always beats a runaway truncation,
// regardless of length-normalized-score quirks (length_penalty=1.0 otherwise lets a long confident
// open skeleton outscore a clean eos hyp — the "more beams = worse" / runaway bug).
struct RigHyp {
    std::vector<int> sequence;        // start_tokens ++ generated, ending at model_eos (if eos)
    double normscore = 0.0;           // score / (gen_len ^ length_penalty)
    bool   eos = false;               // true = emitted model_eos; false = max-length truncation
};

// ----------------------------------------------------------------------------
// THE GPU BEAM LOOP.
//
//   H            : M1Harness with the qwen3 weights registered. Weights are allocated + uploaded ONCE
//                  here (first graph). Must be a FRESH harness used only for this loop.
//   cfg          : Qwen3Cfg.
//   embed_table  : host row-major [vocab, hidden] embed_tokens.weight (host token-embed gather).
//   mesh_cond    : host row-major [n_cond, hidden] R1 mesh-condition.
//   start_tokens : grammar init, == {bos, cls} (model-vocab ids).
//   spec         : grammar spec.
//   num_beams    : 10 (the proven config).
//   max_new_tokens : hard cap on generated tokens (per beam).
//   length_penalty : HF length_penalty (1.0).
//
// Returns the best finished hypothesis sequence (start_tokens ++ generated, ending at model_eos),
// directly comparable to the python output_ids.
// ----------------------------------------------------------------------------
inline std::vector<int> rig_beam_generate(M1Harness& H, const Qwen3Cfg& cfg,
                                          const float* embed_table, /* [vocab, hidden] */
                                          const float* mesh_cond, int n_cond, /* [n_cond, hidden] */
                                          const std::vector<int>& start_tokens,
                                          const GrammarSpec& spec = GrammarSpec(),
                                          int num_beams = 10,
                                          int max_new_tokens = 600,
                                          float length_penalty = 1.0f,
                                          float repetition_penalty = 2.0f,
                                          bool do_sample = true,
                                          float temperature = 1.0f,
                                          int top_k = 5,
                                          float top_p = 0.95f,
                                          uint64_t seed = 0,
                                          bool verbose = true,
                                          std::vector<RigHyp>* finished_out = nullptr) {
    const int hidden = cfg.hidden, hd = cfg.head_dim, V = cfg.vocab;
    const int P = n_cond + (int)start_tokens.size();          // prefix length (cond rows + start tokens)
    // SHARED-PREFIX KV: the P-row prefix (mesh_cond ++ start tokens) is IDENTICAL across all beams, so
    // it is computed ONCE into a single read-only `pre` cache and never copied into a beam. Each beam
    // cache holds only its GENERATED suffix (abs positions [P, P+gen)). This shrinks every beam cache
    // from (P + max_new) to (max_new) positions AND removes the P-row prefix from every per-step fork.
    const int suffix_max = max_new_tokens + 4;                // per-beam cache holds only the suffix
    const int pre_max    = P;                                 // the single shared prefix cache

    // ---- beam-SAMPLE RNG: advanced deterministically across all steps/beams (one shared stream). ----
    std::mt19937_64 rng(seed);

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

    // ---- allocate 2*num_beams + 1 caches: gen-A (num_beams parents) + gen-B (num_beams children) +
    //      1 template (primes gen A at step 0). `owner` keeps them alive for the whole loop; live
    //      ownership is tracked by raw pointers (beams own some; the rest sit in free_pool). DOUBLE
    //      BUFFER: a child is always forked into a free_pool cache (never an active parent) -> the
    //      device-to-device fork can never overwrite a parent another survivor still reads. ----
    // Pool: 2*num_beams SUFFIX caches (gen A parents + gen B children; the prefix is shared so the old
    // +1 prefix-template is no longer needed) PLUS one shared read-only prefix cache `pre`.
    std::vector<std::unique_ptr<Qwen3KvCache>> owner;
    owner.reserve(2 * num_beams + 1);
    for (int b = 0; b < 2 * num_beams; ++b) {
        auto c = std::make_unique<Qwen3KvCache>();
        c->init(H, cfg, suffix_max);
        owner.push_back(std::move(c));
    }
    auto pre_uptr = std::make_unique<Qwen3KvCache>();
    pre_uptr->init(H, cfg, pre_max);
    Qwen3KvCache* pre = pre_uptr.get();
    // free_pool = suffix caches not currently owned by a beam.
    std::vector<Qwen3KvCache*> free_pool;
    free_pool.reserve(2 * num_beams);
    for (auto& c : owner) free_pool.push_back(c.get());

    bool weights_ready = false;
    double vram_peak = rig_vram_used_mib();   // beam-phase device high-watermark (caches live).
    // cudaMemGetInfo isn't free (~0.5ms), and there are ~10*maxnew chunk computes — so sample sparsely:
    // the first few calls (cache alloc + weight upload) and every 64th after (the activation peak grows
    // with klen, so periodic late sampling still catches it). Bounds overhead to ~a few hundred calls.
    long vram_calls = 0;
    auto vram_bump = [&]() {
        if (vram_calls++ < 8 || (vram_calls & 63) == 0) {
            double u = rig_vram_used_mib(); if (u > vram_peak) vram_peak = u;
        }
    };
    if (verbose) printf("[rig_beam] VRAM after pool alloc (%d suffix caches @ %d + 1 shared prefix @ %d): "
                        "%.0f MiB used\n", 2 * num_beams, suffix_max, pre_max, vram_peak);

    // ---- shared chunk runner: build a cache-backed graph for `cache`, decode L tokens at write0
    //      from host embeds `chunk_embeds` [hidden, L], return the chunk's logits [V, L] (host). ----
    auto run_chunk = [&](Qwen3KvCache& cache, int write0, int L,
                         const float* chunk_embeds, bool causal_mask,
                         Qwen3KvCache* pre_c = nullptr, int pre_len = 0) -> std::vector<float> {
        // write0 is the SUFFIX write offset when pre_c is supplied; the ABSOLUTE position of the first
        // token (for RoPE + causal mask) is pre_len + write0. klen = total keys attended (prefix + suffix
        // past + new). For prefill (pre_c==null) abs0==write0 and this reduces to the original path.
        const int abs0 = (pre_c ? pre_len : 0) + write0;
        std::vector<float> cosb, sinb;
        beam_rope_tables(cfg, abs0, L, cosb, sinb);

        const int klen = abs0 + L;
        std::vector<float> maskb;
        if (causal_mask) {
            maskb.assign((size_t)klen * L, 0.f);
            for (int q = 0; q < L; q++) {
                int qabs = abs0 + q;
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
                                                 write0, L, maskT, writes, pre_c, pre_len);   // [vocab, L]
        ggml_set_output(logits);

        ggml_cgraph* gf = new_graph(cctx, 32768);
        ggml_build_forward_expand(gf, logits);
        for (ggml_tensor* w : writes) ggml_build_forward_expand(gf, w);   // ensure cache writes run

        if (!weights_ready) { upload_weights_maybe_f16(H); weights_ready = true; }

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(H.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            ggml_gallocr_free(ga); ggml_free(cctx);
            throw std::runtime_error("rig_beam_generate: gallocr_alloc_graph failed");
        }

        ggml_backend_tensor_set(inp,  chunk_embeds, 0, (size_t)L * hidden * sizeof(float));
        ggml_backend_tensor_set(cosT, cosb.data(),  0, cosb.size() * sizeof(float));
        ggml_backend_tensor_set(sinT, sinb.data(),  0, sinb.size() * sizeof(float));
        if (causal_mask) ggml_backend_tensor_set(maskT, maskb.data(), 0, maskb.size() * sizeof(float));

        ggml_backend_graph_compute(H.backend, gf);
        vram_bump();   // device high-watermark while this chunk's activations + gallocr scratch are live

        std::vector<float> out((size_t)L * V);
        ggml_backend_tensor_get(logits, out.data(), 0, (size_t)L * V * sizeof(float));

        ggml_gallocr_free(ga);
        ggml_free(cctx);
        cache.pos = write0 + L;
        return out;
    };

    // ---- cache fork: copy region [0, pos) of every layer's K/V from src to dst. ----
    // nb2 = bytes per position (== head_dim*n_kv*sizeof(float) for the [hd,n_kv,max_seq] cache).
    // CUDA: a single cudaMemcpyDeviceToDevice per layer on the tensors' DEVICE data pointers (no PCIe
    // round-trip). CPU fallback: the original host bounce. src and dst are ALWAYS distinct caches
    // (double-buffered gen A->gen B), so this never overwrites a region the source still needs.
    std::vector<char> fork_host;   // CPU-fallback reusable host buffer (largest single-layer copy).
    auto fork_cache = [&](const Qwen3KvCache& src, Qwen3KvCache& dst, int pos) {
        if (pos <= 0) { dst.pos = 0; return; }
        for (int l = 0; l < cfg.n_layers; ++l) {
            size_t bytes = (size_t)pos * src.k_cache[l]->nb[2];   // [0,pos) region, one layer
#ifdef M1_USE_CUDA
            cudaMemcpy(dst.k_cache[l]->data, src.k_cache[l]->data, bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(dst.v_cache[l]->data, src.v_cache[l]->data, bytes, cudaMemcpyDeviceToDevice);
#else
            if (fork_host.size() < bytes) fork_host.resize(bytes);
            ggml_backend_tensor_get(src.k_cache[l], fork_host.data(), 0, bytes);
            ggml_backend_tensor_set(dst.k_cache[l], fork_host.data(), 0, bytes);
            ggml_backend_tensor_get(src.v_cache[l], fork_host.data(), 0, bytes);
            ggml_backend_tensor_set(dst.v_cache[l], fork_host.data(), 0, bytes);
#endif
        }
        dst.pos = pos;
    };

    // ============================ PREFILL the shared prefix cache ============================
    pre->reset();
    std::vector<float> prefix_logits_all = run_chunk(*pre, 0, P, prefix_embeds.data(), /*causal=*/true);
    // last-position logits (column P-1) of the prefill.
    std::vector<float> prefix_logits(prefix_logits_all.begin() + (size_t)(P - 1) * V,
                                     prefix_logits_all.begin() + (size_t)P * V);
    if (verbose) printf("[rig_beam] prefill P=%d (n_cond=%d + |start|=%zu), suffix_max=%d, beams=%d, "
                        "do_sample=%d temp=%.2f top_k=%d top_p=%.2f seed=%llu\n",
                        P, n_cond, start_tokens.size(), suffix_max, num_beams,
                        (int)do_sample, temperature, top_k, top_p, (unsigned long long)seed);

    // ============================ STEP 0: expand prefix -> beams ========================
    {
        std::vector<int> allowed = allowed_next_tokens(start_tokens, spec);
        // HF 5.9.0 _beam_search basis: log_softmax FIRST, then ALL processors on the LOG_PROBS in beam
        // order (rep_penalty -> grammar mask -> temperature -> top_k -> top_p). The WARPED logp is BOTH
        // the sampling distribution AND the carried score (HF unifies them; running_beam_scores[0]=0 so
        // step-0 accumulated == warped logp). This matches the steps-1.. scheme below exactly.
        std::vector<float> s0 = prefix_logits;       // raw prefix logits (copy)
        beam_log_softmax_inplace(s0);                // s0 = log_softmax(raw)
        beam_repetition_penalty(s0, start_tokens, repetition_penalty);   // HF applies on log_probs
        apply_grammar_mask(s0, allowed);             // additive -inf for disallowed
        if (do_sample) {
            beam_warp_temperature(s0, temperature);
            beam_warp_top_k(s0, top_k);
            beam_warp_top_p(s0, top_p);
        }
        // cands = (warped_logp, token); beam.score carries the warped logp (== HF accumulated at step 0).
        std::vector<std::pair<float,int>> cands;
        if (do_sample) {
            std::vector<int> drawn = beam_sample_distinct(s0, num_beams, rng);   // multinomial ~ softmax(s0)
            cands.reserve(drawn.size());
            for (int t : drawn) if (std::isfinite(s0[t])) cands.push_back({ s0[t], t });
            // stable: order spawned beams by score (deterministic post-sample, like the steps-1.. pool).
            std::sort(cands.begin(), cands.end(), [](auto&a, auto&b){ return a.first > b.first; });
        } else {
            cands.reserve(allowed.size());
            for (int t : allowed) if (std::isfinite(s0[t])) cands.push_back({ s0[t], t });
            std::sort(cands.begin(), cands.end(), [](auto&a, auto&b){ return a.first > b.first; });
        }
        int nb = std::min((int)cands.size(), num_beams);
        if (nb == 0) throw std::runtime_error("rig_beam_generate: no allowed tokens at step 0");
        // (Tried padding to num_beams with cyclic duplicate first-tokens to match HF's constant beam
        // width — it REGRESSED reliability: duplicate lineages share a cache/sequence and don't add real
        // exploration, they just fill slots and shift the RNG stream. Distinct survivors is better here.)
        // Build the live beams: fork the template cache into beam b's cache, decode the chosen token.
        std::vector<RigBeam> beams;
        beams.reserve(num_beams);
        for (int b = 0; b < nb; ++b) {
            int tok = cands[b].second;
            RigBeam beam;
            beam.cache = free_pool.back(); free_pool.pop_back();  // grab a free (gen-B) suffix cache
            beam.cache->reset();   // fresh suffix (pos=0); the P-row prefix is shared via `pre` (no fork)
            beam.sequence = start_tokens; beam.sequence.push_back(tok);
            const float* erow = embed_row(tok);
            std::vector<float> step_embed(erow, erow + hidden);
            // decode the first generated token at SUFFIX offset 0 (abs position P), attending over `pre`.
            std::vector<float> lg = run_chunk(*beam.cache, 0, 1, step_embed.data(), /*causal=*/false, pre, P);
            beam.last_logits = std::move(lg);              // [V] at position P
            beam.score = cands[b].first;
            beam.finished = false;
            beams.push_back(std::move(beam));
        }
        if (verbose) printf("[rig_beam] step 0: spawned %zu beams (top tokens %d..)\n",
                            beams.size(), beams.empty() ? -1 : beams[0].sequence.back());
        if (std::getenv("RIG_LOGIT_PROBE") && !beams.empty()) {  // step-0 beam-0 decode-logit parity probe
            double s=0,ss=0; int am=0; float mx=-1e30f;
            for (size_t i=0;i<beams[0].last_logits.size();++i){ float v=beams[0].last_logits[i];
                s+=v; ss+=(double)v*v; if(v>mx){mx=v;am=(int)i;} }
            printf("[probe] SEQ step0 beam0 tok=%d : sum=%.5f sumsq=%.5f argmax=%d max=%.6f\n",
                   beams[0].sequence.back(), s, ss, am, mx);
        }

        // ============================ STEPS 1.. (HF 5.9.0 _beam_search faithful port) ====
        // Reference: transformers/generation/utils.py::_beam_search (v5.9.0). Per step:
        //   1+2. s = log_softmax(raw_logits); apply processors ON LOG_PROBS in HF beam order
        //        (rep_penalty -> grammar mask -> temperature -> top_k -> top_p). The WARPED logp IS the
        //        score basis (HF unifies sampling-dist and score; a top_k/top_p-killed token is no
        //        candidate). accumulated[i][t] = warped_logp_i(t) + beam_score_i, where beam_score is
        //        the CUMULATIVE warped logp (HF: running_beam_scores updated to the gathered accumulated).
        //        [Previously this file applied rep_penalty to RAW logits, scored from UN-temperatured
        //         logp, and re-divided the whole cumulative by T at sample time — that over-penalized the
        //         low-prob switch token and under-penalized runaway repeats (rep-penalty on log_probs is
        //         what sinks a repeating runaway), so coherent stop-context beams were pruned. Fixed.]
        //   3. select beams_to_keep = 2*num_beams candidates from the JOINT (beams x vocab) pool:
        //      do_sample -> multinomial WITHOUT replacement ~ softmax(accumulated); else -> top-k by acc.
        //   4. a candidate whose token == model_eos is FINISHED (length-normalized acc -> finished pool,
        //      bounded to num_beams by normscore, like HF BeamHypotheses); the non-eos candidates, top
        //      num_beams by acc, are the next ACTIVE beams (HF _get_running_beams_for_next_iteration).
        //   5. early-stop heuristic (early_stopping=False, length_penalty>=0): keep going while the best
        //      active beam, length-normalized at the current length, could still beat the WORST finished
        //      hyp; STOP when it can't (HF _check_early_stop_heuristic + _beam_search_has_unfinished).
        //   FINAL (pure HF): a still-running mid-skeleton beam is NEVER a winner during the search — HF
        //      only ever returns genuinely-finished hyps (utils.py:3083/3095/3412). We bank the still-
        //      running beams as max-length TRUNCATIONS only when the budget is exhausted (HF banks max-
        //      length completions too), then pick the highest length-normalized score across the pool —
        //      no eos-vs-truncation preference (that would be a non-HF "is it broken" check / sticky tape).
        std::vector<RigHyp> finished;            // model_eos-terminated hyps; kept top num_beams by normscore
        bool improvement_possible = true;        // == any(is_early_stop_heuristic_unsatisfied)
        bool stopped_early = false;              // true if the loop broke before exhausting max_new_tokens
        const int beams_to_keep = 2 * num_beams; // HF: max(2, 1 + n_eos_tokens) * num_beams, n_eos == 1
        auto norm = [&](double score, int gen_len) {
            return score / std::pow((double)std::max(1, gen_len), (double)length_penalty);
        };
        auto worst_finished_norm = [&]() -> double {
            if (finished.empty()) return -1e30;
            double w = std::numeric_limits<double>::infinity();
            for (auto& h : finished) w = std::min(w, h.normscore);
            return w;
        };
        // keep the finished pool bounded to num_beams best (by normscore), like HF's BeamHypotheses.
        auto bank_finished = [&](std::vector<int> seq, double normscore, bool eos) {
            finished.push_back({ std::move(seq), normscore, eos });
            if ((int)finished.size() > num_beams) {
                auto worst = std::min_element(finished.begin(), finished.end(),
                    [](const RigHyp&a, const RigHyp&b){ return a.normscore < b.normscore; });
                finished.erase(worst);
            }
        };

        for (int step = 1; step < max_new_tokens; ++step) {
            const int gen_len = step + 1;        // generated-token count of any candidate built this step
            int active = (int)beams.size();
            if (active == 0) break;

            // ---- 1+2: joint candidate pool. acc = warped_logp + beam_score (cumulative warped logp). ----
            struct JC { double acc; int parent; int tok; };
            std::vector<JC> jc;
            for (int i = 0; i < active; ++i) {
                std::vector<float> s = beams[i].last_logits;            // raw logits (copy)
                std::vector<int> allowed2 = allowed_next_tokens(beams[i].sequence, spec);
                beam_log_softmax_inplace(s);                            // s = log_softmax(raw)
                beam_repetition_penalty(s, beams[i].sequence, repetition_penalty);  // HF: on log_probs
                apply_grammar_mask(s, allowed2);                        // additive -inf for disallowed
                if (do_sample) {
                    beam_warp_temperature(s, temperature);              // /T
                    beam_warp_top_k(s, top_k);
                    beam_warp_top_p(s, top_p);                          // finite entries == candidates
                }
                for (size_t t = 0; t < s.size(); ++t)
                    if (std::isfinite(s[t])) jc.push_back({ beams[i].score + (double)s[t], i, (int)t });
            }
            if (jc.empty()) break;

            // ---- 3: pick beams_to_keep candidates from the joint pool (indices into jc, in pick order). ----
            int want = std::min((int)jc.size(), beams_to_keep);
            std::vector<int> picks; picks.reserve(want);
            if (do_sample) {
                // multinomial WITHOUT replacement ~ softmax(acc) over the joint pool.
                std::vector<char> used(jc.size(), 0);
                double mx = -std::numeric_limits<double>::infinity();
                for (auto& c : jc) mx = std::max(mx, c.acc);
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                for (int n = 0; n < want; ++n) {
                    double sum = 0.0;
                    for (size_t k = 0; k < jc.size(); ++k) if (!used[k]) sum += std::exp(jc[k].acc - mx);
                    if (sum <= 0.0) break;
                    double u = uni(rng) * sum, accum = 0.0; int pick = -1;
                    for (size_t k = 0; k < jc.size(); ++k) {
                        if (used[k]) continue;
                        accum += std::exp(jc[k].acc - mx);
                        if (u < accum) { pick = (int)k; break; }
                    }
                    if (pick < 0) for (size_t k = jc.size(); k-- > 0; ) if (!used[k]) { pick = (int)k; break; }
                    if (pick < 0) break;
                    used[pick] = 1; picks.push_back(pick);
                }
            } else {
                std::vector<int> idx(jc.size());
                for (size_t k = 0; k < jc.size(); ++k) idx[k] = (int)k;
                std::partial_sort(idx.begin(), idx.begin() + want, idx.end(),
                    [&](int a, int b){ return jc[a].acc > jc[b].acc; });
                picks.assign(idx.begin(), idx.begin() + want);
            }

            // ---- 4: eos candidates -> finished pool; non-eos top num_beams by acc -> next active set. ----
            std::vector<std::pair<double,int>> noneos;   // (acc, index into jc)
            noneos.reserve(picks.size());
            // HF 5.9.0 `top_num_beam_mask`: bank an eos candidate as finished ONLY if it's in the top
            // num_beams of the 2*num_beams picks (rank order greedy / draw order sample). Candidates beyond
            // num_beams are live-beam backups that HF masks out of the finished update. Banking ALL eos
            // picks let short low-ranked eos hyps win on length-normalized score -> sparser rigs than Python.
            for (int pi = 0; pi < (int)picks.size(); ++pi) {
                const JC& c = jc[picks[pi]];
                if (c.tok == spec.model_eos) {
                    if (pi < num_beams) {
                        std::vector<int> seq = beams[c.parent].sequence; seq.push_back(spec.model_eos);
                        bank_finished(std::move(seq), norm(c.acc, gen_len), /*eos=*/true);  // genuine termination
                    }
                    // else: eos ranked >= num_beams -> discarded (not finished, not a running beam), per HF.
                } else {
                    noneos.push_back({ c.acc, picks[pi] });
                }
            }
            std::sort(noneos.begin(), noneos.end(),
                      [](const auto&a, const auto&b){ return a.first > b.first; });
            int nkeep = std::min((int)noneos.size(), num_beams);
            struct Survivor { int parent; int tok; double score; std::vector<int> seq; };
            std::vector<Survivor> survivors;
            survivors.reserve(nkeep);
            for (int r = 0; r < nkeep; ++r) {
                const JC& c = jc[noneos[r].second];
                std::vector<int> seq = beams[c.parent].sequence; seq.push_back(c.tok);
                survivors.push_back({ c.parent, c.tok, c.acc, std::move(seq) });
            }

            // ---- DOUBLE-BUFFERED cache fork (ORDER-SAFE, device-to-device). gen A = current `beams`
            // caches (parents); gen B = caches drawn from free_pool. A free_pool cache is by definition
            // NOT owned by any current beam, so forking a parent (gen A) into a child (gen B) can NEVER
            // overwrite a parent that another survivor still needs to fork from. We fork ALL survivors
            // FIRST (parents untouched), THEN return the spent parent caches to free_pool. No snapshot,
            // no host bounce — reads (parents) and writes (children) target disjoint caches.
            const size_t S = survivors.size();
            if (free_pool.size() < S)
                throw std::runtime_error("rig_beam_generate: ran out of free caches");
            std::vector<RigBeam> next_beams;
            next_beams.reserve(S);
            // PASS 1: fork every survivor's parent region [0,ppos) into a fresh gen-B cache + decode.
            for (size_t k = 0; k < S; ++k) {
                Qwen3KvCache* pc  = beams[survivors[k].parent].cache;   // gen-A parent (read-only here)
                Qwen3KvCache* dst = free_pool.back(); free_pool.pop_back();  // gen-B child (write)
                int ppos = pc->pos;                                     // parent SUFFIX length (gen count)
                fork_cache(*pc, *dst, ppos);                            // device-to-device suffix [0,ppos)
                const float* erow = embed_row(survivors[k].tok);
                std::vector<float> step_embed(erow, erow + hidden);
                // decode at SUFFIX offset ppos (abs position P+ppos), attending over shared `pre` + suffix.
                std::vector<float> lg = run_chunk(*dst, ppos, 1, step_embed.data(), /*causal=*/false, pre, P);
                RigBeam nb_beam;
                nb_beam.cache = dst;
                nb_beam.sequence = std::move(survivors[k].seq);
                nb_beam.last_logits = std::move(lg);
                nb_beam.score = survivors[k].score;
                nb_beam.finished = false;
                next_beams.push_back(std::move(nb_beam));
            }
            // PASS 2 (after ALL forks done): retire the gen-A parent caches back to free_pool. Safe now
            // — every child has already read its parent. Next step these become gen B.
            for (auto& ob : beams) if (ob.cache) free_pool.push_back(ob.cache);

            beams = std::move(next_beams);

            // ---- 5: early-stop heuristic (HF _check_early_stop_heuristic, early_stopping=False). The
            //      best active beam (highest cumulative warped score), normalized at the current length,
            //      vs the WORST finished hyp. Once the open beams can no longer beat the finished pool,
            //      improvement is impossible -> stop (HF _beam_search_has_unfinished_sequences). ----
            double best_active = -std::numeric_limits<double>::infinity();
            for (auto& b : beams) best_active = std::max(best_active, b.score);
            double best_active_norm = beams.empty() ? -1e30 : norm(best_active, gen_len);
            improvement_possible = improvement_possible && (best_active_norm > worst_finished_norm());

            if (verbose && (step < 3 || step % 64 == 0))
                printf("[rig_beam] step %d: active=%d finished=%zu worst_fin=%.4f best_act_norm=%.4f\n",
                       step, (int)beams.size(), finished.size(),
                       finished.empty() ? 0.0 : worst_finished_norm(), best_active_norm);

            if (!improvement_possible) { stopped_early = true; break; }   // finished pool dominates -> done
            if (beams.empty())         { stopped_early = true; break; }   // every candidate was eos
        }

        // ---- FINAL (HF-faithful + runaway-proof): the finished pool already holds every eos-terminated
        //      hypothesis. We bank the still-running beams as max-length TRUNCATIONS *only* when the budget
        //      was exhausted (HF banks max-length completions too); on an early-stop / empty exit the open
        //      beams are provably no better than the finished pool, so we never fold them in. Then PREFER
        //      an eos-terminated hyp over any truncation (a clean rig must always beat a runaway), and
        //      among the same class pick the highest normscore.
        bool any_eos = false; for (auto& h : finished) any_eos |= h.eos;
        if (!stopped_early) {                       // budget exhausted -> running beams are max-len truncations
            for (auto& b : beams) {
                int gl = (int)b.sequence.size() - (int)start_tokens.size();
                bank_finished(b.sequence, norm(b.score, gl), /*eos=*/false);
            }
        }
        if (finished.empty()) throw std::runtime_error("rig_beam_generate: no hypotheses produced");

        auto best = std::max_element(finished.begin(), finished.end(),
                        [](const RigHyp& a, const RigHyp& b){ return a.normscore < b.normscore; });
        // See the batched decoder: retain only completed hypotheses for an
        // independent anatomy gate at the production call site.
        if (finished_out) {
            *finished_out = finished;
            std::sort(finished_out->begin(), finished_out->end(),
                      [](const RigHyp& a, const RigHyp& b) { return a.normscore > b.normscore; });
        }
        if (verbose) {
            int gl = (int)best->sequence.size() - (int)start_tokens.size();
            printf("[rig_beam] DONE: %zu hyps (eos-terminated in pool=%d), best %s normscore=%.4f, "
                   "len=%zu (gen=%d), last=%d, terminated=%d\n",
                   finished.size(), (int)any_eos, best->eos ? "EOS" : "trunc", best->normscore,
                   best->sequence.size(), gl, best->sequence.empty() ? -1 : best->sequence.back(),
                   (int)best->eos);
            printf("[rig_beam] VRAM beam-phase peak: %.0f MiB used (%d suffix caches @ %d + shared prefix @ %d)\n",
                   vram_peak, 2 * num_beams, suffix_max, pre_max);
        }
        return best->sequence;
    }
}

} // namespace rig
