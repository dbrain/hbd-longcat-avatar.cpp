// qwen3_batched.hpp — BATCHED-BEAM KV-cache decode for the rig beam search. Decodes ALL active beams'
// next token in ONE forward (batch dim = beams) instead of num_beams sequential single-token decodes.
//
// WHY: nsys showed the sequential per-beam decode is launch-bound + runs the projection matmuls as
// matrix×VECTOR (M=1, bandwidth-bound) — 49% of GPU time. Batching makes them matrix×matrix (M=beams,
// tensor-core efficient) AND cuts kernel launches ~num_beams×. All active beams are always at the SAME
// absolute position (beam search appends exactly one token per beam per step), so a single batched KV
// cache [hd, nkv, suffix_max, beam] + per-beam attention over the beam dim works.
//
// SHARED PREFIX: HF first performs a true B-row prefix forward, so the caller
// uses build_qwen3_prefill_batched() and then retains the identical column-0
// KV cache here as one [hd,nkv,P] prefix.  Suffix attention broadcasts that
// cache over beam; the prefix is never re-materialized as a persistent B-wide
// buffer during decode.
//
// Read-after-write hazard avoidance (mirrors qwen3_decode.hpp): the freshly-computed k/v are written
// into suffix[.,.,step,:] for FUTURE steps, but attention reads suffix[0..step) (PRIOR steps) ++ the
// fresh k_new directly — never the just-written slice in the same graph.
#pragma once
#include "qwen3_forward.hpp"
#include "qwen3_decode.hpp"   // Qwen3KvCache (shared prefix) + Qwen3KvCache::kv_dtype()
#include "m1_ggml.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>

// Batched suffix KV cache: per layer, K and V of [head_dim, n_kv, suffix_max, num_beams]. Beam b's
// generated suffix lives in column b. dtype follows Qwen3KvCache::kv_dtype() (F16 default).
struct Qwen3KvCacheBatched {
    ggml_context*   ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k_cache;   // per layer [hd, nkv, suffix_max, beam]
    std::vector<ggml_tensor*> v_cache;
    int suffix_max = 0;
    int beam = 0;
    int pos = 0;                          // suffix length written (== gen count); == next write index

    void init(M1Harness& H, const Qwen3Cfg& cfg, int suffix_max_, int beam_) {
        suffix_max = suffix_max_; beam = beam_; pos = 0;
        size_t meta = (size_t)(cfg.n_layers * 2 + 8) * ggml_tensor_overhead();
        ggml_init_params ip{ meta, nullptr, /*no_alloc=*/true };
        ctx = ggml_init(ip);
        k_cache.resize(cfg.n_layers);
        v_cache.resize(cfg.n_layers);
        const ggml_type kvt = Qwen3KvCache::kv_dtype();
        for (int l = 0; l < cfg.n_layers; ++l) {
            k_cache[l] = ggml_new_tensor_4d(ctx, kvt, cfg.head_dim, cfg.n_kv, suffix_max, beam);
            v_cache[l] = ggml_new_tensor_4d(ctx, kvt, cfg.head_dim, cfg.n_kv, suffix_max, beam);
            ggml_set_name(k_cache[l], ("bkv.k." + std::to_string(l)).c_str());
            ggml_set_name(v_cache[l], ("bkv.v." + std::to_string(l)).c_str());
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, H.backend);
        if (!buf) throw std::runtime_error("Qwen3KvCacheBatched: alloc_ctx_tensors failed");
    }
    void reset() { pos = 0; }
    ~Qwen3KvCacheBatched() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

// Batched *prefix* prefill.  HF expands inputs_embeds to num_beams rows before
// its first forward.  In BF16 that is observably not numerically equivalent to
// a batch-1 forward followed by a repeat, so the beam generator needs this
// graph in addition to the single-token batched suffix decoder below.
//
// `cache` is intentionally a short-lived Qwen3KvCacheBatched with
// suffix_max=P.  The caller compacts its (identical-input) column zero into
// the long-lived shared 3-D prefix cache after the graph completes.
static inline ggml_tensor* qw_layer_prefill_batched(M1Harness& H, ggml_context* ctx,
                                                    const Qwen3Cfg& cfg, const std::string& p,
                                                    Qwen3KvCacheBatched& cache, int layer,
                                                    ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin,
                                                    int P, int B, ggml_tensor* mask,
                                                    std::vector<ggml_tensor*>& writes,
                                                    std::vector<ggml_tensor*>* attn_last_taps = nullptr,
                                                    std::vector<ggml_tensor*>* post_attn_last_taps = nullptr,
                                                    std::vector<ggml_tensor*>* mlp_last_taps = nullptr,
                                                    std::vector<ggml_tensor*>* gate_last_taps = nullptr,
                                                    std::vector<ggml_tensor*>* up_last_taps = nullptr,
                                                    std::vector<ggml_tensor*>* swiglu_last_taps = nullptr,
                                                    std::vector<ggml_tensor*>* mlp_input_last_taps = nullptr) {
    (void) mask; // causal masking is supplied by the operation-local FA2 contract below.
    const bool trace_layer0_only = std::getenv("RIG_QWEN_LAYER_TRACE_LAYER0_ONLY") != nullptr;
    // Retaining every B-row diagnostic tap keeps the whole prefix graph live
    // and exceeds a 12 GB card.  A selected-layer probe lets parity work walk
    // the stack one block at a time without changing the compute graph.
    const char * trace_layer_env = std::getenv("RIG_QWEN_LAYER_TRACE_LAYER");
    const int trace_layer = trace_layer_env ? std::atoi(trace_layer_env) : -1;
    const bool capture_taps = trace_layer >= 0 ? layer == trace_layer : (!trace_layer0_only || layer == 0);
    const char * trace_pos_env = std::getenv("RIG_QWEN_LAYER_TRACE_POS");
    const int trace_pos = trace_pos_env ? std::clamp(std::atoi(trace_pos_env), 0, P - 1) : P - 1;
    const int hd = cfg.head_dim, nh = cfg.n_heads;
    ggml_tensor *q, *k, *v;
    // qwen3_decode's shared helper is deliberately 3-D for chunks.  Prefix
    // prefill has a real batch dimension, so retain it explicitly here.
    ggml_tensor* h = qw_rmsnorm(ctx, x, H.weight(p + "input_layernorm.weight"), cfg.eps);
    const bool trace_input_norm = std::getenv("RIG_QWEN_LAYER_TRACE_INPUT_NORM") != nullptr;
    if (capture_taps && attn_last_taps && trace_input_norm && layer == 0) {
        ggml_tensor* last = ggml_view_3d(ctx, h, cfg.hidden, 1, 1, h->nb[1], h->nb[2],
                                          (size_t)trace_pos * h->nb[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        attn_last_taps->push_back(last);
    }
    const ggml_type ct = cfg.compute_type;
    q = lin_lp(ctx, H.weight(p + "self_attn.q_proj.weight"), h, ct);
    k = lin_lp(ctx, H.weight(p + "self_attn.k_proj.weight"), h, ct);
    v = lin_lp(ctx, H.weight(p + "self_attn.v_proj.weight"), h, ct);
    const char * qkv_trace = std::getenv("RIG_QWEN_LAYER_TRACE_QKV");
    q = ggml_reshape_4d(ctx, q, hd, cfg.n_heads, P, B);
    k = ggml_reshape_4d(ctx, k, hd, cfg.n_kv,    P, B);
    v = ggml_reshape_4d(ctx, v, hd, cfg.n_kv,    P, B);
    const bool qkv_raw_trace = std::getenv("RIG_QWEN_LAYER_TRACE_QKV_RAW") != nullptr;
    const bool qkv_norm_trace = std::getenv("RIG_QWEN_LAYER_TRACE_QKV_NORM") != nullptr;
    if (capture_taps && attn_last_taps && qkv_trace && qkv_trace[0] && qkv_raw_trace && layer == 0) {
        ggml_tensor * t = qkv_trace[0] == 'q' ? q : qkv_trace[0] == 'k' ? k : v;
        // Raw q/k/v are already [hd, heads, P, B]; qf/kf/vf below need
        // the corresponding permutation because they are [hd, P, heads, B].
        ggml_tensor * last = ggml_view_3d(ctx, t, t->ne[0], t->ne[1], 1,
                                           t->nb[1], t->nb[2], (size_t)trace_pos * t->nb[2]);
        last = ggml_reshape_1d(ctx, ggml_cont(ctx, last), t->ne[0] * t->ne[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        attn_last_taps->push_back(last);
    }
    q = qw_rmsnorm(ctx, q, H.weight(p + "self_attn.q_norm.weight"), cfg.eps);
    k = qw_rmsnorm(ctx, k, H.weight(p + "self_attn.k_norm.weight"), cfg.eps);
    // Selected pre-RoPE Q/K/V diagnostic.  This is intentionally separate
    // from the raw-projection and post-RoPE probes so RMSNorm rounding can be
    // isolated without retaining the full B-row prefix graph.
    if (capture_taps && attn_last_taps && qkv_trace && qkv_trace[0] && qkv_norm_trace && layer == 0) {
        ggml_tensor * t = qkv_trace[0] == 'q' ? q : qkv_trace[0] == 'k' ? k : v;
        ggml_tensor * last = ggml_view_3d(ctx, t, t->ne[0], t->ne[1], 1,
                                           t->nb[1], t->nb[2], (size_t)trace_pos * t->nb[2]);
        last = ggml_reshape_1d(ctx, ggml_cont(ctx, last), t->ne[0] * t->ne[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        attn_last_taps->push_back(last);
    }
    q = qw_rope(ctx, q, cos, sin);
    k = qw_rope(ctx, k, cos, sin);

    ggml_tensor* kc = cache.k_cache[layer];
    ggml_tensor* vc = cache.v_cache[layer];
    // Match cached decode: current attention sees the same rounded K/V
    // representation that later suffix decode will read from the cache.
    if (kc->type != GGML_TYPE_F32) {
        k = ggml_cast(ctx, k, kc->type);
        v = ggml_cast(ctx, v, vc->type);
    }
    writes.push_back(ggml_cpy(ctx, k, kc));
    writes.push_back(ggml_cpy(ctx, v, vc));

    // Match Transformers' B-row prefix kernel rather than reusing ggml's
    // dense masked matmul.  This is an explicit Qwen causal-GQA contract;
    // generic null-mask flash attention remains non-causal.
    if (q->type != GGML_TYPE_BF16) q = ggml_cast(ctx, q, GGML_TYPE_BF16);
    if (k->type != GGML_TYPE_BF16) k = ggml_cast(ctx, k, GGML_TYPE_BF16);
    if (v->type != GGML_TYPE_BF16) v = ggml_cast(ctx, v, GGML_TYPE_BF16);
    // Preserve the Python FA2 physical [B,H,P,D] layout.  The permuted views
    // describe logical [D,P,H,B] to ggml while retaining Python's row/head
    // strides; materialising them swapped the FA2 reduction layout.
    ggml_tensor* qf = ggml_permute(ctx, q, 0, 2, 1, 3); // [hd,P,nh,B], strided
    ggml_tensor* kf = ggml_permute(ctx, k, 0, 2, 1, 3); // [hd,P,nkv,B], strided
    ggml_tensor* vf = ggml_permute(ctx, v, 0, 2, 1, 3); // [hd,P,nkv,B], strided
    // Layer-0-only B20 trace can substitute one pre-FA operand for the
    // attention result. This keeps the trace allocation tiny while isolating
    // Q/K/V rounding from the FlashAttention kernel itself.
    if (capture_taps && attn_last_taps && qkv_trace && qkv_trace[0] && !qkv_raw_trace && !qkv_norm_trace && layer == 0) {
        ggml_tensor * t = qkv_trace[0] == 'q' ? qf : qkv_trace[0] == 'k' ? kf : vf;
        // [hd,P,H,B] -> [hd,H,P,B], then select B0/final P.
        t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
        ggml_tensor * last = ggml_view_3d(ctx, t, t->ne[0], t->ne[1], 1,
                                           t->nb[1], t->nb[2], (size_t)trace_pos * t->nb[2]);
        last = ggml_reshape_1d(ctx, ggml_cont(ctx, last), t->ne[0] * t->ne[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        attn_last_taps->push_back(last);
    }
    ggml_tensor* o = ggml_flash_attn_ext_qwen_causal_gqa(ctx, qf, kf, vf, cfg.attn_scale());
    // flash-attn result already has [hd,nh,P,B] layout.
    if (capture_taps && attn_last_taps && !(qkv_trace && qkv_trace[0] && layer == 0) &&
        !(trace_input_norm && layer == 0)) {
        ggml_tensor* last = ggml_view_3d(ctx, o, hd, nh, 1, o->nb[1], o->nb[2],
                                          (size_t)trace_pos * o->nb[2]);
        last = ggml_reshape_1d(ctx, ggml_cont(ctx, last), hd * nh);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        // Diagnostic graph taps must be explicit outputs; otherwise gallocr
        // may reuse the same temporary for every layer before host readback.
        ggml_set_output(last);
        attn_last_taps->push_back(last);
    }
    o = ggml_reshape_3d(ctx, o, hd * nh, P, B);                        // [hidden,P,B]
    if (qw_materialize_lowprec_activations() && ct != GGML_TYPE_F32) o = ggml_cast(ctx, o, ct);

    o = lin_lp(ctx, H.weight(p + "self_attn.o_proj.weight"), o, ct);
    x = ggml_add(ctx, x, o);
    if (qw_materialize_lowprec_activations() && ct != GGML_TYPE_F32) x = ggml_cast(ctx, x, ct);
    if (capture_taps && post_attn_last_taps) {
        ggml_tensor* last = ggml_view_3d(ctx, x, cfg.hidden, 1, 1,
                                          x->nb[1], x->nb[2], (size_t)trace_pos * x->nb[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        post_attn_last_taps->push_back(last);
    }
    // Transformers keeps the residual stream in BF16 during this prefix
    // forward.  ggml promotes mixed F32/BF16 adds to F32, so the explicit
    // diagnostic materialization mode must restore that boundary here.
    if (qw_materialize_lowprec_activations() && ct != GGML_TYPE_F32) x = ggml_cast(ctx, x, ct);
    ggml_tensor* hn = qw_rmsnorm(ctx, x, H.weight(p + "post_attention_layernorm.weight"), cfg.eps);
    if (capture_taps && mlp_input_last_taps) {
        ggml_tensor* last = ggml_view_3d(ctx, hn, cfg.hidden, 1, 1,
                                          hn->nb[1], hn->nb[2], (size_t)trace_pos * hn->nb[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        mlp_input_last_taps->push_back(last);
    }
    ggml_tensor* g  = lin_lp(ctx, H.weight(p + "mlp.gate_proj.weight"), hn, ct);
    ggml_tensor* u  = lin_lp(ctx, H.weight(p + "mlp.up_proj.weight"),   hn, ct);
    auto tap_intermediate_last = [&](ggml_tensor* value, std::vector<ggml_tensor*>* taps) {
        if (!capture_taps || !taps) return;
        ggml_tensor* last = ggml_view_3d(ctx, value, cfg.intermediate, 1, 1,
                                          value->nb[1], value->nb[2], (size_t)trace_pos * value->nb[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        taps->push_back(last);
    };
    tap_intermediate_last(g, gate_last_taps);
    tap_intermediate_last(u, up_last_taps);
    ggml_tensor* m  = ggml_mul(ctx, ggml_silu(ctx, g), u);
    tap_intermediate_last(m, swiglu_last_taps);
    m = lin_lp(ctx, H.weight(p + "mlp.down_proj.weight"), m, ct);
    if (capture_taps && mlp_last_taps) {
        ggml_tensor* last = ggml_view_3d(ctx, m, cfg.hidden, 1, 1,
                                          m->nb[1], m->nb[2], (size_t)trace_pos * m->nb[1]);
        if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
        ggml_set_output(last);
        mlp_last_taps->push_back(last);
    }
    x = ggml_add(ctx, x, m);
    if (qw_materialize_lowprec_activations() && ct != GGML_TYPE_F32) x = ggml_cast(ctx, x, ct);
    return x;
}

// Full causal prefix forward for B identical beam rows.  Keep only the final
// position for norm/lm_head: generation consumes only that logit column, and
// avoiding a [vocab,P,B] output removes roughly 0.7 GiB at P=514,B=10.
// Returns [vocab,B].  `writes` must be added to the graph by the caller.
static inline ggml_tensor* build_qwen3_prefill_batched(M1Harness& H, ggml_context* ctx,
                                                        const Qwen3Cfg& cfg,
                                                        Qwen3KvCacheBatched& cache,
                                                        ggml_tensor* inputs_embeds,
                                                        ggml_tensor* cos, ggml_tensor* sin,
                                                        int P, int B, ggml_tensor* mask,
                                                        std::vector<ggml_tensor*>& writes,
                                                        std::vector<ggml_tensor*>* layer_last_taps = nullptr,
                                                        std::vector<ggml_tensor*>* attn_last_taps = nullptr,
                                                        std::vector<ggml_tensor*>* post_attn_last_taps = nullptr,
                                                        std::vector<ggml_tensor*>* mlp_last_taps = nullptr,
                                                        std::vector<ggml_tensor*>* gate_last_taps = nullptr,
                                                        std::vector<ggml_tensor*>* up_last_taps = nullptr,
                                                        std::vector<ggml_tensor*>* swiglu_last_taps = nullptr,
                                                        std::vector<ggml_tensor*>* mlp_input_last_taps = nullptr) {
    const std::string m = cfg.prefix + "model.";
    // Transformers' CUDA autocast materializes inputs_embeds as BF16 before
    // the first residual add.  Keeping the source condition F32 here made
    // Q/K/V look right after RMSNorm but left every mesh-token residual
    // slightly different, which then polluted the next layer's KV cache.
    ggml_tensor* x = inputs_embeds;
    if (qw_materialize_lowprec_activations() && cfg.compute_type != GGML_TYPE_F32)
        x = ggml_cast(ctx, x, cfg.compute_type); // [hidden,P,B]
    const char * trace_pos_env = std::getenv("RIG_QWEN_LAYER_TRACE_POS");
    const int trace_pos = trace_pos_env ? std::clamp(std::atoi(trace_pos_env), 0, P - 1) : P - 1;
    for (int l = 0; l < cfg.n_layers; ++l) {
        x = qw_layer_prefill_batched(H, ctx, cfg, m + "layers." + std::to_string(l) + ".",
                                     cache, l, x, cos, sin, P, B, mask, writes, attn_last_taps,
                                     post_attn_last_taps, mlp_last_taps, gate_last_taps, up_last_taps,
                                     swiglu_last_taps, mlp_input_last_taps);
        const char * trace_layer_env = std::getenv("RIG_QWEN_LAYER_TRACE_LAYER");
        const int trace_layer = trace_layer_env ? std::atoi(trace_layer_env) : -1;
        const bool capture_layer_last = trace_layer >= 0 ? l == trace_layer :
            (!std::getenv("RIG_QWEN_LAYER_TRACE_LAYER0_ONLY") || l == 0);
        if (layer_last_taps && capture_layer_last) {
            // Lane zero at the final prefix position.  This is deliberately
            // a graph tap, not a second forward, so it observes the exact
            // activation consumed by the final norm/lm-head path.
            ggml_tensor* last = ggml_view_3d(ctx, x, cfg.hidden, 1, 1,
                                              x->nb[1], x->nb[2],
                                              (size_t)trace_pos * x->nb[1]);
            if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
            // Keep every diagnostic tap live until host readback.  Without an
            // explicit graph output gallocr may recycle the backing storage
            // for later layers, making a layer trace silently incorrect.
            ggml_set_output(last);
            layer_last_taps->push_back(last);
        }
    }
    ggml_tensor* last = ggml_view_3d(ctx, x, cfg.hidden, 1, B,
                                      x->nb[1], x->nb[2], (size_t)(P - 1) * x->nb[1]);
    last = qw_rmsnorm(ctx, last, H.weight(m + "norm.weight"), cfg.eps);
    ggml_tensor* logits = lin_lp(ctx, H.weight(cfg.prefix + "lm_head.weight"), last, cfg.compute_type);
    if (logits->type != GGML_TYPE_F32) logits = ggml_cast(ctx, logits, GGML_TYPE_F32);
    return ggml_reshape_2d(ctx, logits, cfg.vocab, B);
}

// One batched decoder layer. x [hidden, B] (B beams' current-token hidden state); cos/sin [hd,1,B]
// (all B columns = the SAME abs position P+step, replicated). Writes the fresh k/v into suffix col-slice
// at position `step` for FUTURE steps, attends over shared `pre`[0,P) ++ suffix[0,step) ++ k_new.
static inline ggml_tensor* qw_layer_batched(M1Harness& H, ggml_context* ctx, const Qwen3Cfg& cfg,
                                            const std::string& p, Qwen3KvCacheBatched& suf,
                                            Qwen3KvCache& pre, int P, int layer,
                                            ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin,
                                            int step, int B, std::vector<ggml_tensor*>& writes,
                                            std::vector<ggml_tensor*>* decode_layer0_taps = nullptr) {
    const int hd = cfg.head_dim, nh = cfg.n_heads, nkv = cfg.n_kv;
    const ggml_type ct = cfg.compute_type;
    ggml_tensor *q, *k_new, *v_new;
    qw_qkv_for_chunk(H, ctx, cfg, p, x, cos, sin, q, k_new, v_new);   // q[hd,nh,B] k/v[hd,nkv,B]
    if (layer == 0 && decode_layer0_taps) {
        ggml_tensor* tap = ggml_reshape_2d(ctx, q, hd * nh, B);
        if (tap->type != GGML_TYPE_F32) tap = ggml_cast(ctx, tap, GGML_TYPE_F32);
        ggml_set_output(tap);
        decode_layer0_taps->push_back(tap);
        tap = ggml_reshape_2d(ctx, k_new, hd * nkv, B);
        if (tap->type != GGML_TYPE_F32) tap = ggml_cast(ctx, tap, GGML_TYPE_F32);
        ggml_set_output(tap);
        decode_layer0_taps->push_back(tap);
        tap = ggml_reshape_2d(ctx, v_new, hd * nkv, B);
        if (tap->type != GGML_TYPE_F32) tap = ggml_cast(ctx, tap, GGML_TYPE_F32);
        ggml_set_output(tap);
        decode_layer0_taps->push_back(tap);
    }

    ggml_tensor* sk = suf.k_cache[layer];   // [hd, nkv, suffix_max, beam]
    ggml_tensor* sv = suf.v_cache[layer];
    if (sk->type != GGML_TYPE_F32) {
        k_new = ggml_cast(ctx, k_new, sk->type);
        v_new = ggml_cast(ctx, v_new, sv->type);
    }
    ggml_tensor* k_new4 = ggml_reshape_4d(ctx, k_new, hd, nkv, 1, B);   // [hd,nkv,1,B]
    ggml_tensor* v_new4 = ggml_reshape_4d(ctx, v_new, hd, nkv, 1, B);

    // write fresh k/v into suffix[.,.,step, 0:B] (FUTURE steps read it back from the cache).
    ggml_tensor* kw = ggml_view_4d(ctx, sk, hd, nkv, 1, B, sk->nb[1], sk->nb[2], sk->nb[3], (size_t)step * sk->nb[2]);
    ggml_tensor* vw = ggml_view_4d(ctx, sv, hd, nkv, 1, B, sv->nb[1], sv->nb[2], sv->nb[3], (size_t)step * sv->nb[2]);
    writes.push_back(ggml_cpy(ctx, k_new4, kw));
    writes.push_back(ggml_cpy(ctx, v_new4, vw));

    // ---- 2-SEGMENT batched attention: the shared prefix is read DIRECTLY from `pre` via mul_mat
    //      broadcast (beam=1 -> B, GQA nkv->nh) with NO per-beam materialization/concat; only the small
    //      suffix (past ++ new) is concatenated. Scores from both segments are concatenated (cheap:
    //      [klen,1,nh,B], not [hd,nkv,klen,B]) for a single joint softmax, then the output is the sum of
    //      the prefix and suffix value-matmuls. Eliminates the prefix repeat + big K/V concat that the
    //      nsys profile showed were ~48% of batched GPU time. (Output split is a sum of two matmuls vs
    //      one — ~1e-6 fp diff, dwarfed by the kernel-dispatch diff that already makes batched non-bit-
    //      exact; validated by distribution + logit probe.) ----
    ggml_tensor* pk = pre.k_cache[layer];   // [hd, nkv, P]
    ggml_tensor* pv = pre.v_cache[layer];
    ggml_tensor* q4 = ggml_reshape_4d(ctx, q, hd, nh, 1, B);             // [hd,nh,1,B]
    ggml_tensor* qp = ggml_permute(ctx, q4, 0, 2, 1, 3);                 // [hd,1,nh,B]
    const int klen = P + step + 1;
    // HF Qwen3 generation uses FlashAttention-2 over the complete cached
    // sequence.  Keep the shared prefix storage, but materialize it here for
    // the operation-local FA2 contract; the generic split-prefix attention
    // drifts far enough to change sampled beams on Miku.
    const bool full_fa2 = cfg.head_dim == 128 && cfg.n_heads == 16 && cfg.n_kv == 8 &&
        sk->type == GGML_TYPE_BF16;
    ggml_tensor* o = nullptr;
    if (full_fa2) {
        // Upstream cached generation invokes FA2 on one logical K/V sequence.
        // Materialize the shared prefix only in this explicit diagnostic path.
        ggml_tensor* pk4 = ggml_reshape_4d(ctx, pre.k_cache[layer], hd, nkv, P, 1);
        ggml_tensor* pv4 = ggml_reshape_4d(ctx, pre.v_cache[layer], hd, nkv, P, 1);
        ggml_tensor* pt = ggml_new_tensor_4d(ctx, sk->type, hd, nkv, P, B);
        ggml_tensor* preK = ggml_repeat(ctx, pk4, pt);
        ggml_tensor* preV = ggml_repeat(ctx, pv4, pt);
        ggml_tensor* sufK = k_new4;
        ggml_tensor* sufV = v_new4;
        if (step > 0) {
            sufK = ggml_concat(ctx, ggml_view_4d(ctx, sk, hd, nkv, step, B, sk->nb[1], sk->nb[2], sk->nb[3], 0), k_new4, 2);
            sufV = ggml_concat(ctx, ggml_view_4d(ctx, sv, hd, nkv, step, B, sv->nb[1], sv->nb[2], sv->nb[3], 0), v_new4, 2);
        }
        ggml_tensor* qf = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 2, 1, 3));
        if (qf->type != GGML_TYPE_BF16) qf = ggml_cast(ctx, qf, GGML_TYPE_BF16);
        ggml_tensor* kf = ggml_cont(ctx, ggml_permute(ctx, ggml_concat(ctx, preK, sufK, 2), 0, 2, 1, 3));
        ggml_tensor* vf = ggml_cont(ctx, ggml_permute(ctx, ggml_concat(ctx, preV, sufV, 2), 0, 2, 1, 3));
        // PyTorch flash-attn special-cases Q=1 GQA: it reshapes [1,16]
        // query heads into a two-token [2,8] non-GQA operation, then swaps
        // the two axes back.  Keep the normal path as the default; this
        // opt-in supplies an exact launch/layout parity probe.
        if (std::getenv("RIG_QWEN_FA2_GROUPED_DECODE")) {
            ggml_tensor * grouped_q = ggml_reshape_4d(ctx, qf, hd, 2, nkv, B);
            ggml_tensor * grouped_o = ggml_flash_attn_ext_qwen_causal_gqa(ctx, grouped_q, kf, vf, cfg.attn_scale());
            grouped_o = ggml_cont(ctx, ggml_permute(ctx, grouped_o, 0, 2, 1, 3));
            o = ggml_reshape_2d(ctx, grouped_o, hd * nh, B);
        } else {
            o = ggml_reshape_2d(ctx, ggml_flash_attn_ext_qwen_causal_gqa(ctx, qf, kf, vf, cfg.attn_scale()), hd * nh, B);
        }
    } else {

    // prefix scores (broadcast over beam): preK [hd,nkv,P,1] -> [hd,P,nkv,1]; kq_pre [P,1,nh,B].
    ggml_tensor* pk4 = ggml_reshape_4d(ctx, pk, hd, nkv, P, 1);
    ggml_tensor* pkp = ggml_permute(ctx, pk4, 0, 2, 1, 3);              // [hd,P,nkv,1]
    ggml_tensor* kq_pre = ggml_mul_mat(ctx, pkp, qp);                    // [P,1,nh,B]
    ggml_mul_mat_set_prec(kq_pre, GGML_PREC_F32);
    // suffix K = suffix[0,step) ++ k_new  (small; no prefix). [hd,nkv,step+1,B] -> [hd,step+1,nkv,B].
    ggml_tensor* sufK = k_new4;
    if (step > 0) {
        ggml_tensor* sk_past = ggml_view_4d(ctx, sk, hd, nkv, step, B, sk->nb[1], sk->nb[2], sk->nb[3], 0);
        sufK = ggml_concat(ctx, sk_past, k_new4, 2);
    }
    ggml_tensor* skp = ggml_permute(ctx, sufK, 0, 2, 1, 3);            // [hd,step+1,nkv,B]
    ggml_tensor* kq_suf = ggml_mul_mat(ctx, skp, qp);                    // [step+1,1,nh,B]
    ggml_mul_mat_set_prec(kq_suf, GGML_PREC_F32);
    // joint softmax over klen (scores concat is cheap relative to K/V concat).
    ggml_tensor* kq = ggml_concat(ctx, kq_pre, kq_suf, 0);             // [klen,1,nh,B]
    kq = ggml_soft_max_ext(ctx, kq, nullptr, cfg.attn_scale(), 0.0f);
    ggml_tensor* p_pre = ggml_cont(ctx, ggml_view_4d(ctx, kq, P,        1, nh, B, kq->nb[1], kq->nb[2], kq->nb[3], 0));
    ggml_tensor* p_suf = ggml_cont(ctx, ggml_view_4d(ctx, kq, step + 1, 1, nh, B, kq->nb[1], kq->nb[2], kq->nb[3], (size_t)P * kq->nb[0]));
    // prefix value-matmul (broadcast): preV [hd,nkv,P,1] -> [P,hd,nkv,1]; o_pre [hd,1,nh,B].
    ggml_tensor* pv4 = ggml_reshape_4d(ctx, pv, hd, nkv, P, 1);
    ggml_tensor* pvp = ggml_cont(ctx, ggml_permute(ctx, pv4, 1, 2, 0, 3));  // [P,hd,nkv,1]
    ggml_tensor* o_pre = ggml_mul_mat(ctx, pvp, p_pre);                  // [hd,1,nh,B]
    ggml_mul_mat_set_prec(o_pre, GGML_PREC_F32);
    // suffix value-matmul: sufV = suffix[0,step) ++ v_new -> [step+1,hd,nkv,B]; o_suf [hd,1,nh,B].
    ggml_tensor* sufV = v_new4;
    if (step > 0) {
        ggml_tensor* sv_past = ggml_view_4d(ctx, sv, hd, nkv, step, B, sv->nb[1], sv->nb[2], sv->nb[3], 0);
        sufV = ggml_concat(ctx, sv_past, v_new4, 2);
    }
    ggml_tensor* svp = ggml_cont(ctx, ggml_permute(ctx, sufV, 1, 2, 0, 3));  // [step+1,hd,nkv,B]
    ggml_tensor* o_suf = ggml_mul_mat(ctx, svp, p_suf);                  // [hd,1,nh,B]
    ggml_mul_mat_set_prec(o_suf, GGML_PREC_F32);
    ggml_tensor* kqv = ggml_add(ctx, o_pre, o_suf);                      // [hd,1,nh,B]
    o = ggml_permute(ctx, kqv, 0, 2, 1, 3);                              // [hd,nh,1,B]
    o = ggml_cont_2d(ctx, o, hd * nh, B);                                // [hidden, B]
    }
    if (qw_materialize_lowprec_activations() && ct != GGML_TYPE_F32) o = ggml_cast(ctx, o, ct);
    if (layer == 0 && decode_layer0_taps) {
        ggml_tensor* tap = o->type == GGML_TYPE_F32 ? o : ggml_cast(ctx, o, GGML_TYPE_F32);
        ggml_set_output(tap);
        decode_layer0_taps->push_back(tap);
    }
    (void)klen;

    o = lin_lp(ctx, H.weight(p + "self_attn.o_proj.weight"), o, ct);
    if (layer == 0 && decode_layer0_taps) {
        ggml_tensor* tap = o->type == GGML_TYPE_F32 ? o : ggml_cast(ctx, o, GGML_TYPE_F32);
        ggml_set_output(tap);
        decode_layer0_taps->push_back(tap);
    }
    x = ggml_add(ctx, x, o);
    // In the upstream BF16 transformer, the attention residual is materialized
    // before the following RMSNorm.  The diagnostic low-precision path must
    // preserve that boundary for cached decode too (the prefix path already
    // does), otherwise errors compound once beam tokens begin.
    if (qw_materialize_lowprec_activations() && ct != GGML_TYPE_F32) x = ggml_cast(ctx, x, ct);
    // SwiGLU MLP (identical to qw_layer_cached)
    ggml_tensor* hn = qw_rmsnorm(ctx, x, H.weight(p + "post_attention_layernorm.weight"), cfg.eps);
    ggml_tensor* g  = lin_lp(ctx, H.weight(p + "mlp.gate_proj.weight"), hn, ct);
    ggml_tensor* u  = lin_lp(ctx, H.weight(p + "mlp.up_proj.weight"),   hn, ct);
    ggml_tensor* m  = ggml_mul(ctx, ggml_silu(ctx, g), u);
    m = lin_lp(ctx, H.weight(p + "mlp.down_proj.weight"), m, ct);
    x = ggml_add(ctx, x, m);
    if (qw_materialize_lowprec_activations() && ct != GGML_TYPE_F32) x = ggml_cast(ctx, x, ct);
    if (layer == 0 && decode_layer0_taps) {
        ggml_tensor* tap = x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
        ggml_set_output(tap);
        decode_layer0_taps->push_back(tap);
    }
    return x;
}

// Batched decode of B beams' next token. inputs_embeds [hidden, B]; cos/sin [hd,1,B] (abs pos P+step,
// replicated). Returns logits [vocab, B]. `writes` collects suffix-cache writes (caller wires them).
static inline ggml_tensor* build_qwen3_batched(M1Harness& H, ggml_context* ctx, const Qwen3Cfg& cfg,
                                               Qwen3KvCacheBatched& suf, Qwen3KvCache& pre, int P,
                                               ggml_tensor* inputs_embeds, ggml_tensor* cos, ggml_tensor* sin,
                                               int step, int B, std::vector<ggml_tensor*>& writes,
                                               std::vector<ggml_tensor*>* decode_layer0_taps = nullptr) {
    const std::string m = cfg.prefix + "model.";
    ggml_tensor* x = inputs_embeds;
    for (int l = 0; l < cfg.n_layers; ++l)
        x = qw_layer_batched(H, ctx, cfg, m + "layers." + std::to_string(l) + ".",
                             suf, pre, P, l, x, cos, sin, step, B, writes, decode_layer0_taps);
    x = qw_rmsnorm(ctx, x, H.weight(m + "norm.weight"), cfg.eps);
    // Keep cached batched generation in the same compute regime as the full
    // BF16 TokenRig forward, including the output projection.
    ggml_tensor* logits = lin_lp(ctx, H.weight(cfg.prefix + "lm_head.weight"), x, cfg.compute_type);
    return logits->type == GGML_TYPE_F32 ? logits : ggml_cast(ctx, logits, GGML_TYPE_F32); // host sampler reads F32
}
