// qwen3_decode.hpp — KV-cache incremental-decode path for the VALIDATED Qwen3-0.6B forward.
//
// Sits ON TOP of qwen3_forward.hpp (DO NOT modify it). REUSES its weight keys + per-layer math
// (qw_rmsnorm, qw_rope, qw_repeat_kv) so the cached path stays bit-consistent with the full
// teacher-forced forward (qwen3_r2_test). Mirrors llama.cpp's KV cache:
//
//   - A dedicated PERSISTENT backend buffer (its own ggml_context, NOT gallocr-managed) holds,
//     per layer, K and V tensors of shape [head_dim, n_kv, max_seq]:
//        K = post qk-norm, post-RoPE  (RoPE applied per-position AT WRITE TIME so cached keys
//            already carry their positional rotation — decode_step never re-rotates the cache)
//        V = raw (post v_proj, no norm/rope), n_kv heads (GQA repeat happens at read time)
//   - prefill(P tokens): runs the prefix through all layers, WRITES each layer's K,V into the
//     cache at positions [0,P) via ggml_view + ggml_cpy nodes (part of the computed graph), and
//     returns the LAST-position logits.  cache.pos = P.
//   - decode_step(1 token at position `pos`): computes the new token's q/k/v, qk-norm, RoPE for
//     that single position, WRITES its k,v into the cache slice at index `pos`, then the single
//     query attends over cache K/V[0..pos] (all past positions visible -> NO mask needed; the
//     query is always at the newest position).  cache.pos = pos+1.
//
// Per-layer compute is FACTORED (qkv_for_chunk + attn_over_cache) so prefill and decode_step run
// the SAME math, differing only in chunk length L (P vs 1) and the cache write offset.
//
// All fp32 to match the validated path. lin() forces GGML_PREC_F32 (m1_ggml.hpp), so cached-K
// is bit-identical to the full-forward K at each position; the only numerical difference vs the
// full forward is that attention sees keys/values stored-then-reloaded (identical bytes) — so
// logits should match to fp32 round-off (essentially exact).
#pragma once
#include "qwen3_forward.hpp"
#include "m1_ggml.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

struct Qwen3KvCache {
    // Persistent buffer (own ctx, like ctx_w) so gallocr never reuses the cache memory between
    // graph computes.  k_cache[l], v_cache[l] : [head_dim, n_kv, max_seq].
    ggml_context*   ctx   = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k_cache;   // per layer, [hd, n_kv, max_seq]  (post qk-norm + RoPE)
    std::vector<ggml_tensor*> v_cache;   // per layer, [hd, n_kv, max_seq]  (raw v)
    int max_seq = 0;
    int pos     = 0;     // number of positions currently written (== next write index)

    // Allocate the cache tensors in a dedicated persistent backend buffer. Call AFTER the harness
    // backend exists; sizes for `max_seq` positions (e.g. 1152 >= S=1079).
    // Keep F16 as the production default until BF16 cache parity is proven.
    // RIG_KV_TYPE=bf16 is a diagnostic mode for models whose native cache is
    // BF16; the CUDA repeat/concat paths validate that mode explicitly.
    static ggml_type kv_dtype() {
        const char* t = std::getenv("RIG_KV_TYPE");
        if (t) {
            if (std::strcmp(t, "bf16") == 0) return GGML_TYPE_BF16;
            if (std::strcmp(t, "f16") == 0) return GGML_TYPE_F16;
            if (std::strcmp(t, "f32") == 0) return GGML_TYPE_F32;
            throw std::runtime_error("RIG_KV_TYPE must be bf16, f16, or f32");
        }
        const char* e = std::getenv("RIG_KV_F16");
        if (e && e[0] == '0') return GGML_TYPE_F32;
        return GGML_TYPE_F16;
    }
    void init(M1Harness& H, const Qwen3Cfg& cfg, int max_seq_) {
        max_seq = max_seq_;
        pos = 0;
        // metadata ctx: 2 tensors/layer + overhead.
        size_t meta = (size_t)(cfg.n_layers * 2 + 8) * ggml_tensor_overhead();
        ggml_init_params ip{ meta, nullptr, /*no_alloc=*/true };
        ctx = ggml_init(ip);
        k_cache.resize(cfg.n_layers);
        v_cache.resize(cfg.n_layers);
        const ggml_type kvt = kv_dtype();
        for (int l = 0; l < cfg.n_layers; ++l) {
            k_cache[l] = ggml_new_tensor_3d(ctx, kvt, cfg.head_dim, cfg.n_kv, max_seq);
            v_cache[l] = ggml_new_tensor_3d(ctx, kvt, cfg.head_dim, cfg.n_kv, max_seq);
            ggml_set_name(k_cache[l], ("kv.k." + std::to_string(l)).c_str());
            ggml_set_name(v_cache[l], ("kv.v." + std::to_string(l)).c_str());
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, H.backend);
        if (!buf) throw std::runtime_error("Qwen3KvCache: alloc_ctx_tensors failed");
    }
    void reset() { pos = 0; }
    ~Qwen3KvCache() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

// Allocate + upload all registered weights ONCE, converting F32 npy -> F16 for any weight tensor that
// weight() created as F16 (H.w_f16). Replaces the inline upload blocks in the beam decoders so the f16
// weight path is consistent everywhere. (GGUF-backed uploads are passed through untouched.)
static inline void upload_weights_maybe_f16(M1Harness& H) {
    H.wbuf = ggml_backend_alloc_ctx_tensors(H.ctx_w, H.backend);
    for (auto& gu : H.gguf_uploads)
        ggml_backend_tensor_set(gu.first, gu.second, 0, ggml_nbytes(gu.first));
    for (auto& u : H.uploads) {
        NpyArray a = npy_load(u.second);
        if (u.first->type == GGML_TYPE_F16) {
            int64_t n = a.numel();
            std::vector<ggml_fp16_t> h((size_t)n);
            ggml_fp32_to_fp16_row(a.f32(), h.data(), n);
            ggml_backend_tensor_set(u.first, h.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(u.first, a.raw.data(), 0, (size_t)a.numel() * 4);
        }
    }
}

// ----------------------------------------------------------------------------
// shared per-layer compute (factored out of qw_layer, same math)
// ----------------------------------------------------------------------------

// Compute the new chunk's q (post qk-norm + RoPE), k (post qk-norm + RoPE), v (raw) for `x`
// [hidden, L].  cos/sin [hd, 1, L] = RoPE tables for the chunk's positions [write0, write0+L).
// Returns by out-params. Mirrors qw_layer lines up to (but not incl.) repeat_kv.
static inline void qw_qkv_for_chunk(M1Harness& H, ggml_context* ctx, const Qwen3Cfg& cfg,
                                    const std::string& p, ggml_tensor* x,
                                    ggml_tensor* cos, ggml_tensor* sin,
                                    ggml_tensor*& q_out, ggml_tensor*& k_out, ggml_tensor*& v_out,
                                    Qwen3Subtrace* trace = nullptr, int layer = -1) {
    const int hd = cfg.head_dim, nh = cfg.n_heads, nkv = cfg.n_kv;
    const std::string tn = trace && trace->enabled()
        ? std::string("layer_") + (layer < 10 ? "0" : "") + std::to_string(layer) + "." : "";
    ggml_tensor* h = qw_rmsnorm(ctx, x, H.weight(p + "input_layernorm.weight"), cfg.eps);
    if (trace) trace->add_last_2d(ctx, tn + "input_rmsnorm", h);
    const ggml_type ct = cfg.compute_type;
    ggml_tensor* q = lin_lp(ctx, H.weight(p + "self_attn.q_proj.weight"), h, ct);  // [nh*hd, L]
    ggml_tensor* k = lin_lp(ctx, H.weight(p + "self_attn.k_proj.weight"), h, ct);  // [nkv*hd, L]
    ggml_tensor* v = lin_lp(ctx, H.weight(p + "self_attn.v_proj.weight"), h, ct);  // [nkv*hd, L]
    q = ggml_reshape_3d(ctx, q, hd, nh,  q->ne[1]);
    k = ggml_reshape_3d(ctx, k, hd, nkv, k->ne[1]);
    v = ggml_reshape_3d(ctx, v, hd, nkv, v->ne[1]);
    if (trace) {
        trace->add_last_heads(ctx, tn + "q_proj", q);
        trace->add_last_heads(ctx, tn + "k_proj", k);
        trace->add_last_heads(ctx, tn + "v_proj", v);
        if (layer == 0 && std::getenv("RIG_QWEN3_SUBTRACE_FULL"))
            trace->add(ctx, tn + "v_proj_full", v, {v->ne[2], v->ne[1], v->ne[0]});
    }
    q = qw_rmsnorm(ctx, q, H.weight(p + "self_attn.q_norm.weight"), cfg.eps);
    k = qw_rmsnorm(ctx, k, H.weight(p + "self_attn.k_norm.weight"), cfg.eps);
    if (trace) {
        trace->add_last_heads(ctx, tn + "q_norm", q);
        trace->add_last_heads(ctx, tn + "k_norm", k);
    }
    q = qw_rope(ctx, q, cos, sin);
    k = qw_rope(ctx, k, cos, sin);
    if (trace) {
        trace->add_last_heads(ctx, tn + "q_post_rope", q);
        trace->add_last_heads(ctx, tn + "k_post_rope", k);
        // Full layer-0 prefix dump is a deliberately opt-in attention
        // forensic: the final-token taps prove RoPE for one position, while
        // attention consumes all 514 keys.  ggml's [d, heads, tokens] memory
        // is NumPy C-order [tokens, heads, d].
        if (layer == 0 && std::getenv("RIG_QWEN3_SUBTRACE_FULL")) {
            trace->add(ctx, tn + "q_post_rope_full", q, {q->ne[2], q->ne[1], q->ne[0]});
            trace->add(ctx, tn + "k_post_rope_full", k, {k->ne[2], k->ne[1], k->ne[0]});
        }
    }
    q_out = q;                         // [hd, nh,  L]
    k_out = ggml_cont(ctx, k);         // [hd, nkv, L]  (cont so the cpy into cache is clean)
    v_out = ggml_cont(ctx, v);         // [hd, nkv, L]
}

// Write new k,v [hd, nkv, L] into the cache at position `write0`, then return the attention output
// [hidden_q = hd*nh, L] of q [hd, nh, L] attending over cache K/V[0..write0+L).
//   mask : optional additive [k_len, L] (0/-inf). null for the single-query decode (all visible).
// The ggml_cpy(new -> cache_view) nodes are returned via `writes` so the caller wires them into the
// graph (they must be computed; attention reads the cache slice which spans the freshly-written region).
static inline ggml_tensor* qw_attn_over_cache(ggml_context* ctx, const Qwen3Cfg& cfg,
                                              Qwen3KvCache& cache, int layer, int write0, int L,
                                              ggml_tensor* q, ggml_tensor* k_new, ggml_tensor* v_new,
                                              ggml_tensor* mask,
                                              std::vector<ggml_tensor*>& writes,
                                              Qwen3KvCache* pre = nullptr, int pre_len = 0,
                                              Qwen3Subtrace* trace = nullptr) {
    const int hd = cfg.head_dim, nkv = cfg.n_kv;
    ggml_tensor* kc = cache.k_cache[layer];   // [hd, nkv, max_seq]  (the beam's SUFFIX cache when pre!=0)
    ggml_tensor* vc = cache.v_cache[layer];
    // DynamicCache adopts post-RoPE K's dtype.  An F32 prefix therefore also
    // promotes BF16 V on the same cache update; use the cache representation
    // for both this attention and every later decode step.
    if (k_new->type != kc->type) k_new = ggml_cast(ctx, k_new, kc->type);
    if (v_new->type != vc->type) v_new = ggml_cast(ctx, v_new, vc->type);
    // cache write slice [hd, nkv, L] at ne2 offset write0 — for FUTURE steps only. write0 is a SUFFIX
    // offset when a shared read-only prefix cache `pre` is supplied (the n_cond mesh_cond prefix is
    // identical across all beams, so it is held ONCE in `pre` and never copied into the per-beam cache).
    ggml_tensor* kview = ggml_view_3d(ctx, kc, hd, nkv, L, kc->nb[1], kc->nb[2], (size_t)write0 * kc->nb[2]);
    ggml_tensor* vview = ggml_view_3d(ctx, vc, hd, nkv, L, vc->nb[1], vc->nb[2], (size_t)write0 * vc->nb[2]);
    writes.push_back(ggml_cpy(ctx, k_new, kview));
    writes.push_back(ggml_cpy(ctx, v_new, vview));
    // Attention K/V = concat(SHARED prefix pre[0..pre_len), PAST suffix cache[0..write0), NEW k/v). All
    // past regions were written by PRIOR (completed) computes; the new region comes straight from
    // k_new/v_new. We never read the just-written suffix slice back IN THIS GRAPH (ggml tracks deps via
    // src links, not memory aliasing → in-graph read-after-write on the same buffer is a hazard); the
    // shared prefix lives in a SEPARATE buffer and is read-only, so reading it here is always safe.
    const int klen = (pre ? pre_len : 0) + write0 + L;
    std::vector<ggml_tensor*> kparts, vparts;
    if (pre) {
        ggml_tensor* kpc = pre->k_cache[layer];
        ggml_tensor* vpc = pre->v_cache[layer];
        kparts.push_back(ggml_view_3d(ctx, kpc, hd, nkv, pre_len, kpc->nb[1], kpc->nb[2], 0));
        vparts.push_back(ggml_view_3d(ctx, vpc, hd, nkv, pre_len, vpc->nb[1], vpc->nb[2], 0));
    }
    if (write0 > 0) {
        kparts.push_back(ggml_view_3d(ctx, kc, hd, nkv, write0, kc->nb[1], kc->nb[2], 0));
        vparts.push_back(ggml_view_3d(ctx, vc, hd, nkv, write0, vc->nb[1], vc->nb[2], 0));
    }
    kparts.push_back(k_new);   // newest position(s) last
    vparts.push_back(v_new);
    ggml_tensor* kslice = kparts[0];
    ggml_tensor* vslice = vparts[0];
    for (size_t i = 1; i < kparts.size(); ++i) {   // chain the (1..3)-way concat along the position dim
        kslice = ggml_concat(ctx, kslice, kparts[i], 2);   // [hd, nkv, klen]
        vslice = ggml_concat(ctx, vslice, vparts[i], 2);
    }
    // Exact-HF attention for the shared B=1 causal prefix.  FA2 consumes
    // BF16 Q/K/V even when the retained prefix cache uses a different
    // precision.  The source-built bridge receives an operation-local
    // causal-Qwen contract; no ordinary null-mask attention can enter this
    // path.
    if (!pre && write0 == 0 && L > 1 &&
        cfg.head_dim == 128 && cfg.n_heads == 16 && cfg.n_kv == 8 &&
        (kslice->type == GGML_TYPE_F32 || kslice->type == GGML_TYPE_BF16) &&
        (vslice->type == GGML_TYPE_F32 || vslice->type == GGML_TYPE_BF16)) {
        ggml_tensor* qf = ggml_cast(ctx, ggml_cont(ctx, ggml_permute(ctx, q,      0, 2, 1, 3)), GGML_TYPE_BF16);
        ggml_tensor* kf = ggml_cast(ctx, ggml_cont(ctx, ggml_permute(ctx, kslice, 0, 2, 1, 3)), GGML_TYPE_BF16);
        ggml_tensor* vf = ggml_cast(ctx, ggml_cont(ctx, ggml_permute(ctx, vslice, 0, 2, 1, 3)), GGML_TYPE_BF16);
        ggml_tensor* fa = ggml_flash_attn_ext_qwen_causal_gqa(ctx, qf, kf, vf, cfg.attn_scale());
        return ggml_reshape_2d(ctx, fa, hd * cfg.n_heads, L);
    }
    // GQA repeat to n_heads (mirror qw_layer: cont both).
    ggml_tensor* kr = ggml_cont(ctx, qw_repeat_kv(ctx, kslice, cfg.n_rep()));  // [hd, nh, klen]
    ggml_tensor* vr = ggml_cont(ctx, qw_repeat_kv(ctx, vslice, cfg.n_rep()));  // [hd, nh, klen]
    // The Tira oracle's F32 mesh-condition prefix promotes post-RoPE Q/K and
    // the DynamicCache to F32, but HF's FlashAttention-2 wrapper explicitly
    // casts all three operands back to BF16 immediately before the kernel.
    // Keep the cache F32 (it is observed state), perform GQA repeat while it
    // is F32 (ggml CUDA does not support BF16 repeat), then materialize only
    // the kernel operands in BF16.  The attention matmuls still accumulate
    // into F32 as ggml requires.
    if (qw_materialize_lowprec_activations() && cfg.compute_type == GGML_TYPE_BF16) {
        q  = ggml_cast(ctx, q,  GGML_TYPE_BF16);
        kr = ggml_cast(ctx, kr, GGML_TYPE_BF16);
        vr = ggml_cast(ctx, vr, GGML_TYPE_BF16);
    }
    // attention: q [hd, nh, L] over k/v [hd, nh, klen]. Reuse the exact math of qw_causal_attn but
    // with an asymmetric (klen x L) mask (the helper assumes square S x S). Inline it here.
    int64_t d = q->ne[0], nhh = q->ne[1];
    ggml_tensor* kp = ggml_permute(ctx, kr, 0, 2, 1, 3);                  // [d, klen, head]
    ggml_tensor* qp = ggml_permute(ctx, q,  0, 2, 1, 3);                  // [d, L,    head]
    ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, vr, 1, 2, 0, 3)); // [klen, d, head]
    ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);                         // [klen, L, head]
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    if (trace && trace->enabled() && layer == 0)
        trace->add_final_query_scores(ctx, "layer_00.attn_qk_raw", kq);
    kq = ggml_soft_max_ext(ctx, kq, mask, cfg.attn_scale(), 0.0f);       // [klen, L, head]
    if (trace && trace->enabled() && layer == 0)
        trace->add_final_query_scores(ctx, "layer_00.attn_probs", kq);
    ggml_tensor* kqv = ggml_mul_mat(ctx, vp, kq);                        // [d, L, head]
    ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                            // [d, head, L]
    return ggml_cont_2d(ctx, kqv, d * nhh, L);                           // [hidden_q, L]
}

// one cache-backed decoder layer (prefill OR step). x [hidden, L]; cos/sin [hd,1,L] for the chunk's
// positions; write0 = first absolute position of the chunk; mask = additive [klen, L] (or null when
// L==1, all-visible). Appends the two cache-write cpy nodes to `writes`. Returns x' [hidden, L].
static inline ggml_tensor* qw_layer_cached(M1Harness& H, ggml_context* ctx, const Qwen3Cfg& cfg,
                                           const std::string& p, Qwen3KvCache& cache, int layer,
                                           ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin,
                                           int write0, int L, ggml_tensor* mask,
                                           std::vector<ggml_tensor*>& writes,
                                           Qwen3KvCache* pre = nullptr, int pre_len = 0,
                                           Qwen3Subtrace* trace = nullptr) {
    ggml_tensor *q, *k, *v;
    qw_qkv_for_chunk(H, ctx, cfg, p, x, cos, sin, q, k, v, trace, layer);
    ggml_tensor* o = qw_attn_over_cache(ctx, cfg, cache, layer, write0, L, q, k, v, mask, writes, pre, pre_len, trace);
    const std::string tn = trace && trace->enabled()
        ? std::string("layer_") + (layer < 10 ? "0" : "") + std::to_string(layer) + "." : "";
    if (trace) trace->add_last_2d(ctx, tn + "attn_pre_o", o);
    const ggml_type ct = cfg.compute_type;
    o = lin_lp(ctx, H.weight(p + "self_attn.o_proj.weight"), o, ct);
    if (trace) trace->add_last_2d(ctx, tn + "o_proj", o);
    x = ggml_add(ctx, x, o);
    if (trace) trace->add_last_2d(ctx, tn + "post_attn_residual", x);
    // SwiGLU MLP (identical to qw_layer)
    ggml_tensor* hn = qw_rmsnorm(ctx, x, H.weight(p + "post_attention_layernorm.weight"), cfg.eps);
    ggml_tensor* g  = lin_lp(ctx, H.weight(p + "mlp.gate_proj.weight"), hn, ct);
    ggml_tensor* u  = lin_lp(ctx, H.weight(p + "mlp.up_proj.weight"),   hn, ct);
    ggml_tensor* m  = ggml_mul(ctx, ggml_silu(ctx, g), u);
    if (trace) {
        trace->add_last_2d(ctx, tn + "post_attn_rmsnorm", hn);
        trace->add_last_2d(ctx, tn + "gate_proj", g);
        trace->add_last_2d(ctx, tn + "up_proj", u);
        trace->add_last_2d(ctx, tn + "mlp_pre_down", m);
    }
    m = lin_lp(ctx, H.weight(p + "mlp.down_proj.weight"), m, ct);
    if (trace) trace->add_last_2d(ctx, tn + "down_proj", m);
    x = ggml_add(ctx, x, m);
    if (trace) trace->add_last_2d(ctx, tn + "layer_output", x);
    return x;
}

// Build the cache-backed stack for a chunk of L tokens starting at absolute position write0.
// Returns logits [vocab, L]. `writes` collects all cache-write cpy nodes (caller must add them to
// the graph so they execute). mask = additive [klen, L] for prefill (causal), null for L==1 step.
static inline ggml_tensor* build_qwen3_cached(M1Harness& H, ggml_context* ctx, const Qwen3Cfg& cfg,
                                              Qwen3KvCache& cache, ggml_tensor* inputs_embeds,
                                              ggml_tensor* cos, ggml_tensor* sin,
                                              int write0, int L, ggml_tensor* mask,
                                              std::vector<ggml_tensor*>& writes,
                                              Qwen3KvCache* pre = nullptr, int pre_len = 0,
                                              std::vector<ggml_tensor*>* layer_last_taps = nullptr,
                                              Qwen3Subtrace* trace = nullptr) {
    const std::string m = cfg.prefix + "model.";
    ggml_tensor* x = inputs_embeds;
    for (int l = 0; l < cfg.n_layers; ++l) {
        x = qw_layer_cached(H, ctx, cfg, m + "layers." + std::to_string(l) + ".",
                            cache, l, x, cos, sin, write0, L, mask, writes, pre, pre_len, trace);
        if (layer_last_taps) {
            // Diagnostic only: retain the final token's post-residual activation
            // from each actual shared-B1 prefill layer.  The view is F32 unless
            // full low-precision activation materialization is explicitly enabled.
            ggml_tensor* last = ggml_view_2d(ctx, x, cfg.hidden, 1, x->nb[1],
                                              (size_t)(L - 1) * x->nb[1]);
            if (last->type != GGML_TYPE_F32) last = ggml_cast(ctx, last, GGML_TYPE_F32);
            layer_last_taps->push_back(last);
        }
    }
    x = qw_rmsnorm(ctx, x, H.weight(m + "norm.weight"), cfg.eps);
    // Generation is BF16 under TokenRig's CUDA autocast.  Match the full
    // forward path: leaving only lm_head in F32 changes the sampled race.
    ggml_tensor* logits = lin_lp(ctx, H.weight(cfg.prefix + "lm_head.weight"), x, cfg.compute_type);
    return logits->type == GGML_TYPE_F32 ? logits : ggml_cast(ctx, logits, GGML_TYPE_F32); // host sampler reads F32
}
