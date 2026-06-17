// qwen3_batched.hpp — BATCHED-BEAM KV-cache decode for the rig beam search. Decodes ALL active beams'
// next token in ONE forward (batch dim = beams) instead of num_beams sequential single-token decodes.
//
// WHY: nsys showed the sequential per-beam decode is launch-bound + runs the projection matmuls as
// matrix×VECTOR (M=1, bandwidth-bound) — 49% of GPU time. Batching makes them matrix×matrix (M=beams,
// tensor-core efficient) AND cuts kernel launches ~num_beams×. All active beams are always at the SAME
// absolute position (beam search appends exactly one token per beam per step), so a single batched KV
// cache [hd, nkv, suffix_max, beam] + per-beam attention over the beam dim works.
//
// SHARED PREFIX, kept low-VRAM AND bit-exact: the P-row mesh_cond prefix is stored ONCE (pre cache,
// [hd,nkv,P]). For attention it is materialized per-beam only as a graph TRANSIENT (ggml_repeat to the
// beam dim, freed by gallocr) and concatenated with each beam's suffix, so a SINGLE batched matmul
// computes attention. ggml's mul_mat GQA broadcast (i02 = i12/n_rep) is bit-identical to the explicit
// repeat_kv path, and per-beam matmul columns are independent — so this batched forward is BIT-EXACT
// to the sequential shared-prefix decode (qwen3_decode.hpp). It must reproduce the same tokens.
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

// One batched decoder layer. x [hidden, B] (B beams' current-token hidden state); cos/sin [hd,1,B]
// (all B columns = the SAME abs position P+step, replicated). Writes the fresh k/v into suffix col-slice
// at position `step` for FUTURE steps, attends over shared `pre`[0,P) ++ suffix[0,step) ++ k_new.
static inline ggml_tensor* qw_layer_batched(M1Harness& H, ggml_context* ctx, const Qwen3Cfg& cfg,
                                            const std::string& p, Qwen3KvCacheBatched& suf,
                                            Qwen3KvCache& pre, int P, int layer,
                                            ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin,
                                            int step, int B, std::vector<ggml_tensor*>& writes) {
    const int hd = cfg.head_dim, nh = cfg.n_heads, nkv = cfg.n_kv;
    ggml_tensor *q, *k_new, *v_new;
    qw_qkv_for_chunk(H, ctx, cfg, p, x, cos, sin, q, k_new, v_new);   // q[hd,nh,B] k/v[hd,nkv,B]

    ggml_tensor* sk = suf.k_cache[layer];   // [hd, nkv, suffix_max, beam]
    ggml_tensor* sv = suf.v_cache[layer];
    const bool kv_f16 = (sk->type == GGML_TYPE_F16);
    if (kv_f16) { k_new = ggml_cast(ctx, k_new, GGML_TYPE_F16); v_new = ggml_cast(ctx, v_new, GGML_TYPE_F16); }
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
    ggml_tensor* o = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [hd,nh,1,B]
    o = ggml_cont_2d(ctx, o, hd * nh, B);                                // [hidden, B]
    (void)klen;

    const ggml_type ct = cfg.compute_type;
    o = lin_lp(ctx, H.weight(p + "self_attn.o_proj.weight"), o, ct);
    x = ggml_add(ctx, x, o);
    // SwiGLU MLP (identical to qw_layer_cached)
    ggml_tensor* hn = qw_rmsnorm(ctx, x, H.weight(p + "post_attention_layernorm.weight"), cfg.eps);
    ggml_tensor* g  = lin_lp(ctx, H.weight(p + "mlp.gate_proj.weight"), hn, ct);
    ggml_tensor* u  = lin_lp(ctx, H.weight(p + "mlp.up_proj.weight"),   hn, ct);
    ggml_tensor* m  = ggml_mul(ctx, ggml_silu(ctx, g), u);
    m = lin_lp(ctx, H.weight(p + "mlp.down_proj.weight"), m, ct);
    return ggml_add(ctx, x, m);
}

// Batched decode of B beams' next token. inputs_embeds [hidden, B]; cos/sin [hd,1,B] (abs pos P+step,
// replicated). Returns logits [vocab, B]. `writes` collects suffix-cache writes (caller wires them).
static inline ggml_tensor* build_qwen3_batched(M1Harness& H, ggml_context* ctx, const Qwen3Cfg& cfg,
                                               Qwen3KvCacheBatched& suf, Qwen3KvCache& pre, int P,
                                               ggml_tensor* inputs_embeds, ggml_tensor* cos, ggml_tensor* sin,
                                               int step, int B, std::vector<ggml_tensor*>& writes) {
    const std::string m = cfg.prefix + "model.";
    ggml_tensor* x = inputs_embeds;
    for (int l = 0; l < cfg.n_layers; ++l)
        x = qw_layer_batched(H, ctx, cfg, m + "layers." + std::to_string(l) + ".",
                             suf, pre, P, l, x, cos, sin, step, B, writes);
    x = qw_rmsnorm(ctx, x, H.weight(m + "norm.weight"), cfg.eps);
    return lin(ctx, H.weight(cfg.prefix + "lm_head.weight"), nullptr, x);   // [vocab, B]
}
