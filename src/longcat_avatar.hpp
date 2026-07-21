#ifndef __LONGCAT_AVATAR_HPP__
#define __LONGCAT_AVATAR_HPP__

#include "core/ggml_extend.hpp"
#include "model.h"
#include "model/common/rope.hpp"
#include "model/diffusion/model.hpp"  // DiffusionModelRunner + LongCatAvatarDiffusionExtra (sd.cpp #1569)

// LongCat-Video-Avatar 1.5 DiT port (mirrors src/wan.hpp).
//
// The runner includes the Avatar DiT's text/audio conditioning, AudioProjModel,
// continuation-aware RoPE, and cache paths. Numerical/GPU validation remains
// separate from this source integration.
namespace LONGCAT_AVATAR {

    // Per-block single-stream block:
    //   x = x + gate_msa * attn(modulate(mod_norm_attn(x), shift_msa, scale_msa))
    //   x = x + cross_attn(pre_crs_attn_norm(x), y)                  (text, UNGATED)
    //   x = x + audio_gate * audio_cross_attn(...)
    //   x = x + gate_mlp * ffn(modulate(mod_norm_ffn(x), shift_mlp, scale_mlp))
    //
    // mod_norm_attn / mod_norm_ffn : LayerNorm fp32, NO affine.
    // pre_crs_attn_norm / pre_video_crs_attn_norm : affine LayerNorm fp32.
    // self-attn : fused qkv (bias), q_norm/k_norm RMSNorm over head_dim, 3D RoPE.
    // text/audio cross_attn : q_linear + kv_linear(2C) + proj, q_norm/k_norm.
    class LongCatAvatarSingleStreamBlock : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t num_heads;
        int64_t head_dim;
        int64_t ffn_inner;
        float eps;

    public:
        // >1 tiles the SwiGLU FFN over token blocks to bound its activation peak
        // at full length (set by the runner from LONGCAT_FFN_TILES; 1 = original
        // single-shot path, used for the 25f hot path so it stays bit-identical).
        int64_t ffn_token_tiles = 1;
        // >1 tiles the self-attn noise pass over query-row blocks (same idea, for the
        // self-attention working set). 1 = original single-shot call (25f hot path).
        int64_t attn_query_tiles = 1;
        // RUNTIME mouth-exaggeration knob: scales the audio cross-attn gated residual
        // (the SINGLE additive term that drives the face from audio — see audio path
        // below). 1.0 = unchanged (default, bit-identical). <1 = milder mouth, >1 =
        // more exaggerated. Set per-render by the runner from --audio-mouth-scale /
        // the API. Timing is untouched (lives in WHICH tokens attend, not magnitude).
        float audio_mouth_scale = 1.0f;

        // Video-continuation (generate_avc) 3-way self-attn split params. Set per-render
        // by the runner via the avatar forward loop. num_ref_latents>0 enables the
        // ref/cond/noise split; 0 = the ai2v 2-way cond/noise split (single-clip path,
        // bit-identical). n_per_frame maps the mask_frame_range carve-out (in frames)
        // to token offsets.
        int     num_ref_latents  = 0;
        int     ref_img_index    = 10;
        int     mask_frame_range = 3;
        int64_t n_per_frame      = 0;

        LongCatAvatarSingleStreamBlock(int64_t hidden_size,
                                       int64_t num_heads,
                                       int64_t ffn_inner,
                                       int64_t adaln_tembed_dim,
                                       int64_t caption_dim,
                                       int64_t audio_dim,
                                       float eps = 1e-6f)
            : hidden_size(hidden_size),
              num_heads(num_heads),
              ffn_inner(ffn_inner),
              eps(eps) {
            head_dim = hidden_size / num_heads;

            // adaLN modulations: SiLU + Linear(adaln_tembed_dim, k*hidden)
            // NUMERICS: the reference runs the whole avatar DiT body in bf16 with
            // the modulation/LayerNorm/gate-add paths in fp32 (amp.autocast +
            // LayerNorm_FP32 + modulate_fp32). bf16 has the F32 exponent range, so
            // the (large, additively-growing) residual stream never overflows. In
            // ggml the heavy matmuls keep F32 activations, so we force F32 matmul
            // precision (GGML_PREC_F32) on every Linear for parity with bf16's
            // accumulation. The one place this is NOT enough is ffn.w2 (Q8_1
            // activation-quant overflow) — see the ffn.w2 note below.
            const bool pf32 = true;
            blocks["adaLN_modulation.1"]       = std::shared_ptr<GGMLBlock>(new Linear(adaln_tembed_dim, 6 * hidden_size, true, false, pf32));
            blocks["audio_adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(adaln_tembed_dim, 3 * hidden_size, true, false, pf32));

            // non-affine LayerNorms reused for modulation
            blocks["mod_norm_attn"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps, false));
            blocks["mod_norm_ffn"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps, false));

            // affine LayerNorms (weights present in ckpt)
            blocks["pre_crs_attn_norm"]       = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps, true, true));
            blocks["pre_video_crs_attn_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps, true, true));

            // self-attention (fused qkv)
            blocks["attn.qkv"]    = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 3 * hidden_size, true, false, pf32));
            blocks["attn.q_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, eps));
            blocks["attn.k_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, eps));
            blocks["attn.proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, false, pf32));

            // text cross-attention
            blocks["cross_attn.q_linear"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, false, pf32));
            blocks["cross_attn.kv_linear"] = std::shared_ptr<GGMLBlock>(new Linear(caption_dim, 2 * hidden_size, true, false, pf32));
            blocks["cross_attn.proj"]      = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, false, pf32));
            blocks["cross_attn.q_norm"]    = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, eps));
            blocks["cross_attn.k_norm"]    = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, eps));

            // audio cross-attention
            blocks["audio_cross_attn.q_linear"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, false, pf32));
            blocks["audio_cross_attn.kv_linear"] = std::shared_ptr<GGMLBlock>(new Linear(audio_dim, 2 * hidden_size, true, false, pf32));
            blocks["audio_cross_attn.proj"]      = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, false, pf32));
            blocks["audio_cross_attn.q_norm"]    = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, eps));
            blocks["audio_cross_attn.k_norm"]    = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, eps));

            // SwiGLU ffn: w2(silu(w1(x)) * w3(x)), all bias-free.
            // ffn.w2's input is silu(w1)*w3, which is UNBOUNDED — the SwiGLU
            // product reaches O(1e5) (the modulation scale_mlp is genuinely large
            // for this DMD model at t=1000, ~mean 9, so the renormed FFN branch is
            // amplified). For a Q4_K weight the ggml-cuda MMQ path quantizes that
            // activation to Q8_1, whose per-block sum field (sum_q * d) is stored in
            // F16; at activation ~1e5 that field overflows F16 (>65504) -> inf ->
            // NaN. The reference runs this matmul in bf16 (full F32 exponent range),
            // so there is no overflow upstream. We reproduce the safe range by
            // pre-scaling w2's input down (and the output back up) by a constant —
            // ggml_ext_linear does x*=s before the matmul and out*=1/s after, so the
            // result is mathematically identical but the Q8_1 sum field stays in
            // F16 range. s = 1/256 keeps it well below 65504 across all 48 blocks
            // (the FFN-branch magnitude does NOT grow with block depth — it is gated
            // by scale_mlp/the SwiGLU, independent of the additive residual stream).
            const float w2_in_scale = 1.0f / 256.0f;
            blocks["ffn.w1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, ffn_inner, false, false, pf32));
            blocks["ffn.w2"] = std::shared_ptr<GGMLBlock>(new Linear(ffn_inner, hidden_size, false, false, pf32, w2_in_scale));
            blocks["ffn.w3"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, ffn_inner, false, false, pf32));
        }

        // modulate(norm(x), shift, scale) with per-frame (B,T) modulation broadcast
        // over the N//T spatial tokens. x: [N, n_token, C]; mod: [N, T, 1, C].
        ggml_tensor* modulate(GGMLRunnerContext* ctx,
                              std::shared_ptr<LayerNorm> norm,
                              ggml_tensor* x,
                              ggml_tensor* shift,
                              ggml_tensor* scale,
                              int64_t T) {
            // scale+1 then *x then + shift  (broadcast over spatial dim).
            // (scale + 1) is a fused ggml_scale_bias (s=1, b=1) — one elementwise op
            // instead of materializing a full ones tensor (SCALE+REPEAT) and adding it.
            //
            // LongCat lap-27: build scale1 BEFORE the norm/mul/add chain and pre-expand
            // it onto the cgraph so it lands earlier in the post-order topology than
            // the chain. Otherwise scale1's SCALE node would be inserted between NORM
            // and MUL when the main forward is expanded (since MUL.src[0]=NORM_out is
            // DFS'd first), breaking the {NORM, MUL, ADD} autofusion's adjacency check.
            // Pre-expanded ⇒ NORM, MUL, ADD become consecutive ⇒ the auto-fused
            // norm_f32<…,true,true> kernel collapses 3 HBM-bouncing kernels into one.
            auto scale1 = ggml_scale_bias(ctx->ggml_ctx, scale, 1.0f, 1.0f);
            if (ctx->gf != nullptr) {
                // LongCat lap-28: also pre-expand `shift`'s view/CONT chain (the chunk-
                // produced bias). Without this, shift's VIEW+CONT nodes land *between*
                // MUL and ADD in topo order (post-order DFS from ADD finds shift's
                // unbuilt srcs after MUL is already built), which the autofusion's
                // after-MUL view-skip cannot walk past (CONT is not view-like). Pre-
                // expanding pushes them earlier ⇒ ADD lands directly after MUL.
                ggml_build_forward_expand(ctx->gf, scale1);
                ggml_build_forward_expand(ctx->gf, shift);
            }
            x          = norm->forward(ctx, x);
            int64_t Nb = x->ne[2];
            int64_t C  = x->ne[0];
            x          = ggml_reshape_4d(ctx->ggml_ctx, x, C, x->ne[1] / T, T, Nb);  // [N, T, n_token/T, C]
            x           = ggml_mul(ctx->ggml_ctx, x, scale1);
            x           = ggml_add(ctx->ggml_ctx, x, shift);
            x           = ggml_reshape_3d(ctx->ggml_ctx, x, C, x->ne[1] * x->ne[2], Nb);  // [N, n_token, C]
            return x;
        }

        // gate * y, with gate per-frame (B,T,1,C) broadcast over spatial tokens.
        ggml_tensor* gate_add(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* y,
                             ggml_tensor* gate,
                             int64_t T) {
            int64_t Nb = y->ne[2];
            int64_t C  = y->ne[0];
            y          = ggml_reshape_4d(ctx->ggml_ctx, y, C, y->ne[1] / T, T, Nb);  // [N, T, n_token/T, C]
            y          = ggml_mul(ctx->ggml_ctx, y, gate);
            y          = ggml_reshape_3d(ctx->ggml_ctx, y, C, y->ne[1] * y->ne[2], Nb);  // [N, n_token, C]
            return ggml_add(ctx->ggml_ctx, x, y);
        }

        // gate * y (no residual add), gate per-frame (B,T,1,C) broadcast over
        // spatial tokens. Equivalent to gate_add(zeros_like(y), y, gate, T) but
        // skips materializing a full-size zero tensor + its add (the audio path's
        // contribution is prepended with cond zeros separately).
        ggml_tensor* gate_mul(GGMLRunnerContext* ctx,
                              ggml_tensor* y,
                              ggml_tensor* gate,
                              int64_t T) {
            int64_t Nb = y->ne[2];
            int64_t C  = y->ne[0];
            y          = ggml_reshape_4d(ctx->ggml_ctx, y, C, y->ne[1] / T, T, Nb);  // [N, T, n_token/T, C]
            y          = ggml_mul(ctx->ggml_ctx, y, gate);
            y          = ggml_reshape_3d(ctx->ggml_ctx, y, C, y->ne[1] * y->ne[2], Nb);  // [N, n_token, C]
            return y;
        }

        // self attention with fused qkv + qk RMSNorm + 3D RoPE.
        // n_cond_tokens (avatar num_cond_latents split, avatar/attention.py
        // Attention.forward with num_cond_latents==1): the first n_cond_tokens (the
        // ref-image cond latent frame) attend ONLY to themselves; the generated/noise
        // tokens attend to everything. The reference does this as TWO attention calls
        // over a sequence-dim split (q_cond×{k,v}_cond and q_noise×{k,v}_full) then
        // concats. We reproduce that two-pass split exactly — no O(n_token^2) mask, so
        // it works at 480p (~10920 tokens) within the VRAM budget AND keeps flash-attn.
        // n_cond_tokens==0 → plain full self-attention.
        // Helper: flash/non-flash attention over a q row-slice [start, start+len) on the
        // token dim of q_rope, against the given (already cond-style or full) k/v.
        // q_rope is [d_head, n_token, num_heads*N]; v is [d_head, num_heads, n_token, N].
        ggml_tensor* attn_qslice(GGMLRunnerContext* ctx, ggml_tensor* q_rope, ggml_tensor* k, ggml_tensor* v,
                                 int64_t start, int64_t len, bool fa, float kv_scale) {
            auto q_t = ggml_view_3d(ctx->ggml_ctx, q_rope, q_rope->ne[0], len, q_rope->ne[2],
                                    q_rope->nb[1], q_rope->nb[2], q_rope->nb[1] * start);
            q_t      = ggml_cont(ctx->ggml_ctx, q_t);
            return ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q_t, k, v,
                                          num_heads, nullptr, true, fa, kv_scale);
        }

        ggml_tensor* self_attn(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, int64_t n_cond_tokens, int64_t T, int block_idx = -1) {
            auto qkv    = std::dynamic_pointer_cast<Linear>(blocks["attn.qkv"]);
            auto q_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["attn.q_norm"]);
            auto k_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["attn.k_norm"]);
            auto proj   = std::dynamic_pointer_cast<Linear>(blocks["attn.proj"]);

            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            // Self-attn qkv. The fused qkv Linear produces a [3*C, n_token] tensor, and
            // split_qkv() then reshapes+permutes+CONTs the WHOLE thing into a second
            // contiguous [3, C, n_token] buffer before viewing q/k/v. At full length
            // (93f, n_token~37k) qkv_out is ~1.75 GiB and its cont is a SECOND ~1.75 GiB
            // — the two coexist and are the dominant resident compute-buffer peak
            // (~3.5 GiB of the 5.3 GiB monolithic reserve; localized via
            // GGML_ALLOCATOR_DEBUG). Instead, split the qkv WEIGHT (out-dim = 3*C, the
            // ggml ROW dim ne[1] — Q4_K rows are independently quantized, so row-slices
            // are exact) into Wq/Wk/Wv views and run three separate matmuls. Each output
            // is only ~0.58 GiB and no fused [3*C] buffer or its cont is ever
            // materialized: peak self-attn extraction is q+k+v (~1.75 GiB) instead of
            // qkv_out+cont (~3.5 GiB). Mathematically identical to the fused matmul.
            int64_t Cq      = hidden_size;
            auto qkv_w_full = qkv->get_weight();  // ggml ne=[C, 3C]
            auto qkv_b_full = qkv->get_bias();    // ggml ne=[3C]
            const bool pf32_qkv = qkv->get_force_prec_f32();
            auto qkv_part = [&](int idx) {
                // weight rows [idx*C, (idx+1)*C) — contiguous along ne[1].
                auto w = ggml_view_2d(ctx->ggml_ctx, qkv_w_full, qkv_w_full->ne[0], Cq,
                                      qkv_w_full->nb[1], qkv_w_full->nb[1] * Cq * idx);
                ggml_tensor* b = nullptr;
                if (qkv_b_full) {
                    b = ggml_view_1d(ctx->ggml_ctx, qkv_b_full, Cq, qkv_b_full->nb[0] * Cq * idx);
                }
                return ggml_ext_linear(ctx->ggml_ctx, x, w, b, pf32_qkv, 1.0f);  // [C, n_token, N]
            };
            ggml_tensor* parts[3] = {qkv_part(0), qkv_part(1), qkv_part(2)};
            auto q       = ggml_reshape_4d(ctx->ggml_ctx, parts[0], head_dim, num_heads, n_token, N);
            auto k       = ggml_reshape_4d(ctx->ggml_ctx, parts[1], head_dim, num_heads, n_token, N);
            auto v       = ggml_reshape_4d(ctx->ggml_ctx, parts[2], head_dim, num_heads, n_token, N);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            // 3D RoPE. The avatar's rope_3d.py uses `repeat(freqs, "(n r)", r=2)`
            // + `rotate_half` pairing adjacent dims (2i, 2i+1) — i.e. the
            // INTERLEAVED (GPT-J) convention, NOT GPT-NeoX. Numerically validated:
            // the Wan helper's per-axis omega[j]=theta^(-2j/dim) angles are
            // bit-identical to the avatar's, and the {44,42,42} axis split matches
            // dim_t=44/dim_h=42/dim_w=42. So rope_interleaved=true is correct.
            // Apply RoPE to the FULL q/k first (positions match absolute token index,
            // identical to the reference which ropes before the cond/noise split).
            bool fa     = ctx->flash_attn_enabled;
            // Flash-attn casts k/v to F16. q/k are RMS-normed (bounded ~1) but v
            // carries the large (additively-growing, up to ~1e6 over depth) residual
            // stream magnitude — casting it straight to F16 overflows (>65504 -> inf
            // -> NaN), the same class of bug as the ffn.w2 Q8_1 overflow. kv_scale
            // shrinks k/v before the F16 cast and rescales the output back (softmax is
            // invariant to a uniform k scale, and the v scale is undone on the output),
            // so the result is mathematically identical but stays in F16 range. The
            // non-flash path materializes F32 scores and is unaffected (kv_scale==1).
            const float kv_scale = fa ? (1.0f / 256.0f) : 1.0f;
            auto q_rope = Rope::apply_rope(ctx->ggml_ctx, q, pe, true);  // [d_head, n_token, num_heads*N]
            auto k_rope = Rope::apply_rope(ctx->ggml_ctx, k, pe, true);  // [d_head, n_token, num_heads*N]

            ggml_tensor* out;
            if (ctx->cond_kv_cache && ctx->sampler_step > 1 && block_idx >= 0) {
                // ======= COND-FRAME K/V CACHE CONSUME (lap-26, ai2v) =======
                // step>0: x is the NOISE tokens only (forward() sliced off the cond
                // frame). The cond frame's post-RoPE K/V are step-invariant, persisted
                // at step 0; reattach them so the noise queries see the exact same full
                // K/V as the dense path — concat order [cond | noise] matches the
                // original full-length token layout, and the cond K were RoPE'd at
                // their absolute positions [0,n_cond) at step 0, so this is bit-exact.
                GGML_ASSERT(ctx->condkv_k != nullptr && block_idx < (int)ctx->condkv_k->size() && "cond-kv cache missing");
                ggml_tensor* k_cond = (*ctx->condkv_k)[block_idx];
                ggml_tensor* v_cond = (*ctx->condkv_v)[block_idx];
                // LongCat lap-28.2: keep cached F16 K/V (already pre-scaled by kv_scale at
                // persist time), pre-scale + cast the noise K/V to F16, concat F16-with-F16,
                // and tell the wrapper to skip its in-flight kv_scale + F16 cast. Saves the
                // F16→F32→×(1/kv_scale) → concat F32 → ×kv_scale → cast→F16 round-trip
                // (~178 MB per cond+noise tensor × 48 blocks × 7 consume steps). Bit-exact
                // because kv_scale=1/256 is an exact F16 exponent shift and F16→F32 is
                // lossless, so the cancellation is mathematical (not numerical drift).
                auto k_noise_f16 = fa ? ggml_cast(ctx->ggml_ctx,
                                                   ggml_ext_scale(ctx->ggml_ctx, k_rope, kv_scale),
                                                   GGML_TYPE_F16)
                                       : k_rope;
                auto v_noise_f16 = fa ? ggml_cast(ctx->ggml_ctx,
                                                   ggml_ext_scale(ctx->ggml_ctx, v, kv_scale),
                                                   GGML_TYPE_F16)
                                       : v;
                ggml_tensor* k_full;
                ggml_tensor* v_full;
                if (fa) {
                    k_full = ggml_concat(ctx->ggml_ctx, k_cond, k_noise_f16, 1);  // F16 [d_head, n_cond+n_noise, heads]
                    v_full = ggml_concat(ctx->ggml_ctx, v_cond, v_noise_f16, 2);  // F16 [d_head, heads, n_cond+n_noise, N]
                } else {
                    // Non-flash fallback (kv_scale==1): persisted cache happens to be F16(*1)=F16(v).
                    // Round-trip back to F32 and concat as before.
                    k_full = ggml_concat(ctx->ggml_ctx,
                                         ggml_cast(ctx->ggml_ctx, k_cond, GGML_TYPE_F32), k_rope, 1);
                    v_full = ggml_concat(ctx->ggml_ctx,
                                         ggml_cast(ctx->ggml_ctx, v_cond, GGML_TYPE_F32), v, 2);
                }
                // LongCat lap-28.4 (BSA): if a runner-built BSA mask is attached, pass it
                // to flash. Quality trade — not bit-exact (sparse attention). Mask is
                // pre-computed once per render and shared across all 48 blocks + 7
                // consume steps. Persist path (step 0) stays dense for stability.
                out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q_rope, k_full, v_full,
                                             num_heads, ctx->bsa_mask, true, fa, 1.0f);
                if (fa) {
                    out = ggml_ext_scale(ctx->ggml_ctx, out, 1.0f / kv_scale);
                }
            } else if (num_ref_latents > 0 && n_cond_tokens > 0 && n_cond_tokens < n_token && n_per_frame > 0) {
                // ======= VIDEO-CONTINUATION 3-WAY SPLIT (generate_avc) =======
                // Layout: [ref(num_ref_latents) | cond_tail | noise]. The leading
                // n_cond_tokens span = (num_ref + num_cond_tail) frames. This mirrors
                // avatar/attention.py Attention.forward num_cond_latents>1 branch:
                //   q_ref   -> attends ONLY ref            (k_ref, v_ref)
                //   q_cond  -> attends ONLY cond_tail      (k_cond, v_cond)  [NOT ref]
                //   q_noise -> attends FULL (ref+cond+noise) EXCEPT a mask_frame_range
                //              carve-out near ref_img_index where it skips the ref tokens.
                int64_t n_ref_tokens = (int64_t)num_ref_latents * n_per_frame;  // = num_ref_latents_thw
                int64_t L_noise      = n_token - n_cond_tokens;
                int64_t num_cond_latents = n_cond_tokens / n_per_frame;  // ref + cond_tail (total)

                // ref pass: q_ref × {k,v}_ref  (ref tokens see only ref)
                auto k_ref = ggml_view_3d(ctx->ggml_ctx, k_rope, k_rope->ne[0], n_ref_tokens, k_rope->ne[2],
                                          k_rope->nb[1], k_rope->nb[2], 0);
                auto v_ref = ggml_view_4d(ctx->ggml_ctx, v, v->ne[0], v->ne[1], n_ref_tokens, v->ne[3],
                                          v->nb[1], v->nb[2], v->nb[3], 0);
                k_ref       = ggml_cont(ctx->ggml_ctx, k_ref);
                v_ref       = ggml_cont(ctx->ggml_ctx, v_ref);
                auto x_ref  = attn_qslice(ctx, q_rope, k_ref, v_ref, 0, n_ref_tokens, fa, kv_scale);

                // cond pass: q_cond × {k,v}_cond  (cond_tail tokens see only cond_tail)
                int64_t n_condtail = n_cond_tokens - n_ref_tokens;  // cond_tail token span
                auto k_cond = ggml_view_3d(ctx->ggml_ctx, k_rope, k_rope->ne[0], n_condtail, k_rope->ne[2],
                                           k_rope->nb[1], k_rope->nb[2], k_rope->nb[1] * n_ref_tokens);
                auto v_cond = ggml_view_4d(ctx->ggml_ctx, v, v->ne[0], v->ne[1], n_condtail, v->ne[3],
                                           v->nb[1], v->nb[2], v->nb[3], v->nb[2] * n_ref_tokens);
                k_cond      = ggml_cont(ctx->ggml_ctx, k_cond);
                v_cond      = ggml_cont(ctx->ggml_ctx, v_cond);
                auto x_cond = attn_qslice(ctx, q_rope, k_cond, v_cond, n_ref_tokens, n_condtail, fa, kv_scale);

                // noise pass. mask_frame_range carve-out (in NOISE-LOCAL frames):
                //   start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
                //   end_noise   = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1
                // applied only when 0 <= start_noise < end_noise <= num_noisy_frames.
                int64_t num_noisy_frames = T - num_cond_latents;
                int64_t start_noise = (int64_t)ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents;
                int64_t end_noise   = (int64_t)ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1;
                ggml_tensor* x_noise = nullptr;
                if (mask_frame_range > 0 && start_noise >= 0 && end_noise > start_noise && end_noise <= num_noisy_frames) {
                    // k/v EXCLUDING the ref tokens (= k[ n_ref_tokens : ], cond_tail+noise)
                    int64_t n_nonref = n_token - n_ref_tokens;
                    auto k_nonref = ggml_view_3d(ctx->ggml_ctx, k_rope, k_rope->ne[0], n_nonref, k_rope->ne[2],
                                                 k_rope->nb[1], k_rope->nb[2], k_rope->nb[1] * n_ref_tokens);
                    auto v_nonref = ggml_view_4d(ctx->ggml_ctx, v, v->ne[0], v->ne[1], n_nonref, v->ne[3],
                                                 v->nb[1], v->nb[2], v->nb[3], v->nb[2] * n_ref_tokens);
                    k_nonref      = ggml_cont(ctx->ggml_ctx, k_nonref);
                    v_nonref      = ggml_cont(ctx->ggml_ctx, v_nonref);

                    int64_t start_pos = start_noise * n_per_frame;  // noise-local token offsets
                    int64_t end_pos   = end_noise * n_per_frame;
                    // q_noise is the [n_cond_tokens : n_token) span of q_rope. Sub-slice
                    // it into front / maskref / back (noise-local offsets, so the global
                    // q_rope offset is n_cond_tokens + local).
                    ggml_tensor* parts_noise[3] = {nullptr, nullptr, nullptr};
                    if (start_pos > 0) {
                        parts_noise[0] = attn_qslice(ctx, q_rope, k_rope, v, n_cond_tokens, start_pos, fa, kv_scale);
                    }
                    parts_noise[1] = attn_qslice(ctx, q_rope, k_nonref, v_nonref,
                                                 n_cond_tokens + start_pos, end_pos - start_pos, fa, kv_scale);
                    if (end_pos < L_noise) {
                        parts_noise[2] = attn_qslice(ctx, q_rope, k_rope, v,
                                                     n_cond_tokens + end_pos, L_noise - end_pos, fa, kv_scale);
                    }
                    for (int p = 0; p < 3; ++p) {
                        if (parts_noise[p] == nullptr) continue;
                        x_noise = (x_noise == nullptr) ? parts_noise[p]
                                                       : ggml_concat(ctx->ggml_ctx, x_noise, parts_noise[p], 1);
                    }
                } else {
                    // no carve-out: noise attends the full ref+cond+noise k/v.
                    x_noise = attn_qslice(ctx, q_rope, k_rope, v, n_cond_tokens, L_noise, fa, kv_scale);
                }

                out = ggml_concat(ctx->ggml_ctx, x_ref, x_cond, 1);
                out = ggml_concat(ctx->ggml_ctx, out, x_noise, 1);  // [N, n_token, C]
            } else if (n_cond_tokens > 0 && n_cond_tokens < n_token) {
                // v is [d_head, num_heads, n_token, N]; q_rope/k_rope are
                // [d_head, n_token, num_heads*N] (token on ne[1]). Slice on the token
                // dim. (N==1 in the avatar path, so num_heads*N == num_heads.)
                int64_t L_noise = n_token - n_cond_tokens;

                // cond pass: q_cond × {k,v}_cond  (cond tokens see only cond tokens)
                auto q_cond = ggml_view_3d(ctx->ggml_ctx, q_rope, q_rope->ne[0], n_cond_tokens, q_rope->ne[2],
                                           q_rope->nb[1], q_rope->nb[2], 0);
                auto k_cond = ggml_view_3d(ctx->ggml_ctx, k_rope, k_rope->ne[0], n_cond_tokens, k_rope->ne[2],
                                           k_rope->nb[1], k_rope->nb[2], 0);
                auto v_cond = ggml_view_4d(ctx->ggml_ctx, v, v->ne[0], v->ne[1], n_cond_tokens, v->ne[3],
                                           v->nb[1], v->nb[2], v->nb[3], 0);
                q_cond      = ggml_cont(ctx->ggml_ctx, q_cond);
                k_cond      = ggml_cont(ctx->ggml_ctx, k_cond);
                v_cond      = ggml_cont(ctx->ggml_ctx, v_cond);                // Cond-frame K/V cache (lap-26): the cond frame's input (init_latent) and
                // timestep (0) are step-invariant, so post-RoPE k_cond / v_cond are bit-
                // identical every step. Persist them at step 0; steps>0 reuse them and skip
                // the cond compute (consume path — WIP). Default off ⇒ no graph change.
                if (ctx->cond_kv_cache && ctx->sampler_step <= 1 && block_idx >= 0 && ctx->condkv_k &&
                    block_idx < (int)ctx->condkv_k->size()) {
                    // Store F16(k_cond*kv_scale) / F16(v_cond*kv_scale) DIRECTLY into the
                    // persistent buffer (ggml_cpy, expanded by build_graph). Prescale halves
                    // the footprint AND keeps v F16-overflow-safe. Bit-exact on consume: the
                    // wrapper feeds F16(k/256) to flash; /256 is an exact F16 exponent shift,
                    // so F16(k*kv_scale) round-trips to that identical value.
                    auto kc = ggml_cast(ctx->ggml_ctx, ggml_ext_scale(ctx->ggml_ctx, k_cond, kv_scale), GGML_TYPE_F16);
                    auto vc = ggml_cast(ctx->ggml_ctx, ggml_ext_scale(ctx->ggml_ctx, v_cond, kv_scale), GGML_TYPE_F16);
                    auto wk = ggml_cpy(ctx->ggml_ctx, kc, (*ctx->condkv_k)[block_idx]);
                    auto wv = ggml_cpy(ctx->ggml_ctx, vc, (*ctx->condkv_v)[block_idx]);
                    if (ctx->cache_writes) { ctx->cache_writes->push_back(wk); ctx->cache_writes->push_back(wv); }
                    // LongCat lap-27 OFFLOAD FIX: tag each cpy write into the BLOCK'S
                    // existing "post_self_attn" subcut group. The graph-cut planner only
                    // builds segments around subcut-marked tensors + the final output —
                    // without this, the cpy writes belong to NO segment (they're not
                    // marked, and they're not on the residual chain), so the offload's
                    // segmented executor never runs them and condkv_buf stays zeros.
                    // Sharing the group name with the residual's post_self_attn subcut
                    // makes them additional outputs of the same segment — no extra
                    // compute (the cast/scale chain shares ancestors with the attn
                    // path) and minimal plumbing.
                    if (block_idx >= 0) {
                        sd::ggml_graph_cut::mark_graph_cut(
                            wk, "longcat.blocks." + std::to_string(block_idx) + ".post_self_attn", "condkv_k");
                        sd::ggml_graph_cut::mark_graph_cut(
                            wv, "longcat.blocks." + std::to_string(block_idx) + ".post_self_attn", "condkv_v");
                    }
                }
                auto x_cond = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q_cond, k_cond, v_cond,
                                                     num_heads, nullptr, true, fa, kv_scale);  // [N, n_cond, C]

                // noise pass: q_noise × {k,v}_full  (noise tokens see everything).
                // Optionally TILE over query blocks: flash-attn output for a query row
                // is independent of other query rows (each query attends the full K/V),
                // so splitting q_noise into row-blocks and concatenating outputs is
                // mathematically exact. This bounds the per-call q-cont + attention
                // output to a tile (~585 MiB/attn_tiles at 93f) instead of the full
                // L_noise extent — k_rope/v stay shared. attn_query_tiles==1 (the 25f
                // hot path) is the original single-shot call, bit-for-bit.
                ggml_tensor* x_noise = nullptr;
                int64_t qtiles = attn_query_tiles;
                if (qtiles > 1 && L_noise > qtiles) {
                    int64_t base = (L_noise + qtiles - 1) / qtiles;
                    for (int64_t off = 0; off < L_noise; off += base) {
                        int64_t len = std::min<int64_t>(base, L_noise - off);
                        auto q_t    = ggml_view_3d(ctx->ggml_ctx, q_rope, q_rope->ne[0], len, q_rope->ne[2],
                                                   q_rope->nb[1], q_rope->nb[2],
                                                   q_rope->nb[1] * (n_cond_tokens + off));
                        q_t         = ggml_cont(ctx->ggml_ctx, q_t);
                        auto x_t    = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q_t, k_rope, v,
                                                          num_heads, nullptr, true, fa, kv_scale);
                        x_noise     = x_noise == nullptr ? x_t : ggml_concat(ctx->ggml_ctx, x_noise, x_t, 1);
                    }
                } else {
                    auto q_noise = ggml_view_3d(ctx->ggml_ctx, q_rope, q_rope->ne[0], L_noise, q_rope->ne[2],
                                                q_rope->nb[1], q_rope->nb[2], q_rope->nb[1] * n_cond_tokens);
                    q_noise      = ggml_cont(ctx->ggml_ctx, q_noise);
                    x_noise      = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q_noise, k_rope, v,
                                                          num_heads, nullptr, true, fa, kv_scale);  // [N, L_noise, C]
                }

                out = ggml_concat(ctx->ggml_ctx, x_cond, x_noise, 1);  // [N, n_token, C]
            } else {
                out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q_rope, k_rope, v,
                                             num_heads, nullptr, true, fa, kv_scale);  // [N, n_token, C]
            }
            out = proj->forward(ctx, out);
            return out;
        }

        // audio cross attention: per-frame. The spatial tokens of each (noise)
        // latent frame attend ONLY to that frame's 32 audio tokens (block-diagonal
        // over T). Reference SingleStreamAttention._process_cross_attn reshapes
        //   x  "B (N_t S) C -> (B N_t) S C"   (S = tokens-per-frame, here n_per_frame)
        //   kv from audio_hidden_states "(B N_t) 32 768"
        // i.e. a batched attention with batch == n_noise_frames. q from x, kv from
        // the per-frame audio tokens. No RoPE (single-talk, x_ref_attn_map=None).
        //
        // x_noise: [C, n_noise_token, 1] (already cond-stripped) where
        //   n_noise_token = n_noise_frames * n_per_frame.
        // audio: [768, 32, n_noise_frames] (cond frames already stripped by caller).
        // returns [C, n_noise_token, 1].
        ggml_tensor* audio_cross_attn(GGMLRunnerContext* ctx, ggml_tensor* x_noise, ggml_tensor* audio,
                                      int64_t n_noise_frames, int64_t n_per_frame) {
            auto q_linear  = std::dynamic_pointer_cast<Linear>(blocks["audio_cross_attn.q_linear"]);
            auto kv_linear = std::dynamic_pointer_cast<Linear>(blocks["audio_cross_attn.kv_linear"]);
            auto proj      = std::dynamic_pointer_cast<Linear>(blocks["audio_cross_attn.proj"]);
            auto q_norm    = std::dynamic_pointer_cast<UnaryBlock>(blocks["audio_cross_attn.q_norm"]);
            auto k_norm    = std::dynamic_pointer_cast<UnaryBlock>(blocks["audio_cross_attn.k_norm"]);

            int64_t C        = x_noise->ne[0];
            int64_t n_a      = audio->ne[1];  // 32 audio context tokens

            // q from x: [C, n_noise_token, 1] -> per-frame batch [C, n_per_frame, n_noise_frames]
            auto q = q_linear->forward(ctx, x_noise);  // [C, n_noise_token, 1]
            q      = ggml_reshape_4d(ctx->ggml_ctx, q, C, n_per_frame, n_noise_frames, 1);  // batch over frames
            q      = ggml_reshape_3d(ctx->ggml_ctx, q, C, n_per_frame, n_noise_frames);     // [C, n_per_frame, T_n]

            // kv from audio: [768, 32, T_n] -> [2C, 32, T_n], split k/v over the
            // "2" axis. Mirror text_cross_attn's split: reshape [C, 2, 32, T_n] then
            // permute src(0,1,2,3)->dst(0,3,1,2) giving [C, 32, T_n, 2]; k=offset0, v=offset1.
            auto kv = kv_linear->forward(ctx, audio);  // [2C, 32, T_n]
            kv      = ggml_reshape_4d(ctx->ggml_ctx, kv, hidden_size, 2, n_a, n_noise_frames);  // [C, 2, 32, T_n]
            kv      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, kv, 0, 3, 1, 2));    // [C, 32, T_n, 2]
            int64_t off = kv->nb[3];
            auto k      = ggml_view_3d(ctx->ggml_ctx, kv, kv->ne[0], kv->ne[1], kv->ne[2], kv->nb[1], kv->nb[2], off * 0);
            auto v      = ggml_view_3d(ctx->ggml_ctx, kv, kv->ne[0], kv->ne[1], kv->ne[2], kv->nb[1], kv->nb[2], off * 1);
            k           = ggml_cont(ctx->ggml_ctx, k);  // [C, 32, T_n]
            v           = ggml_cont(ctx->ggml_ctx, v);  // [C, 32, T_n]

            // qk RMSNorm over head_dim
            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_per_frame, n_noise_frames);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_a, n_noise_frames);
            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);
            q = ggml_reshape_3d(ctx->ggml_ctx, q, hidden_size, n_per_frame, n_noise_frames);
            k = ggml_reshape_3d(ctx->ggml_ctx, k, hidden_size, n_a, n_noise_frames);

            // Per-frame batched attention uses the N (ne[2]) dim as the batch over
            // T_noise frames. ggml_ext_attention_ext's FLASH path collapses N to 1
            // (its output view is 3D), so it only supports N==1 — force the non-flash
            // materialized-scores path here. K is tiny (32 audio tokens) so the
            // [L_k=32, L_q=n_per_frame, n_head*T_n] score tensor is cheap.
            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false,
                                              false, 1.0f);  // [C, n_per_frame, T_n]
            out      = proj->forward(ctx, out);
            // flatten frames back to tokens: [C, n_per_frame, T_n] -> [C, n_noise_token, 1]
            out      = ggml_reshape_3d(ctx->ggml_ctx, out, C, n_per_frame * n_noise_frames, 1);
            return out;
        }

        // text cross attention: q from x, kv from context. UNGATED.
        ggml_tensor* text_cross_attn(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context, int block_idx = -1) {
            auto q_linear  = std::dynamic_pointer_cast<Linear>(blocks["cross_attn.q_linear"]);
            auto kv_linear = std::dynamic_pointer_cast<Linear>(blocks["cross_attn.kv_linear"]);
            auto proj      = std::dynamic_pointer_cast<Linear>(blocks["cross_attn.proj"]);
            auto q_norm    = std::dynamic_pointer_cast<UnaryBlock>(blocks["cross_attn.q_norm"]);
            auto k_norm    = std::dynamic_pointer_cast<UnaryBlock>(blocks["cross_attn.k_norm"]);

            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];
            int64_t n_ctx   = context->ne[1];

            const bool fa        = ctx->flash_attn_enabled;
            // kv_scale guards the F16 cast in the flash path (see self_attn note); the
            // text-context v is bounded so this is defensive, and a no-op (==1) for the
            // non-flash path which keeps F32 scores.
            const float kv_scale = fa ? (1.0f / 256.0f) : 1.0f;

            // LongCat lap-31: text x-attn K/V cache CONSUME (step>1). The text-context
            // is step-invariant (umT5 runs once pre-sampling), so kv_linear(context) +
            // permute+cont + k_norm + reshape produce byte-identical K/V every step.
            // The cached tensors hold F16(K*kv_scale) and F16(V*kv_scale) in shape
            // [hidden_size, n_ctx, Nb] (post-norm, post-reshape — what the attention
            // wrapper expects when called with kv_prescaled_f16=true on the FA path).
            // Only K and V are cached; Q is recomputed per step (token-dependent).
            const bool xattn_consume = fa && ctx->xattn_text_k && ctx->sampler_step > 1 &&
                                       block_idx >= 0 && block_idx < (int)ctx->xattn_text_k->size();
            if (xattn_consume) {
                auto q = q_linear->forward(ctx, x);
                q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);
                q = q_norm->forward(ctx, q);
                q = ggml_reshape_3d(ctx->ggml_ctx, q, hidden_size, n_token, N);

                ggml_tensor* k_cached = (*ctx->xattn_text_k)[block_idx];
                ggml_tensor* v_cached = (*ctx->xattn_text_v)[block_idx];
                auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k_cached, v_cached,
                                                  num_heads, nullptr, false, fa, 1.0f);
                out = ggml_ext_scale(ctx->ggml_ctx, out, 1.0f / kv_scale);
                out = proj->forward(ctx, out);
                return out;
            }

            auto q  = q_linear->forward(ctx, x);        // [N, n_token, C]
            auto kv = kv_linear->forward(ctx, context);  // [N, n_ctx, 2C]
            kv      = ggml_reshape_4d(ctx->ggml_ctx, kv, hidden_size, 2, n_ctx, N);  // [N, n_ctx, 2, C]
            kv      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, kv, 0, 3, 1, 2));  // [2, N, n_ctx, C]
            int64_t off = kv->nb[2] * kv->ne[2];
            auto k      = ggml_view_3d(ctx->ggml_ctx, kv, kv->ne[0], kv->ne[1], kv->ne[2], kv->nb[1], kv->nb[2], off * 0);
            auto v      = ggml_view_3d(ctx->ggml_ctx, kv, kv->ne[0], kv->ne[1], kv->ne[2], kv->nb[1], kv->nb[2], off * 1);
            k           = ggml_cont(ctx->ggml_ctx, k);
            v           = ggml_cont(ctx->ggml_ctx, v);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_ctx, N);
            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);
            q = ggml_reshape_3d(ctx->ggml_ctx, q, hidden_size, n_token, N);
            k = ggml_reshape_3d(ctx->ggml_ctx, k, hidden_size, n_ctx, N);
            v = ggml_reshape_3d(ctx->ggml_ctx, v, hidden_size, n_ctx, N);

            // LongCat lap-31: text x-attn K/V cache PERSIST (step<=1, FA path only). Store
            // F16(K*kv_scale) / F16(V*kv_scale) into the persistent buffer. Same prescale
            // shift as cond-cache lap-28.2 (exact F16 exponent shift at 1/256), so consume
            // bit-matches dense. Writes get pushed into cache_writes and tagged into the
            // block's post_cross_attn subcut group so the offload-segmented executor sees
            // the cpy in the correct segment.
            if (fa && ctx->xattn_text_k && ctx->sampler_step <= 1 &&
                block_idx >= 0 && block_idx < (int)ctx->xattn_text_k->size() && ctx->cache_writes) {
                auto kc = ggml_cast(ctx->ggml_ctx, ggml_ext_scale(ctx->ggml_ctx, k, kv_scale), GGML_TYPE_F16);
                auto vc = ggml_cast(ctx->ggml_ctx, ggml_ext_scale(ctx->ggml_ctx, v, kv_scale), GGML_TYPE_F16);
                auto wk = ggml_cpy(ctx->ggml_ctx, kc, (*ctx->xattn_text_k)[block_idx]);
                auto wv = ggml_cpy(ctx->ggml_ctx, vc, (*ctx->xattn_text_v)[block_idx]);
                ctx->cache_writes->push_back(wk);
                ctx->cache_writes->push_back(wv);
                sd::ggml_graph_cut::mark_graph_cut(
                    wk, "longcat.blocks." + std::to_string(block_idx) + ".post_cross_attn", "xattn_text_k");
                sd::ggml_graph_cut::mark_graph_cut(
                    wv, "longcat.blocks." + std::to_string(block_idx) + ".post_cross_attn", "xattn_text_v");
            }

            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, fa, kv_scale);
            out      = proj->forward(ctx, out);
            return out;
        }

        // x: [N, n_token, C]
        // t_mod: [N, T, adaln_tembed_dim]  (already silu-free t embed; SiLU applied here)
        // context: [N, n_ctx, caption_dim_proj==C]
        // pe: 3D RoPE
        // n_cond_tokens: number of leading tokens belonging to the ref-image cond
        //   frame(s). They drive (a) the self-attn cond/noise split (cond tokens attend
        //   only to themselves) and (b) the text cross-attn cond-zeroing (cond frame
        //   gets no text). 0 = no split (plain text-to-video).
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* t_mod,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* audio,
                             int64_t T,
                             int64_t n_cond_tokens,
                             int block_idx = -1) {
            // block_idx >= 0 enables INTRA-block graph-cut boundaries (after self-attn,
            // after cross-attn). At full length (93f / ~37k tokens) a single block's
            // activation peak (FFN gate/up/gu all [11008, n_token] F32 ~4.6 GiB +
            // self-attn ~3 GiB) still overruns the 12 GB card when reserved as one
            // segment, so we sub-cut each block into self-attn / cross-attn / FFN
            // sub-segments. Names are unique per block via block_idx. -1 = no sub-cut.
            auto subcut = [&](ggml_tensor* t, const char* tag) {
                if (block_idx >= 0) {
                    sd::ggml_graph_cut::mark_graph_cut(
                        t, "longcat.blocks." + std::to_string(block_idx) + "." + tag, "x");
                }
            };
            auto adaLN          = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);
            auto audio_adaLN    = std::dynamic_pointer_cast<Linear>(blocks["audio_adaLN_modulation.1"]);
            auto mod_norm_attn  = std::dynamic_pointer_cast<LayerNorm>(blocks["mod_norm_attn"]);
            auto mod_norm_ffn   = std::dynamic_pointer_cast<LayerNorm>(blocks["mod_norm_ffn"]);
            auto pre_crs_norm   = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_crs_attn_norm"]);
            auto pre_video_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_video_crs_attn_norm"]);
            auto ffn_w1         = std::dynamic_pointer_cast<Linear>(blocks["ffn.w1"]);
            auto ffn_w2         = std::dynamic_pointer_cast<Linear>(blocks["ffn.w2"]);
            auto ffn_w3         = std::dynamic_pointer_cast<Linear>(blocks["ffn.w3"]);

            int64_t N = x->ne[2];

            // adaLN_modulation(SiLU(t)) -> [N, T, 6*C], chunk into 6 [N, T, C]
            auto t_act = ggml_silu(ctx->ggml_ctx, t_mod);
            auto mod   = adaLN->forward(ctx, t_act);  // [N, T, 6*C]
            mod        = ggml_reshape_4d(ctx->ggml_ctx, mod, hidden_size, 6, T, N);  // [N, T, 6, C]
            auto ms    = ggml_ext_chunk(ctx->ggml_ctx, mod, 6, 1);  // each [N, T, 1, C]
            // ms order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

            // self-attn with modulation
            auto x_m = modulate(ctx, mod_norm_attn, x, ms[0], ms[1], T);
            auto x_s = self_attn(ctx, x_m, pe, n_cond_tokens, T, block_idx);            x        = gate_add(ctx, x, x_s, ms[2], T);
            subcut(x, "post_self_attn");

            // text cross-attn (ungated). With a num_cond_latents split the cond-frame
            // tokens receive NO text conditioning (their output is zeroed).
            auto x_c = text_cross_attn(ctx, pre_crs_norm->forward(ctx, x), context, block_idx);
            if (n_cond_tokens > 0) {
                int64_t C  = x_c->ne[0];
                int64_t Nb = x_c->ne[2];
                // zero the first n_cond_tokens rows: keep [n_cond:] , prepend zeros.
                auto noise = ggml_view_3d(ctx->ggml_ctx, x_c, C, x_c->ne[1] - n_cond_tokens, Nb,
                                          x_c->nb[1], x_c->nb[2], x_c->nb[1] * n_cond_tokens);
                noise      = ggml_cont(ctx->ggml_ctx, noise);
                auto zeros = ggml_ext_zeros(ctx->ggml_ctx, C, n_cond_tokens, Nb, 1);
                x_c        = ggml_concat(ctx->ggml_ctx, zeros, noise, 1);
            }
            x = ggml_add(ctx->ggml_ctx, x, x_c);

            // audio cross-attn (per-frame). Only the GENERATED (noise) frames get
            // audio; the first n_cond_frames ref-image frames get NONE. Reference
            // (longcat_video_dit_avatar.py L162-179):
            //   a_shift,a_scale,a_gate = audio_adaLN(t[:, n_cond:]).chunk(3)
            //   ao = audio_cross_attn(pre_video_crs_attn_norm(x), audio_hidden_states)
            //        -> per-frame, noise frames only; cond output zeroed
            //   ao = modulate(mod_norm_attn, ao, a_shift, a_scale)   # REUSES mod_norm_attn
            //   x  = x + cat([zeros_cond, a_gate * ao])
            // pre_audio_crs_attn_norm is Identity (audio_prenorm=False) so the audio
            // side has no prenorm.
            if (audio != nullptr) {
                int64_t n_token     = x->ne[1];
                int64_t n_per_frame = n_token / T;
                int64_t n_cond_frames = n_cond_tokens > 0 ? (n_cond_tokens / n_per_frame) : 0;
                int64_t T_noise       = T - n_cond_frames;

                if (T_noise > 0) {
                    // pre_video_crs_attn_norm on x, then strip cond tokens for the q side.
                    auto xv = pre_video_norm->forward(ctx, x);  // [C, n_token, 1]
                    ggml_tensor* x_noise;
                    if (n_cond_tokens > 0) {
                        x_noise = ggml_view_3d(ctx->ggml_ctx, xv, xv->ne[0], n_token - n_cond_tokens, xv->ne[2],
                                               xv->nb[1], xv->nb[2], xv->nb[1] * n_cond_tokens);
                        x_noise = ggml_cont(ctx->ggml_ctx, x_noise);
                    } else {
                        x_noise = xv;
                    }

                    // audio: [768, 32, T] -> strip cond frames -> [768, 32, T_noise]
                    ggml_tensor* a_noise = audio;
                    if (n_cond_frames > 0) {
                        a_noise = ggml_view_3d(ctx->ggml_ctx, audio, audio->ne[0], audio->ne[1], T_noise,
                                               audio->nb[1], audio->nb[2], audio->nb[2] * n_cond_frames);
                        a_noise = ggml_cont(ctx->ggml_ctx, a_noise);
                    }

                    auto ao = audio_cross_attn(ctx, x_noise, a_noise, T_noise, n_per_frame);  // [C, n_noise_token, 1]

                    // audio adaLN modulation over noise frames: t[:, n_cond:].
                    // Reuse the SiLU(t_mod) already computed for the main adaLN
                    // (t_act) instead of recomputing it — same input, same op.
                    auto ta_act = t_act;  // [C_t, T, 1]
                    // slice noise frames of t_mod
                    if (n_cond_frames > 0) {
                        ta_act = ggml_view_3d(ctx->ggml_ctx, ta_act, ta_act->ne[0], T_noise, ta_act->ne[2],
                                              ta_act->nb[1], ta_act->nb[2], ta_act->nb[1] * n_cond_frames);
                        ta_act = ggml_cont(ctx->ggml_ctx, ta_act);
                    }
                    auto a_mod = audio_adaLN->forward(ctx, ta_act);  // [3*C, T_noise, 1]
                    a_mod      = ggml_reshape_4d(ctx->ggml_ctx, a_mod, hidden_size, 3, T_noise, 1);  // [C, 3, T_noise, 1]
                    auto am    = ggml_ext_chunk(ctx->ggml_ctx, a_mod, 3, 1);  // a_shift, a_scale, a_gate; each [C,1,T_noise,1]

                    // modulate(mod_norm_attn, ao, a_shift, a_scale) — reuse mod_norm_attn
                    auto ao_m = modulate(ctx, mod_norm_attn, ao, am[0], am[1], T_noise);
                    // gate add into x's noise tokens (a_gate*ao_m, no zeros residual)
                    auto add  = gate_mul(ctx, ao_m, am[2], T_noise);
                    // RUNTIME mouth-exaggeration knob: scale the audio's entire
                    // additive influence on the face. 1.0 (default) skips the op =
                    // bit-identical; <1 = milder mouth, >1 = more exaggerated.
                    if (audio_mouth_scale != 1.0f) {
                        add = ggml_scale(ctx->ggml_ctx, add, audio_mouth_scale);
                    }
                    // add: [C, n_noise_token, 1]. Prepend zeros for cond tokens, add to x.
                    if (n_cond_tokens > 0) {
                        auto zeros = ggml_ext_zeros(ctx->ggml_ctx, hidden_size, n_cond_tokens, x->ne[2], 1);
                        add        = ggml_concat(ctx->ggml_ctx, zeros, add, 1);  // [C, n_token, 1]
                    }
                    x = ggml_add(ctx->ggml_ctx, x, add);
                }
            }

            subcut(x, "post_cross_attn");

            // ffn (SwiGLU) with modulation.
            auto y = modulate(ctx, mod_norm_ffn, x, ms[3], ms[4], T);
            // The SwiGLU inner (w1/w3 -> silu*up -> w2) materializes three
            // [ffn_inner=11008, n_token] F32 transients (g, u, gu). At full length
            // (93f, n_token~37k) that triple is ~4.6 GiB — the dominant resident
            // compute-buffer peak (the FFN sub-segment). The SwiGLU is purely
            // per-token, so we TILE it over token blocks: each tile materializes
            // only [11008, tile] transients, bounding the FFN peak to ~(tile/n_token)
            // of the full triple. Mathematically identical (no token mixing in the
            // FFN). ffn_tiles==1 (the default for the 25f hot path) is the original
            // single-shot path, bit-for-bit. >1 only kicks in at large n_token.
            int64_t n_tok = y->ne[1];
            int64_t tiles = ffn_token_tiles;
            if (tiles > 1 && n_tok > tiles) {
                // tile size rounded up; last tile shorter. No T-divisibility needed
                // (the SwiGLU is per-token; modulation already applied to y).
                int64_t base = (n_tok + tiles - 1) / tiles;
                int64_t C    = y->ne[0];
                int64_t Nb   = y->ne[2];
                ggml_tensor* out_ffn = nullptr;
                for (int64_t off = 0; off < n_tok; off += base) {
                    int64_t len   = std::min<int64_t>(base, n_tok - off);
                    auto y_t      = ggml_view_3d(ctx->ggml_ctx, y, C, len, Nb,
                                                 y->nb[1], y->nb[2], y->nb[1] * off);
                    y_t           = ggml_cont(ctx->ggml_ctx, y_t);
                    auto g_t      = ggml_silu(ctx->ggml_ctx, ffn_w1->forward(ctx, y_t));
                    auto u_t      = ffn_w3->forward(ctx, y_t);
                    auto gu_t     = ggml_mul(ctx->ggml_ctx, g_t, u_t);
                    auto o_t      = ffn_w2->forward(ctx, gu_t);  // [C, len, Nb]
                    out_ffn       = out_ffn == nullptr ? o_t : ggml_concat(ctx->ggml_ctx, out_ffn, o_t, 1);
                }
                y = out_ffn;
            } else {
                auto g  = ggml_silu(ctx->ggml_ctx, ffn_w1->forward(ctx, y));
                auto u  = ffn_w3->forward(ctx, y);
                auto gu = ggml_mul(ctx->ggml_ctx, g, u);
                y       = ffn_w2->forward(ctx, gu);
            }
            x      = gate_add(ctx, x, y, ms[5], T);

            return x;
        }
    };

    // final_layer: norm_final (non-affine LN) modulated by adaLN(SiLU(t)) -> linear
    class FinalLayer : public GGMLBlock {
    protected:
        int64_t hidden_size;
        float eps;

    public:
        FinalLayer(int64_t hidden_size,
                   int64_t num_patch,
                   int64_t out_channels,
                   int64_t adaln_tembed_dim,
                   float eps = 1e-6f)
            : hidden_size(hidden_size), eps(eps) {
            blocks["norm_final"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps, false));
            blocks["linear"]             = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_patch * out_channels, true, false, true));
            blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(adaln_tembed_dim, 2 * hidden_size, true, false, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* t_mod, int64_t T) {
            auto norm_final = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN      = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            int64_t N = x->ne[2];

            auto t_act = ggml_silu(ctx->ggml_ctx, t_mod);
            auto mod   = adaLN->forward(ctx, t_act);  // [N, T, 2*C]
            mod        = ggml_reshape_4d(ctx->ggml_ctx, mod, hidden_size, 2, T, N);
            auto ms    = ggml_ext_chunk(ctx->ggml_ctx, mod, 2, 1);  // shift, scale; each [N, T, 1, C]

            // (scale + 1) fused as ggml_scale_bias (see modulate()).
            // LongCat lap-27: pre-expand scale1 onto ctx->gf so the {NORM, MUL, ADD}
            // autofusion fires here too (otherwise SCALE_BIAS lands between NORM and
            // MUL in topo and breaks adjacency). Same trick as the block modulate path.
            auto scale1 = ggml_scale_bias(ctx->ggml_ctx, ms[1], 1.0f, 1.0f);
            if (ctx->gf != nullptr) {
                // LongCat lap-28: pre-expand shift (ms[0]) too — same reason as the
                // block modulate(): keep shift's view/CONT chain out of the [MUL, ADD]
                // adjacency so the {NORM, MUL, ADD} autofusion fires.
                ggml_build_forward_expand(ctx->gf, scale1);
                ggml_build_forward_expand(ctx->gf, ms[0]);
            }
            x          = norm_final->forward(ctx, x);
            int64_t C  = x->ne[0];
            x          = ggml_reshape_4d(ctx->ggml_ctx, x, C, x->ne[1] / T, T, N);  // [N, T, n_token/T, C]
            x           = ggml_mul(ctx->ggml_ctx, x, scale1);
            x           = ggml_add(ctx->ggml_ctx, x, ms[0]);
            x           = ggml_reshape_3d(ctx->ggml_ctx, x, C, x->ne[1] * x->ne[2], N);

            x = linear->forward(ctx, x);  // [N, n_token, num_patch*out_channels]
            return x;
        }
    };

    // x_embedder.proj: Conv3d with temporal patch=1, stored as a Conv2d-shaped
    // weight [pw, ph, in_channels, hidden] (ggml-reversed of torch [out,in,kh,kw]
    // with the kt=1 axis dropped). We allocate the weight in that exact shape so
    // it loads from the gguf, and run a per-frame Conv2d.
    class PatchEmbed3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t hidden;
        int pw, ph;
        std::string prefix;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, pw, ph, in_channels, hidden);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        }

    public:
        PatchEmbed3D(int64_t in_channels, int64_t hidden, int pw, int ph)
            : in_channels(in_channels), hidden(hidden), pw(pw), ph(ph) {}

        // x: [W, H, T, in_channels]  (single batch). Returns [w_len, h_len, hidden, T].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = params["bias"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
            // x is [W, H, T, C]; ggml_conv_2d wants [W, H, in_channels, batch].
            // Permute frames(T) and channels(C): [W, H, T, C] -> [W, H, C, T].
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // [W, H, C, T]
            x = ggml_conv_2d(ctx->ggml_ctx, w, x, pw, ph, 0, 0, 1, 1);  // [w_len, h_len, hidden, T]
            b = ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, b->ne[0], 1);
            x = ggml_add_inplace(ctx->ggml_ctx, x, b);  // [w_len, h_len, hidden, T]
            return x;
        }
    };

    struct LongCatAvatarParams {
        std::tuple<int, int, int> patch_size = {1, 2, 2};
        int64_t in_channels                  = 16;
        int64_t out_channels                 = 16;
        int64_t hidden_size                  = 4096;
        int num_layers                       = 48;
        int64_t num_heads                    = 32;
        int64_t caption_channels             = 4096;
        int64_t ffn_inner                    = 11008;  // SwiGLU inner
        int64_t adaln_tembed_dim             = 512;
        int frequency_embedding_size         = 256;
        int64_t audio_dim                    = 768;
        float eps                            = 1e-6f;
        int theta                            = 10000;
        // 3D RoPE axes_dim for head_dim 128: dim_t=44, dim_h=42, dim_w=42.
        std::vector<int> axes_dim = {44, 42, 42};
        int64_t axes_dim_sum      = 128;
    };

    class LongCatAvatar : public GGMLBlock {
    protected:
        LongCatAvatarParams params;

    public:
        // RUNTIME mouth-exaggeration knob (default 1.0 = unchanged). Set per-render by
        // the runner; pushed onto every block in the forward loop.
        float audio_mouth_scale = 1.0f;

        // Video-continuation (generate_avc) 3-way-split params. num_ref_latents>0
        // selects the ref/cond/noise split in self_attn; 0 = ai2v 2-way split. Set
        // per-render by the runner.
        int     num_ref_latents  = 0;
        int     ref_img_index    = 10;
        int     mask_frame_range = 3;
        int64_t n_per_frame      = 0;  // spatial tokens per latent frame (h_len*w_len)

    protected:

    public:
        LongCatAvatar() {}
        LongCatAvatar(LongCatAvatarParams params)
            : params(params) {
            int64_t head_dim = params.hidden_size / params.num_heads;
            GGML_ASSERT(head_dim == 128);

            // x_embedder.proj : Conv3d(in, hidden, patch, patch). Temporal patch
            // is 1, weight stored as [kw,kh,in,out] = [2,2,16,4096] (Conv2d-shaped).
            blocks["x_embedder.proj"] = std::shared_ptr<GGMLBlock>(
                new PatchEmbed3D(params.in_channels, params.hidden_size,
                                 std::get<2>(params.patch_size), std::get<1>(params.patch_size)));

            // t_embedder.mlp : Linear(freq, adaln) -> SiLU -> Linear(adaln, adaln)
            blocks["t_embedder.mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(params.frequency_embedding_size, params.adaln_tembed_dim, true, false, true));
            blocks["t_embedder.mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(params.adaln_tembed_dim, params.adaln_tembed_dim, true, false, true));

            // y_embedder.y_proj : Linear -> GELU(tanh) -> Linear
            blocks["y_embedder.y_proj.0"] = std::shared_ptr<GGMLBlock>(new Linear(params.caption_channels, params.hidden_size, true, false, true));
            blocks["y_embedder.y_proj.2"] = std::shared_ptr<GGMLBlock>(new Linear(params.hidden_size, params.hidden_size, true, false, true));

            for (int i = 0; i < params.num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                    new LongCatAvatarSingleStreamBlock(params.hidden_size,
                                                       params.num_heads,
                                                       params.ffn_inner,
                                                       params.adaln_tembed_dim,
                                                       params.caption_channels,
                                                       params.audio_dim,
                                                       params.eps));
            }

            int64_t num_patch = std::get<0>(params.patch_size) * std::get<1>(params.patch_size) * std::get<2>(params.patch_size);
            blocks["final_layer"] = std::shared_ptr<GGMLBlock>(
                new FinalLayer(params.hidden_size, num_patch, params.out_channels, params.adaln_tembed_dim, params.eps));

            // audio_proj : AudioProjModel (avatar/blocks.py). seq_len=5 / seq_len_vf=8
            // (= audio_window+vae_scale-1) / blocks=5 / channels=1280 (whisper d_model):
            //   proj1   : Linear(5*5*1280=32000, 512)
            //   proj1_vf: Linear(8*5*1280=51200, 512)
            //   proj2   : Linear(512, 512)
            //   proj3   : Linear(512, 32*768=24576)
            //   norm    : LayerNorm(768)
            // Invoked in forward() when the audio window inputs are provided.
            blocks["audio_proj.proj1"]    = std::shared_ptr<GGMLBlock>(new Linear(32000, 512, true));
            blocks["audio_proj.proj1_vf"] = std::shared_ptr<GGMLBlock>(new Linear(51200, 512, true));
            blocks["audio_proj.proj2"]    = std::shared_ptr<GGMLBlock>(new Linear(512, 512, true));
            blocks["audio_proj.proj3"]    = std::shared_ptr<GGMLBlock>(new Linear(512, 32 * params.audio_dim, true));
            blocks["audio_proj.norm"]     = std::shared_ptr<GGMLBlock>(new LayerNorm(params.audio_dim, 1e-5f, true, true));
        }

        // AudioProjModel forward (avatar/blocks.py): dual-window MLP.
        //   first:  [32000, 1]   latter: [51200, n_latter]
        //   relu(proj1(first)) ++ relu(proj1_vf(latter)) -> relu(proj2) -> proj3
        //   -> reshape [768, 32, N_t] -> LayerNorm.  Returns [768, 32, N_t].
        ggml_tensor* audio_proj(GGMLRunnerContext* ctx, ggml_tensor* first, ggml_tensor* latter) {
            auto proj1    = std::dynamic_pointer_cast<Linear>(blocks["audio_proj.proj1"]);
            auto proj1_vf = std::dynamic_pointer_cast<Linear>(blocks["audio_proj.proj1_vf"]);
            auto proj2    = std::dynamic_pointer_cast<Linear>(blocks["audio_proj.proj2"]);
            auto proj3    = std::dynamic_pointer_cast<Linear>(blocks["audio_proj.proj3"]);
            auto norm     = std::dynamic_pointer_cast<LayerNorm>(blocks["audio_proj.norm"]);

            auto a = ggml_relu(ctx->ggml_ctx, proj1->forward(ctx, first));      // [512, 1]
            ggml_tensor* c;
            if (latter != nullptr && latter->ne[1] > 0) {
                auto b = ggml_relu(ctx->ggml_ctx, proj1_vf->forward(ctx, latter));  // [512, n_latter]
                c      = ggml_concat(ctx->ggml_ctx, a, b, 1);                       // [512, N_t]
            } else {
                c = a;
            }
            c           = ggml_relu(ctx->ggml_ctx, proj2->forward(ctx, c));         // [512, N_t]
            c           = proj3->forward(ctx, c);                                   // [32*768, N_t]
            int64_t N_t = c->ne[1];
            c           = ggml_reshape_3d(ctx->ggml_ctx, c, 768, 32, N_t);          // [768, 32, N_t]
            c           = norm->forward(ctx, c);
            return c;
        }

        // patch embed: x [W, H, T, C] (C=in_channels). patch_size temporal is 1,
        // so this is a per-frame Conv2d with kernel [pw,ph] stride [pw,ph].
        // Returns [N, t_len*h_len*w_len, hidden].
        ggml_tensor* patch_embed(GGMLRunnerContext* ctx, ggml_tensor* x, int64_t& t_len, int64_t& h_len, int64_t& w_len) {
            auto proj = std::dynamic_pointer_cast<PatchEmbed3D>(blocks["x_embedder.proj"]);

            int64_t T = x->ne[2];

            // PatchEmbed3D produces [w_len, h_len, hidden, T] (per-frame Conv2d).
            x = proj->forward(ctx, x);

            w_len          = x->ne[0];
            h_len          = x->ne[1];
            int64_t hidden = x->ne[2];
            t_len          = T;

            // ggml ne=[w_len,h_len,hidden,T] == torch shape [T,hidden,h_len,w_len].
            // Want the token-flatten order (T,h_len,w_len) with hidden contiguous as the
            // channel, i.e. ggml ne=[hidden,w_len,h_len,T] (== torch [T,h_len,w_len,hidden]).
            // ggml_ext_torch_permute's axis args are the INVERSE mapping (it derives
            // ggml_axes from the inverse perm), so the torch_axes that produce ggml ne
            // [hidden,w_len,h_len,T] are (2,0,1,3), NOT (0,2,3,1). The old (0,2,3,1)
            // yielded ggml ne [w_len,hidden,T,h_len] — a full element-scramble that
            // matched ref std but cos~0 (localized via the torch oracle 2026-05-25; the
            // corrected perm reproduces the reference token order bit-exact).
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
            x = ggml_reshape_3d(ctx->ggml_ctx, x, hidden, w_len * h_len * t_len, 1);  // [N=1, thw, hidden]
            return x;
        }

        ggml_tensor* unpatchify(ggml_context* ctx,
                                ggml_tensor* x,
                                int64_t t_len,
                                int64_t h_len,
                                int64_t w_len) {
            // mirror of WAN::Wan::unpatchify
            int64_t N  = x->ne[3];
            int64_t pt = std::get<0>(params.patch_size);
            int64_t ph = std::get<1>(params.patch_size);
            int64_t pw = std::get<2>(params.patch_size);
            int64_t C  = x->ne[0] / pt / ph / pw;

            GGML_ASSERT(C * pt * ph * pw == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, C, pw * ph * pt, w_len * h_len * t_len, N);
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));
            x = ggml_reshape_4d(ctx, x, pw, ph * pt, w_len, h_len * t_len * C * N);
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph, pt, h_len * t_len * C * N);
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));
            x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph * h_len, t_len * C * N);
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt * t_len, C * N);
            return x;
        }

        // x: [W, H, T, C]   (C => in_channels)
        // timestep: [T] (per-frame) or [1]
        // context: [N, L, caption_channels]
        // pe: 3D RoPE [2,2,axes_dim_sum/2, pos_len]
        // n_cond_tokens = cond frames * spatial tokens (the ai2v num_cond_latents
        //   split, applied per-block in self-attn + text cross-attn). 0 = no split.
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* audio        = nullptr,
                             int64_t n_cond_tokens = 0) {
            auto t_mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["t_embedder.mlp.0"]);
            auto t_mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["t_embedder.mlp.2"]);
            auto y_proj_0 = std::dynamic_pointer_cast<Linear>(blocks["y_embedder.y_proj.0"]);
            auto y_proj_2 = std::dynamic_pointer_cast<Linear>(blocks["y_embedder.y_proj.2"]);
            auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

            // Numerical-oracle taps. Only active when LONGCAT_DUMP_DIR is set
            // (the GGMLRunner debug loop dumps captured taps to .bin there);
            // gated so production builds add zero graph ops. Capture the raw DiT
            // inputs so the torch reference can be fed the EXACT same tensors.
            const bool dump = getenv("LONGCAT_DUMP_DIR") != nullptr;
            auto tap        = [&](const char* name, ggml_tensor* t) {
                if (dump) {
                    ctx->capture_tensor(name, t);
                }
            };
            tap("in_x", x);
            tap("in_timestep", timestep);
            tap("in_context", context);

            int64_t t_len = 0, h_len = 0, w_len = 0;
            x = patch_embed(ctx, x, t_len, h_len, w_len);  // [N=1, thw, hidden]
            tap("tap_patch_embed", x);

            int64_t T = t_len;

            // t_embedder: sinusoidal embedding over timesteps -> [n_ts, adaln].
            // The sampler passes a single (per-batch) timestep; PyTorch expands it
            // to [B, T] over the T latent frames, so we broadcast across T here.
            auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, params.frequency_embedding_size);  // [n_ts, freq]
            t_emb      = t_mlp_0->forward(ctx, t_emb);
            t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
            t_emb      = t_mlp_2->forward(ctx, t_emb);  // [n_ts, adaln]
            int64_t n_ts = t_emb->ne[1];
            if (n_ts == 1 && T > 1) {
                // broadcast single timestep across T frames -> [adaln, T]
                t_emb = ggml_repeat_4d(ctx->ggml_ctx, t_emb, t_emb->ne[0], T, 1, 1);
            }
            // reshape to [N=1, T, adaln]
            t_emb = ggml_reshape_3d(ctx->ggml_ctx, t_emb, t_emb->ne[0], T, 1);
            tap("tap_t_embed", t_emb);

            // Cond-frame K/V cache CONSUME (lap-26): at step>0 the cond frame's forward is
            // step-invariant and was cached at step 0, so process ONLY the noise tokens
            // here. Slice x / t_emb / pe / audio / T down to the noise frames; each block's
            // self_attn reattaches the cached cond K/V (consume branch). The cond output
            // rows are zero-padded back after final_layer (the sampler overwrites the cond
            // latent region with init_latent regardless, so exactness there is free).
            const bool cond_consume = ctx->cond_kv_cache && ctx->sampler_step > 1 &&
                                      n_cond_tokens > 0 && n_cond_tokens < x->ne[1] && n_per_frame > 0;
            int64_t consume_n_cond_tokens = 0;
            ggml_tensor* t_emb_full = t_emb;  // un-sliced t_emb + T for the final_layer (run on full tokens)
            int64_t T_full          = T;
            if (cond_consume) {
                int64_t n_cond_fr     = n_cond_tokens / n_per_frame;
                consume_n_cond_tokens = n_cond_tokens;
                x = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[1] - n_cond_tokens, x->ne[2],
                                                          x->nb[1], x->nb[2], x->nb[1] * n_cond_tokens));
                t_emb = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, t_emb, t_emb->ne[0], T - n_cond_fr, t_emb->ne[2],
                                                             t_emb->nb[1], t_emb->nb[2], t_emb->nb[1] * n_cond_fr));
                if (pe != nullptr) {
                    pe = ggml_cont(ctx->ggml_ctx, ggml_view_4d(ctx->ggml_ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->ne[3] - n_cond_tokens,
                                                              pe->nb[1], pe->nb[2], pe->nb[3], pe->nb[3] * n_cond_tokens));
                }
                if (audio != nullptr) {
                    audio = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, audio, audio->ne[0], audio->ne[1], audio->ne[2] - n_cond_fr,
                                                                 audio->nb[1], audio->nb[2], audio->nb[2] * n_cond_fr));
                }
                T             = T - n_cond_fr;
                n_cond_tokens = 0;  // blocks now see noise-only; self_attn consume branch supplies cond K/V
            }

            // y_embedder: Linear -> GELU(tanh) -> Linear
            context = y_proj_0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context, true);
            context = y_proj_2->forward(ctx, context);  // [N, L, hidden]
            tap("tap_y_embed", context);

            // Graph-cut boundaries (mirrors anima.hpp): mark the cross-block-persistent
            // inputs as a "prelude" group, then mark the residual `x` after every block.
            // This lets the GGMLRunner's graph-cut path reserve ONE block's activation
            // buffer (per-segment) instead of the whole 48-block graph's sum, and stream
            // only that block's weights to GPU. At 93 frames (~37k tokens) the monolithic
            // forward's compute buffer is ~13.3 GiB — over the 12 GB card even with zero
            // resident weights — so full-length renders REQUIRE these cuts (plus
            // --offload-to-cpu + --max-vram, which puts params on a different backend than
            // the runtime; graph-cut only engages then). 25f fits without cuts; the cuts
            // are inert unless the segmented path is taken (no extra ops, marks only).
            sd::ggml_graph_cut::mark_graph_cut(x, "longcat.prelude", "x");
            sd::ggml_graph_cut::mark_graph_cut(t_emb, "longcat.prelude", "t_emb");
            sd::ggml_graph_cut::mark_graph_cut(context, "longcat.prelude", "context");
            if (pe != nullptr) {
                sd::ggml_graph_cut::mark_graph_cut(pe, "longcat.prelude", "pe");
            }
            if (audio != nullptr) {
                sd::ggml_graph_cut::mark_graph_cut(audio, "longcat.prelude", "audio");
            }

            // FFN token-tiling: bound the SwiGLU's [11008, n_token] transient triple
            // at full length. Off (tiles=1, bit-identical single-shot) below a token
            // threshold so the 25f hot path is unchanged; LONGCAT_FFN_TILES overrides
            // (0/unset = auto: 1 below threshold, else a count that keeps each tile
            // ~10k tokens). The FFN is purely per-token so tiling is exact.
            int64_t n_token_total = x->ne[1];
            int64_t ffn_tiles     = 1;
            int64_t attn_tiles    = 1;
            {
                const char* fenv = getenv("LONGCAT_FFN_TILES");
                if (fenv != nullptr && atoi(fenv) > 0) {
                    ffn_tiles = atoi(fenv);
                } else if (n_token_total > 16000) {
                    // auto: target ~10k tokens/tile (matches the 25f single-shot extent
                    // that already fits comfortably).
                    ffn_tiles = (n_token_total + 9999) / 10000;
                } else if (ctx->cond_kv_cache && n_token_total > 2000) {
                    // The cond-K/V cache adds a ~1.23 GB persistent buffer; tile the FFN
                    // (bit-exact, per-token) to shrink the step compute buffer enough to fit
                    // it beside the 8.5 GB resident weights at 480p. tile=2 is the lightest
                    // that fits (tile=1 OOMs); measured fastest of {2,3,4} at 480/8-step.
                    ffn_tiles = 2;
                }
                // attn query-tiling is OPT-IN only (LONGCAT_ATTN_TILES): the
                // concat-accumulated output buffer grows per tile and gallocr can't
                // reuse it, so auto-tiling raised the peak (3629 -> 4494 MiB at 93f).
                // Kept as a knob; not auto-engaged.
                const char* aenv = getenv("LONGCAT_ATTN_TILES");
                if (aenv != nullptr && atoi(aenv) > 0) {
                    attn_tiles = atoi(aenv);
                }
            }

            for (int i = 0; i < params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<LongCatAvatarSingleStreamBlock>(blocks["blocks." + std::to_string(i)]);
                block->ffn_token_tiles  = ffn_tiles;
                block->attn_query_tiles = attn_tiles;
                block->audio_mouth_scale = audio_mouth_scale;
                block->num_ref_latents   = num_ref_latents;
                block->ref_img_index     = ref_img_index;
                block->mask_frame_range  = mask_frame_range;
                block->n_per_frame       = n_per_frame;
                x          = block->forward(ctx, x, t_emb, context, pe, audio, T, n_cond_tokens, i);
                sd::ggml_graph_cut::mark_graph_cut(x, "longcat.blocks." + std::to_string(i) + ".out", "x");
                if (i == 0) {
                    tap("tap_block0", x);
                } else if (i == 1) {
                    tap("tap_block1", x);
                }
            }

            if (cond_consume) {
                // Pad cond (zeros) BEFORE final_layer so it runs on the FULL token set: the
                // final output Linear is F16→cuBLAS (M-dependent FP, unlike the blocks' per-row
                // Q4_K MMQ), so running it on the noise-only 9360 rows drifted ~5e-5 from the
                // dense 10920-row path and compounded over steps. Matching M makes the noise
                // rows bit-exact; the cond rows (zero input) are garbage but the sampler
                // overwrites the cond latent with init_latent. Use the un-sliced t_emb/T.
                auto zeros = ggml_ext_zeros(ctx->ggml_ctx, x->ne[0], consume_n_cond_tokens, x->ne[2], 1);
                x          = ggml_concat(ctx->ggml_ctx, zeros, x, 1);  // [N, full_n_token, C]
            }
            x = final_layer->forward(ctx, x, cond_consume ? t_emb_full : t_emb, cond_consume ? T_full : T);  // [N, thw, num_patch*out_channels]
            tap("tap_final_layer", x);
            x = unpatchify(ctx->ggml_ctx, x, t_len, h_len, w_len);  // [N*C, T, H, W]
            tap("tap_output", x);
            return x;
        }
    };

    struct LongCatAvatarRunner : public GGMLRunner {
    public:
        std::string desc = "Longcat-Video-Avatar";
        LongCatAvatarParams avatar_params;
        LongCatAvatar avatar;
        std::vector<float> pe_vec;
        int num_cond_latents = 0;  // ai2v ref-image cond frames (set per request)
        // Video-continuation (generate_avc) reference-anchor params. num_ref_latents>0
        // selects the 3-way self-attn split (ref / cond / noise) + the ref-positioned
        // 3D-RoPE temporal grid (grid_t = [ref_img_index, 0,1,2,...]). 0 = the ai2v /
        // single-clip path (2-way cond/noise split, plain sequential PE) — UNTOUCHED.
        int num_ref_latents  = 0;   // leading ref-anchor latent frames (1 for the avatar)
        int ref_img_index    = 10;  // ref anchor's temporal grid position (reference default)
        int mask_frame_range = 3;   // noise-near-ref attention carve-out half-width (reference default)
        // Sampler step index (set per compute() by the diffusion-model wrapper). Used by
        // the cond-frame cross-step K/V cache (lap-26): the fixed cond frame's 48-block
        // forward is step-invariant, so step>0 can reuse step-0's per-block cond K/V.
        // -1 = unknown (cache disabled). Gated behind LONGCAT_COND_CACHE (default off).
        int cur_step = -1;
        // Direct persistent cond-K/V cache (lap-26). Allocated once (lazily) in its own
        // ctx+buffer on the runtime backend, so the per-step compute gallocr sees these as
        // pre-allocated leaves (like params). Sidesteps the graph-cut cache mechanism that
        // segfaults / returns garbage as a resident-graph leaf. Stores F16(k*kv_scale) /
        // F16(v*kv_scale): the prescale halves the footprint AND keeps v F16-overflow-safe.
        ggml_context*  condkv_ctx              = nullptr;
        ggml_backend_buffer_t condkv_buf       = nullptr;
        std::vector<ggml_tensor*> condkv_k_vec;  // [n_layers] each [d_head, n_cond, heads]
        std::vector<ggml_tensor*> condkv_v_vec;  // [n_layers] each [d_head, heads, n_cond, N]
        std::vector<ggml_tensor*> condkv_writes; // ggml_cpy nodes for the current graph
        int64_t condkv_ncond                   = -1;

        // LongCat lap-31: text cross-attn K/V cache. Same pattern as condkv: F16
        // prescaled, persistent on the runtime backend, registered with the runner's
        // persistent-tensor set so the segmented executor doesn't orphan the leaves.
        // Cached K is the post-norm, post-permute K (shape [hidden_size, n_ctx, Nb] F16
        // prescaled). Cached V is the post-permute V (same shape). Both align with what
        // text_cross_attn's downstream `ggml_ext_attention_ext` expects.
        ggml_context*  xattn_text_ctx          = nullptr;
        ggml_backend_buffer_t xattn_text_buf   = nullptr;
        std::vector<ggml_tensor*> xattn_text_k_vec;  // [n_layers] each [hidden_size, n_ctx, Nb] F16
        std::vector<ggml_tensor*> xattn_text_v_vec;  // [n_layers] each [hidden_size, n_ctx, Nb] F16
        int64_t xattn_text_nctx                = -1;

        // LongCat BSA: persistent host-built F16 mask tensor for the
        // consume-step self-attention. Computed once per resolution: each query token
        // attends to its 3x3 spatial-cube neighborhood + all tokens in the cond frame
        // (ref anchor). Quality trade, env-gated (LONGCAT_BSA=1) — default-off ⇒ dense.
        ggml_context*         bsa_ctx           = nullptr;
        ggml_backend_buffer_t bsa_buf           = nullptr;
        ggml_tensor*          bsa_mask_tensor   = nullptr;
        int64_t               bsa_cached_n_token = -1;
        int64_t               bsa_cached_n_noise = -1;

        void free_bsa_mask() {
            if (bsa_buf) { ggml_backend_buffer_free(bsa_buf); bsa_buf = nullptr; }
            if (bsa_ctx) { ggml_free(bsa_ctx); bsa_ctx = nullptr; }
            bsa_mask_tensor   = nullptr;
            bsa_cached_n_token = -1;
            bsa_cached_n_noise = -1;
        }

        // (Re)allocate the BSA mask if the shape (n_noise, n_token) changed, fill the
        // -INF/0 mask on the host, and upload. Cube structure [T=full, H=cube_h, W=cube_w]
        // with `radius` spatial-cube window. cond-frame tokens (k_t==0) are always allowed.
        // self_frame_anchor (lap-29-quality): if true, Q at time T also allows ALL K at
        // time T (full intra-frame attention) — addresses edge-cube asymmetry + time-blind
        // cross-temporal mixing that causes the "slow rotation" failure mode.
        // bookend_anchor: if true, additionally allow ALL K at the LAST frame.
        int  bsa_cached_self_frame  = -1;
        int  bsa_cached_bookend     = -1;
        int  bsa_cached_radius      = -1;
        void ensure_bsa_mask(int64_t n_noise, int64_t n_token,
                             int64_t n_per_frame, int64_t h_len, int64_t w_len,
                             int cube_h, int cube_w, int radius,
                             bool self_frame_anchor = false, bool bookend_anchor = false) {
            const int sfa_i = self_frame_anchor ? 1 : 0;
            const int bka_i = bookend_anchor ? 1 : 0;
            if (bsa_buf && bsa_cached_n_token == n_token && bsa_cached_n_noise == n_noise &&
                bsa_cached_self_frame == sfa_i && bsa_cached_bookend == bka_i &&
                bsa_cached_radius == radius) {
                return;  // already built for this shape + flags
            }
            free_bsa_mask();
            ggml_init_params p = { ggml_tensor_overhead() * 4 + 1024, nullptr, /*no_alloc=*/true };
            bsa_ctx = ggml_init(p);
            // Build the mask in F16 directly so the flash-attn wrapper's cast becomes
            // a no-op — otherwise each of 48 attn calls per step ran a 408 MB F32->F16
            // cast (~3 ms each), dominating any sparse-attention savings.
            bsa_mask_tensor = ggml_new_tensor_2d(bsa_ctx, GGML_TYPE_F16, n_token, n_noise);
            ggml_set_name(bsa_mask_tensor, "bsa_mask");
            bsa_buf            = ggml_backend_alloc_ctx_tensors(bsa_ctx, runtime_backend);
            bsa_cached_n_token = n_token;
            bsa_cached_n_noise = n_noise;

            // Build CPU-side. Token layout (per patch_embed): t outer, h middle, w inner.
            // Token offset n_token-n_noise = n_cond_tokens (cond frames precede noise).
            // F16 sentinels: -INF = 0xFC00, 0.0 = 0x0000.
            std::vector<ggml_fp16_t> data((size_t) n_noise * n_token);
            const ggml_fp16_t MASK_DENY  = 0xFC00;  // F16 -infinity
            const ggml_fp16_t MASK_ALLOW = 0x0000;  // F16 +zero
            const int64_t n_cond_local = n_token - n_noise;
            const int64_t T_total      = n_token / n_per_frame;
            const int64_t T_last       = T_total - 1;
            for (int64_t qi = 0; qi < n_noise; ++qi) {
                const int64_t q_orig    = n_cond_local + qi;
                const int64_t q_t       = q_orig / n_per_frame;
                const int64_t q_spatial = q_orig % n_per_frame;
                const int64_t q_h       = q_spatial / w_len;
                const int64_t q_w       = q_spatial % w_len;
                const int     q_cube_h  = (int) (q_h / cube_h);
                const int     q_cube_w  = (int) (q_w / cube_w);
                for (int64_t ki = 0; ki < n_token; ++ki) {
                    const int64_t k_spatial = ki % n_per_frame;
                    const int64_t k_t       = ki / n_per_frame;
                    const int64_t k_h       = k_spatial / w_len;
                    const int64_t k_w       = k_spatial % w_len;
                    const int     k_cube_h  = (int) (k_h / cube_h);
                    const int     k_cube_w  = (int) (k_w / cube_w);
                    bool allow = (std::abs(q_cube_h - k_cube_h) <= radius) &&
                                 (std::abs(q_cube_w - k_cube_w) <= radius);
                    if (k_t == 0) allow = true;  // ref-frame anchor (always on)
                    if (self_frame_anchor && k_t == q_t) allow = true;  // intra-frame anchor
                    if (bookend_anchor && k_t == T_last) allow = true;  // last-frame anchor
                    data[(size_t) qi * n_token + ki] = allow ? MASK_ALLOW : MASK_DENY;
                }
            }
            ggml_backend_tensor_set(bsa_mask_tensor, data.data(), 0, ggml_nbytes(bsa_mask_tensor));

            bsa_cached_self_frame = sfa_i;
            bsa_cached_bookend    = bka_i;
            bsa_cached_radius     = radius;
            LOG_INFO("[BSA] mask built: n_noise=%lld n_token=%lld cube=[%d,%d] radius=%d self_frame=%d bookend=%d mask=%.1f MiB",
                     (long long) n_noise, (long long) n_token, cube_h, cube_w, radius,
                     sfa_i, bka_i,
                     (double) ggml_nbytes(bsa_mask_tensor) / (1024.0 * 1024.0));
        }

        void free_condkv_cache() {
            if (condkv_buf) { ggml_backend_buffer_free(condkv_buf); condkv_buf = nullptr; }
            if (condkv_ctx) { ggml_free(condkv_ctx); condkv_ctx = nullptr; }
            condkv_k_vec.clear(); condkv_v_vec.clear(); condkv_ncond = -1;
        }

        // (Re)allocate the persistent cond-K/V tensors. No-op if already sized for n_cond.
        void ensure_condkv_cache(int64_t d_head, int64_t n_cond, int64_t heads, int64_t Nb, int n_layers) {
            if (condkv_buf != nullptr && condkv_ncond == n_cond) return;
            free_condkv_cache();
            ggml_init_params p = { ggml_tensor_overhead() * (size_t)(2 * n_layers + 8), nullptr, /*no_alloc=*/true };
            condkv_ctx = ggml_init(p);
            condkv_k_vec.resize(n_layers); condkv_v_vec.resize(n_layers);
            for (int i = 0; i < n_layers; ++i) {
                condkv_k_vec[i] = ggml_new_tensor_3d(condkv_ctx, GGML_TYPE_F16, d_head, n_cond, heads);
                condkv_v_vec[i] = ggml_new_tensor_4d(condkv_ctx, GGML_TYPE_F16, d_head, heads, n_cond, Nb);
                ggml_set_name(condkv_k_vec[i], ("condkv.k." + std::to_string(i)).c_str());
                ggml_set_name(condkv_v_vec[i], ("condkv.v." + std::to_string(i)).c_str());
            }
            condkv_buf  = ggml_backend_alloc_ctx_tensors(condkv_ctx, runtime_backend);
            condkv_ncond = n_cond;
        }

        void free_xattn_text_cache() {
            if (xattn_text_buf) { ggml_backend_buffer_free(xattn_text_buf); xattn_text_buf = nullptr; }
            if (xattn_text_ctx) { ggml_free(xattn_text_ctx); xattn_text_ctx = nullptr; }
            xattn_text_k_vec.clear(); xattn_text_v_vec.clear(); xattn_text_nctx = -1;
        }

        // (Re)allocate the persistent text x-attn K/V tensors. No-op if already sized for n_ctx.
        // hidden_size = head_dim * num_heads of the DiT (text-context is reprojected to hidden_size).
        void ensure_xattn_text_cache(int64_t hidden_size, int64_t n_ctx, int64_t Nb, int n_layers) {
            if (xattn_text_buf != nullptr && xattn_text_nctx == n_ctx) return;
            free_xattn_text_cache();
            ggml_init_params p = { ggml_tensor_overhead() * (size_t)(2 * n_layers + 8), nullptr, /*no_alloc=*/true };
            xattn_text_ctx = ggml_init(p);
            xattn_text_k_vec.resize(n_layers); xattn_text_v_vec.resize(n_layers);
            for (int i = 0; i < n_layers; ++i) {
                xattn_text_k_vec[i] = ggml_new_tensor_3d(xattn_text_ctx, GGML_TYPE_F16, hidden_size, n_ctx, Nb);
                xattn_text_v_vec[i] = ggml_new_tensor_3d(xattn_text_ctx, GGML_TYPE_F16, hidden_size, n_ctx, Nb);
                ggml_set_name(xattn_text_k_vec[i], ("xattn_text.k." + std::to_string(i)).c_str());
                ggml_set_name(xattn_text_v_vec[i], ("xattn_text.v." + std::to_string(i)).c_str());
            }
            xattn_text_buf = ggml_backend_alloc_ctx_tensors(xattn_text_ctx, runtime_backend);
            xattn_text_nctx = n_ctx;
        }
        // RUNTIME mouth-exaggeration knob (default 1.0 = unchanged / bit-identical).
        // Set per request (API field, or the CLI via LONGCAT_AUDIO_MOUTH_SCALE).
        float audio_mouth_scale = 1.0f;

        // LongCat lap-32.4: per-request BSA knob. Set by LongCatAvatarModel::compute
        // before each render; build_graph reads these directly. env LONGCAT_BSA*
        // overrides for back-compat with the env-driven bench script tests.
        bool bsa_enabled    = false;
        int  bsa_radius     = 1;
        bool bsa_self_frame = true;
        bool bsa_bookend    = false;
        int  bsa_cube_h     = 4;
        int  bsa_cube_w     = 6;
        // Audio window inputs (host-prepared per request via LONGCAT_AUDIO; empty =
        // no audio → audio cross-attn skipped, identical to text+image-only video).
        sd::Tensor<float> audio_first;   // [32000, 1]
        sd::Tensor<float> audio_latter;  // [51200, n_latter]
        SDVersion version;

        LongCatAvatarRunner(ggml_backend_t backend,
                            const String2TensorStorage& tensor_storage_map = {},
                            const std::string prefix                       = "",
                            SDVersion version                              = VERSION_LONGCAT_AVATAR,
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager), version(version) {
            int num_layers = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                size_t pos = tensor_name.find("blocks.");
                if (pos != std::string::npos) {
                    std::string sub = tensor_name.substr(pos);
                    auto items      = split_string(sub, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > num_layers) {
                            num_layers = block_index + 1;
                        }
                    }
                }
            }
            if (num_layers > 0) {
                avatar_params.num_layers = num_layers;
            }

            LOG_INFO("%s (%d blocks)", desc.c_str(), avatar_params.num_layers);

            avatar = LongCatAvatar(avatar_params);
            avatar.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return desc;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            avatar.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor = {}) {
            ggml_cgraph* gf = new_graph_custom(GGML_DEFAULT_GRAPH_SIZE * 16);

            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_tensor* context   = make_optional_input(context_tensor);

            // 3D RoPE positions over the patchified (t,h,w) grid. For the
            // video-continuation path (num_ref_latents>0) the leading ref-anchor
            // frame(s) get the FIXED temporal position ref_img_index and the rest get
            // 0,1,2,... (matches avatar/rope_3d.py grid_t = [ref_img_index, 0,1,...]);
            // otherwise the plain sequential grid (ai2v / single-clip, UNCHANGED).
            if (num_ref_latents > 0) {
                pe_vec = Rope::gen_wan_pe_ref(static_cast<int>(x->ne[2]),
                                              static_cast<int>(x->ne[1]),
                                              static_cast<int>(x->ne[0]),
                                              std::get<0>(avatar_params.patch_size),
                                              std::get<1>(avatar_params.patch_size),
                                              std::get<2>(avatar_params.patch_size),
                                              1,
                                              avatar_params.theta,
                                              avatar_params.axes_dim,
                                              num_ref_latents,
                                              ref_img_index);
            } else {
                pe_vec = Rope::gen_wan_pe(static_cast<int>(x->ne[2]),
                                          static_cast<int>(x->ne[1]),
                                          static_cast<int>(x->ne[0]),
                                          std::get<0>(avatar_params.patch_size),
                                          std::get<1>(avatar_params.patch_size),
                                          std::get<2>(avatar_params.patch_size),
                                          1,
                                          avatar_params.theta,
                                          avatar_params.axes_dim);
            }
            int pos_len = static_cast<int>(pe_vec.size() / avatar_params.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, avatar_params.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            // num_cond_latents split (ai2v). x is [W,H,T,C]; patch [1,ph,pw]. The
            // cond/noise split is now done via a two-pass attention in self_attn (no
            // O(n_token^2) mask), so it applies at any resolution — including 480p
            // (~10920 tokens) — within the VRAM budget and keeps flash-attn enabled.
            int64_t n_cond_tokens = 0;
            if (num_cond_latents > 0) {
                int64_t t_len       = x->ne[2];
                int64_t h_len       = x->ne[1] / std::get<1>(avatar_params.patch_size);
                int64_t w_len       = x->ne[0] / std::get<2>(avatar_params.patch_size);
                int64_t n_per_frame = h_len * w_len;
                int64_t n_token     = t_len * n_per_frame;
                n_cond_tokens       = std::min<int64_t>(num_cond_latents, t_len) * n_per_frame;
                if (n_cond_tokens <= 0 || n_cond_tokens >= n_token) {
                    n_cond_tokens = 0;  // nothing to split
                }
            }

            auto runner_ctx  = get_context();
            // LongCat lap-27: thread the cgraph into the context so modulate() can
            // pre-expand its small scale_bias node (otherwise it would land between
            // NORM and MUL in topo order and block the {NORM, MUL, ADD} autofusion).
            runner_ctx.gf = gf;
            // Cond-frame cross-step K/V cache. Shipped bit-exact in lap-26 (PSNR 99 +
            // 8% sampling-wall win); stacked with the lap-27.1 fused-modulate to ship
            // ~+9% combined. DEFAULT ON. Opt out via LONGCAT_NO_COND_CACHE=1.
            //
            // Works on BOTH the resident AND --offload-to-cpu paths since lap-27
            // P0 (commit f0b4bcc): the cpy writes are tagged into the block's
            // post_self_attn subcut group so the graph-cut planner attaches them
            // as additional outputs of an existing segment, and the persistent K/V
            // tensors are registered via register_persistent_tensor so the
            // segmented executor's reset_segment_runtime_tensors leaves their
            // buffer/data pointers intact between segments. Without those two
            // pieces in place the offload path's bind_segment_cached_inputs would
            // see zero-initialized leaves and PSNR collapse to ~12 dB.
            //
            // Requires flash-attn: cache stores F16(k*kv_scale) and kv_scale=1/256
            // is what keeps v F16-safe; the non-flash path (kv_scale=1) would
            // overflow.
            static const bool cond_cache_disabled_env = []{
                const char* s = getenv("LONGCAT_NO_COND_CACHE");
                return s && s[0] == '1';
            }();
            runner_ctx.sampler_step = cur_step;
            runner_ctx.cond_kv_cache = !cond_cache_disabled_env &&
                                       (num_cond_latents > 0) &&
                                       (num_ref_latents == 0) &&
                                       runner_ctx.flash_attn_enabled;
            condkv_writes.clear();
            // condkv_writes is the shared receiver for ALL persistent K/V cache writes
            // (lap-26 cond + lap-31 text x-attn). Set unconditionally so the text cache
            // can push without cond cache enabled.
            runner_ctx.cache_writes = &condkv_writes;
            if (runner_ctx.cond_kv_cache) {
                int64_t t_len_c = x->ne[2];
                int64_t h_len_c = x->ne[1] / std::get<1>(avatar_params.patch_size);
                int64_t w_len_c = x->ne[0] / std::get<2>(avatar_params.patch_size);
                int64_t npf_c   = h_len_c * w_len_c;
                int64_t ncond_c = std::min<int64_t>(num_cond_latents, t_len_c) * npf_c;
                int64_t d_head_c = avatar_params.hidden_size / avatar_params.num_heads;
                ensure_condkv_cache(d_head_c, ncond_c, avatar_params.num_heads, 1, (int)avatar_params.num_layers);
                runner_ctx.condkv_k = &condkv_k_vec;
                runner_ctx.condkv_v = &condkv_v_vec;

                // LongCat lap-28.4 / lap-32.4 (BSA): per-request, with env override.
                // Source of truth = runner fields set by LongCatAvatarModel::compute()
                // from sd_vid_gen_params_t. env LONGCAT_BSA* overrides any field that
                // is set, so the bench script and CLI test paths keep working unchanged.
                bool eff_bsa_enabled    = this->bsa_enabled;
                int  eff_bsa_radius     = this->bsa_radius;
                bool eff_bsa_self_frame = this->bsa_self_frame;
                bool eff_bsa_bookend    = this->bsa_bookend;
                int  eff_bsa_cube_h     = this->bsa_cube_h;
                int  eff_bsa_cube_w     = this->bsa_cube_w;
                if (const char* s = getenv("LONGCAT_BSA"))            { eff_bsa_enabled    = (s[0] == '1'); }
                if (const char* s = getenv("LONGCAT_BSA_RADIUS"); s && s[0]) { eff_bsa_radius = atoi(s); }
                if (const char* s = getenv("LONGCAT_BSA_SELF_FRAME")) { eff_bsa_self_frame = (s[0] == '1'); }
                if (const char* s = getenv("LONGCAT_BSA_BOOKEND"))    { eff_bsa_bookend    = (s[0] == '1'); }
                if (const char* s = getenv("LONGCAT_BSA_CUBE_H"); s && s[0]) { eff_bsa_cube_h = atoi(s); }
                if (const char* s = getenv("LONGCAT_BSA_CUBE_W"); s && s[0]) { eff_bsa_cube_w = atoi(s); }
                if (eff_bsa_enabled) {
                    int64_t n_token_b = t_len_c * npf_c;
                    int64_t n_noise_b = n_token_b - ncond_c;
                    if (n_noise_b > 0 && (h_len_c % eff_bsa_cube_h) == 0 && (w_len_c % eff_bsa_cube_w) == 0) {
                        ensure_bsa_mask(n_noise_b, n_token_b, npf_c, h_len_c, w_len_c,
                                        eff_bsa_cube_h, eff_bsa_cube_w, eff_bsa_radius,
                                        eff_bsa_self_frame, eff_bsa_bookend);
                        if (cur_step > 1) {
                            runner_ctx.bsa_mask = bsa_mask_tensor;
                        }
                    } else {
                        LOG_WARN("[BSA] disabled — latent h=%lld w=%lld not divisible by cube [%d,%d]",
                                 (long long) h_len_c, (long long) w_len_c, eff_bsa_cube_h, eff_bsa_cube_w);
                    }
                }
            }

            // LongCat lap-31.1: text cross-attn K/V cache. text_cross_attn.kv_linear(context)
            // is step-invariant (umT5 runs once before sampling), so post-norm K and V
            // are byte-identical every step. Persist at step <=1; consume at step >1
            // (skip kv_linear, k_norm, permute+cont). Measured -0.5s wall, bit-exact.
            // F16 prescaled by kv_scale=1/256 (same exact-exponent shift pattern as
            // lap-28.2 cond cache). Requires --diffusion-fa.
            //
            // DEFAULT ON for prod. Opt out via env LONGCAT_NO_XATTN_TEXT_CACHE=1
            // (or LONGCAT_XATTN_TEXT_CACHE=0). Auto-disables when BSA is enabled
            // AND we're on the resident path (params_backend == runtime_backend),
            // since DiT 8.5 + cond_kv 1.2 + BSA 195 MiB + xattn 384 MiB + compute
            // > 12 GiB at --max-vram 9 would OOM. The prod offload path keeps params
            // on CPU so this stack fits comfortably — xattn stays on there.
            static const bool xattn_text_cache_explicit_off = []{
                const char* s = getenv("LONGCAT_NO_XATTN_TEXT_CACHE");
                if (s && s[0] == '1') return true;
                const char* s2 = getenv("LONGCAT_XATTN_TEXT_CACHE");
                return s2 && s2[0] == '0';
            }();
            static const bool xattn_text_cache_explicit_on = []{
                const char* s = getenv("LONGCAT_XATTN_TEXT_CACHE");
                return s && s[0] == '1';
            }();
            // lap-32.4: BSA may be enabled via API field OR env override. Re-derive
            // the "effective bsa_enabled" the same way build_graph just did above
            // so the OOM-avoidance auto-disable tracks both paths.
            bool bsa_enabled_for_xattn = this->bsa_enabled;
            if (const char* s = getenv("LONGCAT_BSA")) {
                bsa_enabled_for_xattn = (s[0] == '1');
            }
            // ModelManager owns parameter residency; do not infer it from the old
            // dual-backend constructor. Retain the cache unless the user explicitly
            // disables it or the continuation path needs the VRAM headroom.
            const bool would_oom_in_bsa  = bsa_enabled_for_xattn;
            // LongCat continuation OOM fix (regression from lap-32.3 flipping the xattn
            // text cache default-ON). A continuation segment (ref-anchor present) is
            // inherently larger than an equivalent single clip: it prepends an extra
            // ref-anchor latent frame (T = base + num_ref) and runs the 3-way
            // ref/cond/noise self-attn split — both enlarge the compute buffer (resident
            // path) and the graph-cut cross-segment cache_buffer (offload path). The
            // 384 MiB persistent xattn-text K/V cache is the straw that pushes seg>=1
            // over 12 GiB on BOTH paths at 480p (verified: 25f 2-seg OOMs resident at
            // the seg-2 compute-buffer reserve AND offload at the graph-cut cache_buffer;
            // disabling it lets seg-2 fit on both — matching the lap-21 sweet-spot
            // numbers from before the cache was default-on). The cond-K/V cache is
            // already auto-off for continuation (it requires num_ref_latents==0).
            // Single-clip (num_ref_latents==0) keeps the cache → zero perf change on the
            // hot path / bench. Mirrors the would_oom_in_bsa auto-disable above; the
            // explicit LONGCAT_XATTN_TEXT_CACHE=1 override still forces it on (footgun).
            const bool is_continuation_segment = (num_ref_latents > 0 && n_cond_tokens > 0);
            const bool would_oom = would_oom_in_bsa || is_continuation_segment;
            const bool xattn_text_cache_active = !xattn_text_cache_explicit_off &&
                                                 (xattn_text_cache_explicit_on || !would_oom) &&
                                                 runner_ctx.flash_attn_enabled &&
                                                 context != nullptr;
            if (xattn_text_cache_active) {
                int64_t n_ctx_t = context->ne[1];
                int64_t Nb_t    = context->ne[2];
                int     n_layers_t = (int) avatar_params.num_layers;
                int64_t hidden  = avatar_params.hidden_size;
                ensure_xattn_text_cache(hidden, n_ctx_t, Nb_t, n_layers_t);
                runner_ctx.xattn_text_k = &xattn_text_k_vec;
                runner_ctx.xattn_text_v = &xattn_text_v_vec;
            }

            // Audio path: run AudioProjModel on the host-windowed inputs to produce
            // audio_hidden_states [768, 32, N_t], threaded into every block's audio
            // cross-attn. Skipped when no audio was provided.
            ggml_tensor* audio_hidden = nullptr;
            if (!audio_first.empty()) {
                ggml_tensor* a_first  = make_input(audio_first);
                ggml_tensor* a_latter = audio_latter.empty() ? nullptr : make_input(audio_latter);
                audio_hidden          = avatar.audio_proj(&runner_ctx, a_first, a_latter);  // [768, 32, N_t]

                // Video-continuation ref-frame audio prepend + trim (generate_avc;
                // longcat_video_dit_avatar.py L438-441). When a ref-anchor latent frame
                // is prepended to the latent ([ref, cond_tail, noise]), the audio must
                // shift to match: duplicate audio frame 0 for the ref slot, then trim to
                // the latent T (= x->ne[2]). Without this, the ref slot consumes a real
                // audio frame and every noise frame is driven by audio shifted ~1 latent
                // frame early (seam lip-sync drift). Only when continuation ref split is
                // active (num_ref_latents>0 AND there is a cond split).
                if (num_ref_latents > 0 && n_cond_tokens > 0) {
                    int64_t Da = audio_hidden->ne[0];
                    int64_t Sa = audio_hidden->ne[1];
                    // audio_start_ref = audio_hidden[:, [0], :, :] -> duplicate frame 0.
                    auto a_ref = ggml_view_3d(runner_ctx.ggml_ctx, audio_hidden, Da, Sa, 1,
                                              audio_hidden->nb[1], audio_hidden->nb[2], 0);
                    a_ref      = ggml_cont(runner_ctx.ggml_ctx, a_ref);
                    // cat([audio_start_ref, audio_hidden], dim=frame) then trim to last T.
                    audio_hidden       = ggml_concat(runner_ctx.ggml_ctx, a_ref, audio_hidden, 2);
                    int64_t T_latent   = x->ne[2];  // latent frames incl. ref (= 25)
                    int64_t N_audio    = audio_hidden->ne[2];
                    if (N_audio > T_latent) {
                        int64_t off  = N_audio - T_latent;  // keep the LAST T_latent
                        audio_hidden = ggml_view_3d(runner_ctx.ggml_ctx, audio_hidden, Da, Sa, T_latent,
                                                    audio_hidden->nb[1], audio_hidden->nb[2],
                                                    audio_hidden->nb[2] * off);
                        audio_hidden = ggml_cont(runner_ctx.ggml_ctx, audio_hidden);
                    }
                }
            }

            // RUNTIME mouth-exaggeration knob. Source: the runner field (set by the
            // API) unless overridden by LONGCAT_AUDIO_MOUTH_SCALE (the CLI path). Read
            // per-render so it stays runtime (never baked); 1.0 = bit-identical.
            avatar.audio_mouth_scale = audio_mouth_scale;
            if (const char* mse = getenv("LONGCAT_AUDIO_MOUTH_SCALE")) {
                avatar.audio_mouth_scale = (float)atof(mse);
            }

            // Video-continuation 3-way-split params. n_ref_tokens = the leading
            // ref-anchor frames' token span; n_per_frame is the spatial-token stride
            // used to map the mask_frame_range carve-out (in frames) to token offsets.
            // 0 ref latents (ai2v / single-clip) leaves the existing 2-way split intact.
            {
                int64_t h_len_g       = x->ne[1] / std::get<1>(avatar_params.patch_size);
                int64_t w_len_g       = x->ne[0] / std::get<2>(avatar_params.patch_size);
                avatar.n_per_frame    = h_len_g * w_len_g;
                avatar.num_ref_latents  = (num_ref_latents > 0 && n_cond_tokens > 0) ? num_ref_latents : 0;
                avatar.ref_img_index    = ref_img_index;
                avatar.mask_frame_range = mask_frame_range;
            }

            ggml_tensor* out = avatar.forward(&runner_ctx, x, timesteps, context, pe, audio_hidden, n_cond_tokens);

            // LongCat flow-match sign convention: the DiT predicts the flow velocity
            // with the OPPOSITE sign to what the FlowMatchEuler scheduler (and sd.cpp's
            // DiscreteFlowDenoiser: denoised = x - sigma*v, euler v = raw output)
            // expects. EVERY reference pipeline (pipeline_longcat_video{,_avatar}.py)
            // does `noise_pred = -noise_pred  # negate for scheduler compatibility`
            // immediately before `scheduler.step`. Wan does NOT negate, so the shared
            // sd.cpp sampler doesn't either — the avatar must negate its own output.
            // Without this the (masked) cond frame stays correct but the generated
            // frames denoise in the wrong direction → latent std blows up to ~3x → noise.
            out = ggml_scale(runner_ctx.ggml_ctx, out, -1.0f);

            // Cond-K/V cache writes (lap-26): ggml_cpy nodes storing the step-invariant cond
            // K/V into the persistent buffer at step 1. Expand them FIRST (they're off the main
            // output path) so that `out` is expanded LAST and remains the final graph node —
            // the runner reads the output as ggml_graph_node(gf, -1).
            for (ggml_tensor* w : condkv_writes) {
                ggml_build_forward_expand(gf, w);
            }
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context = {}) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }
    };

}  // namespace LONGCAT_AVATAR

// Diffusion-model wrapper (was in diffusion_model.hpp pre-#1569; moved here to match
// upstream sd.cpp's per-model-header layout). Subclasses DiffusionModelRunner.
struct LongCatAvatarModel : public DiffusionModelRunner {
    LONGCAT_AVATAR::LongCatAvatarRunner avatar;
    // Video-continuation (generate_avc) reference-anchor params, set per-render by the
    // generate_video path before sampling. num_ref_latents>0 selects the 3-way self-attn
    // split + ref-positioned PE in the runner; 0 = ai2v / single-clip (UNCHANGED).
    int cont_num_ref_latents  = 0;
    int cont_ref_img_index    = 10;
    int cont_mask_frame_range = 3;
    // LongCat lap-32.4: per-request BSA knob. Set per generate_video call from
    // sd_vid_gen_params_t.bsa_*; the runner consults these in build_graph each
    // render (env LONGCAT_BSA* override for back-compat with the bench script).
    // Defaults match sd_vid_gen_params_init: r=1+self_frame preset, off by default.
    bool bsa_enabled    = false;
    int  bsa_radius     = 1;
    bool bsa_self_frame = true;
    bool bsa_bookend    = false;
    int  bsa_cube_h     = 4;
    int  bsa_cube_w     = 6;

    LongCatAvatarModel(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map = {},
                       const std::string prefix                       = "model.diffusion_model",
                       SDVersion version                              = VERSION_LONGCAT_AVATAR,
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : DiffusionModelRunner(backend, prefix, weight_manager),
          avatar(backend, tensor_storage_map, prefix, version, weight_manager) {
    }

    std::string get_desc() override {
        return avatar.get_desc();
    }

    void runner_done() override {
        avatar.runner_done();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
        avatar.get_param_tensors(tensors, prefix);
    }

    int64_t get_adm_in_channels() {  // not a base method; plain helper (not virtual-dispatched)
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) override {
        avatar.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        avatar.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_stream_layers_enabled(bool enabled) override {
        avatar.set_stream_layers_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        avatar.set_circular_axes(circular_x, circular_y);
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        avatar.set_weight_adapter(adapter);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        // Derive num_cond_latents (ai2v ref-image frames) from the per-frame
        // timesteps: the pipeline forces the cond frame(s) to timestep 0 while the
        // generated frames carry the current step's t (>0). Counting leading-zero
        // frames recovers num_cond_latents without extra plumbing.
        int num_cond_latents     = 0;
        const auto& ts           = *diffusion_params.timesteps;
        const float* ts_data     = ts.data();
        int64_t n_ts             = ts.numel();
        if (n_ts > 1 && ts_data != nullptr) {
            for (int64_t i = 0; i < n_ts; ++i) {
                if (ts_data[i] == 0.0f) {
                    num_cond_latents++;
                } else {
                    break;
                }
            }
            // only treat as a cond split if SOME frames are non-zero (real video)
            if (num_cond_latents == n_ts) {
                num_cond_latents = 0;
            }
        }
        avatar.num_cond_latents = num_cond_latents;
        // Thread the continuation ref-anchor params onto the runner. Only engage the
        // 3-way split when there IS a cond split (num_cond_latents>0); otherwise the
        // ai2v / single-clip path must stay on the 2-way / plain-PE route.
        avatar.num_ref_latents  = (num_cond_latents > 0) ? cont_num_ref_latents : 0;
        avatar.ref_img_index    = cont_ref_img_index;
        avatar.mask_frame_range = cont_mask_frame_range;
        int lc_step = -1;
        if (const auto* lc_extra = std::get_if<LongCatAvatarDiffusionExtra>(&diffusion_params.extra)) {
            lc_step = lc_extra->step;
        }
        avatar.cur_step         = lc_step;  // cond-frame K/V cache (lap-26)
        // LongCat lap-32.4: thread per-request BSA params to the runner. env
        // LONGCAT_BSA* still override these inside build_graph for back-compat.
        avatar.bsa_enabled    = bsa_enabled;
        avatar.bsa_radius     = bsa_radius;
        avatar.bsa_self_frame = bsa_self_frame;
        avatar.bsa_bookend    = bsa_bookend;
        avatar.bsa_cube_h     = bsa_cube_h;
        avatar.bsa_cube_w     = bsa_cube_w;
        return avatar.compute(n_threads,
                              *diffusion_params.x,
                              *diffusion_params.timesteps,
                              tensor_or_empty(diffusion_params.context));
    }
};


#endif  // __LONGCAT_AVATAR_HPP__
