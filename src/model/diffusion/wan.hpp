#ifndef __SD_MODEL_DIFFUSION_WAN_HPP__
#define __SD_MODEL_DIFFUSION_WAN_HPP__

#include <map>
#include <memory>
#include <utility>

#include "model/common/block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model/diffusion/wan_sla.hpp"
#include "model_loader.h"

namespace WAN {

    // 20480: the S2V causal block graph holds the noisy block forward + the cond/sink
    // prefill (a FULL ref forward through all 40 blocks: norm1+mod+self-attn+cross+ffn)
    // + 40 layers of K/V graph outputs. That overflows the old 10240 node budget.
    constexpr int WAN_GRAPH_SIZE = 20480;

    struct WanConfig {
        std::string model_type                 = "t2v";
        std::tuple<int, int, int> patch_size   = {1, 2, 2};
        int64_t text_len                       = 512;
        int64_t in_dim                         = 16;
        int64_t dim                            = 2048;
        int64_t ffn_dim                        = 8192;
        int freq_dim                           = 256;
        int64_t text_dim                       = 4096;
        int64_t out_dim                        = 16;
        int64_t num_heads                      = 16;
        int num_layers                         = 32;
        int vace_layers                        = 0;
        int64_t vace_in_dim                    = 96;
        std::map<int, int> vace_layers_mapping = {};
        bool qk_norm                           = true;
        bool cross_attn_norm                   = true;
        float eps                              = 1e-6f;
        int64_t flf_pos_embed_token_number     = 0;
        int theta                              = 10000;
        // wan2.1 1.3B: 1536/12, wan2.1/2.2 14B: 5120/40, wan2.2 5B: 3074/24
        std::vector<int> axes_dim = {44, 42, 42};
        int64_t axes_dim_sum      = 128;

        static WanConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            WanConfig config;
            config.num_layers = 0;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                size_t pos = name.find("vace_blocks.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > config.vace_layers) {
                            config.vace_layers = block_index + 1;
                        }
                    }
                    continue;
                }
                pos = name.find("blocks.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > config.num_layers) {
                            config.num_layers = block_index + 1;
                        }
                    }
                    continue;
                }
                if (name.find("img_emb") != std::string::npos) {
                    config.model_type = "i2v";
                }
                if (name.find("img_emb.emb_pos") != std::string::npos) {
                    config.flf_pos_embed_token_number = 514;
                }
            }
            LOG_DEBUG("wan: model_type = %s, num_layers = %d, vace_layers = %d, dim = %" PRId64 ", ffn_dim = %" PRId64 ", num_heads = %" PRId64,
                      config.model_type.c_str(),
                      config.num_layers,
                      config.vace_layers,
                      config.dim,
                      config.ffn_dim,
                      config.num_heads);
            return config;
        }
    };

    class WanSelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t head_dim;

    public:
        WanSelfAttention(int64_t dim,
                         int64_t num_heads,
                         bool qk_norm = true,
                         float eps    = 1e-6)
            : num_heads(num_heads) {
            head_dim    = dim / num_heads;
            blocks["q"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["k"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["v"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["o"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));

            if (qk_norm) {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new Identity());
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        virtual ggml_tensor* forward(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* pe,
                                     ggml_tensor* mask = nullptr) {
            // x: [N, n_token, dim]
            // pe: [n_token, d_head/2, 2, 2]
            // return [N, n_token, dim]
            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            // WAN_ATTN_KV_SR half (1) — default OFF => byte-identical. Under the F16 residual
            // stream (WAN_DIT_F16) the K/V nvfp4 linears store-round their output to F16 at the
            // GEMM, and (with WAN_ROPE_F16) K stays F16 through RoPE — so K/V reach the flash
            // kernel already F16, and that rounding bias is rope-phase-locked and sums COHERENTLY
            // over the temporal axis (the frame-count grid). Feed the K/V projections an F32
            // activation so the mm_dst gate (ggml_extend.hpp: F16-dst only when x is F16) emits an
            // F32 dst — the exact nvfp4 path used when WAN_DIT_F16 is off — keeping K/V full
            // precision into build_kqv, where the same env applies stochastic rounding to F16.
            // Q is left F16 (per-query; it does not accumulate across keys). Only the K/V
            // projection INPUT is widened (a local F32 transient), not the whole residual stream.
            static const bool wan_attn_kv_sr = (std::getenv("WAN_ATTN_KV_SR") != nullptr);
            ggml_tensor* x_kv = x;
            if (wan_attn_kv_sr && x->type == GGML_TYPE_F16) {
                x_kv = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }
            auto k = k_proj->forward(ctx, x_kv);
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, x_kv);  // [N, n_token, n_head*d_head]

            // WAN_DIT_F16: under the F16 residual stream the q/k/v Linears emit F16, but
            // the fast fused RoPE (ggml_rope_pe) is F32-only (rope.hpp:953) — an F16 q
            // would fall back to the slow cont+repeat+mul+add chain (big intermediates,
            // the lap-08b smell). Upcast q/k to F32 here so the fused RoPE fires; this is
            // the F32-cast LTX applies for the same reason. The cuDNN flash kernel casts
            // q→F16 internally and, with GGML_CUDNN_ATTN_F16_OUT, stores an F16 output
            // that feeds the F16 o_proj. v is re-cast to F16 inside ggml_ext_attention_ext
            // (build_kqv), so leave it F16. Self-gated on the F16 type → no-op in the
            // default F32 path (byte-identical).
            // WAN_ROPE_F16 (opt-in, default OFF): under WAN_DIT_F16 keep q/k F16 through
            // the fused RoPE instead of upcasting to F32. The fused ggml_rope_pe kernel now
            // has an F16 path (rope-pe.cu) that computes the rotation in F32 and rounds the
            // store to F16 — bit-identical to the F32-rope-then-cast-to-F16 that this path
            // did downstream anyway (k via ggml_cast to F16 for flash, q via cuDNN's internal
            // pool cast). The win is memory: the two full-size F32 rope tensors (2x1237 MB at
            // 1280x704x65f) never enter the compute buffer, and the redundant k->F16 copy is
            // dropped (flash's cast is a no-op on already-F16 k). cuDNN flash accepts F16 Q
            // directly (fattn-cudnn.cu:213; fattn.cu:446 selects on K/V type only). Requires
            // GGML_CUDNN_ATTN (prod always on) — native ggml flash asserts F32 Q. Self-gated on
            // the F16 type; default path unchanged (byte-identical). Owner eye-test is the gate.
            static const bool wan_rope_f16_env = (std::getenv("WAN_ROPE_F16") != nullptr);
            // DEVICE-GATED (Blackwell-only). WAN_ROPE_F16 keeps q/k F16 through the fused
            // RoPE and hands an F16 Q to flash attention — which ONLY cuDNN SDPA accepts,
            // and cuDNN is selected exclusively on cc >= Blackwell. On non-Blackwell (e.g.
            // sm86/RTX 3060) a native ggml flash kernel runs and hard-asserts Q->type==F32
            // (fattn-common.cuh) -> abort. Fall back to the F32 q/k cast below so wan runs
            // on the 3060; the 5060/cuDNN path is unchanged. (CPU build: off.)
#ifdef SD_USE_CUDA
            const bool wan_rope_f16 = wan_rope_f16_env && ggml_backend_cuda_device_has_blackwell_mma(0);
#else
            const bool wan_rope_f16 = false;
#endif
            if (q->type == GGML_TYPE_F16 && !wan_rope_f16) {
                q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_F32);
                k = ggml_cast(ctx->ggml_ctx, k, GGML_TYPE_F32);
            }

            // WAN SLA (Stage 0): the selector block exports mean-pooled (64-block),
            // RoPE'd, head-kept Q and smooth-K so the host can build the next step's
            // content-based skip bitmap (src/model/diffusion/wan_sla.hpp). Faithful to
            // lightx2v sla_util.get_block_map: smooth-k = k - mean_token(k); pool = mean
            // over each 64-token block; both AFTER RoPE (the attention input). Tiny
            // ([d_head, n_blk, n_head]); written into persistent dsts, read back post-step.
            // q/k here are F32 (the upcast above), so the fused RoPE fires. Default OFF.
            if (ctx->sla_capture_now && ctx->sla_pooled_q_dst != nullptr &&
                ctx->sla_pooled_k_dst != nullptr && ctx->gf != nullptr && N == 1) {
                auto    gc   = ctx->ggml_ctx;
                const int blk = ctx->sla_blk;
                const int64_t nqb = (n_token + blk - 1) / blk;
                // RoPE q/k as [d_head, n_head, n_token, 1] → [d_head, n_token, n_head].
                auto q4 = ggml_reshape_4d(gc, q, head_dim, num_heads, n_token, 1);
                auto k4 = ggml_reshape_4d(gc, k, head_dim, num_heads, n_token, 1);
                auto qr = Rope::apply_rope(gc, q4, pe, true, ctx->allow_fused_rope);  // [d_head, n_token, n_head]
                auto kr = Rope::apply_rope(gc, k4, pe, true, ctx->allow_fused_rope);
                // smooth-k: kr - mean over tokens (per d_head, per head).
                auto kperm = ggml_cont(gc, ggml_permute(gc, kr, 1, 0, 2, 3));        // [n_token, d_head, n_head]
                auto kmean = ggml_mean(gc, kperm);                                   // [1, d_head, n_head]
                kmean      = ggml_reshape_3d(gc, kmean, head_dim, num_heads, 1);      // [d_head, n_head, 1]
                kmean      = ggml_cont(gc, ggml_permute(gc, kmean, 0, 2, 1, 3));      // [d_head, 1, n_head]
                auto ksm   = ggml_sub(gc, kr, kmean);                                // broadcast over tokens
                // mean-pool over 64-token blocks (pad the tail block with zeros).
                auto pool = [&](ggml_tensor* t) -> ggml_tensor* {
                    t = ggml_cont(gc, t);                                            // RoPE/sub output → ensure contiguous
                    int64_t pad = nqb * blk - n_token;
                    if (pad > 0) t = ggml_pad(gc, t, 0, (int)pad, 0, 0);             // [d_head, nqb*blk, n_head]
                    t = ggml_reshape_4d(gc, t, head_dim, blk, nqb, num_heads);       // [d_head, blk, nqb, n_head]
                    t = ggml_cont(gc, ggml_permute(gc, t, 1, 0, 2, 3));              // [blk, d_head, nqb, n_head]
                    t = ggml_mean(gc, t);                                            // [1, d_head, nqb, n_head]
                    return ggml_reshape_3d(gc, t, head_dim, nqb, num_heads);         // [d_head, nqb, n_head]
                };
                auto wq = ggml_cpy(gc, pool(qr),  ctx->sla_pooled_q_dst);
                auto wk = ggml_cpy(gc, pool(ksm), ctx->sla_pooled_k_dst);
                // CRITICAL: the graph-cut/offload executor only runs nodes backward-reachable
                // from a subcut-MARKED tensor or the final output — it ignores GGML_TENSOR_FLAG_OUTPUT.
                // These off-residual pooled-export cpys feed neither, so without a mark they belong
                // to NO segment and never execute → persistent pooled_q/k stay zero → selector scores
                // all-equal → a fixed content-independent block selection (the "coherent-but-wrong"
                // SLA bug). Share the selector block's existing "wan.blocks.<sel>" group (line ~921)
                // so they become extra outputs of that segment — no new segment. Mirrors the proven
                // avatar cond-K/V write (longcat_avatar.hpp mark_graph_cut). wq is a view of the
                // persistent pooled_q dst, so the cpy lands in the buffer update_from_pooled reads.
                const std::string sla_grp = "wan.blocks." + std::to_string(ctx->sla_selector_block);
                sd::ggml_graph_cut::mark_graph_cut(wq, sla_grp, "sla_pooled_q");
                sd::ggml_graph_cut::mark_graph_cut(wk, sla_grp, "sla_pooled_k");
                ggml_build_forward_expand(ctx->gf, wq);
                ggml_build_forward_expand(ctx->gf, wk);
            }

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]

            // flash_skip_kv_pad when there's no real mask: the legacy L_k->256 pad otherwise
            // synthesizes a [L_k_pad x L_q] -inf mask (O(n_token^2): ~22 GB at 1280x704x81),
            // which the modern CUDA flash kernel doesn't need (it handles unpadded L_k).
            x = Rope::attention(ctx, q, k, v, pe, mask, /*kv_scale=*/1.0f, /*rope_interleaved=*/true,
                                /*flash_attn=*/true, /*flash_skip_kv_pad=*/mask == nullptr);  // [N, n_token, dim]

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        // Causal KV-cache self-attention (LiveAvatar streaming). Projects x's tokens,
        // RoPE-applies q/k with `pe` (this block's grid), then attends over the
        // concatenated [prev ++ cur ++ cond] keys/values. Exports the current block's
        // RoPE'd K and raw V (both [d_head, L, n_head]) via new_kc/new_vc so the caller
        // can persist them in a host rolling cache. prev_*/cond_* may be null.
        ggml_tensor* forward_kv_cache(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,        // [1, L_blk, dim]
                                      ggml_tensor* pe,       // RoPE for L_blk tokens
                                      ggml_tensor* prev_kc,  // [d_head, L_prev, n_head] or null
                                      ggml_tensor* prev_vc,
                                      ggml_tensor* cond_kc,  // [d_head, L_cond, n_head] or null
                                      ggml_tensor* cond_vc,
                                      ggml_tensor*& new_kc,
                                      ggml_tensor*& new_vc) {
            int64_t L_blk = x->ne[1];
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = norm_q->forward(ctx, q_proj->forward(ctx, x));
            auto k = norm_k->forward(ctx, k_proj->forward(ctx, x));
            auto v = v_proj->forward(ctx, x);
            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, L_blk, 1);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, L_blk, 1);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, L_blk, 1);

            auto q_roped = Rope::apply_rope(ctx->ggml_ctx, q, pe, true, ctx->allow_fused_rope);  // [d_head, L_blk, n_head]
            auto k_roped = Rope::apply_rope(ctx->ggml_ctx, k, pe, true, ctx->allow_fused_rope);  // [d_head, L_blk, n_head]
            auto v_flat  = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));  // [d_head, L_blk, n_head]

            new_kc = k_roped;
            new_vc = v_flat;

            ggml_tensor* k_all = k_roped;
            ggml_tensor* v_all = v_flat;
            // PERF (shotstream floor deep-dive) — CONCAT IN F16, not F32.
            // The prev/cond caches are ALREADY F16 (host make_input tensors or the resident
            // chunk buffers). The legacy path cast the WHOLE growing cache F16->F32 (`to_f32`),
            // concatenated in F32, then build_kqv cast the whole k_all/v_all F32->F16 for the
            // flash kernel — i.e. the entire [prev++cur++cond] cache round-tripped F16->F32->F16
            // EVERY forward (nsys: convert_unary<half,float> ~5k launches + an F32 concat at 2x
            // the bytes). Instead cast only the NEW chunk (small, L_blk tokens) F32->F16 and
            // concat in F16; the cache tensors stay their exact F16 bytes (no conversion), and
            // build_kqv's redundant-cast guard then feeds the F16 K/V straight to flash (no
            // F32->F16 of the whole cache). BYTE-IDENTICAL: the flash kernel already consumed
            // F16 K/V; the new chunk still gets exactly one F32->F16 rounding (previously
            // F32->F32(concat)->F16, same result); cached tokens were F16 and stay bit-for-bit.
            // Halves the concat traffic + the cache's per-forward compute-buffer footprint.
            // Env opt-out SHOTSTREAM_KV_F32_CONCAT=1 restores the legacy F32 concat for A/B.
            static const bool ss_f32_concat = (std::getenv("SHOTSTREAM_KV_F32_CONCAT") != nullptr);
            if (!ss_f32_concat && (prev_kc != nullptr || cond_kc != nullptr)) {
                auto to_f16 = [&](ggml_tensor* t) -> ggml_tensor* {
                    return (t != nullptr && t->type != GGML_TYPE_F16)
                               ? ggml_cast(ctx->ggml_ctx, t, GGML_TYPE_F16)
                               : t;
                };
                k_all = ggml_cast(ctx->ggml_ctx, k_roped, GGML_TYPE_F16);
                v_all = ggml_cast(ctx->ggml_ctx, v_flat, GGML_TYPE_F16);
                if (prev_kc != nullptr) {
                    k_all = ggml_concat(ctx->ggml_ctx, to_f16(prev_kc), k_all, 1);
                    v_all = ggml_concat(ctx->ggml_ctx, to_f16(prev_vc), v_all, 1);
                }
                if (cond_kc != nullptr) {
                    k_all = ggml_concat(ctx->ggml_ctx, k_all, to_f16(cond_kc), 1);
                    v_all = ggml_concat(ctx->ggml_ctx, v_all, to_f16(cond_vc), 1);
                }
            } else {
                // Legacy F32 concat (opt-out, or the no-cache first forward where k_all stays F32).
                auto to_f32 = [&](ggml_tensor* t) -> ggml_tensor* {
                    return (t != nullptr && t->type != GGML_TYPE_F32)
                               ? ggml_cast(ctx->ggml_ctx, t, GGML_TYPE_F32)
                               : t;
                };
                if (prev_kc != nullptr) {
                    k_all = ggml_concat(ctx->ggml_ctx, to_f32(prev_kc), k_all, 1);
                    v_all = ggml_concat(ctx->ggml_ctx, to_f32(prev_vc), v_all, 1);
                }
                if (cond_kc != nullptr) {
                    k_all = ggml_concat(ctx->ggml_ctx, k_all, to_f32(cond_kc), 1);
                    v_all = ggml_concat(ctx->ggml_ctx, v_all, to_f32(cond_vc), 1);
                }
            }
            int64_t L_k = k_all->ne[1];

            auto vv = ggml_reshape_4d(ctx->ggml_ctx,
                                      ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v_all, 0, 2, 1, 3)),
                                      head_dim, num_heads, L_k, 1);  // [d_head, n_head, L_k, 1]
            // Flash-attention over the [prev ++ cur ++ cond] cache. With FA OFF the
            // L_blk x L_k scores tensor is materialized — at 480x832 (L_blk~4680,
            // L_k grows to ~9k) that is several GB and is THE causal compute-buffer
            // peak (measured 5.5 GB), which blocks keeping the Q4_K weights resident
            // on the 12 GB card. FA streams the scores so the buffer drops to a few
            // hundred MB. No causal mask: each block attends ALL cached keys (full
            // attention over the rolling cache), so mask=nullptr is correct. Gated by
            // the runner's flash flag so S2V_NO_FLASH=1 still selects the FA-off path.
            //
            // PERF (shotstream floor deep-dive): pass flash_skip_kv_pad=true (matches the
            // bidirectional WanSelfAttention::forward, which already does this). Without it
            // the wrapper pads L_k->256-multiple AND synthesizes a [L_k_pad x L_q] -inf F16
            // mask on EVERY self-attn call (nsys: pad_f32 + a ~220 MB mask tensor + its cast,
            // ~30 blocks x every forward) — pure waste, since the modern MMA/WMMA flash kernel
            // handles an unpadded L_k with no mask (the bidirectional path proves it). Result
            // is numerically identical (padded keys had a -inf mask => 0 softmax weight anyway).
            // Env opt-out SHOTSTREAM_KV_PAD=1 restores the legacy pad+mask path for A/B.
            static const bool ss_kv_pad = (std::getenv("SHOTSTREAM_KV_PAD") != nullptr);
            auto attn = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend,
                                               q_roped, k_all, vv, num_heads, nullptr,
                                               /*skip_reshape=*/true,
                                               /*flash_attn=*/ctx->flash_attn_enabled,
                                               /*kv_scale=*/1.0f,
                                               /*flash_skip_kv_pad=*/!ss_kv_pad);  // [1, L_blk, dim]
            attn = o_proj->forward(ctx, attn);
            return attn;
        }

        // Cond-prefill: just compute (and export) RoPE'd K + raw V for the ref/cond
        // tokens x [1, L_cond, dim] using cond RoPE `pe_cond`. No attention output is
        // consumed (the sink only fills the cond cache).
        void prefill_cond_kv(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe_cond,
                             ggml_tensor*& kc, ggml_tensor*& vc) {
            int64_t L = x->ne[1];
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);
            auto k = norm_k->forward(ctx, k_proj->forward(ctx, x));
            auto v = v_proj->forward(ctx, x);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, L, 1);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, L, 1);
            kc = Rope::apply_rope(ctx->ggml_ctx, k, pe_cond, true, ctx->allow_fused_rope);  // [d_head, L, n_head]
            vc = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));      // [d_head, L, n_head]
        }
    };

    class WanCrossAttention : public WanSelfAttention {
    public:
        WanCrossAttention(int64_t dim,
                          int64_t num_heads,
                          bool qk_norm = true,
                          float eps    = 1e-6)
            : WanSelfAttention(dim, num_heads, qk_norm, eps) {}
        virtual ggml_tensor* forward(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* context,
                                     int64_t context_img_len) = 0;
    };

    class WanT2VCrossAttention : public WanCrossAttention {
    public:
        WanT2VCrossAttention(int64_t dim,
                             int64_t num_heads,
                             bool qk_norm = true,
                             float eps    = 1e-6)
            : WanCrossAttention(dim, num_heads, qk_norm, eps) {}
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             int64_t context_img_len) override {
            // x: [N, n_token, dim]
            // context: [N, n_context, dim]
            // context_img_len: unused
            // return [N, n_token, dim]
            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, context);  // [N, n_context, dim]
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, context);  // [N, n_context, dim]

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, dim]

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }
    };

    class WanI2VCrossAttention : public WanCrossAttention {
    public:
        WanI2VCrossAttention(int64_t dim,
                             int64_t num_heads,
                             bool qk_norm = true,
                             float eps    = 1e-6)
            : WanCrossAttention(dim, num_heads, qk_norm, eps) {
            blocks["k_img"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["v_img"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));

            if (qk_norm) {
                blocks["norm_k_img"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_k_img"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             int64_t context_img_len) override {
            // x: [N, n_token, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);

            auto k_img_proj = std::dynamic_pointer_cast<Linear>(blocks["k_img"]);
            auto v_img_proj = std::dynamic_pointer_cast<Linear>(blocks["v_img"]);

            auto norm_q     = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k     = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);
            auto norm_k_img = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k_img"]);

            int64_t N               = x->ne[2];
            int64_t n_token         = x->ne[1];
            int64_t dim             = x->ne[0];
            int64_t context_txt_len = context->ne[1] - context_img_len;

            auto context_img = ggml_view_3d(ctx->ggml_ctx, context, dim, context_img_len, N, context->nb[1], context->nb[2], 0);                                 // [N, context_img_len, dim]
            auto context_txt = ggml_view_3d(ctx->ggml_ctx, context, dim, context_txt_len, N, context->nb[1], context->nb[2], context_img_len * context->nb[1]);  // [N, context_txt_len, dim]

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, context_txt);  // [N, context_txt_len, dim]
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, context_txt);  // [N, context_txt_len, dim]

            auto k_img = k_img_proj->forward(ctx, context_img);  // [N, context_img_len, dim]
            k_img      = norm_k_img->forward(ctx, k_img);
            auto v_img = v_img_proj->forward(ctx, context_img);  // [N, context_img_len, dim]

            auto img_x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k_img, v_img, num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, dim]
            x          = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, ctx->flash_attn_enabled);          // [N, n_token, dim]

            x = ggml_add(ctx->ggml_ctx, x, img_x);

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }
    };

    static ggml_tensor* modulate_add(ggml_context* ctx, ggml_tensor* x, ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        // WAN_DIT_F16=0 (F32 residual) guard: the general F32 binbcast asserts nb10 % 4 == 0
        // (binbcast.cu). Normalize the broadcast gate to a fresh 4-aligned F32 buffer under the
        // F32 stream; gated on x being F32 so the prod F16 path is byte-identical (no cont node).
        if (x->type == GGML_TYPE_F32) {
            e = ggml_cont(ctx, e);
        }
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_add(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_add(ctx, x, e);
        }
        return x;
    }

    static ggml_tensor* modulate_mul(ggml_context* ctx, ggml_tensor* x, ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        // WAN_DIT_F16=0 guard (see modulate_add): normalize the F32 broadcast gate to a
        // 4-aligned buffer; no-op/byte-identical on the prod F16 stream.
        if (x->type == GGML_TYPE_F32) {
            e = ggml_cont(ctx, e);
        }
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_mul(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_mul(ctx, x, e);
        }
        return x;
    }

    class WanAttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 6, 1);
        }

    public:
        WanAttentionBlock(bool t2v_cross_attn,
                          int64_t dim,
                          int64_t ffn_dim,
                          int64_t num_heads,
                          bool qk_norm         = true,
                          bool cross_attn_norm = false,
                          float eps            = 1e-6)
            : dim(dim) {
            blocks["norm1"]     = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new WanSelfAttention(dim, num_heads, qk_norm, eps));
            if (cross_attn_norm) {
                blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, true));
            } else {
                blocks["norm3"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
            if (t2v_cross_attn) {
                blocks["cross_attn"] = std::shared_ptr<GGMLBlock>(new WanT2VCrossAttention(dim, num_heads, qk_norm, eps));
            } else {
                blocks["cross_attn"] = std::shared_ptr<GGMLBlock>(new WanI2VCrossAttention(dim, num_heads, qk_norm, eps));
            }

            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));

            blocks["ffn.0"] = std::shared_ptr<GGMLBlock>(new Linear(dim, ffn_dim));
            // ffn.1 is nn.GELU(approximate='tanh')
            blocks["ffn.2"] = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, dim));
        }

        virtual ggml_tensor* forward(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* e,
                                     ggml_tensor* pe,
                                     ggml_tensor* context,
                                     int64_t context_img_len = 257) {
            // x: [N, n_token, dim]
            // e: [N, 6, dim] or [N, T, 6, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]

            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);  // [N, 6, dim] or [N, T, 6, dim]
            // WAN_DIT_F16_MOD (opt-in, default OFF): under the F16 residual stream
            // (WAN_DIT_F16) carry the adaLN modulation gates (the scale/shift/gate es[*]
            // chunks below) in F16 too, so modulate_mul/modulate_add against the F16
            // residual become uniform F16xF16 instead of F16(x)xF32(gate). With the gates
            // F16 the broadcast mul+add fold fires natively in F16 (mul_add_bcast MOD=__half,
            // matcher accepts an F16 gate when the big operand is F16 + GGML_CUDA_F16_BCAST_FUSE
            // is on). The modulation ADD above stays F32 (tiny [6*dim], byte-identical); only
            // the chunked gates are downcast — one small F32->F16 cast per block. Self-gated
            // on x being F16 so the default F32 path and the WAN_DIT_F16-without-this path are
            // unchanged. NOT bit-identical vs F32 gates (F16 modulate arithmetic) — eye-tested.
            // Range-safe: adaLN scale/shift/gate are O(1), well inside F16's 65504 (no overflow).
            static const bool wan_dit_f16_mod = (std::getenv("WAN_DIT_F16_MOD") != nullptr);
            if (wan_dit_f16_mod && x->type == GGML_TYPE_F16) {
                e = ggml_cast(ctx->ggml_ctx, e, GGML_TYPE_F16);
            }
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);  // ([N, 1, dim], ...) or [N, T, 1, dim]

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);

            // self-attention
            auto y = norm1->forward(ctx, x);
            y      = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[1]));
            y      = modulate_add(ctx->ggml_ctx, y, es[0]);
            y      = self_attn->forward(ctx, y, pe);

            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[2]));

            // cross-attention
            x = ggml_add(ctx->ggml_ctx,
                         x,
                         cross_attn->forward(ctx, norm3->forward(ctx, x), context, context_img_len));

            // ffn
            y = norm2->forward(ctx, x);
            y = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[4]));
            y = modulate_add(ctx->ggml_ctx, y, es[3]);

            // Token-tiled FFN (env LONGCAT_FFN_TILE_TOKENS=N, N>0): the FFN is position-wise
            // (ffn_0 Linear -> GELU -> ffn_2 Linear, no token mixing), so process tokens in
            // chunks of N to cap the [ffn_dim, tokens] intermediate — the dominant DiT
            // activation at long video lengths (e.g. 1671 MB at [13824, 63360] f16) — to
            // [ffn_dim, N]. Lossless (same math, concatenated in chunk order), no extra FLOPs
            // (only a few more kernel launches). Off by default (N<=0) so every other path is
            // byte-identical. Only the simple 2D case (single batch) is tiled; anything else
            // falls through to the original path. Mirrors FeedForward::forward in
            // src/model/common/block.hpp. WanAttentionBlock::forward is reused by
            // VaceWanAttentionBlock, so the vace path is covered too.
            int64_t ffn_tile = 0;
            if (const char* env = getenv("LONGCAT_FFN_TILE_TOKENS")) {
                ffn_tile = atoll(env);
            }
            const int64_t n_tok = y->ne[1];
            if (ffn_tile > 0 && n_tok > ffn_tile && y->ne[2] == 1 && y->ne[3] == 1) {
                // Growing-concat accumulate: lower peak VRAM than a preallocated [dim,n_tok]
                // scatter (gallocr reuses the chunk buffers; the scatter kept a full-size `out`
                // + a head-concat temp resident = +1.3 GB, a bad trade under the <=11.5GB cap).
                // Same math, same chunk order -> byte-identical.
                ggml_tensor* out = nullptr;
                for (int64_t c = 0; c < n_tok; c += ffn_tile) {
                    const int64_t len = (n_tok - c < ffn_tile) ? (n_tok - c) : ffn_tile;
                    ggml_tensor* yc = ggml_view_2d(ctx->ggml_ctx, y, y->ne[0], len, y->nb[1], (size_t)c * y->nb[1]);
                    yc              = ggml_cont(ctx->ggml_ctx, yc);
                    yc              = ffn_0->forward(ctx, yc);  // [ffn_dim, len]
                    yc              = ggml_ext_gelu(ctx->ggml_ctx, yc, true);
                    yc              = ffn_2->forward(ctx, yc);  // [dim, len]
                    out             = (out == nullptr) ? yc : ggml_concat(ctx->ggml_ctx, out, yc, 1);
                }
                y = out;
            } else {
                y = ffn_0->forward(ctx, y);
                y = ggml_ext_gelu(ctx->ggml_ctx, y, true);
                y = ffn_2->forward(ctx, y);
            }

            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[5]));

            return x;
        }

        // ShotStream causal block forward. Identical to forward() except the self-attn
        // is routed through WanSelfAttention::forward_kv_cache so this chunk attends over
        // [prev_local_KV ++ this_chunk_KV ++ context_KV] (dual-cache streaming). The
        // block's freshly-computed RoPE'd K + raw V are exported via new_kc/new_vc so the
        // host runner can persist them into the rolling local cache. prev_*/cond_* may be
        // null (empty caches = single 3-frame chunk with no history = bidirectional over
        // its own 4680 tokens, structurally causal by cache contents).
        //   e is per-chunk [N,6,dim] (3 latent frames share one timestep) → es[*] are
        //   [dim,1,1] and the modulate_* broadcast over all L_blk tokens (a no-op T path).
        ggml_tensor* forward_causal(GGMLRunnerContext* ctx,
                                    ggml_tensor* x,        // [1, L_blk, dim]
                                    ggml_tensor* e,        // [N,6,dim] modulation (pre-modulation-add)
                                    ggml_tensor* pe,       // RoPE for L_blk tokens (this chunk)
                                    ggml_tensor* context,  // text [N, ctx, dim]
                                    ggml_tensor* prev_kc,  // [d_head, L_prev, n_head] or null (local cache)
                                    ggml_tensor* prev_vc,
                                    ggml_tensor* cond_kc,  // [d_head, L_ctx, n_head] or null (global cache)
                                    ggml_tensor* cond_vc,
                                    ggml_tensor*& new_kc,  // OUT: this chunk's RoPE'd K
                                    ggml_tensor*& new_vc,  // OUT: this chunk's raw V
                                    int64_t context_img_len = 0) {
            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);  // [N,6,dim]
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);

            // self-attention (causal, dual-KV-cache)
            auto y = norm1->forward(ctx, x);
            y      = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[1]));
            y      = modulate_add(ctx->ggml_ctx, y, es[0]);
            y      = self_attn->forward_kv_cache(ctx, y, pe, prev_kc, prev_vc, cond_kc, cond_vc, new_kc, new_vc);
            x      = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[2]));

            // cross-attention (text; each chunk's frames attend their shot's caption)
            x = ggml_add(ctx->ggml_ctx, x,
                         cross_attn->forward(ctx, norm3->forward(ctx, x), context, context_img_len));

            // ffn
            y = norm2->forward(ctx, x);
            y = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[4]));
            y = modulate_add(ctx->ggml_ctx, y, es[3]);
            y = ffn_0->forward(ctx, y);
            y = ggml_ext_gelu(ctx->ggml_ctx, y, true);
            y = ffn_2->forward(ctx, y);
            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[5]));

            return x;
        }

        // Parity probe (ShotStream block-0 oracle): run ONLY the causal self-attn sub-op on
        // a caller-supplied [1,L_blk,dim] input (standing in for the modulated normed hidden
        // state, exactly like the torch oracle's block0_selfattn_input), with empty caches.
        // Reuses the production forward_kv_cache verbatim; exports this chunk's RoPE'd K + raw
        // V for the block0_roped_k / block0_v goldens. Returns block0_selfattn_out (post-o).
        ggml_tensor* selfattn_only(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe,
                                   ggml_tensor*& new_kc, ggml_tensor*& new_vc) {
            auto self_attn = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            return self_attn->forward_kv_cache(ctx, x, pe, nullptr, nullptr, nullptr, nullptr, new_kc, new_vc);
        }
    };

    class VaceWanAttentionBlock : public WanAttentionBlock {
    protected:
        int block_id;
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 6, 1);
        }

    public:
        VaceWanAttentionBlock(bool t2v_cross_attn,
                              int64_t dim,
                              int64_t ffn_dim,
                              int64_t num_heads,
                              bool qk_norm         = true,
                              bool cross_attn_norm = false,
                              float eps            = 1e-6,
                              int block_id         = 0)
            : WanAttentionBlock(t2v_cross_attn, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps), block_id(block_id) {
            if (block_id == 0) {
                blocks["before_proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            }
            blocks["after_proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* c,
                                                      ggml_tensor* x,
                                                      ggml_tensor* e,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* context,
                                                      int64_t context_img_len = 257) {
            // x: [N, n_token, dim]
            // e: [N, 6, dim] or [N, T, 6, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]
            if (block_id == 0) {
                auto before_proj = std::dynamic_pointer_cast<Linear>(blocks["before_proj"]);

                c = before_proj->forward(ctx, c);
                // Under WAN_DIT_F16 the main residual stream `x` (== x_orig) is F16 while the vace
                // side-stream `c` stays F32. binbcast's <float,float,float> branch (F32 src0/dst)
                // rejects an F16 src1 (nb10 % 4 != 0 -> binbcast.cu:261 assert), so upcast x to F32
                // for this add. No-op when x is already F32 (default path = byte-identical); keeps
                // the vace side-stream in F32 (only the big main stream gets the F16 win).
                ggml_tensor* x_add = (x->type == GGML_TYPE_F16) ? ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32) : x;
                c = ggml_add(ctx->ggml_ctx, c, x_add);
            }

            auto after_proj = std::dynamic_pointer_cast<Linear>(blocks["after_proj"]);

            c           = WanAttentionBlock::forward(ctx, c, e, pe, context, context_img_len);
            auto c_skip = after_proj->forward(ctx, c);

            return {c_skip, c};
        }
    };

    class Head : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 2, 1);
        }

    public:
        Head(int64_t dim,
             int64_t out_dim,
             std::tuple<int, int, int> patch_size,
             float eps = 1e-6)
            : dim(dim) {
            out_dim = out_dim * std::get<0>(patch_size) * std::get<1>(patch_size) * std::get<2>(patch_size);

            blocks["norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["head"] = std::shared_ptr<GGMLBlock>(new Linear(dim, out_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* e) {
            // x: [N, n_token, dim]
            // e: [N, dim] or [N, T, dim]
            // return [N, n_token, out_dim]

            auto modulation = params["modulation"];
            e               = ggml_reshape_4d(ctx->ggml_ctx, e, e->ne[0], 1, e->ne[1], e->ne[2]);  // [N, 1, dim] or [N, T, 1, dim]
            e               = ggml_repeat_4d(ctx->ggml_ctx, e, e->ne[0], 2, e->ne[2], e->ne[3]);   // [N, 2, dim] or [N, T, 2, dim]

            e       = ggml_add(ctx->ggml_ctx, e, modulation);  // [N, 2, dim] or [N, T, 2, dim]
            auto es = ggml_ext_chunk(ctx->ggml_ctx, e, 2, 1);  // ([N, 1, dim], ...) or ([N, T, 1, dim], ...)

            auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto head = std::dynamic_pointer_cast<Linear>(blocks["head"]);

            x = norm->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, x, es[1]));
            x = modulate_add(ctx->ggml_ctx, x, es[0]);
            x = head->forward(ctx, x);
            return x;
        }
    };

    class MLPProj : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t flf_pos_embed_token_number;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            if (flf_pos_embed_token_number > 0) {
                params["emb_pos"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, flf_pos_embed_token_number, 1);
            }
        }

    public:
        MLPProj(int64_t in_dim,
                int64_t out_dim,
                int64_t flf_pos_embed_token_number = 0)
            : in_dim(in_dim), flf_pos_embed_token_number(flf_pos_embed_token_number) {
            blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new LayerNorm(in_dim));
            blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, in_dim));
            // proj.2 is nn.GELU()
            blocks["proj.3"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, out_dim));
            blocks["proj.4"] = std::shared_ptr<GGMLBlock>(new LayerNorm(out_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* image_embeds) {
            if (flf_pos_embed_token_number > 0) {
                auto emb_pos = params["emb_pos"];

                auto a = ggml_ext_slice(ctx->ggml_ctx, image_embeds, 1, 0, emb_pos->ne[1]);
                auto b = ggml_ext_slice(ctx->ggml_ctx, emb_pos, 1, 0, image_embeds->ne[1]);

                image_embeds = ggml_add(ctx->ggml_ctx, a, b);
            }

            auto proj_0 = std::dynamic_pointer_cast<LayerNorm>(blocks["proj.0"]);
            auto proj_1 = std::dynamic_pointer_cast<Linear>(blocks["proj.1"]);
            auto proj_3 = std::dynamic_pointer_cast<Linear>(blocks["proj.3"]);
            auto proj_4 = std::dynamic_pointer_cast<LayerNorm>(blocks["proj.4"]);

            auto x = proj_0->forward(ctx, image_embeds);
            x      = proj_1->forward(ctx, x);
            x      = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            x      = proj_3->forward(ctx, x);
            x      = proj_4->forward(ctx, x);

            return x;  // clip_extra_context_tokens
        }
    };

    class Wan : public GGMLBlock {
    protected:
        WanConfig config;

    public:
        Wan() {}
        Wan(WanConfig config)
            : config(config) {
            // patch_embedding
            blocks["patch_embedding"] = std::shared_ptr<GGMLBlock>(new Conv3d(config.in_dim, config.dim, config.patch_size, config.patch_size));

            // text_embedding
            blocks["text_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(config.text_dim, config.dim));
            // text_embedding.1 is nn.GELU()
            blocks["text_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(config.dim, config.dim));

            // time_embedding
            blocks["time_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(config.freq_dim, config.dim));
            // time_embedding.1 is nn.SiLU()
            blocks["time_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(config.dim, config.dim));

            // time_projection.0 is nn.SiLU()
            blocks["time_projection.1"] = std::shared_ptr<GGMLBlock>(new Linear(config.dim, config.dim * 6));

            // blocks
            for (int i = 0; i < config.num_layers; i++) {
                auto block                            = std::shared_ptr<GGMLBlock>(new WanAttentionBlock(config.model_type == "t2v",
                                                                                                         config.dim,
                                                                                                         config.ffn_dim,
                                                                                                         config.num_heads,
                                                                                                         config.qk_norm,
                                                                                                         config.cross_attn_norm,
                                                                                                         config.eps));
                blocks["blocks." + std::to_string(i)] = block;
            }

            // head
            blocks["head"] = std::shared_ptr<GGMLBlock>(new Head(config.dim, config.out_dim, config.patch_size, config.eps));

            // img_emb
            if (config.model_type == "i2v") {
                blocks["img_emb"] = std::shared_ptr<GGMLBlock>(new MLPProj(1280, config.dim, config.flf_pos_embed_token_number));
            }

            // vace
            if (config.vace_layers > 0) {
                for (int i = 0; i < config.vace_layers; i++) {
                    auto block                                 = std::shared_ptr<GGMLBlock>(new VaceWanAttentionBlock(config.model_type == "t2v",
                                                                                                                      config.dim,
                                                                                                                      config.ffn_dim,
                                                                                                                      config.num_heads,
                                                                                                                      config.qk_norm,
                                                                                                                      config.cross_attn_norm,
                                                                                                                      config.eps,
                                                                                                                      i));
                    blocks["vace_blocks." + std::to_string(i)] = block;
                }

                int step = config.num_layers / config.vace_layers;
                int n    = 0;
                for (int i = 0; i < config.num_layers; i += step) {
                    this->config.vace_layers_mapping[i] = n;
                    n++;
                }

                blocks["vace_patch_embedding"] = std::shared_ptr<GGMLBlock>(new Conv3d(config.vace_in_dim, config.dim, config.patch_size, config.patch_size));
            }
        }

        ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                       ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];

            int pad_t = (std::get<0>(config.patch_size) - T % std::get<0>(config.patch_size)) % std::get<0>(config.patch_size);
            int pad_h = (std::get<1>(config.patch_size) - H % std::get<1>(config.patch_size)) % std::get<1>(config.patch_size);
            int pad_w = (std::get<2>(config.patch_size) - W % std::get<2>(config.patch_size)) % std::get<2>(config.patch_size);
            ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, pad_t, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return x;
        }

        ggml_tensor* unpatchify(ggml_context* ctx,
                                ggml_tensor* x,
                                int64_t t_len,
                                int64_t h_len,
                                int64_t w_len) {
            // x: [N, t_len*h_len*w_len, pt*ph*pw*C]
            // return: [N*C, t_len*pt, h_len*ph, w_len*pw]
            int64_t N  = x->ne[3];
            int64_t pt = std::get<0>(config.patch_size);
            int64_t ph = std::get<1>(config.patch_size);
            int64_t pw = std::get<2>(config.patch_size);
            int64_t C  = x->ne[0] / pt / ph / pw;

            GGML_ASSERT(C * pt * ph * pw == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, C, pw * ph * pt, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, pt*ph*pw, C]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
            x = ggml_reshape_4d(ctx, x, pw, ph * pt, w_len, h_len * t_len * C * N);  // [N*C*t_len*h_len, w_len, pt*ph, pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt*ph, w_len, pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph, pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt, ph, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, ph, pt, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph * h_len, t_len * C * N);  // [N*C*t_len, h_len*ph, pt, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, pt, h_len*ph, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt * t_len, C * N);  // [N*C, t_len*pt, h_len*ph, w_len*pw]
            return x;
        }

        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe,
                                  ggml_tensor* clip_fea     = nullptr,
                                  ggml_tensor* vace_context = nullptr,
                                  float vace_strength       = 1.f,
                                  int64_t N                 = 1) {
            // x: [N*C, T, H, W], C => in_dim
            // vace_context: [N*vace_in_dim, T, H, W]
            // timestep: [N,] or [T]
            // context: [N, L, text_dim]
            // return: [N, t_len*h_len*w_len, out_dim*pt*ph*pw]

            GGML_ASSERT(N == 1);

            auto patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);

            auto text_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto text_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);

            auto time_embedding_0  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto time_embedding_2  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);

            auto head = std::dynamic_pointer_cast<Head>(blocks["head"]);

            // patch_embedding
            x = patch_embedding->forward(ctx, x);                                                    // [N*dim, t_len, h_len, w_len]
            // WAN_DUMP_BLOCKS (grid-divergence detector): capture the DiT token-grid dims from
            // the patch-embed output ([w_len, h_len, t_len, N*dim]) BEFORE the flatten below, so
            // the per-block dump hook can slice one representative frame's [w_len,h_len] plane.
            // Default OFF => these are just read into locals, never used (byte-identical prod).
            const int64_t wan_dump_w_len = x->ne[0];
            const int64_t wan_dump_h_len = x->ne[1];
            const int64_t wan_dump_t_len = x->ne[2];
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3] / N, N);  // [N, dim, t_len*h_len*w_len]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [N, t_len*h_len*w_len, dim]

            // WAN_DIT_F16 (default OFF, prod byte-identical when unset): run the DiT
            // residual stream in F16 to halve the per-block glue/copy/cast HBM traffic
            // (the nsys wall of k_bin_bcast<op_add> 5754-call residual adds, the
            // <op_mul> AdaLN modulates, the convert_unary<half,float> casts) and to feed
            // the cuDNN F16 attention output (GGML_CUDNN_ATTN_F16_OUT) straight into the
            // next Linear with no re-upcast. Mirrors LTX_DIT_F16 (ltxv.hpp:1738). The
            // NVFP4 Linears emit F16 here (cuBLASLt FP4 GEMM: F32 accumulate, F16 store —
            // supports_op gate ggml-cuda.cu:6046, mm_dst gate ggml_extend.hpp:1149);
            // e0/context/pe stay F32 and broadcast into the F16 stream via the
            // F16,F32->F16 binbcast combos (binbcast.cu:378). Cast back to F32 before the
            // head (below) so the model output / sampler / VAE path is unchanged.
            // Only the residual `x` is cast; e0/context stay F32 and mix into x through
            // adds whose F16 `x` is src0 (dst type = F16). VACE is excluded: the VACE
            // block's `ggml_add(c, x_orig)` would mix an F32 `c` (src0) with the F16
            // x_orig (src1) → the <float,float,float> binbcast branch on an F16 src1 =
            // the binbcast.cu:261 stride assert. The prod i2v/t2v path has vace_layers==0,
            // so this just scopes F16 to the supported (and target) path.
            static const bool wan_dit_f16_env = (std::getenv("WAN_DIT_F16") != nullptr);
            // DEVICE-GATED (Blackwell-only). The F16 residual stream feeds the FP4 cuBLASLt
            // GEMM / cuDNN SDPA (Blackwell). On non-Blackwell (sm86/3060) it hands F16 q to
            // the non-RoPE attentions, which reach the native flash kernel (cuDNN is
            // Blackwell-only) and abort on GGML_ASSERT(Q->type==F32) (fattn-common.cuh). Fall
            // back to the well-tested F32 stream so wan runs on the 3060. (CPU build: off.)
#ifdef SD_USE_CUDA
            static const bool wan_dit_f16 = wan_dit_f16_env && ggml_backend_cuda_device_has_blackwell_mma(0);
#else
            static const bool wan_dit_f16 = false;
#endif
            // VACE is now supported: the before_proj add site (above) upcasts the F16 main stream
            // to F32 for the one collision, so the F16 residual win applies to the main stream of
            // both i2v/t2v (vace_layers==0) AND VACE continuation.
            const bool use_dit_f16        = wan_dit_f16;
            if (use_dit_f16 && x->type == GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F16);
            }

            // ── WAN_DUMP_BLOCKS: per-block grid-divergence detector ─────────────────────────
            // Default OFF (env unset) => dump_plane is a no-op that adds ZERO graph nodes, so
            // prod is byte-identical. When WAN_DUMP_BLOCKS=<dir> is set, after each transformer
            // block (and at the DiT input + head, to bracket the whole DiT) we slice ONE
            // representative frame from the residual token grid, channel-mean it to a compact
            // [w_len,h_len] F32 plane, and register it as a debug tap. The tap is snapshotted +
            // read back after graph compute by the existing capture_tensor/debug_tensors path
            // (ggml_extend.hpp), which writes each plane to <dir>/<name>.bin (int64 ndim, then
            // ndim int64 dims in ggml ne order [w_len,h_len], then f32 data, w fastest).
            // tools/grid_divergence.py FFTs the planes to pinpoint the block that injects the
            // mesh. One [w_len,h_len] plane per block (a few KB) so 40 blocks x steps can't OOM.
            // Optional WAN_DUMP_FRAME=<t> picks the frame (default = middle; negative = from end).
            static const bool wan_dump_blocks = (std::getenv("WAN_DUMP_BLOCKS") != nullptr);
            static int wan_dump_call_counter  = 0;
            const int wan_dump_step           = wan_dump_blocks ? wan_dump_call_counter++ : 0;
            auto wan_pad = [](int v, int width) {
                std::string s = std::to_string(v);
                while ((int)s.size() < width) s = "0" + s;
                return s;
            };
            // cut_group: the graph-cut segment group this plane rides in. Under offload the
            // DiT graph is SPLIT into segments (compute_with_graph_cuts), and build_segment_graph
            // rebuilds each segment from ONLY its cut-marked output nodes (ggml_graph_cut.cpp) —
            // a plain output-flagged tap is dropped from every segment graph, so the per-segment
            // dump never sees it (the "0 .bin files" bug). We therefore mark each plane as an
            // extra cut-output of a segment. Block planes reuse the block's OWN existing cut group
            // ("wan.blocks.<i>", see below) so they add ZERO segments; input/head get tiny own
            // groups. Non-segmented runs ignore the mark and dump from the full graph as before.
            auto dump_plane = [&](ggml_tensor* h, const std::string& tag, const std::string& cut_group) {
                if (!wan_dump_blocks || h == nullptr || ctx->debug_tensors == nullptr) return;
                if (wan_dump_h_len <= 0 || wan_dump_w_len <= 0 || wan_dump_t_len <= 0) return;
                const int64_t hw = wan_dump_h_len * wan_dump_w_len;  // tokens per frame
                if (h->ne[1] < hw) return;                           // token axis < one frame => skip (safety)
                ggml_tensor* hc = ggml_is_contiguous(h) ? h : ggml_cont(ctx->ggml_ctx, h);
                int64_t t0 = wan_dump_t_len / 2;                     // representative middle frame
                if (const char* fenv = std::getenv("WAN_DUMP_FRAME")) {
                    long v = std::atol(fenv);
                    t0     = (v < 0) ? (wan_dump_t_len + v) : v;     // negative counts from the end
                }
                if (t0 < 0) t0 = 0;
                if (t0 >= wan_dump_t_len) t0 = wan_dump_t_len - 1;
                // One frame's tokens form a contiguous slab [dim, hw] (token layout: w fastest,
                // h next, t outermost). View it, cast to F32 (CUDA mean is F32+contiguous only),
                // channel-mean over dim -> [1, hw], reshape to the [w_len, h_len] image plane.
                ggml_tensor* frame = ggml_view_2d(ctx->ggml_ctx, hc, hc->ne[0], hw,
                                                  hc->nb[1], (size_t)(t0 * hw) * hc->nb[1]);
                frame              = ggml_cast(ctx->ggml_ctx, frame, GGML_TYPE_F32);
                ggml_tensor* plane = ggml_mean(ctx->ggml_ctx, frame);  // [1, hw]
                plane              = ggml_reshape_2d(ctx->ggml_ctx, plane, wan_dump_w_len, wan_dump_h_len);
                // Concrete output node + cut-mark so it survives segment rebuild; register it in
                // the runner's debug_tensors so the post-compute dump (ggml_extend.hpp, gated on
                // WAN_DUMP_BLOCKS as the output dir) writes <dir>/<tag>_step_SSSS.bin.
                ggml_tensor* snap = ggml_cont(ctx->ggml_ctx, plane);
                ggml_set_output(snap);
                sd::ggml_graph_cut::mark_graph_cut(snap, cut_group, "gridplane");
                ctx->debug_tensors->push_back({snap, tag + "_step_" + wan_pad(wan_dump_step, 4)});
            };
            // DiT input plane (post patch-embed, i.e. block-0 input) — the low bracket.
            dump_plane(x, "input", "wan.dumpblocks.input");

            // time_embedding
            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, config.freq_dim);
            e      = time_embedding_0->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = time_embedding_2->forward(ctx, e);  // [N, dim] or [N, T, dim]

            // time_projection
            auto e0 = ggml_silu(ctx->ggml_ctx, e);
            e0      = time_projection_1->forward(ctx, e0);
            e0      = ggml_reshape_4d(ctx->ggml_ctx, e0, e0->ne[0] / 6, 6, e0->ne[1], e0->ne[2]);  //  [N, 6, dim] or [N, T, 6, dim]

            context = text_embedding_0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context);
            context = text_embedding_2->forward(ctx, context);  // [N, context_txt_len, dim]

            int64_t context_img_len = 0;
            if (clip_fea != nullptr) {
                if (config.model_type == "i2v") {
                    auto img_emb     = std::dynamic_pointer_cast<MLPProj>(blocks["img_emb"]);
                    auto context_img = img_emb->forward(ctx, clip_fea);                      // [N, context_img_len, dim]
                    context          = ggml_concat(ctx->ggml_ctx, context_img, context, 1);  // [N, context_img_len + context_txt_len, dim]
                }
                context_img_len = clip_fea->ne[1];  // 257
            }

            // vace_patch_embedding
            ggml_tensor* c     = nullptr;
            int64_t vace_t_len = 0;  // temporal latent-frame count of the vace control (drives the per-frame ramp)
            if (config.vace_layers > 0) {
                auto vace_patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["vace_patch_embedding"]);

                c = vace_patch_embedding->forward(ctx, vace_context);                                    // [N*dim, t_len, h_len, w_len]  (ggml ne=[w_len,h_len,t_len,N*dim])
                vace_t_len = c->ne[2];                                                                    // temporal latent frames (slowest-varying token block after flatten)
                c = ggml_reshape_3d(ctx->ggml_ctx, c, c->ne[0] * c->ne[1] * c->ne[2], c->ne[3] / N, N);  // [N, dim, t_len*h_len*w_len]
                c = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, c, 1, 0, 2, 3));  // [N, t_len*h_len*w_len, dim]
                // Under WAN_DIT_F16 the main stream `x` is F16, and (with nvfp4 weights) every main-
                // block Linear already emits F16 — but the VACE control stream `c` was left F32, so
                // the vace_blocks' FFN/QKV outputs (the [ffn_dim, T] intermediate is the single
                // biggest activation) stayed F32 and PINNED the compute-buffer peak. Casting `c` to
                // F16 here lets before_proj + the vace-block Linears emit F16 too (nvfp4 weight +
                // F16 input -> F16 dst), halving the vace-stream activations. Self-gated: no-op
                // unless WAN_DIT_F16 made x F16, so the default F32 path is byte-identical.
                if (use_dit_f16 && c->type == GGML_TYPE_F32) {
                    c = ggml_cast(ctx->ggml_ctx, c, GGML_TYPE_F16);
                }
            }
            // VRAM levers (both default OFF => byte-identical to prod; env-gated for A/B).
            //   WAN_VACE_SPLIT           — Lever 1: split each vace-mapped block so the
            //                              main-block attention frees before the vace-block
            //                              attention allocates (peak segment 6268->~3174MB).
            //   WAN_VACE_XORIG_RECOMPUTE — Lever 2: drop the prelude-x graph-cut cache so
            //                              x_orig is retraced (patch_embedding->reshape->cast)
            //                              per vace segment instead of held resident the whole
            //                              DiT (-618MB cache).
            static const bool wan_vace_split = []{
                const char* s = getenv("WAN_VACE_SPLIT");
                return s && s[0] == '1';
            }();
            static const bool wan_vace_xorig_recompute = []{
                const char* s = getenv("WAN_VACE_XORIG_RECOMPUTE");
                return s && s[0] == '1';
            }();
            // Lever 2: the prelude x (== x_orig) is consumed ONLY by the vace blocks (and
            // block 0's main forward). Marking it here pins its F16 618MB tensor in the
            // graph-cut cache for the ENTIRE DiT. When WAN_VACE_XORIG_RECOMPUTE is set we
            // drop the mark so each consuming segment retraces the cheap Conv3d patch
            // embedding from the resident raw-latent leaf instead. Byte-identical
            // (deterministic Conv3d), -618MB cache; default keeps the prelude-x cut.
            if (!wan_vace_xorig_recompute) {
                sd::ggml_graph_cut::mark_graph_cut(x, "wan.prelude", "x");
            }
            // sd::ggml_graph_cut::mark_graph_cut(e, "wan.prelude", "e");
            // sd::ggml_graph_cut::mark_graph_cut(e0, "wan.prelude", "e0");
            // sd::ggml_graph_cut::mark_graph_cut(context, "wan.prelude", "context");
            if (c != nullptr) {
                sd::ggml_graph_cut::mark_graph_cut(c, "wan.prelude", "c");
            }

            auto x_orig = x;

            // VACE_SKIP_BLOCKS: comma-separated vace-block indices whose control residual is
            // NOT added into the main stream. Kijai's documented 2.2-VACE-Fun fix: the PAI
            // adapter's block 0 injects a mis-scaled residual that "flashes"/ripples at full
            // strength + few steps; dropping its add (while still threading c to downstream
            // vace blocks) removes the artifact and keeps the rest of the control intact.
            std::vector<int> vace_skip;
            if (const char* vs = getenv("VACE_SKIP_BLOCKS")) {
                for (const char* p = vs; *p;) {
                    vace_skip.push_back(std::atoi(p));
                    while (*p && *p != ',') ++p;
                    while (*p == ',') ++p;
                }
            }

            // VACE per-latent-frame strength RAMP (default OFF => byte-identical scalar path).
            // Root cause: past the K-frame motion prior the control_video is gray (0.5) = no
            // valid signal, yet every mapped vace_block still synthesizes a residual from that
            // gray control and ADDS it UNIFORMLY to all temporal tokens (including the fast-limb
            // tokens of the free frames), dragging them toward a frozen/gray target => directional
            // smear, worst farthest from the prior. Identity is anchored by the prepended
            // reference-image latent slot + the c_concat prior injection, NOT by this per-frame
            // gray residual — so attenuating the residual on the free-motion TAIL frames while
            // holding the leading ANCHOR frames at full strength releases the smear and keeps
            // identity. The global VACE_STRENGTH scalar can't do this (it weakens the anchored
            // frames too); a per-frame ramp is the distinction. Companion knobs (consumed here
            // via getenv, mirroring the VACE_SKIP_BLOCKS pattern just above):
            //   VACE_STRENGTH_TAIL          — strength floor for the LAST latent frame. Unset (or
            //                                 == VACE_STRENGTH) => ramp OFF => scalar path => prod byte-identical.
            //   VACE_STRENGTH_ANCHOR_FRAMES — leading latent frames held at FULL VACE_STRENGTH
            //                                 before the ramp begins (default 2 = ref slot t=0 + first prior frame).
            ggml_tensor* vace_ramp = nullptr;  // [1,1,t_len,1] per-frame scale; null => use the scalar vace_strength path
            if (c != nullptr && vace_t_len > 0) {
                float tail    = vace_strength;
                bool tail_set = false;
                if (const char* ts = getenv("VACE_STRENGTH_TAIL")) {
                    tail     = (float)atof(ts);
                    tail_set = true;
                }
                int64_t anchor = 2;  // default covers the ref-image latent slot (t=0) + the leading prior latent frame
                if (const char* as = getenv("VACE_STRENGTH_ANCHOR_FRAMES")) {
                    anchor = (int64_t)std::atoi(as);
                }
                if (anchor < 0) anchor = 0;
                const int64_t denom = vace_t_len - 1 - anchor;  // # ramp steps; last frame lands exactly on tail
                // Only ramp when a DIFFERENT tail is requested AND a tail region exists to ramp over;
                // otherwise fall through to the exact existing scalar path => bit-identical to prod.
                const bool ramp_active = tail_set && (tail != vace_strength) && (anchor < vace_t_len) && (denom > 0);
                if (ramp_active) {
                    // s(t) = strength + (tail - strength) * clamp((t - anchor) / denom, 0, 1)
                    //   t <  anchor  : (t-anchor)<0 -> clamp 0 -> s = strength  (anchored identity frames untouched)
                    //   t == t_len-1 : clamp 1       -> s = tail                (free-motion tail attenuated)
                    auto s    = ggml_arange(ctx->ggml_ctx, -(float)anchor, (float)vace_t_len - (float)anchor, 1.0f);  // [t_len] = (t - anchor)
                    s         = ggml_scale(ctx->ggml_ctx, s, 1.0f / (float)denom);                                   // (t-anchor)/denom
                    s         = ggml_clamp(ctx->ggml_ctx, s, 0.0f, 1.0f);                                            // frac in [0,1]
                    s         = ggml_scale(ctx->ggml_ctx, s, tail - vace_strength);                                  // (tail-strength)*frac
                    auto base = ggml_ext_full(ctx->ggml_ctx, vace_strength, vace_t_len, 1, 1, 1);                    // [t_len] const = strength
                    s         = ggml_add(ctx->ggml_ctx, s, base);                                                    // + strength
                    vace_ramp = ggml_reshape_4d(ctx->ggml_ctx, s, 1, 1, vace_t_len, 1);                             // [1,1,t_len,1] broadcast scale
                    static bool logged_ramp = false;
                    if (!logged_ramp) {
                        logged_ramp = true;
                        LOG_INFO("VACE per-frame strength ramp: anchor %lld frames @%.3f, ramp to %.3f over %lld frames (t_len=%lld)",
                                 (long long)anchor, (double)vace_strength, (double)tail,
                                 (long long)(vace_t_len - anchor), (long long)vace_t_len);
                    }
                }
            }

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);

                // WAN SLA: only the configured selector block exports pooled Q/K.
                ctx->sla_capture_now =
                    (ctx->sla_pooled_q_dst != nullptr) && (i == ctx->sla_selector_block);

                x = block->forward(ctx, x, e0, pe, context, context_img_len);
                ctx->sla_capture_now = false;  // don't leak into VACE / later blocks

                auto iter = config.vace_layers_mapping.find(i);
                if (iter != config.vace_layers_mapping.end()) {
                    // Lever 1 (WAN_VACE_SPLIT): cut the main-block output x into its OWN
                    // graph-cut segment so the main-block attention working set completes
                    // and is freed BEFORE the vace-block attention allocates. Without this,
                    // the vace-mapped block is one segment holding BOTH attention working
                    // sets live at once (~6268MB = ~2x a plain block's ~3174MB), which is the
                    // DiT VRAM peak. main-x then crosses into the vace segment (the
                    // `x = x + c_skip` residual add below) as an INPUT_PREVIOUS_CUT, exactly
                    // like the other cross-segment tensors — the cache stays at 3 live tensors
                    // (main-x replaces the previous block's x; x_orig + c are the other two).
                    // Byte-identical: a graph-cut boundary only re-partitions the graph for
                    // per-segment memory reservation; every arithmetic node, and the
                    // `x = x + c_skip` residual add and its order, is unchanged.
                    if (wan_vace_split && c != nullptr) {
                        sd::ggml_graph_cut::mark_graph_cut(x, "wan.blocks." + std::to_string(i) + ".main", "x");
                    }
                    int n = iter->second;

                    auto vace_block = std::dynamic_pointer_cast<VaceWanAttentionBlock>(blocks["vace_blocks." + std::to_string(n)]);

                    auto result = vace_block->forward(ctx, c, x_orig, e0, pe, context, context_img_len);
                    auto c_skip = result.first;
                    c           = result.second;
                    // Drop this vace block's residual injection if listed (keeps c threaded).
                    if (std::find(vace_skip.begin(), vace_skip.end(), n) == vace_skip.end()) {
                        if (vace_ramp != nullptr) {
                            // Per-latent-frame ramp: broadcast-multiply the control residual by s[t].
                            // c_skip is [N, t_len*hw, dim] (ggml ne=[dim, t_len*hw, N]); the token axis
                            // is temporal-OUTERMOST (flat tok = hw_idx + t*hw, so frame t is the slow
                            // block of `hw` tokens). Split that axis into [hw, t_len] (hw innermost,
                            // matching the layout) and multiply by vace_ramp ne=[1,1,t_len,1], which
                            // broadcasts over dim, hw and N so every token of frame t is scaled by s[t];
                            // then reshape back to the flat token layout. Applies to EVERY mapped block.
                            if (!ggml_is_contiguous(c_skip)) {
                                c_skip = ggml_cont(ctx->ggml_ctx, c_skip);
                            }
                            const int64_t dim = c_skip->ne[0];
                            const int64_t hw  = c_skip->ne[1] / vace_t_len;  // h_len*w_len tokens per frame
                            const int64_t Nb  = c_skip->ne[2];
                            auto c4 = ggml_reshape_4d(ctx->ggml_ctx, c_skip, dim, hw, vace_t_len, Nb);  // [dim, hw, t_len, N]
                            c4      = ggml_mul(ctx->ggml_ctx, c4, vace_ramp);                           // per-frame attenuation
                            c_skip  = ggml_reshape_3d(ctx->ggml_ctx, c4, dim, vace_t_len * hw, Nb);     // back to [dim, tok, N]
                        } else {
                            c_skip = ggml_ext_scale(ctx->ggml_ctx, c_skip, vace_strength);
                        }
                        x = ggml_add(ctx->ggml_ctx, x, c_skip);
                    }
                }
                sd::ggml_graph_cut::mark_graph_cut(x, "wan.blocks." + std::to_string(i), "x");
                if (c != nullptr) {
                    sd::ggml_graph_cut::mark_graph_cut(c, "wan.blocks." + std::to_string(i), "c");
                }
                // WAN_DUMP_BLOCKS: capture this block's post-residual (incl. any VACE inject) plane.
                // Ride the block's OWN cut group ("wan.blocks.<i>", marked just above) so the plane
                // joins this block's segment and adds no extra segment.
                dump_plane(x, "block_" + wan_pad(i, 2), "wan.blocks." + std::to_string(i));
            }

            // WAN_DIT_F16: bring the residual stream back to F32 before the head so the
            // norm_out/modulate/proj_out tail + unpatchify + VAE run exactly as in prod
            // (the head Linear then sees an F32 activation → F32 dst, unchanged output).
            if (use_dit_f16 && x->type != GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }
            x = head->forward(ctx, x, e);  // [N, t_len*h_len*w_len, pt*ph*pw*out_dim]

            // WAN_DUMP_BLOCKS: DiT output plane (post final norm/head) — the high bracket.
            dump_plane(x, "head", "wan.dumpblocks.head");

            return x;
        }

        // ------------------------------------------------------------------
        // ShotStream CAUSAL block forward. Runs ONE chunk (nfb latent frames) of the
        // streaming AR loop: patch-embed → time/text embed → 30 causal blocks (each
        // threading the per-layer local + global KV caches) → head → unpatchify. All
        // 3 latent frames of a chunk share one timestep, so the modulation is the plain
        // [N,6,dim] broadcast (no per-frame path). The self-attn of each block attends
        // over [prev_local ++ this_chunk ++ context] and exports its fresh RoPE'd K + raw
        // V into new_kc/new_vc for the host rolling cache.
        //   x_chunk:  [W, H, nfb, in_dim]  (latent; patch t=1 ⇒ t_len=nfb)
        //   timestep: [1] scalar (the warped DMD t, or 0 for the clean rewrite / context)
        //   context:  [text_dim, text_len, N]
        //   pe_block: RoPE table for this chunk's L_blk = nfb*h_len*w_len tokens
        // Returns velocity/flow [W, H, nfb, out_dim] (the sampler converts to x0).
        ggml_tensor* forward_causal_block(GGMLRunnerContext* ctx,
                                          ggml_tensor* x_chunk,
                                          ggml_tensor* timestep,
                                          ggml_tensor* context,
                                          ggml_tensor* pe_block,
                                          const std::vector<ggml_tensor*>& prev_kc,
                                          const std::vector<ggml_tensor*>& prev_vc,
                                          const std::vector<ggml_tensor*>& cond_kc,
                                          const std::vector<ggml_tensor*>& cond_vc,
                                          std::vector<ggml_tensor*>& new_kc,
                                          std::vector<ggml_tensor*>& new_vc) {
            const int64_t N = 1;
            auto patch_embedding  = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);
            auto text_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto text_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);
            auto time_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto time_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);
            auto head             = std::dynamic_pointer_cast<Head>(blocks["head"]);

            int64_t T = x_chunk->ne[2], H = x_chunk->ne[1], W = x_chunk->ne[0];
            int64_t t_len = ((T + (std::get<0>(config.patch_size) / 2)) / std::get<0>(config.patch_size));
            int64_t h_len = ((H + (std::get<1>(config.patch_size) / 2)) / std::get<1>(config.patch_size));
            int64_t w_len = ((W + (std::get<2>(config.patch_size) / 2)) / std::get<2>(config.patch_size));

            // patch_embedding → [1, L_blk, dim]
            auto x = patch_embedding->forward(ctx, x_chunk);                                          // [W_l,H_l,t_len,dim]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3] / N, N);   // [N, dim, L_blk]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));   // [N, L_blk, dim]

            // WAN_DIT_F16 (default OFF, byte-identical when unset): run the causal residual
            // stream in F16 to halve the per-block glue/copy/cast HBM traffic (the binbcast
            // residual adds, the AdaLN modulate muls, the FFN activations) — the same lever
            // as forward() line ~995, replicated here because the causal path is a separate
            // method. NB: GGML_CUDNN_ATTN_F16_OUT must stay OFF on this path — the KV-cache
            // self-attn uses the native ggml FA2 kernel (fattn-common.cuh), which asserts an
            // F32 KQV output; only forward()'s cuDNN attention accepts the F16-out retype.
            // Under WAN_DIT_F16 the block's q/k are still upcast to F32 for the fused RoPE
            // (unless WAN_ROPE_F16), and the F32 attn output re-enters the F16 stream via
            // o_proj (F16 dst). Cast back to F32 before the head below so the sampler/VAE
            // path is unchanged.
            static const bool ss_dit_f16_env = (std::getenv("WAN_DIT_F16") != nullptr);
            // DEVICE-GATED (Blackwell-only) — see the wan_dit_f16 gate above; the F16 stream
            // aborts native flash on sm86, so fall back to F32 on non-Blackwell.
#ifdef SD_USE_CUDA
            static const bool ss_dit_f16 = ss_dit_f16_env && ggml_backend_cuda_device_has_blackwell_mma(0);
#else
            static const bool ss_dit_f16 = false;
#endif
            if (ss_dit_f16 && x->type == GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F16);
            }

            // time embedding (scalar t → single modulation broadcast over all tokens)
            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, config.freq_dim);
            e      = time_embedding_0->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = time_embedding_2->forward(ctx, e);  // [N, dim]

            auto e0 = ggml_silu(ctx->ggml_ctx, e);
            e0      = time_projection_1->forward(ctx, e0);
            e0      = ggml_reshape_4d(ctx->ggml_ctx, e0, e0->ne[0] / 6, 6, e0->ne[1], e0->ne[2]);  // [N,6,dim]

            context = text_embedding_0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context);
            context = text_embedding_2->forward(ctx, context);  // [N, text_len, dim]

            sd::ggml_graph_cut::mark_graph_cut(x, "shotstream.prelude", "x");
            sd::ggml_graph_cut::mark_graph_cut(e0, "shotstream.prelude", "e0");
            sd::ggml_graph_cut::mark_graph_cut(context, "shotstream.prelude", "ctx");
            if (pe_block) sd::ggml_graph_cut::mark_graph_cut(pe_block, "shotstream.prelude", "pe");

            new_kc.assign(config.num_layers, nullptr);
            new_vc.assign(config.num_layers, nullptr);
            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);
                ggml_tensor* nkc = nullptr;
                ggml_tensor* nvc = nullptr;
                x = block->forward_causal(ctx, x, e0, pe_block, context,
                                          prev_kc.empty() ? nullptr : prev_kc[i],
                                          prev_vc.empty() ? nullptr : prev_vc[i],
                                          cond_kc.empty() ? nullptr : cond_kc[i],
                                          cond_vc.empty() ? nullptr : cond_vc[i],
                                          nkc, nvc);
                new_kc[i] = nkc;
                new_vc[i] = nvc;
                sd::ggml_graph_cut::mark_graph_cut(x, "shotstream.blocks." + std::to_string(i) + ".out", "x");
            }

            // WAN_DIT_F16: bring the residual stream back to F32 before the head so the
            // head Linear + unpatchify + flow output run exactly as prod (F32 output).
            if (ss_dit_f16 && x->type != GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }
            auto out = head->forward(ctx, x, e);  // [N, L_blk, pt*ph*pw*out_dim]
            out = unpatchify(ctx->ggml_ctx, out, t_len, h_len, w_len);  // [N*out_dim, t_len, H, W]
            return out;
        }

        // Parity probe: block-0 self-attn only (ShotStream oracle P1). Bridges to
        // WanAttentionBlock::selfattn_only on block 0. new_kc/new_vc = RoPE'd K + raw V.
        ggml_tensor* forward_block0_selfattn(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe,
                                             ggml_tensor*& new_kc, ggml_tensor*& new_vc) {
            auto block = std::dynamic_pointer_cast<WanAttentionBlock>(blocks["blocks.0"]);
            return block->selfattn_only(ctx, x, pe, new_kc, new_vc);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* clip_fea        = nullptr,
                             ggml_tensor* time_dim_concat = nullptr,
                             ggml_tensor* vace_context    = nullptr,
                             float vace_strength          = 1.f,
                             int64_t N                    = 1) {
            // Forward pass of DiT.
            // x: [N*C, T, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // time_dim_concat: [N*C, T2, H, W]
            // return: [N*C, T, H, W]

            GGML_ASSERT(N == 1);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];
            int64_t C = x->ne[3];

            x = pad_to_patch_size(ctx, x);

            int64_t t_len = ((T + (std::get<0>(config.patch_size) / 2)) / std::get<0>(config.patch_size));
            int64_t h_len = ((H + (std::get<1>(config.patch_size) / 2)) / std::get<1>(config.patch_size));
            int64_t w_len = ((W + (std::get<2>(config.patch_size) / 2)) / std::get<2>(config.patch_size));

            if (time_dim_concat != nullptr) {
                time_dim_concat = pad_to_patch_size(ctx, time_dim_concat);
                x               = ggml_concat(ctx->ggml_ctx, x, time_dim_concat, 2);  // [N*C, (T+pad_t) + (T2+pad_t2), H + pad_h, W + pad_w]
                t_len           = ((x->ne[2] + (std::get<0>(config.patch_size) / 2)) / std::get<0>(config.patch_size));
            }

            auto out = forward_orig(ctx, x, timestep, context, pe, clip_fea, vace_context, vace_strength, N);  // [N, t_len*h_len*w_len, pt*ph*pw*C]

            out = unpatchify(ctx->ggml_ctx, out, t_len, h_len, w_len);  // [N*C, (T+pad_t) + (T2+pad_t2), H + pad_h, W + pad_w]

            // slice

            out = ggml_ext_slice(ctx->ggml_ctx, out, 2, 0, T);  // [N*C, T, H + pad_h, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, H);  // [N*C, T, H, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 0, 0, W);  // [N*C, T, H, W]

            return out;
        }
    };

    struct WanRunner : public DiffusionModelRunner {
    public:
        std::string desc = "wan";
        WanConfig config;
        Wan wan;
        std::vector<float> pe_vec;
        SDVersion version;
        // WAN SLA (lightx2v sparse-attention port). Default OFF (WAN_SLA unset).
        sd::WanSlaState sla;

        WanRunner(ggml_backend_t backend,
                  ggml_backend_t params_backend,
                  const String2TensorStorage& tensor_storage_map = {},
                  const std::string prefix                       = "",
                  SDVersion version                              = VERSION_WAN2)
            : DiffusionModelRunner(backend, params_backend, prefix),
              config(WanConfig::detect_from_weights(tensor_storage_map, prefix)) {
            if (config.num_layers == 30) {
                if (version == VERSION_WAN2_2_TI2V) {
                    desc             = "Wan2.2-TI2V-5B";
                    config.dim       = 3072;
                    config.eps       = 1e-06f;
                    config.ffn_dim   = 14336;
                    config.freq_dim  = 256;
                    config.in_dim    = 48;
                    config.num_heads = 24;
                    config.out_dim   = 48;
                    config.text_len  = 512;
                } else {
                    if (config.vace_layers > 0) {
                        desc          = "Wan2.1-VACE-1.3B";
                        config.in_dim = 16;
                    } else if (config.model_type == "i2v") {
                        desc          = "Wan2.1-I2V-1.3B";
                        config.in_dim = 36;
                    } else {
                        desc          = "Wan2.1-T2V-1.3B";
                        config.in_dim = 16;
                    }
                    config.dim       = 1536;
                    config.eps       = 1e-06f;
                    config.ffn_dim   = 8960;
                    config.freq_dim  = 256;
                    config.num_heads = 12;
                    config.out_dim   = 16;
                    config.text_len  = 512;
                }
            } else if (config.num_layers == 40) {
                if (config.model_type == "t2v") {
                    if (version == VERSION_WAN2_2_I2V) {
                        desc          = "Wan2.2-I2V-14B";
                        config.in_dim = 36;
                    } else {
                        if (config.vace_layers > 0) {
                            desc = "Wan2.x-VACE-14B";
                        } else {
                            desc = "Wan2.x-T2V-14B";
                        }
                        config.in_dim = 16;
                    }
                } else {
                    config.in_dim = 36;
                    if (config.flf_pos_embed_token_number > 0) {
                        desc = "Wan2.1-FLF2V-14B";
                    } else {
                        desc = "Wan2.1-I2V-14B";
                    }
                }
                config.dim       = 5120;
                config.eps       = 1e-06f;
                config.ffn_dim   = 13824;
                config.freq_dim  = 256;
                config.num_heads = 40;
                config.out_dim   = 16;
                config.text_len  = 512;
            } else {
                GGML_ABORT("invalid num_layers(%d) of wan", config.num_layers);
            }

            LOG_INFO("%s", desc.c_str());

            wan = Wan(config);
            wan.init(params_ctx, tensor_storage_map, prefix);

            sla.cfg = sd::WanSlaConfig::from_env();
            if (sla.cfg.enabled) {
                LOG_INFO("[WAN-SLA] enabled: sparsity=%.2f (keep top %.0f%%), selector_block=%d, mode=%s%s, warmup=%d",
                         sla.cfg.sparsity, 100.0f * (1.0f - sla.cfg.sparsity), sla.cfg.selector_block,
                         sla.cfg.current_step ? "current-step(2-pass)" : "stale",
                         sla.cfg.per_head ? "+per-head" : "+shared-heads", sla.cfg.warmup_steps);
            }
        }

        std::string get_desc() override {
            return desc;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            wan.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor         = {},
                                 const sd::Tensor<float>& clip_fea_tensor        = {},
                                 const sd::Tensor<float>& c_concat_tensor        = {},
                                 const sd::Tensor<float>& time_dim_concat_tensor = {},
                                 const sd::Tensor<float>& vace_context_tensor    = {},
                                 float vace_strength                             = 1.f) {
            ggml_cgraph* gf = new_graph_custom(WAN_GRAPH_SIZE);

            ggml_tensor* x               = make_input(x_tensor);
            ggml_tensor* timesteps       = make_input(timesteps_tensor);
            ggml_tensor* context         = make_optional_input(context_tensor);
            ggml_tensor* clip_fea        = make_optional_input(clip_fea_tensor);
            ggml_tensor* c_concat        = make_optional_input(c_concat_tensor);
            ggml_tensor* time_dim_concat = make_optional_input(time_dim_concat_tensor);
            ggml_tensor* vace_context    = make_optional_input(vace_context_tensor);

            pe_vec      = Rope::gen_wan_pe(static_cast<int>(x->ne[2]),
                                           static_cast<int>(x->ne[1]),
                                           static_cast<int>(x->ne[0]),
                                           std::get<0>(config.patch_size),
                                           std::get<1>(config.patch_size),
                                           std::get<2>(config.patch_size),
                                           1,
                                           config.theta,
                                           config.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            if (c_concat != nullptr) {
                x = ggml_concat(compute_ctx, x, c_concat, 3);
            }

            auto runner_ctx = get_context();
            runner_ctx.gf   = gf;

            // WAN SLA: size the per-render selector state to this self-attn sequence
            // (pos_len == the blocks' n_token), register the persistent export/bitmap
            // tensors, and thread the capture dsts into the selector block. The device
            // bitmap itself is (re)applied per step in compute(), before graph exec.
            if (sla.cfg.enabled) {
                const int dh = (int)(config.dim / config.num_heads);
                sla.ensure(runtime_backend, (int64_t)pos_len, (int)config.num_heads, dh);
                register_persistent_tensor(sla.pooled_q);
                register_persistent_tensor(sla.pooled_k);
                register_persistent_tensor(sla.bitmap);
                runner_ctx.sla_pooled_q_dst  = sla.pooled_q;
                runner_ctx.sla_pooled_k_dst  = sla.pooled_k;
                runner_ctx.sla_selector_block = sla.cfg.selector_block;
                runner_ctx.sla_blk            = sd::WAN_SLA_BLK;
            }

            ggml_tensor* out = wan.forward(&runner_ctx,
                                           x,
                                           timesteps,
                                           context,
                                           pe,
                                           clip_fea,
                                           time_dim_concat,
                                           vace_context,
                                           vace_strength);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context         = {},
                                  const sd::Tensor<float>& clip_fea        = {},
                                  const sd::Tensor<float>& c_concat        = {},
                                  const sd::Tensor<float>& time_dim_concat = {},
                                  const sd::Tensor<float>& vace_context    = {},
                                  float vace_strength                      = 1.f) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, time_dim_concat, vace_context, vace_strength);
            };

            // WAN SLA Stage 1 — CURRENT-STEP (the ghosting fix): two passes per step.
            // Pass 1 runs DENSE (bitmap cleared) and harvests THIS step's pooled Q/K
            // from the selector block; the host builds the bitmap (per-head if enabled);
            // pass 2 reruns SPARSE with it. Block-`sel`'s Q/K are a pure function of x
            // (unchanged between passes), so the selection is exactly this step's — no
            // one-step-stale echo. Speed/VRAM are irrelevant here (quality validation).
            if (sla.cfg.enabled && sla.cfg.current_step) {
                sla.apply_device_sparse(false);                                  // pass 1: dense
                (void)GGMLRunner::compute<float>(get_graph, n_threads, false);   // harvest pooled
                sla.update_from_pooled();                                        // build CURRENT-step bitmap
                sla.apply_device_sparse(true);                                   // pass 2: sparse
                auto out = restore_trailing_singleton_dims(
                    GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
                LOG_INFO("[WAN-SLA] step %d (current-step%s): live=%.1f%% (target keep %.0f%%), n_blk=%d",
                         sla.step - 1, sla.cfg.per_head ? ",per-head" : "",
                         100.0 * sla.last_live_frac, 100.0f * (1.0f - sla.cfg.sparsity), sla.n_blk);
                sla.apply_device_sparse(false);  // leave device clean for the next model/op
                return out;
            }

            // WAN SLA Stage 0 — one-step-STALE: push the previous step's bitmap before
            // this step's graph, harvest pooled after for the next step. Default OFF.
            if (sla.cfg.enabled) sla.apply_device_sparse(sla.sparse_now());

            auto out = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());

            if (sla.cfg.enabled) {
                sla.update_from_pooled();
                LOG_INFO("[WAN-SLA] step %d: bitmap live=%.1f%% (target keep %.0f%%), n_blk=%d sparse_next=%d",
                         sla.step - 1, 100.0 * sla.last_live_frac, 100.0f * (1.0f - sla.cfg.sparsity),
                         sla.n_blk, (int)sla.sparse_now());
            }

            return out;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            const auto* extra = diffusion_extra_as<WanDiffusionExtra>(diffusion_params);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           tensor_or_empty(diffusion_params.y),
                           tensor_or_empty(diffusion_params.c_concat),
                           sd::Tensor<float>(),
                           tensor_or_empty(extra->vace_context),
                           extra->vace_strength);
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(200 * 1024 * 1024);  // 200 MB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            {
                // cpu f16: pass
                // cuda f16: pass
                // cpu q8_0: pass
                // auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 104, 60, 1, 16);
                // ggml_set_f32(x, 0.01f);
                auto x = sd::load_tensor_from_file_as_tensor<float>("wan_dit_x.bin");
                print_sd_tensor(x);

                std::vector<float> timesteps_vec(3, 1000.f);
                timesteps_vec[0] = 0.f;
                auto timesteps   = sd::Tensor<float>::from_vector(timesteps_vec);

                // auto context = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 512, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = sd::load_tensor_from_file_as_tensor<float>("wan_dit_context.bin");
                print_sd_tensor(context);
                // auto clip_fea = load_tensor_from_file(ctx, "wan_dit_clip_fea.bin");
                // print_ggml_tensor(clip_fea);

                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = compute(8, x, timesteps, context, {}, {}, {}, {}, 1.f);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("wan test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = sd_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_F16;
            LOG_INFO("loading from '%s'", file_path.c_str());

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            for (auto& [name, tensor_storage] : tensor_storage_map) {
                if (ends_with(name, "weight")) {
                    tensor_storage.expected_type = model_data_type;
                }
            }

            std::shared_ptr<WanRunner> wan = std::make_shared<WanRunner>(backend,
                                                                         backend,
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         VERSION_WAN2_2_TI2V);

            if (!wan->alloc_params_buffer()) {
                LOG_ERROR("wan buffer allocation failed");
                return;
            }

            std::map<std::string, ggml_tensor*> tensors;
            wan->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("wan model loaded");

            wan->test();
        }
    };

}  // namespace WAN

#endif  // __SD_MODEL_DIFFUSION_WAN_HPP__
