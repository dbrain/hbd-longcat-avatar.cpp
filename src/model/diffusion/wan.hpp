#ifndef __SD_MODEL_DIFFUSION_WAN_HPP__
#define __SD_MODEL_DIFFUSION_WAN_HPP__

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <cstdlib>

#include "model/common/block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
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
            auto k = k_proj->forward(ctx, x);
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, x);  // [N, n_token, n_head*d_head]

            // WAN_DIT_F16: under the F16 residual stream the q/k/v Linears emit F16, but
            // the fast fused RoPE (ggml_rope_pe) is F32-only (rope.hpp:953) — an F16 q
            // would fall back to the slow cont+repeat+mul+add chain (big intermediates,
            // the lap-08b smell). Upcast q/k to F32 here so the fused RoPE fires; this is
            // the F32-cast LTX applies for the same reason. The cuDNN flash kernel casts
            // q→F16 internally and, with GGML_CUDNN_ATTN_F16_OUT, stores an F16 output
            // that feeds the F16 o_proj. v is re-cast to F16 inside ggml_ext_attention_ext
            // (build_kqv), so leave it F16. Self-gated on the F16 type → no-op in the
            // default F32 path (byte-identical).
            if (q->type == GGML_TYPE_F16) {
                q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_F32);
                k = ggml_cast(ctx->ggml_ctx, k, GGML_TYPE_F32);
            }

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]

            x = Rope::attention(ctx, q, k, v, pe, mask);  // [N, n_token, dim]

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

            auto q_roped = Rope::apply_rope(ctx->ggml_ctx, q, pe, true);  // [d_head, L_blk, n_head]
            auto k_roped = Rope::apply_rope(ctx->ggml_ctx, k, pe, true);  // [d_head, L_blk, n_head]
            auto v_flat  = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));  // [d_head, L_blk, n_head]

            new_kc = k_roped;
            new_vc = v_flat;

            ggml_tensor* k_all = k_roped;
            ggml_tensor* v_all = v_flat;
            // Lever 1 (S2V): the host rolling/cond caches are F16. Cast back to F32
            // (lossless) before concat so the in-graph attention math is unchanged.
            // ggml_concat requires matching dtypes; k_roped/v_flat are F32.
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
            // attention over the rolling cache), so mask=nullptr is correct. The
            // wrapper pads L_k->256 internally. Gated by the runner's flash flag so
            // S2V_NO_FLASH=1 still selects the exact (FA-off) path for A/B.
            auto attn = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend,
                                               q_roped, k_all, vv, num_heads, nullptr,
                                               /*skip_reshape=*/true,
                                               /*flash_attn=*/ctx->flash_attn_enabled);  // [1, L_blk, dim]
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
            kc = Rope::apply_rope(ctx->ggml_ctx, k, pe_cond, true);  // [d_head, L, n_head]
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

            y = ffn_0->forward(ctx, y);
            y = ggml_ext_gelu(ctx->ggml_ctx, y, true);
            y = ffn_2->forward(ctx, y);

            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[5]));

            return x;
        }

        // Causal streaming variant used by ShotStream. The attention cache is
        // deliberately supplied by the caller so the standard Wan model remains
        // stateless and ModelManager-owned.
        ggml_tensor* forward_causal(GGMLRunnerContext* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* e,
                                    ggml_tensor* pe,
                                    ggml_tensor* context,
                                    ggml_tensor* prev_kc,
                                    ggml_tensor* prev_vc,
                                    ggml_tensor* cond_kc,
                                    ggml_tensor* cond_vc,
                                    ggml_tensor*& new_kc,
                                    ggml_tensor*& new_vc,
                                    int64_t context_img_len = 0) {
            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);

            auto y = norm1->forward(ctx, x);
            y      = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[1]));
            y      = modulate_add(ctx->ggml_ctx, y, es[0]);
            y      = self_attn->forward_kv_cache(ctx, y, pe, prev_kc, prev_vc, cond_kc, cond_vc, new_kc, new_vc);
            x      = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[2]));

            x = ggml_add(ctx->ggml_ctx,
                         x,
                         cross_attn->forward(ctx, norm3->forward(ctx, x), context, context_img_len));

            y = norm2->forward(ctx, x);
            y = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[4]));
            y = modulate_add(ctx->ggml_ctx, y, es[3]);
            y = ffn_0->forward(ctx, y);
            y = ggml_ext_gelu(ctx->ggml_ctx, y, true);
            y = ffn_2->forward(ctx, y);
            return ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[5]));
        }

        ggml_tensor* selfattn_only(GGMLRunnerContext* ctx,
                                   ggml_tensor* x,
                                   ggml_tensor* pe,
                                   ggml_tensor*& new_kc,
                                   ggml_tensor*& new_vc) {
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
                c = ggml_add(ctx->ggml_ctx, c, x);
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
            static const bool wan_dit_f16 = (std::getenv("WAN_DIT_F16") != nullptr);
            const bool use_dit_f16        = wan_dit_f16 && config.vace_layers == 0;
            if (use_dit_f16 && x->type == GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F16);
            }

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
            ggml_tensor* c = nullptr;
            int64_t vace_t_len = 0;
            if (config.vace_layers > 0) {
                auto vace_patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["vace_patch_embedding"]);

                c = vace_patch_embedding->forward(ctx, vace_context);                                    // [N*dim, t_len, h_len, w_len]
                vace_t_len = c->ne[2];
                c = ggml_reshape_3d(ctx->ggml_ctx, c, c->ne[0] * c->ne[1] * c->ne[2], c->ne[3] / N, N);  // [N, dim, t_len*h_len*w_len]
                c = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, c, 1, 0, 2, 3));  // [N, t_len*h_len*w_len, dim]
            }
            sd::ggml_graph_cut::mark_graph_cut(x, "wan.prelude", "x");
            // sd::ggml_graph_cut::mark_graph_cut(e, "wan.prelude", "e");
            // sd::ggml_graph_cut::mark_graph_cut(e0, "wan.prelude", "e0");
            // sd::ggml_graph_cut::mark_graph_cut(context, "wan.prelude", "context");
            if (c != nullptr) {
                sd::ggml_graph_cut::mark_graph_cut(c, "wan.prelude", "c");
            }

            auto x_orig = x;

            // VACE continuation uses an anchored leading context and a gray
            // reactive tail.  A scalar control strength over-constrains that
            // tail; optionally attenuate it per latent frame while leaving the
            // leading anchor untouched.  With the variables unset this remains
            // the exact upstream scalar path.
            ggml_tensor* vace_ramp = nullptr;
            if (c != nullptr && vace_t_len > 1) {
                const char* tail_env = getenv("VACE_STRENGTH_TAIL");
                if (tail_env != nullptr && tail_env[0] != '\0') {
                    const float tail = static_cast<float>(atof(tail_env));
                    int64_t anchor = 2;
                    if (const char* anchor_env = getenv("VACE_STRENGTH_ANCHOR_FRAMES");
                        anchor_env != nullptr && anchor_env[0] != '\0') {
                        anchor = std::max<int64_t>(0, atoll(anchor_env));
                    }
                    const int64_t denom = vace_t_len - 1 - anchor;
                    if (tail != vace_strength && anchor < vace_t_len && denom > 0) {
                        auto scale = ggml_arange(ctx->ggml_ctx,
                                                 -static_cast<float>(anchor),
                                                 static_cast<float>(vace_t_len - anchor),
                                                 1.0f);
                        scale = ggml_scale(ctx->ggml_ctx, scale, 1.0f / static_cast<float>(denom));
                        scale = ggml_clamp(ctx->ggml_ctx, scale, 0.0f, 1.0f);
                        scale = ggml_scale(ctx->ggml_ctx, scale, tail - vace_strength);
                        auto base = ggml_ext_full(ctx->ggml_ctx, vace_strength, vace_t_len, 1, 1, 1);
                        scale = ggml_add(ctx->ggml_ctx, scale, base);
                        vace_ramp = ggml_reshape_4d(ctx->ggml_ctx, scale, 1, 1, vace_t_len, 1);
                    }
                }
            }

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);

                x = block->forward(ctx, x, e0, pe, context, context_img_len);

                auto iter = config.vace_layers_mapping.find(i);
                if (iter != config.vace_layers_mapping.end()) {
                    int n = iter->second;

                    auto vace_block = std::dynamic_pointer_cast<VaceWanAttentionBlock>(blocks["vace_blocks." + std::to_string(n)]);

                    auto result = vace_block->forward(ctx, c, x_orig, e0, pe, context, context_img_len);
                    auto c_skip = result.first;
                    c           = result.second;
                    if (vace_ramp != nullptr) {
                        if (!ggml_is_contiguous(c_skip)) {
                            c_skip = ggml_cont(ctx->ggml_ctx, c_skip);
                        }
                        const int64_t dim = c_skip->ne[0];
                        const int64_t hw = c_skip->ne[1] / vace_t_len;
                        const int64_t batch = c_skip->ne[2];
                        auto c4 = ggml_reshape_4d(ctx->ggml_ctx, c_skip, dim, hw, vace_t_len, batch);
                        c4 = ggml_mul(ctx->ggml_ctx, c4, vace_ramp);
                        c_skip = ggml_reshape_3d(ctx->ggml_ctx, c4, dim, vace_t_len * hw, batch);
                    } else {
                        c_skip = ggml_ext_scale(ctx->ggml_ctx, c_skip, vace_strength);
                    }
                    x           = ggml_add(ctx->ggml_ctx, x, c_skip);
                }
                sd::ggml_graph_cut::mark_graph_cut(x, "wan.blocks." + std::to_string(i), "x");
                if (c != nullptr) {
                    sd::ggml_graph_cut::mark_graph_cut(c, "wan.blocks." + std::to_string(i), "c");
                }
            }

            // WAN_DIT_F16: bring the residual stream back to F32 before the head so the
            // norm_out/modulate/proj_out tail + unpatchify + VAE run exactly as in prod
            // (the head Linear then sees an F32 activation → F32 dst, unchanged output).
            if (use_dit_f16 && x->type != GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }
            x = head->forward(ctx, x, e);  // [N, t_len*h_len*w_len, pt*ph*pw*out_dim]

            return x;
        }

        // Causal ShotStream forward. This is a compact wrapper around the normal
        // Wan blocks that threads caller-owned local and global K/V caches through
        // each self-attention layer.
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
            constexpr int64_t N = 1;
            auto patch_embedding  = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);
            auto text_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto text_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);
            auto time_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto time_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);
            auto head             = std::dynamic_pointer_cast<Head>(blocks["head"]);

            int64_t T = x_chunk->ne[2], H = x_chunk->ne[1], W = x_chunk->ne[0];
            int64_t t_len = (T + std::get<0>(config.patch_size) / 2) / std::get<0>(config.patch_size);
            int64_t h_len = (H + std::get<1>(config.patch_size) / 2) / std::get<1>(config.patch_size);
            int64_t w_len = (W + std::get<2>(config.patch_size) / 2) / std::get<2>(config.patch_size);

            auto x = patch_embedding->forward(ctx, x_chunk);
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3] / N, N);
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));

            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, config.freq_dim);
            e      = time_embedding_0->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = time_embedding_2->forward(ctx, e);
            auto e0 = ggml_silu(ctx->ggml_ctx, e);
            e0      = time_projection_1->forward(ctx, e0);
            e0      = ggml_reshape_4d(ctx->ggml_ctx, e0, e0->ne[0] / 6, 6, e0->ne[1], e0->ne[2]);

            context = text_embedding_0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context);
            context = text_embedding_2->forward(ctx, context);

            sd::ggml_graph_cut::mark_graph_cut(x, "shotstream.prelude", "x");
            sd::ggml_graph_cut::mark_graph_cut(e0, "shotstream.prelude", "e0");
            sd::ggml_graph_cut::mark_graph_cut(context, "shotstream.prelude", "ctx");
            if (pe_block != nullptr) {
                sd::ggml_graph_cut::mark_graph_cut(pe_block, "shotstream.prelude", "pe");
            }

            new_kc.assign(config.num_layers, nullptr);
            new_vc.assign(config.num_layers, nullptr);
            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);
                x = block->forward_causal(ctx, x, e0, pe_block, context,
                                          prev_kc.empty() ? nullptr : prev_kc[i],
                                          prev_vc.empty() ? nullptr : prev_vc[i],
                                          cond_kc.empty() ? nullptr : cond_kc[i],
                                          cond_vc.empty() ? nullptr : cond_vc[i],
                                          new_kc[i], new_vc[i]);
                sd::ggml_graph_cut::mark_graph_cut(x, "shotstream.blocks." + std::to_string(i) + ".out", "x");
            }

            auto out = head->forward(ctx, x, e);
            return unpatchify(ctx->ggml_ctx, out, t_len, h_len, w_len);
        }

        ggml_tensor* forward_block0_selfattn(GGMLRunnerContext* ctx,
                                              ggml_tensor* x,
                                              ggml_tensor* pe,
                                              ggml_tensor*& new_kc,
                                              ggml_tensor*& new_vc) {
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

        WanRunner(ggml_backend_t backend,
                  const String2TensorStorage& tensor_storage_map      = {},
                  const std::string prefix                            = "",
                  SDVersion version                                   = VERSION_WAN2,
                  std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
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

            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
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

            auto model_manager        = std::make_shared<ModelManager>();
            ModelLoader& model_loader = model_manager->loader();
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
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         VERSION_WAN2_2_TI2V,
                                                                         model_manager);

            if (!model_manager->register_runner_params("Wan test",
                                                       *wan,
                                                       "model.diffusion_model",
                                                       ModelManager::ResidencyMode::ParamBackend,
                                                       backend,
                                                       backend) ||
                !model_manager->validate_registered_tensors()) {
                LOG_ERROR("register wan tensors with model manager failed");
                return;
            }

            LOG_INFO("wan model loaded");

            wan->test();
        }
    };

}  // namespace WAN

#endif  // __SD_MODEL_DIFFUSION_WAN_HPP__
