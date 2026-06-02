#ifndef __WAN_S2V_HPP__
#define __WAN_S2V_HPP__

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "ggml_extend.hpp"
#include "rope.hpp"
#include "wan.hpp"  // reuse WAN::WanAttentionBlock, WanSelfAttention, WanT2VCrossAttention, Head, modulate_*

// ---------------------------------------------------------------------------
// Wan2.2-S2V-14B DiT (LiveAvatar M1: STOCK non-causal, NO LoRA, drop_motion_frames).
//
// Ground truth: /mnt/hdd/live-avatar/ref/Wan2.2/wan/modules/s2v/model_s2v.py
//   (WanModel_S2V.forward, WanS2VAttentionBlock, Head_S2V, after_transformer_block),
//   audio_utils.py (AudioInjector_WAN), s2v_utils.py (rope_precompute),
//   configs/wan_s2v_14B.py.
//
// Config (14B): dim=5120, num_heads=40, head_dim=128, num_layers=40, ffn_dim=13824,
//   freq_dim=256, text_dim=4096, text_len=512, in_dim=out_dim=16, patch=(1,2,2),
//   eps=1e-6, qk_norm=True, cross_attn_norm=True, cond_dim=16, audio_dim=1024,
//   num_audio_token=4 (-> 5 tokens/frame), enable_adain=True, adain_mode="attn_norm",
//   audio_inject_layers=[0,4,8,12,16,20,24,27,30,33,36,39], zero_timestep=True,
//   enable_framepack=True (M1 drop_motion_frames=True -> motioner SKIPPED).
//
// RoPE: 3D, theta=10000, axes split d-4*(d//6)=44, 2*(d//6)=42, 42 (== stock
//   WanParams default {44,42,42}). The ref latent uses a temporal anchor at grid
//   position 30 (t_f range = 1 single frame, grid [[30,0,0],[31,H,W],[1,H,W]]).
//
// M1 SCOPE NOTES / KNOWN RISKS (see HANDOFF report):
//   - 2-segment modulation: noisy tokens (slot0 @ t) and ref token (slot1 @ t=0).
//     seg_idx = original_seq_len (number of noisy tokens).
//   - audio injection: per-frame cross-attn after the 12 inject layers; each video
//     frame's spatial tokens attend that frame's 5 audio tokens. q:5120, k/v:1024.
//     enable_adain attn_norm path -> AdaLayerNorm(injector_adain_layers) modulates
//     the hidden states with audio_emb_global before the cross-attn.
//   - cond_encoder (Conv3d 16->5120): M1 cond_states=zeros -> +0, but weight loaded.
//   - GGUF names mirror the PyTorch state_dict (match NAMING.md when produced).
namespace WAN_S2V {

    using WAN::Head;
    using WAN::modulate_add;
    using WAN::modulate_mul;
    using WAN::WanAttentionBlock;
    using WAN::WanSelfAttention;
    using WAN::WanT2VCrossAttention;

    struct WanS2VParams {
        std::tuple<int, int, int> patch_size = {1, 2, 2};
        int64_t text_len                     = 512;
        int64_t in_dim                       = 16;
        int64_t dim                          = 5120;
        int64_t ffn_dim                      = 13824;
        int freq_dim                         = 256;
        int64_t text_dim                     = 4096;
        int64_t out_dim                      = 16;
        int64_t num_heads                    = 40;
        int num_layers                       = 40;
        bool qk_norm                         = true;
        bool cross_attn_norm                 = true;
        float eps                            = 1e-6f;
        int theta                            = 10000;
        std::vector<int> axes_dim            = {44, 42, 42};
        int64_t axes_dim_sum                 = 128;

        // S2V extras
        int64_t cond_dim         = 16;
        int64_t audio_dim        = 1024;  // injector k/v dim
        int num_audio_token      = 4;     // -> 5 tokens/frame
        bool enable_adain        = true;
        std::vector<int> audio_inject_layers = {0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39};
        bool zero_timestep       = true;
        int ref_t_offset         = 30;  // ref latent temporal anchor grid pos
    };

    // ----------------------------------------------------------------------
    // WanS2VAttentionBlock: identical parameter layout to stock WanAttentionBlock
    // (modulation [dim,6,1], norm1/2/3, self_attn q/k/v/o + norm_q/k, cross_attn
    // q/k/v/o, ffn.0/ffn.2). The ONLY forward difference is the 2-segment
    // modulation (slot0 for tokens [0:seg], slot1 for [seg:]) which collapses to
    // stock single-modulation when seg==token_count (slots equal w/o zero_timestep).
    // ----------------------------------------------------------------------
    class WanS2VAttentionBlock : public WanAttentionBlock {
    public:
        WanS2VAttentionBlock(int64_t dim,
                             int64_t ffn_dim,
                             int64_t num_heads,
                             bool qk_norm         = true,
                             bool cross_attn_norm = true,
                             float eps            = 1e-6)
            // t2v_cross_attn=true -> WanT2VCrossAttention (text-only cross-attn, no img).
            : WanAttentionBlock(true, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps) {}

        // x: [N, n_token, dim]
        // e: [N, 6, 2, dim]  (slot dim of size 2: [noisy@t, ref@t=0])
        // pe: [n_token, d_head/2, 2, 2]
        // context: [N, ctx_len, dim]
        // seg: number of noisy tokens (= original_seq_len); tokens [0:seg] use slot0.
        ggml_tensor* forward_s2v(GGMLRunnerContext* ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* e,
                                 ggml_tensor* pe,
                                 ggml_tensor* context,
                                 int64_t seg) {
            auto modulation = params["modulation"];  // [dim, 6, 1]
            int64_t dim     = x->ne[0];
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];

            // e: [dim, 6, 2, N]; modulation broadcast over slot/batch -> add.
            // modulation [dim,6,1] -> [dim,6,1,1]
            auto mod4 = ggml_reshape_4d(ctx->ggml_ctx, modulation, dim, 6, 1, 1);
            e         = ggml_add(ctx->ggml_ctx, e, mod4);  // [dim, 6, 2, N]
            // chunk into 6 along dim1 -> each [dim, 1, 2, N]
            auto es = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WAN::WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);

            // helper: extract slot s of a chunked e [dim,1,2,N] -> broadcastable [dim,1,N]
            auto slot = [&](ggml_tensor* ec, int s) -> ggml_tensor* {
                auto v = ggml_view_4d(ctx->ggml_ctx, ec, dim, 1, 1, N,
                                      ec->nb[1], ec->nb[2], ec->nb[3], (size_t)s * ec->nb[2]);
                return ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, v), dim, 1, N);  // [dim,1,N]
            };

            // 2-segment modulate-affine on a [N, n_token, dim] tensor:
            //   tokens[0:seg]  -> y*(1+e_scale[slot0]) + e_shift[slot0]
            //   tokens[seg:]   -> y*(1+e_scale[slot1]) + e_shift[slot1]
            auto seg_modulate = [&](ggml_tensor* y, ggml_tensor* e_shift, ggml_tensor* e_scale) -> ggml_tensor* {
                if (seg <= 0 || seg >= n_token) {
                    // single segment (collapses to stock) — use slot0.
                    auto s0 = slot(e_scale, 0);
                    auto a0 = slot(e_shift, 0);
                    y = ggml_add(ctx->ggml_ctx, y, ggml_mul(ctx->ggml_ctx, y, s0));
                    y = ggml_add(ctx->ggml_ctx, y, a0);
                    return y;
                }
                auto y0 = ggml_view_3d(ctx->ggml_ctx, y, dim, seg, N, y->nb[1], y->nb[2], 0);
                auto y1 = ggml_view_3d(ctx->ggml_ctx, y, dim, n_token - seg, N, y->nb[1], y->nb[2], (size_t)seg * y->nb[1]);
                y0      = ggml_cont(ctx->ggml_ctx, y0);
                y1      = ggml_cont(ctx->ggml_ctx, y1);
                auto s0 = slot(e_scale, 0), s1 = slot(e_scale, 1);
                auto a0 = slot(e_shift, 0), a1 = slot(e_shift, 1);
                y0 = ggml_add(ctx->ggml_ctx, y0, ggml_mul(ctx->ggml_ctx, y0, s0));
                y0 = ggml_add(ctx->ggml_ctx, y0, a0);
                y1 = ggml_add(ctx->ggml_ctx, y1, ggml_mul(ctx->ggml_ctx, y1, s1));
                y1 = ggml_add(ctx->ggml_ctx, y1, a1);
                return ggml_concat(ctx->ggml_ctx, y0, y1, 1);  // [N, n_token, dim]
            };

            // 2-segment gate (y * e_gate[slot]) — no +1.
            auto seg_gate = [&](ggml_tensor* y, ggml_tensor* e_gate) -> ggml_tensor* {
                if (seg <= 0 || seg >= n_token) {
                    return ggml_mul(ctx->ggml_ctx, y, slot(e_gate, 0));
                }
                auto y0 = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, y, dim, seg, N, y->nb[1], y->nb[2], 0));
                auto y1 = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, y, dim, n_token - seg, N, y->nb[1], y->nb[2], (size_t)seg * y->nb[1]));
                y0 = ggml_mul(ctx->ggml_ctx, y0, slot(e_gate, 0));
                y1 = ggml_mul(ctx->ggml_ctx, y1, slot(e_gate, 1));
                return ggml_concat(ctx->ggml_ctx, y0, y1, 1);
            };

            // self-attention
            auto y = norm1->forward(ctx, x);
            y      = seg_modulate(y, es[0], es[1]);  // (1+e1)*x + e0
            y      = self_attn->forward(ctx, y, pe);
            x      = ggml_add(ctx->ggml_ctx, x, seg_gate(y, es[2]));

            // cross-attention (text)
            x = ggml_add(ctx->ggml_ctx, x, cross_attn->forward(ctx, norm3->forward(ctx, x), context, 0));

            // ffn
            y = norm2->forward(ctx, x);
            y = seg_modulate(y, es[3], es[4]);
            y = ffn_0->forward(ctx, y);
            y = ggml_ext_gelu(ctx->ggml_ctx, y, true);
            y = ffn_2->forward(ctx, y);
            x = ggml_add(ctx->ggml_ctx, x, seg_gate(y, es[5]));

            return x;
        }
    };

    // ----------------------------------------------------------------------
    // Head_S2V: same as stock Head (modulation [dim,2,1]); e is [N, dim] (e, not e0).
    // Reuse WAN::Head directly — its forward(x, e) matches Head_S2V.forward.
    // ----------------------------------------------------------------------

    // ----------------------------------------------------------------------
    // AudioInjector cross-attn block (AudioCrossAttention == WanCrossAttention with
    // q:dim, k/v:audio_dim). Plus injector_pre_norm_feat (LN no-affine) OR
    // injector_adain_layers (AdaLayerNorm) per inject layer.
    //
    // AdaLayerNorm(output_dim=dim*2, embedding_dim=dim, chunk_dim=1): a SiLU+Linear
    // producing [scale, shift] from temb=audio_emb_global, applied as
    // LN(x)*(1+scale)+shift. (diffusers AdaLayerNorm.)
    // ----------------------------------------------------------------------
    class AudioCrossAttention : public GGMLBlock {
    protected:
        int64_t dim, kv_dim, num_heads, head_dim;
        bool qk_norm;
        float eps;

    public:
        AudioCrossAttention(int64_t dim, int64_t kv_dim, int64_t num_heads, bool qk_norm = true, float eps = 1e-6f)
            : dim(dim), kv_dim(kv_dim), num_heads(num_heads), qk_norm(qk_norm), eps(eps) {
            head_dim    = dim / num_heads;
            blocks["q"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["k"] = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, dim));
            blocks["v"] = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, dim));
            blocks["o"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            if (qk_norm) {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new Identity());
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        // x: [B*F, n_spatial, dim]; context (audio): [B*F, n_audio_tok, kv_dim].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = norm_q->forward(ctx, q_proj->forward(ctx, x));
            auto k = norm_k->forward(ctx, k_proj->forward(ctx, context));
            auto v = v_proj->forward(ctx, context);
            // NB: this cross-attn is BATCHED over the F latent frames (q ne[2]=F as the
            // batch dim). The flash-attn wrapper's output view assumes N==1 and would
            // assert on the reshape for N>1 — and the attention here is tiny (L_q=h*w,
            // L_k=5 audio tokens), so use the non-flash dense path unconditionally.
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, /*flash_attn=*/false);
            x = o_proj->forward(ctx, x);
            return x;
        }
    };

    // AdaLayerNorm: SiLU(temb) -> Linear(dim -> 2*dim) -> [shift, scale]; LN(x) no-affine
    // then x*(1+scale)+shift. (chunk_dim=1: scale,shift split of the projected vec.)
    class AdaLayerNorm : public GGMLBlock {
    protected:
        int64_t dim;
        float eps;

    public:
        AdaLayerNorm(int64_t dim, float eps = 1e-6f)
            : dim(dim), eps(eps) {
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 2, true));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false, false));
        }

        // x: [B*F, n, dim]; temb: [B*F, dim] (audio_emb_global per frame).
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* temb) {
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto norm   = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto emb    = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, temb));  // [B*F, 2*dim]
            // split into shift, scale along dim0.
            int64_t BF  = emb->ne[1];
            auto shift  = ggml_view_2d(ctx->ggml_ctx, emb, dim, BF, emb->nb[1], 0);
            auto scale  = ggml_view_2d(ctx->ggml_ctx, emb, dim, BF, emb->nb[1], (size_t)dim * emb->nb[0]);
            shift       = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, shift), dim, 1, BF);
            scale       = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, scale), dim, 1, BF);
            x = norm->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale));
            x = ggml_add(ctx->ggml_ctx, x, shift);
            return x;
        }
    };

    // ----------------------------------------------------------------------
    // WanS2V DiT.
    // ----------------------------------------------------------------------
    class WanS2V : public GGMLBlock {
    protected:
        WanS2VParams params;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // trainable_cond_mask is an Embedding(3, dim) — declared as a block below.
            (void)ctx;
            (void)tensor_storage_map;
            (void)prefix;
        }

    public:
        // map inject-layer block index -> injector slot (0..11)
        std::map<int, int> inject_id;

        WanS2V() {}
        WanS2V(WanS2VParams params)
            : params(params) {
            blocks["patch_embedding"]  = std::shared_ptr<GGMLBlock>(new Conv3d(params.in_dim, params.dim, params.patch_size, params.patch_size));
            blocks["cond_encoder"]     = std::shared_ptr<GGMLBlock>(new Conv3d(params.cond_dim, params.dim, params.patch_size, params.patch_size));

            blocks["text_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(params.text_dim, params.dim));
            blocks["text_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim));

            blocks["time_embedding.0"]  = std::shared_ptr<GGMLBlock>(new Linear(params.freq_dim, params.dim));
            blocks["time_embedding.2"]  = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim));
            blocks["time_projection.1"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim * 6));

            for (int i = 0; i < params.num_layers; i++) {
                blocks["blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new WanS2VAttentionBlock(params.dim, params.ffn_dim, params.num_heads,
                                                                        params.qk_norm, params.cross_attn_norm, params.eps));
            }

            blocks["head"]                = std::shared_ptr<GGMLBlock>(new Head(params.dim, params.out_dim, params.patch_size, params.eps));
            blocks["trainable_cond_mask"] = std::shared_ptr<GGMLBlock>(new Embedding(3, params.dim));

            // audio injectors (12)
            int slot = 0;
            for (int layer : params.audio_inject_layers) {
                inject_id[layer] = slot;
                // AudioInjector_WAN.injector = AudioCrossAttention(dim=self.dim, ...): the
                // injector is a WanCrossAttention whose k/v project from dim (5120), NOT
                // audio_dim. The audio context (merged_audio_emb) is already CausalAudio-
                // Encoder out_dim==dim==5120. (audio_dim=1024 is the wav2vec hidden size
                // FEEDING the encoder, not the injector context.)
                blocks["audio_injector.injector." + std::to_string(slot)] =
                    std::shared_ptr<GGMLBlock>(new AudioCrossAttention(params.dim, params.dim, params.num_heads, true, params.eps));
                if (params.enable_adain) {
                    blocks["audio_injector.injector_adain_layers." + std::to_string(slot)] =
                        std::shared_ptr<GGMLBlock>(new AdaLayerNorm(params.dim, params.eps));
                } else {
                    blocks["audio_injector.injector_pre_norm_feat." + std::to_string(slot)] =
                        std::shared_ptr<GGMLBlock>(new LayerNorm(params.dim, params.eps, false, false));
                }
                slot++;
            }
        }

        // Patchify+flatten a latent through a Conv3d -> [N, L, dim].
        ggml_tensor* patch_flatten(GGMLRunnerContext* ctx, std::shared_ptr<Conv3d> conv, ggml_tensor* lat,
                                   int64_t& t_len, int64_t& h_len, int64_t& w_len) {
            auto e = conv->forward(ctx, lat);  // [dim, t_len, h_len, w_len] in ggml ne [w,h,t,dim]
            w_len  = e->ne[0];
            h_len  = e->ne[1];
            t_len  = e->ne[2];
            e = ggml_reshape_3d(ctx->ggml_ctx, e, e->ne[0] * e->ne[1] * e->ne[2], e->ne[3], 1);  // [1, dim, L]
            e = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, e, 1, 0, 2, 3));  // [1, L, dim]
            return e;
        }

        // forward.
        //   x:           [in_dim, T, H, W]  noisy latent (ggml ne [W,H,T,C])
        //   cond_states: [cond_dim, T, H, W] (M1 zeros) OR nullptr
        //   ref_latent:  [in_dim, 1, H, W]
        //   timestep:    [1] (noisy t).
        //   timestep0:   [1] (=0) for the ref slot when zero_timestep; nullptr otherwise.
        //   context:     [text_len, text_dim] (umT5, padded to 512)
        //   cond_mask_ids:[L] int32, 0 for noisy / 1 for ref (built host-side).
        //   audio_tokens: [audio_dim, n_tok(5), F]  per-frame audio (k/v for injectors)
        //   audio_global: [dim, 1, F] per-frame global (AdaLN temb) or nullptr
        //   pe:          precomputed RoPE for the [noisy ++ ref] token sequence.
        // Returns the predicted velocity latent [out_dim, T, H, W].
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* cond_states,
                             ggml_tensor* ref_latent,
                             ggml_tensor* timestep,
                             ggml_tensor* timestep0,
                             ggml_tensor* context,
                             ggml_tensor* cond_mask_ids,
                             ggml_tensor* audio_tokens,
                             ggml_tensor* audio_global,
                             ggml_tensor* pe) {
            auto patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);
            auto cond_encoder    = std::dynamic_pointer_cast<Conv3d>(blocks["cond_encoder"]);
            auto text_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto text_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);
            auto time_embedding_0  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto time_embedding_2  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);
            auto head              = std::dynamic_pointer_cast<Head>(blocks["head"]);
            auto cond_mask         = std::dynamic_pointer_cast<Embedding>(blocks["trainable_cond_mask"]);

            int64_t t_len, h_len, w_len;
            auto xt = patch_flatten(ctx, patch_embedding, x, t_len, h_len, w_len);  // [1, L_noisy, dim]
            if (cond_states != nullptr) {
                int64_t ct, ch, cw;
                auto cond = patch_flatten(ctx, cond_encoder, cond_states, ct, ch, cw);
                xt = ggml_add(ctx->ggml_ctx, xt, cond);
            }
            int64_t L_noisy = xt->ne[1];

            // ref latent -> 1 frame tokens, concat.
            int64_t rt, rh, rw;
            auto ref = patch_flatten(ctx, patch_embedding, ref_latent, rt, rh, rw);  // [1, L_ref, dim]
            int64_t L_ref = ref->ne[1];
            auto xc = ggml_concat(ctx->ggml_ctx, xt, ref, 1);  // [1, L_noisy+L_ref, dim]
            int64_t L = xc->ne[1];
            int64_t dim = params.dim;

            // trainable_cond_mask: 0 for noisy, 1 for ref. ids built host-side.
            {
                if (getenv("S2V_DIT_DEBUG"))
                    fprintf(stderr, "[DiT] cond_mask_ids type=%d (I32=%d) ne=[%lld,%lld] L=%lld\n",
                            (int)cond_mask_ids->type, (int)GGML_TYPE_I32,
                            (long long)cond_mask_ids->ne[0], (long long)cond_mask_ids->ne[1], (long long)L);
                if (getenv("S2V_SKIP_CONDMASK") == nullptr) {
                    auto ids = ggml_reshape_2d(ctx->ggml_ctx, cond_mask_ids, L, 1);  // [L,1] int32
                    auto mask_emb = cond_mask->forward(ctx, ids);  // [1, L, dim]
                    xc = ggml_add(ctx->ggml_ctx, xc, mask_emb);
                }
            }

            // time embedding (noisy t). e: [N, dim].
            auto time_emb = [&](ggml_tensor* ts) -> ggml_tensor* {
                auto te = ggml_ext_timestep_embedding(ctx->ggml_ctx, ts, params.freq_dim);
                te      = time_embedding_0->forward(ctx, te);
                te      = ggml_silu_inplace(ctx->ggml_ctx, te);
                te      = time_embedding_2->forward(ctx, te);  // [N, dim]
                return te;
            };
            auto proj_e0 = [&](ggml_tensor* te) -> ggml_tensor* {
                auto p = ggml_silu(ctx->ggml_ctx, te);
                p      = time_projection_1->forward(ctx, p);
                return ggml_reshape_4d(ctx->ggml_ctx, p, dim, 6, 1, p->ne[1]);  // [dim,6,1,N]
            };

            auto e  = time_emb(timestep);    // [N, dim] (used by Head)
            auto e0 = proj_e0(e);            // [dim, 6, 1, N] noisy slot

            // 2-slot e0: slot0 = e0(t); slot1 = e0(t=0) when zero_timestep, else e0(t).
            ggml_tensor* e0_ref = e0;
            if (params.zero_timestep && timestep0 != nullptr) {
                e0_ref = proj_e0(time_emb(timestep0));
            }
            ggml_tensor* e0_2 = ggml_concat(ctx->ggml_ctx, e0, e0_ref, 2);  // [dim, 6, 2, N]
            int64_t seg = params.zero_timestep ? L_noisy : 0;  // seg_idx

            // text context.
            context = text_embedding_0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context);
            context = text_embedding_2->forward(ctx, context);  // [N, text_len, dim]

            int64_t num_frames = t_len;  // latent frames (audio per-frame count)

            bool dbg = getenv("S2V_DIT_DEBUG") != nullptr;
            auto DD = [&](const char* tag, ggml_tensor* t) {
                if (dbg) fprintf(stderr, "[DiT] %-16s ne=[%lld,%lld,%lld,%lld]\n", tag,
                                 (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
            };
            if (dbg) fprintf(stderr, "[DiT] t_len=%lld h_len=%lld w_len=%lld L_noisy=%lld L_ref=%lld L=%lld seg=%lld\n",
                             (long long)t_len, (long long)h_len, (long long)w_len, (long long)L_noisy, (long long)L_ref, (long long)L, (long long)seg);
            DD("xc(pre-blocks)", xc);
            DD("e0_2", e0_2);
            DD("pe", pe);
            DD("context", context);
            DD("audio_tokens", audio_tokens);

            for (int i = 0; i < params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<WanS2VAttentionBlock>(blocks["blocks." + std::to_string(i)]);
                xc = block->forward_s2v(ctx, xc, e0_2, pe, context, seg);
                if (i == 0) DD("xc(after blk0)", xc);

                auto it = inject_id.find(i);
                if (it != inject_id.end() && audio_tokens != nullptr && getenv("S2V_SKIP_INJECT") == nullptr) {
                    xc = audio_inject(ctx, xc, it->second, L_noisy, num_frames, h_len, w_len,
                                      audio_tokens, audio_global);
                    if (i == 0) DD("xc(after inj0)", xc);
                }
            }

            // slice to noisy tokens and head.
            auto out = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, xc, dim, L_noisy, xc->ne[2], xc->nb[1], xc->nb[2], 0));
            out      = head->forward(ctx, out, e);  // [N, L_noisy, pt*ph*pw*out_dim]

            // unpatchify back to [out_dim, T, H, W].
            out = unpatchify(ctx->ggml_ctx, out, t_len, h_len, w_len);
            return out;
        }

        // after_transformer_block audio injection for one inject layer.
        //   xc: [1, L_noisy+L_ref, dim]; we operate on the first L_noisy tokens.
        //   The noisy tokens are [num_frames, h_len*w_len] spatial; each frame's
        //   (h_len*w_len) tokens cross-attend that frame's audio tokens.
        ggml_tensor* audio_inject(GGMLRunnerContext* ctx, ggml_tensor* xc, int slot, int64_t L_noisy,
                                  int64_t num_frames, int64_t h_len, int64_t w_len,
                                  ggml_tensor* audio_tokens, ggml_tensor* audio_global) {
            int64_t dim = params.dim;
            int64_t n_spatial = h_len * w_len;
            // h = noisy tokens [1, L_noisy, dim] -> reshape [(num_frames), n_spatial, dim]
            // == (b t) n c with b=1.
            auto h = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, xc, dim, L_noisy, 1, xc->nb[1], xc->nb[2], 0));
            h      = ggml_reshape_3d(ctx->ggml_ctx, h, dim, n_spatial, num_frames);  // [num_frames, n_spatial, dim]

            // pre-norm / adain.
            ggml_tensor* hn;
            if (params.enable_adain) {
                auto adain = std::dynamic_pointer_cast<AdaLayerNorm>(blocks["audio_injector.injector_adain_layers." + std::to_string(slot)]);
                // temb = audio_emb_global[:,0] per frame: audio_global [dim, 1, F] -> [dim, F]
                auto temb = ggml_reshape_2d(ctx->ggml_ctx, audio_global, dim, num_frames);
                hn = adain->forward(ctx, h, temb);
            } else {
                auto pre = std::dynamic_pointer_cast<LayerNorm>(blocks["audio_injector.injector_pre_norm_feat." + std::to_string(slot)]);
                hn = pre->forward(ctx, h);
            }

            // audio context per frame: audio_tokens ggml ne = [audio_dim, n_tok, F]
            // == [c, n_tok, batch=F], which matches the cross-attn context layout
            // ([N=F, n_tok, audio_dim]) directly.
            auto inj = std::dynamic_pointer_cast<AudioCrossAttention>(blocks["audio_injector.injector." + std::to_string(slot)]);
            auto out = inj->forward(ctx, hn, audio_tokens);  // [F, n_spatial, dim]

            // rearrange back (b t) n c -> b (t n) c.
            out = ggml_reshape_3d(ctx->ggml_ctx, out, dim, n_spatial * num_frames, 1);  // [1, L_noisy, dim]

            // add residual to the first L_noisy tokens of xc.
            // build padded residual [1, L, dim] (zeros for ref tail) then add.
            int64_t L = xc->ne[1];
            if (L > L_noisy) {
                // NB: must be a TRUE zero fill. The old `ggml_scale(ggml_new_tensor, 0)`
                // multiplied an *uninitialized* compute-buffer tensor by 0 — on a reused
                // graph buffer that scratch can hold Inf/NaN from a prior op, and
                // 0*Inf = NaN, poisoning the whole sequence from the 2nd forward on.
                auto zeros = ggml_ext_zeros(ctx->ggml_ctx, dim, L - L_noisy, 1, 1);
                zeros      = ggml_reshape_3d(ctx->ggml_ctx, zeros, dim, L - L_noisy, 1);
                out        = ggml_concat(ctx->ggml_ctx, out, zeros, 1);  // [1, L, dim]
            }
            xc = ggml_add(ctx->ggml_ctx, xc, out);
            return xc;
        }

        // unpatchify [N, t*h*w, pt*ph*pw*C] -> [N*C, t*pt, h*ph, w*pw].
        ggml_tensor* unpatchify(ggml_context* ctx, ggml_tensor* x, int64_t t_len, int64_t h_len, int64_t w_len) {
            int64_t N  = x->ne[2];
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

        const WanS2VParams& get_params() const { return params; }
    };

    // ----------------------------------------------------------------------
    // WanS2VRunner: owns the WanS2V DiT params, builds the RoPE (noisy grid +
    // ref anchor @ t=30), the cond-mask ids, and runs one forward pass.
    //
    // RoPE: the ref latent's temporal grid anchor is `ref_t_offset` (30) for a
    // single frame; rope_precompute in the reference builds freqs for the
    // concatenated [noisy(grid 0..t-1) ++ ref(grid 30)] sequence. We reproduce
    // this with Rope::gen_vid_ids for the noisy block (t_offset=0) + the ref block
    // (t_offset=ref_t_offset) and embed_nd over the same axes_dim.
    // ----------------------------------------------------------------------
    struct WanS2VRunner : public GGMLRunner {
        WanS2VParams params;
        WanS2V dit;
        std::vector<float> pe_vec;
        // make_input() stores the host data POINTER and copies later (after get_graph
        // returns). These backing tensors MUST outlive the lambda, so keep them as
        // members (like pe_vec) — locals would dangle and feed garbage to the graph.
        sd::Tensor<int32_t> mask_t_hold;
        sd::Tensor<float> zero_ts_hold;
        std::string desc = "Wan2.2-S2V-14B";

        WanS2VRunner(ggml_backend_t backend,
                     ggml_backend_t params_backend,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "model.diffusion_model")
            : GGMLRunner(backend, params_backend) {
            dit = WanS2V(params);
            dit.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return desc; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            dit.get_param_tensors(tensors, prefix);
        }

        // Build the concatenated [noisy ++ ref] RoPE table.
        std::vector<float> build_pe(int t, int h, int w) {
            int pt = std::get<0>(params.patch_size);
            int ph = std::get<1>(params.patch_size);
            int pw = std::get<2>(params.patch_size);
            // noisy grid: t_offset 0
            auto noisy_ids = Rope::gen_vid_ids(t, h, w, pt, ph, pw, 1, 0, 0, 0);
            // ref grid: 1 temporal frame at t_offset = ref_t_offset
            auto ref_ids = Rope::gen_vid_ids(pt /*=1 frame*/, h, w, pt, ph, pw, 1, params.ref_t_offset, 0, 0);
            std::vector<std::vector<float>> ids = noisy_ids;
            ids.insert(ids.end(), ref_ids.begin(), ref_ids.end());
            return Rope::embed_nd(ids, 1, static_cast<float>(params.theta), params.axes_dim);
        }

        // Compute one DiT forward. All inputs are host sd::Tensors in ggml-ne layout.
        //   x:           [W, H, T, in_dim]
        //   ref_latent:  [W, H, 1, in_dim]
        //   cond_states: [W, H, T, cond_dim] (zeros for M1) or empty
        //   timestep:    [1]; the ref slot uses t=0 (zero_timestep).
        //   context:     [text_dim, text_len, 1]
        //   audio_tokens:[audio_dim, n_tok, F]
        //   audio_global:[dim, 1, F] or empty
        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& ref_latent,
                                  const sd::Tensor<float>& cond_states,
                                  const sd::Tensor<float>& timestep,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& audio_tokens,
                                  const sd::Tensor<float>& audio_global) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(WAN::WAN_GRAPH_SIZE);

                ggml_tensor* gx   = make_input(x);
                ggml_tensor* gref = make_input(ref_latent);
                ggml_tensor* gts  = make_input(timestep);
                ggml_tensor* gctx = make_input(context);
                ggml_tensor* gat  = make_input(audio_tokens);
                ggml_tensor* gcond = cond_states.empty() ? nullptr : make_input(cond_states);
                ggml_tensor* gag  = audio_global.empty() ? nullptr : make_input(audio_global);

                // ref-slot t=0.  (member-held: see mask_t_hold note above)
                zero_ts_hold = sd::Tensor<float>::from_vector(std::vector<float>{0.0f});
                ggml_tensor* gts0 = params.zero_timestep ? make_input(zero_ts_hold) : nullptr;

                // dims of the patchified grid (for PE + mask).
                int T = (int)x.shape()[2], H = (int)x.shape()[1], W = (int)x.shape()[0];
                int pt = std::get<0>(params.patch_size), ph = std::get<1>(params.patch_size), pw = std::get<2>(params.patch_size);
                int t_len = (T + pt / 2) / pt, h_len = (H + ph / 2) / ph, w_len = (W + pw / 2) / pw;
                int L_noisy = t_len * h_len * w_len;
                int L_ref   = 1 * h_len * w_len;  // ref is 1 latent frame
                int L = L_noisy + L_ref;

                // cond-mask ids: 0 for noisy, 1 for ref.  (member-held — see note above)
                std::vector<int32_t> mask_ids(L, 0);
                for (int i = L_noisy; i < L; i++) mask_ids[i] = 1;
                mask_t_hold = sd::Tensor<int32_t>::from_vector(mask_ids);
                ggml_tensor* gmask = make_input(mask_t_hold);

                // RoPE.
                pe_vec = build_pe(T, H, W);
                int pos_len = (int)(pe_vec.size() / params.axes_dim_sum / 2);
                auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, params.axes_dim_sum / 2, pos_len);
                set_backend_tensor_data(pe, pe_vec.data());

                auto runner_ctx = get_context();
                ggml_tensor* out = dit.forward(&runner_ctx, gx, gcond, gref, gts, gts0, gctx, gmask, gat, gag, pe);
                ggml_build_forward_expand(gf, out);
                (void)L;
                return gf;
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };

}  // namespace WAN_S2V

#endif  // __WAN_S2V_HPP__
