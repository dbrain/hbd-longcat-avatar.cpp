#ifndef __INFINITETALK_HPP__
#define __INFINITETALK_HPP__

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "ggml_extend.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/wan.hpp"  // reuse WAN::WanAttentionBlock / WanI2VCrossAttention / MLPProj / Head / modulate_*

// ---------------------------------------------------------------------------
// InfiniteTalk (MultiTalk on Wan2.1-I2V-14B) DiT.
//
// Ground truth: github.com/MeiGen-AI/InfiniteTalk wan/modules/multitalk_model.py
//   (WanModel, WanAttentionBlock, AudioProjModel), wan/modules/attention.py
//   (SingleStreamAttention / SingleStreamMutiAttention, human_num==1 path).
//
// The backbone IS the stock Wan2.1-I2V-14B already in wan.hpp (num_layers=40,
// in_dim=36, dim=5120, num_heads=40, ffn_dim=13824, WanI2VCrossAttention with
// k_img/v_img + 257 CLIP tokens, MLPProj img_emb, Head). The ONLY additions are:
//   1) a per-block audio graft inserted AFTER the text cross-attn, BEFORE the FFN:
//        x = x + audio_cross_attn(norm_x(x), audio_embedding)
//      on ALL 40 layers. audio_cross_attn is SingleStreamAttention (human_num==1):
//      q_linear[dim,dim], FUSED kv_linear[2*dim, audio_dim(768)], proj[dim,dim],
//      NO qk_norm; norm_x is an affine LayerNorm. Per-frame cross-attn: each latent
//      frame's spatial tokens attend that frame's 32 audio tokens.
//   2) AudioProjModel (audio_proj.*): wav2vec hidden-state windows -> 32 tokens/frame
//      @768. Runs ONCE per clip window (step/CFG-invariant) via compute_audio_embedding.
//
// Single-speaker only: the SingleStreamMutiAttention RoPE-speaker-class + x_ref_attn_map
// (human_num>1) are skipped, exactly as the reference's `if human_num == 1: return
// super().forward(...)` early-out.
namespace IT {

    using WAN::Head;
    using WAN::MLPProj;
    using WAN::modulate_add;
    using WAN::modulate_mul;
    using WAN::WanAttentionBlock;

    struct InfiniteTalkConfig {
        std::tuple<int, int, int> patch_size = {1, 2, 2};
        int64_t text_len                     = 512;
        int64_t in_dim                       = 36;  // 16 noisy + 20 c_concat (4ch mask + 16ch VAE)
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

        // audio adapter
        int64_t audio_dim     = 768;   // AudioProjModel output / kv_linear input
        int audio_window      = 5;     // first-frame proj1 window
        int vae_scale         = 4;     // temporal VAE downsample (proj1_vf window = audio_window+vae_scale-1 = 8)
        int blocks_per_token  = 12;    // wav2vec hidden layers stacked
        int intermediate_dim  = 512;
        int context_tokens    = 32;    // audio tokens per latent frame
    };

    // ----------------------------------------------------------------------
    // SingleStreamAttention (human_num==1): plain per-frame audio cross-attn.
    //   q  = q_linear(visual)                         [dim,dim]
    //   kv = kv_linear(audio)  -> split k|v            FUSED [2*dim, audio_dim]
    //   x  = proj( attn(q, k, v) )                     [dim,dim]
    // qkv_bias=True, qk_norm=False (q_norm/k_norm are Identity in the reference).
    // ----------------------------------------------------------------------
    class AudioCrossAttention : public GGMLBlock {
    protected:
        int64_t dim, audio_dim, num_heads, head_dim;

    public:
        AudioCrossAttention(int64_t dim, int64_t audio_dim, int64_t num_heads)
            : dim(dim), audio_dim(audio_dim), num_heads(num_heads) {
            head_dim             = dim / num_heads;
            blocks["q_linear"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, true));
            blocks["kv_linear"]  = std::shared_ptr<GGMLBlock>(new Linear(audio_dim, dim * 2, true));
            blocks["proj"]       = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, true));
        }

        // x:     [1, L, dim]            (norm_x(x); L = n_frames * hw)
        // audio: [audio_dim, n_a, n_frames]  (n_a = 32 tokens/frame, batched over n_frames)
        // returns [1, L, dim].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* audio,
                             int64_t n_frames, int64_t hw) {
            auto q_lin  = std::dynamic_pointer_cast<Linear>(blocks["q_linear"]);
            auto kv_lin = std::dynamic_pointer_cast<Linear>(blocks["kv_linear"]);
            auto proj   = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            // rearrange "B (N_t S) C -> (B N_t) S C": batch over the n_frames latent frames.
            x      = ggml_reshape_3d(ctx->ggml_ctx, x, dim, hw, n_frames);  // [n_frames, hw, dim]
            auto q = q_lin->forward(ctx, x);                                // [n_frames, hw, dim]

            // fused kv: split first dim -> k, last dim -> v (view(...,2,heads,hd) outer "2").
            auto kv      = kv_lin->forward(ctx, audio);  // [2*dim, n_a, n_frames]
            int64_t n_a  = audio->ne[1];
            auto k = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, kv, dim, n_a, n_frames,
                                                           kv->nb[1], kv->nb[2], 0));
            auto v = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, kv, dim, n_a, n_frames,
                                                           kv->nb[1], kv->nb[2], (size_t)dim * kv->nb[0]));

            // batched (N=n_frames) dense attention; q L=hw, k/v L=n_a are tiny. The flash
            // wrapper's output view assumes N==1, so use the non-flash path (matches S2V).
            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads,
                                              nullptr, false, /*flash_attn=*/false);  // [n_frames, hw, dim]
            out = proj->forward(ctx, out);
            // rearrange back "(B N_t) S C -> B (N_t S) C".
            out = ggml_reshape_3d(ctx->ggml_ctx, out, dim, hw * n_frames, 1);  // [1, L, dim]
            return out;
        }
    };

    // ----------------------------------------------------------------------
    // InfiniteTalkAttentionBlock = stock WanAttentionBlock (i2v) + audio graft.
    // Adds blocks["audio_cross_attn"] + blocks["norm_x"]; the modulation/norm1/2/3/
    // self_attn/cross_attn/ffn parameters are the parent's (loaded from the base
    // Wan2.1-I2V checkpoint), so the GGUF tensor layout matches verbatim.
    // ----------------------------------------------------------------------
    class InfiniteTalkAttentionBlock : public WanAttentionBlock {
    public:
        InfiniteTalkAttentionBlock(int64_t dim, int64_t ffn_dim, int64_t num_heads, int64_t audio_dim,
                                   bool qk_norm = true, bool cross_attn_norm = true, float eps = 1e-6)
            // t2v_cross_attn=false -> WanI2VCrossAttention (k_img/v_img + 257 CLIP tokens).
            : WanAttentionBlock(false, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps) {
            blocks["audio_cross_attn"] = std::shared_ptr<GGMLBlock>(new AudioCrossAttention(dim, audio_dim, num_heads));
            blocks["norm_x"]           = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, /*affine=*/true, /*bias=*/true));
        }

        // Mirrors WanAttentionBlock::forward with the audio graft inserted between the
        // text cross-attn and the FFN.  e: [dim,6,N] time-projection modulation.
        // audio: [audio_dim, n_a, n_frames] or nullptr (uncond -> graft skipped is NOT
        // correct; the reference feeds zeros, so the caller passes a zero tensor instead).
        ggml_tensor* forward_it(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* e,
                                ggml_tensor* pe,
                                ggml_tensor* context,
                                ggml_tensor* audio,
                                int64_t n_frames,
                                int64_t hw,
                                int64_t context_img_len) {
            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);  // [dim,6,N]
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);  // 6 x [dim,1,N]

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WAN::WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WAN::WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);
            auto audio_attn = std::dynamic_pointer_cast<AudioCrossAttention>(blocks["audio_cross_attn"]);
            auto norm_x     = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_x"]);

            // self-attention
            auto y = norm1->forward(ctx, x);
            y      = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[1]));
            y      = modulate_add(ctx->ggml_ctx, y, es[0]);
            y      = self_attn->forward(ctx, y, pe);
            x      = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[2]));

            // text cross-attention
            x = ggml_add(ctx->ggml_ctx, x,
                         cross_attn->forward(ctx, norm3->forward(ctx, x), context, context_img_len));

            // audio cross-attention graft
            if (audio != nullptr) {
                auto xa = audio_attn->forward(ctx, norm_x->forward(ctx, x), audio, n_frames, hw);
                x       = ggml_add(ctx->ggml_ctx, x, xa);
            }

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
    };

    // ----------------------------------------------------------------------
    // AudioProjModel (audio_proj.*): two parallel input projections (first latent
    // frame's 5-window, latter frames' 8-window), relu, concat along the frame dim,
    // proj2, proj3 -> [context_tokens, output_dim] per frame, output LayerNorm.
    // The first/middle/last window gather (multitalk_model.py:654-666) is done HOST
    // side in build_audio_proj_inputs(); this graph takes the two flattened inputs.
    // ----------------------------------------------------------------------
    class AudioProjModel : public GGMLBlock {
    protected:
        int64_t input_dim, input_dim_vf, intermediate_dim, context_tokens, output_dim;
        bool norm_output;
        float eps;

    public:
        AudioProjModel(const InfiniteTalkConfig& c, bool norm_output = true)
            : intermediate_dim(c.intermediate_dim),
              context_tokens(c.context_tokens),
              output_dim(c.audio_dim),
              norm_output(norm_output),
              eps(1e-5f) {
            int64_t seq_len_vf = c.audio_window + c.vae_scale - 1;  // 8
            input_dim          = (int64_t)c.audio_window * c.blocks_per_token * c.audio_dim;  // 46080
            input_dim_vf       = seq_len_vf * c.blocks_per_token * c.audio_dim;               // 73728

            blocks["proj1"]    = std::shared_ptr<GGMLBlock>(new Linear(input_dim, intermediate_dim, true));
            blocks["proj1_vf"] = std::shared_ptr<GGMLBlock>(new Linear(input_dim_vf, intermediate_dim, true));
            blocks["proj2"]    = std::shared_ptr<GGMLBlock>(new Linear(intermediate_dim, intermediate_dim, true));
            blocks["proj3"]    = std::shared_ptr<GGMLBlock>(new Linear(intermediate_dim, context_tokens * output_dim, true));
            if (norm_output) {
                // nn.LayerNorm(output_dim) (default eps 1e-5, affine).
                blocks["norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(output_dim, eps, true, true));
            }
        }

        // first_in: [input_dim, 1, 1]      (first latent frame's flattened 5-window)
        // vf_in:    [input_dim_vf, n_t, 1] (latter latent frames' flattened 8-windows)
        // returns audio_embedding [output_dim, context_tokens, 1 + n_t].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* first_in, ggml_tensor* vf_in) {
            auto proj1    = std::dynamic_pointer_cast<Linear>(blocks["proj1"]);
            auto proj1_vf = std::dynamic_pointer_cast<Linear>(blocks["proj1_vf"]);
            auto proj2    = std::dynamic_pointer_cast<Linear>(blocks["proj2"]);
            auto proj3    = std::dynamic_pointer_cast<Linear>(blocks["proj3"]);

            auto a  = ggml_relu(ctx->ggml_ctx, proj1->forward(ctx, first_in));    // [inter, 1]
            auto b  = ggml_relu(ctx->ggml_ctx, proj1_vf->forward(ctx, vf_in));    // [inter, n_t]
            auto c  = ggml_concat(ctx->ggml_ctx, a, b, 1);                        // [inter, 1+n_t]
            c       = ggml_relu(ctx->ggml_ctx, proj2->forward(ctx, c));           // [inter, N_t]
            c       = proj3->forward(ctx, c);                                     // [ctx_tok*out_dim, N_t]
            int64_t N_t = c->ne[1];
            c = ggml_reshape_3d(ctx->ggml_ctx, c, output_dim, context_tokens, N_t);  // [out_dim, ctx_tok, N_t]
            if (norm_output) {
                auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
                c         = norm->forward(ctx, c);
            }
            return c;  // [out_dim, ctx_tok, N_t]
        }
    };

    // ----------------------------------------------------------------------
    // The InfiniteTalk DiT: stock Wan2.1-I2V-14B front-end + InfiniteTalkAttentionBlock
    // layers + audio_proj.  forward() mirrors WAN::Wan::forward_orig, threading the
    // per-frame audio embedding to every block.
    // ----------------------------------------------------------------------
    class InfiniteTalk : public GGMLBlock {
    protected:
        InfiniteTalkConfig config;

    public:
        InfiniteTalk() {}
        explicit InfiniteTalk(const InfiniteTalkConfig& config)
            : config(config) {
            blocks["patch_embedding"]   = std::shared_ptr<GGMLBlock>(new Conv3d(config.in_dim, config.dim, config.patch_size, config.patch_size));
            blocks["text_embedding.0"]  = std::shared_ptr<GGMLBlock>(new Linear(config.text_dim, config.dim));
            blocks["text_embedding.2"]  = std::shared_ptr<GGMLBlock>(new Linear(config.dim, config.dim));
            blocks["time_embedding.0"]  = std::shared_ptr<GGMLBlock>(new Linear(config.freq_dim, config.dim));
            blocks["time_embedding.2"]  = std::shared_ptr<GGMLBlock>(new Linear(config.dim, config.dim));
            blocks["time_projection.1"] = std::shared_ptr<GGMLBlock>(new Linear(config.dim, config.dim * 6));

            for (int i = 0; i < config.num_layers; i++) {
                blocks["blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new InfiniteTalkAttentionBlock(config.dim, config.ffn_dim, config.num_heads,
                                                                              config.audio_dim, config.qk_norm,
                                                                              config.cross_attn_norm, config.eps));
            }

            blocks["head"]      = std::shared_ptr<GGMLBlock>(new Head(config.dim, config.out_dim, config.patch_size, config.eps));
            blocks["img_emb"]   = std::shared_ptr<GGMLBlock>(new MLPProj(1280, config.dim));
            blocks["audio_proj"] = std::shared_ptr<GGMLBlock>(new AudioProjModel(config, /*norm_output=*/true));
        }

        const InfiniteTalkConfig& get_config() const { return config; }

        std::shared_ptr<AudioProjModel> get_audio_proj() {
            return std::dynamic_pointer_cast<AudioProjModel>(blocks["audio_proj"]);
        }

        // unpatchify [N, t*h*w, pt*ph*pw*C] -> [N*C, t*pt, h*ph, w*pw]  (== WAN::Wan).
        ggml_tensor* unpatchify(ggml_context* ctx, ggml_tensor* x, int64_t t_len, int64_t h_len, int64_t w_len) {
            int64_t N  = x->ne[3];
            int64_t pt = std::get<0>(config.patch_size);
            int64_t ph = std::get<1>(config.patch_size);
            int64_t pw = std::get<2>(config.patch_size);
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

        ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx, ggml_tensor* x) {
            int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2];
            int pt = std::get<0>(config.patch_size), ph = std::get<1>(config.patch_size), pw = std::get<2>(config.patch_size);
            int pad_t = (pt - T % pt) % pt;
            int pad_h = (ph - H % ph) % ph;
            int pad_w = (pw - W % pw) % pw;
            ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, pad_t, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return x;
        }

        // x:           [in_dim(36), T, H, W]  (noisy 16ch ++ c_concat 20ch already concatenated)
        // timestep:    [N]
        // context:     [text_dim, text_len(512), N]  (umT5, padded)
        // clip_fea:    [1280, 257, N]               (CLIP-H of the cond frame)
        // audio:       [audio_dim, ctx_tok(32), t_len]  per-latent-frame audio tokens (or zeros)
        // pe:          precomputed RoPE table.
        // returns velocity latent [out_dim(16), T, H, W].
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* clip_fea,
                             ggml_tensor* audio,
                             ggml_tensor* pe) {
            auto patch_embedding   = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);
            auto text_embedding_0  = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto text_embedding_2  = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);
            auto time_embedding_0  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto time_embedding_2  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);
            auto head              = std::dynamic_pointer_cast<Head>(blocks["head"]);
            auto img_emb           = std::dynamic_pointer_cast<MLPProj>(blocks["img_emb"]);

            int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2];
            x = pad_to_patch_size(ctx, x);
            int pt = std::get<0>(config.patch_size), ph = std::get<1>(config.patch_size), pw = std::get<2>(config.patch_size);
            int64_t t_len = (T + pt / 2) / pt;
            int64_t h_len = (H + ph / 2) / ph;
            int64_t w_len = (W + pw / 2) / pw;

            // patch_embedding -> [1, L, dim]
            x = patch_embedding->forward(ctx, x);                                                    // [dim, t_len, h_len, w_len]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3], 1);       // [1, dim, L]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [1, L, dim]

            // time embedding
            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, config.freq_dim);
            e      = time_embedding_0->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = time_embedding_2->forward(ctx, e);  // [dim, N]
            auto e0 = ggml_silu(ctx->ggml_ctx, e);
            e0      = time_projection_1->forward(ctx, e0);
            e0      = ggml_reshape_4d(ctx->ggml_ctx, e0, e0->ne[0] / 6, 6, e0->ne[1], e0->ne[2]);  // [dim,6,N]

            // text + clip context
            context = text_embedding_0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context);
            context = text_embedding_2->forward(ctx, context);  // [dim, text_len, N]
            int64_t context_img_len = 0;
            if (clip_fea != nullptr) {
                auto context_img = img_emb->forward(ctx, clip_fea);                   // [dim, 257, N]
                context          = ggml_concat(ctx->ggml_ctx, context_img, context, 1);  // [dim, 257+text_len, N]
                context_img_len  = clip_fea->ne[1];                                   // 257
            }

            int64_t n_frames = t_len;
            int64_t hw       = h_len * w_len;

            sd::ggml_graph_cut::mark_graph_cut(x, "it.prelude", "x");

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<InfiniteTalkAttentionBlock>(blocks["blocks." + std::to_string(i)]);
                x = block->forward_it(ctx, x, e0, pe, context, audio, n_frames, hw, context_img_len);
                sd::ggml_graph_cut::mark_graph_cut(x, "it.blocks." + std::to_string(i), "x");
            }

            x = head->forward(ctx, x, e);  // [1, L, pt*ph*pw*out_dim]
            x = unpatchify(ctx->ggml_ctx, x, t_len, h_len, w_len);  // [out_dim, T(+pad), H(+pad), W(+pad)]

            // slice off any patch padding.
            x = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, T);
            x = ggml_ext_slice(ctx->ggml_ctx, x, 1, 0, H);
            x = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, W);
            return x;
        }
    };

    // ----------------------------------------------------------------------
    // Host-side audio window gather (multitalk_model.py:654-666). Builds the two
    // AudioProjModel inputs for one clip window from the per-video-frame wav2vec
    // stack `full` (ggml ne [audio_dim, blocks, T_total], i.e. [768,12,T]).
    //   window 5 per video frame, centered, clamped (indices -2..+2);
    //   first latent frame uses video frame audio_start_idx (5-window);
    //   each subsequent latent frame groups vae_scale(4) video frames and gathers an
    //   8-window via the first(0:3)/middle(2:3)/last(2:5) selection.
    // Returns first_in [input_dim,1,1] and vf_in [input_dim_vf, n_t, 1] with the
    // (window, block, channel) flatten order proj1/proj1_vf expect (channel fastest).
    // ----------------------------------------------------------------------
    static inline void build_audio_proj_inputs(const sd::Tensor<float>& full,
                                               int audio_start_idx, int frame_num,
                                               const InfiniteTalkConfig& cfg,
                                               sd::Tensor<float>& first_in,
                                               sd::Tensor<float>& vf_in) {
        const int C   = (int)cfg.audio_dim;       // 768
        const int Blk = (int)cfg.blocks_per_token; // 12
        const int Wn  = cfg.audio_window;          // 5
        const int Wvf = cfg.audio_window + cfg.vae_scale - 1;  // 8
        const int mid = Wn / 2;                     // 2
        const int T   = (int)full.shape()[2];       // total video frames available
        const int n_t = (frame_num - 1) / cfg.vae_scale;  // latter latent groups (e.g. 20)
        const float* fd = full.data();
        // full ne [C, Blk, T]: element (c,b,f) at f*Blk*C + b*C + c.
        auto at = [&](int f, int b, int c) -> float {
            int fc = f < 0 ? 0 : (f >= T ? T - 1 : f);
            return fd[((int64_t)fc * Blk + b) * C + c];
        };
        // window value: video frame `vf`, window offset wi in [0,Wn) -> source frame vf+(wi-mid).
        auto win = [&](int vf, int wi, int b, int c) -> float {
            return at(vf + (wi - mid), b, c);
        };

        // first latent frame: video frame `audio_start_idx`, full 5-window.
        first_in = sd::Tensor<float>({(int64_t)Wn * Blk * C, 1, 1});
        {
            float* d = first_in.data();
            int vf   = audio_start_idx;
            for (int w = 0; w < Wn; ++w)
                for (int b = 0; b < Blk; ++b)
                    for (int c = 0; c < C; ++c)
                        d[(w * Blk + b) * C + c] = win(vf, w, b, c);
        }

        // latter latent frames: each groups vae_scale video frames after the first.
        vf_in = sd::Tensor<float>({(int64_t)Wvf * Blk * C, (int64_t)n_t, 1});
        {
            float* d = vf_in.data();
            for (int g = 0; g < n_t; ++g) {
                // the 4 video frames of this group (frames 1.. of the window, 0-based after first).
                int base = audio_start_idx + 1 + g * cfg.vae_scale;
                int f0   = base;                    // first of the 4
                int f3   = base + cfg.vae_scale - 1; // last of the 4
                // gather order == concat(first[0:mid+1], middle[mid:mid+1]*中, last[mid:Wn]) = 8 windows.
                // first frame: windows [0, mid] inclusive (mid+1 = 3 windows).
                int slot = 0;
                float* col = d + (int64_t)g * Wvf * Blk * C;
                for (int w = 0; w <= mid; ++w) {  // 0,1,2
                    for (int b = 0; b < Blk; ++b)
                        for (int c = 0; c < C; ++c)
                            col[(slot * Blk + b) * C + c] = win(f0, w, b, c);
                    slot++;
                }
                // middle frames (the two interior of the group): window `mid` only.
                for (int fmid = base + 1; fmid <= base + cfg.vae_scale - 2; ++fmid) {
                    for (int b = 0; b < Blk; ++b)
                        for (int c = 0; c < C; ++c)
                            col[(slot * Blk + b) * C + c] = win(fmid, mid, b, c);
                    slot++;
                }
                // last frame: windows [mid, Wn) (mid..4 = 3 windows).
                for (int w = mid; w < Wn; ++w) {  // 2,3,4
                    for (int b = 0; b < Blk; ++b)
                        for (int c = 0; c < C; ++c)
                            col[(slot * Blk + b) * C + c] = win(f3, w, b, c);
                    slot++;
                }
                GGML_ASSERT(slot == Wvf);
            }
        }
    }

    // ----------------------------------------------------------------------
    // InfiniteTalkRunner: owns the DiT, builds the 3D RoPE table and the
    // [noisy ++ c_concat] channel concat, and exposes:
    //   compute()                -> one DiT velocity forward for a window
    //   compute_audio_embedding()-> AudioProjModel sub-graph (run once per window)
    // ----------------------------------------------------------------------
    struct InfiniteTalkRunner : public GGMLRunner {
        InfiniteTalkConfig config;
        InfiniteTalk dit;
        std::vector<float> pe_vec;
        std::string desc = "InfiniteTalk (Wan2.1-I2V-14B + MultiTalk audio graft)";

        InfiniteTalkRunner(ggml_backend_t backend,
                           ggml_backend_t params_backend,
                           const String2TensorStorage& tensor_storage_map = {},
                           const std::string prefix                       = "model.diffusion_model")
            : GGMLRunner(backend, params_backend) {
            dit = InfiniteTalk(config);
            dit.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return desc; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            dit.get_param_tensors(tensors, prefix);
        }

        std::vector<float> build_pe(int t, int h, int w) {
            return Rope::gen_wan_pe(t, h, w,
                                    std::get<0>(config.patch_size),
                                    std::get<1>(config.patch_size),
                                    std::get<2>(config.patch_size),
                                    1, config.theta, config.axes_dim);
        }

        // One DiT velocity forward.
        //   x:           [W, H, T, 16]   noisy latent
        //   c_concat:    [W, H, T, 20]   (4ch mask + 16ch VAE-encoded cond)
        //   timestep:    [1]
        //   context:     [text_dim, 512, 1]
        //   clip_fea:    [1280, 257, 1]
        //   audio_emb:   [audio_dim, 32, t_len]  per-latent-frame audio tokens (or zeros)
        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& c_concat,
                                  const sd::Tensor<float>& timestep,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& clip_fea,
                                  const sd::Tensor<float>& audio_emb) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(WAN::WAN_GRAPH_SIZE);

                ggml_tensor* gx     = make_input(x);
                ggml_tensor* gcc    = make_input(c_concat);
                ggml_tensor* gts    = make_input(timestep);
                ggml_tensor* gctx   = make_input(context);
                ggml_tensor* gclip  = make_input(clip_fea);
                ggml_tensor* gaudio = audio_emb.empty() ? nullptr : make_input(audio_emb);

                ggml_tensor* xin = ggml_concat(compute_ctx, gx, gcc, 3);  // [W,H,T,36]

                int T = (int)x.shape()[2], H = (int)x.shape()[1], W = (int)x.shape()[0];
                pe_vec      = build_pe(T, H, W);
                int pos_len = (int)(pe_vec.size() / config.axes_dim_sum / 2);
                auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
                set_backend_tensor_data(pe, pe_vec.data());

                auto rctx        = get_context();
                ggml_tensor* out = dit.forward(&rctx, xin, gts, gctx, gclip, gaudio, pe);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        // AudioProjModel sub-graph: wav2vec windows -> [audio_dim, 32, 1+n_t] audio tokens.
        sd::Tensor<float> compute_audio_embedding(int n_threads,
                                                  const sd::Tensor<float>& first_in,
                                                  const sd::Tensor<float>& vf_in) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf      = new_graph_custom(2048);
                ggml_tensor* gfirst  = make_input(first_in);
                ggml_tensor* gvf     = make_input(vf_in);
                auto rctx            = get_context();
                ggml_tensor* out     = dit.get_audio_proj()->forward(&rctx, gfirst, gvf);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };

}  // namespace IT

#endif  // __INFINITETALK_HPP__
