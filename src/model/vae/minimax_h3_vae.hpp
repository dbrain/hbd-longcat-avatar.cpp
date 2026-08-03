#ifndef __SD_MODEL_VAE_MINIMAX_H3_VAE_HPP__
#define __SD_MODEL_VAE_MINIMAX_H3_VAE_HPP__

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/vae/vae.hpp"
#include "model_loader.h"
#include "model_manager.h"

// MiniMax-H3 video VAE: 16x spatial / 4x temporal, 24 latent channels.
//
// Encoder: `EncoderFCN3D`, a causal 3D CNN (reflect spatial padding, front-only zero
// temporal padding, per-frame group norm).  Decoder: `ViT3DDecoder`, a 36-layer plain
// transformer over one token per latent voxel, with 3-axis rotary embeddings and a
// patch-unfold head -- not a CNN, and not causal.
//
// Tensor names follow the ORIGINAL (reference / ComfyUI) checkpoint spelling:
//   encoder.down.{i}.block.{j}.{norm1,conv1,norm2,conv2,nin_shortcut}
//   encoder.down.{i}.downsample.conv
//   decoder.x_embedder, decoder.transformer_blocks.{i}.{norm1,attn.to_qkv,attn.to_out,
//                                                        scale1,norm2,ff.w1,ff.w2,scale2}
// A diffusers-converted checkpoint (`down_blocks.{i}.resnets.{j}`, split `to_q/to_k/to_v`,
// `ff.net.0.proj`, swapped SwiGLU halves -- see scripts/convert_minimax_h3_to_diffusers.py)
// must be renamed back to the above by name_conversion before it reaches this file.
namespace MiniMaxH3Video {

    // comfy/ldm/minimax/vae.py IMAGENET_MEAN / IMAGENET_STD.
    static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
    static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

    // comfy/ldm/minimax/vae.py LATENTS_MEAN / LATENTS_STD (24 channels).  These live in
    // `video_vae/config.json`, NOT in the weight file -- the conversion script rejects them
    // as unexpected keys -- so they are literals here, exactly as in the reference.
    static const float LATENTS_MEAN[24] = {
        0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
        -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
        -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
        -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
        -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
        -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f};

    static const float LATENTS_STD[24] = {
        1.2223774194717407f, 1.2767263650894165f, 1.68317747116088865f, 1.7549455165863037f,
        1.5636216402053833f, 2.194143533706665f, 0.96531379222869875f, 1.05698859691619875f,
        0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
        0.7996809482574463f, 0.44988900423049925f, 0.7197399735450745f, 0.69362932443618775f,
        2.961095094680786f, 2.7694199085235595f, 3.0496184825897215f, 2.1088054180145265f,
        3.276226282119751f, 3.1627357006073f, 2.28168129920959475f, 2.6127843856811525f};

    static inline bool has_tensor(const String2TensorStorage& tensor_storage_map,
                                  const std::string& name) {
        return tensor_storage_map.find(name) != tensor_storage_map.end();
    }

    static inline int64_t get_tensor_ne(const String2TensorStorage& tensor_storage_map,
                                        const std::string& name,
                                        int index,
                                        int64_t fallback = 0) {
        auto iter = tensor_storage_map.find(name);
        if (iter == tensor_storage_map.end()) {
            return fallback;
        }
        return iter->second.ne[index];
    }

    struct MiniMaxH3VAEConfig {
        // --- encoder -------------------------------------------------------------
        int64_t in_channels  = 3;
        int64_t out_channels = 3;
        // ch * ch_mult with ch=128, ch_mult=(1,2,2,4,4,8) (MiniMaxH3VideoVAE defaults).
        std::vector<int64_t> block_out_channels = {128, 256, 256, 512, 512, 1024};
        int layers_per_block                    = 2;  // num_res_blocks
        // space_down / time_down.  Strides are not recoverable from a weight file, so these
        // stay literals from MiniMaxH3VideoVAE.__init__; only their *length* is validated
        // against the level count detected from the tensor map.
        std::vector<int> spatial_downsample_factors  = {2, 2, 2, 2, 1, 1};
        std::vector<int> temporal_downsample_factors = {1, 2, 2, 1, 1, 1};
        int norm_num_groups                          = 32;     // group_norm_3d()
        float norm_eps                               = 1e-6f;  // group_norm_3d()
        int64_t latent_channels                      = 24;     // z_channels == embed_dim

        // --- decoder -------------------------------------------------------------
        int decoder_num_layers          = 36;      // vit_decoder_kwargs.num_layers
        int64_t decoder_attention_heads = 32;      // vit_decoder_kwargs.heads
        int64_t decoder_head_dim        = 64;      // vit_decoder_kwargs.dim_head
        int decoder_num_register_tokens = 4;       // ViT3DDecoder default
        int decoder_ffn_mult            = 4;       // FeedForward default
        float decoder_rope_theta        = 100.0f;  // vit_decoder_kwargs.rope_theta
        float decoder_rope_dim_ratio    = 0.75f;   // vit_decoder_kwargs.rope_dim_ratio
        float decoder_norm_eps          = 1e-5f;   // ViT3DDecoder eps
        bool has_mask_token             = false;   // unused MAE buffer; carried only to load cleanly

        // --- wrapper -------------------------------------------------------------
        // video_vae/config.json: vae_clip_length / vae_token_drop / vae_tile_size /
        // vae_tile_overlap_min.  Not in the weight file either.
        int64_t clip_length      = 17;
        int64_t token_drop       = 3;
        int64_t tile_size        = 256;
        int64_t tile_overlap_min = 64;
        bool tiling              = true;
        bool has_encoder         = false;

        int64_t num_levels() const { return static_cast<int64_t>(block_out_channels.size()); }

        int64_t decoder_dim() const { return decoder_attention_heads * decoder_head_dim; }

        // int(math.prod(space_down)) / int(math.prod(time_down))
        int64_t vae_ratio() const {
            int64_t r = 1;
            for (int f : spatial_downsample_factors) {
                r *= f;
            }
            return r;
        }

        int64_t vae_ratio_t() const {
            int64_t r = 1;
            for (int f : temporal_downsample_factors) {
                r *= f;
            }
            return r;
        }

        // MiniMaxH3VideoVAE.__init__ temporal-chunking derivations.
        int64_t frame_pre_padding() const {
            const int64_t m = vae_ratio_t();
            return ((m - clip_length % m) % m);  // (-clip_length) % vae_ratio_t
        }

        int64_t tokens_chunk_size() const {
            const int64_t m = vae_ratio_t();
            return (clip_length + m - 1) / m;  // ceil(clip_length / vae_ratio_t)
        }

        int64_t token_overlap() const {
            const int64_t c = tokens_chunk_size();
            return ((c - token_drop % c) % c);  // (-token_drop) % tokens_chunk_size
        }

        int64_t frame_overlap() const {
            return std::max<int64_t>(token_overlap() * vae_ratio_t() - frame_pre_padding(), 0);
        }

        int64_t rope_dim() const {
            return static_cast<int64_t>(decoder_head_dim * decoder_rope_dim_ratio);
        }

        static MiniMaxH3VAEConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                      const std::string& prefix = "") {
            MiniMaxH3VAEConfig config;

            const std::string p = prefix.empty() ? std::string() : prefix + ".";

            // z_channels / embed_dim.  post_quant_conv is Conv3d(embed_dim, z_channels, 1),
            // so its bias width is z_channels and quant_conv's bias width is 2*embed_dim.
            const int64_t post_quant_out = get_tensor_ne(tensor_storage_map, p + "post_quant_conv.bias", 0, 0);
            if (post_quant_out > 0) {
                config.latent_channels = post_quant_out;
            }

            config.has_encoder = has_tensor(tensor_storage_map, p + "encoder.conv_in.weight");

            if (config.has_encoder) {
                // in_channels is deliberately NOT read off encoder.conv_in.weight: a 5-D conv
                // weight can reach the tensor map either as [kW, kH, kT, IC, OC] or already
                // folded to [kW, kH, kT, IC*OC], and the two give different answers with no way
                // to tell them apart.  MiniMaxH3VideoVAE's in_channels is 3 and stays 3.
                std::vector<int64_t> levels;
                for (int level = 0;; ++level) {
                    const std::string bias = p + "encoder.down." + std::to_string(level) + ".block.0.conv1.bias";
                    if (!has_tensor(tensor_storage_map, bias)) {
                        break;
                    }
                    levels.push_back(get_tensor_ne(tensor_storage_map, bias, 0, 0));
                }
                if (!levels.empty()) {
                    config.block_out_channels = levels;
                }

                int blocks_per_level = 0;
                while (has_tensor(tensor_storage_map,
                                  p + "encoder.down.0.block." + std::to_string(blocks_per_level) + ".conv1.bias")) {
                    ++blocks_per_level;
                }
                if (blocks_per_level > 0) {
                    config.layers_per_block = blocks_per_level;
                }
            }

            // The per-level strides are not in the weights.  Keep the reference literals when the
            // level count matches; otherwise fall back to "every level that ships a downsample conv
            // halves space", which is the only stride story a weight file can support, and say so.
            const int64_t detected_levels = config.num_levels();
            if (static_cast<int64_t>(config.spatial_downsample_factors.size()) != detected_levels) {
                LOG_WARN(
                    "minimax_h3_vae: %lld encoder levels detected but the reference stride table has %zu; "
                    "guessing strides from the presence of a downsample conv",
                    (long long)detected_levels,
                    config.spatial_downsample_factors.size());
                config.spatial_downsample_factors.assign(static_cast<size_t>(detected_levels), 1);
                config.temporal_downsample_factors.assign(static_cast<size_t>(detected_levels), 1);
                for (int64_t level = 0; level < detected_levels; ++level) {
                    if (has_tensor(tensor_storage_map,
                                   p + "encoder.down." + std::to_string(level) + ".downsample.conv.weight")) {
                        config.spatial_downsample_factors[static_cast<size_t>(level)] = 2;
                    }
                }
            }

            // Decoder.
            const int64_t detected_dim = get_tensor_ne(tensor_storage_map, p + "decoder.x_embedder.bias", 0, 0);
            if (detected_dim > 0) {
                // dim_head is a config knob with no weight footprint (the qkv projection is fused);
                // keep the released 64 and derive the head count from it.
                config.decoder_attention_heads = detected_dim / config.decoder_head_dim;
            }

            int num_layers = 0;
            while (has_tensor(tensor_storage_map,
                              p + "decoder.transformer_blocks." + std::to_string(num_layers) + ".scale1")) {
                ++num_layers;
            }
            if (num_layers > 0) {
                config.decoder_num_layers = num_layers;
            }

            // register_tokens: torch [1, num_register_tokens, dim] -> ne [dim, num_register_tokens, 1].
            const int64_t num_register = get_tensor_ne(tensor_storage_map, p + "decoder.register_tokens", 1, 0);
            if (num_register > 0) {
                config.decoder_num_register_tokens = static_cast<int>(num_register);
            }

            // ff.w2 is Linear(inner_dim, dim), i.e. ne0 == inner_dim == dim * mult.
            const int64_t ff_inner = get_tensor_ne(tensor_storage_map, p + "decoder.transformer_blocks.0.ff.w2.weight", 0, 0);
            if (ff_inner > 0 && config.decoder_dim() > 0 && ff_inner % config.decoder_dim() == 0) {
                config.decoder_ffn_mult = static_cast<int>(ff_inner / config.decoder_dim());
            }

            // proj_out is Linear(dim, out_channels * patch_size_t * patch_size^2), and the patch
            // sizes are exactly the compression ratios, so out_channels falls out of the width.
            const int64_t proj_out     = get_tensor_ne(tensor_storage_map, p + "decoder.proj_out.bias", 0, 0);
            const int64_t patch_volume = config.vae_ratio() * config.vae_ratio() * config.vae_ratio_t();
            if (proj_out > 0 && patch_volume > 0 && proj_out % patch_volume == 0) {
                config.out_channels = proj_out / patch_volume;
            }

            config.has_mask_token = has_tensor(tensor_storage_map, p + "decoder.mask_token");

            LOG_DEBUG(
                "minimax_h3_vae: levels=%lld layers_per_block=%d latent_channels=%lld "
                "decoder=%dx%lldx%lld register=%d ffn_mult=%d ratio=%lld/%lld encoder=%s",
                (long long)config.num_levels(),
                config.layers_per_block,
                (long long)config.latent_channels,
                config.decoder_num_layers,
                (long long)config.decoder_attention_heads,
                (long long)config.decoder_head_dim,
                config.decoder_num_register_tokens,
                config.decoder_ffn_mult,
                (long long)config.vae_ratio(),
                (long long)config.vae_ratio_t(),
                config.has_encoder ? "yes" : "no");
            return config;
        }
    };

    // ---------------------------------------------------------------------------
    // graph helpers
    // ---------------------------------------------------------------------------

    // ggml has no reflect-pad op for the W/H axes, so torch's F.pad(..., mode="reflect")
    // is expressed as explicit mirrored single-column/row slices plus a concat.  Exact,
    // not an approximation; the cost is 2*pad extra copies per padded conv.
    static inline ggml_tensor* reflect_pad_1d(ggml_context* ctx,
                                              ggml_tensor* x,
                                              int dim,
                                              int left,
                                              int right) {
        if (left == 0 && right == 0) {
            return x;
        }
        const int64_t n = x->ne[dim];
        GGML_ASSERT(n > 1 && "reflect padding needs at least 2 elements along the padded axis");
        GGML_ASSERT(left < n && right < n && "reflect padding wider than the tensor is undefined");

        ggml_tensor* out = x;
        if (left > 0) {
            // torch reflect: out[-i] == x[i], so the prepended block is x[left], ..., x[1].
            ggml_tensor* pad = nullptr;
            for (int i = left; i >= 1; --i) {
                auto column = ggml_ext_slice(ctx, x, dim, i, i + 1);
                pad         = (pad == nullptr) ? column : ggml_concat(ctx, pad, column, dim);
            }
            out = ggml_concat(ctx, pad, out, dim);
        }
        if (right > 0) {
            // out[n-1+i] == x[n-1-i], so the appended block is x[n-2], ..., x[n-1-right].
            ggml_tensor* pad = nullptr;
            for (int i = 1; i <= right; ++i) {
                auto column = ggml_ext_slice(ctx, x, dim, n - 1 - i, n - i);
                pad         = (pad == nullptr) ? column : ggml_concat(ctx, pad, column, dim);
            }
            out = ggml_concat(ctx, out, pad, dim);
        }
        return out;
    }

    // x: [W, H, T, C] (batch folded into C, B == 1 throughout this port)
    static inline ggml_tensor* reflect_pad_spatial(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   int left_w,
                                                   int right_w,
                                                   int left_h,
                                                   int right_h) {
        x = reflect_pad_1d(ctx, x, 0, left_w, right_w);
        x = reflect_pad_1d(ctx, x, 1, left_h, right_h);
        return x;
    }

    // CausalConv3d: reflect padding on W/H, front-only zero padding on T.
    //
    // The reference has a single-frame fast path (`autopad="causal_zero"`) that truncates the
    // temporal taps instead of convolving the all-zero front pad.  That is a *numerical
    // identity* with prepending kernel_t-1 zero frames (the truncated taps multiply zeros), so
    // this port always prepends and never special-cases T == 1.  The only cost is arithmetic on
    // known-zero frames in the image path.
    class CausalConv3d : public Conv3d {
    protected:
        int spatial_padding  = 0;
        int temporal_padding = 0;

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     int kernel_size,
                     std::tuple<int, int, int> stride = {1, 1, 1},
                     int padding                      = 0,
                     bool spatial_pad                 = true,
                     bool bias                        = true)
            : Conv3d(in_channels,
                     out_channels,
                     {kernel_size, kernel_size, kernel_size},
                     stride,
                     {0, 0, 0},
                     {1, 1, 1},
                     bias) {
            // `padding` mirrors the reference's scalar/tuple `causal_padding`: the temporal entry
            // is doubled on the front (CausalConv3d.forward pads causal_padding[0] * 2), and the
            // spatial entries are symmetric.  Downsample3D passes (1, 0, 0) => temporal only.
            temporal_padding = padding * 2;
            spatial_padding  = spatial_pad ? padding : 0;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            if (spatial_padding == 0 && temporal_padding == 0) {
                return Conv3d::forward(ctx, x);
            }
            if (spatial_padding > 0) {
                x = reflect_pad_spatial(ctx->ggml_ctx, x, spatial_padding, spatial_padding, spatial_padding, spatial_padding);
            }
            if (temporal_padding > 0) {
                x = ggml_ext_pad_ext(ctx->ggml_ctx, ctx->backend, x, 0, 0, 0, 0, temporal_padding, 0, 0, 0);
            }
            return Conv3d::forward(ctx, x);
        }
    };

    // TemporalIsolatedGroupNorm: GroupNorm(32, eps=1e-6) whose statistics are computed per
    // frame, i.e. the reference folds T into the batch axis before normalizing.
    //
    // ggml_group_norm groups along ne2 and batches along ne3, so the fold is a permute:
    // [W, H, T, C] -> [W, H, C, T] normalizes each (t) independently over its 32 channel
    // groups, which is exactly the reference.  Doing it without the permute would mix
    // statistics across frames -- silent corruption, no error.
    class TemporalIsolatedGroupNorm : public GroupNorm {
    public:
        TemporalIsolatedGroupNorm(int64_t num_channels, int groups = 32, float eps = 1e-6f)
            : GroupNorm(groups, num_channels, eps, true) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = params["bias"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
            auto h = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // [W,H,C,T]
            h      = ggml_ext_group_norm(ctx->ggml_ctx, h, w, b, num_groups);
            h      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 0, 1, 3, 2));  // [W,H,T,C]
            return h;
        }
    };

    struct Downsample3D : public GGMLBlock {
        int space_stride;

        Downsample3D(int64_t in_channels,
                     int64_t out_channels,
                     int time_stride  = 1,
                     int space_stride = 2)
            : space_stride(space_stride) {
            // padding=(1,0,0): temporal-only causal padding; the spatial pad is the asymmetric
            // bottom/right reflect pad applied in forward().
            blocks["conv"] = std::make_shared<CausalConv3d>(in_channels,
                                                            out_channels,
                                                            3,
                                                            std::tuple<int, int, int>{time_stride, space_stride, space_stride},
                                                            1,
                                                            false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);
            if (space_stride == 2) {
                // F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect") -- right/bottom only.
                x = reflect_pad_spatial(ctx->ggml_ctx, x, 0, 1, 0, 1);
            }
            return conv->forward(ctx, x);
        }
    };

    struct ResnetBlock3D : public GGMLBlock {
        int64_t in_channels;
        int64_t out_channels;

        ResnetBlock3D(int64_t in_channels, int64_t out_channels, int groups = 32, float eps = 1e-6f)
            : in_channels(in_channels), out_channels(out_channels) {
            blocks["norm1"] = std::make_shared<TemporalIsolatedGroupNorm>(in_channels, groups, eps);
            blocks["norm2"] = std::make_shared<TemporalIsolatedGroupNorm>(out_channels, groups, eps);
            blocks["conv1"] = std::make_shared<CausalConv3d>(in_channels, out_channels, 3,
                                                             std::tuple<int, int, int>{1, 1, 1}, 1);
            blocks["conv2"] = std::make_shared<CausalConv3d>(out_channels, out_channels, 3,
                                                             std::tuple<int, int, int>{1, 1, 1}, 1);
            if (in_channels != out_channels) {
                blocks["nin_shortcut"] = std::make_shared<CausalConv3d>(in_channels, out_channels, 1,
                                                                        std::tuple<int, int, int>{1, 1, 1}, 0);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<TemporalIsolatedGroupNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<TemporalIsolatedGroupNorm>(blocks["norm2"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto h = norm1->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = conv1->forward(ctx, h);

            h = norm2->forward(ctx, h);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv2->forward(ctx, h);

            ggml_tensor* residual = x;
            if (in_channels != out_channels) {
                auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["nin_shortcut"]);
                residual      = shortcut->forward(ctx, x);
            }
            return ggml_add(ctx->ggml_ctx, h, residual);
        }
    };

    // encoder.down.{i}: `block.{j}` resnets plus an optional `downsample`.
    struct DownBlock3D : public GGMLBlock {
        int num_res_blocks;
        bool has_downsample;

        DownBlock3D(int64_t in_channels,
                    int64_t out_channels,
                    int num_res_blocks,
                    int time_stride,
                    int space_stride,
                    int groups,
                    float eps)
            : num_res_blocks(num_res_blocks),
              has_downsample(time_stride * space_stride > 1) {
            for (int i = 0; i < num_res_blocks; ++i) {
                blocks["block." + std::to_string(i)] = std::make_shared<ResnetBlock3D>(i == 0 ? in_channels : out_channels,
                                                                                       out_channels,
                                                                                       groups,
                                                                                       eps);
            }
            if (has_downsample) {
                blocks["downsample"] = std::make_shared<Downsample3D>(out_channels, out_channels, time_stride, space_stride);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            for (int i = 0; i < num_res_blocks; ++i) {
                auto block = std::dynamic_pointer_cast<ResnetBlock3D>(blocks["block." + std::to_string(i)]);
                x          = block->forward(ctx, x);
            }
            if (has_downsample) {
                auto downsample = std::dynamic_pointer_cast<Downsample3D>(blocks["downsample"]);
                x               = downsample->forward(ctx, x);
            }
            return x;
        }
    };

    struct EncoderFCN3D : public GGMLBlock {
        int64_t num_levels;

        explicit EncoderFCN3D(const MiniMaxH3VAEConfig& config) {
            num_levels = config.num_levels();

            // block_in = [block_mid[0]] + block_mid[:-1]; block_out == block_mid.
            std::vector<int64_t> block_in(static_cast<size_t>(num_levels));
            for (int64_t i = 0; i < num_levels; ++i) {
                block_in[static_cast<size_t>(i)] = (i == 0) ? config.block_out_channels[0]
                                                            : config.block_out_channels[static_cast<size_t>(i - 1)];
            }

            blocks["conv_in"] = std::make_shared<CausalConv3d>(config.in_channels,
                                                               block_in[0],
                                                               3,
                                                               std::tuple<int, int, int>{1, 1, 1},
                                                               1);

            for (int64_t level = 0; level < num_levels; ++level) {
                blocks["down." + std::to_string(level)] =
                    std::make_shared<DownBlock3D>(block_in[static_cast<size_t>(level)],
                                                  config.block_out_channels[static_cast<size_t>(level)],
                                                  config.layers_per_block,
                                                  config.temporal_downsample_factors[static_cast<size_t>(level)],
                                                  config.spatial_downsample_factors[static_cast<size_t>(level)],
                                                  config.norm_num_groups,
                                                  config.norm_eps);
            }

            const int64_t last = config.block_out_channels.back();
            blocks["norm_out"] = std::make_shared<TemporalIsolatedGroupNorm>(last, config.norm_num_groups, config.norm_eps);
            // double_z=True in MiniMaxH3VideoVAE.__init__.
            blocks["conv_out"] = std::make_shared<CausalConv3d>(last,
                                                                2 * config.latent_channels,
                                                                3,
                                                                std::tuple<int, int, int>{1, 1, 1},
                                                                1);
        }

        // x: [W, H, T, in_channels] -> [W/16, H/16, T_lat, 2*z_channels]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv_in  = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto norm_out = std::dynamic_pointer_cast<TemporalIsolatedGroupNorm>(blocks["norm_out"]);
            auto conv_out = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);

            auto h = conv_in->forward(ctx, x);
            for (int64_t level = 0; level < num_levels; ++level) {
                auto down = std::dynamic_pointer_cast<DownBlock3D>(blocks["down." + std::to_string(level)]);
                h         = down->forward(ctx, h);
            }
            h = norm_out->forward(ctx, h);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            return conv_out->forward(ctx, h);
        }
    };

    // ---------------------------------------------------------------------------
    // ViT3D decoder
    // ---------------------------------------------------------------------------

    // Host-side RotaryEmbeddingND + create_token_ids.
    //
    // create_token_ids: per axis, arange(0.5, n)/n * 2 - 1, meshgrid'd in (t, h, w) order with
    // w fastest.  RotaryEmbeddingND: inv_freq = base ** -arange(0, 1, 2*n_dim/dim), angles =
    // 2*pi * coord * inv_freq, flattened as (axis, freq).  The reference then builds an explicit
    // 2x2 rotation table and calls a split-half rope over the first 2*len(angles) head channels
    // -- the diffusers port spells the same thing as cos/sin tiled twice, which is what we emit.
    //
    // Register/suffix tokens get position 0 in every axis, so their angles are 0 and the rotation
    // is the identity; they are included so the table lines up with the token sequence.
    static inline void build_rope_tables(int64_t latent_t,
                                         int64_t latent_h,
                                         int64_t latent_w,
                                         int64_t num_suffix,
                                         int64_t rope_dim,
                                         float theta,
                                         sd::Tensor<float>* cos_out,
                                         sd::Tensor<float>* sin_out) {
        const int64_t n_axes = 3;
        GGML_ASSERT(rope_dim % (2 * n_axes) == 0 && "rope dim must be divisible by 2 * n_dim");
        const int64_t n_freq  = rope_dim / (2 * n_axes);
        const int64_t n_angle = n_axes * n_freq;  // == rope_dim / 2
        const int64_t patches = latent_t * latent_h * latent_w;
        const int64_t seq_len = patches + num_suffix;

        std::vector<float> inv_freq(static_cast<size_t>(n_freq));
        for (int64_t j = 0; j < n_freq; ++j) {
            const float exponent             = static_cast<float>(j) * static_cast<float>(2 * n_axes) / static_cast<float>(rope_dim);
            inv_freq[static_cast<size_t>(j)] = std::pow(theta, -exponent);
        }

        *cos_out = sd::Tensor<float>({n_angle, 1, seq_len});
        *sin_out = sd::Tensor<float>({n_angle, 1, seq_len});

        auto axis_coord = [](int64_t index, int64_t size) {
            return 2.0f * ((static_cast<float>(index) + 0.5f) / static_cast<float>(size)) - 1.0f;
        };

        for (int64_t s = 0; s < seq_len; ++s) {
            float coords[3] = {0.0f, 0.0f, 0.0f};
            if (s < patches) {
                const int64_t t = s / (latent_h * latent_w);
                const int64_t h = (s / latent_w) % latent_h;
                const int64_t w = s % latent_w;
                coords[0]       = axis_coord(t, latent_t);
                coords[1]       = axis_coord(h, latent_h);
                coords[2]       = axis_coord(w, latent_w);
            }
            for (int64_t a = 0; a < n_axes; ++a) {
                for (int64_t j = 0; j < n_freq; ++j) {
                    // angle_scale = 2 * pi (RotaryEmbeddingND.__init__)
                    const float angle = 6.283185307179586f * coords[a] * inv_freq[static_cast<size_t>(j)];
                    const int64_t idx = s * n_angle + a * n_freq + j;
                    (*cos_out)[idx]   = std::cos(angle);
                    (*sin_out)[idx]   = std::sin(angle);
                }
            }
        }
    }

    // x: [d_head, heads, seq, 1]; cos/sin: [rot_dim/2, 1, seq, 1].
    // Split-half (NeoX style): the first rot_dim channels rotate as
    //   [first, second] -> [first*cos - second*sin, second*cos + first*sin]
    // and the remaining head channels pass through untouched.
    static inline ggml_tensor* apply_rope_split_half(ggml_context* ctx,
                                                     ggml_tensor* x,
                                                     ggml_tensor* rope_cos,
                                                     ggml_tensor* rope_sin,
                                                     int64_t rot_dim) {
        if (rope_cos == nullptr || rope_sin == nullptr || rot_dim <= 0) {
            return x;
        }
        const int64_t d_head = x->ne[0];
        GGML_ASSERT(rot_dim <= d_head);
        const int64_t half = rot_dim / 2;

        ggml_tensor* pass = (rot_dim < d_head) ? ggml_ext_slice(ctx, x, 0, rot_dim, d_head) : nullptr;
        auto first        = ggml_ext_slice(ctx, x, 0, 0, half);
        auto second       = ggml_ext_slice(ctx, x, 0, half, rot_dim);

        auto out_first  = ggml_sub(ctx, ggml_mul(ctx, first, rope_cos), ggml_mul(ctx, second, rope_sin));
        auto out_second = ggml_add(ctx, ggml_mul(ctx, second, rope_cos), ggml_mul(ctx, first, rope_sin));

        auto out = ggml_concat(ctx, out_first, out_second, 0);
        if (pass != nullptr) {
            out = ggml_concat(ctx, out, pass, 0);
        }
        return out;
    }

    struct FeedForward : public GGMLBlock {
        FeedForward(int64_t dim, int mult = 4, bool bias = true) {
            const int64_t inner_dim = dim * mult;
            blocks["w1"]            = std::make_shared<Linear>(dim, inner_dim * 2, bias);
            blocks["w2"]            = std::make_shared<Linear>(inner_dim, dim, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);

            auto h = w1->forward(ctx, x);
            // gate, x = w1(x).chunk(2, dim=-1); w2(silu(gate) * x) -- gate is the FIRST half.
            // (The diffusers conversion swaps these two halves because its SwiGLU reads
            // [up; gate]; this port keeps the original order, so do NOT feed it converted w1.)
            h = ggml_ext_silu_act(ctx->ggml_ctx, h, true);
            return w2->forward(ctx, h);
        }
    };

    struct Attention : public GGMLBlock {
        int64_t heads;
        int64_t dim_head;
        int64_t rot_dim;
        float eps;

        Attention(int64_t heads, int64_t dim_head, int64_t rot_dim, bool bias = true, float eps = 1e-5f)
            : heads(heads), dim_head(dim_head), rot_dim(rot_dim), eps(eps) {
            const int64_t inner_dim = heads * dim_head;
            blocks["to_qkv"]        = std::make_shared<Linear>(inner_dim, inner_dim * 3, bias);
            blocks["to_out"]        = std::make_shared<Linear>(inner_dim, inner_dim, bias);
        }

        // x: [dim, seq, 1]
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* rope_cos,
                             ggml_tensor* rope_sin) {
            auto to_qkv = std::dynamic_pointer_cast<Linear>(blocks["to_qkv"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);

            const int64_t seq = x->ne[1];
            const int64_t n   = x->ne[2];

            auto qkv = to_qkv->forward(ctx, x);  // [3*heads*dim_head, seq, n]
            // The fused projection is PER-HEAD interleaved -- [head0 q|k|v, head1 q|k|v, ...] --
            // which is exactly `qkv.view(B, S, -1, 3*dim_head).chunk(3, dim=-1)` in the reference.
            // Reshaping to [3*dim_head, heads, seq, n] and slicing ne0 reproduces it.
            qkv = ggml_reshape_4d(ctx->ggml_ctx, qkv, 3 * dim_head, heads, seq, n);

            auto q = ggml_ext_slice(ctx->ggml_ctx, qkv, 0, 0, dim_head);
            auto k = ggml_ext_slice(ctx->ggml_ctx, qkv, 0, dim_head, 2 * dim_head);
            auto v = ggml_ext_slice(ctx->ggml_ctx, qkv, 0, 2 * dim_head, 3 * dim_head);

            // norm_q / norm_k are RMSNorm(dim_head, elementwise_affine=False): no weight in the
            // checkpoint, so this is a bare rms_norm rather than the RMSNorm block.
            q = ggml_rms_norm(ctx->ggml_ctx, q, eps);
            k = ggml_rms_norm(ctx->ggml_ctx, k, eps);

            q = apply_rope_split_half(ctx->ggml_ctx, q, rope_cos, rope_sin, rot_dim);
            k = apply_rope_split_half(ctx->ggml_ctx, k, rope_cos, rope_sin, rot_dim);

            // [d_head, heads, seq, n] is contiguous head-major, i.e. the [C, seq, n] layout
            // ggml_ext_attention_ext re-splits internally.
            q = ggml_reshape_3d(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, q), dim_head * heads, seq, n);
            k = ggml_reshape_3d(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, k), dim_head * heads, seq, n);
            v = ggml_reshape_3d(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, v), dim_head * heads, seq, n);

            // The reference follows attention with `.nan_to_num_(0.0)`.  There is no mask and no
            // empty row here, so softmax cannot produce a NaN; the scrub is not ported (ggml has
            // no cheap nan_to_num and adding one would cost a full-size pass per layer).
            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, heads,
                                              nullptr, false, ctx->flash_attn_enabled);
            return to_out->forward(ctx, out);
        }
    };

    struct TransformerBlock : public GGMLBlock {
        int64_t dim;

    protected:
        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale1"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["scale2"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        }

    public:
        TransformerBlock(int64_t heads, int64_t dim_head, int64_t rot_dim, int ffn_mult, bool bias = true, float eps = 1e-5f)
            : dim(heads * dim_head) {
            blocks["norm1"] = std::make_shared<RMSNorm>(dim, eps);
            blocks["attn"]  = std::make_shared<Attention>(heads, dim_head, rot_dim, bias, eps);
            blocks["norm2"] = std::make_shared<RMSNorm>(dim, eps);
            blocks["ff"]    = std::make_shared<FeedForward>(dim, ffn_mult, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* rope_cos,
                             ggml_tensor* rope_sin) {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            auto h = attn->forward(ctx, norm1->forward(ctx, x), rope_cos, rope_sin);
            x      = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, h, params["scale1"]));

            h = ff->forward(ctx, norm2->forward(ctx, x));
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, h, params["scale2"]));
            return x;
        }
    };

    struct ViT3DDecoder : public GGMLBlock {
        int64_t patch_size;
        int64_t patch_size_t;
        int64_t out_channels;
        int64_t in_channels;
        int64_t num_layers;
        int64_t num_register_tokens;
        int64_t dim;
        bool has_mask_token;

    protected:
        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["register_tokens"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, num_register_tokens);
            if (has_mask_token) {
                // Masked-autoencoding buffer.  Never read at inference; registered only so a
                // checkpoint that still ships the key loads without a leftover tensor.
                params["mask_token"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
            }
        }

    public:
        explicit ViT3DDecoder(const MiniMaxH3VAEConfig& config)
            : patch_size(config.vae_ratio()),
              patch_size_t(config.vae_ratio_t()),
              out_channels(config.out_channels),
              in_channels(config.latent_channels),
              num_layers(config.decoder_num_layers),
              num_register_tokens(config.decoder_num_register_tokens),
              dim(config.decoder_dim()),
              has_mask_token(config.has_mask_token) {
            blocks["x_embedder"] = std::make_shared<Linear>(in_channels, dim, true);
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] =
                    std::make_shared<TransformerBlock>(config.decoder_attention_heads,
                                                       config.decoder_head_dim,
                                                       config.rope_dim(),
                                                       config.decoder_ffn_mult,
                                                       true,
                                                       config.decoder_norm_eps);
            }
            blocks["norm_out"] = std::make_shared<LayerNorm>(dim, config.decoder_norm_eps, true, true);
            blocks["proj_out"] = std::make_shared<Linear>(dim, out_channels * patch_size_t * patch_size * patch_size, true);
        }

        int64_t num_suffix_tokens() const { return num_register_tokens + 1; }

        // x: [W_lat, H_lat, T_lat, in_channels] -> [W_lat*16, H_lat*16, T_lat*4, out_channels]
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* rope_cos,
                             ggml_tensor* rope_sin) {
            auto ggml_ctx = ctx->ggml_ctx;

            const int64_t lw = x->ne[0];
            const int64_t lh = x->ne[1];
            const int64_t lt = x->ne[2];

            // x.flatten(2).transpose(1, 2): tokens ordered (t, h, w) with w fastest.
            auto h = ggml_ext_cont(ggml_ctx, ggml_ext_torch_permute(ggml_ctx, x, 3, 0, 1, 2));  // [C, W, H, T]
            h      = ggml_reshape_3d(ggml_ctx, h, in_channels, lw * lh * lt, 1);

            auto x_embedder = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            h               = x_embedder->forward(ctx, h);  // [dim, patches, 1]

            const int64_t num_patches = h->ne[1];

            // [tokens; register_tokens; one all-zero token]
            auto registers = ggml_reshape_3d(ggml_ctx, params["register_tokens"], dim, num_register_tokens, 1);
            h              = ggml_concat(ggml_ctx, h, registers, 1);
            h              = ggml_concat(ggml_ctx, h, ggml_ext_zeros(ggml_ctx, dim, 1, 1, 1), 1);

            for (int64_t i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                h          = block->forward(ctx, h, rope_cos, rope_sin);
            }

            auto norm_out = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            h             = proj_out->forward(ctx, norm_out->forward(ctx, h));
            h             = ggml_ext_slice(ggml_ctx, h, 1, 0, num_patches);

            return unpatchify(ggml_ctx, h, lw, lh, lt);
        }

        // h: [out_channels*pt*ph*pw, T*H*W, 1] -> [W*pw, H*ph, T*pt, out_channels]
        //
        // The reference does
        //   view(B, T, H, W, C, pt, ph, pw).permute(0,4,1,5,2,6,3,7).reshape(B, C, T*pt, H*ph, W*pw)
        // i.e. a 7-axis shuffle.  ggml caps at 4 axes, so it is done one axis at a time: pair pw
        // with W, then ph with H, then pt with T, leaving C outermost.  Each pass is a
        // reshape + permute + cont, and the intermediate orderings are noted per step.
        ggml_tensor* unpatchify(ggml_context* ctx,
                                ggml_tensor* h,
                                int64_t lw,
                                int64_t lh,
                                int64_t lt) const {
            const int64_t pw = patch_size;
            const int64_t ph = patch_size;
            const int64_t pt = patch_size_t;
            const int64_t c  = out_channels;

            // memory order (fastest first): pw, ph, pt, C, W, H, T
            auto x = ggml_reshape_4d(ctx, h, pw, ph * pt * c, lw, lh * lt);
            x      = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // pw, W, (ph pt C), (H T)
            x      = ggml_reshape_4d(ctx, x, pw * lw * ph, pt * c, lh, lt);           // (pw W ph), (pt C), H, T
            x      = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // (pw W ph), H, (pt C), T
            x      = ggml_reshape_4d(ctx, x, pw * lw * ph * lh, pt, c, lt);           // (pw W ph H), pt, C, T
            x      = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 1, 3, 2));  // (pw W ph H), pt, T, C
            return ggml_reshape_4d(ctx, x, lw * pw, lh * ph, lt * pt, c);
        }
    };

    // ---------------------------------------------------------------------------
    // wrapper block: encoder + quant convs + decoder
    // ---------------------------------------------------------------------------

    struct MiniMaxH3VideoVAE : public GGMLBlock {
        MiniMaxH3VAEConfig config;

        explicit MiniMaxH3VideoVAE(const MiniMaxH3VAEConfig& config)
            : config(config) {
            if (config.has_encoder) {
                blocks["encoder"]    = std::make_shared<EncoderFCN3D>(config);
                blocks["quant_conv"] = std::make_shared<Conv3d>(config.latent_channels * 2,
                                                                2 * config.latent_channels,
                                                                std::tuple<int, int, int>{1, 1, 1});
            }
            blocks["post_quant_conv"] = std::make_shared<Conv3d>(config.latent_channels,
                                                                 config.latent_channels,
                                                                 std::tuple<int, int, int>{1, 1, 1});
            blocks["decoder"]         = std::make_shared<ViT3DDecoder>(config);
        }

        // _encode_moments: quant_conv(encoder(x)).  x: [W, H, T, 3]
        //
        // Looked up with find(): operator[] on a decode-only checkpoint would INSERT a null
        // block, which init_blocks/get_param_tensors then dereference.
        ggml_tensor* encode_moments(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto encoder_it    = blocks.find("encoder");
            auto quant_conv_it = blocks.find("quant_conv");
            GGML_ASSERT(encoder_it != blocks.end() && quant_conv_it != blocks.end());
            auto encoder    = std::dynamic_pointer_cast<EncoderFCN3D>(encoder_it->second);
            auto quant_conv = std::dynamic_pointer_cast<Conv3d>(quant_conv_it->second);
            GGML_ASSERT(encoder != nullptr && quant_conv != nullptr);
            return quant_conv->forward(ctx, encoder->forward(ctx, x));
        }

        // _decode_pixels: decoder(post_quant_conv(z)).  z: [W_lat, H_lat, T_lat, 24]
        ggml_tensor* decode_pixels(GGMLRunnerContext* ctx,
                                   ggml_tensor* z,
                                   ggml_tensor* rope_cos,
                                   ggml_tensor* rope_sin) {
            auto post_quant_conv = std::dynamic_pointer_cast<Conv3d>(blocks["post_quant_conv"]);
            auto decoder         = std::dynamic_pointer_cast<ViT3DDecoder>(blocks["decoder"]);
            return decoder->forward(ctx, post_quant_conv->forward(ctx, z), rope_cos, rope_sin);
        }

        int64_t num_suffix_tokens() const {
            return config.decoder_num_register_tokens + 1;
        }
    };

}  // namespace MiniMaxH3Video

// -------------------------------------------------------------------------------
// runner
// -------------------------------------------------------------------------------
//
// The reference's spatial tiling (tile_size 256px, min overlap 64px) and temporal chunking
// (clip_length 17, token_drop 3) are always on at production resolutions, and both are driven
// HOST-SIDE here: one ggml graph per (temporal chunk x spatial tile), with `split_tiles`,
// `blend` and the stitch/canvas bookkeeping ported to sd::Tensor.  Doing it in-graph would put
// 36 transformer layers per tile into a single graph -- 25 tiles at 1024^2 -- and lose the
// per-tile activation ceiling that makes the scheme worth having.
//
// The affine normalizations the reference applies OUTSIDE the conv/transformer stacks (ImageNet
// pixel mean/std, per-channel latents_mean/latents_std) are likewise applied host-side in
// _compute; everything inside the encoder/decoder is in-graph.
struct MiniMaxH3VideoVAERunner : public VAE {
    MiniMaxH3Video::MiniMaxH3VAEConfig config;
    MiniMaxH3Video::MiniMaxH3VideoVAE vae;
    bool decode_only;
    // Defaulted from the reference literals; override with the real numbers when a caller
    // has read them out of `video_vae/config.json` (they are NOT in the weight file).
    std::vector<float> latents_mean_host;
    std::vector<float> latents_std_host;

    MiniMaxH3VideoVAERunner(ggml_backend_t backend,
                            const String2TensorStorage& tensor_storage_map      = {},
                            const std::string& prefix                           = "",
                            bool decode_only                                    = true,
                            SDVersion version                                   = VERSION_COUNT,
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : VAE(version, backend, prefix, weight_manager),
          config(MiniMaxH3Video::MiniMaxH3VAEConfig::detect_from_weights(tensor_storage_map, prefix)),
          vae(config),
          decode_only(decode_only || !config.has_encoder) {
        vae.init(params_ctx, tensor_storage_map, prefix);

        const size_t channels = static_cast<size_t>(config.latent_channels);
        latents_mean_host.assign(MiniMaxH3Video::LATENTS_MEAN, MiniMaxH3Video::LATENTS_MEAN + 24);
        latents_std_host.assign(MiniMaxH3Video::LATENTS_STD, MiniMaxH3Video::LATENTS_STD + 24);
        // A checkpoint with a different latent width than the released 24 has no literals to
        // fall back on; neutral statistics are wrong but at least deterministic and loud.
        if (channels != 24) {
            LOG_WARN(
                "minimax_h3_vae: %zu latent channels but the reference statistics cover 24; "
                "call set_latent_stats() with this checkpoint's values",
                channels);
            latents_mean_host.assign(channels, 0.0f);
            latents_std_host.assign(channels, 1.0f);
        }
    }

    void set_latent_stats(const std::vector<float>& mean, const std::vector<float>& std) {
        if (mean.size() != static_cast<size_t>(config.latent_channels) ||
            std.size() != static_cast<size_t>(config.latent_channels)) {
            LOG_ERROR("minimax_h3_vae: latent statistics must have %lld entries, got %zu/%zu",
                      (long long)config.latent_channels, mean.size(), std.size());
            return;
        }
        latents_mean_host = mean;
        latents_std_host  = std;
    }

    std::string get_desc() override {
        return "minimax_h3_video_vae";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        vae.get_param_tensors(tensors, weight_prefix);
    }

    // This VAE tiles itself, with the reference's own geometry. Leave the generic
    // `--vae-tiling` path OFF: VAE::encode/decode would tile a second time using
    // VAE::get_scale_factor(), which is not virtual and returns 8 for an unrecognized
    // SDVersion -- wrong for a 16x VAE. Tune the internal geometry here instead:
    //   tile_size=<px>  tile_overlap_min=<px>  tiling=<0|1>  clip_length=<frames>
    void set_tiling_params(const sd_tiling_params_t& params) override {
        for (const auto& [key, value] : parse_key_value_args(params.extra_tiling_args, "MiniMax-H3 VAE extra tiling arg")) {
            int parsed = 0;
            if (!parse_strict_int(value, parsed)) {
                LOG_WARN("ignoring invalid MiniMax-H3 VAE extra tiling arg '%s=%s'", key.c_str(), value.c_str());
            } else if (key == "tile_size") {
                config.tile_size = std::max(config.vae_ratio(), static_cast<int64_t>(parsed));
            } else if (key == "tile_overlap_min") {
                config.tile_overlap_min = std::max<int64_t>(0, parsed);
            } else if (key == "tiling") {
                config.tiling = parsed != 0;
            } else if (key == "clip_length") {
                config.clip_length = std::max<int64_t>(1, parsed);
            } else {
                LOG_WARN("ignoring unknown MiniMax-H3 VAE extra tiling arg '%s'", key.c_str());
            }
        }
    }

    int get_encoder_output_channels(int input_channels) override {
        SD_UNUSED(input_channels);
        return static_cast<int>(config.latent_channels);
    }

    // encode() already emits the normalized mean (the reference folds the DiagonalGaussian's
    // mean split and the per-channel statistics into encode itself), so there is nothing left
    // to sample or rescale here.
    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        SD_UNUSED(rng);
        return vae_output;
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    // -----------------------------------------------------------------------
    // graphs
    // -----------------------------------------------------------------------

    ggml_cgraph* build_encode_graph(const sd::Tensor<float>& x_tensor) {
        ggml_cgraph* gf  = new_graph_custom(32768);
        ggml_tensor* x   = make_input(x_tensor);
        auto runner_ctx  = get_context();
        ggml_tensor* out = vae.encode_moments(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    ggml_cgraph* build_decode_graph(const sd::Tensor<float>& z_tensor,
                                    const sd::Tensor<float>& rope_cos_tensor,
                                    const sd::Tensor<float>& rope_sin_tensor) {
        // 36 transformer blocks emit roughly 40 nodes each once the rope slice/concat chain and
        // the fused-qkv split are counted; 65536 leaves plenty of headroom.
        ggml_cgraph* gf  = new_graph_custom(65536);
        ggml_tensor* z   = make_input(z_tensor);
        ggml_tensor* cos = make_input(rope_cos_tensor);
        ggml_tensor* sin = make_input(rope_sin_tensor);
        auto runner_ctx  = get_context();
        ggml_tensor* out = vae.decode_pixels(&runner_ctx, z, cos, sin);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    // One graph per tile: keep the weights staged (auto_free=false, free_compute_params=false)
    // but release each tile's activation buffer so the peak stays at one tile.
    sd::Tensor<float> run_encode_tile(const sd::Tensor<float>& tile, int n_threads) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_encode_graph(tile); };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, true, false), 4);
    }

    sd::Tensor<float> run_decode_tile(const sd::Tensor<float>& tile, int n_threads) {
        sd::Tensor<float> rope_cos;
        sd::Tensor<float> rope_sin;
        MiniMaxH3Video::build_rope_tables(tile.shape()[2],
                                          tile.shape()[1],
                                          tile.shape()[0],
                                          vae.num_suffix_tokens(),
                                          config.rope_dim(),
                                          config.decoder_rope_theta,
                                          &rope_cos,
                                          &rope_sin);
        auto get_graph = [&]() -> ggml_cgraph* { return build_decode_graph(tile, rope_cos, rope_sin); };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, true, false), 4);
    }

    // -----------------------------------------------------------------------
    // spatial tiling (MiniMaxH3VideoVAE.split_tiles / blend / tiled_encode / tiled_decode)
    // -----------------------------------------------------------------------

    struct TilePlan {
        std::vector<int64_t> start;
        std::vector<int64_t> len;
        std::vector<int64_t> overlap;
    };

    TilePlan split_tiles(int64_t input_len) const {
        TilePlan plan;
        const int64_t tile_size = config.tile_size;
        const int64_t ratio     = config.vae_ratio();

        if (tile_size >= input_len) {
            plan.start = {0};
            plan.len   = {input_len};
            return plan;
        }

        int64_t n = (input_len + tile_size - 1) / tile_size;
        std::vector<int64_t> overlaps;
        int64_t remaining = 0;
        while (true) {
            overlaps.assign(static_cast<size_t>(std::max<int64_t>(n - 1, 0)), config.tile_overlap_min);
            int64_t sum = 0;
            for (int64_t o : overlaps) {
                sum += o;
            }
            remaining = tile_size * n - sum - input_len;
            if (remaining < 0) {
                n += 1;
            } else {
                break;
            }
        }

        if (!overlaps.empty()) {
            const int64_t remaining_units = remaining / ratio;
            for (int64_t i = 0; i < remaining_units; ++i) {
                overlaps[static_cast<size_t>(i % static_cast<int64_t>(overlaps.size()))] += ratio;
            }
        }

        plan.start.push_back(0);
        for (int64_t i = 0; i + 1 < n; ++i) {
            plan.start.push_back(plan.start.back() + tile_size - overlaps[static_cast<size_t>(i)]);
        }
        plan.len.assign(static_cast<size_t>(n), tile_size);
        plan.overlap = overlaps;
        return plan;
    }

    // MiniMaxH3VideoVAE.blend: linear cross-fade of a's trailing `extent` into b's leading
    // `extent`, with b's remainder appended untouched.  Torch dims map as -1 -> 0 (W),
    // -2 -> 1 (H), -3 -> 2 (T) in this layout.
    static sd::Tensor<float> blend(const sd::Tensor<float>& a,
                                   const sd::Tensor<float>& b,
                                   int64_t extent,
                                   size_t dim) {
        extent = std::min<int64_t>({a.shape()[dim], b.shape()[dim], extent});
        if (extent <= 0) {
            return b;
        }

        sd::Tensor<float> out = b;

        int64_t inner = 1;
        for (size_t i = 0; i < dim; ++i) {
            inner *= b.shape()[i];
        }
        int64_t outer = 1;
        for (size_t i = dim + 1; i < static_cast<size_t>(b.dim()); ++i) {
            outer *= b.shape()[i];
        }
        const int64_t a_dim = a.shape()[dim];
        const int64_t b_dim = b.shape()[dim];

        const float* ap = a.data();
        const float* bp = b.data();
        float* op       = out.data();

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t k = 0; k < extent; ++k) {
                const float wb      = static_cast<float>(k) / static_cast<float>(extent);
                const float wa      = 1.0f - wb;
                const int64_t a_off = (o * a_dim + (a_dim - extent + k)) * inner;
                const int64_t b_off = (o * b_dim + k) * inner;
                for (int64_t i = 0; i < inner; ++i) {
                    op[b_off + i] = ap[a_off + i] * wa + bp[b_off + i] * wb;
                }
            }
        }
        return out;
    }

    // dst/src: [W, H, T, C] with matching T and C; copies src into dst at (x0, y0).
    //
    // The tile plan can overhang the canvas by up to vae_ratio-1 px when the input is not a
    // clean multiple of the tile geometry (torch would raise on the equivalent canvas write;
    // here the overhang is clipped, because a silent heap overrun is the worse failure).
    static void write_block(sd::Tensor<float>* dst,
                            int64_t x0,
                            int64_t y0,
                            const sd::Tensor<float>& src) {
        const int64_t dw = dst->shape()[0];
        const int64_t dh = dst->shape()[1];
        const int64_t sw = src.shape()[0];
        const int64_t sh = src.shape()[1];
        const int64_t cw = std::min<int64_t>(sw, dw - x0);
        const int64_t ch = std::min<int64_t>(sh, dh - y0);
        if (cw <= 0 || ch <= 0) {
            LOG_WARN("minimax_h3_vae: tile at (%lld, %lld) falls entirely outside the %lldx%lld canvas",
                     (long long)x0, (long long)y0, (long long)dw, (long long)dh);
            return;
        }
        if (cw != sw || ch != sh) {
            LOG_WARN("minimax_h3_vae: clipping a %lldx%lld tile at (%lld, %lld) to fit the %lldx%lld canvas",
                     (long long)sw, (long long)sh, (long long)x0, (long long)y0, (long long)dw, (long long)dh);
        }
        int64_t planes = 1;
        for (size_t i = 2; i < static_cast<size_t>(dst->dim()); ++i) {
            planes *= dst->shape()[i];
        }

        const float* sp = src.data();
        float* dp       = dst->data();
        for (int64_t p = 0; p < planes; ++p) {
            const float* splane = sp + p * sw * sh;
            float* dplane       = dp + p * dw * dh;
            for (int64_t y = 0; y < ch; ++y) {
                std::copy_n(splane + y * sw, cw, dplane + (y0 + y) * dw + x0);
            }
        }
    }

    // MiniMaxH3VideoVAE.tiled_encode.  x: [W, H, T, 3] -> [W/16, H/16, T_lat, 2*z]
    sd::Tensor<float> tiled_encode(const sd::Tensor<float>& x, int n_threads) {
        const int64_t width  = x.shape()[0];
        const int64_t height = x.shape()[1];
        const int64_t ratio  = config.vae_ratio();

        const TilePlan yp = split_tiles(height);
        const TilePlan xp = split_tiles(width);

        std::vector<std::vector<sd::Tensor<float>>> rows;
        for (size_t i = 0; i < yp.start.size(); ++i) {
            std::vector<sd::Tensor<float>> row;
            for (size_t j = 0; j < xp.start.size(); ++j) {
                // torch slicing clamps past the end; the plan can overhang by < vae_ratio when
                // the input is not a clean multiple of the tile geometry.
                const int64_t y1 = std::min<int64_t>(height, yp.start[i] + yp.len[i]);
                const int64_t x1 = std::min<int64_t>(width, xp.start[j] + xp.len[j]);
                auto tile        = sd::ops::slice(x, 1, yp.start[i], y1);
                tile             = sd::ops::slice(tile, 0, xp.start[j], x1);

                auto moments = run_encode_tile(tile, n_threads);
                if (moments.empty()) {
                    return {};
                }
                row.push_back(std::move(moments));
            }
            rows.push_back(std::move(row));
        }

        std::vector<int64_t> latent_y_overlap;
        for (int64_t o : yp.overlap) {
            latent_y_overlap.push_back(o / ratio);
        }
        std::vector<int64_t> latent_x_overlap;
        for (int64_t o : xp.overlap) {
            latent_x_overlap.push_back(o / ratio);
        }

        sd::Tensor<float> out;
        for (size_t i = 0; i < rows.size(); ++i) {
            sd::Tensor<float> out_row;
            for (size_t j = 0; j < rows[i].size(); ++j) {
                sd::Tensor<float> tile = rows[i][j];
                // Blend against the ORIGINAL (unblended) neighbours, as the reference does.
                if (i > 0) {
                    tile = blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], 1);
                }
                if (j > 0) {
                    tile = blend(rows[i][j - 1], tile, latent_x_overlap[j - 1], 0);
                }
                if (i + 1 < rows.size()) {
                    tile = sd::ops::slice(tile, 1, 0, tile.shape()[1] - latent_y_overlap[i]);
                }
                if (j + 1 < rows[i].size()) {
                    tile = sd::ops::slice(tile, 0, 0, tile.shape()[0] - latent_x_overlap[j]);
                }
                out_row = out_row.empty() ? std::move(tile) : sd::ops::concat(out_row, tile, 0);
            }
            out = out.empty() ? std::move(out_row) : sd::ops::concat(out, out_row, 1);
        }
        return out;
    }

    // MiniMaxH3VideoVAE.tiled_decode.  z: [W_lat, H_lat, T_lat, 24] -> [W, H, T_lat*4, 3]
    sd::Tensor<float> tiled_decode(const sd::Tensor<float>& z, int n_threads) {
        const int64_t ratio  = config.vae_ratio();
        const int64_t width  = z.shape()[0] * ratio;
        const int64_t height = z.shape()[1] * ratio;

        const TilePlan yp = split_tiles(height);
        const TilePlan xp = split_tiles(width);

        sd::Tensor<float> canvas;
        std::vector<sd::Tensor<float>> row_tails;
        int64_t out_y       = 0;
        int64_t last_tile_h = 0;

        for (size_t i = 0; i < yp.start.size(); ++i) {
            const int64_t zi = yp.start[i] / ratio;
            const int64_t zl = std::min<int64_t>(z.shape()[1] - zi, yp.len[i] / ratio);

            std::vector<sd::Tensor<float>> new_tails;
            sd::Tensor<float> left_tail;
            int64_t out_x = 0;

            for (size_t j = 0; j < xp.start.size(); ++j) {
                const int64_t zj = xp.start[j] / ratio;
                const int64_t zw = std::min<int64_t>(z.shape()[0] - zj, xp.len[j] / ratio);

                auto z_tile = sd::ops::slice(z, 1, zi, zi + zl);
                z_tile      = sd::ops::slice(z_tile, 0, zj, zj + zw);

                auto tile = run_decode_tile(z_tile, n_threads);
                if (tile.empty()) {
                    return {};
                }

                if (i + 1 < yp.start.size()) {
                    new_tails.push_back(sd::ops::slice(tile, 1, tile.shape()[1] - yp.overlap[i], tile.shape()[1]));
                }
                sd::Tensor<float> next_left_tail;
                if (j + 1 < xp.start.size()) {
                    next_left_tail = sd::ops::slice(tile, 0, tile.shape()[0] - xp.overlap[j], tile.shape()[0]);
                }

                if (i > 0) {
                    tile = blend(row_tails[j], tile, yp.overlap[i - 1], 1);
                }
                if (j > 0) {
                    tile = blend(left_tail, tile, xp.overlap[j - 1], 0);
                }
                left_tail = std::move(next_left_tail);

                if (i + 1 < yp.start.size()) {
                    tile = sd::ops::slice(tile, 1, 0, tile.shape()[1] - yp.overlap[i]);
                }
                if (j + 1 < xp.start.size()) {
                    tile = sd::ops::slice(tile, 0, 0, tile.shape()[0] - xp.overlap[j]);
                }

                if (canvas.empty()) {
                    std::vector<int64_t> shape = tile.shape();
                    shape[0]                   = width;
                    shape[1]                   = height;
                    canvas                     = sd::Tensor<float>(shape);
                    canvas.fill_(0.0f);
                }
                write_block(&canvas, out_x, out_y, tile);
                out_x += tile.shape()[0];
                last_tile_h = tile.shape()[1];
            }
            row_tails = std::move(new_tails);
            out_y += last_tile_h;
        }
        return canvas;
    }

    sd::Tensor<float> adaptive_encode(const sd::Tensor<float>& x, int n_threads) {
        if (config.tiling) {
            return tiled_encode(x, n_threads);
        }
        return run_encode_tile(x, n_threads);
    }

    sd::Tensor<float> adaptive_decode(const sd::Tensor<float>& z, int n_threads) {
        if (config.tiling) {
            return tiled_decode(z, n_threads);
        }
        return run_decode_tile(z, n_threads);
    }

    // -----------------------------------------------------------------------
    // temporal chunking (MiniMaxH3VideoVAE.encode_temporal / decode_temporal)
    // -----------------------------------------------------------------------

    // x[..., -1:, ...] repeated `count` times and appended.  Allocated once rather than
    // concat-in-a-loop: at encode time `x` is a full pixel clip, so the naive version would
    // memcpy the whole thing up to clip_length-1 times.
    static sd::Tensor<float> repeat_last_along(const sd::Tensor<float>& x, size_t dim, int64_t count) {
        if (count <= 0) {
            return x;
        }
        const int64_t n = x.shape()[dim];
        int64_t inner   = 1;
        for (size_t i = 0; i < dim; ++i) {
            inner *= x.shape()[i];
        }
        int64_t outer = 1;
        for (size_t i = dim + 1; i < static_cast<size_t>(x.dim()); ++i) {
            outer *= x.shape()[i];
        }

        std::vector<int64_t> shape = x.shape();
        shape[dim]                 = n + count;
        sd::Tensor<float> out(shape);

        const float* sp = x.data();
        float* dp       = out.data();
        for (int64_t o = 0; o < outer; ++o) {
            const float* src = sp + o * n * inner;
            float* dst       = dp + o * (n + count) * inner;
            std::copy_n(src, n * inner, dst);
            for (int64_t k = 0; k < count; ++k) {
                std::copy_n(src + (n - 1) * inner, inner, dst + (n + k) * inner);
            }
        }
        return out;
    }

    sd::Tensor<float> encode_temporal(const sd::Tensor<float>& x_in, int n_threads) {
        sd::Tensor<float> x  = x_in;
        const int64_t clip   = config.clip_length;
        const int64_t frames = x.shape()[2];
        const int64_t pad    = ((clip - frames % clip) % clip);
        if (pad > 0) {
            x = repeat_last_along(x, 2, pad);
        }

        const int64_t num_chunks = x.shape()[2] / clip;
        sd::Tensor<float> z;
        for (int64_t i = 0; i < num_chunks; ++i) {
            auto clip_x = sd::ops::slice(x, 2, i * clip, (i + 1) * clip);
            auto chunk  = adaptive_encode(clip_x, n_threads);
            if (chunk.empty()) {
                return {};
            }
            z = z.empty() ? std::move(chunk) : sd::ops::concat(z, chunk, 2);
        }
        if (config.token_drop > 0 && !z.empty()) {
            z = sd::ops::slice(z, 2, 0, z.shape()[2] - config.token_drop);
        }
        return z;
    }

    // MiniMaxH3VideoVAE._decode_temporal_pad_frames
    int64_t decode_temporal_pad_frames(int64_t z_len, int64_t pad_tokens) const {
        if (pad_tokens <= 0) {
            return 0;
        }
        const int64_t ratio_t    = config.vae_ratio_t();
        const int64_t intra_tail = config.clip_length % ratio_t;
        if (intra_tail == 0) {
            return pad_tokens * ratio_t;
        }
        const int64_t chunk_tokens = config.tokens_chunk_size();
        const int64_t before       = z_len - pad_tokens;
        int64_t total              = 0;
        for (int64_t k = 0; k < pad_tokens; ++k) {
            total += ((before + k) % chunk_tokens == 0) ? intra_tail : ratio_t;
        }
        return total;
    }

    // MiniMaxH3VideoVAE._decode_temporal_frame_plan
    int64_t decode_temporal_frame_plan(int64_t z_len, int64_t num_chunks, int64_t pad_tokens) const {
        const int64_t chunk_tokens = config.tokens_chunk_size();
        const int64_t ratio_t      = config.vae_ratio_t();
        const int64_t chunk_dec    = chunk_tokens * ratio_t;
        const int64_t split_count  = (config.token_drop > 0 ? 1 : 0) + 1;
        const int64_t overlap      = config.token_overlap();
        const int64_t pre_pad      = config.frame_pre_padding();

        int64_t total_frames         = 0;
        int64_t final_overlap_frames = 0;

        for (int64_t i = 0; i < num_chunks; ++i) {
            const int64_t t_start        = i * chunk_tokens;
            const int64_t t_end          = t_start + chunk_tokens + overlap;
            const int64_t clip_token_len = std::max<int64_t>(0, std::min(t_end, z_len) - std::min(t_start, z_len));
            const int64_t clip_frame_len = clip_token_len * ratio_t;

            for (int64_t j = 0; j < split_count; ++j) {
                const int64_t f_start      = j * chunk_dec;
                const int64_t f_end        = std::min(f_start + chunk_dec, clip_frame_len);
                const int64_t chunk_frames = std::max<int64_t>(0, f_end - f_start - pre_pad);
                if (j == 0) {
                    total_frames += chunk_frames;
                } else {
                    final_overlap_frames = chunk_frames;
                }
            }
        }

        total_frames += final_overlap_frames;
        return total_frames - decode_temporal_pad_frames(z_len, pad_tokens);
    }

    sd::Tensor<float> decode_temporal(const sd::Tensor<float>& z_in, int n_threads) {
        const int64_t chunk_tokens = config.tokens_chunk_size();
        const int64_t ratio_t      = config.vae_ratio_t();
        const int64_t chunk_dec    = chunk_tokens * ratio_t;
        const int64_t split_count  = (config.token_drop > 0 ? 1 : 0) + 1;
        const int64_t overlap      = config.token_overlap();
        const int64_t pre_pad      = config.frame_pre_padding();

        int64_t pseudo_total    = z_in.shape()[2] + config.token_drop;
        int64_t pad_tokens      = 0;
        const int64_t remainder = pseudo_total % chunk_tokens;
        if (remainder != 0) {
            pad_tokens = chunk_tokens - remainder;
            pseudo_total += pad_tokens;
        }

        int64_t num_chunks = pseudo_total / chunk_tokens - (config.token_drop > 0 ? 1 : 0);
        if (num_chunks < 1) {
            // Too few tokens for one chunk (e.g. T_lat == 2): pad one extra chunk.
            pad_tokens += chunk_tokens;
            num_chunks += 1;
        }

        sd::Tensor<float> z = z_in;
        if (pad_tokens > 0) {
            z = repeat_last_along(z, 2, pad_tokens);
        }

        const int64_t output_frames = decode_temporal_frame_plan(z.shape()[2], num_chunks, pad_tokens);

        sd::Tensor<float> dec;
        sd::Tensor<float> dec_overlap;
        int64_t write_pos = 0;

        auto write_part = [&](const sd::Tensor<float>& part) {
            const int64_t part_frames = part.shape()[2];
            if (part_frames <= 0) {
                return;
            }
            if (dec.empty()) {
                std::vector<int64_t> shape = part.shape();
                shape[2]                   = output_frames;
                dec                        = sd::Tensor<float>(shape);
                dec.fill_(0.0f);
            }
            const int64_t copy_frames = std::min(part_frames, std::max<int64_t>(0, output_frames - write_pos));
            if (copy_frames > 0) {
                sd::ops::slice_assign(&dec, 2, write_pos, write_pos + copy_frames,
                                      sd::ops::slice(part, 2, 0, copy_frames));
                write_pos += copy_frames;
            }
        };

        for (int64_t i = 0; i < num_chunks; ++i) {
            const int64_t t_start = std::min<int64_t>(i * chunk_tokens, z.shape()[2]);
            const int64_t t_end   = std::min<int64_t>(t_start + chunk_tokens + overlap, z.shape()[2]);
            if (t_end <= t_start) {
                continue;
            }
            auto clip_z = sd::ops::slice(z, 2, t_start, t_end);

            auto clip_dec = adaptive_decode(clip_z, n_threads);
            if (clip_dec.empty()) {
                return {};
            }

            for (int64_t j = 0; j < split_count; ++j) {
                const int64_t f_start = j * chunk_dec;
                const int64_t f_end   = std::min(f_start + chunk_dec, clip_dec.shape()[2]);
                if (f_end <= f_start) {
                    continue;
                }
                auto part = sd::ops::slice(clip_dec, 2, f_start, f_end);
                if (pre_pad > 0) {
                    if (part.shape()[2] <= pre_pad) {
                        continue;
                    }
                    part = sd::ops::slice(part, 2, pre_pad, part.shape()[2]);
                }

                if (j == 0) {
                    if (!dec_overlap.empty()) {
                        part        = blend(dec_overlap, part, config.frame_overlap(), 2);
                        dec_overlap = sd::Tensor<float>();
                    }
                    write_part(part);
                } else {
                    dec_overlap = std::move(part);
                }
            }

            if (i == num_chunks - 1 && !dec_overlap.empty()) {
                write_part(dec_overlap);
                dec_overlap = sd::Tensor<float>();
            }
        }
        return dec;
    }

    // -----------------------------------------------------------------------
    // host-side affine normalizations
    // -----------------------------------------------------------------------

    // x is [W, H, T, C]: channel c occupies the contiguous plane block [c*W*H*T, (c+1)*W*H*T).
    static void affine_per_channel(sd::Tensor<float>* x, const float* scale, const float* shift, int64_t channels) {
        const int64_t plane = x->shape()[0] * x->shape()[1] * x->shape()[2];
        GGML_ASSERT(x->shape()[3] == channels);
        float* p = x->data();
        for (int64_t c = 0; c < channels; ++c) {
            const float s = scale[c];
            const float b = shift[c];
            float* base   = p + c * plane;
            for (int64_t i = 0; i < plane; ++i) {
                base[i] = base[i] * s + b;
            }
        }
    }

    // -----------------------------------------------------------------------
    // VAE entry points
    // -----------------------------------------------------------------------

    // MiniMaxH3VideoVAE.encode: pixels in [-1, 1] -> normalized latents.
    sd::Tensor<float> encode_video(const sd::Tensor<float>& x_in, int n_threads) {
        sd::Tensor<float> x = x_in;

        // x.add(1).mul_(0.5).sub_(pixel_mean).div_(pixel_std)
        {
            float scale[3];
            float shift[3];
            for (int c = 0; c < 3; ++c) {
                scale[c] = 0.5f / MiniMaxH3Video::IMAGENET_STD[c];
                shift[c] = (0.5f - MiniMaxH3Video::IMAGENET_MEAN[c]) / MiniMaxH3Video::IMAGENET_STD[c];
            }
            affine_per_channel(&x, scale, shift, 3);
        }

        sd::Tensor<float> moments;
        if (x.shape()[2] == 1) {
            moments = adaptive_encode(x, n_threads);
            if (moments.empty()) {
                return {};
            }
            moments = sd::ops::slice(moments, 2, moments.shape()[2] - 1, moments.shape()[2]);
        } else {
            moments = encode_temporal(x, n_threads);
            if (moments.empty()) {
                return {};
            }
        }

        // chunk(moments, 2, dim=channels)[0] -- the DiagonalGaussian mean.
        auto mean = sd::ops::slice(moments, 3, 0, config.latent_channels);

        std::vector<float> scale(static_cast<size_t>(config.latent_channels));
        std::vector<float> shift(static_cast<size_t>(config.latent_channels));
        for (int64_t c = 0; c < config.latent_channels; ++c) {
            const float m                 = latents_mean_host[static_cast<size_t>(c)];
            const float s                 = latents_std_host[static_cast<size_t>(c)];
            scale[static_cast<size_t>(c)] = 1.0f / s;
            shift[static_cast<size_t>(c)] = -m / s;
        }
        affine_per_channel(&mean, scale.data(), shift.data(), config.latent_channels);
        return mean;
    }

    // MiniMaxH3VideoVAE.decode: normalized latents -> pixels in [-1, 1].
    sd::Tensor<float> decode_video(const sd::Tensor<float>& z_in, int n_threads) {
        sd::Tensor<float> z = z_in;

        affine_per_channel(&z, latents_std_host.data(), latents_mean_host.data(), config.latent_channels);

        sd::Tensor<float> dec;
        if (z.shape()[2] == 1) {
            dec = adaptive_decode(z, n_threads);
            if (dec.empty()) {
                return {};
            }
            dec = sd::ops::slice(dec, 2, dec.shape()[2] - 1, dec.shape()[2]);
        } else {
            dec = decode_temporal(z, n_threads);
            if (dec.empty()) {
                return {};
            }
        }

        // dec.mul_(pixel_std).add_(pixel_mean).clamp_(0, 1).mul_(2).sub_(1)
        affine_per_channel(&dec, MiniMaxH3Video::IMAGENET_STD, MiniMaxH3Video::IMAGENET_MEAN, 3);
        float* p = dec.data();
        for (int64_t i = 0; i < dec.numel(); ++i) {
            p[i] = std::min(1.0f, std::max(0.0f, p[i])) * 2.0f - 1.0f;
        }
        return dec;
    }

    // The graph layout is [W, H, T, C] throughout.  A 4-D host tensor is ambiguous between
    // [W, H, T, C] and a batched still, so disambiguate on the channel count -- 24 latent
    // channels for decode, 3 pixel channels for encode -- and treat a 3-D tensor as a single
    // frame.  5-D [W, H, T, C, B] is accepted with B == 1 (make_ggml_tensor folds it anyway).
    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z,
                               bool decode_graph) override {
        if (!decode_graph && decode_only) {
            LOG_ERROR("MiniMax-H3 video VAE encode requires encoder weights");
            return {};
        }

        const size_t expected_dim   = static_cast<size_t>(z.dim());
        const int64_t want_channels = decode_graph ? config.latent_channels : config.in_channels;

        sd::Tensor<float> input = z;
        if (input.dim() == 3) {
            if (input.shape()[2] != want_channels) {
                LOG_ERROR("MiniMax-H3 video VAE expected %lld channels, got a 3-D tensor with %lld",
                          (long long)want_channels, (long long)input.shape()[2]);
                return {};
            }
            input = input.unsqueeze(2);  // [W, H, 1, C]
        } else if (input.dim() == 5) {
            if (input.shape()[4] != 1) {
                LOG_ERROR("MiniMax-H3 video VAE only supports batch size 1, got %lld",
                          (long long)input.shape()[4]);
                return {};
            }
            input = input.squeeze(4);
        } else if (input.dim() != 4) {
            LOG_ERROR("MiniMax-H3 video VAE expects a 3/4/5-D tensor, got dim=%lld", (long long)input.dim());
            return {};
        }

        if (input.shape()[3] != want_channels) {
            LOG_ERROR("MiniMax-H3 video VAE expected %lld channels in the last dimension, got %lld",
                      (long long)want_channels, (long long)input.shape()[3]);
            return {};
        }

        sd::Tensor<float> out = decode_graph ? decode_video(input, n_threads) : encode_video(input, n_threads);

        // Each tile ran with auto_free=false / free_compute_params=false to keep the weights
        // staged across the tile loop; release them once the whole pass is done.
        runner_done();

        if (out.empty()) {
            return {};
        }
        return restore_trailing_singleton_dims(std::move(out), expected_dim);
    }

    void test(const std::string& input_path) {
        auto z = sd::load_tensor_from_file_as_tensor<float>(input_path);
        print_sd_tensor(z, false, "minimax_h3_vae_z");

        const int64_t t0 = ggml_time_ms();
        auto out         = _compute(8, z, true);
        const int64_t t1 = ggml_time_ms();

        GGML_ASSERT(!out.empty());
        print_sd_tensor(out, false, "minimax_h3_vae_out");
        LOG_DEBUG("minimax h3 vae test done in %lldms", (long long)(t1 - t0));
    }

    static void load_from_file_and_test(const std::string& model_path,
                                        const std::string& input_path,
                                        const std::string& prefix = "first_stage_model") {
        ggml_backend_t backend = sd_backend_cpu_init();
        LOG_INFO("loading minimax h3 video vae from '%s'", model_path.c_str());

        auto model_manager        = std::make_shared<ModelManager>();
        ModelLoader& model_loader = model_manager->loader();
        if (!model_loader.init_from_file_and_convert_name(model_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
            return;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();
        auto vae                 = std::make_shared<MiniMaxH3VideoVAERunner>(backend,
                                                                             tensor_storage_map,
                                                                             prefix,
                                                                             true,
                                                                             VERSION_COUNT,
                                                                             model_manager);

        if (!model_manager->register_runner_params("MiniMax H3 video VAE test",
                                                   *vae,
                                                   ModelManager::ResidencyMode::ParamBackend,
                                                   backend,
                                                   backend) ||
            !model_manager->validate_registered_tensors()) {
            LOG_ERROR("register minimax h3 video vae tensors with model manager failed");
            return;
        }

        LOG_INFO("minimax h3 video vae model loaded");
        vae->test(input_path);
    }
};

#endif  // __SD_MODEL_VAE_MINIMAX_H3_VAE_HPP__
