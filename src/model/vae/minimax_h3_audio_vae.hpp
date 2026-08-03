#ifndef __SD_MODEL_VAE_MINIMAX_H3_AUDIO_VAE_HPP__
#define __SD_MODEL_VAE_MINIMAX_H3_AUDIO_VAE_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/vae/ltx_audio_vae.hpp"
#include "model_loader.h"
#include "model_manager.h"

// MiniMax-H3 audio VAE: DAC-lineage waveform encoder + causal-attention
// bottleneck + BigVGAN decoder, 32 kHz, 800 audio samples per latent frame.
//
// Reference (the spec this file was ported from):
//   comfy/ldm/minimax/audio_vae.py            (primary; weight-norm already folded)
//   diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py  (cross-check)
// Reference line numbers quoted below are from the comfy file.
//
// The model itself is MONO. Stereo is carried as two independent batch items;
// this port runs the two channels as two sequential sub-graphs sharing one set
// of weights, which keeps every convolution at ggml batch 1 (the anti-aliased
// depthwise helpers below only handle ne[2] == ne[3] == 1).
//
// CHANNEL LAYOUT -- read this before wiring anything up.
//
//   Latents are CHANNEL-MAJOR (planar), never interleaved:
//       decode() input / encode() output  sd::Tensor<float> shape {T, C_lat, S}
//       ne = (T, C_lat, S): ne0 = latent frame (fastest), ne1 = latent channel,
//       ne2 = stereo channel (slowest).
//       That is exactly torch's [S, C_lat, T] laid out contiguously, i.e. all of
//       stereo channel 0 (its C_lat x T block) and only then stereo channel 1.
//
//   Waveforms are CHANNEL-MAJOR (planar) too:
//       decode() output / encode() input   sd::Tensor<float> shape {L, S}
//       ne = (L, S): ne0 = sample (fastest), ne1 = stereo channel.
//       Sample s of channel c lives at data[c * L + s]. This is NOT the
//       L-R-L-R interleaving that WAV/ffmpeg/miniaudio hand you -- feeding
//       interleaved data here does not error, it silently destroys the stereo
//       field (you get a comb of alternating half-rate channels). Use
//       interleaved_to_planar() / planar_to_interleaved() at the I/O boundary.

namespace MiniMaxH3Audio {

    // Reused verbatim from the LTX audio VAE (same BigVGAN lineage, same house
    // idiom). Do not fork these -- if one needs to change, factor it out of
    // ltx_audio_vae.hpp instead.
    using LTXV::Conv1D;           // torch nn.Conv1d, weight ne = (K, IC, OC)
    using LTXV::ConvTranspose1D;  // torch nn.ConvTranspose1d, weight ne = (K, OC, IC)
    using LTXV::depthwise_conv1d;
    using LTXV::replicate_pad_1d;
    using LTXV::SnakeBeta1D;  // x + sin^2(exp(alpha) x) / (exp(beta) + 1e-9)

    // ---------------------------------------------------------------------
    // audio_vae/config.json latents_mean / latents_std (32 channels). BYTE-IDENTICAL in the
    // released FL2VA and Ref2VA variants.
    //
    // ⚠️ These are the AUDIO twin of MiniMaxH3Video::LATENTS_MEAN / LATENTS_STD, and for the
    // same reason: the reference registers them as buffers filled from `config.json`, the
    // conversion script rejects them as unexpected keys, so they are NOT in the GGUF -- and
    // `detect_from_weights` therefore sets has_latent_stats = false. Without them decode()
    // skips `z * std + mean` entirely and the BigVGAN decoder is handed the DiT's NORMALIZED
    // latents: ~1.5-3.3x too small per channel and offset by up to 0.59. The video path was
    // given its literals; the audio path was not.
    //
    // Reference: comfy/ldm/minimax/audio_vae.py MiniMaxH3AudioVAE.decode (z * std + mean) /
    // encode ((z - mean) / std); diffusers modular_pipelines/minimax_h3/decoders.py L186-188.
    static const float LATENTS_MEAN[32] = {
        -0.020211687488382354f, 0.38764664799505022f, -0.043982797991867668f, -0.28591514936373003f,
        0.081796862145616711f, -0.35782641352446604f, 0.040623809960919084f, -0.015525345019566039f,
        -0.223362481667332f, 0.18210068425090911f, 0.2941778783780663f, -0.079011676019708849f,
        -0.056815072777200999f, -0.36990282218600951f, -0.31616315591624855f, 0.59059513774253913f,
        -0.052139568068853864f, 0.013673160263486295f, -0.036916478646305768f, 0.097326606532981627f,
        -0.33946623287884981f, -0.30685677538541667f, -0.24504598907458763f, -0.034698524462007344f,
        0.028680321847675379f, -0.21217779266454084f, -0.1678263169941987f, 0.32212878890406138f,
        -0.1223055851554907f, 0.43566049281284641f, -0.050259920223625298f, 0.39792583762117972f};

    static const float LATENTS_STD[32] = {
        1.6895524230479284f, 2.76263727217653f, 1.7945344281264435f, 1.6801681847309828f,
        1.6390226546605453f, 2.7788298348882177f, 1.7659090095747236f, 1.6199757612137327f,
        2.6336525640336896f, 1.8539356672817833f, 2.5056497896915633f, 1.811019237886178f,
        1.9579657790720237f, 1.6685498243529284f, 1.4922469314453364f, 3.2986701980673732f,
        1.9491804496832168f, 1.8720003270431442f, 1.8334080103291832f, 1.6488070416529093f,
        1.6176957696319716f, 1.9131449234774398f, 1.5695245398428617f, 1.6943659940415912f,
        1.8318420762504692f, 1.5540637421583379f, 1.9344930328968526f, 1.599198216109855f,
        1.718045989838149f, 1.6307219190837705f, 1.8661226051202384f, 1.5613768203168363f};

    struct MiniMaxH3AudioVAEConfig {
        // Reference defaults: comfy audio_vae.py MiniMaxH3AudioVAE.__init__ (L382-413)
        // and BigVGAN.__init__ (L313-321).
        int sample_rate                                        = 32000;  // L391
        int encoder_dim                                        = 64;     // L383
        std::vector<int> encoder_rates                         = {2, 4, 4, 5, 5};        // L384
        int latent_dim                                         = 2048;                   // L385
        int latent_channels                                    = 32;                     // L388 vae_latent_channels
        int num_attention_heads                                = 8;                      // L402 num_heads=8
        int mlp_hidden_dim                                     = 64;                     // L261 int(out_dim * mlp_ratio), mlp_ratio=2
        int decoder_dim                                        = 1024;                   // L386
        std::vector<int> decoder_rates                         = {5, 5, 2, 2, 2, 2, 2};  // L317
        std::vector<int> decoder_kernel_sizes                  = {9, 9, 4, 4, 4, 4, 4};  // L318
        std::vector<int> resblock_kernel_sizes                 = {3, 7, 11};             // L319
        std::vector<std::vector<int>> resblock_dilation_sizes  = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};  // L320
        int activation_ratio                                   = 2;                      // L139 up_ratio/down_ratio
        int activation_kernel_size                             = 12;                     // L139 up/down_kernel_size
        int audio_channels                                     = 2;                      // stereo carried as two batch items

        // Detected from the tensor map.
        bool has_encoder      = false;
        bool has_latent_stats = false;
        bool detected         = false;

        int hop_length() const {
            int hop = 1;
            for (int rate : encoder_rates) {
                hop *= rate;
            }
            return hop;  // 800
        }

        int upsample_factor() const {
            int factor = 1;
            for (int rate : decoder_rates) {
                factor *= rate;
            }
            return factor;  // must equal hop_length()
        }

        int latents_per_second() const {
            return sample_rate / hop_length();  // 40
        }

        int output_sample_rate() const {
            return sample_rate;
        }

        int head_dim() const {
            return latent_dim / std::max(1, num_attention_heads);  // 256
        }

        int num_upsamples() const {
            return static_cast<int>(decoder_rates.size());
        }

        int decoder_final_channels() const {
            return decoder_dim >> num_upsamples();  // 1024 >> 7 = 8
        }

        // The checkpoint is a name-for-name passthrough of the original module
        // tree (see convert_minimax_h3_to_diffusers.py convert_audio_vae: the
        // keys are *validated*, never renamed), so every shape that is a tensor
        // dimension is read back off the tensor map. What is NOT recoverable
        // from shapes -- head count, conv dilations, upsample strides -- is
        // hardcoded to the reference default and noted above.
        static MiniMaxH3AudioVAEConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                           const std::string& prefix = "") {
            MiniMaxH3AudioVAEConfig config;

            const std::string base = prefix.empty() ? "" : prefix + ".";
            auto find              = [&](const std::string& name) -> const TensorStorage* {
                auto iter = tensor_storage_map.find(base + name);
                if (iter == tensor_storage_map.end()) {
                    return nullptr;
                }
                return &iter->second;
            };

            const TensorStorage* conv_pre    = find("decoder.conv_pre.weight");
            const TensorStorage* dec_in_proj = find("dec_in_proj.weight");
            const TensorStorage* conv_post   = find("decoder.conv_post.weight");
            if (conv_pre == nullptr || dec_in_proj == nullptr || conv_post == nullptr) {
                LOG_DEBUG("minimax_h3_audio_vae: decoder tensors not found under prefix '%s'", prefix.c_str());
                return config;
            }

            // decoder.conv_pre: nn.Conv1d(latent_dim, decoder_dim, 7) -> ne (7, latent_dim, decoder_dim)
            config.latent_dim  = static_cast<int>(conv_pre->ne[1]);
            config.decoder_dim = static_cast<int>(conv_pre->ne[2]);
            // dec_in_proj: nn.Conv1d(latent_channels, latent_dim, 1) -> ne (1, latent_channels, latent_dim)
            config.latent_channels = static_cast<int>(dec_in_proj->ne[1]);

            // decoder.ups.<i>.0: the extra ModuleList nesting of the reference
            // (L330-342) is part of the key, not an accident.
            config.decoder_kernel_sizes.clear();
            for (int i = 0;; ++i) {
                const TensorStorage* up = find("decoder.ups." + std::to_string(i) + ".0.weight");
                if (up == nullptr) {
                    break;
                }
                config.decoder_kernel_sizes.push_back(static_cast<int>(up->ne[0]));
            }
            if (config.decoder_kernel_sizes.empty()) {
                LOG_ERROR("minimax_h3_audio_vae: no decoder.ups.<i>.0.weight tensors found");
                return config;
            }

            // BigVGAN's kernel/stride convention is k = 2r for even r and 2r-1
            // for odd r, so (k + 1) / 2 recovers the stride for both. The
            // product is checked against the encoder hop below; on any mismatch
            // the reference defaults are kept instead of a wrong guess.
            std::vector<int> decoder_rates;
            for (int kernel : config.decoder_kernel_sizes) {
                decoder_rates.push_back((kernel + 1) / 2);
            }

            // Encoder strides ARE exactly recoverable: EncoderBlock's strided
            // conv has kernel_size = 2 * stride (L184).
            std::vector<int> encoder_rates;
            const TensorStorage* enc_conv_in = find("encoder.block.0.weight");
            if (enc_conv_in != nullptr) {
                config.encoder_dim = static_cast<int>(enc_conv_in->ne[2]);
                for (int i = 1;; ++i) {
                    const TensorStorage* strided = find("encoder.block." + std::to_string(i) + ".block.4.weight");
                    if (strided == nullptr) {
                        break;
                    }
                    encoder_rates.push_back(static_cast<int>(strided->ne[0] / 2));
                }
            }
            if (!encoder_rates.empty()) {
                config.encoder_rates = encoder_rates;
            }

            int decoder_factor = 1;
            for (int rate : decoder_rates) {
                decoder_factor *= rate;
            }
            if (decoder_factor == config.hop_length() && decoder_rates.size() == config.decoder_kernel_sizes.size()) {
                config.decoder_rates = decoder_rates;
            } else {
                LOG_WARN("minimax_h3_audio_vae: derived decoder rates (product %d) disagree with the encoder hop (%d); "
                         "keeping the reference defaults",
                         decoder_factor,
                         config.hop_length());
                config.decoder_kernel_sizes = MiniMaxH3AudioVAEConfig().decoder_kernel_sizes;
            }

            // decoder.resblocks.<j>.convs1.0: one AMP block per (upsample stage,
            // resblock kernel). The first num_kernels entries give the kernels.
            int resblock_count = 0;
            while (find("decoder.resblocks." + std::to_string(resblock_count) + ".convs1.0.weight") != nullptr) {
                ++resblock_count;
            }
            const int num_upsamples = static_cast<int>(config.decoder_rates.size());
            if (resblock_count > 0 && num_upsamples > 0 && resblock_count % num_upsamples == 0) {
                const int num_kernels = resblock_count / num_upsamples;
                std::vector<int> resblock_kernel_sizes;
                for (int j = 0; j < num_kernels; ++j) {
                    const TensorStorage* conv = find("decoder.resblocks." + std::to_string(j) + ".convs1.0.weight");
                    resblock_kernel_sizes.push_back(static_cast<int>(conv->ne[0]));
                }
                config.resblock_kernel_sizes = resblock_kernel_sizes;
                // Dilations are not a tensor dimension; hold the per-block
                // dilation triple at the reference default (L320) and widen it
                // to however many parallel blocks the checkpoint actually has.
                config.resblock_dilation_sizes.assign(static_cast<size_t>(num_kernels), {1, 3, 5});
            }

            const TensorStorage* mlp_w0 = find("pre_block.mlp.w0.weight");
            if (mlp_w0 != nullptr) {
                config.mlp_hidden_dim = static_cast<int>(mlp_w0->ne[1]);
            }

            config.has_encoder      = find("encoder.block.0.weight") != nullptr &&
                                 find("pre_block.attn.qkv.weight") != nullptr &&
                                 find("mean_proj.weight") != nullptr;
            config.has_latent_stats = find("latents_mean") != nullptr && find("latents_std") != nullptr;
            config.detected         = true;

            if (config.latent_dim % std::max(1, config.num_attention_heads) != 0 ||
                config.head_dim() % std::max(1, config.latent_channels) != 0) {
                LOG_WARN("minimax_h3_audio_vae: latent_dim %d / heads %d / latent_channels %d do not divide evenly; "
                         "the attention head pooling will assert",
                         config.latent_dim,
                         config.num_attention_heads,
                         config.latent_channels);
            }

            LOG_DEBUG("minimax_h3_audio_vae: latent_dim = %d, latent_channels = %d, decoder_dim = %d, hop = %d, "
                      "latents/s = %d, has_encoder = %s, has_latent_stats = %s",
                      config.latent_dim,
                      config.latent_channels,
                      config.decoder_dim,
                      config.hop_length(),
                      config.latents_per_second(),
                      config.has_encoder ? "true" : "false",
                      config.has_latent_stats ? "true" : "false");
            return config;
        }
    };

    // ---------------------------------------------------------------------
    // Anti-aliased resampling filters.
    //
    // The reference registers these as persistent buffers, so a faithful GGUF
    // would carry ~254 twelve-tap copies of the same two vectors. They are
    // instead rebuilt here from the same closed form (comfy L57-82), because a
    // declared-but-absent parameter is a hard load failure while an unused
    // checkpoint tensor is silently ignored -- and because the construction is
    // exactly reproducible. If the values ever have to be cross-checked, the
    // 12-tap filter for cutoff=0.25 / half_width=0.3 is
    //   0.002028966  0.009389464 -0.025543464 -0.057657375  0.128572609  0.443209800
    //   0.443209800  0.128572609 -0.057657375 -0.025543464  0.009389464  0.002028966
    // (kaiser beta 4.663800128, sums to 1).
    // ---------------------------------------------------------------------

    static double modified_bessel_i0(double x) {
        // I0(x) = sum_k (x/2)^(2k) / (k!)^2. torch.kaiser_window's beta stays
        // well under 10 here, so the series converges in a handful of terms.
        double sum  = 1.0;
        double term = 1.0;
        for (int k = 1; k < 128; ++k) {
            term *= (x * x) / (4.0 * static_cast<double>(k) * static_cast<double>(k));
            sum += term;
            if (term < 1e-17 * sum) {
                break;
            }
        }
        return sum;
    }

    static std::vector<float> kaiser_sinc_filter1d(double cutoff, double half_width, int kernel_size) {
        GGML_ASSERT(kernel_size > 0);
        constexpr double kPi = 3.14159265358979323846;

        const int half_size      = kernel_size / 2;
        const double delta_f     = 4.0 * half_width;
        const double attenuation = 2.285 * (static_cast<double>(half_size) - 1.0) * kPi * delta_f + 7.95;
        double beta              = 0.0;
        if (attenuation > 50.0) {
            beta = 0.1102 * (attenuation - 8.7);
        } else if (attenuation >= 21.0) {
            beta = 0.5842 * std::pow(attenuation - 21.0, 0.4) + 0.07886 * (attenuation - 21.0);
        }

        // torch.kaiser_window(kernel_size, beta=beta, periodic=False)
        const double alpha = (static_cast<double>(kernel_size) - 1.0) / 2.0;
        const double denom = modified_bessel_i0(beta);
        std::vector<double> window(static_cast<size_t>(kernel_size), 1.0);
        if (alpha > 0.0) {
            for (int i = 0; i < kernel_size; ++i) {
                const double ratio = (static_cast<double>(i) - alpha) / alpha;
                const double arg   = std::max(0.0, 1.0 - ratio * ratio);
                window[static_cast<size_t>(i)] = modified_bessel_i0(beta * std::sqrt(arg)) / denom;
            }
        }

        std::vector<double> taps(static_cast<size_t>(kernel_size));
        double sum = 0.0;
        for (int i = 0; i < kernel_size; ++i) {
            const double time = (kernel_size % 2 == 0)
                                    ? (static_cast<double>(-half_size + i) + 0.5)
                                    : static_cast<double>(i - half_size);
            const double arg  = 2.0 * cutoff * time;
            // torch.sinc: sin(pi x) / (pi x), 1 at x == 0.
            const double sinc = (arg == 0.0) ? 1.0 : std::sin(kPi * arg) / (kPi * arg);
            taps[static_cast<size_t>(i)] = 2.0 * cutoff * window[static_cast<size_t>(i)] * sinc;
            sum += taps[static_cast<size_t>(i)];
        }
        GGML_ASSERT(sum != 0.0);

        std::vector<float> filter(static_cast<size_t>(kernel_size));
        for (int i = 0; i < kernel_size; ++i) {
            filter[static_cast<size_t>(i)] = static_cast<float>(taps[static_cast<size_t>(i)] / sum);
        }
        return filter;
    }

    // Depthwise transposed convolution by a filter that is ALREADY time-reversed
    // on the host.
    //
    // LTXV::depthwise_conv_transpose1d does the same thing but flips the filter
    // inside the graph, which costs 2 * kernel_size nodes on every call. The
    // BigVGAN decoder runs 127 anti-aliased activations, so that flip alone is
    // ~3k redundant nodes per decode. Everything else here is LTX's algorithm:
    // zero-stuff by `stride`, cross-correlate with the flipped filter (which is
    // a true convolution), crop, and scale by `stride` for the reference's
    // `.mul_(self.ratio)` (comfy L101).
    static ggml_tensor* depthwise_conv_transpose1d_prereversed(ggml_context* ctx,
                                                               ggml_tensor* x,
                                                               ggml_tensor* reversed_filter,
                                                               int stride) {
        GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1);
        GGML_ASSERT(reversed_filter->ne[1] == 1 && reversed_filter->ne[2] == 1 && reversed_filter->ne[3] == 1);
        GGML_ASSERT(stride >= 1);

        const int64_t time        = x->ne[0];
        const int64_t channels    = x->ne[1];
        const int64_t kernel_size = reversed_filter->ne[0];
        const int64_t out_time    = (time - 1) * stride + kernel_size;

        auto x_flat = ggml_reshape_3d(ctx, x, 1, time, channels);
        if (stride > 1) {
            auto zero_unit = ggml_ext_scale(ctx, x_flat, 0.0f);
            auto zero_tail = zero_unit;
            for (int i = 1; i < stride - 1; ++i) {
                zero_tail = ggml_concat(ctx, zero_tail, zero_unit, 0);
            }
            x_flat = ggml_concat(ctx, x_flat, zero_tail, 0);
        }
        x_flat = ggml_reshape_3d(ctx, x_flat, time * stride, 1, channels);

        auto out = ggml_conv_1d(ctx, reversed_filter, x_flat, 1, static_cast<int>(kernel_size - 1), 1);
        if (out->ne[0] > out_time) {
            out = ggml_ext_slice(ctx, out, 0, 0, out_time);
        }
        GGML_ASSERT(out->ne[0] == out_time);
        GGML_ASSERT(out->ne[1] == 1);
        GGML_ASSERT(out->ne[2] == channels);

        out = ggml_ext_scale(ctx, out, static_cast<float>(stride));
        return ggml_reshape_4d(ctx, out, out_time, channels, 1, 1);
    }

    // The two Kaiser filters every Activation1d shares, carried together so
    // they can be threaded through the block tree as one argument.
    struct AliasFreeFilters {
        ggml_tensor* upsample_reversed = nullptr;  // time-reversed, for the transposed conv
        ggml_tensor* downsample         = nullptr;
    };

    // ---------------------------------------------------------------------
    // Activations
    // ---------------------------------------------------------------------

    // Snake1d (comfy L30-38): per-channel alpha stored as [1, channels, 1],
    // alpha doubles as beta -- x + sin^2(alpha x) / (alpha + 1e-9).
    // NOT the same parameterization as SnakeBeta1D: no exp(), one parameter.
    struct Snake1d : public UnaryBlock {
        int64_t channels;
        float eps = 1e-9f;

        explicit Snake1d(int64_t channels)
            : channels(channels) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            // torch shape [1, channels, 1] -> ne (1, channels, 1, 1). The model
            // manager compares all four ne exactly, so this cannot be flattened.
            params["alpha"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto alpha       = ggml_reshape_4d(ctx->ggml_ctx, params["alpha"], 1, channels, 1, 1);
            auto oscillation = ggml_sin(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, alpha));
            oscillation      = ggml_mul(ctx->ggml_ctx, oscillation, oscillation);
            auto eps_tensor  = ggml_ext_full(ctx->ggml_ctx, eps, 1, 1, 1, 1);
            auto denom       = ggml_add(ctx->ggml_ctx, alpha, eps_tensor);
            return ggml_add(ctx->ggml_ctx, x, ggml_div(ctx->ggml_ctx, oscillation, denom));
        }
    };

    // Activation1d (comfy L136-149): upsample x2 -> SnakeBeta -> downsample x2.
    // Parameter layout matches LTXV::Activation1D exactly ("act.alpha",
    // "act.beta"); the difference is that the Kaiser filters arrive as graph
    // inputs instead of as loaded buffers, which is why this is not reused.
    struct Activation1d : public GGMLBlock {
        int64_t channels;
        int ratio;
        int kernel_size;

        Activation1d(int64_t channels, int ratio = 2, int kernel_size = 12)
            : channels(channels), ratio(ratio), kernel_size(kernel_size) {
            blocks["act"] = std::make_shared<SnakeBeta1D>(channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const AliasFreeFilters& filters) {
            auto act = std::dynamic_pointer_cast<SnakeBeta1D>(blocks["act"]);
            GGML_ASSERT(filters.upsample_reversed != nullptr && filters.downsample != nullptr);

            // UpSample1d (comfy L85-103)
            const int up_pad       = kernel_size / ratio - 1;
            const int up_pad_left  = up_pad * ratio + (kernel_size - ratio) / 2;
            const int up_pad_right = up_pad * ratio + (kernel_size - ratio + 1) / 2;

            x = replicate_pad_1d(ctx, x, up_pad, up_pad);
            x = depthwise_conv_transpose1d_prereversed(ctx->ggml_ctx, x, filters.upsample_reversed, ratio);
            x = ggml_ext_slice(ctx->ggml_ctx, x, 0, up_pad_left, x->ne[0] - up_pad_right);

            x = act->forward(ctx, x);

            // LowPassFilter1d (comfy L106-117)
            const int down_pad_left  = kernel_size / 2 - (kernel_size % 2 == 0 ? 1 : 0);
            const int down_pad_right = kernel_size / 2;
            x                        = replicate_pad_1d(ctx, x, down_pad_left, down_pad_right);
            x                        = depthwise_conv1d(ctx, x, filters.downsample, ratio, 0);
            return x;
        }
    };

    // ---------------------------------------------------------------------
    // DAC encoder
    // ---------------------------------------------------------------------

    // ResidualUnit (comfy L154-170). The block is an nn.Sequential, so the
    // checkpoint keys are block.0 (Snake1d) .. block.3 (Conv1d k=1).
    struct ResidualUnit : public UnaryBlock {
        ResidualUnit(int64_t dim, int dilation) {
            const int pad          = ((7 - 1) * dilation) / 2;
            blocks["block.0"]      = std::make_shared<Snake1d>(dim);
            blocks["block.1"]      = std::make_shared<Conv1D>(dim, dim, 7, 1, pad, dilation);
            blocks["block.2"]      = std::make_shared<Snake1d>(dim);
            blocks["block.3"]      = std::make_shared<Conv1D>(dim, dim, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto snake1 = std::dynamic_pointer_cast<Snake1d>(blocks["block.0"]);
            auto conv1  = std::dynamic_pointer_cast<Conv1D>(blocks["block.1"]);
            auto snake2 = std::dynamic_pointer_cast<Snake1d>(blocks["block.2"]);
            auto conv2  = std::dynamic_pointer_cast<Conv1D>(blocks["block.3"]);

            auto h = snake1->forward(ctx, x);
            h      = conv1->forward(ctx, h);
            h      = snake2->forward(ctx, h);
            h      = conv2->forward(ctx, h);

            // With the reference's padding the dilated conv is length-preserving,
            // so this centre crop is a no-op; it is kept because the reference
            // performs it and a future config could make it bite.
            const int64_t pad = (x->ne[0] - h->ne[0]) / 2;
            if (pad > 0) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 0, pad, x->ne[0] - pad);
            }
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    // EncoderBlock (comfy L173-191): three residual units at dilation 1/3/9,
    // then a channel-doubling strided conv.
    struct EncoderBlock : public UnaryBlock {
        EncoderBlock(int64_t dim, int stride) {
            blocks["block.0"] = std::make_shared<ResidualUnit>(dim / 2, 1);
            blocks["block.1"] = std::make_shared<ResidualUnit>(dim / 2, 3);
            blocks["block.2"] = std::make_shared<ResidualUnit>(dim / 2, 9);
            blocks["block.3"] = std::make_shared<Snake1d>(dim / 2);
            // padding = math.ceil(stride / 2); integer form (stride + 1) / 2
            blocks["block.4"] = std::make_shared<Conv1D>(dim / 2, dim, 2 * stride, stride, (stride + 1) / 2);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            for (int i = 0; i < 3; ++i) {
                auto unit = std::dynamic_pointer_cast<ResidualUnit>(blocks["block." + std::to_string(i)]);
                x         = unit->forward(ctx, x);
            }
            x = std::dynamic_pointer_cast<Snake1d>(blocks["block.3"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Conv1D>(blocks["block.4"])->forward(ctx, x);
            return x;
        }
    };

    // Encoder (comfy L194-208): [samples, 1] -> [samples / hop, latent_dim].
    struct Encoder : public UnaryBlock {
        int num_stages;

        explicit Encoder(const MiniMaxH3AudioVAEConfig& config)
            : num_stages(static_cast<int>(config.encoder_rates.size())) {
            blocks["block.0"] = std::make_shared<Conv1D>(1, config.encoder_dim, 7, 1, 3);
            int dim           = config.encoder_dim;
            for (int i = 0; i < num_stages; ++i) {
                dim *= 2;
                blocks["block." + std::to_string(i + 1)] = std::make_shared<EncoderBlock>(dim, config.encoder_rates[static_cast<size_t>(i)]);
            }
            blocks["block." + std::to_string(num_stages + 1)] = std::make_shared<Snake1d>(dim);
            blocks["block." + std::to_string(num_stages + 2)] = std::make_shared<Conv1D>(dim, config.latent_dim, 3, 1, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            x = std::dynamic_pointer_cast<Conv1D>(blocks["block.0"])->forward(ctx, x);
            for (int i = 0; i < num_stages; ++i) {
                x = std::dynamic_pointer_cast<EncoderBlock>(blocks["block." + std::to_string(i + 1)])->forward(ctx, x);
            }
            x = std::dynamic_pointer_cast<Snake1d>(blocks["block." + std::to_string(num_stages + 1)])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Conv1D>(blocks["block." + std::to_string(num_stages + 2)])->forward(ctx, x);
            return x;
        }
    };

    // ---------------------------------------------------------------------
    // Causal-attention bottleneck (`pre_block`). The LTX audio VAE has no
    // equivalent -- everything below is new.
    // ---------------------------------------------------------------------

    // GeGluMlp (comfy L213-224). Carries its own pre-norm, which is applied on
    // top of AttnProjection.norm2 -- two LayerNorms in a row, as in the reference.
    struct GeGluMlp : public UnaryBlock {
        GeGluMlp(int64_t in_features, int64_t hidden_features) {
            blocks["norm"] = std::make_shared<LayerNorm>(in_features);
            blocks["w0"]   = std::make_shared<Linear>(in_features, hidden_features);
            blocks["w1"]   = std::make_shared<Linear>(in_features, hidden_features);
            blocks["w2"]   = std::make_shared<Linear>(hidden_features, in_features);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto w0   = std::dynamic_pointer_cast<Linear>(blocks["w0"]);
            auto w1   = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2   = std::dynamic_pointer_cast<Linear>(blocks["w2"]);

            x        = norm->forward(ctx, x);
            auto gate = ggml_ext_gelu(ctx->ggml_ctx, w0->forward(ctx, x));  // nn.GELU(approximate="tanh")
            auto up   = w1->forward(ctx, x);
            return w2->forward(ctx, ggml_mul(ctx->ggml_ctx, gate, up));
        }
    };

    // CausalAttention (comfy L227-249).
    //
    // Three things here are unusual and easy to get wrong:
    //   * qkv is a single bias-less Linear; the bias is assembled at runtime as
    //     cat(q_bias, ZEROS, v_bias). The reference stores that zero block as a
    //     `zero_k_bias` buffer; it is materialized in-graph here instead, so the
    //     checkpoint tensor (if present) is simply unused.
    //   * the heads are MEAN-POOLED away, not concatenated.
    //   * the surviving head_dim is adaptive-average-pooled down to out_dim
    //     (256 -> 32 for the released config, i.e. plain 8-wide box averages).
    struct CausalAttention : public GGMLBlock {
        int64_t in_dim;
        int64_t out_dim;
        int num_heads;
        int64_t head_dim;

        CausalAttention(int64_t in_dim, int64_t out_dim, int num_heads)
            : in_dim(in_dim), out_dim(out_dim), num_heads(num_heads), head_dim(in_dim / num_heads) {
            blocks["qkv"]  = std::make_shared<Linear>(in_dim, in_dim * 3, false);
            blocks["proj"] = std::make_shared<Linear>(out_dim, out_dim);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            params["q_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_dim);
            params["v_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_dim);
        }

        // x: [in_dim, T, 1], mask: [T, T, 1, 1] additive causal mask.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* mask) {
            auto gctx = ctx->ggml_ctx;
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto proj     = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            const int64_t seq_len = x->ne[1];
            GGML_ASSERT(x->ne[0] == in_dim);
            GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1);

            auto qkv       = qkv_proj->forward(ctx, x);  // [3 * in_dim, T]
            auto zero_bias = ggml_ext_zeros(gctx, in_dim, 1, 1, 1);
            auto bias      = ggml_concat(gctx, params["q_bias"], zero_bias, 0);
            bias           = ggml_concat(gctx, bias, params["v_bias"], 0);
            bias           = ggml_reshape_4d(gctx, bias, in_dim * 3, 1, 1, 1);
            qkv            = ggml_add(gctx, qkv, bias);

            auto parts = ggml_ext_chunk(gctx, qkv, 3, 0);
            auto q     = ggml_reshape_3d(gctx, parts[0], in_dim, seq_len, 1);
            auto k     = ggml_reshape_3d(gctx, parts[1], in_dim, seq_len, 1);
            auto v     = ggml_reshape_3d(gctx, parts[2], in_dim, seq_len, 1);

            // flash_attn is forced off: the causal mask is supplied explicitly
            // so the result does not depend on which kernel the backend picks.
            auto out = ggml_ext_attention_ext(gctx, ctx->backend, q, k, v, num_heads, mask, false, false);
            // out: [in_dim, T, 1] with ne0 index = head * head_dim + d.

            // torch.mean(x, dim=heads)
            out = ggml_reshape_4d(gctx, out, head_dim, num_heads, seq_len, 1);
            out = ggml_ext_cont(gctx, ggml_permute(gctx, out, 1, 0, 2, 3));  // [num_heads, head_dim, T, 1]
            out = ggml_mean(gctx, out);                                      // [1, head_dim, T, 1]
            out = ggml_reshape_3d(gctx, out, head_dim, seq_len, 1);

            // F.adaptive_avg_pool1d(head_dim -> out_dim). With head_dim an exact
            // multiple of out_dim this is a non-overlapping box average, which is
            // what the released 2048/8/32 config gives (256 -> 32).
            GGML_ASSERT(head_dim % out_dim == 0);
            const int64_t pool = head_dim / out_dim;
            out                = ggml_reshape_4d(gctx, out, pool, out_dim, seq_len, 1);
            out                = ggml_mean(gctx, out);  // [1, out_dim, T, 1]
            out                = ggml_reshape_3d(gctx, out, out_dim, seq_len, 1);

            return proj->forward(ctx, out);
        }
    };

    // AttnProjection (comfy L252-267): rewires latent_dim -> latent_channels.
    struct AttnProjection : public GGMLBlock {
        AttnProjection(int64_t in_dim, int64_t out_dim, int num_heads, int64_t mlp_hidden_dim) {
            blocks["norm1"] = std::make_shared<LayerNorm>(in_dim);
            blocks["attn"]  = std::make_shared<CausalAttention>(in_dim, out_dim, num_heads);
            blocks["proj"]  = std::make_shared<Linear>(in_dim, out_dim);
            blocks["norm3"] = std::make_shared<LayerNorm>(in_dim);
            blocks["norm2"] = std::make_shared<LayerNorm>(out_dim);
            blocks["mlp"]   = std::make_shared<GeGluMlp>(out_dim, mlp_hidden_dim);
        }

        // x: [in_dim, T, 1] -> [out_dim, T, 1]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* mask) {
            auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto attn  = std::dynamic_pointer_cast<CausalAttention>(blocks["attn"]);
            auto proj  = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            auto norm3 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);
            auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto mlp   = std::dynamic_pointer_cast<GeGluMlp>(blocks["mlp"]);

            auto skip = proj->forward(ctx, norm3->forward(ctx, x));
            auto attended = attn->forward(ctx, norm1->forward(ctx, x), mask);
            auto h        = ggml_add(ctx->ggml_ctx, skip, attended);
            return ggml_add(ctx->ggml_ctx, h, mlp->forward(ctx, norm2->forward(ctx, h)));
        }
    };

    // ---------------------------------------------------------------------
    // BigVGAN decoder
    // ---------------------------------------------------------------------

    // AMPBlock1 (comfy L276-304).
    //
    // Near-duplicate of LTXV::AMPBlock1 but NOT reusable: the reference keeps a
    // single flat `activations` ModuleList and interleaves it (acts1 =
    // activations[::2], acts2 = activations[1::2]), so the checkpoint keys are
    // activations.0/2/4 and activations.1/3/5 -- LTX's fork names them
    // acts1.<i> / acts2.<i>.
    struct AMPBlock1 : public GGMLBlock {
        std::vector<int> dilation;

        AMPBlock1(int64_t channels, int kernel_size, const std::vector<int>& dilation, int act_ratio, int act_kernel_size)
            : dilation(dilation) {
            for (size_t j = 0; j < dilation.size(); ++j) {
                const int d = dilation[j];
                blocks["convs1." + std::to_string(j)] = std::make_shared<Conv1D>(channels,
                                                                                 channels,
                                                                                 kernel_size,
                                                                                 1,
                                                                                 (kernel_size * d - d) / 2,
                                                                                 d);
                blocks["convs2." + std::to_string(j)] = std::make_shared<Conv1D>(channels,
                                                                                 channels,
                                                                                 kernel_size,
                                                                                 1,
                                                                                 (kernel_size - 1) / 2,
                                                                                 1);
                blocks["activations." + std::to_string(2 * j)]     = std::make_shared<Activation1d>(channels, act_ratio, act_kernel_size);
                blocks["activations." + std::to_string(2 * j + 1)] = std::make_shared<Activation1d>(channels, act_ratio, act_kernel_size);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const AliasFreeFilters& filters) {
            for (size_t j = 0; j < dilation.size(); ++j) {
                auto act1  = std::dynamic_pointer_cast<Activation1d>(blocks["activations." + std::to_string(2 * j)]);
                auto act2  = std::dynamic_pointer_cast<Activation1d>(blocks["activations." + std::to_string(2 * j + 1)]);
                auto conv1 = std::dynamic_pointer_cast<Conv1D>(blocks["convs1." + std::to_string(j)]);
                auto conv2 = std::dynamic_pointer_cast<Conv1D>(blocks["convs2." + std::to_string(j)]);

                auto h = act1->forward(ctx, x, filters);
                h      = conv1->forward(ctx, h);
                h      = act2->forward(ctx, h, filters);
                h      = conv2->forward(ctx, h);
                x      = ggml_add(ctx->ggml_ctx, h, x);
            }
            return x;
        }
    };

    // BigVGAN (comfy L307-368). use_bias_at_final = false and
    // use_tanh_at_final = false, so conv_post is bias-less and the output is
    // clamped rather than tanh'd.
    struct BigVGANDecoder : public GGMLBlock {
        MiniMaxH3AudioVAEConfig config;

        explicit BigVGANDecoder(const MiniMaxH3AudioVAEConfig& config)
            : config(config) {
            blocks["conv_pre"] = std::make_shared<Conv1D>(config.latent_dim, config.decoder_dim, 7, 1, 3);

            const int num_upsamples = config.num_upsamples();
            const int num_kernels   = static_cast<int>(config.resblock_kernel_sizes.size());
            for (int i = 0; i < num_upsamples; ++i) {
                const int in_channels  = config.decoder_dim >> i;
                const int out_channels = config.decoder_dim >> (i + 1);
                const int kernel       = config.decoder_kernel_sizes[static_cast<size_t>(i)];
                const int rate         = config.decoder_rates[static_cast<size_t>(i)];
                // The reference wraps each upsampler in a one-element ModuleList,
                // so the checkpoint key really is ups.<i>.0.
                blocks["ups." + std::to_string(i) + ".0"] = std::make_shared<ConvTranspose1D>(in_channels,
                                                                                              out_channels,
                                                                                              kernel,
                                                                                              rate,
                                                                                              (kernel - rate) / 2);
                for (int j = 0; j < num_kernels; ++j) {
                    blocks["resblocks." + std::to_string(i * num_kernels + j)] =
                        std::make_shared<AMPBlock1>(out_channels,
                                                    config.resblock_kernel_sizes[static_cast<size_t>(j)],
                                                    config.resblock_dilation_sizes[static_cast<size_t>(j)],
                                                    config.activation_ratio,
                                                    config.activation_kernel_size);
                }
            }

            const int final_channels = config.decoder_final_channels();
            blocks["activation_post"] = std::make_shared<Activation1d>(final_channels,
                                                                       config.activation_ratio,
                                                                       config.activation_kernel_size);
            blocks["conv_post"]       = std::make_shared<Conv1D>(final_channels, 1, 7, 1, 3, 1, false);
        }

        // x: [T, latent_dim, 1] -> [T * upsample_factor, 1, 1]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const AliasFreeFilters& filters) {
            const int num_upsamples = config.num_upsamples();
            const int num_kernels   = static_cast<int>(config.resblock_kernel_sizes.size());

            x = std::dynamic_pointer_cast<Conv1D>(blocks["conv_pre"])->forward(ctx, x);

            for (int i = 0; i < num_upsamples; ++i) {
                auto up = std::dynamic_pointer_cast<ConvTranspose1D>(blocks["ups." + std::to_string(i) + ".0"]);
                x       = up->forward(ctx, x);

                ggml_tensor* sum = nullptr;
                for (int j = 0; j < num_kernels; ++j) {
                    auto resblock  = std::dynamic_pointer_cast<AMPBlock1>(blocks["resblocks." + std::to_string(i * num_kernels + j)]);
                    auto block_out = resblock->forward(ctx, x, filters);
                    sum            = sum == nullptr ? block_out : ggml_add(ctx->ggml_ctx, sum, block_out);
                }
                GGML_ASSERT(sum != nullptr);
                x = ggml_ext_scale(ctx->ggml_ctx, sum, 1.0f / static_cast<float>(num_kernels));
            }

            x = std::dynamic_pointer_cast<Activation1d>(blocks["activation_post"])->forward(ctx, x, filters);
            x = std::dynamic_pointer_cast<Conv1D>(blocks["conv_post"])->forward(ctx, x);
            return ggml_clamp(ctx->ggml_ctx, x, -1.0f, 1.0f);
        }
    };

    // ---------------------------------------------------------------------
    // Top-level VAE
    // ---------------------------------------------------------------------

    struct MiniMaxH3AudioVAE : public GGMLBlock {
        MiniMaxH3AudioVAEConfig config;

        explicit MiniMaxH3AudioVAE(const MiniMaxH3AudioVAEConfig& config)
            : config(config) {
            blocks["dec_in_proj"] = std::make_shared<Conv1D>(config.latent_channels, config.latent_dim, 1);
            blocks["decoder"]     = std::make_shared<BigVGANDecoder>(config);
            if (config.has_encoder) {
                blocks["encoder"]   = std::make_shared<Encoder>(config);
                blocks["pre_block"] = std::make_shared<AttnProjection>(config.latent_dim,
                                                                       config.latent_channels,
                                                                       config.num_attention_heads,
                                                                       config.mlp_hidden_dim);
                blocks["mean_proj"] = std::make_shared<Conv1D>(config.latent_channels, config.latent_channels, 1);
                // `logs_proj` is deliberately absent: the reference never
                // evaluates it at inference (encode returns the posterior mean,
                // comfy L405-407). Its checkpoint tensors are simply ignored.
            }
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
            if (config.has_latent_stats) {
                params["latents_mean"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.latent_channels);
                params["latents_std"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.latent_channels);
            }
        }

        // Prefer host-supplied statistics (diffusers keeps latents_mean /
        // latents_std in config.json, not in the weights); fall back to the
        // checkpoint buffers the comfy tree registers; otherwise skip the
        // normalization entirely.
        std::pair<ggml_tensor*, ggml_tensor*> resolve_stats(ggml_tensor* mean_input, ggml_tensor* std_input) {
            if (mean_input != nullptr && std_input != nullptr) {
                return {mean_input, std_input};
            }
            auto mean_iter = params.find("latents_mean");
            auto std_iter  = params.find("latents_std");
            if (mean_iter != params.end() && std_iter != params.end()) {
                return {mean_iter->second, std_iter->second};
            }
            return {nullptr, nullptr};
        }

        // z: [T, latent_channels, 1] (one stereo channel) -> [T * hop, 1, 1]
        ggml_tensor* decode_channel(GGMLRunnerContext* ctx,
                                    ggml_tensor* z,
                                    ggml_tensor* mean_input,
                                    ggml_tensor* std_input,
                                    const AliasFreeFilters& filters) {
            auto gctx  = ctx->ggml_ctx;
            auto stats = resolve_stats(mean_input, std_input);
            if (stats.first != nullptr) {
                auto mean   = ggml_reshape_4d(gctx, stats.first, 1, config.latent_channels, 1, 1);
                auto stddev = ggml_reshape_4d(gctx, stats.second, 1, config.latent_channels, 1, 1);
                z           = ggml_add(gctx, ggml_mul(gctx, z, stddev), mean);
            }
            auto x = std::dynamic_pointer_cast<Conv1D>(blocks["dec_in_proj"])->forward(ctx, z);
            return std::dynamic_pointer_cast<BigVGANDecoder>(blocks["decoder"])->forward(ctx, x, filters);
        }

        // waveform: [L, 1, 1] with L a multiple of hop_length -> [T, latent_channels, 1]
        ggml_tensor* encode_channel(GGMLRunnerContext* ctx,
                                    ggml_tensor* waveform,
                                    ggml_tensor* causal_mask,
                                    ggml_tensor* mean_input,
                                    ggml_tensor* std_input) {
            GGML_ASSERT(config.has_encoder);
            auto gctx = ctx->ggml_ctx;

            auto h = std::dynamic_pointer_cast<Encoder>(blocks["encoder"])->forward(ctx, waveform);
            // [T, latent_dim] -> [latent_dim, T]: the bottleneck is a token-wise
            // stack, so the feature axis has to be ne0.
            h = ggml_cont(gctx, ggml_transpose(gctx, h));
            h = std::dynamic_pointer_cast<AttnProjection>(blocks["pre_block"])->forward(ctx, h, causal_mask);
            h = ggml_cont(gctx, ggml_transpose(gctx, h));  // [T, latent_channels]

            auto z = std::dynamic_pointer_cast<Conv1D>(blocks["mean_proj"])->forward(ctx, h);

            auto stats = resolve_stats(mean_input, std_input);
            if (stats.first != nullptr) {
                auto mean   = ggml_reshape_4d(gctx, stats.first, 1, config.latent_channels, 1, 1);
                auto stddev = ggml_reshape_4d(gctx, stats.second, 1, config.latent_channels, 1, 1);
                z           = ggml_div(gctx, ggml_sub(gctx, z, mean), stddev);
            }
            return z;
        }
    };

    struct MiniMaxH3AudioVAERunner : public GGMLRunner {
        MiniMaxH3AudioVAEConfig config;
        MiniMaxH3AudioVAE model;
        std::string weight_prefix;

        // Kaiser filters, built once on the host (see kaiser_sinc_filter1d).
        sd::Tensor<float> upsample_filter_reversed;
        sd::Tensor<float> downsample_filter;
        // Optional config-supplied latent statistics.
        sd::Tensor<float> latents_mean_host;
        sd::Tensor<float> latents_std_host;

        MiniMaxH3AudioVAERunner(ggml_backend_t backend,
                                const String2TensorStorage& tensor_storage_map,
                                const std::string& prefix                           = "",
                                std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager),
              config(MiniMaxH3AudioVAEConfig::detect_from_weights(tensor_storage_map, prefix)),
              model(config),
              weight_prefix(prefix) {
            model.init(params_ctx, tensor_storage_map, prefix);

            const double cutoff     = 0.5 / static_cast<double>(config.activation_ratio);
            const double half_width = 0.6 / static_cast<double>(config.activation_ratio);
            auto up_filter          = kaiser_sinc_filter1d(cutoff, half_width, config.activation_kernel_size);
            auto down_filter        = kaiser_sinc_filter1d(cutoff, half_width, config.activation_kernel_size);
            std::reverse(up_filter.begin(), up_filter.end());
            upsample_filter_reversed = sd::Tensor<float>::from_vector(std::move(up_filter));
            downsample_filter        = sd::Tensor<float>::from_vector(std::move(down_filter));

            // Latent statistics. The released GGUFs carry no latents_mean/latents_std tensors
            // (the conversion script drops them; they live in audio_vae/config.json), so without
            // this block resolve_stats() returns {nullptr, nullptr} and BOTH encode() and decode()
            // silently run UNNORMALIZED. A round trip stays self-consistent that way, which is
            // exactly why it never showed up -- but the DiT emits NORMALIZED latents, so the
            // render path fed the BigVGAN decoder inputs ~1.5-3.3x too small with a per-channel
            // offset. Mirrors MiniMaxH3VideoVAERunner's constructor, which has always done this.
            //
            // MINIMAX_H3_AUDIO_LATENT_STATS=0 restores the old (broken) behaviour for a one-env-var
            // A/B without a rebuild.
            const char* stats_env = getenv("MINIMAX_H3_AUDIO_LATENT_STATS");
            const bool stats_on   = stats_env == nullptr || (stats_env[0] != '0' && stats_env[0] != '\0');
            if (!config.has_latent_stats && stats_on) {
                if (config.latent_channels == 32) {
                    latents_mean_host = sd::Tensor<float>::from_vector(
                        std::vector<float>(LATENTS_MEAN, LATENTS_MEAN + 32));
                    latents_std_host = sd::Tensor<float>::from_vector(
                        std::vector<float>(LATENTS_STD, LATENTS_STD + 32));
                    LOG_INFO("minimax_h3_audio_vae: checkpoint carries no latents_mean/latents_std; "
                             "using the released audio_vae/config.json literals (32 channels)");
                } else {
                    LOG_WARN("minimax_h3_audio_vae: %d latent channels but the reference statistics "
                             "cover 32; latents will NOT be de-normalized -- call set_latent_stats()",
                             config.latent_channels);
                }
            } else if (!config.has_latent_stats) {
                LOG_WARN("minimax_h3_audio_vae: MINIMAX_H3_AUDIO_LATENT_STATS=0, latent "
                         "normalization DISABLED (decode will see normalized latents)");
            }
        }

        // For checkpoints that keep latents_mean / latents_std outside the
        // weights (the diffusers export puts them in config.json). Supplying
        // them here overrides any checkpoint buffers.
        void set_latent_stats(const std::vector<float>& mean, const std::vector<float>& stddev) {
            if (static_cast<int>(mean.size()) != config.latent_channels ||
                static_cast<int>(stddev.size()) != config.latent_channels) {
                LOG_ERROR("minimax_h3_audio_vae: latent statistics must have %d entries, got %zu/%zu",
                          config.latent_channels,
                          mean.size(),
                          stddev.size());
                return;
            }
            latents_mean_host = sd::Tensor<float>::from_vector(mean);
            latents_std_host  = sd::Tensor<float>::from_vector(stddev);
        }

        bool has_latent_stats() const {
            return config.has_latent_stats || !latents_std_host.empty();
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
            model.get_param_tensors(tensors, weight_prefix);
        }

        size_t get_params_mem_size() {
            return model.get_params_mem_size();
        }

        std::string get_desc() {
            return "minimax_h3_audio_vae";
        }

        // ------------------------------------------------------------------
        // Planar <-> interleaved helpers. The engine boundary is planar; every
        // container format is interleaved. Converting in one place is the only
        // reason this port cannot silently swap the stereo field.
        // ------------------------------------------------------------------

        static sd::Tensor<float> interleaved_to_planar(const float* interleaved, int64_t samples, int64_t channels) {
            sd::Tensor<float> planar({samples, channels});
            float* out = planar.data();
            for (int64_t channel = 0; channel < channels; ++channel) {
                for (int64_t sample = 0; sample < samples; ++sample) {
                    out[channel * samples + sample] = interleaved[sample * channels + channel];
                }
            }
            return planar;
        }

        // ------------------------------------------------------------------
        // Debug artefact writers.
        //
        // These exist so the audio path can be JUDGED BY EAR and by numpy instead of by
        // summary statistics: every H3 audio A/B so far moved L/R correlation and spectral
        // balance without the result actually sounding right, so the deliverable has to be a
        // playable file plus the raw latent behind it.
        // ------------------------------------------------------------------

        // 32-bit float WAV (format 3). Lossless for a [-1, 1] engine buffer -- a PCM16 write
        // would quantize the thing under test. `samples` frames, `channels` interleaved.
        static bool write_wav_f32(const std::string& path,
                                  const float* interleaved,
                                  int64_t samples,
                                  int64_t channels,
                                  int sample_rate) {
            std::ofstream out(path, std::ios::binary);
            if (!out.is_open()) {
                LOG_ERROR("minimax_h3_audio_vae: cannot write '%s'", path.c_str());
                return false;
            }
            const uint32_t data_bytes  = static_cast<uint32_t>(samples * channels * 4);
            const uint32_t riff_bytes  = 36 + data_bytes;
            const uint16_t fmt_tag     = 3;  // IEEE float
            const uint16_t n_channels  = static_cast<uint16_t>(channels);
            const uint32_t rate        = static_cast<uint32_t>(sample_rate);
            const uint32_t byte_rate   = rate * n_channels * 4;
            const uint16_t block_align = static_cast<uint16_t>(n_channels * 4);
            const uint16_t bits        = 32;
            const uint32_t fmt_bytes   = 16;
            auto w32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
            auto w16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
            out.write("RIFF", 4);
            w32(riff_bytes);
            out.write("WAVE", 4);
            out.write("fmt ", 4);
            w32(fmt_bytes);
            w16(fmt_tag);
            w16(n_channels);
            w32(rate);
            w32(byte_rate);
            w16(block_align);
            w16(bits);
            out.write("data", 4);
            w32(data_bytes);
            out.write(reinterpret_cast<const char*>(interleaved), static_cast<std::streamsize>(data_bytes));
            return out.good();
        }

        // NPY v1.0, '<f4', C-order. sd::Tensor is ne0-FASTEST (fortran-like), so the numpy
        // shape written here is the REVERSE of tensor.shape() -- i.e. a {T, C, S} latent lands
        // in numpy as (S, C, T), which is exactly torch's layout in the reference. Load with
        // `numpy.load(path)`; no header parsing on your side.
        static bool write_npy_f32(const std::string& path,
                                  const float* data,
                                  const std::vector<int64_t>& tensor_shape) {
            std::ofstream out(path, std::ios::binary);
            if (!out.is_open()) {
                LOG_ERROR("minimax_h3_audio_vae: cannot write '%s'", path.c_str());
                return false;
            }
            std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
            int64_t count    = 1;
            for (size_t i = tensor_shape.size(); i-- > 0;) {
                dict += std::to_string(tensor_shape[i]);
                dict += ", ";
                count *= tensor_shape[i];
            }
            dict += "), }";
            // Header (10 magic+version+len bytes + dict + '\n') must be a multiple of 64.
            size_t header_len = dict.size() + 1;
            while ((10 + header_len) % 64 != 0) {
                dict += ' ';
                header_len = dict.size() + 1;
            }
            dict += '\n';
            const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
            const uint16_t len16         = static_cast<uint16_t>(dict.size());
            out.write(reinterpret_cast<const char*>(magic), 8);
            out.write(reinterpret_cast<const char*>(&len16), 2);
            out.write(dict.data(), static_cast<std::streamsize>(dict.size()));
            out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * 4));
            return out.good();
        }

        // Per-latent-channel mean / std of a {T, C, S} latent, logged as a compact table.
        //
        // This is the single number that tells NORMALIZED latents apart from RAW ones: the
        // encoder's output after (z - mean) / std is ~N(0, 1) per channel, while the same
        // latent before normalization has the per-channel std of LATENTS_STD (1.5-3.3). Run it
        // on the DiT's generated latent and on this encoder's and the two must AGREE.
        static void log_latent_stats(const char* tag, const sd::Tensor<float>& latent) {
            if (latent.empty() || latent.dim() < 2) {
                return;
            }
            const int64_t t        = latent.shape()[0];
            const int64_t channels = latent.shape()[1];
            const int64_t stereo   = latent.dim() > 2 ? latent.shape()[2] : 1;
            const float* d         = latent.data();
            double all_mean = 0.0, all_sq = 0.0;
            std::string mean_row, std_row;
            for (int64_t c = 0; c < channels; ++c) {
                double sum = 0.0, sq = 0.0;
                for (int64_t s = 0; s < stereo; ++s) {
                    for (int64_t i = 0; i < t; ++i) {
                        const double v = d[(s * channels + c) * t + i];
                        sum += v;
                        sq += v * v;
                    }
                }
                const double n  = static_cast<double>(t * stereo);
                const double mu = sum / std::max(1.0, n);
                const double sd = std::sqrt(std::max(0.0, sq / std::max(1.0, n) - mu * mu));
                all_mean += mu;
                all_sq += sd;
                char buf[32];
                snprintf(buf, sizeof(buf), "%+.3f ", mu);
                mean_row += buf;
                snprintf(buf, sizeof(buf), "%.3f ", sd);
                std_row += buf;
            }
            LOG_INFO("%s latent {T=%lld, C=%lld, S=%lld}  mean(avg)=%+.4f  std(avg)=%.4f",
                     tag,
                     (long long)t,
                     (long long)channels,
                     (long long)stereo,
                     all_mean / std::max<int64_t>(1, channels),
                     all_sq / std::max<int64_t>(1, channels));
            LOG_INFO("%s per-channel mean: %s", tag, mean_row.c_str());
            LOG_INFO("%s per-channel std : %s", tag, std_row.c_str());
        }

        static std::vector<float> planar_to_interleaved(const sd::Tensor<float>& planar) {
            GGML_ASSERT(planar.dim() == 2);
            const int64_t samples  = planar.shape()[0];
            const int64_t channels = planar.shape()[1];
            std::vector<float> interleaved(static_cast<size_t>(samples * channels));
            const float* in = planar.data();
            for (int64_t channel = 0; channel < channels; ++channel) {
                for (int64_t sample = 0; sample < samples; ++sample) {
                    interleaved[static_cast<size_t>(sample * channels + channel)] = in[channel * samples + sample];
                }
            }
            return interleaved;
        }

        // Decode normalized latents into a planar stereo waveform.
        //   latent:  {T, latent_channels, S}  (S = 1 or 2; see the header comment)
        //   returns: {T * 800, S} at 32 kHz, clamped to [-1, 1]
        sd::Tensor<float> decode(int n_threads, const sd::Tensor<float>& latent_input) {
            // The render path hands this a TRAILING-SINGLETON tensor: minimax_h3_unpack_audio_latent
            // builds {T, S, C, 1} and minimax_h3_swap_audio_axes keeps the fourth axis when it
            // permutes to {T, C, S, 1}. Refusing that is what made every H3 render silently
            // video-only -- the caller only logs a WARN and carries on. A trailing 1 is not a
            // fourth axis, so squeeze it rather than reject it.
            sd::Tensor<float> latent_tensor = latent_input;
            while (!latent_tensor.empty() && latent_tensor.dim() > 3 && latent_tensor.shape().back() == 1) {
                std::vector<int64_t> squeezed(latent_tensor.shape().begin(), latent_tensor.shape().end() - 1);
                latent_tensor = latent_tensor.reshape(std::move(squeezed));
            }
            if (latent_tensor.empty() || latent_tensor.dim() < 2 || latent_tensor.dim() > 3) {
                LOG_ERROR("minimax_h3_audio_vae: decode expects a {frames, latent_channels[, stereo]} tensor");
                return {};
            }
            if (latent_tensor.shape()[1] != config.latent_channels) {
                LOG_ERROR("minimax_h3_audio_vae: decode expects %d latent channels, got %lld",
                          config.latent_channels,
                          static_cast<long long>(latent_tensor.shape()[1]));
                return {};
            }
            const int64_t stereo = latent_tensor.dim() == 3 ? latent_tensor.shape()[2] : 1;

            const int64_t start = ggml_time_ms();
            auto get_graph      = [&]() -> ggml_cgraph* {
                auto latent      = make_input(latent_tensor);
                AliasFreeFilters filters;
                filters.upsample_reversed = make_input(upsample_filter_reversed);
                filters.downsample        = make_input(downsample_filter);
                ggml_tensor* mean_input   = make_optional_input(latents_mean_host);
                ggml_tensor* std_input    = make_optional_input(latents_std_host);

                ggml_cgraph* gf      = new_graph_custom(655360);
                auto runner_ctx      = GGMLRunner::get_context();
                ggml_tensor* stacked = nullptr;
                for (int64_t channel = 0; channel < stereo; ++channel) {
                    auto z = latent_tensor.dim() == 3
                                 ? ggml_ext_slice(runner_ctx.ggml_ctx, latent, 2, channel, channel + 1)
                                 : latent;
                    z      = ggml_reshape_3d(runner_ctx.ggml_ctx, z, z->ne[0], z->ne[1], 1);
                    auto waveform = model.decode_channel(&runner_ctx, z, mean_input, std_input, filters);
                    // ggml_clamp hands back a view; make it a real buffer before
                    // it becomes one half of a concat.
                    waveform      = ggml_cont(runner_ctx.ggml_ctx, waveform);
                    waveform      = ggml_reshape_4d(runner_ctx.ggml_ctx, waveform, waveform->ne[0], 1, 1, 1);
                    stacked       = stacked == nullptr ? waveform
                                                       : ggml_concat(runner_ctx.ggml_ctx, stacked, waveform, 1);
                }
                ggml_build_forward_expand(gf, stacked);
                return gf;
            };
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), 2);
            LOG_INFO("minimax h3 audio vae decode completed, taking %.2fs", (ggml_time_ms() - start) * 1.0f / 1000);
            return result;
        }

        // Encode a planar 32 kHz waveform into normalized latents.
        //   waveform: {L, S} in [-1, 1] (S = 1 or 2); L is right-padded with
        //             zeros to a multiple of 800, exactly as the reference does.
        //   returns:  {ceil(L / 800), latent_channels, S}
        sd::Tensor<float> encode(int n_threads, const sd::Tensor<float>& waveform) {
            if (!config.has_encoder) {
                LOG_ERROR("minimax_h3_audio_vae: encoder weights are unavailable");
                return {};
            }
            if (waveform.empty() || waveform.dim() < 1 || waveform.dim() > 2) {
                LOG_ERROR("minimax_h3_audio_vae: encode expects a {samples[, stereo]} tensor");
                return {};
            }
            const int64_t samples = waveform.shape()[0];
            const int64_t stereo  = waveform.dim() == 2 ? waveform.shape()[1] : 1;
            const int64_t hop     = config.hop_length();
            if (samples <= 0) {
                LOG_ERROR("minimax_h3_audio_vae: encode requires a non-empty waveform");
                return {};
            }

            const int64_t frames         = (samples + hop - 1) / hop;
            const int64_t padded_samples = frames * hop;
            sd::Tensor<float> padded({padded_samples, stereo});
            for (int64_t channel = 0; channel < stereo; ++channel) {
                std::memcpy(padded.data() + channel * padded_samples,
                            waveform.data() + channel * samples,
                            static_cast<size_t>(samples) * sizeof(float));
                std::memset(padded.data() + channel * padded_samples + samples,
                            0,
                            static_cast<size_t>(padded_samples - samples) * sizeof(float));
            }

            // Additive causal mask, [key, query]: 0 where key <= query, -inf
            // above the diagonal. Materialized because ggml_ext_attention_ext
            // has no is_causal flag; O(T^2) floats, ~2.6 MB for 10 s of audio.
            sd::Tensor<float> causal_mask({frames, frames});
            for (int64_t query = 0; query < frames; ++query) {
                for (int64_t key = 0; key < frames; ++key) {
                    causal_mask.data()[query * frames + key] = key <= query ? 0.0f : -INFINITY;
                }
            }

            const int64_t start = ggml_time_ms();
            auto get_graph      = [&]() -> ggml_cgraph* {
                auto padded_input       = make_input(padded);
                auto mask               = make_input(causal_mask);
                ggml_tensor* mean_input = make_optional_input(latents_mean_host);
                ggml_tensor* std_input  = make_optional_input(latents_std_host);

                ggml_cgraph* gf      = new_graph_custom(655360);
                auto runner_ctx      = GGMLRunner::get_context();
                ggml_tensor* stacked = nullptr;
                for (int64_t channel = 0; channel < stereo; ++channel) {
                    auto mono = stereo > 1
                                    ? ggml_ext_slice(runner_ctx.ggml_ctx, padded_input, 1, channel, channel + 1)
                                    : padded_input;
                    mono      = ggml_reshape_3d(runner_ctx.ggml_ctx, mono, padded_samples, 1, 1);
                    auto z    = model.encode_channel(&runner_ctx, mono, mask, mean_input, std_input);
                    z         = ggml_reshape_4d(runner_ctx.ggml_ctx, z, z->ne[0], z->ne[1], 1, 1);
                    stacked   = stacked == nullptr ? z : ggml_concat(runner_ctx.ggml_ctx, stacked, z, 2);
                }
                ggml_build_forward_expand(gf, stacked);
                return gf;
            };
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), 3);
            LOG_INFO("minimax h3 audio vae encode completed, taking %.2fs", (ggml_time_ms() - start) * 1.0f / 1000);
            return result;
        }

        // ------------------------------------------------------------------
        // ENCODE -> DECODE round trip on a known-good waveform, with the DiT, the packed
        // sequence and the muxer all OUT of the loop.
        //
        // Why this exists: encode() was only ever reachable through a ref2va reference and
        // decode() only at the end of a render, so the two were never composed and the VAE was
        // never testable on its own. Every H3 audio finding so far is end-to-end, which cannot
        // separate "the VAE is wrong" from "the latents handed to it are wrong".
        //
        // READ THE RESULT LIKE THIS:
        //   * <prefix>.roundtrip.wav sounds like <prefix>.input.wav  ->  the VAE is FINE and the
        //     defect is in the DiT's audio latents (or in the layout/normalization around them).
        //   * <prefix>.roundtrip.wav is broken                       ->  the VAE (or its weight
        //     fusion / stereo split / filters) is the defect and the DiT is exonerated.
        // Note the round trip is INVARIANT to the latents_mean/std wiring: encode normalizes
        // and decode de-normalizes with the same numbers, so it stays clean either way. Use
        // <prefix>.latent.npy and the per-channel std in the log to judge THAT: with the stats
        // applied the encoder's per-channel std is ~1, without them it is ~1.5-3.3.
        //
        // planar_in: {samples, stereo} at output_sample_rate(), in [-1, 1]. The caller owns
        // resampling and channel duplication -- see encode_minimax_h3_reference_audio().
        bool roundtrip(int n_threads, const sd::Tensor<float>& planar_in, const std::string& out_prefix) {
            if (!config.has_encoder) {
                LOG_ERROR("minimax_h3_audio_vae: round trip needs the encoder half; this checkpoint is decode-only");
                return false;
            }
            if (planar_in.empty() || planar_in.dim() != 2) {
                LOG_ERROR("minimax_h3_audio_vae: round trip expects a {samples, stereo} planar waveform");
                return false;
            }
            const int64_t in_samples  = planar_in.shape()[0];
            const int64_t in_channels = planar_in.shape()[1];
            const int rate            = config.output_sample_rate();

            {
                auto interleaved = planar_to_interleaved(planar_in);
                const std::string p = out_prefix + ".input.wav";
                if (write_wav_f32(p, interleaved.data(), in_samples, in_channels, rate)) {
                    LOG_INFO("minimax_h3_audio_vae: wrote %s (%lld frames, %lld ch, %d Hz)",
                             p.c_str(), (long long)in_samples, (long long)in_channels, rate);
                }
            }

            auto latent = encode(n_threads, planar_in);
            if (latent.empty()) {
                LOG_ERROR("minimax_h3_audio_vae: round trip encode failed");
                return false;
            }
            log_latent_stats("minimax_h3_audio_vae round-trip ENCODED", latent);
            {
                const std::string p = out_prefix + ".latent.npy";
                if (write_npy_f32(p, latent.data(), latent.shape())) {
                    LOG_INFO("minimax_h3_audio_vae: wrote %s (numpy shape is the REVERSE of the "
                             "engine shape, i.e. (S, C, T))",
                             p.c_str());
                }
            }

            auto out = decode(n_threads, latent);
            if (out.empty()) {
                LOG_ERROR("minimax_h3_audio_vae: round trip decode failed");
                return false;
            }
            const int64_t out_samples  = out.shape()[0];
            const int64_t out_channels = out.dim() > 1 ? out.shape()[1] : 1;
            {
                auto interleaved    = planar_to_interleaved(out);
                const std::string p = out_prefix + ".roundtrip.wav";
                if (!write_wav_f32(p, interleaved.data(), out_samples, out_channels, rate)) {
                    return false;
                }
                LOG_INFO("minimax_h3_audio_vae: wrote %s (%lld frames, %lld ch, %d Hz)  <-- LISTEN TO THIS",
                         p.c_str(), (long long)out_samples, (long long)out_channels, rate);
            }

            // Numbers are the SECOND opinion here, never the verdict -- but a per-channel
            // correlation with a best-lag search does separate "reconstructed with a delay"
            // from "unrelated signal", which is the one thing ears are bad at.
            const int64_t n = std::min(in_samples, out_samples);
            for (int64_t c = 0; c < std::min(in_channels, out_channels); ++c) {
                const float* a = planar_in.data() + c * in_samples;
                const float* b = out.data() + c * out_samples;
                double best    = -2.0;
                int64_t best_lag = 0;
                for (int64_t lag = -800; lag <= 800; lag += 8) {
                    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                    int64_t cnt = 0;
                    for (int64_t i = std::max<int64_t>(0, -lag); i < n && i + lag < out_samples; ++i) {
                        const double x = a[i];
                        const double y = b[i + lag];
                        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
                        ++cnt;
                    }
                    if (cnt < 1024) {
                        continue;
                    }
                    const double dn = static_cast<double>(cnt);
                    const double cov = sab / dn - (sa / dn) * (sb / dn);
                    const double va  = saa / dn - (sa / dn) * (sa / dn);
                    const double vb  = sbb / dn - (sb / dn) * (sb / dn);
                    const double r   = (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : 0.0;
                    if (r > best) {
                        best     = r;
                        best_lag = lag;
                    }
                }
                double err = 0, sig = 0;
                for (int64_t i = 0; i < n; ++i) {
                    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                    err += d * d;
                    sig += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                }
                LOG_INFO("minimax_h3_audio_vae round trip ch%lld: corr %.4f @ lag %lld, "
                         "SNR(lag 0) %.2f dB",
                         (long long)c,
                         best,
                         (long long)best_lag,
                         10.0 * std::log10(std::max(1e-20, sig) / std::max(1e-20, err)));
            }
            return true;
        }

        void test(const std::string& input_path) {
            auto z = sd::load_tensor_from_file_as_tensor<float>(input_path);
            GGML_ASSERT(!z.empty());
            print_sd_tensor(z, false, "minimax_h3_audio_vae_z");

            auto out = decode(8, z);
            GGML_ASSERT(!out.empty());
            print_sd_tensor(out, false, "minimax_h3_audio_vae_out");
        }

        static void load_from_file_and_test(const std::string& model_path,
                                            const std::string& input_path,
                                            const std::string& prefix = "") {
            ggml_backend_t backend = sd_backend_cpu_init();
            LOG_INFO("loading minimax h3 audio vae from '%s'", model_path.c_str());

            auto model_manager        = std::make_shared<ModelManager>();
            ModelLoader& model_loader = model_manager->loader();
            if (!model_loader.init_from_file(model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            auto audio_vae           = std::make_shared<MiniMaxH3AudioVAERunner>(backend,
                                                                       tensor_storage_map,
                                                                       prefix,
                                                                       model_manager);

            if (!model_manager->register_runner_params("MiniMax H3 audio VAE test",
                                                       *audio_vae,
                                                       ModelManager::ResidencyMode::ParamBackend,
                                                       backend,
                                                       backend) ||
                !model_manager->validate_registered_tensors()) {
                LOG_ERROR("register minimax h3 audio vae tensors with model manager failed");
                return;
            }

            LOG_INFO("minimax h3 audio vae model loaded");
            audio_vae->test(input_path);
        }
    };

}  // namespace MiniMaxH3Audio

#endif  // __SD_MODEL_VAE_MINIMAX_H3_AUDIO_VAE_HPP__
