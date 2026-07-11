#ifndef __SD_MODEL_DIFFUSION_LTXV_HPP__
#define __SD_MODEL_DIFFUSION_LTXV_HPP__

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "model/common/block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace LTXV {

    constexpr int LTXAV_GRAPH_SIZE = 102400;

    __STATIC_INLINE__ ggml_tensor* rms_norm(ggml_context* ctx,
                                            ggml_tensor* x,
                                            float eps = 1e-6f) {
        return ggml_rms_norm(ctx, x, eps);
    }

    __STATIC_INLINE__ ggml_tensor* align_token_modulation(ggml_context* ctx,
                                                          ggml_tensor* x,
                                                          ggml_tensor* mod) {
        if (mod != nullptr && x != nullptr && mod->ne[1] == 1 && mod->ne[2] == x->ne[1] && x->ne[2] == 1) {
            return ggml_permute(ctx, mod, 0, 2, 1, 3);
        }
        return mod;
    }

    // Modulation token-collapse gather (VRAM win; see LTXAVRunner::build_graph). The
    // per-token AdaLN timestep has only a few UNIQUE values (clean-anchor t=0 vs noise t,
    // plus any graded-overlap levels), so the blocks compute modulation on a COMPACT
    // [dim,coeff,U] table instead of the full [dim,coeff,L_token]; this expands one compact
    // chunk [dim,1,U] back to per-token [dim,1,L] with a pure int32 row-gather (bit-exact —
    // no arithmetic, just copies the already-computed unique columns into token order).
    __STATIC_INLINE__ ggml_tensor* gather_mod_tokens(ggml_context* ctx,
                                                     ggml_tensor* c,     // [dim, 1, U]
                                                     ggml_tensor* sel) { // I32 [L]
        if (sel == nullptr) {
            return c;
        }
        int64_t dim = c->ne[0];
        int64_t U   = c->ne[2];
        auto a = ggml_reshape_2d(ctx, ggml_cont(ctx, c), dim, U);  // [dim, U]
        auto g = ggml_get_rows(ctx, a, sel);                       // [dim, L]
        return ggml_reshape_3d(ctx, g, dim, 1, sel->ne[0]);        // [dim, 1, L]
    }

    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* shift,
                                            ggml_tensor* scale) {
        shift = align_token_modulation(ctx, x, shift);
        scale = align_token_modulation(ctx, x, scale);
        return Flux::modulate(ctx, x, shift, scale, true);
    }

    __STATIC_INLINE__ bool ltx_rms_mod_fuse_enabled() {
        static int v = -1;
        if (v < 0) {
            const char* e = getenv("GGML_CUDA_RMS_MOD_FUSE");
            v = (e && atoi(e)) ? 1 : 0;
        }
        return v == 1;
    }

    // Fused rms_norm + modulate as ONE custom CUDA op: rms(x)*(1+scale)+shift.
    // x_prenorm stays PRE-rms (the op normalizes internally over ne[0]); scale/shift are
    // aligned (permuted) exactly like modulate() does, so the permute guard fires identically.
    // The +1 of (1+scale) is intrinsic to the op, so RAW scale is passed (NOT scale+1).
    // Uses the SAME eps as rms_norm() above (1e-6f) so quality matches the unfused chain.
    __STATIC_INLINE__ ggml_tensor* modulate_fused(ggml_context* ctx,
                                                  ggml_tensor* x_prenorm,
                                                  ggml_tensor* shift,
                                                  ggml_tensor* scale) {
        shift = align_token_modulation(ctx, x_prenorm, shift);
        scale = align_token_modulation(ctx, x_prenorm, scale);
        return ggml_rms_modulate(ctx, x_prenorm, scale, shift, 1e-6f);
    }

    __STATIC_INLINE__ ggml_tensor* apply_gate(ggml_context* ctx,
                                              ggml_tensor* x,
                                              ggml_tensor* gate) {
        gate = align_token_modulation(ctx, x, gate);
        return ggml_mul(ctx, x, gate);
    }

    // The scale_shift modulation tables become the broadcast src1 of ggml_mul/ggml_add in
    // modulate()/apply_gate(); the ggml-cuda binbcast kernel only accepts F32/F16 there. Classic
    // LTX-Video 0.9.x f16 checkpoints store these tables as BF16, so upcast to F32 (no-op for the
    // F32/F16 tables of LTX-2, keeping that path byte-identical).
    __STATIC_INLINE__ ggml_tensor* cast_modulation_table(ggml_context* ctx, ggml_tensor* table) {
        if (table != nullptr && table->type != GGML_TYPE_F32 && table->type != GGML_TYPE_F16) {
            return ggml_cast(ctx, table, GGML_TYPE_F32);
        }
        return table;
    }

    __STATIC_INLINE__ int count_prefix_blocks(const String2TensorStorage& tensor_storage_map,
                                              const std::string& prefix,
                                              const std::string& marker) {
        int max_block = -1;
        for (const auto& [name, _] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }
            size_t pos = name.find(marker);
            if (pos == std::string::npos) {
                continue;
            }
            pos += marker.size();
            size_t end = name.find(".", pos);
            if (end == std::string::npos) {
                continue;
            }
            int block = atoi(name.substr(pos, end - pos).c_str());
            max_block = std::max(max_block, block);
        }
        return max_block + 1;
    }

    struct LTXAVConfig {
        int64_t in_channels                           = 128;
        int64_t out_channels                          = 128;
        int64_t hidden_size                           = 3840;
        int64_t cross_attention_dim                   = 4096;
        int64_t caption_channels                      = 3840;
        int64_t num_attention_heads                   = 30;
        int64_t attention_head_dim                    = 128;
        int64_t num_layers                            = 28;
        float positional_embedding_theta              = 10000.f;
        std::vector<int> positional_embedding_max_pos = {20, 2048, 2048};
        std::tuple<int, int, int> vae_scale_factors   = {8, 32, 32};
        bool causal_temporal_positioning              = true;
        float timestep_scale_multiplier               = 1000.f;

        int64_t audio_in_channels                           = 128;
        int64_t audio_out_channels                          = 128;
        int64_t audio_hidden_size                           = 2048;
        int64_t audio_cross_attention_dim                   = 2048;
        int64_t audio_num_attention_heads                   = 32;
        int64_t audio_attention_head_dim                    = 64;
        std::vector<int> audio_positional_embedding_max_pos = {20};
        float av_ca_timestep_scale_multiplier               = 1000.f;
        int64_t num_audio_channels                          = 8;
        int64_t audio_frequency_bins                        = 16;

        // Video-only (classic LTX-Video 0.9.x) vs audio+video (LTX-2) checkpoint. Auto-detected
        // from the presence of audio_patchify_proj. When false, all audio / audio<->video-cross
        // sub-blocks and their scale_shift tables are NOT constructed (so the loader does not
        // hard-fail on the absent audio tensors), and the dual v+a context split is disabled.
        bool has_audio = true;

        bool use_connector                   = false;
        int64_t connector_hidden_size        = 3840;
        int64_t connector_num_heads          = 30;
        int64_t connector_head_dim           = 128;
        int64_t connector_num_layers         = 2;
        int64_t connector_num_registers      = 128;
        bool connector_rope_interleaved      = false;
        bool connector_apply_gated_attention = false;

        bool use_audio_connector                   = false;
        int64_t audio_connector_hidden_size        = 2048;
        int64_t audio_connector_num_heads          = 32;
        int64_t audio_connector_head_dim           = 64;
        int64_t audio_connector_num_layers         = 2;
        int64_t audio_connector_num_registers      = 128;
        bool audio_connector_rope_interleaved      = false;
        bool audio_connector_apply_gated_attention = false;

        bool video_rope_interleaved  = false;
        bool use_middle_indices_grid = true;
        bool cross_attention_adaln   = false;

        bool use_caption_projection          = true;
        bool use_audio_caption_projection    = true;
        bool caption_proj_before_connector   = true;
        bool caption_projection_first_linear = false;

        bool self_attention_gated  = false;
        bool cross_attention_gated = false;

        static std::pair<int64_t, int64_t> infer_attention_layout(int64_t hidden_size,
                                                                  int64_t preferred_heads = -1) {
            if (preferred_heads > 0 && hidden_size % preferred_heads == 0) {
                return {preferred_heads, hidden_size / preferred_heads};
            }
            const int candidates[] = {128, 96, 80, 64, 48, 40, 32};
            for (int head_dim : candidates) {
                if (hidden_size % head_dim == 0) {
                    int64_t heads = hidden_size / head_dim;
                    if (heads >= 8 && heads <= 64) {
                        return {heads, head_dim};
                    }
                }
            }
            return {32, hidden_size / 32};
        }

        static int64_t infer_gate_heads(const String2TensorStorage& tensor_storage_map,
                                        const std::string& bias_name,
                                        int64_t fallback_heads) {
            auto it = tensor_storage_map.find(bias_name);
            if (it != tensor_storage_map.end()) {
                return it->second.ne[0];
            }
            return fallback_heads;
        }

        static LTXAVConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            LTXAVConfig config;
            auto patchify_proj_iter = tensor_storage_map.find(prefix + ".patchify_proj.weight");
            if (patchify_proj_iter != tensor_storage_map.end()) {
                config.in_channels         = patchify_proj_iter->second.ne[0];
                config.hidden_size         = patchify_proj_iter->second.ne[1];
                int64_t video_heads        = infer_gate_heads(tensor_storage_map, prefix + ".transformer_blocks.0.attn1.to_gate_logits.bias", 32);
                auto attn_layout           = infer_attention_layout(config.hidden_size, video_heads);
                config.num_attention_heads = attn_layout.first;
                config.attention_head_dim  = attn_layout.second;
            }

            auto audio_patchify_proj_iter = tensor_storage_map.find(prefix + ".audio_patchify_proj.weight");
            config.has_audio              = (audio_patchify_proj_iter != tensor_storage_map.end());
            if (audio_patchify_proj_iter != tensor_storage_map.end()) {
                config.audio_in_channels         = audio_patchify_proj_iter->second.ne[0];
                config.audio_hidden_size         = audio_patchify_proj_iter->second.ne[1];
                config.audio_out_channels        = config.audio_in_channels;
                int64_t audio_heads              = infer_gate_heads(tensor_storage_map, prefix + ".transformer_blocks.0.audio_attn1.to_gate_logits.bias", 32);
                auto audio_attn_layout           = infer_attention_layout(config.audio_hidden_size, audio_heads);
                config.audio_num_attention_heads = audio_attn_layout.first;
                config.audio_attention_head_dim  = audio_attn_layout.second;
            }

            auto proj_out_iter = tensor_storage_map.find(prefix + ".proj_out.weight");
            if (proj_out_iter != tensor_storage_map.end()) {
                config.out_channels = proj_out_iter->second.ne[1];
            }
            auto audio_proj_out_iter = tensor_storage_map.find(prefix + ".audio_proj_out.weight");
            if (audio_proj_out_iter != tensor_storage_map.end()) {
                config.audio_out_channels = audio_proj_out_iter->second.ne[1];
            }

            auto attn2_iter = tensor_storage_map.find(prefix + ".transformer_blocks.0.attn2.to_k.weight");
            if (attn2_iter != tensor_storage_map.end()) {
                config.cross_attention_dim = attn2_iter->second.ne[0];
            }
            auto audio_attn2_iter = tensor_storage_map.find(prefix + ".transformer_blocks.0.audio_attn2.to_k.weight");
            if (audio_attn2_iter != tensor_storage_map.end()) {
                config.audio_cross_attention_dim = audio_attn2_iter->second.ne[0];
            }
            if (tensor_storage_map.find(prefix + ".transformer_blocks.0.prompt_scale_shift_table") != tensor_storage_map.end()) {
                config.cross_attention_adaln = true;
            }
            if (tensor_storage_map.find(prefix + ".transformer_blocks.0.attn1.to_gate_logits.weight") != tensor_storage_map.end() ||
                tensor_storage_map.find(prefix + ".transformer_blocks.0.audio_attn1.to_gate_logits.weight") != tensor_storage_map.end()) {
                config.self_attention_gated = true;
            }
            if (tensor_storage_map.find(prefix + ".transformer_blocks.0.attn2.to_gate_logits.weight") != tensor_storage_map.end() ||
                tensor_storage_map.find(prefix + ".transformer_blocks.0.audio_attn2.to_gate_logits.weight") != tensor_storage_map.end()) {
                config.cross_attention_gated = true;
            }
            auto caption_linear1_iter = tensor_storage_map.find(prefix + ".caption_projection.linear_1.weight");
            auto caption_linear2_iter = tensor_storage_map.find(prefix + ".caption_projection.linear_2.weight");
            if (caption_linear1_iter == tensor_storage_map.end() &&
                caption_linear2_iter == tensor_storage_map.end()) {
                config.use_caption_projection = false;
            }
            // Classic LTX-Video 0.9.x: the DiT carries an in-model PixArt caption_projection
            // (linear_1 [caption_channels->hidden], linear_2 [hidden->hidden]) that consumes RAW
            // T5-XXL embeddings (caption_channels = 4096). Detect the input width from linear_1 and
            // route it through the after-connector PixArt path. (LTX-2 has no in-DiT
            // caption_projection — its Gemma states are projected externally by
            // text_embedding_projection — so this branch never fires for LTX-2.)
            if (caption_linear1_iter != tensor_storage_map.end() &&
                caption_linear2_iter != tensor_storage_map.end()) {
                config.caption_channels                = caption_linear1_iter->second.ne[0];
                config.caption_proj_before_connector   = false;
                config.caption_projection_first_linear = false;
                // Classic LTX-Video 0.9.x applies INTERLEAVED rotary position embeddings (LTX-2
                // uses the non-interleaved/rotate-half convention). With the wrong convention the
                // self-attention is spatially incoherent and the DiT decodes to white noise.
                config.video_rope_interleaved = true;
            }
            if (tensor_storage_map.find(prefix + ".audio_caption_projection.linear_1.weight") == tensor_storage_map.end() &&
                tensor_storage_map.find(prefix + ".audio_caption_projection.linear_2.weight") == tensor_storage_map.end()) {
                config.use_audio_caption_projection = false;
            }

            config.num_layers = count_prefix_blocks(tensor_storage_map, prefix + ".", "transformer_blocks.");

            auto connector_iter = tensor_storage_map.find(prefix + ".video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight");
            if (connector_iter != tensor_storage_map.end()) {
                config.use_connector         = true;
                config.connector_hidden_size = connector_iter->second.ne[1];
                int64_t connector_heads      = infer_gate_heads(tensor_storage_map,
                                                                prefix + ".video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.bias",
                                                                32);
                auto connector_layout        = infer_attention_layout(config.connector_hidden_size, connector_heads);
                config.connector_num_heads   = connector_layout.first;
                config.connector_head_dim    = connector_layout.second;
                config.connector_num_layers  = count_prefix_blocks(tensor_storage_map, prefix + ".video_embeddings_connector.", "transformer_1d_blocks.");
                auto register_iter           = tensor_storage_map.find(prefix + ".video_embeddings_connector.learnable_registers");
                if (register_iter != tensor_storage_map.end()) {
                    config.connector_num_registers = register_iter->second.ne[1];
                }
                if (tensor_storage_map.find(prefix + ".video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight") != tensor_storage_map.end()) {
                    config.connector_apply_gated_attention = true;
                }
            }

            auto audio_connector_iter = tensor_storage_map.find(prefix + ".audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight");
            if (audio_connector_iter != tensor_storage_map.end()) {
                config.use_audio_connector         = true;
                config.audio_connector_hidden_size = audio_connector_iter->second.ne[1];
                int64_t connector_heads            = infer_gate_heads(tensor_storage_map,
                                                                      prefix + ".audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.bias",
                                                                      32);
                auto connector_layout              = infer_attention_layout(config.audio_connector_hidden_size, connector_heads);
                config.audio_connector_num_heads   = connector_layout.first;
                config.audio_connector_head_dim    = connector_layout.second;
                config.audio_connector_num_layers  = count_prefix_blocks(tensor_storage_map, prefix + ".audio_embeddings_connector.", "transformer_1d_blocks.");
                auto register_iter                 = tensor_storage_map.find(prefix + ".audio_embeddings_connector.learnable_registers");
                if (register_iter != tensor_storage_map.end()) {
                    config.audio_connector_num_registers = register_iter->second.ne[1];
                }
                if (tensor_storage_map.find(prefix + ".audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight") != tensor_storage_map.end()) {
                    config.audio_connector_apply_gated_attention = true;
                }
            }
            // TEMP A/B: env overrides for the (hardcoded) RoPE / positional config while bringing
            // up the classic LTX-Video 0.9.x path — its self-attention output is spatially
            // incoherent, which points at the RoPE convention differing from LTX-2.
            if (const char* e = std::getenv("LTX_ROPE_INTERLEAVE")) config.video_rope_interleaved = atoi(e) != 0;
            if (const char* e = std::getenv("LTX_MIDDLE_GRID"))     config.use_middle_indices_grid = atoi(e) != 0;
            if (const char* e = std::getenv("LTX_CAUSAL_POS"))      config.causal_temporal_positioning = atoi(e) != 0;
            if (const char* e = std::getenv("LTX_ROPE_THETA"))      config.positional_embedding_theta = (float)atof(e);
            LOG_INFO("ltxav RoPE cfg: interleave=%d middle_grid=%d causal_pos=%d theta=%.1f head_dim=%lld",
                     (int)config.video_rope_interleaved, (int)config.use_middle_indices_grid,
                     (int)config.causal_temporal_positioning, config.positional_embedding_theta,
                     (long long)config.attention_head_dim);
            LOG_DEBUG("ltxav: num_layers = %" PRId64 ", hidden_size = %" PRId64 ", num_attention_heads = %" PRId64 ", audio_hidden_size = %" PRId64 ", audio_num_attention_heads = %" PRId64,
                      config.num_layers,
                      config.hidden_size,
                      config.num_attention_heads,
                      config.audio_hidden_size,
                      config.audio_num_attention_heads);
            return config;
        }
    };

    __STATIC_INLINE__ std::vector<float> generate_freq_grid(float theta,
                                                            int positional_dims,
                                                            int dim) {
        const int n_elem     = 2 * positional_dims;
        const int freq_count = dim / n_elem;

        std::vector<float> out(freq_count);
        if (freq_count <= 0) {
            return out;
        }
        if (freq_count == 1) {
            out[0] = 1.5707963267948966f;
            return out;
        }

        const float half_pi   = 1.5707963267948966f;
        const float log_theta = std::log(theta);
        for (int i = 0; i < freq_count; i++) {
            float ratio = static_cast<float>(i) / static_cast<float>(freq_count - 1);
            out[i]      = std::exp(log_theta * ratio) * half_pi;
        }
        return out;
    }

    __STATIC_INLINE__ std::vector<double> generate_freq_grid_double(double theta,
                                                                    int positional_dims,
                                                                    int dim) {
        const int n_elem     = 2 * positional_dims;
        const int freq_count = dim / n_elem;

        std::vector<double> out(freq_count);
        if (freq_count <= 0) {
            return out;
        }
        if (freq_count == 1) {
            out[0] = 1.5707963267948966;
            return out;
        }

        const double half_pi   = 1.5707963267948966;
        const double log_theta = std::log(theta);
        for (int i = 0; i < freq_count; i++) {
            double ratio = static_cast<double>(i) / static_cast<double>(freq_count - 1);
            out[i]       = std::exp(log_theta * ratio) * half_pi;
        }
        return out;
    }

    __STATIC_INLINE__ std::vector<float> build_rope_matrix_from_frequencies(
        const std::vector<std::vector<float>>& frequencies,
        int dim) {
        const int half_dim = dim / 2;
        std::vector<float> out(static_cast<size_t>(frequencies.size()) * static_cast<size_t>(half_dim) * 4, 0.f);

        for (size_t token = 0; token < frequencies.size(); token++) {
            for (int i = 0; i < half_dim; i++) {
                float angle = i < static_cast<int>(frequencies[token].size()) ? frequencies[token][i] : 0.f;
                float c     = std::cos(angle);
                float s     = std::sin(angle);

                size_t base   = (token * static_cast<size_t>(half_dim) + static_cast<size_t>(i)) * 4;
                out[base + 0] = c;
                out[base + 1] = -s;
                out[base + 2] = s;
                out[base + 3] = c;
            }
        }

        return out;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> split_frequencies_by_heads(
        const std::vector<std::vector<float>>& frequencies,
        int inner_dim,
        int num_heads) {
        GGML_ASSERT(num_heads > 0);
        GGML_ASSERT(inner_dim % num_heads == 0);
        const int inner_half_dim    = inner_dim / 2;
        const int per_head_half_dim = inner_half_dim / num_heads;
        GGML_ASSERT(inner_half_dim % num_heads == 0);

        std::vector<std::vector<float>> out(
            frequencies.size() * static_cast<size_t>(num_heads),
            std::vector<float>(per_head_half_dim, 0.f));

        for (size_t token = 0; token < frequencies.size(); token++) {
            GGML_ASSERT(static_cast<int>(frequencies[token].size()) == inner_half_dim);
            for (int head = 0; head < num_heads; head++) {
                auto& dst = out[token * static_cast<size_t>(num_heads) + static_cast<size_t>(head)];
                std::copy_n(frequencies[token].begin() + head * per_head_half_dim, per_head_half_dim, dst.begin());
            }
        }
        return out;
    }

    __STATIC_INLINE__ std::vector<float> build_video_rope_matrix(int64_t width,
                                                                 int64_t height,
                                                                 int64_t frames,
                                                                 int dim,
                                                                 int num_heads                                      = 1,
                                                                 float frame_rate                                   = 24.f,
                                                                 float theta                                        = 10000.f,
                                                                 const std::vector<int>& max_pos                    = {20, 2048, 2048},
                                                                 const std::tuple<int, int, int>& vae_scale_factors = {8, 32, 32},
                                                                 bool causal_temporal_positioning                   = false,
                                                                 bool use_middle_indices_grid                       = false) {
        GGML_ASSERT(max_pos.size() == 3);
        GGML_ASSERT(dim % num_heads == 0);
        const std::vector<float> indices = generate_freq_grid(theta, 3, dim);
        const int half_dim               = dim / 2;
        const int pad_size               = half_dim - static_cast<int>(indices.size()) * 3;

        std::vector<std::vector<float>> freqs(static_cast<size_t>(width * height * frames), std::vector<float>(half_dim, 0.f));

        const int scale_t = std::get<0>(vae_scale_factors);
        const int scale_h = std::get<1>(vae_scale_factors);
        const int scale_w = std::get<2>(vae_scale_factors);

        size_t token = 0;
        for (int64_t t = 0; t < frames; t++) {
            float pixel_t = static_cast<float>(t * scale_t);
            if (causal_temporal_positioning) {
                pixel_t = std::max(0.f, pixel_t + 1.f - scale_t);
            }
            pixel_t /= frame_rate;
            if (use_middle_indices_grid) {
                float end = static_cast<float>((t + 1) * scale_t);
                if (causal_temporal_positioning) {
                    end = std::max(0.f, end + 1.f - scale_t);
                }
                end /= frame_rate;
                pixel_t = 0.5f * (pixel_t + end);
            }

            for (int64_t h = 0; h < height; h++) {
                float pixel_h = static_cast<float>(h * scale_h);
                if (use_middle_indices_grid) {
                    pixel_h += 0.5f * static_cast<float>(scale_h);
                }
                for (int64_t w = 0; w < width; w++) {
                    float pixel_w = static_cast<float>(w * scale_w);
                    if (use_middle_indices_grid) {
                        pixel_w += 0.5f * static_cast<float>(scale_w);
                    }

                    int out_idx = 0;
                    for (int i = 0; i < pad_size; i++) {
                        freqs[token][out_idx++] = 0.f;
                    }

                    const float coords[3] = {
                        pixel_t / max_pos[0],
                        pixel_h / max_pos[1],
                        pixel_w / max_pos[2],
                    };

                    for (float index : indices) {
                        for (int axis = 0; axis < 3; axis++) {
                            freqs[token][out_idx++] = index * (coords[axis] * 2.f - 1.f);
                        }
                    }
                    token++;
                }
            }
        }

        if (num_heads > 1) {
            return build_rope_matrix_from_frequencies(split_frequencies_by_heads(freqs, dim, num_heads), dim / num_heads);
        }
        return build_rope_matrix_from_frequencies(freqs, dim);
    }

    __STATIC_INLINE__ std::vector<float> build_video_rope_matrix_from_positions(const sd::Tensor<float>& positions,
                                                                                int dim,
                                                                                int num_heads,
                                                                                float theta,
                                                                                const std::vector<int>& max_pos,
                                                                                bool use_middle_indices_grid) {
        GGML_ASSERT(max_pos.size() == 3);
        GGML_ASSERT(dim % num_heads == 0);
        GGML_ASSERT(positions.dim() == 3 || positions.dim() == 4);
        GGML_ASSERT(positions.shape()[0] == 2);
        GGML_ASSERT(positions.shape()[1] == 3);
        if (positions.dim() == 4) {
            GGML_ASSERT(positions.shape()[3] == 1);
        }

        const int64_t tokens             = positions.shape()[2];
        const std::vector<float> indices = generate_freq_grid(theta, 3, dim);
        const int half_dim               = dim / 2;
        const int pad_size               = half_dim - static_cast<int>(indices.size()) * 3;
        std::vector<std::vector<float>> freqs(static_cast<size_t>(tokens), std::vector<float>(half_dim, 0.f));

        for (int64_t token = 0; token < tokens; token++) {
            int out_idx = 0;
            for (int i = 0; i < pad_size; i++) {
                freqs[token][out_idx++] = 0.f;
            }

            float coords[3];
            for (int axis = 0; axis < 3; axis++) {
                float start  = positions.dim() == 4 ? positions.index(0, axis, token, 0)
                                                    : positions.index(0, axis, token);
                float end    = positions.dim() == 4 ? positions.index(1, axis, token, 0)
                                                    : positions.index(1, axis, token);
                float coord  = use_middle_indices_grid ? 0.5f * (start + end) : start;
                coords[axis] = coord / static_cast<float>(max_pos[axis]);
            }

            for (float index : indices) {
                for (int axis = 0; axis < 3; axis++) {
                    freqs[token][out_idx++] = index * (coords[axis] * 2.f - 1.f);
                }
            }
        }

        if (num_heads > 1) {
            return build_rope_matrix_from_frequencies(split_frequencies_by_heads(freqs, dim, num_heads), dim / num_heads);
        }
        return build_rope_matrix_from_frequencies(freqs, dim);
    }

    __STATIC_INLINE__ std::vector<float> build_1d_rope_matrix(int64_t seq_len,
                                                              int dim,
                                                              int num_heads          = 1,
                                                              float theta            = 10000.f,
                                                              float positional_scale = 4096.f,
                                                              bool double_precision  = false) {
        GGML_ASSERT(dim % num_heads == 0);
        const std::vector<float> indices = double_precision ? std::vector<float>() : generate_freq_grid(theta, 1, dim);
        const std::vector<double> indices_d =
            double_precision ? generate_freq_grid_double(static_cast<double>(theta), 1, dim) : std::vector<double>();
        const int half_dim = dim / 2;
        const int pad_size = half_dim - static_cast<int>(double_precision ? indices_d.size() : indices.size());

        std::vector<std::vector<float>> freqs(static_cast<size_t>(seq_len), std::vector<float>(half_dim, 0.f));
        for (int64_t pos = 0; pos < seq_len; pos++) {
            int out_idx = 0;
            for (int i = 0; i < pad_size; i++) {
                freqs[static_cast<size_t>(pos)][out_idx++] = 0.f;
            }

            if (double_precision) {
                double coord = static_cast<double>(pos) / static_cast<double>(positional_scale);
                for (double index : indices_d) {
                    freqs[static_cast<size_t>(pos)][out_idx++] = static_cast<float>(index * (coord * 2.0 - 1.0));
                }
            } else {
                float coord = static_cast<float>(pos) / positional_scale;
                for (float index : indices) {
                    freqs[static_cast<size_t>(pos)][out_idx++] = index * (coord * 2.f - 1.f);
                }
            }
        }

        if (num_heads > 1) {
            return build_rope_matrix_from_frequencies(split_frequencies_by_heads(freqs, dim, num_heads), dim / num_heads);
        }
        return build_rope_matrix_from_frequencies(freqs, dim);
    }

    __STATIC_INLINE__ ggml_tensor* apply_hidden_rope(ggml_context* ctx,
                                                     ggml_tensor* x,
                                                     ggml_tensor* pe,
                                                     int64_t heads,
                                                     int64_t dim_head,
                                                     bool rope_interleaved) {
        GGML_ASSERT(x->ne[0] == heads * dim_head);
        // ggml_rope_pe is F32-only (rope-pe.cu asserts F32 in/pe/out). When the DiT
        // runs the F16 residual stream (dit_f16 + F16-dst Linears), q/k arrive F16 —
        // cast them back to F32 here so RoPE is satisfied. (v skips RoPE and stays F16
        // for the flash-attn path, which casts K/V to F16 anyway.)
        // NB: an F16 rope_pe path was tried (so q/k flow F16 end-to-end, dropping this
        // cast) — measured +16% DiT compute on the proxy (3725→4340 ms/step-pair), a
        // clean LOSS, so reverted. The F16 q/k stream slows the downstream attention
        // reshape/cont + flash path more than the saved cast; F32 rope stays the win.
        if (x->type != GGML_TYPE_F32) {
            x = ggml_cast(ctx, x, GGML_TYPE_F32);
        }
        auto x4 = ggml_reshape_4d(ctx, x, dim_head, heads, x->ne[1], x->ne[2]);
        if (pe != nullptr && pe->ne[3] == x->ne[1] * heads) {
            auto x_flat   = ggml_reshape_4d(ctx, x4, dim_head, 1, x->ne[1] * heads, x->ne[2]);
            auto out_flat = Rope::apply_rope(ctx, x_flat, pe, rope_interleaved);
            auto out4     = ggml_reshape_4d(ctx, out_flat, dim_head, heads, x->ne[1], x->ne[2]);
            return ggml_reshape_3d(ctx, out4, heads * dim_head, x->ne[1], x->ne[2]);
        }
        return Rope::apply_rope(ctx, x4, pe, rope_interleaved);
    }

    struct TimestepEmbedder : public GGMLBlock {
        int frequency_embedding_size;

        TimestepEmbedder(int64_t hidden_size,
                         int frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["linear_1"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true, true);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, hidden_size, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, frequency_embedding_size);
            t_emb      = linear_1->forward(ctx, t_emb);
            t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
            t_emb      = linear_2->forward(ctx, t_emb);
            return t_emb;
        }
    };

    struct AdaLayerNormSingle : public GGMLBlock {
        int64_t embedding_dim;
        int64_t embedding_coefficient;

        AdaLayerNormSingle(int64_t embedding_dim,
                           int64_t embedding_coefficient = 6)
            : embedding_dim(embedding_dim), embedding_coefficient(embedding_coefficient) {
            blocks["emb.timestep_embedder"] = std::make_shared<TimestepEmbedder>(embedding_dim);
            blocks["linear"]                = std::make_shared<Linear>(embedding_dim,
                                                        embedding_coefficient * embedding_dim,
                                                        true,
                                                        true);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* timestep) {
            auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["emb.timestep_embedder"]);
            auto linear            = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto embedded_timestep = timestep_embedder->forward(ctx, timestep);
            auto hidden            = ggml_silu(ctx->ggml_ctx, embedded_timestep);
            auto out               = linear->forward(ctx, hidden);
            return {out, embedded_timestep};
        }
    };

    struct PixArtAlphaTextProjection : public GGMLBlock {
        PixArtAlphaTextProjection(int64_t in_features,
                                  int64_t hidden_size,
                                  int64_t out_features = -1) {
            if (out_features < 0) {
                out_features = hidden_size;
            }
            blocks["linear_1"] = std::make_shared<Linear>(in_features, hidden_size, true, true);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, out_features, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* caption) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            caption = linear_1->forward(ctx, caption);
            caption = ggml_ext_gelu(ctx->ggml_ctx, caption, true);
            caption = linear_2->forward(ctx, caption);
            return caption;
        }
    };

    struct NormSingleLinearTextProjection : public GGMLBlock {
        int64_t in_features;
        int64_t hidden_size;

        NormSingleLinearTextProjection(int64_t in_features,
                                       int64_t hidden_size)
            : in_features(in_features), hidden_size(hidden_size) {
            blocks["linear_1"] = std::make_shared<Linear>(in_features, hidden_size, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* caption) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            caption       = ggml_rms_norm(ctx->ggml_ctx, caption, 1e-6f);
            caption       = ggml_ext_scale(ctx->ggml_ctx, caption, std::sqrt(static_cast<float>(hidden_size) / static_cast<float>(in_features)));
            return linear_1->forward(ctx, caption);
        }
    };

    struct CrossAttention : public GGMLBlock {
        int64_t heads;
        int64_t dim_head;
        bool rope_interleaved;

        CrossAttention(int64_t query_dim,
                       int64_t context_dim,
                       int64_t heads,
                       int64_t dim_head,
                       bool apply_gated_attention = false,
                       bool rope_interleaved      = true)
            : heads(heads), dim_head(dim_head), rope_interleaved(rope_interleaved) {
            int64_t inner_dim = heads * dim_head;
            blocks["q_norm"]  = std::make_shared<RMSNorm>(inner_dim, 1e-5f);
            blocks["k_norm"]  = std::make_shared<RMSNorm>(inner_dim, 1e-5f);
            blocks["to_q"]    = std::make_shared<Linear>(query_dim, inner_dim, true);
            blocks["to_k"]    = std::make_shared<Linear>(context_dim, inner_dim, true);
            blocks["to_v"]    = std::make_shared<Linear>(context_dim, inner_dim, true);
            if (apply_gated_attention) {
                blocks["to_gate_logits"] = std::make_shared<Linear>(query_dim, heads, true);
            }
            blocks["to_out.0"] = std::make_shared<Linear>(inner_dim, query_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context     = nullptr,
                             ggml_tensor* mask        = nullptr,
                             ggml_tensor* pe          = nullptr,
                             ggml_tensor* k_pe        = nullptr,
                             ggml_tensor* nag_context = nullptr) {
            if (context == nullptr) {
                context = x;
            }
            auto gc = ctx->ggml_ctx;

            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            auto q = to_q->forward(ctx, x);
            q      = q_norm->forward(ctx, q);
            if (pe != nullptr) {
                q = apply_hidden_rope(gc, q, pe, heads, dim_head, rope_interleaved);
            }

            // Protective kv_scale: with flash-attn, K/V are cast to F16. On deep-chain cont latents
            // at high res (1280x704) a projected/un-normed K or V element can exceed F16's ±65504 →
            // inf → q·inf = NaN in the (unmasked) self-attn softmax → 100% NaN output (the high-res
            // chain blow-up). Scaling K/V down by 1/256 before the F16 cast keeps them in range; the
            // wrapper divides it back out exactly (1/256 is an exact F16 exponent shift). Override
            // with LTX_ATTN_KV_SCALE=N (N=1 disables → reproduces the NaN).
            float kv_scale = 1.0f / 256.0f;
            if (const char* e = std::getenv("LTX_ATTN_KV_SCALE")) {
                float d = (float)atof(e);
                if (d > 0.0f) kv_scale = 1.0f / d;
            }
            // When there is no real attention mask (self-attention), opt into
            // flash_skip_kv_pad: the legacy L_k->256 pad otherwise synthesizes a full
            // [L_k_pad x L_q] all-zeros mask (~N^2, F32 + an F16 copy) purely to pad —
            // at 1280x704 that is the dominant VRAM peak AND the super-linear (frames^2)
            // term. Modern ggml flash_attn_ext handles unpadded L_k with mask==nullptr
            // directly (same path the avatar already uses). No precision change (the
            // mask is zeros), and it skips building+casting the tensor (faster). A real
            // mask (cross-attn) keeps the legacy path.
            const bool skip_kv_pad = (mask == nullptr);
            if (k_pe == nullptr) {
                k_pe = pe;
            }

            // Run one attention with the shared (roped, normed) q against the K/V projected from a
            // given context. Factored out so NAG can run it twice (positive + negative context)
            // with the SAME q. When nag_context is null (the default for every caller except the
            // video text cross-attn on a NAG step) this collapses to exactly the legacy single pass.
            auto attend = [&](ggml_tensor* kv_context) -> ggml_tensor* {
                auto k = to_k->forward(ctx, kv_context);
                auto v = to_v->forward(ctx, kv_context);
                k      = k_norm->forward(ctx, k);
                if (pe != nullptr) {
                    k = apply_hidden_rope(gc, k, k_pe, heads, dim_head, rope_interleaved);
                }
                return ggml_ext_attention_ext(gc,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              heads,
                                              mask,
                                              false,
                                              ctx->flash_attn_enabled,
                                              kv_scale,
                                              skip_kv_pad);
            };

            // Apply the optional gated-attention gate (LTX cross_attention_gated / self_attention
            // _gated). The gate is a function of x (the query hidden) only, so it is IDENTICAL for
            // the positive and negative attention outputs and is applied to each.
            auto apply_gate_if_any = [&](ggml_tensor* out) -> ggml_tensor* {
                if (blocks.count("to_gate_logits") == 0) {
                    return out;
                }
                auto to_gate_logits = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);
                auto gate_logits    = to_gate_logits->forward(ctx, x);
                auto gates          = ggml_sigmoid(gc, gate_logits);
                gates               = ggml_ext_scale(gc, gates, 2.0f, true);
                gates               = ggml_reshape_4d(gc, gates, 1, heads, gate_logits->ne[1], gate_logits->ne[2]);
                auto out4           = ggml_reshape_4d(gc, out, dim_head, heads, out->ne[1], out->ne[2]);
                gates               = ggml_repeat(gc, gates, out4);
                out4                = ggml_mul(gc, out4, gates);
                return ggml_reshape_3d(gc, out4, heads * dim_head, out4->ne[2], out4->ne[3]);
            };

            auto out = apply_gate_if_any(attend(context));

            // ── NAG (Normalized Attention Guidance) ────────────────────────────────────────────
            // Only fires when a negative context is supplied AND NAG is enabled on the runner ctx
            // (ltx_nag_scale != 0). Attention-space negative guidance: extrapolate the positive
            // attention output away from the negative one, clamp the per-token L2-norm growth to
            // tau*||z_pos||, then mix by alpha. Everything is post-`to_out.0` (final hidden space),
            // matching the NAG reference processor (to_out is linear; the norm-clamp is not, so the
            // space it is computed in matters — we use the projected space).
            if (nag_context != nullptr && ctx->ltx_nag_scale != 0.0f) {
                float nag_scale = ctx->ltx_nag_scale;
                float nag_alpha = ctx->ltx_nag_alpha;
                float nag_tau   = ctx->ltx_nag_tau;

                auto out_neg     = apply_gate_if_any(attend(nag_context));
                auto z_pos_o     = to_out_0->forward(ctx, out);      // [query_dim, tokens, batch]
                auto z_neg_o     = to_out_0->forward(ctx, out_neg);
                ggml_type z_dtype = z_pos_o->type;
                // The whole NAG blend runs in F32: ggml_sum_rows requires F32, and it keeps every
                // op same-typed (mixed F16/F32 mul/div assert). Cast the result back at the end.
                auto z_pos = z_pos_o->type == GGML_TYPE_F32 ? z_pos_o : ggml_cast(gc, z_pos_o, GGML_TYPE_F32);
                auto z_neg = z_neg_o->type == GGML_TYPE_F32 ? z_neg_o : ggml_cast(gc, z_neg_o, GGML_TYPE_F32);

                // z_ext = z_pos + scale * (z_pos - z_neg)   (extrapolate in feature space)
                auto z_ext = ggml_add(gc, z_pos, ggml_scale(gc, ggml_sub(gc, z_pos, z_neg), nag_scale));

                // per-token (per-row over the feature dim ne[0]) L2 norms -> [1, tokens, batch]
                auto l2norm_rows = [&](ggml_tensor* z) -> ggml_tensor* {
                    return ggml_sqrt(gc, ggml_sum_rows(gc, ggml_sqr(gc, z)));
                };
                auto n_pos = l2norm_rows(z_pos);
                auto n_ext = l2norm_rows(z_ext);
                // eps-guard the denominator so an all-zero z_ext token can't NaN the ratio.
                n_ext      = ggml_clamp(gc, n_ext, 1e-6f, 3.0e38f);
                // factor = min(1, tau * ||z_pos|| / ||z_ext||)   (clamp the extrapolated norm to <= tau*||z_pos||)
                auto factor = ggml_clamp(gc, ggml_div(gc, ggml_scale(gc, n_pos, nag_tau), n_ext), 0.0f, 1.0f);
                auto z_nag  = ggml_mul(gc, z_ext, factor);  // factor [1,tok,b] broadcasts over feature dim
                // z_out = alpha * z_nag + (1 - alpha) * z_pos
                auto z_out  = ggml_add(gc, ggml_scale(gc, z_nag, nag_alpha), ggml_scale(gc, z_pos, 1.0f - nag_alpha));
                return z_dtype == GGML_TYPE_F32 ? z_out : ggml_cast(gc, z_out, z_dtype);
            }

            return to_out_0->forward(ctx, out);
        }
    };

    struct BasicTransformerBlock : public GGMLBlock {
        int64_t dim;
        bool cross_attention_adaln;
        bool self_attention_gated;
        bool cross_attention_gated;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ggml_type wtype             = get_type(prefix + "scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, dim, cross_attention_adaln ? 9 : 6);
            if (cross_attention_adaln) {
                ggml_type prompt_wtype             = get_type(prefix + "prompt_scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
                params["prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, prompt_wtype, dim, 2);
            }
        }

        BasicTransformerBlock(int64_t dim,
                              int64_t n_heads,
                              int64_t d_head,
                              int64_t context_dim,
                              bool rope_interleaved      = true,
                              bool cross_attention_adaln = false,
                              bool self_attention_gated  = false,
                              bool cross_attention_gated = false)
            : dim(dim),
              cross_attention_adaln(cross_attention_adaln),
              self_attention_gated(self_attention_gated),
              cross_attention_gated(cross_attention_gated) {
            blocks["attn1"] = std::make_shared<CrossAttention>(dim, dim, n_heads, d_head, self_attention_gated, rope_interleaved);
            blocks["attn2"] = std::make_shared<CrossAttention>(dim, context_dim, n_heads, d_head, cross_attention_gated, false);
            blocks["ff"]    = std::make_shared<FeedForward>(dim, dim, 4, FeedForward::Activation::GELU);
        }

        std::vector<ggml_tensor*> get_scale_shift_values(GGMLRunnerContext* ctx,
                                                         ggml_tensor* timestep) {
            auto table    = params["scale_shift_table"];
            int64_t batch = timestep->ne[1];

            int64_t coeff = cross_attention_adaln ? 9 : 6;
            auto t        = ggml_reshape_3d(ctx->ggml_ctx, timestep, dim, coeff, batch);
            auto s        = ggml_reshape_3d(ctx->ggml_ctx, table, dim, coeff, 1);
            auto e        = ggml_new_tensor_3d(ctx->ggml_ctx, timestep->type, dim, coeff, batch);
            s             = ggml_repeat(ctx->ggml_ctx, s, e);
            t             = ggml_repeat(ctx->ggml_ctx, t, e);
            auto out      = ggml_add(ctx->ggml_ctx, s, t);
            return ggml_ext_chunk(ctx->ggml_ctx, out, static_cast<int>(coeff), 1);
        }

        std::vector<ggml_tensor*> get_prompt_scale_shift_values(GGMLRunnerContext* ctx,
                                                                ggml_tensor* prompt_timestep) {
            auto table    = params["prompt_scale_shift_table"];
            int64_t batch = prompt_timestep->ne[1];

            auto t   = ggml_reshape_3d(ctx->ggml_ctx, prompt_timestep, dim, 2, batch);
            auto s   = ggml_reshape_3d(ctx->ggml_ctx, table, dim, 2, 1);
            auto e   = ggml_new_tensor_3d(ctx->ggml_ctx, prompt_timestep->type, dim, 2, batch);
            s        = ggml_repeat(ctx->ggml_ctx, s, e);
            t        = ggml_repeat(ctx->ggml_ctx, t, e);
            auto out = ggml_add(ctx->ggml_ctx, s, t);
            return ggml_ext_chunk(ctx->ggml_ctx, out, 2, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             ggml_tensor* timestep,
                             ggml_tensor* prompt_timestep,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask      = nullptr,
                             ggml_tensor* self_attention_mask = nullptr) {
            auto attn1 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
            auto attn2 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn2"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            auto mods      = get_scale_shift_values(ctx, timestep);
            auto shift_msa = mods[0];
            auto scale_msa = mods[1];
            auto gate_msa  = mods[2];
            auto shift_mlp = mods[3];
            auto scale_mlp = mods[4];
            auto gate_mlp  = mods[5];

            auto x_norm = rms_norm(ctx->ggml_ctx, x);
            x_norm      = modulate(ctx->ggml_ctx, x_norm, shift_msa, scale_msa);
            auto msa    = attn1->forward(ctx, x_norm, nullptr, self_attention_mask, pe);
            x           = ggml_add(ctx->ggml_ctx, x, apply_gate(ctx->ggml_ctx, msa, gate_msa));

            if (cross_attention_adaln) {
                auto shift_q = mods[6];
                auto scale_q = mods[7];
                auto gate_q  = mods[8];

                auto q = rms_norm(ctx->ggml_ctx, x);
                q      = modulate(ctx->ggml_ctx, q, shift_q, scale_q);

                auto context_mod = context;
                if (prompt_timestep != nullptr) {
                    auto prompt_mods = get_prompt_scale_shift_values(ctx, prompt_timestep);
                    context_mod      = modulate(ctx->ggml_ctx, context_mod, prompt_mods[0], prompt_mods[1]);
                }

                auto mca = attn2->forward(ctx, q, context_mod, attention_mask, nullptr, nullptr);
                x        = ggml_add(ctx->ggml_ctx, x, apply_gate(ctx->ggml_ctx, mca, gate_q));
            } else {
                auto mca = attn2->forward(ctx, x, context, attention_mask, nullptr, nullptr);
                x        = ggml_add(ctx->ggml_ctx, x, mca);
            }

            auto y       = rms_norm(ctx->ggml_ctx, x);
            y            = modulate(ctx->ggml_ctx, y, shift_mlp, scale_mlp);
            auto mlp_out = ff->forward(ctx, y);
            x            = ggml_add(ctx->ggml_ctx, x, apply_gate(ctx->ggml_ctx, mlp_out, gate_mlp));
            return x;
        }
    };

    struct BasicTransformerBlock1D : public GGMLBlock {
        BasicTransformerBlock1D(int64_t dim,
                                int64_t n_heads,
                                int64_t d_head,
                                bool rope_interleaved,
                                bool apply_gated_attention = false) {
            blocks["attn1"] = std::make_shared<CrossAttention>(dim, dim, n_heads, d_head, apply_gated_attention, rope_interleaved);
            blocks["ff"]    = std::make_shared<FeedForward>(dim, dim, 4, FeedForward::Activation::GELU);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask = nullptr) {
            auto attn1 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            auto h = rms_norm(ctx->ggml_ctx, x);
            h      = attn1->forward(ctx, h, nullptr, attention_mask, pe);
            x      = ggml_add(ctx->ggml_ctx, x, h);

            h = rms_norm(ctx->ggml_ctx, x);
            h = ff->forward(ctx, h);
            x = ggml_add(ctx->ggml_ctx, x, h);
            return x;
        }
    };

    struct Embeddings1DConnector : public GGMLBlock {
        int64_t hidden_size;
        int64_t num_attention_heads;
        int64_t attention_head_dim;
        int64_t num_layers;
        int64_t num_learnable_registers;
        bool rope_interleaved;
        bool apply_gated_attention;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            if (num_learnable_registers > 0) {
                ggml_type wtype               = get_type(prefix + "learnable_registers", tensor_storage_map, GGML_TYPE_F32);
                params["learnable_registers"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, num_learnable_registers);
            }
        }

        Embeddings1DConnector(int64_t hidden_size,
                              int64_t num_attention_heads     = 30,
                              int64_t attention_head_dim      = 128,
                              int64_t num_layers              = 2,
                              int64_t num_learnable_registers = 128,
                              bool rope_interleaved           = false,
                              bool apply_gated_attention      = false)
            : hidden_size(hidden_size),
              num_attention_heads(num_attention_heads),
              attention_head_dim(attention_head_dim),
              num_layers(num_layers),
              num_learnable_registers(num_learnable_registers),
              rope_interleaved(rope_interleaved),
              apply_gated_attention(apply_gated_attention) {
            for (int i = 0; i < num_layers; i++) {
                blocks["transformer_1d_blocks." + std::to_string(i)] =
                    std::make_shared<BasicTransformerBlock1D>(hidden_size,
                                                              num_attention_heads,
                                                              attention_head_dim,
                                                              rope_interleaved,
                                                              apply_gated_attention);
            }
        }

        ggml_tensor* append_registers(GGMLRunnerContext* ctx,
                                      ggml_tensor* hidden_states) {
            if (num_learnable_registers <= 0 || params.count("learnable_registers") == 0) {
                return hidden_states;
            }

            int64_t seq_len       = hidden_states->ne[1];
            int64_t target_len    = std::max<int64_t>(1024, seq_len);
            int64_t duplications  = (target_len + num_learnable_registers - 1) / num_learnable_registers;
            int64_t total_to_keep = duplications * num_learnable_registers - seq_len;
            if (total_to_keep <= 0) {
                return hidden_states;
            }

            auto regs = ggml_reshape_3d(ctx->ggml_ctx, params["learnable_registers"], hidden_size, num_learnable_registers, 1);
            // The learnable_registers param may be stored at a different precision than the
            // running hidden_states (e.g. BF16 registers in an NVFP4 gguf vs F32 activations).
            // ggml_concat below requires uniform type, so match the registers to hidden_states.
            if (regs->type != hidden_states->type) {
                regs = ggml_cast(ctx->ggml_ctx, regs, hidden_states->type);
            }
            auto temp = ggml_new_tensor_3d(ctx->ggml_ctx, regs->type, regs->ne[0], regs->ne[1], hidden_states->ne[2]);
            regs      = ggml_repeat(ctx->ggml_ctx, regs, temp);

            auto regs_full = regs;
            for (int64_t i = 1; i < duplications; i++) {
                regs_full = ggml_concat(ctx->ggml_ctx, regs_full, regs, 1);
            }
            regs_full = ggml_ext_slice(ctx->ggml_ctx, regs_full, 1, seq_len, seq_len + total_to_keep);
            return ggml_concat(ctx->ggml_ctx, hidden_states, regs_full, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask = nullptr) {
            hidden_states = append_registers(ctx, hidden_states);

            for (int i = 0; i < num_layers; i++) {
                auto block    = std::dynamic_pointer_cast<BasicTransformerBlock1D>(blocks["transformer_1d_blocks." + std::to_string(i)]);
                hidden_states = block->forward(ctx, hidden_states, pe, attention_mask);
            }

            return ggml_rms_norm(ctx->ggml_ctx, hidden_states, 1e-6f);
        }
    };

    __STATIC_INLINE__ std::pair<int64_t, int64_t> infer_attention_layout(int64_t hidden_size,
                                                                         int64_t preferred_heads = -1) {
        if (preferred_heads > 0 && hidden_size % preferred_heads == 0) {
            return {preferred_heads, hidden_size / preferred_heads};
        }
        const int candidates[] = {128, 96, 80, 64, 48, 40, 32};
        for (int head_dim : candidates) {
            if (hidden_size % head_dim == 0) {
                int64_t heads = hidden_size / head_dim;
                if (heads >= 8 && heads <= 64) {
                    return {heads, head_dim};
                }
            }
        }
        return {32, hidden_size / 32};
    }

    __STATIC_INLINE__ std::vector<float> build_1d_rope_matrix_from_coords(const std::vector<float>& coords,
                                                                          int dim,
                                                                          int num_heads         = 1,
                                                                          float theta           = 10000.f,
                                                                          float max_pos         = 20.f,
                                                                          bool double_precision = false) {
        GGML_ASSERT(dim % num_heads == 0);
        const std::vector<float> indices = double_precision ? std::vector<float>() : generate_freq_grid(theta, 1, dim);
        const std::vector<double> indices_d =
            double_precision ? generate_freq_grid_double(static_cast<double>(theta), 1, dim) : std::vector<double>();
        const int half_dim = dim / 2;
        const int pad_size = half_dim - static_cast<int>(double_precision ? indices_d.size() : indices.size());

        std::vector<std::vector<float>> freqs(coords.size(), std::vector<float>(half_dim, 0.f));
        for (size_t pos = 0; pos < coords.size(); pos++) {
            int out_idx = 0;
            for (int i = 0; i < pad_size; i++) {
                freqs[pos][out_idx++] = 0.f;
            }
            if (double_precision) {
                double coord = static_cast<double>(coords[pos]) / static_cast<double>(max_pos);
                for (double index : indices_d) {
                    freqs[pos][out_idx++] = static_cast<float>(index * (coord * 2.0 - 1.0));
                }
            } else {
                float coord = coords[pos] / max_pos;
                for (float index : indices) {
                    freqs[pos][out_idx++] = index * (coord * 2.f - 1.f);
                }
            }
        }
        if (num_heads > 1) {
            return build_rope_matrix_from_frequencies(split_frequencies_by_heads(freqs, dim, num_heads), dim / num_heads);
        }
        return build_rope_matrix_from_frequencies(freqs, dim);
    }

    __STATIC_INLINE__ float video_latent_corner_to_time_sec(int64_t corner_index,
                                                            int scale_t,
                                                            float frame_rate,
                                                            bool causal_temporal_positioning) {
        float pixel_t = static_cast<float>(corner_index * scale_t);
        if (causal_temporal_positioning) {
            pixel_t = std::max(0.f, pixel_t + 1.f - scale_t);
        }
        return pixel_t / frame_rate;
    }

    __STATIC_INLINE__ std::vector<float> build_video_temporal_rope_matrix(int64_t width,
                                                                          int64_t height,
                                                                          int64_t frames,
                                                                          int dim,
                                                                          int num_heads,
                                                                          float frame_rate,
                                                                          float theta,
                                                                          int max_pos_t,
                                                                          int scale_t,
                                                                          bool causal_temporal_positioning,
                                                                          bool use_middle_indices_grid) {
        std::vector<float> coords;
        coords.reserve(static_cast<size_t>(width * height * frames));
        for (int64_t t = 0; t < frames; t++) {
            float coord = video_latent_corner_to_time_sec(t, scale_t, frame_rate, causal_temporal_positioning);
            if (use_middle_indices_grid) {
                float end = video_latent_corner_to_time_sec(t + 1, scale_t, frame_rate, causal_temporal_positioning);
                coord     = 0.5f * (coord + end);
            }
            for (int64_t h = 0; h < height; h++) {
                for (int64_t w = 0; w < width; w++) {
                    coords.push_back(coord);
                }
            }
        }
        return build_1d_rope_matrix_from_coords(coords, dim, num_heads, theta, static_cast<float>(max_pos_t));
    }

    __STATIC_INLINE__ std::vector<float> build_video_temporal_rope_matrix_from_positions(const sd::Tensor<float>& positions,
                                                                                         int dim,
                                                                                         int num_heads,
                                                                                         float theta,
                                                                                         int max_pos_t,
                                                                                         bool use_middle_indices_grid) {
        GGML_ASSERT(positions.dim() == 3 || positions.dim() == 4);
        GGML_ASSERT(positions.shape()[0] == 2);
        GGML_ASSERT(positions.shape()[1] >= 1);
        if (positions.dim() == 4) {
            GGML_ASSERT(positions.shape()[3] == 1);
        }

        std::vector<float> coords;
        coords.reserve(static_cast<size_t>(positions.shape()[2]));
        for (int64_t token = 0; token < positions.shape()[2]; token++) {
            float start = positions.dim() == 4 ? positions.index(0, 0, token, 0)
                                               : positions.index(0, 0, token);
            float end   = positions.dim() == 4 ? positions.index(1, 0, token, 0)
                                               : positions.index(1, 0, token);
            coords.push_back(use_middle_indices_grid ? 0.5f * (start + end) : start);
        }
        return build_1d_rope_matrix_from_coords(coords, dim, num_heads, theta, static_cast<float>(max_pos_t));
    }

    __STATIC_INLINE__ float audio_latent_start_time_sec(int64_t latent_index,
                                                        int audio_latent_downsample_factor = 4,
                                                        int hop_length                     = 160,
                                                        int sample_rate                    = 16000,
                                                        bool causal                        = true) {
        float mel_frame = static_cast<float>(latent_index * audio_latent_downsample_factor);
        if (causal) {
            mel_frame = std::max(0.f, mel_frame + 1.f - static_cast<float>(audio_latent_downsample_factor));
        }
        return mel_frame * static_cast<float>(hop_length) / static_cast<float>(sample_rate);
    }

    __STATIC_INLINE__ std::vector<float> build_audio_rope_matrix(int64_t seq_len,
                                                                 int dim,
                                                                 int num_heads,
                                                                 float theta                  = 10000.f,
                                                                 int max_pos_t                = 20,
                                                                 bool use_middle_indices_grid = false) {
        std::vector<float> coords(static_cast<size_t>(seq_len), 0.f);
        for (int64_t t = 0; t < seq_len; t++) {
            float start = audio_latent_start_time_sec(t);
            if (use_middle_indices_grid) {
                float end                      = audio_latent_start_time_sec(t + 1);
                coords[static_cast<size_t>(t)] = 0.5f * (start + end);
            } else {
                coords[static_cast<size_t>(t)] = start;
            }
        }
        return build_1d_rope_matrix_from_coords(coords, dim, num_heads, theta, static_cast<float>(max_pos_t));
    }

    struct BasicAVTransformerBlock : public GGMLBlock {
        int64_t v_dim;
        int64_t a_dim;
        bool cross_attention_adaln;
        bool has_audio = true;
        // LTX_BLOCK_STOP_AT_SUBOP: when this block is the truncation block (set by
        // forward_core for the LTX_DIT_STOP_AT_BLOCK block), return vx early after a
        // chosen video sub-op so the surviving terminal output isolates WHICH op in the
        // offending block first goes NaN. 0=after self-attn (attn1), 1=after text cross-attn
        // (attn2), 2=after audio_to_video cross-attn (a2v), 3=after video ffn (== full block).
        // Unset/-1 = run the whole block. Composes with LTX_DIT_STOP_AT_BLOCK.
        int stop_at_subop = -1;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            int64_t coeff                     = cross_attention_adaln ? 9 : 6;
            ggml_type vw                      = get_type(prefix + "scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
            params["scale_shift_table"]       = ggml_new_tensor_2d(ctx, vw, v_dim, coeff);

            if (cross_attention_adaln) {
                ggml_type vpw                      = get_type(prefix + "prompt_scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
                params["prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, vpw, v_dim, 2);
            }

            if (has_audio) {
                ggml_type aw                      = get_type(prefix + "audio_scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
                params["audio_scale_shift_table"] = ggml_new_tensor_2d(ctx, aw, a_dim, coeff);
                if (cross_attention_adaln) {
                    ggml_type apw                            = get_type(prefix + "audio_prompt_scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
                    params["audio_prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, apw, a_dim, 2);
                }
                ggml_type avw                            = get_type(prefix + "scale_shift_table_a2v_ca_audio", tensor_storage_map, GGML_TYPE_F32);
                ggml_type vaw                            = get_type(prefix + "scale_shift_table_a2v_ca_video", tensor_storage_map, GGML_TYPE_F32);
                params["scale_shift_table_a2v_ca_audio"] = ggml_new_tensor_2d(ctx, avw, a_dim, 5);
                params["scale_shift_table_a2v_ca_video"] = ggml_new_tensor_2d(ctx, vaw, v_dim, 5);
            }
        }

        BasicAVTransformerBlock(int64_t v_dim,
                                int64_t a_dim,
                                int64_t v_heads,
                                int64_t a_heads,
                                int64_t vd_head,
                                int64_t ad_head,
                                int64_t v_context_dim,
                                int64_t a_context_dim,
                                bool apply_gated_attention,
                                bool cross_attention_adaln,
                                bool video_rope_interleaved,
                                bool has_audio = true)
            : v_dim(v_dim),
              a_dim(a_dim),
              cross_attention_adaln(cross_attention_adaln),
              has_audio(has_audio) {
            blocks["attn1"]               = std::make_shared<CrossAttention>(v_dim, v_dim, v_heads, vd_head, apply_gated_attention, video_rope_interleaved);
            blocks["attn2"]               = std::make_shared<CrossAttention>(v_dim, v_context_dim, v_heads, vd_head, apply_gated_attention, false);
            blocks["ff"]                  = std::make_shared<FeedForward>(v_dim, v_dim, 4, FeedForward::Activation::GELU);
            if (has_audio) {
                blocks["audio_attn1"]         = std::make_shared<CrossAttention>(a_dim, a_dim, a_heads, ad_head, apply_gated_attention, false);
                blocks["audio_attn2"]         = std::make_shared<CrossAttention>(a_dim, a_context_dim, a_heads, ad_head, apply_gated_attention, false);
                blocks["audio_to_video_attn"] = std::make_shared<CrossAttention>(v_dim, a_dim, a_heads, ad_head, apply_gated_attention, false);
                blocks["video_to_audio_attn"] = std::make_shared<CrossAttention>(a_dim, v_dim, a_heads, ad_head, apply_gated_attention, false);
                blocks["audio_ff"]            = std::make_shared<FeedForward>(a_dim, a_dim, 4, FeedForward::Activation::GELU);
            }
        }

        std::vector<ggml_tensor*> get_ada_values(GGMLRunnerContext* ctx,
                                                 ggml_tensor* table,
                                                 ggml_tensor* timestep,
                                                 int64_t dim,
                                                 int64_t coeff,
                                                 int64_t start           = 0,
                                                 int64_t count           = -1,
                                                 ggml_tensor* expand_sel = nullptr) {
            if (count < 0) {
                count = coeff - start;
            }
            // `timestep` is [coeff*dim, batch]. When modulation token-collapse is active for
            // this stream, batch == U (the few unique timestep values) and expand_sel maps
            // each token back to its column; otherwise batch == n_tokens (per-token, old path)
            // and expand_sel is null.
            auto t      = ggml_reshape_3d(ctx->ggml_ctx, timestep, dim, coeff, timestep->ne[1]);
            auto s      = ggml_reshape_3d(ctx->ggml_ctx, table, dim, coeff, 1);
            auto e      = ggml_new_tensor_3d(ctx->ggml_ctx, timestep->type, dim, coeff, timestep->ne[1]);
            t           = ggml_repeat(ctx->ggml_ctx, t, e);
            s           = ggml_repeat(ctx->ggml_ctx, s, e);
            auto out    = ggml_add(ctx->ggml_ctx, s, t);
            auto chunks = ggml_ext_chunk(ctx->ggml_ctx, out, static_cast<int>(coeff), 1);
            std::vector<ggml_tensor*> sel(chunks.begin() + start, chunks.begin() + start + count);
            if (expand_sel != nullptr) {
                for (auto& c : sel) {
                    c = gather_mod_tokens(ctx->ggml_ctx, c, expand_sel);
                }
            }
            return sel;
        }

        ggml_tensor* apply_text_cross_attention(GGMLRunnerContext* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* context,
                                                CrossAttention* attn,
                                                ggml_tensor* table,
                                                ggml_tensor* prompt_table,
                                                ggml_tensor* timestep,
                                                ggml_tensor* prompt_timestep,
                                                int64_t dim,
                                                ggml_tensor* attention_mask,
                                                ggml_tensor* expand_sel   = nullptr,
                                                ggml_tensor* context_neg   = nullptr) {
            // NAG: when context_neg is supplied it is the NEGATIVE text context. It must receive the
            // SAME prompt-token modulation (prompt_scale_shift) as the positive context before being
            // handed to CrossAttention as the nag_context (which projects its own K/V from it and
            // NAG-blends against the positive attention output). nullptr = no NAG (legacy).
            if (cross_attention_adaln) {
                auto q_mods      = get_ada_values(ctx, table, timestep, dim, 9, 6, 3, expand_sel);
                ggml_tensor* q   = nullptr;
                if (ltx_rms_mod_fuse_enabled()) {
                    q = modulate_fused(ctx->ggml_ctx, x, q_mods[0], q_mods[1]);
                } else {
                    q = rms_norm(ctx->ggml_ctx, x);
                    q = modulate(ctx->ggml_ctx, q, q_mods[0], q_mods[1]);
                }
                auto context_mod     = context;
                auto context_neg_mod = context_neg;
                if (prompt_timestep != nullptr && prompt_table != nullptr) {
                    auto p_mods = get_ada_values(ctx, prompt_table, prompt_timestep, dim, 2);
                    context_mod = modulate(ctx->ggml_ctx, context_mod, p_mods[0], p_mods[1]);
                    if (context_neg_mod != nullptr) {
                        context_neg_mod = modulate(ctx->ggml_ctx, context_neg_mod, p_mods[0], p_mods[1]);
                    }
                }
                auto out = attn->forward(ctx, q, context_mod, attention_mask, nullptr, nullptr, context_neg_mod);
                return apply_gate(ctx->ggml_ctx, out, q_mods[2]);
            }

            auto q = rms_norm(ctx->ggml_ctx, x);
            return attn->forward(ctx, q, context, attention_mask, nullptr, nullptr, context_neg);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* vx,
                                                      ggml_tensor* ax,
                                                      ggml_tensor* v_context,
                                                      ggml_tensor* a_context,
                                                      ggml_tensor* attention_mask,
                                                      ggml_tensor* v_timestep,
                                                      ggml_tensor* a_timestep,
                                                      ggml_tensor* v_pe,
                                                      ggml_tensor* a_pe,
                                                      ggml_tensor* v_cross_pe,
                                                      ggml_tensor* a_cross_pe,
                                                      ggml_tensor* v_cross_scale_shift_timestep,
                                                      ggml_tensor* a_cross_scale_shift_timestep,
                                                      ggml_tensor* v_cross_gate_timestep,
                                                      ggml_tensor* a_cross_gate_timestep,
                                                      ggml_tensor* v_prompt_timestep,
                                                      ggml_tensor* a_prompt_timestep,
                                                      ggml_tensor* self_attention_mask = nullptr,
                                                      ggml_tensor* v_context_neg       = nullptr) {
            auto attn1               = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
            auto audio_attn1         = std::dynamic_pointer_cast<CrossAttention>(blocks["audio_attn1"]);
            auto attn2               = std::dynamic_pointer_cast<CrossAttention>(blocks["attn2"]);
            auto audio_attn2         = std::dynamic_pointer_cast<CrossAttention>(blocks["audio_attn2"]);
            auto audio_to_video_attn = std::dynamic_pointer_cast<CrossAttention>(blocks["audio_to_video_attn"]);
            auto video_to_audio_attn = std::dynamic_pointer_cast<CrossAttention>(blocks["video_to_audio_attn"]);
            auto ff                  = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
            auto audio_ff            = std::dynamic_pointer_cast<FeedForward>(blocks["audio_ff"]);

            auto v_table = cast_modulation_table(ctx->ggml_ctx, params["scale_shift_table"]);
            auto a_table = cast_modulation_table(ctx->ggml_ctx, params["audio_scale_shift_table"]);

            bool run_ax  = ax != nullptr && ggml_nelements(ax) > 0 && ax->ne[1] > 0;
            // A2V modality-guidance "mod" pass: drop the audio<->video cross-attention so the
            // video is predicted as if it ignored the driving audio (ctx->ltx_skip_a2v). The
            // audio self/cross stack (audio_attn1/2) still runs so the frozen audio stays valid;
            // only the coupling into/out of the video stream is severed.
            bool run_a2v = run_ax && !ctx->ltx_skip_a2v;
            bool run_v2a = run_ax && !ctx->ltx_skip_a2v;

            auto v_mods = get_ada_values(ctx, v_table, v_timestep, v_dim, cross_attention_adaln ? 9 : 6, 0, -1, ctx->ltx_video_token_sel);
            ggml_tensor* v_norm = nullptr;
            if (ltx_rms_mod_fuse_enabled()) {
                v_norm = modulate_fused(ctx->ggml_ctx, vx, v_mods[0], v_mods[1]);
            } else {
                v_norm = rms_norm(ctx->ggml_ctx, vx);
                v_norm = modulate(ctx->ggml_ctx, v_norm, v_mods[0], v_mods[1]);
            }
            auto v_sa   = attn1->forward(ctx, v_norm, nullptr, self_attention_mask, v_pe);
            vx          = ggml_add(ctx->ggml_ctx, vx, apply_gate(ctx->ggml_ctx, v_sa, v_mods[2]));
            if (stop_at_subop == 0) {
                return {vx, ax};  // truncate after video self-attn (attn1)
            }
            auto v_txt  = apply_text_cross_attention(ctx,
                                                     vx,
                                                     v_context,
                                                     attn2.get(),
                                                     v_table,
                                                    cross_attention_adaln ? params["prompt_scale_shift_table"] : nullptr,
                                                     v_timestep,
                                                     v_prompt_timestep,
                                                     v_dim,
                                                     attention_mask,
                                                     ctx->ltx_video_token_sel,
                                                     v_context_neg);  // NAG negative video text context (null unless a NAG step)
            vx          = ggml_add(ctx->ggml_ctx, vx, v_txt);
            if (stop_at_subop == 1) {
                return {vx, ax};  // truncate after video text cross-attn (attn2)
            }

            if (run_ax) {
                auto a_mods = get_ada_values(ctx, a_table, a_timestep, a_dim, cross_attention_adaln ? 9 : 6);
                ggml_tensor* a_norm = nullptr;
                if (ltx_rms_mod_fuse_enabled()) {
                    a_norm = modulate_fused(ctx->ggml_ctx, ax, a_mods[0], a_mods[1]);
                } else {
                    a_norm = rms_norm(ctx->ggml_ctx, ax);
                    a_norm = modulate(ctx->ggml_ctx, a_norm, a_mods[0], a_mods[1]);
                }
                auto a_sa   = audio_attn1->forward(ctx, a_norm, nullptr, nullptr, a_pe);
                ax          = ggml_add(ctx->ggml_ctx, ax, apply_gate(ctx->ggml_ctx, a_sa, a_mods[2]));
                auto a_txt  = apply_text_cross_attention(ctx,
                                                         ax,
                                                         a_context,
                                                         audio_attn2.get(),
                                                         a_table,
                                                        cross_attention_adaln ? params["audio_prompt_scale_shift_table"] : nullptr,
                                                         a_timestep,
                                                         a_prompt_timestep,
                                                         a_dim,
                                                         attention_mask);
                ax          = ggml_add(ctx->ggml_ctx, ax, a_txt);

                auto vx_norm3 = rms_norm(ctx->ggml_ctx, vx);
                auto ax_norm3 = rms_norm(ctx->ggml_ctx, ax);

                if (run_a2v) {
                    auto a2v_audio_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_audio"], 1, 0, 4);
                    auto a2v_video_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_video"], 1, 0, 4);
                    auto a2v_audio       = get_ada_values(ctx, a2v_audio_table, a_cross_scale_shift_timestep, a_dim, 4);
                    auto a2v_video       = get_ada_values(ctx, a2v_video_table, v_cross_scale_shift_timestep, v_dim, 4);
                    auto vx_scaled       = modulate(ctx->ggml_ctx, vx_norm3, a2v_video[1], a2v_video[0]);
                    auto ax_scaled       = modulate(ctx->ggml_ctx, ax_norm3, a2v_audio[1], a2v_audio[0]);
                    auto a2v_out         = audio_to_video_attn->forward(ctx, vx_scaled, ax_scaled, nullptr, v_cross_pe, a_cross_pe);
                    auto a2v_gate_table  = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_video"], 1, 4, 5);
                    auto a2v_gate        = get_ada_values(ctx, a2v_gate_table, v_cross_gate_timestep, v_dim, 1)[0];
                    vx                   = ggml_add(ctx->ggml_ctx, vx, apply_gate(ctx->ggml_ctx, a2v_out, a2v_gate));
                }
                if (stop_at_subop == 2) {
                    return {vx, ax};  // truncate after audio_to_video cross-attn (a2v)
                }

                if (run_v2a) {
                    auto v2a_audio_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_audio"], 1, 0, 4);
                    auto v2a_video_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_video"], 1, 0, 4);
                    auto v2a_audio       = get_ada_values(ctx, v2a_audio_table, a_cross_scale_shift_timestep, a_dim, 4);
                    auto v2a_video       = get_ada_values(ctx, v2a_video_table, v_cross_scale_shift_timestep, v_dim, 4);
                    auto ax_scaled       = modulate(ctx->ggml_ctx, ax_norm3, v2a_audio[3], v2a_audio[2]);
                    auto vx_scaled       = modulate(ctx->ggml_ctx, vx_norm3, v2a_video[3], v2a_video[2]);
                    auto v2a_out         = video_to_audio_attn->forward(ctx, ax_scaled, vx_scaled, nullptr, a_cross_pe, v_cross_pe);
                    auto v2a_gate_table  = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_audio"], 1, 4, 5);
                    auto v2a_gate        = get_ada_values(ctx, v2a_gate_table, a_cross_gate_timestep, a_dim, 1)[0];
                    ax                   = ggml_add(ctx->ggml_ctx, ax, apply_gate(ctx->ggml_ctx, v2a_out, v2a_gate));
                }
                auto a_ff_mods = get_ada_values(ctx, a_table, a_timestep, a_dim, cross_attention_adaln ? 9 : 6, 3, 3);
                ggml_tensor* ax_scaled = nullptr;
                if (ltx_rms_mod_fuse_enabled()) {
                    ax_scaled = modulate_fused(ctx->ggml_ctx, ax, a_ff_mods[0], a_ff_mods[1]);
                } else {
                    ax_scaled = rms_norm(ctx->ggml_ctx, ax);
                    ax_scaled = modulate(ctx->ggml_ctx, ax_scaled, a_ff_mods[0], a_ff_mods[1]);
                }
                auto a_ff_out  = audio_ff->forward(ctx, ax_scaled);
                ax             = ggml_add(ctx->ggml_ctx, ax, apply_gate(ctx->ggml_ctx, a_ff_out, a_ff_mods[2]));
            }

            auto v_ff_mods = get_ada_values(ctx, v_table, v_timestep, v_dim, cross_attention_adaln ? 9 : 6, 3, 3, ctx->ltx_video_token_sel);
            ggml_tensor* vx_scaled = nullptr;
            if (ltx_rms_mod_fuse_enabled()) {
                vx_scaled = modulate_fused(ctx->ggml_ctx, vx, v_ff_mods[0], v_ff_mods[1]);
            } else {
                vx_scaled = rms_norm(ctx->ggml_ctx, vx);
                vx_scaled = modulate(ctx->ggml_ctx, vx_scaled, v_ff_mods[0], v_ff_mods[1]);
            }
            auto v_ff_out  = ff->forward(ctx, vx_scaled);
            vx             = ggml_add(ctx->ggml_ctx, vx, apply_gate(ctx->ggml_ctx, v_ff_out, v_ff_mods[2]));

            return {vx, ax};
        }
    };

    struct LTXAVModelBlock : public GGMLBlock {
        LTXAVConfig config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"]       = ggml_new_tensor_2d(ctx,
                                                                   get_type(prefix + "scale_shift_table", tensor_storage_map, GGML_TYPE_F32),
                                                                   config.hidden_size,
                                                                   2);
            if (config.has_audio) {
                params["audio_scale_shift_table"] = ggml_new_tensor_2d(ctx,
                                                                       get_type(prefix + "audio_scale_shift_table", tensor_storage_map, GGML_TYPE_F32),
                                                                       config.audio_hidden_size,
                                                                       2);
            }
        }

        LTXAVModelBlock(const LTXAVConfig& config)
            : config(config) {
            blocks["patchify_proj"]       = std::make_shared<Linear>(config.in_channels, config.hidden_size, true, true);
            blocks["adaln_single"]        = std::make_shared<AdaLayerNormSingle>(config.hidden_size, config.cross_attention_adaln ? 9 : 6);
            if (config.has_audio) {
                blocks["audio_patchify_proj"] = std::make_shared<Linear>(config.audio_in_channels, config.audio_hidden_size, true, true);
                blocks["audio_adaln_single"]  = std::make_shared<AdaLayerNormSingle>(config.audio_hidden_size, config.cross_attention_adaln ? 9 : 6);
                if (config.cross_attention_adaln) {
                    blocks["prompt_adaln_single"]       = std::make_shared<AdaLayerNormSingle>(config.hidden_size, 2);
                    blocks["audio_prompt_adaln_single"] = std::make_shared<AdaLayerNormSingle>(config.audio_hidden_size, 2);
                }
                blocks["av_ca_video_scale_shift_adaln_single"] = std::make_shared<AdaLayerNormSingle>(config.hidden_size, 4);
                blocks["av_ca_a2v_gate_adaln_single"]          = std::make_shared<AdaLayerNormSingle>(config.hidden_size, 1);
                blocks["av_ca_audio_scale_shift_adaln_single"] = std::make_shared<AdaLayerNormSingle>(config.audio_hidden_size, 4);
                blocks["av_ca_v2a_gate_adaln_single"]          = std::make_shared<AdaLayerNormSingle>(config.audio_hidden_size, 1);
            } else if (config.cross_attention_adaln) {
                blocks["prompt_adaln_single"] = std::make_shared<AdaLayerNormSingle>(config.hidden_size, 2);
            }

            if (config.use_caption_projection) {
                if (config.caption_proj_before_connector) {
                    if (config.caption_projection_first_linear) {
                        blocks["caption_projection"] = std::make_shared<NormSingleLinearTextProjection>(config.caption_channels, config.hidden_size);
                    }
                } else {
                    blocks["caption_projection"] = std::make_shared<PixArtAlphaTextProjection>(config.caption_channels, config.hidden_size, config.hidden_size);
                }
            }
            if (config.use_audio_caption_projection) {
                if (config.caption_proj_before_connector) {
                    if (config.caption_projection_first_linear) {
                        blocks["audio_caption_projection"] = std::make_shared<NormSingleLinearTextProjection>(config.caption_channels, config.audio_hidden_size);
                    }
                } else {
                    blocks["audio_caption_projection"] = std::make_shared<PixArtAlphaTextProjection>(config.caption_channels, config.audio_hidden_size, config.audio_hidden_size);
                }
            }

            if (config.use_connector) {
                blocks["video_embeddings_connector"] = std::make_shared<Embeddings1DConnector>(config.connector_hidden_size,
                                                                                               config.connector_num_heads,
                                                                                               config.connector_head_dim,
                                                                                               config.connector_num_layers,
                                                                                               config.connector_num_registers,
                                                                                               config.connector_rope_interleaved,
                                                                                               config.connector_apply_gated_attention);
            }
            if (config.use_audio_connector) {
                blocks["audio_embeddings_connector"] = std::make_shared<Embeddings1DConnector>(config.audio_connector_hidden_size,
                                                                                               config.audio_connector_num_heads,
                                                                                               config.audio_connector_head_dim,
                                                                                               config.audio_connector_num_layers,
                                                                                               config.audio_connector_num_registers,
                                                                                               config.audio_connector_rope_interleaved,
                                                                                               config.audio_connector_apply_gated_attention);
            }

            for (int i = 0; i < config.num_layers; i++) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<BasicAVTransformerBlock>(config.hidden_size,
                                                                                                              config.audio_hidden_size,
                                                                                                              config.num_attention_heads,
                                                                                                              config.audio_num_attention_heads,
                                                                                                              config.attention_head_dim,
                                                                                                              config.audio_attention_head_dim,
                                                                                                              config.cross_attention_dim,
                                                                                                              config.audio_cross_attention_dim,
                                                                                                              config.self_attention_gated || config.cross_attention_gated,
                                                                                                              config.cross_attention_adaln,
                                                                                                              config.video_rope_interleaved,
                                                                                                              config.has_audio);
            }

            blocks["norm_out"]       = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["proj_out"]       = std::make_shared<Linear>(config.hidden_size, config.out_channels, true, true);
            if (config.has_audio) {
                blocks["audio_norm_out"] = std::make_shared<LayerNorm>(config.audio_hidden_size, 1e-6f, false);
                blocks["audio_proj_out"] = std::make_shared<Linear>(config.audio_hidden_size, config.audio_out_channels, true, true);
            }
        }

        ggml_tensor* patchify_video(GGMLRunnerContext* ctx, ggml_tensor* x, int64_t n) {
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3] / n, n);
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            return x;
        }

        ggml_tensor* unpatchify_video(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,
                                      int64_t width,
                                      int64_t height,
                                      int64_t frames) {
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x = ggml_reshape_4d(ctx->ggml_ctx, x, width, height, frames, x->ne[1] * x->ne[2]);
            return x;
        }

        ggml_tensor* patchify_audio(GGMLRunnerContext* ctx, ggml_tensor* ax) {
            // ax: [b, c, t, f]
            ax = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, ax, 0, 2, 1, 3));  // [b, t, c, f]
            ax = ggml_reshape_3d(ctx->ggml_ctx, ax, ax->ne[0] * ax->ne[1], ax->ne[2], ax->ne[3]);  // [b, t, c*f]
            return ax;
        }

        ggml_tensor* repeat_scalar_timestep_like(GGMLRunnerContext* ctx, ggml_tensor* timestep, ggml_tensor* like) {
            GGML_ASSERT(timestep != nullptr && like != nullptr);
            if (timestep->ne[0] == like->ne[0]) {
                return timestep;
            }
            GGML_ASSERT(timestep->ne[0] == 1);
            return ggml_repeat(ctx->ggml_ctx, timestep, ggml_new_tensor_1d(ctx->ggml_ctx, timestep->type, like->ne[0]));
        }

        ggml_tensor* unpatchify_audio(GGMLRunnerContext* ctx, ggml_tensor* ax, int64_t audio_length) {
            if (ax == nullptr) {
                return nullptr;
            }
            ax = ggml_reshape_4d(ctx->ggml_ctx, ax, config.audio_frequency_bins, config.num_audio_channels, audio_length, ax->ne[2]);  // [b, t, c, f]
            ax = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, ax, 0, 2, 1, 3));                                      // [b, c, t, f]
            return ax;
        }

        std::pair<ggml_tensor*, ggml_tensor*> preprocess_contexts(GGMLRunnerContext* ctx,
                                                                  ggml_tensor* context,
                                                                  ggml_tensor* video_connector_pe,
                                                                  ggml_tensor* audio_connector_pe,
                                                                  bool process_audio_context) {
            if (context == nullptr) {
                return {nullptr, nullptr};
            }

            // Video-only checkpoints feed a raw [caption_channels, tokens] T5 context whose width
            // (4096) can numerically equal cross_attention_dim + audio_cross_attention_dim; the
            // dual v+a split must be disabled so that context is NOT sliced in half.
            bool is_fully_processed_context =
                config.has_audio &&
                context->ne[0] == config.cross_attention_dim + config.audio_cross_attention_dim &&
                context->ne[1] >= 1024;
            bool is_unprocessed_dual_context =
                config.has_audio &&
                context->ne[0] == config.cross_attention_dim + config.audio_cross_attention_dim &&
                context->ne[1] < 1024;

            if (is_fully_processed_context) {
                auto v_context         = ggml_ext_slice(ctx->ggml_ctx, context, 0, 0, config.cross_attention_dim);
                ggml_tensor* a_context = nullptr;
                if (process_audio_context) {
                    a_context = ggml_ext_slice(ctx->ggml_ctx, context, 0, config.cross_attention_dim, config.cross_attention_dim + config.audio_cross_attention_dim);
                }
                return {v_context, a_context};
            }

            ggml_tensor* v_context = context;
            ggml_tensor* a_context = process_audio_context ? context : nullptr;
            if (is_unprocessed_dual_context) {
                v_context = ggml_ext_slice(ctx->ggml_ctx, context, 0, 0, config.cross_attention_dim);
                if (process_audio_context) {
                    a_context = ggml_ext_slice(ctx->ggml_ctx, context, 0, config.cross_attention_dim, config.cross_attention_dim + config.audio_cross_attention_dim);
                }
            } else if (context->ne[0] == config.caption_channels * 2) {
                v_context = ggml_ext_slice(ctx->ggml_ctx, context, 0, 0, config.caption_channels);
                if (process_audio_context) {
                    a_context = ggml_ext_slice(ctx->ggml_ctx, context, 0, config.caption_channels, config.caption_channels * 2);
                }
            }

            if (config.caption_proj_before_connector) {
                if (config.use_caption_projection &&
                    blocks.count("caption_projection") > 0 &&
                    v_context != nullptr &&
                    v_context->ne[0] == config.caption_channels) {
                    auto caption_projection = std::dynamic_pointer_cast<NormSingleLinearTextProjection>(blocks["caption_projection"]);
                    if (caption_projection != nullptr) {
                        v_context = caption_projection->forward(ctx, v_context);
                    }
                }
                if (process_audio_context &&
                    config.use_audio_caption_projection &&
                    blocks.count("audio_caption_projection") > 0 &&
                    a_context != nullptr &&
                    a_context->ne[0] == config.caption_channels) {
                    auto caption_projection = std::dynamic_pointer_cast<NormSingleLinearTextProjection>(blocks["audio_caption_projection"]);
                    if (caption_projection != nullptr) {
                        a_context = caption_projection->forward(ctx, a_context);
                    }
                }
            }

            if (config.use_connector && v_context != nullptr && v_context->ne[0] == config.connector_hidden_size) {
                auto connector = std::dynamic_pointer_cast<Embeddings1DConnector>(blocks["video_embeddings_connector"]);
                v_context      = connector->forward(ctx, v_context, video_connector_pe);
            }
            if (process_audio_context &&
                config.use_audio_connector &&
                a_context != nullptr &&
                a_context->ne[0] == config.audio_connector_hidden_size) {
                auto connector = std::dynamic_pointer_cast<Embeddings1DConnector>(blocks["audio_embeddings_connector"]);
                a_context      = connector->forward(ctx, a_context, audio_connector_pe);
            }

            if (!config.caption_proj_before_connector &&
                config.use_caption_projection &&
                blocks.count("caption_projection") > 0 &&
                v_context != nullptr &&
                v_context->ne[0] == config.caption_channels) {
                auto caption_projection = std::dynamic_pointer_cast<PixArtAlphaTextProjection>(blocks["caption_projection"]);
                if (caption_projection != nullptr) {
                    v_context = caption_projection->forward(ctx, v_context);
                }
            }
            if (process_audio_context &&
                !config.caption_proj_before_connector &&
                config.use_audio_caption_projection &&
                blocks.count("audio_caption_projection") > 0 &&
                a_context != nullptr &&
                a_context->ne[0] == config.caption_channels) {
                auto caption_projection = std::dynamic_pointer_cast<PixArtAlphaTextProjection>(blocks["audio_caption_projection"]);
                if (caption_projection != nullptr) {
                    a_context = caption_projection->forward(ctx, a_context);
                }
            }

            return {v_context, a_context};
        }

        std::vector<ggml_tensor*> get_output_scale_shift(GGMLRunnerContext* ctx,
                                                         ggml_tensor* table,
                                                         ggml_tensor* embedded_timestep,
                                                         int64_t dim,
                                                         ggml_tensor* expand_sel = nullptr) {
            auto temp = ggml_new_tensor_3d(ctx->ggml_ctx, embedded_timestep->type, dim, 2, embedded_timestep->ne[1]);
            auto t    = ggml_repeat(ctx->ggml_ctx, ggml_reshape_3d(ctx->ggml_ctx, embedded_timestep, dim, 1, embedded_timestep->ne[1]), temp);
            auto s    = ggml_repeat(ctx->ggml_ctx, ggml_reshape_3d(ctx->ggml_ctx, table, dim, 2, 1), temp);
            auto out  = ggml_add(ctx->ggml_ctx, s, t);
            auto chunks = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 1);
            if (expand_sel != nullptr) {
                for (auto& c : chunks) {
                    c = gather_mod_tokens(ctx->ggml_ctx, c, expand_sel);
                }
            }
            return chunks;
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* vx,
                                                      ggml_tensor* ax,
                                                      ggml_tensor* timestep,
                                                      ggml_tensor* audio_timestep,
                                                      ggml_tensor* context,
                                                      ggml_tensor* v_pe,
                                                      ggml_tensor* a_pe,
                                                      ggml_tensor* v_cross_pe,
                                                      ggml_tensor* a_cross_pe,
                                                      ggml_tensor* video_connector_pe,
                                                      ggml_tensor* audio_connector_pe,
                                                      ggml_tensor* vx_ref      = nullptr,
                                                      ggml_tensor* nag_context = nullptr) {
            auto patchify_proj       = std::dynamic_pointer_cast<Linear>(blocks["patchify_proj"]);
            auto audio_patchify_proj = std::dynamic_pointer_cast<Linear>(blocks["audio_patchify_proj"]);
            auto adaln_single        = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["adaln_single"]);
            auto audio_adaln_single  = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["audio_adaln_single"]);
            auto norm_out            = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out            = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            auto audio_norm_out      = std::dynamic_pointer_cast<LayerNorm>(blocks["audio_norm_out"]);
            auto audio_proj_out      = std::dynamic_pointer_cast<Linear>(blocks["audio_proj_out"]);

            GGML_ASSERT(vx->ne[3] % config.in_channels == 0);
            int64_t n          = vx->ne[3] / config.in_channels;
            int64_t width      = vx->ne[0];
            int64_t height     = vx->ne[1];
            int64_t frames     = vx->ne[2];
            // Video-only (classic LTX-Video 0.9.x) checkpoints have no audio sub-blocks; never run
            // the audio path even if an audio latent is somehow supplied (would null-deref).
            int64_t audio_time = (config.has_audio && ax != nullptr) ? ax->ne[1] : 0;

            vx = patchify_video(ctx, vx, n);
            vx = patchify_proj->forward(ctx, vx);  // [hidden, target_tokens, n]
            // FIX A2 separable half-res relip reference: patchify the SEPARATE [W/N,H/N,ref,C]
            // reference grid with the SAME patchify_proj and append its tokens to the video
            // sequence (token axis = ne[1]). The combined sequence then flows through the blocks
            // with the combined video_pe / per-token timesteps (built in build_graph); the ref
            // tokens are sliced back off before unpatchify_video below. vx_ref==nullptr (every
            // non-separable path, incl. N==1 full-res concat) leaves vx exactly as before.
            const int64_t target_token_count = width * height * frames;
            if (vx_ref != nullptr) {
                GGML_ASSERT(vx_ref->ne[3] % config.in_channels == 0);
                int64_t n_ref = vx_ref->ne[3] / config.in_channels;
                auto rx       = patchify_video(ctx, vx_ref, n_ref);
                rx            = patchify_proj->forward(ctx, rx);  // [hidden, ref_tokens, n]
                vx            = ggml_concat(ctx->ggml_ctx, vx, rx, 1);
            }
            if (ax != nullptr && ggml_nelements(ax) > 0 && audio_time > 0) {
                ax = patchify_audio(ctx, ax);
                ax = audio_patchify_proj->forward(ctx, ax);
            } else {
                ax = nullptr;
            }

            // WORKSTREAM B experiment (LTX_DIT_F16, default OFF) — run the DiT residual
            // stream in F16 to halve glue memory traffic (§9c). MEASURED 2026-06-22: F16 is
            // range-SAFE (nnan=0, clean latents at 1280x704 — the kv_scale overflow note does
            // NOT bite the residual stream) BUT a NET LOSS at prod scale: 193f sampling
            // 228s(F32) -> 304s(F16), +33%, per-segment compute 1767->5485ms. Cause: F16
            // activations fed into the FP4 cuBLASLt GEMM fall OFF the fast tensor-core path
            // (the quantized matmul wants F32 activation). So naive activation-dtype casting
            // is capped by the matmul. The real glue+matmul lever = a fast FP4 GEMM that
            // takes low-precision (FP8/FP16) activations directly (comfy's native scaled_mm)
            // — a ggml-cuda nvfp4-cublaslt change, NOT this cast. Kept env-gated as the
            // scaffold for that follow-up; leave OFF.
            static const bool dit_f16 = (std::getenv("LTX_DIT_F16") != nullptr);
            if (dit_f16) {
                vx = ggml_cast(ctx->ggml_ctx, vx, GGML_TYPE_F16);
                if (ax != nullptr) {
                    ax = ggml_cast(ctx->ggml_ctx, ax, GGML_TYPE_F16);
                }
            }

            bool run_ax    = ax != nullptr && ggml_nelements(ax) > 0 && audio_time > 0;
            auto contexts  = preprocess_contexts(ctx, context, video_connector_pe, audio_connector_pe, run_ax);
            auto v_context = contexts.first;
            auto a_context = contexts.second != nullptr ? contexts.second : contexts.first;
            if (contexts.second != nullptr) {
                a_context = ggml_cont(ctx->ggml_ctx, a_context);
            }

            // NAG: preprocess the NEGATIVE text context through the SAME caption_projection +
            // connector as the positive VIDEO context (video branch only — NAG steers video text
            // cross-attn). Reuses video_connector_pe, which is sized for the positive context's
            // sequence length in build_graph — so the negative context MUST be encoded/padded to
            // that same length (the Gemma TE path pads both prompts to a fixed max_len, so this
            // holds; see CPP-CHANGES.md build-verification checklist). Only runs on NAG steps.
            ggml_tensor* v_context_neg = nullptr;
            if (nag_context != nullptr && ctx->ltx_nag_scale != 0.0f) {
                auto neg_contexts = preprocess_contexts(ctx, nag_context, video_connector_pe, audio_connector_pe, false);
                v_context_neg     = neg_contexts.first;
            }

            auto v_timestep_scaled = ggml_ext_scale(ctx->ggml_ctx, timestep, config.timestep_scale_multiplier);
            auto v_pair            = adaln_single->forward(ctx, v_timestep_scaled);
            auto v_timestep_mod    = v_pair.first;
            auto v_embedded_time   = v_pair.second;

            // Audio + audio<->video-cross timestep modulations exist only on AV (LTX-2)
            // checkpoints. On a video-only (classic LTX-Video 0.9.x) checkpoint the audio /
            // av_ca adaLN blocks are not built, run_ax is always false, and none of these are
            // consumed downstream — so leave them null.
            ggml_tensor* a_timestep_mod        = nullptr;
            ggml_tensor* a_embedded_time       = nullptr;
            ggml_tensor* v_prompt_timestep_mod = nullptr;
            ggml_tensor* a_prompt_timestep_mod = nullptr;
            ggml_tensor* av_ca_video_scale_shift_timestep = nullptr;
            ggml_tensor* av_ca_a2v_gate_noise_timestep    = nullptr;
            ggml_tensor* av_ca_audio_scale_shift_timestep = nullptr;
            ggml_tensor* av_ca_v2a_gate_noise_timestep    = nullptr;
            if (config.has_audio) {
                ggml_tensor* effective_audio_timestep = audio_timestep != nullptr ? audio_timestep : timestep;
                auto a_timestep_scaled                = ggml_ext_scale(ctx->ggml_ctx, effective_audio_timestep, config.timestep_scale_multiplier);
                auto a_pair                           = audio_adaln_single->forward(ctx, a_timestep_scaled);
                a_timestep_mod                        = a_pair.first;
                a_embedded_time                       = a_pair.second;

                if (config.cross_attention_adaln) {
                    auto prompt_adaln_single       = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["prompt_adaln_single"]);
                    auto audio_prompt_adaln_single = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["audio_prompt_adaln_single"]);
                    v_prompt_timestep_mod          = prompt_adaln_single->forward(ctx, a_timestep_scaled).first;
                    a_prompt_timestep_mod          = audio_prompt_adaln_single->forward(ctx, a_timestep_scaled).first;
                }

                // When modulation token-collapse is active (ctx->ltx_video_token_sel set), `timestep`
                // is the COMPACT video timestep [U], so the old repeat-to-video-length would size this
                // to U (wrong). The video-side cross-attn timestep is derived from the (scalar) audio
                // timestep and is therefore CONSTANT across video tokens — keep it scalar so get_ada_values
                // broadcasts it (bit-exact to the per-token-constant path), no per-token materialization.
                const bool mod_collapse   = ctx->ltx_video_token_sel != nullptr;
                auto av_ca_video_timestep = (mod_collapse && effective_audio_timestep->ne[0] == 1)
                                                ? effective_audio_timestep
                                                : repeat_scalar_timestep_like(ctx, effective_audio_timestep, timestep);
                auto av_ca_audio_timestep = effective_audio_timestep;
                auto av_ca_factor         = config.av_ca_timestep_scale_multiplier / config.timestep_scale_multiplier;
                av_ca_video_scale_shift_timestep =
                    std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_video_scale_shift_adaln_single"])->forward(ctx, av_ca_video_timestep).first;
                av_ca_a2v_gate_noise_timestep =
                    std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_a2v_gate_adaln_single"])
                        ->forward(ctx, ggml_ext_scale(ctx->ggml_ctx, av_ca_video_timestep, av_ca_factor))
                        .first;
                av_ca_audio_scale_shift_timestep =
                    std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_audio_scale_shift_adaln_single"])->forward(ctx, av_ca_audio_timestep).first;
                av_ca_v2a_gate_noise_timestep =
                    std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_v2a_gate_adaln_single"])
                        ->forward(ctx, ggml_ext_scale(ctx->ggml_ctx, av_ca_audio_timestep, av_ca_factor))
                        .first;
            }

            sd::ggml_graph_cut::mark_graph_cut(vx, "ltxav.prelude", "vx");
            sd::ggml_graph_cut::mark_graph_cut(ax, "ltxav.prelude", "ax");

            // LTX_BLOCK_NAN: capture the modulation tables + patchified inputs BEFORE the blocks, so
            // if block 0 is already NaN we can tell whether the AdaLN modulation (the #1 collapse
            // output) / the patchify input is the source vs the first block's ops.
            if (getenv("LTX_BLOCK_NAN") != nullptr) {
                auto cap = [&](ggml_tensor* t, const char* nm) {
                    if (t == nullptr || ggml_nelements(t) == 0) return;
                    int64_t k = std::min<int64_t>(512, ggml_nelements(t));
                    ctx->capture_tensor(std::string("pre_") + nm, ggml_view_1d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, t), k, 0));
                };
                cap(vx, "vx_in");
                cap(v_timestep_mod, "v_mod");
                cap(a_timestep_mod, "a_mod");
                cap(av_ca_video_scale_shift_timestep, "avca_v");
                cap(v_prompt_timestep_mod, "v_prompt");
            }
            // LTX_DIT_STOP_AT_BLOCK=N: graph-cut-SURVIVING NaN bisection. The per-block
            // capture_tensor taps above get pruned by the offload graph-cut (mid-DiT nodes
            // aren't consumed at a segment boundary, so the snapshot node is dropped — only
            // mark_graph_cut'd / terminal-output tensors survive). To localize the first NaN
            // block we instead TRUNCATE the network after block N: vx then flows through the
            // real norm_out/proj_out/unpatchify tail to the actual model output, so it survives
            // the cut and the existing LTX_NAN_DEBUG step-2 `out` scan reports nan-count for the
            // network-truncated-at-N. Bisect N: out clean at N=k but NaN at N=k+1 ⇒ block k+1
            // is the source (then the BSA self-attn / cross-attn / ffn inside it). N<0 / unset
            // = full network. Tiny cost (just runs fewer blocks). One run per bisection point.
            static const int dit_stop_at_block = [] {
                const char* e = getenv("LTX_DIT_STOP_AT_BLOCK");
                return e != nullptr ? atoi(e) : -1;
            }();
            static const int dit_stop_at_subop = [] {
                const char* e = getenv("LTX_BLOCK_STOP_AT_SUBOP");
                return e != nullptr ? atoi(e) : -1;
            }();
            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<BasicAVTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                // Apply the intra-block sub-op truncation only on the LAST block we run, so
                // it composes with LTX_DIT_STOP_AT_BLOCK (e.g. STOP_AT_BLOCK=3 SUBOP=0 =
                // "run blocks 0-2 whole, then block 3 only up to its self-attn").
                block->stop_at_subop = (dit_stop_at_block >= 0 && i == dit_stop_at_block) ? dit_stop_at_subop : -1;
                auto out   = block->forward(ctx,
                                            vx,
                                            ax,
                                            v_context,
                                            a_context,
                                            nullptr,
                                            v_timestep_mod,
                                            a_timestep_mod,
                                            v_pe,
                                            a_pe,
                                            v_cross_pe,
                                            a_cross_pe,
                                            av_ca_video_scale_shift_timestep,
                                            av_ca_audio_scale_shift_timestep,
                                            av_ca_a2v_gate_noise_timestep,
                                            av_ca_v2a_gate_noise_timestep,
                                            v_prompt_timestep_mod,
                                            a_prompt_timestep_mod,
                                            nullptr,          // self_attention_mask (unused on AV path)
                                            v_context_neg);   // NAG negative video text context (null unless a NAG step)
                vx         = out.first;
                ax         = out.second;
                // LTX_BLOCK_NAN: capture a small slice of each block's vx/ax for the post-compute
                // nnan readback — pinpoints which transformer block first goes NaN (the NaN is 100%
                // of elements, so a 512-element slice is enough). Tiny VRAM. One run localizes the op.
                static const bool block_nan_dbg = (getenv("LTX_BLOCK_NAN") != nullptr);
                if (block_nan_dbg) {
                    int64_t tv = std::min<int64_t>(512, ggml_nelements(vx));
                    ctx->capture_tensor("blk" + std::to_string(i) + "_vx", ggml_view_1d(ctx->ggml_ctx, vx, tv, 0));
                    if (ax != nullptr && ggml_nelements(ax) > 0) {
                        int64_t ta = std::min<int64_t>(512, ggml_nelements(ax));
                        ctx->capture_tensor("blk" + std::to_string(i) + "_ax", ggml_view_1d(ctx->ggml_ctx, ax, ta, 0));
                    }
                }
                sd::ggml_graph_cut::mark_graph_cut(vx, "ltxav.transformer_blocks." + std::to_string(i), "vx");
                sd::ggml_graph_cut::mark_graph_cut(ax, "ltxav.transformer_blocks." + std::to_string(i), "ax");
                if (dit_stop_at_block >= 0 && i >= dit_stop_at_block) {
                    LOG_INFO("[LTX_DIT_STOP] truncating DiT after block %d (of %lld)", i, (long long)config.num_layers);
                    break;
                }
            }

            if (dit_f16 && vx->type != GGML_TYPE_F32) {
                vx = ggml_cast(ctx->ggml_ctx, vx, GGML_TYPE_F32);
            }
            auto v_shift_scale = get_output_scale_shift(ctx, cast_modulation_table(ctx->ggml_ctx, params["scale_shift_table"]), v_embedded_time, config.hidden_size, ctx->ltx_video_token_sel);
            vx                 = norm_out->forward(ctx, vx);
            vx                 = modulate(ctx->ggml_ctx, vx, v_shift_scale[0], v_shift_scale[1]);
            vx                 = proj_out->forward(ctx, vx);  // [out_dim, total_tokens, n]
            // FIX A2 separable relip: drop the appended reference tokens, keeping only the
            // target tokens, so unpatchify reconstructs the target [W,H,frames] grid (the output
            // is already target-only => the post-sampling frame crop is a no-op). No-op when no
            // reference was appended (target_token_count == total tokens).
            if (vx_ref != nullptr && vx->ne[1] != target_token_count) {
                vx = ggml_cont(ctx->ggml_ctx,
                               ggml_view_3d(ctx->ggml_ctx, vx,
                                            vx->ne[0], target_token_count, vx->ne[2],
                                            vx->nb[1], vx->nb[2], 0));
            }
            vx                 = unpatchify_video(ctx, vx, width, height, frames);

            if (ax != nullptr && audio_time > 0) {
                if (dit_f16 && ax->type != GGML_TYPE_F32) {
                    ax = ggml_cast(ctx->ggml_ctx, ax, GGML_TYPE_F32);
                }
                auto a_shift_scale = get_output_scale_shift(ctx, params["audio_scale_shift_table"], a_embedded_time, config.audio_hidden_size);
                ax                 = audio_norm_out->forward(ctx, ax);
                ax                 = modulate(ctx->ggml_ctx, ax, a_shift_scale[0], a_shift_scale[1]);
                ax                 = audio_proj_out->forward(ctx, ax);
                ax                 = unpatchify_audio(ctx, ax, audio_time);
            }

            return {vx, ax};
        }
    };

    struct LTXAVRunner : public DiffusionModelRunner {
        LTXAVConfig config;
        LTXAVModelBlock model;
        std::vector<float> video_pe_vec;
        std::vector<float> audio_pe_vec;
        std::vector<float> video_cross_pe_vec;
        std::vector<float> audio_cross_pe_vec;
        std::vector<float> connector_pe_vec;
        std::vector<float> audio_connector_pe_vec;
        sd::Tensor<float> vx_input_cache;
        sd::Tensor<float> ax_input_cache;
        // FIX A2 separable half-res relip reference: the [W/N,H/N,ref,C] reference grid fed as a
        // separate DiT token block. Held as a member so its data survives until the backend
        // reads it (same lifetime contract as vx_input_cache). Empty unless N>1 relip.
        sd::Tensor<float> vx_ref_input_cache;
        sd::Tensor<float> v_timestep_combined_cache;  // target per-token ts ++ ref frozen t=0 (separable relip)
        // Modulation token-collapse (VRAM win). The conditioned video timestep is per-token
        // but has only a few UNIQUE values; we feed the blocks the compact unique set and a
        // per-token selector so get_ada_values gathers each compact chunk back per-token.
        // These persist as members (the backend reads their data after build_graph returns).
        sd::Tensor<float> v_timestep_compact_cache;  // [U] unique video timesteps
        std::vector<int32_t> v_token_sel_vec;        // [L_video] token -> unique-column index
        bool skip_a2v_cross_attn_ = false;           // A2V modality-guidance "mod" pass (set per compute)
        // NAG (Normalized Attention Guidance) params, set per-compute from LTXAVDiffusionExtra and
        // forwarded to the runner ctx in build_graph. nag_scale_ == 0 => NAG off (default).
        float nag_scale_ = 0.0f;
        float nag_alpha_ = 0.35f;
        float nag_tau_   = 2.5f;

        LTXAVRunner(ggml_backend_t backend,
                    ggml_backend_t params_backend,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string& prefix                      = "model.diffusion_model")
            : DiffusionModelRunner(backend, params_backend, prefix),
              config(LTXAVConfig::detect_from_weights(tensor_storage_map, prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "ltxav";
        }

        bool has_audio_stream() const override {
            return config.has_audio;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> split_av_latents(const sd::Tensor<float>& x_tensor,
                                                                         int audio_length) const {
            if (x_tensor.empty()) {
                return {{}, {}};
            }

            GGML_ASSERT(x_tensor.dim() == 4 || x_tensor.dim() == 5);
            if (x_tensor.dim() == 5) {
                GGML_ASSERT(x_tensor.shape()[4] == 1);
            }
            int64_t width          = x_tensor.shape()[0];
            int64_t height         = x_tensor.shape()[1];
            int64_t frames         = x_tensor.shape()[2];
            int64_t total_channels = x_tensor.shape()[3];
            int64_t spatial_size   = width * height * frames;

            GGML_ASSERT(total_channels >= config.in_channels);

            sd::Tensor<float> vx({width, height, frames, config.in_channels});
            size_t video_values = static_cast<size_t>(config.in_channels * spatial_size);
            std::copy_n(x_tensor.data(), video_values, vx.data());

            if (audio_length <= 0 || total_channels == config.in_channels) {
                return {vx, {}};
            }

            int64_t needed_audio_values = static_cast<int64_t>(audio_length) * config.num_audio_channels * config.audio_frequency_bins;
            int64_t packed_audio_values = (total_channels - config.in_channels) * spatial_size;
            GGML_ASSERT(packed_audio_values >= needed_audio_values);

            sd::Tensor<float> ax({config.audio_frequency_bins, audio_length, config.num_audio_channels, 1});
            const float* audio_src = x_tensor.data() + video_values;
            std::copy_n(audio_src, static_cast<size_t>(needed_audio_values), ax.data());
            return {vx, ax};
        }

        ggml_tensor* merge_av_latents(ggml_context* ctx,
                                      ggml_tensor* vx,
                                      ggml_tensor* ax) const {
            if (ax == nullptr || ggml_nelements(ax) == 0 || ax->ne[1] == 0) {
                return vx;
            }

            int64_t width        = vx->ne[0];
            int64_t height       = vx->ne[1];
            int64_t frames       = vx->ne[2];
            int64_t divisor      = width * height * frames;
            int64_t audio_values = ax->ne[0] * ax->ne[1] * ax->ne[2] * ax->ne[3];
            int64_t pad_values   = (divisor - (audio_values % divisor)) % divisor;
            int64_t padded_len   = audio_values + pad_values;

            ax = ggml_cont(ctx, ax);
            ax = ggml_reshape_4d(ctx, ax, audio_values, 1, 1, 1);
            if (pad_values > 0) {
                ax = ggml_ext_pad(ctx, ax, static_cast<int>(pad_values), 0, 0, 0);
            }
            int64_t extra_channels = padded_len / divisor;
            ax                     = ggml_reshape_4d(ctx, ax, width, height, frames, extra_channels);
            return ggml_concat(ctx, vx, ax, 3);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor         = {},
                                 const sd::Tensor<float>& audio_x_tensor         = {},
                                 const sd::Tensor<float>& audio_timesteps_tensor = {},
                                 int audio_length                                = 0,
                                 float frame_rate                                = 24.f,
                                 const sd::Tensor<float>& video_positions_tensor = {},
                                 const sd::Tensor<float>& audio_positions_tensor = {},
                                 const sd::Tensor<float>& video_reference_tensor = {},
                                 const sd::Tensor<float>& nag_context_tensor     = {}) {
            auto split_inputs = split_av_latents(x_tensor, audio_length);
            vx_input_cache    = split_inputs.first;
            if (!audio_x_tensor.empty()) {
                ax_input_cache = audio_x_tensor;
            } else {
                ax_input_cache = split_inputs.second;
            }

            ggml_tensor* vx         = make_input(vx_input_cache);
            ggml_tensor* ax         = make_optional_input(ax_input_cache);

            // FIX A2 separable half-res relip reference: a non-empty video_reference_tensor is a
            // SEPARATE [W/N,H/N,ref,C] grid that the DiT patchifies on its own and appends to the
            // video token sequence (see forward()). Its per-token timesteps are appended here as
            // frozen t=0 so the combined timestep vector matches the combined token count.
            ggml_tensor* vx_ref     = nullptr;
            int64_t ref_token_count = 0;
            if (!video_reference_tensor.empty()) {
                vx_ref_input_cache = video_reference_tensor;
                vx_ref             = make_input(vx_ref_input_cache);
                GGML_ASSERT(vx_ref->ne[3] % config.in_channels == 0);
                ref_token_count = vx_ref->ne[0] * vx_ref->ne[1] * vx_ref->ne[2];
            }

            // --- Modulation token-collapse (VRAM win; ports avatar 98e8d16 to LTX) ---
            // The conditioned video timestep is per-token (len = video_token_count) but holds
            // only a few UNIQUE values (clean-anchor t=0, noise t, plus any graded-overlap
            // levels). Feed the blocks just those unique values + a per-token selector so each
            // per-block AdaLN materializes [dim,coeff,U] (U≈2-5) instead of [dim,coeff,L_video]
            // (~5k) — cuts the conditioned DiT compute buffer by ~2GB. Pure t2v generate
            // already passes a length-1 timestep (broadcast) and needs nothing. Bit-exactness:
            // the gather is a pure copy; the only divergence vs the old path is that the compact
            // adaln matmul runs at width U (may route to a different ggml-cuda kernel for
            // quantized weights — same effect as avatar 98e8d16, slightly MORE accurate).
            // Escape hatches: LTX_MOD_COLLAPSE=0 (old per-token path), LTX_MOD_NO_DEDUP=1
            // (compact==per-token, sel=identity → isolates/validates the gather as bit-exact).
            ggml_tensor* timesteps         = nullptr;
            ggml_tensor* v_token_sel_input = nullptr;
            bool collapse_env              = true;
            if (const char* e = std::getenv("LTX_MOD_COLLAPSE")) {
                collapse_env = e[0] != '0';
            }
            bool no_dedup = std::getenv("LTX_MOD_NO_DEDUP") != nullptr;
            // FIX A2 separable relip: append `ref_token_count` frozen (t=0) per-token timesteps
            // for the appended reference token block. The reference is a clean conditioning
            // signal (no noise), so t=0 is correct and dedups to a single extra unique value in
            // the modulation collapse. When ref_token_count==0 the effective timesteps ARE the
            // original tensor (no copy) => N==1 path byte-identical.
            const sd::Tensor<float>* eff_ts = &timesteps_tensor;
            if (ref_token_count > 0 && timesteps_tensor.numel() > 0) {
                std::vector<float> combined(timesteps_tensor.data(), timesteps_tensor.data() + timesteps_tensor.numel());
                combined.insert(combined.end(), static_cast<size_t>(ref_token_count), 0.0f);
                v_timestep_combined_cache = sd::Tensor<float>({static_cast<int64_t>(combined.size())}, combined);
                eff_ts                    = &v_timestep_combined_cache;
            }
            int64_t n_ts  = static_cast<int64_t>(eff_ts->numel());
            if (collapse_env && n_ts > 1) {
                const float* td = eff_ts->data();
                std::vector<float> uniq;
                v_token_sel_vec.resize(static_cast<size_t>(n_ts));
                for (int64_t i = 0; i < n_ts; ++i) {
                    float v = td[i];
                    int idx = -1;
                    if (!no_dedup) {
                        for (size_t u = 0; u < uniq.size(); ++u) {
                            if (uniq[u] == v) {
                                idx = static_cast<int>(u);
                                break;
                            }
                        }
                    }
                    if (idx < 0) {
                        idx = static_cast<int>(uniq.size());
                        uniq.push_back(v);
                    }
                    v_token_sel_vec[static_cast<size_t>(i)] = idx;
                }
                v_timestep_compact_cache = sd::Tensor<float>({static_cast<int64_t>(uniq.size())}, uniq);
                timesteps                = make_input(v_timestep_compact_cache);
                v_token_sel_input        = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_ts);
                ggml_set_name(v_token_sel_input, "ltxav_video_token_sel");
                set_backend_tensor_data(v_token_sel_input, v_token_sel_vec.data());
                LOG_DEBUG("ltxav modulation collapse: %lld video tokens -> %zu unique timesteps", (long long)n_ts, uniq.size());
            } else {
                timesteps = make_input(*eff_ts);
            }
            ggml_tensor* a_timestep = make_optional_input(audio_timesteps_tensor);
            ggml_tensor* context    = make_optional_input(context_tensor);
            // NAG negative text context (null unless a NAG step). Same lifetime contract as
            // `context` (its data lives on the DiffusionParams/extra across the compute call).
            ggml_tensor* nag_context = make_optional_input(nag_context_tensor);

            ggml_cgraph* gf = new_graph_custom(LTXAV_GRAPH_SIZE);

            float video_frame_rate    = frame_rate > 0.f ? frame_rate : 24.f;
            // Separable relip: the DiT sequence is target tokens (vx grid) + reference tokens
            // (vx_ref grid). All PE/positions/token-sel are sized for the COMBINED count; the
            // reference tokens are sliced off in forward() before unpatchify. ref_token_count==0
            // (no vx_ref) => count is the plain target grid (legacy, byte-identical).
            int64_t video_token_count = vx->ne[0] * vx->ne[1] * vx->ne[2] + ref_token_count;
            bool has_video_positions  = !video_positions_tensor.empty();
            if (has_video_positions) {
                GGML_ASSERT(video_positions_tensor.shape()[2] == video_token_count);
                video_pe_vec = build_video_rope_matrix_from_positions(video_positions_tensor,
                                                                      static_cast<int>(config.hidden_size),
                                                                      static_cast<int>(config.num_attention_heads),
                                                                      config.positional_embedding_theta,
                                                                      config.positional_embedding_max_pos,
                                                                      config.use_middle_indices_grid);
            } else {
                video_pe_vec = build_video_rope_matrix(vx->ne[0],
                                                       vx->ne[1],
                                                       vx->ne[2],
                                                       static_cast<int>(config.hidden_size),
                                                       static_cast<int>(config.num_attention_heads),
                                                       video_frame_rate,
                                                       config.positional_embedding_theta,
                                                       config.positional_embedding_max_pos,
                                                       config.vae_scale_factors,
                                                       config.causal_temporal_positioning,
                                                       config.use_middle_indices_grid);
            }
            auto video_pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.attention_head_dim / 2, video_token_count * config.num_attention_heads);
            ggml_set_name(video_pe, "ltxav_video_pe");
            set_backend_tensor_data(video_pe, video_pe_vec.data());

            ggml_tensor* audio_pe       = nullptr;
            ggml_tensor* video_cross_pe = nullptr;
            ggml_tensor* audio_cross_pe = nullptr;
            if (ax != nullptr && ggml_nelements(ax) > 0 && ax->ne[1] > 0) {
                const bool has_audio_positions = !audio_positions_tensor.empty();
                if (has_audio_positions) {
                    GGML_ASSERT(audio_positions_tensor.numel() == ax->ne[1]);
                    std::vector<float> coords(audio_positions_tensor.data(),
                                              audio_positions_tensor.data() + audio_positions_tensor.numel());
                    audio_pe_vec = build_1d_rope_matrix_from_coords(coords,
                                                                      static_cast<int>(config.audio_hidden_size),
                                                                      static_cast<int>(config.audio_num_attention_heads),
                                                                      config.positional_embedding_theta,
                                                                      static_cast<float>(config.audio_positional_embedding_max_pos[0]));
                } else {
                    audio_pe_vec = build_audio_rope_matrix(ax->ne[1],
                                                           static_cast<int>(config.audio_hidden_size),
                                                           static_cast<int>(config.audio_num_attention_heads),
                                                           config.positional_embedding_theta,
                                                           config.audio_positional_embedding_max_pos[0],
                                                           config.use_middle_indices_grid);
                }
                audio_pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.audio_attention_head_dim / 2, ax->ne[1] * config.audio_num_attention_heads);
                ggml_set_name(audio_pe, "ltxav_audio_pe");
                set_backend_tensor_data(audio_pe, audio_pe_vec.data());

                int temporal_max_pos = std::max(config.positional_embedding_max_pos[0], config.audio_positional_embedding_max_pos[0]);
                if (has_video_positions) {
                    video_cross_pe_vec = build_video_temporal_rope_matrix_from_positions(video_positions_tensor,
                                                                                         static_cast<int>(config.audio_cross_attention_dim),
                                                                                         static_cast<int>(config.audio_num_attention_heads),
                                                                                         config.positional_embedding_theta,
                                                                                         temporal_max_pos,
                                                                                         true);
                } else {
                    video_cross_pe_vec = build_video_temporal_rope_matrix(vx->ne[0],
                                                                          vx->ne[1],
                                                                          vx->ne[2],
                                                                          static_cast<int>(config.audio_cross_attention_dim),
                                                                          static_cast<int>(config.audio_num_attention_heads),
                                                                          video_frame_rate,
                                                                          config.positional_embedding_theta,
                                                                          temporal_max_pos,
                                                                          std::get<0>(config.vae_scale_factors),
                                                                          config.causal_temporal_positioning,
                                                                          true);
                }
                video_cross_pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.audio_attention_head_dim / 2, video_token_count * config.audio_num_attention_heads);
                ggml_set_name(video_cross_pe, "ltxav_video_cross_pe");
                set_backend_tensor_data(video_cross_pe, video_cross_pe_vec.data());

                if (has_audio_positions) {
                    std::vector<float> coords(audio_positions_tensor.data(),
                                              audio_positions_tensor.data() + audio_positions_tensor.numel());
                    audio_cross_pe_vec = build_1d_rope_matrix_from_coords(coords,
                                                                            static_cast<int>(config.audio_cross_attention_dim),
                                                                            static_cast<int>(config.audio_num_attention_heads),
                                                                            config.positional_embedding_theta,
                                                                            static_cast<float>(temporal_max_pos));
                } else {
                    audio_cross_pe_vec = build_audio_rope_matrix(ax->ne[1],
                                                                 static_cast<int>(config.audio_cross_attention_dim),
                                                                 static_cast<int>(config.audio_num_attention_heads),
                                                                 config.positional_embedding_theta,
                                                                 temporal_max_pos,
                                                                 true);
                }
                audio_cross_pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.audio_attention_head_dim / 2, ax->ne[1] * config.audio_num_attention_heads);
                ggml_set_name(audio_cross_pe, "ltxav_audio_cross_pe");
                set_backend_tensor_data(audio_cross_pe, audio_cross_pe_vec.data());
            }

            bool needs_video_connector_pe =
                config.use_connector &&
                context != nullptr &&
                (context->ne[0] == config.connector_hidden_size ||
                 ((context->ne[0] == config.cross_attention_dim + config.audio_cross_attention_dim ||
                   context->ne[0] == config.caption_channels * 2) &&
                  context->ne[1] < 1024));
            ggml_tensor* video_connector_pe = nullptr;
            if (needs_video_connector_pe) {
                int64_t seq_len      = context->ne[1];
                int64_t target_len   = std::max<int64_t>(1024, seq_len);
                int64_t duplications = (target_len + config.connector_num_registers - 1) / config.connector_num_registers;
                int64_t full_len     = seq_len + duplications * config.connector_num_registers - seq_len;
                connector_pe_vec     = build_1d_rope_matrix(full_len, static_cast<int>(config.connector_hidden_size), static_cast<int>(config.connector_num_heads), 10000.f, 4096.f, true);
                video_connector_pe   = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.connector_head_dim / 2, full_len * config.connector_num_heads);
                ggml_set_name(video_connector_pe, "ltxav_video_connector_pe");
                set_backend_tensor_data(video_connector_pe, connector_pe_vec.data());
            }

            bool run_audio_context =
                ax != nullptr &&
                ggml_nelements(ax) > 0 &&
                ax->ne[1] > 0;
            bool needs_audio_connector_pe =
                run_audio_context &&
                config.use_audio_connector &&
                context != nullptr &&
                (context->ne[0] == config.audio_connector_hidden_size ||
                 ((context->ne[0] == config.cross_attention_dim + config.audio_cross_attention_dim ||
                   context->ne[0] == config.caption_channels * 2) &&
                  context->ne[1] < 1024));
            ggml_tensor* audio_connector_pe = nullptr;
            if (needs_audio_connector_pe) {
                int64_t seq_len        = context->ne[1];
                int64_t target_len     = std::max<int64_t>(1024, seq_len);
                int64_t duplications   = (target_len + config.audio_connector_num_registers - 1) / config.audio_connector_num_registers;
                int64_t full_len       = seq_len + duplications * config.audio_connector_num_registers - seq_len;
                audio_connector_pe_vec = build_1d_rope_matrix(full_len, static_cast<int>(config.audio_connector_hidden_size), static_cast<int>(config.audio_connector_num_heads), 10000.f, 4096.f, true);
                audio_connector_pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.audio_connector_head_dim / 2, full_len * config.audio_connector_num_heads);
                ggml_set_name(audio_connector_pe, "ltxav_audio_connector_pe");
                set_backend_tensor_data(audio_connector_pe, audio_connector_pe_vec.data());
            }

            auto runner_ctx                 = get_context();
            runner_ctx.ltx_video_token_sel  = v_token_sel_input;  // null unless modulation collapse active
            runner_ctx.ltx_skip_a2v         = skip_a2v_cross_attn_;  // A2V modality-guidance "mod" pass
            // NAG (Normalized Attention Guidance): carry the scale/alpha/tau to the cross-attn.
            // nag_context is null (and these unused) on every non-NAG forward, so the block-level
            // gate `nag_context != nullptr && ctx->ltx_nag_scale != 0` keeps legacy byte-identical.
            runner_ctx.ltx_nag_scale        = nag_scale_;
            runner_ctx.ltx_nag_alpha        = nag_alpha_;
            runner_ctx.ltx_nag_tau          = nag_tau_;
            auto out_pair                   = model.forward(&runner_ctx,
                                            vx,
                                            ax,
                                            timesteps,
                                            a_timestep,
                                            context,
                                            video_pe,
                                            audio_pe,
                                            video_cross_pe,
                                            audio_cross_pe,
                                            video_connector_pe,
                                            audio_connector_pe,
                                            vx_ref,
                                            nag_context);
            auto out        = merge_av_latents(compute_ctx, out_pair.first, out_pair.second);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context         = {},
                                  const sd::Tensor<float>& audio_x         = {},
                                  const sd::Tensor<float>& audio_timesteps = {},
                                  int audio_length                         = 0,
                                  float frame_rate                         = 24.f,
                                  const sd::Tensor<float>& video_positions = {},
                                  const sd::Tensor<float>& audio_positions = {},
                                  const sd::Tensor<float>& video_reference = {},
                                  const sd::Tensor<float>& nag_context     = {}) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, audio_x, audio_timesteps, audio_length, frame_rate, video_positions, audio_positions, video_reference, nag_context);
            };
            auto out = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
            return out;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            const auto* extra   = diffusion_extra_as<LTXAVDiffusionExtra>(diffusion_params);
            skip_a2v_cross_attn_ = extra->skip_a2v;  // consumed by build_graph -> runner_ctx.ltx_skip_a2v
            // NAG: pull the per-step scale/alpha/tau + negative context off the extra. nag_context
            // is null (nag_scale 0) on every non-NAG step, so this is a no-op then.
            nag_scale_ = extra->nag_scale;
            nag_alpha_ = extra->nag_alpha;
            nag_tau_   = extra->nag_tau;
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           tensor_or_empty(extra->audio_x),
                           tensor_or_empty(extra->audio_timesteps),
                           extra->audio_length,
                           extra->frame_rate,
                           tensor_or_empty(extra->video_positions),
                           tensor_or_empty(extra->audio_positions),
                           tensor_or_empty(extra->video_reference),
                           tensor_or_empty(extra->nag_context));
        }

        void test(const std::string& x_path,
                  const std::string& timesteps_path       = "",
                  const std::string& context_path         = "",
                  const std::string& audio_x_path         = "",
                  const std::string& audio_timesteps_path = "") {
            auto x = sd::load_tensor_from_file_as_tensor<float>(x_path);
            GGML_ASSERT(!x.empty());
            print_sd_tensor(x, false, "ltxav_x");

            sd::Tensor<float> timesteps;
            if (!timesteps_path.empty()) {
                timesteps = sd::load_tensor_from_file_as_tensor<float>(timesteps_path);
            } else {
                timesteps = sd::Tensor<float>::from_vector(std::vector<float>{1.f});
            }
            GGML_ASSERT(!timesteps.empty());
            print_sd_tensor(timesteps, false, "ltxav_timesteps");

            sd::Tensor<float> context;
            if (!context_path.empty()) {
                context = sd::load_tensor_from_file_as_tensor<float>(context_path);
                GGML_ASSERT(!context.empty());
                print_sd_tensor(context, false, "ltxav_context");
            }

            sd::Tensor<float> audio_x;
            int audio_length = 0;
            if (!audio_x_path.empty()) {
                audio_x = sd::load_tensor_from_file_as_tensor<float>(audio_x_path);
                GGML_ASSERT(!audio_x.empty());
                GGML_ASSERT(audio_x.dim() >= 2);
                audio_length = static_cast<int>(audio_x.shape()[1]);
                print_sd_tensor(audio_x, false, "ltxav_audio_x");
            }

            sd::Tensor<float> audio_timesteps;
            if (!audio_timesteps_path.empty()) {
                audio_timesteps = sd::load_tensor_from_file_as_tensor<float>(audio_timesteps_path);
                GGML_ASSERT(!audio_timesteps.empty());
            } else if (!audio_x.empty()) {
                audio_timesteps = timesteps;
            }
            if (!audio_timesteps.empty()) {
                print_sd_tensor(audio_timesteps, false, "ltxav_audio_timesteps");
            }

            int64_t t0   = ggml_time_ms();
            auto out_opt = compute(8, x, timesteps, context, audio_x, audio_timesteps, audio_length);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            print_sd_tensor(out_opt, false, "ltxav_out");
            LOG_DEBUG("ltxav test done in %lldms", t1 - t0);
        }

        static void load_from_file_and_test(const std::string& model_path,
                                            const std::string& x_path,
                                            const std::string& timesteps_path       = "",
                                            const std::string& context_path         = "",
                                            const std::string& embeddings_path      = "",
                                            const std::string& audio_x_path         = "",
                                            const std::string& audio_timesteps_path = "") {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend = sd_backend_cpu_init();
            LOG_INFO("loading ltxav from '%s'", model_path.c_str());

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(model_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
                return;
            }
            if (!embeddings_path.empty()) {
                LOG_INFO("loading ltxav embeddings from '%s'", embeddings_path.c_str());
                if (!model_loader.init_from_file(embeddings_path)) {
                    LOG_ERROR("init embeddings model loader from file failed: '%s'", embeddings_path.c_str());
                    return;
                }
            }

            auto& tensor_storage_map           = model_loader.get_tensor_storage_map();
            std::shared_ptr<LTXAVRunner> ltxav = std::make_shared<LTXAVRunner>(backend,
                                                                               backend,
                                                                               tensor_storage_map,
                                                                               "model.diffusion_model");

            if (!ltxav->alloc_params_buffer()) {
                LOG_ERROR("ltxav buffer allocation failed");
                return;
            }
            std::map<std::string, ggml_tensor*> tensors;
            ltxav->get_param_tensors(tensors, "model.diffusion_model");

            if (!model_loader.load_tensors(tensors)) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("ltxav model loaded");
            ltxav->test(x_path, timesteps_path, context_path, audio_x_path, audio_timesteps_path);
        }
    };

};  // namespace LTXV

#endif  // __SD_MODEL_DIFFUSION_LTXV_HPP__
