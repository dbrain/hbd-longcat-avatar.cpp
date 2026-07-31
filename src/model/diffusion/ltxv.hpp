#ifndef __SD_MODEL_DIFFUSION_LTXV_HPP__
#define __SD_MODEL_DIFFUSION_LTXV_HPP__

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

    // LTX_DIT_F16 — run the DiT residual stream (vx/ax) in F16 instead of F32.
    //
    // The DiT is bandwidth-bound on elementwise glue, not on the GEMMs: the AdaLN
    // modulate/gate multiplies, the residual adds, the per-Linear bias adds and the
    // attention layout copies all move the full [hidden x tokens] activation. Halving
    // its element width halves all of that traffic, and it halves the compute-buffer
    // footprint too.
    //
    // ONLY safe when the FP4 GEMM can consume F16 activations directly. If it cannot, every
    // DiT Linear fails ggml_backend_supports_op and ggml_backend_sched drops it to the CPU
    // backend -- so this is a correctness gate, not just a perf one. Hence the device/env
    // probe below rather than a bare env check.
    //
    // Value-honouring: `LTX_DIT_F16=0` must DISABLE it. A presence-only test (getenv() !=
    // nullptr) would silently keep F16 on for anyone bisecting with an explicit 0.
    __STATIC_INLINE__ bool ltx_dit_f16_env() {
        static int v = -1;
        if (v < 0) {
            const char* e = getenv("LTX_DIT_F16");
            v = (e != nullptr && atoi(e) != 0) ? 1 : 0;
        }
        return v == 1;
    }

    // Full gate: the env opt-in AND a device/backend that will actually serve an NVFP4
    // mul_mat with an F16 activation + F16 destination (cuBLASLt FP4, Blackwell-only,
    // GGML_NVFP4_CUBLASLT=1). On an sm86 box (e.g. the 3060 the same image is deployed to)
    // this is false and the well-tested F32 stream runs unchanged — which also keeps the
    // cross-attentions off the F16-Q path that only the Blackwell cuDNN SDPA accepts.
    // Escape hatch for the adapter gate below: LTX_DIT_F16_WITH_ADAPTER=1 restores the old
    // behaviour (F16 stream even under an adapter that cannot consume F16), so the two can be
    // A/B'd on one binary. Do not ship it on without a measurement.
    __STATIC_INLINE__ bool ltx_dit_f16_with_adapter_env() {
        static int v = -1;
        if (v < 0) {
            const char* e = getenv("LTX_DIT_F16_WITH_ADAPTER");
            v             = (e != nullptr && atoi(e) != 0) ? 1 : 0;
        }
        return v == 1;
    }

    __STATIC_INLINE__ bool ltx_dit_f16_enabled(GGMLRunnerContext* ctx) {
        if (!ltx_dit_f16_env()) {
            return false;
        }
        if (ctx == nullptr) {
            return false;
        }
        // 🔴 A RUNTIME LoRA MUST BE ABLE TO EAT THE STREAM IT IS HANDED.
        //
        // This cast is what puts the whole transformer body in F16, and forward_with_lora()
        // then feeds that same activation to the ADAPTER's own tensors as src1. On CUDA an F16
        // src1 against a non-F16, non-cuBLASLt-served src0 is REJECTED by
        // ggml_backend_cuda_device_supports_op(), so a Q8_0 adapter's down-GEMM is not merely
        // slower in the F16 stream — it is a node the CUDA backend says it cannot run.
        //
        // krea2 has had the equivalent gate since bring-up (krea2.hpp: any adapter => F32
        // stream) and measured 3.05 s of what looked like "adapter cost" to be the base DiT
        // losing F16. LTX never had one: the ONLY adapter-aware F16 gate on this model was
        // ltx_attn_f16_out_enabled() below, which covers one tensor per attention.
        //
        // Asking the adapter, rather than blanket-rejecting, is what lets an all-NVFP4
        // rank%64==0 adapter KEEP the F16 stream — which is the entire reason to convert an
        // adapter to NVFP4 in the first place.
        if (ctx->weight_adapter != nullptr && !ctx->weight_adapter->supports_f16_activation() &&
            !ltx_dit_f16_with_adapter_env()) {
            return false;
        }
        return ggml_cuda_nvfp4_f16_dst_available(ctx->backend);
    }

    // GGML_CUDNN_ATTN_F16_OUT -- let the DiT attention hand its output to `to_out.0` as F16
    // instead of F32. Independent of LTX_DIT_F16 on purpose: it pays off in BOTH streams.
    //   * F32 stream: to_out's activation-quant reads half-width and its dst is F16, which
    //     then broadcasts into the F32 residual via the F32,F16->F32 binbcast combo.
    //   * F16 stream (LTX_DIT_F16): it also removes the ONLY F32 island left inside a block --
    //     today the attention returns F32, so to_out falls off the F16-dst gate in
    //     ggml_ext_linear and the residual add pays a full-width F32 read at every one of the
    //     ~6 attentions x 28 blocks per forward.
    // Value-honouring (`=0` disables), same as LTX_DIT_F16.
    __STATIC_INLINE__ bool ltx_attn_f16_out_env() {
        static int v = -1;
        if (v < 0) {
            const char* e = getenv("GGML_CUDNN_ATTN_F16_OUT");
            v             = (e != nullptr && atoi(e) != 0) ? 1 : 0;
        }
        return v == 1;
    }

    // Full gate. An F16 attention output is only ever handed to `to_out.0`, so the question is
    // exactly "will that ONE mul_mat be served with an F16 src1". Answering it per-Linear
    // instead of per-device is what makes this safe to enable on a non-NVFP4 checkpoint:
    // ggml_backend_cuda_device_supports_op() REJECTS an F16 src1 against a non-F16, non-served
    // weight, and a rejected DiT Linear is not slow, it is dropped to the CPU backend.
    //   * device/backend must advertise the cuBLASLt FP4 route (Blackwell + GGML_NVFP4_CUBLASLT)
    //     -- the same probe LTX_DIT_F16 uses. Non-Blackwell / no-cuDNN builds keep today's F32
    //     path, which is also what prod's has_blackwell_mma() gate buys.
    //   * the to_out weight must actually be NVFP4, so ggml_ext_linear's mm_dst gate is
    //     guaranteed to fire and emit the F16 dst rather than asking for an F32 dst from an
    //     F16 activation.
    //   * a weight-adapter must SAY it can take an F16 activation. forward_with_lora() builds
    //     its own delta chain out of the adapter's tensors, and an F16 src1 against a non-NVFP4
    //     src0 is not merely unproven, it is REJECTED by supports_op and dropped to the CPU
    //     backend. WeightAdapter::supports_f16_activation() answers that per adapter (an
    //     all-NVFP4, rank%64==0 adapter says yes; a Q8_0 or rank-32 one says no), which is what
    //     lets an NVFP4 adapter keep the F16 attention output that a Q8_0 one has to give up.
    //     Blanket-rejecting any adapter — as this used to — cost the F16 output on EVERY
    //     adaptered render, including the ones that could have kept it.
    __STATIC_INLINE__ bool ltx_attn_f16_out_enabled(GGMLRunnerContext* ctx, const ggml_tensor* to_out_weight) {
        if (!ltx_attn_f16_out_env()) {
            return false;
        }
        if (ctx == nullptr) {
            return false;
        }
        if (ctx->weight_adapter != nullptr && !ctx->weight_adapter->supports_f16_activation()) {
            return false;
        }
        if (to_out_weight == nullptr || to_out_weight->type != GGML_TYPE_NVFP4) {
            return false;
        }
        return ggml_cuda_nvfp4_f16_dst_available(ctx->backend);
    }

    __STATIC_INLINE__ ggml_tensor* align_token_modulation(ggml_context* ctx,
                                                          ggml_tensor* x,
                                                          ggml_tensor* mod) {
        if (mod != nullptr && x != nullptr && mod->ne[1] == 1 && mod->ne[2] == x->ne[1] && x->ne[2] == 1) {
            return ggml_permute(ctx, mod, 0, 2, 1, 3);
        }
        return mod;
    }

    // Expand a compact [dim, 1, unique_timestep_count] AdaLN chunk back to
    // token order. This is a gather only: the compact modulation remains a
    // graph-cut/cached prelude value until a transformer block consumes it.
    __STATIC_INLINE__ ggml_tensor* gather_mod_tokens(ggml_context* ctx,
                                                      ggml_tensor* c,
                                                      ggml_tensor* sel) {
        if (sel == nullptr) {
            return c;
        }
        const int64_t dim = c->ne[0];
        const int64_t unique_count = c->ne[2];
        auto compact = ggml_reshape_2d(ctx, ggml_cont(ctx, c), dim, unique_count);
        auto gathered = ggml_get_rows(ctx, compact, sel);
        return ggml_reshape_3d(ctx, gathered, dim, 1, sel->ne[0]);
    }

    // Reference AdaLN modulation: out = x + x*scale + shift == 1 MUL + 2 ADD, all at
    // full activation width. Kept as the unfused reference; the live path is modulate_v2().
    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* shift,
                                            ggml_tensor* scale) {
        shift = align_token_modulation(ctx, x, shift);
        scale = align_token_modulation(ctx, x, scale);
        return Flux::modulate(ctx, x, shift, scale, true);
    }

    // Fold the +1 of (1+scale) into an AdaLN scale chunk.
    //
    // MUST be applied exactly once, and MUST be applied to the COMPACT [dim, 1, U]
    // modulation chunk (i.e. before gather_mod_tokens() expands it to per-token order):
    // on the compact table this costs dim*U elements, on the expanded one it would cost
    // dim*L — exactly the full-width ADD that modulate_v2() is removing, i.e. a wash.
    //
    // ggml_scale is F32-only on the CUDA backend, while classic LTX-Video 0.9.x
    // checkpoints can store the scale_shift tables as F16/BF16. Upcast the (tiny) chunk
    // first; a no-op for the F32 tables of LTX-2, so that path is unaffected.
    __STATIC_INLINE__ ggml_tensor* adaln_scale_plus_one(ggml_context* ctx, ggml_tensor* scale) {
        if (scale == nullptr) {
            return nullptr;
        }
        if (scale->type != GGML_TYPE_F32) {
            scale = ggml_cast(ctx, scale, GGML_TYPE_F32);
        }
        return ggml_scale_bias(ctx, scale, 1.0f, 1.0f);  // scale -> scale + 1
    }

    // Reassociated AdaLN modulation: out = x*(1+scale) + shift.
    //
    // `scale_plus1` must ALREADY carry the +1 (applied by adaln_scale_plus_one() on the
    // compact modulation chunk — see get_ada_values()/get_scale_shift_values()).
    //
    // vs modulate(): 1 MUL + 2 ADD -> 1 MUL + 1 ADD, dropping one full-width ADD per
    // modulation site. It also leaves the preceding ggml_rms_norm() output with exactly
    // ONE consumer (modulate() fed it to both the MUL and the first ADD), which is the
    // use-count precondition of the ggml-cuda {RMS_NORM, MUL, ADD} fusion matcher.
    //
    // Numerics: x + x*s + shift and x*(1+s) + shift differ only in rounding. This is the
    // same form the reference fused AdaLN kernel computes (rms_norm(x)*(mul + 1) + shift),
    // so the reassociation is the accepted one.
    __STATIC_INLINE__ ggml_tensor* modulate_v2(ggml_context* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* shift,
                                               ggml_tensor* scale_plus1) {
        shift       = align_token_modulation(ctx, x, shift);
        scale_plus1 = align_token_modulation(ctx, x, scale_plus1);
        return ggml_add(ctx, ggml_mul(ctx, x, scale_plus1), shift);
    }

    __STATIC_INLINE__ ggml_tensor* apply_gate(ggml_context* ctx,
                                              ggml_tensor* x,
                                              ggml_tensor* gate) {
        gate = align_token_modulation(ctx, x, gate);
        return ggml_mul(ctx, x, gate);
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
        // Classic LTX-Video 0.9.x checkpoints are video-only.  Infer this
        // from the absent audio patchifier before constructing any AV blocks.
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
            config.has_audio              = audio_patchify_proj_iter != tensor_storage_map.end();
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
            if (caption_linear1_iter != tensor_storage_map.end() &&
                caption_linear2_iter != tensor_storage_map.end()) {
                config.caption_channels                = caption_linear1_iter->second.ne[0];
                config.caption_proj_before_connector   = false;
                config.caption_projection_first_linear = false;
                // LTX 0.9 uses interleaved rotary embeddings (LTX-2 does not).
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

    // TASS-RoPE source-phase tag (LTX-Best-Face-ID / ST-DRC arXiv:2606.02441).
    //
    // Overlap reference conditioning places the reference latent on the target's
    // frame-0 RoPE grid.  To stop the reference from being confused with the first
    // frame the model is asked to GENERATE, every token carries a per-source
    // multiplicative rotary phase:
    //
    //     phase[d] = source_id * phase_scale * theta^(-d / L)
    //
    // `d` and `L` are PER ATTENTION HEAD, not over the flat rotary half-dim.  Upstream
    // (ComfyUI `freqs_cis_matrix`) reshapes the flat freq vector to
    // [B, T, num_heads, inner_dim / (2*num_heads), 2, 2] before the reference node's
    // `_rotate_ref_block` reads `L = matrix.shape[-3]`, so L = attention_head_dim / 2
    // and the SAME phase ramp is broadcast to every head.  Indexing `d` across the flat
    // half-dim instead gives head h a near-constant phase of seg * theta^(-h/num_heads):
    // the first couple of heads get their whole position encoding rotated by ~2 rad
    // while the remaining heads get no tag at all.
    //
    // Target tokens use source_id = 0, which makes the phase identically zero and
    // therefore an EXACT no-op — a run with no references is bit-identical to the
    // pre-TASS path.  References use source_id = 2, 3, 4, ... so multiple distinct
    // subjects stay separable ("who is who").
    //
    // The phase is an angle offset: RoPE rotates by exp(i*freq), so multiplying by
    // exp(i*phase) is the same as adding phase to the angle before cos/sin.
    __STATIC_INLINE__ float ltxv_tass_source_phase(float source_id, float phase_scale, float theta, int d, int head_half_dim) {
        if (source_id == 0.f || phase_scale == 0.f || head_half_dim <= 0) {
            return 0.f;
        }
        const double ratio = static_cast<double>(d % head_half_dim) / static_cast<double>(head_half_dim);
        return static_cast<float>(static_cast<double>(source_id) * static_cast<double>(phase_scale) *
                                  std::pow(static_cast<double>(theta), -ratio));
    }

    __STATIC_INLINE__ std::vector<float> build_video_rope_matrix_from_positions(const sd::Tensor<float>& positions,
                                                                                int dim,
                                                                                int num_heads,
                                                                                float theta,
                                                                                const std::vector<int>& max_pos,
                                                                                bool use_middle_indices_grid,
                                                                                const std::vector<float>* source_ids = nullptr,
                                                                                float phase_scale                    = 1.f) {
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

        if (source_ids != nullptr) {
            GGML_ASSERT(static_cast<int64_t>(source_ids->size()) == tokens);
        }

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

            // Apply the source-phase tag.  The ramp is per attention head: `d` restarts at
            // every head boundary (see ltxv_tass_source_phase).  source_id == 0
            // short-circuits to a no-op so the untagged path stays bit-identical.
            const float source_id = source_ids != nullptr ? (*source_ids)[static_cast<size_t>(token)] : 0.f;
            // phase_scale == 0 is the UNTAGGED (Echo overlap) layout: skip the whole
            // per-dim loop rather than adding a column of exact zeros.
            if (source_id != 0.f && phase_scale != 0.f) {
                const int head_half_dim = half_dim / std::max(num_heads, 1);
                for (int d = 0; d < half_dim; d++) {
                    freqs[token][d] += ltxv_tass_source_phase(source_id, phase_scale, theta, d, head_half_dim);
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
        // Rope::apply_rope builds its rotation out of ggml_mul/ggml_add against the F32 `pe`
        // table and hands the result straight to attention. Under LTX_DIT_F16 the F16-dst
        // Linears deliver an F16 q/k here, which would (a) carry F16 through the whole
        // rotation and (b) leave attention with an F16 Q that only the Blackwell cuDNN SDPA
        // accepts -- every native ggml flash kernel asserts Q->type == F32
        // (fattn-common.cuh:990). Normalize back to F32 so the RoPE chain and the attention
        // that follows are bit-identical to the F32 stream. (v skips RoPE and stays F16; the
        // attention wrapper casts K/V to F16 anyway.) No-op when x is already F32.
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

    // Target-video self-attention temperature (JoyAI-Echo §3.4). Reference tokens
    // dilute the target's own self-attention; sharpening it back is an inference-only
    // counterweight -- softmax(QK^T / (tau * sqrt(d))) == softmax((Q/tau) K^T / sqrt(d)),
    // so it is a scale on Q and nothing else. tau < 1 sharpens.
    //
    // 1.0 is an EXACT no-op: the scale node is not emitted at all, so the untagged
    // graph stays bit-identical to a build without this knob.
    // Fold factor for the reference-bias mask: how many mask rows each query frame gets.
    //
    // This is a pure COST knob -- every row of a frame carries the same bias, so any divisor is
    // numerically identical. It trades mask size against kernel shape, and the shape is what
    // matters: m = 1 leaves `tokens_per_frame` broadcast query-heads (880 at 1280x704), which
    // MEASURED 2x slower than no mask at all. Picking m so the broadcast lands near 64 costs a few
    // MB of mask and gives the kernel something it likes.
    //
    // `m` must divide tokens_per_frame, so this walks the divisors rather than guessing.
    inline int64_t ltxav_reference_mask_fold(int64_t tokens_per_frame) {
        if (const char* raw = getenv("LTX_REF_MASK_FOLD"); raw != nullptr && *raw != '\0') {
            const long forced = std::strtol(raw, nullptr, 10);
            if (forced > 0 && tokens_per_frame % forced == 0) {
                return forced;
            }
            LOG_WARN("LTX_REF_MASK_FOLD=%s does not divide %lld tokens per frame; auto-selecting",
                     raw, (long long)tokens_per_frame);
        }
        constexpr int64_t kTargetBroadcast = 64;
        int64_t best = 1;
        for (int64_t m = 1; m <= tokens_per_frame; ++m) {
            if (tokens_per_frame % m != 0) {
                continue;
            }
            best = m;
            if (tokens_per_frame / m <= kTargetBroadcast) {
                break;
            }
        }
        return best;
    }

    // How hard the TASS reference block bites: LTX-2's IC-LoRA `conditioning_attention_strength`.
    // Implemented with its real semantics -- an additive log(strength) bias on the reference KEYS,
    // PRE-SOFTMAX -- so reducing it hands the reference's attention mass back to the target rather
    // than merely fading the reference out. See `ref_strength_bias_vec` for how it is made cheap.
    //
    // Useful because reference influence is currently all-or-nothing. Measured earlier: a
    // reference taking 39.5% of the sequence CAPTURES the shot's framing while 11.8% is clean, and
    // the only remedies available today are fewer references, longer shots or a bigger render --
    // all of which change the shot rather than the reference. This turns it down instead.
    //
    // 1.0 is an EXACT no-op: no node is emitted, so the graph stays bit-identical without it.
    inline float ltxav_reference_strength() {
        static const float strength = [] {
            const char* raw = getenv("LTX_REF_STRENGTH");
            if (raw == nullptr || *raw == '\0') {
                return 1.f;
            }
            const float parsed = std::strtof(raw, nullptr);
            if (!(parsed >= 0.f)) {
                LOG_WARN("LTX_REF_STRENGTH=%s is negative; ignoring", raw);
                return 1.f;
            }
            LOG_INFO("LTX_REF_STRENGTH=%.3f (TASS reference contribution)", parsed);
            return parsed;
        }();
        return strength;
    }

    inline float ltxv_video_self_attn_temp() {
        static const float temp = [] {
            const char* raw = getenv("LTX_TASS_VSELF_TEMP");
            if (raw == nullptr || *raw == '\0') {
                return 1.f;
            }
            const float parsed = std::strtof(raw, nullptr);
            if (!(parsed > 0.f)) {
                LOG_WARN("LTX_TASS_VSELF_TEMP=%s is not positive; ignoring", raw);
                return 1.f;
            }
            LOG_INFO("LTX_TASS_VSELF_TEMP=%.3f (target-video self-attention temperature)", parsed);
            return parsed;
        }();
        return temp;
    }

    // ---------------------------------------------------------------------
    // Relay self-mask, SPATIALLY FOLDED (LTX_RELAY_SELF_MASK_FOLD).
    // ---------------------------------------------------------------------
    //
    // Everything the folded path needs to rebuild the video self-attention with
    // the frame axis, rather than the token axis, on Q's query dimension. See
    // ltx_relay.hpp self_mask_fold() for why that is the whole trick. `mask` is
    // null on every path that has not opted in, and the struct is then never
    // constructed at all.
    // LTX_RELAY_SEGMENT_MERGE -- the MASK-FREE key split. See ltx_relay.hpp
    // build_self_segment_bias() for why a piecewise-constant bias can be replaced by
    // per-segment unmasked attention plus a log-sum-exp recombination, and for the
    // measured cost of the mask this exists to avoid.
    struct SelfSegmentPlan {
        // First KEY TOKEN of each segment; strictly increasing, [0] == 0, and the last
        // segment runs to L_k. TASS reference keys sit after the target tokens and are
        // off the timeline, so when they are present they form their own final segment
        // with a zero bias column.
        std::vector<int64_t> start_token;
        // [F_q, n_seg] F32, contiguous, ne0 == query frame: the constant additive bias
        // every key in segment s carries for query frame q. Compact and expanded to the
        // token axis inside each graph-cut segment, exactly like the dense mask -- see
        // expand_self_frame_bias() for why the expansion is GPU-side.
        ggml_tensor* bias        = nullptr;
        int64_t frames           = 0;
        int64_t tokens_per_frame = 0;
        int64_t target_tokens    = 0;
    };

    // FNV-1a over the inputs that determine a RoPE table. Cheap next to the trig it guards.
    inline uint64_t ltxav_rope_cache_key(const sd::Tensor<float>& positions,
                                         const std::vector<float>* source_ids,
                                         float phase_scale,
                                         int64_t token_count,
                                         uint64_t salt) {
        uint64_t h = 1469598103934665603ull ^ salt;
        auto mix = [&h](const void* data, size_t bytes) {
            const auto* p = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < bytes; ++i) {
                h = (h ^ p[i]) * 1099511628211ull;
            }
        };
        mix(&token_count, sizeof(token_count));
        mix(&phase_scale, sizeof(phase_scale));
        if (!positions.empty()) {
            mix(positions.data(), static_cast<size_t>(positions.numel()) * sizeof(float));
        }
        if (source_ids != nullptr && !source_ids->empty()) {
            mix(source_ids->data(), source_ids->size() * sizeof(float));
        }
        // Zero is the "nothing cached yet" sentinel, so never return it.
        return h == 0 ? 1 : h;
    }

    struct SelfAttnPlan {
        // LTX_RELAY_SELF_MASK_FOLD.
        ggml_tensor* mask        = nullptr;  // [L_k, frames*fold] F16, contiguous
        int64_t fold             = 1;        // m: mask rows per frame
        int64_t tokens_per_frame = 0;        // W_lat * H_lat
        int64_t frames           = 0;        // latent frames
        int64_t target_tokens    = 0;        // frames * tokens_per_frame
        // LTX_RELAY_SEGMENT_MERGE. Mutually exclusive with `mask` above: the segment
        // merge exists precisely so that no mask reaches the kernel.
        const SelfSegmentPlan* segments = nullptr;

        bool active() const { return mask != nullptr || segments != nullptr; }
    };

    // Video self-attention with the folded mask.
    //
    // q/k/v arrive exactly as ggml_ext_attention_ext would receive them --
    // [n_head*d_head, L, 1], post-RoPE, q already normalised to F32 -- and the
    // result comes back in the same [C, L, 1] layout, so the caller's gating and
    // to_out are untouched.
    //
    // Query order is preserved by construction: the fold is a pure PERMUTATION of
    // Q's axes (token == frame*tpf + spatial, spatial == a*(tpf/m) + b, so
    // (frame, a) is a contiguous outer axis and b an inner one), and the inverse
    // permutation is applied to the result. No token is reordered relative to any
    // other; only which axis the kernel calls "query" changes.
    //
    // TASS reference tokens sit AFTER the target tokens on the sequence axis and
    // may carry a different spatial grid (ltxv.hpp:2044-2049), so L is not
    // guaranteed to factor. They are therefore run as a second, unmasked
    // attention against the SAME K/V and concatenated back on -- which is what a
    // zero bias row would have produced anyway.
    inline ggml_tensor* ltx_folded_self_attention(GGMLRunnerContext* ctx,
                                                  ggml_tensor* q,
                                                  ggml_tensor* k,
                                                  ggml_tensor* v,
                                                  int64_t n_head,
                                                  int64_t d_head,
                                                  const SelfAttnPlan& fold) {
        ggml_context* gctx = ctx->ggml_ctx;
        const int64_t C    = n_head * d_head;
        const int64_t L    = q->ne[1];
        const int64_t L_k  = k->ne[1];
        const int64_t T    = fold.target_tokens;
        const int64_t R    = L - T;
        const int64_t m    = fold.fold;
        const int64_t rows = fold.frames * m;          // mask rows == folded query count
        const int64_t g    = fold.tokens_per_frame / m;  // queries hidden on the head axis
        const float scale  = 1.0f / std::sqrt((float)d_head);

        // Every reshape below reads q/k/v as flat [token][head][channel]; a
        // non-contiguous projection output would silently reinterpret. No-op on
        // the path that actually runs (Linear returns contiguous).
        q = ggml_ext_cont(gctx, q);
        k = ggml_ext_cont(gctx, k);
        v = ggml_ext_cont(gctx, v);

        // K and V as [d_head, L_k, 1, n_head]: one KV head shared by all `g`
        // fake query heads (GQA), batched over the real heads. The cont here is
        // the same copy ggml_ext_attention_ext already makes at
        // ggml_extend.hpp:1463 -- same bytes, different destination shape.
        auto to_kv = [&](ggml_tensor* t) {
            auto t4 = ggml_reshape_4d(gctx, t, d_head, n_head, L_k, 1);
            t4      = ggml_ext_cont(gctx, ggml_permute(gctx, t4, 0, 3, 1, 2));
            if (t4->type != GGML_TYPE_F16) {
                t4 = ggml_cast(gctx, t4, GGML_TYPE_F16);
            }
            return t4;
        };
        auto k4 = to_kv(k);
        auto v4 = to_kv(v);

        // Target tokens, folded. The prefix view is contiguous (offset 0), so the
        // reshape below is free and the cont costs exactly what the unfolded
        // permute+cont at ggml_extend.hpp:1459 costs.
        auto q_t = ggml_view_3d(gctx, q, C, T, 1, q->nb[1], q->nb[1] * T, 0);
        auto q_f = ggml_reshape_4d(gctx, q_t, d_head, n_head, g, rows);
        q_f      = ggml_ext_cont(gctx, ggml_permute(gctx, q_f, 0, 3, 2, 1));  // [d_head, rows, g, n_head]

        auto out = ggml_flash_attn_ext(gctx, q_f, k4, v4, fold.mask, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        // [d_head, g, rows, n_head] -> [d_head, n_head, g, rows] -> [C, T, 1]
        out = ggml_ext_cont(gctx, ggml_permute(gctx, out, 0, 2, 3, 1));
        out = ggml_reshape_3d(gctx, out, C, T, 1);

        if (R > 0) {
            auto q_r = ggml_view_3d(gctx, q, C, R, 1, q->nb[1], q->nb[1] * R, q->nb[1] * T);
            q_r      = ggml_ext_cont(gctx, q_r);
            auto q_r4 = ggml_reshape_4d(gctx, q_r, d_head, n_head, R, 1);
            q_r4      = ggml_ext_cont(gctx, ggml_permute(gctx, q_r4, 0, 3, 1, 2));  // [d_head, R, 1, n_head]

            auto out_r = ggml_flash_attn_ext(gctx, q_r4, k4, v4, nullptr, scale, 0.f, 0.f);
            ggml_flash_attn_ext_set_prec(out_r, GGML_PREC_F32);
            out_r = ggml_ext_cont(gctx, ggml_permute(gctx, out_r, 0, 2, 3, 1));
            out_r = ggml_reshape_3d(gctx, out_r, C, R, 1);
            out   = ggml_concat(gctx, out, out_r, 1);
        }
        return out;
    }

    // Video self-attention with a MASK-FREE temporal restriction (LTX_RELAY_SEGMENT_MERGE).
    //
    // q/k/v arrive exactly as ggml_ext_attention_ext would receive them -- [C, L, 1],
    // post-RoPE, q already F32 -- and the result comes back in the same layout, so the
    // caller's gating and to_out are untouched.
    //
    // The key axis is cut at the points where the relay's frame-pair bias changes, each
    // piece runs an ORDINARY unmasked attention against the full Q, and the pieces are
    // recombined with the per-segment log-sum-exp:
    //
    //     O = sum_s w_s O_s,   w_s = softmax_s(LSE_s + b_s)
    //
    // which is algebraically the same softmax the mask would have produced -- the segments
    // partition the keys, so the denominators simply add. Every ggml_flash_attn_ext_lse
    // call here passes mask == nullptr and the un-folded BHSD shape, which is what keeps
    // BOTH fast kernels eligible (cuDNN's mask-free window; SA3's q->ne[2] == 32, no GQA).
    //
    // Cost, at 1920x1088x121 (L = 32640, H = 32, D = 128), per attention call:
    //   * the per-segment ggml_ext_cont of K/V is ONE full copy of each in total (the
    //     segments partition the key axis, they do not duplicate it): 510 MiB written,
    //     ~1.07 GB of traffic, ~2.4 ms at 448 GB/s -- against the ~320 s a mask costs over
    //     a whole render, this is noise. It exists because a key-axis slice is contiguous
    //     only WITHIN a head's slab, not across the 32 of them.
    //   * one extra full-size F32 attention output per extra segment, live simultaneously
    //     for the merge (510 MiB each). That, not the copy, is the reason for the
    //     LTX_RELAY_SEGMENT_MAX ceiling.
    //
    // TASS reference tokens need no special case: they are off the timeline, so their bias
    // column is zero, and a query whose bias is equal across every segment recombines to
    // exactly the unsegmented attention.
    inline ggml_tensor* ltx_segmented_self_attention(GGMLRunnerContext* ctx,
                                                     ggml_tensor* q,
                                                     ggml_tensor* k,
                                                     ggml_tensor* v,
                                                     int64_t n_head,
                                                     int64_t d_head,
                                                     const SelfSegmentPlan& seg) {
        ggml_context* gctx  = ctx->ggml_ctx;
        const int64_t C     = n_head * d_head;
        const int64_t L     = q->ne[1];
        const int64_t L_k   = k->ne[1];
        const int64_t n_seg = (int64_t)seg.start_token.size();
        const float scale   = 1.0f / std::sqrt((float)d_head);

        GGML_ASSERT(n_seg >= 2);
        GGML_ASSERT(seg.bias != nullptr && seg.bias->ne[0] == seg.frames && seg.bias->ne[1] == n_seg);

        // Plain BHSD, the same tensors ggml_ext_attention_ext builds: [d_head, L, n_head, 1].
        // No fold -- without a mask the fold has nothing left to buy, and the un-folded shape
        // is the one SA3's contract demands.
        auto to_bhsd = [&](ggml_tensor* t, bool as_f16) {
            auto t4 = ggml_reshape_4d(gctx, t, d_head, n_head, t->ne[1], 1);
            t4      = ggml_ext_cont(gctx, ggml_permute(gctx, t4, 0, 2, 1, 3));
            if (as_f16 && t4->type != GGML_TYPE_F16) {
                t4 = ggml_cast(gctx, t4, GGML_TYPE_F16);
            }
            return t4;
        };
        auto q4 = to_bhsd(q, false);
        auto k4 = to_bhsd(k, true);
        auto v4 = to_bhsd(v, true);

        // Segment s of the compact bias, expanded from one entry per query FRAME to one per
        // query TOKEN. Video tokens are frame-major (patchify_video: token == frame*tpf +
        // spatial), so a single repeat over a size-1 inner axis does it exactly. Reference
        // queries are appended with a zero bias, which is the additive identity.
        auto bias_tokens = [&](int64_t s) {
            auto col = ggml_view_2d(gctx, seg.bias, seg.frames, 1, seg.bias->nb[1], (size_t)s * seg.bias->nb[1]);
            auto e   = ggml_reshape_4d(gctx, col, 1, seg.frames, 1, 1);
            auto rep = ggml_repeat_4d(gctx, e, seg.tokens_per_frame, seg.frames, 1, 1);
            auto flat = ggml_reshape_1d(gctx, rep, seg.target_tokens);
            if (L > seg.target_tokens) {
                flat = ggml_pad(gctx, flat, (int)(L - seg.target_tokens), 0, 0, 0);
            }
            return flat;  // [L]
        };

        std::vector<ggml_tensor*> outs((size_t)n_seg, nullptr);
        ggml_tensor* stacked = nullptr;
        for (int64_t s = 0; s < n_seg; ++s) {
            const int64_t off = seg.start_token[(size_t)s];
            const int64_t len = (s + 1 < n_seg ? seg.start_token[(size_t)(s + 1)] : L_k) - off;
            GGML_ASSERT(len > 0 && off + len <= L_k);

            // A key-axis slice is strided across the head axis, and every consumer here
            // (cuDNN's packed BHSD plan, SA3's packed preprocessing) reads packed pointers,
            // so materialise it. Summed over the segments this is exactly one copy of K/V.
            auto ks = ggml_ext_cont(gctx, ggml_view_4d(gctx, k4, d_head, len, n_head, 1,
                                                       k4->nb[1], k4->nb[2], k4->nb[3],
                                                       (size_t)off * k4->nb[1]));
            auto vs = ggml_ext_cont(gctx, ggml_view_4d(gctx, v4, d_head, len, n_head, 1,
                                                       v4->nb[1], v4->nb[2], v4->nb[3],
                                                       (size_t)off * v4->nb[1]));

            auto node = ggml_flash_attn_ext_lse(gctx, q4, ks, vs, nullptr, scale, 0.f, 0.f);
            ggml_flash_attn_ext_set_prec(node, GGML_PREC_F32);

            outs[(size_t)s] = ggml_flash_attn_ext_lse_out(gctx, node);  // [d_head, n_head, L, 1]
            auto lse        = ggml_flash_attn_ext_lse_stats(gctx, node);  // [L, n_head, 1]

            // LSE_s + b_s, then stacked on a new INNERMOST axis so one soft_max normalises
            // across segments per (query, head) -- that softmax is exactly w_s.
            auto biased = ggml_add(gctx, lse, bias_tokens(s));
            auto row    = ggml_reshape_4d(gctx, biased, 1, L, n_head, 1);
            stacked     = stacked == nullptr ? row : ggml_concat(gctx, stacked, row, 0);
        }

        auto weights = ggml_soft_max(gctx, stacked);  // [n_seg, L, n_head, 1]

        ggml_tensor* out = nullptr;
        for (int64_t s = 0; s < n_seg; ++s) {
            auto w = ggml_view_4d(gctx, weights, 1, L, n_head, 1,
                                  weights->nb[1], weights->nb[2], weights->nb[3],
                                  (size_t)s * weights->nb[0]);
            // [1, L, n_head, 1] -> [1, n_head, L, 1], the axis order of the attention output.
            w        = ggml_ext_cont(gctx, ggml_permute(gctx, w, 0, 2, 1, 3));
            auto term = ggml_mul(gctx, outs[(size_t)s], w);
            out       = out == nullptr ? term : ggml_add(gctx, out, term);
        }

        // The flash output's memory order is already (L, n_head, d_head) with d_head
        // innermost, so this is the same free reshape ggml_ext_attention_ext ends on.
        out = ggml_ext_cont(gctx, out);
        return ggml_reshape_3d(gctx, out, C, L, 1);
    }

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
                             ggml_tensor* context = nullptr,
                             ggml_tensor* mask    = nullptr,
                             ggml_tensor* pe      = nullptr,
                             ggml_tensor* k_pe    = nullptr,
                             float logit_scale    = 1.f,
                             // LTX_RELAY_SELF_MASK_FOLD. Null everywhere except the
                             // video self-attention of an opted-in relay render, so
                             // every other call site keeps the graph it always had.
                             const SelfAttnPlan* fold = nullptr) {
            if (context == nullptr) {
                context = x;
            }

            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            auto q = to_q->forward(ctx, x);
            auto k = to_k->forward(ctx, context);
            auto v = to_v->forward(ctx, context);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            if (pe != nullptr) {
                if (k_pe == nullptr) {
                    k_pe = pe;
                }
                q = apply_hidden_rope(ctx->ggml_ctx, q, pe, heads, dim_head, rope_interleaved);
                k = apply_hidden_rope(ctx->ggml_ctx, k, k_pe, heads, dim_head, rope_interleaved);
            }

            // Every CROSS-attention passes pe == nullptr, so it skips the RoPE cast above and
            // under LTX_DIT_F16 would hand attention an F16 Q. ggml's native CUDA flash
            // kernels assert Q->type == GGML_TYPE_F32 (fattn-common.cuh:990) -- only the
            // Blackwell cuDNN SDPA consumes an F16 Q, and ggml_cuda_get_best_fattn_kernel()
            // never inspects Q->type, so it will happily select a native kernel (whenever
            // GGML_CUDNN_ATTN is off, cuDNN is unavailable, or the shape falls outside
            // cuDNN's mask-free / D in {64,128} window) and then abort mid-render.
            //
            // Normalizing Q to F32 here makes the F16 stream independent of that selection
            // instead of silently depending on it. K/V are unaffected: the attention wrapper
            // casts them to F16 regardless, so they keep the half-width win.
            //
            // GGML_CUDNN_ATTN_F16_OUT does NOT change this. That optimization retypes the
            // attention RESULT (a CPY node after the flash node), never ggml_flash_attn_ext's
            // Q or its destination, so it cannot make an F16 Q reachable by a native kernel.
            // The cast therefore stays exactly as landed, and the "no native fattn kernel ever
            // sees a non-F32 Q" property is preserved unconditionally, on every device, with
            // the flag on or off.
            if (q->type != GGML_TYPE_F32) {
                q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_F32);
            }

            // Attention temperature, applied to Q only. Deliberately AFTER the F32
            // normalization above: ggml_scale is F32-only on CUDA (scale.cu:28-29
            // asserts, no fallback), so scaling an LTX_DIT_F16 Q would abort here.
            if (logit_scale != 1.f) {
                q = ggml_scale(ctx->ggml_ctx, q, logit_scale);
            }

            // GGML_CUDNN_ATTN_F16_OUT opt-in. Decided per Linear (see ltx_attn_f16_out_enabled)
            // because the F16 attention output has exactly one consumer: to_out.0 below.
            const bool f16_out_ok = ltx_attn_f16_out_enabled(ctx, to_out_0->get_weight());

            ggml_tensor* out = nullptr;
            if (fold != nullptr && fold->segments != nullptr) {
                // MASK-FREE: `mask` is deliberately ignored, exactly as in the folded path --
                // the segment split IS the bias, and passing both would apply it twice.
                out = ltx_segmented_self_attention(ctx, q, k, v, heads, dim_head, *fold->segments);
            } else if (fold != nullptr && fold->mask != nullptr) {
                // Same softmax, same bias values, same query order -- only the
                // axis the kernel calls "query" differs. `mask` is deliberately
                // ignored here: the folded mask IS the expanded one, so passing
                // both would double the bias.
                out = ltx_folded_self_attention(ctx, q, k, v, heads, dim_head, *fold);
            } else {
                out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                             ctx->backend,
                                             q,
                                             k,
                                             v,
                                             heads,
                                             mask,
                                             false,
                                             ctx->flash_attn_enabled,
                                             /*kv_scale=*/1.0f,
                                             /*kv_prescaled_f16=*/false,
                                             /*f16_out_ok=*/f16_out_ok);
            }

            if (blocks.count("to_gate_logits") > 0) {
                auto to_gate_logits = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);
                auto gate_logits    = to_gate_logits->forward(ctx, x);
                // ggml_scale is F32-ONLY on CUDA (scale.cu:28-29 asserts, it does not fall
                // back), and under LTX_DIT_F16 this Linear inherits x's F16 and returns an
                // F16 dst -- which would abort in the first attention of the first block.
                // Bring the gate back to F32 before sigmoid so the whole gate computation is
                // numerically identical to the F32 stream. It is [n_head, tokens], i.e. ~1/128
                // of an activation, so the cast is free. Every LTX-2 attention carries
                // to_gate_logits, so this path is NOT optional.
                if (gate_logits->type != GGML_TYPE_F32) {
                    gate_logits = ggml_cast(ctx->ggml_ctx, gate_logits, GGML_TYPE_F32);
                }
                auto gates          = ggml_sigmoid(ctx->ggml_ctx, gate_logits);
                gates               = ggml_ext_scale(ctx->ggml_ctx, gates, 2.0f, true);
                gates               = ggml_reshape_4d(ctx->ggml_ctx, gates, 1, heads, gate_logits->ne[1], gate_logits->ne[2]);

                auto out4 = ggml_reshape_4d(ctx->ggml_ctx, out, dim_head, heads, out->ne[1], out->ne[2]);
                gates     = ggml_repeat(ctx->ggml_ctx, gates, out4);
                out4      = ggml_mul(ctx->ggml_ctx, out4, gates);
                out       = ggml_reshape_3d(ctx->ggml_ctx, out4, heads * dim_head, out4->ne[2], out4->ne[3]);
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
            auto chunks   = ggml_ext_chunk(ctx->ggml_ctx, out, static_cast<int>(coeff), 1);
            // Chunk layout is (shift, scale, gate) per group: msa = 0..2, mlp = 3..5 and,
            // under cross_attention_adaln, cross-attn q = 6..8. Pre-bias the SCALE chunks
            // for modulate_v2(); shift/gate chunks are left untouched.
            chunks[1] = adaln_scale_plus_one(ctx->ggml_ctx, chunks[1]);
            chunks[4] = adaln_scale_plus_one(ctx->ggml_ctx, chunks[4]);
            if (cross_attention_adaln) {
                chunks[7] = adaln_scale_plus_one(ctx->ggml_ctx, chunks[7]);
            }
            return chunks;
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
            auto chunks = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 1);
            // (shift, scale) — pre-bias the SCALE chunk for modulate_v2().
            chunks[1]   = adaln_scale_plus_one(ctx->ggml_ctx, chunks[1]);
            return chunks;
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
            x_norm      = LTXV::modulate_v2(ctx->ggml_ctx, x_norm, shift_msa, scale_msa);
            auto msa    = attn1->forward(ctx, x_norm, nullptr, self_attention_mask, pe);
            x           = ggml_add(ctx->ggml_ctx, x, apply_gate(ctx->ggml_ctx, msa, gate_msa));

            if (cross_attention_adaln) {
                auto shift_q = mods[6];
                auto scale_q = mods[7];
                auto gate_q  = mods[8];

                auto q = rms_norm(ctx->ggml_ctx, x);
                q      = LTXV::modulate_v2(ctx->ggml_ctx, q, shift_q, scale_q);

                auto context_mod = context;
                if (prompt_timestep != nullptr) {
                    auto prompt_mods = get_prompt_scale_shift_values(ctx, prompt_timestep);
                    context_mod      = LTXV::modulate_v2(ctx->ggml_ctx, context_mod, prompt_mods[0], prompt_mods[1]);
                }

                auto mca = attn2->forward(ctx, q, context_mod, attention_mask, nullptr, nullptr);
                x        = ggml_add(ctx->ggml_ctx, x, apply_gate(ctx->ggml_ctx, mca, gate_q));
            } else {
                auto mca = attn2->forward(ctx, x, context, attention_mask, nullptr, nullptr);
                x        = ggml_add(ctx->ggml_ctx, x, mca);
            }

            auto y       = rms_norm(ctx->ggml_ctx, x);
            y            = LTXV::modulate_v2(ctx->ggml_ctx, y, shift_mlp, scale_mlp);
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

    // Post-connector text context length -- the key axis every text
    // cross-attention in the DiT sees. LTX-2 tops the prompt up to this many keys
    // with learnable registers (128 unique vectors, tiled); upstream ComfyUI's
    // Embeddings1DConnector uses the same ceil(max(1024, seq_len)/num_registers).
    //
    // Do NOT shorten it to give Prompt Relay a larger share of the keys. Tried and
    // GPU-disproven 2026-07-28: the registers are the channel the DiT reads the
    // prompt THROUGH, not padding diluting it. Suppressing the block rendered a
    // completely different scene (a one-woman cream-wall prompt came back as two
    // people in a corner) -- sharp and coherent, but unrelated to the prompt.
    constexpr int64_t kConnectorTargetLen = 1024;

    // Post-connector length for a `seq_len`-token context. The register top-up is
    // tiled in whole copies of the register table, so the result is a multiple of
    // num_registers. Shared by the connector, the RoPE-table planner and the relay
    // mask so the three can never disagree about how long the key axis is.
    __STATIC_INLINE__ int64_t ltx_connector_output_len(int64_t seq_len, int64_t num_registers) {
        if (num_registers <= 0 || kConnectorTargetLen <= seq_len) {
            return seq_len;
        }
        const int64_t duplications = (kConnectorTargetLen + num_registers - 1) / num_registers;
        return std::max(duplications * num_registers, seq_len);
    }

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
            int64_t out_len       = ltx_connector_output_len(seq_len, num_learnable_registers);
            int64_t total_to_keep = out_len - seq_len;
            if (total_to_keep <= 0) {
                return hidden_states;
            }
            int64_t duplications = (out_len + num_learnable_registers - 1) / num_learnable_registers;

            auto regs = ggml_reshape_3d(ctx->ggml_ctx, params["learnable_registers"], hidden_size, num_learnable_registers, 1);
            // NVFP4 GGUFs may keep the learned registers in BF16 while the
            // activations are promoted to F32. ggml_concat requires matching
            // element types, so align the registers before expanding them.
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

        // `scale_idx`: indices INTO THE RETURNED vector whose chunk is an AdaLN *scale*
        // destined for modulate_v2(). Those chunks get the +1 of (1+scale) folded in here,
        // while they are still the COMPACT [dim, 1, U] table — i.e. before the optional
        // gather_mod_tokens() expansion to per-token order, which is what keeps the fold
        // cheap. Pass {} (the default) to get the raw chunks for modulate()/apply_gate().
        std::vector<ggml_tensor*> get_ada_values(GGMLRunnerContext* ctx,
                                                 ggml_tensor* table,
                                                 ggml_tensor* timestep,
                                                 int64_t dim,
                                                 int64_t coeff,
                                                 int64_t start = 0,
                                                 int64_t count = -1,
                                                 ggml_tensor* expand_sel = nullptr,
                                                 const std::vector<int>& scale_idx = {}) {
            if (count < 0) {
                count = coeff - start;
            }
            auto t      = ggml_reshape_3d(ctx->ggml_ctx, timestep, dim, coeff, timestep->ne[1]);
            auto s      = ggml_reshape_3d(ctx->ggml_ctx, table, dim, coeff, 1);
            auto e      = ggml_new_tensor_3d(ctx->ggml_ctx, timestep->type, dim, coeff, timestep->ne[1]);
            t           = ggml_repeat(ctx->ggml_ctx, t, e);
            s           = ggml_repeat(ctx->ggml_ctx, s, e);
            auto out    = ggml_add(ctx->ggml_ctx, s, t);
            auto chunks = ggml_ext_chunk(ctx->ggml_ctx, out, static_cast<int>(coeff), 1);
            std::vector<ggml_tensor*> selected(chunks.begin() + start, chunks.begin() + start + count);
            for (int i : scale_idx) {
                GGML_ASSERT(i >= 0 && i < (int) selected.size());
                selected[i] = adaln_scale_plus_one(ctx->ggml_ctx, selected[i]);
            }
            if (expand_sel != nullptr) {
                for (auto& chunk : selected) {
                    chunk = gather_mod_tokens(ctx->ggml_ctx, chunk, expand_sel);
                }
            }
            return selected;
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
                                                ggml_tensor* expand_sel = nullptr) {
            if (cross_attention_adaln) {
                // q_mods = (shift, scale, gate) -> scale at 1
                auto q_mods      = get_ada_values(ctx, table, timestep, dim, 9, 6, 3, expand_sel, {1});
                auto q           = rms_norm(ctx->ggml_ctx, x);
                q                = LTXV::modulate_v2(ctx->ggml_ctx, q, q_mods[0], q_mods[1]);
                auto context_mod = context;
                if (prompt_timestep != nullptr && prompt_table != nullptr) {
                    // p_mods = (shift, scale) -> scale at 1
                    auto p_mods = get_ada_values(ctx, prompt_table, prompt_timestep, dim, 2, 0, -1, nullptr, {1});
                    context_mod = LTXV::modulate_v2(ctx->ggml_ctx, context_mod, p_mods[0], p_mods[1]);
                }
                auto out = attn->forward(ctx, q, context_mod, attention_mask, nullptr, nullptr);
                return apply_gate(ctx->ggml_ctx, out, q_mods[2]);
            }

            auto q = rms_norm(ctx->ggml_ctx, x);
            return attn->forward(ctx, q, context, attention_mask, nullptr, nullptr);
        }

        // `v_attention_mask` biases the VIDEO text cross-attention and
        // `a_attention_mask` the AUDIO one. They are separate parameters
        // because the two streams have different query->time mappings: a video
        // query is a (frame, h, w) patch token and an audio query is an audio
        // latent frame, so one mask cannot serve both.
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* vx,
                                                      ggml_tensor* ax,
                                                      ggml_tensor* v_context,
                                                      ggml_tensor* a_context,
                                                      ggml_tensor* v_attention_mask,
                                                      ggml_tensor* a_attention_mask,
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
                                                      // LTX_RELAY_SELF_MASK_FOLD; mutually exclusive
                                                      // with self_attention_mask above.
                                                      const SelfAttnPlan* self_attention_fold = nullptr) {
            auto attn1               = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
            auto audio_attn1         = std::dynamic_pointer_cast<CrossAttention>(blocks["audio_attn1"]);
            auto attn2               = std::dynamic_pointer_cast<CrossAttention>(blocks["attn2"]);
            auto audio_attn2         = std::dynamic_pointer_cast<CrossAttention>(blocks["audio_attn2"]);
            auto audio_to_video_attn = std::dynamic_pointer_cast<CrossAttention>(blocks["audio_to_video_attn"]);
            auto video_to_audio_attn = std::dynamic_pointer_cast<CrossAttention>(blocks["video_to_audio_attn"]);
            auto ff                  = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
            auto audio_ff            = std::dynamic_pointer_cast<FeedForward>(blocks["audio_ff"]);

            auto v_table = params["scale_shift_table"];
            auto a_table = params["audio_scale_shift_table"];

            bool run_ax  = has_audio && ax != nullptr && ggml_nelements(ax) > 0 && ax->ne[1] > 0;
            bool run_a2v = run_ax && !ctx->ltx_skip_a2v_cross_attn;
            bool run_v2a = run_ax && !ctx->ltx_skip_a2v_cross_attn;

            // v_mods[0..2] = (shift, scale, gate) for the video self-attn -> scale at 1.
            // Entries 3.. are re-fetched separately below (v_ff_mods / q_mods), so only
            // index 1 is consumed here and only index 1 needs the +1 fold.
            auto v_mods = get_ada_values(ctx, v_table, v_timestep, v_dim, cross_attention_adaln ? 9 : 6, 0, -1, ctx->ltx_video_token_sel, {1});
            auto v_norm = rms_norm(ctx->ggml_ctx, vx);
            v_norm      = LTXV::modulate_v2(ctx->ggml_ctx, v_norm, v_mods[0], v_mods[1]);
            // 1/tau: sharpen target-video self-attention against reference-token dilution.
            // Unset (1.0) emits no node at all -- see ltxv_video_self_attn_temp().
            const float v_sa_temp = ltxv_video_self_attn_temp();
            auto v_sa   = attn1->forward(ctx, v_norm, nullptr, self_attention_mask, v_pe, nullptr,
                                         v_sa_temp == 1.f ? 1.f : 1.f / v_sa_temp,
                                         self_attention_fold);
            vx          = ggml_add(ctx->ggml_ctx, vx, apply_gate(ctx->ggml_ctx, v_sa, v_mods[2]));
            auto v_txt  = apply_text_cross_attention(ctx,
                                                     vx,
                                                     v_context,
                                                     attn2.get(),
                                                     v_table,
                                                    cross_attention_adaln ? params["prompt_scale_shift_table"] : nullptr,
                                                     v_timestep,
                                                     v_prompt_timestep,
                                                     v_dim,
                                                     v_attention_mask,
                                                     ctx->ltx_video_token_sel);
            vx          = ggml_add(ctx->ggml_ctx, vx, v_txt);

            if (run_ax) {
                // a_mods[0..2] = (shift, scale, gate) for the audio self-attn -> scale at 1.
                auto a_mods = get_ada_values(ctx, a_table, a_timestep, a_dim, cross_attention_adaln ? 9 : 6, 0, -1, nullptr, {1});
                auto a_norm = rms_norm(ctx->ggml_ctx, ax);
                a_norm      = LTXV::modulate_v2(ctx->ggml_ctx, a_norm, a_mods[0], a_mods[1]);
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
                                                         a_attention_mask);
                ax          = ggml_add(ctx->ggml_ctx, ax, a_txt);

                auto vx_norm3 = rms_norm(ctx->ggml_ctx, vx);
                auto ax_norm3 = rms_norm(ctx->ggml_ctx, ax);

                if (run_a2v) {
                    auto a2v_audio_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_audio"], 1, 0, 4);
                    auto a2v_video_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_video"], 1, 0, 4);
                    // a2v uses chunk layout (scale, shift, ...) -> scale at 0.
                    auto a2v_audio       = get_ada_values(ctx, a2v_audio_table, a_cross_scale_shift_timestep, a_dim, 4, 0, -1, nullptr, {0});
                    auto a2v_video       = get_ada_values(ctx, a2v_video_table, v_cross_scale_shift_timestep, v_dim, 4, 0, -1, nullptr, {0});
                    auto vx_scaled       = LTXV::modulate_v2(ctx->ggml_ctx, vx_norm3, a2v_video[1], a2v_video[0]);
                    auto ax_scaled       = LTXV::modulate_v2(ctx->ggml_ctx, ax_norm3, a2v_audio[1], a2v_audio[0]);
                    auto a2v_out         = audio_to_video_attn->forward(ctx, vx_scaled, ax_scaled, nullptr, v_cross_pe, a_cross_pe);
                    auto a2v_gate_table  = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_video"], 1, 4, 5);
                    auto a2v_gate        = get_ada_values(ctx, a2v_gate_table, v_cross_gate_timestep, v_dim, 1)[0];
                    vx                   = ggml_add(ctx->ggml_ctx, vx, apply_gate(ctx->ggml_ctx, a2v_out, a2v_gate));
                }

                if (run_v2a) {
                    auto v2a_audio_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_audio"], 1, 0, 4);
                    auto v2a_video_table = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_video"], 1, 0, 4);
                    // v2a consumes the SECOND pair of the same 4-wide table: (scale, shift)
                    // at 2,3 -> scale at 2. (Separate get_ada_values() calls from the a2v
                    // branch above, so these are distinct graph nodes — no double-fold.)
                    auto v2a_audio       = get_ada_values(ctx, v2a_audio_table, a_cross_scale_shift_timestep, a_dim, 4, 0, -1, nullptr, {2});
                    auto v2a_video       = get_ada_values(ctx, v2a_video_table, v_cross_scale_shift_timestep, v_dim, 4, 0, -1, nullptr, {2});
                    auto ax_scaled       = LTXV::modulate_v2(ctx->ggml_ctx, ax_norm3, v2a_audio[3], v2a_audio[2]);
                    auto vx_scaled       = LTXV::modulate_v2(ctx->ggml_ctx, vx_norm3, v2a_video[3], v2a_video[2]);
                    auto v2a_out         = video_to_audio_attn->forward(ctx, ax_scaled, vx_scaled, nullptr, a_cross_pe, v_cross_pe);
                    auto v2a_gate_table  = ggml_ext_slice(ctx->ggml_ctx, params["scale_shift_table_a2v_ca_audio"], 1, 4, 5);
                    auto v2a_gate        = get_ada_values(ctx, v2a_gate_table, a_cross_gate_timestep, a_dim, 1)[0];
                    ax                   = ggml_add(ctx->ggml_ctx, ax, apply_gate(ctx->ggml_ctx, v2a_out, v2a_gate));
                }
                // a_ff_mods = chunks 3..5 rebased to (shift, scale, gate) -> scale at 1.
                auto a_ff_mods = get_ada_values(ctx, a_table, a_timestep, a_dim, cross_attention_adaln ? 9 : 6, 3, 3, nullptr, {1});
                auto ax_scaled = rms_norm(ctx->ggml_ctx, ax);
                ax_scaled      = LTXV::modulate_v2(ctx->ggml_ctx, ax_scaled, a_ff_mods[0], a_ff_mods[1]);
                auto a_ff_out  = audio_ff->forward(ctx, ax_scaled);
                ax             = ggml_add(ctx->ggml_ctx, ax, apply_gate(ctx->ggml_ctx, a_ff_out, a_ff_mods[2]));
            }

            // v_ff_mods = chunks 3..5 rebased to (shift, scale, gate) -> scale at 1.
            auto v_ff_mods = get_ada_values(ctx, v_table, v_timestep, v_dim, cross_attention_adaln ? 9 : 6, 3, 3, ctx->ltx_video_token_sel, {1});
            auto vx_scaled = rms_norm(ctx->ggml_ctx, vx);
            vx_scaled      = LTXV::modulate_v2(ctx->ggml_ctx, vx_scaled, v_ff_mods[0], v_ff_mods[1]);
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
            if (config.has_audio && config.use_audio_caption_projection) {
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
            if (config.has_audio && config.use_audio_connector) {
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

        // Expand the relay's compact [F_k, F_q] frame-pair bias into the dense
        // [L_k, L_q] F16 self-attention mask ggml_flash_attn_ext demands.
        //
        // Done with ggml ops, on the GPU, INSIDE each block's graph-cut segment,
        // rather than by uploading a dense CPU-built mask like the text relay
        // does. Two reasons, both structural:
        //   * A graph-cut segment re-uploads every INPUT_EXTERNAL it touches
        //     (ggml_extend.hpp copy_data_to_backend_tensor, called per segment
        //     with the data map preserved). A dense mask is an external input to
        //     all ~48 block segments, so at 1280x704 that is 48 x 378 MiB of
        //     host->device traffic PER DENOISE STEP. The compact seed is 512
        //     bytes, so its 48 uploads are free.
        //   * Built here, the dense tensor is an ordinary intermediate: it dies
        //     with the segment that made it instead of staying resident for the
        //     whole forward.
        //
        // The expansion is exact, not an approximation, because patchify_video
        // lays video tokens out FRAME-MAJOR (token == frame * W*H + spatial), so
        // the dense mask is the compact matrix tensored with an all-ones
        // [tokens_per_frame, tokens_per_frame] block. Two ggml_repeats over a
        // size-1 axis do that; the reshapes between them are views.
        ggml_tensor* expand_self_frame_bias(GGMLRunnerContext* ctx,
                                            ggml_tensor* seed,
                                            int64_t tokens_per_frame,
                                            int64_t ref_tokens) {
            const int64_t key_frames   = seed->ne[0];
            const int64_t query_frames = seed->ne[1];
            const int64_t L_k          = key_frames * tokens_per_frame;

            // [1, F_k, F_q] -> [tpf, F_k, F_q]: broadcast over the KEY's spatial
            // slot, then fold it into ne0 so ne0 == k_frame * tpf + k_spatial.
            auto mask = ggml_reshape_4d(ctx->ggml_ctx, seed, 1, key_frames, query_frames, 1);
            mask      = ggml_repeat_4d(ctx->ggml_ctx, mask, tokens_per_frame, key_frames, query_frames, 1);
            mask      = ggml_reshape_4d(ctx->ggml_ctx, mask, L_k, 1, query_frames, 1);
            // [L_k, 1, F_q] -> [L_k, tpf, F_q]: same trick on the QUERY axis, so
            // ne1 == q_frame * tpf + q_spatial.
            mask = ggml_repeat_4d(ctx->ggml_ctx, mask, L_k, tokens_per_frame, query_frames, 1);
            mask = ggml_reshape_2d(ctx->ggml_ctx, mask, L_k, query_frames * tokens_per_frame);

            if (ref_tokens > 0) {
                // TASS reference tokens are appended after the target tokens and
                // are not on the timeline, so they get zero bias in both
                // directions -- exactly what the no-mask path feeds. ggml_pad
                // fills with zero, which IS the identity for an additive bias.
                mask = ggml_pad(ctx->ggml_ctx, mask, (int)ref_tokens, (int)ref_tokens, 0, 0);
            }
            return mask;
        }

        // The SPATIALLY FOLDED mask: [L_k, frames * fold] instead of [L_k, L_q].
        //
        // Only the KEY axis is expanded to tokens; the query axis keeps one row
        // per (frame, fold slot), which is all the bias ever distinguishes. The
        // attention side then permutes Q so its query dimension is exactly that
        // axis and the remaining tokens_per_frame/fold spatial positions ride the
        // head axis, which ggml_flash_attn_ext DOES broadcast the mask over
        // (ggml.c:5541). Result: 3.98 MiB instead of 2.03 GiB at 1920x1088x121.
        //
        // `fold` rows per frame rather than one, because the CUDA selector tiles
        // the query axis in 64s (fattn.cu:25-37) and a 16-row mask would leave
        // 48/64 of every tile idle. See ltx_relay.hpp self_mask_fold().
        ggml_tensor* fold_self_frame_bias(GGMLRunnerContext* ctx,
                                          ggml_tensor* seed,
                                          int64_t tokens_per_frame,
                                          int64_t fold,
                                          int64_t ref_tokens) {
            const int64_t key_frames   = seed->ne[0];
            const int64_t query_frames = seed->ne[1];
            const int64_t L_k          = key_frames * tokens_per_frame;

            // Key axis: identical to the dense builder.
            auto mask = ggml_reshape_4d(ctx->ggml_ctx, seed, 1, key_frames, query_frames, 1);
            mask      = ggml_repeat_4d(ctx->ggml_ctx, mask, tokens_per_frame, key_frames, query_frames, 1);
            mask      = ggml_reshape_4d(ctx->ggml_ctx, mask, L_k, 1, query_frames, 1);
            // Query axis: `fold` copies of each frame's row, NOT tokens_per_frame.
            // Row index comes out as q_frame * fold + slot, which is the order the
            // attention side's permutation produces.
            mask = ggml_repeat_4d(ctx->ggml_ctx, mask, L_k, fold, query_frames, 1);
            mask = ggml_reshape_2d(ctx->ggml_ctx, mask, L_k, query_frames * fold);

            if (ref_tokens > 0) {
                // Reference KEYS still have to exist on the key axis (the kernel
                // reads a full K->ne[1] row); zero is the additive identity, so
                // padding is exactly "no bias". Reference QUERIES are not here at
                // all -- they get their own unmasked attention call.
                mask = ggml_pad(ctx->ggml_ctx, mask, (int)ref_tokens, 0, 0, 0);
            }
            return mask;
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
                                                                  bool process_audio_context,
                                                                  ggml_tensor* v_connector_mask = nullptr,
                                                                  ggml_tensor* a_connector_mask = nullptr) {
            if (context == nullptr) {
                return {nullptr, nullptr};
            }

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
                v_context      = connector->forward(ctx, v_context, video_connector_pe, v_connector_mask);
            }
            if (process_audio_context &&
                config.use_audio_connector &&
                a_context != nullptr &&
                a_context->ne[0] == config.audio_connector_hidden_size) {
                auto connector = std::dynamic_pointer_cast<Embeddings1DConnector>(blocks["audio_embeddings_connector"]);
                a_context      = connector->forward(ctx, a_context, audio_connector_pe, a_connector_mask);
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
            // (shift, scale) — pre-bias the SCALE chunk for modulate_v2(), while it is
            // still compact (before the optional per-token gather below).
            chunks[1]   = adaln_scale_plus_one(ctx->ggml_ctx, chunks[1]);
            if (expand_sel != nullptr) {
                for (auto& chunk : chunks) {
                    chunk = gather_mod_tokens(ctx->ggml_ctx, chunk, expand_sel);
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
                                                      ggml_tensor* v_relay_mask     = nullptr,
                                                      ggml_tensor* a_relay_mask     = nullptr,
                                                      ggml_tensor* ref_vx           = nullptr,
                                                      ggml_tensor* v_connector_mask = nullptr,
                                                      ggml_tensor* a_connector_mask = nullptr,
                                                      // Compact [F_k, F_q] relay self-attention bias, or null.
                                                      // Null is the only value the production path has ever
                                                      // passed, so the graph is unchanged unless
                                                      // LTX_RELAY_SELF_MASK is set.
                                                      ggml_tensor* v_self_mask_seed = nullptr,
                                                      // TASS reference-strength bias: [L_k, 1] F16,
                                                      // zero on target keys and log(strength) on
                                                      // the reference block's. Null = no-op.
                                                      ggml_tensor* ref_strength_mask = nullptr,
                                                      int64_t ref_strength_fold      = 1,
                                                      // LTX_RELAY_SEGMENT_MERGE. Null unless the
                                                      // env gate is set AND the split fits under
                                                      // LTX_RELAY_SEGMENT_MAX; when it is non-null
                                                      // it REPLACES the mask above, and it falls
                                                      // back to it if the backend cannot serve
                                                      // ggml_flash_attn_ext_lse.
                                                      SelfSegmentPlan* v_self_segments = nullptr) {
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
            int64_t audio_time = (config.has_audio && ax != nullptr) ? ax->ne[1] : 0;

            vx = patchify_video(ctx, vx, n);
            vx = patchify_proj->forward(ctx, vx);

            // TASS overlap references. Patchified with the SAME projection as the
            // target and appended on the token axis (ne[1]); after patchify the
            // sequence is flat, so a reference may carry a different spatial grid
            // (native-resolution 1536x1024 character sheet vs a 768x448 video) with
            // all geometry supplied by the positions + source-phase RoPE tag.
            // Target tokens stay FIRST and contiguous so the tail can be dropped
            // before unpatchify.
            const int64_t target_tokens = width * height * frames;
            if (ref_vx != nullptr && ggml_nelements(ref_vx) > 0) {
                GGML_ASSERT(ref_vx->ne[3] % config.in_channels == 0);
                const int64_t ref_n = ref_vx->ne[3] / config.in_channels;
                GGML_ASSERT(ref_n == n);
                ggml_tensor* rx = patchify_video(ctx, ref_vx, ref_n);
                rx              = patchify_proj->forward(ctx, rx);
                vx              = ggml_concat(ctx->ggml_ctx, vx, rx, 1);
            }

            if (ax != nullptr && ggml_nelements(ax) > 0 && audio_time > 0) {
                ax = patchify_audio(ctx, ax);
                ax = audio_patchify_proj->forward(ctx, ax);
            } else {
                ax = nullptr;
            }

            // LTX_DIT_F16: enter the F16 residual stream. Everything upstream of this point
            // (patchify + patchify_proj) and everything downstream of the matching un-cast
            // before norm_out stays F32, so only the transformer-block body — where all the
            // full-width elementwise glue lives — changes width. The text/audio CONTEXTS are
            // deliberately NOT cast: they feed the cross-attention to_k/to_v, which then keep
            // an F32 dst and so keep the wglobal->alpha fold's F32 shape as well.
            const bool dit_f16 = ltx_dit_f16_enabled(ctx);
            if (dit_f16) {
                vx = ggml_cast(ctx->ggml_ctx, vx, GGML_TYPE_F16);
                if (ax != nullptr) {
                    ax = ggml_cast(ctx->ggml_ctx, ax, GGML_TYPE_F16);
                }
            }

            bool run_ax    = ax != nullptr && ggml_nelements(ax) > 0 && audio_time > 0;
            auto contexts  = preprocess_contexts(ctx, context, video_connector_pe, audio_connector_pe, run_ax,
                                                v_connector_mask, a_connector_mask);
            auto v_context = contexts.first;
            auto a_context = contexts.second != nullptr ? contexts.second : contexts.first;
            if (contexts.second != nullptr) {
                a_context = ggml_cont(ctx->ggml_ctx, a_context);
            }

            auto v_timestep_scaled = ggml_ext_scale(ctx->ggml_ctx, timestep, config.timestep_scale_multiplier);
            auto v_pair            = adaln_single->forward(ctx, v_timestep_scaled);
            auto v_timestep_mod    = v_pair.first;
            auto v_embedded_time   = v_pair.second;

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
                // With video modulation collapse, `timestep` is the compact
                // unique-value table.  A scalar audio timestep is already the
                // correct broadcast for the AV connector; expanding it to the
                // compact width creates an incompatible modulation shape.
                const bool mod_collapse = ctx->ltx_video_token_sel != nullptr;
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
            // The transformer blocks all consume the preprocessed text/audio
            // contexts and AdaLN values below.  Keeping only vx/ax at this
            // boundary makes every block segment walk back through the whole
            // connector/prelude graph, so the weight manager stages several
            // GiB of shared parameters for every block.  Materialize those
            // invariant values with the prelude and pass them as graph-cut
            // inputs instead.  This preserves the graph numerics while making
            // per-block parameter residency reflect the actual block.
            {
                sd::ggml_graph_cut::mark_graph_cut(v_context, "ltxav.prelude", "v_context");
                if (a_context != v_context) {
                    sd::ggml_graph_cut::mark_graph_cut(a_context, "ltxav.prelude", "a_context");
                }
                // These are invariant for every transformer block.  The graph-cut
                // cache aliases views that share a backing activation, so keeping
                // these at the prelude boundary does not multiply their storage.
                sd::ggml_graph_cut::mark_graph_cut(v_timestep_mod, "ltxav.prelude", "v_timestep_mod");
                sd::ggml_graph_cut::mark_graph_cut(v_embedded_time, "ltxav.prelude", "v_embedded_time");
                sd::ggml_graph_cut::mark_graph_cut(a_timestep_mod, "ltxav.prelude", "a_timestep_mod");
                sd::ggml_graph_cut::mark_graph_cut(a_embedded_time, "ltxav.prelude", "a_embedded_time");
                sd::ggml_graph_cut::mark_graph_cut(v_prompt_timestep_mod, "ltxav.prelude", "v_prompt_timestep_mod");
                sd::ggml_graph_cut::mark_graph_cut(a_prompt_timestep_mod, "ltxav.prelude", "a_prompt_timestep_mod");
                sd::ggml_graph_cut::mark_graph_cut(av_ca_video_scale_shift_timestep, "ltxav.prelude", "av_ca_video_scale_shift");
                sd::ggml_graph_cut::mark_graph_cut(av_ca_a2v_gate_noise_timestep, "ltxav.prelude", "av_ca_a2v_gate_noise");
                sd::ggml_graph_cut::mark_graph_cut(av_ca_audio_scale_shift_timestep, "ltxav.prelude", "av_ca_audio_scale_shift");
                sd::ggml_graph_cut::mark_graph_cut(av_ca_v2a_gate_noise_timestep, "ltxav.prelude", "av_ca_v2a_gate_noise");
            }

            // Video self-attention leak block. The seed is null unless
            // LTX_RELAY_SELF_MASK is set AND a relay plan with beats exists, so
            // `v_self_mask` stays null and attn1 keeps the exact argument it has
            // always had. Expanded per block rather than once above the loop
            // because every graph-cut segment reconstructs its own dependencies
            // anyway (ggml_graph_cut.cpp:378-420 walks back from the segment's
            // outputs), so hoisting it would buy nothing and only blur which
            // segment owns the ~378 MiB intermediate.
            // LTX_RELAY_SELF_MASK_FOLD: resolve the fold factor ONCE, before the
            // loop, because it also decides whether the dense expansion above runs
            // at all. `fold` divides tokens_per_frame and is raised until the mask
            // has at least 64 rows, so the CUDA query tiling (fattn.cu:25-37,
            // ncols1 == 64 whenever the GQA path is off, which it always is here
            // because L_k is never a multiple of FATTN_KQ_STRIDE) sees a full tile
            // and the launch geometry matches the unfolded attention exactly.
            // LTX_RELAY_SEGMENT_MERGE: viability probe, once, before the loop.
            //
            // ggml_flash_attn_ext_lse is only implementable where a backend can hand back the
            // softmax denominator -- today that is cuDNN SDPA on Blackwell and nothing else --
            // so ASK rather than assume, exactly like the folded-mask probe below. A rejection
            // here silently falls back to the mask path, which is the reference oracle and is
            // still correct, just slow. These probe tensors are never added to the graph;
            // supports_op reads metadata only.
            if (v_self_segments != nullptr) {
                const int64_t d_head = config.attention_head_dim;
                const int64_t n_head = config.num_attention_heads;
                const int64_t L      = vx->ne[1];
                auto probe_q         = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F32, d_head, L, n_head, 1);
                auto probe_k         = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F16, d_head, L, n_head, 1);
                auto probe           = ggml_flash_attn_ext_lse(ctx->ggml_ctx, probe_q, probe_k, probe_k, nullptr,
                                                     1.f / std::sqrt((float)d_head), 0.f, 0.f);
                ggml_flash_attn_ext_set_prec(probe, GGML_PREC_F32);
                if (!ctx->flash_attn_enabled || !ggml_backend_supports_op(ctx->backend, probe)) {
                    LOG_WARN("ltxav relay segment merge: backend cannot serve ggml_flash_attn_ext_lse "
                             "(flash_attn=%d); falling back to the LTX_RELAY_SELF_MASK path",
                             (int)ctx->flash_attn_enabled);
                    v_self_segments = nullptr;
                } else {
                    LOG_DEBUG("ltxav relay segment merge: %zu key segments, no mask -- "
                              "%.0f MiB of extra F32 attention outputs held live for the merge",
                              v_self_segments->start_token.size(),
                              (double)(v_self_segments->start_token.size() - 1) * (double)d_head *
                                  (double)n_head * (double)L * sizeof(float) / 1048576.0);
                }
            }

            int64_t self_fold = 0;
            if (v_self_segments == nullptr && v_self_mask_seed != nullptr && sd::ltx_relay::self_mask_fold() > 0) {
                const int64_t tpf     = width * height;
                const int64_t frames  = v_self_mask_seed->ne[1];
                const int64_t want    = sd::ltx_relay::self_mask_fold();
                for (int64_t m = want; m <= tpf; ++m) {
                    if (tpf % m == 0 && frames * m >= 64) {
                        self_fold = m;
                        break;
                    }
                }
                if (self_fold == 0) {
                    LOG_WARN("ltxav relay self-mask fold: no divisor of tokens_per_frame=%lld at or above %lld "
                             "reaches 64 rows; falling back to the dense mask",
                             (long long)tpf,
                             (long long)want);
                } else {
                    const int64_t L_k  = frames * tpf + (vx->ne[1] - target_tokens);
                    const int64_t rows = frames * self_fold;
                    // The folded geometry is unusual (one KV head serving
                    // tokens_per_frame/fold query heads, batched over the real
                    // heads), so ASK the backend rather than assume. A rejection
                    // here would otherwise land in ggml_ext_attention_ext's
                    // non-flash fallback, which builds KQ at [L_k, L_q, n_head] --
                    // ~19 GiB at 1280x704 and an instant OOM. These probe tensors
                    // are never added to the graph; supports_op reads metadata only.
                    const int64_t d_head = config.attention_head_dim;
                    const int64_t n_head = config.num_attention_heads;
                    auto probe_q         = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F32, d_head, rows, tpf / self_fold, n_head);
                    auto probe_k         = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F16, d_head, L_k, 1, n_head);
                    auto probe_m         = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F16, L_k, rows, 1, 1);
                    auto probe           = ggml_flash_attn_ext(ctx->ggml_ctx, probe_q, probe_k, probe_k, probe_m,
                                                     1.f / std::sqrt((float)d_head), 0.f, 0.f);
                    ggml_flash_attn_ext_set_prec(probe, GGML_PREC_F32);
                    if (!ctx->flash_attn_enabled || !ggml_backend_supports_op(ctx->backend, probe)) {
                        LOG_WARN("ltxav relay self-mask fold: backend rejects the folded attention shape "
                                 "(flash_attn=%d); falling back to the dense mask",
                                 (int)ctx->flash_attn_enabled);
                        self_fold = 0;
                    }
                }
                if (self_fold > 0) {
                    const int64_t L_k  = frames * tpf + (vx->ne[1] - target_tokens);
                    const int64_t rows = frames * self_fold;
                    LOG_DEBUG("ltxav relay self-mask fold: m=%lld -> mask [%lld,%lld] F16 (%.2f MiB), "
                              "%lld spatial positions per row on the broadcast head axis",
                              (long long)self_fold,
                              (long long)L_k,
                              (long long)rows,
                              (double)L_k * (double)rows * sizeof(ggml_fp16_t) / 1048576.0,
                              (long long)(tpf / self_fold));
                }
            }

            // BACKEND PROBE for the reference-strength bias, mirroring what the relay does for its
            // identical folded path. `ltx_folded_self_attention` calls ggml_flash_attn_ext with no
            // fallback, so a backend that rejects this shape does not degrade gracefully -- the
            // relay's own comment records that a rejection lands in a ~19 GiB KQ allocation. The
            // >= 64 row floor is the relay's too. On rejection we drop the BIAS, never the render:
            // the reference simply keeps full strength.
            bool ref_strength_fold_ok = false;
            if (ref_strength_mask != nullptr) {
                const int64_t probe_rows = ref_strength_mask->ne[1];
                ref_strength_fold_ok     = probe_rows >= 64;
                if (ref_strength_fold_ok) {
                    const int64_t d_head = config.attention_head_dim;
                    const int64_t n_head = config.num_attention_heads;
                    const int64_t L_k    = ref_strength_mask->ne[0];
                    auto probe_q = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F32, d_head, probe_rows,
                                                      (width * height) / ref_strength_fold, n_head);
                    auto probe_k = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F16, d_head, L_k, 1, n_head);
                    auto probe_m = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F16, L_k, probe_rows, 1, 1);
                    auto probe   = ggml_flash_attn_ext(ctx->ggml_ctx, probe_q, probe_k, probe_k, probe_m,
                                                       1.f / std::sqrt((float)d_head), 0.f, 0.f);
                    ggml_flash_attn_ext_set_prec(probe, GGML_PREC_F32);
                    ref_strength_fold_ok = ctx->flash_attn_enabled &&
                                           ggml_backend_supports_op(ctx->backend, probe);
                }
                if (!ref_strength_fold_ok) {
                    LOG_WARN("LTX_REF_STRENGTH ignored: backend rejects the folded attention shape "
                             "(rows=%lld, needs >= 64 and flash-attn). Reference keeps full strength.",
                             (long long)probe_rows);
                }
            }

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<BasicAVTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                ggml_tensor* v_self_mask = nullptr;
                SelfAttnPlan v_self_fold;
                if (v_self_segments != nullptr) {
                    v_self_fold.segments = v_self_segments;
                } else if (v_self_mask_seed != nullptr && self_fold > 0) {
                    v_self_fold.mask             = fold_self_frame_bias(ctx,
                                                                        v_self_mask_seed,
                                                                        width * height,
                                                                        self_fold,
                                                                        vx->ne[1] - target_tokens);
                    v_self_fold.fold             = self_fold;
                    v_self_fold.tokens_per_frame = width * height;
                    v_self_fold.frames           = v_self_mask_seed->ne[1];
                    v_self_fold.target_tokens    = target_tokens;
                } else if (v_self_mask_seed != nullptr) {
                    v_self_mask = expand_self_frame_bias(ctx,
                                                         v_self_mask_seed,
                                                         width * height,
                                                         vx->ne[1] - target_tokens);
                } else if (ref_strength_mask != nullptr && ref_strength_fold_ok) {
                    // Already [L_k, frames * fold], built once outside the block loop.
                    v_self_fold.mask             = ref_strength_mask;
                    v_self_fold.fold             = ref_strength_fold;
                    v_self_fold.tokens_per_frame = width * height;
                    v_self_fold.frames           = frames;
                    v_self_fold.target_tokens    = target_tokens;
                }
                auto out   = block->forward(ctx,
                                            vx,
                                            ax,
                                            v_context,
                                            a_context,
                                            v_relay_mask,
                                            a_relay_mask,
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
                                            v_self_mask,
                                            v_self_fold.active() ? &v_self_fold : nullptr);
                vx         = out.first;
                ax         = out.second;
                sd::ggml_graph_cut::mark_graph_cut(vx, "ltxav.transformer_blocks." + std::to_string(i), "vx");
                sd::ggml_graph_cut::mark_graph_cut(ax, "ltxav.transformer_blocks." + std::to_string(i), "ax");
            }

            // Leave the F16 residual stream before the head: norm_out is a LayerNorm
            // (ggml_norm) and proj_out is an F32 Linear, and the sampler/VAE downstream all
            // expect F32. One cast per forward, vs. the per-op traffic it just saved.
            if (dit_f16 && vx->type != GGML_TYPE_F32) {
                vx = ggml_cast(ctx->ggml_ctx, vx, GGML_TYPE_F32);
            }
            auto v_shift_scale = get_output_scale_shift(ctx, params["scale_shift_table"], v_embedded_time, config.hidden_size, ctx->ltx_video_token_sel);
            vx                 = norm_out->forward(ctx, vx);
            vx                 = LTXV::modulate_v2(ctx->ggml_ctx, vx, v_shift_scale[0], v_shift_scale[1]);
            vx                 = proj_out->forward(ctx, vx);
            // Drop the TASS reference tokens before unpatchify: they were appended
            // after the target tokens, are pure conditioning, and have no place in
            // the decoded w*h*f grid.
            if (vx->ne[1] > target_tokens) {
                vx = ggml_cont(ctx->ggml_ctx,
                               ggml_view_3d(ctx->ggml_ctx, vx,
                                            vx->ne[0], target_tokens, vx->ne[2],
                                            vx->nb[1], vx->nb[2], 0));
            }
            vx                 = unpatchify_video(ctx, vx, width, height, frames);

            if (ax != nullptr && audio_time > 0) {
                if (dit_f16 && ax->type != GGML_TYPE_F32) {
                    ax = ggml_cast(ctx->ggml_ctx, ax, GGML_TYPE_F32);
                }
                auto a_shift_scale = get_output_scale_shift(ctx, params["audio_scale_shift_table"], a_embedded_time, config.audio_hidden_size);
                ax                 = audio_norm_out->forward(ctx, ax);
                ax                 = LTXV::modulate_v2(ctx->ggml_ctx, ax, a_shift_scale[0], a_shift_scale[1]);
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
        // RoPE TABLE CACHE. `build_video_rope_matrix*` is ~25M cos+sin and ~394k allocations over
        // a 12k-token sequence -- measured 310 ms for video_pe and 157 ms for video_cross_pe PER
        // DiT CALL, single-threaded, on the critical path ahead of compute. Positions do not change
        // within a render, so across an 8-step render that is ~3.7 s of pure serial CPU thrown away
        // (and ~4.3 GB of host->device traffic re-uploading identical bytes).
        //
        // The relay masks sitting beside these already cache on a key; these never did. Keyed on
        // everything that feeds the build -- the positions bytes, the source ids, the phase scale
        // and the token count -- so it is BIT-EXACT by construction: a key hit means the inputs
        // were identical, and a miss just rebuilds. Hashing ~316 KB costs ~0.3 ms against 310 ms.
        uint64_t video_pe_key = 0;
        std::vector<float> audio_pe_vec;
        std::vector<float> video_cross_pe_vec;
        uint64_t video_cross_pe_key = 0;
        std::vector<float> audio_cross_pe_vec;
        std::vector<float> connector_pe_vec;
        std::vector<float> audio_connector_pe_vec;
        sd::Tensor<float> vx_input_cache;
        sd::Tensor<float> ax_input_cache;
        // TASS overlap reference latents (may carry a different spatial grid than vx).
        sd::Tensor<float> ref_vx_input_cache;
        // These must outlive graph construction because the selector is an
        // externally-backed graph input.
        sd::Tensor<float> v_timestep_compact_cache;
        std::vector<int32_t> v_token_sel_vec;
        // Prompt Relay masks. Same lifetime requirement, plus a cache: the mask
        // is identical for every step of a window, so it is materialised once
        // per (plan revision, shape) rather than on every graph build.
        std::vector<ggml_fp16_t> relay_video_mask_vec;
        std::vector<ggml_fp16_t> relay_audio_mask_vec;
        std::vector<ggml_fp16_t> connector_isolate_vec;
        std::pair<uint64_t, int64_t> connector_isolate_key{0, 0};
        // Post-connector beat ownership. Without piece isolation this is just the
        // conditioner's token map (registers stay global); with it, the registers
        // are split per piece so relay can address them.
        std::vector<int32_t> relay_key_beat;
        std::pair<uint64_t, int64_t> relay_key_beat_key{0, 0};
        std::tuple<uint64_t, int64_t, int64_t> relay_video_key{0, 0, 0};
        std::tuple<uint64_t, int64_t, int64_t> relay_audio_key{0, 0, 0};
        // Compact [F_k, F_q] video self-attention bias (LTX_RELAY_SELF_MASK).
        // Keyed on latent frames rather than a token count: the dense mask this
        // seeds is derived, so the seed is invariant across spatial resolution.
        std::vector<ggml_fp16_t> relay_self_bias_vec;
        // TASS reference-strength bias, PRE-SOFTMAX. One F16 per key: zero for the target's keys
        // and log(strength) for the reference block's, so the softmax weight on every reference
        // key is multiplied by `strength` and the mass it loses is redistributed across the
        // target's own keys. That redistribution is what makes this the real
        // `conditioning_attention_strength` rather than a post-softmax attenuation.
        //
        // It rides the RELAY'S FOLDED-MASK PATH with fold m = 1, which is what makes it
        // affordable: the kernel then sees `rows = latent frames` queries instead of L, so the
        // mask is [L_k, frames] -- ~1.7 MiB at 13k tokens -- rather than the [L_k, L] dense form,
        // which is ~350 MB and is re-uploaded per step. The bias depends only on the KEY index, so
        // every query row is identical and the fold costs nothing in accuracy.
        std::vector<ggml_fp16_t> ref_strength_bias_vec;
        std::pair<uint64_t, int64_t> relay_self_key{0, 0};
        // LTX_RELAY_SEGMENT_MERGE: the key-frame cuts and the [F_q, n_seg] F32 segment bias,
        // cached against the same {revision, frames} key as the mask above. The extra
        // all-zero column for the TASS reference keys is part of the cached vector, so the
        // cache key also carries whether references are present.
        std::vector<int64_t> relay_seg_cut_frames;
        std::vector<float> relay_seg_bias_vec;
        std::tuple<uint64_t, int64_t, int64_t> relay_seg_key{0, 0, 0};

        // LTX_RELAY_ISOLATE partitions the connector's registers across the
        // pieces; without it the map is the conditioner's token map and every
        // register stays global (the historical behaviour).
        const std::vector<int32_t>& relay_key_beat_for(const sd::ltx_relay::Plan* relay, int64_t length) {
            if (!sd::ltx_relay::isolate_enabled()) {
                return relay->token_beat;
            }
            if (relay_key_beat_key != std::make_pair(relay->revision, length)) {
                sd::ltx_relay::build_connector_key_beat(relay->token_beat,
                                                        relay->beats.size(),
                                                        length,
                                                        relay_key_beat);
                relay_key_beat_key = {relay->revision, length};
            }
            return relay_key_beat;
        }

        static std::tuple<uint64_t, int64_t, int64_t> relay_mask_key(const sd::ltx_relay::Plan* relay,
                                                                    int64_t L_q,
                                                                    int64_t L_k) {
            return {relay->revision, L_q, L_k};
        }

        LTXAVRunner(ggml_backend_t backend,
                    const String2TensorStorage& tensor_storage_map      = {},
                    const std::string& prefix                           = "model.diffusion_model",
                    std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
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
                                 bool skip_a2v                                  = false,
                                 const sd::ltx_relay::Plan* relay               = nullptr,
                                 const sd::Tensor<float>& ref_video_x_tensor    = {},
                                 const std::vector<float>* video_source_ids     = nullptr,
                                 float tass_phase_scale                         = 1.f) {
            auto split_inputs = split_av_latents(x_tensor, audio_length);
            vx_input_cache    = split_inputs.first;
            if (!audio_x_tensor.empty()) {
                ax_input_cache = audio_x_tensor;
            } else {
                ax_input_cache = split_inputs.second;
            }

            ggml_tensor* vx         = make_input(vx_input_cache);
            // TASS overlap references: a separate latent block carrying its own spatial
            // grid. Kept out of vx so the target keeps a clean w*h*f grid for unpatchify.
            ref_vx_input_cache      = ref_video_x_tensor;
            ggml_tensor* ref_vx     = make_optional_input(ref_vx_input_cache);
            ggml_tensor* ax         = make_optional_input(ax_input_cache);
            ggml_tensor* timesteps  = nullptr;
            ggml_tensor* v_token_sel_input = nullptr;
            // Production-fork default: scalar T2V is unchanged; masked
            // continuation uses compact modulation unless explicitly disabled.
            bool collapse_enabled = true;
            if (const char* value = std::getenv("LTX_MOD_COLLAPSE")) {
                collapse_enabled = value[0] != '0';
            }
            const bool no_dedup = std::getenv("LTX_MOD_NO_DEDUP") != nullptr;
            const int64_t n_timesteps = static_cast<int64_t>(timesteps_tensor.numel());
            if (collapse_enabled && n_timesteps > 1) {
                const float* timestep_data = timesteps_tensor.data();
                std::vector<float> unique_timesteps;
                v_token_sel_vec.resize(static_cast<size_t>(n_timesteps));
                for (int64_t i = 0; i < n_timesteps; ++i) {
                    const float timestep = timestep_data[i];
                    int32_t unique_index = -1;
                    if (!no_dedup) {
                        for (size_t j = 0; j < unique_timesteps.size(); ++j) {
                            if (unique_timesteps[j] == timestep) {
                                unique_index = static_cast<int32_t>(j);
                                break;
                            }
                        }
                    }
                    if (unique_index < 0) {
                        unique_index = static_cast<int32_t>(unique_timesteps.size());
                        unique_timesteps.push_back(timestep);
                    }
                    v_token_sel_vec[static_cast<size_t>(i)] = unique_index;
                }
                v_timestep_compact_cache = sd::Tensor<float>({static_cast<int64_t>(unique_timesteps.size())}, unique_timesteps);
                timesteps = make_input(v_timestep_compact_cache);
                v_token_sel_input = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_timesteps);
                ggml_set_name(v_token_sel_input, "ltxav_video_token_sel");
                set_backend_tensor_data(v_token_sel_input, v_token_sel_vec.data());
                LOG_DEBUG("ltxav modulation collapse: %lld video tokens -> %zu unique timesteps", (long long)n_timesteps, unique_timesteps.size());
            } else {
                timesteps = make_input(timesteps_tensor);
            }
            ggml_tensor* a_timestep = make_optional_input(audio_timesteps_tensor);
            // The audio/prompt AdaLN path is indexed independently from video
            // tokens.  It must retain the legacy per-frame timeline when the
            // video path is compacted; otherwise its text-context broadcast
            // sees the compact U=2 width and cannot repeat it safely.
            if (a_timestep == nullptr && collapse_enabled && n_timesteps > 1) {
                a_timestep = make_input(timesteps_tensor);
            }
            ggml_tensor* context    = make_optional_input(context_tensor);

            ggml_cgraph* gf = new_graph_custom(LTXAV_GRAPH_SIZE);

            float video_frame_rate    = frame_rate > 0.f ? frame_rate : 24.f;
            int64_t target_token_count = vx->ne[0] * vx->ne[1] * vx->ne[2];
            // TASS reference tokens are appended after the target tokens, so every
            // per-token vector (positions, source ids, timesteps, RoPE rows) is sized
            // over target ++ reference.
            int64_t ref_token_count   = (ref_vx != nullptr && ggml_nelements(ref_vx) > 0)
                                            ? ref_vx->ne[0] * ref_vx->ne[1] * ref_vx->ne[2]
                                            : 0;
            int64_t video_token_count = target_token_count + ref_token_count;
            bool has_video_positions  = !video_positions_tensor.empty();
            GGML_ASSERT(ref_token_count == 0 || has_video_positions);
            // Skip the rebuild when nothing that feeds it changed -- see `video_pe_key`. The whole
            // guarded region is pure: same inputs, same bytes.
            const uint64_t want_video_pe_key =
                has_video_positions
                    ? ltxav_rope_cache_key(video_positions_tensor, video_source_ids, tass_phase_scale,
                                           video_token_count, 0x11)
                    : ltxav_rope_cache_key({}, nullptr, video_frame_rate,
                                           (vx->ne[0] * 73856093) ^ (vx->ne[1] * 19349663) ^ (vx->ne[2] * 83492791),
                                           0x12);
            const bool video_pe_hit = video_pe_key == want_video_pe_key && !video_pe_vec.empty();
            if (video_pe_hit) {
                // nothing to do: video_pe_vec already holds exactly these bytes
            } else if (has_video_positions) {
                GGML_ASSERT(video_positions_tensor.shape()[2] == video_token_count);
                if (video_source_ids != nullptr) {
                    GGML_ASSERT(static_cast<int64_t>(video_source_ids->size()) == video_token_count);
                }
                video_pe_vec = build_video_rope_matrix_from_positions(video_positions_tensor,
                                                                      static_cast<int>(config.hidden_size),
                                                                      static_cast<int>(config.num_attention_heads),
                                                                      config.positional_embedding_theta,
                                                                      config.positional_embedding_max_pos,
                                                                      config.use_middle_indices_grid,
                                                                      video_source_ids,
                                                                      tass_phase_scale);
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
            video_pe_key = want_video_pe_key;
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
                    audio_pe_vec = build_1d_rope_matrix_from_coords(
                        coords,
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
                // Same cache as video_pe -- a different salt so the two keys cannot alias, and the
                // temporal max joins the key because it is the one config value that can differ
                // between otherwise identical calls.
                const uint64_t want_cross_key =
                    has_video_positions
                        ? ltxav_rope_cache_key(video_positions_tensor, nullptr,
                                               static_cast<float>(temporal_max_pos), video_token_count, 0x21)
                        : ltxav_rope_cache_key({}, nullptr, video_frame_rate,
                                               (vx->ne[0] * 73856093) ^ (vx->ne[1] * 19349663) ^ (vx->ne[2] * 83492791),
                                               0x22);
                if (video_cross_pe_key == want_cross_key && !video_cross_pe_vec.empty()) {
                    // cached
                } else if (has_video_positions) {
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
                video_cross_pe_key = want_cross_key;
                video_cross_pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.audio_attention_head_dim / 2, video_token_count * config.audio_num_attention_heads);
                ggml_set_name(video_cross_pe, "ltxav_video_cross_pe");
                set_backend_tensor_data(video_cross_pe, video_cross_pe_vec.data());

                if (has_audio_positions) {
                    std::vector<float> coords(audio_positions_tensor.data(),
                                              audio_positions_tensor.data() + audio_positions_tensor.numel());
                    audio_cross_pe_vec = build_1d_rope_matrix_from_coords(
                        coords,
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
            // Key length the video text cross-attention will actually see. The
            // connector appends learnable registers, so the post-connector
            // sequence is longer than the conditioner's token count; the relay
            // mask has to be built against this, not against context->ne[1].
            int64_t video_context_len = context != nullptr ? context->ne[1] : 0;
            if (needs_video_connector_pe) {
                int64_t seq_len   = context->ne[1];
                int64_t full_len  = ltx_connector_output_len(seq_len, config.connector_num_registers);
                video_context_len = full_len;
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
            int64_t audio_context_len       = context != nullptr ? context->ne[1] : 0;
            if (needs_audio_connector_pe) {
                int64_t seq_len   = context->ne[1];
                int64_t full_len  = ltx_connector_output_len(seq_len, config.audio_connector_num_registers);
                audio_context_len = full_len;
                audio_connector_pe_vec = build_1d_rope_matrix(full_len, static_cast<int>(config.audio_connector_hidden_size), static_cast<int>(config.audio_connector_num_heads), 10000.f, 4096.f, true);
                audio_connector_pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.audio_connector_head_dim / 2, full_len * config.audio_connector_num_heads);
                ggml_set_name(audio_connector_pe, "ltxav_audio_connector_pe");
                set_backend_tensor_data(audio_connector_pe, audio_connector_pe_vec.data());
            }

            // Connector self-attention isolation. Off unless LTX_RELAY_ISOLATE=1,
            // and null without a relay plan, so the ordinary graph is untouched.
            ggml_tensor* v_connector_mask = nullptr;
            ggml_tensor* a_connector_mask = nullptr;
            if (sd::ltx_relay::isolate_enabled() && relay != nullptr && !relay->token_beat.empty() && context != nullptr) {
                if (needs_video_connector_pe && video_context_len > 0) {
                    if (connector_isolate_key != std::make_pair(relay->revision, video_context_len)) {
                        sd::ltx_relay::build_connector_isolation_mask_f16(relay_key_beat_for(relay, video_context_len),
                                                                          video_context_len,
                                                                          connector_isolate_vec);
                        connector_isolate_key = {relay->revision, video_context_len};
                    }
                    v_connector_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F16, video_context_len, video_context_len);
                    ggml_set_name(v_connector_mask, "ltxav_connector_isolate_video");
                    set_backend_tensor_data(v_connector_mask, connector_isolate_vec.data());
                }
                if (needs_audio_connector_pe && audio_context_len == video_context_len && v_connector_mask != nullptr) {
                    // Same token map and same length, so the video mask is the
                    // audio mask; sharing it keeps one upload instead of two.
                    a_connector_mask = v_connector_mask;
                }
            }

            // Prompt Relay masks. Built on CPU, uploaded pre-shaped
            // ([L_k, L_q], ne1 == L_q) and pre-F16 so that
            // ggml_ext_attention_ext's ggml_repeat + ggml_cast fallback never
            // fires -- that fallback is the difference between one mask and
            // ~48x its size in graph churn. A null plan leaves both null and
            // the graph identical to a build without relay.
            ggml_tensor* v_relay_mask = nullptr;
            ggml_tensor* a_relay_mask = nullptr;
            if (relay != nullptr && context != nullptr) {
                const int64_t tokens_per_frame = vx->ne[0] * vx->ne[1];
                if (relay->has_video() && video_context_len > 0 &&
                    static_cast<int64_t>(relay->video_frame_time.size()) == vx->ne[2]) {
                    // The mask is built over the TARGET frames only. TASS reference
                    // tokens are appended after them and get zero bias rows: a
                    // reference is not on the timeline, so no beat should penalise
                    // its view of the prompt. Zero rows are also exactly what the
                    // no-relay path feeds, so relay + references compose.
                    const int64_t L_q = static_cast<int64_t>(relay->video_frame_time.size()) * tokens_per_frame +
                                        ref_token_count;
                    if (relay_video_key != relay_mask_key(relay, L_q, video_context_len)) {
                        sd::ltx_relay::build_mask_f16(*relay,
                                                      relay_key_beat_for(relay, video_context_len),
                                                      relay->video_frame_time,
                                                      tokens_per_frame,
                                                      video_context_len,
                                                      relay->eps,
                                                      relay_video_mask_vec);
                        relay_video_mask_vec.resize(static_cast<size_t>(L_q * video_context_len),
                                                    ggml_fp32_to_fp16(0.f));
                        relay_video_key = relay_mask_key(relay, L_q, video_context_len);
                        LOG_DEBUG("ltxav prompt relay: video mask [%lld,%lld] F16 (%.1f MiB), %zu beats",
                                  (long long)video_context_len,
                                  (long long)L_q,
                                  relay_video_mask_vec.size() * sizeof(ggml_fp16_t) / 1048576.0,
                                  relay->beats.size());
                    }
                    v_relay_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F16, video_context_len, L_q);
                    ggml_set_name(v_relay_mask, "ltxav_relay_video_mask");
                    set_backend_tensor_data(v_relay_mask, relay_video_mask_vec.data());
                } else if (relay->has_video()) {
                    LOG_WARN("ltxav prompt relay: video mask skipped (frame table %zu vs %lld latent frames)",
                             relay->video_frame_time.size(),
                             (long long)vx->ne[2]);
                }
                if (ax != nullptr && relay->has_audio() && audio_context_len > 0 &&
                    static_cast<int64_t>(relay->audio_frame_time.size()) == ax->ne[1]) {
                    const int64_t L_q = static_cast<int64_t>(relay->audio_frame_time.size());
                    if (relay_audio_key != relay_mask_key(relay, L_q, audio_context_len)) {
                        sd::ltx_relay::build_mask_f16(*relay,
                                                      relay_key_beat_for(relay, audio_context_len),
                                                      relay->audio_frame_time,
                                                      1,
                                                      audio_context_len,
                                                      relay->audio_eps > 0.f ? relay->audio_eps : relay->eps,
                                                      relay_audio_mask_vec);
                        relay_audio_key = relay_mask_key(relay, L_q, audio_context_len);
                    }
                    a_relay_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F16, audio_context_len, L_q);
                    ggml_set_name(a_relay_mask, "ltxav_relay_audio_mask");
                    set_backend_tensor_data(a_relay_mask, relay_audio_mask_vec.data());
                }
            }

            // Relay temporal SELF-attention bias (LTX_RELAY_SELF_MASK), EXPERIMENTAL.
            //
            // Uploaded COMPACT, one entry per (key frame, query frame); the DiT
            // expands it to the dense [L, L] mask inside each block's segment.
            // See ltx_relay.hpp for why the bias is one-sided and
            // LTXAVModelBlock::expand_self_frame_bias for why the expansion is
            // GPU-side. Deliberately independent of relay->token_beat, so the
            // self-attention half can be A/B'd against the text half alone.
            //
            // Off (unset / <= 0) allocates nothing and never names a tensor, so
            // the graph is byte-identical to a build without this feature -- which
            // matters here because relay renders are gated on a graph-shape
            // fingerprint.
            ggml_tensor* v_self_mask_seed = nullptr;
            if (relay != nullptr && sd::ltx_relay::self_mask_strength() > 0.f && !relay->beats.empty() &&
                static_cast<int64_t>(relay->video_frame_time.size()) == vx->ne[2]) {
                const int64_t frames           = vx->ne[2];
                const int64_t tokens_per_frame = vx->ne[0] * vx->ne[1];
                const int64_t L                = frames * tokens_per_frame + ref_token_count;
                const double dense_mib         = (double)L * (double)L * sizeof(ggml_fp16_t) / 1048576.0;
                // LTX_RELAY_SELF_MASK_FOLD never materialises the dense mask, so
                // the MiB refusal below does not apply to it. The DiT still
                // downgrades to dense (and hence back under this ceiling) if the
                // backend rejects the folded shape, so the ceiling is not lost --
                // it just stops being the thing that decides.
                const bool folded = sd::ltx_relay::self_mask_fold() > 0;
                if (!folded && dense_mib > (double)sd::ltx_relay::self_mask_max_mib()) {
                    // Refuse rather than OOM mid-render: ggml_ext_attention_ext
                    // casts the mask again on the way into flash attention, so the
                    // live peak is twice what is printed here.
                    LOG_WARN("ltxav relay self-mask: REFUSED, dense [%lld,%lld] F16 needs %.0f MiB (x2 live) > "
                             "LTX_RELAY_SELF_MASK_MAX_MIB=%lld",
                             (long long)L,
                             (long long)L,
                             dense_mib,
                             (long long)sd::ltx_relay::self_mask_max_mib());
                } else {
                    if (relay_self_key != std::make_pair(relay->revision, frames)) {
                        sd::ltx_relay::build_self_frame_bias_f16(*relay,
                                                                 sd::ltx_relay::self_mask_strength(),
                                                                 relay_self_bias_vec);
                        relay_self_key = {relay->revision, frames};
                        LOG_DEBUG("ltxav relay self-mask: seed [%lld,%lld] F16 -> dense [%lld,%lld] (%.0f MiB), "
                                  "%zu beats, strength %.3f",
                                  (long long)frames,
                                  (long long)frames,
                                  (long long)L,
                                  (long long)L,
                                  dense_mib,
                                  relay->beats.size(),
                                  sd::ltx_relay::self_mask_strength());
                    }
                    v_self_mask_seed = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F16, frames, frames);
                    ggml_set_name(v_self_mask_seed, "ltxav_relay_self_bias");
                    set_backend_tensor_data(v_self_mask_seed, relay_self_bias_vec.data());
                }
            }

            // TASS reference-strength bias. Built only when there IS a reference block and the
            // strength is off its no-op, and only when the relay is not already driving the video
            // self-attention -- the two would otherwise both claim `SelfAttnPlan`, and the relay's
            // bias is per-frame-pair where this one is per-key, so they cannot simply be added
            // without deciding whose fold factor wins. The relay keeps priority; a render asking
            // for both logs and takes the relay.
            ggml_tensor* ref_strength_mask = nullptr;
            int64_t ref_strength_fold      = 1;
            const float ref_strength       = ltxav_reference_strength();
            if (ref_token_count > 0 && ref_strength != 1.f) {
                const int64_t frames           = vx->ne[2];
                const int64_t tokens_per_frame = vx->ne[0] * vx->ne[1];
                const int64_t L_k              = tokens_per_frame * frames + ref_token_count;
                const int64_t target           = L_k - ref_token_count;
                ref_strength_fold              = ltxav_reference_mask_fold(tokens_per_frame);
                // log(0) is -inf, the additive identity for "cannot attend". It is clamped, and
                // the clamp VALUE matters: -1e4 produced pure garbage in the suppressed frame even
                // though it is representable in F16. exp(-30) is ~1e-13, which is zero for every
                // practical purpose while staying in the range the kernel's accumulation handles,
                // and it is what the usual masking convention uses. Do not make this more extreme.
                auto to_bias = [](float s) { return s > 0.f ? std::max(std::log(s), -30.f) : -30.f; };

                // ONE key-row, uploaded and repeated ON DEVICE: the host sends L_k halves (~26 KB)
                // rather than the whole mask (~5 MB), and the earlier attention-mask work measured
                // re-upload as the dominant cost.
                ref_strength_bias_vec.assign(static_cast<size_t>(L_k), ggml_fp32_to_fp16(0.f));
                for (int64_t i = target; i < L_k; ++i) {
                    ref_strength_bias_vec[static_cast<size_t>(i)] = ggml_fp32_to_fp16(to_bias(ref_strength));
                }
                auto key_row = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F16, L_k, 1);
                ggml_set_name(key_row, "ltxav_ref_strength_row");
                set_backend_tensor_data(key_row, ref_strength_bias_vec.data());
                ref_strength_mask = ggml_repeat_4d(compute_ctx, key_row, L_k, frames * ref_strength_fold, 1, 1);
                LOG_INFO("ltxav reference strength %.3f, fold m=%lld -> "
                         "mask [%lld,%lld] F16 (%.2f MiB), %lld broadcast query-heads",
                         ref_strength,
                         (long long)ref_strength_fold,
                         (long long)L_k,
                         (long long)(frames * ref_strength_fold),
                         (double)L_k * (double)(frames * ref_strength_fold) * sizeof(ggml_fp16_t) / 1048576.0,
                         (long long)(tokens_per_frame / ref_strength_fold));
            }

            // LTX_RELAY_SEGMENT_MERGE -- the MASK-FREE form of the same bias.
            //
            // Built INDEPENDENTLY of the dense-mask refusal above: the whole point is that no
            // dense [L, L] tensor is ever materialised, so the MiB ceiling that governs the
            // mask has nothing to say here. Deliberately gated on the same
            // LTX_RELAY_SELF_MASK strength so the two paths are A/B-comparable by flipping one
            // extra variable, and the mask seed is still built above so the DiT can fall back
            // to it if the backend cannot serve ggml_flash_attn_ext_lse.
            SelfSegmentPlan v_self_segments;
            if (sd::ltx_relay::segment_merge_enabled() && relay != nullptr &&
                sd::ltx_relay::self_mask_strength() > 0.f && !relay->beats.empty() &&
                static_cast<int64_t>(relay->video_frame_time.size()) == vx->ne[2]) {
                const int64_t frames           = vx->ne[2];
                const int64_t tokens_per_frame = vx->ne[0] * vx->ne[1];
                const int64_t target_tokens    = frames * tokens_per_frame;

                if (relay_seg_key != std::make_tuple(relay->revision, frames, ref_token_count)) {
                    std::vector<float> seg_bias;
                    relay_seg_cut_frames.clear();
                    relay_seg_bias_vec.clear();
                    if (sd::ltx_relay::build_self_segment_bias(*relay,
                                                               sd::ltx_relay::self_mask_strength(),
                                                               // one slot is spent on the
                                                               // reference-key segment
                                                               sd::ltx_relay::segment_merge_max() -
                                                                   (ref_token_count > 0 ? 1 : 0),
                                                               relay_seg_cut_frames,
                                                               seg_bias)) {
                        relay_seg_bias_vec = std::move(seg_bias);
                        if (ref_token_count > 0) {
                            // Reference keys are off the timeline: one more segment, zero bias.
                            relay_seg_bias_vec.resize(relay_seg_bias_vec.size() + (size_t)frames, 0.f);
                        }
                    } else {
                        LOG_WARN("ltxav relay segment merge: the bias does not split into at most "
                                 "LTX_RELAY_SEGMENT_MAX=%lld key segments (or is uniform); "
                                 "using the LTX_RELAY_SELF_MASK path",
                                 (long long)sd::ltx_relay::segment_merge_max());
                    }
                    relay_seg_key = {relay->revision, frames, ref_token_count};
                }

                if (!relay_seg_cut_frames.empty()) {
                    const int64_t n_seg = (int64_t)relay_seg_cut_frames.size() + (ref_token_count > 0 ? 1 : 0);
                    v_self_segments.start_token.reserve((size_t)n_seg);
                    for (int64_t cut : relay_seg_cut_frames) {
                        v_self_segments.start_token.push_back(cut * tokens_per_frame);
                    }
                    if (ref_token_count > 0) {
                        v_self_segments.start_token.push_back(target_tokens);
                    }
                    v_self_segments.bias = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, frames, n_seg);
                    ggml_set_name(v_self_segments.bias, "ltxav_relay_segment_bias");
                    set_backend_tensor_data(v_self_segments.bias, relay_seg_bias_vec.data());
                    v_self_segments.frames           = frames;
                    v_self_segments.tokens_per_frame = tokens_per_frame;
                    v_self_segments.target_tokens    = target_tokens;
                    std::string cut_list;
                    for (int64_t cut : relay_seg_cut_frames) {
                        cut_list += std::to_string(cut) + " ";
                    }
                    LOG_DEBUG("ltxav relay segment merge: %lld key segments, cuts at frames %s(+refs), "
                              "%zu beats, strength %.3f",
                              (long long)n_seg,
                              cut_list.c_str(),
                              relay->beats.size(),
                              sd::ltx_relay::self_mask_strength());
                }
            }

            auto runner_ctx = get_context();
            runner_ctx.ltx_video_token_sel = v_token_sel_input;
            runner_ctx.ltx_skip_a2v_cross_attn = skip_a2v;
            auto out_pair   = model.forward(&runner_ctx,
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
                                            v_relay_mask,
                                            a_relay_mask,
                                            ref_vx,
                                            v_connector_mask,
                                            a_connector_mask,
                                            v_self_mask_seed,
                                            ref_strength_mask,
                                            ref_strength_fold,
                                            v_self_segments.bias != nullptr ? &v_self_segments : nullptr);
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
                                  bool skip_a2v = false,
                                  const sd::ltx_relay::Plan* relay = nullptr,
                                  const sd::Tensor<float>& ref_video_x       = {},
                                  const std::vector<float>* video_source_ids = nullptr,
                                  float tass_phase_scale                     = 1.f) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x,
                                   timesteps,
                                   context,
                                   audio_x,
                                   audio_timesteps,
                                   audio_length,
                                   frame_rate,
                                   video_positions,
                                   audio_positions,
                                   skip_a2v,
                                   relay,
                                   ref_video_x,
                                   video_source_ids,
                                   tass_phase_scale);
            };
            auto out = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
            return out;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            const auto* extra = diffusion_extra_as<LTXAVDiffusionExtra>(diffusion_params);
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
                           extra->skip_a2v,
                           extra->relay,
                           tensor_or_empty(extra->ref_video_x),
                           extra->video_source_ids,
                           extra->tass_phase_scale);
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

            auto model_manager        = std::make_shared<ModelManager>();
            ModelLoader& model_loader = model_manager->loader();
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
                                                                               tensor_storage_map,
                                                                               "model.diffusion_model",
                                                                               model_manager);

            if (!model_manager->register_runner_params("LTXAV test",
                                                       *ltxav,
                                                       "model.diffusion_model",
                                                       ModelManager::ResidencyMode::ParamBackend,
                                                       backend,
                                                       backend) ||
                !model_manager->validate_registered_tensors()) {
                LOG_ERROR("register ltxav tensors with model manager failed");
                return;
            }

            LOG_INFO("ltxav model loaded");
            ltxav->test(x_path, timesteps_path, context_path, audio_x_path, audio_timesteps_path);
        }
    };

};  // namespace LTXV

#endif  // __SD_MODEL_DIFFUSION_LTXV_HPP__
