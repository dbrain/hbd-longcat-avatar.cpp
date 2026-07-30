#ifndef __SD_MODEL_DIFFUSION_KREA2_HPP__
#define __SD_MODEL_DIFFUSION_KREA2_HPP__

#include <inttypes.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/ggml_graph_cut.h"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace Krea2 {
    constexpr int KREA2_GRAPH_SIZE = 65536;

    // ---------------------------------------------------------------------------------------
    // KREA2_DIT_F16 — run the DiT residual stream (the [features x tokens] hidden state that
    // flows through `blocks.*`) in F16 instead of F32.
    //
    // WHY: at 2048x2048 there are 16,384 image tokens and the sampling peak is the DiT's own
    // activation working set (~4,000 MB), not the VAE. The SwiGLU dominates it: the
    // intermediate is [16384, 16384] and SwiGLU materializes gate + up + product, i.e.
    // 3 x 1,024 MB in F32. Lowering KREA2_MAX_VRAM does NOT help — a graph cut splits the
    // BLOCK sequence, but every activation inside a segment still spans the full token
    // sequence (measured: 11 -> 9 -> 7 GiB moved the peak by 2 MiB). Halving the element width
    // is the only lever that touches those tensors.
    //
    // HOW IT IS SAFE: ggml_ext_linear (ggml_extend.hpp) emits an F16 matmul DESTINATION when
    // the activation is F16 and the weight is NVFP4, because the cuBLASLt FP4 GEMM accumulates
    // in F32 and stores F16. That GEMM is ALSO the only route in ggml_cuda_mul_mat() that can
    // consume an F16 activation against a quantized weight at all: for anything else
    // ggml_backend_cuda_device_supports_op() REJECTS an F16 src1 (ggml-cuda.cu ~5334), and a
    // rejected DiT Linear is not slow, it is silently dropped to the CPU backend. So this is a
    // correctness gate, not a perf one — hence the device probe AND the weight-type probe
    // below rather than a bare env check.
    //
    // Value-honouring: `KREA2_DIT_F16=0` must DISABLE it. A presence-only test (getenv() !=
    // nullptr) would silently keep F16 on for anyone bisecting with an explicit 0.
    __STATIC_INLINE__ bool krea2_dit_f16_env() {
        static int v = -1;
        if (v < 0) {
            const char* e = getenv("KREA2_DIT_F16");
            v             = (e != nullptr && atoi(e) != 0) ? 1 : 0;
        }
        return v == 1;
    }

    // Device/backend half of the gate: will this backend serve an NVFP4 mul_mat with an F16
    // activation AND an F16 destination? True only for CUDA + GGML_NVFP4_CUBLASLT=1 +
    // Blackwell. On the sm86 3060 that the same image is deployed to this is false and the
    // well-tested F32 stream runs unchanged — there is no F16-Q path there, and an ungated
    // LTX_DIT_F16 is exactly what crashed a 3060 once already.
    //
    // A runtime weight adapter used to refuse the F16 stream UNCONDITIONALLY: LoRA takes
    // WeightAdapter::forward_with_lora(), whose delta chain runs `x @ A^T @ B^T` against adapter
    // tensors of unknown type, and an F32 adapter weight against an F16 x is precisely the
    // supports_op rejection above — the Linear would be dropped to the CPU, not merely slowed.
    //
    // That blanket refusal is now narrowed to what it was actually protecting against. An NVFP4
    // adapter is the one case where the premise is FALSE: supports_op explicitly serves NVFP4
    // src0 against an F16 src1 (the cuBLASLt FP4 GEMM quantises the activation to E2M1 itself),
    // and both LoRA GEMMs go through ggml_ext_linear(), so its F16-dst gate fires for them too
    // and the whole delta chain stays half-width. WeightAdapter::supports_f16_activation()
    // answers this per adapter and defaults to false, so every other adapter type — F32, F16,
    // Q8_0, mixed, or an NVFP4 one whose rank is not a multiple of 64 — keeps today's behaviour.
    //
    // Both halves are required: the adapter must be F16-safe AND the device must serve an F16
    // NVFP4 destination at all.
    __STATIC_INLINE__ bool krea2_dit_f16_device_ok(GGMLRunnerContext* ctx) {
        if (ctx == nullptr) {
            return false;
        }
        if (ctx->weight_adapter != nullptr && !ctx->weight_adapter->supports_f16_activation()) {
            return false;
        }
        return ggml_cuda_nvfp4_f16_dst_available(ctx->backend);
    }
    // ---------------------------------------------------------------------------------------

    // Value-honouring env read for the per-render caches below: `=0` must DISABLE, exactly as
    // for KREA2_DIT_F16. A presence-only test would ignore anyone bisecting with an explicit 0.
    __STATIC_INLINE__ bool krea2_env_flag(const char* name, bool dflt) {
        const char* e = getenv(name);
        if (e == nullptr || e[0] == '\0') {
            return dflt;
        }
        return atoi(e) != 0;
    }

    // FNV-1a, byte at a time. Fine for the few-hundred-byte scalar keys.
    __STATIC_INLINE__ uint64_t krea2_fnv1a(uint64_t h, const void* data, size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < bytes; ++i) {
            h = (h ^ p[i]) * 1099511628211ull;
        }
        return h;
    }

    template <typename T>
    __STATIC_INLINE__ uint64_t krea2_fnv1a_val(uint64_t h, const T& v) {
        return krea2_fnv1a(h, &v, sizeof(T));
    }

    // Bulk digest for the CONDITIONING, which is ~74 MB ([2560*12, 605] F32). The byte-at-a-time
    // loop above is a serial multiply chain — ~100 ms on that buffer, i.e. more than the cache it
    // guards saves. Eight independent lanes over 8-byte words turn it into a memory-bandwidth
    // walk (~5 ms), which is roughly what the 74 MB H2D upload this change REMOVES was already
    // costing per step. Still an exact identity check: a hit needs the same byte length and the
    // same 64-bit digest over every byte.
    __STATIC_INLINE__ uint64_t krea2_bulk_hash(const void* data, size_t bytes) {
        constexpr int LANES = 8;
        uint64_t h[LANES];
        for (int i = 0; i < LANES; ++i) {
            h[i] = 1469598103934665603ull + static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull;
        }
        const auto* p     = static_cast<const unsigned char*>(data);
        const size_t words = bytes / 8;
        size_t i           = 0;
        for (; i + LANES <= words; i += LANES) {
            for (int l = 0; l < LANES; ++l) {
                uint64_t v;
                std::memcpy(&v, p + (i + static_cast<size_t>(l)) * 8, sizeof(v));
                h[l] = (h[l] ^ v) * 1099511628211ull;
            }
        }
        uint64_t out = 1469598103934665603ull;
        out          = krea2_fnv1a_val(out, bytes);
        for (int l = 0; l < LANES; ++l) {
            out = krea2_fnv1a_val(out, h[l]);
        }
        // Tail: whole words the lane loop could not fill, plus any trailing bytes.
        out = krea2_fnv1a(out, p + i * 8, bytes - i * 8);
        // Zero is the "nothing cached yet" sentinel, so never return it.
        return out == 0 ? 1 : out;
    }

    // Self-attention for the F16 residual stream: RoPE, then attention, with the two type
    // normalizations the CUDA kernels actually require.
    //
    //   * Q BACK TO F32. Rope::apply_rope keeps F16 end to end (rope-pe.cu is templated on the
    //     element type and does its math in F32 either way), but ggml's native CUDA flash
    //     kernels assert Q->type == GGML_TYPE_F32 (fattn-common.cuh:990). Only the Blackwell
    //     cuDNN SDPA takes an F16 Q, and ggml_cuda_get_best_fattn_kernel() never inspects
    //     Q->type — so it would happily pick a native kernel and abort mid-render. Normalizing
    //     Q here makes the F16 stream independent of that selection instead of silently
    //     depending on it. K/V are unaffected: they are F16 on both streams (the wrapper casts
    //     them anyway), which is what kv_prescaled_f16 then lets us skip re-doing.
    //
    //   * f16_out_ok. Only meaningful under the F16 stream, and deliberately NOT enabled for
    //     the F32 one even though GGML_CUDNN_ATTN_F16_OUT is set in the service env: Krea2's
    //     attention output is immediately multiplied by sigmoid(gate(x)), and ggml fuses that
    //     pair into ggml_cuda_op_unary_mul, which GGML_ASSERTs both operands share a type
    //     (unary.cu ~592). An F16 attention output against an F32 gate — which is what a
    //     bare GGML_CUDNN_ATTN_F16_OUT would produce on the F32 stream — aborts the render.
    //     The caller re-normalizes `out` to F16 for the same reason, since this flag can still
    //     decline on shape/env grounds.
    __STATIC_INLINE__ ggml_tensor* krea2_rope_attention(GGMLRunnerContext* ctx,
                                                        ggml_tensor* q,
                                                        ggml_tensor* k,
                                                        ggml_tensor* v,
                                                        ggml_tensor* pe,
                                                        ggml_tensor* mask,
                                                        bool f16_stream) {
        int64_t n_head = q->ne[1];

        q = Rope::apply_rope(ctx->ggml_ctx, q, pe);  // [N*n_head, L, d_head]
        k = Rope::apply_rope(ctx->ggml_ctx, k, pe);  // [N*n_kv_head, L, d_head]

        if (q->type != GGML_TYPE_F32) {
            q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_F32);
        }

        // kv_scale stays 1.0, so "prescaled" here just means "already F16" and skips a
        // redundant full-size ggml_cast of K and V inside the wrapper.
        const bool kv_prescaled_f16 = k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;

        return ggml_ext_attention_ext(ctx->ggml_ctx,
                                      ctx->backend,
                                      q,
                                      k,
                                      v,
                                      n_head,
                                      mask,
                                      /*skip_reshape=*/true,
                                      ctx->flash_attn_enabled,
                                      /*kv_scale=*/1.0f,
                                      kv_prescaled_f16,
                                      /*f16_out_ok=*/f16_stream);
    }

    struct Krea2Config {
        int patch_size            = 2;
        int64_t in_channels       = 16;
        int64_t out_channels      = 16;
        int64_t features          = 6144;
        int64_t timestep_dim      = 256;
        int64_t text_dim          = 2560;
        int64_t text_layers       = 12;
        int64_t layers            = 28;
        int64_t heads             = 48;
        int64_t kv_heads          = 12;
        int64_t text_heads        = 20;
        int64_t text_kv_heads     = 20;
        int64_t mlp_multiplier    = 4;
        float theta               = 1000.f;
        float norm_eps            = 1e-5f;
        std::vector<int> axes_dim = {32, 48, 48};
        int axes_dim_sum          = 128;

        int64_t head_dim() const {
            return features / heads;
        }

        static int64_t count_blocks(const String2TensorStorage& tensor_storage_map,
                                    const std::string& prefix,
                                    const std::string& block_prefix) {
            int64_t count           = 0;
            std::string full_prefix = prefix.empty() ? block_prefix : prefix + "." + block_prefix;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, full_prefix)) {
                    continue;
                }
                std::string tail = name.substr(full_prefix.size());
                size_t dot       = tail.find('.');
                if (dot == std::string::npos) {
                    continue;
                }
                int block_index = std::atoi(tail.substr(0, dot).c_str());
                count           = std::max<int64_t>(count, block_index + 1);
            }
            return count;
        }

        void update_axes_dim() {
            int64_t dim_head = head_dim();
            int64_t unit     = dim_head / 16;
            axes_dim         = {
                        static_cast<int>(dim_head - 12 * unit),
                        static_cast<int>(6 * unit),
                        static_cast<int>(6 * unit),
            };
            axes_dim_sum = axes_dim[0] + axes_dim[1] + axes_dim[2];
        }

        static Krea2Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                               const std::string& prefix) {
            Krea2Config config;
            int64_t detected_head_dim      = 0;
            int64_t detected_text_head_dim = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "first.weight") && tensor_storage.n_dims == 2) {
                    config.in_channels  = tensor_storage.ne[0] / (config.patch_size * config.patch_size);
                    config.out_channels = config.in_channels;
                    config.features     = tensor_storage.ne[1];
                } else if (ends_with(name, "blocks.0.attn.qknorm.qnorm.scale") && tensor_storage.n_dims == 1) {
                    detected_head_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "blocks.0.attn.wq.weight") && tensor_storage.n_dims == 2) {
                    if (detected_head_dim > 0) {
                        config.heads = tensor_storage.ne[1] / detected_head_dim;
                    }
                } else if (ends_with(name, "blocks.0.attn.wk.weight") && tensor_storage.n_dims == 2) {
                    if (detected_head_dim > 0) {
                        config.kv_heads = tensor_storage.ne[1] / detected_head_dim;
                    }
                } else if (ends_with(name, "txtfusion.projector.weight") && tensor_storage.n_dims == 2) {
                    config.text_layers = tensor_storage.ne[0];
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.prenorm.scale") && tensor_storage.n_dims == 1) {
                    config.text_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale") && tensor_storage.n_dims == 1) {
                    detected_text_head_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.attn.wq.weight") && tensor_storage.n_dims == 2) {
                    if (detected_text_head_dim > 0) {
                        config.text_heads = tensor_storage.ne[1] / detected_text_head_dim;
                    }
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.attn.wk.weight") && tensor_storage.n_dims == 2) {
                    if (detected_text_head_dim > 0) {
                        config.text_kv_heads = tensor_storage.ne[1] / detected_text_head_dim;
                    }
                } else if (ends_with(name, "last.linear.weight") && tensor_storage.n_dims == 2) {
                    config.out_channels = tensor_storage.ne[1] / (config.patch_size * config.patch_size);
                }
            }

            config.layers = std::max<int64_t>(1, count_blocks(tensor_storage_map, prefix, "blocks."));
            if (detected_head_dim > 0 && config.features > 0) {
                config.heads = config.features / detected_head_dim;
            }
            if (detected_head_dim > 0) {
                std::string wk_name = prefix.empty() ? "blocks.0.attn.wk.weight" : prefix + ".blocks.0.attn.wk.weight";
                auto it             = tensor_storage_map.find(wk_name);
                if (it != tensor_storage_map.end() && it->second.n_dims == 2) {
                    config.kv_heads = it->second.ne[1] / detected_head_dim;
                }
            }
            if (detected_text_head_dim > 0 && config.text_dim > 0) {
                config.text_heads = config.text_dim / detected_text_head_dim;
            }
            if (detected_text_head_dim > 0) {
                std::string wk_name = prefix.empty() ? "txtfusion.layerwise_blocks.0.attn.wk.weight" : prefix + ".txtfusion.layerwise_blocks.0.attn.wk.weight";
                auto it             = tensor_storage_map.find(wk_name);
                if (it != tensor_storage_map.end() && it->second.n_dims == 2) {
                    config.text_kv_heads = it->second.ne[1] / detected_text_head_dim;
                }
            }
            config.update_axes_dim();

            LOG_DEBUG("krea2: layers=%" PRId64 ", features=%" PRId64 ", heads=%" PRId64 ", kv_heads=%" PRId64 ", text_dim=%" PRId64 ", text_layers=%" PRId64 ", text_heads=%" PRId64 ", text_kv_heads=%" PRId64 ", channels=%" PRId64,
                      config.layers,
                      config.features,
                      config.heads,
                      config.kv_heads,
                      config.text_dim,
                      config.text_layers,
                      config.text_heads,
                      config.text_kv_heads,
                      config.in_channels);
            return config;
        }
    };

    __STATIC_INLINE__ int64_t ceil_to_multiple(int64_t value, int64_t multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    class KreaRMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;
        std::string prefix;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            this->prefix    = prefix;
            params["scale"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        }

    public:
        KreaRMSNorm(int64_t hidden_size, float eps = 1e-5f)
            : hidden_size(hidden_size),
              eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* scale = params["scale"];
            scale              = ggml_add(ctx->ggml_ctx, scale, ggml_ext_ones(ctx->ggml_ctx, scale->ne[0], 1, 1, 1));
            x                  = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            x                  = ggml_mul_inplace(ctx->ggml_ctx, x, scale);
            return x;
        }
    };

    class KreaSwiGLU : public UnaryBlock {
    public:
        KreaSwiGLU(int64_t features, int64_t multiplier) {
            int64_t mlp_dim = ceil_to_multiple(((2 * features) / 3) * multiplier, 128);
            blocks["gate"]  = std::make_shared<Linear>(features, mlp_dim, false);
            blocks["up"]    = std::make_shared<Linear>(features, mlp_dim, false);
            blocks["down"]  = std::make_shared<Linear>(mlp_dim, features, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gate = std::dynamic_pointer_cast<Linear>(blocks["gate"]);
            auto up   = std::dynamic_pointer_cast<Linear>(blocks["up"]);
            auto down = std::dynamic_pointer_cast<Linear>(blocks["down"]);

            // ⚠️ OPERAND ORDER IS LOAD-BEARING. ggml only fuses ops at strictly consecutive node
            // indices (ggml_can_fuse_ext, ggml-impl.h:669) and the graph DFS visits src[0] before
            // src[1]. With the SiLU as src[0] the DFS emits MM(gate), SILU, MM(up), MUL — two apart
            // — so {UNARY(SILU), MUL} (ggml-cuda.cu:3941) never matches and the SiLU runs as its own
            // full pass over [16384, L]. With the SiLU as src[1] it emits MM(up), MM(gate), SILU,
            // MUL and ggml_cuda_op_unary_mul (unary.cu:578) fires: 5 memory passes -> 3.
            // Bit-exact: FP multiply is commutative and no intermediate is round-tripped (verified
            // by md5-identical renders across a 4-arm A/B).
            // KreaAttention (below, ~line 289) already had the lucky order, which is why the profile
            // showed its sigmoid fused while this SiLU was not.
            auto up_x  = up->forward(ctx, x);
            auto gated = ggml_silu(ctx->ggml_ctx, gate->forward(ctx, x));
            x          = ggml_mul(ctx->ggml_ctx, up_x, gated);
            return down->forward(ctx, x);
        }

        // See KreaAttention::linears_all_nvfp4().
        bool linears_all_nvfp4() {
            for (const char* name : {"gate", "up", "down"}) {
                auto linear = std::dynamic_pointer_cast<Linear>(blocks[name]);
                if (linear == nullptr || linear->get_weight() == nullptr ||
                    linear->get_weight()->type != GGML_TYPE_NVFP4) {
                    return false;
                }
            }
            return true;
        }
    };

    class KreaAttention : public GGMLBlock {
    protected:
        int64_t features;
        int64_t heads;
        int64_t kv_heads;
        int64_t head_dim_;

        ggml_tensor* attention_no_rope(GGMLRunnerContext* ctx,
                                       ggml_tensor* q,
                                       ggml_tensor* k,
                                       ggml_tensor* v,
                                       ggml_tensor* mask) {
            int64_t Lq = q->ne[2];
            int64_t Lk = k->ne[2];
            int64_t N  = q->ne[3];
            q          = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, q), head_dim_ * heads, Lq, N);
            k          = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, k), head_dim_ * kv_heads, Lk, N);
            v          = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, v), head_dim_ * kv_heads, Lk, N);
            // Same Q-must-be-F32 rule as krea2_rope_attention (native CUDA fattn kernels assert
            // it). Unreachable today — this branch is txtfusion's, which stays F32 — but the
            // invariant "no fattn kernel ever sees a non-F32 Q" should hold by construction,
            // not by which caller happens to be F32 this week.
            if (q->type != GGML_TYPE_F32) {
                q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_F32);
            }
            return ggml_ext_attention_ext(ctx->ggml_ctx,
                                          ctx->backend,
                                          q,
                                          k,
                                          v,
                                          heads,
                                          mask,
                                          false,
                                          ctx->flash_attn_enabled);
        }

    public:
        KreaAttention(int64_t features,
                      int64_t heads,
                      int64_t kv_heads,
                      float eps = 1e-5f)
            : features(features),
              heads(heads),
              kv_heads(kv_heads),
              head_dim_(features / heads) {
            blocks["wq"]           = std::make_shared<Linear>(features, heads * head_dim_, false);
            blocks["wk"]           = std::make_shared<Linear>(features, kv_heads * head_dim_, false);
            blocks["wv"]           = std::make_shared<Linear>(features, kv_heads * head_dim_, false);
            blocks["gate"]         = std::make_shared<Linear>(features, features, false);
            blocks["qknorm.qnorm"] = std::make_shared<KreaRMSNorm>(head_dim_, eps);
            blocks["qknorm.knorm"] = std::make_shared<KreaRMSNorm>(head_dim_, eps);
            blocks["wo"]           = std::make_shared<Linear>(features, features, false);
        }

        // out_lo/out_hi: keep ONLY tokens [out_lo, out_hi) from the point where attention stops
        // mixing rows. Everything after `attn_out` here — the gate Linear, the sigmoid multiply
        // and `wo` — is strictly row-wise, so dropping rows there computes exactly the rows the
        // caller keeps. Used by the LAST DiT block only, where the caller throws the text and
        // reference rows away immediately afterwards. -1 (the default) means "all tokens".
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe   = nullptr,
                             ggml_tensor* mask = nullptr,
                             int64_t out_lo    = -1,
                             int64_t out_hi    = -1) {
            auto wq    = std::dynamic_pointer_cast<Linear>(blocks["wq"]);
            auto wk    = std::dynamic_pointer_cast<Linear>(blocks["wk"]);
            auto wv    = std::dynamic_pointer_cast<Linear>(blocks["wv"]);
            auto gate  = std::dynamic_pointer_cast<Linear>(blocks["gate"]);
            auto qnorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["qknorm.qnorm"]);
            auto knorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["qknorm.knorm"]);
            auto wo    = std::dynamic_pointer_cast<Linear>(blocks["wo"]);

            if (sd_backend_is(ctx->backend, "Vulkan") || sd_backend_is(ctx->backend, "ROCm")) {
                wo->set_force_prec_f32(true);
            }

            int64_t L = x->ne[1];
            int64_t N = x->ne[2];

            auto q = wq->forward(ctx, x);
            q      = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim_, heads, L, N);
            auto k = wk->forward(ctx, x);
            k      = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim_, kv_heads, L, N);
            auto v = wv->forward(ctx, x);
            v      = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim_, kv_heads, L, N);

            q = qnorm->forward(ctx, q);
            k = knorm->forward(ctx, k);

            // The residual stream's own type IS the mode flag: under KREA2_DIT_F16 `x` arrives
            // F16, and every Linear here has an NVFP4 weight, so wq/wk/wv/gate all come back
            // F16 from ggml_ext_linear's F16-dst gate. Deriving it from `x` instead of
            // plumbing a bool down means the block body cannot desync from the stream.
            const bool f16_stream = x->type == GGML_TYPE_F16;

            auto out = pe != nullptr ? krea2_rope_attention(ctx, q, k, v, pe, mask, f16_stream)
                                     : attention_no_rope(ctx, q, k, v, mask);

            // TOKEN TRIM (last block only). Attention is the last op that mixes rows, so this is
            // the earliest point a row can be dropped without changing any row that survives.
            //
            // `x` is sliced TOO, not just `out`. Two reasons, both load-bearing:
            //   * the gate Linear reads `x`, so leaving it full width would keep a 6144x6144 GEMM
            //     at the full sequence for an output that is immediately discarded;
            //   * slicing the SIGMOID's output instead of its input would put a cont node between
            //     the sigmoid and the multiply, and ggml only fuses at strictly consecutive node
            //     indices — that would break ggml_cuda_op_unary_mul and cost a full extra pass
            //     over [features x tokens], which is most of what the trim just saved.
            if (out_lo >= 0) {
                GGML_ASSERT(out_hi > out_lo && out_hi <= out->ne[1] && out_hi <= x->ne[1]);
                out = ggml_ext_slice(ctx->ggml_ctx, out, 1, out_lo, out_hi);
                x   = ggml_ext_slice(ctx->ggml_ctx, x, 1, out_lo, out_hi);
            }

            // Re-join the stream. `out` is F16 already when GGML_CUDNN_ATTN_F16_OUT is on AND
            // the shape gate fired; it is F32 when that flag is off, or when flash attention
            // declined and the manual mul_mat fallback ran (that path always writes F32).
            // Both must end up F16 here, because the multiply below fuses with the sigmoid
            // into ggml_cuda_op_unary_mul, which GGML_ASSERTs its two operands share a type —
            // and gate->forward(x) is F16. A mismatch is an abort, not a slow path.
            if (f16_stream && out->type != GGML_TYPE_F16) {
                out = ggml_cast(ctx->ggml_ctx, out, GGML_TYPE_F16);
            }

            // ⚠️ OPERAND ORDER IS LOAD-BEARING here too, for the same reason as KreaSwiGLU:
            // the SiLU/sigmoid must be src[1] so the DFS emits UNARY then MUL adjacently.
            out = ggml_mul(ctx->ggml_ctx, out, ggml_sigmoid(ctx->ggml_ctx, gate->forward(ctx, x)));
            out = wo->forward(ctx, out);
            return out;
        }

        // True iff every Linear in this attention carries an NVFP4 weight — i.e. every one of
        // them can be fed an F16 activation. See Krea2Model::blocks_accept_f16().
        bool linears_all_nvfp4() {
            for (const char* name : {"wq", "wk", "wv", "gate", "wo"}) {
                auto linear = std::dynamic_pointer_cast<Linear>(blocks[name]);
                if (linear == nullptr || linear->get_weight() == nullptr ||
                    linear->get_weight()->type != GGML_TYPE_NVFP4) {
                    return false;
                }
            }
            return true;
        }
    };

    class KreaDoubleSharedModulation : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            GGML_UNUSED(prefix);
            params["lin"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim * 6);
        }

    public:
        KreaDoubleSharedModulation(int64_t dim)
            : dim(dim) {}

        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            auto lin = ggml_repeat(ctx->ggml_ctx, params["lin"], vec);
            auto out = ggml_add(ctx->ggml_ctx, vec, lin);
            return ggml_ext_chunk(ctx->ggml_ctx, out, 6, 0);
        }
    };

    class KreaFinalModulation : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            GGML_UNUSED(prefix);
            params["lin"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
        }

    public:
        KreaFinalModulation(int64_t dim)
            : dim(dim) {}

        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            auto out = ggml_add(ctx->ggml_ctx, params["lin"], vec);
            return ggml_ext_chunk(ctx->ggml_ctx, out, 2, 1);
        }
    };

    class KreaTextFusionBlock : public UnaryBlock {
    public:
        KreaTextFusionBlock(int64_t dim,
                            int64_t heads,
                            int64_t kv_heads,
                            int64_t multiplier,
                            float eps) {
            blocks["prenorm"]  = std::make_shared<KreaRMSNorm>(dim, eps);
            blocks["postnorm"] = std::make_shared<KreaRMSNorm>(dim, eps);
            blocks["attn"]     = std::make_shared<KreaAttention>(dim, heads, kv_heads, eps);
            blocks["mlp"]      = std::make_shared<KreaSwiGLU>(dim, multiplier);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto prenorm  = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["prenorm"]);
            auto postnorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["postnorm"]);
            auto attn     = std::dynamic_pointer_cast<KreaAttention>(blocks["attn"]);
            auto mlp      = std::dynamic_pointer_cast<KreaSwiGLU>(blocks["mlp"]);

            x = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, prenorm->forward(ctx, x)));
            x = ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, postnorm->forward(ctx, x)));
            return x;
        }
    };

    class KreaTextFusionTransformer : public UnaryBlock {
    protected:
        Krea2Config config;

    public:
        explicit KreaTextFusionTransformer(Krea2Config config)
            : config(std::move(config)) {
            for (int i = 0; i < 2; ++i) {
                blocks["layerwise_blocks." + std::to_string(i)] = std::make_shared<KreaTextFusionBlock>(this->config.text_dim,
                                                                                                        this->config.text_heads,
                                                                                                        this->config.text_kv_heads,
                                                                                                        this->config.mlp_multiplier,
                                                                                                        this->config.norm_eps);
                blocks["refiner_blocks." + std::to_string(i)]   = std::make_shared<KreaTextFusionBlock>(this->config.text_dim,
                                                                                                      this->config.text_heads,
                                                                                                      this->config.text_kv_heads,
                                                                                                      this->config.mlp_multiplier,
                                                                                                      this->config.norm_eps);
            }
            blocks["projector"] = std::make_shared<Linear>(this->config.text_layers, 1, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* context) override {
            int64_t text_tokens = context->ne[1];
            int64_t batch       = context->ne[2];

            context = ggml_reshape_3d(ctx->ggml_ctx,
                                      context,
                                      config.text_dim,
                                      config.text_layers,
                                      text_tokens * batch);

            for (int i = 0; i < 2; ++i) {
                auto block = std::dynamic_pointer_cast<KreaTextFusionBlock>(blocks["layerwise_blocks." + std::to_string(i)]);
                context    = block->forward(ctx, context);
            }

            context        = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, context, 1, 0, 2, 3));
            auto projector = std::dynamic_pointer_cast<Linear>(blocks["projector"]);
            context        = projector->forward(ctx, context);
            context        = ggml_reshape_3d(ctx->ggml_ctx, context, config.text_dim, text_tokens, batch);

            for (int i = 0; i < 2; ++i) {
                auto block = std::dynamic_pointer_cast<KreaTextFusionBlock>(blocks["refiner_blocks." + std::to_string(i)]);
                context    = block->forward(ctx, context);
            }
            return context;
        }
    };

    class KreaSingleStreamBlock : public UnaryBlock {
    public:
        explicit KreaSingleStreamBlock(Krea2Config config) {
            blocks["mod"]      = std::make_shared<KreaDoubleSharedModulation>(config.features);
            blocks["prenorm"]  = std::make_shared<KreaRMSNorm>(config.features, config.norm_eps);
            blocks["postnorm"] = std::make_shared<KreaRMSNorm>(config.features, config.norm_eps);
            blocks["attn"]     = std::make_shared<KreaAttention>(config.features, config.heads, config.kv_heads, config.norm_eps);
            blocks["mlp"]      = std::make_shared<KreaSwiGLU>(config.features, config.mlp_multiplier);
        }

        // out_lo/out_hi — see KreaAttention::forward. Only the LAST block passes them, and only
        // ever the IMAGE row range, which is why the trimmed path below can use `mods_main`
        // unconditionally: the image rows sit entirely inside [0, ref_start), i.e. inside the
        // "main" modulation half, never in the reference half.
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* vec,
                             ggml_tensor* pe,
                             ggml_tensor* vec_refs = nullptr,
                             int64_t ref_start     = -1,
                             int64_t out_lo        = -1,
                             int64_t out_hi        = -1) {
            auto mod      = std::dynamic_pointer_cast<KreaDoubleSharedModulation>(blocks["mod"]);
            auto prenorm  = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["prenorm"]);
            auto postnorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["postnorm"]);
            auto attn     = std::dynamic_pointer_cast<KreaAttention>(blocks["attn"]);
            auto mlp      = std::dynamic_pointer_cast<KreaSwiGLU>(blocks["mlp"]);

            if (out_lo >= 0 && ref_start >= 0 && vec_refs) {
                // TRIMMED + per-reference modulation. The attention input still spans the whole
                // sequence (references must still be attended to), so the pre-attention half is
                // unchanged, verbatim, from the untrimmed branch below. Only the post-attention
                // half collapses: [out_lo, out_hi) is a subrange of [0, ref_start), so every
                // surviving row takes `mods_main` and the view/concat dance disappears.
                GGML_ASSERT(out_hi > out_lo && out_hi <= ref_start);

                auto mods_main = mod->forward(ctx, vec);
                auto mods_refs = mod->forward(ctx, vec_refs);

                int64_t D  = x->ne[0];
                int64_t N  = x->ne[1];
                int64_t B  = x->ne[2];
                size_t nb1 = x->nb[1];
                size_t nb2 = x->nb[2];

                int64_t len_main = ref_start;
                int64_t len_refs = N - ref_start;

                auto pre_x = prenorm->forward(ctx, x);

                auto pre_x_main = ggml_view_3d(ctx->ggml_ctx, pre_x, D, len_main, B, nb1, nb2, 0);
                auto pre_x_refs = ggml_view_3d(ctx->ggml_ctx, pre_x, D, len_refs, B, nb1, nb2, len_main * nb1);

                auto attn_in_main = Flux::modulate(ctx->ggml_ctx, pre_x_main, mods_main[1], mods_main[0], true);
                auto attn_in_refs = Flux::modulate(ctx->ggml_ctx, pre_x_refs, mods_refs[1], mods_refs[0], true);

                auto attn_input = ggml_concat(ctx->ggml_ctx, attn_in_main, attn_in_refs, 1);

                auto attn_out = attn->forward(ctx, attn_input, pe, nullptr, out_lo, out_hi);

                x = ggml_ext_slice(ctx->ggml_ctx, x, 1, out_lo, out_hi);
                x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, mods_main[2]));

                auto mlp_input = Flux::modulate(ctx->ggml_ctx,
                                                postnorm->forward(ctx, x),
                                                mods_main[4],
                                                mods_main[3],
                                                true);
                auto mlp_out   = mlp->forward(ctx, mlp_input);
                x              = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, mods_main[5]));
                return x;
            }

            if (ref_start >= 0 && vec_refs) {
                // same as normal, but since vec is different for refs and the rest, needs a lot of views and concats
                auto mods_main = mod->forward(ctx, vec);
                auto mods_refs = mod->forward(ctx, vec_refs);

                int64_t D  = x->ne[0];
                int64_t N  = x->ne[1];
                int64_t B  = x->ne[2];
                size_t nb1 = x->nb[1];
                size_t nb2 = x->nb[2];

                int64_t len_main = ref_start;
                int64_t len_refs = N - ref_start;

                auto pre_x = prenorm->forward(ctx, x);

                auto pre_x_main = ggml_view_3d(ctx->ggml_ctx, pre_x, D, len_main, B, nb1, nb2, 0);
                auto pre_x_refs = ggml_view_3d(ctx->ggml_ctx, pre_x, D, len_refs, B, nb1, nb2, len_main * nb1);

                auto attn_in_main = Flux::modulate(ctx->ggml_ctx, pre_x_main, mods_main[1], mods_main[0], true);
                auto attn_in_refs = Flux::modulate(ctx->ggml_ctx, pre_x_refs, mods_refs[1], mods_refs[0], true);

                auto attn_input = ggml_concat(ctx->ggml_ctx, attn_in_main, attn_in_refs, 1);

                auto attn_out = attn->forward(ctx, attn_input, pe);

                auto attn_out_main = ggml_view_3d(ctx->ggml_ctx, attn_out, D, len_main, B, attn_out->nb[1], attn_out->nb[2], 0);
                auto attn_out_refs = ggml_view_3d(ctx->ggml_ctx, attn_out, D, len_refs, B, attn_out->nb[1], attn_out->nb[2], len_main * attn_out->nb[1]);

                auto res_main = ggml_mul(ctx->ggml_ctx, attn_out_main, mods_main[2]);
                auto res_refs = ggml_mul(ctx->ggml_ctx, attn_out_refs, mods_refs[2]);

                auto attn_res = ggml_concat(ctx->ggml_ctx, res_main, res_refs, 1);

                x = ggml_add(ctx->ggml_ctx, x, attn_res);

                auto post_x = postnorm->forward(ctx, x);

                auto post_x_main = ggml_view_3d(ctx->ggml_ctx, post_x, D, len_main, B, post_x->nb[1], post_x->nb[2], 0);
                auto post_x_refs = ggml_view_3d(ctx->ggml_ctx, post_x, D, len_refs, B, post_x->nb[1], post_x->nb[2], len_main * post_x->nb[1]);

                auto mlp_in_main = Flux::modulate(ctx->ggml_ctx, post_x_main, mods_main[4], mods_main[3], true);
                auto mlp_in_refs = Flux::modulate(ctx->ggml_ctx, post_x_refs, mods_refs[4], mods_refs[3], true);

                auto mlp_input = ggml_concat(ctx->ggml_ctx, mlp_in_main, mlp_in_refs, 1);
                auto mlp_out   = mlp->forward(ctx, mlp_input);

                auto mlp_out_main = ggml_view_3d(ctx->ggml_ctx, mlp_out, D, len_main, B, mlp_out->nb[1], mlp_out->nb[2], 0);
                auto mlp_out_refs = ggml_view_3d(ctx->ggml_ctx, mlp_out, D, len_refs, B, mlp_out->nb[1], mlp_out->nb[2], len_main * mlp_out->nb[1]);

                auto mlp_res_main = ggml_mul(ctx->ggml_ctx, mlp_out_main, mods_main[5]);
                auto mlp_res_refs = ggml_mul(ctx->ggml_ctx, mlp_out_refs, mods_refs[5]);

                auto mlp_res = ggml_concat(ctx->ggml_ctx, mlp_res_main, mlp_res_refs, 1);
                x            = ggml_add(ctx->ggml_ctx, x, mlp_res);
            } else {
                auto mods       = mod->forward(ctx, vec);
                auto attn_input = Flux::modulate(ctx->ggml_ctx,
                                                 prenorm->forward(ctx, x),
                                                 mods[1],
                                                 mods[0],
                                                 true);
                auto attn_out   = attn->forward(ctx, attn_input, pe, nullptr, out_lo, out_hi);
                if (out_lo >= 0) {
                    x = ggml_ext_slice(ctx->ggml_ctx, x, 1, out_lo, out_hi);
                }
                x               = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, mods[2]));

                auto mlp_input = Flux::modulate(ctx->ggml_ctx,
                                                postnorm->forward(ctx, x),
                                                mods[4],
                                                mods[3],
                                                true);
                auto mlp_out   = mlp->forward(ctx, mlp_input);
                x              = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, mods[5]));
            }
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_UNUSED(ctx);
            GGML_UNUSED(x);
            GGML_ABORT("KreaSingleStreamBlock requires conditioning");
            return nullptr;
        }

        // See KreaAttention::linears_all_nvfp4(). `mod`, `prenorm` and `postnorm` are not
        // asked about on purpose: they are plain F32 parameter tensors consumed by
        // rms_norm / binbcast, all of which take an F16 activation against an F32 operand.
        bool linears_all_nvfp4() {
            auto attn = std::dynamic_pointer_cast<KreaAttention>(blocks["attn"]);
            auto mlp  = std::dynamic_pointer_cast<KreaSwiGLU>(blocks["mlp"]);
            return attn != nullptr && mlp != nullptr &&
                   attn->linears_all_nvfp4() && mlp->linears_all_nvfp4();
        }
    };

    class KreaTimeMLP : public UnaryBlock {
    public:
        explicit KreaTimeMLP(Krea2Config config) {
            blocks["0"] = std::make_shared<Linear>(config.timestep_dim, config.features, true);
            blocks["2"] = std::make_shared<Linear>(config.features, config.features, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto linear_0 = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);
            x             = linear_0->forward(ctx, x);
            x             = ggml_ext_gelu(ctx->ggml_ctx, x, false);
            x             = linear_2->forward(ctx, x);
            return x;
        }
    };

    class KreaTProj : public UnaryBlock {
    public:
        explicit KreaTProj(Krea2Config config) {
            blocks["1"] = std::make_shared<Linear>(config.features, config.features * 6, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["1"]);
            x             = ggml_ext_gelu(ctx->ggml_ctx, x, false);
            x             = linear_1->forward(ctx, x);
            return x;
        }
    };

    class KreaTextMLP : public UnaryBlock {
    public:
        explicit KreaTextMLP(Krea2Config config) {
            blocks["0"] = std::make_shared<KreaRMSNorm>(config.text_dim, config.norm_eps);
            blocks["1"] = std::make_shared<Linear>(config.text_dim, config.features, true);
            blocks["3"] = std::make_shared<Linear>(config.features, config.features, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto norm     = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["0"]);
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["1"]);
            auto linear_3 = std::dynamic_pointer_cast<Linear>(blocks["3"]);
            x             = norm->forward(ctx, x);
            x             = linear_1->forward(ctx, x);
            x             = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            x             = linear_3->forward(ctx, x);
            return x;
        }
    };

    class KreaLastLayer : public GGMLBlock {
    public:
        explicit KreaLastLayer(Krea2Config config) {
            blocks["norm"]       = std::make_shared<KreaRMSNorm>(config.features, config.norm_eps);
            blocks["linear"]     = std::make_shared<Linear>(config.features, config.patch_size * config.patch_size * config.out_channels, true);
            blocks["modulation"] = std::make_shared<KreaFinalModulation>(config.features);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* vec) {
            auto norm       = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["norm"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto modulation = std::dynamic_pointer_cast<KreaFinalModulation>(blocks["modulation"]);

            auto mods = modulation->forward(ctx, vec);
            x         = Flux::modulate(ctx->ggml_ctx,
                                       norm->forward(ctx, x),
                                       mods[1],
                                       mods[0],
                                       true);
            x         = linear->forward(ctx, x);
            return x;
        }
    };

    class Krea2Model : public GGMLBlock {
    protected:
        Krea2Config config;

    public:
        Krea2Model() = default;
        explicit Krea2Model(Krea2Config config)
            : config(std::move(config)) {
            blocks["first"]     = std::make_shared<Linear>(this->config.patch_size * this->config.patch_size * this->config.in_channels,
                                                       this->config.features,
                                                       true);
            blocks["tmlp"]      = std::make_shared<KreaTimeMLP>(this->config);
            blocks["txtfusion"] = std::make_shared<KreaTextFusionTransformer>(this->config);
            blocks["txtmlp"]    = std::make_shared<KreaTextMLP>(this->config);
            blocks["tproj"]     = std::make_shared<KreaTProj>(this->config);
            for (int i = 0; i < this->config.layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<KreaSingleStreamBlock>(this->config);
            }
            blocks["last"] = std::make_shared<KreaLastLayer>(this->config);
        }

        // Weight half of the KREA2_DIT_F16 gate: EVERY Linear inside `blocks.*` must be NVFP4,
        // because the F16 stream is all-or-nothing — a single non-NVFP4 weight on the path is
        // an F16 src1 that ggml_backend_cuda_device_supports_op() refuses, and ggml_backend_sched
        // then runs that Linear on the CPU.
        //
        // This is NOT hypothetical for Krea2. The shipping checkpoint is
        // krea2-bf16hold-snofs.gguf — "bf16 holdouts" — and it really does keep 5 weights in
        // BF16: tmlp.0, tmlp.2, tproj.1, txtmlp.1, txtmlp.3. All five sit OUTSIDE the block
        // stack (timestep/text projection), which is exactly why the F16 region below starts
        // after `first` and after `txtmlp`, and ends before `last` (both of which are F32
        // weights in that same checkpoint). Verified per-tensor: all 224 block Linears
        // (28 x {wq,wk,wv,gate,wo,mlp.gate,mlp.up,mlp.down}) are NVFP4. A future re-quant that
        // holds one of them back turns the stream off here rather than falling off a cliff.
        bool blocks_accept_f16() {
            for (int i = 0; i < config.layers; ++i) {
                auto block = std::dynamic_pointer_cast<KreaSingleStreamBlock>(blocks["blocks." + std::to_string(i)]);
                if (block == nullptr || !block->linears_all_nvfp4()) {
                    return false;
                }
            }
            return true;
        }

        // The TEXT half of the DiT, on its own. txtfusion -> txtmlp is a pure function of
        // `context`: no timestep, no x, no noise, nothing that varies across a sampling
        // schedule. Krea2Runner computes it ONCE per conditioning per render and feeds the
        // result back into forward() as `txt_in`, instead of rebuilding these ~2.5 TFLOP (and
        // dragging 32 of the LoRA modules through the graph) on every one of the 10 steps.
        //
        // Kept as its own method rather than a flag inside forward() so the two paths cannot
        // drift: the body below is the exact pair of calls forward() used to make inline.
        ggml_tensor* forward_text(GGMLRunnerContext* ctx, ggml_tensor* context) {
            auto txtfusion = std::dynamic_pointer_cast<KreaTextFusionTransformer>(blocks["txtfusion"]);
            auto txtmlp    = std::dynamic_pointer_cast<KreaTextMLP>(blocks["txtmlp"]);
            GGML_ASSERT(context != nullptr);
            auto txt = txtfusion->forward(ctx, context);
            txt      = txtmlp->forward(ctx, txt);
            return txt;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             std::vector<ggml_tensor*> ref_latents = {},
                             bool zero_timestep_refs               = false,
                             ggml_tensor* txt_in                   = nullptr) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t N = x->ne[3];
            GGML_ASSERT(N == 1);

            auto first     = std::dynamic_pointer_cast<Linear>(blocks["first"]);
            auto tmlp      = std::dynamic_pointer_cast<KreaTimeMLP>(blocks["tmlp"]);
            auto txtfusion = std::dynamic_pointer_cast<KreaTextFusionTransformer>(blocks["txtfusion"]);
            auto txtmlp    = std::dynamic_pointer_cast<KreaTextMLP>(blocks["txtmlp"]);
            auto tproj     = std::dynamic_pointer_cast<KreaTProj>(blocks["tproj"]);
            auto last      = std::dynamic_pointer_cast<KreaLastLayer>(blocks["last"]);

            auto img        = DiT::pad_and_patchify(ctx, x, config.patch_size, config.patch_size, true);
            int64_t img_len = img->ne[1];
            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, config.patch_size, config.patch_size, true);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }
            int64_t ref_len = img->ne[1] - img_len;
            img             = first->forward(ctx, img);

            // KREA2_DIT_F16: enter the F16 residual stream.
            //
            // The cast is AFTER `first`, not before it: first.weight is F32 in the shipping
            // checkpoint, and an F16 activation against an F32 weight is the supports_op
            // rejection described on blocks_accept_f16() — it would move the patch embed to
            // the CPU. Same reasoning at the other end: `last.linear.weight` is F32 too, so
            // the stream leaves F16 before the head.
            //
            // txt is cast separately rather than casting the concat, so the [features x
            // tokens] F32 concat buffer is never materialized: img is ~16,384 tokens at
            // 2048x2048 and txt is a few hundred, so this trades one 384 MB F32 intermediate
            // for a 192 MB F16 one.
            const bool dit_f16 = krea2_dit_f16_env() && krea2_dit_f16_device_ok(ctx) && blocks_accept_f16();
            if (dit_f16) {
                img = ggml_cast(ctx->ggml_ctx, img, GGML_TYPE_F16);
            }
            // graph_build runs once per sampler step, so log the DECISION, not every build.
            // Worth logging at all because every way this gate can decline (wrong card, LoRA
            // attached, GGML_NVFP4_CUBLASLT unset, a re-quant that held a block weight back)
            // is silent otherwise — the render just stays at the F32 VRAM peak.
            {
                static int logged = -1;
                const int decision = dit_f16 ? 1 : 0;
                if (logged != decision) {
                    logged = decision;
                    LOG_INFO("krea2: DiT residual stream = %s", dit_f16 ? "F16 (KREA2_DIT_F16)" : "F32");
                }
            }

            auto t    = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, static_cast<int>(config.timestep_dim), 10000, 1000.f);
            t         = tmlp->forward(ctx, t);
            t         = ggml_reshape_3d(ctx->ggml_ctx, t, t->ne[0], 1, t->ne[1]);
            auto tvec = tproj->forward(ctx, t);

            ggml_tensor* tvec_0 = nullptr;
            if (ref_latents.size() > 0 && zero_timestep_refs) {
                // "index_timestep_zero" mode: use timestep = 0 for ref latents
                auto timestep_0 = ggml_scale(ctx->ggml_ctx, timestep, 0.0f);
                auto t_0        = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep_0, static_cast<int>(config.timestep_dim), 10000, 1000.f);
                t_0             = tmlp->forward(ctx, t_0);
                t_0             = ggml_reshape_3d(ctx->ggml_ctx, t_0, t_0->ne[0], 1, t_0->ne[1]);
                tvec_0          = tproj->forward(ctx, t_0);
            }

            // txtfusion (2 layerwise + 2 refiner blocks at width 2560) deliberately stays F32.
            // Its Linears ARE NVFP4, so it COULD run F16 — but it is not worth the risk:
            //   * it is not where the memory is. It runs on text tokens (a few hundred) x 12
            //     encoder layers, at width 2560 with a 6912 intermediate. That working set is
            //     ~2 orders of magnitude below the image stack's, which is the whole problem.
            //   * its consumer forces F32 anyway: txtmlp.1 / txtmlp.3 are BF16 in the shipping
            //     checkpoint, so txtfusion's output has to be F32 by the time it gets there.
            //     Running the middle in F16 would buy nothing and add two casts.
            //   * `projector` is a Linear(12 -> 1) applied across a permuted layer axis — a
            //     12-element reduction where F16 rounding is proportionally worst, feeding
            //     every downstream token. Bad place to spend precision for no VRAM.
            //
            // `txt_in` is that same result, already computed by forward_text() in a one-shot
            // graph and handed back as an F32 input. It is byte-identical to what the two calls
            // below produce — same ops, same weights, same order, and the D2H/H2D round trip is a
            // raw copy of F32 — so the only thing that changes is HOW OFTEN it is computed. When
            // it is absent (cache disabled, or an empty conditioning) the original inline path
            // runs unchanged.
            ggml_tensor* txt = txt_in;
            if (txt == nullptr) {
                txt = txtfusion->forward(ctx, context);
                txt = txtmlp->forward(ctx, txt);
            }
            int64_t txt_len = txt->ne[1];

            if (dit_f16) {
                txt = ggml_cast(ctx->ggml_ctx, txt, GGML_TYPE_F16);
            }

            // ggml_concat requires src0->type == src1->type == dst->type (concat.cu:287), so
            // both halves must have been cast by here.
            auto hidden_states = ggml_concat(ctx->ggml_ctx, txt, img, 1);
            int64_t ref_start  = hidden_states->ne[1] - ref_len;

            // LAST-BLOCK TOKEN TRIM. The slice below keeps only the image rows, so in block 27
            // `wo`, the gate/sigmoid multiply and the whole SwiGLU were being computed for the
            // text and reference rows and then thrown away — at 1024^2 with one reference that
            // is 4701 of 8797 rows, 53% of ~87% of a block.
            //
            // The layout is `concat(txt, img, refs)` and `ref_start = ne[1] - ref_len`, so the
            // image rows are txt_len .. txt_len+img_len == ref_start — the MIDDLE of the
            // sequence, hence a two-sided range rather than a prefix. Asserted below, because
            // getting it wrong corrupts the output silently rather than failing.
            //
            // ⚠️ NOT BIT-IDENTICAL under GGML_NVFP4_QUANT_TWOLEVEL=1 (which the krea2 service
            // sets). It is exact in real arithmetic — everything after attention is row-wise —
            // but the two-level activation quant takes a PER-TENSOR amax over the whole M x K
            // activation (nvfp4-cublaslt.cu, quant_act_kernel: the block scale is
            // ue4m3((amax/6)/per_tensor) and per_tensor is also carried in the GEMM alpha), and
            // dropping rows can lower that amax, which moves the rounding grid for the rows that
            // remain. The change can only SHRINK the amax, so the survivors are quantised no
            // worse than before; but the last bits of the output can differ, content-dependently
            // — if the largest activation happens to live in an image row it is bit-identical,
            // and that is not knowable in advance.
            //
            // DEFAULT OFF for that reason, and only that reason. Optimisations 1-3 in this change
            // ARE bit-identical, so they validate as a maxdiff-0 assertion against the previous
            // binary; leaving an output-changing edit switched on in the same build would make any
            // pixel difference ambiguous between "the trim working as designed" and "one of the
            // other three subtly wrong". KREA2_LAST_BLOCK_TRIM=1 turns it on as a single-flag A/B
            // once that first assertion has passed. Flip the default only with its own evidence.
            static const bool last_block_trim = krea2_env_flag("KREA2_LAST_BLOCK_TRIM", false);
            const int64_t trim_lo             = last_block_trim ? txt_len : -1;
            const int64_t trim_hi             = last_block_trim ? txt_len + img_len : -1;
            if (last_block_trim) {
                GGML_ASSERT(trim_hi == ref_start);
                GGML_ASSERT(trim_lo >= 0 && trim_hi > trim_lo && trim_hi <= hidden_states->ne[1]);
            }

            for (int i = 0; i < config.layers; ++i) {
                auto block          = std::dynamic_pointer_cast<KreaSingleStreamBlock>(blocks["blocks." + std::to_string(i)]);
                const bool is_last  = (i == config.layers - 1);
                hidden_states       = block->forward(ctx,
                                                     hidden_states,
                                                     tvec,
                                                     pe,
                                                     tvec_0,
                                                     ref_start,
                                                     is_last ? trim_lo : -1,
                                                     is_last ? trim_hi : -1);
                sd::ggml_graph_cut::mark_graph_cut(hidden_states, "krea2.blocks." + std::to_string(i), "hidden_states");
            }

            if (!last_block_trim) {
                hidden_states = ggml_ext_slice(ctx->ggml_ctx, hidden_states, 1, txt_len, txt_len + img_len);
            }
            GGML_ASSERT(hidden_states->ne[1] == img_len);
            // Leave the F16 stream before the head. Cast AFTER the slice, so only the image
            // tokens pay for it, and the text and ref tokens are dropped while still half
            // width. KreaLastLayer is F32 throughout: last.norm is an RMSNorm whose output
            // feeds Flux::modulate, and last.linear.weight is an F32 tensor that would refuse
            // an F16 activation outright. The sampler and VAE downstream want F32 anyway.
            if (dit_f16 && hidden_states->type != GGML_TYPE_F32) {
                hidden_states = ggml_cast(ctx->ggml_ctx, hidden_states, GGML_TYPE_F32);
            }
            hidden_states = last->forward(ctx, hidden_states, t);
            hidden_states = DiT::unpatchify_and_crop(ctx->ggml_ctx, hidden_states, H, W, config.patch_size, config.patch_size, true);
            return hidden_states;
        }
    };

    __STATIC_INLINE__ std::vector<float> gen_krea2_pe(int h,
                                                      int w,
                                                      int patch_size,
                                                      int bs,
                                                      int context_len,
                                                      float theta,
                                                      const std::vector<int>& axes_dim,
                                                      const std::vector<ggml_tensor*>& ref_latents,
                                                      Rope::RefIndexMode ref_index_mode) {
        auto txt_ids = Rope::gen_flux_txt_ids(bs, context_len, 3, {});
        auto img_ids = Rope::gen_flux_img_ids(h, w, patch_size, bs, 3, 0, 0, 0, false);
        auto ids     = Rope::concat_ids(txt_ids, img_ids, bs);
        if (ref_latents.size() > 0) {
            auto refs_ids = Rope::gen_refs_ids(patch_size, bs, 3, 1, ref_latents, ref_index_mode, 1.0f, false, 0);
            ids           = Rope::concat_ids(ids, refs_ids, bs);
        }
        return Rope::embed_nd(ids, bs, theta, axes_dim);
    }

    // Key for the RoPE table cache below: FNV-1a over EVERY argument gen_krea2_pe() reads.
    // That is what makes the cache exact by construction rather than by argument — a hit means
    // the inputs were identical, a miss just rebuilds.
    //
    // The reference latents contribute their SHAPES, not their contents, because that is all
    // Rope::gen_refs_ids() looks at (`ref->ne[1]` and `ref->ne[0]` feed gen_flux_img_ids, and
    // the running h/w offsets are derived from the same two). All four ne are mixed in anyway:
    // free, and it means a future gen_refs_ids that starts reading ne[2]/ne[3] cannot silently
    // alias. The reference COUNT is implicit in the per-ref mixing but is mixed explicitly too,
    // so [A] and [A, A] cannot collide with each other's ordering.
    __STATIC_INLINE__ uint64_t krea2_pe_cache_key(int h,
                                                  int w,
                                                  int patch_size,
                                                  int bs,
                                                  int context_len,
                                                  float theta,
                                                  const std::vector<int>& axes_dim,
                                                  const std::vector<ggml_tensor*>& ref_latents,
                                                  Rope::RefIndexMode ref_index_mode) {
        uint64_t hh = 1469598103934665603ull;
        hh          = krea2_fnv1a_val(hh, h);
        hh          = krea2_fnv1a_val(hh, w);
        hh          = krea2_fnv1a_val(hh, patch_size);
        hh          = krea2_fnv1a_val(hh, bs);
        hh          = krea2_fnv1a_val(hh, context_len);
        hh          = krea2_fnv1a_val(hh, theta);
        hh          = krea2_fnv1a_val(hh, static_cast<int>(ref_index_mode));
        hh          = krea2_fnv1a_val(hh, static_cast<uint64_t>(axes_dim.size()));
        for (int d : axes_dim) {
            hh = krea2_fnv1a_val(hh, d);
        }
        hh = krea2_fnv1a_val(hh, static_cast<uint64_t>(ref_latents.size()));
        for (const ggml_tensor* ref : ref_latents) {
            const uint64_t null_marker = 0;
            if (ref == nullptr) {
                hh = krea2_fnv1a_val(hh, null_marker);
                continue;
            }
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                hh = krea2_fnv1a_val(hh, ref->ne[d]);
            }
        }
        // Zero is the "nothing cached yet" sentinel, so never return it.
        return hh == 0 ? 1 : hh;
    }

    struct Krea2Runner : public DiffusionModelRunner {
        Krea2Config config;
        Krea2Model model;
        std::vector<float> pe_vec;

        // ROPE TABLE CACHE. gen_krea2_pe() ran inside build_graph, i.e. ONCE PER SAMPLING STEP.
        // At 1024^2 with one reference that is an 8797x3 transpose, three rope() passes
        // (~1.1M sin + 1.1M cos), an `emb` built out of 8797 separate std::vector<float>(256)
        // rows and a flatten() — order 45k heap allocations — all single-threaded and all on the
        // critical path AHEAD of any GPU work, followed by a ~9 MB H2D upload. Ten times a render.
        //
        // The positions do not change within a render, so the whole thing is pure. This is the
        // same fix (and the same keying pattern) as b6c0bc2 for LTX, which measured 310 ms and
        // 157 ms per call at 12k tokens; krea2 was simply missed. Keyed on every input the build
        // reads (see krea2_pe_cache_key), so a resolution change, a reference-count change or a
        // new request invalidates it and a hit is byte-identical to a rebuild by construction.
        //
        // KREA2_PE_CACHE=0 disables it; KREA2_PE_TIMING=1 logs the per-call cost. Run the two
        // together to get the raw uncached number for every step.
        uint64_t pe_key = 0;

        // TEXT-BRANCH CACHE — see Krea2Model::forward_text. N-way, because CFG evaluates two
        // conditionings per step and a batch evaluates more; a single slot would thrash and
        // recompute on every call, which is the bug this replaces, not a fix for it.
        struct TxtCacheEntry {
            uint64_t key = 0;
            sd::Tensor<float> txt;
        };
        static constexpr size_t KREA2_TXT_CACHE_SLOTS = 4;
        std::array<TxtCacheEntry, KREA2_TXT_CACHE_SLOTS> txt_cache;
        size_t txt_cache_next = 0;

        Krea2Runner(ggml_backend_t backend,
                    const String2TensorStorage& tensor_storage_map      = {},
                    const std::string prefix                            = "",
                    std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(Krea2Config::detect_from_weights(tensor_storage_map, prefix)) {
            model = Krea2Model(config);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "krea2";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        // GALLOCR RE-RESERVE PROBE. The text pass and the DiT graph share one gallocr
        // (free_compute_buffer=false on both), so a tiny text graph is followed by the big DiT
        // graph and ggml_gallocr_alloc_graph has to grow the buffer -- once per render, not once
        // per step, because after step 0 the text branch is a cache hit and never builds a graph.
        //
        // ggml's own evidence for that is unusable here: BOTH "reallocating buffers
        // automatically" (ggml-alloc.c:1070) and "reallocating <buft> buffer from size X to Y"
        // (ggml-alloc.c:944) sit inside `#ifndef NDEBUG`, and CMakeLists.txt forces Release, so
        // neither is compiled in. Hence this: KREA2_BUF_DIAG=1 prints the gallocr's actual buffer
        // size at each phase, which is the number those lines would have reported.
        void log_compute_buffer(const char* phase) {
            static const bool diag = krea2_env_flag("KREA2_BUF_DIAG", false);
            if (!diag || compute_allocr == nullptr) {
                return;
            }
            LOG_INFO("[krea2-buf] %s: compute buffer %.2f MiB",
                     phase,
                     ggml_gallocr_get_buffer_size(compute_allocr, 0) / (1024.0 * 1024.0));
        }

        // One-shot graph for the text branch. Deliberately builds NOTHING else: the point is that
        // its result depends on `context` alone, so it must not pick up a timestep or an x.
        ggml_cgraph* build_text_graph(const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf      = new_graph_custom(KREA2_GRAPH_SIZE);
            ggml_tensor* context = make_input(context_tensor);
            auto runner_ctx      = get_context();
            ggml_tensor* out     = model.forward_text(&runner_ctx, context);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        // Returns the cached txtfusion->txtmlp result for this conditioning, computing it on a
        // miss. nullptr means "not available — run the original inline path", which is the only
        // behaviour change when KREA2_TXT_CACHE=0 or the one-shot compute fails.
        //
        // Keyed on the conditioning BYTES, not on its address: the sampler holds one
        // SDCondition per branch for the whole render, but an address alone would let a later
        // render inherit a previous one's text through a recycled allocation. Hashing ~74 MB
        // costs a few ms (see krea2_bulk_hash) against a text branch worth ~2.5 TFLOP, and it
        // replaces a 74 MB context upload per step with a ~15 MB one.
        //
        // MUST be called BEFORE the outer GGMLRunner::compute — it runs a compute of its own,
        // and compute() resets the shared compute ctx.
        //
        // NULL/EMPTY CONDITIONING: a null `diffusion_params.context` arrives here as
        // tensor_or_empty()'s shared default-constructed tensor, so `empty()` is true and we bail
        // BEFORE touching data()/numel() — no hash of a null pointer, no zero-length digest. That
        // path then hits build_graph's pre-existing `GGML_ASSERT(!context_tensor.empty())`,
        // exactly as it did before this change; the DiT cannot run without conditioning either
        // way, so this adds no new failure mode. Note that an empty negative PROMPT is not an
        // empty conditioning TENSOR — the encoder still emits a full-length embedding of the
        // empty string — so the common t2i case takes the normal cached path, and at cfg 1.0
        // there is one conditioning, hence one text pass and nine hits per 10-step render.
        const sd::Tensor<float>* get_or_build_txt(int n_threads, const sd::Tensor<float>& context_tensor) {
            static const bool enabled = krea2_env_flag("KREA2_TXT_CACHE", true);
            if (!enabled || context_tensor.empty()) {
                return nullptr;
            }
            uint64_t key = krea2_bulk_hash(context_tensor.data(),
                                           static_cast<size_t>(context_tensor.numel()) * sizeof(float));
            key          = krea2_fnv1a_val(key, static_cast<uint64_t>(context_tensor.dim()));
            for (int64_t d : context_tensor.shape()) {
                key = krea2_fnv1a_val(key, d);
            }
            if (key == 0) {
                key = 1;
            }

            for (auto& entry : txt_cache) {
                if (entry.key == key && !entry.txt.empty()) {
                    return &entry.txt;
                }
            }

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_text_graph(context_tensor);
            };
            auto out = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            log_compute_buffer("after-text-pass");
            if (!out.has_value() || out->empty()) {
                LOG_WARN("krea2: text-branch precompute failed; falling back to the in-graph path");
                return nullptr;
            }

            // Round-robin eviction. With CFG the two conditionings settle into two slots on the
            // first step and every later step is a hit, so replacement policy barely matters —
            // round robin just avoids one branch permanently evicting the other at 2 slots.
            const size_t slot_index = txt_cache_next;
            txt_cache_next          = (txt_cache_next + 1) % KREA2_TXT_CACHE_SLOTS;
            TxtCacheEntry& slot     = txt_cache[slot_index];
            slot.txt                = std::move(*out);
            slot.key                = key;
            LOG_DEBUG("krea2: text branch computed (%" PRId64 " tokens), cached in slot %zu",
                      slot.txt.dim() >= 2 ? slot.txt.shape()[1] : static_cast<int64_t>(0),
                      slot_index);
            return &slot.txt;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 const RefImageParams& ref_image_params                   = REF_IMAGE_PRESETS.at("krea2_ostris_edit"),
                                 const sd::Tensor<float>* txt_tensor                      = nullptr) {
            ggml_cgraph* gf        = new_graph_custom(KREA2_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            // With the text branch hoisted, the raw conditioning is no longer part of this graph
            // at all -- neither the ops nor the ~74 MB upload. `txt` (~15 MB) takes its place.
            ggml_tensor* context = txt_tensor != nullptr ? nullptr : make_input(context_tensor);
            ggml_tensor* txt     = txt_tensor != nullptr ? make_input(*txt_tensor) : nullptr;

            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            // Text token count for the RoPE ids. txtfusion and txtmlp both preserve the token
            // axis (ne[1]), so the hoisted txt carries exactly the conditioning's token count --
            // this is the same number either way, not an approximation of it.
            const int64_t context_len = txt != nullptr ? txt->ne[1] : context->ne[1];

            const uint64_t want_pe_key = krea2_pe_cache_key(static_cast<int>(x->ne[1]),
                                                            static_cast<int>(x->ne[0]),
                                                            config.patch_size,
                                                            static_cast<int>(x->ne[3]),
                                                            static_cast<int>(context_len),
                                                            config.theta,
                                                            config.axes_dim,
                                                            ref_latents,
                                                            ref_image_params.ref_index_mode);
            static const bool pe_cache_enabled = krea2_env_flag("KREA2_PE_CACHE", true);
            static const bool pe_timing        = krea2_env_flag("KREA2_PE_TIMING", false);
            const bool pe_hit                  = pe_cache_enabled && pe_key == want_pe_key && !pe_vec.empty();
            if (pe_hit) {
                if (pe_timing) {
                    LOG_INFO("[krea2-pe] reused (key=%016" PRIx64 ")", want_pe_key);
                }
            } else {
                const int64_t t0 = pe_timing ? ggml_time_ms() : 0;
                pe_vec           = gen_krea2_pe(static_cast<int>(x->ne[1]),
                                                static_cast<int>(x->ne[0]),
                                                config.patch_size,
                                                static_cast<int>(x->ne[3]),
                                                static_cast<int>(context_len),
                                                config.theta,
                                                config.axes_dim,
                                                ref_latents,
                                                ref_image_params.ref_index_mode);
                if (pe_timing) {
                    LOG_INFO("[krea2-pe] built in %" PRId64 " ms (%zu floats, key=%016" PRIx64 ")",
                             ggml_time_ms() - t0,
                             pe_vec.size(),
                             want_pe_key);
                }
            }
            pe_key      = want_pe_key;
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = model.forward(&runner_ctx, x, timesteps, context, pe, ref_latents, ref_image_params.force_ref_timestep_zero, txt);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  const RefImageParams& ref_image_params            = REF_IMAGE_PRESETS.at("krea2_ostris_edit")) {
            // Resolved here, OUTSIDE get_graph: on a miss it runs a compute of its own, and
            // compute() resets the shared compute ctx, so it cannot happen during graph build.
            const sd::Tensor<float>* txt = get_or_build_txt(n_threads, context);
            auto get_graph               = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, ref_image_params, txt);
            };
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
            log_compute_buffer("after-dit-step");
            return result;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            static const std::vector<sd::Tensor<float>> empty_ref_latents;
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           diffusion_params.ref_latents && diffusion_params.ref_image_params.pass_to_dit ? *diffusion_params.ref_latents : empty_ref_latents,
                           diffusion_params.ref_image_params);
        }
    };
}  // namespace Krea2

#endif  // __SD_MODEL_DIFFUSION_KREA2_HPP__
