#ifndef __SD_MODEL_DIFFUSION_MODEL_HPP__
#define __SD_MODEL_DIFFUSION_MODEL_HPP__

#include <string>
#include <utility>
#include <variant>

#include "core/ggml_extend.hpp"
#include "core/tensor_ggml.hpp"

struct UNetDiffusionExtra {
    int num_video_frames                           = -1;
    const std::vector<sd::Tensor<float>>* controls = nullptr;
    float control_strength                         = 0.f;
};

struct SkipLayerDiffusionExtra {
    const std::vector<int>* skip_layers = nullptr;
};

struct FluxDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const std::vector<int>* skip_layers = nullptr;
};

struct AnimaDiffusionExtra {
    const sd::Tensor<int32_t>* t5_ids   = nullptr;
    const sd::Tensor<float>* t5_weights = nullptr;
};

struct WanDiffusionExtra {
    const sd::Tensor<float>* vace_context = nullptr;
    float vace_strength                   = 1.f;
};

struct HiDreamO1DiffusionExtra {
    const sd::Tensor<int32_t>* input_ids                               = nullptr;
    const sd::Tensor<int32_t>* input_pos                               = nullptr;
    const sd::Tensor<int32_t>* token_types                             = nullptr;
    const sd::Tensor<int32_t>* vinput_mask                             = nullptr;
    const std::vector<std::pair<int, sd::Tensor<float>>>* image_embeds = nullptr;
};

struct LTXAVDiffusionExtra {
    const sd::Tensor<float>* audio_x         = nullptr;
    const sd::Tensor<float>* audio_timesteps = nullptr;
    int audio_length                         = 0;
    float frame_rate                         = 24.f;
    const sd::Tensor<float>* video_positions = nullptr;
    bool skip_a2v                            = false;  // A2V modality-guidance "mod" pass: skip audio<->video cross-attn
    // FIX A2 separable half-res relip reference (LTXAV_RELIP_REF_DOWNSCALE>1). When set, the
    // reference clip is a SEPARATE [W/N,H/N,ref,C] latent grid (NOT concatenated into the
    // sampler grid). The DiT patchifies it on its own and appends its tokens to the video
    // token sequence (positions for the ref block come from video_positions; its per-token
    // timesteps are appended as frozen t=0 inside build_graph), then slices them off before
    // unpatchify. nullptr / empty = legacy full-res concat path (N==1, byte-identical).
    const sd::Tensor<float>* video_reference = nullptr;
};

struct LongCatAvatarDiffusionExtra {
    int step = -1;  // sampler step index for the cond-frame K/V cache (lap-26); -1 = unknown
};

using DiffusionExtraParams = std::variant<std::monostate,
                                          UNetDiffusionExtra,
                                          SkipLayerDiffusionExtra,
                                          FluxDiffusionExtra,
                                          AnimaDiffusionExtra,
                                          WanDiffusionExtra,
                                          HiDreamO1DiffusionExtra,
                                          LTXAVDiffusionExtra,
                                          LongCatAvatarDiffusionExtra>;

struct DiffusionParams {
    const sd::Tensor<float>* x                        = nullptr;
    const sd::Tensor<float>* timesteps                = nullptr;
    const sd::Tensor<float>* context                  = nullptr;
    const sd::Tensor<float>* c_concat                 = nullptr;
    const sd::Tensor<float>* y                        = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents = nullptr;
    bool increase_ref_index                           = false;
    DiffusionExtraParams extra                        = std::monostate{};
};

template <typename T>
static inline const T* diffusion_extra_as(const DiffusionParams& params) {
    const auto* extra = std::get_if<T>(&params.extra);
    GGML_ASSERT(extra != nullptr);
    return extra;
}

template <typename T>
static inline const sd::Tensor<T>& tensor_or_empty(const sd::Tensor<T>* tensor) {
    static const sd::Tensor<T> kEmpty;
    return tensor != nullptr ? *tensor : kEmpty;
}

struct DiffusionModelRunner : public GGMLRunner {
protected:
    std::string prefix;

public:
    DiffusionModelRunner(ggml_backend_t backend,
                         ggml_backend_t params_backend,
                         const std::string& prefix)
        : GGMLRunner(backend, params_backend),
          prefix(prefix) {}

    virtual sd::Tensor<float> compute(int n_threads,
                                      const DiffusionParams& diffusion_params) = 0;

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        get_param_tensors(tensors, prefix);
    }

    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                   const std::string& prefix) = 0;

    // longcat-avatar additions. set_circular_axes is avatar-only (default no-op).
    // set_max_graph_vram_bytes and set_flash_attention_enabled MUST forward to the
    // GGMLRunner base by default — FluxRunner / LTXAVRunner et al. don't override them,
    // and a no-op here silently name-hides the real GGMLRunner setter. For
    // set_flash_attention_enabled that left flash_attn_enabled=false → unfused softmax
    // (flux2 lap-1). For set_max_graph_vram_bytes it left max_graph_vram_bytes=0 → the
    // graph-cut segmentation never engages → --max-vram is silently ignored and the whole
    // DiT is offloaded to the GPU at once → OOM on the LTX vid_gen path. Wrappers with
    // nested runners (avatar, hidream) still override to delegate to the inner runner.
    virtual void set_circular_axes(bool /*cx*/, bool /*cy*/) {}
    virtual void set_max_graph_vram_bytes(size_t max_vram_bytes) { GGMLRunner::set_max_graph_vram_bytes(max_vram_bytes); }
    virtual void set_flash_attention_enabled(bool enabled) { GGMLRunner::set_flash_attention_enabled(enabled); }

    // GGMLRunner param/adapter methods are non-virtual; make them virtual here so a
    // wrapper model (e.g. LongCatAvatarModel) can delegate to a nested runner. Default
    // forwards to GGMLRunner (correct for models whose params live in their own ctx).
    virtual bool   alloc_params_buffer()      { return GGMLRunner::alloc_params_buffer(); }
    virtual void   free_params_buffer()       { GGMLRunner::free_params_buffer(); }
    virtual void   free_compute_buffer()      { GGMLRunner::free_compute_buffer(); }
    virtual size_t get_params_buffer_size()   { return GGMLRunner::get_params_buffer_size(); }
    virtual void   set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) { GGMLRunner::set_weight_adapter(adapter); }
};

#endif  // __SD_MODEL_DIFFUSION_MODEL_HPP__
