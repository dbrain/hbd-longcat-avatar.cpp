#ifndef __SD_MODEL_DIFFUSION_MODEL_HPP__
#define __SD_MODEL_DIFFUSION_MODEL_HPP__

#include <string>
#include <utility>
#include <variant>

#include "core/ggml_extend.hpp"
#include "core/tensor_ggml.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/ltx_relay.hpp"
#include "model_manager.h"

enum class RefImageResizeMode {
    NONE,
    LONGEST_SIDE,
    AREA,
};

struct RefImageParams {
    bool pass_to_vlm                   = false;
    bool pass_to_dit                   = true;
    Rope::RefIndexMode ref_index_mode  = Rope::RefIndexMode::FIXED;
    bool force_ref_timestep_zero       = false;
    bool resize_before_vae             = true;
    int vae_input_max_pixels           = -1;
    RefImageResizeMode vlm_resize_mode = RefImageResizeMode::AREA;
    int vlm_min_size                   = -1;
    int vlm_max_size                   = -1;
    bool resize_vae_to_target          = false;
    // Center-crop the reference to the TARGET aspect ratio before the resize that
    // resize_vae_to_target performs. Without it a mismatched-AR source is stretched;
    // the krea2_edit trainer/nodes crop (their "crop (legacy)" geometry) instead.
    bool crop_vae_to_target_ar = false;
    // Prefix each VLM vision block with "Picture N: ". Upstream's Krea2 conditioner
    // always does; the lbouaraba/conradlocke krea2_edit recipe trains WITHOUT it
    // (its template is bare <|vision_start|><|image_pad|>...<|vision_end|>), and the
    // grounding half of that recipe only lands if inference matches training.
    bool vlm_picture_labels = true;
    // Give the DiT reference tokens the SAME positional span as the target grid rather than
    // a unit-step sub-rectangle of it (krea2 only, today). Without it a reference that is not
    // the same latent grid as the target lands in the top-left corner of the positional plane
    // and the (i,j)<->(i,j) correspondence an edit adapter trains on is lost — which is what
    // resize_vae_to_target + crop_vae_to_target_ar exist to sidestep by forcing the grids equal.
    // With it, a reference of ANY aspect ratio or size spans the target, so it can be passed at
    // its native geometry. Off everywhere by default; deployed presets are unaffected.
    bool rescale_ref_ids = false;
    // The other way to put a small reference where the target is: keep the step at exactly 1
    // (in-distribution) and TRANSLATE the reference to the centre of the target plane instead
    // of leaving it in the top-left corner. Wins over rescale_ref_ids if both are set.
    //
    // MEASURED, 1920x1088 target, n=3 seeds x 2 subjects x 2 reference shapes: centring beats
    // spanning on the identity marker (square ref 3/3 vs 2/3; PORTRAIT ref 3/3 vs 0/3) and ties
    // it at 0/6 composition failures. Spanning has to apply a fractional, anisotropic step
    // (0.798 rows x 2.532 cols on the portrait case) that the adapter — trained at ref grid ==
    // target grid, i.e. step exactly 1.0 — has never seen. Centring is only a translation, so
    // it never leaves the training distribution. This is the one the restage preset carries.
    bool center_ref_ids = false;
    // Decode reference images at their OWN geometry instead of centre-cropping them to the
    // request's aspect ratio and rescaling them to the request's resolution.
    //
    // ⚠️ THIS IS RESOLVED BEFORE THE DECODE, not here — see sd_ref_image_args_want_native_geometry()
    // and the call site in SDGenerationParams::from_json_str(). By the time anything reads this
    // struct the reference bytes already exist, so flipping it here would be far too late. It is
    // carried on RefImageParams only so that the preset table is the single place the two halves
    // agree, and so `native_ref` is not rejected as an unknown arg.
    //
    // Off by default: for an IN-PLACE EDIT the crop is correct, because the reference IS the
    // canvas. It is wrong for RESTAGE, where the reference is an identity source and the crop is
    // silent data loss — a 768x1344 portrait into a 16:9 render keeps a 32% middle band, which
    // does not contain the head.
    bool native_ref_geometry = false;
};

const std::unordered_map<std::string, RefImageParams> REF_IMAGE_PRESETS = {
    {"flux_kontext", {false, true, Rope::RefIndexMode::FIXED, false, true, -1, RefImageResizeMode::NONE, -1, -1}},
    {"longcat", {true, true, Rope::RefIndexMode::FIXED, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"flux2", {false, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::NONE, -1, -1}},
    {"qwen", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"qwen_layered", {true, true, Rope::RefIndexMode::DECREASE, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"mage_flow", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::LONGEST_SIDE, -1, 384, true}},
    {"z_image_omni", {true, true, Rope::RefIndexMode::FIXED, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"krea2_ostris_edit", {true, true, Rope::RefIndexMode::INCREASE, true, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"krea2_edit", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::LONGEST_SIDE, 768, 768}},
    // conradlocke/krea2-identity-edit (v1.x) + lbouaraba/comfyui-krea2edit nodes.
    // Differences from "krea2_edit" above, all training-matched to those nodes:
    //   resize_vae_to_target + crop_vae_to_target_ar  the node fits the SOURCE to the
    //       target latent grid exactly (center-crop to the target AR, then resize), so
    //       ref grid == target grid and the anchored ref RoPE ids are the centered ones.
    //   vlm_picture_labels=false  its grounded-encode template has no "Picture N: ".
    //   vlm min/max 768           = the nodes' grounding_px default (trained 384-768).
    {"krea2_identity_edit", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::LONGEST_SIDE, 768, 768, true, true, false}},
    // Same adapter as krea2_identity_edit, different job: RESTAGE, where the reference is an
    // IDENTITY SOURCE whose aspect ratio has nothing to do with the output's. Three things have
    // to be true together, and it is worth being explicit about which does what, because they
    // were confused for each other for a long time:
    //   native_ref_geometry=true      the reference is decoded at its own size, so the identifying
    //       features are still IN THE INPUT. This is the one that saves the markers.
    //   resize_vae_to_target=false    ... and the preset does not then re-impose the target
    //   crop_vae_to_target_ar=false       geometry that native_ref_geometry just avoided.
    //   center_ref_ids=true           the reference is translated to the MIDDLE of the positional
    //       plane. Without it a native-size reference sits in the top-left corner, the model reads
    //       it as "a picture hanging here" rather than "this is the subject", and pastes it — 4/6
    //       cells came back as a diptych or a duplicated subject.
    // NOT rescale_ref_ids: spanning is the losing arm, see the comment on the field.
    // Also 30% cheaper than the edit preset at 1920x1088 (4,096 reference tokens, not 8,160),
    // because the reference is not upscaled to the target first.
    {"krea2_identity_restage", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::LONGEST_SIDE, 768, 768, false, false, false, /*rescale_ref_ids=*/false, /*center_ref_ids=*/true, /*native_ref_geometry=*/true}},
    {"cosmos_reference", {false, true, Rope::RefIndexMode::INCREASE, false, false, -1, RefImageResizeMode::NONE, -1, -1}},
};

struct UNetDiffusionExtra {
    int num_video_frames                           = -1;
    const std::vector<sd::Tensor<float>>* controls = nullptr;
    float control_strength                         = 0.f;
    const sd::Tensor<float>* ip_context            = nullptr;
    float ip_scale                                 = 1.f;
};

struct SkipLayerDiffusionExtra {
    const std::vector<int>* skip_layers = nullptr;
};

struct FluxDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const std::vector<int>* skip_layers = nullptr;
    const sd::Tensor<float>* pulid_id   = nullptr;
    float pulid_id_weight               = 1.0f;
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
    const sd::Tensor<float>* audio_positions = nullptr;
    bool skip_a2v                           = false;
    // TASS overlap reference conditioning (LTX-Best-Face-ID identity transfer).
    //
    // `ref_video_x` is one or more already-VAE-encoded reference latents packed on
    // the frame axis. They are patchified with the SAME patchify_proj as the target
    // and appended on the token axis, so a reference may carry its own spatial grid
    // (a 1536x1024 character sheet next to a 768x448 video) — after patchify the
    // sequence is flat and all geometry lives in the positions/source-id vectors.
    //
    // `video_source_ids` is per-token over (target tokens ++ reference tokens):
    // 0 for target (exact RoPE no-op), 2/3/4/... one distinct id per reference
    // subject. Null => no references, and the graph is identical to a build
    // without TASS.
    const sd::Tensor<float>* ref_video_x     = nullptr;
    const std::vector<float>* video_source_ids = nullptr;
    float tass_phase_scale                   = 1.f;
    // Prompt Relay. Null is the ordinary path: no cross-attention mask is
    // built, the flash fast path keeps its null-mask shape, and the graph is
    // byte-identical to a build without relay. The sampler nulls this out for
    // the unconditional pass (whose token ranges do not match) and once the
    // configured step fraction has elapsed.
    const sd::ltx_relay::Plan* relay = nullptr;
};

struct LongCatAvatarDiffusionExtra {
    // Sampler step controls the cross-step cond/text K/V caches.
    int step = -1;
};

struct MiniT2IDiffusionExtra {
    const sd::Tensor<float>* mask = nullptr;
};

struct HunyuanVideoDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const sd::Tensor<float>* byt5       = nullptr;
    const sd::Tensor<float>* vision     = nullptr;
    const sd::Tensor<float>* timestep_r = nullptr;
};

using DiffusionExtraParams = std::variant<std::monostate,
                                          UNetDiffusionExtra,
                                          SkipLayerDiffusionExtra,
                                          FluxDiffusionExtra,
                                          AnimaDiffusionExtra,
                                          WanDiffusionExtra,
                                          HiDreamO1DiffusionExtra,
                                          LTXAVDiffusionExtra,
                                          LongCatAvatarDiffusionExtra,
                                          MiniT2IDiffusionExtra,
                                          HunyuanVideoDiffusionExtra>;

struct DiffusionParams {
    const sd::Tensor<float>* x                        = nullptr;
    const sd::Tensor<float>* timesteps                = nullptr;
    const sd::Tensor<float>* context                  = nullptr;
    const sd::Tensor<float>* c_concat                 = nullptr;
    const sd::Tensor<float>* y                        = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents = nullptr;
    RefImageParams ref_image_params                   = {false, false, Rope::RefIndexMode::FIXED, false};
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
                         const std::string& prefix,
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          prefix(prefix) {}

    virtual sd::Tensor<float> compute(int n_threads,
                                      const DiffusionParams& diffusion_params) = 0;

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        get_param_tensors(tensors, prefix);
    }

    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                   const std::string& prefix) = 0;

    // Classic LTX-Video 0.9.x checkpoints are video-only. The LTX-2 default
    // remains audio+video; the 0.9 runner overrides this to prevent the
    // pipeline constructing a nonexistent audio stream.
    virtual bool has_audio_stream() const { return true; }
};

#endif  // __SD_MODEL_DIFFUSION_MODEL_HPP__
