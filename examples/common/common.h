#ifndef __EXAMPLES_COMMON_COMMON_H__
#define __EXAMPLES_COMMON_COMMON_H__

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "log.h"
#include "resource_owners.hpp"
#include "stable-diffusion.h"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

extern const char* const modes_str[];
#define SD_ALL_MODES_STR "img_gen, vid_gen, convert, upscale, metadata"

enum SDMode {
    IMG_GEN,
    VID_GEN,
    CONVERT,
    UPSCALE,
    METADATA,
    MODE_COUNT
};

struct StringOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::string* target;
};

struct IntOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    int* target;
};

struct FloatOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    float* target;
};

struct BoolOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    bool keep_true;
    bool* target;
};

struct ManualFunction {
    std::function<int(int, const char**, int, bool&)> _func;

    ManualFunction() = default;

    ManualFunction(std::function<int(int argc, const char** argv, int index, bool& valid)> func)
        : _func(std::move(func)) {
    }

    template <typename F>
    ManualFunction(F func)
        : _func(make_function(func)) {
    }

    int operator()(int argc, const char** argv, int index, bool& valid) const {
        return _func(argc, argv, index, valid);
    }

private:
    template <typename F>
    static std::function<int(int, const char**, int, bool&)> make_function(F func) {
        if constexpr (std::is_invocable_v<F, int, const char**, int, bool&>) {
            return func;
        } else {
            return [func](int argc, const char** argv, int index, bool&) {
                return func(argc, argv, index);
            };
        }
    }
};

struct ManualOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    ManualFunction cb;
};

struct ArgOptions {
    std::vector<StringOption> string_options;
    std::vector<IntOption> int_options;
    std::vector<FloatOption> float_options;
    std::vector<BoolOption> bool_options;
    std::vector<ManualOption> manual_options;

    static std::string wrap_text(const std::string& text, size_t width, size_t indent);
    void print() const;
};

bool parse_options(int argc, const char** argv, const std::vector<ArgOptions>& options_list);
bool decode_base64_image(const std::string& encoded_input,
                         int target_channels,
                         int expected_width,
                         int expected_height,
                         SDImageOwner& out_image);

struct SDContextParams {
    int n_threads = -1;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string uncond_diffusion_model_path;
    std::string embeddings_connectors_path;
    std::string vae_path;
    std::string vae_format = "auto";
    std::string audio_vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string control_net_path;
    std::string embedding_dir;
    std::string photo_maker_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    std::string lora_model_dir = ".";
    std::string hires_upscalers_dir;

    std::map<std::string, std::string> embedding_map;
    std::vector<sd_embedding_t> embedding_vec;

    rng_type_t rng_type         = CUDA_RNG;
    rng_type_t sampler_rng_type = RNG_TYPE_COUNT;
    bool offload_params_to_cpu  = false;
    float max_vram              = 0.f;
    bool stream_layers          = false;
    std::string backend;
    std::string params_backend;
    bool enable_mmap           = false;
    bool control_net_cpu       = false;
    bool clip_on_cpu           = false;
    bool vae_on_cpu            = false;
    bool flash_attn            = false;
    bool diffusion_flash_attn  = false;
    bool diffusion_conv_direct = false;
    bool vae_conv_direct       = false;

    bool circular   = false;
    bool circular_x = false;
    bool circular_y = false;

    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask  = false;
    int chroma_t5_mask_pad   = 1;

    bool qwen_image_zero_cond_t = false;

    prediction_t prediction           = PREDICTION_COUNT;
    lora_apply_mode_t lora_apply_mode = LORA_APPLY_AUTO;

    bool force_sdxl_vae_conv_scale = false;

    float flow_shift = INFINITY;
    ArgOptions get_options();
    void build_embedding_map();
    bool resolve(SDMode mode);
    bool validate(SDMode mode);
    bool resolve_and_validate(SDMode mode);
    std::string to_string() const;
    sd_ctx_params_t to_sd_ctx_params_t(bool vae_decode_only, bool free_params_immediately, bool taesd_preview);
};

struct SDGenerationParams {
    // User-facing input fields.
    std::string prompt;
    std::string negative_prompt;
    int clip_skip              = -1;  // <= 0 represents unspecified
    int width                  = -1;
    int height                 = -1;
    int batch_count            = 1;
    int64_t seed               = 42;
    float strength             = 0.75f;
    float control_strength     = 0.9f;
    bool auto_resize_ref_image = true;
    bool increase_ref_index    = false;
    bool embed_image_metadata  = true;

    std::string init_image_path;
    std::string end_image_path;
    std::string mask_image_path;
    std::string control_image_path;
    std::vector<std::string> ref_image_paths;
    // LTXAV multi-keyframe conditioning (--keyframe PATH@FRAME, repeatable): image paths and their
    // VIDEO (pixel) frame indices in [0, video_frames), applied as frozen 1-frame guides.
    std::vector<std::string> keyframe_paths;
    std::vector<int>         keyframe_indices;
    std::string control_video_path;
    std::string audio_path;  // LongCat-Avatar audio-driven lip-sync (16kHz mono wav)
    std::string drive_audio_path;  // LTX-2.3 audio-driven lip-sync (16kHz wav, needs encoder -ENC audio-vae)

    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    std::string extra_sample_args;
    std::string high_noise_extra_sample_args;
    std::vector<int> skip_layers            = {7, 8, 9};
    std::vector<int> high_noise_skip_layers = {7, 8, 9};

    std::vector<float> custom_sigmas;

    std::string cache_mode;
    std::string cache_option;
    std::string scm_mask;
    bool scm_policy_dynamic = true;
    sd_cache_params_t cache_params{};

    float moe_boundary                   = 0.875f;
    int video_frames                     = 1;
    // LongCat-Avatar continuation chaining (multi-segment clips).
    int segments                         = 1;   // >1 -> chain N resident segments
    int cont_cond_frames                 = 13;  // prior-segment tail frames used as conditioning
    std::string cont_latent_path;               // LTXAV file-based latent continuation: prior seg's saved video latent
    int cont_latent_take                 = 3;   // how many tail latent frames to use as the motion-carrying overlap
    int ltx_chain_segments               = 0;   // LTXAV in-process N-segment chaining: >0 -> render N segments, DiT resident
    std::string ltx_chain_prompts_path;         // LTXAV chain: file with one prompt per line (per-segment "director"); empty -> reuse -p for all
    std::string ltx_chain_audio_dir;            // LTXAV chain: dir of pre-sliced per-segment 16k wavs (aud_<seg>.wav) for lip-sync drive; empty -> no audio
    std::string cont_anchor_path;               // LTXAV appearance anchor: latent whose frame0 = original char (anti-drift)
    int cont_ref_img_index               = 10;  // continuation ref-anchor temporal grid position (generate_avc default)
    int cont_mask_frame_range            = 3;   // continuation noise-near-ref attention carve-out half-width
    // LTXAV relip (lip-dub) RUNTIME knobs — bridged to the shared engine via env EVERY render
    // (apply_ltx_relip_env), so the warm resident worker honours per-request values and never
    // bleeds a prior render's value (the sticky-setenv bug). Defaults == the engine defaults.
    float a2v_guidance                   = 1.0f; // audio->video modality guidance (1=faithful/fast; owner up to 3)
    float a2v_ramp_end                   = 1.0f; // a2v guidance ramp: fraction of (scale-1) kept at lowest-sigma step
    // Production Relip keeps the reference at full spatial resolution for identity, then
    // retains every fourth latent frame.  This is consumed by the video-worker env bridge;
    // a request may still explicitly supply another positive stride.
    int   relip_ref_tstride              = 4;    // temporal subsample of the relip reference (identity-safe token cut)
    // FEATURE 2 (NAG — Normalized Attention Guidance) RUNTIME knobs, bridged to the engine via env
    // (apply_ltx_relip_env). Attention-space negative guidance on the LTX video text cross-attn.
    // Default OFF (scale 0). Workflow values: scale 14, alpha 0.35, tau 2.5.
    float nag_scale                      = 0.0f; // 0 = NAG OFF; Denoise-AI workflow uses 14 (see CPP-CHANGES.md scale-convention note)
    float nag_alpha                      = 0.35f;// blend weight of the NAG-guided output vs positive
    float nag_tau                        = 2.5f; // per-token L2-norm clamp ceiling (z_ext norm <= tau*||z_pos||)
    float nag_until_sigma                = 0.9f; // apply NAG only while sigma >= this (early/high-noise steps; workflow S1-only)
    // LongCat-Avatar mouth-exaggeration knobs (RUNTIME).
    float audio_mouth_scale              = 1.0f; // scale the audio->face residual (1.0 = unchanged)
    float audio_lowpass                  = 0.0f; // input-wav low-pass cutoff Hz (0 = off)
    int fps                              = 16;
    float vace_strength                  = 1.f;
    sd_tiling_params_t vae_tiling_params = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    std::string extra_tiling_args;

    std::string pm_id_images_dir;
    std::string pm_id_embed_path;
    float pm_style_strength = 20.f;

    int upscale_repeats   = 1;
    int upscale_tile_size = 128;

    bool hires_enabled         = false;
    std::string hires_upscaler = "Latent";
    std::string hires_upscaler_model_path;
    float hires_scale              = 2.f;
    int hires_width                = 0;
    int hires_height               = 0;
    int hires_steps                = 0;
    float hires_denoising_strength = 0.7f;
    int hires_upscale_tile_size    = 128;
    std::vector<float> hires_custom_sigmas;
    // FEATURE 1 (--hires-lora): a SECOND LoRA set applied only on the hires/refine pass, so the
    // refine can run a different distillation/detailer strength than the base pass (Denoise-AI
    // workflow: distill@0.65 base -> distill@0.8 + detailer@0.7 refine). `hires_lora_spec` is the
    // raw CLI/JSON spec ("<lora:name:strength>[,name:strength...]"); it is parsed + path-resolved
    // in resolve() (needs lora_model_dir) into `hires_lora_map` (resolved-path -> strength), then
    // flattened into `hires_lora_vec` (the sd_lora_t backing storage) in to_sd_vid_gen_params_t().
    std::string hires_lora_spec;
    std::map<std::string, float> hires_lora_map;

    std::map<std::string, float> lora_map;
    std::map<std::string, float> high_noise_lora_map;

    // Derived and normalized fields.
    std::string prompt_with_lora;  // for metadata record only
    std::vector<sd_lora_t> lora_vec;
    std::vector<sd_lora_t> hires_lora_vec;  // FEATURE 1: backing storage for params.hires.loras
    sd_hires_upscaler_t resolved_hires_upscaler;

    // Owned execution payload.
    SDImageOwner init_image;
    SDImageOwner end_image;
    std::vector<SDImageOwner> keyframe_images;  // LTXAV multi-keyframe owned images
    std::vector<SDImageOwner> ref_images;
    SDImageOwner mask_image;
    SDImageOwner control_image;
    std::vector<SDImageOwner> pm_id_images;
    std::vector<SDImageOwner> control_frames;

    // Backing storage for sd_img_gen_params_t view fields.
    std::vector<sd_image_t> ref_image_views;
    std::vector<sd_image_t> pm_id_image_views;
    std::vector<sd_image_t> control_frame_views;
    std::vector<sd_image_t> keyframe_views;  // backing storage for sd_vid_gen_params_t.keyframes

    SDGenerationParams();
    SDGenerationParams(const SDGenerationParams& other)                = default;
    SDGenerationParams& operator=(const SDGenerationParams& other)     = default;
    SDGenerationParams(SDGenerationParams&& other) noexcept            = default;
    SDGenerationParams& operator=(SDGenerationParams&& other) noexcept = default;
    ArgOptions get_options();
    bool from_json_str(const std::string& json_str,
                       const std::function<std::string(const std::string&)>& lora_path_resolver = {});
    bool initialize_cache_params();
    void extract_and_remove_lora(const std::string& lora_model_dir);
    bool width_and_height_are_set() const;
    void set_width_and_height_if_unset(int w, int h);
    int get_resolved_width() const;
    int get_resolved_height() const;
    bool resolve(const std::string& lora_model_dir, const std::string& hires_upscalers_dir, bool strict = false);
    bool validate(SDMode mode);
    bool resolve_and_validate(SDMode mode,
                              const std::string& lora_model_dir,
                              const std::string& hires_upscalers_dir,
                              bool strict = false);
    // A2-safe env bridge: setenv the LTXAV relip knobs (a2v_guidance / a2v_ramp_end /
    // relip_ref_tstride) to THIS request's values, overwriting any stale value from a prior
    // warm-worker render. Call once per render before generate_video(_chain). Inert for non-LTX.
    void apply_ltx_relip_env() const;
    sd_img_gen_params_t to_sd_img_gen_params_t();
    sd_vid_gen_params_t to_sd_vid_gen_params_t();
    std::string to_string() const;
};

std::string version_string();
std::string build_sdcpp_image_metadata_json(const SDContextParams& ctx_params,
                                            const SDGenerationParams& gen_params,
                                            int64_t seed,
                                            SDMode mode = IMG_GEN);
std::string get_image_params(const SDContextParams& ctx_params,
                             const SDGenerationParams& gen_params,
                             int64_t seed,
                             SDMode mode = IMG_GEN);

#endif  // __EXAMPLES_COMMON_COMMON_H__
