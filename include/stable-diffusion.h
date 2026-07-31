#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

// The LongCat avatar model has a FIXED native save_fps: video frame k always carries the
// mouth shape for audio time k/SD_AVATAR_NATIVE_FPS, independent of the requested output fps.
// The audio window config and the output-fps default must agree on this value or lipsync
// drifts (muxing at 16 instead of 25 plays the mouth at 0.64x). Defined once so the engine
// and the CLI/server layer cannot disagree.
#define SD_AVATAR_NATIVE_FPS 25

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
    CPU_RNG,
    RNG_TYPE_COUNT
};

enum sample_method_t {
    EULER_SAMPLE_METHOD,
    EULER_A_SAMPLE_METHOD,
    HEUN_SAMPLE_METHOD,
    DPM2_SAMPLE_METHOD,
    DPMPP2S_A_SAMPLE_METHOD,
    DPMPP2M_SAMPLE_METHOD,
    DPMPP2Mv2_SAMPLE_METHOD,
    IPNDM_SAMPLE_METHOD,
    IPNDM_V_SAMPLE_METHOD,
    LCM_SAMPLE_METHOD,
    DDIM_TRAILING_SAMPLE_METHOD,
    TCD_SAMPLE_METHOD,
    RES_MULTISTEP_SAMPLE_METHOD,
    RES_2S_SAMPLE_METHOD,
    ER_SDE_SAMPLE_METHOD,
    EULER_CFG_PP_SAMPLE_METHOD,
    EULER_A_CFG_PP_SAMPLE_METHOD,
    EULER_GE_SAMPLE_METHOD,
    DPMPP2M_SDE_SAMPLE_METHOD,
    DPMPP2M_SDE_BT_SAMPLE_METHOD,
    SAMPLE_METHOD_COUNT
};

enum scheduler_t {
    DISCRETE_SCHEDULER,
    KARRAS_SCHEDULER,
    EXPONENTIAL_SCHEDULER,
    AYS_SCHEDULER,
    GITS_SCHEDULER,
    SGM_UNIFORM_SCHEDULER,
    SIMPLE_SCHEDULER,
    SMOOTHSTEP_SCHEDULER,
    KL_OPTIMAL_SCHEDULER,
    LCM_SCHEDULER,
    BONG_TANGENT_SCHEDULER,
    LTX2_SCHEDULER,
    LOGIT_NORMAL_SCHEDULER,
    FLUX2_SCHEDULER,
    FLUX_SCHEDULER,
    BETA_SCHEDULER,
    SCHEDULER_COUNT
};

enum prediction_t {
    EPS_PRED,
    V_PRED,
    EDM_V_PRED,
    FLOW_PRED,
    FLUX_FLOW_PRED,
    SEFI_FLOW_PRED,
    MINIT2I_FLOW_PRED,
    PREDICTION_COUNT
};

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // SD_TYPE_Q4_0_4_8 = 32,
    // SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0 = 34,
    SD_TYPE_TQ2_0 = 35,
    // SD_TYPE_IQ4_NL_4_4 = 36,
    // SD_TYPE_IQ4_NL_4_8 = 37,
    // SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4 = 39,  // MXFP4 (1 block)
    SD_TYPE_NVFP4 = 40,  // NVFP4 (4 blocks, E4M3 scale)
    SD_TYPE_Q1_0  = 41,
    SD_TYPE_COUNT = 42,
};

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

enum preview_t {
    PREVIEW_NONE,
    PREVIEW_PROJ,
    PREVIEW_TAE,
    PREVIEW_VAE,
    PREVIEW_COUNT
};

enum lora_apply_mode_t {
    LORA_APPLY_AUTO,
    LORA_APPLY_IMMEDIATELY,
    LORA_APPLY_AT_RUNTIME,
    LORA_APPLY_MODE_COUNT,
};

typedef struct {
    bool enabled;
    bool temporal_tiling;
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
    float rel_size_x;
    float rel_size_y;
    const char* extra_tiling_args;
} sd_tiling_params_t;

typedef struct {
    const char* name;
    const char* path;
} sd_embedding_t;

enum sd_vae_format_t {
    SD_VAE_FORMAT_AUTO = -1,
    SD_VAE_FORMAT_FLUX,
    SD_VAE_FORMAT_SD3,
    SD_VAE_FORMAT_FLUX2,
    SD_VAE_FORMAT_WAN,
    SD_VAE_FORMAT_COUNT,
};

typedef struct {
    const char* model_path;
    const char* clip_l_path;
    const char* clip_g_path;
    const char* clip_vision_path;
    const char* t5xxl_path;
    const char* llm_path;
    const char* llm_vision_path;
    const char* diffusion_model_path;
    const char* high_noise_diffusion_model_path;
    const char* uncond_diffusion_model_path;
    const char* embeddings_connectors_path;
    const char* vae_path;
    const char* audio_vae_path;
    const char* taesd_path;
    const char* control_net_path;
    const char* motion_module_path;
    const sd_embedding_t* embeddings;
    uint32_t embedding_count;
    const char* photo_maker_path;
    const char* pulid_weights_path;
    const char* tensor_type_rules;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type;
    enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode;
    bool enable_mmap;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool tae_preview_only;
    bool diffusion_conv_direct;
    bool vae_conv_direct;
    bool force_sdxl_vae_conv_scale;
    enum sd_vae_format_t vae_format;
    const char* max_vram;  // GiB budget or backend assignment spec for graph-cut segmented param offload (0 = disabled, -1 = auto)
    bool stream_layers;  // Enable residency+prefetch streaming on top of --max-vram (no effect without --max-vram)
    bool eager_load;  // Load all params into the params backend at model-load time instead of lazily on first use
    const char* backend;
    const char* params_backend;
    const char* split_mode;  // weight distribution for multi-device modules: layer (default) or row, or per-module assignments e.g. "diffusion=row"
    bool auto_fit;
    const char* rpc_servers;
    const char* model_args;
} sd_ctx_params_t;

typedef struct {
    uint32_t sample_rate;
    uint32_t channels;
    uint64_t sample_count;
    float* data;
} sd_audio_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct {
    int* layers;
    size_t layer_count;
    float layer_start;
    float layer_end;
    float scale;
} sd_slg_params_t;

typedef struct {
    float txt_cfg;
    float img_cfg;
    float distilled_guidance;
    sd_slg_params_t slg;
} sd_guidance_params_t;

typedef struct {
    sd_guidance_params_t guidance;
    enum scheduler_t scheduler;
    enum sample_method_t sample_method;
    int sample_steps;
    float eta;
    int shifted_timestep;
    float* custom_sigmas;
    int custom_sigmas_count;
    float flow_shift;
    const char* extra_sample_args;
} sd_sample_params_t;

typedef struct {
    sd_image_t* id_images;
    int id_images_count;
    const char* id_embed_path;
    float style_strength;
} sd_pm_params_t;  // photo maker

typedef struct {
    const char* id_embedding_path;
    float id_weight;
} sd_pulid_params_t;

enum sd_cache_mode_t {
    SD_CACHE_DISABLED = 0,
    SD_CACHE_EASYCACHE,
    SD_CACHE_UCACHE,
    SD_CACHE_DBCACHE,
    SD_CACHE_TAYLORSEER,
    SD_CACHE_CACHE_DIT,
    SD_CACHE_SPECTRUM,
};

typedef struct {
    enum sd_cache_mode_t mode;
    float reuse_threshold;
    float start_percent;
    float end_percent;
    float error_decay_rate;
    bool use_relative_threshold;
    bool reset_error_on_compute;
    int Fn_compute_blocks;
    int Bn_compute_blocks;
    float residual_diff_threshold;
    int max_warmup_steps;
    int max_cached_steps;
    int max_continuous_cached_steps;
    int taylorseer_n_derivatives;
    int taylorseer_skip_interval;
    const char* scm_mask;
    bool scm_policy_dynamic;
    float spectrum_w;
    int spectrum_m;
    float spectrum_lam;
    int spectrum_window_size;
    float spectrum_flex_window;
    int spectrum_warmup_steps;
    float spectrum_stop_percent;
} sd_cache_params_t;

typedef struct {
    bool is_high_noise;
    float multiplier;
    const char* path;
} sd_lora_t;

enum sd_hires_upscaler_t {
    SD_HIRES_UPSCALER_NONE,
    SD_HIRES_UPSCALER_LATENT,
    SD_HIRES_UPSCALER_LATENT_NEAREST,
    SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT,
    SD_HIRES_UPSCALER_LATENT_ANTIALIASED,
    SD_HIRES_UPSCALER_LATENT_BICUBIC,
    SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED,
    SD_HIRES_UPSCALER_LANCZOS,
    SD_HIRES_UPSCALER_NEAREST,
    SD_HIRES_UPSCALER_MODEL,
    SD_HIRES_UPSCALER_COUNT,
};

typedef struct {
    bool enabled;
    enum sd_hires_upscaler_t upscaler;
    const char* model_path;
    float scale;
    int target_width;
    int target_height;
    int steps;
    float denoising_strength;
    int upscale_tile_size;
    float* custom_sigmas;
    int custom_sigmas_count;
    // Refine stages may use a sampler and CFG distinct from the base pass.
    // SAMPLE_METHOD_COUNT and NAN retain the legacy inherited values.
    enum sample_method_t sample_method;
    float cfg;
} sd_hires_params_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t* ref_images;
    int ref_images_count;
    const char* ref_image_args;
    sd_image_t mask_image;
    int width;
    int height;
    sd_sample_params_t sample_params;
    float strength;
    int64_t seed;
    int batch_count;
    sd_image_t control_image;
    float control_strength;
    sd_pm_params_t pm_params;
    sd_pulid_params_t pulid_params;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
    int qwen_image_layers;
    bool circular_x;
    bool circular_y;
} sd_img_gen_params_t;

// Prompt Relay beat (arXiv 2604.10030). The shot's `prompt` stays the global
// setting and conditions every frame with zero penalty; each beat is a short
// clause that is given cross-attention priority around its own moment. Fewer
// than one beat leaves the ordinary null-mask path byte-identical.
typedef struct {
    // Pixel-frame index on the shot's VISIBLE timeline -- the frame a viewer sees, counting from
    // the start of the shot as it appears in the finished clip. On a continuation shot the chain
    // adds that shot's seam drop internally, so callers never handle the trim.
    int frame;
    const char* text;
    // Multiplies the attention penalty. Zero or negative selects 1.0.
    float strength;
    // Flat-top half-width in seconds. Negative selects the paper's ablated
    // best, w = L - 2 latent frames, i.e. a two-latent-frame crossfade.
    float window;
} sd_ltx_beat_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    // Prompt Relay. `prompt` remains the global anchor; these are the timed
    // sub-prompts. A non-empty list is REQUIRED to have a non-empty global
    // prompt: the zero-penalty global tokens are what keeps the masked softmax
    // from degenerating far away from every beat.
    const sd_ltx_beat_t* beats;
    int beat_count;
    // Gaussian tail floor. Zero selects 0.01 (the paper uses 0.001, which is
    // near-hard partitioning at our latent-frame counts).
    float relay_eps;
    // Separate floor for the audio cross-attention stream. Zero inherits
    // relay_eps; a negative value disables the audio relay mask entirely.
    float relay_audio_eps;
    // Fraction of the base sampling schedule that carries the mask. Semantic
    // layout is decided early, so a value below one costs less. Zero selects 1.
    float relay_steps_frac;
    sd_image_t init_image;
    sd_image_t end_image;
    // LTXAV frame-pinned image guides. Each image is frozen at the matching
    // pixel-frame index on the generated timeline. A zero count preserves the
    // ordinary t2v/i2v/continuation paths.
    sd_image_t* keyframes;
    int* keyframe_frame_indices;
    int keyframes_size;
    // LTXAV TASS overlap reference conditioning (LTX-Best-Face-ID character
    // sheets). Each image is VAE encoded AT THE RESOLUTION IT IS HANDED IN --
    // the caller decides between the sheet's native resolution and the render
    // bucket -- then appended on the DiT token axis with its own rotary source
    // tag, so a shot can hold an identity without spending an i2v guide frame.
    // A zero count leaves every existing path bit-identical.
    sd_image_t* character_refs;
    // Optional per-reference rotary source ids. A null pointer, or a
    // non-positive entry, assigns 2, 3, 4, ... in array order. Zero and one are
    // reserved (zero is the target's exact no-op tag).
    int* character_ref_source_ids;
    int character_refs_size;
    // Optional per-reference SEGMENT SCOPE for chained renders, as two parallel
    // flat arrays: `character_ref_segment_counts` holds one count per reference
    // (`character_refs_size` entries) and `character_ref_segments` is their
    // concatenation in reference order, so reference i owns the counts[i] entries
    // starting at sum(counts[0..i-1]).
    //
    // A NULL counts pointer means every reference applies to every segment, which
    // is what every caller predating this field gets. A count of ZERO scopes that
    // reference to no segment at all -- deliberately distinct from the null case,
    // never a synonym for "all". A segment left with no references in scope
    // renders exactly as if the request carried no character_refs.
    //
    // Indices are in rendered-segment index space and are ignored outside
    // generate_video_chain, where a single render is the whole request.
    int* character_ref_segments;
    int* character_ref_segment_counts;
    // Rotary source-phase multiplier. NEGATIVE means "unsupplied" and selects the
    // trained default of 1.0; ZERO is a meaningful value that selects the UNTAGGED
    // layout (phase = source_id * scale * theta^-d/L collapses to an exact no-op, so
    // the references sit on the target's own RoPE grid with nothing marking them).
    // Echo- and MSR-derived weights never saw a phase tag in training and need zero.
    float tass_phase_scale;
    // LTXAV MSR (Licon Multiple-Subject-Reference) in-context reference STRIP.
    //
    // Unlike `character_refs` -- a set of stills, each encoded to one latent frame --
    // this is a short VIDEO composited to the RENDER RESOLUTION and encoded in ONE
    // VAE pass, so a subject occupies whole latent frames rather than a single slot.
    // The strip is appended by the same TASS overlap path, so it composes with every
    // conditioning layout and costs nothing when `msr_frames` is zero.
    //
    // `msr_frames` must be 1 modulo 8 (the checkpoint's menu is 17/25/33/41/49/57/65);
    // zero disables the whole path. `msr_background` is REQUIRED when enabled -- it is
    // the substrate every frame starts from, not one slot among many.
    sd_image_t* msr_background;
    sd_image_t* msr_subjects;
    int msr_subjects_size;
    int msr_frames;
    // Optional SEGMENT SCOPE for the strip, in rendered-segment index space -- the same
    // space as `character_ref_segments`. A NULL pointer means every segment, which is what
    // a single render and every caller predating this field gets. A shot that scopes the
    // strip out takes the untouched no-reference path, so injecting a location into shot 2
    // leaves shot 1 bit-identical to a request that never mentioned MSR.
    int* msr_segments;
    int msr_segments_size;
    // REFERENCE HEAD-FRAME TRIM.
    //
    // A TASS reference sits at the target's latent-frame-0 RoPE address, and on some
    // checkpoints the decoder hands that address straight back: the opening pixel frame
    // of a bare t2v shot renders the reference VERBATIM. With a location plate in the
    // set frame 0 is the empty plate (which reads as "the subject is missing"); with only
    // a character sheet it is that sheet's own source photo, background and all.
    //
    // The contaminated run is measured in PIXEL frames and is 1 + 8*(K-1), where K is the
    // largest reference's LATENT frame count. Reference COUNT does not change it (verified
    // at 1, 2 and 4 references): every reference is given the same latent-frame origin, and
    // latent frame 0 is the only latent frame that decodes to a single pixel frame, because
    // ltxv_latent_corner_to_pixel_frame() is max(0, 8t-7). For stills (K == 1) that is
    // exactly ONE pixel frame.
    //
    //   0  -- OFF. The default, and byte-identical to a build that never had this field.
    //   -1 -- AUTO. The ENGINE derives 1 + 8*(K-1) from the references it actually encoded,
    //         so no caller ever hard-codes the formula.
    //   >0 -- trim exactly that many pixel frames.
    //
    // Caller-controlled rather than automatic because it is CHECKPOINT-specific: echo-e50
    // leaks, msr-v2 and echo-full do not. It additionally SELF-GATES to a no-op on any shot
    // that already pins frame 0 -- an i2v init image, a keyframe at index 0, or a
    // continuation tail -- and on any shot carrying no references at all; i2v and
    // continuation shots were measured clean. Each gate logs.
    //
    // The shot is TRIMMED, not re-rendered: it returns (N - trim) frames. Rendering the
    // frames back would cost a whole latent frame (LTX requires frames % 8 == 1, so the
    // smallest legal increase is 8 pixel frames) and that VRAM is not available. A matching
    // trim/fps seconds is taken off the HEAD of the OUTPUT audio so A/V stays exact.
    int reference_head_trim;
    sd_image_t* control_frames;
    int control_frames_size;
    int width;
    int height;
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    float moe_boundary;
    float strength;
    // LTX video-to-video mode. Zero is the production LipDub/relip path: the
    // source clip is VAE encoded and appended as frozen, timeline-aligned
    // reference tokens while a supplied drive-audio latent controls the mouth.
    // One selects pixel-source SDEdit and two selects guide-edit, which also
    // accepts a saved video-latent source.
    int v2v_mode;
    // LipDub timeline-reference temporal stride. One preserves every latent
    // frame; larger values retain every nth reference frame.
    int relip_ref_tstride;
    // LTXAV audio-to-video modality guidance. One is inert; larger values add
    // a second audio-decoupled DiT forward that increases articulation.
    float a2v_guidance;
    float a2v_ramp_end;
    bool lipdub_two_stage;
    // Guide-edit uses this instead of `strength` when positive. A value of one
    // is a full restyle; lower values preserve more of the source scene.
    float v2v_guide_strength;
    // Trusted absolute path to a saved video-only LTX latent. Consumed only by
    // v2v_mode 2; callers of the HTTP API must use the job-bank root.
    const char* v2v_guide_latent_path;
    int64_t seed;
    int video_frames;
    int fps;
    float vace_strength;
    // Wan-VACE continuation input.  When provided, the leading temporal latent
    // frames of the VACE inactive context are replaced from this prior sampled
    // video latent, avoiding a lossy pixel decode/re-encode at a chain seam.
    // This is consumed only by Wan VACE models.
    const float* vace_cont_latent;
    int vace_cont_latent_width;
    int vace_cont_latent_height;
    int vace_cont_latent_frames;
    int vace_cont_latent_channels;
    int vace_cont_frames;
    int vace_cont_latent_drop_tail;
    // LTX multi-segment continuation input. The caller supplies the prior
    // segment's video-only diffusion-latent tail in ggml-ne order
    // [width, height, frames, channels, 1]. The LTX path pins that tail at the
    // head of the next segment, avoiding a lossy VAE decode/re-encode seam.
    const float* cont_latent;
    int cont_latent_width;
    int cont_latent_height;
    int cont_latent_frames;
    int cont_latent_channels;
    // Optional frozen video-latent guide at the end of an LTX window.  Chain
    // retake uses the opening latent frames of the unchanged next shot here so
    // the re-rendered shot is constrained at both seams.
    const float* end_cont_latent;
    int end_cont_latent_width;
    int end_cont_latent_height;
    int end_cont_latent_frames;
    int end_cont_latent_channels;
    // LongCat-Video-Avatar 1.5 driving audio.  This is a WAV file consumed by
    // the Whisper audio encoder; the same source track is returned as optional
    // generated audio, trimmed to the rendered clip.
    const char* audio_path;
    // LTX-AV driving audio. This 16 kHz mono WAV is encoded by an encoder-capable
    // LTX audio VAE and held fixed while video is denoised, so the generated
    // motion follows the supplied speech. This is distinct from audio_path,
    // which remains the LongCat Avatar Whisper input.
    const char* drive_audio_path;
    // AUDIO GAP-FILL (inpainting, stage 1). Non-zero generates the regions of
    // drive_audio_path that are SILENT while holding the regions that carry real
    // signal, instead of holding the whole clip. Zero (default) is the ordinary
    // behaviour: a supplied drive clip conditions the render and is held entire.
    //
    // The deliverable still wants the SOURCE re-substituted over the held regions
    // at mux time — a round trip through the audio VAE is measurably lossy, so
    // what this buys is generated content in the gaps, not a transparent copy of
    // what was supplied.
    int audio_fill_gaps;
    // Start position in the driving-audio timeline, measured at the Avatar
    // model's fixed 25 fps.  Zero is the normal single-clip case.
    int audio_frame_offset;
    // LongCat Avatar block-sparse attention mask. Dense is the default;
    // enabling BSA changes the temporal/spatial attention receptive field.
    int bsa_enabled;
    int bsa_radius;
    int bsa_self_frame;
    int bsa_bookend;
    int bsa_cube_h;
    int bsa_cube_w;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
    // LTX supports an ordered sequence of latent-upscale + SDEdit refine
    // stages. A null/empty chain preserves the legacy single `hires` pass.
    const sd_hires_params_t* hires_chain;
    int hires_chain_count;
    // Optional previews emitted after each non-final upscale stage. Frame
    // storage is borrowed for the duration of the callback only.
    int emit_stages;
    void (*on_stage)(int seg_index, int stage_scale, int width, int height,
                     const sd_image_t* frames, int frame_count, void* user);
    void* on_stage_user;
    int stage_seg_index;
    bool circular_x;
    bool circular_y;
} sd_vid_gen_params_t;

// LTX or LongCat Avatar multi-window render and durable continuation. Completed
// windows can be saved as `seg_<n>.bin` under `bank_dir`; a non-zero
// `start_segment` restores that prefix and renders only the remaining windows.
// Output ownership matches generate_video.
typedef struct {
    int n_segments;
    const char* const* segment_prompts;
    // Optional per-window render lengths. A zero entry uses video_frames from the
    // base parameters; a continuation window must still leave room for its overlap.
    const int* segment_video_frames;
    // Optional fresh-scene controls. A scene-cut window starts from its prompt without
    // a continuation tail; an image entry pins its opening frame and also starts fresh.
    const int* segment_scene_cuts;
    const sd_image_t* const* segment_init_images;
    // Optional per-window LTXAV frame-pinned image guides. Each segment owns
    // `segment_keyframe_counts[i]` images and matching pixel-frame indices.
    sd_image_t* const* segment_keyframes;
    const int* const* segment_keyframe_indices;
    const int* segment_keyframe_counts;
    // Optional per-window V2V sources. Modes 1 and 2 replace the window's
    // starting video latent; they are fresh scenes rather than continuations.
    sd_image_t* const* segment_control_frames;
    const int* segment_control_frame_counts;
    const int* segment_v2v_modes;
    // Negative entries retain base_params->strength; non-negative entries are
    // the SDEdit denoising strength for their corresponding V2V window.
    const float* segment_v2v_strengths;
    // Optional trusted saved-video-latent source for a mode-2 V2V window.
    const char* const* segment_v2v_guide_latent_paths;
    int cont_latent_frames;
    int start_segment;
    const char* bank_dir;
    // Optional PER-SHOT override for the bank a RESTORED shot is read from.
    // Null, or a null/empty entry, keeps `bank_dir` — i.e. today's behaviour.
    //
    // What this buys: a resume or a retake can restore a MIXTURE — shot 0 out of
    // one job's bank, shot 1 out of another's — which is what "render again from
    // the takes I picked" needs. `bank_dir` alone can only ever name one bank,
    // so a retake of a later shot continued from whatever the single bank last
    // held rather than from the take the user actually selected.
    //
    // Entries are RESOLVED DIRECTORIES, not job ids: id->dir policy (the bank_id
    // indirection, the persist/transient root search) lives in the server layer
    // next to resolve_ltx_bank_dir, exactly as segment_v2v_guide_latent_paths
    // carries a resolved path rather than a reference.
    //
    // WRITES ignore this and always go to `bank_dir`. Restoring is a read of a
    // shot's chosen past; banking is what this job produces.
    const char* const* segment_bank_dirs;
    // Optional full-timeline WAV inputs for LTX chains. The core slices
    // chain_audio_full per generated window and holds each slice fixed as the
    // drive signal; bank_dir is required so those durable slices survive the
    // asynchronous job. chain_audio_track is muxed over the final stitched
    // video at its actual post-overlap length.
    const char* chain_audio_full;
    const char* chain_audio_track;
    int chain_audio_offset_frames;
    // Legacy pre-sliced per-window 16 kHz driving WAVs, named aud_<n>.wav.
    // Kept for the Koblem relip/window API; whole-timeline chain_audio_full
    // takes precedence when both forms are supplied.
    const char* chain_audio_dir;
    // Optional safe-boundary hook invoked immediately before each window is
    // sampled. Returning false aborts the chain. It is intended for callers
    // that need to lease an architecture-compatible DiT variant per segment;
    // no frame or latent storage is live at this point.
    bool (*before_segment)(int segment_index, void* user);
    void* before_segment_user;
    // Optional notification after a complete segment has been sampled and
    // decoded. The frame storage remains owned by the chain and is valid only
    // for the callback; consumers that need it asynchronously must copy it.
    //
    // `audio` is that segment's own decoded audio, already trimmed to the kept
    // (post-overlap) portion of the shot, or null when the render has no audio.
    // It carries the same borrowed-for-the-callback lifetime as `frames`: the
    // chain frees it immediately afterwards, so a consumer that encodes
    // asynchronously must copy the samples. Supplying it is what lets a
    // progressive per-shot preview have SOUND — without it every preview is
    // silent while only the final stitched clip carries the track.
    void (*on_segment)(int segment_index,
                       const sd_image_t* frames,
                       int frame_count,
                       const sd_audio_t* audio,
                       void* user);
    void* on_segment_user;
    // WINDOWED STREAMING FINALIZE.  Set this and the chain stops accumulating the whole decoded
    // timeline in host RAM: everything older than the small suffix a future seam op can still
    // reach into is handed over here, IN ORDER, as it becomes final.  Peak frame memory drops
    // from the entire clip to roughly one segment plus that suffix, which is the difference
    // between ~14 GB and ~200 MB on a 3.5-minute 1280x704 chain (and ~32 GB at 1920x1088).
    //
    // The callee TAKES OWNERSHIP of each frame's .data and must free it.  Leaving this null keeps
    // the historical accumulate-everything behaviour, so existing callers are unaffected.
    void (*on_flush_frames)(const sd_image_t* frames, int frame_count, void* user);
    void* on_flush_frames_user;
    // Re-render one durable banked shot in place.  `enable_retake` is separate
    // so legacy zero-initialized chain structs retain their ordinary behavior;
    // set it whenever retake_segment is supplied (including segment zero).
    bool enable_retake;
    // The prefix is restored from
    // bank_dir, the selected shot is sampled with both neighbour guides when
    // available, and the unchanged suffix is decoded from its banks.  -1 is
    // disabled when enable_retake is false.
    int retake_segment;
    // Continuation seam policy.  A positive scalar pins every ordinary
    // continuation.  Zero uses the derived 8*K drop, and PINS it whenever the
    // engine owns the audio -- the drive window is cut a priori against that
    // number, so only equality keeps A/V frame-exact.  A NEGATIVE scalar forces
    // the content-adaptive seam search back on even with audio present, which
    // desyncs by the per-segment prediction error; diagnostics only.  Per-shot
    // entries supersede it; a negative ENTRY selects the scalar/derived value.
    int cont_seam_drop_frames;
    const int* segment_seam_drop_frames;
    // Optional per-shot audio files.  full drives that window's LTX audio
    // conditioning; track is appended from its own t=0 for exactly the kept
    // portion of that shot (padding short tracks with silence).
    const char* const* segment_audio_full;
    const char* const* segment_audio_track;
    // Optional per-shot Prompt Relay beats.  Frame indices are on that shot's
    // own rendered timeline.  A null array, or a zero count for a shot, leaves
    // that shot on the byte-identical null-mask path.
    const sd_ltx_beat_t* const* segment_beats;
    const int* segment_beat_counts;
    // Optional per-shot sampling overrides.  These exist so one shot can be
    // retaken, or pushed onto a cfg-capable variant for relay, without
    // disturbing the rest of the chain.  A null array inherits everywhere.
    // Seeds: negative entries inherit `base_params->seed + segment`.
    const int64_t* segment_seeds;
    // Steps: entries <= 0 inherit the base sample step count.
    const int* segment_steps;
    // Text CFG: entries < 0 inherit the base guidance.
    const float* segment_cfg;
    // Negative prompts: null or empty entries inherit the base negative prompt.
    const char* const* segment_negative_prompts;
    // Optional per-shot reference_head_trim, same encoding as the base field
    // (0 = off, -1 = auto, >0 = an explicit pixel-frame count). A NULL array
    // applies base_params->reference_head_trim to every shot; a SUPPLIED array is
    // authoritative for every entry, INCLUDING a zero -- the same rule the beat
    // arrays follow, so a caller can switch the trim off for one shot of a project
    // that has it on.
    const int* segment_reference_head_trim;
} sd_vid_chain_params_t;

typedef struct sd_ctx_t sd_ctx_t;
struct ggml_tensor;

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);
typedef bool (*sd_graph_eval_callback_t)(struct ggml_tensor* t, bool ask, void* user_data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);
SD_API void sd_set_backend_eval_callback(sd_graph_eval_callback_t cb, void* data);
SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx);
// Replace the registered diffusion-model weights while preserving the
// architecture-compatible runner and the other model components. The caller
// must serialize this with generation; the outgoing DiT residency is released
// before the replacement is registered.
SD_API bool sd_ctx_swap_diffusion_model(sd_ctx_t* sd_ctx, const char* diffusion_model_path);

// Re-apply the runtime LoRA set on an already-loaded context, REPLACING whatever is active
// (pass lora_count = 0 to clear). Same serialization requirement as the DiT swap above: the
// caller must not have generation in flight.
//
// Exists so a chain can vary adapters PER SEGMENT the way it already varies the DiT variant.
// Safe to call repeatedly: apply_loras_at_runtime() clears set_loras/runtime_lora_models and
// the adapters before rebuilding, so this is a replace and not an accumulate.
//
// NOTE the asymmetry with the DiT swap: a variant swap re-registers weights, but an adapter
// change only rebuilds the MultiLoraAdapter, so it is far cheaper (~1s measured vs ~4s).
SD_API bool sd_ctx_apply_loras(sd_ctx_t* sd_ctx, const sd_lora_t* loras, uint32_t lora_count);

// ControlNet hot-swap APIs are not safe to call while generation is in flight.
SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path);
SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx);

// Does this reference-image args string ask for references to be decoded at their NATIVE
// geometry, rather than centre-cropped to the request's aspect ratio and rescaled to the
// request's resolution?
//
// This exists because that crop happens in the CALLER, at HTTP decode time (load_image_common,
// driven by the request's width/height), long before an sd_ctx_t sees a RefImageParams. The
// front end therefore has to be able to ask the question before it has decoded anything, and it
// must get the same answer the library would — so both go through the same preset table here.
//
// Returns true if `preset=` names a preset with native geometry (krea2_identity_restage), or if
// an explicit `native_ref=1` overrides it either way. Safe on NULL/empty.
SD_API bool sd_ref_image_args_want_native_geometry(const char* ref_image_args);

SD_API const char* sd_type_name(enum sd_type_t type);
SD_API enum sd_type_t str_to_sd_type(const char* str);
SD_API const char* sd_rng_type_name(enum rng_type_t rng_type);
SD_API enum rng_type_t str_to_rng_type(const char* str);
SD_API const char* sd_sample_method_name(enum sample_method_t sample_method);
SD_API enum sample_method_t str_to_sample_method(const char* str);
SD_API const char* sd_scheduler_name(enum scheduler_t scheduler);
SD_API enum scheduler_t str_to_scheduler(const char* str);
SD_API const char* sd_prediction_name(enum prediction_t prediction);
SD_API enum prediction_t str_to_prediction(const char* str);
SD_API const char* sd_preview_name(enum preview_t preview);
SD_API enum preview_t str_to_preview(const char* str);
SD_API const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode);
SD_API enum lora_apply_mode_t str_to_lora_apply_mode(const char* str);
SD_API const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler);
SD_API enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str);

SD_API void sd_cache_params_init(sd_cache_params_t* cache_params);
SD_API void sd_hires_params_init(sd_hires_params_t* hires_params);

SD_API void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);
SD_API char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params);

SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);
// Release the DiT's staged compute and parameter residency while retaining the
// context, model metadata, VAE, and text encoders. The next generation call reloads
// the registered DiT parameters through ModelManager. Call only when idle.
SD_API void sd_ctx_free_diffusion_model(sd_ctx_t* sd_ctx);
SD_API void free_sd_audio(sd_audio_t* audio);

SD_API void sd_sample_params_init(sd_sample_params_t* sample_params);
SD_API char* sd_sample_params_to_str(const sd_sample_params_t* sample_params);

SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method);

SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
SD_API char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params);
SD_API bool generate_image(sd_ctx_t* sd_ctx,
                           const sd_img_gen_params_t* sd_img_gen_params,
                           sd_image_t** images_out,
                           int* num_images_out);

enum sd_cancel_mode_t {
    // Stop the current generation as soon as possible.
    SD_CANCEL_ALL,
    // Finish the current image sample, then skip additional batch latents and return completed images.
    SD_CANCEL_NEW_LATENTS,
    // Clear a pending cancellation request.
    SD_CANCEL_RESET
};

SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode);

SD_API void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params);
SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out);

// Like generate_video, and additionally returns a malloc-owned copy of the
// sampled video latent before VAE decoding.  The shape is ggml-ne ordered
// [width, height, frames, channels, 1].  It is primarily the lossless hand-off
// used by Wan-VACE continuation; any output pointer may be NULL.
SD_API bool generate_video_ex(sd_ctx_t* sd_ctx,
                              const sd_vid_gen_params_t* sd_vid_gen_params,
                              sd_image_t** frames_out,
                              int* num_frames_out,
                              sd_audio_t** audio_out,
                              float** final_latent_out,
                              int* latent_width_out,
                              int* latent_height_out,
                              int* latent_frames_out,
                              int* latent_channels_out,
                              // Reports the reference head-frame trim this render RESOLVED (after
                              // AUTO derivation, self-gating and clamping) WITHOUT applying it, so
                              // the caller can decide where the cut lands. generate_video() applies
                              // it to the frames and audio it returns; generate_video_chain() folds
                              // it into that shot's output-side head drop, which keeps the durable
                              // seg_<n>.bin / .len / .audio bank internally consistent (the bank
                              // holds the shot AS RENDERED and .len records what the timeline kept).
                              // Null when the caller does not care.
                              int* reference_head_trim_out);

SD_API bool generate_video_chain(sd_ctx_t*                    sd_ctx,
                                 const sd_vid_gen_params_t*   base_params,
                                 const sd_vid_chain_params_t* chain_params,
                                 sd_image_t**                 frames_out,
                                 int*                         num_frames_out,
                                 sd_audio_t**                 audio_out);

// Wan2.2-VACE multi-window continuation.  Each window carries both its kept
// decoded pixel tail and the prior sampled latent directly in memory.  The
// overlap/discard values are pixel-frame counts; zero selects the production
// defaults (5, 4, and 1 respectively).  Output ownership matches
// generate_video.
typedef struct {
    int n_segments;
    const char* const* segment_prompts;
    int overlap_frames;
    int discard_tail_frames;
    int drop_latent_tail_frames;
    // Optional state for continuing a previously completed prefix.  The caller
    // owns all buffers for the duration of this call.  `start_segment` must be
    // the number of already-completed windows; the supplied pixel tail and
    // sampled latent seed that next window.
    int start_segment;
    const sd_image_t* resume_control_frames;
    int resume_control_frames_size;
    const float* resume_latent;
    int resume_latent_width;
    int resume_latent_height;
    int resume_latent_frames;
    int resume_latent_channels;
    // Optional durable bank.  Each completed window is saved as seg_<n>.bin
    // under this directory.  When start_segment is non-zero and no in-memory
    // state is supplied, the chain VAE-decodes this bank to rebuild its prefix.
    const char* bank_dir;
    // Called after each newly completed window.  `frames` are the kept frames
    // for that window; `latent` is non-NULL whenever another window remains.
    void (*on_segment)(int segment_index,
                       const sd_image_t* frames,
                       int frame_count,
                       const float* latent,
                       int latent_width,
                       int latent_height,
                       int latent_frames,
                       int latent_channels,
                       void* user);
    void* on_segment_user;
} sd_wan_vace_chain_params_t;

SD_API bool generate_wan_vace_chain(sd_ctx_t*                          sd_ctx,
                                    const sd_vid_gen_params_t*         base_params,
                                    const sd_wan_vace_chain_params_t*  chain_params,
                                    sd_image_t**                       frames_out,
                                    int*                               num_frames_out,
                                    sd_audio_t**                       audio_out);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        bool direct,
                                        int n_threads,
                                        int tile_size,
                                        const char* backend,
                                        const char* params_backend);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API bool upscale(upscaler_ctx_t* upscaler_ctx,
                    sd_image_t input_image,
                    uint32_t upscale_factor,
                    sd_image_t** images_out,
                    int* num_images_out);

SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);

typedef struct adetailer_ctx_t adetailer_ctx_t;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    const char* extra_ad_args;
} sd_adetailer_params_t;

SD_API adetailer_ctx_t* new_adetailer_ctx(const char* detector_path,
                                          int n_threads,
                                          const char* backend,
                                          const char* params_backend);
SD_API void free_adetailer_ctx(adetailer_ctx_t* adetailer_ctx);
SD_API bool adetail_image(adetailer_ctx_t* adetailer_ctx,
                          sd_ctx_t* sd_ctx,
                          sd_image_t input_image,
                          const sd_adetailer_params_t* adetailer_params,
                          const sd_img_gen_params_t* inpaint_params,
                          sd_image_t** images_out,
                          int* num_images_out);

SD_API bool convert(const char* input_path,
                    const char* vae_path,
                    const char* output_path,
                    enum sd_type_t output_type,
                    const char* tensor_type_rules,
                    bool convert_name);

SD_API bool convert_with_components(const char* model_path,
                                    const char* clip_l_path,
                                    const char* clip_g_path,
                                    const char* t5xxl_path,
                                    const char* diffusion_model_path,
                                    const char* vae_path,
                                    const char* output_path,
                                    enum sd_type_t output_type,
                                    const char* tensor_type_rules,
                                    bool convert_name,
                                    int n_threads);

SD_API bool preprocess_canny(sd_image_t image,
                             float high_threshold,
                             float low_threshold,
                             float weak,
                             float strong,
                             bool inverse);

SD_API bool load_imatrix(const char* imatrix_path);
SD_API void save_imatrix(const char* imatrix_path);
SD_API void enable_imatrix_collection(void);
SD_API void disable_imatrix_collection(void);

SD_API const char* sd_commit(void);
SD_API const char* sd_version(void);

// List available ggml backend devices, one `name<TAB>description` per line.
// The names are the device names accepted by the --backend / --params-backend
// assignment specs. Returns the number of bytes required, excluding the null
// terminator. Passing nullptr or buffer_size 0 only queries the required size.
SD_API size_t sd_list_devices(char* buffer, size_t buffer_size);

// for C API, caller needs to call free_sd_images to free the memory after use
// This helps avoid CRT problems on Windows when memory is allocated in the library but freed in the caller, which may use a different CRT.
SD_API void free_sd_images(sd_image_t* result_images, int num_images);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
