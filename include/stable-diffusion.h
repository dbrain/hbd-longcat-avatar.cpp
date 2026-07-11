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
    SCHEDULER_COUNT
};

enum prediction_t {
    EPS_PRED,
    V_PRED,
    EDM_V_PRED,
    FLOW_PRED,
    FLUX_FLOW_PRED,
    FLUX2_FLOW_PRED,
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
    const sd_embedding_t* embeddings;
    uint32_t embedding_count;
    const char* photo_maker_path;
    const char* tensor_type_rules;
    bool vae_decode_only;
    bool free_params_immediately;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type;
    enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode;
    bool offload_params_to_cpu;
    bool enable_mmap;
    bool keep_clip_on_cpu;
    bool keep_control_net_on_cpu;
    bool keep_vae_on_cpu;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool tae_preview_only;
    bool diffusion_conv_direct;
    bool vae_conv_direct;
    bool circular_x;
    bool circular_y;
    bool force_sdxl_vae_conv_scale;
    bool chroma_use_dit_mask;
    bool chroma_use_t5_mask;
    int chroma_t5_mask_pad;
    bool qwen_image_zero_cond_t;
    enum sd_vae_format_t vae_format;
    float max_vram;  // GiB budget for graph-cut segmented param offload (0 = disabled, -1 = auto free VRAM minus 1 GiB)
    bool stream_layers;  // Enable residency+prefetch streaming on top of --max-vram (no effect without --max-vram)
    const char* backend;
    const char* params_backend;
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
    // FEATURE 1 (--hires-lora): optional per-phase LoRA set applied only on the hires/refine pass
    // (video path). NULL/0 = reuse the base pass's LoRA state (legacy behavior). Consumed by
    // generate_video, which calls apply_loras(loras, lora_count) just before the refine sample();
    // because apply_loras diffs against curr_lora_state, passing the FULL refine set transitions
    // cleanly from the base set. Backing storage is owned by the caller (SDGenerationParams).
    const sd_lora_t* loras;
    uint32_t lora_count;
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
    bool auto_resize_ref_image;
    bool increase_ref_index;
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
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
} sd_img_gen_params_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t end_image;
    // LTXAV MULTI-KEYFRAME conditioning: arbitrary (image, video-frame-index) guide pairs
    // pinned as frozen 1-frame conditioning frames on the target timeline. Generalises the
    // single start/end i2v pins (init_image @ frame 0, end_image @ frames-1) to N caller-placed
    // keyframes: keyframes[i] is applied at keyframe_frame_indices[i], a VIDEO (pixel) frame
    // index in [0, video_frames) — the same unit end_image uses. Only consumed on the LTXAV
    // path; a size of 0 (the default) leaves the single-image i2v / t2v / continuation paths
    // byte-identical.
    sd_image_t* keyframes;
    int*        keyframe_frame_indices;
    int         keyframes_size;
    sd_image_t* control_frames;
    int control_frames_size;
    int width;
    int height;
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    float moe_boundary;
    float strength;
    int64_t seed;
    int video_frames;
    int fps;
    float vace_strength;
    const char* audio_path;  // LongCat-Avatar audio-driven lip-sync (16kHz mono wav)
    // LTX-2.3 (LTXAV) audio-DRIVEN lip-sync: when set, this 16kHz wav is encoded by the
    // audio VAE into the joint AV latent's audio slot and held FIXED (denoise-mask 0,
    // audio-timestep 0) so the model lip-syncs video to THIS audio instead of inventing
    // its own. Requires an --audio-vae built with the encoder + mel basis (the -ENC gguf
    // from tools/convert_ltx_audio_vae.py --with-encoder). Distinct from audio_path above,
    // which is the LongCat-Avatar (NAVA) Whisper path, NOT the LTXAV path.
    const char* drive_audio_path;
    // LongCat-Avatar continuation chaining: when cont_latent != NULL these are the
    // LAST cont_latent_frames latent frames of the PRIOR segment (the diffusion
    // latents themselves, NOT pixels — avoids a lossy VAE decode/re-encode roundtrip
    // and the multi-frame VAE-encode path). They become the fixed conditioning
    // (num_cond_latents = cont_latent_frames; denoise-mask 0, timestep 0) and the new
    // segment continues from them. cont_latent layout is the avatar diffusion-latent
    // ggml-ne order [W_lat, H_lat, cont_latent_frames, C_lat, 1] (contiguous f32),
    // W_lat=width/vae_scale, H_lat=height/vae_scale, C_lat = the model's latent channels.
    // audio_frame_offset is this segment's start position (in 25fps video frames) in
    // the global audio timeline so lip-sync continues across segments.
    const float* cont_latent;
    int cont_latent_frames;
    // LTXAV two-stage continuation: the last refined HIGH-RES VIDEO-only latent frames from
    // the prior segment. These are not interchangeable with cont_latent: cont_latent carries
    // the base-grid motion tail into stage 1, while cont_refine_latent is supplied as a frozen
    // reference token block only to the next segment's stage-2 refine. Layout is ggml-ne
    // [W_lat, H_lat, frames, video_channels, 1], contiguous f32. NULL/0 is byte-identical to
    // the pre-existing two-stage path. Audio is intentionally excluded; every segment keeps its
    // own drive-audio latent/timeline.
    const float* cont_refine_latent;
    int cont_refine_latent_frames;
    int cont_refine_latent_width;
    int cont_refine_latent_height;
    int cont_refine_latent_channels;
    // LTXAV LATENT continuation (file-based, CLI-friendly): path to a saved VIDEO latent
    // (LTXAV_SAVE_VIDEO_LATENT output from the prior segment). When set on the LTXAV path,
    // the last cont_latent_frames latent frames are placed at the head of the new segment's
    // init_latent and held fixed (motion-carrying overlap; mask via LTXAV_CONT_OVERLAP_MASK).
    // Distinct from cont_latent (the avatar's in-process float* tail).
    const char* cont_latent_path;
    // LTXAV appearance ANCHOR (anti-drift): path to a saved video latent whose FRAME 0 is the
    // original character. Pinned at the head of every continuation segment so the rendering
    // style can't migrate off the source over a long chain (StreamingT2V-style appearance memory).
    const char* cont_anchor_path;
    int audio_frame_offset;
    // LongCat-Avatar continuation REFERENCE ANCHOR (generate_avc). When cont_ref_latent
    // != NULL it is the ORIGINAL portrait's diffusion latent (1 frame, same ggml-ne
    // layout [W_lat, H_lat, 1, C_lat, 1]) — the un-drifted clean anchor the reference
    // keeps on EVERY continuation segment. It is PREPENDED ahead of the cont_latent cond
    // tail (layout [ref(1), cond_tail(N), noise...]), held fixed (denoise-mask 0, ts 0),
    // and drives the 3-way self-attn split + ref-positioned 3D-RoPE in the DiT. Without
    // it, continuation frames drift off the already-drifted prior tail (watercolour melt).
    // Only consumed when cont_latent is also set (segments>1 continuation path).
    const float* cont_ref_latent;
    int cont_ref_img_index;     // ref anchor's temporal grid position (default 10)
    int cont_mask_frame_range;  // noise-near-ref attention carve-out half-width (default 3)
    // LongCat-Avatar Block-Sparse Attention (BSA). Per-request quality/speed knob;
    // not bit-exact (mild quality trade for ~-2s wall on the BSA path). Set
    // bsa_enabled=1 to engage; otherwise dense attention runs (the default).
    // Defaults to the "r=1+self_frame" config the project has tested with —
    // mild camera-like rotation drift, owner-approved at lap-29.2 quality.
    int   bsa_enabled;     // 0=dense (default), 1=BSA
    int   bsa_radius;      // cube-near window half-width (default 1)
    int   bsa_self_frame;  // 0/1, intra-frame anchor (default 1 — tested config)
    int   bsa_bookend;     // 0/1, last-frame anchor (default 0)
    int   bsa_cube_h;      // cube height (default 4)
    int   bsa_cube_w;      // cube width (default 6)
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
} sd_vid_gen_params_t;

// LTXAV in-process multi-segment chain. Renders n_segments video segments with the DiT
// kept RESIDENT across all of them (no per-segment reload), each segment continuing from
// the prior segment's video-latent tail (motion carry, in-memory float* — no disk/VAE
// roundtrip), and stitches them into one continuous clip (dropping the re-rendered
// overlap head of each segment>0). A relip caller may instead provide a distinct source-video
// frame range for every segment: those independent V2V windows are stitched without a generated
// latent overlap, while still sharing the resident DiT. Per-segment prompts (the "director" layer) are
// pre-encoded up front in ONE text-encoder window (sd_ctx_precompute_chain_text_conds) so
// no gemma encode is interleaved between segments. Per-segment lip-sync audio, when
// present, is read from chain_audio_dir/aud_<i>.wav (16kHz mono, absolute timeline).
typedef struct {
    int          n_segments;          // number of segments to render + stitch (>=1)
    int          cont_latent_frames;  // K overlap latent frames carried between segments
    const char** segment_prompts;     // n_segments entries; a NULL entry reuses base prompt
    const char*  chain_audio_dir;     // dir with aud_<i>.wav per segment, or NULL (no lip-sync)
    const char*  save_dir;            // optional: bank each seg's video latent + webm to save_dir/seg_<i>.{bin,webm}
    int          resume_from;         // resume: skip+reload segments [0, resume_from) from save_dir banked latents (0 = fresh)
    // Optional per-segment V2V source ranges for chained relip. Both arrays have n_segments
    // entries; segment_control_frames[i] points at the first frame and the matching count gives
    // its length. When supplied for a segment, it replaces base_params->control_frames and that
    // segment deliberately does not receive the generated continuation latent (the source video
    // itself is the temporal control). NULL preserves ordinary LTX continuation byte-for-byte.
    sd_image_t* const* segment_control_frames;
    const int*         segment_control_frame_counts;
    // Optional per-segment i2v scene image ("Director" multi-scene). n_segments entries; a
    // non-NULL entry with .data makes that segment START A FRESH SCENE from the image
    // (like seg-0 i2v) instead of continuing the prior segment's motion. The audio timeline
    // is untouched — the per-segment aud_<i>.wav still lands on its slot and the scene-cut
    // segment's stitch drop is 0, so the soundtrack flows unbroken across the cut. Seg-0's
    // entry is redundant with base_params->init_image (both = the opener). NULL = ordinary
    // continuation, byte-identical to before.
    sd_image_t* const* segment_init_images;
    // Optional: invoked once per stitched segment, in order, with that segment's kept frames
    // (overlap head already dropped; the frames stay owned by the chain). Lets the caller
    // (server layer) bank a viewable per-segment webm as it's produced, without the core lib
    // depending on the example-layer media encoder. Fired only for freshly-rendered segments
    // (skipped ones on resume already have their webm on disk). NULL = no callback.
    void (*on_segment)(int seg_index, const sd_image_t* frames, int frame_count, void* user);
    void*        on_segment_user;
} sd_vid_chain_params_t;

typedef struct sd_ctx_t sd_ctx_t;

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);

SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);

// Cooperative render cancellation. sd_request_cancel() sets a process-global
// atomic that the sampler loop (between steps) and the avatar segment loop
// (between segments) poll; on observe they bail cleanly, freeing compute
// buffers and returning failure up the call chain WITHOUT tearing down the
// context (weights stay resident — this is NOT unload). Callers MUST
// sd_clear_cancel() at the start of each render so a stale cancel can't abort
// the next request.
SD_API void sd_request_cancel(void);
SD_API void sd_clear_cancel(void);
SD_API bool sd_is_cancel_requested(void);
SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx);

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
SD_API void free_sd_audio(sd_audio_t* audio);

SD_API void sd_sample_params_init(sd_sample_params_t* sample_params);
SD_API char* sd_sample_params_to_str(const sd_sample_params_t* sample_params);

SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method);

SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
SD_API char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params);
SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);

SD_API void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params);
SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out);

// Like generate_video, but for LongCat-Avatar continuation chaining it also returns
// the final diffusion latent (caller-freed via free()) so the caller can feed the
// tail back as the next segment's cont_latent. final_latent_out / shape outputs may
// be NULL if not needed. Shape is the avatar latent ggml-ne order
// [latent_width, latent_height, latent_frames, latent_channels, 1].
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
                              float** refined_latent_out,
                              int* refined_latent_width_out,
                              int* refined_latent_height_out,
                              int* refined_latent_frames_out,
                              int* refined_latent_channels_out);

// Render + stitch an LTXAV multi-segment chain (see sd_vid_chain_params_t). base_params is
// the per-segment template; the chain overrides prompt / cont_latent / drive_audio / seed /
// audio_frame_offset per segment. Internally flips keep_diffusion_model_resident on and
// pre-encodes all segment prompts in one TE window. On success *frames_out is a malloc'd
// array of *num_frames_out stitched frames (caller frees each frame's .data, then the
// array) and *audio_out is the optional segment-0 / generated audio (free via
// free_sd_audio). Returns false on any segment failure (partial outputs are freed).
SD_API bool generate_video_chain(sd_ctx_t*                    sd_ctx,
                                 const sd_vid_gen_params_t*   base_params,
                                 const sd_vid_chain_params_t* chain_params,
                                 sd_image_t**                 frames_out,
                                 int*                         num_frames_out,
                                 sd_audio_t**                 audio_out);

// Render + stitch a Wan2.2-VACE multi-window continuation chain (the warm-server wan path).
// Window 0 is the base (t2v clean = VACE_STRENGTH 0, or i2v/ref-anchored via base_params->
// init_image); windows 1..N-1 are VACE continuations fed the prior window's kept pixel tail
// (the --control-video equivalent) + its full diffusion latent (the VACE_CONT_LATENT
// equivalent) entirely IN MEMORY — no cont_bank/*.bin or cont_tail/ disk round-trip. The
// per-window VACE env is injected here: VACE_SKIP_BLOCKS=0 always, VACE_STRENGTH_TAIL /
// VACE_STRENGTH_ANCHOR_FRAMES on continuation windows only. Overlap/discard knobs are read
// from env (WAN_CHAIN_K=5, WAN_CHAIN_DISCARD=4, WAN_CHAIN_DROPLAT=1 by default). Output
// ownership matches generate_video_chain. Returns false on any window failure.
SD_API bool generate_wan_vace_chain(sd_ctx_t*                    sd_ctx,
                                    const sd_vid_gen_params_t*   base_params,
                                    const sd_vid_chain_params_t* chain_params,
                                    sd_image_t**                 frames_out,
                                    int*                         num_frames_out,
                                    sd_audio_t**                 audio_out);

// Keep the diffusion (and whisper) model params resident across back-to-back
// generate_video[_ex] calls — required for LongCat-Avatar continuation chaining so
// later segments don't render against freed GPU memory.
SD_API void sd_ctx_keep_diffusion_model_resident(sd_ctx_t* sd_ctx, bool keep);

// Pre-encode the text conditioning for a set of chained-segment prompts in ONE text-
// encoder residency window (avatar/LTXAV resident chains only). Populates an internal
// prompt-keyed cache so the per-segment renders all hit it, letting the resident DiT run
// as one uninterrupted phase with no gemma encode interleaved between segments. The
// negative prompt is constant across the chain; when needed, its uncond is encoded once
// and shared.
// Call once, after sd_ctx_keep_diffusion_model_resident(true), before the segment loop.
SD_API void sd_ctx_precompute_chain_text_conds(sd_ctx_t*    sd_ctx,
                                               const char** prompts,
                                               int          n_prompts,
                                               const char*  negative_prompt,
                                               int          clip_skip,
                                               bool         need_uncond);

// Hot-swap the diffusion (DiT) model weights in place from a different gguf,
// reusing the existing backend + the resident VAE/text-encoder. Intended for the
// FLUX.2-Klein base<->edit swap: both variants share the same DiT architecture +
// tensor names, only the ~5.6 GB of weights differ, so the runner object/param
// graph is reused and only its param buffer is freed + refilled. Returns false on
// load failure (and leaves the DiT params buffer freed — caller should treat the
// ctx as unusable for img_gen until a successful swap/reload). The caller MUST
// guarantee no render is in flight (call from the serial async worker thread).
SD_API bool sd_ctx_swap_diffusion_model(sd_ctx_t* sd_ctx, const char* diffusion_model_path);

// Free the diffusion (DiT) model's VRAM (compute + param buffers) without touching
// the resident VAE / text encoder. Used by the server's /v1/admin/unload so an
// external GPU gate can reclaim the card; the next render must reload the DiT via
// sd_ctx_swap_diffusion_model() first. Caller MUST ensure no render is in flight.
SD_API void sd_ctx_free_diffusion_model(sd_ctx_t* sd_ctx);

// LongCat-Avatar continuation chaining (drift sink): VAE-encode a stack of decoded
// RGB frames back to a DIFFUSION latent, matching the space of generate_video_ex's
// final_latent_out / the cont_latent consumed by the next segment. This is the
// reference pipeline's decode->re-encode round-trip: re-regularizing the prior
// segment's tail to the Wan-VAE manifold every chain step kills the per-seam
// color/identity drift that a raw latent passthrough compounds. The frames are the
// last N decoded video frames of the prior segment (same W/H as the render).
// Returns a malloc'd latent (caller frees via free()) in ggml-ne order
// [latent_width, latent_height, latent_frames, latent_channels, 1], or NULL on error.
SD_API float* sd_ctx_encode_video_frames(sd_ctx_t* sd_ctx,
                                         const sd_image_t* frames,
                                         int num_frames,
                                         int width,
                                         int height,
                                         int* latent_width_out,
                                         int* latent_height_out,
                                         int* latent_frames_out,
                                         int* latent_channels_out);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        bool offload_params_to_cpu,
                                        bool direct,
                                        int n_threads,
                                        int tile_size,
                                        const char* backend,
                                        const char* params_backend);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx,
                          sd_image_t input_image,
                          uint32_t upscale_factor);

SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);

SD_API bool convert(const char* input_path,
                    const char* vae_path,
                    const char* output_path,
                    enum sd_type_t output_type,
                    const char* tensor_type_rules,
                    bool convert_name);

SD_API bool preprocess_canny(sd_image_t image,
                             float high_threshold,
                             float low_threshold,
                             float weak,
                             float strong,
                             bool inverse);

SD_API const char* sd_commit(void);
SD_API const char* sd_version(void);

// ---- nvfp4-twolevel diffusion imatrix collection ----
// Enable per-DiT-Linear activation importance collection (AWQ-style per-input
// column second moment). `name_filter` restricts collection to weights whose
// name contains the substring (e.g. "diffusion_model"); pass NULL/"" for all.
SD_API void sd_imatrix_collect_begin(const char* name_filter);
// Write the accumulated imatrix as a gguf (one f32 1-D tensor per weight, keyed
// by the prefix-stripped weight name). Returns true on success and reports the
// number of tensors written via *n_written (may be NULL).
SD_API bool sd_imatrix_collect_write_gguf(const char* out_path, int* n_written);

// ---- nvfp4-twolevel convert imatrix feed ----
// Like convert(), but additionally loads an imatrix gguf and feeds AWQ-style
// per-column importance into the nvfp4 (and other) quantizers. Pass NULL/"" for
// imatrix_path to behave exactly like convert().
SD_API bool convert_with_imatrix(const char* input_path,
                                 const char* vae_path,
                                 const char* output_path,
                                 enum sd_type_t output_type,
                                 const char* tensor_type_rules,
                                 bool convert_name,
                                 const char* imatrix_path);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
