#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

#include "gguf.h"

#include "core/ggml_extend.hpp"
#include "core/ggml_graph_cut.h"

#include "core/rng.hpp"
#include "core/rng_mt19937.hpp"
#include "core/rng_philox.hpp"
#include "core/util.h"
#include "model_loader.h"
#include "stable-diffusion.h"

#include "conditioning/conditioner.hpp"
#include "extensions/generation_extension.h"
#include "model/adapter/lora.hpp"
#include "model/diffusion/anima.hpp"
#include "model/diffusion/control.hpp"
#include "model/diffusion/ernie_image.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/hidream_o1.hpp"
#include "model/diffusion/ideogram4.hpp"
#include "model/diffusion/lens.hpp"
#include "model/diffusion/ltxv.hpp"
#include "model/diffusion/mmdit.hpp"
#include "model/diffusion/model.hpp"
#include "model/diffusion/pid.hpp"
#include "model/diffusion/qwen_image.hpp"
#include "model/diffusion/unet.hpp"
#include "model/diffusion/wan.hpp"
#include "model/diffusion/z_image.hpp"
#include "model/upscaler/esrgan.hpp"
#include "model/upscaler/ltx_latent_upscaler.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "model/vae/ltx_audio_vae.hpp"
#include "model/vae/ltx_vae.hpp"
#include "model/vae/tae.hpp"
#include "model/vae/vae.hpp"
#include "model/vae/wan_vae.hpp"
#include "runtime/denoiser.hpp"
#include "runtime/guidance.h"
#include "runtime/sample-cache.h"
// fork-only models (flat src/ root; src is an include root)
#include "longcat_audio.hpp"
#include "longcat_avatar.hpp"  // LongCatAvatarModel (relocated from model/diffusion/model.hpp, sd.cpp #1569)
#include "upscaler.h"

#include "name_conversion.h"
#include "runtime/latent-preview.h"

const char* sd_vae_format_name(enum sd_vae_format_t format);
static SDVersion sd_vae_format_to_version(enum sd_vae_format_t format, SDVersion fallback);

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 1.x Inpaint",
    "Instruct-Pix2Pix",
    "SD 1.x Tiny UNet",
    "SD 2.x",
    "SD 2.x Inpaint",
    "SD 2.x Instruct-Pix2Pix",
    "SD 2.x Tiny UNet",
    "SDXS (512-DS)",
    "SDXS (09)",
    "SDXL",
    "SDXL Inpaint",
    "SDXL Instruct-Pix2Pix",
    "SDXL (Vega)",
    "SDXL (SSD1B)",
    "SVD",
    "SD3.x",
    "Flux",
    "Flux Fill",
    "Flux Control",
    "Flex.2",
    "Chroma Radiance",
    "Wan 2.x",
    "Wan 2.2 I2V",
    "Wan 2.2 TI2V",
    "Qwen Image",
    "Anima",
    "Flux.2",
    "Flux.2 klein",
    "LTXAV",
    "HiDream O1",
    "Z-Image",
    "Ovis Image",
    "Ernie Image",
    "Lens",
    "Longcat-Image",
    "Longcat-Video-Avatar",
    "PiD",
    "Ideogram 4",
    "Wan 2.2 S2V (LiveAvatar)",
};

// model_version_to_str is indexed by SDVersion (see the LOG_INFO at "Version: %s"), so a new
// enum value inserted anywhere but the end silently shifts every name after it. Catch that at
// compile time instead of shipping mislabelled models.
static_assert(sizeof(model_version_to_str) / sizeof(model_version_to_str[0]) == VERSION_COUNT,
              "model_version_to_str is out of sync with enum SDVersion");

const char* sampling_methods_str[] = {
    "Euler",
    "Euler A",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD",
    "Res Multistep",
    "Res 2s",
    "ER-SDE",
    "Euler CFG++",
    "Euler A CFG++",
    "Euler GE",
};

/*================================================== Helper Functions ================================================*/

static bool sd_version_supports_ref_latent_img_cfg(SDVersion version) {
    return version == VERSION_FLUX ||
           sd_version_is_flux2(version) ||
           sd_version_is_qwen_image(version) ||
           sd_version_is_longcat(version) ||
           sd_version_is_z_image(version);
}

static bool sd_version_supports_img_cfg(SDVersion version, bool has_ref_images) {
    return sd_version_is_inpaint_or_unet_edit(version) ||
           (has_ref_images && sd_version_supports_ref_latent_img_cfg(version));
}

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120f,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
    float reuse_threshold = params.reuse_threshold;
    if (reuse_threshold == INFINITY) {
        if (params.mode == SD_CACHE_EASYCACHE) {
            reuse_threshold = 0.2f;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0f;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

// P2 (nvfp4 patchy fix): read the per-tensor weight globals (ModelOpt weight_scale_2)
// from an UNFOLDED-import DiT gguf. They are stored as tiny sibling F32 tensors named
// "<weight>.wglobal". Uses no_alloc=true (metadata only — must NOT alloc the 16 GB of
// real tensors) and reads only the 4-byte wglobal payloads by file offset. out is keyed
// by the BARE gguf name "<weight>.wglobal". A legacy folded gguf has none -> empty map.
static void load_nvfp4_weight_globals(const std::string& path, std::map<std::string, float>& out) {
    struct gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* gctx = gguf_init_from_file(path.c_str(), gp);
    if (gctx == nullptr) {
        return;
    }
    const size_t data_off = gguf_get_data_offset(gctx);
    FILE* f = fopen(path.c_str(), "rb");
    if (f != nullptr) {
        const int64_t n = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < n; ++i) {
            const char* name = gguf_get_tensor_name(gctx, i);
            const size_t L   = name ? strlen(name) : 0;
            if (L < 8 || strcmp(name + L - 8, ".wglobal") != 0) {
                continue;
            }
            const size_t off = data_off + gguf_get_tensor_offset(gctx, i);
            float g = 1.0f;
            if (fseek(f, (long)off, SEEK_SET) == 0 && fread(&g, sizeof(float), 1, f) == 1) {
                out[name] = g;
            }
        }
        fclose(f);
    }
    gguf_free(gctx);
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    std::vector<MmapTensorStore> mmap_tensor_store;
    SDBackendManager backend_manager;

    SDVersion version;
    bool vae_decode_only         = false;
    bool external_vae_is_invalid = false;
    bool free_params_immediately = false;
    // LongCat-Avatar continuation chaining: keep the DiT params resident across
    // back-to-back generate_video calls (segments). The TE is still freed once by the
    // deferred-load flow, but the DiT must NOT be freed after each segment or the next
    // segment renders against freed GPU memory (illegal access). Set by the caller via
    // sd_ctx_keep_diffusion_model_resident() before a chained run.
    bool keep_diffusion_model_resident = false;
    // Prompt-keyed text-conditioning cache for chained video segments. The avatar's
    // constant-prompt path keeps a single entry (computed lazily on segment 0, reused
    // thereafter). The LTX director chain pre-encodes EVERY distinct per-segment prompt
    // up front (sd_ctx_precompute_chain_text_conds) in one TE residency window, so the
    // resident DiT then runs as one uninterrupted phase with ZERO gemma encodes
    // interleaved between segments. SDCondition holds owning sd::Tensor<float> host
    // buffers, so cached copies never alias a reused encode buffer. Key =
    // prompt + '\x1f' + negative_prompt (the negative is constant across a chain).
    struct CachedTextCond {
        SDCondition cond;
        SDCondition uncond;
        bool        has_uncond = false;
    };
    std::map<std::string, CachedTextCond> avatar_cond_cache;
    static std::string text_cond_key(const std::string& prompt, const std::string& negative_prompt) {
        return prompt + std::string(1, '\x1f') + negative_prompt;
    }
    // Standalone umT5 reload support (Conditioner has no load_from_file): a ModelLoader
    // copy + the TE tensor subset captured at init, mirroring the deferred-DiT loader.
    // Lets reload_cond_stage_model() re-alloc the params buffer and refill it after the
    // TE was freed, without retaining sd_ctx_params.
    std::shared_ptr<ModelLoader> te_reload_loader;
    std::map<std::string, ggml_tensor*> te_reload_tensors;
    std::set<std::string> te_reload_ignore_tensors;
    bool te_reload_use_mmap = false;

    // FIX 3 (LTXAV_TWOSTAGE_FREE_UNUSED): reload state so the video + audio VAE params can be
    // brought back for the final decode after being freed to make VRAM headroom during the stage-2
    // DiT forward. Mirrors the umT5 te_reload pattern: free_params_buffer() nulls the tensors' data
    // pointers, and a retained ModelLoader + load_tensors() re-materializes the weights from disk
    // (~0.1s for the ~1.4GB VAE with a warm page cache). Captured only for LTXAV + no-mmap (mmap'd
    // weights are never freed, so there is neither VRAM to reclaim nor a reload to do).
    std::shared_ptr<ModelLoader> resident_reload_loader;
    bool                          resident_reload_use_mmap = false;

    // Lever 3 (WAN_VAE_FREE_DURING_DIT) reload support: same mechanism as the TE reload
    // above but for the first-stage (VAE) params. Lets reload_first_stage_model() re-alloc
    // + refill the VAE params buffer after they were freed for the DiT sample loop, so the
    // ~254MB VAE weights don't sit resident+unused through the whole sample. Captured only
    // when the lever env is set (one-shot CLI, !keep_diffusion_model_resident).
    std::shared_ptr<ModelLoader> vae_reload_loader;
    std::map<std::string, ggml_tensor*> vae_reload_tensors;
    std::set<std::string> vae_reload_ignore_tensors;
    bool vae_reload_use_mmap = false;
    // Dual-DiT (base/edit) hot-swap state. The flags mirror the boot mmap
    // config (sd_ctx_params isn't retained); the store holds the mmap of the
    // CURRENTLY-SWAPPED DiT so its file-backed pages stay alive. Reassigned on
    // each swap (drops the prior swapped DiT's mapping). The boot variant's DiT
    // mapping lives in `mmap_tensor_store` and is left mapped (reclaimable).
    bool dit_swap_enable_mmap   = false;
    bool dit_swap_writable_mmap = false;
    std::vector<MmapTensorStore> dit_swap_mmap_store;

    bool circular_x = false;
    bool circular_y = false;

    std::shared_ptr<RNG> rng         = std::make_shared<PhiloxRNG>();
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float default_flow_shift         = INFINITY;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd or wan2.1 i2v
    std::shared_ptr<DiffusionModelRunner> diffusion_model;
    std::shared_ptr<DiffusionModelRunner> high_noise_diffusion_model;
    std::shared_ptr<VAE> first_stage_model;
    std::shared_ptr<VAE> preview_vae;
    std::shared_ptr<LTXV::LTXAudioVAERunner> audio_vae_model;
    std::shared_ptr<LTXVUpsampler::LatentUpsamplerRunner> ltx_latent_upsampler;
    std::string ltx_latent_upsampler_path;
    std::shared_ptr<LONGCAT_AUDIO::WhisperEncoderRunner> whisper_encoder_model;  // LongCat-Avatar audio
    std::shared_ptr<ControlNet> control_net;
    std::vector<std::shared_ptr<GenerationExtension>> generation_extensions;
    std::vector<std::shared_ptr<LoraModel>> cond_stage_lora_models;
    std::vector<std::shared_ptr<LoraModel>> diffusion_lora_models;
    std::vector<std::shared_ptr<LoraModel>> first_stage_lora_models;
    bool apply_lora_immediately = false;

    std::string taesd_path;
    sd_tiling_params_t vae_tiling_params = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
    bool offload_params_to_cpu           = false;
    float max_vram                       = 0.f;
    bool stream_layers                   = false;
    std::string backend_spec;
    std::string params_backend_spec;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    std::map<std::string, ggml_tensor*> tensors;

    // Deferred DiT weight load (avatar umT5-on-GPU path): when the text encoder is
    // resident on the GPU, umT5 (~6 GB) + DiT (~8.5 GB) cannot coexist at load. We
    // load+encode+free the TE first, then load the DiT weights. The loader is kept
    // alive so the second load_tensors() pass can read the DiT tensors from the file.
    std::shared_ptr<ModelLoader> deferred_loader;
    std::map<std::string, ggml_tensor*> deferred_dit_tensors;
    std::set<std::string> deferred_ignore_tensors;
    bool dit_load_deferred  = false;
    bool deferred_use_mmap  = false;

    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() = default;

    ggml_backend_t backend_for(SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.runtime_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    ggml_backend_t params_backend_for(SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.params_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    bool ensure_backend_pair(SDBackendModule module) {
        if (backend_for(module) == nullptr) {
            return false;
        }
        return params_backend_for(module) != nullptr;
    }

    bool init_backend(const sd_ctx_params_t* sd_ctx_params) {
        std::string error;
        if (!backend_manager.init(sd_ctx_params->backend,
                                  sd_ctx_params->params_backend,
                                  offload_params_to_cpu,
                                  sd_ctx_params->keep_clip_on_cpu,
                                  sd_ctx_params->keep_vae_on_cpu,
                                  sd_ctx_params->keep_control_net_on_cpu,
                                  &error)) {
            LOG_ERROR("backend config failed: %s", error.c_str());
            return false;
        }
        return ensure_backend_pair(SDBackendModule::DIFFUSION);
    }

    std::shared_ptr<RNG> get_rng(rng_type_t rng_type) {
        if (rng_type == STD_DEFAULT_RNG) {
            return std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CPU_RNG) {
            return std::make_shared<MT19937RNG>();
        } else {  // default: CUDA_RNG
            return std::make_shared<PhiloxRNG>();
        }
    }

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;
        max_vram                = sd_ctx_params->max_vram;
        stream_layers           = sd_ctx_params->stream_layers;
        backend_spec            = SAFE_STR(sd_ctx_params->backend);
        params_backend_spec     = SAFE_STR(sd_ctx_params->params_backend);
        if (stream_layers && max_vram == 0.f) {
            LOG_WARN("--stream-layers has no effect without --max-vram set; ignoring");
            stream_layers = false;
        }
        if (stream_layers && !offload_params_to_cpu && params_backend_spec.empty()) {
            // Streaming needs CPU-resident params.
            LOG_WARN("--stream-layers has no effect without --offload-to-cpu (or --params-backend); ignoring");
            stream_layers = false;
        }

        bool use_tae       = false;
        bool use_audio_vae = false;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        if (!init_backend(sd_ctx_params)) {
            return false;
        }
        max_vram = sd::ggml_graph_cut::resolve_max_vram_gib(max_vram, backend_for(SDBackendModule::DIFFUSION));

        ModelLoader model_loader;

        if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
            LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
            if (!model_loader.init_from_file(sd_ctx_params->model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", sd_ctx_params->model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) > 0) {
            LOG_INFO("loading diffusion model from '%s'", sd_ctx_params->diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->diffusion_model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->diffusion_model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
            LOG_INFO("loading high noise diffusion model from '%s'", sd_ctx_params->high_noise_diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->high_noise_diffusion_model_path, "model.high_noise_diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->high_noise_diffusion_model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->uncond_diffusion_model_path)) > 0) {
            LOG_INFO("loading unconditional diffusion model from '%s'", sd_ctx_params->uncond_diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->uncond_diffusion_model_path, "model.diffusion_model.uncond.")) {
                LOG_WARN("loading unconditional diffusion model from '%s' failed", sd_ctx_params->uncond_diffusion_model_path);
            }
        }

        bool is_unet = sd_version_is_unet(model_loader.get_sd_version());

        if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
            LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
            std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, prefix)) {
                LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
            LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
            std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, prefix)) {
                LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
            LOG_INFO("loading clip_vision from '%s'", sd_ctx_params->clip_vision_path);
            std::string prefix = "cond_stage_model.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_vision_path, prefix)) {
                LOG_WARN("loading clip_vision from '%s' failed", sd_ctx_params->clip_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_path)) > 0) {
            LOG_INFO("loading llm from '%s'", sd_ctx_params->llm_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_path, "text_encoders.llm.")) {
                LOG_WARN("loading llm from '%s' failed", sd_ctx_params->llm_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_vision_path)) > 0) {
            LOG_INFO("loading llm vision from '%s'", sd_ctx_params->llm_vision_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_vision_path, "text_encoders.llm.visual.")) {
                LOG_WARN("loading llm vision from '%s' failed", sd_ctx_params->llm_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
            LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
                external_vae_is_invalid = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0) {
            LOG_INFO("loading tae from '%s'", sd_ctx_params->taesd_path);
            if (!model_loader.init_from_file(sd_ctx_params->taesd_path, "tae.")) {
                LOG_WARN("loading tae from '%s' failed", sd_ctx_params->taesd_path);
            } else {
                use_tae = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->embeddings_connectors_path)) > 0) {
            LOG_INFO("loading embeddings connectors from '%s'", sd_ctx_params->embeddings_connectors_path);
            if (!model_loader.init_from_file(sd_ctx_params->embeddings_connectors_path)) {
                LOG_WARN("loading embeddings connectors from '%s' failed", sd_ctx_params->embeddings_connectors_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->audio_vae_path)) > 0) {
            LOG_INFO("loading LTX audio VAE from '%s'", sd_ctx_params->audio_vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->audio_vae_path)) {
                LOG_WARN("loading LTX audio VAE weights from '%s' failed", sd_ctx_params->audio_vae_path);
            } else {
                use_audio_vae = true;
            }
        }

        model_loader.convert_tensors_name();

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
            return false;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        ggml_type wtype               = (int)sd_ctx_params->wtype < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)
                                            ? (ggml_type)sd_ctx_params->wtype
                                            : GGML_TYPE_COUNT;
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }

        std::map<ggml_type, uint32_t> wtype_stat                 = model_loader.get_wtype_stat();
        std::map<ggml_type, uint32_t> conditioner_wtype_stat     = model_loader.get_conditioner_wtype_stat();
        std::map<ggml_type, uint32_t> diffusion_model_wtype_stat = model_loader.get_diffusion_model_wtype_stat();
        std::map<ggml_type, uint32_t> vae_wtype_stat             = model_loader.get_vae_wtype_stat();

        auto wtype_stat_to_str = [](const std::map<ggml_type, uint32_t>& m, int key_width = 8, int value_width = 5) -> std::string {
            std::ostringstream oss;
            bool first = true;
            for (const auto& [type, count] : m) {
                if (!first)
                    oss << "|";
                first = false;
                oss << std::right << std::setw(key_width) << ggml_type_name(type)
                    << ": "
                    << std::left << std::setw(value_width) << count;
            }
            return oss.str();
        };

        LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
        LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
        LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
        LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (sd_ctx_params->lora_apply_mode == LORA_APPLY_AUTO) {
            bool have_quantized_weight = false;
            if (wtype != GGML_TYPE_COUNT && ggml_is_quantized(wtype)) {
                have_quantized_weight = true;
            } else {
                for (const auto& [type, _] : wtype_stat) {
                    if (ggml_is_quantized(type)) {
                        have_quantized_weight = true;
                        break;
                    }
                }
            }
            // Avoid full-model LoRA merge buffers on constrained setups.
            const bool streaming_constrained = stream_layers ||
                                               sd_ctx_params->offload_params_to_cpu;
            if (have_quantized_weight || streaming_constrained) {
                apply_lora_immediately = false;
            } else {
                apply_lora_immediately = true;
            }
        } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
            apply_lora_immediately = true;
        } else {
            apply_lora_immediately = false;
        }

        std::map<std::string, ggml_tensor*> mmap_able_tensors;
        bool enable_mmap_tensors = false;
        bool needs_writable_mmap = false;
        if (sd_ctx_params->enable_mmap) {
            if (apply_lora_immediately) {
                needs_writable_mmap = true;
                LOG_WARN("in mode 'immediately', LoRAs will cause extra memory usage with mmap");
            }
            enable_mmap_tensors = true;
        }
        // Capture the mmap flags so swap_diffusion_model() (which has no
        // sd_ctx_params) can re-map a swapped DiT the same way as the boot load
        // — keeping the swapped weights in reclaimable file-backed page cache
        // instead of anon RAM (preserves the --mmap host-RAM win across swaps).
        dit_swap_enable_mmap   = enable_mmap_tensors;
        dit_swap_writable_mmap = needs_writable_mmap;

        // split definition to avoid msvc choking on the extra parameter handling
        auto module_can_mmap = [&](SDBackendModule module) {
            return enable_mmap_tensors &&
                   (backend_manager.runtime_backend_is_cpu(module) ||
                    backend_manager.params_backend_is_cpu(module) ||
                    backend_manager.runtime_backend_supports_host_buffer(module));
        };

        // LongCat: force the DiT params into a real (pinned) host buffer instead
        // of mmap. Under --offload-to-cpu + CUDA, alloc_params_buffer() then puts
        // the weights in cudaMallocHost pinned memory, so the per-step weight H2D
        // is async/batched DMA (~6-12 GB/s) instead of the pageable-mmap sync
        // bounce path. Measured on Q4 LTX-2.3: the mmap offload was 14.4 s/step
        // for ~16.4 GB (= ~1.1 GB/s, sync-per-tensor bound). TE + VAE stay mmap'd
        // (reclaimable file pages), so only the DiT (~16.4 GB) is pinned — fits a
        // 31 GB host where full --no-mmap (~27.5 GB pinned) does not. Env-gated,
        // OFF by default: the prod avatar (and every other recipe) is unchanged.
        static const bool dit_no_mmap = []{
            const char* s = getenv("LONGCAT_DIT_NO_MMAP");
            return s && s[0] == '1';
        }();

        auto get_param_tensors_p = [&](auto&& model, bool do_mmap, const char* prefix) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp, prefix);
            for (const auto& [key, tensor] : temp) {
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        auto get_param_tensors = [&](auto&& model, bool do_mmap) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp);
            for (const auto& [key, tensor] : temp) {
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        if (sd_version_is_control(version)) {
            // Might need vae encode for control cond
            vae_decode_only = false;
        }
        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            tae_preview_only = false;
            use_tae          = true;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(max_vram);

        {
            if (!ensure_backend_pair(SDBackendModule::TE) ||
                !ensure_backend_pair(SDBackendModule::DIFFUSION)) {
                return false;
            }

            if (sd_version_is_sd3(version)) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                     params_backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map);
                diffusion_model  = std::make_shared<MMDiTRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                params_backend_for(SDBackendModule::DIFFUSION),
                                                                tensor_storage_map,
                                                                "model.diffusion_model");
            } else if (sd_version_is_pid(version)) {
                vae_decode_only  = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Pid::PiDRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                   params_backend_for(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model.net");
            } else if (sd_version_is_ideogram4(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false);
                diffusion_model  = std::make_shared<Ideogram4::Ideogram4Runner>(backend_for(SDBackendModule::DIFFUSION),
                                                                               params_backend_for(SDBackendModule::DIFFUSION),
                                                                               tensor_storage_map,
                                                                               "model.diffusion_model");
            } else if (sd_version_is_flux(version)) {
                bool is_chroma = false;
                for (auto pair : tensor_storage_map) {
                    if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                        is_chroma = true;
                        break;
                    }
                }
                if (is_chroma) {
                    if ((sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) && sd_ctx_params->chroma_use_dit_mask) {
                        LOG_WARN(
                            "!!!It looks like you are using Chroma with flash attention. "
                            "This is currently unsupported. "
                            "If you find that the generated images are broken, "
                            "try either disabling flash attention or specifying "
                            "--chroma-disable-dit-mask as a workaround.");
                    }

                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                        params_backend_for(SDBackendModule::TE),
                                                                        tensor_storage_map,
                                                                        sd_ctx_params->chroma_use_t5_mask,
                                                                        sd_ctx_params->chroma_t5_mask_pad);
                } else if (version == VERSION_OVIS_IMAGE) {
                    cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                     params_backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     false);
                } else {
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                          params_backend_for(SDBackendModule::TE),
                                                                          tensor_storage_map);
                }
                diffusion_model = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_ltxav(version)) {
                // Classic LTX-Video 0.9.x checkpoints carry an in-DiT PixArt caption_projection
                // that consumes RAW T5-XXL (4096-dim) embeddings, and have NO Gemma text encoder /
                // text_embedding_projection / connector tensors. Detect that signature and use the
                // reusable T5-XXL conditioner instead of the LTX-2 Gemma path. (LTX-2 has no in-DiT
                // caption_projection, so this branch never fires for it.)
                bool ltxv_t5_caption =
                    tensor_storage_map.find("model.diffusion_model.caption_projection.linear_1.weight") != tensor_storage_map.end() &&
                    tensor_storage_map.find("text_embedding_projection.linear_1.weight") == tensor_storage_map.end() &&
                    tensor_storage_map.find("text_encoders.llm.model.embed_tokens.weight") == tensor_storage_map.end();
                if (ltxv_t5_caption) {
                    LOG_INFO("LTX-Video 0.9.x checkpoint detected (in-DiT T5-XXL caption_projection); using T5-XXL text encoder");
                    auto ltx_t5 = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                   params_backend_for(SDBackendModule::TE),
                                                                   tensor_storage_map,
                                                                   /*use_mask=*/false,
                                                                   /*mask_pad=*/0,
                                                                   /*is_umt5=*/false);
                    // The LTX DiT cross-attends without a padding mask; trim the ~500 pad tokens
                    // so the real prompt tokens are not drowned out (was → washed-out output).
                    ltx_t5->trim_to_valid = true;
                    cond_stage_model      = ltx_t5;
                } else {
                    cond_stage_model = std::make_shared<LTXAVEmbedder>(backend_for(SDBackendModule::TE),
                                                                       params_backend_for(SDBackendModule::TE),
                                                                       tensor_storage_map);
                }
                diffusion_model  = std::make_shared<LTXV::LTXAVRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                      params_backend_for(SDBackendModule::DIFFUSION),
                                                                      tensor_storage_map,
                                                                      "model.diffusion_model");
            } else if (sd_version_is_wan(version)) {
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                    params_backend_for(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true);
                diffusion_model  = std::make_shared<WAN::WanRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                   params_backend_for(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model",
                                                                   version);
                if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                    high_noise_diffusion_model = std::make_shared<WAN::WanRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                  params_backend_for(SDBackendModule::DIFFUSION),
                                                                                  tensor_storage_map,
                                                                                  "model.high_noise_diffusion_model",
                                                                                  version);
                }
                if ((diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                     diffusion_model->get_desc() == "Wan2.1-FLF2V-14B" ||
                     diffusion_model->get_desc() == "Wan2.1-I2V-1.3B") &&
                    strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
                    if (!ensure_backend_pair(SDBackendModule::CLIP_VISION)) {
                        return false;
                    }
                    clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend_for(SDBackendModule::CLIP_VISION),
                                                                             params_backend_for(SDBackendModule::CLIP_VISION),
                                                                             tensor_storage_map);
                    clip_vision->set_max_graph_vram_bytes(max_graph_vram_bytes);
                    get_param_tensors(clip_vision, module_can_mmap(SDBackendModule::CLIP_VISION));
                }
            } else if (sd_version_is_qwen_image(version)) {
                bool enable_vision = false;
                if (!vae_decode_only) {
                    enable_vision = true;
                }
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                diffusion_model  = std::make_shared<Qwen::QwenImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                          params_backend_for(SDBackendModule::DIFFUSION),
                                                                          tensor_storage_map,
                                                                          "model.diffusion_model",
                                                                          version,
                                                                          sd_ctx_params->qwen_image_zero_cond_t);
            } else if (sd_version_is_longcat(version)) {
                bool enable_vision = false;
                if (!vae_decode_only) {
                    enable_vision = true;
                }
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_longcat_avatar(version)) {
                // umT5 text encoder (loaded under text_encoders.t5xxl, is_umt5 auto-detected).
                // TODO(audio): the whisper audio encoder + AudioProjModel are not wired yet.
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                    params_backend_for(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true);
                diffusion_model  = std::make_shared<LongCatAvatarModel>(backend_for(SDBackendModule::DIFFUSION),
                                                                       params_backend_for(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.diffusion_model",
                                                                       version);
            } else if (version == VERSION_HIDREAM_O1) {
                cond_stage_model = std::make_shared<HiDreamO1::HiDreamO1Conditioner>(backend_for(SDBackendModule::TE),
                                                                                     params_backend_for(SDBackendModule::TE),
                                                                                     tensor_storage_map);
                diffusion_model  = std::make_shared<HiDreamO1::HiDreamO1Runner>(backend_for(SDBackendModule::DIFFUSION),
                                                                               params_backend_for(SDBackendModule::DIFFUSION),
                                                                               tensor_storage_map,
                                                                               "model");
            } else if (sd_version_is_anima(version)) {
                cond_stage_model = std::make_shared<AnimaConditioner>(backend_for(SDBackendModule::TE),
                                                                      params_backend_for(SDBackendModule::TE),
                                                                      tensor_storage_map);
                diffusion_model  = std::make_shared<Anima::AnimaRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                       params_backend_for(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.diffusion_model");
            } else if (sd_version_is_z_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<ZImage::ZImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                         params_backend_for(SDBackendModule::DIFFUSION),
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         version);
            } else if (sd_version_is_ernie_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<ErnieImage::ErnieImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                 params_backend_for(SDBackendModule::DIFFUSION),
                                                                                 tensor_storage_map,
                                                                                 "model.diffusion_model");
            } else if (sd_version_is_lens(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Lens::LensRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model");
            } else {  // SD1.x SD2.x SDXL
                std::map<std::string, std::string> embbeding_map;
                for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                    embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
                }
                cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(backend_for(SDBackendModule::TE),
                                                                                       params_backend_for(SDBackendModule::TE),
                                                                                       tensor_storage_map,
                                                                                       embbeding_map,
                                                                                       version);
                diffusion_model  = std::make_shared<UNetModelRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                    params_backend_for(SDBackendModule::DIFFUSION),
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model",
                                                                    version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the diffusion model");
                    diffusion_model->set_conv2d_direct_enabled(true);
                }
            }

            // LONGCAT_ENCODE_MAX_VRAM: give the text encoder (cond stage) its own,
            // typically lower, graph-cut budget. The encoder runs in its own phase
            // (gemma is freed before the DiT), and its graph-cut greedily fills the
            // budget with resident weights — so at the shared --max-vram it parks the
            // render peak at the ceiling even though the DiT/VAE phases sit well under
            // it (LTX 1280x704: encode 7504 vs DiT 6832 vs VAE 4914). Capping the encode
            // near the DiT peak drops the overall peak to DiT-bound (~700 MB headroom)
            // with the DiT still at full --max-vram speed. Pairs with LTX_TEXT_MINLEN
            // (smaller N^2 activation => cheaper low-budget encode). GiB; <=0/unset keeps
            // the shared budget.
            size_t cond_graph_vram_bytes = max_graph_vram_bytes;
            if (const char* e = getenv("LONGCAT_ENCODE_MAX_VRAM")) {
                float enc_gib = (float)atof(e);
                if (enc_gib > 0.f) {
                    cond_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(enc_gib);
                    LOG_INFO("LONGCAT_ENCODE_MAX_VRAM: text-encoder graph budget = %.2f GiB (DiT keeps %.2f)", enc_gib, max_vram);
                }
            }
            cond_stage_model->set_max_graph_vram_bytes(cond_graph_vram_bytes);
            get_param_tensors(cond_stage_model, module_can_mmap(SDBackendModule::TE));

            diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
            diffusion_model->set_stream_layers_enabled(stream_layers);
            get_param_tensors(diffusion_model, module_can_mmap(SDBackendModule::DIFFUSION) && !dit_no_mmap);

            if (sd_version_is_unet_edit(version)) {
                vae_decode_only = false;
            }

            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
                high_noise_diffusion_model->set_stream_layers_enabled(stream_layers);
                get_param_tensors(high_noise_diffusion_model, module_can_mmap(SDBackendModule::DIFFUSION) && !dit_no_mmap);
            }

            if (!ensure_backend_pair(SDBackendModule::VAE)) {
                return false;
            }

            auto create_tae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version) ||
                    sd_version_is_ltxav(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(backend_for(SDBackendModule::VAE),
                                                                  params_backend_for(SDBackendModule::VAE),
                                                                  tensor_storage_map,
                                                                  "decoder",
                                                                  vae_decode_only,
                                                                  version);

                } else {
                    auto model = std::make_shared<TinyImageAutoEncoder>(backend_for(SDBackendModule::VAE),
                                                                        params_backend_for(SDBackendModule::VAE),
                                                                        tensor_storage_map,
                                                                        "decoder.layers",
                                                                        vae_decode_only,
                                                                        version);
                    return model;
                }
            };

            sd_vae_format_t vae_format = sd_ctx_params->vae_format;
            if (vae_format < SD_VAE_FORMAT_AUTO || vae_format >= SD_VAE_FORMAT_COUNT) {
                LOG_WARN("invalid VAE format override, using auto");
                vae_format = SD_VAE_FORMAT_AUTO;
            }
            SDVersion vae_version = version;
            if (sd_version_is_pid(version) && vae_format != SD_VAE_FORMAT_AUTO) {
                vae_version = sd_vae_format_to_version(vae_format, vae_version);
            }

            auto create_vae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_ltxav(version)) {
                    return std::make_shared<LTXVideoVAE>(backend_for(SDBackendModule::VAE),
                                                         params_backend_for(SDBackendModule::VAE),
                                                         tensor_storage_map,
                                                         "first_stage_model",
                                                         vae_decode_only,
                                                         version);
                } else if (sd_version_is_wan(version) ||
                           sd_version_is_qwen_image(version) ||
                           sd_version_is_anima(version) ||
                           sd_version_is_longcat_avatar(version)) {
                    return std::make_shared<WAN::WanVAERunner>(backend_for(SDBackendModule::VAE),
                                                               params_backend_for(SDBackendModule::VAE),
                                                               tensor_storage_map,
                                                               "first_stage_model",
                                                               vae_decode_only,
                                                               version);
                } else {
                    auto model = std::make_shared<AutoEncoderKL>(backend_for(SDBackendModule::VAE),
                                                                 params_backend_for(SDBackendModule::VAE),
                                                                 tensor_storage_map,
                                                                 "first_stage_model",
                                                                 vae_decode_only,
                                                                 false,
                                                                 vae_version);
                    if (sd_version_is_sdxl(version) &&
                        (strlen(SAFE_STR(sd_ctx_params->vae_path)) == 0 || sd_ctx_params->force_sdxl_vae_conv_scale || external_vae_is_invalid)) {
                        float vae_conv_2d_scale = 1.f / 32.f;
                        LOG_WARN(
                            "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, "
                            "using Conv2D scale %.3f",
                            vae_conv_2d_scale);
                        model->set_conv2d_scale(vae_conv_2d_scale);
                    }
                    return model;
                }
            };

            bool vae_mmap = module_can_mmap(SDBackendModule::VAE);

            if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1) {
                LOG_INFO("using FakeVAE");
                first_stage_model = std::make_shared<FakeVAE>(version,
                                                              backend_for(SDBackendModule::VAE),
                                                              params_backend_for(SDBackendModule::VAE));
            } else if (use_tae && !tae_preview_only) {
                LOG_INFO("using TAE for encoding / decoding");
                first_stage_model = create_tae();
                first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
                get_param_tensors_p(first_stage_model, vae_mmap, "tae");
            } else {
                LOG_INFO("using VAE for encoding / decoding");
                first_stage_model = create_vae();
                first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
                get_param_tensors_p(first_stage_model, vae_mmap, "first_stage_model");
                if (use_tae && tae_preview_only) {
                    LOG_INFO("using TAE for preview");
                    preview_vae = create_tae();
                    preview_vae->set_max_graph_vram_bytes(max_graph_vram_bytes);
                    get_param_tensors_p(preview_vae, vae_mmap, "tae");
                }
            }

            if (use_audio_vae && sd_version_is_longcat_avatar(version)) {
                // For LongCat-Avatar, the "audio_vae_path" file is the whisper-large-v3
                // ENCODER gguf (audio features for lip-sync), not an LTX audio VAE.
                use_audio_vae         = false;
                whisper_encoder_model = std::make_shared<LONGCAT_AUDIO::WhisperEncoderRunner>(
                    backend_for(SDBackendModule::TE),
                    params_backend_for(SDBackendModule::TE),
                    tensor_storage_map,
                    "audio_encoder");
                whisper_encoder_model->set_flash_attention_enabled(false);
                get_param_tensors_p(whisper_encoder_model, module_can_mmap(SDBackendModule::TE), "audio_encoder");
            } else if (use_audio_vae) {
                audio_vae_model = std::make_shared<LTXV::LTXAudioVAERunner>(backend_for(SDBackendModule::VAE),
                                                                            params_backend_for(SDBackendModule::VAE),
                                                                            tensor_storage_map);
                get_param_tensors_p(audio_vae_model, vae_mmap, "");
            }

            // GGML_CUDNN_CONV=1 routes VAE 2D convs through GGML_OP_CONV_2D so the
            // env-gated cuDNN implicit-GEMM conv path in ggml-cuda intercepts them
            // (replacing the heavy im2col+GEMM VAE decode convs).
            //
            // Do NOT enable conv2d-direct for GGML_CUDNN_CONV3D alone. That env scopes cuDNN
            // to the *3D* convs (the im2col_3d IC*27 VRAM blowup); the 2D convs (3x3 resample +
            // 1x1 attention qkv/proj) must stay on the fast im2col+GEMM (tensor-core) path.
            // Forcing them to GGML_OP_CONV_2D here, while the cuDNN-conv2d interceptor only
            // activates for GGML_CUDNN_CONV (conv2d-cudnn.cu), dropped every 2D conv onto the
            // naive spatial conv2d_kernel (conv2d.cu) -> 2.7x slower VAE decode (the naive
            // kernel was ~59% of GPU time; the 1x1 pointwise convs are matmuls and suffer most).
            // With direct off these 2D convs go ggml_conv_2d -> IM2COL + MUL_MAT = the validated
            // baseline path (zero conv2d_kernel), while 3D convs still take cuDNN (low VRAM).
            // (vae_conv_direct + GGML_CUDNN_CONV3D is honored: conv2d-cudnn now intercepts both
            // envs, so that explicit combo gets the cuDNN 2D path rather than the naive kernel.)
            if (sd_ctx_params->vae_conv_direct || getenv("GGML_CUDNN_CONV")) {
                LOG_INFO("Using Conv2d direct in the vae model");
                first_stage_model->set_conv2d_direct_enabled(true);
                if (preview_vae) {
                    preview_vae->set_conv2d_direct_enabled(true);
                }
            }

            if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
                if (!ensure_backend_pair(SDBackendModule::CONTROL_NET)) {
                    return false;
                }
                control_net = std::make_shared<ControlNet>(backend_for(SDBackendModule::CONTROL_NET),
                                                           params_backend_for(SDBackendModule::CONTROL_NET),
                                                           tensor_storage_map,
                                                           version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the control net");
                    control_net->set_conv2d_direct_enabled(true);
                }
            }

            {
                generation_extensions.clear();
                auto photomaker_extension = create_photomaker_extension();
                GenerationExtensionInitContext extension_ctx{
                    sd_ctx_params,
                    version,
                    tensor_storage_map,
                    model_loader,
                    n_threads,
                    [this](SDBackendModule module) { return ensure_backend_pair(module); },
                    [this](SDBackendModule module) { return backend_for(module); },
                    [this](SDBackendModule module) { return params_backend_for(module); },
                };
                if (!photomaker_extension->init(extension_ctx)) {
                    return false;
                }
                if (photomaker_extension->is_enabled()) {
                    generation_extensions.push_back(photomaker_extension);
                }
            }
            {
                GenerationExtensionTensorContext extension_tensor_ctx{
                    tensors,
                    mmap_able_tensors,
                    module_can_mmap,
                };
                for (auto& extension : generation_extensions) {
                    extension->collect_param_tensors(extension_tensor_ctx);
                }
            }

            if (sd_ctx_params->flash_attn) {
                LOG_INFO("Using flash attention");
                cond_stage_model->set_flash_attention_enabled(true);
                if (clip_vision) {
                    clip_vision->set_flash_attention_enabled(true);
                }
                if (first_stage_model) {
                    first_stage_model->set_flash_attention_enabled(true);
                }
                if (preview_vae) {
                    preview_vae->set_flash_attention_enabled(true);
                }
            }

            if (sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) {
                LOG_INFO("Using flash attention in the diffusion model");
                diffusion_model->set_flash_attention_enabled(true);
                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->set_flash_attention_enabled(true);
                }
            }

            diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            if (control_net) {
                control_net->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            circular_x = sd_ctx_params->circular_x;
            circular_y = sd_ctx_params->circular_y;
        }

        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tae && !tae_preview_only) {
            ignore_tensors.insert("first_stage_model.");
        }
        for (auto& extension : generation_extensions) {
            extension->add_ignore_tensors(ignore_tensors);
        }
        ignore_tensors.insert("model.diffusion_model.__x0__");
        ignore_tensors.insert("model.diffusion_model.__32x32__");
        ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.conv1");
            ignore_tensors.insert("first_stage_model.quant");
            ignore_tensors.insert("tae.encoder");
            ignore_tensors.insert("text_encoders.llm.visual.");
        }
        if (audio_vae_model && !audio_vae_model->config.has_encoder) {
            // Decode-only audio VAE (no baked mel basis) → skip the unused encoder convs.
            // When the file DOES carry the encoder + mel_stft basis (the -ENC gguf), the
            // encoder block IS built and its tensors must load (LTXAV audio-DRIVE path).
            ignore_tensors.insert("audio_vae.encoder");
        }
        if (version == VERSION_OVIS_IMAGE) {
            ignore_tensors.insert("text_encoders.llm.vision_model.");
            ignore_tensors.insert("text_encoders.llm.visual_tokenizer.");
            ignore_tensors.insert("text_encoders.llm.vte.");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        if (sd_version_is_ernie_image(version)) {
            ignore_tensors.insert("text_encoders.llm.vision_tower.");
            ignore_tensors.insert("text_encoders.llm.multi_modal_projector.");
        }
        if (sd_version_is_lens(version)) {
            ignore_tensors.insert("text_encoders.llm.tokenizer_json");
            ignore_tensors.insert("text_encoders.llm.model.layers.0.mlp.experts.gate_up_proj.weight_scale_2");
            ignore_tensors.insert("text_encoders.llm.model.layers.0.mlp.experts.down_proj.weight_scale_2");
        }
        if (sd_version_is_ideogram4(version)) {
            ignore_tensors.insert("text_encoders.llm.lm_head.");
            ignore_tensors.insert("text_encoders.llm.visual.");
            ignore_tensors.insert("text_encoders.llm.vision_model.");
            ignore_tensors.insert("text_encoders.llm.tokenizer_json");
        }
        if (version == VERSION_HIDREAM_O1) {
            ignore_tensors.insert("lm_head.");
            ignore_tensors.insert("model.visual.deepstack_merger_list.");
        }

        if (enable_mmap_tensors) {
            if (mmap_able_tensors.empty()) {
                LOG_DEBUG("no tensors could be memory-mapped");
            } else {
                mmap_tensor_store = model_loader.mmap_tensors(mmap_able_tensors, ignore_tensors, needs_writable_mmap);
            }
        }

        if (clip_vision && !clip_vision->alloc_params_buffer()) {
            LOG_ERROR("CLIP vision params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (cond_stage_model && !cond_stage_model->alloc_params_buffer()) {
            LOG_ERROR("Conditioner model params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }

        // Avatar umT5-on-GPU: defer the DiT weight load so umT5 (~6 GB) and the DiT
        // (~8.5 GB) never coexist on the 12 GB card. We alloc+load+encode+free the
        // text encoder on the GPU first, then alloc+load the DiT. Only engage when the
        // TE actually lives on the GPU (otherwise nothing is gained) and weights are
        // freed-immediately (one-shot CLI). The DiT tensors are split out of the load
        // map here and loaded by finalize_deferred_dit_load() after the TE is freed.
        dit_load_deferred = false;
        if (sd_version_is_longcat_avatar(version) && diffusion_model && cond_stage_model &&
            free_params_immediately && !sd_ctx_params->enable_mmap &&
            !sd_backend_is_cpu(params_backend_for(SDBackendModule::TE)) &&
            !sd_backend_is_cpu(params_backend_for(SDBackendModule::DIFFUSION))) {
            for (auto it = tensors.begin(); it != tensors.end();) {
                if (starts_with(it->first, "model.diffusion_model.")) {
                    deferred_dit_tensors[it->first] = it->second;
                    it                              = tensors.erase(it);
                } else {
                    ++it;
                }
            }
            if (!deferred_dit_tensors.empty()) {
                dit_load_deferred        = true;
                deferred_ignore_tensors  = ignore_tensors;
                deferred_use_mmap        = sd_ctx_params->enable_mmap;
                deferred_loader          = std::make_shared<ModelLoader>(model_loader);
                LOG_INFO("avatar: deferring DiT weight load (umT5-on-GPU); %zu DiT tensors",
                         deferred_dit_tensors.size());
            }
        }

        // Capture umT5 reload state: the TE is freed after each text encode
        // (free_params_immediately), but a prompt change on a warm resident worker
        // must recompute the conditioning, which requires reloading umT5. Keep a
        // loader copy + the TE tensor subset so reload_cond_stage_model() can
        // re-alloc + refill the params buffer on demand. This is needed REGARDLESS of
        // whether the TE lives on GPU or CPU (--clip-on-cpu): the deferred-DiT block
        // above only engages for umT5-on-GPU, but the free-then-reload-on-prompt-change
        // requirement is independent of placement. Gate only on avatar + freed-weights +
        // no-mmap (mmap'd weights are never freed, so no reload is needed).
        // Existing avatar (umT5, !mmap) PLUS the flux2 IMAGE path (incl. --mmap): a warm
        // resident image worker re-encodes a fresh prompt every /generate, but with NVFP4
        // offload-off the TE (~4.7 GB Qwen3) would otherwise sit on-GPU between requests.
        // Capturing the loader + TE tensor subset lets prepare_image_generation_embeds free
        // it after each cond and reload_cond_stage_model() restore it before the next prompt
        // (release-after-encode -> resident ~6 GB vs ~11). The reload loader re-reads the
        // file, so this works under --mmap too. This condition preserves the avatar's exact
        // prior behavior (avatar => captured only when !mmap) and adds the image path.
        if (cond_stage_model && free_params_immediately &&
            (!sd_ctx_params->enable_mmap || !sd_version_is_longcat_avatar(version))) {
            for (const auto& [key, tensor] : tensors) {
                if (starts_with(key, "text_encoders.t5xxl.transformer") ||
                    starts_with(key, "text_encoders.llm")) {
                    te_reload_tensors[key] = tensor;
                }
            }
            if (!te_reload_tensors.empty()) {
                te_reload_ignore_tensors = ignore_tensors;
                te_reload_use_mmap       = sd_ctx_params->enable_mmap;
                te_reload_loader         = std::make_shared<ModelLoader>(model_loader);
            }
        }

        // FIX 3: capture a loader copy so the video/audio VAE can be reloaded after being freed for
        // the stage-2 DiT forward (LTXAV two-stage relip). model_loader already holds every model
        // file's converted tensor metadata; reload re-reads the weights from disk. Skip under mmap
        // (weights are never freed => no VRAM reclaimed, no reload needed).
        if (free_params_immediately && version == VERSION_LTXAV && !sd_ctx_params->enable_mmap) {
            resident_reload_loader   = std::make_shared<ModelLoader>(model_loader);
            resident_reload_use_mmap = sd_ctx_params->enable_mmap;
        }

        // Lever 3 (WAN_VAE_FREE_DURING_DIT): capture the VAE reload state so the ~254MB
        // first-stage params can be freed before the DiT sample loop and reloaded before
        // decode (the VAE is otherwise resident+unused through the whole sample). Same
        // capture pattern as the TE above, gated additionally on the lever env so the
        // default path retains no extra loader. One-shot CLI only (the warm resident
        // worker keeps the VAE across /generate; it has no reload hook on that side).
        static const bool wan_vae_free_during_dit = []{
            const char* s = getenv("WAN_VAE_FREE_DURING_DIT");
            return s && s[0] == '1';
        }();
        if (wan_vae_free_during_dit && first_stage_model && free_params_immediately) {
            for (const auto& [key, tensor] : tensors) {
                if (starts_with(key, "first_stage_model")) {
                    vae_reload_tensors[key] = tensor;
                }
            }
            if (!vae_reload_tensors.empty()) {
                vae_reload_ignore_tensors = ignore_tensors;
                vae_reload_use_mmap       = sd_ctx_params->enable_mmap;
                vae_reload_loader         = std::make_shared<ModelLoader>(model_loader);
            }
        }
        // Skip the eager DiT alloc when deferred (avatar umT5-on-GPU loads it later in
        // finalize_deferred_dit_load); otherwise alloc now and surface failure (upstream).
        if (diffusion_model && !dit_load_deferred && !diffusion_model->alloc_params_buffer()) {
            LOG_ERROR("Diffusion model params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (high_noise_diffusion_model && !high_noise_diffusion_model->alloc_params_buffer()) {
            LOG_ERROR("High noise diffusion model params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (first_stage_model && !first_stage_model->alloc_params_buffer()) {
            LOG_ERROR("VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (preview_vae && !preview_vae->alloc_params_buffer()) {
            LOG_ERROR("preview VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (audio_vae_model && !audio_vae_model->alloc_params_buffer()) {
            LOG_ERROR("LTX audio VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (whisper_encoder_model && !whisper_encoder_model->alloc_params_buffer()) {
            LOG_ERROR("whisper audio encoder params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        for (auto& extension : generation_extensions) {
            if (!extension->alloc_params_buffer()) {
                LOG_ERROR("%s params buffer allocation failed", extension->name());
                ggml_free(ctx);
                return false;
            }
        }

        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads, sd_ctx_params->enable_mmap);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("finished loaded file");

        // P2 (nvfp4 patchy fix): if the DiT gguf is an UNFOLDED import, it carries per-tensor
        // weight globals as "<weight>.wglobal" sibling tensors. Register each against the
        // corresponding graph weight tensor's ACTUAL name (ggml_set_name truncates to
        // GGML_MAX_NAME, so we must register t->name — what src0->name will be at GEMM time —
        // not the untruncated map key). The FP4 cuBLASLt GEMM then folds the global into alpha
        // (alpha = A_global * W_global). A legacy folded gguf has no .wglobal -> map empty ->
        // nothing registered -> GEMM multiplier defaults to 1.0 (byte-identical legacy path).
        // MoE models (e.g. Wan2.2) load TWO DiT ggufs, each under its own runtime prefix:
        // the low-noise expert (--diffusion-model) as "model.diffusion_model." and the
        // high-noise expert (--high-noise-diffusion-model) as "model.high_noise_diffusion_model.".
        // Register each gguf's wglobals against its own prefix — registering only the low
        // expert leaves the high expert's weights with w_global=1.0 (~1e4x too large -> NaN/wash).
        {
            const std::pair<const char*, std::string> dit_legs[] = {
                {SAFE_STR(sd_ctx_params->diffusion_model_path),            "model.diffusion_model."},
                {SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path), "model.high_noise_diffusion_model."},
            };
            for (const auto& leg : dit_legs) {
                if (strlen(leg.first) == 0) {
                    continue;
                }
                std::map<std::string, float> wglobals;
                load_nvfp4_weight_globals(leg.first, wglobals);
                if (wglobals.empty()) {
                    continue;
                }
                const std::string& pfx = leg.second;
                const bool f8_dbg = (getenv("GGML_F8_DBG") != nullptr && atoi(getenv("GGML_F8_DBG")) != 0);
                size_t n_reg = 0;
                for (auto& kv : tensors) {
                    const std::string& full = kv.first;
                    if (full.compare(0, pfx.size(), pfx) != 0 || kv.second == nullptr) {
                        continue;
                    }
                    const std::string bare = full.substr(pfx.size());
                    auto it = wglobals.find(bare + ".wglobal");
                    if (it != wglobals.end()) {
                        // Register under BOTH the graph tensor's actual ->name (what src0->name is
                        // at GEMM time) AND the map-key `full` AND the bare name — belt-and-
                        // suspenders so the FP8/FP4 GEMM's nvfp4_weight_global_for(src0->name) lookup
                        // hits regardless of whether the offload/mmap path leaves src0->name as the
                        // prefixed, bare, or ggml-set name. Extra keys are harmless (same value).
                        ggml_cuda_nvfp4_register_weight_global(kv.second->name, it->second);
                        ggml_cuda_nvfp4_register_weight_global(full.c_str(), it->second);
                        ggml_cuda_nvfp4_register_weight_global(bare.c_str(), it->second);
                        if (f8_dbg && n_reg < 8) {
                            LOG_INFO("[F8_DBG] register wglobal=%.8g  ggml_name='%s'  map_key='%s'  bare='%s'",
                                     it->second, kv.second->name, full.c_str(), bare.c_str());
                        }
                        ++n_reg;
                    }
                }
                LOG_INFO("nvfp4: registered %zu/%zu weight globals (unfolded import) for %s",
                         n_reg, wglobals.size(), pfx.c_str());
            }
        }

        {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            if (high_noise_diffusion_model) {
                unet_params_mem_size += high_noise_diffusion_model->get_params_buffer_size();
            }
            size_t vae_params_mem_size = 0;
            vae_params_mem_size        = first_stage_model->get_params_buffer_size();
            if (preview_vae) {
                vae_params_mem_size += preview_vae->get_params_buffer_size();
            }
            if (audio_vae_model) {
                vae_params_mem_size += audio_vae_model->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(SAFE_STR(sd_ctx_params->control_net_path), n_threads)) {
                    ggml_free(ctx);
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t extension_params_mem_size = 0;
            for (auto& extension : generation_extensions) {
                extension_params_mem_size += extension->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            auto add_params_memory        = [&](size_t size, SDBackendModule module) {
                if (size == 0) {
                    return true;
                }
                ggml_backend_t module_backend = params_backend_for(module);
                if (module_backend == nullptr) {
                    return false;
                }
                if (sd_backend_is_cpu(module_backend)) {
                    total_params_ram_size += size;
                } else {
                    total_params_vram_size += size;
                }
                return true;
            };
            auto params_memory_location = [&](size_t size, SDBackendModule module) {
                if (size == 0) {
                    return "N/A";
                }
                ggml_backend_t module_backend = params_backend_for(module);
                if (module_backend == nullptr) {
                    return "N/A";
                }
                return sd_backend_is_cpu(module_backend) ? "RAM" : "VRAM";
            };

            if (!add_params_memory(clip_params_mem_size, SDBackendModule::TE) ||
                !add_params_memory(extension_params_mem_size, SDBackendModule::PHOTOMAKER) ||
                !add_params_memory(unet_params_mem_size, SDBackendModule::DIFFUSION) ||
                !add_params_memory(vae_params_mem_size, SDBackendModule::VAE) ||
                !add_params_memory(control_net_params_mem_size, SDBackendModule::CONTROL_NET)) {
                ggml_free(ctx);
                return false;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), extensions %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(clip_params_mem_size, SDBackendModule::TE),
                unet_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(unet_params_mem_size, SDBackendModule::DIFFUSION),
                vae_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(vae_params_mem_size, SDBackendModule::VAE),
                control_net_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(control_net_params_mem_size, SDBackendModule::CONTROL_NET),
                extension_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(extension_params_mem_size, SDBackendModule::PHOTOMAKER));
        }

        // init denoiser
        {
            prediction_t pred_type = sd_ctx_params->prediction;

            if (pred_type == PREDICTION_COUNT) {
                if (sd_version_is_sd2(version)) {
                    // check is_using_v_parameterization_for_sd2
                    if (is_using_v_parameterization_for_sd2(sd_version_is_inpaint(version))) {
                        pred_type = V_PRED;
                    } else {
                        pred_type = EPS_PRED;
                    }
                } else if (sd_version_is_sdxl(version)) {
                    if (tensor_storage_map.find("edm_vpred.sigma_max") != tensor_storage_map.end()) {
                        // CosXL models
                        // TODO: get sigma_min and sigma_max values from file
                        pred_type = EDM_V_PRED;
                    } else if (tensor_storage_map.find("v_pred") != tensor_storage_map.end()) {
                        pred_type = V_PRED;
                    } else {
                        pred_type = EPS_PRED;
                    }
                } else if (sd_version_is_sd3(version) ||
                           sd_version_is_wan(version) ||
                           sd_version_is_qwen_image(version) ||
                           version == VERSION_HIDREAM_O1 ||
                           sd_version_is_anima(version) ||
                           sd_version_is_ernie_image(version) ||
                           sd_version_is_longcat_avatar(version) ||
                           sd_version_is_z_image(version) ||
                           sd_version_is_pid(version) ||
                           sd_version_is_ideogram4(version)) {
                    pred_type = FLOW_PRED;
                    if (sd_version_is_wan(version)) {
                        default_flow_shift = 5.f;
                    } else if (sd_version_is_longcat_avatar(version)) {
                        default_flow_shift = 7.f;  // FlowMatchEulerDiscrete shift 7.0
                    } else if (sd_version_is_ernie_image(version)) {
                        default_flow_shift = 4.f;
                    } else if (sd_version_is_pid(version)) {
                        default_flow_shift = 1.5f;
                    } else if (sd_version_is_ideogram4(version)) {
                        default_flow_shift = 1.0f;
                    } else {
                        default_flow_shift = 3.f;
                    }
                } else if (sd_version_is_flux(version) ||
                           sd_version_is_longcat(version) ||
                           sd_version_is_lens(version) ||
                           sd_version_is_ltxav(version)) {
                    pred_type = FLUX_FLOW_PRED;

                    default_flow_shift = 1.0f;  // TODO: validate
                    for (const auto& [name, tensor_storage] : tensor_storage_map) {
                        if (starts_with(name, "model.diffusion_model.guidance_in.in_layer.weight")) {
                            default_flow_shift = 1.15f;
                            break;
                        }
                    }
                    if (sd_version_is_longcat(version)) {
                        default_flow_shift = 3.0f;
                    } else if (sd_version_is_lens(version)) {
                        default_flow_shift = 1.83f;
                    } else if (sd_version_is_ltxav(version)) {
                        default_flow_shift = 2.37f;
                    }
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else {
                    pred_type = EPS_PRED;
                }
            }

            switch (pred_type) {
                case EPS_PRED:
                    LOG_INFO("running in eps-prediction mode");
                    break;
                case V_PRED:
                    LOG_INFO("running in v-prediction mode");
                    denoiser = std::make_shared<CompVisVDenoiser>();
                    break;
                case EDM_V_PRED:
                    LOG_INFO("running in v-prediction EDM mode");
                    denoiser = std::make_shared<EDMVDenoiser>();
                    break;
                case FLOW_PRED: {
                    if (sd_version_is_ltxav(version)) {
                        LOG_INFO("running in LTXAV FLOW mode");
                        denoiser = std::make_shared<FluxFlowDenoiser>();
                    } else {
                        LOG_INFO("running in FLOW mode");
                        denoiser = std::make_shared<DiscreteFlowDenoiser>();
                    }
                    break;
                }
                case FLUX_FLOW_PRED: {
                    LOG_INFO("running in Flux FLOW mode");
                    denoiser = std::make_shared<FluxFlowDenoiser>();
                    break;
                }
                case FLUX2_FLOW_PRED: {
                    LOG_INFO("running in Flux2 FLOW mode");
                    denoiser = std::make_shared<Flux2FlowDenoiser>();
                    break;
                }
                default: {
                    LOG_ERROR("Unknown predition type %i", pred_type);
                    ggml_free(ctx);
                    return false;
                }
            }

            auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
            if (comp_vis_denoiser) {
                for (int i = 0; i < TIMESTEPS; i++) {
                    comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                    comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
                }
            }
        }

        ggml_free(ctx);
        return true;
    }

    // Avatar umT5-on-GPU: alloc the DiT param buffer (GPU) and load its weights from
    // the stashed loader, AFTER the text encoder has been freed. Idempotent / no-op
    // when the load was not deferred. Returns false on alloc/load failure.
    bool finalize_deferred_dit_load() {
        if (!dit_load_deferred) {
            return true;
        }
        dit_load_deferred = false;  // run once
        int64_t t0        = ggml_time_ms();
        if (diffusion_model) {
            diffusion_model->alloc_params_buffer();
        }
        if (!deferred_loader) {
            LOG_ERROR("deferred DiT load: loader missing");
            return false;
        }
        bool ok = deferred_loader->load_tensors(deferred_dit_tensors,
                                                deferred_ignore_tensors,
                                                n_threads,
                                                deferred_use_mmap);
        deferred_loader.reset();
        deferred_dit_tensors.clear();
        deferred_ignore_tensors.clear();
        if (!ok) {
            LOG_ERROR("deferred DiT weight load failed");
            return false;
        }
        LOG_INFO("avatar: deferred DiT weight load completed, taking %.2fs",
                 (ggml_time_ms() - t0) * 1.0f / 1000);
        return true;
    }

    // Reload the umT5 text-encoder params after they were freed
    // (free_params_immediately). Conditioner has no load_from_file, so this mirrors
    // the deferred-DiT loader: re-alloc the params buffer (against the surviving
    // params_ctx tensors) and refill it from the captured ModelLoader. No-op (returns
    // true) if the TE is still resident or no reload state was captured.
    bool reload_cond_stage_model() {
        if (!cond_stage_model) {
            return false;
        }
        if (cond_stage_model->get_params_buffer_size() != 0) {
            return true;  // still resident, nothing to reload
        }
        if (!te_reload_loader || te_reload_tensors.empty()) {
            LOG_ERROR("text-encoder reload requested but no reload state captured");
            return false;
        }
        int64_t t0 = ggml_time_ms();
        cond_stage_model->alloc_params_buffer();
        bool ok = te_reload_loader->load_tensors(te_reload_tensors,
                                                 te_reload_ignore_tensors,
                                                 n_threads,
                                                 te_reload_use_mmap);
        if (!ok) {
            LOG_ERROR("text-encoder reload failed");
            return false;
        }
        LOG_INFO("text-encoder reloaded for prompt change, taking %.2fs",
                 (ggml_time_ms() - t0) * 1.0f / 1000);
        return true;
    }

    // Re-materialize a VAE whose params were freed to make DiT VRAM headroom. No-op (returns true)
    // when still resident, so callers can invoke unconditionally before decode. Dispatches to
    // whichever reload state was captured — the two paths are mutually exclusive (different model
    // versions / gating): resident_reload_loader = LTXAV two-stage relip (FIX 3); vae_reload_loader
    // = the WAN_VAE_FREE_DURING_DIT CLI lever. Both re-alloc a fresh params buffer against the
    // surviving params_ctx tensor structs (only their data pointers were nulled by free) and reload
    // the weights from disk via the retained loader — the reload_cond_stage_model path proven for umT5.
    bool reload_first_stage_model() {
        if (!first_stage_model) {
            return false;
        }
        if (first_stage_model->get_params_buffer_size() != 0) {
            return true;  // still resident
        }
        int64_t t0 = ggml_time_ms();
        first_stage_model->alloc_params_buffer();
        if (resident_reload_loader) {
            // LTXAV two-stage relip path.
            std::map<std::string, ggml_tensor*> t;
            first_stage_model->get_param_tensors(t, "first_stage_model");
            std::set<std::string> ignore;
            if (!resident_reload_loader->load_tensors(t, ignore, n_threads, resident_reload_use_mmap)) {
                LOG_ERROR("video VAE reload failed");
                return false;
            }
        } else if (vae_reload_loader && !vae_reload_tensors.empty()) {
            // WAN_VAE_FREE_DURING_DIT lever path.
            if (!vae_reload_loader->load_tensors(vae_reload_tensors, vae_reload_ignore_tensors,
                                                 n_threads, vae_reload_use_mmap)) {
                LOG_ERROR("VAE reload failed");
                return false;
            }
        } else {
            LOG_ERROR("video VAE reload requested but no reload state captured");
            return false;
        }
        LOG_INFO("video VAE reloaded for decode, taking %.2fs", (ggml_time_ms() - t0) * 1.0f / 1000);
        return true;
    }

    bool reload_audio_vae_model() {
        if (!audio_vae_model) {
            return false;
        }
        if (audio_vae_model->get_params_buffer_size() != 0) {
            return true;  // still resident
        }
        if (!resident_reload_loader) {
            LOG_ERROR("audio VAE reload requested but no reload state captured");
            return false;
        }
        int64_t t0 = ggml_time_ms();
        audio_vae_model->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> t;
        audio_vae_model->get_param_tensors(t, "");
        std::set<std::string> ignore;
        if (!resident_reload_loader->load_tensors(t, ignore, n_threads, resident_reload_use_mmap)) {
            LOG_ERROR("audio VAE reload failed");
            return false;
        }
        LOG_INFO("audio VAE reloaded for decode, taking %.2fs", (ggml_time_ms() - t0) * 1.0f / 1000);
        return true;
    }
    // Pre-encode the text conditioning for a whole set of chained-segment prompts in ONE
    // text-encoder residency window, populating avatar_cond_cache. The per-segment
    // prepare_video_generation_embeds() calls then all hit the cache, so the resident DiT
    // runs as one uninterrupted phase with NO gemma encode interleaved between segments
    // (the snappy long-form path). The negative prompt is constant across a chain, so its
    // uncond is encoded once and shared into every entry. After pre-encoding we free the
    // TE on the deferred GPU-TE path (frees VRAM for the resident DiT) and bring the DiT
    // onto the GPU once; under --mmap both are skipped (mmap'd weights are never freed and
    // finalize_deferred_dit_load is a no-op). No-op unless this is an avatar/LTXAV resident
    // chain with a conditioner.
    void precompute_chain_text_conds(const std::vector<std::string>& prompts,
                                     const std::string&              negative_prompt,
                                     int                             clip_skip,
                                     bool                            need_uncond) {
        if ((!sd_version_is_longcat_avatar(version) && version != VERSION_LTXAV) ||
            !keep_diffusion_model_resident || !cond_stage_model) {
            return;
        }
        if (!reload_cond_stage_model()) {
            LOG_ERROR("LTXAV chain: TE reload failed; cannot pre-encode text conds");
            return;
        }
        int64_t t0 = ggml_time_ms();
        // Constant negative prompt across the chain -> encode its uncond once, share it
        // only when the resolved sampling path will actually consume it.
        SDCondition shared_uncond;
        if (need_uncond) {
            ConditionerParams cp;
            cp.clip_skip       = clip_skip;
            cp.zero_out_masked = true;
            cp.text            = negative_prompt;
            shared_uncond      = cond_stage_model->get_learned_condition(n_threads, cp);
        }
        int encoded = 0;
        for (const auto& p : prompts) {
            std::string key = text_cond_key(p, negative_prompt);
            if (avatar_cond_cache.count(key)) {
                continue;  // distinct prompt already encoded (or repeated across segments)
            }
            ConditionerParams cp;
            cp.clip_skip       = clip_skip;
            cp.zero_out_masked = true;
            cp.text            = p;
            CachedTextCond entry;
            entry.cond             = cond_stage_model->get_learned_condition(n_threads, cp);
            entry.uncond           = shared_uncond;
            entry.has_uncond       = need_uncond;
            avatar_cond_cache[key] = std::move(entry);
            ++encoded;
        }
        LOG_INFO("LTXAV chain: pre-encoded %d distinct text cond(s)%s over %zu segment prompt(s) "
                 "in one TE window, taking %.2fs",
                 encoded, need_uncond ? " + shared uncond" : "", prompts.size(),
                 (ggml_time_ms() - t0) * 1.0f / 1000);
        // We now hold every cond; on the deferred GPU-TE path free the TE to give the
        // resident DiT its VRAM back. Skipped under --mmap (weights are not freed).
        if (free_params_immediately && dit_load_deferred) {
            cond_stage_model->free_params_buffer();
        }
        finalize_deferred_dit_load();
    }

    // Hot-swap the diffusion model (DiT) weights in place from a different gguf,
    // reusing the existing backend, the resident VAE + text encoder, and the
    // already-built DiffusionModelRunner object/param-tensor graph. This is the
    // FLUX.2-Klein base<->edit swap path: both variants share the exact same DiT
    // architecture + tensor names ("model.diffusion_model.*"), only the weights
    // differ (~5.6 GB). We therefore do NOT rebuild the runner; we just free its
    // param buffer (VRAM) and refill it from the new file — the same free-then-
    // reload mechanism reload_cond_stage_model() uses for umT5, which is why the
    // dangling-pointer fix in GGMLRunner::free_params_buffer() is load-bearing here.
    //
    // Must be called with no render in flight (serial async worker thread).
    bool swap_diffusion_model(const std::string& new_diffusion_model_path) {
        if (!diffusion_model) {
            LOG_ERROR("swap_diffusion_model: no diffusion model loaded");
            return false;
        }
        if (new_diffusion_model_path.empty()) {
            LOG_ERROR("swap_diffusion_model: empty path");
            return false;
        }

        int64_t t0 = ggml_time_ms();

        // 1. A variant swap replaces the outgoing DiT, so it must not retain that
        //    model's warm GPU residency while the incoming weights are loaded.
        //    free_params_buffer() alone is insufficient for the LTXAV offload
        //    path: it releases params_buffer/runtime_params_buffer but deliberately
        //    leaves resident_runtime_params_buffer (LONGCAT_SHARED_RESIDENT), the
        //    graph-cut streaming/prefetch buffers, and the temporal cache alive.
        //    Those are owned by the old weights and otherwise stack with the new
        //    variant. Default-on; setting LTXAV_SWAP_FREE_OUTGOING=0 preserves the
        //    old swap behaviour for A/B comparison.
        const char* swap_free_outgoing_env = getenv("LTXAV_SWAP_FREE_OUTGOING");
        const bool free_outgoing_ltxav = sd_version_is_ltxav(version) &&
                                         (swap_free_outgoing_env == nullptr ||
                                          std::string(swap_free_outgoing_env) != "0");
        diffusion_model->free_compute_buffer();
        if (free_outgoing_ltxav) {
            if (diffusion_model->params_offloaded_to_host()) {
                // This also frees runtime + shared-resident param buffers, partial
                // and prefetched streaming buffers, and the graph-cut cache buffer.
                diffusion_model->release_all_gpu_param_residency();
            } else {
                // Direct-GPU params have no host-backed re-offload path; their
                // params buffer is released below. Still drop the old graph-cut
                // streaming/cache buffers before loading the replacement weights.
                diffusion_model->free_streaming_scratch_buffers();
                diffusion_model->free_cache_ctx_and_buffer();
            }
        }

        // free_params_buffer() nulls the tensor data/buffer pointers so the re-alloc
        // below doesn't trip the "already allocated" fast-path. For an mmap'd DiT
        // the weight data lives in a MmapTensorStore (the boot variant in
        // `mmap_tensor_store`, a prior swap in `dit_swap_mmap_store`), not the
        // params buffer.
        diffusion_model->free_params_buffer();
        if (free_outgoing_ltxav) {
            // The old DiT's graph scratch has returned to the ggml VMM pool; make
            // it real headroom before the incoming variant allocates. cuDNN's
            // attention/conv execution plans and async-mempool pages sit outside
            // that pool, so reset them too. The serialized worker guarantees no
            // render is in flight; both calls rebuild lazily for the new variant.
            ggml_backend_cuda_trim_pools(backend_for(SDBackendModule::DIFFUSION));
            ggml_backend_cuda_release_cudnn_plans();
            LOG_INFO("LTXAV_SWAP_FREE_OUTGOING: released outgoing DiT GPU residency "
                     "(runtime + shared-resident + streaming/prefetch + graph-cut cache), "
                     "trimmed DIFFUSION pool, and reset cuDNN plans before variant swap");
        }

        // 2. Open the new gguf with the same prefix the runner's tensor keys use.
        ModelLoader swap_loader;
        if (!swap_loader.init_from_file(new_diffusion_model_path, "model.diffusion_model.")) {
            LOG_ERROR("swap_diffusion_model: init loader from '%s' failed", new_diffusion_model_path.c_str());
            return false;
        }

        // 3. Collect the runner's param tensors (keys are "model.diffusion_model.*").
        std::map<std::string, ggml_tensor*> dit_tensors;
        diffusion_model->get_param_tensors(dit_tensors);
        if (dit_tensors.empty()) {
            LOG_ERROR("swap_diffusion_model: runner exposed no param tensors");
            return false;
        }

        // 4. Mirror the boot load ORDER so the swapped DiT has the SAME RAM
        //    characteristics as the boot one: when mmap is on, map the new file
        //    first (points the tensors at reclaimable file-backed pages — this is
        //    what preserves the --mmap host-RAM win across swaps; without it the
        //    swapped DiT would sit in ~5.6 GB of non-reclaimable anon RAM), THEN
        //    alloc the params buffer for any non-mappable remainder, THEN load
        //    that remainder. Drop the prior swap's mapping first so we never hold
        //    two swapped DiTs mapped at once. mmap_tensors skips any tensor whose
        //    dtype/shape differs from the runner graph (built from the boot
        //    gguf); load_tensors(use_mmap=true) then fills exactly those, so a
        //    base/edit recipe mismatch degrades to a partial anon load rather
        //    than silent garbage.
        dit_swap_mmap_store.clear();
        if (dit_swap_enable_mmap) {
            dit_swap_mmap_store =
                swap_loader.mmap_tensors(dit_tensors, /*ignore_tensors=*/{}, dit_swap_writable_mmap);
        }
        if (!diffusion_model->alloc_params_buffer()) {
            LOG_ERROR("swap_diffusion_model: alloc params buffer failed");
            return false;
        }
        bool ok = swap_loader.load_tensors(dit_tensors,
                                           /*ignore_tensors=*/{},
                                           n_threads,
                                           /*use_mmap=*/dit_swap_enable_mmap);
        if (!ok) {
            LOG_ERROR("swap_diffusion_model: load tensors from '%s' failed", new_diffusion_model_path.c_str());
            return false;
        }

        // 5. Re-register the INCOMING gguf's nvfp4 per-tensor weight globals.
        //    Boot registers these ONLY for diffusion_model_path / high_noise_diffusion_model_path
        //    (see the dit_legs block at load). Every selectable DiT — "base", "edit" and each
        //    --diffusion-model-variants entry — is hot-swapped through here instead
        //    (build_model_variants, examples/server/runtime.cpp), so without this an UNFOLDED
        //    import swapped in as a variant kept the BOOT model's map: nvfp4_weight_global_for()
        //    missed every tensor, defaulted w_global to 1.0, and the weights ran ~2688x too large
        //    -> pure white frames, silently, with no warning.
        //    CLEAR FIRST: g_wglobal is process-global, so an unfolded -> folded swap must not
        //    leave the outgoing globals registered (a folded gguf folds its global into the block
        //    scale and would be scaled twice). A folded gguf exports no .wglobal -> empty map ->
        //    nothing re-registered -> w_global stays 1.0 = byte-identical legacy behaviour.
        {
            ggml_cuda_nvfp4_clear_weight_globals();
            std::map<std::string, float> wglobals;
            load_nvfp4_weight_globals(new_diffusion_model_path, wglobals);
            const std::string pfx = "model.diffusion_model.";
            size_t n_reg = 0;
            for (auto& kv : dit_tensors) {
                if (wglobals.empty()) {
                    break;
                }
                const std::string& full = kv.first;
                if (full.compare(0, pfx.size(), pfx) != 0 || kv.second == nullptr) {
                    continue;
                }
                const std::string bare = full.substr(pfx.size());
                auto it = wglobals.find(bare + ".wglobal");
                if (it != wglobals.end()) {
                    // Same belt-and-suspenders keying as the boot path: the FP4 GEMM looks up
                    // src0->name, which may be the ggml (truncated) name, the prefixed key, or
                    // the bare stem depending on the offload/mmap path. Extra keys are harmless.
                    ggml_cuda_nvfp4_register_weight_global(kv.second->name, it->second);
                    ggml_cuda_nvfp4_register_weight_global(full.c_str(), it->second);
                    ggml_cuda_nvfp4_register_weight_global(bare.c_str(), it->second);
                    ++n_reg;
                }
            }
            LOG_INFO("swap_diffusion_model: nvfp4 weight globals re-registered %zu/%zu for '%s'%s",
                     n_reg, wglobals.size(), new_diffusion_model_path.c_str(),
                     wglobals.empty() ? " (folded gguf: none present, w_global=1.0)" : "");
        }

        LOG_INFO("swap_diffusion_model: loaded DiT from '%s' (mmap=%d), taking %.2fs",
                 new_diffusion_model_path.c_str(),
                 (int)dit_swap_enable_mmap,
                 (ggml_time_ms() - t0) * 1.0f / 1000);
        return true;
    }

    bool is_using_v_parameterization_for_sd2(bool is_inpaint = false) {
        sd::Tensor<float> x_t   = sd::full<float>({8, 8, 4, 1}, 0.5f);
        sd::Tensor<float> c     = sd::full<float>({1024, 2, 1, 1}, 0.5f);
        sd::Tensor<float> steps = sd::full<float>({1}, 999.0f);
        sd::Tensor<float> concat;
        if (is_inpaint) {
            concat = sd::zeros<float>({8, 8, 5, 1});
        } else if (sd_version_is_unet_edit(version)) {
            // ip2p conv_in takes 8 ch = [x(4), img_latent(4)]. Without the image half this
            // probe would feed a 4-ch x into an 8-ch conv_in and assert. Pass --prediction
            // explicitly to skip the probe entirely.
            concat = sd::zeros<float>({8, 8, 4, 1});
        }

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> out;
        DiffusionParams diffusion_params;
        diffusion_params.x         = &x_t;
        diffusion_params.timesteps = &steps;
        diffusion_params.context   = &c;
        diffusion_params.extra     = UNetDiffusionExtra{};
        if (!concat.empty()) {
            diffusion_params.c_concat = &concat;
        }
        auto out_opt = diffusion_model->compute(n_threads, diffusion_params);
        GGML_ASSERT(!out_opt.empty());
        out = std::move(out_opt);
        diffusion_model->free_compute_buffer();

        double result = static_cast<double>((out - x_t).mean());
        int64_t t1    = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    std::shared_ptr<LoraModel> load_lora_model_from_file(const std::string& lora_id,
                                                         float multiplier,
                                                         SDBackendModule module,
                                                         LoraModel::filter_t lora_tensor_filter = nullptr) {
        std::string lora_path             = lora_id;
        static std::string high_noise_tag = "|high_noise|";
        bool is_high_noise                = false;
        if (starts_with(lora_path, high_noise_tag)) {
            lora_path     = lora_path.substr(high_noise_tag.size());
            is_high_noise = true;
            LOG_DEBUG("high noise lora: %s", lora_path.c_str());
        }
        if (!ensure_backend_pair(module)) {
            return nullptr;
        }
        auto lora = std::make_shared<LoraModel>(lora_id,
                                                backend_for(module),
                                                backend_for(module),
                                                lora_path,
                                                is_high_noise ? "model.high_noise_" : "",
                                                version);
        if (!lora->load_from_file(n_threads, lora_tensor_filter)) {
            LOG_WARN("load lora tensors from %s failed", lora_path.c_str());
            return nullptr;
        }

        lora->multiplier = multiplier;
        return lora;
    }

    void apply_loras_immediately(const std::unordered_map<std::string, float>& lora_state) {
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;
            lora_state_diff[lora_name] += multiplier;
        }
        for (auto& kv : curr_lora_state) {
            const std::string& lora_name = kv.first;
            float curr_multiplier        = kv.second;
            lora_state_diff[lora_name] -= curr_multiplier;
        }

        if (lora_state_diff.empty()) {
            return;
        }

        LOG_INFO("apply lora immediately");

        size_t rm = lora_state_diff.size() - lora_state.size();
        if (rm != 0) {
            LOG_INFO("attempting to apply %lu LoRAs (removing %lu applied LoRAs)", lora_state.size(), rm);
        } else {
            LOG_INFO("attempting to apply %lu LoRAs", lora_state.size());
        }

        for (auto& kv : lora_state_diff) {
            int64_t t0 = ggml_time_ms();

            auto lora = load_lora_model_from_file(kv.first, kv.second, SDBackendModule::DIFFUSION);
            if (!lora || lora->lora_tensors.empty()) {
                continue;
            }
            lora->apply(tensors, version, n_threads);
            lora->free_params_buffer();

            int64_t t1 = ggml_time_ms();

            LOG_INFO("lora '%s' applied, taking %.2fs", kv.first.c_str(), (t1 - t0) * 1.0f / 1000);
        }

        curr_lora_state = lora_state;
    }

    void apply_loras_at_runtime(const std::unordered_map<std::string, float>& lora_state) {
        cond_stage_lora_models.clear();
        diffusion_lora_models.clear();
        first_stage_lora_models.clear();
        if (cond_stage_model) {
            cond_stage_model->set_weight_adapter(nullptr);
        }
        if (diffusion_model) {
            diffusion_model->set_weight_adapter(nullptr);
        }
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->set_weight_adapter(nullptr);
        }
        if (first_stage_model) {
            first_stage_model->set_weight_adapter(nullptr);
        }
        if (lora_state.empty()) {
            return;
        }
        LOG_INFO("apply lora at runtime");
        if (cond_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : cond_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            cond_stage_lora_models  = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_cond_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_id = kv.first;
                float multiplier           = kv.second;

                auto lora = load_lora_model_from_file(lora_id, multiplier, SDBackendModule::TE, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    cond_stage_lora_models.push_back(lora);
                }
            }
            // Only attach the adapter when there are LoRAs targeting the cond_stage model.
            // An empty MultiLoraAdapter still routes every linear/conv through
            // forward_with_lora() instead of the direct kernel path — slower for no benefit.
            if (!cond_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(cond_stage_lora_models);
                cond_stage_model->set_weight_adapter(multi_lora_adapter);
            }
        }
        if (diffusion_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : diffusion_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            diffusion_lora_models   = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_diffusion_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, SDBackendModule::DIFFUSION, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    diffusion_lora_models.push_back(lora);
                }
            }
            if (!diffusion_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(diffusion_lora_models);
                diffusion_model->set_weight_adapter(multi_lora_adapter);
                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->set_weight_adapter(multi_lora_adapter);
                }
            }
        }

        if (first_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : first_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            first_stage_lora_models = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_first_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, SDBackendModule::VAE, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    first_stage_lora_models.push_back(lora);
                }
            }
            if (!first_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(first_stage_lora_models);
                first_stage_model->set_weight_adapter(multi_lora_adapter);
            }
        }
    }

    void lora_stat() {
        if (!cond_stage_lora_models.empty()) {
            LOG_INFO("cond_stage_lora_models:");
            for (auto& lora_model : cond_stage_lora_models) {
                lora_model->stat();
            }
        }

        if (!diffusion_lora_models.empty()) {
            LOG_INFO("diffusion_lora_models:");
            for (auto& lora_model : diffusion_lora_models) {
                lora_model->stat();
            }
        }

        if (!first_stage_lora_models.empty()) {
            LOG_INFO("first_stage_lora_models:");
            for (auto& lora_model : first_stage_lora_models) {
                lora_model->stat();
            }
        }
    }

    void apply_loras(const sd_lora_t* loras, uint32_t lora_count) {
        std::unordered_map<std::string, float> lora_f2m;
        for (uint32_t i = 0; i < lora_count; i++) {
            std::string lora_id = SAFE_STR(loras[i].path);
            if (loras[i].is_high_noise) {
                lora_id = "|high_noise|" + lora_id;
            }
            lora_f2m[lora_id] = loras[i].multiplier;
            LOG_DEBUG("lora %s:%.2f", lora_id.c_str(), loras[i].multiplier);
        }
        int64_t t0 = ggml_time_ms();
        if (apply_lora_immediately) {
            apply_loras_immediately(lora_f2m);
        } else {
            apply_loras_at_runtime(lora_f2m);
        }
        int64_t t1 = ggml_time_ms();
        if (!lora_f2m.empty()) {
            LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        }
    }

    void reset_generation_extensions() {
        for (auto& extension : generation_extensions) {
            extension->reset_runtime_condition();
        }
    }

    void prepare_generation_extensions(const sd_pm_params_t& pm_params,
                                       ConditionerParams& condition_params,
                                       int total_steps) {
        reset_generation_extensions();
        GenerationExtensionConditionContext ctx{
            cond_stage_model.get(),
            condition_params,
            pm_params,
            tensors,
            version,
            n_threads,
            total_steps,
            free_params_immediately,
        };

        for (auto& extension : generation_extensions) {
            extension->prepare_condition(ctx);
        }
    }

    sd::Tensor<float> get_clip_vision_output(const sd::Tensor<float>& image,
                                             bool return_pooled   = true,
                                             int clip_skip        = -1,
                                             bool zero_out_masked = false) {
        sd::Tensor<float> output;
        if (zero_out_masked) {
            if (return_pooled) {
                output = sd::zeros<float>({clip_vision->vision_model.projection_dim});
            } else {
                output = sd::zeros<float>({clip_vision->vision_model.hidden_size, 257});
            }
        } else {
            auto pixel_values = clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);
            auto output_opt   = clip_vision->compute(n_threads, pixel_values, return_pooled, clip_skip);
            if (output_opt.empty()) {
                LOG_ERROR("clip_vision compute failed");
                return {};
            }
            output = std::move(output_opt);
        }
        return output;
    }

    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask) {
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B" || sd_version_is_longcat_avatar(version) ||
            (version == VERSION_WAN2_2_I2V && !denoise_mask.empty())) {
            // Per-frame timesteps. The avatar sets EVERY fixed-cond latent frame's
            // timestep to 0 (clean) so its adaLN/attention conditioning is treated as
            // a fully-denoised anchor. For ai2v that is the single ref-image frame
            // (`timestep[:, :1] = 0`); for video-continuation (generate_avc) it is ALL
            // num_cond_latents frames — ref(1) + cond_tail(N) — matching
            // `timestep[:, :num_cond_latents] = 0` (pipeline L1405-1406). The fixed-cond
            // frames are exactly those the denoise_mask pins (mask==0), so drive the
            // zeroing PER-FRAME off the mask rather than only frame 0; otherwise the
            // cond_tail frames' modulation (the K/V the noise frames attend) is computed
            // at the noisy `t` and contaminates the generated frames (watercolour melt).
            int64_t T          = init_latent.shape()[2];
            auto new_timesteps = std::vector<float>(static_cast<size_t>(T), timesteps[0]);

            // upstream #1604 guard: only per-frame-zero when the mask actually has a
            // matching frame dim (otherwise leave the uniform timesteps untouched).
            if (!denoise_mask.empty() && denoise_mask.dim() >= 4 && denoise_mask.shape()[2] == T) {
                for (int64_t f = 0; f < T; ++f) {
                    float value = denoise_mask.dim() == 5 ? denoise_mask.index(0, 0, f, 0, 0)
                                                          : denoise_mask.index(0, 0, f, 0);
                    if (value == 0.f) {
                        new_timesteps[static_cast<size_t>(f)] = 0.f;
                    }
                }
            }
            return new_timesteps;
        } else {
            return timesteps;
        }
    }

    std::vector<float> process_ltxav_video_timesteps(const std::vector<float>& timesteps,
                                                     const sd::Tensor<float>& init_latent,
                                                     const sd::Tensor<float>& denoise_mask) {
        if (timesteps.empty() || denoise_mask.empty() || init_latent.dim() < 4 || denoise_mask.dim() < 4) {
            return timesteps;
        }

        int64_t width  = init_latent.shape()[0];
        int64_t height = init_latent.shape()[1];
        int64_t frames = init_latent.shape()[2];
        if (denoise_mask.shape()[0] != width ||
            denoise_mask.shape()[1] != height ||
            denoise_mask.shape()[2] != frames ||
            denoise_mask.shape()[3] < 1) {
            LOG_WARN("unexpected LTXAV denoise mask shape for timestep processing");
            return timesteps;
        }

        std::vector<float> video_timesteps(static_cast<size_t>(width * height * frames));
        size_t idx = 0;
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    float mask             = denoise_mask.dim() == 5 ? denoise_mask.index(w, h, t, 0, 0)
                                                                     : denoise_mask.index(w, h, t, 0);
                    video_timesteps[idx++] = mask * timesteps[0];
                }
            }
        }
        return video_timesteps;
    }

    // LTX-AV's audio stream is packed after the video channels.  For LipDub it
    // contains `[noisy target | clean reference]`: the reference block is held
    // by the denoise mask and, just as importantly, must receive timestep zero.
    // Supplying a uniform noisy timestep makes the clean block look like another
    // generation target to adaLN/cross-attention, which defeats audio-reference
    // conditioning even though its waveform latent is present.
    std::vector<float> process_ltxav_audio_timesteps(const std::vector<float>& timesteps,
                                                     const sd::Tensor<float>& init_latent,
                                                     const sd::Tensor<float>& denoise_mask,
                                                     int audio_length,
                                                     bool audio_fixed) {
        if (timesteps.empty() || audio_length <= 0) {
            return timesteps;
        }
        if (audio_fixed) {
            // Preserve the original direct-drive contract: a single zero is
            // broadcast across the audio stream.  Per-token timesteps are
            // needed only for the experimental target/reference layout below;
            // using them here changed the established LipDub path.
            return {0.f};
        }
        if (denoise_mask.empty() || init_latent.dim() < 4 || denoise_mask.dim() < 4 ||
            denoise_mask.shape() != init_latent.shape()) {
            return timesteps;
        }

        constexpr int64_t kAudioFreqBins = 16;
        constexpr int64_t kAudioChannels = 8;
        const int64_t audio_values       = kAudioFreqBins * kAudioChannels * audio_length;
        const int64_t spatial_size       = init_latent.shape()[0] * init_latent.shape()[1] * init_latent.shape()[2];
        if (spatial_size <= 0 || audio_values <= 0) {
            return timesteps;
        }
        const int64_t extra_channels = (audio_values + spatial_size - 1) / spatial_size;
        if (init_latent.shape()[3] < extra_channels) {
            LOG_WARN("unexpected LTXAV packed latent shape for audio timestep processing");
            return timesteps;
        }

        // pack_ltxav_audio_and_video_denoise_mask writes the compact audio mask
        // at the start of the trailing extra-channel block.  One value per time
        // index is sufficient: all 16x8 components of an audio latent token use
        // the same mask value.
        const int64_t audio_offset = spatial_size * (init_latent.shape()[3] - extra_channels);
        if (audio_offset + audio_values > denoise_mask.numel()) {
            LOG_WARN("truncated LTXAV packed audio mask for timestep processing");
            return timesteps;
        }
        std::vector<float> audio_timesteps(static_cast<size_t>(audio_length), timesteps[0]);
        const float* mask = denoise_mask.data() + audio_offset;
        for (int t = 0; t < audio_length; ++t) {
            audio_timesteps[static_cast<size_t>(t)] = mask[static_cast<int64_t>(t) * kAudioFreqBins] * timesteps[0];
        }
        return audio_timesteps;
    }

    void preview_image(int step,
                       const sd::Tensor<float>& latents,
                       enum SDVersion version,
                       preview_t preview_mode,
                       std::function<void(int, int, sd_image_t*, bool, void*)> step_callback,
                       void* step_callback_data,
                       bool is_noisy) {
        bool is_video = preview_latent_tensor_is_video(latents);
        uint32_t dim  = is_video ? static_cast<uint32_t>(latents.shape()[3]) : static_cast<uint32_t>(latents.shape()[2]);
        int channels  = get_latent_channel();
        auto _latents = channels != dim ? is_video ? sd::ops::slice(latents, 3, 0, channels)
                                                   : sd::ops::slice(latents, 2, 0, channels)
                                        : latents;
        if (preview_mode == PREVIEW_PROJ) {
            int patch_sz                     = 1;
            const float(*latent_rgb_proj)[3] = nullptr;
            float* latent_rgb_bias           = nullptr;

            if (channels == 128) {
                if (sd_version_uses_flux2_vae(version)) {
                    latent_rgb_proj = flux2_latent_rgb_proj;
                    latent_rgb_bias = flux2_latent_rgb_bias;
                    patch_sz        = 2;
                } else if (version == VERSION_LTXAV) {
                    latent_rgb_proj = ltxav_latent_rgb_proj;
                    latent_rgb_bias = ltxav_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 48) {
                if (sd_version_is_wan(version)) {
                    latent_rgb_proj = wan_22_latent_rgb_proj;
                    latent_rgb_bias = wan_22_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 16) {
                if (sd_version_is_sd3(version)) {
                    latent_rgb_proj = sd3_latent_rgb_proj;
                    latent_rgb_bias = sd3_latent_rgb_bias;
                } else if (sd_version_is_flux(version) || sd_version_is_z_image(version) || sd_version_is_longcat(version)) {
                    latent_rgb_proj = flux_latent_rgb_proj;
                    latent_rgb_bias = flux_latent_rgb_bias;
                } else if (sd_version_is_wan(version) || sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
                    latent_rgb_proj = wan_21_latent_rgb_proj;
                    latent_rgb_bias = wan_21_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 4) {
                if (sd_version_is_sdxl(version)) {
                    latent_rgb_proj = sdxl_latent_rgb_proj;
                    latent_rgb_bias = sdxl_latent_rgb_bias;
                } else if (sd_version_is_sd1(version) || sd_version_is_sd2(version)) {
                    latent_rgb_proj = sd_latent_rgb_proj;
                    latent_rgb_bias = sd_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels != 3) {
                LOG_WARN("No latent to RGB projection known for this model (dim = %d)", dim);
                return;
            }

            uint32_t frames     = is_video ? static_cast<uint32_t>(_latents.shape()[2]) : 1;
            uint32_t img_width  = static_cast<uint32_t>(_latents.shape()[0]) * patch_sz;
            uint32_t img_height = static_cast<uint32_t>(_latents.shape()[1]) * patch_sz;

            uint8_t* data = (uint8_t*)malloc(frames * img_width * img_height * 3 * sizeof(uint8_t));
            GGML_ASSERT(data != nullptr);
            preview_latent_video(data, _latents, latent_rgb_proj, latent_rgb_bias, patch_sz);
            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            GGML_ASSERT(images != nullptr);
            for (uint32_t i = 0; i < frames; i++) {
                images[i] = {img_width, img_height, 3, data + i * img_width * img_height * 3};
            }
            step_callback(step, frames, images, is_noisy, step_callback_data);
            free(data);
            free(images);
            return;
        }

        if (preview_mode == PREVIEW_VAE || preview_mode == PREVIEW_TAE) {
            sd::Tensor<float> vae_latents;
            sd::Tensor<float> decoded;
            if (preview_vae) {
                preview_vae->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
                vae_latents = preview_vae->diffusion_to_vae_latents(_latents);
                decoded     = preview_vae->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
            } else {
                first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
                vae_latents = first_stage_model->diffusion_to_vae_latents(_latents);
                decoded     = first_stage_model->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
            }
            if (decoded.empty()) {
                LOG_ERROR("preview decode failed at step %d", step);
                return;
            }

            is_video           = preview_latent_tensor_is_video(decoded);
            uint32_t frames    = is_video ? static_cast<uint32_t>(decoded.shape()[2]) : 1;
            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            GGML_ASSERT(images != nullptr);
            for (uint32_t i = 0; i < frames; ++i) {
                images[i] = tensor_to_sd_image(decoded, static_cast<int>(i));
            }

            step_callback(step, frames, images, is_noisy, step_callback_data);
            for (uint32_t i = 0; i < frames; ++i) {
                free(images[i].data);
            }
            free(images);
            return;
        }

        if (preview_mode != PREVIEW_NONE) {
            LOG_WARN("Unsupported preview mode: %d", static_cast<int>(preview_mode));
        }
    }

    std::vector<float> prepare_sample_timesteps(float sigma,
                                                int shifted_timestep) {
        float t = denoiser->sigma_to_t(sigma);
        if (shifted_timestep > 0) {
            float shifted_t_float = t * (float(shifted_timestep) / float(TIMESTEPS));
            int64_t shifted_t     = static_cast<int64_t>(roundf(shifted_t_float));
            shifted_t             = std::max((int64_t)0, std::min((int64_t)(TIMESTEPS - 1), shifted_t));
            LOG_DEBUG("shifting timestep from %.2f to %" PRId64 " (sigma: %.4f)", t, shifted_t, sigma);
            return std::vector<float>{(float)shifted_t};
        }
        if (sd_version_is_anima(version)) {
            return std::vector<float>{t / static_cast<float>(TIMESTEPS)};
        }
        if (version == VERSION_HIDREAM_O1) {
            return std::vector<float>{1.0f - (t / static_cast<float>(TIMESTEPS))};
        }
        if (sd_version_is_z_image(version) || sd_version_is_ideogram4(version)) {
            return std::vector<float>{1000.f - t};
        }
        return std::vector<float>{t};
    }

    void adjust_sample_step_scalings(int shifted_timestep,
                                     const std::vector<float>& timesteps_vec,
                                     float c_in,
                                     float* c_skip,
                                     float* c_out) {
        GGML_ASSERT(c_skip != nullptr);
        GGML_ASSERT(c_out != nullptr);
        if (shifted_timestep <= 0) {
            return;
        }

        int64_t shifted_t_idx              = static_cast<int64_t>(roundf(timesteps_vec[0]));
        float shifted_sigma                = denoiser->t_to_sigma((float)shifted_t_idx);
        std::vector<float> shifted_scaling = denoiser->get_scalings(shifted_sigma);
        float shifted_c_skip               = shifted_scaling[0];
        float shifted_c_out                = shifted_scaling[1];
        float shifted_c_in                 = shifted_scaling[2];

        *c_skip = shifted_c_skip * c_in / shifted_c_in;
        *c_out  = shifted_c_out;
    }

    struct SamplePreviewContext {
        sd_preview_cb_t callback = nullptr;
        void* data               = nullptr;
        preview_t mode           = PREVIEW_NONE;
    };

    SamplePreviewContext prepare_sample_preview_context() {
        return SamplePreviewContext{sd_get_preview_callback(),
                                    sd_get_preview_callback_data(),
                                    sd_get_preview_mode()};
    }

    void report_sample_progress(int step, size_t total_steps, int64_t t0) {
        int64_t t1 = ggml_time_us();
        if (step > 0 || step == -(int)total_steps) {
            int showstep = std::abs(step);
            pretty_progress(showstep, (int)total_steps, (t1 - t0) / 1000000.f / showstep);
        }
    }

    void compute_sample_controls(const sd::Tensor<float>& control_image,
                                 const sd::Tensor<float>& noised_input,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const SDCondition& condition,
                                 std::vector<sd::Tensor<float>>* controls) {
        GGML_ASSERT(controls != nullptr);
        controls->clear();
        if (control_image.empty() || control_net == nullptr) {
            return;
        }

        auto control_result = control_net->compute(n_threads,
                                                   noised_input,
                                                   control_image,
                                                   timesteps_tensor,
                                                   condition.c_crossattn,
                                                   condition.c_vector);
        if (!control_result.has_value()) {
            LOG_ERROR("controlnet compute failed");
            return;
        }

        *controls = std::move(*control_result);
    }

    sd::Tensor<float> sample(const std::shared_ptr<DiffusionModelRunner>& work_diffusion_model,
                             bool inverse_noise_scaling,
                             const sd::Tensor<float>& init_latent,
                             sd::Tensor<float> noise,
                             const SDCondition& cond,
                             const SDCondition& uncond,
                             const SDCondition& img_uncond,
                             const sd::Tensor<float>& control_image,
                             float control_strength,
                             const sd_guidance_params_t& guidance,
                             float eta,
                             int shifted_timestep,
                             sample_method_t method,
                             bool is_flow_denoiser,
                             const char* extra_sample_args,
                             const std::vector<float>& sigmas,
                             const std::vector<sd::Tensor<float>>& ref_latents,
                             bool increase_ref_index,
                             const sd::Tensor<float>& denoise_mask,
                             const sd::Tensor<float>& vace_context,
                             float vace_strength,
                             int audio_length,
                             float frame_rate,
                             const sd_cache_params_t* cache_params,
                             const sd::Tensor<float>& video_positions = {},
                             const sd::Tensor<float>& audio_positions = {},
                             bool ltxav_audio_fixed                    = false,
                             const sd::Tensor<float>& video_reference  = {},
                             bool return_denoised                       = false) {
        if (getenv("LONGCAT_VRAM_BREAKDOWN") != nullptr) {
            double dit = work_diffusion_model ? work_diffusion_model->gpu_footprint_bytes()/1048576.0 : 0.0;
            double vae = first_stage_model ? first_stage_model->gpu_footprint_bytes()/1048576.0 : 0.0;
            double avae = audio_vae_model ? audio_vae_model->gpu_footprint_bytes()/1048576.0 : 0.0;
            LOG_INFO("[VRAM-ATTR sample-entry] DiT_gpu=%.0f MB  VAE_gpu=%.0f MB  audioVAE_gpu=%.0f MB", dit, vae, avae);
        }
        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);
        float cfg_scale     = guidance.txt_cfg;
        float img_cfg_scale = guidance.img_cfg;
        float slg_scale     = guidance.slg.scale;
        bool slg_uncond     = sd::guidance::parse_skip_layer_guidance_uncond_arg(extra_sample_args);

        sd_sample::SampleCacheRuntime cache_runtime = sd_sample::init_sample_cache_runtime(version,
                                                                                           cache_params,
                                                                                           denoiser.get(),
                                                                                           sigmas);

        bool needs_uncond_denoised = method == EULER_CFG_PP_SAMPLE_METHOD || method == EULER_A_CFG_PP_SAMPLE_METHOD;
        // Spectrum cache is not supported for CFG++ samplers
        if (needs_uncond_denoised) {
            if (cache_runtime.spectrum_enabled) {
                LOG_WARN("Spectrum cache requested but not supported for CFG++ samplers");
                cache_runtime.spectrum_enabled = false;
            }
        }

        size_t steps       = sigmas.size() - 1;
        // SA3 is FP4 approximate attention.  Its upstream implementation recommends
        // a hybrid policy for models whose all-step FP4 path is not lossless: run a
        // precise attention backend at the first and final denoise steps, and SA3 in
        // the middle.  The CUDA dispatcher reads GGML_LTX_SA3 at each attention call,
        // so scope the process environment to this sample only.  This is deliberately
        // opt-in; the unset production default remains cuDNN.
        const char* sa3_policy_env = std::getenv("GGML_LTX_SA3_POLICY");
        const std::string sa3_policy = sa3_policy_env != nullptr ? sa3_policy_env : "";
        const bool sa3_step_policy = sd_version_is_ltxav(version) &&
                                     (sa3_policy == "first" || sa3_policy == "last" || sa3_policy == "middle");
        // The precise cuDNN step needs two full-length F16 buffers (converted Q
        // and output).  Tile only that step's independent query rows; the SA3
        // steps retain their one-shot graph and throughput.  The legacy global
        // LTX_ATTN_QTILE keeps precedence when explicitly configured.
        const char* precision_qtile_env = std::getenv("GGML_LTX_SA3_PRECISION_QTILE");
        const bool precision_qtile_enabled = sa3_step_policy && precision_qtile_env != nullptr &&
                                             atoll(precision_qtile_env) > 0 &&
                                             (std::getenv("LTX_ATTN_QTILE") == nullptr || atoll(std::getenv("LTX_ATTN_QTILE")) == 0);
        struct SA3EnvRestore {
            bool active = false;
            bool had_value = false;
            std::string value;
            ~SA3EnvRestore() {
                if (!active) return;
                if (had_value) setenv("GGML_LTX_SA3", value.c_str(), 1);
                else unsetenv("GGML_LTX_SA3");
            }
        } sa3_env_restore;
        struct PrecisionQTileEnvRestore {
            bool active = false;
            bool had_value = false;
            std::string value;
            ~PrecisionQTileEnvRestore() {
                if (!active) return;
                if (had_value) setenv("GGML_CUDNN_LTX_QTILE", value.c_str(), 1);
                else unsetenv("GGML_CUDNN_LTX_QTILE");
            }
        } precision_qtile_restore;
        if (sa3_step_policy) {
            if (const char* e = std::getenv("GGML_LTX_SA3")) {
                sa3_env_restore.had_value = true;
                sa3_env_restore.value = e;
            }
            sa3_env_restore.active = true;
            LOG_INFO("LTX SA3 policy=%s (%zu total sampler steps)", sa3_policy.c_str(), steps);
        }
        if (precision_qtile_enabled) {
            if (const char* e = std::getenv("GGML_CUDNN_LTX_QTILE")) {
                precision_qtile_restore.had_value = true;
                precision_qtile_restore.value = e;
            }
            precision_qtile_restore.active = true;
            LOG_INFO("LTX SA3 precision-step query tile=%s", precision_qtile_env);
        }
        bool has_skiplayer = (slg_scale != 0.0f || slg_uncond) && !skip_layers.empty();
        if (has_skiplayer && !sd_version_is_dit(version)) {
            has_skiplayer = false;
            LOG_WARN("SLG is incompatible with this model type");
        }
        sd::guidance::AdaptiveProjectedGuidanceParams apg_params = sd::guidance::parse_adaptive_projected_guidance_args(extra_sample_args);
        bool use_apg_guidance                                    = sd::guidance::is_adaptive_projected_guidance_enabled(apg_params);
        if (use_apg_guidance) {
            LOG_INFO("using Adaptive Projected Guidance (APG)");
        }
        sd::guidance::ClassifierFreeGuidance classifier_free_guidance(cfg_scale, img_cfg_scale);
        sd::guidance::AdaptiveProjectedGuidance adaptive_projected_guidance(cfg_scale, img_cfg_scale, apg_params);
        const sd::guidance::BaseGuidance& primary_guidance = use_apg_guidance
                                                                 ? static_cast<const sd::guidance::BaseGuidance&>(adaptive_projected_guidance)
                                                                 : static_cast<const sd::guidance::BaseGuidance&>(classifier_free_guidance);
        sd::guidance::SkipLayerGuidance skip_layer_guidance(has_skiplayer ? skip_layers : std::vector<int>(),
                                                            has_skiplayer ? slg_scale : 0.0f,
                                                            guidance.slg.layer_start,
                                                            guidance.slg.layer_end);

        if (version == VERSION_HIDREAM_O1 && !noise.empty()) {
            noise *= eta;
        }

        int64_t t0                   = ggml_time_us();
        sd::Tensor<float> x_t        = !noise.empty()
                                           ? denoiser->noise_scaling(sigmas[0], noise, init_latent)
                                           : init_latent;
        sd::Tensor<float> denoised   = x_t;
        SamplePreviewContext preview = prepare_sample_preview_context();

        // Official LTX motion-continuity for hard-conditioning (continuation overlap +
        // i2v) frames: instead of feeding them perfectly clean (sigma 0), inject
        // timestep-dependent noise each step (diffusers add_noise_to_image_conditioning_latents):
        //   cond = init_latent + image_cond_noise_scale * t^2 * eps,  t = sigma in [0,1].
        // A frozen clean conditioning wall creates a velocity discontinuity at the seam
        // (the continuation segment jumps then catches up); matched-noise conditioning lets
        // motion flow across. Gated by LTXAV_COND_NOISE_SCALE (default 0 = legacy frozen).
        float cond_noise_scale = 0.0f;
        if (const char* e = std::getenv("LTXAV_COND_NOISE_SCALE")) {
            cond_noise_scale = std::max(0.0f, (float)atof(e));
        }
        std::shared_ptr<RNG> cond_noise_rng;
        if (cond_noise_scale > 0.0f) {
            cond_noise_rng = std::make_shared<MT19937RNG>(0x10ec0de5ULL);
            LOG_INFO("LTXAV cond-noise: image_cond_noise_scale=%.4f (timestep-dependent noise on hard-conditioning frames)",
                     cond_noise_scale);
        }

        // LTXAV A2V (audio-to-video) modality guidance — drives lip-sync on the distilled model.
        // The DiT is an audio-video model but the video won't track a *frozen* driving-audio
        // latent on its own (the strong reference dominates). LTX-2's MultiModalGuider fixes this
        // with a modality term: pred = cond + (scale-1)*(cond - mod), where the "mod" forward has
        // the audio<->video cross-attention severed (the video as if the audio were absent).
        // Extrapolating away from it amplifies how strongly the mouth follows the audio. It
        // perturbs the AUDIO modality (not text), so unlike text CFG it needs no negative prompt
        // and survives distillation. Costs a 2nd forward/step. Off by default (1.0).
        float a2v_guidance_scale = 1.0f;
        if (const char* e = std::getenv("LTXAV_A2V_GUIDANCE")) {
            a2v_guidance_scale = (float)atof(e);
        }
        // A2V schedule: ramp the guidance across the denoise trajectory (the official guider
        // varies its params by sigma; a constant scale over-drives the LOW-sigma detail steps
        // and smears the lips). LTXAV_A2V_RAMP_END = the fraction of (scale-1) kept at the LAST
        // (lowest-sigma) step: 1.0 = constant [default]; 0.0 = ramp to OFF so late steps refine
        // a clean mouth; 0.5 = ramp to half. Interpolated linearly in sigma between sigmas[0]
        // (full strength) and ~0 (ramp_end strength).
        float a2v_ramp_end = 1.0f;
        if (const char* e = std::getenv("LTXAV_A2V_RAMP_END")) {
            a2v_ramp_end = std::clamp((float)atof(e), 0.0f, 1.0f);
        }
        if (a2v_guidance_scale != 1.0f && sd_version_is_ltxav(version)) {
            LOG_INFO("LTXAV A2V (audio->video) modality guidance scale=%.2f ramp_end=%.2f (lip-sync; +1 audio-skipped forward/step)",
                     a2v_guidance_scale, a2v_ramp_end);
        }

        // NAG (Normalized Attention Guidance) — attention-space negative guidance on the video text
        // cross-attn (Denoise-AI workflow: scale 14, alpha 0.35, tau 2.5, applied on the high-noise
        // preview sub-stage only). Read per-render from env (mirrors the A2V knob above so the warm
        // resident worker never bleeds a prior render's value). scale == 0 => OFF (default).
        //   LTXAV_NAG_SCALE / LTXAV_NAG_ALPHA / LTXAV_NAG_TAU / LTXAV_NAG_UNTIL_SIGMA
        // The sigma gate (apply NAG only while sigma >= until_sigma, default 0.9) faithfully limits
        // NAG to the early/high-noise steps (the workflow's S1-only application) AND lets us ablate.
        float nag_scale       = 0.0f;
        float nag_alpha       = 0.35f;
        float nag_tau         = 2.5f;
        float nag_until_sigma = 0.9f;
        if (const char* e = std::getenv("LTXAV_NAG_SCALE"))       { nag_scale = (float)atof(e); }
        if (const char* e = std::getenv("LTXAV_NAG_ALPHA"))       { nag_alpha = (float)atof(e); }
        if (const char* e = std::getenv("LTXAV_NAG_TAU"))         { nag_tau = (float)atof(e); }
        if (const char* e = std::getenv("LTXAV_NAG_UNTIL_SIGMA")) { nag_until_sigma = (float)atof(e); }
        const bool nag_enabled = (nag_scale != 0.0f) && sd_version_is_ltxav(version);
        // NAG supplies negative guidance in attention space, so it REPLACES output-level CFG. When
        // enabled at cfg<=1 we skip the separate uncond forward (its cfg-1 combine is a no-op) and
        // instead feed the negative context INTO the single cond forward. The negative embedding is
        // materialized because generate_video forces use_uncond when NAG is on (see resolve()).
        const bool nag_owns_guidance = nag_enabled && cfg_scale <= 1.0f;
        if (nag_enabled) {
            LOG_INFO("LTXAV NAG (normalized attention guidance) scale=%.2f alpha=%.2f tau=%.2f until_sigma=%.3f (video text cross-attn; +1 cross-attn/step while gated on)",
                     nag_scale, nag_alpha, nag_tau, nag_until_sigma);
        }

        auto denoise = [&](const sd::Tensor<float>& x, float sigma, int step) -> sd::guidance::GuiderOutput {
            // Cooperative cancel: client disconnected mid-render. Bail before launching
            // this step's DiT compute. An empty pred makes sample_k_diffusion yield an
            // empty latent, which the caller treats as failure and frees the compute
            // buffers (weights stay resident — NOT unload).
            if (sd_is_cancel_requested()) {
                LOG_INFO("render cancelled (client disconnect) at sampler step %d/%zu", step, (size_t)steps);
                return {};
            }
            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
            }

            if (sa3_step_policy) {
                const int sample_step = std::abs(step);
                const bool use_sa3 = sa3_policy == "first" ? sample_step > 1
                                     : sa3_policy == "last" ? sample_step < (int)steps
                                                            : sample_step > 1 && sample_step < (int)steps;
                setenv("GGML_LTX_SA3", use_sa3 ? "1" : "0", 1);
                if (precision_qtile_enabled) {
                    if (use_sa3) unsetenv("GGML_CUDNN_LTX_QTILE");
                    else setenv("GGML_CUDNN_LTX_QTILE", precision_qtile_env, 1);
                }
                LOG_DEBUG("LTX SA3 policy=%s: step %d/%zu -> %s", sa3_policy.c_str(), sample_step, steps,
                          use_sa3 ? "SA3" : "cuDNN");
            }

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            std::vector<float> base_timesteps_vec = prepare_sample_timesteps(sigma, shifted_timestep);
            std::vector<float> timesteps_vec      = base_timesteps_vec;
            sd::Tensor<float> audio_timesteps_tensor;
            if (sd_version_is_ltxav(version) && !denoise_mask.empty()) {
                timesteps_vec = process_ltxav_video_timesteps(base_timesteps_vec, init_latent, denoise_mask);
                // Per-token audio timesteps mirror the audio denoise mask.  This
                // matters for LipDub's appended clean reference block, while
                // retaining the legacy all-fixed behaviour for driven A2V.
                std::vector<float> audio_ts = process_ltxav_audio_timesteps(base_timesteps_vec,
                                                                             init_latent,
                                                                             denoise_mask,
                                                                             audio_length,
                                                                             ltxav_audio_fixed);
                // GGML's audio adaLN treats dim-0 as the scalar/timestep
                // component and dim-1 as the audio-token axis.  A one-element
                // vector remains the legacy broadcast scalar; a mixed
                // LipDub target/reference timeline must therefore be [1, T],
                // not a flat [T] tensor.
                audio_timesteps_tensor = audio_ts.size() > 1
                                             ? sd::Tensor<float>({1, static_cast<int64_t>(audio_ts.size())}, audio_ts)
                                             : sd::Tensor<float>({1}, audio_ts);
            } else {
                timesteps_vec = process_timesteps(timesteps_vec, init_latent, denoise_mask);
            }
            const std::vector<float>& scaling_timesteps_vec = (sd_version_is_ltxav(version) && !denoise_mask.empty())
                                                                  ? base_timesteps_vec
                                                                  : timesteps_vec;
            adjust_sample_step_scalings(shifted_timestep, scaling_timesteps_vec, c_in, &c_skip, &c_out);

            sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
            sd::Tensor<float> guidance_tensor({1}, std::vector<float>{guidance.distilled_guidance});
            sd::Tensor<float> noised_input = x * c_in;
            if (!denoise_mask.empty() && (version == VERSION_WAN2_2_TI2V || version == VERSION_WAN2_2_I2V || sd_version_is_ltxav(version) || sd_version_is_longcat_avatar(version))) {
                // ai2v: the first num_cond_latents temporal latent frames ARE the
                // VAE-encoded reference image and must be held fixed (mask=0) through
                // the whole denoise loop; only the generated frames (mask=1) evolve.
                // (pipeline keeps latents[:,:,:1] = cond_latents every step.)
                if (cond_noise_scale > 0.0f && cond_noise_rng && sd_version_is_ltxav(version)) {
                    sd::Tensor<float> cond_eps    = sd::randn_like<float>(init_latent, cond_noise_rng);
                    float             s           = cond_noise_scale * sigma * sigma;
                    sd::Tensor<float> cond_latent = init_latent + cond_eps * s;
                    noised_input = noised_input * denoise_mask + cond_latent * (1.0f - denoise_mask);
                } else {
                    noised_input = noised_input * denoise_mask + init_latent * (1.0f - denoise_mask);
                }
            }

            if (cache_runtime.spectrum_enabled && cache_runtime.spectrum.should_predict()) {
                cache_runtime.spectrum.predict(&denoised);
                if (!denoise_mask.empty()) {
                    denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
                }
                if (sd_should_preview_denoised() && preview.callback != nullptr) {
                    preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
                }
                report_sample_progress(step, steps, t0);
                sd::guidance::GuiderOutput output;
                output.pred = denoised;
                return output;
            }

            if (sd_should_preview_noisy() && preview.callback != nullptr) {
                preview_image(step, noised_input, version, preview.mode, preview.callback, preview.data, true);
            }

            sd::Tensor<float> cond_out;
            sd::Tensor<float> uncond_out;
            sd::Tensor<float> img_uncond_out;
            sd_sample::SampleStepCacheDispatcher step_cache(cache_runtime, step, sigma);
            std::vector<sd::Tensor<float>> controls;
            DiffusionParams diffusion_params;
            diffusion_params.x                  = &noised_input;
            diffusion_params.timesteps          = &timesteps_tensor;
            diffusion_params.increase_ref_index = increase_ref_index;
            sd::guidance::GuidanceInput step_guidance_input;
            step_guidance_input.step          = step;
            step_guidance_input.schedule_size = sigmas.size();
            bool is_skiplayer_step            = skip_layer_guidance.is_enabled_for_step(step_guidance_input);

            compute_sample_controls(control_image,
                                    noised_input,
                                    timesteps_tensor,
                                    cond,
                                    &controls);

            static const std::vector<sd::Tensor<float>> empty_ref_latents;
            bool uncond_without_ref_latents = !img_uncond.empty() &&
                                              !ref_latents.empty() &&
                                              sd_version_supports_ref_latent_img_cfg(version);

            auto run_condition = [&](const SDCondition& condition,
                                     const sd::Tensor<float>* c_concat_override                 = nullptr,
                                     const std::vector<int>* local_skip_layers                  = nullptr,
                                     const std::vector<sd::Tensor<float>>* ref_latents_override = nullptr,
                                     bool skip_a2v_pass                                         = false,
                                     bool nag_pass                                              = false) -> sd::Tensor<float> {
                diffusion_params.context     = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
                diffusion_params.c_concat    = c_concat_override != nullptr ? c_concat_override : (condition.c_concat.empty() ? nullptr : &condition.c_concat);
                diffusion_params.y           = condition.c_vector.empty() ? nullptr : &condition.c_vector;
                diffusion_params.ref_latents = ref_latents_override != nullptr ? ref_latents_override : (condition.c_ref_images.empty() ? &ref_latents : &condition.c_ref_images);

                if (sd_version_is_unet(version)) {
                    diffusion_params.extra = UNetDiffusionExtra{-1, &controls, control_strength};
                } else if (sd_version_is_sd3(version)) {
                    diffusion_params.extra = SkipLayerDiffusionExtra{local_skip_layers};
                } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) || sd_version_is_longcat(version)) {
                    diffusion_params.extra = FluxDiffusionExtra{&guidance_tensor,
                                                                local_skip_layers};
                } else if (sd_version_is_anima(version)) {
                    diffusion_params.extra = AnimaDiffusionExtra{condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids,
                                                                 condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights};
                } else if (sd_version_is_wan(version)) {
                    diffusion_params.extra = WanDiffusionExtra{vace_context.empty() ? nullptr : &vace_context,
                                                               vace_strength};
                } else if (version == VERSION_HIDREAM_O1) {
                    diffusion_params.extra = HiDreamO1DiffusionExtra{
                        condition.c_input_ids.empty() ? nullptr : &condition.c_input_ids,
                        condition.c_position_ids.empty() ? nullptr : &condition.c_position_ids,
                        condition.c_token_types.empty() ? nullptr : &condition.c_token_types,
                        condition.c_vinput_mask.empty() ? nullptr : &condition.c_vinput_mask,
                        condition.c_image_embeds.empty() ? nullptr : &condition.c_image_embeds};
                } else if (sd_version_is_ltxav(version)) {
                    LTXAVDiffusionExtra ltx_extra{
                        nullptr,
                        audio_timesteps_tensor.empty() ? nullptr : &audio_timesteps_tensor,
                        audio_length,
                        frame_rate,
                        video_positions.empty() ? nullptr : &video_positions,
                        audio_positions.empty() ? nullptr : &audio_positions,
                        skip_a2v_pass,
                        video_reference.empty() ? nullptr : &video_reference};
                    // NAG: only on the primary cond forward (nag_pass), only while sigma is high
                    // enough (the workflow applies NAG to the high-noise sub-stage), and only when a
                    // negative context is actually available. Otherwise NAG stays off (legacy).
                    if (nag_pass && nag_enabled && sigma >= nag_until_sigma && !uncond.c_crossattn.empty()) {
                        ltx_extra.nag_context = &uncond.c_crossattn;
                        ltx_extra.nag_scale   = nag_scale;
                        ltx_extra.nag_alpha   = nag_alpha;
                        ltx_extra.nag_tau     = nag_tau;
                    }
                    diffusion_params.extra = ltx_extra;
                } else if (sd_version_is_longcat_avatar(version)) {
                    diffusion_params.extra = LongCatAvatarDiffusionExtra{step};  // cond-frame K/V cache (lap-26)
                } else {
                    diffusion_params.extra = std::monostate{};
                }

                sd::Tensor<float> cached_output;
                if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                    return std::move(cached_output);
                }

                auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
                if (output_opt.empty()) {
                    LOG_ERROR("diffusion model compute failed");
                    return sd::Tensor<float>();
                }

                step_cache.after_condition(&condition, noised_input, output_opt);
                return output_opt;
            };

            const SDCondition* positive_condition      = &cond;
            const sd::Tensor<float>* c_concat_override = nullptr;
            for (const auto& extension : generation_extensions) {
                const SDCondition& next_condition = extension->before_condition(step, *positive_condition);
                if (&next_condition != positive_condition) {
                    positive_condition = &next_condition;
                    if (positive_condition != &cond) {
                        c_concat_override = cond.c_concat.empty() ? nullptr : &cond.c_concat;
                    }
                    break;
                }
            }

            cond_out = run_condition(*positive_condition, c_concat_override, nullptr, nullptr, false, /*nag_pass=*/true);
            if (cond_out.empty()) {
                return {};
            }

            // LTX_NAN_DEBUG: per-step health scan of the DiT input (noised_input = cont+noise) and
            // raw model output (cond_out). Pinpoints WHICH sampler step first overflows and whether
            // the input was already corrupt (cont-latent issue) vs the forward producing inf/nan
            // (FA/matmul overflow). Zero cost unless the env is set.
            if (getenv("LTX_NAN_DEBUG") != nullptr) {
                auto scan = [&](const sd::Tensor<float>& t, const char* tag) {
                    const float* d = t.data();
                    int64_t n      = t.numel();
                    long nn = 0, ni = 0;
                    float mx = -1e30f, mn = 1e30f;
                    for (int64_t i = 0; i < n; ++i) {
                        float v = d[i];
                        if (std::isnan(v)) {
                            nn++;
                        } else if (std::isinf(v)) {
                            ni++;
                        } else {
                            if (v > mx) mx = v;
                            if (v < mn) mn = v;
                        }
                    }
                    LOG_INFO("[LTX_NAN] step=%d %s nan=%ld inf=%ld range=%.3f..%.3f", step, tag, nn, ni, mn, mx);
                };
                scan(noised_input, "in ");
                scan(cond_out, "out");
            }

            // A2V modality guidance: re-run the DiT with audio<->video cross-attn severed (the
            // "mod" pass) and extrapolate cond away from it to amplify lip-sync. Only meaningful
            // when a driving audio is held fixed. pred = cond + (eff_scale-1)*(cond - mod), where
            // eff_scale ramps from the full scale at sigmas[0] to scale-ramped at sigma~0 so the
            // detail (low-sigma) steps don't over-drive and smear the lips.
            if (a2v_guidance_scale != 1.0f && sd_version_is_ltxav(version) && ltxav_audio_fixed) {
                float eff_scale = a2v_guidance_scale;
                if (a2v_ramp_end < 1.0f && !sigmas.empty() && sigmas[0] > 0.0f) {
                    float frac = std::clamp(1.0f - sigma / sigmas[0], 0.0f, 1.0f);  // 0 at high sigma -> 1 at low
                    float w    = 1.0f + (a2v_ramp_end - 1.0f) * frac;               // 1.0 -> a2v_ramp_end
                    eff_scale  = 1.0f + (a2v_guidance_scale - 1.0f) * w;
                }
                // Skip the (expensive) 2nd forward once the scheduled scale is negligible — with a
                // ramp-to-off this drops the audio-severed pass on the low-sigma detail steps, so
                // the schedule buys back compute, not just cleaner lips.
                if (std::fabs(eff_scale - 1.0f) > 0.02f) {
                    sd::Tensor<float> mod_out =
                        run_condition(*positive_condition, c_concat_override, nullptr, nullptr, /*skip_a2v_pass=*/true);
                    if (mod_out.empty()) {
                        return {};
                    }
                    cond_out = cond_out + (cond_out - mod_out) * (eff_scale - 1.0f);
                }
            }

            // NAG owns guidance in attention space (cfg<=1): skip the redundant output-level uncond
            // forward. The negative context was already consumed inside the cond forward above; its
            // cfg-1 CFG combine would be a numeric no-op anyway. (When cfg>1 the user wants BOTH
            // NAG and CFG, so the uncond forward still runs.)
            if (!uncond.empty() && !nag_owns_guidance) {
                if (!step_cache.is_step_skipped()) {
                    compute_sample_controls(control_image,
                                            noised_input,
                                            timesteps_tensor,
                                            uncond,
                                            &controls);
                }
                const std::vector<int>* uncond_skip_layers = nullptr;
                if (is_skiplayer_step && slg_uncond) {
                    LOG_DEBUG("Skipping layers at uncond step %d\n", step);
                    uncond_skip_layers = &skip_layer_guidance.layers();
                }
                uncond_out = run_condition(uncond,
                                           uncond.c_concat.empty() ? nullptr : &uncond.c_concat,
                                           uncond_skip_layers);
                if (uncond_out.empty()) {
                    return {};
                }
            }
            if (!img_uncond.empty()) {
                img_uncond_out = run_condition(img_uncond,
                                               img_uncond.c_concat.empty() ? nullptr : &img_uncond.c_concat,
                                               nullptr,
                                               uncond_without_ref_latents ? &empty_ref_latents : nullptr);
                if (img_uncond_out.empty()) {
                    return {};
                }
            }
            sd::guidance::GuidanceInput guidance_input;
            guidance_input.step            = step;
            guidance_input.schedule_size   = sigmas.size();
            guidance_input.pred_cond       = &cond_out;
            guidance_input.pred_uncond     = uncond_out.empty() ? nullptr : &uncond_out;
            guidance_input.pred_img_uncond = img_uncond_out.empty() ? nullptr : &img_uncond_out;

            sd::guidance::GuiderOutput guided = primary_guidance.forward(guidance_input, {});
            if (guided.pred.empty()) {
                return {};
            }

            if (is_skiplayer_step && slg_scale != 0.0f) {
                LOG_DEBUG("Skipping layers at step %d\n", step);
                if (!step_cache.is_step_skipped()) {
                    guidance_input.predict_skip_layer = [&]() -> sd::Tensor<float> {
                        return run_condition(cond,
                                             cond.c_concat.empty() ? nullptr : &cond.c_concat,
                                             &skip_layer_guidance.layers());
                    };
                }
            }

            guided = skip_layer_guidance.forward(guidance_input, std::move(guided));
            if (guided.pred.empty()) {
                return {};
            }

            denoised = guided.pred * c_out + x * c_skip;
            sd::guidance::GuiderOutput output;
            output.pred = denoised;
            if (needs_uncond_denoised) {
                const sd::Tensor<float>& base_uncond = !img_uncond_out.empty()
                                                           ? img_uncond_out
                                                           : (!uncond_out.empty() ? uncond_out : cond_out);
                output.pred_uncond                   = base_uncond * c_out + x * c_skip;
            }
            if (cache_runtime.spectrum_enabled) {
                cache_runtime.spectrum.update(denoised);
            }
            if (!denoise_mask.empty()) {
                denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
            }
            // [STEP_DELTA] non-perturbing: log relative-L1 change of the denoised x0
            // estimate between consecutive sampler steps. Measures DMD-step reuse window
            // (block/step-cache viability). Gated by LONGCAT_STEP_DELTA; zero cost when unset.
            if (getenv("LONGCAT_STEP_DELTA") != nullptr) {
                static std::vector<float> sd_prev_denoised;
                const float* dd = denoised.data();
                int64_t      dn = denoised.numel();
                double l1abs = 0.0, l1den = 0.0;
                if ((int64_t)sd_prev_denoised.size() == dn) {
                    for (int64_t i = 0; i < dn; ++i) {
                        l1abs += std::fabs(dd[i] - sd_prev_denoised[i]);
                        l1den += std::fabs(sd_prev_denoised[i]);
                    }
                    LOG_INFO("[STEP_DELTA] step %d rel_L1(denoised_t - denoised_{t-1}) = %.4f (abs %.5g / prev %.5g)",
                             step, l1den > 0 ? (l1abs / l1den) : 0.0, l1abs, l1den);
                }
                sd_prev_denoised.assign(dd, dd + dn);
            }
            if (sd_should_preview_denoised() && preview.callback != nullptr) {
                preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
            }
            report_sample_progress(step, steps, t0);
            output.pred = denoised;
            return output;
        };

        auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser, extra_sample_args);
        if (x0_opt.empty()) {
            LOG_ERROR("Diffusion model sampling failed");
            if (control_net) {
                control_net->free_control_ctx();
                control_net->free_compute_buffer();
            }
            if (work_diffusion_model) {
                work_diffusion_model->free_compute_buffer();
            }
            return {};
        }

        auto x0 = std::move(x0_opt);
        sd_sample::log_sample_cache_summary(cache_runtime, steps);
        // ComfyUI's SamplerCustomAdvanced exposes both the sampler trajectory
        // (`output`) and the last model x0 prediction (`denoised_output`).  A
        // partial LCM trajectory is not an x0 estimate: LCM has just mixed fresh
        // noise into it for its next sigma.  The LTX ver3 two-stage graph feeds
        // denoised_output to its learned latent upsampler, so expose that same
        // value for the narrowly-gated hires path below.
        if (return_denoised) {
            x0 = std::move(denoised);
        } else if (inverse_noise_scaling) {
            x0 = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x0);
        }

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        if (work_diffusion_model) {
            work_diffusion_model->free_compute_buffer();
        }
        return x0;
    }

    int get_vae_scale_factor() {
        if (sd_version_is_pid(version)) {
            return 1;
        }
        return first_stage_model->get_scale_factor();
    }

    int get_diffusion_model_down_factor() {
        int down_factor = 8;  // unet
        if (sd_version_is_dit(version)) {
            if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version)) {
                down_factor = 2;
            } else {
                down_factor = 1;
            }
        }
        return down_factor;
    }

    int get_latent_channel() {
        int latent_channel = 4;
        if (sd_version_is_dit(version)) {
            if (sd_version_is_ltxav(version)) {
                latent_channel = 128;
            } else if (version == VERSION_WAN2_2_TI2V) {
                latent_channel = 48;
            } else if (version == VERSION_HIDREAM_O1) {
                latent_channel = 3;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                latent_channel = 3;
            } else if (sd_version_is_pid(version)) {
                latent_channel = 3;
            } else if (sd_version_uses_flux2_vae(version)) {
                latent_channel = 128;
            } else {
                latent_channel = 16;
            }
        }
        return latent_channel;
    }

    int get_image_seq_len(int h, int w) {
        int vae_scale_factor = get_vae_scale_factor();
        return (h / vae_scale_factor) * (w / vae_scale_factor);
    }

    sd::Tensor<float> generate_init_latent(int width,
                                           int height,
                                           int frames = 1,
                                           bool video = false) {
        int vae_scale_factor = get_vae_scale_factor();
        int W                = width / vae_scale_factor;
        int H                = height / vae_scale_factor;
        int T                = video_frames_to_latent_frames(frames);
        int C                = get_latent_channel();
        if (video) {
            return sd::zeros<float>({W, H, T, C, 1});
        }
        return sd::zeros<float>({W, H, C, 1});
    }

    int video_frames_to_latent_frames(int frames) {
        int latent_frames = frames;
        if (sd_version_is_ltxav(version)) {
            latent_frames = ((frames - 1) / 8) + 1;
        } else if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version)) {
            latent_frames = ((frames - 1) / 4) + 1;
        }
        return latent_frames;
    }

    int latent_frames_to_video_frames(int latent_frames) {
        if (latent_frames <= 0) {
            return latent_frames;
        }
        if (sd_version_is_ltxav(version)) {
            return (latent_frames - 1) * 8 + 1;
        }
        if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version)) {
            return (latent_frames - 1) * 4 + 1;
        }
        return latent_frames;
    }

    int align_video_frames(int frames) {
        return latent_frames_to_video_frames(video_frames_to_latent_frames(frames));
    }

    sd::Tensor<float> encode_to_vae_latents(const sd::Tensor<float>& x) {
        // Mirror decode_first_stage: enable temporal tiling on the encoder too, else a
        // long clip (e.g. the LTXAV relip reference video) is encoded in ONE buffer and
        // OOMs at full res (1x1 spatial). The decode path set this; the encode path did
        // not, so --temporal-tiling silently had no effect on encode.
        first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
        // Decode can run full-spatial (1x1, temporal-streamed) but the encode's 4-pixel-frame
        // temporal groups overrun both conv paths at full spatial (cuDNN CONV_3D 2^31-element
        // limit / im2col IC*27 OOM). LONGCAT_VAE_ENCODE_REL_TILE=R (0<R<1) spatially tiles the
        // ENCODE only (e.g. 0.5 -> 0.5x0.5) while decode keeps its own (1x1) vae_tiling_params.
        sd_tiling_params_t enc_tp = vae_tiling_params;
        if (const char* s = getenv("LONGCAT_VAE_ENCODE_REL_TILE")) {
            float r = atof(s);
            if (r > 0.f && r < 1.f) {
                enc_tp.enabled    = true;
                enc_tp.tile_size_x = 0;
                enc_tp.tile_size_y = 0;
                enc_tp.rel_size_x = r;
                enc_tp.rel_size_y = r;
            }
        }
        auto latents = first_stage_model->encode(n_threads, x, enc_tp, circular_x, circular_y);
        if (latents.empty()) {
            return {};
        }
        latents = first_stage_model->vae_output_to_latents(latents, rng);
        return latents;
    }

    sd::Tensor<float> encode_first_stage(const sd::Tensor<float>& x) {
        auto latents = encode_to_vae_latents(x);
        if (latents.empty()) {
            return {};
        }
        // diffusers ip2p does NOT scale the conditioning image latent by the VAE scaling factor.
        if (!sd_version_is_sd_unet_edit(version)) {
            latents = first_stage_model->vae_to_diffusion_latents(latents);
        }
        return latents;
    }

    // TEMPORAL-CHUNKED video encode for contexts whose VAE encode OOMs when the WHOLE
    // temporal stack is fed at once. The base VAE encode() (vae.hpp:131) spatially tiles
    // but — unlike decode() (vae.hpp:188 set_tiling_params) — never propagates temporal
    // tiling, and the Wan VAE's temporal streaming is hard-gated decode-only
    // (wan_vae.hpp:1748 `chunk >= 1 && decode_graph`), so a 25-81 frame encode at
    // 832x456+ runs the cuDNN CONV_3D / im2col(IC*27) intermediate over ALL frames at
    // once -> the 14.7GB OOM at "failed to encode VACE inactive context".
    //
    // The Wan VAE encode is 4x causal-temporal: pixel group 0 = frame[0:1] -> 1 latent
    // frame; group i = frame[1+4(i-1):1+4i] -> 1 latent frame. Encoding each group through
    // the (still spatially-tiled) encode_first_stage and concatenating on the temporal
    // axis bounds the per-pass VRAM to <=4 frames x one spatial tile. For a CONSTANT (gray)
    // input every group is bit-identical to the monolithic encode (the causal conv sees
    // gray history either way). The VACE inactive/reactive context is gray everywhere
    // except the kept tail frames, and those are overwritten by VACE_CONT_LATENT the
    // instant after this returns, so the chunked result is exact wherever it is actually
    // used. Wan-only (the 4x grouping); off-switch WAN_VACE_ENCODE_NO_TCHUNK=1.
    sd::Tensor<float> encode_first_stage_temporal_chunked(const sd::Tensor<float>& x) {
        static const bool disabled = [] {
            const char* e = getenv("WAN_VACE_ENCODE_NO_TCHUNK");
            return e != nullptr && e[0] == '1';
        }();
        // <=4 pixel frames already encode in a single bounded pass; non-Wan VAEs do not
        // use the 1+4k grouping, so fall back to the plain (byte-identical) encode.
        if (disabled || !sd_version_is_wan(version) || x.dim() < 3 || x.shape()[2] <= 4) {
            return encode_first_stage(x);
        }
        const int64_t T = x.shape()[2];
        sd::Tensor<float> out;
        int64_t i = 0;
        for (int64_t fs = 0; fs < T;) {
            int64_t fe    = (i == 0) ? std::min<int64_t>(T, 1) : std::min<int64_t>(T, fs + 4);
            auto    group = sd::ops::slice(x, 2, fs, fe);                 // [W,H,<=4,C,..]
            auto    enc   = encode_first_stage(group);                    // 1 latent frame, spatially tiled
            if (enc.empty()) {
                LOG_ERROR("temporal-chunked encode failed on group %lld [%lld,%lld)",
                          (long long)i, (long long)fs, (long long)fe);
                return {};
            }
            out = out.empty() ? std::move(enc) : sd::ops::concat(out, enc, 2);
            fs  = fe;
            i++;
        }
        LOG_INFO("encode_first_stage_temporal_chunked: %lld pixel frames in %lld groups -> %lld latent frames",
                 (long long)T, (long long)i, (long long)(out.dim() > 2 ? out.shape()[2] : 1));
        return out;
    }

    sd::Tensor<float> decode_first_stage(const sd::Tensor<float>& x, bool decode_video = false) {
        if (getenv("LONGCAT_VRAM_BREAKDOWN") != nullptr) {
            double dit = diffusion_model ? diffusion_model->gpu_footprint_bytes()/1048576.0 : 0.0;
            double vae = first_stage_model ? first_stage_model->gpu_footprint_bytes()/1048576.0 : 0.0;
            double avae = audio_vae_model ? audio_vae_model->gpu_footprint_bytes()/1048576.0 : 0.0;
            LOG_INFO("[VRAM-ATTR decode-entry] DiT_gpu=%.0f MB  VAE_gpu=%.0f MB  audioVAE_gpu=%.0f MB (DiT should be ~0 here)", dit, vae, avae);
        }
        // GGML_F8_DBG: report the post-sampling latent magnitude entering the VAE. Saturated
        // (max|latent| >> ~10) => the DiT produced garbage (upstream/fp8 GEMM); reasonable
        // (~O(1-5)) => the DiT is fine and any white is a DECODE issue.
        if (getenv("GGML_F8_DBG") != nullptr && atoi(getenv("GGML_F8_DBG")) != 0) {
            const float* d = x.data();
            const size_t n = x.numel();
            float mx = 0.f, sum = 0.f;
            for (size_t i = 0; i < n; ++i) { const float a = std::fabs(d[i]); if (a > mx) mx = a; sum += a; }
            LOG_INFO("[F8_DBG] pre-VAE latent: numel=%zu  max|latent|=%.6g  mean|latent|=%.6g",
                     n, mx, n ? sum / (float)n : 0.f);
        }
        if (sd_version_is_pid(version)) {
            return sd::ops::clamp((x + 1.f) * 0.5f, 0.0f, 1.0f);
        }
        // Free resident diffusion params before VAE allocates its compute buffer.
        // Covers BOTH the streaming-layers residency and the LTX cross-step shared-
        // resident payload (lap-C), which lives on the offload path (stream_layers
        // off). release_streaming_residency() -> restore_resident_params() is an
        // idempotent no-op when nothing is resident, so it is safe to call always.
        if (diffusion_model) {
            diffusion_model->release_streaming_residency();
        }
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->release_streaming_residency();
        }
        auto latents = first_stage_model->diffusion_to_vae_latents(x);
        first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
        return first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
    }

    sd::Tensor<float> normalize_ltx_video_latents(const sd::Tensor<float>& x) {
        auto ltx_vae = std::dynamic_pointer_cast<LTXVideoVAE>(first_stage_model);
        if (!ltx_vae) {
            LOG_ERROR("LTX latent normalization requires LTX video VAE");
            return {};
        }
        return ltx_vae->normalize_latents(n_threads, x);
    }

    sd::Tensor<float> un_normalize_ltx_video_latents(const sd::Tensor<float>& x) {
        auto ltx_vae = std::dynamic_pointer_cast<LTXVideoVAE>(first_stage_model);
        if (!ltx_vae) {
            LOG_ERROR("LTX latent un-normalization requires LTX video VAE");
            return {};
        }
        return ltx_vae->un_normalize_latents(n_threads, x);
    }

    sd::Tensor<float> decode_ltx_audio_latent(const sd::Tensor<float>& audio_latent) {
        if (audio_vae_model == nullptr || audio_latent.empty()) {
            return {};
        }
        auto waveform = audio_vae_model->decode(n_threads, audio_latent);
        // !keep_diffusion_model_resident: on a warm resident chain (in-process N-segment
        // chaining) the audio VAE, like the video VAE/DiT/TE, must survive across segments —
        // free_params_buffer() nulls its tensors, and the next segment's decode would then
        // hit GGML_ASSERT(buffer). The video-VAE/DiT frees already carry this guard.
        if (free_params_immediately && !keep_diffusion_model_resident) {
            audio_vae_model->free_params_buffer();
        }
        return waveform;
    }

    // LTXAV chain inter-segment GPU reclaim. A warm resident chain (keep_diffusion_model_resident)
    // keeps every runner's CPU params alive so the next segment re-offloads without a disk reload,
    // but it ALSO leaves each runner's GPU-side residency + temporal/causal-conv CACHE buffer
    // resident across the segment boundary. Those stack onto the next segment's DiT sampling: a
    // 2-seg chain peaks ~12.5 GB @ mv10.5 vs ~11.1 GB single-seg (the +1.4 GB "chain anchor") —
    // measured leftover after seg-0's decode = ~1.74 GB (audio-VAE params left on-GPU by its
    // free_compute_buffer_immediately=false decode + the DiT/video-VAE/audio-VAE compute & cache
    // buffers). Everything here is rebuilt by the next segment's own compute, so release it. We
    // only drop the GPU-side residency + caches; each runner KEEPS its host params buffer
    // (free_compute_buffer restores the offload to host; free_params_buffer is NOT called), so the
    // next segment re-offloads from RAM with no reload and the resident-chain contract holds.
    void release_chain_segment_gpu_residency() {
        auto reclaim = [](auto& runner) {
            if (!runner) {
                return;
            }
            runner->log_gpu_residency_ledger("chain-boundary:before-release");
            runner->release_streaming_residency();    // drop any cross-step shared-resident payload
            runner->free_compute_buffer();            // restore offloaded params to host + free activations
            // SEAM FIX (97f continuation refine VRAM): free_compute_buffer's restore_partial_params
            // RETURNS the partial buffer to prefetch_buf_pool_ (pool_return, not cudaFree) and leaves
            // prefetched_state_.buf, so seg-1's ~420-627 MB of idle DiT streaming scratch would carry
            // into seg-2 and squat through its base+refine SAMPLING peak (the true 97f ceiling) —
            // ggml_backend_cuda_trim_pools can't reach these (they're backend buffers outside the
            // ggml_cuda_pool). Free them at the boundary; the next segment's stream re-creates them
            // lazily (one-time buffer alloc, no weight re-stream). Must run AFTER free_compute_buffer
            // (which repopulates the pool). No-op when empty → single-render byte-identical.
            runner->free_streaming_scratch_buffers();
            runner->free_cache_ctx_and_buffer();      // free the temporal/causal-conv cache buffer
            runner->log_gpu_residency_ledger("chain-boundary:after-release");
            // Diagnostic (b): param-buffer state right after the between-segment reclaim's
            // free_compute_buffer (which runs restore_partial/all_params). If the DiT
            // buffer is NULL here despite keep_diffusion_model_resident=true, the reclaim
            // (not a later step) is where it goes null.
            if (getenv("LTXAV_PARAM_BUF_TRACE") == nullptr || getenv("LTXAV_PARAM_BUF_TRACE")[0] != '0') {
                runner->log_param_buffer_state("release_chain_segment_gpu_residency:after-free-compute");
            }
        };
        reclaim(diffusion_model);
        reclaim(high_noise_diffusion_model);
        reclaim(first_stage_model);
        reclaim(audio_vae_model);

        // P3: optionally return the ggml CUDA VMM pool's committed high-water to the OS
        // between segments (real cuMemUnmap, not the prior empty-counter trim). MEASURED
        // PEAK-NEUTRAL (2026-06-25): a 3-seg chain peaks identically with this on or off,
        // because the peak is the genuine within-segment continuation working set (resident
        // DiT + the ~3.1 GB attention/FFN transient + offload) which re-commits every
        // segment — NOT cross-seg pool accumulation. So this is DEFAULT-OFF (opt-in). It is
        // still correct + harmless (lowers the inter-segment idle valley for gate
        // co-tenancy) and never frees params buffers, so SHARED_RESIDENT is preserved. The
        // real peak lever remains --max-vram (mv9.0 -> ~10.77 GB flat at +~31% sampling).
        if (getenv("LTXAV_CHAIN_POOL_TRIM") != nullptr) {
            ggml_backend_cuda_trim_pools(backend_for(SDBackendModule::DIFFUSION));
            ggml_backend_cuda_trim_pools(backend_for(SDBackendModule::VAE));
            ggml_backend_cuda_trim_pools(backend_for(SDBackendModule::TE));
        }

        // LTXAV_CHAIN_CUDNN_RESET (opt-in, default off = byte-identical): with cuDNN
        // enabled (GGML_CUDNN_ATTN / GGML_CUDNN_CONV3D — the prod continuation recipe),
        // the SDPA + conv3d execution-plan caches are process-global, keyed by shape.
        // Current cuDNN/CUDA can keep freed internal allocations in CUDA's async
        // mempool after frontend graphs/handles are destroyed; that memory is outside
        // the ggml VMM pool, so ggml_backend_cuda_trim_pools cannot reclaim it. Reset
        // the graph caches, this CUDA worker thread's handles, and CUDA mempools at the
        // segment boundary. The next segment recreates handles/plans lazily and rebuilds
        // only what it needs. Nothing is in flight here (all segment compute has
        // completed and synced). No-op on non-cuDNN builds.
        if (getenv("LTXAV_CHAIN_CUDNN_RESET") != nullptr) {
            ggml_backend_cuda_release_cudnn_plans();
        }
    }

    void set_flow_shift(float flow_shift = INFINITY) {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        if (flow_denoiser) {
            if (flow_shift == INFINITY) {
                flow_shift = default_flow_shift;
            }
            flow_denoiser->set_shift(flow_shift);
        }
    }

    bool is_flow_denoiser() {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        return !!flow_denoiser;
    }
};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

const char* sd_type_name(enum sd_type_t type) {
    if ((int)type < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)) {
        return ggml_type_name((ggml_type)type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_ancestral_cfg_pp",
    "euler_ge",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    // Contract spelling used by the LTX quality profiles. Keep the historical short
    // spelling accepted too, since it is exposed by the generic sd.cpp CLI/API.
    if (!strcmp(str, "euler_ancestral_cfg_pp") || !strcmp(str, "euler_a_cfg_pp")) {
        return EULER_A_CFG_PP_SAMPLE_METHOD;
    }
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "ltx2",
    "linear_quadratic",
};

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "flux2_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return NONE_STR;
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

const char* hires_upscaler_to_str[] = {
    "None",
    "Latent",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Lanczos",
    "Nearest",
    "Model",
};

const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler) {
    if (upscaler >= SD_HIRES_UPSCALER_NONE && upscaler < SD_HIRES_UPSCALER_COUNT) {
        return hires_upscaler_to_str[upscaler];
    }
    return NONE_STR;
}

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str) {
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; i++) {
        if (!strcmp(str, hires_upscaler_to_str[i])) {
            return (enum sd_hires_upscaler_t)i;
        }
    }
    return SD_HIRES_UPSCALER_COUNT;
}

const char* sd_vae_format_name(enum sd_vae_format_t format) {
    switch (format) {
        case SD_VAE_FORMAT_AUTO:
            return "auto";
        case SD_VAE_FORMAT_FLUX:
            return "flux";
        case SD_VAE_FORMAT_SD3:
            return "sd3";
        case SD_VAE_FORMAT_FLUX2:
            return "flux2";
        default:
            return NONE_STR;
    }
}

static SDVersion sd_vae_format_to_version(enum sd_vae_format_t format, SDVersion fallback) {
    switch (format) {
        case SD_VAE_FORMAT_FLUX:
            return VERSION_FLUX;
        case SD_VAE_FORMAT_SD3:
            return VERSION_SD3;
        case SD_VAE_FORMAT_FLUX2:
            return VERSION_FLUX2;
        case SD_VAE_FORMAT_AUTO:
        default:
            return fallback;
    }
}

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = INFINITY;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
    cache_params->spectrum_w                  = 0.40f;
    cache_params->spectrum_m                  = 3;
    cache_params->spectrum_lam                = 1.0f;
    cache_params->spectrum_window_size        = 2;
    cache_params->spectrum_flex_window        = 0.50f;
    cache_params->spectrum_warmup_steps       = 4;
    cache_params->spectrum_stop_percent       = 0.9f;
}

void sd_hires_params_init(sd_hires_params_t* hires_params) {
    *hires_params                     = {};
    hires_params->enabled             = false;
    hires_params->upscaler            = SD_HIRES_UPSCALER_LATENT;
    hires_params->model_path          = nullptr;
    hires_params->scale               = 2.0f;
    hires_params->target_width        = 0;
    hires_params->target_height       = 0;
    hires_params->steps               = 0;
    hires_params->denoising_strength  = 0.7f;
    hires_params->upscale_tile_size   = 128;
    hires_params->custom_sigmas       = nullptr;
    hires_params->custom_sigmas_count = 0;
    hires_params->loras               = nullptr;  // FEATURE 1 (--hires-lora): default = reuse base pass LoRAs
    hires_params->lora_count          = 0;
    hires_params->sample_method       = SAMPLE_METHOD_COUNT;  // sentinel = inherit base pass sampler
    hires_params->cfg                 = NAN;                  // legacy = inherit base cfg
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->sampler_rng_type        = RNG_TYPE_COUNT;
    sd_ctx_params->prediction              = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode         = LORA_APPLY_AUTO;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->max_vram                = 0.f;
    sd_ctx_params->stream_layers           = false;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
    sd_ctx_params->vae_format              = SD_VAE_FORMAT_AUTO;
    sd_ctx_params->backend                 = nullptr;
    sd_ctx_params->params_backend          = nullptr;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(8192);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 8192 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "clip_vision_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "diffusion_model_path: %s\n"
             "high_noise_diffusion_model_path: %s\n"
             "uncond_diffusion_model_path: %s\n"
             "embeddings_connectors_path: %s\n"
             "vae_path: %s\n"
             "audio_vae_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "photo_maker_path: %s\n"
             "tensor_type_rules: %s\n"
             "vae_decode_only: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "offload_params_to_cpu: %s\n"
             "max_vram: %.3f\n"
             "stream_layers: %s\n"
             "backend: %s\n"
             "params_backend: %s\n"
             "keep_clip_on_cpu: %s\n"
             "keep_control_net_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "chroma_use_dit_mask: %s\n"
             "chroma_use_t5_mask: %s\n"
             "chroma_t5_mask_pad: %d\n"
             "vae_format: %s\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->clip_vision_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path),
             SAFE_STR(sd_ctx_params->uncond_diffusion_model_path),
             SAFE_STR(sd_ctx_params->embeddings_connectors_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->audio_vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->photo_maker_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             BOOL_STR(sd_ctx_params->offload_params_to_cpu),
             sd_ctx_params->max_vram,
             BOOL_STR(sd_ctx_params->stream_layers),
             SAFE_STR(sd_ctx_params->backend),
             SAFE_STR(sd_ctx_params->params_backend),
             BOOL_STR(sd_ctx_params->keep_clip_on_cpu),
             BOOL_STR(sd_ctx_params->keep_control_net_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->circular_x),
             BOOL_STR(sd_ctx_params->circular_y),
             BOOL_STR(sd_ctx_params->chroma_use_dit_mask),
             BOOL_STR(sd_ctx_params->chroma_use_t5_mask),
             sd_ctx_params->chroma_t5_mask_pad,
             sd_vae_format_name(sd_ctx_params->vae_format));

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
    sample_params->extra_sample_args           = nullptr;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "shifted_timestep: %d, "
             "flow_shift: %.2f, "
             "extra_sample_args: %s)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->shifted_timestep,
             sample_params->flow_shift,
             SAFE_STR(sample_params->extra_sample_args));

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip         = -1;
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->vae_tiling_params = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_cache_params_init(&sd_img_gen_params->cache);
    sd_hires_params_init(&sd_img_gen_params->hires);
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "auto_resize_ref_image: %s\n"
             "increase_ref_index: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s (temporal=%s, extra_tiling_args=%s)\n"
             "hires: {enabled=%s, upscaler=%s, model_path=%s, scale=%.2f, target=%dx%d, steps=%d, denoising_strength=%.2f}\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             BOOL_STR(sd_img_gen_params->auto_resize_ref_image),
             BOOL_STR(sd_img_gen_params->increase_ref_index),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.temporal_tiling),
             SAFE_STR(sd_img_gen_params->vae_tiling_params.extra_tiling_args),
             BOOL_STR(sd_img_gen_params->hires.enabled),
             sd_hires_upscaler_name(sd_img_gen_params->hires.upscaler),
             SAFE_STR(sd_img_gen_params->hires.model_path),
             sd_img_gen_params->hires.scale,
             sd_img_gen_params->hires.target_width,
             sd_img_gen_params->hires.target_height,
             sd_img_gen_params->hires.steps,
             sd_img_gen_params->hires.denoising_strength);
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    } else if (sd_img_gen_params->cache.mode == SD_CACHE_UCACHE) {
        cache_mode_str = "ucache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             get_cache_reuse_threshold(sd_img_gen_params->cache),
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}

void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    *sd_vid_gen_params = {};
    sd_sample_params_init(&sd_vid_gen_params->sample_params);
    sd_sample_params_init(&sd_vid_gen_params->high_noise_sample_params);
    sd_vid_gen_params->high_noise_sample_params.sample_steps = -1;
    sd_vid_gen_params->width                                 = 512;
    sd_vid_gen_params->height                                = 512;
    sd_vid_gen_params->strength                              = 0.75f;
    sd_vid_gen_params->v2v_mode                              = 0;
    sd_vid_gen_params->v2v_guide_strength                    = 1.0f;
    sd_vid_gen_params->v2v_guide_latent_path                 = nullptr;
    sd_vid_gen_params->character_reference_latent            = nullptr;
    sd_vid_gen_params->character_reference_latent_lo         = nullptr;
    sd_vid_gen_params->character_reference_latent_lo_width   = 0;
    sd_vid_gen_params->character_reference_latent_lo_height  = 0;
    sd_vid_gen_params->character_reference_latent_lo_frames  = 0;
    sd_vid_gen_params->character_reference_latent_lo_channels = 0;
    sd_vid_gen_params->character_reference_latent_hi         = nullptr;
    sd_vid_gen_params->character_reference_latent_hi_width   = 0;
    sd_vid_gen_params->character_reference_latent_hi_height  = 0;
    sd_vid_gen_params->character_reference_latent_hi_frames  = 0;
    sd_vid_gen_params->character_reference_latent_hi_channels = 0;
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->fps                                   = 16;
    sd_vid_gen_params->drive_audio_path                      = nullptr;
    sd_vid_gen_params->cont_latent_path                      = nullptr;
    sd_vid_gen_params->cont_anchor_path                      = nullptr;
    sd_vid_gen_params->cont_latent                           = nullptr;
    sd_vid_gen_params->cont_latent_frames                    = 0;
    sd_vid_gen_params->chain_latent_offset                   = 0;  // 0 = legacy/no absolute offset
    sd_vid_gen_params->end_cont_latent                       = nullptr;
    sd_vid_gen_params->end_cont_latent_frames                = 0;
    sd_vid_gen_params->cont_refine_latent                    = nullptr;
    sd_vid_gen_params->cont_refine_latent_frames             = 0;
    sd_vid_gen_params->cont_refine_latent_width              = 0;
    sd_vid_gen_params->cont_refine_latent_height             = 0;
    sd_vid_gen_params->cont_refine_latent_channels           = 0;
    sd_vid_gen_params->cont_refine_latent_lo                 = nullptr;
    sd_vid_gen_params->cont_refine_latent_lo_frames          = 0;
    sd_vid_gen_params->cont_refine_latent_lo_width           = 0;
    sd_vid_gen_params->cont_refine_latent_lo_height          = 0;
    sd_vid_gen_params->cont_refine_latent_lo_channels        = 0;
    sd_vid_gen_params->keyframes                             = nullptr;
    sd_vid_gen_params->keyframe_frame_indices                = nullptr;
    sd_vid_gen_params->keyframes_size                        = 0;
    sd_vid_gen_params->audio_frame_offset                    = 0;
    sd_vid_gen_params->cont_ref_latent                       = nullptr;
    sd_vid_gen_params->cont_ref_img_index                    = 10;
    sd_vid_gen_params->cont_mask_frame_range                 = 3;
    // LongCat-Avatar BSA: dense by default (off); "r=1+self_frame" preset
    // baked into the defaults so a caller flipping bsa_enabled=1 gets exactly
    // the config the project tested with (mild rotation-drift trade, owner-OK).
    sd_vid_gen_params->bsa_enabled                           = 0;
    sd_vid_gen_params->bsa_radius                            = 1;
    sd_vid_gen_params->bsa_self_frame                        = 1;
    sd_vid_gen_params->bsa_bookend                           = 0;
    sd_vid_gen_params->bsa_cube_h                            = 4;
    sd_vid_gen_params->bsa_cube_w                            = 6;
    sd_vid_gen_params->moe_boundary                          = 0.875f;
    sd_vid_gen_params->vace_strength                         = 1.f;
    sd_vid_gen_params->vae_tiling_params                     = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_vid_gen_params->hires.enabled                         = false;
    sd_vid_gen_params->hires.upscaler                        = SD_HIRES_UPSCALER_LATENT;
    sd_vid_gen_params->hires.scale                           = 2.f;
    sd_vid_gen_params->hires.target_width                    = 0;
    sd_vid_gen_params->hires.target_height                   = 0;
    sd_vid_gen_params->hires.steps                           = 0;
    sd_vid_gen_params->hires.denoising_strength              = 0.7f;
    sd_vid_gen_params->hires.upscale_tile_size               = 128;
    sd_vid_gen_params->hires.custom_sigmas                   = nullptr;
    sd_vid_gen_params->hires.custom_sigmas_count             = 0;
    sd_vid_gen_params->hires.sample_method                   = SAMPLE_METHOD_COUNT;  // sentinel = inherit base sampler
    sd_vid_gen_params->hires.cfg                             = NAN;
    sd_vid_gen_params->hires_chain                           = nullptr;
    sd_vid_gen_params->hires_chain_count                     = 0;
    sd_vid_gen_params->emit_stages                           = 0;
    sd_vid_gen_params->on_stage                              = nullptr;
    sd_vid_gen_params->on_stage_user                         = nullptr;
    sd_vid_gen_params->stage_seg_index                       = 0;
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

static bool sd_version_supports_video_generation(SDVersion version) {
    return version == VERSION_SVD || sd_version_is_wan(version) || sd_version_is_ltxav(version) || sd_version_is_longcat_avatar(version);
}

static bool sd_version_supports_image_generation(SDVersion version) {
    return !sd_version_supports_video_generation(version);
}

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

static sd_audio_t* waveform_to_sd_audio(const StableDiffusionGGML* sd,
                                        const sd::Tensor<float>& waveform) {
    if (sd == nullptr || waveform.empty()) {
        return nullptr;
    }

    int64_t sample_count = waveform.shape()[0];
    int64_t channels     = waveform.shape().size() > 1 ? waveform.shape()[1] : 1;
    if (sample_count <= 0 || channels <= 0) {
        return nullptr;
    }

    sd_audio_t* audio = (sd_audio_t*)malloc(sizeof(sd_audio_t));
    if (audio == nullptr) {
        return nullptr;
    }

    audio->sample_rate  = static_cast<uint32_t>(sd->audio_vae_model != nullptr ? sd->audio_vae_model->config.output_sample_rate() : 0);
    audio->channels     = static_cast<uint32_t>(channels);
    audio->sample_count = static_cast<uint64_t>(sample_count);
    size_t sample_bytes = waveform.numel() * sizeof(float);
    audio->data         = (float*)malloc(sample_bytes);
    if (audio->data == nullptr) {
        free(audio);
        return nullptr;
    }

    auto wavaform_t = waveform.permute({1, 0, 2, 3});
    std::memcpy(audio->data, wavaform_t.data(), sample_bytes);

    return audio;
}

void free_sd_audio(sd_audio_t* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_video_generation(sd_ctx->sd->version);
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        if (sd_version_is_pid(sd_ctx->sd->version)) {
            return LCM_SAMPLE_METHOD;
        }
        if (sd_version_is_dit(sd_ctx->sd->version)) {
            return EULER_SAMPLE_METHOD;
        }
    }
    return EULER_A_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        auto edm_v_denoiser = std::dynamic_pointer_cast<EDMVDenoiser>(sd_ctx->sd->denoiser);
        if (edm_v_denoiser) {
            return EXPONENTIAL_SCHEDULER;
        }
    }
    if (sample_method == LCM_SAMPLE_METHOD || sample_method == TCD_SAMPLE_METHOD) {
        return LCM_SCHEDULER;
    } else if (sample_method == DDIM_TRAILING_SAMPLE_METHOD) {
        return SIMPLE_SCHEDULER;
    } else if (sd_ctx != nullptr && sd_ctx->sd != nullptr && sd_version_is_ltxav(sd_ctx->sd->version)) {
        return LTX2_SCHEDULER;
    }
    return DISCRETE_SCHEDULER;
}

static int64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    srand((int)time(nullptr));
    return rand();
}

// LongCat-Video-Avatar 1.5 DMD distilled sigma schedule.
// Reproduces pipeline_longcat_video_avatar.get_timesteps_sigmas(use_distill=True,
// model_type="avatar-v1.5") followed by FlowMatchEulerDiscreteScheduler.set_timesteps
// (shift applied, terminal 0 appended). Returns flow-match sigmas (size = steps + 1).
static std::vector<float> build_longcat_dmd_sigmas(int distill_steps, int num_train_timesteps, float shift) {
    // distill_indices = round(arange(1..steps) * (T/steps)); distill_indices = T - distill_indices
    // sigmas = flip(linspace(0,1,T))[distill_indices] ; flipped back to ascending step order
    std::vector<float> sigmas;
    sigmas.reserve(distill_steps + 1);
    for (int i = distill_steps; i >= 1; --i) {  // produce in flipped order -> step order
        long di = std::lround((double)i * (double)(num_train_timesteps / distill_steps));
        di      = num_train_timesteps - di;  // index into the descending-sigma array
        // flip(linspace(0,1,T))[di] == (T-1-di)/(T-1)
        double raw = (double)(num_train_timesteps - 1 - di) / (double)(num_train_timesteps - 1);
        // resolution-independent (linear) shift, shift=7.0
        double s = (double)shift * raw / (1.0 + ((double)shift - 1.0) * raw);
        sigmas.push_back((float)s);
    }
    sigmas.push_back(0.0f);  // terminal
    return sigmas;
}

// Sigma grid for the Wan2.2 lightx2v DMD distill, built from n_high (structure) + n_low (detail) steps
// around the trained boundary t=500 (boundary_step_index=2 of the trained [1000,750,500,250]). The two
// MoE experts have distinct jobs, each its own lever:
//   HIGH steps -> coarse structure/object coherence (the high-noise expert). More high steps fix the
//     "two-halves car" structure failures (FINDINGS-L8f: seed123/seed7 2 high = split, 4 high = coherent).
//     n_high evals are spaced evenly in (500,1000]: n_high=2 -> [1000,750]; n_high=4 -> [1000,875,750,625].
//   LOW steps -> detail/cleanup; more low steps kill the few-step "dotty" grain (FINDINGS-L8d: 2+2 dotty,
//     2+4 smooth). low evals start [500,250] then HALVE the tail (125,62,..) to refine the final denoise.
// Trained config = n_high=2,n_low=2 -> exactly [1000,750,500,250]. Shift is the LINEAR warp
// s=shift*r/(1+(shift-1)r), r=t/T (matches Wan fm_solvers; NOT exp/mu). The caller's --high-noise-steps
// sets n_high (the MoE high<->low switch index), --steps sets n_low; we do NOT clobber that split.
static std::vector<float> build_wan_distill_sigmas(int n_high, int n_low, int num_train_timesteps, float shift) {
    const int boundary_t = num_train_timesteps / 2;  // 500: trained high<->low switch
    if (n_high < 1) n_high = 2;
    if (n_low < 1) n_low = 2;
    std::vector<int> ts;
    // HIGH region: n_high evals evenly from T down to just above the boundary (structure phase).
    for (int i = 0; i < n_high; ++i)
        ts.push_back(num_train_timesteps - (int)std::lround((double)i * (double)boundary_t / (double)n_high));
    // LOW region: [boundary, boundary/2] then halve the tail (detail/cleanup phase).
    ts.push_back(boundary_t);
    int last = boundary_t;
    for (int i = 1; i < n_low; ++i) { last = last > 1 ? last / 2 : 1; ts.push_back(last); }
    std::vector<float> sigmas;
    sigmas.reserve(ts.size() + 1);
    for (int t : ts) {
        double r = (double)t / (double)num_train_timesteps;
        double s = (double)shift * r / (1.0 + ((double)shift - 1.0) * r);
        sigmas.push_back((float)s);
    }
    sigmas.push_back(0.0f);  // terminal
    return sigmas;
}

// Flow-shift for the Wan2.2 lightx2v DMD distill. NOTE: deliberately NOT resolution-coupled.
// A multi-res shift A/B (FINDINGS-L8c: 768x432 / 640x640 / 1280x720 x shift {3,5,7,9,11}, face
// close-up) showed the shift lever is ~FLAT for 5..11 at EVERY resolution (only shift 3 is
// clearly worse — dark/murky) and shows NO resolution dependence — confirming the lightx2v
// finding that "the shift lever is nearly dead at 4-step distill" (res-coupling matters for the
// FULL-STEP base path, shift 12 native, not the distill). So a single fixed shift ~7 is correct
// across 432p..720p; 7 is in the sweet spot everywhere and marginally crisper than 5. Overridable
// via WAN_DISTILL_SHIFT for experimentation. (seq_len arg kept for the log line only.)
static float wan_distill_shift(int /*spatial_seq_len*/) {
    const char* s = getenv("WAN_DISTILL_SHIFT");
    if (s != nullptr && s[0] != '\0') {
        float v = static_cast<float>(atof(s));
        if (v > 0.0f) return v;
    }
    return 7.0f;
}

static enum sample_method_t resolve_sample_method(sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sample_method == SAMPLE_METHOD_COUNT) {
        return sd_get_default_sample_method(sd_ctx);
    }
    return sample_method;
}

static scheduler_t resolve_scheduler(sd_ctx_t* sd_ctx,
                                     scheduler_t scheduler,
                                     enum sample_method_t sample_method) {
    if (scheduler == SCHEDULER_COUNT) {
        return sd_get_default_scheduler(sd_ctx, sample_method);
    }
    return scheduler;
}

static float resolve_eta(sd_ctx_t* sd_ctx,
                         float eta,
                         enum sample_method_t sample_method) {
    if (eta == INFINITY) {
        if (sd_ctx->sd->version == VERSION_HIDREAM_O1) {
            return 8.f;
        }
        switch (sample_method) {
            case DDIM_TRAILING_SAMPLE_METHOD:
            case TCD_SAMPLE_METHOD:
            case RES_MULTISTEP_SAMPLE_METHOD:
            case RES_2S_SAMPLE_METHOD:
                return 0.0f;
            case EULER_A_SAMPLE_METHOD:
            case DPMPP2S_A_SAMPLE_METHOD:
            case ER_SDE_SAMPLE_METHOD:
            case EULER_A_CFG_PP_SAMPLE_METHOD:
                return 1.0f;
            default:;
        }
        return 0.0f;
    }
    return eta;
}

struct GenerationRequest {
    std::string prompt;
    std::string negative_prompt;
    int width                                = -1;
    int height                               = -1;
    int clip_skip                            = -1;
    int vae_scale_factor                     = -1;
    int diffusion_model_down_factor          = -1;
    int64_t seed                             = -1;
    bool use_uncond                          = false;
    bool use_img_uncond                      = false;
    bool use_high_noise_uncond               = false;
    bool use_high_noise_img_uncond           = false;
    bool has_ref_images                      = false;
    const sd_cache_params_t* cache_params    = nullptr;
    int batch_count                          = 1;
    int shifted_timestep                     = 0;
    float strength                           = 1.f;
    float control_strength                   = 0.f;
    float eta                                = 0.f;
    bool increase_ref_index                  = false;
    bool auto_resize_ref_image               = false;
    sd_guidance_params_t guidance            = {};
    sd_guidance_params_t high_noise_guidance = {};
    sd_pm_params_t pm_params                 = {};
    sd_hires_params_t hires                  = {};
    int frames                               = -1;
    int requested_frames                     = -1;
    int fps                                  = 16;
    float vace_strength                      = 1.f;
    enum sample_method_t sample_method       = SAMPLE_METHOD_COUNT;

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
        prompt                      = SAFE_STR(sd_img_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_img_gen_params->negative_prompt);
        width                       = sd_img_gen_params->width;
        height                      = sd_img_gen_params->height;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_img_gen_params->seed;
        batch_count                 = sd_img_gen_params->batch_count;
        clip_skip                   = sd_img_gen_params->clip_skip;
        shifted_timestep            = sd_img_gen_params->sample_params.shifted_timestep;
        strength                    = sd_img_gen_params->strength;
        control_strength            = sd_img_gen_params->control_strength;
        eta                         = sd_img_gen_params->sample_params.eta;
        increase_ref_index          = sd_img_gen_params->increase_ref_index;
        auto_resize_ref_image       = sd_img_gen_params->auto_resize_ref_image;
        has_ref_images              = sd_img_gen_params->ref_images_count > 0;
        guidance                    = sd_img_gen_params->sample_params.guidance;
        pm_params                   = sd_img_gen_params->pm_params;
        hires                       = sd_img_gen_params->hires;
        cache_params                = &sd_img_gen_params->cache;
        sample_method               = sd_img_gen_params->sample_params.sample_method;
        resolve(sd_ctx);
    }

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params) {
        prompt                      = SAFE_STR(sd_vid_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_vid_gen_params->negative_prompt);
        width                       = sd_vid_gen_params->width;
        height                      = sd_vid_gen_params->height;
        requested_frames            = std::max(1, sd_vid_gen_params->video_frames);
        frames                      = sd_ctx->sd->align_video_frames(requested_frames);
        clip_skip                   = sd_vid_gen_params->clip_skip;
        fps                         = std::max(1, sd_vid_gen_params->fps);
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_vid_gen_params->seed;
        strength                    = sd_vid_gen_params->strength;
        cache_params                = &sd_vid_gen_params->cache;
        vace_strength               = sd_vid_gen_params->vace_strength;
        // VACE_STRENGTH env override: scales the vace branch residual (c_skip *= strength).
        // Set 0 for pure-t2v seg0 (gray control -> the undistilled vace branch stipples; zeroing
        // it = pure base t2v = clean) and keep 1 for i2v/continuation (real reference control).
        if (const char* vs = getenv("VACE_STRENGTH")) {
            vace_strength = (float)atof(vs);
            LOG_INFO("VACE_STRENGTH override: vace branch scaled to %.3f", vace_strength);
        }
        guidance                    = sd_vid_gen_params->sample_params.guidance;
        high_noise_guidance         = sd_vid_gen_params->high_noise_sample_params.guidance;
        hires                       = sd_vid_gen_params->hires;
        sample_method               = sd_vid_gen_params->sample_params.sample_method;
        resolve(sd_ctx);
        if (frames != requested_frames) {
            LOG_WARN("align video frames from %d to %d for %s",
                     requested_frames,
                     frames,
                     model_version_to_str[sd_ctx->sd->version]);
        }
    }

    void align_generation_request_size() {
        align_image_size(&width, &height, "generation request");
    }

    void align_image_size(int* target_width, int* target_height, const char* label) {
        int spatial_multiple = vae_scale_factor * diffusion_model_down_factor;
        int width_offset     = align_up_offset(*target_width, spatial_multiple);
        int height_offset    = align_up_offset(*target_height, spatial_multiple);
        if (width_offset <= 0 && height_offset <= 0) {
            return;
        }

        int original_width  = *target_width;
        int original_height = *target_height;

        *target_width += width_offset;
        *target_height += height_offset;
        LOG_WARN("align %s up %dx%d to %dx%d (multiple=%d)",
                 label,
                 original_width,
                 original_height,
                 *target_width,
                 *target_height,
                 spatial_multiple);
    }

    void resolve_hires() {
        if (!hires.enabled) {
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_NONE) {
            hires.enabled = false;
            return;
        }
        if (hires.upscaler < SD_HIRES_UPSCALER_NONE || hires.upscaler >= SD_HIRES_UPSCALER_COUNT) {
            LOG_WARN("hires upscaler '%d' is invalid, disabling hires", hires.upscaler);
            hires.enabled = false;
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_MODEL && strlen(SAFE_STR(hires.model_path)) == 0) {
            LOG_WARN("hires model upscaler requires a model path, disabling hires");
            hires.enabled = false;
            return;
        }
        if (hires.scale <= 0.f && hires.target_width <= 0 && hires.target_height <= 0) {
            LOG_WARN("hires scale must be positive when no target size is set, disabling hires");
            hires.enabled = false;
            return;
        }
        if (hires.custom_sigmas_count < 0) {
            LOG_WARN("hires custom sigmas count is negative, ignoring custom sigmas");
            hires.custom_sigmas       = nullptr;
            hires.custom_sigmas_count = 0;
        }
        if (hires.custom_sigmas_count > 0 && hires.custom_sigmas == nullptr) {
            LOG_WARN("hires custom sigmas count is positive but custom sigmas are null, ignoring custom sigmas");
            hires.custom_sigmas_count = 0;
        }
        if (hires.custom_sigmas_count == 1) {
            LOG_WARN("hires custom sigmas requires at least two values, ignoring custom sigmas");
            hires.custom_sigmas       = nullptr;
            hires.custom_sigmas_count = 0;
        }
        hires.denoising_strength = std::clamp(hires.denoising_strength, 0.0001f, 1.f);
        hires.steps              = std::max(0, hires.steps);

        if (hires.target_width > 0 && hires.target_height > 0) {
            // pass
        } else if (hires.target_width > 0) {
            hires.target_height = hires.target_width;
        } else if (hires.target_height > 0) {
            hires.target_width = hires.target_height;
        } else {
            hires.target_width  = static_cast<int>(std::round(width * hires.scale));
            hires.target_height = static_cast<int>(std::round(height * hires.scale));
        }

        if (hires.target_width <= 0 || hires.target_height <= 0) {
            LOG_WARN("hires target size is not positive, disabling hires");
            hires.enabled = false;
            return;
        }
        align_image_size(&hires.target_width, &hires.target_height, "hires target");
    }

    static void resolve_guidance(sd_ctx_t* sd_ctx,
                                 sd_guidance_params_t* guidance,
                                 bool* use_uncond,
                                 bool* use_img_uncond,
                                 bool has_ref_images,
                                 enum sample_method_t sample_method,
                                 const char* stage_name = nullptr) {
        GGML_ASSERT(guidance != nullptr);
        GGML_ASSERT(use_uncond != nullptr);
        GGML_ASSERT(use_img_uncond != nullptr);
        // out_img_uncond + text_cfg_scale * (out_cond - out_uncond) + image_cfg_scale * (out_uncond - out_img_uncond)
        // -> text_cfg_scale * out_cond + (image_cfg_scale - text_cfg_scale) * out_uncond + (1 - image_cfg_scale) * out_img_uncond
        // out_cond       : prompt, image latent
        // out_uncond     : negative prompt, image latent
        // out_img_uncond : negative prompt, zero image latent
        // image_cfg_scale == 1 reduces 3-cond CFG to 2-cond CFG.
        bool img_cfg_was_set = std::isfinite(guidance->img_cfg);
        if (!img_cfg_was_set) {
            guidance->img_cfg = 1.f;
        }

        if (!sd_version_supports_img_cfg(sd_ctx->sd->version, has_ref_images)) {
            if (img_cfg_was_set && guidance->img_cfg != 1.f) {
                LOG_WARN("3-conditioning CFG is not supported with this model, disabling it for better performance");
            }
            guidance->img_cfg = 1.f;
        }

        if (guidance->img_cfg != guidance->txt_cfg || guidance->txt_cfg > 1.f) {
            *use_uncond = true;
        }

        // CFG++ samplers require a real uncond prediction every step, even at cfg==1
        // (ComfyUI forces this via disable_cfg1_optimization). Without it pred_uncond is
        // empty and the cfg_pp step degenerates / blows up.
        if (sample_method == EULER_CFG_PP_SAMPLE_METHOD || sample_method == EULER_A_CFG_PP_SAMPLE_METHOD) {
            *use_uncond = true;
        }

        if (guidance->img_cfg != 1.f) {
            *use_img_uncond = true;
        }

        if (guidance->txt_cfg < 1.f) {
            const char* prefix = stage_name == nullptr ? "" : stage_name;
            if (guidance->txt_cfg == 0.f) {
                LOG_WARN("%sunconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)",
                         prefix);
            } else {
                LOG_WARN("%scfg value out of expected range may produce unexpected results", prefix);
            }
        }
    }

    void resolve(sd_ctx_t* sd_ctx) {
        align_generation_request_size();
        resolve_hires();
        seed = resolve_seed(seed);

        resolve_guidance(sd_ctx, &guidance, &use_uncond, &use_img_uncond, has_ref_images, sample_method);
        // NAG needs the NEGATIVE text context materialized even at cfg=1 (where use_uncond would
        // otherwise be false and the negative prompt never encoded). Force it on so
        // prepare_video_generation_embeds encodes embeds.uncond and generate_video passes it into
        // sample() as the `uncond` carrier; the sampler then feeds uncond.c_crossattn into the
        // single cond forward as the NAG negative context (and suppresses the redundant cfg-1 uncond
        // forward). Gated on the same per-render env the sampler reads, so it can't get sticky.
        if (sd_version_is_ltxav(sd_ctx->sd->version)) {
            if (const char* e = std::getenv("LTXAV_NAG_SCALE")) {
                if ((float)atof(e) != 0.0f) {
                    use_uncond = true;
                }
            }
        }
        if (sd_ctx->sd->high_noise_diffusion_model) {
            resolve_guidance(sd_ctx,
                             &high_noise_guidance,
                             &use_high_noise_uncond,
                             &use_high_noise_img_uncond,
                             has_ref_images,
                             sample_method,
                             "high noise: ");
        }

        if (shifted_timestep > 0 && !sd_version_is_sdxl(sd_ctx->sd->version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
    }
};

struct SamplePlan {
    enum sample_method_t sample_method            = SAMPLE_METHOD_COUNT;
    enum sample_method_t high_noise_sample_method = SAMPLE_METHOD_COUNT;
    const char* extra_sample_args                 = nullptr;
    const char* high_noise_extra_sample_args      = nullptr;
    float eta                                     = 0.f;
    float high_noise_eta                          = 0.f;
    int sample_steps                              = 0;
    int high_noise_sample_steps                   = 0;
    int total_steps                               = 0;
    float moe_boundary                            = 0.f;
    // GUIDE-EDIT (sd_vid_gen_params->v2v_mode==2): DEFAULT the source-as-guide edit path to the
    // `linear_quadratic` schedule (the Director-2 guide schedule; prod LTX_CUSTOM_SIGMAS reproduces
    // it bit-for-bit) when the request left scheduler unset, running the full step budget. Only a
    // default — an explicit scheduler / LTX_CUSTOM_SIGMAS / custom_sigmas still win (for GPU A/B).
    // false on every other path.
    bool v2v_guide_edit                           = false;
    // control_frames_size at construction — 0 for plain t2v/i2v. Gates the LTXAV ver3 baked sigma
    // default: relip / SDEdit / guide-edit provide control frames and keep the env / scheduler path.
    int control_frames_count                      = 0;
    std::vector<float> sigmas;

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_img_gen_params_t* sd_img_gen_params,
               const GenerationRequest& request) {
        sample_method     = sd_img_gen_params->sample_params.sample_method;
        extra_sample_args = sd_img_gen_params->sample_params.extra_sample_args;
        eta               = sd_img_gen_params->sample_params.eta;
        sample_steps      = sd_img_gen_params->sample_params.sample_steps;
        resolve(sd_ctx, &request, &sd_img_gen_params->sample_params);
    }

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_vid_gen_params_t* sd_vid_gen_params,
               const GenerationRequest& request) {
        sample_method     = sd_vid_gen_params->sample_params.sample_method;
        extra_sample_args = sd_vid_gen_params->sample_params.extra_sample_args;
        eta               = sd_vid_gen_params->sample_params.eta;
        sample_steps      = sd_vid_gen_params->sample_params.sample_steps;
        if (sd_ctx->sd->high_noise_diffusion_model) {
            high_noise_sample_steps      = sd_vid_gen_params->high_noise_sample_params.sample_steps;
            high_noise_sample_method     = sd_vid_gen_params->high_noise_sample_params.sample_method;
            high_noise_extra_sample_args = sd_vid_gen_params->high_noise_sample_params.extra_sample_args;
            high_noise_eta               = sd_vid_gen_params->high_noise_sample_params.eta;
        }
        moe_boundary    = sd_vid_gen_params->moe_boundary;
        v2v_guide_edit  = (sd_vid_gen_params->v2v_mode == 2);
        control_frames_count = sd_vid_gen_params->control_frames_size;
        resolve(sd_ctx, &request, &sd_vid_gen_params->sample_params);
    }

    void resolve(sd_ctx_t* sd_ctx,
                 const GenerationRequest* request,
                 const sd_sample_params_t* sample_params) {
        sample_method = resolve_sample_method(sd_ctx, sample_method);

        total_steps = sample_steps + std::max(0, high_noise_sample_steps);

        // A/B hook: LTX_CUSTOM_SIGMAS="1.0,0.99375,...,0.0" overrides the scheduler with an
        // explicit sigma list, used RAW (no flow-shift) exactly like sample_params.custom_sigmas.
        // Lets us feed the official LTX-2.3 distilled 8-step schedule for a scheduler A/B.
        std::vector<float> env_sigmas;
        if (const char* e = getenv("LTX_CUSTOM_SIGMAS"); e != nullptr && e[0] != '\0') {
            const char* p = e;
            while (*p != '\0') {
                char* end = nullptr;
                float v   = std::strtof(p, &end);
                if (end == p) {
                    break;
                }
                env_sigmas.push_back(v);
                p = end;
                while (*p == ',' || *p == ' ') {
                    ++p;
                }
            }
        }

        if (sample_params->custom_sigmas_count > 0) {
            // Per-request explicit sigmas WIN over the LTX_CUSTOM_SIGMAS env A/B hook, so a caller
            // (e.g. koblem's ver3 6-step base schedule) can override the container default. The env
            // stays a fallback for callers that send no custom_sigmas (relip, direct scheduler A/B).
            sigmas      = std::vector<float>(sample_params->custom_sigmas,
                                        sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
            }
            LOG_INFO("request custom_sigmas: %d sigmas => %d steps [%.5f .. %.5f]",
                     static_cast<int>(sigmas.size()), total_steps, sigmas.front(), sigmas.back());
        } else if (sd_version_is_ltxav(sd_ctx->sd->version) && !v2v_guide_edit && control_frames_count <= 0) {
            // LTXAV plain-generate baked default = the ver3 "distilled" shape, and it beats the
            // LTX_CUSTOM_SIGMAS env so plain t2v/i2v renders default to ver3 without any env/koblem
            // help. Two-pass (a hires refine will run) splits the distilled schedule — the base
            // denoises 1.0 -> 0.725 in 6 steps and the hires refine finishes 0.725 -> 0; single-pass
            // runs the full 8-step schedule. Gated to plain t2v/i2v (no control_frames, not
            // guide-edit) so relip / SDEdit / guide-edit fall through to the env / scheduler path
            // below unchanged. Explicit request custom_sigmas still wins above.
            if (request->hires.enabled) {
                static const float kVer3TwoPassBase[] = {1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f};
                sigmas = std::vector<float>(kVer3TwoPassBase, kVer3TwoPassBase + 7);
            } else {
                static const float kVer3SinglePass[] = {1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f};
                sigmas = std::vector<float>(kVer3SinglePass, kVer3SinglePass + 9);
            }
            total_steps = static_cast<int>(sigmas.size()) - 1;
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
            }
            LOG_INFO("LTXAV ver3 baked default (%s): %d sigmas => %d steps [%.5f .. %.5f]",
                     request->hires.enabled ? "two-pass 6-step base" : "single-pass 8-step",
                     static_cast<int>(sigmas.size()), total_steps, sigmas.front(), sigmas.back());
        } else if (env_sigmas.size() >= 2) {
            // Reached only by relip / SDEdit / guide-edit (the baked branch above owns plain t2v/i2v).
            // LTX_CUSTOM_SIGMAS supplies their base schedule (prod distilled 8-step), unchanged.
            sigmas      = env_sigmas;
            total_steps = static_cast<int>(sigmas.size()) - 1;
            sample_steps = std::min(sample_steps, total_steps);
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
            }
            LOG_INFO("LTX_CUSTOM_SIGMAS override: %d sigmas => %d steps [%.5f .. %.5f]",
                     static_cast<int>(sigmas.size()), total_steps, sigmas.front(), sigmas.back());
        } else {
            scheduler_t scheduler = resolve_scheduler(sd_ctx,
                                                      sample_params->scheduler,
                                                      sample_method);
            // GUIDE-EDIT defaults to `linear_quadratic` — the schedule the Director-2 workflow uses
            // for guide-conditioned edits (and what prod LTX_CUSTOM_SIGMAS reproduces bit-for-bit).
            // It front-loads the mid/low-noise band (σ≈0.4–0.9) that the guide leans on to hold the
            // source scene, where LTX2's back-loaded flow curve wastes those steps up high. Only
            // DEFAULTS it (when the request left scheduler unset) so a GPU pass can A/B by naming an
            // explicit scheduler. (Explicit LTX_CUSTOM_SIGMAS / custom_sigmas above still win.)
            if (v2v_guide_edit && sd_version_is_ltxav(sd_ctx->sd->version) &&
                sample_params->scheduler == SCHEDULER_COUNT) {
                scheduler = LINEAR_QUADRATIC_SCHEDULER;
                LOG_INFO("LTXAV guide-edit: defaulting to linear_quadratic scheduler (%d full steps; "
                         "SDEdit truncation by edit strength applied after)", total_steps);
            }
            int sample_seq_len    = sd_ctx->sd->get_image_seq_len(request->height, request->width);
            if (sd_version_is_ltxav(sd_ctx->sd->version) && request->frames > 0) {
                int latent_frames = ((request->frames - 1) / 8) + 1;
                sample_seq_len *= latent_frames;
            }
            sigmas = sd_ctx->sd->denoiser->get_sigmas(total_steps,
                                                      sample_seq_len,
                                                      scheduler,
                                                      sd_ctx->sd->version,
                                                      sample_params->extra_sample_args);
        }

        // LongCat-Video-Avatar 1.5 is distilled (the DMD LoRA is folded into the
        // q4_K weights at convert time). Inference uses the model's own fixed DMD
        // sigma schedule (num_distill_sample_steps=8, num_train_timesteps=1000,
        // FlowMatchEuler shift 7.0) — see pipeline get_timesteps_sigmas(avatar-v1.5,
        // use_distill=True) + scheduler.set_timesteps. We override whatever scheduler
        // sigmas were computed above with this exact schedule so the few-step path
        // actually denoises. (Honour an explicit custom sigma list if the user gave one.)
        if (sd_version_is_longcat_avatar(sd_ctx->sd->version) &&
            sample_params->custom_sigmas_count <= 0) {
            // The DMD distill schedule is natively 8 steps. build_longcat_dmd_sigmas
            // re-derives a valid distilled schedule for any step count (the distill
            // indices scale with the count), so --steps can trade a little quality
            // for speed. Default to 8 (the count the LoRA was distilled for) unless
            // the user explicitly asked for fewer via --steps.
            int dmd_steps = (sample_steps > 0 && sample_steps < 8) ? sample_steps : 8;
            sigmas        = build_longcat_dmd_sigmas(dmd_steps, 1000, 7.0f);
            total_steps   = static_cast<int>(sigmas.size()) - 1;
            sample_steps  = total_steps;
            LOG_INFO("LongCat-Avatar DMD distilled schedule: %d steps", total_steps);
        }

        // Wan2.2 lightx2v DMD distill (wan22-*-distill GGUFs): SAME bug as LongCat above —
        // the few-step distill needs its native DMD sigma grid ([1000,750,500,250] @ shift,
        // = build_longcat_dmd_sigmas), but Wan2.2 falls through to the generic Discrete
        // scheduler which emits t=[999,666,333,0]+0 = a WASTED step on the WRONG grid ->
        // under-denoised "murk" (FINDINGS-L8, confirmed by --sigmas A/B). Opt-in via
        // WAN_DISTILL_SIGMAS=1 (the FULL-STEP VACE-Fun base must NOT use this — no model-side
        // marker distinguishes them, so it's an explicit env gate). Shift is resolution-coupled
        // like LTX2. Unlike the avatar branch this MUST NOT clobber sample_steps /
        // high_noise_sample_steps — Wan2.2 is MoE and that split drives the high<->low expert
        // switch; we replace ONLY the sigma grid (count stays total_steps).
        else if (sd_version_is_wan(sd_ctx->sd->version) &&
                 getenv("WAN_DISTILL_SIGMAS") != nullptr &&
                 sample_params->custom_sigmas_count <= 0) {
            int n_high    = high_noise_sample_steps > 0 ? high_noise_sample_steps : 2;
            int n_low     = sample_steps > 0 ? sample_steps : 2;
            int seq_len   = sd_ctx->sd->get_image_seq_len(request->height, request->width);
            float shift   = wan_distill_shift(seq_len);
            sigmas        = build_wan_distill_sigmas(n_high, n_low, 1000, shift);
            LOG_INFO("Wan2.2 DMD distilled schedule: %d high + %d low steps, shift=%.2f (seq_len=%d)",
                     n_high, n_low, shift, seq_len);
        }

        eta = resolve_eta(sd_ctx, eta, sample_method);

        if (high_noise_sample_steps < 0) {
            for (size_t i = 0; i < sigmas.size(); ++i) {
                if (sigmas[i] < moe_boundary) {
                    high_noise_sample_steps = static_cast<int>(i);
                    break;
                }
            }
            LOG_DEBUG("switching from high noise model at step %d", high_noise_sample_steps);
        }

        LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
        if (high_noise_sample_steps > 0) {
            high_noise_sample_method = resolve_sample_method(sd_ctx,
                                                             high_noise_sample_method);
            high_noise_eta           = resolve_eta(sd_ctx, high_noise_eta, high_noise_sample_method);
            LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);
        }
    }
};

struct ImageGenerationLatents {
    sd::Tensor<float> init_latent;
    sd::Tensor<float> concat_latent;
    sd::Tensor<float> img_uncond_concat_latent;
    sd::Tensor<float> audio_latent;
    sd::Tensor<float> audio_positions;
    sd::Tensor<float> video_positions;
    // FIX A2 separable half-res relip reference (LTXAV_RELIP_REF_DOWNSCALE>1): the [W/N,H/N,ref,C]
    // reference latent fed to the DiT as a separate token block (NOT concatenated into init_latent).
    // Empty on every other path (incl. N==1 full-res concat) => legacy behaviour untouched.
    sd::Tensor<float> video_reference;
    sd::Tensor<float> control_image;
    std::vector<sd::Tensor<float>> ref_images;
    std::vector<sd::Tensor<float>> ref_latents;
    sd::Tensor<float> denoise_mask;
    sd::Tensor<float> clip_vision_output;
    sd::Tensor<float> vace_context;
    int64_t ref_image_num                  = 0;
    int64_t video_conditioning_frame_count = 0;
    int64_t video_target_frame_count       = 0;
    int audio_length                       = 0;
    int audio_target_length                = 0;
    bool audio_reference_conditioning      = false;
    // Generic V2V (SDEdit): set when the video channels of init_latent were seeded from a source
    // clip and the caller must truncate the sigma schedule by `strength` before sampling.
    bool v2v_sdedit                        = false;
    bool audio_fixed                       = false;  // LTXAV: hold audio latent fixed (drive lip-sync to a given wav)
    // Two-stage lipdub relip (LTXAV_RELIP_TWOSTAGE=1): stage-1 ran at half res from-noise; the
    // hires latent-upscale stage-2 must RE-APPLY the relip reference at full res (Change B in
    // apply_ltxv_refine_image_conditioning). These carry the stage-1 setup forward so stage-2 can
    // re-encode the reference at the upscaled resolution. relip_twostage=false on every other path.
    bool relip_twostage                    = false;
    int relip_ref_downscale                = 1;  // reference_downscale_factor (spatial), per LoRA / env
    int relip_ref_tstride                  = 1;  // temporal subsample stride, per env
};

static float ltxv_latent_corner_to_pixel_frame(int64_t corner_index,
                                               int temporal_scale,
                                               bool causal_temporal_positioning) {
    float pixel_t = static_cast<float>(corner_index * temporal_scale);
    if (causal_temporal_positioning) {
        pixel_t = std::max(0.f, pixel_t + 1.f - static_cast<float>(temporal_scale));
    }
    return pixel_t;
}

static void set_ltxv_video_position(sd::Tensor<float>* positions,
                                    int64_t token,
                                    float t_start,
                                    float t_end,
                                    float h_start,
                                    float h_end,
                                    float w_start,
                                    float w_end) {
    positions->index(0, 0, token, 0) = t_start;
    positions->index(1, 0, token, 0) = t_end;
    positions->index(0, 1, token, 0) = h_start;
    positions->index(1, 1, token, 0) = h_end;
    positions->index(0, 2, token, 0) = w_start;
    positions->index(1, 2, token, 0) = w_end;
}

// Positions for a temporal sub-window on the original AV timeline. Unlike the
// default RoPE builder, `latent_start` is not reset to zero: frozen overlap
// frames and the matching audio slice must retain their shared absolute time.
static sd::Tensor<float> build_ltxav_window_video_positions(int64_t width,
                                                            int64_t height,
                                                            int64_t latent_start,
                                                            int64_t latent_frames,
                                                            int fps,
                                                            int spatial_scale,
                                                            int temporal_scale = 8) {
    GGML_ASSERT(width > 0 && height > 0 && latent_frames > 0 && fps > 0);
    sd::Tensor<float> positions({2, 3, width * height * latent_frames, 1});
    int64_t token = 0;
    for (int64_t t = 0; t < latent_frames; ++t) {
        const float t_start = ltxv_latent_corner_to_pixel_frame(latent_start + t, temporal_scale, true) /
                              static_cast<float>(fps);
        const float t_end = ltxv_latent_corner_to_pixel_frame(latent_start + t + 1, temporal_scale, true) /
                            static_cast<float>(fps);
        for (int64_t h = 0; h < height; ++h) {
            const float h_start = static_cast<float>(h * spatial_scale);
            const float h_end = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; ++w) {
                set_ltxv_video_position(&positions,
                                        token++,
                                        t_start,
                                        t_end,
                                        h_start,
                                        h_end,
                                        static_cast<float>(w * spatial_scale),
                                        static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    return positions;
}

// The continuation high-res guide is a separate frozen video-token block.  A
// temporal refine tile therefore needs positions for both its absolute-time
// target frames and the unchanged guide frames; passing target-only positions
// makes the DiT reject the appended guide-token sequence.  This mirrors
// build_ltxv_video_positions(), but preserves the target window's position on
// the original timeline instead of rebasing it to zero.
static sd::Tensor<float> build_ltxav_window_video_positions_with_reference(int64_t width,
                                                                             int64_t height,
                                                                             int64_t latent_start,
                                                                             int64_t latent_frames,
                                                                             int64_t reference_frames,
                                                                             int fps,
                                                                             int spatial_scale,
                                                                             int temporal_scale = 8,
                                                                             int64_t reference_latent_start = -1) {
    GGML_ASSERT(width > 0 && height > 0 && latent_frames > 0 && reference_frames > 0 && fps > 0);
    sd::Tensor<float> positions({2, 3, width * height * (latent_frames + reference_frames), 1});
    int64_t token = 0;
    for (int64_t t = 0; t < latent_frames; ++t) {
        const float t_start = ltxv_latent_corner_to_pixel_frame(latent_start + t, temporal_scale, true) /
                              static_cast<float>(fps);
        const float t_end = ltxv_latent_corner_to_pixel_frame(latent_start + t + 1, temporal_scale, true) /
                            static_cast<float>(fps);
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                set_ltxv_video_position(&positions, token++, t_start, t_end,
                                        static_cast<float>(h * spatial_scale),
                                        static_cast<float>((h + 1) * spatial_scale),
                                        static_cast<float>(w * spatial_scale),
                                        static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    // The legacy continuation/I2V guide keeps its established frame-zero
    // full-interval convention (the -1 default). A rolling appearance anchor
    // instead carries the absolute location at which its low-motion frame was
    // selected, so it remains a late local handoff rather than being
    // reinterpreted as a global first-frame portrait.
    for (int64_t t = 0; t < reference_frames; ++t) {
        const float t_start = reference_latent_start < 0
                                  ? static_cast<float>(t * temporal_scale) / static_cast<float>(fps)
                                  : ltxv_latent_corner_to_pixel_frame(reference_latent_start + t, temporal_scale, true) /
                                        static_cast<float>(fps);
        const float t_end = reference_latent_start < 0
                                ? static_cast<float>((t + 1) * temporal_scale) / static_cast<float>(fps)
                                : ltxv_latent_corner_to_pixel_frame(reference_latent_start + t + 1, temporal_scale, true) /
                                      static_cast<float>(fps);
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                set_ltxv_video_position(&positions, token++, t_start, t_end,
                                        static_cast<float>(h * spatial_scale),
                                        static_cast<float>((h + 1) * spatial_scale),
                                        static_cast<float>(w * spatial_scale),
                                        static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    return positions;
}

// The multi-keyframe conditioner is a set of single-frame image guides at
// arbitrary *pixel-frame* positions, not a contiguous continuation tail.  A
// temporal refine tile keeps its target on the absolute clip timeline and
// appends those guides at the same timeline locations used by stage one.
static sd::Tensor<float> build_ltxav_window_video_positions_with_keyframes(int64_t width,
                                                                            int64_t height,
                                                                            int64_t latent_start,
                                                                            int64_t latent_frames,
                                                                            const std::vector<int>& keyframe_frame_indices,
                                                                            int fps,
                                                                            int spatial_scale,
                                                                            int temporal_scale = 8) {
    GGML_ASSERT(width > 0 && height > 0 && latent_frames > 0 && fps > 0);
    const int64_t n_keyframes = static_cast<int64_t>(keyframe_frame_indices.size());
    GGML_ASSERT(n_keyframes > 0);
    sd::Tensor<float> positions({2, 3, width * height * (latent_frames + n_keyframes), 1});
    int64_t token = 0;
    for (int64_t t = 0; t < latent_frames; ++t) {
        const float t_start = ltxv_latent_corner_to_pixel_frame(latent_start + t, temporal_scale, true) /
                              static_cast<float>(fps);
        const float t_end = ltxv_latent_corner_to_pixel_frame(latent_start + t + 1, temporal_scale, true) /
                            static_cast<float>(fps);
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                set_ltxv_video_position(&positions, token++, t_start, t_end,
                                        static_cast<float>(h * spatial_scale),
                                        static_cast<float>((h + 1) * spatial_scale),
                                        static_cast<float>(w * spatial_scale),
                                        static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    for (int frame_idx : keyframe_frame_indices) {
        const float t_start = static_cast<float>(frame_idx) / static_cast<float>(fps);
        const float t_end   = static_cast<float>(frame_idx + 1) / static_cast<float>(fps);
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                set_ltxv_video_position(&positions, token++, t_start, t_end,
                                        static_cast<float>(h * spatial_scale),
                                        static_cast<float>((h + 1) * spatial_scale),
                                        static_cast<float>(w * spatial_scale),
                                        static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    return positions;
}

static sd::Tensor<float> build_ltxav_window_audio_positions(int64_t audio_start, int64_t audio_frames) {
    std::vector<float> positions(static_cast<size_t>(audio_frames));
    for (int64_t t = 0; t < audio_frames; ++t) {
        positions[static_cast<size_t>(t)] = LTXV::audio_latent_start_time_sec(audio_start + t);
    }
    return sd::Tensor<float>({audio_frames}, positions);
}

static sd::Tensor<float> build_ltxv_video_positions(int64_t width,
                                                    int64_t height,
                                                    int64_t target_latent_frames,
                                                    int64_t keyframe_latent_frames,
                                                    int keyframe_frame_idx,
                                                    int keyframe_pixel_frames,
                                                    int fps,
                                                    int spatial_scale,
                                                    int temporal_scale,
                                                    bool causal_temporal_positioning) {
    GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0);
    GGML_ASSERT(keyframe_latent_frames > 0);
    GGML_ASSERT(fps > 0);

    int64_t total_tokens = width * height * (target_latent_frames + keyframe_latent_frames);
    sd::Tensor<float> positions({2, 3, total_tokens, 1});
    int64_t token = 0;

    for (int64_t t = 0; t < target_latent_frames; t++) {
        float t_start = ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        float t_end   = ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        for (int64_t h = 0; h < height; h++) {
            float h_start = static_cast<float>(h * spatial_scale);
            float h_end   = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; w++) {
                float w_start = static_cast<float>(w * spatial_scale);
                float w_end   = static_cast<float>((w + 1) * spatial_scale);
                set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
            }
        }
    }

    // [SEAM 2026-07-17] The appended block's temporal grid MUST match the target's, or the guide
    // claims coordinates for content that lives elsewhere on the timeline.
    //
    // The TARGET loop above maps latent corner t through ltxv_latent_corner_to_pixel_frame(), which
    // under causal_temporal_positioning is max(0, 8t-7): frame 0 -> [0,1), frame 1 -> [1,9),
    // frame 2 -> [9,17). This loop instead used a UNIFORM stride-8 grid: frame 0 -> [0,8),
    // frame 1 -> [8,16), frame 2 -> [16,24) — ignoring causal_temporal_positioning entirely. So a
    // CONTINUATION guide (a K-latent-frame VIDEO tail depicting exactly the target's overlap
    // frames) was placed up to temporal_scale-1 = 7 pixel frames LATER than the content it depicts.
    // Candidate mechanism for the measured "anchor holds ~3 frames then falls off a cliff", and for
    // the echo/ghost that got LTXAV_CONT_LEGACY_HEAD rejected.
    //
    // keyframe_pixel_frames == 1 is a genuine single-INSTANT image pin (i2v / Director keyframe):
    // one pixel frame at its own index, no causal span. That path is unchanged.
    //
    // Env-gated for A/B; unset => the historical uniform grid, byte-identical. VALUE-gated, never
    // presence-gated (compose "${VAR:-}" yields an empty string and getenv returns non-null for it).
    const char* guide_causal_env = std::getenv("LTXAV_GUIDE_CAUSAL_POS");
    const bool  guide_causal     = guide_causal_env != nullptr && guide_causal_env[0] == '1' &&
                                   guide_causal_env[1] == '\0' && keyframe_pixel_frames != 1;
    for (int64_t t = 0; t < keyframe_latent_frames; t++) {
        float t_start, t_end;
        if (guide_causal) {
            t_start = keyframe_frame_idx +
                      ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning);
            t_end   = keyframe_frame_idx +
                      ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning);
        } else {
            t_start = static_cast<float>(keyframe_frame_idx + t * temporal_scale);
            t_end   = static_cast<float>(keyframe_frame_idx + (t + 1) * temporal_scale);
            if (keyframe_pixel_frames == 1) {
                t_end = t_start + 1.f;
            }
        }
        t_start /= static_cast<float>(fps);
        t_end /= static_cast<float>(fps);
        for (int64_t h = 0; h < height; h++) {
            float h_start = static_cast<float>(h * spatial_scale);
            float h_end   = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; w++) {
                float w_start = static_cast<float>(w * spatial_scale);
                float w_end   = static_cast<float>((w + 1) * spatial_scale);
                set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
            }
        }
    }

    return positions;
}

// LTX-2.3 MULTI-KEYFRAME positions. Generalises build_ltxv_video_positions (target block + a
// single keyframe block) to N single-latent-frame keyframe blocks, each pinned at its own target
// latent frame index. Token order is: the full target (denoised) block first, then one 1-frame
// keyframe block per conditioning image in keyframe_frame_indices order — matching the init_latent
// concat order in the caller so positions line up token-for-token. Each keyframe uses a single
// pixel-frame temporal extent ([idx, idx+1)/fps), identical to build_ltxv_video_positions' keyframe
// block with keyframe_pixel_frames==1.
static sd::Tensor<float> build_ltxv_multi_keyframe_video_positions(int64_t width,
                                                                   int64_t height,
                                                                   int64_t target_latent_frames,
                                                                   const std::vector<int>& keyframe_frame_indices,
                                                                   int fps,
                                                                   int spatial_scale,
                                                                   int temporal_scale,
                                                                   bool causal_temporal_positioning) {
    GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0);
    GGML_ASSERT(fps > 0);

    int64_t n_keyframes  = static_cast<int64_t>(keyframe_frame_indices.size());
    int64_t total_tokens = width * height * (target_latent_frames + n_keyframes);
    sd::Tensor<float> positions({2, 3, total_tokens, 1});
    int64_t token = 0;

    for (int64_t t = 0; t < target_latent_frames; t++) {
        float t_start = ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        float t_end   = ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        for (int64_t h = 0; h < height; h++) {
            float h_start = static_cast<float>(h * spatial_scale);
            float h_end   = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; w++) {
                float w_start = static_cast<float>(w * spatial_scale);
                float w_end   = static_cast<float>((w + 1) * spatial_scale);
                set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
            }
        }
    }

    for (int64_t k = 0; k < n_keyframes; k++) {
        float t_start = static_cast<float>(keyframe_frame_indices[k]);
        float t_end   = t_start + 1.f;  // single pixel-frame extent (keyframe_pixel_frames == 1)
        t_start /= static_cast<float>(fps);
        t_end /= static_cast<float>(fps);
        for (int64_t h = 0; h < height; h++) {
            float h_start = static_cast<float>(h * spatial_scale);
            float h_end   = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; w++) {
                float w_start = static_cast<float>(w * spatial_scale);
                float w_end   = static_cast<float>((w + 1) * spatial_scale);
                set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
            }
        }
    }

    return positions;
}

// One appended guide block for the mixed continuation+keyframe path: a run of `latent_frames`
// latent frames placed starting at pixel-frame `frame_idx`. `pixel_frames==1` gives each frame a
// single-instant extent (still image / identity keyframe); otherwise each frame spans
// `temporal_scale` pixel frames (a real motion window, e.g. the continuation overlap tail).
struct LtxvGuideSpec {
    int frame_idx;
    int latent_frames;
    int pixel_frames;
};

// LTX-2.3 positions for a target block + N heterogeneous appended guide blocks (Director v2:
// continuation motion tail AND frozen identity keyframes on the SAME shot). Generalises
// build_ltxv_video_positions / build_ltxv_multi_keyframe_video_positions to per-guide temporal
// spans — the motion tail needs `temporal_scale`-wide frames while image keyframes are instants.
static sd::Tensor<float> build_ltxv_guides_video_positions(int64_t width,
                                                           int64_t height,
                                                           int64_t target_latent_frames,
                                                           const std::vector<LtxvGuideSpec>& guides,
                                                           int fps,
                                                           int spatial_scale,
                                                           int temporal_scale,
                                                           bool causal_temporal_positioning) {
    GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0 && fps > 0);
    int64_t guide_frames = 0;
    for (const auto& g : guides) {
        guide_frames += g.latent_frames;
    }
    int64_t           total_tokens = width * height * (target_latent_frames + guide_frames);
    sd::Tensor<float> positions({2, 3, total_tokens, 1});
    int64_t           token = 0;

    for (int64_t t = 0; t < target_latent_frames; t++) {
        float t_start = ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        float t_end   = ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        for (int64_t h = 0; h < height; h++) {
            float h_start = static_cast<float>(h * spatial_scale);
            float h_end   = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; w++) {
                float w_start = static_cast<float>(w * spatial_scale);
                float w_end   = static_cast<float>((w + 1) * spatial_scale);
                set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
            }
        }
    }

    for (const auto& g : guides) {
        for (int64_t t = 0; t < g.latent_frames; t++) {
            float t_start = static_cast<float>(g.frame_idx + t * temporal_scale);
            float t_end   = static_cast<float>(g.frame_idx + (t + 1) * temporal_scale);
            if (g.pixel_frames == 1) {
                t_end = t_start + 1.f;
            }
            t_start /= static_cast<float>(fps);
            t_end /= static_cast<float>(fps);
            for (int64_t h = 0; h < height; h++) {
                float h_start = static_cast<float>(h * spatial_scale);
                float h_end   = static_cast<float>((h + 1) * spatial_scale);
                for (int64_t w = 0; w < width; w++) {
                    float w_start = static_cast<float>(w * spatial_scale);
                    float w_end   = static_cast<float>((w + 1) * spatial_scale);
                    set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
                }
            }
        }
    }

    return positions;
}

// LTX-2.3 V2V LIPDUB RELIP positions. The appended reference-clip tokens occupy the SAME
// timeline coordinates as the target tokens (1:1 frame overlap; reference_downscale_factor=1
// and reference_temporal_scale=1 per the lipdub-0.9 IC-LoRA metadata), so the model copies the
// reference appearance + motion per-frame while the frozen driving-audio latent moves the mouth.
// Both the target and reference blocks use the identical causal corner mapping; with equal frame
// counts the reference token positions exactly mirror the target token positions. This differs
// from build_ltxv_video_positions, whose keyframe block offsets the guide onto a separate
// (past/future) timeline for continuation.
// ref_width/ref_height/ref_spatial_scale default to the target grid's values (-1 sentinel),
// which makes the reference block IDENTICAL to the target block (full-res concat path, N==1,
// byte-identical to the original two-equal-block builder). For the separable half-res path
// (LTXAV_RELIP_REF_DOWNSCALE=N>1) the reference is encoded at W/N x H/N, so its block uses
// ref_width=W/N, ref_height=H/N, ref_spatial_scale=spatial_scale*N — the *N keeps each half-res
// reference latent cell spanning the SAME pixel extent on the shared timeline as a full-res cell.
static sd::Tensor<float> build_ltxv_relip_video_positions(int64_t width,
                                                          int64_t height,
                                                          int64_t target_latent_frames,
                                                          int64_t reference_latent_frames,
                                                          int fps,
                                                          int spatial_scale,
                                                          int temporal_scale,
                                                          bool causal_temporal_positioning,
                                                          int64_t ref_width        = -1,
                                                          int64_t ref_height       = -1,
                                                          int ref_spatial_scale    = -1,
                                                          int ref_temporal_stride  = 1) {
    if (ref_width < 0) ref_width = width;
    if (ref_height < 0) ref_height = height;
    if (ref_spatial_scale < 0) ref_spatial_scale = spatial_scale;
    if (ref_temporal_stride < 1) ref_temporal_stride = 1;
    GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0 && reference_latent_frames > 0);
    GGML_ASSERT(ref_width > 0 && ref_height > 0);
    GGML_ASSERT(fps > 0);

    int64_t total_tokens = width * height * target_latent_frames + ref_width * ref_height * reference_latent_frames;
    sd::Tensor<float> positions({2, 3, total_tokens, 1});
    int64_t token = 0;

    // tstride = the original-latent-frame step between successive emitted reference frames
    // (LTXAV_RELIP_REF_TSTRIDE). The j-th subsampled reference frame represents original latent
    // frame j*tstride, so it must carry THAT frame's temporal coordinate (a single-frame extent
    // [corner(j*tstride), corner(j*tstride+1)]) to stay timeline-aligned with the target. tstride=1
    // (target block, and ref when no temporal subsampling) = the original per-frame coordinates.
    auto emit_block = [&](int64_t bw, int64_t bh, int bscale, int btstride, int64_t frames) {
        for (int64_t t = 0; t < frames; t++) {
            int64_t src_corner = t * btstride;
            float t_start = ltxv_latent_corner_to_pixel_frame(src_corner, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
            float t_end   = ltxv_latent_corner_to_pixel_frame(src_corner + 1, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
            for (int64_t h = 0; h < bh; h++) {
                float h_start = static_cast<float>(h * bscale);
                float h_end   = static_cast<float>((h + 1) * bscale);
                for (int64_t w = 0; w < bw; w++) {
                    float w_start = static_cast<float>(w * bscale);
                    float w_end   = static_cast<float>((w + 1) * bscale);
                    set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
                }
            }
        }
    };

    emit_block(width, height, spatial_scale, 1, target_latent_frames);                            // target (denoised) tokens
    emit_block(ref_width, ref_height, ref_spatial_scale, ref_temporal_stride, reference_latent_frames);  // reference (frozen) tokens

    return positions;
}

// Character-reference identity conditioning is driven by the REQUEST (the presence of a
// character-reference image), NOT by requiring the LTXAV_CHARACTER_REF env to be SET. The old
// env-must-be-present gate left the koblem "character reference" toggle inert in prod (the field
// was sent + decoded but never attached). LTXAV_CHARACTER_REF=0 / "false" / "no" is now an
// explicit kill switch; unset (or any truthy value) keeps it enabled so a provided reference is
// honoured. Requests that provide no reference are unaffected (the callers also check that the
// character_reference latent/data is non-null), so flag-off token layout is byte-identical.
static bool ltxav_character_ref_enabled() {
    const char* v = std::getenv("LTXAV_CHARACTER_REF");
    if (v == nullptr) return true;
    return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
}

// Whether the two-stage continuation attaches the prior segment's refined VIDEO tail as a separate
// guide token block on the hires refine stage(s). Enabled unless LTXAV_CHAIN_HIRES_REFERENCE=0
// (mirrors the inline env read the chain caller uses).
static bool ltxav_chain_hires_reference_enabled() {
    const char* v = std::getenv("LTXAV_CHAIN_HIRES_REFERENCE");
    return v == nullptr || (v[0] != '\0' && std::string(v) != "0");
}

// Default ON (LTXAV_REFINE_HIRES_IDENTITY=0/false/no to disable): re-pin the init/end/keyframe
// IDENTITY at each hires-chain refine stage's FULL resolution, so the final upscale can't reroll
// the look from the low-res base pass. Stage 0 already re-pins (apply_ltxv_refine_image_conditioning);
// the later hires_chain stages (the 4x output stage) do NOT by themselves — they SDEdit the upscaled
// latent with an all-ones mask, so the frozen frame-0 image drifts. With this the final stage(s)
// re-encode + re-pin the source image at the stage resolution (a single-frame VAE encode per stage;
// VRAM-modest). Only fires for image-guide shots (i2v opener / scene cut / keyframes); continuation
// and no-image shots are unaffected, so default-on is byte-identical for them.
static bool ltxav_refine_hires_identity_enabled() {
    const char* v = std::getenv("LTXAV_REFINE_HIRES_IDENTITY");
    if (v == nullptr) return true;
    return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
}

// Append a frozen identity block to the LTXAV DiT sequence. Reference tensors are flattened
// to one token axis so a relip grid and a character-image grid may coexist without pretending
// they share spatial dimensions; RoPE retains each block's own spatial coordinates.
static bool append_ltxav_character_reference(ImageGenerationLatents* latents,
                                             const sd::Tensor<float>& character,
                                             int fps, int spatial_scale, int temporal_scale) {
    if (latents == nullptr || character.empty() || latents->init_latent.empty()) return false;
    const int64_t tw = latents->init_latent.shape()[0], th = latents->init_latent.shape()[1], tf = latents->init_latent.shape()[2];
    const int64_t cw = character.shape()[0], ch = character.shape()[1], cf = character.shape()[2];
    auto char_pos_all = build_ltxv_relip_video_positions(tw, th, tf, cf, fps, spatial_scale, temporal_scale, true, cw, ch, spatial_scale);
    const int64_t target_tokens = tw * th * tf;
    auto char_pos = sd::ops::slice(char_pos_all, 2, target_tokens, target_tokens + cw * ch * cf);
    auto char_flat = character.reshape({cw * ch * cf, 1, 1, character.shape()[3], 1});
    // Append the char identity tokens to the reference block (create it if this segment has none).
    if (latents->video_reference.empty()) {
        latents->video_reference = std::move(char_flat);
    } else {
        const auto old = latents->video_reference.reshape({latents->video_reference.shape()[0] * latents->video_reference.shape()[1] * latents->video_reference.shape()[2], 1, 1, latents->video_reference.shape()[3], 1});
        latents->video_reference = sd::ops::concat(old, char_flat, 0);
    }
    // Positions: KEY on video_positions.empty(), NOT video_reference.empty(). A CONTINUATION (or
    // keyframe) segment stores its guide frames in init_latent and leaves video_reference empty, but
    // has ALREADY built video_positions (guide frames pinned at their past-anchor RoPE slots). The
    // old code keyed on video_reference.empty() and OVERWROTE those positions with char_pos_all's
    // sequential relip layout — relabelling the guide frames from past-anchor to future-tail and
    // destroying continuation. Only replace positions when there are none yet (pure t2v: char_pos_all
    // = [target block][char block]); otherwise append ONLY the char rows, preserving the existing
    // target(+guide/relip) layout token-for-token.
    if (latents->video_positions.empty()) {
        latents->video_positions = std::move(char_pos_all);
    } else {
        latents->video_positions = sd::ops::concat(latents->video_positions, char_pos, 2);
    }
    return true;
}

// Which resolution tier of the character-identity reference a given DiT pass should attach.
// Base = the base pass (base_params->width/height ~15x8 latent grid); Lo = the first hires/refine
// stage (base*2); Hi = the later hires_chain stages (final res). The chain encodes all three once
// (VAE available) and threads them onto every segment's params.
enum class LtxavCharTier { Base, Lo, Hi };

// Materialize the character-reference latent matched to a refine stage. Picks the _lo/_hi latent
// for the Lo/Hi tiers when the caller supplied one, else FALLS BACK to the base-res latent (older
// callers, single-stage renders, or the base pass itself). Returns an empty tensor only when no
// character reference is present at all — so a no-reference request stays byte-identical.
static sd::Tensor<float> ltxav_character_latent_for_stage(const sd_vid_gen_params_t* p, LtxavCharTier tier) {
    const float* d  = p->character_reference_latent;
    int64_t      cw = p->character_reference_latent_width;
    int64_t      chh = p->character_reference_latent_height;
    int64_t      cf = p->character_reference_latent_frames;
    int64_t      cc = p->character_reference_latent_channels;
    if (tier == LtxavCharTier::Lo && p->character_reference_latent_lo != nullptr) {
        d   = p->character_reference_latent_lo;
        cw  = p->character_reference_latent_lo_width;
        chh = p->character_reference_latent_lo_height;
        cf  = p->character_reference_latent_lo_frames;
        cc  = p->character_reference_latent_lo_channels;
    } else if (tier == LtxavCharTier::Hi && p->character_reference_latent_hi != nullptr) {
        d   = p->character_reference_latent_hi;
        cw  = p->character_reference_latent_hi_width;
        chh = p->character_reference_latent_hi_height;
        cf  = p->character_reference_latent_hi_frames;
        cc  = p->character_reference_latent_hi_channels;
    }
    if (d == nullptr || cw <= 0 || chh <= 0 || cf <= 0 || cc <= 0) {
        return {};
    }
    sd::Tensor<float> character({cw, chh, cf, cc, 1});
    std::memcpy(character.data(), d, (size_t)character.numel() * sizeof(float));
    return character;
}

// FIX 1 (VRAM) — host-level TEMPORAL-CHUNKED reference encode. The LTX video-VAE encode builds a
// SINGLE graph for the whole clip: temporal-tiled streaming is DECODE-only (gated on decode_graph
// @ ltx_vae.hpp:1449; there is no encode_tiled_chunk), so the encode compute buffer scales with
// frame count (~2.6 GB @193f, temporal-dominated) and OOMs before stage-2 sampling. This slices
// the PIXEL reference into causal 8k+1-aligned temporal groups, encodes each with the caller's
// already-set spatial tiling, and concats the latent frames — bounding the buffer to ~(8*G+1)
// frames of activations regardless of clip length. The LTX VAE is causal (temporal compression 8:
// F=8k+1 -> k+1 latent frames; latent 0 = 1-frame seed, latent i>=1 = an 8-pixel-frame chunk).
// Groups after the first are prefixed with a 1-frame causal seed (the previous group's last pixel
// frame) whose latent is dropped; that seed under-provides causal history vs a full-clip encode
// (the decode path uses a multi-feature cache we lack on the encoder) — a small boundary
// approximation, harmless for a FROZEN conditioning reference. The full latent frame count is
// preserved, so this composes with the TSTRIDE latent-subsample downstream (TSTRIDE-independent).
// group_latent_chunks (G) = LTXAV_RELIP_ENCODE_TFRAMES; the helper is skipped unless that env is
// set (default path = one monolithic encode, byte-identical).
static sd::Tensor<float> encode_relip_reference_temporal_chunked(sd_ctx_t* sd_ctx,
                                                                 const sd::Tensor<float>& ref_video,
                                                                 int group_latent_chunks) {
    const int64_t F = ref_video.shape()[2];   // pixel frames (already snapped to 8k+1)
    const int64_t k = (F - 1) / 8;             // 8-frame chunks (=> k+1 latent frames)
    const int G     = std::max(1, group_latent_chunks);
    if (F <= 1 || k < 1 || static_cast<int64_t>(G) >= k) {
        // Trivial clip or group covers the whole clip => single monolithic encode (no benefit).
        return sd_ctx->sd->encode_first_stage(ref_video);
    }
    // Group 0: pixel [0, 8*g0+1) -> keeps latent [0, g0] (seed frame + g0 chunks).
    const int64_t g0      = std::min<int64_t>(G, k);
    sd::Tensor<float> out = sd_ctx->sd->encode_first_stage(sd::ops::slice(ref_video, 2, 0, 8 * g0 + 1));
    if (out.empty()) {
        return {};
    }
    int64_t done = g0;   // chunk latent frames produced so far (excluding seed latent 0)
    while (done < k) {
        const int64_t g       = std::min<int64_t>(G, k - done);
        const int64_t seed_px = 8 * done;   // last pixel frame already covered = causal seed (sub-clip frame 0)
        auto sub              = sd::ops::slice(ref_video, 2, seed_px, seed_px + 8 * g + 1);  // seed + 8g frames = 8g+1
        auto lat              = sd_ctx->sd->encode_first_stage(sub);                          // seed + g chunks = g+1 latent
        if (lat.empty() || lat.shape()[2] < g + 1) {
            LOG_WARN("LTXAV relip temporal-chunked encode: unexpected chunk latent frames %lld (want >= %lld); falling back to monolithic encode",
                     lat.empty() ? -1LL : (long long)lat.shape()[2], (long long)(g + 1));
            return sd_ctx->sd->encode_first_stage(ref_video);
        }
        out = sd::ops::concat(out, sd::ops::slice(lat, 2, 1, g + 1), 2);   // drop seed latent frame 0
        done += g;
    }
    LOG_INFO("LTXAV relip: temporal-chunked reference encode G=%d -> %lld latent frames (buffer bounded to ~%lld-frame groups)",
             G, (long long)out.shape()[2], (long long)(8 * G + 1));
    return out;
}

// Optional small temporal-budget trim for separable relip references. Unlike TSTRIDE this
// preserves every early adjacent latent frame (and therefore mouth motion) and only omits the
// final trailing frame(s), whose source appearance is already covered by the preceding causal
// VAE latent. Unset/default is exactly no-op. It is deliberately applied after TSTRIDE.
static void limit_relip_reference_latent_frames(sd::Tensor<float>* reference_latent, const char* phase) {
    if (reference_latent == nullptr || reference_latent->empty() || reference_latent->shape()[2] <= 1) {
        return;
    }
    const char* e = getenv("LTXAV_RELIP_REF_MAX_TFRAMES");
    const int max_frames = e != nullptr ? atoi(e) : 0;
    const int64_t have = reference_latent->shape()[2];
    if (max_frames > 0 && max_frames < have) {
        *reference_latent = sd::ops::slice(*reference_latent, 2, 0, max_frames);
        LOG_INFO("LTXAV relip: %s reference temporal budget %lld -> %d latent frames (trailing frame trim)",
                 phase, (long long)have, max_frames);
    }
}

static sd::Tensor<float> pack_ltxav_audio_and_video_latents(const sd::Tensor<float>& video_latent,
                                                            const sd::Tensor<float>& audio_latent) {
    if (audio_latent.empty()) {
        return video_latent;
    }

    GGML_ASSERT(video_latent.dim() == 4 || video_latent.dim() == 5);
    GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
    if (video_latent.dim() == 5) {
        GGML_ASSERT(video_latent.shape()[4] == 1);
    }
    if (audio_latent.dim() == 4) {
        GGML_ASSERT(audio_latent.shape()[3] == 1);
    }

    int64_t width        = video_latent.shape()[0];
    int64_t height       = video_latent.shape()[1];
    int64_t frames       = video_latent.shape()[2];
    int64_t video_ch     = video_latent.shape()[3];
    int64_t spatial_size = width * height * frames;
    int64_t audio_values = audio_latent.numel();
    int64_t extra_ch     = (audio_values + spatial_size - 1) / spatial_size;

    std::vector<int64_t> packed_shape = video_latent.shape();
    packed_shape[3]                   = video_ch + extra_ch;
    sd::Tensor<float> packed          = sd::zeros<float>(packed_shape);

    std::copy_n(video_latent.data(), video_latent.numel(), packed.data());
    std::copy_n(audio_latent.data(), audio_latent.numel(), packed.data() + video_latent.numel());
    return packed;
}

static sd::Tensor<float> pack_ltxav_audio_and_video_denoise_mask(const sd::Tensor<float>& video_mask,
                                                                 const sd::Tensor<float>& video_latent,
                                                                 const sd::Tensor<float>& audio_latent,
                                                                 float audio_mask_value = 1.f,
                                                                 const sd::Tensor<float>* per_audio_mask = nullptr) {
    if (video_mask.empty() || audio_latent.empty()) {
        return video_mask;
    }

    GGML_ASSERT(video_latent.dim() == 4 || video_latent.dim() == 5);
    GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
    if (video_latent.dim() == 5) {
        GGML_ASSERT(video_latent.shape()[4] == 1);
    }
    if (audio_latent.dim() == 4) {
        GGML_ASSERT(audio_latent.shape()[3] == 1);
    }

    int64_t width        = video_latent.shape()[0];
    int64_t height       = video_latent.shape()[1];
    int64_t frames       = video_latent.shape()[2];
    int64_t video_ch     = video_latent.shape()[3];
    int64_t spatial_size = width * height * frames;
    int64_t audio_values = audio_latent.numel();
    int64_t extra_ch     = (audio_values + spatial_size - 1) / spatial_size;

    GGML_ASSERT(video_mask.dim() == video_latent.dim());
    GGML_ASSERT(video_mask.shape()[0] == width);
    GGML_ASSERT(video_mask.shape()[1] == height);
    GGML_ASSERT(video_mask.shape()[2] == frames);
    if (video_mask.dim() == 5) {
        GGML_ASSERT(video_mask.shape()[4] == video_latent.shape()[4]);
    }

    int64_t mask_ch = video_mask.shape()[3];
    if (mask_ch == video_ch + extra_ch) {
        return video_mask;
    }
    GGML_ASSERT(mask_ch == 1 || mask_ch == video_ch);

    sd::Tensor<float> video_mask_full = video_mask;
    if (mask_ch == 1 && video_ch != 1) {
        video_mask_full = video_mask * sd::Tensor<float>::ones(video_latent.shape());
    }

    std::vector<int64_t> audio_mask_shape = video_latent.shape();
    audio_mask_shape[3]                   = extra_ch;
    // audio_mask_value 1.0 = denoise/generate audio (default); 0.0 = hold audio FIXED
    // (drive lip-sync to the supplied encoded wav, pinned every step via line ~2329).
    auto audio_mask                       = sd::full<float>(audio_mask_shape, audio_mask_value);
    if (per_audio_mask != nullptr && !per_audio_mask->empty()) {
        GGML_ASSERT(per_audio_mask->numel() == audio_values);
        std::copy_n(per_audio_mask->data(), audio_values, audio_mask.data());
    }
    return sd::ops::concat(video_mask_full, audio_mask, 3);
}

static sd::Tensor<float> make_ltxav_video_denoise_mask(const sd::Tensor<float>& video_latent, float value = 1.f) {
    if (video_latent.empty()) {
        return {};
    }
    return sd::full<float>({video_latent.shape()[0],
                            video_latent.shape()[1],
                            video_latent.shape()[2],
                            1,
                            1},
                           value);
}

// LTXAV_SHARED_REFINE_NOISE (value-gated, default off = byte-identical).
//
// The chain renders long video as segments whose refine re-noises to sigma0 with FRESH noise per
// segment. A continuation segment re-renders the prior segment's tail (the K-frame overlap), so that
// timeline region is denoised twice — from two INDEPENDENT noise draws. Two independent inventions
// of the same region cannot agree, which is the seam. Keying the noise to the ABSOLUTE timeline
// position makes both segments' refines see the same eps there, so they fall toward the same basin
// instead of diverging. Same idea as LTX's tiled sampler (one full_noise, sliced per tile) and
// FreeNoise/CoNo; it is a strict refinement of LTX_REFINE_CONST_SEED, which already makes the refine
// noise chain-constant but aligns it by LOCAL frame index — the overlap sits at DIFFERENT local
// indices in the two segments (seg N's last K vs seg N+1's first K), so const-seed cannot align it.
static bool ltxav_shared_refine_noise_enabled() {
    // VALUE-gated, never presence-gated: compose passes "${VAR:-}" which yields an EMPTY STRING, and
    // getenv returns non-null for it. A presence check would silently enable this on every render.
    const char* e = std::getenv("LTXAV_SHARED_REFINE_NOISE");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
}

// Chain-CONSTANT base seed for the position-keyed noise. It must NOT depend on the segment: the chain
// gives each segment seed = base + seg (generate_video_chain) precisely so base motion varies, and
// inheriting that here would hand the same absolute frame a different seed in each segment — defeating the whole
// mechanism. Following LTX_REFINE_CONST_SEED's established precedent, the value is arbitrary (only
// cross-segment consistency matters) and seed variety keeps living in the base pass.
static uint64_t ltxav_shared_refine_noise_seed() {
    if (const char* e = std::getenv("LTXAV_SHARED_REFINE_NOISE_SEED"); e != nullptr && e[0] != '\0') {
        return std::strtoull(e, nullptr, 10);
    }
    return 0x5D1F'C0DEull;
}

// splitmix64: a counter-based bijective mixer (period 2^64, full avalanche). Deliberately NOT
// STDDefaultRNG re-seeded per frame: that is minstd_rand0, a 31-bit LCG whose single ~2.1e9 orbit is
// shared by every seed, so per-frame re-seeding drops ~200 frames x ~6e5 draws onto one cycle and
// adjacent frames overlap with high probability — duplicated noise runs that would read as a temporal
// artefact, not a crash. A counter-based mixer has no orbit to collide on and stays O(1) in memory
// (the alternative, drawing one contiguous stream from frame 0 and slicing, is O(n^2) over the chain).
static inline uint64_t ltxav_splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E37'79B9'7F4A'7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58'476D'1CE4'E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D0'49BB'1331'11EBull;
    return z ^ (z >> 31);
}

// Fill `dst` with n i.i.d. N(0,1) draws deterministically derived from `key` (Box-Muller).
static void ltxav_positional_normals(float* dst, int64_t n, uint64_t key) {
    uint64_t state = key;
    // Decorrelate keys that differ only in the low bits (adjacent absolute frames).
    state          = ltxav_splitmix64(state);
    for (int64_t i = 0; i < n; i += 2) {
        // (0,1]: never 0, so log() is finite.
        const double u1 = static_cast<double>((ltxav_splitmix64(state) >> 11) + 1) * (1.0 / 9007199254740993.0);
        const double u2 = static_cast<double>(ltxav_splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
        const double r  = std::sqrt(-2.0 * std::log(u1));
        const double th = 2.0 * M_PI * u2;
        dst[i]          = static_cast<float>(r * std::cos(th));
        if (i + 1 < n) {
            dst[i + 1] = static_cast<float>(r * std::sin(th));
        }
    }
}

// Position-keyed replacement for Tensor::randn_like at the refine sites.
//
// `like` is the refine's [W, H, T, C, 1] latent, possibly AV-PACKED (audio occupies the channels above
// video_channels as a flat blob — pack_ltxav_audio_and_video_latents, sd:5658). Layout is verified from
// Tensor::offset_of (core/tensor.hpp:308-325): stride starts at 1 and multiplies by shape_[i] as i
// ascends, so shape[0] is FASTEST-varying and (t,c) is a contiguous W*H plane at flat offset
// (c*T + t)*W*H.
//
// Frame -> absolute index follows the tensor's TWO-BLOCK layout: frames [0, target_frames) are this
// segment's own output at abs frame_offset + t, while any trailing guide block appended by
// apply_ltxav_video_guide_by_keyframe_index is the PRIOR segment's tail carrying RoPE
// keyframe_frame_idx=0 — i.e. it truly sits at the segment START, abs frame_offset + (t - target_frames).
// Keying it there makes seg N+1's guide agree with seg N's own tail frames at the same absolute time.
//
// AUDIO is left on `rng` (one draw for the whole trailing region, video-frame-count-keyed rather than
// timeline-keyed): the packed audio blob has its own temporal geometry and its own known seam bug, and
// position-keying it would be a guess. Unchanged-from-today behaviour for audio, no silent corruption.
static sd::Tensor<float> positional_randn_like(const sd::Tensor<float>& like,
                                               const std::shared_ptr<RNG>& rng,
                                               uint64_t seed,
                                               int64_t frame_offset,
                                               int64_t video_channels,
                                               int64_t target_frames) {
    GGML_ASSERT(like.dim() == 5);
    const int64_t W = like.shape()[0];
    const int64_t H = like.shape()[1];
    const int64_t T = like.shape()[2];
    const int64_t C = like.shape()[3];
    GGML_ASSERT(video_channels > 0 && video_channels <= C);
    GGML_ASSERT(target_frames > 0 && target_frames <= T);

    sd::Tensor<float> out(like.shape());
    const int64_t plane = W * H;
    std::vector<float> frame_noise(static_cast<size_t>(plane * video_channels));
    for (int64_t t = 0; t < T; ++t) {
        const int64_t abs_t = frame_offset + (t < target_frames ? t : t - target_frames);
        ltxav_positional_normals(frame_noise.data(), plane * video_channels,
                                 seed ^ (static_cast<uint64_t>(abs_t) * 0x9E37'79B9'7F4A'7C15ull));
        // Scatter this frame's draws across the channel-strided planes.
        for (int64_t c = 0; c < video_channels; ++c) {
            std::copy_n(frame_noise.data() + c * plane, plane, out.data() + (c * T + t) * plane);
        }
    }
    const int64_t audio_start = plane * T * video_channels;
    const int64_t audio_n     = out.numel() - audio_start;
    if (audio_n > 0) {
        std::vector<float> audio_noise = rng->randn(static_cast<uint32_t>(audio_n));
        std::copy_n(audio_noise.data(), audio_n, out.data() + audio_start);
    }
    return out;
}

static sd::Tensor<float> encode_ltxav_condition_image(sd_ctx_t* sd_ctx,
                                                      const sd::Tensor<float>& image,
                                                      const char* name) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image.empty()) {
        return {};
    }
    auto condition_image  = image.reshape({image.shape()[0],
                                           image.shape()[1],
                                           1,
                                           image.shape()[2],
                                           image.shape()[3]});
    auto condition_latent = sd_ctx->sd->encode_first_stage(condition_image);
    if (condition_latent.empty()) {
        LOG_ERROR("failed to encode LTXAV %s image", name);
    }
    return condition_latent;
}

static bool apply_ltxav_condition_by_latent_index(sd::Tensor<float>* video_latent,
                                                  sd::Tensor<float>* video_mask,
                                                  const sd::Tensor<float>& condition_latent,
                                                  int64_t latent_idx,
                                                  const char* name,
                                                  float conditioned_mask) {
    if (video_latent == nullptr || video_mask == nullptr || video_latent->empty() || video_mask->empty()) {
        return false;
    }
    if (condition_latent.empty() ||
        condition_latent.shape()[0] != video_latent->shape()[0] ||
        condition_latent.shape()[1] != video_latent->shape()[1] ||
        condition_latent.shape()[3] != video_latent->shape()[3]) {
        LOG_ERROR("invalid LTXAV %s condition latent shape", name);
        return false;
    }
    int64_t latent_frames    = video_latent->shape()[2];
    int64_t condition_frames = condition_latent.shape()[2];
    if (latent_idx < 0 || condition_frames <= 0 || latent_idx + condition_frames > latent_frames) {
        LOG_ERROR("invalid LTXAV %s image latent range: start=%" PRId64 ", length=%" PRId64 ", latent_frames=%" PRId64,
                  name,
                  latent_idx,
                  condition_frames,
                  latent_frames);
        return false;
    }

    sd::ops::slice_assign(video_latent, 2, latent_idx, latent_idx + condition_frames, condition_latent);
    sd::ops::fill_slice(video_mask, 2, latent_idx, latent_idx + condition_frames, conditioned_mask);
    return true;
}

// LTXAV_PIN_REFINE_OVERLAP (value-gated, default off = byte-identical).
//
// THE BUG. A continuation segment's hires refine is an SDEdit at sigma0=0.85 — the OFFICIAL LTX-2.3
// recipe (Lightricks' shipped ComfyUI workflow, the "Deno" workflows, Wan2GP's
// DISTILLED_8_STEPS_STAGE_2_SIGMA_VALUES all agree), so 0.85 is not negotiable. The segment's target
// frames [0,K) ARE the prior segment's tail re-rendered (the chain drops 1+(K-1)*8 pixel frames off
// this head at stitch), yet they carry denoise mask 1.0 like every other frame — so the refine
// re-noises the SHARED overlap to 0.85 and RE-INVENTS it. Two independent inventions of one timeline
// region cannot agree: that is the seam. The prior tail rides today only as `hires_video_reference`,
// APPENDED guide tokens (clean, frozen) — a real anchor, but an advisory one that measurably decays
// within ~3 frames (frame-diff 1.6 at the cut -> 30.4 three frames later).
//
// THE FIX (what every implementation that actually runs a strong refine on chained video does — they
// PIN, they do not match noise; matching noise was measured here and made the seam WORSE, 1.14 ->
// 1.26, and StreamingT2V/Wan2GP corroborate):
//   - Wan2GP (models/ltx2/ltx2.py:1299 _append_prefix_entries) feeds the clean overlap prefix to BOTH
//     stages; VideoConditionByKeyframeIndex sets denoise_mask = 1.0 - strength with
//     input_video_strength=1.0 => mask 0 => per-token timestep 0 + a hard restore every step.
//   - Lightricks LTXVLoopingSampler: noise_mask = 1 - strength clamp on the overlap.
//   - RuneXX Extend: overlap as a known encoded latent + LTXVAddLatentGuide strength 1.0.
// We do exactly that: write the prior segment's REFINED tail into the refine's x_t over the overlap
// frames and drop the mask there to 1.0 - strength. sample() then gives those tokens timestep 0
// (process_ltxav_video_timesteps: per-token t = mask * t0) and restores them every step
// (`noised_input`/`denoised` = x * mask + init_latent * (1 - mask), :2982/:2991/:3246). The refine
// physically CANNOT re-invent the overlap; it can only harmonize the frames after it.
//
// Reuses apply_ltxav_condition_by_latent_index — the SAME helper the base path's LTXAV_CONT_LEGACY_HEAD
// uses (:8091) — rather than a parallel implementation. It fits exactly: pass the UNPACKED video latent
// and the VIDEO-ONLY mask and its shape check (shape[0]/[1]/[3] equality), slice_assign on the frame
// dim, and fill_slice of the mask over the same frames are precisely the pin.
static bool ltxav_pin_refine_overlap_enabled() {
    // VALUE-gated, NEVER presence-gated: compose passes "${VAR:-}" which yields an EMPTY STRING and
    // getenv returns non-null for it, so a presence check silently enables the feature on every
    // render. That has bitten this codebase twice — LTX_REFINE_CONTEXT_FRAMES (commented out in
    // docker-compose.yml for exactly this) and LTXAV_CHAIN_HIRES_REFERENCE_FRAMES (:12990).
    const char* e = std::getenv("LTXAV_PIN_REFINE_OVERLAP");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
}

// Pin strength, mirroring Wan2GP's `input_video_strength` and our existing LTXAV_CONT_OVERLAP_MASK /
// LTXAV_RELIP_REF_STRENGTH levers: 1.0 (default) = mask 0.0 = fully frozen overlap. Lower values let
// the refine partially re-diffuse the overlap so its texture blends with the frames after it, at the
// cost of loosening the lock. Explicit e[0] != '\0' guard so an empty "${VAR:-}" cannot atof() to 0.0
// and silently turn the pin off while the feature reports as engaged.
static float ltxav_pin_refine_overlap_strength() {
    if (const char* e = std::getenv("LTXAV_PIN_REFINE_OVERLAP_STRENGTH"); e != nullptr && e[0] != '\0') {
        return std::clamp(static_cast<float>(atof(e)), 0.f, 1.f);
    }
    return 1.0f;
}

// Pin `tail_frames` clean prior-segment refined latent frames over this refine's overlap and hold them
// at mask = 1 - strength. Returns true only when frames were actually written.
//
// `overlap_frames` is K (sd_vid_gen_params->cont_latent_frames) — the base overlap this segment
// re-renders; `tail_frames` is the transported refined tail (hires_ref_K), which may be SMALLER.
static bool ltxav_pin_refine_overlap(sd::Tensor<float>* latent,
                                     sd::Tensor<float>* mask,
                                     const float* tail_data,
                                     int64_t tail_frames,
                                     int64_t tail_w,
                                     int64_t tail_h,
                                     int64_t tail_c,
                                     int64_t video_ch,
                                     int64_t overlap_frames,
                                     int64_t target_frames,
                                     const char* stage_label) {
    if (latent == nullptr || mask == nullptr || latent->empty() || tail_data == nullptr) {
        LOG_WARN("LTXAV_PIN_REFINE_OVERLAP (%s): SKIPPED, pinned 0 frames - no refine latent or no transported tail",
                 stage_label);
        return false;
    }
    if (latent->dim() != 5 || video_ch <= 0) {
        LOG_WARN("LTXAV_PIN_REFINE_OVERLAP (%s): SKIPPED, pinned 0 frames - unexpected refine latent rank %d",
                 stage_label, (int)latent->dim());
        return false;
    }
    const int64_t W     = latent->shape()[0];
    const int64_t H     = latent->shape()[1];
    const int64_t T_all = latent->shape()[2];
    const int64_t C_all = latent->shape()[3];
    // video_target_frame_count owns the [target | guide] split. Pin the TARGET's own frames only —
    // the appended guide block at [T_tgt, T_tgt+K) is a separate, already-working anchor and must not
    // be confused with the target's first K frames. (On the default path LTX_REFINE_CONTEXT_FRAMES
    // has already trimmed the guide out of x_t, so T_tgt == T_all; an explicit env value can retain
    // it, hence the clamp.)
    const int64_t T_tgt = (target_frames > 0 && target_frames < T_all) ? target_frames : T_all;

    // ALIGNMENT (a silent-corruption trap). The tail is the prior segment's LAST `tail_frames`
    // refined frames; this segment's target frames [0,K) re-render the prior segment's LAST K frames,
    // 1:1 in order. So prior frame (T_prev - K + j) == our frame j, and the tail therefore lands at
    // [K - tail_frames, K) — NOT at [0, tail_frames). With the shipped defaults K=3 and
    // hires_ref_K=min(3,K)=3 the two coincide, but a larger --cont-latent-take (or
    // LTXAV_CHAIN_HIRES_REFERENCE_FRAMES < K) makes tail_frames < K, and pinning at 0 would place the
    // prior tail (K - tail_frames) frames EARLIER than its true timeline slot — a temporal stutter
    // welded in at full strength, with no error anywhere.
    const int64_t pin_start = overlap_frames - tail_frames;
    const int64_t pin_end   = overlap_frames;

    const bool shape_ok = tail_w == W && tail_h == H && tail_c == video_ch &&
                          tail_frames > 0 && overlap_frames > 0 && tail_frames <= overlap_frames &&
                          pin_start >= 0 && pin_end <= T_tgt && C_all >= video_ch;
    if (!shape_ok) {
        // Mirrors the hires-guide gate's shape_ok WARN (:11456): never corrupt, just decline.
        LOG_WARN("LTXAV_PIN_REFINE_OVERLAP (%s): SKIPPED, pinned 0 frames - tail [w=%lld,h=%lld,t=%lld,c=%lld] vs refine "
                 "[w=%lld,h=%lld,t=%lld(target %lld),c=%lld], video_ch=%lld, K=%lld -> wanted target frames [%lld,%lld)",
                 stage_label, (long long)tail_w, (long long)tail_h, (long long)tail_frames, (long long)tail_c,
                 (long long)W, (long long)H, (long long)T_all, (long long)T_tgt, (long long)C_all,
                 (long long)video_ch, (long long)overlap_frames, (long long)pin_start, (long long)pin_end);
        return false;
    }

    // PACKED AV. pack_ltxav_audio_and_video_latents (:5659) lays the VIDEO block out as a CONTIGUOUS
    // flat prefix of W*H*T*video_ch floats with the audio blob memcpy'd after it, so slicing dim 3 at
    // [0, video_ch) and assigning that same range back is exactly a read/write of the video prefix —
    // the audio bytes are never touched. Deliberately NOT the unpack -> repack pattern used at
    // :11384: repacking rebuilds the audio mask from a flat audio_mask_value and would SILENTLY DROP
    // the LipDub per-token audio mask (make_ltxav_lipdub_audio_mask) that the relip stage-2 path
    // installs at :9881. Slice-and-assign-back preserves the audio mask bit-for-bit.
    const bool latent_packed = C_all > video_ch;
    sd::Tensor<float> video  = latent_packed ? sd::ops::slice(*latent, 3, 0, video_ch) : *latent;

    // EMPTY MASK. On the plain continuation refine apply_ltxv_refine_image_conditioning returns early
    // (:9903-9924) leaving hires_denoise_mask EMPTY — it only fills it for i2v/keyframe/relip or an
    // audio-pinned (audio_fixed) refine. sample() skips the whole mask branch on an empty mask, so
    // without creating one here the pin would write x_t and then be silently overwritten at step 1.
    // An all-ones mask is arithmetically inert for every UNPINNED frame (x*1 + init*0 == x) and for
    // the audio channels (a 1-channel mask broadcasts to 1.0 over them = "generate", which is what an
    // absent mask already meant — and audio_fixed renders never reach here with an empty mask).
    bool mask_created = mask->empty();
    bool mask_packed  = false;
    sd::Tensor<float> video_mask;
    if (mask_created) {
        video_mask = make_ltxav_video_denoise_mask(video, 1.f);
    } else if (mask->dim() != 5 || mask->shape()[0] != W || mask->shape()[1] != H || mask->shape()[2] != T_all) {
        LOG_WARN("LTXAV_PIN_REFINE_OVERLAP (%s): SKIPPED, pinned 0 frames - refine denoise mask shape mismatch "
                 "(rank %d, [%lld,%lld,%lld] vs latent [%lld,%lld,%lld])",
                 stage_label, (int)mask->dim(),
                 (long long)(mask->dim() > 0 ? mask->shape()[0] : 0),
                 (long long)(mask->dim() > 1 ? mask->shape()[1] : 0),
                 (long long)(mask->dim() > 2 ? mask->shape()[2] : 0),
                 (long long)W, (long long)H, (long long)T_all);
        return false;
    } else if (mask->shape()[3] > video_ch) {
        // Packed mask: extract the VIDEO channels only. fill_slice on the frame dim iterates every
        // channel, so pinning the packed mask directly would zero the trailing AUDIO mask channels at
        // those flat offsets and mute/freeze arbitrary audio tokens.
        video_mask  = sd::ops::slice(*mask, 3, 0, video_ch);
        mask_packed = true;
    } else {
        video_mask = *mask;
    }

    const float strength = ltxav_pin_refine_overlap_strength();
    const float pin_mask = 1.0f - strength;  // Wan2GP's input_video_strength=1.0 -> denoise_mask 0.0

    sd::Tensor<float> tail({W, H, tail_frames, video_ch, 1});
    std::memcpy(tail.data(), tail_data, (size_t)tail.numel() * sizeof(float));

    if (!apply_ltxav_condition_by_latent_index(&video, &video_mask, tail, pin_start,
                                               "refine overlap pin", pin_mask)) {
        return false;
    }

    if (latent_packed) {
        sd::ops::slice_assign(latent, 3, 0, video_ch, video);
    } else {
        *latent = std::move(video);
    }
    if (mask_packed) {
        sd::ops::slice_assign(mask, 3, 0, video_ch, video_mask);
    } else {
        *mask = std::move(video_mask);
    }

    // Log ONCE per stage when engaged. Absence of a log line has twice been mistaken here for absence
    // of the thing, so state the stage, K, the exact frames pinned, the mask value and the tail dims;
    // every decline above logs a distinct "SKIPPED, pinned 0 frames" WARN so a no-op is never silent.
    LOG_INFO("LTXAV_PIN_REFINE_OVERLAP (%s): PINNED %lld overlap frame(s) at target [%lld,%lld) of %lld (K=%lld), "
             "strength=%.3f -> denoise mask=%.3f (0=frozen), tail [w=%lld,h=%lld,t=%lld,c=%lld], refine latent "
             "[w=%lld,h=%lld,t=%lld,c=%lld] (audio channels=%lld untouched), mask=%s",
             stage_label, (long long)tail_frames, (long long)pin_start, (long long)pin_end, (long long)T_tgt,
             (long long)overlap_frames, strength, pin_mask,
             (long long)tail_w, (long long)tail_h, (long long)tail_frames, (long long)tail_c,
             (long long)W, (long long)H, (long long)T_all, (long long)C_all, (long long)(C_all - video_ch),
             mask_created ? "CREATED all-ones (was empty)"
                          : (mask_packed ? "existing packed (video channels only)" : "existing video-only"));
    return true;
}

// LTXAV CONTINUATION via the ComfyUI LTX-Director keyframe convention: append the prior
// segment's motion-carrying guide latent as EXTRA tokens at the TAIL of the sequence (NOT
// overwriting output frames 0..K), give those guide tokens their OWN true-past-timeline RoPE
// position via keyframe_frame_idx, hold them (conditioned_mask, frozen at 0), and let the rest
// of the segment denoise freely. The appended guide tokens are cropped off the output after
// sampling (video_conditioning_frame_count). This avoids the head-placement bug where the guide
// frames steal the new segment's low RoPE slots 0..K-1 and over-anchor the immediately-adjacent
// generated frames into a faded echo — an effect that gets WORSE at higher fps (the frozen
// anchors sit closer in pixel-time to the generated content). frame_idx 0 pins the guide at the
// segment start (smooth continuation); the generated output evolves rather than freezing.
static bool apply_ltxav_video_guide_by_keyframe_index(ImageGenerationLatents* latents,
                                                      const sd::Tensor<float>& guide,
                                                      int keyframe_frame_idx,
                                                      int fps,
                                                      int spatial_scale,
                                                      float conditioned_mask) {
    if (latents == nullptr || latents->init_latent.empty() || latents->denoise_mask.empty() || guide.empty()) {
        return false;
    }
    if (guide.shape()[0] != latents->init_latent.shape()[0] ||
        guide.shape()[1] != latents->init_latent.shape()[1] ||
        guide.shape()[3] != latents->init_latent.shape()[3]) {
        LOG_ERROR("invalid LTXAV continuation guide latent shape");
        return false;
    }
    int64_t keyframe_frames                 = guide.shape()[2];
    latents->video_target_frame_count       = latents->init_latent.shape()[2];
    latents->video_conditioning_frame_count = keyframe_frames;
    latents->init_latent                    = sd::ops::concat(latents->init_latent, guide, 2);

    auto keyframe_mask = sd::full<float>({guide.shape()[0], guide.shape()[1], keyframe_frames, 1, 1}, conditioned_mask);
    latents->denoise_mask = sd::ops::concat(latents->denoise_mask, keyframe_mask, 2);
    // keyframe_pixel_frames must be != 1 so each guide latent frame spans a full temporal_scale
    // window (real video motion), not the single-pixel image-keyframe convention.
    latents->video_positions = build_ltxv_video_positions(latents->init_latent.shape()[0],
                                                          latents->init_latent.shape()[1],
                                                          latents->video_target_frame_count,
                                                          keyframe_frames,
                                                          keyframe_frame_idx,
                                                          /*keyframe_pixel_frames*/ 8,
                                                          fps,
                                                          spatial_scale,
                                                          8,
                                                          true);
    return true;
}

// LTX-2.3 V2V LIPDUB RELIP (IC-LoRA). Append the FULL reference clip's VAE latents as EXTRA
// CLEAN tokens at the tail of the sequence, occupying the SAME timeline positions as the target
// frames (1:1 overlap). The reference tokens are held FROZEN (denoise mask 0) so the lipdub
// IC-LoRA copies the reference appearance + motion per-frame while the frozen driving-audio
// latent (--drive-audio, audio_fixed) drives the mouth. The appended tokens are cropped off the
// output after sampling (video_conditioning_frame_count). Reuses the continuation machinery
// (concat tail, frozen mask, crop) but with timeline-ALIGNED positions instead of an offset.
static bool apply_ltxav_video_relip_reference(ImageGenerationLatents* latents,
                                              const sd::Tensor<float>& reference,
                                              int fps,
                                              int spatial_scale) {
    if (latents == nullptr || latents->init_latent.empty() || latents->denoise_mask.empty() || reference.empty()) {
        return false;
    }
    if (reference.shape()[0] != latents->init_latent.shape()[0] ||
        reference.shape()[1] != latents->init_latent.shape()[1] ||
        reference.shape()[3] != latents->init_latent.shape()[3]) {
        LOG_ERROR("invalid LTXAV relip reference latent shape");
        return false;
    }
    int64_t reference_frames                = reference.shape()[2];
    latents->video_target_frame_count       = latents->init_latent.shape()[2];
    latents->video_conditioning_frame_count = reference_frames;
    if (reference_frames != latents->video_target_frame_count) {
        // Not fatal — positions still align per-index — but 1:1 relip wants equal counts.
        LOG_WARN("LTXAV relip: reference latent frames %lld != target %lld (expect 1:1 overlap)",
                 (long long)reference_frames, (long long)latents->video_target_frame_count);
    }
    latents->init_latent = sd::ops::concat(latents->init_latent, reference, 2);

    // Reference conditioning strength (the official IC-LoRA `reference_strength`, default 1.0):
    // denoise mask = 1 - strength. strength 1.0 = fully frozen crisp reference — faithful but,
    // in our SINGLE full-res from-noise pass (the official runs a two-stage half-res→refine),
    // it pins every frame to its source pixels at the same spacetime coord: that clamps the
    // mouth shut (a2v can't overpower it) and smears moving content into a temporal echo.
    // Lowering strength (~0.7-0.9) renoises the reference tokens a little each step so the lock
    // loosens — the audio can open the mouth and the double-exposure eases, at some cost to
    // identity/scene fidelity. Env LTXAV_RELIP_REF_STRENGTH (default 1.0 = legacy frozen).
    float ref_strength = 1.0f;
    if (const char* e = std::getenv("LTXAV_RELIP_REF_STRENGTH")) {
        ref_strength = std::clamp((float)atof(e), 0.0f, 1.0f);
    }
    float ref_mask_val = 1.0f - ref_strength;
    if (ref_strength != 1.0f) {
        LOG_INFO("LTXAV relip: reference_strength=%.2f (reference denoise mask=%.2f; <1 loosens the frozen reference)",
                 ref_strength, ref_mask_val);
    }
    auto reference_mask   = sd::full<float>({reference.shape()[0], reference.shape()[1], reference_frames, 1, 1}, ref_mask_val);
    latents->denoise_mask = sd::ops::concat(latents->denoise_mask, reference_mask, 2);
    latents->video_positions = build_ltxv_relip_video_positions(latents->init_latent.shape()[0],
                                                                latents->init_latent.shape()[1],
                                                                latents->video_target_frame_count,
                                                                reference_frames,
                                                                fps,
                                                                spatial_scale,
                                                                8,
                                                                true);
    return true;
}

static bool apply_ltxav_condition_image_by_latent_index(sd_ctx_t* sd_ctx,
                                                        const sd::Tensor<float>& image,
                                                        sd::Tensor<float>* video_latent,
                                                        sd::Tensor<float>* video_mask,
                                                        int64_t latent_idx,
                                                        const char* name,
                                                        float strength) {
    auto condition_latent = encode_ltxav_condition_image(sd_ctx, image, name);
    return !condition_latent.empty() &&
           apply_ltxav_condition_by_latent_index(video_latent,
                                                 video_mask,
                                                 condition_latent,
                                                 latent_idx,
                                                 name,
                                                 1.0f - std::clamp(strength, 0.f, 1.f));
}

static sd::Tensor<float> unpack_ltxav_audio_latent(const sd::Tensor<float>& packed_latent,
                                                   int audio_length,
                                                   int video_channels) {
    if (packed_latent.empty() || audio_length <= 0) {
        return {};
    }

    GGML_ASSERT(packed_latent.dim() == 4 || packed_latent.dim() == 5);
    int64_t width          = packed_latent.shape()[0];
    int64_t height         = packed_latent.shape()[1];
    int64_t frames         = packed_latent.shape()[2];
    int64_t total_channels = packed_latent.shape()[3];
    int64_t spatial_size   = width * height * frames;
    if (total_channels <= video_channels) {
        return {};
    }

    constexpr int kLtxavAudioFrequencyBins = 16;
    constexpr int kLtxavAudioChannels      = 8;
    int64_t required_values                = static_cast<int64_t>(audio_length) * kLtxavAudioFrequencyBins * kLtxavAudioChannels;
    int64_t packed_values                  = (total_channels - video_channels) * spatial_size;
    if (packed_values < required_values) {
        return {};
    }

    sd::Tensor<float> audio_latent({kLtxavAudioFrequencyBins, audio_length, kLtxavAudioChannels, 1});
    const float* audio_src = packed_latent.data() + static_cast<size_t>(video_channels) * static_cast<size_t>(spatial_size);
    std::copy_n(audio_src, static_cast<size_t>(required_values), audio_latent.data());
    return audio_latent;
}

static sd::Tensor<float> make_ltxav_empty_audio_latent(int audio_length) {
    if (audio_length <= 0) {
        return {};
    }
    constexpr int kLtxavAudioFrequencyBins = 16;
    constexpr int kLtxavAudioChannels      = 8;
    return sd::zeros<float>({kLtxavAudioFrequencyBins, audio_length, kLtxavAudioChannels, 1});
}

// Official LipDub audio conditioning uses `[target | clean reference]` tokens.
// Target positions start at zero; the appended reference is shifted into negative
// time so it is an unambiguous conditioning block rather than a replacement for
// the target audio timeline.
static sd::Tensor<float> build_ltxav_lipdub_audio_positions(int target_length, int reference_length) {
    const int total = target_length + reference_length;
    if (total <= 0) return {};
    std::vector<float> positions(static_cast<size_t>(total), 0.f);
    for (int t = 0; t < target_length; ++t) {
        positions[static_cast<size_t>(t)] = LTXV::audio_latent_start_time_sec(t);
    }
    const float ref_duration = reference_length > 0 ? LTXV::audio_latent_start_time_sec(reference_length - 1) : 0.f;
    for (int t = 0; t < reference_length; ++t) {
        positions[static_cast<size_t>(target_length + t)] = LTXV::audio_latent_start_time_sec(t) - ref_duration - 0.04f;
    }
    return sd::Tensor<float>({total}, positions);
}

static sd::Tensor<float> resize_ltxav_audio_latent(const sd::Tensor<float>& audio_latent,
                                                   int target_audio_length) {
    auto resized = make_ltxav_empty_audio_latent(target_audio_length);
    if (resized.empty() || audio_latent.empty()) {
        return resized;
    }
    GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
    int copy_length = std::min(static_cast<int>(audio_latent.shape()[1]), target_audio_length);
    if (copy_length > 0) {
        auto copied = sd::ops::slice(audio_latent, 1, 0, copy_length);
        sd::ops::slice_assign(&resized, 1, 0, copy_length, copied);
    }
    return resized;
}

static sd::Tensor<float> make_ltxav_lipdub_audio_mask(const sd::Tensor<float>& audio_latent,
                                                       int target_length,
                                                       bool freeze_target) {
    if (audio_latent.empty()) {
        return {};
    }
    auto mask = sd::full<float>(audio_latent.shape(), 1.f);
    const int64_t target_values = static_cast<int64_t>(std::max(0, target_length)) * 16 * 8;
    if (freeze_target) {
        std::fill(mask.data(), mask.data() + mask.numel(), 0.f);
    } else if (target_values < mask.numel()) {
        std::fill(mask.data() + target_values, mask.data() + mask.numel(), 0.f);
    }
    return mask;
}

// LTXAV audio-DRIVE: load a 16kHz wav, run it through the audio-VAE encoder, and lay the
// result out as the packed audio latent [freq=16, audio_length, chan=8, 1] so it can be
// spliced into the joint AV latent and held fixed. Returns {} (caller falls back to
// generated/zeros audio) if there's no encoder or the wav can't be read.
static sd::Tensor<float> encode_ltxav_drive_audio(sd_ctx_t* sd_ctx, const char* wav_path, int audio_length) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || wav_path == nullptr || wav_path[0] == '\0' || audio_length <= 0) {
        return {};
    }
    auto& avae = sd_ctx->sd->audio_vae_model;
    if (!avae || !avae->config.has_encoder) {
        LOG_ERROR("--drive-audio needs an audio VAE with the encoder + mel basis "
                  "(build the -ENC gguf via tools/convert_ltx_audio_vae.py --with-encoder)");
        return {};
    }
    std::vector<float> wav;
    if (!LONGCAT_AUDIO::load_wav_16k_mono(wav_path, wav) || wav.empty()) {
        LOG_ERROR("failed to load drive audio wav: %s", wav_path);
        return {};
    }
    // encoder expects waveform ne [samples, channels]; mono is duplicated to its 2 channels.
    sd::Tensor<float> waveform({static_cast<int64_t>(wav.size()), 1}, wav);
    auto enc = avae->encode(sd_ctx->sd->n_threads, waveform);  // [128, T_enc, (1)]
    if (enc.empty()) {
        LOG_ERROR("audio VAE encode failed for %s", wav_path);
        return {};
    }
    constexpr int kF = 16;  // frequency bins (fastest in the 128 feature axis)
    constexpr int kC = 8;   // latent channels
    GGML_ASSERT(enc.shape()[0] == kF * kC);
    int64_t T_enc = enc.shape()[1];
    // [128 = (freq16 fastest, chan8), T_enc] -> [freq16, audio_length, chan8, 1] (zero-pad/trim T).
    auto out          = make_ltxav_empty_audio_latent(audio_length);
    float* dst        = out.data();
    const float* src  = enc.data();
    int64_t feat      = enc.shape()[0];
    int64_t Tcopy     = std::min<int64_t>(T_enc, audio_length);
    for (int64_t c = 0; c < kC; ++c) {
        for (int64_t t = 0; t < Tcopy; ++t) {
            for (int64_t f = 0; f < kF; ++f) {
                dst[c * static_cast<int64_t>(kF) * audio_length + t * kF + f] = src[t * feat + c * kF + f];
            }
        }
    }
    LOG_INFO("LTXAV audio-drive: encoded %s (%zu samples) -> audio latent T_enc=%" PRId64 " -> %d (lip-sync target)",
             wav_path, wav.size(), T_enc, audio_length);
    // LTX_NAN_DEBUG: scan the encoded + packed audio latent so the audio's role in any NaN is
    // visible (e.g. a silent/clipped slice → extreme encode → tips the joint forward to NaN).
    if (getenv("LTX_NAN_DEBUG") != nullptr) {
        auto scan = [](const sd::Tensor<float>& t, const char* tag) {
            const float* d = t.data();
            int64_t n      = t.numel();
            long nn = 0, ni = 0;
            float mx = -1e30f, mn = 1e30f;
            double sum = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                float v = d[i];
                if (std::isnan(v)) { nn++; } else if (std::isinf(v)) { ni++; } else { if (v > mx) mx = v; if (v < mn) mn = v; sum += v; }
            }
            LOG_INFO("[LTX_NAN] audio %s n=%lld nan=%ld inf=%ld range=%.3f..%.3f mean=%.4f", tag, (long long)n, nn, ni, mn, mx, n ? sum / n : 0.0);
        };
        scan(enc, "enc ");
        scan(out, "pack");
    }
    return out;
}

static int get_ltxav_num_audio_latents(int frames, int fps) {
    GGML_ASSERT(frames > 0);
    GGML_ASSERT(fps > 0);
    constexpr float kSampleRate            = 16000.0f;
    constexpr float kMelHopLength          = 160.0f;
    constexpr float kAudioLatentDownsample = 4.0f;
    constexpr float kLatentsPerSecond      = kSampleRate / kMelHopLength / kAudioLatentDownsample;
    return static_cast<int>(std::ceil((static_cast<float>(frames) / static_cast<float>(fps)) * kLatentsPerSecond));
}

struct ImageGenerationEmbeds {
    SDCondition cond;
    SDCondition uncond;
    SDCondition img_uncond;
};

struct CircularAxesState {
    bool circular_x = false;
    bool circular_y = false;
};

static CircularAxesState configure_image_vae_axes(sd_ctx_t* sd_ctx,
                                                  const sd_img_gen_params_t* sd_img_gen_params,
                                                  const GenerationRequest& request) {
    CircularAxesState original_axes = {sd_ctx->sd->circular_x, sd_ctx->sd->circular_y};

    if (!sd_img_gen_params->vae_tiling_params.enabled) {
        if (sd_ctx->sd->first_stage_model) {
            sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        if (sd_ctx->sd->preview_vae) {
            sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        return original_axes;
    }

    int tile_size_x, tile_size_y;
    float overlap;
    int latent_size_x = request.width / request.vae_scale_factor;
    int latent_size_y = request.height / request.vae_scale_factor;
    sd_ctx->sd->first_stage_model->get_tile_sizes(tile_size_x,
                                                  tile_size_y,
                                                  overlap,
                                                  sd_img_gen_params->vae_tiling_params,
                                                  latent_size_x,
                                                  latent_size_y);

    sd_ctx->sd->circular_x = sd_ctx->sd->circular_x && (tile_size_x >= latent_size_x);
    sd_ctx->sd->circular_y = sd_ctx->sd->circular_y && (tile_size_y >= latent_size_y);

    if (sd_ctx->sd->first_stage_model) {
        sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }
    if (sd_ctx->sd->preview_vae) {
        sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }

    sd_ctx->sd->circular_x = original_axes.circular_x && (tile_size_x < latent_size_x);
    sd_ctx->sd->circular_y = original_axes.circular_y && (tile_size_y < latent_size_y);

    return original_axes;
}

static void restore_image_vae_axes(sd_ctx_t* sd_ctx, const CircularAxesState& original_axes) {
    sd_ctx->sd->circular_x = original_axes.circular_x;
    sd_ctx->sd->circular_y = original_axes.circular_y;
}

class ImageVaeAxesGuard {
private:
    sd_ctx_t* sd_ctx = nullptr;
    CircularAxesState original_axes;

public:
    ImageVaeAxesGuard(sd_ctx_t* sd_ctx,
                      const sd_img_gen_params_t* sd_img_gen_params,
                      const GenerationRequest& request)
        : sd_ctx(sd_ctx),
          original_axes(configure_image_vae_axes(sd_ctx, sd_img_gen_params, request)) {}

    ~ImageVaeAxesGuard() {
        restore_image_vae_axes(sd_ctx, original_axes);
    }

    ImageVaeAxesGuard(const ImageVaeAxesGuard&)            = delete;
    ImageVaeAxesGuard& operator=(const ImageVaeAxesGuard&) = delete;
};

static std::optional<ImageGenerationLatents> prepare_image_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_img_gen_params_t* sd_img_gen_params,
                                                                              GenerationRequest* request,
                                                                              SamplePlan* plan) {
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> init_image_tensor;
    sd::Tensor<float> control_image_tensor;
    sd::Tensor<float> mask_image_tensor;

    if (sd_img_gen_params->init_image.data != nullptr) {
        LOG_INFO("IMG2IMG");

        if (request->strength < 1.f) {
            size_t t_enc = static_cast<size_t>(plan->sample_steps * request->strength);
            if (t_enc == static_cast<size_t>(plan->sample_steps)) {
                t_enc--;
            }
            LOG_INFO("target t_enc is %zu steps", t_enc);
            std::vector<float> sigma_sched;
            sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());
            plan->sigmas       = std::move(sigma_sched);
            plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
        }

        init_image_tensor = sd_image_to_tensor(sd_img_gen_params->init_image, request->width, request->height);
    }

    if (sd_img_gen_params->mask_image.data != nullptr) {
        mask_image_tensor = sd_image_to_tensor(sd_img_gen_params->mask_image, request->width, request->height);
        mask_image_tensor = sd::ops::round(mask_image_tensor);
    }

    if (sd_img_gen_params->control_image.data != nullptr) {
        control_image_tensor = sd_image_to_tensor(sd_img_gen_params->control_image, request->width, request->height);
    }

    if (init_image_tensor.empty() || mask_image_tensor.empty()) {
        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            LOG_WARN("inpainting model requires both an init image and a mask image.");
        }
    }

    if (mask_image_tensor.empty()) {
        mask_image_tensor = sd::full<float>({request->width, request->height, 1, 1}, 1.f);
    }

    sd::Tensor<float> latent_mask = sd::ops::interpolate(mask_image_tensor,
                                                         {request->width / request->vae_scale_factor,
                                                          request->height / request->vae_scale_factor,
                                                          1,
                                                          1},
                                                         sd::ops::InterpolateMode::NearestMax);

    sd::Tensor<float> init_latent;
    sd::Tensor<float> control_latent;
    if (init_image_tensor.empty()) {
        init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(init_image_tensor);
        if (init_latent.empty()) {
            LOG_ERROR("failed to encode init image");
            return std::nullopt;
        }
    }

    if (!control_image_tensor.empty() && !sd_ctx->sd->vae_decode_only) {
        control_latent = sd_ctx->sd->encode_first_stage(control_image_tensor);
        if (control_latent.empty()) {
            LOG_ERROR("failed to encode control image");
            return std::nullopt;
        }
    }

    std::vector<sd::Tensor<float>> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(sd_image_to_tensor(sd_img_gen_params->ref_images[i]));
    }

    if (ref_images.empty() && sd_version_is_unet_edit(sd_ctx->sd->version)) {
        LOG_WARN("This model needs at least one reference image; using an empty reference");
        ref_images.push_back(sd::zeros<float>({request->width, request->height, 3, 1}));
        request->guidance.img_cfg = request->guidance.txt_cfg;
        request->use_img_uncond   = false;
    }

    if (!ref_images.empty()) {
        LOG_INFO("EDIT mode");
    }

    std::vector<sd::Tensor<float>> ref_latents;
    for (size_t i = 0; i < ref_images.size(); i++) {
        if (sd_ctx->sd->version == VERSION_HIDREAM_O1) {
            continue;
        }
        sd::Tensor<float> ref_latent;
        if (request->auto_resize_ref_image && !sd_version_is_pid(sd_ctx->sd->version)) {
            LOG_DEBUG("auto resize ref images");
            int vae_image_size = std::min(1024 * 1024, request->width * request->height);
            double vae_width   = sqrt(vae_image_size * ref_images[i].shape()[0] / ref_images[i].shape()[1]);
            double vae_height  = vae_width * ref_images[i].shape()[1] / ref_images[i].shape()[0];

            int factor = sd_version_is_qwen_image(sd_ctx->sd->version) ? 32 : 16;
            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            auto resized_ref_img = sd::ops::interpolate(ref_images[i],
                                                        {static_cast<int>(vae_width), static_cast<int>(vae_height), 3, 1});

            LOG_DEBUG("resize vae ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64,
                      static_cast<int>(i),
                      ref_images[i].shape()[1],
                      ref_images[i].shape()[0],
                      resized_ref_img.shape()[1],
                      resized_ref_img.shape()[0]);

            ref_latent = sd_ctx->sd->encode_first_stage(resized_ref_img);
        } else {
            ref_latent = sd_ctx->sd->encode_first_stage(ref_images[i]);
        }
        if (ref_latent.empty()) {
            LOG_ERROR("failed to encode reference image %d", static_cast<int>(i));
            return std::nullopt;
        }

        ref_latents.push_back(std::move(ref_latent));
    }

    if (sd_version_is_pid(sd_ctx->sd->version)) {
        if (ref_latents.empty()) {
            LOG_ERROR("PiD requires a reference image");
            return std::nullopt;
        }
    }

    sd::Tensor<float> concat_latent;
    sd::Tensor<float> img_uncond_concat_latent;
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        sd::Tensor<float> masked_init_latent;

        if (sd_ctx->sd->version != VERSION_FLEX_2) {
            if (!init_image_tensor.empty()) {
                auto masked_image  = ((1.0f - mask_image_tensor) * (init_image_tensor - 0.5f)) + 0.5f;
                masked_init_latent = sd_ctx->sd->encode_first_stage(masked_image);
                if (masked_init_latent.empty()) {
                    LOG_ERROR("failed to encode masked init image");
                    return std::nullopt;
                }
            } else {
                masked_init_latent = sd::Tensor<float>::zeros_like(init_latent);
            }
        } else {
            masked_init_latent = ((1.0f - latent_mask) * init_latent);
        }

        auto uncond_masked_init_latent = sd::Tensor<float>::zeros_like(masked_init_latent);

        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            auto mask = mask_image_tensor.reshape({request->vae_scale_factor,
                                                   request->width / request->vae_scale_factor,
                                                   request->vae_scale_factor,
                                                   request->height / request->vae_scale_factor});
            mask      = mask.permute({1, 3, 0, 2}).reshape({request->width / request->vae_scale_factor, request->height / request->vae_scale_factor, request->vae_scale_factor * request->vae_scale_factor, 1});

            concat_latent            = sd::ops::concat(masked_init_latent, mask, 2);
            img_uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, mask, 2);
        } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
            concat_latent = sd::ops::concat(masked_init_latent, latent_mask, 2);
            if (!control_latent.empty()) {
                concat_latent = sd::ops::concat(concat_latent, control_latent, 2);
            } else {
                concat_latent = sd::ops::concat(concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
            }

            img_uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, latent_mask, 2);
            img_uncond_concat_latent = sd::ops::concat(img_uncond_concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
        } else {  // SD1.x SD2.x SDXL inpaint
            concat_latent            = sd::ops::concat(latent_mask, masked_init_latent, 2);
            img_uncond_concat_latent = sd::ops::concat(latent_mask, uncond_masked_init_latent, 2);
        }
    }
    if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        concat_latent            = sd::ops::interpolate<float>(ref_latents[0], init_latent.shape());
        img_uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }
    if (sd_ctx->sd->version == VERSION_FLUX_CONTROLS) {
        if (!control_latent.empty()) {
            concat_latent = control_latent;
        } else {
            concat_latent = sd::Tensor<float>::zeros_like(init_latent);
        }
        img_uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }

    if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0) {
        int64_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);
    }

    ImageGenerationLatents latents;
    latents.init_latent              = std::move(init_latent);
    latents.concat_latent            = std::move(concat_latent);
    latents.img_uncond_concat_latent = std::move(img_uncond_concat_latent);
    latents.control_image            = std::move(control_image_tensor);
    latents.ref_images               = std::move(ref_images);
    latents.ref_latents              = std::move(ref_latents);

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        latent_mask = sd::ops::max_pool_2d(latent_mask,
                                           {3, 3},
                                           {1, 1},
                                           {1, 1});
    }
    latents.denoise_mask = std::move(latent_mask);

    return latents;
}

static std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds(sd_ctx_t* sd_ctx,
                                                                            const sd_img_gen_params_t* sd_img_gen_params,
                                                                            GenerationRequest* request,
                                                                            SamplePlan* plan,
                                                                            ImageGenerationLatents* latents) {
    ConditionerParams condition_params;
    condition_params.text       = request->prompt;
    condition_params.clip_skip  = request->clip_skip;
    condition_params.width      = request->width;
    condition_params.height     = request->height;
    condition_params.ref_images = &latents->ref_images;

    sd_ctx->sd->prepare_generation_extensions(request->pm_params,
                                              condition_params,
                                              plan->total_steps);
    int64_t prepare_start_ms         = ggml_time_ms();
    condition_params.zero_out_masked = false;

    // Release-after-encode (warm image worker): the TE may have been freed after the
    // previous request's cond to keep the NVFP4 DiT's resident footprint low (~6 GB vs
    // ~11). Reload it from the captured loader before this prompt's encode. No-op if the
    // TE is still resident or no reload state was captured (one-shot CLI / non-warm).
    if (!sd_ctx->sd->reload_cond_stage_model()) {
        LOG_ERROR("text-encoder reload failed; text conditioning may be unavailable");
    }

    // flux2 lap-11 Lever A: CFG does two consecutive text encodes (cond below,
    // uncond just after). With --offload-to-cpu the text-encoder params get
    // restored to CPU after the cond encode and fully re-uploaded for the uncond
    // encode (~0.6s redundant H2D). Keep them resident across both encodes; the
    // params are already paid for by the first encode and stay the same size, so
    // VRAM peak is unchanged. They are restored to CPU below (before the UNet
    // loads) by set_keep_params_resident(false) / free_params_buffer().
    const bool keep_te_resident_across_cfg = (request->use_uncond || request->use_high_noise_uncond);
    if (keep_te_resident_across_cfg) {
        sd_ctx->sd->cond_stage_model->set_keep_params_resident(true);
    }

    auto cond                        = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                           condition_params);
    if (cond.c_concat.empty()) {
        cond.c_concat = latents->concat_latent;  // TODO: optimize
    }

    bool use_ref_latent_img_cfg = request->use_img_uncond &&
                                  !latents->ref_images.empty() &&
                                  sd_version_supports_ref_latent_img_cfg(sd_ctx->sd->version);

    SDCondition uncond;
    if (request->use_uncond || request->use_high_noise_uncond) {
        if (sd_version_is_ideogram4(sd_ctx->sd->version)) {
            uncond.c_vector = sd::Tensor<float>::from_vector({1.0f});
        } else {
            bool zero_out_masked = false;
            if (sd_version_is_sdxl(sd_ctx->sd->version) &&
                request->negative_prompt.empty() &&
                !sd_ctx->sd->is_using_edm_v_parameterization) {
                zero_out_masked = true;
            }
            condition_params.text            = request->negative_prompt;
            condition_params.zero_out_masked = zero_out_masked;
            uncond                           = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                                   condition_params);
        }
        if (uncond.c_concat.empty()) {
            uncond.c_concat = latents->concat_latent;  // TODO: optimize
        }
    }

    SDCondition img_uncond;
    if (request->use_img_uncond) {
        if ((request->use_uncond || request->use_high_noise_uncond) && (latents->ref_images.empty() || !use_ref_latent_img_cfg)) {
            img_uncond = SDCondition(uncond.c_crossattn, uncond.c_vector, latents->img_uncond_concat_latent);
        } else {
            bool zero_out_masked = false;
            if (sd_version_is_sdxl(sd_ctx->sd->version) &&
                request->negative_prompt.empty() &&
                !sd_ctx->sd->is_using_edm_v_parameterization) {
                zero_out_masked = true;
            }
            condition_params.text            = request->negative_prompt;
            condition_params.zero_out_masked = zero_out_masked;
            if (use_ref_latent_img_cfg) {
                std::vector<sd::Tensor<float>> empty_ref_images;
                condition_params.ref_images = &empty_ref_images;
            }
            img_uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                             condition_params);
            if (img_uncond.c_concat.empty()) {
                img_uncond.c_concat = latents->img_uncond_concat_latent;  // TODO: optimize
            }
        }
    }

    // flux2 lap-11 Lever A: both CFG encodes done — restore text-encoder params
    // back to the params backend (frees runtime VRAM) before the UNet loads.
    if (keep_te_resident_across_cfg) {
        sd_ctx->sd->cond_stage_model->set_keep_params_resident(false);
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    // Only free the text encoder when this is a one-shot context (CLI). On a warm
    // RESIDENT worker (worker isolation / image API) keep_diffusion_model_resident is
    // set and every /generate re-encodes a fresh prompt, so the TE must survive across
    // requests. The image path has no TE reload mechanism (unlike the avatar video path
    // at get_learned_condition_for_video, which captures reload state + calls
    // reload_cond_stage_model()), so freeing it here would leave the conditioner's param
    // tensors with NULL buffers — free_params_buffer() nulls t->data/buffer even for the
    // mmap fast-path (params_buffer==nullptr) — and the NEXT encode aborts in
    // ggml_backend_tensor_copy (GGML_ASSERT(buffer)). Mirror the !keep_diffusion_model_resident
    // guard the VAE/DiT frees already use (decode_first_stage / sample). Under
    // --offload-to-cpu the TE params live on the CPU params backend (the per-compute GPU
    // copy is freed by free_compute_buffer), so keeping them resident costs no VRAM; under
    // --mmap they are reclaimable file-backed pages.
    //
    // Release-after-encode: the warm resident IMAGE worker frees the TE TOO when we
    // captured reload state above — with NVFP4 offload-off the Qwen3 encoder (~4.7 GB)
    // would otherwise stay GPU-resident between requests; freeing it here drops the warm
    // footprint to ~6 GB and reload_cond_stage_model() (top of prepare_image_generation_embeds)
    // restores it before the next prompt from the captured mmap'd loader (~1-2s/render).
    const bool warm_release_te = sd_ctx->sd->keep_diffusion_model_resident &&
                                 sd_ctx->sd->te_reload_loader &&
                                 !sd_ctx->sd->te_reload_tensors.empty();
    if (sd_ctx->sd->free_params_immediately &&
        (!sd_ctx->sd->keep_diffusion_model_resident || warm_release_te)) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    ImageGenerationEmbeds embeds;
    embeds.img_uncond = std::move(img_uncond);
    embeds.cond       = std::move(cond);
    embeds.uncond     = std::move(uncond);

    return embeds;
}

static sd_image_t* decode_image_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const std::vector<sd::Tensor<float>>& final_latents) {
    if (final_latents.size() != static_cast<size_t>(request.batch_count)) {
        LOG_ERROR("expected %d latents, got %zu", request.batch_count, final_latents.size());
        return nullptr;
    }
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<sd::Tensor<float>> decoded_images;
    int64_t t0 = ggml_time_ms();

    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);
        if (image.empty()) {
            LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
            if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
                sd_ctx->sd->first_stage_model->free_params_buffer();
            }
            return nullptr;
        }
        decoded_images.push_back(std::move(image));
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %zu decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t0) * 1.0f / 1000);
    // !keep_diffusion_model_resident: the VAE, like the DiT/TE, must survive across
    // /generate calls on a warm resident worker (no reload path on the image side).
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_image_t* result_images = (sd_image_t*)calloc(request.batch_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    memset(result_images, 0, request.batch_count * sizeof(sd_image_t));

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i] = tensor_to_sd_image(decoded_images[i]);
    }

    return result_images;
}

static sd::Tensor<float> upscale_hires_latent(sd_ctx_t* sd_ctx,
                                              const sd::Tensor<float>& latent,
                                              const GenerationRequest& request,
                                              UpscalerGGML* upscaler) {
    auto get_hires_latent_target_shape = [&]() {
        std::vector<int64_t> target_shape = latent.shape();
        if (target_shape.size() < 2) {
            target_shape.clear();
            return target_shape;
        }
        target_shape[0] = request.hires.target_width / request.vae_scale_factor;
        target_shape[1] = request.hires.target_height / request.vae_scale_factor;
        return target_shape;
    };

    if (request.hires.upscaler == SD_HIRES_UPSCALER_LATENT ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_ANTIALIASED ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED) {
        std::vector<int64_t> target_shape = get_hires_latent_target_shape();
        if (target_shape.empty()) {
            LOG_ERROR("latent has invalid shape for hires upscale");
            return {};
        }

        sd::ops::InterpolateMode mode = sd::ops::InterpolateMode::Nearest;
        bool antialias                = false;
        switch (request.hires.upscaler) {
            case SD_HIRES_UPSCALER_LATENT:
                mode = sd::ops::InterpolateMode::Bilinear;
                break;
            case SD_HIRES_UPSCALER_LATENT_NEAREST:
                mode = sd::ops::InterpolateMode::Nearest;
                break;
            case SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT:
                mode = sd::ops::InterpolateMode::NearestExact;
                break;
            case SD_HIRES_UPSCALER_LATENT_ANTIALIASED:
                mode      = sd::ops::InterpolateMode::Bilinear;
                antialias = true;
                break;
            case SD_HIRES_UPSCALER_LATENT_BICUBIC:
                mode = sd::ops::InterpolateMode::Bicubic;
                break;
            case SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED:
                mode      = sd::ops::InterpolateMode::Bicubic;
                antialias = true;
                break;
            default:
                break;
        }

        LOG_INFO("hires %s upscale %" PRId64 "x%" PRId64 " -> %" PRId64 "x%" PRId64,
                 sd_hires_upscaler_name(request.hires.upscaler),
                 latent.shape()[0],
                 latent.shape()[1],
                 target_shape[0],
                 target_shape[1]);

        return sd::ops::interpolate(latent, target_shape, mode, false, antialias);
    } else if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL ||
               request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS ||
               request.hires.upscaler == SD_HIRES_UPSCALER_NEAREST) {
        if (sd_ctx->sd->vae_decode_only) {
            LOG_ERROR("hires %s upscaler requires VAE encoder weights; create the context with vae_decode_only=false",
                      sd_hires_upscaler_name(request.hires.upscaler));
            return {};
        }
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL && upscaler == nullptr) {
            LOG_ERROR("hires model upscaler context is null");
            return {};
        }

        sd::Tensor<float> decoded = sd_ctx->sd->decode_first_stage(latent);
        if (decoded.empty()) {
            LOG_ERROR("decode_first_stage failed before hires %s upscale",
                      sd_hires_upscaler_name(request.hires.upscaler));
            return {};
        }

        sd::Tensor<float> upscaled_tensor;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            upscaled_tensor = upscaler->upscale_tensor(decoded);
            if (upscaled_tensor.empty()) {
                LOG_ERROR("hires model upscale failed");
                return {};
            }

            if (upscaled_tensor.shape()[0] != request.hires.target_width ||
                upscaled_tensor.shape()[1] != request.hires.target_height) {
                upscaled_tensor = sd::ops::interpolate(upscaled_tensor,
                                                       {request.hires.target_width,
                                                        request.hires.target_height,
                                                        upscaled_tensor.shape()[2],
                                                        upscaled_tensor.shape()[3]});
            }
        } else {
            sd::ops::InterpolateMode mode = request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS
                                                ? sd::ops::InterpolateMode::Lanczos
                                                : sd::ops::InterpolateMode::Nearest;
            LOG_INFO("hires %s image upscale %" PRId64 "x%" PRId64 " -> %dx%d",
                     sd_hires_upscaler_name(request.hires.upscaler),
                     decoded.shape()[0],
                     decoded.shape()[1],
                     request.hires.target_width,
                     request.hires.target_height);
            upscaled_tensor = sd::ops::interpolate(decoded,
                                                   {request.hires.target_width,
                                                    request.hires.target_height,
                                                    decoded.shape()[2],
                                                    decoded.shape()[3]},
                                                   mode);
            upscaled_tensor = sd::ops::clamp(upscaled_tensor, 0.0f, 1.0f);
        }

        sd::Tensor<float> upscaled_latent = sd_ctx->sd->encode_first_stage(upscaled_tensor);
        if (upscaled_latent.empty()) {
            LOG_ERROR("encode_first_stage failed after hires %s upscale",
                      sd_hires_upscaler_name(request.hires.upscaler));
        }
        return upscaled_latent;
    }

    LOG_ERROR("unsupported hires upscaler '%s'", sd_hires_upscaler_name(request.hires.upscaler));
    return {};
}

static std::vector<float> make_hires_sigma_schedule(sd_ctx_t* sd_ctx,
                                                    const sd_hires_params_t& hires,
                                                    const sd_sample_params_t& sample_params,
                                                    sample_method_t sample_method,
                                                    int default_steps,
                                                    int sample_seq_len,
                                                    int* scheduler_steps_out) {
    if (scheduler_steps_out != nullptr) {
        *scheduler_steps_out = 0;
    }

    if (hires.custom_sigmas_count > 0 && hires.custom_sigmas != nullptr) {
        std::vector<float> custom_sigmas(hires.custom_sigmas,
                                         hires.custom_sigmas + hires.custom_sigmas_count);
        if (scheduler_steps_out != nullptr) {
            *scheduler_steps_out = static_cast<int>(custom_sigmas.size()) - 1;
        }
        return custom_sigmas;
    }

    int effective_steps = hires.steps > 0 ? hires.steps : default_steps;
    effective_steps     = std::max(1, effective_steps);

    // sd-webui behavior: scale up total steps so trimming by denoising_strength yields exactly hires_steps effective steps,
    // unlike img2img which trims from a fixed step count.
    int scheduler_steps = static_cast<int>(effective_steps / hires.denoising_strength);
    scheduler_steps     = std::max(1, scheduler_steps);

    scheduler_t scheduler     = resolve_scheduler(sd_ctx,
                                                  sample_params.scheduler,
                                                  sample_method);
    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(scheduler_steps,
                                                                 sample_seq_len,
                                                                 scheduler,
                                                                 sd_ctx->sd->version,
                                                                 sample_params.extra_sample_args);
    size_t t_enc              = static_cast<size_t>(scheduler_steps * hires.denoising_strength);
    if (t_enc >= static_cast<size_t>(scheduler_steps)) {
        t_enc = static_cast<size_t>(scheduler_steps) - 1;
    }
    if (scheduler_steps_out != nullptr) {
        *scheduler_steps_out = scheduler_steps;
    }
    return std::vector<float>(sigmas.begin() + scheduler_steps - static_cast<int>(t_enc) - 1,
                              sigmas.end());
}

SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    LOG_INFO("generate_image %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan);
    if (!latents_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents);
    if (!embeds_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    std::vector<sd::Tensor<float>> final_latents;
    int64_t denoise_start = ggml_time_ms();
    // flux2 lap-11 Lever C: for a multi-seed batch the UNet weights are re-uploaded
    // for every seed (each sample() ends by restoring params to CPU). Nothing else
    // touches VRAM between seeds (the text encoder is already freed; the VAE runs
    // after this loop), so keep the UNet resident across the seeds — saves one
    // ~0.8s 5.6GB H2D per extra seed. VRAM peak is unchanged (one UNet, activations
    // still freed between seeds). Cleared below before the UNet is freed / the VAE
    // loads. No-op for batch_count==1 (single-gen path untouched).
    const bool keep_unet_resident = request.batch_count > 1;
    if (keep_unet_resident) {
        sd_ctx->sd->diffusion_model->set_keep_params_resident(true);
    }
    for (int b = 0; b < request.batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = request.seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

        // FP4 bench harness: replay a fixed initial latent across configs so the ONLY
        // difference between renders is the compute path -> LPIPS/PSNR/latent-cosine
        // become meaningful. FLUX2_INIT_LATENT=<file> loads the noise (raw f32, must
        // match numel); FLUX2_SAVE_LATENT=<file> banks the freshly-sampled noise.
        // File format: magic "FLX2LAT1" + int64 numel + numel*f32 (row-major).
        if (const char* lp = getenv("FLUX2_INIT_LATENT")) {
            FILE* f = fopen(lp, "rb");
            if (!f) { LOG_ERROR("FLUX2_INIT_LATENT: cannot open %s", lp); }
            else {
                char magic[8] = {0}; int64_t nel = 0;
                if (fread(magic, 1, 8, f) == 8 && memcmp(magic, "FLX2LAT1", 8) == 0 &&
                    fread(&nel, sizeof(nel), 1, f) == 1 && nel == noise.numel() &&
                    fread(noise.data(), sizeof(float), (size_t)nel, f) == (size_t)nel) {
                    LOG_INFO("FLUX2_INIT_LATENT: loaded %lld-elem fixed latent from %s", (long long)nel, lp);
                } else {
                    LOG_ERROR("FLUX2_INIT_LATENT: bad/mismatched file %s (numel=%lld) — using RNG noise",
                              lp, (long long)noise.numel());
                }
                fclose(f);
            }
        }
        if (const char* sp = getenv("FLUX2_SAVE_LATENT")) {
            FILE* f = fopen(sp, "wb");
            if (!f) { LOG_ERROR("FLUX2_SAVE_LATENT: cannot open %s", sp); }
            else {
                int64_t nel = noise.numel();
                fwrite("FLX2LAT1", 1, 8, f);
                fwrite(&nel, sizeof(nel), 1, f);
                fwrite(noise.data(), sizeof(float), (size_t)nel, f);
                fclose(f);
                LOG_INFO("FLUX2_SAVE_LATENT: banked %lld-elem latent to %s", (long long)nel, sp);
            }
        }

        sd::Tensor<float> x_0 = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                   true,
                                                   latents.init_latent,
                                                   std::move(noise),
                                                   embeds.cond,
                                                   embeds.uncond,
                                                   embeds.img_uncond,
                                                   latents.control_image,
                                                   request.control_strength,
                                                   request.guidance,
                                                   plan.eta,
                                                   request.shifted_timestep,
                                                   plan.sample_method,
                                                   sd_ctx->sd->is_flow_denoiser(),
                                                   plan.extra_sample_args,
                                                   plan.sigmas,
                                                   latents.ref_latents,
                                                   request.increase_ref_index,
                                                   latents.denoise_mask,
                                                   sd::Tensor<float>(),
                                                   1.f,
                                                   0,
                                                   static_cast<float>(request.fps),
                                                   request.cache_params);
        int64_t sampling_end  = ggml_time_ms();
        if (!x_0.empty()) {
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(std::move(x_0));
            continue;
        }

        LOG_ERROR("sampling for image %d/%d failed after %.2fs",
                  b + 1,
                  request.batch_count,
                  (sampling_end - sampling_start) * 1.0f / 1000);
        if (keep_unet_resident) {
            sd_ctx->sd->diffusion_model->set_keep_params_resident(false);
        }
        // !keep_diffusion_model_resident: never free the DiT on a warm resident
        // worker (image API / isolation) — it has no reload path and the next
        // request would abort with NULL-buffer params. See the TE free in
        // prepare_image_generation_embeds for the full rationale.
        if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        return nullptr;
    }
    // flux2 lap-11 Lever C: all seeds done — drop the keep-resident hold, which
    // restores the UNet params off the runtime backend (frees VRAM) before the
    // free below / the VAE decode loads.
    if (keep_unet_resident) {
        sd_ctx->sd->diffusion_model->set_keep_params_resident(false);
    }
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident && !request.hires.enabled) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %zu latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);

    if (request.hires.enabled && request.hires.target_width > 0) {
        LOG_INFO("hires fix: upscaling to %dx%d", request.hires.target_width, request.hires.target_height);

        std::unique_ptr<UpscalerGGML> hires_upscaler;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            LOG_INFO("hires fix: loading model upscaler from '%s'", request.hires.model_path);
            hires_upscaler                    = std::make_unique<UpscalerGGML>(sd_ctx->sd->n_threads,
                                                            false,
                                                            request.hires.upscale_tile_size,
                                                            sd_ctx->sd->backend_spec,
                                                            sd_ctx->sd->params_backend_spec);
            const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(sd_ctx->sd->max_vram);
            hires_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
            if (!hires_upscaler->load_from_file(request.hires.model_path,
                                                sd_ctx->sd->offload_params_to_cpu,
                                                sd_ctx->sd->n_threads)) {
                LOG_ERROR("load hires model upscaler failed");
                if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }
        }

        int hires_scheduler_steps = 0;
        std::vector<float> hires_sigma_sched =
            make_hires_sigma_schedule(sd_ctx,
                                      request.hires,
                                      sd_img_gen_params->sample_params,
                                      plan.sample_method,
                                      plan.sample_steps,
                                      sd_ctx->sd->get_image_seq_len(request.hires.target_height, request.hires.target_width),
                                      &hires_scheduler_steps);
        LOG_INFO("hires fix: scheduler_steps=%d, denoising_strength=%.2f, sigma_sched_size=%zu%s",
                 hires_scheduler_steps,
                 request.hires.denoising_strength,
                 hires_sigma_sched.size(),
                 request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");

        std::vector<sd::Tensor<float>> hires_final_latents;
        int64_t hires_denoise_start = ggml_time_ms();
        for (int b = 0; b < (int)final_latents.size(); b++) {
            int64_t cur_seed = request.seed + b;
            sd_ctx->sd->rng->manual_seed(cur_seed);
            sd_ctx->sd->sampler_rng->manual_seed(cur_seed);

            sd::Tensor<float> upscaled = upscale_hires_latent(sd_ctx,
                                                              final_latents[b],
                                                              request,
                                                              hires_upscaler.get());
            if (upscaled.empty()) {
                if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }

            sd::Tensor<float> noise = sd::randn_like<float>(upscaled, sd_ctx->sd->rng);

            sd::Tensor<float> hires_denoise_mask;
            if (!latents.denoise_mask.empty()) {
                std::vector<int64_t> mask_shape = latents.denoise_mask.shape();
                mask_shape[0]                   = upscaled.shape()[0];
                mask_shape[1]                   = upscaled.shape()[1];
                hires_denoise_mask              = sd::ops::interpolate(latents.denoise_mask,
                                                                       mask_shape,
                                                                       sd::ops::InterpolateMode::NearestMax);
            }

            int64_t hires_sample_start = ggml_time_ms();
            sd::Tensor<float> x_0      = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                            true,
                                                            upscaled,
                                                            std::move(noise),
                                                            embeds.cond,
                                                            embeds.uncond,
                                                            embeds.img_uncond,
                                                            latents.control_image,
                                                            request.control_strength,
                                                            request.guidance,
                                                            plan.eta,
                                                            request.shifted_timestep,
                                                            plan.sample_method,
                                                            sd_ctx->sd->is_flow_denoiser(),
                                                            plan.extra_sample_args,
                                                            hires_sigma_sched,
                                                            latents.ref_latents,
                                                            request.increase_ref_index,
                                                            hires_denoise_mask,
                                                            sd::Tensor<float>(),
                                                            1.f,
                                                            0,
                                                            static_cast<float>(request.fps),
                                                            request.cache_params);
            int64_t hires_sample_end   = ggml_time_ms();
            if (!x_0.empty()) {
                LOG_INFO("hires sampling %d/%d completed, taking %.2fs",
                         b + 1,
                         (int)final_latents.size(),
                         (hires_sample_end - hires_sample_start) * 1.0f / 1000);
                hires_final_latents.push_back(std::move(x_0));
                continue;
            }

            LOG_ERROR("hires sampling for image %d/%d failed after %.2fs",
                      b + 1,
                      (int)final_latents.size(),
                      (hires_sample_end - hires_sample_start) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return nullptr;
        }
        if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        int64_t hires_denoise_end = ggml_time_ms();
        LOG_INFO("hires fix completed, taking %.2fs", (hires_denoise_end - hires_denoise_start) * 1.0f / 1000);

        final_latents = std::move(hires_final_latents);
    }

    auto result = decode_image_outputs(sd_ctx, request, final_latents);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}

static std::optional<ImageGenerationLatents> prepare_video_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_vid_gen_params_t* sd_vid_gen_params,
                                                                              GenerationRequest* request) {
    ImageGenerationLatents latents;
    int64_t prepare_start_ms = ggml_time_ms();

    // Reset the avatar continuation ref-anchor params each render so a prior segment's
    // 3-way-split state never leaks into a single-clip / ai2v / seg0 render (the model
    // is resident across chained calls). The cont-latent branch below re-arms them.
    // Also thread the per-request BSA params (lap-32.4) onto the model so build_graph
    // sees this caller's BSA choice. Defaults from sd_vid_gen_params_init mean a caller
    // that never touches the BSA fields gets dense attention (bit-exact, the prior
    // default behavior); flipping bsa_enabled=1 engages the BSA path for this render
    // only with the r=1+self_frame preset.
    if (sd_version_is_longcat_avatar(sd_ctx->sd->version)) {
        if (auto avatar_model = std::dynamic_pointer_cast<LongCatAvatarModel>(sd_ctx->sd->diffusion_model)) {
            avatar_model->cont_num_ref_latents = 0;
            avatar_model->bsa_enabled    = sd_vid_gen_params->bsa_enabled != 0;
            avatar_model->bsa_radius     = sd_vid_gen_params->bsa_radius;
            avatar_model->bsa_self_frame = sd_vid_gen_params->bsa_self_frame != 0;
            avatar_model->bsa_bookend    = sd_vid_gen_params->bsa_bookend != 0;
            avatar_model->bsa_cube_h     = sd_vid_gen_params->bsa_cube_h;
            avatar_model->bsa_cube_w     = sd_vid_gen_params->bsa_cube_w;
        }
    }

    sd::Tensor<float> start_image;
    sd::Tensor<float> end_image;

    if (sd_vid_gen_params->init_image.data) {
        start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, request->width, request->height);
    }

    if (sd_vid_gen_params->end_image.data) {
        end_image = sd_image_to_tensor(sd_vid_gen_params->end_image, request->width, request->height);
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version)) {
        // A guide-edit (v2v_mode==2) whose source is a banked latent (path) carries no control_frames
        // but still needs the 8k+1 target so the target latent lines up 1:1 with the banked guide.
        const bool ltxav_guide_latent = (sd_vid_gen_params->v2v_mode == 2 &&
                                         SAFE_STR(sd_vid_gen_params->v2v_guide_latent_path)[0] != '\0');
        // LTX-2.3 V2V lipdub relip (control_frames) requires an 8k+1 frame count (VAE temporal
        // stride 8). Snap here — before the audio length is derived — so the reference clip,
        // target latent, and audio all agree. Gated on control_frames (or a latent-in guide) so
        // every other LTXAV path (i2v, t2v, continuation, --drive-audio) is byte-identical.
        if (sd_vid_gen_params->control_frames_size > 0 || ltxav_guide_latent) {
            int snapped = ((request->frames - 1) / 8) * 8 + 1;
            if (snapped < 1) {
                snapped = 1;
            }
            if (snapped != request->frames) {
                LOG_INFO("LTXAV relip: snapping frames %d -> %d (8k+1)", request->frames, snapped);
                request->frames = snapped;
            }
        }
        // Video-only classic LTX-Video 0.9.x checkpoints have no audio stream — synthesizing an
        // audio latent here would feed the DiT an audio stream it has no tensors for (null deref).
        bool ltxav_has_audio = sd_ctx->sd->diffusion_model == nullptr || sd_ctx->sd->diffusion_model->has_audio_stream();
        if (ltxav_has_audio) {
            latents.audio_length = get_ltxav_num_audio_latents(request->frames, request->fps);
            const char* drive_wav = SAFE_STR(sd_vid_gen_params->drive_audio_path);
            if (strlen(drive_wav) > 0) {
                auto driven = encode_ltxav_drive_audio(sd_ctx, drive_wav, latents.audio_length);
                if (!driven.empty()) {
                    // The direct fixed-audio path is the established V2V LipDub
                    // behaviour: it keeps the supplied waveform latent clean in
                    // both stages while the IC-LoRA guide preserves the video.
                    // Retain the official generated-audio-reference layout as an
                    // explicit experiment only (LTXAV_LIPDUB_AUDIO_MODE=reference).
                    const char* audio_mode = std::getenv("LTXAV_LIPDUB_AUDIO_MODE");
                    const bool direct_fixed_audio = audio_mode == nullptr || std::string(audio_mode) != "reference";
                    if (direct_fixed_audio) {
                        latents.audio_latent = std::move(driven);
                        latents.audio_fixed  = true;
                        LOG_INFO("LTXAV LipDub A/B: using direct fixed driving-audio conditioning");
                    } else {
                    // Official LipDub semantics: the dubbed waveform is a CLEAN audio-reference
                    // block at negative positions, not a frozen replacement for the target audio
                    // stream.  Stage 1 denoises target audio/video against this reference; stage 2
                    // later freezes the stage-1 target audio.
                    const int target_length = latents.audio_length;
                    latents.audio_target_length           = target_length;
                    latents.audio_latent                  = sd::ops::concat(make_ltxav_empty_audio_latent(target_length), driven, 1);
                    latents.audio_positions               = build_ltxav_lipdub_audio_positions(target_length, (int)driven.shape()[1]);
                    latents.audio_length                  = target_length + (int)driven.shape()[1];
                    latents.audio_reference_conditioning  = true;
                    latents.audio_fixed                   = false;
                    }
                } else {
                    latents.audio_latent = make_ltxav_empty_audio_latent(latents.audio_length);
                }
            } else {
                latents.audio_latent = make_ltxav_empty_audio_latent(latents.audio_length);
            }
        } else {
            latents.audio_length = 0;
        }
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version)) {
        if (sd_vid_gen_params->control_frames_size > 0 && sd_vid_gen_params->v2v_mode == 0) {
            // LTX-2.3 V2V LIPDUB RELIP: the control_frames are an existing video clip (e.g. a
            // Wan2.2 render). Encode the WHOLE clip to video latents and append it as a frozen,
            // timeline-aligned IC-LoRA reference (apply_ltxav_video_relip_reference). With the
            // lipdub IC-LoRA loaded (--lora) + a frozen driving-audio latent (--drive-audio), the
            // model preserves the input video and only re-lips the mouth. Off-switch:
            // LTXAV_RELIP_DISABLE restores the legacy "not implemented" rejection for A/B.
            // (v2v_mode==1 = SDEdit and v2v_mode==2 = guide-edit are handled separately below.)
            if (std::getenv("LTXAV_RELIP_DISABLE") != nullptr) {
                LOG_ERROR("LTXAV control_frames (relip) disabled via LTXAV_RELIP_DISABLE");
                return std::nullopt;
            }
            if (sd_ctx->sd->vae_decode_only) {
                LOG_ERROR("LTXAV relip (control_frames) requires VAE encoder weights; create the context with vae_decode_only=false");
                return std::nullopt;
            }

            int64_t t1      = ggml_time_ms();
            int64_t cf_size = sd_vid_gen_params->control_frames_size;
            if (cf_size < request->frames) {
                LOG_WARN("LTXAV relip: %lld control_frames < %d target frames; holding the last frame for the remainder",
                         (long long)cf_size, request->frames);
            }

            // FIX A2 (env LTXAV_RELIP_REF_DOWNSCALE, default 1 = full-res reference, the prod
            // path; N>1 = separable half-res reference): the lipdub-0.9 IC-LoRA carries
            // reference_downscale_factor and was trained with a HALF-resolution reference.
            // The reference currently DOUBLES the DiT video-token count (L_q=L_k ~= 44000) and
            // is the binding driver of the 4524 MiB DiT compute buffer that OOMs at 193 frames
            // (render-proven: --max-vram / *_RESIDENT / TE-free all still OOM at the SAME alloc).
            //
            // N==1 (default/prod): UNCHANGED. Reference encoded at full res and grid-concatenated
            //   into the sampler latent (apply_ltxav_video_relip_reference) — byte-identical.
            // N>1 (opt-in): reference encoded at W/N x H/N and kept as a SEPARATE latent grid
            //   (latents.video_reference). It is NOT in the sampler grid (that stays target-only,
            //   denoise_mask all-ones). The DiT patchifies it on its own and appends its (N^2
            //   fewer) tokens to the video sequence; positions for the ref block come from the
            //   half-res emit_block of build_ltxv_relip_video_positions; per-token ref timesteps
            //   are appended as frozen t=0 inside build_graph; the ref tokens are sliced off
            //   before unpatchify so the output is target-only (crop becomes a no-op).
            //
            // FIX A2t (env LTXAV_RELIP_REF_TSTRIDE, default 1): TEMPORAL subsample — keep FULL
            // spatial resolution (face identity intact) but emit only every T-th reference latent
            // frame. ref tokens drop by ~T with NO spatial loss. Composes with DOWNSCALE (both
            // shrink the ref token block); either >1 forces the same separable-token path (a
            // shorter/lower-res reference can't grid-concat onto the full target). The subsampled
            // ref frames keep their TRUE timeline coordinates (frame j -> original frame j*T) via
            // ref_temporal_stride in build_ltxv_relip_video_positions, so each still maps to the
            // right target time.
            int relip_ref_downscale = 1;
            if (const char* e = std::getenv("LTXAV_RELIP_REF_DOWNSCALE")) {
                int v = std::atoi(e);
                if (v >= 1) {
                    relip_ref_downscale = v;
                }
            }
            int relip_ref_tstride = 1;
            if (const char* e = std::getenv("LTXAV_RELIP_REF_TSTRIDE")) {
                int v = std::atoi(e);
                if (v >= 1) {
                    relip_ref_tstride = v;
                }
            }
            const int ref_ds = relip_ref_downscale;
            // Reference encode dims. N==1 => full target res (prod path, byte-identical). N>1:
            // encode the reference at the OFFICIAL aspect-preserving downscale (content =
            // width/N x height/N; iclora_utils.py:108-109) — NOT floored to a VAE-scale multiple.
            // The old floor `(dim/N/vsf)*vsf` anisotropically squished the face (e.g. 352->160
            // instead of 176) AND dropped a latent row, so the reference RoPE span (ref_lat*vsf*N)
            // undershot the target extent and identity failed to bind. Our VAE derives latent dims
            // by floor division (vae.hpp:145), so we PAD the content (neutral gray, bottom/right) up
            // to the next VAE-scale multiple before encoding => ceil(content/vsf) rows, matching the
            // official VAE's internal padding; the pad lands in extra ref tokens whose positions
            // extend just past the target (the official overhang).
            int64_t ref_content_w = request->width;
            int64_t ref_content_h = request->height;
            int64_t ref_enc_w     = request->width;
            int64_t ref_enc_h     = request->height;
            if (ref_ds > 1) {
                int64_t vsf = std::max<int64_t>(1, request->vae_scale_factor);
                int64_t cw  = request->width / ref_ds;
                int64_t ch  = request->height / ref_ds;
                if (cw >= vsf && ch >= vsf) {
                    ref_content_w = cw;
                    ref_content_h = ch;
                    ref_enc_w     = ((cw + vsf - 1) / vsf) * vsf;
                    ref_enc_h     = ((ch + vsf - 1) / vsf) * vsf;
                    LOG_INFO("LTXAV relip: SEPARABLE half-res reference, downscale=%d -> resize ref to %lldx%lld (aspect-preserving) + pad to %lldx%lld for encode (target %dx%d)",
                             ref_ds, (long long)ref_content_w, (long long)ref_content_h,
                             (long long)ref_enc_w, (long long)ref_enc_h, request->width, request->height);
                } else {
                    LOG_WARN("LTXAV relip: LTXAV_RELIP_REF_DOWNSCALE=%d too large for %dx%d (vae_scale %lld); using full-res reference",
                             ref_ds, request->width, request->height, (long long)vsf);
                    relip_ref_downscale = 1;
                }
            }
            // Two-stage lipdub (LTXAV_RELIP_TWOSTAGE=1, set up in generate_video which already
            // halved request->w/h for stage-1): force the separable token path in stage-1 so the
            // reference is a separate block (matching the official, and so stage-2 can re-apply it
            // the same way). Recorded on latents so the hires refine (Change B) re-encodes the
            // reference at the upscaled full res.
            const bool relip_twostage = (std::getenv("LTXAV_RELIP_TWOSTAGE") != nullptr &&
                                         std::string(std::getenv("LTXAV_RELIP_TWOSTAGE")) != "0");
            const bool relip_separable = (relip_ref_downscale > 1) || (relip_ref_tstride > 1) || relip_twostage;
            if (relip_twostage) {
                latents.relip_twostage   = true;
                latents.relip_ref_downscale = relip_ref_downscale;
                latents.relip_ref_tstride   = relip_ref_tstride;
                LOG_INFO("LTXAV two-stage: stage1 %dx%d from-noise (relip separable ref, downscale=%d tstride=%d)",
                         request->width, request->height, relip_ref_downscale, relip_ref_tstride);
            }
            // Assemble the reference clip as [Wenc, Henc, frames, 3, 1] (same layout the VACE/i2v
            // encode paths use), then VAE-encode it directly (no -0.5; LTX feeds image tensors
            // straight to the encoder). Clamp index past the supplied frames to hold the last.
            sd::Tensor<float> ref_video = sd::full<float>({ref_content_w, ref_content_h, request->frames, 3, 1}, 0.5f);
            for (int64_t i = 0; i < request->frames; ++i) {
                int64_t src       = std::min<int64_t>(i, cf_size - 1);
                auto reference_fr = sd_image_to_tensor(sd_vid_gen_params->control_frames[src], ref_content_w, ref_content_h);
                sd::ops::slice_assign(&ref_video, 2, i, i + 1, reference_fr.unsqueeze(2));
            }
            // Neutral-gray PAD (bottom/right) up to the VAE-scale-aligned encode dims so the VAE's
            // floor-division latent sizing yields ceil(content/vsf) rows without squishing the face.
            if (ref_enc_h > ref_content_h) {
                ref_video = sd::ops::concat(ref_video,
                                            sd::full<float>({ref_content_w, ref_enc_h - ref_content_h, request->frames, 3, 1}, 0.5f),
                                            1);
            }
            if (ref_enc_w > ref_content_w) {
                ref_video = sd::ops::concat(ref_video,
                                            sd::full<float>({ref_enc_w - ref_content_w, ref_enc_h, request->frames, 3, 1}, 0.5f),
                                            0);
            }

            // RELIP REFERENCE ENCODE — VRAM lever. The VAE encoder builds ONE graph for the
            // whole clip, so the compute buffer holds every frame's activations at once
            // (~12GB @25f, ~40GB @81f at full res) and OOMs on long clips. Spatially tile the
            // ENCODE aggressively so each tile's buffer is small (16 tiles @0.25 fit 81f),
            // while keeping the output DECODE at the caller's tile (prod 1x1 = no seam) by
            // saving/restoring the params. The reference is a frozen conditioning signal (the
            // relip regenerates pixels at decode), so spatial tile-blend in it is harmless.
            // Tune via LTXAV_RELIP_ENCODE_TILE (default 0.25; 1.0 = old whole-frame behaviour).
            // (A streaming temporal encode would be cleaner, but this is the cheap lever.)
            sd_tiling_params_t relip_saved_tiling = sd_ctx->sd->vae_tiling_params;
            // Hoisted to outer scope: extra_tiling_args is a const char*, so the backing string
            // must outlive the encode_first_stage() call below (which parses it).
            std::string relip_enc_tiling_args;
            {
                float enc_tile = 0.25f;
                if (const char* e = getenv("LTXAV_RELIP_ENCODE_TILE")) {
                    float v = (float)atof(e);
                    if (v > 0.f && v <= 1.f) enc_tile = v;
                }
                sd_ctx->sd->vae_tiling_params.enabled    = true;
                sd_ctx->sd->vae_tiling_params.rel_size_x = enc_tile;
                sd_ctx->sd->vae_tiling_params.rel_size_y = enc_tile;
                // NOTE: the LTX VAE temporal_tile_frames arg is DECODE-ONLY (the streaming path is
                // gated on decode_graph @ ltx_vae.hpp:1449; there is no encode_tiled_chunk), so it is
                // INERT for this encode — the encoder builds one whole-clip graph and the compute
                // buffer scales with frame count (~2.6 GB @193f). The temporal buffer bound now comes
                // from the host-level chunked encode below (encode_relip_reference_temporal_chunked,
                // FIX 1), driven by LTXAV_RELIP_ENCODE_TFRAMES. We still set the (harmless) arg for
                // parity with the decode tiling config; spatial tiling (LTXAV_RELIP_ENCODE_TILE) is
                // the only VAE-level tiling that actually reduces the encode buffer.
                int enc_tframes = 1;
                if (const char* e = getenv("LTXAV_RELIP_ENCODE_TFRAMES")) {
                    int v = atoi(e);
                    if (v >= 1) enc_tframes = v;
                }
                relip_enc_tiling_args = "temporal_tile_frames=" + std::to_string(enc_tframes) + ",temporal_tile_overlap=0";
                sd_ctx->sd->vae_tiling_params.temporal_tiling   = true;
                sd_ctx->sd->vae_tiling_params.extra_tiling_args = relip_enc_tiling_args.c_str();
                LOG_INFO("LTXAV relip: encoding reference with spatial tile %.2f (temporal buffer bound via host-chunked encode, TFRAMES=%d)",
                         enc_tile, enc_tframes);
            }
            sd::Tensor<float> reference_latent;
            if (const char* e = getenv("LTXAV_RELIP_ENCODE_TFRAMES")) {
                // FIX 1: host-level temporal-chunked encode (bounds the encode buffer at high frame
                // counts). Default (env unset) keeps the monolithic single-graph encode.
                reference_latent = encode_relip_reference_temporal_chunked(sd_ctx, ref_video, atoi(e));
            } else {
                reference_latent = sd_ctx->sd->encode_first_stage(ref_video);  // [Wl, Hl, Tl, Cl, 1]
            }
            sd_ctx->sd->vae_tiling_params = relip_saved_tiling;                 // restore for the output decode
            if (reference_latent.empty()) {
                LOG_ERROR("failed to encode LTXAV relip reference video");
                return std::nullopt;
            }
            // Free the video-VAE encode compute buffer NOW. The text-encoder (gemma) runs next
            // and its graph-cut weights would otherwise co-reside with this ~4.4 GB encode buffer
            // for one transient spike (measured 15.2 GB peak vs an 11.3 GB steady state at
            // 1280x704). The VAE compute buffer re-allocates lazily at the output decode, so this
            // costs nothing but drops the relip's peak under the prod VRAM ceiling.
            sd_ctx->sd->first_stage_model->free_compute_buffer();

            // Target = full-length video noise + fully-denoised mask (the whole frame is
            // re-generated; the reference + lipdub LoRA constrain it to the input clip).
            latents.init_latent  = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);

            LOG_INFO("LTXAV LIPDUB RELIP: reference %lld latent frames over %lld target frames",
                     (long long)reference_latent.shape()[2], (long long)latents.init_latent.shape()[2]);
            if (relip_separable) {
                // SEPARABLE path (half-res via DOWNSCALE and/or temporal-subsampled via TSTRIDE).
                // Keep the sampler grid = target only (already set above: full-length noise +
                // all-ones denoise_mask). Hand the reference latent to the DiT as a separate token
                // block (latents.video_reference) with combined target+ref RoPE positions; the DiT
                // slices the ref tokens off before unpatchify so the output is target-only => the
                // post-sampling crop is a no-op (video_conditioning_frame_count left 0).
                if (reference_latent.shape()[3] != latents.init_latent.shape()[3]) {
                    LOG_ERROR("LTXAV relip separable: reference channels %lld != target %lld",
                              (long long)reference_latent.shape()[3], (long long)latents.init_latent.shape()[3]);
                    return std::nullopt;
                }
                // TEMPORAL subsample (LTXAV_RELIP_REF_TSTRIDE>1): keep every T-th reference LATENT
                // frame at full spatial resolution. The kept frames retain their original-frame
                // identity; their true timeline coordinate (frame j -> original j*T) is restored
                // via ref_temporal_stride in the position builder below.
                if (relip_ref_tstride > 1 && reference_latent.shape()[2] > 1) {
                    int64_t orig_f = reference_latent.shape()[2];
                    int64_t n_sub  = (orig_f + relip_ref_tstride - 1) / relip_ref_tstride;
                    std::vector<int64_t> sub_shape = reference_latent.shape();
                    sub_shape[2]                   = n_sub;
                    sd::Tensor<float> sub(sub_shape);
                    int64_t j = 0;
                    for (int64_t f = 0; f < orig_f; f += relip_ref_tstride) {
                        sd::ops::slice_assign(&sub, 2, j, j + 1, sd::ops::slice(reference_latent, 2, f, f + 1));
                        ++j;
                    }
                    reference_latent = std::move(sub);
                    LOG_INFO("LTXAV relip: TEMPORAL subsample tstride=%d -> %lld of %lld ref latent frames (FULL spatial res)",
                             relip_ref_tstride, (long long)n_sub, (long long)orig_f);
                }
                limit_relip_reference_latent_frames(&reference_latent, "stage1");
                int64_t target_lat_frames = latents.init_latent.shape()[2];
                int64_t ref_lat_frames    = reference_latent.shape()[2];
                int64_t target_lat_w      = latents.init_latent.shape()[0];
                int64_t target_lat_h      = latents.init_latent.shape()[1];
                int64_t ref_lat_w         = reference_latent.shape()[0];
                int64_t ref_lat_h         = reference_latent.shape()[1];
                LOG_INFO("LTXAV two-stage: stage1 ref encode %lldx%lld px -> ref latent grid %lldx%lld x %lld frames (downscale=%d); target latent %lldx%lld",
                         (long long)ref_enc_w, (long long)ref_enc_h,
                         (long long)ref_lat_w, (long long)ref_lat_h, (long long)reference_latent.shape()[2],
                         relip_ref_downscale, (long long)target_lat_w, (long long)target_lat_h);
                latents.video_reference   = reference_latent;  // [Wl/N, Hl/N, ceil(ref/T), Cl]
                latents.video_target_frame_count       = target_lat_frames;
                latents.video_conditioning_frame_count = 0;  // output already target-only => crop no-op
                // Combined positions: target block at spatial_scale=vae_scale_factor; ref block at
                // vae_scale_factor*N (each half-res cell spans the same pixel extent) with
                // ref_temporal_stride=T so each subsampled ref frame j carries original frame j*T's
                // timeline coordinate.
                latents.video_positions = build_ltxv_relip_video_positions(target_lat_w,
                                                                           target_lat_h,
                                                                           target_lat_frames,
                                                                           ref_lat_frames,
                                                                           request->fps,
                                                                           request->vae_scale_factor,
                                                                           8,
                                                                           true,
                                                                           ref_lat_w,
                                                                           ref_lat_h,
                                                                           request->vae_scale_factor * relip_ref_downscale,
                                                                           relip_ref_tstride);
                LOG_INFO("LTXAV relip separable: target tokens %lld + ref tokens %lld (downscale=%d tstride=%d) = %lld (vs %lld at full-res concat)",
                         (long long)(target_lat_w * target_lat_h * target_lat_frames),
                         (long long)(ref_lat_w * ref_lat_h * ref_lat_frames),
                         relip_ref_downscale,
                         relip_ref_tstride,
                         (long long)(target_lat_w * target_lat_h * target_lat_frames + ref_lat_w * ref_lat_h * ref_lat_frames),
                         (long long)(target_lat_w * target_lat_h * (target_lat_frames + ref_lat_frames)));
            } else if (!apply_ltxav_video_relip_reference(&latents, reference_latent, request->fps, request->vae_scale_factor)) {
                return std::nullopt;
            }
            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage (relip reference) completed, taking %" PRId64 " ms", t2 - t1);
        } else if (sd_vid_gen_params->v2v_mode == 2 &&
                   (sd_vid_gen_params->control_frames_size > 0 ||
                    SAFE_STR(sd_vid_gen_params->v2v_guide_latent_path)[0] != '\0')) {
            // GENERIC V2V GUIDE-EDIT (Director-2 "LTXDirectorGuide"): "keep the scene, add an
            // element." This is the SAME conditioning primitive as the i2v / keyframe guide
            // (VAE-encode reference frame(s) -> append them as guide tokens at their timeline
            // positions -> hold them via denoise_mask = 1 - guide_strength), but applied to the
            // WHOLE source clip instead of a single image, and run over the FULL schedule. The
            // target block denoises freely (mask all-ones) and attends to the clean-ish guide
            // frames at aligned positions, so the source structure is reconstructed where the guide
            // is strong (guide_strength->1) and the prompt takes over where it is weak.
            //
            // It is NOT SDEdit (v2v_mode==1, which seeds the init latent and TRUNCATES the sigma
            // schedule — a restyle that washes structure out over a deep denoise). And unlike relip
            // (v2v_mode==0) it needs NO lipdub IC-LoRA and NO frozen drive-audio: the audio channels
            // stay the plain generated t2v stream (set above; muxed/replaced downstream). The caller
            // forces the LTX2 scheduler over the FULL step budget for this path (no LTX_CUSTOM_SIGMAS).
            int64_t t1 = ggml_time_ms();

            // Target = full-length video noise + all-ones denoise mask (every output frame denoises).
            latents.init_latent              = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask             = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            int64_t target_lat_frames        = latents.init_latent.shape()[2];
            latents.video_target_frame_count = target_lat_frames;
            int64_t Wl                       = latents.init_latent.shape()[0];
            int64_t Hl                       = latents.init_latent.shape()[1];
            int64_t Cl                       = latents.init_latent.shape()[3];

            // Two ways to obtain the guide latent [Wl,Hl,lf,Cl,1]:
            //   (A) LATENT-IN (PREFERRED — the source is a shot WE rendered): load the banked
            //       diffusion VIDEO latent (the seg_<i>.bin the chain already writes) straight from
            //       v2v_guide_latent_path. NO pixel decode->re-encode roundtrip and its compression
            //       artifacts — cleaner + faster; the whole point of staying in latent space.
            //   (B) PIXEL-ENCODE fallback (FOREIGN/uploaded clip, no banked latent): VAE-encode
            //       control_frames one-graph (like the SDEdit path; long guides can OOM the encode
            //       buffer — the relip path's encode-tiling knobs could be ported if needed).
            // Both produce a 5D [Wl,Hl,lf,Cl,1] latent matching the target grid, appended below.
            sd::Tensor<float> guide_latent;
            int               lf        = 0;
            const char*       lat_path  = SAFE_STR(sd_vid_gen_params->v2v_guide_latent_path);
            const bool        latent_in = (lat_path[0] != '\0');
            if (latent_in) {
                sd::Tensor<float> loaded;
                try {
                    loaded = sd::load_tensor_from_file_as_tensor<float>(lat_path);
                } catch (const std::exception& e) {
                    LOG_ERROR("LTXAV V2V guide-edit: failed to load banked guide latent %s: %s", lat_path, e.what());
                    return std::nullopt;
                }
                if (loaded.empty() || loaded.dim() < 4) {
                    LOG_ERROR("LTXAV V2V guide-edit: banked guide latent %s empty/malformed", lat_path);
                    return std::nullopt;
                }
                if (loaded.shape()[0] != Wl || loaded.shape()[1] != Hl || loaded.shape()[3] != Cl) {
                    LOG_ERROR("LTXAV V2V guide-edit: banked guide grid %lldx%lldx%lldc != target %lldx%lldx%lldc "
                              "(must match render width/height/model)",
                              (long long)loaded.shape()[0], (long long)loaded.shape()[1], (long long)loaded.shape()[3],
                              (long long)Wl, (long long)Hl, (long long)Cl);
                    return std::nullopt;
                }
                lf = (int)loaded.shape()[2];
                // Normalise to 5D [Wl,Hl,lf,Cl,1] (a banked latent may be saved as 4D) so the concat
                // below matches generate_init_latent's layout. numel == Wl*Hl*lf*Cl either way.
                guide_latent = sd::Tensor<float>({Wl, Hl, (int64_t)lf, Cl, 1});
                std::memcpy(guide_latent.data(), loaded.data(), (size_t)Wl * Hl * lf * Cl * sizeof(float));
                LOG_INFO("LTXAV V2V guide-edit: LATENT-IN guide %s (%d latent frames, no re-encode)", lat_path, lf);
            } else {
                if (sd_ctx->sd->vae_decode_only) {
                    LOG_ERROR("LTXAV V2V guide-edit (pixel encode) requires VAE encoder weights; create the context with vae_decode_only=false");
                    return std::nullopt;
                }
                int    lw = 0, lh = 0, lc = 0;
                float* src = sd_ctx_encode_video_frames(sd_ctx, sd_vid_gen_params->control_frames,
                                                        sd_vid_gen_params->control_frames_size,
                                                        request->width, request->height, &lw, &lh, &lf, &lc);
                if (src == nullptr) {
                    LOG_ERROR("LTXAV V2V guide-edit: source VAE encode failed");
                    return std::nullopt;
                }
                if (lw != (int)Wl || lh != (int)Hl || lc != (int)Cl) {
                    LOG_ERROR("LTXAV V2V guide-edit: source latent grid %dx%dx%dc != target %lldx%lldx%lldc",
                              lw, lh, lc, (long long)Wl, (long long)Hl, (long long)Cl);
                    free(src);
                    return std::nullopt;
                }
                guide_latent = sd::Tensor<float>({(int64_t)lw, (int64_t)lh, (int64_t)lf, (int64_t)lc, 1});
                std::memcpy(guide_latent.data(), src, (size_t)lw * lh * lf * lc * sizeof(float));
                free(src);
                // Free the VAE encode compute buffer before the text-encoder runs (VRAM hygiene).
                sd_ctx->sd->first_stage_model->free_compute_buffer();
                LOG_INFO("LTXAV V2V guide-edit: PIXEL-ENCODE guide from %d source frames (%d latent frames)",
                         sd_vid_gen_params->control_frames_size, lf);
            }
            if (lf != (int)target_lat_frames) {
                // Not fatal — the guide covers its own lf latent frames on the timeline; but a 1:1
                // guide (source length == output length) is what "edit this clip" wants.
                LOG_WARN("LTXAV V2V guide-edit: guide %d latent frames != target %lld (expect 1:1)",
                         lf, (long long)target_lat_frames);
            }

            // Director-2 "edit" strength = SDEdit denoise fraction: higher = bigger change from the
            // prompt, lower = keep more of the source scene (~0.45 adds a localized element while
            // holding the scene). Drives the sigma-schedule truncation in generate_video's SDEdit
            // block (via v2v_guide_strength). 0 falls back to a full re-render.
            float guide_strength = sd_vid_gen_params->v2v_guide_strength;
            if (guide_strength <= 0.f) {
                guide_strength = 1.f;
            }
            guide_strength = std::clamp(guide_strength, 0.f, 1.f);

            // BASE-NATIVE structure-preserving edit = SDEdit — the mechanism Director-2's LTXVAddGuide
            // uses for a whole-clip video guide (traced from comfy_extras/nodes_lt.py + the LTX
            // pipeline): VAE-encode the source CLEAN, seed it into the init latent, then start sampling
            // from a TRUNCATED sigma so only the top (guide_strength) fraction of the schedule denoises.
            // Low-frequency scene structure survives everywhere and the new element enters through the
            // PROMPT — NO IC-LoRA, NO spatial mask, NO appended tokens.
            //
            // This REPLACES the old per-step partial-freeze mask (denoise_mask = 1 - guide_strength +
            // clean-x0 re-injection every step). That mixed clean x0 into a near-pure-noise field at
            // high sigma -> horizontal striping. LTX never injects clean x0 mid-schedule; it places the
            // conditioning token at a schedule-consistent noise level (lerp(noise, x0, strength)), which
            // the SDEdit truncation does for the whole seeded clip. So mode==2 is now bit-identical to
            // the mode==1 restyle path, differing only in the LATENT-IN guide source (edit a banked
            // render with no re-encode) + the edit-tuned strength default. Standard t2v positions,
            // all-ones mask (truncation, not a freeze mask, preserves the scene).
            if (lf != (int)target_lat_frames) {
                LOG_ERROR("LTXAV V2V guide-edit: source %d latent frames != target %lld (need a 1:1 clip)",
                          lf, (long long)target_lat_frames);
                return std::nullopt;
            }
            std::memcpy(latents.init_latent.data(), guide_latent.data(),
                        (size_t)Wl * (size_t)Hl * (size_t)lf * (size_t)Cl * sizeof(float));
            latents.denoise_mask                   = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            latents.video_conditioning_frame_count = 0;
            latents.v2v_sdedit                     = true;
            int64_t t2 = ggml_time_ms();
            LOG_INFO("LTXAV V2V EDIT (SDEdit): %s guide -> %d latent frames over %lld target "
                     "(edit strength=%.2f, truncated schedule); prep %" PRId64 " ms",
                     latent_in ? "LATENT-IN" : "PIXEL-ENCODE", lf, (long long)target_lat_frames,
                     guide_strength, t2 - t1);
        } else if (sd_vid_gen_params->keyframes != nullptr && sd_vid_gen_params->keyframes_size > 0 &&
                   sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0) {
            // MERGED (Director v2): a CONTINUATION shot that ALSO pins identity keyframes mid-flow
            // (a reveal / identity swap without restarting the motion). The prior segment's motion
            // tail rides as a held guide at frame 0 (temporal_scale-wide), and each keyframe is a
            // frozen instant at its own frame index; the DiT continues the motion while snapping to
            // the pinned images. Placed ABOVE the keyframe-only and continuation branches so it only
            // fires when BOTH are present.
            if (sd_ctx->sd->vae_decode_only) {
                LOG_ERROR("LTXAV merged continuation+keyframe conditioning requires VAE encoder weights");
                return std::nullopt;
            }
            int64_t t1                       = ggml_time_ms();
            latents.init_latent              = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask             = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            int64_t target_lat_frames        = latents.init_latent.shape()[2];
            latents.video_target_frame_count = target_lat_frames;

            float conditioning_strength = std::clamp(request->strength, 0.f, 1.f);
            float conditioned_mask      = 1.0f - conditioning_strength;

            int64_t Wl = latents.init_latent.shape()[0];
            int64_t Hl = latents.init_latent.shape()[1];
            int64_t Cl = latents.init_latent.shape()[3];
            int64_t K  = sd_vid_gen_params->cont_latent_frames;
            if (K > target_lat_frames) {
                LOG_ERROR("cont latent frames %lld exceed segment latent frames %lld",
                          (long long)K, (long long)target_lat_frames);
                return std::nullopt;
            }
            float omask = 0.0f;  // motion tail frozen (same env knob as the pure-continuation path)
            if (const char* e = std::getenv("LTXAV_CONT_OVERLAP_MASK")) {
                omask = std::clamp((float)atof(e), 0.f, 1.f);
            }

            std::vector<LtxvGuideSpec> guides;
            int64_t                    appended = 0;

            // 1. continuation motion tail (held, temporal-window frames) at frame 0.
            {
                sd::Tensor<float> cont_tail({Wl, Hl, K, Cl, 1});
                std::memcpy(cont_tail.data(), sd_vid_gen_params->cont_latent, (size_t)cont_tail.numel() * sizeof(float));
                latents.init_latent  = sd::ops::concat(latents.init_latent, cont_tail, 2);
                auto cont_mask       = sd::full<float>({Wl, Hl, K, 1, 1}, omask);
                latents.denoise_mask = sd::ops::concat(latents.denoise_mask, cont_mask, 2);
                guides.push_back({0, (int)K, 8});
                appended += K;
            }

            // 2. identity keyframes (frozen instants) at their frame indices.
            for (int i = 0; i < sd_vid_gen_params->keyframes_size; ++i) {
                int frame_idx = sd_vid_gen_params->keyframe_frame_indices != nullptr
                                    ? sd_vid_gen_params->keyframe_frame_indices[i]
                                    : 0;
                if (frame_idx < 0 || frame_idx >= request->frames) {
                    LOG_ERROR("LTXAV merged keyframe %d frame index %d out of range [0, %d)", i, frame_idx, request->frames);
                    return std::nullopt;
                }
                if (sd_vid_gen_params->keyframes[i].data == nullptr) {
                    LOG_ERROR("LTXAV merged keyframe %d has null image data", i);
                    return std::nullopt;
                }
                sd::Tensor<float> kf_image  = sd_image_to_tensor(sd_vid_gen_params->keyframes[i], request->width, request->height);
                auto              kf_latent = encode_ltxav_condition_image(sd_ctx, kf_image, "keyframe");
                if (kf_latent.empty()) {
                    return std::nullopt;
                }
                if (kf_latent.shape()[0] != Wl || kf_latent.shape()[1] != Hl || kf_latent.shape()[3] != Cl) {
                    LOG_ERROR("invalid LTXAV merged keyframe %d latent shape", i);
                    return std::nullopt;
                }
                int64_t kf_frames    = kf_latent.shape()[2];
                latents.init_latent  = sd::ops::concat(latents.init_latent, kf_latent, 2);
                auto kf_mask         = sd::full<float>({Wl, Hl, kf_frames, 1, 1}, conditioned_mask);
                latents.denoise_mask = sd::ops::concat(latents.denoise_mask, kf_mask, 2);
                guides.push_back({frame_idx, (int)kf_frames, 1});
                appended += kf_frames;
                LOG_INFO("LTXAV merged keyframe %d/%d pinned at frame %d (continuation K=%lld held, strength=%.2f)",
                         i + 1, sd_vid_gen_params->keyframes_size, frame_idx, (long long)K, conditioning_strength);
            }

            latents.video_conditioning_frame_count = appended;
            latents.video_positions                = build_ltxv_guides_video_positions(
                Wl, Hl, target_lat_frames, guides, request->fps, request->vae_scale_factor, 8, true);
            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage (continuation + %d merged keyframes) completed, taking %" PRId64 " ms",
                     sd_vid_gen_params->keyframes_size, t2 - t1);
        } else if (sd_vid_gen_params->keyframes != nullptr && sd_vid_gen_params->keyframes_size > 0) {
            // LTXAV MULTI-KEYFRAME conditioning: arbitrary (image, latent-frame-index) pairs pinned
            // as frozen 1-frame guides on the target timeline. Generalises the start/end i2v path
            // (one image at latent idx 0, optionally one at frames-1) to N caller-placed images.
            // Each keyframe is VAE-encoded to a 1-frame latent, appended after the target block,
            // held via the denoise mask, and given its own frame_idx RoPE position; the DiT
            // interpolates the generated (denoised) frames between the pinned keyframes. Placed
            // BEFORE the single start/end branch so keyframes_size==0 leaves i2v/t2v untouched.
            if (sd_ctx->sd->vae_decode_only) {
                LOG_ERROR("LTXAV keyframe conditioning requires VAE encoder weights; create the context with vae_decode_only=false");
                return std::nullopt;
            }

            int64_t t1          = ggml_time_ms();
            latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);

            float conditioning_strength = std::clamp(request->strength, 0.f, 1.f);
            float conditioned_mask      = 1.0f - conditioning_strength;
            latents.denoise_mask        = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);

            int64_t target_lat_frames        = latents.init_latent.shape()[2];
            latents.video_target_frame_count = target_lat_frames;

            // frame_idx is a VIDEO (pixel) frame index on the shared timeline — the SAME unit the
            // end-image i2v path uses (it pins at request->frames - 1) and the unit
            // build_ltxv_video_positions expects for keyframe_frame_idx (added to pixel-corner
            // values, then /fps). Validate against the pixel frame count, not the latent count.
            std::vector<int> keyframe_positions;  // one entry per appended keyframe latent frame
            int64_t          appended_frames = 0;
            for (int i = 0; i < sd_vid_gen_params->keyframes_size; ++i) {
                int frame_idx = sd_vid_gen_params->keyframe_frame_indices != nullptr
                                    ? sd_vid_gen_params->keyframe_frame_indices[i]
                                    : 0;
                if (frame_idx < 0 || frame_idx >= request->frames) {
                    LOG_ERROR("LTXAV keyframe %d frame index %d out of range [0, %d) video frames",
                              i, frame_idx, request->frames);
                    return std::nullopt;
                }
                if (sd_vid_gen_params->keyframes[i].data == nullptr) {
                    LOG_ERROR("LTXAV keyframe %d has null image data", i);
                    return std::nullopt;
                }
                sd::Tensor<float> kf_image = sd_image_to_tensor(sd_vid_gen_params->keyframes[i], request->width, request->height);
                auto kf_latent = encode_ltxav_condition_image(sd_ctx, kf_image, "keyframe");
                if (kf_latent.empty()) {
                    return std::nullopt;
                }
                if (kf_latent.shape()[0] != latents.init_latent.shape()[0] ||
                    kf_latent.shape()[1] != latents.init_latent.shape()[1] ||
                    kf_latent.shape()[3] != latents.init_latent.shape()[3]) {
                    LOG_ERROR("invalid LTXAV keyframe %d latent shape", i);
                    return std::nullopt;
                }
                int64_t kf_frames    = kf_latent.shape()[2];
                latents.init_latent  = sd::ops::concat(latents.init_latent, kf_latent, 2);
                auto kf_mask         = sd::full<float>({kf_latent.shape()[0],
                                                        kf_latent.shape()[1],
                                                        kf_frames,
                                                        1,
                                                        1},
                                               conditioned_mask);
                latents.denoise_mask = sd::ops::concat(latents.denoise_mask, kf_mask, 2);
                for (int64_t f = 0; f < kf_frames; ++f) {
                    keyframe_positions.push_back(frame_idx);
                }
                appended_frames += kf_frames;
                LOG_INFO("LTXAV keyframe %d/%d pinned at video frame %d (%lld latent frame(s), strength=%.2f)",
                         i + 1, sd_vid_gen_params->keyframes_size, frame_idx, (long long)kf_frames, conditioning_strength);
            }

            latents.video_conditioning_frame_count = appended_frames;
            latents.video_positions = build_ltxv_multi_keyframe_video_positions(latents.init_latent.shape()[0],
                                                                                latents.init_latent.shape()[1],
                                                                                target_lat_frames,
                                                                                keyframe_positions,
                                                                                request->fps,
                                                                                request->vae_scale_factor,
                                                                                8,
                                                                                true);

            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage (%d keyframes) completed, taking %" PRId64 " ms",
                     sd_vid_gen_params->keyframes_size, t2 - t1);
        } else if (sd_vid_gen_params->end_cont_latent != nullptr && sd_vid_gen_params->end_cont_latent_frames > 0) {
            // RETAKE BIDIRECTIONAL PIN (true single-segment retake). Re-render THIS segment pinned
            // by BOTH neighbours: the START by seg_{N-1}'s tail (cont_latent, a held guide at frame
            // 0 — the ordinary continuation) OR a fresh opener image (init_image at target frame 0),
            // and the END by seg_{N+1}'s HEAD latent frames (end_cont_latent), appended as a frozen
            // guide at a FUTURE timeline position so this segment's generated tail converges to the
            // next segment's opening (N ends exactly where the unchanged, banked N+1 begins).
            // Composed with build_ltxv_guides_video_positions (the same multi-guide machinery as the
            // merged continuation+keyframe path). Fires ONLY when end_cont_latent is set (retake
            // mode), so every ordinary render is byte-identical.
            if (sd_ctx->sd->vae_decode_only && !start_image.empty()) {
                LOG_ERROR("LTXAV retake opener image conditioning requires VAE encoder weights");
                return std::nullopt;
            }
            int64_t t1                       = ggml_time_ms();
            latents.init_latent              = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask             = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            int64_t target_lat_frames        = latents.init_latent.shape()[2];
            latents.video_target_frame_count = target_lat_frames;

            const int64_t Wl             = latents.init_latent.shape()[0];
            const int64_t Hl             = latents.init_latent.shape()[1];
            const int64_t Cl             = latents.init_latent.shape()[3];
            const int     temporal_scale = 8;

            float conditioning_strength = std::clamp(request->strength, 0.f, 1.f);

            float omask = 0.0f;  // start motion-tail frozen (same knob as the pure-continuation path)
            if (const char* e = std::getenv("LTXAV_CONT_OVERLAP_MASK")) {
                omask = std::clamp((float)atof(e), 0.f, 1.f);
            }
            float end_mask = 0.0f;  // end pin frozen by default; loosen with LTXAV_RETAKE_END_MASK
            if (const char* e = std::getenv("LTXAV_RETAKE_END_MASK")) {
                end_mask = std::clamp((float)atof(e), 0.f, 1.f);
            }

            std::vector<LtxvGuideSpec> guides;

            // ── START pin ────────────────────────────────────────────────────────────────
            if (!start_image.empty()) {
                // Fresh opener / scene-cut retake (N==0 or an image-anchored shot): pin the image at
                // target frame 0 (part of the target block — no appended guide token).
                if (!apply_ltxav_condition_image_by_latent_index(sd_ctx, start_image, &latents.init_latent,
                                                                 &latents.denoise_mask, 0, "retake-init",
                                                                 conditioning_strength)) {
                    return std::nullopt;
                }
                LOG_INFO("LTXAV RETAKE: start pinned by opener image (target frame 0)");
            } else if (sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0) {
                // Continuation retake: seg_{N-1}'s tail as a held guide at frame 0.
                int64_t K = sd_vid_gen_params->cont_latent_frames;
                if (K > target_lat_frames) {
                    LOG_ERROR("retake cont latent frames %lld exceed segment latent frames %lld",
                              (long long)K, (long long)target_lat_frames);
                    return std::nullopt;
                }
                sd::Tensor<float> cont_tail({Wl, Hl, K, Cl, 1});
                std::memcpy(cont_tail.data(), sd_vid_gen_params->cont_latent, (size_t)cont_tail.numel() * sizeof(float));
                latents.init_latent  = sd::ops::concat(latents.init_latent, cont_tail, 2);
                auto cont_mask       = sd::full<float>({Wl, Hl, K, 1, 1}, omask);
                latents.denoise_mask = sd::ops::concat(latents.denoise_mask, cont_mask, 2);
                guides.push_back({0, (int)K, temporal_scale});
                LOG_INFO("LTXAV RETAKE: start pinned by prior tail (%lld held frames at frame_idx 0, mask=%.2f)",
                         (long long)K, omask);
            } else {
                LOG_INFO("LTXAV RETAKE: no start pin (fresh opener from noise)");
            }

            // ── END pin (seg_{N+1} head), appended AFTER any start guide so the tail-crop
            //    (video_conditioning_frame_count) removes both appended blocks together. ────
            {
                int64_t Ke = sd_vid_gen_params->end_cont_latent_frames;
                if (Ke > target_lat_frames) {
                    LOG_ERROR("retake end_cont latent frames %lld exceed segment latent frames %lld",
                              (long long)Ke, (long long)target_lat_frames);
                    return std::nullopt;
                }
                sd::Tensor<float> end_head({Wl, Hl, Ke, Cl, 1});
                std::memcpy(end_head.data(), sd_vid_gen_params->end_cont_latent, (size_t)end_head.numel() * sizeof(float));
                latents.init_latent  = sd::ops::concat(latents.init_latent, end_head, 2);
                auto end_msk         = sd::full<float>({Wl, Hl, Ke, 1, 1}, end_mask);
                latents.denoise_mask = sd::ops::concat(latents.denoise_mask, end_msk, 2);
                // Shadow the target's CLOSING Ke frames (symmetric to the start guide shadowing the
                // opening K): place the guide at the pixel position of target latent frame
                // (target_lat_frames - Ke) so the tail re-renders toward the next segment's head.
                // Override with LTXAV_RETAKE_END_FRAME_IDX (pixel-frame units) for GPU tuning.
                int end_frame_idx = (int)ltxv_latent_corner_to_pixel_frame(target_lat_frames - Ke, temporal_scale, true);
                if (const char* e = std::getenv("LTXAV_RETAKE_END_FRAME_IDX")) {
                    end_frame_idx = atoi(e);
                }
                guides.push_back({end_frame_idx, (int)Ke, temporal_scale});
                LOG_INFO("LTXAV RETAKE: end pinned by next-segment head (%lld frozen frames at frame_idx %d, mask=%.2f)",
                         (long long)Ke, end_frame_idx, end_mask);
            }

            int64_t appended = 0;
            for (const auto& g : guides) {
                appended += g.latent_frames;
            }
            latents.video_conditioning_frame_count = appended;
            latents.video_positions                = build_ltxv_guides_video_positions(
                Wl, Hl, target_lat_frames, guides, request->fps, request->vae_scale_factor, temporal_scale, true);
            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage (RETAKE bidirectional pin, %zu guide block(s)) completed, taking %" PRId64 " ms",
                     guides.size(), t2 - t1);
        } else if (!start_image.empty() || !end_image.empty()) {
            if (sd_ctx->sd->vae_decode_only) {
                LOG_ERROR("LTXAV image conditioning requires VAE encoder weights; create the context with vae_decode_only=false");
                return std::nullopt;
            }

            if (!start_image.empty() && !end_image.empty()) {
                LOG_INFO("FLF2V");
            } else if (!start_image.empty()) {
                LOG_INFO("IMG2VID");
            } else {
                LOG_INFO("END2VID");
            }

            int64_t t1          = ggml_time_ms();
            latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);

            float conditioning_strength = std::clamp(request->strength, 0.f, 1.f);
            float conditioned_mask      = 1.0f - conditioning_strength;
            latents.denoise_mask        = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);

            auto apply_video_condition_by_keyframe_index = [&](const sd::Tensor<float>& keyframes,
                                                               int frame_idx,
                                                               const char* name) -> bool {
                int64_t keyframe_frames = keyframes.shape()[2];
                if (keyframe_frames <= 0 || keyframes.shape()[0] != latents.init_latent.shape()[0] ||
                    keyframes.shape()[1] != latents.init_latent.shape()[1] ||
                    keyframes.shape()[3] != latents.init_latent.shape()[3]) {
                    LOG_ERROR("invalid LTXAV %s keyframe latent shape", name);
                    return false;
                }

                latents.video_target_frame_count       = latents.init_latent.shape()[2];
                latents.video_conditioning_frame_count = keyframe_frames;
                latents.init_latent                    = sd::ops::concat(latents.init_latent, keyframes, 2);

                auto keyframe_mask      = sd::full<float>({keyframes.shape()[0],
                                                           keyframes.shape()[1],
                                                           keyframes.shape()[2],
                                                           1,
                                                           1},
                                                     conditioned_mask);
                latents.denoise_mask    = sd::ops::concat(latents.denoise_mask, keyframe_mask, 2);
                latents.video_positions = build_ltxv_video_positions(latents.init_latent.shape()[0],
                                                                     latents.init_latent.shape()[1],
                                                                     latents.video_target_frame_count,
                                                                     keyframe_frames,
                                                                     frame_idx,
                                                                     1,
                                                                     request->fps,
                                                                     request->vae_scale_factor,
                                                                     8,
                                                                     true);
                return true;
            };

            if (!start_image.empty()) {
                if (!apply_ltxav_condition_image_by_latent_index(sd_ctx,
                                                                 start_image,
                                                                 &latents.init_latent,
                                                                 &latents.denoise_mask,
                                                                 0,
                                                                 "init",
                                                                 conditioning_strength)) {
                    return std::nullopt;
                }
            }

            if (!end_image.empty()) {
                auto end_image_latent = encode_ltxav_condition_image(sd_ctx, end_image, "end");
                if (end_image_latent.empty()) {
                    return std::nullopt;
                }

                int frame_idx = request->frames - 1;
                bool ok       = frame_idx == 0 ? apply_ltxav_condition_by_latent_index(&latents.init_latent,
                                                                                       &latents.denoise_mask,
                                                                                       end_image_latent,
                                                                                       0,
                                                                                       "end",
                                                                                       conditioned_mask)
                                               : apply_video_condition_by_keyframe_index(end_image_latent, frame_idx, "end");
                if (!ok) {
                    return std::nullopt;
                }
            }

            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
        } else if (sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0) {
            // LTXAV IN-MEMORY LATENT CONTINUATION (in-process N-segment chaining): identical
            // semantics to the file-based --cont-latent path below, but the prior segment's
            // motion-carrying tail arrives as a contiguous float* (cont_latent) instead of a
            // file. This keeps the ~11GB DiT RESIDENT across segments (one sd-cli process
            // renders N segments; no per-segment reload). The caller (examples/cli/main.cpp
            // --ltx-chain-segments loop) slices the prior segment's returned latent to exactly
            // K = cont_latent_frames tail frames AND the first get_latent_channel() (==128)
            // VIDEO channels (audio channels stripped), in ggml-ne order [Wl, Hl, K, Cl] with
            // W fastest. We rebuild that as an sd::Tensor and apply it with the SAME logic as
            // the file path: place at the head of init_latent, hold (graded) via denoise mask.
            latents.init_latent  = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            // Geometry from the segment's own init latent so Wl/Hl/Cl always match what
            // apply_ltxav_condition_by_latent_index expects (shape()[0/1/3] equality). The
            // init_latent is 5D [Wl, Hl, T, Cl, 1]; cont_tail is 4D [Wl, Hl, K, Cl] (the file
            // path's cont_tail is also 4D after slicing the saved 4D video latent), and the
            // apply check compares shape()[3]==Cl only, so 4D is fine.
            int64_t Wl = latents.init_latent.shape()[0];
            int64_t Hl = latents.init_latent.shape()[1];
            int64_t Cl = latents.init_latent.shape()[3];  // == get_latent_channel() == 128 for LTXAV (VIDEO channels)
            int64_t K  = sd_vid_gen_params->cont_latent_frames;
            // 5D [Wl,Hl,K,Cl,1] to match init_latent's rank — apply_ltxav_condition_by_latent_index
            // -> slice_assign requires equal rank (the file path's cont_full is 5D after slicing).
            sd::Tensor<float> cont_tail({Wl, Hl, K, Cl, 1});
            std::memcpy(cont_tail.data(), sd_vid_gen_params->cont_latent,
                        (size_t)cont_tail.numel() * sizeof(float));
            if (K > latents.init_latent.shape()[2]) {
                LOG_ERROR("cont latent frames %lld exceed segment latent frames %lld",
                          (long long)K, (long long)latents.init_latent.shape()[2]);
                return std::nullopt;
            }
            // Overlap mask value: 0 = frozen (carries motion, may stall); >0 lets the overlap
            // re-denoise so motion evolves instead of freezing. Tunable via LTXAV_CONT_OVERLAP_MASK
            // (same env knob as the file path).
            float omask = 0.0f;
            if (const char* e = std::getenv("LTXAV_CONT_OVERLAP_MASK")) {
                omask = std::clamp((float)atof(e), 0.f, 1.f);
            }
            // NOTE: appearance anchor (cont_anchor) is intentionally NOT supported on the
            // in-memory path for now — it's a follow-up (would need an in-memory anchor float*).
            // The in-process chain relies on the motion overlap alone.
            //
            // DEFAULT (Director keyframe convention): append the guide at the TAIL with its own
            // true-past RoPE position and crop it off the output — avoids the high-fps echo/ghost
            // the legacy head-placement causes. LTXAV_CONT_LEGACY_HEAD=1 restores the old
            // head-placement (guide overwrites output frames 0..K) for A/B comparison.
            bool legacy_head = false;
            if (const char* e = std::getenv("LTXAV_CONT_LEGACY_HEAD")) {
                legacy_head = atoi(e) != 0;
            }
            if (legacy_head) {
                int64_t base_idx = 0;
                LOG_INFO("LTXAV CONTINUATION (in-memory, LEGACY head-place): %lld prior latent frames at idx %lld, mask=%.2f",
                         (long long)K, (long long)base_idx, omask);
                if (!apply_ltxav_condition_by_latent_index(&latents.init_latent, &latents.denoise_mask,
                                                           cont_tail, base_idx, "cont-mem", omask)) {
                    return std::nullopt;
                }
            } else {
                // Guide pinned at the segment start (frame_idx 0): this segment re-renders the
                // overlap region (warm-up) then continues. The re-render gives the join frame a
                // settled trajectory (a cold start at a negative frame_idx jumps harder), and the
                // stitcher auto-aligns the trim to the smoothest continuation (ltxav_auto_trim_drop)
                // — this beat guide-in-the-past on the seam metric (raw seam 1.58 vs 3.1).
                int kf_idx = 0;
                if (const char* e = std::getenv("LTXAV_CONT_KEYFRAME_IDX")) {
                    kf_idx = atoi(e);
                }
                LOG_INFO("LTXAV CONTINUATION (in-memory, keyframe-append): %lld prior latent frames at frame_idx %d, mask=%.2f",
                         (long long)K, kf_idx, omask);
                if (!apply_ltxav_video_guide_by_keyframe_index(&latents, cont_tail, kf_idx,
                                                               request->fps, request->vae_scale_factor, omask)) {
                    return std::nullopt;
                }
            }
        } else if (sd_vid_gen_params->cont_latent_path != nullptr && sd_vid_gen_params->cont_latent_path[0] != '\0' &&
                   sd_vid_gen_params->cont_latent_frames > 0) {
            // LTXAV LATENT CONTINUATION: condition on the LAST K diffusion-latent frames of
            // the prior segment (motion-carrying; no pixel decode/re-encode). Place them at
            // the head of init_latent and hold them (graded) fixed; the rest denoise and
            // continue the motion via the DiT's full temporal attention over the T axis.
            sd::Tensor<float> cont_full;
            try {
                cont_full = sd::load_tensor_from_file_as_tensor<float>(sd_vid_gen_params->cont_latent_path);
            } catch (const std::exception& e) {
                LOG_ERROR("failed to load --cont-latent %s: %s", sd_vid_gen_params->cont_latent_path, e.what());
                return std::nullopt;
            }
            int64_t Tprev = cont_full.shape()[2];
            int64_t K     = std::min<int64_t>(sd_vid_gen_params->cont_latent_frames, Tprev);
            auto cont_tail = sd::ops::slice(cont_full, 2, Tprev - K, Tprev);  // last K latent frames
            latents.init_latent  = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            if (K > latents.init_latent.shape()[2]) {
                LOG_ERROR("cont latent frames %lld exceed segment latent frames %lld",
                          (long long)K, (long long)latents.init_latent.shape()[2]);
                return std::nullopt;
            }
            // Overlap mask value: 0 = frozen (carries motion, may stall); >0 lets the overlap
            // re-denoise so motion evolves instead of freezing. Tunable via LTXAV_CONT_OVERLAP_MASK.
            float omask = 0.0f;
            if (const char* e = std::getenv("LTXAV_CONT_OVERLAP_MASK")) {
                omask = std::clamp((float)atof(e), 0.f, 1.f);
            }
            // APPEARANCE ANCHOR (anti-drift): optionally pin the ORIGINAL character (frame 0 of
            // --cont-anchor) at the head so style can't migrate off the source over a long chain.
            // Layout [anchor(1), motion_tail(K), generated...]; anchor+overlap are dropped at stitch.
            int64_t base_idx = 0;
            if (sd_vid_gen_params->cont_anchor_path != nullptr && sd_vid_gen_params->cont_anchor_path[0] != '\0') {
                try {
                    auto anchor_full = sd::load_tensor_from_file_as_tensor<float>(sd_vid_gen_params->cont_anchor_path);
                    auto anchor      = sd::ops::slice(anchor_full, 2, 0, 1);  // original char = frame 0
                    if (apply_ltxav_condition_by_latent_index(&latents.init_latent, &latents.denoise_mask,
                                                              anchor, 0, "cont-anchor", 0.0f)) {
                        base_idx = 1;
                        LOG_INFO("LTXAV CONTINUATION: + appearance anchor pinned at frame 0");
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("failed to load --cont-anchor %s: %s", sd_vid_gen_params->cont_anchor_path, e.what());
                }
            }
            if (base_idx + K > latents.init_latent.shape()[2]) {
                LOG_ERROR("anchor+cont frames (%lld) exceed segment latent frames %lld",
                          (long long)(base_idx + K), (long long)latents.init_latent.shape()[2]);
                return std::nullopt;
            }
            LOG_INFO("LTXAV CONTINUATION: %lld prior latent frames (of %lld) as motion overlap at idx %lld, mask=%.2f",
                     (long long)K, (long long)Tprev, (long long)base_idx, omask);
            if (!apply_ltxav_condition_by_latent_index(&latents.init_latent, &latents.denoise_mask,
                                                       cont_tail, base_idx, "cont", omask)) {
                return std::nullopt;
            }
        } else {
            // Pure text-to-video (no init/end image, no continuation). Without this branch the
            // LTXAV t2v path set no init_latent and fell through to the generic image-latent
            // init ([W,H,C,1], no frames dim) -> split_av_latents misread channels as frames and
            // asserted. Mirror the image-cond branch: full-length video noise + fully-denoised mask.
            LOG_INFO("TXT2VID (LTXAV)");
            latents.init_latent  = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
        }

        // GENERIC V2V (SDEdit): a control-frame source with v2v_mode==1 fell through to the t2v
        // branch above (relip gated off) so init_latent/mask/positions/audio are the plain t2v
        // setup. Now VAE-encode the source clip and overwrite the VIDEO channels of init_latent
        // with it; the caller then truncates the sigma schedule by `strength` so sampling starts
        // from a partial sigma (img2img convention: 1.0 = full re-render / source ignored, lower =
        // keep more of the source). The audio channels stay as the t2v init (they denoise weakly
        // and are discarded — v2v audio is muxed downstream). NULL/0 v2v_mode = byte-identical.
        if (sd_vid_gen_params->v2v_mode == 1 && sd_vid_gen_params->control_frames_size > 0 &&
            latents.init_latent.dim() >= 4) {
            if (sd_ctx->sd->vae_decode_only) {
                LOG_ERROR("LTXAV V2V (SDEdit) requires VAE encoder weights (vae_decode_only=false)");
                return std::nullopt;
            }
            int    lw = 0, lh = 0, lf = 0, lc = 0;
            float* src = sd_ctx_encode_video_frames(sd_ctx, sd_vid_gen_params->control_frames,
                                                    sd_vid_gen_params->control_frames_size,
                                                    request->width, request->height, &lw, &lh, &lf, &lc);
            if (src == nullptr) {
                LOG_ERROR("LTXAV V2V: source VAE encode failed");
                return std::nullopt;
            }
            int64_t Wl       = latents.init_latent.shape()[0];
            int64_t Hl       = latents.init_latent.shape()[1];
            int64_t Tl       = latents.init_latent.shape()[2];
            size_t  src_vals = (size_t)lw * (size_t)lh * (size_t)lf * (size_t)lc;
            if (lw != (int)Wl || lh != (int)Hl || lf != (int)Tl ||
                src_vals > (size_t)latents.init_latent.numel()) {
                LOG_ERROR("LTXAV V2V: source latent %dx%dx%dx%d != target grid %lldx%lldx%lld",
                          lw, lh, lf, lc, (long long)Wl, (long long)Hl, (long long)Tl);
                free(src);
                return std::nullopt;
            }
            // init_latent is ggml-ne [W,H,T,C,1] contiguous: video channels are the first `lc`
            // channels (= the first src_vals floats), audio channels follow. Seed video only.
            std::memcpy(latents.init_latent.data(), src, src_vals * sizeof(float));
            free(src);
            latents.v2v_sdedit = true;
            LOG_INFO("LTXAV V2V (SDEdit): seeded %d source frames into init_latent video channels (strength=%.2f)",
                     sd_vid_gen_params->control_frames_size, sd_vid_gen_params->strength);
        }
    }

    if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
        LOG_INFO("IMG2VID");

        // WAN LATENT CHAINING: number of leading conditioning (mask=1) latent frames.
        // 0 in the normal (non-chained) path -> the mask below marks only frame 0 (or
        // nothing) exactly as before; the WAN_CONT_LATENT block (after the VAE encode)
        // raises this to anchor(0|1)+K when a prior-segment tail is injected.
        int64_t wan_cont_known = 0;

        if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
            if (sd_ctx->sd->clip_vision == nullptr) {
                // Wan2.1-I2V with no --clip-vision model: feed zeroed clip_fea [1280,257]
                // instead of failing. Degrades identity (the 257 image tokens are an
                // appearance anchor) but lets the i2v path run from c_concat alone.
                LOG_WARN("Wan2.1-I2V with no clip_vision: feeding zero clip_fea [1280,257]");
                latents.clip_vision_output = sd::zeros<float>({1280, 257});
            } else if (!start_image.empty()) {
                auto clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2);
                if (clip_vision_output.empty()) {
                    LOG_ERROR("failed to compute clip vision output for init image");
                    return std::nullopt;
                }
                latents.clip_vision_output = std::move(clip_vision_output);
            } else {
                latents.clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2, true);
            }

            if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
                sd::Tensor<float> end_image_clip_vision_output;
                if (!end_image.empty()) {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2);
                    if (end_image_clip_vision_output.empty()) {
                        LOG_ERROR("failed to compute clip vision output for end image");
                        return std::nullopt;
                    }
                } else {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2, true);
                }
                latents.clip_vision_output = sd::ops::concat(latents.clip_vision_output, end_image_clip_vision_output, 1);
            }

            int64_t t1 = ggml_time_ms();
            LOG_INFO("get_clip_vision_output completed, taking %" PRId64 " ms", t1 - prepare_start_ms);
        }

        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        if (!start_image.empty()) {
            sd::ops::slice_assign(&image, 2, 0, 1, start_image.unsqueeze(2));
        }
        if (!end_image.empty()) {
            sd::ops::slice_assign(&image, 2, request->frames - 1, request->frames, end_image.unsqueeze(2));
        }

        auto concat_latent = sd_ctx->sd->encode_first_stage(image);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (concat_latent.empty()) {
            LOG_ERROR("failed to encode video conditioning frames");
            return std::nullopt;
        }
        latents.concat_latent = std::move(concat_latent);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);

        // ====================================================================
        // WAN LATENT CHAINING (velocity-preserving long-form i2v). DEFAULT OFF:
        // this whole block is skipped unless a prior-segment tail is supplied
        // (sd_vid_gen_params->cont_latent in-memory, OR env WAN_CONT_LATENT=path),
        // so the normal i2v/t2v path stays byte-identical.
        //
        // Wan2.x-I2V conditions purely via the c_concat side channel:
        //   concat_latent = [4 mask channels ++ VAE-encoded conditioning video]
        // (the latent is built just above; the mask is prepended just below). The
        // mask channels flag which latent frames are "known"; the DiT copies those
        // and synthesizes the rest. To chain segments WITHOUT restarting motion we
        // inject the PRIOR segment's last K diffusion-latent frames (its motion-
        // carrying tail) into the HEAD of concat_latent and mark them known. With
        // K>1 the DiT sees the trajectory (position AND velocity), so it continues
        // the motion instead of re-accelerating from a still. The banked tail
        // (WAN_SAVE_LATENT) is already in encode_first_stage's (mu-mean)/std space
        // — same as concat_latent — so it drops in directly, with NO pixel
        // decode/re-encode roundtrip. Mirrors VACE_CONT_LATENT (the inactive-
        // context backdoor) and the LTXAV in-memory/file cont-latent paths.
        // ====================================================================
        {
            const float* mem_tail        = nullptr;
            int64_t       mem_tail_frames = 0;
            if (sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0) {
                mem_tail        = sd_vid_gen_params->cont_latent;
                mem_tail_frames = sd_vid_gen_params->cont_latent_frames;
            }
            const char* file_tail = getenv("WAN_CONT_LATENT");
            bool        have_file = (file_tail != nullptr && file_tail[0] != '\0');

            if (mem_tail != nullptr || have_file) {
                int64_t Wl = latents.concat_latent.shape()[0];
                int64_t Hl = latents.concat_latent.shape()[1];
                int64_t Tl = latents.concat_latent.shape()[2];
                int64_t Cl = latents.concat_latent.dim() > 3 ? latents.concat_latent.shape()[3] : 1;

                // K in LATENT frames from WAN_CONT_K (in PIXEL frames): the Wan VAE is
                // 4x temporal, so K_lat = (K_pix - 1) / 4 + 1 (same arithmetic as
                // VACE_CONT_LATENT). Default 5 pixel frames -> 2 latent overlap frames.
                int kpix = 5;
                if (const char* e = getenv("WAN_CONT_K")) {
                    int v = atoi(e);
                    if (v > 0) kpix = v;
                }
                int64_t K_lat = std::min<int64_t>((std::min<int64_t>(kpix, request->frames) - 1) / 4 + 1, Tl);
                if (K_lat < 1) K_lat = 1;

                // per-channel mean/std over (W,H,T) for each latent channel c. The ggml
                // layout is channel-major-contiguous (idx = ((c*T+t)*H+h)*W+w), so each
                // channel occupies a contiguous T*H*W block at offset c*T*H*W.
                auto per_channel_stats = [](const sd::Tensor<float>& tt,
                                            std::vector<double>& mean,
                                            std::vector<double>& stdv) {
                    int64_t W = tt.shape()[0], H = tt.shape()[1], T = tt.shape()[2];
                    int64_t C   = tt.dim() > 3 ? tt.shape()[3] : 1;
                    int64_t per = W * H * T;
                    const float* d = tt.data();
                    mean.assign((size_t)C, 0.0);
                    stdv.assign((size_t)C, 0.0);
                    for (int64_t c = 0; c < C; ++c) {
                        const float* base = d + c * per;
                        double s = 0.0, sq = 0.0;
                        for (int64_t i = 0; i < per; ++i) {
                            double v = base[i];
                            s += v;
                            sq += v * v;
                        }
                        double m   = s / (double)per;
                        double var = sq / (double)per - m * m;
                        mean[(size_t)c] = m;
                        stdv[(size_t)c] = var > 0.0 ? std::sqrt(var) : 0.0;
                    }
                };

                // Build the [Wl,Hl,K_lat,Cl,1] tail tensor from whichever source.
                sd::Tensor<float> cont_tail;
                bool              ok = false;
                if (have_file) {
                    try {
                        auto    cont_full = sd::load_tensor_from_file_as_tensor<float>(file_tail);
                        int64_t Tprev     = cont_full.shape()[2];
                        int64_t Cprev     = cont_full.dim() > 3 ? cont_full.shape()[3] : 1;
                        int64_t K         = std::min<int64_t>(K_lat, Tprev);
                        if (cont_full.shape()[0] != Wl || cont_full.shape()[1] != Hl || Cprev != Cl) {
                            LOG_ERROR("WAN_CONT_LATENT: shape mismatch, saved latent (%dx%dx%dx%d) "
                                      "incompatible with concat_latent (%dx%dx%dx%d); skipping injection",
                                      (int)cont_full.shape()[0], (int)cont_full.shape()[1], (int)Tprev, (int)Cprev,
                                      (int)Wl, (int)Hl, (int)Tl, (int)Cl);
                        } else {
                            cont_tail = sd::ops::slice(cont_full, 2, Tprev - K, Tprev);  // last K latent frames
                            cont_tail.reshape_({Wl, Hl, K, Cl, 1});
                            K_lat = K;
                            ok    = true;
                        }
                    } catch (const std::exception& e) {
                        LOG_ERROR("WAN_CONT_LATENT: failed to load %s: %s (keeping gray conditioning)", file_tail, e.what());
                    }
                } else {
                    // In-memory (in-process chaining): the caller passes exactly the last
                    // mem_tail_frames latent frames, [Wl,Hl,K,Cl] W-fastest (LTXAV contract).
                    int64_t K = std::min<int64_t>(K_lat, mem_tail_frames);
                    cont_tail = sd::Tensor<float>({Wl, Hl, K, Cl, 1});
                    std::memcpy(cont_tail.data(), mem_tail, (size_t)cont_tail.numel() * sizeof(float));
                    K_lat = K;
                    ok    = true;
                }

                if (ok && K_lat >= 1) {
                    // ---- Phase 3: ABSOLUTE-ANCHOR AGC (anti-drift) BEFORE injection ----
                    // The carried tail is the prior segment's highest-variance frames; re-
                    // injecting it as conditioning ratchets contrast/colour up every cut
                    // (the VACE 0.64->0.80->0.92 runaway). We pull it back to a FIXED
                    // reference (seg-0), NOT to the previous segment — an absolute anchor
                    // is a regulator, a relative match is a random walk (why the pixel
                    // exposure-match drifted). Two modes:
                    //   WAN_CONT_AGC_REF=<seg0 latent>  -> per-channel mean+std match to
                    //       seg-0 (controls BOTH contrast and colour/luma drift; fix opt 2).
                    //   else                            -> global std -> WAN_CONT_AGC_TARGET
                    //       (default 0.65), uniform gain about the global mean
                    //       (VACE_CONT_AGC-equivalent; preserves brightness + channel ratios).
                    if (const char* a = getenv("WAN_CONT_AGC"); a != nullptr && a[0] == '1') {
                        const char* refp = getenv("WAN_CONT_AGC_REF");
                        if (refp != nullptr && refp[0] != '\0') {
                            try {
                                auto                ref = sd::load_tensor_from_file_as_tensor<float>(refp);
                                std::vector<double> rm, rs, cm, cs;
                                per_channel_stats(ref, rm, rs);
                                per_channel_stats(cont_tail, cm, cs);
                                int64_t W = cont_tail.shape()[0], H = cont_tail.shape()[1], T = cont_tail.shape()[2];
                                int64_t C   = cont_tail.dim() > 3 ? cont_tail.shape()[3] : 1;
                                int64_t per = W * H * T;
                                if ((int64_t)rm.size() == C) {
                                    float* d = cont_tail.data();
                                    for (int64_t c = 0; c < C; ++c) {
                                        double g = (cs[(size_t)c] > 1e-6) ? (rs[(size_t)c] / cs[(size_t)c]) : 1.0;
                                        if (g < 0.5) g = 0.5;  // gentle clamp: de-drift, don't regrade
                                        if (g > 2.0) g = 2.0;
                                        float* base = d + c * per;
                                        for (int64_t i = 0; i < per; ++i)
                                            base[i] = (float)((base[i] - cm[(size_t)c]) * g + rm[(size_t)c]);
                                    }
                                    LOG_INFO("WAN_CONT_AGC: per-channel mean+std anchored to ref %s (%lld ch, absolute)",
                                             refp, (long long)C);
                                } else {
                                    LOG_WARN("WAN_CONT_AGC: ref channel count %lld != tail %lld; skipping AGC",
                                             (long long)rm.size(), (long long)C);
                                }
                            } catch (const std::exception& e) {
                                LOG_ERROR("WAN_CONT_AGC: failed to load ref %s: %s (skipping AGC)", refp, e.what());
                            }
                        } else {
                            float target = 0.65f;
                            if (const char* t = getenv("WAN_CONT_AGC_TARGET"); t != nullptr && t[0] != '\0') {
                                float v = (float)atof(t);
                                if (v > 0.0f) target = v;
                            }
                            float*  d = cont_tail.data();
                            int64_t n = cont_tail.numel();
                            if (n > 1) {
                                double sum = 0.0, sq = 0.0;
                                for (int64_t i = 0; i < n; ++i) {
                                    sum += d[i];
                                    sq += (double)d[i] * d[i];
                                }
                                double mean = sum / (double)n;
                                double var  = sq / (double)n - mean * mean;
                                double cur  = var > 0.0 ? std::sqrt(var) : 0.0;
                                if (cur > 1e-6) {
                                    float g = (float)(target / cur);
                                    for (int64_t i = 0; i < n; ++i)
                                        d[i] = (float)((d[i] - mean) * g + mean);
                                    LOG_INFO("WAN_CONT_AGC: global std %.4f -> %.4f (gain %.3f, fixed target)",
                                             cur, target, g);
                                }
                            }
                        }
                    }

                    // ---- Phase 3: APPEARANCE ANCHOR (anti style/identity drift) ----
                    // Every M segments, ALSO pin seg-0's frame-0 latent at the head (mask=1)
                    // so the character can't migrate off the source over a long chain. The
                    // anchor source defaults to WAN_CONT_AGC_REF (the seg-0 latent already
                    // used by AGC) or a dedicated WAN_CONT_ANCHOR path. Segment index is
                    // passed by the chain driver via WAN_CONT_SEG_INDEX. Layout becomes
                    // [anchor(1), tail(K), generated...]; the anchor+overlap are dropped at
                    // stitch time. Mirrors the LTXAV cont-anchor.
                    int64_t base_idx = 0;
                    // ---- INIT-IMG APPEARANCE ANCHOR (VACE-style clean-reference pin) ----
                    // The clean --init-img VAE encoding already sits at concat_latent idx 0
                    // (the [init, gray, gray...] encode at ~:6104). The plain chain (base_idx
                    // 0) and the seg1-frame-0 anchor below both OVERWRITE it — the former with
                    // the motion tail, the latter with seg-1's already-diffused (drifted)
                    // frame-0 latent. Either way the DiT re-derives appearance from a non-
                    // pristine source -> identity slides to a similar-but-different face
                    // (owner: "like i2v on the blurry final frame, not the original
                    // character"). VACE held identity by pinning the ORIGINAL reference image
                    // through its ref slot; mirror that here: PRESERVE the clean init-img
                    // latent at idx 0 (mask=1, untouched) and inject the motion tail at idx 1+.
                    // So every continuation segment keeps the true source character as a hard
                    // known anchor, while the tail still carries velocity. Off by default.
                    bool initimg_anchor = false;
                    if (const char* e = getenv("WAN_CONT_ANCHOR_INITIMG");
                        e != nullptr && e[0] == '1' && !start_image.empty() && (K_lat + 1) <= Tl) {
                        base_idx       = 1;  // leave concat_latent[idx 0] = clean init-img encoding
                        initimg_anchor = true;
                        LOG_INFO("WAN_CONT_ANCHOR_INITIMG: pinned clean --init-img latent at idx 0 "
                                 "(VACE-style ref); motion tail injects at idx 1");
                    }
                    int     anchor_every = 0;
                    if (const char* e = getenv("WAN_CONT_ANCHOR_EVERY")) anchor_every = atoi(e);
                    int seg_index = 0;
                    if (const char* e = getenv("WAN_CONT_SEG_INDEX")) seg_index = atoi(e);
                    const char* anchor_path = getenv("WAN_CONT_ANCHOR");
                    if (anchor_path == nullptr || anchor_path[0] == '\0') anchor_path = getenv("WAN_CONT_AGC_REF");
                    if (!initimg_anchor && anchor_every > 0 && seg_index > 0 && (seg_index % anchor_every) == 0 &&
                        anchor_path != nullptr && anchor_path[0] != '\0' && (K_lat + 1) <= Tl) {
                        try {
                            auto anchor_full = sd::load_tensor_from_file_as_tensor<float>(anchor_path);
                            int64_t Cancc    = anchor_full.dim() > 3 ? anchor_full.shape()[3] : 1;
                            if (anchor_full.shape()[0] == Wl && anchor_full.shape()[1] == Hl && Cancc == Cl) {
                                auto anchor = sd::ops::slice(anchor_full, 2, 0, 1);  // original char = frame 0
                                anchor.reshape_({Wl, Hl, 1, Cl, 1});
                                sd::ops::slice_assign(&latents.concat_latent, 2, 0, 1, anchor);
                                base_idx = 1;
                                LOG_INFO("WAN_CONT_ANCHOR: pinned seg-0 frame-0 latent at idx 0 (seg %d, every %d)",
                                         seg_index, anchor_every);
                            } else {
                                LOG_WARN("WAN_CONT_ANCHOR: shape mismatch %s; skipping anchor", anchor_path);
                            }
                        } catch (const std::exception& e) {
                            LOG_ERROR("WAN_CONT_ANCHOR: failed to load %s: %s", anchor_path, e.what());
                        }
                    }

                    // clamp the overlap so anchor + tail fit the segment
                    if (base_idx + K_lat > Tl) {
                        K_lat = Tl - base_idx;
                        cont_tail = sd::ops::slice(cont_tail, 2, 0, K_lat);
                        cont_tail.reshape_({Wl, Hl, K_lat, Cl, 1});
                    }

                    // ---- inject the (AGC'd) tail into the C-channel latent head ----
                    // concat_latent is still the bare C-channel latent here (the 4 mask
                    // channels are prepended below), so this writes the full Cl channels of
                    // frames [base_idx .. base_idx+K_lat).
                    sd::ops::slice_assign(&latents.concat_latent, 2, base_idx, base_idx + K_lat, cont_tail);
                    wan_cont_known = base_idx + K_lat;
                    LOG_INFO("WAN_CONT_LATENT: injected %lld tail latent frames at idx %lld "
                             "(known=%lld of %lld, K_pix=%d, %s) bypassing pixel re-encode",
                             (long long)K_lat, (long long)base_idx, (long long)wan_cont_known,
                             (long long)Tl, kpix, have_file ? "file" : "in-memory");

                    // ---- i2v FREEZE-BASED CONTINUATION (sidesteps the broken VACE
                    // distill; the clean uniformly-distilled i2v expert does the chain) ----
                    // The c_concat tail-inject above only TELLS the DiT "these head frames
                    // are known," but Wan2.2-I2V was trained for SINGLE-image c_concat
                    // conditioning, so the tail alone often fails to truly continue motion
                    // (the old WAN_CONT_LATENT path that "failed"). Additionally PIN the
                    // injected tail in the diffusion latent itself: write it into
                    // init_latent and hold those frames fixed via denoise_mask=0 + a clean
                    // (t=0) per-frame timestep — exactly the TI2V/avatar input-freeze
                    // (sample loop ~:2543 + process_timesteps ~:2130, now ungated for
                    // VERSION_WAN2_2_I2V). The remaining frames (mask=1) generate and
                    // continue from the frozen tail.
                    //   - byte-identical for every non-continuation path: this whole block
                    //     only runs when a cont tail was injected (mem_tail/WAN_CONT_LATENT).
                    //   - off-switch: WAN_I2V_CONT_FREEZE_DISABLE=1 reverts to the old
                    //     c_concat-tail-inject-only behaviour (mask/init_latent untouched).
                    if (sd_ctx->sd->version == VERSION_WAN2_2_I2V &&
                        getenv("WAN_I2V_CONT_FREEZE_DISABLE") == nullptr) {
                        // zeros base (generated frames) + the AGC'd tail at [base_idx, base_idx+K_lat)
                        latents.init_latent = sd_ctx->sd->generate_init_latent(
                            request->width, request->height, request->frames, true);  // sd::zeros, no RNG
                        if (latents.init_latent.shape()[2] == Tl && latents.init_latent.shape()[3] == Cl) {
                            sd::ops::slice_assign(&latents.init_latent, 2, base_idx, base_idx + K_lat, cont_tail);
                            latents.denoise_mask = sd::full<float>({Wl, Hl, Tl, 1, 1}, 1.f);
                            sd::ops::fill_slice(&latents.denoise_mask, 2, base_idx, base_idx + K_lat, 0.0f);
                            LOG_INFO("WAN_I2V_CONT_FREEZE: pinned %lld tail latent frames [%lld,%lld) "
                                     "at clean timestep (mask=0); %lld frames generate",
                                     (long long)K_lat, (long long)base_idx, (long long)(base_idx + K_lat),
                                     (long long)(Tl - (base_idx + K_lat)));
                        } else {
                            LOG_WARN("WAN_I2V_CONT_FREEZE: init_latent (T=%lld C=%lld) != concat (T=%lld C=%lld); "
                                     "skipping freeze (c_concat-only continuation)",
                                     (long long)latents.init_latent.shape()[2], (long long)latents.init_latent.shape()[3],
                                     (long long)Tl, (long long)Cl);
                            latents.init_latent = sd::Tensor<float>();  // clear -> default rebuild at ~:6734
                        }
                    }
                }
            }
        }

        sd::Tensor<float> concat_mask = sd::zeros<float>({latents.concat_latent.shape()[0],
                                                          latents.concat_latent.shape()[1],
                                                          latents.concat_latent.shape()[2],
                                                          4,
                                                          1});  // [b, 4, t, h/vae_scale_factor, w/vae_scale_factor]
        if (!start_image.empty()) {
            sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
        }
        // WAN LATENT CHAINING: mark the injected anchor+overlap frames [0..wan_cont_known)
        // as known (mask=1) so the DiT treats them as given context and continues from
        // them. 0 in the non-chained path -> behaviour unchanged.
        if (wan_cont_known > 0) {
            sd::ops::fill_slice(&concat_mask, 2, 0, wan_cont_known, 1.0f);
        }
        if (!end_image.empty()) {
            auto last_channel = sd::ops::slice(concat_mask, 3, 3, 4);
            sd::ops::fill_slice(&last_channel, 2, last_channel.shape()[2] - 1, last_channel.shape()[2], 1.0f);
            sd::ops::slice_assign(&concat_mask, 3, 3, 4, last_channel);
        }
        latents.concat_latent = sd::ops::concat(concat_mask, latents.concat_latent, 3);  // [b, 4+c, t, h/vae_scale_factor, w/vae_scale_factor]
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-TI2V-5B" && !start_image.empty()) {
        LOG_INFO("IMG2VID");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode init video frame");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        if (!end_image.empty()) {
            auto end_img          = end_image.reshape({end_image.shape()[0], end_image.shape()[1], 1, end_image.shape()[2], 1});
            auto end_image_latent = sd_ctx->sd->encode_first_stage(end_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
            if (end_image_latent.empty()) {
                LOG_ERROR("failed to encode end video frame");
                return std::nullopt;
            }
            sd::ops::slice_assign(&latents.init_latent, 2, latents.init_latent.shape()[2] - 1, latents.init_latent.shape()[2], end_image_latent);
            sd::ops::fill_slice(&latents.denoise_mask, 2, latents.init_latent.shape()[2] - 1, latents.init_latent.shape()[2], 0.0f);
        }

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_version_is_longcat_avatar(sd_ctx->sd->version) &&
               sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0) {
        // CONTINUATION (video continuation, generate_vc): instead of a single ref
        // image, condition on the LAST N LATENT frames of the PRIOR segment. The
        // caller passes the prior segment's diffusion-latent tail directly (no VAE
        // decode/re-encode roundtrip), so num_cond_latents = cont_latent_frames.
        // The cond latents are written to the head of init_latent and held fixed
        // (denoise_mask 0, timestep 0); the remaining frames are noise and continue
        // from them. The cond/noise self-attn two-pass split + the per-frame denoise
        // mask already generalize from N==1 to N>1 (the avatar derives num_cond_latents
        // from the leading zero-timestep frames at DiT-forward time).
        int64_t num_cond_tail = sd_vid_gen_params->cont_latent_frames;
        // REFERENCE ANCHOR (generate_avc): when the caller supplies the original
        // portrait's latent, PREPEND it as a persistent un-drifted ref frame so the
        // layout is [ref(1), cond_tail(N), noise...] — the reference keeps this clean
        // anchor on EVERY segment. num_ref = 1 then, and the cond split / mask / PE
        // engage the 3-way path. Absent (older raw-latent flow) -> ref-free 2-way path.
        const bool   have_ref  = sd_vid_gen_params->cont_ref_latent != nullptr;
        const int64_t num_ref  = have_ref ? 1 : 0;
        const int64_t num_cond_latents = num_ref + num_cond_tail;  // total fixed-cond frames
        LOG_INFO("CONTINUATION (ref-anchor=%d + %lld prior-segment tail latent frames, ref_img_index=%d, mask_frame_range=%d)",
                 (int)num_ref, (long long)num_cond_tail,
                 sd_vid_gen_params->cont_ref_img_index, sd_vid_gen_params->cont_mask_frame_range);

        int64_t t1 = ggml_time_ms();
        // Base noise latent at the requested length (T = request->frames latents, e.g.
        // 24 for 93 frames). The reference `prepare_latents` builds exactly this, then
        // the ref is PREPENDED as an EXTRA latent frame (cat([ref_latent, latents]),
        // pipeline L1378 → T grows to 25), denoised, and STRIPPED before decode
        // (L1494-1496) — so NO generated frame is lost. We mirror that: keep
        // request->frames unchanged (it drives the audio windowing) and grow the latent
        // tensor by num_ref, marking ref_image_num so the trailing decode strips it.
        sd::Tensor<float> base_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
        int64_t Wl = base_latent.shape()[0];
        int64_t Hl = base_latent.shape()[1];
        int64_t Tb = base_latent.shape()[2];  // base (non-ref) latent frames
        int64_t Cl = base_latent.shape()[3];
        if (num_cond_tail > Tb) {
            LOG_ERROR("continuation: cont_latent_frames(%lld) exceeds segment latent frames %lld",
                      (long long)num_cond_tail, (long long)Tb);
            return std::nullopt;
        }
        // Overwrite the cond_tail (prior segment's re-encoded tail) into the head of the
        // base noise latent: latents[:, :, :num_cond_tail] = cond_tail (pipeline L1365).
        {
            sd::Tensor<float> cont_latent({Wl, Hl, num_cond_tail, Cl, 1});
            std::memcpy(cont_latent.data(), sd_vid_gen_params->cont_latent,
                        (size_t)cont_latent.numel() * sizeof(float));
            sd::ops::slice_assign(&base_latent, 2, 0, num_cond_tail, cont_latent);
        }
        // Prepend the ref anchor as an EXTRA temporal frame → [ref(num_ref), base(Tb)].
        if (have_ref) {
            int64_t T_full      = Tb + num_ref;  // = 25 with num_ref=1
            latents.init_latent = sd::full<float>({Wl, Hl, T_full, Cl, 1}, 0.f);
            sd::Tensor<float> ref_latent({Wl, Hl, num_ref, Cl, 1});
            std::memcpy(ref_latent.data(), sd_vid_gen_params->cont_ref_latent,
                        (size_t)ref_latent.numel() * sizeof(float));
            sd::ops::slice_assign(&latents.init_latent, 2, 0, num_ref, ref_latent);
            sd::ops::slice_assign(&latents.init_latent, 2, num_ref, num_ref + Tb, base_latent);
            latents.ref_image_num = num_ref;  // strip the ref frame(s) before VAE decode
        } else {
            latents.init_latent = std::move(base_latent);
        }

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, num_cond_latents, 0.0f);

        // Set the 3-way-split / ref-PE params on the avatar model for this render.
        if (have_ref) {
            auto avatar_model = std::dynamic_pointer_cast<LongCatAvatarModel>(sd_ctx->sd->diffusion_model);
            if (avatar_model) {
                avatar_model->cont_num_ref_latents  = (int)num_ref;
                avatar_model->cont_ref_img_index    = sd_vid_gen_params->cont_ref_img_index;
                avatar_model->cont_mask_frame_range = sd_vid_gen_params->cont_mask_frame_range;
            }
        }

        int64_t t2 = ggml_time_ms();
        LOG_INFO("continuation latent conditioning prepared, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_version_is_longcat_avatar(sd_ctx->sd->version) && !start_image.empty()) {
        // ai2v: the reference portrait is VAE-encoded to ONE temporal cond latent
        // (num_cond_latents = 1), prepended as the first latent frame; generated
        // frames follow. The cond frame is held fixed via the denoise_mask (=0) and
        // its per-frame timestep is forced to 0. (pipeline_longcat_video_avatar.py
        // generate_ai2v: prepare_latents(image, num_cond_frames=1) ->
        // latents[:,:,:1]=cond; loop steps only latents[:,:,1:].)
        LOG_INFO("AI2V (reference-image conditioning)");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);  // [b, c, 1, h/8, w/8]
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode reference image");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);  // [b, c, t, h/8, w/8]
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
               sd_ctx->sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B") {
        LOG_INFO("VACE");
        // Continuation mode: keep the first K control frames (mask=0, inactive context) and
        // generate the rest, so the segment continues from the prior segment's carried-over
        // tail. Default 0 = control mode (all frames reactive/generated). Env-toggled to match
        // the codebase's experimental-flag pattern (no public-struct churn).
        int vace_cont_frames = 0;
        if (const char* e = getenv("VACE_CONT_FRAMES")) {
            vace_cont_frames = std::max(0, atoi(e));
        }
        int64_t t1 = ggml_time_ms();
        sd::Tensor<float> ref_image_latent;
        if (!start_image.empty()) {
            auto ref_img     = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
            auto encoded_ref = sd_ctx->sd->encode_first_stage(ref_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
            if (encoded_ref.empty()) {
                LOG_ERROR("failed to encode VACE reference image");
                return std::nullopt;
            }
            ref_image_latent = sd::ops::concat(encoded_ref, sd::zeros<float>(encoded_ref.shape()), 3);  // [b, 2*c, 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        sd::Tensor<float> control_video = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        int64_t control_frame_count     = std::min<int64_t>(request->frames, sd_vid_gen_params->control_frames_size);
        for (int64_t i = 0; i < control_frame_count; ++i) {
            auto control_frame = sd_image_to_tensor(sd_vid_gen_params->control_frames[i], request->width, request->height);
            sd::ops::slice_assign(&control_video, 2, i, i + 1, control_frame.unsqueeze(2));
        }

        sd::Tensor<float> mask = sd::full<float>({request->width, request->height, request->frames, 1, 1}, 1.0f);
        if (vace_cont_frames > 0) {
            int kpx = std::min<int>(vace_cont_frames, static_cast<int>(request->frames));
            sd::ops::fill_slice(&mask, 2, 0, kpx, 0.0f);  // keep first K pixel frames (continuation context)
            LOG_INFO("VACE continuation: %d kept pixel frames (mask=0)", kpx);
        }

        control_video              = control_video - 0.5f;
        sd::Tensor<float> inactive = control_video * (1.0f - mask) + 0.5f;
        sd::Tensor<float> reactive = control_video * mask + 0.5f;

        // VACE GRAY-LATENT FAST PATH (lever #1 — the ~30s control-context encode).
        // For a fresh segment with no active control frames, BOTH inactive and reactive
        // collapse to full(0.5) gray; for a continuation, reactive is still all-gray and
        // inactive's gray tail dominates. VAE-encoding a constant is deterministic and
        // input-independent, so we encode the gray volume ONCE per (W,H,T) and reuse it
        // (in-run dedup of inactive==reactive). An optional cross-run disk cache
        // (VACE_GRAY_CACHE_DIR) makes the gray encode free on every segment of a chain.
        // Substituting cache[x]==encode(x) for an all-gray x is bit-exact by construction.
        // VACE_NO_GRAY_FAST=1 disables (for A/B).
        const bool vace_gray_fast = !(getenv("VACE_NO_GRAY_FAST") && getenv("VACE_NO_GRAY_FAST")[0] == '1');
        // Gray-SUFFIX fast encode (default ON): the `inactive` context is [real kept tail
        // (kpx px) + gray (rest)] — is_const_gray is false so it falls to the full
        // temporal-chunked encode (~179s @65f). But that encode groups the frames as
        // 1 + 4k and encodes each group INDEPENDENTLY, so every group past the tail is the
        // SAME constant-gray latent, and the leading (tail) latent frames get OVERWRITTEN by
        // VACE_CONT_LATENT the instant after. So we only need: encode the real prefix
        // (rounded up to a group boundary) + ONE gray group, then tile the gray latent for
        // the suffix. Bit-exact by construction. Off-switch VACE_NO_GRAY_SUFFIX=1.
        const bool vace_gray_suffix = !(getenv("VACE_NO_GRAY_SUFFIX") && getenv("VACE_NO_GRAY_SUFFIX")[0] == '1');
        const char* gray_cache_dir = getenv("VACE_GRAY_CACHE_DIR");
        static std::map<std::array<int64_t, 3>, sd::Tensor<float>> vace_gray_cache;
        auto is_const_gray = [](const sd::Tensor<float>& x) {
            if (x.numel() <= 0) return false;
            const float* d = x.data();
            for (int64_t i = 0, n = x.numel(); i < n; ++i) {
                if (d[i] != 0.5f) return false;
            }
            return true;
        };
        auto vace_encode_ctx = [&](const sd::Tensor<float>& x, const char* tag) -> sd::Tensor<float> {
            if (!vace_gray_fast || !is_const_gray(x)) {
                // Gray-SUFFIX fast path: the continuation `inactive` = [real kept tail px +
                // gray px]. encode_first_stage_temporal_chunked groups frames 1 + 4k and
                // encodes each group INDEPENDENTLY (no cross-group causal state), so every
                // group past the real tail is the SAME constant-gray latent. Encode only the
                // real prefix (rounded up to a group boundary — the extra gray frames inside
                // it are byte-identical to the full encode's groups) + ONE gray group, and
                // tile the gray latent for the rest. Bit-exact; ~6x fewer VAE graphs.
                if (vace_gray_fast && vace_gray_suffix && sd_version_is_wan(sd_ctx->sd->version) &&
                    x.dim() >= 3 && x.shape()[2] > 5) {
                    const int64_t T = x.shape()[2];
                    int64_t last_real = -1;
                    for (int64_t t = 0; t < T; ++t) {
                        if (!is_const_gray(sd::ops::slice(x, 2, t, t + 1))) last_real = t;
                    }
                    if (last_real >= 0 && last_real < T - 1) {
                        const int64_t k  = (last_real + 3) / 4;  // ceil(last_real/4): #groups covering real px past frame 0
                        const int64_t sb = 1 + 4 * k;            // first temporal-group boundary at/after the real px
                        if (sb < T && (T - sb) % 4 == 0) {
                            auto prefix     = sd_ctx->sd->encode_first_stage_temporal_chunked(sd::ops::slice(x, 2, 0, sb));
                            auto gray_group = sd::ops::slice(x, 2, sb, sb + 4);           // 4 gray px -> 1 gray latent frame
                            auto gray_lat   = sd_ctx->sd->encode_first_stage(gray_group);
                            if (!prefix.empty() && !gray_lat.empty()) {
                                const int64_t n_suf = (T - sb) / 4;
                                auto suffix = gray_lat;
                                for (int64_t i = 1; i < n_suf; ++i) suffix = sd::ops::concat(suffix, gray_lat, 2);
                                auto out = sd::ops::concat(prefix, suffix, 2);
                                LOG_INFO("VACE gray-suffix fast encode (%s): real prefix [0,%lld) + %lld tiled gray "
                                         "latent frames (was %lld temporal groups)",
                                         tag, (long long)sb, (long long)n_suf, (long long)(1 + (T - 1 + 3) / 4));
                                return out;
                            }
                        }
                    }
                }
                // Temporal-chunked fallback: the inactive context (real kept tail + gray) is a
                // genuine 25-81 frame video encode that OOMs full-temporal at useful res.
                return sd_ctx->sd->encode_first_stage_temporal_chunked(x);
            }
            std::array<int64_t, 3> key{request->width, request->height, x.shape()[2]};
            if (auto it = vace_gray_cache.find(key); it != vace_gray_cache.end()) {
                LOG_INFO("VACE gray-latent cache hit (%s, skipped VAE encode)", tag);
                return it->second;
            }
            std::string gray_path;
            if (gray_cache_dir && gray_cache_dir[0] != '\0') {
                gray_path = std::string(gray_cache_dir) + "/vace_gray_" + std::to_string(key[0]) + "x" +
                            std::to_string(key[1]) + "x" + std::to_string(key[2]) + ".bin";
                try {
                    auto cached = sd::load_tensor_from_file_as_tensor<float>(gray_path);
                    if (!cached.empty()) {
                        LOG_INFO("VACE gray-latent disk-cache hit (%s, %s, skipped VAE encode)", tag, gray_path.c_str());
                        vace_gray_cache[key] = cached;
                        return cached;
                    }
                } catch (const std::exception&) { /* miss → compute below */ }
            }
            auto enc = sd_ctx->sd->encode_first_stage_temporal_chunked(x);
            if (!enc.empty()) {
                vace_gray_cache[key] = enc;
                if (!gray_path.empty()) {
                    try {
                        sd::save_tensor_to_file(gray_path, enc, "vace_gray");
                        LOG_INFO("VACE gray-latent computed + disk-cached (%s, %s)", tag, gray_path.c_str());
                    } catch (const std::exception& e) {
                        LOG_WARN("VACE gray-latent disk-cache write failed (%s): %s", gray_path.c_str(), e.what());
                    }
                }
            }
            return enc;
        };

        inactive = vace_encode_ctx(inactive, "inactive");  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (inactive.empty()) {
            LOG_ERROR("failed to encode VACE inactive context");
            return std::nullopt;
        }

        // VACE LATENT BACKDOOR: replace the re-encoded pixels in the kept (mask=0) head
        // frames of `inactive` with the prior segment's TAIL diffusion latents, bypassing
        // the lossy VAE decode->re-encode roundtrip. The saved latent (VACE_SAVE_LATENT)
        // is already in encode_first_stage's (mu-mean)/std diffusion space, so it drops in
        // directly. Composes with VACE_CONT_FRAMES (which sets mask=0 on those slots);
        // unset => the re-encoded pixel path above is untouched (byte-identical default).
        if (vace_cont_frames > 0) {
            // Prior-window latent source. The warm Wan-VACE server hands the FULL prior-window
            // diffusion latent IN MEMORY (sd_vid_gen_params->cont_latent, cont_latent_frames =
            // its temporal length), which takes precedence over the disk file the shell chain
            // uses (VACE_CONT_LATENT=path). Either way we slice the last K latent frames and
            // inject them into the inactive head, bypassing the lossy pixel decode->re-encode.
            // The in-memory path is what kills the cont_bank/*.bin round-trip.
            const bool  vace_mem_cont =
                sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0;
            const char* lp = getenv("VACE_CONT_LATENT");
            if (vace_mem_cont || (lp != nullptr && lp[0] != '\0')) {
                int64_t klat = std::min<int64_t>(
                    (std::min<int64_t>(vace_cont_frames, request->frames) - 1) / 4 + 1,
                    inactive.shape()[2]);
                try {
                    sd::Tensor<float> cont_full;
                    if (vace_mem_cont) {
                        // Rebuild the prior latent [Wl,Hl,Tprev,Cl,1] from the in-memory tail.
                        // Geometry comes from `inactive` (same W/H/C = same res + VAE), so the
                        // downstream shape check + slice_assign match exactly as the file path.
                        int64_t Tmem = sd_vid_gen_params->cont_latent_frames;
                        cont_full    = sd::Tensor<float>(
                            {inactive.shape()[0], inactive.shape()[1], Tmem, inactive.shape()[3], 1});
                        std::memcpy(cont_full.data(), sd_vid_gen_params->cont_latent,
                                    (size_t)cont_full.numel() * sizeof(float));
                    } else {
                        cont_full = sd::load_tensor_from_file_as_tensor<float>(lp);
                    }
                    int64_t Tprev  = cont_full.shape()[2];
                    // Discard-last-frames at the CONDITIONING level (companion to the stitch-level
                    // discard in the chain harness): the prior segment's TERMINAL latent frames are
                    // its most striping-degraded (farthest from that segment's own motion prior).
                    // Ignoring the last N latent frames re-anchors the next window on the still-clean
                    // frames BEFORE the degradation, so blur doesn't propagate across cuts (the Wan2GP
                    // "discard last frames" fix). Default unset => byte-identical to prod.
                    if (const char* dt = getenv("VACE_CONT_LATENT_DROP_TAIL")) {
                        int64_t drop = std::max<int64_t>(0, (int64_t)std::atoi(dt));
                        if (drop > 0 && Tprev - drop >= klat) {
                            Tprev -= drop;
                            LOG_INFO("VACE_CONT_LATENT_DROP_TAIL: ignoring %lld degraded trailing latent frame(s) "
                                     "(prior tail now ends at latent frame %lld)",
                                     (long long)drop, (long long)Tprev);
                        }
                    }
                    int64_t K      = std::min<int64_t>(klat, Tprev);
                    // last K latent frames of the prior segment (its tail = seg2's overlap head)
                    auto cont_tail = sd::ops::slice(cont_full, 2, Tprev - K, Tprev);
                    // rank/shape-match `inactive` ([W,H,T,C,1]) for slice_assign
                    std::vector<int64_t> tgt = inactive.shape();
                    tgt[2]                   = K;
                    if (cont_tail.numel() != [&] { int64_t n = 1; for (auto d : tgt) n *= d; return n; }()) {
                        LOG_ERROR("VACE_CONT_LATENT: shape mismatch, saved latent (%dx%dx%dx%d) "
                                  "incompatible with inactive (%dx%dx%dx%d); skipping injection",
                                  (int)cont_full.shape()[0], (int)cont_full.shape()[1],
                                  (int)cont_full.shape()[2],
                                  (int)(cont_full.dim() > 3 ? cont_full.shape()[3] : 1),
                                  (int)inactive.shape()[0], (int)inactive.shape()[1],
                                  (int)inactive.shape()[2], (int)inactive.shape()[3]);
                    } else {
                        // VACE continuation latent-AGC (FINDINGS-L12 fix): the carried tail is the
                        // prior segment's HIGHEST-variance frames; re-injecting it as the next
                        // segment's VACE conditioning ratchets the latent std up every cut
                        // (measured 0.64->0.80->0.92...), blowing out contrast over a long chain.
                        // Pull the tail's global std back to a canonical reference (the measured
                        // fresh-segment scale) so every segment re-seeds from the SAME baseline —
                        // contrast-only AGC: a single uniform gain about the mean, so it preserves
                        // brightness + per-channel ratios (no colour distortion), just de-drifts
                        // the contrast. Opt-in (output is NOT bit-exact): VACE_CONT_AGC=1 enables;
                        // VACE_CONT_AGC_TARGET overrides the target std (default 0.65 = seg0 global
                        // latent std at the 3+3 distill — res/model-robust as a global scalar).
                        if (const char* a = getenv("VACE_CONT_AGC"); a != nullptr && a[0] == '1') {
                            float target = 0.65f;
                            if (const char* t = getenv("VACE_CONT_AGC_TARGET"); t != nullptr && t[0] != '\0') {
                                float v = static_cast<float>(atof(t));
                                if (v > 0.0f) target = v;
                            }
                            float* d        = cont_tail.data();
                            const int64_t n = cont_tail.numel();
                            if (n > 1) {
                                double sum = 0.0, sq = 0.0;
                                for (int64_t i = 0; i < n; ++i) {
                                    sum += d[i];
                                    sq += static_cast<double>(d[i]) * d[i];
                                }
                                double mean = sum / static_cast<double>(n);
                                double var  = sq / static_cast<double>(n) - mean * mean;
                                double cur_std = var > 0.0 ? std::sqrt(var) : 0.0;
                                if (cur_std > 1e-6) {
                                    float g = static_cast<float>(target / cur_std);
                                    for (int64_t i = 0; i < n; ++i) {
                                        d[i] = static_cast<float>((d[i] - mean) * g + mean);
                                    }
                                    LOG_INFO("VACE_CONT_AGC: carried-tail std %.4f -> %.4f (gain %.3f, mean %.4f kept)",
                                             cur_std, target, g, mean);
                                }
                            }
                        }
                        cont_tail.reshape_(tgt);
                        sd::ops::slice_assign(&inactive, 2, 0, K, cont_tail);
                        LOG_INFO("VACE_CONT_LATENT: injected %lld tail latent frames (of %lld) into "
                                 "inactive head, bypassing pixel re-encode (%s)",
                                 (long long)K, (long long)Tprev, vace_mem_cont ? "in-memory" : "disk");
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("VACE_CONT_LATENT: failed to load %s: %s (keeping re-encoded pixels)",
                              vace_mem_cont ? "<in-memory>" : lp, e.what());
                }
            }
        }

        reactive = vace_encode_ctx(reactive, "reactive");  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (reactive.empty()) {
            LOG_ERROR("failed to encode VACE reactive context");
            return std::nullopt;
        }

        int64_t length = inactive.shape()[2];
        if (!ref_image_latent.empty()) {
            length += 1;
            request->frames       = static_cast<int>((length - 1) * 4 + 1);
            latents.ref_image_num = 1;
        }
        auto vace_context = sd::ops::concat(inactive, reactive, 3);  // [b, 2*c, t, h/vae_scale_factor, w/vae_scale_factor]

        mask              = sd::full<float>({request->width, request->height, inactive.shape()[2], 1, 1}, 1.0f);
        if (vace_cont_frames > 0) {
            int klat = std::min<int>((std::min<int>(vace_cont_frames, static_cast<int>(request->frames)) - 1) / 4 + 1,
                                     static_cast<int>(inactive.shape()[2]));
            sd::ops::fill_slice(&mask, 2, 0, klat, 0.0f);  // keep first K-equiv latent frames (continuation context)
        }
        auto mask_context = mask.reshape({request->vae_scale_factor,
                                          inactive.shape()[0],
                                          request->vae_scale_factor,
                                          inactive.shape()[1],
                                          inactive.shape()[2]});   // [t, h/vae_scale_factor, vae_scale_factor, w/vae_scale_factor, vae_scale_factor]
        mask_context      = mask_context.permute({1, 3, 4, 0, 2})  // [vae_scale_factor, vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]
                           .reshape({inactive.shape()[0],
                                     inactive.shape()[1],
                                     inactive.shape()[2],
                                     request->vae_scale_factor * request->vae_scale_factor});  // [vae_scale_factor*vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]

        if (!ref_image_latent.empty()) {
            vace_context  = sd::ops::concat(ref_image_latent, vace_context, 2);  // [b, 2*c, t+1, h/vae_scale_factor, w/vae_scale_factor]
            auto mask_pad = sd::zeros<float>({mask_context.shape()[0],
                                              mask_context.shape()[1],
                                              1,
                                              mask_context.shape()[3]});  // [vae_scale_factor*vae_scale_factor, 1, h/vae_scale_factor, w/vae_scale_factor]
            mask_context  = sd::ops::concat(mask_pad, mask_context, 2);   // [vae_scale_factor*vae_scale_factor, t + 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        mask_context.unsqueeze_(mask_context.dim());  // [b, vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]

        latents.vace_context = sd::ops::concat(vace_context, mask_context, 3);  // [b, 2*c + vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]
        int64_t t2           = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    }

    if (latents.init_latent.empty()) {
        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version) && !latents.audio_latent.empty()) {
        // Driving audio needs a denoise mask so the audio slot can be pinned (mask=0)
        // every step. If none exists (pure t2v), make a fully-generated video mask (1.0)
        // so only the audio is held fixed.
        if ((latents.audio_fixed || latents.audio_reference_conditioning) && latents.denoise_mask.empty()) {
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
        }
        if (!latents.denoise_mask.empty()) {
            sd::Tensor<float> audio_ref_mask;
            if (latents.audio_reference_conditioning) {
                audio_ref_mask = make_ltxav_lipdub_audio_mask(latents.audio_latent,
                                                               latents.audio_target_length,
                                                               false);
            }
            latents.denoise_mask = pack_ltxav_audio_and_video_denoise_mask(latents.denoise_mask,
                                                                           latents.init_latent,
                                                                           latents.audio_latent,
                                                                           latents.audio_fixed ? 0.0f : 1.0f,
                                                                           audio_ref_mask.empty() ? nullptr : &audio_ref_mask);
        }
        latents.init_latent = pack_ltxav_audio_and_video_latents(latents.init_latent, latents.audio_latent);
    }

    // Request-driven: only inspect the character-reference field when the request actually
    // provides one (and it is not killed via LTXAV_CHARACTER_REF=0). This placement is after all
    // legacy conditioning, so the no-reference token layout is byte-identical to before.
    if (sd_version_is_ltxav(sd_ctx->sd->version) && ltxav_character_ref_enabled() &&
        sd_vid_gen_params->character_reference_latent != nullptr) {
        // Base pass runs at base resolution -> attach the base-res identity block (the higher-res
        // _lo/_hi latents are reserved for the 2x/4x refine stages where the extra tokens pay off).
        sd::Tensor<float> character = ltxav_character_latent_for_stage(sd_vid_gen_params, LtxavCharTier::Base);
        if (!append_ltxav_character_reference(&latents, character, request->fps, request->vae_scale_factor, 8)) {
            LOG_ERROR("failed to attach LTXAV character reference");
            return std::nullopt;
        }
    }

    return latents;
}

static ImageGenerationEmbeds prepare_video_generation_embeds(sd_ctx_t* sd_ctx,
                                                             const sd_vid_gen_params_t* sd_vid_gen_params,
                                                             const GenerationRequest& request,
                                                             const ImageGenerationLatents& latents) {
    ImageGenerationEmbeds embeds;
    ConditionerParams condition_params;
    condition_params.clip_skip       = request.clip_skip;
    condition_params.text            = request.prompt;
    condition_params.zero_out_masked = true;

    int64_t prepare_start_ms = ggml_time_ms();
    // Chained avatar segments reuse the cached text conditioning (the umT5 weights
    // were freed after segment 0). The avatar leaves c_concat/c_vector empty, so the
    // cached cond/uncond carry the full text conditioning.
    // Prompt-keyed avatar text cache. Valid only when the prompt AND negative prompt
    // match what it was computed for. On a warm resident worker the umT5 weights were
    // freed (free_params_immediately), so a prompt change forces a reload + recompute;
    // an identical prompt (the common boilerplate case) reuses the cache with no reload
    // and no recompute.
    // Both the LongCat avatar AND LTXAV keep the DiT resident across chain segments and
    // re-run the (expensive) text encode + projection each segment — so both benefit from
    // the prompt-keyed cache + the up-front batch precompute. (VERSION_LTXAV is distinct
    // from VERSION_LONGCAT_AVATAR, so the avatar-only predicate isn't enough here.)
    bool        avatar_resident = (sd_version_is_longcat_avatar(sd_ctx->sd->version) ||
                            sd_ctx->sd->version == VERSION_LTXAV) &&
                           sd_ctx->sd->keep_diffusion_model_resident;
    std::string cache_key       = StableDiffusionGGML::text_cond_key(request.prompt, request.negative_prompt);
    auto        cache_it        = avatar_resident ? sd_ctx->sd->avatar_cond_cache.find(cache_key)
                                                  : sd_ctx->sd->avatar_cond_cache.end();
    bool        use_text_cache  = (cache_it != sd_ctx->sd->avatar_cond_cache.end());
    if (use_text_cache) {
        LOG_INFO("avatar: reusing cached text conditioning (prompt unchanged)");
        embeds.cond = cache_it->second.cond;
        if (request.use_uncond && cache_it->second.has_uncond) {
            embeds.uncond = cache_it->second.uncond;
        }
    } else {
        if (avatar_resident && !sd_ctx->sd->avatar_cond_cache.empty()) {
            LOG_INFO("avatar: prompt not in cache, reloading umT5 to recompute conditioning");
        }
        // umT5 may have been freed after the previous encode; reload before use.
        if (!sd_ctx->sd->reload_cond_stage_model()) {
            LOG_ERROR("avatar: umT5 reload failed; text conditioning unavailable");
        }
        embeds.cond          = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                   condition_params);
        if (request.use_uncond) {
            condition_params.text = request.negative_prompt;
            embeds.uncond         = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                        condition_params);
        }
        // Stash for subsequent renders before the TE is freed, keyed by prompt+negative.
        if (avatar_resident) {
            StableDiffusionGGML::CachedTextCond entry;
            entry.cond                          = embeds.cond;
            entry.uncond                        = embeds.uncond;
            entry.has_uncond                    = request.use_uncond;
            sd_ctx->sd->avatar_cond_cache[cache_key] = std::move(entry);
        }
    }
    embeds.cond.c_concat     = latents.concat_latent;
    embeds.cond.c_vector     = latents.clip_vision_output;
    if (request.use_uncond) {
        embeds.uncond.c_concat = latents.concat_latent;
        embeds.uncond.c_vector = latents.clip_vision_output;
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    // FIX A3 (env LTXAV_FREE_TE_COMPUTE, default ON; set "0" to disable): the gemma-3-12b
    // text-encoder leaves a large encode compute buffer (~2.9GB) resident after the
    // conditioning above. embeds.cond/.uncond are now fully-materialized CPU-owned
    // sd::Tensors (c_crossattn), so the encode compute buffer is no longer aliased and
    // can be freed before the DiT compute buffer (4524 MiB @193f) is allocated — this is
    // the direct unblock for the COLD-run OOM (warm prod worker pays this once). Params
    // buffer is left intact (freed separately below if applicable) so a resident TE can
    // re-encode the next segment's prompt without a reload.
    {
        const char* free_te_env = std::getenv("LTXAV_FREE_TE_COMPUTE");
        bool free_te_compute    = (free_te_env == nullptr) || (std::string(free_te_env) != "0");
        if (free_te_compute && sd_ctx->sd->cond_stage_model) {
            sd_ctx->sd->cond_stage_model->free_compute_buffer();
        }
    }

    // FIX A3b (env LTXAV_FREE_TE_PARAMS, default OFF): release the text-encoder PARAMS
    // (gemma ~3840 MB + text-projection/embeddings-connector ~2205 MB ≈ 6 GB) before the DiT
    // compute buffer (4524 MiB @193f) is allocated. The normal gate below SKIPS the param
    // free whenever keep_diffusion_model_resident is set — which is TRUE for the relip path
    // (avatar_resident = (longcat_avatar || LTXAV) && keep_diffusion_model_resident), so the
    // ~6 GB of TE params stay resident through the DiT phase to allow per-chain-segment
    // re-encode. For a SINGLE relip render that residency is pure waste and is the dominant
    // term in the 193f OOM (profiled). embeds.cond/.uncond are already CPU-owned sd::Tensors
    // (same A3 invariant — not aliasing the params), so freeing here is safe; if a later
    // chain segment needs a different prompt, reload_cond_stage_model() re-allocs + refills
    // the TE params on demand (one reload per segment, the existing umT5 mechanism). Default
    // OFF preserves the chain residency behaviour; opt in for single/long renders that OOM.
    if (sd_ctx->sd->cond_stage_model && std::getenv("LTXAV_FREE_TE_PARAMS") != nullptr &&
        std::string(std::getenv("LTXAV_FREE_TE_PARAMS")) != "0") {
        if (sd_ctx->sd->cond_stage_model->get_params_buffer_size() != 0) {
            // The warm relip worker keeps the TE's shared/runtime GPU residency even after
            // free_params_buffer(). Release that offloaded copy and trim its shared CUDA pool
            // first; embeddings are already CPU-owned and reload_cond_stage_model() restores
            // the TE on the next segment/request.
            sd_ctx->sd->cond_stage_model->release_all_gpu_param_residency();
            sd_ctx->sd->cond_stage_model->free_params_buffer();
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::TE));
            LOG_INFO("LTXAV: released+freed TE GPU params (gemma+projection) before DiT, ~6GB (LTXAV_FREE_TE_PARAMS=1)");
        }
    }

    // !keep_diffusion_model_resident: an in-process chain with PER-SEGMENT prompts must re-encode
    // the text encoder every segment, so it cannot be freed after seg0 (the avatar dodges this by
    // caching one cond). In --offload-to-cpu mode the TE param buffer is host RAM (mmap), not VRAM,
    // so keeping it resident costs host RAM, not GPU; it still streams to the GPU per encode.
    if (sd_ctx->sd->free_params_immediately && !use_text_cache && !sd_ctx->sd->keep_diffusion_model_resident) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }
    // Avatar umT5-on-GPU: now that the TE is freed, bring the DiT weights onto the
    // GPU (deferred at load to avoid TE+DiT coexisting). No-op when not deferred.
    sd_ctx->sd->finalize_deferred_dit_load();
    return embeds;
}

static sd_image_t* decode_video_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const sd::Tensor<float>& final_latent,
                                        int* num_frames_out) {
    if (final_latent.empty()) {
        LOG_ERROR("no latent video to decode");
        return nullptr;
    }
    // LTX latent-reuse harness: bank the raw post-sampling latent so the VAE-tiling
    // quality ladder can re-decode it (LTX_LOAD_LATENTS) without re-running the DiT.
    if (const char* save_path = getenv("LTX_SAVE_LATENTS"); save_path != nullptr && save_path[0] != '\0') {
        try {
            sd::save_tensor_to_file<float>(save_path, final_latent, "ltx_final_latent");
            LOG_INFO("LTX_SAVE_LATENTS: wrote post-sampling latent (%dx%dx%dx%d) to %s",
                     (int)final_latent.shape()[0], (int)final_latent.shape()[1],
                     (int)final_latent.shape()[2], (int)(final_latent.dim() > 3 ? final_latent.shape()[3] : 1),
                     save_path);
        } catch (const std::exception& e) {
            LOG_ERROR("LTX_SAVE_LATENTS failed: %s", e.what());
        }
    }
    sd::Tensor<float> video_latent = final_latent;
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        video_latent.shape()[3] > sd_ctx->sd->get_latent_channel()) {
        video_latent = sd::ops::slice(video_latent, 3, 0, sd_ctx->sd->get_latent_channel());
    }
    // LTXAV latent chaining: bank the clean VIDEO latent (channels only, audio stripped) so
    // the next segment can condition on its motion-carrying tail via --cont-latent.
    if (const char* sp = getenv("LTXAV_SAVE_VIDEO_LATENT"); sp != nullptr && sp[0] != '\0' &&
        sd_version_is_ltxav(sd_ctx->sd->version)) {
        try {
            sd::save_tensor_to_file<float>(sp, video_latent, "ltxav_video_latent");
            LOG_INFO("LTXAV_SAVE_VIDEO_LATENT: wrote video latent (%dx%dx%dx%d) to %s",
                     (int)video_latent.shape()[0], (int)video_latent.shape()[1],
                     (int)video_latent.shape()[2], (int)video_latent.shape()[3], sp);
        } catch (const std::exception& e) {
            LOG_ERROR("LTXAV_SAVE_VIDEO_LATENT failed: %s", e.what());
        }
    }
    // VACE latent chaining: bank the post-sampling diffusion-space latent (already
    // ref-frame stripped, same (mu-mean)/std space as encode_first_stage's output) so
    // the next VACE segment can inject its kept-frame tail directly into the `inactive`
    // context via VACE_CONT_LATENT, bypassing the lossy pixel decode->re-encode roundtrip.
    if (const char* sp = getenv("VACE_SAVE_LATENT"); sp != nullptr && sp[0] != '\0') {
        try {
            sd::save_tensor_to_file<float>(sp, video_latent, "vace_video_latent");
            LOG_INFO("VACE_SAVE_LATENT: wrote diffusion latent (%dx%dx%dx%d) to %s",
                     (int)video_latent.shape()[0], (int)video_latent.shape()[1],
                     (int)video_latent.shape()[2],
                     (int)(video_latent.dim() > 3 ? video_latent.shape()[3] : 1), sp);
        } catch (const std::exception& e) {
            LOG_ERROR("VACE_SAVE_LATENT failed: %s", e.what());
        }
    }
    // ---- WAN chaining OUTPUT anchor (anti-drift) ----
    // The injected-tail WAN_CONT_AGC only constrains the K overlap frames; the
    // freely-generated frames still let the distilled few-step DiT inflate std each
    // segment (measured ratchet 0.73->0.78->0.83->0.86 + mean drift, even with the
    // tail AGC on). Anchoring the WHOLE output latent's per-channel mean+std to the
    // fixed seg-0 reference BEFORE banking/decoding stops the compounding at its
    // source (the banked latent that feeds the next segment) and keeps the displayed
    // video consistent. Gated, default OFF; reuses WAN_CONT_AGC_REF (seg-0 latent).
    // seg-0 has no REF (establishes it) so it is never regraded.
    if (const char* oa = getenv("WAN_CONT_OUTPUT_AGC"); oa != nullptr && oa[0] == '1') {
        const char* refp = getenv("WAN_CONT_AGC_REF");
        if (refp != nullptr && refp[0] != '\0') {
            try {
                auto    ref  = sd::load_tensor_from_file_as_tensor<float>(refp);
                int64_t W    = video_latent.shape()[0], H = video_latent.shape()[1];
                int64_t T    = video_latent.shape()[2];
                int64_t C    = video_latent.dim() > 3 ? video_latent.shape()[3] : 1;
                int64_t per  = W * H * T;
                int64_t Cref = ref.dim() > 3 ? ref.shape()[3] : 1;
                int64_t perR = ref.shape()[0] * ref.shape()[1] * ref.shape()[2];
                if (Cref == C) {
                    const float* rd = ref.data();
                    float*       d  = video_latent.data();
                    for (int64_t c = 0; c < C; ++c) {
                        const float* rb  = rd + c * perR;
                        double       rs0 = 0, rsq = 0;
                        for (int64_t i = 0; i < perR; ++i) { rs0 += rb[i]; rsq += (double)rb[i] * rb[i]; }
                        double rm   = rs0 / (double)perR;
                        double rstd = rsq / (double)perR - rm * rm;
                        rstd        = rstd > 0 ? std::sqrt(rstd) : 0.0;
                        float* base = d + c * per;
                        double cs0 = 0, csq = 0;
                        for (int64_t i = 0; i < per; ++i) { cs0 += base[i]; csq += (double)base[i] * base[i]; }
                        double cm   = cs0 / (double)per;
                        double cstd = csq / (double)per - cm * cm;
                        cstd        = cstd > 0 ? std::sqrt(cstd) : 0.0;
                        double g = (cstd > 1e-6) ? (rstd / cstd) : 1.0;
                        if (g < 0.5) g = 0.5;
                        if (g > 2.0) g = 2.0;
                        for (int64_t i = 0; i < per; ++i)
                            base[i] = (float)((base[i] - cm) * g + rm);
                    }
                    LOG_INFO("WAN_CONT_OUTPUT_AGC: full-output per-channel mean+std anchored to seg-0 ref %s (%lld ch)",
                             refp, (long long)C);
                } else {
                    LOG_WARN("WAN_CONT_OUTPUT_AGC: ref ch %lld != latent ch %lld; skipping",
                             (long long)Cref, (long long)C);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("WAN_CONT_OUTPUT_AGC: failed to load ref %s: %s", refp, e.what());
            }
        }
    }
    LOG_DEBUG("decode_video_outputs latent %dx%dx%dx%d",
              (int)video_latent.shape()[0],
              (int)video_latent.shape()[1],
              (int)video_latent.shape()[2],
              (int)video_latent.shape()[3]);
    {
        // TEMP DEBUG: stats of the diffusion-space latent right before decode.
        const float* d = video_latent.data();
        int64_t      n = video_latent.numel();
        double sum = 0, sq = 0, mn = 1e30, mx = -1e30;
        int64_t nnan = 0;
        for (int64_t i = 0; i < n; ++i) {
            float v = d[i];
            if (std::isnan(v) || std::isinf(v)) { nnan++; continue; }
            sum += v; sq += (double)v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double mean = sum / (double)n;
        double var  = sq / (double)n - mean * mean;
        LOG_INFO("[DBG predecode latent] numel=%lld mean=%.5f std=%.5f min=%.4f max=%.4f nnan=%lld",
                 (long long)n, mean, var > 0 ? sqrt(var) : 0.0, mn, mx, (long long)nnan);
        // also per-frame mean (axis T = shape[2])
        int64_t W = video_latent.shape()[0], H = video_latent.shape()[1];
        int64_t Tn = video_latent.shape()[2], C = video_latent.dim() > 3 ? video_latent.shape()[3] : 1;
        for (int64_t t = 0; t < Tn && t < 8; ++t) {
            double fs = 0, fsq = 0;
            int64_t cnt = 0;
            for (int64_t c = 0; c < C; ++c)
                for (int64_t hh = 0; hh < H; ++hh)
                    for (int64_t ww = 0; ww < W; ++ww) {
                        // ne order [W,H,T,C] -> index
                        int64_t idx = ((c * Tn + t) * H + hh) * W + ww;
                        float v = d[idx];
                        fs += v; fsq += (double)v * v; cnt++;
                    }
            double fm = fs / (double)cnt;
            double fv = fsq / (double)cnt - fm * fm;
            LOG_INFO("[DBG predecode latent] frame %lld mean=%.5f std=%.5f", (long long)t, fm, fv > 0 ? sqrt(fv) : 0.0);
        }
    }
    // Dump the pre-VAE diffusion latent for offline VAE eval (LightVAE/LightTAE A/B).
    // Forwards via the WAN_ prefix. Same [W,H,T,C] f32 the
    // lighttae_eval harness reads; same space decode_first_stage de-norms internally.
    if (const char* sp = getenv("WAN_SAVE_LATENT"); sp != nullptr && sp[0] != '\0') {
        try {
            sd::save_tensor_to_file<float>(sp, video_latent, "wan_video_latent");
            LOG_INFO("WAN_SAVE_LATENT: wrote diffusion latent (%dx%dx%dx%d) to %s",
                     (int)video_latent.shape()[0], (int)video_latent.shape()[1],
                     (int)video_latent.shape()[2],
                     (int)(video_latent.dim() > 3 ? video_latent.shape()[3] : 1), sp);
        } catch (const std::exception& e) {
            LOG_ERROR("WAN_SAVE_LATENT failed: %s", e.what());
        }
        // WAN LATENT CHAINING drift metric: per-segment global std + per-channel
        // mean/std of the banked latent. This is the signal that exposed the VACE
        // 0.64->0.80->0.92 contrast runaway: if WAN_CONT_AGC holds, global std and
        // each channel's mean/std stay flat across the chain instead of ratcheting.
        // Harnesses can grep "[WAN_CHAIN_STATS]".
        {
            int64_t      W = video_latent.shape()[0], H = video_latent.shape()[1];
            int64_t      T = video_latent.shape()[2];
            int64_t      C = video_latent.dim() > 3 ? video_latent.shape()[3] : 1;
            int64_t      per = W * H * T;
            const float* d   = video_latent.data();
            double       gsum = 0.0, gsq = 0.0;
            int64_t      gn = W * H * T * C;
            for (int64_t i = 0; i < gn; ++i) {
                gsum += d[i];
                gsq += (double)d[i] * d[i];
            }
            double gmean = gsum / (double)gn;
            double gvar  = gsq / (double)gn - gmean * gmean;
            LOG_INFO("[WAN_CHAIN_STATS] global mean=%.5f std=%.5f (C=%lld T=%lld)",
                     gmean, gvar > 0 ? sqrt(gvar) : 0.0, (long long)C, (long long)T);
            for (int64_t c = 0; c < C; ++c) {
                const float* base = d + c * per;
                double       s = 0.0, sq = 0.0;
                for (int64_t i = 0; i < per; ++i) {
                    s += base[i];
                    sq += (double)base[i] * base[i];
                }
                double m   = s / (double)per;
                double var = sq / (double)per - m * m;
                LOG_INFO("[WAN_CHAIN_STATS] ch %02lld mean=%.5f std=%.5f",
                         (long long)c, m, var > 0 ? sqrt(var) : 0.0);
            }
        }
    }
    // auto z = sd::load_tensor_from_file_as_tensor<float>("ltx_vae_z.bin");
    // LTX_LOAD_LATENT (diagnostic): replace the just-sampled diffusion latent with one
    // saved earlier via WAN_SAVE_LATENT, so the SAME latent can be decoded through
    // different VAE weights/precisions for an airtight VAE-only A/B (the nvfp4/fp8 DiT is
    // not bit-deterministic run-to-run). No effect on prod (unset).
    if (const char* lp = getenv("LTX_LOAD_LATENT"); lp != nullptr && lp[0] != '\0') {
        try {
            video_latent = sd::load_tensor_from_file_as_tensor<float>(lp);
            LOG_INFO("LTX_LOAD_LATENT: decoding saved latent (%dx%dx%dx%d) from %s",
                     (int)video_latent.shape()[0], (int)video_latent.shape()[1],
                     (int)video_latent.shape()[2],
                     (int)(video_latent.dim() > 3 ? video_latent.shape()[3] : 1), lp);
        } catch (const std::exception& e) {
            LOG_ERROR("LTX_LOAD_LATENT failed: %s", e.what());
        }
    }
    int64_t t4            = ggml_time_ms();
    sd::Tensor<float> vid = sd_ctx->sd->decode_first_stage(video_latent, true);
    int64_t t5            = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t5 - t4) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (vid.empty()) {
        LOG_ERROR("decode_first_stage failed for video");
        return nullptr;
    }
    LOG_DEBUG("decode_video_outputs decoded %dx%dx%dx%d",
              (int)vid.shape()[0],
              (int)vid.shape()[1],
              (int)vid.shape()[2],
              (int)vid.shape()[3]);
    if (request.frames > 0 &&
        vid.shape()[2] > request.frames) {
        vid = sd::ops::slice(vid, 2, 0, request.frames);
    }

    sd_image_t* result_images = (sd_image_t*)calloc(vid.shape()[2], sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = static_cast<int>(vid.shape()[2]);
    }

    for (int64_t i = 0; i < vid.shape()[2]; i++) {
        result_images[i] = tensor_to_sd_image(vid, static_cast<int>(i));
    }

    return result_images;
}

static sd::Tensor<float> upscale_ltx_spatial_video_latent(sd_ctx_t* sd_ctx,
                                                          const char* model_path,
                                                          const sd::Tensor<float>& packed_latent,
                                                          int audio_length) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || packed_latent.empty()) {
        return {};
    }
    if (strlen(SAFE_STR(model_path)) == 0) {
        LOG_ERROR("LTX latent spatial upscale requires a model path");
        return {};
    }
    if (!sd_ctx->sd->ensure_backend_pair(SDBackendModule::UPSCALER)) {
        return {};
    }

    int latent_channels            = sd_ctx->sd->get_latent_channel();
    sd::Tensor<float> video_latent = packed_latent;
    sd::Tensor<float> audio_latent;
    if (packed_latent.shape()[3] > latent_channels) {
        video_latent = sd::ops::slice(packed_latent, 3, 0, latent_channels);
        audio_latent = unpack_ltxav_audio_latent(packed_latent, audio_length, latent_channels);
    }

    LOG_INFO("LTX latent spatial upscale: latent %dx%dx%dx%d -> model output",
             (int)video_latent.shape()[0],
             (int)video_latent.shape()[1],
             (int)video_latent.shape()[2],
             (int)video_latent.shape()[3]);

    sd::Tensor<float> unnormalized = sd_ctx->sd->un_normalize_ltx_video_latents(video_latent);
    if (unnormalized.empty()) {
        LOG_ERROR("LTX latent un-normalization failed before spatial upscale");
        return {};
    }

    std::shared_ptr<LTXVUpsampler::LatentUpsamplerRunner> upsampler = sd_ctx->sd->ltx_latent_upsampler;
    const std::string requested_model_path = SAFE_STR(model_path);
    if (!upsampler || sd_ctx->sd->ltx_latent_upsampler_path != requested_model_path) {
        int64_t load_start = ggml_time_ms();
        upsampler =
            std::make_shared<LTXVUpsampler::LatentUpsamplerRunner>(sd_ctx->sd->backend_for(SDBackendModule::UPSCALER),
                                                                   sd_ctx->sd->params_backend_for(SDBackendModule::UPSCALER));
        const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(sd_ctx->sd->max_vram);
        upsampler->set_max_graph_vram_bytes(max_graph_vram_bytes);
        if (!upsampler->load_from_file(model_path, sd_ctx->sd->n_threads)) {
            sd_ctx->sd->ltx_latent_upsampler.reset();
            sd_ctx->sd->ltx_latent_upsampler_path.clear();
            LOG_ERROR("load LTX latent upsampler failed");
            return {};
        }
        sd_ctx->sd->ltx_latent_upsampler      = upsampler;
        sd_ctx->sd->ltx_latent_upsampler_path = requested_model_path;
        LOG_INFO("[LTX_PHASE] latent upsampler load/cache fill took %.3fs", (ggml_time_ms() - load_start) * 1.0f / 1000);
    } else {
        LOG_INFO("LTX latent upsampler cache hit: %s", requested_model_path.c_str());
    }

    const char* upscale_window_env = std::getenv("LTX_UPSCALER_TEMPORAL_WINDOW");
    const bool temporal_windowing = upscale_window_env != nullptr && upscale_window_env[0] != '\0' &&
                                    std::string(upscale_window_env) != "0" && unnormalized.shape()[2] > 1;
    sd::Tensor<float> upscaled;
    if (!temporal_windowing) {
        upscaled = upsampler->compute(sd_ctx->sd->n_threads, unnormalized);
    } else {
        int window = std::max(2, std::atoi(upscale_window_env));
        int overlap = 2;
        if (const char* overlap_env = std::getenv("LTX_UPSCALER_TEMPORAL_OVERLAP"); overlap_env != nullptr) {
            overlap = std::max(1, std::atoi(overlap_env));
        }
        const int64_t total_frames = unnormalized.shape()[2];
        window = std::clamp(window, 2, static_cast<int>(total_frames));
        overlap = std::clamp(overlap, 1, window - 1);
        const int64_t stride = window - overlap;
        const int64_t input_channels = unnormalized.shape()[3];
        int tile_index = 0;
        int64_t produced_end = 0;
        for (int64_t start = 0;; start += stride, ++tile_index) {
            // Keep the final temporal tile at the regular window shape. A short
            // tail makes the upsampler build a different, larger graph-cut
            // working set. Re-anchor it against already-produced context and
            // emit only its unseen tail instead.
            const int64_t tile_start = start + window >= total_frames
                                           ? std::max<int64_t>(0, total_frames - window)
                                           : start;
            const int64_t end = std::min<int64_t>(total_frames, tile_start + window);
            const int64_t length = end - tile_start;
            auto input_tile = sd::ops::slice(unnormalized, 2, tile_start, end);
            LOG_INFO("LTX latent spatial upscale temporal-window tile %d: latent [%lld,%lld), window=%d retained-overlap=%lld",
                     tile_index,
                     (long long)tile_start,
                     (long long)end,
                     window,
                     (long long)(tile_index == 0 ? 0 : std::min<int64_t>(length, std::max<int64_t>(0, produced_end - tile_start))));
            auto output_tile = upsampler->compute(sd_ctx->sd->n_threads, input_tile);
            if (output_tile.empty()) {
                upscaled = {};
                break;
            }
            if (upscaled.empty()) {
                auto shape = output_tile.shape();
                shape[2] = total_frames;
                upscaled = sd::Tensor<float>(shape);
                upscaled.fill_(0.0f);
            }
            const int64_t frozen = tile_index == 0 ? 0 : std::min<int64_t>(length, std::max<int64_t>(0, produced_end - tile_start));
            const int64_t plane = output_tile.shape()[0] * output_tile.shape()[1];
            const int64_t output_channels = output_tile.shape()[3];
            if (output_channels != input_channels || output_tile.shape()[2] != length) {
                LOG_ERROR("LTX latent spatial upscale temporal window returned unexpected shape %lldx%lldx%lldx%lld for %lld input frames",
                          (long long)output_tile.shape()[0],
                          (long long)output_tile.shape()[1],
                          (long long)output_tile.shape()[2],
                          (long long)output_channels,
                          (long long)length);
                upscaled = {};
                break;
            }
            const float* src = output_tile.data();
            float* dst = upscaled.data();
            for (int64_t local = frozen; local < length; ++local) {
                for (int64_t channel = 0; channel < output_channels; ++channel) {
                    std::memcpy(dst + plane * (tile_start + local + total_frames * channel),
                                src + plane * (local + length * channel),
                                static_cast<size_t>(plane) * sizeof(float));
                }
            }
            produced_end = std::max(produced_end, end);
            if (end == total_frames) {
                break;
            }
        }
    }
    upsampler->free_compute_buffer();
    upsampler->release_all_gpu_param_residency();
    if (upscaled.empty()) {
        LOG_ERROR("LTX latent spatial upscale failed");
        return {};
    }

    upscaled = sd_ctx->sd->normalize_ltx_video_latents(upscaled);
    if (upscaled.empty()) {
        LOG_ERROR("LTX latent normalization failed after spatial upscale");
        return {};
    }

    if (!audio_latent.empty()) {
        upscaled = pack_ltxav_audio_and_video_latents(upscaled, audio_latent);
    }
    return upscaled;
}

// The LTX 2.3 temporal upscaler has a temporal pixel-shuffle followed by a
// [1, end) temporal slice, so T input latent frames produce 2*T-1 output
// latent frames.  This intentionally accepts VIDEO latents only: audio's
// independently packed timeline is not resampled by this test-stage.
static sd::Tensor<float> upscale_ltx_temporal_video_latent(sd_ctx_t* sd_ctx,
                                                           const char* model_path,
                                                           const sd::Tensor<float>& video_latent) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || video_latent.empty()) {
        return {};
    }
    if (strlen(SAFE_STR(model_path)) == 0) {
        LOG_ERROR("LTX latent temporal upscale requires a model path");
        return {};
    }
    if (!sd_ctx->sd->ensure_backend_pair(SDBackendModule::UPSCALER)) {
        return {};
    }

    const int64_t input_t = video_latent.shape()[2];
    sd::Tensor<float> unnormalized = sd_ctx->sd->un_normalize_ltx_video_latents(video_latent);
    if (unnormalized.empty()) {
        LOG_ERROR("LTX latent un-normalization failed before temporal upscale");
        return {};
    }

    std::shared_ptr<LTXVUpsampler::LatentUpsamplerRunner> upsampler = sd_ctx->sd->ltx_latent_upsampler;
    const std::string requested_model_path = SAFE_STR(model_path);
    if (!upsampler || sd_ctx->sd->ltx_latent_upsampler_path != requested_model_path) {
        int64_t load_start = ggml_time_ms();
        upsampler = std::make_shared<LTXVUpsampler::LatentUpsamplerRunner>(
            sd_ctx->sd->backend_for(SDBackendModule::UPSCALER),
            sd_ctx->sd->params_backend_for(SDBackendModule::UPSCALER));
        const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(sd_ctx->sd->max_vram);
        upsampler->set_max_graph_vram_bytes(max_graph_vram_bytes);
        if (!upsampler->load_from_file(model_path, sd_ctx->sd->n_threads)) {
            sd_ctx->sd->ltx_latent_upsampler.reset();
            sd_ctx->sd->ltx_latent_upsampler_path.clear();
            LOG_ERROR("load LTX temporal latent upsampler failed");
            return {};
        }
        sd_ctx->sd->ltx_latent_upsampler      = upsampler;
        sd_ctx->sd->ltx_latent_upsampler_path = requested_model_path;
        LOG_INFO("[LTX_PHASE] temporal latent upsampler load/cache fill took %.3fs",
                 (ggml_time_ms() - load_start) * 1.0f / 1000);
    } else {
        LOG_INFO("LTX temporal latent upsampler cache hit: %s", requested_model_path.c_str());
    }

    sd::Tensor<float> upscaled = upsampler->compute(sd_ctx->sd->n_threads, unnormalized);
    upsampler->free_compute_buffer();
    upsampler->release_all_gpu_param_residency();
    if (upscaled.empty()) {
        LOG_ERROR("LTX latent temporal upscale failed");
        return {};
    }
    if (upscaled.shape()[2] != 2 * input_t - 1) {
        LOG_ERROR("LTX temporal upscale returned unexpected latent T=%lld (expected %lld)",
                  (long long)upscaled.shape()[2], (long long)(2 * input_t - 1));
        return {};
    }

    upscaled = sd_ctx->sd->normalize_ltx_video_latents(upscaled);
    if (upscaled.empty()) {
        LOG_ERROR("LTX latent normalization failed after temporal upscale");
        return {};
    }
    return upscaled;
}

// Two-stage lipdub (Change B helper): encode the relip reference video at `out_w x out_h`
// (already the full target res / reference_downscale_factor), tile-encoded exactly like the
// stage-1 relip encode block, then temporally subsampled by `ref_tstride`. Returns the
// reference latent [Wl, Hl, ceil(Tl/T), Cl] (empty on failure). Standalone (does not touch the
// stage-1 encode block) so the default single-stage path stays byte-identical.
static sd::Tensor<float> encode_ltxav_relip_reference_latent(sd_ctx_t* sd_ctx,
                                                             const sd_vid_gen_params_t* sd_vid_gen_params,
                                                             int64_t content_w,
                                                             int64_t content_h,
                                                             int frames,
                                                             int ref_tstride) {
    int64_t cf_size = sd_vid_gen_params->control_frames_size;
    if (cf_size <= 0 || content_w <= 0 || content_h <= 0 || frames <= 0) {
        return {};
    }
    // content_w/h = OFFICIAL aspect-preserving downscale (full/scale). Resize the reference frames
    // to those dims (no squish), then neutral-gray PAD (bottom/right) up to the next VAE-scale
    // multiple: encode_first_stage derives latent dims by floor division (vae.hpp:145), so padding
    // (not resizing) yields ceil(content/vsf) latent rows — matching the official VAE's internal
    // padding so the ref RoPE span covers the full target extent.
    int64_t vsf   = std::max<int64_t>(1, sd_ctx->sd->get_vae_scale_factor());
    int64_t enc_w = ((content_w + vsf - 1) / vsf) * vsf;
    int64_t enc_h = ((content_h + vsf - 1) / vsf) * vsf;
    sd::Tensor<float> ref_video = sd::full<float>({content_w, content_h, frames, 3, 1}, 0.5f);
    for (int64_t i = 0; i < frames; ++i) {
        int64_t src       = std::min<int64_t>(i, cf_size - 1);
        auto reference_fr = sd_image_to_tensor(sd_vid_gen_params->control_frames[src], content_w, content_h);
        sd::ops::slice_assign(&ref_video, 2, i, i + 1, reference_fr.unsqueeze(2));
    }
    if (enc_h > content_h) {
        ref_video = sd::ops::concat(ref_video,
                                    sd::full<float>({content_w, enc_h - content_h, frames, 3, 1}, 0.5f), 1);
    }
    if (enc_w > content_w) {
        ref_video = sd::ops::concat(ref_video,
                                    sd::full<float>({enc_w - content_w, enc_h, frames, 3, 1}, 0.5f), 0);
    }
    LOG_INFO("LTXAV two-stage: stage2 ref encode content %lldx%lld -> padded %lldx%lld px",
             (long long)content_w, (long long)content_h, (long long)enc_w, (long long)enc_h);
    sd_tiling_params_t relip_saved_tiling = sd_ctx->sd->vae_tiling_params;
    std::string relip_enc_tiling_args;
    {
        float enc_tile = 0.25f;
        if (const char* e = getenv("LTXAV_RELIP_ENCODE_TILE")) {
            float v = (float)atof(e);
            if (v > 0.f && v <= 1.f) enc_tile = v;
        }
        sd_ctx->sd->vae_tiling_params.enabled    = true;
        sd_ctx->sd->vae_tiling_params.rel_size_x = enc_tile;
        sd_ctx->sd->vae_tiling_params.rel_size_y = enc_tile;
        int enc_tframes = 1;
        if (const char* e = getenv("LTXAV_RELIP_ENCODE_TFRAMES")) {
            int v = atoi(e);
            if (v >= 1) enc_tframes = v;
        }
        relip_enc_tiling_args = "temporal_tile_frames=" + std::to_string(enc_tframes) + ",temporal_tile_overlap=0";
        sd_ctx->sd->vae_tiling_params.temporal_tiling   = true;
        sd_ctx->sd->vae_tiling_params.extra_tiling_args = relip_enc_tiling_args.c_str();
    }
    sd::Tensor<float> reference_latent;
    if (const char* e = getenv("LTXAV_RELIP_ENCODE_TFRAMES")) {
        // FIX 1: host-level temporal-chunked encode (bounds the full-res stage-2 encode buffer,
        // the 193f OOM). Default (env unset) keeps the monolithic single-graph encode.
        reference_latent = encode_relip_reference_temporal_chunked(sd_ctx, ref_video, atoi(e));
    } else {
        reference_latent = sd_ctx->sd->encode_first_stage(ref_video);
    }
    sd_ctx->sd->vae_tiling_params = relip_saved_tiling;
    if (reference_latent.empty()) {
        return {};
    }
    // Free the VAE-encode compute buffer NOW, before the stage-2 DiT alloc (mirrors :5963 / the
    // stage-1 encode). The decode re-allocates lazily.
    sd_ctx->sd->first_stage_model->free_compute_buffer();

    if (ref_tstride > 1 && reference_latent.shape()[2] > 1) {
        int64_t orig_f = reference_latent.shape()[2];
        int64_t n_sub  = (orig_f + ref_tstride - 1) / ref_tstride;
        std::vector<int64_t> sub_shape = reference_latent.shape();
        sub_shape[2]                   = n_sub;
        sd::Tensor<float> sub(sub_shape);
        int64_t j = 0;
        for (int64_t f = 0; f < orig_f; f += ref_tstride) {
            sd::ops::slice_assign(&sub, 2, j, j + 1, sd::ops::slice(reference_latent, 2, f, f + 1));
            ++j;
        }
        reference_latent = std::move(sub);
    }
    limit_relip_reference_latent_frames(&reference_latent, "stage2");
    return reference_latent;
}

static bool apply_ltxv_refine_image_conditioning(sd_ctx_t* sd_ctx,
                                                 const sd_vid_gen_params_t* sd_vid_gen_params,
                                                 const GenerationRequest& request,
                                                 const ImageGenerationLatents& latents,
                                                 sd::Tensor<float>* latent,
                                                 sd::Tensor<float>* denoise_mask,
                                                 sd::Tensor<float>* video_positions,
                                                 sd::Tensor<float>* video_reference_out = nullptr) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_vid_gen_params == nullptr ||
        latent == nullptr || latent->empty() || denoise_mask == nullptr || video_positions == nullptr) {
        return true;
    }

    // Change B — Two-stage lipdub: RE-APPLY the relip reference at the upscaled full res. The
    // *latent here is the upscaled stage-1 latent (target-only grid); the reference is encoded
    // fresh at full-res / reference_downscale_factor and handed to stage-2 as a SEPARATE token
    // block (video_reference_out) with combined target+ref positions — NEVER grid-concatenated
    // (that would be the 2L-token OOM). denoise_mask stays all-ones target. Stage-2 then refines
    // the target while attending to the (downscaled) full-res reference, sharpening identity.
    if (latents.relip_twostage) {
        if (sd_ctx->sd->vae_decode_only) {
            LOG_ERROR("LTXV two-stage relip refine requires VAE encoder weights");
            return false;
        }
        if (video_reference_out == nullptr) {
            LOG_ERROR("LTXV two-stage relip refine requires a video_reference output");
            return false;
        }
        int lat_ch       = sd_ctx->sd->get_latent_channel();
        sd::Tensor<float> video_latent = *latent;
        sd::Tensor<float> audio_latent;
        if (latent->shape()[3] > lat_ch) {
            video_latent = sd::ops::slice(*latent, 3, 0, lat_ch);
            audio_latent = unpack_ltxav_audio_latent(*latent, latents.audio_length, lat_ch);
        }
        // Official LipDub stage 2 freezes the stage-1 target audio and appends that
        // same clean target as the negative-position audio reference.  Do not carry
        // the original drive block through this pass: stage 1 has already translated
        // it into the model's target-audio representation.
        if (latents.audio_reference_conditioning && !audio_latent.empty()) {
            const int target_length = latents.audio_target_length;
            if (target_length <= 0 || audio_latent.shape()[1] < target_length) {
                LOG_ERROR("LTXAV LipDub stage2: invalid target audio length %d for latent T=%lld",
                          target_length, (long long)audio_latent.shape()[1]);
                return false;
            }
            auto stage1_target = sd::ops::slice(audio_latent, 1, 0, target_length);
            audio_latent = sd::ops::concat(stage1_target, stage1_target, 1);
            LOG_INFO("LTXAV LipDub stage2: froze %d stage1 target-audio tokens and appended them as clean audio reference",
                     target_length);
        }
        int64_t target_lat_w   = video_latent.shape()[0];
        int64_t target_lat_h   = video_latent.shape()[1];
        int64_t target_lat_f   = video_latent.shape()[2];
        int full_w             = static_cast<int>(target_lat_w) * request.vae_scale_factor;
        int full_h             = static_cast<int>(target_lat_h) * request.vae_scale_factor;
        int ds                 = std::max(1, latents.relip_ref_downscale);
        int64_t vsf            = std::max<int64_t>(1, request.vae_scale_factor);
        // Aspect-preserving content dims (official: full/scale). The encode helper resizes to these
        // (no squish) then pads up to a VAE-scale multiple internally (ceil rows). At full res
        // full/ds is usually already a vsf multiple (e.g. 704/2=352), so the pad is a no-op here;
        // the fix matters most at the half-res stage-1 (e.g. 352/2=176 -> pad to 192).
        int64_t content_w      = full_w / ds;
        int64_t content_h      = full_h / ds;
        if (ds > 1 && (content_w < vsf || content_h < vsf)) {
            LOG_WARN("LTXV two-stage relip: downscale=%d too large for %dx%d; using full-res reference in stage-2", ds, full_w, full_h);
            ds        = 1;
            content_w = full_w;
            content_h = full_h;
        }
        sd::Tensor<float> reference_latent = encode_ltxav_relip_reference_latent(sd_ctx,
                                                                                sd_vid_gen_params,
                                                                                content_w,
                                                                                content_h,
                                                                                request.frames,
                                                                                latents.relip_ref_tstride);
        if (reference_latent.empty() || reference_latent.shape()[3] != video_latent.shape()[3]) {
            LOG_ERROR("LTXV two-stage relip: stage-2 reference encode failed (or channel mismatch)");
            return false;
        }
        int64_t ref_lat_w = reference_latent.shape()[0];
        int64_t ref_lat_h = reference_latent.shape()[1];
        int64_t ref_lat_f = reference_latent.shape()[2];
        *video_reference_out = reference_latent;
        *video_positions     = build_ltxv_relip_video_positions(target_lat_w,
                                                            target_lat_h,
                                                            target_lat_f,
                                                            ref_lat_f,
                                                            request.fps,
                                                            request.vae_scale_factor,
                                                            8,
                                                            true,
                                                            ref_lat_w,
                                                            ref_lat_h,
                                                            request.vae_scale_factor * ds,
                                                            latents.relip_ref_tstride);
        // Target grid stays as-is (upscaled), denoise_mask all-ones (the target is fully refined;
        // the reference is a separate frozen token block handled in the DiT).
        sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video_latent, 1.f);
        if (!audio_latent.empty()) {
            *latent       = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
            // Freeze the (driven) audio slot when audio_fixed, mirroring stage-1 (:6543-6546):
            // audio_mask_value 0 = pinned every step, 1 = generated.
            auto audio_mask = latents.audio_reference_conditioning
                                  ? make_ltxav_lipdub_audio_mask(audio_latent, latents.audio_target_length, true)
                                  : sd::Tensor<float>();
            *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent,
                                                                    latents.audio_fixed ? 0.0f : 1.0f,
                                                                    audio_mask.empty() ? nullptr : &audio_mask);
        } else {
            *latent       = std::move(video_latent);
            *denoise_mask = std::move(video_mask);
        }
        LOG_INFO("LTXAV two-stage: stage2 %dx%d refine, ref content %lldx%lld -> ref latent grid %lldx%lld x %lld frames (downscale=%d tstride=%d); target latent %lldx%lld: target tokens %lld + ref tokens %lld = %lld",
                 full_w, full_h, (long long)content_w, (long long)content_h,
                 (long long)ref_lat_w, (long long)ref_lat_h, (long long)ref_lat_f, ds, latents.relip_ref_tstride,
                 (long long)target_lat_w, (long long)target_lat_h,
                 (long long)(target_lat_w * target_lat_h * target_lat_f),
                 (long long)(ref_lat_w * ref_lat_h * ref_lat_f),
                 (long long)(target_lat_w * target_lat_h * target_lat_f + ref_lat_w * ref_lat_h * ref_lat_f));
        return true;
    }

    const bool has_timeline_keyframes = sd_vid_gen_params->keyframes != nullptr &&
                                        sd_vid_gen_params->keyframes_size > 0;
    if (sd_vid_gen_params->init_image.data == nullptr &&
        sd_vid_gen_params->end_image.data == nullptr && !has_timeline_keyframes) {
        // No image conditioning to re-apply in the refine. For an AUDIO-DRIVEN LTXAV render
        // (latents.audio_fixed) the stage-2 refine must still PIN the driving-audio slot
        // (mask=0) every step, exactly like stage-1 (:7793-7801) and the relip stage-2 path
        // (:8417) — otherwise the empty hires denoise_mask lets stage-2 re-denoise the driving
        // audio and washes out the lip-sync stage-1 established. Build a full-generated video
        // mask (1.0) + audio-pinned mask when a packed audio slot is present. Gated on
        // audio_fixed so t2v/i2v and no-drive LTXAV renders keep the empty-mask path (unchanged).
        if (sd_version_is_ltxav(sd_ctx->sd->version) && latents.audio_fixed) {
            int lat_ch = sd_ctx->sd->get_latent_channel();
            if (latent->shape()[3] > lat_ch) {
                sd::Tensor<float> video_latent = sd::ops::slice(*latent, 3, 0, lat_ch);
                sd::Tensor<float> audio_latent = unpack_ltxav_audio_latent(*latent, latents.audio_length, lat_ch);
                if (!audio_latent.empty()) {
                    sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video_latent, 1.f);
                    *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent,
                                                                           latents.audio_fixed ? 0.0f : 1.0f);
                }
            }
        }
        return true;
    }
    if (sd_ctx->sd->vae_decode_only) {
        LOG_ERROR("LTXV refine image conditioning requires VAE encoder weights; create the context with vae_decode_only=false");
        return false;
    }

    constexpr float conditioning_strength = 1.f;
    int latent_channels                   = sd_ctx->sd->get_latent_channel();
    sd::Tensor<float> video_latent        = *latent;
    sd::Tensor<float> audio_latent;
    if (latent->shape()[3] > latent_channels) {
        video_latent = sd::ops::slice(*latent, 3, 0, latent_channels);
        audio_latent = unpack_ltxav_audio_latent(*latent, latents.audio_length, latent_channels);
        if (audio_latent.empty()) {
            LOG_ERROR("failed to unpack LTXAV audio latent before image-to-video inplace conditioning");
            return false;
        }
    }

    int image_width              = static_cast<int>(video_latent.shape()[0]) * request.vae_scale_factor;
    int image_height             = static_cast<int>(video_latent.shape()[1]) * request.vae_scale_factor;
    sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video_latent, 1.f);

    if (has_timeline_keyframes) {
        if (video_reference_out == nullptr) {
            LOG_ERROR("LTXV refine multi-keyframe conditioning requires a video_reference output");
            return false;
        }
        std::vector<int> keyframe_positions;
        sd::Tensor<float> keyframe_reference;
        for (int i = 0; i < sd_vid_gen_params->keyframes_size; ++i) {
            const int frame_idx = sd_vid_gen_params->keyframe_frame_indices != nullptr
                                      ? sd_vid_gen_params->keyframe_frame_indices[i]
                                      : 0;
            if (frame_idx < 0 || frame_idx >= request.frames || sd_vid_gen_params->keyframes[i].data == nullptr) {
                LOG_ERROR("invalid LTXV refine keyframe %d (frame=%d, range=[0,%d), image=%p)",
                          i, frame_idx, request.frames, (void*)sd_vid_gen_params->keyframes[i].data);
                return false;
            }
            auto keyframe_image = sd_image_to_tensor(sd_vid_gen_params->keyframes[i], image_width, image_height);
            auto keyframe_latent = encode_ltxav_condition_image(sd_ctx, keyframe_image, "refine keyframe");
            if (keyframe_latent.empty() || keyframe_latent.shape()[0] != video_latent.shape()[0] ||
                keyframe_latent.shape()[1] != video_latent.shape()[1] || keyframe_latent.shape()[2] != 1 ||
                keyframe_latent.shape()[3] != video_latent.shape()[3]) {
                LOG_ERROR("invalid LTXV refine keyframe %d latent shape", i);
                return false;
            }
            keyframe_reference = keyframe_reference.empty()
                                     ? std::move(keyframe_latent)
                                     : sd::ops::concat(keyframe_reference, keyframe_latent, 2);
            keyframe_positions.push_back(frame_idx);
        }
        *video_reference_out = std::move(keyframe_reference);
        *video_positions = build_ltxv_multi_keyframe_video_positions(video_latent.shape()[0],
                                                                       video_latent.shape()[1],
                                                                       video_latent.shape()[2],
                                                                       keyframe_positions,
                                                                       request.fps,
                                                                       request.vae_scale_factor,
                                                                       8,
                                                                       true);
        LOG_INFO("LTXV refine: re-applied %d timeline keyframe guide(s) at %dx%d",
                 sd_vid_gen_params->keyframes_size, image_width, image_height);
    } else if (sd_vid_gen_params->init_image.data != nullptr) {
        // Frame 0 (the i2v opener) is normally pinned at strength 1.0 -> denoise mask 0.0, i.e.
        // frozen: it is the only frame that skips the SDEdit refine's texture-harmonizing pass, so it
        // comes out as a raw VAE round-trip of the init image (over-sharp "crunch") while every other
        // frame gets the softer refined look. LTXAV_REFINE_INIT_STRENGTH < 1.0 lets the refine partially
        // re-diffuse frame 0 so its texture matches its neighbours while still anchoring identity.
        float init_strength = conditioning_strength;
        if (const char* e = std::getenv("LTXAV_REFINE_INIT_STRENGTH")) {
            init_strength = std::clamp(static_cast<float>(atof(e)), 0.f, 1.f);
            LOG_INFO("LTXV refine init pin strength overridden -> %.3f (mask=%.3f) via LTXAV_REFINE_INIT_STRENGTH",
                     init_strength, 1.f - init_strength);
        }
        sd::Tensor<float> start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, image_width, image_height);
        if (!apply_ltxav_condition_image_by_latent_index(sd_ctx,
                                                         start_image,
                                                         &video_latent,
                                                         &video_mask,
                                                         0,
                                                         "init",
                                                         init_strength)) {
            return false;
        }
    }

    if (sd_vid_gen_params->end_image.data != nullptr) {
        sd::Tensor<float> end_image        = sd_image_to_tensor(sd_vid_gen_params->end_image, image_width, image_height);
        sd::Tensor<float> end_image_latent = encode_ltxav_condition_image(sd_ctx, end_image, "end");
        if (end_image_latent.empty()) {
            return false;
        }

        int frame_idx = request.frames - 1;
        if (frame_idx == 0) {
            if (!apply_ltxav_condition_by_latent_index(&video_latent,
                                                       &video_mask,
                                                       end_image_latent,
                                                       0,
                                                       "end",
                                                       1.f - conditioning_strength)) {
                return false;
            }
        } else {
            if (latents.video_conditioning_frame_count <= 0 || latents.video_target_frame_count <= 0) {
                LOG_ERROR("LTXV FLF2V refine conditioning requires low-resolution keyframe conditioning metadata");
                return false;
            }
            int64_t target_latent_frames = latents.video_target_frame_count;
            if (!apply_ltxav_condition_by_latent_index(&video_latent,
                                                       &video_mask,
                                                       end_image_latent,
                                                       target_latent_frames,
                                                       "end",
                                                       1.f - conditioning_strength)) {
                return false;
            }
            *video_positions = build_ltxv_video_positions(video_latent.shape()[0],
                                                          video_latent.shape()[1],
                                                          target_latent_frames,
                                                          end_image_latent.shape()[2],
                                                          frame_idx,
                                                          1,
                                                          request.fps,
                                                          request.vae_scale_factor,
                                                          8,
                                                          true);
        }
    }

    if (!audio_latent.empty()) {
        *latent       = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
        // Pin the driving-audio slot (mask=0) when audio_fixed, mirroring stage-1 (:7800) and the
        // relip stage-2 path (:8417); the default 1.0 (generated) let the refine denoise the driving
        // audio and drop lip-sync. No-drive renders have audio_fixed=false -> 1.0 = unchanged.
        *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent,
                                                                latents.audio_fixed ? 0.0f : 1.0f);
    } else {
        *latent       = std::move(video_latent);
        *denoise_mask = std::move(video_mask);
    }
    LOG_INFO("LTXV refine image conditioning applied at %dx%d", image_width, image_height);
    return true;
}

SD_API bool generate_video_ex(sd_ctx_t* sd_ctx,
                              const sd_vid_gen_params_t* sd_vid_gen_params,
                              sd_image_t** frames_out,
                              int* num_frames_out,
                              sd_audio_t** audio_out,
                              int* output_fps,
                              float** final_latent_out,
                              int* latent_width_out,
                              int* latent_height_out,
                              int* latent_frames_out,
                              int* latent_channels_out,
                              float** refined_latent_out,
                              int* refined_latent_width_out,
                              int* refined_latent_height_out,
                              int* refined_latent_frames_out,
                              int* refined_latent_channels_out,
                              float** refined_latent_lo_out,
                              int* refined_latent_lo_width_out,
                              int* refined_latent_lo_height_out,
                              int* refined_latent_lo_frames_out,
                              int* refined_latent_lo_channels_out) {
    if (sd_ctx == nullptr || sd_vid_gen_params == nullptr) {
        return false;
    }
    if (frames_out != nullptr) {
        *frames_out = nullptr;
    }
    if (audio_out != nullptr) {
        *audio_out = nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = 0;
    }
    int effective_output_fps = std::max(1, sd_vid_gen_params->fps);
    if (output_fps != nullptr) {
        *output_fps = effective_output_fps;
    }
    if (final_latent_out != nullptr) {
        *final_latent_out = nullptr;
    }
    if (refined_latent_out != nullptr) {
        *refined_latent_out = nullptr;
    }
    if (refined_latent_lo_out != nullptr) {
        *refined_latent_lo_out = nullptr;
    }
    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
    // Avatar: the full-clip Wan-VAE temporal decode at 480p OOMs a 12GB card
    // (~11.9 GiB peak -> CUDA OOM). Spatial tiling bounds the per-tile activation
    // footprint (peak ~10.8 GiB) and decodes ~10x faster than CPU (54s vs 570s).
    // So when the VAE runs on GPU and tiling was not explicitly requested, turn it
    // on by default for the avatar. Forcing the VAE to CPU (--vae-on-cpu) or
    // passing --vae-tiling/--vae-tile-size explicitly both still take precedence.
    if (sd_version_is_longcat_avatar(sd_ctx->sd->version) &&
        !sd_ctx->sd->vae_tiling_params.enabled &&
        !sd_backend_is_cpu(sd_ctx->sd->backend_for(SDBackendModule::VAE))) {
        sd_ctx->sd->vae_tiling_params.enabled = true;
        // lap-21: VAE decode time is ∝ total tile area (overcompute). The stock
        // 0.5 overlap recomputes ~64% extra; at the avatar's 60x104 latent / 32
        // tile it yields 2x5=10 tiles. Dropping to 0.25 overlap gives 2x4=8 tiles
        // (the code floors at ~8 here either way) -> VAE decode -20% (80.5->64.2s
        // @ 37f, validated coherent ac16 0.833-0.842, no visible seams). Bigger
        // tiles OOM (full decode needs ~20 GiB; 60x60 OOMs; 60x40 is slower+10GB),
        // so 32-tile/0.25-overlap is the measured optimum. Only lower it from the
        // stock 0.5 default (an explicit --vae-tile-overlap of any other value is
        // respected).
        if (sd_ctx->sd->vae_tiling_params.target_overlap == 0.5f) {
            sd_ctx->sd->vae_tiling_params.target_overlap = 0.25f;
        }
        LOG_INFO("avatar: enabling VAE spatial tiling by default (GPU decode; avoids OOM, ~10x faster than CPU; overlap %.2f). Pass --vae-on-cpu to disable.",
                 sd_ctx->sd->vae_tiling_params.target_overlap);
    }
    // For the avatar path the conditioning audio is an INPUT; capture the 16k
    // mono waveform here so it can be muxed back into the output container
    // (so clips come out viewable WITH sound, no manual ffmpeg).
    std::vector<float> avatar_input_wav;
    GenerationRequest request(sd_ctx, sd_vid_gen_params);
    // A populated chain replaces (rather than augments) legacy hires. Selecting stage 0
    // here deliberately leaves the no-chain control flow byte-for-byte on its old path.
    const bool hires_chain_enabled = sd_vid_gen_params->hires_chain != nullptr &&
                                     sd_vid_gen_params->hires_chain_count > 0;
    if (hires_chain_enabled) {
        request.hires = sd_vid_gen_params->hires_chain[0];
        request.hires.enabled = true;
        request.resolve_hires();
        // The base embeds must include uncond when any later stage needs CFG/CFG++, too.
        for (int i = 0; i < sd_vid_gen_params->hires_chain_count; ++i) {
            const sd_hires_params_t& stage = sd_vid_gen_params->hires_chain[i];
            if (stage.cfg > 1.f || stage.sample_method == EULER_CFG_PP_SAMPLE_METHOD ||
                stage.sample_method == EULER_A_CFG_PP_SAMPLE_METHOD) {
                request.use_uncond = true;
                break;
            }
        }
    }

    // Change A — Two-stage lipdub relip (LTXAV_RELIP_TWOSTAGE=1). Port of the official
    // ltx_pipelines/lipdub.py two-stage: stage-1 renders the relip at HALF resolution
    // from-noise (cheap, tiny DiT buffer), then the existing LTX latent-upscale path
    // upsamples the latent 2x and stage-2 REFINES at full res with the reference re-applied
    // (Change B) at a low-noise schedule (Change D). This is the only configuration that fits
    // 193f at full identity: stage-2 is a 1.25L-token refine of an already-faithful upscaled
    // latent, not a 2L-token from-noise generation. Default OFF => byte-identical single-stage.
    // Requires the caller to also pass the LTX latent upsampler (--hires-model); we cannot
    // synthesize that path. Sets up the hires latent-upscale BEFORE latent_upscale_enabled /
    // hires_request are read below.
    bool relip_twostage = false;
    if (std::getenv("LTXAV_RELIP_TWOSTAGE") != nullptr &&
        std::string(std::getenv("LTXAV_RELIP_TWOSTAGE")) != "0" &&
        sd_version_is_ltxav(sd_ctx->sd->version) &&
        sd_vid_gen_params->control_frames_size > 0 &&
        sd_vid_gen_params->v2v_mode == 0) {  // relip (lipdub) ONLY — NOT SDEdit(1)/guide-edit(2),
                                             // which share control_frames but must not two-stage
                                             // (else the prod LTXAV_RELIP_TWOSTAGE=1 env breaks all v2v)
        if (strlen(SAFE_STR(request.hires.model_path)) == 0) {
            LOG_ERROR("LTXAV_RELIP_TWOSTAGE=1 requires the LTX latent upsampler model (pass --hires-model <spatial_upsampler.safetensors>)");
            return false;
        }
        // Two-stage needs full W,H divisible by 64 so the half-res stage-1 is latent-aligned
        // (half divisible by 32 >= vae_scale_factor) and the 2x upscale returns exactly to W,H.
        if (request.width % 64 != 0 || request.height % 64 != 0) {
            LOG_ERROR("LTXAV_RELIP_TWOSTAGE requires width/height divisible by 64 (got %dx%d); two-stage needs a latent-aligned half-res stage-1",
                      request.width, request.height);
            return false;
        }
        // Hardening: at REFDS>1 the half-res stage-1 reference is downscaled again, to
        // (W/2)/REFDS x (H/2)/REFDS. When that isn't a VAE-scale multiple it gets neutral-gray
        // padded up to one (handled correctly in prepare_video_generation_latents) — log it so the
        // extra padded ref row in the render log isn't mistaken for a regression. Divisible dims
        // (full W,H % (2*REFDS*vae_scale_factor) == 0) need no padding.
        if (const char* e = std::getenv("LTXAV_RELIP_REF_DOWNSCALE")) {
            int refds   = std::atoi(e);
            int vsf     = std::max(1, sd_ctx->sd->get_vae_scale_factor());
            int align   = refds * vsf;
            if (refds > 1 && ((request.width / 2) % align != 0 || (request.height / 2) % align != 0)) {
                LOG_WARN("LTXAV_RELIP_TWOSTAGE: half-res stage-1 dims %dx%d are not divisible by REFDS*vae_scale (%d); "
                         "the downscaled reference will be neutral-gray padded to the next %d-multiple before encode "
                         "(use full W,H divisible by %d for an exact downscale)",
                         request.width / 2, request.height / 2, align, vsf, 2 * align);
            }
        }
        int full_w = request.width;
        int full_h = request.height;
        relip_twostage         = true;
        request.hires.enabled  = true;
        request.hires.upscaler = SD_HIRES_UPSCALER_MODEL;  // LTX learned latent upsampler (model_path)
        request.hires.scale    = 2.0f;
        // Change D — default stage-2 sigmas = official STAGE_2_DISTILLED_SIGMAS (3-step refine,
        // sigma0=0.909375 = the noise added to the upscaled latent), unless the caller passed
        // --hires-sigmas. Static backing array => valid for the lifetime of the request.
        //
        // LTXAV_TWOSTAGE_USE_HIRES_STRENGTH=1 is an explicit diagnostic path: leave
        // custom_sigmas empty so make_hires_sigma_schedule() can build/trim the schedule from
        // hires.denoising_strength. Without this, --hires-denoising-strength is logged but
        // ignored because the official custom sigma vector returns early from schedule creation.
        bool use_hires_strength_schedule = false;
        if (const char* e = std::getenv("LTXAV_TWOSTAGE_USE_HIRES_STRENGTH")) {
            use_hires_strength_schedule = e[0] != '\0' && std::string(e) != "0";
        }
        if (!use_hires_strength_schedule &&
            (request.hires.custom_sigmas_count <= 0 || request.hires.custom_sigmas == nullptr)) {
            static float kStage2DistilledSigmas[] = {0.909375f, 0.725f, 0.421875f, 0.0f};
            request.hires.custom_sigmas       = kStage2DistilledSigmas;
            request.hires.custom_sigmas_count = 4;
        } else if (use_hires_strength_schedule &&
                   (request.hires.custom_sigmas_count <= 0 || request.hires.custom_sigmas == nullptr)) {
            LOG_INFO("LTXAV_TWOSTAGE_USE_HIRES_STRENGTH=1: using generated hires schedule trimmed by denoising_strength=%.2f",
                     request.hires.denoising_strength);
        }
        // Stage-1 at HALF the final resolution (the relip block below renders at request.w/h).
        request.width  = full_w / 2;
        request.height = full_h / 2;
        LOG_INFO("LTXAV two-stage: stage1 %dx%d from-noise (half-res) -> latent-upscale 2x -> stage2 %dx%d refine",
                 request.width, request.height, full_w, full_h);
    }

    bool same_res_refine_enabled = false;
    if (const char* e = std::getenv("LTX_REFINE_NO_UPSCALE")) {
        same_res_refine_enabled = e[0] != '\0' && std::string(e) != "0";
    }
    same_res_refine_enabled      = same_res_refine_enabled && request.hires.enabled;
    bool latent_refine_enabled   = request.hires.enabled;
    bool latent_upscale_enabled  = latent_refine_enabled && !same_res_refine_enabled;
    // LTX_HIRES_CONTINUE: reproduce the ver3 two-stage graph.  Its partial LCM
    // base stage publishes the final denoised x0 estimate (not the LCM trajectory
    // point at σ=0.725); the learned upsampler operates on that clean latent, and
    // the Euler refine SDEdit-noises the upscaled x0 at σ=0.725 with fresh noise.
    // Off (default) = today's existing SDEdit hires behavior.
    // Gated to plain two-pass t2v/i2v (latent upscale; not relip / SDEdit / guide-edit) so those paths
    // are byte-identical. NOTE: base+refine sigmas must be contiguous (base ends where refine begins).
    bool hires_continue_mode = false;
    if (const char* e = std::getenv("LTX_HIRES_CONTINUE")) {
        hires_continue_mode = e[0] != '\0' && std::string(e) != "0";
    }
    hires_continue_mode = hires_continue_mode && latent_upscale_enabled &&
                          sd_version_is_ltxav(sd_ctx->sd->version) && !relip_twostage &&
                          sd_vid_gen_params->v2v_mode == 0 && sd_vid_gen_params->control_frames_size <= 0;
    if (hires_continue_mode) {
        LOG_INFO("LTX_HIRES_CONTINUE: ver3 x0/SDEdit hires (partial LCM base denoised_output -> upscale -> fresh-noise Euler refine)");
    }
    GenerationRequest hires_request = request;
    if (latent_refine_enabled) {
        if (!sd_version_is_ltxav(sd_ctx->sd->version)) {
            LOG_ERROR("LTX latent refine is only supported for LTX video models");
            return false;
        }
        if (latent_upscale_enabled && request.hires.upscaler != SD_HIRES_UPSCALER_MODEL) {
            LOG_ERROR("LTX latent spatial upscale currently requires hires upscaler MODEL");
            return false;
        }
        if (latent_upscale_enabled && strlen(SAFE_STR(request.hires.model_path)) == 0) {
            LOG_ERROR("LTX latent spatial upscale is enabled but hires model path was not provided");
            return false;
        }
    }

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_vid_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count);
    sd_ctx->sd->reset_generation_extensions();

    SamplePlan plan(sd_ctx, sd_vid_gen_params, request);
    auto latent_inputs_opt = prepare_video_generation_latents(sd_ctx, sd_vid_gen_params, &request);
    if (!latent_inputs_opt.has_value()) {
        return false;
    }
    ImageGenerationLatents latents = std::move(*latent_inputs_opt);

    // GENERIC V2V (SDEdit): the source clip is seeded into init_latent's video channels; truncate
    // the sigma schedule by `strength` so sampling starts from a partial sigma. img2img convention
    // (mirrors prepare_image_generation_latents): t_enc = sample_steps*strength; keep the tail of
    // the schedule. strength >= 1.0 leaves the full schedule (start==0 → source ignored, a fresh
    // render). NULL v2v = v2v_sdedit false → untouched.
    if (latents.v2v_sdedit) {
        // v2v (SDEdit restyle=mode 1 / edit=mode 2) truncates by the guide-strength slider, NOT the
        // image-pin `strength` (which defaults to 1.0 = no truncation = full re-render). koblem sends
        // the slider as v2v_guide_strength for every v2v shot; both modes share this schedule slice.
        float v2v_strength = sd_vid_gen_params->strength;
        if (sd_vid_gen_params->v2v_mode >= 1 && sd_vid_gen_params->v2v_guide_strength > 0.f) {
            v2v_strength = sd_vid_gen_params->v2v_guide_strength;
        }
        float strength = std::clamp(v2v_strength, 0.f, 1.f);
        if (strength < 1.f && plan.sample_steps > 0 &&
            (int)plan.sigmas.size() == plan.sample_steps + 1) {
            int t_enc = (int)(plan.sample_steps * strength);
            if (t_enc >= plan.sample_steps) {
                t_enc = plan.sample_steps - 1;
            }
            if (t_enc < 0) {
                t_enc = 0;
            }
            int start = plan.sample_steps - t_enc - 1;
            if (start > 0 && start < (int)plan.sigmas.size()) {
                std::vector<float> sched(plan.sigmas.begin() + start, plan.sigmas.end());
                plan.sigmas       = std::move(sched);
                plan.sample_steps = (int)plan.sigmas.size() - 1;
                LOG_INFO("LTXAV V2V: SDEdit strength=%.2f -> %d steps (t_enc=%d), sigma[0]=%.4f",
                         strength, plan.sample_steps, t_enc, plan.sigmas.front());
            }
        }
    }

    ImageGenerationEmbeds embeds = prepare_video_generation_embeds(sd_ctx,
                                                                   sd_vid_gen_params,
                                                                   request,
                                                                   latents);
    if (latent_upscale_enabled) {
        LOG_INFO("generate_video %dx%dx%d -> LTX latent spatial upscale",
                 request.width,
                 request.height,
                 request.frames);
    } else if (latent_refine_enabled) {
        LOG_INFO("generate_video %dx%dx%d -> LTX same-resolution latent refine",
                 request.width,
                 request.height,
                 request.frames);
    } else {
        LOG_INFO("generate_video %dx%dx%d",
                 request.width,
                 request.height,
                 request.frames);
    }

    // Lever 3 (WAN_VAE_FREE_DURING_DIT): prepare_video_generation_embeds above finished all
    // VAE encoding (ref/control/init latents); the VAE is not needed again until decode. Free
    // its ~254MB params now so they don't sit resident through the VRAM-peak DiT sample loop
    // (both the high-noise and main passes). reload_first_stage_model() restores them right
    // before decode_video_outputs. Gated on the captured reload state (present only when the
    // lever env is set) + one-shot CLI (!keep_diffusion_model_resident).
    if (sd_ctx->sd->free_params_immediately &&
        !sd_ctx->sd->keep_diffusion_model_resident &&
        sd_ctx->sd->vae_reload_loader && !sd_ctx->sd->vae_reload_tensors.empty() &&
        sd_ctx->sd->first_stage_model) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
        LOG_INFO("WAN_VAE_FREE_DURING_DIT: freed VAE params for the DiT sample loop");
    }

    // LTXAV_VAE_LAZY: on an LTXAV render the video VAE (~1385 MB) + audio VAE (~353 MB) sit on
    // the GPU but are UNUSED between here and the final decode — they squat ~1.7 GB through the
    // DiT sample AND the (latent-upscale) refine, whose reserve is the binding [VRAM] peak
    // (~11.9 GB). All VAE ENCODING (ref/control/init latents) finished inside the
    // prepare_video_generation_* calls above, so release both VAEs' GPU VRAM now for the whole
    // sample+refine and re-establish it for decode. Two residency regimes, handled separately:
    //
    //  (A) --offload-to-cpu (our prod recipe, incl. --mmap): the params' home lives on the CPU
    //      params_backend and is UPLOADED to a GPU runtime buffer per-compute (the log line
    //      "ltx_video_vae offload params (1385 MB) to runtime backend (CUDA0)"). That GPU copy is
    //      what squats. set_keep_params_resident(false) -> restore_all_params() frees the GPU copy
    //      while the CPU/mmap-backed home survives; the decode's execute_graph re-runs
    //      offload_all_params() to re-upload from that home (the SAME mechanism that loaded it).
    //      No reload loader needed and mmap-safe — the reload path is proven-present (params_buffer
    //      still allocated + execute_graph always re-offloads before compute).
    //
    //  (B) params directly on the GPU (no offload): restore_all_params() is a no-op, so free the
    //      buffers and reload from disk before decode via resident_reload_loader (captured for
    //      every LTXAV no-mmap ctx, :1335). reload_first_stage_model() (:9128, unconditional) and
    //      reload_audio_vae_model() (:9027, audio-decode block) re-materialize them (~0.1s each).
    //      Gated on that loader's presence so the free never happens without a proven reload path;
    //      absent (e.g. --mmap WITHOUT offload = params file-backed on GPU) it is skipped.
    //
    // Two-stage relip also needs the VAE later, but its stage-2 reference encode is itself an
    // ordinary offloaded compute that re-materializes the GPU copy on demand. Releasing it here
    // therefore saves the VAE's shared-resident payload through the entire stage-1 sample without
    // changing the reference; it is re-offloaded just before that stage-2 encode. Opt-in, default
    // off, so no measured baseline shifts.
    static const bool ltxav_vae_lazy = [] {
        const char* s = getenv("LTXAV_VAE_LAZY");
        return s && s[0] == '1';
    }();
    if (ltxav_vae_lazy &&
        sd_version_is_ltxav(sd_ctx->sd->version) &&
        sd_ctx->sd->first_stage_model) {
        int64_t phase_t0 = ggml_time_ms();
        auto& vvae = sd_ctx->sd->first_stage_model;
        auto& avae = sd_ctx->sd->audio_vae_model;
        if (vvae->params_offloaded_to_host()) {
            // Regime (A): release the offloaded GPU copies; CPU/mmap home survives, decode
            // re-offloads. Harmless (no-op) on a VAE that isn't currently offload-resident.
            //
            // FIX: set_keep_params_resident(false) alone does NOT free the VAE's GPU squat under the
            // prod recipe (LONGCAT_SHARED_RESIDENT=1). It calls only restore_all_params(), which frees
            // runtime_params_buffer — but the VAE's ~1385 MB lives in resident_runtime_params_buffer
            // (offload_resident_params(), the cross-step shared payload that is deliberately kept pinned
            // across the tiled-decode segments and warm renders). Only restore_resident_params() frees
            // that. release_all_gpu_param_residency() funnels resident + partial + runtime buffers
            // through their real freers (-> cudaFree) and drops the cross-step token/cache so the decode
            // re-offloads from the CPU/mmap home. (This is why the earlier restore_all_params()+pool-trim
            // left VAE_gpu at 1385 — the trim can't reclaim a still-referenced resident buffer.)
            vvae->release_all_gpu_param_residency();
            if (avae) {
                avae->release_all_gpu_param_residency();
            }
            // Trim the VAE backend's pool so the freed VRAM actually leaves the board and becomes
            // real headroom for the sample/refine. Mirrors the pre-sample DIFFUSION-pool trim (:~8957).
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::VAE));
            // The VAE params just moved (re-offloaded to host); their device pointers are now
            // stale, so the cuDNN conv3d weight-reorder cache (raw cudaMalloc, keyed by weight
            // ptr) holds orphaned buffers from the prior segment. Free them now or they leak
            // ~1.4 GB/segment on a continuation. Zero perf cost (they re-reorder at decode anyway).
            ggml_backend_cuda_release_cudnn_conv3d_weights();
            LOG_INFO("LTXAV_VAE_LAZY: released offloaded video+audio VAE GPU params (runtime + shared-resident) + trimmed VAE pool + freed conv3d reorder weights before DiT sample+refine; re-offload from host at decode");
        } else if (sd_ctx->sd->resident_reload_loader) {
            // Regime (B): free the GPU-resident buffers; reload from disk before decode.
            size_t freed = vvae->get_params_buffer_size();
            vvae->free_params_buffer();
            if (avae) {
                freed += avae->get_params_buffer_size();
                avae->free_params_buffer();
            }
            LOG_INFO("LTXAV_VAE_LAZY: freed resident video+audio VAE params (~%.0f MB) before DiT sample+refine; reload from disk at decode",
                     freed / (1024.f * 1024.f));
        } else {
            LOG_INFO("LTXAV_VAE_LAZY: VAE params are neither host-offloaded nor reloadable (e.g. --mmap without --offload-to-cpu); skipping to stay abort-safe");
        }
        LOG_INFO("[LTX_PHASE] pre-sample VAE eviction/setup took %.3fs", (ggml_time_ms() - phase_t0) * 1.0f / 1000);
    }

    int64_t latent_start = ggml_time_ms();
    int W                = request.width / request.vae_scale_factor;
    int H                = request.height / request.vae_scale_factor;
    int T                = static_cast<int>(latents.init_latent.shape()[2]);

    sd::Tensor<float> x_t   = latents.init_latent;
    sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);

    // LongCat-Avatar audio path: wav -> log-mel -> whisper encoder -> windowed
    // AudioProjModel inputs, set on the avatar runner so every DiT block's audio
    // cross-attn drives lip-sync from the speech. No-op (silent video) if absent.
    if (sd_version_is_longcat_avatar(sd_ctx->sd->version)) {
        auto avatar_model = std::dynamic_pointer_cast<LongCatAvatarModel>(sd_ctx->sd->diffusion_model);
        if (avatar_model) {
            avatar_model->avatar.audio_first  = sd::Tensor<float>();
            avatar_model->avatar.audio_latter = sd::Tensor<float>();
            const char* apath                 = SAFE_STR(sd_vid_gen_params->audio_path);
            if (strlen(apath) > 0 && sd_ctx->sd->whisper_encoder_model) {
                std::vector<float> wav;
                if (LONGCAT_AUDIO::load_wav_16k_mono(apath, wav)) {
                    avatar_input_wav = wav;  // keep a copy to mux into the output
                    // T_video = generated video frames. The audio windowing maps
                    // T_video -> T latent frames (vae_scale=4). avatar-v1.5 uses a
                    // FIXED save_fps=25 / audio_stride=1 for the audio<->frame
                    // alignment (model constants), independent of the output --fps.
                    int T_video = request.frames;
                    int fps     = 25;

                    LONGCAT_AUDIO::WhisperMel mel_fe;
                    int n_mel_frames = 0;
                    std::vector<float> logmel = mel_fe.log_mel(wav, n_mel_frames);
                    LOG_INFO("audio: log-mel %d frames (%zu samples)", n_mel_frames, wav.size());

                    if (n_mel_frames > 0) {
                        // mel tensor [T_mel, n_mels, 1] (ggml-ne: dim0=T_mel). logmel
                        // is laid out [mel][frame] (mel outer), so transpose to [frame][mel].
                        const int n_mels = LONGCAT_AUDIO::WhisperMel::kNMels;
                        sd::Tensor<float> mel_t({(int64_t)n_mel_frames, (int64_t)n_mels, 1});
                        float* md = mel_t.data();
                        for (int m = 0; m < n_mels; m++) {
                            for (int t = 0; t < n_mel_frames; t++) {
                                md[(size_t)m * n_mel_frames + t] = logmel[(size_t)m * n_mel_frames + t];
                            }
                        }

                        sd_ctx->sd->whisper_encoder_model->set_flash_attention_enabled(false);
                        sd::Tensor<float> whisper_hs = sd_ctx->sd->whisper_encoder_model->compute(sd_ctx->sd->n_threads, mel_t);  // [1280, T_enc, 33]
                        sd_ctx->sd->whisper_encoder_model->free_compute_buffer();
                        // Keep whisper params resident across chained segments (each
                        // segment re-windows its own audio slice); otherwise free them.
                        if (!sd_ctx->sd->keep_diffusion_model_resident) {
                            sd_ctx->sd->whisper_encoder_model->free_params_buffer();
                        }
                        LOG_INFO("audio: whisper encoder out [%lld, %lld, %lld]",
                                 (long long)whisper_hs.shape()[0], (long long)whisper_hs.shape()[1], (long long)whisper_hs.shape()[2]);

                        LONGCAT_AUDIO::AudioWindowConfig acfg;
                        acfg.fps = static_cast<float>(fps);
                        // video_length = int(audio_duration * fps) (reference). The
                        // whisper features interpolate to that length; the ±2 window
                        // then indexes the first T_video of them (clamped at the ends).
                        double audio_duration = (double)wav.size() / 16000.0;
                        int video_length      = std::max(1, (int)(audio_duration * fps));
                        std::vector<float> full = LONGCAT_AUDIO::build_full_audio_emb(whisper_hs, video_length, acfg);
                        LOG_INFO("audio: duration %.2fs -> video_length %d (fps %d), windowing first %d frames",
                                 audio_duration, video_length, fps, T_video);
                        sd::Tensor<float> first, latter;
                        int audio_off = sd_vid_gen_params->audio_frame_offset;
                        if (audio_off != 0) {
                            LOG_INFO("audio: continuation frame_offset %d (global timeline)", audio_off);
                        }
                        int N_t_audio = LONGCAT_AUDIO::build_proj_inputs(full, T_video, acfg, first, latter, audio_off);
                        LOG_INFO("audio: window inputs first[%lld,%lld] latter[%lld,%lld] N_t=%d (latent T=%d)",
                                 (long long)first.shape()[0], (long long)first.shape()[1],
                                 (long long)latter.shape()[0], (long long)latter.shape()[1], N_t_audio, T);
                        avatar_model->avatar.audio_first  = std::move(first);
                        avatar_model->avatar.audio_latter = std::move(latter);
                    }
                }
            }
        }
    }

    if (plan.high_noise_sample_steps > 0) {
        LOG_DEBUG("sample(high noise) %dx%dx%d", W, H, T);

        int64_t sampling_start = ggml_time_ms();
        std::vector<float> high_noise_sigmas(plan.sigmas.begin(), plan.sigmas.begin() + plan.high_noise_sample_steps + 1);
        plan.sigmas = std::vector<float>(plan.sigmas.begin() + plan.high_noise_sample_steps, plan.sigmas.end());

        sd::Tensor<float> x_t_sampled = sd_ctx->sd->sample(sd_ctx->sd->high_noise_diffusion_model,
                                                           false,
                                                           x_t,
                                                           std::move(noise),
                                                           embeds.cond,
                                                           request.use_high_noise_uncond ? embeds.uncond : SDCondition(),
                                                           embeds.img_uncond,
                                                           sd::Tensor<float>(),
                                                           0.f,
                                                           request.high_noise_guidance,
                                                           plan.high_noise_eta,
                                                           request.shifted_timestep,
                                                           plan.high_noise_sample_method,
                                                           sd_ctx->sd->is_flow_denoiser(),
                                                           plan.high_noise_extra_sample_args,
                                                           high_noise_sigmas,
                                                           std::vector<sd::Tensor<float>>{},
                                                           false,
                                                           latents.denoise_mask,
                                                           latents.vace_context,
                                                           request.vace_strength,
                                                           latents.audio_length,
                                                           static_cast<float>(request.fps),
                                                           request.cache_params,
                                                           latents.video_positions);
        int64_t sampling_end          = ggml_time_ms();
        if (x_t_sampled.empty()) {
            LOG_ERROR("sampling(high noise) failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            // On a warm resident worker (keep_diffusion_model_resident, the Wan2.2 dual-MoE chain)
            // the high-noise expert must SURVIVE across renders exactly like the low-noise model:
            // there is no per-render DiT reload, and free_params_buffer() nulls the tensor pointers,
            // so freeing it here would fault the NEXT render's high-noise phase (GGML_ASSERT(buffer)
            // null). Mirror the low-noise guard below (free only when not resident).
            if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
                sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
            }
            return false;
        }

        x_t   = std::move(x_t_sampled);
        noise = {};
        LOG_INFO("sampling(high noise) completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        // Resident warm worker: keep the high-noise expert's params buffer alive across renders
        // (mirrors the low-noise diffusion_model's resident-guarded free after the main sample).
        // The per-window GPU residency is reclaimed by release_chain_segment_gpu_residency(); the
        // host params copy stays so the next window/render re-offloads without a reload.
        if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
            sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
        }
    }

    // LTXAV relip VRAM (pre-sample pool trim, default on): the phases that ran just before
    // the DiT sample loop — the relip reference VAE encode (a multi-GB tiled encode compute
    // buffer) and, on the two-stage path, the latent upscaler (~950 MB params + its compute)
    // — reserve large scratch blocks in the ggml CUDA VMM pool. ggml frees those blocks but
    // the pool RETAINS them as committed high-water, so they co-reside with the DiT forward's
    // own resident+partial weights and inflate the sampling-peak driver_used by ~1.7 GB
    // (two-stage stage-2) to ~4 GB. Return the pool high-water to the OS here, before the
    // forward reserves its (smaller) compute buffer, so the sampling peak reflects only the
    // genuine DiT working set. Nothing is in flight at this point (encode/upscale done, no DiT
    // compute buffer live yet) and the offloaded DiT params live in real backend buffers, not
    // the pool — so this only drops freed scratch. The pool rebuilds lazily on the next alloc.
    // All SD modules resolve to the one CUDA device backend, so trimming DIFFUSION trims the
    // shared device pool (VAE + upscaler included). Opt out with LTXAV_PRE_SAMPLE_POOL_TRIM=0.
    if (sd_version_is_ltxav(sd_ctx->sd->version)) {
        const char* e = getenv("LTXAV_PRE_SAMPLE_POOL_TRIM");
        if (e == nullptr || std::string(e) != "0") {
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
        }
    }

    LOG_DEBUG("sample %dx%dx%d", W, H, T);
    int64_t sampling_start = ggml_time_ms();
    sd::Tensor<float> final_latent;
    // Relip's stage-1 is a short (8-step) pass but otherwise pins the same 5.4 GB
    // cross-segment DiT set as a long continuation. Allow the existing hot-first
    // resident cap to apply to it too: the separable reference has already reduced
    // its context, so streaming the cold tail is a useful final VRAM lever. Both
    // variables are required; every normal render and the locked default are inert.
    const bool relip_base_resident_cap = latents.video_conditioning_frame_count > 0 &&
                                         getenv("LTXAV_RELIP_BASE_RESIDENT_CAP") != nullptr &&
                                         std::string(getenv("LTXAV_RELIP_BASE_RESIDENT_CAP")) != "0" &&
                                         getenv("LONGCAT_SHARED_RESIDENT_MAX_MB") != nullptr;
    if (relip_base_resident_cap && sd_ctx->sd->diffusion_model) {
        sd_ctx->sd->diffusion_model->set_refine_resident_scope(true);
        LOG_INFO("LTXAV relip: applying LONGCAT_SHARED_RESIDENT_MAX_MB to stage-1 sample");
    }
    bool final_latent_prestripped = false;
    // LTX latent-reuse harness: skip the (expensive) DiT sampling and load a banked
    // latent so the VAE-tiling quality ladder re-decodes the SAME latent under
    // different tiling. The text encode above still runs (~cheap); only sampling is
    // skipped. Bank one with LTX_SAVE_LATENTS first (see decode_video_outputs).
    // VACE_DECODE_LATENT is the VACE-side alias: it loads a latent banked by
    // VACE_SAVE_LATENT (which is ALREADY ref-stripped/post-decode shape) so we also
    // skip the post-sampling strips below — lets the VAE-tiling sweep re-decode the
    // SAME latent at ~33s/run (encode+T5+decode) instead of ~111s (full DiT).
    const char* vace_decode_path = getenv("VACE_DECODE_LATENT");
    const char* ltx_load_path    = getenv("LTX_LOAD_LATENTS");
    const char* load_path        = (vace_decode_path && vace_decode_path[0]) ? vace_decode_path : ltx_load_path;
    if (load_path != nullptr && load_path[0] != '\0') {
        try {
            final_latent = sd::load_tensor_from_file_as_tensor<float>(load_path);
            final_latent_prestripped = (vace_decode_path && vace_decode_path[0]);
            LOG_INFO("%s: loaded cached latent (%dx%dx%dx%d) from %s, SKIPPING DiT sampling",
                     final_latent_prestripped ? "VACE_DECODE_LATENT" : "LTX_LOAD_LATENTS",
                     (int)final_latent.shape()[0], (int)final_latent.shape()[1],
                     (int)final_latent.shape()[2], (int)(final_latent.dim() > 3 ? final_latent.shape()[3] : 1),
                     load_path);
        } catch (const std::exception& e) {
            LOG_ERROR("latent load failed (%s); falling back to sampling", e.what());
        }
    }
    auto sample_base_window = [&](const sd::Tensor<float>& window_latent,
                                  sd::Tensor<float> window_noise,
                                  const sd::Tensor<float>& window_mask,
                                  int window_audio_length,
                                  const sd::Tensor<float>& window_video_positions,
                                  const sd::Tensor<float>& window_audio_positions,
                                  const sd::Tensor<float>& window_video_reference) {
        return sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                  true,
                                  window_latent,
                                  std::move(window_noise),
                                  embeds.cond,
                                  request.use_uncond ? embeds.uncond : SDCondition(),
                                  embeds.img_uncond,
                                  sd::Tensor<float>(),
                                  0.f,
                                  sd_vid_gen_params->sample_params.guidance,
                                  plan.eta,
                                  sd_vid_gen_params->sample_params.shifted_timestep,
                                  plan.sample_method,
                                  sd_ctx->sd->is_flow_denoiser(),
                                  plan.extra_sample_args,
                                  plan.sigmas,
                                  std::vector<sd::Tensor<float>>{},
                                  false,
                                  window_mask,
                                  latents.vace_context,
                                  request.vace_strength,
                                  window_audio_length,
                                  static_cast<float>(request.fps),
                                  request.cache_params,
                                  window_video_positions,
                                  window_audio_positions,
                                  latents.audio_fixed,
                                  window_video_reference,
                                  // The ver3 workflow wires SamplerCustomAdvanced's
                                  // denoised_output, not its noisy LCM output, into
                                  // the latent upsampler.  This also works per tile.
                                  hires_continue_mode);
    };
    if (final_latent.empty()) {
        const char* base_window_env = std::getenv("LTX_BASE_TEMPORAL_WINDOW");
        // A char-reference block (latents.video_reference) and/or a PAST-ANCHOR continuation/keyframe
        // guide (init_latent tail frames, => latents.video_positions non-empty) can now COMPOSE with
        // base temporal windowing: each is re-attached to every tile below (bounded token cost).
        // Previously any such conditioning DISABLED windowing and forced a single full-length base
        // pass whose VRAM grows with segment length. audio_fixed is no longer a windowing PRECONDITION
        // for the reference path — it only selects the audio denoise-mask value — so a char-ref t2v
        // render (audio_fixed=false) can window too.
        //
        // The reference path re-maps positions by EXTRACTING the guide/char rows VERBATIM from
        // latents.video_positions (never rebuilding them), so it is agnostic to the guide's RoPE
        // convention (past-anchor continuation, keyframe indices, mixed Director guides). It is still
        // gated to NON-v2v renders (control_frames_size==0 && v2v_mode==0): that excludes relip-lipdub
        // (timeline-aligned reference; note its DEFAULT direct-audio mode does NOT set audio_positions,
        // so audio_positions.empty() alone would not exclude it), SDEdit (v2v_mode==1) and guide-edit
        // (v2v_mode==2), all of which keep the historical full pass byte-for-byte. vace_context and a
        // separate audio_positions block (reference-audio lipdub) also remain hard-disabled.
        const bool base_window_ref =
            (!latents.video_positions.empty() || !latents.video_reference.empty()) &&
            !latents.video_positions.empty() && sd_vid_gen_params->control_frames_size == 0 &&
            sd_vid_gen_params->v2v_mode == 0;
        const bool base_window_legacy_audio =
            latents.audio_fixed && latents.video_positions.empty() && latents.video_reference.empty();
        const bool base_temporal_windowing = base_window_env != nullptr && base_window_env[0] != '\0' &&
                                             std::string(base_window_env) != "0" &&
                                             sd_version_is_ltxav(sd_ctx->sd->version) &&
                                             (base_window_legacy_audio || base_window_ref) &&
                                             latents.audio_positions.empty() && latents.vace_context.empty() &&
                                             (x_t.dim() == 4 || (x_t.dim() == 5 && x_t.shape()[4] == 1)) &&
                                             x_t.shape()[2] > 1;
        if (!base_temporal_windowing) {
            final_latent = sample_base_window(x_t,
                                               std::move(noise),
                                               latents.denoise_mask,
                                               latents.audio_length,
                                               latents.video_positions,
                                               latents.audio_positions,
                                               latents.video_reference);
        } else {
            int temporal_window = std::max(2, std::atoi(base_window_env));
            int temporal_overlap = 4;
            if (const char* overlap_env = std::getenv("LTX_BASE_TEMPORAL_OVERLAP"); overlap_env != nullptr) {
                temporal_overlap = std::max(1, std::atoi(overlap_env));
            }
            const int64_t total_frames = x_t.shape()[2];
            const int64_t latent_channels = sd_ctx->sd->get_latent_channel();
            // A continuation/keyframe guide lives as extra FROZEN frames at the TAIL of the video
            // grid (init_latent); only the leading target frames are windowed. A char-ref block is
            // separate (latents.video_reference) and never in the grid.
            const int64_t base_guide_frames  = base_window_ref ? latents.video_conditioning_frame_count : 0;
            const int64_t base_target_frames = total_frames - base_guide_frames;
            const bool base_window_has_audio = latents.audio_length > 0 && x_t.shape()[3] > latent_channels;
            const int64_t base_char_ref_tokens =
                (base_window_ref && !latents.video_reference.empty()) ? latents.video_reference.shape()[0] : 0;
            // Extraction invariant: video_positions must be exactly [main grid (total_frames) ++ char].
            // If a caller ever lays it out differently, fall back to the full pass rather than corrupt
            // RoPE — the windowed reference path relies on this contiguity.
            const int64_t base_expected_pos_rows =
                x_t.shape()[0] * x_t.shape()[1] * total_frames + base_char_ref_tokens;
            const bool base_window_ref_layout_ok =
                !base_window_ref ||
                (!latents.video_positions.empty() &&
                 latents.video_positions.shape()[2] == base_expected_pos_rows);
            temporal_window =
                std::clamp(temporal_window, 2, static_cast<int>(std::max<int64_t>(2, base_target_frames)));
            temporal_overlap = std::clamp(temporal_overlap, 1, temporal_window - 1);
            const int64_t stride = temporal_window - temporal_overlap;
            if (base_window_ref && (base_target_frames <= temporal_window || !base_window_ref_layout_ok)) {
                // Single tile (or unexpected layout): identical to the historical full base pass.
                if (!base_window_ref_layout_ok) {
                    LOG_WARN("LTX base temporal-window: unexpected video_positions layout (%lld rows, expected "
                             "%lld); using full base pass",
                             (long long)latents.video_positions.shape()[2], (long long)base_expected_pos_rows);
                }
                final_latent = sample_base_window(x_t,
                                                  std::move(noise),
                                                  latents.denoise_mask,
                                                  latents.audio_length,
                                                  latents.video_positions,
                                                  latents.audio_positions,
                                                  latents.video_reference);
            } else if (!base_window_has_audio && !base_window_ref) {
                LOG_ERROR("LTX base temporal-window requires packed fixed driving audio");
            } else {
                auto full_video = sd::ops::slice(x_t, 3, 0, latent_channels);
                auto full_noise = sd::ops::slice(noise, 3, 0, latent_channels);
                const int64_t Wl = full_video.shape()[0];
                const int64_t Hl = full_video.shape()[1];
                const int64_t video_ch = full_video.shape()[3];
                // Keep I2V/keyframe locks from the packed stage-one mask. The
                // per-window audio mask is rebuilt for each sliced audio span;
                // only its leading video channels belong to this timeline.
                sd::Tensor<float> full_video_denoise_mask;
                if (!latents.denoise_mask.empty()) {
                    full_video_denoise_mask = sd::ops::slice(latents.denoise_mask, 3, 0, latent_channels);
                }
                // The in-grid continuation/keyframe guide (init_latent tail, base_guide_frames) becomes
                // a separate frozen reference GRID re-attached to EVERY tile; the flattened char-ref
                // block keeps its own token rows. Positions are NOT rebuilt — the guide+char position
                // rows are EXTRACTED VERBATIM from latents.video_positions (everything past the target
                // block), so their exact RoPE convention (past-anchor / keyframe / mixed) is preserved
                // and late tiles keep continuing the prior motion. Both blocks are bounded (guide K +
                // char tokens), so per-window token count stays ~one window regardless of segment
                // length — restoring the VRAM budget for char-ref/continuation renders.
                sd::Tensor<float> guide_ref_grid;
                if (base_guide_frames > 0) {
                    guide_ref_grid = sd::ops::slice(full_video, 2, base_target_frames, total_frames);
                }
                const sd::Tensor<float>& char_ref_block = latents.video_reference;
                const int64_t char_ref_tokens = char_ref_block.empty() ? 0 : char_ref_block.shape()[0];
                // Guide (K frames) + char rows = everything after the windowed target block. Reused
                // unchanged for every tile (the reference sits at fixed absolute positions).
                sd::Tensor<float> reference_positions;
                if (!latents.video_positions.empty()) {
                    reference_positions = sd::ops::slice(latents.video_positions, 2, Wl * Hl * base_target_frames,
                                                         latents.video_positions.shape()[2]);
                }
                sd::Tensor<float> full_audio;
                if (base_window_has_audio) {
                    full_audio =
                        unpack_ltxav_audio_latent(x_t, latents.audio_length, static_cast<int>(latent_channels));
                }
                if (base_window_has_audio && full_audio.empty()) {
                    LOG_ERROR("LTX base temporal-window could not unpack driving audio latent");
                } else {
                    sd::Tensor<float> video_result(full_video.shape());
                    video_result.fill_(0.0f);
                    const int64_t plane = Wl * Hl;
                    const int64_t audio_rate = 25;  // LTXAV audio latent tokens per second.
                    // audio_fixed => hold driving audio (mask 0); otherwise generate audio (mask 1).
                    const float audio_mask_value = latents.audio_fixed ? 0.0f : 1.0f;
                    int tile_index = 0;
                    int64_t produced_end = 0;
                    for (int64_t start = 0;; start += stride, ++tile_index) {
                        // The short final tail otherwise gets a different, much
                        // larger graph-cut weight layout. Align it to a normal
                        // window and use the already-sampled latent history as
                        // additional frozen context; only unseen tail frames are
                        // emitted below.
                        const int64_t tile_start = start + temporal_window >= base_target_frames
                                                       ? std::max<int64_t>(0, base_target_frames - temporal_window)
                                                       : start;
                        const int64_t end = std::min<int64_t>(base_target_frames, tile_start + temporal_window);
                        const int64_t length = end - tile_start;
                        const int64_t frozen = tile_index == 0
                                                   ? 0
                                                   : std::min<int64_t>(length, std::max<int64_t>(0, produced_end - tile_start));
                        auto video_tile = sd::ops::slice(full_video, 2, tile_start, end);
                        auto noise_tile = sd::ops::slice(full_noise, 2, tile_start, end);
                        if (frozen > 0) {
                            float* tile_data = video_tile.data();
                            const float* result_data = video_result.data();
                            for (int64_t channel = 0; channel < latent_channels; ++channel) {
                                for (int64_t local = 0; local < frozen; ++local) {
                                    std::memcpy(tile_data + plane * (local + length * channel),
                                                result_data + plane * (tile_start + local + total_frames * channel),
                                                static_cast<size_t>(plane) * sizeof(float));
                                }
                            }
                        }
                        // Each latent step advances eight display frames after the initial frame.
                        // Rebase both modalities to the local window while retaining the exact
                        // audio time slice that corresponds to these video frames.
                        int64_t audio_start = 0;
                        int64_t audio_end   = 0;
                        sd::Tensor<float> audio_tile;
                        if (base_window_has_audio) {
                            const int64_t pixel_start = tile_start * 8;
                            const int64_t pixel_end = std::min<int64_t>(request.frames, (end - 1) * 8 + 1);
                            audio_start = std::clamp<int64_t>(
                                (pixel_start * audio_rate) / request.fps, 0, full_audio.shape()[1]);
                            audio_end = std::clamp<int64_t>(
                                (pixel_end * audio_rate + request.fps - 1) / request.fps,
                                audio_start + 1,
                                full_audio.shape()[1]);
                            audio_tile = sd::ops::slice(full_audio, 1, audio_start, audio_end);
                        }
                        // Preserve source/keyframe mask values instead of
                        // recreating a fully denoisable tile, then freeze the
                        // carried temporal overlap without changing audio.
                        auto video_mask = full_video_denoise_mask.empty()
                                              ? make_ltxav_video_denoise_mask(video_tile, 1.0f)
                                              : sd::ops::slice(full_video_denoise_mask, 2, tile_start, end);
                        if (frozen > 0) {
                            float* mask_data = video_mask.data();
                            const int64_t mask_channels = video_mask.shape()[3];
                            for (int64_t channel = 0; channel < mask_channels; ++channel) {
                                for (int64_t local = 0; local < frozen; ++local) {
                                    std::fill_n(mask_data + plane * (local + length * channel), plane, 0.0f);
                                }
                            }
                        }
                        sd::Tensor<float> latent_tile = base_window_has_audio
                                                            ? pack_ltxav_audio_and_video_latents(video_tile, audio_tile)
                                                            : video_tile;
                        sd::Tensor<float> packed_noise = base_window_has_audio
                                                             ? pack_ltxav_audio_and_video_latents(noise_tile, audio_tile)
                                                             : noise_tile;
                        sd::Tensor<float> mask_tile =
                            base_window_has_audio
                                ? pack_ltxav_audio_and_video_denoise_mask(video_mask, video_tile, audio_tile,
                                                                          audio_mask_value)
                                : video_mask;
                        // Per-tile RoPE rows: the windowed target rows sliced VERBATIM from the full
                        // video_positions (byte-identical to the full pass for these absolute frames),
                        // then the extracted guide+char rows unchanged — matching the reference token
                        // block assembled below token-for-token. No-reference tiles build target rows
                        // directly (byte-identical to the historical audio-driven path).
                        sd::Tensor<float> video_positions;
                        if (base_window_ref) {
                            auto target_positions = sd::ops::slice(latents.video_positions, 2,
                                                                   Wl * Hl * tile_start, Wl * Hl * end);
                            video_positions = reference_positions.empty()
                                                  ? target_positions
                                                  : sd::ops::concat(target_positions, reference_positions, 2);
                        } else {
                            video_positions = build_ltxav_window_video_positions(
                                Wl, Hl, tile_start, length, request.fps, request.vae_scale_factor);
                        }
                        // Reference token block: a grid guide alone; the flattened char-ref alone; or
                        // (both) the guide flattened + char-ref concatenated — matching the position
                        // row order [target ++ guide ++ char] above.
                        sd::Tensor<float> tile_reference;
                        if (base_guide_frames > 0 && char_ref_tokens > 0) {
                            auto guide_flat =
                                guide_ref_grid.reshape({Wl * Hl * base_guide_frames, 1, 1, video_ch, 1});
                            tile_reference = sd::ops::concat(guide_flat, char_ref_block, 0);
                        } else if (base_guide_frames > 0) {
                            tile_reference = guide_ref_grid;
                        } else if (char_ref_tokens > 0) {
                            tile_reference = char_ref_block;
                        }
                        sd::Tensor<float> audio_positions;
                        int tile_audio_length = 0;
                        if (base_window_has_audio) {
                            audio_positions = build_ltxav_window_audio_positions(audio_start, audio_tile.shape()[1]);
                            tile_audio_length = static_cast<int>(audio_tile.shape()[1]);
                        }
                        LOG_INFO("LTX base temporal-window tile %d: latent [%lld,%lld), frozen-overlap=%lld, "
                                 "audio [%lld,%lld), guide=%lld charref=%lld",
                                 tile_index, (long long)tile_start, (long long)end, (long long)frozen,
                                 (long long)audio_start, (long long)audio_end,
                                 (long long)base_guide_frames, (long long)char_ref_tokens);
                        auto tile = sample_base_window(latent_tile,
                                                       std::move(packed_noise),
                                                       mask_tile,
                                                       tile_audio_length,
                                                       video_positions,
                                                       audio_positions,
                                                       tile_reference);
                        // Each window may materialize a different graph-cut/shared-resident DiT
                        // set. Keeping that set across the next audio window accumulates several
                        // GiB of dead weights even though the host-offloaded model can re-stream
                        // them exactly. Release it, then trim the graph scratch before the next
                        // window. This is allocation-only and preserves the sampled latent.
                        if (sd_ctx->sd->diffusion_model->params_offloaded_to_host()) {
                            sd_ctx->sd->diffusion_model->release_all_gpu_param_residency();
                            sd_ctx->sd->diffusion_model->free_compute_buffer();
                        }
                        ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
                        if (tile.empty()) {
                            final_latent = {};
                            break;
                        }
                        auto refined_video = sd::ops::slice(tile, 3, 0, latent_channels);
                        const float* src = refined_video.data();
                        float* dst = video_result.data();
                        for (int64_t local = frozen; local < length; ++local) {
                            for (int64_t channel = 0; channel < latent_channels; ++channel) {
                                std::memcpy(dst + plane * (tile_start + local + total_frames * channel),
                                            src + plane * (local + length * channel),
                                            static_cast<size_t>(plane) * sizeof(float));
                            }
                        }
                        produced_end = std::max(produced_end, end);
                        if (end == base_target_frames) {
                            // Carry the frozen in-grid guide frames through unchanged so the emitted
                            // latent keeps its historical [target + guide] shape for the downstream
                            // crop keyed on video_conditioning_frame_count.
                            if (base_guide_frames > 0) {
                                const float* g = full_video.data();
                                float* rd = video_result.data();
                                for (int64_t channel = 0; channel < latent_channels; ++channel) {
                                    for (int64_t local = 0; local < base_guide_frames; ++local) {
                                        const int64_t f = base_target_frames + local;
                                        std::memcpy(rd + plane * (f + total_frames * channel),
                                                    g + plane * (f + total_frames * channel),
                                                    static_cast<size_t>(plane) * sizeof(float));
                                    }
                                }
                            }
                            final_latent = base_window_has_audio
                                               ? pack_ltxav_audio_and_video_latents(video_result, full_audio)
                                               : video_result;
                            break;
                        }
                    }
                }
            }
        }
    }
    if (relip_base_resident_cap && sd_ctx->sd->diffusion_model) {
        sd_ctx->sd->diffusion_model->set_refine_resident_scope(false);
    }

    int64_t sampling_end = ggml_time_ms();
    if (final_latent.empty()) {
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        LOG_ERROR("sampling failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        return false;
    }
    LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);

    // ---- OPT-IN progressive UPSCALE-STAGE previews (emit_stages) ----
    // Only meaningful when the caller asked for stage previews AND this is a real LTX refine render
    // (base < final resolution); a no-refine render's single decode already IS the final output, so
    // no BASE preview is emitted (byte-identical). The hook fires a fast low-res BASE preview here
    // (before the first refine) and a mid-res STAGE-0 preview after the first refine on a >=2-stage
    // chain. The FINAL full-res frames are the ordinary per-segment output (on_segment / frames_out),
    // never re-decoded here.
    const bool emit_stages_on = sd_vid_gen_params->emit_stages != 0 &&
                                sd_vid_gen_params->on_stage != nullptr &&
                                sd_version_is_ltxav(sd_ctx->sd->version) &&
                                latent_refine_enabled;
    // Re-evict the video+audio VAE after a valley decode, restoring exactly the headroom the
    // following upscale/refine expects. Mirrors the LTXAV_VAE_LAZY eviction block; a no-op when the
    // lever is off (the VAE was never evicted, so it must stay resident). Idempotent under a
    // double-call (the refine's own 2nd eviction is then a no-op).
    auto reevict_vae_lazy = [&]() {
        if (!(ltxav_vae_lazy && sd_ctx->sd->first_stage_model)) {
            return;
        }
        auto& vvae = sd_ctx->sd->first_stage_model;
        auto& avae = sd_ctx->sd->audio_vae_model;
        if (vvae->params_offloaded_to_host()) {
            vvae->release_all_gpu_param_residency();
            if (avae) {
                avae->release_all_gpu_param_residency();
            }
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::VAE));
            ggml_backend_cuda_release_cudnn_conv3d_weights();
        } else if (sd_ctx->sd->resident_reload_loader) {
            vvae->free_params_buffer();
            if (avae) {
                avae->free_params_buffer();
            }
        }
    };
    // Decode an intermediate stage latent to pixels and hand the frames to on_stage. The VAE was
    // freed for the sample loop (LTXAV_VAE_LAZY prod recipe), so this runs in the VALLEY between
    // sample stages: it (1) releases the DiT GPU residency — the surrounding refine path releases it
    // at the very next stage boundary anyway (:free_dit_before_upscale / hires-chain handoff), so this
    // just moves that release a few lines earlier so DiT+VAE peaks never stack; (2) reloads the VAE;
    // (3) decodes a CLEAN video-only latent (audio channels + trailing guide frames + leading ref
    // frames stripped, mirroring the refined_latent_lo export / final decode strips); (4) dispatches
    // the frames (the callee copies them onto its own encoder thread); (5) re-evicts the VAE. All of
    // this is skipped entirely when emit_stages is off, so the default path is byte-identical.
    auto emit_stage_preview = [&](int stage_scale, const sd::Tensor<float>& raw_latent) {
        if (!emit_stages_on || raw_latent.empty()) {
            return;
        }
        int64_t stage_t0 = ggml_time_ms();
        const int64_t     video_ch = sd_ctx->sd->get_latent_channel();
        sd::Tensor<float> lat      = raw_latent;
        if (lat.dim() > 3 && lat.shape()[3] > video_ch) {
            lat = sd::ops::slice(lat, 3, 0, video_ch);
        }
        if (!final_latent_prestripped && latents.video_conditioning_frame_count > 0) {
            int64_t target_frames = latents.video_target_frame_count > 0
                                        ? latents.video_target_frame_count
                                        : lat.shape()[2] - latents.video_conditioning_frame_count;
            if (target_frames > 0 && target_frames <= lat.shape()[2]) {
                lat = sd::ops::slice(lat, 2, 0, target_frames);
            }
        }
        if (!final_latent_prestripped && latents.ref_image_num > 0 &&
            latents.ref_image_num < lat.shape()[2]) {
            lat = sd::ops::slice(lat, 2, latents.ref_image_num, lat.shape()[2]);
        }
        if (lat.empty() || lat.shape()[2] <= 0) {
            return;
        }
        // (1) release the DiT so the valley decode never stacks DiT+VAE (host-offload re-offloads at
        // the next sample; skip on GPU-resident params — no re-offload path).
        if (sd_ctx->sd->diffusion_model &&
            sd_ctx->sd->diffusion_model->params_offloaded_to_host()) {
            sd_ctx->sd->diffusion_model->release_all_gpu_param_residency();
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
        }
        // (2) bring the VAE back (no-op if resident / the LAZY lever is off).
        if (!sd_ctx->sd->reload_first_stage_model()) {
            LOG_WARN("emit_stages: VAE reload for stage-%d preview failed; skipping this preview", stage_scale);
            return;
        }
        // (3) decode the clean video-only latent to pixels.
        sd::Tensor<float> vid = sd_ctx->sd->decode_first_stage(lat, true);
        if (vid.empty()) {
            LOG_WARN("emit_stages: stage-%d preview decode failed", stage_scale);
            reevict_vae_lazy();
            return;
        }
        int n = (int)vid.shape()[2];
        int w = (int)vid.shape()[0];
        int h = (int)vid.shape()[1];
        if (n > 0) {
            sd_image_t* imgs = (sd_image_t*)calloc((size_t)n, sizeof(sd_image_t));
            if (imgs != nullptr) {
                for (int i = 0; i < n; ++i) {
                    imgs[i] = tensor_to_sd_image(vid, i);
                }
                // (4) dispatch — the callee (server) copies frames onto its background encoder.
                sd_vid_gen_params->on_stage(sd_vid_gen_params->stage_seg_index, stage_scale, w, h,
                                            imgs, n, sd_vid_gen_params->on_stage_user);
                for (int i = 0; i < n; ++i) {
                    free(imgs[i].data);
                }
                free(imgs);
            }
        }
        // (5) restore the pre-refine headroom.
        reevict_vae_lazy();
        LOG_INFO("emit_stages: seg %d stage-%d preview (%dx%d, %d frames) decoded+dispatched in %.2fs",
                 sd_vid_gen_params->stage_seg_index, stage_scale, w, h, n,
                 (ggml_time_ms() - stage_t0) * 1.0f / 1000);
    };

    // Stage-one timeline keyframes are appended guide tokens.  They must not be
    // spatially upscaled or treated as output frames: stage two re-encodes the
    // source images at its own resolution and passes those guides separately.
    // Strip the stage-one tail before the upscale, preserving packed audio as a
    // segment-global stream rather than shearing it by slicing the video axis.
    const bool timeline_keyframes = sd_vid_gen_params->keyframes != nullptr &&
                                    sd_vid_gen_params->keyframes_size > 0;
    if (latent_refine_enabled && timeline_keyframes && latents.video_conditioning_frame_count > 0 &&
        latents.video_target_frame_count > 0 &&
        final_latent.shape()[2] == latents.video_target_frame_count + latents.video_conditioning_frame_count) {
        const int64_t target_frames = latents.video_target_frame_count;
        const int64_t latent_channels = sd_ctx->sd->get_latent_channel();
        if (latents.audio_length > 0 && final_latent.shape()[3] > latent_channels) {
            auto video_latent = sd::ops::slice(final_latent, 3, 0, latent_channels);
            auto audio_latent = unpack_ltxav_audio_latent(final_latent, latents.audio_length, (int)latent_channels);
            if (audio_latent.empty()) {
                LOG_ERROR("could not preserve packed audio while stripping stage-one keyframe guides");
                return false;
            }
            video_latent = sd::ops::slice(video_latent, 2, 0, target_frames);
            final_latent = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
        } else {
            final_latent = sd::ops::slice(final_latent, 2, 0, target_frames);
        }
        latents.video_conditioning_frame_count = 0;
        latents.video_positions = {};
        LOG_INFO("LTXV refine: stripped %d stage-one keyframe guide frame(s); stage-two will re-apply them separately",
                 sd_vid_gen_params->keyframes_size);
    }

    // Continuation + hires: when LTX latent spatial upscale is enabled, the continuation
    // latent handed back to the caller (final_latent_out) must be the BASE (pre-upscale)
    // latent, so the NEXT chained segment seeds + samples at the same base resolution it
    // will upscale from. Capture it here BEFORE the hires block overwrites final_latent
    // with the upscaled latent; the decoded frames_out below still come from the upscaled
    // latent. Spatial upscale preserves the temporal frame count, so the caller's overlap
    // bookkeeping is unchanged. (Deep copy: final_latent is reassigned by the hires pass.)
    sd::Tensor<float> chain_base_latent;
    if (latent_upscale_enabled && final_latent_out != nullptr) {
        int64_t phase_t0 = ggml_time_ms();
        chain_base_latent = final_latent;
        LOG_INFO("[LTX_PHASE] chain base latent capture took %.3fs", (ggml_time_ms() - phase_t0) * 1.0f / 1000);
    }

    // BASE stage preview (stage_scale 1): the fast low-res base latent, decoded BEFORE any refine so
    // a client gets a rough preview as early as possible. No-op unless emit_stages was requested.
    emit_stage_preview(1, final_latent);

    if (latent_refine_enabled) {
        int64_t refine_total_start = ggml_time_ms();
        int64_t upscale_start             = ggml_time_ms();
        // The stage-1 DiT has finished before the latent upsampler runs. Keeping
        // its streamed/shared GPU residency through the upsampler makes the
        // latter's largest Conv3D allocation overlap with otherwise-dead DiT
        // weights. On a 45-latent-frame continuation this was the real 13.4 GB
        // board spike (the VAE decode itself is 11.26 GB). Release it here and
        // let the stage-2 sample re-stream the exact same weights from host.
        // This is phase-local allocation lifetime only: no latent, conditioning,
        // or sampler value changes. It applies only to host-offloaded LTXAV
        // two-stage renders; legacy/non-LTX callers retain their lifetime.
        const char* relip_free_before_upscale_env = getenv("LTXAV_RELIP_FREE_DIT_BEFORE_UPSCALE");
        const bool free_dit_before_upscale = sd_version_is_ltxav(sd_ctx->sd->version) &&
                                             (relip_free_before_upscale_env == nullptr ||
                                              std::string(relip_free_before_upscale_env) != "0") &&
                                             sd_ctx->sd->diffusion_model &&
                                             sd_ctx->sd->diffusion_model->params_offloaded_to_host();
        if (free_dit_before_upscale) {
            sd_ctx->sd->diffusion_model->release_all_gpu_param_residency();
            sd_ctx->sd->diffusion_model->free_compute_buffer();
            // release_all_gpu_param_residency returns buffers to the CUDA VMM
            // pool. Trim that pool before the upsampler asks cuDNN for its
            // largest Conv3D allocation; otherwise the driver-visible peak is
            // unchanged even though the DiT is no longer usable here.
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
            LOG_INFO("LTXAV two-stage: released stage-1 DiT residency before latent upscale");
        }
        sd::Tensor<float> upscaled_latent;
        if (same_res_refine_enabled) {
            upscaled_latent = final_latent;
            LOG_INFO("LTX same-resolution latent refine: skipping spatial upscale");
        } else {
            upscaled_latent = upscale_ltx_spatial_video_latent(sd_ctx,
                                                               request.hires.model_path,
                                                               final_latent,
                                                               latents.audio_length);
        }
        int64_t upscale_end               = ggml_time_ms();
        if (upscaled_latent.empty()) {
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return false;
        }
        if (!same_res_refine_enabled) {
            LOG_INFO("LTX latent spatial upscale completed, taking %.2fs",
                     (upscale_end - upscale_start) * 1.0f / 1000);
        }

        x_t                        = std::move(upscaled_latent);
        hires_request.width        = static_cast<int>(x_t.shape()[0]) * hires_request.vae_scale_factor;
        hires_request.height       = static_cast<int>(x_t.shape()[1]) * hires_request.vae_scale_factor;
        int upscaled_latent_frames = static_cast<int>(x_t.shape()[2]);
        int upscaled_frames        = sd_ctx->sd->latent_frames_to_video_frames(upscaled_latent_frames);
        if (upscaled_frames != hires_request.frames) {
            LOG_INFO("LTX latent upsampler output latent frames %d, frames %d -> %d",
                     upscaled_latent_frames,
                     hires_request.frames,
                     upscaled_frames);
            hires_request.frames = upscaled_frames;
        }
        if (sd_version_is_ltxav(sd_ctx->sd->version) && latents.audio_length > 0) {
            int target_audio_length = get_ltxav_num_audio_latents(hires_request.frames, hires_request.fps);
            const int expected_audio_length = latents.audio_reference_conditioning
                                                  ? target_audio_length * 2
                                                  : target_audio_length;
            if (expected_audio_length != latents.audio_length) {
                int latent_channels            = sd_ctx->sd->get_latent_channel();
                sd::Tensor<float> video_latent = x_t;
                sd::Tensor<float> audio_latent = latents.audio_latent;
                if (x_t.shape()[3] > latent_channels) {
                    video_latent = sd::ops::slice(x_t, 3, 0, latent_channels);
                    audio_latent = unpack_ltxav_audio_latent(x_t, latents.audio_length, latent_channels);
                }
                if (latents.audio_reference_conditioning) {
                    // Keep the official `[target | clean reference]` layout through
                    // spatial upscale. Both halves represent the same timeline.
                    auto target = sd::ops::slice(audio_latent, 1, 0, latents.audio_target_length);
                    auto reference = sd::ops::slice(audio_latent, 1, latents.audio_target_length, audio_latent.shape()[1]);
                    target          = resize_ltxav_audio_latent(target, target_audio_length);
                    reference       = resize_ltxav_audio_latent(reference, target_audio_length);
                    audio_latent    = sd::ops::concat(target, reference, 1);
                    latents.audio_target_length = target_audio_length;
                    latents.audio_positions = build_ltxav_lipdub_audio_positions(target_audio_length, target_audio_length);
                } else {
                    audio_latent = resize_ltxav_audio_latent(audio_latent, target_audio_length);
                }
                if (audio_latent.empty()) {
                    LOG_ERROR("failed to resize LTX audio latent for latent upscale: %d -> %d",
                              latents.audio_length,
                              expected_audio_length);
                    if (sd_ctx->sd->free_params_immediately) {
                        sd_ctx->sd->diffusion_model->free_params_buffer();
                    }
                    return false;
                }
                x_t                  = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
                latents.audio_latent = std::move(audio_latent);
                LOG_INFO("LTX audio latent length adjusted for latent upscale: %d -> %d",
                         latents.audio_length,
                         expected_audio_length);
                latents.audio_length = expected_audio_length;
            }
        }
        if (!same_res_refine_enabled &&
            (request.hires.target_width > 0 || request.hires.target_height > 0) &&
            (request.hires.target_width != hires_request.width || request.hires.target_height != hires_request.height)) {
            LOG_WARN("LTX latent spatial upsampler output is %dx%d; ignoring hires target %dx%d",
                     hires_request.width,
                     hires_request.height,
                     request.hires.target_width,
                     request.hires.target_height);
        }
        // FIX 2 (LTXAV_TWOSTAGE_FREE_STAGE1, default on): the stage-1 setup on `latents` (half-res
        // init latent + the stage-1 relip reference token block + its positions) is dead once the
        // upscaled latent is in x_t — stage-2 rebuilds all of it (hires_* + a freshly re-encoded
        // full-res reference). Release those host tensors and free the stage-1 DiT/VAE compute
        // buffers BEFORE the stage-2 reference VAE encode (inside apply_ltxv_refine_image_conditioning)
        // so that encode buffer does not co-reside with stage-1's working set (the ~441 MB that OOM'd
        // the 193f transition). Two-stage path only; gated so it can be A/B'd.
        if (latents.relip_twostage) {
            bool free_stage1 = true;
            if (const char* e = std::getenv("LTXAV_TWOSTAGE_FREE_STAGE1")) {
                free_stage1 = std::string(e) != "0";
            }
            if (free_stage1) {
                latents.video_reference = {};  // stage-1 half-res ref block (replaced by hires_video_reference)
                latents.init_latent     = {};  // stage-1 init latent (replaced by x_t = upscaled)
                latents.video_positions = {};  // stage-1 positions (replaced by hires_video_positions)
                if (sd_ctx->sd->diffusion_model) {
                    sd_ctx->sd->diffusion_model->free_compute_buffer();
                }
                sd_ctx->sd->first_stage_model->free_compute_buffer();
                LOG_INFO("LTXAV two-stage: freed stage-1 latents + DiT/VAE compute buffers before stage-2 reference encode");
            }
        }
        sd::Tensor<float> hires_denoise_mask;
        sd::Tensor<float> hires_video_positions;
        sd::Tensor<float> hires_video_reference;  // Change B/C: stage-2 re-applied relip reference
        int64_t refine_conditioning_start = ggml_time_ms();
        if (!apply_ltxv_refine_image_conditioning(sd_ctx,
                                                  sd_vid_gen_params,
                                                  hires_request,
                                                  latents,
                                                  &x_t,
                                                  &hires_denoise_mask,
                                                  &hires_video_positions,
                                                  &hires_video_reference)) {
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return false;
        }
        LOG_INFO("[LTX_PHASE] refine image/audio conditioning took %.3fs", (ggml_time_ms() - refine_conditioning_start) * 1.0f / 1000);

        // LTX_REFINE_CONTEXT_FRAMES=N (default unset = byte-identical): "refine only the surviving
        // frames" knob. A continuation segment appends the prior-segment guide as K extra TAIL frames
        // (video_conditioning_frame_count, sd:5138/6774) whose RoPE lands at the segment start; the base
        // pass samples the full [target(T), guide(K)] sequence for coherence, but those K guide frames are
        // CROPPED off the latent (:9439) BEFORE decode — the hires/refine sample() spends compute+VRAM on
        // frames that are thrown away (measured: seg-1 13f refine = 10531 MiB vs seg-2 16f refine = 12995,
        // purely the K=3 guide frames; 13f fits ≤11.5). This slices the upscaled latent to [target(T) + the
        // N guide frames NEAREST the surviving set] before the refine so it denoises only the surviving
        // target frames plus N seam-context frames. N=0 drops the whole guide (max VRAM save, max seam
        // risk); N>=K = full refine = unchanged. Retained context frames sit at tensor indices [T, T+N) and
        // are discarded by the SAME crop at :9439 (slice start=0, len=video_target_frame_count=T), so NO
        // downstream reassembly is needed; the surviving target frames keep their exact HEAD positions so
        // the (empty) refine positions' default RoPE is identical for them — only the trailing context set
        // they attend to shrinks. chain_base_latent (next-seg seed, captured pre-upscale at :9057) is
        // unaffected.
        //
        // AUDIO (the deployed lipdub/driven-audio continuation): audio is packed as frame-count-keyed extra
        // CHANNELS (pack_ltxav_audio_and_video_latents, sd:4964), so we CANNOT slice the packed frame dim
        // directly (that shears the flat audio blob). Instead unpack -> slice the VIDEO frames only ->
        // repack the SAME audio latent (audio is a segment-global timeline, independent of the video guide
        // frames; unpack returns {16, audio_length, 8} regardless of video T) -> rebuild the audio-pinned
        // denoise_mask for the sliced video. audio_length is unchanged so the post-refine audio decode
        // (:9384) and crop (:9439) behave exactly as with the full frame count.
        //
        // GUARDED to the continuation refine whose refine positions are empty (default RoPE rebuilt by
        // sample from the latent shape): LTXAV, K>0, T>0, the exact [T+K] video-frame layout, no relip
        // two-stage, no init/end image, and empty hires_video_positions (non-empty positions ONLY arise
        // from the relip/end-image branches, sd:8414/8538, which are frame-coupled and separately handled).
        // audio_length>0 and a non-empty audio-pinned mask are BOTH handled. Any other path is a no-op
        // (byte-identical), with the specific failing guard logged.
        const char* ctx_env = std::getenv("LTX_REFINE_CONTEXT_FRAMES");
        const bool default_hires_ref_trim = ctx_env == nullptr &&
                                            sd_vid_gen_params->cont_refine_latent != nullptr &&
                                            sd_vid_gen_params->cont_refine_latent_frames > 0;
        if (ctx_env != nullptr || default_hires_ref_trim) {
            // Once the separately transported refined guide is present, the appended low-res
            // base guide is redundant in stage 2 and is cropped from the final output anyway.
            // Drop it by default; the full-resolution refined guide is attached immediately
            // below. An explicit environment value still wins for A/B and recovery.
            int ctx_n         = ctx_env != nullptr ? atoi(ctx_env) : 0;
            int64_t T_tgt     = latents.video_target_frame_count;
            int64_t K_cond    = latents.video_conditioning_frame_count;
            int64_t x_frames  = x_t.dim() > 2 ? x_t.shape()[2] : 0;
            int64_t lat_ch    = sd_ctx->sd->get_latent_channel();
            bool has_audio    = latents.audio_length > 0 && x_t.shape()[3] > lat_ch;
            bool ltxav        = sd_version_is_ltxav(sd_ctx->sd->version);
            bool no_image     = sd_vid_gen_params->init_image.data == nullptr &&
                                sd_vid_gen_params->end_image.data == nullptr;
            bool pos_ok       = hires_video_positions.empty();     // non-empty => relip/end-image (frame-coupled)
            bool ref_ok       = hires_video_reference.empty();     // non-empty => relip two-stage
            bool layout_ok    = (x_frames == T_tgt + K_cond);
            bool eligible     = ctx_n >= 0 && ltxav && T_tgt > 0 && K_cond > 0 &&
                                !latents.relip_twostage && no_image && pos_ok && ref_ok && layout_ok;
            if (eligible) {
                int64_t keep = std::min<int64_t>(ctx_n, K_cond);
                if (keep < K_cond) {
                    int64_t refine_frames = T_tgt + keep;
                    if (has_audio) {
                        // unpack -> slice video frames -> repack (audio + audio-pinned mask preserved)
                        sd::Tensor<float> video_latent = sd::ops::slice(x_t, 3, 0, lat_ch);
                        sd::Tensor<float> audio_latent = unpack_ltxav_audio_latent(x_t, latents.audio_length, (int)lat_ch);
                        video_latent                   = sd::ops::slice(video_latent, 2, 0, refine_frames);
                        x_t = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
                        if (!hires_denoise_mask.empty()) {
                            // rebuild the audio-pinned mask for the sliced video (mirror apply_ltxv_refine :8464/8433)
                            sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video_latent, 1.f);
                            hires_denoise_mask = pack_ltxav_audio_and_video_denoise_mask(
                                video_mask, video_latent, audio_latent, latents.audio_fixed ? 0.0f : 1.0f);
                        }
                    } else {
                        x_t = sd::ops::slice(x_t, 2, 0, refine_frames);
                    }
                    LOG_INFO("LTX_REFINE_CONTEXT_FRAMES=%d%s: refine sliced to %lld surviving + %lld guide-context frames "
                             "(dropped %lld of %lld throwaway guide frames from the refine; audio=%s) -> refine T=%lld",
                             ctx_n, default_hires_ref_trim ? " (default with hires reference)" : "",
                             (long long)T_tgt, (long long)keep, (long long)(K_cond - keep),
                             (long long)K_cond, has_audio ? "preserved" : "none", (long long)refine_frames);
                } else {
                    LOG_INFO("LTX_REFINE_CONTEXT_FRAMES=%d >= guide frames %lld: refining all frames (no slice)",
                             ctx_n, (long long)K_cond);
                }
            } else {
                LOG_INFO("LTX_REFINE_CONTEXT_FRAMES set but refine ineligible -> refining all %lld frames unchanged "
                         "(n=%d ltxav=%d T_tgt=%lld K_cond=%lld layout_ok=%d relip=%d img=%d pos_empty=%d ref_empty=%d audio_len=%d)",
                         (long long)x_frames, ctx_n, (int)ltxav, (long long)T_tgt, (long long)K_cond, (int)layout_ok,
                         (int)latents.relip_twostage, (int)!no_image, (int)pos_ok, (int)ref_ok, latents.audio_length);
            }
        }

        // Dual-resolution continuation: stage 1 already received the prior BASE-grid tail via
        // cont_latent. Add the prior segment's actual REFINED VIDEO tail only after any optional
        // base-guide context trim above, so the refine target excludes those otherwise-throwaway
        // base-guide frames while retaining a full-fidelity appearance guide. This is the FIRST
        // hires stage (base*2), so prefer the matching-resolution stage-0 (lower-res) guide when the
        // caller transported one; fall back to cont_refine_latent (the final full-res tail — matches
        // only on the single-stage 2x path where this block IS the sole refine). It runs BEFORE the
        // character-reference reattach below so that append composes the identity rows ONTO this
        // guide layout instead of the two excluding each other.
        const float* cont_src        = sd_vid_gen_params->cont_refine_latent;
        int          cont_src_frames = sd_vid_gen_params->cont_refine_latent_frames;
        int          cont_src_width  = sd_vid_gen_params->cont_refine_latent_width;
        int          cont_src_height = sd_vid_gen_params->cont_refine_latent_height;
        int          cont_src_chan   = sd_vid_gen_params->cont_refine_latent_channels;
        if (sd_vid_gen_params->cont_refine_latent_lo != nullptr &&
            sd_vid_gen_params->cont_refine_latent_lo_frames > 0) {
            cont_src        = sd_vid_gen_params->cont_refine_latent_lo;
            cont_src_frames = sd_vid_gen_params->cont_refine_latent_lo_frames;
            cont_src_width  = sd_vid_gen_params->cont_refine_latent_lo_width;
            cont_src_height = sd_vid_gen_params->cont_refine_latent_lo_height;
            cont_src_chan   = sd_vid_gen_params->cont_refine_latent_lo_channels;
        }
        if (ltxav_chain_hires_reference_enabled() && !latents.relip_twostage &&
            cont_src != nullptr && cont_src_frames > 0) {
            const int64_t video_ch = sd_ctx->sd->get_latent_channel();
            const int64_t target_w = x_t.shape()[0];
            const int64_t target_h = x_t.shape()[1];
            const int64_t target_t = x_t.shape()[2];
            const int64_t guide_t  = cont_src_frames;
            // Keep the empty()-guard: with the character-ref reattach moved AFTER this block it no
            // longer pre-populates the reference/positions, so an empty check here NO LONGER blocks
            // the char-ref compose (char-ref concatenates onto this guide below). The only thing it
            // still blocks is a KEYFRAME refine layout (apply_ltxv_refine_image_conditioning sets
            // video_reference/positions for kf_cont merged shots): those must not be clobbered by the
            // continuation guide. Mirrors the stage-1 gate's stage_reference/stage_positions empty check.
            const bool shape_ok = cont_src_width == target_w &&
                                  cont_src_height == target_h &&
                                  cont_src_chan == video_ch &&
                                  guide_t <= target_t &&
                                  sd_vid_gen_params->init_image.data == nullptr &&
                                  sd_vid_gen_params->end_image.data == nullptr &&
                                  hires_video_reference.empty() && hires_video_positions.empty();
            if (!shape_ok) {
                LOG_WARN("LTX hires continuation reference skipped: prior [%d,%d,%d,%d] vs target [%lld,%lld,%lld,%lld], image/ref conditioning=%d/%d/%d",
                         cont_src_width, cont_src_height, cont_src_frames, cont_src_chan,
                         (long long)target_w, (long long)target_h, (long long)target_t, (long long)video_ch,
                         (int)(sd_vid_gen_params->init_image.data != nullptr),
                         (int)(sd_vid_gen_params->end_image.data != nullptr),
                         (int)(!hires_video_reference.empty() || !hires_video_positions.empty()));
            } else {
                hires_video_reference = sd::Tensor<float>({target_w, target_h, guide_t, video_ch, 1});
                std::memcpy(hires_video_reference.data(), cont_src,
                            (size_t)hires_video_reference.numel() * sizeof(float));
                hires_video_positions = build_ltxv_video_positions(target_w, target_h, target_t, guide_t,
                                                                     /*keyframe_frame_idx*/ 0,
                                                                     /*keyframe_pixel_frames*/ 8,
                                                                     hires_request.fps,
                                                                     hires_request.vae_scale_factor,
                                                                     8,
                                                                     true);
                LOG_INFO("LTX hires continuation reference (stage 0, %s): %lld refined VIDEO tail frames as separate guide tokens; target=%lld + guide=%lld frames",
                         (sd_vid_gen_params->cont_refine_latent_lo != nullptr &&
                          sd_vid_gen_params->cont_refine_latent_lo_frames > 0) ? "lower-res" : "full-res",
                         (long long)guide_t, (long long)target_t, (long long)guide_t);
            }
        }

        // apply_ltxv_refine_image_conditioning deliberately replaces the stage-1 relip block, and the
        // continuation apply above builds the target/guide layout. Re-append the persistent identity
        // block LAST, for relip / ordinary hires / continuation alike: append_ltxav_character_reference
        // CONCATENATES the identity rows onto whatever reference/positions already exist (keying on
        // video_positions.empty()), so the character reference COMPOSES with the continuation guide
        // instead of the two excluding each other.
        if (sd_version_is_ltxav(sd_ctx->sd->version) && ltxav_character_ref_enabled() &&
            sd_vid_gen_params->character_reference_latent != nullptr) {
            // First hires/refine stage (base*2): attach the stage-0 (_lo) identity block so this
            // refine gets a resolution-matched, higher-detail character than the base-res hint;
            // falls back to the base-res latent when no _lo was supplied (byte-identical).
            sd::Tensor<float> character = ltxav_character_latent_for_stage(sd_vid_gen_params, LtxavCharTier::Lo);
            ImageGenerationLatents refine_refs;
            refine_refs.init_latent = x_t;
            refine_refs.video_reference = std::move(hires_video_reference);
            refine_refs.video_positions = std::move(hires_video_positions);
            if (!append_ltxav_character_reference(&refine_refs, character, hires_request.fps,
                                                  hires_request.vae_scale_factor, 8)) {
                return false;
            }
            hires_video_reference = std::move(refine_refs.video_reference);
            hires_video_positions = std::move(refine_refs.video_positions);
        }

        // LTXAV_PIN_REFINE_OVERLAP (stage 0): hold this segment's overlap frames CLEAN through the
        // 0.85 SDEdit instead of letting it re-invent them. Uses the SAME `cont_src` the guide block
        // above selected (stage-0's own lower-res tail when transported, else the full-res tail —
        // whichever matches THIS stage's [W,H]; K is temporal so it is resolution-independent).
        // Runs AFTER the guide/character-reference appends so it sees the final target/guide layout,
        // and BEFORE the noise + temporal-window blocks so the pin composes with both: the blend path
        // slices its per-tile latent out of x_t and its per-tile mask out of hires_denoise_mask, so
        // pinned values and zeroed mask frames both survive the tiling. (That path is INERT in prod
        // regardless — it is gated `&& !hires_continue_mode` at :11813 and prod sets LTX_HIRES_CONTINUE=1.)
        //
        // Gated to the PURE continuation shape (no init/end image, no timeline keyframes, not relip
        // two-stage): those refines legitimately pin frame 0 / keyframe indices to re-encoded images
        // via apply_ltxv_refine_image_conditioning, and an overlap pin over [0,K) would fight them.
        if (ltxav_pin_refine_overlap_enabled() && sd_version_is_ltxav(sd_ctx->sd->version) &&
            !latents.relip_twostage && cont_src != nullptr && cont_src_frames > 0 &&
            sd_vid_gen_params->init_image.data == nullptr &&
            sd_vid_gen_params->end_image.data == nullptr &&
            (sd_vid_gen_params->keyframes == nullptr || sd_vid_gen_params->keyframes_size == 0)) {
            ltxav_pin_refine_overlap(&x_t,
                                     &hires_denoise_mask,
                                     cont_src,
                                     cont_src_frames,
                                     cont_src_width,
                                     cont_src_height,
                                     cont_src_chan,
                                     sd_ctx->sd->get_latent_channel(),
                                     sd_vid_gen_params->cont_latent_frames,
                                     latents.video_target_frame_count,
                                     "stage 0");
        }

        // LTX_REFINE_CONST_SEED (chain identity-stability): the chain gives each segment a DISTINCT
        // seed (base+seg, sd:10183) for base-motion variety, but the stage-2 refine INHERITS that
        // per-segment RNG -> each segment's refine adds DIFFERENT noise -> re-denoises the face/skin
        // differently -> a skin-tone/identity "flash" at every seam. Re-seeding the refine noise to a
        // CONSTANT (seg-independent) makes the re-roll consistent across segments so identity holds,
        // while the base sampling keeps its per-segment variety. Value is arbitrary (only consistency
        // matters); default 42 when the toggle is "1". Off by default = byte-identical.
        int64_t refine_noise_start = ggml_time_ms();
        if (const char* e = std::getenv("LTX_REFINE_CONST_SEED"); e != nullptr && e[0] != '\0' && std::string(e) != "0") {
            uint64_t rseed = std::strtoull(e, nullptr, 10);
            if (rseed <= 1) {
                rseed = 42;
            }
            sd_ctx->sd->rng->manual_seed(rseed);
            LOG_INFO("LTX_REFINE_CONST_SEED: refine noise re-seeded to constant %llu (identity-stable across chain segments)",
                     (unsigned long long)rseed);
        }
        // LTXAV_SHARED_REFINE_NOISE: key this refine's noise to the ABSOLUTE timeline so the
        // continuation overlap is re-noised identically in both segments that cover it (see
        // positional_randn_like). Runs BEFORE the fallback below; off => the fallback is the only
        // path and is byte-identical to today.
        bool refine_noise_positional = false;
        if (ltxav_shared_refine_noise_enabled() && sd_version_is_ltxav(sd_ctx->sd->version) &&
            x_t.dim() == 5 && x_t.shape()[2] > 0 && x_t.shape()[3] >= sd_ctx->sd->get_latent_channel()) {
            const int64_t video_ch = sd_ctx->sd->get_latent_channel();
            const int64_t T_all    = x_t.shape()[2];
            // video_target_frame_count owns the [target | guide] split. The guide is trimmed out of
            // x_t above on the default (cont_refine_latent) path, so clamp to the tensor we have.
            const int64_t T_tgt = latents.video_target_frame_count > 0
                                      ? std::min<int64_t>(latents.video_target_frame_count, T_all)
                                      : T_all;
            const uint64_t pseed = ltxav_shared_refine_noise_seed();
            noise = positional_randn_like(x_t, sd_ctx->sd->rng, pseed,
                                          sd_vid_gen_params->chain_latent_offset, video_ch, T_tgt);
            refine_noise_positional = true;
            LOG_INFO("LTXAV shared refine noise (stage 0): abs latent offset %lld, target=%lld guide=%lld frames, "
                     "video_ch=%lld, seed=%llu",
                     (long long)sd_vid_gen_params->chain_latent_offset, (long long)T_tgt,
                     (long long)(T_all - T_tgt), (long long)video_ch, (unsigned long long)pseed);
        }
        // In continue mode this matches ComfyUI stage two's RandomNoise:
        // sample() applies (1-σ) * upscaled_x0 + σ * fresh_noise.
        if (!refine_noise_positional) {
            noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);
        }
        LOG_INFO("[LTX_PHASE] refine noise tensor setup took %.3fs", (ggml_time_ms() - refine_noise_start) * 1.0f / 1000);

        // FIX 3 (LTXAV_TWOSTAGE_FREE_UNUSED, default on): the stage-2 reference VAE encode is now
        // done and the stage-2 DiT forward (below) does NOT touch the video/audio VAE — those only
        // run at the final decode. Free their resident params (video VAE ~1385 MB + audio VAE
        // ~823 MB) to make ~2.2 GB of headroom for the forward; reload_{first_stage,audio_vae}_model()
        // re-materializes them before their decodes (~0.1s). Gated + two-stage only + reload-loader
        // required (only captured for LTXAV no-mmap), so the default single-stage path is untouched
        // and the free never happens without a proven reload path.
        if (latents.relip_twostage && sd_ctx->sd->resident_reload_loader) {
            bool free_unused = true;
            if (const char* e = std::getenv("LTXAV_TWOSTAGE_FREE_UNUSED")) {
                free_unused = std::string(e) != "0";
            }
            if (free_unused) {
                size_t before = sd_ctx->sd->first_stage_model->get_params_buffer_size();
                sd_ctx->sd->first_stage_model->free_params_buffer();
                if (sd_ctx->sd->audio_vae_model) {
                    before += sd_ctx->sd->audio_vae_model->get_params_buffer_size();
                    sd_ctx->sd->audio_vae_model->free_params_buffer();
                }
                LOG_INFO("LTXAV two-stage: freed video+audio VAE params (~%.0f MB) before stage-2 forward; reload at decode",
                         before / (1024.f * 1024.f));
            }
        }

        W                                   = hires_request.width / hires_request.vae_scale_factor;
        H                                   = hires_request.height / hires_request.vae_scale_factor;
        T                                   = static_cast<int>(x_t.shape()[2]);
        // Independent refine-pass sampler. Precedence: JSON hires.sample_method (request.hires.sample_method)
        // > LTXAV_HIRES_SAMPLE_METHOD env backstop > inherit the base pass sampler (plan.sample_method).
        // Unset (both sentinel/absent) keeps today's behavior IDENTICAL: refine inherits the base sampler.
        sample_method_t hires_sample_method =
            (request.hires.sample_method != SAMPLE_METHOD_COUNT) ? request.hires.sample_method : plan.sample_method;
        if (request.hires.sample_method == SAMPLE_METHOD_COUNT) {
            if (const char* e = std::getenv("LTXAV_HIRES_SAMPLE_METHOD"); e && *e) {
                sample_method_t m = str_to_sample_method(e);
                if (m != SAMPLE_METHOD_COUNT) {
                    hires_sample_method = m;
                }
            }
        }
        // Plain-generate ver3 refine default: a plain t2v/i2v two-pass render that provided no
        // hires.custom_sigmas defaults to the ver3 2-step refine [0.725, 0.421875, 0.0] so the server
        // fully defaults to ver3 (symmetric with the 6-step base baked in SamplePlan; base ends at
        // 0.725, this refine finishes it). Gated to plain generate (no control_frames, not guide-edit);
        // relip sets its own 3-step custom_sigmas upstream so it never reaches here.
        if (sd_version_is_ltxav(sd_ctx->sd->version) &&
            sd_vid_gen_params->control_frames_size <= 0 &&
            sd_vid_gen_params->v2v_mode != 2 &&
            (request.hires.custom_sigmas_count <= 0 || request.hires.custom_sigmas == nullptr)) {
            static float kVer3RefineSigmas[] = {0.725f, 0.421875f, 0.0f};
            request.hires.custom_sigmas       = kVer3RefineSigmas;
            request.hires.custom_sigmas_count = 3;
            LOG_INFO("LTXAV ver3 baked refine default: 3 sigmas => 2 steps [0.72500 .. 0.00000]");
        }
        int hires_scheduler_steps           = 0;
        int64_t refine_schedule_start       = ggml_time_ms();
        std::vector<float> hires_sigma_sched =
            make_hires_sigma_schedule(sd_ctx,
                                      request.hires,
                                      sd_vid_gen_params->sample_params,
                                      hires_sample_method,
                                      plan.sample_steps,
                                      sd_ctx->sd->get_image_seq_len(hires_request.height, hires_request.width) * T,
                                      &hires_scheduler_steps);
        float hires_eta = resolve_eta(sd_ctx,
                                      sd_vid_gen_params->sample_params.eta,
                                      hires_sample_method);
        LOG_INFO("[LTX_PHASE] refine sigma/scheduler setup took %.3fs", (ggml_time_ms() - refine_schedule_start) * 1.0f / 1000);

        LOG_DEBUG("%s %dx%dx%d", same_res_refine_enabled ? "sample(same-res refine)" : "sample(latent upscale)", W, H, T);
        LOG_INFO("LTX %s refine: scheduler_steps=%d, denoising_strength=%.2f, sampler=%s, sigma_sched_size=%zu%s",
                 same_res_refine_enabled ? "same-resolution latent" : "latent spatial upscale",
                 hires_scheduler_steps,
                 request.hires.denoising_strength,
                 sampling_methods_str[hires_sample_method],
                 hires_sigma_sched.size(),
                 request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");
        if (hires_continue_mode && !plan.sigmas.empty() && !hires_sigma_sched.empty()) {
            LOG_INFO("LTX_HIRES_CONTINUE: base denoised x0 -> latent upscale -> noise_scaling(sigma=%.5f, fresh_noise) -> %s refine (base_end=%.5f)",
                     hires_sigma_sched.front(),
                     sampling_methods_str[hires_sample_method],
                     plan.sigmas.back());
        }

        // Pre-sample pool trim for the two-stage stage-2 refine (see the same trim before the
        // stage-1 / single-stage sample above). The latent upscaler (~950 MB offload) and the
        // full-res stage-2 reference VAE encode both just ran; their freed scratch is still
        // committed in the CUDA VMM pool and would otherwise inflate this refine's peak. Trim it
        // back before the refine reserves its compute buffer. Gated by LTXAV_PRE_SAMPLE_POOL_TRIM.
        if (sd_version_is_ltxav(sd_ctx->sd->version)) {
            const char* e = getenv("LTXAV_PRE_SAMPLE_POOL_TRIM");
            if (e == nullptr || std::string(e) != "0") {
                int64_t phase_t0 = ggml_time_ms();
                ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
                // REFINE-PEAK FIX (97f continuation): the trim above reclaims the VMM pool
                // high-water, but the DiT's idle param-streaming buffers (prefetch_buf_pool_ +
                // the last prefetched_state_.buf, ~420-627 MB left by the just-finished base
                // sample) live OUTSIDE that pool and squat through the refine — the true 97f
                // ceiling. Nothing is in flight here (base sample + upscale done, refine buffer
                // not reserved yet; the trim already synced), so free them before the refine
                // reserves its compute buffer. The refine's own stream re-creates them lazily
                // (one-time buffer alloc, NOT a weight re-stream → no slowdown). No-op when empty
                // → single-render / 720p-flat byte-identical.
                if (sd_ctx->sd->diffusion_model) {
                    sd_ctx->sd->diffusion_model->free_streaming_scratch_buffers();
                }
                LOG_INFO("[LTX_PHASE] pre-refine pool/streaming trim took %.3fs", (ggml_time_ms() - phase_t0) * 1.0f / 1000);
            }
        }

        // FEATURE 1 (--hires-lora): swap in the per-phase refine LoRA set immediately before the
        // refine sample(). apply_loras diffs against curr_lora_state (set by the up-front base
        // apply_loras at :8390), so passing the FULL refine set (e.g. distill@0.8 [+detailer@0.7])
        // transitions cleanly from the base set (e.g. distill@0.65). For quantized (nvfp4) bases
        // this is the cheap runtime-multiplier swap (apply_loras_at_runtime) — no re-fold, same
        // VRAM/step-time as prod. No teardown needed: the next chained generate_video re-asserts
        // the base set via its own up-front apply_loras. Detailer tensors that don't dim-match are
        // skipped inside load_lora_model_from_file (fails safe to distill-only on the refine).
        if (request.hires.lora_count > 0 && request.hires.loras != nullptr) {
            LOG_INFO("hires: applying %u per-phase refine LoRA(s) before the refine sample", request.hires.lora_count);
            sd_ctx->sd->apply_loras(request.hires.loras, request.hires.lora_count);
        }

        // LTXAV_VAE_LAZY (2nd eviction — the two-stage refine): the render is TWO passes (base
        // sample -> latent x2 upscale -> this hires/refine sample). The pre-BASE eviction at :8615
        // frees the VAE, but the offload pipeline RE-OFFLOADS the video VAE (1385 MB) between the two
        // samples ("ltx_video_vae offload params (1385.02 MB, ...) to runtime backend (CUDA0)" at the
        // hires stage), so by the refine VAE_gpu is back to 1385 and the refine (DiT 5471 + VAE 1385 +
        // compute ~2767) is the ~11913 MB binding peak. The plain t2v two-stage refine is a pure latent
        // denoise — it does NOT touch the video/audio VAE (only the final decode does, which re-offloads
        // it: decode-entry correctly shows VAE_gpu=1385). So release both VAEs' GPU residency again here.
        // Relip has completed its stage-2 reference encode before this point, so it too can release
        // the VAE before refine and re-offload it lazily for the final decode. The only requirement
        // is host-offloaded params, which makes the re-offload mmap-safe. Opt-in via LTXAV_VAE_LAZY.
        if (ltxav_vae_lazy &&
            sd_version_is_ltxav(sd_ctx->sd->version) &&
            sd_ctx->sd->first_stage_model &&
            sd_ctx->sd->first_stage_model->params_offloaded_to_host()) {
            int64_t phase_t0 = ggml_time_ms();
            sd_ctx->sd->first_stage_model->release_all_gpu_param_residency();
            if (sd_ctx->sd->audio_vae_model) {
                sd_ctx->sd->audio_vae_model->release_all_gpu_param_residency();
            }
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::VAE));
            // Same reason as the first eviction: VAE param pointers are now stale, so free the
            // cuDNN conv3d reorder-weight buffers keyed by those pointers (else ~1.4 GB/segment leak).
            ggml_backend_cuda_release_cudnn_conv3d_weights();
            LOG_INFO("LTXAV_VAE_LAZY: re-released offloaded video+audio VAE GPU params (runtime + shared-resident) + trimmed VAE pool + freed conv3d reorder weights before the hires/refine sample; re-offload from host at decode");
            LOG_INFO("[LTX_PHASE] pre-refine VAE re-eviction took %.3fs", (ggml_time_ms() - phase_t0) * 1.0f / 1000);
        }

        // Refine-scoped resident byte cap (LONGCAT_SHARED_RESIDENT_MAX_MB): engage the
        // cap ONLY for this 3-step hires/refine sample so it pins just the hottest ~4 GB
        // (fits ≤11776 on the 97f 1080p continuation) while the 8-step base above keeps
        // its full pin for speed. No-op when the cap env is unset (byte-identical).
        float old_max_vram = sd_ctx->sd->max_vram;
        bool refine_max_vram_override = false;
        size_t old_max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(old_max_vram);
        if (const char* e = std::getenv("LTXAV_REFINE_MAX_VRAM"); e != nullptr && e[0] != '\0') {
            float refine_max_vram = (float)atof(e);
            if (refine_max_vram > 0.f && refine_max_vram != old_max_vram) {
                refine_max_vram_override = true;
                sd_ctx->sd->max_vram = refine_max_vram;
                size_t refine_max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(refine_max_vram);
                if (sd_ctx->sd->diffusion_model) {
                    sd_ctx->sd->diffusion_model->set_max_graph_vram_bytes(refine_max_graph_vram_bytes);
                }
                LOG_INFO("LTXAV_REFINE_MAX_VRAM: using %.2f GiB graph budget for refine sample only (base remains %.2f GiB)",
                         refine_max_vram,
                         old_max_vram);
            }
        }
        if (sd_ctx->sd->diffusion_model) {
            if (const char* e = std::getenv("LTXAV_REFINE_SHARED_RESIDENT_MAX_MB"); e != nullptr && e[0] != '\0') {
                const float mb = static_cast<float>(std::atof(e));
                if (mb > 0.f) {
                    sd_ctx->sd->diffusion_model->set_shared_resident_max_mb_override(mb);
                    LOG_INFO("LTXAV_REFINE_SHARED_RESIDENT_MAX_MB: using %.0f MB resident cap for refine only", mb);
                }
            }
            sd_ctx->sd->diffusion_model->set_refine_resident_scope(true);
            // The base pass deliberately keeps its full hot DiT residency for speed.
            // Return that stale base residency before the first high-resolution
            // sample materialises the DiT under its bounded refine policy.
            if (sd_ctx->sd->diffusion_model->params_offloaded_to_host()) {
                sd_ctx->sd->diffusion_model->release_all_gpu_param_residency();
                ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
                LOG_INFO("LTXAV refine handoff: released full base DiT GPU residency before bounded refine reuse");
            }
        }

        LOG_INFO("[LTX_PHASE] refine setup total before sample took %.3fs", (ggml_time_ms() - refine_total_start) * 1.0f / 1000);
        sampling_start = ggml_time_ms();
        sd_guidance_params_t hires_guidance = sd_vid_gen_params->sample_params.guidance;
        if (std::isfinite(request.hires.cfg)) {
            hires_guidance.txt_cfg = request.hires.cfg;
        }
        const bool hires_needs_uncond = request.use_uncond || hires_guidance.txt_cfg > 1.f ||
                                        hires_sample_method == EULER_CFG_PP_SAMPLE_METHOD ||
                                        hires_sample_method == EULER_A_CFG_PP_SAMPLE_METHOD;
        auto sample_refine_window = [&](const sd::Tensor<float>& window_latent,
                                        sd::Tensor<float> window_noise,
                                        const sd::Tensor<float>& window_mask,
                                        const sd::Tensor<float>& window_video_positions,
                                        const sd::Tensor<float>& window_audio_positions,
                                        const sd::Tensor<float>& window_video_reference) {
            return sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                      true,
                                      window_latent,
                                      std::move(window_noise),
                                      embeds.cond,
                                      hires_needs_uncond ? embeds.uncond : SDCondition(),
                                      embeds.img_uncond,
                                      sd::Tensor<float>(),
                                      0.f,
                                      hires_guidance,
                                      hires_eta,
                                      sd_vid_gen_params->sample_params.shifted_timestep,
                                      hires_sample_method,
                                      sd_ctx->sd->is_flow_denoiser(),
                                      plan.extra_sample_args,
                                      hires_sigma_sched,
                                      std::vector<sd::Tensor<float>>{},
                                      false,
                                      window_mask,
                                      sd::Tensor<float>(),
                                      hires_request.vace_strength,
                                      latents.audio_length,
                                      static_cast<float>(hires_request.fps),
                                      hires_request.cache_params,
                                      window_video_positions,
                                      window_audio_positions,
                                      // LipDub stage 2 freezes both the stage-1 target audio
                                      // and its appended clean reference.  Mark it fixed here
                                      // as well so the audio adaLN sees timestep zero.
                                      latents.audio_fixed || latents.audio_reference_conditioning,
                                      window_video_reference);
        };

        // Continuation's high-res guide has the same spatial grid as the
        // stage-2 target. It is safe to reuse for every temporal refine tile
        // when paired with a combined target+guide position tensor below.
        // Keep the relip/reference-image variants on their established full
        // path: their guide grid can have different spatial geometry.
        const bool refine_reference_windowable = !hires_video_reference.empty() &&
                                                 hires_video_reference.dim() >= 4 &&
                                                 hires_video_reference.shape()[0] == x_t.shape()[0] &&
                                                 hires_video_reference.shape()[1] == x_t.shape()[1] &&
                                                 hires_video_reference.shape()[3] == sd_ctx->sd->get_latent_channel();
        const bool refine_timeline_keyframes_windowable = refine_reference_windowable &&
                                                          sd_vid_gen_params->keyframes != nullptr &&
                                                          sd_vid_gen_params->keyframes_size > 0 &&
                                                          hires_video_reference.shape()[2] == sd_vid_gen_params->keyframes_size;
        const char* temporal_refine_env = std::getenv("LTX_REFINE_TEMPORAL_BLEND");
        const bool temporal_refine_enabled = temporal_refine_env != nullptr && temporal_refine_env[0] != '\0' &&
                                             std::string(temporal_refine_env) != "0" &&
                                             sd_version_is_ltxav(sd_ctx->sd->version) &&
                                             (hires_video_positions.empty() || refine_reference_windowable) &&
                                             x_t.dim() == 5 && x_t.shape()[2] > 1 &&
                                             // Keep the existing temporal-blend alternate path out of the
                                             // strictly-gated ComfyUI two-pass workflow.
                                             !hires_continue_mode;
        if (!temporal_refine_enabled) {
            final_latent = sample_refine_window(x_t,
                                                std::move(noise),
                                                hires_denoise_mask,
                                                hires_video_positions,
                                                latents.audio_positions,
                                                hires_video_reference);
        } else {
            int temporal_window = 8;
            int temporal_overlap = 2;
            if (const char* e = std::getenv("LTX_REFINE_TBLEND_FRAMES"); e != nullptr) {
                temporal_window = std::max(2, std::atoi(e));
            }
            if (const char* e = std::getenv("LTX_REFINE_TBLEND_OVERLAP"); e != nullptr) {
                temporal_overlap = std::max(1, std::atoi(e));
            }
            temporal_window = std::clamp(temporal_window, 2, static_cast<int>(x_t.shape()[2]));
            temporal_overlap = std::clamp(temporal_overlap, 1, temporal_window - 1);
            const int64_t total_frames = x_t.shape()[2];
            const int64_t stride = temporal_window - temporal_overlap;
            const int64_t latent_channels = sd_ctx->sd->get_latent_channel();
            // A fixed overlap carries immediate motion, but it falls out of the
            // attention horizon on the following tile.  Optionally retain a small,
            // late appearance reference as a separate token block: two memories,
            // one contiguous for motion and one non-contiguous for identity.  The
            // appearance reference rolls forward after every tile, never reaches
            // back to tile zero, so it can follow legitimate scene changes.
            int anchor_frames = 0;
            if (hires_video_reference.empty()) {
                if (const char* e = std::getenv("LTX_REFINE_TEMPORAL_ANCHOR_FRAMES"); e != nullptr) {
                    anchor_frames = std::max(0, std::atoi(e));
                }
            }
            anchor_frames = std::min<int>(anchor_frames, temporal_window - 1);
            const bool has_audio = latents.audio_length > 0 && x_t.shape()[3] > latent_channels;
            sd::Tensor<float> full_video = sd::ops::slice(x_t, 3, 0, latent_channels);
            sd::Tensor<float> full_noise = sd::ops::slice(noise, 3, 0, latent_channels);
            // Stage-two I2V/keyframe conditioning has replaced its locations in
            // x_t with full-resolution VAE latents and locked them in this mask.
            // Preserve that mask when refine is split into temporal windows. For
            // packed AV tensors, leading latent_channels are the video mask and
            // trailing channels are the separately rebuilt audio mask.
            sd::Tensor<float> full_video_denoise_mask;
            if (!hires_denoise_mask.empty()) {
                // A VIDEO-ONLY mask is 1-channel (make_ltxav_video_denoise_mask), not latent_channels:
                // that is what the no-audio refine already stores (:10065) and what the overlap pin
                // creates. resolve_slice_bounds THROWS when end > dim_size, so the unclamped
                // slice(..., 3, 0, latent_channels) was a live crash on any 1-channel mask reaching
                // here (i2v + LTX_REFINE_TEMPORAL_BLEND=1 + LTX_HIRES_CONTINUE=0 today). Clamping is a
                // no-op for every packed mask (min picks latent_channels) and keeps the 1-channel mask
                // whole — it broadcasts over the video channels exactly as sample() expects.
                const int64_t mask_video_ch = std::min<int64_t>(latent_channels, hires_denoise_mask.shape()[3]);
                full_video_denoise_mask     = sd::ops::slice(hires_denoise_mask, 3, 0, mask_video_ch);
            }
            sd::Tensor<float> audio_latent;
            sd::Tensor<float> audio_noise;
            bool audio_unpack_ok = true;
            if (has_audio) {
                audio_latent = unpack_ltxav_audio_latent(x_t, latents.audio_length, static_cast<int>(latent_channels));
                audio_noise = unpack_ltxav_audio_latent(noise, latents.audio_length, static_cast<int>(latent_channels));
                if (audio_latent.empty() || audio_noise.empty()) {
                    LOG_ERROR("LTX refine temporal-blend could not unpack packed audio latent/noise");
                    audio_unpack_ok = false;
                }
            }

            if (!audio_unpack_ok) {
                final_latent = {};
            } else {
                // The DiT normally derives RoPE coordinates from the input tensor shape.
                // That rebases every temporal refine tile to t=0 while its audio remains on
                // the segment-wide timeline, so the model reinterprets the same phonemes at
                // each tile boundary. Supply the original timeline explicitly, just as the
                // audio-windowed base pass does. The audio stays segment-global here, hence
                // its absolute positions cover the full packed audio sequence for every tile.
                const sd::Tensor<float> refine_audio_positions =
                    latents.audio_positions.empty() && has_audio
                        ? build_ltxav_window_audio_positions(0, latents.audio_length)
                        : latents.audio_positions;
                // Independent denoise+feather windows re-roll face/detail at every stride.
                // Keep the already-refined overlap frozen in the next window instead, then emit
                // only that window's new frames. The base pass remains globally coherent and the
                // high-res detail stage receives a real local temporal history.
                sd::Tensor<float> video_result(full_video.shape());
                video_result.fill_(0.0f);
                // Keep the original high-resolution I2V latent as an immutable
                // non-contiguous reference for every later refine tile.  The short
                // frozen overlap remains the local motion memory; this separate
                // frame-zero guide is identity/appearance memory and must never be
                // refreshed from generated motion.  It mirrors LTX's negative-index
                // guide semantics without forcing the source frame into the local
                // target timeline.
                sd::Tensor<float> immutable_i2v_guide;
                if (hires_video_reference.empty() && sd_vid_gen_params->init_image.data != nullptr &&
                    full_video.shape()[2] > 0) {
                    immutable_i2v_guide = sd::ops::slice(full_video, 2, 0, 1);
                    LOG_INFO("LTX refine temporal immutable I2V guide: source latent frame 0 retained for all tiles");
                }
                sd::Tensor<float> appearance_anchor;
                int64_t            appearance_anchor_start = 0;
                sd::Tensor<float> refined_audio;
                int tile_index = 0;
                int64_t produced_end = 0;
                for (int64_t start = 0;; start += stride, ++tile_index) {
                    const int64_t tile_start = start + temporal_window >= total_frames
                                                   ? std::max<int64_t>(0, total_frames - temporal_window)
                                                   : start;
                    const int64_t end = std::min<int64_t>(total_frames, tile_start + temporal_window);
                    const int64_t length = end - tile_start;
                    auto video_tile = sd::ops::slice(full_video, 2, tile_start, end);
                    auto noise_tile = sd::ops::slice(full_noise, 2, tile_start, end);
                    const int64_t frozen = tile_index == 0
                                               ? 0
                                               : std::min<int64_t>(length, std::max<int64_t>(0, produced_end - tile_start));
                    const int64_t plane = full_video.shape()[0] * full_video.shape()[1];
                    if (frozen > 0) {
                        float* tile_data = video_tile.data();
                        const float* result_data = video_result.data();
                        for (int64_t channel = 0; channel < latent_channels; ++channel) {
                            for (int64_t local = 0; local < frozen; ++local) {
                                std::memcpy(tile_data + plane * (local + length * channel),
                                            result_data + plane * (tile_start + local + total_frames * channel),
                                            static_cast<size_t>(plane) * sizeof(float));
                            }
                        }
                    }
                    sd::Tensor<float> latent_tile = video_tile;
                    sd::Tensor<float> packed_noise = noise_tile;
                    sd::Tensor<float> mask_tile;
                    // Slice the original stage-two mask first, then overlay the
                    // carried-overlap freeze. Recreating an all-ones tile mask
                    // here re-denoises locked I2V/keyframe latents.
                    auto video_mask = full_video_denoise_mask.empty()
                                          ? make_ltxav_video_denoise_mask(video_tile, 1.0f)
                                          : sd::ops::slice(full_video_denoise_mask, 2, tile_start, end);
                    if (frozen > 0) {
                        float* mask_data = video_mask.data();
                        const int64_t mask_channels = video_mask.shape()[3];
                        for (int64_t channel = 0; channel < mask_channels; ++channel) {
                            for (int64_t local = 0; local < frozen; ++local) {
                                std::fill_n(mask_data + plane * (local + length * channel), plane, 0.0f);
                            }
                        }
                    }
                    if (has_audio) {
                        latent_tile = pack_ltxav_audio_and_video_latents(video_tile, audio_latent);
                        packed_noise = pack_ltxav_audio_and_video_latents(noise_tile, audio_noise);
                        mask_tile = pack_ltxav_audio_and_video_denoise_mask(
                            video_mask, video_tile, audio_latent, latents.audio_fixed ? 0.0f : 1.0f);
                    } else {
                        mask_tile = std::move(video_mask);
                    }

                    LOG_INFO("LTX refine temporal-window tile %d: latent [%lld,%lld), window=%d frozen-overlap=%lld",
                             tile_index, (long long)tile_start, (long long)end, temporal_window, (long long)frozen);
                    const bool use_appearance_anchor = !appearance_anchor.empty();
                    const sd::Tensor<float>& identity_guide = refine_reference_windowable
                                                                   ? hires_video_reference
                                                                   : !immutable_i2v_guide.empty()
                                                                   ? immutable_i2v_guide
                                                                   : appearance_anchor;
                    std::vector<int> refine_keyframe_positions;
                    if (refine_timeline_keyframes_windowable) {
                        refine_keyframe_positions.reserve(sd_vid_gen_params->keyframes_size);
                        for (int i = 0; i < sd_vid_gen_params->keyframes_size; ++i) {
                            refine_keyframe_positions.push_back(sd_vid_gen_params->keyframe_frame_indices != nullptr
                                                                    ? sd_vid_gen_params->keyframe_frame_indices[i]
                                                                    : 0);
                        }
                    }
                    auto refine_video_positions = refine_timeline_keyframes_windowable
                        ? build_ltxav_window_video_positions_with_keyframes(video_tile.shape()[0],
                                                                              video_tile.shape()[1],
                                                                              tile_start,
                                                                              length,
                                                                              refine_keyframe_positions,
                                                                              hires_request.fps,
                                                                              hires_request.vae_scale_factor)
                        : refine_reference_windowable
                        ? build_ltxav_window_video_positions_with_reference(video_tile.shape()[0],
                                                                              video_tile.shape()[1],
                                                                              tile_start,
                                                                              length,
                                                                              hires_video_reference.shape()[2],
                                                                              hires_request.fps,
                                                                              hires_request.vae_scale_factor)
                        : !immutable_i2v_guide.empty()
                        ? build_ltxav_window_video_positions_with_reference(video_tile.shape()[0],
                                                                              video_tile.shape()[1],
                                                                              tile_start,
                                                                              length,
                                                                              immutable_i2v_guide.shape()[2],
                                                                              hires_request.fps,
                                                                              hires_request.vae_scale_factor)
                        : use_appearance_anchor
                        ? build_ltxav_window_video_positions_with_reference(video_tile.shape()[0],
                                                                              video_tile.shape()[1],
                                                                              tile_start,
                                                                              length,
                                                                              appearance_anchor.shape()[2],
                                                                              hires_request.fps,
                                                                              hires_request.vae_scale_factor,
                                                                              8,
                                                                              appearance_anchor_start)
                        : build_ltxav_window_video_positions(video_tile.shape()[0],
                                                             video_tile.shape()[1],
                                                             tile_start,
                                                             length,
                                                             hires_request.fps,
                                                             hires_request.vae_scale_factor);
                    auto tile = sample_refine_window(latent_tile,
                                                     std::move(packed_noise),
                                                     mask_tile,
                                                     refine_video_positions,
                                                     refine_audio_positions,
                                                     identity_guide);
                    if (tile.empty()) {
                        final_latent = {};
                        break;
                    }
                    auto refined_video = sd::ops::slice(tile, 3, 0, latent_channels);
                    if (has_audio && refined_audio.empty()) {
                        refined_audio = unpack_ltxav_audio_latent(tile, latents.audio_length, static_cast<int>(latent_channels));
                        if (refined_audio.empty()) {
                            LOG_ERROR("LTX refine temporal-blend could not unpack refined audio latent");
                            final_latent = {};
                            break;
                        }
                    }

                    const float* src = refined_video.data();
                    float* dst = video_result.data();
                    for (int64_t local = frozen; local < length; ++local) {
                        const int64_t global = tile_start + local;
                        for (int64_t channel = 0; channel < latent_channels; ++channel) {
                            const float* sp = src + plane * (local + length * channel);
                            float* dp = dst + plane * (global + total_frames * channel);
                            for (int64_t pixel = 0; pixel < plane; ++pixel) {
                                dp[pixel] = sp[pixel];
                            }
                        }
                    }
                    produced_end = std::max(produced_end, end);
                    // T2V's guide is deliberately rolling rather than global: every
                    // completed tile contributes a late, low-motion appearance frame
                    // for its immediate successor, so legitimate scene/character
                    // changes remain possible on longer clips.
                    const bool establish_anchor = immutable_i2v_guide.empty() && anchor_frames > 0;
                    if (establish_anchor) {
                        int64_t anchor_end = end;
                        int64_t anchor_start = std::max<int64_t>(tile_start + frozen, anchor_end - anchor_frames);
                        // T2V has no external identity image.  For the first guide only,
                        // choose a completed generated frame that is both locally detailed
                        // and low-motion relative to its predecessor instead of blindly
                        // anchoring the moving terminal frame of tile zero.  The contiguous
                        // frozen overlap still carries actual motion into the next tile.
                        const bool select_sharp_handoff =
                            std::getenv("LTX_REFINE_TEMPORAL_ANCHOR_SELECT") != nullptr &&
                            std::string(std::getenv("LTX_REFINE_TEMPORAL_ANCHOR_SELECT")) == "sharp";
                        if (select_sharp_handoff && anchor_end - anchor_start > 1) {
                            const int64_t width = full_video.shape()[0];
                            const int64_t height = full_video.shape()[1];
                            int64_t best = anchor_start;
                            double best_score = std::numeric_limits<double>::infinity();
                            const float* data = video_result.data();
                            for (int64_t frame = std::max<int64_t>(anchor_start, 1); frame < anchor_end; ++frame) {
                                double motion = 0.0, detail = 0.0;
                                int64_t n = 0;
                                for (int64_t ch = 0; ch < latent_channels; ch += 16) {
                                    const float* cur = data + plane * (frame + total_frames * ch);
                                    const float* prev = data + plane * (frame - 1 + total_frames * ch);
                                    for (int64_t y = 1; y < height; y += 4) {
                                        for (int64_t x = 1; x < width; x += 4) {
                                            const int64_t p = x + width * y;
                                            motion += std::fabs(cur[p] - prev[p]);
                                            detail += std::fabs(cur[p] - cur[p - 1]) + std::fabs(cur[p] - cur[p - width]);
                                            ++n;
                                        }
                                    }
                                }
                                const double score = motion / (detail + 1e-6);
                                if (n > 0 && score < best_score) { best_score = score; best = frame; }
                            }
                            anchor_start = best;
                            anchor_end = std::min<int64_t>(end, best + 1);
                            LOG_INFO("LTX refine temporal rolling sharp-handoff selected latent %lld (score %.6f)",
                                     (long long)best, best_score);
                        }
                        const int64_t count = anchor_end - anchor_start;
                        if (count > 0) {
                            const bool had_appearance_anchor = !appearance_anchor.empty();
                            appearance_anchor = sd::ops::slice(video_result, 2, anchor_start, anchor_end);
                            appearance_anchor_start = anchor_start;
                            LOG_INFO("LTX refine temporal appearance-anchor: %s latent [%lld,%lld) for later tiles",
                                     had_appearance_anchor ? "updated" : "established",
                                     (long long)appearance_anchor_start, (long long)anchor_end);
                        }
                    }
                    if (end == total_frames) {
                        final_latent = has_audio ? pack_ltxav_audio_and_video_latents(video_result, refined_audio) : std::move(video_result);
                        break;
                    }
                }
            }
        }
        sampling_end   = ggml_time_ms();
        if (sd_ctx->sd->diffusion_model) {
            sd_ctx->sd->diffusion_model->set_refine_resident_scope(false);
            sd_ctx->sd->diffusion_model->set_shared_resident_max_mb_override(0.f);
        }
        if (refine_max_vram_override) {
            sd_ctx->sd->max_vram = old_max_vram;
            if (sd_ctx->sd->diffusion_model) {
                sd_ctx->sd->diffusion_model->set_max_graph_vram_bytes(old_max_graph_vram_bytes);
            }
        }
        // FIX (1080p hires chain seg-2 crash): this hires/latent-upscale success-path free was
        // gated ONLY on free_params_immediately, missing the !keep_diffusion_model_resident
        // guard the non-hires branch below (:9292) has. On a warm N-segment chain
        // (keep_diffusion_model_resident=true) it nulled every non-resident DiT param
        // (data=null for ~4138/4444 tensors: the per-block attn/ff/scale_shift weights), so the
        // NEXT segment's forward read freed weights -> segfault ("peer closed cleanly"). 720p-flat
        // never hit it (no hires -> takes the correctly-gated :9292 branch), which is why only the
        // hires path died. Keep the DiT host-resident across the seam like the non-hires path.
        if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        if (final_latent.empty()) {
            LOG_ERROR("%s failed after %.2fs",
                      same_res_refine_enabled ? "sampling(same-res refine)" : "sampling(latent upscale)",
                      (sampling_end - sampling_start) * 1.0f / 1000);
            return false;
        }
        LOG_INFO("%s completed, taking %.2fs",
                 same_res_refine_enabled ? "sampling(same-res refine)" : "sampling(latent upscale)",
                 (sampling_end - sampling_start) * 1.0f / 1000);
    } else if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }

    // Dual-resolution continuation: export the STAGE-0 (pre stage-1 upscale) refined VIDEO tail so
    // the NEXT segment's stage-0 refine gets a MATCHING-RESOLUTION continuation guide (the post-loop
    // refined_latent_out export below only captures the final full-res stage-1 tail). Operate on a
    // COPY — the stage-1 loop still consumes final_latent unchanged. Gated on hires_chain_count>=2 so
    // the single-stage (2x) path never writes _lo (byte-identical). Mirrors the strip+export at the
    // refined_latent_out path: drop packed audio channels, then trailing guide frames, then leading
    // ref-image frames, then malloc+memcpy the video-only tail.
    if (refined_latent_lo_out != nullptr && latent_upscale_enabled &&
        sd_vid_gen_params->hires_chain_count >= 2 && !final_latent.empty()) {
        const int64_t video_ch = sd_ctx->sd->get_latent_channel();
        sd::Tensor<float> lo    = final_latent;
        if (lo.dim() > 3 && lo.shape()[3] > video_ch) {
            lo = sd::ops::slice(lo, 3, 0, video_ch);
        }
        if (!final_latent_prestripped && latents.video_conditioning_frame_count > 0) {
            int64_t target_frames = latents.video_target_frame_count > 0
                                        ? latents.video_target_frame_count
                                        : lo.shape()[2] - latents.video_conditioning_frame_count;
            lo = sd::ops::slice(lo, 2, 0, target_frames);
        }
        if (!final_latent_prestripped && latents.ref_image_num > 0) {
            lo = sd::ops::slice(lo, 2, latents.ref_image_num, lo.shape()[2]);
        }
        if (lo.dim() > 3 && lo.shape()[3] == video_ch) {
            const int64_t Wl = lo.shape()[0];
            const int64_t Hl = lo.shape()[1];
            const int64_t Tl = lo.shape()[2];
            float* buf = (float*)malloc((size_t)lo.numel() * sizeof(float));
            if (buf != nullptr) {
                std::memcpy(buf, lo.data(), (size_t)lo.numel() * sizeof(float));
                *refined_latent_lo_out = buf;
                if (refined_latent_lo_width_out) *refined_latent_lo_width_out = (int)Wl;
                if (refined_latent_lo_height_out) *refined_latent_lo_height_out = (int)Hl;
                if (refined_latent_lo_frames_out) *refined_latent_lo_frames_out = (int)Tl;
                if (refined_latent_lo_channels_out) *refined_latent_lo_channels_out = (int)video_ch;
                LOG_INFO("LTX hires continuation: exported STAGE-0 refined video latent [%lld,%lld,%lld,%lld]",
                         (long long)Wl, (long long)Hl, (long long)Tl, (long long)video_ch);
            }
        }
    }

    // MID stage preview (stage_scale 2): the STAGE-0 (base*2, e.g. 960) refined latent, decoded after
    // the first refine but BEFORE the next hires-chain upscale — the intermediate sharpen. Only fired
    // on a genuine >=2-stage chain (else the first refine IS the final = the per-segment output, no
    // intermediate). At this point final_latent is the STAGE-0 refined latent. No-op unless emit_stages.
    if (emit_stages_on && latent_upscale_enabled && sd_vid_gen_params->hires_chain_count >= 2) {
        emit_stage_preview(2, final_latent);
    }

    // Stages after stage 0 intentionally use the same AV packing discipline as the
    // established refine path: upscale VIDEO while audio is separate, re-pack it,
    // then build a fresh audio-pinned SDEdit mask for that stage. The old single-hires
    // branch above remains entirely untouched when hires_chain is absent/empty.
    if (hires_chain_enabled) {
        for (int stage_index = 1; stage_index < sd_vid_gen_params->hires_chain_count; ++stage_index) {
            const sd_hires_params_t& stage = sd_vid_gen_params->hires_chain[stage_index];
            // Each later SDEdit stage is a same-model transition.  Release the
            // prior stage's full DiT residency before reserving the larger graph.
            if (sd_ctx->sd->diffusion_model) {
                sd_ctx->sd->diffusion_model->set_refine_resident_scope(true);
                if (sd_ctx->sd->diffusion_model->params_offloaded_to_host()) {
                    sd_ctx->sd->diffusion_model->release_all_gpu_param_residency();
                    ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
                    LOG_INFO("LTXAV hires-chain handoff: released full DiT GPU residency before stage %d bounded reuse",
                             stage_index);
                }
            }
            sd::Tensor<float> stage_latent = upscale_ltx_spatial_video_latent(sd_ctx, stage.model_path,
                                                                                final_latent, latents.audio_length);
            if (stage_latent.empty()) {
                LOG_ERROR("LTX hires_chain stage %d upscale failed", stage_index);
                return false;
            }
            // The spatial upscaler has re-offloaded the VAE for its encode pass.
            // Refinement is latent-only, so that GPU residency is dead until final
            // decode and must not coexist with the high-resolution DiT graph.
            if (ltxav_vae_lazy &&
                sd_version_is_ltxav(sd_ctx->sd->version) &&
                sd_ctx->sd->first_stage_model &&
                sd_ctx->sd->first_stage_model->params_offloaded_to_host()) {
                sd_ctx->sd->first_stage_model->release_all_gpu_param_residency();
                if (sd_ctx->sd->audio_vae_model) {
                    sd_ctx->sd->audio_vae_model->release_all_gpu_param_residency();
                }
                ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::VAE));
                ggml_backend_cuda_release_cudnn_conv3d_weights();
                LOG_INFO("LTXAV hires-chain stage %d: released VAE GPU residency before latent refine",
                         stage_index);
            }
            if (sd_ctx->sd->diffusion_model) {
                sd_ctx->sd->diffusion_model->free_streaming_scratch_buffers();
                ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
            }
            GenerationRequest stage_request = hires_request;
            stage_request.hires = stage;
            stage_request.width = (int)stage_latent.shape()[0] * stage_request.vae_scale_factor;
            stage_request.height = (int)stage_latent.shape()[1] * stage_request.vae_scale_factor;
            const int T_stage = (int)stage_latent.shape()[2];
            int scheduler_steps = 0;
            std::vector<float> sigmas = make_hires_sigma_schedule(
                sd_ctx, stage, sd_vid_gen_params->sample_params, stage.sample_method, stage.steps,
                sd_ctx->sd->get_image_seq_len(stage_request.height, stage_request.width) * T_stage, &scheduler_steps);
            if (sigmas.size() < 2) {
                LOG_ERROR("LTX hires_chain stage %d has no usable SDEdit sigma schedule", stage_index);
                return false;
            }
            sd_guidance_params_t guidance = sd_vid_gen_params->sample_params.guidance;
            guidance.txt_cfg = stage.cfg;
            const bool use_uncond = request.use_uncond || stage.cfg > 1.f ||
                                    stage.sample_method == EULER_CFG_PP_SAMPLE_METHOD ||
                                    stage.sample_method == EULER_A_CFG_PP_SAMPLE_METHOD;
            sd::Tensor<float> video = stage_latent;
            sd::Tensor<float> audio;
            const int latent_channels = sd_ctx->sd->get_latent_channel();
            if (latents.audio_length > 0 && stage_latent.shape()[3] > latent_channels) {
                video = sd::ops::slice(stage_latent, 3, 0, latent_channels);
                audio = unpack_ltxav_audio_latent(stage_latent, latents.audio_length, latent_channels);
                if (audio.empty()) {
                    LOG_ERROR("LTX hires_chain stage %d could not separate audio latent", stage_index);
                    return false;
                }
            }
            sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video, 1.f);
            sd::Tensor<float> mask = audio.empty()
                                        ? video_mask
                                        : pack_ltxav_audio_and_video_denoise_mask(video_mask, video, audio,
                                                                                   latents.audio_fixed ? 0.f : 1.f);
            sd::Tensor<float> stage_reference;
            sd::Tensor<float> stage_positions;
            // #4 (opt-in, default off): re-pin the source image identity at THIS stage's FULL
            // resolution so the final upscale anchors on the hi-res image instead of rerolling it
            // from the low-res base pass. Mirrors stage 0 — apply_ltxv_refine_image_conditioning
            // re-encodes init/end/keyframe at the stage res and freezes those frames in the mask.
            // Only for real image-guide shots (i2v opener / scene cut / keyframes); relip and
            // continuation-only shots keep the all-ones mask. OFF = historical byte-identical path.
            if (ltxav_refine_hires_identity_enabled() && !latents.relip_twostage &&
                (sd_vid_gen_params->init_image.data != nullptr ||
                 sd_vid_gen_params->end_image.data != nullptr ||
                 (sd_vid_gen_params->keyframes != nullptr && sd_vid_gen_params->keyframes_size > 0))) {
                sd::Tensor<float> repin_latent = stage_latent;
                sd::Tensor<float> repin_mask;
                sd::Tensor<float> repin_positions;
                sd::Tensor<float> repin_reference;
                if (!apply_ltxv_refine_image_conditioning(sd_ctx, sd_vid_gen_params, stage_request, latents,
                                                          &repin_latent, &repin_mask, &repin_positions,
                                                          &repin_reference)) {
                    return false;
                }
                if (!repin_mask.empty()) {
                    stage_latent    = std::move(repin_latent);
                    mask            = std::move(repin_mask);
                    stage_positions = std::move(repin_positions);
                    stage_reference = std::move(repin_reference);
                    LOG_INFO("LTX hires_chain stage %d: re-pinned hi-res identity at %dx%d",
                             stage_index, stage_request.width, stage_request.height);
                }
            }
            // Dual-resolution continuation (this hires stage): attach the prior segment's REFINED
            // full-resolution VIDEO tail (cont_refine_latent — the final 4x tail matches THIS stage's
            // grid) as a separate guide token block so the final upscale anchors the character on its
            // own prior appearance instead of rerolling it. Matching-resolution counterpart of the
            // stage-0 apply; runs before the identity reattach below so the two COMPOSE. Only for a
            // continuation-only shot (no image re-pin: gated on init/end==null and an empty
            // stage_reference/stage_positions left by the #4 block above).
            if (ltxav_chain_hires_reference_enabled() && !latents.relip_twostage &&
                sd_vid_gen_params->cont_refine_latent != nullptr &&
                sd_vid_gen_params->cont_refine_latent_frames > 0 &&
                sd_vid_gen_params->init_image.data == nullptr &&
                sd_vid_gen_params->end_image.data == nullptr &&
                stage_reference.empty() && stage_positions.empty()) {
                const int64_t video_ch = sd_ctx->sd->get_latent_channel();
                const int64_t target_w = stage_latent.shape()[0];
                const int64_t target_h = stage_latent.shape()[1];
                const int64_t target_t = stage_latent.shape()[2];
                const int64_t guide_t  = sd_vid_gen_params->cont_refine_latent_frames;
                const bool shape_ok = sd_vid_gen_params->cont_refine_latent_width == target_w &&
                                      sd_vid_gen_params->cont_refine_latent_height == target_h &&
                                      sd_vid_gen_params->cont_refine_latent_channels == video_ch &&
                                      guide_t <= target_t;
                if (!shape_ok) {
                    LOG_WARN("LTX hires_chain stage %d continuation reference skipped: prior [%d,%d,%d,%d] vs target [%lld,%lld,%lld,%lld]",
                             stage_index,
                             sd_vid_gen_params->cont_refine_latent_width,
                             sd_vid_gen_params->cont_refine_latent_height,
                             sd_vid_gen_params->cont_refine_latent_frames,
                             sd_vid_gen_params->cont_refine_latent_channels,
                             (long long)target_w, (long long)target_h, (long long)target_t, (long long)video_ch);
                } else {
                    stage_reference = sd::Tensor<float>({target_w, target_h, guide_t, video_ch, 1});
                    std::memcpy(stage_reference.data(), sd_vid_gen_params->cont_refine_latent,
                                (size_t)stage_reference.numel() * sizeof(float));
                    stage_positions = build_ltxv_video_positions(target_w, target_h, target_t, guide_t,
                                                                   /*keyframe_frame_idx*/ 0,
                                                                   /*keyframe_pixel_frames*/ 8,
                                                                   stage_request.fps,
                                                                   stage_request.vae_scale_factor,
                                                                   8,
                                                                   true);
                    LOG_INFO("LTX hires_chain stage %d continuation reference: %lld refined VIDEO tail frames as separate guide tokens; target=%lld + guide=%lld frames",
                             stage_index, (long long)guide_t, (long long)target_t, (long long)guide_t);
                }
            }
            // Keep the identity-only DiT reference on every chain stage. It is a separate token
            // block (never an output-frame pin), matching stage 0; it COMPOSES onto any hi-res
            // re-pin positions/reference or continuation guide set just above (the fixed append
            // helper concatenates the identity rows, it never clobbers an existing target/guide layout).
            if (ltxav_character_ref_enabled() &&
                sd_vid_gen_params->character_reference_latent != nullptr) {
                // Later hires_chain stages (final res): attach the final-res (_hi) identity block so
                // the highest-resolution refine anchors on a full-detail character instead of the
                // base-res hint; falls back to the base-res latent when no _hi was supplied.
                sd::Tensor<float> character = ltxav_character_latent_for_stage(sd_vid_gen_params, LtxavCharTier::Hi);
                ImageGenerationLatents refs;
                refs.init_latent     = stage_latent;
                refs.video_reference = std::move(stage_reference);
                refs.video_positions = std::move(stage_positions);
                if (!append_ltxav_character_reference(&refs, character, stage_request.fps,
                                                      stage_request.vae_scale_factor, 8)) {
                    return false;
                }
                stage_reference = std::move(refs.video_reference);
                stage_positions = std::move(refs.video_positions);
            }
            // LTXAV_PIN_REFINE_OVERLAP (this hires stage): identical mechanism to stage 0, against
            // the FULL-RESOLUTION refined tail (cont_refine_latent) whose grid matches THIS stage's —
            // the same source the stage's guide block just above uses. K is a LATENT-FRAME count, so
            // it is unchanged by this stage's doubled [W,H]; the tail TENSOR must match the stage's
            // grid, which the helper's shape_ok check enforces (WARN + skip, never corrupt).
            // Unlike stage 0 the mask here always exists (built at :12268), so this is the
            // packed/video-only mask path, never the created path.
            if (ltxav_pin_refine_overlap_enabled() && sd_version_is_ltxav(sd_ctx->sd->version) &&
                !latents.relip_twostage &&
                sd_vid_gen_params->cont_refine_latent != nullptr &&
                sd_vid_gen_params->cont_refine_latent_frames > 0 &&
                sd_vid_gen_params->init_image.data == nullptr &&
                sd_vid_gen_params->end_image.data == nullptr &&
                (sd_vid_gen_params->keyframes == nullptr || sd_vid_gen_params->keyframes_size == 0)) {
                char stage_label[32];
                snprintf(stage_label, sizeof(stage_label), "stage %d", stage_index);
                ltxav_pin_refine_overlap(&stage_latent,
                                         &mask,
                                         sd_vid_gen_params->cont_refine_latent,
                                         sd_vid_gen_params->cont_refine_latent_frames,
                                         sd_vid_gen_params->cont_refine_latent_width,
                                         sd_vid_gen_params->cont_refine_latent_height,
                                         sd_vid_gen_params->cont_refine_latent_channels,
                                         latent_channels,
                                         sd_vid_gen_params->cont_latent_frames,
                                         latents.video_target_frame_count,
                                         stage_label);
            }
            // LTXAV_SHARED_REFINE_NOISE (see stage 0): the offset is in LATENT FRAMES, which are
            // temporal and therefore resolution-independent — this stage's [W,H] have doubled since
            // stage 0 but its frame t is still the same absolute timeline position, so the SAME
            // offset applies unchanged. Per-frame draws scale with this stage's plane size.
            sd::Tensor<float> noise;
            if (ltxav_shared_refine_noise_enabled() && sd_version_is_ltxav(sd_ctx->sd->version) &&
                stage_latent.dim() == 5 && stage_latent.shape()[2] > 0 &&
                stage_latent.shape()[3] >= latent_channels) {
                const int64_t T_all = stage_latent.shape()[2];
                const int64_t T_tgt = latents.video_target_frame_count > 0
                                          ? std::min<int64_t>(latents.video_target_frame_count, T_all)
                                          : T_all;
                const uint64_t pseed = ltxav_shared_refine_noise_seed();
                noise = positional_randn_like(stage_latent, sd_ctx->sd->rng, pseed,
                                              sd_vid_gen_params->chain_latent_offset, latent_channels, T_tgt);
                LOG_INFO("LTXAV shared refine noise (stage %d): abs latent offset %lld, target=%lld guide=%lld frames, seed=%llu",
                         stage_index, (long long)sd_vid_gen_params->chain_latent_offset, (long long)T_tgt,
                         (long long)(T_all - T_tgt), (unsigned long long)pseed);
            } else {
                noise = sd::Tensor<float>::randn_like(stage_latent, sd_ctx->sd->rng);
            }
            LOG_INFO("LTX hires_chain stage %d: upscale -> SDEdit sigma0=%.5f, steps=%d, sampler=%s, cfg=%.3f",
                     stage_index, sigmas.front(), scheduler_steps, sampling_methods_str[stage.sample_method], stage.cfg);
            final_latent = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model, true, stage_latent, std::move(noise),
                                              embeds.cond, use_uncond ? embeds.uncond : SDCondition(), embeds.img_uncond,
                                              sd::Tensor<float>(), 0.f, guidance,
                                              resolve_eta(sd_ctx, sd_vid_gen_params->sample_params.eta, stage.sample_method),
                                              sd_vid_gen_params->sample_params.shifted_timestep, stage.sample_method,
                                              sd_ctx->sd->is_flow_denoiser(), plan.extra_sample_args, sigmas,
                                              std::vector<sd::Tensor<float>>{}, false, mask, sd::Tensor<float>(),
                                              stage_request.vace_strength, latents.audio_length, (float)stage_request.fps,
                                              stage_request.cache_params, stage_positions, latents.audio_positions,
                                              latents.audio_fixed, stage_reference);
            if (final_latent.empty()) {
                LOG_ERROR("LTX hires_chain stage %d refine failed", stage_index);
                return false;
            }
            hires_request = std::move(stage_request);
        }
        if (sd_ctx->sd->diffusion_model) {
            sd_ctx->sd->diffusion_model->set_refine_resident_scope(false);
        }
    }

    int64_t latent_end = ggml_time_ms();
    LOG_INFO("generating latent video completed, taking %.2fs", (latent_end - latent_start) * 1.0f / 1000);

    // LTXAV_DIT_FREE_DURING_DECODE (opt-in, default off): symmetric counterpart to LTXAV_VAE_LAZY.
    // All DiT sampling (base + latent-upscale refine) is now done; nothing below — the audio-VAE
    // decode, the continuation-latent copy, and the video-VAE decode_video_outputs — touches the
    // diffusion model. Yet on a warm resident chain (keep_diffusion_model_resident) the DiT free at
    // :9089/:9097 is intentionally skipped, so its ~5471 MB stays GPU-resident and squats through the
    // VAE decode — the render's TRUE VRAM peak. That resident weight lives in the DiT's cross-step
    // shared-resident buffer (LONGCAT_SHARED_RESIDENT), which neither set_keep_params_resident(false)
    // nor free_params_buffer() releases. release_all_gpu_param_residency() genuinely frees it (runtime +
    // shared-resident -> cudaFree) while leaving the CPU/mmap home intact, so a chained next generate's
    // sample() re-offloads it via execute_graph — no reload loader, mmap-safe. Gated on the DiT being
    // host-offloaded (params_offloaded_to_host); if params live directly on the GPU there is no
    // re-offload path, so we skip to stay abort-safe (the DiT is not needed for decode either way).
    static const bool ltxav_dit_free_during_decode = [] {
        const char* s = getenv("LTXAV_DIT_FREE_DURING_DECODE");
        return s && s[0] == '1';
    }();
    if (ltxav_dit_free_during_decode &&
        sd_version_is_ltxav(sd_ctx->sd->version) &&
        sd_ctx->sd->diffusion_model) {
        int64_t phase_t0 = ggml_time_ms();
        if (sd_ctx->sd->diffusion_model->params_offloaded_to_host()) {
            sd_ctx->sd->diffusion_model->release_all_gpu_param_residency();
            // Trim the DIFFUSION backend pool so the freed VRAM leaves the board as real headroom for
            // the VAE decode peak (mirrors the pre-sample DIFFUSION trim + the LTXAV_VAE_LAZY VAE trim).
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::DIFFUSION));
            // FIX 2 (belt-and-suspenders): also trim the VAE backend's pool. On the single-GPU default
            // recipe VAE and DIFFUSION resolve to the same cached cuda0 backend, so this is a no-op
            // (the pool was just trimmed); if a recipe ever splits the VAE onto its own backend it
            // reclaims that pool's committed high-water (e.g. the continuation reference-encode /
            // prior-segment decode scratch) which the DIFFUSION-only trim can't reach. Harmless +
            // byte-identical either way.
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::VAE));
            LOG_INFO("LTXAV_DIT_FREE_DURING_DECODE: released offloaded DiT GPU params (runtime + shared-resident + streaming/prefetch buffers) + trimmed DIFFUSION+VAE pools before decode; a chained next generate re-offloads from host");
        } else {
            LOG_INFO("LTXAV_DIT_FREE_DURING_DECODE: DiT params live on the runtime backend (no host offload); skipping to stay abort-safe");
        }
        LOG_INFO("[LTX_PHASE] pre-decode DiT release took %.3fs", (ggml_time_ms() - phase_t0) * 1.0f / 1000);
    }

    sd_audio_t* generated_audio = nullptr;
    static const bool ltxav_skip_audio_decode = [] {
        const char* s = getenv("LTXAV_SKIP_AUDIO_DECODE");
        return s && s[0] == '1';
    }();
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        latents.audio_length > 0 &&
        sd_ctx->sd->audio_vae_model != nullptr) {
        int64_t audio_latent_decode_start = ggml_time_ms();

        if (ltxav_skip_audio_decode) {
            LOG_INFO("LTXAV_SKIP_AUDIO_DECODE=1: skipping generated audio latent decode");
        } else {
            int64_t audio_vae_reload_start = ggml_time_ms();
            // FIX 3: bring the audio VAE back if it was freed for the stage-2 forward (no-op if resident).
            sd_ctx->sd->reload_audio_vae_model();
            LOG_INFO("[LTX_PHASE] audio VAE reload before decode took %.3fs", (ggml_time_ms() - audio_vae_reload_start) * 1.0f / 1000);

            auto audio_latent = unpack_ltxav_audio_latent(final_latent,
                                                          latents.audio_reference_conditioning
                                                              ? latents.audio_target_length
                                                              : latents.audio_length,
                                                          sd_ctx->sd->get_latent_channel());
            if (!audio_latent.empty()) {
                LOG_DEBUG("decode audio latent %dx%dx%dx%d",
                          (int)audio_latent.shape()[0],
                          (int)audio_latent.shape()[1],
                          (int)audio_latent.shape()[2],
                          (int)audio_latent.shape()[3]);
                auto waveform = sd_ctx->sd->decode_ltx_audio_latent(audio_latent);
                if (!waveform.empty()) {
                    generated_audio = waveform_to_sd_audio(sd_ctx->sd, waveform);
                } else {
                    LOG_WARN("LTX audio latent decode failed; continuing with silent video output");
                }
            }
        }
        int64_t audio_latent_decode_end = ggml_time_ms();
        LOG_INFO("decoding audio latent completed, taking %.2fs", (audio_latent_decode_end - audio_latent_decode_start) * 1.0f / 1000);

        // Audio decoding is complete before the much larger video-VAE decode starts.  In a
        // warm chain, keep_diffusion_model_resident deliberately keeps the audio VAE's
        // host-backed GPU residency alive for the next segment.  That leaves ~350 MB of
        // otherwise unused audio-VAE weights resident beside the video decode's peak buffer.
        // Unlike free_params_buffer(), this only releases the GPU copy: the mmap/host home
        // remains intact and the next segment re-offloads it on demand.  This is allocation
        // only, so it cannot change the generated waveform or video.
        if (sd_ctx->sd->audio_vae_model->params_offloaded_to_host()) {
            const double audio_vae_gpu_mib =
                sd_ctx->sd->audio_vae_model->gpu_footprint_bytes() / (1024.0 * 1024.0);
            sd_ctx->sd->audio_vae_model->release_all_gpu_param_residency();
            LOG_INFO("LTXAV: released audio VAE GPU residency (%.0f MiB) before video decode",
                     audio_vae_gpu_mib);
        }
    }

    // Avatar: mux the INPUT conditioning audio back into the output container,
    // trimmed to the generated video's duration so audio/video stay in sync.
    if (sd_version_is_longcat_avatar(sd_ctx->sd->version) && !avatar_input_wav.empty()) {
        const uint32_t in_sr = 16000;  // load_wav_16k_mono always returns 16k mono
        int out_fps          = request.fps > 0 ? request.fps : 25;
        // For a continuation segment, the muxed audio starts at the segment's
        // offset in the global timeline (audio_frame_offset @ 25fps).
        double seg_start_dur = (double)sd_vid_gen_params->audio_frame_offset / 25.0;
        size_t start_off     = (size_t)(seg_start_dur * (double)in_sr);
        if (start_off > avatar_input_wav.size()) {
            start_off = avatar_input_wav.size();
        }
        // request.frames is the requested video length; clamp the audio to it.
        double video_dur   = (double)request.frames / (double)out_fps;
        size_t want        = (size_t)(video_dur * (double)in_sr);
        size_t n           = avatar_input_wav.size() - start_off;
        if (want > 0 && want < n) {
            n = want;  // trim trailing audio beyond the rendered video
        }
        sd_audio_t* a = (sd_audio_t*)malloc(sizeof(sd_audio_t));
        if (a != nullptr) {
            a->sample_rate  = in_sr;
            a->channels     = 1;
            a->sample_count = (uint64_t)n;
            a->data         = (float*)malloc(n * sizeof(float));
            if (a->data != nullptr) {
                std::memcpy(a->data, avatar_input_wav.data() + start_off, n * sizeof(float));
                generated_audio = a;
                LOG_INFO("avatar: muxing input audio (%zu samples @ %u Hz, %.2fs) into output", n, in_sr, (double)n / (double)in_sr);
            } else {
                free(a);
            }
        }
    }

    if (!final_latent_prestripped && latents.video_conditioning_frame_count > 0) {
        int64_t target_frames = latents.video_target_frame_count > 0 ? latents.video_target_frame_count
                                                                     : final_latent.shape()[2] - latents.video_conditioning_frame_count;
        final_latent          = sd::ops::slice(final_latent, 2, 0, target_frames);
        // mirror the same temporal trim on the base continuation latent (same Tl)
        if (!chain_base_latent.empty()) {
            chain_base_latent = sd::ops::slice(chain_base_latent, 2, 0, target_frames);
        }
    }

    if (!final_latent_prestripped && latents.ref_image_num > 0) {
        final_latent = sd::ops::slice(final_latent, 2, latents.ref_image_num, final_latent.shape()[2]);
        if (!chain_base_latent.empty()) {
            chain_base_latent = sd::ops::slice(chain_base_latent, 2, latents.ref_image_num, chain_base_latent.shape()[2]);
        }
    }

    // Test-only final temporal interpolation.  Keep this after all sampling/refine
    // bookkeeping so its changed timeline never reaches RoPE, masks, or continuation
    // state. Audio is decoded before this stage and keeps its duration while the video
    // switches to twice the presentation fps.
    bool temporal_upscale_applied = false;
    bool temporal_upscale_audio_present = false;
    const char* temporal_upscale_env = std::getenv("LTXAV_TEMPORAL_UPSCALE");
    if (temporal_upscale_env != nullptr && temporal_upscale_env[0] != '\0' &&
        std::string(temporal_upscale_env) != "0" && std::string(temporal_upscale_env) != "false") {
        const int64_t video_channels = sd_ctx->sd->get_latent_channel();
        const bool packed_audio_present = latents.audio_length > 0 &&
                                          final_latent.dim() > 3 &&
                                          final_latent.shape()[3] > video_channels;
        const bool continuation_active = final_latent_out != nullptr ||
                                         sd_vid_gen_params->cont_latent != nullptr ||
                                         sd_vid_gen_params->cont_refine_latent != nullptr ||
                                         sd_vid_gen_params->cont_refine_latent_lo != nullptr ||
                                         sd_vid_gen_params->end_cont_latent != nullptr;
        if (!sd_version_is_ltxav(sd_ctx->sd->version)) {
            LOG_INFO("LTX temporal upscale skipped: non-LTXAV request");
        } else if (continuation_active) {
            LOG_INFO("LTX temporal upscale skipped: continuation/chain request unsupported in test build");
        } else {
            constexpr const char* temporal_model_path =
                "/models/ltx2/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors";
            const int64_t input_t = final_latent.shape()[2];
            sd::Tensor<float> video_latent = final_latent;
            if (packed_audio_present) {
                video_latent = sd::ops::slice(final_latent, 3, 0, video_channels);
                LOG_INFO("LTX temporal upscale: audio present; stripped packed audio channels and upscaling VIDEO latent only");
            }
            sd::Tensor<float> temporal_latent =
                upscale_ltx_temporal_video_latent(sd_ctx, temporal_model_path, video_latent);
            if (temporal_latent.empty()) {
                LOG_ERROR("LTX temporal upscale failed; leaving final latent unchanged");
            } else {
                final_latent = std::move(temporal_latent);
                temporal_upscale_applied = true;
                temporal_upscale_audio_present = packed_audio_present;
                // decode_video_outputs normally limits video to request.frames.
                // This test stage intentionally returns the interpolated timeline.
                if (latent_refine_enabled) {
                    hires_request.frames = 0;
                } else {
                    request.frames = 0;
                }
                if (temporal_upscale_audio_present) {
                    const int old_fps = effective_output_fps;
                    effective_output_fps *= 2;
                    if (output_fps != nullptr) {
                        *output_fps = effective_output_fps;
                    }
                    LOG_INFO("LTX temporal upscale: audio present; output fps %d -> %d to preserve audio/video duration",
                             old_fps, effective_output_fps);
                }
                LOG_INFO("LTX temporal upscale: latent T=%d -> %d",
                         (int)input_t, (int)final_latent.shape()[2]);
            }
        }
    }

    // Return a second, video-only continuation state for the next segment's hires refine.
    // The ordinary final_latent_out intentionally remains the BASE-grid state, because it is
    // sampled at base resolution on the next segment. The refined state is only meaningful
    // when a latent spatial upscale was performed; strip packed audio before exporting it.
    if (refined_latent_out != nullptr && latent_upscale_enabled && !final_latent.empty()) {
        const int64_t video_ch = sd_ctx->sd->get_latent_channel();
        sd::Tensor<float> refined_video = final_latent;
        if (refined_video.dim() > 3 && refined_video.shape()[3] > video_ch) {
            refined_video = sd::ops::slice(refined_video, 3, 0, video_ch);
        }
        if (refined_video.dim() > 3 && refined_video.shape()[3] == video_ch) {
            const int64_t Wl = refined_video.shape()[0];
            const int64_t Hl = refined_video.shape()[1];
            const int64_t Tl = refined_video.shape()[2];
            float* buf = (float*)malloc((size_t)refined_video.numel() * sizeof(float));
            if (buf != nullptr) {
                std::memcpy(buf, refined_video.data(), (size_t)refined_video.numel() * sizeof(float));
                *refined_latent_out = buf;
                if (refined_latent_width_out) *refined_latent_width_out = (int)Wl;
                if (refined_latent_height_out) *refined_latent_height_out = (int)Hl;
                if (refined_latent_frames_out) *refined_latent_frames_out = (int)Tl;
                if (refined_latent_channels_out) *refined_latent_channels_out = (int)video_ch;
                LOG_INFO("LTX hires continuation: exported refined video latent [%lld,%lld,%lld,%lld]",
                         (long long)Wl, (long long)Hl, (long long)Tl, (long long)video_ch);
            }
        }
    }

    // Continuation chaining: hand the post-sampling diffusion latent back to the
    // caller (before VAE decode) so the tail can condition the next segment without
    // a lossy decode/re-encode roundtrip. With hires on, hand back the BASE pre-upscale
    // latent (chain_base_latent) so the next segment chains at base resolution.
    const sd::Tensor<float>& cont_latent_src = (!chain_base_latent.empty()) ? chain_base_latent : final_latent;
    if (final_latent_out != nullptr && !cont_latent_src.empty()) {
        int64_t phase_t0 = ggml_time_ms();
        int64_t Wl = cont_latent_src.shape()[0];
        int64_t Hl = cont_latent_src.shape()[1];
        int64_t Tl = cont_latent_src.shape()[2];
        int64_t Cl = cont_latent_src.dim() > 3 ? cont_latent_src.shape()[3] : 1;
        size_t n   = (size_t)cont_latent_src.numel();
        float* buf = (float*)malloc(n * sizeof(float));
        if (buf != nullptr) {
            std::memcpy(buf, cont_latent_src.data(), n * sizeof(float));
            *final_latent_out = buf;
            if (latent_width_out) *latent_width_out = (int)Wl;
            if (latent_height_out) *latent_height_out = (int)Hl;
            if (latent_frames_out) *latent_frames_out = (int)Tl;
            if (latent_channels_out) *latent_channels_out = (int)Cl;
        }
        LOG_INFO("[LTX_PHASE] continuation latent host copy took %.3fs", (ggml_time_ms() - phase_t0) * 1.0f / 1000);
    }

    // FIX 3 (LTXAV two-stage) / WAN_VAE_FREE_DURING_DIT (CLI lever): bring the video VAE back if it
    // was freed to make DiT VRAM headroom. No-op (returns true) if still resident; the merged
    // reload_first_stage_model() dispatches to whichever reload loader was captured. On a genuine
    // reload FAILURE it has alloc'd but not refilled the params buffer, so abort here rather than
    // decode garbage (the pre-merge WAN-lever path checked this; the LTX path did not).
    int64_t video_vae_reload_start = ggml_time_ms();
    if (!sd_ctx->sd->reload_first_stage_model()) {
        LOG_ERROR("video VAE reload before decode failed");
        free_sd_audio(generated_audio);
        return false;
    }
    LOG_INFO("[LTX_PHASE] video VAE reload before decode took %.3fs", (ggml_time_ms() - video_vae_reload_start) * 1.0f / 1000);
    int64_t video_decode_start = ggml_time_ms();
    auto result = decode_video_outputs(sd_ctx, latent_refine_enabled ? hires_request : request, final_latent, num_frames_out);
    if (result == nullptr) {
        free_sd_audio(generated_audio);
        return false;
    }
    if (temporal_upscale_applied && num_frames_out != nullptr) {
        if (temporal_upscale_audio_present) {
            const double audio_duration = generated_audio != nullptr && generated_audio->sample_rate > 0
                                              ? (double)generated_audio->sample_count / generated_audio->sample_rate
                                              : 0.0;
            LOG_INFO("LTX temporal upscale: final frames=%d fps=%d audio_duration=%.3fs",
                     *num_frames_out, effective_output_fps, audio_duration);
        } else {
            LOG_INFO("LTX temporal upscale: final decoded frame count=%d", *num_frames_out);
        }
    }
    LOG_INFO("[LTX_PHASE] video decode outputs total took %.3fs", (ggml_time_ms() - video_decode_start) * 1.0f / 1000);

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_video completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    if (frames_out != nullptr) {
        *frames_out = result;
    }
    if (audio_out != nullptr) {
        *audio_out = generated_audio;
    } else {
        free_sd_audio(generated_audio);
    }
    return true;
}

SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out,
                           int* output_fps) {
    return generate_video_ex(sd_ctx, sd_vid_gen_params, frames_out, num_frames_out, audio_out, output_fps,
                             nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr);
}

// Decode a banked segment VIDEO latent (save_dir/seg_<i>.bin, written by the
// LTXAV_SAVE_VIDEO_LATENT path) back to pixel frames via the VAE, with NO DiT sampling.
// Used on resume to cheaply rebuild the already-rendered prefix. Returns a malloc'd
// sd_image_t array (caller frees each .data + the array), or nullptr on failure.
static sd_image_t* decode_banked_video_latent(sd_ctx_t* sd_ctx, const std::string& path, int* count_out) {
    if (count_out != nullptr) {
        *count_out = 0;
    }
    sd::Tensor<float> latent;
    try {
        latent = sd::load_tensor_from_file_as_tensor<float>(path);
    } catch (const std::exception& e) {
        LOG_ERROR("resume: failed to load banked latent %s: %s", path.c_str(), e.what());
        return nullptr;
    }
    if (latent.empty()) {
        LOG_ERROR("resume: banked latent %s is empty", path.c_str());
        return nullptr;
    }
    sd::Tensor<float> vid = sd_ctx->sd->decode_first_stage(latent, true);
    if (vid.empty()) {
        LOG_ERROR("resume: decode_first_stage failed for %s", path.c_str());
        return nullptr;
    }
    int         n    = (int)vid.shape()[2];
    sd_image_t* imgs = (sd_image_t*)calloc((size_t)n, sizeof(sd_image_t));
    if (imgs == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < n; ++i) {
        imgs[i] = tensor_to_sd_image(vid, i);
    }
    if (count_out != nullptr) {
        *count_out = n;
    }
    return imgs;
}

// Raw per-segment audio banking (header + planar [channel][sample] floats) so a resumed
// chain can reload the already-rendered prefix's audio — the VIDEO-only seg_<i>.bin latent
// can't carry it. Mirrors the sd_audio_t planar layout (waveform_to_sd_audio).
static bool write_seg_audio(const std::string& path, const sd_audio_t* a) {
    if (a == nullptr || a->data == nullptr || a->sample_count == 0) {
        return false;
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    uint32_t sr = a->sample_rate, ch = a->channels;
    uint64_t n  = a->sample_count;
    fwrite(&sr, sizeof(sr), 1, f);
    fwrite(&ch, sizeof(ch), 1, f);
    fwrite(&n, sizeof(n), 1, f);
    fwrite(a->data, sizeof(float), (size_t)n * ch, f);
    fclose(f);
    return true;
}
// RETAKE length-pinning: bank each rendered segment's KEPT frame count (post overlap-trim)
// next to its latent, so a later single-segment retake reproduces the exact timeline offsets
// (the content-adaptive seam auto-trim would otherwise re-derive a different drop against the
// re-rendered segment and shift every downstream segment -> audio/beat desync). The bidirectional
// end-pin makes the retaken segment land on the banked boundaries, so reusing the banked kept
// count is both length-stable AND seam-clean.
static void write_seg_len(const std::string& path, int kept) {
    FILE* f = fopen(path.c_str(), "w");
    if (f != nullptr) {
        fprintf(f, "%d\n", kept);
        fclose(f);
    }
}
static int read_seg_len(const std::string& path) {  // -1 = absent/unreadable (fall back to auto-trim)
    FILE* f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return -1;
    }
    int v = -1;
    if (fscanf(f, "%d", &v) != 1 || v < 0) {
        v = -1;
    }
    fclose(f);
    return v;
}
static sd_audio_t* read_seg_audio(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return nullptr;
    }
    uint32_t sr = 0, ch = 0;
    uint64_t n = 0;
    if (fread(&sr, sizeof(sr), 1, f) != 1 || fread(&ch, sizeof(ch), 1, f) != 1 ||
        fread(&n, sizeof(n), 1, f) != 1) {
        fclose(f);
        return nullptr;
    }
    sd_audio_t* a   = (sd_audio_t*)malloc(sizeof(sd_audio_t));
    a->sample_rate  = sr;
    a->channels     = ch;
    a->sample_count = n;
    size_t total    = (size_t)n * ch;
    a->data         = (float*)malloc(total * sizeof(float));
    if (a->data == nullptr || fread(a->data, sizeof(float), total, f) != total) {
        fclose(f);
        free(a->data);
        free(a);
        return nullptr;
    }
    fclose(f);
    return a;
}

// ── ENGINE-OWNED AUDIO helpers (chain_audio_full / chain_audio_track) ────────────────────
// The chain slices the caller's full-timeline clip itself instead of trusting a client to
// predict the seam trim. See sd_vid_chain_params_t::chain_audio_full for the why.

// Minimal 16 kHz mono PCM16 WAV writer. The core lib does NOT link the example layer's
// media_io (that's the whole point of the on_segment/on_flush_frames callbacks), so the
// per-segment drive slices get their own local writer — same precedent as write_seg_audio
// above. Writing a real file (rather than plumbing samples in-memory) keeps
// encode_ltxav_drive_audio's path-based contract untouched AND leaves the exact audio that
// drove each segment on disk next to its latent, which is a debugging affordance the
// pre-sliced path also had.
static bool write_wav_16k_mono(const std::string& path, const float* samples, size_t n) {
    if (samples == nullptr) {
        return false;
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        LOG_ERROR("chain audio: cannot write drive slice '%s'", path.c_str());
        return false;
    }
    const uint32_t sr        = 16000;
    const uint16_t ch        = 1;
    const uint16_t bits      = 16;
    const uint32_t data_len  = (uint32_t)(n * 2);
    const uint32_t riff_len  = 36 + data_len;
    const uint32_t byte_rate = sr * ch * (bits / 8);
    const uint16_t blk_align = (uint16_t)(ch * (bits / 8));
    const uint32_t fmt_len   = 16;
    const uint16_t fmt_tag   = 1;
    fwrite("RIFF", 1, 4, f);
    fwrite(&riff_len, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    fwrite(&fmt_len, 4, 1, f);
    fwrite(&fmt_tag, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&blk_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_len, 4, 1, f);
    for (size_t i = 0; i < n; ++i) {
        float   v = samples[i];
        int32_t s = (int32_t)llround(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        int16_t o = (int16_t)std::max(-32768, std::min(32767, s));
        fwrite(&o, 2, 1, f);
    }
    fclose(f);
    return true;
}

// The caller's full-timeline audio, loaded once and sliced per segment by the chain.
// `samples` is 16 kHz MONO for the drive clip (that's all the audio VAE encoder consumes),
// and VERBATIM interleaved for the deliverable track (rate/channels preserved so the Opus
// encode is the only generation loss).
struct ChainFullAudio {
    std::vector<float> samples;
    uint32_t           sample_rate = 0;
    uint32_t           channels    = 0;
    bool               loaded      = false;

    size_t frames() const {  // audio frames (sample groups), NOT video frames
        return (channels == 0) ? 0 : samples.size() / channels;
    }

    // Cut [start_frame, start_frame + n_frames) of the VIDEO timeline out of this clip,
    // given the clip's t=0 sits at `offset_frames` on that timeline. Reads out of range are
    // ZERO-FILLED rather than clamped: silence is the honest answer for "before the clip
    // starts" / "after it ends", whereas clamping would smear the first/last sample and a
    // short read would silently retime everything after it.
    std::vector<float> window(long long start_frame, long long n_frames, float fps, long long offset_frames) const {
        std::vector<float> out;
        if (!loaded || channels == 0 || n_frames <= 0 || fps <= 0.0f) {
            return out;
        }
        const long long total = (long long)frames();
        // Clip-local audio-frame range for this video-frame window.
        const long long a0 = llround((double)(start_frame - offset_frames) * (double)sample_rate / (double)fps);
        const long long a1 = llround((double)(start_frame - offset_frames + n_frames) * (double)sample_rate / (double)fps);
        const long long want = std::max(0LL, a1 - a0);
        out.assign((size_t)want * channels, 0.0f);
        for (long long i = 0; i < want; ++i) {
            const long long src = a0 + i;
            if (src < 0 || src >= total) {
                continue;  // zero-fill outside the clip
            }
            for (uint32_t c = 0; c < channels; ++c) {
                out[(size_t)i * channels + c] = samples[(size_t)src * channels + c];
            }
        }
        return out;
    }
};

// Stitches per-segment audio onto one continuous timeline.  LTXAV continuation appends
// its video guide as extra, cropped tokens, while the driven audio remains at target time
// zero; therefore the kept video window maps to the HEAD of each generated audio segment.
struct ChainAudioAcc {
    uint32_t           sample_rate = 0;
    uint32_t           channels    = 0;
    std::vector<float> frames;  // INTERLEAVED: [f0c0, f0c1, f1c0, f1c1, ...]

    // sd_audio_t.data is interleaved (channel-minor) — same layout the single-segment
    // path feeds the opus/pcm writers, which read interleaved. The old per-channel
    // (planar) accumulate/rebuild scrambled the stereo pair across segments, which is
    // why multi-segment renders sounded broken while single-segment was clean.
    void append_window(const sd_audio_t* a, int64_t start_samples, int64_t keep_samples) {
        if (a == nullptr || a->data == nullptr || a->sample_count == 0) {
            return;
        }
        if (channels == 0) {
            channels    = a->channels;
            sample_rate = a->sample_rate;
        }
        if (a->channels != channels) {
            LOG_WARN("chain audio: segment channel count %u != %u; skipping its audio", a->channels, channels);
            return;
        }
        const int64_t sc    = static_cast<int64_t>(a->sample_count);  // frames
        const int64_t start = std::clamp(start_samples, int64_t{0}, sc);
        const int64_t count = std::min(std::max(int64_t{0}, keep_samples), sc - start);
        frames.insert(frames.end(),
                      a->data + (size_t)start * channels,
                      a->data + (size_t)(start + count) * channels);
    }
    sd_audio_t* build() const {
        if (channels == 0 || frames.empty()) {
            return nullptr;
        }
        size_t      per = frames.size() / channels;  // frames
        sd_audio_t* out = (sd_audio_t*)malloc(sizeof(sd_audio_t));
        out->sample_rate  = sample_rate;
        out->channels     = channels;
        out->sample_count = per;
        out->data         = (float*)malloc(frames.size() * sizeof(float));
        if (out->data == nullptr) {
            free(out);
            return nullptr;
        }
        std::memcpy(out->data, frames.data(), frames.size() * sizeof(float));
        return out;
    }
};

// Seam auto-trim for keyframe-append continuation. The warm-up segment re-renders the overlap
// region then continues; the re-render is slightly phase-shifted from the prior segment's true
// tail, so a fixed overlap_px trim leaves a visible "skip" at the join. Search a small window of
// candidate trim points for the frame whose downsampled luma best matches the prior segment's
// last frame — that frame is the smoothest continuation. Returns the trim (drop) count. (A
// per-segment exposure match — seg N+1 renders ~0.85 luma darker — further flattens the residual
// tone step; validated offline, not yet ported here. See HANDOFF.)
static int ltxav_auto_trim_drop(const sd_image_t& prev_last, const sd_image_t* frames,
                                int n_frames, int overlap_px) {
    if (frames == nullptr || n_frames <= 0 || prev_last.data == nullptr) return overlap_px;
    int lo = std::max(1, overlap_px - 6);
    int hi = std::min(n_frames - 2, overlap_px + 14);
    if (hi <= lo) return std::min(overlap_px, std::max(0, n_frames - 1));
    const int GW = 32, GH = 18;
    auto grid = [](const sd_image_t& im, std::vector<float>& out, int gw, int gh) {
        out.assign((size_t)gw * gh, 0.f);
        int ch = (int)im.channel;
        for (int gy = 0; gy < gh; ++gy) {
            int sy = std::min((int)im.height - 1, (int)(((float)gy + 0.5f) * im.height / gh));
            for (int gx = 0; gx < gw; ++gx) {
                int sx = std::min((int)im.width - 1, (int)(((float)gx + 0.5f) * im.width / gw));
                const uint8_t* p = im.data + ((size_t)sy * im.width + sx) * ch;
                out[(size_t)gy * gw + gx] = (ch >= 3) ? (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2])
                                                      : (float)p[0];
            }
        }
    };
    std::vector<float> ref, cur;
    grid(prev_last, ref, GW, GH);
    int   best_t = overlap_px;
    float best   = 1e30f;
    for (int t = lo; t <= hi; ++t) {
        if (frames[t].data == nullptr) continue;
        grid(frames[t], cur, GW, GH);
        float mae = 0.f;
        for (size_t i = 0; i < ref.size(); ++i) mae += std::fabs(ref[i] - cur[i]);
        mae /= (float)ref.size();
        if (mae < best) { best = mae; best_t = t; }
    }
    return best_t;
}

// P1-C seam polish: per-segment exposure/tone match. A conditioned continuation segment renders
// ~0.85 luma darker than the prior segment, leaving a visible tone STEP at the join even after the
// auto-trim aligns content. Correct it with a gentle global per-channel gain+offset on the new
// segment so its colour stats match the prior segment's tail across the seam — a continuity
// correction, NOT a cross-fade (the whole segment shifts uniformly, so no doubling/ghost returns).
// Stats: last N kept frames of the accumulated output (pre) vs first N kept frames of the new
// segment (post). gain = pre_std/post_std clamped [0.9,1.1] (only correct drift, don't regrade);
// offset = pre_mean - gain*post_mean. Applied in-place to ALL `n_new` frames. Ported from the
// validated expo_match.py. It is opt-in: a whole-segment RGB regrade can itself be more visible
// than the seam it attempts to correct. Set LTXAV_EXPOSURE_MATCH=1 for the former behaviour.
static void ltxav_exposure_match(const sd_image_t* prev_tail, int n_prev,
                                 sd_image_t* new_frames, int n_new) {
    const char* enabled = getenv("LTXAV_EXPOSURE_MATCH");
    if (enabled == nullptr || enabled[0] == '\0' || std::string(enabled) == "0") return;
    if (prev_tail == nullptr || new_frames == nullptr || n_prev <= 0 || n_new <= 0) return;
    const int N  = 16;
    const int ch = (int)new_frames[0].channel;
    if (ch < 3) return;  // luma-only frames: nothing per-channel to match
    const int np = std::min(N, n_prev);
    const int ns = std::min(N, n_new);

    // accumulate per-channel mean and mean-of-squares over the two windows
    auto stats = [ch](const sd_image_t* fr, int base, int cnt, double* mean, double* var) {
        double sum[4] = {0,0,0,0}, sq[4] = {0,0,0,0};
        size_t npx = 0;
        for (int f = 0; f < cnt; ++f) {
            const sd_image_t& im = fr[base + f];
            if (im.data == nullptr) continue;
            size_t pix = (size_t)im.width * im.height;
            for (size_t p = 0; p < pix; ++p) {
                const uint8_t* px = im.data + p * (size_t)im.channel;
                for (int c = 0; c < ch; ++c) { double v = px[c]; sum[c] += v; sq[c] += v * v; }
            }
            npx += pix;
        }
        if (npx == 0) return false;
        for (int c = 0; c < ch; ++c) {
            mean[c] = sum[c] / (double)npx;
            double v = sq[c] / (double)npx - mean[c] * mean[c];
            var[c]  = v > 0.0 ? v : 0.0;
        }
        return true;
    };

    double pre_mean[4], pre_var[4], post_mean[4], post_var[4];
    if (!stats(prev_tail, n_prev - np, np, pre_mean, pre_var)) return;
    if (!stats(new_frames, 0, ns, post_mean, post_var)) return;

    float gain[4], off[4];
    for (int c = 0; c < ch; ++c) {
        double g = std::sqrt(pre_var[c]) / std::max(std::sqrt(post_var[c]), 1e-3);
        g        = std::min(1.1, std::max(0.9, g));  // gentle: only drift, no regrade
        gain[c]  = (float)g;
        off[c]   = (float)(pre_mean[c] - g * post_mean[c]);
    }
    LOG_INFO("generate_video_chain: exposure-match gain[%.3f %.3f %.3f] off[%.1f %.1f %.1f] "
             "(pre luma %.1f vs post %.1f)",
             gain[0], gain[1], gain[2], off[0], off[1], off[2],
             0.299 * pre_mean[0] + 0.587 * pre_mean[1] + 0.114 * pre_mean[2],
             0.299 * post_mean[0] + 0.587 * post_mean[1] + 0.114 * post_mean[2]);

    for (int f = 0; f < n_new; ++f) {
        sd_image_t& im = new_frames[f];
        if (im.data == nullptr) continue;
        size_t pix = (size_t)im.width * im.height;
        for (size_t p = 0; p < pix; ++p) {
            uint8_t* px = im.data + p * (size_t)im.channel;
            for (int c = 0; c < ch; ++c) {
                float v = gain[c] * (float)px[c] + off[c];
                px[c]   = (uint8_t)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v + 0.5f));
            }
        }
    }
}

// Seam cross-dissolve (env LTXAV_SEAM_CROSSFADE=<W frames>, default 0=off). The auto-trim leaves a
// residual "skip" at the join (seg N+1's re-rendered overlap phase-shifts from seg N's tail). Instead
// of a hard cut, fade the last W frames of `stitched` (seg N's tail) toward seg N+1's re-rendered
// overlap frames [drop-W, drop) — which cover the SAME timeline window and are about to be discarded —
// with a smootherstep alpha 0->1. Turns the content skip into a short dissolve. Frames must match dims.
static void ltxav_seam_crossfade(std::vector<sd_image_t>& stitched, const sd_image_t* seg_video,
                                 int drop, int W) {
    if (W <= 0 || seg_video == nullptr || (int)stitched.size() < W || drop < W) { return; }
    for (int j = 0; j < W; ++j) {
        sd_image_t&       dst = stitched[stitched.size() - (size_t)W + j];
        const sd_image_t& src = seg_video[drop - W + j];
        if (dst.data == nullptr || src.data == nullptr ||
            dst.width != src.width || dst.height != src.height || dst.channel != src.channel) { continue; }
        double x = (double)(j + 1) / (double)(W + 1);
        double a = x * x * x * (x * (6.0 * x - 15.0) + 10.0);  // smootherstep 0->1
        size_t n = (size_t)dst.width * dst.height * dst.channel;
        for (size_t p = 0; p < n; ++p) {
            dst.data[p] = (uint8_t)std::lround((1.0 - a) * (double)dst.data[p] + a * (double)src.data[p]);
        }
    }
}

// LATENT-SPACE per-channel affine continuity-match (anti-drift, pixel-free). The in-process
// chain carries a video-latent tail (cont_buf) from segment to segment; over a long chain the
// tail's per-channel colour/exposure statistics drift, and because each segment conditions on the
// PRIOR tail the drift compounds. These two helpers capture SEG-0's per-channel mean/std as the
// reference and remap each later segment's tail back onto those stats BEFORE it conditions the
// next segment, so drift can't accumulate — the latent analogue of continuity_match_segment
// (examples/cli/main.cpp), but on the 128 latent channels instead of RGB pixels (no VAE decode/
// re-encode). The tail layout is [Wl, Hl, Kf, Cv] contiguous (W fastest, channel slowest), so
// channel c occupies the contiguous block [c*Wl*Hl*Kf, (c+1)*Wl*Hl*Kf).
static void ltxav_compute_latent_channel_stats(const float*        buf,
                                               int                 Wl,
                                               int                 Hl,
                                               int                 Kf,
                                               int                 Cv,
                                               std::vector<float>& mean,
                                               std::vector<float>& std_out) {
    mean.assign(Cv, 0.f);
    std_out.assign(Cv, 0.f);
    const size_t block = (size_t)Wl * (size_t)Hl * (size_t)Kf;  // elements per channel
    if (block == 0) {
        return;
    }
    for (int c = 0; c < Cv; ++c) {
        const float* p = buf + (size_t)c * block;
        double       s = 0.0, s2 = 0.0;
        for (size_t i = 0; i < block; ++i) {
            double x = p[i];
            s += x;
            s2 += x * x;
        }
        double m   = s / (double)block;
        double var = s2 / (double)block - m * m;
        mean[c]    = (float)m;
        // Guard std with an epsilon (latents are unbounded; mirror the pixel version's shape).
        std_out[c] = (float)(var > 1e-8 ? std::sqrt(var) : 1e-4);
    }
}

static void ltxav_latent_channel_affine_match(float*                    tail,
                                              int                       Wl,
                                              int                       Hl,
                                              int                       Kf,
                                              int                       Cv,
                                              const std::vector<float>& ref_mean,
                                              const std::vector<float>& ref_std,
                                              float                     strength) {
    const size_t block = (size_t)Wl * (size_t)Hl * (size_t)Kf;  // elements per channel
    if (block == 0 || (int)ref_mean.size() < Cv || (int)ref_std.size() < Cv) {
        return;
    }
    float g0 = 0.f, b0 = 0.f;  // channel-0 sample for logging
    for (int c = 0; c < Cv; ++c) {
        float* p = tail + (size_t)c * block;
        // this segment's tail stats for channel c
        double s = 0.0, s2 = 0.0;
        for (size_t i = 0; i < block; ++i) {
            double x = p[i];
            s += x;
            s2 += x * x;
        }
        double tm  = s / (double)block;
        double tvar = s2 / (double)block - tm * tm;
        double tstd = tvar > 1e-8 ? std::sqrt(tvar) : 1e-4;
        // affine map onto the reference: out = ref_mean + (x - tail_mean)*(ref_std/tail_std),
        // blended toward identity by `strength` (mirrors continuity_match_segment exactly).
        double gain_raw = (double)ref_std[c] / tstd;
        double bias_raw = (double)ref_mean[c] - tm * gain_raw;
        double gain     = 1.0 + (double)strength * (gain_raw - 1.0);
        double bias     = (double)strength * bias_raw;
        if (c == 0) {
            g0 = (float)gain;
            b0 = (float)bias;
        }
        // No clamping: latents are unbounded (unlike uint8 pixels).
        for (size_t i = 0; i < block; ++i) {
            p[i] = (float)((double)p[i] * gain + bias);
        }
    }
    LOG_INFO("generate_video_chain: latent channel-affine continuity-match applied (strength=%.2f, ch0 gain=%.4f bias=%.4f)",
             strength, g0, b0);
}

SD_API bool generate_video_chain(sd_ctx_t*                    sd_ctx,
                                 const sd_vid_gen_params_t*   base_params,
                                 const sd_vid_chain_params_t* chain_params,
                                 sd_image_t**                 frames_out,
                                 int*                         num_frames_out,
                                 sd_audio_t**                 audio_out) {
    if (frames_out != nullptr) {
        *frames_out = nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = 0;
    }
    if (audio_out != nullptr) {
        *audio_out = nullptr;
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || base_params == nullptr || chain_params == nullptr ||
        frames_out == nullptr || num_frames_out == nullptr) {
        LOG_ERROR("generate_video_chain: null argument");
        return false;
    }

    const int n_chain = chain_params->n_segments;
    if (n_chain < 1) {
        LOG_ERROR("generate_video_chain: n_segments must be >= 1 (got %d)", n_chain);
        return false;
    }
    int K = std::max(1, chain_params->cont_latent_frames);  // overlap latent frames
    const char* hires_ref_env = std::getenv("LTXAV_CHAIN_HIRES_REFERENCE");
    const bool chain_hires_reference = hires_ref_env == nullptr ||
                                       (hires_ref_env[0] != '\0' && std::string(hires_ref_env) != "0");
    // Three full-resolution refined tail frames preserve identity over repeated seams. The
    // stage-2 base-guide frames are dropped when this transport is active, keeping the validated
    // 97-frame x2 chain under the 11.5 GiB envelope. Override for diagnostics/recovery.
    int hires_ref_K = std::min(3, K);
    if (const char* e = std::getenv("LTXAV_CHAIN_HIRES_REFERENCE_FRAMES")) {
        hires_ref_K = std::clamp(atoi(e), 1, K);
    }
    if (chain_hires_reference) {
        LOG_INFO("generate_video_chain: high-res stage-2 reference enabled (%d/%d tail frames)",
                 hires_ref_K, K);
    }

    // LTX causal-VAE temporal: K latent frames decode to 1+(K-1)*8 pixel frames. That is
    // the head overlap we drop on seg>0 (the prior tail's re-render).
    int overlap_px = 1 + (K - 1) * 8;
    if (overlap_px >= base_params->video_frames) {
        LOG_ERROR("generate_video_chain: cont_latent_frames (%d -> %d overlap pixel frames) must leave "
                  "new frames in video_frames (%d)",
                  K, overlap_px, base_params->video_frames);
        return false;
    }
    // LTXAV VIDEO latent channel count. generate_video_ex's returned latent packs AUDIO
    // into trailing channels (Cl_full > video_channels); the next segment is fed ONLY the
    // video channels. For LTXAV get_latent_channel()==128.
    const int LTXAV_VIDEO_LATENT_CHANNELS = 128;
    LOG_INFO("generate_video_chain: %d segments, K=%d overlap latent frames (%d overlap pixel frames dropped/seg>0)",
             n_chain, K, overlap_px);

    // Keep the DiT resident across segments (the TE is freed once by the GPU-TE deferred
    // flow; the DiT must persist or later segments render against freed GPU memory).
    sd_ctx_keep_diffusion_model_resident(sd_ctx, true);

    // Encode once for the whole chain; this frozen DiT-only block is not an image pin.
    // In addition to the base-res block for the base pass, encode the SAME reference image at the
    // refine stages' resolutions (_lo = first hires stage = base*2, _hi = final = base*2^n_stages)
    // so each SDEdit refine attaches a resolution-matched, higher-detail identity block instead of
    // upscaling the coarse base-res one. This MUST happen here: the VAE encoder is available at
    // chain level, but the per-segment LTXAV_VAE_LAZY path frees it before the DiT sample+refine
    // (so the refine stages cannot re-encode without a costly reload). The refine attach sites fall
    // back to the base-res latent when a tier is empty, so a failed/absent tier is non-fatal.
    sd::Tensor<float> chain_character_latent;
    sd::Tensor<float> chain_character_latent_lo;
    sd::Tensor<float> chain_character_latent_hi;
    const bool character_ref_enabled = sd_version_is_ltxav(sd_ctx->sd->version) &&
                                       ltxav_character_ref_enabled() &&
                                       base_params->character_reference.data != nullptr;
    if (character_ref_enabled) {
        if (sd_ctx->sd->vae_decode_only) { LOG_ERROR("LTXAV character reference requires VAE encoder weights"); return false; }
        auto image = sd_image_to_tensor(base_params->character_reference, base_params->width, base_params->height);
        chain_character_latent = encode_ltxav_condition_image(sd_ctx, image, "character reference");
        if (chain_character_latent.empty()) return false;
        LOG_INFO("LTXAV character reference: encoded once for %d chain segments", n_chain);
        // Every LTX latent-upscale refine stage is a fixed 2x, so stage 0 = base*2 and the final
        // hires_chain stage = base*2^n_stages (n_stages = hires_chain_count, or 1 for legacy single
        // hires). Aspect matches the base encode (sd_image_to_tensor rescales the same portrait).
        const int  hc            = base_params->hires_chain_count;
        const bool will_refine   = hc >= 1 || base_params->hires.enabled;
        const int  n_stages      = std::max(1, hc);
        const int  base_w        = base_params->width;
        const int  base_h        = base_params->height;
        if (will_refine) {
            auto lo_image          = sd_image_to_tensor(base_params->character_reference, base_w * 2, base_h * 2);
            chain_character_latent_lo = encode_ltxav_condition_image(sd_ctx, lo_image, "character reference (lo)");
            if (chain_character_latent_lo.empty()) {
                LOG_WARN("LTXAV character reference: stage-0 (base*2) encode failed; refine falls back to base-res identity");
            }
        }
        if (hc >= 2) {
            const int hi_scale     = 1 << n_stages;  // base*2^n_stages (2-stage chain -> 4)
            auto hi_image          = sd_image_to_tensor(base_params->character_reference, base_w * hi_scale, base_h * hi_scale);
            chain_character_latent_hi = encode_ltxav_condition_image(sd_ctx, hi_image, "character reference (hi)");
            if (chain_character_latent_hi.empty()) {
                LOG_WARN("LTXAV character reference: final-res (base*%d) encode failed; hires-chain refine falls back to base-res identity", hi_scale);
            }
        }
        LOG_INFO("LTXAV character reference: refine identity tiers lo=%lldx%lld hi=%lldx%lld (0x0 = fall back to base-res %lldx%lld)",
                 chain_character_latent_lo.empty() ? 0LL : (long long)chain_character_latent_lo.shape()[0],
                 chain_character_latent_lo.empty() ? 0LL : (long long)chain_character_latent_lo.shape()[1],
                 chain_character_latent_hi.empty() ? 0LL : (long long)chain_character_latent_hi.shape()[0],
                 chain_character_latent_hi.empty() ? 0LL : (long long)chain_character_latent_hi.shape()[1],
                 (long long)chain_character_latent.shape()[0], (long long)chain_character_latent.shape()[1]);
    }

    // Pre-encode EVERY distinct per-segment prompt in one text-encoder window so the
    // resident DiT chain runs uninterrupted (no per-segment gemma encode interleaved).
    {
        std::vector<std::string> eff_prompts;
        eff_prompts.reserve(n_chain);
        for (int seg = 0; seg < n_chain; ++seg) {
            const char* p = (chain_params->segment_prompts != nullptr && chain_params->segment_prompts[seg] != nullptr)
                                ? chain_params->segment_prompts[seg]
                                : base_params->prompt;
            eff_prompts.emplace_back(p != nullptr ? p : "");
        }
        std::vector<const char*> cptrs;
        cptrs.reserve(eff_prompts.size());
        for (const auto& s : eff_prompts) {
            cptrs.push_back(s.c_str());
        }
        GenerationRequest precompute_request(sd_ctx, base_params);
        bool need_precomputed_uncond = precompute_request.use_uncond ||
                                       precompute_request.use_high_noise_uncond;
        sd_ctx_precompute_chain_text_conds(
            sd_ctx, cptrs.data(), (int)cptrs.size(),
            base_params->negative_prompt != nullptr ? base_params->negative_prompt : "",
            base_params->clip_skip,
            need_precomputed_uncond);
    }

    // Prior segment's captured video-channel-only latent tail, ggml-ne order
    // [Wl, Hl, K, Cv] contiguous (W fastest, channel slowest), fed as cont_latent.
    std::vector<float>      cont_buf;
    int                     cont_Wl = 0, cont_Hl = 0, cont_Cv = 0;  // cont_buf spatial dims + channel count
    // Separate stage-2 appearance tail. Unlike cont_buf this stays on the refined spatial grid
    // and is never fed to the base sampler or latent-matched with the base transport.
    std::vector<float>      cont_refine_buf;
    int                     cont_refine_Wl = 0, cont_refine_Hl = 0, cont_refine_Cv = 0;
    // Parallel stage-0 (lower-res) appearance tail, fed to the next segment's FIRST hires stage so
    // its refine gets a matching-resolution continuation guide. Only captured on >=2-stage chains.
    std::vector<float>      cont_refine_lo_buf;
    int                     cont_refine_lo_Wl = 0, cont_refine_lo_Hl = 0, cont_refine_lo_Cv = 0;
    std::vector<sd_image_t> stitched;   // adopts each kept frame's .data (streaming: rolling window only)
    ChainAudioAcc           audio_acc;  // per-segment audio stitched onto one timeline

    // WINDOWED STREAMING FINALIZE (chain_params->on_flush_frames set). Instead of accumulating the
    // whole decoded timeline into `stitched` (≈31.5 GB for a 3.5-min 1080p chain → swap thrash), keep
    // only the last WINDOW_KEEP frames — the maximum a future segment's seam op can reach back into:
    //   exposure_match reads the last min(16, N) frames of `stitched`; crossfade mutates the last W.
    // Everything OLDER than that suffix is final forever, so it is flushed to the caller's encoder
    // (which frees it) as we go. Because the window is always a suffix of the full timeline, every
    // seam op sees byte-identical inputs to the non-streaming path. flush_window(false) drains all but
    // the tail; flush_window(true) drains everything. No-op unless streaming.
    const bool streaming = chain_params->on_flush_frames != nullptr;
    int        crossfade_W = 0;
    if (const char* e = getenv("LTXAV_SEAM_CROSSFADE")) {
        crossfade_W = std::max(0, atoi(e));
    }
    const int WINDOW_KEEP = std::max(16, crossfade_W);  // exposure_match N=16, crossfade W
    long long flushed_total = 0;

    // ── ENGINE-OWNED AUDIO: load the caller's full timeline ONCE ─────────────────────────
    // `chain_audio_full` drives lip-sync (16k mono — all the audio VAE takes); `chain_audio_track`
    // is the deliverable, kept verbatim. Same path for both = the common "voice drives AND
    // delivers" case, loaded twice at different fidelities on purpose.
    const long long audio_offset_frames =
        (chain_params->chain_audio_full != nullptr || chain_params->chain_audio_track != nullptr)
            ? (long long)chain_params->chain_audio_offset_frames
            : 0;
    ChainFullAudio drive_full;
    if (chain_params->chain_audio_full != nullptr && chain_params->chain_audio_full[0] != '\0') {
        if (LONGCAT_AUDIO::load_wav_16k_mono(chain_params->chain_audio_full, drive_full.samples) &&
            !drive_full.samples.empty()) {
            drive_full.sample_rate = 16000;
            drive_full.channels    = 1;
            drive_full.loaded      = true;
            LOG_INFO("generate_video_chain: engine-owned DRIVE audio '%s' (%zu frames @16k mono, timeline offset %lld)",
                     chain_params->chain_audio_full, drive_full.frames(), audio_offset_frames);
        } else {
            // Hard-fail rather than silently reverting to chain_audio_dir or to no lip-sync:
            // an unreadable clip that renders anyway burns a full chain and looks like a
            // model regression. (A missing per-segment aud_<i>.wav on the LEGACY path is
            // deliberately silent-not-fatal; this is a new, explicit contract.)
            LOG_ERROR("generate_video_chain: chain_audio_full '%s' unreadable (need RIFF/WAVE PCM16/PCM32/float)",
                      chain_params->chain_audio_full);
            return false;
        }
    }
    ChainFullAudio track_full;
    if (chain_params->chain_audio_track != nullptr && chain_params->chain_audio_track[0] != '\0') {
        if (LONGCAT_AUDIO::load_wav_full(chain_params->chain_audio_track, track_full.samples,
                                         track_full.sample_rate, track_full.channels) &&
            !track_full.samples.empty()) {
            track_full.loaded = true;
            LOG_INFO("generate_video_chain: engine-owned TRACK audio '%s' (%zu frames @%u Hz x%u)",
                     chain_params->chain_audio_track, track_full.frames(),
                     track_full.sample_rate, track_full.channels);
        } else {
            LOG_ERROR("generate_video_chain: chain_audio_track '%s' unreadable", chain_params->chain_audio_track);
            return false;
        }
    }
    // Where the drive slices are written. Prefer save_dir (banked next to the latents, so a
    // resume/retake can see exactly what drove each segment); else the audio dir; else skip.
    std::string drive_slice_dir;
    if (chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0') {
        drive_slice_dir = chain_params->save_dir;
    } else if (chain_params->chain_audio_dir != nullptr && chain_params->chain_audio_dir[0] != '\0') {
        drive_slice_dir = chain_params->chain_audio_dir;
    }
    if (drive_full.loaded && drive_slice_dir.empty()) {
        LOG_ERROR("generate_video_chain: chain_audio_full needs save_dir or chain_audio_dir to stage slices");
        return false;
    }

    auto flush_window = [&](bool final_flush) {
        if (!streaming) {
            return;
        }
        const int keep = final_flush ? 0 : WINDOW_KEEP;
        if ((int)stitched.size() <= keep) {
            return;
        }
        const int n_flush = (int)stitched.size() - keep;
        // Hand the now-final prefix to the encoder IN ORDER; the callback consumes+frees each .data.
        chain_params->on_flush_frames(stitched.data(), n_flush, chain_params->on_flush_frames_user);
        stitched.erase(stitched.begin(), stitched.begin() + n_flush);  // .data already freed by callee
        flushed_total += n_flush;
    };

    // FEATURE A — latent-space per-channel affine continuity-match (anti-drift, opt-in). Capture
    // SEG-0's per-channel latent mean/std as the reference, then remap every later segment's tail
    // onto those stats before it conditions the next segment so colour/exposure drift can't
    // compound over a long chain. Purely in latent space (no VAE roundtrip). Default OFF; enable
    // with LTXAV_CONT_LATENT_MATCH, blend strength via LTXAV_CONT_MATCH_STRENGTH (0..1, default 1).
    std::vector<float>       ref_mean, ref_std;
    bool                     ref_ready       = false;
    static const bool        latent_match_on = getenv("LTXAV_CONT_LATENT_MATCH") != nullptr;
    float                    latent_match_strength = 1.0f;
    if (const char* e = getenv("LTXAV_CONT_MATCH_STRENGTH")) {
        latent_match_strength = std::clamp((float)atof(e), 0.f, 1.f);
    }

    // ── Resume: skip segments [0, resume_from) by reloading their banked artifacts ──
    // Rebuild the prefix frames by VAE-decoding the banked seg_<i>.bin latents (cheap, no
    // sampling), and seed cont_buf from seg_{resume_from-1}.bin so segment resume_from
    // continues the motion. resume_from == n_chain is finalize-only: reload every banked
    // segment and skip the sampler entirely. On any failure we fall back to a fresh full render.
    int start_seg = 0;
    // RETAKE (bidirectional single-segment splice) vs ordinary resume. When retake_active, the
    // prefix reload stops at retake_seg and the sampler renders ONLY that one segment (end-pinned
    // to seg_{retake_seg+1}'s head); the downstream segments are spliced from their banked latents
    // after the loop. retake_segment < 0 (the default) leaves this fully inert.
    const bool retake_active = chain_params->retake_segment >= 0 &&
                               chain_params->retake_segment < n_chain &&
                               chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0';
    const int  retake_seg       = retake_active ? chain_params->retake_segment : -1;
    const int  effective_resume = retake_active ? retake_seg : chain_params->resume_from;
    const int  render_end       = retake_active ? retake_seg : (n_chain - 1);
    if (retake_active) {
        LOG_INFO("generate_video_chain: RETAKE segment %d (reload prefix [0,%d), render only %d, splice banked tail [%d,%d))",
                 retake_seg, retake_seg, retake_seg, retake_seg + 1, n_chain);
    }
    if (effective_resume > 0 && chain_params->save_dir != nullptr &&
        chain_params->save_dir[0] != '\0') {
        int               resume_k = std::min(effective_resume, n_chain);
        const std::string sd_dir   = chain_params->save_dir;
        LOG_INFO("generate_video_chain: %s from segment %d/%d (reloading banked prefix from %s)",
                 resume_k == n_chain ? "FINALIZE-ONLY" : "RESUME",
                 resume_k, n_chain, sd_dir.c_str());
        bool prefix_ok = true;
        for (int seg = 0; seg < resume_k && prefix_ok; ++seg) {
            std::string p   = sd_dir + "/seg_" + std::to_string(seg) + ".bin";
            int         cnt = 0;
            sd_image_t* fr  = decode_banked_video_latent(sd_ctx, p, &cnt);
            if (fr == nullptr || cnt <= 0) {
                LOG_ERROR("generate_video_chain: resume prefix decode failed at seg %d (%s)", seg, p.c_str());
                free(fr);
                prefix_ok = false;
                break;
            }
            // mirror the live-stitch overlap drop (legacy head-place trims; keyframe-append = 0)
            bool legacy_head_r = false;
            if (const char* e = std::getenv("LTXAV_CONT_LEGACY_HEAD")) {
                legacy_head_r = atoi(e) != 0;
            }
            int drop = (seg == 0) ? 0 : (legacy_head_r ? overlap_px : 0);
            if (const char* e = std::getenv("LTXAV_CHAIN_OVERLAP_DROP")) {
                if (seg > 0) {
                    drop = atoi(e);
                }
            }
            // Length-pin: keep exactly the banked kept-count so the reloaded prefix reproduces the
            // ORIGINAL stitched length. The live chain auto-trims continuation overlaps (~overlap_px),
            // but this reload defaulted to drop=0 for keyframe-append -> the prefix grew by the
            // untrimmed overlap and shifted the whole timeline (the retake length drift). Falls back
            // to the default drop when no .len is banked (older jobs). Off via LTXAV_RETAKE_NO_PIN_LENGTH=1.
            if (getenv("LTXAV_RETAKE_NO_PIN_LENGTH") == nullptr) {
                int banked = read_seg_len(sd_dir + "/seg_" + std::to_string(seg) + ".len");
                if (banked >= 0 && banked <= cnt) {
                    drop = cnt - banked;
                }
            }
            if (drop > cnt) {
                drop = cnt;
            }
            for (int i = 0; i < cnt; ++i) {
                if (i < drop) {
                    free(fr[i].data);
                } else {
                    stitched.push_back(fr[i]);
                }
            }
            free(fr);
            // Reload this prefix segment's banked audio onto the timeline (best-effort —
            // a missing seg audio just means that slice is silent, not a resume failure).
            // Match each segment's saved audio contribution to its saved kept-frame count.
            sd_audio_t* pa = read_seg_audio(sd_dir + "/seg_" + std::to_string(seg) + ".audio");
            if (pa != nullptr) {
                const int kept_n = cnt - drop;
                long long head_offset_samples = llround((double)drop * pa->sample_rate / base_params->fps);
                long long keep_samples = llround((double)kept_n * pa->sample_rate / base_params->fps);
                audio_acc.append_window(pa, head_offset_samples, keep_samples);
                free_sd_audio(pa);
            }
            // Stream the reloaded prefix out as we go (finalize-only-resume reloads the WHOLE
            // timeline — keeping it all in RAM is exactly the 31.5 GB peak we're killing). Reloaded
            // frames carry no seam ops, so only the WINDOW_KEEP tail must survive for the first
            // freshly-rendered segment's exposure/crossfade lookback.
            flush_window(false);
        }
        if (prefix_ok) {
            std::string ptail = sd_dir + "/seg_" + std::to_string(resume_k - 1) + ".bin";
            try {
                sd::Tensor<float> tail = sd::load_tensor_from_file_as_tensor<float>(ptail);
                int    lw = (int)tail.shape()[0], lh = (int)tail.shape()[1];
                int    lt = (int)tail.shape()[2], cv = (int)tail.shape()[3];
                int    keep  = std::min(K, lt);
                size_t plane = (size_t)lw * lh;
                cont_buf.assign(plane * (size_t)keep * (size_t)cv, 0.f);
                const float* src_base = tail.data();
                for (int c = 0; c < cv; ++c) {
                    for (int nf = 0; nf < keep; ++nf) {
                        int          src_t = lt - keep + nf;
                        const float* src   = src_base + ((size_t)c * lt + src_t) * plane;
                        float*       dst   = cont_buf.data() + ((size_t)c * keep + nf) * plane;
                        std::memcpy(dst, src, plane * sizeof(float));
                    }
                }
                cont_Wl = lw;
                cont_Hl = lh;
                cont_Cv = cv;
                K         = keep;
                start_seg = resume_k;
            } catch (const std::exception& e) {
                LOG_ERROR("generate_video_chain: resume cont-latent load failed (%s): %s", ptail.c_str(), e.what());
                prefix_ok = false;
            }
        }
        if (!prefix_ok && streaming && flushed_total > 0) {
            // Some prefix frames were already streamed to (and freed by) the encoder — we can't
            // silently restart a fresh full render on top of them. Fail cleanly; the job can retry.
            for (auto& f : stitched) {
                free(f.data);
            }
            stitched.clear();
            LOG_ERROR("generate_video_chain: resume prefix failed after streaming %lld frame(s); "
                      "aborting (no fresh-render fallback once frames are emitted)", flushed_total);
            return false;
        }
        if (!prefix_ok) {
            LOG_WARN("generate_video_chain: resume failed — falling back to a fresh full render");
            for (auto& f : stitched) {
                free(f.data);
            }
            stitched.clear();
            cont_buf.clear();
            start_seg = 0;
            K         = std::max(1, chain_params->cont_latent_frames);
        }
    }

    // RETAKE END-PIN: load seg_{retake_seg+1}'s HEAD K latent frames (video channels only) so the
    // re-rendered segment terminates exactly where the unchanged, banked next segment begins.
    // Absent (last-segment retake, or a decode failure) degrades to an Option-A single-segment
    // render with no end pin — still safe.
    std::vector<float> end_cont_buf;
    int                end_cont_Ke = 0;
    if (retake_active && retake_seg + 1 < n_chain) {
        const std::string sd_dir = chain_params->save_dir;
        const std::string nhead  = sd_dir + "/seg_" + std::to_string(retake_seg + 1) + ".bin";
        try {
            sd::Tensor<float> nxt = sd::load_tensor_from_file_as_tensor<float>(nhead);
            int    lw = (int)nxt.shape()[0], lh = (int)nxt.shape()[1];
            int    lt = (int)nxt.shape()[2], cv = (int)nxt.shape()[3];
            int    keep  = std::min(K, lt);
            int    cvk   = std::min(LTXAV_VIDEO_LATENT_CHANNELS, cv);
            size_t plane = (size_t)lw * lh;
            end_cont_buf.assign(plane * (size_t)keep * (size_t)cvk, 0.f);
            const float* src_base = nxt.data();
            for (int c = 0; c < cvk; ++c) {
                for (int nf = 0; nf < keep; ++nf) {
                    int          src_t = nf;  // HEAD frames [0, keep)
                    const float* src   = src_base + ((size_t)c * lt + src_t) * plane;
                    float*       dst   = end_cont_buf.data() + ((size_t)c * keep + nf) * plane;
                    std::memcpy(dst, src, plane * sizeof(float));
                }
            }
            end_cont_Ke = keep;
            LOG_INFO("generate_video_chain: RETAKE end-pin loaded %d head frame(s) from seg_%d.bin", keep, retake_seg + 1);
        } catch (const std::exception& e) {
            LOG_WARN("generate_video_chain: RETAKE end-pin load failed (%s): %s — degrading to no end pin",
                     nhead.c_str(), e.what());
        }
    }

    // LTXAV_SHARED_REFINE_NOISE: running ABSOLUTE latent-frame position of the segment being rendered.
    //   offset(0) = 0;  offset(s) = offset(s-1) + T_lat(s-1) - overlap(s)
    // T_lat = 1 + (pixel_frames - 1) / 8 (the LTX causal-VAE 8:1 temporal ratio this function already
    // relies on for overlap_px at :13001), and overlap(s) is the K frames segment s re-renders from
    // the prior tail — exactly vp.cont_latent_frames, which the conditioning branch below already sets
    // to K for a continuation and 0 for a fresh anchor (seg 0 / scene cut / text cut / fresh keyframe
    // shot). Taking it from vp keeps this in lockstep with that branch instead of re-deriving it.
    //
    // RESUME (start_seg > 0): the skipped prefix was rendered by an earlier process, so its true
    // absolute positions must be reconstructed. This pre-roll assumes the prefix was all continuations
    // (what a resume means), mirroring the existing audio_frame_offset closed form. A scene cut inside
    // the skipped prefix would shift the base — which costs the feature at the resume seam only (that
    // seam degrades to today's independent noise) and cannot corrupt output.
    int64_t chain_latent_offset     = 0;
    int64_t prev_seg_latent_frames  = 0;
    for (int s = 0; s < start_seg; ++s) {
        const int frames_s = (chain_params->segment_video_frames != nullptr &&
                              chain_params->segment_video_frames[s] > 0)
                                 ? chain_params->segment_video_frames[s]
                                 : base_params->video_frames;
        chain_latent_offset += (1 + (frames_s - 1) / 8) - K;
    }

    for (int seg = start_seg; seg <= render_end; ++seg) {
        // The preceding segment has released its GPU-only chain residency, so a
        // host may safely switch to this segment's compatible fused DiT here.
        if (chain_params->before_segment != nullptr &&
            !chain_params->before_segment(seg, chain_params->before_segment_user)) {
            LOG_ERROR("generate_video_chain: DiT residency lease failed before segment %d", seg + 1);
            return false;
        }
        sd_vid_gen_params_t vp = *base_params;  // per-segment copy of the template
        vp.stage_seg_index     = seg;           // progressive stage previews (emit_stages) report THIS segment
        if (!chain_character_latent.empty()) {
            vp.character_reference_latent = chain_character_latent.data();
            vp.character_reference_latent_width = (int)chain_character_latent.shape()[0];
            vp.character_reference_latent_height = (int)chain_character_latent.shape()[1];
            vp.character_reference_latent_frames = (int)chain_character_latent.shape()[2];
            vp.character_reference_latent_channels = (int)chain_character_latent.shape()[3];
            // Stage-matched higher-res identity tiers (empty => refine falls back to base-res).
            if (!chain_character_latent_lo.empty()) {
                vp.character_reference_latent_lo          = chain_character_latent_lo.data();
                vp.character_reference_latent_lo_width    = (int)chain_character_latent_lo.shape()[0];
                vp.character_reference_latent_lo_height   = (int)chain_character_latent_lo.shape()[1];
                vp.character_reference_latent_lo_frames   = (int)chain_character_latent_lo.shape()[2];
                vp.character_reference_latent_lo_channels = (int)chain_character_latent_lo.shape()[3];
            }
            if (!chain_character_latent_hi.empty()) {
                vp.character_reference_latent_hi          = chain_character_latent_hi.data();
                vp.character_reference_latent_hi_width    = (int)chain_character_latent_hi.shape()[0];
                vp.character_reference_latent_hi_height   = (int)chain_character_latent_hi.shape()[1];
                vp.character_reference_latent_hi_frames   = (int)chain_character_latent_hi.shape()[2];
                vp.character_reference_latent_hi_channels = (int)chain_character_latent_hi.shape()[3];
            }
        }
        // Director variable-length: this shot renders its own frame count when the caller
        // supplied one (else the uniform base_params->video_frames). NULL/0 = byte-identical.
        if (chain_params->segment_video_frames != nullptr &&
            chain_params->segment_video_frames[seg] > 0) {
            vp.video_frames = chain_params->segment_video_frames[seg];
        }
        // RETAKE end-pin never leaks onto a non-retake segment: cleared here, set only for retake_seg.
        vp.end_cont_latent        = nullptr;
        vp.end_cont_latent_frames = 0;
        const bool segmented_relip = chain_params->segment_control_frames != nullptr &&
                                     chain_params->segment_control_frame_counts != nullptr &&
                                     chain_params->segment_control_frames[seg] != nullptr &&
                                     chain_params->segment_control_frame_counts[seg] > 0;
        if (segmented_relip) {
            vp.control_frames      = chain_params->segment_control_frames[seg];
            vp.control_frames_size = chain_params->segment_control_frame_counts[seg];
            // Per-segment mode override (else keep base_params->v2v_mode): 0 = lipdub relip
            // reference append, 1 = SDEdit denoise, 2 = guide-edit (source-as-guide + edit prompt).
            if (chain_params->segment_v2v_mode != nullptr) {
                int m = chain_params->segment_v2v_mode[seg];
                vp.v2v_mode = (m == 1 || m == 2) ? m : 0;
            }
            const char* v2v_desc = vp.v2v_mode == 1 ? "V2V SDEdit"
                                 : vp.v2v_mode == 2 ? "V2V guide-edit"
                                                    : "V2V relip";
            LOG_INFO("generate_video_chain seg %d: %s source window (%d frames)",
                     seg + 1, v2v_desc, vp.control_frames_size);
        }

        // Guide-edit LATENT-IN (v2v_mode==2 with a banked seg_<i>.bin): the common "edit a shot we
        // rendered" case — no pixels, just the banked latent path. When the per-segment array is
        // present it OWNS this segment's guide source (an entry makes it a guide-edit even without
        // control_frames; a NULL entry clears any inherited base path so this seg isn't a latent-in
        // guide). A NULL array leaves the base_params->v2v_guide_latent_path inheritance intact
        // (single-segment / global source).
        if (chain_params->segment_v2v_guide_latent_paths != nullptr) {
            const char* gp = chain_params->segment_v2v_guide_latent_paths[seg];
            if (gp != nullptr && gp[0] != '\0') {
                vp.v2v_guide_latent_path = gp;
                vp.v2v_mode              = 2;
                LOG_INFO("generate_video_chain seg %d: V2V guide-edit LATENT-IN %s", seg + 1, gp);
            } else {
                vp.v2v_guide_latent_path = nullptr;
            }
        }

        // Director keyframes: a shot with per-segment keyframes (image+frame pins) renders FRESH
        // via the LTXAV keyframe branch — frame 0 = scene start, last = end frame, a middle index
        // = a mid-shot reveal. Takes precedence over a plain scene-cut image and over continuation
        // (it's a fresh i2v-style shot, not a continuation of the prior motion).
        const bool has_keyframes = !segmented_relip &&
                                   chain_params->segment_keyframes != nullptr &&
                                   chain_params->segment_keyframe_counts != nullptr &&
                                   chain_params->segment_keyframe_counts[seg] > 0 &&
                                   chain_params->segment_keyframes[seg] != nullptr;

        // A keyframe shot with a pin at frame 0 (or seg 0) is FRESH — the pin defines the start.
        // A seg>0 keyframe shot with NO frame-0 pin CONTINUES the prior motion and injects the pins
        // mid-flow (Director v2 merged: reveal/identity-swap without restarting the scene).
        bool kf_frame0 = false;
        if (has_keyframes) {
            const int* kidx = chain_params->segment_keyframe_indices != nullptr
                                  ? chain_params->segment_keyframe_indices[seg]
                                  : nullptr;
            int kcnt = chain_params->segment_keyframe_counts[seg];
            for (int k = 0; kidx != nullptr && k < kcnt; ++k) {
                if (kidx[k] == 0) {
                    kf_frame0 = true;
                    break;
                }
            }
        }
        const bool kf_cont  = has_keyframes && seg > 0 && !segmented_relip && !kf_frame0;
        const bool kf_fresh = has_keyframes && !kf_cont;

        // Director multi-scene: a seg>0 with its own init image starts a FRESH i2v scene here
        // (a scene cut) rather than continuing the prior motion. Seg-0 already uses the opener
        // via base_params->init_image, so per-segment images only matter for seg>0.
        const bool scene_cut = seg > 0 && !segmented_relip && !has_keyframes &&
                               chain_params->segment_init_images != nullptr &&
                               chain_params->segment_init_images[seg] != nullptr &&
                               chain_params->segment_init_images[seg]->data != nullptr;

        // TEXT-ONLY scene cut: a seg>0 shot explicitly flagged as a new scene with NO image and no
        // keyframes — render it fresh from the prompt alone (mutually exclusive with scene_cut/kf).
        const bool text_scene_cut = seg > 0 && !segmented_relip && !has_keyframes && !scene_cut &&
                                    chain_params->segment_scene_cut != nullptr &&
                                    chain_params->segment_scene_cut[seg] != 0;

        // A fresh shot does not continue the prior tail; its stitch drop is 0 and it re-anchors the
        // continuity references. A merged (kf_cont) shot DOES continue, so it is NOT a fresh anchor.
        const bool fresh_anchor = scene_cut || kf_fresh || text_scene_cut;

        if (kf_fresh) {
            // Fresh keyframe shot: pin the caller's images at their frame indices; the keyframe
            // branch owns conditioning (init_image ignored, no continuation latent). Audio stays
            // on its normal per-segment path (drop=0 below keeps it aligned).
            vp.keyframes                 = chain_params->segment_keyframes[seg];
            vp.keyframe_frame_indices    = const_cast<int*>(chain_params->segment_keyframe_indices[seg]);
            vp.keyframes_size            = chain_params->segment_keyframe_counts[seg];
            vp.init_image.data           = nullptr;
            vp.cont_latent               = nullptr;
            vp.cont_latent_frames        = 0;
            vp.cont_refine_latent        = nullptr;
            vp.cont_refine_latent_frames = 0;
            vp.cont_refine_latent_lo        = nullptr;
            vp.cont_refine_latent_lo_frames = 0;
            vp.audio_frame_offset        = (seg == 0) ? 0 : seg * (base_params->video_frames - overlap_px);
            LOG_INFO("generate_video_chain seg %d: KEYFRAME shot (%d frame-pinned image(s), fresh)",
                     seg + 1, vp.keyframes_size);
        } else if (kf_cont) {
            // Merged (v2): continue the prior motion tail AND inject the frame pins mid-flow. The
            // engine's merged branch fires when BOTH keyframes and cont_latent are set.
            vp.keyframes                 = chain_params->segment_keyframes[seg];
            vp.keyframe_frame_indices    = const_cast<int*>(chain_params->segment_keyframe_indices[seg]);
            vp.keyframes_size            = chain_params->segment_keyframe_counts[seg];
            vp.init_image.data           = nullptr;
            vp.cont_latent               = cont_buf.data();
            vp.cont_latent_frames        = K;
            vp.cont_refine_latent        = cont_refine_buf.empty() ? nullptr : cont_refine_buf.data();
            vp.cont_refine_latent_frames = cont_refine_buf.empty() ? 0 : hires_ref_K;
            vp.cont_refine_latent_width  = cont_refine_Wl;
            vp.cont_refine_latent_height = cont_refine_Hl;
            vp.cont_refine_latent_channels = cont_refine_Cv;
            vp.cont_refine_latent_lo          = cont_refine_lo_buf.empty() ? nullptr : cont_refine_lo_buf.data();
            vp.cont_refine_latent_lo_frames   = cont_refine_lo_buf.empty() ? 0 : hires_ref_K;
            vp.cont_refine_latent_lo_width    = cont_refine_lo_Wl;
            vp.cont_refine_latent_lo_height   = cont_refine_lo_Hl;
            vp.cont_refine_latent_lo_channels = cont_refine_lo_Cv;
            vp.audio_frame_offset        = seg * (base_params->video_frames - overlap_px);
            LOG_INFO("generate_video_chain seg %d: MERGED continuation + %d keyframe pin(s)",
                     seg + 1, vp.keyframes_size);
        } else if (seg == 0 || segmented_relip) {
            vp.cont_latent        = nullptr;
            vp.cont_latent_frames = 0;
            vp.cont_refine_latent = nullptr;
            vp.cont_refine_latent_frames = 0;
            vp.cont_refine_latent_lo        = nullptr;
            vp.cont_refine_latent_lo_frames = 0;
            vp.audio_frame_offset = 0;
        } else if (scene_cut) {
            // Fresh scene from this shot's image: i2v start, no continuation latent (nothing to
            // continue). Audio is deliberately left on its normal per-segment path — aud_<seg>.wav
            // still drives this segment and drop=0 below keeps it aligned — so the soundtrack runs
            // unbroken across the cut. audio_frame_offset stays the continuation value (inert for
            // the LTXAV per-segment-wav path).
            vp.init_image         = *chain_params->segment_init_images[seg];
            vp.cont_latent        = nullptr;
            vp.cont_latent_frames = 0;
            vp.cont_refine_latent = nullptr;
            vp.cont_refine_latent_frames = 0;
            vp.cont_refine_latent_lo        = nullptr;
            vp.cont_refine_latent_lo_frames = 0;
            vp.audio_frame_offset = seg * (base_params->video_frames - overlap_px);
            LOG_INFO("generate_video_chain seg %d: SCENE CUT (fresh i2v from per-segment image %dx%d)",
                     seg + 1, vp.init_image.width, vp.init_image.height);
        } else if (text_scene_cut) {
            // Fresh scene from this shot's PROMPT ALONE — a pure t2v opener mid-chain: no init
            // image (clear the lingering opener), no continuation latent. Same fresh path as seg 0,
            // just decoupled from the presence of an image. drop=0 + re-anchor via fresh_anchor;
            // audio stays on the per-segment aud_<seg>.wav path (inert offset like scene_cut).
            vp.init_image.data           = nullptr;
            vp.cont_latent               = nullptr;
            vp.cont_latent_frames        = 0;
            vp.cont_refine_latent        = nullptr;
            vp.cont_refine_latent_frames = 0;
            vp.cont_refine_latent_lo        = nullptr;
            vp.cont_refine_latent_lo_frames = 0;
            vp.audio_frame_offset        = seg * (base_params->video_frames - overlap_px);
            LOG_INFO("generate_video_chain seg %d: TEXT SCENE CUT (fresh t2v, no image)", seg + 1);
        } else {
            // Clear the init image for seg>0: prepare_video_generation_latents checks the
            // start image BEFORE the cont-latent branch, so a lingering init image would
            // re-render i2v from the same portrait and ignore the continuation.
            vp.init_image.data    = nullptr;
            vp.cont_latent        = cont_buf.data();
            vp.cont_latent_frames = K;
            vp.cont_refine_latent = cont_refine_buf.empty() ? nullptr : cont_refine_buf.data();
            vp.cont_refine_latent_frames = cont_refine_buf.empty() ? 0 : hires_ref_K;
            vp.cont_refine_latent_width = cont_refine_Wl;
            vp.cont_refine_latent_height = cont_refine_Hl;
            vp.cont_refine_latent_channels = cont_refine_Cv;
            vp.cont_refine_latent_lo          = cont_refine_lo_buf.empty() ? nullptr : cont_refine_lo_buf.data();
            vp.cont_refine_latent_lo_frames   = cont_refine_lo_buf.empty() ? 0 : hires_ref_K;
            vp.cont_refine_latent_lo_width    = cont_refine_lo_Wl;
            vp.cont_refine_latent_lo_height   = cont_refine_lo_Hl;
            vp.cont_refine_latent_lo_channels = cont_refine_lo_Cv;
            vp.audio_frame_offset = seg * (base_params->video_frames - overlap_px);
        }
        // RETAKE end-pin: this segment (retake_seg) terminates at seg_{retake_seg+1}'s head so the
        // banked next segment still flows from its freshly re-rendered tail. Composes with whatever
        // START conditioning the branch above set (cont_latent for a continuation shot, or init_image
        // for a seg-0 opener / scene cut). A missing next segment (last-seg retake) leaves it off.
        if (retake_active && seg == retake_seg && end_cont_Ke > 0) {
            vp.end_cont_latent        = end_cont_buf.data();
            vp.end_cont_latent_frames = end_cont_Ke;
            LOG_INFO("generate_video_chain seg %d: RETAKE end-pin active (%d frozen next-head frames)",
                     seg + 1, end_cont_Ke);
        }

        // LTXAV_SHARED_REFINE_NOISE: advance to THIS segment's absolute latent-frame start. Placed
        // after the conditioning branch so vp.cont_latent_frames already reflects whether this segment
        // continues the prior tail (K) or re-anchors fresh (0). Inert unless the engine-side gate is on.
        const int64_t seg_latent_frames = 1 + (vp.video_frames - 1) / 8;
        if (seg > start_seg) {
            const int64_t seg_overlap = (vp.cont_latent != nullptr && vp.cont_latent_frames > 0)
                                            ? vp.cont_latent_frames
                                            : 0;
            chain_latent_offset += prev_seg_latent_frames - seg_overlap;
        }
        vp.chain_latent_offset = chain_latent_offset;
        prev_seg_latent_frames = seg_latent_frames;

        // distinct seed per segment so the noise frames differ
        vp.seed = (base_params->seed < 0) ? base_params->seed : base_params->seed + seg;

        // Per-segment prompt (director): this segment's line, or the base prompt.
        if (chain_params->segment_prompts != nullptr && chain_params->segment_prompts[seg] != nullptr) {
            vp.prompt = chain_params->segment_prompts[seg];
            LOG_INFO("generate_video_chain seg %d prompt: %s", seg, vp.prompt);
        }

        // Per-segment lip-sync audio. The string must outlive the generate call, so keep it
        // loop-scoped.
        std::string seg_audio_path;
        if (drive_full.loaded) {
            // ── ENGINE-OWNED SLICE ───────────────────────────────────────────────────────
            // Anchor this segment's drive window to the TRUE accumulated timeline rather than
            // to a predicted grid. `timeline_pos` is where this segment's first KEPT frame
            // will land — every earlier segment's ACTUAL (auto-trimmed) kept length is already
            // baked into it, which is exactly what the client-side slicer could not know.
            //
            // We render seg_frames and later drop `drop` from the head, so segment-local frame
            // `drop` must carry timeline_pos's audio => the window starts drop_pred frames EARLY.
            // `drop` is content-adaptive (ltxav_auto_trim_drop) and only measurable AFTER the
            // render, so drop_pred is the a-priori estimate. A miss of δ frames offsets lip-sync
            // by δ WITHIN this segment only (~40ms/frame @24fps) and CANNOT accumulate: the next
            // segment re-anchors to the corrected timeline_pos. That self-correction is the whole
            // point of moving this in-engine — the old client-side constant had no feedback path,
            // so its error compounded across every seam.
            const long long timeline_pos = flushed_total + (long long)stitched.size();
            int drop_pred = (seg == 0 || segmented_relip || fresh_anchor) ? 0 : overlap_px;
            if (const char* e = std::getenv("LTXAV_CHAIN_OVERLAP_DROP")) {
                // Value-gate: "${VAR:-}" yields an EMPTY STRING and getenv returns non-null, so a
                // bare presence check silently enables drop=0 via atoi(""). Only a real integer pins.
                if (e[0] != '\0' && seg > 0 && !(segmented_relip || fresh_anchor)) {
                    char* end = nullptr;
                    long  v   = strtol(e, &end, 10);
                    if (end != nullptr && *end == '\0' && v >= 0) {
                        drop_pred = (int)v;
                    }
                }
            }
            const long long win_start = timeline_pos - drop_pred;
            std::vector<float> win =
                drive_full.window(win_start, (long long)vp.video_frames, (float)base_params->fps, audio_offset_frames);
            seg_audio_path = drive_slice_dir + "/aud_" + std::to_string(seg) + ".wav";
            if (!win.empty() && write_wav_16k_mono(seg_audio_path, win.data(), win.size())) {
                vp.drive_audio_path = seg_audio_path.c_str();
                LOG_INFO("generate_video_chain seg %d drive-audio: engine slice [%lld,%lld) frames "
                         "(timeline_pos=%lld drop_pred=%d) -> %s",
                         seg, win_start, win_start + vp.video_frames, timeline_pos, drop_pred,
                         seg_audio_path.c_str());
            } else {
                LOG_ERROR("generate_video_chain seg %d: drive slice failed; segment renders without lip-sync", seg);
                seg_audio_path.clear();
            }
        } else if (chain_params->chain_audio_dir != nullptr && chain_params->chain_audio_dir[0] != '\0') {
            // LEGACY: caller pre-sliced aud_<i>.wav against its own guess of the seam trim.
            seg_audio_path     = std::string(chain_params->chain_audio_dir) + "/aud_" + std::to_string(seg) + ".wav";
            vp.drive_audio_path = seg_audio_path.c_str();
            LOG_INFO("generate_video_chain seg %d drive-audio: %s (legacy pre-sliced)", seg, seg_audio_path.c_str());
        }

        LOG_INFO("=== generate_video_chain segment %d/%d ===", seg + 1, n_chain);

        // Optional: bank each segment's saved VIDEO latent so a single failing segment can
        // be replayed standalone (via cont_latent_path) instead of re-running the chain.
        std::string seg_save_path;
        if (chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0') {
            seg_save_path = std::string(chain_params->save_dir) + "/seg_" + std::to_string(seg) + ".bin";
            setenv("LTXAV_SAVE_VIDEO_LATENT", seg_save_path.c_str(), 1);
        }

        sd_image_t* seg_video    = nullptr;
        int         seg_count    = 0;
        sd_audio_t* seg_audio    = nullptr;
        float*      lat_out      = nullptr;
        int         lw = 0, lh = 0, lt = 0, lc = 0;  // lc = FULL channel count (video + audio)
        float*      refined_out = nullptr;
        int         rw = 0, rh = 0, rt = 0, rc = 0;
        float*      refined_lo_out = nullptr;
        int         rlow = 0, rloh = 0, rlot = 0, rloc = 0;
        bool        want_latent  = (seg + 1 < n_chain);
        if (!generate_video_ex(sd_ctx, &vp, &seg_video, &seg_count, &seg_audio, nullptr,
                               want_latent ? &lat_out : nullptr,
                               want_latent ? &lw : nullptr, want_latent ? &lh : nullptr,
                               want_latent ? &lt : nullptr, want_latent ? &lc : nullptr,
                               (want_latent && chain_hires_reference) ? &refined_out : nullptr,
                               (want_latent && chain_hires_reference) ? &rw : nullptr,
                               (want_latent && chain_hires_reference) ? &rh : nullptr,
                               (want_latent && chain_hires_reference) ? &rt : nullptr,
                               (want_latent && chain_hires_reference) ? &rc : nullptr,
                               (want_latent && chain_hires_reference) ? &refined_lo_out : nullptr,
                               (want_latent && chain_hires_reference) ? &rlow : nullptr,
                               (want_latent && chain_hires_reference) ? &rloh : nullptr,
                               (want_latent && chain_hires_reference) ? &rlot : nullptr,
                               (want_latent && chain_hires_reference) ? &rloc : nullptr)) {
            LOG_ERROR("generate_video_chain segment %d failed", seg + 1);
            free_sd_audio(seg_audio);
            free(seg_video);
            free(lat_out);
            free(refined_out);
            free(refined_lo_out);
            // Free everything collected so far (audio_acc frees itself on scope exit).
            for (auto& f : stitched) {
                free(f.data);
            }
            return false;
        }

        // DRIFT-SINK (env LONGCAT_CONT_REENCODE=1): condition the next segment on the
        // VAE-ENCODED pixels of this segment's tail instead of the raw diffusion latent.
        // Official LTX continuation conditions on VAE-encoded frames; our raw DMD-x0 latent
        // carries high-freq sampling structure that, frozen as conditioning, can ghost
        // throughout the continuation segment. Decode→re-encode snaps it onto the VAE
        // manifold. Mirrors the main.cpp manual-chain path. Default off (raw latent).
        bool reencoded = false;
        if (want_latent && getenv("LONGCAT_CONT_REENCODE") != nullptr && seg_video != nullptr && seg_count > 0 &&
            ((int)seg_video[0].width != base_params->width || (int)seg_video[0].height != base_params->height)) {
            LOG_WARN("generate_video_chain: LONGCAT_CONT_REENCODE is incompatible with LTX hires "
                     "(%ux%u rendered vs %dx%d base transport); using the raw base latent tail",
                     seg_video[0].width, seg_video[0].height, base_params->width, base_params->height);
        } else if (want_latent && getenv("LONGCAT_CONT_REENCODE") != nullptr && seg_video != nullptr && seg_count > 0) {
            int tail = std::min(overlap_px, seg_count);  // pixel frames that re-encode to K latents
            int rlw = 0, rlh = 0, rlt = 0, rlc = 0;
            float* reenc = sd_ctx_encode_video_frames(sd_ctx, seg_video + (seg_count - tail), tail,
                                                      seg_video[0].width, seg_video[0].height,
                                                      &rlw, &rlh, &rlt, &rlc);
            if (reenc != nullptr && rlt > 0) {
                int    cv    = std::min(LTXAV_VIDEO_LATENT_CHANNELS, rlc);
                int    keep  = std::min(K, rlt);
                size_t plane = (size_t)rlw * rlh;
                cont_buf.assign(plane * (size_t)keep * (size_t)cv, 0.f);
                for (int c = 0; c < cv; ++c) {
                    for (int nf = 0; nf < keep; ++nf) {
                        int          src_t = rlt - keep + nf;
                        const float* src   = reenc + ((size_t)c * rlt + src_t) * plane;
                        float*       dst   = cont_buf.data() + ((size_t)c * keep + nf) * plane;
                        std::memcpy(dst, src, plane * sizeof(float));
                    }
                }
                free(reenc);
                cont_Wl    = rlw;
                cont_Hl    = rlh;
                cont_Cv    = cv;
                K          = keep;
                reencoded  = true;
                LOG_INFO("generate_video_chain: cont via VAE RE-ENCODE of last %d px frames -> %d cond latents (drift sink)",
                         tail, keep);
            } else {
                free(reenc);
                LOG_WARN("generate_video_chain: re-encode failed; falling back to raw latent tail");
            }
        }

        // Capture the LAST K latent frames + the first LTXAV_VIDEO_LATENT_CHANNELS VIDEO
        // channels into cont_buf for the next segment. Returned latent layout is
        // [Wl, Hl, Tl, Cl] contiguous (ggml-ne): index(w,h,t,c) = ((c*Tl + t)*Hl + h)*Wl + w.
        // Audio lives in channels >= video_channels and is dropped here.
        if (!reencoded && want_latent && lat_out != nullptr) {
            int cv = std::min(LTXAV_VIDEO_LATENT_CHANNELS, lc);  // video channels to keep
            if (cv < lc) {
                LOG_INFO("generate_video_chain: latent has %d channels; keeping first %d as VIDEO (audio stripped)", lc, cv);
            }
            int keep = std::min(K, lt);  // last `keep` latent frames
            if (keep < K) {
                LOG_WARN("generate_video_chain: prior segment produced only %d latent frames (< K=%d); using %d", lt, K, keep);
            }
            size_t plane = (size_t)lw * lh;
            cont_buf.assign(plane * (size_t)keep * (size_t)cv, 0.f);
            for (int c = 0; c < cv; ++c) {
                for (int nf = 0; nf < keep; ++nf) {
                    int          src_t = lt - keep + nf;
                    const float* src   = lat_out + ((size_t)c * lt + src_t) * plane;
                    float*       dst   = cont_buf.data() + ((size_t)c * keep + nf) * plane;
                    std::memcpy(dst, src, plane * sizeof(float));
                }
            }
            cont_Wl = lw;
            cont_Hl = lh;
            cont_Cv = cv;
            K = keep;  // K passed to the next segment must equal the frames actually captured
        }
        if (chain_hires_reference && want_latent && refined_out != nullptr && rw > 0 && rh > 0 && rt > 0 && rc == LTXAV_VIDEO_LATENT_CHANNELS) {
            const int keep = std::min(hires_ref_K, rt);
            const size_t plane = (size_t)rw * rh;
            cont_refine_buf.assign(plane * (size_t)keep * (size_t)rc, 0.f);
            for (int c = 0; c < rc; ++c) {
                for (int nf = 0; nf < keep; ++nf) {
                    const int src_t = rt - keep + nf;
                    const float* src = refined_out + ((size_t)c * rt + src_t) * plane;
                    float* dst = cont_refine_buf.data() + ((size_t)c * keep + nf) * plane;
                    std::memcpy(dst, src, plane * sizeof(float));
                }
            }
            cont_refine_Wl = rw;
            cont_refine_Hl = rh;
            cont_refine_Cv = rc;
            LOG_INFO("generate_video_chain: captured %d high-res refined VIDEO tail frames [%d,%d,%d] for next stage-2 reference",
                     keep, rw, rh, rc);
        } else if (chain_hires_reference && want_latent && base_params->hires.enabled) {
            LOG_WARN("generate_video_chain: missing/invalid high-res refined continuation state; stage-2 will use base-only continuation");
            cont_refine_buf.clear();
            cont_refine_Wl = cont_refine_Hl = cont_refine_Cv = 0;
        }
        // Parallel capture of the STAGE-0 (lower-res) refined tail for the next segment's FIRST hires
        // stage. Only exported by generate_video_ex on a >=2-stage chain, so on the single-stage (2x)
        // path refined_lo_out stays null and cont_refine_lo_buf stays empty (byte-identical).
        if (chain_hires_reference && want_latent && refined_lo_out != nullptr &&
            rlow > 0 && rloh > 0 && rlot > 0 && rloc == LTXAV_VIDEO_LATENT_CHANNELS) {
            const int keep = std::min(hires_ref_K, rlot);
            const size_t plane = (size_t)rlow * rloh;
            cont_refine_lo_buf.assign(plane * (size_t)keep * (size_t)rloc, 0.f);
            for (int c = 0; c < rloc; ++c) {
                for (int nf = 0; nf < keep; ++nf) {
                    const int src_t = rlot - keep + nf;
                    const float* src = refined_lo_out + ((size_t)c * rlot + src_t) * plane;
                    float* dst = cont_refine_lo_buf.data() + ((size_t)c * keep + nf) * plane;
                    std::memcpy(dst, src, plane * sizeof(float));
                }
            }
            cont_refine_lo_Wl = rlow;
            cont_refine_lo_Hl = rloh;
            cont_refine_lo_Cv = rloc;
            LOG_INFO("generate_video_chain: captured %d stage-0 refined VIDEO tail frames [%d,%d,%d] for next stage-0 reference",
                     keep, rlow, rloh, rloc);
        } else {
            cont_refine_lo_buf.clear();
            cont_refine_lo_Wl = cont_refine_lo_Hl = cont_refine_lo_Cv = 0;
        }
        free(lat_out);
        free(refined_out);
        free(refined_lo_out);

        // FEATURE A: capture SEG-0's per-channel latent stats as the anti-drift reference, and
        // (seg>0) remap this segment's freshly-captured tail onto them BEFORE it conditions the
        // next segment at the top of the following iteration. Runs only when the tail was actually
        // (re)filled this segment (want_latent) and the dims are known.
        if (want_latent && !cont_buf.empty() && cont_Wl > 0 && cont_Hl > 0 && cont_Cv > 0 && K > 0) {
            if ((seg == 0 && !ref_ready) || fresh_anchor) {
                // seg-0, or a fresh shot (scene cut / keyframe shot), RE-ANCHORS the anti-drift
                // reference to THIS scene's stats — so the continuation shots that follow match
                // the new scene, not the one before the cut.
                ltxav_compute_latent_channel_stats(cont_buf.data(), cont_Wl, cont_Hl, K, cont_Cv, ref_mean, ref_std);
                ref_ready = true;
                if (latent_match_on) {
                    LOG_INFO("generate_video_chain: captured %s latent channel reference stats (%d channels) for continuity-match",
                             scene_cut ? "scene-cut" : (has_keyframes ? "keyframe" : "seg-0"), cont_Cv);
                }
            } else if (seg > 0 && ref_ready && latent_match_on) {
                ltxav_latent_channel_affine_match(cont_buf.data(), cont_Wl, cont_Hl, K, cont_Cv,
                                                  ref_mean, ref_std, latent_match_strength);
            }
        }

        // Stitch: seg0 keeps all frames. For seg>0 the LEGACY head-placement path re-renders
        // the overlap at the head and must drop it (overlap_px); the keyframe-append path
        // (default) places the guide in the PAST and generates only NEW frames, so it drops
        // nothing — that re-render+trim is exactly what caused the seam "skip". Override the
        // drop with LTXAV_CHAIN_OVERLAP_DROP for tuning.
        bool legacy_head = false;
        if (const char* e = std::getenv("LTXAV_CONT_LEGACY_HEAD")) {
            legacy_head = atoi(e) != 0;
        }
        int drop;
        if (seg == 0 || segmented_relip || fresh_anchor) {
            // A fresh shot (scene cut or keyframe shot) generates a wholly new scene — nothing to
            // trim against the prior segment (and drop=0 keeps its audio aligned, like seg-0).
            drop = 0;
        } else if (legacy_head) {
            drop = overlap_px;  // legacy head-placement re-renders + trims the fixed overlap
        } else {
            // keyframe-append (default): auto-align the trim to the smoothest continuation of
            // the prior segment's last kept frame, instead of a fixed overlap_px (the phase
            // mismatch at a fixed cut is the seam "skip").
            drop = ltxav_auto_trim_drop(stitched.empty() ? sd_image_t{} : stitched.back(),
                                        seg_video, seg_count, overlap_px);
            LOG_INFO("generate_video_chain: seam auto-trim -> drop %d (fixed overlap_px would be %d)",
                     drop, overlap_px);
        }
        if (const char* e = std::getenv("LTXAV_CHAIN_OVERLAP_DROP")) {
            // Only pin the trim for genuine continuations. A fresh shot (scene cut / keyframe-fresh /
            // relip anchor) already resolved drop=0 above and must NOT be trimmed — else its brand-new
            // scene loses its opening frames and the whole timeline (and audio) shifts by the overlap.
            if (seg > 0 && !(segmented_relip || fresh_anchor)) {
                drop = atoi(e);
            }
        }
        // RETAKE length-pin: reproduce the ORIGINAL kept length for the retaken segment so the
        // timeline (and audio/beat sync) does not shift. The end-pin lands the new tail on the same
        // boundary as the banked tail, so pinning the leading drop is seam-clean. Off via
        // LTXAV_RETAKE_NO_PIN_LENGTH=1 (falls back to content-adaptive auto-trim).
        if (retake_active && seg == retake_seg && getenv("LTXAV_RETAKE_NO_PIN_LENGTH") == nullptr &&
            chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0') {
            int banked = read_seg_len(std::string(chain_params->save_dir) + "/seg_" + std::to_string(seg) + ".len");
            if (banked >= 0 && banked <= seg_count) {
                LOG_INFO("generate_video_chain: RETAKE length-pin seg %d -> drop %d (auto-trim was %d; banked kept=%d)",
                         seg, seg_count - banked, drop, banked);
                drop = seg_count - banked;
            }
        }
        size_t kept_start = stitched.size();
        if (drop > seg_count) {
            drop = seg_count;
        }
        // P1-C: tone-match this segment's kept frames to the prior segment's tail before stitching,
        // flattening the ~0.85-luma step at the seam (continuity correction, not a cross-fade). seg0
        // has nothing to match against. Operates on the post-trim kept frames so the matched window
        // lines up with what actually lands in the timeline.
        if (seg > 0 && !fresh_anchor && !stitched.empty() && seg_count - drop > 0) {
            // Skip at a fresh shot: the new scene must NOT be tone-graded toward the old one.
            ltxav_exposure_match(stitched.data(), (int)stitched.size(),
                                 seg_video + drop, seg_count - drop);
        }
        // Seam cross-dissolve (opt-in): fade stitched's tail toward this segment's re-rendered overlap
        // so the join dissolves instead of hard-cutting. Runs AFTER exposure-match, BEFORE the discard.
        if (const char* e = getenv("LTXAV_SEAM_CROSSFADE"); e != nullptr && seg > 0 && !stitched.empty()) {
            ltxav_seam_crossfade(stitched, seg_video, drop, atoi(e));
        }
        for (int i = 0; i < seg_count; ++i) {
            if (i < drop) {
                free(seg_video[i].data);  // discard re-rendered overlap frame
            } else {
                stitched.push_back(seg_video[i]);  // adopt ownership of .data
            }
        }
        free(seg_video);

        // Hand this segment's kept frames to the caller (server) so it can bank a viewable
        // per-segment webm as it's produced. Frames stay owned by `stitched`; the callback
        // must copy anything it needs to outlive this call.
        int kept_n = (int)(stitched.size() - kept_start);
        if (chain_params->on_segment != nullptr && kept_n > 0) {
            chain_params->on_segment(seg, stitched.data() + kept_start, kept_n, chain_params->on_segment_user);
        }
        // Bank this segment's kept-frame count so a later retake can pin the timeline (see
        // read_seg_len). On a retake render this rewrites seg_<retake_seg>.len with the same value
        // it was pinned to, so the banked length stays authoritative.
        if (chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0' && kept_n > 0) {
            write_seg_len(std::string(chain_params->save_dir) + "/seg_" + std::to_string(seg) + ".len", kept_n);
        }

        // Audio: bank this segment's audio (for resume) + stitch it onto the timeline,
        // length-matching each continuation to its kept video-frame count.
        if (chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0' && seg_audio != nullptr) {
            write_seg_audio(std::string(chain_params->save_dir) + "/seg_" + std::to_string(seg) + ".audio", seg_audio);
        }
        if (seg_audio != nullptr) {
            long long head_offset_samples = llround((double)drop * seg_audio->sample_rate / base_params->fps);
            long long keep_samples = llround((double)kept_n * seg_audio->sample_rate / base_params->fps);
            audio_acc.append_window(seg_audio, head_offset_samples, keep_samples);
        }
        free_sd_audio(seg_audio);

        // Reclaim the per-segment GPU residency + caches before the NEXT segment's DiT
        // sampling allocates its full footprint on top (the +1.4 GB chain anchor). Skipped
        // after the last segment (nothing follows) and gated so it only runs on the warm
        // resident chain path. Default-on; env LTXAV_NO_CHAIN_GPU_RECLAIM=1 restores the
        // old (leaky) behaviour for A/B.
        if (seg + 1 < n_chain && sd_ctx->sd->keep_diffusion_model_resident &&
            getenv("LTXAV_NO_CHAIN_GPU_RECLAIM") == nullptr) {
            sd_ctx->sd->release_chain_segment_gpu_residency();
        }

        // Windowed streaming: this segment's kept frames are now stitched and reported (on_segment
        // fired above), and next segment's seam ops can only reach the WINDOW_KEEP tail — flush and
        // free everything older so peak RAM stays ≈ the live window, not the whole timeline.
        flush_window(false);
    }

    // ── RETAKE tail splice ──────────────────────────────────────────────────────────────────
    // Re-attach the UNCHANGED downstream segments [retake_seg+1, n_chain) from their banked
    // latents (VAE-decode only, NO sampling). Because segment retake_seg was end-pinned to
    // seg_{retake_seg+1}'s head, its new tail lands where the banked tail already begins, so the
    // join is continuous without re-rendering or modifying the tail pixels — the downstream
    // segments stay bit-identical to their banked webms (only the head auto-trim can differ). Each
    // continuation tail segment re-derives its overlap drop against the NEW preceding frame so the
    // N/N+1 join is clean; a fresh shot (scene cut / fresh keyframe) trims 0 as in the live path.
    if (retake_active && chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0') {
        const std::string sd_dir = chain_params->save_dir;
        bool              legacy_head_s = false;
        if (const char* e = std::getenv("LTXAV_CONT_LEGACY_HEAD")) {
            legacy_head_s = atoi(e) != 0;
        }
        for (int seg = retake_seg + 1; seg < n_chain; ++seg) {
            std::string p   = sd_dir + "/seg_" + std::to_string(seg) + ".bin";
            int         cnt = 0;
            sd_image_t* fr  = decode_banked_video_latent(sd_ctx, p, &cnt);
            if (fr == nullptr || cnt <= 0) {
                LOG_ERROR("generate_video_chain: RETAKE tail decode failed at seg %d (%s)", seg, p.c_str());
                free(fr);
                for (auto& f : stitched) {
                    free(f.data);
                }
                return false;
            }
            // Fresh shot (scene cut / fresh keyframe) => trim 0; continuation => auto-trim vs the
            // NEW preceding stitched frame. Mirrors the live-loop drop logic.
            bool kf_here = chain_params->segment_keyframes != nullptr &&
                           chain_params->segment_keyframe_counts != nullptr &&
                           chain_params->segment_keyframe_counts[seg] > 0 &&
                           chain_params->segment_keyframes[seg] != nullptr;
            bool kf_frame0_here = false;
            if (kf_here && chain_params->segment_keyframe_indices != nullptr &&
                chain_params->segment_keyframe_indices[seg] != nullptr) {
                int kc = chain_params->segment_keyframe_counts[seg];
                for (int k = 0; k < kc; ++k) {
                    if (chain_params->segment_keyframe_indices[seg][k] == 0) {
                        kf_frame0_here = true;
                        break;
                    }
                }
            }
            bool kf_fresh_here  = kf_here && (seg == 0 || kf_frame0_here);
            bool scene_cut_here = seg > 0 && !kf_here &&
                                  chain_params->segment_init_images != nullptr &&
                                  chain_params->segment_init_images[seg] != nullptr &&
                                  chain_params->segment_init_images[seg]->data != nullptr;
            bool text_scene_cut_here = seg > 0 && !kf_here && !scene_cut_here &&
                                       chain_params->segment_scene_cut != nullptr &&
                                       chain_params->segment_scene_cut[seg] != 0;
            bool fresh_here = scene_cut_here || kf_fresh_here || text_scene_cut_here;
            int  drop;
            if (fresh_here) {
                drop = 0;
            } else if (legacy_head_s) {
                drop = overlap_px;
            } else {
                drop = ltxav_auto_trim_drop(stitched.empty() ? sd_image_t{} : stitched.back(), fr, cnt, overlap_px);
            }
            if (const char* e = std::getenv("LTXAV_CHAIN_OVERLAP_DROP")) {
                drop = atoi(e);
            }
            // RETAKE length-pin: reproduce each banked tail segment's ORIGINAL kept length so the
            // total timeline is unchanged (only segment retake_seg's pixels differ). Off via
            // LTXAV_RETAKE_NO_PIN_LENGTH=1.
            if (getenv("LTXAV_RETAKE_NO_PIN_LENGTH") == nullptr && !fresh_here) {
                int banked = read_seg_len(sd_dir + "/seg_" + std::to_string(seg) + ".len");
                if (banked >= 0 && banked <= cnt) {
                    LOG_INFO("generate_video_chain: RETAKE tail length-pin seg %d -> drop %d (auto-trim was %d; banked kept=%d)",
                             seg, cnt - banked, drop, banked);
                    drop = cnt - banked;
                }
            }
            if (drop > cnt) {
                drop = cnt;
            }
            for (int i = 0; i < cnt; ++i) {
                if (i < drop) {
                    free(fr[i].data);
                } else {
                    stitched.push_back(fr[i]);
                }
            }
            free(fr);
            sd_audio_t* pa = read_seg_audio(sd_dir + "/seg_" + std::to_string(seg) + ".audio");
            if (pa != nullptr) {
                const int kept_n = cnt - drop;
                long long head_offset_samples = llround((double)drop * pa->sample_rate / base_params->fps);
                long long keep_samples = llround((double)kept_n * pa->sample_rate / base_params->fps);
                audio_acc.append_window(pa, head_offset_samples, keep_samples);
                free_sd_audio(pa);
            }
            // Windowed streaming: the spliced tail only needs stitched.back() for the next tail
            // segment's auto-trim, so flush all but the WINDOW_KEEP suffix.
            flush_window(false);
        }
        LOG_INFO("generate_video_chain: RETAKE spliced banked tail [%d, %d) after re-rendered segment %d",
                 retake_seg + 1, n_chain, retake_seg);
    }

    // End-of-render GPU reclaim (opt-in: LTXAV_END_RENDER_RECLAIM=1). The between-segment
    // release above is gated `seg + 1 < n_chain`, so the FINAL segment's compute/cache
    // buffers, shared residency, and the grown VMM scratch-pool high-water persist into
    // the idle warm worker and stack under the NEXT render's peak — a ~2.4 GB cross-render
    // VRAM creep (render 1 ~11.5 GB -> render 2+ ~13-14 GB) on the long-lived server worker.
    // Releasing here returns each render to the clean fresh-worker baseline. Safe: all frames
    // are already copied host-side (stitched/audio below touch no GPU). Does not free params
    // (weights stay warm; idle floor unchanged). Honours LTXAV_NO_CHAIN_GPU_RECLAIM as a
    // master off-switch; LTXAV_CHAIN_POOL_TRIM still controls whether the pool high-water is
    // returned to the OS inside the reclaim.
    if (sd_ctx->sd->keep_diffusion_model_resident &&
        getenv("LTXAV_END_RENDER_RECLAIM") != nullptr &&
        getenv("LTXAV_NO_CHAIN_GPU_RECLAIM") == nullptr) {
        sd_ctx->sd->release_chain_segment_gpu_residency();
    }

    // Don't let the per-segment latent-save env leak into a later (non-chain) render in the
    // same process — it's re-set per segment on the next chain anyway.
    if (chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0') {
        unsetenv("LTXAV_SAVE_VIDEO_LATENT");
    }

    // Assemble the stitched audio timeline (null if no segment produced audio).
    sd_audio_t* chain_audio = audio_acc.build();

    // ── ENGINE-OWNED TRACK: deliver the caller's audio, cut to the ACTUAL timeline ────────
    // The generated per-segment audio was only ever lip-sync conditioning; the deliverable is
    // the user's own clip. Doing this here (rather than the client re-muxing afterwards) is
    // what makes it correct: `total_frames` is the REAL stitched length after every
    // content-adaptive auto-trim, so the cut lands where the video actually ends. A client
    // muxing "the full track with -shortest" instead aligns only at t=0 and lets any
    // accumulated trim disagreement ride as a progressive A/V drift.
    if (track_full.loaded) {
        const long long total_frames = flushed_total + (long long)stitched.size();
        std::vector<float> win =
            track_full.window(0, total_frames, (float)base_params->fps, audio_offset_frames);
        if (!win.empty()) {
            sd_audio_t* t = (sd_audio_t*)malloc(sizeof(sd_audio_t));
            if (t != nullptr) {
                t->sample_rate  = track_full.sample_rate;
                t->channels     = track_full.channels;
                t->sample_count = win.size() / track_full.channels;
                t->data         = (float*)malloc(win.size() * sizeof(float));
                if (t->data != nullptr) {
                    std::memcpy(t->data, win.data(), win.size() * sizeof(float));
                    LOG_INFO("generate_video_chain: engine-owned track -> %llu frames @%u Hz x%u "
                             "(cut to %lld video frames @%.3f fps)",
                             (unsigned long long)t->sample_count, t->sample_rate, t->channels,
                             total_frames, (double)base_params->fps);
                    free_sd_audio(chain_audio);  // generated audio was conditioning only
                    chain_audio = t;
                } else {
                    free(t);
                    LOG_ERROR("generate_video_chain: track alloc failed; keeping generated audio");
                }
            }
        } else {
            LOG_WARN("generate_video_chain: engine-owned track produced an empty window "
                     "(offset %lld vs %lld video frames); keeping generated audio",
                     audio_offset_frames, flushed_total + (long long)stitched.size());
        }
    }

    // Windowed streaming finalize: drain the residual window, then hand back only metadata + audio
    // (frames were flushed+freed incrementally). Peak frame RAM stayed ≈ WINDOW_KEEP + one segment.
    if (streaming) {
        flush_window(true);
        const int total = (int)flushed_total;
        LOG_INFO("generate_video_chain: streamed %d segments -> %d frames (audio: %llu samples @ %u Hz x%u)",
                 n_chain, total,
                 chain_audio != nullptr ? (unsigned long long)chain_audio->sample_count : 0ULL,
                 chain_audio != nullptr ? chain_audio->sample_rate : 0,
                 chain_audio != nullptr ? chain_audio->channels : 0);
        if (total <= 0) {
            free_sd_audio(chain_audio);
            LOG_ERROR("generate_video_chain: no frames produced");
            return false;
        }
        *frames_out     = nullptr;  // frames already streamed to the caller's encoder
        *num_frames_out = total;
        if (audio_out != nullptr) {
            *audio_out = chain_audio;
        } else {
            free_sd_audio(chain_audio);
        }
        return true;
    }

    const int total = (int)stitched.size();
    LOG_INFO("generate_video_chain: stitched %d segments -> %d frames (audio: %llu samples @ %u Hz x%u)",
             n_chain, total,
             chain_audio != nullptr ? (unsigned long long)chain_audio->sample_count : 0ULL,
             chain_audio != nullptr ? chain_audio->sample_rate : 0,
             chain_audio != nullptr ? chain_audio->channels : 0);
    if (total <= 0) {
        free_sd_audio(chain_audio);
        LOG_ERROR("generate_video_chain: no frames produced");
        return false;
    }

    sd_image_t* out = (sd_image_t*)malloc((size_t)total * sizeof(sd_image_t));
    if (out == nullptr) {
        for (auto& f : stitched) {
            free(f.data);
        }
        free_sd_audio(chain_audio);
        LOG_ERROR("generate_video_chain: out-of-memory allocating result array");
        return false;
    }
    for (int i = 0; i < total; ++i) {
        out[i] = stitched[i];  // shallow copy; .data ownership transfers to the caller
    }
    *frames_out     = out;
    *num_frames_out = total;
    if (audio_out != nullptr) {
        *audio_out = chain_audio;
    } else {
        free_sd_audio(chain_audio);
    }
    return true;
}

// Deep-copy one decoded frame (owns a fresh .data). Used to keep a window's pixel tail
// (the next window's --control-video) alive after the window's frames are handed to the
// stitched timeline.
static sd_image_t wan_copy_frame(const sd_image_t& s) {
    sd_image_t c = s;
    size_t     sz = (size_t)s.width * s.height * s.channel;
    c.data        = (uint8_t*)malloc(sz);
    if (c.data != nullptr && s.data != nullptr) {
        std::memcpy(c.data, s.data, sz);
    }
    return c;
}

// Per-window VACE env (A2-safe: always setenv/unsetenv so a warm resident worker can never
// inherit a prior render's ramp). VACE_SKIP_BLOCKS=0 is correct on EVERY window (the Kijai
// block-0 flash/grid fix). The strength ramp (VACE_STRENGTH_TAIL + ANCHOR_FRAMES) is correct
// whenever there is a live control residual to attenuate toward the tail — i.e. a continuation
// window (overlap-frame anchor) OR an i2v render (init-image ref anchor). Without the ramp, an
// i2v ref over-constrains every frame uniformly → vertical striping on the moving parts (hands).
// Only a PURE base t2v (no control at all) must skip it (VACE_STRENGTH=0 there anyway).
//
// PER-MODE ramp (fixes the i2v identity loss): continuation and i2v want DIFFERENT curves.
//  - Continuation: the injected prior-window latent carries identity+motion at full strength on
//    the anchor frames, so the tail can drop hard (0.2) to loosen the seam. Floor 0.2 / anchor 2
//    (anchor later re-tracked to klat in the is_cont block).
//  - i2v base: the ONLY identity anchor is the VACE reference-image slot at latent t=0; its
//    influence on later frames rides the SAME control residual the ramp attenuates. A hard 0.2
//    tail therefore starves identity through the second half of the clip (a bearded male init
//    drifted female / lost the beard). Use a MILDER, higher-floor ramp (0.5) + a longer full-
//    strength anchor (3 = ref slot + 2 leading frames) so the reference holds across the whole
//    clip while STILL halving the uniform gray-residual on the moving tail (the hand-striping the
//    ramp was added to relieve). This is the striping<->identity balance: lower floor = cleaner
//    hands but weaker identity, higher = stronger identity but more striping; 0.5/3 is the eye-
//    test midpoint start. WAN_VACE_STRENGTH_TAIL/_ANCHOR_FRAMES override the cont curve;
//    WAN_VACE_I2V_STRENGTH_TAIL/_ANCHOR_FRAMES override the i2v curve.
static void apply_wan_vace_env(bool has_control_residual, bool is_cont_window) {
    setenv("VACE_SKIP_BLOCKS", "0", 1);
    if (has_control_residual) {
        const char* tail;
        const char* anchor;
        const char* tail_dflt;
        const char* anchor_dflt;
        if (is_cont_window) {
            tail        = getenv("WAN_VACE_STRENGTH_TAIL");
            anchor      = getenv("WAN_VACE_STRENGTH_ANCHOR_FRAMES");
            tail_dflt   = "0.2";
            anchor_dflt = "2";
        } else {
            // i2v base: milder, higher-floor ramp so the init-image reference holds identity.
            tail        = getenv("WAN_VACE_I2V_STRENGTH_TAIL");
            anchor      = getenv("WAN_VACE_I2V_STRENGTH_ANCHOR_FRAMES");
            tail_dflt   = "0.5";
            anchor_dflt = "3";
        }
        setenv("VACE_STRENGTH_TAIL", (tail != nullptr && tail[0] != '\0') ? tail : tail_dflt, 1);
        setenv("VACE_STRENGTH_ANCHOR_FRAMES", (anchor != nullptr && anchor[0] != '\0') ? anchor : anchor_dflt, 1);
    } else {
        unsetenv("VACE_STRENGTH_TAIL");
        unsetenv("VACE_STRENGTH_ANCHOR_FRAMES");
    }
}

SD_API bool generate_wan_vace_chain(sd_ctx_t*                    sd_ctx,
                                    const sd_vid_gen_params_t*   base_params,
                                    const sd_vid_chain_params_t* chain_params,
                                    sd_image_t**                 frames_out,
                                    int*                         num_frames_out,
                                    sd_audio_t**                 audio_out) {
    if (frames_out != nullptr) {
        *frames_out = nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = 0;
    }
    if (audio_out != nullptr) {
        *audio_out = nullptr;  // wan2.2-VACE t2v carries no audio track
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || base_params == nullptr || chain_params == nullptr ||
        frames_out == nullptr || num_frames_out == nullptr) {
        LOG_ERROR("generate_wan_vace_chain: null argument");
        return false;
    }

    const int n_chain = chain_params->n_segments;
    if (n_chain < 1) {
        LOG_ERROR("generate_wan_vace_chain: n_segments must be >= 1 (got %d)", n_chain);
        return false;
    }

    // Continuation knobs — the ground-truth wan chain recipe (chain_long.sh / render_shot.sh):
    //   K       = pixel overlap frames carried between windows (--control-video tail length).
    //   discard = degraded terminal PIXEL frames dropped from each continuation window.
    //   droplat = degraded terminal LATENT frames dropped when rolling the latent forward.
    auto env_int = [](const char* k, int dflt) {
        const char* e = getenv(k);
        if (e != nullptr && e[0] != '\0') {
            int v = atoi(e);
            if (v > 0) {
                return v;
            }
        }
        return dflt;
    };
    int K       = env_int("WAN_CHAIN_K", 5);
    int discard = env_int("WAN_CHAIN_DISCARD", 4);
    int droplat = env_int("WAN_CHAIN_DROPLAT", 1);

    // Mode signal: an init image on the template = i2v / ref-anchored (window 0 uses it, and
    // continuation windows re-carry it as the VACE identity anchor). No init image = pure t2v
    // (window 0 renders VACE_STRENGTH=0 clean, the render_shot base window).
    const bool is_t2v = (base_params->init_image.data == nullptr);
    LOG_INFO("generate_wan_vace_chain: %d windows, K=%d overlap px, discard=%d, droplat=%d, mode=%s",
             n_chain, K, discard, droplat, is_t2v ? "t2v" : "i2v/ref");

    // Keep the dual-MoE DiT weights warm across windows, and reclaim the transient GPU working
    // set (activations + offloaded params + cache buffers) BETWEEN windows so window 2's compute
    // buffer doesn't stack on window 1's — the shell gets this clean slate for free by running
    // each window in a fresh process. Freed buffers return to the ggml VMM pool for the next
    // window to reuse; we deliberately do NOT trim the pool to the OS (real cuMemUnmap), because
    // that invalidates the resident cublas handle's workspace and faults the next GEMM. The peak
    // lever is --max-vram (the graph-cut sizes the compute buffer to the budget): the wan chain
    // recipe uses a low budget (chain_long.sh: --max-vram 3) so the per-window buffer is small
    // enough that two windows coexist. (No text-cond precompute: umT5 re-encodes per window
    // cheaply and the LTXAV precompute path is gemma-specific.)
    //
    // Keep the DiT weights resident across windows (default; mirrors the LTXAV chain). This is
    // the architecturally-correct intent: the weights stay warm and each window re-streams them.
    //
    // FIXED (2026-07-04): the Wan2.2 DUAL-MoE + --offload-to-cpu warm worker used to fault on a
    //   SECOND in-process render (window 2, or a second single-window job). Root cause was a
    //   residency ASYMMETRY between the two experts in generate_video_ex: the low-noise
    //   diffusion_model's post-sample free is guarded by (free_params_immediately &&
    //   !keep_diffusion_model_resident) so a warm chain KEEPS it, but the high-noise expert's
    //   free was guarded by free_params_immediately ALONE — so on a warm resident chain window 1
    //   still called high_noise_diffusion_model->free_params_buffer(), which nulls the param
    //   tensor pointers. There is no per-render DiT reload, so window 2's high-noise phase read
    //   the nulled buffer -> GGML_ASSERT(buffer). The high-noise free now mirrors the low-noise
    //   resident guard (both sites in the plan.high_noise_sample_steps>0 block), so both experts
    //   survive across windows; release_chain_segment_gpu_residency() still reclaims their GPU
    //   working set between windows. The in-memory cont-latent hand-off already worked.
    //   WAN_CHAIN_RESIDENT=0 forces per-window free (for A/B of that path).
    const bool wan_chain_resident =
        !(getenv("WAN_CHAIN_RESIDENT") != nullptr && getenv("WAN_CHAIN_RESIDENT")[0] == '0');
    sd_ctx_keep_diffusion_model_resident(sd_ctx, wan_chain_resident);

    std::vector<sd_image_t> stitched;       // the output timeline (adopts each kept frame's .data)
    std::vector<sd_image_t> control_tail;   // last K kept px frames -> next window --control-video (owned)
    std::vector<float>      prior_latent;   // full prior-window diffusion latent (owned)
    int                     pl_w = 0, pl_h = 0, pl_t = 0, pl_c = 0;

    // ── Continuation contrast/std-ratchet corrector (fixes the chain "speeds up" percept) ──
    // The distilled few-step DiT inflates the output latent's per-channel mean+std a little every
    // window; contrast-normalised motion is flat across windows but raw brightness/contrast
    // ratchet MONOTONICALLY (measured brightness 58->61->67, std 45->48 over 3 windows), and the
    // pixel exposure-match only aligns each seam to the prior (already-drifted) tail so the intra-
    // window slope COMPOUNDS — the higher contrast makes the same motion "pop" (reads as speed-up)
    // and amplifies the tail gray-residual striping. WAN_CONT_OUTPUT_AGC anchors each window's
    // WHOLE output latent to window-0's per-channel mean+std BEFORE decode/bank (stops the ratchet
    // at source: the regraded latent is also what feeds the next window). It previously read a ref
    // FILE the warm in-memory chain never wrote, so it was a dead no-op in prod; here we bank
    // window-0's latent to a one-shot ref file and point the corrector at it. Default ON for a
    // multi-window chain; WAN_CHAIN_NO_OUTPUT_AGC=1 (or WAN_CONT_OUTPUT_AGC=0) disables.
    const bool chain_output_agc =
        n_chain > 1 &&
        !(getenv("WAN_CHAIN_NO_OUTPUT_AGC") != nullptr && getenv("WAN_CHAIN_NO_OUTPUT_AGC")[0] == '1') &&
        !(getenv("WAN_CONT_OUTPUT_AGC") != nullptr && getenv("WAN_CONT_OUTPUT_AGC")[0] == '0');
    std::string agc_ref_path;  // one-shot window-0 latent ref file (removed at chain end)
    auto cleanup_output_agc = [&]() {
        unsetenv("WAN_CONT_OUTPUT_AGC");
        unsetenv("WAN_CONT_AGC_REF");
        if (!agc_ref_path.empty()) {
            std::remove(agc_ref_path.c_str());
            agc_ref_path.clear();
        }
    };

    auto free_control_tail = [&]() {
        for (auto& f : control_tail) {
            free(f.data);
        }
        control_tail.clear();
    };

    // ── Resume: rebuild the banked prefix [0, resume_from) from save_dir/seg_<i>.bin ──
    int start_seg = 0;
    if (chain_params->resume_from > 0 && chain_params->save_dir != nullptr &&
        chain_params->save_dir[0] != '\0') {
        const int         resume_k = std::min(chain_params->resume_from, n_chain - 1);
        const std::string sd_dir   = chain_params->save_dir;
        LOG_INFO("generate_wan_vace_chain: RESUME from window %d/%d (reloading banked prefix from %s)",
                 resume_k, n_chain, sd_dir.c_str());
        bool ok = true;
        for (int seg = 0; seg < resume_k && ok; ++seg) {
            std::string       p   = sd_dir + "/seg_" + std::to_string(seg) + ".bin";
            int               cnt = 0;
            sd_image_t*       fr  = decode_banked_video_latent(sd_ctx, p, &cnt);
            if (fr == nullptr || cnt <= 0) {
                free(fr);
                ok = false;
                break;
            }
            int drop_head = (seg == 0) ? 0 : std::min(K, cnt);
            int keep      = (seg == 0) ? cnt : std::max(0, cnt - discard);
            // The LAST prefix window seeds the resumed continuation: its full latent (prior_latent)
            // and its kept pixel tail (control_tail) must survive the frame hand-off, so copy first.
            if (seg == resume_k - 1) {
                try {
                    sd::Tensor<float> lat = sd::load_tensor_from_file_as_tensor<float>(p);
                    pl_w = (int)lat.shape()[0];
                    pl_h = (int)lat.shape()[1];
                    pl_t = (int)lat.shape()[2];
                    pl_c = (int)(lat.dim() > 3 ? lat.shape()[3] : 1);
                    prior_latent.assign(lat.data(), lat.data() + lat.numel());
                } catch (const std::exception& e) {
                    LOG_ERROR("generate_wan_vace_chain: resume latent load failed (%s): %s", p.c_str(), e.what());
                    ok = false;
                }
                int tstart = std::max(drop_head, keep - K);
                free_control_tail();
                for (int i = tstart; i < keep; ++i) {
                    control_tail.push_back(wan_copy_frame(fr[i]));
                }
            }
            for (int i = 0; i < cnt; ++i) {
                if (i >= drop_head && i < keep) {
                    stitched.push_back(fr[i]);
                } else {
                    free(fr[i].data);
                }
            }
            free(fr);
        }
        if (ok && !prior_latent.empty()) {
            start_seg = resume_k;
        } else {
            LOG_WARN("generate_wan_vace_chain: resume failed — falling back to a fresh full render");
            for (auto& f : stitched) {
                free(f.data);
            }
            stitched.clear();
            free_control_tail();
            prior_latent.clear();
            start_seg = 0;
        }
    }

    for (int seg = start_seg; seg < n_chain; ++seg) {
        const bool          is_cont = (seg > 0);
        sd_vid_gen_params_t vp      = *base_params;
        vp.seed                     = (base_params->seed < 0) ? base_params->seed : base_params->seed + seg;
        vp.stage_seg_index          = seg;  // stage previews (emit_stages) report THIS window (Wan has no refine → inert)

        if (chain_params->segment_prompts != nullptr && chain_params->segment_prompts[seg] != nullptr) {
            vp.prompt = chain_params->segment_prompts[seg];
        }

        // Per-window VACE env + prior-window in-memory hand-off. The tail ramp applies wherever
        // there's a control residual: a continuation window (is_cont) OR an i2v render (!is_t2v,
        // init-image ref). Only a pure base t2v window skips it. Fixes i2v hand-striping.
        apply_wan_vace_env(is_cont || !is_t2v, is_cont);
        unsetenv("VACE_CONT_LATENT");  // never let a stale disk path shadow the in-memory tail
        if (!is_cont) {
            vp.control_frames      = nullptr;
            vp.control_frames_size = 0;
            vp.cont_latent         = nullptr;
            vp.cont_latent_frames  = 0;
            unsetenv("VACE_CONT_FRAMES");
            unsetenv("VACE_CONT_LATENT_DROP_TAIL");
            if (is_t2v) {
                setenv("VACE_STRENGTH", "0", 1);  // clean base t2v (no control residual)
            } else {
                unsetenv("VACE_STRENGTH");  // i2v: full VACE ref-anchor strength
            }
        } else {
            vp.control_frames      = control_tail.data();
            vp.control_frames_size = (int)control_tail.size();
            vp.cont_latent         = prior_latent.data();
            vp.cont_latent_frames  = pl_t;  // FULL prior-window latent T (last K sliced in-engine)
            setenv("VACE_CONT_FRAMES", std::to_string(K).c_str(), 1);
            // DROP_TAIL alignment (render_shot.sh:205 vs :238): the FIRST continuation rolls forward
            // from the pristine BASE window (window 0), whose latent tail is clean, so PRIOR_DROP=0;
            // only LATER continuations (prior window is itself a continuation, hence striping-
            // degraded) drop the terminal latent frame (DROPLAT). We were dropping droplat on EVERY
            // continuation incl. the first, discarding one clean base-tail latent frame at seam 1.
            // prior-window index == seg-1, so the prior is the pristine base iff seg==1 (holds on
            // resume too: window 0 is always the base). Match the shell: 0 on the first cut, droplat after.
            const int this_drop = (seg == 1) ? 0 : droplat;
            setenv("VACE_CONT_LATENT_DROP_TAIL", std::to_string(this_drop).c_str(), 1);
            unsetenv("VACE_STRENGTH");  // full strength; the tail ramp attenuates the seam

            // ANCHOR the FULL injected-context region at full VACE strength — ramp ONLY the
            // genuinely-free frames. The prior-window tail is injected into the first
            //   klat = (min(K, frames) - 1)/4 + 1   (Wan VAE temporal factor 4)
            // latent frames (mask=0); those carry the anti-drift motion/identity prior and must
            // stay at full strength. The ramp's ANCHOR_FRAMES default (apply_wan_vace_env) is a
            // STATIC 2 — correct only when klat==2, i.e. the OLD K=5. The recipe raised K to 13
            // (klat=4), so 2 of the 4 context frames fell INSIDE the ramp's attenuation zone,
            // weakening the anchor exactly at the context->free seam. Two consequences, both
            // compounding once the prior tail is itself a degraded continuation (seg>=2, where the
            // second seam lives): (1) the under-anchored boundary lets striping bleed through
            // (first seam clean because window 0's tail is pristine); (2) the continuation no
            // longer faithfully re-plays the K overlap it is meant to reproduce, so motion "runs
            // ahead" inside the dropped head and the kept frames start further along each cut ->
            // the chain drifts progressively faster. Tracking the anchor to klat (+1 for the i2v
            // ref slot, which occupies latent t=0) keeps the whole injected context at full
            // strength regardless of K. Only widens the anchored region (never removes the
            // free-frame ramp), so it is a strict anti-drift improvement; user override still wins.
            const bool keep_ref =
                getenv("WAN_CHAIN_KEEP_REF") != nullptr && getenv("WAN_CHAIN_KEEP_REF")[0] == '1';
            if (!(getenv("WAN_VACE_STRENGTH_ANCHOR_FRAMES") != nullptr &&
                  getenv("WAN_VACE_STRENGTH_ANCHOR_FRAMES")[0] != '\0')) {
                const int  frames      = std::max(1, base_params->video_frames);
                const int  klat        = (std::min(K, frames) - 1) / 4 + 1;
                const bool ref_on_cont = keep_ref && base_params->init_image.data != nullptr;
                const int  anchor      = klat + (ref_on_cont ? 1 : 0);
                setenv("VACE_STRENGTH_ANCHOR_FRAMES", std::to_string(anchor).c_str(), 1);
            }

            // REFERENCE-IMAGE CONFLICT FIX — the "window-1-appears-as-a-hat" continuation bug.
            // render_shot.sh:224 gates --init-img to i2v mode ONLY: a t2v continuation window passes
            // NO reference image and is driven PURELY by --control-video (the kept K-frame pixel tail)
            // + VACE_CONT_LATENT (the prior window's diffusion-latent tail). In this warm single-
            // VACE-model server the prior window's identity+motion is ALREADY carried losslessly by
            // the in-memory cont_latent, so re-feeding the request's init/reference image into the
            // VACE reference-image slot on a continuation window is redundant AND actively harmful:
            // the VACE ref path (stable-diffusion.cpp:7222-7231 encode + 7482-7513 prepend) composites
            // that full frame as a small MISPLACED spatial reference object ("hat") that fights the
            // control continuation and pulls the generated identity off the prior window. Keep the
            // reference only on window 0 (identity/scene establishment) and let cont_latent+control
            // carry every continuation — this matches the render_shot t2v recipe and the in-memory
            // hand-off's own design intent. WAN_CHAIN_KEEP_REF=1 restores the old re-fed-ref behaviour
            // for A/B (e.g. to mimic render_shot i2v, which re-anchors each window with a CLEAN
            // identity still rather than a busy full-scene frame).
            if (!keep_ref) {
                vp.init_image.data = nullptr;  // skip the VACE reference-image path this window
            }
        }

        // Output-AGC (contrast/std de-ratchet): regrade THIS continuation window's output latent
        // to window-0's per-channel mean+std before decode/bank. Window 0 establishes the ref and
        // must NEVER be regraded (it IS the reference). A2-safe: set/unset every window.
        if (chain_output_agc && is_cont && !agc_ref_path.empty()) {
            setenv("WAN_CONT_OUTPUT_AGC", "1", 1);
            setenv("WAN_CONT_AGC_REF", agc_ref_path.c_str(), 1);
        } else {
            unsetenv("WAN_CONT_OUTPUT_AGC");
        }

        LOG_INFO("=== generate_wan_vace_chain window %d/%d [%s] ===",
                 seg + 1, n_chain, is_cont ? "vace-cont" : (is_t2v ? "t2v-base" : "i2v-base"));

        sd_image_t* seg_video   = nullptr;
        int         seg_count   = 0;
        sd_audio_t* seg_audio   = nullptr;
        float*      lat_out     = nullptr;
        int         lw = 0, lh = 0, lt = 0, lc = 0;
        const bool  want_latent = (seg + 1 < n_chain);
        if (!generate_video_ex(sd_ctx, &vp, &seg_video, &seg_count, &seg_audio, nullptr,
                               want_latent ? &lat_out : nullptr,
                               want_latent ? &lw : nullptr, want_latent ? &lh : nullptr,
                               want_latent ? &lt : nullptr, want_latent ? &lc : nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr) ||
            seg_video == nullptr || seg_count <= 0) {
            LOG_ERROR("generate_wan_vace_chain window %d failed", seg + 1);
            free_sd_audio(seg_audio);
            free(seg_video);
            free(lat_out);
            free_control_tail();
            cleanup_output_agc();
            for (auto& f : stitched) {
                free(f.data);
            }
            return false;
        }
        free_sd_audio(seg_audio);

        // Carry this window's full latent forward (the VACE_CONT_LATENT-equivalent, in memory).
        if (want_latent && lat_out != nullptr) {
            prior_latent.assign(lat_out, lat_out + (size_t)lw * lh * lt * lc);
            pl_w = lw;
            pl_h = lh;
            pl_t = lt;
            pl_c = lc;
        }
        free(lat_out);

        // Bank window-0's pristine output latent ONCE as the output-AGC reference, so every
        // later window anchors its per-channel mean+std to it and the contrast ratchet stops at
        // source. Uses save_dir when banking is on, else a temp file (removed at chain end).
        if (chain_output_agc && seg == 0 && want_latent && !prior_latent.empty()) {
            const bool have_save_dir =
                chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0';
            const char* td   = getenv("TMPDIR");
            std::string base = have_save_dir
                                   ? std::string(chain_params->save_dir)
                                   : std::string(td != nullptr && td[0] != '\0' ? td : "/tmp");
            agc_ref_path = base + "/wan_agc_ref_seg0_" + std::to_string((long long)ggml_time_ms()) + ".bin";
            try {
                sd::Tensor<float> ref_save({pl_w, pl_h, pl_t, pl_c, 1});
                std::memcpy(ref_save.data(), prior_latent.data(), (size_t)ref_save.numel() * sizeof(float));
                sd::save_tensor_to_file<float>(agc_ref_path, ref_save, "wan_agc_ref");
                LOG_INFO("generate_wan_vace_chain: banked window-0 output-AGC ref -> %s", agc_ref_path.c_str());
            } catch (const std::exception& e) {
                LOG_WARN("generate_wan_vace_chain: output-AGC ref bank failed: %s (de-ratchet disabled)",
                         e.what());
                agc_ref_path.clear();
            }
        }

        // Bank the full latent so a failed chain can resume from the last completed window.
        if (want_latent && chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0' &&
            !prior_latent.empty()) {
            try {
                sd::Tensor<float> lt_save({pl_w, pl_h, pl_t, pl_c, 1});
                std::memcpy(lt_save.data(), prior_latent.data(), (size_t)lt_save.numel() * sizeof(float));
                sd::save_tensor_to_file<float>(
                    std::string(chain_params->save_dir) + "/seg_" + std::to_string(seg) + ".bin",
                    lt_save, "wan_vace_video_latent");
            } catch (const std::exception& e) {
                LOG_WARN("generate_wan_vace_chain: bank latent seg %d failed: %s", seg, e.what());
            }
        }

        // Stitch: window 0 keeps every frame; a continuation drops the re-rendered overlap head
        // (first K px, the held context) and the degraded discard tail. keep = seg_count-discard.
        const int drop_head = is_cont ? std::min(K, seg_count) : 0;
        const int keep      = is_cont ? std::max(0, seg_count - discard) : seg_count;

        // Build the next window's --control-video BEFORE handing frames to the timeline: the
        // last K frames of [0, keep) (chain_long.sh: last K of the kept, pre-overlap-drop).
        if (want_latent) {
            const int tstart = std::max(drop_head, keep - K);
            free_control_tail();
            for (int i = tstart; i < keep; ++i) {
                control_tail.push_back(wan_copy_frame(seg_video[i]));
            }
        }

        // Seam continuity: tone-match this window's kept frames to the prior tail (no cross-fade;
        // the python colour-lock post-pass is a further quality refinement, tracked separately).
        if (is_cont && !stitched.empty() && keep - drop_head > 0) {
            ltxav_exposure_match(stitched.data(), (int)stitched.size(),
                                 seg_video + drop_head, keep - drop_head);
        }

        const size_t kept_start = stitched.size();
        for (int i = 0; i < seg_count; ++i) {
            if (i >= drop_head && i < keep) {
                stitched.push_back(seg_video[i]);  // adopt ownership
            } else {
                free(seg_video[i].data);
            }
        }
        free(seg_video);

        // Hand the kept frames to the server so it can bank a viewable per-window webm preview.
        const int kept_n = (int)(stitched.size() - kept_start);
        if (chain_params->on_segment != nullptr && kept_n > 0) {
            chain_params->on_segment(seg, stitched.data() + kept_start, kept_n, chain_params->on_segment_user);
        }

        // Reclaim the per-window GPU working set before the next window's footprint stacks on
        // top (only meaningful on the resident chain; the default per-window free+reload already
        // returns to a clean slate). WAN_NO_CHAIN_GPU_RECLAIM=1 disables for A/B.
        if (wan_chain_resident && seg + 1 < n_chain && getenv("WAN_NO_CHAIN_GPU_RECLAIM") == nullptr) {
            sd_ctx->sd->release_chain_segment_gpu_residency();
        }
    }

    free_control_tail();

    // A2-safety: don't let the per-window VACE ramp/continuation env leak into a later render.
    unsetenv("VACE_STRENGTH_TAIL");
    unsetenv("VACE_STRENGTH_ANCHOR_FRAMES");
    unsetenv("VACE_CONT_FRAMES");
    unsetenv("VACE_CONT_LATENT_DROP_TAIL");
    unsetenv("VACE_STRENGTH");
    cleanup_output_agc();

    const int total = (int)stitched.size();
    if (total <= 0) {
        LOG_ERROR("generate_wan_vace_chain: no frames produced");
        return false;
    }
    sd_image_t* out = (sd_image_t*)malloc((size_t)total * sizeof(sd_image_t));
    if (out == nullptr) {
        for (auto& f : stitched) {
            free(f.data);
        }
        LOG_ERROR("generate_wan_vace_chain: out-of-memory allocating result array");
        return false;
    }
    for (int i = 0; i < total; ++i) {
        out[i] = stitched[i];
    }
    *frames_out     = out;
    *num_frames_out = total;
    LOG_INFO("generate_wan_vace_chain: stitched %d windows -> %d frames", n_chain, total);
    return true;
}

SD_API void sd_ctx_keep_diffusion_model_resident(sd_ctx_t* sd_ctx, bool keep) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        sd_ctx->sd->keep_diffusion_model_resident = keep;
    }
}

SD_API void sd_ctx_precompute_chain_text_conds(sd_ctx_t*    sd_ctx,
                                               const char** prompts,
                                               int          n_prompts,
                                               const char*  negative_prompt,
                                               int          clip_skip,
                                               bool         need_uncond) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || prompts == nullptr || n_prompts <= 0) {
        return;
    }
    std::vector<std::string> ps;
    ps.reserve(n_prompts);
    for (int i = 0; i < n_prompts; ++i) {
        ps.emplace_back(prompts[i] != nullptr ? prompts[i] : "");
    }
    sd_ctx->sd->precompute_chain_text_conds(ps, negative_prompt != nullptr ? negative_prompt : "", clip_skip, need_uncond);
}

SD_API bool sd_ctx_swap_diffusion_model(sd_ctx_t* sd_ctx, const char* diffusion_model_path) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || diffusion_model_path == nullptr) {
        return false;
    }
    // Hot-swap the DiT weights in place. The VAE + text encoder stay resident; an
    // LTXAV swap also tears down the outgoing DiT's GPU-only residency before
    // refilling the runner from the new gguf. Caller MUST ensure no render is in flight.
    return sd_ctx->sd->swap_diffusion_model(diffusion_model_path);
}

SD_API void sd_ctx_set_diffusion_model_residency_id(sd_ctx_t* sd_ctx, const char* model_id) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || !sd_ctx->sd->diffusion_model) {
        return;
    }
    sd_ctx->sd->diffusion_model->set_residency_model_id(model_id != nullptr ? model_id : "");
}

SD_API void sd_ctx_free_diffusion_model(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || !sd_ctx->sd->diffusion_model) {
        return;
    }
    // Free the DiT compute + param buffers (releases its VRAM) without touching the
    // resident VAE/text-encoder. Used by /v1/admin/unload so the external GPU gate
    // can reclaim the card for the LLM/avatar. The next render must reload the DiT
    // via sd_ctx_swap_diffusion_model() first. free_params_buffer() nulls the tensor
    // data/buffer pointers so a later alloc+reload is clean.
    sd_ctx->sd->diffusion_model->free_compute_buffer();
    sd_ctx->sd->diffusion_model->free_params_buffer();
    // Also drop any swapped-DiT mmap so its file-backed pages are released (the
    // boot variant's mapping in mmap_tensor_store is left alone; a reload
    // re-maps via sd_ctx_swap_diffusion_model()).
    sd_ctx->sd->dit_swap_mmap_store.clear();
}

SD_API float* sd_ctx_encode_video_frames(sd_ctx_t* sd_ctx,
                                         const sd_image_t* frames,
                                         int num_frames,
                                         int width,
                                         int height,
                                         int* latent_width_out,
                                         int* latent_height_out,
                                         int* latent_frames_out,
                                         int* latent_channels_out) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || frames == nullptr || num_frames <= 0) {
        return nullptr;
    }
    // Assemble the RGB frame stack into a [W, H, T, C, 1] [0,1] tensor (the layout
    // encode_first_stage expects for the Wan-VAE temporal encode), then VAE-encode to
    // a DIFFUSION latent — exactly the space generate_video_ex hands back / the next
    // segment consumes as cont_latent. This is the chaining drift sink.
    int C = (int)frames[0].channel;
    sd::Tensor<float> x({(int64_t)width, (int64_t)height, (int64_t)num_frames, (int64_t)C, 1});
    for (int t = 0; t < num_frames; ++t) {
        const sd_image_t& img = frames[t];
        if ((int)img.width != width || (int)img.height != height || (int)img.channel != C) {
            LOG_ERROR("sd_ctx_encode_video_frames: frame %d shape %ux%ux%u != expected %dx%dx%d",
                      t, img.width, img.height, img.channel, width, height, C);
            return nullptr;
        }
        for (int ic = 0; ic < C; ++ic) {
            for (int ih = 0; ih < height; ++ih) {
                for (int iw = 0; iw < width; ++iw) {
                    x.index(iw, ih, t, ic, 0) = sd_image_get_f32(img, iw, ih, ic, true);
                }
            }
        }
    }

    auto latent = sd_ctx->sd->encode_first_stage(x);  // diffusion latent [W,H,T,C,1]
    if (latent.empty()) {
        LOG_ERROR("sd_ctx_encode_video_frames: VAE encode failed");
        return nullptr;
    }
    int64_t Wl = latent.shape()[0];
    int64_t Hl = latent.shape()[1];
    int64_t Tl = latent.shape()[2];
    int64_t Cl = latent.dim() > 3 ? latent.shape()[3] : 1;
    size_t n   = (size_t)latent.numel();
    float* buf = (float*)malloc(n * sizeof(float));
    if (buf == nullptr) {
        return nullptr;
    }
    std::memcpy(buf, latent.data(), n * sizeof(float));
    if (latent_width_out) *latent_width_out = (int)Wl;
    if (latent_height_out) *latent_height_out = (int)Hl;
    if (latent_frames_out) *latent_frames_out = (int)Tl;
    if (latent_channels_out) *latent_channels_out = (int)Cl;
    return buf;
}
