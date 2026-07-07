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
                                     int                             clip_skip) {
        if ((!sd_version_is_longcat_avatar(version) && version != VERSION_LTXAV) ||
            !keep_diffusion_model_resident || !cond_stage_model) {
            return;
        }
        if (!reload_cond_stage_model()) {
            LOG_ERROR("LTXAV chain: TE reload failed; cannot pre-encode text conds");
            return;
        }
        int64_t t0 = ggml_time_ms();
        // Constant negative prompt across the chain -> encode its uncond once, share it.
        SDCondition shared_uncond;
        {
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
            entry.has_uncond       = true;
            avatar_cond_cache[key] = std::move(entry);
            ++encoded;
        }
        LOG_INFO("LTXAV chain: pre-encoded %d distinct text cond(s) over %zu segment prompt(s) "
                 "in one TE window, taking %.2fs",
                 encoded, prompts.size(), (ggml_time_ms() - t0) * 1.0f / 1000);
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

        // 1. Free the current DiT compute + params buffers (releases VRAM and any
        //    anon params backing). free_params_buffer() nulls the tensor
        //    data/buffer pointers so the re-alloc below doesn't trip the
        //    "already allocated" fast-path. For an mmap'd DiT the weight data
        //    lives in a MmapTensorStore (the boot variant in `mmap_tensor_store`,
        //    a prior swap in `dit_swap_mmap_store`), not the params buffer.
        diffusion_model->free_compute_buffer();
        diffusion_model->free_params_buffer();

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
                             bool ltxav_audio_fixed                    = false,
                             const sd::Tensor<float>& video_reference  = {}) {
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
                // Driven audio is held fixed (clean): tell the model its audio tokens are
                // at timestep 0, just like i2v zeroes the fixed frame-0 video timesteps.
                std::vector<float> audio_ts = ltxav_audio_fixed
                                                  ? std::vector<float>(base_timesteps_vec.size(), 0.0f)
                                                  : base_timesteps_vec;
                audio_timesteps_tensor = sd::Tensor<float>({static_cast<int64_t>(audio_ts.size())}, audio_ts);
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
        if (inverse_noise_scaling) {
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
        if (version != VERSION_SD1_PIX2PIX) {
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
        // the SDPA + conv3d execution-plan caches are process-global, keyed by shape,
        // and pin cuDNN-backend device memory that is NEVER returned to the driver and
        // that ggml_backend_cuda_trim_pools cannot reach (it lives outside the ggml VMM
        // pool). A continuation segment's LONGER base pass (e.g. 16 latent frames = 13
        // gen + 3 keyframe) and the first segment's VAE decode each build a bigger plan
        // whose reservation then squats through EVERY later phase — measured as a uniform
        // ~+1.7 GB reserve-time high-water on seg-2's refine AND decode vs seg-1 (which
        // ran before those plans existed). Dropping the plan caches at the segment
        // boundary returns that memory to the driver; the next segment rebuilds only the
        // plans it actually needs (one-time build, not per-step → the refine/decode is
        // not slowed). Nothing is in flight here (all segment compute has completed and
        // synced). No-op on non-cuDNN builds.
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
    "euler_a_cfg_pp",
    "euler_ge",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
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
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->fps                                   = 16;
    sd_vid_gen_params->drive_audio_path                      = nullptr;
    sd_vid_gen_params->cont_latent_path                      = nullptr;
    sd_vid_gen_params->cont_anchor_path                      = nullptr;
    sd_vid_gen_params->cont_latent                           = nullptr;
    sd_vid_gen_params->cont_latent_frames                    = 0;
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

        if (guidance->img_cfg != guidance->txt_cfg) {
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
        moe_boundary = sd_vid_gen_params->moe_boundary;
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

        if (env_sigmas.size() >= 2) {
            sigmas      = env_sigmas;
            total_steps = static_cast<int>(sigmas.size()) - 1;
            sample_steps = std::min(sample_steps, total_steps);
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
            }
            LOG_INFO("LTX_CUSTOM_SIGMAS override: %d sigmas => %d steps [%.5f .. %.5f]",
                     static_cast<int>(sigmas.size()), total_steps, sigmas.front(), sigmas.back());
        } else if (sample_params->custom_sigmas_count > 0) {
            sigmas      = std::vector<float>(sample_params->custom_sigmas,
                                        sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set total_steps to %d", total_steps);
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set high_noise_sample_steps to %d", high_noise_sample_steps);
            }
        } else {
            scheduler_t scheduler = resolve_scheduler(sd_ctx,
                                                      sample_params->scheduler,
                                                      sample_method);
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

    for (int64_t t = 0; t < keyframe_latent_frames; t++) {
        float t_start = static_cast<float>(keyframe_frame_idx + t * temporal_scale);
        float t_end   = static_cast<float>(keyframe_frame_idx + (t + 1) * temporal_scale);
        if (keyframe_pixel_frames == 1) {
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
                                                                 float audio_mask_value = 1.f) {
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
        // LTX-2.3 V2V lipdub relip (control_frames) requires an 8k+1 frame count (VAE temporal
        // stride 8). Snap here — before the audio length is derived — so the reference clip,
        // target latent, and audio all agree. Gated on control_frames so every other LTXAV path
        // (i2v, t2v, continuation, --drive-audio without control_frames) is byte-identical.
        if (sd_vid_gen_params->control_frames_size > 0) {
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
                    latents.audio_latent = driven;       // lip-sync to THIS audio
                    latents.audio_fixed  = true;         // hold it fixed through the denoise loop
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
        if (sd_vid_gen_params->control_frames_size > 0) {
            // LTX-2.3 V2V LIPDUB RELIP: the control_frames are an existing video clip (e.g. a
            // Wan2.2 render). Encode the WHOLE clip to video latents and append it as a frozen,
            // timeline-aligned IC-LoRA reference (apply_ltxav_video_relip_reference). With the
            // lipdub IC-LoRA loaded (--lora) + a frozen driving-audio latent (--drive-audio), the
            // model preserves the input video and only re-lips the mouth. Off-switch:
            // LTXAV_RELIP_DISABLE restores the legacy "not implemented" rejection for A/B.
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
        if (latents.audio_fixed && latents.denoise_mask.empty()) {
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
        }
        if (!latents.denoise_mask.empty()) {
            latents.denoise_mask = pack_ltxav_audio_and_video_denoise_mask(latents.denoise_mask,
                                                                           latents.init_latent,
                                                                           latents.audio_latent,
                                                                           latents.audio_fixed ? 0.0f : 1.0f);
        }
        latents.init_latent = pack_ltxav_audio_and_video_latents(latents.init_latent, latents.audio_latent);
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
            sd_ctx->sd->cond_stage_model->free_params_buffer();
            LOG_INFO("LTXAV: freed TE params (gemma+projection) before DiT, ~6GB (LTXAV_FREE_TE_PARAMS=1)");
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
    // Forwards via the WAN_ prefix in run_wan22_i2v_nvfp4.sh. Same [W,H,T,C] f32 the
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
        // run_wan_latent_cont.sh greps "[WAN_CHAIN_STATS]".
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

    std::unique_ptr<LTXVUpsampler::LatentUpsamplerRunner> upsampler =
        std::make_unique<LTXVUpsampler::LatentUpsamplerRunner>(sd_ctx->sd->backend_for(SDBackendModule::UPSCALER),
                                                               sd_ctx->sd->params_backend_for(SDBackendModule::UPSCALER));
    const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(sd_ctx->sd->max_vram);
    upsampler->set_max_graph_vram_bytes(max_graph_vram_bytes);
    if (!upsampler->load_from_file(model_path, sd_ctx->sd->n_threads)) {
        LOG_ERROR("load LTX latent upsampler failed");
        return {};
    }

    sd::Tensor<float> upscaled = upsampler->compute(sd_ctx->sd->n_threads, unnormalized);
    upsampler.reset();
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
            *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent,
                                                                    latents.audio_fixed ? 0.0f : 1.0f);
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

    if (sd_vid_gen_params->init_image.data == nullptr &&
        sd_vid_gen_params->end_image.data == nullptr) {
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

    if (sd_vid_gen_params->init_image.data != nullptr) {
        sd::Tensor<float> start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, image_width, image_height);
        if (!apply_ltxav_condition_image_by_latent_index(sd_ctx,
                                                         start_image,
                                                         &video_latent,
                                                         &video_mask,
                                                         0,
                                                         "init",
                                                         conditioning_strength)) {
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
                              float** final_latent_out,
                              int* latent_width_out,
                              int* latent_height_out,
                              int* latent_frames_out,
                              int* latent_channels_out) {
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
    if (final_latent_out != nullptr) {
        *final_latent_out = nullptr;
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
        sd_vid_gen_params->control_frames_size > 0) {
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
        if (request.hires.custom_sigmas_count <= 0 || request.hires.custom_sigmas == nullptr) {
            static float kStage2DistilledSigmas[] = {0.909375f, 0.725f, 0.421875f, 0.0f};
            request.hires.custom_sigmas       = kStage2DistilledSigmas;
            request.hires.custom_sigmas_count = 4;
        }
        // Stage-1 at HALF the final resolution (the relip block below renders at request.w/h).
        request.width  = full_w / 2;
        request.height = full_h / 2;
        LOG_INFO("LTXAV two-stage: stage1 %dx%d from-noise (half-res) -> latent-upscale 2x -> stage2 %dx%d refine",
                 request.width, request.height, full_w, full_h);
    }

    bool latent_upscale_enabled     = request.hires.enabled;
    GenerationRequest hires_request = request;
    if (latent_upscale_enabled) {
        if (!sd_version_is_ltxav(sd_ctx->sd->version)) {
            LOG_ERROR("LTX latent spatial upscale is only supported for LTX video models");
            return false;
        }
        if (request.hires.upscaler != SD_HIRES_UPSCALER_MODEL) {
            LOG_ERROR("LTX latent spatial upscale currently requires hires upscaler MODEL");
            return false;
        }
        if (strlen(SAFE_STR(request.hires.model_path)) == 0) {
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

    ImageGenerationEmbeds embeds = prepare_video_generation_embeds(sd_ctx,
                                                                   sd_vid_gen_params,
                                                                   request,
                                                                   latents);
    if (latent_upscale_enabled) {
        LOG_INFO("generate_video %dx%dx%d -> LTX latent spatial upscale",
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
    // NOT engaged on the relip two-stage path: its stage-2 reference re-encode
    // (apply_ltxv_refine_image_conditioning, relip only) still needs the video VAE AFTER the base
    // sample, so FIX-3 (LTXAV_TWOSTAGE_FREE_UNUSED) frees it at the correct later point (:8907).
    // Opt-in, default off, so no measured baseline shifts.
    static const bool ltxav_vae_lazy = [] {
        const char* s = getenv("LTXAV_VAE_LAZY");
        return s && s[0] == '1';
    }();
    if (ltxav_vae_lazy && !relip_twostage &&
        sd_version_is_ltxav(sd_ctx->sd->version) &&
        sd_ctx->sd->first_stage_model) {
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
            LOG_INFO("LTXAV_VAE_LAZY: released offloaded video+audio VAE GPU params (runtime + shared-resident) + trimmed VAE pool before DiT sample+refine; re-offload from host at decode");
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
    if (final_latent.empty()) {
        final_latent = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                        true,
                                                        x_t,
                                                        std::move(noise),
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
                                                        latents.denoise_mask,
                                                        latents.vace_context,
                                                        request.vace_strength,
                                                        latents.audio_length,
                                                        static_cast<float>(request.fps),
                                                        request.cache_params,
                                                        latents.video_positions,
                                                        latents.audio_fixed,
                                                        latents.video_reference);
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

    // Continuation + hires: when LTX latent spatial upscale is enabled, the continuation
    // latent handed back to the caller (final_latent_out) must be the BASE (pre-upscale)
    // latent, so the NEXT chained segment seeds + samples at the same base resolution it
    // will upscale from. Capture it here BEFORE the hires block overwrites final_latent
    // with the upscaled latent; the decoded frames_out below still come from the upscaled
    // latent. Spatial upscale preserves the temporal frame count, so the caller's overlap
    // bookkeeping is unchanged. (Deep copy: final_latent is reassigned by the hires pass.)
    sd::Tensor<float> chain_base_latent;
    if (latent_upscale_enabled && final_latent_out != nullptr) {
        chain_base_latent = final_latent;
    }

    if (latent_upscale_enabled) {
        int64_t upscale_start             = ggml_time_ms();
        sd::Tensor<float> upscaled_latent = upscale_ltx_spatial_video_latent(sd_ctx,
                                                                             request.hires.model_path,
                                                                             final_latent,
                                                                             latents.audio_length);
        int64_t upscale_end               = ggml_time_ms();
        if (upscaled_latent.empty()) {
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return false;
        }
        LOG_INFO("LTX latent spatial upscale completed, taking %.2fs",
                 (upscale_end - upscale_start) * 1.0f / 1000);

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
            if (target_audio_length != latents.audio_length) {
                int latent_channels            = sd_ctx->sd->get_latent_channel();
                sd::Tensor<float> video_latent = x_t;
                sd::Tensor<float> audio_latent = latents.audio_latent;
                if (x_t.shape()[3] > latent_channels) {
                    video_latent = sd::ops::slice(x_t, 3, 0, latent_channels);
                    audio_latent = unpack_ltxav_audio_latent(x_t, latents.audio_length, latent_channels);
                }
                audio_latent = resize_ltxav_audio_latent(audio_latent, target_audio_length);
                if (audio_latent.empty()) {
                    LOG_ERROR("failed to resize LTX audio latent for latent upscale: %d -> %d",
                              latents.audio_length,
                              target_audio_length);
                    if (sd_ctx->sd->free_params_immediately) {
                        sd_ctx->sd->diffusion_model->free_params_buffer();
                    }
                    return false;
                }
                x_t                  = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
                latents.audio_latent = std::move(audio_latent);
                LOG_INFO("LTX audio latent length adjusted for latent upscale: %d -> %d",
                         latents.audio_length,
                         target_audio_length);
                latents.audio_length = target_audio_length;
            }
        }
        if ((request.hires.target_width > 0 || request.hires.target_height > 0) &&
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
        if (const char* ctx_env = std::getenv("LTX_REFINE_CONTEXT_FRAMES")) {
            int ctx_n         = atoi(ctx_env);
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
                    LOG_INFO("LTX_REFINE_CONTEXT_FRAMES=%d: refine sliced to %lld surviving + %lld guide-context frames "
                             "(dropped %lld of %lld throwaway guide frames from the refine; audio=%s) -> refine T=%lld",
                             ctx_n, (long long)T_tgt, (long long)keep, (long long)(K_cond - keep),
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

        noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);

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
        sample_method_t hires_sample_method = plan.sample_method;
        int hires_scheduler_steps           = 0;
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

        LOG_DEBUG("sample(latent upscale) %dx%dx%d", W, H, T);
        LOG_INFO("LTX latent spatial upscale refine: scheduler_steps=%d, denoising_strength=%.2f, sampler=%s, sigma_sched_size=%zu%s",
                 hires_scheduler_steps,
                 request.hires.denoising_strength,
                 sampling_methods_str[hires_sample_method],
                 hires_sigma_sched.size(),
                 request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");

        // Pre-sample pool trim for the two-stage stage-2 refine (see the same trim before the
        // stage-1 / single-stage sample above). The latent upscaler (~950 MB offload) and the
        // full-res stage-2 reference VAE encode both just ran; their freed scratch is still
        // committed in the CUDA VMM pool and would otherwise inflate this refine's peak. Trim it
        // back before the refine reserves its compute buffer. Gated by LTXAV_PRE_SAMPLE_POOL_TRIM.
        if (sd_version_is_ltxav(sd_ctx->sd->version)) {
            const char* e = getenv("LTXAV_PRE_SAMPLE_POOL_TRIM");
            if (e == nullptr || std::string(e) != "0") {
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
        // Gated identically to regime A (:8597): !relip_twostage (relip's stage-2 reference re-encode
        // DOES need the VAE during refine — freed later by LTXAV_TWOSTAGE_FREE_UNUSED) +
        // params_offloaded_to_host() (re-offloadable, mmap-safe). Opt-in via LTXAV_VAE_LAZY; default off.
        if (ltxav_vae_lazy && !relip_twostage &&
            sd_version_is_ltxav(sd_ctx->sd->version) &&
            sd_ctx->sd->first_stage_model &&
            sd_ctx->sd->first_stage_model->params_offloaded_to_host()) {
            sd_ctx->sd->first_stage_model->release_all_gpu_param_residency();
            if (sd_ctx->sd->audio_vae_model) {
                sd_ctx->sd->audio_vae_model->release_all_gpu_param_residency();
            }
            ggml_backend_cuda_trim_pools(sd_ctx->sd->backend_for(SDBackendModule::VAE));
            LOG_INFO("LTXAV_VAE_LAZY: re-released offloaded video+audio VAE GPU params (runtime + shared-resident) + trimmed VAE pool before the hires/refine sample; re-offload from host at decode");
        }

        // Refine-scoped resident byte cap (LONGCAT_SHARED_RESIDENT_MAX_MB): engage the
        // cap ONLY for this 3-step hires/refine sample so it pins just the hottest ~4 GB
        // (fits ≤11776 on the 97f 1080p continuation) while the 8-step base above keeps
        // its full pin for speed. No-op when the cap env is unset (byte-identical).
        if (sd_ctx->sd->diffusion_model) {
            sd_ctx->sd->diffusion_model->set_refine_resident_scope(true);
        }
        sampling_start = ggml_time_ms();
        final_latent   = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                            true,
                                            x_t,
                                            std::move(noise),
                                            embeds.cond,
                                          hires_request.use_uncond ? embeds.uncond : SDCondition(),
                                            embeds.img_uncond,
                                            sd::Tensor<float>(),
                                            0.f,
                                            sd_vid_gen_params->sample_params.guidance,
                                            hires_eta,
                                            sd_vid_gen_params->sample_params.shifted_timestep,
                                            hires_sample_method,
                                            sd_ctx->sd->is_flow_denoiser(),
                                            plan.extra_sample_args,
                                            hires_sigma_sched,
                                            std::vector<sd::Tensor<float>>{},
                                            false,
                                            hires_denoise_mask,
                                            sd::Tensor<float>(),
                                            hires_request.vace_strength,
                                            latents.audio_length,
                                            static_cast<float>(hires_request.fps),
                                            hires_request.cache_params,
                                            hires_video_positions,
                                            latents.audio_fixed,
                                            hires_video_reference);
        sampling_end   = ggml_time_ms();
        if (sd_ctx->sd->diffusion_model) {
            sd_ctx->sd->diffusion_model->set_refine_resident_scope(false);
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
            LOG_ERROR("sampling(latent upscale) failed after %.2fs",
                      (sampling_end - sampling_start) * 1.0f / 1000);
            return false;
        }
        LOG_INFO("sampling(latent upscale) completed, taking %.2fs",
                 (sampling_end - sampling_start) * 1.0f / 1000);
    } else if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->keep_diffusion_model_resident) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
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
    }

    sd_audio_t* generated_audio = nullptr;
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        latents.audio_length > 0 &&
        sd_ctx->sd->audio_vae_model != nullptr) {
        int64_t audio_latent_decode_start = ggml_time_ms();

        // FIX 3: bring the audio VAE back if it was freed for the stage-2 forward (no-op if resident).
        sd_ctx->sd->reload_audio_vae_model();

        auto audio_latent = unpack_ltxav_audio_latent(final_latent,
                                                      latents.audio_length,
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
        int64_t audio_latent_decode_end = ggml_time_ms();
        LOG_INFO("decoding audio latent completed, taking %.2fs", (audio_latent_decode_end - audio_latent_decode_start) * 1.0f / 1000);
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

    // Continuation chaining: hand the post-sampling diffusion latent back to the
    // caller (before VAE decode) so the tail can condition the next segment without
    // a lossy decode/re-encode roundtrip. With hires on, hand back the BASE pre-upscale
    // latent (chain_base_latent) so the next segment chains at base resolution.
    const sd::Tensor<float>& cont_latent_src = (!chain_base_latent.empty()) ? chain_base_latent : final_latent;
    if (final_latent_out != nullptr && !cont_latent_src.empty()) {
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
    }

    // FIX 3 (LTXAV two-stage) / WAN_VAE_FREE_DURING_DIT (CLI lever): bring the video VAE back if it
    // was freed to make DiT VRAM headroom. No-op (returns true) if still resident; the merged
    // reload_first_stage_model() dispatches to whichever reload loader was captured. On a genuine
    // reload FAILURE it has alloc'd but not refilled the params buffer, so abort here rather than
    // decode garbage (the pre-merge WAN-lever path checked this; the LTX path did not).
    if (!sd_ctx->sd->reload_first_stage_model()) {
        LOG_ERROR("video VAE reload before decode failed");
        free_sd_audio(generated_audio);
        return false;
    }
    auto result = decode_video_outputs(sd_ctx, latent_upscale_enabled ? hires_request : request, final_latent, num_frames_out);
    if (result == nullptr) {
        free_sd_audio(generated_audio);
        return false;
    }

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
                           sd_audio_t** audio_out) {
    return generate_video_ex(sd_ctx, sd_vid_gen_params, frames_out, num_frames_out, audio_out,
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

// Stitches per-segment audio onto one continuous timeline (planar, channel-major), dropping
// each seg>0's overlap head so it stays aligned with the video stitch. Replaces the old
// "keep only seg0 audio" behaviour, which left multi-segment clips silent after seg0 (and
// dropped audio entirely on resume).
struct ChainAudioAcc {
    uint32_t           sample_rate = 0;
    uint32_t           channels    = 0;
    std::vector<float> frames;  // INTERLEAVED: [f0c0, f0c1, f1c0, f1c1, ...]

    // sd_audio_t.data is interleaved (channel-minor) — same layout the single-segment
    // path feeds the opus/pcm writers, which read interleaved. The old per-channel
    // (planar) accumulate/rebuild scrambled the stereo pair across segments, which is
    // why multi-segment renders sounded broken while single-segment was clean.
    void append(const sd_audio_t* a, int drop_head_samples) {
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
        int sc    = (int)a->sample_count;  // frames
        int start = std::min(std::max(0, drop_head_samples), sc);
        // Drop `start` whole frames off the head, then append the rest interleaved.
        frames.insert(frames.end(),
                      a->data + (size_t)start * channels,
                      a->data + (size_t)sc * channels);
    }
    // Audio samples (frames) to drop for a seg>0 to match the dropped overlap_px video frames.
    int drop_for(const sd_audio_t* a, int overlap_px, int fps) const {
        if (a == nullptr || fps <= 0) {
            return 0;
        }
        return (int)llround((double)overlap_px * a->sample_rate / fps);
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
// validated expo_match.py. Default-on for the keyframe path; LTXAV_NO_EXPOSURE_MATCH=1 disables.
static void ltxav_exposure_match(const sd_image_t* prev_tail, int n_prev,
                                 sd_image_t* new_frames, int n_new) {
    if (getenv("LTXAV_NO_EXPOSURE_MATCH") != nullptr) return;
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
        sd_ctx_precompute_chain_text_conds(
            sd_ctx, cptrs.data(), (int)cptrs.size(),
            base_params->negative_prompt != nullptr ? base_params->negative_prompt : "",
            base_params->clip_skip);
    }

    // Prior segment's captured video-channel-only latent tail, ggml-ne order
    // [Wl, Hl, K, Cv] contiguous (W fastest, channel slowest), fed as cont_latent.
    std::vector<float>      cont_buf;
    int                     cont_Wl = 0, cont_Hl = 0, cont_Cv = 0;  // cont_buf spatial dims + channel count
    std::vector<sd_image_t> stitched;   // adopts each kept frame's .data
    ChainAudioAcc           audio_acc;  // per-segment audio stitched onto one timeline

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
    // continues the motion. On any failure we fall back to a fresh full render.
    int start_seg = 0;
    if (chain_params->resume_from > 0 && chain_params->save_dir != nullptr &&
        chain_params->save_dir[0] != '\0') {
        int               resume_k = std::min(chain_params->resume_from, n_chain - 1);
        const std::string sd_dir   = chain_params->save_dir;
        LOG_INFO("generate_video_chain: RESUME from segment %d/%d (reloading banked prefix from %s)",
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
            sd_audio_t* pa = read_seg_audio(sd_dir + "/seg_" + std::to_string(seg) + ".audio");
            if (pa != nullptr) {
                audio_acc.append(pa, (seg == 0) ? 0 : audio_acc.drop_for(pa, overlap_px, base_params->fps));
                free_sd_audio(pa);
            }
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

    for (int seg = start_seg; seg < n_chain; ++seg) {
        sd_vid_gen_params_t vp = *base_params;  // per-segment copy of the template

        if (seg == 0) {
            vp.cont_latent        = nullptr;
            vp.cont_latent_frames = 0;
            vp.audio_frame_offset = 0;
        } else {
            // Clear the init image for seg>0: prepare_video_generation_latents checks the
            // start image BEFORE the cont-latent branch, so a lingering init image would
            // re-render i2v from the same portrait and ignore the continuation.
            vp.init_image.data    = nullptr;
            vp.cont_latent        = cont_buf.data();
            vp.cont_latent_frames = K;
            vp.audio_frame_offset = seg * (base_params->video_frames - overlap_px);
        }
        // distinct seed per segment so the noise frames differ
        vp.seed = (base_params->seed < 0) ? base_params->seed : base_params->seed + seg;

        // Per-segment prompt (director): this segment's line, or the base prompt.
        if (chain_params->segment_prompts != nullptr && chain_params->segment_prompts[seg] != nullptr) {
            vp.prompt = chain_params->segment_prompts[seg];
            LOG_INFO("generate_video_chain seg %d prompt: %s", seg, vp.prompt);
        }

        // Per-segment lip-sync audio: chain_audio_dir/aud_<seg>.wav (16kHz mono). The
        // string must outlive the generate call, so keep it loop-scoped.
        std::string seg_audio_path;
        if (chain_params->chain_audio_dir != nullptr && chain_params->chain_audio_dir[0] != '\0') {
            seg_audio_path     = std::string(chain_params->chain_audio_dir) + "/aud_" + std::to_string(seg) + ".wav";
            vp.drive_audio_path = seg_audio_path.c_str();
            LOG_INFO("generate_video_chain seg %d drive-audio: %s", seg, seg_audio_path.c_str());
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
        bool        want_latent  = (seg + 1 < n_chain);
        if (!generate_video_ex(sd_ctx, &vp, &seg_video, &seg_count, &seg_audio,
                               want_latent ? &lat_out : nullptr,
                               want_latent ? &lw : nullptr, want_latent ? &lh : nullptr,
                               want_latent ? &lt : nullptr, want_latent ? &lc : nullptr)) {
            LOG_ERROR("generate_video_chain segment %d failed", seg + 1);
            free_sd_audio(seg_audio);
            free(seg_video);
            free(lat_out);
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
        if (want_latent && getenv("LONGCAT_CONT_REENCODE") != nullptr && seg_video != nullptr && seg_count > 0) {
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
        free(lat_out);

        // FEATURE A: capture SEG-0's per-channel latent stats as the anti-drift reference, and
        // (seg>0) remap this segment's freshly-captured tail onto them BEFORE it conditions the
        // next segment at the top of the following iteration. Runs only when the tail was actually
        // (re)filled this segment (want_latent) and the dims are known.
        if (want_latent && !cont_buf.empty() && cont_Wl > 0 && cont_Hl > 0 && cont_Cv > 0 && K > 0) {
            if (seg == 0 && !ref_ready) {
                ltxav_compute_latent_channel_stats(cont_buf.data(), cont_Wl, cont_Hl, K, cont_Cv, ref_mean, ref_std);
                ref_ready = true;
                if (latent_match_on) {
                    LOG_INFO("generate_video_chain: captured seg-0 latent channel reference stats (%d channels) for continuity-match",
                             cont_Cv);
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
        if (seg == 0) {
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
            if (seg > 0) {
                drop = atoi(e);
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
        if (seg > 0 && !stitched.empty() && seg_count - drop > 0) {
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

        // Audio: bank this segment's audio (for resume) + stitch it onto the timeline,
        // dropping the overlap head on seg>0 to stay aligned with the video stitch.
        if (chain_params->save_dir != nullptr && chain_params->save_dir[0] != '\0' && seg_audio != nullptr) {
            write_seg_audio(std::string(chain_params->save_dir) + "/seg_" + std::to_string(seg) + ".audio", seg_audio);
        }
        if (seg_audio != nullptr) {
            audio_acc.append(seg_audio, (seg == 0) ? 0 : audio_acc.drop_for(seg_audio, drop, base_params->fps));
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
        if (!generate_video_ex(sd_ctx, &vp, &seg_video, &seg_count, &seg_audio,
                               want_latent ? &lat_out : nullptr,
                               want_latent ? &lw : nullptr, want_latent ? &lh : nullptr,
                               want_latent ? &lt : nullptr, want_latent ? &lc : nullptr) ||
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
                                               int          clip_skip) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || prompts == nullptr || n_prompts <= 0) {
        return;
    }
    std::vector<std::string> ps;
    ps.reserve(n_prompts);
    for (int i = 0; i < n_prompts; ++i) {
        ps.emplace_back(prompts[i] != nullptr ? prompts[i] : "");
    }
    sd_ctx->sd->precompute_chain_text_conds(ps, negative_prompt != nullptr ? negative_prompt : "", clip_skip);
}

SD_API bool sd_ctx_swap_diffusion_model(sd_ctx_t* sd_ctx, const char* diffusion_model_path) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || diffusion_model_path == nullptr) {
        return false;
    }
    // Hot-swap the DiT weights in place (e.g. FLUX.2-Klein base<->edit). The VAE +
    // text encoder stay resident; only the diffusion model's param buffer is freed
    // and refilled from the new gguf. Caller MUST ensure no render is in flight.
    return sd_ctx->sd->swap_diffusion_model(diffusion_model_path);
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
