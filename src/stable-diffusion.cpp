#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/ggml_graph_cut.h"
#include "core/layer_split_partition.h"

#include "core/rng.hpp"
#include "core/rng_mt19937.hpp"
#include "core/rng_philox.hpp"
#include "core/tensor_ggml.hpp"
#include "core/util.h"
#include "gguf.h"
#include "model_loader.h"
#include "model_manager.h"
#include "stable-diffusion.h"

#include "conditioning/conditioner.hpp"
#include "core/backend_fit.h"
#include "extensions/generation_extension.h"
#include "model/adapter/lora.hpp"
#include "model/diffusion/anima.hpp"
#include "model/diffusion/animatediff.hpp"
#include "model/diffusion/boogu.hpp"
#include "model/diffusion/control.hpp"
#include "model/diffusion/ernie_image.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/hidream_o1.hpp"
#include "model/diffusion/hunyuan.hpp"
#include "model/diffusion/ideogram4.hpp"
#include "model/diffusion/krea2.hpp"
#include "model/diffusion/lens.hpp"
#include "model/diffusion/lingbot_video.hpp"
#include "model/diffusion/ltxv.hpp"
#include "longcat_audio.hpp"
#include "longcat_avatar.hpp"
#include "model/diffusion/mage_flow.hpp"
#include "model/diffusion/minit2i.hpp"
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
#include "model/vae/hunyuan_vae.hpp"
#include "model/vae/ltx_audio_vae.hpp"
#include "model/vae/ltx_vae.hpp"
#include "model/vae/mage_vae.hpp"
#include "model/vae/tae.hpp"
#include "model/vae/vae.hpp"
#include "model/vae/wan_vae.hpp"
#include "runtime/denoiser.hpp"
#include "runtime/guidance.h"
#include "runtime/sample-cache.h"
#include "upscaler.h"

#include "name_conversion.h"
#include "runtime/latent-preview.h"

#include <atomic>

const char* sd_vae_format_name(enum sd_vae_format_t format);
static SDVersion sd_vae_format_to_version(enum sd_vae_format_t format, SDVersion fallback);

static bool sd_version_supports_animatediff(SDVersion version) {
    return version == VERSION_SD1 || version == VERSION_SD1_INPAINT || version == VERSION_SD1_PIX2PIX;
}

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
    "LingBot Video",
    "Qwen Image",
    "Qwen Image Layered",
    "Hunyuan Video",
    "Anima",
    "Flux.2",
    "Flux.2 klein",
    "LTXAV",
    "HiDream O1",
    "Z-Image",
    "Boogu Image",
    "Ovis Image",
    "Ernie Image",
    "Lens",
    "MiniT2I",
    "Longcat-Image",
    "Longcat-Video-Avatar",
    "PiD",
    "Ideogram 4",
    "SeFi-Image",
    "Krea2",
    "Mage Flow",
    "ESRGAN",
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
           sd_version_is_mage_flow(version) ||
           sd_version_is_longcat(version) ||
           sd_version_is_z_image(version) ||
           sd_version_is_boogu_image(version);
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

// Read the per-tensor NVFP4 weight globals (ModelOpt weight_scale_2) out of an UNFOLDED
// import.  They are stored as tiny sibling F32 tensors named "<weight>.wglobal".
// Deliberately reads the file directly (gguf metadata with no_alloc=true, then a 4-byte
// read per sidecar by file offset) rather than going through the shared ModelLoader:
// this must be unambiguous about WHICH file it is describing, and the loader accumulates
// every model file ever init'd -- including the DiT that a hot-swap replaced.  `out` is
// keyed by the BARE gguf name "<weight>.wglobal"; a legacy folded gguf has none, so an
// empty map is the normal, correct result for those.
static void load_nvfp4_weight_globals(const std::string& path, std::map<std::string, float>& out) {
    struct gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* gctx         = gguf_init_from_file(path.c_str(), gp);
    if (gctx == nullptr) {
        return;
    }
    const size_t data_off = gguf_get_data_offset(gctx);
    FILE* f               = fopen(path.c_str(), "rb");
    if (f != nullptr) {
        const int64_t n = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < n; ++i) {
            const char* name = gguf_get_tensor_name(gctx, i);
            const size_t len = name ? strlen(name) : 0;
            if (len < 8 || strcmp(name + len - 8, ".wglobal") != 0) {
                continue;
            }
            // Only the exact shape this fold understands: one f32 scalar.  Anything else
            // is left unregistered, so its Linear keeps the graph-level multiply.
            if (gguf_get_tensor_type(gctx, i) != GGML_TYPE_F32 || gguf_get_tensor_size(gctx, i) != sizeof(float)) {
                continue;
            }
            const size_t off = data_off + gguf_get_tensor_offset(gctx, i);
            float g          = 1.0f;
            if (fseek(f, (long)off, SEEK_SET) == 0 && fread(&g, sizeof(float), 1, f) == 1) {
                out[name] = g;
            }
        }
        fclose(f);
    }
    gguf_free(gctx);
}

/*=============================================== StableDiffusionGGML ================================================*/

template <typename T, typename = void>
struct has_set_runtime_backends : std::false_type {};
template <typename T>
struct has_set_runtime_backends<T,
                                std::void_t<decltype(std::declval<T&>().set_runtime_backends(
                                    std::declval<const std::vector<ggml_backend_t>&>()))>> : std::true_type {};

static_assert(std::atomic<sd_cancel_mode_t>::is_always_lock_free,
              "sd_cancel_mode_t must be lock-free");

class StableDiffusionGGML {
public:
    SDBackendManager backend_manager;

    SDVersion version;
    bool external_vae_is_invalid = false;

    bool circular_x = false;
    bool circular_y = false;

    std::shared_ptr<RNG> rng         = std::make_shared<PhiloxRNG>();
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float default_flow_shift         = INFINITY;

    // {model file, runtime tensor-name prefix} of every leg that boot scanned for NVFP4
    // .wglobal sidecars. Kept so a DiT hot-swap, which rebuilds the whole (process-global)
    // registry, can re-register the legs it is NOT replacing -- e.g. a MoE high-noise
    // expert loaded under its own prefix.
    std::vector<std::pair<std::string, std::string>> nvfp4_weight_global_legs;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd or wan2.1 i2v
    std::shared_ptr<DiffusionModelRunner> diffusion_model;
    std::shared_ptr<DiffusionModelRunner> high_noise_diffusion_model;
    std::shared_ptr<VAE> first_stage_model;
    std::shared_ptr<VAE> preview_vae;
    std::shared_ptr<LTXV::LTXAudioVAERunner> audio_vae_model;
    std::shared_ptr<LONGCAT_AUDIO::WhisperEncoderRunner> whisper_encoder_model;
    std::shared_ptr<ControlNet> control_net;
    std::vector<std::shared_ptr<GenerationExtension>> generation_extensions;
    std::vector<std::shared_ptr<LoraModel>> runtime_lora_models;
    bool apply_lora_immediately = false;

    // ── caches that must not outlive the weights they were built against ─────
    //
    // Bumped by anything that changes what is registered with `model_manager`
    // while the sd_ctx lives: a DiT hot-swap (which REUSES the same runner
    // object, so a runner pointer cannot detect it) and control-net load/unload
    // (which changes `model_manager->tensor_names()`, the input to
    // LoraModel::preprocess_lora_tensors). Every cache below joins it into its
    // key, so a stale entry becomes unreachable rather than silently applied
    // against different weights.
    uint64_t base_model_epoch = 0;

    // Canonical description of the LoRA set resolved for the current request:
    // path, file identity, multiplier, high-noise flag and prefix filter of each
    // spec, in order, plus the apply mode. Empty string == no adapter. Recomputed
    // in apply_loras(); read by the runtime-adapter cache and handed to the
    // conditioner so the VLM embed cache can key on it.
    std::string current_lora_signature;

    // One request's worth of runtime LoRA state, kept so an unchanged `lora`
    // array does not re-parse and re-register the adapter gguf on every render
    // (measured: apply_loras 0.42-0.55 s per img_gen, CPU, outside sampling).
    //
    // The models hold ResidencyMode::ParamBackend registrations in their OWN
    // private ModelManager (~926 MiB for the r256 Q8_0 adapter).
    //
    // MEASURED: caching this costs ZERO extra idle VRAM -- 2938 vs 2940 MiB over
    // three renders, i.e. noise. An earlier draft of this comment claimed the
    // binding "pins" that memory between renders and was WRONG: it does, but so
    // does the code without it. apply_loras_at_runtime frees the previous adapter
    // at the START of the next request, not at the end of the render, so the
    // adapter is already resident between renders in both arms. The cache changes
    // nothing about idle residency.
    //
    // Still a single binding rather than an LRU: a second adapter set WOULD cost
    // its full resident size, for a hit rate the edit workflow does not need.
    // Releasing drops the last shared_ptr, and ~ModelManager -> release_all()
    // returns the memory.
    struct RuntimeLoraBinding {
        std::string key;
        std::vector<std::shared_ptr<LoraModel>> models;
        std::shared_ptr<MultiLoraAdapter> cond_stage_adapter;
        std::shared_ptr<MultiLoraAdapter> diffusion_adapter;
        std::shared_ptr<MultiLoraAdapter> first_stage_adapter;
    };
    std::optional<RuntimeLoraBinding> runtime_lora_binding;

    // Reference-image VAE latents, content-addressed. See
    // encode_reference_latent_cached().
    sd_cache::LruCache<sd::Tensor<float>> reference_latent_cache{
        sd_cache::entries_from_env("SD_REF_LATENT_CACHE_ENTRIES", 4, 64)};

    bool animatediff_loaded     = false;
    int animatediff_num_frames  = 0;

    std::string taesd_path;
    sd_tiling_params_t vae_tiling_params = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
    bool enable_mmap                     = false;
    sd::ggml_graph_cut::MaxVramAssignment max_vram_assignment;
    bool stream_layers = false;
    bool eager_load    = false;
    std::string backend_spec;
    std::string params_backend_spec;
    std::string split_mode_spec;
    bool auto_fit_enabled = false;

    bool diffusion_conv_direct = false;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    size_t control_net_params_mem_size = 0;

    std::shared_ptr<ModelManager> model_manager;

    // LongCat Avatar chains reuse the same image, audio and prompt for each
    // continuation window.  Text conditioning is independent of the window,
    // but umT5 is costly to run, so retain its host-owned result only for the
    // lifetime of one chain.  Do not share this with LTX: its per-window
    // reference-image conditioning is part of SDCondition.
    struct AvatarChainTextCond {
        SDCondition cond;
        SDCondition uncond;
        bool has_uncond = false;
    };
    bool avatar_chain_text_cache_active = false;
    std::map<std::string, AvatarChainTextCond> avatar_chain_text_cache;

    static std::string avatar_chain_text_key(const std::string& prompt,
                                             const std::string& negative_prompt) {
        return prompt + std::string(1, '\x1f') + negative_prompt;
    }

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();
    std::vector<float> file_alphas_cumprod;

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

    std::atomic<sd_cancel_mode_t> cancellation_flag = SD_CANCEL_RESET;

    void set_cancel_flag(enum sd_cancel_mode_t flag) {
        cancellation_flag.store(flag, std::memory_order_release);
    }

    void reset_cancel_flag() {
        set_cancel_flag(SD_CANCEL_RESET);
    }

    enum sd_cancel_mode_t get_cancel_flag() {
        return cancellation_flag.load(std::memory_order_acquire);
    }

    size_t max_graph_vram_bytes_for_module(SDBackendModule module) {
        return max_vram_assignment.bytes_for_backend(backend_for(module));
    }

    std::vector<size_t> layer_split_vram_limits_for_backends(const std::vector<ggml_backend_t>& backends) {
        std::vector<size_t> limits;
        limits.reserve(backends.size());
        for (ggml_backend_t backend : backends) {
            limits.push_back(max_vram_assignment.bytes_for_backend(backend));
        }
        return limits;
    }

    bool ensure_backend_pair(SDBackendModule module) {
        if (backend_for(module) == nullptr) {
            return false;
        }
        return params_backend_for(module) != nullptr;
    }

    // Keep CPU/mmap parameter residency warm across LTX continuation windows,
    // but return every transient GPU allocation before the next window plans
    // its graph-cut cache. This is an upstream weight-manager boundary, not a
    // return to the fork's resident-weight implementation.
    void reclaim_ltx_chain_window_gpu_memory() {
        // LTXAV_CHAIN_RECLAIM=0 disables this boundary reclaim. Default 1 = current behaviour.
        //
        // Suspect in a reproducible mid-chain crash: segment 1 of a chained render dies on its
        // FIRST cuBLAS GEMM immediately after this runs —
        //     LTX chain boundary: reclaimed transient staged weights, CUDA VMM pages, ...
        //     resize input image from 1920x1088 to 960x544
        //     sampling using Euler A CFG++ method
        //     [ERROR] CUDA error: an internal operation failed   (cublasGemmEx, ggml-cuda.cu:110)
        // seen at 92.1 s and 94.7 s into two different renders (koblem gpu_logs 845, 847), i.e.
        // deterministic rather than a random eviction. The worker then dies, so the caller's next
        // poll gets the supervisor's "worker is unloaded" 410 and it LOOKS like the gate evicted it.
        //
        // Mechanism to test: ggml_backend_cuda_trim_memory() hands VMM pages back to the driver and
        // ggml_backend_cuda_release_cudnn_conv3d_weights() frees raw cudaMalloc'd buffers. If cuBLAS
        // still holds a workspace carved from that pool, pulling it out from under the handle fails
        // the next GEMM exactly this way. The fork gated the equivalent trim behind opt-in
        // LTXAV_CHAIN_POOL_TRIM and measured it peak-NEUTRAL; the rebuild made it unconditional.
        static const bool chain_reclaim_enabled = [] {
            const char* s = getenv("LTXAV_CHAIN_RECLAIM");
            return s == nullptr || s[0] != '0';
        }();
        if (!chain_reclaim_enabled) {
            LOG_INFO("LTX chain boundary: reclaim DISABLED (LTXAV_CHAIN_RECLAIM=0)");
            return;
        }
        auto finish_runner = [](auto& runner) {
            if (!runner) {
                return;
            }
            runner->runner_done();
            runner->free_cache_ctx_and_buffer();
        };
        finish_runner(diffusion_model);
        finish_runner(high_noise_diffusion_model);
        finish_runner(first_stage_model);
        finish_runner(audio_vae_model);
        // The cached runtime adapter set pins its own params (hundreds of MiB for
        // a rank-256 adapter) in its private ModelManager, and nothing above can
        // see it. A chain boundary exists to hand VRAM back, so hand this back
        // too; the next segment reloads it exactly as it does today.
        release_runtime_lora_binding();
        if (model_manager) {
            model_manager->reclaim_transient_compute_buffers();
        }

        std::set<ggml_backend_t> backends;
        for (SDBackendModule module : {SDBackendModule::DIFFUSION, SDBackendModule::VAE, SDBackendModule::TE}) {
            if (ggml_backend_t backend = backend_for(module); backend != nullptr && ggml_backend_is_cuda(backend)) {
                backends.insert(backend);
            }
        }
        for (ggml_backend_t backend : backends) {
            ggml_backend_synchronize(backend);
            ggml_backend_cuda_trim_memory(backend);
        }

        // LTX's cuDNN Conv3D path holds a reordered copy of each VAE weight, outside
        // the VMM pool, so a boundary that exists to hand VRAM back frees them here.
        // This is no longer a CORRECTNESS requirement: the cache is keyed by stable
        // identity (tensor name + buffer + shape + type + device), not by the temporary
        // staged address, so keeping it across a window would be safe -- it is kept as a
        // VRAM reclaim. Do not touch the CUDA runtime for a CPU-only LTX invocation.
        if (!backends.empty()) {
            ggml_backend_cuda_release_cudnn_conv3d_weights();
        }
        LOG_INFO("LTX chain boundary: reclaimed transient staged weights, CUDA VMM pages, and cuDNN Conv3D reorder buffers");
    }

    template <typename T>
    bool register_runner_params(const std::string& desc,
                                const std::shared_ptr<T>& model,
                                SDBackendModule module,
                                size_t* params_mem_size = nullptr) {
        if (model == nullptr) {
            return true;
        }
        std::map<std::string, ggml_tensor*> group_tensors;
        model->get_param_tensors(group_tensors);
        if (model_manager == nullptr) {
            return true;
        }
        ModelManager::ResidencyMode residency_mode =
            backend_manager.params_backend_is_disk(module) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend;

        std::vector<ggml_backend_t> module_backends = backend_manager.runtime_backends(module);
        if (module_backends.size() > 1) {
            if constexpr (has_set_runtime_backends<T>::value) {
                if (module == SDBackendModule::DIFFUSION || module == SDBackendModule::TE) {
                    if (backend_manager.split_mode(module) == SDSplitMode::ROW) {
                        return register_row_split_runner_params(desc,
                                                                model,
                                                                module,
                                                                module_backends,
                                                                std::move(group_tensors),
                                                                residency_mode,
                                                                params_mem_size);
                    }
                    return register_layer_split_runner_params(desc,
                                                              model,
                                                              module,
                                                              module_backends,
                                                              std::move(group_tensors),
                                                              residency_mode,
                                                              params_mem_size);
                }
            }
            LOG_WARN("%s module does not support multiple runtime backends; using %s",
                     sd_backend_module_name(module),
                     sd::layer_split_backend_device_display_name(module_backends[0]).c_str());
        }
        return model_manager->register_param_tensors(desc,
                                                     std::move(group_tensors),
                                                     residency_mode,
                                                     backend_for(module),
                                                     params_backend_for(module),
                                                     params_mem_size);
    }

    template <typename T>
    bool register_row_split_runner_params(const std::string& desc,
                                          const std::shared_ptr<T>& model,
                                          SDBackendModule module,
                                          const std::vector<ggml_backend_t>& module_backends,
                                          std::map<std::string, ggml_tensor*> group_tensors,
                                          ModelManager::ResidencyMode residency_mode,
                                          size_t* params_mem_size) {
        ggml_backend_t main_backend = module_backends[0];

        auto fall_back_to_layer_split = [&](const char* reason) {
            LOG_WARN("%s: row split unavailable (%s); falling back to layer split", desc.c_str(), reason);
            return register_layer_split_runner_params(desc,
                                                      model,
                                                      module,
                                                      module_backends,
                                                      std::move(group_tensors),
                                                      residency_mode,
                                                      params_mem_size);
        };

        ggml_backend_dev_t main_dev = ggml_backend_get_device(main_backend);
        ggml_backend_reg_t reg      = main_dev != nullptr ? ggml_backend_dev_backend_reg(main_dev) : nullptr;
        if (reg == nullptr) {
            return fall_back_to_layer_split("no backend registry");
        }
        const size_t reg_dev_count = ggml_backend_reg_dev_count(reg);
        std::vector<float> tensor_split(reg_dev_count, 0.0f);
        constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
        for (ggml_backend_t backend : module_backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            int reg_index          = -1;
            for (size_t i = 0; i < reg_dev_count; i++) {
                if (ggml_backend_reg_dev_get(reg, i) == dev) {
                    reg_index = (int)i;
                    break;
                }
            }
            if (reg_index < 0) {
                return fall_back_to_layer_split("devices span different backend registries");
            }
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            int64_t usable_bytes    = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes,
                                                     (int64_t)free_bytes / 8);
            tensor_split[reg_index] = usable_bytes > 0 ? (float)((double)usable_bytes / (1024.0 * 1024.0)) : 1.0f;
        }

        ggml_backend_buffer_type_t split_buft = backend_manager.split_buffer_type(main_backend, tensor_split);
        if (split_buft == nullptr) {
            return fall_back_to_layer_split("backend has no split buffer type");
        }
        model_manager->set_split_buffer_type(main_backend, split_buft);

        std::map<std::string, ggml_tensor*> split_tensors;
        if constexpr (std::is_base_of_v<Conditioner, T>) {
            model->get_layer_split_param_tensors(split_tensors);
        } else {
            split_tensors = group_tensors;
        }

        std::map<std::string, ggml_tensor*> row_split_map;
        std::map<std::string, ggml_tensor*> regular_map;
        size_t row_split_bytes = 0;
        for (const auto& kv : group_tensors) {
            if (split_tensors.count(kv.first) != 0 &&
                sd::layer_split_tensor_block_index(kv.first) >= 0 &&
                ModelManager::tensor_shape_supports_split_buffer(kv.second)) {
                row_split_map[kv.first] = kv.second;
                row_split_bytes += ggml_nbytes(kv.second);
            } else {
                regular_map[kv.first] = kv.second;
            }
        }
        if (row_split_map.empty()) {
            return fall_back_to_layer_split("no row-splittable transformer block weights found");
        }

        LOG_INFO("%s row split: %zu tensors (%.1f MB) split across %zu devices (main %s)",
                 desc.c_str(),
                 row_split_map.size(),
                 row_split_bytes / (1024.f * 1024.f),
                 module_backends.size(),
                 sd::layer_split_backend_device_display_name(main_backend).c_str());

        if (!model_manager->register_param_tensors(desc,
                                                   std::move(row_split_map),
                                                   residency_mode,
                                                   main_backend,
                                                   params_backend_for(module),
                                                   params_mem_size,
                                                   /*allow_split_buffer=*/true)) {
            return false;
        }
        return model_manager->register_param_tensors(desc,
                                                     std::move(regular_map),
                                                     residency_mode,
                                                     main_backend,
                                                     params_backend_for(module),
                                                     params_mem_size);
    }

    // Register graph-cut layer-split tensors on the primary backend first.
    // The first real graph assigns each param tensor to a runtime backend
    // before weights are loaded or staged.
    template <typename T>
    bool register_layer_split_runner_params(const std::string& desc,
                                            const std::shared_ptr<T>& model,
                                            SDBackendModule module,
                                            const std::vector<ggml_backend_t>& module_backends,
                                            std::map<std::string, ggml_tensor*> group_tensors,
                                            ModelManager::ResidencyMode residency_mode,
                                            size_t* params_mem_size) {
        bool has_cpu_device = false;
        for (ggml_backend_t backend : module_backends) {
            has_cpu_device = has_cpu_device || sd_backend_is_cpu(backend);
        }
        if (has_cpu_device) {
            // The scheduler reserves the CPU slot for its fallback backend, and
            // CPU weight participation is what --params-backend <module>=cpu is
            // for; a CPU device in a split list is almost certainly a mistake.
            LOG_WARN(
                "%s: layer split across a CPU device is not supported; using %s "
                "(use --params-backend %s=cpu to keep weights in RAM)",
                desc.c_str(),
                sd::layer_split_backend_device_display_name(module_backends[0]).c_str(),
                sd_backend_module_name(module));
            return model_manager->register_param_tensors(desc,
                                                         std::move(group_tensors),
                                                         residency_mode,
                                                         module_backends[0],
                                                         params_backend_for(module),
                                                         params_mem_size);
        }

        model->set_runtime_backends(module_backends);
        model->set_graph_cut_layer_split_backend_vram_limits(layer_split_vram_limits_for_backends(module_backends));
        model->set_graph_cut_layer_split_enabled(true);
        const bool params_follow_runtime = backend_manager.params_backend_follows_runtime(module) ||
                                           backend_manager.params_backend_is_disk(module);
        ggml_backend_t initial_params_backend = params_follow_runtime ? module_backends[0] : params_backend_for(module);
        if (initial_params_backend == nullptr) {
            return false;
        }

        LOG_INFO("%s graph-cut layer split: deferring %zu tensors across %zu runtime backends until first graph",
                 desc.c_str(),
                 group_tensors.size(),
                 module_backends.size());

        return model_manager->register_param_tensors(desc,
                                                     std::move(group_tensors),
                                                     residency_mode,
                                                     module_backends[0],
                                                     initial_params_backend,
                                                     params_mem_size,
                                                     false,
                                                     params_follow_runtime);
    }

    // Drop everything cached against the currently registered weights or against
    // `model_manager->tensor_names()`. Cheap; call it on any doubt.
    void invalidate_weight_dependent_caches() {
        base_model_epoch++;
        reference_latent_cache.clear();
        release_runtime_lora_binding();
    }

    bool unload_control_net() {
        if (control_net == nullptr) {
            return true;
        }
        invalidate_weight_dependent_caches();
        if (model_manager != nullptr) {
            if (!model_manager->unregister_param_tensors("ControlNet", &control_net_params_mem_size)) {
                return false;
            }
        }
        control_net.reset();
        control_net_params_mem_size = 0;
        return true;
    }

    // Rebuild the process-global NVFP4 weight-global registry from the given model files.
    // Each leg is {file path, runtime tensor-name prefix} exactly as it was handed to
    // ModelLoader::init_from_file, because the sidecars are stored under bare gguf names.
    //
    // Registering a scalar is what licenses Linear::forward to drop its full-size
    // ggml_mul: the FP4 cuBLASLt GEMM then folds the scalar into the matmul alpha.  The
    // failure mode is asymmetric -- a scalar that is registered but keyed wrong (or stale
    // from a previous model) is silently wrong output, while one that is simply absent
    // only costs the multiply we were trying to remove.  So this ALWAYS clears first, and
    // on any inconsistency it clears again and leaves every Linear on the graph path.
    void register_nvfp4_weight_globals(const std::vector<std::pair<std::string, std::string>>& legs,
                                       const char* context) {
        // Never additive: a hot-swapped variant must not inherit the outgoing model's
        // scalars (a FOLDED gguf has already folded its global into the block scales and
        // would be scaled a second time).
        ggml_cuda_nvfp4_clear_weight_globals();

        if (model_manager == nullptr) {
            return;
        }

        // Ground truth for "how many Linears would otherwise emit the graph multiply":
        // the currently registered param tensors, which is exactly the set the graph
        // builder bound a "weight.wglobal" param for.
        size_t expected_sidecars = 0;
        for (const std::string& name : model_manager->tensor_names()) {
            if (ends_with(name, ".wglobal")) {
                ++expected_sidecars;
            }
        }
        if (expected_sidecars == 0) {
            return;  // folded gguf (or no NVFP4 model at all): nothing to fold, nothing to log
        }

        // ggml_set_name truncates to GGML_MAX_NAME, and the CUDA backend only ever sees
        // that truncated name -- so key by tensor->name, and refuse to fold at all if two
        // distinct weights of the SAME file collide onto one key with different scalars.
        // Across legs, later wins: that mirrors ModelLoader, where a later init_from_file()
        // overrides an earlier file's tensor of the same name (e.g. --diffusion-model
        // replacing the DiT inside a full checkpoint).
        std::map<std::string, float> by_ggml_name;
        size_t n_found = 0, n_unmatched = 0, n_conflict = 0;
        for (const auto& leg : legs) {
            if (leg.first.empty()) {
                continue;
            }
            std::map<std::string, float> wglobals;
            load_nvfp4_weight_globals(leg.first, wglobals);
            std::map<std::string, float> leg_by_ggml_name;
            for (const auto& kv : wglobals) {
                ++n_found;
                // "<bare weight>.wglobal" -> the runtime name of the weight it belongs to
                const std::string weight_name = leg.second + kv.first.substr(0, kv.first.size() - 8);
                ggml_tensor* weight           = model_manager->find_tensor(weight_name);
                if (weight == nullptr) {
                    LOG_WARN("%s: NVFP4 weight global '%s' from '%s' matches no registered tensor '%s'; "
                             "that Linear keeps its graph-level scale",
                             context, kv.first.c_str(), leg.first.c_str(), weight_name.c_str());
                    ++n_unmatched;
                    continue;
                }
                if (!std::isfinite(kv.second) || kv.second <= 0.f) {
                    LOG_ERROR("%s: NVFP4 weight global for '%s' is not a usable scale (%g)",
                              context, weight_name.c_str(), kv.second);
                    ++n_conflict;
                    continue;
                }
                auto it = leg_by_ggml_name.find(weight->name);
                if (it != leg_by_ggml_name.end() && it->second != kv.second) {
                    LOG_ERROR("%s: NVFP4 weight globals of '%s' collide on ggml name '%s' (%g vs %g) -- "
                              "GGML_MAX_NAME truncation makes the fold ambiguous",
                              context, leg.first.c_str(), weight->name, it->second, kv.second);
                    ++n_conflict;
                    continue;
                }
                leg_by_ggml_name[weight->name] = kv.second;
            }
            for (const auto& kv : leg_by_ggml_name) {
                by_ggml_name[kv.first] = kv.second;
            }
        }

        if (n_conflict > 0) {
            LOG_ERROR("%s: %zu NVFP4 weight-global conflicts -- NOT folding any of them into the "
                      "GEMM alpha; every Linear keeps its explicit scale (slower, correct)",
                      context, n_conflict);
            ggml_cuda_nvfp4_clear_weight_globals();
            return;
        }
        if (by_ggml_name.empty()) {
            LOG_WARN("%s: model declares %zu NVFP4 .wglobal sidecars but none of them could be matched "
                     "back to a registered weight (%zu read from file, %zu unmatched); keeping the "
                     "graph-level scale for all of them",
                     context, expected_sidecars, n_found, n_unmatched);
            return;
        }

        for (const auto& kv : by_ggml_name) {
            ggml_cuda_nvfp4_register_weight_global(kv.first.c_str(), kv.second);
        }

        if (by_ggml_name.size() != expected_sidecars || n_unmatched > 0) {
            LOG_ERROR("%s: registered %zu of %zu NVFP4 weight globals (%zu read from the model files, "
                      "%zu unmatched); the remainder stay on the graph-level scale path",
                      context, by_ggml_name.size(), expected_sidecars, n_found, n_unmatched);
        }

        // Say plainly whether the fold is actually live: registration alone is not enough,
        // a cuBLASLt path (FP4 alpha fold, or the FP8 FFN weight promotion) also has to be
        // enabled and the device Blackwell-class.  NB this probes ONE representative name;
        // the FP8 gate is per-name (GGML_FP8_LAYERS), so with FP8 on and FP4 off the true
        // answer varies per Linear -- the per-Linear decision is made in
        // ggml_ext_nvfp4_weight_global_folded_in_gemm(), this is only a startup summary.
        const bool folded = ggml_cuda_nvfp4_weight_global_folded(backend_manager.runtime_backend(SDBackendModule::DIFFUSION),
                                                                 by_ggml_name.begin()->first.c_str());
        if (folded) {
            LOG_INFO("%s: folding %zu NVFP4 weight globals into the cuBLASLt GEMM "
                     "(FP4 alpha / FP8 weight promotion; per-Linear scale multiply elided)",
                     context, by_ggml_name.size());
        } else {
            LOG_WARN("%s: %zu NVFP4 weight globals registered but the cuBLASLt GEMM fold is not "
                     "active for the diffusion backend (GGML_NVFP4_CUBLASLT / GGML_FP8_FFN off, "
                     "non-CUDA, or pre-Blackwell); keeping the graph-level scale",
                     context, by_ggml_name.size());
        }
    }

    // Upstream's ModelManager owns all DiT parameter residency.  Re-register
    // the same runner graph against a replacement GGUF at a serial boundary,
    // which drops the outgoing CPU/VRAM blocks before the new loader source is
    // made available. This deliberately supports only architecture-compatible
    // variants (the existing tensor metadata validation enforces that).
    bool swap_diffusion_model(const std::string& path) {
        if (path.empty() || diffusion_model == nullptr || model_manager == nullptr) {
            LOG_ERROR("swap_diffusion_model: missing path, diffusion runner, or model manager");
            return false;
        }
        // Invalidate every weight-dependent cache BEFORE anything is touched, and
        // unconditionally: the swap reuses the SAME runner object, so nothing
        // downstream can tell one variant from another by identity, and a failed
        // swap can still leave the registration mutated. A runtime adapter cached
        // against the outgoing DiT would be applied to the incoming one.
        invalidate_weight_dependent_caches();

        ModelLoader compatibility_probe;
        if (!compatibility_probe.init_from_file(path, "model.diffusion_model.")) {
            LOG_ERROR("swap_diffusion_model: could not inspect '%s'", path.c_str());
            return false;
        }
        std::vector<std::string> required_weight_globals;
        for (const std::string& name : compatibility_probe.get_tensor_names()) {
            if (ends_with(name, ".wglobal")) {
                required_weight_globals.push_back(name);
            }
        }

        diffusion_model->runner_done();
        if (!model_manager->unregister_param_tensors("Diffusion model")) {
            LOG_ERROR("swap_diffusion_model: could not release outgoing diffusion tensors");
            return false;
        }

        ModelLoader& loader = model_manager->loader();
        if (!loader.init_from_file(path, "model.diffusion_model.")) {
            LOG_ERROR("swap_diffusion_model: could not load '%s'", path.c_str());
            return false;
        }
        if (!register_runner_params("Diffusion model", diffusion_model, SDBackendModule::DIFFUSION)) {
            LOG_ERROR("swap_diffusion_model: could not register '%s'", path.c_str());
            return false;
        }
        const std::set<std::string> registered_names = model_manager->tensor_names();
        for (const std::string& name : required_weight_globals) {
            if (registered_names.find(name) == registered_names.end()) {
                LOG_ERROR("swap_diffusion_model: '%s' has unsupported ModelOpt NVFP4 sidecar '%s'",
                          path.c_str(), name.c_str());
                model_manager->unregister_param_tensors("Diffusion model");
                return false;
            }
        }
        if (!model_manager->validate_registered_tensors()) {
            LOG_ERROR("swap_diffusion_model: '%s' is not architecture-compatible", path.c_str());
            model_manager->unregister_param_tensors("Diffusion model");
            return false;
        }
        // Re-point the NVFP4 weight-global registry at the INCOMING gguf.  This is a full
        // rebuild, not an update: the registry is process-global and every selectable DiT
        // variant comes through here, so an unfolded -> folded swap must not leave the
        // outgoing scalars registered (the folded weights would be scaled twice), and an
        // unfolded -> unfolded swap must not keep the outgoing model's values.
        //
        // The registry is keyed by tensor name and the swap reuses the same runner graph,
        // so the keys are identical across variants -- only the values change.
        //
        // Legs that the swap does NOT replace (a MoE high-noise expert, an uncond DiT) must
        // be re-registered from their own files, or the rebuild would silently drop them
        // back onto the slower graph-level scale path.  The boot legs that DO cover
        // "model.diffusion_model." (including a single-file checkpoint, prefix "") are
        // dropped: their copy of these weights is the one being replaced.
        std::vector<std::pair<std::string, std::string>> legs = {{path, "model.diffusion_model."}};
        for (const auto& leg : nvfp4_weight_global_legs) {
            if (!leg.first.empty() && !leg.second.empty() && leg.second != "model.diffusion_model.") {
                legs.push_back(leg);
            }
        }
        register_nvfp4_weight_globals(legs, "swap_diffusion_model");
        LOG_INFO("swap_diffusion_model: selected '%s'; weights will load lazily", path.c_str());
        return true;
    }

    bool load_control_net_from_file(const std::string& path) {
        if (path.empty()) {
            LOG_ERROR("sd_ctx_load_control_net: empty path");
            return false;
        }
        if (model_manager == nullptr) {
            LOG_ERROR("sd_ctx_load_control_net: model_manager not initialized");
            return false;
        }

        if (!unload_control_net()) {
            return false;
        }
        // unload_control_net() short-circuits when nothing was loaded, so bump
        // again here: this call registers new tensors either way, and
        // `model_manager->tensor_names()` feeds preprocess_lora_tensors().
        invalidate_weight_dependent_caches();

        ModelLoader& shared_loader = model_manager->loader();
        if (!shared_loader.init_from_file(path)) {
            LOG_ERROR("sd_ctx_load_control_net: failed to load '%s'", path.c_str());
            return false;
        }
        shared_loader.convert_tensors_name();

        if (!ensure_backend_pair(SDBackendModule::CONTROL_NET)) {
            LOG_ERROR("sd_ctx_load_control_net: control_net backend unavailable");
            return false;
        }

        control_net = std::make_shared<ControlNet>(backend_for(SDBackendModule::CONTROL_NET),
                                                   params_backend_for(SDBackendModule::CONTROL_NET),
                                                   shared_loader.get_tensor_storage_map(),
                                                   version,
                                                   "",
                                                   model_manager);
        if (diffusion_conv_direct) {
            LOG_INFO("Using Conv2d direct in the control net");
            control_net->set_conv2d_direct_enabled(true);
        }
        if (!register_runner_params("ControlNet",
                                    control_net,
                                    SDBackendModule::CONTROL_NET,
                                    &control_net_params_mem_size)) {
            LOG_ERROR("sd_ctx_load_control_net: register_runner_params failed");
            control_net.reset();
            control_net_params_mem_size = 0;
            return false;
        }
        if (!model_manager->validate_registered_tensors()) {
            LOG_ERROR("sd_ctx_load_control_net: registered tensors validation failed");
            unload_control_net();
            return false;
        }
        LOG_INFO("sd_ctx_load_control_net: loaded '%s' (%.2f MB)",
                 path.c_str(),
                 control_net_params_mem_size / 1024.0 / 1024.0);
        return true;
    }

    bool init_backend() {
        std::string error;
        if (!backend_manager.init(backend_spec.c_str(),
                                  params_backend_spec.c_str(),
                                  split_mode_spec.c_str(),
                                  &error)) {
            LOG_ERROR("backend config failed: %s", error.c_str());
            return false;
        }
        return ensure_backend_pair(SDBackendModule::DIFFUSION);
    }

    bool row_split_active() {
        for (SDBackendModule module : {SDBackendModule::DIFFUSION, SDBackendModule::TE}) {
            if (backend_manager.split_mode(module) == SDSplitMode::ROW &&
                backend_manager.runtime_backends(module).size() > 1) {
                return true;
            }
        }
        return false;
    }

    bool graph_cut_layer_split_active() {
        for (SDBackendModule module : {SDBackendModule::DIFFUSION, SDBackendModule::TE}) {
            if (backend_manager.split_mode(module) == SDSplitMode::LAYER &&
                backend_manager.runtime_backends(module).size() > 1) {
                return true;
            }
        }
        return false;
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

    void refresh_compvis_denoiser_sigmas() {
        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
        if (!comp_vis_denoiser) {
            return;
        }
        std::vector<float> alphas_cumprod(TIMESTEPS);
        if (file_alphas_cumprod.size() == TIMESTEPS) {
            alphas_cumprod = file_alphas_cumprod;
        } else {
            calculate_alphas_cumprod(alphas_cumprod.data());
        }
        for (int i = 0; i < TIMESTEPS; i++) {
            comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
            comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
        }
    }

    void load_alphas_cumprod(ModelLoader& model_loader) {
        file_alphas_cumprod.clear();

        std::vector<float> loaded_alphas;
        if (!model_loader.load_float_tensor("alphas_cumprod", loaded_alphas, n_threads, enable_mmap)) {
            return;
        }
        if (loaded_alphas.size() != TIMESTEPS) {
            LOG_WARN("ignore alphas_cumprod from model file: expected %d values, got %zu",
                     TIMESTEPS,
                     loaded_alphas.size());
            return;
        }
        for (float alpha : loaded_alphas) {
            if (!std::isfinite(alpha) || alpha <= 0.0f || alpha > 1.0f) {
                LOG_WARN("ignore invalid alphas_cumprod from model file");
                return;
            }
        }

        file_alphas_cumprod = std::move(loaded_alphas);
        LOG_DEBUG("loaded alphas_cumprod from model file");
    }

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads           = sd_ctx_params->n_threads;
        enable_mmap         = sd_ctx_params->enable_mmap;
        stream_layers       = sd_ctx_params->stream_layers;
        eager_load          = sd_ctx_params->eager_load;
        backend_spec        = SAFE_STR(sd_ctx_params->backend);
        params_backend_spec = SAFE_STR(sd_ctx_params->params_backend);
        split_mode_spec     = SAFE_STR(sd_ctx_params->split_mode);
        auto_fit_enabled    = sd_ctx_params->auto_fit;
        max_vram_assignment.reset(0.f);
        {
            std::string error;
            if (!max_vram_assignment.parse(SAFE_STR(sd_ctx_params->max_vram), &error)) {
                LOG_ERROR("%s", error.c_str());
                return false;
            }
        }

        std::string rpc_servers_spec = SAFE_STR(sd_ctx_params->rpc_servers);
        add_rpc_devices(rpc_servers_spec);

        bool use_tae         = false;
        bool use_audio_vae   = false;
        bool use_control_net = false;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        model_manager = std::make_shared<ModelManager>();
        model_manager->set_n_threads(n_threads);
        model_manager->set_enable_mmap(enable_mmap);
        ModelLoader& model_loader = model_manager->loader();

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

        if (strlen(SAFE_STR(sd_ctx_params->pulid_weights_path)) > 0) {
            LOG_INFO("loading PuLID weights from '%s'", sd_ctx_params->pulid_weights_path);
            if (!model_loader.init_from_file(sd_ctx_params->pulid_weights_path,
                                             "model.diffusion_model.")) {
                LOG_WARN("loading PuLID weights from '%s' failed", sd_ctx_params->pulid_weights_path);
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

        if (strlen(SAFE_STR(sd_ctx_params->motion_module_path)) > 0) {
            LOG_INFO("loading motion module (AnimateDiff) from '%s'", sd_ctx_params->motion_module_path);
            if (!model_loader.init_from_file(sd_ctx_params->motion_module_path,
                                             "model.diffusion_model.motion_module.")) {
                LOG_WARN("loading motion module from '%s' failed", sd_ctx_params->motion_module_path);
            } else {
                animatediff_loaded = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
            if (!model_loader.init_from_file(sd_ctx_params->control_net_path)) {
                LOG_ERROR("init control net model loader from file failed: '%s'", sd_ctx_params->control_net_path);
                return false;
            } else {
                use_control_net = true;
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
        ggml_type wtype               = sd_type_to_ggml_type(sd_ctx_params->wtype);
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }

        if (auto_fit_enabled) {
            if (!sd::backend_fit::derive_backend_specs(model_loader,
                                                       wtype,
                                                       max_vram_assignment,
                                                       backend_spec,
                                                       params_backend_spec)) {
                return false;
            }
        }

        if (!init_backend()) {
            return false;
        }
        {
            std::string error;
            if (!max_vram_assignment.canonicalize_backend_keys(&error)) {
                LOG_ERROR("%s", error.c_str());
                return false;
            }
        }
        if (stream_layers && !backend_manager.params_backend_is_cpu(SDBackendModule::DIFFUSION)) {
            LOG_WARN("--stream-layers has no effect unless diffusion params backend is cpu; ignoring");
            stream_layers = false;
        }
        if (eager_load && graph_cut_layer_split_active()) {
            LOG_WARN("--eager-load is not supported with graph-cut layer split; weights will be prepared lazily");
            eager_load = false;
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
            const bool params_offloaded      = params_backend_for(SDBackendModule::DIFFUSION) != backend_for(SDBackendModule::DIFFUSION);
            const bool streaming_constrained = stream_layers || params_offloaded;
            // SD_LORA_FOLD=1 selects the params-backend fold (model/adapter/lora_fold.hpp),
            // which is exactly the case these two exclusions were written for: it merges on
            // the HOST copy before staging, so offload is fine, and it writes NVFP4 block
            // bytes directly, so a quantised base is fine. Row split still is not -- those
            // tensors are sharded across backends and the fold only sees one shard.
            const char* fold_env  = std::getenv("SD_LORA_FOLD");
            const bool fold_loras = fold_env != nullptr && fold_env[0] != '\0' && fold_env[0] != '0';
            if (row_split_active()) {
                apply_lora_immediately = false;
            } else if (fold_loras) {
                apply_lora_immediately = true;
            } else if (have_quantized_weight || streaming_constrained) {
                apply_lora_immediately = false;
            } else {
                apply_lora_immediately = true;
            }
        } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
            if (row_split_active()) {
                LOG_WARN(
                    "row-split tensors do not support the immediately LoRA apply mode; "
                    "LoRAs will not be applied to them (use --lora-apply-mode at_runtime)");
            }
            apply_lora_immediately = true;
        } else {
            apply_lora_immediately = false;
        }

        bool needs_writable_mmap = enable_mmap && apply_lora_immediately;
        model_manager->set_writable_mmap(needs_writable_mmap);
        if (enable_mmap && apply_lora_immediately) {
            LOG_WARN("in mode 'immediately', LoRAs will cause extra memory usage with mmap");
        }
        model_loader.process_model_files(enable_mmap, needs_writable_mmap);
        load_alphas_cumprod(model_loader);

        diffusion_conv_direct = sd_ctx_params->diffusion_conv_direct;

        size_t text_encoder_params_mem_size = 0;
        size_t unet_params_mem_size         = 0;
        size_t vae_params_mem_size          = 0;
        control_net_params_mem_size         = 0;
        size_t extension_params_mem_size    = 0;

        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            tae_preview_only = false;
            use_tae          = true;
        }

        {
            if (!ensure_backend_pair(SDBackendModule::TE) ||
                !ensure_backend_pair(SDBackendModule::DIFFUSION)) {
                return false;
            }

            if (sd_version_is_sd3(version)) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map,
                                                                     model_manager);
                diffusion_model  = std::make_shared<MMDiTRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                tensor_storage_map,
                                                                "model.diffusion_model",
                                                                model_manager);
            } else if (sd_version_is_pid(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Pid::PiDRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model.net",
                                                                   model_manager);
            } else if (sd_version_is_ideogram4(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Ideogram4::Ideogram4Runner>(backend_for(SDBackendModule::DIFFUSION),
                                                                               tensor_storage_map,
                                                                               "model.diffusion_model",
                                                                               model_manager);
            } else if (sd_version_is_krea2(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 true,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Krea2::Krea2Runner>(backend_for(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.diffusion_model",
                                                                       model_manager);
            } else if (sd_version_is_flux(version)) {
                bool is_chroma = false;
                for (auto pair : tensor_storage_map) {
                    if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                        is_chroma = true;
                        break;
                    }
                }
                if (is_chroma) {
                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                        tensor_storage_map,
                                                                        false,
                                                                        1,
                                                                        false,
                                                                        model_manager,
                                                                        sd_ctx_params->model_args);
                } else if (version == VERSION_OVIS_IMAGE) {
                    cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     false,
                                                                     model_manager);
                } else {
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                          tensor_storage_map,
                                                                          model_manager);
                }
                diffusion_model = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     model_manager,
                                                                     sd_ctx_params->model_args);
            } else if (sd_version_is_flux2(version) || sd_version_is_sefi_image(version)) {
                bool is_chroma   = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     model_manager,
                                                                     sd_ctx_params->model_args);
            } else if (sd_version_is_ltxav(version)) {
                const bool ltxv_t5_caption =
                    tensor_storage_map.find("model.diffusion_model.caption_projection.linear_1.weight") != tensor_storage_map.end() &&
                    tensor_storage_map.find("text_embedding_projection.linear_1.weight") == tensor_storage_map.end() &&
                    tensor_storage_map.find("text_encoders.llm.model.embed_tokens.weight") == tensor_storage_map.end();
                if (ltxv_t5_caption) {
                    LOG_INFO("LTX-Video 0.9.x checkpoint detected; using T5-XXL text encoder");
                    auto ltx_t5 = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    /*use_mask=*/false,
                                                                    /*mask_pad=*/0,
                                                                    /*is_umt5=*/false,
                                                                    model_manager);
                    ltx_t5->trim_to_valid = true;
                    cond_stage_model      = ltx_t5;
                } else {
                    cond_stage_model = std::make_shared<LTXAVEmbedder>(backend_for(SDBackendModule::TE),
                                                                       tensor_storage_map,
                                                                       "text_encoders.llm",
                                                                       "text_embedding_projection",
                                                                       model_manager);
                }
                diffusion_model  = std::make_shared<LTXV::LTXAVRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                      tensor_storage_map,
                                                                      "model.diffusion_model",
                                                                      model_manager);
            } else if (sd_version_is_hunyuan_video(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Hunyuan::HunyuanVideoRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                tensor_storage_map,
                                                                                "model.diffusion_model",
                                                                                version,
                                                                                model_manager);
            } else if (sd_version_is_wan(version)) {
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true,
                                                                    model_manager);
                diffusion_model  = std::make_shared<WAN::WanRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model",
                                                                   version,
                                                                   model_manager);
                if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                    high_noise_diffusion_model = std::make_shared<WAN::WanRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                  tensor_storage_map,
                                                                                  "model.high_noise_diffusion_model",
                                                                                  version,
                                                                                  model_manager);
                }
                if (diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-FLF2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-I2V-1.3B") {
                    if (!ensure_backend_pair(SDBackendModule::CLIP_VISION)) {
                        return false;
                    }
                    clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend_for(SDBackendModule::CLIP_VISION),
                                                                             tensor_storage_map,
                                                                             model_manager);
                    clip_vision->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::CLIP_VISION));
                    if (!register_runner_params("CLIP vision",
                                                clip_vision,
                                                SDBackendModule::CLIP_VISION)) {
                        return false;
                    }
                }
            } else if (sd_version_is_lingbot_video(version)) {
                bool enable_vision = false;
                for (const auto& [name, _] : tensor_storage_map) {
                    if (starts_with(name, "text_encoders.llm.visual.")) {
                        enable_vision = true;
                        break;
                    }
                }
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision,
                                                                 model_manager);
                diffusion_model  = std::make_shared<LingBotVideo::LingBotVideoRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                     tensor_storage_map,
                                                                                     "model.diffusion_model",
                                                                                     model_manager,
                                                                                     sd_ctx_params->model_args);
            } else if (sd_version_is_qwen_image(version)) {
                bool enable_vision = version != VERSION_QWEN_IMAGE_LAYERED;
                cond_stage_model   = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision,
                                                                 model_manager);
                diffusion_model    = std::make_shared<Qwen::QwenImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                          tensor_storage_map,
                                                                          "model.diffusion_model",
                                                                          version,
                                                                          model_manager,
                                                                          sd_ctx_params->model_args);
            } else if (sd_version_is_mage_flow(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 true,
                                                                 model_manager);
                diffusion_model  = std::make_shared<MageFlow::MageFlowRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                             tensor_storage_map,
                                                                             "model.diffusion_model",
                                                                             model_manager);
            } else if (sd_version_is_longcat(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 true,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     model_manager,
                                                                     sd_ctx_params->model_args);
            } else if (sd_version_is_longcat_avatar(version)) {
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true,
                                                                    model_manager,
                                                                    sd_ctx_params->model_args);
                diffusion_model = std::make_shared<LongCatAvatarModel>(backend_for(SDBackendModule::DIFFUSION),
                                                                        tensor_storage_map,
                                                                        "model.diffusion_model",
                                                                        version,
                                                                        model_manager);
            } else if (version == VERSION_HIDREAM_O1) {
                cond_stage_model = std::make_shared<HiDreamO1::HiDreamO1Conditioner>(backend_for(SDBackendModule::TE),
                                                                                     tensor_storage_map,
                                                                                     model_manager);
                diffusion_model  = std::make_shared<HiDreamO1::HiDreamO1Runner>(backend_for(SDBackendModule::DIFFUSION),
                                                                               tensor_storage_map,
                                                                               "model",
                                                                               model_manager);
            } else if (sd_version_is_minit2i(version)) {
                cond_stage_model = std::make_shared<MiniT2IConditioner>(backend_for(SDBackendModule::TE),
                                                                        tensor_storage_map,
                                                                        model_manager);
                diffusion_model  = std::make_shared<MiniT2I::MiniT2IRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                           tensor_storage_map,
                                                                           "model.diffusion_model.model.net",
                                                                           model_manager);
            } else if (sd_version_is_anima(version)) {
                cond_stage_model = std::make_shared<AnimaConditioner>(backend_for(SDBackendModule::TE),
                                                                      tensor_storage_map,
                                                                      model_manager);
                diffusion_model  = std::make_shared<Anima::AnimaRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.diffusion_model",
                                                                       model_manager);
            } else if (sd_version_is_z_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false,
                                                                 model_manager);
                diffusion_model  = std::make_shared<ZImage::ZImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         version,
                                                                         model_manager);
            } else if (sd_version_is_boogu_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 true,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Boogu::BooguImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            version,
                                                                            model_manager);
            } else if (sd_version_is_ernie_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false,
                                                                 model_manager);
                diffusion_model  = std::make_shared<ErnieImage::ErnieImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                 tensor_storage_map,
                                                                                 "model.diffusion_model",
                                                                                 model_manager);
            } else if (sd_version_is_lens(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 false,
                                                                 model_manager);
                diffusion_model  = std::make_shared<Lens::LensRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     model_manager);
            } else {  // SD1.x SD2.x SDXL
                std::map<std::string, std::string> embbeding_map;
                for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                    embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
                }
                cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(backend_for(SDBackendModule::TE),
                                                                                       tensor_storage_map,
                                                                                       embbeding_map,
                                                                                       version,
                                                                                       model_manager);
                diffusion_model  = std::make_shared<UNetModelRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model",
                                                                    version,
                                                                    model_manager);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the diffusion model");
                    diffusion_model->set_conv2d_direct_enabled(true);
                }
            }

            cond_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::TE));
            if (!register_runner_params("Conditioner model",
                                        cond_stage_model,
                                        SDBackendModule::TE,
                                        &text_encoder_params_mem_size)) {
                return false;
            }

            diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::DIFFUSION));
            diffusion_model->set_stream_layers_enabled(stream_layers);
            if (!register_runner_params("Diffusion model",
                                        diffusion_model,
                                        SDBackendModule::DIFFUSION,
                                        &unet_params_mem_size)) {
                return false;
            }

            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::DIFFUSION));
                high_noise_diffusion_model->set_stream_layers_enabled(stream_layers);
                if (!register_runner_params("High noise diffusion model",
                                            high_noise_diffusion_model,
                                            SDBackendModule::DIFFUSION,
                                            &unet_params_mem_size)) {
                    return false;
                }
            }

            if (!ensure_backend_pair(SDBackendModule::VAE)) {
                return false;
            }

            auto create_tae = [&](bool decode_only) -> std::shared_ptr<VAE> {
                if (sd_version_uses_wan_vae(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(backend_for(SDBackendModule::VAE),
                                                                  tensor_storage_map,
                                                                  "decoder",
                                                                  decode_only,
                                                                  version,
                                                                  model_manager);

                } else {
                    auto model = std::make_shared<TinyImageAutoEncoder>(backend_for(SDBackendModule::VAE),
                                                                        tensor_storage_map,
                                                                        "decoder.layers",
                                                                        decode_only,
                                                                        version,
                                                                        model_manager);
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
                                                         tensor_storage_map,
                                                         "first_stage_model",
                                                         false,
                                                         version,
                                                         model_manager);
                } else if (sd_version_is_mage_flow(vae_version)) {
                    return std::make_shared<MageVAE::MageVAERunner>(backend_for(SDBackendModule::VAE),
                                                                    tensor_storage_map,
                                                                    "first_stage_model",
                                                                    model_manager);
                } else if (sd_version_uses_hunyuan_video_vae(vae_version)) {
                    return std::make_shared<Hunyuan::HunyuanVideoVAERunner>(backend_for(SDBackendModule::VAE),
                                                                            tensor_storage_map,
                                                                            "first_stage_model",
                                                                            false,
                                                                            vae_version,
                                                                            model_manager);
                } else if (sd_version_uses_wan_vae(vae_version)) {
                    return std::make_shared<WAN::WanVAERunner>(backend_for(SDBackendModule::VAE),
                                                               tensor_storage_map,
                                                               "first_stage_model",
                                                               false,
                                                               vae_version,
                                                               model_manager);
                } else {
                    auto model = std::make_shared<AutoEncoderKL>(backend_for(SDBackendModule::VAE),
                                                                 tensor_storage_map,
                                                                 "first_stage_model",
                                                                 false,
                                                                 false,
                                                                 vae_version,
                                                                 model_manager);
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

            if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1 || sd_version_is_minit2i(version)) {
                LOG_INFO("using FakeVAE");
                first_stage_model = std::make_shared<FakeVAE>(version,
                                                              backend_for(SDBackendModule::VAE),
                                                              model_manager);
                if (!register_runner_params("VAE",
                                            first_stage_model,
                                            SDBackendModule::VAE,
                                            &vae_params_mem_size)) {
                    return false;
                }
            } else if (use_tae && !tae_preview_only) {
                LOG_INFO("using TAE for encoding / decoding");
                first_stage_model = create_tae(false);
                first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::VAE));
                if (!register_runner_params("VAE",
                                            first_stage_model,
                                            SDBackendModule::VAE,
                                            &vae_params_mem_size)) {
                    return false;
                }
            } else {
                LOG_INFO("using VAE for encoding / decoding");
                first_stage_model = create_vae();
                first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::VAE));
                if (!register_runner_params("VAE",
                                            first_stage_model,
                                            SDBackendModule::VAE,
                                            &vae_params_mem_size)) {
                    return false;
                }
                if (use_tae && tae_preview_only) {
                    LOG_INFO("using TAE for preview");
                    preview_vae = create_tae(true);
                    preview_vae->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::VAE));
                    if (!register_runner_params("preview VAE",
                                                preview_vae,
                                                SDBackendModule::VAE,
                                                &vae_params_mem_size)) {
                        return false;
                    }
                }
            }

            if (use_audio_vae && sd_version_is_longcat_avatar(version)) {
                // For LongCat Avatar, audio_vae_path identifies its Whisper
                // encoder GGUF rather than an LTX audio VAE.
                use_audio_vae = false;
                whisper_encoder_model = std::make_shared<LONGCAT_AUDIO::WhisperEncoderRunner>(backend_for(SDBackendModule::TE),
                                                                                               tensor_storage_map,
                                                                                               "audio_encoder",
                                                                                               model_manager);
                whisper_encoder_model->set_flash_attention_enabled(false);
                if (!register_runner_params("LongCat Avatar Whisper encoder",
                                            whisper_encoder_model,
                                            SDBackendModule::TE,
                                            &text_encoder_params_mem_size)) {
                    return false;
                }
            } else if (use_audio_vae) {
                audio_vae_model = std::make_shared<LTXV::LTXAudioVAERunner>(backend_for(SDBackendModule::VAE),
                                                                            tensor_storage_map,
                                                                            "",
                                                                            model_manager);
                if (!register_runner_params("LTX audio VAE",
                                            audio_vae_model,
                                            SDBackendModule::VAE,
                                            &vae_params_mem_size)) {
                    return false;
                }
            }

            // GGML_CUDNN_CONV is the route prod uses to enable this: the flux2 entrypoint
            // passes no --vae-conv-direct, so the env gate was the ONLY way in, and the
            // rebuild dropping it left every VAE 2D conv on ggml_conv_2d -> IM2COL + MUL_MAT.
            // At 1920x1088 that im2col arena asks for a single ~9.9 GB compute buffer and
            // OOMs; conv-direct + the cuDNN interceptor (conv2d-cudnn.cu) is both smaller
            // and faster. Measured: 1024^2 11210 -> 8070 MiB, 1920x1088 OOM -> 10288 MiB.
            // Value-honouring: `GGML_CUDNN_CONV=0` must DISABLE this. The presence test that used
            // to be here meant an explicit 0 turned conv2d-direct ON, so anyone bisecting cuDNN
            // conv measured it enabled in both arms. Same class of bug as the two gates inside
            // ggml-cuda (conv2d-cudnn.cu / conv3d-cudnn.cu), fixed there too.
            const char* cudnn_conv2d_env = getenv("GGML_CUDNN_CONV");
            const bool cudnn_conv2d_on   = cudnn_conv2d_env != nullptr && cudnn_conv2d_env[0] != '\0' &&
                                         atoi(cudnn_conv2d_env) != 0;
            if (sd_ctx_params->vae_conv_direct || cudnn_conv2d_on) {
                LOG_INFO("Using Conv2d direct in the vae model");
                first_stage_model->set_conv2d_direct_enabled(true);
                if (preview_vae) {
                    preview_vae->set_conv2d_direct_enabled(true);
                }
            }

            if (use_control_net) {
                if (!ensure_backend_pair(SDBackendModule::CONTROL_NET)) {
                    return false;
                }
                control_net = std::make_shared<ControlNet>(backend_for(SDBackendModule::CONTROL_NET),
                                                           params_backend_for(SDBackendModule::CONTROL_NET),
                                                           model_loader.get_tensor_storage_map(),
                                                           version,
                                                           "",
                                                           model_manager);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the control net");
                    control_net->set_conv2d_direct_enabled(true);
                }
                if (!register_runner_params("ControlNet",
                                            control_net,
                                            SDBackendModule::CONTROL_NET,
                                            &control_net_params_mem_size)) {
                    return false;
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
                    model_manager,
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

                auto pulid_extension = create_pulid_extension();
                if (!pulid_extension->init(extension_ctx)) {
                    return false;
                }
                if (pulid_extension->is_enabled()) {
                    generation_extensions.push_back(pulid_extension);
                }
            }
            for (auto& extension : generation_extensions) {
                if (!register_runner_params(extension->name(),
                                            extension,
                                            SDBackendModule::PHOTOMAKER,
                                            &extension_params_mem_size)) {
                    return false;
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
        }

        LOG_DEBUG("validating model metadata");

        std::set<std::string> ignore_tensors;
        if (use_tae && !tae_preview_only) {
            ignore_tensors.insert("first_stage_model.");
        }
        for (auto& extension : generation_extensions) {
            extension->add_ignore_tensors(ignore_tensors);
        }
        ignore_tensors.insert("model.diffusion_model.__x0__");
        ignore_tensors.insert("model.diffusion_model.__32x32__");
        ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

        // NOTE: do NOT add "audio_vae.encoder" here. This set is handed to
        // ModelManager::set_common_ignore_tensors() below, and should_ignore() gates not just
        // metadata validation but LOADING (model_manager.cpp:337) and STAGING TO THE COMPUTE
        // BACKEND (model_manager.cpp:398) as well. The audio VAE encoder ships in its own GGUF
        // (LTX_AUDIO_VAE=..._audio_vae-ENC-f16.gguf), so ignoring it as "absent from the main
        // checkpoint" is right for validation and catastrophic for staging: its weights are then
        // never managed, and after the first chain window tears down its params-backend storage
        // they fall back to the raw mmap host address forever. The next window hands that HOST
        // pointer to cublasSgemm, which returns a bare "an internal operation failed" —
        // killing every multi-segment render WITH audio at window 2 while the window that
        // caused it completed perfectly. Named by GGML_CUDA_CHECK_MM_PTRS=1 as
        // 'audio_vae.encoder.mel_stft.forward_basis (reshaped)' at a stable host address.
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

        model_manager->set_common_ignore_tensors(ignore_tensors);
        size_t weight_global_count = 0;
        const std::set<std::string> registered_names = model_manager->tensor_names();
        for (const std::string& name : model_loader.get_tensor_names()) {
            if (!ends_with(name, ".wglobal")) {
                continue;
            }
            ++weight_global_count;
            if (registered_names.find(name) == registered_names.end()) {
                LOG_ERROR("unfolded NVFP4 GGUF sidecar '%s' is not attached to a supported Linear weight", name.c_str());
                return false;
            }
        }
        if (weight_global_count > 0) {
            // Hand the scalars to the CUDA FP4 GEMM so it can fold them into the matmul
            // alpha, instead of paying a full-size elementwise multiply per Linear.  Legs
            // mirror the init_from_file() calls above, since the sidecars carry bare gguf
            // names and it is the load prefix that makes them runtime tensor names.
            nvfp4_weight_global_legs = {
                {SAFE_STR(sd_ctx_params->model_path), ""},
                {SAFE_STR(sd_ctx_params->diffusion_model_path), "model.diffusion_model."},
                {SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path), "model.high_noise_diffusion_model."},
                {SAFE_STR(sd_ctx_params->uncond_diffusion_model_path), "model.diffusion_model.uncond."},
            };
            register_nvfp4_weight_globals(nvfp4_weight_global_legs, "nvfp4 weight globals");
        } else {
            // No unfolded sidecars in this model: make sure a previously created context in
            // this process cannot leave stale scalars behind in the global registry.
            ggml_cuda_nvfp4_clear_weight_globals();
        }
        if (!model_manager->validate_registered_tensors()) {
            LOG_ERROR("model metadata validation failed");
            return false;
        }

        if (eager_load) {
            if (!model_manager->load_all_params_eagerly()) {
                LOG_ERROR("model params eager load failed");
                return false;
            }
            LOG_DEBUG("model metadata validated; weights pre-loaded to params backend");
        } else {
            LOG_DEBUG("model metadata validated; weights will be prepared lazily");
        }

        {
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

            if (!add_params_memory(text_encoder_params_mem_size, SDBackendModule::TE) ||
                !add_params_memory(extension_params_mem_size, SDBackendModule::PHOTOMAKER) ||
                !add_params_memory(unet_params_mem_size, SDBackendModule::DIFFUSION) ||
                !add_params_memory(vae_params_mem_size, SDBackendModule::VAE) ||
                !add_params_memory(control_net_params_mem_size, SDBackendModule::CONTROL_NET)) {
                return false;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), extensions %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                text_encoder_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(text_encoder_params_mem_size, SDBackendModule::TE),
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
                    pred_type = is_using_v_parameterization_for_sd2(sd_version_is_inpaint(version)) ? V_PRED : EPS_PRED;
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
                           sd_version_is_hunyuan_video(version) ||
                           sd_version_is_lingbot_video(version) ||
                           sd_version_is_qwen_image(version) ||
                           sd_version_is_mage_flow(version) ||
                           version == VERSION_HIDREAM_O1 ||
                           sd_version_is_anima(version) ||
                           sd_version_is_ernie_image(version) ||
                           sd_version_is_longcat_avatar(version) ||
                           sd_version_is_z_image(version) ||
                           sd_version_is_boogu_image(version) ||
                           sd_version_is_pid(version) ||
                           sd_version_is_ideogram4(version)) {
                    pred_type = FLOW_PRED;
                    if (sd_version_is_wan(version)) {
                        default_flow_shift = 5.f;
                    } else if (sd_version_is_longcat_avatar(version)) {
                        // LongCat Avatar 1.5 is trained with FlowMatchEulerDiscrete
                        // timestep shifting, not epsilon prediction. Keep this distinct
                        // from the generic Wan 5.0 shift.
                        default_flow_shift = 7.f;
                    } else if (sd_version_is_hunyuan_video(version)) {
                        default_flow_shift = 7.f;
                    } else if (sd_version_is_ernie_image(version)) {
                        default_flow_shift = 4.f;
                    } else if (sd_version_is_pid(version)) {
                        default_flow_shift = 1.5f;
                    } else if (sd_version_is_ideogram4(version)) {
                        default_flow_shift = 1.0f;
                    } else if (sd_version_is_boogu_image(version)) {
                        default_flow_shift = 3.16f;
                    } else if (sd_version_is_mage_flow(version)) {
                        default_flow_shift = 6.f;
                    } else {
                        default_flow_shift = 3.f;
                    }
                } else if (sd_version_is_flux(version) ||
                           sd_version_is_flux2(version) ||
                           sd_version_is_longcat(version) ||
                           sd_version_is_lens(version) ||
                           sd_version_is_ltxav(version) ||
                           sd_version_is_krea2(version)) {
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
                    } else if (sd_version_is_krea2(version)) {
                        default_flow_shift = 1.15f;
                    }
                } else if (sd_version_is_sefi_image(version)) {
                    pred_type = SEFI_FLOW_PRED;
                } else if (sd_version_is_minit2i(version)) {
                    pred_type = MINIT2I_FLOW_PRED;
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
                case SEFI_FLOW_PRED: {
                    LOG_INFO("running in SeFi-Image dual-time FLOW mode");
                    denoiser = std::make_shared<SefiFlowDenoiser>();
                    break;
                }
                case MINIT2I_FLOW_PRED: {
                    LOG_INFO("running in MiniT2I FLOW mode");
                    denoiser = std::make_shared<MiniT2IFlowDenoiser>();
                    break;
                }
                default: {
                    LOG_ERROR("Unknown predition type %i", pred_type);
                    return false;
                }
            }

            refresh_compvis_denoiser_sigmas();
        }

        return true;
    }

    bool is_using_v_parameterization_for_sd2(bool is_inpaint = false) {
        struct RunnerDoneOnExit {
            GGMLRunner* runner = nullptr;
            ~RunnerDoneOnExit() {
                if (runner != nullptr) {
                    runner->runner_done();
                }
            }
        };
        RunnerDoneOnExit diffusion_runner_done{diffusion_model.get()};

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

        double result = static_cast<double>((out - x_t).mean());
        int64_t t1    = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    static std::string lora_log_id(const ModelManager::LoraSpec& lora) {
        return lora.is_high_noise ? "|high_noise|" + lora.path : lora.path;
    }

    std::shared_ptr<LoraModel> load_lora_model(const ModelManager::LoraSpec& lora_spec,
                                               SDBackendModule module,
                                               LoraModel::filter_t module_filter = nullptr) {
        if (!ensure_backend_pair(module)) {
            return nullptr;
        }
        if (lora_spec.is_high_noise) {
            LOG_DEBUG("high noise lora: %s", lora_spec.path.c_str());
        }
        auto lora                              = std::make_shared<LoraModel>(lora_log_id(lora_spec),
                                                backend_for(module),
                                                backend_for(module),
                                                lora_spec.path,
                                                lora_spec.is_high_noise ? "model.high_noise_" : "",
                                                version);
        LoraModel::filter_t lora_tensor_filter = module_filter;
        if (!lora_spec.tensor_name_prefix_filter.empty()) {
            lora_tensor_filter = [module_filter, prefix = lora_spec.tensor_name_prefix_filter](const std::string& tensor_name) {
                return starts_with(tensor_name, prefix) && (!module_filter || module_filter(tensor_name));
            };
        }
        if (!lora->load_from_file(n_threads, lora_tensor_filter)) {
            LOG_WARN("load lora tensors from %s failed", lora_spec.path.c_str());
            return nullptr;
        }

        lora->multiplier = lora_spec.multiplier;
        return lora;
    }

    void clear_lora_adapters() {
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
    }

    std::vector<std::shared_ptr<LoraModel>> load_runtime_loras_for_module(const std::vector<ModelManager::LoraSpec>& loras,
                                                                          const std::set<std::string>& model_tensor_names,
                                                                          SDBackendModule module,
                                                                          LoraModel::filter_t module_filter = nullptr) {
        std::vector<std::shared_ptr<LoraModel>> module_lora_models;
        for (const auto& lora_spec : loras) {
            auto lora = load_lora_model(lora_spec, module, module_filter);
            if (lora == nullptr) {
                if (lora_spec.required) {
                    LOG_ERROR("required lora load failed: %s", lora_spec.path.c_str());
                }
                continue;
            }
            if (lora->lora_tensors.empty()) {
                continue;
            }

            lora->preprocess_lora_tensors(model_tensor_names);
            runtime_lora_models.push_back(lora);
            module_lora_models.push_back(std::move(lora));
        }
        return module_lora_models;
    }

    void apply_loras_immediately(const std::vector<ModelManager::LoraSpec>& loras) {
        if (model_manager == nullptr) {
            if (!loras.empty()) {
                LOG_WARN("model manager is not available for immediate lora");
            }
            return;
        }

        clear_lora_adapters();
        runtime_lora_models.clear();
        // The immediate path folds into the base weights and never consults the
        // runtime binding, so a live binding here would be dead VRAM.
        release_runtime_lora_binding();

        model_manager->set_loras(loras, version);
    }

    // Drop the cached runtime adapter set and, with it, its VRAM. The models are
    // the last strong references to their private ModelManagers, whose destructor
    // calls release_all(); release_loaded_tensors() first makes that deterministic
    // rather than dependent on shared_ptr order, and matches how ModelManager
    // itself disposes of a fold LoRA (model_manager.cpp:567,1283).
    void release_runtime_lora_binding() {
        if (!runtime_lora_binding.has_value()) {
            return;
        }
        RuntimeLoraBinding binding = std::move(*runtime_lora_binding);
        runtime_lora_binding.reset();
        for (auto& model : binding.models) {
            if (model != nullptr) {
                model->release_loaded_tensors();
            }
        }
    }

    // Everything outside the LoRA specs themselves that decides what
    // apply_loras_at_runtime() builds. Joined into the binding key.
    std::string runtime_lora_environment_key() const {
        std::string key = "v" + std::to_string(static_cast<int>(version)) +
                          "-t" + std::to_string(n_threads) +
                          "-e" + std::to_string((unsigned long long)base_model_epoch);
        // Which module runners exist decides which of the three filtered loads
        // runs at all, and each load allocates on that module's backend pair.
        key += cond_stage_model ? "-te1" : "-te0";
        key += diffusion_model ? "-dit1" : "-dit0";
        key += high_noise_diffusion_model ? "-hn1" : "-hn0";
        key += first_stage_model ? "-vae1" : "-vae0";
        return key;
    }

    void apply_loras_at_runtime(const std::vector<ModelManager::LoraSpec>& loras) {
        if (model_manager != nullptr) {
            model_manager->set_loras({}, version);
        }

        // ── cached fast path ─────────────────────────────────────────────────
        //
        // The adapter gguf is otherwise re-opened, re-parsed, re-allocated and
        // re-staged on EVERY img_gen (measured 0.42-0.55 s of CPU, outside
        // sampling) for a `lora` array the user is not changing while they
        // iterate on seeds.
        //
        // KEY = current_lora_signature (per spec, in order: resolved path, file
        // size, file mtime, multiplier bit pattern, is_high_noise, tensor-name
        // prefix filter, required flag) + runtime_lora_environment_key()
        // (SDVersion, n_threads, base_model_epoch, which module runners exist).
        // A hit rebuilds NOTHING -- the same LoraModel objects, holding the same
        // tensors at the same addresses, are re-attached -- so the graph the
        // adapter contributes is identical by construction.
        //
        // Multiplier is in the key even though LoraModel::multiplier is only read
        // at graph-build time (lora.hpp forward_lora / get_lora_weight_diff), so
        // assigning it on reuse would also be correct. Keying on it costs one
        // reload when the strength changes -- exactly today's cost -- and removes
        // the need for that argument to stay true.
        //
        // `current_lora_signature` is set by apply_loras(), the only caller. If a
        // future caller reaches here without setting it, an empty signature would
        // make every non-empty set share one key -- so that case declines to
        // cache rather than guessing.
        //
        // Escape hatch: SD_RUNTIME_LORA_CACHE=0 restores the reload-every-request
        // behaviour exactly, and is the reference arm for any A/B of this change.
        // It is also the lever if holding the adapter resident between renders
        // ever costs a co-resident model its headroom.
        const bool cache_enabled = sd_cache::entries_from_env("SD_RUNTIME_LORA_CACHE", 1, 1) != 0;
        const bool cacheable     = cache_enabled && !loras.empty() && !current_lora_signature.empty();

        const std::string binding_key = cacheable
                                            ? current_lora_signature + "|" + runtime_lora_environment_key()
                                            : std::string();
        if (cacheable && runtime_lora_binding.has_value() && runtime_lora_binding->key == binding_key) {
            runtime_lora_models = runtime_lora_binding->models;
            clear_lora_adapters();
            if (cond_stage_model && runtime_lora_binding->cond_stage_adapter) {
                cond_stage_model->set_weight_adapter(runtime_lora_binding->cond_stage_adapter);
            }
            if (diffusion_model && runtime_lora_binding->diffusion_adapter) {
                diffusion_model->set_weight_adapter(runtime_lora_binding->diffusion_adapter);
                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->set_weight_adapter(runtime_lora_binding->diffusion_adapter);
                }
            }
            if (first_stage_model && runtime_lora_binding->first_stage_adapter) {
                first_stage_model->set_weight_adapter(runtime_lora_binding->first_stage_adapter);
            }
            LOG_INFO("[CACHE] lora-binding HIT: reusing the loaded adapter set (no re-read)");
            return;
        }
        if (!loras.empty()) {
            LOG_INFO("[CACHE] lora-binding MISS (%s)",
                     !cache_enabled ? "disabled"
                                    : (runtime_lora_binding.has_value() ? "key changed" : "cold"));
        }

        runtime_lora_models.clear();
        clear_lora_adapters();
        // Release BEFORE loading the replacement, never after: the outgoing
        // adapter's resident params must be returned before the incoming set is
        // allocated, or a switch would transiently need both (~926 MiB each for
        // the r256 Q8_0 adapter). This also covers the switch to NO adapter --
        // an empty request drops the binding rather than pinning dead VRAM
        // through a t2i render.
        release_runtime_lora_binding();
        if (loras.empty()) {
            return;
        }

        std::set<std::string> model_tensor_names;
        if (model_manager != nullptr) {
            model_tensor_names = model_manager->tensor_names();
        }

        RuntimeLoraBinding binding;
        binding.key = binding_key;

        LOG_INFO("apply lora at runtime");
        if (cond_stage_model) {
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_cond_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            auto cond_stage_lora_models =
                load_runtime_loras_for_module(loras,
                                              model_tensor_names,
                                              SDBackendModule::TE,
                                              lora_tensor_filter);
            // Only attach the adapter when there are LoRAs targeting the cond_stage model.
            // An empty MultiLoraAdapter still routes every linear/conv through
            // forward_with_lora() instead of the direct kernel path — slower for no benefit.
            if (!cond_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(cond_stage_lora_models);
                cond_stage_model->set_weight_adapter(multi_lora_adapter);
                binding.cond_stage_adapter = std::move(multi_lora_adapter);
            }
        }
        if (diffusion_model) {
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_diffusion_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            auto diffusion_lora_models =
                load_runtime_loras_for_module(loras,
                                              model_tensor_names,
                                              SDBackendModule::DIFFUSION,
                                              lora_tensor_filter);
            if (!diffusion_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(diffusion_lora_models);
                diffusion_model->set_weight_adapter(multi_lora_adapter);
                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->set_weight_adapter(multi_lora_adapter);
                }
                binding.diffusion_adapter = std::move(multi_lora_adapter);
            }
        }

        if (first_stage_model) {
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_first_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            auto first_stage_lora_models =
                load_runtime_loras_for_module(loras,
                                              model_tensor_names,
                                              SDBackendModule::VAE,
                                              lora_tensor_filter);
            if (!first_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(first_stage_lora_models);
                first_stage_model->set_weight_adapter(multi_lora_adapter);
                binding.first_stage_adapter = std::move(multi_lora_adapter);
            }
        }

        // Only cache a set that actually produced something. A key whose loads
        // all failed or matched no tensors must be retried, not remembered as
        // "the adapter is already applied".
        if (cacheable && !runtime_lora_models.empty()) {
            binding.models = runtime_lora_models;
            runtime_lora_binding.emplace(std::move(binding));
        }
    }

    void lora_stat() {
        if (!runtime_lora_models.empty()) {
            LOG_INFO("runtime_lora_models:");
            for (auto& lora_model : runtime_lora_models) {
                lora_model->stat();
            }
        }
    }

    // Canonical, order-sensitive description of a resolved LoRA set.
    //
    // FILE IDENTITY IS IN IT, not just the path: an adapter is routinely rebuilt
    // in place under the same name (tools/convert_lora_q8.py writes over the
    // served gguf), and a cache keyed on the path alone would keep serving the
    // previous build's weights for the rest of the worker's life. Size + mtime
    // is the same identity ModelLoader would have re-read anyway; a content hash
    // would mean reading ~1 GB per request, which is the cost being removed.
    //
    // A path that cannot be stat()ed contributes `id=0` with zeroed size/mtime,
    // which never equals a successful stat of the same file, so an unreadable
    // file can never produce a hit against a readable one.
    static std::string lora_set_signature(const std::vector<ModelManager::LoraSpec>& loras,
                                          bool apply_immediately) {
        std::string signature = apply_immediately ? "mode=immediate" : "mode=runtime";
        for (const auto& spec : loras) {
            uint64_t size_bytes = 0;
            int64_t mtime_ticks = 0;
            bool identified     = false;
            try {
                std::error_code error;
                const std::filesystem::path path(spec.path);
                const auto file_size = std::filesystem::file_size(path, error);
                if (!error) {
                    const auto write_time = std::filesystem::last_write_time(path, error);
                    if (!error) {
                        size_bytes  = static_cast<uint64_t>(file_size);
                        mtime_ticks = static_cast<int64_t>(write_time.time_since_epoch().count());
                        identified  = true;
                    }
                }
            } catch (const std::exception&) {
                identified = false;
            }
            // Compare the multiplier by BIT PATTERN: two floats that print the
            // same are not necessarily the same float.
            uint32_t multiplier_bits = 0;
            static_assert(sizeof(multiplier_bits) == sizeof(spec.multiplier),
                          "LoraSpec::multiplier must be a 32-bit float to key on its bits");
            std::memcpy(&multiplier_bits, &spec.multiplier, sizeof(multiplier_bits));

            char scratch[128];
            snprintf(scratch, sizeof(scratch),
                     "|sz=%llu|mt=%lld|mul=%08x|hn=%d|req=%d|id=%d|pfx=",
                     (unsigned long long)size_bytes,
                     (long long)mtime_ticks,
                     (unsigned)multiplier_bits,
                     spec.is_high_noise ? 1 : 0,
                     spec.required ? 1 : 0,
                     identified ? 1 : 0);
            signature += "\n" + spec.path + scratch + spec.tensor_name_prefix_filter;
        }
        return signature;
    }

    void apply_loras(const sd_lora_t* loras, uint32_t lora_count) {
        std::vector<ModelManager::LoraSpec> all_loras;
        all_loras.reserve(lora_count);
        for (uint32_t i = 0; i < lora_count; i++) {
            std::string lora_id = SAFE_STR(loras[i].path);
            ModelManager::LoraSpec lora_spec;
            lora_spec.path          = lora_id;
            lora_spec.multiplier    = loras[i].multiplier;
            lora_spec.is_high_noise = loras[i].is_high_noise;
            all_loras.push_back(std::move(lora_spec));
            if (loras[i].is_high_noise) {
                lora_id = "|high_noise|" + lora_id;
            }
            LOG_DEBUG("lora %s:%.2f", lora_id.c_str(), loras[i].multiplier);
        }

        for (auto& extension : generation_extensions) {
            extension->collect_loras(all_loras);
        }

        // Computed AFTER collect_loras(), so extension-supplied adapters (PuLID,
        // PhotoMaker) are in it, and before apply_*, which reads it.
        current_lora_signature = all_loras.empty() ? std::string()
                                                   : lora_set_signature(all_loras, apply_lora_immediately);

        int64_t t0 = ggml_time_ms();
        if (apply_lora_immediately) {
            apply_loras_immediately(all_loras);
        } else {
            apply_loras_at_runtime(all_loras);
        }
        int64_t t1 = ggml_time_ms();
        if (!all_loras.empty()) {
            LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        }
    }

    void reset_generation_extensions() {
        for (auto& extension : generation_extensions) {
            extension->reset_runtime_condition();
        }
    }

    void prepare_generation_extensions(const sd_pm_params_t& pm_params,
                                       const sd_pulid_params_t& pulid_params,
                                       ConditionerParams& condition_params,
                                       int total_steps) {
        reset_generation_extensions();
        GenerationExtensionConditionContext ctx{
            cond_stage_model.get(),
            condition_params,
            pm_params,
            pulid_params,
            n_threads,
            total_steps,
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
                                         const sd::Tensor<float>& denoise_mask,
                                         int step) {
        if (auto sefi_denoiser = std::dynamic_pointer_cast<SefiFlowDenoiser>(denoiser)) {
            int sched_idx = step > 0 ? step - 1 : 0;
            if (sched_idx >= static_cast<int>(sefi_denoiser->tex_timesteps.size())) {
                sched_idx = static_cast<int>(sefi_denoiser->tex_timesteps.size()) - 1;
            }
            return {sefi_denoiser->sem_timesteps[sched_idx],
                    sefi_denoiser->tex_timesteps[sched_idx]};
        }
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B" ||
            sd_version_is_longcat_avatar(version) ||
            (version == VERSION_WAN2_2_I2V && !denoise_mask.empty())) {
            // Avatar reference latents are clean conditioning anchors. Their
            // per-frame timestep must be zero whenever the denoise mask pins
            // them, matching the model's trained AI2V conditioning path.
            int64_t frame_count = init_latent.shape()[2];
            auto new_timesteps  = std::vector<float>(static_cast<size_t>(frame_count), timesteps[0]);

            if (!denoise_mask.empty() && denoise_mask.dim() >= 4 && denoise_mask.shape()[2] == frame_count) {
                for (int64_t frame = 0; frame < frame_count; ++frame) {
                    float value = denoise_mask.dim() == 5 ? denoise_mask.index(0, 0, frame, 0, 0) : denoise_mask.index(0, 0, frame, 0);
                    if (value == 0.f) {
                        new_timesteps[static_cast<size_t>(frame)] = 0.f;
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
                                                     const sd::Tensor<float>& denoise_mask,
                                                     int64_t ref_token_count = 0) {
        if (timesteps.empty() || denoise_mask.empty() || init_latent.dim() < 4 || denoise_mask.dim() < 4) {
            GGML_ASSERT(ref_token_count == 0);
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
            GGML_ASSERT(ref_token_count == 0);
            return timesteps;
        }

        // TASS reference tokens are appended after the target grid and are never
        // denoised, so they carry timestep 0.  They must be present: the graph
        // sizes its per-token modulation over target ++ reference.  The
        // modulation-collapse dedup folds them all into one bucket.
        std::vector<float> video_timesteps(static_cast<size_t>(width * height * frames + ref_token_count), 0.f);
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
                } else if (sd_version_uses_flux_vae(version)) {
                    latent_rgb_proj = flux_latent_rgb_proj;
                    latent_rgb_bias = flux_latent_rgb_bias;
                } else if (sd_version_uses_wan_vae(version)) {
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
        if (sd_version_is_boogu_image(version)) {
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

    void report_sample_progress(int step, size_t total_steps, int64_t* last_progress_us) {
        if (step > 0 || step == -(int)total_steps) {
            int64_t now        = ggml_time_us();
            int showstep       = std::abs(step);
            float step_seconds = last_progress_us != nullptr && *last_progress_us > 0
                                     ? (now - *last_progress_us) / 1000000.f
                                     : 0.f;
            pretty_progress(showstep, (int)total_steps, step_seconds);
            if (last_progress_us != nullptr) {
                *last_progress_us = now;
            }
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

    // Does the packed denoise mask ask for ANY audio element to be generated?
    //
    // Used instead of a separate "gap fill" flag so a flag and a mask can never disagree: the mask
    // IS the instruction, and the timestep schedule simply follows it. With audio held (the
    // ordinary supplied-drive case) the audio block is uniformly 0 and this returns false, giving
    // the one-element {0.f} timestep exactly as before — byte-identical. With gap-fill it returns
    // true, so the audio runs the FULL schedule and the re-injection blend keeps the supplied
    // regions pinned while the gaps actually denoise. A scalar timestep of 0 tells the model all
    // audio is already clean, which is why the gaps would otherwise never be generated.
    //
    // The audio occupies the LAST `extra_ch` channels of the packed mask, where extra_ch is the
    // number of [W,H,F] planes the audio latent rounds up to — derived here from the mask's own
    // shape so it cannot drift from what the packer did.
    static bool ltxav_audio_mask_has_free_elements(const sd::Tensor<float>& denoise_mask, int audio_length) {
        constexpr int64_t kFrequencyBins = 16;
        constexpr int64_t kAudioChannels = 8;
        if (denoise_mask.empty() || denoise_mask.dim() < 4 || audio_length <= 0) {
            return false;
        }
        const int64_t width  = denoise_mask.shape()[0];
        const int64_t height = denoise_mask.shape()[1];
        const int64_t frames = denoise_mask.shape()[2];
        const int64_t total_ch = denoise_mask.shape()[3];
        const int64_t spatial  = width * height * frames;
        if (spatial <= 0) {
            return false;
        }
        const int64_t audio_values = kFrequencyBins * static_cast<int64_t>(audio_length) * kAudioChannels;
        const int64_t extra_ch     = (audio_values + spatial - 1) / spatial;
        if (extra_ch <= 0 || extra_ch > total_ch) {
            return false;
        }
        const float* data = denoise_mask.data();
        // Only the REAL audio elements are consulted; the rounding padding beyond them carries
        // whatever the uniform fill left there and says nothing about the caller's intent.
        const int64_t first = (total_ch - extra_ch) * spatial;
        for (int64_t i = 0; i < audio_values; ++i) {
            if (data[first + i] > 0.5f) {
                return true;
            }
        }
        return false;
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
                             const RefImageParams& ref_image_params,
                             const sd::Tensor<float>& denoise_mask,
                             const sd::Tensor<float>& vace_context,
                             float vace_strength,
                             int audio_length,
                             float frame_rate,
                             const sd_cache_params_t* cache_params,
                             const sd::Tensor<float>& video_positions = {},
                             const sd::Tensor<float>& audio_positions = {},
                             bool ltxav_audio_fixed                   = false,
                             float a2v_guidance_scale                 = 1.f,
                             float a2v_ramp_end                       = 1.f,
                             const sd::ltx_relay::Plan* relay_plan    = nullptr,
                             float relay_steps_frac                   = 1.f,
                             // TASS overlap references. Empty is the ordinary path:
                             // no extra tokens, source ids null, and the RoPE phase
                             // is an exact no-op.
                             const sd::Tensor<float>& ref_video_x     = {},
                             const std::vector<float>* video_source_ids = nullptr,
                             float tass_phase_scale                   = 1.f) {
        struct RunnerDoneOnExit {
            GGMLRunner* runner = nullptr;
            ~RunnerDoneOnExit() {
                if (runner != nullptr) {
                    runner->runner_done();
                }
            }
        };
        RunnerDoneOnExit sample_diffusion_runner_done{work_diffusion_model.get()};

        RunnerDoneOnExit sample_control_runner_done{!control_image.empty() && control_net != nullptr ? control_net.get() : nullptr};

        // TASS overlap references extend every per-token vector the DiT builds
        // (positions, source ids, timesteps).  A reference is a clean latent, so
        // its tokens are pinned at timestep zero; the per-token timestep vector
        // only exists on the LTXAV masked path, which is why the caller is
        // required to have supplied a denoise mask alongside the references.
        const int64_t ref_video_token_count =
            ref_video_x.empty() ? 0
                                : ref_video_x.shape()[0] * ref_video_x.shape()[1] * ref_video_x.shape()[2];
        GGML_ASSERT(ref_video_token_count == 0 || (sd_version_is_ltxav(version) && !denoise_mask.empty()));

        a2v_ramp_end = std::clamp(a2v_ramp_end, 0.f, 1.f);
        if (a2v_guidance_scale != 1.f && sd_version_is_ltxav(version)) {
            LOG_INFO("LTXAV A2V guidance scale=%.2f ramp_end=%.2f (+1 audio-decoupled forward/step)",
                     a2v_guidance_scale, a2v_ramp_end);
        }

        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);
        float cfg_scale     = guidance.txt_cfg;
        float img_cfg_scale = guidance.img_cfg;
        float slg_scale     = guidance.slg.scale;
        bool slg_uncond     = sd::guidance::parse_skip_layer_guidance_uncond_arg(extra_sample_args);

        std::vector<float> guidance_schedule = sd::guidance::parse_guidance_schedule(extra_sample_args);
        if (!guidance_schedule.empty() && guidance_schedule.size() != sigmas.size() - 1) {
            if (guidance_schedule.size() > sigmas.size()) {
                LOG_WARN("guidance_schedule length (%zu) is greater than number of steps (%zu)", guidance_schedule.size(), sigmas.size() - 1);
                LOG_WARN("truncating guidance_schedule to match step count");
                guidance_schedule.resize(sigmas.size() - 1);
            } else {
                LOG_INFO("padding guidance_schedule with cfg_scale");
                while (guidance_schedule.size() < sigmas.size() - 1) {
                    guidance_schedule.push_back(cfg_scale);
                }
            }
        }

        if (!guidance_schedule.empty()) {
            std::string schedule_str = "[";
            for (size_t i = 0; i < guidance_schedule.size(); ++i) {
                schedule_str += std::to_string(guidance_schedule[i]);
                if (i < guidance_schedule.size() - 1) {
                    schedule_str += ", ";
                }
            }
            schedule_str += "]";
            LOG_DEBUG("using guidance schedule: %s", schedule_str.c_str());
        }

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
        // precise attention backend at the first (and optionally last/middle) denoise
        // steps, and SA3 in the others.  The CUDA dispatcher reads GGML_LTX_SA3 at each
        // attention call (ggml-cuda/fattn.cu ggml_cuda_flash_attn_ext, per-forward
        // getenv, not cached), so we scope the process environment to this sample only.
        // This is deliberately opt-in: GGML_LTX_SA3_POLICY unset => behaviour is exactly
        // today's (whatever static GGML_LTX_SA3 the caller set, untouched).
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

        int64_t last_progress_us     = ggml_time_us();
        SamplePreviewContext preview = prepare_sample_preview_context();

        sd::Tensor<float> x_t      = !noise.empty()
                                         ? denoiser->noise_scaling(sigmas[0], noise, init_latent)
                                         : init_latent;
        sd::Tensor<float> denoised = x_t;

        auto denoise = [&](const sd::Tensor<float>& x, float sigma, int step) -> sd::guidance::GuiderOutput {
            if (get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_DEBUG("cancelling generation");
                return {};
            }

            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
                last_progress_us = ggml_time_us();
            }

            // Per-step SA3 policy toggle. Sampler `step` is 1-based and its sign is a
            // flag (a negative step marks the uncond/CFG pass), so compare on abs().
            // Toggled via setenv BEFORE this step's DiT forward launches; the CUDA
            // attention dispatcher re-reads GGML_LTX_SA3 each forward. No-op unless
            // GGML_LTX_SA3_POLICY selected an LTX-AV policy above.
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
                timesteps_vec          = process_ltxav_video_timesteps(base_timesteps_vec, init_latent, denoise_mask,
                                                                       ref_video_token_count);
                // Follow the MASK, not the flag: audio held everywhere -> the historical
                // one-element {0.f}; any element left free (gap-fill) -> the full schedule, so
                // the gaps denoise while the blend re-pins the supplied regions each step.
                const bool audio_needs_denoise =
                    !ltxav_audio_fixed || ltxav_audio_mask_has_free_elements(denoise_mask, audio_length);
                const std::vector<float> audio_timesteps = audio_needs_denoise
                                                               ? base_timesteps_vec
                                                               : std::vector<float>{0.f};
                audio_timesteps_tensor = sd::Tensor<float>({static_cast<int64_t>(audio_timesteps.size())}, audio_timesteps);
            } else {
                timesteps_vec = process_timesteps(timesteps_vec, init_latent, denoise_mask, step);
            }
            const std::vector<float>& scaling_timesteps_vec = (sd_version_is_ltxav(version) && !denoise_mask.empty())
                                                                  ? base_timesteps_vec
                                                                  : timesteps_vec;
            adjust_sample_step_scalings(shifted_timestep, scaling_timesteps_vec, c_in, &c_skip, &c_out);

            sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
            sd::Tensor<float> guidance_tensor({1}, std::vector<float>{guidance.distilled_guidance});
            sd::Tensor<float> hunyuan_timestep_r_tensor;
            if (sd_version_is_hunyuan_video(version) && step + 1 < sigmas.size()) {
                hunyuan_timestep_r_tensor = sd::Tensor<float>::from_vector({sigmas[step + 1]});
            }
            sd::Tensor<float> noised_input = x * c_in;
            if (!denoise_mask.empty() && (version == VERSION_WAN2_2_TI2V || sd_version_is_ltxav(version) || sd_version_is_lingbot_video(version) || sd_version_is_longcat_avatar(version))) {
                noised_input = noised_input * denoise_mask + init_latent * (1.0f - denoise_mask);
            }

            if (cache_runtime.spectrum_enabled && cache_runtime.spectrum.should_predict()) {
                cache_runtime.spectrum.predict(&denoised);
                if (!denoise_mask.empty()) {
                    denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
                }
                if (sd_should_preview_denoised() && preview.callback != nullptr) {
                    preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
                }
                report_sample_progress(step, steps, &last_progress_us);
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
            diffusion_params.x                = &noised_input;
            diffusion_params.timesteps        = &timesteps_tensor;
            diffusion_params.ref_image_params = ref_image_params;
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

            // Prompt Relay is a bias on the POSITIVE prompt's cross-attention
            // only. The unconditional encode is a different string with a
            // different token count, so its key axis does not match the mask at
            // all; and the mask only shapes semantic layout, which is decided in
            // the first steps, so it can be retired early (relay_steps_frac).
            // `step` is ONE-based here (the denoiser calls model(x, sigma, i+1)), so the
            // count of masked steps IS the last masked step -- no -1. The -1 that used to
            // be here assumed a 0-based counter and retired the mask a step early: at the
            // default frac=1.0 the FINAL step ran unmasked, and at small fractions it cost
            // a whole step out of two or three.
            const int relay_last_step = relay_plan == nullptr
                                            ? 0
                                            : std::max(1, static_cast<int>(std::ceil(
                                                              std::clamp(relay_steps_frac, 0.f, 1.f) *
                                                              static_cast<float>(steps))));
            // A NEGATIVE step is some samplers' marker for "this is the primary
            // evaluation", not a position in the schedule. Those keep the mask
            // unconditionally, exactly as before -- narrowing them to "unmasked" here
            // would silently switch relay off for every sampler except euler_a.
            const sd::ltx_relay::Plan* step_relay =
                (relay_plan != nullptr && (step < 0 || step <= relay_last_step)) ? relay_plan : nullptr;

            auto run_condition = [&](const SDCondition& condition,
                                     const sd::Tensor<float>* c_concat_override                 = nullptr,
                                     const std::vector<int>* local_skip_layers                  = nullptr,
                                     const std::vector<sd::Tensor<float>>* ref_latents_override = nullptr,
                                     bool skip_a2v_pass                                          = false) -> sd::Tensor<float> {
                diffusion_params.context     = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
                diffusion_params.c_concat    = c_concat_override != nullptr ? c_concat_override : (condition.c_concat.empty() ? nullptr : &condition.c_concat);
                diffusion_params.y           = condition.c_vector.empty() ? nullptr : &condition.c_vector;
                diffusion_params.ref_latents = ref_latents_override != nullptr ? ref_latents_override : (condition.c_ref_images.empty() ? &ref_latents : &condition.c_ref_images);

                if (sd_version_is_unet(version)) {
                    int nvf = -1;
                    if (animatediff_loaded && noised_input.dim() >= 4 && noised_input.shape()[3] > 1) {
                        nvf = static_cast<int>(noised_input.shape()[3]);
                    }
                    diffusion_params.extra = UNetDiffusionExtra{nvf, &controls, control_strength};
                } else if (sd_version_is_sd3(version)) {
                    diffusion_params.extra = SkipLayerDiffusionExtra{local_skip_layers};
                } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) || sd_version_is_longcat(version) || sd_version_is_sefi_image(version)) {
                    diffusion_params.extra = FluxDiffusionExtra{&guidance_tensor,
                                                                local_skip_layers};
                } else if (sd_version_is_anima(version)) {
                    diffusion_params.extra = AnimaDiffusionExtra{condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids,
                                                                 condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights};
                } else if (sd_version_is_wan(version)) {
                    diffusion_params.extra = WanDiffusionExtra{vace_context.empty() ? nullptr : &vace_context,
                                                               vace_strength};
                } else if (sd_version_is_hunyuan_video(version)) {
                    diffusion_params.extra = HunyuanVideoDiffusionExtra{
                        &guidance_tensor,
                        condition.extra_c_crossattns.empty() ? nullptr : &condition.extra_c_crossattns[0],
                        condition.c_vector.empty() ? nullptr : &condition.c_vector,
                        hunyuan_timestep_r_tensor.empty() ? nullptr : &hunyuan_timestep_r_tensor};
                } else if (version == VERSION_HIDREAM_O1) {
                    diffusion_params.extra = HiDreamO1DiffusionExtra{
                        condition.c_input_ids.empty() ? nullptr : &condition.c_input_ids,
                        condition.c_position_ids.empty() ? nullptr : &condition.c_position_ids,
                        condition.c_token_types.empty() ? nullptr : &condition.c_token_types,
                        condition.c_vinput_mask.empty() ? nullptr : &condition.c_vinput_mask,
                        condition.c_image_embeds.empty() ? nullptr : &condition.c_image_embeds};
                } else if (sd_version_is_ltxav(version)) {
                    // c_token_pieces is populated only by the piece-wise
                    // positive encode, so it is exactly the discriminator for
                    // "this condition's key axis is the one the mask was built
                    // against" -- the unconditional encode, and any condition an
                    // extension substituted, leave it empty.
                    diffusion_params.extra = LTXAVDiffusionExtra{
                        nullptr,
                        audio_timesteps_tensor.empty() ? nullptr : &audio_timesteps_tensor,
                        audio_length,
                        frame_rate,
                        video_positions.empty() ? nullptr : &video_positions,
                        audio_positions.empty() ? nullptr : &audio_positions,
                        skip_a2v_pass,
                        ref_video_x.empty() ? nullptr : &ref_video_x,
                        (ref_video_x.empty() || video_source_ids == nullptr) ? nullptr : video_source_ids,
                        tass_phase_scale,
                        condition.c_token_pieces.empty() ? nullptr : step_relay};
                } else if (sd_version_is_longcat_avatar(version)) {
                    diffusion_params.extra = LongCatAvatarDiffusionExtra{step};
                } else if (sd_version_is_minit2i(version)) {
                    diffusion_params.extra = MiniT2IDiffusionExtra{
                        condition.c_vector.empty() ? nullptr : &condition.c_vector};
                } else {
                    diffusion_params.extra = std::monostate{};
                }

                sd::Tensor<float> cached_output;
                if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                    return std::move(cached_output);
                }

                for (const auto& extension : generation_extensions) {
                    extension->before_diffusion(diffusion_params, step);
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

            cond_out = run_condition(*positive_condition, c_concat_override);
            if (cond_out.empty()) {
                return {};
            }

            if (a2v_guidance_scale != 1.f && sd_version_is_ltxav(version) && ltxav_audio_fixed) {
                float effective_scale = a2v_guidance_scale;
                if (a2v_ramp_end < 1.f && !sigmas.empty() && sigmas[0] > 0.f) {
                    const float fraction = std::clamp(1.f - sigma / sigmas[0], 0.f, 1.f);
                    const float ramp = 1.f + (a2v_ramp_end - 1.f) * fraction;
                    effective_scale = 1.f + (a2v_guidance_scale - 1.f) * ramp;
                }
                if (std::fabs(effective_scale - 1.f) > .02f) {
                    sd::Tensor<float> audio_decoupled =
                        run_condition(*positive_condition, c_concat_override, nullptr, nullptr, true);
                    if (audio_decoupled.empty()) {
                        return {};
                    }
                    cond_out = cond_out + (cond_out - audio_decoupled) * (effective_scale - 1.f);
                }
            }

            if (!uncond.empty()) {
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

            sd::guidance::GuiderOutput guided = guidance_schedule.empty() ? primary_guidance.forward(guidance_input, {}) : primary_guidance.forward(guidance_input, {}, guidance_schedule[guidance_schedule.size() - 1 - step]);
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
            if (sd_should_preview_denoised() && preview.callback != nullptr) {
                preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
            }
            report_sample_progress(step, steps, &last_progress_us);
            output.pred = denoised;
            return output;
        };

        auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser, extra_sample_args, denoiser);
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
            if (sd_version_is_wan(version) || sd_version_is_lingbot_video(version) || sd_version_is_longcat_avatar(version)) {
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
            } else if (sd_version_is_hunyuan_video(version)) {
                latent_channel = 32;
            } else if (version == VERSION_HIDREAM_O1) {
                latent_channel = 3;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                latent_channel = 3;
            } else if (sd_version_is_minit2i(version)) {
                latent_channel = 3;
            } else if (sd_version_is_pid(version)) {
                latent_channel = 3;
            } else if (sd_version_is_sefi_image(version)) {
                latent_channel = 144;
            } else if (sd_version_uses_flux2_vae(version)) {
                latent_channel = 128;
            } else if (sd_version_is_mage_flow(version)) {
                latent_channel = 128;
            } else {
                latent_channel = 16;
            }
        }
        return latent_channel;
    }

    int get_image_channels() const {
        return version == VERSION_QWEN_IMAGE_LAYERED ? 4 : 3;
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
        } else if (sd_version_is_wan(version) || sd_version_is_lingbot_video(version) || sd_version_is_hunyuan_video(version) || sd_version_is_longcat_avatar(version)) {
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
        if (sd_version_is_wan(version) || sd_version_is_lingbot_video(version) || sd_version_is_hunyuan_video(version) || sd_version_is_longcat_avatar(version)) {
            return (latent_frames - 1) * 4 + 1;
        }
        return latent_frames;
    }

    int align_video_frames(int frames) {
        return latent_frames_to_video_frames(video_frames_to_latent_frames(frames));
    }

    sd::Tensor<float> encode_to_vae_latents(const sd::Tensor<float>& x) {
        auto latents = first_stage_model->encode(n_threads, x, vae_tiling_params, circular_x, circular_y);
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

    // ── reference-image VAE latent, content-addressed ────────────────────────
    //
    // A reference is an IDENTITY, not a shot: the same pixels encode to the same
    // latent for every seed the user tries on one edit. Today it is re-encoded
    // per request, serialised ahead of sampling inside the measured 3.30 s of
    // pre-sampling GPU idle, and the encode owns a 2356 MB compute buffer.
    // ltx-video already keys its reference latents this way
    // (ltxav_encode_with_reference_cache below); this is the same idea in RAM.
    //
    // KEY, part 1: the content+shape hash of `pixels` -- the EXACT tensor handed
    // to the VAE, after the crop and the resize. Every geometry input is
    // therefore in the key by construction rather than by enumeration:
    // resize_before_vae, resize_vae_to_target, crop_vae_to_target_ar,
    // vae_input_max_pixels, the vae_scale_factor rounding, and the source image
    // itself. Shape is joined explicitly because a transposed pair of uniform
    // tensors has identical bytes.
    //
    // KEY, part 2 (`env`): everything encode_first_stage() reads that is NOT in
    // those bytes -- see encode_reference_latent_env().
    //
    // GATED TO krea2 ON PURPOSE. AutoencoderKL::vae_output_to_latents() draws
    // from `rng` to sample the posterior (auto_encoder_kl.hpp:761), so for those
    // models the encode is neither reproducible nor side-effect-free: serving a
    // cached latent would both differ from a fresh sample and leave the RNG
    // stream un-advanced, changing the sampling noise and hence the whole image.
    // krea2 uses the Wan VAE, whose vae_output_to_latents is `SD_UNUSED(rng);
    // return vae_output;` (wan_vae.hpp:1318) -- deterministic, no draw. Widening
    // this needs that check repeated per VAE, not an assumption.
    //
    // Both the REQUESTED and the current circular axes are in the env, and the
    // pair is load-bearing. configure_image_vae_axes() splits one request flag
    // into two values -- the VAE runner's internal axes (`requested && tile >=
    // latent`) and the `sd->circular_*` that encode() is passed (`requested &&
    // tile < latent`) -- and the second does NOT determine the first: with
    // tiling on and a tile at least as large as the latent, sd->circular_x is
    // false for a requested value of either true or false, while the runner's
    // internal axis follows the request. Keying on the post-guard value alone
    // would collide a circular encode with a non-circular one.
    std::string encode_reference_latent_env(int target_width,
                                            int target_height,
                                            bool requested_circular_x,
                                            bool requested_circular_y) const {
        char scratch[512];
        snprintf(scratch, sizeof(scratch),
                 "v%d|e%llu|n%d|w%d|h%d|rcx%d|rcy%d|cx%d|cy%d|tile%d,%d,%d,%d,%.9g,%.9g,%.9g|xa=",
                 static_cast<int>(version),
                 (unsigned long long)base_model_epoch,
                 n_threads,
                 target_width,
                 target_height,
                 requested_circular_x ? 1 : 0,
                 requested_circular_y ? 1 : 0,
                 circular_x ? 1 : 0,
                 circular_y ? 1 : 0,
                 vae_tiling_params.enabled ? 1 : 0,
                 vae_tiling_params.temporal_tiling ? 1 : 0,
                 vae_tiling_params.tile_size_x,
                 vae_tiling_params.tile_size_y,
                 (double)vae_tiling_params.target_overlap,
                 (double)vae_tiling_params.rel_size_x,
                 (double)vae_tiling_params.rel_size_y);
        std::string env = scratch;
        env += vae_tiling_params.extra_tiling_args != nullptr ? vae_tiling_params.extra_tiling_args : "";
        // A runtime LoRA can target the first_stage model, and the immediate path
        // folds LoRA deltas into the VAE weights outright. This is the whole
        // request's adapter set, i.e. a superset of the VAE-targeting subset: it
        // can only cost hit rate, never correctness.
        env += "|lora=" + current_lora_signature;
        return env;
    }

    sd::Tensor<float> encode_reference_latent_cached(const sd::Tensor<float>& pixels,
                                                     int target_width,
                                                     int target_height,
                                                     bool requested_circular_x,
                                                     bool requested_circular_y,
                                                     int index) {
        const size_t entries = sd_cache::entries_from_env("SD_REF_LATENT_CACHE_ENTRIES", 4, 64);
        if (entries == 0 || pixels.empty() || !sd_version_is_krea2(version)) {
            if (!pixels.empty() && sd_version_is_krea2(version)) {
                LOG_INFO("[CACHE] ref-latent %d MISS (disabled)", index);
            }
            return encode_first_stage(pixels);
        }
        reference_latent_cache.set_capacity(entries);

        const std::string key = sd_cache::tensor_content_key(pixels) + "|" +
                                encode_reference_latent_env(target_width,
                                                            target_height,
                                                            requested_circular_x,
                                                            requested_circular_y);
        if (const sd::Tensor<float>* hit = reference_latent_cache.get(key); hit != nullptr) {
            LOG_INFO("[CACHE] ref-latent %d HIT (no VAE encode pass)", index);
            return *hit;
        }
        LOG_INFO("[CACHE] ref-latent %d MISS (encoding; %zu entries resident)",
                 index, reference_latent_cache.size());
        auto latent = encode_first_stage(pixels);
        if (!latent.empty()) {
            reference_latent_cache.put(key, latent);
        }
        return latent;
    }

    sd::Tensor<float> decode_first_stage(const sd::Tensor<float>& x, bool decode_video = false) {
        if (sd_version_is_pid(version) || sd_version_is_minit2i(version)) {
            return sd::ops::clamp((x + 1.f) * 0.5f, 0.0f, 1.0f);
        }
        auto latents = first_stage_model->diffusion_to_vae_latents(x);
        first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
        auto decoded = first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
        if (decoded.empty() && auto_fit_enabled) {
            bool prefer_temporal_tiling = decode_video && std::dynamic_pointer_cast<LTXVideoVAE>(first_stage_model) != nullptr;
            if (sd::backend_fit::prepare_vae_decode_retry_tiling(vae_tiling_params, prefer_temporal_tiling)) {
                first_stage_model->free_compute_buffer();
                first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
                decoded = first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
            }
        }
        return decoded;
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
        return waveform;
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

    std::string get_default_ref_image_preset(SDVersion version) const {
        if (sd_version_is_longcat(version)) {
            return "longcat";
        } else if (sd_version_is_flux(version)) {
            return "flux_kontext";
        } else if (sd_version_is_flux2(version) || sd_version_is_sefi_image(version)) {
            return "flux2";
        } else if (version == VERSION_QWEN_IMAGE_LAYERED) {
            return "qwen_layered";
        } else if (sd_version_is_qwen_image(version)) {
            return "qwen";
        } else if (sd_version_is_mage_flow(version)) {
            return "mage_flow";
        } else if (sd_version_is_z_image(version) || sd_version_is_boogu_image(version)) {
            return "z_image_omni";
        } else if (sd_version_is_krea2(version)) {
            // have to make a choice between "krea2_edit" mode (for lbouaraba/krea2edit)
            // and "krea2_ostris_edit" (for krea2 ostris edit)
            // since krea2 ostris edit support predates, it should probably be default
            return "krea2_ostris_edit";
        } else if (sd_version_is_anima(version)) {
            return "cosmos_reference";
        }
        return "default";
    }

    RefImageParams resolve_ref_image_params(const char* ref_image_args) const {
        RefImageParams params;
        std::string preset_name = get_default_ref_image_preset(version);

        for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
            if (key == "preset") {
                std::string requested_preset_name = value;
                if (REF_IMAGE_PRESETS.count(requested_preset_name)) {
                    preset_name = requested_preset_name;
                } else if (value != "default") {
                    std::string valid_list;
                    for (auto const& [name, _] : REF_IMAGE_PRESETS) {
                        valid_list += (valid_list.empty() ? "" : ", ") + name;
                    }
                    LOG_WARN("ignoring invalid reference image preset '%s'. Valid options: [%s]", value.c_str(), valid_list.c_str());
                }
                break;
            }
        }
        if (preset_name != "default") {
            LOG_INFO("Using '%s' preset for reference images", preset_name.c_str());
            params = REF_IMAGE_PRESETS.at(preset_name);
        }

        for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
            if (key == "pass_to_vlm") {
                if (!parse_strict_bool(value, params.pass_to_vlm)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "pass_to_dit") {
                if (!parse_strict_bool(value, params.pass_to_dit)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "ref_index_mode") {
                if (value == "fixed") {
                    params.ref_index_mode = Rope::RefIndexMode::FIXED;
                } else if (value == "increase") {
                    params.ref_index_mode = Rope::RefIndexMode::INCREASE;
                } else if (value == "decrease") {
                    params.ref_index_mode = Rope::RefIndexMode::DECREASE;
                } else {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "force_ref_timestep_zero") {
                if (!parse_strict_bool(value, params.force_ref_timestep_zero)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "resize_before_vae") {
                if (!parse_strict_bool(value, params.resize_before_vae)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "vae_input_max_pixels") {
                if (!parse_strict_int(value, params.vae_input_max_pixels)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "vlm_resize_mode") {
                if (value == "longest_side") {
                    params.vlm_resize_mode = RefImageResizeMode::LONGEST_SIDE;
                } else if (value == "area") {
                    params.vlm_resize_mode = RefImageResizeMode::AREA;
                } else if (value == "none") {
                    params.vlm_resize_mode = RefImageResizeMode::NONE;
                } else {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "vlm_max_size") {
                if (!parse_strict_int(value, params.vlm_max_size)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "vlm_min_size") {
                if (!parse_strict_int(value, params.vlm_min_size)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "resize_vae_to_target") {
                if (!parse_strict_bool(value, params.resize_vae_to_target)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "crop_vae_to_target_ar") {
                if (!parse_strict_bool(value, params.crop_vae_to_target_ar)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "vlm_picture_labels") {
                if (!parse_strict_bool(value, params.vlm_picture_labels)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "rescale_ref_ids") {
                if (!parse_strict_bool(value, params.rescale_ref_ids)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "center_ref_ids") {
                if (!parse_strict_bool(value, params.center_ref_ids)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "native_ref") {
                // Parsed here only so it is not reported as unknown, and so the resolved value is
                // visible for the sanity check below. It has ALREADY been acted on, by
                // sd_ref_image_args_want_native_geometry() at HTTP decode time; setting it at this
                // point changes nothing, because the reference bytes exist by now.
                if (!parse_strict_bool(value, params.native_ref_geometry)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key != "preset" && key != "vlm_size") {
                LOG_WARN("ignoring unknown reference image arg '%s'", key.c_str());
            }
        }
        for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
            if (key == "vlm_size") {
                int vlm_size;
                if (!parse_strict_int(value, vlm_size)) {
                    LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
                } else {
                    LOG_INFO("vlm_size override: setting both min and max size to %ld", (long)vlm_size);
                    params.vlm_min_size = vlm_size;
                    params.vlm_max_size = vlm_size;
                }
                break;
            }
        }
        if (params.force_ref_timestep_zero && !sd_version_is_krea2(version)) {
            LOG_WARN("force_ref_timestep_zero is only supported by Krea2 architecture for now");
        }
        if ((params.rescale_ref_ids || params.center_ref_ids) && !sd_version_is_krea2(version)) {
            LOG_WARN("rescale_ref_ids/center_ref_ids are only supported by Krea2 architecture for now");
        }
        if (params.rescale_ref_ids && params.center_ref_ids) {
            LOG_WARN("rescale_ref_ids and center_ref_ids are mutually exclusive; centring wins");
        }
        if (params.native_ref_geometry && (params.resize_vae_to_target || params.crop_vae_to_target_ar)) {
            // Both halves of the restage recipe have to agree. Decoding the reference natively and
            // then resizing/cropping it to the target here just re-imposes the geometry the native
            // decode existed to avoid — the markers are thrown away either way, and the only
            // symptom is a quietly wrong picture.
            LOG_WARN("native_ref is set but resize_vae_to_target/crop_vae_to_target_ar will re-impose the target geometry");
        }
        return params;
    }
};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

bool sd_ref_image_args_want_native_geometry(const char* ref_image_args) {
    // Deliberately NOT resolve_ref_image_params(): that is a method on the context, needs the
    // loaded model's version to pick a default preset, and runs long after the reference bytes
    // have been decoded. The caller has to decide this before it decodes anything, so it needs a
    // free function — but it must still read the SAME preset table, or the two halves of one
    // request could disagree about whether the reference was cropped.
    //
    // No version is needed here because every version-default preset leaves this off; only an
    // explicit preset or an explicit native_ref= can turn it on.
    bool native = false;
    for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
        if (key == "preset") {
            auto it = REF_IMAGE_PRESETS.find(value);
            if (it != REF_IMAGE_PRESETS.end()) {
                native = it->second.native_ref_geometry;
            }
            break;
        }
    }
    // An explicit native_ref= wins over the preset, in both directions: it is how you restage
    // with the edit preset, or ask the restage preset for a reference already at target geometry.
    for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
        if (key == "native_ref") {
            if (!parse_strict_bool(value, native)) {
                LOG_WARN("ignoring invalid reference image arg 'native_ref=%s'", value.c_str());
            }
        }
    }
    return native;
}

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
    "dpm++2m_sde",
    "dpm++2m_sde_bt",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

// Wire-compatibility aliases for sampler names upstream has renamed. Callers (koblem,
// saved Director projects, stored presets) still send the older spelling, and an
// unrecognised name silently degrades to the default sampler rather than erroring — so a
// rename is a quiet output change, not a loud failure. Keep old names working forever.
static const struct {
    const char* alias;
    const char* canonical;
} sample_method_aliases[] = {
    // upstream renamed euler_ancestral_cfg_pp -> euler_a_cfg_pp during the rebuild
    {"euler_ancestral_cfg_pp", "euler_a_cfg_pp"},
    {"euler_ancestral", "euler_a"},
};

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    for (const auto& a : sample_method_aliases) {
        if (!strcmp(str, a.alias)) {
            for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
                if (!strcmp(a.canonical, sample_method_to_str[i])) {
                    return (enum sample_method_t)i;
                }
            }
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
    "logit_normal",
    "flux2",
    "flux",
    "beta",
};

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    if (!strcmp(str, "normal")) {
        return DISCRETE_SCHEDULER;
    }
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
    "sefi_flow",
    "minit2i_flow",
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
        case SD_VAE_FORMAT_WAN:
            return "wan";
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
        case SD_VAE_FORMAT_WAN:
            return VERSION_WAN2;
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
    hires_params->sample_method       = SAMPLE_METHOD_COUNT;
    hires_params->cfg                 = NAN;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                      = {};
    sd_ctx_params->n_threads            = sd_get_num_physical_cores();
    sd_ctx_params->wtype                = SD_TYPE_COUNT;
    sd_ctx_params->rng_type             = CUDA_RNG;
    sd_ctx_params->sampler_rng_type     = RNG_TYPE_COUNT;
    sd_ctx_params->prediction           = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode      = LORA_APPLY_AUTO;
    sd_ctx_params->max_vram             = nullptr;
    sd_ctx_params->stream_layers        = false;
    sd_ctx_params->eager_load           = false;
    sd_ctx_params->enable_mmap          = false;
    sd_ctx_params->diffusion_flash_attn = false;
    sd_ctx_params->vae_format           = SD_VAE_FORMAT_AUTO;
    sd_ctx_params->backend              = nullptr;
    sd_ctx_params->params_backend       = nullptr;
    sd_ctx_params->split_mode           = nullptr;
    sd_ctx_params->auto_fit             = false;
    sd_ctx_params->rpc_servers          = nullptr;
    sd_ctx_params->model_args           = nullptr;
    sd_ctx_params->pulid_weights_path   = nullptr;
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
             "pulid_weights_path: %s\n"
             "tensor_type_rules: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "max_vram: %s\n"
             "stream_layers: %s\n"
             "eager_load: %s\n"
             "backend: %s\n"
             "params_backend: %s\n"
             "split_mode: %s\n"
             "model_args: %s\n"
             "auto_fit: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
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
             SAFE_STR(sd_ctx_params->pulid_weights_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             SAFE_STR(sd_ctx_params->max_vram),
             BOOL_STR(sd_ctx_params->stream_layers),
             BOOL_STR(sd_ctx_params->eager_load),
             SAFE_STR(sd_ctx_params->backend),
             SAFE_STR(sd_ctx_params->params_backend),
             SAFE_STR(sd_ctx_params->split_mode),
             SAFE_STR(sd_ctx_params->model_args),
             BOOL_STR(sd_ctx_params->auto_fit),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
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
    sd_img_gen_params->ref_image_args    = "";
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->qwen_image_layers = 3;
    sd_img_gen_params->circular_x        = false;
    sd_img_gen_params->circular_y        = false;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->pulid_params      = {nullptr, 1.0f};
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
             "qwen_image_layers: %d\n"
             "ref_images_count: %d\n"
             "ref_image_args: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s (temporal=%s, extra_tiling_args=%s)\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
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
             sd_img_gen_params->qwen_image_layers,
             sd_img_gen_params->ref_images_count,
             SAFE_STR(sd_img_gen_params->ref_image_args),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.temporal_tiling),
             SAFE_STR(sd_img_gen_params->vae_tiling_params.extra_tiling_args),
             BOOL_STR(sd_img_gen_params->circular_x),
             BOOL_STR(sd_img_gen_params->circular_y),
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
    sd_vid_gen_params->relip_ref_tstride                     = 1;
    sd_vid_gen_params->a2v_guidance                           = 1.f;
    sd_vid_gen_params->a2v_ramp_end                           = 1.f;
    sd_vid_gen_params->lipdub_two_stage                       = false;
    sd_vid_gen_params->v2v_guide_strength                    = 1.f;
    sd_vid_gen_params->v2v_guide_latent_path                 = nullptr;
    sd_vid_gen_params->keyframes                             = nullptr;
    sd_vid_gen_params->keyframe_frame_indices                = nullptr;
    sd_vid_gen_params->keyframes_size                        = 0;
    sd_vid_gen_params->character_refs                        = nullptr;
    sd_vid_gen_params->character_ref_source_ids              = nullptr;
    sd_vid_gen_params->character_refs_size                   = 0;
    // Null counts == every reference in every segment, so an untouched params
    // block keeps the unscoped behaviour.
    sd_vid_gen_params->character_ref_segments                = nullptr;
    sd_vid_gen_params->character_ref_segment_counts          = nullptr;
    // Negative == unsupplied (engine default 1.0). Zero is a MEANINGFUL value: it
    // requests the untagged / JoyAI-Echo overlap layout, so it cannot double as
    // the "not set" sentinel the way it used to.
    sd_vid_gen_params->tass_phase_scale                      = -1.f;
    sd_vid_gen_params->msr_background                        = nullptr;
    sd_vid_gen_params->msr_subjects                          = nullptr;
    sd_vid_gen_params->msr_subjects_size                     = 0;
    sd_vid_gen_params->msr_frames                            = 0;
    sd_vid_gen_params->msr_segments                          = nullptr;
    sd_vid_gen_params->msr_segments_size                     = 0;
    // OFF. The trim is checkpoint-specific (echo-e50 leaks, msr-v2 and echo-full do
    // not), so it is opt-in and an untouched params block renders exactly as before.
    sd_vid_gen_params->reference_head_trim                   = 0;
    sd_vid_gen_params->beats                                 = nullptr;
    sd_vid_gen_params->beat_count                            = 0;
    sd_vid_gen_params->relay_eps                             = 0.f;
    sd_vid_gen_params->relay_audio_eps                       = 0.f;
    sd_vid_gen_params->relay_steps_frac                      = 0.f;
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->fps                                   = 16;
    sd_vid_gen_params->moe_boundary                          = 0.875f;
    sd_vid_gen_params->vace_strength                         = 1.f;
    sd_vid_gen_params->vace_cont_latent                       = nullptr;
    sd_vid_gen_params->vace_cont_latent_width                 = 0;
    sd_vid_gen_params->vace_cont_latent_height                = 0;
    sd_vid_gen_params->vace_cont_latent_frames                = 0;
    sd_vid_gen_params->vace_cont_latent_channels              = 0;
    sd_vid_gen_params->vace_cont_frames                       = 0;
    sd_vid_gen_params->vace_cont_latent_drop_tail             = 0;
    sd_vid_gen_params->cont_latent                            = nullptr;
    sd_vid_gen_params->cont_latent_width                      = 0;
    sd_vid_gen_params->cont_latent_height                     = 0;
    sd_vid_gen_params->cont_latent_frames                     = 0;
    sd_vid_gen_params->cont_latent_channels                   = 0;
    sd_vid_gen_params->end_cont_latent                         = nullptr;
    sd_vid_gen_params->end_cont_latent_width                   = 0;
    sd_vid_gen_params->end_cont_latent_height                  = 0;
    sd_vid_gen_params->end_cont_latent_frames                  = 0;
    sd_vid_gen_params->end_cont_latent_channels                = 0;
    sd_vid_gen_params->audio_path                            = nullptr;
    sd_vid_gen_params->audio_frame_offset                    = 0;
    sd_vid_gen_params->bsa_enabled                           = 0;
    sd_vid_gen_params->bsa_radius                            = 1;
    sd_vid_gen_params->bsa_self_frame                        = 1;
    sd_vid_gen_params->bsa_bookend                           = 0;
    sd_vid_gen_params->bsa_cube_h                            = 4;
    sd_vid_gen_params->bsa_cube_w                            = 6;
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
    sd_vid_gen_params->hires.sample_method                   = SAMPLE_METHOD_COUNT;
    sd_vid_gen_params->hires.cfg                             = NAN;
    sd_vid_gen_params->hires_chain                           = nullptr;
    sd_vid_gen_params->hires_chain_count                     = 0;
    sd_vid_gen_params->emit_stages                           = 0;
    sd_vid_gen_params->on_stage                              = nullptr;
    sd_vid_gen_params->on_stage_user                         = nullptr;
    sd_vid_gen_params->stage_seg_index                       = 0;
    sd_vid_gen_params->circular_x                            = false;
    sd_vid_gen_params->circular_y                            = false;
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

static bool sd_version_supports_video_generation(SDVersion version) {
    return version == VERSION_SVD || sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_lingbot_video(version) || sd_version_is_ltxav(version) || sd_version_is_longcat_avatar(version);
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

SD_API void sd_ctx_free_diffusion_model(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->diffusion_model == nullptr) {
        return;
    }

    // ModelManager owns parameter residency. runner_done() releases the DiT's
    // staged compute and parameter residency through that owner; a later compute
    // prepares the same registered tensors again on demand.
    sd_ctx->sd->diffusion_model->runner_done();
}

SD_API bool sd_ctx_swap_diffusion_model(sd_ctx_t* sd_ctx, const char* diffusion_model_path) {
    return sd_ctx != nullptr && sd_ctx->sd != nullptr && diffusion_model_path != nullptr &&
           sd_ctx->sd->swap_diffusion_model(diffusion_model_path);
}

SD_API bool sd_ctx_apply_loras(sd_ctx_t* sd_ctx, const sd_lora_t* loras, uint32_t lora_count) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    if (lora_count > 0 && loras == nullptr) {
        return false;
    }
    // apply_loras() tolerates an empty set: apply_loras_at_runtime() clears the adapters and
    // returns early, which is exactly how a segment drops an inherited adapter.
    sd_ctx->sd->apply_loras(loras, lora_count);
    return true;
}

SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode) {
    if (sd_ctx && sd_ctx->sd) {
        if (mode < SD_CANCEL_ALL || mode > SD_CANCEL_RESET) {
            mode = SD_CANCEL_ALL;
        }
        sd_ctx->sd->set_cancel_flag(mode);
    }
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
    if (sd_ctx->sd->animatediff_loaded && sd_version_supports_animatediff(sd_ctx->sd->version)) {
        return true;
    }
    return sd_version_supports_video_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || path == nullptr) {
        return false;
    }
    return sd_ctx->sd->load_control_net_from_file(path);
}

SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_ctx->sd->unload_control_net();
}

SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_ctx->sd->control_net != nullptr;
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
    } else if (sd_ctx != nullptr && sd_ctx->sd != nullptr && sd_version_is_flux(sd_ctx->sd->version)) {
        return FLUX_SCHEDULER;
    } else if (sd_ctx != nullptr && sd_ctx->sd != nullptr && sd_version_is_flux2(sd_ctx->sd->version)) {
        return FLUX2_SCHEDULER;
    } else if (sd_ctx != nullptr && sd_ctx->sd != nullptr && sd_version_is_ltxav(sd_ctx->sd->version)) {
        return LTX2_SCHEDULER;
    } else if (sd_ctx != nullptr && sd_ctx->sd != nullptr && sd_version_is_ideogram4(sd_ctx->sd->version)) {
        return LOGIT_NORMAL_SCHEDULER;
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

// Wan2.2 lightx2v DMD distill uses a fixed high/low expert sigma grid rather
// than the generic discrete scheduler.  The server selects it explicitly for
// the distilled GGUFs with WAN_DISTILL_SIGMAS=1; full-step Wan must retain the
// normal scheduler.
// LongCat-Video-Avatar 1.5 DMD distilled sigma schedule.  The folded DMD
// checkpoint is trained on this FlowMatch grid, not the generic discrete grid.
static std::vector<float> build_longcat_dmd_sigmas(int distill_steps, int num_train_timesteps, float shift) {
    std::vector<float> sigmas;
    sigmas.reserve(distill_steps + 1);
    for (int i = distill_steps; i >= 1; --i) {
        long index = std::lround(static_cast<double>(i) * (num_train_timesteps / distill_steps));
        index = num_train_timesteps - index;
        const double raw = static_cast<double>(num_train_timesteps - 1 - index) / (num_train_timesteps - 1);
        const double shifted = static_cast<double>(shift) * raw / (1.0 + (static_cast<double>(shift) - 1.0) * raw);
        sigmas.push_back(static_cast<float>(shifted));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

static std::vector<float> build_wan_distill_sigmas(int n_high, int n_low, int num_train_timesteps, float shift) {
    const int boundary_t = num_train_timesteps / 2;
    n_high = std::max(1, n_high);
    n_low  = std::max(1, n_low);
    std::vector<int> ts;
    for (int i = 0; i < n_high; ++i) {
        ts.push_back(num_train_timesteps - static_cast<int>(std::lround(
            static_cast<double>(i) * static_cast<double>(boundary_t) / static_cast<double>(n_high))));
    }
    ts.push_back(boundary_t);
    int last = boundary_t;
    for (int i = 1; i < n_low; ++i) {
        last = last > 1 ? last / 2 : 1;
        ts.push_back(last);
    }
    std::vector<float> sigmas;
    sigmas.reserve(ts.size() + 1);
    for (int t : ts) {
        const double r = static_cast<double>(t) / static_cast<double>(num_train_timesteps);
        sigmas.push_back(static_cast<float>(static_cast<double>(shift) * r /
                                            (1.0 + (static_cast<double>(shift) - 1.0) * r)));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

static float wan_distill_shift(int /* spatial_seq_len */) {
    if (const char* value = getenv("WAN_DISTILL_SHIFT"); value != nullptr && value[0] != '\0') {
        const float shift = static_cast<float>(atof(value));
        if (shift > 0.0f) return shift;
    }
    return 7.0f;
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
            case DPMPP2M_SDE_SAMPLE_METHOD:
            case DPMPP2M_SDE_BT_SAMPLE_METHOD:
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
    int qwen_image_layers                    = 3;
    int shifted_timestep                     = 0;
    float strength                           = 1.f;
    float control_strength                   = 0.f;
    float eta                                = 0.f;
    sd_guidance_params_t guidance            = {};
    sd_guidance_params_t high_noise_guidance = {};
    sd_pm_params_t pm_params                 = {};
    sd_pulid_params_t pulid_params           = {};
    sd_hires_params_t hires                  = {};
    int frames                               = -1;
    int requested_frames                     = -1;
    int fps                                  = 16;
    float vace_strength                      = 1.f;

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
        prompt                      = SAFE_STR(sd_img_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_img_gen_params->negative_prompt);
        width                       = sd_img_gen_params->width;
        height                      = sd_img_gen_params->height;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_img_gen_params->seed;
        batch_count                 = sd_img_gen_params->batch_count;
        qwen_image_layers           = std::max(0, sd_img_gen_params->qwen_image_layers);
        clip_skip                   = sd_img_gen_params->clip_skip;
        shifted_timestep            = sd_img_gen_params->sample_params.shifted_timestep;
        strength                    = sd_img_gen_params->strength;
        control_strength            = sd_img_gen_params->control_strength;
        eta                         = sd_img_gen_params->sample_params.eta;
        has_ref_images              = sd_img_gen_params->ref_images_count > 0;
        guidance                    = sd_img_gen_params->sample_params.guidance;
        pm_params                   = sd_img_gen_params->pm_params;
        pulid_params                = sd_img_gen_params->pulid_params;
        hires                       = sd_img_gen_params->hires;
        cache_params                = &sd_img_gen_params->cache;
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
        guidance                    = sd_vid_gen_params->sample_params.guidance;
        high_noise_guidance         = sd_vid_gen_params->high_noise_sample_params.guidance;
        hires                       = sd_vid_gen_params->hires;
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

        resolve_guidance(sd_ctx, &guidance, &use_uncond, &use_img_uncond, has_ref_images);
        if (sd_ctx->sd->high_noise_diffusion_model) {
            resolve_guidance(sd_ctx,
                             &high_noise_guidance,
                             &use_high_noise_uncond,
                             &use_high_noise_img_uncond,
                             has_ref_images,
                             "high noise: ");
        }

        if (shifted_timestep > 0 && !sd_version_is_sdxl(sd_ctx->sd->version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
    }

    // Reconcile a CFG++ sampler with the guidance scale actually in force. Must run AFTER
    // SamplePlan has resolved the method, since only then is the sampler final.
    //
    // A CFG++ sampler steps along the UNCOND direction while anchoring on the cfg-combined
    // x0. That differs from the plain ancestral step ONLY through the cond/uncond gap, so it
    // needs a real uncond prediction on EVERY step — a second model evaluation, which the
    // usual cfg == 1 optimisation skips (ComfyUI forces it via disable_cfg1_optimization).
    //
    // At cfg == 1 there is no gap: pred == pred_uncond, and the CFG++ update is algebraically
    // the plain ancestral update. Paying for the uncond pass there buys nothing — measured on
    // the 960x544/145f-schedule i2v repro at 25 frames: sampling 27.2 s -> 56.3 s (+107%) for
    // identical colour metrics (SATAVG drift -2.0% vs -2.1%, latent ramp 0.985 vs 0.986).
    // So at cfg == 1 fall back to the plain sampler and say so; the distilled LTX recipes all
    // run cfg 1, which is why koblem's euler_ancestral_cfg_pp requests land here.
    //
    // Above cfg 1 the gap is real: honour CFG++ and take the uncond pass.
    //
    // LTXAV_CFG_PP_FORCE_UNCOND=1 keeps true CFG++ at cfg == 1 (what the pre-rebuild engine
    // did) for A/B measurement — it is a comparison escape, not a quality setting.
    void reconcile_cfg_pp_sampler(enum sample_method_t* sample_method) {
        GGML_ASSERT(sample_method != nullptr);
        if (*sample_method != EULER_CFG_PP_SAMPLE_METHOD && *sample_method != EULER_A_CFG_PP_SAMPLE_METHOD) {
            return;
        }
        const char* force = getenv("LTXAV_CFG_PP_FORCE_UNCOND");
        const bool forced = force != nullptr && force[0] != '0';
        const bool cfg_is_one = guidance.txt_cfg == 1.f && guidance.img_cfg == 1.f;

        if (cfg_is_one && !forced) {
            enum sample_method_t plain = *sample_method == EULER_A_CFG_PP_SAMPLE_METHOD
                                             ? EULER_A_SAMPLE_METHOD
                                             : EULER_SAMPLE_METHOD;
            LOG_INFO("sampling using %s method (requested %s; at cfg 1.0 CFG++ is algebraically "
                     "the plain step, so the per-step uncond pass would cost ~2x for nothing)",
                     sd_sample_method_name(plain),
                     sd_sample_method_name(*sample_method));
            *sample_method = plain;
            return;
        }
        if (!use_uncond) {
            LOG_INFO("CFG++ sampler: forcing an uncond prediction per step (cfg=%.2f)", guidance.txt_cfg);
        }
        use_uncond = true;
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

        if (sample_params->custom_sigmas_count > 0) {
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

        if (sd_version_is_longcat_avatar(sd_ctx->sd->version) &&
            sample_params->custom_sigmas_count <= 0) {
            const int dmd_steps = (sample_steps > 0 && sample_steps < 8) ? sample_steps : 8;
            sigmas = build_longcat_dmd_sigmas(dmd_steps, 1000, 7.0f);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            sample_steps = total_steps;
            LOG_INFO("LongCat-Avatar DMD distilled schedule: %d steps", total_steps);
        }

        if (sd_version_is_wan(sd_ctx->sd->version) &&
            getenv("WAN_DISTILL_SIGMAS") != nullptr &&
            sample_params->custom_sigmas_count <= 0) {
            const int n_high = high_noise_sample_steps > 0 ? high_noise_sample_steps : 2;
            const int n_low  = sample_steps > 0 ? sample_steps : 2;
            const int seq_len = sd_ctx->sd->get_image_seq_len(request->height, request->width);
            const float shift = wan_distill_shift(seq_len);
            sigmas = build_wan_distill_sigmas(n_high, n_low, 1000, shift);
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

        // The REQUESTED sampler. reconcile_cfg_pp_sampler() may still swap a CFG++ method for
        // its plain equivalent at cfg 1 after this point, and logs the swap when it does.
        LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
        if (high_noise_sample_steps > 0) {
            high_noise_sample_method = resolve_sample_method(sd_ctx,
                                                             high_noise_sample_method);
            high_noise_eta           = resolve_eta(sd_ctx, high_noise_eta, high_noise_sample_method);
            LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);
        }
    }
};

// TASS overlap reference conditioning (LTX-Best-Face-ID identity transfer).
//
// Layout: the generated target grid first, then one block per reference. Every
// reference sits on the target's FRAME-0 temporal grid ("overlap") while keeping
// its OWN spatial extent, so a native-resolution 1536x1024 character sheet can
// condition a 768x448 render without being squashed to the video's bucket.
//
// What separates a reference from the real first frame is not its position but the
// per-token source id: targets are 0 (an exact RoPE no-op) and references are
// 2, 3, 4, ... — one distinct rotary tag per subject, which is also what lets
// several character sheets coexist in one shot.
struct LtxvTassRefGrid {
    int64_t width     = 0;
    int64_t height    = 0;
    int64_t frames    = 0;
    float source_id   = 0.f;
};

struct ImageGenerationLatents {
    sd::Tensor<float> init_latent;
    sd::Tensor<float> concat_latent;
    sd::Tensor<float> img_uncond_concat_latent;
    sd::Tensor<float> audio_latent;
    sd::Tensor<float> video_positions;
    // TASS overlap reference conditioning. `ref_video_x` is the character-sheet
    // latents packed on the frame axis; `video_source_ids` is per-token over
    // (target tokens ++ reference tokens) and is empty when no sheet was given,
    // which keeps the RoPE phase an exact no-op.
    sd::Tensor<float> ref_video_x;
    std::vector<float> video_source_ids;
    // The reference grids that produced the tail of `video_positions`.  Kept so a
    // temporal window can rebuild "tile target grid ++ the same reference block"
    // for its own frame range instead of inheriting the full-length layout.
    std::vector<LtxvTassRefGrid> ref_grids;
    // True when the TASS layout was built over an EMPTY base, i.e. the target part
    // of `video_positions` is exactly the implicit plain-t2v grid.  Only then may a
    // temporal window regenerate the target positions from scratch.
    bool tass_positions_only = false;
    float tass_phase_scale = 1.f;
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
    bool audio_fixed                        = false;
    // Per-element audio denoise mask for gap-fill inpainting. Empty = the ordinary uniform
    // behaviour (all held when audio_fixed, all generated otherwise).
    std::vector<float> audio_gap_mask;
    bool v2v_sdedit                        = false;
    bool relip_twostage                    = false;
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

// ── REFERENCE HEAD-FRAME TRIM ────────────────────────────────────────────────────────────────
//
// The LTX VAE's temporal stride. `build_ltxv_tass_ref_video_positions` places every reference on
// the causal grid with this same scale, which is what makes the contaminated run derivable rather
// than a magic number.
static constexpr int kLtxvTemporalScale = 8;
// A caller cannot ask for an unbounded trim. This is deliberately generous -- it only has to catch
// a units mistake (milliseconds, samples, a whole clip length) before it reaches the frame loop;
// the real bound is the shot's own length, applied in ltxv_resolve_reference_head_trim().
static constexpr int kLtxvMaxReferenceHeadTrim = 512;

// Resolve sd_vid_gen_params_t::reference_head_trim for ONE rendered shot: AUTO derivation,
// self-gating, and the upper bound. Returns the number of PIXEL frames to drop off the head, or
// zero for "leave this shot exactly as it rendered".
//
// The AUTO derivation lives here, and nowhere else, so no caller ever hard-codes 1 + 8*(K-1).
// K is read off the reference grids the encode path ACTUALLY built, not off the request: a shot
// that scoped every reference out has no grids and therefore no trim.
static int ltxv_resolve_reference_head_trim(const sd_vid_gen_params_t* params,
                                            const ImageGenerationLatents& latents,
                                            bool is_ltxav,
                                            int frame_count) {
    if (params == nullptr || params->reference_head_trim == 0 || frame_count <= 0) {
        return 0;
    }
    if (params->reference_head_trim < -1 || params->reference_head_trim > kLtxvMaxReferenceHeadTrim) {
        LOG_WARN("reference_head_trim=%d is out of range [-1, %d]; treating it as OFF",
                 params->reference_head_trim, kLtxvMaxReferenceHeadTrim);
        return 0;
    }
    if (!is_ltxav) {
        LOG_WARN("reference_head_trim is an LTXAV TASS-reference control; ignoring it on this model");
        return 0;
    }
    // No references encoded == nothing sitting at latent frame 0's address == nothing to trim.
    // This is the state a shot that scoped every sheet out arrives in, so it is a silent no-op
    // at DEBUG rather than a warning.
    if (latents.ref_grids.empty()) {
        LOG_DEBUG("reference head trim: no TASS references on this shot; nothing to trim");
        return 0;
    }
    // SELF-GATES. Every one of these already pins pixel frame 0 with real content, and i2v and
    // continuation shots were measured NOT to leak. Trimming them would silently shorten a shot
    // for no reason, so the gate is the engine's job and not the caller's.
    const char* gate = nullptr;
    if (params->init_image.data != nullptr) {
        gate = "an i2v init image pins frame 0";
    } else if (params->cont_latent != nullptr && params->cont_latent_frames > 0) {
        gate = "this is a continuation segment";
    } else if (params->keyframes != nullptr && params->keyframe_frame_indices != nullptr) {
        for (int i = 0; i < params->keyframes_size; ++i) {
            if (params->keyframe_frame_indices[i] == 0) {
                gate = "a keyframe is pinned at frame 0";
                break;
            }
        }
    }
    if (gate != nullptr) {
        LOG_INFO("reference head trim: SELF-GATED to a no-op -- %s", gate);
        return 0;
    }

    int64_t ref_latent_frames = 0;
    for (const auto& grid : latents.ref_grids) {
        ref_latent_frames = std::max(ref_latent_frames, grid.frames);
    }
    // 1 + 8*(K-1), spelled as the position math it comes from: the references occupy latent
    // corners [0, K), and corner K maps to pixel frame max(0, 8K-7).
    const int derived = static_cast<int>(
        ltxv_latent_corner_to_pixel_frame(ref_latent_frames, kLtxvTemporalScale, true));
    int trim = params->reference_head_trim < 0 ? derived : params->reference_head_trim;
    if (trim <= 0) {
        return 0;
    }
    // NEVER trim a shot away. One frame has to survive or the caller gets an empty render out of
    // a successful sample.
    const int max_trim = frame_count - 1;
    if (trim > max_trim) {
        LOG_WARN("reference head trim: %d frame(s) would leave nothing of a %d-frame shot; "
                 "clamping to %d",
                 trim, frame_count, max_trim);
        trim = max_trim;
    }
    if (trim <= 0) {
        return 0;
    }
    LOG_INFO("reference head trim: dropping %d head pixel frame(s) of %d (%s; %d reference(s), "
             "largest is %lld latent frame(s) -> 1 + 8*(K-1) = %d)",
             trim,
             frame_count,
             params->reference_head_trim < 0 ? "AUTO" : "caller-specified",
             static_cast<int>(latents.ref_grids.size()),
             (long long)ref_latent_frames,
             derived);
    return trim;
}

// Drop `trim` pixel frames off the head of a decoded shot AND the matching trim/fps seconds off
// the head of its audio, so the two stay frame-exact.
//
// This is an OUTPUT-side cut, deliberately not routed through the pre-render drive-audio window.
// That contract (`drop_applied == drop_predicted`, see the seam-trim block in
// generate_video_chain) exists because a seam trim is measured AFTER the render while the drive
// audio must be cut BEFORE it. This trim moves no content and is a pure function of the
// references, so cutting both outputs by the same amount is exact and needs no prediction.
static void ltxv_apply_reference_head_trim(sd_image_t** frames,
                                           int* frame_count,
                                           sd_audio_t* audio,
                                           int trim,
                                           int fps) {
    if (trim <= 0 || frames == nullptr || *frames == nullptr || frame_count == nullptr ||
        *frame_count <= trim) {
        return;
    }
    sd_image_t* list = *frames;
    for (int i = 0; i < trim; ++i) {
        free(list[i].data);
    }
    const int kept = *frame_count - trim;
    std::memmove(list, list + trim, static_cast<size_t>(kept) * sizeof(sd_image_t));
    *frame_count = kept;

    if (audio == nullptr || audio->data == nullptr || audio->sample_count == 0 ||
        audio->channels == 0 || audio->sample_rate == 0) {
        return;
    }
    const int safe_fps = std::max(1, fps);
    const uint64_t drop = std::min<uint64_t>(
        audio->sample_count,
        static_cast<uint64_t>(std::llround(static_cast<double>(trim) * audio->sample_rate / safe_fps)));
    if (drop == 0) {
        return;
    }
    const uint64_t remaining = audio->sample_count - drop;
    if (remaining > 0) {
        std::memmove(audio->data,
                     audio->data + static_cast<size_t>(drop) * audio->channels,
                     static_cast<size_t>(remaining) * audio->channels * sizeof(float));
    }
    audio->sample_count = remaining;
    LOG_INFO("reference head trim: cut %llu audio sample(s) (%.3fs) off the head to match the "
             "%d trimmed frame(s)",
             (unsigned long long)drop,
             static_cast<double>(drop) / audio->sample_rate,
             trim);
}

// [SEAM] An appended guide block's temporal grid MUST match the target's, or the guide claims
// coordinates for content that lives somewhere else on the timeline.
//
// The target loop maps latent corner t through ltxv_latent_corner_to_pixel_frame(), which under
// causal_temporal_positioning is max(0, 8t-7): frame 0 -> [0,1), frame 1 -> [1,9), frame 2 ->
// [9,17). A UNIFORM stride-8 grid instead gives frame 1 -> [8,16). So a CONTINUATION guide — a
// K-latent-frame video tail depicting exactly the target's overlap frames — was placed up to
// temporal_scale-1 = 7 pixel frames LATER than the content it depicts, which is the "guide holds
// ~3 frames then falls off a cliff" seam at a segment crossover.
//
// pixel_frames == 1 is a genuine single-INSTANT image pin (i2v / Director keyframe): one pixel
// frame at its own index, no causal span. That convention is unchanged.
//
// LTXAV_GUIDE_CAUSAL_POS=0 restores the historical uniform grid for A/B.
static bool ltxav_guide_causal_positions() {
    const char* e = std::getenv("LTXAV_GUIDE_CAUSAL_POS");
    return e == nullptr || e[0] == '\0' || std::string(e) != "0";
}

static void ltxv_guide_frame_span(int frame_idx,
                                  int64_t t,
                                  int temporal_scale,
                                  int pixel_frames,
                                  bool causal_temporal_positioning,
                                  bool guide_causal,
                                  float* t_start,
                                  float* t_end) {
    if (pixel_frames == 1) {
        *t_start = static_cast<float>(frame_idx + t * temporal_scale);
        *t_end   = *t_start + 1.f;
        return;
    }
    if (guide_causal) {
        *t_start = frame_idx + ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning);
        *t_end   = frame_idx + ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning);
    } else {
        *t_start = static_cast<float>(frame_idx + t * temporal_scale);
        *t_end   = static_cast<float>(frame_idx + (t + 1) * temporal_scale);
    }
}

// ---------------------------------------------------------------------------
// Prompt Relay plan assembly (arXiv 2604.10030; see model/diffusion/ltx_relay.hpp)
// ---------------------------------------------------------------------------

static float ltx_relay_env_float(const char* name, float fallback) {
    if (const char* value = std::getenv(name); value != nullptr && value[0] != '\0') {
        return std::strtof(value, nullptr);
    }
    return fallback;
}

// Seconds occupied by one video latent frame at this frame rate. The paper's
// ablated-best flat top is L minus two latent frames, so this is the unit the
// default crossfade is measured in.
static float ltx_relay_latent_frame_seconds(int fps, int temporal_scale) {
    const int safe_fps = std::max(1, fps);
    return static_cast<float>(temporal_scale) / static_cast<float>(safe_fps);
}

// Beats carry a time, not a span. Derive spans by Voronoi over the rendered
// segment: beat s owns the interval to the midpoints of its neighbours, the
// first beat reaches back to frame zero and the last forward to the final
// frame. L is the larger half of that cell, which is the permissive reading
// (the penalty reaches eps at the far edge rather than before it).
static bool build_ltx_relay_plan(const sd_vid_gen_params_t* params,
                                 const std::vector<int32_t>& token_pieces,
                                 int fps,
                                 int rendered_frames,
                                 int temporal_scale,
                                 sd::ltx_relay::Plan* plan) {
    if (params == nullptr || plan == nullptr || params->beats == nullptr || params->beat_count < 1) {
        return false;
    }
    if (token_pieces.empty()) {
        LOG_WARN("LTX prompt relay: no token->piece map from the text encoder, relay disabled");
        return false;
    }
    if (params->prompt == nullptr || params->prompt[0] == '\0') {
        // The zero-penalty global tokens are the only guarantee that a query
        // token far from every beat keeps an unpenalised key. Without them the
        // masked softmax is degenerate, so this is a hard requirement.
        LOG_WARN("LTX prompt relay: beats require a non-empty global prompt, relay disabled");
        return false;
    }

    const int safe_fps      = std::max(1, fps);
    const float latent_secs = ltx_relay_latent_frame_seconds(safe_fps, temporal_scale);
    const float clip_secs   = static_cast<float>(std::max(1, rendered_frames) - 1) / static_cast<float>(safe_fps);

    struct Sorted {
        int index;
        float time;
    };
    std::vector<Sorted> sorted;
    sorted.reserve(static_cast<size_t>(params->beat_count));
    for (int beat = 0; beat < params->beat_count; ++beat) {
        const float time = std::clamp(static_cast<float>(params->beats[beat].frame) / static_cast<float>(safe_fps),
                                      0.f,
                                      clip_secs);
        sorted.push_back({beat, time});
    }
    std::stable_sort(sorted.begin(), sorted.end(), [](const Sorted& a, const Sorted& b) { return a.time < b.time; });

    const float env_window   = ltx_relay_env_float("LTX_RELAY_W", -1.f);
    const float env_strength = ltx_relay_env_float("LTX_RELAY_STRENGTH", -1.f);

    // Beats stay in the CALLER's order: the plan's beat k must line up with
    // prompt piece k+1, which the conditioner encoded in that order.
    plan->beats.assign(static_cast<size_t>(params->beat_count), sd::ltx_relay::Beat{});
    for (size_t position = 0; position < sorted.size(); ++position) {
        const int beat_index = sorted[position].index;
        const float time     = sorted[position].time;
        const float lo       = position == 0 ? 0.f : 0.5f * (sorted[position - 1].time + time);
        const float hi       = position + 1 == sorted.size() ? clip_secs : 0.5f * (time + sorted[position + 1].time);
        // The NEARER cell edge, not the further one.
        //
        // This was max() -- the "permissive reading", so the penalty only reached eps at
        // the far edge. It silently disables the mask whenever a beat sits off-centre in
        // CAP the beat's reach. This one line is what makes Prompt Relay work.
        //
        // eps DEFINES the penalty at distance L -- that is the paper's construction -- so
        // the strongest suppression available anywhere INSIDE a beat's own cell is eps
        // itself, 100x at the default 0.01. That is nowhere near enough to stop the model
        // establishing an object. Suppression only becomes total OUTSIDE the cell, where
        // the quadratic runs past the floor toward max_cost.
        //
        // A lone beat owns the whole clip under Voronoi, so uncapped its cell radius is
        // seconds wide and no frame is ever outside it -- the mask is then a formality.
        // GPU-proven 2026-07-28, one beat, one continuous action, same prompt and seed
        // (the model's own default is the object present from frame 0):
        //
        //   beat 4.0s, L=1.0s (t=0 sits 4L away) -> object arrives ON the beat
        //   beat 4.0s, L=4.0s (t=0 exactly 1L)   -> object present from frame 0
        //
        // Evenly-spaced beats hid this for months: their cells are already ~1s, so the cap
        // barely binds and the defaults looked fine right up until a shot carried one beat.
        //
        // KNOWN BOUND: a beat before ~3s of a 5s shot may still not hold, even at total
        // suppression -- the model cannot be stopped from establishing something the scene
        // needs early. That is not a mask problem; do not chase it with strength or eps.
        const float max_span = ltx_relay_env_float("LTX_RELAY_MAX_L", 1.0f);
        // Keep max() for the half-cell. min() was tried and is a REGRESSION: it narrows
        // multi-beat windows 4x (0.33s -> 0.08s on a 3-beat shot) and the model then fails
        // to express the beat at all -- measured, an "eyes glow red" beat stopped firing
        // entirely. The cap alone fixes the lone-beat case without disturbing the rest.
        const float span     = std::clamp(std::max({time - lo, hi - time, latent_secs}),
                                      latent_secs,
                                      std::max(latent_secs, max_span));

        const float requested_window = env_window >= 0.f ? env_window : params->beats[beat_index].window;
        const float window           = requested_window >= 0.f
                                           ? std::min(requested_window, span - 0.05f * latent_secs)
                                           : std::max(0.f, span - 2.f * latent_secs);
        const float requested_strength = env_strength > 0.f ? env_strength : params->beats[beat_index].strength;

        auto& beat     = plan->beats[static_cast<size_t>(beat_index)];
        beat.mid       = time;
        beat.half_span = span;
        beat.window    = std::max(0.f, window);
        beat.strength  = requested_strength > 0.f ? requested_strength : 1.f;
    }

    plan->token_beat = token_pieces;
    plan->eps        = ltx_relay_env_float("LTX_RELAY_EPS",
                                    params->relay_eps > 0.f ? params->relay_eps : 0.01f);
    plan->audio_eps  = ltx_relay_env_float("LTX_RELAY_AUDIO_EPS",
                                          params->relay_audio_eps != 0.f ? params->relay_audio_eps : plan->eps);
    plan->max_cost   = ltx_relay_env_float("LTX_RELAY_MAX_COST", 60.f);
    // LTX_RELAY_SIGMA_LF selects the reference implementation's sigma: a constant
    // in LATENT FRAMES rather than the paper's cell-scaled one. 0.1448 is what
    // WhatDreamsCost-ComfyUI/prompt_relay.py renders with at its default eps.
    const float sigma_lf = ltx_relay_env_float("LTX_RELAY_SIGMA_LF", 0.f);
    plan->sigma_fixed    = sigma_lf > 0.f ? sigma_lf * latent_secs : 0.f;
    // Set from the mask's own content, per window, in sample_base_window: a
    // counter here would alias across shots in a chain.
    plan->revision = 0;

    int32_t mapped_beats = 0;
    for (int32_t piece : plan->token_beat) {
        mapped_beats += static_cast<int32_t>(piece >= 0);
    }
    // Both denominators, because they differ by ~30x and only the second one is
    // the relay's actual authority. The connector tops the prompt up to
    // kConnectorTargetLen keys with learnable registers, and a register key sits
    // past token_beat -- permanently unpenalised. Reporting the share of prompt
    // tokens alone reads as ~45% when the share of cross-attention keys is ~1%.
    const size_t key_count = std::max<size_t>(static_cast<size_t>(LTXV::kConnectorTargetLen),
                                              plan->token_beat.size());
    // With piece isolation the registers are beat-owned too, so the key figure is
    // not the token figure. Keep both: the first says how much of the PROMPT is
    // beat text, the second how much of the mask can actually act.
    int32_t mapped_keys = mapped_beats;
    if (sd::ltx_relay::isolate_enabled()) {
        std::vector<int32_t> key_beat;
        sd::ltx_relay::build_connector_key_beat(plan->token_beat,
                                                plan->beats.size(),
                                                static_cast<int64_t>(key_count),
                                                key_beat);
        mapped_keys = 0;
        for (int32_t piece : key_beat) {
            mapped_keys += static_cast<int32_t>(piece >= 0);
        }
    }
    LOG_INFO("LTX prompt relay: %d beat(s) over %.2fs, eps=%.4g audio_eps=%.4g%s%s, %d/%zu prompt tokens "
             "beat-owned = %d/%zu cross-attention keys addressable (%.1f%%)",
             params->beat_count,
             clip_secs,
             plan->eps,
             plan->audio_eps,
             plan->sigma_fixed > 0.f ? " [fixed sigma]" : "",
             sd::ltx_relay::isolate_enabled() ? " [piece isolation]" : "",
             mapped_beats,
             plan->token_beat.size(),
             mapped_keys,
             key_count,
             100.0 * static_cast<double>(mapped_keys) / static_cast<double>(std::max<size_t>(1, key_count)));
    for (size_t beat = 0; beat < plan->beats.size(); ++beat) {
        LOG_INFO("  beat %zu: mid=%.3fs L=%.3fs w=%.3fs sigma=%.4fs strength=%.2f",
                 beat,
                 plan->beats[beat].mid,
                 plan->beats[beat].half_span,
                 plan->beats[beat].window,
                 sd::ltx_relay::beat_sigma(plan->beats[beat], plan->eps, plan->sigma_fixed),
                 plan->beats[beat].strength);
    }
    return true;
}

// Per-latent-frame query times, in seconds, for one sampler grid. When the
// caller already built an explicit RoPE position table (i2v, keyframes,
// continuation, windowed tiles) that table is authoritative -- it is what the
// model itself uses for time, appended reference tokens included.
static std::vector<float> ltx_relay_video_frame_times(const sd::Tensor<float>& video_positions,
                                                      int64_t latent_width,
                                                      int64_t latent_height,
                                                      int64_t latent_frames,
                                                      int64_t latent_frame_offset,
                                                      int fps,
                                                      int temporal_scale) {
    std::vector<float> times(static_cast<size_t>(std::max<int64_t>(0, latent_frames)), 0.f);
    const int safe_fps         = std::max(1, fps);
    const int64_t tokens_frame = std::max<int64_t>(1, latent_width * latent_height);
    const bool from_positions  = !video_positions.empty() &&
                                video_positions.dim() == 4 &&
                                video_positions.shape()[2] >= latent_frames * tokens_frame;
    for (int64_t frame = 0; frame < latent_frames; ++frame) {
        times[static_cast<size_t>(frame)] =
            from_positions
                ? video_positions.index(0, 0, frame * tokens_frame, 0)
                : ltxv_latent_corner_to_pixel_frame(latent_frame_offset + frame, temporal_scale, true) /
                      static_cast<float>(safe_fps);
    }
    return times;
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

// Positions for a temporal sub-window on the original video timeline.  The
// target frames are sampled in tiles, but their RoPE locations must remain
// absolute so a frozen overlap retains its original temporal coordinates.
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

    const bool guide_causal = ltxav_guide_causal_positions();
    for (int64_t t = 0; t < keyframe_latent_frames; t++) {
        float t_start = 0.f;
        float t_end   = 0.f;
        ltxv_guide_frame_span(keyframe_frame_idx, t, temporal_scale, keyframe_pixel_frames,
                              causal_temporal_positioning, guide_causal, &t_start, &t_end);
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

// LTX-2.3 LipDub/relip positions.  The generated target grid is followed by
// reference frames at the *same* temporal locations, rather than keyframe
// positions at the end of the clip.  This is the IC-LoRA conditioning layout
// used by the production edit model.
static sd::Tensor<float> build_ltxv_relip_video_positions(int64_t width,
                                                           int64_t height,
                                                           int64_t target_latent_frames,
                                                           int64_t reference_latent_frames,
                                                           int fps,
                                                           int spatial_scale,
                                                           int temporal_scale,
                                                           int reference_temporal_stride = 1) {
    GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0 && reference_latent_frames > 0 && fps > 0);
    reference_temporal_stride = std::max(1, reference_temporal_stride);
    sd::Tensor<float> positions({2, 3, width * height * (target_latent_frames + reference_latent_frames), 1});
    int64_t token = 0;
    auto append = [&](int64_t frames, int temporal_stride) {
        for (int64_t t = 0; t < frames; ++t) {
            const int64_t source_t = t * temporal_stride;
            const float t_start = ltxv_latent_corner_to_pixel_frame(source_t, temporal_scale, true) / static_cast<float>(fps);
            const float t_end = ltxv_latent_corner_to_pixel_frame(source_t + 1, temporal_scale, true) / static_cast<float>(fps);
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    set_ltxv_video_position(&positions, token++, t_start, t_end,
                                             static_cast<float>(h * spatial_scale), static_cast<float>((h + 1) * spatial_scale),
                                             static_cast<float>(w * spatial_scale), static_cast<float>((w + 1) * spatial_scale));
                }
            }
        }
    };
    append(target_latent_frames, 1);
    append(reference_latent_frames, reference_temporal_stride);
    return positions;
}

// `base_positions` is whatever position layout the shot already committed to
// (keyframe, relip, end-image, ...).  It is copied through unchanged so TASS
// composes with every existing conditioning path; when it is empty the plain
// t2v grid is generated here, byte-for-byte the same coordinates the implicit
// `build_video_rope_matrix` path would have produced.
//
// The frame-0 artifact: a reference sharing target latent frame 0's RoPE address is decoded into
// frame 0. SEVEN lever classes were tried and REMOVED, all failing the same way -- they reduced the
// leak only by reducing the conditioning, because on echo-e50 the reference's pull on frame 0 and
// its hold on identity are the same attention:
//   * 9-frame clip encode (content)         -- no effect at all
//   * rescaling the reference (extent)      -- no effect, and CANNOT work: these are absolute pixel
//                                              coords from a shared origin, so a bigger reference
//                                              adds tokens at the SAME coords, it does not move them
//   * reference ORDER                       -- no effect; with a plate present the plate always wins
//   * temporal placement prefix/suffix/gap  -- relocates the leak (suffix dissolves the tail,
//                                              prefix opens on the reference's own scene)
//   * spatial coord scale (upstream's       -- removes the leak by DISABLING the reference: at 1.2
//     memory_downscale_factor)                 and above the render collapses to prompt-only
//   * uniform attention strength            -- clears frame 0 but changes the character
//   * frame-0-only attention suppression    -- hard: frame 0 renders garbage; half: works but the
//                                              composition shifts, since frame 0 seeds it
// The reference is ALREADY "unrenderable but referrable": pinned at timestep zero and sliced off
// before decode. The leak is frame 0's own token being denoised toward it, which is the mechanism
// i2v relies on deliberately. Do not re-try an address- or weight-based fix without a
// NO-REFERENCE arm alongside -- that control is what exposed the spatial one as a disable.

// `ref_frame_origin` is the LATENT FRAME the references are placed RELATIVE TO.  It is
// 0 for a full-length pass and the tile start under temporal windowing, so a tile keeps
// the same relationship to its own frames that a full pass has to the whole shot:
// pinning the sheet at global t=0 would place it seconds in the past for every tile but
// the first — a temporal relationship the checkpoint never saw. The placement mode above
// decides whether the references land ON that frame, before it, or after the target.
static sd::Tensor<float> build_ltxv_tass_ref_video_positions(const sd::Tensor<float>& base_positions,
                                                             int64_t target_width,
                                                             int64_t target_height,
                                                             int64_t target_frames,
                                                             const std::vector<LtxvTassRefGrid>& refs,
                                                             int fps,
                                                             int spatial_scale,
                                                             int temporal_scale,
                                                             std::vector<float>* source_ids_out,
                                                             int64_t ref_frame_origin = 0) {
    GGML_ASSERT(target_width > 0 && target_height > 0 && target_frames > 0 && fps > 0);
    GGML_ASSERT(!refs.empty());

    const bool have_base = !base_positions.empty();
    if (have_base) {
        GGML_ASSERT(base_positions.dim() == 4);
        GGML_ASSERT(base_positions.shape()[0] == 2 && base_positions.shape()[1] == 3);
    }
    const int64_t base_tokens = have_base ? base_positions.shape()[2]
                                          : target_width * target_height * target_frames;

    int64_t total_tokens = base_tokens;
    for (const auto& ref : refs) {
        GGML_ASSERT(ref.width > 0 && ref.height > 0 && ref.frames > 0);
        total_tokens += ref.width * ref.height * ref.frames;
    }

    sd::Tensor<float> positions({2, 3, total_tokens, 1});
    if (source_ids_out != nullptr) {
        source_ids_out->assign(static_cast<size_t>(total_tokens), 0.f);
    }

    int64_t token = 0;
    if (have_base) {
        for (int64_t base_token = 0; base_token < base_tokens; ++base_token, ++token) {
            for (int corner = 0; corner < 2; ++corner) {
                for (int axis = 0; axis < 3; ++axis) {
                    positions.index(corner, axis, token, 0) = base_positions.index(corner, axis, base_token, 0);
                }
            }
        }
    }

    auto append = [&](int64_t w_count, int64_t h_count, int64_t f_count, float source_id,
                      int64_t frame_origin) {
        const float step = static_cast<float>(spatial_scale);
        for (int64_t t = 0; t < f_count; ++t) {
            // `frame_origin` is 0 for a full-length pass and the tile start under temporal
            // windowing; the placement mode decides what it is relative to.
            const float t_start = ltxv_latent_corner_to_pixel_frame(frame_origin + t, temporal_scale, true) / static_cast<float>(fps);
            const float t_end   = ltxv_latent_corner_to_pixel_frame(frame_origin + t + 1, temporal_scale, true) / static_cast<float>(fps);
            for (int64_t h = 0; h < h_count; ++h) {
                for (int64_t w = 0; w < w_count; ++w) {
                    if (source_ids_out != nullptr && source_id != 0.f) {
                        (*source_ids_out)[static_cast<size_t>(token)] = source_id;
                    }
                    set_ltxv_video_position(&positions, token++, t_start, t_end,
                                            static_cast<float>(h) * step, static_cast<float>(h + 1) * step,
                                            static_cast<float>(w) * step, static_cast<float>(w + 1) * step);
                }
            }
        }
    };

    if (!have_base) {
        append(target_width, target_height, target_frames, 0.f, 0);
    }
    for (const auto& ref : refs) {
        // Every reference overlaps the SAME frame. Upstream (JoyAI-Echo `legacy`) spreads them
        // 0..K-1 instead; that was tried and changed nothing about the artifact, so this stays as
        // it shipped rather than carrying an untested divergence.
        append(ref.width, ref.height, ref.frames, ref.source_id, ref_frame_origin);
    }
    GGML_ASSERT(token == total_tokens);
    return positions;
}

// LTXAV multi-keyframe positions: the generated target grid is followed by
// one frozen guide frame per image, placed at that image's pixel-frame index.
// A guide block appended after the target grid. `pixel_frames == 1` is a single-INSTANT image pin
// (Director keyframe / i2v anchor); anything else is real video, where each latent frame spans a
// full temporal_scale window — see ltxv_guide_frame_span().
struct LtxvGuideSpec {
    int frame_idx;
    int latent_frames;
    int pixel_frames;
};

// LTX-2.3 positions for a target block + N heterogeneous appended guide blocks, so a continuation
// motion tail and frozen identity keyframes can coexist on the SAME shot. Generalises the former
// single-keyframe and multi-keyframe builders to per-guide temporal spans: the motion tail needs
// temporal_scale-wide frames on the causal grid, while image keyframes are instants.
//
// The guide blocks are emitted in list order, which MUST match the order their latents were
// concatenated onto init_latent.
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
    for (const auto& guide : guides) {
        guide_frames += guide.latent_frames;
    }
    GGML_ASSERT(guide_frames > 0);

    sd::Tensor<float> positions({2, 3, width * height * (target_latent_frames + guide_frames), 1});
    int64_t token = 0;
    for (int64_t t = 0; t < target_latent_frames; ++t) {
        const float t_start = ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning) /
                              static_cast<float>(fps);
        const float t_end = ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning) /
                            static_cast<float>(fps);
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                set_ltxv_video_position(&positions, token++, t_start, t_end,
                                        static_cast<float>(h * spatial_scale), static_cast<float>((h + 1) * spatial_scale),
                                        static_cast<float>(w * spatial_scale), static_cast<float>((w + 1) * spatial_scale));
            }
        }
    }
    const bool guide_causal = ltxav_guide_causal_positions();
    for (const auto& guide : guides) {
        for (int64_t t = 0; t < guide.latent_frames; ++t) {
            float t_start = 0.f;
            float t_end   = 0.f;
            ltxv_guide_frame_span(guide.frame_idx, t, temporal_scale, guide.pixel_frames,
                                  causal_temporal_positioning, guide_causal, &t_start, &t_end);
            t_start /= static_cast<float>(fps);
            t_end /= static_cast<float>(fps);
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    set_ltxv_video_position(&positions, token++, t_start, t_end,
                                            static_cast<float>(h * spatial_scale), static_cast<float>((h + 1) * spatial_scale),
                                            static_cast<float>(w * spatial_scale), static_cast<float>((w + 1) * spatial_scale));
                }
            }
        }
    }
    return positions;
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

// `audio_mask_elements`, when non-null and non-empty, replaces the uniform audio mask with a
// PER-ELEMENT one — 0 holds that audio latent element at the supplied value, 1 lets the model
// generate it. This is what makes gap-fill inpainting possible: the existing re-injection blend
// (noised = noised*mask + init*(1-mask)) already reads the full packed mask including the audio
// channels, so a per-element block is all the mechanism needs. Entry k corresponds to audio latent
// element k, which the packer lays down at flat offset video_latent.numel() + k.
//
// Elements past the end of the supplied vector (the padding that rounds the audio up to whole
// [W,H,F] channels) keep `audio_mask_value`: they are not real audio and must behave exactly as
// they did before.
static sd::Tensor<float> pack_ltxav_audio_and_video_denoise_mask(const sd::Tensor<float>& video_mask,
                                                                 const sd::Tensor<float>& video_latent,
                                                                 const sd::Tensor<float>& audio_latent,
                                                                 float audio_mask_value = 1.f,
                                                                 const std::vector<float>* audio_mask_elements = nullptr) {
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
    auto audio_mask                       = sd::full<float>(audio_mask_shape, audio_mask_value);
    if (audio_mask_elements != nullptr && !audio_mask_elements->empty()) {
        const int64_t n = std::min<int64_t>(static_cast<int64_t>(audio_mask_elements->size()),
                                            audio_mask.numel());
        std::copy_n(audio_mask_elements->data(), n, audio_mask.data());
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

// Production LipDub/relip: append the source clip as frozen, timeline-aligned
// tokens.  The sampled grid remains the target video; the caller strips this
// appended reference tail after sampling using video_conditioning_frame_count.
static bool apply_ltxav_video_relip_reference(ImageGenerationLatents* latents,
                                               const sd::Tensor<float>& reference,
                                               int fps,
                                               int spatial_scale,
                                               int reference_temporal_stride = 1) {
    if (latents == nullptr || latents->init_latent.empty() || latents->denoise_mask.empty() || reference.empty() ||
        reference.shape()[0] != latents->init_latent.shape()[0] ||
        reference.shape()[1] != latents->init_latent.shape()[1] ||
        reference.shape()[3] != latents->init_latent.shape()[3]) {
        LOG_ERROR("invalid LTXAV relip reference latent shape");
        return false;
    }
    const int64_t reference_frames = reference.shape()[2];
    latents->video_target_frame_count = latents->init_latent.shape()[2];
    latents->video_conditioning_frame_count = reference_frames;
    latents->init_latent = sd::ops::concat(latents->init_latent, reference, 2);
    auto reference_mask = sd::full<float>({reference.shape()[0], reference.shape()[1], reference_frames, 1, 1}, 0.f);
    latents->denoise_mask = sd::ops::concat(latents->denoise_mask, reference_mask, 2);
    latents->video_positions = build_ltxv_relip_video_positions(latents->init_latent.shape()[0],
                                                                  latents->init_latent.shape()[1],
                                                                  latents->video_target_frame_count,
                                                                  reference_frames,
                                                                  fps,
                                                                  spatial_scale,
                                                                  8,
                                                                  reference_temporal_stride);
    return true;
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

// How many pixel frames a TASS reference still is expanded to before its LAST
// latent frame is kept. ONE -- the default -- encodes the still directly, which is
// what has always shipped.
//
// Nine is what JoyAI-Echo's own pipeline does (`utils.py:99-101`,
// `clip_num_frames: 9`, `latent[:, -1:]`; the community node spells it
// `[_ref_pil] * 9`), and the reasoning for adopting it was that encoding ONE image
// yields latent frame 0 -- the causal SEED latent, bit-for-bit what an i2v pin
// writes into that slot -- so the model would be reading the reference as an
// opening-frame pin. A last-of-9 latent encodes a temporal chunk instead.
//
// THAT REASONING IS WRONG, and it is recorded here rather than deleted so nobody
// spends the GPU time again. A/B at 1280x704, seed 9001, echo-e50, two
// render-resolution references, `tass_phase_scale: 0`: BOTH arms open on the empty
// scene with the subject appearing at frame 1. The clip encode measurably changes
// the render -- the two arms are different takes -- but it does not touch the
// artifact, because the artifact is POSITIONAL.
//
// So this stays a knob rather than a default: it is upstream's convention and may
// yet matter for identity fidelity, but it costs 9x the VAE work on a cache miss
// and has no measured benefit, and turning it on changes the conditioning of every
// existing project. The count must be 1 modulo 8, the LTX VAE's temporal stride.
static int ltxav_reference_clip_frames() {
    static const int frames = []() {
        const char* configured = getenv("LTX_REF_CLIP_FRAMES");
        if (configured == nullptr || configured[0] == '\0') {
            return 1;
        }
        const int requested = atoi(configured);
        if (requested < 1 || requested % 8 != 1) {
            LOG_WARN("LTX_REF_CLIP_FRAMES=%s is not 1 modulo 8; using 1", configured);
            return 1;
        }
        return requested;
    }();
    return frames;
}

// Encode one reference still the way the checkpoint was trained to read it: as a
// short clip of that image, keeping only the trailing latent frame. See
// `ltxav_reference_clip_frames` for why a single-frame encode is the wrong tensor
// rather than merely a cheaper one.
static sd::Tensor<float> encode_ltxav_reference_image(sd_ctx_t* sd_ctx,
                                                      const sd::Tensor<float>& image,
                                                      const char* name) {
    const int clip_frames = ltxav_reference_clip_frames();
    if (clip_frames <= 1) {
        return encode_ltxav_condition_image(sd_ctx, image, name);
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image.empty()) {
        return {};
    }
    // Replicate on the frame axis. The VAE sees a still clip, which is what the
    // reference implementation hands it -- not a still.
    auto frame = image.reshape({image.shape()[0], image.shape()[1], 1, image.shape()[2], image.shape()[3]});
    sd::Tensor<float> clip({image.shape()[0], image.shape()[1], clip_frames, image.shape()[2], image.shape()[3]});
    for (int f = 0; f < clip_frames; ++f) {
        sd::ops::slice_assign(&clip, 2, f, f + 1, frame);
    }
    auto clip_latent = sd_ctx->sd->encode_first_stage(clip);
    if (clip_latent.empty() || clip_latent.dim() < 4) {
        LOG_ERROR("failed to encode LTXAV %s reference clip", name);
        return {};
    }
    const int64_t latent_frames = clip_latent.shape()[2];
    if (latent_frames <= 1) {
        // A VAE with no temporal compression collapses the clip back to a seed
        // latent; nothing is gained and nothing is lost by saying so out loud.
        LOG_WARN("LTXAV %s reference clip encoded to %lld latent frame(s); keeping it as-is",
                 name,
                 (long long)latent_frames);
        return clip_latent;
    }
    return sd::ops::slice(clip_latent, 2, latent_frames - 1, latent_frames);
}

// ── content-addressed cache for encoded reference latents ────────────────────
//
// A character reference is an IDENTITY, not a shot: the same pixels, encoded to
// the same latent, for every segment of a chain and for every chain that names
// that character. Today it is re-encoded per segment, inside the GPU mutex, so a
// ten-shot chain pays the VAE ten times for one unchanging face -- and pays it
// again on the next render of the same project. The clip encode above makes that
// worse, since it now feeds the VAE nine frames rather than one.
//
// Both problems have the same answer: key the latent on its input and keep it.
//
// The key is taken over the DECODED, ALREADY-RESIZED pixel tensor -- the exact
// thing handed to the VAE -- so resolution and channel layout are part of the key
// by construction rather than by remembering to include them. Two independent
// 64-bit FNV-1a passes give a 128-bit key; a collision would silently swap one
// person's face for another's, which is worth eight extra bytes to make absurd.
//
// Empty cache dir disables the whole thing and restores the encode-every-time
// behaviour exactly. `LTX_REF_LATENT_CACHE_TAG` joins the key so a deliberate VAE
// change can invalidate every entry without finding them.
static std::string ltxav_reference_cache_dir() {
    static const std::string dir = []() {
        const char* configured = getenv("LTX_REF_LATENT_CACHE_DIR");
        return configured != nullptr ? std::string(configured) : std::string();
    }();
    return dir;
}

static std::string ltxav_reference_cache_key(const sd::Tensor<float>& input, const char* kind, int variant) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(input.data());
    const size_t size = static_cast<size_t>(input.numel()) * sizeof(float);
    uint64_t h1 = 1469598103934665603ull;   // FNV-1a 64
    uint64_t h2 = 0x9e3779b97f4a7c15ull;    // a second basis, so the pair is 128 bits
    for (size_t i = 0; i < size; ++i) {
        h1 = (h1 ^ bytes[i]) * 1099511628211ull;
        h2 = (h2 ^ bytes[i]) * 0x100000001b3ull;
        h2 ^= h2 >> 29;
    }
    // SHAPE goes in the key explicitly. "Resolution is in the key by construction because it is in
    // the bytes" is FALSE for a transposed pair: a uniform-content reference at [W,H] and [H,W] has
    // the same bytes, the same numel and the same rank, so it would collide and hand back a latent
    // on the wrong spatial grid.
    std::string dims;
    for (int64_t axis = 0; axis < input.dim(); ++axis) {
        dims += (axis ? "x" : "") + std::to_string(input.shape()[axis]);
    }
    char key[256];
    const char* tag = getenv("LTX_REF_LATENT_CACHE_TAG");
    snprintf(key, sizeof(key), "%s-%016llx%016llx-v%d-s%s-%s",
             kind,
             (unsigned long long)h1,
             (unsigned long long)h2,
             variant,
             dims.c_str(),
             tag != nullptr && tag[0] != '\0' ? tag : "v1");
    return key;
}

// Run `encode` on `input`, reusing a previously written latent when these exact
// bytes have been through here before. Falls back to encoding on ANY cache
// trouble: a cache is an optimisation, and a render must never fail because of one.
template <typename Encoder>
static sd::Tensor<float> ltxav_encode_with_reference_cache(const sd::Tensor<float>& input,
                                                           const char* kind,
                                                           int variant,
                                                           const char* name,
                                                           Encoder&& encode) {
    const std::string dir = ltxav_reference_cache_dir();
    if (dir.empty() || input.empty()) {
        return encode();
    }
    std::string path;
    try {
        std::error_code error;
        std::filesystem::create_directories(dir, error);
        if (error) {
            LOG_WARN("LTXAV reference latent cache dir %s is unusable: %s", dir.c_str(), error.message().c_str());
            return encode();
        }
        path = (std::filesystem::path(dir) / (ltxav_reference_cache_key(input, kind, variant) + ".bin")).string();
        if (std::filesystem::is_regular_file(path, error)) {
            auto cached = sd::load_tensor_from_file_as_tensor<float>(path);
            if (!cached.empty() && cached.dim() >= 4) {
                LOG_INFO("LTXAV %s reference: cache HIT %lldx%lldx%lld latent (no VAE pass)",
                         name,
                         (long long)cached.shape()[0],
                         (long long)cached.shape()[1],
                         (long long)cached.shape()[2]);
                return cached;
            }
            LOG_WARN("LTXAV reference latent cache entry %s is unusable; re-encoding", path.c_str());
        }
    } catch (const std::exception& exception) {
        LOG_WARN("LTXAV reference latent cache read failed (%s); re-encoding", exception.what());
        return encode();
    }

    auto encoded = encode();
    if (encoded.empty() || path.empty()) {
        return encoded;
    }
    // Write to a unique temporary and rename. Two workers can encode the same
    // reference concurrently; rename is atomic, so the loser overwrites the winner
    // with byte-identical content instead of a reader seeing a half-written latent.
    try {
        // A stack address is NOT unique across threads -- two workers at the same call depth share
        // it. Use the thread id plus a monotonic counter so concurrent encodes cannot collide on
        // the temporary and truncate each other's write.
        static std::atomic<uint64_t> temp_seq{0};
        std::ostringstream temp_name;
        temp_name << path << ".tmp" << std::hash<std::thread::id>{}(std::this_thread::get_id())
                  << "-" << temp_seq.fetch_add(1);
        const std::string temporary = temp_name.str();
        sd::save_tensor_to_file(temporary, encoded, "ltxav_reference_latent");
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error) {
            std::filesystem::remove(temporary, error);
        }
    } catch (const std::exception& exception) {
        LOG_WARN("LTXAV reference latent cache write failed (%s); the render is unaffected", exception.what());
    }
    return encoded;
}

static sd::Tensor<float> encode_ltxav_reference_image_cached(sd_ctx_t* sd_ctx,
                                                             const sd::Tensor<float>& image,
                                                             const char* name) {
    const int clip_frames = ltxav_reference_clip_frames();
    return ltxav_encode_with_reference_cache(image, "ref", clip_frames, name, [&]() {
        return encode_ltxav_reference_image(sd_ctx, image, name);
    });
}

// Fit one image onto a `width` x `height` canvas.
//
// `cover` scales by max() and centre-CROPS the excess -- what a background plate
// wants, since it must fill the frame. Otherwise we scale by min() and centre the
// whole image on a WHITE canvas, never cropping it: that letterbox is the trained
// convention for an MSR subject, not a presentation choice, so the padding value
// matters as much as the geometry.
//
// Both modes are the same sampling loop. The source-space offset is positive when
// covering (we start inside the image) and negative when containing (we start
// outside it), and every destination pixel that lands outside the resized source
// is white by construction.
static sd::Tensor<float> ltxav_fit_image_to_canvas(const sd_image_t& image,
                                                   int width,
                                                   int height,
                                                   bool cover,
                                                   const char* name) {
    if (image.data == nullptr || image.width == 0 || image.height == 0) {
        LOG_ERROR("LTXAV MSR %s has no image data", name);
        return {};
    }
    auto source = sd_image_to_tensor(image);
    if (source.empty() || source.dim() != 4) {
        LOG_ERROR("failed to read LTXAV MSR %s", name);
        return {};
    }
    int64_t source_width  = source.shape()[0];
    int64_t source_height = source.shape()[1];
    const int64_t channels = source.shape()[2];

    // COVER CROPS IN SOURCE SPACE FIRST. Scaling the whole source and cropping afterwards
    // makes the intermediate grow with the SOURCE's aspect ratio, which is request-controlled
    // and unbounded: a 1x10000 PNG is ~1 KB on the wire and asks for a 1280 x 12,800,000
    // float tensor -- ~196 GB. That allocation throws, and the async job worker is a
    // std::thread entry point with no catch, so it would take the whole engine process down
    // rather than failing one render. Cropping first bounds every intermediate to the canvas.
    if (cover) {
        const double target_aspect = static_cast<double>(width) / static_cast<double>(height);
        const double source_aspect = static_cast<double>(source_width) / static_cast<double>(source_height);
        int64_t crop_w = source_width;
        int64_t crop_h = source_height;
        if (source_aspect > target_aspect) {
            crop_w = std::max<int64_t>(1, static_cast<int64_t>(std::llround(static_cast<double>(source_height) * target_aspect)));
        } else if (source_aspect < target_aspect) {
            crop_h = std::max<int64_t>(1, static_cast<int64_t>(std::llround(static_cast<double>(source_width) / target_aspect)));
        }
        if (crop_w != source_width || crop_h != source_height) {
            const int64_t left = (source_width - crop_w) / 2;
            const int64_t top  = (source_height - crop_h) / 2;
            auto cropped = sd::ops::slice(sd::ops::slice(source, 0, left, left + crop_w), 1, top, top + crop_h);
            if (cropped.empty()) {
                LOG_ERROR("failed to crop LTXAV MSR %s", name);
                return {};
            }
            source        = std::move(cropped);
            source_width  = crop_w;
            source_height = crop_h;
        }
    }

    const double scale_x = static_cast<double>(width) / static_cast<double>(source_width);
    const double scale_y = static_cast<double>(height) / static_cast<double>(source_height);
    const double scale   = cover ? std::max(scale_x, scale_y) : std::min(scale_x, scale_y);

    int64_t resized_width  = std::max<int64_t>(1, static_cast<int64_t>(std::llround(scale * static_cast<double>(source_width))));
    int64_t resized_height = std::max<int64_t>(1, static_cast<int64_t>(std::llround(scale * static_cast<double>(source_height))));
    // Rounding can leave a cover one pixel short of the canvas (or a contain one
    // pixel over it), which would show as a white seam on a background plate.
    if (cover) {
        resized_width  = std::max<int64_t>(resized_width, width);
        resized_height = std::max<int64_t>(resized_height, height);
    } else {
        resized_width  = std::min<int64_t>(resized_width, width);
        resized_height = std::min<int64_t>(resized_height, height);
    }
    // Post-crop this can only exceed the canvas by rounding, but the bound is cheap and it
    // is the last thing standing between a malformed reference and a dead worker.
    if (resized_width > width * 4 || resized_height > height * 4) {
        LOG_ERROR("LTXAV MSR %s would resize to %lldx%lld for a %dx%d canvas; refusing",
                  name, (long long)resized_width, (long long)resized_height, width, height);
        return {};
    }

    // Bicubic + antialias matches the plugin's torch fallback; its cv2 path uses
    // INTER_AREA/LANCZOS4. Bilinear without antialias visibly softens a 2848px sheet
    // being pulled down to a 1280px canvas, and the sheet's detail IS the reference.
    auto resized = sd::ops::interpolate(source,
                                        {resized_width, resized_height, channels, 1},
                                        sd::ops::InterpolateMode::Bicubic,
                                        false,
                                        true);
    if (resized.empty()) {
        LOG_ERROR("failed to resize LTXAV MSR %s", name);
        return {};
    }

    const int64_t offset_x = (resized_width - static_cast<int64_t>(width)) / 2;
    const int64_t offset_y = (resized_height - static_cast<int64_t>(height)) / 2;

    auto canvas = sd::zeros<float>({width, height, channels, 1});
    for (int64_t x = 0; x < width; ++x) {
        const int64_t sx = x + offset_x;
        for (int64_t y = 0; y < height; ++y) {
            const int64_t sy      = y + offset_y;
            const bool    covered = sx >= 0 && sx < resized_width && sy >= 0 && sy < resized_height;
            for (int64_t c = 0; c < channels; ++c) {
                canvas.index(x, y, c, 0) = covered ? resized.index(sx, sy, c, 0) : 1.f;
            }
        }
    }
    return canvas;
}

// Composite an MSR (Licon Multiple-Subject-Reference) in-context reference strip.
//
// The strip is a VIDEO, and its layout is a TRAINED convention that
// ComfyUI-Licon-MSR builds before handing it to LTXAddVideoICLoRAGuide:
//
//   * the BACKGROUND covers the canvas and initialises EVERY frame -- it is the
//     substrate the shot happens in, not one slot among many;
//   * each SUBJECT is letterboxed on white and overwrites its own frame window.
//
// Windows are aligned to the temporal VAE's 8x compression so a subject lands on
// whole latent frames: latent slot 0 is pixel frame 0, and slot N >= 1 covers
// pixel frames [1 + (N-1)*8 .. N*8].
//
// With K subjects and L latent slots the subjects share slots 1..L-1. The exactly
// known case is L-1 == K -- one slot each, i.e. `frames == 8*K + 1` -- and that is
// what the 17/25/33 end of the checkpoint's menu is for. When there are spare slots
// they are shared out with the remainder going to the EARLIER subjects, matching the
// plugin's documented "prioritise the first subject" budgeting.
static sd::Tensor<float> build_ltxav_msr_strip(const sd_image_t& background,
                                               const sd_image_t* subjects,
                                               int subjects_size,
                                               int width,
                                               int height,
                                               int frames) {
    if (width <= 0 || height <= 0 || frames <= 0) {
        LOG_ERROR("LTXAV MSR strip needs a positive canvas and frame count");
        return {};
    }
    if (frames % 8 != 1) {
        LOG_ERROR("LTXAV MSR strip frames must be 1 modulo 8 (17/25/33/41/49/57/65), got %d", frames);
        return {};
    }
    const int64_t latent_slots = (frames - 1) / 8 + 1;
    if (subjects_size < 0 || (subjects_size > 0 && subjects == nullptr)) {
        LOG_ERROR("LTXAV MSR subject array is inconsistent");
        return {};
    }
    if (static_cast<int64_t>(subjects_size) > latent_slots - 1) {
        LOG_ERROR("LTXAV MSR strip has %d subjects but only %lld slots at %d frames; use at least %d frames",
                  subjects_size,
                  (long long)(latent_slots - 1),
                  frames,
                  subjects_size * 8 + 1);
        return {};
    }

    auto background_canvas = ltxav_fit_image_to_canvas(background, width, height, true, "background");
    if (background_canvas.empty()) {
        return {};
    }
    const int64_t channels = background_canvas.shape()[2];

    std::vector<sd::Tensor<float>> subject_canvases;
    subject_canvases.reserve(static_cast<size_t>(subjects_size));
    for (int i = 0; i < subjects_size; ++i) {
        const std::string name = "subject " + std::to_string(i + 1);
        auto canvas            = ltxav_fit_image_to_canvas(subjects[i], width, height, false, name.c_str());
        if (canvas.empty()) {
            return {};
        }
        subject_canvases.push_back(std::move(canvas));
    }

    // Frame ownership, ported from ComfyUI-Licon-MSR `_expand_frames` /
    // `_allocate_subject_latent_counts` / `_latent_to_frame_range`. Two details here are
    // NOT what you would guess and were read from that source rather than inferred:
    //
    //   * the subject cursor starts at latent 0, so subject 1 owns pixel frame 0 -- the
    //     causal anchor -- and the BACKGROUND is what fills the tail slots, not the head;
    //   * spare slots go to the FIRST subject (up to three) before anyone else gets a
    //     second, rather than being shared out evenly.
    //
    // Background initialises every frame and subjects overwrite their own windows, so any
    // slot nobody claims stays background for free.
    std::vector<int> frame_owner(static_cast<size_t>(frames), -1);
    if (subjects_size > 0) {
        const int64_t budget = std::max<int64_t>(0, latent_slots - 1);
        std::vector<int64_t> counts(static_cast<size_t>(subjects_size), 1);
        int64_t extra = budget - subjects_size;
        if (extra > 0) {
            counts[0] += 1;
            extra -= 1;
        }
        for (int index = 1; extra > 0 && subjects_size > 1;) {
            bool any_short = false;
            for (int i = 1; i < subjects_size; ++i) {
                if (counts[static_cast<size_t>(i)] < 2) any_short = true;
            }
            if (!any_short) break;
            if (counts[static_cast<size_t>(index)] < 2) {
                counts[static_cast<size_t>(index)] += 1;
                extra -= 1;
            }
            index = (index + 1 < subjects_size) ? index + 1 : 1;
        }
        if (extra > 0 && counts[0] < 3) {
            counts[0] += 1;
            extra -= 1;
        }
        for (int index = 0; extra > 0; index = (index + 1) % subjects_size) {
            counts[static_cast<size_t>(index)] += 1;
            extra -= 1;
        }

        int64_t cursor = 0;
        for (int i = 0; i < subjects_size; ++i) {
            const int64_t latent_start = cursor;
            const int64_t latent_end   = cursor + counts[static_cast<size_t>(i)] - 1;
            cursor                     = latent_end + 1;
            const int64_t frame_start  = latent_start <= 0 ? 0 : 1 + (latent_start - 1) * 8;
            const int64_t frame_end    = latent_end <= 0 ? 0 : latent_end * 8;
            for (int64_t f = std::max<int64_t>(0, frame_start);
                 f <= std::min<int64_t>(frames - 1, frame_end); ++f) {
                frame_owner[static_cast<size_t>(f)] = i;
            }
        }
    }

    sd::Tensor<float> strip({width, height, frames, channels, 1});
    for (int frame = 0; frame < frames; ++frame) {
        const int owner = frame_owner[static_cast<size_t>(frame)];
        const sd::Tensor<float>& canvas =
            owner >= 0 ? subject_canvases[static_cast<size_t>(owner)] : background_canvas;
        sd::ops::slice_assign(&strip, 2, frame, frame + 1, canvas.unsqueeze(2));
    }
    LOG_INFO("LTXAV MSR strip: %dx%d, %d frames -> %lld latent slots, %d subject(s) + background",
             width,
             height,
             frames,
             (long long)latent_slots,
             subjects_size);
    return strip;
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

// AUDIO GAP-FILL (inpainting, stage 1).
//
// Build a per-element denoise mask for the audio latent from the supplied clip's own coverage:
// 0 where the clip actually has signal (HOLD — the existing re-injection blend puts the supplied
// audio back every step), 1 where it is silent or absent (GENERATE). Handing that to
// pack_ltxav_audio_and_video_denoise_mask is the whole mechanism; no graph change is involved.
//
// ★ The threshold is in dB, deliberately. A linear "is it zero" test calls quiet room tone a gap
// and the model then invents over the top of real material. Only near-digital-silence should read
// as absent, which is what a low dBFS floor expresses.
//
// ★ Layout. The audio latent is [frequency=16, time=audio_length, channel=8, 1] with
//   flat = channel*16*audio_length + time*16 + frequency
// so a decision made per TIME step has to be written across every frequency AND every channel of
// that step. Getting this wrong does not crash — it masks a diagonal smear through the spectrum
// and sounds like a broken codec rather than like a mis-built mask.
//
// Returns an empty vector when every step is covered (nothing to generate), which the caller
// treats as "no gap-fill", so a fully-supplied clip stays byte-identical to the old path.
static std::vector<float> build_ltxav_audio_gap_mask(const std::vector<float>& wav,
                                                     uint32_t wav_channels,
                                                     int64_t source_tokens,
                                                     int audio_length,
                                                     float silence_db) {
    constexpr int kFrequencyBins = 16;
    constexpr int kChannels      = 8;
    if (wav.empty() || wav_channels == 0 || source_tokens <= 0 || audio_length <= 0) {
        return {};
    }
    const int64_t frames = static_cast<int64_t>(wav.size() / wav_channels);
    if (frames <= 0) {
        return {};
    }
    const double silence_amplitude = std::pow(10.0, silence_db / 20.0);

    // One decision per latent time step, measured over the samples that step covers.
    std::vector<float> per_time(static_cast<size_t>(audio_length), 1.f);   // default: generate
    int64_t covered = 0;
    const int64_t usable = std::min<int64_t>(source_tokens, audio_length);
    for (int64_t t = 0; t < usable; ++t) {
        const int64_t first = frames * t / source_tokens;
        const int64_t last  = std::min<int64_t>(frames, frames * (t + 1) / source_tokens);
        double acc = 0.0;
        int64_t n  = 0;
        for (int64_t f = first; f < last; ++f) {
            for (uint32_t c = 0; c < wav_channels; ++c) {
                const double v = wav[static_cast<size_t>(f) * wav_channels + c];
                acc += v * v;
                ++n;
            }
        }
        const double rms = n > 0 ? std::sqrt(acc / static_cast<double>(n)) : 0.0;
        if (rms > silence_amplitude) {
            per_time[static_cast<size_t>(t)] = 0.f;   // hold: the clip has real signal here
            ++covered;
        }
    }
    if (covered == 0) {
        LOG_WARN("LTX audio gap-fill: the supplied clip is silent everywhere above %.1f dBFS; "
                 "nothing would be held, so gap-fill is skipped",
                 silence_db);
        return {};
    }
    if (covered == audio_length) {
        // Fully covered: there is no gap to fill, so leave the ordinary fixed-audio path alone.
        return {};
    }

    std::vector<float> mask(static_cast<size_t>(kFrequencyBins) * audio_length * kChannels, 1.f);
    for (int64_t channel = 0; channel < kChannels; ++channel) {
        for (int64_t t = 0; t < audio_length; ++t) {
            const float value = per_time[static_cast<size_t>(t)];
            const int64_t base = channel * kFrequencyBins * audio_length + t * kFrequencyBins;
            for (int64_t frequency = 0; frequency < kFrequencyBins; ++frequency) {
                mask[static_cast<size_t>(base + frequency)] = value;
            }
        }
    }
    LOG_INFO("LTX audio gap-fill: holding %lld of %d audio latent steps from the supplied clip, "
             "generating the other %lld (silence floor %.1f dBFS)",
             (long long)covered, audio_length, (long long)(audio_length - covered), silence_db);
    return mask;
}

static sd::Tensor<float> make_ltxav_empty_audio_latent(int audio_length) {
    if (audio_length <= 0) {
        return {};
    }
    constexpr int kLtxavAudioFrequencyBins = 16;
    constexpr int kLtxavAudioChannels      = 8;
    return sd::zeros<float>({kLtxavAudioFrequencyBins, audio_length, kLtxavAudioChannels, 1});
}

// Load a 16 kHz drive WAV, encode it with the optional LTX audio-VAE encoder,
// then lay it out in the joint AV latent's [frequency, time, channel, batch]
// format. The caller holds this latent fixed during video denoising.
// How many channels of the drive WAV reach the LTX audio VAE.
//
// The VAE is a 2-channel model (ltx_audio_vae.hpp asserts audio_channels == 2)
// and its encode() has always accepted a stereo waveform, duplicating channel 0
// when handed mono. The drive loader was the only thing forcing mono, which is
// why delivered LTX audio measured L/R NCC 0.993 — effectively mono — against
// source material at 0.43-0.50.
//
// LTX_DRIVE_AUDIO_CHANNELS=1 restores the old mono downmix on the same binary,
// which is what makes this A/B-able. Reads the VALUE, not merely the presence:
// a presence test would make "=1" enable the very thing it asks to disable, and
// that exact bug has bitten four gates in this tree already.
static int ltx_drive_audio_channels() {
    if (const char* env = std::getenv("LTX_DRIVE_AUDIO_CHANNELS"); env != nullptr && env[0] != '\0') {
        const int want = atoi(env);
        if (want >= 1 && want <= 2) {
            return want;
        }
        LOG_WARN("LTX_DRIVE_AUDIO_CHANNELS='%s' is not 1 or 2; using 2", env);
    }
    return 2;
}

// Gap-fill knobs. LTX_AUDIO_GAP_FILL=1 turns on stage-1 audio inpainting: regions of the supplied
// drive clip that are silent are GENERATED while the rest is held. Off by default — a supplied
// drive clip means "condition on this", and generating over its quiet parts is a different request.
// LTX_AUDIO_GAP_SILENCE_DB moves the floor that decides what counts as absent (default -60 dBFS,
// low enough that genuine room tone reads as signal rather than as a gap).
static bool ltx_audio_gap_fill_enabled() {
    if (const char* env = std::getenv("LTX_AUDIO_GAP_FILL"); env != nullptr && env[0] != '\0') {
        return !(env[0] == '0' && env[1] == '\0');
    }
    return false;
}

static float ltx_audio_gap_silence_db() {
    if (const char* env = std::getenv("LTX_AUDIO_GAP_SILENCE_DB"); env != nullptr && env[0] != '\0') {
        const float value = static_cast<float>(atof(env));
        if (value < 0.f && value > -120.f) {
            return value;
        }
        LOG_WARN("LTX_AUDIO_GAP_SILENCE_DB='%s' is not a sane negative dBFS floor; using -60", env);
    }
    return -60.f;
}

// `gap_mask_out`, when non-null, receives the per-element audio denoise mask described by
// build_ltxav_audio_gap_mask. Empty means "no gaps" and the caller keeps the ordinary held-audio
// behaviour.
static sd::Tensor<float> encode_ltxav_drive_audio(sd_ctx_t* sd_ctx,
                                                  const char* wav_path,
                                                  int audio_length,
                                                  std::vector<float>* gap_mask_out = nullptr,
                                                  bool request_gap_fill = false) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || wav_path == nullptr || wav_path[0] == '\0' || audio_length <= 0) {
        return {};
    }
    auto& audio_vae = sd_ctx->sd->audio_vae_model;
    if (audio_vae == nullptr || !audio_vae->config.has_encoder) {
        LOG_ERROR("LTX drive audio requires an audio VAE GGUF with encoder weights and mel bases");
        return {};
    }

    std::vector<float> wav;
    uint32_t           wav_channels = 0;
    if (!LONGCAT_AUDIO::load_wav_16k(wav_path, wav, wav_channels, ltx_drive_audio_channels()) || wav.empty()) {
        LOG_ERROR("failed to load LTX drive audio WAV: %s", wav_path);
        return {};
    }
    // encode() reads its waveform PLANAR (channel c starts at c*samples), while
    // every WAV buffer in this file is interleaved. De-interleave here rather
    // than teaching the loader a second layout — the interleaved convention is
    // load-bearing elsewhere (see load_wav_full's comment about the historical
    // planar-read bug).
    const int64_t      wav_frames = static_cast<int64_t>(wav.size() / wav_channels);
    std::vector<float> planar(wav.size());
    for (uint32_t c = 0; c < wav_channels; ++c) {
        for (int64_t f = 0; f < wav_frames; ++f) {
            planar[static_cast<size_t>(c) * wav_frames + f] = wav[static_cast<size_t>(f) * wav_channels + c];
        }
    }
    sd::Tensor<float> waveform({wav_frames, static_cast<int64_t>(wav_channels)}, planar);
    const auto encoded = audio_vae->encode(sd_ctx->sd->n_threads, waveform);
    constexpr int kFrequencyBins = 16;
    constexpr int kChannels      = 8;
    if (encoded.empty() || encoded.dim() < 2 || encoded.shape()[0] != kFrequencyBins * kChannels) {
        LOG_ERROR("LTX audio VAE produced an invalid drive latent for %s", wav_path);
        return {};
    }

    auto output               = make_ltxav_empty_audio_latent(audio_length);
    const int64_t source_time = encoded.shape()[1];
    const int64_t copy_time   = std::min(source_time, static_cast<int64_t>(audio_length));
    const float* source       = encoded.data();
    float* destination        = output.data();
    for (int64_t channel = 0; channel < kChannels; ++channel) {
        for (int64_t time = 0; time < copy_time; ++time) {
            for (int64_t frequency = 0; frequency < kFrequencyBins; ++frequency) {
                destination[channel * static_cast<int64_t>(kFrequencyBins) * audio_length + time * kFrequencyBins + frequency] =
                    source[time * encoded.shape()[0] + channel * kFrequencyBins + frequency];
            }
        }
    }
    if (gap_mask_out != nullptr && (request_gap_fill || ltx_audio_gap_fill_enabled())) {
        *gap_mask_out = build_ltxav_audio_gap_mask(wav, wav_channels, source_time, audio_length,
                                                   ltx_audio_gap_silence_db());
    }
    LOG_INFO("LTX drive audio encoded: %s (%lld samples x%u ch, %lld source tokens -> %d target tokens)",
             wav_path,
             static_cast<long long>(wav_frames),
             wav_channels,
             source_time,
             audio_length);
    return output;
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

struct ConditionerRunnerDoneOnExit {
    Conditioner* conditioner = nullptr;
    ~ConditionerRunnerDoneOnExit() {
        if (conditioner != nullptr) {
            conditioner->runner_done();
        }
    }
};

struct CircularAxesState {
    bool circular_x = false;
    bool circular_y = false;
};

static void apply_circular_axes_to_diffusion(sd_ctx_t* sd_ctx, bool circular_x, bool circular_y) {
    sd_ctx->sd->circular_x = circular_x;
    sd_ctx->sd->circular_y = circular_y;
    if (sd_ctx->sd->diffusion_model) {
        sd_ctx->sd->diffusion_model->set_circular_axes(circular_x, circular_y);
    }
    if (sd_ctx->sd->high_noise_diffusion_model) {
        sd_ctx->sd->high_noise_diffusion_model->set_circular_axes(circular_x, circular_y);
    }
    if (sd_ctx->sd->control_net) {
        sd_ctx->sd->control_net->set_circular_axes(circular_x, circular_y);
    }
    if (circular_x || circular_y) {
        LOG_INFO("Using circular padding for convolutions (x=%s, y=%s)",
                 circular_x ? "true" : "false",
                 circular_y ? "true" : "false");
    }
}

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

static sd::Tensor<float> ensure_image_tensor_channels(sd::Tensor<float> image, int channels) {
    if (image.empty()) {
        return image;
    }
    GGML_ASSERT(image.dim() == 4);
    int64_t current_channels = image.shape()[2];
    if (current_channels == channels) {
        return image;
    }
    if (channels == 4) {
        sd::Tensor<float> alpha = sd::full<float>({image.shape()[0], image.shape()[1], 1, image.shape()[3]}, 1.f);
        if (current_channels == 3) {
            return sd::ops::concat(image, alpha, 2);
        }
        if (current_channels == 1) {
            sd::Tensor<float> rgb = sd::ops::concat(image, image, 2);
            rgb                   = sd::ops::concat(rgb, image, 2);
            return sd::ops::concat(rgb, alpha, 2);
        }
    }
    if (channels == 3 && current_channels >= 3) {
        return sd::ops::slice(image, 2, 0, 3);
    }
    GGML_ABORT("cannot convert image tensor from %lld to %d channels",
               (long long)current_channels,
               channels);
}

static std::optional<ImageGenerationLatents> prepare_image_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_img_gen_params_t* sd_img_gen_params,
                                                                              GenerationRequest* request,
                                                                              SamplePlan* plan,
                                                                              const RefImageParams& ref_image_params) {
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> init_image_tensor;
    sd::Tensor<float> control_image_tensor;
    sd::Tensor<float> mask_image_tensor;
    int image_channels = sd_ctx->sd->get_image_channels();

    if (sd_img_gen_params->init_image.data != nullptr) {
        LOG_INFO("IMG2IMG");

        if (request->strength < 1.f) {
            bool strength_as_noise_level = false;
            bool force_first_sigma       = false;
            for (const auto& [key, value] : parse_key_value_args(sd_img_gen_params->sample_params.extra_sample_args, "img2img arg")) {
                if (key == "strength_as_noise_level") {
                    if (!parse_strict_bool(value, strength_as_noise_level)) {
                        LOG_WARN("ignoring invalid img2img sample arg '%s=%s'", key.c_str(), value.c_str());
                    }
                } else if (key == "force_first_sigma") {
                    if (!parse_strict_bool(value, force_first_sigma)) {
                        LOG_WARN("ignoring invalid img2img sample arg '%s=%s'", key.c_str(), value.c_str());
                    }
                }
            }

            size_t t_enc;
            float target_sigma = -1;
            if (!strength_as_noise_level) {
                t_enc = static_cast<size_t>(plan->sample_steps * request->strength);
                if (t_enc == static_cast<size_t>(plan->sample_steps)) {
                    t_enc--;
                }
            } else {
                LOG_DEBUG("Interpreting denoise strength as relative noise level");
                // assume x_noised = K * (x * (1-noise_level) + noise * noise_level) = K * lerp(x, noise, noise_level)
                // K = 1, noise_level = sigma for flow models
                // K = 1+sigma, noise_level=sigma/(1+sigma) for diffusion models
                float target_noise_level = request->strength;
                target_sigma             = sd_ctx->sd->denoiser->noise_level_to_sigma(target_noise_level);
                size_t start_index       = 0;
                for (size_t i = 0; i < plan->sigmas.size(); ++i) {
                    if (plan->sigmas[i] <= target_sigma) {
                        start_index = i;
                        break;
                    }
                }

                if (start_index >= plan->sigmas.size() - 1) {
                    start_index = plan->sigmas.size() - 2;  // Leave at least 1 step
                }
                t_enc = plan->sample_steps - start_index - 1;
            }
            LOG_INFO("target t_enc is %zu steps", t_enc);
            std::vector<float> sigma_sched;
            sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());

            if (target_sigma > 0 && force_first_sigma && strength_as_noise_level) {
                LOG_DEBUG("force_first_sigma to %.4f (from %.4f)", target_sigma, sigma_sched[0]);
                sigma_sched[0] = target_sigma;
            }

            plan->sigmas       = std::move(sigma_sched);
            plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
        }

        init_image_tensor = ensure_image_tensor_channels(sd_image_to_tensor(sd_img_gen_params->init_image, request->width, request->height),
                                                         image_channels);
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
        if (sd_ctx->sd->version == VERSION_QWEN_IMAGE_LAYERED) {
            init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->qwen_image_layers + 1, true);
        } else {
            init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height);
        }
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(init_image_tensor);
        if (init_latent.empty()) {
            LOG_ERROR("failed to encode init image");
            return std::nullopt;
        }
    }

    if (sd_ctx->sd->animatediff_num_frames > 1 &&
        init_latent.dim() >= 4 && init_latent.shape()[3] == 1) {
        int n_frames = sd_ctx->sd->animatediff_num_frames;
        std::vector<int64_t> shape(init_latent.shape().begin(), init_latent.shape().end());
        shape[3] = n_frames;
        if (!init_image_tensor.empty()) {
            sd::Tensor<float> replicated(shape);
            for (int f = 0; f < n_frames; ++f) {
                sd::ops::slice_assign(&replicated, 3, f, f + 1, init_latent);
            }
            init_latent = std::move(replicated);
        } else {
            init_latent = sd::Tensor<float>(std::move(shape));
        }
    }

    if (!control_image_tensor.empty()) {
        control_latent = sd_ctx->sd->encode_first_stage(control_image_tensor);
        if (control_latent.empty()) {
            LOG_ERROR("failed to encode control image");
            return std::nullopt;
        }
    }

    std::vector<sd::Tensor<float>> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(ensure_image_tensor_channels(sd_image_to_tensor(sd_img_gen_params->ref_images[i]),
                                                          image_channels));
    }

    if (ref_images.empty() && sd_version_is_unet_edit(sd_ctx->sd->version)) {
        LOG_WARN("This model needs at least one reference image; using an empty reference");
        ref_images.push_back(sd::zeros<float>({request->width, request->height, image_channels, 1}));
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
        if (ref_image_params.resize_before_vae && !sd_version_is_pid(sd_ctx->sd->version)) {
            LOG_DEBUG("auto resize ref images");
            double vae_width;
            double vae_height;
            if (ref_image_params.resize_vae_to_target) {
                vae_width  = request->width;
                vae_height = request->height;
            } else {
                int target_pixels  = ref_image_params.vae_input_max_pixels > 0 ? ref_image_params.vae_input_max_pixels : 1024 * 1024;
                int vae_image_size = std::min(target_pixels, request->width * request->height);
                vae_width          = sqrt(vae_image_size * ref_images[i].shape()[0] / ref_images[i].shape()[1]);
                vae_height         = vae_width * ref_images[i].shape()[1] / ref_images[i].shape()[0];
            }

            int factor = sd_version_is_qwen_image(sd_ctx->sd->version) ? 32 : 16;
            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            // crop_vae_to_target_ar: a plain interpolate to a target of a different
            // aspect ratio STRETCHES the reference — for an identity-preserving edit
            // that is a distorted face, not a framing choice. Center-crop to the
            // target AR first (the krea2_edit trainer's geometry) so the resize is
            // pure scale. Only meaningful together with resize_vae_to_target.
            // Deliberately a LOCAL copy: ref_images[] is what the VLM grounds on, and
            // the reference nodes ground on the UNCROPPED image.
            const sd::Tensor<float>* vae_ref_src = &ref_images[i];
            sd::Tensor<float> cropped_ref;
            if (ref_image_params.crop_vae_to_target_ar && vae_width > 0 && vae_height > 0) {
                const int64_t src_w = ref_images[i].shape()[0];
                const int64_t src_h = ref_images[i].shape()[1];
                const double scale  = std::max(vae_width / static_cast<double>(src_w),
                                              vae_height / static_cast<double>(src_h));
                const int64_t crop_w = std::min<int64_t>(src_w, std::llround(vae_width / scale));
                const int64_t crop_h = std::min<int64_t>(src_h, std::llround(vae_height / scale));
                if (crop_w > 0 && crop_h > 0 && (crop_w != src_w || crop_h != src_h)) {
                    const int64_t x0 = (src_w - crop_w) / 2;
                    const int64_t y0 = (src_h - crop_h) / 2;
                    LOG_DEBUG("center-crop ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64 " (target AR)",
                              static_cast<int>(i), src_w, src_h, crop_w, crop_h);
                    cropped_ref = sd::ops::slice(sd::ops::slice(ref_images[i], 0, x0, x0 + crop_w),
                                                 1,
                                                 y0,
                                                 y0 + crop_h);
                    vae_ref_src = &cropped_ref;
                }
            }

            auto resized_ref_img = sd::ops::interpolate(*vae_ref_src,
                                                        {static_cast<int>(vae_width),
                                                         static_cast<int>(vae_height),
                                                         vae_ref_src->shape()[2],
                                                         vae_ref_src->shape()[3]});

            LOG_DEBUG("resize vae ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64,
                      static_cast<int>(i),
                      vae_ref_src->shape()[1],
                      vae_ref_src->shape()[0],
                      resized_ref_img.shape()[1],
                      resized_ref_img.shape()[0]);

            ref_latent = sd_ctx->sd->encode_reference_latent_cached(resized_ref_img,
                                                                    request->width,
                                                                    request->height,
                                                                    sd_img_gen_params->circular_x,
                                                                    sd_img_gen_params->circular_y,
                                                                    static_cast<int>(i));
        } else {
            ref_latent = sd_ctx->sd->encode_reference_latent_cached(ref_images[i],
                                                                    request->width,
                                                                    request->height,
                                                                    sd_img_gen_params->circular_x,
                                                                    sd_img_gen_params->circular_y,
                                                                    static_cast<int>(i));
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
                                                                            ImageGenerationLatents* latents,
                                                                            const RefImageParams& ref_image_params) {
    ConditionerRunnerDoneOnExit conditioner_runner_done{sd_ctx->sd->cond_stage_model.get()};

    ConditionerParams condition_params;
    condition_params.text      = request->prompt;
    condition_params.clip_skip = request->clip_skip;
    condition_params.width     = request->width;
    condition_params.height    = request->height;
    if (ref_image_params.pass_to_vlm) {
        condition_params.ref_images = &latents->ref_images;
    }

    condition_params.ref_image_params = ref_image_params;
    // Names the adapter set attached to the cond_stage model for this request, so
    // the VLM image-embed cache cannot serve an embedding computed under a
    // different (or absent) LoRA. See ConditionerParams::weight_adapter_signature.
    condition_params.weight_adapter_signature = sd_ctx->sd->current_lora_signature;

    sd_ctx->sd->prepare_generation_extensions(request->pm_params,
                                              request->pulid_params,
                                              condition_params,
                                              plan->total_steps);
    int64_t prepare_start_ms         = ggml_time_ms();
    condition_params.zero_out_masked = false;
    auto cond                        = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                           condition_params);
    if (cond.c_concat.empty() && ref_image_params.pass_to_dit) {
        cond.c_concat = latents->concat_latent;  // TODO: optimize
    }

    bool use_ref_latent_img_cfg = request->use_img_uncond &&
                                  !latents->ref_images.empty() &&
                                  sd_version_supports_ref_latent_img_cfg(sd_ctx->sd->version);

    SDCondition uncond;
    if (request->use_uncond || request->use_high_noise_uncond) {
        if (sd_version_is_ideogram4(sd_ctx->sd->version)) {
            uncond.c_vector = sd::Tensor<float>::from_vector({1.0f});
        } else if (sd_version_is_minit2i(sd_ctx->sd->version)) {
            // MiniT2I derives the unconditional signal from the same T5 hidden
            // states with a zeroed prompt mask, so no extra text encode is needed.
            uncond.c_crossattn = cond.c_crossattn;
            uncond.c_vector    = sd::Tensor<float>::zeros_like(cond.c_vector);
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
        if (uncond.c_concat.empty() && ref_image_params.pass_to_dit) {
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
            if (img_uncond.c_concat.empty() && ref_image_params.pass_to_dit) {
                img_uncond.c_concat = latents->img_uncond_concat_latent;  // TODO: optimize
            }
        }
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    ImageGenerationEmbeds embeds;
    embeds.img_uncond = std::move(img_uncond);
    embeds.cond       = std::move(cond);
    embeds.uncond     = std::move(uncond);

    return embeds;
}

static sd_image_t* decode_image_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const std::vector<sd::Tensor<float>>& final_latents,
                                        int* num_images_out) {
    if (final_latents.empty()) {
        LOG_ERROR("no latent images to decode");
        return nullptr;
    }
    if (final_latents.size() > static_cast<size_t>(request.batch_count)) {
        LOG_ERROR("expected at most %d latents, got %zu", request.batch_count, final_latents.size());
        return nullptr;
    }
    if (final_latents.size() < static_cast<size_t>(request.batch_count)) {
        LOG_INFO("decoding %zu/%d latents", final_latents.size(), request.batch_count);
    } else {
        LOG_INFO("decoding %zu latents", final_latents.size());
    }
    std::vector<sd::Tensor<float>> decoded_images;
    int64_t t0     = ggml_time_ms();
    bool cancelled = false;

    for (size_t i = 0; i < final_latents.size(); i++) {
        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling latent decodings");
            cancelled = true;
            break;
        }
        int64_t t1 = ggml_time_ms();
        if (sd_ctx->sd->version == VERSION_QWEN_IMAGE_LAYERED) {
            int qwen_image_latent_layers = request.qwen_image_layers + 1;
            if (final_latents[i].dim() < 5 || final_latents[i].shape()[2] < qwen_image_latent_layers) {
                LOG_ERROR("qwen image layered expected at least %d latent layers, got shape dim=%d",
                          qwen_image_latent_layers,
                          final_latents[i].dim());
                return nullptr;
            }
            for (int layer_index = 0; layer_index < qwen_image_latent_layers; layer_index++) {
                if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
                    LOG_ERROR("cancelling latent decodings");
                    cancelled = true;
                    break;
                }
                sd::Tensor<float> layer_latent = sd::ops::slice(final_latents[i], 2, layer_index, layer_index + 1);
                layer_latent.squeeze_(2);
                sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(layer_latent);
                if (image.empty()) {
                    LOG_ERROR("decode_first_stage failed for latent %zu layer %d", i + 1, layer_index + 1);
                    return nullptr;
                }
                decoded_images.push_back(std::move(image));
            }
            if (cancelled) {
                break;
            }
        } else if (sd_ctx->sd->animatediff_num_frames > 1 &&
                   final_latents[i].dim() >= 4 &&
                   final_latents[i].shape()[3] == sd_ctx->sd->animatediff_num_frames) {
            int n_frames = sd_ctx->sd->animatediff_num_frames;
            for (int f = 0; f < n_frames; ++f) {
                if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
                    LOG_ERROR("cancelling latent decodings");
                    cancelled = true;
                    break;
                }
                sd::Tensor<float> frame_latent = sd::ops::slice(final_latents[i], 3, f, f + 1);
                sd::Tensor<float> image        = sd_ctx->sd->decode_first_stage(frame_latent);
                if (image.empty()) {
                    LOG_ERROR("decode_first_stage failed for AnimateDiff frame %d/%d", f + 1, n_frames);
                    return nullptr;
                }
                decoded_images.push_back(std::move(image));
            }
        } else {
            sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);
            if (image.empty()) {
                LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
                return nullptr;
            }
            decoded_images.push_back(std::move(image));
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %zu decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t0) * 1.0f / 1000);
    if (decoded_images.empty()) {
        LOG_ERROR(cancelled ? "cancelled before any latent images were decoded" : "no decoded images");
        return nullptr;
    }

    int image_count           = static_cast<int>(decoded_images.size());
    sd_image_t* result_images = (sd_image_t*)calloc(image_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    if (num_images_out != nullptr) {
        *num_images_out = image_count;
    }

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i] = tensor_to_sd_image(decoded_images[i]);
    }

    return result_images;
}

static sd::Tensor<float> upscale_hires_latent(sd_ctx_t* sd_ctx,
                                              const sd::Tensor<float>& latent,
                                              const GenerationRequest& request,
                                              UpscalerGGML* upscaler) {
    if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
        LOG_ERROR("cancelling hires latent upscale");
        return {};
    }

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
        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling hires image upscale");
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

        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling hires latent encode");
            return {};
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

SD_API bool generate_image(sd_ctx_t* sd_ctx,
                           const sd_img_gen_params_t* sd_img_gen_params,
                           sd_image_t** images_out,
                           int* num_images_out) {
    if (images_out != nullptr) {
        *images_out = nullptr;
    }
    if (num_images_out != nullptr) {
        *num_images_out = 0;
    }
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return false;
    }

    sd_ctx->sd->reset_cancel_flag();

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    LOG_INFO("generate_image %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);
    apply_circular_axes_to_diffusion(sd_ctx, sd_img_gen_params->circular_x, sd_img_gen_params->circular_y);

    const RefImageParams ref_image_params = sd_ctx->sd->resolve_ref_image_params(sd_img_gen_params->ref_image_args);

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    request.reconcile_cfg_pp_sampler(&plan.sample_method);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan,
                                                        ref_image_params);
    if (!latents_opt.has_value()) {
        return false;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents,
                                                      ref_image_params);
    if (!embeds_opt.has_value()) {
        return false;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    std::vector<sd::Tensor<float>> final_latents;
    int64_t denoise_start = ggml_time_ms();
    for (int b = 0; b < request.batch_count; b++) {
        sd_cancel_mode_t cancel = sd_ctx->sd->get_cancel_flag();
        if (cancel == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation");
            return false;
        }
        if (cancel == SD_CANCEL_NEW_LATENTS) {
            LOG_INFO("cancelling new latent generation, returning %zu/%d completed latents",
                     final_latents.size(),
                     request.batch_count);
            break;
        }

        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = request.seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

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
                                                   ref_image_params,
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
        return false;
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %zu latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);
    if (final_latents.empty()) {
        LOG_ERROR("no latent images generated");
        return false;
    }

    if (request.hires.enabled && request.hires.target_width > 0) {
        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation before hires fix");
            return false;
        }
        LOG_INFO("hires fix: upscaling to %dx%d", request.hires.target_width, request.hires.target_height);

        std::unique_ptr<UpscalerGGML> hires_upscaler;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation before hires model load");
                return false;
            }
            LOG_INFO("hires fix: loading model upscaler from '%s'", request.hires.model_path);
            hires_upscaler                    = std::make_unique<UpscalerGGML>(sd_ctx->sd->n_threads,
                                                            false,
                                                            request.hires.upscale_tile_size,
                                                            sd_ctx->sd->backend_spec,
                                                            sd_ctx->sd->params_backend_spec);
            const size_t max_graph_vram_bytes = sd_ctx->sd->max_graph_vram_bytes_for_module(SDBackendModule::UPSCALER);
            hires_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
            if (!hires_upscaler->load_from_file(request.hires.model_path,
                                                sd_ctx->sd->n_threads)) {
                LOG_ERROR("load hires model upscaler failed");
                return false;
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
            if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation during hires fix");
                return false;
            }
            int64_t cur_seed = request.seed + b;
            sd_ctx->sd->rng->manual_seed(cur_seed);
            sd_ctx->sd->sampler_rng->manual_seed(cur_seed);

            sd::Tensor<float> upscaled = upscale_hires_latent(sd_ctx,
                                                              final_latents[b],
                                                              request,
                                                              hires_upscaler.get());
            if (upscaled.empty()) {
                return false;
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
                                                            ref_image_params,
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
            return false;
        }
        int64_t hires_denoise_end = ggml_time_ms();
        LOG_INFO("hires fix completed, taking %.2fs", (hires_denoise_end - hires_denoise_start) * 1.0f / 1000);

        final_latents = std::move(hires_final_latents);
    }

    int num_images = 0;
    auto result    = decode_image_outputs(sd_ctx, request, final_latents, &num_images);
    if (result == nullptr) {
        return false;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    if (num_images_out != nullptr) {
        *num_images_out = num_images;
    }
    if (images_out != nullptr) {
        *images_out = result;
    } else {
        free_sd_images(result, num_images);
    }
    return true;
}

static std::optional<ImageGenerationLatents> prepare_video_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_vid_gen_params_t* sd_vid_gen_params,
                                                                              GenerationRequest* request) {
    ImageGenerationLatents latents;
    int64_t prepare_start_ms = ggml_time_ms();

    if (sd_version_is_longcat_avatar(sd_ctx->sd->version)) {
        if (auto avatar_model = std::dynamic_pointer_cast<LongCatAvatarModel>(sd_ctx->sd->diffusion_model)) {
            avatar_model->cont_num_ref_latents = 0;
            avatar_model->bsa_enabled          = sd_vid_gen_params->bsa_enabled != 0;
            avatar_model->bsa_radius           = sd_vid_gen_params->bsa_radius;
            avatar_model->bsa_self_frame       = sd_vid_gen_params->bsa_self_frame != 0;
            avatar_model->bsa_bookend          = sd_vid_gen_params->bsa_bookend != 0;
            avatar_model->bsa_cube_h           = sd_vid_gen_params->bsa_cube_h;
            avatar_model->bsa_cube_w           = sd_vid_gen_params->bsa_cube_w;
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
        const bool has_audio = sd_ctx->sd->diffusion_model == nullptr || sd_ctx->sd->diffusion_model->has_audio_stream();
        if (has_audio) {
            latents.audio_length = get_ltxav_num_audio_latents(request->frames, request->fps);
            const char* drive_audio_path = sd_vid_gen_params->drive_audio_path;
            if (drive_audio_path != nullptr && drive_audio_path[0] != '\0') {
                latents.audio_latent = encode_ltxav_drive_audio(sd_ctx, drive_audio_path, latents.audio_length,
                                                                &latents.audio_gap_mask,
                                                                sd_vid_gen_params->audio_fill_gaps != 0);
                if (latents.audio_latent.empty()) {
                    LOG_ERROR("LTX drive audio was requested but could not be encoded");
                    return std::nullopt;
                }
                latents.audio_fixed = true;
            } else {
                latents.audio_latent = make_ltxav_empty_audio_latent(latents.audio_length);
            }
        } else {
            latents.audio_length = 0;
        }
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version)) {
        const bool has_guide_latent = sd_vid_gen_params->v2v_guide_latent_path != nullptr &&
                                      sd_vid_gen_params->v2v_guide_latent_path[0] != '\0';
        if ((sd_vid_gen_params->control_frames_size > 0 || has_guide_latent) &&
            sd_vid_gen_params->v2v_mode != 0 && sd_vid_gen_params->v2v_mode != 1 && sd_vid_gen_params->v2v_mode != 2) {
            LOG_ERROR("invalid LTXAV v2v_mode");
            return std::nullopt;
        }

        // Guide blocks appended after the target grid, in the order their latents are
        // concatenated onto init_latent: the continuation motion tail first (if any), then any
        // Director keyframes. Positions are built ONCE from this list by
        // ltxav_finalize_guides(), which is what lets a continuation shot also carry keyframes.
        std::vector<LtxvGuideSpec> ltxav_guides;
        int64_t ltxav_guide_target_frames = 0;
        int64_t ltxav_guide_appended      = 0;

        // Append a guide latent at the TAIL of the sequence with its own timeline position, and
        // hold it frozen. `instant` marks a single-frame image pin; otherwise the block is real
        // video and each latent frame spans a full temporal window.
        auto ltxav_append_guide = [&](const sd::Tensor<float>& guide,
                                      int frame_idx,
                                      bool instant,
                                      float mask_value) {
            if (ltxav_guide_target_frames == 0) {
                ltxav_guide_target_frames = latents.init_latent.shape()[2];
            }
            const int64_t frames = guide.shape()[2];
            latents.init_latent  = sd::ops::concat(latents.init_latent, guide, 2);
            auto guide_mask =
                sd::full<float>({guide.shape()[0], guide.shape()[1], frames, 1, 1}, mask_value);
            latents.denoise_mask = sd::ops::concat(latents.denoise_mask, guide_mask, 2);
            if (instant) {
                // One instant spec per latent frame, so an image pin keeps its historical
                // single-pixel-frame position regardless of how many latent frames it encoded to.
                for (int64_t f = 0; f < frames; ++f) {
                    ltxav_guides.push_back({frame_idx, 1, 1});
                }
            } else {
                ltxav_guides.push_back({frame_idx, static_cast<int>(frames), 8});
            }
            ltxav_guide_appended += frames;
        };

        auto ltxav_finalize_guides = [&]() {
            if (ltxav_guides.empty()) {
                return;
            }
            latents.video_target_frame_count       = ltxav_guide_target_frames;
            latents.video_conditioning_frame_count = ltxav_guide_appended;
            latents.video_positions                = build_ltxv_guides_video_positions(
                latents.init_latent.shape()[0], latents.init_latent.shape()[1],
                ltxav_guide_target_frames, ltxav_guides, request->fps,
                request->vae_scale_factor, 8, true);
        };

        auto append_ltxav_keyframes = [&]() -> bool {
            if (sd_vid_gen_params->keyframes == nullptr || sd_vid_gen_params->keyframes_size <= 0) {
                ltxav_finalize_guides();
                return true;
            }
            for (int i = 0; i < sd_vid_gen_params->keyframes_size; ++i) {
                const int frame_idx = sd_vid_gen_params->keyframe_frame_indices != nullptr
                                          ? sd_vid_gen_params->keyframe_frame_indices[i]
                                          : 0;
                if (frame_idx < 0 || frame_idx >= request->frames) {
                    LOG_ERROR("LTXAV keyframe %d frame index %d out of range [0, %d)",
                              i, frame_idx, request->frames);
                    return false;
                }
                if (sd_vid_gen_params->keyframes[i].data == nullptr) {
                    LOG_ERROR("LTXAV keyframe %d has null image data", i);
                    return false;
                }
                auto image = sd_image_to_tensor(sd_vid_gen_params->keyframes[i], request->width, request->height);
                auto keyframe_latent = encode_ltxav_condition_image(sd_ctx, image, "keyframe");
                if (keyframe_latent.empty() || keyframe_latent.shape()[0] != latents.init_latent.shape()[0] ||
                    keyframe_latent.shape()[1] != latents.init_latent.shape()[1] ||
                    keyframe_latent.shape()[3] != latents.init_latent.shape()[3]) {
                    LOG_ERROR("invalid LTXAV keyframe %d latent shape", i);
                    return false;
                }
                ltxav_append_guide(keyframe_latent, frame_idx, /*instant*/ true,
                                   1.0f - std::clamp(request->strength, 0.f, 1.f));
                LOG_INFO("LTXAV keyframe %d/%d pinned at video frame %d", i + 1,
                         sd_vid_gen_params->keyframes_size, frame_idx);
            }
            ltxav_finalize_guides();
            return true;
        };

        if ((sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0) ||
            (sd_vid_gen_params->end_cont_latent != nullptr && sd_vid_gen_params->end_cont_latent_frames > 0)) {
            // LTX chain continuation: retain the sampled video-space tail directly
            // as fixed leading context for the next window. This stays entirely in
            // latent space; decoding is only needed when rebuilding a durable prefix.
            latents.init_latent = sd_ctx->sd->generate_init_latent(request->width,
                                                                    request->height,
                                                                    request->frames,
                                                                    true);
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            const int64_t width = latents.init_latent.shape()[0];
            const int64_t height = latents.init_latent.shape()[1];
            const int64_t frames = latents.init_latent.shape()[2];
            const int64_t channels = latents.init_latent.shape()[3];
            const int64_t carried_frames = sd_vid_gen_params->cont_latent_frames;
            if (sd_vid_gen_params->cont_latent != nullptr &&
                (sd_vid_gen_params->cont_latent_width != width ||
                 sd_vid_gen_params->cont_latent_height != height ||
                 sd_vid_gen_params->cont_latent_channels != channels ||
                 carried_frames <= 0 || carried_frames > frames)) {
                LOG_ERROR("LTX continuation latent shape mismatch: got %dx%dx%dx%d, expected %lldx%lldx1..%lldx%lld",
                          sd_vid_gen_params->cont_latent_width,
                          sd_vid_gen_params->cont_latent_height,
                          sd_vid_gen_params->cont_latent_frames,
                          sd_vid_gen_params->cont_latent_channels,
                          (long long)width,
                          (long long)height,
                          (long long)frames,
                          (long long)channels);
                return std::nullopt;
            }
            float mask_value = 0.f;
            if (const char* configured = std::getenv("LTXAV_CONT_OVERLAP_MASK")) {
                mask_value = std::clamp(static_cast<float>(atof(configured)), 0.f, 1.f);
            }
            if (sd_vid_gen_params->cont_latent != nullptr) {
                sd::Tensor<float> continuation({width, height, carried_frames, channels, 1});
                std::memcpy(continuation.data(),
                            sd_vid_gen_params->cont_latent,
                            static_cast<size_t>(continuation.numel()) * sizeof(float));
                // DEFAULT (Director keyframe convention): append the prior segment's
                // motion-carrying tail as EXTRA tokens at the TAIL of the sequence — NOT
                // overwriting output frames 0..K-1 — give those tokens their own timeline
                // position, hold them frozen, and crop them off after sampling
                // (video_conditioning_frame_count). Placing the guide at the HEAD instead makes
                // it steal the new segment's low RoPE slots 0..K-1 and over-anchor the
                // immediately-adjacent generated frames into a faded echo — worse at higher fps,
                // and the jumpy seam at a segment crossover.
                //
                // The guide is pinned at the segment start (frame_idx 0) so this segment
                // re-renders the overlap region as a warm-up and then continues; the chain trims
                // that overlap back off at the pinned 8*K seam drop.
                //
                // LTXAV_CONT_LEGACY_HEAD=1 restores the old head placement for A/B.
                bool legacy_head = false;
                if (const char* e = std::getenv("LTXAV_CONT_LEGACY_HEAD")) {
                    legacy_head = atoi(e) != 0;
                }
                if (legacy_head) {
                    if (!apply_ltxav_condition_by_latent_index(&latents.init_latent,
                                                               &latents.denoise_mask,
                                                               continuation,
                                                               0,
                                                               "continuation",
                                                               mask_value)) {
                        return std::nullopt;
                    }
                    LOG_INFO("LTX continuation (LEGACY head-place): pinned %lld carried latent frame(s), mask=%.2f",
                             (long long)carried_frames,
                             mask_value);
                } else {
                    int kf_idx = 0;
                    if (const char* e = std::getenv("LTXAV_CONT_KEYFRAME_IDX")) {
                        kf_idx = atoi(e);
                    }
                    ltxav_append_guide(continuation, kf_idx, /*instant*/ false, mask_value);
                    LOG_INFO("LTX continuation (keyframe-append): %lld carried latent frame(s) at frame_idx %d, mask=%.2f",
                             (long long)carried_frames,
                             kf_idx,
                             mask_value);
                }
            }
            const int64_t end_frames = sd_vid_gen_params->end_cont_latent_frames;
            if (sd_vid_gen_params->end_cont_latent != nullptr) {
                if (sd_vid_gen_params->end_cont_latent_width != width ||
                    sd_vid_gen_params->end_cont_latent_height != height ||
                    sd_vid_gen_params->end_cont_latent_channels != channels ||
                    end_frames <= 0 || end_frames > frames) {
                    LOG_ERROR("LTX end continuation latent shape mismatch: got %dx%dx%dx%d, expected %lldx%lldx1..%lldx%lld",
                              sd_vid_gen_params->end_cont_latent_width,
                              sd_vid_gen_params->end_cont_latent_height,
                              sd_vid_gen_params->end_cont_latent_frames,
                              sd_vid_gen_params->end_cont_latent_channels,
                              (long long)width,
                              (long long)height,
                              (long long)frames,
                              (long long)channels);
                    return std::nullopt;
                }
                sd::Tensor<float> end_continuation({width, height, end_frames, channels, 1});
                std::memcpy(end_continuation.data(),
                            sd_vid_gen_params->end_cont_latent,
                            static_cast<size_t>(end_continuation.numel()) * sizeof(float));
                if (!apply_ltxav_condition_by_latent_index(&latents.init_latent,
                                                           &latents.denoise_mask,
                                                           end_continuation,
                                                           frames - end_frames,
                                                           "end continuation",
                                                           mask_value)) {
                    return std::nullopt;
                }
                LOG_INFO("LTX retake: pinned %lld end latent frame(s), mask=%.2f",
                         (long long)end_frames,
                         mask_value);
            }
            if (!append_ltxav_keyframes()) {
                return std::nullopt;
            }
        } else if (sd_vid_gen_params->keyframes != nullptr && sd_vid_gen_params->keyframes_size > 0) {
            latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            if (!append_ltxav_keyframes()) {
                return std::nullopt;
            }
        } else if (!start_image.empty() || !end_image.empty()) {
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
        }
    }

    if (sd_version_is_hunyuan_video(sd_ctx->sd->version) &&
        (!start_image.empty() || !end_image.empty())) {
        LOG_INFO("Hunyuan Video IMG2VID");

        int64_t t1                  = ggml_time_ms();
        auto concat_latent          = sd_ctx->sd->generate_init_latent(request->width,
                                                                       request->height,
                                                                       request->frames,
                                                                       true);
        auto encode_condition_frame = [&](const sd::Tensor<float>& image,
                                          int64_t latent_frame,
                                          const char* name) -> bool {
            auto encoded = sd_ctx->sd->encode_first_stage(image);
            if (encoded.empty()) {
                LOG_ERROR("failed to encode Hunyuan Video %s conditioning frame", name);
                return false;
            }
            if (encoded.dim() == 4) {
                encoded.unsqueeze_(2);
            }
            if (encoded.dim() != 5 ||
                encoded.shape()[0] != concat_latent.shape()[0] ||
                encoded.shape()[1] != concat_latent.shape()[1] ||
                encoded.shape()[3] != concat_latent.shape()[3]) {
                LOG_ERROR("invalid Hunyuan Video %s conditioning latent shape", name);
                return false;
            }
            sd::ops::slice_assign(&concat_latent,
                                  2,
                                  latent_frame,
                                  latent_frame + 1,
                                  sd::ops::slice(encoded, 2, 0, 1));
            return true;
        };

        if (!start_image.empty() && !encode_condition_frame(start_image, 0, "start")) {
            return std::nullopt;
        }
        if (!end_image.empty() &&
            !encode_condition_frame(end_image, concat_latent.shape()[2] - 1, "end")) {
            return std::nullopt;
        }

        sd::Tensor<float> concat_mask = sd::zeros<float>({concat_latent.shape()[0],
                                                          concat_latent.shape()[1],
                                                          concat_latent.shape()[2],
                                                          1,
                                                          1});
        if (!start_image.empty()) {
            sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
        }
        if (!end_image.empty()) {
            sd::ops::fill_slice(&concat_mask, 2, concat_mask.shape()[2] - 1, concat_mask.shape()[2], 1.0f);
        }
        latents.concat_latent = sd::ops::concat(concat_latent, concat_mask, 3);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    }

    if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
        LOG_INFO("IMG2VID");

        if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
            if (!start_image.empty()) {
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

        sd::Tensor<float> concat_mask = sd::zeros<float>({latents.concat_latent.shape()[0],
                                                          latents.concat_latent.shape()[1],
                                                          latents.concat_latent.shape()[2],
                                                          4,
                                                          1});  // [b, 4, t, h/vae_scale_factor, w/vae_scale_factor]
        if (!start_image.empty()) {
            sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
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
    } else if (sd_version_is_lingbot_video(sd_ctx->sd->version) && !start_image.empty()) {
        LOG_INFO("LingBot Video IMG2VID");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode init video frame");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        latents.ref_images.push_back(start_image);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_version_is_longcat_avatar(sd_ctx->sd->version) &&
               sd_vid_gen_params->cont_latent != nullptr && sd_vid_gen_params->cont_latent_frames > 0) {
        // LongCat Avatar continuation uses the prior sampled latent tail as
        // leading fixed context. This is the same latent space generated by
        // generate_video_ex, so no decoded-frame round-trip is involved.
        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width,
                                                                request->height,
                                                                request->frames,
                                                                true);
        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0],
                                                latents.init_latent.shape()[1],
                                                latents.init_latent.shape()[2],
                                                1,
                                                1},
                                                1.f);
        const int64_t width = latents.init_latent.shape()[0];
        const int64_t height = latents.init_latent.shape()[1];
        const int64_t frames = latents.init_latent.shape()[2];
        const int64_t channels = latents.init_latent.shape()[3];
        const int64_t carried_frames = sd_vid_gen_params->cont_latent_frames;
        if (sd_vid_gen_params->cont_latent_width != width ||
            sd_vid_gen_params->cont_latent_height != height ||
            sd_vid_gen_params->cont_latent_channels != channels ||
            carried_frames <= 0 || carried_frames > frames) {
            LOG_ERROR("LongCat Avatar continuation latent shape mismatch: got %dx%dx%dx%d, expected %lldx%lldx1..%lldx%lld",
                      sd_vid_gen_params->cont_latent_width,
                      sd_vid_gen_params->cont_latent_height,
                      sd_vid_gen_params->cont_latent_frames,
                      sd_vid_gen_params->cont_latent_channels,
                      (long long)width,
                      (long long)height,
                      (long long)frames,
                      (long long)channels);
            return std::nullopt;
        }
        sd::Tensor<float> continuation({width, height, carried_frames, channels, 1});
        std::memcpy(continuation.data(),
                    sd_vid_gen_params->cont_latent,
                    static_cast<size_t>(continuation.numel()) * sizeof(float));
        sd::ops::slice_assign(&latents.init_latent, 2, 0, carried_frames, continuation);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, carried_frames, 0.f);
        LOG_INFO("LongCat Avatar continuation: pinned %lld carried latent frame(s)",
                 (long long)carried_frames);
    } else if (sd_version_is_longcat_avatar(sd_ctx->sd->version) && !start_image.empty()) {
        // Avatar AI2V conditions its portrait in the first VAE latent frame and
        // pins that frame throughout sampling.  The DiT derives the cond split
        // from the leading zero timestep entries.
        LOG_INFO("LongCat Avatar AI2V (reference-image conditioning)");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode LongCat Avatar reference image");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);
        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0],
                                                latents.init_latent.shape()[1],
                                                latents.init_latent.shape()[2],
                                                1,
                                                1},
                                                1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
               sd_ctx->sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B") {
        LOG_INFO("VACE");
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
        if (control_frame_count > 0) {
            // Held control frames are inactive VACE context, not a reactive
            // target.  This is what lets a continuation preserve its pixel tail
            // while the remaining frames stay free to synthesize motion.
            sd::ops::fill_slice(&mask, 2, 0, control_frame_count, 0.0f);
        }

        control_video              = control_video - 0.5f;
        sd::Tensor<float> inactive = control_video * (1.0f - mask) + 0.5f;
        sd::Tensor<float> reactive = control_video * mask + 0.5f;

        inactive = sd_ctx->sd->encode_first_stage(inactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (inactive.empty()) {
            LOG_ERROR("failed to encode VACE inactive context");
            return std::nullopt;
        }

        if (sd_vid_gen_params->vace_cont_latent != nullptr) {
            const int64_t expected_w = inactive.shape()[0];
            const int64_t expected_h = inactive.shape()[1];
            const int64_t expected_c = inactive.shape()[3];
            if (sd_vid_gen_params->vace_cont_latent_width != expected_w ||
                sd_vid_gen_params->vace_cont_latent_height != expected_h ||
                sd_vid_gen_params->vace_cont_latent_channels != expected_c ||
                sd_vid_gen_params->vace_cont_latent_frames <= 0) {
                LOG_ERROR("VACE continuation latent shape mismatch: got %dx%dx%dx%d, expected %lldx%lldxT x%lld",
                          sd_vid_gen_params->vace_cont_latent_width,
                          sd_vid_gen_params->vace_cont_latent_height,
                          sd_vid_gen_params->vace_cont_latent_frames,
                          sd_vid_gen_params->vace_cont_latent_channels,
                          (long long)expected_w,
                          (long long)expected_h,
                          (long long)expected_c);
                return std::nullopt;
            }
            const int pixel_frames = std::max(1, sd_vid_gen_params->vace_cont_frames);
            const int64_t take = std::min<int64_t>(
                inactive.shape()[2],
                (std::min<int>(pixel_frames, request->frames) - 1) / 4 + 1);
            const int64_t drop_tail = std::min<int64_t>(
                std::max(0, sd_vid_gen_params->vace_cont_latent_drop_tail),
                std::max<int64_t>(0, sd_vid_gen_params->vace_cont_latent_frames - take));
            const int64_t src_end = sd_vid_gen_params->vace_cont_latent_frames - drop_tail;
            const int64_t src_beg = src_end - take;
            sd::Tensor<float> prior({expected_w,
                                     expected_h,
                                     sd_vid_gen_params->vace_cont_latent_frames,
                                     expected_c,
                                     1});
            std::memcpy(prior.data(),
                        sd_vid_gen_params->vace_cont_latent,
                        static_cast<size_t>(prior.numel()) * sizeof(float));
            auto tail = sd::ops::slice(prior, 2, src_beg, src_end);
            sd::ops::slice_assign(&inactive, 2, 0, take, tail);
            LOG_INFO("VACE continuation: injected %lld prior latent frame(s)", (long long)take);
        }

        reactive = sd_ctx->sd->encode_first_stage(reactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
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

    // Production LipDub/relip (v2v_mode 0): source pixels become a frozen,
    // timeline-aligned IC-LoRA reference block.  Unlike SDEdit this samples a
    // new target from noise, and unlike guide-edit the reference is fully
    // frozen while the fixed driving-audio latent supplies lip motion.
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        sd_vid_gen_params->v2v_mode == 0 && sd_vid_gen_params->control_frames_size > 0) {
        if (sd_vid_gen_params->control_frames == nullptr) {
            LOG_ERROR("LTXAV relip control_frames pointer is null");
            return std::nullopt;
        }
        const int64_t supplied = sd_vid_gen_params->control_frames_size;
        sd::Tensor<float> reference_video({request->width, request->height, request->frames, 3, 1});
        for (int frame = 0; frame < request->frames; ++frame) {
            const int64_t source = std::min<int64_t>(frame, supplied - 1);
            if (sd_vid_gen_params->control_frames[source].data == nullptr) {
                LOG_ERROR("LTXAV relip control frame %lld is empty", (long long)source);
                return std::nullopt;
            }
            const auto image = sd_image_to_tensor(sd_vid_gen_params->control_frames[source], request->width, request->height);
            sd::ops::slice_assign(&reference_video, 2, frame, frame + 1, image.unsqueeze(2));
        }
        auto reference = sd_ctx->sd->encode_first_stage(reference_video);
        if (reference.empty()) {
            LOG_ERROR("failed to encode LTXAV relip reference video");
            return std::nullopt;
        }
        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
        latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
        const int reference_stride = std::max(1, sd_vid_gen_params->relip_ref_tstride);
        if (reference_stride > 1 && reference.shape()[2] > 1) {
            const int64_t original_frames = reference.shape()[2];
            const int64_t kept_frames = (original_frames + reference_stride - 1) / reference_stride;
            auto shape = reference.shape();
            shape[2] = kept_frames;
            sd::Tensor<float> subsampled(shape);
            for (int64_t dst = 0, source = 0; source < original_frames; ++dst, source += reference_stride) {
                sd::ops::slice_assign(&subsampled, 2, dst, dst + 1, sd::ops::slice(reference, 2, source, source + 1));
            }
            reference = std::move(subsampled);
        }
        if (!apply_ltxav_video_relip_reference(&latents, reference, request->fps,
                                                request->vae_scale_factor, reference_stride)) {
            return std::nullopt;
        }
        LOG_INFO("LTXAV LIPDUB RELIP: target %lld latent frames, reference %lld (temporal stride %d)%s",
                 (long long)latents.video_target_frame_count, (long long)reference.shape()[2], reference_stride,
                 latents.audio_fixed ? ", fixed drive audio" : "");
    }

    // LTX V2V deliberately reaches this common t2v fallback first so
    // the normal video positions, empty audio latent, and target grid exist.
    // It then replaces only the video latent with the source clip and trims
    // the sampling schedule below according to strength.
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        (sd_vid_gen_params->v2v_mode == 1 || sd_vid_gen_params->v2v_mode == 2)) {
        const bool latent_in = sd_vid_gen_params->v2v_mode == 2 &&
                               sd_vid_gen_params->v2v_guide_latent_path != nullptr &&
                               sd_vid_gen_params->v2v_guide_latent_path[0] != '\0';
        const bool pixel_source = sd_vid_gen_params->control_frames_size > 0;
        if (!latent_in && !pixel_source) {
            LOG_ERROR("LTXAV V2V requires control_frames or a guide latent path");
            return std::nullopt;
        }
        sd::Tensor<float> source_latent;
        if (latent_in) {
            try {
                auto loaded = sd::load_tensor_from_file_as_tensor<float>(sd_vid_gen_params->v2v_guide_latent_path);
                if (loaded.empty() || loaded.dim() < 4) {
                    LOG_ERROR("LTXAV guide latent is empty or malformed: %s", sd_vid_gen_params->v2v_guide_latent_path);
                    return std::nullopt;
                }
                source_latent = sd::Tensor<float>({loaded.shape()[0], loaded.shape()[1], loaded.shape()[2], loaded.shape()[3], 1});
                std::memcpy(source_latent.data(), loaded.data(),
                            static_cast<size_t>(source_latent.numel()) * sizeof(float));
            } catch (const std::exception& error) {
                LOG_ERROR("failed to load LTX guide latent %s: %s",
                          sd_vid_gen_params->v2v_guide_latent_path,
                          error.what());
                return std::nullopt;
            }
        } else {
            if (sd_vid_gen_params->control_frames == nullptr ||
                sd_vid_gen_params->control_frames_size != request->frames) {
                LOG_ERROR("LTXAV V2V requires exactly one control frame per output frame");
                return std::nullopt;
            }
            sd::Tensor<float> source_video({request->width, request->height, request->frames, 3, 1});
            for (int frame = 0; frame < request->frames; ++frame) {
                if (sd_vid_gen_params->control_frames[frame].data == nullptr) {
                    LOG_ERROR("LTXAV V2V control frame %d is empty", frame);
                    return std::nullopt;
                }
                const auto source_image = sd_image_to_tensor(sd_vid_gen_params->control_frames[frame],
                                                              request->width,
                                                              request->height);
                sd::ops::slice_assign(&source_video, 2, frame, frame + 1, source_image.unsqueeze(2));
            }
            source_latent = sd_ctx->sd->encode_first_stage(source_video);
        }
        if (source_latent.empty() || source_latent.dim() < 4 ||
            source_latent.shape()[0] != latents.init_latent.shape()[0] ||
            source_latent.shape()[1] != latents.init_latent.shape()[1] ||
            source_latent.shape()[2] != latents.init_latent.shape()[2] ||
            source_latent.shape()[3] != latents.init_latent.shape()[3]) {
            LOG_ERROR("LTXAV SDEdit source latent does not match the target latent grid");
            return std::nullopt;
        }
        std::memcpy(latents.init_latent.data(), source_latent.data(),
                    static_cast<size_t>(source_latent.numel()) * sizeof(float));
        latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
        latents.v2v_sdedit   = true;
        LOG_INFO("LTXAV V2V %s: seeded %d video frames at strength %.2f",
                 latent_in ? "guide-edit latent-in" : sd_vid_gen_params->v2v_mode == 2 ? "guide-edit" : "SDEdit",
                 request->frames,
                 sd_vid_gen_params->v2v_mode == 2 ? sd_vid_gen_params->v2v_guide_strength : sd_vid_gen_params->strength);
    }

    // TASS overlap reference conditioning (LTX-Best-Face-ID character sheets).
    // Deliberately the LAST thing done to the video grid: it composes with every
    // conditioning path above by copying whatever positions that path committed
    // to and appending reference blocks after them.  Unlike the keyframe/relip
    // references these are NOT frame-axis concatenations, so they are exempt from
    // the matching-spatial-dims requirement -- carrying a 1536x1024 sheet into a
    // 768x448 render at full detail is the entire point.
    const bool has_character_refs = sd_vid_gen_params->character_refs != nullptr &&
                                    sd_vid_gen_params->character_refs_size > 0;
    const bool has_msr_strip = sd_vid_gen_params->msr_frames > 0;
    if (sd_version_is_ltxav(sd_ctx->sd->version) && (has_character_refs || has_msr_strip)) {
        if (latents.init_latent.empty()) {
            LOG_ERROR("LTXAV character references need a target latent grid");
            return std::nullopt;
        }
        const int scale = std::max(1, request->vae_scale_factor);
        std::vector<sd::Tensor<float>> ref_latents;
        std::vector<LtxvTassRefGrid> ref_grids;
        int next_source_id = 2;
        // The MSR strip goes FIRST so its slot order -- background, then subject 1,
        // 2, ... -- is the order the prompt's numbered figures refer to. It is also
        // the one reference encoded as a video rather than a still, and it is
        // composited at the RENDER resolution: the checkpoint ships
        // `reference_downscale_factor = 1`, so the guide shares the target's grid.
        if (has_msr_strip) {
            if (sd_vid_gen_params->msr_background == nullptr) {
                LOG_ERROR("LTXAV MSR needs a background image; it is the substrate every frame starts from");
                return std::nullopt;
            }
            auto strip = build_ltxav_msr_strip(*sd_vid_gen_params->msr_background,
                                               sd_vid_gen_params->msr_subjects,
                                               sd_vid_gen_params->msr_subjects_size,
                                               request->width,
                                               request->height,
                                               sd_vid_gen_params->msr_frames);
            if (strip.empty()) {
                return std::nullopt;
            }
            // The strip is recomposited from the same images for every shot it is
            // scoped to, so it is the single biggest beneficiary of the cache: one
            // VAE pass over a 17-to-65-frame video, once, instead of once per shot.
            auto strip_latent =
                ltxav_encode_with_reference_cache(strip, "msr", sd_vid_gen_params->msr_frames, "MSR strip", [&]() {
                    return sd_ctx->sd->encode_first_stage(strip);
                });
            if (strip_latent.empty() || strip_latent.dim() < 4) {
                LOG_ERROR("failed to encode the LTXAV MSR reference strip");
                return std::nullopt;
            }
            if (strip_latent.shape()[3] != latents.init_latent.shape()[3]) {
                LOG_ERROR("LTXAV MSR strip encoded to %lld channels, expected %lld",
                          (long long)strip_latent.shape()[3],
                          (long long)latents.init_latent.shape()[3]);
                return std::nullopt;
            }
            const int source_id = next_source_id++;
            ref_grids.push_back(LtxvTassRefGrid{strip_latent.shape()[0],
                                                strip_latent.shape()[1],
                                                strip_latent.shape()[2],
                                                static_cast<float>(source_id)});
            LOG_INFO("LTXAV MSR reference strip: %dx%d px x %d frames -> %lldx%lldx%lld latent, source_id=%d",
                     request->width,
                     request->height,
                     sd_vid_gen_params->msr_frames,
                     (long long)strip_latent.shape()[0],
                     (long long)strip_latent.shape()[1],
                     (long long)strip_latent.shape()[2],
                     source_id);
            ref_latents.push_back(std::move(strip_latent));
        }
        for (int i = 0; has_character_refs && i < sd_vid_gen_params->character_refs_size; ++i) {
            const sd_image_t& ref_image = sd_vid_gen_params->character_refs[i];
            if (ref_image.data == nullptr || ref_image.width == 0 || ref_image.height == 0) {
                LOG_ERROR("LTXAV character reference %d has no image data", i);
                return std::nullopt;
            }
            // The caller already chose the resolution (the sheet's native size, or
            // the render bucket); all that is left is snapping it onto the VAE's
            // spatial grid.
            const int ref_width  = std::max<int>(scale, static_cast<int>((ref_image.width + scale / 2) / scale) * scale);
            const int ref_height = std::max<int>(scale, static_cast<int>((ref_image.height + scale / 2) / scale) * scale);
            auto image           = sd_image_to_tensor(ref_image, ref_width, ref_height);
            auto ref_latent      = encode_ltxav_reference_image_cached(sd_ctx, image, "character");
            if (ref_latent.empty() || ref_latent.dim() < 4) {
                return std::nullopt;
            }
            if (ref_latent.shape()[3] != latents.init_latent.shape()[3]) {
                LOG_ERROR("LTXAV character reference %d encoded to %lld channels, expected %lld",
                          i,
                          (long long)ref_latent.shape()[3],
                          (long long)latents.init_latent.shape()[3]);
                return std::nullopt;
            }
            // The references are packed on the frame axis of ONE tensor, so they
            // have to share a spatial grid with each other.  They still need not
            // match the target: the token axis is what frees them from that.
            if (!ref_latents.empty() &&
                (ref_latent.shape()[0] != ref_latents.front().shape()[0] ||
                 ref_latent.shape()[1] != ref_latents.front().shape()[1])) {
                LOG_ERROR("LTXAV character reference %d is %lldx%lld latent but reference 0 is %lldx%lld; "
                          "all references in one shot must share a resolution",
                          i,
                          (long long)ref_latent.shape()[0],
                          (long long)ref_latent.shape()[1],
                          (long long)ref_latents.front().shape()[0],
                          (long long)ref_latents.front().shape()[1]);
                return std::nullopt;
            }
            int source_id = next_source_id;
            if (sd_vid_gen_params->character_ref_source_ids != nullptr &&
                sd_vid_gen_params->character_ref_source_ids[i] > 1) {
                source_id = sd_vid_gen_params->character_ref_source_ids[i];
            }
            next_source_id = source_id + 1;
            ref_grids.push_back(LtxvTassRefGrid{ref_latent.shape()[0],
                                                ref_latent.shape()[1],
                                                ref_latent.shape()[2],
                                                static_cast<float>(source_id)});
            LOG_INFO("LTXAV character reference %d/%d: %ux%u px -> %lldx%lldx%lld latent, source_id=%d, clip=%d%s",
                     i + 1,
                     sd_vid_gen_params->character_refs_size,
                     ref_image.width,
                     ref_image.height,
                     (long long)ref_latent.shape()[0],
                     (long long)ref_latent.shape()[1],
                     (long long)ref_latent.shape()[2],
                     source_id,
                     ltxav_reference_clip_frames(),
                     ltxav_reference_clip_frames() > 1 ? " (last-of-clip)" : " (SEED LATENT)");
            ref_latents.push_back(std::move(ref_latent));
        }

        if (ref_latents.empty()) {
            LOG_ERROR("LTXAV reference conditioning produced no references");
            return std::nullopt;
        }
        sd::Tensor<float> packed_refs = ref_latents.front();
        for (size_t i = 1; i < ref_latents.size(); ++i) {
            packed_refs = sd::ops::concat(packed_refs, ref_latents[i], 2);
        }
        latents.ref_video_x = std::move(packed_refs);
        if (latents.denoise_mask.empty()) {
            // Plain t2v carries a scalar timestep. The references need their own
            // (clean) timestep, which only the masked per-token path can express.
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
        }
        // Remembered so a temporal window can rebuild "tile grid ++ the same
        // reference block" for its own frame range; `tass_positions_only` records
        // that nothing but TASS wrote these positions, which is what makes that
        // rebuild safe (any other conditioning path owns a layout we must not
        // regenerate).
        latents.tass_positions_only = latents.video_positions.empty();
        latents.ref_grids           = ref_grids;
        // >= 0 is an explicit request; exactly 0 means UNTAGGED (Echo's native layout:
        // references share the target's RoPE grid with no source-phase tag at all).
        // An unsupplied phase scale defaults to the Best-Face-ID convention of 1.0,
        // EXCEPT when an MSR strip is present: MSR was trained through ComfyUI's
        // IC-LoRA guide, which tags nothing, so tagging it would be off-recipe. A
        // caller that asks for a scale explicitly still gets exactly what it asked for.
        //
        // Resolved BEFORE the positions are built: the placement policy reads it, because the
        // frame-0 collision only exists in the untagged regime.
        latents.tass_phase_scale = sd_vid_gen_params->tass_phase_scale >= 0.f
                                       ? sd_vid_gen_params->tass_phase_scale
                                       : (has_msr_strip ? 0.f : 1.f);
        latents.video_positions  = build_ltxv_tass_ref_video_positions(latents.video_positions,
                                                                      latents.init_latent.shape()[0],
                                                                      latents.init_latent.shape()[1],
                                                                      latents.init_latent.shape()[2],
                                                                      ref_grids,
                                                                      request->fps,
                                                                      request->vae_scale_factor,
                                                                      8,
                                                                      &latents.video_source_ids);
        LOG_INFO("LTXAV TASS overlap references: %d sheet(s), %lld reference latent frame(s), "
                 "%lld reference token(s), phase_scale=%.2f%s",
                 static_cast<int>(ref_grids.size()),
                 (long long)latents.ref_video_x.shape()[2],
                 (long long)(latents.ref_video_x.shape()[0] * latents.ref_video_x.shape()[1] *
                             latents.ref_video_x.shape()[2]),
                 latents.tass_phase_scale,
                 latents.tass_phase_scale == 0.f ? " (UNTAGGED / Echo layout)" : "");
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version) && !latents.audio_latent.empty()) {
        // Supplied DRIVE audio has to be PINNED (mask 0) every step or the model treats it as noise
        // to denoise rather than as clean conditioning -- the audio is fed in, nothing holds it, and
        // the mouth just does generic talking instead of following the words.
        //
        // The mask is what carries that pin, and the conditioning branches above only build one when
        // there IS video conditioning (continuation, keyframes, image). A PURE T2V render reaches
        // here with an empty mask, so the pin was silently dropped and drive audio did nothing --
        // which is why lip-sync failed on plain t2v and on segment 0 of a chain, while later
        // continuation segments (which do have a mask) worked.
        //
        // So: when audio is fixed and nothing else made a mask, make a fully-generated video mask
        // (1.0 = denoise all video normally) purely so the audio slot has somewhere to be held at 0.
        if (latents.audio_fixed && latents.denoise_mask.empty()) {
            latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
        }
        if (!latents.denoise_mask.empty()) {
            latents.denoise_mask = pack_ltxav_audio_and_video_denoise_mask(latents.denoise_mask,
                                                                           latents.init_latent,
                                                                           latents.audio_latent,
                                                                           latents.audio_fixed ? 0.f : 1.f,
                                                                           latents.audio_gap_mask.empty()
                                                                               ? nullptr
                                                                               : &latents.audio_gap_mask);
        }
        latents.init_latent = pack_ltxav_audio_and_video_latents(latents.init_latent, latents.audio_latent);
    }

    return latents;
}

static ImageGenerationEmbeds prepare_video_generation_embeds(sd_ctx_t* sd_ctx,
                                                             const sd_vid_gen_params_t* sd_vid_gen_params,
                                                             const GenerationRequest& request,
                                                             const ImageGenerationLatents& latents) {
    ConditionerRunnerDoneOnExit conditioner_runner_done{sd_ctx->sd->cond_stage_model.get()};

    ImageGenerationEmbeds embeds;
    ConditionerParams condition_params;
    condition_params.clip_skip       = request.clip_skip;
    condition_params.text            = request.prompt;
    condition_params.zero_out_masked = true;
    condition_params.ref_images      = &latents.ref_images;
    // Not reachable today (the VLM image-embed cache is krea2-only and krea2 is
    // an image model), but set for the same reason the image path does: an empty
    // signature is a real key value, and leaving it empty here would make an
    // adapter-attached video encode collide with an adapter-free one.
    condition_params.weight_adapter_signature = sd_ctx->sd->current_lora_signature;
    if (sd_version_is_lingbot_video(sd_ctx->sd->version)) {
        condition_params.ref_image_params.vlm_resize_mode = RefImageResizeMode::AREA;
    }

    // Prompt Relay: piece 0 is the shot prompt (the global anchor, zero penalty
    // everywhere) and pieces 1..N are the beat clauses, in the caller's order.
    // Encoding them as separate pieces of ONE Gemma pass is what makes the
    // per-beat token ranges exact. Gemma is causal, so beat k sees the setting
    // and every earlier beat but no later one, which is the right dependency
    // direction inside a shot.
    std::vector<std::string> relay_pieces;
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        sd_vid_gen_params->beats != nullptr && sd_vid_gen_params->beat_count > 0) {
        relay_pieces.reserve(static_cast<size_t>(sd_vid_gen_params->beat_count) + 1);
        relay_pieces.push_back(request.prompt);
        for (int beat = 0; beat < sd_vid_gen_params->beat_count; ++beat) {
            relay_pieces.push_back(SAFE_STR(sd_vid_gen_params->beats[beat].text));
        }
        condition_params.text_pieces = &relay_pieces;
    }

    int64_t prepare_start_ms = ggml_time_ms();
    const bool use_avatar_chain_cache =
        sd_version_is_longcat_avatar(sd_ctx->sd->version) &&
        sd_ctx->sd->avatar_chain_text_cache_active;
    const std::string cache_key = use_avatar_chain_cache
                                      ? StableDiffusionGGML::avatar_chain_text_key(request.prompt,
                                                                                     request.negative_prompt)
                                      : std::string();
    const auto cache_it = use_avatar_chain_cache
                              ? sd_ctx->sd->avatar_chain_text_cache.find(cache_key)
                              : sd_ctx->sd->avatar_chain_text_cache.end();
    if (cache_it != sd_ctx->sd->avatar_chain_text_cache.end() &&
        (!request.use_uncond || cache_it->second.has_uncond)) {
        LOG_INFO("LongCat Avatar: reusing chained text conditioning (prompt unchanged)");
        embeds.cond = cache_it->second.cond;
        if (request.use_uncond && cache_it->second.has_uncond) {
            embeds.uncond = cache_it->second.uncond;
        }
    } else {
        embeds.cond = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                            condition_params);
        if (request.use_uncond) {
            condition_params.text = request.negative_prompt;
            // The negative prompt has no beats; leaving pieces set would encode
            // the positive beats into the unconditional branch.
            condition_params.text_pieces = nullptr;
            embeds.uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                   condition_params);
        }
        if (use_avatar_chain_cache) {
            StableDiffusionGGML::AvatarChainTextCond cached;
            cached.cond = embeds.cond;
            cached.uncond = embeds.uncond;
            cached.has_uncond = request.use_uncond;
            sd_ctx->sd->avatar_chain_text_cache.emplace(cache_key, std::move(cached));
        }
    }
    embeds.cond.c_concat = latents.concat_latent;
    embeds.cond.c_vector = latents.clip_vision_output;
    if (request.use_uncond) {
        embeds.uncond.c_concat = latents.concat_latent;
        embeds.uncond.c_vector = latents.clip_vision_output;
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

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
    if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
        LOG_ERROR("cancelling video decode");
        return nullptr;
    }
    sd::Tensor<float> video_latent = final_latent;
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        video_latent.shape()[3] > sd_ctx->sd->get_latent_channel()) {
        video_latent = sd::ops::slice(video_latent, 3, 0, sd_ctx->sd->get_latent_channel());
    }
    LOG_DEBUG("decode_video_outputs latent %dx%dx%dx%d",
              (int)video_latent.shape()[0],
              (int)video_latent.shape()[1],
              (int)video_latent.shape()[2],
              (int)video_latent.shape()[3]);
    // Diagnostic-only latent handoff.  Sampling with NVFP4/FP8 is intentionally
    // not bit-stable across implementations, so a visual VAE A/B must decode
    // the exact same diffusion-space tensor.
    if (const char* path = std::getenv("WAN_SAVE_LATENT"); path != nullptr && path[0] != '\0') {
        sd::save_tensor_to_file<float>(path, video_latent, "wan_video_latent");
        LOG_INFO("WAN_SAVE_LATENT: wrote %s", path);
    }
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

    auto upsampler_manager = std::make_shared<ModelManager>();
    upsampler_manager->set_n_threads(sd_ctx->sd->n_threads);
    upsampler_manager->set_enable_mmap(sd_ctx->sd->enable_mmap);
    ModelLoader& model_loader = upsampler_manager->loader();
    if (!model_loader.init_from_file(model_path)) {
        LOG_ERROR("init LTX latent upsampler model loader from file failed: '%s'", model_path);
        return {};
    }

    std::unique_ptr<LTXVUpsampler::LatentUpsamplerRunner> upsampler =
        std::make_unique<LTXVUpsampler::LatentUpsamplerRunner>(sd_ctx->sd->backend_for(SDBackendModule::UPSCALER),
                                                               model_loader.get_tensor_storage_map(),
                                                               upsampler_manager);
    const size_t max_graph_vram_bytes = sd_ctx->sd->max_graph_vram_bytes_for_module(SDBackendModule::UPSCALER);
    upsampler->set_max_graph_vram_bytes(max_graph_vram_bytes);
    if (upsampler->model == nullptr) {
        LOG_ERROR("init LTX latent upsampler from metadata failed");
        return {};
    }

    std::map<std::string, ggml_tensor*> tensors;
    upsampler->get_param_tensors(tensors);
    if (!upsampler_manager->register_param_tensors("LTX latent upsampler",
                                                   std::move(tensors),
                                                   ModelManager::ResidencyMode::ParamBackend,
                                                   sd_ctx->sd->backend_for(SDBackendModule::UPSCALER),
                                                   sd_ctx->sd->params_backend_for(SDBackendModule::UPSCALER)) ||
        !upsampler_manager->validate_registered_tensors()) {
        LOG_ERROR("register LTX latent upsampler tensors with model manager failed");
        return {};
    }

    sd::Tensor<float> upscaled = upsampler->compute(sd_ctx->sd->n_threads, unnormalized);
    upsampler_manager.reset();
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

static bool apply_ltxv_refine_image_conditioning(sd_ctx_t* sd_ctx,
                                                 const sd_vid_gen_params_t* sd_vid_gen_params,
                                                 const GenerationRequest& request,
                                                 const ImageGenerationLatents& latents,
                                                 sd::Tensor<float>* latent,
                                                 sd::Tensor<float>* denoise_mask,
                                                 sd::Tensor<float>* video_positions) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_vid_gen_params == nullptr ||
        latent == nullptr || latent->empty() || denoise_mask == nullptr || video_positions == nullptr) {
        return true;
    }
    if (latents.relip_twostage) {
        if (sd_vid_gen_params->control_frames == nullptr || sd_vid_gen_params->control_frames_size <= 0) {
            LOG_ERROR("LTX LipDub two-stage refine requires VAE encoding and source control_frames");
            return false;
        }
        const int latent_channels = sd_ctx->sd->get_latent_channel();
        sd::Tensor<float> target_video = *latent;
        sd::Tensor<float> audio_latent;
        if (latent->shape()[3] > latent_channels) {
            target_video = sd::ops::slice(*latent, 3, 0, latent_channels);
            audio_latent = unpack_ltxav_audio_latent(*latent, latents.audio_length, latent_channels);
            if (audio_latent.empty()) {
                LOG_ERROR("LTX LipDub two-stage refine could not unpack audio latent");
                return false;
            }
        }
        const int image_width = static_cast<int>(target_video.shape()[0]) * request.vae_scale_factor;
        const int image_height = static_cast<int>(target_video.shape()[1]) * request.vae_scale_factor;
        sd::Tensor<float> reference_video({image_width, image_height, request.frames, 3, 1});
        for (int frame = 0; frame < request.frames; ++frame) {
            const int source = std::min(frame, sd_vid_gen_params->control_frames_size - 1);
            if (sd_vid_gen_params->control_frames[source].data == nullptr) {
                LOG_ERROR("LTX LipDub two-stage control frame %d is empty", source);
                return false;
            }
            const auto image = sd_image_to_tensor(sd_vid_gen_params->control_frames[source], image_width, image_height);
            sd::ops::slice_assign(&reference_video, 2, frame, frame + 1, image.unsqueeze(2));
        }
        sd::Tensor<float> reference = sd_ctx->sd->encode_first_stage(reference_video);
        if (reference.empty() || reference.shape()[3] != target_video.shape()[3]) {
            LOG_ERROR("LTX LipDub two-stage full-resolution reference encode failed");
            return false;
        }
        const int stride = std::max(1, sd_vid_gen_params->relip_ref_tstride);
        if (stride > 1 && reference.shape()[2] > 1) {
            const int64_t source_frames = reference.shape()[2];
            auto shape = reference.shape();
            shape[2] = (source_frames + stride - 1) / stride;
            sd::Tensor<float> subsampled(shape);
            for (int64_t output = 0, source = 0; source < source_frames; ++output, source += stride) {
                sd::ops::slice_assign(&subsampled, 2, output, output + 1, sd::ops::slice(reference, 2, source, source + 1));
            }
            reference = std::move(subsampled);
        }
        ImageGenerationLatents relip_latents;
        relip_latents.init_latent = target_video;
        relip_latents.denoise_mask = make_ltxav_video_denoise_mask(target_video, 1.f);
        relip_latents.video_target_frame_count = target_video.shape()[2];
        if (!apply_ltxav_video_relip_reference(&relip_latents, reference, request.fps,
                                                request.vae_scale_factor, stride)) {
            return false;
        }
        *video_positions = std::move(relip_latents.video_positions);
        if (audio_latent.empty()) {
            *latent = std::move(relip_latents.init_latent);
            *denoise_mask = std::move(relip_latents.denoise_mask);
        } else {
            *latent = pack_ltxav_audio_and_video_latents(relip_latents.init_latent, audio_latent);
            *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(relip_latents.denoise_mask,
                                                                       relip_latents.init_latent,
                                                                       audio_latent,
                                                                       latents.audio_fixed ? 0.f : 1.f);
        }
        LOG_INFO("LTX LipDub two-stage: re-applied %lld full-resolution reference latent frame(s) for refine",
                 (long long)reference.shape()[2]);
        return true;
    }
    if (sd_vid_gen_params->init_image.data == nullptr &&
        sd_vid_gen_params->end_image.data == nullptr) {
        // No image to re-pin -- but an AUDIO-DRIVEN render still needs its drive audio held at
        // mask 0 through stage 2. Returning here with an empty hires mask lets the refine
        // re-denoise the driving audio and undoes the lip-sync stage 1 established. Build a
        // pin-only mask (all video generated at 1.0, audio held at 0.0) for that case.
        if (latents.audio_fixed && latent != nullptr && !latent->empty() && denoise_mask != nullptr) {
            const int latent_channels = sd_ctx->sd->get_latent_channel();
            if (latent->shape()[3] > latent_channels) {
                auto video_only = sd::ops::slice(*latent, 3, 0, latent_channels);
                auto audio_only = unpack_ltxav_audio_latent(*latent, latents.audio_length, latent_channels);
                if (!audio_only.empty()) {
                    *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(
                        make_ltxav_video_denoise_mask(video_only, 1.f), video_only, audio_only, 0.f);
                }
            }
        }
        return true;
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
        *latent = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
        // Keep the driving audio PINNED through the refine too. The default here is 1.0
        // ("generate"), which lets stage 2 re-denoise the drive audio and washes out the lip-sync
        // stage 1 established.
        *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent,
                                                                latents.audio_fixed ? 0.f : 1.f);
    } else {
        *latent       = std::move(video_latent);
        *denoise_mask = std::move(video_mask);
    }
    LOG_INFO("LTXV refine image conditioning applied at %dx%d", image_width, image_height);
    return true;
}

static bool generate_animatediff_video(sd_ctx_t* sd_ctx,
                                       const sd_vid_gen_params_t* sd_vid_gen_params,
                                       sd_image_t** frames_out,
                                       int* num_frames_out) {
    int n_frames = sd_vid_gen_params->video_frames;
    if (n_frames < 1) {
        LOG_ERROR("AnimateDiff: --video-frames must be >= 1");
        return false;
    }
    if (n_frames > 32) {
        LOG_WARN("AnimateDiff motion modules have a 32-frame positional-encoding context; capping to 32");
        n_frames = 32;
    }

    sd_img_gen_params_t img_gen_params;
    sd_img_gen_params_init(&img_gen_params);
    img_gen_params.loras             = sd_vid_gen_params->loras;
    img_gen_params.lora_count        = sd_vid_gen_params->lora_count;
    img_gen_params.prompt            = sd_vid_gen_params->prompt;
    img_gen_params.negative_prompt   = sd_vid_gen_params->negative_prompt;
    img_gen_params.clip_skip         = sd_vid_gen_params->clip_skip;
    img_gen_params.width             = sd_vid_gen_params->width;
    img_gen_params.height            = sd_vid_gen_params->height;
    img_gen_params.sample_params     = sd_vid_gen_params->sample_params;
    img_gen_params.strength          = sd_vid_gen_params->strength;
    img_gen_params.init_image        = sd_vid_gen_params->init_image;
    img_gen_params.seed              = sd_vid_gen_params->seed;
    img_gen_params.batch_count       = 1;
    img_gen_params.control_strength  = 1.0f;
    img_gen_params.vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
    img_gen_params.cache             = sd_vid_gen_params->cache;
    img_gen_params.hires             = sd_vid_gen_params->hires;
    img_gen_params.qwen_image_layers = 0;
    img_gen_params.circular_x        = sd_vid_gen_params->circular_x;
    img_gen_params.circular_y        = sd_vid_gen_params->circular_y;

    sd_ctx->sd->animatediff_num_frames = n_frames;
    bool ok                            = generate_image(sd_ctx, &img_gen_params, frames_out, num_frames_out);
    sd_ctx->sd->animatediff_num_frames = 0;
    return ok;
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
                              int* latent_channels_out,
                              int* reference_head_trim_out) {
    if (reference_head_trim_out != nullptr) {
        *reference_head_trim_out = 0;
    }
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
    if (latent_width_out != nullptr) {
        *latent_width_out = 0;
    }
    if (latent_height_out != nullptr) {
        *latent_height_out = 0;
    }
    if (latent_frames_out != nullptr) {
        *latent_frames_out = 0;
    }
    if (latent_channels_out != nullptr) {
        *latent_channels_out = 0;
    }

    if (sd_ctx->sd->animatediff_loaded && sd_version_supports_animatediff(sd_ctx->sd->version)) {
        LOG_INFO("AnimateDiff dispatch: %d frames, %dx%d",
                 sd_vid_gen_params->video_frames, sd_vid_gen_params->width, sd_vid_gen_params->height);
        return generate_animatediff_video(sd_ctx, sd_vid_gen_params, frames_out, num_frames_out);
    }

    sd_ctx->sd->reset_cancel_flag();

    const RefImageParams ref_image_params;

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
    apply_circular_axes_to_diffusion(sd_ctx, sd_vid_gen_params->circular_x, sd_vid_gen_params->circular_y);
    GenerationRequest request(sd_ctx, sd_vid_gen_params);
    const bool hires_chain_enabled = sd_vid_gen_params->hires_chain != nullptr &&
                                     sd_vid_gen_params->hires_chain_count > 0;
    if (hires_chain_enabled) {
        request.hires = sd_vid_gen_params->hires_chain[0];
        request.hires.enabled = true;
        request.resolve_hires();
        if (std::isfinite(request.hires.cfg)) {
            request.guidance.txt_cfg = request.hires.cfg;
            request.use_uncond = request.hires.cfg > 1.f;
        }
    }

    // LipDub's production recipe is a half-resolution from-noise pass followed
    // by the learned x2 latent upsampler and a full-resolution reference-aware
    // refine.  This is deliberately restricted to mode-0 relip; SDEdit and
    // guide-edit use the ordinary hires path.
    const bool relip_twostage = sd_vid_gen_params->lipdub_two_stage &&
                               sd_version_is_ltxav(sd_ctx->sd->version) &&
                               sd_vid_gen_params->v2v_mode == 0 &&
                               sd_vid_gen_params->control_frames != nullptr &&
                               sd_vid_gen_params->control_frames_size > 0;
    if (relip_twostage) {
        if (!request.hires.enabled || request.hires.upscaler != SD_HIRES_UPSCALER_MODEL ||
            strlen(SAFE_STR(request.hires.model_path)) == 0) {
            LOG_ERROR("LTX LipDub two_stage requires the LTX spatial upsampler");
            return false;
        }
        if (request.width % 64 != 0 || request.height % 64 != 0) {
            LOG_ERROR("LTX LipDub two_stage requires width and height divisible by 64 (got %dx%d)",
                      request.width, request.height);
            return false;
        }
        request.width /= 2;
        request.height /= 2;
        request.hires.enabled = true;
        request.hires.scale = 2.f;
        // Production's distilled LipDub stage-2 recipe is a fixed three-step
        // low-noise refine unless the caller deliberately supplied sigmas.
        if (request.hires.custom_sigmas == nullptr || request.hires.custom_sigmas_count <= 0) {
            static float k_lipdub_stage2_sigmas[] = {0.909375f, 0.725f, 0.421875f, 0.f};
            request.hires.custom_sigmas = k_lipdub_stage2_sigmas;
            request.hires.custom_sigmas_count = 4;
        }
        LOG_INFO("LTX LipDub two-stage: half-res base %dx%d then x2 reference-aware refine",
                 request.width, request.height);
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
    request.reconcile_cfg_pp_sampler(&plan.sample_method);
    auto latent_inputs_opt = prepare_video_generation_latents(sd_ctx, sd_vid_gen_params, &request);
    if (!latent_inputs_opt.has_value()) {
        return false;
    }
    ImageGenerationLatents latents = std::move(*latent_inputs_opt);
    latents.relip_twostage = relip_twostage;

    if (latents.v2v_sdedit) {
        const float requested_strength = sd_vid_gen_params->v2v_mode == 2 && sd_vid_gen_params->v2v_guide_strength > 0.f
                                             ? sd_vid_gen_params->v2v_guide_strength
                                             : sd_vid_gen_params->strength;
        const float strength = std::clamp(requested_strength, 0.f, 1.f);
        if (strength < 1.f && plan.sample_steps > 0 &&
            static_cast<int>(plan.sigmas.size()) == plan.sample_steps + 1) {
            int t_enc = static_cast<int>(plan.sample_steps * strength);
            t_enc = std::clamp(t_enc, 0, plan.sample_steps - 1);
            const int start = plan.sample_steps - t_enc - 1;
            if (start > 0 && start < static_cast<int>(plan.sigmas.size())) {
                plan.sigmas = std::vector<float>(plan.sigmas.begin() + start, plan.sigmas.end());
                plan.sample_steps = static_cast<int>(plan.sigmas.size()) - 1;
                LOG_INFO("LTXAV SDEdit: strength=%.2f -> %d sampling steps (t_enc=%d)",
                         strength,
                         plan.sample_steps,
                         t_enc);
            }
        }
    }

    // All VAE input-encoding (init/keyframe/ref/relip/v2v via encode_first_stage in
    // prepare_video_generation_latents) is done. Release the video-VAE compute buffer
    // now, BEFORE the gemma3 text encoder allocates its own ~819 MiB compute buffer in
    // prepare_video_generation_embeds — otherwise on i2v the two stack and gemma3 OOMs
    // (cudaMalloc "failed to allocate the compute buffer"). The VAE compute buffer
    // re-allocates lazily at the output decode, so this is free. Mirrors the per-branch
    // free prod does in the relip/v2v paths, hoisted to the one seam that covers every
    // encode sub-path.
    if (sd_ctx->sd->first_stage_model) {
        sd_ctx->sd->first_stage_model->free_compute_buffer();
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
    } else {
        LOG_INFO("generate_video %dx%dx%d",
                 request.width,
                 request.height,
                 request.frames);
    }

    // LTXAV_VAE_LAZY: on an LTXAV (a2v/i2v) render the video VAE (~1385 MB) + audio VAE
    // (~353 MB) are UNUSED between here and the final output decode, yet their encode-staged
    // GPU weights squat ~1.7 GB through the VRAM-peak DiT sample loop (and the latent
    // refine/upscale). All VAE input-encoding (init/keyframe/ref/relip/v2v via
    // encode_first_stage, plus the a2v audio encode) finished before prepare_video_generation_*
    // above; the video-VAE compute BUFFER was already dropped at the encode seam (:7831). What
    // remains resident is the ModelManager COMPUTE STAGING for those encodes: LTX VAE _compute
    // and the audio VAE encode both call GGMLRunner::compute(..., free_compute_params=false,
    // auto_free=false), so the staged weights stay in runner_param_tensors until the next
    // runner_done() (normally not until decode/tiling or the chain boundary). Release them now.
    //
    // Fork parity: the fork calls release_all_gpu_param_residency() on both VAEs here. The
    // rebuild has no resident_runtime_params_buffer — DiT/VAE residency is ModelManager-owned.
    // runner_done() is the equivalent freer: it funnels the encode-staged params through
    // release_compute_backend_params() -> finish_compute_backend_usage() ->
    // release_compute_staging_blocks(), freeing the GPU staging buffer while
    // free_compute_staging_block() swaps each tensor back to its params-backend home (host RAM
    // under --offload-to-cpu, or mmap/disk under Disk residency). Decode automatically re-homes
    // the VAE: prepare_execute_graph_weights() -> prepare_params() re-stages from that home. This
    // is the SAME operation reclaim_ltx_chain_window_gpu_memory() already runs on these runners
    // at every chain boundary (finish_runner -> runner_done), so the encode->free->decode-reload
    // cycle is already proven for multi-segment reuse. Under a no-offload recipe (params_backend
    // == compute_backend) nothing was ever staged, so runner_done() is a harmless VRAM no-op.
    //
    // Gate: unset/0 is byte-identical to today (no eviction). Prod sets LTXAV_VAE_LAZY=1
    // (docker-compose), which enables it. Trims the VAE backend's CUDA pool + drops the cuDNN
    // Conv3D reorder cache (keyed by the now-stale staged weight pointers) so the freed VRAM
    // becomes real DiT headroom, mirroring the chain-boundary reclaim.
    static const bool ltxav_vae_lazy = [] {
        const char* s = getenv("LTXAV_VAE_LAZY");
        return s != nullptr && s[0] == '1';
    }();
    if (ltxav_vae_lazy &&
        sd_version_is_ltxav(sd_ctx->sd->version) &&
        sd_ctx->sd->first_stage_model) {
        int64_t phase_t0 = ggml_time_ms();
        sd_ctx->sd->first_stage_model->runner_done();
        if (sd_ctx->sd->audio_vae_model) {
            sd_ctx->sd->audio_vae_model->runner_done();
        }
        if (ggml_backend_t vae_backend = sd_ctx->sd->backend_for(SDBackendModule::VAE);
            vae_backend != nullptr && ggml_backend_is_cuda(vae_backend)) {
            ggml_backend_synchronize(vae_backend);
            ggml_backend_cuda_trim_memory(vae_backend);
            // The staged VAE weights just moved (compute copies freed, home restored). The
            // cuDNN Conv3D reorder cache is keyed by stable identity, not by that address, so
            // it is not stale -- but this phase exists to hand VRAM back and those reorder
            // buffers are outside the VMM pool. Free them here; they re-reorder at decode.
            ggml_backend_cuda_release_cudnn_conv3d_weights();
        }
        LOG_INFO("LTXAV_VAE_LAZY: released encode-staged video+audio VAE GPU params + trimmed VAE pool "
                 "+ freed cuDNN Conv3D reorder weights before the DiT sample loop; re-staged at decode "
                 "(%.3fs)",
                 (ggml_time_ms() - phase_t0) * 1.0f / 1000);
    }

    int64_t latent_start = ggml_time_ms();
    int W                = request.width / request.vae_scale_factor;
    int H                = request.height / request.vae_scale_factor;
    int T                = static_cast<int>(latents.init_latent.shape()[2]);

    sd::Tensor<float> x_t   = latents.init_latent;
    sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);
    std::vector<float> avatar_input_wav;

    // LongCat Avatar's speech pathway is intentionally outside DiffusionParams:
    // Whisper creates the fixed per-frame inputs and the Avatar runner consumes
    // them from its audio projection blocks on every DiT forward.
    if (sd_version_is_longcat_avatar(sd_ctx->sd->version)) {
        auto avatar_model = std::dynamic_pointer_cast<LongCatAvatarModel>(sd_ctx->sd->diffusion_model);
        if (avatar_model == nullptr) {
            LOG_ERROR("LongCat Avatar model runner is unavailable");
            return false;
        }
        avatar_model->avatar.audio_first  = {};
        avatar_model->avatar.audio_latter = {};

        const char* audio_path = SAFE_STR(sd_vid_gen_params->audio_path);
        if (strlen(audio_path) > 0) {
            if (sd_ctx->sd->whisper_encoder_model == nullptr) {
                LOG_ERROR("LongCat Avatar audio_path requires audio_vae_path with the Whisper encoder GGUF");
                return false;
            }
            struct RunnerDoneOnExit {
                GGMLRunner* runner = nullptr;
                ~RunnerDoneOnExit() {
                    if (runner != nullptr) {
                        runner->runner_done();
                    }
                }
            } whisper_runner_done{sd_ctx->sd->whisper_encoder_model.get()};
            if (!LONGCAT_AUDIO::load_wav_16k_mono(audio_path, avatar_input_wav)) {
                return false;
            }

            LONGCAT_AUDIO::WhisperMel mel_extractor;
            int n_mel_frames = 0;
            std::vector<float> log_mel = mel_extractor.log_mel(avatar_input_wav, n_mel_frames);
            if (n_mel_frames <= 0) {
                LOG_ERROR("LongCat Avatar audio has no usable mel frames");
                return false;
            }

            const int n_mels = LONGCAT_AUDIO::WhisperMel::kNMels;
            sd::Tensor<float> mel({static_cast<int64_t>(n_mel_frames), static_cast<int64_t>(n_mels), 1});
            float* mel_data = mel.data();
            for (int m = 0; m < n_mels; ++m) {
                for (int t = 0; t < n_mel_frames; ++t) {
                    mel_data[static_cast<size_t>(m) * n_mel_frames + t] = log_mel[static_cast<size_t>(m) * n_mel_frames + t];
                }
            }

            auto whisper = sd_ctx->sd->whisper_encoder_model->compute(sd_ctx->sd->n_threads, mel);
            if (whisper.empty()) {
                LOG_ERROR("LongCat Avatar Whisper encoder failed");
                return false;
            }

            constexpr int avatar_fps = SD_AVATAR_NATIVE_FPS;
            LONGCAT_AUDIO::AudioWindowConfig audio_config;
            audio_config.fps = static_cast<float>(avatar_fps);
            const int video_length = std::max(1, static_cast<int>(avatar_input_wav.size() * avatar_fps / 16000));
            auto full_audio = LONGCAT_AUDIO::build_full_audio_emb(whisper, video_length, audio_config);
            sd::Tensor<float> first;
            sd::Tensor<float> latter;
            // The VAE may align a requested pixel-frame count before building
            // latents.  Audio projection must follow that actual latent
            // timeline, not the pre-alignment request, otherwise short Avatar
            // clips build incompatible audio tensors (for example 16 vs 4).
            const int audio_video_frames = sd_ctx->sd->latent_frames_to_video_frames(T);
            const int latent_audio_frames = LONGCAT_AUDIO::build_proj_inputs(full_audio,
                                                                               audio_video_frames,
                                                                               audio_config,
                                                                               first,
                                                                               latter,
                                                                               sd_vid_gen_params->audio_frame_offset);
            if (latent_audio_frames != T) {
                LOG_ERROR("LongCat Avatar audio window produced %d latent frames; video requires %d",
                          latent_audio_frames,
                          T);
                return false;
            }
            avatar_model->avatar.audio_first  = std::move(first);
            avatar_model->avatar.audio_latter = std::move(latter);
            LOG_INFO("LongCat Avatar audio conditioning: %d mel frames, %d video frames, %d latent frames",
                     n_mel_frames,
                     request.frames,
                     T);
        }
    }

    if (plan.high_noise_sample_steps > 0) {
        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation before high-noise sampling");
            return false;
        }
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
                                                           ref_image_params,
                                                           latents.denoise_mask,
                                                           latents.vace_context,
                                                           request.vace_strength,
                                                           latents.audio_length,
                                                           static_cast<float>(request.fps),
                                                           request.cache_params,
                                                           latents.video_positions,
                                                           sd::Tensor<float>(),
                                                           latents.audio_fixed);
        int64_t sampling_end          = ggml_time_ms();
        if (x_t_sampled.empty()) {
            LOG_ERROR("sampling(high noise) failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            return false;
        }

        x_t   = std::move(x_t_sampled);
        noise = {};
        LOG_INFO("sampling(high noise) completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
    }

    if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
        LOG_ERROR("cancelling generation before sampling");
        return false;
    }
    LOG_DEBUG("sample %dx%dx%d", W, H, T);
    int64_t sampling_start = ggml_time_ms();

    // Prompt Relay. Built once per shot from the caller's beats and the
    // positive encode's token->piece map; the per-window query-time tables are
    // filled in immediately before each sample() so a temporal tile is
    // penalised against GLOBAL time, not against its own frame zero.
    constexpr int kLtxTemporalScale = 8;
    sd::ltx_relay::Plan relay_plan;
    const bool relay_enabled =
        sd_version_is_ltxav(sd_ctx->sd->version) &&
        build_ltx_relay_plan(sd_vid_gen_params,
                             embeds.cond.c_token_pieces,
                             request.fps,
                             request.frames,
                             kLtxTemporalScale,
                             &relay_plan);
    const float relay_steps_frac = std::clamp(sd_vid_gen_params->relay_steps_frac > 0.f
                                                  ? sd_vid_gen_params->relay_steps_frac
                                                  : ltx_relay_env_float("LTX_RELAY_STEPS_FRAC", 1.f),
                                              0.f,
                                              1.f);

    auto sample_base_window = [&](const sd::Tensor<float>& window_latent,
                                  sd::Tensor<float> window_noise,
                                  const sd::Tensor<float>& window_mask,
                                  int window_audio_length,
                                  const sd::Tensor<float>& window_video_positions,
                                  const sd::Tensor<float>& window_audio_positions = {},
                                  int64_t window_latent_start                     = 0,
                                  int64_t window_audio_start                      = 0,
                                  // TASS source ids for THIS window (target tokens ++ reference
                                  // tokens). Null falls back to the full-pass vector.
                                  const std::vector<float>* window_source_ids     = nullptr) {
        const sd::ltx_relay::Plan* window_relay = nullptr;
        if (relay_enabled) {
            relay_plan.video_frame_time = ltx_relay_video_frame_times(window_video_positions,
                                                                      window_latent.shape()[0],
                                                                      window_latent.shape()[1],
                                                                      window_latent.shape()[2],
                                                                      window_latent_start,
                                                                      request.fps,
                                                                      kLtxTemporalScale);
            relay_plan.audio_frame_time.clear();
            if (window_audio_length > 0) {
                relay_plan.audio_frame_time.resize(static_cast<size_t>(window_audio_length));
                for (int frame = 0; frame < window_audio_length; ++frame) {
                    relay_plan.audio_frame_time[static_cast<size_t>(frame)] =
                        LTXV::audio_latent_start_time_sec(window_audio_start + frame);
                }
            }
            // Content-addressed, NOT a counter. The runner's mask cache outlives
            // this call, so a per-shot counter makes shot 2's first window
            // collide with shot 1's -- same revision, and on a uniform chain the
            // same L_q and L_k too -- and shot 2 silently renders shot 1's beats.
            relay_plan.revision = sd::ltx_relay::plan_fingerprint(relay_plan);
            window_relay        = &relay_plan;
        }
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
                                  ref_image_params,
                                  window_mask,
                                  latents.vace_context,
                                  request.vace_strength,
                                  window_audio_length,
                                  static_cast<float>(request.fps),
                                  request.cache_params,
                                  window_video_positions,
                                  window_audio_positions,
                                  latents.audio_fixed,
                                  sd_vid_gen_params->a2v_guidance,
                                  sd_vid_gen_params->a2v_ramp_end,
                                  window_relay,
                                  relay_steps_frac,
                                  latents.ref_video_x,
                                  window_source_ids != nullptr
                                      ? window_source_ids
                                      : (latents.video_source_ids.empty() ? nullptr : &latents.video_source_ids),
                                  latents.tass_phase_scale);
    };

    sd::Tensor<float> final_latent;
    const char* base_window_env = std::getenv("LTX_BASE_TEMPORAL_WINDOW");
    // This is deliberately the small, safe subset of production temporal tiling:
    // supplied fixed audio and unconditioned video only.  I2V/keyframe/continuation
    // guides and V2V need appended-reference position handling and stay full-pass.
    //
    // TASS references DO tile.  The reference latent is a separate tensor from the
    // target, so a tile only has to rebuild the per-token vectors: its own frame
    // range's positions, then the same reference block re-appended (see
    // `base_window_tass`).  What it may NOT do is inherit a position layout some
    // other conditioning path committed to, which is what `tass_positions_only`
    // gates -- with references present the positions are non-empty by construction,
    // so the plain "positions must be empty" test has to be relaxed to "positions
    // are nothing but the implicit t2v grid plus the reference tail".
    // An i2v opener is a single FROZEN frame pinned at latent index 0 with mask 0 -- it lives in
    // the target grid, not in an appended block, and the loop below already slices the denoise
    // mask per tile. So tile 0 carries the pin and later tiles simply continue from the frozen
    // overlap, exactly as a continuation shot does across a segment boundary. It therefore does
    // NOT need to block windowing. (An end_image pins near the last frame and appends a guide
    // block instead, so it is covered by the guide path below.)
    const bool base_window_unconditioned =
        sd_vid_gen_params->keyframes_size == 0 && sd_vid_gen_params->cont_latent == nullptr &&
        sd_vid_gen_params->end_cont_latent == nullptr;
    // TASS reference block present at all. `base_window_tass` is the stricter form used by the
    // PLAIN path, which regenerates its target rows from scratch and so may only do that when
    // nothing else owns the layout. The guide path below EXTRACTS its rows verbatim instead, so
    // it can carry a reference block safely even though tass_positions_only is false there.
    const bool base_window_tass_refs = !latents.ref_video_x.empty() && !latents.ref_grids.empty();
    const bool base_window_tass      = base_window_tass_refs && latents.tass_positions_only;
    const bool base_window_positions_plain =
        latents.video_positions.empty() || (base_window_tass && !latents.video_positions.empty());
    // Where a tile's reference block sits on the timeline.  Default: the tile's OWN
    // first frame, so the sheet keeps the zero temporal offset it was trained with
    // for every tile.  Pinning it at global frame 0 instead (LTX_TASS_WINDOW_REF_ABS=1)
    // puts it seconds in the past for tiles 1..N, a relationship the checkpoint never
    // saw; kept as an env switch so the two can be A/B'd without a rebuild.
    const bool tass_ref_abs_origin = [] {
        const char* env = std::getenv("LTX_TASS_WINDOW_REF_ABS");
        return env != nullptr && env[0] != '\0' && std::string(env) != "0";
    }();
    auto tass_ref_frame_origin = [&](int64_t tile_start) -> int64_t {
        return tass_ref_abs_origin ? 0 : tile_start;
    };
    const bool base_window_fixed_audio =
        base_window_unconditioned && latents.audio_fixed && base_window_positions_plain && !latents.v2v_sdedit;
    const bool base_window_plain_t2v =
        base_window_unconditioned && !latents.audio_fixed && latents.audio_length == 0 && base_window_positions_plain &&
        !latents.v2v_sdedit;
    // A tail-appended GUIDE block -- the continuation motion tail and/or Director keyframes, which
    // apply_ltxav_video_guide/append_ltxav_keyframes concatenate past the target grid -- can ride
    // EVERY tile at a bounded token cost, so a conditioned render tiles instead of taking one
    // full-length pass whose VRAM grows with segment length. The guide's position rows are
    // EXTRACTED VERBATIM from latents.video_positions rather than rebuilt, so this is agnostic to
    // the guide's RoPE convention (causal continuation tail, keyframe instants, or a mix).
    //
    // Restricted to supplied-fixed-audio renders on purpose: a plain t2v (no drive audio) that
    // tiles is a known-bad regime in how the AV checkpoint handles the video+audio pair, and is
    // not worth stressing.
    //
    // Extraction invariant: video_positions must OPEN with the main grid, one row block per latent
    // frame over (target ++ guide). Anything past that is a TASS reference tail, which each tile
    // rebuilds for itself, so it is allowed -- the guide rows are sliced by explicit index and
    // never overlap it.
    const int64_t base_guide_frames  = latents.video_conditioning_frame_count;
    const int64_t base_target_frames = latents.video_target_frame_count;
    const int64_t base_grid_pos_rows = x_t.shape()[0] * x_t.shape()[1] * x_t.shape()[2];
    const bool base_window_guide_layout_ok =
        base_guide_frames > 0 && base_target_frames > 0 &&
        base_target_frames + base_guide_frames == x_t.shape()[2] && !latents.video_positions.empty() &&
        latents.video_positions.shape()[2] >= base_grid_pos_rows &&
        (latents.video_positions.shape()[2] == base_grid_pos_rows || base_window_tass_refs);
    const bool base_window_guide =
        base_window_guide_layout_ok && latents.audio_fixed && !latents.v2v_sdedit &&
        sd_vid_gen_params->v2v_mode == 0 && sd_vid_gen_params->control_frames_size == 0 &&
        !latents.relip_twostage;
    const bool base_temporal_windowing =
        base_window_env != nullptr && base_window_env[0] != '\0' && std::string(base_window_env) != "0" &&
        sd_version_is_ltxav(sd_ctx->sd->version) &&
        (base_window_fixed_audio || base_window_plain_t2v || base_window_guide) &&
        latents.vace_context.empty() && (x_t.dim() == 4 || (x_t.dim() == 5 && x_t.shape()[4] == 1)) &&
        x_t.shape()[2] > 1;
    if (!base_temporal_windowing) {
        final_latent = sample_base_window(x_t,
                                          std::move(noise),
                                          latents.denoise_mask,
                                          latents.audio_length,
                                          latents.video_positions,
                                          sd::Tensor<float>());
    } else {
        int temporal_window = std::max(2, std::atoi(base_window_env));
        int temporal_overlap = 4;
        if (const char* overlap_env = std::getenv("LTX_BASE_TEMPORAL_OVERLAP"); overlap_env != nullptr) {
            temporal_overlap = std::max(1, std::atoi(overlap_env));
        }
        const int64_t total_frames = x_t.shape()[2];
        // Only the TARGET frames are windowed; a tail-appended guide is re-attached to every tile
        // and carried through unchanged at the end.
        const int64_t windowed_frames = base_window_guide ? base_target_frames : total_frames;
        const int64_t guide_frames    = base_window_guide ? base_guide_frames : 0;
        const int64_t latent_channels = sd_ctx->sd->get_latent_channel();
        const bool has_fixed_audio = latents.audio_fixed && latents.audio_length > 0 &&
                                     x_t.shape()[3] > latent_channels;
        temporal_window = std::clamp(temporal_window, 2, static_cast<int>(std::max<int64_t>(2, windowed_frames)));
        temporal_overlap = std::clamp(temporal_overlap, 1, temporal_window - 1);
        if (windowed_frames <= temporal_window) {
            final_latent = sample_base_window(x_t,
                                              std::move(noise),
                                              latents.denoise_mask,
                                              latents.audio_length,
                                              latents.video_positions,
                                              sd::Tensor<float>());
        } else {
            auto full_video = sd::ops::slice(x_t, 3, 0, latent_channels);
            auto full_noise = sd::ops::slice(noise, 3, 0, latent_channels);
            sd::Tensor<float> full_audio;
            if (has_fixed_audio) {
                full_audio = unpack_ltxav_audio_latent(x_t, latents.audio_length, static_cast<int>(latent_channels));
            }
            if (has_fixed_audio && full_audio.empty()) {
                LOG_ERROR("LTX base temporal-window could not unpack fixed driving audio latent");
            } else {
                sd::Tensor<float> full_video_mask;
                if (!latents.denoise_mask.empty()) {
                    full_video_mask = sd::ops::slice(latents.denoise_mask, 3, 0, latent_channels);
                }
                sd::Tensor<float> video_result(full_video.shape());
                video_result.fill_(0.0f);
                const int64_t plane = full_video.shape()[0] * full_video.shape()[1];
                const int64_t stride = temporal_window - temporal_overlap;
                const int64_t audio_rate = 25;
                // The guide block: latent frames, mask and position rows, all sliced once and
                // re-attached to every tile. Positions are taken verbatim (never rebuilt).
                const int64_t pos_row_stride = x_t.shape()[0] * x_t.shape()[1];
                sd::Tensor<float> guide_grid, guide_mask_grid, guide_positions;
                if (guide_frames > 0) {
                    guide_grid = sd::ops::slice(full_video, 2, windowed_frames, total_frames);
                    if (!full_video_mask.empty()) {
                        guide_mask_grid = sd::ops::slice(full_video_mask, 2, windowed_frames, total_frames);
                    }
                    guide_positions = sd::ops::slice(latents.video_positions, 2,
                                                     pos_row_stride * windowed_frames,
                                                     pos_row_stride * total_frames);
                }
                int64_t produced_end = 0;
                int tile_index = 0;
                for (int64_t start = 0;; start += stride, ++tile_index) {
                    const int64_t tile_start = start + temporal_window >= windowed_frames
                                                   ? std::max<int64_t>(0, windowed_frames - temporal_window)
                                                   : start;
                    const int64_t end = std::min<int64_t>(windowed_frames, tile_start + temporal_window);
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
                    int64_t audio_start = 0;
                    int64_t audio_end = 0;
                    sd::Tensor<float> audio_tile;
                    if (has_fixed_audio) {
                        const int64_t pixel_start = tile_start * 8;
                        const int64_t pixel_end = std::min<int64_t>(request.frames, (end - 1) * 8 + 1);
                        audio_start = std::clamp<int64_t>((pixel_start * audio_rate) / request.fps,
                                                          0,
                                                          full_audio.shape()[1]);
                        audio_end = std::clamp<int64_t>((pixel_end * audio_rate + request.fps - 1) / request.fps,
                                                        audio_start + 1,
                                                        full_audio.shape()[1]);
                        audio_tile = sd::ops::slice(full_audio, 1, audio_start, audio_end);
                    }
                    auto video_mask = full_video_mask.empty()
                                          ? make_ltxav_video_denoise_mask(video_tile, 1.0f)
                                          : sd::ops::slice(full_video_mask, 2, tile_start, end);
                    if (frozen > 0) {
                        float* mask_data = video_mask.data();
                        const int64_t mask_channels = video_mask.shape()[3];
                        for (int64_t channel = 0; channel < mask_channels; ++channel) {
                            for (int64_t local = 0; local < frozen; ++local) {
                                std::fill_n(mask_data + plane * (local + length * channel), plane, 0.0f);
                            }
                        }
                    }
                    // Re-attach the frozen guide block to THIS tile. Done after the frozen-overlap
                    // fixups above so their `length`-based indexing stays right.
                    if (guide_frames > 0) {
                        video_tile = sd::ops::concat(video_tile, guide_grid, 2);
                        noise_tile = sd::ops::concat(noise_tile,
                                                     sd::ops::slice(full_noise, 2, windowed_frames, total_frames), 2);
                        video_mask = sd::ops::concat(video_mask,
                                                     guide_mask_grid.empty()
                                                         ? make_ltxav_video_denoise_mask(guide_grid, 0.0f)
                                                         : guide_mask_grid,
                                                     2);
                    }
                    auto latent_tile = has_fixed_audio
                                           ? pack_ltxav_audio_and_video_latents(video_tile, audio_tile)
                                           : video_tile;
                    auto packed_noise = has_fixed_audio
                                            ? pack_ltxav_audio_and_video_latents(noise_tile, audio_tile)
                                            : noise_tile;
                    auto mask_tile = has_fixed_audio
                                         ? pack_ltxav_audio_and_video_denoise_mask(video_mask,
                                                                                   video_tile,
                                                                                   audio_tile,
                                                                                   0.0f)
                                         : video_mask;
                    // With a guide block the tile's rows are taken VERBATIM from the full layout
                    // (target slice ++ the guide rows), so the tile is byte-identical to the full
                    // pass for these absolute frames and the guide keeps whatever RoPE convention
                    // built it. Without one, the target rows are rebuilt for the tile's range.
                    sd::Tensor<float> video_positions;
                    if (guide_frames > 0) {
                        video_positions = sd::ops::concat(
                            sd::ops::slice(latents.video_positions, 2,
                                           pos_row_stride * tile_start, pos_row_stride * end),
                            guide_positions, 2);
                    } else {
                        video_positions = build_ltxav_window_video_positions(video_tile.shape()[0],
                                                                             video_tile.shape()[1],
                                                                             tile_start,
                                                                             length,
                                                                             request.fps,
                                                                             request.vae_scale_factor);
                    }
                    // TASS overlap references ride along with EVERY tile.  The
                    // reference latent itself is tile-invariant (it is a separate
                    // tensor, never sliced), so all a tile rebuilds is the per-token
                    // tail: the same reference grids, tagged with the same source ids,
                    // placed on THIS tile's first frame.  Handing them only to tile 0
                    // would leave every later tile holding the identity through four
                    // frozen overlap frames alone -- which is the drift this whole
                    // feature exists to remove -- and would silently switch the model
                    // between a tagged and an untagged graph mid-shot.
                    std::vector<float> tile_source_ids;
                    if (base_window_tass || (guide_frames > 0 && base_window_tass_refs)) {
                        // Row count covers the target slice PLUS any guide block, so the source-id
                        // vector lines up token-for-token with the positions built above.
                        video_positions = build_ltxv_tass_ref_video_positions(video_positions,
                                                                              video_tile.shape()[0],
                                                                              video_tile.shape()[1],
                                                                              length + guide_frames,
                                                                              latents.ref_grids,
                                                                              request.fps,
                                                                              request.vae_scale_factor,
                                                                              kLtxTemporalScale,
                                                                              &tile_source_ids,
                                                                              tass_ref_frame_origin(tile_start));
                    }
                    auto audio_positions = has_fixed_audio
                                               ? build_ltxav_window_audio_positions(audio_start, audio_tile.shape()[1])
                                               : sd::Tensor<float>();
                    LOG_INFO("LTX base temporal-window tile %d: latent [%lld,%lld) of %lld, frozen-overlap=%lld, "
                             "guide=%lld, audio [%lld,%lld)%s",
                             tile_index,
                             (long long)tile_start,
                             (long long)end,
                             (long long)windowed_frames,
                             (long long)frozen,
                             (long long)guide_frames,
                             (long long)audio_start,
                             (long long)audio_end,
                             base_window_tass
                                 ? (", +" + std::to_string(latents.ref_video_x.shape()[0] *
                                                           latents.ref_video_x.shape()[1] *
                                                           latents.ref_video_x.shape()[2]) +
                                    " TASS ref token(s) @ latent frame " +
                                    std::to_string(tass_ref_frame_origin(tile_start)))
                                       .c_str()
                                 : "");
                    auto tile = sample_base_window(latent_tile,
                                                   std::move(packed_noise),
                                                   mask_tile,
                                                   has_fixed_audio ? static_cast<int>(audio_tile.shape()[1]) : 0,
                                                   video_positions,
                                                   audio_positions,
                                                   tile_start,
                                                   audio_start,
                                                   tile_source_ids.empty() ? nullptr : &tile_source_ids);
                    if (tile.empty()) {
                        final_latent = {};
                        break;
                    }
                    auto refined_video = sd::ops::slice(tile, 3, 0, latent_channels);
                    const float* src = refined_video.data();
                    float* dst = video_result.data();
                    // The sampled tile is [target slice ++ guide], so its per-channel frame stride
                    // includes the guide; only the target frames are emitted.
                    const int64_t tile_frames = length + guide_frames;
                    for (int64_t local = frozen; local < length; ++local) {
                        for (int64_t channel = 0; channel < latent_channels; ++channel) {
                            std::memcpy(dst + plane * (tile_start + local + total_frames * channel),
                                        src + plane * (local + tile_frames * channel),
                                        static_cast<size_t>(plane) * sizeof(float));
                        }
                    }
                    produced_end = std::max(produced_end, end);
                    if (end == windowed_frames) {
                        // Carry the frozen guide frames through unchanged so the emitted latent
                        // keeps its [target ++ guide] shape for the downstream crop keyed on
                        // video_conditioning_frame_count.
                        if (guide_frames > 0) {
                            const float* g = full_video.data();
                            float* rd = video_result.data();
                            for (int64_t channel = 0; channel < latent_channels; ++channel) {
                                for (int64_t local = 0; local < guide_frames; ++local) {
                                    const int64_t f = windowed_frames + local;
                                    std::memcpy(rd + plane * (f + total_frames * channel),
                                                g + plane * (f + total_frames * channel),
                                                static_cast<size_t>(plane) * sizeof(float));
                                }
                            }
                        }
                        final_latent = has_fixed_audio
                                           ? pack_ltxav_audio_and_video_latents(video_result, full_audio)
                                           : std::move(video_result);
                        break;
                    }
                }
            }
        }
    }

    int64_t sampling_end = ggml_time_ms();
    if (final_latent.empty()) {
        LOG_ERROR("sampling failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        return false;
    }
    LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
    // CHAINING ACROSS A REFINE. The next segment seeds and samples at BASE resolution, so its
    // continuation tail must be the base latent -- captured here, before the hires block replaces
    // final_latent with the upscaled one. Exporting the upscaled latent instead makes segment 2 of
    // any chain that carries a hires_chain die on "continuation latent shape mismatch". The
    // decoded frames still come from the upscaled latent; spatial upscale preserves the frame
    // count, so the caller's overlap bookkeeping is unchanged. Deep copy: final_latent is
    // reassigned by the hires pass.
    sd::Tensor<float> chain_base_latent;
    if (latent_upscale_enabled && final_latent_out != nullptr) {
        chain_base_latent = final_latent;
    }
    if (latent_upscale_enabled) {
        // The base relip pass includes frozen source-reference frames in its
        // sampler grid.  The learned spatial upsampler must only see the
        // generated target; the refine below re-encodes and re-attaches a
        // fresh full-resolution reference block.
        if (latents.relip_twostage && latents.video_conditioning_frame_count > 0) {
            const int latent_channels = sd_ctx->sd->get_latent_channel();
            sd::Tensor<float> target_video = final_latent;
            sd::Tensor<float> audio_latent;
            if (final_latent.shape()[3] > latent_channels) {
                target_video = sd::ops::slice(final_latent, 3, 0, latent_channels);
                audio_latent = unpack_ltxav_audio_latent(final_latent, latents.audio_length, latent_channels);
            }
            target_video = sd::ops::slice(target_video, 2, 0, latents.video_target_frame_count);
            final_latent = audio_latent.empty()
                               ? std::move(target_video)
                               : pack_ltxav_audio_and_video_latents(target_video, audio_latent);
            LOG_INFO("LTX LipDub two-stage: removed %lld base reference latent frame(s) before x2 upscale",
                     (long long)latents.video_conditioning_frame_count);
        }
        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation before latent upscale");
            return false;
        }
        int64_t upscale_start             = ggml_time_ms();
        sd::Tensor<float> upscaled_latent = upscale_ltx_spatial_video_latent(sd_ctx,
                                                                             request.hires.model_path,
                                                                             final_latent,
                                                                             latents.audio_length);
        int64_t upscale_end               = ggml_time_ms();
        if (upscaled_latent.empty()) {
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
        sd::Tensor<float> hires_denoise_mask;
        sd::Tensor<float> hires_video_positions;
        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation before latent upscale refine");
            return false;
        }
        if (!apply_ltxv_refine_image_conditioning(sd_ctx,
                                                  sd_vid_gen_params,
                                                  hires_request,
                                                  latents,
                                                  &x_t,
                                                  &hires_denoise_mask,
                                                  &hires_video_positions)) {
            return false;
        }
        noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);

        W                                   = hires_request.width / hires_request.vae_scale_factor;
        H                                   = hires_request.height / hires_request.vae_scale_factor;
        T                                   = static_cast<int>(x_t.shape()[2]);
        sample_method_t hires_sample_method = request.hires.sample_method == SAMPLE_METHOD_COUNT
                                                  ? plan.sample_method
                                                  : request.hires.sample_method;
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

        sampling_start = ggml_time_ms();
        sd_guidance_params_t hires_guidance = sd_vid_gen_params->sample_params.guidance;
        if (std::isfinite(request.hires.cfg)) {
            hires_guidance.txt_cfg = request.hires.cfg;
        }
        final_latent   = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                            true,
                                            x_t,
                                            std::move(noise),
                                            embeds.cond,
                                          hires_request.use_uncond ? embeds.uncond : SDCondition(),
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
                                            ref_image_params,
                                            hires_denoise_mask,
                                            sd::Tensor<float>(),
                                            hires_request.vace_strength,
                                            latents.audio_length,
                                            static_cast<float>(hires_request.fps),
                                            hires_request.cache_params,
                                            hires_video_positions,
                                            sd::Tensor<float>(),
                                            latents.audio_fixed);
        sampling_end   = ggml_time_ms();
        if (final_latent.empty()) {
            LOG_ERROR("sampling(latent upscale) failed after %.2fs",
                      (sampling_end - sampling_start) * 1.0f / 1000);
            return false;
        }
        LOG_INFO("sampling(latent upscale) completed, taking %.2fs",
                 (sampling_end - sampling_start) * 1.0f / 1000);
    }

    // Every later stage is the same established LTX latent-upscale + SDEdit
    // operation, with its own model/sigma/sampler configuration.  Stage zero
    // above deliberately remains the legacy single-hires path.
    if (hires_chain_enabled) {
        for (int stage_index = 1; stage_index < sd_vid_gen_params->hires_chain_count; ++stage_index) {
            const sd_hires_params_t& stage = sd_vid_gen_params->hires_chain[stage_index];
            if (stage.upscaler != SD_HIRES_UPSCALER_MODEL || strlen(SAFE_STR(stage.model_path)) == 0) {
                LOG_ERROR("LTX hires_chain stage %d requires a model upscaler", stage_index);
                return false;
            }
            if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation before hires_chain stage %d", stage_index);
                return false;
            }
            sd::Tensor<float> stage_latent = upscale_ltx_spatial_video_latent(sd_ctx,
                                                                                stage.model_path,
                                                                                final_latent,
                                                                                latents.audio_length);
            if (stage_latent.empty()) {
                LOG_ERROR("LTX hires_chain stage %d upscale failed", stage_index);
                return false;
            }
            GenerationRequest stage_request = hires_request;
            stage_request.hires = stage;
            stage_request.width = static_cast<int>(stage_latent.shape()[0]) * stage_request.vae_scale_factor;
            stage_request.height = static_cast<int>(stage_latent.shape()[1]) * stage_request.vae_scale_factor;
            stage_request.frames = sd_ctx->sd->latent_frames_to_video_frames(static_cast<int>(stage_latent.shape()[2]));
            const int latent_channels = sd_ctx->sd->get_latent_channel();
            sd::Tensor<float> video_latent = stage_latent;
            sd::Tensor<float> audio_latent;
            if (latents.audio_length > 0 && stage_latent.shape()[3] > latent_channels) {
                video_latent = sd::ops::slice(stage_latent, 3, 0, latent_channels);
                audio_latent = unpack_ltxav_audio_latent(stage_latent, latents.audio_length, latent_channels);
                if (audio_latent.empty()) {
                    LOG_ERROR("LTX hires_chain stage %d could not separate audio latent", stage_index);
                    return false;
                }
            }
            sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video_latent, 1.f);
            sd::Tensor<float> stage_mask = audio_latent.empty()
                                                ? video_mask
                                                : pack_ltxav_audio_and_video_denoise_mask(video_mask,
                                                                                           video_latent,
                                                                                           audio_latent,
                                                                                           latents.audio_fixed ? 0.f : 1.f);
            const sample_method_t stage_method = stage.sample_method == SAMPLE_METHOD_COUNT
                                                     ? plan.sample_method
                                                     : stage.sample_method;
            int scheduler_steps = 0;
            std::vector<float> sigmas = make_hires_sigma_schedule(
                sd_ctx, stage, sd_vid_gen_params->sample_params, stage_method, stage.steps,
                sd_ctx->sd->get_image_seq_len(stage_request.height, stage_request.width) *
                    static_cast<int>(stage_latent.shape()[2]),
                &scheduler_steps);
            if (sigmas.size() < 2) {
                LOG_ERROR("LTX hires_chain stage %d has no usable SDEdit sigma schedule", stage_index);
                return false;
            }
            sd_guidance_params_t stage_guidance = sd_vid_gen_params->sample_params.guidance;
            if (std::isfinite(stage.cfg)) stage_guidance.txt_cfg = stage.cfg;
            const bool stage_use_uncond = request.use_uncond || stage_guidance.txt_cfg > 1.f ||
                                          stage_method == EULER_CFG_PP_SAMPLE_METHOD ||
                                          stage_method == EULER_A_CFG_PP_SAMPLE_METHOD;
            final_latent = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                               true,
                                               stage_latent,
                                               sd::Tensor<float>::randn_like(stage_latent, sd_ctx->sd->rng),
                                               embeds.cond,
                                               stage_use_uncond ? embeds.uncond : SDCondition(),
                                               embeds.img_uncond,
                                               sd::Tensor<float>(),
                                               0.f,
                                               stage_guidance,
                                               resolve_eta(sd_ctx, sd_vid_gen_params->sample_params.eta, stage_method),
                                               sd_vid_gen_params->sample_params.shifted_timestep,
                                               stage_method,
                                               sd_ctx->sd->is_flow_denoiser(),
                                               plan.extra_sample_args,
                                               sigmas,
                                               std::vector<sd::Tensor<float>>{},
                                               ref_image_params,
                                               stage_mask,
                                               sd::Tensor<float>(),
                                               stage_request.vace_strength,
                                               latents.audio_length,
                                               static_cast<float>(stage_request.fps),
                                               stage_request.cache_params,
                                               sd::Tensor<float>(),
                                               sd::Tensor<float>(),
                                               latents.audio_fixed);
            if (final_latent.empty()) {
                LOG_ERROR("LTX hires_chain stage %d refine failed", stage_index);
                return false;
            }
            hires_request = std::move(stage_request);
            LOG_INFO("LTX hires_chain stage %d completed (%d steps)", stage_index, scheduler_steps);
        }
    }

    int64_t latent_end = ggml_time_ms();
    LOG_INFO("generating latent video completed, taking %.2fs", (latent_end - latent_start) * 1.0f / 1000);

    sd_audio_t* generated_audio = nullptr;
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        latents.audio_length > 0 &&
        sd_ctx->sd->audio_vae_model != nullptr) {
        if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation before audio decode");
            return false;
        }
        int64_t audio_latent_decode_start = ggml_time_ms();

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
    if (sd_version_is_longcat_avatar(sd_ctx->sd->version) && !avatar_input_wav.empty()) {
        constexpr uint32_t sample_rate = 16000;
        const int output_fps = request.fps > 0 ? request.fps : 25;
        const size_t start = std::min(avatar_input_wav.size(),
                                      static_cast<size_t>(std::max(0, sd_vid_gen_params->audio_frame_offset)) * sample_rate / 25);
        const size_t requested_samples = static_cast<size_t>(request.frames) * sample_rate / output_fps;
        const size_t count = std::min(requested_samples, avatar_input_wav.size() - start);
        generated_audio = (sd_audio_t*)calloc(1, sizeof(sd_audio_t));
        if (generated_audio == nullptr) {
            LOG_WARN("LongCat Avatar could not allocate output audio; continuing with video");
        } else {
            generated_audio->sample_rate  = sample_rate;
            generated_audio->channels     = 1;
            generated_audio->sample_count = count;
            generated_audio->data         = (float*)malloc(count * sizeof(float));
            if (generated_audio->data == nullptr) {
                free(generated_audio);
                generated_audio = nullptr;
                LOG_WARN("LongCat Avatar could not allocate output audio samples; continuing with video");
            } else {
                std::memcpy(generated_audio->data, avatar_input_wav.data() + start, count * sizeof(float));
            }
        }
    }

    if (latents.video_conditioning_frame_count > 0) {
        int64_t target_frames = latents.video_target_frame_count > 0 ? latents.video_target_frame_count
                                                                     : final_latent.shape()[2] - latents.video_conditioning_frame_count;
        final_latent          = sd::ops::slice(final_latent, 2, 0, target_frames);
        if (!chain_base_latent.empty()) {
            chain_base_latent = sd::ops::slice(chain_base_latent, 2, 0, target_frames);
        }
    }

    if (latents.ref_image_num > 0) {
        final_latent = sd::ops::slice(final_latent, 2, latents.ref_image_num, final_latent.shape()[2]);
        if (!chain_base_latent.empty()) {
            chain_base_latent =
                sd::ops::slice(chain_base_latent, 2, latents.ref_image_num, chain_base_latent.shape()[2]);
        }
    }

    if (final_latent_out != nullptr) {
        // Chain on the BASE latent when a refine ran; otherwise the single-stage result IS the base.
        const sd::Tensor<float>& exported_latent = chain_base_latent.empty() ? final_latent : chain_base_latent;
        if (exported_latent.dim() < 4 || exported_latent.dim() > 5) {
            LOG_ERROR("cannot export video latent with %d dimensions", exported_latent.dim());
            free_sd_audio(generated_audio);
            return false;
        }
        const size_t bytes = static_cast<size_t>(exported_latent.numel()) * sizeof(float);
        float* exported = static_cast<float*>(malloc(bytes));
        if (exported == nullptr) {
            LOG_ERROR("failed to allocate %zu-byte video latent export", bytes);
            free_sd_audio(generated_audio);
            return false;
        }
        std::memcpy(exported, exported_latent.data(), bytes);
        *final_latent_out = exported;
        if (latent_width_out != nullptr) {
            *latent_width_out = static_cast<int>(exported_latent.shape()[0]);
        }
        if (latent_height_out != nullptr) {
            *latent_height_out = static_cast<int>(exported_latent.shape()[1]);
        }
        if (latent_frames_out != nullptr) {
            *latent_frames_out = static_cast<int>(exported_latent.shape()[2]);
        }
        if (latent_channels_out != nullptr) {
            *latent_channels_out = static_cast<int>(exported_latent.shape()[3]);
        }
    }

    if (sd_ctx->sd->get_cancel_flag() == SD_CANCEL_ALL) {
        LOG_ERROR("cancelling generation before video decode");
        free_sd_audio(generated_audio);
        return false;
    }
    auto result = decode_video_outputs(sd_ctx, latent_upscale_enabled ? hires_request : request, final_latent, num_frames_out);
    if (result == nullptr) {
        free_sd_audio(generated_audio);
        return false;
    }

    // Resolve -- but do NOT apply -- the reference head-frame trim. The cut belongs to whoever
    // owns the output timeline: generate_video() applies it directly, while generate_video_chain()
    // folds it into that shot's head drop so the durable bank keeps holding the shot AS RENDERED.
    if (reference_head_trim_out != nullptr) {
        *reference_head_trim_out = ltxv_resolve_reference_head_trim(
            sd_vid_gen_params, latents, sd_version_is_ltxav(sd_ctx->sd->version), *num_frames_out);
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
    int reference_head_trim = 0;
    if (!generate_video_ex(sd_ctx,
                           sd_vid_gen_params,
                           frames_out,
                           num_frames_out,
                           audio_out,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           &reference_head_trim)) {
        return false;
    }
    // Single-shot render: this call owns the whole output timeline, so the trim lands here on both
    // the frames and the audio at once.
    ltxv_apply_reference_head_trim(frames_out,
                                   num_frames_out,
                                   audio_out != nullptr ? *audio_out : nullptr,
                                   reference_head_trim,
                                   sd_vid_gen_params != nullptr ? sd_vid_gen_params->fps : 0);
    // Reclaim this job's GPU working set, exactly as generate_video_chain() does at its end
    // (see the note there). The two are alternative branches of the SAME server route
    // (routes_longcat.cpp: `segment_count == 1 ? generate_video : generate_video_chain`), so
    // without this a one-segment render leaves the last window's committed VMM high-water and
    // the cuDNN conv3d reorder cache resident into the next job on a persistent worker — the
    // exact starvation the chain path added its own end-of-job reclaim to prevent.
    //
    // ⚠️ SCOPED TO LTX ON PURPOSE. generate_video() is also the entry point for wan-vace and
    // longcat-avatar, and this reclaim is the prime suspect in a reproducible mid-chain cuBLAS
    // failure (see reclaim_ltx_chain_window_gpu_memory()'s header). Widening it to every video
    // model would be trading a latent leak for a live crash class in services that never had it.
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr && sd_version_is_ltxav(sd_ctx->sd->version)) {
        sd_ctx->sd->reclaim_ltx_chain_window_gpu_memory();
    }
    return true;
}

static sd_image_t copy_video_frame(const sd_image_t& source) {
    sd_image_t copy = source;
    const size_t bytes = static_cast<size_t>(source.width) * source.height * source.channel;
    copy.data = static_cast<uint8_t*>(malloc(bytes));
    if (copy.data != nullptr && source.data != nullptr) {
        std::memcpy(copy.data, source.data, bytes);
    }
    return copy;
}

// CONTENT-ADAPTIVE SEAM TRIM. A derived trim is only ever an ESTIMATE of how much of a continuation
// shot repeats the prior one -- the true settling length varies per clip, and trimming too long
// throws away newly rendered frames, which reads as the picture JUMPING FORWARD at the join.
// Instead, search a band around the predicted centre for the frame whose downsampled luma best
// matches the prior segment's last kept frame: that frame is the smoothest continuation.
//
// READ THE ERROR, NOT JUST THE ARGMIN -- see the confidence gate below. An argmin always exists; a
// MATCH does not, and on a seam where the shot genuinely CHANGES SCENE none of the candidates match
// at all. Measured 2026-07-30 on an alley->beach crane the whole band sat at MAE 41.3..45.1 out of
// 255, a 9% spread with no minimum, and the argmin was a two-way tie. That noise used to be applied
// as a trim (24 here; 10 on the same recipe without drive audio), which is where the wandering came
// from. Below the gate the answer stands; above it the caller's centre is returned unchanged.
// LTXAV_TRIM_DEBUG=1 prints the whole curve, the band median, and the verdict.
//
// The centre is whatever the caller predicted for this seam, and the band is only [-8, +10] around
// it, so the answer is NOT independent of the prediction -- a prediction that is off by more than
// the band rails the search at an edge. (Historically the centre was 8*K; centring on overlap_px
// let the band rail-clip as K shrank, which produced the wild per-segment trims 22/17/8 at K=2.)
//
// Cheap by construction: a 32x18 luma grid per candidate, ~19 candidates, no full-res compare.
static int ltxav_auto_trim_drop(const sd_image_t& prev_last,
                                const sd_image_t* frames,
                                int n_frames,
                                int centre) {
    if (frames == nullptr || n_frames <= 0 || prev_last.data == nullptr) {
        return centre;
    }
    const int lo = std::max(1, centre - 8);
    const int hi = std::min(n_frames - 2, centre + 10);
    if (hi <= lo) {
        return std::min(centre, std::max(0, n_frames - 1));
    }
    constexpr int GW = 32, GH = 18;
    auto grid = [](const sd_image_t& image, std::vector<float>& out) {
        out.assign(static_cast<size_t>(GW) * GH, 0.f);
        const int channels = static_cast<int>(image.channel);
        for (int gy = 0; gy < GH; ++gy) {
            const int sy = std::min<int>(static_cast<int>(image.height) - 1,
                                         static_cast<int>((gy + 0.5f) * image.height / GH));
            for (int gx = 0; gx < GW; ++gx) {
                const int sx = std::min<int>(static_cast<int>(image.width) - 1,
                                             static_cast<int>((gx + 0.5f) * image.width / GW));
                const uint8_t* pixel = image.data + (static_cast<size_t>(sy) * image.width + sx) * channels;
                out[static_cast<size_t>(gy) * GW + gx] =
                    channels >= 3 ? (0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2])
                                  : static_cast<float>(pixel[0]);
            }
        }
    };
    std::vector<float> reference, candidate;
    grid(prev_last, reference);
    std::vector<float> errors;
    errors.reserve(static_cast<size_t>(hi - lo + 1));
    int best_frame = centre;
    float best_error = 1e30f;
    // LTXAV_TRIM_DEBUG=1 dumps the whole match curve. Worth having permanently: the single number
    // this function returns cannot tell you whether it found a real minimum or just rail-clipped
    // at the edge of the band, and those two want opposite fixes.
    static const bool trim_debug = [] {
        const char* raw = getenv("LTXAV_TRIM_DEBUG");
        return raw != nullptr && *raw == '1';
    }();
    std::string curve;
    for (int frame = lo; frame <= hi; ++frame) {
        if (frames[frame].data == nullptr) {
            continue;
        }
        grid(frames[frame], candidate);
        float mae = 0.f;
        for (size_t i = 0; i < reference.size(); ++i) {
            mae += std::fabs(reference[i] - candidate[i]);
        }
        mae /= static_cast<float>(reference.size());
        if (trim_debug) {
            char cell[32];
            snprintf(cell, sizeof(cell), " %d:%.2f", frame, mae);
            curve += cell;
        }
        errors.push_back(mae);
        if (mae < best_error) {
            best_error = mae;
            best_frame = frame;
        }
    }
    // CONFIDENCE GATE -- refuse to answer when nothing in the band actually matches.
    //
    // An argmin always exists; a MATCH does not. Measured 2026-07-30, three seams, same clip
    // family, LTXAV_TRIM_DEBUG=1:
    //   continuous walk, seam 1  best 5.62 @16, band median 21.65  -> real minimum, ratio 0.26
    //   continuous walk, seam 2  best 3.89 @16, band median 29.52  -> real minimum, ratio 0.13
    //   alley -> beach crane     best 41.28,    band median 41.94  -> NO minimum,   ratio 0.98
    // The third is a scene change: the previous shot's last frame is nowhere in the continuation,
    // the whole band is flat noise, and the argmin was a two-way tie decided by float ordering.
    // That is exactly the number that used to be applied as a trim (24 here, 10 on the same recipe
    // without drive audio), and on an audio chain it also got carried into the NEXT seam's
    // prediction, over-trimming a seam whose own search had a clean answer at 17.
    //
    // So: below the threshold the measurement stands; at or above it, fall back to the caller's
    // centre, which is a deliberate estimate rather than the argmin of noise. On a genuinely
    // static shot every candidate matches, the ratio goes to 1, and falling back is also correct --
    // every trim looks the same, so there is nothing to gain by moving the cut.
    constexpr float CONFIDENT_RATIO = 0.6f;
    float band_median = 0.f;
    if (!errors.empty()) {
        std::vector<float> sorted = errors;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        band_median = sorted[sorted.size() / 2];
    }
    const bool confident = band_median > 0.f && best_error <= CONFIDENT_RATIO * band_median;
    if (trim_debug) {
        LOG_INFO("LTXAV auto-trim curve (centre %d, band %d..%d, best %d @ %.2f, band median %.2f, "
                 "ratio %.2f -> %s):%s",
                 centre, lo, hi, best_frame, best_error, band_median,
                 band_median > 0.f ? best_error / band_median : 0.f,
                 confident ? "CONFIDENT" : "no match, using the centre", curve.c_str());
    }
    if (!confident) {
        LOG_INFO("LTXAV auto-trim: no frame in %d..%d reproduces the previous shot's last frame "
                 "(best MAE %.1f vs band median %.1f) -- this seam is a scene change, not a "
                 "continuation; keeping the predicted %d-frame trim instead of the argmin of noise",
                 lo, hi, best_error, band_median, centre);
        return std::min(centre, std::max(0, n_frames - 1));
    }
    // best_frame is the frame that most closely REPRODUCES the prior segment's last frame -- i.e.
    // the duplicate of a moment the viewer has already seen. Keeping it freezes the action for one
    // frame at every join: measured 1.7 frame-to-frame difference against ~7 either side, which
    // reads as a slight hitch or jump rather than a cut. Drop it as well, so the join ADVANCES by
    // one frame of motion instead of repeating one. Bounded by the search band (hi <= n_frames-2),
    // so this stays inside the array.
    //
    // +1 AND NO FURTHER (tested, negative): advancing past the duplicate by 7 or 8 frames, to make
    // the trim land on 8*K, visibly JUMPED on a locked-off constant-velocity walk. Those frames
    // carry real motion -- only best_frame is a duplicate.
    return std::min(best_frame + 1, n_frames - 1);
}

// Per-segment bank SIDECARS. seg_<i>.bin holds VIDEO latents only, so a resumed chain can restore
// the picture of an already-rendered prefix but not its sound or its exact length. Two files fix
// that, and both are best-effort: a chain whose bank predates them just behaves as before.
//
//   seg_<i>.audio  the segment's DECODED audio: a header (rate u32, channels u32, samples u64)
//                  followed by INTERLEAVED float32 — the sd_audio_t layout. ★ INTERLEAVED, not
//                  planar: this comment said "planar" for a long time and it is wrong, which is
//                  the same trap load_wav_full warns about ("the planar reading of that buffer was
//                  a real historical bug"). Reading it planar does not fail loudly; it silently
//                  shifts every time offset by the channel count, so a measurement lands on the
//                  wrong part of the clip and quietly reports the wrong answer.
//                  Without this file, a chain resumed at segment N replays its whole prefix
//                  SILENTLY whenever the audio was model-generated rather than supplied.
//   seg_<i>.len    the KEPT frame count after the seam trim. Without it the restore path
//                  re-derives a drop from its own fresh-vs-continuation classification, which is
//                  exactly the disagreement the pinned trim exists to defeat -- so a retake
//                  reproduces a different prefix length and slides the whole timeline.
//
// ★ THE THREE MOVE TOGETHER, PER SHOT. Every RESTORE must take .bin, .len and .audio for a given
// shot from the SAME directory. Resolving the .bin out of one bank while .len still looks in
// another finds no .len, silently re-derives the drop, and slides every downstream segment and its
// audio along the timeline -- a fault that sounds and looks nothing like "wrong take" and would be
// blamed on anything but the bank. `bank_stem_for` below is the ONLY way a restore should build
// these paths, so the three cannot disagree by construction.
static std::string bank_stem_for(const sd_vid_chain_params_t* chain_params, int segment) {
    const char* dir = nullptr;
    if (chain_params->segment_bank_dirs != nullptr && segment >= 0 && segment < chain_params->n_segments &&
        chain_params->segment_bank_dirs[segment] != nullptr &&
        chain_params->segment_bank_dirs[segment][0] != '\0') {
        dir = chain_params->segment_bank_dirs[segment];
    } else {
        dir = chain_params->bank_dir;
    }
    return std::string(dir != nullptr ? dir : "") + "/seg_" + std::to_string(segment);
}

static bool write_seg_audio(const std::string& path, const sd_audio_t* audio) {
    if (audio == nullptr || audio->data == nullptr || audio->sample_count == 0) {
        return false;
    }
    FILE* file = fopen(path.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }
    const uint32_t rate     = audio->sample_rate;
    const uint32_t channels = audio->channels;
    const uint64_t samples  = audio->sample_count;
    bool ok = fwrite(&rate, sizeof(rate), 1, file) == 1 &&
              fwrite(&channels, sizeof(channels), 1, file) == 1 &&
              fwrite(&samples, sizeof(samples), 1, file) == 1;
    const size_t total = static_cast<size_t>(samples) * channels;
    ok = ok && fwrite(audio->data, sizeof(float), total, file) == total;
    fclose(file);
    return ok;
}

static sd_audio_t* read_seg_audio(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return nullptr;
    }
    uint32_t rate = 0, channels = 0;
    uint64_t samples = 0;
    if (fread(&rate, sizeof(rate), 1, file) != 1 || fread(&channels, sizeof(channels), 1, file) != 1 ||
        fread(&samples, sizeof(samples), 1, file) != 1 || rate == 0 || channels == 0 || samples == 0) {
        fclose(file);
        return nullptr;
    }
    const size_t total = static_cast<size_t>(samples) * channels;
    auto* audio = (sd_audio_t*)calloc(1, sizeof(sd_audio_t));
    if (audio == nullptr) {
        fclose(file);
        return nullptr;
    }
    audio->sample_rate  = rate;
    audio->channels     = channels;
    audio->sample_count = samples;
    audio->data         = (float*)malloc(total * sizeof(float));
    if (audio->data == nullptr || fread(audio->data, sizeof(float), total, file) != total) {
        fclose(file);
        free(audio->data);
        free(audio);
        return nullptr;
    }
    fclose(file);
    return audio;
}

static void write_seg_len(const std::string& path, int kept) {
    FILE* file = fopen(path.c_str(), "w");
    if (file != nullptr) {
        fprintf(file, "%d\n", kept);
        fclose(file);
    }
}

static int read_seg_len(const std::string& path) {  // -1 = absent/unreadable => fall back to re-deriving
    FILE* file = fopen(path.c_str(), "r");
    if (file == nullptr) {
        return -1;
    }
    int kept = -1;
    if (fscanf(file, "%d", &kept) != 1 || kept < 0) {
        kept = -1;
    }
    fclose(file);
    return kept;
}

static sd_image_t* decode_banked_video_latent(sd_ctx_t* sd_ctx, const std::string& path, int* count_out) {
    if (count_out != nullptr) {
        *count_out = 0;
    }
    try {
        auto latent = sd::load_tensor_from_file_as_tensor<float>(path);
        if (latent.empty()) {
            LOG_ERROR("video continuation bank %s is empty", path.c_str());
            return nullptr;
        }
        auto decoded = sd_ctx->sd->decode_first_stage(latent, true);
        if (decoded.empty()) {
            LOG_ERROR("could not VAE-decode video continuation bank %s", path.c_str());
            return nullptr;
        }
        const int count = static_cast<int>(decoded.shape()[2]);
        auto* frames = static_cast<sd_image_t*>(calloc(static_cast<size_t>(count), sizeof(sd_image_t)));
        if (frames == nullptr) {
            return nullptr;
        }
        for (int frame = 0; frame < count; ++frame) {
            frames[frame] = tensor_to_sd_image(decoded, frame);
            if (frames[frame].data == nullptr) {
                for (int i = 0; i < frame; ++i) {
                    free(frames[i].data);
                }
                free(frames);
                return nullptr;
            }
        }
        if (count_out != nullptr) {
            *count_out = count;
        }
        return frames;
    } catch (const std::exception& error) {
        LOG_ERROR("could not load video continuation bank %s: %s", path.c_str(), error.what());
        return nullptr;
    }
}

static bool save_banked_video_latent(const std::string& path, const sd::Tensor<float>& latent) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }
    const int32_t dimensions = static_cast<int32_t>(latent.dim());
    const std::string name = "wan_vace_video_latent";
    const int32_t name_length = static_cast<int32_t>(name.size());
    const int32_t type = GGML_TYPE_F32;
    output.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    output.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
    output.write(reinterpret_cast<const char*>(&type), sizeof(type));
    for (int32_t dimension = 0; dimension < dimensions; ++dimension) {
        const int32_t length = static_cast<int32_t>(latent.shape()[dimension]);
        output.write(reinterpret_cast<const char*>(&length), sizeof(length));
    }
    output.write(name.data(), name.size());
    output.write(reinterpret_cast<const char*>(latent.data()),
                 static_cast<std::streamsize>(latent.numel() * sizeof(float)));
    return output.good();
}

// Stage one window of drive audio into the job bank as a 16 kHz WAV, which the
// per-window render then re-reads through encode_ltxav_drive_audio.
//
// `samples` is INTERLEAVED with `n_channels` channels. This used to hardcode
// mono, which silently collapsed a stereo drive track at the one point between
// the loader and the VAE where nothing else would have noticed.
static bool write_ltx_drive_audio_wav(const std::string& path,
                                      const std::vector<float>& samples,
                                      uint32_t n_channels = 1) {
    if (samples.empty() || n_channels == 0 || samples.size() % n_channels != 0) {
        return false;
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        LOG_ERROR("generate_video_chain: could not create drive-audio slice %s", path.c_str());
        return false;
    }
    const uint32_t sample_rate = 16000;
    const uint16_t channels = static_cast<uint16_t>(n_channels);
    const uint16_t bits = 16;
    const uint32_t data_length = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t riff_length = 36 + data_length;
    const uint32_t byte_rate = sample_rate * channels * (bits / 8);
    const uint16_t block_align = channels * (bits / 8);
    const uint32_t fmt_length = 16;
    const uint16_t format = 1;
    output.write("RIFF", 4);
    output.write(reinterpret_cast<const char*>(&riff_length), sizeof(riff_length));
    output.write("WAVEfmt ", 8);
    output.write(reinterpret_cast<const char*>(&fmt_length), sizeof(fmt_length));
    output.write(reinterpret_cast<const char*>(&format), sizeof(format));
    output.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    output.write(reinterpret_cast<const char*>(&sample_rate), sizeof(sample_rate));
    output.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    output.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    output.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
    output.write("data", 4);
    output.write(reinterpret_cast<const char*>(&data_length), sizeof(data_length));
    for (const float sample : samples) {
        const int16_t pcm = static_cast<int16_t>(std::lround(std::clamp(sample, -1.f, 1.f) * 32767.f));
        output.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
    }
    return output.good();
}

struct LTXChainAudio {
    std::vector<float> samples;
    uint32_t sample_rate = 0;
    uint32_t channels = 0;

    bool loaded() const {
        return sample_rate != 0 && channels != 0 && !samples.empty();
    }

    std::vector<float> window(int64_t start_frame,
                              int64_t frame_count,
                              int fps,
                              int64_t offset_frames) const {
        if (!loaded() || frame_count <= 0 || fps <= 0) {
            return {};
        }
        const int64_t first = std::llround(static_cast<double>(start_frame - offset_frames) * sample_rate / fps);
        const int64_t last = std::llround(static_cast<double>(start_frame + frame_count - offset_frames) * sample_rate / fps);
        const int64_t wanted = std::max<int64_t>(0, last - first);
        std::vector<float> result(static_cast<size_t>(wanted) * channels, 0.f);
        const int64_t available = static_cast<int64_t>(samples.size() / channels);
        for (int64_t frame = 0; frame < wanted; ++frame) {
            const int64_t source_frame = first + frame;
            if (source_frame < 0 || source_frame >= available) continue;
            std::copy_n(samples.data() + static_cast<size_t>(source_frame) * channels,
                        channels,
                        result.data() + static_cast<size_t>(frame) * channels);
        }
        return result;
    }
};

static sd_audio_t* make_ltx_chain_audio(const std::vector<float>& samples,
                                        uint32_t sample_rate,
                                        uint32_t channels) {
    if (samples.empty() || sample_rate == 0 || channels == 0 || samples.size() % channels != 0) {
        return nullptr;
    }
    auto* audio = static_cast<sd_audio_t*>(calloc(1, sizeof(sd_audio_t)));
    if (audio == nullptr) return nullptr;
    audio->sample_rate = sample_rate;
    audio->channels = channels;
    audio->sample_count = samples.size() / channels;
    audio->data = static_cast<float*>(malloc(samples.size() * sizeof(float)));
    if (audio->data == nullptr) {
        free(audio);
        return nullptr;
    }
    std::memcpy(audio->data, samples.data(), samples.size() * sizeof(float));
    return audio;
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
        frames_out == nullptr || num_frames_out == nullptr || chain_params->n_segments < 1) {
        LOG_ERROR("generate_video_chain: invalid arguments");
        return false;
    }
    const bool ltx_chain = sd_version_is_ltxav(sd_ctx->sd->version);
    const bool avatar_chain = sd_version_is_longcat_avatar(sd_ctx->sd->version);
    if (!ltx_chain && !avatar_chain) {
        LOG_ERROR("generate_video_chain requires an LTX or LongCat Avatar video model");
        return false;
    }
    if (!ltx_chain && ((chain_params->chain_audio_full != nullptr && chain_params->chain_audio_full[0] != '\0') ||
                       (chain_params->chain_audio_track != nullptr && chain_params->chain_audio_track[0] != '\0'))) {
        LOG_ERROR("generate_video_chain: full-timeline audio inputs are supported only for LTX chains");
        return false;
    }

    // Scope the Avatar cache strictly to this request.  A later request with a
    // different image or audio must never inherit a condition from this chain.
    struct AvatarChainTextCacheGuard {
        StableDiffusionGGML* sd;
        ~AvatarChainTextCacheGuard() {
            sd->avatar_chain_text_cache.clear();
            sd->avatar_chain_text_cache_active = false;
        }
    } avatar_chain_text_cache_guard{sd_ctx->sd};
    if (avatar_chain) {
        sd_ctx->sd->avatar_chain_text_cache.clear();
        sd_ctx->sd->avatar_chain_text_cache_active = true;
    }

    // Last seam the trim search actually measured, fed forward as the next segment's a-priori
    // estimate -- and, on an audio chain, as the next segment's APPLIED trim, since the two are
    // pinned together there. The drive-audio window has to be cut BEFORE a segment renders but the
    // trim can only be measured AFTER, so the two can only agree if the applied value is the
    // predicted one; carrying the previous seam's measurement forward is how the predicted value
    // still tracks the content. Seams in one clip sit close together (8 and 8, or 8 and 14, on
    // measured chains). -1 = nothing measured yet.
    int last_measured_seam_drop = -1;
    // Per-seam trim log so a long chain reports its behaviour in ONE line at the end, instead of
    // the operator having to spot a mid-render WARN. `applied` == `predicted` is the normal case
    // (that is the pin); `measured` is what the advisory search wanted, and a wide
    // measured-vs-applied spread is the honest signal that the prediction models this clip badly.
    struct SeamTrim {
        int predicted;
        int applied;
        int measured;
    };
    std::vector<SeamTrim> seam_trims;
    int carried_frames = std::max(1, chain_params->cont_latent_frames);
    const int overlap_frames = 1 + (carried_frames - 1) * (ltx_chain ? 8 : 4);
    // Seam trim fallback when the caller supplies no pin.
    //
    // This USED to be 8*K on the theory that the causal VAE decodes the K carried latents to
    // overlap_px = 1+(K-1)*8 real frames and the model then re-renders 7 "settling" frames, making
    // 8*K = overlap_px + 7 the true trim. EYE TEST 2026-07-28 REFUTES THAT: on a locked-off
    // constant-velocity walk (the case where a skip is unmissable), trimming to 8*K visibly JUMPS,
    // while trimming to overlap_px is clean. Those 7 frames carry real motion, they are not
    // duplicates of content already shown.
    //
    // Which also resolves the A/V desync properly. The rule is `drop_applied == drop_predicted`,
    // and drop_predicted must be known BEFORE the render because it cuts the drive audio. It does
    // not have to be 8*K to satisfy that — overlap_px is equally a pure function of K. The
    // auto-trim search independently lands on overlap_px (measured: 9 at K=2, 17 at K=3), so
    // predicting overlap_px makes prediction and application agree WITHOUT forcing the picture to
    // a cut that jumps. 8*K made them agree by moving the wrong one.
    //
    // Callers that know better still win via cont_seam_drop_frames / segment_seam_drop_frames.
    const int seam_drop_default = overlap_frames;
    if (overlap_frames >= base_params->video_frames &&
        (chain_params->segment_video_frames == nullptr || chain_params->segment_video_frames[0] <= 0)) {
        LOG_ERROR("generate_video_chain: continuation overlap %d leaves no new frames in a %d-frame segment",
                  overlap_frames,
                  base_params->video_frames);
        return false;
    }
    if (chain_params->start_segment < 0 || chain_params->start_segment >= chain_params->n_segments) {
        LOG_ERROR("generate_video_chain: invalid resume segment %d/%d",
                  chain_params->start_segment,
                  chain_params->n_segments);
        return false;
    }
    if (chain_params->start_segment > 0 && (chain_params->bank_dir == nullptr || chain_params->bank_dir[0] == '\0')) {
        LOG_ERROR("generate_video_chain: resume requires a durable bank directory");
        return false;
    }
    const bool retake_active = chain_params->enable_retake && chain_params->retake_segment >= 0;
    if (retake_active && (chain_params->retake_segment >= chain_params->n_segments ||
                          chain_params->bank_dir == nullptr || chain_params->bank_dir[0] == '\0')) {
        LOG_ERROR("generate_video_chain: retake requires a valid segment and durable bank directory");
        return false;
    }
    const int sample_start = retake_active ? chain_params->retake_segment : chain_params->start_segment;
    const int sample_end = retake_active ? chain_params->retake_segment : chain_params->n_segments - 1;

    const int64_t audio_offset = (chain_params->chain_audio_full != nullptr || chain_params->chain_audio_track != nullptr)
                                     ? chain_params->chain_audio_offset_frames
                                     : 0;
    LTXChainAudio drive_audio;
    if (chain_params->chain_audio_full != nullptr && chain_params->chain_audio_full[0] != '\0') {
        uint32_t drive_channels = 0;
        if (chain_params->bank_dir == nullptr || chain_params->bank_dir[0] == '\0' ||
            !LONGCAT_AUDIO::load_wav_16k(chain_params->chain_audio_full,
                                         drive_audio.samples,
                                         drive_channels,
                                         ltx_drive_audio_channels()) ||
            drive_audio.samples.empty()) {
            LOG_ERROR("generate_video_chain: drive audio requires a readable WAV and durable bank directory");
            return false;
        }
        drive_audio.sample_rate = 16000;
        drive_audio.channels = drive_channels;
    }
    LTXChainAudio track_audio;
    if (chain_params->chain_audio_track != nullptr && chain_params->chain_audio_track[0] != '\0') {
        if (!LONGCAT_AUDIO::load_wav_full(chain_params->chain_audio_track,
                                          track_audio.samples,
                                          track_audio.sample_rate,
                                          track_audio.channels) ||
            !track_audio.loaded()) {
            LOG_ERROR("generate_video_chain: track audio is not a readable WAV");
            return false;
        }
    }

    // ── SEAM TRIM: PINNED vs CONTENT-ADAPTIVE ───────────────────────────────────────────────
    // A/V is exact IFF the trim actually APPLIED equals the trim this segment's drive-audio window
    // was CUT against. The window is cut before the render; a content-adaptive trim can only be
    // measured after it. Nothing closes that gap -- not even feeding the previous measurement
    // forward, which is wrong at the very first seam by construction and wrong again at every seam
    // whose content differs from the last one. Whatever the search returns that the estimate did
    // not predict lands as a within-shot A/V offset, and the damage is not just the audible
    // repeat in the generated track: the shot was RENDERED against audio in the wrong place, so
    // the lip-sync error is baked into the pixels and survives any re-mux of the source song.
    //
    // Therefore: PIN the trim whenever the engine owns the audio -- i.e. whenever there is anything
    // to desync -- and let the adaptive search run only when a miss cannot cost anything (a
    // picture-only chain). Seam smoothness on the pinned path is the job of frame-count-PRESERVING
    // polish, not of moving the cut. Caller pin wins; cont_seam_drop_frames == 0 uses the derived
    // prediction (overlap_px, then the previous seam's measurement); a NEGATIVE
    // cont_seam_drop_frames forces the adaptive search back on even with audio (a diagnostic
    // escape hatch -- it will desync, and the tripwire below says so).
    const bool has_shot_audio    = chain_params->segment_audio_full != nullptr ||
                                   chain_params->segment_audio_track != nullptr;
    const bool has_windowed_audio_dir = chain_params->chain_audio_dir != nullptr &&
                                        chain_params->chain_audio_dir[0] != '\0';
    const bool engine_owns_audio = drive_audio.loaded() || track_audio.loaded() || has_shot_audio ||
                                   has_windowed_audio_dir;
    const bool force_adaptive_seam = chain_params->cont_seam_drop_frames < 0;
    // PIN TO THE PREDICTION WHENEVER THE ENGINE OWNS THE AUDIO.
    //
    // The previous revision left the adaptive search APPLIED over audio on the argument that
    // "the auto-trim search independently lands on overlap_px (measured: 9 at K=2, 17 at K=3), so
    // predicting overlap_px makes prediction and application agree". MEASURED 2026-07-30, that is
    // false: on a 2-shot i2v+continuation chain at K=3 with a supplied drive song, the estimate
    // was 17 and the search applied 24; the SAME recipe and seed with no drive audio applied 10.
    // The search is a content match — it moves with the content, and the content moves with the
    // audio conditioning, so no a-priori value can track it.
    //
    // The damage is measurable, not theoretical. Envelope cross-correlation of the delivered
    // Opus track against the drive song (envelopes, not samples — the audio VAE round-trips the
    // waveform) put shot 1 at +0.022 s and shot 2 at +0.315 s: the song SKIPS FORWARD by
    // applied-minus-predicted = 7 frames = 292 ms across the seam, and the shot itself is
    // rendered 292 ms off the song. Picture and its own generated audio stay together (both take
    // the same head drop), so the audible defect on this path is a jump in the music; on any path
    // where the caller re-muxes the SOURCE song against the final timeline it is 292 ms of true
    // lip-sync error, baked into the pixels.
    //
    // So the applied trim is the ESTIMATE, always, whenever there is audio to desync. Note what
    // that estimate is: NOT the old 8*K, which the 2026-07-28 eye test showed visibly jumping on a
    // locked-off walk. It is overlap_px at the first seam (the cut that eye test found clean) and
    // the PREVIOUS seam's measured trim at every seam after it. From the second seam on the
    // applied trim is therefore content-adaptive AND frame-exact against the audio; only the first
    // seam of a chain pays for the fact that a trim cannot be measured before the render it has to
    // precede. The search still runs on this path as an ADVISORY: it reports the gap and feeds the
    // next seam's estimate, it just no longer moves this shot.
    //
    // LTXAV_PIN_SEAM=0 restores apply-the-measurement for A/B (it will desync, and the tripwire
    // below says so), as does a NEGATIVE cont_seam_drop_frames.
    static const bool pin_seam_env = [] {
        const char* raw = getenv("LTXAV_PIN_SEAM");
        return raw == nullptr || *raw != '0';
    }();
    const bool pin_derived_seam = pin_seam_env && engine_owns_audio && !force_adaptive_seam &&
                                  chain_params->cont_seam_drop_frames == 0;
    if (ltx_chain && chain_params->n_segments > 1) {
        if (chain_params->cont_seam_drop_frames > 0) {
            LOG_INFO("generate_video_chain: seam trim PINNED by the caller to %d frames%s -> the "
                     "drive-audio window and the applied trim agree by construction",
                     chain_params->cont_seam_drop_frames,
                     chain_params->segment_seam_drop_frames != nullptr ? " (per-shot entries supersede)" : "");
        } else if (pin_derived_seam) {
            LOG_INFO("generate_video_chain: seam trim PINNED to this seam's PREDICTION (first seam %d = "
                     "guide overlap_px, K=%d; later seams inherit the previous seam's measured trim) "
                     "because the engine owns the audio -> the drive-audio window and the applied trim "
                     "agree at every seam. The auto-trim search still runs as an advisory.",
                     seam_drop_default, carried_frames);
        } else {
            LOG_INFO("generate_video_chain: seam trim CONTENT-ADAPTIVE (search centred on %d, guide "
                     "overlap_px=%d)%s",
                     seam_drop_default,
                     overlap_frames,
                     engine_owns_audio ? " -- FORCED ON with audio present: A/V will wander by the "
                                         "per-segment prediction error"
                                       : "");
        }
    }

    std::vector<sd_image_t> stitched;
    std::vector<float> previous_tail;
    int tail_width = 0;
    int tail_height = 0;
    int tail_channels = 0;
    uint32_t chain_audio_rate = 0;
    uint32_t chain_audio_channels = 0;
    std::vector<float> chain_audio_samples;

    auto release_stitched = [&]() {
        for (auto& frame : stitched) {
            free(frame.data);
        }
        stitched.clear();
    };

    // WINDOWED STREAMING FINALIZE. Without it `stitched` grows to the entire decoded timeline --
    // about 14 GB for a 3.5-minute 1280x704 chain and 32 GB at 1920x1088, which is swap-thrash or
    // an OOM rather than a slow render. With on_flush_frames set we instead keep only the last
    // WINDOW_KEEP frames: the furthest back any later operation can still reach.
    //
    // What reaches backwards, and how far:
    //   * ltxav_auto_trim_drop compares against stitched.back()      -> 1 frame
    //   * a future seam polish (exposure match) reads the last 16    -> 16 frames
    // Everything older than that suffix is final forever, so it is handed to the caller's encoder
    // (which takes ownership and frees it) as we go. Because the retained window is always a
    // SUFFIX of the full timeline, every one of those operations sees byte-identical input to the
    // accumulate-everything path -- streaming changes memory, not pixels.
    const bool streaming = chain_params->on_flush_frames != nullptr;
    constexpr int WINDOW_KEEP = 16;
    long long flushed_total = 0;
    if (streaming) {
        LOG_INFO("generate_video_chain: streaming finalize ON (keeping a %d-frame window; frames are "
                 "handed over and freed as they become final)", WINDOW_KEEP);
    }
    auto flush_window = [&](bool final_flush) {
        if (!streaming) {
            return;
        }
        const int keep = final_flush ? 0 : WINDOW_KEEP;
        if (static_cast<int>(stitched.size()) <= keep) {
            return;
        }
        const int n_flush = static_cast<int>(stitched.size()) - keep;
        // Hand the now-final prefix over IN ORDER; the callee consumes and frees each .data.
        chain_params->on_flush_frames(stitched.data(), n_flush, chain_params->on_flush_frames_user);
        stitched.erase(stitched.begin(), stitched.begin() + n_flush);
        flushed_total += n_flush;
    };
    // Frames emitted so far, whether still resident or already flushed. This is the true timeline
    // position -- stitched.size() alone is only the resident tail once streaming is on.
    auto timeline_frames = [&]() -> long long {
        return flushed_total + static_cast<long long>(stitched.size());
    };
    // ── DRIVE-AUDIO TIMELINE SKEW ───────────────────────────────────────────────────────────────
    // Frames of the DRIVE timeline that were rendered and then discarded, and whose audio no other
    // shot covers. Output timeline frame t is driven by drive frame `t + drive_head_trimmed`, and
    // everything that reads the drive or deliverable timeline has to add it.
    //
    // Only the REFERENCE HEAD TRIM contributes. A seam drop discards frames too, but those are the
    // continuation overlap -- the previous shot already showed that stretch of the song, so the
    // timeline does not advance past it. A reference head trim discards frames nothing else covers,
    // so the song genuinely runs on underneath them.
    //
    // Measured 2026-07-30 on a bare t2v opener with two character references (AUTO -> 1 frame):
    // envelope lag against the drive song was +0.022 s with the trim off and +0.062 s with it on,
    // a clean +0.040 s = 0.96 frames, while both deliverable-track paths handed back the source at
    // NCC 0.9977 lag 0.00. One frame of lip desync, 41.7 ms, in every shipped clip that trims.
    //
    // Unlike the seam trim this is repairable output-side: shot 0 ESTABLISHES the timeline instead
    // of having to match one, so nothing has to be predicted before the render -- the trim is
    // reported back after it and every consumer of the offset runs later.
    int64_t drive_head_trimmed = 0;
    auto fail = [&]() {
        release_stitched();
        // Reclaim GPU memory on a failed/aborted job too, so a persistent worker
        // does not carry this job's committed pool into the next request.
        sd_ctx->sd->reclaim_ltx_chain_window_gpu_memory();
        return false;
    };
    auto append_audio = [&](const sd_audio_t* audio, int drop_frames, int kept_frames) {
        if (audio == nullptr || audio->data == nullptr || audio->sample_count == 0 ||
            audio->sample_rate == 0 || audio->channels == 0) {
            return;
        }
        if (chain_audio_channels == 0) {
            chain_audio_rate = audio->sample_rate;
            chain_audio_channels = audio->channels;
        }
        if (audio->sample_rate != chain_audio_rate || audio->channels != chain_audio_channels) {
            LOG_WARN("generate_video_chain: ignoring segment audio with incompatible format %u Hz x%u",
                     audio->sample_rate,
                     audio->channels);
            return;
        }
        const int fps = std::max(1, base_params->fps);
        const uint64_t start = std::min<uint64_t>(audio->sample_count,
                                                   static_cast<uint64_t>(std::llround(
                                                       static_cast<double>(std::max(0, drop_frames)) *
                                                       audio->sample_rate / fps)));
        const uint64_t wanted = static_cast<uint64_t>(std::llround(
            static_cast<double>(std::max(0, kept_frames)) * audio->sample_rate / fps));
        const uint64_t count = std::min<uint64_t>(wanted, audio->sample_count - start);
        const size_t begin = static_cast<size_t>(start) * audio->channels;
        const size_t end = static_cast<size_t>(start + count) * audio->channels;
        chain_audio_samples.insert(chain_audio_samples.end(), audio->data + begin, audio->data + end);
    };
    // A per-shot deliverable track is on the SAME clock as that shot's per-shot drive WAV, which
    // is the shot AS RENDERED -- frame 0 of the track is frame 0 of the render, seam overlap and
    // contaminated reference head included. So it takes the same head drop the picture and the
    // generated audio take (see append_audio, which has always done this). Reading it from t=0
    // instead shipped the shot's picture `drop` frames ahead of its own deliverable: measured
    // 2026-07-30, one frame (41.7 ms) on a head-trimmed t2v opener, and it would be the whole
    // seam drop -- 17 frames, 708 ms -- on any continuation shot carrying per-shot audio.
    auto append_shot_track = [&](const char* path, int drop_frames, int kept_frames) -> bool {
        if (path == nullptr || path[0] == '\0') {
            return true;
        }
        std::vector<float> samples;
        uint32_t sample_rate = 0;
        uint32_t channels = 0;
        if (!LONGCAT_AUDIO::load_wav_full(path, samples, sample_rate, channels) || sample_rate == 0 || channels == 0) {
            LOG_ERROR("generate_video_chain: per-shot audio track is not a readable WAV: %s", path);
            return false;
        }
        if (chain_audio_channels == 0) {
            chain_audio_rate = sample_rate;
            chain_audio_channels = channels;
        }
        if (sample_rate != chain_audio_rate || channels != chain_audio_channels) {
            LOG_ERROR("generate_video_chain: per-shot audio track format %u Hz x%u does not match %u Hz x%u",
                      sample_rate, channels, chain_audio_rate, chain_audio_channels);
            return false;
        }
        const int fps = std::max(1, base_params->fps);
        const size_t available = samples.size() / channels;
        const size_t start = std::min<size_t>(available,
                                              static_cast<size_t>(std::llround(
                                                  static_cast<double>(std::max(0, drop_frames)) * sample_rate / fps)));
        const size_t wanted = static_cast<size_t>(std::llround(
            static_cast<double>(std::max(0, kept_frames)) * sample_rate / fps));
        const size_t copy = std::min(wanted, available - start);
        chain_audio_samples.insert(chain_audio_samples.end(),
                                   samples.begin() + start * channels,
                                   samples.begin() + (start + copy) * channels);
        chain_audio_samples.insert(chain_audio_samples.end(), (wanted - copy) * channels, 0.f);
        return true;
    };
    auto adopt_frames = [&](sd_image_t* segment_frames, int count, int drop) {
        drop = std::clamp(drop, 0, count);
        for (int frame = 0; frame < count; ++frame) {
            if (frame < drop) {
                free(segment_frames[frame].data);
            } else {
                stitched.push_back(segment_frames[frame]);
            }
        }
        free(segment_frames);
    };
    auto extract_tail = [&](const float* raw,
                            int width,
                            int height,
                            int frames,
                            int channels) -> bool {
        const int video_channels = sd_ctx->sd->get_latent_channel();
        if (raw == nullptr || width <= 0 || height <= 0 || frames < carried_frames ||
            channels < video_channels) {
            LOG_ERROR("generate_video_chain: invalid exported LTX latent %dx%dx%dx%d for K=%d",
                      width,
                      height,
                      frames,
                      channels,
                      carried_frames);
            return false;
        }
        const size_t plane = static_cast<size_t>(width) * height;
        previous_tail.assign(plane * static_cast<size_t>(carried_frames) * video_channels, 0.f);
        for (int channel = 0; channel < video_channels; ++channel) {
            for (int frame = 0; frame < carried_frames; ++frame) {
                const size_t from = (static_cast<size_t>(channel) * frames + (frames - carried_frames + frame)) * plane;
                const size_t to = (static_cast<size_t>(channel) * carried_frames + frame) * plane;
                std::memcpy(previous_tail.data() + to, raw + from, plane * sizeof(float));
            }
        }
        tail_width = width;
        tail_height = height;
        tail_channels = video_channels;
        return true;
    };
    auto extract_head = [&](const float* raw,
                            int width,
                            int height,
                            int frames,
                            int channels,
                            std::vector<float>& head) -> bool {
        const int video_channels = sd_ctx->sd->get_latent_channel();
        if (raw == nullptr || width <= 0 || height <= 0 || frames < carried_frames || channels < video_channels) {
            return false;
        }
        const size_t plane = static_cast<size_t>(width) * height;
        head.assign(plane * static_cast<size_t>(carried_frames) * video_channels, 0.f);
        for (int channel = 0; channel < video_channels; ++channel) {
            std::memcpy(head.data() + static_cast<size_t>(channel) * carried_frames * plane,
                        raw + static_cast<size_t>(channel) * frames * plane,
                        static_cast<size_t>(carried_frames) * plane * sizeof(float));
        }
        return true;
    };
    auto save_video_latent = [&](const std::string& path,
                                  const float* raw,
                                  int width,
                                  int height,
                                  int frames,
                                  int channels) -> bool {
        const int video_channels = sd_ctx->sd->get_latent_channel();
        if (raw == nullptr || width <= 0 || height <= 0 || frames <= 0 || channels < video_channels) {
            return false;
        }
        sd::Tensor<float> video({width, height, frames, video_channels, 1});
        const size_t plane = static_cast<size_t>(width) * height;
        for (int channel = 0; channel < video_channels; ++channel) {
            const size_t count = plane * static_cast<size_t>(frames);
            std::memcpy(video.data() + static_cast<size_t>(channel) * count,
                        raw + static_cast<size_t>(channel) * count,
                        count * sizeof(float));
        }
        return save_banked_video_latent(path, video);
    };

    // Rebuild a durable prefix exactly as the original chain stitched it, and
    // recover the final bank's latent tail for the next sampled window.
    for (int segment = 0; segment < sample_start; ++segment) {
        const std::string path = bank_stem_for(chain_params, segment) + ".bin";
        int count = 0;
        sd_image_t* banked_frames = decode_banked_video_latent(sd_ctx, path, &count);
        if (banked_frames == nullptr || count <= 0) {
            free(banked_frames);
            return fail();
        }
        try {
            auto banked = sd::load_tensor_from_file_as_tensor<float>(path);
            if (banked.empty() || banked.dim() < 4 ||
                !extract_tail(banked.data(),
                              static_cast<int>(banked.shape()[0]),
                              static_cast<int>(banked.shape()[1]),
                              static_cast<int>(banked.shape()[2]),
                              static_cast<int>(banked.shape()[3]))) {
                for (int frame = 0; frame < count; ++frame) free(banked_frames[frame].data);
                free(banked_frames);
                return fail();
            }
        } catch (const std::exception& error) {
            LOG_ERROR("generate_video_chain: could not restore LTX bank %s: %s", path.c_str(), error.what());
            for (int frame = 0; frame < count; ++frame) free(banked_frames[frame].data);
            free(banked_frames);
            return fail();
        }
        bool keyframe_at_start = false;
        if (chain_params->segment_keyframes != nullptr && chain_params->segment_keyframe_counts != nullptr &&
            chain_params->segment_keyframe_counts[segment] > 0 &&
            chain_params->segment_keyframe_indices != nullptr &&
            chain_params->segment_keyframe_indices[segment] != nullptr) {
            for (int keyframe = 0; keyframe < chain_params->segment_keyframe_counts[segment]; ++keyframe) {
                if (chain_params->segment_keyframe_indices[segment][keyframe] == 0) {
                    keyframe_at_start = true;
                    break;
                }
            }
        }
        const bool fresh_scene = chain_params->segment_scene_cuts != nullptr &&
                                 chain_params->segment_scene_cuts[segment] != 0;
        const bool has_scene_image = chain_params->segment_init_images != nullptr &&
                                     chain_params->segment_init_images[segment] != nullptr &&
                                     chain_params->segment_init_images[segment]->data != nullptr;
        const int v2v_mode = chain_params->segment_v2v_modes == nullptr ? 0 : chain_params->segment_v2v_modes[segment];
        const bool has_v2v_control = chain_params->segment_control_frames != nullptr &&
                                    chain_params->segment_control_frame_counts != nullptr &&
                                    chain_params->segment_control_frames[segment] != nullptr &&
                                    chain_params->segment_control_frame_counts[segment] > 0;
        const bool has_v2v_latent = chain_params->segment_v2v_guide_latent_paths != nullptr &&
                                    chain_params->segment_v2v_guide_latent_paths[segment] != nullptr &&
                                    chain_params->segment_v2v_guide_latent_paths[segment][0] != '\0';
        const bool has_v2v_source = (v2v_mode == 0 && has_v2v_control) ||
                                    (v2v_mode == 1 && has_v2v_control) ||
                                    (v2v_mode == 2 && (has_v2v_control || has_v2v_latent));
        const int declared_drop = chain_params->segment_seam_drop_frames != nullptr
                                      ? chain_params->segment_seam_drop_frames[segment]
                                      : -1;
        int drop = segment == 0 || fresh_scene || has_scene_image || keyframe_at_start || has_v2v_source
                       ? 0
                       : (declared_drop >= 0 ? declared_drop
                                             : (chain_params->cont_seam_drop_frames > 0
                                                    ? chain_params->cont_seam_drop_frames
                                                    : seam_drop_default));
        // LENGTH PIN: if this segment banked its kept count, honour it verbatim. Re-deriving the
        // drop here would let a retake reproduce a DIFFERENT prefix length and slide every
        // downstream segment (and its audio) along the timeline.
        const int seam_policy_drop = drop;
        const int banked_kept = read_seg_len(bank_stem_for(chain_params, segment) + ".len");
        if (banked_kept >= 0 && banked_kept <= count) {
            const int pinned_drop = count - banked_kept;
            if (pinned_drop != drop) {
                LOG_INFO("generate_video_chain: window %d restored to its banked length (kept %d, "
                         "drop %d, would have re-derived %d)",
                         segment + 1, banked_kept, pinned_drop, drop);
            }
            drop = pinned_drop;
        }
        // A banked drop on a shot the seam policy says to drop NOTHING from is a reference head
        // trim this prefix shot took when it was first rendered. It skewed the drive clock then
        // and it has to skew it again now, or the shots this resume actually re-renders get their
        // drive windows cut against a clock the restored prefix does not share.
        //
        // Gated on seam_policy_drop == 0 and not just on the difference: only an opener-shaped
        // shot can carry a head trim (it self-gates off i2v, keyframe-at-0 and continuation, all
        // of which also force this drop to 0), so on a continuation shot a banked-vs-policy
        // difference means the CALLER changed the seam policy between renders, which is not a
        // trim and must not move the clock.
        const int restored_head_trim = seam_policy_drop == 0 ? std::max(0, drop) : 0;
        if (restored_head_trim > 0) {
            drive_head_trimmed += restored_head_trim;
            LOG_INFO("generate_video_chain: restored window %d carried a %d-frame reference head "
                     "trim -> drive clock now %lld frame(s) ahead of the output timeline",
                     segment + 1, restored_head_trim, (long long)drive_head_trimmed);
        }
        if (!track_audio.loaded() && chain_params->segment_audio_track != nullptr &&
            chain_params->segment_audio_track[segment] != nullptr &&
            chain_params->segment_audio_track[segment][0] != '\0' &&
            !append_shot_track(chain_params->segment_audio_track[segment],
                               std::min(drop, count),
                               count - std::min(drop, count))) {
            for (int frame = 0; frame < count; ++frame) free(banked_frames[frame].data);
            free(banked_frames);
            return fail();
        }
        // Restore this prefix segment's own audio. Without it a resumed chain whose audio was
        // MODEL-GENERATED (rather than supplied as a track) plays its whole prefix silent.
        if (!track_audio.loaded() &&
            (chain_params->segment_audio_track == nullptr ||
             chain_params->segment_audio_track[segment] == nullptr ||
             chain_params->segment_audio_track[segment][0] == '\0')) {
            const std::string audio_path = bank_stem_for(chain_params, segment) + ".audio";
            if (sd_audio_t* banked_audio = read_seg_audio(audio_path); banked_audio != nullptr) {
                append_audio(banked_audio, drop, count - std::min(drop, count));
                free_sd_audio(banked_audio);
            }
        }
        adopt_frames(banked_frames, count, drop);
        // Restored prefix frames are final the moment they land -- nothing re-renders them.
        flush_window(false);
    }

    // TASS character-reference SHOT SCOPING. A sheet is an identity, so it arrives once at the top
    // level, but a chain's shots do not all contain the same people: a reference scoped to shot 1
    // must contribute no tokens at all to shot 0, not merely be ignored there.
    //
    // The rotary source ids are settled ONCE, over the whole array, using the same rule the encode
    // path applies (2, 3, 4, ... in order, resuming one past any explicit id). Recomputing them per
    // segment would renumber whatever survived the filter, and an identity that changes its source
    // tag between shots is a different identity to the model.
    std::vector<int> tass_source_ids;
    const bool tass_scoped = base_params->character_refs != nullptr && base_params->character_refs_size > 0 &&
                             base_params->character_ref_segment_counts != nullptr;
    std::vector<int> tass_scope_offsets;
    if (base_params->character_refs != nullptr && base_params->character_refs_size > 0) {
        tass_source_ids.reserve(base_params->character_refs_size);
        // The encode path numbers the MSR strip FIRST and takes 2 for it, so a chain that carries
        // both a strip and character references must resume at 3 or reference 0 collides with the
        // strip. Inert while the phase scale is zero -- the tag multiplies by zero -- but a
        // landmine the moment anyone sets it positive.
        int next_source_id = base_params->msr_frames > 0 ? 3 : 2;
        for (int i = 0; i < base_params->character_refs_size; ++i) {
            int source_id = next_source_id;
            if (base_params->character_ref_source_ids != nullptr && base_params->character_ref_source_ids[i] > 1) {
                source_id = base_params->character_ref_source_ids[i];
            }
            next_source_id = source_id + 1;
            tass_source_ids.push_back(source_id);
        }
        if (tass_scoped) {
            tass_scope_offsets.reserve(base_params->character_refs_size);
            int offset = 0;
            for (int i = 0; i < base_params->character_refs_size; ++i) {
                tass_scope_offsets.push_back(offset);
                offset += std::max(0, base_params->character_ref_segment_counts[i]);
            }
        }
    }
    // Backing storage for the per-segment filtered views. Rebuilt each iteration and only ever read
    // by the generate_video call inside that same iteration.
    std::vector<sd_image_t> segment_character_refs;
    std::vector<int> segment_character_ref_source_ids;

    for (int segment = sample_start; segment <= sample_end; ++segment) {
        if (chain_params->before_segment != nullptr &&
            !chain_params->before_segment(segment, chain_params->before_segment_user)) {
            LOG_ERROR("generate_video_chain: segment %d model lease failed", segment + 1);
            return fail();
        }
        sd_vid_gen_params_t params = *base_params;
        // Narrow the references to the ones this shot is in scope for. Handing the encode path a
        // null pointer and a zero count is exactly the state a request with no character_refs
        // arrives in, so a shot that scopes out every sheet takes the untouched code path -- no
        // reference tokens, no denoise-mask forcing, no change to its output.
        // The MSR strip carries its own scope. It is ONE reference rather than an array, so
        // scoping it out is just making it inert -- and clearing the images too keeps the
        // shot's params indistinguishable from a request that never carried a strip.
        params.msr_segments      = nullptr;
        params.msr_segments_size = 0;
        if (base_params->msr_frames > 0 && base_params->msr_segments != nullptr) {
            const int* scope = base_params->msr_segments;
            const int count  = std::max(0, base_params->msr_segments_size);
            if (std::find(scope, scope + count, segment) == scope + count) {
                params.msr_background    = nullptr;
                params.msr_subjects      = nullptr;
                params.msr_subjects_size = 0;
                params.msr_frames        = 0;
            } else {
                LOG_INFO("LTXAV MSR scope: window %d takes the reference strip", segment + 1);
            }
        }
        if (!tass_source_ids.empty()) {
            params.character_ref_segments       = nullptr;
            params.character_ref_segment_counts = nullptr;
            if (tass_scoped) {
                segment_character_refs.clear();
                segment_character_ref_source_ids.clear();
                for (int i = 0; i < base_params->character_refs_size; ++i) {
                    const int count = std::max(0, base_params->character_ref_segment_counts[i]);
                    // Zero is inert, and it is also the one case where the flat index buffer is
                    // allowed to be null, so it has to be answered before the pointer arithmetic.
                    if (count == 0 || base_params->character_ref_segments == nullptr) {
                        continue;
                    }
                    const int* scope = base_params->character_ref_segments + tass_scope_offsets[i];
                    if (std::find(scope, scope + count, segment) == scope + count) {
                        continue;
                    }
                    segment_character_refs.push_back(base_params->character_refs[i]);
                    segment_character_ref_source_ids.push_back(tass_source_ids[i]);
                }
                params.character_refs           = segment_character_refs.empty() ? nullptr
                                                                                 : segment_character_refs.data();
                params.character_ref_source_ids = segment_character_ref_source_ids.empty()
                                                      ? nullptr
                                                      : segment_character_ref_source_ids.data();
                params.character_refs_size      = static_cast<int>(segment_character_refs.size());
                LOG_INFO("LTXAV TASS scope: window %d takes %d of %d character reference(s)",
                         segment + 1,
                         params.character_refs_size,
                         base_params->character_refs_size);
            } else {
                // Unscoped: every sheet, but still carrying the settled numbering so the ids a
                // scoped and an unscoped request produce are the same function of the array.
                params.character_ref_source_ids = tass_source_ids.data();
            }
        }
        if (chain_params->segment_video_frames != nullptr && chain_params->segment_video_frames[segment] > 0) {
            params.video_frames = chain_params->segment_video_frames[segment];
        }
        params.seed = base_params->seed < 0 ? base_params->seed : base_params->seed + segment;
        if (chain_params->segment_seeds != nullptr && chain_params->segment_seeds[segment] >= 0) {
            params.seed = chain_params->segment_seeds[segment];
        }
        if (chain_params->segment_prompts != nullptr && chain_params->segment_prompts[segment] != nullptr) {
            params.prompt = chain_params->segment_prompts[segment];
        }
        if (chain_params->segment_negative_prompts != nullptr &&
            chain_params->segment_negative_prompts[segment] != nullptr &&
            chain_params->segment_negative_prompts[segment][0] != '\0') {
            params.negative_prompt = chain_params->segment_negative_prompts[segment];
        }
        if (chain_params->segment_steps != nullptr && chain_params->segment_steps[segment] > 0) {
            params.sample_params.sample_steps = chain_params->segment_steps[segment];
            // A per-shot step count and a baked custom sigma schedule are
            // mutually exclusive -- the schedule already fixes the step count,
            // and it would silently win.
            params.sample_params.custom_sigmas       = nullptr;
            params.sample_params.custom_sigmas_count = 0;
        }
        if (chain_params->segment_cfg != nullptr && chain_params->segment_cfg[segment] >= 0.f) {
            params.sample_params.guidance.txt_cfg = chain_params->segment_cfg[segment];
        }
        // A supplied per-shot array is authoritative for every entry, zero included -- otherwise
        // "off for this one shot" would be unexpressible in a project that has the trim on.
        if (chain_params->segment_reference_head_trim != nullptr) {
            params.reference_head_trim = chain_params->segment_reference_head_trim[segment];
        }
        // A supplied per-shot array is authoritative, including when a shot's
        // count is zero. Only a chain that carries no beat array at all falls
        // back to whatever the base params hold (the single-shot CLI/HTTP form).
        if (chain_params->segment_beats != nullptr && chain_params->segment_beat_counts != nullptr) {
            params.beats      = chain_params->segment_beat_counts[segment] > 0
                                    ? chain_params->segment_beats[segment]
                                    : nullptr;
            params.beat_count = chain_params->segment_beat_counts[segment] > 0
                                    ? chain_params->segment_beat_counts[segment]
                                    : 0;
        }
        const int v2v_mode = chain_params->segment_v2v_modes == nullptr ? 0 : chain_params->segment_v2v_modes[segment];
        const bool has_v2v_control = chain_params->segment_control_frames != nullptr &&
                                    chain_params->segment_control_frame_counts != nullptr &&
                                    chain_params->segment_control_frames[segment] != nullptr &&
                                    chain_params->segment_control_frame_counts[segment] > 0;
        const bool has_v2v_latent = chain_params->segment_v2v_guide_latent_paths != nullptr &&
                                    chain_params->segment_v2v_guide_latent_paths[segment] != nullptr &&
                                    chain_params->segment_v2v_guide_latent_paths[segment][0] != '\0';
        const bool has_v2v_source = (v2v_mode == 0 && has_v2v_control) ||
                                    (v2v_mode == 1 && has_v2v_control) ||
                                    (v2v_mode == 2 && (has_v2v_control || has_v2v_latent));
        bool keyframe_at_start = false;
        const bool has_keyframes = !has_v2v_source &&
                                   chain_params->segment_keyframes != nullptr &&
                                   chain_params->segment_keyframe_counts != nullptr &&
                                   chain_params->segment_keyframes[segment] != nullptr &&
                                   chain_params->segment_keyframe_counts[segment] > 0;
        if (has_keyframes && chain_params->segment_keyframe_indices != nullptr &&
            chain_params->segment_keyframe_indices[segment] != nullptr) {
            for (int keyframe = 0; keyframe < chain_params->segment_keyframe_counts[segment]; ++keyframe) {
                if (chain_params->segment_keyframe_indices[segment][keyframe] == 0) {
                    keyframe_at_start = true;
                    break;
                }
            }
        }
        const bool fresh_scene = segment > 0 &&
                                 ((chain_params->segment_scene_cuts != nullptr && chain_params->segment_scene_cuts[segment] != 0) ||
                                  (chain_params->segment_init_images != nullptr &&
                                   chain_params->segment_init_images[segment] != nullptr &&
                                   chain_params->segment_init_images[segment]->data != nullptr) ||
                                  keyframe_at_start ||
                                  has_v2v_source);
        if (segment > 0 && !fresh_scene && overlap_frames >= params.video_frames) {
            LOG_ERROR("generate_video_chain: continuation overlap %d leaves no new frames in segment %d (%d frames)",
                      overlap_frames,
                      segment + 1,
                      params.video_frames);
            return fail();
        }
        if (fresh_scene) {
            params.init_image.data = nullptr;
            if (chain_params->segment_init_images != nullptr &&
                chain_params->segment_init_images[segment] != nullptr) {
                params.init_image = *chain_params->segment_init_images[segment];
            }
        }
        if (has_keyframes) {
            params.keyframes = chain_params->segment_keyframes[segment];
            params.keyframe_frame_indices = chain_params->segment_keyframe_indices != nullptr &&
                                             chain_params->segment_keyframe_indices[segment] != nullptr
                                                 ? const_cast<int*>(chain_params->segment_keyframe_indices[segment])
                                                 : nullptr;
            params.keyframes_size = chain_params->segment_keyframe_counts[segment];
        }
        if (has_v2v_source) {
            params.init_image.data = nullptr;
            params.control_frames = has_v2v_control ? chain_params->segment_control_frames[segment] : nullptr;
            params.control_frames_size = has_v2v_control ? chain_params->segment_control_frame_counts[segment] : 0;
            params.v2v_mode = v2v_mode;
            params.v2v_guide_latent_path = has_v2v_latent ? chain_params->segment_v2v_guide_latent_paths[segment] : nullptr;
            if (chain_params->segment_v2v_strengths != nullptr &&
                chain_params->segment_v2v_strengths[segment] >= 0.f) {
                params.strength = chain_params->segment_v2v_strengths[segment];
                params.v2v_guide_strength = chain_params->segment_v2v_strengths[segment];
            }
        }
        if (segment > 0 && !fresh_scene) {
            if (previous_tail.empty()) {
                LOG_ERROR("generate_video_chain: continuation segment %d has no prior latent tail", segment);
                return fail();
            }
            params.init_image.data = nullptr;
            params.cont_latent = previous_tail.data();
            params.cont_latent_width = tail_width;
            params.cont_latent_height = tail_height;
            params.cont_latent_frames = carried_frames;
            params.cont_latent_channels = tail_channels;
            if (avatar_chain) {
                params.audio_frame_offset = segment * (base_params->video_frames - overlap_frames);
            }
        } else {
            params.cont_latent = nullptr;
            params.cont_latent_frames = 0;
        }

        // A retaken shot is additionally pinned to the head of its unchanged
        // successor.  Read only the small latent head; the suffix itself is
        // decoded and spliced after this window has completed.
        std::vector<float> next_head;
        if (retake_active && segment + 1 < chain_params->n_segments) {
            const std::string next_path = bank_stem_for(chain_params, segment + 1) + ".bin";
            try {
                auto next = sd::load_tensor_from_file_as_tensor<float>(next_path);
                if (next.empty() || next.dim() < 4 ||
                    !extract_head(next.data(), static_cast<int>(next.shape()[0]), static_cast<int>(next.shape()[1]),
                                  static_cast<int>(next.shape()[2]), static_cast<int>(next.shape()[3]), next_head)) {
                    LOG_ERROR("generate_video_chain: could not restore retake end guide from %s", next_path.c_str());
                    return fail();
                }
                params.end_cont_latent = next_head.data();
                params.end_cont_latent_width = static_cast<int>(next.shape()[0]);
                params.end_cont_latent_height = static_cast<int>(next.shape()[1]);
                params.end_cont_latent_frames = carried_frames;
                params.end_cont_latent_channels = sd_ctx->sd->get_latent_channel();
            } catch (const std::exception& error) {
                LOG_ERROR("generate_video_chain: could not read retake end guide %s: %s", next_path.c_str(), error.what());
                return fail();
            }
        }

        std::string drive_audio_path;
        const int declared_drop = chain_params->segment_seam_drop_frames != nullptr
                                      ? chain_params->segment_seam_drop_frames[segment]
                                      : -1;
        // A-priori estimate. On an audio chain this is not merely the estimate, it IS the trim
        // (pin_derived_seam), because it is also what cut this segment's drive-audio window and
        // only equality keeps the shot on the song. So it has to be as close to the content's
        // preference as something knowable-before-the-render can be:
        //
        //   first seam of a chain -> seam_drop_default = overlap_px, the guide's decoded length
        //                            and the cut the 2026-07-28 eye test found clean;
        //   every seam after it   -> last_measured_seam_drop, what the advisory search actually
        //                            measured at the previous seam of THIS clip.
        //
        // Seams within one clip sit close together, so the carry-forward makes the applied trim
        // content-adaptive from the second seam on without ever letting it disagree with the audio
        // window. It applies on the pinned path too -- that is the whole point of measuring a trim
        // we are not going to apply here.
        const int seam_drop_estimate =
            (declared_drop < 0 && chain_params->cont_seam_drop_frames <= 0 &&
             last_measured_seam_drop >= 0)
                ? last_measured_seam_drop
                : seam_drop_default;
        const int seam_drop = segment == 0 || fresh_scene
                                  ? 0
                                  : std::min(params.video_frames,
                                             declared_drop >= 0 ? declared_drop
                                                                : (chain_params->cont_seam_drop_frames > 0
                                                                       ? chain_params->cont_seam_drop_frames
                                                                       : seam_drop_estimate));
        // Beat frames arrive on the shot's VISIBLE timeline -- what a viewer sees, and the only
        // frame of reference a caller can reasonably have. A continuation shot renders its seam
        // overlap first and trims it back off, so a beat at visible frame 0 sits `seam_drop`
        // frames into what is actually rendered. Shift them here rather than making every caller
        // (and every UI) know the trim. Beats that would land before the shot starts clamp to 0.
        std::vector<sd_ltx_beat_t> shifted_beats;
        if (seam_drop > 0 && params.beats != nullptr && params.beat_count > 0) {
            shifted_beats.assign(params.beats, params.beats + params.beat_count);
            for (auto& beat : shifted_beats) {
                beat.frame = std::max(0, beat.frame) + seam_drop;
            }
            params.beats = shifted_beats.data();
            LOG_INFO("LTX prompt relay: shifted %d beat(s) by the %d-frame seam drop for window %d",
                     params.beat_count, seam_drop, segment + 1);
        }
        if (chain_params->segment_audio_full != nullptr &&
            chain_params->segment_audio_full[segment] != nullptr &&
            chain_params->segment_audio_full[segment][0] != '\0') {
            if (chain_params->bank_dir == nullptr || chain_params->bank_dir[0] == '\0') {
                LOG_ERROR("generate_video_chain: per-shot drive audio requires a durable bank directory");
                return fail();
            }
            std::vector<float> shot_drive;
            uint32_t           shot_drive_channels = 0;
            if (!LONGCAT_AUDIO::load_wav_16k(chain_params->segment_audio_full[segment],
                                             shot_drive,
                                             shot_drive_channels,
                                             ltx_drive_audio_channels()) ||
                shot_drive.empty()) {
                LOG_ERROR("generate_video_chain: per-shot drive audio is not a readable WAV: %s",
                          chain_params->segment_audio_full[segment]);
                return fail();
            }
            drive_audio_path = std::string(chain_params->bank_dir) + "/aud_shot_" + std::to_string(segment) + ".wav";
            if (!write_ltx_drive_audio_wav(drive_audio_path, shot_drive, shot_drive_channels)) {
                LOG_ERROR("generate_video_chain: could not stage per-shot drive audio for window %d", segment + 1);
                return fail();
            }
            params.drive_audio_path = drive_audio_path.c_str();
        } else if (drive_audio.loaded()) {
            const int drop = seam_drop;
            // + drive_head_trimmed: an earlier shot discarded that many rendered frames off its
            // head, so the song is that far ahead of the output timeline from there on. Without
            // it every shot after a head-trimmed one would be re-cut against a stale clock and the
            // fix for shot 0 would break shot 1.
            const int64_t timeline_start =
                static_cast<int64_t>(timeline_frames()) - drop + drive_head_trimmed;
            const auto slice = drive_audio.window(timeline_start,
                                                  params.video_frames,
                                                  std::max(1, params.fps),
                                                  audio_offset);
            drive_audio_path = std::string(chain_params->bank_dir) + "/aud_" + std::to_string(segment) + ".wav";
            if (!write_ltx_drive_audio_wav(drive_audio_path, slice, drive_audio.channels)) {
                LOG_ERROR("generate_video_chain: could not stage drive audio for window %d", segment + 1);
                return fail();
            }
            params.drive_audio_path = drive_audio_path.c_str();
        } else if (chain_params->chain_audio_dir != nullptr && chain_params->chain_audio_dir[0] != '\0') {
            // Koblem's older windowed relip path sends one already-sliced WAV
            // per chain segment.  Keep it durable in the job bank and use it
            // when no whole-timeline drive signal was supplied.
            drive_audio_path = std::string(chain_params->chain_audio_dir) + "/aud_" +
                               std::to_string(segment) + ".wav";
            std::ifstream input(drive_audio_path, std::ios::binary);
            if (input.good()) {
                params.drive_audio_path = drive_audio_path.c_str();
            }
        }

        sd_image_t* segment_frames = nullptr;
        int segment_count = 0;
        sd_audio_t* segment_audio = nullptr;
        float* latent = nullptr;
        int width = 0;
        int height = 0;
        int frames = 0;
        int channels = 0;
        int reference_head_trim = 0;
        if (!generate_video_ex(sd_ctx,
                               &params,
                               &segment_frames,
                               &segment_count,
                               &segment_audio,
                               &latent,
                               &width,
                               &height,
                               &frames,
                               &channels,
                               &reference_head_trim) ||
            segment_frames == nullptr || segment_count <= 0 || latent == nullptr) {
            free_sd_audio(segment_audio);
            free(segment_frames);
            free(latent);
            LOG_ERROR("generate_video_chain: LTX window %d failed", segment + 1);
            return fail();
        }
        if (chain_params->bank_dir != nullptr && chain_params->bank_dir[0] != '\0' &&
            !save_video_latent(std::string(chain_params->bank_dir) + "/seg_" + std::to_string(segment) + ".bin",
                               latent,
                               width,
                               height,
                               frames,
                               channels)) {
            LOG_ERROR("generate_video_chain: could not save LTX bank for window %d", segment + 1);
            free_sd_audio(segment_audio);
            for (int frame = 0; frame < segment_count; ++frame) free(segment_frames[frame].data);
            free(segment_frames);
            free(latent);
            return fail();
        }
        if (segment + 1 < chain_params->n_segments && !extract_tail(latent, width, height, frames, channels)) {
            free_sd_audio(segment_audio);
            for (int frame = 0; frame < segment_count; ++frame) free(segment_frames[frame].data);
            free(segment_frames);
            free(latent);
            return fail();
        }
        if (chain_params->on_segment != nullptr) {
            // Frames AND audio are both still untrimmed here -- adopt_frames() applies the head
            // drop below (the seam overlap PLUS any reference head trim), and append_audio()
            // applies the matching audio_drop. So the preview is the shot exactly AS RENDERED,
            // overlap head and contaminated reference frame included, and its picture and sound
            // stay in sync with each other. Trimming belongs to the stitched timeline, not to a
            // preview of one shot.
            chain_params->on_segment(segment,
                                     segment_frames,
                                     segment_count,
                                     segment_audio,
                                     chain_params->on_segment_user);
        }
        free(latent);
        // The trim: an explicit pin from the caller is authoritative, and the PREDICTION is itself
        // a pin whenever the engine owns the audio -- because the drive slice was cut a priori
        // against exactly this number, and only equality keeps the shot on the song. The
        // content-adaptive search, which aligns the cut to the smoothest continuation rather than
        // to the prediction, is APPLIED only on a picture-only chain (or when a caller forces it),
        // where a miss costs nothing but a slightly different join.
        //
        // It still RUNS on the pinned path, because a measurement we do not apply here is exactly
        // what makes the NEXT seam of this clip both adaptive and exact: it becomes that seam's
        // prediction, and therefore also cuts that seam's drive-audio window.
        //
        // Using one drop for BOTH the video trim and this segment's audio head-drop keeps those two
        // aligned within the segment, and the next segment re-anchors its drive window to the true
        // accumulated timeline, so a miss cannot accumulate across seams. It still cannot undo the
        // fact that the shot was RENDERED against a mispositioned drive window -- which is why the
        // pin exists.
        int effective_seam_drop = seam_drop;
        // Beats pin the trim for exactly the reason audio does. This shot's beat
        // frames were already shifted by `seam_drop` (the a-priori estimate) so
        // that a beat authored at visible t=0 lands on the first frame the viewer
        // keeps. If the adaptive search then applies a different trim, every beat
        // in the shot is off by the difference -- baked into the pixels, and not
        // knowable at shift time because the search runs after the render.
        const bool caller_pinned_seam = declared_drop >= 0 || chain_params->cont_seam_drop_frames > 0;
        const bool seam_drop_pinned = caller_pinned_seam || pin_derived_seam || params.beat_count > 0;
        if (seam_drop > 0 && !stitched.empty() && segment_count > 2) {
            const int measured = ltxav_auto_trim_drop(stitched.back(), segment_frames, segment_count, seam_drop);
            if (!seam_drop_pinned) {
                // TRIPWIRE. The drive window for THIS segment was already cut against seam_drop, so
                // a measured trim that differs by delta offsets this segment against the song by
                // delta -- baked into the pixels, not fixable by re-muxing. With the pin above,
                // this branch is unreachable whenever there is any audio at all, so the warning
                // should never fire; if it does, a caller forced the adaptive search on over audio
                // (LTXAV_PIN_SEAM=0 or a negative cont_seam_drop_frames) or the pin has regressed.
                if (measured != seam_drop) {
                    const int delta = std::abs(measured - seam_drop);
                    LOG_INFO("generate_video_chain: window %d seam auto-trim -> drop %d (estimate was %d)",
                             segment + 1, measured, seam_drop);
                    if (engine_owns_audio) {
                        LOG_WARN("generate_video_chain: window %d drive audio was cut for a %d-frame trim "
                                 "but %d was applied — this shot sits %d frame(s) off the song; the next "
                                 "shot re-anchors to the true timeline so it cannot accumulate",
                                 segment + 1, seam_drop, measured, delta);
                    }
                }
                effective_seam_drop = measured;
            } else if (measured != seam_drop) {
                // ADVISORY ONLY. Applying this would move the picture off the audio window that
                // was already cut for `seam_drop`, which is the desync this pin exists to prevent.
                // It is still worth measuring and worth saying out loud: a large or wandering gap
                // means the prediction is a poor model of this clip's content, which is a real
                // (picture-side) finding even though nothing here acts on it.
                LOG_INFO("generate_video_chain: window %d seam auto-trim ADVISORY -> would have dropped "
                         "%d, applied the predicted %d (delta %d)%s",
                         segment + 1, measured, seam_drop, std::abs(measured - seam_drop),
                         caller_pinned_seam ? " — caller pin, not carried forward"
                                            : " — carried forward as the next seam's prediction");
            }
            // Feed the measurement forward unless the caller dictated this seam's trim: at the next
            // seam it becomes both the applied trim and the drive-audio cut, so the chain converges
            // on the content's preference without ever letting the two disagree.
            if (!caller_pinned_seam) {
                last_measured_seam_drop = measured;
            }
            seam_trims.push_back({seam_drop, effective_seam_drop, measured});
        }
        // The REFERENCE HEAD TRIM rides on the same output-side head drop as the seam trim: the
        // frames and this shot's own audio are cut by the same amount, and nothing before the
        // render moves. In practice the two never coexist -- the trim self-gates off continuation
        // shots, which are the only ones with a seam drop -- but summing is the honest expression
        // of "drop this many frames off the head", and both cuts are pure output bookkeeping.
        //
        // Doing it HERE rather than inside generate_video_ex is what keeps the durable bank
        // consistent: seg_<n>.bin holds the shot AS RENDERED (untrimmed), seg_<n>.audio holds its
        // untrimmed audio, and seg_<n>.len records what the timeline kept. A resume or a retake
        // re-derives the drop as `count - banked_kept` and applies it to BOTH, which reproduces
        // this cut exactly. Banking an already-trimmed audio sidecar next to an untrimmed latent
        // would silently trim the audio twice on every restore.
        if (reference_head_trim > 0) {
            LOG_INFO("generate_video_chain: window %d drops %d reference head frame(s)%s",
                     segment + 1,
                     reference_head_trim,
                     effective_seam_drop > 0 ? " on top of its seam drop" : "");
        }
        // With the trim OFF this is the historical `min(effective_seam_drop, segment_count)` to
        // the byte; the leave-one-frame floor only comes into force once a trim is in play.
        const int head_drop_limit = reference_head_trim > 0 ? std::max(0, segment_count - 1) : segment_count;
        const int audio_drop      = std::min(effective_seam_drop + reference_head_trim, head_drop_limit);
        // Only the part of the head drop that is a REFERENCE trim advances the drive clock -- and
        // only by as much as actually survived the head_drop_limit clamp. The seam part is overlap
        // the previous shot already showed.
        const int drive_skipped = std::max(0, audio_drop - effective_seam_drop);
        if (chain_params->segment_audio_track != nullptr) {
            if (!append_shot_track(chain_params->segment_audio_track[segment], audio_drop,
                                   segment_count - audio_drop)) {
                free_sd_audio(segment_audio);
                for (int frame = 0; frame < segment_count; ++frame) free(segment_frames[frame].data);
                free(segment_frames);
                return fail();
            }
        } else {
            append_audio(segment_audio, audio_drop, segment_count - audio_drop);
        }
        // Bank the sidecars next to seg_<i>.bin: this shot's own audio (the latent is video-only,
        // so a resumed prefix would otherwise be silent) and its KEPT length (so a retake
        // reproduces these exact timeline offsets instead of re-deriving a drop). Both are
        // best-effort -- a failure here must not lose a completed render.
        if (chain_params->bank_dir != nullptr && chain_params->bank_dir[0] != '\0') {
            const std::string stem = std::string(chain_params->bank_dir) + "/seg_" + std::to_string(segment);
            write_seg_len(stem + ".len", segment_count - audio_drop);
            if (segment_audio != nullptr) {
                write_seg_audio(stem + ".audio", segment_audio);
            }
        }
        free_sd_audio(segment_audio);
        adopt_frames(segment_frames, segment_count, audio_drop);
        // Advance the drive clock AFTER this shot's own windows are cut and its audio appended:
        // the skew applies to everything downstream of the discarded frames, not to them.
        if (drive_skipped > 0) {
            drive_head_trimmed += drive_skipped;
            LOG_INFO("generate_video_chain: window %d discarded %d rendered frame(s) of drive audio "
                     "no other shot covers -> the song now runs %lld frame(s) ahead of the output "
                     "timeline; later drive windows and the deliverable track follow it",
                     segment + 1, drive_skipped, (long long)drive_head_trimmed);
        }
        // Everything but the retained suffix is final now; hand it over before the next window
        // allocates, so peak RAM is one segment plus the window rather than the whole clip.
        flush_window(false);
        if (ltx_chain && segment + 1 < chain_params->n_segments) {
            // previous_tail and the adopted frames are host-owned now; no GPU
            // allocation from this window is needed by the continuation setup.
            sd_ctx->sd->reclaim_ltx_chain_window_gpu_memory();
        }
    }

    // Retake leaves downstream shots untouched: restore their durable latents,
    // decode them, and apply the same declared seam policy used by the live
    // chain.  This avoids re-sampling an entire timeline for a one-shot edit.
    if (retake_active) {
        for (int segment = sample_end + 1; segment < chain_params->n_segments; ++segment) {
            const std::string path = bank_stem_for(chain_params, segment) + ".bin";
            int segment_count = 0;
            sd_image_t* segment_frames = decode_banked_video_latent(sd_ctx, path, &segment_count);
            if (segment_frames == nullptr || segment_count <= 0) {
                free(segment_frames);
                LOG_ERROR("generate_video_chain: could not decode retake suffix bank %s", path.c_str());
                return fail();
            }
            const int declared_drop = chain_params->segment_seam_drop_frames != nullptr
                                          ? chain_params->segment_seam_drop_frames[segment]
                                          : -1;
            int drop = std::min(segment_count,
                                declared_drop >= 0 ? declared_drop
                                                   : (chain_params->cont_seam_drop_frames > 0
                                                          ? chain_params->cont_seam_drop_frames
                                                          : seam_drop_default));
            // LENGTH PIN -- the same one the RESUME path applies above, and for the same reason.
            // A suffix segment's kept length is whatever it banked, which is NOT re-derivable from
            // the seam policy alone: an auto-trim search lands where it lands, and a reference
            // head trim adds to it. Re-deriving here reproduces a DIFFERENT prefix length and
            // slides every downstream shot (and its audio) along the timeline. The resume path had
            // this pin; the retake suffix never got it, so a retake on a project with references
            // was the one path that could still drift.
            const int banked_kept = read_seg_len(bank_stem_for(chain_params, segment) + ".len");
            if (banked_kept >= 0 && banked_kept <= segment_count) {
                const int pinned_drop = segment_count - banked_kept;
                if (pinned_drop != drop) {
                    LOG_INFO("generate_video_chain: retake suffix %d restored to its banked length "
                             "(kept %d, drop %d, would have re-derived %d)",
                             segment + 1, banked_kept, pinned_drop, drop);
                }
                drop = pinned_drop;
            }
            if (chain_params->segment_audio_track != nullptr &&
                chain_params->segment_audio_track[segment] != nullptr &&
                chain_params->segment_audio_track[segment][0] != '\0' &&
                !append_shot_track(chain_params->segment_audio_track[segment], drop,
                                   segment_count - drop)) {
                for (int frame = 0; frame < segment_count; ++frame) free(segment_frames[frame].data);
                free(segment_frames);
                return fail();
            }
            // Restore this SUFFIX segment's own audio, exactly as the prefix path above does and
            // for the same reason: a retake on a project whose audio is MODEL-GENERATED rather
            // than a supplied track otherwise plays everything after the retaken shot silent.
            // The prefix path grew this restore; the retake suffix never did, so the picture came
            // back and the sound did not -- and a retake is precisely when that bites, since
            // retaking shot i is the only way to reach this loop.
            if (!track_audio.loaded() &&
                (chain_params->segment_audio_track == nullptr ||
                 chain_params->segment_audio_track[segment] == nullptr ||
                 chain_params->segment_audio_track[segment][0] == '\0')) {
                const std::string audio_path = bank_stem_for(chain_params, segment) + ".audio";
                if (sd_audio_t* banked_audio = read_seg_audio(audio_path); banked_audio != nullptr) {
                    append_audio(banked_audio, drop, segment_count - drop);
                    free_sd_audio(banked_audio);
                }
            }
            adopt_frames(segment_frames, segment_count, drop);
            flush_window(false);
        }
        LOG_INFO("generate_video_chain: retook segment %d and spliced %d banked suffix segment(s)",
                 sample_end, chain_params->n_segments - sample_end - 1);
    }

    // The deliverable track is cut against the FULL timeline, including anything already flushed.
    const int64_t final_frame_count = static_cast<int64_t>(timeline_frames());
    if (final_frame_count <= 0) {
        return fail();
    }
    if (streaming) {
        // Drain the residual window and hand back metadata only -- the frames were flushed and
        // freed as they became final, which is the entire point. Peak frame RAM stayed at roughly
        // one segment plus WINDOW_KEEP instead of the whole clip.
        flush_window(true);
        LOG_INFO("generate_video_chain: streamed %lld frame(s) out; peak frame memory stayed at one "
                 "segment plus the window instead of the whole timeline",
                 (long long)flushed_total);
        if (!seam_trims.empty()) {
            std::string detail;
            int worst = 0;
            int widest_advice = 0;
            for (const auto& trim : seam_trims) {
                detail += " " + std::to_string(trim.applied);
                if (trim.predicted != trim.applied) {
                    detail += "(pred " + std::to_string(trim.predicted) + ")";
                }
                if (trim.measured != trim.applied) {
                    detail += "[search wanted " + std::to_string(trim.measured) + "]";
                }
                worst = std::max(worst, std::abs(trim.applied - trim.predicted));
                widest_advice = std::max(widest_advice, std::abs(trim.measured - trim.applied));
            }
            LOG_INFO("generate_video_chain: seam trims applied:%s | worst predicted-vs-applied gap %d "
                     "frame(s)%s | widest unapplied search advice %d frame(s)",
                     detail.c_str(), worst,
                     worst == 0 ? " — every shot sits exactly where its drive-audio window put it"
                                : " (that shot sits that far off the song)",
                     widest_advice);
        }
        *frames_out = nullptr;
        *num_frames_out = static_cast<int>(final_frame_count);
    } else {
        auto* output = static_cast<sd_image_t*>(malloc(stitched.size() * sizeof(sd_image_t)));
        if (output == nullptr) {
            return fail();
        }
        std::memcpy(output, stitched.data(), stitched.size() * sizeof(sd_image_t));
        *frames_out = output;
        *num_frames_out = static_cast<int>(stitched.size());
    }
    if (audio_out != nullptr) {
        if (track_audio.loaded()) {
            // Starts at drive_head_trimmed, not 0: the render consumed that much of the song
            // driving frames it then discarded, so output timeline 0 IS song frame
            // drive_head_trimmed. Handing back the song from 0 while the picture sings from
            // drive_head_trimmed is a straight lip desync, and this is the DELIVERABLE path --
            // the track is copied verbatim, so the error lands whole rather than smeared by the
            // audio VAE.
            //
            // One contiguous window can only carry ONE offset. Every shot but a bare t2v opener
            // self-gates the reference trim off, so in practice only shot 0 ever moves this; a
            // chain where a later shot trims too would need the deliverable accumulated per shot,
            // and says so rather than shipping a wrong number quietly.
            if (drive_head_trimmed > 0) {
                LOG_INFO("generate_video_chain: deliverable track starts %lld frame(s) into the "
                         "supplied audio, matching the reference head trim the picture took",
                         (long long)drive_head_trimmed);
            }
            const auto track = track_audio.window(drive_head_trimmed,
                                                  final_frame_count,
                                                  std::max(1, base_params->fps),
                                                  audio_offset);
            *audio_out = make_ltx_chain_audio(track, track_audio.sample_rate, track_audio.channels);
            if (*audio_out == nullptr) {
                LOG_WARN("generate_video_chain: could not allocate requested deliverable audio track");
            }
        } else {
            *audio_out = make_ltx_chain_audio(chain_audio_samples, chain_audio_rate, chain_audio_channels);
            if (*audio_out == nullptr && !chain_audio_samples.empty()) {
                LOG_WARN("generate_video_chain: could not allocate stitched audio");
            }
        }
    }
    LOG_INFO("generate_video_chain: stitched %d LTX windows -> %d frames",
             chain_params->n_segments,
             *num_frames_out);
    // Reclaim this job's GPU working set at job end. The between-window reclaim
    // above skips the FINAL window, so without this the last window's committed
    // VMM pool high-water + cuDNN conv3d reorder cache survive into the NEXT job
    // on this persistent worker — starving the next job's text-encoder graph-cut
    // cudaMalloc (a fresh alloc outside the pool) and OOMing it. Uses the same
    // upstream-native trim the between-window path already uses.
    sd_ctx->sd->reclaim_ltx_chain_window_gpu_memory();
    return true;
}

class ScopedWanVaceEnvironment {
public:
    ScopedWanVaceEnvironment() {
        for (const char* name : {"VACE_SKIP_BLOCKS", "VACE_STRENGTH_TAIL", "VACE_STRENGTH_ANCHOR_FRAMES"}) {
            const char* value = getenv(name);
            values.emplace_back(name, value != nullptr, value != nullptr ? value : "");
        }
    }

    ~ScopedWanVaceEnvironment() {
        for (const auto& [name, was_set, value] : values) {
            if (was_set) {
                setenv(name.c_str(), value.c_str(), 1);
            } else {
                unsetenv(name.c_str());
            }
        }
    }

private:
    std::vector<std::tuple<std::string, bool, std::string>> values;
};

SD_API bool generate_wan_vace_chain(sd_ctx_t*                         sd_ctx,
                                    const sd_vid_gen_params_t*        base_params,
                                    const sd_wan_vace_chain_params_t* chain_params,
                                    sd_image_t**                      frames_out,
                                    int*                              num_frames_out,
                                    sd_audio_t**                      audio_out) {
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
        frames_out == nullptr || num_frames_out == nullptr || chain_params->n_segments < 1) {
        LOG_ERROR("generate_wan_vace_chain: invalid arguments");
        return false;
    }
    const auto& description = sd_ctx->sd->diffusion_model->get_desc();
    if (description != "Wan2.1-VACE-1.3B" && description != "Wan2.x-VACE-14B") {
        LOG_ERROR("generate_wan_vace_chain requires a Wan VACE diffusion model (got %s)", description.c_str());
        return false;
    }

    const int overlap = chain_params->overlap_frames > 0 ? chain_params->overlap_frames : 5;
    const int discard = chain_params->discard_tail_frames > 0 ? chain_params->discard_tail_frames : 4;
    const int drop_latent_tail = std::max(0, chain_params->drop_latent_tail_frames > 0
                                                  ? chain_params->drop_latent_tail_frames
                                                  : 1);
    const bool is_t2v = base_params->init_image.data == nullptr;
    ScopedWanVaceEnvironment restore_environment;
    std::vector<sd_image_t> stitched;
    std::vector<sd_image_t> control_tail;
    std::vector<float> prior_latent;
    int latent_width = 0;
    int latent_height = 0;
    int latent_frames = 0;
    int latent_channels = 0;

    auto clear_tail = [&]() {
        for (auto& frame : control_tail) {
            free(frame.data);
        }
        control_tail.clear();
    };
    auto fail = [&]() {
        clear_tail();
        for (auto& frame : stitched) {
            free(frame.data);
        }
        stitched.clear();
        return false;
    };

    if (chain_params->start_segment < 0 || chain_params->start_segment >= chain_params->n_segments) {
        LOG_ERROR("generate_wan_vace_chain: invalid resume segment %d/%d",
                  chain_params->start_segment, chain_params->n_segments);
        return false;
    }
    if (chain_params->start_segment > 0) {
        if (chain_params->resume_control_frames != nullptr && chain_params->resume_control_frames_size > 0 &&
            chain_params->resume_latent != nullptr && chain_params->resume_latent_width > 0 &&
            chain_params->resume_latent_height > 0 && chain_params->resume_latent_frames > 0 &&
            chain_params->resume_latent_channels > 0) {
            for (int frame = 0; frame < chain_params->resume_control_frames_size; ++frame) {
                sd_image_t copy = copy_video_frame(chain_params->resume_control_frames[frame]);
                if (copy.data == nullptr) {
                    clear_tail();
                    LOG_ERROR("generate_wan_vace_chain: could not copy resume control tail");
                    return false;
                }
                control_tail.push_back(copy);
            }
            latent_width = chain_params->resume_latent_width;
            latent_height = chain_params->resume_latent_height;
            latent_frames = chain_params->resume_latent_frames;
            latent_channels = chain_params->resume_latent_channels;
            const size_t count = static_cast<size_t>(latent_width) * latent_height * latent_frames * latent_channels;
            prior_latent.assign(chain_params->resume_latent, chain_params->resume_latent + count);
        } else if (chain_params->bank_dir != nullptr && chain_params->bank_dir[0] != '\0') {
            for (int segment = 0; segment < chain_params->start_segment; ++segment) {
                // NOT bank_stem_for: Wan-VACE chains carry sd_wan_vace_chain_params_t, a
                // different struct with no per-shot bank overrides. Take selection is an LTX
                // Director feature and Wan-VACE has no take model, so this stays single-bank.
                const std::string path = std::string(chain_params->bank_dir) + "/seg_" + std::to_string(segment) + ".bin";
                int count = 0;
                sd_image_t* frames = decode_banked_video_latent(sd_ctx, path, &count);
                if (frames == nullptr || count <= 0) {
                    free(frames);
                    return fail();
                }
                const bool continuation = segment > 0;
                const int first_keep = continuation ? std::min(overlap, count) : 0;
                const int last_keep = continuation ? std::max(first_keep, count - discard) : count;
                if (segment + 1 == chain_params->start_segment) {
                    try {
                        auto latent = sd::load_tensor_from_file_as_tensor<float>(path);
                        if (latent.empty() || latent.dim() < 4) {
                            throw std::runtime_error("invalid latent shape");
                        }
                        latent_width = static_cast<int>(latent.shape()[0]);
                        latent_height = static_cast<int>(latent.shape()[1]);
                        latent_frames = static_cast<int>(latent.shape()[2]);
                        latent_channels = static_cast<int>(latent.shape()[3]);
                        prior_latent.assign(latent.data(), latent.data() + latent.numel());
                    } catch (const std::exception& error) {
                        LOG_ERROR("generate_wan_vace_chain: could not restore latent %s: %s", path.c_str(), error.what());
                        for (int i = 0; i < count; ++i) free(frames[i].data);
                        free(frames);
                        return fail();
                    }
                    const int tail_start = std::max(first_keep, last_keep - overlap);
                    for (int frame = tail_start; frame < last_keep; ++frame) {
                        sd_image_t copy = copy_video_frame(frames[frame]);
                        if (copy.data == nullptr) {
                            for (int i = 0; i < count; ++i) free(frames[i].data);
                            free(frames);
                            return fail();
                        }
                        control_tail.push_back(copy);
                    }
                }
                for (int frame = 0; frame < count; ++frame) {
                    if (frame >= first_keep && frame < last_keep) {
                        stitched.push_back(frames[frame]);
                    } else {
                        free(frames[frame].data);
                    }
                }
                free(frames);
            }
        } else {
            LOG_ERROR("generate_wan_vace_chain: resume state is incomplete");
            return false;
        }
    }

    LOG_INFO("generate_wan_vace_chain: %d windows, overlap=%d, discard=%d, mode=%s",
             chain_params->n_segments, overlap, discard, is_t2v ? "t2v" : "i2v");
    for (int segment = chain_params->start_segment; segment < chain_params->n_segments; ++segment) {
        const bool continuation = segment > 0;
        const bool need_latent = true;
        sd_vid_gen_params_t params = *base_params;
        params.seed = base_params->seed < 0 ? base_params->seed : base_params->seed + segment;
        if (chain_params->segment_prompts != nullptr && chain_params->segment_prompts[segment] != nullptr) {
            params.prompt = chain_params->segment_prompts[segment];
        }

        setenv("VACE_SKIP_BLOCKS", "0", 1);
        if (continuation) {
            if (prior_latent.empty() || control_tail.empty()) {
                LOG_ERROR("generate_wan_vace_chain: continuation %d has no prior context", segment);
                return fail();
            }
            params.control_frames = control_tail.data();
            params.control_frames_size = static_cast<int>(control_tail.size());
            params.vace_cont_latent = prior_latent.data();
            params.vace_cont_latent_width = latent_width;
            params.vace_cont_latent_height = latent_height;
            params.vace_cont_latent_frames = latent_frames;
            params.vace_cont_latent_channels = latent_channels;
            params.vace_cont_frames = overlap;
            params.vace_cont_latent_drop_tail = segment == 1 ? 0 : drop_latent_tail;
            params.vace_strength = base_params->vace_strength;
            // The carried latent is the continuity anchor.  Do not re-inject a
            // still image into later windows unless a caller deliberately uses
            // a fresh chain request for that experiment.
            params.init_image.data = nullptr;
            const int anchor = (std::min(overlap, std::max(1, params.video_frames)) - 1) / 4 + 1;
            const char* tail = getenv("WAN_VACE_STRENGTH_TAIL");
            setenv("VACE_STRENGTH_TAIL", tail != nullptr && tail[0] != '\0' ? tail : "0.2", 1);
            const char* configured_anchor = getenv("WAN_VACE_STRENGTH_ANCHOR_FRAMES");
            setenv("VACE_STRENGTH_ANCHOR_FRAMES",
                   configured_anchor != nullptr && configured_anchor[0] != '\0'
                       ? configured_anchor
                       : std::to_string(anchor).c_str(),
                   1);
        } else if (is_t2v) {
            params.control_frames = nullptr;
            params.control_frames_size = 0;
            params.vace_cont_latent = nullptr;
            params.vace_cont_latent_frames = 0;
            params.vace_strength = 0.0f;
            unsetenv("VACE_STRENGTH_TAIL");
            unsetenv("VACE_STRENGTH_ANCHOR_FRAMES");
        } else {
            const char* tail = getenv("WAN_VACE_I2V_STRENGTH_TAIL");
            const char* anchor = getenv("WAN_VACE_I2V_STRENGTH_ANCHOR_FRAMES");
            setenv("VACE_STRENGTH_TAIL", tail != nullptr && tail[0] != '\0' ? tail : "0.5", 1);
            setenv("VACE_STRENGTH_ANCHOR_FRAMES", anchor != nullptr && anchor[0] != '\0' ? anchor : "3", 1);
        }

        sd_image_t* segment_frames = nullptr;
        int segment_count = 0;
        sd_audio_t* segment_audio = nullptr;
        float* exported_latent = nullptr;
        int out_width = 0;
        int out_height = 0;
        int out_frames = 0;
        int out_channels = 0;
        const bool generated = generate_video_ex(sd_ctx,
                                                 &params,
                                                 &segment_frames,
                                                 &segment_count,
                                                 &segment_audio,
                                                 &exported_latent,
                                                 &out_width,
                                                 &out_height,
                                                 &out_frames,
                                                 &out_channels,
                                                 // Wan VACE has no TASS references; the trim
                                                 // resolves to zero here by construction.
                                                 nullptr);
        free_sd_audio(segment_audio);
        if (!generated || segment_frames == nullptr || segment_count <= 0 ||
            exported_latent == nullptr) {
            LOG_ERROR("generate_wan_vace_chain: window %d failed", segment + 1);
            free(segment_frames);
            free(exported_latent);
            return fail();
        }

        {
            const size_t count = static_cast<size_t>(out_width) * out_height * out_frames * out_channels;
            prior_latent.assign(exported_latent, exported_latent + count);
            latent_width = out_width;
            latent_height = out_height;
            latent_frames = out_frames;
            latent_channels = out_channels;
            if (chain_params->bank_dir != nullptr && chain_params->bank_dir[0] != '\0') {
                sd::Tensor<float> bank({latent_width, latent_height, latent_frames, latent_channels, 1});
                std::memcpy(bank.data(), prior_latent.data(), count * sizeof(float));
                if (!save_banked_video_latent(std::string(chain_params->bank_dir) + "/seg_" +
                                                  std::to_string(segment) + ".bin",
                                              bank)) {
                    LOG_ERROR("generate_wan_vace_chain: could not save segment %d bank", segment);
                    free(exported_latent);
                    for (int frame = 0; frame < segment_count; ++frame) free(segment_frames[frame].data);
                    free(segment_frames);
                    return fail();
                }
            }
        }
        free(exported_latent);

        const int first_keep = continuation ? std::min(overlap, segment_count) : 0;
        const int last_keep = continuation ? std::max(first_keep, segment_count - discard) : segment_count;
        if (segment + 1 < chain_params->n_segments) {
            clear_tail();
            const int tail_start = std::max(first_keep, last_keep - overlap);
            for (int frame = tail_start; frame < last_keep; ++frame) {
                sd_image_t copy = copy_video_frame(segment_frames[frame]);
                if (copy.data == nullptr) {
                    for (int i = 0; i < segment_count; ++i) {
                        free(segment_frames[i].data);
                    }
                    free(segment_frames);
                    LOG_ERROR("generate_wan_vace_chain: could not copy continuation pixel tail");
                    return fail();
                }
                control_tail.push_back(copy);
            }
        }
        for (int frame = 0; frame < segment_count; ++frame) {
            if (frame >= first_keep && frame < last_keep) {
                stitched.push_back(segment_frames[frame]);
            } else {
                free(segment_frames[frame].data);
            }
        }
        free(segment_frames);
        if (chain_params->on_segment != nullptr && last_keep > first_keep) {
            chain_params->on_segment(segment,
                                     stitched.data() + stitched.size() - (last_keep - first_keep),
                                     last_keep - first_keep,
                                     need_latent ? prior_latent.data() : nullptr,
                                     need_latent ? latent_width : 0,
                                     need_latent ? latent_height : 0,
                                     need_latent ? latent_frames : 0,
                                     need_latent ? latent_channels : 0,
                                     chain_params->on_segment_user);
        }
    }

    clear_tail();
    if (stitched.empty()) {
        LOG_ERROR("generate_wan_vace_chain: no frames produced");
        return false;
    }
    sd_image_t* output = static_cast<sd_image_t*>(malloc(stitched.size() * sizeof(sd_image_t)));
    if (output == nullptr) {
        for (auto& frame : stitched) {
            free(frame.data);
        }
        LOG_ERROR("generate_wan_vace_chain: output allocation failed");
        return false;
    }
    std::memcpy(output, stitched.data(), stitched.size() * sizeof(sd_image_t));
    *frames_out = output;
    *num_frames_out = static_cast<int>(stitched.size());
    LOG_INFO("generate_wan_vace_chain: stitched %d windows -> %d frames",
             chain_params->n_segments, *num_frames_out);
    return true;
}

SD_API void free_sd_images(sd_image_t* result_images, int num_images) {
    if (result_images == nullptr) {
        return;
    }

    for (int i = 0; i < num_images; ++i) {
        if (result_images[i].data != nullptr) {
            free(result_images[i].data);
            result_images[i].data = nullptr;
        }
    }

    free(result_images);
}
