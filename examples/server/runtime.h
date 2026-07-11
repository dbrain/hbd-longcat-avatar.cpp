#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <json.hpp>
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "stable-diffusion.h"

using json = nlohmann::json;

struct ArgOptions;
struct SDContextParams;
struct AsyncJobManager;
namespace longcat_avatar { class WorkerSession; }

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    std::string serve_html_path;
    // FLUX.2-Klein edit-variant DiT path. The BASE DiT comes from the context
    // params (--diffusion-model); this is the EDIT DiT (--diffusion-model-edit),
    // swapped in per request. Empty = single-model (back-compat) mode.
    std::string diffusion_model_edit_path;
    // Additional named base-DiT variants for the per-request "model" selector, beyond
    // base/edit. Format: "name=path;name=path" (--diffusion-model-variants). Each entry
    // is a self-contained GGUF (e.g. a LoRA folded into the nvfp4 base). Empty = none.
    std::string diffusion_model_variants_spec;
    bool normal_exit = false;
    bool verbose     = false;
    bool color       = false;

    ArgOptions get_options();
    bool validate();
    bool resolve_and_validate();
    std::string to_string() const;
};

struct LoraEntry {
    std::string name;
    std::string path;
    std::string fullpath;
};

struct UpscalerEntry {
    std::string name;
    std::string path;
    std::string fullpath;
    std::string model_name;
    int scale = 4;
};

// Tracks which FLUX.2-Klein DiT variant is currently resident on the GPU, plus the
// admin drain/unload state for the external GPU gate. Mutated only on the serial
// async worker thread (swaps) + the HTTP threads (atomics), so no extra locking is
// needed beyond the existing sd_ctx_mutex around generation.
struct ModelSwapState {
    std::string base_path;   // --diffusion-model
    std::string edit_path;   // --diffusion-model-edit (empty = single-model mode)
    // Every selectable DiT variant name -> its GGUF path. Always contains "base" (and
    // "edit" when configured); extra entries come from --diffusion-model-variants. The
    // per-request "model" field is resolved through this map. Populated once at startup.
    std::map<std::string, std::string> variants;
    // The variant name whose weights are resident in sd_ctx right now (a key of `variants`).
    std::string loaded_variant = "base";
    std::atomic<bool> loaded{true};       // false after /v1/admin/unload freed VRAM
    std::atomic<bool> draining{false};    // true after /v1/admin/drain
};

// Parse a "name=path;name=path" spec into a variant map, seeded with base/edit. `base_path`
// registers as "base"; a non-empty `edit_path` as "edit". Spec entries add/override by name.
// Blank names/paths and malformed entries are skipped. Defined in runtime.cpp.
std::map<std::string, std::string> build_model_variants(const std::string& spec,
                                                        const std::string& base_path,
                                                        const std::string& edit_path);

struct ServerRuntime {
    sd_ctx_t* sd_ctx;
    std::mutex* sd_ctx_mutex;
    const SDSvrParams* svr_params;
    const SDContextParams* ctx_params;
    const SDGenerationParams* default_gen_params;
    std::vector<LoraEntry>* lora_cache;
    std::mutex* lora_mutex;
    std::vector<UpscalerEntry>* upscaler_cache;
    std::mutex* upscaler_mutex;
    AsyncJobManager* async_job_manager;
    ModelSwapState* model_swap;
    // Worker-isolation (flux2 image mode): when non-null, the GPU compute runs in
    // a forked CUDA-owning child and `sd_ctx` is null in this (parent) runtime.
    // The async worker routes img_gen jobs through worker->render_image(), and
    // /v1/admin/unload SIGKILLs the child. Null = legacy in-process (sd_ctx set).
    longcat_avatar::WorkerSession* worker = nullptr;
    // LTXAV video-chain mode (LTX_VIDEO_ISOLATION). When true, the async worker routes
    // VidGen jobs through worker->render_video_chain(job.vid_chain_request_json) (the chain
    // runs in the CUDA child), instead of the in-process single-segment execute_vid_gen_job.
    bool ltx_video_mode = false;
};

struct ImgGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "png";
    int output_compression    = 100;
    // Requested DiT variant ("base" | "edit"), parsed from the optional top-level
    // "model" field of the img_gen body. Empty = caller did not specify, so the
    // worker renders with whatever variant is currently loaded (no forced swap).
    std::string model_variant;

    sd_img_gen_params_t to_sd_img_gen_params_t() {
        return gen_params.to_sd_img_gen_params_t();
    }
};

struct VidGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "webm";
    int output_compression    = 100;

    sd_vid_gen_params_t to_sd_vid_gen_params_t() {
        return gen_params.to_sd_vid_gen_params_t();
    }
};

std::string base64_encode(const std::vector<uint8_t>& bytes);
std::string normalize_output_format(std::string output_format);
std::vector<std::string> supported_img_output_formats(bool allow_webp = true);
std::vector<std::string> supported_vid_output_formats();
bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message);
bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message);
std::string video_mime_type(const std::string& output_format);
bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode);
std::string unsupported_generation_mode_error(SDMode mode);
void refresh_lora_cache(ServerRuntime& rt);
std::string get_lora_full_path(ServerRuntime& rt, const std::string& path);
void refresh_upscaler_cache(ServerRuntime& rt);
int64_t unix_timestamp_now();
