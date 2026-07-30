#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <map>
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

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    std::string serve_html_path;
    // Optional alternate DiT used by the worker-isolated `model: "edit"`
    // contract. A variant change recycles the CUDA child instead of retaining
    // old allocations in a long-lived context.
    std::string diffusion_model_edit_path;
    // Extra named DiT variants for worker-isolated deployments. Format:
    // "name=path;name=path". Each request's top-level `model` selects one;
    // the supervisor destroys the old CUDA-owning process before spawning the
    // requested variant, so switches cannot retain the old model in VRAM.
    std::string diffusion_model_variants_spec;
    bool normal_exit = false;
    bool verbose     = false;
    bool color       = false;

    ArgOptions get_options();
    bool validate();
    bool resolve_and_validate();
    std::string to_string() const;
};

// One adapter in --lora-model-dir, plus whatever `loras.json` in that same directory says about
// it. The manifest half exists because a file name cannot carry the two things a CLIENT must know
// to use the adapter correctly, and getting either wrong is silent — the render succeeds and looks
// like the adapter did nothing:
//
//   * the multiplier it was TRAINED at. None of these adapters ship `.alpha` tensors, so the
//     engine applies `multiplier` raw; an adapter trained at alpha/r = 2.0 rendered at 1.0 is
//     simply half an adapter, and nothing on the wire says so.
//   * the TRIGGER PHRASE some adapters need in the prompt before they do anything at all.
//
// It lives NEXT TO THE WEIGHTS on purpose. The directory is bind-mounted, so adding an adapter is
// "drop the .gguf, add a manifest entry" — no engine rebuild, no client rebuild, and no second
// hand-maintained list of file names anywhere to drift out of step with the directory.
// Every field is optional; a directory with no `loras.json` behaves exactly as it did before.
struct LoraEntry {
    std::string name;
    std::string path;
    std::string fullpath;
    // ── from loras.json, empty/absent when the manifest says nothing ──
    std::string label;
    std::string blurb;
    // < 0 means "unstated" — NOT 1.0. A client must be able to tell "the manifest says render this
    // at 1.0" from "the manifest is silent", because only the first is a claim about the adapter.
    float default_multiplier = -1.0f;
    std::string trigger;
    bool trigger_required = false;
};

struct UpscalerEntry {
    std::string name;
    std::string path;
    std::string fullpath;
    std::string model_name;
    int scale = 4;
};

// Server-side ownership state for cooperative GPU sharing. The model context
// remains owned by main; this only gates new work and records DiT residency.
struct GPUSharingState {
    std::atomic<bool> draining{false};
    std::atomic<bool> diffusion_loaded{true};
    // Logical name of the DiT variant currently resident (FLUX.2-Klein dual-DiT:
    // "base"/"edit"). Unlike the atomics above this is guarded by
    // ServerRuntime::sd_ctx_mutex — only the serialized async worker reads/writes
    // it around a swap. new_sd_ctx() loads --diffusion-model (base) at boot.
    std::string loaded_variant = "base";
};

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
    GPUSharingState* gpu_sharing;
};

bool runtime_is_draining(const ServerRuntime& runtime);
std::map<std::string, std::string> runtime_diffusion_model_variants(const ServerRuntime& runtime);

struct ImgGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "png";
    int output_compression    = 100;
    // Optional top-level "model":"base"|"edit" DiT selector (FLUX.2-Klein dual
    // DiT). Empty = leave the resident DiT untouched (single-model byte-identical).
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
bool base64_decode(const std::string& text, std::vector<uint8_t>& bytes);
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
// Scan a LoRA directory and read its `loras.json`. FILESYSTEM ONLY — no CUDA, no model context —
// which is what lets the CUDA-free supervisor answer `/sdapi/v1/loras` itself instead of proxying
// it into the worker. That matters: koblem asks for this list on every video-tab bootstrap, and
// proxying would cold-start a ~16 GB CUDA child to list a directory. Same reasoning as the
// supervisor's `DELETE /ltx/v1/job`.
std::vector<LoraEntry> scan_lora_dir(const std::string& lora_model_dir);

// ONE wire shape for a LoRA entry, shared by the worker's `/sdapi/v1/loras` and the supervisor's.
// The two answer the same route depending on isolation mode; a client that saw different JSON from
// each would be debugging a phantom.
nlohmann::json lora_entry_json(const LoraEntry& entry);

void refresh_lora_cache(ServerRuntime& rt);
std::string get_lora_full_path(ServerRuntime& rt, const std::string& path);
void refresh_upscaler_cache(ServerRuntime& rt);
int64_t unix_timestamp_now();
