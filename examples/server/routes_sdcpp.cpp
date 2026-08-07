#include "routes.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "async_jobs.h"
#include "common/common.h"

namespace fs = std::filesystem;

// The request document, from either body shape.
//
// JSON body: the whole body IS the document, which is every caller before the media-transport
// work. Multipart: a `request` part holds the document and the binary parts alongside it carry the
// media that used to be base64 inside it (`docs/media-transport.md` §4). `/ltx/v1/generate` and
// `/wan/v1/generate` were already multipart; this is what brings `/sdcpp/v1/*` up to them.
static bool extract_sdcpp_request(const httplib::Request& req, json& out) {
    if (!req.is_multipart_form_data()) {
        if (req.body.empty()) {
            return false;
        }
        out = json::parse(req.body);
        return true;
    }
    std::string document;
    if (req.form.has_field("request")) {
        document = req.form.get_field("request");
    } else if (req.form.has_file("request")) {
        document = req.form.get_file("request").content;
    } else {
        return false;
    }
    if (document.empty()) {
        return false;
    }
    out = json::parse(document);
    return true;
}

static bool parse_cache_mode(const std::string& mode_str, sd_cache_mode_t& mode_out) {
    if (mode_str == "disabled") {
        mode_out = SD_CACHE_DISABLED;
        return true;
    }
    if (mode_str == "easycache") {
        mode_out = SD_CACHE_EASYCACHE;
        return true;
    }
    if (mode_str == "ucache") {
        mode_out = SD_CACHE_UCACHE;
        return true;
    }
    if (mode_str == "dbcache") {
        mode_out = SD_CACHE_DBCACHE;
        return true;
    }
    if (mode_str == "taylorseer") {
        mode_out = SD_CACHE_TAYLORSEER;
        return true;
    }
    if (mode_str == "cache-dit") {
        mode_out = SD_CACHE_CACHE_DIT;
        return true;
    }
    if (mode_str == "spectrum") {
        mode_out = SD_CACHE_SPECTRUM;
        return true;
    }
    return false;
}

static json finite_number_or_null(float value) {
    return std::isfinite(value) ? json(value) : json(nullptr);
}

static const char* capability_scheduler_name(enum scheduler_t scheduler) {
    return scheduler < SCHEDULER_COUNT ? sd_scheduler_name(scheduler) : "default";
}

static const char* capability_sample_method_name(enum sample_method_t sample_method) {
    return sample_method < SAMPLE_METHOD_COUNT ? sd_sample_method_name(sample_method) : "default";
}

static json make_vae_tiling_json(const sd_tiling_params_t& params) {
    return {
        {"enabled", params.enabled},
        {"temporal_tiling", params.temporal_tiling},
        {"tile_size_x", params.tile_size_x},
        {"tile_size_y", params.tile_size_y},
        {"target_overlap", params.target_overlap},
        {"rel_size_x", params.rel_size_x},
        {"rel_size_y", params.rel_size_y},
        {"extra_tiling_args", params.extra_tiling_args ? params.extra_tiling_args : ""},
    };
}

static fs::path resolve_display_model_path(const ServerRuntime& runtime) {
    const auto& ctx = *runtime.ctx_params;
    if (!ctx.model_path.empty()) {
        return fs::path(ctx.model_path);
    }
    if (!ctx.diffusion_model_path.empty()) {
        return fs::path(ctx.diffusion_model_path);
    }
    return {};
}

static json make_sample_params_json(const sd_sample_params_t& sample_params, const std::vector<int>& skip_layers) {
    const auto& guidance = sample_params.guidance;
    return {
        {"scheduler", capability_scheduler_name(sample_params.scheduler)},
        {"sample_method", capability_sample_method_name(sample_params.sample_method)},
        {"sample_steps", sample_params.sample_steps},
        {"eta", finite_number_or_null(sample_params.eta)},
        {"shifted_timestep", sample_params.shifted_timestep},
        {"flow_shift", finite_number_or_null(sample_params.flow_shift)},
        {"guidance",
         {
             {"txt_cfg", guidance.txt_cfg},
             {"img_cfg", finite_number_or_null(guidance.img_cfg)},
             {"distilled_guidance", guidance.distilled_guidance},
             {"slg",
              {
                  {"layers", skip_layers},
                  {"layer_start", guidance.slg.layer_start},
                  {"layer_end", guidance.slg.layer_end},
                  {"scale", guidance.slg.scale},
              }},
         }},
    };
}

static json make_hires_json(const SDGenerationParams& defaults) {
    return {
        {"enabled", defaults.hires_enabled},
        {"upscaler", defaults.hires_upscaler},
        {"scale", defaults.hires_scale},
        {"target_width", defaults.hires_width},
        {"target_height", defaults.hires_height},
        {"steps", defaults.hires_steps},
        {"denoising_strength", defaults.hires_denoising_strength},
        {"custom_sigmas", defaults.hires_custom_sigmas},
        {"upscale_tile_size", defaults.hires_upscale_tile_size},
    };
}

static json make_img_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format) {
    return {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width > 0 ? defaults.width : 512},
        {"height", defaults.height > 0 ? defaults.height : 512},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"batch_count", defaults.batch_count},
        {"qwen_image_layers", defaults.qwen_image_layers},
        {"auto_resize_ref_image", defaults.auto_resize_ref_image},
        {"increase_ref_index", defaults.increase_ref_index},
        {"ref_image_args", defaults.ref_image_args},
        {"control_strength", defaults.control_strength},
        {"ip_adapter_strength", defaults.ip_adapter_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers)},
        {"hires", make_hires_json(defaults)},
        {"vae_tiling_params", make_vae_tiling_json(defaults.vae_tiling_params)},
        {"cache_mode", defaults.cache_mode},
        {"cache_option", defaults.cache_option},
        {"scm_mask", defaults.scm_mask},
        {"scm_policy_dynamic", defaults.scm_policy_dynamic},
        {"output_format", output_format},
        {"output_compression", 100},
    };
}

static json make_vid_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format) {
    return {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width > 0 ? defaults.width : 512},
        {"height", defaults.height > 0 ? defaults.height : 512},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"video_frames", defaults.video_frames},
        {"fps", defaults.fps},
        {"moe_boundary", defaults.moe_boundary},
        {"vace_strength", defaults.vace_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers)},
        {"high_noise_sample_params", make_sample_params_json(defaults.high_noise_sample_params, defaults.high_noise_skip_layers)},
        {"hires", make_hires_json(defaults)},
        {"vae_tiling_params", make_vae_tiling_json(defaults.vae_tiling_params)},
        {"cache_mode", defaults.cache_mode},
        {"cache_option", defaults.cache_option},
        {"scm_mask", defaults.scm_mask},
        {"scm_policy_dynamic", defaults.scm_policy_dynamic},
        {"output_format", output_format},
        {"output_compression", 100},
    };
}

static json make_img_gen_features_json() {
    return {
        {"init_image", true},
        {"mask_image", true},
        {"control_image", true},
        {"ip_adapter_image", true},
        {"ref_images", true},
        {"ref_image_args", true},
        {"lora", true},
        {"vae_tiling", true},
        {"hires", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", false},
    };
}

static json make_vid_gen_features_json() {
    return {
        {"init_image", true},
        {"end_image", true},
        {"control_frames", true},
        {"high_noise_sample_params", true},
        {"lora", true},
        {"vae_tiling", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", false},
    };
}

static json make_capabilities_json(ServerRuntime& runtime) {
    refresh_lora_cache(runtime);
    refresh_upscaler_cache(runtime);

    AsyncJobManager& manager  = *runtime.async_job_manager;
    const auto& defaults      = *runtime.default_gen_params;
    const fs::path model_path = resolve_display_model_path(runtime);
    const bool supports_img   = runtime_supports_generation_mode(runtime, IMG_GEN);
    const bool supports_vid   = runtime_supports_generation_mode(runtime, VID_GEN);
    json samplers             = json::array();
    json schedulers           = json::array();
    json image_output_formats = supported_img_output_formats();
    json video_output_formats = supported_vid_output_formats();
    json available_loras      = json::array();
    json available_upscalers  = json::array();
    json supported_modes      = json::array();

    for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i) {
        samplers.push_back(sd_sample_method_name((sample_method_t)i));
    }

    for (int i = 0; i < SCHEDULER_COUNT; ++i) {
        schedulers.push_back(sd_scheduler_name((scheduler_t)i));
        if (i == DISCRETE_SCHEDULER) {
            schedulers.push_back("normal");
        }
    }

    {
        std::lock_guard<std::mutex> lock(*runtime.lora_mutex);
        for (const auto& entry : *runtime.lora_cache) {
            available_loras.push_back({
                {"name", entry.name},
                {"path", entry.path},
            });
        }
    }

    available_upscalers.push_back({
        {"name", "None"},
    });
    available_upscalers.push_back({
        {"name", "Lanczos"},
    });
    available_upscalers.push_back({
        {"name", "Nearest"},
    });
    available_upscalers.push_back({
        {"name", "Latent"},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest)"},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest-exact)"},
    });
    available_upscalers.push_back({
        {"name", "Latent (antialiased)"},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic)"},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic antialiased)"},
    });
    {
        std::lock_guard<std::mutex> lock(*runtime.upscaler_mutex);
        for (const auto& entry : *runtime.upscaler_cache) {
            available_upscalers.push_back({
                {"name", entry.name},
            });
        }
    }

    if (supports_img) {
        supported_modes.push_back("img_gen");
    }
    if (supports_vid) {
        supported_modes.push_back("vid_gen");
    }

    std::string default_img_output_format = "png";
    std::string default_vid_output_format = "avi";
    if (!image_output_formats.empty()) {
        default_img_output_format = image_output_formats[0].get<std::string>();
    }
    if (!video_output_formats.empty()) {
        default_vid_output_format = video_output_formats[0].get<std::string>();
    }

    json defaults_by_mode       = json::object();
    json output_formats_by_mode = json::object();
    json features_by_mode       = json::object();
    if (supports_img) {
        defaults_by_mode["img_gen"]       = make_img_gen_defaults_json(defaults, default_img_output_format);
        output_formats_by_mode["img_gen"] = image_output_formats;
        features_by_mode["img_gen"]       = make_img_gen_features_json();
    }
    if (supports_vid) {
        defaults_by_mode["vid_gen"]       = make_vid_gen_defaults_json(defaults, default_vid_output_format);
        output_formats_by_mode["vid_gen"] = video_output_formats;
        features_by_mode["vid_gen"]       = make_vid_gen_features_json();
    }

    json top_level_defaults       = json::object();
    json top_level_output_formats = json::array();
    json top_level_features       = {
              {"cancel_queued", true},
              {"cancel_generating", false},
    };
    std::string current_mode = "";
    if (supports_img) {
        current_mode             = "img_gen";
        top_level_defaults       = defaults_by_mode["img_gen"];
        top_level_output_formats = output_formats_by_mode["img_gen"];
        top_level_features       = features_by_mode["img_gen"];
    } else if (supports_vid) {
        current_mode             = "vid_gen";
        top_level_defaults       = defaults_by_mode["vid_gen"];
        top_level_output_formats = output_formats_by_mode["vid_gen"];
        top_level_features       = features_by_mode["vid_gen"];
    }

    json result;
    result["model"] = {
        {"name", model_path.filename().u8string()},
        {"stem", model_path.stem().u8string()},
        {"path", model_path.u8string()},
    };
    result["current_mode"]     = current_mode;
    result["supported_modes"]  = supported_modes;
    result["defaults"]         = top_level_defaults;
    result["defaults_by_mode"] = defaults_by_mode;
    result["limits"]           = {
                  {"min_width", 64},
                  {"max_width", 4096},
                  {"min_height", 64},
                  {"max_height", 4096},
                  {"max_batch_count", 8},
                  {"max_queue_size", manager.max_pending_jobs},
    };
    result["samplers"]               = samplers;
    result["schedulers"]             = schedulers;
    result["output_formats"]         = top_level_output_formats;
    result["output_formats_by_mode"] = output_formats_by_mode;
    result["features"]               = top_level_features;
    result["features_by_mode"]       = features_by_mode;
    result["loras"]                  = available_loras;
    result["upscalers"]              = available_upscalers;
    return result;
}

static bool parse_img_gen_request(const json& body,
                                  ServerRuntime& runtime,
                                  ImgGenJobRequest& request,
                                  std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;

    refresh_lora_cache(runtime);
    if (!request.gen_params.from_json_str(body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }

    std::string output_format = body.value("output_format", "png");
    int output_compression    = body.value("output_compression", 100);
    if (!assign_output_options(request, output_format, output_compression, true, error_message)) {
        return false;
    }

    // FLUX.2-Klein dual-DiT: optional top-level "model":"base"|"edit" selects
    // which DiT renders this request (koblem sends it on every img_gen call).
    // Validate against the runtime variant map so an unknown name is rejected up
    // front; "edit" on a server started without --diffusion-model-edit is absent
    // from that map. When absent/empty, model_variant stays empty and the worker
    // renders with whatever DiT is resident (single-model behaviour byte-
    // identical). Mirrors prod routes_sdcpp.cpp:392-406.
    if (body.contains("model") && body["model"].is_string()) {
        const std::string variant = body["model"].get<std::string>();
        if (!variant.empty()) {
            // Consult the runtime variant map (which reflects the worker-isolation
            // SD_SERVER_WORKER_VARIANT_MAP), NOT svr_params->diffusion_model_edit_path:
            // under isolation the request-handling child has --diffusion-model-edit
            // stripped from its argv (worker_supervisor.cpp), so that field is always
            // empty even though the edit DiT is loaded and selectable via the map.
            const auto variants = runtime_diffusion_model_variants(runtime);
            if (variant == "edit" && variants.find("edit") == variants.end()) {
                error_message = "edit model not available (server started without --diffusion-model-edit)";
                return false;
            }
            if (variants.find(variant) == variants.end()) {
                error_message = "unknown model variant '" + variant + "'";
                return false;
            }
            request.model_variant = variant;
        }
    }

    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
}

static bool parse_vid_gen_request(const json& body,
                                  ServerRuntime& runtime,
                                  VidGenJobRequest& request,
                                  std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;

    refresh_lora_cache(runtime);
    if (!request.gen_params.from_json_str(body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }

    std::string output_format = body.value("output_format", "webm");
    int output_compression    = body.value("output_compression", 100);
    if (!assign_output_options(request, output_format, output_compression, error_message)) {
        return false;
    }
    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(VID_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
}

void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    svr.Get("/sdcpp/v1/capabilities", [runtime](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(make_capabilities_json(*runtime).dump(), "application/json");
    });

    svr.Post("/sdcpp/v1/img_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty() && !req.is_multipart_form_data()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, IMG_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(IMG_GEN)}}).dump(), "application/json");
                return;
            }

            // `part:<name>` media (`docs/media-transport.md` §4). Registered for this thread
            // BEFORE anything parses, because the top-level images are decoded during the parse,
            // and moved onto the job below, because the per-shot ones are decoded on the worker
            // thread minutes later. The move is the last use — nothing decodes between it and the
            // end of this scope, where the registration lapses.
            MediaPartTable media_parts = collect_media_parts(req);
            if (const std::string bad = first_media_part_hash_mismatch(media_parts); !bad.empty()) {
                // §9.3: a hash-named part whose bytes do not hash to its name was truncated or
                // corrupted in transit. One hash to refuse it; accepting it renders something
                // subtly wrong and returns 200.
                res.status = 400;
                res.set_content(json({{"error", "part " + bad + " does not match its content hash"}}).dump(),
                                "application/json");
                return;
            }
            ScopedMediaParts parts_guard(&media_parts);
            json body;
            if (!extract_sdcpp_request(req, body)) {
                res.status = 400;
                res.set_content(R"({"error":"missing or invalid request"})", "application/json");
                return;
            }
            ImgGenJobRequest request;
            std::string error_message;
            if (!parse_img_gen_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            AsyncJobManager& manager                = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job = std::make_shared<AsyncGenerationJob>();
            job->kind                               = AsyncJobKind::ImgGen;
            job->status                             = AsyncJobStatus::Queued;
            job->created_at                         = unix_timestamp_now();
            job->media_parts = std::move(media_parts);
            job->img_gen                            = std::move(request);

            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id               = make_async_job_id(manager);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }

            manager.cv.notify_one();

            json out;
            out["id"]       = job->id;
            out["kind"]     = async_job_kind_name(job->kind);
            out["status"]   = async_job_status_name(job->status);
            out["created"]  = job->created_at;
            out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;

            res.status = 202;
            res.set_content(out.dump(), "application/json");
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/sdcpp/v1/vid_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty() && !req.is_multipart_form_data()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, VID_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(VID_GEN)}}).dump(), "application/json");
                return;
            }

            // `part:<name>` media (`docs/media-transport.md` §4). Registered for this thread
            // BEFORE anything parses, because the top-level images are decoded during the parse,
            // and moved onto the job below, because the per-shot ones are decoded on the worker
            // thread minutes later. The move is the last use — nothing decodes between it and the
            // end of this scope, where the registration lapses.
            MediaPartTable media_parts = collect_media_parts(req);
            if (const std::string bad = first_media_part_hash_mismatch(media_parts); !bad.empty()) {
                // §9.3: a hash-named part whose bytes do not hash to its name was truncated or
                // corrupted in transit. One hash to refuse it; accepting it renders something
                // subtly wrong and returns 200.
                res.status = 400;
                res.set_content(json({{"error", "part " + bad + " does not match its content hash"}}).dump(),
                                "application/json");
                return;
            }
            ScopedMediaParts parts_guard(&media_parts);
            json body;
            if (!extract_sdcpp_request(req, body)) {
                res.status = 400;
                res.set_content(R"({"error":"missing or invalid request"})", "application/json");
                return;
            }
            VidGenJobRequest request;
            std::string error_message;
            if (!parse_vid_gen_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            AsyncJobManager& manager                = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job = std::make_shared<AsyncGenerationJob>();
            job->kind                               = AsyncJobKind::VidGen;
            job->status                             = AsyncJobStatus::Queued;
            job->created_at                         = unix_timestamp_now();
            job->media_parts = std::move(media_parts);
            job->vid_gen                            = std::move(request);

            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id               = make_async_job_id(manager);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }

            manager.cv.notify_one();

            json out;
            out["id"]       = job->id;
            out["kind"]     = async_job_kind_name(job->kind);
            out["status"]   = async_job_status_name(job->status);
            out["created"]  = job->created_at;
            out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;

            res.status = 202;
            res.set_content(out.dump(), "application/json");
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", e.what()}}).dump(), "application/json");
        }
    });

    svr.Get(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+))", [runtime](const httplib::Request& req, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        purge_expired_jobs(manager);

        std::string job_id = req.matches[1];
        auto it            = manager.jobs.find(job_id);
        if (it == manager.jobs.end()) {
            if (manager.expired_jobs.find(job_id) != manager.expired_jobs.end()) {
                res.status = 410;
                res.set_content(R"({"error":"job expired"})", "application/json");
            } else {
                res.status = 404;
                res.set_content(R"({"error":"job not found"})", "application/json");
            }
            return;
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, *it->second).dump(), "application/json");
    });

    svr.Get(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+)/media)", [runtime](const httplib::Request& req, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        // Take a REFERENCE to the job under the lock and let go of it before writing the body
        // (`docs/media-transport.md` §4.1). With `SDCPP_JOB_MEDIA_B64=0` this becomes the normal
        // way every render is collected, and `manager.mutex` also serialises every status poll on
        // the service — holding it across a 70 MB copy would stall the poll loop of any other
        // in-flight job. The shared_ptr keeps the record alive even if the sweeper drops it.
        std::shared_ptr<AsyncGenerationJob> job;
        {
            std::lock_guard<std::mutex> lock(manager.mutex);
            purge_expired_jobs(manager);
            const auto it = manager.jobs.find(req.matches[1]);
            if (it == manager.jobs.end()) {
                res.status = manager.expired_jobs.count(req.matches[1]) ? 410 : 404;
                res.set_content(R"({"error":"job not found or expired"})", "application/json");
                return;
            }
            job = it->second;
        }
        if (job->status != AsyncJobStatus::Completed) {
            res.status = 409;
            res.set_content(R"({"error":"job is not complete"})", "application/json");
            return;
        }
        // B5: the same route serves an IMAGE job's output n, so an image result is fetched raw
        // instead of read out of `result.images[].b64_json`. `index` defaults to 0, which is the
        // only image on every single-image render.
        if (job->kind == AsyncJobKind::ImgGen) {
            size_t index = 0;
            if (req.has_param("index")) {
                const std::string raw = req.get_param_value("index");
                try {
                    const long long parsed = std::stoll(raw);
                    if (parsed < 0) throw std::out_of_range("negative");
                    index = static_cast<size_t>(parsed);
                } catch (const std::exception&) {
                    res.status = 400;
                    res.set_content(R"({"error":"index must be a non-negative integer"})", "application/json");
                    return;
                }
            }
            if (index >= job->result_images.size()) {
                res.status = 404;
                res.set_content(json({{"error", "no image at index " + std::to_string(index)},
                                      {"images", job->result_images.size()}})
                                    .dump(),
                                "application/json");
                return;
            }
            res.status = 200;
            res.set_content(reinterpret_cast<const char*>(job->result_images[index].data()),
                            job->result_images[index].size(),
                            image_mime_type(job->img_gen.output_format));
            return;
        }
        if (job->kind != AsyncJobKind::VidGen) {
            res.status = 409;
            res.set_content(R"({"error":"job has no media"})", "application/json");
            return;
        }
        if (job->result_media.empty()) {
            res.status = 500;
            res.set_content(R"({"error":"stored video result is invalid"})", "application/json");
            return;
        }
        // Served straight out of the job record: no decode, and no second buffer.
        res.status = 200;
        res.set_content(reinterpret_cast<const char*>(job->result_media.data()),
                        job->result_media.size(),
                        job->result_media_mime_type);
    });

    svr.Post(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+)/cancel)", [runtime](const httplib::Request& req, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        purge_expired_jobs(manager);

        std::string job_id = req.matches[1];
        auto it            = manager.jobs.find(job_id);
        if (it == manager.jobs.end()) {
            if (manager.expired_jobs.find(job_id) != manager.expired_jobs.end()) {
                res.status = 410;
                res.set_content(R"({"error":"job expired"})", "application/json");
            } else {
                res.status = 404;
                res.set_content(R"({"error":"job not found"})", "application/json");
            }
            return;
        }

        auto& job = *it->second;
        if (job.status == AsyncJobStatus::Queued) {
            if (!cancel_queued_job(manager, job)) {
                res.status = 409;
                res.set_content(R"({"error":"job queue state changed before cancellation"})", "application/json");
                return;
            }
            res.status = 200;
            res.set_content(make_async_job_json(manager, job).dump(), "application/json");
            return;
        }

        if (job.status == AsyncJobStatus::Generating) {
            job.cancel_requested = true;
            sd_cancel_generation(runtime->sd_ctx, SD_CANCEL_ALL);
            res.status = 202;
            res.set_content(json({{"id", job.id}, {"status", "cancelling"}}).dump(), "application/json");
            return;
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, job).dump(), "application/json");
    });
}
