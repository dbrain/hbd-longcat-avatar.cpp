// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <iomanip>
#include <sstream>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "worker_session.h"

const char* async_job_kind_name(AsyncJobKind kind) {
    switch (kind) {
        case AsyncJobKind::ImgGen:
            return "img_gen";
        case AsyncJobKind::VidGen:
            return "vid_gen";
        default:
            return "img_gen";
    }
}

const char* async_job_status_name(AsyncJobStatus status) {
    switch (status) {
        case AsyncJobStatus::Queued:
            return "queued";
        case AsyncJobStatus::Generating:
            return "generating";
        case AsyncJobStatus::Completed:
            return "completed";
        case AsyncJobStatus::Failed:
            return "failed";
        case AsyncJobStatus::Cancelled:
            return "cancelled";
        default:
            return "failed";
    }
}

void purge_expired_jobs(AsyncJobManager& manager) {
    const int64_t now = unix_timestamp_now();

    for (auto it = manager.expired_jobs.begin(); it != manager.expired_jobs.end();) {
        if (it->second <= now) {
            it = manager.expired_jobs.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = manager.jobs.begin(); it != manager.jobs.end();) {
        const auto& job = it->second;
        if (job->completed_at == 0) {
            ++it;
            continue;
        }

        int64_t ttl_seconds = job->status == AsyncJobStatus::Completed
                                  ? manager.completed_ttl_seconds
                                  : manager.failed_ttl_seconds;
        if (now - job->completed_at >= ttl_seconds) {
            manager.expired_jobs[job->id] = now + std::max<int64_t>(ttl_seconds, 60);
            it                            = manager.jobs.erase(it);
        } else {
            ++it;
        }
    }
}

size_t count_pending_jobs(const AsyncJobManager& manager) {
    size_t pending = 0;
    for (const auto& entry : manager.jobs) {
        if (entry.second->status == AsyncJobStatus::Queued ||
            entry.second->status == AsyncJobStatus::Generating) {
            ++pending;
        }
    }
    return pending;
}

std::string make_async_job_id(AsyncJobManager& manager) {
    std::ostringstream oss;
    oss << "job_" << std::hex << unix_timestamp_now() << "_" << std::setw(8)
        << std::setfill('0') << manager.next_id++;
    return oss.str();
}

bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job) {
    auto new_end = std::remove(manager.queue.begin(), manager.queue.end(), job.id);
    if (new_end == manager.queue.end()) {
        return false;
    }

    manager.queue.erase(new_end, manager.queue.end());
    job.status       = AsyncJobStatus::Cancelled;
    job.completed_at = unix_timestamp_now();
    job.result_images_b64.clear();
    job.result_media_b64.clear();
    job.result_media_mime_type.clear();
    job.result_frame_count = 0;
    job.result_fps         = 0;
    job.error_code         = "cancelled";
    job.error_message      = "job cancelled by client";
    return true;
}

json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job) {
    json result;
    result["id"]             = job.id;
    result["kind"]           = async_job_kind_name(job.kind);
    result["status"]         = async_job_status_name(job.status);
    result["created"]        = job.created_at;
    result["started"]        = job.started_at == 0 ? json(nullptr) : json(job.started_at);
    result["completed"]      = job.completed_at == 0 ? json(nullptr) : json(job.completed_at);
    result["queue_position"] = 0;

    if (job.status == AsyncJobStatus::Queued) {
        size_t position = 1;
        for (const auto& queued_id : manager.queue) {
            if (queued_id == job.id) {
                result["queue_position"] = position;
                break;
            }
            ++position;
        }
    }

    if (job.status == AsyncJobStatus::Completed) {
        if (job.kind == AsyncJobKind::VidGen) {
            result["result"] = {
                {"output_format", job.vid_gen.output_format},
                {"mime_type", job.result_media_mime_type},
                {"fps", job.result_fps},
                {"frame_count", job.result_frame_count},
                {"b64_json", job.result_media_b64},
            };
        } else {
            json images = json::array();
            for (size_t i = 0; i < job.result_images_b64.size(); ++i) {
                images.push_back({{"index", i}, {"b64_json", job.result_images_b64[i]}});
            }
            result["result"] = {
                {"output_format", job.img_gen.output_format},
                {"images", images},
            };
        }
        result["error"] = nullptr;
    } else if (job.status == AsyncJobStatus::Failed ||
               job.status == AsyncJobStatus::Cancelled) {
        result["result"] = nullptr;
        result["error"]  = {
             {"code",
             job.error_code.empty()
                  ? (job.status == AsyncJobStatus::Cancelled ? "cancelled" : "generation_failed")
                  : job.error_code},
             {"message", job.error_message},
        };
    } else {
        result["result"] = nullptr;
        result["error"]  = nullptr;
    }

    return result;
}

bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         std::string& error_message) {
    sd_img_gen_params_t params = job.img_gen.to_sd_img_gen_params_t();

    SDImageVec results;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = generate_image(runtime.sd_ctx, &params);
        results.adopt(raw_results, params.batch_count);
    }

    const int num_results = results.count();
    if (num_results <= 0) {
        error_message = "generate_image returned no results";
        return false;
    }

    EncodedImageFormat encoded_format = EncodedImageFormat::PNG;
    if (job.img_gen.output_format == "jpeg") {
        encoded_format = EncodedImageFormat::JPEG;
    } else if (job.img_gen.output_format == "webp") {
        encoded_format = EncodedImageFormat::WEBP;
    }

    for (int i = 0; i < num_results; ++i) {
        if (results[i].data == nullptr) {
            continue;
        }

        const std::string metadata = job.img_gen.gen_params.embed_image_metadata
                                         ? get_image_params(*runtime.ctx_params,
                                                            job.img_gen.gen_params,
                                                            job.img_gen.gen_params.seed + i)
                                         : "";
        auto image_bytes           = encode_image_to_vector(encoded_format,
                                                            results[i].data,
                                                            results[i].width,
                                                            results[i].height,
                                                            results[i].channel,
                                                            metadata,
                                                            job.img_gen.output_compression);
        if (image_bytes.empty()) {
            continue;
        }
        output_images.push_back(base64_encode(image_bytes));
    }

    if (output_images.empty()) {
        error_message = "generate_image returned empty encoded outputs";
        return false;
    }

    return true;
}

bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         std::string& error_message) {
    sd_vid_gen_params_t params = job.vid_gen.to_sd_vid_gen_params_t();

    SDImageVec results;
    int num_results             = 0;
    sd_audio_t* generated_audio = nullptr;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = nullptr;
        if (!generate_video(runtime.sd_ctx, &params, &raw_results, &num_results, &generated_audio)) {
            raw_results = nullptr;
        }
        results.adopt(raw_results, num_results);
    }

    num_results = results.count();
    if (num_results <= 0) {
        free_sd_audio(generated_audio);
        error_message = "generate_video returned no results";
        return false;
    }

    std::vector<uint8_t> video_bytes = create_video_from_sd_images_to_vector(job.vid_gen.output_format,
                                                                             results.data(),
                                                                             num_results,
                                                                             job.vid_gen.gen_params.fps,
                                                                             job.vid_gen.output_compression,
                                                                             generated_audio);
    free_sd_audio(generated_audio);
    if (video_bytes.empty()) {
        error_message = "failed to encode generated video container";
        return false;
    }

    output_media_b64       = base64_encode(video_bytes);
    output_media_mime_type = video_mime_type(job.vid_gen.output_format);
    output_frame_count     = num_results;
    output_fps             = job.vid_gen.gen_params.fps;
    return true;
}

bool run_vid_chain_job(ServerRuntime& runtime,
                       const std::string& chain_request_json,
                       std::vector<uint8_t>& out_video,
                       std::string& out_mime,
                       int& out_frame_count,
                       int& out_fps,
                       std::string& error_message) {
    json body;
    try {
        body = json::parse(chain_request_json);
    } catch (const std::exception& e) {
        error_message = std::string("chain request JSON parse: ") + e.what();
        return false;
    }

    // Base per-segment generation params: defaults + request JSON (loads inline base64
    // init_image, W/H/fps/steps/cfg/sampler/scheduler/seed/negative, and the chain knobs
    // ltx_chain_segments + cont_latent_frames).
    SDGenerationParams gen_params = *runtime.default_gen_params;
    if (!gen_params.from_json_str(body.dump())) {
        error_message = "invalid generation parameters";
        return false;
    }
    if (!gen_params.resolve_and_validate(VID_GEN, runtime.ctx_params->lora_model_dir,
                                         runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters (resolve_and_validate)";
        return false;
    }

    int n_segments = std::max(1, gen_params.ltx_chain_segments);

    // Per-segment prompts (the director layer): a JSON array. Fewer than n_segments reuses
    // the last; none reuses the base prompt. Storage must outlive the chain call.
    std::vector<std::string> seg_prompts;
    if (body.contains("prompts") && body["prompts"].is_array()) {
        for (const auto& p : body["prompts"]) {
            if (p.is_string()) {
                seg_prompts.push_back(p.get<std::string>());
            }
        }
    }
    std::vector<std::string> resolved_prompts;
    resolved_prompts.reserve(n_segments);
    for (int seg = 0; seg < n_segments; ++seg) {
        if (!seg_prompts.empty()) {
            resolved_prompts.push_back(seg_prompts[std::min((size_t)seg, seg_prompts.size() - 1)]);
        } else {
            resolved_prompts.emplace_back();
        }
    }
    std::vector<const char*> prompt_ptrs;
    prompt_ptrs.reserve(n_segments);
    for (const auto& s : resolved_prompts) {
        prompt_ptrs.push_back(s.empty() ? nullptr : s.c_str());
    }

    // Per-segment lip-sync audio dir (aud_<i>.wav), written by the parent route handler.
    std::string audio_dir     = body.value("chain_audio_dir", std::string());
    std::string output_format = body.value("output_format", std::string("webm"));
    int         output_compression = body.value("output_compression", 90);

    sd_vid_gen_params_t base = gen_params.to_sd_vid_gen_params_t();

    sd_vid_chain_params_t chain = {};
    chain.n_segments         = n_segments;
    chain.cont_latent_frames = std::max(1, gen_params.cont_latent_take);
    chain.segment_prompts    = prompt_ptrs.data();
    chain.chain_audio_dir    = audio_dir.empty() ? nullptr : audio_dir.c_str();
    chain.save_dir           = nullptr;

    sd_image_t* frames      = nullptr;
    int         frame_count = 0;
    sd_audio_t* audio       = nullptr;
    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        if (!generate_video_chain(runtime.sd_ctx, &base, &chain, &frames, &frame_count, &audio)) {
            error_message = "generate_video_chain failed";
            return false;
        }
    }
    if (frame_count <= 0 || frames == nullptr) {
        free_sd_audio(audio);
        error_message = "generate_video_chain produced no frames";
        return false;
    }

    out_video = create_video_from_sd_images_to_vector(output_format, frames, frame_count,
                                                      gen_params.fps, output_compression, audio);
    for (int i = 0; i < frame_count; ++i) {
        free(frames[i].data);
    }
    free(frames);
    free_sd_audio(audio);

    if (out_video.empty()) {
        error_message = "failed to encode generated video container";
        return false;
    }
    out_mime        = video_mime_type(output_format);
    out_frame_count = frame_count;
    out_fps         = gen_params.fps;
    return true;
}

bool ensure_variant_loaded(ServerRuntime& runtime,
                           const std::string& target_variant,
                           std::string& error_message) {
    ModelSwapState* swap = runtime.model_swap;
    if (swap == nullptr) {
        return true;  // dual-DiT not wired (should not happen in-process)
    }

    // Resolve the variant we actually need resident. An explicit request wins;
    // otherwise keep the currently-loaded variant (no forced swap — preserves
    // single-model byte-identical behaviour when no "model" field is sent).
    std::string want = target_variant.empty() ? swap->loaded_variant : target_variant;
    if (want.empty()) {
        want = "base";
    }

    const bool unloaded     = !swap->loaded.load();
    const bool variant_diff = (want != swap->loaded_variant);

    // Fast path: requested variant already resident and nothing was unloaded.
    if (!unloaded && !variant_diff) {
        return true;
    }

    // Resolve the gguf path for the wanted variant.
    std::string path = (want == "edit") ? swap->edit_path : swap->base_path;
    if (path.empty()) {
        error_message = "no diffusion model path for variant '" + want + "'";
        return false;
    }

    // Swap is serialized with rendering via sd_ctx_mutex (worker is single-threaded,
    // but admin/unload may also touch the DiT under this lock).
    std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);

    if (variant_diff || unloaded) {
        LOG_INFO("img_gen: swapping DiT variant %s -> %s (unloaded=%d)",
                 swap->loaded_variant.c_str(), want.c_str(), (int)unloaded);
        if (!sd_ctx_swap_diffusion_model(runtime.sd_ctx, path.c_str())) {
            error_message = "failed to load diffusion model variant '" + want + "'";
            // On failure the DiT params buffer is freed → mark unloaded so the next
            // attempt forces a fresh reload rather than assuming it's resident.
            swap->loaded.store(false);
            return false;
        }
        swap->loaded_variant = want;
        swap->loaded.store(true);
    }
    return true;
}

void async_job_worker(ServerRuntime& runtime) {
    AsyncJobManager& manager = *runtime.async_job_manager;

    while (true) {
        std::shared_ptr<AsyncGenerationJob> job;
        {
            std::unique_lock<std::mutex> lock(manager.mutex);
            manager.cv.wait(lock, [&]() { return manager.stop || !manager.queue.empty(); });

            if (manager.stop && manager.queue.empty()) {
                break;
            }

            purge_expired_jobs(manager);
            if (manager.queue.empty()) {
                continue;
            }

            const std::string job_id = manager.queue.front();
            manager.queue.pop_front();

            auto it = manager.jobs.find(job_id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job             = it->second;
            job->status     = AsyncJobStatus::Generating;
            job->started_at = unix_timestamp_now();
            // Publish the active job + clear any stale cancel so a cancel aimed at
            // an already-finished job cannot abort this fresh render.
            manager.generating_job_id = job->id;
            sd_clear_cancel();
        }

        std::vector<std::string> output_images;
        std::string output_media_b64;
        std::string output_media_mime_type;
        int output_frame_count = 0;
        int output_fps         = 0;
        std::string error_message;
        bool ok = false;

        if (job->kind == AsyncJobKind::ImgGen) {
            if (runtime.worker) {
                // Worker isolation: render in the CUDA-owning child over IPC. The
                // child re-parses the raw request and runs the SAME
                // parse_img_gen_request + ensure_variant_loaded + execute_img_gen_job.
                longcat_avatar::ImageRenderResult r = runtime.worker->render_image(job->img_gen_request_json);
                ok            = r.ok;
                error_message = r.error;
                if (ok) {
                    output_images = std::move(r.images);
                    // Keep the parent's model_swap roughly in sync for
                    // /capabilities + /health (the child owns the real swap state).
                    if (runtime.model_swap) {
                        runtime.model_swap->loaded.store(true);
                        if (!job->img_gen.model_variant.empty()) {
                            runtime.model_swap->loaded_variant = job->img_gen.model_variant;
                        }
                    }
                }
            } else if (ensure_variant_loaded(runtime, job->img_gen.model_variant, error_message)) {
                // FLUX.2-Klein dual-DiT: ensure the requested variant (or the tracked
                // one, after an admin unload) is resident before rendering. The swap is
                // serial on this worker thread, so it never races a live render.
                ok = execute_img_gen_job(runtime, *job, output_images, error_message);
            }
        } else if (job->kind == AsyncJobKind::VidGen && runtime.worker) {
            if (runtime.ltx_video_mode && !job->vid_chain_request_json.empty()) {
                // LTXAV multi-segment chain in the CUDA-owning child. The child re-parses
                // the chain request and runs generate_video_chain via run_vid_chain_job.
                longcat_avatar::RenderResult r =
                    runtime.worker->render_video_chain(job->vid_chain_request_json);
                ok            = r.ok;
                error_message = r.error;
                if (ok) {
                    output_media_b64       = base64_encode(r.video_bytes);
                    output_media_mime_type = video_mime_type(job->vid_gen.output_format);
                    output_frame_count     = r.frame_count;
                    output_fps             = r.fps;
                }
            } else {
                // flux2 image-isolation parent has no sd_ctx and isn't an LTX video server.
                error_message = "vid_gen not supported under flux2 image isolation";
            }
        } else if (job->kind == AsyncJobKind::VidGen && !job->vid_chain_request_json.empty()) {
            // In-process LTXAV chain (no worker isolation).
            std::vector<uint8_t> video_bytes;
            ok = run_vid_chain_job(runtime, job->vid_chain_request_json, video_bytes,
                                   output_media_mime_type, output_frame_count, output_fps,
                                   error_message);
            if (ok) {
                output_media_b64 = base64_encode(video_bytes);
            }
        } else if (job->kind == AsyncJobKind::VidGen) {
            ok = execute_vid_gen_job(runtime,
                                     *job,
                                     output_media_b64,
                                     output_media_mime_type,
                                     output_frame_count,
                                     output_fps,
                                     error_message);
        } else {
            error_message = "unsupported job kind";
        }

        {
            std::lock_guard<std::mutex> lock(manager.mutex);
            auto it = manager.jobs.find(job->id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job->completed_at = unix_timestamp_now();
            manager.generating_job_id.clear();
            if (ok) {
                job->status                 = AsyncJobStatus::Completed;
                job->result_images_b64      = std::move(output_images);
                job->result_media_b64       = std::move(output_media_b64);
                job->result_media_mime_type = std::move(output_media_mime_type);
                job->result_frame_count     = output_frame_count;
                job->result_fps             = output_fps;
                job->error_code.clear();
                job->error_message.clear();
            } else if (sd_is_cancel_requested()) {
                // Cooperative cancel: the render bailed by request, not a fault.
                job->status        = AsyncJobStatus::Cancelled;
                job->error_code    = "cancelled";
                job->error_message = "job cancelled by client";
                job->result_images_b64.clear();
                job->result_media_b64.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps         = 0;
            } else {
                job->status        = AsyncJobStatus::Failed;
                job->error_code    = "generation_failed";
                job->error_message = error_message.empty() ? "unknown generation error" : error_message;
                job->result_images_b64.clear();
                job->result_media_b64.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps         = 0;
            }

            purge_expired_jobs(manager);
        }
    }
}
