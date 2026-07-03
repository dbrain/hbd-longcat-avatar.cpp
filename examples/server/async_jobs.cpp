// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "worker_session.h"

namespace {

// Background writer that banks a viewable per-segment webm (save_dir/seg_<i>.webm) as the
// chain produces each segment, OFF the GPU sampling thread (encoding inline would stall
// sampling — costly under swap). Strictly bounded to one segment's frames in flight via a
// busy flag: the chain's on_segment callback blocks until the prior encode drains, so the
// extra RAM is ~one segment of decoded frames, never an unbounded pile. The banked webm is
// a partial-preview artifact; resume itself reloads the seg_<i>.bin latents, not the webm.
struct SegWebmWriter {
    std::string dir;
    int         fps     = 24;
    int         quality = 90;

    std::mutex              m;
    std::condition_variable cv;
    struct Task {
        int                     seg = 0;
        std::vector<sd_image_t> frames;  // owned deep copies, freed after encode
    };
    std::deque<Task> q;
    bool             busy = false;
    bool             done = false;
    std::thread      th;

    void start() {
        th = std::thread([this] {
            for (;;) {
                Task t;
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return done || !q.empty(); });
                    if (q.empty()) {
                        if (done) {
                            break;
                        }
                        continue;
                    }
                    t    = std::move(q.front());
                    q.pop_front();
                    busy = true;
                }
                std::string path  = dir + "/seg_" + std::to_string(t.seg) + ".webm";
                auto        bytes = create_video_from_sd_images_to_vector(
                    "webm", t.frames.data(), (int)t.frames.size(), fps, quality, nullptr);
                if (!bytes.empty()) {
                    std::ofstream out(path, std::ios::binary | std::ios::trunc);
                    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
                }
                for (auto& f : t.frames) {
                    free(f.data);
                }
                {
                    std::lock_guard<std::mutex> lk(m);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }

    void enqueue(int seg, const sd_image_t* fr, int n) {
        {  // back-pressure: wait until the prior segment has drained
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return done || (q.empty() && !busy); });
            if (done) {
                return;
            }
        }
        std::vector<sd_image_t> copies;
        copies.reserve(n);
        for (int i = 0; i < n; ++i) {
            sd_image_t c = fr[i];
            size_t     sz = (size_t)c.width * c.height * c.channel;
            c.data        = (uint8_t*)malloc(sz);
            if (c.data != nullptr && fr[i].data != nullptr) {
                memcpy(c.data, fr[i].data, sz);
            }
            copies.push_back(c);
        }
        {
            std::lock_guard<std::mutex> lk(m);
            q.push_back(Task{seg, std::move(copies)});
        }
        cv.notify_all();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lk(m);
            done = true;
        }
        cv.notify_all();
        if (th.joinable()) {
            th.join();
        }
    }
};

void ltx_seg_webm_cb(int seg, const sd_image_t* frames, int n, void* user) {
    static_cast<SegWebmWriter*>(user)->enqueue(seg, frames, n);
}

}  // namespace

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
    // Artifact dir (when job persistence is on). Its presence flags a resumable VidGen job:
    // a failed/cancelled chain can be continued via resume_job_id from the banked segments.
    result["job_dir"]        = job.job_dir.empty() ? json(nullptr) : json(job.job_dir);
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
    // A2-safe per-render bridge: honour this request's a2v / ramp / ref-tstride, overwriting any
    // stale value from a prior warm-worker render (the whole chain shares one a2v via process env).
    gen_params.apply_ltx_relip_env();

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

    // Durable artifact dir (per-segment latent/webm banking + final.webm) and the resume
    // point (skip segments already banked there). Both injected by the LTX /generate route.
    std::string job_dir     = body.value("ltx_job_dir", std::string());
    int         resume_from = body.value("resume_from", 0);

    sd_vid_gen_params_t base = gen_params.to_sd_vid_gen_params_t();

    sd_vid_chain_params_t chain = {};
    chain.n_segments         = n_segments;
    chain.cont_latent_frames = std::max(1, gen_params.cont_latent_take);
    chain.segment_prompts    = prompt_ptrs.data();
    chain.chain_audio_dir    = audio_dir.empty() ? nullptr : audio_dir.c_str();
    chain.save_dir           = job_dir.empty() ? nullptr : job_dir.c_str();
    chain.resume_from        = std::max(0, resume_from);

    // Bank a viewable per-segment webm as each segment lands (off the GPU thread). On by
    // default when a job dir is set; LTX_BANK_SEG_WEBM=0 disables it (e.g. under tight RAM —
    // resume only needs the seg_<i>.bin latents, which are banked regardless).
    std::unique_ptr<SegWebmWriter> seg_writer;
    if (!job_dir.empty()) {
        const char* bank = getenv("LTX_BANK_SEG_WEBM");
        bool        on   = (bank == nullptr) || (bank[0] != '0');
        if (on) {
            seg_writer          = std::make_unique<SegWebmWriter>();
            seg_writer->dir     = job_dir;
            seg_writer->fps     = gen_params.fps;
            seg_writer->quality = output_compression;
            seg_writer->start();
            chain.on_segment      = &ltx_seg_webm_cb;
            chain.on_segment_user = seg_writer.get();
        }
    }

    sd_image_t* frames      = nullptr;
    int         frame_count = 0;
    sd_audio_t* audio       = nullptr;
    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        bool chain_ok = generate_video_chain(runtime.sd_ctx, &base, &chain, &frames, &frame_count, &audio);
        if (seg_writer) {
            seg_writer->finish();  // drain pending seg webms before reporting outcome
        }
        if (!chain_ok) {
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

    // Persist the finished container so it survives the in-RAM result TTL and a client
    // disconnect (re-fetchable via GET /sdcpp/v1/jobs/{id}/media).
    if (!job_dir.empty()) {
        std::string final_path = job_dir + "/final." + output_format;
        std::ofstream out(final_path, std::ios::binary | std::ios::trunc);
        if (out) {
            out.write(reinterpret_cast<const char*>(out_video.data()),
                      static_cast<std::streamsize>(out_video.size()));
        }
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
