// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include "common/log.h"
#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

namespace fs = std::filesystem;

namespace {

// A single bounded CPU encoder for progressive LTX segments.  The core owns
// and frees its decoded frames, so the callback copies one segment at a time;
// back-pressure prevents previews accumulating while a long chain samples.
struct SegmentPreviewWriter {
    std::string directory;
    int fps = 24;
    int quality = 90;
    std::mutex mutex;
    std::condition_variable cv;
    int segment = -1;
    std::vector<sd_image_t> frames;
    bool pending = false;
    bool busy = false;
    bool stop = false;
    std::thread thread;

    void start() {
        thread = std::thread([this] {
            for (;;) {
                int queued_segment = -1;
                std::vector<sd_image_t> queued_frames;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&] { return stop || pending; });
                    if (!pending && stop) break;
                    queued_segment = segment;
                    queued_frames = std::move(frames);
                    pending = false;
                    busy = true;
                }
                const auto bytes = create_video_from_sd_images_to_vector(
                    "webm", queued_frames.data(), static_cast<int>(queued_frames.size()), fps, quality, nullptr);
                if (!bytes.empty()) {
                    const fs::path final_path = fs::path(directory) /
                                                ("seg_" + std::to_string(queued_segment) + ".webm");
                    const fs::path temp_path = final_path.string() + ".tmp";
                    std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
                    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                    output.close();
                    if (output.good()) {
                        std::error_code error;
                        fs::rename(temp_path, final_path, error);
                    }
                }
                for (auto& frame : queued_frames) free(frame.data);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }

    void enqueue(int index, const sd_image_t* source, int count) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return stop || (!pending && !busy); });
        if (stop) return;
        std::vector<sd_image_t> copies;
        copies.reserve(static_cast<size_t>(count));
        for (int frame = 0; frame < count; ++frame) {
            sd_image_t copy = source[frame];
            const size_t size = static_cast<size_t>(copy.width) * copy.height * copy.channel;
            copy.data = static_cast<uint8_t*>(malloc(size));
            if (copy.data == nullptr || source[frame].data == nullptr) {
                for (auto& allocated : copies) free(allocated.data);
                return;
            }
            memcpy(copy.data, source[frame].data, size);
            copies.push_back(copy);
        }
        segment = index;
        frames = std::move(copies);
        pending = true;
        lock.unlock();
        cv.notify_all();
    }

    void finish() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return !pending && !busy; });
            stop = true;
        }
        cv.notify_all();
        if (thread.joinable()) thread.join();
    }
};

void write_segment_preview(int segment, const sd_image_t* frames, int count, void* user) {
    static_cast<SegmentPreviewWriter*>(user)->enqueue(segment, frames, count);
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

    // Koblem polls this optional list and fetches the URLs best-effort.  The
    // files are atomically published by SegmentPreviewWriter, so a listed
    // segment is always a complete, playable WebM rather than a partial write.
    if (job.ltx_emit_segments && !job.ltx_bank_dir.empty() &&
        (job.status == AsyncJobStatus::Generating || job.status == AsyncJobStatus::Completed)) {
        json partials = json::array();
        for (size_t segment = 0; segment < job.ltx_prompts.size(); ++segment) {
            std::error_code error;
            const fs::path path = fs::path(job.ltx_bank_dir) / ("seg_" + std::to_string(segment) + ".webm");
            if (fs::is_regular_file(path, error)) {
                partials.push_back({
                    {"segment_index", static_cast<int>(segment)},
                    {"stage", 4},
                    {"url", "/sdcpp/v1/jobs/" + job.id + "/segments/" + std::to_string(segment)},
                });
            }
        }
        if (!partials.empty()) result["partials"] = std::move(partials);
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
        sd_image_t* raw_results = nullptr;
        int num_results         = 0;
        const bool generated = generate_image(runtime.sd_ctx, &params, &raw_results, &num_results);
        if (!generated) {
            raw_results = nullptr;
            num_results = 0;
        } else if (runtime.gpu_sharing != nullptr) {
            runtime.gpu_sharing->diffusion_loaded.store(true);
        }
        results.adopt(raw_results, num_results);
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

    int batch_count      = job.img_gen.gen_params.batch_count;
    int images_per_batch = batch_count > 0 ? std::max(1, num_results / batch_count) : 1;
    for (int i = 0; i < num_results; ++i) {
        if (results[i].data == nullptr) {
            continue;
        }

        const std::string metadata = job.img_gen.gen_params.embed_image_metadata
                                         ? get_image_params(*runtime.ctx_params,
                                                            job.img_gen.gen_params,
                                                            job.img_gen.gen_params.seed + i / images_per_batch)
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

    std::vector<SDImageOwner> ltx_scene_image_owners;
    std::vector<const sd_image_t*> ltx_scene_images;
    std::vector<std::vector<SDImageOwner>> ltx_v2v_image_owners;
    std::vector<std::vector<sd_image_t>> ltx_v2v_images;
    std::vector<sd_image_t*> ltx_v2v_image_windows;
    std::vector<int> ltx_v2v_frame_counts;
    std::vector<const char*> ltx_v2v_guide_latent_paths;
    if (!job.ltx_prompts.empty()) {
        ltx_scene_image_owners.resize(job.ltx_prompts.size());
        ltx_scene_images.resize(job.ltx_prompts.size(), nullptr);
        for (size_t segment = 1; segment < job.ltx_segment_init_images.size(); ++segment) {
            const std::string& encoded = job.ltx_segment_init_images[segment];
            if (encoded.empty()) {
                continue;
            }
            if (!decode_base64_image(encoded,
                                     3,
                                     params.width,
                                     params.height,
                                     ltx_scene_image_owners[segment])) {
                error_message = "failed to decode LTX scene image for segment " + std::to_string(segment + 1);
                return false;
            }
            ltx_scene_images[segment] = &ltx_scene_image_owners[segment].get();
        }
        ltx_v2v_image_owners.resize(job.ltx_prompts.size());
        ltx_v2v_images.resize(job.ltx_prompts.size());
        ltx_v2v_image_windows.resize(job.ltx_prompts.size(), nullptr);
        ltx_v2v_frame_counts.resize(job.ltx_prompts.size(), 0);
        ltx_v2v_guide_latent_paths.resize(job.ltx_prompts.size(), nullptr);
        for (size_t segment = 0; segment < job.ltx_segment_v2v_guide_latent_paths.size(); ++segment) {
            if (!job.ltx_segment_v2v_guide_latent_paths[segment].empty()) {
                ltx_v2v_guide_latent_paths[segment] = job.ltx_segment_v2v_guide_latent_paths[segment].c_str();
            }
        }
        for (size_t segment = 0; segment < job.ltx_segment_control_frames.size(); ++segment) {
            const auto& encoded_frames = job.ltx_segment_control_frames[segment];
            if (encoded_frames.empty()) continue;
            auto& owners = ltx_v2v_image_owners[segment];
            auto& images = ltx_v2v_images[segment];
            owners.resize(encoded_frames.size());
            images.reserve(encoded_frames.size());
            for (size_t frame = 0; frame < encoded_frames.size(); ++frame) {
                if (!decode_base64_image(encoded_frames[frame], 3, params.width, params.height, owners[frame])) {
                    error_message = "failed to decode LTX V2V frame " + std::to_string(frame + 1) +
                                    " for segment " + std::to_string(segment + 1);
                    return false;
                }
                images.push_back(owners[frame].get());
            }
            ltx_v2v_image_windows[segment] = images.data();
            ltx_v2v_frame_counts[segment] = static_cast<int>(images.size());
        }
    }

    SegmentPreviewWriter segment_writer;
    const bool write_segment_previews = !job.ltx_prompts.empty() && job.ltx_emit_segments &&
                                        !job.ltx_bank_dir.empty();
    if (write_segment_previews) {
        segment_writer.directory = job.ltx_bank_dir;
        segment_writer.fps = params.fps;
        segment_writer.quality = job.vid_gen.output_compression;
        segment_writer.start();
    }

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = nullptr;
        bool generated = false;
        if (!job.ltx_prompts.empty()) {
            std::vector<const char*> prompts;
            prompts.reserve(job.ltx_prompts.size());
            for (const auto& prompt : job.ltx_prompts) {
                prompts.push_back(prompt.c_str());
            }
            sd_vid_chain_params_t chain = {};
            chain.n_segments = static_cast<int>(prompts.size());
            chain.segment_prompts = prompts.data();
            chain.segment_video_frames = job.ltx_segment_frames.empty() ? nullptr : job.ltx_segment_frames.data();
            chain.segment_scene_cuts = job.ltx_segment_scene_cuts.empty() ? nullptr : job.ltx_segment_scene_cuts.data();
            chain.segment_init_images = ltx_scene_images.empty() ? nullptr : ltx_scene_images.data();
            chain.segment_control_frames = ltx_v2v_image_windows.empty() ? nullptr : ltx_v2v_image_windows.data();
            chain.segment_control_frame_counts = ltx_v2v_frame_counts.empty() ? nullptr : ltx_v2v_frame_counts.data();
            chain.segment_v2v_modes = job.ltx_segment_v2v_modes.empty() ? nullptr : job.ltx_segment_v2v_modes.data();
            chain.segment_v2v_strengths = job.ltx_segment_v2v_strengths.empty() ? nullptr : job.ltx_segment_v2v_strengths.data();
            chain.segment_v2v_guide_latent_paths = ltx_v2v_guide_latent_paths.empty() ? nullptr : ltx_v2v_guide_latent_paths.data();
            chain.cont_latent_frames = job.ltx_cont_latent_frames;
            chain.start_segment = job.ltx_resume_from;
            chain.bank_dir = job.ltx_bank_dir.empty() ? nullptr : job.ltx_bank_dir.c_str();
            chain.chain_audio_full = job.ltx_chain_audio_full.empty() ? nullptr : job.ltx_chain_audio_full.c_str();
            chain.chain_audio_track = job.ltx_chain_audio_track.empty() ? nullptr : job.ltx_chain_audio_track.c_str();
            chain.chain_audio_offset_frames = job.ltx_chain_audio_offset_frames;
            chain.chain_audio_dir = job.ltx_chain_audio_dir.empty() ? nullptr : job.ltx_chain_audio_dir.c_str();
            chain.on_segment = write_segment_previews ? write_segment_preview : nullptr;
            chain.on_segment_user = write_segment_previews ? &segment_writer : nullptr;
            generated = generate_video_chain(runtime.sd_ctx,
                                              &params,
                                              &chain,
                                              &raw_results,
                                              &num_results,
                                              &generated_audio);
        } else if (!job.wan_vace_prompts.empty()) {
            std::vector<const char*> prompts;
            prompts.reserve(job.wan_vace_prompts.size());
            for (const auto& prompt : job.wan_vace_prompts) {
                prompts.push_back(prompt.c_str());
            }
            sd_wan_vace_chain_params_t chain = {};
            chain.n_segments = static_cast<int>(prompts.size());
            chain.segment_prompts = prompts.data();
            chain.start_segment = job.wan_vace_resume_from;
            chain.bank_dir = job.wan_vace_bank_dir.empty() ? nullptr : job.wan_vace_bank_dir.c_str();
            generated = generate_wan_vace_chain(runtime.sd_ctx,
                                                 &params,
                                                 &chain,
                                                 &raw_results,
                                                 &num_results,
                                                 &generated_audio);
        }
        if (!generated) {
            raw_results = nullptr;
        } else if (runtime.gpu_sharing != nullptr) {
            runtime.gpu_sharing->diffusion_loaded.store(true);
        }
        results.adopt(raw_results, num_results);
    }
    if (write_segment_previews) {
        segment_writer.finish();
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
        }

        std::vector<std::string> output_images;
        std::string output_media_b64;
        std::string output_media_mime_type;
        int output_frame_count = 0;
        int output_fps         = 0;
        std::string error_message;
        bool ok = false;

        if (job->kind == AsyncJobKind::ImgGen) {
            ok = execute_img_gen_job(runtime, *job, output_images, error_message);
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
            if (job->cancel_requested) {
                job->status = AsyncJobStatus::Cancelled;
                job->error_code = "cancelled";
                job->error_message = "job cancelled by client";
                job->result_images_b64.clear();
                job->result_media_b64.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps = 0;
            } else if (ok) {
                job->status                 = AsyncJobStatus::Completed;
                job->result_images_b64      = std::move(output_images);
                job->result_media_b64       = std::move(output_media_b64);
                job->result_media_mime_type = std::move(output_media_mime_type);
                job->result_frame_count     = output_frame_count;
                job->result_fps             = output_fps;
                job->error_code.clear();
                job->error_message.clear();
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
