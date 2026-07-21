// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

WanVaceResumeBank::~WanVaceResumeBank() {
    for (auto& frame : prefix_frames) {
        free(frame.data);
    }
    for (auto& frame : control_tail) {
        free(frame.data);
    }
}

static sd_image_t copy_wan_frame(const sd_image_t& source) {
    sd_image_t copy = source;
    const size_t bytes = static_cast<size_t>(source.width) * source.height * source.channel;
    copy.data = static_cast<uint8_t*>(malloc(bytes));
    if (copy.data != nullptr && source.data != nullptr) {
        std::memcpy(copy.data, source.data, bytes);
    }
    return copy;
}

static void bank_wan_segment(int segment_index,
                             const sd_image_t* frames,
                             int frame_count,
                             const float* latent,
                             int latent_width,
                             int latent_height,
                             int latent_frames,
                             int latent_channels,
                             void* user) {
    auto* bank = static_cast<WanVaceResumeBank*>(user);
    if (bank == nullptr || frames == nullptr || frame_count <= 0 || segment_index != bank->completed_segments) {
        return;
    }
    std::vector<sd_image_t> copied;
    copied.reserve(frame_count);
    for (int frame = 0; frame < frame_count; ++frame) {
        sd_image_t copy = copy_wan_frame(frames[frame]);
        if (copy.data == nullptr) {
            for (auto& allocated : copied) {
                free(allocated.data);
            }
            return;
        }
        copied.push_back(copy);
    }
    for (auto& frame : bank->control_tail) {
        free(frame.data);
    }
    bank->control_tail.clear();
    const int overlap = std::min(frame_count, 5);
    for (int frame = frame_count - overlap; frame < frame_count; ++frame) {
        sd_image_t copy = copy_wan_frame(frames[frame]);
        if (copy.data == nullptr) {
            for (auto& allocated : copied) {
                free(allocated.data);
            }
            return;
        }
        bank->control_tail.push_back(copy);
    }
    bank->prefix_frames.insert(bank->prefix_frames.end(), copied.begin(), copied.end());
    bank->completed_segments++;
    if (latent != nullptr && latent_width > 0 && latent_height > 0 && latent_frames > 0 && latent_channels > 0) {
        const size_t count = static_cast<size_t>(latent_width) * latent_height * latent_frames * latent_channels;
        bank->latent.assign(latent, latent + count);
        bank->latent_width = latent_width;
        bank->latent_height = latent_height;
        bank->latent_frames = latent_frames;
        bank->latent_channels = latent_channels;
    }
}

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
    bool wan_resumed = false;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = nullptr;
        bool generated = false;
        if (job.wan_vace_prompts.empty()) {
            generated = generate_video(runtime.sd_ctx, &params, &raw_results, &num_results, &generated_audio);
        } else {
            std::vector<const char*> prompts;
            prompts.reserve(job.wan_vace_prompts.size());
            for (const auto& prompt : job.wan_vace_prompts) {
                prompts.push_back(prompt.c_str());
            }
            sd_wan_vace_chain_params_t chain = {};
            chain.n_segments = static_cast<int>(prompts.size());
            chain.segment_prompts = prompts.data();
            if (job.wan_vace_bank != nullptr && job.wan_vace_bank->completed_segments > 0) {
                wan_resumed = true;
                chain.start_segment = job.wan_vace_bank->completed_segments;
                chain.resume_control_frames = job.wan_vace_bank->control_tail.data();
                chain.resume_control_frames_size = static_cast<int>(job.wan_vace_bank->control_tail.size());
                chain.resume_latent = job.wan_vace_bank->latent.data();
                chain.resume_latent_width = job.wan_vace_bank->latent_width;
                chain.resume_latent_height = job.wan_vace_bank->latent_height;
                chain.resume_latent_frames = job.wan_vace_bank->latent_frames;
                chain.resume_latent_channels = job.wan_vace_bank->latent_channels;
            }
            if (job.wan_vace_bank == nullptr) {
                job.wan_vace_bank = std::make_shared<WanVaceResumeBank>();
            }
            chain.on_segment = bank_wan_segment;
            chain.on_segment_user = job.wan_vace_bank.get();
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

    // A resumed core chain returns only its newly rendered suffix.  Prepend
    // deep copies of the retained prefix for container encoding; the bank stays
    // valid for a later resume even after this job's output is released.
    if (wan_resumed && job.wan_vace_bank != nullptr &&
        job.wan_vace_bank->completed_segments > 0 && !job.wan_vace_bank->prefix_frames.empty()) {
        const int suffix_count = results.count();
        const int prefix_count = static_cast<int>(job.wan_vace_bank->prefix_frames.size());
        sd_image_t* combined = static_cast<sd_image_t*>(malloc(static_cast<size_t>(prefix_count + suffix_count) * sizeof(sd_image_t)));
        if (combined == nullptr) {
            free_sd_audio(generated_audio);
            error_message = "failed to allocate resumed video frame list";
            return false;
        }
        int copied = 0;
        for (; copied < prefix_count; ++copied) {
            combined[copied] = copy_wan_frame(job.wan_vace_bank->prefix_frames[copied]);
            if (combined[copied].data == nullptr) {
                break;
            }
        }
        if (copied != prefix_count) {
            for (int i = 0; i < copied; ++i) {
                free(combined[i].data);
            }
            free(combined);
            free_sd_audio(generated_audio);
            error_message = "failed to copy resumed video prefix";
            return false;
        }
        for (int frame = 0; frame < suffix_count; ++frame) {
            combined[prefix_count + frame] = results[frame];
            results[frame].data = nullptr;
        }
        results.adopt(combined, prefix_count + suffix_count);
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
