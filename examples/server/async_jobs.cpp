// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <cmath>
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

struct LTXModelLease {
    ServerRuntime* runtime = nullptr;
    const std::vector<std::string>* models = nullptr;
    std::map<std::string, std::string> variants;
    std::string active;
    std::string error;
};

bool lease_ltx_segment_model(int segment, void* user) {
    auto* lease = static_cast<LTXModelLease*>(user);
    if (lease == nullptr || lease->runtime == nullptr || lease->models == nullptr ||
        segment < 0 || static_cast<size_t>(segment) >= lease->models->size()) {
        return false;
    }
    const std::string& wanted = (*lease->models)[segment];
    if (wanted.empty() || wanted == lease->active) return true;
    const auto variant = lease->variants.find(wanted);
    if (variant == lease->variants.end()) {
        lease->error = "unknown LTX segment model variant '" + wanted + "'";
        return false;
    }
    if (!sd_ctx_swap_diffusion_model(lease->runtime->sd_ctx, variant->second.c_str())) {
        lease->error = "could not load LTX segment model variant '" + wanted + "'";
        return false;
    }
    lease->active = wanted;
    return true;
}

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
    int stage = 4;
    std::vector<sd_image_t> frames;
    // Owned copy of the segment's audio. The chain frees its own immediately after the callback
    // returns, and this writer encodes on a background thread, so borrowing would be a use-after-free.
    std::vector<float> audio_samples;
    uint32_t audio_rate = 0;
    uint32_t audio_channels = 0;
    bool pending = false;
    bool busy = false;
    bool stop = false;
    std::thread thread;

    void start() {
        thread = std::thread([this] {
            for (;;) {
                int queued_segment = -1;
                int queued_stage = 4;
                std::vector<sd_image_t> queued_frames;
                std::vector<float> queued_audio;
                uint32_t queued_rate = 0;
                uint32_t queued_channels = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&] { return stop || pending; });
                    if (!pending && stop) break;
                    queued_segment = segment;
                    queued_stage = stage;
                    queued_frames = std::move(frames);
                    queued_audio = std::move(audio_samples);
                    queued_rate = audio_rate;
                    queued_channels = audio_channels;
                    pending = false;
                    busy = true;
                }
                // Give the preview its own sound. A silent per-shot preview is a poor proxy for a
                // shot whose whole point may be the audio (lipsync, music reactivity), and the
                // encoder already takes a track -- only the plumbing was missing.
                sd_audio_t queued_audio_view{};
                const sd_audio_t* audio_arg = nullptr;
                if (!queued_audio.empty() && queued_rate > 0 && queued_channels > 0) {
                    queued_audio_view.sample_rate = queued_rate;
                    queued_audio_view.channels = queued_channels;
                    queued_audio_view.sample_count = queued_audio.size() / queued_channels;
                    queued_audio_view.data = queued_audio.data();
                    audio_arg = &queued_audio_view;
                }
                const auto bytes = create_video_from_sd_images_to_vector(
                    "webm", queued_frames.data(), static_cast<int>(queued_frames.size()), fps, quality, audio_arg);
                if (!bytes.empty()) {
                    const std::string name = queued_stage == 4
                                                 ? "seg_" + std::to_string(queued_segment) + ".webm"
                                                 : "seg_" + std::to_string(queued_segment) + "_stage" +
                                                       std::to_string(queued_stage) + ".webm";
                    const fs::path final_path = fs::path(directory) / name;
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

    void enqueue(int index,
                 int preview_stage,
                 const sd_image_t* source,
                 int count,
                 const sd_audio_t* source_audio) {
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
        // Copy the samples too: the chain frees its sd_audio_t as soon as this returns, and the
        // encode happens later on the writer thread.
        std::vector<float> audio_copy;
        uint32_t rate = 0;
        uint32_t channels = 0;
        if (source_audio != nullptr && source_audio->data != nullptr &&
            source_audio->sample_count > 0 && source_audio->channels > 0) {
            const size_t total = static_cast<size_t>(source_audio->sample_count) * source_audio->channels;
            audio_copy.assign(source_audio->data, source_audio->data + total);
            rate = source_audio->sample_rate;
            channels = source_audio->channels;
        }
        segment = index;
        stage = preview_stage;
        frames = std::move(copies);
        audio_samples = std::move(audio_copy);
        audio_rate = rate;
        audio_channels = channels;
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

void write_segment_preview(int segment,
                           const sd_image_t* frames,
                           int count,
                           const sd_audio_t* audio,
                           void* user) {
    static_cast<SegmentPreviewWriter*>(user)->enqueue(segment, 4, frames, count, audio);
}

// The hires-stage previews are intermediate upscales of a shot whose audio is unchanged by the
// refine, and the stage callback carries none — so they stay silent, deliberately. Only the
// completed shot preview gets sound.
void write_ltx_stage_preview(int segment, int stage, int, int,
                             const sd_image_t* frames, int count, void* user) {
    static_cast<SegmentPreviewWriter*>(user)->enqueue(segment, stage, frames, count, nullptr);
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

bool async_job_in_flight(AsyncJobManager* manager) {
    if (manager == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(manager->mutex);
    for (const auto& entry : manager->jobs) {
        const auto& job = entry.second;
        if (job && (job->status == AsyncJobStatus::Queued || job->status == AsyncJobStatus::Generating)) {
            return true;
        }
    }
    return false;
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
    if ((job.ltx_emit_segments || job.ltx_emit_stages) && !job.ltx_bank_dir.empty() &&
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
            if (job.ltx_emit_stages) {
                for (int stage = 1; stage <= 3; ++stage) {
                    error.clear();
                    const fs::path stage_path = fs::path(job.ltx_bank_dir) /
                                                ("seg_" + std::to_string(segment) + "_stage" +
                                                 std::to_string(stage) + ".webm");
                    if (fs::is_regular_file(stage_path, error)) {
                        partials.push_back({
                            {"segment_index", static_cast<int>(segment)},
                            {"stage", stage},
                            {"url", "/sdcpp/v1/jobs/" + job.id + "/segments/" +
                                        std::to_string(segment) + "?stage=" + std::to_string(stage)},
                        });
                    }
                }
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

        // FLUX.2-Klein dual-DiT: honour the request's top-level model variant by
        // swapping the resident DiT before sampling. Empty variant (the default,
        // and every non-flux2 img_gen caller) leaves the loaded DiT untouched, so
        // single-model renders stay byte-identical. Swap when the wanted variant
        // differs from the resident one, or after an /v1/admin/unload freed the
        // DiT (diffusion_loaded == false) — mirrors prod ensure_variant_loaded
        // (routes_sdcpp.cpp:392-406 -> async_jobs.cpp:1316) using this tree's
        // runtime_diffusion_model_variants() map + sd_ctx_swap_diffusion_model.
        if (!job.img_gen.model_variant.empty() && runtime.gpu_sharing != nullptr) {
            const bool need_swap = !runtime.gpu_sharing->diffusion_loaded.load() ||
                                   runtime.gpu_sharing->loaded_variant != job.img_gen.model_variant;
            if (need_swap) {
                const auto variants = runtime_diffusion_model_variants(runtime);
                const auto variant  = variants.find(job.img_gen.model_variant);
                if (variant == variants.end()) {
                    error_message = "unknown model variant '" + job.img_gen.model_variant + "'";
                    return false;
                }
                if (!sd_ctx_swap_diffusion_model(runtime.sd_ctx, variant->second.c_str())) {
                    error_message = "could not load model variant '" + job.img_gen.model_variant + "'";
                    return false;
                }
                runtime.gpu_sharing->loaded_variant = job.img_gen.model_variant;
            }
        }

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
    // The production LipDub client sends its source frames at the top level,
    // with v2v_mode omitted (therefore mode 0).  Keep the selection in durable
    // job state so queued requests cannot inherit a previous worker setting.
    params.v2v_mode = job.ltx_v2v_mode;
    params.relip_ref_tstride = std::max(1, job.ltx_relip_ref_tstride);
    std::vector<sd_hires_params_t> ltx_hires_chain;
    if (!job.ltx_hires_stages.empty()) {
        ltx_hires_chain.reserve(job.ltx_hires_stages.size());
        for (size_t index = 0; index < job.ltx_hires_stages.size(); ++index) {
            sd_hires_params_t stage = job.ltx_hires_stages[index].to_sd_vid_gen_params_t().hires;
            stage.sample_method = index < job.ltx_hires_stage_methods.size()
                                      ? job.ltx_hires_stage_methods[index]
                                      : SAMPLE_METHOD_COUNT;
            stage.cfg = index < job.ltx_hires_stage_cfgs.size()
                            ? job.ltx_hires_stage_cfgs[index]
                            : NAN;
            ltx_hires_chain.push_back(stage);
        }
        params.hires_chain = ltx_hires_chain.data();
        params.hires_chain_count = static_cast<int>(ltx_hires_chain.size());
    }

    SDImageVec results;
    int num_results             = 0;
    sd_audio_t* generated_audio = nullptr;

    std::vector<SDImageOwner> ltx_scene_image_owners;
    std::vector<const sd_image_t*> ltx_scene_images;
    std::vector<std::vector<SDImageOwner>> ltx_keyframe_image_owners;
    std::vector<std::vector<sd_image_t>> ltx_keyframe_images;
    std::vector<sd_image_t*> ltx_keyframe_image_windows;
    std::vector<const int*> ltx_keyframe_indices;
    std::vector<int> ltx_keyframe_counts;
    std::vector<std::vector<SDImageOwner>> ltx_v2v_image_owners;
    std::vector<std::vector<sd_image_t>> ltx_v2v_images;
    std::vector<sd_image_t*> ltx_v2v_image_windows;
    std::vector<int> ltx_v2v_frame_counts;
    std::vector<const char*> ltx_v2v_guide_latent_paths;
    if (!job.ltx_prompts.empty()) {
        ltx_scene_image_owners.resize(job.ltx_prompts.size());
        ltx_scene_images.resize(job.ltx_prompts.size(), nullptr);
        ltx_keyframe_image_owners.resize(job.ltx_prompts.size());
        ltx_keyframe_images.resize(job.ltx_prompts.size());
        ltx_keyframe_image_windows.resize(job.ltx_prompts.size(), nullptr);
        ltx_keyframe_indices.resize(job.ltx_prompts.size(), nullptr);
        ltx_keyframe_counts.resize(job.ltx_prompts.size(), 0);
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
        for (size_t segment = 0; segment < job.ltx_segment_keyframes.size(); ++segment) {
            const auto& encoded_keyframes = job.ltx_segment_keyframes[segment];
            if (encoded_keyframes.empty()) continue;
            if (segment >= job.ltx_segment_keyframe_indices.size() ||
                encoded_keyframes.size() != job.ltx_segment_keyframe_indices[segment].size()) {
                error_message = "invalid LTX keyframe job state for segment " + std::to_string(segment + 1);
                return false;
            }
            auto& owners = ltx_keyframe_image_owners[segment];
            auto& images = ltx_keyframe_images[segment];
            owners.resize(encoded_keyframes.size());
            images.reserve(encoded_keyframes.size());
            for (size_t keyframe = 0; keyframe < encoded_keyframes.size(); ++keyframe) {
                if (!decode_base64_image(encoded_keyframes[keyframe], 3, params.width, params.height, owners[keyframe])) {
                    error_message = "failed to decode LTX keyframe " + std::to_string(keyframe + 1) +
                                    " for segment " + std::to_string(segment + 1);
                    return false;
                }
                images.push_back(owners[keyframe].get());
            }
            ltx_keyframe_image_windows[segment] = images.data();
            ltx_keyframe_indices[segment] = job.ltx_segment_keyframe_indices[segment].data();
            ltx_keyframe_counts[segment] = static_cast<int>(images.size());
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
        // Koblem LipDub predates the per-segment Director schema: it posts a
        // single top-level control_frames array plus audio_0.  The common
        // request parser owns these decoded frame views in `params`; bridge
        // that exact borrowed span into segment zero so generate_video_chain
        // reaches the mode-0 reference path rather than dropping it.
        if (job.ltx_v2v_mode == 0 && job.ltx_prompts.size() == 1 &&
            params.control_frames != nullptr && params.control_frames_size > 0) {
            ltx_v2v_image_windows[0] = params.control_frames;
            ltx_v2v_frame_counts[0] = params.control_frames_size;
        }
    }

    // TASS overlap character references. These are the one image input the
    // engine deliberately does NOT force into the render bucket: the sheet is
    // appended on the DiT token axis, so `native_resolution` keeps every pixel
    // of a 1536x1024 sheet while the video stays 768x448.
    std::vector<SDImageOwner> ltx_character_ref_owners;
    std::vector<sd_image_t> ltx_character_ref_images;
    std::vector<int> ltx_character_ref_source_ids;
    if (!job.ltx_character_refs.empty()) {
        ltx_character_ref_owners.resize(job.ltx_character_refs.size());
        ltx_character_ref_images.reserve(job.ltx_character_refs.size());
        ltx_character_ref_source_ids.reserve(job.ltx_character_refs.size());
        for (size_t index = 0; index < job.ltx_character_refs.size(); ++index) {
            const LtxCharacterRef& reference = job.ltx_character_refs[index];
            const int target_width  = reference.match_target ? params.width : 0;
            const int target_height = reference.match_target ? params.height : 0;
            if (!reference.image.empty() && reference.image[0] == '/') {
                sd_image_t loaded = {};
                if (!load_sd_image_from_file(&loaded, reference.image.c_str(), target_width, target_height, 3)) {
                    error_message = "failed to load LTX character reference " + std::to_string(index + 1);
                    return false;
                }
                ltx_character_ref_owners[index].reset(loaded);
            } else if (!decode_base64_image(reference.image,
                                            3,
                                            target_width,
                                            target_height,
                                            ltx_character_ref_owners[index])) {
                error_message = "failed to decode LTX character reference " + std::to_string(index + 1);
                return false;
            }
            ltx_character_ref_images.push_back(ltx_character_ref_owners[index].get());
            ltx_character_ref_source_ids.push_back(reference.source_id);
        }
        params.character_refs           = ltx_character_ref_images.data();
        params.character_ref_source_ids = ltx_character_ref_source_ids.data();
        params.character_refs_size      = static_cast<int>(ltx_character_ref_images.size());
        params.tass_phase_scale         = job.ltx_tass_phase_scale;
    }

    SegmentPreviewWriter segment_writer;
    const bool write_segment_previews = !job.ltx_prompts.empty() && (job.ltx_emit_segments || job.ltx_emit_stages) &&
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
            chain.segment_keyframes = ltx_keyframe_image_windows.empty() ? nullptr : ltx_keyframe_image_windows.data();
            chain.segment_keyframe_indices = ltx_keyframe_indices.empty() ? nullptr : ltx_keyframe_indices.data();
            chain.segment_keyframe_counts = ltx_keyframe_counts.empty() ? nullptr : ltx_keyframe_counts.data();
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
            std::vector<const char*> segment_audio_full;
            std::vector<const char*> segment_audio_track;
            if (!job.ltx_segment_audio_full.empty()) {
                segment_audio_full.reserve(job.ltx_segment_audio_full.size());
                for (const auto& path : job.ltx_segment_audio_full) {
                    segment_audio_full.push_back(path.empty() ? nullptr : path.c_str());
                }
            }
            if (!job.ltx_segment_audio_track.empty()) {
                segment_audio_track.reserve(job.ltx_segment_audio_track.size());
                for (const auto& path : job.ltx_segment_audio_track) {
                    segment_audio_track.push_back(path.empty() ? nullptr : path.c_str());
                }
            }
            // Prompt Relay beat views. The sd_ltx_beat_t text pointers borrow
            // the job's own strings, which outlive this call.
            std::vector<std::vector<sd_ltx_beat_t>> segment_beat_storage;
            std::vector<const sd_ltx_beat_t*> segment_beat_views;
            std::vector<int> segment_beat_counts;
            if (!job.ltx_segment_beats.empty()) {
                segment_beat_storage.resize(job.ltx_prompts.size());
                segment_beat_views.resize(job.ltx_prompts.size(), nullptr);
                segment_beat_counts.resize(job.ltx_prompts.size(), 0);
                for (size_t segment = 0;
                     segment < job.ltx_segment_beats.size() && segment < job.ltx_prompts.size();
                     ++segment) {
                    const auto& beats = job.ltx_segment_beats[segment];
                    if (beats.empty()) {
                        continue;
                    }
                    auto& storage = segment_beat_storage[segment];
                    storage.reserve(beats.size());
                    for (const auto& beat : beats) {
                        storage.push_back(sd_ltx_beat_t{beat.frame,
                                                        beat.text.c_str(),
                                                        beat.strength,
                                                        beat.window});
                    }
                    segment_beat_views[segment]  = storage.data();
                    segment_beat_counts[segment] = static_cast<int>(storage.size());
                }
                chain.segment_beats       = segment_beat_views.data();
                chain.segment_beat_counts = segment_beat_counts.data();
            }
            std::vector<const char*> segment_negative_prompts;
            if (!job.ltx_segment_negative_prompts.empty()) {
                segment_negative_prompts.reserve(job.ltx_segment_negative_prompts.size());
                for (const auto& negative : job.ltx_segment_negative_prompts) {
                    segment_negative_prompts.push_back(negative.empty() ? nullptr : negative.c_str());
                }
                chain.segment_negative_prompts = segment_negative_prompts.data();
            }
            chain.segment_seeds = job.ltx_segment_seeds.empty() ? nullptr : job.ltx_segment_seeds.data();
            chain.segment_steps = job.ltx_segment_steps.empty() ? nullptr : job.ltx_segment_steps.data();
            chain.segment_cfg   = job.ltx_segment_cfg.empty() ? nullptr : job.ltx_segment_cfg.data();
            chain.retake_segment = job.ltx_retake_segment;
            chain.enable_retake = job.ltx_retake_segment >= 0;
            chain.cont_seam_drop_frames = job.ltx_cont_seam_drop_frames;
            chain.segment_seam_drop_frames = job.ltx_segment_seam_drop_frames.empty()
                                                 ? nullptr
                                                 : job.ltx_segment_seam_drop_frames.data();
            chain.segment_audio_full = segment_audio_full.empty() ? nullptr : segment_audio_full.data();
            chain.segment_audio_track = segment_audio_track.empty() ? nullptr : segment_audio_track.data();
            LTXModelLease model_lease;
            model_lease.runtime = &runtime;
            model_lease.models = &job.ltx_segment_models;
            model_lease.variants = runtime_diffusion_model_variants(runtime);
            model_lease.active = job.ltx_default_model;
            chain.before_segment = job.ltx_segment_models.empty() ? nullptr : lease_ltx_segment_model;
            chain.before_segment_user = job.ltx_segment_models.empty() ? nullptr : &model_lease;
            chain.on_segment = write_segment_previews ? write_segment_preview : nullptr;
            chain.on_segment_user = write_segment_previews ? &segment_writer : nullptr;
            params.emit_stages = job.ltx_emit_stages ? 1 : 0;
            params.on_stage = job.ltx_emit_stages ? write_ltx_stage_preview : nullptr;
            params.on_stage_user = job.ltx_emit_stages ? &segment_writer : nullptr;
            generated = generate_video_chain(runtime.sd_ctx,
                                              &params,
                                              &chain,
                                              &raw_results,
                                              &num_results,
                                              &generated_audio);
            // A child can serve another job after this one. Restore its
            // supervisor-selected model so a segment override cannot leak into
            // a subsequent request whose top-level model was unchanged.
            if (!job.ltx_segment_models.empty() && model_lease.active != job.ltx_default_model) {
                const auto default_variant = model_lease.variants.find(job.ltx_default_model);
                if (default_variant == model_lease.variants.end() ||
                    !sd_ctx_swap_diffusion_model(runtime.sd_ctx, default_variant->second.c_str())) {
                    generated = false;
                    error_message = "could not restore LTX default model variant '" + job.ltx_default_model + "'";
                }
            }
            if (!generated && error_message.empty() && !model_lease.error.empty()) {
                error_message = model_lease.error;
            }
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
