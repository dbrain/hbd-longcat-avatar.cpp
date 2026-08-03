// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <chrono>
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

// Per-segment DiT variant AND runtime-LoRA state for one chain. Both ride the SAME
// `chain.before_segment` hook, because that hook is a plain bool(int, void*) with nothing
// model-specific about it. Either array may be empty; whichever is populated is leased.
using LTXLoraSet = std::vector<std::pair<std::string, float>>;   // resolved full path -> multiplier

struct LTXModelLease {
    ServerRuntime* runtime = nullptr;
    const std::vector<std::string>* models = nullptr;
    // Per-segment adapter sets. A non-empty entry REPLACES the request's top-level set for that
    // segment (it does not add to it), so a segment can also drop an inherited adapter by
    // naming a different set. An EMPTY entry means "inherit", i.e. leave whatever is active.
    const std::vector<LTXLoraSet>* loras = nullptr;
    std::map<std::string, std::string> variants;
    std::string active;
    // What the adapter stack currently holds, so an unchanged segment costs nothing. Seeded by
    // the caller with the request's top-level set (see the chain wiring below), otherwise the
    // first segment would re-apply an identical stack for ~1s of nothing.
    LTXLoraSet active_loras;
    std::string error;
};

// Apply one segment's adapter set, if it differs from what is resident. Split out so the
// end-of-chain restore can reuse it verbatim rather than duplicating the sd_lora_t marshalling.
bool apply_ltx_lora_set(LTXModelLease* lease, const LTXLoraSet& wanted) {
    if (wanted == lease->active_loras) return true;
    std::vector<sd_lora_t> specs;
    specs.reserve(wanted.size());
    for (const auto& [path, multiplier] : wanted) {
        sd_lora_t spec{};
        spec.path          = path.c_str();
        spec.multiplier    = multiplier;
        spec.is_high_noise = false;
        specs.push_back(spec);
    }
    // Empty set is legal and meaningful: it CLEARS the adapters for this segment.
    if (!sd_ctx_apply_loras(lease->runtime->sd_ctx, specs.empty() ? nullptr : specs.data(),
                            static_cast<uint32_t>(specs.size()))) {
        lease->error = "could not apply LTX segment LoRAs";
        return false;
    }
    lease->active_loras = wanted;
    return true;
}

bool lease_ltx_segment_model(int segment, void* user) {
    auto* lease = static_cast<LTXModelLease*>(user);
    if (lease == nullptr || lease->runtime == nullptr || segment < 0) {
        return false;
    }
    // The DiT variant first: swapping weights rebuilds the runner, and an adapter is bound to
    // the model it was applied to, so re-applying AFTER the swap is the only safe order.
    if (lease->models != nullptr && static_cast<size_t>(segment) < lease->models->size()) {
        const std::string& wanted = (*lease->models)[segment];
        if (!wanted.empty() && wanted != lease->active) {
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
            // A variant swap re-registers the DiT weights, so the adapter stack that was bound
            // to the outgoing model is gone. Forget it, or the next comparison would wrongly
            // conclude the wanted set is already resident and skip re-applying it.
            lease->active_loras.clear();
        }
    }
    if (lease->loras != nullptr && static_cast<size_t>(segment) < lease->loras->size()) {
        const LTXLoraSet& wanted = (*lease->loras)[segment];
        // Empty entry = inherit whatever is active (the top-level set, or the previous shot's).
        if (!wanted.empty() && !apply_ltx_lora_set(lease, wanted)) {
            return false;
        }
    }
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

namespace {
// Chain flush callback: the chain hands over frames that are final and TRANSFERS OWNERSHIP, so we
// encode each one and free it immediately. That is what keeps a long chain's peak memory at one
// segment rather than the whole decoded timeline.
void stream_flush_frames(const sd_image_t* frames, int frame_count, void* user) {
    auto* encoder = static_cast<IncrementalWebmEncoder*>(user);
    for (int i = 0; i < frame_count; ++i) {
        if (encoder != nullptr) {
            encoder->append(frames[i]);
        }
        free(frames[i].data);
    }
}
}  // namespace

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

    // The derived task ("t2va" / "fl2va" / "ref2va", or "mixed" when the shots differ) is echoed on
    // every poll, not only in the 202, so a client that reattached after a restart still learns what
    // the job it is watching actually is.
    if (!job.h3_task.empty()) {
        result["task"]     = job.h3_task;
        result["segments"] = static_cast<int>(job.h3_prompts.size());
    }

    // MiniMax-H3 publishes the SAME artefact under the SAME URL as LTX -- one seg_<n>.webm per
    // shot, listed here as it lands -- so koblem's progressive-delivery machinery needs no fork.
    // The bank holds nothing else: there is no latent to resume from and no stage previews,
    // because an H3 shot renders in a single pass.
    if (job.h3_emit_segments && !job.h3_bank_dir.empty() &&
        (job.status == AsyncJobStatus::Generating || job.status == AsyncJobStatus::Completed)) {
        json partials = json::array();
        for (size_t segment = 0; segment < job.h3_prompts.size(); ++segment) {
            std::error_code error;
            const fs::path path = fs::path(job.h3_bank_dir) / ("seg_" + std::to_string(segment) + ".webm");
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
    params.audio_fill_gaps = job.ltx_audio_fill_gaps ? 1 : 0;
    // The production LipDub client sends its source frames at the top level,
    // with v2v_mode omitted (therefore mode 0).  Keep the selection in durable
    // job state so queued requests cannot inherit a previous worker setting.
    params.v2v_mode = job.ltx_v2v_mode;
    params.relip_ref_tstride = std::max(1, job.ltx_relip_ref_tstride);
    // Opt-in reference head-frame trim (0 off / -1 auto / >0 explicit). Durable job state for the
    // same reason the seam policy is: a queued request must render under the contract it was
    // staged with, not under whatever the worker happens to default to.
    params.reference_head_trim = job.ltx_reference_head_trim;
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
    std::vector<const char*> ltx_segment_bank_dirs;
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
        if (!job.ltx_segment_bank_dirs.empty()) {
            ltx_segment_bank_dirs.resize(job.ltx_prompts.size(), nullptr);
            for (size_t segment = 0;
                 segment < job.ltx_segment_bank_dirs.size() && segment < ltx_segment_bank_dirs.size();
                 ++segment) {
                if (!job.ltx_segment_bank_dirs[segment].empty()) {
                    ltx_segment_bank_dirs[segment] = job.ltx_segment_bank_dirs[segment].c_str();
                }
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
    // Optional per-sheet shot scope, flattened for the C API: one count per reference plus the
    // concatenation of their segment-index lists. Left empty -- and the params pointers left null
    // -- unless at least one reference actually carries a scope, so unscoped requests reach the
    // engine in exactly the shape they always did.
    std::vector<int> ltx_character_ref_segments;
    std::vector<int> ltx_character_ref_segment_counts;
    if (!job.ltx_character_refs.empty()) {
        ltx_character_ref_owners.resize(job.ltx_character_refs.size());
        ltx_character_ref_images.reserve(job.ltx_character_refs.size());
        ltx_character_ref_source_ids.reserve(job.ltx_character_refs.size());
        for (size_t index = 0; index < job.ltx_character_refs.size(); ++index) {
            const LtxCharacterRef& reference = job.ltx_character_refs[index];
            const int target_width  = reference.match_target ? params.width : 0;
            const int target_height = reference.match_target ? params.height : 0;
            if (ltx_source_is_path(reference.image)) {
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

        const bool any_scoped = std::any_of(job.ltx_character_refs.begin(),
                                            job.ltx_character_refs.end(),
                                            [](const LtxCharacterRef& ref) { return ref.scoped; });
        if (any_scoped) {
            // The flat form has no "applies everywhere" encoding, so once ANY sheet is scoped the
            // unscoped ones are written out as the full segment enumeration. That keeps a count of
            // zero unambiguously meaning "no segment".
            const int segment_count = std::max<int>(1, static_cast<int>(job.ltx_prompts.size()));
            ltx_character_ref_segment_counts.reserve(job.ltx_character_refs.size());
            for (const LtxCharacterRef& reference : job.ltx_character_refs) {
                if (reference.scoped) {
                    ltx_character_ref_segment_counts.push_back(static_cast<int>(reference.segments.size()));
                    ltx_character_ref_segments.insert(ltx_character_ref_segments.end(),
                                                      reference.segments.begin(),
                                                      reference.segments.end());
                } else {
                    ltx_character_ref_segment_counts.push_back(segment_count);
                    for (int segment = 0; segment < segment_count; ++segment) {
                        ltx_character_ref_segments.push_back(segment);
                    }
                }
            }
            params.character_ref_segments       = ltx_character_ref_segments.empty()
                                                      ? nullptr
                                                      : ltx_character_ref_segments.data();
            params.character_ref_segment_counts = ltx_character_ref_segment_counts.data();
        }
    }

    // MSR reference strip. The images are decoded at their NATIVE size -- the engine
    // fits them onto the render canvas itself, because cover-crop for the background
    // and letterbox-on-white for a subject are different operations and only the
    // engine knows which is which. Owners live to the end of the render, like every
    // other image view handed across the C API.
    SDImageOwner ltx_msr_background_owner;
    std::vector<SDImageOwner> ltx_msr_subject_owners;
    std::vector<sd_image_t> ltx_msr_subject_images;
    if (job.ltx_msr_frames > 0) {
        auto load_msr_image = [&](const std::string& source, SDImageOwner& owner, const std::string& what) {
            if (ltx_source_is_path(source)) {
                sd_image_t loaded = {};
                if (!load_sd_image_from_file(&loaded, source.c_str(), 0, 0, 3)) {
                    error_message = "failed to load LTX MSR " + what;
                    return false;
                }
                owner.reset(loaded);
                return true;
            }
            if (!decode_base64_image(source, 3, 0, 0, owner)) {
                error_message = "failed to decode LTX MSR " + what;
                return false;
            }
            return true;
        };
        if (!load_msr_image(job.ltx_msr_background, ltx_msr_background_owner, "background")) {
            return false;
        }
        ltx_msr_subject_owners.resize(job.ltx_msr_subjects.size());
        ltx_msr_subject_images.reserve(job.ltx_msr_subjects.size());
        for (size_t index = 0; index < job.ltx_msr_subjects.size(); ++index) {
            if (!load_msr_image(job.ltx_msr_subjects[index],
                                ltx_msr_subject_owners[index],
                                "subject " + std::to_string(index + 1))) {
                return false;
            }
            ltx_msr_subject_images.push_back(ltx_msr_subject_owners[index].get());
        }
        params.msr_background    = &ltx_msr_background_owner.get();
        params.msr_subjects      = ltx_msr_subject_images.empty() ? nullptr : ltx_msr_subject_images.data();
        params.msr_subjects_size = static_cast<int>(ltx_msr_subject_images.size());
        params.msr_frames        = job.ltx_msr_frames;
        if (!job.ltx_msr_segments.empty()) {
            params.msr_segments      = job.ltx_msr_segments.data();
            params.msr_segments_size = static_cast<int>(job.ltx_msr_segments.size());
        }
        // The strip and the sheets share ONE TASS block, so the phase scale is a
        // single decision for the request. The character-ref path already forwarded
        // it when there were sheets; forward it here only when there were none.
        if (job.ltx_character_refs.empty()) {
            params.tass_phase_scale = job.ltx_tass_phase_scale;
        }
    }

    // MiniMax-H3 `ref2va` references. The route has already validated these, resolved the 2 fps
    // conditioner frame subset and computed the block timestamps; this decodes each reference's
    // pixels ONCE for the whole job and records where they landed, so a chain of N shots pays one
    // decode rather than N.
    //
    // ⚠️ ORDER IS SEMANTIC and is preserved exactly: a reference's position assigns its
    // `<Picture i>` / `<Video k>` / `<Audio j>` ordinal AND advances the shared rotary clock, so
    // grouping by kind here would silently render a different request than the one submitted.
    // Filtering a shot's references out narrows the list but never reorders it.
    struct H3StagedReference {
        int kind         = 0;
        int frame_count  = 0;
        size_t frame_offset = 0;  // into h3_frames
        std::vector<float> block_timestamps;
        const MiniMaxH3JobReference* source = nullptr;
    };
    std::vector<SDImageOwner> h3_frame_owners;
    std::vector<sd_image_t> h3_frames;
    std::vector<H3StagedReference> h3_staged_refs;
    if (!job.h3_references.empty()) {
        size_t total_frames = 0;
        for (const MiniMaxH3JobReference& reference : job.h3_references) {
            total_frames += reference.frames.size() + (reference.image.empty() ? 0 : 1);
        }
        h3_frame_owners.resize(total_frames);
        h3_frames.reserve(total_frames);
        h3_staged_refs.reserve(job.h3_references.size());

        size_t owner_cursor = 0;
        auto decode_frame = [&](const std::string& source, const std::string& what) {
            if (ltx_source_is_path(source)) {
                sd_image_t loaded = {};
                if (!load_sd_image_from_file(&loaded, source.c_str(), 0, 0, 3)) {
                    error_message = "failed to load MiniMax-H3 " + what;
                    return false;
                }
                h3_frame_owners[owner_cursor].reset(loaded);
            } else if (!decode_base64_image(source, 3, 0, 0, h3_frame_owners[owner_cursor])) {
                error_message = "failed to decode MiniMax-H3 " + what;
                return false;
            }
            h3_frames.push_back(h3_frame_owners[owner_cursor].get());
            owner_cursor++;
            return true;
        };

        for (size_t index = 0; index < job.h3_references.size(); ++index) {
            const MiniMaxH3JobReference& reference = job.h3_references[index];
            const std::string label                = "reference " + std::to_string(index + 1);
            H3StagedReference staged;
            staged.source       = &reference;
            staged.frame_offset = h3_frames.size();
            switch (reference.kind) {
                case MiniMaxH3JobReference::Kind::Image: {
                    if (!decode_frame(reference.image, label)) {
                        return false;
                    }
                    staged.kind        = 0;
                    staged.frame_count = 1;
                    break;
                }
                case MiniMaxH3JobReference::Kind::Video: {
                    for (size_t frame = 0; frame < reference.frames.size(); ++frame) {
                        if (!decode_frame(reference.frames[frame], label + " frame " + std::to_string(frame + 1))) {
                            return false;
                        }
                    }
                    staged.kind        = 1;
                    staged.frame_count = static_cast<int>(reference.frames.size());
                    break;
                }
                case MiniMaxH3JobReference::Kind::Audio: {
                    // No pixels at all. A count of ZERO here means exactly that -- it is never a
                    // synonym for "all", which is the same distinction character_ref_segment_counts
                    // documents on the LTX side.
                    staged.kind        = 2;
                    staged.frame_count = 0;
                    break;
                }
            }
            staged.block_timestamps.reserve(reference.block_timestamps.size());
            for (double timestamp : reference.block_timestamps) {
                staged.block_timestamps.push_back(static_cast<float>(timestamp));
            }
            h3_staged_refs.push_back(std::move(staged));
        }
    }
    if (!job.h3_prompts.empty()) {
        // The VIDEO shift is already folded into sample_params.flow_shift by the route -- that is
        // the schedule the sampler walks. Only the audio shift needs its own field.
        params.minimax_h3_sigma_shift_audio = job.h3_sigma_shift_audio;
    }

    SegmentPreviewWriter segment_writer;
    const bool write_h3_segment_previews =
        !job.h3_prompts.empty() && job.h3_emit_segments && !job.h3_bank_dir.empty();
    const bool write_segment_previews = (!job.ltx_prompts.empty() && (job.ltx_emit_segments || job.ltx_emit_stages) &&
                                         !job.ltx_bank_dir.empty()) ||
                                        write_h3_segment_previews;
    if (write_segment_previews) {
        segment_writer.directory = write_h3_segment_previews ? job.h3_bank_dir : job.ltx_bank_dir;
        segment_writer.fps = params.fps;
        segment_writer.quality = job.vid_gen.output_compression;
        segment_writer.start();
    }

    IncrementalWebmEncoder stream_encoder;
    // The MiniMax-H3 chain accumulates its own output (it drives N generate_video calls rather
    // than one), so the shared `results.adopt()` below must not run and clear it.
    bool results_already_owned = false;
    // ...and because it accumulates, a mid-chain failure leaves REAL frames behind. The generic
    // tail below infers success from `num_results > 0`, which for every other path is zero after a
    // failure precisely because `adopt(nullptr, 0)` cleared it. Without this flag a 12-shot job
    // that died on shot 7 would be delivered as a complete 6-shot video.
    bool h3_incomplete = false;
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
            chain.segment_pin_strengths = job.ltx_segment_pin_strengths.empty() ? nullptr : job.ltx_segment_pin_strengths.data();
            chain.segment_v2v_guide_latent_paths = ltx_v2v_guide_latent_paths.empty() ? nullptr : ltx_v2v_guide_latent_paths.data();
            chain.cont_latent_frames = job.ltx_cont_latent_frames;
            chain.start_segment = job.ltx_resume_from;
            chain.bank_dir = job.ltx_bank_dir.empty() ? nullptr : job.ltx_bank_dir.c_str();
            chain.segment_bank_dirs = ltx_segment_bank_dirs.empty() ? nullptr : ltx_segment_bank_dirs.data();
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
            chain.segment_reference_head_trim = job.ltx_segment_reference_head_trim.empty()
                                                    ? nullptr
                                                    : job.ltx_segment_reference_head_trim.data();
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
            model_lease.loras = &job.ltx_segment_loras;
            model_lease.variants = runtime_diffusion_model_variants(runtime);
            model_lease.active = job.ltx_default_model;
            // Seed with the request's TOP-LEVEL adapter stack, which sd_vid_gen_params already
            // applied before the chain started. Without this the first segment that names the
            // same set would re-apply an identical stack for ~1s of nothing.
            model_lease.active_loras = job.ltx_default_loras;
            // One hook serves both knobs; arm it if EITHER array is populated.
            const bool lease_per_segment =
                !job.ltx_segment_models.empty() || !job.ltx_segment_loras.empty();
            chain.before_segment = lease_per_segment ? lease_ltx_segment_model : nullptr;
            chain.before_segment_user = lease_per_segment ? &model_lease : nullptr;
            chain.on_segment = write_segment_previews ? write_segment_preview : nullptr;
            chain.on_segment_user = write_segment_previews ? &segment_writer : nullptr;
            // STREAM the timeline out instead of accumulating it. A 3.5-minute 1280x704 chain
            // holds ~14 GB of decoded frames if we keep them all (~32 GB at 1920x1088); with
            // this the chain hands over each frame as it becomes final and we encode it
            // immediately, so only compressed packets survive. Only for webm - the other
            // containers have no incremental path - and begin() fails without VP9, in which
            // case we simply do not set the callback and the old behaviour stands.
            if (job.vid_gen.output_format == "webm" &&
                stream_encoder.begin(params.width, params.height, params.fps)) {
                chain.on_flush_frames = stream_flush_frames;
                chain.on_flush_frames_user = &stream_encoder;
            }
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
                } else {
                    // The swap dropped the adapter stack with the outgoing DiT, so the restore
                    // below must re-apply rather than compare against a stale record.
                    model_lease.active_loras.clear();
                }
            }
            // Same contract for the adapters: a per-shot override must not leak into the next
            // request. Restoring to the job's TOP-LEVEL set (not to empty) is what makes the
            // next job's first render match a fresh worker's.
            if (!job.ltx_segment_loras.empty() && !apply_ltx_lora_set(&model_lease, job.ltx_default_loras)) {
                generated = false;
                if (error_message.empty()) {
                    error_message = "could not restore LTX default runtime LoRAs";
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
        } else if (!job.h3_prompts.empty()) {
            // ── MiniMax-H3: LTX's job shape over H3's one-sequence-per-denoise model ──────────
            //
            // ONE PACKED SEQUENCE PER DENOISE IS FIXED; ONE PROMPT PER REQUEST WAS NOT. So this is
            // a loop of independent renders rather than a chain: no latent is carried across the
            // seam, no bank is written, and nothing here can resume. A shot's `first_frame` /
            // `last_frame` are its fl2va anchors -- the reference implementation pins those two
            // keyframes at resolved indices 0 and frame_count-1 and nowhere else -- and they are
            // supplied by the CALLER, which is what replaces LTX's `cont_latent_frames` overlap.
            //
            // ★ ONE RESIDENCY PER STAGE FOR THE WHOLE JOB, which is the reason the segment list
            // exists at all. The Qwen3-VL text encoder is ~14.5 GB resident at NVFP4 and the DiT
            // is ~11 GB, against a 15,888 MiB card: they cannot be co-resident, so the swap is
            // unavoidable and the only question is whether it happens once or N times. So:
            //
            //   PHASE 1  encode EVERY shot's prompt, in one text-encoder residency
            //   PHASE 2  reclaim the text encoder -- params home, cache buffer and CUDA pool
            //   PHASE 3  render every shot; the DiT is loaded once and each shot's conditioning
            //            comes back out of the staging cache instead of off the TE
            //
            // Everything else is amortised by the loop simply living here: it holds `sd_ctx_mutex`
            // for the whole run so nothing can interleave, the reference pixels above are decoded
            // once, and nothing between shots calls reclaim_ltx_chain_window_gpu_memory() --
            // generate_video() scopes that to LTX -- so the params-backend homes stay warm.
            //
            // Staging is an OPTIMISATION, never a correctness dependency: if any shot fails to
            // stage, the cache is dropped and every shot encodes in place exactly as it would
            // have, one swap each. `MINIMAX_H3_STAGED_TE=0` takes that path deliberately, which is
            // the runtime rollback for a path that has never met real weights.
            const size_t segment_count = job.h3_prompts.size();
            std::vector<float> h3_audio_samples;
            uint32_t h3_audio_rate     = 0;
            uint32_t h3_audio_channels = 0;
            const bool h3_stream = job.vid_gen.output_format == "webm" &&
                                   stream_encoder.begin(params.width, params.height, params.fps);

            // Every shot's fully-resolved request, built BEFORE either phase runs because phase 1
            // needs the same `sd_vid_gen_params_t` phase 3 will render from -- the conditioner's
            // presentation depends on the references in scope for that shot, so staging a shot
            // from anything other than its own params would encode a different request.
            //
            // ⚠️ `plans` is sized ONCE and never grows. Each plan's params holds raw pointers into
            // that plan's own vectors, so a reallocation would dangle every one of them.
            struct H3SegmentPlan {
                SDImageOwner init_owner;
                SDImageOwner end_owner;
                std::vector<sd_image_t> ref_frames;
                std::vector<int> ref_kinds;
                std::vector<int> ref_frame_counts;
                std::vector<int> ref_conditioner_frames;
                std::vector<int> ref_conditioner_frame_counts;
                std::vector<float> ref_block_timestamps;
                std::vector<int> ref_block_timestamp_counts;
                std::vector<const char*> ref_audio_paths;
                sd_vid_gen_params_t params = {};
            };
            std::vector<H3SegmentPlan> plans(segment_count);

            generated = true;
            for (size_t segment = 0; segment < segment_count && generated; ++segment) {
                H3SegmentPlan& plan = plans[segment];
                plan.params         = params;
                plan.params.prompt  = job.h3_prompts[segment].c_str();
                plan.params.video_frames = job.h3_segment_frames[segment];
                // LTX's sentinels exactly: a negative seed and a zero step count inherit the
                // request-level value rather than meaning "seed 0" / "zero steps".
                if (job.h3_segment_seeds[segment] >= 0) {
                    plan.params.seed = job.h3_segment_seeds[segment];
                }
                if (job.h3_segment_steps[segment] > 0) {
                    plan.params.sample_params.sample_steps = job.h3_segment_steps[segment];
                }

                // fl2va anchors, decoded to the render canvas. Segment 0 takes exactly this path
                // too -- the route folded the request's top-level init_image/end_image into
                // h3_segment_*_images[0] and erased them from the body -- so no shot is special.
                const auto decode_anchor = [&](const std::string& source,
                                               SDImageOwner& owner,
                                               const char* what) {
                    if (source.empty()) {
                        return true;
                    }
                    if (ltx_source_is_path(source)) {
                        sd_image_t loaded = {};
                        if (!load_sd_image_from_file(&loaded, source.c_str(), params.width, params.height, 3)) {
                            error_message = std::string("failed to load MiniMax-H3 ") + what + " for segment " +
                                            std::to_string(segment);
                            return false;
                        }
                        owner.reset(loaded);
                        return true;
                    }
                    if (!decode_base64_image(source, 3, params.width, params.height, owner)) {
                        error_message = std::string("failed to decode MiniMax-H3 ") + what + " for segment " +
                                        std::to_string(segment);
                        return false;
                    }
                    return true;
                };
                if (!decode_anchor(job.h3_segment_init_images[segment], plan.init_owner, "first_frame") ||
                    !decode_anchor(job.h3_segment_end_images[segment], plan.end_owner, "last_frame")) {
                    generated = false;
                    break;
                }
                plan.params.init_image = plan.init_owner.get();
                plan.params.end_image  = plan.end_owner.get();

                // References in scope for THIS shot, in request order. The flat arrays are rebuilt
                // per shot because the C API has no "applies everywhere" encoding, and because a
                // shot that scopes every reference out must reach the engine in exactly the state
                // a request with no references arrives in -- null pointers and a zero count --
                // rather than in a special empty-array state nothing else ever produces.
                for (const H3StagedReference& staged : h3_staged_refs) {
                    const MiniMaxH3JobReference& reference = *staged.source;
                    if (reference.scoped &&
                        std::find(reference.segments.begin(), reference.segments.end(),
                                  static_cast<int>(segment)) == reference.segments.end()) {
                        continue;
                    }
                    plan.ref_kinds.push_back(staged.kind);
                    plan.ref_frame_counts.push_back(staged.frame_count);
                    for (int frame = 0; frame < staged.frame_count; ++frame) {
                        plan.ref_frames.push_back(h3_frames[staged.frame_offset + static_cast<size_t>(frame)]);
                    }
                    // The counts arrays are per REFERENCE, not per video reference: an image or
                    // audio entry contributes a zero so index i always means reference i.
                    plan.ref_conditioner_frame_counts.push_back(
                        static_cast<int>(reference.conditioner_frame_indices.size()));
                    plan.ref_conditioner_frames.insert(plan.ref_conditioner_frames.end(),
                                                       reference.conditioner_frame_indices.begin(),
                                                       reference.conditioner_frame_indices.end());
                    plan.ref_block_timestamp_counts.push_back(static_cast<int>(staged.block_timestamps.size()));
                    plan.ref_block_timestamps.insert(plan.ref_block_timestamps.end(),
                                                     staged.block_timestamps.begin(),
                                                     staged.block_timestamps.end());
                    plan.ref_audio_paths.push_back(reference.audio_path.empty() ? nullptr
                                                                                : reference.audio_path.c_str());
                }
                if (!plan.ref_kinds.empty()) {
                    plan.params.minimax_h3_ref_kinds  = plan.ref_kinds.data();
                    plan.params.minimax_h3_ref_frames = plan.ref_frames.empty() ? nullptr
                                                                                : plan.ref_frames.data();
                    plan.params.minimax_h3_ref_frame_counts = plan.ref_frame_counts.data();
                    plan.params.minimax_h3_ref_conditioner_frames =
                        plan.ref_conditioner_frames.empty() ? nullptr : plan.ref_conditioner_frames.data();
                    plan.params.minimax_h3_ref_conditioner_frame_counts =
                        plan.ref_conditioner_frame_counts.data();
                    plan.params.minimax_h3_ref_block_timestamps =
                        plan.ref_block_timestamps.empty() ? nullptr : plan.ref_block_timestamps.data();
                    plan.params.minimax_h3_ref_block_timestamp_counts =
                        plan.ref_block_timestamp_counts.data();
                    plan.params.minimax_h3_ref_audio_paths = plan.ref_audio_paths.data();
                    plan.params.minimax_h3_refs_size       = static_cast<int>(plan.ref_kinds.size());
                }
            }

            // ── PHASE 1: every prompt, one text-encoder residency ────────────────────────────
            static const bool h3_staged_te_enabled = [] {
                const char* value = getenv("MINIMAX_H3_STAGED_TE");
                return value == nullptr || value[0] != '0';
            }();
            bool h3_staged_te = false;
            if (generated && h3_staged_te_enabled) {
                const auto staging_started = std::chrono::steady_clock::now();
                h3_staged_te                     = true;
                sd_ctx_minimax_h3_clear_staged_conditioning(runtime.sd_ctx);
                for (size_t segment = 0; segment < segment_count; ++segment) {
                    if (!sd_ctx_minimax_h3_stage_conditioning(runtime.sd_ctx, &plans[segment].params,
                                                              static_cast<int>(segment))) {
                        h3_staged_te = false;
                        break;
                    }
                }
                // ── PHASE 2: hand the card to the DiT ────────────────────────────────────────
                //
                // UNCONDITIONAL, including on the failure path. The staging pass pins the text
                // encoder resident (that is what makes it one residency instead of N), so skipping
                // this after a partial pass would carry ~14.5 GB into the DiT phase on a card that
                // cannot hold both.
                sd_ctx_minimax_h3_reclaim_text_encoder(runtime.sd_ctx);
                if (h3_staged_te) {
                    const double staging_seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - staging_started).count();
                    LOG_INFO("MiniMax-H3: encoded %zu shot prompt(s) in one text-encoder residency (%.2fs)",
                             segment_count, staging_seconds);
                } else {
                    // Never fatal. Dropping the cache makes every lookup miss, which is exactly
                    // the un-staged behaviour -- one swap per shot, same output.
                    sd_ctx_minimax_h3_clear_staged_conditioning(runtime.sd_ctx);
                    LOG_WARN("MiniMax-H3: could not stage the shot prompts up front; each shot will "
                             "encode its own, costing a text-encoder/DiT swap per shot");
                }
            }

            // ── PHASE 3: render every shot against the resident DiT ──────────────────────────
            //
            // FL2VA serves t2va + fl2va; Ref2VA is a separate checkpoint. Select it from the
            // conditioning-derived task rather than trusting a caller-controlled model name.
            // The staged conditioner cache survives a DiT-only swap, so mixed timelines still
            // pay one text-encoder residency.
            const auto h3_variants = runtime_diffusion_model_variants(runtime);
            std::string h3_active_variant =
                runtime.gpu_sharing != nullptr ? runtime.gpu_sharing->loaded_variant : "base";
            const auto swap_h3_variant = [&](const std::string& wanted) {
                if (wanted == h3_active_variant) {
                    return true;
                }
                const auto variant = h3_variants.find(wanted);
                if (variant == h3_variants.end()) {
                    error_message = wanted == "ref2va"
                                        ? "MiniMax-H3 reference conditioning requires a configured "
                                          "'ref2va' diffusion-model variant"
                                        : "unknown MiniMax-H3 diffusion-model variant '" + wanted + "'";
                    return false;
                }
                if (!sd_ctx_swap_diffusion_model(runtime.sd_ctx, variant->second.c_str())) {
                    error_message = "could not load MiniMax-H3 diffusion-model variant '" + wanted + "'";
                    return false;
                }
                h3_active_variant = wanted;
                if (runtime.gpu_sharing != nullptr) {
                    runtime.gpu_sharing->loaded_variant = wanted;
                }
                return true;
            };
            for (size_t segment = 0; segment < segment_count && generated; ++segment) {
                if (job.cancel_requested) {
                    error_message = "job cancelled by client";
                    generated     = false;
                    break;
                }
                const std::string wanted_variant =
                    segment < job.h3_segment_tasks.size() && job.h3_segment_tasks[segment] == "ref2va"
                        ? "ref2va"
                        : "base";
                if (!swap_h3_variant(wanted_variant)) {
                    generated = false;
                    break;
                }
                // -1 disables the lookup outright, so an un-staged run cannot half-hit.
                sd_ctx_minimax_h3_use_staged_conditioning(runtime.sd_ctx,
                                                          h3_staged_te ? static_cast<int>(segment) : -1);

                sd_image_t* segment_frames = nullptr;
                int segment_frame_count    = 0;
                sd_audio_t* segment_audio  = nullptr;
                const bool segment_ok = generate_video(runtime.sd_ctx, &plans[segment].params, &segment_frames,
                                                       &segment_frame_count, &segment_audio);
                SDImageVec segment_results;
                if (!segment_ok) {
                    segment_frames = nullptr;
                }
                segment_results.adopt(segment_frames, segment_frame_count);
                if (!segment_ok || segment_results.empty()) {
                    free_sd_audio(segment_audio);
                    if (error_message.empty()) {
                        error_message = "MiniMax-H3 shot " + std::to_string(segment + 1) + " of " +
                                        std::to_string(segment_count) + " returned no results";
                    }
                    generated = false;
                    break;
                }
                if (runtime.gpu_sharing != nullptr) {
                    runtime.gpu_sharing->diffusion_loaded.store(true);
                }

                // One partial per shot, published under the same file name and the same URL LTX
                // uses, so koblem's progressive delivery needs no fork.
                if (write_h3_segment_previews) {
                    segment_writer.enqueue(static_cast<int>(segment), 4, segment_results.data(),
                                           segment_results.count(), segment_audio);
                }

                if (segment_audio != nullptr && segment_audio->data != nullptr &&
                    segment_audio->channels > 0 && segment_audio->sample_count > 0) {
                    if (h3_audio_channels == 0) {
                        h3_audio_rate     = segment_audio->sample_rate;
                        h3_audio_channels = segment_audio->channels;
                    }
                    if (segment_audio->sample_rate == h3_audio_rate &&
                        segment_audio->channels == h3_audio_channels) {
                        const size_t total = static_cast<size_t>(segment_audio->sample_count) *
                                             segment_audio->channels;
                        h3_audio_samples.insert(h3_audio_samples.end(), segment_audio->data,
                                                segment_audio->data + total);
                    } else {
                        // Every shot runs the same audio VAE at the same rate, so this cannot
                        // happen without a real bug. Say so rather than splicing mismatched PCM
                        // into the deliverable, which would desync everything after it.
                        LOG_WARN("MiniMax-H3 shot %zu returned %u Hz x%u audio, expected %u Hz x%u; dropping it",
                                 segment + 1, segment_audio->sample_rate, segment_audio->channels,
                                 h3_audio_rate, h3_audio_channels);
                    }
                }
                free_sd_audio(segment_audio);

                if (h3_stream) {
                    // Hand each frame to the encoder and free it immediately: peak frame memory
                    // stays at one shot rather than the whole timeline, which is what makes a
                    // 35-shot Director run survivable.
                    for (int frame = 0; frame < segment_results.count(); ++frame) {
                        stream_encoder.append(segment_results[frame]);
                    }
                } else {
                    for (int frame = 0; frame < segment_results.count(); ++frame) {
                        results.push_back(segment_results[frame]);
                        segment_results[frame].data = nullptr;  // ownership moved to `results`
                    }
                }
            }
            // A request ending on Ref2VA must not leak that checkpoint into the next t2va/fl2va
            // request. Restore even after a failed segment; if restoration itself fails, the
            // worker is no longer trustworthy and the job must fail.
            if (h3_active_variant != "base" && !swap_h3_variant("base")) {
                generated = false;
            }
            // The cache is CONTEXT state, not job state: leaving it behind would condition the
            // next job's shot i on this job's shot i, and the render would look completely normal.
            sd_ctx_minimax_h3_clear_staged_conditioning(runtime.sd_ctx);

            results_already_owned = true;
            h3_incomplete         = !generated;
            if (generated && !h3_audio_samples.empty() && h3_audio_channels > 0) {
                auto* stitched = static_cast<sd_audio_t*>(calloc(1, sizeof(sd_audio_t)));
                float* pcm     = static_cast<float*>(malloc(h3_audio_samples.size() * sizeof(float)));
                if (stitched != nullptr && pcm != nullptr) {
                    memcpy(pcm, h3_audio_samples.data(), h3_audio_samples.size() * sizeof(float));
                    stitched->sample_rate  = h3_audio_rate;
                    stitched->channels     = h3_audio_channels;
                    stitched->sample_count = h3_audio_samples.size() / h3_audio_channels;
                    stitched->data         = pcm;
                    generated_audio        = stitched;
                } else {
                    free(pcm);
                    free(stitched);
                    LOG_WARN("MiniMax-H3: could not allocate the stitched soundtrack");
                }
            }
        }
        if (!generated) {
            raw_results = nullptr;
        } else if (runtime.gpu_sharing != nullptr) {
            runtime.gpu_sharing->diffusion_loaded.store(true);
        }
        if (!results_already_owned) {
            results.adopt(raw_results, num_results);
        } else {
            free(raw_results);
        }
    }
    if (write_segment_previews) {
        segment_writer.finish();
    }

    if (h3_incomplete) {
        free_sd_audio(generated_audio);
        if (error_message.empty()) {
            error_message = "MiniMax-H3 chain did not complete";
        }
        return false;
    }

    // When the chain streamed, the frames were encoded and freed as they arrived, so `results`
    // is deliberately empty and the frame count comes from the encoder.
    const bool streamed = stream_encoder.active() && stream_encoder.frames() > 0;
    num_results = streamed ? stream_encoder.frames() : results.count();
    if (num_results <= 0) {
        free_sd_audio(generated_audio);
        // Only when nothing more specific was recorded. A render that failed for a KNOWN reason
        // (a LoRA lease that could not be restored, an unsupported H3 conditioning shape) already
        // set error_message; overwriting it with the generic line replaced a diagnosis with a
        // symptom, and the diagnosis is the only part the caller cannot re-derive.
        if (error_message.empty()) {
            error_message = "generate_video returned no results";
        }
        return false;
    }

    std::vector<uint8_t> video_bytes =
        streamed ? stream_encoder.finalize(generated_audio, job.vid_gen.output_compression)
                 : create_video_from_sd_images_to_vector(job.vid_gen.output_format,
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
