// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "worker_session.h"

namespace fs = std::filesystem;

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
        int                     seg   = 0;
        int                     stage = 0;  // 0 = final seg_<i>.webm; >0 = intermediate seg_<i>_stage<k>.webm
        std::vector<sd_image_t> frames;     // owned deep copies, freed after encode
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
                // stage 0 = the final full-res per-segment webm (seg_<i>.webm, existing behaviour);
                // stage>0 = an intermediate progressive-preview webm (seg_<i>_stage<k>.webm).
                std::string path  = t.stage > 0
                                        ? dir + "/seg_" + std::to_string(t.seg) + "_stage" + std::to_string(t.stage) + ".webm"
                                        : dir + "/seg_" + std::to_string(t.seg) + ".webm";
                auto        bytes = create_video_from_sd_images_to_vector(
                    "webm", t.frames.data(), (int)t.frames.size(), fps, quality, nullptr);
                if (!bytes.empty()) {
                    // Publish atomically (mirrors the final.webm write below): a status poller /
                    // segment fetch must never observe a half-written seg webm, so encode into
                    // seg_<i>.webm.tmp and rename() it into place once the bytes are fully flushed.
                    std::string     tmp_path = path + ".tmp";
                    std::error_code ec;
                    fs::remove(tmp_path, ec);
                    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
                    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
                    out.close();
                    fs::rename(tmp_path, path, ec);
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

    // stage 0 = the final per-segment webm (on_segment); stage>0 = an intermediate progressive
    // preview (on_stage). Both share this one bounded writer: back-pressure keeps at most ~one
    // segment's decoded frames in flight, so a mid-segment stage preview and the final segment webm
    // never pile up. Copies the frames (the core frees the originals right after the callback).
    void enqueue(int seg, const sd_image_t* fr, int n, int stage = 0) {
        {  // back-pressure: wait until the prior task has drained
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
            q.push_back(Task{seg, stage, std::move(copies)});
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

// Progressive UPSCALE-STAGE preview (opt-in emit_stages). Fired from generate_video_ex mid-segment
// after an intermediate stage's latent is VAE-decoded: stage_scale 1 = base (fast, low-res), 2 =
// STAGE-0 refine. Banked to seg_<seg>_stage<scale>.webm off the GPU thread via the shared writer, so
// the encode is CPU/background and never stalls the render. The final full-res webm still lands via
// ltx_seg_webm_cb (seg_<seg>.webm). width/height are informational (the frames already carry them).
void ltx_stage_webm_cb(int seg, int stage_scale, int width, int height,
                       const sd_image_t* frames, int n, void* user) {
    (void)width;
    (void)height;
    static_cast<SegWebmWriter*>(user)->enqueue(seg, frames, n, stage_scale);
}

// Kept alive by run_vid_chain_job while generate_video_chain is executing.
// The core invokes this only at a segment boundary, after the previous segment
// released its GPU-only buffers.
struct ChainDitLease {
    ServerRuntime*                  runtime = nullptr;
    const std::vector<std::string>* variants = nullptr;
    std::string                     error;
};

bool ltx_chain_dit_lease_cb(int seg, void* user) {
    auto* lease = static_cast<ChainDitLease*>(user);
    if (lease == nullptr || lease->runtime == nullptr || lease->variants == nullptr ||
        seg < 0 || static_cast<size_t>(seg) >= lease->variants->size()) {
        return false;
    }
    ServerRuntime& runtime = *lease->runtime;
    ModelSwapState* swap = runtime.model_swap;
    const std::string& want = (*lease->variants)[seg];
    if (swap == nullptr || want.empty()) {
        sd_ctx_set_diffusion_model_residency_id(runtime.sd_ctx, want.c_str());
        return true;
    }
    if (!swap->loaded.load() || swap->loaded_variant != want) {
        const auto it = swap->variants.find(want);
        if (it == swap->variants.end() || it->second.empty()) {
            lease->error = "no diffusion model path for chain variant '" + want + "'";
            return false;
        }
        LOG_INFO("LTXAV_DIT_LEASE: segment %d switching DiT %s -> %s after outgoing GPU lease release",
                 seg + 1, swap->loaded_variant.c_str(), want.c_str());
        if (!sd_ctx_swap_diffusion_model(runtime.sd_ctx, it->second.c_str())) {
            swap->loaded.store(false);
            lease->error = "failed to load chain diffusion model variant '" + want + "'";
            return false;
        }
        swap->loaded_variant = want;
        swap->loaded.store(true);
    }
    sd_ctx_set_diffusion_model_residency_id(runtime.sd_ctx, want.c_str());
    LOG_INFO("LTXAV_DIT_LEASE: segment %d active_dit_models=[%s]", seg + 1, want.c_str());
    return true;
}

// Windowed-streaming finalize: the chain hands us the now-final leading frames IN ORDER. Encode
// (spool) each and free its pixels immediately, so the chain never materialises the whole timeline.
// Synchronous by contract — the chain frees nothing here and considers these frames gone on return.
void ltx_stream_flush_cb(sd_image_t* frames, int n, void* user) {
    auto* w = static_cast<IncrementalWebmWriter*>(user);
    for (int i = 0; i < n; ++i) {
        w->append_video_frame(frames[i]);  // records an internal failure flag on error
        free(frames[i].data);
        frames[i].data = nullptr;
    }
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

    // Progressive previews (opt-in emit_segments / emit_stages). List every banked preview webm so a
    // client can play each as it lands. Two shapes: seg_<n>.webm = the FINAL full-res shot (stage 4),
    // and seg_<n>_stage<k>.webm = an intermediate WITHIN-shot upscale preview (stage k: 1=base low-res,
    // 2=mid-res). Read straight off the shared job dir here on the status HTTP thread — the partials
    // NEVER ride the blocking render IPC channel. Present while Generating and once Completed; absent
    // when no preview webms exist (the byte-identical off path). Cheap directory scan; skips the
    // in-flight *.webm.tmp files (they don't end in ".webm").
    if (!job.job_dir.empty() && (job.status == AsyncJobStatus::Generating ||
                                 job.status == AsyncJobStatus::Completed)) {
        static const std::string kPrefix = "seg_";
        static const std::string kSuffix = ".webm";
        struct PartialEntry {
            int  seg;
            int  stage;
            bool staged;  // seg_<n>_stage<k>.webm vs the final seg_<n>.webm
        };
        std::vector<PartialEntry> entries;
        std::error_code           ec;
        for (auto it = fs::directory_iterator(job.job_dir, ec);
             !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec)) {
                continue;
            }
            const std::string name = it->path().filename().string();
            if (name.size() <= kPrefix.size() + kSuffix.size() ||
                name.compare(0, kPrefix.size(), kPrefix) != 0 ||
                name.compare(name.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0) {
                continue;  // not seg_*.webm (also drops the seg_*.webm.tmp in-flight files)
            }
            // core = the part between "seg_" and ".webm": "<n>" (final) or "<n>_stage<k>" (staged).
            const std::string core =
                name.substr(kPrefix.size(), name.size() - kPrefix.size() - kSuffix.size());
            const std::string kStage = "_stage";
            const size_t      sp     = core.find(kStage);
            std::string       seg_d, stage_d;
            bool              staged = false;
            if (sp != std::string::npos) {
                seg_d   = core.substr(0, sp);
                stage_d = core.substr(sp + kStage.size());
                staged  = true;
                if (stage_d.empty() || stage_d.find_first_not_of("0123456789") != std::string::npos) {
                    continue;
                }
            } else {
                seg_d = core;
            }
            if (seg_d.empty() || seg_d.find_first_not_of("0123456789") != std::string::npos) {
                continue;
            }
            entries.push_back(PartialEntry{std::stoi(seg_d),
                                           staged ? std::stoi(stage_d) : 4,  // final = 4x
                                           staged});
        }
        // Only add the "partials" key when at least one preview webm exists. With both flags off (and
        // no env override) none are written, so the key is absent and the status JSON is byte-identical
        // to today — never an empty "partials": [].
        if (!entries.empty()) {
            std::sort(entries.begin(), entries.end(), [](const PartialEntry& a, const PartialEntry& b) {
                return a.seg != b.seg ? a.seg < b.seg : a.stage < b.stage;
            });
            json partials = json::array();
            for (const auto& e : entries) {
                // Staged previews carry ?stage=<k>; the final shot keeps its plain URL (unchanged).
                std::string url = "/sdcpp/v1/jobs/" + job.id + "/segments/" + std::to_string(e.seg);
                if (e.staged) {
                    url += "?stage=" + std::to_string(e.stage);
                }
                partials.push_back({
                    {"segment_index", e.seg},
                    {"stage", e.stage},
                    {"url", url},
                });
            }
            result["partials"] = std::move(partials);
        }
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
    int effective_output_fps     = job.vid_gen.gen_params.fps;
    sd_audio_t* generated_audio = nullptr;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = nullptr;
        if (!generate_video(runtime.sd_ctx, &params, &raw_results, &num_results, &generated_audio,
                            &effective_output_fps)) {
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
                                                                             effective_output_fps,
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
    output_fps             = effective_output_fps;
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

    // Engine select: the wan route stamps "engine":"wan" so the SAME chain job path drives the
    // Wan2.2-VACE continuation (generate_wan_vace_chain) instead of the LTXAV chain. Everything
    // else — param parse, prompt list, job dir, per-window webm banking — is shared.
    const bool wan_mode = body.value("engine", std::string()) == "wan";

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

    // FLUX-style dual-DiT variant swap for the video chain (LTX/Wan). An explicit
    // top-level {"model":"base"|"edit"} selects which DiT must be resident before the
    // chain renders — the LTX lipdub relip path sends {"model":"edit"} to load the
    // merged lipdub GGUF (--diffusion-model-edit). Absent/"base" keeps the base DiT,
    // so every existing non-relip render is byte-identical (ensure_variant_loaded with
    // an empty target keeps the currently-loaded variant). This mirrors the FLUX image
    // path (parse in routes_sdcpp.cpp:392-406 -> ensure_variant_loaded at IMG_GEN_REQ /
    // async_jobs.cpp:650). It MUST run here — inside run_vid_chain_job, which executes in
    // the CUDA-owning process (the worker child's VIDGEN_CHAIN_REQ handler, or the
    // in-process path) — and BEFORE the generate_video_chain sd_ctx_mutex lock below,
    // because ensure_variant_loaded takes that same mutex internally. koblem calls
    // /v1/admin/unload before the edit render, so the swap re-forks/loads the edit DiT
    // fresh. Validation mirrors the FLUX route so the error text matches.
    std::string chain_default_variant;
    {
        std::string want_variant;  // empty = keep whatever DiT is loaded (base) — byte-identical
        if (body.contains("model") && body["model"].is_string()) {
            want_variant = body["model"].get<std::string>();
            // Validate against the registered variant map (base/edit + --diffusion-model-variants).
            if (!want_variant.empty() && runtime.model_swap) {
                const auto& vars = runtime.model_swap->variants;
                if (vars.find(want_variant) == vars.end()) {
                    std::string names;
                    for (const auto& kv : vars) {
                        if (!names.empty()) {
                            names += ", ";
                        }
                        names += '"' + kv.first + '"';
                    }
                    error_message = "unknown model variant \"" + want_variant +
                                    "\"; registered: " + (names.empty() ? "(none)" : names);
                    return false;
                }
            }
        }
        if (!ensure_variant_loaded(runtime, want_variant, error_message)) {
            return false;  // error_message set by ensure_variant_loaded
        }
        chain_default_variant = runtime.model_swap != nullptr
                                    ? runtime.model_swap->loaded_variant
                                    : want_variant;
    }

    int n_segments = std::max(1, gen_params.ltx_chain_segments);

    // A chain normally keeps its selected DiT. A segment may override it with a
    // registered compatible fused variant; every transition is validated before
    // rendering and leased at the safe boundary in the core chain loop.
    std::vector<std::string> segment_model_variants(n_segments, chain_default_variant);
    if (!wan_mode && body.contains("segments") && body["segments"].is_array()) {
        const auto& segs = body["segments"];
        for (int seg = 0; seg < n_segments && seg < (int)segs.size(); ++seg) {
            if (!segs[seg].is_object()) {
                continue;
            }
            const auto it = segs[seg].find("model");
            if (it == segs[seg].end() || !it->is_string() || it->get<std::string>().empty()) {
                continue;
            }
            const std::string candidate = it->get<std::string>();
            if (runtime.model_swap != nullptr &&
                runtime.model_swap->variants.find(candidate) == runtime.model_swap->variants.end()) {
                error_message = "unknown model variant \"" + candidate + "\" for chain segment " +
                                std::to_string(seg + 1);
                return false;
            }
            segment_model_variants[seg] = candidate;
        }
    }
    ChainDitLease chain_dit_lease{&runtime, &segment_model_variants, {}};

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
    // Engine-owned whole-timeline audio (supersedes the pre-sliced aud_<i>.wav when present):
    // `full` drives lip-sync, `track` is muxed as the deliverable. Both staged by the route.
    std::string audio_full    = body.value("chain_audio_full", std::string());
    std::string audio_track   = body.value("chain_audio_track", std::string());
    int         audio_offset_frames = body.value("chain_audio_offset_frames", 0);
    // Seam trim pin (frames dropped from each continuation's head). The caller MUST pad each
    // continuation's render by the same number for produced == requested visible length.
    // 0 = derive 8*K and pin it whenever the engine owns the audio; <0 = force content-adaptive.
    int         cont_seam_drop = body.value("cont_seam_drop_frames", 0);
    // Per-segment override: drop[i] = render_frames[i] - visible_frames[i]. Stating it per shot is
    // what makes produced == requested regardless of whether caller and engine classify a shot as
    // fresh the same way (they have diverged). A negative/short entry = let the engine decide.
    // Per-shot audio: each shot's own clip, staged by the route. `full` drives that shot's lip-sync,
    // `track` is its deliverable audio; both compose onto the whole-clip bed above.
    std::vector<std::string> seg_audio_full((size_t)n_segments);
    std::vector<std::string> seg_audio_track((size_t)n_segments);
    std::vector<const char*> seg_audio_full_ptrs;
    std::vector<const char*> seg_audio_track_ptrs;
    auto load_seg_audio = [&](const char* key, std::vector<std::string>& out) {
        bool any = false;
        if (body.contains(key) && body[key].is_array()) {
            const auto& arr = body[key];
            for (size_t i = 0; i < arr.size() && i < out.size(); ++i) {
                if (arr[i].is_string()) {
                    out[i] = arr[i].get<std::string>();
                    any    = any || !out[i].empty();
                }
            }
        }
        return any;
    };
    if (load_seg_audio("segment_audio_full", seg_audio_full)) {
        seg_audio_full_ptrs.reserve((size_t)n_segments);
        for (const auto& s : seg_audio_full) {
            seg_audio_full_ptrs.push_back(s.empty() ? nullptr : s.c_str());
        }
    }
    if (load_seg_audio("segment_audio_track", seg_audio_track)) {
        seg_audio_track_ptrs.reserve((size_t)n_segments);
        for (const auto& s : seg_audio_track) {
            seg_audio_track_ptrs.push_back(s.empty() ? nullptr : s.c_str());
        }
    }
    std::vector<int> seam_drops;
    if (body.contains("segment_seam_drop_frames") && body["segment_seam_drop_frames"].is_array()) {
        seam_drops.assign((size_t)n_segments, -1);
        const auto& arr = body["segment_seam_drop_frames"];
        for (size_t i = 0; i < arr.size() && i < seam_drops.size(); ++i) {
            if (arr[i].is_number_integer()) {
                seam_drops[i] = arr[i].get<int>();
            }
        }
    }
    std::string output_format = body.value("output_format", std::string("webm"));
    int         output_compression = body.value("output_compression", 90);

    // Durable artifact dir (per-segment latent/webm banking + final.webm) and the resume
    // point (skip segments already banked there). Both injected by the LTX /generate route.
    std::string job_dir     = body.value("ltx_job_dir", std::string());
    int         resume_from = body.value("resume_from", 0);

    sd_vid_gen_params_t base = gen_params.to_sd_vid_gen_params_t();
    // Character identity reference is intentionally separate from init_image/control_frames.
    // It is decoded once here and generate_video_chain VAE-encodes it once for all segments.
    SDImageOwner character_reference_owner;
    if (!wan_mode && body.contains("character_reference") && body["character_reference"].is_string() &&
        !body["character_reference"].get<std::string>().empty()) {
        if (!decode_base64_image(body["character_reference"].get<std::string>(), 3, base.width, base.height,
                                 character_reference_owner)) {
            error_message = "failed to decode character reference image";
            return false;
        }
        base.character_reference = character_reference_owner.get();
    }
    // Generic V2V mode selector for control_frames (see sd_vid_gen_params_t.v2v_mode):
    //   0 = lipdub relip (default), 1 = SDEdit denoise, 2 = guide-edit (Director-2 source-as-guide).
    // Accept an explicit integer "v2v_mode" (0/1/2); fall back to the legacy boolean "v2v" (true=1
    // SDEdit). Applies to every segment by default; per-segment segments[i].v2v_mode overrides below.
    if (body.contains("v2v_mode") && body["v2v_mode"].is_number_integer()) {
        int m         = body["v2v_mode"].get<int>();
        base.v2v_mode = (m == 1 || m == 2) ? m : 0;
    } else {
        base.v2v_mode = body.value("v2v", false) ? 1 : 0;
    }
    // Guide-edit (v2v_mode==2) LTXDirectorGuide scale ("edit strength"): 1.0 = hold the source scene
    // hard, ~0.5 = looser guide / bigger edit. Global to the chain in v1. Ignored for modes 0/1.
    base.v2v_guide_strength = body.value("v2v_guide_strength", 1.0f);
    // Guide-edit SOURCE (mode 2), LATENT-IN (preferred when we own the source, e.g. editing a shot we
    // rendered): absolute path to a banked diffusion VIDEO latent (a prior job's seg_<i>.bin). When
    // set, the guide loads straight from this latent — no pixel re-encode. Global/seg-0 here;
    // per-segment via segments[i].v2v_source_latent_path below. Falls back to control_frames (pixel
    // encode) when absent. (Owner string must outlive generate_video_chain — declared in this scope.)
    std::string v2v_guide_latent_path = body.value("v2v_guide_latent_path", std::string());
    base.v2v_guide_latent_path = v2v_guide_latent_path.empty() ? nullptr : v2v_guide_latent_path.c_str();
    if (base.v2v_mode == 2) {
        LOG_INFO("run_vid_chain_job: guide-edit v2v (mode 2), guide strength %.2f, source=%s",
                 base.v2v_guide_strength,
                 base.v2v_guide_latent_path ? "latent-in" : "pixel-encode(control_frames)");
    }

    // Chained V2V relip accepts one contiguous control_frames array and partitions it into
    // per-segment source windows. Default is exactly video_frames per segment; callers may pass
    // relip_control_frame_counts for an explicit partition (useful for a short final window).
    // Supplying enough frames is intentionally required rather than silently reusing segment 0:
    // replaying the first source clip on every window is not a V2V chain.
    std::vector<sd_image_t*> relip_control_starts;
    std::vector<int>         relip_control_counts;
    if (n_segments > 1 && !gen_params.control_frame_views.empty()) {
        std::vector<int> requested_counts;
        if (body.contains("relip_control_frame_counts") && body["relip_control_frame_counts"].is_array()) {
            for (const auto& v : body["relip_control_frame_counts"]) {
                if (!v.is_number_integer() || v.get<int>() <= 0) {
                    error_message = "relip_control_frame_counts must contain positive integers";
                    return false;
                }
                requested_counts.push_back(v.get<int>());
            }
            if ((int)requested_counts.size() != n_segments) {
                error_message = "relip_control_frame_counts must have one entry per segment";
                return false;
            }
        } else {
            requested_counts.assign(n_segments, gen_params.video_frames);
        }
        size_t offset = 0;
        relip_control_starts.reserve(n_segments);
        relip_control_counts.reserve(n_segments);
        for (int count : requested_counts) {
            if (offset + (size_t)count > gen_params.control_frame_views.size()) {
                error_message = "chained relip needs distinct control_frames for every segment";
                return false;
            }
            relip_control_starts.push_back(gen_params.control_frame_views.data() + offset);
            relip_control_counts.push_back(count);
            offset += (size_t)count;
        }
        LOG_INFO("run_vid_chain_job: chained V2V relip uses %d distinct source windows (%zu control frames)",
                 n_segments, offset);
    }

    // Director multi-scene: per-segment i2v scene images from body["segments"][i]["init_image"]
    // (inline base64, resized to the render grid like the top-level init_image). A seg>0 with
    // one starts a FRESH scene there; seg-0's is the opener and already rides base.init_image.
    // Owners must outlive generate_video_chain below; seg_img_values is sized once so the
    // pointers stay stable.
    std::vector<SDImageOwner> seg_img_owners(n_segments);
    std::vector<sd_image_t>   seg_img_values(n_segments);          // value-init: data=nullptr
    std::vector<sd_image_t*>  seg_img_ptrs(n_segments, nullptr);
    bool                      any_seg_img = false;
    if (!wan_mode && body.contains("segments") && body["segments"].is_array()) {
        const auto& segs = body["segments"];
        for (int seg = 1; seg < n_segments && seg < (int)segs.size(); ++seg) {
            if (!segs[seg].is_object()) {
                continue;
            }
            auto it = segs[seg].find("init_image");
            if (it == segs[seg].end() || !it->is_string() || it->get<std::string>().empty()) {
                continue;
            }
            if (!decode_base64_image(it->get<std::string>(), 3, base.width, base.height,
                                     seg_img_owners[seg])) {
                error_message = "failed to decode segment " + std::to_string(seg + 1) + " scene image";
                return false;
            }
            seg_img_values[seg] = seg_img_owners[seg].get();
            seg_img_ptrs[seg]   = &seg_img_values[seg];
            any_seg_img         = true;
            LOG_INFO("run_vid_chain_job: segment %d scene image %ux%u (mid-chain i2v scene cut)",
                     seg + 1, seg_img_values[seg].width, seg_img_values[seg].height);
        }
    }

    // Director keyframes: per-segment frame-pinned images from body["segments"][i]["keyframes"] =
    // [{"image":"<b64>","frame":<int>}, ...] (inline base64, resized to the render grid). A shot
    // with keyframes renders fresh (i2v keyframe branch). Owners outlive generate_video_chain;
    // the value/index vectors are sized once so the per-segment array pointers stay stable.
    std::vector<std::vector<SDImageOwner>> kf_owners(n_segments);
    std::vector<std::vector<sd_image_t>>   kf_values(n_segments);
    std::vector<std::vector<int>>          kf_indices(n_segments);
    std::vector<sd_image_t*>               kf_img_ptrs(n_segments, nullptr);
    std::vector<const int*>                kf_idx_ptrs(n_segments, nullptr);
    std::vector<int>                       kf_counts(n_segments, 0);
    bool                                   any_kf = false;
    if (!wan_mode && body.contains("segments") && body["segments"].is_array()) {
        const auto& segs = body["segments"];
        for (int seg = 0; seg < n_segments && seg < (int)segs.size(); ++seg) {
            if (!segs[seg].is_object()) {
                continue;
            }
            auto it = segs[seg].find("keyframes");
            if (it == segs[seg].end() || !it->is_array() || it->empty()) {
                continue;
            }
            for (const auto& kf : *it) {
                if (!kf.is_object()) {
                    continue;
                }
                auto img_it = kf.find("image");
                if (img_it == kf.end() || !img_it->is_string() || img_it->get<std::string>().empty()) {
                    continue;
                }
                SDImageOwner owner;
                if (!decode_base64_image(img_it->get<std::string>(), 3, base.width, base.height, owner)) {
                    error_message = "failed to decode segment " + std::to_string(seg + 1) + " keyframe image";
                    return false;
                }
                kf_owners[seg].push_back(std::move(owner));
                kf_indices[seg].push_back(kf.value("frame", 0));
            }
            int cnt = (int)kf_owners[seg].size();
            if (cnt > 0) {
                kf_values[seg].resize(cnt);
                for (int k = 0; k < cnt; ++k) {
                    kf_values[seg][k] = kf_owners[seg][k].get();
                }
                kf_img_ptrs[seg] = kf_values[seg].data();
                kf_idx_ptrs[seg] = kf_indices[seg].data();
                kf_counts[seg]   = cnt;
                any_kf           = true;
                LOG_INFO("run_vid_chain_job: segment %d has %d keyframe pin(s)", seg + 1, cnt);
            }
        }
    }

    // Director variable-length: per-segment frame count from body["segments"][i]["frames"].
    // A 0/absent entry falls back to the uniform gen_params.video_frames. Any positive entry
    // switches this render to variable-length (segment_video_frames wired below).
    std::vector<int> seg_frames(n_segments, 0);
    bool             any_seg_frames = false;
    // Per-segment TEXT-ONLY SCENE CUT. segments[i].scene_cut=true makes that seg>0 shot a fresh
    // scene from its prompt alone (no image, no continuation). Off/absent = ordinary continuation.
    std::vector<int> seg_scene(n_segments, 0);
    bool             any_seg_scene = false;
    // Per-segment generic-V2V mode. Default = base.v2v_mode; override via segments[i].v2v_mode
    // (integer 0/1/2) or the legacy boolean segments[i].v2v (true=1 SDEdit).
    std::vector<int> seg_v2v(n_segments, base.v2v_mode);
    bool             any_seg_v2v = false;
    // Per-segment guide-edit LATENT-IN source: absolute path to that segment's banked seg_<i>.bin
    // (from segments[i].v2v_source_latent_path). Owners (strings) outlive generate_video_chain; the
    // const char* array is stable-sized. An entry makes that segment a latent-in guide-edit.
    std::vector<std::string> seg_guide_paths(n_segments);
    std::vector<const char*> seg_guide_ptrs(n_segments, nullptr);
    bool                     any_seg_guide_lat = false;
    if (!wan_mode && body.contains("segments") && body["segments"].is_array()) {
        const auto& segs = body["segments"];
        for (int seg = 0; seg < n_segments && seg < (int)segs.size(); ++seg) {
            if (!segs[seg].is_object()) {
                continue;
            }
            auto it = segs[seg].find("frames");
            if (it != segs[seg].end() && it->is_number_integer() && it->get<int>() > 0) {
                seg_frames[seg] = it->get<int>();
                any_seg_frames  = true;
                LOG_INFO("run_vid_chain_job: segment %d frame count %d (variable-length)",
                         seg + 1, seg_frames[seg]);
            }
            auto scit = segs[seg].find("scene_cut");
            if (scit != segs[seg].end() && scit->is_boolean() && scit->get<bool>()) {
                seg_scene[seg] = 1;
                any_seg_scene  = true;
                LOG_INFO("run_vid_chain_job: segment %d TEXT SCENE CUT (fresh from prompt)", seg + 1);
            }
            bool seg_set_v2v = false;
            auto mit         = segs[seg].find("v2v_mode");
            if (mit != segs[seg].end() && mit->is_number_integer()) {
                int m        = mit->get<int>();
                seg_v2v[seg] = (m == 1 || m == 2) ? m : 0;
                seg_set_v2v  = true;
            } else {
                auto vit = segs[seg].find("v2v");
                if (vit != segs[seg].end() && vit->is_boolean()) {
                    seg_v2v[seg] = vit->get<bool>() ? 1 : 0;
                    seg_set_v2v  = true;
                }
            }
            if (seg_set_v2v) {
                any_seg_v2v   = true;
                const char* d = seg_v2v[seg] == 1 ? "SDEdit" : seg_v2v[seg] == 2 ? "guide-edit" : "relip";
                LOG_INFO("run_vid_chain_job: segment %d v2v mode = %d (%s)", seg + 1, seg_v2v[seg], d);
            }
            auto git = segs[seg].find("v2v_source_latent_path");
            if (git != segs[seg].end() && git->is_string() && !git->get<std::string>().empty()) {
                seg_guide_paths[seg] = git->get<std::string>();
                seg_guide_ptrs[seg]  = seg_guide_paths[seg].c_str();
                any_seg_guide_lat    = true;
                // A latent-in guide segment is a guide-edit (mode 2) even without control_frames.
                if (!seg_set_v2v) {
                    seg_v2v[seg] = 2;
                    any_seg_v2v  = true;
                }
                LOG_INFO("run_vid_chain_job: segment %d guide-edit LATENT-IN %s", seg + 1, seg_guide_paths[seg].c_str());
            }
        }
    }

    sd_vid_chain_params_t chain = {};
    // RETAKE (bidirectional single-segment splice): body "retake_segment" (a FILTERED segment index
    // into the banked seg_<i>.bin files) re-renders ONLY that shot, start-pinned by seg_{N-1}'s tail
    // and end-pinned by seg_{N+1}'s head, then splices the banked tail. -1 (default) = ordinary
    // chain/resume. Aggregate {} would give 0 (a valid index), so this MUST be set explicitly.
    chain.retake_segment     = body.value("retake_segment", -1);
    chain.n_segments         = n_segments;
    chain.cont_latent_frames = std::max(1, gen_params.cont_latent_take);
    chain.segment_prompts    = prompt_ptrs.data();
    chain.segment_video_frames = any_seg_frames ? seg_frames.data() : nullptr;
    chain.segment_scene_cut    = any_seg_scene ? seg_scene.data() : nullptr;
    chain.segment_v2v_mode     = any_seg_v2v ? seg_v2v.data() : nullptr;
    chain.segment_v2v_guide_latent_paths = any_seg_guide_lat ? seg_guide_ptrs.data() : nullptr;
    chain.chain_audio_dir    = audio_dir.empty() ? nullptr : audio_dir.c_str();
    chain.chain_audio_full   = audio_full.empty() ? nullptr : audio_full.c_str();
    chain.chain_audio_track  = audio_track.empty() ? nullptr : audio_track.c_str();
    chain.chain_audio_offset_frames = audio_offset_frames;
    chain.cont_seam_drop_frames     = cont_seam_drop;
    chain.segment_seam_drop_frames  = seam_drops.empty() ? nullptr : seam_drops.data();
    chain.segment_audio_full        = seg_audio_full_ptrs.empty() ? nullptr : seg_audio_full_ptrs.data();
    chain.segment_audio_track       = seg_audio_track_ptrs.empty() ? nullptr : seg_audio_track_ptrs.data();
    chain.save_dir           = job_dir.empty() ? nullptr : job_dir.c_str();
    chain.resume_from        = std::max(0, resume_from);
    chain.segment_control_frames = relip_control_starts.empty() ? nullptr : relip_control_starts.data();
    chain.segment_control_frame_counts = relip_control_counts.empty() ? nullptr : relip_control_counts.data();
    chain.segment_init_images = any_seg_img ? seg_img_ptrs.data() : nullptr;
    chain.segment_keyframes        = any_kf ? kf_img_ptrs.data() : nullptr;
    chain.segment_keyframe_indices = any_kf ? kf_idx_ptrs.data() : nullptr;
    chain.segment_keyframe_counts  = any_kf ? kf_counts.data() : nullptr;
    chain.before_segment      = &ltx_chain_dit_lease_cb;
    chain.before_segment_user = &chain_dit_lease;

    // Bank a viewable per-segment webm as each segment lands (off the GPU thread) for
    // progressive delivery — a client plays each shot while the next renders. OPT-IN: only when
    // the request set chain["emit_segments"]=true, OR the LTX_BANK_SEG_WEBM env override forces
    // it on. When neither, on_segment stays null and frames flow only through the normal
    // stitch/streaming path (byte-identical: nothing extra is encoded and no partials appear).
    // (resume needs only the seg_<i>.bin latents, banked regardless of this flag.)
    // emit_stages (opt-in) additionally banks intermediate progressive-preview webms
    // (seg_<i>_stage<k>.webm — a fast low-res base, then the mid-res refine) WITHIN each shot so the
    // client sees a rough preview ASAP that sharpens. It implies the per-segment final webm too (the
    // stage-4 = seg_<i>.webm), so a full 1->2->4 ladder is available. Byte-identical when off.
    const bool emit_segments = body.value("emit_segments", false);
    const bool emit_stages   = body.value("emit_stages", false);

    // LTX native temporal frame-upscaler (LTXAV_TEMPORAL_UPSCALE): generate_video_ex turns the
    // decoded video into 2T-1 frames but leaves the base fps request untouched, so the container
    // must be muxed at 2x fps to keep real-time playback (frames double + fps double => duration
    // unchanged; the ORIGINAL audio, already that duration, still lines up 1:1 and is NOT touched
    // here). The upscaler only fires on a single-segment LTX chain — multi-segment chains are
    // continuation-gated in the core and skip it, so their frame count is NOT doubled and their
    // fps must stay put. No-op (byte-identical) when the flag is unset.
    int effective_fps = gen_params.fps;
    {
        const char* tu     = getenv("LTXAV_TEMPORAL_UPSCALE");
        const bool  tu_on  = tu != nullptr && tu[0] != '\0' &&
                            std::string(tu) != "0" && std::string(tu) != "false";
        if (tu_on && !wan_mode && n_segments == 1) {
            effective_fps = std::max(1, gen_params.fps) * 2;
            LOG_INFO("LTX temporal upscale active (single-segment LTX chain): muxing container at 2x fps "
                     "(%d -> %d); audio passes through unchanged",
                     gen_params.fps, effective_fps);
        }
    }

    std::unique_ptr<SegWebmWriter> seg_writer;
    if (!job_dir.empty()) {
        const char* bank   = getenv("LTX_BANK_SEG_WEBM");
        const bool  env_on = (bank != nullptr) && (bank[0] != '0');
        const bool  on     = emit_segments || emit_stages || env_on;
        if (on) {
            seg_writer          = std::make_unique<SegWebmWriter>();
            seg_writer->dir     = job_dir;
            seg_writer->fps     = effective_fps;
            seg_writer->quality = output_compression;
            seg_writer->start();
            chain.on_segment      = &ltx_seg_webm_cb;
            chain.on_segment_user = seg_writer.get();
            if (emit_stages) {
                // Route the core's per-stage hook through the SAME bounded writer (base_params
                // propagates on_stage into every per-segment vp; the chain sets vp.stage_seg_index).
                base.emit_stages   = 1;
                base.on_stage      = &ltx_stage_webm_cb;
                base.on_stage_user = seg_writer.get();
            }
        }
    }

    // WINDOWED STREAMING FINALIZE. For webm chains with a durable job dir (the prod path — a 3.5-min
    // 1080p chain is ~31.5 GB of decoded frames), stream each finalized frame straight into the WebM
    // encoder instead of materialising the whole timeline. generate_video_chain keeps only a small
    // rolling window and flushes the rest through on_flush_frames; the writer spools encoded VP8 and
    // muxes the length-matched audio + publishes final.<fmt> atomically at finalize(). Byte-identical
    // to the whole-array path. Other formats / no job dir fall back to the legacy whole-array encode.
    std::string lc_format = output_format;
    std::transform(lc_format.begin(), lc_format.end(), lc_format.begin(),
                   [](unsigned char c) { return static_cast<char>(tolower(c)); });
    std::unique_ptr<IncrementalWebmWriter> stream_writer;
    bool streaming = false;
    // LTX chain only: generate_wan_vace_chain does not honour on_flush_frames (it returns the whole
    // array), and Wan windows are short — the 31.5 GB peak is the LTX long-chain problem.
    if (!wan_mode && lc_format == "webm" && !job_dir.empty()) {
        const fs::path final_path = fs::path(job_dir) / ("final." + output_format);
        stream_writer             = std::make_unique<IncrementalWebmWriter>();
        if (stream_writer->open(final_path.string(), effective_fps, output_compression)) {
            streaming                  = true;
            chain.on_flush_frames      = &ltx_stream_flush_cb;
            chain.on_flush_frames_user = stream_writer.get();
        } else {
            stream_writer.reset();  // spool open failed — degrade to whole-array (still correct)
        }
    }

    sd_image_t* frames      = nullptr;
    int         frame_count = 0;
    sd_audio_t* audio       = nullptr;
    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        bool chain_ok = wan_mode
                            ? generate_wan_vace_chain(runtime.sd_ctx, &base, &chain, &frames, &frame_count, &audio)
                            : generate_video_chain(runtime.sd_ctx, &base, &chain, &frames, &frame_count, &audio);
        if (seg_writer) {
            seg_writer->finish();  // drain pending seg webms before reporting outcome
        }
        if (!chain_ok) {
            // Streaming: stream_writer's dtor drops the spool + any half-written .tmp (nothing published).
            free_sd_audio(audio);
            error_message = chain_dit_lease.error.empty()
                                ? "generate_video_chain failed"
                                : chain_dit_lease.error;
            return false;
        }
    }

    if (streaming) {
        // Frames were streamed+freed incrementally; only metadata + audio came back.
        if (frame_count <= 0) {
            free_sd_audio(audio);
            error_message = "generate_video_chain produced no frames";
            return false;
        }
        stream_writer->set_audio(audio);
        free_sd_audio(audio);
        out_video = stream_writer->finalize();  // muxes spool + audio, publishes final.<fmt> atomically
        if (out_video.empty()) {
            error_message = "failed to encode generated video container";
            return false;
        }
        out_mime        = video_mime_type(output_format);
        out_frame_count = frame_count;
        out_fps         = effective_fps;
        return true;
    }

    if (frame_count <= 0 || frames == nullptr) {
        free_sd_audio(audio);
        error_message = "generate_video_chain produced no frames";
        return false;
    }

    // The final WebM encoder owns the chain's decoded pixels and releases each frame as
    // soon as VP8 has consumed it. The nulling contract makes this cleanup loop safe.
    out_video = create_video_from_sd_images_to_vector(output_format, frames, frame_count,
                                                      effective_fps, output_compression, audio, true);
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
        const fs::path final_path = fs::path(job_dir) / ("final." + output_format);
        const fs::path tmp_path   = final_path.string() + ".tmp";
        std::error_code ec;
        fs::remove(tmp_path, ec);
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            error_message = "failed to open final video artifact";
            return false;
        }
        out.write(reinterpret_cast<const char*>(out_video.data()),
                  static_cast<std::streamsize>(out_video.size()));
        if (!out) {
            error_message = "failed to write final video artifact";
            return false;
        }
        out.close();
        if (!out) {
            error_message = "failed to close final video artifact";
            return false;
        }
        fs::rename(tmp_path, final_path, ec);
        if (ec) {
            error_message = "failed to publish final video artifact: " + ec.message();
            return false;
        }
    }

    out_mime        = video_mime_type(output_format);
    out_frame_count = frame_count;
    out_fps         = effective_fps;
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

    // Resolve the gguf path for the wanted variant from the startup-built map (seeded with
    // base/edit + any --diffusion-model-variants). Fall back to base/edit for safety if the
    // map is somehow unpopulated (single-model in-process paths).
    std::string path;
    {
        auto it = swap->variants.find(want);
        if (it != swap->variants.end()) {
            path = it->second;
        } else if (want == "edit") {
            path = swap->edit_path;
        } else {
            path = swap->base_path;
        }
    }
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
                // FIX (back-to-back worker collision, 2026-07-07): reap the resident CUDA child
                // after each chain render so the NEXT render re-forks cold. The warm-reuse fast
                // path (worker_session ensure_loaded) keeps DiT/VAE resident across renders and
                // the success path never reaps it, so render N+1 stacks its offload buffers on
                // N's retained set and the OOM-killer SIGKILLs the child mid-hires — the observed
                // every-other-render "VIDGEN_CHAIN_RESP recv failed: peer closed cleanly". shutdown()
                // is idempotent (a dead child is a no-op). Cost: cold weight reload per chain render
                // (~seconds on a minutes-long render). Env-gated default-ON; LTX_REAP_CHILD_AFTER_RENDER=0
                // restores the (currently fatal) warm-reuse path. See WORKER-ISOLATION-BACKTOBACK-BUG.md.
                {
                    const char * reap = getenv("LTX_REAP_CHILD_AFTER_RENDER");
                    if ((!reap || reap[0] != '0') && runtime.worker) {
                        runtime.worker->shutdown();
                    }
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
