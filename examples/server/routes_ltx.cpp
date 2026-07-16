// routes_ltx.cpp — LTXAV multi-segment video chain API (worker-isolated, async).
//
// POST /ltx/v1/generate submits a chain render. The whole chain runs in the CUDA child
// (worker isolation) via worker->render_video_chain(), or in-process when no worker is
// set; both go through the shared run_vid_chain_job(). Poll/cancel reuse the existing
// /sdcpp/v1/jobs/{id} endpoints (registered by register_sdcpp_api_endpoints), and the GPU
// gate uses the /v1/admin/{drain,unload,load} endpoints (register_sdcpp_admin_endpoints).
//
// Request (JSON, or multipart with a `request` part + `audio_<i>` wav parts):
//   { "segments": [ {"prompt": "..."}, ... ]   // OR "prompts": ["...", ...]
//     "n_segments": N,                          // optional; defaults to segments.length
//     "cont_latent_frames": 3,                  // K overlap latent frames
//     "width","height","fps","steps","cfg_scale","sampling_method","scheduler","seed",
//     "negative_prompt": "",
//     "init_image": "<base64|data-uri>",        // optional, seg0 i2v (from_json_str loads it)
//     "character_reference": "<base64|data-uri>", // optional LTXAV identity-only DiT reference
//     "control_frames": ["<base64|data-uri>", ...], // V2V source: N consecutive source windows
//     "relip_control_frame_counts": [97, ...],    // optional exact source-frame count per window
//     "v2v_mode": 0,                              // how control_frames are used: 0=lipdub relip
//                                                 //   (default), 1=SDEdit restyle, 2=guide-edit
//                                                 //   (Director-2 keep-scene-add-element). Legacy
//                                                 //   bool "v2v":true == mode 1. Per-segment via
//                                                 //   segments[i].v2v_mode (int) or segments[i].v2v.
//     "v2v_guide_strength": 1.0,                  // guide-edit (mode 2) LTXDirectorGuide scale:
//                                                 //   1.0=hold scene, ~0.5=bigger edit (chain-global)
//     "v2v_guide_latent_path": "/…/seg_0.bin",    // guide-edit LATENT-IN source (PREFERRED when we
//                                                 //   rendered the source): a banked diffusion latent
//                                                 //   to guide from with NO pixel re-encode. Per-seg
//                                                 //   via segments[i].v2v_source_latent_path. Absent
//                                                 //   => fall back to control_frames (pixel encode).
//     "model": "edit",                            // selects the lipdub DiT variant
//     "hires": { ... },                         // optional legacy single spatial upscaler
//     "hires_chain": [ { "upscaler": "<spatial-upscaler-x2 name>",
//                         "custom_sigmas": [0.85,0.725,0.421875,0.0],
//                         "sample_method": "euler_cfg_pp", "cfg": 1.0, "steps": 3 } ],
//                                                 // optional ordered upscale+SDEdit stages; non-empty replaces hires
//     "output_format": "webm" }
// Per-segment lip-sync wavs (16kHz mono) ride as multipart parts audio_0, audio_1, …; the
// parent writes them to a shared /tmp dir and passes the dir to the chain. For relip every
// source window is independent (no generated latent carry / overlap trim): provide all source
// frames in segment order, or an explicit relip_control_frame_counts partition.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "async_jobs.h"
#include "common/log.h"
#include "routes.h"
#include "runtime.h"

namespace fs = std::filesystem;
using json   = nlohmann::json;

static void ltx_write_blob(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Root for durable per-job artifact dirs (request + per-segment latents/webm + final.webm).
// Overridable via LTX_JOB_DIR; mounted in compose so renders survive a container restart.
static fs::path ltx_job_root() {
    const char* e = getenv("LTX_JOB_DIR");
    return fs::path((e != nullptr && e[0] != '\0') ? e : "/var/lib/ltx-video/jobs");
}

// Count the contiguous run of banked segment latents (seg_0.bin, seg_1.bin, …) in a job
// dir. This is the resume point: segments [0, k) are done, so a resume renders [k, N).
static int ltx_banked_segment_count(const fs::path& dir) {
    int k = 0;
    std::error_code ec;
    while (fs::exists(dir / ("seg_" + std::to_string(k) + ".bin"), ec)) {
        ++k;
    }
    return k;
}

// Keep only the newest LTX_JOB_KEEP job dirs (default 20), deleting older ones so the
// artifact root doesn't grow unbounded. `keep_dir` (the current/resumed job) is never
// swept. Best-effort: filesystem errors are ignored.
static void ltx_sweep_old_jobs(const fs::path& root, const fs::path& keep_dir) {
    const char* e   = getenv("LTX_JOB_KEEP");
    int         max = (e != nullptr && e[0] != '\0') ? std::atoi(e) : 20;
    if (max <= 0) {
        return;
    }
    std::error_code ec;
    std::vector<std::pair<fs::file_time_type, fs::path>> dirs;
    for (auto it = fs::directory_iterator(root, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_directory(ec)) {
            continue;
        }
        dirs.emplace_back(fs::last_write_time(it->path(), ec), it->path());
    }
    if ((int)dirs.size() <= max) {
        return;
    }
    std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = (size_t)max; i < dirs.size(); ++i) {
        if (dirs[i].second == keep_dir) {
            continue;
        }
        fs::remove_all(dirs[i].second, ec);
    }
}

void register_ltx_video_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    svr.Post("/ltx/v1/generate", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (runtime->model_swap && runtime->model_swap->draining.load()) {
                res.status = 503;
                res.set_content(R"({"error":"service draining — not accepting new jobs"})", "application/json");
                return;
            }

            json body;
            std::vector<std::pair<int, std::string>> audio_parts;  // (segment index, wav bytes)
            if (req.is_multipart_form_data()) {
                std::string req_json;
                if (req.form.has_field("request")) {
                    req_json = req.form.get_field("request");
                } else if (req.form.has_file("request")) {
                    req_json = req.form.get_file("request").content;
                }
                if (req_json.empty()) {
                    res.status = 400;
                    res.set_content(R"({"error":"missing 'request' part"})", "application/json");
                    return;
                }
                body = json::parse(req_json);
                for (int i = 0;; ++i) {
                    std::string key = "audio_" + std::to_string(i);
                    if (!req.form.has_file(key)) {
                        break;
                    }
                    audio_parts.emplace_back(i, req.form.get_file(key).content);
                }
            } else {
                if (req.body.empty()) {
                    res.status = 400;
                    res.set_content(R"({"error":"empty body"})", "application/json");
                    return;
                }
                body = json::parse(req.body);
            }

            // Per-segment prompts (the director layer). Accept segments[] of objects/strings
            // or a flat prompts[].
            std::vector<std::string> prompts;
            if (body.contains("segments") && body["segments"].is_array()) {
                for (const auto& s : body["segments"]) {
                    if (s.is_string()) {
                        prompts.push_back(s.get<std::string>());
                    } else if (s.is_object()) {
                        prompts.push_back(s.value("prompt", std::string()));
                    }
                }
            } else if (body.contains("prompts") && body["prompts"].is_array()) {
                for (const auto& p : body["prompts"]) {
                    if (p.is_string()) {
                        prompts.push_back(p.get<std::string>());
                    }
                }
            }
            int n_segments = body.value("n_segments", static_cast<int>(prompts.size()));
            if (n_segments < 1) {
                n_segments = 1;
            }

            // Resume: a prior job's id whose banked segments we continue from. The new job
            // renders into the SAME artifact dir, skipping segments already on disk.
            std::string resume_job_id = body.value("resume_job_id", std::string());

            // The chain request the worker (or the in-process path) re-parses. Start from the
            // client's params (W/H/fps/steps/cfg/sampler/scheduler/seed/negative/init_image/
            // hires), then inject the chain extras. from_json_str reads the gen params + the
            // inline base64 init_image + ltx_chain_segments; run_vid_chain_job reads prompts[],
            // chain_audio_dir, ltx_job_dir and resume_from.
            json chain                  = body;
            chain["ltx_chain_segments"] = n_segments;
            chain["prompts"]            = prompts;
            if (!prompts.empty()) {
                chain["prompt"] = prompts[0];  // base prompt = seg0 (seg0 / fallback)
            }
            // Clip length: clients send "frames" (the wan/koblem convention), but the gen-params
            // parser (SDGenerationParams::from_json_str) reads "video_frames". chain = body already,
            // yet without this alias the client's requested length is silently dropped and the
            // render falls back to the server's --video-frames default. Mirror routes_wan.cpp.
            // The value should be model-valid (8k+1, e.g. 97/257); snapping is the client's job.
            if (body.contains("frames")) {
                chain["video_frames"] = body["frames"];
            }
            // RETAKE (bidirectional single-segment splice) is OFF unless a valid retake_segment is
            // supplied against a banked (resume) job below. Default it off so a stray body field or
            // a fresh (non-resume) job can never trigger a spurious single-segment render.
            chain["retake_segment"] = -1;
            // Progressive per-segment delivery (opt-in). When true, run_vid_chain_job banks a
            // viewable seg_<n>.webm as each shot lands and the job status lists it under
            // "partials" so a client can play each shot while the next renders. Default false =
            // byte-identical (no seg encode, no partials). Normalize the default here so the
            // downstream reader (async_jobs.cpp) sees an explicit bool.
            chain["emit_segments"] = body.value("emit_segments", false);
            std::string output_format = body.value("output_format", std::string("webm"));

            // Register the job (assign an id) before touching the filesystem, so a fresh job's
            // artifact dir can be keyed by its own id.
            AsyncJobManager&                    manager = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job     = std::make_shared<AsyncGenerationJob>();
            job->kind                  = AsyncJobKind::VidGen;
            job->status                = AsyncJobStatus::Queued;
            job->created_at            = unix_timestamp_now();
            job->vid_gen.output_format = output_format;  // drives make_async_job_json mime
            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id               = make_async_job_id(manager);
                manager.jobs[job->id] = job;  // registered but not queued until inputs are on disk
            }

            // Resolve the artifact dir: an existing dir on resume, else this job's own.
            fs::path        root = ltx_job_root();
            fs::path        job_dir;
            int             resume_from = 0;
            std::error_code ec;
            if (!resume_job_id.empty()) {
                job_dir = root / resume_job_id;
                if (!fs::exists(job_dir, ec)) {
                    std::lock_guard<std::mutex> lock(manager.mutex);
                    manager.jobs.erase(job->id);
                    res.status = 404;
                    res.set_content(R"({"error":"resume_job_id not found"})", "application/json");
                    return;
                }
                resume_from = ltx_banked_segment_count(job_dir);
                if (resume_from >= n_segments) {
                    // All segment latents survived but final.webm did not (for example a
                    // crash during finalization): reload and stitch every banked segment,
                    // then encode only. Keep the old last-segment rerender behaviour when a
                    // completed final artifact is already present.
                    if (!fs::exists(job_dir / ("final." + output_format), ec)) {
                        resume_from = n_segments;
                    } else {
                        resume_from = std::max(0, n_segments - 1);
                    }
                }
                // RETAKE: re-render ONLY this banked shot, pinned by both neighbours, then splice
                // the banked tail. The engine derives its own prefix-reload point from
                // retake_segment; align resume_from to it so request.json stays consistent.
                // Accept koblem's existing "retake_from" as an alias (same FILTERED-index meaning).
                int retake_segment = body.value("retake_segment", body.value("retake_from", -1));
                if (retake_segment >= 0 && retake_segment < n_segments) {
                    chain["retake_segment"] = retake_segment;
                    resume_from             = retake_segment;  // reload banked prefix [0, retake_segment)
                }
            } else {
                job_dir = root / job->id;
            }
            fs::create_directories(job_dir, ec);

            // Pre-sliced per-segment lip-sync wavs (16kHz mono). On a fresh job they ride in
            // as multipart parts; on resume they already sit in the dir from the first submit.
            std::string audio_dir;
            if (!audio_parts.empty()) {
                fs::create_directories(job_dir / "audio", ec);
                for (const auto& [idx, bytes] : audio_parts) {
                    ltx_write_blob((job_dir / "audio" / ("aud_" + std::to_string(idx) + ".wav")).string(), bytes);
                }
            }
            if (fs::exists(job_dir / "audio", ec)) {
                audio_dir = (job_dir / "audio").string();
            }
            if (!audio_dir.empty()) {
                chain["chain_audio_dir"] = audio_dir;
            }
            chain["ltx_job_dir"] = job_dir.string();
            chain["resume_from"] = resume_from;

            // Persist the inputs so the job is fully replayable / resumable. request.json
            // carries every param incl. the inline base64 init image; prompts.txt is the
            // human-readable director script.
            job->job_dir                = job_dir.string();
            job->vid_chain_request_json = chain.dump();
            ltx_write_blob((job_dir / "request.json").string(), job->vid_chain_request_json);
            {
                std::string ptxt;
                for (const auto& p : prompts) {
                    ptxt += p;
                    ptxt += '\n';
                }
                ltx_write_blob((job_dir / "prompts.txt").string(), ptxt);
            }
            ltx_sweep_old_jobs(root, job_dir);

            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                manager.queue.push_back(job->id);
            }
            manager.cv.notify_one();

            json out;
            out["id"]          = job->id;
            out["kind"]        = async_job_kind_name(job->kind);
            out["status"]      = async_job_status_name(job->status);
            out["created"]     = job->created_at;
            out["poll_url"]    = "/sdcpp/v1/jobs/" + job->id;
            out["media_url"]   = "/sdcpp/v1/jobs/" + job->id + "/media";
            out["segments"]    = n_segments;
            out["resume_from"] = resume_from;
            out["retake_segment"] = chain.value("retake_segment", -1);
            out["job_dir"]     = job_dir.string();

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

    // GET /sdcpp/v1/jobs/{id}/media — stream the finished final.webm from the job's artifact
    // dir. Survives the in-RAM result TTL and a koblem disconnect: the file is read from disk
    // by job id, falling back to LTX_JOB_DIR/<id> when the job has already aged out of RAM.
    svr.Get(R"(/sdcpp/v1/jobs/([^/]+)/media)", [runtime](const httplib::Request& req, httplib::Response& res) {
        const std::string job_id = req.matches[1];
        fs::path          dir;
        {
            AsyncJobManager&            manager = *runtime->async_job_manager;
            std::lock_guard<std::mutex> lock(manager.mutex);
            auto                        it = manager.jobs.find(job_id);
            if (it != manager.jobs.end() && !it->second->job_dir.empty()) {
                dir = it->second->job_dir;
            }
        }
        if (dir.empty()) {
            dir = ltx_job_root() / job_id;  // job aged out of RAM; serve from disk
        }
        fs::path        final_webm = dir / "final.webm";
        std::error_code ec;
        if (!fs::exists(final_webm, ec)) {
            res.status = 404;
            res.set_content(R"({"error":"no final.webm for job"})", "application/json");
            return;
        }
        std::ifstream in(final_webm.string(), std::ios::binary);
        std::string   bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        res.set_content(bytes, "video/webm");
    });

    // GET /sdcpp/v1/jobs/{id}/segments/{n} — stream a finished per-segment preview webm
    // (seg_<n>.webm) from the job's artifact dir. This is the progressive-delivery fetch: the
    // client polls /sdcpp/v1/jobs/{id}, reads the "partials" list, and pulls each seg webm as it
    // lands. 404 until that segment's atomic seg_<n>.webm exists. Mirrors the /media handler:
    // resolve the dir from RAM (falling back to LTX_JOB_DIR/<id> after the in-RAM TTL) and stream.
    svr.Get(R"(/sdcpp/v1/jobs/([^/]+)/segments/(\d+))",
            [runtime](const httplib::Request& req, httplib::Response& res) {
        const std::string job_id = req.matches[1];
        const std::string seg_n  = req.matches[2];
        fs::path          dir;
        {
            AsyncJobManager&            manager = *runtime->async_job_manager;
            std::lock_guard<std::mutex> lock(manager.mutex);
            auto                        it = manager.jobs.find(job_id);
            if (it != manager.jobs.end() && !it->second->job_dir.empty()) {
                dir = it->second->job_dir;
            }
        }
        if (dir.empty()) {
            dir = ltx_job_root() / job_id;  // job aged out of RAM; serve from disk
        }
        fs::path        seg_webm = dir / ("seg_" + seg_n + ".webm");
        std::error_code ec;
        if (!fs::exists(seg_webm, ec)) {
            res.status = 404;
            res.set_content(R"({"error":"segment webm not available"})", "application/json");
            return;
        }
        std::ifstream in(seg_webm.string(), std::ios::binary);
        std::string   bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        res.set_content(bytes, "video/webm");
    });

    LOG_INFO("ltx-video: POST /ltx/v1/generate + GET /sdcpp/v1/jobs/{id}/media + "
             "GET /sdcpp/v1/jobs/{id}/segments/{n} registered "
             "(async chain; poll /sdcpp/v1/jobs/{id})\n");
}
