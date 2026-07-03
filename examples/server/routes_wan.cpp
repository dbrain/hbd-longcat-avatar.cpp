// routes_wan.cpp — Wan2.2-VACE multi-window continuation API (worker-isolated, async).
//
// POST /wan/v1/generate submits a continuation render. The whole clip runs in the CUDA child
// (worker isolation) via worker->render_video_chain(), or in-process, both through the shared
// run_vid_chain_job() which — seeing engine=wan — drives generate_wan_vace_chain() instead of
// the LTXAV chain. The prior window's diffusion latent + pixel tail are held IN MEMORY between
// windows (no cont_bank/*.bin round-trip). Poll/cancel/media reuse the /sdcpp/v1/jobs/{id}
// endpoints (registered by register_sdcpp_api_endpoints + register_ltx_video_endpoints).
//
// Request (multipart: a `request` JSON part + optional `init_image` / `ref_image` file parts):
//   { "mode": "t2v"|"i2v"|"continuation",
//     "segments": [ {"prompt": "..."}, ... ],   // one prompt per window
//     "n_segments": N,                           // optional; defaults to segments.length
//     "width","height","fps","frames","seed",
//     "quant": "nvfp4"|"q4_k",                   // informational (models are container-resident)
//     "negative_prompt": "",                     // optional
//     "has_init_image": bool, "has_reference_image": bool,
//     "resume_job_id": "<job>",                  // optional (banked-window resume)
//     "output_format": "webm" }
// The init/reference images ride as multipart file parts; they are inlined (base64) into the
// engine request as init_image (the i2v start / VACE identity anchor).

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "async_jobs.h"
#include "common/log.h"
#include "routes.h"
#include "runtime.h"

namespace fs = std::filesystem;
using json   = nlohmann::json;

// Job artifact root. Shares LTX_JOB_DIR with the LTX chain so the /sdcpp/v1/jobs/{id}/media
// disk fallback (registered by register_ltx_video_endpoints) resolves wan jobs too; a
// dedicated WAN_JOBS_DIR overrides. The in-RAM media path uses the job's stored job_dir
// regardless, so this only matters for post-restart re-fetch.
static fs::path wan_job_root() {
    if (const char* e = getenv("WAN_JOB_DIR"); e != nullptr && e[0] != '\0') {
        return fs::path(e);
    }
    if (const char* e = getenv("LTX_JOB_DIR"); e != nullptr && e[0] != '\0') {
        return fs::path(e);
    }
    return fs::path("/var/lib/wan-vace/jobs");
}

static int wan_banked_window_count(const fs::path& dir) {
    int             k = 0;
    std::error_code ec;
    while (fs::exists(dir / ("seg_" + std::to_string(k) + ".bin"), ec)) {
        ++k;
    }
    return k;
}

void register_wan_video_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    svr.Post("/wan/v1/generate", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (runtime->model_swap && runtime->model_swap->draining.load()) {
                res.status = 503;
                res.set_content(R"({"error":"service draining — not accepting new jobs"})", "application/json");
                return;
            }

            json                 body;
            std::vector<uint8_t> init_bytes;
            std::vector<uint8_t> ref_bytes;
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
                if (req.form.has_file("init_image")) {
                    const auto& c = req.form.get_file("init_image").content;
                    init_bytes.assign(c.begin(), c.end());
                }
                if (req.form.has_file("ref_image")) {
                    const auto& c = req.form.get_file("ref_image").content;
                    ref_bytes.assign(c.begin(), c.end());
                }
            } else {
                if (req.body.empty()) {
                    res.status = 400;
                    res.set_content(R"({"error":"empty body"})", "application/json");
                    return;
                }
                body = json::parse(req.body);
            }

            // Per-window prompts (segments[] of objects/strings, or a flat prompts[]).
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

            std::string resume_job_id = body.value("resume_job_id", std::string());
            std::string output_format = body.value("output_format", std::string("webm"));

            // Translate the koblem wan schema into the engine chain request that
            // run_vid_chain_job re-parses. engine=wan routes it to generate_wan_vace_chain;
            // ltx_chain_segments carries N; frames -> video_frames (the per-window length);
            // the init/reference image is inlined (base64 data-uri) as init_image, the VACE
            // identity anchor / i2v start. Sampler config (euler_a, steps 2, flow-shift 5, …)
            // comes from the container's default gen params (CLI), not the request.
            json chain;
            chain["engine"]             = "wan";
            chain["ltx_chain_segments"] = n_segments;
            chain["prompts"]            = prompts;
            if (!prompts.empty()) {
                chain["prompt"] = prompts[0];
            }
            if (body.contains("width")) {
                chain["width"] = body["width"];
            }
            if (body.contains("height")) {
                chain["height"] = body["height"];
            }
            if (body.contains("fps")) {
                chain["fps"] = body["fps"];
            }
            if (body.contains("frames")) {
                chain["video_frames"] = body["frames"];  // engine reads video_frames (per window)
            }
            if (body.contains("seed")) {
                chain["seed"] = body["seed"];
            }
            if (body.contains("negative_prompt")) {
                chain["negative_prompt"] = body["negative_prompt"];
            }
            chain["mode"]          = body.value("mode", std::string("t2v"));
            chain["output_format"] = output_format;

            // Inline the init / reference image (prefer an explicit init image; else the
            // reference anchor). t2v mode sends neither.
            const std::vector<uint8_t>& anchor = !init_bytes.empty() ? init_bytes : ref_bytes;
            if (!anchor.empty()) {
                chain["init_image"] = std::string("data:image/png;base64,") + base64_encode(anchor);
            }

            AsyncJobManager&                    manager = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job     = std::make_shared<AsyncGenerationJob>();
            job->kind                  = AsyncJobKind::VidGen;
            job->status                = AsyncJobStatus::Queued;
            job->created_at            = unix_timestamp_now();
            job->vid_gen.output_format = output_format;
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
            }

            // Resolve the artifact dir: an existing dir on resume, else this job's own.
            fs::path        root = wan_job_root();
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
                resume_from = wan_banked_window_count(job_dir);
                if (resume_from >= n_segments) {
                    resume_from = std::max(0, n_segments - 1);
                }
            } else {
                job_dir = root / job->id;
            }
            fs::create_directories(job_dir, ec);

            chain["ltx_job_dir"] = job_dir.string();
            chain["resume_from"] = resume_from;

            job->job_dir                = job_dir.string();
            job->vid_chain_request_json = chain.dump();
            {
                std::ofstream out((job_dir / "request.json").string(), std::ios::binary | std::ios::trunc);
                out << job->vid_chain_request_json;
            }

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

    LOG_INFO("wan-vace: POST /wan/v1/generate registered (async continuation chain; poll /sdcpp/v1/jobs/{id})\n");
}
