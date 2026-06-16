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
//     "hires": { ... },                         // optional spatial upscaler
//     "output_format": "webm" }
// Per-segment lip-sync wavs (16kHz mono) ride as multipart parts audio_0, audio_1, …; the
// parent writes them to a shared /tmp dir and passes the dir to the chain.

#include <atomic>
#include <filesystem>
#include <fstream>
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

            // Per-job scratch dir for the pre-sliced audio (shared with the CUDA child via
            // /tmp). The dir name is a process-unique sequence, independent of the job id.
            static std::atomic<uint64_t> seq{0};
            uint64_t    my_seq = seq.fetch_add(1);
            fs::path    job_dir = fs::temp_directory_path() / "ltx-video-jobs" / std::to_string(my_seq);
            std::string audio_dir;
            if (!audio_parts.empty()) {
                std::error_code ec;
                fs::create_directories(job_dir / "audio", ec);
                for (const auto& [idx, bytes] : audio_parts) {
                    ltx_write_blob((job_dir / "audio" / ("aud_" + std::to_string(idx) + ".wav")).string(), bytes);
                }
                audio_dir = (job_dir / "audio").string();
            }

            // The chain request the worker (or the in-process path) re-parses. Start from the
            // client's params (W/H/fps/steps/cfg/sampler/scheduler/seed/negative/init_image/
            // hires), then inject the chain extras. from_json_str reads the gen params + the
            // inline base64 init_image + ltx_chain_segments; run_vid_chain_job reads prompts[]
            // and chain_audio_dir.
            json chain                 = body;
            chain["ltx_chain_segments"] = n_segments;
            chain["prompts"]            = prompts;
            if (!prompts.empty()) {
                chain["prompt"] = prompts[0];  // base prompt = seg0 (seg0 / fallback)
            }
            if (!audio_dir.empty()) {
                chain["chain_audio_dir"] = audio_dir;
            }
            std::string output_format = body.value("output_format", std::string("webm"));

            AsyncJobManager&                    manager = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job     = std::make_shared<AsyncGenerationJob>();
            job->kind                     = AsyncJobKind::VidGen;
            job->status                   = AsyncJobStatus::Queued;
            job->created_at               = unix_timestamp_now();
            job->vid_gen.output_format    = output_format;  // drives make_async_job_json mime
            job->vid_chain_request_json   = chain.dump();

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
            out["segments"] = n_segments;

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

    LOG_INFO("ltx-video: POST /ltx/v1/generate registered (async chain; poll /sdcpp/v1/jobs/{id})\n");
}
