#include "routes.h"

#include <string>
#include <vector>

#include "async_jobs.h"

namespace {

bool parse_wan_video_request(const json& body,
                             ServerRuntime& runtime,
                             VidGenJobRequest& request,
                             std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;
    refresh_lora_cache(runtime);
    if (!request.gen_params.from_json_str(body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }
    if (!assign_output_options(request,
                               body.value("output_format", std::string("webm")),
                               body.value("output_compression", 100),
                               error_message)) {
        return false;
    }
    if (!request.gen_params.resolve_and_validate(VID_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
}

bool extract_wan_request(const httplib::Request& req, json& body, std::vector<uint8_t>& anchor) {
    if (!req.is_multipart_form_data()) {
        if (req.body.empty()) {
            return false;
        }
        body = json::parse(req.body);
        return true;
    }
    std::string request_json;
    if (req.form.has_field("request")) {
        request_json = req.form.get_field("request");
    } else if (req.form.has_file("request")) {
        request_json = req.form.get_file("request").content;
    }
    if (request_json.empty()) {
        return false;
    }
    body = json::parse(request_json);
    const char* names[] = {"init_image", "ref_image"};
    for (const char* name : names) {
        if (!anchor.empty() || !req.form.has_file(name)) {
            continue;
        }
        const auto& content = req.form.get_file(name).content;
        anchor.assign(content.begin(), content.end());
    }
    return true;
}

}  // namespace

void register_wan_video_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;
    svr.Post("/wan/v1/generate", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (runtime_is_draining(*runtime)) {
                res.status = 503;
                res.set_content(R"({"error":"service draining — not accepting new jobs"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, VID_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(VID_GEN)}}).dump(), "application/json");
                return;
            }
            json body;
            std::vector<uint8_t> anchor;
            if (!extract_wan_request(req, body, anchor)) {
                res.status = 400;
                res.set_content(R"({"error":"missing or invalid request"})", "application/json");
                return;
            }
            std::vector<std::string> prompts;
            if (body.contains("segments") && body["segments"].is_array()) {
                for (const auto& segment : body["segments"]) {
                    if (segment.is_string()) {
                        prompts.push_back(segment.get<std::string>());
                    } else if (segment.is_object()) {
                        prompts.push_back(segment.value("prompt", std::string()));
                    }
                }
            } else if (body.contains("prompts") && body["prompts"].is_array()) {
                for (const auto& prompt : body["prompts"]) {
                    if (prompt.is_string()) {
                        prompts.push_back(prompt.get<std::string>());
                    }
                }
            }
            const int requested_segments = body.value("n_segments", static_cast<int>(prompts.size()));
            if (requested_segments < 1 || prompts.empty() || requested_segments != static_cast<int>(prompts.size())) {
                res.status = 400;
                res.set_content(R"({"error":"segments must contain exactly n_segments non-empty prompts"})", "application/json");
                return;
            }
            body["prompt"] = prompts.front();
            if (body.contains("frames")) {
                body["video_frames"] = body["frames"];
            }
            if (!anchor.empty()) {
                body["init_image"] = std::string("data:image/png;base64,") + base64_encode(anchor);
            }

            VidGenJobRequest request;
            std::string error_message;
            if (!parse_wan_video_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }
            AsyncJobManager& manager = *runtime->async_job_manager;
            auto job = std::make_shared<AsyncGenerationJob>();
            job->kind = AsyncJobKind::VidGen;
            job->status = AsyncJobStatus::Queued;
            job->created_at = unix_timestamp_now();
            job->vid_gen = std::move(request);
            job->wan_vace_prompts = std::move(prompts);
            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id = make_async_job_id(manager);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }
            manager.cv.notify_one();
            res.status = 202;
            res.set_content(json({{"id", job->id},
                                  {"kind", "wan_vace"},
                                  {"status", async_job_status_name(job->status)},
                                  {"created", job->created_at},
                                  {"poll_url", "/sdcpp/v1/jobs/" + job->id},
                                  {"media_url", "/sdcpp/v1/jobs/" + job->id + "/media"},
                                  {"segments", static_cast<int>(job->wan_vace_prompts.size())}})
                                .dump(),
                            "application/json");
        } catch (const json::parse_error& error) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", error.what()}}).dump(), "application/json");
        } catch (const std::exception& error) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", error.what()}}).dump(), "application/json");
        }
    });
}
