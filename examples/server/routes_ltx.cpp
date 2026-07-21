#include "routes.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "async_jobs.h"

namespace {

namespace fs = std::filesystem;

fs::path ltx_bank_root() {
    if (const char* configured = getenv("LTX_JOB_DIR"); configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    return "/var/lib/ltx-video/jobs";
}

bool valid_ltx_bank_id(const std::string& id) {
    return !id.empty() && id.size() <= 128 &&
           std::all_of(id.begin(), id.end(), [](unsigned char ch) {
               return std::isalnum(ch) || ch == '_' || ch == '-';
           });
}

bool resolve_ltx_bank_dir(const std::string& requested_id, fs::path& bank_dir, std::string& bank_id) {
    if (!valid_ltx_bank_id(requested_id)) {
        return false;
    }
    bank_id = requested_id;
    std::ifstream reference(ltx_bank_root() / requested_id / "bank_id");
    if (reference.is_open()) {
        std::string referenced_id;
        std::getline(reference, referenced_id);
        if (!valid_ltx_bank_id(referenced_id)) {
            return false;
        }
        bank_id = std::move(referenced_id);
    }
    bank_dir = ltx_bank_root() / bank_id;
    return true;
}

bool write_ltx_bank_reference(const std::string& job_id, const std::string& bank_id) {
    if (job_id == bank_id) {
        return true;
    }
    std::error_code error;
    const fs::path reference_dir = ltx_bank_root() / job_id;
    fs::create_directories(reference_dir, error);
    if (error) {
        return false;
    }
    std::ofstream reference(reference_dir / "bank_id", std::ios::trunc);
    reference << bank_id << '\n';
    return reference.good();
}

bool extract_ltx_request(const httplib::Request& req, json& body) {
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
    return true;
}

bool parse_ltx_video_request(const json& body,
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

}  // namespace

void register_ltx_video_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;
    svr.Post("/ltx/v1/generate", [runtime](const httplib::Request& req, httplib::Response& res) {
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
            if (!extract_ltx_request(req, body)) {
                res.status = 400;
                res.set_content(R"({"error":"missing or invalid request"})", "application/json");
                return;
            }
            std::vector<std::string> prompts;
            std::vector<int> segment_frames;
            if (body.contains("segments") && body["segments"].is_array()) {
                for (const auto& segment : body["segments"]) {
                    if (segment.is_string()) {
                        prompts.push_back(segment.get<std::string>());
                        segment_frames.push_back(0);
                    } else if (segment.is_object()) {
                        prompts.push_back(segment.value("prompt", std::string()));
                        const int frames = segment.value("frames", 0);
                        if (frames < 0 || (frames > 0 && (frames - 1) % 8 != 0)) {
                            res.status = 400;
                            res.set_content(R"({"error":"each LTX segment frames value must be 8k+1"})", "application/json");
                            return;
                        }
                        segment_frames.push_back(frames);
                    }
                }
            } else if (body.contains("prompts") && body["prompts"].is_array()) {
                for (const auto& prompt : body["prompts"]) {
                    if (prompt.is_string()) {
                        prompts.push_back(prompt.get<std::string>());
                        segment_frames.push_back(0);
                    }
                }
            }
            const int requested_segments = body.value("n_segments", static_cast<int>(prompts.size()));
            if (requested_segments < 1 || prompts.empty() || requested_segments != static_cast<int>(prompts.size()) ||
                std::any_of(prompts.begin(), prompts.end(), [](const std::string& prompt) { return prompt.empty(); })) {
                res.status = 400;
                res.set_content(R"({"error":"segments must contain exactly n_segments non-empty prompts"})", "application/json");
                return;
            }
            body["prompt"] = prompts.front();
            if (body.contains("frames")) {
                body["video_frames"] = body["frames"];
            }

            VidGenJobRequest request;
            std::string error_message;
            if (!parse_ltx_video_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }
            const int continuation_frames = body.value("cont_latent_frames", 3);
            if (continuation_frames < 1) {
                res.status = 400;
                res.set_content(R"({"error":"cont_latent_frames must be positive"})", "application/json");
                return;
            }

            AsyncJobManager& manager = *runtime->async_job_manager;
            auto job = std::make_shared<AsyncGenerationJob>();
            job->kind = AsyncJobKind::VidGen;
            job->status = AsyncJobStatus::Queued;
            job->created_at = unix_timestamp_now();
            job->vid_gen = std::move(request);
            job->ltx_prompts = std::move(prompts);
            job->ltx_segment_frames = std::move(segment_frames);
            job->ltx_cont_latent_frames = continuation_frames;
            const std::string resume_job_id = body.value("resume_job_id", std::string());
            int resume_from = 0;
            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                fs::path bank_dir;
                std::string bank_id;
                if (!resume_job_id.empty()) {
                    if (!resolve_ltx_bank_dir(resume_job_id, bank_dir, bank_id)) {
                        res.status = 400;
                        res.set_content(R"({"error":"invalid resume_job_id"})", "application/json");
                        return;
                    }
                    const auto prior = manager.jobs.find(resume_job_id);
                    if (prior != manager.jobs.end() &&
                        (prior->second->status == AsyncJobStatus::Queued || prior->second->status == AsyncJobStatus::Generating)) {
                        res.status = 409;
                        res.set_content(R"({"error":"resume_job_id is still rendering"})", "application/json");
                        return;
                    }
                    for (; fs::exists(bank_dir / ("seg_" + std::to_string(resume_from) + ".bin")); ++resume_from) {}
                    if (resume_from <= 0 || resume_from >= static_cast<int>(job->ltx_prompts.size())) {
                        res.status = 404;
                        res.set_content(R"({"error":"resume_job_id has no resumable LTX latent bank"})", "application/json");
                        return;
                    }
                } else {
                    job->id = make_async_job_id(manager);
                    bank_dir = ltx_bank_root() / job->id;
                    bank_id = job->id;
                }
                std::error_code error;
                fs::create_directories(bank_dir, error);
                if (error) {
                    res.status = 500;
                    res.set_content(json({{"error", "could not create LTX bank directory"}, {"message", error.message()}}).dump(), "application/json");
                    return;
                }
                if (job->id.empty()) job->id = make_async_job_id(manager);
                if (!write_ltx_bank_reference(job->id, bank_id)) {
                    res.status = 500;
                    res.set_content(R"({"error":"could not persist LTX bank reference"})", "application/json");
                    return;
                }
                job->ltx_bank_dir = bank_dir.string();
                job->ltx_bank_id = bank_id;
                job->ltx_resume_from = resume_from;
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }
            manager.cv.notify_one();
            res.status = 202;
            res.set_content(json({{"id", job->id},
                                  {"kind", "ltx"},
                                  {"status", async_job_status_name(job->status)},
                                  {"created", job->created_at},
                                  {"poll_url", "/sdcpp/v1/jobs/" + job->id},
                                  {"media_url", "/sdcpp/v1/jobs/" + job->id + "/media"},
                                  {"segments", static_cast<int>(job->ltx_prompts.size())},
                                  {"resume_from", resume_from},
                                  {"resume_job_id", job->ltx_bank_id}})
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
