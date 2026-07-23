#include "routes.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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

fs::path ltx_persist_root() {
    if (const char* configured = getenv("LTX_PERSIST_DIR"); configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    return "/var/lib/ltx-video/persist";
}

std::vector<fs::path> ltx_bank_roots() {
    std::vector<fs::path> roots;
    const fs::path persist = ltx_persist_root();
    const fs::path transient = ltx_bank_root();
    roots.push_back(persist);
    if (transient != persist) roots.push_back(transient);
    return roots;
}

bool valid_ltx_bank_id(const std::string& id) {
    return !id.empty() && id.size() <= 128 &&
           std::all_of(id.begin(), id.end(), [](unsigned char ch) {
               return std::isalnum(ch) || ch == '_' || ch == '-';
           });
}

bool resolve_ltx_bank_dir(const std::string& requested_id,
                          fs::path& bank_dir,
                          std::string& bank_id,
                          fs::path* bank_root = nullptr) {
    if (!valid_ltx_bank_id(requested_id)) {
        return false;
    }
    for (const fs::path& root : ltx_bank_roots()) {
        const fs::path requested_dir = root / requested_id;
        std::error_code error;
        if (!fs::is_directory(requested_dir, error)) {
            continue;
        }
        std::string resolved_id = requested_id;
        std::ifstream reference(requested_dir / "bank_id");
        if (reference.is_open()) {
            std::string referenced_id;
            std::getline(reference, referenced_id);
            if (!valid_ltx_bank_id(referenced_id)) {
                return false;
            }
            resolved_id = std::move(referenced_id);
        }
        const fs::path resolved_dir = root / resolved_id;
        if (!fs::is_directory(resolved_dir, error)) {
            return false;
        }
        bank_id = std::move(resolved_id);
        bank_dir = resolved_dir;
        if (bank_root != nullptr) *bank_root = root;
        return true;
    }
    return false;
}

bool write_ltx_bank_reference(const fs::path& root, const std::string& job_id, const std::string& bank_id) {
    if (job_id == bank_id) {
        return true;
    }
    std::error_code error;
    const fs::path reference_dir = root / job_id;
    fs::create_directories(reference_dir, error);
    if (error) {
        return false;
    }
    std::ofstream reference(reference_dir / "bank_id", std::ios::trunc);
    reference << bank_id << '\n';
    return reference.good();
}

bool resolve_ltx_guide_latent_path(const std::string& requested_path, std::string& resolved_path) {
    std::error_code error;
    const fs::path candidate = fs::weakly_canonical(requested_path, error);
    if (error || !fs::is_regular_file(candidate, error)) return false;
    bool within_bank_root = false;
    for (const fs::path& bank_root : ltx_bank_roots()) {
        error.clear();
        const fs::path root = fs::weakly_canonical(bank_root, error);
        if (error) continue;
        const fs::path relative = candidate.lexically_relative(root);
        if (!relative.empty() && !relative.is_absolute() && relative.begin() != relative.end() &&
            relative.begin()->string() != "..") {
            within_bank_root = true;
            break;
        }
    }
    if (!within_bank_root) return false;
    const std::string filename = candidate.filename().string();
    if (filename.size() <= 8 || filename.rfind("seg_", 0) != 0 ||
        filename.compare(filename.size() - 4, 4, ".bin") != 0 ||
        !std::all_of(filename.begin() + 4, filename.end() - 4,
                     [](unsigned char ch) { return std::isdigit(ch); })) {
        return false;
    }
    resolved_path = candidate.string();
    return true;
}

bool resolve_ltx_guide_latent_reference(const std::string& job_id, int segment, std::string& resolved_path) {
    if (segment < 0) return false;
    fs::path bank_dir;
    std::string bank_id;
    if (!resolve_ltx_bank_dir(job_id, bank_dir, bank_id)) return false;
    std::error_code error;
    const fs::path candidate = bank_dir / ("seg_" + std::to_string(segment) + ".bin");
    if (!fs::is_regular_file(candidate, error)) return false;
    resolved_path = candidate.string();
    return true;
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
            std::string drive_audio_bytes;
            std::string track_audio_bytes;
            std::vector<std::pair<int, std::string>> legacy_audio_parts;
            std::vector<std::pair<int, std::string>> shot_audio_full_parts;
            std::vector<std::pair<int, std::string>> shot_audio_track_parts;
            if (req.is_multipart_form_data()) {
                // Backward-compatible per-window lip-sync WAVs.  Koblem's
                // older relip/window path sends contiguous audio_<n> parts;
                // retain them in the durable LTX bank so retries/resumes keep
                // exactly the same conditioning inputs.
                for (int index = 0;; ++index) {
                    const std::string key = "audio_" + std::to_string(index);
                    if (!req.form.has_file(key)) break;
                    const std::string& bytes = req.form.get_file(key).content;
                    if (bytes.empty()) {
                        res.status = 400;
                        res.set_content(json({{"error", key + " must be a non-empty WAV upload"}}).dump(), "application/json");
                        return;
                    }
                    legacy_audio_parts.emplace_back(index, bytes);
                }
                if (req.form.has_file("audio_full")) {
                    drive_audio_bytes = req.form.get_file("audio_full").content;
                    if (drive_audio_bytes.empty()) {
                        res.status = 400;
                        res.set_content(R"({"error":"audio_full must be a non-empty WAV upload"})", "application/json");
                        return;
                    }
                }
                if (req.form.has_file("audio_track")) {
                    track_audio_bytes = req.form.get_file("audio_track").content;
                    if (track_audio_bytes.empty()) {
                        res.status = 400;
                        res.set_content(R"({"error":"audio_track must be a non-empty WAV upload"})", "application/json");
                        return;
                    }
                }
            }
            std::vector<std::string> prompts;
            std::vector<std::string> segment_models;
            std::vector<int> segment_frames;
            std::vector<int> segment_scene_cuts;
            std::vector<std::string> segment_init_images;
            std::vector<std::vector<std::string>> segment_keyframes;
            std::vector<std::vector<int>> segment_keyframe_indices;
            std::vector<std::vector<std::string>> segment_control_frames;
            std::vector<int> segment_v2v_modes;
            std::vector<float> segment_v2v_strengths;
            std::vector<std::string> segment_v2v_guide_latent_paths;
            if (body.contains("segments") && body["segments"].is_array()) {
                for (const auto& segment : body["segments"]) {
                    if (segment.is_string()) {
                        prompts.push_back(segment.get<std::string>());
                        segment_models.emplace_back();
                        segment_frames.push_back(0);
                        segment_scene_cuts.push_back(0);
                        segment_init_images.emplace_back();
                        segment_keyframes.emplace_back();
                        segment_keyframe_indices.emplace_back();
                        segment_control_frames.emplace_back();
                        segment_v2v_modes.push_back(0);
                        segment_v2v_strengths.push_back(-1.f);
                        segment_v2v_guide_latent_paths.emplace_back();
                    } else if (segment.is_object()) {
                        prompts.push_back(segment.value("prompt", std::string()));
                        if (segment.contains("model") && !segment["model"].is_string()) {
                            res.status = 400;
                            res.set_content(R"({"error":"each LTX segment model must be a string"})", "application/json");
                            return;
                        }
                        segment_models.push_back(segment.value("model", std::string()));
                        const int frames = segment.value("frames", 0);
                        if (frames < 0 || (frames > 0 && (frames - 1) % 8 != 0)) {
                            res.status = 400;
                            res.set_content(R"({"error":"each LTX segment frames value must be 8k+1"})", "application/json");
                            return;
                        }
                        segment_frames.push_back(frames);
                        segment_scene_cuts.push_back(segment.value("scene_cut", false) ? 1 : 0);
                        segment_init_images.push_back(segment.value("init_image", std::string()));
                        std::vector<std::string> keyframes;
                        std::vector<int> keyframe_indices;
                        if (segment.contains("keyframes")) {
                            if (!segment["keyframes"].is_array() || segment["keyframes"].empty()) {
                                res.status = 400;
                                res.set_content(R"({"error":"LTX keyframes must be a non-empty array of {image, frame}"})", "application/json");
                                return;
                            }
                            for (const auto& keyframe : segment["keyframes"]) {
                                if (!keyframe.is_object() || !keyframe.contains("image") ||
                                    !keyframe["image"].is_string() || keyframe["image"].get<std::string>().empty() ||
                                    !keyframe.contains("frame") || !keyframe["frame"].is_number_integer()) {
                                    res.status = 400;
                                    res.set_content(R"({"error":"each LTX keyframe needs a non-empty image and integer frame"})", "application/json");
                                    return;
                                }
                                keyframes.push_back(keyframe["image"].get<std::string>());
                                keyframe_indices.push_back(keyframe["frame"].get<int>());
                            }
                        }
                        segment_keyframes.push_back(std::move(keyframes));
                        segment_keyframe_indices.push_back(std::move(keyframe_indices));
                        const int v2v_mode = segment.value("v2v_mode", 0);
                        if (v2v_mode != 0 && v2v_mode != 1 && v2v_mode != 2) {
                            res.status = 400;
                            res.set_content("{\"error\":\"LTX v2v_mode must be 0, 1 (SDEdit), or 2 (guide edit)\"}", "application/json");
                            return;
                        }
                        std::string guide_latent_path;
                        const bool has_legacy_guide_path = segment.contains("v2v_source_latent_path");
                        const bool has_guide_reference = segment.contains("v2v_source_job_id") ||
                                                         segment.contains("v2v_source_segment");
                        if (has_legacy_guide_path && has_guide_reference) {
                            res.status = 400;
                            res.set_content(R"({"error":"provide either v2v_source_latent_path or v2v_source_job_id plus v2v_source_segment"})", "application/json");
                            return;
                        }
                        if (has_legacy_guide_path) {
                            if (!segment["v2v_source_latent_path"].is_string() ||
                                !resolve_ltx_guide_latent_path(segment["v2v_source_latent_path"].get<std::string>(), guide_latent_path)) {
                                res.status = 400;
                                res.set_content(R"({"error":"v2v_source_latent_path must be an existing LTX job-bank seg_<n>.bin file"})", "application/json");
                                return;
                            }
                            if (v2v_mode != 2) {
                                res.status = 400;
                                res.set_content(R"({"error":"v2v_source_latent_path requires v2v_mode 2"})", "application/json");
                                return;
                            }
                        }
                        if (has_guide_reference) {
                            if (!segment.contains("v2v_source_job_id") || !segment["v2v_source_job_id"].is_string() ||
                                !segment.contains("v2v_source_segment") || !segment["v2v_source_segment"].is_number_integer() ||
                                !resolve_ltx_guide_latent_reference(segment["v2v_source_job_id"].get<std::string>(),
                                                                    segment["v2v_source_segment"].get<int>(),
                                                                    guide_latent_path)) {
                                res.status = 400;
                                res.set_content(R"({"error":"v2v_source_job_id plus v2v_source_segment must identify an existing LTX banked latent"})", "application/json");
                                return;
                            }
                            if (v2v_mode != 2) {
                                res.status = 400;
                                res.set_content(R"({"error":"v2v_source_job_id requires v2v_mode 2"})", "application/json");
                                return;
                            }
                        }
                        std::vector<std::string> controls;
                        if (segment.contains("control_frames")) {
                            if (!segment["control_frames"].is_array() ||
                                !std::all_of(segment["control_frames"].begin(), segment["control_frames"].end(),
                                             [](const json& frame) { return frame.is_string() && !frame.get<std::string>().empty(); })) {
                                res.status = 400;
                                res.set_content(R"({"error":"LTX control_frames must be a non-empty base64 image array"})", "application/json");
                                return;
                            }
                            for (const auto& frame : segment["control_frames"]) controls.push_back(frame.get<std::string>());
                        }
                        if ((v2v_mode == 1 || v2v_mode == 2) && controls.empty() && guide_latent_path.empty()) {
                            res.status = 400;
                            res.set_content(R"({"error":"LTX V2V requires control_frames or a guide latent path"})", "application/json");
                            return;
                        }
                        if (v2v_mode == 0 && (!controls.empty() || !guide_latent_path.empty())) {
                            res.status = 400;
                            res.set_content(R"({"error":"V2V sources require v2v_mode 1 or 2"})", "application/json");
                            return;
                        }
                        if (!controls.empty() && !guide_latent_path.empty()) {
                            res.status = 400;
                            res.set_content(R"({"error":"guide edit accepts either control_frames or a banked LTX guide source, not both"})", "application/json");
                            return;
                        }
                        const float v2v_strength = segment.value("v2v_guide_strength", -1.f);
                        if (v2v_strength < -1.f || v2v_strength > 1.f) {
                            res.status = 400;
                            res.set_content(R"({"error":"v2v_guide_strength must be between 0 and 1"})", "application/json");
                            return;
                        }
                        segment_control_frames.push_back(std::move(controls));
                        segment_v2v_modes.push_back(v2v_mode);
                        segment_v2v_strengths.push_back(v2v_strength);
                        segment_v2v_guide_latent_paths.push_back(std::move(guide_latent_path));
                    }
                }
            } else if (body.contains("prompts") && body["prompts"].is_array()) {
                for (const auto& prompt : body["prompts"]) {
                    if (prompt.is_string()) {
                        prompts.push_back(prompt.get<std::string>());
                        segment_models.emplace_back();
                        segment_frames.push_back(0);
                        segment_scene_cuts.push_back(0);
                        segment_init_images.emplace_back();
                        segment_keyframes.emplace_back();
                        segment_keyframe_indices.emplace_back();
                        segment_control_frames.emplace_back();
                        segment_v2v_modes.push_back(0);
                        segment_v2v_strengths.push_back(-1.f);
                        segment_v2v_guide_latent_paths.emplace_back();
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
            if (req.is_multipart_form_data()) {
                // Sparse indexed per-shot audio deliberately does not stop at a
                // missing index: Director may replace just shot 7 on a retake.
                for (int index = 0; index < requested_segments; ++index) {
                    const std::string full_key = "audio_full_" + std::to_string(index);
                    const std::string track_key = "audio_track_" + std::to_string(index);
                    if (req.form.has_file(full_key)) {
                        const std::string& bytes = req.form.get_file(full_key).content;
                        if (bytes.empty()) {
                            res.status = 400;
                            res.set_content(json({{"error", full_key + " must be a non-empty WAV upload"}}).dump(), "application/json");
                            return;
                        }
                        shot_audio_full_parts.emplace_back(index, bytes);
                    }
                    if (req.form.has_file(track_key)) {
                        const std::string& bytes = req.form.get_file(track_key).content;
                        if (bytes.empty()) {
                            res.status = 400;
                            res.set_content(json({{"error", track_key + " must be a non-empty WAV upload"}}).dump(), "application/json");
                            return;
                        }
                        shot_audio_track_parts.emplace_back(index, bytes);
                    }
                }
            }
            body["prompt"] = prompts.front();
            if (body.contains("frames")) {
                body["video_frames"] = body["frames"];
            }

            // Koblem's LipDub contract asks for the production two-stage
            // recipe with `two_stage: true`; it does not need to redundantly
            // expose the worker's configured latent-upscaler model. Enable
            // that configured model before parameter resolution so the
            // request reaches the engine with a valid model path.
            if (body.value("two_stage", false)) {
                json& hires = body["hires"];
                if (!hires.is_object()) hires = json::object();
                if (!hires.contains("enabled")) hires["enabled"] = true;
                if (!hires.contains("upscaler")) {
                    hires["upscaler"] = runtime->default_gen_params->hires_upscaler;
                }
            }

            VidGenJobRequest request;
            std::string error_message;
            if (!parse_ltx_video_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }
            // Koblem's LipDub endpoint keeps its source clip at the top level
            // (`control_frames`) and omits v2v_mode: production defines that
            // default as mode 0 (relip), distinct from SDEdit/guide-edit.
            int top_level_v2v_mode = body.value("v2v_mode", 0);
            if (top_level_v2v_mode != 0 && top_level_v2v_mode != 1 && top_level_v2v_mode != 2) {
                res.status = 400;
                res.set_content("{\"error\":\"LTX v2v_mode must be 0 (lipdub), 1 (SDEdit), or 2 (guide edit)\"}",
                                "application/json");
                return;
            }
            int relip_ref_tstride = body.value("relip_ref_tstride", 1);
            if (relip_ref_tstride < 1) {
                res.status = 400;
                res.set_content(R"({"error":"relip_ref_tstride must be positive"})", "application/json");
                return;
            }
            if (top_level_v2v_mode == 0 && !request.gen_params.control_frames.empty() &&
                request.gen_params.control_frames.size() < static_cast<size_t>(request.gen_params.video_frames)) {
                res.status = 400;
                res.set_content(R"({"error":"LTX lipdub requires one control frame per output frame"})", "application/json");
                return;
            }
            std::vector<SDGenerationParams> hires_stages;
            std::vector<sample_method_t> hires_stage_methods;
            std::vector<float> hires_stage_cfgs;
            if (body.contains("hires_chain")) {
                if (!body["hires_chain"].is_array() || body["hires_chain"].empty()) {
                    res.status = 400;
                    res.set_content(R"({"error":"hires_chain must be a non-empty array"})", "application/json");
                    return;
                }
                for (const auto& stage : body["hires_chain"]) {
                    if (!stage.is_object()) {
                        res.status = 400;
                        res.set_content(R"({"error":"each hires_chain stage must be an object"})", "application/json");
                        return;
                    }
                    json stage_body = body;
                    json stage_hires = stage;
                    stage_hires["enabled"] = true;
                    stage_body["hires"] = std::move(stage_hires);
                    stage_body.erase("hires_chain");
                    SDGenerationParams parsed = request.gen_params;
                    if (!parsed.from_json_str(stage_body.dump(), [&](const std::string& path) {
                            return get_lora_full_path(*runtime, path);
                        }) ||
                        !parsed.resolve_and_validate(VID_GEN, "", runtime->ctx_params->hires_upscalers_dir, true)) {
                        res.status = 400;
                        res.set_content(R"({"error":"invalid hires_chain stage"})", "application/json");
                        return;
                    }
                    sample_method_t method = SAMPLE_METHOD_COUNT;
                    if (stage.contains("sample_method")) {
                        if (!stage["sample_method"].is_string()) {
                            res.status = 400;
                            res.set_content(R"({"error":"hires_chain sample_method must be a string"})", "application/json");
                            return;
                        }
                        method = str_to_sample_method(stage["sample_method"].get<std::string>().c_str());
                        if (method == SAMPLE_METHOD_COUNT) {
                            res.status = 400;
                            res.set_content(R"({"error":"invalid hires_chain sample_method"})", "application/json");
                            return;
                        }
                    }
                    float cfg = NAN;
                    if (stage.contains("cfg")) {
                        if (!stage["cfg"].is_number()) {
                            res.status = 400;
                            res.set_content(R"({"error":"hires_chain cfg must be numeric"})", "application/json");
                            return;
                        }
                        cfg = stage["cfg"].get<float>();
                    }
                    hires_stages.push_back(std::move(parsed));
                    hires_stage_methods.push_back(method);
                    hires_stage_cfgs.push_back(cfg);
                }
            }
            const std::string default_model = body.value("model", std::string("base"));
            const auto variants = runtime_diffusion_model_variants(*runtime);
            if (variants.find(default_model) == variants.end()) {
                res.status = 400;
                res.set_content(json({{"error", "unknown LTX model variant '" + default_model + "'"}}).dump(), "application/json");
                return;
            }
            for (size_t index = 0; index < segment_models.size(); ++index) {
                if (segment_models[index].empty()) segment_models[index] = default_model;
                if (variants.find(segment_models[index]) == variants.end()) {
                    res.status = 400;
                    res.set_content(json({{"error", "unknown LTX segment model variant '" + segment_models[index] +
                                                       "' for segment " + std::to_string(index + 1)}}).dump(),
                                    "application/json");
                    return;
                }
            }
            for (size_t segment = 0; segment < segment_v2v_modes.size(); ++segment) {
                if (segment_v2v_modes[segment] != 1 &&
                    !(segment_v2v_modes[segment] == 2 && segment_v2v_guide_latent_paths[segment].empty())) continue;
                const int frames = segment_frames[segment] > 0 ? segment_frames[segment] : request.gen_params.video_frames;
                if (static_cast<int>(segment_control_frames[segment].size()) != frames) {
                    res.status = 400;
                    res.set_content(R"({"error":"LTX SDEdit requires one control frame per segment output frame"})", "application/json");
                    return;
                }
            }
            for (size_t segment = 0; segment < segment_keyframe_indices.size(); ++segment) {
                const int frames = segment_frames[segment] > 0 ? segment_frames[segment] : request.gen_params.video_frames;
                for (const int frame : segment_keyframe_indices[segment]) {
                    if (frame < 0 || frame >= frames) {
                        res.status = 400;
                        res.set_content(json({{"error", "LTX keyframe frame must be within segment " +
                                                         std::to_string(segment + 1) + " output frames"}}).dump(),
                                        "application/json");
                        return;
                    }
                }
            }
            const int continuation_frames = body.value("cont_latent_frames", 3);
            const bool single_lipdub = top_level_v2v_mode == 0 && !request.gen_params.control_frames.empty() &&
                                       requested_segments == 1;
            if (continuation_frames < 0 || (continuation_frames == 0 && !single_lipdub)) {
                res.status = 400;
                res.set_content(R"({"error":"cont_latent_frames must be positive except for a single lipdub window"})", "application/json");
                return;
            }

            AsyncJobManager& manager = *runtime->async_job_manager;
            auto job = std::make_shared<AsyncGenerationJob>();
            job->kind = AsyncJobKind::VidGen;
            job->status = AsyncJobStatus::Queued;
            job->created_at = unix_timestamp_now();
            job->vid_gen = std::move(request);
            job->ltx_v2v_mode = top_level_v2v_mode;
            job->ltx_relip_ref_tstride = relip_ref_tstride;
            job->ltx_hires_stages = std::move(hires_stages);
            job->ltx_hires_stage_methods = std::move(hires_stage_methods);
            job->ltx_hires_stage_cfgs = std::move(hires_stage_cfgs);
            job->ltx_emit_stages = body.value("emit_stages", false);
            job->ltx_prompts = std::move(prompts);
            job->ltx_segment_models = std::move(segment_models);
            job->ltx_default_model = default_model;
            job->ltx_segment_frames = std::move(segment_frames);
            job->ltx_segment_scene_cuts = std::move(segment_scene_cuts);
            job->ltx_segment_init_images = std::move(segment_init_images);
            job->ltx_segment_keyframes = std::move(segment_keyframes);
            job->ltx_segment_keyframe_indices = std::move(segment_keyframe_indices);
            job->ltx_segment_control_frames = std::move(segment_control_frames);
            job->ltx_segment_v2v_modes = std::move(segment_v2v_modes);
            job->ltx_segment_v2v_strengths = std::move(segment_v2v_strengths);
            job->ltx_segment_v2v_guide_latent_paths = std::move(segment_v2v_guide_latent_paths);
            job->ltx_cont_latent_frames = continuation_frames;
            job->ltx_emit_segments = body.value("emit_segments", false);
            const bool persist_bank = body.value("persist", false);
            const std::string resume_job_id = body.value("resume_job_id", std::string());
            const int retake_segment = body.value("retake_segment", body.value("retake_from", -1));
            if (retake_segment < -1 || retake_segment >= requested_segments) {
                res.status = 400;
                res.set_content(R"({"error":"retake_segment must name an existing segment"})", "application/json");
                return;
            }
            if (retake_segment >= 0 && resume_job_id.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"retake_segment requires resume_job_id"})", "application/json");
                return;
            }
            if (body.contains("cont_seam_drop_frames") && !body["cont_seam_drop_frames"].is_number_integer()) {
                res.status = 400;
                res.set_content(R"({"error":"cont_seam_drop_frames must be an integer"})", "application/json");
                return;
            }
            const int cont_seam_drop_frames = body.value("cont_seam_drop_frames", 0);
            std::vector<int> segment_seam_drop_frames;
            if (body.contains("segment_seam_drop_frames")) {
                if (!body["segment_seam_drop_frames"].is_array()) {
                    res.status = 400;
                    res.set_content(R"({"error":"segment_seam_drop_frames must be an array"})", "application/json");
                    return;
                }
                segment_seam_drop_frames.assign(static_cast<size_t>(requested_segments), -1);
                const auto& drops = body["segment_seam_drop_frames"];
                for (size_t index = 0; index < drops.size() && index < segment_seam_drop_frames.size(); ++index) {
                    if (!drops[index].is_number_integer()) {
                        res.status = 400;
                        res.set_content(R"({"error":"segment_seam_drop_frames entries must be integers"})", "application/json");
                        return;
                    }
                    segment_seam_drop_frames[index] = drops[index].get<int>();
                }
            }
            if (body.contains("audio_offset_frames") && !body["audio_offset_frames"].is_number_integer()) {
                res.status = 400;
                res.set_content(R"({"error":"audio_offset_frames must be an integer"})", "application/json");
                return;
            }
            const bool has_audio_offset_frames = body.contains("audio_offset_frames");
            const int audio_offset_frames = body.value("audio_offset_frames", 0);
            if (audio_offset_frames < 0) {
                res.status = 400;
                res.set_content(R"({"error":"audio_offset_frames must not be negative"})", "application/json");
                return;
            }
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
                fs::path bank_root;
                std::string bank_id;
                if (!resume_job_id.empty()) {
                    if (!drive_audio_bytes.empty() || !track_audio_bytes.empty() || !legacy_audio_parts.empty() ||
                        !shot_audio_full_parts.empty() || !shot_audio_track_parts.empty()) {
                        res.status = 400;
                        res.set_content(R"({"error":"resumed LTX jobs reuse their banked audio; do not upload new audio"})", "application/json");
                        return;
                    }
                    if (!resolve_ltx_bank_dir(resume_job_id, bank_dir, bank_id, &bank_root)) {
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
                    if (resume_from <= 0 || (resume_from >= static_cast<int>(job->ltx_prompts.size()) && retake_segment < 0)) {
                        res.status = 404;
                        res.set_content(R"({"error":"resume_job_id has no resumable LTX latent bank"})", "application/json");
                        return;
                    }
                    if (retake_segment >= 0) {
                        if (retake_segment >= resume_from) {
                            res.status = 404;
                            res.set_content(R"({"error":"retake_segment is not present in the LTX latent bank"})", "application/json");
                            return;
                        }
                        resume_from = retake_segment;
                    }
                } else {
                    job->id = make_async_job_id(manager);
                    bank_root = persist_bank ? ltx_persist_root() : ltx_bank_root();
                    bank_dir = bank_root / job->id;
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
                if (!write_ltx_bank_reference(bank_root, job->id, bank_id)) {
                    res.status = 500;
                    res.set_content(R"({"error":"could not persist LTX bank reference"})", "application/json");
                    return;
                }
                auto stage_audio = [&](const std::string& bytes,
                                       const char* filename,
                                       std::string& staged_path) -> bool {
                    if (bytes.empty()) return true;
                    const fs::path path = bank_dir / filename;
                    std::ofstream output(path, std::ios::binary | std::ios::trunc);
                    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                    if (!output.good()) return false;
                    staged_path = path.string();
                    return true;
                };
                if (!resume_job_id.empty()) {
                    std::error_code audio_dir_error;
                    const fs::path legacy_dir = bank_dir / "audio";
                    if (fs::is_directory(legacy_dir, audio_dir_error)) {
                        job->ltx_chain_audio_dir = legacy_dir.string();
                    }
                    const fs::path drive_path = bank_dir / "audio_full.wav";
                    const fs::path track_path = bank_dir / "audio_track.wav";
                    std::error_code audio_error;
                    if (fs::is_regular_file(drive_path, audio_error)) job->ltx_chain_audio_full = drive_path.string();
                    audio_error.clear();
                    if (fs::is_regular_file(track_path, audio_error)) job->ltx_chain_audio_track = track_path.string();
                    std::ifstream offset_file(bank_dir / "audio_offset_frames");
                    int stored_offset = 0;
                    if (offset_file >> stored_offset) {
                        if (stored_offset < 0 || (has_audio_offset_frames && stored_offset != audio_offset_frames)) {
                            res.status = 400;
                            res.set_content(R"({"error":"resume audio_offset_frames does not match the banked audio timeline"})", "application/json");
                            return;
                        }
                        job->ltx_chain_audio_offset_frames = stored_offset;
                    }
                } else if (!stage_audio(drive_audio_bytes, "audio_full.wav", job->ltx_chain_audio_full) ||
                           !stage_audio(track_audio_bytes, "audio_track.wav", job->ltx_chain_audio_track)) {
                    std::error_code remove_error;
                    fs::remove(bank_dir / "audio_full.wav", remove_error);
                    remove_error.clear();
                    fs::remove(bank_dir / "audio_track.wav", remove_error);
                    res.status = 500;
                    res.set_content(R"({"error":"could not stage LTX audio upload"})", "application/json");
                    return;
                } else if (!legacy_audio_parts.empty()) {
                    const fs::path legacy_dir = bank_dir / "audio";
                    std::error_code audio_dir_error;
                    fs::create_directories(legacy_dir, audio_dir_error);
                    if (audio_dir_error) {
                        res.status = 500;
                        res.set_content(R"({"error":"could not create LTX legacy audio directory"})", "application/json");
                        return;
                    }
                    for (const auto& [index, bytes] : legacy_audio_parts) {
                        std::ofstream output(legacy_dir / ("aud_" + std::to_string(index) + ".wav"),
                                             std::ios::binary | std::ios::trunc);
                        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                        if (!output.good()) {
                            res.status = 500;
                            res.set_content(R"({"error":"could not stage LTX legacy audio upload"})", "application/json");
                            return;
                        }
                    }
                    job->ltx_chain_audio_dir = legacy_dir.string();
                } else if (!drive_audio_bytes.empty() || !track_audio_bytes.empty()) {
                    std::ofstream offset_file(bank_dir / "audio_offset_frames", std::ios::trunc);
                    offset_file << audio_offset_frames << '\n';
                    if (!offset_file.good()) {
                        std::error_code remove_error;
                        fs::remove(bank_dir / "audio_full.wav", remove_error);
                        remove_error.clear();
                        fs::remove(bank_dir / "audio_track.wav", remove_error);
                        res.status = 500;
                        res.set_content(R"({"error":"could not persist LTX audio timeline metadata"})", "application/json");
                        return;
                    }
                }
                // Per-shot audio is stored independently of the legacy audio_<n>
                // directory. On resume it is rediscovered so retakes never need a
                // second upload and sparse shot indices stay sparse.
                const fs::path shot_audio_dir = bank_dir / "audio";
                if (!resume_job_id.empty() || !shot_audio_full_parts.empty() || !shot_audio_track_parts.empty()) {
                    std::error_code shot_error;
                    if (!shot_audio_full_parts.empty() || !shot_audio_track_parts.empty()) {
                        fs::create_directories(shot_audio_dir, shot_error);
                        if (shot_error) {
                            res.status = 500;
                            res.set_content(R"({"error":"could not create LTX per-shot audio directory"})", "application/json");
                            return;
                        }
                        for (const auto& [index, bytes] : shot_audio_full_parts) {
                            std::ofstream output(shot_audio_dir / ("shot_" + std::to_string(index) + "_full.wav"), std::ios::binary | std::ios::trunc);
                            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                            if (!output.good()) { res.status = 500; res.set_content(R"({"error":"could not stage LTX per-shot audio upload"})", "application/json"); return; }
                        }
                        for (const auto& [index, bytes] : shot_audio_track_parts) {
                            std::ofstream output(shot_audio_dir / ("shot_" + std::to_string(index) + "_track.wav"), std::ios::binary | std::ios::trunc);
                            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                            if (!output.good()) { res.status = 500; res.set_content(R"({"error":"could not stage LTX per-shot audio upload"})", "application/json"); return; }
                        }
                    }
                    job->ltx_segment_audio_full.assign(static_cast<size_t>(requested_segments), std::string());
                    job->ltx_segment_audio_track.assign(static_cast<size_t>(requested_segments), std::string());
                    for (int index = 0; index < requested_segments; ++index) {
                        const fs::path full = shot_audio_dir / ("shot_" + std::to_string(index) + "_full.wav");
                        const fs::path track = shot_audio_dir / ("shot_" + std::to_string(index) + "_track.wav");
                        if (fs::is_regular_file(full, shot_error)) job->ltx_segment_audio_full[index] = full.string();
                        shot_error.clear();
                        if (fs::is_regular_file(track, shot_error)) job->ltx_segment_audio_track[index] = track.string();
                    }
                }
                job->ltx_bank_dir = bank_dir.string();
                job->ltx_bank_id = bank_id;
                job->ltx_resume_from = resume_from;
                job->ltx_retake_segment = retake_segment;
                job->ltx_cont_seam_drop_frames = cont_seam_drop_frames;
                job->ltx_segment_seam_drop_frames = std::move(segment_seam_drop_frames);
                if (job->ltx_chain_audio_offset_frames == 0) {
                    job->ltx_chain_audio_offset_frames = audio_offset_frames;
                }
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
                                  {"retake_segment", retake_segment},
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

    // Progressive LTX shot artifact.  This is deliberately durable-bank based
    // rather than RAM-job based, so Koblem can retrieve a published segment
    // after its in-memory job record has expired or after a route restart.
    svr.Get(R"(/sdcpp/v1/jobs/([^/]+)/segments/(\d+))",
            [](const httplib::Request& req, httplib::Response& res) {
        const std::string job_id = req.matches[1];
        const std::string segment = req.matches[2];
        fs::path bank_dir;
        std::string bank_id;
        if (!resolve_ltx_bank_dir(job_id, bank_dir, bank_id)) {
            res.status = 404;
            res.set_content(R"({"error":"unknown LTX job"})", "application/json");
            return;
        }
        fs::path artifact = bank_dir / ("seg_" + segment + ".webm");
        const std::string stage = req.get_param_value("stage");
        if (!stage.empty() && stage != "4" &&
            stage.find_first_not_of("0123456789") == std::string::npos) {
            artifact = bank_dir / ("seg_" + segment + "_stage" + stage + ".webm");
        }
        std::error_code error;
        if (!fs::is_regular_file(artifact, error)) {
            res.status = 404;
            res.set_content(R"({"error":"segment webm not available"})", "application/json");
            return;
        }
        std::ifstream input(artifact, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        res.set_content(bytes, "video/webm");
    });

    // Explicit lifecycle cleanup for Koblem Director projects. The client only
    // sends an opaque engine job id; never accept a host path. A durable bank
    // may have a resume alias, so remove the requested alias and its bank
    // payload together.
    svr.Delete("/ltx/v1/job", [runtime](const httplib::Request& req, httplib::Response& res) {
        const std::string job_id = req.get_param_value("id");
        if (!valid_ltx_bank_id(job_id)) {
            res.status = 400;
            res.set_content(R"({"error":"id must be an LTX job id"})", "application/json");
            return;
        }
        {
            AsyncJobManager& manager = *runtime->async_job_manager;
            std::lock_guard<std::mutex> lock(manager.mutex);
            const auto it = manager.jobs.find(job_id);
            if (it != manager.jobs.end() &&
                (it->second->status == AsyncJobStatus::Queued || it->second->status == AsyncJobStatus::Generating)) {
                res.status = 409;
                res.set_content(R"({"error":"cannot delete an active LTX job"})", "application/json");
                return;
            }
        }

        fs::path bank_dir;
        fs::path bank_root;
        std::string bank_id;
        if (!resolve_ltx_bank_dir(job_id, bank_dir, bank_id, &bank_root)) {
            res.set_content(json({{"status", "missing"}, {"deleted", false}}).dump(), "application/json");
            return;
        }
        std::error_code error;
        uintmax_t removed = 0;
        for (const fs::path& root : ltx_bank_roots()) {
            if (root == bank_root) {
                removed += fs::remove_all(root / bank_id, error);
                if (error) break;
            }
            if (job_id != bank_id) {
                error.clear();
                removed += fs::remove_all(root / job_id, error);
                if (error) break;
            }
        }
        if (error) {
            res.status = 500;
            res.set_content(json({{"error", "could not delete LTX job bank"}, {"message", error.message()}}).dump(),
                            "application/json");
            return;
        }
        res.set_content(json({{"status", "deleted"}, {"deleted", removed > 0}}).dump(), "application/json");
    });
}
