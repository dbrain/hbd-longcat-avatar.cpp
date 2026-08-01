#include "routes.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
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

// `reference_head_trim` wire encoding: 0 = off, -1 = AUTO (the engine derives 1 + 8*(K-1) from the
// references it actually encoded), >0 = trim exactly that many pixel frames. The ceiling only has
// to stop a units mistake reaching the engine -- the real bound is the shot's own length, which
// only the engine knows, and which it enforces by refusing to leave a shot with no frames.
constexpr int kMaxReferenceHeadTrim = 512;

bool valid_reference_head_trim(int value) {
    return value >= -1 && value <= kMaxReferenceHeadTrim;
}

std::string reference_head_trim_error(const std::string& field) {
    return R"({"error":")" + field + " must be -1 (auto), 0 (off), or 1.." +
           std::to_string(kMaxReferenceHeadTrim) + R"( frames"})";
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

// Is a resumed/retaken job allowed to FORK its own bank instead of writing into
// the one it resumed? LTX_BANK_FORK=0 restores the old shared-bank behaviour on
// the same binary. Reads the VALUE, not merely the presence.
bool ltx_bank_fork_enabled() {
    if (const char* env = std::getenv("LTX_BANK_FORK"); env != nullptr && env[0] != '\0') {
        return !(env[0] == '0' && env[1] == '\0');
    }
    return true;
}

// Give a resumed job its OWN bank, sharing every unchanged shot with the bank it
// resumed by HARD LINK.
//
// ★ WHY THIS EXISTS — measured, not theorised. Before this, a retake resolved
// `bank_dir` to the RESUMED job's directory and re-rendered shot i straight over
// `seg_<i>.bin` there, while `write_ltx_bank_reference` made the new job id a
// mere pointer back to the same bank. So a retake DESTROYED the take it
// replaced: rendering a 2-shot chain and retaking shot 0 leaves seg_0.bin and
// seg_0.audio with new contents and nowhere holding the old ones. Every take's
// `latent_job_id` resolved to one directory, so a UI that records which take a
// shot used was recording a distinction that did not exist on disk.
//
// Copying is not an option: a production 42-shot 1920x1088 bank is 929 MB and
// the store is ext4 (no reflinks), so an overnight critic run at two retakes a
// shot would write ~78 GB. Hard links make the fork cost one directory entry per
// file and zero data blocks — the shots this job will NOT re-render are
// genuinely the same bytes, which is exactly what a hard link says.
//
// The re-rendered shots' links are then REMOVED, so when the render opens
// `seg_<i>.bin` for writing it creates a fresh inode and the resumed bank's copy
// is left untouched. Unlinking first is the whole trick — writing in place
// through a shared inode would corrupt the other take rather than fork from it.
bool fork_ltx_bank(const fs::path& resume_bank,
                   const fs::path& destination,
                   const std::vector<std::string>& segment_bank_dirs,
                   int n_segments,
                   const std::vector<int>& rerendered_segments,
                   std::string& why_not) {
    std::error_code error;
    fs::create_directories(destination, error);
    if (error) {
        why_not = "could not create the forked bank directory: " + error.message();
        return false;
    }

    auto link_or_copy = [&](const fs::path& from, const fs::path& to) -> bool {
        std::error_code remove_error;
        fs::remove(to, remove_error);
        std::error_code link_error;
        fs::create_hard_link(from, to, link_error);
        if (!link_error) return true;
        // Cross-device, or a filesystem without links: a real copy is still
        // correct, just expensive. Only the fallback pays for the bytes.
        std::error_code copy_error;
        fs::copy_file(from, to, fs::copy_options::overwrite_existing, copy_error);
        if (copy_error) {
            why_not = "could not link or copy '" + from.filename().string() + "': " + copy_error.message();
            return false;
        }
        return true;
    };

    // Everything that is NOT a per-shot artefact — the staged drive/track WAVs,
    // audio_offset_frames, the legacy audio/ slice directory — comes from the
    // resumed job wholesale. `bank_id` is deliberately NOT carried over: this is a
    // real bank now, and copying the pointer would make it resolve elsewhere.
    for (const auto& entry : fs::recursive_directory_iterator(resume_bank, error)) {
        if (error) break;
        const fs::path relative = entry.path().lexically_relative(resume_bank);
        const std::string top = relative.begin()->string();
        if (top == "bank_id" || top.rfind("seg_", 0) == 0) continue;
        if (entry.is_directory(error)) {
            std::error_code dir_error;
            fs::create_directories(destination / relative, dir_error);
            continue;
        }
        if (!entry.is_regular_file(error)) continue;
        if (!link_or_copy(entry.path(), destination / relative)) return false;
    }
    if (error) {
        why_not = "could not read the source bank: " + error.message();
        return false;
    }

    // ★ Per-shot artefacts come from THE BANK THAT SHOT IS RESTORED FROM, not
    // blanket from the resumed job.
    //
    // This is what makes the forked bank a faithful record of what was actually
    // rendered. A mixture render — shot 0 out of take A, shot 1 out of take C —
    // restores each shot from its own bank; if the fork copied both from the
    // resumed job instead, the finished bank would describe a DIFFERENT mixture
    // than the video it accompanies, and the next resume of this job would
    // silently render a combination the user never picked. The output would look
    // right and the bank would lie.
    //
    // Shots this job RE-RENDERS are skipped entirely: no link is made, so the
    // render's own write creates a fresh inode and no other take is touched.
    for (int segment = 0; segment < n_segments; ++segment) {
        if (std::find(rerendered_segments.begin(), rerendered_segments.end(), segment) !=
            rerendered_segments.end()) {
            continue;
        }
        fs::path source = resume_bank;
        if (segment < static_cast<int>(segment_bank_dirs.size()) && !segment_bank_dirs[segment].empty()) {
            source = segment_bank_dirs[segment];
        }
        // The trailing '.' is load-bearing: without it "seg_1" would also match
        // "seg_10.bin" and a ten-shot chain would fork the wrong artefacts.
        const std::string stem = "seg_" + std::to_string(segment) + ".";
        std::error_code scan_error;
        for (const auto& entry : fs::directory_iterator(source, scan_error)) {
            if (scan_error) break;
            if (!entry.is_regular_file(scan_error)) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind(stem, 0) != 0) continue;
            if (!link_or_copy(entry.path(), destination / name)) return false;
        }
    }
    return true;
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
            // Per-segment runtime LoRAs, resolved here to full paths (empty entry = inherit).
            std::vector<std::vector<std::pair<std::string, float>>> segment_loras;
            std::vector<int> segment_frames;
            std::vector<int> segment_scene_cuts;
            std::vector<std::string> segment_init_images;
            std::vector<std::vector<std::string>> segment_keyframes;
            std::vector<std::vector<int>> segment_keyframe_indices;
            std::vector<std::vector<std::string>> segment_control_frames;
            std::vector<int> segment_v2v_modes;
            std::vector<float> segment_v2v_strengths;
            // Per-shot image-pin hold strength (init image / keyframes / end image), -1 = inherit
            // the request's top-level `strength`. Only staged onto the job if some shot actually
            // names one, so a request that never mentions pin_strength reaches the core with a
            // null pointer -- byte-identical to before this existed.
            std::vector<float> segment_pin_strengths;
            std::vector<std::string> segment_v2v_guide_latent_paths;
            // Per-shot "restore THIS shot from THAT job's bank" — the take the caller picked for
            // this shot. Collected as raw job ids and resolved to directories once, below, so the
            // core is only ever handed resolved paths.
            std::vector<std::string> segment_bank_job_ids;
            std::vector<std::vector<LtxSegmentBeat>> segment_beats;
            std::vector<int64_t> segment_seeds;
            std::vector<int> segment_steps;
            std::vector<float> segment_cfg;
            std::vector<std::string> segment_negative_prompts;
            // Per-shot reference_head_trim, collected with an "absent" sentinel because the
            // top-level default it inherits from is not parsed until ~500 lines below. Resolved
            // into the wire encoding (0 off / -1 auto / >0 explicit) once both are known.
            constexpr int kHeadTrimInherit = std::numeric_limits<int>::min();
            std::vector<int> segment_reference_head_trim;
            bool any_segment_head_trim = false;
            bool any_segment_beats     = false;
            bool any_segment_overrides = false;
            // Only needed to turn a beat's `time` (seconds) into a frame index; the authoritative
            // fps is parsed later with the rest of the generation params.
            int beat_fps = body.value("fps", 24);
            if (beat_fps <= 0) {
                beat_fps = 24;
            }
            // Per-segment `lora` names are resolved inside the loop below, but the cache they
            // resolve against is only refreshed by parse_ltx_video_request(), ~300 lines later.
            // On a cold process it is therefore EMPTY here and every per-shot adapter 400s with
            // "unknown LTX segment lora" -- including names that demonstrably work at request
            // root. It only ever appeared to work when some earlier request in the same process
            // had already warmed the cache, which is why this survived being tested.
            refresh_lora_cache(*runtime);
            if (body.contains("segments") && body["segments"].is_array()) {
                for (const auto& segment : body["segments"]) {
                    if (segment.is_string()) {
                        prompts.push_back(segment.get<std::string>());
                        segment_models.emplace_back();
                        segment_loras.emplace_back();   // keep per-segment arrays aligned
                        segment_frames.push_back(0);
                        segment_scene_cuts.push_back(0);
                        segment_init_images.emplace_back();
                        segment_keyframes.emplace_back();
                        segment_keyframe_indices.emplace_back();
                        segment_control_frames.emplace_back();
                        segment_v2v_modes.push_back(0);
                        segment_v2v_strengths.push_back(-1.f);
                        segment_pin_strengths.push_back(-1.f);
                        segment_v2v_guide_latent_paths.emplace_back();
                        segment_bank_job_ids.emplace_back();
                        segment_beats.emplace_back();
                        segment_seeds.push_back(-1);
                        segment_steps.push_back(0);
                        segment_cfg.push_back(-1.f);
                        segment_negative_prompts.emplace_back();
                        segment_reference_head_trim.push_back(kHeadTrimInherit);
                    } else if (segment.is_object()) {
                        prompts.push_back(segment.value("prompt", std::string()));
                        if (segment.contains("model") && !segment["model"].is_string()) {
                            res.status = 400;
                            res.set_content(R"({"error":"each LTX segment model must be a string"})", "application/json");
                            return;
                        }
                        segment_models.push_back(segment.value("model", std::string()));
                        // Per-shot adapters. Same object shape as the request-root `lora`
                        // array, and resolved through the SAME cache, so a bare stem (no
                        // extension) hard-fails here instead of silently rendering the shot
                        // without its adapter.
                        std::vector<std::pair<std::string, float>> seg_loras;
                        if (segment.contains("lora") && !segment["lora"].is_null()) {
                            if (!segment["lora"].is_array()) {
                                res.status = 400;
                                res.set_content(R"({"error":"each LTX segment lora must be an array"})", "application/json");
                                return;
                            }
                            for (const auto& item : segment["lora"]) {
                                if (!item.is_object()) {
                                    res.status = 400;
                                    res.set_content(R"({"error":"each LTX segment lora entry must be an object"})", "application/json");
                                    return;
                                }
                                const std::string lora_path = item.value("path", std::string());
                                const std::string resolved = lora_path.empty() ? std::string()
                                                                               : get_lora_full_path(*runtime, lora_path);
                                if (resolved.empty()) {
                                    res.status = 400;
                                    res.set_content(json({{"error", "unknown LTX segment lora '" + lora_path + "'"}}).dump(),
                                                    "application/json");
                                    return;
                                }
                                seg_loras.emplace_back(resolved, item.value("multiplier", 1.0f));
                            }
                        }
                        segment_loras.push_back(std::move(seg_loras));
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
                        // How hard THIS shot holds its pinned image(s) -- the init image just
                        // parsed, the keyframes above, an end image. Absent (and only absent)
                        // inherits the request's top-level `strength`; the -1 sentinel is the
                        // same one `v2v_guide_strength` uses two blocks down, deliberately, so
                        // there is one convention for "this shot did not say".
                        //
                        // 0 is a LEGAL value and means something (the pin is fully re-denoised),
                        // which is why the sentinel is negative rather than zero and why the
                        // range check rejects anything below it instead of clamping.
                        float pin_strength = -1.f;
                        if (segment.contains("pin_strength") && !segment["pin_strength"].is_null()) {
                            if (!segment["pin_strength"].is_number()) {
                                res.status = 400;
                                res.set_content(R"({"error":"each LTX segment pin_strength must be a number between 0 and 1"})", "application/json");
                                return;
                            }
                            pin_strength = segment["pin_strength"].get<float>();
                            if (pin_strength < 0.f || pin_strength > 1.f) {
                                res.status = 400;
                                res.set_content(R"({"error":"LTX segment pin_strength must be between 0 and 1"})", "application/json");
                                return;
                            }
                        }
                        segment_pin_strengths.push_back(pin_strength);
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
                        // Which TAKE this shot is restored from. `bank_job_id` names the job whose
                        // bank holds the take the caller selected for this shot; absent means "the
                        // resumed job's own bank", i.e. exactly today's behaviour. Only meaningful
                        // for shots this render RESTORES rather than re-renders.
                        segment_bank_job_ids.push_back(segment.value("bank_job_id", std::string()));

                        // Prompt Relay beats. The shot's own `prompt` stays the
                        // global setting; each beat is a short clause pinned to a
                        // frame on this shot's rendered timeline. Zero beats is
                        // the ordinary byte-identical path.
                        std::vector<LtxSegmentBeat> beats;
                        if (segment.contains("beats")) {
                            if (!segment["beats"].is_array()) {
                                res.status = 400;
                                res.set_content(R"({"error":"LTX beats must be an array of {frame, text}"})", "application/json");
                                return;
                            }
                            for (const auto& beat : segment["beats"]) {
                                // `time` (seconds into the shot as the viewer sees it) is the
                                // natural unit for a caller; `frame` stays accepted for exact
                                // control. Both are on the shot's VISIBLE timeline -- the engine
                                // adds a continuation shot's seam drop itself.
                                const bool has_time = beat.is_object() && beat.contains("time") &&
                                                      beat["time"].is_number();
                                const bool has_frame = beat.is_object() && beat.contains("frame") &&
                                                       beat["frame"].is_number_integer();
                                if (!beat.is_object() || (!has_time && !has_frame) ||
                                    (has_frame && beat["frame"].get<int>() < 0) ||
                                    (has_time && beat["time"].get<double>() < 0.0) ||
                                    !beat.contains("text") || !beat["text"].is_string() ||
                                    beat["text"].get<std::string>().empty()) {
                                    res.status = 400;
                                    res.set_content(R"({"error":"each LTX beat needs a non-negative time (seconds) or frame, and non-empty text"})", "application/json");
                                    return;
                                }
                                LtxSegmentBeat parsed;
                                parsed.frame = has_frame
                                                   ? beat["frame"].get<int>()
                                                   : static_cast<int>(std::lround(beat["time"].get<double>() * beat_fps));
                                parsed.text     = beat["text"].get<std::string>();
                                parsed.strength = beat.value("strength", 0.f);
                                parsed.window   = beat.value("window", -1.f);
                                beats.push_back(std::move(parsed));
                            }
                            if (!beats.empty() && segment.value("prompt", std::string()).empty()) {
                                res.status = 400;
                                res.set_content(R"({"error":"LTX beats require a non-empty shot prompt to anchor them"})", "application/json");
                                return;
                            }
                        }
                        any_segment_beats = any_segment_beats || !beats.empty();
                        segment_beats.push_back(std::move(beats));

                        // Per-shot sampling overrides.
                        const int64_t shot_seed = segment.value("seed", static_cast<int64_t>(-1));
                        const int shot_steps    = segment.value("steps", 0);
                        const float shot_cfg    = segment.value("cfg", -1.f);
                        if (shot_steps < 0 || shot_steps > 200) {
                            res.status = 400;
                            res.set_content(R"({"error":"LTX segment steps must be between 0 (inherit) and 200"})", "application/json");
                            return;
                        }
                        segment_seeds.push_back(shot_seed);
                        segment_steps.push_back(shot_steps);
                        segment_cfg.push_back(shot_cfg);
                        segment_negative_prompts.push_back(segment.value("negative_prompt", std::string()));
                        any_segment_overrides = any_segment_overrides ||
                                                shot_seed >= 0 || shot_steps > 0 || shot_cfg >= 0.f ||
                                                !segment_negative_prompts.back().empty();

                        // Per-shot reference head-frame trim. ABSENT inherits the request-level
                        // value; PRESENT is authoritative for this shot, zero included, so one
                        // shot of a project that has the trim on can switch it off.
                        int shot_head_trim = kHeadTrimInherit;
                        if (segment.contains("reference_head_trim") &&
                            !segment["reference_head_trim"].is_null()) {
                            if (!segment["reference_head_trim"].is_number_integer()) {
                                res.status = 400;
                                res.set_content(R"({"error":"each LTX segment reference_head_trim must be an integer: 0 off, -1 auto, >0 frames"})",
                                                "application/json");
                                return;
                            }
                            shot_head_trim = segment["reference_head_trim"].get<int>();
                            if (!valid_reference_head_trim(shot_head_trim)) {
                                res.status = 400;
                                res.set_content(reference_head_trim_error("LTX segment reference_head_trim"),
                                                "application/json");
                                return;
                            }
                            any_segment_head_trim = true;
                        }
                        segment_reference_head_trim.push_back(shot_head_trim);
                    }
                }
            } else if (body.contains("prompts") && body["prompts"].is_array()) {
                for (const auto& prompt : body["prompts"]) {
                    if (prompt.is_string()) {
                        prompts.push_back(prompt.get<std::string>());
                        segment_models.emplace_back();
                        segment_loras.emplace_back();   // keep per-segment arrays aligned
                        segment_frames.push_back(0);
                        segment_scene_cuts.push_back(0);
                        segment_init_images.emplace_back();
                        segment_keyframes.emplace_back();
                        segment_keyframe_indices.emplace_back();
                        segment_control_frames.emplace_back();
                        segment_v2v_modes.push_back(0);
                        segment_v2v_strengths.push_back(-1.f);
                        segment_pin_strengths.push_back(-1.f);
                        segment_v2v_guide_latent_paths.emplace_back();
                        segment_bank_job_ids.emplace_back();
                        segment_beats.emplace_back();
                        segment_seeds.push_back(-1);
                        segment_steps.push_back(0);
                        segment_cfg.push_back(-1.f);
                        segment_negative_prompts.emplace_back();
                        segment_reference_head_trim.push_back(kHeadTrimInherit);
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

            // SEGMENT 0 HAS NO PER-SEGMENT SCENE IMAGE -- its opener is the request's TOP-LEVEL
            // `init_image`. That is deliberate on both sides: segment 0 is always a fresh scene, so
            // `generate_video_chain` gates the per-segment image behind `segment > 0`
            // (stable-diffusion.cpp, `fresh_scene`) and the job staging below only decodes indices
            // >= 1. Do NOT relax either gate; two fields fighting over the same opener with no
            // defined precedence is worse than one field.
            //
            // The trap is that `segments[0].init_image` still PARSES. It lands in
            // segment_init_images[0], is never read, and the render succeeds looking plausible
            // while the reference never conditioned anything -- measured at 10.42 dB on frame 0
            // against the source image, where a working i2v scores 28-39 dB. A caller cannot tell
            // that apart from success, so answer it here rather than let it ship a silent t2v.
            //
            // Fold when the opener slot is free (the caller's intent is unambiguous); refuse when
            // both are set, because picking a winner would be guessing at which image the caller
            // meant to open on.
            if (body.contains("segments") && body["segments"].is_array() && !body["segments"].empty() &&
                body["segments"][0].is_object() && body["segments"][0].contains("init_image") &&
                body["segments"][0]["init_image"].is_string() &&
                !body["segments"][0]["init_image"].get<std::string>().empty()) {
                const bool has_top_level = body.contains("init_image") && body["init_image"].is_string() &&
                                           !body["init_image"].get<std::string>().empty();
                if (has_top_level) {
                    res.status = 400;
                    res.set_content(
                        R"({"error":"segments[0].init_image conflicts with the top-level init_image; segment 0's opener is the top-level field, so send exactly one of them"})",
                        "application/json");
                    return;
                }
                body["init_image"] = body["segments"][0]["init_image"];
                LOG_INFO("LTX: folded segments[0].init_image up to the top-level opener "
                         "(segment 0 has no per-segment scene image; send it at the request root)");
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
            // An EMPTY hires_chain means "no refine chain", not "malformed request".
            // koblem always emits the key and sends [] whenever upscaler_ratio is "none",
            // so rejecting it 400s every un-refined render — which is most of them. The
            // pre-rebuild engine had no hires_chain validation at all and simply ignored
            // the field, so treating [] as absent is what restores that wire contract.
            // A non-array is still an error; only the empty array is tolerated.
            if (body.contains("hires_chain") &&
                !(body["hires_chain"].is_array() && body["hires_chain"].empty())) {
                if (!body["hires_chain"].is_array()) {
                    res.status = 400;
                    res.set_content(R"({"error":"hires_chain must be an array"})", "application/json");
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
            // TASS overlap character references (LTX-Best-Face-ID sheets).
            //   "character_refs": [{"image": "<base64|/abs/path>",
            //                       "source_id": 2,
            //                       "resize_mode": "native_resolution",
            //                       "segments": [0, 2]}]
            // The sheet is an identity, not a shot, so this stays a TOP-LEVEL list
            // and the per-segment `segments[i].character_refs` copies remain
            // ignored -- this scope is authoritative. Absent, everything below is
            // inert and the render is unchanged.
            //
            // `segments` is optional and is in RENDERED SEGMENT INDEX space. Absent
            // means every segment (the original behaviour, which callers rely on
            // being byte-identical); an explicit empty array applies to NO segment.
            std::vector<LtxCharacterRef> character_refs;
            if (body.contains("character_refs") && !body["character_refs"].is_null()) {
                if (!body["character_refs"].is_array()) {
                    res.status = 400;
                    res.set_content(R"({"error":"character_refs must be an array"})", "application/json");
                    return;
                }
                std::vector<int> explicit_source_ids;
                for (const auto& entry : body["character_refs"]) {
                    if (!entry.is_object() || !entry.contains("image") || !entry["image"].is_string() ||
                        entry["image"].get<std::string>().empty()) {
                        res.status = 400;
                        res.set_content(R"({"error":"each character_ref needs a non-empty image: base64 or absolute path"})",
                                        "application/json");
                        return;
                    }
                    LtxCharacterRef reference;
                    reference.image = entry["image"].get<std::string>();
                    if (ltx_source_is_path(reference.image)) {
                        std::error_code error;
                        if (!fs::is_regular_file(reference.image, error)) {
                            res.status = 400;
                            res.set_content(R"({"error":"character_ref image path is not a readable file"})",
                                            "application/json");
                            return;
                        }
                    }
                    if (entry.contains("source_id") && !entry["source_id"].is_null()) {
                        // Zero is the target's own tag (an exact RoPE no-op) and one
                        // is reserved by the checkpoint, so subjects start at two.
                        if (!entry["source_id"].is_number_integer() || entry["source_id"].get<int>() < 2) {
                            res.status = 400;
                            res.set_content(R"({"error":"character_ref source_id must be an integer >= 2"})",
                                            "application/json");
                            return;
                        }
                        reference.source_id = entry["source_id"].get<int>();
                        if (std::find(explicit_source_ids.begin(), explicit_source_ids.end(), reference.source_id) !=
                            explicit_source_ids.end()) {
                            res.status = 400;
                            res.set_content(R"({"error":"character_ref source_id values must be distinct"})",
                                            "application/json");
                            return;
                        }
                        explicit_source_ids.push_back(reference.source_id);
                    }
                    if (entry.contains("resize_mode") && !entry["resize_mode"].is_null()) {
                        if (!entry["resize_mode"].is_string()) {
                            res.status = 400;
                            res.set_content(R"({"error":"character_ref resize_mode must be a string"})",
                                            "application/json");
                            return;
                        }
                        const std::string mode = entry["resize_mode"].get<std::string>();
                        reference.resize_mode_explicit = true;
                        if (mode == "match_target") {
                            reference.match_target = true;
                        } else if (mode != "native_resolution") {
                            res.status = 400;
                            res.set_content(R"({"error":"character_ref resize_mode must be native_resolution or match_target"})",
                                            "application/json");
                            return;
                        }
                    }
                    if (entry.contains("segments") && !entry["segments"].is_null()) {
                        if (!entry["segments"].is_array()) {
                            res.status = 400;
                            res.set_content(R"({"error":"character_ref segments must be an array of segment indices"})",
                                            "application/json");
                            return;
                        }
                        // Present-but-empty is a real state, not a missing key: it scopes the
                        // sheet to nothing. Recorded via `scoped` so the two never collapse.
                        reference.scoped = true;
                        for (const auto& index : entry["segments"]) {
                            if (!index.is_number_integer() || index.get<int>() < 0 ||
                                index.get<int>() >= static_cast<int>(prompts.size())) {
                                res.status = 400;
                                res.set_content(json({{"error", "character_ref segments must be integers in [0, " +
                                                                    std::to_string(prompts.size()) + ")"}}).dump(),
                                                "application/json");
                                return;
                            }
                            const int segment_index = index.get<int>();
                            if (std::find(reference.segments.begin(), reference.segments.end(), segment_index) ==
                                reference.segments.end()) {
                                reference.segments.push_back(segment_index);
                            }
                        }
                    }
                    character_refs.push_back(std::move(reference));
                }
            }
            // Negative == "not supplied" -> the engine picks its 1.0 default.
            //
            // ZERO IS NOW LEGAL and means UNTAGGED: phase = source_id * scale * theta^-d/L,
            // so a zero scale makes the source tag an exact no-op and the references sit on
            // the target's own RoPE grid with nothing distinguishing them -- which is
            // precisely JoyAI-Echo's native memory layout (`position_mode: reference`).
            // Echo-derived weights never saw a phase tag during training, so running them
            // with the Best-Face-ID tagging convention would be off-recipe.
            float tass_phase_scale = -1.f;
            if (body.contains("tass_phase_scale") && !body["tass_phase_scale"].is_null()) {
                if (!body["tass_phase_scale"].is_number()) {
                    res.status = 400;
                    res.set_content(R"({"error":"tass_phase_scale must be numeric"})", "application/json");
                    return;
                }
                tass_phase_scale = body["tass_phase_scale"].get<float>();
                if (!(tass_phase_scale >= 0.f)) {
                    res.status = 400;
                    res.set_content(R"({"error":"tass_phase_scale must be zero (untagged) or positive"})",
                                    "application/json");
                    return;
                }
            }

            // MSR (Licon Multiple-Subject-Reference) in-context reference strip:
            //   "msr": {"background": "<base64|/abs/path>",
            //           "subjects": ["<base64|/abs/path>", ...],
            //           "frames": 17}
            // The BACKGROUND is required -- it is the substrate every frame of the strip
            // starts from, not one slot among many. `frames` defaults to the tightest
            // strip that gives each subject its own latent slot (8*K + 1).
            std::string msr_background;
            std::vector<std::string> msr_subjects;
            std::vector<int> msr_segments;
            int msr_frames = 0;
            if (body.contains("msr") && !body["msr"].is_null()) {
                const auto& msr = body["msr"];
                if (!msr.is_object()) {
                    res.status = 400;
                    res.set_content(R"({"error":"msr must be an object"})", "application/json");
                    return;
                }
                auto read_msr_image = [&](const json& value, const char* what, std::string* out) {
                    if (!value.is_string() || value.get<std::string>().empty()) {
                        res.status = 400;
                        res.set_content(json({{"error", std::string("msr ") + what +
                                                            " must be a non-empty base64 payload or absolute path"}})
                                            .dump(),
                                        "application/json");
                        return false;
                    }
                    *out = value.get<std::string>();
                    if (ltx_source_is_path(*out)) {
                        std::error_code error;
                        if (!fs::is_regular_file(*out, error)) {
                            res.status = 400;
                            res.set_content(json({{"error", std::string("msr ") + what +
                                                                " path is not a readable file"}})
                                                .dump(),
                                            "application/json");
                            return false;
                        }
                    }
                    return true;
                };
                if (!msr.contains("background") || msr["background"].is_null()) {
                    res.status = 400;
                    res.set_content(R"({"error":"msr needs a background image"})", "application/json");
                    return;
                }
                if (!read_msr_image(msr["background"], "background", &msr_background)) {
                    return;
                }
                if (msr.contains("subjects") && !msr["subjects"].is_null()) {
                    if (!msr["subjects"].is_array()) {
                        res.status = 400;
                        res.set_content(R"({"error":"msr subjects must be an array"})", "application/json");
                        return;
                    }
                    // Four is the checkpoint's trained ceiling (plus the background, that is
                    // the "2 to 5 reference images" the model card describes).
                    if (msr["subjects"].size() > 4) {
                        res.status = 400;
                        res.set_content(R"({"error":"msr supports at most 4 subjects"})", "application/json");
                        return;
                    }
                    for (const auto& entry : msr["subjects"]) {
                        std::string subject;
                        if (!read_msr_image(entry, "subject", &subject)) {
                            return;
                        }
                        msr_subjects.push_back(std::move(subject));
                    }
                }
                // 8*K+1 gives each subject its own latent slot. With NO subjects every slot is
                // background, which is the highest-weight form of a location injection -- so it
                // gets the checkpoint menu's shortest real strip rather than a nonsensical 1.
                msr_frames = msr_subjects.empty() ? 17 : static_cast<int>(msr_subjects.size()) * 8 + 1;
                if (msr.contains("frames") && !msr["frames"].is_null()) {
                    if (!msr["frames"].is_number_integer()) {
                        res.status = 400;
                        res.set_content(R"({"error":"msr frames must be an integer"})", "application/json");
                        return;
                    }
                    msr_frames = msr["frames"].get<int>();
                }
                // 1 modulo 8 is what makes each reference land on a whole latent frame. The
                // checkpoint's own menu starts at 17, but 9 is a real and useful length -- two
                // latent slots, one subject plus the background, the cheapest strip that composes
                // anything -- so it is accepted, and the error says so rather than listing a menu
                // the validator does not actually enforce.
                if (msr_frames < 9 || msr_frames > 65 || msr_frames % 8 != 1) {
                    res.status = 400;
                    res.set_content(R"json({"error":"msr frames must be 1 modulo 8 in [9, 65]: 9, 17, 25, 33, 41, 49, 57 or 65. The checkpoint's own menu starts at 17; 9 is the minimal one-subject strip."})json",
                                    "application/json");
                    return;
                }
                if (static_cast<int>(msr_subjects.size()) > (msr_frames - 1) / 8) {
                    res.status = 400;
                    res.set_content(json({{"error", "msr has " + std::to_string(msr_subjects.size()) +
                                                        " subjects but only " + std::to_string((msr_frames - 1) / 8) +
                                                        " slots at " + std::to_string(msr_frames) + " frames"}})
                                        .dump(),
                                    "application/json");
                    return;
                }
                // Optional shot scope. Absent means every segment; an explicit list injects the
                // location only where it belongs, which is the whole point when shot 1 is
                // somewhere else entirely.
                if (msr.contains("segments") && !msr["segments"].is_null()) {
                    if (!msr["segments"].is_array() || msr["segments"].empty()) {
                        res.status = 400;
                        res.set_content(R"({"error":"msr segments must be a non-empty array of segment indices"})",
                                        "application/json");
                        return;
                    }
                    for (const auto& index : msr["segments"]) {
                        if (!index.is_number_integer() || index.get<int>() < 0 ||
                            index.get<int>() >= static_cast<int>(prompts.size())) {
                            res.status = 400;
                            res.set_content(json({{"error", "msr segments must be integers in [0, " +
                                                                std::to_string(prompts.size()) + ")"}})
                                                .dump(),
                                            "application/json");
                            return;
                        }
                        const int segment_index = index.get<int>();
                        if (std::find(msr_segments.begin(), msr_segments.end(), segment_index) ==
                            msr_segments.end()) {
                            msr_segments.push_back(segment_index);
                        }
                    }
                }
                // MSR trained through ComfyUI's IC-LoRA guide, which tags nothing, so the strip
                // needs the untagged layout. That resolution is left to the ENGINE, which does
                // it PER SEGMENT: a shot carrying the strip gets 0, a shot the strip is scoped
                // out of keeps the Best-Face-ID default of 1.0 for its own character sheets.
                //
                // Forcing 0 here instead would apply it to the whole request and silently flip
                // the sheets on every shot the strip does not appear in -- the exact opposite of
                // what scoping promises. An explicit caller-supplied scale is still honoured.

                // A strip and a character sheet in the same shot must share a LATENT GRID: the
                // engine packs every reference onto the frame axis of one tensor. The strip is
                // always composited at the render resolution, so a sheet kept at its native size
                // cannot join it -- and that failure used to happen deep in the encode path, AFTER
                // the job was queued and the model warm, surfacing as a generic `generation_failed`
                // with the real reason only in the engine log. It is decidable here, from the
                // request alone, so decide it here.
                //
                // An UNSTATED resize mode is adopted to the strip's grid: `match_target` is already
                // what the Echo/MSR memory regime asks for, so this is the mode such a caller meant.
                // An explicitly stated `native_resolution` is refused rather than overridden.
                for (size_t index = 0; index < character_refs.size(); ++index) {
                    LtxCharacterRef& reference = character_refs[index];
                    if (reference.match_target) {
                        continue;
                    }
                    if (!reference.resize_mode_explicit) {
                        reference.match_target = true;
                        continue;
                    }
                    res.status = 400;
                    res.set_content(json({{"error", "character_ref " + std::to_string(index + 1) +
                                                        " uses resize_mode native_resolution, which cannot share a "
                                                        "latent grid with the msr strip (composited at the render "
                                                        "resolution); use match_target or drop the msr block"}})
                                        .dump(),
                                    "application/json");
                    return;
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
            // Only carry the per-segment adapter arrays when at least one shot actually names
            // a set — an all-empty array would arm the before_segment hook for nothing.
            if (std::any_of(segment_loras.begin(), segment_loras.end(),
                            [](const auto& set) { return !set.empty(); })) {
                job->ltx_segment_loras = std::move(segment_loras);
            }
            // The already-resolved top-level stack, so the lease can seed from it and restore
            // to it after the chain.
            for (const auto& [path, multiplier] : request.gen_params.lora_map) {
                job->ltx_default_loras.emplace_back(path, multiplier);
            }
            job->ltx_segment_frames = std::move(segment_frames);
            job->ltx_segment_scene_cuts = std::move(segment_scene_cuts);
            job->ltx_segment_init_images = std::move(segment_init_images);
            job->ltx_segment_keyframes = std::move(segment_keyframes);
            job->ltx_segment_keyframe_indices = std::move(segment_keyframe_indices);
            job->ltx_segment_control_frames = std::move(segment_control_frames);
            job->ltx_character_refs = std::move(character_refs);
            job->ltx_tass_phase_scale = tass_phase_scale;
            job->ltx_msr_background   = std::move(msr_background);
            job->ltx_msr_subjects     = std::move(msr_subjects);
            job->ltx_msr_frames       = msr_frames;
            job->ltx_msr_segments     = std::move(msr_segments);
            job->ltx_segment_v2v_modes = std::move(segment_v2v_modes);
            job->ltx_segment_v2v_strengths = std::move(segment_v2v_strengths);
            // Only carry the pin-strength array when a shot actually names one. An all-inherit
            // array would reach the core as a non-null pointer of -1s -- harmless, but then
            // "pin_strength absent" and "pin_strength present at the chain default" would take
            // different code paths, and the no-op claim would rest on the sentinel check rather
            // than on there being nothing to check. Same reason ltx_segment_loras is gated.
            if (std::any_of(segment_pin_strengths.begin(), segment_pin_strengths.end(),
                            [](float strength) { return strength >= 0.f; })) {
                job->ltx_segment_pin_strengths = std::move(segment_pin_strengths);
            }
            job->ltx_segment_v2v_guide_latent_paths = std::move(segment_v2v_guide_latent_paths);
            // Resolve each shot's chosen take to a DIRECTORY here, once, so the core never learns
            // the bank_id indirection or the persist/transient root search — same split as
            // v2v_guide_latent_paths, which also crosses this boundary already resolved.
            //
            // An unresolvable id is a hard 400 rather than a silent fall back to the resumed
            // bank: falling back would render a DIFFERENT take from the one the user picked and
            // look completely successful, which is the exact failure this whole feature exists to
            // stop. Better to refuse than to quietly disagree with the UI.
            if (std::any_of(segment_bank_job_ids.begin(), segment_bank_job_ids.end(),
                            [](const std::string& id) { return !id.empty(); })) {
                std::vector<std::string> resolved(segment_bank_job_ids.size());
                for (size_t segment = 0; segment < segment_bank_job_ids.size(); ++segment) {
                    if (segment_bank_job_ids[segment].empty()) continue;
                    fs::path segment_bank_dir;
                    std::string segment_bank_id;
                    if (!resolve_ltx_bank_dir(segment_bank_job_ids[segment], segment_bank_dir, segment_bank_id)) {
                        res.status = 400;
                        res.set_content(json({{"error", "invalid bank_job_id"},
                                              {"segment", static_cast<int>(segment)},
                                              {"bank_job_id", segment_bank_job_ids[segment]}})
                                            .dump(),
                                        "application/json");
                        return;
                    }
                    resolved[segment] = segment_bank_dir.string();
                }
                job->ltx_segment_bank_dirs = std::move(resolved);
            }
            // Only carry the per-shot arrays when a shot actually asked for
            // something: an all-inert array would make the chain treat every
            // shot as "explicitly no beats", which is the same result but costs
            // durable job state on every ordinary render.
            if (any_segment_beats) {
                job->ltx_segment_beats = std::move(segment_beats);
            }
            if (any_segment_overrides) {
                job->ltx_segment_seeds = std::move(segment_seeds);
                job->ltx_segment_steps = std::move(segment_steps);
                job->ltx_segment_cfg = std::move(segment_cfg);
                job->ltx_segment_negative_prompts = std::move(segment_negative_prompts);
            }
            // Reference head-frame trim. Request-level value first, then the per-shot overrides
            // resolved against it -- the array is only carried when a shot actually named the
            // field, so an ordinary project keeps the single scalar in durable job state.
            if (body.contains("reference_head_trim") && !body["reference_head_trim"].is_null()) {
                if (!body["reference_head_trim"].is_number_integer()) {
                    res.status = 400;
                    res.set_content(R"({"error":"reference_head_trim must be an integer: 0 off, -1 auto, >0 frames"})",
                                    "application/json");
                    return;
                }
                job->ltx_reference_head_trim = body["reference_head_trim"].get<int>();
                if (!valid_reference_head_trim(job->ltx_reference_head_trim)) {
                    res.status = 400;
                    res.set_content(reference_head_trim_error("reference_head_trim"), "application/json");
                    return;
                }
            }
            if (any_segment_head_trim) {
                segment_reference_head_trim.resize(static_cast<size_t>(requested_segments), kHeadTrimInherit);
                for (auto& value : segment_reference_head_trim) {
                    if (value == kHeadTrimInherit) {
                        value = job->ltx_reference_head_trim;
                    }
                }
                job->ltx_segment_reference_head_trim = std::move(segment_reference_head_trim);
            }
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
            // AUDIO GAP-FILL. Generate the SILENT stretches of the supplied drive clip while
            // holding the rest, instead of holding the whole clip. Default off: a supplied drive
            // clip normally means "condition on all of this".
            if (body.contains("audio_fill_gaps") && !body["audio_fill_gaps"].is_boolean() &&
                !body["audio_fill_gaps"].is_number_integer()) {
                res.status = 400;
                res.set_content(R"({"error":"audio_fill_gaps must be a boolean"})", "application/json");
                return;
            }
            const bool audio_fill_gaps = body.contains("audio_fill_gaps")
                                             ? (body["audio_fill_gaps"].is_boolean()
                                                    ? body["audio_fill_gaps"].get<bool>()
                                                    : body["audio_fill_gaps"].get<int>() != 0)
                                             : false;
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
                // FORK the resumed bank rather than writing back into it, so this
                // job's take is added to the project instead of replacing the one
                // it resumed. See fork_ltx_bank for why hard links and not a copy.
                //
                // Every shot this job does NOT re-render is still restored from
                // this bank, byte-identical, because the link IS the same file.
                // Failing to fork is not fatal: it degrades to the previous
                // shared-bank behaviour, which is destructive but is also exactly
                // what every render before this change did. Loud, not fatal.
                if (!resume_job_id.empty() && ltx_bank_fork_enabled()) {
                    // The resume branch above never allocated one — it used to have no
                    // need, since the job wrote into the bank it resumed. It needs an id
                    // now because the fork is named after it.
                    if (job->id.empty()) job->id = make_async_job_id(manager);
                    const fs::path source_bank = bank_dir;
                    const fs::path forked = bank_root / job->id;
                    std::vector<int> rerendered;
                    if (retake_segment >= 0) {
                        rerendered.push_back(retake_segment);
                    } else {
                        for (int segment = resume_from;
                             segment < static_cast<int>(job->ltx_prompts.size()); ++segment) {
                            rerendered.push_back(segment);
                        }
                    }
                    std::string why_not;
                    if (forked != source_bank &&
                        fork_ltx_bank(source_bank, forked, job->ltx_segment_bank_dirs,
                                      static_cast<int>(job->ltx_prompts.size()), rerendered, why_not)) {
                        bank_dir = forked;
                        bank_id = job->id;   // a real bank now, not a pointer
                        printf("[ltx] forked bank %s -> %s (re-rendering %zu shot(s); every other shot "
                               "is hard-linked, so the resumed take is preserved)\n",
                               source_bank.filename().string().c_str(), job->id.c_str(), rerendered.size());
                    } else if (forked != source_bank) {
                        printf("[ltx] WARNING: could not fork bank %s (%s) — falling back to writing "
                               "into the resumed bank, which OVERWRITES the take being resumed\n",
                               source_bank.filename().string().c_str(), why_not.c_str());
                    }
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
                    job->ltx_audio_fill_gaps = audio_fill_gaps;
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
