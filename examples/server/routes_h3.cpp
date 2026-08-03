#include "routes.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "async_jobs.h"
#include "minimax_h3_wire.h"

namespace {

namespace fs = std::filesystem;

// MiniMax-H3 generates video and synchronised stereo audio in ONE denoising pass over one packed
// token sequence. That is a MODEL property and it is fixed.
//
// One prompt per HTTP REQUEST is NOT a model property -- the entry point is ours -- so this route
// deliberately mirrors `/ltx/v1/generate`: it takes a `segments` list and renders the shots back to
// back inside one job. The goal is explicit parity, not resemblance: a koblem client that already
// drives the LTX Director should need the smallest possible diff, so field names and types are
// taken FROM LTX rather than improved on. `examples/server/api.md` carries the field-by-field
// mapping table, and every LTX Director field is in exactly one of four states there:
// identical, renamed/reshaped, N/A-refused, or N/A-ignored.
//
// Two things follow from the model, and both stay:
//
//   * there is NO CONTINUITY MACHINERY. No latent bank, no `resume_job_id`, no banked seg_<n>.bin,
//     no prefix reload, no retake splice, no `cont_latent_frames` overlap. H3 has nothing to
//     resume from; its seam is an `fl2va` RGB frame the CALLER supplies. A retake costs exactly
//     one shot, which is a real H3 advantage over LTX and is worth preserving rather than
//     emulating away. Anything whose only purpose is resumption is refused, not accepted-and-
//     ignored, because "resume" that silently re-renders everything is expensive AND wrong.
//   * there is NO GUIDANCE. The checkpoint is guidance-distilled: exactly one forward per step, no
//     unconditional branch. A `negative_prompt` or a `cfg` above 1 is not "ignored", it is a
//     request the model cannot honour, so it is refused rather than silently dropped -- a caller
//     that thinks it sent a negative prompt and got a plausible clip has no way to notice.
//   * the frame grid, the temporal VAE ratio and the audio latent rate are all defined AT 24 fps,
//     so fps is not a free parameter.
//
// The wire vocabulary (202 + `/sdcpp/v1/jobs/{id}` poll, `/media`, `/segments/{n}`, `/cancel`) is
// identical to LTX's, so a client that already drives ltx-video drives this with a different submit
// URL and the same job loop.

std::string h3_error(const std::string& message) {
    return json({{"error", message}}).dump();
}

// One shot's worth of LTX-shaped request, before it is staged onto the job.
struct H3Segment {
    std::string prompt;
    int requested_frames = 0;
    int frames           = 0;
    int64_t seed         = -1;  // LTX sentinel: negative inherits the request seed
    int steps            = 0;   // LTX sentinel: zero inherits the request steps
    std::string init_image;
    std::string end_image;
    std::string task;
};

// Accumulates the verdict on every LTX field this request carries.
//
// `refuse` is for anything that WOULD CHANGE THE RESULT if H3 could honour it -- those get a loud
// 400. `ignore` is for fields that are genuinely inert here (an LTX default that H3 already is, or
// a knob for machinery H3 does not have); those are accepted so an otherwise-valid Director request
// still renders, and every one of them is NAMED BACK TO THE CALLER in the 202's `ignored` array.
// Silently dropping a field is how someone spends an hour wondering why their setting does nothing;
// listing it in the response costs one array and removes the whole failure mode.
struct H3Compat {
    std::vector<std::string> ignored;
    std::string error;

    void refuse(const std::string& message) {
        if (error.empty()) {
            error = message;
        }
    }
    void ignore(const std::string& field) {
        if (std::find(ignored.begin(), ignored.end(), field) == ignored.end()) {
            ignored.push_back(field);
        }
    }
};

bool json_present(const json& body, const char* key) {
    return body.contains(key) && !body[key].is_null();
}

// LTX fields at the request root. Ordered refused-first so the caller sees the substantive
// objection rather than a note about an inert field.
void h3_check_ltx_top_level(const json& body, H3Compat& compat) {
    if (json_present(body, "negative_prompt")) {
        if (body["negative_prompt"].is_string() && !body["negative_prompt"].get<std::string>().empty()) {
            compat.refuse(
                "MiniMax-H3 is guidance-distilled and has no unconditional branch; negative_prompt is not supported");
        } else {
            compat.ignore("negative_prompt");
        }
    }
    for (const char* key : {"cfg", "cfg_scale", "guidance_scale"}) {
        if (!json_present(body, key) || !body[key].is_number()) {
            continue;
        }
        // 1.0 is H3's ONLY legal value and it is genuinely honoured, so an LTX request that merely
        // carries the default is not refused and is not "ignored" either.
        if (body[key].get<float>() != 1.0f) {
            compat.refuse(std::string("MiniMax-H3 is guidance-distilled; ") + key +
                          " must be 1.0 (one forward per step, no CFG)");
        }
    }
    if (body.value("batch_count", 1) != 1) {
        compat.refuse("MiniMax-H3 packs one request into one sequence; batch_count must be 1");
    }
    if (json_present(body, "fps") && body["fps"].is_number() && body["fps"].get<int>() != kMiniMaxH3Fps) {
        compat.refuse(
            "MiniMax-H3 is a 24 fps model: the 17n+5 frame grid and the 40 Hz audio latent rate are both defined "
            "there, so fps is not a free parameter");
    }

    // ── continuity machinery: refused, never ignored ─────────────────────────────────────────
    //
    // These promise reuse of banked work that does not exist. Accepting `resume_job_id` and
    // re-rendering the whole timeline would bill a full render as a resume and look successful.
    if (json_present(body, "resume_job_id") && body["resume_job_id"].is_string() &&
        !body["resume_job_id"].get<std::string>().empty()) {
        compat.refuse(
            "MiniMax-H3 keeps no latent bank, so there is nothing to resume from; resubmit the shots you want and "
            "note that a retake costs exactly one shot");
    }
    for (const char* key : {"retake_segment", "retake_from"}) {
        if (json_present(body, key) && body[key].is_number_integer() && body[key].get<int>() >= 0) {
            compat.refuse(std::string("MiniMax-H3 keeps no latent bank, so ") + key +
                          " cannot splice a re-render into banked shots; submit the single shot instead");
        }
    }
    // The seam knobs are inert rather than dangerous: H3's shots do not overlap at all, so an
    // overlap or a seam trim describes an operation that never happens.
    for (const char* key : {"cont_latent_frames", "cont_seam_drop_frames", "segment_seam_drop_frames"}) {
        if (json_present(body, key)) {
            compat.ignore(key);
        }
    }

    // ── task-selected checkpoints, no caller override, no adapters or refine chain ───────────
    if (json_present(body, "model") && body["model"].is_string()) {
        const std::string model = body["model"].get<std::string>();
        if (!model.empty() && model != "base") {
            compat.refuse("MiniMax-H3 selects FL2VA or Ref2VA from each shot's conditioning; "
                          "caller-selected model variant '" + model + "' is not supported");
        } else if (!model.empty()) {
            compat.ignore("model");
        }
    }
    if (json_present(body, "lora") && body["lora"].is_array()) {
        if (!body["lora"].empty()) {
            compat.refuse("MiniMax-H3 has no runtime-LoRA path; remove the lora array");
        } else {
            compat.ignore("lora");
        }
    }
    if (json_present(body, "hires_chain") && body["hires_chain"].is_array()) {
        if (!body["hires_chain"].empty()) {
            compat.refuse("MiniMax-H3 has no latent-upscale refine chain; hires_chain must be empty or absent");
        } else {
            // koblem emits `hires_chain: []` on every un-refined LTX render.
            compat.ignore("hires_chain");
        }
    }
    if (json_present(body, "two_stage")) {
        if (body["two_stage"].is_boolean() && body["two_stage"].get<bool>()) {
            compat.refuse("MiniMax-H3 has no two-stage refine; two_stage must be false or absent");
        } else {
            compat.ignore("two_stage");
        }
    }
    if (json_present(body, "emit_stages")) {
        if (body["emit_stages"].is_boolean() && body["emit_stages"].get<bool>()) {
            compat.refuse("MiniMax-H3 renders each shot in one pass, so there are no intermediate stages to emit");
        } else {
            compat.ignore("emit_stages");
        }
    }

    // ── TASS / MSR identity conditioning ─────────────────────────────────────────────────────
    //
    // H3's equivalent is `references`, and it is a different mechanism (VLM-routed omni references
    // packed into the sequence, not rotary-tagged latent overlap), so these cannot be translated.
    if (json_present(body, "character_refs") && body["character_refs"].is_array() &&
        !body["character_refs"].empty()) {
        compat.refuse(
            "MiniMax-H3 has no TASS overlap path; send identity sheets as references[] entries with kind 'image'");
    }
    if (json_present(body, "msr")) {
        compat.refuse(
            "MiniMax-H3 has no MSR reference strip; send the background and subjects as references[] entries with "
            "kind 'image'");
    }
    if (json_present(body, "tass_phase_scale")) {
        compat.refuse("tass_phase_scale controls the LTX rotary source tag, which MiniMax-H3 does not have");
    }

    // ── video-to-video / lipdub ──────────────────────────────────────────────────────────────
    if (json_present(body, "v2v_mode") && body["v2v_mode"].is_number_integer() &&
        body["v2v_mode"].get<int>() != 0) {
        compat.refuse("MiniMax-H3 has no video-to-video path; v2v_mode must be 0 or absent");
    }
    if (json_present(body, "control_frames") && body["control_frames"].is_array() &&
        !body["control_frames"].empty()) {
        compat.refuse(
            "MiniMax-H3 has no lipdub/SDEdit control-frame path; a source clip belongs in references[] with kind "
            "'video'");
    }
    if (json_present(body, "relip_ref_tstride") && body["relip_ref_tstride"].is_number_integer() &&
        body["relip_ref_tstride"].get<int>() != 1) {
        compat.refuse("relip_ref_tstride belongs to the LTX lipdub path, which MiniMax-H3 does not have");
    }

    // ── drive audio ──────────────────────────────────────────────────────────────────────────
    //
    // H3 GENERATES its soundtrack in the same denoise as the picture. A supplied voice track is not
    // a conditioning input here, so accepting one would deliver a clip whose audio is not the audio
    // the caller uploaded, and nothing about the response would say so.
    for (const char* key : {"audio_offset_frames", "audio_fill_gaps"}) {
        if (!json_present(body, key)) {
            continue;
        }
        const bool inert = (body[key].is_number_integer() && body[key].get<int>() == 0) ||
                           (body[key].is_boolean() && !body[key].get<bool>());
        if (inert) {
            compat.ignore(key);
        } else {
            compat.refuse(std::string(key) +
                          " belongs to the LTX drive-audio timeline; MiniMax-H3 generates its own soundtrack");
        }
    }
    if (json_present(body, "reference_head_trim") && body["reference_head_trim"].is_number_integer() &&
        body["reference_head_trim"].get<int>() != 0) {
        compat.refuse(
            "reference_head_trim removes an LTX reference leak at latent frame 0; MiniMax-H3 does not place "
            "references on the target's rotary grid and has nothing to trim");
    }

    // ── inert request-level defaults every LTX client carries ────────────────────────────────
    //
    // `strength` is the LTX image-pin hold; an H3 fl2va keyframe is anchored outright, so there is
    // no partial hold to set. It is ACCEPTED (rather than refused like the per-shot `pin_strength`)
    // because a Director sends it on every request as a request-level default, whereas nobody
    // sends `pin_strength` by accident -- that one is a deliberate per-shot intent and deserves the
    // 400. `clip_skip` cannot bite either: the H3 conditioner reads a fixed hidden layer.
    for (const char* key : {"strength", "clip_skip"}) {
        if (json_present(body, key)) {
            compat.ignore(key);
        }
    }
}

// The same audit for one `segments[i]` object.
void h3_check_ltx_segment(const json& entry, size_t index, H3Compat& compat) {
    const std::string where = " on segment " + std::to_string(index);

    if (json_present(entry, "negative_prompt")) {
        if (entry["negative_prompt"].is_string() && !entry["negative_prompt"].get<std::string>().empty()) {
            compat.refuse("MiniMax-H3 is guidance-distilled; negative_prompt" + where + " is not supported");
        } else {
            compat.ignore("segments[].negative_prompt");
        }
    }
    if (json_present(entry, "cfg") && entry["cfg"].is_number() && entry["cfg"].get<float>() >= 0.f &&
        entry["cfg"].get<float>() != 1.0f) {
        compat.refuse("MiniMax-H3 is guidance-distilled; cfg" + where + " must be 1.0");
    }
    if (json_present(entry, "model") && entry["model"].is_string()) {
        const std::string model = entry["model"].get<std::string>();
        if (!model.empty() && model != "base") {
            compat.refuse("MiniMax-H3 selects FL2VA or Ref2VA from each shot's conditioning; "
                          "caller-selected segment model variant '" + model + "' is not supported");
        } else if (!model.empty()) {
            compat.ignore("segments[].model");
        }
    }
    if (json_present(entry, "lora") && entry["lora"].is_array()) {
        if (!entry["lora"].empty()) {
            compat.refuse("MiniMax-H3 has no runtime-LoRA path; remove the lora array" + where);
        } else {
            compat.ignore("segments[].lora");
        }
    }
    // Every H3 shot IS a scene cut -- shots share no latent state at all -- so the flag is
    // structurally inert rather than merely unimplemented.
    if (json_present(entry, "scene_cut")) {
        compat.ignore("segments[].scene_cut");
    }
    // LTX spells "this shot did not say" as a NEGATIVE number, and koblem emits the key on every
    // shot. Refusing the sentinel would 400 every Director request; refusing a real value is the
    // point, because an explicit hold strength that silently does nothing is the failure mode.
    if (json_present(entry, "pin_strength") && entry["pin_strength"].is_number() &&
        entry["pin_strength"].get<float>() >= 0.f) {
        compat.refuse(
            "pin_strength is the LTX image-pin hold strength; MiniMax-H3 anchors an fl2va keyframe outright and has "
            "no partial hold" +
            where);
    } else if (json_present(entry, "pin_strength")) {
        compat.ignore("segments[].pin_strength");
    }
    if (json_present(entry, "keyframes") && entry["keyframes"].is_array() && !entry["keyframes"].empty()) {
        compat.refuse(
            "MiniMax-H3 pins only the FIRST and LAST frame of a shot (fl2va); mid-shot keyframes are not "
            "supported" +
            where);
    }
    if (json_present(entry, "beats") && entry["beats"].is_array() && !entry["beats"].empty()) {
        compat.refuse("Prompt Relay beats are an LTX conditioning layout; MiniMax-H3 encodes one prompt per shot" +
                      where);
    }
    if (json_present(entry, "v2v_mode") && entry["v2v_mode"].is_number_integer() &&
        entry["v2v_mode"].get<int>() != 0) {
        compat.refuse("MiniMax-H3 has no video-to-video path; v2v_mode must be 0 or absent" + where);
    }
    // Same sentinel discipline for the V2V family: an empty array, an empty string and LTX's
    // negative "inherit" number are all inert and are accepted so a Director that always emits the
    // keys still renders. Anything that names a real source or a real strength is refused.
    for (const char* key : {"control_frames", "v2v_source_latent_path", "v2v_source_job_id", "v2v_source_segment",
                            "v2v_guide_strength"}) {
        if (!json_present(entry, key)) {
            continue;
        }
        const json& value = entry[key];
        const bool inert  = (value.is_array() && value.empty()) ||
                           (value.is_string() && value.get<std::string>().empty()) ||
                           (value.is_number() && value.get<double>() < 0.0);
        if (inert) {
            compat.ignore(std::string("segments[].") + key);
        } else {
            compat.refuse(std::string(key) + " belongs to the LTX video-to-video path" + where);
        }
    }
    if (json_present(entry, "bank_job_id") && entry["bank_job_id"].is_string() &&
        !entry["bank_job_id"].get<std::string>().empty()) {
        compat.refuse("bank_job_id selects a banked LTX take; MiniMax-H3 keeps no latent bank" + where);
    } else if (json_present(entry, "bank_job_id")) {
        compat.ignore("segments[].bank_job_id");
    }
    if (json_present(entry, "reference_head_trim") && entry["reference_head_trim"].is_number_integer() &&
        entry["reference_head_trim"].get<int>() != 0) {
        compat.refuse("reference_head_trim is an LTX reference-leak workaround" + where);
    }
}

bool extract_h3_request(const httplib::Request& req, json& body) {
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

bool parse_h3_video_request(const json& body,
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

// A reference image/frame is either a base64 payload or a trusted absolute path. `ltx_source_is_path`
// asks the FILESYSTEM rather than looking at the leading character, which matters because base64
// JPEG begins "/9j/"; the `ltx_` prefix is historical, the predicate is not LTX-specific.
bool valid_h3_source(const std::string& source) {
    if (source.empty()) {
        return false;
    }
    if (!ltx_source_is_path(source)) {
        return true;  // treated as a base64 payload; it fails to decode loudly if it is not one
    }
    std::error_code error;
    return fs::is_regular_file(source, error);
}

bool decode_h3_wav(const std::string& value, std::vector<uint8_t>& bytes) {
    if (ltx_source_is_path(value)) {
        std::ifstream input(value, std::ios::binary);
        if (!input.is_open()) {
            return false;
        }
        const std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        bytes.assign(content.begin(), content.end());
        return !bytes.empty();
    }
    std::string encoded = value;
    if (encoded.rfind("data:", 0) == 0) {
        const size_t comma = encoded.find(',');
        if (comma == std::string::npos) {
            return false;
        }
        encoded = encoded.substr(comma + 1);
    }
    return base64_decode(encoded, bytes) && !bytes.empty();
}

}  // namespace

void register_minimax_h3_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    // STATIC METADATA + PURE ARITHMETIC. No model, no filesystem. Registered on the CUDA child for
    // non-isolated deployments and answered by the supervisor when isolation is on -- both from
    // h3_capabilities_json(), so the two can never disagree. A client bootstrapping its video tab
    // must be able to learn the 17n+5 rule without a ~20 GB CUDA child being started to tell it.
    svr.Get("/h3/v1/capabilities", [](const httplib::Request& req, httplib::Response& res) {
        const auto number_param = [&](const char* name) {
            const std::string value = req.get_param_value(name);
            return value.empty() ? 0 : std::atoi(value.c_str());
        };
        res.set_content(h3_capabilities_json(number_param("width"),
                                             number_param("height"),
                                             number_param("frames"))
                            .dump(),
                        "application/json");
    });

    svr.Post("/h3/v1/generate", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (runtime_is_draining(*runtime)) {
                res.status = 503;
                res.set_content(R"({"error":"service draining — not accepting new jobs"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, VID_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(VID_GEN)}}).dump(),
                                "application/json");
                return;
            }
            json body;
            if (!extract_h3_request(req, body)) {
                res.status = 400;
                res.set_content(R"({"error":"missing or invalid request"})", "application/json");
                return;
            }

            H3Compat compat;
            h3_check_ltx_top_level(body, compat);

            // LTX's drive/track audio uploads arrive as multipart parts rather than JSON keys, so
            // they are audited here. Same reasoning as `audio_fill_gaps`: H3 generates its own
            // soundtrack, and delivering a clip whose audio is not the audio that was uploaded
            // would look exactly like success.
            if (req.is_multipart_form_data()) {
                for (const char* part : {"audio_full", "audio_track"}) {
                    if (req.form.has_file(part)) {
                        compat.refuse(std::string(part) +
                                      " is the LTX drive-audio upload; MiniMax-H3 generates its soundtrack in the "
                                      "same denoise as the picture. A reference soundtrack goes on a references[] "
                                      "entry, as ref_audio_<i>");
                    }
                }
                for (const char* prefix : {"audio_", "audio_full_", "audio_track_"}) {
                    if (req.form.has_file(std::string(prefix) + "0")) {
                        compat.refuse(std::string(prefix) +
                                      "<n> is the LTX per-shot drive-audio upload; MiniMax-H3 generates its own "
                                      "soundtrack");
                    }
                }
            }

            // ── segments ─────────────────────────────────────────────────────────────────────
            //
            // The LTX shape exactly: `segments` may be an array of bare strings or of objects,
            // `prompts` is the legacy string-array alias, and `n_segments` (when sent) must agree.
            // A bare top-level `prompt` with no list is ONE segment -- which is what every request
            // written against the pre-segment-list H3 route already sends, so those keep working.
            std::vector<H3Segment> segments;
            const auto push_bare_prompt = [&](std::string prompt) {
                H3Segment segment;
                segment.prompt = std::move(prompt);
                segments.push_back(std::move(segment));
            };
            if (body.contains("segments") && body["segments"].is_array()) {
                for (size_t index = 0; index < body["segments"].size(); ++index) {
                    const auto& entry = body["segments"][index];
                    if (entry.is_string()) {
                        push_bare_prompt(entry.get<std::string>());
                        continue;
                    }
                    if (!entry.is_object()) {
                        res.status = 400;
                        res.set_content(R"({"error":"each segment must be a prompt string or an object"})",
                                        "application/json");
                        return;
                    }
                    h3_check_ltx_segment(entry, index, compat);
                    H3Segment segment;
                    segment.prompt = entry.value("prompt", std::string());
                    // `frames` keeps LTX's name and its per-shot meaning. Only the GRID differs:
                    // LTX wants 8k+1, H3 snaps UP to 17n+5 and reports both numbers, so a Director
                    // that sends its LTX shot lengths gets a legal H3 clip and is told the length
                    // it actually got rather than silently rendering something else.
                    segment.requested_frames = entry.value("frames", 0);
                    if (segment.requested_frames < 0) {
                        res.status = 400;
                        res.set_content(R"({"error":"each segment frames value must be positive"})",
                                        "application/json");
                        return;
                    }
                    segment.seed  = entry.value("seed", static_cast<int64_t>(-1));
                    segment.steps = entry.value("steps", 0);
                    if (segment.steps < 0 || segment.steps > 200) {
                        res.status = 400;
                        res.set_content(R"({"error":"segment steps must be between 0 (inherit) and 200"})",
                                        "application/json");
                        return;
                    }
                    segment.init_image = entry.value("first_frame", entry.value("init_image", std::string()));
                    segment.end_image  = entry.value("last_frame", entry.value("end_image", std::string()));
                    if (entry.contains("task") && !entry["task"].is_null()) {
                        if (!entry["task"].is_string()) {
                            res.status = 400;
                            res.set_content(R"({"error":"segment task must be one of t2va, fl2va, ref2va"})",
                                            "application/json");
                            return;
                        }
                        segment.task = entry["task"].get<std::string>();
                    }
                    segments.push_back(std::move(segment));
                }
            } else if (body.contains("prompts") && body["prompts"].is_array()) {
                for (const auto& prompt : body["prompts"]) {
                    if (prompt.is_string()) {
                        push_bare_prompt(prompt.get<std::string>());
                    }
                }
            } else if (body.contains("prompt") && body["prompt"].is_string()) {
                push_bare_prompt(body["prompt"].get<std::string>());
            }

            const int requested_segments = body.value("n_segments", static_cast<int>(segments.size()));
            if (requested_segments < 1 || segments.empty() ||
                requested_segments != static_cast<int>(segments.size()) ||
                std::any_of(segments.begin(), segments.end(),
                            [](const H3Segment& segment) { return segment.prompt.empty(); })) {
                res.status = 400;
                res.set_content(R"({"error":"segments must contain exactly n_segments non-empty prompts"})",
                                "application/json");
                return;
            }

            // ── refusals that exist because the alternative is a silent wrong render ──────────
            //
            // Reported here, after the segment list has parsed, so the message names the field
            // rather than a symptom -- and reported as ONE 400 rather than the first of many.
            if (!compat.error.empty()) {
                res.status = 400;
                res.set_content(json({{"error", compat.error}, {"ignored", compat.ignored}}).dump(),
                                "application/json");
                return;
            }

            // ── geometry ─────────────────────────────────────────────────────────────────────
            //
            // Frames snap UP to 17n+5 (the reference rule), and the 202 reports the snapped value
            // per shot so a UI can show the real duration instead of the one it asked for.
            //
            // The omitted-field defaults come from the WORKER'S OWN configuration
            // (--video-frames / -W / -H), so a deployment's recipe governs, falling back to the
            // model's native 124-frame 1344x768 only when the configured value is not a legal H3
            // shape -- which it will not be on a server whose defaults were resolved for images.
            const SDGenerationParams& defaults = *runtime->default_gen_params;
            const int default_frames = h3_frame_count_is_aligned(defaults.video_frames) ? defaults.video_frames : 124;
            const int request_frames = body.contains("frames") && body["frames"].is_number_integer()
                                           ? body["frames"].get<int>()
                                           : body.value("video_frames", default_frames);
            if (request_frames < 1) {
                res.status = 400;
                res.set_content(R"({"error":"frames must be positive"})", "application/json");
                return;
            }
            for (H3Segment& segment : segments) {
                if (segment.requested_frames <= 0) {
                    segment.requested_frames = request_frames;
                }
                segment.frames = h3_align_frame_count(segment.requested_frames);
            }

            const bool defaults_are_legal_canvas =
                h3_canvas_is_legal(defaults.get_resolved_width(), defaults.get_resolved_height());
            int width  = body.value("width", defaults_are_legal_canvas ? defaults.get_resolved_width() : 1344);
            int height = body.value("height", defaults_are_legal_canvas ? defaults.get_resolved_height() : 768);
            const bool adapt_canvas = body.value("adapt_canvas", false);
            if (!h3_canvas_is_legal(width, height)) {
                int adapted_width  = 0;
                int adapted_height = 0;
                h3_adapt_canvas(width, height, adapted_width, adapted_height);
                // Adopting a caller's off-grid size WITHOUT being asked would render something
                // other than what was requested and report success, so it is opt-in. The refusal
                // names the size that would have been used, and GET /h3/v1/capabilities computes
                // the same answer without submitting anything.
                if (!adapt_canvas) {
                    res.status = 400;
                    res.set_content(json({{"error", "width and height must be multiples of 32 with width*height <= " +
                                                        std::to_string(kMiniMaxH3MaxPixels) +
                                                        "; send adapt_canvas:true to accept the model's own canvas"},
                                          {"width", adapted_width},
                                          {"height", adapted_height}})
                                        .dump(),
                                    "application/json");
                    return;
                }
                // The reference canvas rule rounds each axis to 32 AFTER the area cap, so an
                // extreme aspect can round back over it (3712x288 for 1344x100). Refuse rather
                // than render one: the cap is what the sequence length was trained at, and
                // "adapt_canvas" promised the model's own canvas, not any canvas at all.
                if (!h3_canvas_is_legal(adapted_width, adapted_height)) {
                    res.status = 400;
                    res.set_content(json({{"error", "aspect ratio is too extreme: the model's own canvas rule lands at " +
                                                        std::to_string(adapted_width) + "x" +
                                                        std::to_string(adapted_height) + ", over the " +
                                                        std::to_string(kMiniMaxH3MaxPixels) + " pixel cap"},
                                          {"width", adapted_width},
                                          {"height", adapted_height}})
                                        .dump(),
                                    "application/json");
                    return;
                }
                width  = adapted_width;
                height = adapted_height;
            }

            float sigma_shift_video = body.value("sigma_shift_video", kMiniMaxH3SigmaShiftVideo);
            float sigma_shift_audio = body.value("sigma_shift_audio", kMiniMaxH3SigmaShiftAudio);
            if (!(sigma_shift_video > 0.f) || !(sigma_shift_audio > 0.f)) {
                res.status = 400;
                res.set_content(R"({"error":"sigma_shift_video and sigma_shift_audio must be positive"})",
                                "application/json");
                return;
            }

            // ── fl2va anchors ────────────────────────────────────────────────────────────────
            //
            // The request's top-level `init_image` / `end_image` seed SEGMENT 0 -- that is LTX's
            // own convention for the opener and it is what keeps an LTX-shaped request working.
            //
            // Unlike LTX, `segments[0].init_image` is NOT a trap here: H3 shots share no latent
            // state, so every shot legitimately owns a first and last frame. It is folded up to
            // segment 0 exactly as LTX folds it, and the conflict between the two spellings is
            // refused for the same reason LTX refuses it -- picking a winner would be guessing at
            // which image the caller meant to open on.
            const std::string top_first_frame = body.value("first_frame", body.value("init_image", std::string()));
            const std::string top_last_frame  = body.value("last_frame", body.value("end_image", std::string()));
            if (!top_first_frame.empty()) {
                if (!segments.front().init_image.empty()) {
                    res.status = 400;
                    res.set_content(
                        R"({"error":"segments[0].init_image conflicts with the top-level init_image; send exactly one of them"})",
                        "application/json");
                    return;
                }
                segments.front().init_image = top_first_frame;
            }
            if (!top_last_frame.empty()) {
                if (!segments.front().end_image.empty()) {
                    res.status = 400;
                    res.set_content(
                        R"({"error":"segments[0].end_image conflicts with the top-level end_image; send exactly one of them"})",
                        "application/json");
                    return;
                }
                segments.front().end_image = top_last_frame;
            }
            for (size_t index = 0; index < segments.size(); ++index) {
                for (const auto& [name, source] :
                     {std::pair<const char*, const std::string&>{"first_frame", segments[index].init_image},
                      std::pair<const char*, const std::string&>{"last_frame", segments[index].end_image}}) {
                    if (!source.empty() && !valid_h3_source(source)) {
                        res.status = 400;
                        res.set_content(h3_error(std::string(name) + " on segment " + std::to_string(index) +
                                                 " must be a base64 image or a readable absolute path"),
                                        "application/json");
                        return;
                    }
                }
            }

            // ── references ───────────────────────────────────────────────────────────────────
            //
            // TOP-LEVEL, with an optional per-entry `segments` scope -- the same placement and the
            // same scoping convention as LTX's `character_refs`, and for the same reason: a
            // reference is a cast member or a location, not a shot. Absent scope means every shot;
            // an explicit empty array means no shot.
            std::vector<MiniMaxH3JobReference> references;
            // Parallel to `references`: the multipart field a reference's soundtrack arrived in,
            // empty when it came inline. Kept SEPARATE from MiniMaxH3JobReference::audio_path
            // rather than encoded into it with a sentinel prefix, so a caller-supplied string can
            // never be mistaken for a form-part name.
            std::vector<std::string> reference_audio_parts;
            int ref_images = 0;
            int ref_videos = 0;
            int ref_audios = 0;
            // The longest shot bounds a reference clip: it is truncated per request, not per shot,
            // because one decoded reference list serves every shot it is in scope for.
            const int longest_segment_frames =
                std::max_element(segments.begin(), segments.end(),
                                 [](const H3Segment& a, const H3Segment& b) { return a.frames < b.frames; })
                    ->frames;
            if (body.contains("references") && !body["references"].is_null()) {
                if (!body["references"].is_array()) {
                    res.status = 400;
                    res.set_content(R"({"error":"references must be an array"})", "application/json");
                    return;
                }
                for (const auto& entry : body["references"]) {
                    if (!entry.is_object()) {
                        res.status = 400;
                        res.set_content(R"({"error":"each reference must be an object"})", "application/json");
                        return;
                    }
                    const std::string kind = entry.value("kind", std::string());
                    MiniMaxH3JobReference reference;
                    const size_t index = references.size();

                    // A soundtrack is accepted as JSON (base64 / absolute path) or as a multipart
                    // file `ref_audio_<i>`, indexed by POSITION IN THIS ARRAY, so an image at
                    // index 0 and a video at index 1 cannot be confused about whose audio it is.
                    std::string audio_source = entry.value("audio", std::string());
                    std::string audio_part;
                    const std::string audio_part_name = "ref_audio_" + std::to_string(index);
                    if (req.is_multipart_form_data() && req.form.has_file(audio_part_name)) {
                        if (!audio_source.empty()) {
                            res.status = 400;
                            res.set_content(h3_error("reference " + std::to_string(index) +
                                                     " has both an inline audio field and a " + audio_part_name +
                                                     " upload; send one"),
                                            "application/json");
                            return;
                        }
                        audio_part = audio_part_name;
                    }
                    const bool has_audio = !audio_source.empty() || !audio_part.empty();

                    if (kind == "image") {
                        reference.kind  = MiniMaxH3JobReference::Kind::Image;
                        reference.image = entry.value("image", std::string());
                        if (!valid_h3_source(reference.image)) {
                            res.status = 400;
                            res.set_content(h3_error("reference " + std::to_string(index) +
                                                     " needs a base64 image or a readable absolute path"),
                                            "application/json");
                            return;
                        }
                        if (has_audio) {
                            res.status = 400;
                            res.set_content(h3_error("an image reference carries no soundtrack; use kind 'audio'"),
                                            "application/json");
                            return;
                        }
                        ref_images++;
                    } else if (kind == "video") {
                        reference.kind = MiniMaxH3JobReference::Kind::Video;
                        if (!entry.contains("frames") || !entry["frames"].is_array()) {
                            res.status = 400;
                            res.set_content(h3_error("reference " + std::to_string(index) +
                                                     " needs a frames array (the clip at 24 fps, one entry per frame)"),
                                            "application/json");
                            return;
                        }
                        for (const auto& frame : entry["frames"]) {
                            if (!frame.is_string() || !valid_h3_source(frame.get<std::string>())) {
                                res.status = 400;
                                res.set_content(h3_error("reference " + std::to_string(index) +
                                                         " frames must be base64 images or readable absolute paths"),
                                                "application/json");
                                return;
                            }
                            reference.frames.push_back(frame.get<std::string>());
                        }
                        // The reference is VAE-encoded whole, so its own length lands on the same
                        // 17n+5 grid, trimmed DOWN, and never longer than the longest clip being
                        // rendered in this job.
                        if (static_cast<int>(reference.frames.size()) > longest_segment_frames) {
                            reference.frames.resize(static_cast<size_t>(longest_segment_frames));
                        }
                        const int trimmed = h3_trim_reference_frame_count(static_cast<int>(reference.frames.size()));
                        if (trimmed == 0) {
                            res.status = 400;
                            res.set_content(h3_error("reference " + std::to_string(index) +
                                                     " needs at least 5 frames (~0.2 s at 24 fps)"),
                                            "application/json");
                            return;
                        }
                        reference.frames.resize(static_cast<size_t>(trimmed));
                        // ★ 2 fps conditioner sampling, decided HERE. The route is the right home:
                        // it is pure index arithmetic over a frame list the route already holds,
                        // and the conditioner GGML_ASSERTs that the sampled frame count and the
                        // block timestamps agree. Producing them at the same place, from the same
                        // number, makes that agreement structural instead of a convention two
                        // layers have to remember.
                        reference.conditioner_frame_indices = h3_sample_reference_frame_indices(trimmed);
                        reference.block_timestamps =
                            h3_reference_block_timestamps(reference.conditioner_frame_indices.size());
                        ref_videos++;
                    } else if (kind == "audio") {
                        reference.kind = MiniMaxH3JobReference::Kind::Audio;
                        if (!has_audio) {
                            res.status = 400;
                            res.set_content(h3_error("reference " + std::to_string(index) +
                                                     " needs an audio payload (base64 WAV, absolute path, or a " +
                                                     audio_part_name + " upload)"),
                                            "application/json");
                            return;
                        }
                        ref_audios++;
                    } else {
                        res.status = 400;
                        res.set_content(R"({"error":"reference kind must be image, video or audio"})",
                                        "application/json");
                        return;
                    }
                    // Optional shot scope, in rendered-segment index space -- the same space as the
                    // request's own `segments[]` positions, and the same encoding LtxCharacterRef
                    // uses: `scoped` records that the key was PRESENT, so an explicit empty array
                    // (applies to nothing) is never confused with an absent one (applies to all).
                    if (entry.contains("segments") && !entry["segments"].is_null()) {
                        if (!entry["segments"].is_array()) {
                            res.status = 400;
                            res.set_content(R"({"error":"reference segments must be an array of segment indices"})",
                                            "application/json");
                            return;
                        }
                        reference.scoped = true;
                        for (const auto& scope : entry["segments"]) {
                            if (!scope.is_number_integer() || scope.get<int>() < 0 ||
                                scope.get<int>() >= static_cast<int>(segments.size())) {
                                res.status = 400;
                                res.set_content(json({{"error", "reference segments must be integers in [0, " +
                                                                    std::to_string(segments.size()) + ")"}})
                                                    .dump(),
                                                "application/json");
                                return;
                            }
                            const int segment_index = scope.get<int>();
                            if (std::find(reference.segments.begin(), reference.segments.end(), segment_index) ==
                                reference.segments.end()) {
                                reference.segments.push_back(segment_index);
                            }
                        }
                    }
                    reference.audio_path = audio_source;  // staged to the bank once the id exists
                    references.push_back(std::move(reference));
                    reference_audio_parts.push_back(std::move(audio_part));
                }
            }
            if (ref_images > kMiniMaxH3MaxRefImages || ref_videos > kMiniMaxH3MaxRefVideos ||
                ref_audios > kMiniMaxH3MaxRefAudios ||
                references.size() > static_cast<size_t>(kMiniMaxH3MaxReferences)) {
                res.status = 400;
                res.set_content(json({{"error", "too many references"},
                                      {"max_images", kMiniMaxH3MaxRefImages},
                                      {"max_videos", kMiniMaxH3MaxRefVideos},
                                      {"max_audios", kMiniMaxH3MaxRefAudios},
                                      {"max_total", kMiniMaxH3MaxReferences}})
                                    .dump(),
                                "application/json");
                return;
            }

            // ── task selection, per shot ─────────────────────────────────────────────────────
            //
            // `task` is optional: the conditioning the caller actually sent determines it, and an
            // explicit value is checked against that rather than trusted. A request that names
            // ref2va but carries no references would otherwise render a plain t2va and look fine.
            const std::string requested_task = body.contains("task") && body["task"].is_string()
                                                   ? body["task"].get<std::string>()
                                                   : std::string();
            if (!requested_task.empty() && requested_task != "t2va" && requested_task != "fl2va" &&
                requested_task != "ref2va") {
                res.status = 400;
                res.set_content(R"({"error":"task must be one of t2va, fl2va, ref2va"})", "application/json");
                return;
            }
            for (size_t index = 0; index < segments.size(); ++index) {
                H3Segment& segment = segments[index];
                const bool has_keyframes = !segment.init_image.empty() || !segment.end_image.empty();
                const bool has_references =
                    std::any_of(references.begin(), references.end(),
                                [&](const MiniMaxH3JobReference& reference) {
                                    return !reference.scoped ||
                                           std::find(reference.segments.begin(), reference.segments.end(),
                                                     static_cast<int>(index)) != reference.segments.end();
                                });
                const bool has_visual_reference =
                    std::any_of(references.begin(), references.end(),
                                [&](const MiniMaxH3JobReference& reference) {
                                    const bool in_scope =
                                        !reference.scoped ||
                                        std::find(reference.segments.begin(), reference.segments.end(),
                                                  static_cast<int>(index)) != reference.segments.end();
                                    return in_scope && reference.kind != MiniMaxH3JobReference::Kind::Audio;
                                });
                if (has_references && !has_visual_reference) {
                    res.status = 400;
                    res.set_content(
                        json({{"error", "audio references cannot be the only references for segment " +
                                            std::to_string(index) +
                                            "; pair them with at least one image or video reference"},
                              {"segment", static_cast<int>(index)}})
                            .dump(),
                        "application/json");
                    return;
                }
                const std::string derived = has_references ? "ref2va" : (has_keyframes ? "fl2va" : "t2va");
                const std::string stated  = segment.task.empty() ? requested_task : segment.task;
                if (!stated.empty()) {
                    if (stated != "t2va" && stated != "fl2va" && stated != "ref2va") {
                        res.status = 400;
                        res.set_content(R"({"error":"task must be one of t2va, fl2va, ref2va"})", "application/json");
                        return;
                    }
                    if (stated != derived) {
                        res.status = 400;
                        res.set_content(json({{"error", "task '" + stated + "' disagrees with the conditioning "
                                                        "supplied for segment " + std::to_string(index) +
                                                        "; that shot looks like '" + derived + "'"},
                                              {"segment", static_cast<int>(index)},
                                              {"task", derived},
                                              {"first_frame", !segment.init_image.empty()},
                                              {"last_frame", !segment.end_image.empty()},
                                              {"references", has_references}})
                                            .dump(),
                                        "application/json");
                        return;
                    }
                }
                if (derived == "ref2va" && has_keyframes) {
                    res.status = 400;
                    res.set_content(json({{"error", "ref2va and fl2va are different tasks; segment " +
                                                        std::to_string(index) +
                                                        " carries both references and first/last frames"},
                                          {"segment", static_cast<int>(index)}})
                                        .dump(),
                                    "application/json");
                    return;
                }
                segment.task = derived;
            }
            const bool tasks_agree = std::all_of(segments.begin(), segments.end(), [&](const H3Segment& segment) {
                return segment.task == segments.front().task;
            });
            const std::string job_task = tasks_agree ? segments.front().task : std::string("mixed");

            // ── hand the generic fields to the shared parameter parser ───────────────────────
            //
            // Segment 0's resolved geometry is written back into the body so the generic parser,
            // and therefore the core, sees the SNAPPED values rather than what the caller typed.
            //
            // The fl2va anchors are REMOVED from the body deliberately: every shot's images -- including
            // segment 0's -- are decoded by the executor from `h3_segment_init_images` /
            // `h3_segment_end_images`, so all shots take one code path. Leaving segment 0's copy in
            // the body would decode it twice and give shot 0 a different path from every other shot,
            // which is exactly the asymmetry that hides bugs.
            body["prompt"]       = segments.front().prompt;
            body["width"]        = width;
            body["height"]       = height;
            body["video_frames"] = segments.front().frames;
            body["fps"]          = kMiniMaxH3Fps;
            body.erase("init_image");
            body.erase("end_image");
            body.erase("first_frame");
            body.erase("last_frame");
            // The VIDEO shift IS the schedule the sampler walks, and that already has a wire
            // field. Only fill it in when the caller did not state one directly, so an explicit
            // `sample_params.flow_shift` still wins.
            const bool caller_set_flow_shift = body.contains("sample_params") &&
                                               body["sample_params"].is_object() &&
                                               body["sample_params"].contains("flow_shift") &&
                                               body["sample_params"]["flow_shift"].is_number();
            if (caller_set_flow_shift) {
                sigma_shift_video = body["sample_params"]["flow_shift"].get<float>();
            } else {
                json& sample_params = body["sample_params"];
                if (!sample_params.is_object()) {
                    sample_params = json::object();
                }
                sample_params["flow_shift"] = sigma_shift_video;
            }

            VidGenJobRequest request;
            std::string error_message;
            if (!parse_h3_video_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(h3_error(error_message), "application/json");
                return;
            }

            AsyncJobManager& manager = *runtime->async_job_manager;
            auto job        = std::make_shared<AsyncGenerationJob>();
            job->kind       = AsyncJobKind::VidGen;
            job->status     = AsyncJobStatus::Queued;
            job->created_at = unix_timestamp_now();
            job->vid_gen    = std::move(request);
            job->h3_task    = job_task;
            job->h3_sigma_shift_video = sigma_shift_video;
            job->h3_sigma_shift_audio = sigma_shift_audio;
            job->h3_emit_segments     = body.value("emit_segments", false);
            job->h3_prompts.reserve(segments.size());
            for (const H3Segment& segment : segments) {
                job->h3_prompts.push_back(segment.prompt);
                job->h3_segment_tasks.push_back(segment.task);
                job->h3_segment_frames.push_back(segment.frames);
                job->h3_segment_latent_frames.push_back(h3_video_latent_t(segment.frames));
                job->h3_segment_audio_latent_frames.push_back(h3_audio_latent_t(segment.frames));
                job->h3_segment_seeds.push_back(segment.seed);
                job->h3_segment_steps.push_back(segment.steps);
                job->h3_segment_init_images.push_back(segment.init_image);
                job->h3_segment_end_images.push_back(segment.end_image);
            }

            const bool persist_bank = body.value("persist", false);
            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id = make_async_job_id(manager);

                // A bank is created when there is reference audio to keep OR when the caller wants
                // progressive per-shot partials. A t2va/fl2va request that wants neither leaves
                // nothing behind, so it also leaves nothing to clean up.
                const bool needs_bank =
                    job->h3_emit_segments ||
                    std::any_of(references.begin(), references.end(),
                                [](const MiniMaxH3JobReference& reference) { return !reference.audio_path.empty(); }) ||
                    std::any_of(reference_audio_parts.begin(), reference_audio_parts.end(),
                                [](const std::string& part) { return !part.empty(); });
                if (needs_bank) {
                    const fs::path bank_dir = (persist_bank ? h3_persist_root() : h3_bank_root()) / job->id;
                    std::error_code error;
                    fs::create_directories(bank_dir, error);
                    if (error) {
                        res.status = 500;
                        res.set_content(json({{"error", "could not create MiniMax-H3 job directory"},
                                              {"message", error.message()}})
                                            .dump(),
                                        "application/json");
                        return;
                    }
                    job->h3_bank_dir = bank_dir.string();
                    for (size_t index = 0; index < references.size(); index++) {
                        std::string& source            = references[index].audio_path;
                        const std::string& upload_part = reference_audio_parts[index];
                        if (source.empty() && upload_part.empty()) {
                            continue;
                        }
                        std::vector<uint8_t> wav;
                        if (!upload_part.empty()) {
                            const std::string& bytes = req.form.get_file(upload_part).content;
                            wav.assign(bytes.begin(), bytes.end());
                        } else if (!decode_h3_wav(source, wav)) {
                            wav.clear();
                        }
                        if (wav.empty()) {
                            res.status = 400;
                            res.set_content(h3_error("reference " + std::to_string(index) +
                                                     " audio must be non-empty WAV bytes"),
                                            "application/json");
                            return;
                        }
                        const fs::path staged = bank_dir / ("ref_" + std::to_string(index) + ".wav");
                        std::ofstream output(staged, std::ios::binary | std::ios::trunc);
                        output.write(reinterpret_cast<const char*>(wav.data()),
                                     static_cast<std::streamsize>(wav.size()));
                        if (!output.good()) {
                            res.status = 500;
                            res.set_content(R"({"error":"could not stage MiniMax-H3 reference audio"})",
                                            "application/json");
                            return;
                        }
                        source = staged.string();
                    }
                }
                job->h3_references    = std::move(references);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }
            manager.cv.notify_one();

            json segment_plans = json::array();
            int total_frames   = 0;
            for (size_t index = 0; index < segments.size(); ++index) {
                total_frames += segments[index].frames;
                segment_plans.push_back({{"index", static_cast<int>(index)},
                                         {"task", segments[index].task},
                                         {"requested_frames", segments[index].requested_frames},
                                         {"frames", segments[index].frames},
                                         {"duration_seconds",
                                          static_cast<double>(segments[index].frames) / kMiniMaxH3Fps},
                                         {"latent_frames", h3_video_latent_t(segments[index].frames)},
                                         {"audio_latent_frames", h3_audio_latent_t(segments[index].frames)}});
            }
            res.status = 202;
            res.set_content(json({{"id", job->id},
                                  {"kind", "minimax_h3"},
                                  {"status", async_job_status_name(job->status)},
                                  {"created", job->created_at},
                                  {"poll_url", "/sdcpp/v1/jobs/" + job->id},
                                  {"media_url", "/sdcpp/v1/jobs/" + job->id + "/media"},
                                  {"cancel_url", "/sdcpp/v1/jobs/" + job->id + "/cancel"},
                                  {"segments", static_cast<int>(segments.size())},
                                  // LTX parity, and deliberately constant: H3 keeps no latent
                                  // bank, so a chain always starts at shot 0, nothing is ever
                                  // retaken in place, and there is no bank id to resume.
                                  {"resume_from", 0},
                                  {"retake_segment", -1},
                                  {"resume_job_id", nullptr},
                                  {"task", job_task},
                                  {"segment_plans", std::move(segment_plans)},
                                  {"width", width},
                                  {"height", height},
                                  {"fps", kMiniMaxH3Fps},
                                  {"frames", total_frames},
                                  {"duration_seconds", static_cast<double>(total_frames) / kMiniMaxH3Fps},
                                  {"references", {{"images", ref_images},
                                                  {"videos", ref_videos},
                                                  {"audios", ref_audios}}},
                                  // Every LTX field this request carried that H3 accepted and
                                  // discarded. Named back rather than dropped in silence.
                                  {"ignored", compat.ignored}})
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

    // Filesystem-only cleanup, mirroring DELETE /ltx/v1/job. The supervisor answers this itself
    // when isolation is on, so a client can tidy up after an /unload without waking the worker.
    svr.Delete("/h3/v1/job", [runtime](const httplib::Request& req, httplib::Response& res) {
        const std::string job_id = req.get_param_value("id");
        {
            AsyncJobManager& manager = *runtime->async_job_manager;
            std::lock_guard<std::mutex> lock(manager.mutex);
            const auto it = manager.jobs.find(job_id);
            if (it != manager.jobs.end() && (it->second->status == AsyncJobStatus::Queued ||
                                             it->second->status == AsyncJobStatus::Generating)) {
                res.status = 409;
                res.set_content(R"({"error":"cannot delete an active MiniMax-H3 job"})", "application/json");
                return;
            }
        }
        bool deleted = false;
        std::string error_message;
        if (!h3_delete_bank(job_id, deleted, error_message)) {
            res.status = 400;
            res.set_content(h3_error(error_message), "application/json");
            return;
        }
        res.set_content(json({{"status", deleted ? "deleted" : "missing"}, {"deleted", deleted}}).dump(),
                        "application/json");
    });
}
