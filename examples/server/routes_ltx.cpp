// routes_ltx.cpp — LTXAV multi-segment video chain API (worker-isolated, async).
//
// POST /ltx/v1/generate submits a chain render. The whole chain runs in the CUDA child
// (worker isolation) via worker->render_video_chain(), or in-process when no worker is
// set; both go through the shared run_vid_chain_job(). Poll/cancel reuse the existing
// /sdcpp/v1/jobs/{id} endpoints (registered by register_sdcpp_api_endpoints), and the GPU
// gate uses the /v1/admin/{drain,unload,load} endpoints (register_sdcpp_admin_endpoints).
//
// Request (JSON, or multipart with a `request` part + wav parts: `audio_<i>` pre-sliced per-segment
// drive (legacy), `audio_full`/`audio_track` whole-timeline drive+deliverable, or
// `audio_full_<i>`/`audio_track_<i>` PER-SHOT drive+deliverable — each per-shot clip starts at its
// own shot and is composited over the whole-timeline pair, so a music bed and per-shot VO combine):
//   { "segments": [ {"prompt": "..."}, ... ]   // OR "prompts": ["...", ...]
//     "n_segments": N,                          // optional; defaults to segments.length
//     "cont_latent_frames": 3,                  // K overlap latent frames
//     "cont_seam_drop_frames": 24,              // optional: pin the seam trim (default: 8*K when
//                                               //   the engine owns the audio). Pad each
//                                               //   continuation's render by the SAME number.
//     "segment_seam_drop_frames": [0, 16, ...], // optional, per shot: frames[i] - visible[i].
//                                               //   Preferred — produced == requested even if the
//                                               //   caller and engine classify a shot differently.
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
//                                                 //   PREFER by-reference: segments[i].v2v_source_job_id
//                                                 //   (opaque job id) + v2v_source_segment (int) — the
//                                                 //   engine resolves those to the banked seg_<n>.bin in
//                                                 //   its own artifact roots (no path on the wire). A
//                                                 //   raw segments[i].v2v_source_latent_path is still
//                                                 //   honoured (engine-internal). Absent => fall back
//                                                 //   to control_frames (pixel encode).
//     "persist": false,                           // when true a FRESH job's artifacts land in the
//                                                 //   persist root (LTX_PERSIST_DIR, never swept) so a
//                                                 //   saved shot's latents survive as retake sources;
//                                                 //   default false => FIFO root (swept, LTX_JOB_KEEP).
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
#include <cstdint>
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

// Root for jobs the client marked persist:true (saved Director shots). NEVER swept — user data
// lives here until an explicit DELETE. Overridable via LTX_PERSIST_DIR. A sibling of the FIFO
// root so a job id resolves in exactly one of the two.
static fs::path ltx_persist_root() {
    const char* e = getenv("LTX_PERSIST_DIR");
    return fs::path((e != nullptr && e[0] != '\0') ? e : "/var/lib/ltx-video/persist");
}

// A job id arriving from the wire (resume_job_id, v2v_source_job_id, DELETE ?id) must be a single
// path component. The engine resolves ids against its own roots, so this is the confinement guard:
// a crafted id can never traverse out of a root.
static bool ltx_id_is_safe(const std::string& id) {
    return !id.empty() && id.find('/') == std::string::npos && id.find('\\') == std::string::npos &&
           id.find("..") == std::string::npos;
}

// Resolve an opaque job id to its artifact dir, checking the persist root first, then the FIFO
// root. Returns an empty path for an unsafe id or when neither root holds it. This is how a
// by-reference v2v source / resume finds its banked latents without a path ever crossing the wire.
static fs::path ltx_resolve_job_dir(const std::string& id) {
    if (!ltx_id_is_safe(id)) {
        return {};
    }
    std::error_code ec;
    fs::path        p = ltx_persist_root() / id;
    if (fs::exists(p, ec)) {
        return p;
    }
    p = ltx_job_root() / id;
    if (fs::exists(p, ec)) {
        return p;
    }
    return {};
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

// Blackwell's F16 LTX text cross-attention is not repeatable for very short query
// sequences.  The smallest demonstrated stable sequence is 128 latent tokens
// (256x256/9f); accepting smaller clips would knowingly enqueue a generation that
// can change for the same seed.  This is an API admission check, not a sampler
// fallback: supported requests retain their model, VRAM use and timing unchanged.
static bool ltx_validate_deterministic_token_floor(const ServerRuntime& runtime,
                                                   const json& chain,
                                                   int n_segments,
                                                   std::string& error) {
    SDGenerationParams params = *runtime.default_gen_params;
    if (!params.from_json_str(chain.dump())) {
        // The normal worker-side parser reports malformed requests with its existing
        // error path; this check must not turn that into a misleading determinism error.
        return true;
    }

    const int64_t spatial_tokens =
        static_cast<int64_t>(params.get_resolved_width() / 32) * (params.get_resolved_height() / 32);
    int default_frames = params.video_frames;
    if (default_frames <= 0) {
        return true;  // normal validation owns this malformed request
    }

    const json* segments = chain.contains("segments") && chain["segments"].is_array()
                               ? &chain["segments"]
                               : nullptr;
    for (int seg = 0; seg < n_segments; ++seg) {
        int frames = default_frames;
        if (segments != nullptr && seg < static_cast<int>(segments->size()) && (*segments)[seg].is_object()) {
            const auto it = (*segments)[seg].find("frames");
            if (it != (*segments)[seg].end() && it->is_number_integer() && it->get<int>() > 0) {
                frames = it->get<int>();
            }
        }
        const int64_t latent_frames = (static_cast<int64_t>(frames) - 1) / 8 + 1;
        const int64_t tokens        = spatial_tokens * latent_frames;
        if (tokens < 128) {
            error = "LTX deterministic-mode minimum is 128 latent tokens; segment " +
                    std::to_string(seg + 1) + " resolves to " + std::to_string(tokens) +
                    " (" + std::to_string(params.get_resolved_width()) + "x" +
                    std::to_string(params.get_resolved_height()) + ", " +
                    std::to_string(frames) + " frames). Increase resolution and/or length.";
            return false;
        }
    }
    return true;
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
            std::string audio_full_part;   // whole-timeline drive clip (engine slices it)
            std::string audio_track_part;  // whole-timeline deliverable track (engine muxes it)
            // Per-shot clips (index, bytes), sparse: audio_full_<i> / audio_track_<i>.
            std::vector<std::pair<int, std::string>> shot_audio_full_parts;
            std::vector<std::pair<int, std::string>> shot_audio_track_parts;
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
                // ENGINE-OWNED AUDIO: the whole timeline, once. `audio_full` drives lip-sync,
                // `audio_track` is the deliverable; either may be omitted. When present these
                // SUPERSEDE the pre-sliced audio_<i> parts above — the engine cuts each segment's
                // window from its own seam trim instead of trusting the client to predict it.
                if (req.form.has_file("audio_full")) {
                    audio_full_part = req.form.get_file("audio_full").content;
                }
                if (req.form.has_file("audio_track")) {
                    audio_track_part = req.form.get_file("audio_track").content;
                }
                // PER-SHOT audio: `audio_full_<i>` / `audio_track_<i>`. Each clip starts at its own
                // shot and is composited onto the whole-clip bed above (which may be absent). Sparse
                // by design — a shot with no part contributes nothing — so probe every index rather
                // than stopping at the first gap, unlike the contiguous pre-sliced audio_<i> loop.
                int n_seg_hint = body.value("n_segments", 0);
                if (body.contains("segments") && body["segments"].is_array()) {
                    n_seg_hint = std::max(n_seg_hint, (int)body["segments"].size());
                }
                if (body.contains("prompts") && body["prompts"].is_array()) {
                    n_seg_hint = std::max(n_seg_hint, (int)body["prompts"].size());
                }
                for (int i = 0; i < n_seg_hint; ++i) {
                    std::string fk = "audio_full_" + std::to_string(i);
                    std::string tk = "audio_track_" + std::to_string(i);
                    if (req.form.has_file(fk)) {
                        shot_audio_full_parts.emplace_back(i, req.form.get_file(fk).content);
                    }
                    if (req.form.has_file(tk)) {
                        shot_audio_track_parts.emplace_back(i, req.form.get_file(tk).content);
                    }
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
            // Progressive WITHIN-shot upscale-stage previews (opt-in). When true, the engine banks
            // seg_<n>_stage<k>.webm (a fast low-res base, then the mid-res refine) as each shot renders,
            // and the job status lists them under "partials" with their stage, so a client sees a rough
            // preview ASAP that sharpens. Implies emit_segments' final per-shot webm too. Default false =
            // byte-identical (no extra decode/encode, no stage partials).
            chain["emit_stages"] = body.value("emit_stages", false);
            std::string output_format = body.value("output_format", std::string("webm"));

            std::string determinism_error;
            if (!ltx_validate_deterministic_token_floor(*runtime, chain, n_segments, determinism_error)) {
                res.status = 422;
                res.set_content(json({{"error", "nondeterministic_request"},
                                      {"message", determinism_error}}).dump(),
                                "application/json");
                return;
            }

            // V2V source-by-reference: a retake names its banked source by opaque job id + segment
            // (segments[i].v2v_source_job_id / v2v_source_segment) — never a filesystem path. Resolve
            // each to the engine-internal absolute latent path (v2v_source_latent_path) the chain
            // renderer already consumes; the id is confined to the engine's own artifact roots and the
            // path never crosses the wire. A ref that doesn't resolve (source aged out of the FIFO
            // store / never existed) fails the request before the job is registered.
            if (chain.contains("segments") && chain["segments"].is_array()) {
                for (auto& seg : chain["segments"]) {
                    if (!seg.is_object()) {
                        continue;
                    }
                    auto jit = seg.find("v2v_source_job_id");
                    if (jit == seg.end() || !jit->is_string() || jit->get<std::string>().empty()) {
                        continue;
                    }
                    const std::string src_id  = jit->get<std::string>();
                    int               src_seg = 0;
                    if (auto sit = seg.find("v2v_source_segment");
                        sit != seg.end() && sit->is_number_integer()) {
                        src_seg = sit->get<int>();
                    }
                    fs::path        src_dir = ltx_resolve_job_dir(src_id);
                    fs::path        lat = src_dir.empty() ? fs::path()
                                                          : src_dir / ("seg_" + std::to_string(src_seg) + ".bin");
                    std::error_code lec;
                    if (lat.empty() || !fs::exists(lat, lec)) {
                        res.status = 404;
                        res.set_content(json({{"error", "v2v_source_not_found"},
                                              {"message", "banked latent for v2v_source_job_id '" + src_id +
                                                              "' segment " + std::to_string(src_seg) +
                                                              " not found"}})
                                            .dump(),
                                        "application/json");
                        return;
                    }
                    seg["v2v_source_latent_path"] = lat.string();
                }
            }

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

            // Resolve the artifact dir: an existing dir on resume, else this job's own. A fresh job
            // marked persist:true is keyed into the persist root (never swept); everything else lands
            // in the FIFO root (swept to LTX_JOB_KEEP). A resume finds its dir in whichever root holds
            // it, so a persisted chain keeps accumulating there.
            bool            persist = body.value("persist", false);
            fs::path        root    = persist ? ltx_persist_root() : ltx_job_root();
            fs::path        job_dir;
            int             resume_from = 0;
            std::error_code ec;
            if (!resume_job_id.empty()) {
                job_dir = ltx_resolve_job_dir(resume_job_id);
                if (job_dir.empty()) {
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

            // ENGINE-OWNED AUDIO: stage the whole-timeline clips next to the per-segment dir.
            // Written on the FIRST submit and then simply re-discovered by path on every
            // resume/retake (same trick as chain_audio_dir above) — a retake of shot 12 of a
            // 3-minute clip needs no re-upload and, crucially, no client-side hunt for "which
            // chunk of audio does shot 12 use": the engine re-derives that window itself from
            // the banked seg_<i>.len prefix it already replays.
            if (!audio_full_part.empty() || !audio_track_part.empty()) {
                fs::create_directories(job_dir / "audio", ec);
            }
            if (!audio_full_part.empty()) {
                ltx_write_blob((job_dir / "audio" / "full.wav").string(), audio_full_part);
            }
            if (!audio_track_part.empty()) {
                ltx_write_blob((job_dir / "audio" / "track.wav").string(), audio_track_part);
            }
            // PER-SHOT audio, staged next to the whole-clip bed and re-discovered by path on every
            // resume/retake exactly like it. Written on the first submit only; the arrays below are
            // rebuilt from what is on disk, so a retake needs no re-upload.
            if (!shot_audio_full_parts.empty() || !shot_audio_track_parts.empty()) {
                fs::create_directories(job_dir / "audio", ec);
                for (const auto& [idx, bytes] : shot_audio_full_parts) {
                    ltx_write_blob(
                        (job_dir / "audio" / ("shot_" + std::to_string(idx) + "_full.wav")).string(), bytes);
                }
                for (const auto& [idx, bytes] : shot_audio_track_parts) {
                    ltx_write_blob(
                        (job_dir / "audio" / ("shot_" + std::to_string(idx) + "_track.wav")).string(), bytes);
                }
            }
            {
                // n_segments is authoritative here (the chain object is already built).
                int n_seg = chain.value("n_segments", 0);
                if (n_seg <= 0 && chain.contains("prompts") && chain["prompts"].is_array()) {
                    n_seg = (int)chain["prompts"].size();
                }
                json full_paths  = json::array();
                json track_paths = json::array();
                bool any_full = false, any_track = false;
                for (int i = 0; i < n_seg; ++i) {
                    auto fp = job_dir / "audio" / ("shot_" + std::to_string(i) + "_full.wav");
                    auto tp = job_dir / "audio" / ("shot_" + std::to_string(i) + "_track.wav");
                    bool hf = fs::exists(fp, ec);
                    bool ht = fs::exists(tp, ec);
                    full_paths.push_back(hf ? fp.string() : std::string());
                    track_paths.push_back(ht ? tp.string() : std::string());
                    any_full  = any_full || hf;
                    any_track = any_track || ht;
                }
                if (any_full) {
                    chain["segment_audio_full"] = full_paths;
                }
                if (any_track) {
                    chain["segment_audio_track"] = track_paths;
                }
            }
            if (fs::exists(job_dir / "audio" / "full.wav", ec)) {
                chain["chain_audio_full"] = (job_dir / "audio" / "full.wav").string();
            }
            if (fs::exists(job_dir / "audio" / "track.wav", ec)) {
                chain["chain_audio_track"] = (job_dir / "audio" / "track.wav").string();
            }
            // Timeline frame at which both clips' t=0 sits (default 0 = clip starts with the render).
            if (body.contains("chain_audio_offset_frames")) {
                chain["chain_audio_offset_frames"] = body.value("chain_audio_offset_frames", 0);
            }
            // Seam trim pin. Send the SAME number the caller padded each continuation's render by:
            // picture and track are then frame-exact at every seam and the produced segment length
            // equals the requested visible length. Omit to let the engine derive 8*K.
            if (body.contains("cont_seam_drop_frames")) {
                chain["cont_seam_drop_frames"] = body.value("cont_seam_drop_frames", 0);
            }
            // Per-shot trim (render - visible). Preferred over the scalar: it holds even when the
            // caller and the engine disagree about which shots are fresh.
            if (body.contains("segment_seam_drop_frames")) {
                chain["segment_seam_drop_frames"] = body["segment_seam_drop_frames"];
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
            // Only ever sweep the FIFO root — the persist store is user data and is never evicted.
            // Passing job_dir as keep_dir is harmless when it lives in the persist root (it simply
            // won't match anything the FIFO sweep iterates).
            ltx_sweep_old_jobs(ltx_job_root(), job_dir);

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

    // DELETE /ltx/v1/job?id=<job_id> — remove a job's artifact dir from whichever root holds it
    // (persist or FIFO). koblem calls this on project-delete / tier-wipe so persisted user data is
    // cleaned up on demand — the persist root is otherwise never swept. Confined to the roots by the
    // single-component id guard. Idempotent: deleting a missing id succeeds with deleted:false.
    svr.Delete("/ltx/v1/job", [](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.has_param("id") ? req.get_param_value("id") : std::string();
        if (!ltx_id_is_safe(id)) {
            res.status = 400;
            res.set_content(R"({"error":"invalid id"})", "application/json");
            return;
        }
        std::error_code ec;
        bool            removed = false;
        for (const fs::path& r : {ltx_persist_root(), ltx_job_root()}) {
            fs::path d = r / id;
            if (fs::exists(d, ec)) {
                fs::remove_all(d, ec);
                removed = true;
            }
        }
        res.status = 200;
        res.set_content(json({{"deleted", removed}, {"id", id}}).dump(), "application/json");
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
            // job aged out of RAM; serve from disk — check both roots (a persisted job lives in
            // the persist store), falling back to the FIFO path so a truly-absent id still 404s.
            dir = ltx_resolve_job_dir(job_id);
            if (dir.empty()) {
                dir = ltx_job_root() / job_id;
            }
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

    // GET /sdcpp/v1/jobs/{id}/segments/{n}[?stage=<k>] — stream a banked preview webm from the job's
    // artifact dir. Default / stage=4 = the FINAL full-res shot (seg_<n>.webm). ?stage=1 or 2 = an
    // intermediate WITHIN-shot progressive preview (seg_<n>_stage<k>.webm: 1=base low-res, 2=mid-res),
    // banked only when the request set emit_stages. This is the progressive-delivery fetch: the client
    // polls /sdcpp/v1/jobs/{id}, reads the "partials" list (each carries its stage + url), and pulls
    // each as it lands. 404 until that atomic webm exists. Mirrors the /media handler: resolve the dir
    // from RAM (falling back to LTX_JOB_DIR/<id> after the in-RAM TTL) and stream.
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
            // job aged out of RAM; serve from disk — check both roots (a persisted job lives in
            // the persist store), falling back to the FIFO path so a truly-absent id still 404s.
            dir = ltx_resolve_job_dir(job_id);
            if (dir.empty()) {
                dir = ltx_job_root() / job_id;
            }
        }
        // stage=4 (or absent) is the final seg_<n>.webm; stage 1/2 select the intermediate previews.
        const std::string stage_q = req.has_param("stage") ? req.get_param_value("stage") : "";
        fs::path          seg_webm;
        if (!stage_q.empty() && stage_q != "4" &&
            stage_q.find_first_not_of("0123456789") == std::string::npos) {
            seg_webm = dir / ("seg_" + seg_n + "_stage" + stage_q + ".webm");
        } else {
            seg_webm = dir / ("seg_" + seg_n + ".webm");
        }
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
