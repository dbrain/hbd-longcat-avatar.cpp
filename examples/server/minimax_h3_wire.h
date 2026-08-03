#pragma once

// MiniMax-H3 wire arithmetic and durable-bank helpers.
//
// EVERYTHING IN HERE IS CUDA-FREE AND MODEL-FREE ON PURPOSE. Two processes answer H3 requests
// depending on deployment: the worker-isolated CUDA-less SUPERVISOR, and the CUDA child it
// proxies into. `GET /h3/v1/capabilities` and `DELETE /h3/v1/job` are answered by whichever of
// the two received them -- the supervisor must NOT cold-start a ~20 GB CUDA child to describe a
// frame grid or to remove a directory -- so the two answers have to be produced by the SAME code
// or a client would be debugging a phantom. Same reasoning, and the same shape, as
// `scan_lora_dir` / `lora_entry_json` in runtime.h.
//
// The frame/latent arithmetic is transcribed from the reference nodes
// (comfy_extras/nodes_minimax_h3.py: `align_frame_count`, `video_latent_t`, `temporal_shape`,
// `adapt_canvas`) and from diffusers' `sample_reference_video_frames`
// (modular_pipelines/minimax_h3/packing_ref2va.py). Get any of it wrong and nothing crashes --
// the layout simply describes a different sequence than the one the VAEs produced.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <json.hpp>

#include "model/diffusion/minimax_h3_layout.hpp"

// ── model constants ────────────────────────────────────────────────────────────────────────────
// H3 is a fixed-fps model: the 17n+5 frame grid, the 4x temporal VAE and the 40 Hz audio latent
// rate are all defined AT 24 fps. A request at another fps is not a slower render, it is a
// different (and wrong) audio length, which is why the route refuses one rather than rescaling.
//
// ★ The frame-grid arithmetic itself now lives in `model/diffusion/minimax_h3_layout.hpp` and
// this header FORWARDS to it. The route quotes the geometry in its 202 and the core builds the
// packed sequence from it; two copies drifting apart would not crash, it would just describe a
// different sequence than the one the VAEs produced.
constexpr int kMiniMaxH3Fps             = MiniMaxH3::FPS;
constexpr int kMiniMaxH3AudioLatentFps  = MiniMaxH3::AUDIO_LATENT_FPS;
constexpr int kMiniMaxH3AudioSampleRate = 32000;
// Qwen3-VL reads a reference video at 2 fps and merges the sampled frames in pairs.
constexpr int kMiniMaxH3QwenSampleFps     = 2;
constexpr int kMiniMaxH3QwenTemporalPatch = 2;
// Canvas: 32-pixel multiples (16x VAE * 2x2 patch), 768 short edge for the default canvas.
constexpr int kMiniMaxH3CanvasMultiple = 32;
constexpr int kMiniMaxH3BaseShortEdge  = 768;

// ⚠️ AREA CAP — 768*1344 is the reference's DEFAULT CANVAS, not a limit. Do not re-tighten it.
//
// This constant started at 768*1344 = 1,032,192 px, read out of the reference PRs, and enforcing
// it as validation REFUSED 1920x1088 (2,088,960 px) -- a resolution we actively intend to test.
// That was a bug: it turned a default into a limit.
//
// The reference's own node declares no cap at all:
//     io.Int.Input("width",  default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32)
//     io.Int.Input("height", default=768,  min=32, max=nodes.MAX_RESOLUTION, step=32)
// `MAX_PIXELS` appears ONLY inside `adapt_canvas()`, a helper that derives a canvas from an aspect
// ratio (and is applied to REFERENCE VIDEO canvases), never to the target canvas.
// Corroborating, though weaker because it is a commercial host's own copy: fal.ai documents every
// H3 endpoint as 2K -- 1440 short edge, ~3.7 MP at 21:9 (2976x1248).
//
// So the number below is a sanity bound, not a model constraint: it keeps a fat-fingered request
// from allocating absurdly while letting VRAM be the real limit, which is self-evidently
// diagnosable in a way a 400 is not.
constexpr int kMiniMaxH3MaxPixels = 2976 * 1248;  // 3,714,048 — fal's documented 2K maximum
// Reference-image sizing: "max" keeps a 2048 short edge for identity fidelity.
constexpr int kMiniMaxH3RefImageShortEdge = 2048;
// Reference counts the trained node exposes. Not architectural limits -- refusing at the
// boundary is simply cheaper than discovering it after a 20 GB model is resident.
constexpr int kMiniMaxH3MaxRefImages = 9;
constexpr int kMiniMaxH3MaxRefVideos = 3;
constexpr int kMiniMaxH3MaxRefAudios = 3;
constexpr int kMiniMaxH3MaxReferences = 12;
// Guidance-distilled: one forward per step, no CFG, no negative prompt.
constexpr float kMiniMaxH3SigmaShiftVideo = 12.0f;
constexpr float kMiniMaxH3SigmaShiftAudio = 3.0f;

// ── frame grid ─────────────────────────────────────────────────────────────────────────────────

inline int h3_align_frame_count(int frames) {
    return MiniMaxH3::align_frame_count(frames);
}

inline bool h3_frame_count_is_aligned(int frames) {
    return MiniMaxH3::frame_count_is_aligned(frames);
}

inline int h3_video_latent_t(int aligned_frames) {
    return MiniMaxH3::video_latent_t(aligned_frames);
}

inline int h3_audio_latent_t(int aligned_frames) {
    return MiniMaxH3::audio_latent_t(aligned_frames);
}

inline bool h3_canvas_is_legal(int width, int height) {
    return width >= kMiniMaxH3CanvasMultiple && height >= kMiniMaxH3CanvasMultiple &&
           width % kMiniMaxH3CanvasMultiple == 0 && height % kMiniMaxH3CanvasMultiple == 0 &&
           static_cast<int64_t>(width) * height <= kMiniMaxH3MaxPixels;
}

// The reference canvas rule: keep the aspect, put the short edge at 768, cap the area, then round
// each axis to a 32 multiple. Used to ANSWER a caller that sent an off-grid size -- the route only
// adopts it when the request opts in, so a caller that pinned a resolution never silently renders
// at another one.
inline void h3_adapt_canvas(int width, int height, int& out_width, int& out_height) {
    const double w = std::max(1, width);
    const double h = std::max(1, height);
    const double ratio = w / h;
    double nominal_w = ratio >= 1.0 ? kMiniMaxH3BaseShortEdge * ratio : kMiniMaxH3BaseShortEdge;
    double nominal_h = ratio >= 1.0 ? kMiniMaxH3BaseShortEdge : kMiniMaxH3BaseShortEdge / ratio;
    if (nominal_w * nominal_h > kMiniMaxH3MaxPixels) {
        const double scale = std::sqrt(kMiniMaxH3MaxPixels / (nominal_w * nominal_h));
        nominal_w *= scale;
        nominal_h *= scale;
    }
    const auto snap = [](double value) {
        const int multiple = kMiniMaxH3CanvasMultiple;
        return std::max(multiple, static_cast<int>(std::lround(value / multiple)) * multiple);
    };
    out_width  = snap(nominal_w);
    out_height = snap(nominal_h);
}

// ── reference video sampling ───────────────────────────────────────────────────────────────────

// A reference video is VAE-encoded whole, so its own length must land on the 17n+5 grid too. It is
// trimmed DOWN (the reference is an observation, not a duration request) and needs at least the
// 5-frame minimum, i.e. ~0.2 s at 24 fps.
inline int h3_trim_reference_frame_count(int frames) {
    int trimmed = frames;
    while (trimmed >= 5 && trimmed % 17 != 5) {
        trimmed--;
    }
    return trimmed < 5 ? 0 : trimmed;
}

// Python's round() is round-half-to-even, and `sample_reference_video_frames` is written against
// it. With a 24/2 stride the cursor only ever takes exact integers, so no tie is actually reached
// -- but transcribing the rounding rather than the reachable subset means a future stride change
// cannot silently shift every index by one.
inline int64_t h3_round_half_even(double value) {
    const double floor_value = std::floor(value);
    const double fraction    = value - floor_value;
    int64_t result           = static_cast<int64_t>(floor_value);
    if (fraction > 0.5) {
        result += 1;
    } else if (fraction == 0.5) {
        result += (result % 2 == 0) ? 0 : 1;
    }
    return result;
}

// Which frames of a 24 fps reference the CONDITIONER sees: every 12th, deduplicated by the rounded
// cursor. The VAE still encodes all `frame_count` frames -- these indices only select what Qwen3-VL
// is shown, and they must agree row-for-row with the block timestamps below.
inline std::vector<int> h3_sample_reference_frame_indices(int frame_count) {
    const double stride = static_cast<double>(kMiniMaxH3Fps) / kMiniMaxH3QwenSampleFps;
    std::vector<int> indices;
    double cursor = 0.0;
    while (h3_round_half_even(cursor) < frame_count) {
        const int64_t index = h3_round_half_even(cursor);
        if (indices.empty() || index > indices.back()) {
            indices.push_back(static_cast<int>(index));
        }
        cursor += stride;
    }
    return indices;
}

// One timestamp per MERGED 2-frame vision block: the mean of the pair's two 2 fps timestamps, with
// the last timestamp repeated when the sampled count is odd -- exactly the repeat-padding the
// conditioner applies to the frames themselves. The conditioner GGML_ASSERTs that these two counts
// agree, so they are produced together, here, and never by two independent callers.
inline std::vector<double> h3_reference_block_timestamps(size_t sampled_frames) {
    std::vector<double> timestamps;
    timestamps.reserve(sampled_frames + 1);
    for (size_t index = 0; index < sampled_frames; index++) {
        timestamps.push_back(static_cast<double>(index) / kMiniMaxH3QwenSampleFps);
    }
    if (timestamps.empty()) {
        return {};
    }
    while (timestamps.size() % kMiniMaxH3QwenTemporalPatch != 0) {
        timestamps.push_back(timestamps.back());
    }
    std::vector<double> block_timestamps;
    block_timestamps.reserve(timestamps.size() / kMiniMaxH3QwenTemporalPatch);
    for (size_t index = 0; index + kMiniMaxH3QwenTemporalPatch <= timestamps.size();
         index += kMiniMaxH3QwenTemporalPatch) {
        block_timestamps.push_back((timestamps[index] + timestamps[index + kMiniMaxH3QwenTemporalPatch - 1]) / 2.0);
    }
    return block_timestamps;
}

// ── durable bank ───────────────────────────────────────────────────────────────────────────────

inline std::filesystem::path h3_bank_root() {
    if (const char* configured = std::getenv("MINIMAX_H3_JOB_DIR");
        configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    return "/var/lib/minimax-h3/jobs";
}

inline std::filesystem::path h3_persist_root() {
    if (const char* configured = std::getenv("MINIMAX_H3_PERSIST_DIR");
        configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    return "/var/lib/minimax-h3/persist";
}

inline std::vector<std::filesystem::path> h3_bank_roots() {
    const std::filesystem::path persist   = h3_persist_root();
    const std::filesystem::path transient = h3_bank_root();
    if (persist == transient) {
        return {transient};
    }
    return {persist, transient};
}

inline bool h3_valid_bank_id(const std::string& id) {
    return !id.empty() && id.size() <= 128 &&
           std::all_of(id.begin(), id.end(), [](unsigned char ch) {
               return std::isalnum(ch) || ch == '_' || ch == '-';
           });
}

// Remove a job's staged reference payload. FILESYSTEM ONLY, which is what lets the supervisor
// answer `DELETE /h3/v1/job` without waking the worker -- the same call the LTX bank cleanup makes
// and for the same reason. Idempotent: a missing bank is `deleted:false`, not an error.
inline bool h3_delete_bank(const std::string& job_id, bool& deleted, std::string& error_message) {
    if (!h3_valid_bank_id(job_id)) {
        error_message = "id must be a MiniMax-H3 job id";
        return false;
    }
    for (const std::filesystem::path& root : h3_bank_roots()) {
        std::error_code error;
        const std::filesystem::path directory = root / job_id;
        if (!std::filesystem::is_directory(directory, error)) {
            continue;
        }
        const uintmax_t removed = std::filesystem::remove_all(directory, error);
        if (error) {
            error_message = error.message();
            return false;
        }
        deleted = deleted || removed > 0;
    }
    return true;
}

// ── capabilities ───────────────────────────────────────────────────────────────────────────────

// The static description of the model's request shape, plus -- when the caller supplies
// `width`/`height`/`frames` -- the exact geometry a request with those values would render at.
// Answering the plan here is what stops a UI guessing the 17n+5 rule client-side and drifting.
inline nlohmann::json h3_capabilities_json(int requested_width, int requested_height, int requested_frames) {
    nlohmann::json capabilities = {
        {"model", "minimax-h3"},
        {"tasks", nlohmann::json::array({"t2va", "fl2va", "ref2va"})},
        {"fps", kMiniMaxH3Fps},
        {"frame_grid", "17n+5"},
        {"min_frames", 5},
        {"batch_size", 1},
        {"guidance_distilled", true},
        {"supports_cfg", false},
        {"supports_negative_prompt", false},
        {"joint_audio", true},
        {"audio", {{"channels", 2},
                   {"sample_rate", kMiniMaxH3AudioSampleRate},
                   {"latent_fps", kMiniMaxH3AudioLatentFps},
                   {"latent_channels", 32}}},
        {"video", {{"spatial_compression", 16}, {"temporal_compression", 4}, {"latent_channels", 24}}},
        {"canvas", {{"multiple", kMiniMaxH3CanvasMultiple},
                    {"base_short_edge", kMiniMaxH3BaseShortEdge},
                    {"max_pixels", kMiniMaxH3MaxPixels}}},
        {"references", {{"max_images", kMiniMaxH3MaxRefImages},
                        {"max_videos", kMiniMaxH3MaxRefVideos},
                        {"max_audios", kMiniMaxH3MaxRefAudios},
                        {"max_total", kMiniMaxH3MaxReferences},
                        {"audio_only", false},
                        {"video_min_frames", 5},
                        {"video_frame_grid", "17n+5"},
                        {"conditioner_sample_fps", kMiniMaxH3QwenSampleFps},
                        {"image_short_edge_max", kMiniMaxH3RefImageShortEdge}}},
        {"sigma_shift", {{"video", kMiniMaxH3SigmaShiftVideo}, {"audio", kMiniMaxH3SigmaShiftAudio}}},
        {"endpoints", nlohmann::json::array({"POST /h3/v1/generate",
                                             "GET /h3/v1/capabilities",
                                             "DELETE /h3/v1/job"})},
        {"poll_url_template", "/sdcpp/v1/jobs/{id}"},
        {"media_url_template", "/sdcpp/v1/jobs/{id}/media"},
        {"cancel_url_template", "/sdcpp/v1/jobs/{id}/cancel"},
    };

    if (requested_frames > 0 || requested_width > 0 || requested_height > 0) {
        const int frames  = h3_align_frame_count(requested_frames > 0 ? requested_frames : 124);
        nlohmann::json plan = {
            {"requested_frames", requested_frames},
            {"frames", frames},
            {"fps", kMiniMaxH3Fps},
            {"duration_seconds", static_cast<double>(frames) / kMiniMaxH3Fps},
            {"latent_frames", h3_video_latent_t(frames)},
            {"audio_latent_frames", h3_audio_latent_t(frames)},
        };
        if (requested_width > 0 && requested_height > 0) {
            const bool legal   = h3_canvas_is_legal(requested_width, requested_height);
            int adapted_width  = 0;
            int adapted_height = 0;
            h3_adapt_canvas(requested_width, requested_height, adapted_width, adapted_height);
            // ★ Quote the canvas the ROUTE would render at, not the adapted one unconditionally.
            // `/h3/v1/generate` only adapts when the requested canvas is ILLEGAL (routes_h3.cpp),
            // so reporting 1376x768 for a perfectly legal 1280x704 made this document contradict
            // the 202 it is supposed to predict -- and a UI sizing its VRAM/duration estimate off
            // `latent_width` got a number for a sequence nothing will ever render.
            const int planned_width  = legal ? requested_width : adapted_width;
            const int planned_height = legal ? requested_height : adapted_height;
            plan["requested_width"]  = requested_width;
            plan["requested_height"] = requested_height;
            plan["width"]            = planned_width;
            plan["height"]           = planned_height;
            plan["canvas_is_legal"]  = legal;
            // The adapted canvas stays visible so a caller that means to send adapt_canvas:true
            // can still see what it would get.
            plan["adapted_width"]    = adapted_width;
            plan["adapted_height"]   = adapted_height;
            plan["latent_width"]     = planned_width / 16;
            plan["latent_height"]    = planned_height / 16;
        }
        capabilities["plan"] = std::move(plan);
    }
    return capabilities;
}
