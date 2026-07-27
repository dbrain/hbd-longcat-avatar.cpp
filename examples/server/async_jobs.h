#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>


#include "runtime.h"

enum class AsyncJobKind {
    ImgGen,
    VidGen,
};

enum class AsyncJobStatus {
    Queued,
    Generating,
    Completed,
    Failed,
    Cancelled,
};

const char* async_job_kind_name(AsyncJobKind kind);
const char* async_job_status_name(AsyncJobStatus status);

// One LTX Prompt Relay beat as it arrived on the wire. Frame indices are on
// that shot's own rendered timeline (the caller adds its own seam drop).
struct LtxSegmentBeat {
    int frame      = 0;
    std::string text;
    float strength = 0.f;
    float window   = -1.f;
};

// One LTX TASS character reference as it arrived on the wire. `image` is either
// a base64 payload or a trusted absolute path; `source_id` zero means "assign
// 2, 3, 4, ... in array order", and `match_target` selects resizing to the
// render bucket instead of keeping the sheet's native resolution.
struct LtxCharacterRef {
    std::string image;
    int source_id     = 0;
    bool match_target = false;
};

struct AsyncGenerationJob {
    std::string id;
    AsyncJobKind kind     = AsyncJobKind::ImgGen;
    AsyncJobStatus status = AsyncJobStatus::Queued;
    int64_t created_at    = unix_timestamp_now();
    int64_t started_at    = 0;
    int64_t completed_at  = 0;
    ImgGenJobRequest img_gen;
    VidGenJobRequest vid_gen;
    // Non-empty only for /wan/v1/generate.  The generic async job lifecycle and
    // result schema are deliberately shared with regular video generation.
    std::vector<std::string> wan_vace_prompts;
    std::string wan_vace_bank_dir;
    std::string wan_vace_bank_id;
    int wan_vace_resume_from = 0;
    // Non-empty only for /ltx/v1/generate. The same async job and media result
    // lifecycle is used for LTX and Wan chains.
    std::vector<std::string> ltx_prompts;
    // Variant identifiers, aligned to ltx_prompts. The worker leases these
    // only at core-provided segment boundaries.
    std::vector<std::string> ltx_segment_models;
    std::string ltx_default_model = "base";
    // Top-level V2V selector is retained separately because SDGenerationParams
    // deliberately has no LTX-specific transport fields.  Mode 0 is LipDub.
    int ltx_v2v_mode = 0;
    int ltx_relip_ref_tstride = 1;
    std::vector<int> ltx_segment_frames;
    std::vector<int> ltx_segment_scene_cuts;
    std::vector<std::string> ltx_segment_init_images;
    std::vector<std::vector<std::string>> ltx_segment_keyframes;
    std::vector<std::vector<int>> ltx_segment_keyframe_indices;
    std::vector<std::vector<std::string>> ltx_segment_control_frames;
    // TASS overlap character references (LTX-Best-Face-ID sheets). Applies to
    // every segment of the request: a sheet is an identity, not a shot.
    std::vector<LtxCharacterRef> ltx_character_refs;
    // Zero inherits the checkpoint's trained default of 1.0.
    float ltx_tass_phase_scale = 0.f;
    std::vector<int> ltx_segment_v2v_modes;
    std::vector<float> ltx_segment_v2v_strengths;
    std::vector<std::string> ltx_segment_v2v_guide_latent_paths;
    // Owned stage parameter views for the optional LTX hires chain.  The
    // generation objects retain model-path and sigma backing storage until the
    // worker has completed the request.
    std::vector<SDGenerationParams> ltx_hires_stages;
    std::vector<sample_method_t> ltx_hires_stage_methods;
    std::vector<float> ltx_hires_stage_cfgs;
    bool ltx_emit_stages = false;
    std::string ltx_bank_dir;
    std::string ltx_bank_id;
    // Durable full-timeline WAV inputs staged alongside the LTX latent bank.
    std::string ltx_chain_audio_dir;
    std::string ltx_chain_audio_full;
    std::string ltx_chain_audio_track;
    int ltx_chain_audio_offset_frames = 0;
    // Retake/seam/audio policy is durable job state rather than an endpoint-only
    // concern: a queued job must retain precisely the contract that was staged.
    int ltx_retake_segment = -1;
    int ltx_cont_seam_drop_frames = 0;
    std::vector<int> ltx_segment_seam_drop_frames;
    std::vector<std::string> ltx_segment_audio_full;
    std::vector<std::string> ltx_segment_audio_track;
    // Per-shot Prompt Relay beats and sampling overrides. Empty vectors mean
    // "nothing per-shot"; within a populated vector, an inert entry (zero beats,
    // negative seed, non-positive steps, negative cfg, empty string) inherits.
    std::vector<std::vector<LtxSegmentBeat>> ltx_segment_beats;
    std::vector<int64_t> ltx_segment_seeds;
    std::vector<int> ltx_segment_steps;
    std::vector<float> ltx_segment_cfg;
    std::vector<std::string> ltx_segment_negative_prompts;
    int ltx_resume_from = 0;
    int ltx_cont_latent_frames = 3;
    bool ltx_emit_segments = false;
    std::vector<std::string> result_images_b64;
    std::string result_media_b64;
    std::string result_media_mime_type;
    int result_frame_count = 0;
    int result_fps         = 0;
    bool cancel_requested  = false;
    std::string error_code;
    std::string error_message;
};

struct AsyncJobManager {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<std::string, std::shared_ptr<AsyncGenerationJob>> jobs;
    std::unordered_map<std::string, int64_t> expired_jobs;
    std::deque<std::string> queue;
    uint64_t next_id              = 0;
    bool stop                     = false;
    size_t max_pending_jobs       = 64;
    int64_t completed_ttl_seconds = 600;
    int64_t failed_ttl_seconds    = 600;
};

void purge_expired_jobs(AsyncJobManager& manager);
size_t count_pending_jobs(const AsyncJobManager& manager);
// True while any job is Queued or Generating. Used by the child's /health `busy` field so the
// supervisor's in_flight() (and therefore Koblem's GPU gate) can see async renders, which return
// 202 immediately and would otherwise look idle for their whole duration.
bool async_job_in_flight(AsyncJobManager* manager);
std::string make_async_job_id(AsyncJobManager& manager);
bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job);
json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job);
bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         std::string& error_message);
bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         std::string& error_message);
void async_job_worker(ServerRuntime& runtime);
