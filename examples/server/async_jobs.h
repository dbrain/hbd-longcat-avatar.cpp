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
    std::vector<int> ltx_segment_frames;
    std::vector<int> ltx_segment_scene_cuts;
    std::vector<std::string> ltx_segment_init_images;
    std::vector<std::vector<std::string>> ltx_segment_control_frames;
    std::vector<int> ltx_segment_v2v_modes;
    std::vector<float> ltx_segment_v2v_strengths;
    std::vector<std::string> ltx_segment_v2v_guide_latent_paths;
    std::string ltx_bank_dir;
    std::string ltx_bank_id;
    // Durable full-timeline WAV inputs staged alongside the LTX latent bank.
    std::string ltx_chain_audio_dir;
    std::string ltx_chain_audio_full;
    std::string ltx_chain_audio_track;
    int ltx_chain_audio_offset_frames = 0;
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
