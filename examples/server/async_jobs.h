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

// Process-local Wan continuation bank.  Completed windows are retained as raw
// frames plus their sampled latent so a cancelled/failed request can resume at
// the exact next VACE seam without re-rendering its prefix.
struct WanVaceResumeBank {
    ~WanVaceResumeBank();
    std::vector<sd_image_t> prefix_frames;
    std::vector<sd_image_t> control_tail;
    std::vector<float> latent;
    int completed_segments = 0;
    int latent_width = 0;
    int latent_height = 0;
    int latent_frames = 0;
    int latent_channels = 0;
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
    std::shared_ptr<WanVaceResumeBank> wan_vace_bank;
    std::string wan_vace_bank_dir;
    int wan_vace_resume_from = 0;
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
