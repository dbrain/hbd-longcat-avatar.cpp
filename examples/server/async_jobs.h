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
    std::vector<std::string> result_images_b64;
    std::string result_media_b64;
    std::string result_media_mime_type;
    int result_frame_count = 0;
    int result_fps         = 0;
    std::string error_code;
    std::string error_message;
};

struct AsyncJobManager {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<std::string, std::shared_ptr<AsyncGenerationJob>> jobs;
    std::unordered_map<std::string, int64_t> expired_jobs;
    std::deque<std::string> queue;
    std::string generating_job_id;  // id of the job currently on the GPU ("" = none)
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
// Ensure the DiT variant required by `target_variant` ("base"|"edit") is resident
// in sd_ctx, swapping (or reloading after an admin unload) if necessary. An empty
// target_variant means "use whatever is loaded" — but if the model was unloaded by
// admin, the currently-tracked variant is reloaded so the render has weights. Runs
// on the serial async worker thread (holds sd_ctx_mutex internally). Returns false
// and fills error_message on swap/reload failure.
bool ensure_variant_loaded(ServerRuntime& runtime,
                           const std::string& target_variant,
                           std::string& error_message);

void async_job_worker(ServerRuntime& runtime);
