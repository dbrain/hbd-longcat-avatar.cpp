// worker_session.h — parent-side handle on the longcat-avatar render worker.
//
// Shape mirrors qwen3-tts.cpp/src/worker_session.h. One worker child per
// server process, owns sd_ctx + CUDA. Parent stays CUDA-free until
// /v1/admin/unload SIGKILLs the child (which clears ALL VRAM — primary
// context + cuBLAS workspace + cubin cache).

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

namespace longcat_avatar {

struct RenderResult {
    bool ok = false;
    std::string error;
    std::vector<uint8_t> video_bytes;
    int   segments_rendered = 0;
    int   frame_count       = 0;
    int   fps               = 25;
    double render_sec       = 0.0;
};

class WorkerSession {
public:
    explicit WorkerSession(const char* argv0, std::vector<std::string> extra_argv = {});
    ~WorkerSession();

    // Spawn the worker if it isn't running. Sends a LOAD_REQ wait-for-ok
    // gating handshake. Returns true once the worker is ready to accept
    // RENDER_REQ. Sets last_error_ on failure.
    bool ensure_loaded();

    // SIGKILL + waitpid. Idempotent.
    void shutdown();

    bool is_alive() const { return pid_ > 0; }
    pid_t pid() const     { return pid_; }
    const std::string& last_error() const { return last_error_; }

    // Drive a render through the worker. JSON is the SDGenerationParams
    // JSON (server's "generate" body). image/audio are raw bytes (png/jpg,
    // wav). Caller serialises via the in_flight gate; this method takes
    // io_mutex_ for the full request/response round-trip.
    RenderResult render(const std::string& gen_json,
                        const std::vector<uint8_t>& image,
                        const std::vector<uint8_t>& audio);

private:
    void kill_worker_locked();

    std::string              argv0_;
    std::vector<std::string> extra_argv_;
    pid_t                    pid_         = -1;
    int                      fd_          = -1;
    bool                     loaded_      = false;
    mutable std::mutex       io_mutex_;
    std::string              last_error_;
    std::atomic<uint32_t>    next_req_id_{1};
};

// Child-side dispatch loop. Called from main() when --worker <fd> is on the
// command line. Owns the sd_ctx + warm weights; services RENDER_REQ until
// EOF / SHUTDOWN. Returns the process exit code (0 = clean shutdown).
int run_worker_loop(int fd, int argc, const char** argv);

} // namespace longcat_avatar
