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

// Result of a FLUX.2 image render driven through the worker child.
struct ImageRenderResult {
    bool ok = false;
    std::string error;
    std::string output_format = "png";
    std::vector<std::string> images;  // base64-encoded image bytes, one per batch
    double render_sec = 0.0;
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

    // Drive a FLUX.2 image render through the worker. `request_json` is the raw
    // /sdcpp/v1/img_gen body (the init/edit image rides inside as base64). The
    // child re-parses it and runs the in-process compute. Same io_mutex_ +
    // current_render_req_id_ machinery as render(), so cancel_in_flight() works.
    ImageRenderResult render_image(const std::string& request_json);

    // Drive an LTXAV multi-segment video chain through the worker. `chain_json` is the
    // chain request (base gen-params + prompts[] + n_segments + cont_latent_frames +
    // chain_audio_dir + inline base64 init image). The child re-parses it and runs
    // generate_video_chain via run_vid_chain_job. Same io_mutex_ + cancel machinery as
    // render(); result.video_bytes is the encoded container (webm).
    RenderResult render_video_chain(const std::string& chain_json);

    // Send CANCEL_REQ to the worker for the currently in-flight render.
    // Idempotent: no-op if no render is in flight (current_render_req_id_==0).
    // Safe to call from another thread WHILE render() holds io_mutex_ in its
    // blocking recv — uses a separate send_mutex_ so it can't deadlock. Does NOT
    // SIGKILL the worker; the worker bails cooperatively and stays warm.
    void cancel_in_flight();

private:
    void kill_worker_locked();

    std::string              argv0_;
    std::vector<std::string> extra_argv_;
    pid_t                    pid_         = -1;
    int                      fd_          = -1;
    bool                     loaded_      = false;
    mutable std::mutex       io_mutex_;
    // Held only while a frame is written to fd_. Distinct from io_mutex_ so a
    // cancel can be sent while io_mutex_ is held by the render's blocking recv.
    mutable std::mutex       send_mutex_;
    std::string              last_error_;
    std::atomic<uint32_t>    next_req_id_{1};
    // Published by render() AFTER the RENDER_REQ send, cleared on every return
    // path via RAII. Read by cancel_in_flight(). 0 ⇒ no render in flight.
    std::atomic<uint32_t>    current_render_req_id_{0};
};

// Child-side dispatch loop. Called from main() when --worker <fd> is on the
// command line. Owns the sd_ctx + warm weights; services RENDER_REQ until
// EOF / SHUTDOWN. Returns the process exit code (0 = clean shutdown).
int run_worker_loop(int fd, int argc, const char** argv);

} // namespace longcat_avatar
