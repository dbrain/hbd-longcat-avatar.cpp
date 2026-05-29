// worker_session.cpp — see worker_session.h.

#include "worker_session.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include <json.hpp>

#include "common/common.h"
#include "common/log.h"
#include "common/resource_owners.hpp"
#include "common/avatar_render.h"
#include "stable-diffusion.h"
#include "runtime.h"
#include "worker_ipc.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace longcat_avatar {

// ─── parent-side handle ──────────────────────────────────────────────────────

WorkerSession::WorkerSession(const char* argv0, std::vector<std::string> extra_argv)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)) {}

WorkerSession::~WorkerSession() {
    shutdown();
}

void WorkerSession::kill_worker_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int status = 0;
        while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
        pid_    = -1;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    loaded_ = false;
}

void WorkerSession::shutdown() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    kill_worker_locked();
}

bool WorkerSession::ensure_loaded() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ > 0 && fd_ >= 0 && loaded_) return true;

    // Re-spawn if pid_ is stale (child died).
    if (pid_ > 0) {
        int status = 0;
        pid_t r = waitpid(pid_, &status, WNOHANG);
        if (r != 0) {  // child already gone
            pid_ = -1;
        }
    }
    if (pid_ <= 0) {
        int parent_fd = -1;
        pid_t pid = spawn_worker(argv0_.c_str(), extra_argv_, &parent_fd);
        if (pid <= 0) {
            last_error_ = "spawn_worker failed";
            return false;
        }
        pid_ = pid;
        fd_  = parent_fd;
    }

    // Send LOAD_REQ (empty JSON — child rebuilds args from extra_argv via argv).
    auto err = send_frame(fd_, WorkerFrame::LOAD_REQ, 0, std::string("{}"));
    if (err != IpcError::OK) {
        last_error_ = std::string("LOAD_REQ send failed: ") + ipc_error_str(err);
        kill_worker_locked();
        return false;
    }
    // Wait for LOAD_RESP.
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    err = recv_frame(fd_, &hdr, &payload);
    if (err != IpcError::OK || hdr.type != (uint32_t)WorkerFrame::LOAD_RESP) {
        last_error_ = std::string("LOAD_RESP recv failed: ") + ipc_error_str(err);
        kill_worker_locked();
        return false;
    }
    try {
        json resp = json::parse(std::string(payload.begin(), payload.end()));
        if (!resp.value("ok", false)) {
            last_error_ = resp.value("error", "worker LOAD failed");
            kill_worker_locked();
            return false;
        }
    } catch (const std::exception& e) {
        last_error_ = std::string("LOAD_RESP parse: ") + e.what();
        kill_worker_locked();
        return false;
    }
    loaded_ = true;
    return true;
}

RenderResult WorkerSession::render(const std::string& gen_json,
                                   const std::vector<uint8_t>& image,
                                   const std::vector<uint8_t>& audio) {
    RenderResult result;
    if (!ensure_loaded()) {
        result.error = last_error_;
        return result;
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    uint32_t req_id = next_req_id_.fetch_add(1);

    auto payload = pack_render_request(gen_json, image, audio);
    auto err     = send_frame(fd_, WorkerFrame::RENDER_REQ, req_id, payload);
    if (err != IpcError::OK) {
        result.error = std::string("RENDER_REQ send failed: ") + ipc_error_str(err);
        kill_worker_locked();
        last_error_ = result.error;
        return result;
    }

    FrameHeader hdr{};
    std::vector<uint8_t> resp_payload;
    err = recv_frame(fd_, &hdr, &resp_payload);
    if (err != IpcError::OK || hdr.type != (uint32_t)WorkerFrame::RENDER_RESP) {
        result.error = std::string("RENDER_RESP recv failed: ") + ipc_error_str(err);
        kill_worker_locked();
        last_error_ = result.error;
        return result;
    }

    std::string meta;
    if (!unpack_render_response(resp_payload, &meta, &result.video_bytes)) {
        result.error = "RENDER_RESP unpack failed";
        return result;
    }
    try {
        json m = json::parse(meta);
        result.ok = m.value("ok", false);
        result.error = m.value("error", "");
        result.segments_rendered = m.value("segments", 0);
        result.frame_count       = m.value("frame_count", 0);
        result.fps               = m.value("fps", 25);
        result.render_sec        = m.value("render_sec", 0.0);
    } catch (const std::exception& e) {
        result.ok = false;
        result.error = std::string("RENDER_RESP meta parse: ") + e.what();
    }
    return result;
}

// ─── child-side dispatch loop ────────────────────────────────────────────────

// Forward decl from main.cpp — the child re-parses CLI args the same way the
// parent does, but using its own argv (passed through via spawn_worker's
// execv shape). We don't want to duplicate the option lists here, so we
// expose this helper from main.cpp.
extern bool parse_server_args_into(int argc, const char** argv,
                                   SDSvrParams* svr_params,
                                   SDContextParams* ctx_params,
                                   SDGenerationParams* default_gen_params);

static void write_blob_to_file(const std::vector<uint8_t>& blob, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(blob.data()),
              static_cast<std::streamsize>(blob.size()));
}

int run_worker_loop(int fd, int argc, const char** argv) {
    // Drop the "--worker <fd>" tokens — every other arg is identical to
    // what the parent saw, which means parse_server_args_into reads the
    // same model/quant/host/port flags. Host/port are unused inside the
    // worker (only the parent binds) but reusing the parser keeps the
    // argv shape consistent.
    std::vector<const char*> filt;
    filt.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--worker" || a == "--worker-aligner") {
            // skip flag + fd
            if (i + 1 < argc) i++;
            continue;
        }
        filt.push_back(argv[i]);
    }
    SDSvrParams svr_params;
    SDContextParams ctx_params;
    SDGenerationParams default_gen_params;
    if (!parse_server_args_into(static_cast<int>(filt.size()),
                                filt.data(),
                                &svr_params,
                                &ctx_params,
                                &default_gen_params)) {
        std::fprintf(stderr, "worker: failed to parse args\n");
        return 2;
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(false, false, false);
    SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));
    if (!sd_ctx) {
        std::fprintf(stderr, "worker: new_sd_ctx failed\n");
        // Still service LOAD_REQ so the parent gets a clean error before
        // we exit. But there's no graceful way to recover.
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        recv_frame(fd, &hdr, &payload);
        send_frame(fd, WorkerFrame::LOAD_RESP, hdr.req_id,
                   std::string(R"({"ok":false,"error":"new_sd_ctx failed"})"));
        return 3;
    }
    std::mutex sd_ctx_mutex;

    // DiT residency is decided PER RENDER from the request's `offload` flag
    // (see RENDER_REQ below): offload=false keeps the DiT resident across
    // renders (warm-weights fast path, ~10.5 GB); offload=true lets
    // --offload-to-cpu stream weights per layer so peak drops to ~5.7 GB,
    // leaving VRAM for the light GPU peers. Don't pin it unconditionally here.

    // Tmp dir for materialising image+audio per render. /tmp survives
    // across renders; we just overwrite per request.
    fs::path tmp_dir = fs::temp_directory_path() / "longcat-avatar-worker";
    std::error_code ec;
    fs::create_directories(tmp_dir, ec);

    fprintf(stderr, "worker: ready (pid=%d)\n", (int)getpid());

    for (;;) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        auto err = recv_frame(fd, &hdr, &payload);
        if (err == IpcError::EofClean) {
            fprintf(stderr, "worker: parent closed cleanly, exiting\n");
            return 0;
        }
        if (err != IpcError::OK) {
            fprintf(stderr, "worker: recv_frame failed: %s\n", ipc_error_str(err));
            return 1;
        }
        switch (static_cast<WorkerFrame>(hdr.type)) {
            case WorkerFrame::LOAD_REQ: {
                // sd_ctx is already loaded; ack immediately.
                send_frame(fd, WorkerFrame::LOAD_RESP, hdr.req_id,
                           std::string(R"({"ok":true})"));
                break;
            }
            case WorkerFrame::PING: {
                send_frame(fd, WorkerFrame::PONG, hdr.req_id,
                           std::string("{}"));
                break;
            }
            case WorkerFrame::SHUTDOWN: {
                fprintf(stderr, "worker: SHUTDOWN received\n");
                return 0;
            }
            case WorkerFrame::RENDER_REQ: {
                std::string gen_json;
                std::vector<uint8_t> image_bytes, audio_bytes;
                if (!unpack_render_request(payload, &gen_json, &image_bytes, &audio_bytes)) {
                    json err_meta = {{"ok", false}, {"error", "RENDER_REQ unpack failed"}};
                    auto resp = pack_render_response(err_meta.dump(), {});
                    send_frame(fd, WorkerFrame::RENDER_RESP, hdr.req_id, resp);
                    break;
                }
                // Materialise image + audio so the avatar lib can read by path.
                std::string image_path = (tmp_dir / "ref.bin").string();
                std::string audio_path = (tmp_dir / "speech.wav").string();
                write_blob_to_file(image_bytes, image_path);
                write_blob_to_file(audio_bytes, audio_path);

                // Build a SDGenerationParams from defaults + JSON.
                SDGenerationParams gen_params = default_gen_params;
                if (!gen_params.from_json_str(gen_json)) {
                    json err_meta = {{"ok", false}, {"error", "from_json_str failed"}};
                    auto resp = pack_render_response(err_meta.dump(), {});
                    send_frame(fd, WorkerFrame::RENDER_RESP, hdr.req_id, resp);
                    break;
                }
                gen_params.init_image_path = image_path;
                gen_params.audio_path      = audio_path;
                if (!gen_params.resolve_and_validate(VID_GEN,
                                                     ctx_params.lora_model_dir,
                                                     ctx_params.hires_upscalers_dir,
                                                     true)) {
                    json err_meta = {{"ok", false}, {"error", "resolve_and_validate failed"}};
                    auto resp = pack_render_response(err_meta.dump(), {});
                    send_frame(fd, WorkerFrame::RENDER_RESP, hdr.req_id, resp);
                    break;
                }
                // Output format default is webm (the koblem consumer).
                json req_extra;
                try { req_extra = json::parse(gen_json); } catch (...) {}
                std::string output_format = req_extra.value("output_format", std::string("webm"));
                int output_quality        = req_extra.value("output_compression", 90);

                // Per-request DiT residency from `offload`. Residency is the
                // offload-vs-resident switch in this build: resident=true loads
                // the DiT onto the GPU (params_backend == runtime_backend →
                // graph-cut segmented compute disabled → ~10.8 GB monolithic
                // path); resident=false leaves the DiT on CPU so --offload-to-cpu
                // streams it via graph-cut and the peak drops to ~5.7 GB at 53f
                // — leaving room for the light GPU peers (TTS/STT/vision).
                //
                // Applies to chains too: each segment is its own 53f render at
                // the same per-segment VRAM (the 13-frame overlap doesn't raise
                // it), and the lap-20 ggml-alloc fix made segmented/offload
                // coherent at all frame counts. With --clip-on-cpu the
                // per-segment text re-encode just runs on CPU.
                //
                // offload=true (prod default) → resident=false → ~5.7 GB.
                // offload=false → resident=true → ~10.8 GB, warm-weights fast.
                // "auto"/any non-"false" string means on.
                bool offload = true;
                if (req_extra.contains("offload")) {
                    const auto& o = req_extra["offload"];
                    if (o.is_boolean()) {
                        offload = o.get<bool>();
                    } else if (o.is_string()) {
                        offload = o.get<std::string>() != "false";
                    }
                }
                sd_ctx_keep_diffusion_model_resident(sd_ctx.get(), !offload);

                std::vector<uint8_t> video_bytes;
                int segments_rendered = 0;
                std::string err_msg;
                auto t0 = std::chrono::steady_clock::now();
                bool ok = false;
                {
                    std::lock_guard<std::mutex> lock(sd_ctx_mutex);
                    ok = render_avatar_to_video_bytes(sd_ctx.get(),
                                                      gen_params,
                                                      output_format,
                                                      output_quality,
                                                      video_bytes,
                                                      segments_rendered,
                                                      err_msg);
                }
                auto t1 = std::chrono::steady_clock::now();
                double render_sec = std::chrono::duration<double>(t1 - t0).count();

                json meta = {
                    {"ok", ok},
                    {"error", err_msg},
                    {"segments", segments_rendered},
                    {"frame_count", segments_rendered > 0 ? (int)video_bytes.size() : 0},
                    {"render_sec", render_sec},
                    {"fps", gen_params.fps},
                };
                auto resp = pack_render_response(meta.dump(), video_bytes);
                auto serr = send_frame(fd, WorkerFrame::RENDER_RESP, hdr.req_id, resp);
                if (serr != IpcError::OK) {
                    fprintf(stderr, "worker: RENDER_RESP send failed: %s\n",
                            ipc_error_str(serr));
                    return 1;
                }
                break;
            }
            default: {
                fprintf(stderr, "worker: unexpected frame type 0x%x\n", hdr.type);
                break;
            }
        }
    }
}

} // namespace longcat_avatar
