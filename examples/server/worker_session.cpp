// worker_session.cpp — see worker_session.h.

#include "worker_session.h"

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
#include "routes.h"
#include "async_jobs.h"
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

void WorkerSession::cancel_in_flight() {
    uint32_t req_id = current_render_req_id_.load(std::memory_order_acquire);
    if (req_id == 0) {
        return;  // no render in flight
    }
    int fd = fd_;
    if (fd < 0) {
        return;  // worker already gone
    }
    // Deliberately NOT io_mutex_ — the render we're cancelling holds it inside its
    // blocking recv_frame.
    std::lock_guard<std::mutex> slk(send_mutex_);
    IpcError e = send_frame(fd, WorkerFrame::CANCEL_REQ, req_id, nullptr, 0);
    if (e != IpcError::OK) {
        fprintf(stderr, "worker-session: CANCEL_REQ send failed (req_id=%u): %s\n",
                req_id, ipc_error_str(e));
        return;
    }
    fprintf(stderr, "worker-session: CANCEL_REQ sent (req_id=%u)\n", req_id);
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
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        auto err = send_frame(fd_, WorkerFrame::RENDER_REQ, req_id, payload);
        if (err != IpcError::OK) {
            result.error = std::string("RENDER_REQ send failed: ") + ipc_error_str(err);
            kill_worker_locked();
            last_error_ = result.error;
            return result;
        }
    }
    // Publish req_id AFTER the send so a cancel racing in before the worker has even
    // seen RENDER_REQ is a harmless no-op. Clear on every return path via RAII.
    current_render_req_id_.store(req_id, std::memory_order_release);
    struct CurrentRenderGuard {
        std::atomic<uint32_t>& cur;
        ~CurrentRenderGuard() { cur.store(0, std::memory_order_release); }
    } current_render_guard{current_render_req_id_};

    FrameHeader hdr{};
    std::vector<uint8_t> resp_payload;
    auto err = recv_frame(fd_, &hdr, &resp_payload);
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

ImageRenderResult WorkerSession::render_image(const std::string& request_json) {
    ImageRenderResult result;
    if (!ensure_loaded()) {
        result.error = last_error_;
        return result;
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    uint32_t req_id = next_req_id_.fetch_add(1);

    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        auto err = send_frame(fd_, WorkerFrame::IMG_GEN_REQ, req_id, request_json);
        if (err != IpcError::OK) {
            result.error = std::string("IMG_GEN_REQ send failed: ") + ipc_error_str(err);
            kill_worker_locked();
            last_error_ = result.error;
            return result;
        }
    }
    // Publish AFTER the send so a cancel racing in early is a harmless no-op.
    current_render_req_id_.store(req_id, std::memory_order_release);
    struct CurrentRenderGuard {
        std::atomic<uint32_t>& cur;
        ~CurrentRenderGuard() { cur.store(0, std::memory_order_release); }
    } current_render_guard{current_render_req_id_};

    FrameHeader hdr{};
    std::vector<uint8_t> resp_payload;
    auto err = recv_frame(fd_, &hdr, &resp_payload);
    if (err != IpcError::OK || hdr.type != (uint32_t)WorkerFrame::IMG_GEN_RESP) {
        result.error = std::string("IMG_GEN_RESP recv failed: ") + ipc_error_str(err);
        kill_worker_locked();
        last_error_ = result.error;
        return result;
    }
    try {
        json m = json::parse(std::string(resp_payload.begin(), resp_payload.end()));
        result.ok            = m.value("ok", false);
        result.error         = m.value("error", "");
        result.output_format = m.value("output_format", std::string("png"));
        result.render_sec    = m.value("render_sec", 0.0);
        if (m.contains("images") && m["images"].is_array()) {
            for (const auto& im : m["images"]) {
                if (im.is_string()) result.images.push_back(im.get<std::string>());
            }
        }
    } catch (const std::exception& e) {
        result.ok = false;
        result.error = std::string("IMG_GEN_RESP meta parse: ") + e.what();
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

    // free_params_immediately=TRUE: frees the umT5 text encoder (~5.7 GB host RAM)
    // right after the one-shot text encode. On the avatar vid_gen path this is the
    // ONLY weight it frees — the DiT (stable-diffusion.cpp:5872), VAE (5251) and
    // whisper (5604) frees are all gated on !keep_diffusion_model_resident, and the
    // worker sets keep_resident=true below, so those stay warm across renders/segments.
    // Matches the CLI (main.cpp passes true). umT5 is dead after its encode (chained
    // segments reuse the cached text conditioning), so holding it was pure waste.
    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(false, true, false);
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

    // Resident DiT across renders is what gives us the warm-weights win.
    sd_ctx_keep_diffusion_model_resident(sd_ctx.get(), true);

    // Tmp dir for materialising image+audio per render. /tmp survives
    // across renders; we just overwrite per request.
    fs::path tmp_dir = fs::temp_directory_path() / "longcat-avatar-worker";
    std::error_code ec;
    fs::create_directories(tmp_dir, ec);

    // FLUX.2 image-isolation mode reuses the in-process compute (parse_img_gen_request
    // + ensure_variant_loaded + execute_img_gen_job) via a child-local ServerRuntime.
    // Built unconditionally — it's cheap, and the avatar path simply never sends
    // IMG_GEN_REQ. The DiT swap state lives HERE (the real sd_ctx is in this child).
    std::vector<LoraEntry>     lora_cache;
    std::mutex                 lora_mutex;
    std::vector<UpscalerEntry> upscaler_cache;
    std::mutex                 upscaler_mutex;
    ModelSwapState             model_swap;
    model_swap.base_path      = ctx_params.diffusion_model_path;
    model_swap.edit_path      = svr_params.diffusion_model_edit_path;
    model_swap.loaded_variant = "base";
    model_swap.loaded.store(true);
    model_swap.draining.store(false);
    ServerRuntime child_runtime = {
        sd_ctx.get(),       &sd_ctx_mutex,    &svr_params,
        &ctx_params,        &default_gen_params,
        &lora_cache,        &lora_mutex,      &upscaler_cache,
        &upscaler_mutex,    nullptr /*async_job_manager*/, &model_swap,
    };

    fprintf(stderr, "worker: ready (pid=%d)\n", (int)getpid());

    // Reader thread demultiplexes the socket so CANCEL_REQ is observed WHILE a render
    // runs on the main thread. CANCEL_REQ is handled inline by the reader (flips the
    // cooperative cancel flag); every other frame is queued for the main thread.
    struct WorkerCtrl {
        std::deque<std::pair<FrameHeader, std::vector<uint8_t>>> work_queue;
        std::mutex              queue_mutex;
        std::condition_variable queue_cv;
        std::atomic<bool>       reader_done{false};
        int                     reader_exit_code = 0;
        // Published by the main thread BEFORE a render begins, cleared AFTER. The
        // reader matches CANCEL_REQ.req_id against this; mismatch/0 ⇒ drop.
        std::atomic<uint32_t>   active_render_req_id{0};
    };
    WorkerCtrl ctrl;

    std::thread reader_thread([fd, &ctrl]() {
        for (;;) {
            FrameHeader hdr{};
            std::vector<uint8_t> payload;
            IpcError e = recv_frame(fd, &hdr, &payload);
            if (e == IpcError::EofClean) {
                ctrl.reader_done.store(true);
                ctrl.queue_cv.notify_all();
                return;
            }
            if (e != IpcError::OK) {
                fprintf(stderr, "worker-reader: recv_frame failed: %s\n", ipc_error_str(e));
                ctrl.reader_exit_code = 1;
                ctrl.reader_done.store(true);
                ctrl.queue_cv.notify_all();
                return;
            }
            WorkerFrame ft = static_cast<WorkerFrame>(hdr.type);
            if (ft == WorkerFrame::CANCEL_REQ) {
                uint32_t active = ctrl.active_render_req_id.load(std::memory_order_acquire);
                if (active != 0 && active == hdr.req_id) {
                    sd_request_cancel();
                    fprintf(stderr, "worker-reader: CANCEL_REQ req_id=%u accepted\n", hdr.req_id);
                } else {
                    fprintf(stderr, "worker-reader: CANCEL_REQ req_id=%u dropped (active=%u)\n",
                            hdr.req_id, active);
                }
                continue;
            }
            if (ft == WorkerFrame::SHUTDOWN) {
                std::lock_guard<std::mutex> lk(ctrl.queue_mutex);
                ctrl.work_queue.emplace_back(hdr, std::move(payload));
                ctrl.reader_done.store(true);
                ctrl.queue_cv.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lk(ctrl.queue_mutex);
                ctrl.work_queue.emplace_back(hdr, std::move(payload));
            }
            ctrl.queue_cv.notify_all();
        }
    });
    struct ReaderJoinGuard {
        std::thread& t;
        ~ReaderJoinGuard() { if (t.joinable()) t.join(); }
    } reader_join{reader_thread};

    for (;;) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        {
            std::unique_lock<std::mutex> lk(ctrl.queue_mutex);
            ctrl.queue_cv.wait(lk, [&] {
                return !ctrl.work_queue.empty() || ctrl.reader_done.load();
            });
            if (ctrl.work_queue.empty()) {
                return ctrl.reader_exit_code;  // reader done, nothing left
            }
            hdr     = ctrl.work_queue.front().first;
            payload = std::move(ctrl.work_queue.front().second);
            ctrl.work_queue.pop_front();
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

                std::vector<uint8_t> video_bytes;
                int segments_rendered = 0;
                std::string err_msg;
                auto t0 = std::chrono::steady_clock::now();
                bool ok = false;
                // Cancel coordination: clear any stale cancel from a prior render BEFORE
                // publishing this render's req_id, so a CANCEL_REQ arriving after the
                // store is guaranteed to see the cleared flag (release/acquire with the
                // reader's load). RAII unpublish on every exit.
                sd_clear_cancel();
                ctrl.active_render_req_id.store(hdr.req_id, std::memory_order_release);
                struct RenderActiveGuard {
                    std::atomic<uint32_t>& active;
                    ~RenderActiveGuard() { active.store(0, std::memory_order_release); }
                } render_active_guard{ctrl.active_render_req_id};
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
            case WorkerFrame::IMG_GEN_REQ: {
                // FLUX.2 image render. Payload is the raw /sdcpp/v1/img_gen body;
                // re-parse + render with the same code the in-process server runs.
                std::string request_json(payload.begin(), payload.end());
                bool ok = false;
                std::string err_msg;
                std::string out_fmt = "png";
                std::vector<std::string> images;
                auto t0 = std::chrono::steady_clock::now();
                // Cancel coordination — mirrors RENDER_REQ so /sdcpp job-cancel and
                // force /v1/admin/unload reach this render via the reader thread.
                sd_clear_cancel();
                ctrl.active_render_req_id.store(hdr.req_id, std::memory_order_release);
                struct ImgActiveGuard {
                    std::atomic<uint32_t>& active;
                    ~ImgActiveGuard() { active.store(0, std::memory_order_release); }
                } img_active_guard{ctrl.active_render_req_id};
                try {
                    json body = json::parse(request_json);
                    ImgGenJobRequest request;
                    if (!parse_img_gen_request(body, child_runtime, request, err_msg)) {
                        // err_msg already set by the parser.
                    } else {
                        out_fmt = request.output_format;
                        AsyncGenerationJob ijob;
                        ijob.kind    = AsyncJobKind::ImgGen;
                        ijob.img_gen = std::move(request);
                        if (ensure_variant_loaded(child_runtime, ijob.img_gen.model_variant, err_msg)) {
                            ok = execute_img_gen_job(child_runtime, ijob, images, err_msg);
                        }
                    }
                } catch (const std::exception& e) {
                    err_msg = std::string("img_gen worker exception: ") + e.what();
                }
                double render_sec = std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - t0).count();
                json meta = {
                    {"ok", ok},
                    {"error", err_msg},
                    {"output_format", out_fmt},
                    {"render_sec", render_sec},
                    {"images", images},
                };
                auto serr = send_frame(fd, WorkerFrame::IMG_GEN_RESP, hdr.req_id, meta.dump());
                if (serr != IpcError::OK) {
                    fprintf(stderr, "worker: IMG_GEN_RESP send failed: %s\n",
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
