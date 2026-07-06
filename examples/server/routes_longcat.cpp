// routes_longcat.cpp — POST /generate (avatar render) + /v1/admin/* endpoints
// for the native longcat-avatar-server. Mirrors the wire shape of the v0
// supervisor (tools/avatar_server.py) so koblem's HTTP client + the
// kob-gpu-gate eviction path keep working unchanged.

#include "routes_longcat.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "common/log.h"
#include "worker_session.h"

namespace fs = std::filesystem;

// ─── base64 decode (the supervisor's request encodes image+audio as b64) ─────

static int b64_index(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}
static std::vector<uint8_t> base64_decode(const std::string& s) {
    std::vector<uint8_t> out;
    out.reserve(s.size() * 3 / 4);
    int val = 0, bits = -8;
    for (char c : s) {
        if (c == '=' || c == '\r' || c == '\n' || c == ' ') continue;
        int idx = b64_index(c);
        if (idx < 0) continue;
        val = (val << 6) | idx;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    in.seekg(0, std::ios::end);
    auto n = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(n > 0 ? (size_t)n : 0);
    if (n > 0) in.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

// "data:image/png;base64,XXXX" / raw b64 / fs path / http url. The native
// server runs server-side so http downloads are fine. Matches the wire
// shape of avatar_server.py::_materialize.
static std::vector<uint8_t> materialize_blob(const std::string& spec) {
    if (spec.empty()) return {};
    if (spec.size() < 4096) {
        if (fs::exists(spec) && fs::is_regular_file(spec)) {
            return read_file(spec);
        }
    }
    if (spec.rfind("data:", 0) == 0) {
        auto comma = spec.find(',');
        if (comma != std::string::npos) {
            return base64_decode(spec.substr(comma + 1));
        }
        return {};
    }
    // For now treat anything else as raw base64. The Python supervisor also
    // handles http(s) URLs; native users can pre-fetch or send b64.
    return base64_decode(spec);
}

// ─── BSA env block — propagated via setenv() onto the worker child ────────────
//
// The DiT reads LONGCAT_BSA* env vars at sd_ctx init time. We're a long-lived
// worker so we can't re-init per render; the env-block fields therefore only
// take effect if the worker was SIGKILL'd between requests with the env
// preset, which is exactly what /v1/admin/unload does. The supervisor handled
// this by spawn-per-render; the native server's tradeoff is documented in
// HANDOFF-native-server.md (BSA toggle requires worker recycle).
//
// We still parse + return the requested env so callers can verify the
// payload, and we setenv() them in the parent so the next worker spawn picks
// them up. This is the same shape as the supervisor's extra_env handling but
// applied at child-spawn time rather than per-render.
static void apply_bsa_env_from_json(const json& bsa_obj,
                                    nlohmann::json& applied_meta) {
    if (!bsa_obj.is_object()) return;
    if (bsa_obj.value("enable", false)) {
        setenv("LONGCAT_BSA", "1", 1);
        applied_meta["LONGCAT_BSA"] = "1";
    } else {
        return;
    }
    auto set_int = [&](const char* k, const char* env_k) {
        if (bsa_obj.contains(k) && bsa_obj[k].is_number()) {
            std::string v = std::to_string((int)bsa_obj[k].get<double>());
            setenv(env_k, v.c_str(), 1);
            applied_meta[env_k] = v;
        }
    };
    if (bsa_obj.value("self_frame", false)) {
        setenv("LONGCAT_BSA_SELF_FRAME", "1", 1);
        applied_meta["LONGCAT_BSA_SELF_FRAME"] = "1";
    }
    if (bsa_obj.value("bitmap", false)) {
        setenv("LONGCAT_BSA_BITMAP", "1", 1);
        applied_meta["LONGCAT_BSA_BITMAP"] = "1";
    }
    set_int("radius",  "LONGCAT_BSA_RADIUS");
    set_int("bookend", "LONGCAT_BSA_BOOKEND");
    set_int("cube_h",  "LONGCAT_BSA_CUBE_H");
    set_int("cube_w",  "LONGCAT_BSA_CUBE_W");
}

// Namespace guard: only env keys that look like LongCat / GGML / CUDA knobs.
static void apply_debug_env_from_json(const json& env_obj,
                                      nlohmann::json& applied_meta) {
    if (!env_obj.is_object()) return;
    for (auto it = env_obj.begin(); it != env_obj.end(); ++it) {
        const std::string& k = it.key();
        if (k.rfind("LONGCAT_", 0) != 0 &&
            k.rfind("GGML_", 0)    != 0 &&
            k.rfind("CUDA_", 0)    != 0) {
            continue;
        }
        std::string v;
        if (it.value().is_string()) v = it.value().get<std::string>();
        else                         v = it.value().dump();
        setenv(k.c_str(), v.c_str(), 1);
        applied_meta[k] = v;
    }
}

// ─── /generate handler ────────────────────────────────────────────────────────

static void handle_generate(const httplib::Request& req,
                            httplib::Response& res,
                            LongCatRuntime& ctx) {
    if (ctx.state.draining.load()) {
        res.status = 503;
        res.set_content(R"({"error":"service draining — not accepting new renders"})",
                        "application/json");
        return;
    }
    int prev_inflight = ctx.state.in_flight.fetch_add(1);
    if (prev_inflight > 0) {
        ctx.state.in_flight.fetch_sub(1);
        res.status = 429;
        res.set_content(R"json({"error":"busy: a render is already in flight (one GPU job at a time)"})json",
                        "application/json");
        return;
    }

    struct InFlightGuard {
        LongCatRuntime& ctx;
        ~InFlightGuard() { ctx.state.in_flight.fetch_sub(1); }
    } guard{ctx};

    std::string image_spec, audio_spec;
    std::vector<uint8_t> image_bytes, audio_bytes;
    json body;
    std::string return_mode = "file";

    // multipart: `request` JSON part + `image` file part + `audio` file part.
    // Single-shot path is JSON-with-base64 (current koblem wire shape).
    if (req.is_multipart_form_data()) {
        // `request` may arrive either as a text field or a file part.
        std::string req_json;
        if (req.form.has_field("request")) {
            req_json = req.form.get_field("request");
        } else if (req.form.has_file("request")) {
            req_json = req.form.get_file("request").content;
        }
        if (!req_json.empty()) {
            try {
                body = json::parse(req_json);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json({{"error", std::string("bad multipart 'request' json: ") + e.what()}}).dump(),
                                "application/json");
                return;
            }
        } else {
            body = json::object();
        }
        if (req.form.has_file("image")) {
            const auto& f = req.form.get_file("image");
            image_bytes.assign(f.content.begin(), f.content.end());
        }
        if (req.form.has_file("audio")) {
            const auto& f = req.form.get_file("audio");
            audio_bytes.assign(f.content.begin(), f.content.end());
        }
    } else {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"empty body"})", "application/json");
            return;
        }
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", std::string("bad json: ") + e.what()}}).dump(),
                            "application/json");
            return;
        }
        image_spec = body.value("image", std::string());
        audio_spec = body.value("audio", std::string());
        image_bytes = materialize_blob(image_spec);
        audio_bytes = materialize_blob(audio_spec);
    }

    if (image_bytes.empty() || audio_bytes.empty()) {
        res.status = 400;
        res.set_content(R"({"error":"image and audio are required"})",
                        "application/json");
        return;
    }
    if (body.contains("return")) {
        return_mode = body["return"].get<std::string>();
    }
    if (return_mode != "bytes" && return_mode != "file") {
        // Default to file (matches supervisor); koblem always asks for "bytes".
        return_mode = "file";
    }
    // Accept: video/webm forces bytes too.
    auto accept_hdr = req.get_header_value("Accept");
    if (accept_hdr.find("video/webm") != std::string::npos) {
        return_mode = "bytes";
    }

    // BSA + debug_env apply BEFORE we (re)spawn the worker, so the child
    // picks them up at sd_ctx init time. If the worker is already alive
    // they're recorded for the next recycle.
    json applied_env = json::object();
    if (body.contains("bsa")) {
        apply_bsa_env_from_json(body["bsa"], applied_env);
    }
    if (body.contains("debug_env")) {
        apply_debug_env_from_json(body["debug_env"], applied_env);
    }

    // Audio knob aliases the supervisor accepts.
    if (body.contains("audio_lowpass_hz") && !body.contains("audio_lowpass")) {
        body["audio_lowpass"] = body["audio_lowpass_hz"];
    }

    // Honour `duration_sec` → segments solver (same formula as the supervisor)
    if (body.contains("duration_sec") && body["duration_sec"].is_number()) {
        double duration = body["duration_sec"].get<double>();
        int seg_frames = body.value("segment_frames", body.value("video_frames", 53));
        int cond       = body.value("cont_cond_frames", 13);
        int new_per    = std::max(1, seg_frames - cond);
        int target     = std::max(1, (int)std::round(duration * 25.0));
        int n          = 1;
        if (target > seg_frames) n = 1 + (target - seg_frames + new_per - 1) / new_per;
        body["segments"] = std::max(1, n);
    }

    // Now drive the worker. ensure_loaded() spawns the child on first call;
    // subsequent calls reuse the warm worker (weights stay resident).
    if (!ctx.worker->ensure_loaded()) {
        res.status = 500;
        res.set_content(json({{"error", std::string("worker spawn failed: ") + ctx.worker->last_error()}}).dump(),
                        "application/json");
        return;
    }

    // Disconnect watchdog. render() blocks on the worker round-trip; if the client
    // goes away mid-render we poll req.is_connection_closed (httplib wires this per
    // request against the socket) every ~200 ms and ask the worker to cancel
    // cooperatively. The worker bails ASAP and stays warm — this is NOT /unload.
    std::atomic<bool> wd_stop{false};
    std::atomic<bool> client_gone{false};
    std::thread watchdog([&]() {
        while (!wd_stop.load(std::memory_order_relaxed)) {
            if (req.is_connection_closed && req.is_connection_closed()) {
                client_gone.store(true, std::memory_order_relaxed);
                LOG_INFO("generate: client disconnected -> cancel in-flight render");
                ctx.worker->cancel_in_flight();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });
    struct WatchdogJoinGuard {
        std::atomic<bool>& stop;
        std::thread& t;
        ~WatchdogJoinGuard() { stop.store(true, std::memory_order_relaxed); if (t.joinable()) t.join(); }
    } watchdog_join{wd_stop, watchdog};

    auto t0 = std::chrono::steady_clock::now();
    longcat_avatar::RenderResult rr = ctx.worker->render(body.dump(), image_bytes, audio_bytes);
    auto t1 = std::chrono::steady_clock::now();
    double wall_sec = std::chrono::duration<double>(t1 - t0).count();

    if (!rr.ok) {
        // Cancelled by client disconnect: nothing to send (client is gone), but
        // return a distinct status for logs/proxies. 499 = client closed request.
        if (client_gone.load(std::memory_order_relaxed)) {
            res.status = 499;
            res.set_content(R"json({"error":"render cancelled (client disconnected)"})json",
                            "application/json");
            return;
        }
        res.status = 500;
        res.set_content(json({{"error", rr.error.empty() ? "render failed" : rr.error}}).dump(),
                        "application/json");
        return;
    }
    ctx.state.renders_done.fetch_add(1);
    ctx.state.last_render_sec.store(rr.render_sec > 0 ? rr.render_sec : wall_sec);

    // Build response meta — the supervisor's X-Avatar-* header set + JSON
    // envelope. koblem reads X-Avatar-segments / X-Avatar-render_sec /
    // X-Avatar-offload (see koblem/api/src/avatar.rs::generate).
    auto add_meta_header = [&](const std::string& k, const std::string& v) {
        res.set_header("X-Avatar-" + k, v);
    };
    int  segments_used = std::max(1, rr.segments_rendered);
    bool offload_used  = body.value("offload", true);
    if (body.contains("offload") && body["offload"].is_string()) {
        // auto -> the lib decides; we don't try to second-guess.
        offload_used = body["offload"].get<std::string>() == "true";
    }
    add_meta_header("segments",   std::to_string(segments_used));
    add_meta_header("render_sec", std::to_string(rr.render_sec));
    add_meta_header("offload",    offload_used ? "true" : "false");

    if (return_mode == "bytes") {
        res.set_content(reinterpret_cast<const char*>(rr.video_bytes.data()),
                        rr.video_bytes.size(),
                        "video/webm");
        return;
    }

    // file mode: write the video bytes to a known tmp path and return the
    // metadata envelope (the supervisor's shape).
    fs::path outdir = ctx.outdir;
    std::error_code ec;
    fs::create_directories(outdir, ec);
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    fs::path out_path = outdir / ("avatar-" + std::to_string(now_us) + ".webm");
    {
        std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(rr.video_bytes.data()),
                  static_cast<std::streamsize>(rr.video_bytes.size()));
    }
    json envelope = {
        {"status",      "ok"},
        {"video_path",  out_path.string()},
        {"video_bytes", (int64_t)rr.video_bytes.size()},
        {"render_sec",  rr.render_sec},
        {"segments",    segments_used},
        {"fps",         rr.fps},
    };
    if (!applied_env.empty()) envelope["applied_env"] = applied_env;
    res.set_content(envelope.dump(), "application/json");
}

// ─── /health, /v1/admin/{unload,drain,load} ──────────────────────────────────

static void handle_health(const httplib::Request&, httplib::Response& res,
                          LongCatRuntime& ctx) {
    bool loaded = ctx.worker && ctx.worker->is_alive();
    json h = {
        {"status",            "ok"},
        {"busy",              ctx.state.in_flight.load() > 0},
        {"draining",          ctx.state.draining.load()},
        {"loaded",            loaded},
        {"in_flight",         ctx.state.in_flight.load()},
        {"renders_done",      ctx.state.renders_done.load()},
        {"last_render_sec",   ctx.state.last_render_sec.load()},
        {"uptime_sec",        std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - ctx.state.started_at).count()},
    };
    if (loaded) h["worker_pid"] = (int)ctx.worker->pid();
    res.set_content(h.dump(), "application/json");
}

// GPU residency announce for the koblem gate.
static void handle_gpu_status(const httplib::Request&, httplib::Response& res,
                              LongCatRuntime& ctx) {
    const bool loaded = ctx.worker && ctx.worker->is_alive();
    json body = {{"loaded", loaded}};
    if (loaded && !ctx.worker->worker_gpu().empty())
        body["gpu"] = ctx.worker->worker_gpu();
    else
        body["gpu"] = nullptr;
    res.set_content(body.dump(), "application/json");
}

static void handle_unload(const httplib::Request& req, httplib::Response& res,
                          LongCatRuntime& ctx) {
    bool force = false;
    if (!req.body.empty()) {
        try {
            json b = json::parse(req.body);
            force = b.value("force", false);
        } catch (...) {}
    }
    bool was_loaded = ctx.worker && ctx.worker->is_alive();
    if (ctx.state.in_flight.load() > 0 && !force) {
        res.status = 200;
        res.set_content(json({{"status", "busy (pass force=true to kill)"}}).dump(),
                        "application/json");
        return;
    }
    if (ctx.worker) {
        ctx.worker->shutdown();
    }
    // Clear the drain flag now the worker is dead + VRAM reclaimed: drain's job was to
    // hold new renders until in-flight finished ahead of the unload. Leaving it set
    // wedges the service — the generate guard (:153) hard-503s while draining, blocking
    // the request that would re-spawn the worker. The gate does drain->unload but never
    // /load, so nothing else clears it. Mirrors handle_load().
    ctx.state.draining.store(false);
    res.set_content(json({
        {"status", was_loaded ? "unloaded" : "idle"},
        {"unloaded", was_loaded},
    }).dump(), "application/json");
}

static void handle_drain(const httplib::Request&, httplib::Response& res,
                         LongCatRuntime& ctx) {
    ctx.state.draining.store(true);
    res.set_content(json({
        {"status",   "draining"},
        {"busy",     ctx.state.in_flight.load() > 0},
        {"in_flight",ctx.state.in_flight.load()},
    }).dump(), "application/json");
}

static void handle_load(const httplib::Request&, httplib::Response& res,
                        LongCatRuntime& ctx) {
    ctx.state.draining.store(false);
    // Optional pre-warm: spawn the worker now so the first /generate doesn't
    // pay cold-start latency. Best-effort: failure is reported but doesn't
    // 5xx (matches the supervisor's lenient response).
    bool spawned = false;
    if (ctx.worker && !ctx.worker->is_alive()) {
        spawned = ctx.worker->ensure_loaded();
    }
    res.set_content(json({
        {"status",  "ok"},
        {"loaded",  ctx.worker && ctx.worker->is_alive()},
        {"spawned", spawned},
    }).dump(), "application/json");
}

void register_longcat_endpoints(httplib::Server& svr, LongCatRuntime& ctx) {
    LongCatRuntime* ctxp = &ctx;

    svr.Get("/health", [ctxp](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res, *ctxp);
    });
    svr.Get("/v1/gpu/status", [ctxp](const httplib::Request& req, httplib::Response& res) {
        handle_gpu_status(req, res, *ctxp);
    });
    svr.Post("/generate", [ctxp](const httplib::Request& req, httplib::Response& res) {
        handle_generate(req, res, *ctxp);
    });
    svr.Post("/v1/admin/unload", [ctxp](const httplib::Request& req, httplib::Response& res) {
        handle_unload(req, res, *ctxp);
    });
    svr.Post("/unload", [ctxp](const httplib::Request& req, httplib::Response& res) {
        handle_unload(req, res, *ctxp);
    });
    svr.Post("/v1/admin/drain", [ctxp](const httplib::Request& req, httplib::Response& res) {
        handle_drain(req, res, *ctxp);
    });
    svr.Post("/v1/admin/load", [ctxp](const httplib::Request& req, httplib::Response& res) {
        handle_load(req, res, *ctxp);
    });

    // Body-less POSTs deadlock on httplib's read_content_without_length
    // until the keep-alive timeout fires; the supervisor inherited the same
    // workaround from qwen3-tts.cpp. Inject Content-Length: 0 so the body
    // read returns immediately.
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response&) {
        if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
            !req.has_header("Content-Length") &&
            req.get_header_value("Transfer-Encoding") != "chunked") {
            const_cast<httplib::Request&>(req).set_header("Content-Length", "0");
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
}
