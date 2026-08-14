// avatar_e2e — one native call: photo + text prompt -> rigged, animated GLB.
//
// This is a THIN driver over avatar_pipeline.hpp. It exists to demonstrate and to run the harness;
// the API is meant to link the library, not to shell to this. Everything interesting is in
// avatar::Engine, and this file deliberately contains no pipeline logic at all.
//
//   ./avatar_e2e --image photo.png --prompt "a person waves" --out out/anim.glb [--stage-dir DIR]
//   ./avatar_e2e --serve-demo ...     two requests through ONE Engine, to show model reuse
//   ./avatar_e2e --motion-only --prompt "..." --out clip.json
//
// The 3060 lock is OFF unless --lock is passed: the library must not grab a global lock behind a
// caller's back. On this box the demo passes it, because the card is shared.
#include "avatar_pipeline.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Ctrl-C -> cancel, so the demo driver exercises the same path a service's /cancel endpoint does.
// (image_to_rig, the A/B harness, deliberately keeps its old kill-me behaviour — see cancel_hook.hpp.)
static avatar::CancelToken* g_sigint_token = nullptr;
static void on_sigint(int) {
    if (g_sigint_token) g_sigint_token->cancel();   // an atomic store: async-signal-safe
}

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void usage() {
    std::printf(
        "usage: avatar_e2e --image IMG --prompt TEXT --out OUT.glb [options]\n"
        "  --image PATH          any photo; RMBG-2.0 mattes it in-process\n"
        "  --prompt TEXT         motion prompt (HY-Motion, Qwen3-8B + CLIP-L conditioning)\n"
        "  --uncond              the checkpoint's trained null conditioning (no text encoder)\n"
        "  --out PATH            the animated, rigged delivery GLB\n"
        "  --stage-dir DIR       keep the geometry/DC intermediates + the resume cache\n"
        "  --from-refined DIR    resume at bake+rig from a previous run's stage dir\n"
        "  --rig-cache DIR       share ONE skeleton across LOD tiers\n"
        "  --clip-json PATH      also write the raw smplh22 motion clip\n"
        "  --seconds F           clip length (default 4)\n"
        "  --steps N             Euler steps (default 50)\n"
        "  --cfg F               text guidance (default 5)\n"
        "  --seed N              motion seed (default 0)\n"
        "  --texsize N           atlas size (default 2048)\n"
        "  --decimate N          target faces (default 220000)\n"
        "  --resolution N        geometry lattice (default: the binary's own)\n"
        "  --variant ID          fix|nofix|base|rot|rotbase (default rotbase)\n"
        "  --smooth N            temporal quaternion smoothing window (default 2)\n"
        "  --no-exercise         skip the skeleton-exercise eye-test track\n"
        "  --lock [PATH]         serialise GPU stages on an flock (default off)\n"
        "  --motion-only         prompt -> clip JSON, no geometry\n"
        "  --rig-only            image -> rigged GLB, no motion\n"
        "  --animate-only RIG    retarget onto an existing rigged GLB\n"
        "  --clip-in PATH        (with --animate-only) use a clip already on disk; repeatable.\n"
        "                        With it the animate path needs no GPU at all.\n"
        "  --idle-unload S       drop resident weights after S idle seconds (0 = never; default 300)\n"
        "  --serve-demo          run TWO motion requests through one Engine, then the full chain,\n"
        "                        to show that weights are loaded once and reused\n"
        "  --cancel-after S      trip the request's CancelToken S seconds in and report the measured\n"
        "                        latency from cancel() to the call actually returning\n"
        "  --queue-demo N        fire N concurrent requests at one Engine to show single-flight\n"
        "                        queueing: admission order, queue_position, and /health as it runs\n"
        "  --hymotion PATH --llm PATH --clip-l PATH --geo-model DIR   weight overrides\n");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    avatar::EngineConfig cfg;
    cfg.hymotion_gguf = "/mnt/hdd/3d/avatar-shootout/_weights/hymotion/gguf/hymotion-1.0-f16.gguf";
    cfg.llm_gguf = "/home/dbrain/dev/flux2.cpp/models/text_encoders/Qwen3-8B-UD-Q4_K_XL.gguf";
    cfg.clip_l_gguf = "/mnt/hdd/3d/avatar-shootout/_weights/hymotion/gguf/hymotion-clip-l-f16.gguf";

    avatar::Request req;
    std::string clip_json_out, animate_only_rig;
    std::vector<std::string> clip_in;   // reuse a clip already on disk instead of generating one
    bool motion_only = false, rig_only = false, serve_demo = false;
    double cancel_after = 0;   // seconds; 0 = never
    int    queue_demo = 0;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--image") req.rig.image = next();
        else if (a == "--prompt") req.motion.prompt = next();
        else if (a == "--uncond") req.motion.uncond = true;
        else if (a == "--out") req.out_glb = next();
        else if (a == "--stage-dir") req.rig.stage_dir = next();
        else if (a == "--from-refined") req.rig.from_refined = next();
        else if (a == "--rig-cache") req.rig.rig_cache = next();
        else if (a == "--clip-json") clip_json_out = next();
        else if (a == "--seconds") req.motion.seconds = (float)std::atof(next().c_str());
        else if (a == "--steps") req.motion.steps = std::atoi(next().c_str());
        else if (a == "--cfg") req.motion.cfg = (float)std::atof(next().c_str());
        else if (a == "--seed") req.motion.seed = (uint32_t)std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--texsize") req.rig.texsize = std::atoi(next().c_str());
        else if (a == "--decimate") req.rig.decimate = std::atoi(next().c_str());
        else if (a == "--resolution") req.rig.resolution = std::atoi(next().c_str());
        else if (a == "--variant") req.variant = next();
        else if (a == "--smooth") req.smooth = std::atoi(next().c_str());
        else if (a == "--no-exercise") req.exercise = false;
        else if (a == "--lock") {
            if (i + 1 < argc && argv[i + 1][0] != '-') cfg.gpu_lock_path = next();
            else cfg.gpu_lock_path = "/mnt/hdd/3d/avatar-shootout/_shootout_out/.3060-image-to-rig.lock";
        }
        else if (a == "--motion-only") motion_only = true;
        else if (a == "--rig-only") rig_only = true;
        else if (a == "--animate-only") animate_only_rig = next();
        else if (a == "--clip-in") clip_in.push_back(next());   // repeatable; skips the motion stage
        else if (a == "--serve-demo") serve_demo = true;
        else if (a == "--cancel-after") cancel_after = std::atof(next().c_str());
        else if (a == "--queue-demo") queue_demo = std::atoi(next().c_str());
        else if (a == "--hymotion") cfg.hymotion_gguf = next();
        else if (a == "--llm") cfg.llm_gguf = next();
        else if (a == "--clip-l") cfg.clip_l_gguf = next();
        else if (a == "--geo-model") cfg.geo_model_dir = next();
        else if (a == "--cpu") cfg.use_cuda = false;
        else if (a == "--idle-unload") cfg.idle_unload_seconds = std::atoi(next().c_str());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }
    req.rig.out_glb = req.out_glb.empty() ? std::string() : req.out_glb + ".rigged.glb";
    req.motion.out_json = clip_json_out;
    req.motion.name = "prompted";

    avatar::Engine engine(std::move(cfg));
    std::string err;

    // The cancellation token a service would keep in its job record. One per request; tripped from
    // any thread. --cancel-after is how the worst-case cancel latency in the report was measured:
    // it timestamps the cancel() and the return, and the difference IS the latency.
    auto tok = std::make_shared<avatar::CancelToken>();
    g_sigint_token = tok.get();
    std::signal(SIGINT, on_sigint);
    std::atomic<double> cancel_at{0};
    std::thread canceller;
    auto arm_canceller = [&]() {
        if (cancel_after <= 0) return;
        canceller = std::thread([&, tok] {
            std::this_thread::sleep_for(std::chrono::duration<double>(cancel_after));
            cancel_at.store(now_s());
            std::printf("\n[cancel] tripping the token %.1fs in\n", cancel_after);
            std::fflush(stdout);
            tok->cancel();
        });
    };
    auto report_cancel = [&](const char* what) {
        if (canceller.joinable()) canceller.join();
        if (cancel_at.load() > 0)
            std::printf("[cancel] %s returned %.3fs after cancel() — THAT is the worst-case latency "
                        "for the point it stopped at\n", what, now_s() - cancel_at.load());
    };
    auto print_health = [&](const char* tag) {
        avatar::Health h = engine.health();
        std::printf("[health %s] {status:%s busy:%s in_flight:%d queued:%d draining:%s "
                    "waiting_for_gpu_lock:%s loaded:%s renders_done:%ld last_render_sec:%.1f "
                    "idle_sec:%.1f idle_unload_seconds:%d}\n",
                    tag, h.status.c_str(), h.busy ? "true" : "false", h.in_flight, h.queued,
                    h.draining ? "true" : "false", h.waiting_for_gpu_lock ? "true" : "false",
                    h.loaded ? "true" : "false",
                    h.renders_done, h.last_render_sec, h.idle_sec, h.idle_unload_seconds);
        std::fflush(stdout);
    };

    if (queue_demo > 0) {
        // SINGLE-FLIGHT PROOF: N threads, one card. Every one is admitted, none is rejected, and
        // they run strictly one at a time in arrival order.
        std::printf("==== queue-demo: %d concurrent requests, one Engine ====\n", queue_demo);
        std::mutex io;
        std::vector<std::thread> th;
        for (int k = 0; k < queue_demo; k++) {
            th.emplace_back([&, k] {
                const double t_sub = now_s();
                auto s = engine.session();
                { std::lock_guard<std::mutex> g(io);
                  std::printf("  req %d: admitted after %.2fs (status %d)\n", k, now_s() - t_sub, s.status_code()); }
                if (!s.ok()) return;
                avatar::MotionRequest m = req.motion;
                m.name = "q" + std::to_string(k);
                m.seed = (uint32_t)k;
                avatar::MotionResult mr;
                std::string e;
                const bool okk = engine.generate_motion(m, mr, e);
                { std::lock_guard<std::mutex> g(io);
                  std::printf("  req %d: %s (%.1fs)\n", k, okk ? "done" : e.c_str(), mr.seconds); }
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(200));   // stagger, so the order is arrival order
        }
        for (int i = 0; i < 6 && i < queue_demo * 4; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(700));
            print_health("during");
        }
        for (auto& t : th) t.join();
        print_health("after");
        return 0;
    }

    if (motion_only) {
        if (req.out_glb.empty() && clip_json_out.empty()) { usage(); return 2; }
        if (req.motion.out_json.empty()) req.motion.out_json = req.out_glb;
        avatar::MotionResult mr;
        auto sess = engine.session(tok.get());
        if (!sess.ok()) { std::fprintf(stderr, "FAIL(%d): %s\n", sess.status_code(), sess.error().c_str()); return 1; }
        arm_canceller();
        if (!engine.generate_motion(req.motion, mr, err)) { report_cancel("generate_motion"); std::fprintf(stderr, "FAIL: %s\n", err.c_str()); return err == "cancelled" ? 130 : 1; }
        report_cancel("generate_motion");
        std::printf("motion: %d frames @ %g fps in %.1fs (load %.1fs, encode %.1fs, solve %.1fs; DiT was %s)\n",
                    mr.frames, mr.fps, mr.seconds, mr.load_s, mr.encode_s, mr.solve_s,
                    mr.dit_was_resident ? "RESIDENT" : "cold");
        std::printf("peak VRAM used: %zu MiB device-wide (%zu MiB was already held by a co-tenant at start)\n",
                    engine.peak_vram_used_mib(), engine.baseline_vram_used_mib());
        return 0;
    }

    if (rig_only) {
        if (req.rig.image.empty() || req.out_glb.empty()) { usage(); return 2; }
        req.rig.out_glb = req.out_glb;
        avatar::RigResult rr;
        auto sess = engine.session(tok.get());
        if (!sess.ok()) { std::fprintf(stderr, "FAIL(%d): %s\n", sess.status_code(), sess.error().c_str()); return 1; }
        arm_canceller();
        if (!engine.build_rig(req.rig, rr, err)) { report_cancel("build_rig"); std::fprintf(stderr, "FAIL: %s\n", err.c_str()); return err == "cancelled" ? 130 : 1; }
        report_cancel("build_rig");
        std::printf("rig: %s  J=%d named_core=%d/22  %.1fs\n", rr.glb.c_str(), rr.joints, rr.named_core, rr.seconds);
        std::printf("peak VRAM used: %zu MiB device-wide (%zu MiB was already held by a co-tenant at start)\n",
                    engine.peak_vram_used_mib(), engine.baseline_vram_used_mib());
        return 0;
    }

    if (!animate_only_rig.empty()) {
        if (req.out_glb.empty()) { usage(); return 2; }
        avatar::AnimateRequest ar;
        if (clip_in.empty()) {
            avatar::MotionResult mr;
            if (!engine.generate_motion(req.motion, mr, err)) { std::fprintf(stderr, "FAIL: %s\n", err.c_str()); return 1; }
            ar.clip_json.push_back(mr.clip_json);
        } else {
            ar.clip_path = clip_in;
        }
        ar.rig_glb = animate_only_rig;
        ar.variant = req.variant;
        ar.smooth = req.smooth;
        ar.exercise = req.exercise;
        ar.out_glb = req.out_glb;
        avatar::AnimateResult res;
        if (!engine.animate(ar, res, err)) { std::fprintf(stderr, "FAIL: %s\n", err.c_str()); return 1; }
        std::printf("animated: %s  %d animations, map=%s %d/22, swap=%s, exercise %d joints  %.2fs\n",
                    res.glb.c_str(), res.animations, res.map_kind.c_str(), res.mapped,
                    res.swap ? "on" : "off", res.exercise_joints, res.seconds);
        std::printf("peak VRAM used: %zu MiB device-wide (%zu MiB was already held by a co-tenant at start)\n",
                    engine.peak_vram_used_mib(), engine.baseline_vram_used_mib());
        return 0;
    }

    if (serve_demo) {
        // THE PROPERTY THREE BINARIES CANNOT GIVE US: two requests, one process, weights loaded
        // once. The second motion request must report load 0.0s / DiT RESIDENT.
        std::printf("==== serve-demo: two motion requests through ONE Engine ====\n");
        // Hold the card across both requests: without this each one re-queues behind whatever else
        // shares the 3060, and the demo measures the queue rather than the engine.
        auto session = engine.session(tok.get());
        if (!session.ok()) { std::fprintf(stderr, "FAIL(%d): %s\n", session.status_code(), session.error().c_str()); return 1; }
        for (int k = 0; k < 2; k++) {
            avatar::MotionRequest m = req.motion;
            m.name = k ? "second" : "first";
            m.seed = (uint32_t)k;
            m.out_json = req.out_glb + "." + m.name + ".clip.json";
            avatar::MotionResult mr;
            if (!engine.generate_motion(m, mr, err)) { std::fprintf(stderr, "FAIL: %s\n", err.c_str()); return 1; }
            std::printf("  request %d: %d frames, wall %.1fs = load %.1fs + encode %.1fs + solve %.1fs   DiT %s\n",
                        k + 1, mr.frames, mr.seconds, mr.load_s, mr.encode_s, mr.solve_s,
                        mr.dit_was_resident ? "REUSED (0 load)" : "loaded");
        }
        std::printf("peak VRAM used after two requests: %zu MiB device-wide (%zu MiB co-tenant at start)\n",
                    engine.peak_vram_used_mib(), engine.baseline_vram_used_mib());
        if (req.rig.image.empty()) return 0;
        std::printf("==== and now the full chain in the SAME process ====\n");
    }

    if (req.rig.image.empty() || req.out_glb.empty()) { usage(); return 2; }
    avatar::Result res;
    req.cancel = tok.get();
    print_health("before");
    arm_canceller();
    if (!engine.run(req, res, err)) {
        report_cancel("run");
        std::fprintf(stderr, "%s: %s\n", res.cancelled ? "CANCELLED" : "FAIL", err.c_str());
        print_health("after-cancel");
        // The proof that a cancel gives the card back, not just stops using it.
        avatar::GpuStatus g = engine.gpu_status();
        std::printf("[gpu] {loaded:%s gpu:%s mem_free_mb:%ld}\n",
                    g.loaded ? "true" : "false", g.gpu.empty() ? "null" : g.gpu.c_str(), g.mem_free_mb);
        return res.cancelled ? 130 : 1;
    }
    report_cancel("run");

    std::printf("\n==== avatar_e2e DONE ====\n");
    std::printf("  motion   %6.1fs   %d frames @ %g fps  (load %.1f / encode %.1f / solve %.1f, DiT %s)\n",
                res.motion.seconds, res.motion.frames, res.motion.fps, res.motion.load_s,
                res.motion.encode_s, res.motion.solve_s, res.motion.dit_was_resident ? "reused" : "loaded");
    std::printf("  rig      %6.1fs   %s  J=%d  named_core=%d/22\n", res.rig.seconds, res.rig.glb.c_str(),
                res.rig.joints, res.rig.named_core);
    std::printf("  animate  %6.2fs   %d animations, map=%s %d/22, swap=%s, rest mismatch arm/leg/spine "
                "%.1f/%.1f/%.1f deg, exercise %d joints\n",
                res.anim.seconds, res.anim.animations, res.anim.map_kind.c_str(), res.anim.mapped,
                res.anim.swap ? "on" : "off", res.anim.rest_arm_deg, res.anim.rest_leg_deg,
                res.anim.rest_spine_deg, res.anim.exercise_joints);
    std::printf("  TOTAL    %6.1fs   peak VRAM %zu MiB device-wide, %zu MiB above the %zu MiB a co-tenant\n"
                "                     already held when the engine started\n",
                res.total_s, res.peak_vram_used_mib,
                res.peak_vram_used_mib > res.baseline_vram_used_mib ? res.peak_vram_used_mib - res.baseline_vram_used_mib : 0,
                res.baseline_vram_used_mib);
    std::printf("  -> %s\n", res.anim.glb.c_str());
    return 0;
}
