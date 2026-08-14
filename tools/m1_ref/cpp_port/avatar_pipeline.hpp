// avatar_pipeline.hpp — image + text prompt -> rigged, animated GLB, in ONE process.
//
// THIS HEADER IS THE SEAM AN API IS WRITTEN AGAINST. It is deliberately free of ggml, CUDA, glTF
// and stb: a service links libavatar_pipeline.a and includes this, and needs none of the model
// stack in its own translation units.
//
// WHAT IT REPLACES.  Three binaries and a bash driver:
//     image_to_rig                 photo -> textured + rigged GLB   (matte, geometry, DC, bake, rig)
//     hymotion                     prompt -> smplh22 motion clip
//     motion_retarget              rig + clip -> animated GLB
//     shootout/final_e2e_dc_rig.sh the 3060 lock, VRAM sampling, QC gates, exercise clip
// All four are still there and still work; the A/B harness needs them. Nothing here shells out to
// any of them.
//
// ============================================================================================
// THE CALL SEQUENCE AN API AUTHOR WANTS. Read this before anything else.
// ============================================================================================
//
//   // ---- once, at process start (slow: this is where weights become resident) --------------
//   avatar::EngineConfig cfg;
//   cfg.hymotion_gguf = ...; cfg.llm_gguf = ...; cfg.clip_l_gguf = ...; cfg.geo_model_dir = ...;
//   cfg.gpu_lock_path = "/mnt/hdd/3d/avatar-shootout/_shootout_out/.3060-image-to-rig.lock";
//   static avatar::Engine engine(cfg);          // ONE per process. NOT thread-safe. Never copied.
//   std::setvbuf(stdout, nullptr, _IOLBF, 0);   // the stages log progress to stdout; without this
//                                               // libc block-buffers it and a live render looks dead
//
//   // ---- GET /health : answer from ANY thread, at ANY time, taking no render lock ----------
//   avatar::Health h = engine.health();
//   // -> {status, busy, in_flight, queued, draining, waiting_for_gpu_lock, loaded, loaded_model,
//   //     renders_done, last_render_sec, uptime_sec, idle_sec, idle_unload_seconds, keep_resident}
//   // `busy` is "on the card". `queued` is "waiting for the card" and is REPORTED SEPARATELY on
//   // purpose: a health that only reports busy shows an idle service that is actually backed up,
//   // and every admin action that must not race a pending request (unload, swap, shutdown) then
//   // fires at the wrong moment. `in_flight` = busy + queued, which is what an idle-unload
//   // watchdog must gate on.
//
//   // ---- GET /v1/gpu/status : residency, NOT activity (fleet convention keeps VRAM out of
//   //      /health because /health must answer while the models are unloaded) ------------------
//   avatar::GpuStatus g = engine.gpu_status();   // {loaded, gpu, mem_free_mb}
//
//   // ---- POST /render : ONE request on the card at a time; the rest QUEUE -------------------
//   auto tok = std::make_shared<avatar::CancelToken>();   // keep it in your job record
//   {
//       avatar::Engine::Session s = engine.session(tok.get());  // FIFO. Blocks until it is our turn.
//       if (!s.ok()) return http(s.status_code(), s.error());   // 503 draining / 499 cancelled / 504 lock
//       avatar::Request  req;  req.rig.image = ...; req.motion.prompt = ...; req.out_glb = ...;
//       req.cancel = tok.get();                                 // (redundant here: the Session has it)
//       avatar::Result   res;  std::string err;
//       if (!engine.run(req, res, err))
//           return res.cancelled ? http(499, "cancelled") : http(500, err);
//       return http(200, res.anim.glb);
//   }   // ~Session: releases the card, disarms the token, wakes the next waiter
//
//   // ---- POST /render/{id}/cancel : from ANOTHER thread, any time -------------------------
//   tok->cancel();   // returns immediately; the render stops at its next cancellation point
//
// The stages (build_rig / generate_motion / animate) are also callable one at a time, and a real
// service usually wants them: rigs are cached per character and only the motion is per-request.
// Open ONE Session around a group of stage calls — see Session's own comment for why.
//
// ============================================================================================
// THE SHAPE, AND WHY IT SUITS A LONG-LIVED SERVICE
//   * Engine is constructed ONCE with weights + device policy and lives for the process. It is not
//     a function that happens to be called repeatedly: it owns residency.
//   * Per-request work is in build_rig / generate_motion / animate, each of which takes a request
//     struct and fills a result struct. run() is sugar that chains the three.
//   * Nothing between the stages goes through the filesystem unless the caller asks for it: the
//     motion clip crosses as JSON TEXT, not a path. Files that ARE written are artifacts, not IPC.
//   * The engine is NOT thread-safe and does not pretend to be. One 3060, one CUDA context, one
//     request at a time; a service SERIALISES by queueing (Session), not by threading this object.
//     Session, health() and CancelToken::cancel() ARE safe to call from other threads — they are
//     the whole thread-safe surface, and they are the only three a request handler needs.
//
// THE OPERATIONAL CONCERNS, AND WHERE THEY LANDED (see the report / the .cpp for the reasoning)
//   * IN-PROCESS QUEUEING vs THE CROSS-PROCESS FLOCK — they compose, in this order and only this
//     order: Session first (FIFO among OUR requests), then the flock (arbitration against the
//     other agents' shell drivers on this box), taken ONCE for the whole request. Never the
//     reverse: queueing while holding the flock would park the whole card behind our own queue.
//     The flock stays REQUEST-scoped with counted re-entrancy — a stage-scoped lock re-queues at
//     every stage and starved two runs for ~50 minutes — and it still FAILS CLOSED on timeout.
//   * Peak VRAM: IN, because it costs a cudaMemGetInfo on a timer and removes a bash+nvidia-smi
//     sampler that could only ever watch a whole process.
//   * The QC gates (rig_weight_health.py, rig_pose_smoke.py) are OUT and stay Python. They are
//     fail-closed PUBLISH gates that never run inside an API, and rewriting a working gate only
//     risks one that wrongly passes.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace avatar {

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------
// A render is ~450 s. The caller WILL go away, and the user WILL press stop. Hand a token to
// Engine::session() and trip it from any thread; the render stops at its next cancellation point,
// writes nothing, and gives the card back.
//
// Cancellation points sit at every stage boundary AND inside the three long loops: the geometry
// diffusion sampler (one per Euler step, and every geometry DiT goes through that one loop), the
// narrow-band DC ladder (one per level), and the rig's beam decode (one per token). A cancel that
// only landed at stage boundaries would be nearly useless here — the geometry stage alone is
// minutes long.
//
// Not copyable and not movable ON PURPOSE: the engine holds a pointer to it for the duration of the
// request, so its address has to be stable. Keep it in your job record, e.g. a shared_ptr.
class CancelToken {
public:
    CancelToken() = default;
    CancelToken(const CancelToken&) = delete;
    CancelToken& operator=(const CancelToken&) = delete;

    void cancel()          { flag_.store(true, std::memory_order_release); }
    bool cancelled() const { return flag_.load(std::memory_order_acquire); }
    void reset()           { flag_.store(false, std::memory_order_release); }   // reuse for a retry

private:
    friend class Engine;
    std::atomic<bool> flag_{false};
};

// ---------------------------------------------------------------------------
// Process-lifetime configuration. Weights and device policy — never per request.
// ---------------------------------------------------------------------------
struct EngineConfig {
    // --- geometry + rig (image_to_rig's stage) ---
    std::string geo_model_dir = "/home/dbrain/models/3d/geo_f16_native";
    std::string rig_r1w;        // "" = image_to_rig's own default
    std::string rig_qwen3_w;    // ditto (this is the RIG's Qwen3, not the motion text encoder)
    std::string rig_skin_vae;   // ditto

    // --- motion (HY-Motion) ---
    std::string hymotion_gguf;
    std::string llm_gguf;       // Qwen3-8B, the motion TEXT encoder
    std::string clip_l_gguf;

    bool use_cuda = true;
    int  threads = 4;
    bool verbose = true;

    // Serialisation of GPU work ACROSS PROCESSES. "" = the caller owns it (default).
    // This is the flock the shell A/B drivers on this box also take, which is the only reason it
    // is in here: the card is shared. It is taken per REQUEST (see Session), not per stage.
    std::string gpu_lock_path;
    int         gpu_lock_timeout_s = 7200;

    // In-process peak free-VRAM sampler (cudaMemGetInfo on a timer). Cheap; on by default.
    bool sample_vram = true;
    int  vram_sample_ms = 250;

    // Keep the ~2.1 GB motion DiT resident between requests. A motion-only service wants this ON;
    // a service that interleaves rig builds wants it OFF, because the rig stage peaks near 10.7 GB
    // of the 3060's 11.9 GB and 2.1 GB of squatting weights is the difference between fitting and
    // not. run() therefore releases it around the rig stage regardless (see the .cpp).
    bool keep_motion_resident = true;

    // Idle unload, the fleet's shape: a 1 s-tick watchdog drops resident weights after this many
    // idle seconds. 0 = never. It only ever fires with in_flight == 0 (queued counts as in flight,
    // so it cannot evict under a request that has not reached the card yet) and it takes the card
    // NON-BLOCKINGLY, so it can never delay an arriving request.
    int idle_unload_seconds = 300;
};

// ---------------------------------------------------------------------------
// Stage 1: image -> textured + rigged GLB
// ---------------------------------------------------------------------------
struct RigRequest {
    std::string image;        // ANY photo. RMBG-2.0 mattes it in-process; an existing matte is
                              // classified and passed through untouched.
    std::string out_glb;
    std::string stage_dir;    // "" = no intermediates
    std::string from_refined; // resume: skip geometry + DC, re-enter at bake+rig
    std::string rig_cache;    // share ONE skeleton across LOD tiers
    int  texsize = 2048;
    int  decimate = 220000;
    int  resolution = 0;      // 0 = the binary's default lattice
    int  dc_band = 0;         // 0 = default
    bool dc_remesh = true;    // the Python-parity O-Voxel narrow-band DC path
    // BEST-OF-N over conditioning draws. The rig decode is a DRAW LOTTERY: six draws of the shipped
    // miku mesh (identical weights, beam config and decode seed — only the 8192-point surface sample
    // varying) gave J = 37/153/37/161/196/192, and four of the six welded every vertex to one joint.
    // A new subject is a coin flip. Each extra draw costs one rig decode (~20-27 s of a ~450 s
    // request) and the selector keeps the best one. 0 = the historical one-shot path, unchanged.
    //
    // 🔴 THE DELIVERY DEFAULT IS 2, and the CLI harness's is still 0. A rejected draw is usually a
    // ~214 s runaway, so 2 bounds the worst case near ~430 s of rig — worth it against a coin flip,
    // and it also turns the pose gate on (image_to_rig defaults that to `rig_retries > 0`). The
    // image_to_rig binary keeps its historical 0 so the A/B harness's numbers stay comparable.
    int  rig_retries = 2;
    // ONE more draw beyond that budget, and only in the corner where the budget ran out with nothing
    // accepted AND nothing even passing the deformation gate. -1 = the stage's own default (1),
    // 0 = never. It is an OPTIMISATION for a bad run: exhausting it does not fail the request, it
    // just ships the best-ranked draw with its verdict attached.
    int  rig_extra_retries = -1;
    // Post-process the delivered skin to remove the "spike" artefact — vertices bound to a joint
    // half a body away, and weight fields that vary faster than the mesh can carry. See
    // rig_weight_cleanup.hpp. It is an OFFER, not a gate: it verifies itself against the shipped
    // pose gate and keeps the original skin whenever it cannot prove an improvement, so on a rig
    // that is already good it is a no-op that costs about a second.
    bool skin_cleanup = true;
    // Escape hatch onto image_to_rig's full flag surface, for the options this struct does not
    // name. Parsed by image_to_rig's OWN parser, applied ON TOP of the fields above — so a flag
    // here wins, exactly as a later argv element does.
    std::vector<std::string> extra_args;
};

// Reserved for the rig-selector work: the structural/anatomy gate's outcome, so a service can say
// WHY a rig was rejected instead of only that it was. `ran == false` until the gate reports.
struct RigGateResult {
    bool        ran = false;
    bool        passed = false;
    int         hypotheses = 0;     // decoded beams considered
    int         rejected = 0;       // ...and how many the gate threw out
    std::string reason;             // "" when passed

    // --- the DRAW selector (RigRequest::rig_retries), filled by the rig stage --------------------
    // `accepted == false` means no draw cleared the predicate and the best-ranked one was delivered
    // anyway. IT IS ADVISORY, NOT A REFUSAL: a rig can fail the deformation gate and still be a
    // decent asset (char1 reads 25.9 against a limit of 6.0 and moves correctly — the defects are
    // cosmetic spikes), so the delivery ships with the verdict attached and the caller decides. The
    // one thing that DOES fail closed is `skin_ok`: a rig with zero weights deforms nothing and
    // would poison the shared rig cache for every LOD tier.
    //
    // This is a verdict computed BEFORE the write, so it is not recoverable from the GLB afterwards.
    int    draws = 0;                 // conditioning draws decoded; 0 = the skeleton came from cache
    int    accepted_draw = -1;        // which draw was accepted; -1 = none, best-ranked was shipped
    bool   accepted = false;
    bool   skin_ok = false;           // R4 decoded weights at all (false = the rig deforms NOTHING)
    bool   humanoid_gate_ok = false;  // the strict 22-bone anatomy gate accepted
    bool   generic_gate_ok = false;   // the size-aware creature gate accepted (a DIFFERENT claim)
    int    selector_named_core = 0;   // SMPL-22 core bones NAMED on the shipped draw
    bool   pose_gate_ran = false;
    bool   pose_gate_pass = false;
    double pose_gate_worst = 0;       // worst all-influential / per-component p999 stretch; limit 6.0
    std::string selector_summary;     // the one-line verdict, already formatted
};

struct RigResult {
    std::string glb;
    int    joints = 0;        // J, read back off the written GLB (not scraped from a log)
    int    named_core = 0;    // how many of the SMPL-22 core bones got standard names
    std::string skin_cleanup; // one-line verdict from the skin-weight cleanup ("" = not run)
    double seconds = 0;
    RigGateResult gate;
};

// ---------------------------------------------------------------------------
// Stage 2: text prompt -> SMPL-H 22 motion clip
// ---------------------------------------------------------------------------
struct MotionRequest {
    std::string prompt;
    bool        uncond = false;   // the checkpoint's own trained null conditioning (no text encoder)
    float       seconds = 4.0f;
    float       cfg = 5.0f;
    int         steps = 50;
    uint32_t    seed = 0;
    std::string name = "motion";
    std::string out_json;         // optional artifact; the pipeline does not need it
};

struct MotionResult {
    std::string clip_json;        // the clip itself, in memory
    std::string path;             // "" unless out_json was set
    int    frames = 0;
    double fps = 30;
    double seconds = 0;
    // The reuse evidence: 0 load_s with dit_was_resident means the DiT came from the last request.
    bool   dit_was_resident = false;
    double load_s = 0, encode_s = 0, solve_s = 0;
};

// ---------------------------------------------------------------------------
// Stage 3: rigged GLB + clips -> animated GLB
// ---------------------------------------------------------------------------
struct AnimateRequest {
    std::string rig_glb;
    std::vector<std::string> clip_json;   // clip TEXT, not paths
    std::vector<std::string> clip_path;   // ...or paths, for clips already on disk
    std::string variant = "rotbase";      // fix | nofix | base | rot | rotbase
    int  smooth = 2;
    bool exercise = true;                 // also bake the skeleton-exercise eye-test clip
    std::string out_glb;
};

struct AnimateResult {
    std::string glb;
    int    animations = 0;
    int    mapped = 0;          // SMPL-22 slots the bone mapper filled
    std::string map_kind;       // "names" | "topology"
    bool   swap = false;        // the measured L/R decision
    double rest_arm_deg = 0, rest_leg_deg = 0, rest_spine_deg = 0;
    int    exercise_joints = 0;
    double seconds = 0;
};

// ---------------------------------------------------------------------------
// The one call
// ---------------------------------------------------------------------------
struct Request {
    RigRequest    rig;
    MotionRequest motion;
    std::string   out_glb;      // the animated delivery
    bool          exercise = true;
    std::string   variant = "rotbase";
    int           smooth = 2;
    // The token run() should arm. Set it OR open your own Session with one — not both, and if you
    // do both the Session wins (run() nests inside it and never re-arms). It is on the request
    // because run() opens its own Session internally, and a token you cannot hand it is a
    // cancellation that silently never fires.
    CancelToken*  cancel = nullptr;
};

struct Result {
    RigResult     rig;
    MotionResult  motion;
    AnimateResult anim;
    double        total_s = 0;
    size_t        peak_vram_used_mib = 0;      // device-wide, sampled in process
    size_t        baseline_vram_used_mib = 0;  // what a co-tenant already held when we started
    // Set when the request stopped because its token was tripped. A cancelled call returns FALSE
    // with err = "cancelled"; this is how a service tells that apart from a real failure (499 vs
    // 500) without string-matching the error.
    bool          cancelled = false;
};

// ---------------------------------------------------------------------------
// What /health and /v1/gpu/status report. Field names follow the rest of this box's GPU fleet
// (ltx-video, longcat-avatar, vision-server), so a dashboard that already reads those reads this.
// ---------------------------------------------------------------------------
struct Health {
    std::string status = "ok";       // "ok" | "degraded" (fatal non-empty)
    bool   busy = false;             // our one slot is taken. NOT the same as "kernels are running":
                                     // see waiting_for_gpu_lock.
    int    in_flight = 0;            // running OR queued — what an idle watchdog must gate on
    int    queued = 0;               // waiting for the card. Reported separately ON PURPOSE.
    bool   draining = false;         // refusing new admissions
    // The admitted request has our slot but is stuck waiting for the CROSS-PROCESS flock, i.e.
    // something else on this box owns the card. Without this field `busy` is ambiguous — an
    // operator watching a service that is making no progress cannot tell "our render is slow" from
    // "another process has the GPU and we have not started". It is the honest answer to "is the
    // in-process queue authoritative?": it is authoritative over OUR requests only.
    bool   waiting_for_gpu_lock = false;
    bool   loaded = false;           // weights resident. NOT the same thing as busy.
    std::string loaded_model;        // "" when nothing is resident
    long   renders_done = 0;
    double last_render_sec = 0;
    double uptime_sec = 0;
    double idle_sec = 0;             // since the last request finished
    int    idle_unload_seconds = 0;  // echoes the config, so the reader can explain an unload
    bool   keep_resident = true;
    std::string fatal;               // non-empty => status "degraded"
};

struct GpuStatus {
    bool        loaded = false;      // weights resident, NOT "rendering"
    std::string gpu;                 // "" when nothing is resident
    long        mem_free_mb = -1;    // -1 = could not ask the driver
};

class Engine {
public:
    // ---------------------------------------------------------------------------------------
    // Session — ADMISSION. Constructing one says "this request wants the card"; it blocks, FIFO,
    // until every earlier request has finished, then takes the cross-process flock ONCE for the
    // whole request and arms the cancellation token.
    //
    // Open ONE around a whole request, not one per stage. Without it each stage queues for the
    // card independently, which on a shared box lets a multi-stage request be starved between its
    // own stages: measured here, two consecutive runs lost the card between the motion and rig
    // stages and made no progress in ~50 minutes. run() opens one internally; a caller driving the
    // stages itself (batch of clips, rig + N motions) should open one too. RAII, and re-entrant
    // with run() and with the stage calls — nesting is COUNTED, never re-taken, because an flock
    // attaches to the open file description and a second LOCK_EX would deadlock against ourselves.
    //
    // ok() == false means DO NOT TOUCH THE GPU. status_code() gives the HTTP status the fleet uses
    // for that reason: 503 draining, 499 cancelled while queued, 504 the flock timed out.
    // ---------------------------------------------------------------------------------------
    class Session {
    public:
        ~Session();
        Session(Session&&) noexcept;
        Session(const Session&) = delete;
        bool ok() const;
        const std::string& error() const;
        int  status_code() const;        // 200 | 503 draining | 499 cancelled | 504 lock timeout
        int  queue_position() const;     // 1-based while queued, 0 once it owns the card
    private:
        friend class Engine;
        Session(Engine& e, CancelToken* tok, bool blocking);
        struct State;
        std::unique_ptr<State> st_;
    };
    // The normal one: queue for the card, cancellable while queued.
    Session session(CancelToken* tok = nullptr);
    // Non-blocking: ok() == false immediately if the card is busy. This is what the idle-unload
    // watchdog uses so it can never delay an arriving request.
    Session try_session();

    explicit Engine(EngineConfig cfg);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool build_rig(const RigRequest&, RigResult&, std::string& err);
    bool generate_motion(const MotionRequest&, MotionResult&, std::string& err);
    bool animate(const AnimateRequest&, AnimateResult&, std::string& err);

    // rig -> motion -> retarget -> exercise, in this process.
    bool run(const Request&, Result&, std::string& err);

    // Drop every resident model. A service calls this under memory pressure, on drain, or never.
    // Callers other than the engine itself must hold a Session: it touches the CUDA context.
    void release_gpu();

    // Thread-safe, lock-free, and answerable while a render is in flight or while unloaded.
    Health    health() const;
    GpuStatus gpu_status() const;
    void      drain();     // refuse new admissions; in-flight work finishes
    void      undrain();

    size_t peak_vram_used_mib() const;
    size_t baseline_vram_used_mib() const;   // co-tenant usage at engine construction
    void   reset_vram_peak();

    const EngineConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace avatar
