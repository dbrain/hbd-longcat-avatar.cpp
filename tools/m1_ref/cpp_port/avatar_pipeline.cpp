// avatar_pipeline.cpp — the implementation behind avatar_pipeline.hpp.
//
// Three stages, one process, no fork:
//   build_rig()        -> i2r::run(i2r::Options), the SAME translation unit the CLI is, compiled
//                         with -DIMAGE_TO_RIG_LIB. A TYPED STRUCT, not a synthesised argv.
//   generate_motion()  -> HYMotion::Engine, which keeps the DiT resident across requests.
//   animate()          -> mret:: retarget + bake, and rigex:: for the exercise clip.
//
// WHAT STILL LEAVES THE PROCESS: nothing on this path. There is no popen, no system(), no
// posix_spawn and no fork anywhere in the chain. The only external calls are open/read/write on
// weights and artifacts, and the CUDA driver.
#include "avatar_pipeline.hpp"

#include "cancel_hook.hpp"
#include "image_to_rig_options.hpp"
#include "motion_glb_anim.hpp"
#include "motion_retarget.hpp"
#include "rig_bone_names.hpp"
#include "rig_exercise.hpp"
#include "rig_stage_report.hpp"
#include "smpl_rest_joints.hpp"

#include "hymotion_engine.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <sys/file.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace avatar {

namespace {

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur.push_back(p[i]);
        if (p[i] == '/' || i + 1 == p.size()) ::mkdir(cur.c_str(), 0755);
    }
}
void mkparent(const std::string& p) {
    const size_t s = p.find_last_of('/');
    if (s != std::string::npos && s > 0) mkdirs(p.substr(0, s));
}

// --- artifact hygiene -------------------------------------------------------
// A cancelled render must not leave a half-written GLB behind for someone to serve. The stages'
// writers are the LAST thing each stage does and there is no cancellation point inside them, so a
// truncated file is not actually reachable today — but "not reachable today" is one refactor away
// from a corrupt delivery, so the guard is explicit: unlink an output this request created unless
// the request committed it. A file that ALREADY EXISTED is never touched (a retry that fails must
// not destroy the previous good delivery — the same mistake that once destroyed the takes it was
// meant to replace).
class OutputGuard {
public:
    explicit OutputGuard(std::string p) : p_(std::move(p)) {
        if (!p_.empty()) existed_ = (::access(p_.c_str(), F_OK) == 0);
    }
    void commit() { committed_ = true; }
    ~OutputGuard() {
        if (!committed_ && !existed_ && !p_.empty()) ::unlink(p_.c_str());
    }
    OutputGuard(const OutputGuard&) = delete;
private:
    std::string p_;
    bool existed_ = false, committed_ = false;
};

// --- GPU serialisation, ACROSS PROCESSES ------------------------------------
// An RAII flock. This is the same lock file the shell drivers use, so a library request and a
// shell A/B still serialise against each other — which is the only reason it is in here at all:
// the card is shared with other agents on this box. It is REQUEST-scoped (taken by Session), not
// stage-scoped.
class GpuLock {
public:
    // `depth` is the caller's re-entrancy counter. flock() locks attach to the OPEN FILE
    // DESCRIPTION, so a second LOCK_EX on a second fd of the same file BLOCKS FOREVER against
    // ourselves — nesting has to be counted, not attempted.
    // `cancel` lets a caller abandon the WAIT (not just the render): without it a cancelled
    // request would sit here for up to gpu_lock_timeout_s (7200 s) after the user let go.
    GpuLock(const std::string& path, int timeout_s, bool verbose, int* depth,
            const std::atomic<bool>* cancel)
        : depth_(depth) {
        if (depth_ && (*depth_)++ > 0) { ok_ = true; return; }   // an outer scope already holds it
        if (path.empty()) { ok_ = true; return; }   // caller owns serialisation: nothing to take
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (fd_ < 0) { err_ = "cannot open the GPU lock file " + path; return; }
        const double t0 = now_s();
        while (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            if (cancel && cancel->load(std::memory_order_relaxed)) {
                err_ = "cancelled while waiting for the GPU lock";
                cancelled_ = true;
                ::close(fd_);
                fd_ = -1;
                return;
            }
            if (now_s() - t0 > timeout_s) {
                // FAIL, do not proceed unlocked. A caller that asked for serialisation and is
                // silently given none will run a 10.7 GB stage next to whatever already owns the
                // card and OOM — with a stack trace that blames the wrong thing.
                err_ = "timed out after " + std::to_string(timeout_s) + "s waiting for " + path;
                if (verbose) std::fprintf(stderr, "avatar: %s\n", err_.c_str());
                ::close(fd_);
                fd_ = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (verbose && now_s() - t0 > 1.0)
            std::fprintf(stderr, "avatar: took the GPU lock after %.0fs\n", now_s() - t0);
        held_ = true;
        ok_ = true;
    }
    bool ok() const { return ok_; }
    bool cancelled() const { return cancelled_; }
    const std::string& error() const { return err_; }
    ~GpuLock() {
        if (depth_) --(*depth_);
        if (fd_ >= 0) { if (held_) ::flock(fd_, LOCK_UN); ::close(fd_); }
    }
    GpuLock(const GpuLock&) = delete;
    GpuLock& operator=(const GpuLock&) = delete;
private:
    int*        depth_ = nullptr;
    int         fd_ = -1;
    bool        held_ = false;
    bool        ok_ = false;
    bool        cancelled_ = false;
    std::string err_;
};

// --- peak VRAM --------------------------------------------------------------
// cudaMemGetInfo, on a timer, in this process. It reports DEVICE-WIDE used memory (the card is
// shared, so this is an upper bound on what we cost, not a claim about us alone) — which is
// exactly what the nvidia-smi sampler it replaces reported, minus the subprocess.
class VramSampler {
public:
    void start(int period_ms) {
        if (running_) return;
        // Baseline = what the card already holds before we do anything. The 3060 is shared, so a
        // device-wide peak on its own says nothing about our cost; peak-minus-baseline is the
        // honest bound (still an upper one if a co-tenant grows).
        { size_t f = 0, t = 0; if (cudaMemGetInfo(&f, &t) == cudaSuccess && t) baseline_ = (t - f) >> 20; }
        running_ = true;
        th_ = std::thread([this, period_ms] {
            while (running_) {
                size_t freeb = 0, totb = 0;
                if (cudaMemGetInfo(&freeb, &totb) == cudaSuccess && totb) {
                    const size_t used = (totb - freeb) >> 20;
                    free_mib_.store(freeb >> 20);
                    size_t prev = peak_.load();
                    while (used > prev && !peak_.compare_exchange_weak(prev, used)) {}
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
            }
        });
    }
    void stop() {
        if (!running_) return;
        running_ = false;
        if (th_.joinable()) th_.join();
    }
    ~VramSampler() { stop(); }
    size_t peak() const { return peak_.load(); }
    size_t baseline() const { return baseline_; }
    long   free_mib() const { return (long)free_mib_.load(); }
    void reset() { peak_.store(0); }
private:
    std::atomic<bool>   running_{false};
    std::atomic<size_t> peak_{0};
    std::atomic<size_t> free_mib_{0};
    size_t              baseline_ = 0;
    std::thread         th_;
};

void hymo_log_cb(enum sd_log_level_t level, const char* text, void*) {
    std::fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
    std::fflush(level == SD_LOG_ERROR ? stderr : stdout);
}

}  // namespace

// ---------------------------------------------------------------------------
struct Engine::Impl {
    EngineConfig cfg;
    VramSampler  vram;
    Engine*      owner = nullptr;

    // --- admission: ONE request on the card, the rest QUEUE ---------------------------------
    // The fleet's shape (vision-server's mutex, ltx-video's deque + condvar), reduced to what a
    // homelab service actually needs: a FIFO of waiters, a condvar, and counters an outsider can
    // read without touching any of it. No thread pool, no work stealing — the card is the
    // resource and there is one of it.
    //
    // FIFO matters: a plain std::mutex is not fair, and on a queue of 450 s renders an unfair
    // wakeup is a request that never runs. A deque of ticket ids gives real ordering AND a real
    // queue_position to report.
    std::mutex               qmx;
    std::condition_variable  qcv;
    std::deque<uint64_t>     waiters;
    uint64_t                 next_ticket = 1;
    // atomic because health() reads it WITHOUT the queue mutex (it must answer while a 450 s
    // render holds everything); it is only ever WRITTEN under qmx.
    std::atomic<bool>        card_busy{false};
    int                      session_depth = 0;   // re-entrancy: nested Sessions are a no-op
    int                      lock_depth = 0;      // ...and so is the flock they would take
    // WHOSE re-entrancy. Without this, "a Session already exists" would wave through a Session
    // opened on ANOTHER thread and the queue would be defeated by the first concurrent caller.
    // Nesting is a property of one request's call stack, so it is keyed on that thread.
    std::thread::id          card_owner;

    std::atomic<bool>   draining{false};
    // Set while the admitted request sits in the flock retry loop, i.e. ANOTHER PROCESS owns
    // the card. This is the one piece of state the in-process queue cannot know by itself.
    std::atomic<bool>   waiting_lock{false};
    std::atomic<int>    queued{0};
    std::atomic<long>   renders_done{0};
    std::atomic<bool>   motion_loaded{false};
    std::atomic<double> last_render_sec{0};
    std::atomic<double> last_done_s{0};
    double              started_s = now_s();
    std::string         fatal;

    // --- idle unload watchdog ---------------------------------------------------------------
    std::thread       watchdog;
    std::atomic<bool> watchdog_stop{false};

    std::unique_ptr<HYMotion::Engine> motion;

    explicit Impl(EngineConfig c) : cfg(std::move(c)) {
        if (cfg.sample_vram && cfg.use_cuda) vram.start(cfg.vram_sample_ms);
        last_done_s.store(started_s);
    }
    ~Impl() { stop_watchdog(); vram.stop(); }

    void stop_watchdog() {
        watchdog_stop.store(true);
        if (watchdog.joinable()) watchdog.join();
    }

    HYMotion::Engine* motion_engine() {
        if (!motion) {
            static std::once_flag once;
            std::call_once(once, [] { sd_set_log_callback(hymo_log_cb, nullptr); });
            HYMotion::EngineConfig mc;
            mc.model_gguf = cfg.hymotion_gguf;
            mc.qwen3_gguf = cfg.llm_gguf;
            mc.clip_l_gguf = cfg.clip_l_gguf;
            mc.force_cpu = !cfg.use_cuda;
            mc.threads = cfg.threads;
            mc.verbose = cfg.verbose;
            motion = std::make_unique<HYMotion::Engine>(mc);
        }
        return motion.get();
    }
};

// --- Session ---------------------------------------------------------------
struct Engine::Session::State {
    Engine*     e = nullptr;
    bool        outer = false;      // false => a nested Session; the outer one owns everything
    bool        holds_card = false;
    uint64_t    ticket = 0;
    CancelToken* tok = nullptr;
    std::unique_ptr<GpuLock> lock;
    std::string err;
    int         code = 200;
    double      t0 = 0;
};

Engine::Session::Session(Engine& e, CancelToken* tok, bool blocking) : st_(new State) {
    Impl& im = *e.impl_;
    st_->e = &e;
    st_->tok = tok;

    {
        std::unique_lock<std::mutex> lk(im.qmx);
        if (im.session_depth > 0 && im.card_owner == std::this_thread::get_id()) {
            im.session_depth++;              // nested on OUR thread: the outer Session owns the card
            st_->outer = false;
            return;
        }
        if (im.draining.load()) {
            st_->err = "service draining — not accepting new requests";
            st_->code = 503;
            return;
        }
        st_->outer = true;
        st_->ticket = im.next_ticket++;
        im.waiters.push_back(st_->ticket);
        im.queued.fetch_add(1);

        // Wait our turn. The 50 ms tick is what makes a CANCEL WHILE QUEUED land promptly without
        // wiring a condvar into CancelToken: a queued request is not on the card, so 50 ms of
        // latency costs nothing and the token stays a plain atomic anyone can trip.
        for (;;) {
            const bool mine = !im.waiters.empty() && im.waiters.front() == st_->ticket;
            if (mine && !im.card_busy.load()) break;
            if (tok && tok->cancelled()) {
                st_->err = "cancelled while queued";
                st_->code = 499;
                break;
            }
            if (im.draining.load()) {
                st_->err = "service draining — not accepting new requests";
                st_->code = 503;
                break;
            }
            if (!blocking) {
                st_->err = "busy: a request is already on the card";
                st_->code = 429;
                break;
            }
            im.qcv.wait_for(lk, std::chrono::milliseconds(50));
        }
        if (st_->code != 200) {                       // left the queue without the card
            for (auto it = im.waiters.begin(); it != im.waiters.end(); ++it)
                if (*it == st_->ticket) { im.waiters.erase(it); break; }
            im.queued.fetch_sub(1);
            im.qcv.notify_all();
            st_->outer = false;
            return;
        }
        im.waiters.pop_front();
        im.queued.fetch_sub(1);
        im.card_busy.store(true);
        im.session_depth = 1;
        im.card_owner = std::this_thread::get_id();
        st_->holds_card = true;
        st_->t0 = now_s();
    }

    // ORDER MATTERS, and only this order is right: in-process queue FIRST, cross-process flock
    // SECOND. Taking the flock before queueing would park the entire card — including the other
    // agents' shell drivers — behind OUR internal queue.
    cancelhook::arm(tok ? &tok->flag_ : nullptr);
    im.waiting_lock.store(true);
    st_->lock.reset(new GpuLock(im.cfg.gpu_lock_path, im.cfg.gpu_lock_timeout_s, im.cfg.verbose,
                                &im.lock_depth, tok ? &tok->flag_ : nullptr));
    im.waiting_lock.store(false);
    if (!st_->lock->ok()) {
        st_->err = st_->lock->error();
        st_->code = st_->lock->cancelled() ? 499 : 504;
    }
}

Engine::Session::Session(Session&&) noexcept = default;

Engine::Session::~Session() {
    if (!st_ || !st_->e) return;
    Impl& im = *st_->e->impl_;
    if (!st_->outer) {
        if (!st_->holds_card && st_->code == 200) {   // a nested Session that never queued
            std::lock_guard<std::mutex> lk(im.qmx);
            if (im.session_depth > 0) im.session_depth--;
        }
        return;
    }
    cancelhook::disarm();
    st_->lock.reset();                                 // release the flock BEFORE the in-process card
    {
        std::lock_guard<std::mutex> lk(im.qmx);
        im.card_busy.store(false);
        im.session_depth = 0;
        im.card_owner = std::thread::id();
    }
    im.last_done_s.store(now_s());
    if (st_->code == 200) {                            // a lock timeout is not a completed render
        im.last_render_sec.store(now_s() - st_->t0);
        im.renders_done.fetch_add(1);
    }
    im.qcv.notify_all();
}

bool Engine::Session::ok() const { return st_ && st_->code == 200; }
const std::string& Engine::Session::error() const {
    static const std::string none;
    return st_ ? st_->err : none;
}
int Engine::Session::status_code() const { return st_ ? st_->code : 500; }
int Engine::Session::queue_position() const {
    if (!st_ || !st_->e || st_->holds_card || !st_->ticket) return 0;
    Impl& im = *st_->e->impl_;
    std::lock_guard<std::mutex> lk(im.qmx);
    int pos = 1;
    for (uint64_t t : im.waiters) { if (t == st_->ticket) return pos; ++pos; }
    return 0;
}

Engine::Session Engine::session(CancelToken* tok) { return Session(*this, tok, /*blocking=*/true); }
Engine::Session Engine::try_session() { return Session(*this, nullptr, /*blocking=*/false); }

// ---------------------------------------------------------------------------
Engine::Engine(EngineConfig cfg) : impl_(new Impl(std::move(cfg))) {
    impl_->owner = this;
    // Idle unload, the fleet's watchdog shape: 1 s tick, gated on in_flight == 0 (QUEUED counts,
    // so it cannot evict under a request that has not reached the card yet), and it takes the card
    // NON-BLOCKINGLY so it can never delay an arriving request.
    if (impl_->cfg.idle_unload_seconds > 0) {
        impl_->watchdog = std::thread([this] {
            Impl& im = *impl_;
            while (!im.watchdog_stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                if (im.watchdog_stop.load()) break;
                if (!im.motion_loaded.load()) continue;
                if (now_s() - im.last_done_s.load() < (double)im.cfg.idle_unload_seconds) continue;
                Session s = try_session();
                if (!s.ok()) continue;                       // a request owns the card; try later
                if (im.cfg.verbose)
                    std::fprintf(stderr, "avatar: idle %ds — unloading resident weights\n",
                                 im.cfg.idle_unload_seconds);
                release_gpu();
            }
        });
    }
}

Engine::~Engine() {
    impl_->stop_watchdog();
}

const EngineConfig& Engine::config() const { return impl_->cfg; }
size_t Engine::peak_vram_used_mib() const { return impl_->vram.peak(); }
size_t Engine::baseline_vram_used_mib() const { return impl_->vram.baseline(); }
void Engine::reset_vram_peak() { impl_->vram.reset(); }
void Engine::release_gpu() {
    if (impl_->motion) impl_->motion->unload();
    impl_->motion_loaded.store(false);
}
void Engine::drain()   { impl_->draining.store(true);  impl_->qcv.notify_all(); }
void Engine::undrain() { impl_->draining.store(false); impl_->qcv.notify_all(); }

Health Engine::health() const {
    const Impl& im = *impl_;
    Health h;
    // NO LOCK. /health has to answer while a 450 s render holds everything, and it has to answer
    // while nothing is loaded at all. Every field below is an atomic or a constant.
    h.busy      = im.card_busy.load();            // benign race: it is a status line, not a decision
    h.queued    = im.queued.load();
    h.in_flight = (h.busy ? 1 : 0) + h.queued;
    h.draining  = im.draining.load();
    h.waiting_for_gpu_lock = im.waiting_lock.load();
    h.loaded    = im.motion_loaded.load();
    h.loaded_model = h.loaded ? im.cfg.hymotion_gguf : std::string();
    h.renders_done = im.renders_done.load();
    h.last_render_sec = im.last_render_sec.load();
    h.uptime_sec = now_s() - im.started_s;
    h.idle_sec = h.in_flight ? 0.0 : now_s() - im.last_done_s.load();
    h.idle_unload_seconds = im.cfg.idle_unload_seconds;
    h.keep_resident = im.cfg.keep_motion_resident;
    h.fatal = im.fatal;
    h.status = im.fatal.empty() ? "ok" : "degraded";
    return h;
}

GpuStatus Engine::gpu_status() const {
    GpuStatus g;
    g.loaded = impl_->motion_loaded.load();
    if (g.loaded) g.gpu = "cuda:0";
    g.mem_free_mb = impl_->vram.free_mib();
    if (g.mem_free_mb == 0) {
        size_t f = 0, t = 0;
        g.mem_free_mb = (cudaMemGetInfo(&f, &t) == cudaSuccess) ? (long)(f >> 20) : -1;
    }
    return g;
}

// ---------------------------------------------------------------------------
// Stage 1 — image -> rigged GLB
// ---------------------------------------------------------------------------
bool Engine::build_rig(const RigRequest& req, RigResult& out, std::string& err) {
    if (req.image.empty() || req.out_glb.empty()) { err = "build_rig needs an image and an out_glb"; return false; }
    Session sess = session();      // no-op if the caller already holds one
    if (!sess.ok()) { err = sess.error(); return false; }
    const EngineConfig& cfg = impl_->cfg;
    mkparent(req.out_glb);
    if (!req.stage_dir.empty()) mkdirs(req.stage_dir);
    if (!req.rig_cache.empty()) mkdirs(req.rig_cache);
    OutputGuard guard(req.out_glb);

    // THE TYPED SEAM. This used to build an argv array and let image_to_rig's CLI parser re-read
    // it; now the same fields are set on the same struct the parser fills, so nothing is
    // stringified and nothing can be lost to a quoting bug.
    i2r::Options o;
    o.model = cfg.geo_model_dir.empty() ? std::string("/home/dbrain/models/3d/geo_f16_native") : cfg.geo_model_dir;
    o.image = req.image;
    o.out   = req.out_glb;
    o.dc_remesh = req.dc_remesh;
    // _set = true is not decoration: it is what `--texsize N` on the command line means, and it is
    // what stops --dc-remesh silently raising these to its own defaults underneath the caller.
    o.texsize = req.texsize;   o.texsize_set = true;
    o.decimate = req.decimate; o.decimate_set = true;
    if (req.resolution > 0) o.geo_resolution = req.resolution;
    if (req.dc_band > 0)    o.dc_band = req.dc_band;
    if (!req.stage_dir.empty())    o.stage_dir = req.stage_dir;
    if (!req.from_refined.empty()) o.from_refined = req.from_refined;
    if (!req.rig_cache.empty())    o.rig_cache = req.rig_cache;
    if (req.rig_retries > 0)       o.rig_retries = req.rig_retries;
    if (req.rig_extra_retries >= 0) o.rig_extra_retries = req.rig_extra_retries;
    if (!cfg.rig_r1w.empty())      o.r1w = cfg.rig_r1w;
    if (!cfg.rig_qwen3_w.empty())  o.qwen3_w = cfg.rig_qwen3_w;
    if (!cfg.rig_skin_vae.empty()) o.skinvae = cfg.rig_skin_vae;
    o.use_cuda = cfg.use_cuda;
    // The escape hatch keeps working, through image_to_rig's OWN parser applied on top — one
    // parser, so an extra_args flag behaves exactly as it does on the command line.
    if (!req.extra_args.empty()) {
        std::vector<std::string> a{"image_to_rig"};
        for (const auto& e : req.extra_args) a.push_back(e);
        std::vector<char*> argv;
        argv.reserve(a.size());
        for (auto& s : a) argv.push_back(const_cast<char*>(s.c_str()));
        const int prc = i2r::parse_args((int)argv.size(), argv.data(), o);
        if (prc >= 0) { err = "build_rig: bad extra_args"; return false; }
    }

    const double t0 = now_s();
    const int rc = i2r::run(o);
    out.seconds = now_s() - t0;
    if (rc == i2r::RC_CANCELLED) { err = "cancelled"; return false; }
    if (rc != 0) { err = "image_to_rig stage failed (rc " + std::to_string(rc) + ")"; return false; }
    out.glb = req.out_glb;
    guard.commit();
    // The draw selector's verdict. It is computed BEFORE the write (a gate result, not a property of
    // the file), so unlike J and named_core below it cannot be recovered by reading the GLB back —
    // it has to come across from the stage. See rig_stage_report.hpp for why that is a last-run
    // global and why one-request-at-a-time makes it well defined.
    {
        const rig::StageReport& sr = image_to_rig_last_rig_report();
        out.gate.draws               = sr.valid ? sr.draws : 0;
        out.gate.accepted_draw       = sr.valid ? sr.accepted_draw : -1;
        out.gate.accepted            = sr.accepted;
        out.gate.skin_ok             = sr.skin_ok;
        out.gate.humanoid_gate_ok    = sr.humanoid_gate_ok;
        out.gate.generic_gate_ok     = sr.generic_gate_ok;
        out.gate.selector_named_core = sr.named_core;
        out.gate.pose_gate_ran       = sr.pose_gate_ran;
        out.gate.pose_gate_pass      = sr.pose_gate_pass;
        out.gate.pose_gate_worst     = sr.pose_gate_worst;
        out.gate.selector_summary    = sr.summary;
    }

    // Read the delivered rig back rather than scraping the stage's stdout: the artifact is the
    // contract, and a log line is not.
    mret::Rig rig;
    std::string e2;
    if (mret::load_rig(req.out_glb.c_str(), rig, e2)) {
        out.joints = (int)rig.joints.size();
        std::vector<std::string> want(rig::kMixamoNames, rig::kMixamoNames + rig::SMPL_N);
        for (int n : rig.joints)
            for (const auto& w : want)
                if (rig.orig[(size_t)n] == w) { out.named_core++; break; }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stage 2 — prompt -> motion clip
// ---------------------------------------------------------------------------
bool Engine::generate_motion(const MotionRequest& req, MotionResult& out, std::string& err) {
    Session sess = session();
    if (!sess.ok()) { err = sess.error(); return false; }
    const EngineConfig& cfg = impl_->cfg;
    HYMotion::Engine* eng = impl_->motion_engine();
    HYMotion::GenRequest g;
    g.prompt = req.prompt;
    g.uncond = req.uncond;
    g.cfg = req.cfg;
    g.seconds = req.seconds;
    g.steps = req.steps;
    g.seed = req.seed;
    g.name = req.name;

    HYMotion::DecodedMotion m;
    HYMotion::GenStats st;
    OutputGuard guard(req.out_json);
    const double t0 = now_s();
    try {
        if (!eng->generate(g, m, &st, err)) return false;
    } catch (const cancelhook::Cancelled&) {
        // The sampler threw mid-solve. The engine's resident weights are still valid, but an
        // abandoned request should not keep 2.1 GB of the 3060 warm for a caller that left.
        release_gpu();
        err = "cancelled";
        return false;
    }
    impl_->motion_loaded.store(true);
    out.seconds = now_s() - t0;
    out.frames = (int)m.frames;
    out.fps = (double)m.fps;
    out.dit_was_resident = st.dit_was_resident;
    out.load_s = st.load_s;
    out.encode_s = st.encode_s;
    out.solve_s = st.solve_s;
    out.clip_json = HYMotion::to_clip_json(m, req.name);
    if (!req.out_json.empty()) {
        mkparent(req.out_json);
        FILE* f = std::fopen(req.out_json.c_str(), "w");
        if (!f) { err = "cannot write " + req.out_json; return false; }
        std::fwrite(out.clip_json.data(), 1, out.clip_json.size(), f);
        std::fclose(f);
        out.path = req.out_json;
    }
    guard.commit();
    if (!cfg.keep_motion_resident) { eng->unload(); impl_->motion_loaded.store(false); }
    return true;
}

// ---------------------------------------------------------------------------
// Stage 3 — rigged GLB + clips -> animated GLB
// ---------------------------------------------------------------------------
bool Engine::animate(const AnimateRequest& req, AnimateResult& out, std::string& err) {
    using namespace mret;
    const double t0 = now_s();
    if (req.rig_glb.empty() || req.out_glb.empty()) { err = "animate needs a rig_glb and an out_glb"; return false; }
    // No Session here: this stage is CPU-only (retarget + bake). Taking the card for it would
    // block a queued render behind arithmetic. It is still cancellable, at its own boundaries.
    OutputGuard guard(req.out_glb);

    // The SMPL rest pose is BAKED (smpl_rest_joints.hpp): no 167 MB npz on the delivery path.
    const std::vector<V3> src_rest0 = smpl_rest_joints_baked_v3();
    Prep p;
    if (!prepare(req.rig_glb, src_rest0, p, err)) return false;
    out.mapped = p.map.mapped();
    out.map_kind = p.map.which;
    out.swap = p.swap.swap;
    out.rest_arm_deg = p.rest_arm_mean;
    out.rest_leg_deg = p.rest_leg_mean;
    out.rest_spine_deg = p.rest_spine_mean;

    const Variant* v = find_variant(req.variant);
    if (!v) { err = "unknown variant '" + req.variant + "'"; return false; }

    std::vector<std::string> texts = req.clip_json;
    // Animation naming: a clip that arrives as TEXT has only its own `name` field, but a clip that
    // arrives as a PATH is named by its filename stem — because that is what motion_retarget's
    // single-shot mode does, and the eye pages and manifests key off those names. Two shipped
    // paths that name the same animation differently is a trap, so they agree here.
    std::vector<std::string> names(texts.size());
    for (const auto& path : req.clip_path) {
        std::vector<uint8_t> buf;
        if (!read_file(path.c_str(), buf)) { err = "cannot open " + path; return false; }
        texts.emplace_back((const char*)buf.data(), buf.size());
        std::string base = path.substr(path.find_last_of('/') + 1);
        const size_t dot = base.find_last_of('.');
        names.push_back(dot == std::string::npos ? base : base.substr(0, dot));
    }
    if (texts.empty()) { err = "animate needs at least one clip"; return false; }

    std::vector<BakedClip> baked;
    for (size_t i = 0; i < texts.size(); i++) {
        if (cancelhook::cancelled()) { err = "cancelled"; return false; }
        HymoClip hc;
        if (!load_hymo_string(texts[i].data(), texts[i].size(), hc, err)) return false;
        if (v->rot && hc.local.empty()) { err = "variant '" + req.variant + "' needs a rotation-carrying source"; return false; }
        // The legacy float32 position bridge is still computed: the positional variants need it,
        // and the rotation-native ones do not use it. hymo_positions() reproduces hymo_to_npy.py's
        // float32 round deliberately — it is the bridge's behaviour, not a bug to fix here.
        std::vector<V3> pos = hymo_positions(hc, src_rest0);
        std::vector<Q4> rot = smpl_global_rots(hc);
        const int T = (int)(pos.size() / 22);
        const bool swap = (req.variant == "nofix") ? !p.swap.swap : p.swap.swap;
        std::vector<V3> mm = pos, rest = src_rest0;
        std::vector<Q4> mr = rot;
        if (swap) {
            // relabel motion AND source rest identically — lr_swap indexes the JOINT axis, so it
            // works unchanged on rotations.
            lr_swap_inplace(mm, T);
            lr_swap_inplace(rest, 1);
            if (!mr.empty()) lr_swap_inplace(mr, T);
        }
        RestAlign ra;
        ra.a[GRP_ARM] = v->arm_alpha;
        Clip c;
        if (!retarget_delta(p.rig, p.map, mm, rest, hc.fps, false, ra, v->rot ? &mr : nullptr, c, err)) return false;
        smooth_quats(c, req.smooth);
        BakedClip bc;
        bc.name = !names[i].empty() ? names[i] : (hc.name.empty() ? ("clip" + std::to_string(i)) : hc.name);
        bc.bones = c.bones; bc.fps = hc.fps; bc.T = c.T; bc.N = c.N; bc.quats = c.quats;
        baked.push_back(std::move(bc));
    }

    if (cancelhook::cancelled()) { err = "cancelled"; return false; }
    mkparent(req.out_glb);
    const int n = bake_animations(p.rig, baked, req.out_glb.c_str(), err);
    if (n < 0) return false;
    out.animations = n;
    out.glb = req.out_glb;

    // The skeleton-exercise clip goes into the SAME delivery: it is an eye-test track alongside the
    // prompted motion, not a separate file to keep in sync. (bake_exercise reads the whole GLB
    // before it writes, so in-place is safe.)
    if (req.exercise) {
        rigex::Options eo;
        rigex::Result er;
        std::string e2;
        if (rigex::bake_exercise(req.out_glb.c_str(), req.out_glb.c_str(), eo, er, e2)) {
            out.exercise_joints = er.joints;
            out.animations++;
        } else if (impl_->cfg.verbose) {
            std::fprintf(stderr, "avatar: exercise clip skipped (%s)\n", e2.c_str());
        }
    }
    guard.commit();
    out.seconds = now_s() - t0;
    return true;
}

// ---------------------------------------------------------------------------
// The one call
// ---------------------------------------------------------------------------
bool Engine::run(const Request& req, Result& out, std::string& err) {
    const double t0 = now_s();
    // ONE admission for the whole chain: the stages nest inside it and neither re-queue nor
    // re-take the flock. See Session.
    Session sess = session(req.cancel);
    if (!sess.ok()) { err = sess.error(); out.cancelled = (sess.status_code() == 499); return false; }

    // ORDER: motion FIRST, then the rig. Not cosmetic — the rig stage peaks near 10.7 GB of the
    // 3060's 11.9 GB, so the ~2.1 GB motion DiT must not be resident across it. Doing motion first
    // and releasing before the rig means the card only ever holds one of the two.
    if (!generate_motion(req.motion, out.motion, err)) { out.cancelled = cancelhook::cancelled(); return false; }
    if (impl_->motion) { impl_->motion->unload(); impl_->motion_loaded.store(false); }

    if (!build_rig(req.rig, out.rig, err)) { out.cancelled = cancelhook::cancelled(); return false; }

    AnimateRequest ar;
    ar.rig_glb = out.rig.glb;
    ar.clip_json.push_back(out.motion.clip_json);
    ar.variant = req.variant;
    ar.smooth = req.smooth;
    ar.exercise = req.exercise;
    ar.out_glb = req.out_glb;
    if (!animate(ar, out.anim, err)) { out.cancelled = cancelhook::cancelled(); return false; }

    out.total_s = now_s() - t0;
    out.peak_vram_used_mib = impl_->vram.peak();
    out.baseline_vram_used_mib = impl_->vram.baseline();
    return true;
}

}  // namespace avatar
