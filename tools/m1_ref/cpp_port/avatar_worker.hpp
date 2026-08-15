// avatar_worker.hpp — worker isolation for avatar_server, and the NVML reading that makes it
// possible. Header-only, included by avatar_server.cpp and by nothing else.
//
// ============================================================================================
// WHY THIS EXISTS: ggml POOLS FREED VRAM, SO "UNLOAD" NEVER GAVE THE CARD BACK
// ============================================================================================
// Measured 2026-08-14 on the 3060, one container, one job:
//     fresh avatar_server process ......... 122 MiB   (the CUDA primary context, made at startup)
//     after ONE motion or rig job ........ 1558 MiB   with /health reporting loaded: false
//     POST /v1/admin/unload -d '{}' ...... 1558 MiB   {"status":"idle","unloaded":false}
//     docker restart ...................... 122 MiB   <- the only thing that ever worked
// Engine::release_gpu() drops the motion DiT and nothing else; every block ggml's CUDA allocator
// ever handed out stays pooled for reuse. That retained pool pushed the service under its own
// LONGCAT_RIG_MIN_FREE_MIB floor and made it refuse its own next request — one job per container
// start. Lowering the floor 11000 -> 8000 hid it; it did not fix it.
//
// PROCESS EXIT IS THE ONLY COMPLETE, PORTABLE VRAM RELEASE. So the same shape the rest of this
// fork's fleet already uses (examples/server/worker_supervisor.{h,cpp}, the sd-server GPU gate):
//
//     PARENT  (default)  owns the public listener, the job manager, admission, the VRAM floor and
//                        every route. It NEVER constructs avatar::Engine, so it never creates a
//                        CUDA context and holds 0 MiB on the card.
//     CHILD   (--worker) is today's avatar_server, unchanged, on a private loopback port. It owns
//                        the Engine, the CUDA context, the ggml pools and the GPU flock.
//
// An unload is a SIGKILL of the child. The kernel then reclaims the context, the pools and the
// flock together, and there is nothing left to leak.
//
// 🔴 THE PARENT MUST NEVER CALL cudaMemGetInfo. Engine::gpu_status() does, and that call ALONE
// creates the CUDA primary context (~122 MiB) — which would defeat the entire point. The parent
// reads free VRAM through NVML instead: no CUDA context, and it is the same number nvidia-smi
// prints. See nvml_free_mib() for the trap that lookup has.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "httplib.h"

namespace avsup {

// ---------------------------------------------------------------------------
// NVML — "how much of the card is free", asked WITHOUT creating a CUDA context
// ---------------------------------------------------------------------------
// 🔴 NVML DOES NOT HONOUR CUDA_VISIBLE_DEVICES. An index-based lookup here would read whichever
// card the driver enumerates first, which on this box is not necessarily ours — and reading the
// 5060 Ti's free VRAM to decide whether a 3060 job fits is a wrong answer that looks right. So
// every lookup here is BY UUID. A numeric CUDA_VISIBLE_DEVICES is honoured as an NVML index only as
// a fallback for the default card, and said so out loud, because CUDA's device order and NVML's
// agree only under CUDA_DEVICE_ORDER=PCI_BUS_ID.
//
// 🔴 SINCE PLACEMENT, THE CARD IS A PARAMETER, NOT A SINGLETON. The worker can be spawned on any
// card in `LONGCAT_RIG_GPUS`, so "free VRAM" is only a question you can ask ABOUT A CARD — asking
// it about "the pinned card" is what the single-device version of this file did, and it is exactly
// the reading that would admit a job onto a card it does not fit. Handles are cached per UUID.
//
// dlopen rather than -lnvidia-ml: libnvidia-ml.so.1 is injected by the NVIDIA container runtime and
// is absent from the build image entirely. A link-time dependency would make the binary refuse to
// start anywhere the driver is not mounted; a dlopen degrades to "cannot ask", which is a state the
// callers already handle (-1 => admit anyway, never invent a refusal).
namespace detail {

struct NvmlMemory {
    unsigned long long total = 0, free = 0, used = 0;
};

struct NvmlState {
    void* lib = nullptr;
    int (*get_memory)(void*, NvmlMemory*) = nullptr;
    int (*by_uuid)(const char*, void**) = nullptr;
    int (*by_index)(unsigned, void**) = nullptr;
    void* fallback_device = nullptr;   // the container's own CUDA_VISIBLE_DEVICES, resolved once
    bool ready = false;                // the LIBRARY is usable; a given UUID may still not resolve
    std::string note = "not initialised";
    std::mutex mx;                     // guards `cache` only
    std::vector<std::pair<std::string, void*>> cache;
};

inline NvmlState& nvml_state() {
    static NvmlState s;
    static std::once_flag once;
    std::call_once(once, [] {
        s.lib = ::dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!s.lib) s.lib = ::dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
        if (!s.lib) { s.note = "libnvidia-ml.so.1 not loadable (no NVIDIA driver mounted?)"; return; }
        auto init    = (int (*)())::dlsym(s.lib, "nvmlInit_v2");
        s.by_uuid    = (int (*)(const char*, void**))::dlsym(s.lib, "nvmlDeviceGetHandleByUUID");
        s.by_index   = (int (*)(unsigned, void**))::dlsym(s.lib, "nvmlDeviceGetHandleByIndex_v2");
        s.get_memory = (int (*)(void*, NvmlMemory*))::dlsym(s.lib, "nvmlDeviceGetMemoryInfo");
        if (!init || !s.get_memory || (!s.by_uuid && !s.by_index)) {
            s.note = "libnvidia-ml.so.1 is missing the symbols we need";
            return;
        }
        if (init() != 0) { s.note = "nvmlInit_v2 failed"; return; }
        s.ready = true;

        const char* env = std::getenv("CUDA_VISIBLE_DEVICES");
        const std::string pin = (env && *env) ? std::string(env) : std::string();
        // Only the FIRST entry: a comma list means CUDA device 0 is the first one, and device 0 is
        // the card every stage in this pipeline uses.
        const std::string first = pin.substr(0, pin.find(','));
        if (s.by_uuid && (first.rfind("GPU-", 0) == 0 || first.rfind("MIG-", 0) == 0)) {
            if (s.by_uuid(first.c_str(), &s.fallback_device) == 0) { s.note = "by UUID " + first; return; }
            s.note = "nvmlDeviceGetHandleByUUID(" + first + ") failed";
            return;
        }
        if (!s.by_index) { s.note = "no index lookup available"; return; }
        unsigned idx = 0;
        if (!first.empty() && first.find_first_not_of("0123456789") == std::string::npos)
            idx = (unsigned)std::atoi(first.c_str());
        if (s.by_index(idx, &s.fallback_device) == 0) {
            s.note = "by INDEX " + std::to_string(idx) +
                     " — CUDA_VISIBLE_DEVICES is not a UUID, so this may be the WRONG CARD unless "
                     "CUDA_DEVICE_ORDER=PCI_BUS_ID";
            return;
        }
        s.note = "nvmlDeviceGetHandleByIndex_v2(" + std::to_string(idx) + ") failed";
    });
    return s;
}

// The NVML handle for a UUID, or the container-default device when `uuid` is empty. nullptr = the
// card could not be resolved, which callers must treat as "cannot ask" and never as a refusal.
inline void* nvml_device(const std::string& uuid) {
    NvmlState& s = nvml_state();
    if (!s.ready) return nullptr;
    if (uuid.empty()) return s.fallback_device;
    std::lock_guard<std::mutex> lk(s.mx);
    for (const auto& [k, d] : s.cache)
        if (k == uuid) return d;
    void* dev = nullptr;
    // A UUID that does not resolve is cached as nullptr on purpose: it is a config error (a card
    // this host does not have, or one the container was not given), and re-asking NVML about it on
    // every admission would turn a typo into a per-request syscall.
    if (s.by_uuid && s.by_uuid(uuid.c_str(), &dev) != 0) dev = nullptr;
    s.cache.emplace_back(uuid, dev);
    return dev;
}

// Free VRAM on `uuid` (empty = the container default), in MiB. -1 = could not ask, which every
// caller must treat as "no opinion" and NOT as a refusal.
inline long nvml_free_mib(const std::string& uuid) {
    NvmlState& s = nvml_state();
    void* dev = nvml_device(uuid);
    if (!dev) return -1;
    NvmlMemory m;
    if (s.get_memory(dev, &m) != 0) return -1;
    return (long)(m.free >> 20);
}

inline long nvml_total_mib(const std::string& uuid) {
    NvmlState& s = nvml_state();
    void* dev = nvml_device(uuid);
    if (!dev) return -1;
    NvmlMemory m;
    if (s.get_memory(dev, &m) != 0) return -1;
    return (long)(m.total >> 20);
}

}  // namespace detail

// Free / total VRAM on a named card, MiB. -1 = could not ask.
inline long nvml_free_mib(const std::string& uuid) { return detail::nvml_free_mib(uuid); }
inline long nvml_total_mib(const std::string& uuid) { return detail::nvml_total_mib(uuid); }

// The same, for the container's own CUDA_VISIBLE_DEVICES. Kept because /v1/gpu/status and the boot
// preflight report the container's default card whether or not placement is in use.
inline long nvml_free_mib() { return detail::nvml_free_mib(std::string()); }
inline long nvml_total_mib() { return detail::nvml_total_mib(std::string()); }

// True if NVML can actually answer for this card — i.e. the UUID names a GPU this host has. The
// admission path uses it to reject a bad `gpu` field with a 400 instead of silently placing the
// worker on the default card, which is the failure mode that makes a scheduler untrustworthy.
inline bool nvml_knows(const std::string& uuid) { return detail::nvml_device(uuid) != nullptr; }

// How the default device was resolved (or why it was not), for /v1/gpu/status and the boot log.
inline const std::string& nvml_note() { return detail::nvml_state().note; }

// ---------------------------------------------------------------------------
// Worker — one lazily-spawned CUDA-owning child of this same binary
// ---------------------------------------------------------------------------
// Modelled on WorkerSupervisor (examples/server/worker_supervisor.cpp) for the mechanics —
// loopback port reservation, readiness wait, PDEATHSIG, SIGKILL-and-reap, lease accounting — and
// deliberately NOT reusing that class, which is welded to sd-server's model-variant switching.
// avatar_server has one child, one configuration, and no variant to switch between.
class Worker {
public:
    // A lease keeps the child alive for as long as it is held. Nothing kills a worker with an
    // outstanding lease — not idle-unload, not the demand-driven reclaim, not a plain
    // /v1/admin/unload (that one answers 409 instead).
    class Lease {
    public:
        Lease(Lease&& o) noexcept : w_(o.w_), port_(o.port_), err_(std::move(o.err_)) { o.w_ = nullptr; o.port_ = 0; }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() { if (w_) w_->release_lease(); }
        bool ok() const { return port_ > 0; }
        int  port() const { return port_; }
        const std::string& error() const { return err_; }

    private:
        friend class Worker;
        Lease(Worker* w, int port, std::string err) : w_(w), port_(port), err_(std::move(err)) {}
        Worker* w_ = nullptr;
        int port_ = 0;
        std::string err_;
    };

    Worker(std::string exe, std::vector<std::string> args, std::function<void(const char*)> log)
        : exe_(std::move(exe)), args_(std::move(args)), log_(std::move(log)) {
        last_active_.store(mono());
        spawner_ = std::thread([this] { spawner_loop(); });
    }
    ~Worker() {
        { std::lock_guard<std::mutex> lk(mx_); (void)kill_locked(); }
        {
            std::lock_guard<std::mutex> lk(smx_);
            spawn_stop_ = true;
        }
        scv_.notify_all();
        if (spawner_.joinable()) spawner_.join();
    }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // The card an un-targeted request lands on (WORKER_DEFAULT_GPU). Empty = whatever the
    // container's CUDA_VISIBLE_DEVICES already says, which is the pre-placement behaviour and the
    // reason this whole change is backward compatible: no default and no `gpu` field ⇒ no setenv
    // ⇒ the child inherits the container pin exactly as it always did.
    void set_default_gpu(std::string uuid) {
        std::lock_guard<std::mutex> lk(mx_);
        default_gpu_ = std::move(uuid);
    }
    std::string default_gpu() const {
        std::lock_guard<std::mutex> lk(mx_);
        return default_gpu_;
    }

    // 🔴 THE GPU FLOCK IS PER CARD, AND THE CONFIGURED ONE BELONGS TO EXACTLY ONE OF THEM. `--lock`
    // names a real file that the HOST-SIDE shell drivers also take (its default is literally
    // `.3060-image-to-rig.lock`), so it arbitrates one physical GPU. Once the worker can be placed,
    // taking that same lock for a render on the OTHER card would be wrong twice over: it would block
    // a host driver waiting for the card we are not using — for up to the 7200 s lock timeout, which
    // looks exactly like a hang — while not excluding anything at all on the card we ARE using.
    //
    // So: `home_card` keeps the configured path (that is the card the lock was always about), and
    // every other card gets its own sibling lock file. `base` empty = no flock, unchanged.
    void set_lock_policy(std::string base, std::string home_card) {
        std::lock_guard<std::mutex> lk(mx_);
        lock_base_ = std::move(base);
        lock_home_card_ = std::move(home_card);
    }

    // The card the resident worker is actually on. Empty when there is no worker — never a guess,
    // because "where would it go" and "where is it" are different questions and only the second one
    // may be used to decide whether a relocation is needed.
    std::string worker_gpu() const {
        std::lock_guard<std::mutex> lk(gmx_);
        return pid_a_.load() > 0 ? worker_gpu_ : std::string();
    }

    // Spawn the child if it is not already up, and hold it for the caller. Blocking (the readiness
    // wait), but only ever for the ~seconds a cold start costs: constructing avatar::Engine is
    // lazy, so this is process exec + the CUDA primary context, not a weight load.
    //
    // `want_gpu` is the placement target: empty means the default. A resident worker on the WRONG
    // card is killed and respawned on the right one — an in-flight job never migrates, so this can
    // only happen while nothing holds a lease (see ensure_locked).
    Lease lease(const std::string& want_gpu = std::string()) {
        std::unique_lock<std::mutex> lk(mx_);
        const std::string target = want_gpu.empty() ? default_gpu_ : want_gpu;
        // 🔴 A RELOCATION WAITS FOR THE OTHER CARD'S RENDER, IT DOES NOT REFUSE IT. Relocating
        // needs an idle worker, and more than one job can hold a lease at once — the parent runs a
        // thread per admitted job and the FIFO that actually serialises them lives in the child. So
        // a second submit naming the other card arrives while the first is still running, which is
        // ordinary, not an error: without this wait it would fail on a queue it should simply have
        // joined. `wait_for` drops mx_ while it waits, so /health, poll() and the release itself
        // all keep working.
        //
        // The bound matches the flock's own 7200 s timeout — the other thing in this engine that
        // waits for a card — so a stuck render surfaces as one message, not two different ones.
        const bool ready = relocate_cv_.wait_for(lk, std::chrono::seconds(7200), [&] {
            return pid_ <= 0 || worker_gpu_ == target || leases_.load() == 0;
        });
        if (!ready) {
            return Lease(nullptr, 0,
                         "waited 7200 s for the GPU worker to go idle so it could be moved to " +
                             (target.empty() ? std::string("the container default card") : target) +
                             ", and it never did");
        }
        std::string err;
        if (!ensure_locked(err, want_gpu)) return Lease(nullptr, 0, err);
        leases_.fetch_add(1);
        return Lease(this, port_, std::string());
    }

    // 🔴 THE STATUS READS TAKE NO LOCK, and that is load-bearing rather than tidy. lease() holds
    // mx_ across a worker cold start, and /health is polled by the container healthcheck with a 5 s
    // timeout — a /health that blocked behind a spawn would report a perfectly good service as
    // unhealthy for as long as the start took. So pid/port are mirrored into atomics under the
    // mutex and read out of it. poll() is how a dead child still gets reaped promptly.
    bool alive() const { return pid_a_.load() > 0; }
    int  pid()   const { return pid_a_.load(); }
    int  port()  const { return pid_a_.load() > 0 ? port_a_.load() : 0; }
    int  leases() const { return leases_.load(); }

    // Reap a child that exited on its own. Best effort: skipped entirely if the worker is busy
    // being spawned or killed, because the caller is a watchdog and will be back in half a second.
    void poll() {
        std::unique_lock<std::mutex> lk(mx_, std::try_to_lock);
        if (lk.owns_lock()) reap_locked();
    }
    long spawns() const { return spawns_.load(); }

    // SIGKILL + reap. TRUE means we can PROVE no CUDA-owning worker of ours remains — which is the
    // only claim on which a caller may admit competing GPU work. It is also true when there was
    // nothing to kill, for the same reason: the claim is about the end state, not about the work.
    // FALSE means the kill or the wait failed and the context may still be alive.
    bool kill_worker() {
        std::lock_guard<std::mutex> lk(mx_);
        return kill_locked();
    }

    // Kill ONLY if nothing holds a lease. Returns true if a worker was actually killed. This is
    // the demand-driven reclaim: an idle child still shows its pooled VRAM as used, so a request
    // that would otherwise be refused for VRAM gets one chance to prove the shortfall is ours.
    bool kill_if_idle() {
        std::lock_guard<std::mutex> lk(mx_);
        reap_locked();
        if (pid_ <= 0 || leases_.load() > 0) return false;
        const int was = pid_;
        if (!kill_locked()) return false;
        return was > 0;
    }

    // Seconds since the last lease was released (0 while any lease is held).
    double idle_seconds() const {
        if (leases_.load() > 0) return 0.0;
        return mono() - last_active_.load();
    }

private:
    friend class Lease;

    static double mono() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
    }
    void say(const std::string& s) { if (log_) log_(s.c_str()); }

    // No mx_ here, deliberately: a lease is released from the job thread and taking the supervisor
    // lock would make every completion queue behind a cold start. The notify is what lets a lease()
    // blocked on a relocation notice that the card just went idle.
    void release_lease() {
        last_active_.store(mono());
        leases_.fetch_sub(1);
        relocate_cv_.notify_all();
    }

    // Called under mx_ whenever the child is gone. worker_gpu_ is cleared with it: "which card is
    // the worker on" must have no answer when there is no worker, or the next ensure_locked() would
    // compare its target against a dead process's card and skip a spawn it needs.
    void clear_locked() {
        pid_ = -1;
        port_ = 0;
        { std::lock_guard<std::mutex> glk(gmx_); worker_gpu_.clear(); }
        publish_locked();
    }
    void publish_locked() { pid_a_.store(pid_); port_a_.store(port_); }

    bool reap_locked() {
        if (pid_ <= 0) return false;
        int status = 0;
        const pid_t waited = ::waitpid(pid_, &status, WNOHANG);
        if (waited == 0) return false;
        if (waited == pid_ || (waited < 0 && errno == ECHILD)) {
            // SAY HOW, not just that. A worker that vanishes looks the same from here whether it
            // OOMed, hit a CUDA fault, or was signalled by something we did to ourselves — and the
            // first bug this supervisor had was exactly the last of those (see spawner_loop).
            std::string how = "reason unknown";
            if (WIFEXITED(status))        how = "exit " + std::to_string(WEXITSTATUS(status));
            else if (WIFSIGNALED(status)) how = "signal " + std::to_string(WTERMSIG(status));
            say("worker pid " + std::to_string(pid_) + " exited on its own (" + how + ")");
            clear_locked();
            return true;
        }
        return false;
    }

    bool kill_locked() {
        reap_locked();
        if (pid_ <= 0) return true;
        const int pid = pid_;
        // SIGKILL is intentional and is the whole mechanism: the child owns CUDA's primary context
        // and every ggml backend pool, so process exit is the only complete release. A polite
        // SIGTERM would let it try to unwind a CUDA context that a render may be mid-kernel in.
        if (::kill(pid, SIGKILL) != 0 && errno != ESRCH) return false;
        int status = 0;
        pid_t waited = -1;
        do { waited = ::waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        if (waited != pid && !(waited < 0 && (errno == ECHILD || errno == ESRCH))) return false;
        say("worker pid " + std::to_string(pid) + " killed — CUDA context, ggml pools and the GPU "
            "flock are all released with it");
        clear_locked();
        last_active_.store(mono());
        return true;
    }

    static int reserve_loopback_port() {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return 0;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) { ::close(fd); return 0; }
        socklen_t len = sizeof(a);
        if (::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) != 0) { ::close(fd); return 0; }
        const int p = ntohs(a.sin_port);
        ::close(fd);
        return p;
    }

    bool wait_until_ready_locked(std::string& err) {
        httplib::Client c("127.0.0.1", port_);
        c.set_connection_timeout(1, 0);
        c.set_read_timeout(2, 0);
        for (int i = 0; i < 1200; i++) {          // 120 s: a cold CUDA context, not a weight load
            int status = 0;
            if (::waitpid(pid_, &status, WNOHANG) == pid_) {
                pid_ = -1;
                publish_locked();
                if (WIFEXITED(status))        err = "worker exited during startup (exit " + std::to_string(WEXITSTATUS(status)) + ")";
                else if (WIFSIGNALED(status)) err = "worker died during startup (signal " + std::to_string(WTERMSIG(status)) + ")";
                else                          err = "worker exited during startup";
                return false;
            }
            const auto r = c.Get("/health");
            if (r && r->status >= 200 && r->status < 300) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        err = "worker did not answer /health within 120 s";
        return false;
    }

    bool ensure_locked(std::string& err, const std::string& want_gpu) {
        reap_locked();
        const std::string target = want_gpu.empty() ? default_gpu_ : want_gpu;
        if (pid_ > 0) {
            // 🔴 THE REUSE CHECK IS (alive AND same card), never just (alive). Reusing a worker
            // that is on the other GPU is how a placement decision becomes a lie: the gate reserved
            // VRAM on one card and the render would run on another, so both cards' accounting is
            // wrong and the one actually used was never checked against the floor.
            if (worker_gpu_ == target) return true;
            // An in-flight render does NOT migrate. Nothing here may touch a worker under a lease —
            // that is the same rule kill_if_idle() follows, and for the same reason.
            if (leases_.load() > 0) {
                err = "the GPU worker is busy on " + (worker_gpu_.empty() ? std::string("its default card")
                                                                          : worker_gpu_) +
                      " and a render cannot migrate mid-flight; retry once it finishes";
                return false;
            }
            say("relocating the CUDA worker from " +
                (worker_gpu_.empty() ? std::string("the container default") : worker_gpu_) + " to " +
                (target.empty() ? std::string("the container default") : target));
            if (!kill_locked()) { err = "could not kill the resident worker to relocate it"; return false; }
        }
        port_ = reserve_loopback_port();
        if (port_ == 0) { err = "could not reserve a worker loopback port"; return false; }

        // Our own argv, then the overrides. avatar_server's parser is last-wins, so appending is
        // what makes --port/--host/--idle-unload ours rather than the entrypoint's. Everything
        // else — the weight paths, --lock, --min-free-mib, --jobs-dir, --rig-retries — reaches the
        // child unchanged, which is the point: the child IS today's server.
        std::vector<std::string> args = args_;
        args.push_back("--worker");
        args.push_back("--host");
        args.push_back("127.0.0.1");
        args.push_back("--port");
        args.push_back(std::to_string(port_));
        // 🔴 The PARENT owns idle-unload now, because "unload" means "kill the child". A child that
        // also ran the Engine's own watchdog would drop the motion DiT on a timer the parent cannot
        // see, and give back 2.1 GB where the parent is about to give back all of it.
        args.push_back("--idle-unload");
        args.push_back("0");
        // 🔴 AND THE PARENT OWNS THE VRAM FLOOR, for the same reason it owns the unload: a child
        // enforcing it would re-create the exact bug this change exists to fix. After one job the
        // child's own pooled blocks read as USED to cudaMemGetInfo, so its second rig would refuse
        // itself on memory that is already its own and already reusable — one job per worker
        // instead of one job per container. The parent reads NVML, and when the number is short it
        // can KILL THE IDLE WORKER AND RE-MEASURE, which is the only way anyone on this box can
        // tell "a co-tenant took the card" from "we are still holding our own pool". It checks at
        // submit and again immediately before dispatch; see vram_ok() and run_proxy().
        args.push_back("--min-free-mib");
        args.push_back("0");
        // Last-wins parsing is what lets this override the entrypoint's own --lock. Only when the
        // worker is going somewhere other than the card the configured lock belongs to.
        if (!lock_base_.empty() && !target.empty() && target != lock_home_card_) {
            // A short, stable suffix: the UUID tail is unique across the cards in one box and does
            // not turn the path into something no human can read in an `ls`.
            const std::string tail = target.size() > 8 ? target.substr(target.size() - 8) : target;
            args.push_back("--lock");
            args.push_back(lock_base_ + "." + tail);
        }

        const pid_t pid = spawn_on_spawner_thread(args, target);
        if (pid < 0) { err = "fork failed"; port_ = 0; return false; }
        pid_ = (int)pid;
        {
            std::lock_guard<std::mutex> glk(gmx_);
            worker_gpu_ = target;
        }
        publish_locked();
        spawns_.fetch_add(1);
        say("spawned CUDA worker pid " + std::to_string(pid_) + " on 127.0.0.1:" +
            std::to_string(port_) +
            (target.empty() ? std::string(" (container default card)") : (" on card " + target)));
        if (!wait_until_ready_locked(err)) { (void)kill_locked(); return false; }
        last_active_.store(mono());
        return true;
    }

    // ---- the spawner thread ------------------------------------------------------------------
    // 🔴 PR_SET_PDEATHSIG IS KEYED TO THE FORKING **THREAD**, NOT THE PROCESS. Linux delivers the
    // parent-death signal when the thread that called fork() exits, and a supervisor is a threaded
    // program — so forking from the per-job thread that happens to need the worker means the worker
    // is SIGKILLed the instant that job's thread returns.
    //
    // MEASURED, on the first build of this change: a motion job completed normally, the parent
    // logged the result, and the very next line was "worker pid 140 exited on its own". Every job
    // therefore paid a cold start, idle-unload had nothing left to unload, and /health reported no
    // worker seconds after a successful render. The symptom looks exactly like a crash and is not
    // one, which is why it is written down here rather than merely fixed.
    //
    // So every fork happens on ONE thread that lives as long as the process does. PDEATHSIG then
    // means what it is there to mean: if the SUPERVISOR dies, the CUDA-owning child dies with it,
    // and never leaves 10 GB of a shared card to an orphan.
    void spawner_loop() {
        std::unique_lock<std::mutex> lk(smx_);
        for (;;) {
            scv_.wait(lk, [this] { return spawn_want_ || spawn_stop_; });
            if (spawn_stop_) return;
            spawn_want_ = false;
            const std::vector<std::string> args = spawn_args_;
            const std::string gpu = spawn_gpu_;
            lk.unlock();
            const pid_t pid = fork_exec(args, gpu);
            lk.lock();
            spawn_pid_ = pid;
            spawn_done_ = true;
            sdone_.notify_all();
        }
    }

    pid_t spawn_on_spawner_thread(const std::vector<std::string>& args, const std::string& gpu) {
        std::unique_lock<std::mutex> lk(smx_);
        spawn_args_ = args;
        spawn_gpu_ = gpu;
        spawn_done_ = false;
        spawn_want_ = true;
        scv_.notify_one();
        sdone_.wait(lk, [this] { return spawn_done_; });
        return spawn_pid_;
    }

    pid_t fork_exec(const std::vector<std::string>& args, const std::string& gpu) {
        const pid_t me = ::getpid();
        const pid_t pid = ::fork();
        if (pid != 0) return pid;
        // A parent that crashes must not leave a CUDA-owning orphan holding 10 GB of a shared card.
        // PID 1 is a normal parent inside a Docker PID namespace, so compare against the pre-fork
        // pid exactly rather than treating 1 as universally orphaned.
        if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) ::_exit(125);
        if (::getppid() != me) ::_exit(126);
        ::setenv("AVATAR_SERVER_WORKER_CHILD", "1", 1);
        // 🔴 THIS LINE IS THE ENTIRE PLACEMENT MECHANISM. The parent is CUDA-free by construction
        // (that is what worker isolation is FOR), so nothing here has read CUDA_VISIBLE_DEVICES
        // yet — the child's first CUDA call does, after execv, and lands on this card. No
        // `main_gpu` plumbing, no ggml change: every stage still calls ggml_backend_cuda_init(0),
        // and device 0 is now whichever card we named.
        //
        // ⚠️ UUID, NEVER AN INDEX. CUDA enumerates fastest-first, which on this box is the OPPOSITE
        // of nvidia-smi's order, so "1" means different cards to different tools. A wrong index
        // here would not fail — it would rig on the other card and report success.
        if (!gpu.empty()) ::setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(exe_.c_str()));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        ::execv(exe_.c_str(), argv.data());
        ::_exit(127);
    }

    std::string exe_;
    std::vector<std::string> args_;
    std::function<void(const char*)> log_;
    mutable std::mutex mx_;
    // Where an un-targeted request goes, and where the resident worker actually is. `worker_gpu_`
    // has its own mutex so worker_gpu() can be read by /health without queueing behind a cold
    // start, which is the same reason pid_/port_ are mirrored into atomics.
    std::string default_gpu_;
    std::string lock_base_, lock_home_card_;
    // Signalled when a lease is released, so a lease() waiting to relocate wakes on the card going
    // idle rather than on a timer.
    std::condition_variable relocate_cv_;
    mutable std::mutex gmx_;
    std::string worker_gpu_;
    std::thread spawner_;
    std::mutex smx_;
    std::condition_variable scv_, sdone_;
    std::vector<std::string> spawn_args_;
    std::string spawn_gpu_;
    pid_t spawn_pid_ = -1;
    bool spawn_want_ = false, spawn_done_ = false, spawn_stop_ = false;
    int pid_ = -1;
    int port_ = 0;
    // Lock-free mirrors of pid_/port_ for the status readers. Written only under mx_.
    std::atomic<int> pid_a_{-1};
    std::atomic<int> port_a_{0};
    std::atomic<int>  leases_{0};
    std::atomic<long> spawns_{0};
    std::atomic<double> last_active_{0};
};

}  // namespace avsup
