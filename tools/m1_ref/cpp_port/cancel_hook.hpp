// cancel_hook.hpp — the one-line cancellation point the long native loops call.
//
// WHY A PROCESS-GLOBAL AND NOT A PARAMETER. The chain that has to become interruptible is
// image_to_rig -> pixal3d_chain -> geometry_e2e::flow_sampler, narrow_band_dc (a .cu TU), and
// rig_beam_generate_batched. Threading a token through those signatures is a wide, merge-hostile
// diff across files two other agents are editing, and every one of those call sites is reached from
// exactly one request because avatar::Engine is single-threaded by contract: one card, one request.
// So the token is armed for the duration of a request and the loops ask a global.
//
// It is deliberately NOT a bool: `arm()` publishes a POINTER to the caller's flag, so the caller
// keeps ownership and a stale arm from a finished request cannot cancel the next one (disarm()
// clears it, and the pointer identity is checked by nobody because there is only ever one).
//
// HOW CANCELLATION UNWINDS. check() THROWS. Return codes were not an option: flow_sampler returns
// the latent by value and a partially-stepped latent is not distinguishable from a finished one,
// and the DiT stages hold their weights in M1Harness, whose DESTRUCTOR is what frees the CUDA
// buffers. Throwing runs those destructors; a `return` from the middle of a lambda does not even
// exist as an option there. The rule for a new hook site is therefore: only call check() where a
// throw is safe, i.e. where every live GPU allocation is owned by something with a destructor.
// rig_beam_generate_batched's per-token ggml_context is raw, so the hook goes in the OUTER token
// loop (between calls, where nothing raw is live), not inside the graph build.
//
// This header pulls in <atomic> and <stdexcept> and nothing else, so a .cu TU can include it.
#pragma once
#include <atomic>
#include <stdexcept>

namespace cancelhook {

// Thrown by check(). Derives from std::exception so an existing `catch (const std::exception&)`
// does not terminate the process — but every such catch on the pipeline path RE-THROWS this type
// first, so a cancel is never reported as "refine failed".
struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled by request") {}
};

inline std::atomic<const std::atomic<bool>*>& slot() {
    static std::atomic<const std::atomic<bool>*> g{nullptr};
    return g;
}

// Arm/disarm for the duration of ONE request. avatar::Engine::Session does this; nothing else
// should. NOTE the image_to_rig CLI deliberately does NOT arm anything — it is the A/B harness
// and shell drivers depend on its exact behaviour, so Ctrl-C there still kills the process as it
// always has. avatar_e2e (a driver, not the harness) does install a SIGINT handler.
inline void arm(const std::atomic<bool>* flag) { slot().store(flag, std::memory_order_release); }
inline void disarm() { slot().store(nullptr, std::memory_order_release); }

// The hot path: one relaxed atomic load, then usually one more. Called at most once per DiT step /
// per token / per DC ladder level, so its cost is unmeasurable next to the work it guards.
inline bool cancelled() {
    const std::atomic<bool>* f = slot().load(std::memory_order_acquire);
    return f != nullptr && f->load(std::memory_order_relaxed);
}

inline void check() {
    if (cancelled()) throw Cancelled();
}

}  // namespace cancelhook
