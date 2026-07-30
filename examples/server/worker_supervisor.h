#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "httplib.h"

// CUDA-free sd-server parent used by the GPU gate.  It owns no sd_ctx and proxies
// regular API traffic to one lazily-spawned child.  Killing that child is therefore
// a real CUDA-context teardown, rather than a best-effort model-buffer release.
class WorkerSupervisor {
public:
    WorkerSupervisor(std::string argv0,
                     std::vector<std::string> original_args,
                     std::string base_model,
                     std::string edit_model,
                     std::string variants_spec,
                     std::string default_gpu);
    ~WorkerSupervisor();

    WorkerSupervisor(const WorkerSupervisor&) = delete;
    WorkerSupervisor& operator=(const WorkerSupervisor&) = delete;

    // Starts/reuses a child suitable for this request and forwards its HTTP request.
    // `model` is a named server variant (base/edit); empty retains the active child.
    bool proxy(const httplib::Request& request, httplib::Response& response, const std::string& model);
    static std::string request_model(const httplib::Request& request);

    // SIGKILL is intentional: the child owns CUDA's primary context and all backend
    // pools, so process exit is the only complete/portable VRAM release mechanism.
    // Returns only after the child has exited. A false result means the parent
    // could not prove the CUDA-owning worker is gone, so callers must not admit
    // a competing GPU workload.
    bool unload();
    bool loaded();
    int worker_pid();
    void drain();
    void reopen();
    bool draining() const;
    std::string active_model();
    std::string active_gpu();
    int in_flight();

private:
    bool ensure_worker_locked(const std::string& model, const std::string& gpu, std::string& error);
    bool reap_exited_locked();
    void clear_worker_locked();
    bool unload_locked();
    bool wait_until_ready_locked(std::string& error);
    int reserve_loopback_port() const;
    std::vector<std::string> child_args(const std::string& model, int port) const;
    static std::string request_gpu(const httplib::Request& request);
    bool child_busy();
    static bool child_busy_on_port(int port);
    static bool inproc_model_switch_enabled();
    static bool swap_child_model(int port, const std::string& model);
    static std::map<std::string, std::string> build_variants(const std::string& base_model,
                                                              const std::string& edit_model,
                                                              const std::string& variants_spec);

    std::string argv0_;
    std::vector<std::string> original_args_;
    std::map<std::string, std::string> variants_;
    std::string default_gpu_;
    mutable std::mutex mutex_;
    int pid_ = -1;
    int port_ = 0;
    // mutable: draining() is const but self-heals an abandoned drain in place.
    mutable std::atomic<bool> draining_{false};
    // Wall-clock (seconds since epoch) when draining_ was last raised. A drain is only
    // ever meant to bridge a gate's drain -> unload handover; if the gate dies, times
    // out, or is otherwise abandoned mid-sequence, nothing clears the flag and the
    // service refuses every generation request forever while still holding VRAM. Bound
    // it so an abandoned drain self-heals. See draining().
    mutable std::atomic<long long> drain_started_at_{0};
    std::atomic<int> active_generation_requests_{0};
    std::string active_model_;
    std::string active_gpu_;
};

bool worker_isolation_requested();
bool worker_isolation_child();
void register_worker_supervisor_endpoints(httplib::Server& server, WorkerSupervisor& supervisor);
