#pragma once

#include <atomic>
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
    void unload();
    bool loaded() const;
    void drain();
    void reopen();
    bool draining() const;
    std::string active_model() const;
    std::string active_gpu() const;

private:
    bool ensure_worker_locked(const std::string& model, const std::string& gpu, std::string& error);
    void unload_locked();
    bool wait_until_ready_locked(std::string& error);
    int reserve_loopback_port() const;
    std::vector<std::string> child_args(const std::string& model, int port) const;
    static std::string request_gpu(const httplib::Request& request);

    std::string argv0_;
    std::vector<std::string> original_args_;
    std::string base_model_;
    std::string edit_model_;
    std::string default_gpu_;
    mutable std::mutex mutex_;
    int pid_ = -1;
    int port_ = 0;
    std::atomic<bool> draining_{false};
    std::string active_model_;
    std::string active_gpu_;
};

bool worker_isolation_requested();
bool worker_isolation_child();
void register_worker_supervisor_endpoints(httplib::Server& server, WorkerSupervisor& supervisor);
