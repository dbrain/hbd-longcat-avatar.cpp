#ifndef __WEIGHT_MANAGER_H__
#define __WEIGHT_MANAGER_H__

#include <vector>

#include "ggml-backend.h"

struct ggml_tensor;

struct RunnerWeightManager {
    virtual ~RunnerWeightManager()                                                        = default;
    virtual bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                        ggml_backend_t compute_backend)                   = 0;
    virtual bool prepare_params(const std::vector<ggml_tensor*>& tensors)                 = 0;
    virtual bool retain_compute_backend_params(const std::vector<ggml_tensor*>& tensors)  = 0;
    virtual void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) = 0;
    virtual void release_retained_compute_backend_params(const std::vector<ggml_tensor*>& tensors) = 0;
    virtual void release_params_backend_params(const std::vector<ggml_tensor*>& tensors)  = 0;

    // Copy/compute overlap for the weight-streaming path, driven by the graph-cut segment
    // loop. Three phases, and the phase a call runs on is part of the contract:
    //
    //   begin_stage_prefetch(next segment's params)  MAIN thread, after this segment's own
    //       params are staged and before its graph is enqueued. Allocates the device buffer
    //       (a cudaMalloc, so it must not happen with kernels in flight) and returns false
    //       if it declined -- no prefetch, no worker, no cleanup owed.
    //   run_stage_prefetch()                         WORKER thread, while this segment's
    //       graph runs. Host->device copies ONLY. Touches no manager bookkeeping.
    //   finish_stage_prefetch()                      MAIN thread, after the worker is joined.
    //       Publishes the staging block, or discards it if the worker never ran.
    //
    // Default-off no-ops so a manager that does not implement streaming is unaffected.
    virtual bool begin_stage_prefetch(const std::vector<ggml_tensor*>& /*tensors*/) { return false; }
    virtual void run_stage_prefetch() {}
    virtual void finish_stage_prefetch() {}
};

#endif  // __WEIGHT_MANAGER_H__
