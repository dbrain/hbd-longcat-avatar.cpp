#include "routes.h"

#include <cstdlib>

#include "async_jobs.h"
#include "runtime.h"

namespace {

int active_jobs(const AsyncJobManager& manager) {
    int count = 0;
    for (const auto& [_, job] : manager.jobs) {
        if (job != nullptr && job->status == AsyncJobStatus::Generating) {
            ++count;
        }
    }
    return count;
}

void cancel_queued_jobs(AsyncJobManager& manager) {
    for (const auto& id : manager.queue) {
        const auto it = manager.jobs.find(id);
        if (it == manager.jobs.end() || it->second == nullptr) {
            continue;
        }
        AsyncGenerationJob& job = *it->second;
        job.status              = AsyncJobStatus::Cancelled;
        job.completed_at        = unix_timestamp_now();
        job.error_code          = "unloaded";
        job.error_message       = "cancelled while releasing GPU residency";
    }
    manager.queue.clear();
}

void set_json(httplib::Response& res, json body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

}  // namespace

void register_gpu_sharing_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    svr.Get("/health", [runtime](const httplib::Request&, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        const int in_flight = active_jobs(manager);
        // `accepts_part_refs` is the deploy-order gate (`docs/media-transport.md` §9.4): koblem
        // may only send `part:<name>` media to an engine that says it understands it. Without a
        // flag, rolling an engine back while koblem is on the new path turns every marker into a
        // string that fails to decode — a corrupt request rather than a clean fallback.
        set_json(res,
                 {
                     {"status", "ok"},
                     {"busy", in_flight > 0},
                     {"in_flight", in_flight},
                     {"draining", runtime_is_draining(*runtime)},
                     {"loaded", runtime->gpu_sharing != nullptr && runtime->gpu_sharing->diffusion_loaded.load()},
                     {"accepts_part_refs", true},
                 });
    });

    svr.Get("/v1/gpu/status", [runtime](const httplib::Request&, httplib::Response& res) {
        const bool loaded = runtime->gpu_sharing != nullptr && runtime->gpu_sharing->diffusion_loaded.load();
        json body         = {{"loaded", loaded}, {"gpu", nullptr}};
        if (const char* visible = std::getenv("CUDA_VISIBLE_DEVICES"); visible != nullptr && visible[0] != '\0') {
            body["gpu"] = visible;
        }
        set_json(res, std::move(body));
    });

    svr.Post("/v1/admin/drain", [runtime](const httplib::Request&, httplib::Response& res) {
        if (runtime->gpu_sharing != nullptr) {
            runtime->gpu_sharing->draining.store(true);
        }
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        const int in_flight = active_jobs(manager);
        set_json(res, {{"status", "draining"}, {"busy", in_flight > 0}, {"in_flight", in_flight}});
    });

    svr.Post("/v1/admin/unload", [runtime](const httplib::Request& req, httplib::Response& res) {
        bool force = false;
        if (!req.body.empty()) {
            try {
                force = json::parse(req.body).value("force", false);
            } catch (...) {
                // Preserve the established lenient admin contract for an empty/bad body.
            }
        }

        AsyncJobManager& manager = *runtime->async_job_manager;
        {
            std::lock_guard<std::mutex> lock(manager.mutex);
            const int in_flight = active_jobs(manager);
            if (in_flight > 0 && !force) {
                set_json(res, {{"status", "busy (pass force=true to cancel + unload)"}});
                return;
            }
            if (in_flight > 0) {
                sd_cancel_generation(runtime->sd_ctx, SD_CANCEL_ALL);
            }
            cancel_queued_jobs(manager);
        }

        // Locking serializes against a forced active generation until it observes the
        // cancellation request. The DiT can then be released without racing ggml.
        bool was_loaded = runtime->gpu_sharing == nullptr || runtime->gpu_sharing->diffusion_loaded.load();
        {
            std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
            sd_ctx_free_diffusion_model(runtime->sd_ctx);
        }
        if (runtime->gpu_sharing != nullptr) {
            runtime->gpu_sharing->diffusion_loaded.store(false);
            // The external gate issues drain -> unload, not load. Reopen now that
            // residency is gone so a later request may lazily reload the DiT.
            runtime->gpu_sharing->draining.store(false);
        }
        set_json(res, {{"status", was_loaded ? "unloaded" : "idle"}, {"unloaded", was_loaded}});
    });

    // Switch the resident DiT variant IN PROCESS, instead of the supervisor killing this
    // child and forking a new one to reload ~29 GB.
    //
    // The recycle's stated justification -- "a full process recycle to release the prior CUDA
    // primary context" -- does not hold: this same binary already swaps the top-level DiT in
    // process on the img_gen path and per chain segment, and the pre-rebuild production tree
    // did it too and never recycled. What the recycle genuinely bought was a guaranteed-zero
    // allocator watermark, which is an empirical question about heap fragmentation, not a
    // correctness one.
    //
    // 409 ON A QUEUED JOB, NOT JUST A RUNNING ONE. /health's `busy` counts only Generating
    // jobs, so a job sitting in manager.queue reads as idle. The recycle path SIGKILLs such a
    // job silently today; an in-process swap would be worse -- it would pull the weights out
    // from under a job that is about to start. The queue is the authority here.
    svr.Post("/v1/admin/swap_model", [runtime](const httplib::Request& req, httplib::Response& res) {
        std::string wanted;
        try {
            wanted = json::parse(req.body).value("model", std::string());
        } catch (...) {
            wanted.clear();
        }
        if (wanted.empty()) {
            set_json(res, {{"error", "swap_model requires a non-empty \"model\""}}, 400);
            return;
        }
        const auto variants = runtime_diffusion_model_variants(*runtime);
        const auto variant  = variants.find(wanted);
        if (variant == variants.end()) {
            set_json(res, {{"error", "unknown diffusion model variant '" + wanted + "'"}}, 400);
            return;
        }
        {
            AsyncJobManager& manager = *runtime->async_job_manager;
            std::lock_guard<std::mutex> lock(manager.mutex);
            if (active_jobs(manager) > 0 || !manager.queue.empty()) {
                set_json(res, {{"error", "worker has running or queued jobs"}}, 409);
                return;
            }
        }
        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
            ok = sd_ctx_swap_diffusion_model(runtime->sd_ctx, variant->second.c_str());
        }
        if (!ok) {
            // A failed swap can leave the DiT UNREGISTERED (see swap_diffusion_model's late
            // failure paths), so this is not a soft "keep the old model" outcome. The caller
            // must treat a non-200 as "recycle the worker", which is what it does.
            set_json(res, {{"error", "could not swap to '" + wanted + "'"}}, 500);
            return;
        }
        if (runtime->gpu_sharing != nullptr) {
            runtime->gpu_sharing->diffusion_loaded.store(true);
        }
        set_json(res, {{"status", "ok"}, {"model", wanted}});
    });

    svr.Post("/v1/admin/load", [runtime](const httplib::Request&, httplib::Response& res) {
        if (runtime->gpu_sharing != nullptr) {
            runtime->gpu_sharing->draining.store(false);
        }
        // ModelManager reloads lazily on the next generation. Avoid an eager load
        // here so this endpoint remains safe while GPU capacity is still contested.
        const bool loaded = runtime->gpu_sharing != nullptr && runtime->gpu_sharing->diffusion_loaded.load();
        set_json(res, {{"status", "ok"}, {"loaded", loaded}});
    });
}
