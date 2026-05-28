// longcat-avatar-server main. Started life as the sd.cpp `sd-server`; the
// LongCat-Video-Avatar production server replaces tools/avatar_server.py
// with a warm-weights native binary + worker-subprocess isolation
// (LONGCAT_AVATAR_WORKER_ISOLATION=1) matching the qwen3-tts.cpp /
// siglip2.cpp / parakeet.cpp pattern. See HANDOFF-native-server.md.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

#include "async_jobs.h"
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "routes.h"
#include "routes_longcat.h"
#include "runtime.h"
#include "worker_ipc.h"
#include "worker_session.h"

#ifdef HAVE_INDEX_HTML
#include "frontend/dist/gen_index_html.h"
#endif

static void print_usage(const char* argv0, const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv0 << " [options]\n\n";
    std::cout << "Svr Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nDefault Generation Options:\n";
    options_list[2].print();
}

static bool parse_args_with(int argc,
                            const char** argv,
                            SDSvrParams& svr_params,
                            SDContextParams& ctx_params,
                            SDGenerationParams& default_gen_params,
                            bool* out_normal_exit = nullptr) {
    std::vector<ArgOptions> options_vec = {
        svr_params.get_options(),
        ctx_params.get_options(),
        default_gen_params.get_options(),
    };

    if (!parse_options(argc, argv, options_vec)) {
        if (out_normal_exit) *out_normal_exit = svr_params.normal_exit;
        print_usage(argv[0], options_vec);
        return false;
    }
    const bool random_seed_requested = default_gen_params.seed < 0;

    if (!svr_params.resolve_and_validate() ||
        !ctx_params.resolve_and_validate(IMG_GEN) ||
        !default_gen_params.resolve_and_validate(IMG_GEN,
                                                 ctx_params.lora_model_dir,
                                                 ctx_params.hires_upscalers_dir)) {
        print_usage(argv[0], options_vec);
        return false;
    }
    if (random_seed_requested) {
        default_gen_params.seed = -1;
    }
    return true;
}

// Exposed for worker_session.cpp's child-side dispatch loop so it doesn't
// have to duplicate the option vectors.
namespace longcat_avatar {
bool parse_server_args_into(int argc,
                            const char** argv,
                            SDSvrParams* svr_params,
                            SDContextParams* ctx_params,
                            SDGenerationParams* default_gen_params) {
    return parse_args_with(argc, argv, *svr_params, *ctx_params, *default_gen_params);
}
} // namespace longcat_avatar

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDSvrParams* svr_params = (SDSvrParams*)data;
    log_print(level, log, svr_params->verbose, svr_params->color);
}

static bool env_truthy(const char* k) {
    const char* v = std::getenv(k);
    return v != nullptr && v[0] && v[0] != '0';
}

int main(int argc, const char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

    // ─── Worker subprocess role ──────────────────────────────────────────────
    // longcat-avatar-server --worker <fd> ... → child-side dispatch.
    if (argc >= 3 && std::string(argv[1]) == "--worker") {
        int fd = std::atoi(argv[2]);
        if (fd <= 0) {
            std::fprintf(stderr, "worker: bad fd argument\n");
            return 2;
        }
        return longcat_avatar::run_worker_loop(fd, argc, argv);
    }

    SDSvrParams svr_params;
    SDContextParams ctx_params;
    SDGenerationParams default_gen_params;
    bool normal_exit = false;
    if (!parse_args_with(argc, argv, svr_params, ctx_params, default_gen_params, &normal_exit)) {
        return normal_exit ? 0 : 1;
    }

    sd_set_log_callback(sd_log_cb, (void*)&svr_params);
    log_verbose = svr_params.verbose;
    log_color   = svr_params.color;

    LOG_DEBUG("version: %s", version_string().c_str());
    LOG_DEBUG("%s", sd_get_system_info());
    LOG_DEBUG("%s", svr_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", default_gen_params.to_string().c_str());

    bool worker_iso = env_truthy("LONGCAT_AVATAR_WORKER_ISOLATION");

    // ─── Worker-isolation parent. CUDA-free; just HTTP + admin endpoints. ───
    if (worker_iso) {
        LOG_INFO("worker-isolation: ENABLED (weights load in subprocess on first /generate)\n");

        // Strip --worker-iso-internal sentinels just in case; pass the
        // remaining argv through to the child via WorkerSession's extra_argv.
        std::vector<std::string> child_extra_argv;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--worker" && i + 1 < argc) { i++; continue; }
            child_extra_argv.push_back(a);
        }

        LongCatRuntime ctx;
        ctx.worker = std::make_unique<longcat_avatar::WorkerSession>(argv[0], child_extra_argv);
        const char* outdir_env = std::getenv("LONGCAT_AVATAR_OUTDIR");
        ctx.outdir = outdir_env ? outdir_env : "/tmp/avatar-out";

        httplib::Server svr;
        svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            std::string origin = req.get_header_value("Origin");
            if (origin.empty()) origin = "*";
            res.set_header("Access-Control-Allow-Origin", origin);
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "*");
            res.set_header("Access-Control-Allow-Headers", "*");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
            // body-less POST workaround — see routes_longcat.cpp.
            if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
                !req.has_header("Content-Length") &&
                req.get_header_value("Transfer-Encoding") != "chunked") {
                const_cast<httplib::Request&>(req).set_header("Content-Length", "0");
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        // httplib's default payload limit is 8 MiB; bump for multipart
        // image+audio + base64-bloated JSON renders (a 1-minute b64 png +
        // wav can exceed 16 MiB).
        svr.set_payload_max_length(512ull * 1024 * 1024);
        // Avatar renders take minutes — the default httplib server-side
        // read+write timeout (5s) drops the connection mid-render. Bump to
        // 30 minutes so a chained long-form clip can stream the WebM back
        // even on the cold-load path.
        svr.set_read_timeout(60 * 30, 0);
        svr.set_write_timeout(60 * 30, 0);
        svr.set_idle_interval(60, 0);

        register_longcat_endpoints(svr, ctx);

        LOG_INFO("listening on: http://%s:%d (worker-isolation, lazy worker spawn)\n",
                 svr_params.listen_ip.c_str(), svr_params.listen_port);
        svr.listen(svr_params.listen_ip, svr_params.listen_port);
        ctx.worker->shutdown();
        return 0;
    }

    // ─── In-process mode (legacy sd-server + native /generate). ──────────────
    // Used when LONGCAT_AVATAR_WORKER_ISOLATION is not set: keeps the existing
    // sd.cpp HTTP framework (img_gen / vid_gen / sdapi / openai endpoints)
    // intact and ALSO mounts the avatar `/generate` route directly on the
    // resident sd_ctx (no worker fork).
    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(false, false, false);
    SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));
    if (sd_ctx == nullptr) {
        LOG_ERROR("new_sd_ctx_t failed");
        return 1;
    }

    std::mutex sd_ctx_mutex;
    std::vector<LoraEntry>     lora_cache;
    std::mutex                 lora_mutex;
    std::vector<UpscalerEntry> upscaler_cache;
    std::mutex                 upscaler_mutex;
    AsyncJobManager async_job_manager;
    ServerRuntime runtime = {
        sd_ctx.get(),
        &sd_ctx_mutex,
        &svr_params,
        &ctx_params,
        &default_gen_params,
        &lora_cache,
        &lora_mutex,
        &upscaler_cache,
        &upscaler_mutex,
        &async_job_manager,
    };

    std::thread async_worker(async_job_worker, std::ref(runtime));

    httplib::Server svr;

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) origin = "*";
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    svr.set_payload_max_length(512ull * 1024 * 1024);
    svr.set_read_timeout(60 * 30, 0);
    svr.set_write_timeout(60 * 30, 0);
    svr.set_idle_interval(60, 0);

    std::string index_html;
#ifdef HAVE_INDEX_HTML
    index_html.assign(reinterpret_cast<const char*>(index_html_bytes), index_html_size);
#else
    index_html = "longcat-avatar-server is running";
#endif
    register_index_endpoints(svr, svr_params, index_html);
    register_openai_api_endpoints(svr, runtime);
    register_sdapi_endpoints(svr, runtime);
    register_sdcpp_api_endpoints(svr, runtime);

    LOG_INFO("listening on: http://%s:%d (in-process)\n",
             svr_params.listen_ip.c_str(), svr_params.listen_port);
    svr.listen(svr_params.listen_ip, svr_params.listen_port);

    {
        std::lock_guard<std::mutex> lock(async_job_manager.mutex);
        async_job_manager.stop = true;
    }
    async_job_manager.cv.notify_all();
    async_worker.join();
    return 0;
}
