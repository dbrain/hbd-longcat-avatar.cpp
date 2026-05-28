// routes_longcat.h — POST /generate + /v1/admin/* endpoint set for the
// native longcat-avatar-server.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include <json.hpp>

#include "httplib.h"
#include "worker_session.h"

using json = nlohmann::json;

struct LongCatState {
    std::atomic<int>     in_flight{0};
    std::atomic<int>     renders_done{0};
    std::atomic<double>  last_render_sec{0.0};
    std::atomic<bool>    draining{false};
    std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now();
};

struct LongCatRuntime {
    std::unique_ptr<longcat_avatar::WorkerSession> worker;
    LongCatState state;
    std::string outdir;  // where "file"-mode returns persist webm files
};

void register_longcat_endpoints(httplib::Server& svr, LongCatRuntime& ctx);
