#include "worker_supervisor.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>

#include <arpa/inet.h>
#include <json.hpp>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using json = nlohmann::json;

bool truthy(const char* value) {
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

bool takes_value(const std::string& arg) {
    return arg == "--listen-ip" || arg == "-l" || arg == "--listen-port" || arg == "-p" ||
           arg == "--diffusion-model" || arg == "--diffusion-model-edit" ||
           arg == "--diffusion-model-variants";
}

bool is_admin_path(const std::string& path) {
    return path == "/health" || path == "/v1/gpu/status" || path == "/v1/admin/drain" ||
           path == "/v1/admin/unload" || path == "/v1/admin/load";
}

bool is_generation_request(const httplib::Request& request) {
    if (request.method != "POST") return false;
    return request.path == "/sdcpp/v1/img_gen" || request.path == "/sdcpp/v1/vid_gen" ||
           request.path == "/ltx/v1/generate" || request.path == "/wan/v1/generate" ||
           request.path == "/generate" || request.path == "/sdapi/v1/txt2img" ||
           request.path == "/sdapi/v1/img2img" || request.path == "/v1/images/generations" ||
           request.path == "/v1/images/edits";
}

void copy_response(const httplib::Response& source, httplib::Response& destination) {
    destination.status = source.status;
    for (const auto& header : source.headers) {
        if (header.first == "Content-Length" || header.first == "Connection" ||
            header.first == "Transfer-Encoding") {
            continue;
        }
        destination.set_header(header.first, header.second);
    }
    destination.set_content(source.body, source.get_header_value("Content-Type", "application/octet-stream"));
}

}  // namespace

WorkerSupervisor::WorkerSupervisor(std::string argv0,
                                   std::vector<std::string> original_args,
                                   std::string base_model,
                                   std::string edit_model,
                                   std::string variants_spec,
                                   std::string default_gpu)
    : argv0_(std::move(argv0)),
      original_args_(std::move(original_args)),
      variants_(build_variants(base_model, edit_model, variants_spec)),
      default_gpu_(std::move(default_gpu)) {}

WorkerSupervisor::~WorkerSupervisor() {
    unload();
}

bool WorkerSupervisor::loaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pid_ > 0;
}

bool WorkerSupervisor::draining() const {
    return draining_.load();
}

void WorkerSupervisor::drain() {
    draining_.store(true);
}

void WorkerSupervisor::reopen() {
    draining_.store(false);
}

std::string WorkerSupervisor::active_model() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_model_;
}

std::string WorkerSupervisor::active_gpu() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_gpu_;
}

std::map<std::string, std::string> WorkerSupervisor::build_variants(const std::string& base_model,
                                                                      const std::string& edit_model,
                                                                      const std::string& variants_spec) {
    std::map<std::string, std::string> variants;
    if (!base_model.empty()) variants["base"] = base_model;
    if (!edit_model.empty()) variants["edit"] = edit_model;
    std::stringstream entries(variants_spec);
    std::string entry;
    while (std::getline(entries, entry, ';')) {
        const size_t delimiter = entry.find('=');
        if (delimiter == std::string::npos || delimiter == 0 || delimiter + 1 >= entry.size()) continue;
        const std::string name = entry.substr(0, delimiter);
        const std::string path = entry.substr(delimiter + 1);
        const bool valid_name = std::all_of(name.begin(), name.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-';
        });
        if (valid_name && !path.empty()) variants[name] = path;
    }
    return variants;
}

void WorkerSupervisor::unload_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int status = 0;
        while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
    }
    pid_ = -1;
    port_ = 0;
    active_model_.clear();
    active_gpu_.clear();
}

void WorkerSupervisor::unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    unload_locked();
    // Koblem's normal gate sequence is drain -> unload; reopening makes a later
    // request able to lazily create a fresh worker without requiring an extra /load.
    draining_.store(false);
}

int WorkerSupervisor::reserve_loopback_port() const {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return 0;
    }
    socklen_t length = sizeof(address);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        ::close(fd);
        return 0;
    }
    const int port = ntohs(address.sin_port);
    ::close(fd);
    return port;
}

std::vector<std::string> WorkerSupervisor::child_args(const std::string& model, int port) const {
    std::vector<std::string> out;
    const auto variant = variants_.find(model);
    const std::string& model_path = variant->second;
    bool replaced_model = false;
    for (size_t i = 0; i < original_args_.size(); ++i) {
        const std::string& arg = original_args_[i];
        if (takes_value(arg)) {
            if (i + 1 >= original_args_.size()) break;
            if (arg == "--diffusion-model") {
                out.push_back(arg);
                out.push_back(model_path);
                replaced_model = true;
            }
            ++i;
            continue;
        }
        out.push_back(arg);
    }
    if (!replaced_model && !model_path.empty()) {
        out.push_back("--diffusion-model");
        out.push_back(model_path);
    }
    out.push_back("--listen-ip");
    out.push_back("127.0.0.1");
    out.push_back("--listen-port");
    out.push_back(std::to_string(port));
    return out;
}

bool WorkerSupervisor::wait_until_ready_locked(std::string& error) {
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(1, 0);
    client.set_read_timeout(1, 0);
    for (int attempt = 0; attempt < 300; ++attempt) {
        int status = 0;
        if (::waitpid(pid_, &status, WNOHANG) == pid_) {
            pid_ = -1;
            error = "worker exited during startup";
            return false;
        }
        const auto response = client.Get("/health");
        if (response && response->status >= 200 && response->status < 300) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    error = "worker did not become healthy within 30 seconds";
    return false;
}

bool WorkerSupervisor::ensure_worker_locked(const std::string& requested_model,
                                             const std::string& requested_gpu,
                                             std::string& error) {
    const std::string model = requested_model.empty() ?
        (active_model_.empty() ? std::string("base") : active_model_) : requested_model;
    if (variants_.find(model) == variants_.end()) {
        error = "unknown model variant '" + model + "'";
        return false;
    }
    const std::string gpu = requested_gpu.empty() ? default_gpu_ : requested_gpu;
    if (pid_ > 0 && active_model_ == model && active_gpu_ == gpu) return true;
    // Switching must be a full process recycle to release the prior CUDA primary
    // context.  Do not make that cleanup destructive to an existing render: the
    // Koblem contract drains/unloads before a switch, and direct callers receive
    // a retryable conflict instead of a silent cancelled job.
    if (pid_ > 0 && (active_generation_requests_.load() > 1 || child_busy_on_port(port_))) {
        error = "model or GPU switch requires the active worker to be idle";
        return false;
    }
    unload_locked();

    port_ = reserve_loopback_port();
    if (port_ == 0) {
        error = "could not reserve a worker loopback port";
        return false;
    }
    const auto args = child_args(model, port_);
    const pid_t pid = ::fork();
    if (pid < 0) {
        error = "fork failed";
        port_ = 0;
        return false;
    }
    if (pid == 0) {
        ::setenv("SD_SERVER_WORKER_CHILD", "1", 1);
        // The child receives the selected DiT as --diffusion-model, so retain
        // the complete logical variant map separately for safe in-chain LTX
        // leases (including a later return from edit to base).
        std::string variant_map;
        for (const auto& [name, path] : variants_) {
            if (!variant_map.empty()) variant_map += ';';
            variant_map += name + "=" + path;
        }
        ::setenv("SD_SERVER_WORKER_VARIANT_MAP", variant_map.c_str(), 1);
        if (!gpu.empty()) ::setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(argv0_.c_str()));
        for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        ::execv(argv0_.c_str(), argv.data());
        ::_exit(127);
    }
    pid_ = static_cast<int>(pid);
    active_model_ = model;
    active_gpu_ = gpu;
    if (!wait_until_ready_locked(error)) {
        unload_locked();
        return false;
    }
    return true;
}

bool WorkerSupervisor::child_busy_on_port(int port) {
    if (port <= 0) return false;
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(1, 0);
    client.set_read_timeout(1, 0);
    const auto response = client.Get("/health");
    if (!response || response->status < 200 || response->status >= 300) return false;
    try {
        return json::parse(response->body).value("busy", false);
    } catch (...) {
        return false;
    }
}

bool WorkerSupervisor::child_busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pid_ > 0 && child_busy_on_port(port_);
}

int WorkerSupervisor::in_flight() const {
    return active_generation_requests_.load() + (child_busy() ? 1 : 0);
}

std::string WorkerSupervisor::request_model(const httplib::Request& request) {
    if (request.method != "POST") return {};
    std::string body = request.body;
    if (request.is_multipart_form_data() && request.form.has_field("request")) {
        body = request.form.get_field("request");
    }
    try {
        const json parsed = json::parse(body);
        if (parsed.contains("model") && parsed["model"].is_string()) return parsed["model"].get<std::string>();
    } catch (...) {}
    return {};
}

std::string WorkerSupervisor::request_gpu(const httplib::Request& request) {
    if (request.method != "POST") return {};
    std::string body = request.body;
    if (request.is_multipart_form_data() && request.form.has_field("request")) {
        body = request.form.get_field("request");
    }
    try {
        const json parsed = json::parse(body);
        if (parsed.contains("gpu") && parsed["gpu"].is_string()) return parsed["gpu"].get<std::string>();
    } catch (...) {}
    return {};
}

bool WorkerSupervisor::proxy(const httplib::Request& request,
                             httplib::Response& response,
                             const std::string& requested_model) {
    if (draining() && is_generation_request(request)) {
        response.status = 503;
        response.set_content(R"({"error":"service draining — not accepting new generation requests"})", "application/json");
        return true;
    }
    struct ActiveGenerationGuard {
        std::atomic<int>* counter = nullptr;
        ~ActiveGenerationGuard() { if (counter != nullptr) counter->fetch_sub(1); }
    } active_guard;
    if (is_generation_request(request)) {
        active_generation_requests_.fetch_add(1);
        active_guard.counter = &active_generation_requests_;
    } else if (!loaded()) {
        // An unload is a strict residency boundary.  In particular, Koblem may
        // issue one final job/media poll after the gate has evicted a service;
        // that read must never lazily recreate a CUDA context and make VRAM
        // reappear.  Only a new generation request is allowed to cold-start a
        // worker after /unload (or after a normal idle shutdown).
        response.status = 410;
        response.set_content(R"({"error":"worker is unloaded; submit a new generation request to start it"})",
                             "application/json");
        return true;
    }
    int worker_port = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string error;
        if (!ensure_worker_locked(requested_model, request_gpu(request), error)) {
            response.status = error == "model or GPU switch requires the active worker to be idle" ? 409 : 503;
            response.set_content(json({{"error", "worker unavailable"}, {"message", error}}).dump(), "application/json");
            return true;
        }
        worker_port = port_;
    }
    httplib::Client client("127.0.0.1", worker_port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(60 * 60, 0);
    client.set_write_timeout(60 * 60, 0);
    httplib::Request proxied;
    proxied.method = request.method;
    proxied.path = request.path;
    proxied.params = request.params;
    proxied.body = request.body;
    for (const auto& header : request.headers) {
        if (header.first != "Host" && header.first != "Connection" && header.first != "Content-Length") {
            proxied.headers.emplace(header.first, header.second);
        }
    }
    const auto result = client.send(proxied);
    if (!result) {
        response.status = 502;
        response.set_content(json({{"error", "worker proxy failed"}, {"detail", httplib::to_string(result.error())}}).dump(), "application/json");
        return true;
    }
    copy_response(*result, response);
    return true;
}

bool worker_isolation_requested() {
    return truthy(std::getenv("SD_SERVER_WORKER_ISOLATION")) || truthy(std::getenv("SD_IMAGE_ISOLATION")) ||
           truthy(std::getenv("LTX_VIDEO_ISOLATION")) || truthy(std::getenv("WAN_VIDEO_ISOLATION")) ||
           truthy(std::getenv("LONGCAT_AVATAR_WORKER_ISOLATION"));
}

bool worker_isolation_child() {
    return truthy(std::getenv("SD_SERVER_WORKER_CHILD"));
}

void register_worker_supervisor_endpoints(httplib::Server& server, WorkerSupervisor& supervisor) {
    server.Get("/health", [&supervisor](const httplib::Request&, httplib::Response& response) {
        const int in_flight = supervisor.in_flight();
        response.set_content(json({{"status", "ok"}, {"busy", in_flight > 0}, {"in_flight", in_flight},
                                   {"draining", supervisor.draining()}, {"loaded", supervisor.loaded()},
                                   {"loaded_model", supervisor.active_model()}}).dump(),
                             "application/json");
    });
    server.Get("/v1/gpu/status", [&supervisor](const httplib::Request&, httplib::Response& response) {
        response.set_content(json({{"loaded", supervisor.loaded()}, {"gpu", supervisor.loaded() ? json(supervisor.active_gpu()) : json(nullptr)}}).dump(),
                             "application/json");
    });
    server.Post("/v1/admin/drain", [&supervisor](const httplib::Request&, httplib::Response& response) {
        supervisor.drain();
        const int in_flight = supervisor.in_flight();
        response.set_content(json({{"status", "draining"}, {"busy", in_flight > 0}, {"in_flight", in_flight}}).dump(),
                             "application/json");
    });
    server.Post("/v1/admin/unload", [&supervisor](const httplib::Request& request, httplib::Response& response) {
        bool force = false;
        if (!request.body.empty()) {
            try {
                force = json::parse(request.body).value("force", false);
            } catch (...) {
                // Preserve the established lenient admin contract for malformed bodies.
            }
        }
        const int in_flight = supervisor.in_flight();
        if (in_flight > 0 && !force) {
            response.set_content(json({{"status", "busy (pass force=true to cancel + unload)"},
                                       {"busy", true}, {"in_flight", in_flight}}).dump(),
                                 "application/json");
            return;
        }
        const bool was_loaded = supervisor.loaded();
        supervisor.unload();
        response.set_content(json({{"status", was_loaded ? "unloaded" : "idle"}, {"unloaded", was_loaded},
                                   {"cuda_context_released", true}, {"forced", force && in_flight > 0}}).dump(),
                             "application/json");
    });
    server.Post("/v1/admin/load", [&supervisor](const httplib::Request&, httplib::Response& response) {
        supervisor.reopen();
        response.set_content(json({{"status", "ok"}, {"loaded", supervisor.loaded()}}).dump(), "application/json");
    });
    server.set_pre_routing_handler([&supervisor](const httplib::Request& request, httplib::Response& response) {
        std::string origin = request.get_header_value("Origin");
        if (origin.empty()) origin = "*";
        response.set_header("Access-Control-Allow-Origin", origin);
        response.set_header("Access-Control-Allow-Credentials", "true");
        response.set_header("Access-Control-Allow-Methods", "*");
        response.set_header("Access-Control-Allow-Headers", "*");
        if (request.method == "OPTIONS") {
            response.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        if (is_admin_path(request.path)) return httplib::Server::HandlerResponse::Unhandled;
        supervisor.proxy(request, response, WorkerSupervisor::request_model(request));
        return httplib::Server::HandlerResponse::Handled;
    });
}
