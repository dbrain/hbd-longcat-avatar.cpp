#include "worker_supervisor.h"

#include "common/log.h"
// For scan_lora_dir / lora_entry_json — the CUDA-free half of the LoRA listing, so the supervisor
// can answer /sdapi/v1/loras without waking the worker.
#include "runtime.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include <arpa/inet.h>
#include <json.hpp>
#include <netinet/in.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

bool truthy(const char* value) {
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
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

bool valid_ltx_bank_id(const std::string& id) {
    return !id.empty() && id.size() <= 128 &&
           std::all_of(id.begin(), id.end(), [](unsigned char ch) {
               return std::isalnum(ch) || ch == '_' || ch == '-';
           });
}

std::vector<fs::path> ltx_bank_roots() {
    const char* jobs_env = std::getenv("LTX_JOB_DIR");
    const char* persist_env = std::getenv("LTX_PERSIST_DIR");
    const fs::path jobs = jobs_env != nullptr && jobs_env[0] != '\0'
        ? jobs_env : "/var/lib/ltx-video/jobs";
    const fs::path persist = persist_env != nullptr && persist_env[0] != '\0'
        ? persist_env : "/var/lib/ltx-video/persist";
    return persist == jobs ? std::vector<fs::path>{jobs} : std::vector<fs::path>{persist, jobs};
}

bool delete_ltx_bank(const std::string& job_id, bool& deleted, std::string& error_message) {
    if (!valid_ltx_bank_id(job_id)) {
        error_message = "id must be an LTX job id";
        return false;
    }
    for (const fs::path& root : ltx_bank_roots()) {
        const fs::path requested_dir = root / job_id;
        std::error_code error;
        if (!fs::is_directory(requested_dir, error)) continue;
        std::string bank_id = job_id;
        std::ifstream reference(requested_dir / "bank_id");
        if (reference.is_open()) {
            std::getline(reference, bank_id);
            if (!valid_ltx_bank_id(bank_id)) {
                error_message = "invalid LTX bank reference";
                return false;
            }
        }
        const uintmax_t bank_removed = fs::remove_all(root / bank_id, error);
        if (error) {
            error_message = error.message();
            return false;
        }
        uintmax_t alias_removed = 0;
        if (bank_id != job_id) {
            alias_removed = fs::remove_all(requested_dir, error);
            if (error) {
                error_message = error.message();
                return false;
            }
        }
        deleted = deleted || bank_removed > 0 || alias_removed > 0;
    }
    return true;
}

// Header names are case-insensitive on the wire (RFC 9110 §5.1), and callers do not
// agree on a spelling: curl sends "Content-Length", Rust hyper/reqwest (Koblem) sends
// "content-length".  These names describe THIS hop's framing, so they must never be
// forwarded — the proxy re-encodes the body and httplib re-derives them.  Matching them
// case-sensitively let Koblem's lowercase content-length + stale multipart boundary
// through, and because httplib's client looks headers up case-INsensitively it then
// declined to emit its own Content-Length.  The child was therefore told to expect the
// caller's byte count while receiving the (shorter) re-encoded body, starved on the
// missing tail, and returned httplib's empty-bodied 400 after its 5 s read timeout.
bool is_hop_by_hop_header(const std::string& name) {
    static const char* const kDropped[] = {"host", "connection", "content-length", "content-type",
                                           "transfer-encoding", "expect", "keep-alive", "upgrade", "te",
                                           "trailer", "proxy-connection", "proxy-authorization"};
    std::string lowered;
    lowered.reserve(name.size());
    for (const unsigned char c : name) lowered.push_back(static_cast<char>(std::tolower(c)));
    for (const char* dropped : kDropped) {
        if (lowered == dropped) return true;
    }
    return false;
}

void copy_response(const httplib::Response& source, httplib::Response& destination) {
    destination.status = source.status;
    for (const auto& header : source.headers) {
        // set_content() below erases and re-sets Content-Type from the source response,
        // so dropping it here is a no-op rather than a loss.
        if (is_hop_by_hop_header(header.first)) {
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
    (void)unload();
}

bool WorkerSupervisor::loaded() {
    std::lock_guard<std::mutex> lock(mutex_);
    reap_exited_locked();
    return pid_ > 0;
}

int WorkerSupervisor::worker_pid() {
    std::lock_guard<std::mutex> lock(mutex_);
    reap_exited_locked();
    return pid_ > 0 ? pid_ : 0;
}

// Longest a drain may stay raised before it is treated as abandoned. A drain exists only
// to bridge a gate's drain -> unload handover, which takes seconds; anything beyond this
// means the gate died, timed out, or was interrupted (e.g. a cancelled
// /gpu/services/<name> toggle) and is never coming back to unload us. Without a bound the
// service refuses every generation request with 503 "service draining" indefinitely while
// still holding its worker + VRAM, and only a manual POST /v1/admin/load recovers it.
// Generous by default so a legitimately slow handover is never cut short.
static long long drain_max_seconds() {
    static const long long value = [] {
        const char* env = getenv("SD_DRAIN_MAX_SECONDS");
        if (env != nullptr && env[0] != '\0') {
            const long long parsed = atoll(env);
            if (parsed > 0) {
                return parsed;
            }
        }
        return 600LL;  // 10 minutes
    }();
    return value;
}

bool WorkerSupervisor::draining() const {
    if (!draining_.load()) {
        return false;
    }
    const long long started = drain_started_at_.load();
    if (started > 0) {
        const long long now = (long long)::time(nullptr);
        if (now - started >= drain_max_seconds()) {
            // Self-heal: report (and latch) not-draining so the service starts serving
            // again instead of staying wedged behind an abandoned handover.
            draining_.store(false);
            drain_started_at_.store(0);
            LOG_WARN("drain abandoned after %llds (no unload followed); reopening for requests",
                     (long long)(now - started));
            return false;
        }
    }
    return true;
}

void WorkerSupervisor::drain() {
    drain_started_at_.store((long long)::time(nullptr));
    draining_.store(true);
}

void WorkerSupervisor::reopen() {
    draining_.store(false);
    drain_started_at_.store(0);
}

std::string WorkerSupervisor::active_model() {
    std::lock_guard<std::mutex> lock(mutex_);
    reap_exited_locked();
    return active_model_;
}

std::string WorkerSupervisor::active_gpu() {
    std::lock_guard<std::mutex> lock(mutex_);
    reap_exited_locked();
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

void WorkerSupervisor::clear_worker_locked() {
    pid_ = -1;
    port_ = 0;
    active_model_.clear();
    active_gpu_.clear();
}

bool WorkerSupervisor::reap_exited_locked() {
    if (pid_ <= 0) return false;
    int status = 0;
    const pid_t waited = ::waitpid(pid_, &status, WNOHANG);
    if (waited == 0) return false;
    if (waited == pid_ || (waited < 0 && errno == ECHILD)) {
        clear_worker_locked();
        return true;
    }
    return false;
}

bool WorkerSupervisor::unload_locked() {
    reap_exited_locked();
    if (pid_ <= 0) {
        return true;
    }
    const int pid = pid_;
    if (::kill(pid, SIGKILL) != 0 && errno != ESRCH) {
        return false;
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = ::waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    // ESRCH means it was already gone; ECHILD means no process owned by this
    // parent remains to retain the CUDA primary context. Any other wait error is
    // treated as a failed residency transition rather than claiming VRAM is free.
    if (waited != pid && !(waited < 0 && (errno == ECHILD || errno == ESRCH))) {
        return false;
    }
    clear_worker_locked();
    return true;
}

bool WorkerSupervisor::unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Koblem's normal gate sequence is drain -> unload; reopening makes a later
    // request able to lazily create a fresh worker without requiring an extra /load.
    //
    // Clear the flag on EVERY path, including when there was nothing to unload.
    // "unload" is the end of the gate sequence regardless of whether a worker
    // happened to be resident, so it must always reopen. Clearing it only on the
    // success path meant a drain followed by an unload that found no worker (already
    // exited, a raced double-unload, or a worker that died on its own) left
    // draining_ latched true forever: every subsequent generation request 503s with
    // "service draining" and nothing ever resets it, so the service looks hung and
    // only a manual POST /v1/admin/load recovers it.
    const bool unloaded = unload_locked();
    draining_.store(false);
    drain_started_at_.store(0);
    return unloaded;
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
    const std::string* model_path = variant == variants_.end() ? nullptr : &variant->second;
    bool replaced_model = false;
    for (size_t i = 0; i < original_args_.size(); ++i) {
        const std::string& arg = original_args_[i];
        // The parent owns the public listen address and the child owns its
        // private loopback address.  Likewise, the parent selects one model
        // variant per worker and exports the complete logical map separately
        // for LTX's in-chain variant leases.  Those are the only options that
        // must be consumed here.  Every other argument (notably --vae,
        // --t5xxl, --llm, --audio-vae and tiling/offload settings) belongs to
        // the real server and must reach the child unchanged.
        if (arg == "--listen-ip" || arg == "-l" || arg == "--listen-port" || arg == "-p" ||
            arg == "--diffusion-model-edit" || arg == "--diffusion-model-variants") {
            if (i + 1 >= original_args_.size()) break;
            ++i;
            continue;
        }
        if (arg == "--diffusion-model") {
            if (i + 1 >= original_args_.size()) break;
            out.push_back(arg);
            // Legacy single-model services (LongCat Avatar, and any server
            // launched with -m) do not publish a logical variant map.  Keep
            // their original model argument intact; only variant-aware
            // services substitute the selected --diffusion-model path.
            out.push_back(model_path != nullptr ? *model_path : original_args_[i + 1]);
            replaced_model = true;
            ++i;
            continue;
        }
        out.push_back(arg);
    }
    if (!replaced_model && model_path != nullptr && !model_path->empty()) {
        out.push_back("--diffusion-model");
        out.push_back(*model_path);
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
            if (WIFEXITED(status)) {
                error = "worker exited during startup (exit " + std::to_string(WEXITSTATUS(status)) + ")";
            } else if (WIFSIGNALED(status)) {
                error = "worker died during startup (signal " + std::to_string(WTERMSIG(status)) + ")";
            } else {
                error = "worker exited during startup";
            }
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
    reap_exited_locked();
    const std::string model = requested_model.empty() ?
        (active_model_.empty() ? std::string("base") : active_model_) : requested_model;
    // A server started with the legacy single-model `-m` option has no
    // diffusion variant map.  It is still a valid isolated worker: preserve
    // its original argv and use the stable logical name "base" for health.
    // Variant-aware services keep the strict validation needed for safe
    // Flux/LTX model switches.
    if (!variants_.empty() && variants_.find(model) == variants_.end()) {
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
    // Try an IN-PROCESS variant swap before recycling. Saves ~36 s per switch (~40 s recycle
    // + ~29 GB reload vs ~4 s swap). Preconditions, all of them load-bearing:
    //   * same GPU -- CUDA_VISIBLE_DEVICES is applied before execv, so a GPU change genuinely
    //     needs a new process and can never be a swap;
    //   * a variant map -- a legacy single-model `-m` server has nothing to swap between;
    //   * child alive, and idle including its QUEUE (the child re-checks and 409s).
    // Any non-200 falls through to the recycle below, which is required and not merely
    // preferred: sd_ctx_swap_diffusion_model's late failure paths can leave the DiT
    // unregistered, so the respawn is the repair.
    if (inproc_model_switch_enabled() && pid_ > 0 && active_gpu_ == gpu && !variants_.empty() &&
        variants_.find(model) != variants_.end() && swap_child_model(port_, model)) {
        active_model_ = model;
        return true;
    }

    if (!unload_locked()) {
        error = "could not terminate the prior GPU worker";
        return false;
    }

    port_ = reserve_loopback_port();
    if (port_ == 0) {
        error = "could not reserve a worker loopback port";
        return false;
    }
    const auto args = child_args(model, port_);
    const pid_t supervisor_pid = ::getpid();
    const pid_t pid = ::fork();
    if (pid < 0) {
        error = "fork failed";
        port_ = 0;
        return false;
    }
    if (pid == 0) {
        // Do not let a supervisor crash/restart orphan a CUDA-owning child.
        // Explicit /unload still waits for this PID, but parent-death signalling
        // covers every other exit path as well.
        // PID 1 is a normal supervisor PID inside a Docker PID namespace, so
        // compare against the pre-fork parent exactly rather than treating 1
        // as universally orphaned.
        if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
            ::_exit(125);
        }
        if (::getppid() != supervisor_pid) {
            ::_exit(126);
        }
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
        (void)unload_locked();
        return false;
    }
    return true;
}

// LTX_INPROC_MODEL_SWITCH=1 opts in. Default OFF: the swap replaces a guaranteed-zero
// allocator watermark with a live-context free/realloc, and whether that fragments the CUDA
// heap over many switches is empirical.
bool WorkerSupervisor::inproc_model_switch_enabled() {
    const char* value = std::getenv("LTX_INPROC_MODEL_SWITCH");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool WorkerSupervisor::swap_child_model(int port, const std::string& model) {
    if (port <= 0) return false;
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    // Registration walks every tensor of the incoming DiT; the weights themselves load
    // lazily afterwards, but this call is not instant.
    client.set_read_timeout(180, 0);
    const auto response = client.Post("/v1/admin/swap_model",
                                      json({{"model", model}}).dump(), "application/json");
    return response && response->status == 200;
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

bool WorkerSupervisor::child_busy() {
    std::lock_guard<std::mutex> lock(mutex_);
    reap_exited_locked();
    return pid_ > 0 && child_busy_on_port(port_);
}

int WorkerSupervisor::in_flight() {
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
    const bool generation = is_generation_request(request);
    if (draining() && generation) {
        response.status = 503;
        response.set_content(R"({"error":"service draining — not accepting new generation requests"})", "application/json");
        return true;
    }
    struct ActiveGenerationGuard {
        std::atomic<int>* counter = nullptr;
        ~ActiveGenerationGuard() { if (counter != nullptr) counter->fetch_sub(1); }
    } active_guard;
    if (generation) {
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
        reap_exited_locked();
        // Only a GENERATION request may move the worker between DiT variants or
        // cards.  A status/media poll carries neither field — request_model() and
        // request_gpu() both return early for non-POST — so resolving one through
        // ensure_worker_locked() compares the RUNNING worker against the spawn
        // DEFAULTS rather than against anything the caller asked for.  Whenever
        // Koblem's GPU gate placed the worker via the per-request `gpu` field and
        // that UUID differs from WORKER_DEFAULT_GPU (unset counts: the fallback is
        // then ""), every poll for the life of that render read as a pending GPU
        // switch and was answered 409 "requires the active worker to be idle".
        // Observed 2026-07-27: ltx-video recreated without docker-compose.override.yml
        // lost WORKER_DEFAULT_GPU, so the render ran to completion on the GPU while
        // Koblem 409'd on all 20 poll attempts and abandoned the job.
        //
        // A poll must therefore attach to whatever worker is already up.  Cold
        // start stays a generation-only privilege: the !loaded() check above is
        // unlocked, so a worker that exits (or is evicted by the gate) in the
        // window before we take the mutex would otherwise fall through to
        // ensure_worker_locked() and let a poll spawn a fresh CUDA child —
        // precisely the residency violation that 410 exists to prevent.  Re-test
        // it here, where the answer cannot go stale.
        if (!generation) {
            if (pid_ <= 0) {
                response.status = 410;
                response.set_content(
                    R"({"error":"worker is unloaded; submit a new generation request to start it"})",
                    "application/json");
                return true;
            }
        } else {
            std::string error;
            if (!ensure_worker_locked(requested_model, request_gpu(request), error)) {
                response.status = error == "model or GPU switch requires the active worker to be idle" ? 409 : 503;
                response.set_content(json({{"error", "worker unavailable"}, {"message", error}}).dump(),
                                     "application/json");
                return true;
            }
        }
        worker_port = port_;
    }
    httplib::Client client("127.0.0.1", worker_port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(60 * 60, 0);
    client.set_write_timeout(60 * 60, 0);
    httplib::Headers headers;
    for (const auto& header : request.headers) {
        if (!is_hop_by_hop_header(header.first)) {
            headers.emplace(header.first, header.second);
        }
    }
    httplib::Result result;
    if (request.is_multipart_form_data()) {
        // cpp-httplib parses multipart requests into `form` and intentionally
        // leaves Request::body empty.  Re-encode the parsed fields/files for
        // the child instead of forwarding an empty body with the caller's
        // stale multipart boundary.  This is the Koblem contract for LTX,
        // Wan-VACE and LongCat Avatar (request JSON plus image/audio files).
        httplib::UploadFormDataItems items;
        items.reserve(request.form.fields.size() + request.form.files.size());
        for (const auto& field : request.form.fields) {
            items.push_back({field.second.name, field.second.content, "", ""});
        }
        for (const auto& file : request.form.files) {
            items.push_back({file.second.name, file.second.content, file.second.filename, file.second.content_type});
        }
        if (request.method == "POST") {
            result = client.Post(request.path, headers, items);
        } else if (request.method == "PUT") {
            result = client.Put(request.path, headers, items);
        } else if (request.method == "PATCH") {
            result = client.Patch(request.path, headers, items);
        } else {
            response.status = 405;
            response.set_content(R"({"error":"multipart proxy method is unsupported"})", "application/json");
            return true;
        }
    } else {
        httplib::Request proxied;
        proxied.method = request.method;
        proxied.path = request.path;
        proxied.params = request.params;
        proxied.body = request.body;
        proxied.headers = std::move(headers);
        result = client.send(proxied);
    }
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

void register_worker_supervisor_endpoints(httplib::Server& server, WorkerSupervisor& supervisor,
                                          const std::string& lora_model_dir) {
    // Answered HERE, not proxied. Scanning the LoRA directory (and its `loras.json`) touches only
    // the filesystem, and the catch-all below would have sent it to the worker — cold-starting a
    // ~16 GB CUDA child every time koblem bootstraps the video tab, which is where it reads this
    // list. Exactly the reasoning behind `DELETE /ltx/v1/job` further down.
    //
    // Scanned per request rather than cached: the directory is bind-mounted, so dropping in an
    // adapter or editing the manifest is meant to take effect without a restart, and the cost is a
    // readdir over a handful of files.
    if (!lora_model_dir.empty()) {
        server.Get("/sdapi/v1/loras",
                   [lora_model_dir](const httplib::Request&, httplib::Response& response) {
                       json result = json::array();
                       for (const auto& e : scan_lora_dir(lora_model_dir)) {
                           result.push_back(lora_entry_json(e));
                       }
                       response.set_content(result.dump(), "application/json");
                   });
    }

    server.Get("/health", [&supervisor](const httplib::Request&, httplib::Response& response) {
        const int in_flight = supervisor.in_flight();
        response.set_content(json({{"status", "ok"}, {"busy", in_flight > 0}, {"in_flight", in_flight},
                                   {"draining", supervisor.draining()}, {"loaded", supervisor.loaded()},
                                   {"loaded_model", supervisor.active_model()},
                                   {"worker_pid", supervisor.worker_pid() > 0 ? json(supervisor.worker_pid()) : json(nullptr)}}).dump(),
                             "application/json");
    });
    server.Get("/v1/gpu/status", [&supervisor](const httplib::Request&, httplib::Response& response) {
        response.set_content(json({{"loaded", supervisor.loaded()},
                                   {"gpu", supervisor.loaded() ? json(supervisor.active_gpu()) : json(nullptr)},
                                   {"worker_pid", supervisor.worker_pid() > 0 ? json(supervisor.worker_pid()) : json(nullptr)}}).dump(),
                             "application/json");
    });
    // Drain must BLOCK until in-flight work finishes, because that is the contract Koblem's GPU
    // gate relies on. evict_for_placement() does:
    //     drain (long-timeout client, return value discarded) -> unload with force=true
    // and the force flag deliberately bypasses the "busy" guard in /v1/admin/unload below. So a
    // drain that returns immediately means the gate SIGKILLs a live render: the caller then sees
    // its next job/media poll answered with the supervisor's 410 "worker is unloaded" — observed
    // in prod as a render dying 92 s in. Returning only once the work is done makes the gate's
    // "drain then force" sequence safe, which is what it already assumes ("so an in-flight
    // multi-minute review finishes before we unload").
    //
    // Bounded so a wedged worker cannot pin the gate forever; on timeout we still return and the
    // gate's force-unload proceeds exactly as before. LTX renders at 1920x1088/145f take ~6 min,
    // so the default allows for a comfortably longer chain.
    //
    // Kept as a named lambda (not an inline route body) because the pre-routing hook has to be
    // able to run it too — see the empty-body dispatch there.
    const auto admin_drain = [&supervisor](const httplib::Request&, httplib::Response& response) {
        supervisor.drain();
        int wait_seconds = 1800;
        if (const char* e = std::getenv("SD_DRAIN_WAIT_SECONDS")) {
            wait_seconds = std::max(0, atoi(e));
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(wait_seconds);
        int in_flight = supervisor.in_flight();
        while (in_flight > 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            in_flight = supervisor.in_flight();
        }
        const bool timed_out = in_flight > 0;
        response.set_content(json({{"status", timed_out ? "drain timed out" : "drained"},
                                   {"busy", in_flight > 0},
                                   {"in_flight", in_flight},
                                   {"timed_out", timed_out}}).dump(),
                             "application/json");
    };
    server.Post("/v1/admin/drain", admin_drain);
    const auto admin_unload = [&supervisor](const httplib::Request& request, httplib::Response& response) {
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
        if (!supervisor.unload()) {
            response.status = 503;
            response.set_content(json({{"status", "unload failed"}, {"loaded", supervisor.loaded()},
                                       {"worker_pid", supervisor.worker_pid()},
                                       {"cuda_context_released", false}}).dump(),
                                 "application/json");
            return;
        }
        response.set_content(json({{"status", was_loaded ? "unloaded" : "idle"}, {"unloaded", was_loaded},
                                   {"cuda_context_released", true}, {"worker_pid", nullptr},
                                   {"forced", force && in_flight > 0}}).dump(),
                             "application/json");
    };
    server.Post("/v1/admin/unload", admin_unload);
    const auto admin_load = [&supervisor](const httplib::Request&, httplib::Response& response) {
        supervisor.reopen();
        response.set_content(json({{"status", "ok"}, {"loaded", supervisor.loaded()}}).dump(), "application/json");
    };
    server.Post("/v1/admin/load", admin_load);

    const auto ltx_delete_job = [&supervisor](const httplib::Request& request, httplib::Response& response) {
        // Durable LTX cleanup is metadata/filesystem-only. Let Koblem delete a
        // retired Director bank after /unload without cold-starting a CUDA child.
        if (supervisor.in_flight() > 0) {
            response.status = 409;
            response.set_content(R"({"error":"cannot delete an active LTX job"})", "application/json");
            return;
        }
        bool deleted = false;
        std::string error;
        if (!delete_ltx_bank(request.get_param_value("id"), deleted, error)) {
            response.status = 400;
            response.set_content(json({{"error", error}}).dump(), "application/json");
            return;
        }
        response.set_content(json({{"status", deleted ? "deleted" : "missing"}, {"deleted", deleted}}).dump(),
                             "application/json");
    };
    server.Delete("/ltx/v1/job", ltx_delete_job);

    const auto proxy_request = [&supervisor](const httplib::Request& request, httplib::Response& response) {
        supervisor.proxy(request, response, WorkerSupervisor::request_model(request));
    };

    // Capture the handlers BY VALUE. This function only registers and returns —
    // listen() happens later in main.cpp — so the lambdas above are dead by the time
    // any request arrives. The route table is safe because httplib copies each
    // handler it is given; the pre-routing hook has to copy them too. Capturing them
    // by reference here crashed the supervisor on the first bodyless request. Each
    // copy still holds &supervisor, which is caller-owned and outlives the server.
    server.set_pre_routing_handler([admin_drain, admin_load, admin_unload, ltx_delete_job,
                                    proxy_request](const httplib::Request& request,
                                                   httplib::Response& response) {
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
        // cpp-httplib consumes a request body before ordinary route dispatch, and
        // with NEITHER Content-Length NOR Transfer-Encoding it has no way to know
        // where that body ends — so it waits for a close that a keep-alive client
        // never sends. Every bodyless request to a body-bearing method therefore
        // hangs until the read timeout, which this server sets long enough for a
        // multi-minute render. Both curl (`curl -X POST url`) and reqwest's
        // bodyless `.post(url).send()` send exactly that shape.
        //
        // This hook runs BEFORE the body read, so it is the only place the request
        // can still be rescued. It used to rescue /v1/admin/unload alone, which
        // silently left every other bodyless caller hanging:
        //   - POST /v1/admin/drain          — koblem's GPU gate evicts with
        //     "drain (900s client) then force unload" and its drain carries no
        //     body. The drain never returned, so the force unload after it never
        //     ran, the worker was never killed, and the gate kept the service's
        //     whole VRAM reservation while reporting it disabled. Symptom: an idle,
        //     not-busy engine that cannot be evicted and a
        //     `POST /api/v1/gpu/services/{name}` that never answers.
        //   - POST /v1/admin/load           — the documented way to clear a stuck
        //     drain flag, itself stuck.
        //   - POST /sdcpp/v1/jobs/{id}/cancel — every cancel path in koblem
        //     (flux2.rs, krea2.rs, ltx_video.rs) posts it with no body, so a
        //     cancel hung and took its GPU guard with it.
        //   - DELETE /ltx/v1/job            — bodyless by nature.
        //
        // So dispatch generically rather than per-route: anything with no body gets
        // answered here, admin routes inline and everything else through the same
        // proxy the catch-all would have used. Requests that DO carry a body (or
        // are chunked) fall through untouched, so JSON and multipart still reach
        // the regular handlers intact.
        const bool has_length = !request.get_header_value("Content-Length").empty();
        const bool chunked = !request.get_header_value("Transfer-Encoding").empty();
        const bool body_bearing = request.method == "POST" || request.method == "PUT" ||
                                  request.method == "PATCH" || request.method == "DELETE";
        if (body_bearing && !has_length && !chunked) {
            if (request.method == "POST" && request.path == "/v1/admin/drain") {
                admin_drain(request, response);
            } else if (request.method == "POST" && request.path == "/v1/admin/load") {
                admin_load(request, response);
            } else if (request.method == "POST" && request.path == "/v1/admin/unload") {
                admin_unload(request, response);
            } else if (request.method == "DELETE" && request.path == "/ltx/v1/job") {
                ltx_delete_job(request, response);
            } else {
                proxy_request(request, response);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    constexpr const char* kAnyPath = R"(/.*)";
    server.Get(kAnyPath, proxy_request);
    server.Post(kAnyPath, proxy_request);
    server.Put(kAnyPath, proxy_request);
    server.Patch(kAnyPath, proxy_request);
    server.Delete(kAnyPath, proxy_request);
}
