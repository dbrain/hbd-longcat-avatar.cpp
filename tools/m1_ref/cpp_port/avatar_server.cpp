// avatar_server — the HTTP front end for avatar_pipeline.hpp.
//
// It links libavatar_pipeline.a and includes avatar_pipeline.hpp, and NOTHING ELSE from the model
// stack: no ggml, no CUDA, no glTF in this translation unit. That was the point of the seam.
//
// ============================================================================================
// THE SHAPE, AND WHICH FLEET SERVICE IT COPIES
// ============================================================================================
// Modelled on spar3d-riganything's supervisor (kobbler/docker/spar3d-riganything/supervisor.py) —
// the box's other image->rigged-asset service — for the JOB LIFECYCLE, and on the sd.cpp fork's
// server (examples/server/routes_gpu_sharing.cpp) for the ADMIN + /health surface. Both were
// followed rather than improved on, so a dashboard or a client written against either reads this:
//
//   POST /v1/rig|/v1/motion|/v1/animate|/v1/render   -> 202 {id, status:"queued", ...}
//   POST .../{path}?wait=1                           -> blocks, 200 with the finished job
//   GET  /v1/jobs/{id}                               -> the job, polled
//   GET  /v1/jobs/{id}/result                        -> the ARTIFACT BYTES (GLB or clip JSON)
//   POST /v1/jobs/{id}/cancel                        -> trips the CancelToken; lands in <= 7 s
//   GET  /health          busy / in_flight / queued / waiting_for_gpu_lock, SEPARATELY
//   GET  /v1/gpu/status   residency + free VRAM (NOT activity — /health answers while unloaded)
//   GET  /v1/metrics      accepted / completed / failed / cancelled counters
//   POST /v1/admin/drain | /v1/admin/load | /v1/admin/unload
// `/rig`, `/motion`, `/animate` and `/render` are registered as unversioned aliases of the /v1
// paths, because that is the surface the service was specified against.
//
// WHY THREE ENDPOINTS AND NOT ONE. The economics are the product: a rig is ~450 s and is cached per
// character, a motion clip is ~11 s, and the retarget is 0.22 s. A re-roll is therefore motion +
// animate = ~12 s with NO re-rigging, and a single fat endpoint would spend 450 s to change the
// prompt. /v1/render exists for the one-shot case and is the exception, not the interface.
//
// EVERY STAGE IS RE-RUNNABLE AGAINST A PREVIOUS RESULT, which is the other half of driving this
// step by step. Every job publishes stable paths for its intermediates — `stage_dir` (the geometry),
// `rig_cache` (the skeleton), and one GLB per LOD tier — and every request field that names one of
// those accepts EITHER a path OR the id of an earlier job:
//     {"geometry_from":"<rig job id>"}   re-bake + re-rig an existing geometry, ~60-180 s, no
//                                        diffusion, no DC. This is `--from-refined`.
//     {"rig_cache_from":"<rig job id>"}  reuse the SAME skeleton — no rig decode at all.
//     {"rig":"<rig job id>","tier":"game"}   animate a tier of an earlier rig.
//     {"clips":["<motion job id>", ...]}     retarget clips generated earlier.
// So "re-rig THIS geometry" and "type another prompt against THIS rig" never redo the 450 s.
//
// THE LOD LADDER. A rig job returns FOUR tiers — hero 220k / high 100k / medium 40k / game 15k —
// and they share ONE skeleton through `--rig-cache`, so a clip authored against the hero plays on
// the game tier unchanged. Tiers after the first cost a bake, not a rig: the geometry comes from
// this job's own stage dir and the skeleton from its own cache. `{"tiers":["hero"]}` opts out.
//
// THE QC GATES ARE ADVISORY. A rig that fails the pose gate is still delivered, with the verdict
// attached (`result.gate`, and a `warnings` array), because a rig can be judged decent by eye while
// the gate reads 25.9 — the defects are cosmetic spikes, and refusing to hand back an asset the
// owner would have shipped is the wrong failure. The ONE exception is `skin_ok`: a rig with zero
// skin weights deforms NOTHING and would poison the shared rig cache for every tier, so the stage
// itself still fails closed on that (image_to_rig.cpp, `--allow-zero-skin` to override).
//
// ============================================================================================
// CONCURRENCY. One thread per admitted job, and the ENGINE's queue is the authority.
// ============================================================================================
// Each job gets a thread that opens an avatar::Engine::Session and therefore blocks in the
// engine's own FIFO. That is deliberate: it means /health's `queued` and `waiting_for_gpu_lock`
// come straight out of Engine::health() and describe the real thing, rather than a second queue
// of our own that could disagree with it. Threads are bounded by LONGCAT_RIG_MAX_PENDING.
//
// The exception is /v1/animate, which follows the library: Engine::animate() takes NO session
// because it is CPU-only arithmetic, so a 0.22 s retarget is not parked behind a 450 s rig. See
// the note at run_animate() for the one wrinkle that creates.
//
// ============================================================================================
// VRAM, WHICH IS THE FAILURE THIS SERVICE ACTUALLY HAS
// ============================================================================================
// The rig stage peaks ~10.7 GB of a 12 GB 3060 AND DIES RATHER THAN DEGRADING, and it shares the
// card with a resident LLM that does not take our flock. So the floor is checked TWICE and neither
// check is the entrypoint's:
//   * at SUBMIT, but only when the engine is idle — if a render of ours is running, low free VRAM
//     is us, and the request must queue rather than be refused;
//   * at RUN, once we hold the session and the flock, immediately before build_rig().
// Both fail with `error_code: "insufficient_vram"` and a message naming the number. The point is
// that the operator learns this in milliseconds instead of as a CUDA OOM seven minutes in.
//
// ============================================================================================
// WORKER ISOLATION — THE PROCESS IS TWO PROCESSES NOW. Read avatar_worker.hpp first.
// ============================================================================================
// ggml's CUDA allocator pools every block it ever frees, so `release_gpu()` — and therefore
// /v1/admin/unload — could never return VRAM to the driver: measured, 122 MiB fresh, 1558 MiB after
// ONE job with `loaded: false`, and a container restart the only way back. So this binary now runs
// as a PAIR, exactly as the sd.cpp fork's GPU gate does:
//
//   PARENT (the default)   this file, with `engine == nullptr`. It never constructs avatar::Engine,
//                          so it never creates a CUDA context and holds ZERO MiB on the card. It
//                          owns the public listener, the job manager, admission, the VRAM floor
//                          (read via NVML — see the 🔴 in avatar_worker.hpp), idle-unload, and
//                          every route. Each GPU job is ONE synchronous `?wait=1` request proxied
//                          to the child, whose result it records as its own.
//   CHILD  (--worker)      this file, with `engine != nullptr`: today's server, byte for byte the
//                          same code path, on a private loopback port. It owns the Engine, the CUDA
//                          context, the ggml pools, and the GPU flock.
//
// `POST /v1/admin/unload` SIGKILLs the child. That is a real context teardown, so the card comes
// back to the parent-only baseline of 0 MiB.
//
// WHY THE PARENT KEEPS THE JOBS rather than proxying /v1/jobs as well: killing the child must not
// erase job history. `geometry_from: <job id>`, `rig_cache_from`, `animate {rig: <job id>}` and
// `GET /v1/jobs/{id}/result` all have to keep working across an unload, and the artifacts are on a
// shared volume that outlives any child. The parent therefore MINTS the job id and hands it to the
// child as `_job_id`, so both processes name the same job and the same `/out/jobs/<id>/` directory,
// and every path the child publishes is one the parent can serve afterwards.
//
// LONGCAT_RIG_WORKER_ISOLATION=0 (or --no-worker-isolation) restores the single-process behaviour,
// pooled VRAM and all. It exists to bisect against, not to run.
#include "avatar_pipeline.hpp"

#include "httplib.h"
#include "json.hpp"

#include "avatar_worker.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
namespace {

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
double wall_now() { return (double)::time(nullptr); }

// Parent and child both log to the same inherited stderr, so they say which one they are. Without
// this every `docker logs` line is ambiguous the moment isolation is on.
const char* g_log_tag = "avatar-server";

void log_line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log_line(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[%s] %s\n", g_log_tag, buf);
    std::fflush(stderr);
}

void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur.push_back(p[i]);
        if (p[i] == '/' || i + 1 == p.size()) ::mkdir(cur.c_str(), 0755);
    }
}

bool is_regular_file(const std::string& p) {
    struct stat st;
    return !p.empty() && ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool read_file(const std::string& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

bool write_file(const std::string& p, const std::string& bytes) {
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write(bytes.data(), (std::streamsize)bytes.size());
    return (bool)f;
}

// ---------------------------------------------------------------------------
// Submit-time validation
// ---------------------------------------------------------------------------
// THE RULE: anything the caller can fix by changing the request is a 400 WITH THE REASON, decided
// before a job exists. Not a job that starts, occupies the card, and fails.
//
// This is not tidiness. The product is a step-wise pipeline a person drives — upload, look, re-run
// a stage — so the person IS the caller, and "rc 1" arriving after a delay with the real reason in
// a container log they cannot read is a dead end for them. Measured the hard way: a mislabelled
// file (`miku.png`, actually WebP) reached the geometry stage and died there with `unknown image
// type` visible only in `docker logs`.
//
// Everything here is cheap and GPU-free: magic bytes, not a decode; a stat, not a read.

// What the image actually IS, by magic bytes. stb_image — which is what the pipeline loads with —
// reads PNG/JPEG/BMP/PSD/TGA/GIF/HDR/PIC/PNM and nothing else, so the useful answer names both the
// format found and whether it can be used.
struct ImageKind {
    std::string name;       // "PNG", "WebP", ...; "" = unrecognised
    bool supported = false;
};
ImageKind detect_image(const std::string& d) {
    auto has = [&](size_t off, const char* sig, size_t n) {
        return d.size() >= off + n && std::memcmp(d.data() + off, sig, n) == 0;
    };
    if (has(0, "\x89PNG\r\n\x1a\n", 8))                      return {"PNG", true};
    if (has(0, "\xff\xd8\xff", 3))                           return {"JPEG", true};
    if (has(0, "BM", 2))                                     return {"BMP", true};
    if (has(0, "GIF87a", 6) || has(0, "GIF89a", 6))          return {"GIF", true};
    if (has(0, "8BPS", 4))                                   return {"PSD", true};
    if (has(0, "#?RADIANCE", 10) || has(0, "#?RGBE", 6))     return {"Radiance HDR", true};
    if (d.size() >= 3 && d[0] == 'P' && d[1] >= '1' && d[1] <= '6') return {"PNM", true};
    // The ones that will NOT load, named explicitly — a caller who sent one needs to know which.
    if (has(0, "RIFF", 4) && has(8, "WEBP", 4))              return {"WebP", false};
    if (has(4, "ftypavif", 8) || has(4, "ftypavis", 8))      return {"AVIF", false};
    if (has(4, "ftypheic", 8) || has(4, "ftypheix", 8) ||
        has(4, "ftypmif1", 8))                               return {"HEIC", false};
    if (has(0, "II*\0", 4) || has(0, "MM\0*", 4))            return {"TIFF", false};
    if (has(0, "<?xml", 5) || has(0, "<svg", 4))             return {"SVG", false};
    if (has(0, "%PDF", 4))                                   return {"PDF", false};
    if (has(0, "\x1a\x45\xdf\xa3", 4))                       return {"Matroska/WebM video", false};
    return {"", false};
}

// The first bytes of a file, for the magic check. Also answers "is it readable at all".
bool read_header(const std::string& path, std::string& out, size_t n = 32) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.resize(n);
    f.read(&out[0], (std::streamsize)n);
    out.resize((size_t)f.gcount());
    return true;
}

// Reject what stb_image cannot read, naming what it actually is.
bool image_bytes_ok(const std::string& head, const std::string& what, std::string& err) {
    if (head.empty()) { err = what + " is empty"; return false; }
    const ImageKind k = detect_image(head);
    if (k.supported) return true;
    if (!k.name.empty()) {
        err = what + " is a " + k.name + " file, which this pipeline cannot read. Send PNG, JPEG, "
              "BMP, GIF, PSD, TGA, HDR or PNM. (A file NAMED .png is not necessarily a PNG — "
              "convert it, do not rename it.)";
        return false;
    }
    err = what + " is not a recognised image (first bytes match no supported format). Send PNG, "
          "JPEG, BMP, GIF, PSD, TGA, HDR or PNM.";
    return false;
}

// Is this media field a trusted absolute PATH rather than a base64 payload?
//
// Lifted verbatim in spirit from the fleet's `ltx_source_is_path` (examples/server/async_jobs.h),
// including WHY it asks the filesystem: base64-encoded JPEG begins "/9j/", so a leading '/' alone
// reads a payload as a path. A real path exists; base64 never does.
bool source_is_path(const std::string& s) {
    if (s.empty() || s[0] != '/' || s.size() >= 4096 || s.find('\n') != std::string::npos) return false;
    return is_regular_file(s);
}

// RFC 4648 base64 decode. Tolerates whitespace and a `data:...;base64,` prefix, because callers
// paste both. Returns false on a character that is neither.
bool base64_decode(const std::string& in, std::string& out) {
    static int8_t table[256];
    static bool ready = false;
    if (!ready) {
        std::memset(table, -1, sizeof(table));
        const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) table[(unsigned char)abc[i]] = (int8_t)i;
        table[(unsigned char)'-'] = 62;   // URL-safe alphabet, accepted on input
        table[(unsigned char)'_'] = 63;
        ready = true;
    }
    size_t start = 0;
    const size_t comma = in.find(',');
    if (in.compare(0, 5, "data:") == 0 && comma != std::string::npos) start = comma + 1;
    out.clear();
    out.reserve((in.size() - start) * 3 / 4 + 3);
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = start; i < in.size(); i++) {
        const unsigned char c = (unsigned char)in[i];
        if (c == '=' ) break;
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int8_t v = table[c];
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char)((acc >> bits) & 0xff));
        }
    }
    return !out.empty();
}

std::string getenv_str(const char* k, const std::string& dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : dflt;
}
int getenv_int(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : dflt;
}

}  // namespace

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------
namespace {

struct Job {
    std::string id;
    std::string kind;                  // "rig" | "motion" | "animate" | "render"
    std::string status = "queued";     // queued | running | completed | failed | cancelled
    double created_at = wall_now();
    double started_at = 0;
    double completed_at = 0;
    double seconds = 0;                // wall time of the stage(s), measured
    std::string error, error_code;
    std::string result_path;           // the artifact on disk
    std::string result_mime;
    std::string dir;
    json request;                      // the parsed request, WITHOUT any base64 payload
    json result;                       // stage results (gate verdict, frames, timings, ...)
    std::shared_ptr<avatar::CancelToken> cancel = std::make_shared<avatar::CancelToken>();
    std::atomic<bool> done{false};
};

using JobPtr = std::shared_ptr<Job>;

struct Manager {
    std::mutex mx;
    std::map<std::string, JobPtr> jobs;
    std::deque<std::string> order;      // insertion order, for eviction + listing
    uint64_t next_id = 0;
    size_t max_records = 512;           // records, NOT artifacts: files are never auto-deleted
    long accepted = 0, completed = 0, failed = 0, cancelled = 0, rejected = 0;
};

// Job ids are `<seconds since epoch in hex><counter>`, monotone and sortable, matching the fleet's
// habit of an id you can eyeball for age. Never random: a duplicate would silently overwrite.
std::string new_job_id(Manager& m) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%08lx%04llx", (unsigned long)::time(nullptr),
                  (unsigned long long)(++m.next_id & 0xffff));
    return buf;
}

json job_public(const Job& j) {
    json b = {
        {"id", j.id},
        {"kind", j.kind},
        {"status", j.status},
        {"created_at", j.created_at},
        {"started_at", j.started_at},
        {"completed_at", j.completed_at},
        {"seconds", j.seconds},
        {"request", j.request},
    };
    if (!j.result.is_null()) b["result"] = j.result;
    if (!j.result_path.empty()) {
        b["result_path"] = j.result_path;
        // Published because the isolation parent has to reproduce it: it records a child's finished
        // job as its own, and the artifact's media type is not derivable from the job kind alone
        // (a `render` delivers a GLB, a `motion` delivers clip JSON, and a future stage may differ).
        b["result_mime"] = j.result_mime;
        // The bytes are fetched, never inlined: a rigged GLB is tens of MB and base64 in a poll
        // body is the transport mistake `docs/media-transport.md` exists to undo.
        b["result_url"] = "/v1/jobs/" + j.id + "/result";
    }
    if (!j.error.empty()) b["error"] = j.error;
    if (!j.error_code.empty()) b["error_code"] = j.error_code;
    return b;
}

// Drop the OLDEST finished records once we are over the cap. Only the record: the artifact on disk
// stays, because a sweep that deletes files a client is still fetching is a bug this fleet has
// already paid for once (the chain-timeout incident deleted a render's inputs mid-render).
void evict_locked(Manager& m) {
    while (m.order.size() > m.max_records) {
        bool evicted = false;
        for (auto it = m.order.begin(); it != m.order.end(); ++it) {
            auto ji = m.jobs.find(*it);
            if (ji == m.jobs.end()) { m.order.erase(it); evicted = true; break; }
            if (ji->second->done.load()) {
                m.jobs.erase(ji);
                m.order.erase(it);
                evicted = true;
                break;
            }
        }
        if (!evicted) break;   // everything left is live
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------
namespace {

struct Config {
    std::string host = "0.0.0.0";
    int         port = 8102;
    std::string jobs_dir = "/out/jobs";
    int         min_free_mib = 0;      // 0 = do not check
    int         rig_retries = 2;       // the shipped default; see the block at parse time
    int         max_pending = 32;
};

// EXACTLY ONE of `engine` and `worker` is set, and which one it is IS the mode:
//   engine != nullptr  we own the CUDA context: the --worker child, or the whole service when
//                      isolation is switched off.
//   worker != nullptr  we are the CUDA-free parent and the context lives in a child process.
struct State {
    avatar::Engine*  engine = nullptr;
    avsup::Worker*   worker = nullptr;
    bool worker_mode = false;          // we are the child (accept `_job_id` from our parent)
    int  idle_unload_seconds = 0;      // parent-owned when `worker` is set
    Config cfg;
    Manager jm;
    std::atomic<bool> stopping{false};
    std::atomic<bool> draining{false}; // parent-side admission gate (the child has the Engine's)
    double started_s = 0;
};

State* g_state = nullptr;
httplib::Server* g_svr = nullptr;

void set_json(httplib::Response& res, const json& body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

// How many jobs are admitted but not finished. Bounds the thread count and is what the
// `max_pending` refusal is measured against.
int live_jobs(Manager& m) {
    int n = 0;
    for (const auto& [_, j] : m.jobs)
        if (!j->done.load()) n++;
    return n;
}

// ---------------------------------------------------------------------------
// The backend, whichever process actually owns the card
// ---------------------------------------------------------------------------

// GET something small off the child. Short timeouts: these are status reads on a loopback socket,
// and /health in particular is answered by the child without taking any lock, even mid-render.
bool worker_get(State& st, const char* path, json& out) {
    if (!st.worker) return false;
    const int port = st.worker->port();
    if (port <= 0) return false;
    httplib::Client c("127.0.0.1", port);
    c.set_connection_timeout(1, 0);
    c.set_read_timeout(3, 0);
    const auto r = c.Get(path);
    if (!r || r->status != 200) return false;
    try { out = json::parse(r->body); } catch (...) { return false; }
    return true;
}

// Free VRAM on the pinned card, as the driver sees it right now. -1 = could not ask.
//
// 🔴 The parent asks NVML, NEVER the engine: Engine::gpu_status() calls cudaMemGetInfo, and that
// call alone creates the CUDA primary context (~122 MiB) in a process whose entire purpose is not
// to have one.
long free_vram_mib(State& st) {
    if (st.engine) return st.engine->gpu_status().mem_free_mb;
    return avsup::nvml_free_mib();
}

// /health's fields, from whichever process can answer them. In parent mode the card-facing
// half (busy / queued / waiting_for_gpu_lock) comes from the child, because the child's Engine is
// the thing that queues and the thing that takes the flock; the rest is ours.
avatar::Health backend_health(State& st) {
    if (st.engine) return st.engine->health();
    avatar::Health h;
    h.draining = st.draining.load();
    // `loaded` now means "a CUDA-owning worker exists". That is the honest reading in this mode and
    // it is what /v1/admin/unload acts on — the motion DiT's residency is no longer a thing the
    // parent can see, or a thing anyone can act on separately.
    h.loaded = st.worker && st.worker->alive();
    json ch;
    if (h.loaded && worker_get(st, "/health", ch)) {
        h.busy                 = ch.value("busy", false);
        h.queued               = ch.value("queued", 0);
        h.in_flight            = ch.value("in_flight", 0);
        h.waiting_for_gpu_lock = ch.value("waiting_for_gpu_lock", false);
        h.loaded_model         = ch.value("loaded_model", std::string());
        h.renders_done         = ch.value("renders_done", (long)0);
        h.last_render_sec      = ch.value("last_render_sec", 0.0);
        h.fatal                = ch.value("fatal", std::string());
    }
    h.uptime_sec = now_s() - st.started_s;
    h.idle_sec = st.worker ? st.worker->idle_seconds() : 0.0;
    h.idle_unload_seconds = st.idle_unload_seconds;
    h.keep_resident = true;
    h.status = h.fatal.empty() ? "ok" : "degraded";
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------
namespace {

// The body, whatever shape it arrived in:
//   application/json                    -> parsed
//   multipart/form-data                 -> the `json` part parsed (if present); files kept separately
//   image/*  (or anything else binary)  -> {} and the raw body treated as the image
bool parse_body(const httplib::Request& req, json& body, std::string& err) {
    body = json::object();
    const std::string ct = req.get_header_value("Content-Type");
    if (ct.find("multipart/form-data") != std::string::npos) {
        if (req.form.has_file("json")) {
            try {
                body = json::parse(req.form.get_file("json").content);
            } catch (const std::exception& e) {
                err = std::string("bad json part: ") + e.what();
                return false;
            }
        }
        return true;
    }
    if (ct.find("application/json") != std::string::npos || (!req.body.empty() && req.body[0] == '{')) {
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            err = std::string("bad json body: ") + e.what();
            return false;
        }
    }
    return true;
}

// Resolve an image to a path ON DISK, writing the payload into the job dir if it arrived inline.
// Order: multipart file part, then the `image` field (path or base64), then a raw binary body.
bool resolve_image(const httplib::Request& req, const json& body, const std::string& job_dir,
                   std::string& out_path, std::string& err) {
    if (req.form.has_file("image")) {
        const auto f = req.form.get_file("image");
        if (!image_bytes_ok(f.content, "the uploaded image", err)) return false;
        out_path = job_dir + "/input";
        if (!write_file(out_path, f.content)) { err = "cannot write " + out_path; return false; }
        return true;
    }
    if (body.contains("image") && body["image"].is_string()) {
        const std::string s = body["image"].get<std::string>();
        if (s.empty()) { err = "`image` is an empty string"; return false; }
        // A string that MEANS a path gets a path's error message. Falling through to "could not
        // decode base64" for a typo'd filename sends the caller looking in the wrong place —
        // base64 never starts with '/', so this is a safe thing to be definite about.
        const bool looks_like_path = s[0] == '/' && s.size() < 4096 && s.find('\n') == std::string::npos;
        if (looks_like_path && !is_regular_file(s)) {
            err = "no such file inside the container: " + s +
                  " (paths are resolved in the CONTAINER's filesystem — /out is the job volume; "
                  "send base64 or a multipart part for a file the container cannot see)";
            return false;
        }
        if (source_is_path(s)) {
            std::string head;
            if (!read_header(s, head)) { err = "cannot read " + s; return false; }
            if (!image_bytes_ok(head, s, err)) return false;
            out_path = s;
            return true;
        }
        std::string bytes;
        if (!base64_decode(s, bytes)) {
            err = "`image` is neither a readable absolute path nor decodable base64";
            return false;
        }
        if (!image_bytes_ok(bytes, "the decoded image", err)) return false;
        out_path = job_dir + "/input";
        if (!write_file(out_path, bytes)) { err = "cannot write " + out_path; return false; }
        return true;
    }
    const std::string ct = req.get_header_value("Content-Type");
    if (!req.body.empty() && ct.compare(0, 6, "image/") == 0) {
        if (!image_bytes_ok(req.body, "the request body", err)) return false;
        out_path = job_dir + "/input";
        if (!write_file(out_path, req.body)) { err = "cannot write " + out_path; return false; }
        return true;
    }
    err = "no image: send `image` as an absolute path or base64, a multipart `image` file part, or a raw image/* body";
    return false;
}

JobPtr find_job(State& st, const std::string& id) {
    std::lock_guard<std::mutex> lk(st.jm.mx);
    auto it = st.jm.jobs.find(id);
    return it == st.jm.jobs.end() ? nullptr : it->second;
}

// One tier's GLB out of a finished rig/render job. `tier` empty = the hero (the first tier built).
bool tier_glb(const Job& j, const std::string& tier, std::string& out_path, std::string& err) {
    if (!j.result.contains("tiers") || !j.result["tiers"].is_array() || j.result["tiers"].empty()) {
        err = "job " + j.id + " built no LOD tiers";
        return false;
    }
    for (const auto& t : j.result["tiers"]) {
        if (tier.empty() || t.value("tier", std::string()) == tier) {
            out_path = t.value("glb", std::string());
            return !out_path.empty();
        }
    }
    std::string have;
    for (const auto& t : j.result["tiers"]) have += (have.empty() ? "" : ", ") + t.value("tier", std::string());
    err = "job " + j.id + " has no tier '" + tier + "' (has: " + have + ")";
    return false;
}

// A rig GLB, named either directly (path) or by the id of a completed rig/render job, optionally at
// a named LOD tier. Resolving job ids here is what makes "rig once, re-roll the motion" a two-call
// loop for the client, and what makes "animate the game tier of that rig" one field.
bool resolve_rig(State& st, const std::string& s, const std::string& tier, std::string& out_path,
                 std::string& err) {
    if (s.empty()) { err = "animate needs a `rig` (a path, or the id of a completed rig job)"; return false; }
    if (is_regular_file(s)) { out_path = s; return true; }
    JobPtr j = find_job(st, s);
    if (!j) { err = "rig '" + s + "' is neither a readable file nor a known job id"; return false; }
    if (j->status != "completed") { err = "rig job " + s + " is " + j->status + ", not completed"; return false; }
    if (j->kind == "motion") { err = "job " + s + " is a motion clip, not a rig"; return false; }
    if (!tier_glb(*j, tier, out_path, err)) return false;
    if (!is_regular_file(out_path)) { err = "job " + s + " has no rig GLB at " + out_path; return false; }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// The stages
// ---------------------------------------------------------------------------
namespace {

json gate_json(const avatar::RigGateResult& g) {
    return json{
        {"draws", g.draws},
        // FALSE means no draw cleared the acceptance predicate and the best-ranked one was shipped
        // anyway. It is surfaced rather than swallowed on purpose: it is a verdict computed before
        // the write and it CANNOT be recovered by reading the GLB back.
        {"accepted", g.accepted},
        {"skin_ok", g.skin_ok},
        {"humanoid_gate_ok", g.humanoid_gate_ok},
        {"generic_gate_ok", g.generic_gate_ok},
        {"named_core", g.selector_named_core},
        {"pose_gate_ran", g.pose_gate_ran},
        {"pose_gate_pass", g.pose_gate_pass},
        {"pose_gate_worst", g.pose_gate_worst},
        {"summary", g.selector_summary},
    };
}

// Publishing a result is a WRITE that a concurrent GET /v1/jobs/{id} may be reading, so it takes the
// same lock the readers do. (A json assignment is not atomic and the reader copies the whole node.)
void publish(State& st, JobPtr j, json result, const std::string& path, const char* mime,
             double seconds) {
    std::lock_guard<std::mutex> lk(st.jm.mx);
    j->result = std::move(result);
    j->result_path = path;
    j->result_mime = mime;
    j->seconds = seconds;
}

void finish(State& st, JobPtr j, bool ok, const std::string& err, const std::string& code) {
    std::lock_guard<std::mutex> lk(st.jm.mx);
    j->completed_at = wall_now();
    if (ok) {
        j->status = "completed";
        st.jm.completed++;
    } else if (code == "cancelled") {
        j->status = "cancelled";
        j->error = err;
        j->error_code = code;
        st.jm.cancelled++;
    } else {
        j->status = "failed";
        j->error = err;
        j->error_code = code;
        st.jm.failed++;
    }
    j->done.store(true);
    evict_locked(st.jm);
    log_line("job %s %s in %.1fs%s", j->id.c_str(), j->status.c_str(), j->seconds,
             j->error.empty() ? "" : (" — " + j->error).c_str());
}

// The VRAM floor, checked with the card in hand. `where` is only for the message.
bool vram_ok(State& st, std::string& err) {
    if (st.cfg.min_free_mib <= 0) return true;
    long free_mib = free_vram_mib(st);
    if (free_mib < 0) return true;   // driver would not answer; do not invent a refusal
    if (free_mib >= st.cfg.min_free_mib) return true;

    // ---- the parent's second look --------------------------------------------------------------
    // This is what worker isolation BUYS at admission time, and it is the answer to the failure
    // that made this floor a brick: an idle child still shows every block ggml ever pooled as used,
    // so the shortfall may be entirely ours. The parent can now settle that instead of guessing —
    // kill the idle worker (a real CUDA teardown) and re-read the card. If the memory was ours we
    // now fit, and the next request pays a ~seconds cold start instead of a 503 it did not deserve.
    // Nothing is killed while a lease is held, so this can never touch a running render.
    if (!st.engine && st.worker && st.worker->kill_if_idle()) {
        const long after = free_vram_mib(st);
        log_line("VRAM floor: %ld MiB free < %d — killed the idle worker; %ld MiB free now",
                 free_mib, st.cfg.min_free_mib, after);
        if (after >= st.cfg.min_free_mib) return true;
        if (after >= 0) free_mib = after;
    }
    if (!st.engine) {
        err = "only " + std::to_string(free_mib) + " MiB free on the pinned card, need >= " +
              std::to_string(st.cfg.min_free_mib) +
              ". This process holds no CUDA context and its GPU worker has already been killed and "
              "re-measured, so the shortfall is a CO-TENANT, not us — kobbler-llama-server-1 takes "
              "~9 GB of the 3060 without taking our lock. Free the card or lower "
              "LONGCAT_RIG_MIN_FREE_MIB.";
        return false;
    }
    // Name OUR OWN retained pool, because after one job it is usually the whole reason we are here
    // and the old message blamed llama-server unconditionally. ggml's CUDA allocator keeps freed
    // blocks for reuse, so the driver reports ~1.4 GB used by this process while nothing is loaded;
    // /v1/admin/unload cannot reclaim it (it only drops the motion DiT) and a container restart is
    // the only way back to the baseline. That memory IS available to our next job — the driver just
    // does not say so — which is why the floor must be set against the measured PEAK and not against
    // free-as-the-driver-sees-it. Measured peaks on this configuration: 6055 MiB (four-tier ladder)
    // and 6959 MiB (re-rig from cached geometry).
    const long retained = (long)st.engine->gpu_status().mem_free_mb < 0
                              ? 0
                              : (long)st.engine->peak_vram_used_mib() -
                                    (long)st.engine->baseline_vram_used_mib();
    err = "only " + std::to_string(free_mib) + " MiB free on the pinned card, need >= " +
          std::to_string(st.cfg.min_free_mib) +
          ". This process has already peaked at " + std::to_string(retained > 0 ? retained : 0) +
          " MiB above its baseline and ggml keeps that pool for reuse, so part of the shortfall may "
          "be our own retained memory rather than a co-tenant — check `nvidia-smi` before assuming. "
          "Otherwise free the card (kobbler-llama-server-1 takes ~9 GB of the 3060 without taking "
          "our lock) or lower LONGCAT_RIG_MIN_FREE_MIB.";
    return false;
}

// The LOD ladder. One skeleton, four decimation targets — a clip authored on the hero plays on the
// game tier unchanged BECAUSE they share the skeleton, which is what `--rig-cache` is for. Tier 0 is
// the only one that pays for geometry and for the rig decode; the rest resume from tier 0's stage
// dir and read its cache, which is a bake (~60-180 s), not a rig.
struct Tier {
    std::string name;
    int decimate;
};
const std::vector<Tier>& default_tiers() {
    static const std::vector<Tier> t = {
        {"hero", 220000}, {"high", 100000}, {"medium", 40000}, {"game", 15000},
    };
    return t;
}

// The rest of the caller-fixable surface, checked before a job exists. Each one of these otherwise
// fails INSIDE a stage: an unknown variant after the retarget has loaded the rig, a zero step count
// as a sampler that produces nothing, an empty prompt as a clip of noise.
bool validate_common(const json& b, bool needs_prompt, std::string& err) {
    if (needs_prompt && !b.value("uncond", false)) {
        if (!b.contains("prompt") || !b["prompt"].is_string()) {
            err = "`prompt` is required (or `uncond: true` for the checkpoint's trained null conditioning)";
            return false;
        }
        std::string p = b["prompt"].get<std::string>();
        const size_t s = p.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) { err = "`prompt` is empty"; return false; }
    }
    // Ranges, named with the value that was sent — a bare "invalid" makes the caller guess which
    // field. The bounds are generous: this is a guard against 0 and typos, not a policy.
    struct { const char* k; double lo, hi; } nums[] = {
        {"seconds", 0.1, 60.0}, {"steps", 1, 500}, {"cfg", 0.0, 50.0},
        {"smooth", 0, 100}, {"texsize", 16, 8192}, {"decimate", 100, 5000000},
        {"resolution", 0, 2032}, {"rig_retries", 0, 16}, {"rig_extra_retries", -1, 16},
        // Seeds are the retry knob (see fill_rig_request), so a negative one must not silently
        // wrap into a different draw than the caller asked for.
        {"seed", 0, 4294967295.0}, {"rig_seed", 0, 4294967295.0}, {"geo_seed", 0, 2147483647.0},
    };
    for (const auto& n : nums) {
        if (!b.contains(n.k) || !b[n.k].is_number()) continue;
        const double v = b[n.k].get<double>();
        if (v < n.lo || v > n.hi) {
            err = std::string("`") + n.k + "` is " + std::to_string(v) + "; expected " +
                  std::to_string(n.lo) + " to " + std::to_string(n.hi);
            return false;
        }
    }
    // `resolution` has a SHAPE as well as a range, and the typed seam bypasses the CLI parser that
    // used to enforce it: avatar_pipeline sets `o.geo_resolution` directly, so `--resolution`'s
    // "multiple of 16" check never runs on an API request. Without this, `resolution: 777` is
    // accepted, occupies the card for the whole diffusion stage, and dies in the DC — which peels
    // factors of two off the value and refuses a coarsest level above 128 (narrow_band_dc_cuda.cu:
    // KEY_MAX_COORD 2047, MAX_LADDER_START 128). Every multiple of 16 in 32..2032 peels to <= 127,
    // so the multiple-of-16 rule is exactly the DC's admissible set, not an approximation of it.
    if (b.contains("resolution") && b["resolution"].is_number()) {
        const int r = b["resolution"].get<int>();
        if (r != 0 && (r % 16 != 0 || r < 32)) {
            err = "`resolution` is " + std::to_string(r) +
                  "; expected 0 (the default lattice) or a multiple of 16 from 32 to 2032. 1024 is "
                  "the default; 1536 is the only other value with any measured result, and it is "
                  "per-subject (better geometry, worse rig on some characters).";
            return false;
        }
    }
    return true;
}

// The retarget variants, checked here because animate() only learns the name is wrong after it has
// loaded the rig and the clips.
bool validate_variant(const json& b, std::string& err) {
    if (!b.contains("variant")) return true;
    if (!b["variant"].is_string()) { err = "`variant` must be a string"; return false; }
    const std::string v = b["variant"].get<std::string>();
    for (const char* k : {"fix", "nofix", "base", "rot", "rotbase"})
        if (v == k) return true;
    err = "unknown variant '" + v + "' (fix | nofix | base | rot | rotbase)";
    return false;
}

// `tiers` may be omitted (all four), a list of names, or a list of {name, decimate} objects.
bool parse_tiers(const json& b, std::vector<Tier>& out, std::string& err) {
    out.clear();
    if (!b.contains("tiers")) { out = default_tiers(); return true; }
    if (!b["tiers"].is_array() || b["tiers"].empty()) { err = "`tiers` must be a non-empty array"; return false; }
    for (const auto& t : b["tiers"]) {
        if (t.is_string()) {
            const std::string n = t.get<std::string>();
            bool found = false;
            for (const auto& d : default_tiers())
                if (d.name == n) { out.push_back(d); found = true; break; }
            if (!found) { err = "unknown tier '" + n + "' (hero|high|medium|game, or {name,decimate})"; return false; }
        } else if (t.is_object() && t.contains("name") && t.contains("decimate")) {
            out.push_back({t["name"].get<std::string>(), t["decimate"].get<int>()});
        } else {
            err = "each tier is a name or {\"name\":..., \"decimate\":N}";
            return false;
        }
    }
    return true;
}

void fill_rig_request(State& st, const json& b, const std::string& image, const std::string& out_glb,
                      avatar::RigRequest& r) {
    r.image = image;
    r.out_glb = out_glb;
    r.texsize   = b.value("texsize", 2048);
    r.decimate  = b.value("decimate", 220000);
    r.resolution = b.value("resolution", 0);        // 0 = the binary's default lattice. OPT-IN only:
                                                    // 1536 was not visually separable and is not a
                                                    // default anyone asked for. Same for `--solid`,
                                                    // which is reachable through extra_args and is
                                                    // NOT a default: it rigs inconsistently.
    r.dc_band   = b.value("dc_band", 0);
    r.dc_remesh = b.value("dc_remesh", true);
    // BEST-OF-N. Default 2 (Config::rig_retries): a rejected draw is usually a ~214 s runaway, so
    // N=2 bounds the worst case near ~430 s while still buying a second look at a draw lottery that
    // welded every vertex to one joint in four of six measured draws. `rig_extra_retries` grants ONE
    // more draw, and only when the budget ran out with nothing accepted AND nothing passing the pose
    // gate — an optimisation for a bad run, never a precondition for returning an asset.
    r.rig_retries = b.value("rig_retries", st.cfg.rig_retries);
    r.rig_extra_retries = b.value("rig_extra_retries", -1);   // -1 = the stage's own default
    // 🔴 A "try again" button MUST send a new `rig_seed` or it is a no-op — the rig decode is
    // deterministic in (mesh, seed), so re-rigging an unchanged request against a cached geometry
    // returns the same draw it just rejected. Named here rather than left to `extra_args` precisely
    // because that is the mistake a client makes silently. `geo_seed` redraws the MESH instead, and
    // costs the full geometry stage.
    r.rig_seed = b.value("rig_seed", (uint64_t)0);
    r.geo_seed = b.value("geo_seed", -1);
    // Skin-weight cleanup: ON, because it verifies itself against the shipped pose gate and keeps
    // the original skin byte-for-byte when it cannot prove an improvement. It removes the vertices
    // bound to a joint half a body away that get flung across the character when it rotates — the
    // "spears off the arms" defect — and on a rig that is already good it is a ~1 s no-op.
    r.skin_cleanup = b.value("skin_cleanup", true);
    if (b.contains("extra_args") && b["extra_args"].is_array())
        for (const auto& a : b["extra_args"])
            if (a.is_string()) r.extra_args.push_back(a.get<std::string>());
}

// Build every requested tier under ONE admission. Returns false only on a real failure — a rig that
// merely fails the pose gate is a SUCCESS with a warning attached (see the header).
bool build_ladder(State& st, JobPtr j, const json& b, const std::string& image,
                  const std::string& stage_dir, const std::string& rig_cache,
                  json& tiers_out, json& warnings, std::string& err, std::string& code) {
    std::vector<Tier> tiers;
    if (!parse_tiers(b, tiers, err)) { code = "bad_request"; return false; }
    tiers_out = json::array();

    // A caller-named geometry to resume from (`--from-refined`) applies to the FIRST tier only;
    // after that our own stage dir is the resume point, whether we generated it or inherited it.
    std::string resume_from = b.value("from_refined", std::string());

    for (size_t i = 0; i < tiers.size(); i++) {
        if (j->cancel->cancelled()) { err = "cancelled"; code = "cancelled"; return false; }
        avatar::RigRequest r;
        fill_rig_request(st, b, image, j->dir + "/" + tiers[i].name + ".glb", r);
        r.decimate = tiers[i].decimate;
        r.stage_dir = stage_dir;
        r.rig_cache = rig_cache;
        r.from_refined = resume_from;

        avatar::RigResult rr;
        std::string e;
        log_line("job %s: tier %s (decimate %d)%s", j->id.c_str(), tiers[i].name.c_str(),
                 tiers[i].decimate, resume_from.empty() ? "" : " resuming from a cached geometry");
        if (!st.engine->build_rig(r, rr, e)) {
            err = "tier " + tiers[i].name + ": " + e;
            code = (e == "cancelled") ? "cancelled" : "rig_failed";
            // A ladder that produced the hero and then failed on a cheaper tier still delivered
            // something worth having, so say which tiers exist rather than throwing them away.
            if (!tiers_out.empty()) warnings.push_back(err + " — earlier tiers were delivered");
            return tiers_out.empty() ? false : true;
        }
        json t = {
            {"tier", tiers[i].name}, {"decimate", tiers[i].decimate},
            {"glb", rr.glb}, {"joints", rr.joints}, {"named_core", rr.named_core},
            {"seconds", rr.seconds},
            // "" = the cleanup did not run. Otherwise the one-line verdict, including the case where
            // it found nothing to improve and kept the original skin.
            {"skin_cleanup", rr.skin_cleanup},
            {"result_url", "/v1/jobs/" + j->id + "/result?tier=" + tiers[i].name},
        };
        // The gate verdict belongs to the DRAW, so it is reported on the tier that decoded it. Tiers
        // that came off the cache report `draws: 0` — they inherited the skeleton and re-ran nothing.
        t["gate"] = gate_json(rr.gate);
        if (!rr.gate.accepted && rr.gate.draws > 0) {
            warnings.push_back("tier " + tiers[i].name + ": NO draw satisfied the acceptance "
                               "predicate — shipped the best-ranked draw. " + rr.gate.selector_summary);
        } else if (rr.gate.pose_gate_ran && !rr.gate.pose_gate_pass) {
            warnings.push_back("tier " + tiers[i].name + ": the deformation (pose) gate FAILED at " +
                               std::to_string(rr.gate.pose_gate_worst) +
                               " (limit 6.0) — the asset is delivered anyway; expect stretch artefacts.");
        }
        tiers_out.push_back(t);
        // From here on the geometry is ours: every later tier re-enters at bake+rig and pays neither
        // the diffusion nor the DC remesh.
        resume_from = stage_dir;
    }
    return true;
}

void run_rig(State& st, JobPtr j, const json& b, const std::string& image) {
    // OUR session, with OUR token: build_rig() would otherwise open one with no token and the
    // cancellation hook would never be armed for this request. One admission for the whole ladder.
    auto sess = st.engine->session(j->cancel.get());
    if (!sess.ok()) {
        finish(st, j, false, sess.error(), sess.status_code() == 499 ? "cancelled" : "unavailable");
        return;
    }
    // 🔴 GIVE THE MOTION DiT BACK FIRST. Engine::run() does this between its own stages and says
    // why: the rig stage peaks near 10.7 GB of the 3060's 11.9 GB, so the ~2.1 GB motion DiT must
    // not be resident across it. This endpoint drives the stages itself, so it owes the same
    // release — without it a /v1/rig that follows a /v1/motion runs the biggest stage in the
    // pipeline with our own weights still on the card. Safe here and only here: we hold the
    // session, so nothing else is using the CUDA context.
    st.engine->release_gpu();
    std::string verr;
    if (!vram_ok(st, verr)) { finish(st, j, false, verr, "insufficient_vram"); return; }

    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        j->status = "running";
        j->started_at = wall_now();
    }
    // Always OURS unless the caller named one: a job that keeps no stage dir cannot be re-rigged
    // cheaply, and the whole point of publishing these paths is that the next request can.
    const std::string stage_dir = b.value("stage_dir", j->dir + "/stage");
    const std::string rig_cache = b.value("rig_cache", j->dir + "/rig_cache");
    json tiers_out, warnings = json::array();
    std::string err, code;
    const double t0 = now_s();
    const bool ok = build_ladder(st, j, b, image, stage_dir, rig_cache, tiers_out, warnings, err, code);
    const double took = now_s() - t0;
    if (!ok) { j->seconds = took; finish(st, j, false, err, code); return; }

    json result = {
        {"tiers", tiers_out},
        // The re-run handles. A later job passes either of these — or just this job's id — and skips
        // the expensive half: `geometry_from` re-bakes and RE-RIGS (~60-180 s), `rig_cache_from`
        // keeps the skeleton and re-bakes only.
        {"stage_dir", stage_dir},
        {"rig_cache", rig_cache},
        {"image", image},
        {"peak_vram_used_mib", (long)st.engine->peak_vram_used_mib()},
        {"baseline_vram_used_mib", (long)st.engine->baseline_vram_used_mib()},
    };
    if (!warnings.empty()) result["warnings"] = warnings;
    // The hero is the default artifact; `?tier=` picks any other.
    publish(st, j, std::move(result),
            tiers_out.empty() ? std::string() : tiers_out[0]["glb"].get<std::string>(),
            "model/gltf-binary", took);
    finish(st, j, true, "", "");
}

void run_motion(State& st, JobPtr j, const json& b) {
    avatar::MotionRequest m;
    m.prompt  = b.value("prompt", std::string());
    m.uncond  = b.value("uncond", false);
    m.seconds = b.value("seconds", 4.0f);
    m.cfg     = b.value("cfg", 5.0f);
    m.steps   = b.value("steps", 50);
    m.seed    = (uint32_t)b.value("seed", 0);
    m.name    = b.value("name", std::string("motion"));
    m.out_json = j->dir + "/clip.json";

    auto sess = st.engine->session(j->cancel.get());
    if (!sess.ok()) {
        finish(st, j, false, sess.error(), sess.status_code() == 499 ? "cancelled" : "unavailable");
        return;
    }
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        j->status = "running";
        j->started_at = wall_now();
    }
    avatar::MotionResult mr;
    std::string err;
    const double t0 = now_s();
    const bool ok = st.engine->generate_motion(m, mr, err);
    const double took = now_s() - t0;
    if (!ok) { j->seconds = took; finish(st, j, false, err, err == "cancelled" ? "cancelled" : "motion_failed"); return; }
    json result = {
        {"frames", mr.frames}, {"fps", mr.fps}, {"seconds", mr.seconds},
        {"dit_was_resident", mr.dit_was_resident},
        {"load_s", mr.load_s}, {"encode_s", mr.encode_s}, {"solve_s", mr.solve_s},
    };
    publish(st, j, std::move(result), mr.path, "application/json", took);
    finish(st, j, true, "", "");
}

// Retarget. NO SESSION, following the library: animate() is CPU-only and taking the card for it
// would park a 0.22 s job behind a 450 s rig, which is exactly the interaction loop this service
// exists to protect.
//
// THE ONE WRINKLE: cancelhook's armed flag is a process global (cancel_hook.hpp explains why), so
// while a rig holds the card with ITS token armed, a concurrent retarget reads THAT flag. A cancel
// aimed at the rig can therefore abort an unrelated 0.22 s retarget. Rather than pretend it cannot
// happen, retry once when the stage says "cancelled" and OUR token is not tripped — the retarget is
// idempotent and costs a fifth of a second.
void run_animate(State& st, JobPtr j, const json& b, const std::string& rig_glb,
                 std::vector<std::string> clip_text, std::vector<std::string> clip_paths) {
    avatar::AnimateRequest a;
    a.rig_glb = rig_glb;
    a.clip_json = std::move(clip_text);
    a.clip_path = std::move(clip_paths);
    a.variant = b.value("variant", std::string("rotbase"));
    a.smooth  = b.value("smooth", 2);
    a.exercise = b.value("exercise", true);
    a.out_glb = j->dir + "/anim.glb";

    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        j->status = "running";
        j->started_at = wall_now();
    }
    avatar::AnimateResult ar;
    std::string err;
    const double t0 = now_s();
    bool ok = st.engine->animate(a, ar, err);
    if (!ok && err == "cancelled" && !j->cancel->cancelled()) {
        log_line("job %s: retarget saw ANOTHER request's cancel — retrying once", j->id.c_str());
        ok = st.engine->animate(a, ar, err);
    }
    const double took = now_s() - t0;
    if (!ok) { j->seconds = took; finish(st, j, false, err, err == "cancelled" ? "cancelled" : "animate_failed"); return; }
    json result = {
        {"glb", ar.glb}, {"animations", ar.animations}, {"mapped", ar.mapped},
        {"map_kind", ar.map_kind}, {"swap", ar.swap},
        {"rest_arm_deg", ar.rest_arm_deg}, {"rest_leg_deg", ar.rest_leg_deg},
        {"rest_spine_deg", ar.rest_spine_deg},
        {"exercise_joints", ar.exercise_joints}, {"seconds", ar.seconds},
    };
    publish(st, j, std::move(result), ar.glb, "model/gltf-binary", took);
    finish(st, j, true, "", "");
}

// The one-shot chain. It is NOT Engine::run(): run() builds a single rig, and this service serves
// the whole LOD ladder. What it DOES keep is run()'s ordering, which is not cosmetic —
//
//     motion FIRST, then release the DiT, THEN the rig.
//
// The rig stage peaks near 10.7 GB of the 3060's 11.9 GB, so the ~2.1 GB motion DiT must not still
// be resident across it. Doing it the other way round is an OOM, not a slowdown.
void run_render(State& st, JobPtr j, const json& b, const std::string& image) {
    auto sess = st.engine->session(j->cancel.get());
    if (!sess.ok()) {
        finish(st, j, false, sess.error(), sess.status_code() == 499 ? "cancelled" : "unavailable");
        return;
    }
    std::string verr;
    if (!vram_ok(st, verr)) { finish(st, j, false, verr, "insufficient_vram"); return; }
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        j->status = "running";
        j->started_at = wall_now();
    }
    const double t0 = now_s();
    json warnings = json::array();

    // ---- 1/3 motion -------------------------------------------------------------------------
    avatar::MotionRequest m;
    m.prompt  = b.value("prompt", std::string());
    m.uncond  = b.value("uncond", false);
    m.seconds = b.value("seconds", 4.0f);
    m.cfg     = b.value("cfg", 5.0f);
    m.steps   = b.value("steps", 50);
    m.seed    = (uint32_t)b.value("seed", 0);
    m.name    = b.value("name", std::string("prompted"));
    m.out_json = j->dir + "/clip.json";
    avatar::MotionResult mr;
    std::string err;
    if (!st.engine->generate_motion(m, mr, err)) {
        j->seconds = now_s() - t0;
        finish(st, j, false, err, err == "cancelled" ? "cancelled" : "motion_failed");
        return;
    }
    st.engine->release_gpu();   // 2.1 GB back before the rig stage asks for 10.7

    // ---- 2/3 the rig ladder -----------------------------------------------------------------
    const std::string stage_dir = b.value("stage_dir", j->dir + "/stage");
    const std::string rig_cache = b.value("rig_cache", j->dir + "/rig_cache");
    json tiers_out;
    std::string code;
    if (!build_ladder(st, j, b, image, stage_dir, rig_cache, tiers_out, warnings, err, code)) {
        j->seconds = now_s() - t0;
        finish(st, j, false, err, code);
        return;
    }

    // ---- 3/3 retarget onto one tier ---------------------------------------------------------
    // The hero by default. The clip is authored against the SHARED skeleton, so animating a
    // different tier is a request field and not a different clip.
    const std::string want_tier = b.value("animate_tier", std::string("hero"));
    std::string rig_glb = tiers_out.empty() ? std::string() : tiers_out[0]["glb"].get<std::string>();
    for (const auto& t : tiers_out)
        if (t["tier"] == want_tier) rig_glb = t["glb"].get<std::string>();
    avatar::AnimateRequest a;
    a.rig_glb = rig_glb;
    a.clip_json.push_back(mr.clip_json);
    a.variant = b.value("variant", std::string("rotbase"));
    a.smooth  = b.value("smooth", 2);
    a.exercise = b.value("exercise", true);
    a.out_glb = j->dir + "/anim.glb";
    avatar::AnimateResult ar;
    if (!st.engine->animate(a, ar, err)) {
        j->seconds = now_s() - t0;
        finish(st, j, false, err, err == "cancelled" ? "cancelled" : "animate_failed");
        return;
    }
    const double took = now_s() - t0;

    json result = {
        {"total_s", took},
        {"peak_vram_used_mib", (long)st.engine->peak_vram_used_mib()},
        {"baseline_vram_used_mib", (long)st.engine->baseline_vram_used_mib()},
        {"tiers", tiers_out},
        {"stage_dir", stage_dir},
        {"rig_cache", rig_cache},
        {"image", image},
        {"animated_tier", want_tier},
        {"clip", mr.path},
        {"motion", {{"frames", mr.frames}, {"fps", mr.fps}, {"seconds", mr.seconds},
                    {"dit_was_resident", mr.dit_was_resident}}},
        {"animate", {{"glb", ar.glb}, {"animations", ar.animations}, {"mapped", ar.mapped},
                     {"map_kind", ar.map_kind}, {"seconds", ar.seconds},
                     {"exercise_joints", ar.exercise_joints}}},
    };
    if (!warnings.empty()) result["warnings"] = warnings;
    publish(st, j, std::move(result), ar.glb, "model/gltf-binary", took);
    finish(st, j, true, "", "");
}

// ---------------------------------------------------------------------------
// The stage, run in the CUDA child — the parent's version of run_rig/run_motion/...
// ---------------------------------------------------------------------------
// ONE synchronous `?wait=1` request per job. The child is the same binary with the same handlers,
// so what happens on the far side of this call is exactly the in-process path above; the only
// difference is which process the CUDA context belongs to.
//
// `_job_id` is why the parent can keep the job history: the child mints nothing, adopts our id, and
// therefore writes into the SAME /out/jobs/<id>/ directory and publishes the same `result_url`s and
// the same `stage_dir` / `rig_cache` handles. Kill the child afterwards and every one of those
// still resolves, because they were only ever paths on a shared volume.
void run_proxy(State& st, JobPtr j, const std::string& path, json body, bool needs_rig_vram) {
    if (j->cancel->cancelled()) { finish(st, j, false, "cancelled", "cancelled"); return; }
    // THE SECOND FLOOR CHECK, and the parent's replacement for the one run_rig used to make with
    // the card in hand. A job can sit in the queue for a quarter of an hour, so the number that
    // mattered at submit is not the number that matters now.
    //
    // Only when NOTHING OF OURS is on the card (`leases() == 0`), which is the same rule the old
    // in-process check followed and for the same reason: if one of our renders is running then low
    // free VRAM IS that render, and the request must queue behind it rather than be refused. When
    // we are idle, vram_ok() may kill the worker and re-measure, so the refusal that survives that
    // names a genuine co-tenant.
    if (needs_rig_vram && st.worker->leases() == 0) {
        std::string verr;
        if (!vram_ok(st, verr)) { finish(st, j, false, verr, "insufficient_vram"); return; }
    }
    body["_job_id"] = j->id;
    // Already resolved by the parent against the parent's job table; the child has no job table to
    // resolve them against and would only reject them.
    body.erase("geometry_from");
    body.erase("rig_cache_from");

    // The lease both spawns the worker and pins it: nothing can unload it under us.
    avsup::Worker::Lease lease = st.worker->lease();
    if (!lease.ok()) { finish(st, j, false, "GPU worker unavailable: " + lease.error(), "unavailable"); return; }
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        j->status = "running";
        j->started_at = wall_now();
    }

    // A cancel has to REACH the child, and it may be tripped at any point — including in the window
    // between our POST leaving and the child registering the job, which is why this retries until
    // the child acknowledges rather than firing once. Cancellation stays a <= 7 s promise.
    std::atomic<bool> settled{false};
    const int port = lease.port();
    const std::string jid = j->id;
    std::thread canceller([&settled, j, port, jid] {
        bool sent = false;
        while (!settled.load()) {
            if (!sent && j->cancel->cancelled()) {
                httplib::Client c("127.0.0.1", port);
                c.set_connection_timeout(2, 0);
                c.set_read_timeout(5, 0);
                const auto r = c.Post("/v1/jobs/" + jid + "/cancel", "{}", "application/json");
                if (r && r->status == 200) sent = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(10, 0);
    // A four-tier ladder is ~15 minutes and a /v1/render measured 576.7 s. httplib's 5 s default
    // would drop the socket mid-render and turn a healthy job into a failure.
    cli.set_read_timeout(6 * 60 * 60, 0);
    cli.set_write_timeout(10 * 60, 0);
    const double t0 = now_s();
    const auto resp = cli.Post(path + "?wait=1", body.dump(), "application/json");
    const double took = now_s() - t0;
    settled.store(true);
    canceller.join();

    if (!resp) {
        j->seconds = took;
        // A forced unload SIGKILLs the worker out from under a job it has already cancelled. That
        // is a cancellation, not a failure, and calling it a failure would make `force` look broken.
        if (j->cancel->cancelled()) { finish(st, j, false, "cancelled", "cancelled"); return; }
        // Otherwise the worker died on us — an OOM kill, or a crash.
        finish(st, j, false,
               "the GPU worker did not answer (" + httplib::to_string(resp.error()) +
                   ") — it may have died mid-job; see the container log",
               "worker_failed");
        return;
    }
    json rb = json::object();
    try { rb = json::parse(resp->body); } catch (...) {}
    if (resp->status == 200) {
        const std::string mime = rb.value("result_mime", std::string("application/octet-stream"));
        publish(st, j, rb.value("result", json()), rb.value("result_path", std::string()),
                mime.c_str(), rb.value("seconds", took));
        finish(st, j, true, "", "");
        return;
    }
    j->seconds = rb.value("seconds", took);
    const std::string code = rb.value("error_code",
                                      std::string(resp->status == 499 ? "cancelled" : "worker_failed"));
    finish(st, j, false,
           rb.value("error", "the GPU worker returned " + std::to_string(resp->status)), code);
}

}  // namespace

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------
namespace {

// The job id. In the CHILD it is our parent's, adopted verbatim so that both processes name one
// job and one job directory (see run_proxy). Everywhere else it is minted here as before.
//
// The hex check is not cosmetic: this id is concatenated into `jobs_dir` by make_job_dir(), so it
// has to be incapable of naming anything outside it.
std::string mint_job_id(State& st, json& b) {
    std::string adopted;
    if (b.contains("_job_id") && b["_job_id"].is_string()) {
        adopted = b["_job_id"].get<std::string>();
        b.erase("_job_id");
    }
    std::lock_guard<std::mutex> lk(st.jm.mx);
    if (st.worker_mode && !adopted.empty() && adopted.size() <= 32 &&
        adopted.find_first_not_of("0123456789abcdef") == std::string::npos) {
        st.jm.next_id++;      // keep our own counter moving, so a locally minted id cannot collide
        return adopted;
    }
    return new_job_id(st.jm);
}

// Cancelling a job means cancelling it wherever it is running. The token is the whole story in
// single-process mode; under isolation run_proxy's canceller thread watches the same token and
// relays it to the child, so this stays one call for every caller.
void cancel_job(JobPtr j) { j->cancel->cancel(); }

// Everything a submit handler shares: admission, the job record, the thread, and `?wait=1`.
// `work` is the stage body, already bound to its parsed request.
void submit(State& st, const httplib::Request& req, httplib::Response& res, const std::string& kind,
            const json& request_echo,
            std::function<void(JobPtr)> work, const std::string& job_dir, const std::string& job_id) {
    JobPtr j = std::make_shared<Job>();
    j->id = job_id;
    j->kind = kind;
    j->dir = job_dir;
    j->request = request_echo;

    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        st.jm.jobs[j->id] = j;
        st.jm.order.push_back(j->id);
        st.jm.accepted++;
    }
    log_line("job %s accepted (%s)", j->id.c_str(), kind.c_str());

    std::thread([&st, j, work] {
        work(j);
        // A job whose body returned without calling finish() would hang a waiter forever.
        if (!j->done.load()) finish(st, j, false, "stage returned without a verdict", "internal");
    }).detach();

    // `?wait=1` — the synchronous shape spar3d offers. It is the SAME job underneath, so a client
    // that times out can still poll the id it was given.
    if (req.has_param("wait") && req.get_param_value("wait") == "1") {
        while (!j->done.load()) {
            // CLIENT DISCONNECT IS A CANCEL. A 450 s render whose caller has gone away is 450 s of
            // a shared 12 GB card spent on nothing. CancelToken lands in <= 7 s.
            if (req.is_connection_closed()) {
                log_line("job %s: client disconnected — cancelling", j->id.c_str());
                cancel_job(j);
                return;   // nothing to write; the socket is gone
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::lock_guard<std::mutex> lk(st.jm.mx);
        const int code = j->status == "completed" ? 200 : (j->status == "cancelled" ? 499 : 500);
        set_json(res, job_public(*j), code);
        return;
    }
    std::lock_guard<std::mutex> lk(st.jm.mx);
    set_json(res, job_public(*j), 202);
}

// Admission checks common to every submit: draining, the pending cap, and — for the RIG stage only
// — the VRAM floor.
//
// ⚠️ `needs_rig_vram` IS NOT "touches the GPU". The floor is ~10.7 GB because that is what the RIG
// stage peaks at; a motion request needs a fraction of it. Applying the rig's floor to every GPU
// stage is not conservative, it is wrong: measured 2026-08-14, the motion DiT stays resident by
// design (~2.1 GB), which drops free VRAM to ~9.6 GB, which then refused EVERY subsequent request
// — including motion ones that would have fitted — until idle-unload fired 300 s later. A guard
// that bricks the service after one successful job is worse than the OOM it was guarding against.
bool admit(State& st, httplib::Response& res, bool needs_rig_vram) {
    const avatar::Health h = backend_health(st);
    if (h.draining) {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        st.jm.rejected++;
        set_json(res, {{"error", "service draining — not accepting new requests"}, {"error_code", "draining"}}, 503);
        return false;
    }
    int live = 0;
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        live = live_jobs(st.jm);
        if (live >= st.cfg.max_pending) {
            st.jm.rejected++;
            set_json(res, {{"error", "queue full"}, {"error_code", "queue_full"},
                           {"max_pending", st.cfg.max_pending}}, 503);
            return false;
        }
    }
    // Only when WE are idle AND holding nothing: if one of our renders has the card, low free VRAM
    // is us and the request must queue; if our weights are merely RESIDENT, the job will hand them
    // back before the rig stage (run_rig calls release_gpu), so raw free VRAM here is not the
    // number that matters. In both cases the authoritative check is the one inside the job, made
    // with the card in hand and after the release.
    //
    // Under worker isolation the `!h.loaded` half is DROPPED, and deliberately: `loaded` there
    // means "a CUDA worker exists", and skipping the floor whenever one exists would skip it
    // permanently. The check is safe to run in that mode because vram_ok() can kill an idle worker
    // and re-measure, so a resident-weights reading can no longer produce a false refusal.
    const bool idle_enough = st.engine ? (h.in_flight == 0 && !h.loaded)
                                       : (h.in_flight == 0 && live == 0);
    if (needs_rig_vram && idle_enough) {
        std::string verr;
        if (!vram_ok(st, verr)) {
            std::lock_guard<std::mutex> lk(st.jm.mx);
            st.jm.rejected++;
            set_json(res, {{"error", verr}, {"error_code", "insufficient_vram"},
                           {"free_mib", free_vram_mib(st)}, {"min_free_mib", st.cfg.min_free_mib}}, 503);
            return false;
        }
    }
    return true;
}

std::string make_job_dir(State& st, const std::string& id) {
    const std::string d = st.cfg.jobs_dir + "/" + id;
    mkdirs(d);
    return d;
}

}  // namespace

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------
namespace {

// `geometry_from` / `rig_cache_from`: a path, or the id of an earlier rig job. This is the whole
// "drive it step by step" story — resolving them here means the client re-runs a stage by naming a
// job, never by knowing where we put files. Fills `b` in place with the resolved paths, and inherits
// that job's input image when the caller did not send one.
bool resolve_reruns(State& st, json& b, std::string& err) {
    if (b.contains("geometry_from") && b["geometry_from"].is_string()) {
        const std::string s = b["geometry_from"].get<std::string>();
        std::string dir = s;
        if (!is_regular_file(s + "/refined.glb")) {
            JobPtr src = find_job(st, s);
            if (!src || src->status != "completed") {
                err = "geometry_from '" + s + "' is neither a stage dir (no refined.glb) nor a completed job id";
                return false;
            }
            dir = src->result.value("stage_dir", std::string());
            if (dir.empty() || !is_regular_file(dir + "/refined.glb")) {
                err = "job " + s + " kept no reusable geometry (no stage dir with refined.glb)";
                return false;
            }
            if (!b.contains("image")) b["image"] = src->result.value("image", std::string());
        }
        b["from_refined"] = dir;
    }
    if (b.contains("rig_cache_from") && b["rig_cache_from"].is_string()) {
        const std::string s = b["rig_cache_from"].get<std::string>();
        std::string dir = s;
        if (!is_regular_file(s + "/rig_joints.bin")) {
            JobPtr src = find_job(st, s);
            if (!src || src->status != "completed") {
                err = "rig_cache_from '" + s + "' is neither a rig cache dir nor a completed job id";
                return false;
            }
            dir = src->result.value("rig_cache", std::string());
            if (dir.empty() || !is_regular_file(dir + "/rig_joints.bin")) {
                err = "job " + s + " kept no rig cache";
                return false;
            }
            if (!b.contains("image")) b["image"] = src->result.value("image", std::string());
        }
        // Used AS the cache dir: the stage reads it and writes the same skeleton back, so pointing
        // at the source job's cache shares one skeleton across jobs by construction.
        b["rig_cache"] = dir;
    }
    return true;
}

void handle_rig(State& st, const httplib::Request& req, httplib::Response& res) {
    json b;
    std::string err;
    if (!parse_body(req, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!validate_common(b, /*needs_prompt=*/false, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!resolve_reruns(st, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    std::vector<Tier> probe;
    if (!parse_tiers(b, probe, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!admit(st, res, /*needs_rig_vram=*/true)) return;

    const std::string id = mint_job_id(st, b);
    const std::string dir = make_job_dir(st, id);
    std::string image;
    if (!resolve_image(req, b, dir, image, err)) { set_json(res, {{"error", err}}, 400); return; }
    b["image"] = image;

    json echo = b;
    echo["rig_retries"] = b.value("rig_retries", st.cfg.rig_retries);
    echo["tiers"] = json::array();
    for (const auto& t : probe) echo["tiers"].push_back(json{{"name", t.name}, {"decimate", t.decimate}});
    submit(st, req, res, "rig", echo,
           [&st, b, image](JobPtr j) {
               if (st.worker) run_proxy(st, j, "/v1/rig", b, /*needs_rig_vram=*/true);
               else           run_rig(st, j, b, image);
           }, dir, id);
}

void handle_motion(State& st, const httplib::Request& req, httplib::Response& res) {
    json b;
    std::string err;
    if (!parse_body(req, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!validate_common(b, /*needs_prompt=*/true, err)) { set_json(res, {{"error", err}}, 400); return; }
    // needs_rig_vram=false: the motion stage is nowhere near the rig's 10.7 GB, and holding it to
    // that floor refuses work that fits (see admit()).
    if (!admit(st, res, /*needs_rig_vram=*/false)) return;
    const std::string id = mint_job_id(st, b);
    const std::string dir = make_job_dir(st, id);
    submit(st, req, res, "motion", b,
           [&st, b](JobPtr j) {
               if (st.worker) run_proxy(st, j, "/v1/motion", b, /*needs_rig_vram=*/false);
               else           run_motion(st, j, b);
           }, dir, id);
}

void handle_animate(State& st, const httplib::Request& req, httplib::Response& res) {
    json b;
    std::string err;
    if (!parse_body(req, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!validate_common(b, /*needs_prompt=*/false, err) || !validate_variant(b, err)) {
        set_json(res, {{"error", err}}, 400);
        return;
    }

    std::string rig_glb;
    if (!resolve_rig(st, b.value("rig", std::string()), b.value("tier", std::string()), rig_glb, err)) {
        set_json(res, {{"error", err}}, 400);
        return;
    }
    // Clips: a job id, a path, or the clip TEXT itself. Same three-way resolution as `rig`, because
    // a client that just called /v1/motion has an id and nothing else.
    std::vector<std::string> texts, paths;
    if (b.contains("clips") && b["clips"].is_array()) {
        for (const auto& c : b["clips"]) {
            if (!c.is_string()) continue;
            const std::string s = c.get<std::string>();
            if (is_regular_file(s)) { paths.push_back(s); continue; }
            JobPtr cj;
            {
                std::lock_guard<std::mutex> lk(st.jm.mx);
                auto it = st.jm.jobs.find(s);
                if (it != st.jm.jobs.end()) cj = it->second;
            }
            if (cj) {
                if (cj->status != "completed" || cj->kind != "motion") {
                    set_json(res, {{"error", "clip job " + s + " is not a completed motion job"}}, 400);
                    return;
                }
                paths.push_back(cj->result_path);
                continue;
            }
            if (!s.empty() && s[0] == '{') { texts.push_back(s); continue; }
            set_json(res, {{"error", "clip '" + s.substr(0, 64) + "' is not a path, a motion job id, or clip JSON"}}, 400);
            return;
        }
    }
    if (texts.empty() && paths.empty()) {
        set_json(res, {{"error", "animate needs `clips`: an array of motion job ids, clip paths, or clip JSON"}}, 400);
        return;
    }
    // needs_card=false: the retarget never touches the GPU, so it must not be refused for VRAM.
    if (!admit(st, res, /*needs_rig_vram=*/false)) return;
    const std::string id = mint_job_id(st, b);
    const std::string dir = make_job_dir(st, id);
    json echo = b;
    echo["rig"] = rig_glb;
    // 🔴 The retarget is CPU-only, and it STILL goes to the child under isolation, because it is a
    // method on avatar::Engine and constructing one is precisely what the parent must never do. The
    // cost is a worker cold start (~seconds, no weights) when a bare /v1/animate arrives with no
    // child up; the alternative — a CUDA context in the parent forever — is the bug being fixed.
    json cb = b;
    cb["rig"] = rig_glb;
    cb.erase("tier");                      // already applied when we resolved the tier's GLB
    cb["clips"] = json::array();
    for (const auto& p : paths) cb["clips"].push_back(p);
    for (const auto& t : texts) cb["clips"].push_back(t);
    submit(st, req, res, "animate", echo,
           [&st, b, cb, rig_glb, texts, paths](JobPtr j) {
               if (st.worker) run_proxy(st, j, "/v1/animate", cb, /*needs_rig_vram=*/false);
               else           run_animate(st, j, b, rig_glb, texts, paths);
           },
           dir, id);
}

void handle_render(State& st, const httplib::Request& req, httplib::Response& res) {
    json b;
    std::string err;
    if (!parse_body(req, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!validate_common(b, /*needs_prompt=*/true, err) || !validate_variant(b, err)) {
        set_json(res, {{"error", err}}, 400);
        return;
    }
    if (!resolve_reruns(st, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    std::vector<Tier> probe;
    if (!parse_tiers(b, probe, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!admit(st, res, /*needs_rig_vram=*/true)) return;
    const std::string id = mint_job_id(st, b);
    const std::string dir = make_job_dir(st, id);
    std::string image;
    if (!resolve_image(req, b, dir, image, err)) { set_json(res, {{"error", err}}, 400); return; }
    b["image"] = image;
    json echo = b;
    submit(st, req, res, "render", echo,
           [&st, b, image](JobPtr j) {
               if (st.worker) run_proxy(st, j, "/v1/render", b, /*needs_rig_vram=*/true);
               else           run_render(st, j, b, image);
           }, dir, id);
}

void register_routes(httplib::Server& svr, State& st) {
    // ---- health -------------------------------------------------------------------------------
    // Every field comes straight off Engine::health(), which takes NO LOCK and answers while a
    // 450 s render owns everything.
    //
    // 🔴 busy, queued and waiting_for_gpu_lock are REPORTED SEPARATELY and none of them is
    // derivable from another. `busy` = our one slot is taken. `queued` = admitted requests waiting
    // for that slot — a health that hid this shows an idle service that is actually backed up.
    // `waiting_for_gpu_lock` = we hold our own slot but ANOTHER PROCESS on this box owns the card;
    // without it an operator cannot tell "our render is slow" from "we have not started".
    svr.Get("/health", [&st](const httplib::Request&, httplib::Response& res) {
        const avatar::Health h = backend_health(st);
        int jobs_queued = 0, jobs_running = 0;
        {
            std::lock_guard<std::mutex> lk(st.jm.mx);
            for (const auto& [_, j] : st.jm.jobs) {
                if (j->done.load()) continue;
                if (j->status == "running") jobs_running++;
                else jobs_queued++;
            }
        }
        set_json(res, {
            {"status", h.status},
            {"busy", h.busy},
            {"in_flight", h.in_flight},
            {"queued", h.queued},
            {"waiting_for_gpu_lock", h.waiting_for_gpu_lock},
            {"draining", h.draining},
            {"loaded", h.loaded},
            {"loaded_model", h.loaded_model},
            {"renders_done", h.renders_done},
            {"last_render_sec", h.last_render_sec},
            {"uptime_sec", h.uptime_sec},
            {"idle_sec", h.idle_sec},
            {"idle_unload_seconds", h.idle_unload_seconds},
            {"keep_resident", h.keep_resident},
            {"jobs_running", jobs_running},
            {"jobs_queued", jobs_queued},
            {"min_free_mib", st.cfg.min_free_mib},
            {"rig_retries_default", st.cfg.rig_retries},
            {"fatal", h.fatal},
            // Which of the two processes is answering, so an operator reading a log or a probe can
            // tell a CUDA-free parent from the worker it supervises without guessing.
            {"worker_isolation", st.worker != nullptr},
            {"worker_pid", st.worker ? st.worker->pid() : 0},
            {"worker_spawns", st.worker ? st.worker->spawns() : 0},
            {"role", st.worker ? "parent" : (st.worker_mode ? "worker" : "single-process")},
        });
    });

    svr.Get("/v1/gpu/status", [&st](const httplib::Request&, httplib::Response& res) {
        // In parent mode `loaded` means a CUDA-owning worker exists, and the free-VRAM number comes
        // from NVML — never from cudaMemGetInfo, which would create the very context this process
        // exists in order not to have. Peak/baseline are the WORKER's, and they reset with it: a
        // fresh child starts a fresh watermark, which is the honest thing to report given the pool
        // it is measuring no longer survives an unload.
        json b;
        if (st.engine) {
            const avatar::GpuStatus g = st.engine->gpu_status();
            b = {
                {"loaded", g.loaded},
                {"gpu", g.gpu.empty() ? json(nullptr) : json(g.gpu)},
                {"mem_free_mb", g.mem_free_mb},
                {"peak_vram_used_mib", (long)st.engine->peak_vram_used_mib()},
                {"baseline_vram_used_mib", (long)st.engine->baseline_vram_used_mib()},
                {"vram_source", "cudaMemGetInfo"},
            };
        } else {
            json cg;
            const bool got = worker_get(st, "/v1/gpu/status", cg);
            b = {
                {"loaded", st.worker && st.worker->alive()},
                {"gpu", got && cg.contains("gpu") ? cg["gpu"] : json(nullptr)},
                {"mem_free_mb", avsup::nvml_free_mib()},
                {"mem_total_mb", avsup::nvml_total_mib()},
                {"peak_vram_used_mib", got ? cg.value("peak_vram_used_mib", (long)0) : (long)0},
                {"baseline_vram_used_mib", got ? cg.value("baseline_vram_used_mib", (long)0) : (long)0},
                {"vram_source", "nvml"},
                {"nvml", avsup::nvml_note()},
                {"worker_pid", st.worker ? st.worker->pid() : 0},
                {"worker_spawns", st.worker ? st.worker->spawns() : 0},
            };
        }
        if (const char* v = std::getenv("CUDA_VISIBLE_DEVICES"); v && *v) b["cuda_visible_devices"] = v;
        set_json(res, b);
    });

    svr.Get("/v1/metrics", [&st](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        set_json(res, {
            {"accepted", st.jm.accepted}, {"completed", st.jm.completed},
            {"failed", st.jm.failed}, {"cancelled", st.jm.cancelled},
            {"rejected", st.jm.rejected}, {"live", live_jobs(st.jm)},
            {"records", (long)st.jm.jobs.size()},
        });
    });

    // ---- the three stages, plus the one-shot chain ---------------------------------------------
    for (const char* p : {"/v1/rig", "/rig"})
        svr.Post(p, [&st](const httplib::Request& q, httplib::Response& r) { handle_rig(st, q, r); });
    for (const char* p : {"/v1/motion", "/motion"})
        svr.Post(p, [&st](const httplib::Request& q, httplib::Response& r) { handle_motion(st, q, r); });
    for (const char* p : {"/v1/animate", "/animate"})
        svr.Post(p, [&st](const httplib::Request& q, httplib::Response& r) { handle_animate(st, q, r); });
    for (const char* p : {"/v1/render", "/render"})
        svr.Post(p, [&st](const httplib::Request& q, httplib::Response& r) { handle_render(st, q, r); });

    // ---- jobs -----------------------------------------------------------------------------------
    svr.Get("/v1/jobs", [&st](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        std::lock_guard<std::mutex> lk(st.jm.mx);
        for (const auto& id : st.jm.order) {
            auto it = st.jm.jobs.find(id);
            if (it != st.jm.jobs.end()) arr.push_back(job_public(*it->second));
        }
        set_json(res, {{"jobs", arr}});
    });

    svr.Get(R"(/v1/jobs/([A-Za-z0-9_-]+))", [&st](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        auto it = st.jm.jobs.find(req.matches[1]);
        if (it == st.jm.jobs.end()) { set_json(res, {{"error", "job_not_found"}}, 404); return; }
        set_json(res, job_public(*it->second));
    });

    svr.Get(R"(/v1/jobs/([A-Za-z0-9_-]+)/result)", [&st](const httplib::Request& req, httplib::Response& res) {
        JobPtr j;
        {
            std::lock_guard<std::mutex> lk(st.jm.mx);
            auto it = st.jm.jobs.find(req.matches[1]);
            if (it != st.jm.jobs.end()) j = it->second;
        }
        if (!j) { set_json(res, {{"error", "job_not_found"}}, 404); return; }
        if (j->status != "completed") {
            set_json(res, {{"error", "result_not_ready"}, {"status", j->status}}, 409);
            return;
        }
        // `?tier=game` picks an LOD tier off a rig job; without it the hero (or, for a render job,
        // the animated delivery) is served.
        std::string path = j->result_path, terr;
        std::string tier;
        if (req.has_param("tier")) {
            tier = req.get_param_value("tier");
            if (!tier_glb(*j, tier, path, terr)) { set_json(res, {{"error", terr}}, 404); return; }
        }
        std::string bytes;
        if (path.empty() || !read_file(path, bytes)) {
            set_json(res, {{"error", "result missing on disk: " + path}}, 410);
            return;
        }
        const std::string name = j->id + (tier.empty() ? "" : "." + tier) +
                                 (j->result_mime == "application/json" ? ".clip.json" : ".glb");
        res.set_header("Content-Disposition", "attachment; filename=\"" + name + "\"");
        res.set_content(bytes, j->result_mime.c_str());
    });

    svr.Post(R"(/v1/jobs/([A-Za-z0-9_-]+)/cancel)", [&st](const httplib::Request& req, httplib::Response& res) {
        JobPtr j;
        {
            std::lock_guard<std::mutex> lk(st.jm.mx);
            auto it = st.jm.jobs.find(req.matches[1]);
            if (it != st.jm.jobs.end()) j = it->second;
        }
        if (!j) { set_json(res, {{"error", "job_not_found"}}, 404); return; }
        cancel_job(j);   // returns immediately; the render stops at its next cancellation point
        set_json(res, {{"id", j->id}, {"status", j->status}, {"cancel_requested", true}});
    });

    // ---- admin, the fleet's names ----------------------------------------------------------------
    svr.Post("/v1/admin/drain", [&st](const httplib::Request&, httplib::Response& res) {
        // The parent's own admission gate. It is NOT forwarded to the child: draining the child's
        // Engine would refuse the very proxied requests that are already in flight, and "in-flight
        // work finishes" is the whole meaning of a drain.
        if (st.engine) st.engine->drain();
        else           st.draining.store(true);
        const avatar::Health h = backend_health(st);
        set_json(res, {{"status", "draining"}, {"busy", h.busy}, {"in_flight", h.in_flight}, {"queued", h.queued}});
    });

    svr.Post("/v1/admin/load", [&st](const httplib::Request&, httplib::Response& res) {
        // Undrain only. Weights reload lazily on the next request, exactly as the fleet's
        // /v1/admin/load does — an eager load here would grab the card while it is still contested.
        // Under isolation it also does NOT spawn a worker, for the same reason: the CUDA context is
        // the expensive thing now, and it is created by the first request that needs it.
        if (st.engine) st.engine->undrain();
        else           st.draining.store(false);
        set_json(res, {{"status", "ok"}, {"loaded", backend_health(st).loaded}});
    });

    // 🔴 WHAT `unloaded` MEANS NOW. Under worker isolation it is the fleet's WorkerSupervisor
    // contract: TRUE means this process can PROVE no CUDA-owning worker of ours remains — the
    // context, every ggml pool and the GPU flock are gone with it — and is therefore the only
    // answer on which a caller may admit competing GPU work. FALSE means the kill or the wait
    // failed and the card may still be held. It is true when there was nothing to kill, because the
    // claim is about the end state, not about whether work was done; `worker_killed` says which.
    //
    // This is a deliberate change from the old `{"status":"idle","unloaded":false}`, which was
    // literally true (the motion DiT was not resident) and operationally useless: nothing was
    // freed either way, because release_gpu() cannot reach a pooled allocator.
    svr.Post("/v1/admin/unload", [&st](const httplib::Request& req, httplib::Response& res) {
        bool force = false;
        if (!req.body.empty()) {
            try { force = json::parse(req.body).value("force", false); } catch (...) {}
        }
        const avatar::Health before = backend_health(st);
        int live = 0;
        {
            std::lock_guard<std::mutex> lk(st.jm.mx);
            live = live_jobs(st.jm);
        }
        const int busy = st.engine ? before.in_flight : std::max(before.in_flight, live);
        if (busy > 0 && !force) {
            set_json(res, {{"status", "busy (pass force=true to cancel + unload)"},
                           {"in_flight", busy}, {"queued", before.queued}}, 409);
            return;
        }
        if (force) {
            std::lock_guard<std::mutex> lk(st.jm.mx);
            for (auto& [_, j] : st.jm.jobs)
                if (!j->done.load()) cancel_job(j);
        }
        if (st.engine) {
            // release_gpu() touches the CUDA context, so it must not race a render: take the card
            // non-blockingly and say so if we could not.
            auto s = st.engine->try_session();
            if (!s.ok()) {
                set_json(res, {{"status", "busy"}, {"error", s.error()}}, 409);
                return;
            }
            const bool was_loaded = before.loaded;
            st.engine->release_gpu();
            st.engine->undrain();
            set_json(res, {{"status", was_loaded ? "unloaded" : "idle"}, {"unloaded", was_loaded},
                           {"worker_killed", false},
                           {"note", "single-process mode: ggml's CUDA pools are NOT returned to the "
                                    "driver by this call. Only a process exit does that."}});
            return;
        }
        // A forced unload has to outrun the cancels it just issued: a cancel lands in <= 7 s, and
        // waiting that long turns "force" into a promise the SIGKILL would keep anyway.
        if (force) {
            for (int i = 0; i < 35 && st.worker->leases() > 0; i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        const int pid = st.worker->pid();
        const bool ok = st.worker->kill_worker();
        st.draining.store(false);
        set_json(res, {{"status", ok ? (pid > 0 ? "unloaded" : "idle") : "failed"},
                       {"unloaded", ok},
                       {"worker_killed", ok && pid > 0},
                       {"worker_pid", pid}},
                 ok ? 200 : 500);
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            "longcat-rig avatar_server\n"
            "  POST /v1/rig       {image}                     -> 202 job; builds the LOD ladder\n"
            "                     {geometry_from:<job|dir>}    re-rig an existing geometry (~60-180s)\n"
            "                     {rig_cache_from:<job|dir>}   reuse the skeleton, re-bake only\n"
            "                     {tiers:[hero,high,medium,game]}   default: all four\n"
            "                     {rig_seed:N}    A RETRY MUST MOVE THIS. The rig decode is\n"
            "                                     deterministic in (mesh, seed): re-rigging the same\n"
            "                                     request returns the draw you just rejected.\n"
            "                     {geo_seed:N}    redraws the MESH instead (pays the ~250 s geometry)\n"
            "  POST /v1/motion    {prompt}                    -> 202 job (~11 s)\n"
            "  POST /v1/animate   {rig:<job|path>, tier, clips:[<job|path|json>]}  -> 202 job (0.22 s)\n"
            "  POST /v1/render    {image, prompt}             -> 202 job, the whole chain\n"
            "  add ?wait=1 to any of the four to block until it finishes\n"
            "  GET  /v1/jobs/{id}                   GET /v1/jobs/{id}/result[?tier=game]\n"
            "  POST /v1/jobs/{id}/cancel\n"
            "  GET  /health   /v1/gpu/status   /v1/metrics\n"
            "  POST /v1/admin/drain | /v1/admin/load | /v1/admin/unload\n"
            "\n"
            "  The GPU runs in a SUPERVISED CHILD PROCESS; this one holds no CUDA context and no\n"
            "  VRAM. /v1/admin/unload SIGKILLs that child, which is what actually returns the card\n"
            "  (ggml pools every block it frees, so an in-process release never could). Jobs and\n"
            "  their artifacts survive it: they live here and on /out, not in the child.\n",
            "text/plain");
    });
}

void usage() {
    std::printf(
        "usage: avatar_server [options]\n"
        "  --host IP             listen address (default 0.0.0.0, or HOST)\n"
        "  --port N              listen port (default 8102, or PORT)\n"
        "  --jobs-dir DIR        where job artifacts land (default /out/jobs)\n"
        "  --min-free-mib N      refuse a GPU stage below this much free VRAM (0 = never)\n"
        "  --rig-retries N       default best-of-N conditioning draws per rig (default 2)\n"
        "  --max-pending N       admitted-but-unfinished job cap (default 32)\n"
        "  --idle-unload S       kill the idle GPU worker after S idle seconds (0 = never)\n"
        "  --lock [PATH]         cross-process GPU flock (default off; held by the WORKER)\n"
        "  --geo-model DIR --hymotion PATH --llm PATH --clip-l PATH   weight overrides\n"
        "  --cpu                 no CUDA (diagnostics only)\n"
        "  --worker              INTERNAL: run as the CUDA-owning child of a supervisor parent.\n"
        "                        Spawned automatically; never pass this by hand.\n"
        "  --no-worker-isolation own the CUDA context in THIS process (the pre-isolation shape).\n"
        "                        ggml pools freed VRAM, so /v1/admin/unload frees NOTHING in this\n"
        "                        mode and only a restart returns the card. To bisect against, not\n"
        "                        to run. Env: LONGCAT_RIG_WORKER_ISOLATION=0.\n");
}

void on_signal(int) {
    if (g_state) g_state->stopping.store(true);
    if (g_svr) g_svr->stop();
}

}  // namespace

int main(int argc, char** argv) {
    // Line-buffered: the stages log progress to stdout and libc block-buffers a pipe, which makes a
    // live 450 s render look dead in `docker logs`.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);

    avatar::EngineConfig ecfg;
    ecfg.hymotion_gguf = getenv_str("LONGCAT_RIG_HYMOTION",
        "/mnt/hdd/3d/avatar-shootout/_weights/hymotion/gguf/hymotion-1.0-f16.gguf");
    ecfg.llm_gguf = getenv_str("LONGCAT_RIG_LLM",
        "/home/dbrain/dev/flux2.cpp/models/text_encoders/Qwen3-8B-UD-Q4_K_XL.gguf");
    ecfg.clip_l_gguf = getenv_str("LONGCAT_RIG_CLIP_L",
        "/mnt/hdd/3d/avatar-shootout/_weights/hymotion/gguf/hymotion-clip-l-f16.gguf");
    ecfg.geo_model_dir = getenv_str("LONGCAT_RIG_GEO_MODEL", ecfg.geo_model_dir);
    ecfg.gpu_lock_path = getenv_str("LONGCAT_RIG_LOCK", "");
    ecfg.idle_unload_seconds = getenv_int("LONGCAT_RIG_IDLE_UNLOAD", 300);

    Config cfg;
    cfg.host = getenv_str("HOST", cfg.host);
    cfg.port = getenv_int("PORT", getenv_int("LONGCAT_RIG_PORT", cfg.port));
    cfg.jobs_dir = getenv_str("LONGCAT_RIG_JOBS_DIR", getenv_str("LONGCAT_RIG_OUTDIR", "/out") + "/jobs");
    cfg.min_free_mib = getenv_int("LONGCAT_RIG_MIN_FREE_MIB", 0);
    cfg.rig_retries = getenv_int("LONGCAT_RIG_RETRIES", 2);
    cfg.max_pending = getenv_int("LONGCAT_RIG_MAX_PENDING", 32);

    // Our own argv, verbatim, kept before the parser touches it: it is what the child is launched
    // with, plus the overrides avatar_worker.hpp appends. Every weight path, --lock, --min-free-mib
    // and --jobs-dir therefore reaches the worker unchanged, and so does the whole environment
    // (fork inherits it, CUDA_VISIBLE_DEVICES and every LONGCAT_RIG_* with it).
    const std::vector<std::string> original_args(argv + 1, argv + argc);
    bool worker_mode = std::getenv("AVATAR_SERVER_WORKER_CHILD") != nullptr;
    bool isolation = getenv_int("LONGCAT_RIG_WORKER_ISOLATION", 1) != 0;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--host") cfg.host = next();
        else if (a == "--port") cfg.port = std::atoi(next().c_str());
        else if (a == "--jobs-dir") cfg.jobs_dir = next();
        else if (a == "--min-free-mib") cfg.min_free_mib = std::atoi(next().c_str());
        else if (a == "--rig-retries") cfg.rig_retries = std::atoi(next().c_str());
        else if (a == "--max-pending") cfg.max_pending = std::atoi(next().c_str());
        else if (a == "--idle-unload") ecfg.idle_unload_seconds = std::atoi(next().c_str());
        else if (a == "--lock") {
            if (i + 1 < argc && argv[i + 1][0] != '-') ecfg.gpu_lock_path = next();
            else ecfg.gpu_lock_path = "/mnt/hdd/3d/avatar-shootout/_shootout_out/.3060-image-to-rig.lock";
        }
        else if (a == "--geo-model") ecfg.geo_model_dir = next();
        else if (a == "--hymotion") ecfg.hymotion_gguf = next();
        else if (a == "--llm") ecfg.llm_gguf = next();
        else if (a == "--clip-l") ecfg.clip_l_gguf = next();
        else if (a == "--cpu") ecfg.use_cuda = false;
        else if (a == "--worker") worker_mode = true;
        else if (a == "--no-worker-isolation") isolation = false;
        else if (a == "--serve") { /* accepted and ignored: the entrypoint's own flag */ }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }

    mkdirs(cfg.jobs_dir);

    // The child is a supervised process, never a service in its own right: it must not be reachable
    // from outside the container even by accident, and it must not inherit the public port.
    if (worker_mode) {
        g_log_tag = "avatar-worker";
        if (cfg.host != "127.0.0.1") cfg.host = "127.0.0.1";
    }
    const bool supervise = isolation && !worker_mode;

    State st;
    st.cfg = cfg;
    st.worker_mode = worker_mode;
    st.idle_unload_seconds = ecfg.idle_unload_seconds;
    st.started_s = now_s();
    g_state = &st;

    // ONE Engine, in whichever process owns the card. Constructing it is cheap in wall time but it
    // is NOT free on the GPU: it creates the CUDA primary context (~122 MiB, measured). That is the
    // single reason the supervising parent below never constructs one.
    std::unique_ptr<avatar::Engine> engine;
    std::unique_ptr<avsup::Worker> worker;
    if (!supervise) {
        engine = std::make_unique<avatar::Engine>(ecfg);
        st.engine = engine.get();
    } else {
        // /proc/self/exe, not argv[0]: it is the path the kernel actually loaded, so it survives a
        // PATH lookup, a relative invocation and a rename.
        char exe[4096];
        const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        std::string exe_path = (n > 0) ? std::string(exe, (size_t)n) : std::string(argv[0]);
        worker = std::make_unique<avsup::Worker>(exe_path, original_args,
                                                 [](const char* m) { log_line("%s", m); });
        st.worker = worker.get();
        log_line("worker isolation ON — this process owns NO CUDA context; NVML device %s",
                 avsup::nvml_note().c_str());
        if (avsup::nvml_free_mib() < 0 && cfg.min_free_mib > 0)
            log_line("WARN: NVML cannot be read, so the %d MiB VRAM floor cannot be enforced at "
                     "submit time. The worker still enforces it with the card in hand.",
                     cfg.min_free_mib);
    }

    httplib::Server svr;
    g_svr = &svr;
    // A rig is ~450 s and a GLB is tens of MB. httplib's 5 s default read/write timeout drops the
    // connection mid-body on a large upload (an empty-bodied 400) or mid-response on a long
    // `?wait=1` render — the same two traps the sd.cpp server documents.
    svr.set_payload_max_length(512ull * 1024 * 1024);
    svr.set_read_timeout(60 * 60, 0);
    svr.set_write_timeout(60 * 60, 0);
    svr.set_idle_interval(1, 0);
    // 🔴 A `?wait=1` rig OCCUPIES ITS HTTPLIB WORKER for the whole ladder — a quarter of an hour.
    // httplib's default pool is 8, so eight patient clients would leave /health with no thread to
    // answer on, and a service that stops answering /health while it is working perfectly is a
    // service that looks dead to every probe on the box. The work itself is still single-flight
    // (the Engine's queue decides that, not this number); these threads are almost all asleep.
    svr.new_task_queue = [] { return new httplib::ThreadPool(64); };
    svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "unknown exception";
        try { std::rethrow_exception(ep); }
        catch (const std::exception& e) { what = e.what(); }
        catch (...) {}
        res.status = 500;
        res.set_content(json({{"error", what}}).dump(), "application/json");
    });

    register_routes(svr, st);

    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);

    // ---- idle unload, which now means "kill the CUDA worker" ---------------------------------
    // 🔴 THE PARENT OWNS THIS. The Engine's own watchdog only ever dropped the motion DiT, and the
    // child is launched with --idle-unload 0 so the two cannot both act. Killing the process is
    // what actually returns the card, so this is the only idle-unload that has ever been true to
    // its name. It takes the worker's lock and refuses to kill anything with a lease outstanding,
    // so it can never evict a running render.
    std::thread idle_watchdog;
    if (supervise) {
        idle_watchdog = std::thread([&st, &worker, idle = ecfg.idle_unload_seconds] {
            while (!st.stopping.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (st.stopping.load()) break;
                worker->poll();          // reap a child that died on its own, so it is not a zombie
                if (idle <= 0) continue; // 0 = never unload; the reap above still has to happen
                if (!worker->alive() || worker->leases() > 0) continue;
                if (worker->idle_seconds() < (double)idle) continue;
                {
                    std::lock_guard<std::mutex> lk(st.jm.mx);
                    if (live_jobs(st.jm) > 0) continue;
                }
                log_line("idle %ds — killing the GPU worker (this is what returns the VRAM)", idle);
                (void)worker->kill_worker();
            }
        });
    }

    log_line("listening on http://%s:%d  (jobs %s, idle-unload %ds, rig-retries %d, min-free %d MiB%s, %s)",
             cfg.host.c_str(), cfg.port, cfg.jobs_dir.c_str(), ecfg.idle_unload_seconds,
             cfg.rig_retries, cfg.min_free_mib,
             ecfg.gpu_lock_path.empty() ? ", no flock" : ", flock held per request",
             supervise ? "worker isolation" : (worker_mode ? "CUDA worker" : "single process"));
    const bool bound = svr.listen(cfg.host, cfg.port);
    st.stopping.store(true);
    if (idle_watchdog.joinable()) idle_watchdog.join();
    if (!bound) {
        log_line("FATAL: could not bind %s:%d", cfg.host.c_str(), cfg.port);
        if (worker) (void)worker->kill_worker();
        return 1;
    }
    // A SIGTERM during a 450 s render: refuse new work, then let the in-flight one give the card
    // back. Docker's 10 s grace is shorter than that, so cancel rather than pretend.
    if (engine) engine->drain();
    else        st.draining.store(true);
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        for (auto& [_, j] : st.jm.jobs)
            if (!j->done.load()) cancel_job(j);
    }
    // And then take the card back for real. PR_SET_PDEATHSIG covers a parent that is killed
    // outright; this covers the orderly exit, and does it before the process teardown that would
    // otherwise leave a worker briefly re-parented to init still holding 10 GB.
    if (worker) (void)worker->kill_worker();
    log_line("stopped");
    return 0;
}
