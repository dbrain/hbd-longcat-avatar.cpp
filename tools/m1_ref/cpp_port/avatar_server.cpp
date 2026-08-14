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
#include "avatar_pipeline.hpp"

#include "httplib.h"
#include "json.hpp"

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

void log_line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log_line(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[avatar-server] %s\n", buf);
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

struct State {
    avatar::Engine* engine = nullptr;
    Config cfg;
    Manager jm;
    std::atomic<bool> stopping{false};
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

// Free VRAM on the pinned card, as the driver sees it right now. -1 = could not ask.
long free_vram_mib(State& st) { return st.engine->gpu_status().mem_free_mb; }

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
        out_path = job_dir + "/input";
        if (!write_file(out_path, f.content)) { err = "cannot write " + out_path; return false; }
        return true;
    }
    if (body.contains("image") && body["image"].is_string()) {
        const std::string s = body["image"].get<std::string>();
        if (s.empty()) { err = "image is empty"; return false; }
        if (source_is_path(s)) { out_path = s; return true; }
        std::string bytes;
        if (!base64_decode(s, bytes)) {
            err = "image is neither a readable absolute path nor decodable base64";
            return false;
        }
        out_path = job_dir + "/input";
        if (!write_file(out_path, bytes)) { err = "cannot write " + out_path; return false; }
        return true;
    }
    const std::string ct = req.get_header_value("Content-Type");
    if (!req.body.empty() && ct.compare(0, 6, "image/") == 0) {
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
    const long free_mib = free_vram_mib(st);
    if (free_mib < 0) return true;   // driver would not answer; do not invent a refusal
    if (free_mib >= st.cfg.min_free_mib) return true;
    err = "only " + std::to_string(free_mib) + " MiB free on the pinned card, need >= " +
          std::to_string(st.cfg.min_free_mib) +
          ". The rig stage peaks ~10.7 GB and DOES NOT DEGRADE — it would OOM minutes in. Free the "
          "card (kobbler-llama-server-1 holds ~9 GB of the 3060 without taking our lock) or lower "
          "LONGCAT_RIG_MIN_FREE_MIB.";
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

}  // namespace

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------
namespace {

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
                j->cancel->cancel();
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
    const avatar::Health h = st.engine->health();
    if (h.draining) {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        st.jm.rejected++;
        set_json(res, {{"error", "service draining — not accepting new requests"}, {"error_code", "draining"}}, 503);
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        if (live_jobs(st.jm) >= st.cfg.max_pending) {
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
    if (needs_rig_vram && h.in_flight == 0 && !h.loaded) {
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
    if (!resolve_reruns(st, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    std::vector<Tier> probe;
    if (!parse_tiers(b, probe, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!admit(st, res, /*needs_rig_vram=*/true)) return;

    std::string id, dir;
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        id = new_job_id(st.jm);
    }
    dir = make_job_dir(st, id);
    std::string image;
    if (!resolve_image(req, b, dir, image, err)) { set_json(res, {{"error", err}}, 400); return; }
    b["image"] = image;

    json echo = b;
    echo["rig_retries"] = b.value("rig_retries", st.cfg.rig_retries);
    echo["tiers"] = json::array();
    for (const auto& t : probe) echo["tiers"].push_back(json{{"name", t.name}, {"decimate", t.decimate}});
    submit(st, req, res, "rig", echo,
           [&st, b, image](JobPtr j) { run_rig(st, j, b, image); }, dir, id);
}

void handle_motion(State& st, const httplib::Request& req, httplib::Response& res) {
    json b;
    std::string err;
    if (!parse_body(req, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!b.contains("prompt") && !b.value("uncond", false)) {
        set_json(res, {{"error", "motion needs a `prompt` (or uncond:true for the checkpoint's trained null conditioning)"}}, 400);
        return;
    }
    // needs_rig_vram=false: the motion stage is nowhere near the rig's 10.7 GB, and holding it to
    // that floor refuses work that fits (see admit()).
    if (!admit(st, res, /*needs_rig_vram=*/false)) return;
    std::string id, dir;
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        id = new_job_id(st.jm);
    }
    dir = make_job_dir(st, id);
    submit(st, req, res, "motion", b,
           [&st, b](JobPtr j) { run_motion(st, j, b); }, dir, id);
}

void handle_animate(State& st, const httplib::Request& req, httplib::Response& res) {
    json b;
    std::string err;
    if (!parse_body(req, b, err)) { set_json(res, {{"error", err}}, 400); return; }

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
    std::string id, dir;
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        id = new_job_id(st.jm);
    }
    dir = make_job_dir(st, id);
    json echo = b;
    echo["rig"] = rig_glb;
    submit(st, req, res, "animate", echo,
           [&st, b, rig_glb, texts, paths](JobPtr j) { run_animate(st, j, b, rig_glb, texts, paths); },
           dir, id);
}

void handle_render(State& st, const httplib::Request& req, httplib::Response& res) {
    json b;
    std::string err;
    if (!parse_body(req, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!b.contains("prompt") && !b.value("uncond", false)) {
        set_json(res, {{"error", "render needs a `prompt`"}}, 400);
        return;
    }
    if (!resolve_reruns(st, b, err)) { set_json(res, {{"error", err}}, 400); return; }
    std::vector<Tier> probe;
    if (!parse_tiers(b, probe, err)) { set_json(res, {{"error", err}}, 400); return; }
    if (!admit(st, res, /*needs_rig_vram=*/true)) return;
    std::string id, dir;
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        id = new_job_id(st.jm);
    }
    dir = make_job_dir(st, id);
    std::string image;
    if (!resolve_image(req, b, dir, image, err)) { set_json(res, {{"error", err}}, 400); return; }
    b["image"] = image;
    json echo = b;
    submit(st, req, res, "render", echo,
           [&st, b, image](JobPtr j) { run_render(st, j, b, image); }, dir, id);
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
        const avatar::Health h = st.engine->health();
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
        });
    });

    svr.Get("/v1/gpu/status", [&st](const httplib::Request&, httplib::Response& res) {
        const avatar::GpuStatus g = st.engine->gpu_status();
        json b = {
            {"loaded", g.loaded},
            {"gpu", g.gpu.empty() ? json(nullptr) : json(g.gpu)},
            {"mem_free_mb", g.mem_free_mb},
            {"peak_vram_used_mib", (long)st.engine->peak_vram_used_mib()},
            {"baseline_vram_used_mib", (long)st.engine->baseline_vram_used_mib()},
        };
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
        j->cancel->cancel();   // returns immediately; the render stops at its next cancellation point
        set_json(res, {{"id", j->id}, {"status", j->status}, {"cancel_requested", true}});
    });

    // ---- admin, the fleet's names ----------------------------------------------------------------
    svr.Post("/v1/admin/drain", [&st](const httplib::Request&, httplib::Response& res) {
        st.engine->drain();
        const avatar::Health h = st.engine->health();
        set_json(res, {{"status", "draining"}, {"busy", h.busy}, {"in_flight", h.in_flight}, {"queued", h.queued}});
    });

    svr.Post("/v1/admin/load", [&st](const httplib::Request&, httplib::Response& res) {
        // Undrain only. Weights reload lazily on the next request, exactly as the fleet's
        // /v1/admin/load does — an eager load here would grab the card while it is still contested.
        st.engine->undrain();
        set_json(res, {{"status", "ok"}, {"loaded", st.engine->health().loaded}});
    });

    svr.Post("/v1/admin/unload", [&st](const httplib::Request& req, httplib::Response& res) {
        bool force = false;
        if (!req.body.empty()) {
            try { force = json::parse(req.body).value("force", false); } catch (...) {}
        }
        const avatar::Health before = st.engine->health();
        if (before.in_flight > 0 && !force) {
            set_json(res, {{"status", "busy (pass force=true to cancel + unload)"},
                           {"in_flight", before.in_flight}, {"queued", before.queued}}, 409);
            return;
        }
        if (force) {
            std::lock_guard<std::mutex> lk(st.jm.mx);
            for (auto& [_, j] : st.jm.jobs)
                if (!j->done.load()) j->cancel->cancel();
        }
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
        set_json(res, {{"status", was_loaded ? "unloaded" : "idle"}, {"unloaded", was_loaded}});
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            "longcat-rig avatar_server\n"
            "  POST /v1/rig       {image}                     -> 202 job; builds the LOD ladder\n"
            "                     {geometry_from:<job|dir>}    re-rig an existing geometry (~60-180s)\n"
            "                     {rig_cache_from:<job|dir>}   reuse the skeleton, re-bake only\n"
            "                     {tiers:[hero,high,medium,game]}   default: all four\n"
            "  POST /v1/motion    {prompt}                    -> 202 job (~11 s)\n"
            "  POST /v1/animate   {rig:<job|path>, tier, clips:[<job|path|json>]}  -> 202 job (0.22 s)\n"
            "  POST /v1/render    {image, prompt}             -> 202 job, the whole chain\n"
            "  add ?wait=1 to any of the four to block until it finishes\n"
            "  GET  /v1/jobs/{id}                   GET /v1/jobs/{id}/result[?tier=game]\n"
            "  POST /v1/jobs/{id}/cancel\n"
            "  GET  /health   /v1/gpu/status   /v1/metrics\n"
            "  POST /v1/admin/drain | /v1/admin/load | /v1/admin/unload\n",
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
        "  --idle-unload S       drop resident weights after S idle seconds (0 = never)\n"
        "  --lock [PATH]         cross-process GPU flock (default off)\n"
        "  --geo-model DIR --hymotion PATH --llm PATH --clip-l PATH   weight overrides\n"
        "  --cpu                 no CUDA (diagnostics only)\n");
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
        else if (a == "--serve") { /* accepted and ignored: the entrypoint's own flag */ }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }

    mkdirs(cfg.jobs_dir);

    // ONE Engine for the process. Constructing it is cheap: weights become resident on first use
    // (the fleet's LAZY_LOAD shape), and the idle watchdog gives them back after
    // idle_unload_seconds with in_flight == 0.
    avatar::Engine engine(ecfg);
    State st;
    st.engine = &engine;
    st.cfg = cfg;
    g_state = &st;

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

    log_line("listening on http://%s:%d  (jobs %s, idle-unload %ds, rig-retries %d, min-free %d MiB%s)",
             cfg.host.c_str(), cfg.port, cfg.jobs_dir.c_str(), ecfg.idle_unload_seconds,
             cfg.rig_retries, cfg.min_free_mib,
             ecfg.gpu_lock_path.empty() ? ", no flock" : ", flock held per request");
    if (!svr.listen(cfg.host, cfg.port)) {
        log_line("FATAL: could not bind %s:%d", cfg.host.c_str(), cfg.port);
        return 1;
    }
    // A SIGTERM during a 450 s render: refuse new work, then let the in-flight one give the card
    // back. Docker's 10 s grace is shorter than that, so cancel rather than pretend.
    engine.drain();
    {
        std::lock_guard<std::mutex> lk(st.jm.mx);
        for (auto& [_, j] : st.jm.jobs)
            if (!j->done.load()) j->cancel->cancel();
    }
    log_line("stopped");
    return 0;
}
