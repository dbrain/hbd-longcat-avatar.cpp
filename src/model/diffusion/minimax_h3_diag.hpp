#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "core/util.h"

// MiniMax-H3 DIAGNOSTIC BUNDLE.
//
// One render, one machine-readable artifact.  The point is NOT to test a hypothesis -- it is to
// make the NEXT twenty hypotheses testable by re-reading a file instead of re-rendering.  A render
// costs 3-15 minutes on a card that is also serving; a JSON costs milliseconds.
//
//   MINIMAX_H3_DIAG=<path-prefix>   enable.  Writes <prefix>.diag.json.  Everything below is
//                                   host-side arithmetic on tensors the pipeline already has in
//                                   RAM -- no extra graph nodes, no extra GPU work.
//   MINIMAX_H3_DIAG_GRAPH=1         additionally capture, per DiT block, the AdaLN modulation
//                                   tensors and a strided row subsample of the residual stream,
//                                   split by modality.  This DOES add graph outputs (see
//                                   minimax_h3.hpp) and is therefore opt-in.
//   MINIMAX_H3_DIAG_ROWS=<n>        rows per modality per block for the residual subsample
//                                   (default 8).  Cost is linear in this.
//   MINIMAX_H3_DIAG_NPY=<prefix>    also write .npy sidecars of the per-step audio/video
//                                   velocity + x0 arrays.  Off by default; this is the only knob
//                                   here that writes full tensors.
//
// ★ The file is rewritten after every step, atomically (tmp + rename).  A crashed or cancelled
// render therefore still leaves a VALID json describing everything up to the crash, which is the
// most valuable kind -- `"complete": false` says so.
//
// This header is deliberately free of ggml and of sd::Tensor so it can be included from the DiT,
// from stable-diffusion.cpp and from a test without dragging either in.  Everything takes raw
// pointers.
namespace MiniMaxH3Diag {

    // -------------------------------------------------------------------------------------
    // Aggregate statistics.  Aggregates, never tensors -- that is what makes this cheap enough
    // to leave on.
    // -------------------------------------------------------------------------------------

    struct Stats {
        int64_t n         = 0;
        double min        = 0.0;
        double max        = 0.0;
        double mean       = 0.0;
        double std        = 0.0;
        double absmax     = 0.0;
        double rms        = 0.0;
        int64_t nonfinite = 0;
    };

    inline Stats stats_of(const float* p, int64_t n, int64_t stride = 1) {
        Stats s;
        if (p == nullptr || n <= 0 || stride <= 0) {
            return s;
        }
        double sum = 0.0, sq = 0.0;
        bool first = true;
        int64_t counted = 0;
        for (int64_t i = 0; i < n; i += stride) {
            const double v = static_cast<double>(p[i]);
            if (!std::isfinite(v)) {
                s.nonfinite++;
                continue;
            }
            if (first) {
                s.min = s.max = v;
                first         = false;
            } else {
                s.min = std::min(s.min, v);
                s.max = std::max(s.max, v);
            }
            s.absmax = std::max(s.absmax, std::fabs(v));
            sum += v;
            sq += v * v;
            counted++;
        }
        s.n = counted;
        if (counted > 0) {
            const double dn = static_cast<double>(counted);
            s.mean          = sum / dn;
            s.rms           = std::sqrt(std::max(0.0, sq / dn));
            s.std           = std::sqrt(std::max(0.0, sq / dn - s.mean * s.mean));
        }
        return s;
    }

    // Pearson correlation of two equal-length arrays.  Returned as 0 when either side is
    // constant, which is itself diagnostic (a flat velocity has no correlation to report).
    inline double corr_of(const float* a, const float* b, int64_t n) {
        if (a == nullptr || b == nullptr || n <= 1) {
            return 0.0;
        }
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        int64_t counted = 0;
        for (int64_t i = 0; i < n; i++) {
            const double p = static_cast<double>(a[i]);
            const double q = static_cast<double>(b[i]);
            if (!std::isfinite(p) || !std::isfinite(q)) {
                continue;
            }
            sx += p;
            sy += q;
            sxx += p * p;
            syy += q * q;
            sxy += p * q;
            counted++;
        }
        if (counted <= 1) {
            return 0.0;
        }
        const double dn  = static_cast<double>(counted);
        const double cov = sxy / dn - (sx / dn) * (sy / dn);
        const double vx  = std::max(0.0, sxx / dn - (sx / dn) * (sx / dn));
        const double vy  = std::max(0.0, syy / dn - (sy / dn) * (sy / dn));
        return (vx > 0.0 && vy > 0.0) ? cov / std::sqrt(vx * vy) : 0.0;
    }

    // -------------------------------------------------------------------------------------
    // Tiny JSON emitter.  No dependency, no escaping surprises: keys are literals from this
    // tree and the only free-form strings are warnings and env values, both escaped below.
    // -------------------------------------------------------------------------------------

    inline std::string jesc(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 8);
        for (char c : s) {
            switch (c) {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    }

    inline std::string jstr(const std::string& s) {
        return "\"" + jesc(s) + "\"";
    }

    // JSON has no NaN/Inf.  Emitting `null` for one is not a formatting nicety: it is the ONLY
    // way a reader can tell "this number blew up" from "this number is huge but real".
    inline std::string jnum(double v, int prec = 6) {
        if (!std::isfinite(v)) {
            return "null";
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%.*g", prec, v);
        return buf;
    }

    inline std::string jint(int64_t v) {
        return std::to_string(v);
    }

    inline std::string jbool(bool v) {
        return v ? "true" : "false";
    }

    inline std::string jarray(const std::vector<std::string>& items, const char* sep = ",") {
        std::string out = "[";
        for (size_t i = 0; i < items.size(); i++) {
            if (i != 0) {
                out += sep;
            }
            out += items[i];
        }
        out += "]";
        return out;
    }

    // A Stats as a compact object.  `nonfinite` is always emitted, even at zero: its ABSENCE
    // would be indistinguishable from "nobody looked".
    inline std::string jstats(const Stats& s) {
        return "{\"n\":" + jint(s.n) +
               ",\"min\":" + jnum(s.min) +
               ",\"max\":" + jnum(s.max) +
               ",\"mean\":" + jnum(s.mean) +
               ",\"std\":" + jnum(s.std) +
               ",\"rms\":" + jnum(s.rms) +
               ",\"absmax\":" + jnum(s.absmax) +
               ",\"nonfinite\":" + jint(s.nonfinite) + "}";
    }

    // -------------------------------------------------------------------------------------
    // Recorder
    // -------------------------------------------------------------------------------------

    class Recorder {
    public:
        static Recorder& get() {
            static Recorder instance;
            return instance;
        }

        bool enabled() const { return enabled_; }
        // In-graph capture (AdaLN + per-block residual).  Implies enabled().
        bool graph_capture() const { return enabled_ && graph_; }
        // Per-block PRE-GATE attention / MLP outputs.  A THIRD level, not part of
        // MINIMAX_H3_DIAG_GRAPH, because it is the only capture here with a real VRAM cost:
        // holding a view of `attn_out` until the end of the block extends that tensor's lifetime
        // from mid-block to block end, which is one extra [hidden, seq] allocation live (~711 MB
        // F32 / ~356 MB F16 at production length).  Worth it when the question is "attention or
        // MLP", not worth it by default.
        bool graph_branch() const { return enabled_ && graph_ && branch_; }
        int graph_rows() const { return rows_; }
        const std::string& npy_prefix() const { return npy_prefix_; }
        bool npy() const { return enabled_ && !npy_prefix_.empty(); }

        // The sampler step currently being computed, so the DiT -- which is never told -- can
        // label what it captures.  Set from the sampler; `abs()` because a negative step is this
        // tree's flag for the unconditional pass.
        void set_step(int step) { step_ = step < 0 ? -step : step; }
        int step() const { return step_; }

        void begin_render() {
            if (!enabled_) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            sections_.clear();
            arrays_.clear();
            warnings_.clear();
            complete_ = false;
            render_++;
            start_ms_ = now_ms();
        }

        // Replace a whole named object section ("geometry", "schedule", "outputs", ...).
        void set_section(const std::string& name, const std::string& json_object) {
            if (!enabled_) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            if (!section_order_contains(name)) {
                section_order_.push_back(name);
            }
            sections_[name] = json_object;
        }

        bool has_section(const std::string& name) const {
            if (!enabled_) {
                return false;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            return sections_.find(name) != sections_.end();
        }

        // Append one element to a named array section ("trajectory", "blocks", ...).
        void append(const std::string& name, const std::string& json_element) {
            if (!enabled_) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            if (!array_order_contains(name)) {
                array_order_.push_back(name);
            }
            arrays_[name].push_back(json_element);
        }

        // ★ An anomaly the run should TELL us about, rather than us noticing three hypotheses
        // later.  Also logged at WARN so it is visible in a tailed container log.
        void warn(const std::string& text) {
            if (!enabled_) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const std::string& existing : warnings_) {
                    if (existing == text) {
                        return;  // de-duplicate: a per-step check fires every step
                    }
                }
                warnings_.push_back(text);
            }
            LOG_WARN("MiniMax-H3 DIAG: %s", text.c_str());
        }

        void mark_complete() {
            if (!enabled_) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                complete_ = true;
            }
            flush();
        }

        // Atomic rewrite.  Called after every step; the whole file is small (a few MB at 50
        // blocks x 10 steps with graph capture on) and a partial file is worse than a slow one.
        void flush() {
            if (!enabled_) {
                return;
            }
            std::string body;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                body = render_json();
            }
            const std::string tmp = path_ + ".tmp";
            FILE* f               = fopen(tmp.c_str(), "wb");
            if (f == nullptr) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    LOG_WARN("MiniMax-H3 DIAG: cannot write '%s'", tmp.c_str());
                }
                return;
            }
            fwrite(body.data(), 1, body.size(), f);
            fclose(f);
            std::rename(tmp.c_str(), path_.c_str());
        }

        const std::string& path() const { return path_; }

        // ~Recorder runs at process exit, which is where an aborted render's diagnostics would
        // otherwise be lost.  flush() is idempotent.
        ~Recorder() {
            if (enabled_) {
                flush();
            }
        }

    private:
        Recorder() {
            const char* prefix = getenv("MINIMAX_H3_DIAG");
            if (prefix == nullptr || prefix[0] == '\0' || (prefix[0] == '0' && prefix[1] == '\0')) {
                return;
            }
            enabled_ = true;
            // `MINIMAX_H3_DIAG=1` is the convenience form: land the bundle next to whatever
            // else this job writes.
            std::string p = prefix;
            if (p == "1") {
                const char* job_dir = getenv("MINIMAX_H3_JOB_DIR");
                p = (job_dir != nullptr && job_dir[0] != '\0') ? std::string(job_dir) + "/minimax_h3"
                                                               : std::string("minimax_h3");
            }
            path_ = p + ".diag.json";

            graph_  = env_flag("MINIMAX_H3_DIAG_GRAPH", false);
            branch_ = env_flag("MINIMAX_H3_DIAG_BRANCH", false);
            rows_   = env_int("MINIMAX_H3_DIAG_ROWS", 8);
            rows_  = std::max(1, std::min(rows_, 4096));
            if (const char* npy = getenv("MINIMAX_H3_DIAG_NPY"); npy != nullptr && npy[0] != '\0' && strcmp(npy, "0") != 0) {
                npy_prefix_ = (strcmp(npy, "1") == 0) ? p : std::string(npy);
            }
            start_ms_ = now_ms();
            LOG_INFO("MiniMax-H3 DIAG enabled -> %s (graph capture %s, branch capture %s, "
                     "%d subsample rows/modality/block%s)",
                     path_.c_str(),
                     graph_ ? "ON" : "off",
                     branch_ ? "ON" : "off",
                     rows_,
                     npy_prefix_.empty() ? "" : ", npy sidecars ON");
        }

        Recorder(const Recorder&)            = delete;
        Recorder& operator=(const Recorder&) = delete;

        static bool env_flag(const char* name, bool dflt) {
            const char* v = getenv(name);
            if (v == nullptr || v[0] == '\0') {
                return dflt;
            }
            return !(v[0] == '0' && v[1] == '\0');
        }

        static int env_int(const char* name, int dflt) {
            const char* v = getenv(name);
            if (v == nullptr || v[0] == '\0') {
                return dflt;
            }
            return atoi(v);
        }

        static int64_t now_ms() {
            return static_cast<int64_t>(time(nullptr)) * 1000;
        }

        bool section_order_contains(const std::string& n) const {
            return std::find(section_order_.begin(), section_order_.end(), n) != section_order_.end();
        }
        bool array_order_contains(const std::string& n) const {
            return std::find(array_order_.begin(), array_order_.end(), n) != array_order_.end();
        }

        std::string render_json() const {
            std::string out = "{\n";
            out += "  \"schema\": 1,\n";
            out += "  \"model\": \"minimax-h3\",\n";
            out += "  \"render_index\": " + jint(render_) + ",\n";
            out += "  \"started_unix_ms\": " + jint(start_ms_) + ",\n";
            out += "  \"complete\": " + std::string(complete_ ? "true" : "false") + ",\n";
            out += "  \"env\": " + env_json() + ",\n";
            for (const std::string& name : section_order_) {
                auto it = sections_.find(name);
                if (it == sections_.end()) {
                    continue;
                }
                out += "  " + jstr(name) + ": " + it->second + ",\n";
            }
            for (const std::string& name : array_order_) {
                auto it = arrays_.find(name);
                if (it == arrays_.end()) {
                    continue;
                }
                out += "  " + jstr(name) + ": [\n";
                for (size_t i = 0; i < it->second.size(); i++) {
                    out += "    " + it->second[i];
                    if (i + 1 < it->second.size()) {
                        out += ",";
                    }
                    out += "\n";
                }
                out += "  ],\n";
            }
            out += "  \"warning_count\": " + jint(static_cast<int64_t>(warnings_.size())) + ",\n";
            out += "  \"warnings\": [\n";
            for (size_t i = 0; i < warnings_.size(); i++) {
                out += "    " + jstr(warnings_[i]);
                if (i + 1 < warnings_.size()) {
                    out += ",";
                }
                out += "\n";
            }
            out += "  ]\n}\n";
            return out;
        }

        // Every env var that changes H3 numerics, recorded verbatim.  A bundle whose numbers
        // disagree with another bundle's is only interpretable if both say what they were run
        // with -- "I think that run had SECANT on" has cost us renders already.
        static std::string env_json() {
            static const char* const names[] = {
                "MINIMAX_H3_DIAG",
                "MINIMAX_H3_DIAG_GRAPH",
                "MINIMAX_H3_DIAG_BRANCH",
                "MINIMAX_H3_DIAG_ROWS",
                // These three change nothing about the numbers and everything about the VRAM
                // curve, which is why a bundle that does not name them cannot be compared to one
                // that does -- the same render at the same settings has two different peaks
                // depending on whether the VAEs were evicted and whether staging was prefetched.
                "MINIMAX_H3_DIAG_VRAM",
                "MINIMAX_H3_EVICT_VAE",
                "MINIMAX_H3_STAGE_PREFETCH",
                "MINIMAX_H3_AUDIO_SECANT",
                "MINIMAX_H3_DIT_F16",
                "MINIMAX_H3_AUDIO_X_SCALE",
                "MINIMAX_H3_AUDIO_X_ZERO",
                "MINIMAX_H3_AUDIO_X_IMPULSE",
                "MINIMAX_H3_INJECT_AUDIO_LATENT",
                "MINIMAX_H3_LOAD_LATENT",
                "MINIMAX_H3_MLP_CHUNK",
                "MINIMAX_H3_PE_CACHE",
                "MINIMAX_H3_QKV_DEINTERLEAVE",
                "MINIMAX_H3_ISOLATION",
                "MINIMAX_H3_STAGED_TE",
                "MINIMAX_H3_MODALITY_NUM",
                "MINIMAX_H3_AUDIO_LATENT_STATS",
                "GGML_NVFP4_CUBLASLT",
                "SD_FLASH_ATTENTION",
            };
            std::string out = "{";
            bool first      = true;
            for (const char* n : names) {
                const char* v = getenv(n);
                if (!first) {
                    out += ",";
                }
                first = false;
                out += jstr(n) + ":" + (v == nullptr ? "null" : jstr(v));
            }
            out += "}";
            return out;
        }

        mutable std::mutex mutex_;
        bool enabled_ = false;
        bool graph_   = false;
        bool branch_  = false;
        int rows_     = 8;
        int step_     = 0;
        int64_t render_   = 0;
        int64_t start_ms_ = 0;
        bool complete_    = false;
        std::string path_;
        std::string npy_prefix_;
        std::vector<std::string> section_order_;
        std::vector<std::string> array_order_;
        std::map<std::string, std::string> sections_;
        std::map<std::string, std::vector<std::string>> arrays_;
        std::vector<std::string> warnings_;
    };

    inline Recorder& rec() {
        return Recorder::get();
    }

    inline bool on() {
        return Recorder::get().enabled();
    }

    inline bool graph_on() {
        return Recorder::get().graph_capture();
    }

    // -------------------------------------------------------------------------------------
    // Thresholds the run checks itself against.
    //
    // The audio numbers are the ComfyUI reference clip's, measured on the same prompt; they are
    // TARGETS, not tolerances, and are written into the bundle so a reader never has to go
    // looking for what "too loud" meant.
    // -------------------------------------------------------------------------------------
    struct Thresholds {
        // decoded PCM
        double pcm_std_ref     = 0.0085;
        double pcm_peak_ref    = 0.0486;
        double pcm_lr_corr_ref = 0.607;
        // a peak past full scale is a clip, full stop
        double pcm_peak_clip = 1.0;
        // how far from the reference std before it is worth a warning (ratio, both directions)
        double pcm_std_ratio_warn = 4.0;
        double pcm_lr_corr_min    = 0.2;
        // a correct rectified flow's velocity/noise correlation at sigma ~ 1
        double rho_min_at_high_sigma = 0.2;
        double high_sigma            = 0.9;
        // consecutive-block residual RMS ratio that counts as a step change
        double block_rms_ratio_warn = 10.0;
    };

    inline const Thresholds& thresholds() {
        static const Thresholds t;
        return t;
    }

    inline std::string thresholds_json() {
        const Thresholds& t = thresholds();
        return "{\"pcm_std_ref\":" + jnum(t.pcm_std_ref) +
               ",\"pcm_peak_ref\":" + jnum(t.pcm_peak_ref) +
               ",\"pcm_lr_corr_ref\":" + jnum(t.pcm_lr_corr_ref) +
               ",\"pcm_peak_clip\":" + jnum(t.pcm_peak_clip) +
               ",\"pcm_std_ratio_warn\":" + jnum(t.pcm_std_ratio_warn) +
               ",\"pcm_lr_corr_min\":" + jnum(t.pcm_lr_corr_min) +
               ",\"rho_min_at_high_sigma\":" + jnum(t.rho_min_at_high_sigma) +
               ",\"high_sigma\":" + jnum(t.high_sigma) +
               ",\"block_rms_ratio_warn\":" + jnum(t.block_rms_ratio_warn) + "}";
    }

}  // namespace MiniMaxH3Diag

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_HPP__
