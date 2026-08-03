#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_VRAM_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_VRAM_HPP__

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "model/diffusion/minimax_h3_diag.hpp"

// PHASE-RESOLVED VRAM ACCOUNTING for MiniMax-H3.
//
// The question this exists to answer, in numbers rather than in argument: **at the peak of a
// render, how many MiB are model weights sitting idle, and how many are live activations?**
// Until this header, the only breakdown available was the load-time `total params memory size`
// line, which describes PLACEMENT AT REGISTRATION and says nothing about what is resident when
// the ceiling is actually hit.
//
// ★ THE LEDGER IS DOUBLE-ENTRY ON PURPOSE.  Every sample records what the DRIVER says is used
// (`ggml_backend_dev_memory`) and, separately, what this process can ACCOUNT for (params blocks,
// staging blocks, prefetch buffers, the graph compute buffer, the runner cache buffer).  The
// difference is emitted as `unattributed_mib` and it is the most informative field in the whole
// bundle: it is CUDA pool slack, fragmentation, cuBLAS/cuDNN workspaces and any other process on
// the card.  A breakdown that silently balances is a breakdown that has been fitted.
//
//   MINIMAX_H3_DIAG_VRAM=1   enable.  Independent of MINIMAX_H3_DIAG: with DIAG on the ledger is
//                            written into the bundle as the "vram" section; with DIAG off it
//                            still logs a compact line per phase, so a bare perf run gets the
//                            numbers without paying for the rest of the bundle.
//   MINIMAX_H3_DIAG_VRAM_SEGMENTS=1
//                            also sample at EVERY graph-cut segment boundary, not just at phase
//                            boundaries and sampler steps.  This is where the true ceiling lives
//                            (peak VRAM is reached inside step 1, inside one segment), so it is
//                            on by default; set to 0 to drop it if the cudaMemGetInfo cost ever
//                            shows up (~20 us x segments x steps).
//   MINIMAX_H3_DIAG_VRAM_MAX_SAMPLES=<n>
//                            cap on retained samples (default 4096).  Phase/step samples are
//                            always kept; segment samples are kept only when they RAISE the peak,
//                            because a segment that does not move the ceiling is not evidence.
//
// ★ Every entry point is inert unless the env var is set, so ltx-video, ltx-2b, flux2, flux2-4b,
// wan-vace and longcat-avatar see one `bool` test per graph segment and nothing else.
namespace MiniMaxH3Diag {

    // -----------------------------------------------------------------------------------------
    // Enablement
    // -----------------------------------------------------------------------------------------

    inline bool vram_env_flag(const char* name, bool dflt) {
        const char* v = getenv(name);
        if (v == nullptr || v[0] == '\0') {
            return dflt;
        }
        return !(v[0] == '0' && v[1] == '\0');
    }

    inline bool vram_on() {
        static const bool on = vram_env_flag("MINIMAX_H3_DIAG_VRAM", false);
        return on;
    }

    inline bool vram_segments_on() {
        static const bool on = vram_on() && vram_env_flag("MINIMAX_H3_DIAG_VRAM_SEGMENTS", true);
        return on;
    }

    // -----------------------------------------------------------------------------------------
    // What the ModelManager can tell us.  Mirrors ModelManager::ParamsResidency field for field;
    // deliberately a separate POD so this header does not include model_manager.h (which drags in
    // the loader and, through lora.hpp, half the tree).
    // -----------------------------------------------------------------------------------------

    struct ModuleResidency {
        std::string desc;
        // Weights parked in the params backend.  `vram` is the interesting one: those bytes are
        // held for the whole process life once loaded, whatever phase is running.
        int64_t params_vram = 0;
        int64_t params_ram  = 0;
        // Weights copied onto the compute backend for the graph currently in flight.  Transient
        // by construction -- released when the segment's staging block is freed.
        int64_t staged_vram    = 0;
        int64_t params_tensors = 0;
        int64_t staged_tensors = 0;
    };

    struct Residency {
        std::vector<ModuleResidency> modules;
        int64_t params_vram   = 0;
        int64_t params_ram    = 0;
        int64_t staged_vram   = 0;
        int64_t prefetch_vram = 0;
        int64_t params_blocks = 0;
        int64_t staged_blocks = 0;
    };

    // -----------------------------------------------------------------------------------------
    // Which module each phase is actually USING.  Everything else that holds device memory in
    // that phase is, by definition, idle -- and that is the number the whole exercise is for.
    //
    // ★ This is a table, not an inference.  Getting it wrong would misattribute idle bytes to
    // live ones, which is exactly the error that a plausible-looking breakdown hides.
    // -----------------------------------------------------------------------------------------
    inline bool module_is_live_in_phase(const std::string& phase, const std::string& desc) {
        auto has = [&desc](const char* needle) { return desc.find(needle) != std::string::npos; };
        if (phase == "text_encode") {
            return has("Conditioner") || has("CLIP vision") || has("Whisper");
        }
        if (phase == "vae_encode") {
            return has("VAE");
        }
        if (phase == "sampling") {
            return has("Diffusion model") || has("High noise") || has("ControlNet") || has("IP-Adapter");
        }
        if (phase == "video_decode") {
            return desc == "VAE" || desc == "preview VAE";
        }
        if (phase == "audio_decode") {
            return has("audio VAE");
        }
        // "start" / "end" / anything unlabelled: nothing is running, so nothing is live.
        return false;
    }

    // -----------------------------------------------------------------------------------------
    // One observation
    // -----------------------------------------------------------------------------------------

    struct VramSample {
        std::string phase;  // text_encode | vae_encode | sampling | video_decode | audio_decode | start | end
        std::string label;  // free-form: the runner desc, the segment, why the probe fired
        int step    = -1;   // sampler step, or -1
        int segment = -1;   // graph-cut segment index, or -1
        int segments_total = -1;

        int64_t dev_free  = 0;
        int64_t dev_total = 0;

        Residency res;
        // The graph's activation working set.  ggml sizes this ONCE per graph, at reserve time, to
        // the graph's own peak -- so this is not a sample of a fluctuating quantity, it IS the
        // activation peak for the graph in flight.
        int64_t compute_buffer = 0;
        // Rope tables and reusable graph scratch the runner keeps between graphs.
        int64_t cache_buffer = 0;

        int64_t dev_used() const { return dev_total - dev_free; }

        int64_t accounted() const {
            return res.params_vram + res.staged_vram + res.prefetch_vram + compute_buffer + cache_buffer;
        }

        // Driver-reported usage minus everything this process can name.  Pool slack, fragmentation,
        // cuBLAS/cuDNN workspaces, other processes on the card.  Can go negative if another
        // process frees between the two reads; emitted signed rather than clamped, because a
        // negative here means the two clocks disagree and the reader must know that.
        int64_t unattributed() const { return dev_used() - accounted(); }

        // ★ THE ANSWER.  Weights resident on the device that this phase does not touch.
        int64_t idle_weight_vram() const {
            int64_t idle = 0;
            for (const ModuleResidency& m : res.modules) {
                if (!module_is_live_in_phase(phase, m.desc)) {
                    idle += m.params_vram;
                }
            }
            return idle;
        }
    };

    // -----------------------------------------------------------------------------------------
    // Ledger
    // -----------------------------------------------------------------------------------------

    class VramLedger {
    public:
        static VramLedger& get() {
            static VramLedger instance;
            return instance;
        }

        void begin_render() {
            if (!vram_on()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            samples_.clear();
            have_peak_ = false;
            peak_      = VramSample();
        }

        void record(VramSample sample) {
            if (!vram_on()) {
                return;
            }
            bool new_peak = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!have_peak_ || sample.dev_used() > peak_.dev_used()) {
                    peak_      = sample;
                    have_peak_ = true;
                    new_peak   = true;
                }
                const bool is_boundary = sample.segment < 0;
                if ((is_boundary || new_peak) &&
                    static_cast<int>(samples_.size()) < max_samples()) {
                    samples_.push_back(sample);
                }
            }
            // ★ Logged, not merely filed.  A number that only exists in a JSON nobody opens is a
            // number that does not exist; the peak line in particular has to be visible in a
            // tailed container log while a render is still running.
            if (sample.segment < 0 || new_peak) {
                LOG_INFO("[H3_VRAM] %-12s %-28s used=%.0f MiB | weights=%.0f (idle %.0f) staged=%.0f "
                         "prefetch=%.0f compute=%.0f cache=%.0f | unattributed=%.0f%s",
                         sample.phase.c_str(),
                         sample.label.c_str(),
                         mib(sample.dev_used()),
                         mib(sample.res.params_vram),
                         mib(sample.idle_weight_vram()),
                         mib(sample.res.staged_vram),
                         mib(sample.res.prefetch_vram),
                         mib(sample.compute_buffer),
                         mib(sample.cache_buffer),
                         mib(sample.unattributed()),
                         new_peak ? "  <- PEAK" : "");
            }
        }

        // Writes the "vram" section of the diagnostic bundle.  No-op when MINIMAX_H3_DIAG is off,
        // which is the case a bare perf run wants: the log lines above are the whole product then.
        void emit() const {
            if (!vram_on() || !Recorder::get().enabled()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            std::string js = "{\"schema\":1";
            js += ",\"device_total_mib\":" + jnum(mib(have_peak_ ? peak_.dev_total : 0), 8);
            js += ",\"peak\":" + (have_peak_ ? sample_json(peak_) : std::string("null"));
            if (have_peak_) {
                // The one-line verdict, spelled out so a reader never re-derives it.
                js += ",\"verdict\":{";
                js += "\"peak_phase\":" + jstr(peak_.phase);
                js += ",\"peak_used_mib\":" + jnum(mib(peak_.dev_used()), 8);
                js += ",\"weights_resident_mib\":" + jnum(mib(peak_.res.params_vram), 8);
                js += ",\"weights_idle_mib\":" + jnum(mib(peak_.idle_weight_vram()), 8);
                js += ",\"weights_staged_mib\":" + jnum(mib(peak_.res.staged_vram), 8);
                js += ",\"activations_mib\":" + jnum(mib(peak_.compute_buffer), 8);
                js += ",\"unattributed_mib\":" + jnum(mib(peak_.unattributed()), 8);
                js += "}";
            }
            std::vector<std::string> rows;
            rows.reserve(samples_.size());
            for (const VramSample& s : samples_) {
                rows.push_back(sample_json(s));
            }
            js += ",\"samples\":" + jarray(rows);
            js += "}";
            Recorder::get().set_section("vram", js);
        }

        bool have_peak() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return have_peak_;
        }

        VramSample peak() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return peak_;
        }

    private:
        VramLedger()                             = default;
        VramLedger(const VramLedger&)            = delete;
        VramLedger& operator=(const VramLedger&) = delete;

        static double mib(int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

        static int max_samples() {
            static const int n = [] {
                const char* v = getenv("MINIMAX_H3_DIAG_VRAM_MAX_SAMPLES");
                const int parsed = (v != nullptr && v[0] != '\0') ? atoi(v) : 4096;
                return std::max(16, std::min(parsed, 1 << 16));
            }();
            return n;
        }

        static std::string sample_json(const VramSample& s) {
            std::string js = "{\"phase\":" + jstr(s.phase);
            js += ",\"label\":" + jstr(s.label);
            js += ",\"step\":" + jint(s.step);
            js += ",\"segment\":" + jint(s.segment);
            js += ",\"segments_total\":" + jint(s.segments_total);
            js += ",\"dev_used_mib\":" + jnum(mib(s.dev_used()), 8);
            js += ",\"dev_free_mib\":" + jnum(mib(s.dev_free), 8);
            js += ",\"weights_vram_mib\":" + jnum(mib(s.res.params_vram), 8);
            js += ",\"weights_ram_mib\":" + jnum(mib(s.res.params_ram), 8);
            js += ",\"weights_idle_vram_mib\":" + jnum(mib(s.idle_weight_vram()), 8);
            js += ",\"staged_vram_mib\":" + jnum(mib(s.res.staged_vram), 8);
            js += ",\"prefetch_vram_mib\":" + jnum(mib(s.res.prefetch_vram), 8);
            js += ",\"compute_buffer_mib\":" + jnum(mib(s.compute_buffer), 8);
            js += ",\"cache_buffer_mib\":" + jnum(mib(s.cache_buffer), 8);
            js += ",\"accounted_mib\":" + jnum(mib(s.accounted()), 8);
            js += ",\"unattributed_mib\":" + jnum(mib(s.unattributed()), 8);
            js += ",\"params_blocks\":" + jint(s.res.params_blocks);
            js += ",\"staged_blocks\":" + jint(s.res.staged_blocks);
            std::vector<std::string> mods;
            mods.reserve(s.res.modules.size());
            for (const ModuleResidency& m : s.res.modules) {
                mods.push_back("{\"desc\":" + jstr(m.desc) +
                               ",\"params_vram_mib\":" + jnum(mib(m.params_vram), 8) +
                               ",\"params_ram_mib\":" + jnum(mib(m.params_ram), 8) +
                               ",\"staged_vram_mib\":" + jnum(mib(m.staged_vram), 8) +
                               ",\"params_tensors\":" + jint(m.params_tensors) +
                               ",\"staged_tensors\":" + jint(m.staged_tensors) +
                               ",\"live_in_phase\":" + jbool(module_is_live_in_phase(s.phase, m.desc)) + "}");
            }
            js += ",\"modules\":" + jarray(mods);
            js += "}";
            return js;
        }

        mutable std::mutex mutex_;
        std::vector<VramSample> samples_;
        VramSample peak_;
        bool have_peak_ = false;
    };

    inline VramLedger& vram() {
        return VramLedger::get();
    }

    // Convenience: read the device counters off a backend.  Returns false for a null or non-device
    // backend, in which case the caller should not record at all -- a sample with dev_total 0 would
    // report an absurd `unattributed` and poison the peak.
    inline bool query_device_memory(ggml_backend_t backend, int64_t* free_bytes, int64_t* total_bytes) {
        if (backend == nullptr) {
            return false;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) {
            return false;
        }
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(dev, &f, &t);
        if (t == 0) {
            return false;
        }
        *free_bytes  = static_cast<int64_t>(f);
        *total_bytes = static_cast<int64_t>(t);
        return true;
    }

}  // namespace MiniMaxH3Diag

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_VRAM_HPP__
