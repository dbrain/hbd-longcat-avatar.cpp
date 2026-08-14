// hymotion_engine.hpp — HY-Motion 1.0 text->motion as a RESIDENT object, not a process.
//
// examples/hymotion/main.cpp used to be the whole implementation: load, encode, solve, decode,
// write, exit. That is fine for a CLI and useless for a service, which wants to pay the DiT load
// ONCE and then answer many prompts. This header is that implementation lifted out of main(),
// unchanged in what it computes, with an explicit residency policy attached.
//
// RESIDENCY, WHICH IS THE WHOLE POINT
//   * The DiT (~2.1 GB f16) is loaded by load() and STAYS on the backend until unload() or
//     destruction. Every generate() after the first pays no load at all.
//   * The TEXT ENCODERS are the opposite: Qwen3-8B is ~5 GB and is needed for exactly one forward
//     per request (the reference computes conditioning once, before the ODE solve —
//     motion_diffusion.py:508). encode() loads them, runs them, and FREES them inside the call.
//     Keeping an 8B model resident to spend 0.5 s on 128 tokens would cost a service 40% of a
//     3060 permanently.
//   * On the FIRST request the encoders run before the DiT is allocated (`load_dit_after_encode`),
//     which is the ordering the standalone tool always used. On later requests the DiT is already
//     resident and the encoders load beside it: 2.1 + ~5.3 GB, which fits the 3060 with room. Set
//     `release_dit_for_encode` if a caller ever needs the old strict isolation back.
//
// WHAT IS NOT HERE: no file I/O beyond reading weights, no clip JSON, no process exit. generate()
// returns a DecodedMotion. The caller decides what that becomes.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include "core/tensor.hpp"
#include "model_loader.h"
#include "model/diffusion/hymotion.hpp"
#include "model/diffusion/hymotion_decode.hpp"
#include "model/te/hymotion_text.hpp"
#include "weight_manager.h"

// Cooperative cancellation. Relative because this header lives in examples/ and the hook lives
// with the pipeline that shares it; it is header-only and pulls in nothing but <atomic>.
#include "../../tools/m1_ref/cpp_port/cancel_hook.hpp"

namespace HYMotion {

// Pass-through weight manager. Master routes every runner's parameter residency through
// ModelManager: a GGMLRunner refuses to execute a graph touching parameter tensors unless a
// RunnerWeightManager is attached to stage them. This engine needs none of that machinery — each
// runner is allocated wholly onto the one compute backend via alloc_all_params_on() and freed
// outright — so the honest implementation is to say so rather than fake a staging step.
struct ResidentWeightManager : public RunnerWeightManager {
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return true; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return true; }
    bool retain_compute_backend_params(const std::vector<ggml_tensor*>&) override { return true; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void release_retained_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void release_params_backend_params(const std::vector<ggml_tensor*>&) override {}
};

struct EngineConfig {
    std::string model_gguf;    // hymotion-1.0-*.gguf
    std::string qwen3_gguf;    // Qwen3-8B (llama.cpp naming is converted on load)
    std::string clip_l_gguf;   // CLIP-L (tools/convert_hymotion_clip.py)
    bool        force_cpu = false;
    int         threads = 4;
    bool        verbose = true;
    // Strict VRAM isolation: drop the DiT before the encoders load, reload it after. OFF by
    // default because 2.1 + 5.3 GB fits an 11.9 GB card and reloading costs seconds per request.
    bool        release_dit_for_encode = false;
};

struct GenRequest {
    std::string prompt;              // "" + uncond, or a real prompt
    bool        uncond = false;      // the ckpt's own trained null conditioning (no text encoder)
    float       cfg = 5.0f;
    float       seconds = 4.0f;      // <= 12
    int         steps = 50;
    uint32_t    seed = 0;            // our own RNG; NOT torch-compatible (see main.cpp)
    std::string noise_path;          // optional raw f32 [360*201] y0, for a golden comparison
    std::string ctxt_path, vtxt_path;// optional precomputed conditioning
    int64_t     ctxt_len = 128;
    std::string name = "hymotion";
};

struct GenStats {
    double load_s = 0;     // 0 on every request after the first — that IS the reuse
    double encode_s = 0;
    double solve_s = 0;
    bool   dit_was_resident = false;
};

class Engine {
public:
    explicit Engine(EngineConfig cfg) : cfg_(std::move(cfg)) {}
    ~Engine() { unload(); free_backend(); }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool dit_resident() const { return dit_buf_ != nullptr; }
    const Params& params() const { return p_; }

    // Load the DiT + the pipeline-level tensors onto the backend and KEEP them there.
    bool load(std::string& err) {
        if (dit_buf_) return true;
        if (!backend_ && !init_backend(err)) return false;
        loader_ = std::make_unique<ModelLoader>();
        if (!loader_->init_from_file(cfg_.model_gguf)) {
            err = "failed to load " + cfg_.model_gguf;
            return false;
        }
        runner_ = std::make_unique<HYMotionRunner>(backend_, wm_, p_, loader_->get_tensor_storage_map());
        dit_buf_ = runner_->alloc_all_params_on(backend_);
        if (!dit_buf_) { err = "DiT alloc failed"; return false; }
        std::map<std::string, ggml_tensor*> tensors;
        runner_->get_param_tensors(tensors, "");

        // mean/std/null feats live at the ckpt TOP level, not inside motion_transformer, so they
        // are not part of the DiT block tree. Allocate them in their own context and let the same
        // loader pass fill them alongside the block weights. NB they come from the CHECKPOINT, not
        // from stats/{Mean,Std}.npy — the reference registers the npy and then load_state_dict
        // overwrites it with these, so the ckpt values are the authoritative ones.
        struct ggml_init_params ip = {
            /*mem_size  =*/ggml_tensor_overhead() * 8 + (size_t)(201 + 201 + 768 + 4096) * sizeof(float) + 4096,
            /*mem_buffer=*/nullptr,
            /*no_alloc  =*/false,
        };
        pipe_ctx_ = ggml_init(ip);
        auto t_mean = ggml_new_tensor_1d(pipe_ctx_, GGML_TYPE_F32, 201);
        auto t_std = ggml_new_tensor_1d(pipe_ctx_, GGML_TYPE_F32, 201);
        auto t_nvtxt = ggml_new_tensor_1d(pipe_ctx_, GGML_TYPE_F32, 768);
        auto t_nctxt = ggml_new_tensor_1d(pipe_ctx_, GGML_TYPE_F32, 4096);
        tensors["pipe.mean"] = t_mean;
        tensors["pipe.std"] = t_std;
        tensors["pipe.null_vtxt_feat"] = t_nvtxt;
        tensors["pipe.null_ctxt_input"] = t_nctxt;
        if (!loader_->load_tensors(tensors)) { err = "failed to load tensors"; return false; }
        mean_.resize(201); std_.resize(201); null_vtxt_.resize(768); null_ctxt_.resize(4096);
        memcpy(mean_.data(), t_mean->data, mean_.size() * sizeof(float));
        memcpy(std_.data(), t_std->data, std_.size() * sizeof(float));
        memcpy(null_vtxt_.data(), t_nvtxt->data, null_vtxt_.size() * sizeof(float));
        memcpy(null_ctxt_.data(), t_nctxt->data, null_ctxt_.size() * sizeof(float));
        return true;
    }

    // Drop the DiT's VRAM. The backend is kept (a service will want it again).
    void unload() {
        runner_.reset();
        if (dit_buf_) { ggml_backend_buffer_free(dit_buf_); dit_buf_ = nullptr; }
        if (pipe_ctx_) { ggml_free(pipe_ctx_); pipe_ctx_ = nullptr; }
        loader_.reset();   // ModelLoader is move-only; drop it rather than reassign
    }

    // Qwen3-8B -> ctxt[128*4096], CLIP-L -> vtxt[768]. BOTH encoders are freed before returning.
    bool encode(const std::string& prompt, std::vector<float>& ctxt, int64_t& ctxt_length,
                std::vector<float>& vtxt, std::string& err) {
        if (!backend_ && !init_backend(err)) return false;
        if (cfg_.qwen3_gguf.empty() || cfg_.clip_l_gguf.empty()) {
            err = "a prompt needs both --qwen3 and --clip-l weights";
            return false;
        }
        if (strlen(HYMotionText::SYSTEM_PROMPT) != HYMotionText::SYSTEM_PROMPT_LEN) {
            err = "SYSTEM_PROMPT literal has been altered (leading \"\\n    \" / trailing \"\\n\" matter)";
            return false;
        }
        {   // --- Qwen3-8B ---
            ModelLoader qloader;
            if (!qloader.init_from_file(cfg_.qwen3_gguf, "text_encoders.llm.")) {
                err = "failed to load qwen3 from " + cfg_.qwen3_gguf;
                return false;
            }
            qloader.convert_tensors_name();
            LLM::LLMRunner llm(LLM::LLMArch::QWEN3, backend_, qloader.get_tensor_storage_map(),
                               "text_encoders.llm", /*enable_vision*/ false, wm_);
            ggml_backend_buffer_t buf = llm.alloc_all_params_on(backend_);
            if (!buf) { err = "qwen3 alloc failed"; return false; }
            std::map<std::string, ggml_tensor*> qt;
            llm.get_param_tensors(qt, "text_encoders.llm");
            if (!qloader.load_tensors(qt)) { ggml_backend_buffer_free(buf); err = "failed to load qwen3 tensors"; return false; }
            Qwen2Tokenizer qtok;   // Qwen3 reuses the Qwen2 BPE vocab/merges
            std::string rep;
            const int64_t crop_start = HYMotionText::compute_crop_start(qtok, &rep);
            if (cfg_.verbose) fprintf(stderr, "hymotion: %s\n", rep.c_str());
            if (crop_start < 0) { ggml_backend_buffer_free(buf); err = "crop_start derivations disagreed"; return false; }
            const bool ok = HYMotionText::encode_ctxt(llm, qtok, prompt, crop_start, p_.max_length_llm,
                                                      p_.ctxt_input_dim, cfg_.threads, ctxt, ctxt_length,
                                                      cfg_.verbose);
            ggml_backend_buffer_free(buf);       // <-- the 8B model leaves VRAM here
            if (!ok) { err = "qwen3 encode failed"; return false; }
        }
        {   // --- CLIP-L ---
            ModelLoader cloader;
            if (!cloader.init_from_file(cfg_.clip_l_gguf)) { err = "failed to load clip-l from " + cfg_.clip_l_gguf; return false; }
            CLIPTextModelRunner clip(backend_, cloader.get_tensor_storage_map(), "text_model",
                                     OPENAI_CLIP_VIT_L_14, /*with_final_ln*/ true,
                                     /*force_clip_f32*/ false, wm_);
            ggml_backend_buffer_t buf = clip.alloc_all_params_on(backend_);
            if (!buf) { err = "clip-l alloc failed"; return false; }
            std::map<std::string, ggml_tensor*> ct;
            clip.get_param_tensors(ct, "text_model");
            if (!cloader.load_tensors(ct)) { ggml_backend_buffer_free(buf); err = "failed to load clip-l tensors"; return false; }
            CLIPTokenizer ctok;
            const bool ok = HYMotionText::encode_vtxt(clip, ctok, prompt, p_.vtxt_input_dim, cfg_.threads,
                                                      vtxt, cfg_.verbose);
            ggml_backend_buffer_free(buf);
            if (!ok) { err = "clip-l encode failed"; return false; }
        }
        return true;
    }

    // One request. Loads the DiT on the first call only.
    bool generate(const GenRequest& req, DecodedMotion& out, GenStats* stats, std::string& err) {
        GenStats st;
        st.dit_was_resident = dit_resident();

        std::vector<float> ctxt, vtxt;
        int64_t text_valid = req.ctxt_len;
        const bool have_prompt = !req.prompt.empty() && !req.uncond;
        float cfg = req.cfg;

        // PHASE-BOUNDARY CANCELLATION POINTS. The text encode (Qwen3-8B load + one prefill) and the
        // DiT load are each a single call into sd.cpp's loader/graph and have no interior hook, so a
        // cancel arriving inside one cannot land until it ends. Checking BETWEEN the phases is what
        // stops a cancelled request from going on to pay for the next one: MEASURED, a cancel 8 s
        // into a cold request used to return 18.5 s later because it still ran the encode AND the
        // DiT load AND reached the first Euler step. A warm engine skips both phases entirely.
        cancelhook::check();
        if (have_prompt) {
            // Strict isolation is opt-in; see the header comment on residency.
            if (cfg_.release_dit_for_encode && dit_resident()) unload();
            const double t = now_s();
            if (!encode(req.prompt, ctxt, text_valid, vtxt, err)) return false;
            st.encode_s = now_s() - t;
        }
        cancelhook::check();

        if (!dit_resident()) {
            const double t = now_s();
            if (!load(err)) return false;
            st.load_s = now_s() - t;
        }
        cancelhook::check();

        const int64_t T_full = p_.train_frames;
        int64_t frames = (int64_t)llround(req.seconds * (double)p_.fps);
        frames = std::max<int64_t>(20, std::min<int64_t>(T_full, frames));
        const int64_t L_t = p_.max_length_llm;

        if (req.uncond) {
            // Broadcast the learned null feats: null_ctxt_input is (1,1,4096) and the reference
            // .expand()s it across all 128 text positions.
            ctxt.assign((size_t)(L_t * p_.ctxt_input_dim), 0.0f);
            for (int64_t i = 0; i < L_t; ++i)
                std::copy(null_ctxt_.begin(), null_ctxt_.end(), ctxt.begin() + (size_t)(i * p_.ctxt_input_dim));
            vtxt = null_vtxt_;
            text_valid = L_t;
            cfg = 1.0f;    // uncondition_mode disables CFG in the reference
        } else if (!have_prompt) {
            if (req.ctxt_path.empty() || req.vtxt_path.empty()) {
                err = "need a prompt, uncond, or both ctxt and vtxt";
                return false;
            }
            ctxt = read_f32(req.ctxt_path, (size_t)(L_t * p_.ctxt_input_dim), err);
            vtxt = read_f32(req.vtxt_path, (size_t)p_.vtxt_input_dim, err);
            if (!err.empty()) return false;
        }

        const bool do_cfg = cfg > 1.0f && !req.uncond;
        const int64_t N = do_cfg ? 2 : 1;
        runner_->L_m = T_full;
        runner_->L_t = L_t;
        runner_->N = N;

        // CFG batch order is [uncond, cond] -- x_pred = basic + scale*(text - basic).
        runner_->ctxt_vec.assign((size_t)(N * L_t * p_.ctxt_input_dim), 0.0f);
        runner_->vtxt_vec.assign((size_t)(N * p_.vtxt_input_dim), 0.0f);
        if (do_cfg) {
            // enable_ctxt_null_feat: the uncond branch uses null_ctxt_input broadcast over all
            // 128 positions, NOT the real text.
            for (int64_t i = 0; i < L_t; ++i)
                std::copy(null_ctxt_.begin(), null_ctxt_.end(),
                          runner_->ctxt_vec.begin() + (size_t)(i * p_.ctxt_input_dim));
            std::copy(null_vtxt_.begin(), null_vtxt_.end(), runner_->vtxt_vec.begin());
            std::copy(ctxt.begin(), ctxt.end(), runner_->ctxt_vec.begin() + (size_t)(L_t * p_.ctxt_input_dim));
            std::copy(vtxt.begin(), vtxt.end(), runner_->vtxt_vec.begin() + (size_t)p_.vtxt_input_dim);
        } else {
            runner_->ctxt_vec = ctxt;
            runner_->vtxt_vec = vtxt;
        }

        std::vector<float> y((size_t)(T_full * p_.input_dim));
        if (!req.noise_path.empty()) {
            y = read_f32(req.noise_path, y.size(), err);
            if (!err.empty()) return false;
        } else {
            std::mt19937 rng(req.seed);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto& v : y) v = nd(rng);
        }

        // Euler ODE solve, t = linspace(0,1,steps+1):  y_{n+1} = y_n + dt * f(t_n, y_n)
        const double t_solve = now_s();
        const float dt = 1.0f / (float)req.steps;
        for (int s = 0; s < req.steps; ++s) {
            // CANCELLATION POINT — the motion diffusion sampler. Quantum = one Euler step.
            cancelhook::check();
            const float t = (float)s / (float)req.steps;
            runner_->x_vec.assign((size_t)(N * T_full * p_.input_dim), 0.0f);
            for (int64_t n = 0; n < N; ++n)
                std::copy(y.begin(), y.end(), runner_->x_vec.begin() + (size_t)(n * T_full * p_.input_dim));
            auto pred = runner_->compute(cfg_.threads, t, frames, text_valid);
            if (pred.empty()) { err = "forward failed at step " + std::to_string(s); return false; }
            const float* d = pred.data();
            const size_t stride = (size_t)(T_full * p_.input_dim);
            for (size_t i = 0; i < stride; ++i) {
                const float v = do_cfg ? (d[i] + cfg * (d[stride + i] - d[i])) : d[i];
                y[i] += dt * v;
            }
            if (cfg_.verbose) fprintf(stderr, "\rhymotion: step %d/%d", s + 1, req.steps);
        }
        if (cfg_.verbose) fprintf(stderr, "\n");
        st.solve_s = now_s() - t_solve;

        // The reference always denoises train_frames and crops: trajectory[-1][:, :length]
        std::vector<float> cropped((size_t)(frames * p_.input_dim));
        std::copy(y.begin(), y.begin() + (size_t)(frames * p_.input_dim), cropped.begin());
        out = decode(cropped.data(), frames, mean_.data(), std_.data(), (int)p_.fps, /*smooth*/ true);
        if (stats) *stats = st;
        return true;
    }

private:
    static double now_s() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
    }
    static std::vector<float> read_f32(const std::string& path, size_t expect, std::string& err) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "cannot open " + path; return {}; }
        const size_t bytes = (size_t)f.tellg();
        f.seekg(0);
        if (expect && bytes != expect * sizeof(float)) { err = path + ": unexpected size"; return {}; }
        std::vector<float> v(bytes / sizeof(float));
        f.read((char*)v.data(), (std::streamsize)bytes);
        return v;
    }
    bool init_backend(std::string& err) {
#ifdef GGML_USE_CUDA
        if (!cfg_.force_cpu) {
            backend_ = ggml_backend_cuda_init(0);
            if (!backend_ && cfg_.verbose) fprintf(stderr, "hymotion: CUDA init failed, falling back to CPU\n");
        }
#endif
        if (!backend_) backend_ = ggml_backend_cpu_init();
        if (!backend_) { err = "no ggml backend"; return false; }
        owns_backend_ = true;
        return true;
    }
    void free_backend() {
        if (backend_ && owns_backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
    }

    EngineConfig cfg_;
    Params       p_;
    ggml_backend_t backend_ = nullptr;
    bool           owns_backend_ = false;
    std::shared_ptr<ResidentWeightManager> wm_ = std::make_shared<ResidentWeightManager>();
    std::unique_ptr<ModelLoader> loader_;
    std::unique_ptr<HYMotionRunner> runner_;
    ggml_backend_buffer_t dit_buf_ = nullptr;
    ggml_context*  pipe_ctx_ = nullptr;
    std::vector<float> mean_, std_, null_vtxt_, null_ctxt_;
};

}  // namespace HYMotion
