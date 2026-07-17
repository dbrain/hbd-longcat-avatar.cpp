// ---------------------------------------------------------------------------
// hymotion -- HY-Motion 1.0 text->humanoid-motion, native ggml.
//
// Standalone by design: motion is per-character x per-prompt, the rigged asset is
// generated once, and the asset pipeline already peaks at 10751/12288 MiB on the
// 3060 -- the two must not co-reside. This is NOT wired into image_to_rig.
//
// CONDITIONING -- three ways in, all still supported:
//
//   --prompt TEXT       The real thing: Qwen3-8B hidden states + CLIP-L pooled,
//                       encoded natively (src/model/te/hymotion_text.hpp). Needs
//                       --qwen3 and --clip-l. The encoders are loaded, run ONCE,
//                       and FREED before the DiT is allocated -- the reference
//                       computes conditioning once before the ODE solve and never
//                       re-invokes Qwen3 during denoising (motion_diffusion.py:508),
//                       so nothing needs the 8B model resident next to the DiT.
//
//   --uncond            Generate from the checkpoint's OWN trained null
//                       conditioning (null_vtxt_feat / null_ctxt_input, both real
//                       tensors in the ckpt, learned via train_cfg.cond_mask_prob
//                       = 0.1). NO Qwen3, NO CLIP, NO torch, NO text encoder of
//                       any kind. Produces generic human motion, not prompt-
//                       following motion -- kept for A/B against --prompt.
//
//                       NB: this is NOT upstream's `uncondition_mode`, which uses
//                       ctxt of sequence length 1 with ctxt_length=1
//                       (motion_diffusion.py:524-527). Here the null feat is
//                       broadcast over all 128 text slots with text_valid=128.
//                       Since RoPE is applied over the concatenated stream, 128
//                       null tokens are NOT equivalent to 1. This is our own
//                       null baseline, not a reproduction of theirs.
//
//   --ctxt/--vtxt FILE  Feed precomputed conditioning (raw f32 dumps) -- golden
//                       comparison against a python dump.
//
// Output: the clip JSON that puppy-eyetest/anim already plays, with SMPL-H 22
// LOCAL joint rotations as xyzw quats + root translation.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
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

static void usage() {
    printf(
        "usage: hymotion -m MODEL.gguf -o OUT.json [options]\n"
        "\n"
        "  -m, --model PATH      hymotion gguf (from tools/convert_hymotion.py)\n"
        "  -o, --out PATH        output clip JSON\n"
        "  -p, --prompt TEXT     text prompt (needs --qwen3 and --clip-l)\n"
        "      --qwen3 PATH      Qwen3-8B gguf (llama.cpp-format is fine)\n"
        "      --clip-l PATH     CLIP-L gguf (from tools/convert_hymotion_clip.py)\n"
        "      --uncond          generate from the ckpt's trained null conditioning\n"
        "                        (no text encoder needed) -- the fastest smoke test\n"
        "      --ctxt PATH       raw f32 [128*4096] Qwen3-8B hidden states\n"
        "      --vtxt PATH       raw f32 [768] CLIP-L pooled\n"
        "      --ctxt-len N      valid text tokens (default 128)\n"
        "      --dump-cond PFX   write PFX.ctxt.f32 / PFX.vtxt.f32 / PFX.meta.txt\n"
        "      --encode-only     encode the prompt, dump, and exit (no DiT, no solve)\n"
        "      --selftest-text   check the template/crop_start/tokenizer only.\n"
        "                        No weights, no backend, no GPU. See\n"
        "                        tools/falsify_hymotion_text.py\n"
        "      --noise PATH      raw f32 [360*201] initial noise y0 (overrides --seed;\n"
        "                        use this to match a reference run exactly)\n"
        "  -s, --seed N          RNG seed (default 0). NOTE: this does NOT reproduce\n"
        "                        the reference's torch CPU RNG -- see below.\n"
        "      --steps N         Euler steps (default 50)\n"
        "      --cfg F           text guidance scale (default 5.0)\n"
        "      --seconds F       clip length in seconds, <= 12 (default 4)\n"
        "  -t, --threads N       CPU threads (default 4)\n"
        "      --cpu             force the CPU backend (default: CUDA if built)\n"
        "      --name NAME       clip name in the JSON (default \"hymotion\")\n"
        "      --no-smooth       skip slerp/savgol post-smoothing\n"
        "\n"
        "SEEDS ARE NOT REFERENCE-COMPATIBLE. The reference draws y0 with torch's CPU\n"
        "MT19937 (random_generator_on_gpu: false). Reproducing torch's exact normal\n"
        "sampler bit-for-bit is a pointless dependency; --seed here is our own RNG.\n"
        "For a golden comparison, dump y0 from python and pass --noise.\n");
}

static std::vector<float> read_f32(const std::string& path, size_t expect) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "hymotion: cannot open %s\n", path.c_str());
        exit(1);
    }
    const size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    if (expect && bytes != expect * sizeof(float)) {
        fprintf(stderr, "hymotion: %s is %zu bytes, expected %zu (%zu f32)\n",
                path.c_str(), bytes, expect * sizeof(float), expect);
        exit(1);
    }
    std::vector<float> v(bytes / sizeof(float));
    f.read((char*)v.data(), (std::streamsize)bytes);
    return v;
}

// ---------------------------------------------------------------------------
// --selftest-text: everything about the text contract that can be checked with NO
// model weights, NO backend and NO GPU. Prints machine-readable lines that
// tools/falsify_hymotion_text.py checks against the Qwen3 GGUF's own vocab -- an
// independent source our C++ never reads (it builds its vocab from qwen_merges.hpp).
// If our BPE and HF's disagree by even one id, crop_start moves and the DiT is fed
// the system prompt's hidden states instead of the user's, at a perfectly valid shape.
// ---------------------------------------------------------------------------
static int selftest_text_main(const std::string& prompt) {
    printf("SYSTEM_PROMPT_LEN %zu %zu\n",
           strlen(HYMotionText::SYSTEM_PROMPT), HYMotionText::SYSTEM_PROMPT_LEN);

    Qwen2Tokenizer qtok;
    std::string rep;
    const int64_t crop = HYMotionText::compute_crop_start(qtok, &rep);
    printf("CROP_REPORT %s\n", rep.c_str());
    printf("CROP_START %lld\n", (long long)crop);

    const std::string chat = HYMotionText::chat_string(prompt);
    printf("CHAT_BYTES %zu\n", chat.size());
    std::vector<int> ids = qtok.encode(chat);
    printf("N_TOKENS %zu\n", ids.size());
    printf("IDS");
    for (int id : ids) printf(" %d", id);
    printf("\n");

    CLIPTokenizer ctok;
    std::vector<int> cids = ctok.encode(prompt);
    std::vector<float> cw(cids.size(), 1.0f);
    ctok.pad_tokens(cids, &cw, nullptr, 77, 77, false);
    auto it = std::find(cids.begin(), cids.end(), ctok.EOS_TOKEN_ID);
    printf("CLIP_N %zu\n", cids.size());
    printf("CLIP_EOS_IDX %lld\n",
           it == cids.end() ? -1LL : (long long)std::distance(cids.begin(), it));
    printf("CLIP_IDS");
    for (int id : cids) printf(" %d", id);
    printf("\n");

    return crop < 0 ? 1 : 0;
}

// Route sd.cpp's LOG_ERROR/LOG_WARN/LOG_INFO somewhere visible. WITHOUT THIS THEY GO NOWHERE:
// every other example (delight, vae-roundtrip, server) wires this and hymotion never did, so the
// whole text-encoder path was blindfolded -- encode_ctxt's failure returned 1 with NO message and
// its success line ("N tokens, crop_start=..., ctxt_length=...") never printed either. The bug was
// not silent by nature; the diagnostics were connected to a sink that did not exist.
static void hymo_log_cb(enum sd_log_level_t level, const char* text, void* data) {
    (void)data;
    fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
    fflush(level == SD_LOG_ERROR ? stderr : stdout);
}

int main(int argc, char** argv) {
    sd_set_log_callback(hymo_log_cb, nullptr);
    std::string model_path, out_path, ctxt_path, vtxt_path, noise_path;
    std::string prompt, qwen3_path, clip_l_path, dump_cond;
    std::string clip_name = "hymotion";
    bool uncond = false, smooth = true, force_cpu = false, encode_only = false;
    bool have_prompt = false, selftest_text = false;
    int steps = 50, threads = 4;
    int64_t ctxt_len = 128;
    uint32_t seed    = 0;
    float cfg = 5.0f, seconds = 4.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next     = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "-o" || a == "--out") out_path = next();
        else if (a == "-p" || a == "--prompt") { prompt = next(); have_prompt = true; }
        else if (a == "--qwen3") qwen3_path = next();
        else if (a == "--clip-l") clip_l_path = next();
        else if (a == "--dump-cond") dump_cond = next();
        else if (a == "--encode-only") encode_only = true;
        else if (a == "--selftest-text") selftest_text = true;
        else if (a == "--uncond") uncond = true;
        else if (a == "--ctxt") ctxt_path = next();
        else if (a == "--vtxt") vtxt_path = next();
        else if (a == "--ctxt-len") ctxt_len = std::stoll(next());
        else if (a == "--noise") noise_path = next();
        else if (a == "-s" || a == "--seed") seed = (uint32_t)std::stoul(next());
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--cfg") cfg = std::stof(next());
        else if (a == "--seconds") seconds = std::stof(next());
        else if (a == "-t" || a == "--threads") threads = std::stoi(next());
        else if (a == "--name") clip_name = next();
        else if (a == "--no-smooth") smooth = false;
        else if (a == "--cpu") force_cpu = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "hymotion: unknown arg %s\n", a.c_str()); usage(); return 1; }
    }
    // Before ANY model load or backend init: this path must never touch a GPU.
    if (selftest_text) {
        return selftest_text_main(prompt);
    }

    if (model_path.empty()) { usage(); return 1; }
    if (out_path.empty() && !encode_only) { usage(); return 1; }
    if (have_prompt && (qwen3_path.empty() || clip_l_path.empty())) {
        fprintf(stderr, "hymotion: --prompt needs both --qwen3 and --clip-l\n");
        return 1;
    }
    if (have_prompt && uncond) {
        fprintf(stderr, "hymotion: --prompt and --uncond are mutually exclusive\n");
        return 1;
    }
    if (!have_prompt && !uncond && (ctxt_path.empty() || vtxt_path.empty())) {
        fprintf(stderr, "hymotion: need --prompt, --uncond, or both --ctxt and --vtxt\n");
        return 1;
    }
    if (encode_only && !have_prompt) {
        fprintf(stderr, "hymotion: --encode-only requires --prompt\n");
        return 1;
    }
    if (encode_only && dump_cond.empty()) {
        fprintf(stderr, "hymotion: --encode-only requires --dump-cond\n");
        return 1;
    }

    // ---- load ----
    // NB: init_from_file only reads metadata; the DiT weights are not resident until
    // alloc_params_buffer() below. That is what lets the 8B encoder run first and be
    // freed before the DiT ever touches VRAM.
    ModelLoader loader;
    if (!loader.init_from_file(model_path)) {
        fprintf(stderr, "hymotion: failed to load %s\n", model_path.c_str());
        return 1;
    }

    HYMotion::Params p;
    // Read the arch straight out of the GGUF so nothing is hardcoded twice.
    // (convert_hymotion.py derives these from the weights and asserts them.)
    const int64_t T_full = p.train_frames;
    int64_t frames       = (int64_t)llround(seconds * (double)p.fps);
    frames               = std::max<int64_t>(20, std::min<int64_t>(T_full, frames));

    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
    if (!force_cpu) {
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            fprintf(stderr, "hymotion: CUDA init failed, falling back to CPU\n");
        }
    }
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    (void)force_cpu;

    // ---- text encoding (BEFORE the DiT is allocated) -------------------------
    // Order is load -> encode -> free, per encoder. The 3060 has 11906 MiB and other
    // stages of this pipeline already peak near 90% of it; an 8B encoder must never
    // be resident alongside the DiT. It does not need to be: conditioning is computed
    // once, before the ODE solve (motion_diffusion.py:508).
    std::vector<float> enc_ctxt, enc_vtxt;
    int64_t enc_ctxt_length = 0;
    if (have_prompt) {
        if (strlen(HYMotionText::SYSTEM_PROMPT) != HYMotionText::SYSTEM_PROMPT_LEN) {
            fprintf(stderr, "hymotion: SYSTEM_PROMPT is %zu chars, expected %zu -- the template "
                            "literal has been altered (leading \"\\n    \" / trailing \"\\n\" matter)\n",
                    strlen(HYMotionText::SYSTEM_PROMPT), HYMotionText::SYSTEM_PROMPT_LEN);
            return 1;
        }

        // --- Qwen3-8B -> ctxt [128, 4096] ---
        {
            ModelLoader qloader;
            // Same call the production path uses for a standalone --llm gguf
            // (stable-diffusion.cpp:417): prefix, then convert_tensors_name() maps
            // llama.cpp naming (blk.0.attn_q.weight) onto the HF names llm.hpp wants.
            if (!qloader.init_from_file(qwen3_path, "text_encoders.llm.")) {
                fprintf(stderr, "hymotion: failed to load qwen3 from %s\n", qwen3_path.c_str());
                return 1;
            }
            qloader.convert_tensors_name();

            LLM::LLMRunner llm(LLM::LLMArch::QWEN3, backend, backend,
                               qloader.get_tensor_storage_map(), "text_encoders.llm", false);
            if (!llm.alloc_params_buffer()) {
                fprintf(stderr, "hymotion: qwen3 alloc failed\n");
                return 1;
            }
            std::map<std::string, ggml_tensor*> qt;
            llm.get_param_tensors(qt, "text_encoders.llm");
            if (!qloader.load_tensors(qt)) {
                fprintf(stderr, "hymotion: failed to load qwen3 tensors\n");
                return 1;
            }

            Qwen2Tokenizer qtok;  // Qwen3 reuses the Qwen2 BPE vocab/merges
            std::string rep;
            const int64_t crop_start = HYMotionText::compute_crop_start(qtok, &rep);
            fprintf(stderr, "hymotion: %s\n", rep.c_str());
            if (crop_start < 0) {
                return 1;  // the two crop_start derivations disagreed; do not guess
            }
            if (!HYMotionText::encode_ctxt(llm, qtok, prompt, crop_start,
                                           p.max_length_llm, p.ctxt_input_dim, threads,
                                           enc_ctxt, enc_ctxt_length, true)) {
                return 1;
            }
            llm.free_params_buffer();  // <-- the 8B model leaves VRAM here
        }

        // --- CLIP-L -> vtxt [768] ---
        {
            ModelLoader cloader;
            // Emitted verbatim as text_model.* by tools/convert_hymotion_clip.py, so
            // NO name conversion: init_from_file() only.
            if (!cloader.init_from_file(clip_l_path)) {
                fprintf(stderr, "hymotion: failed to load clip-l from %s\n", clip_l_path.c_str());
                return 1;
            }
            CLIPTextModelRunner clip(backend, backend, cloader.get_tensor_storage_map(),
                                     "text_model", OPENAI_CLIP_VIT_L_14,
                                     /*with_final_ln*/ true, /*force_clip_f32*/ false);
            if (!clip.alloc_params_buffer()) {
                fprintf(stderr, "hymotion: clip-l alloc failed\n");
                return 1;
            }
            std::map<std::string, ggml_tensor*> ct;
            clip.get_param_tensors(ct, "text_model");
            if (!cloader.load_tensors(ct)) {
                fprintf(stderr, "hymotion: failed to load clip-l tensors\n");
                return 1;
            }
            CLIPTokenizer ctok;
            if (!HYMotionText::encode_vtxt(clip, ctok, prompt, p.vtxt_input_dim, threads,
                                           enc_vtxt, true)) {
                return 1;
            }
            clip.free_params_buffer();
        }

        if (!dump_cond.empty()) {
            auto dump = [&](const std::string& suffix, const std::vector<float>& v) {
                std::ofstream f(dump_cond + suffix, std::ios::binary);
                f.write((const char*)v.data(), (std::streamsize)(v.size() * sizeof(float)));
            };
            dump(".ctxt.f32", enc_ctxt);
            dump(".vtxt.f32", enc_vtxt);
            std::ofstream m(dump_cond + ".meta.txt");
            m << "prompt=" << prompt << "\n"
              << "ctxt_length=" << enc_ctxt_length << "\n"
              << "ctxt_dim=" << p.ctxt_input_dim << "\n"
              << "ctxt_rows=" << p.max_length_llm << "\n"
              << "vtxt_dim=" << p.vtxt_input_dim << "\n";
            fprintf(stderr, "hymotion: dumped conditioning to %s.{ctxt,vtxt}.f32\n", dump_cond.c_str());
        }
        if (encode_only) {
            fprintf(stderr, "hymotion: --encode-only, stopping before the DiT\n");
            return 0;
        }
    }

    // GGMLRunner asserts BOTH backends are non-null (ggml_extend.hpp:4054-4055): this fork keeps a
    // separate params_backend so weights can live somewhere other than the compute backend (that is
    // what --offload-to-cpu is). Passing the same backend for both = weights resident on the compute
    // device, which is what we want here (the DiT is only ~2.1GB f16).
    HYMotion::HYMotionRunner runner(backend, backend, p, loader.get_tensor_storage_map());
    runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors, "");

    // The pipeline-level params (null/game feats, mean, std) live at the ckpt top
    // level, not inside motion_transformer, so they are not part of the DiT block
    // tree. Allocate them in their own context and let the same loader pass fill
    // them alongside the block weights.
    //
    // NB: mean/std come from the CHECKPOINT, not from stats/{Mean,Std}.npy. The
    // reference registers them as buffers from the npy and then load_state_dict
    // overwrites them with these -- so the ckpt values are the authoritative ones
    // and the LFS npy files (which are pointers in a shallow clone anyway) are not
    // needed at all.
    struct ggml_init_params ip = {
        /*mem_size  =*/ggml_tensor_overhead() * 8 + (size_t)(201 + 201 + 768 + 4096) * sizeof(float) + 4096,
        /*mem_buffer=*/nullptr,
        /*no_alloc  =*/false,
    };
    ggml_context* pipe_ctx = ggml_init(ip);
    auto t_mean            = ggml_new_tensor_1d(pipe_ctx, GGML_TYPE_F32, 201);
    auto t_std             = ggml_new_tensor_1d(pipe_ctx, GGML_TYPE_F32, 201);
    auto t_nvtxt           = ggml_new_tensor_1d(pipe_ctx, GGML_TYPE_F32, 768);
    auto t_nctxt           = ggml_new_tensor_1d(pipe_ctx, GGML_TYPE_F32, 4096);
    tensors["pipe.mean"]                  = t_mean;
    tensors["pipe.std"]                   = t_std;
    tensors["pipe.null_vtxt_feat"]        = t_nvtxt;
    tensors["pipe.null_ctxt_input"]       = t_nctxt;

    if (!loader.load_tensors(tensors)) {
        fprintf(stderr, "hymotion: failed to load tensors\n");
        return 1;
    }

    std::vector<float> mean(201), std_(201), null_vtxt(768), null_ctxt(4096);
    memcpy(mean.data(), t_mean->data, mean.size() * sizeof(float));
    memcpy(std_.data(), t_std->data, std_.size() * sizeof(float));
    memcpy(null_vtxt.data(), t_nvtxt->data, null_vtxt.size() * sizeof(float));
    memcpy(null_ctxt.data(), t_nctxt->data, null_ctxt.size() * sizeof(float));

    fprintf(stderr, "hymotion: model loaded (%lld frames @ %lld fps, %d steps, cfg %.1f)\n",
            (long long)frames, (long long)p.fps, steps, (double)cfg);

    // ---- conditioning ----
    const int64_t L_t = p.max_length_llm;
    std::vector<float> ctxt, vtxt;
    int64_t text_valid = ctxt_len;
    if (uncond) {
        // Broadcast the learned null feats: null_ctxt_input is (1,1,4096) and the
        // reference .expand()s it across all 128 text positions.
        ctxt.assign((size_t)(L_t * p.ctxt_input_dim), 0.0f);
        for (int64_t i = 0; i < L_t; ++i) {
            std::copy(null_ctxt.begin(), null_ctxt.end(), ctxt.begin() + (size_t)(i * p.ctxt_input_dim));
        }
        vtxt       = null_vtxt;
        text_valid = L_t;
        cfg        = 1.0f;  // uncondition_mode disables CFG in the reference
    } else if (have_prompt) {
        // Encoded above, before the DiT was allocated.
        ctxt       = std::move(enc_ctxt);
        vtxt       = std::move(enc_vtxt);
        text_valid = enc_ctxt_length;
    } else {
        ctxt = read_f32(ctxt_path, (size_t)(L_t * p.ctxt_input_dim));
        vtxt = read_f32(vtxt_path, (size_t)p.vtxt_input_dim);
    }

    const bool do_cfg = cfg > 1.0f && !uncond;
    const int64_t N   = do_cfg ? 2 : 1;

    runner.L_m = T_full;
    runner.L_t = L_t;
    runner.N   = N;

    // CFG batch order is [uncond, cond] -- x_pred = basic + scale*(text - basic).
    runner.ctxt_vec.assign((size_t)(N * L_t * p.ctxt_input_dim), 0.0f);
    runner.vtxt_vec.assign((size_t)(N * p.vtxt_input_dim), 0.0f);
    if (do_cfg) {
        // enable_ctxt_null_feat: true -> the uncond branch uses null_ctxt_input
        // broadcast over all 128 positions (NOT the real text).
        for (int64_t i = 0; i < L_t; ++i) {
            std::copy(null_ctxt.begin(), null_ctxt.end(),
                      runner.ctxt_vec.begin() + (size_t)(i * p.ctxt_input_dim));
        }
        std::copy(null_vtxt.begin(), null_vtxt.end(), runner.vtxt_vec.begin());
        std::copy(ctxt.begin(), ctxt.end(), runner.ctxt_vec.begin() + (size_t)(L_t * p.ctxt_input_dim));
        std::copy(vtxt.begin(), vtxt.end(), runner.vtxt_vec.begin() + (size_t)p.vtxt_input_dim);
    } else {
        runner.ctxt_vec = ctxt;
        runner.vtxt_vec = vtxt;
    }

    // ---- initial noise ----
    std::vector<float> y0((size_t)(T_full * p.input_dim));
    if (!noise_path.empty()) {
        y0 = read_f32(noise_path, y0.size());
        fprintf(stderr, "hymotion: y0 from %s\n", noise_path.c_str());
    } else {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : y0) v = nd(rng);
    }

    // ---- Euler ODE solve, t = linspace(0,1,steps+1) ----
    // torchdiffeq odeint(method="euler"): y_{n+1} = y_n + dt * f(t_n, y_n)
    std::vector<float> y = y0;
    const float dt       = 1.0f / (float)steps;
    // Diagnostic falsifiers for the CUDA-only step-17 NaN cliff (env-gated, default OFF
    // => prod path is byte-identical). The cliff happens on a SMOOTHLY varying input
    // (pred stays in ~[-5.5,4.9], dt=0.02), so "activations blow up" does not fit. These
    // separate the three candidate variables:
    //   HYMOTION_FREEZE_Y — never advance the ODE: every step re-runs the SAME input y0.
    //   HYMOTION_FREEZE_T — pin t=0: every step uses the SAME timestep embedding.
    // Both set => every compute() is bit-identical work. If the NaN STILL lands on the
    // same step index, the trigger is per-compute STATE (offload/prefetch/alloc reuse),
    // not the data and not t.
    const bool freeze_y = getenv("HYMOTION_FREEZE_Y") != nullptr;
    const bool freeze_t = getenv("HYMOTION_FREEZE_T") != nullptr;
    for (int s = 0; s < steps; ++s) {
        const float t = freeze_t ? 0.0f : (float)s / (float)steps;

        runner.x_vec.assign((size_t)(N * T_full * p.input_dim), 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            std::copy(y.begin(), y.end(), runner.x_vec.begin() + (size_t)(n * T_full * p.input_dim));
        }

        if (getenv("HYMOTION_DEBUG")) {
            // Measure the INPUT too, not just the output: a NaN in pred with a clean y
            // localises the fault INSIDE this compute; a dirty y means it arrived earlier.
            size_t ynan = 0;
            float ymn = 1e30f, ymx = -1e30f;
            for (size_t i = 0; i < y.size(); ++i) {
                if (!std::isfinite(y[i])) { ynan++; } else { ymn = std::min(ymn, y[i]); ymx = std::max(ymx, y[i]); }
            }
            fprintf(stderr, "\n[dbg] step %d t=%.3f y_in: nan=%zu/%zu min=%.4f max=%.4f\n",
                    s, (double)t, ynan, y.size(), (double)ymn, (double)ymx);
        }

        auto pred = runner.compute(threads, t, frames, text_valid);
        if (pred.empty()) {
            fprintf(stderr, "hymotion: forward failed at step %d\n", s);
            return 1;
        }
        const float* d = pred.data();
        const size_t stride = (size_t)(T_full * p.input_dim);
        if (getenv("HYMOTION_DEBUG")) {
            size_t nan = 0;
            float mn = 1e30f, mx = -1e30f;
            for (size_t i = 0; i < stride * (size_t)N; ++i) {
                if (!std::isfinite(d[i])) { nan++; } else { mn = std::min(mn, d[i]); mx = std::max(mx, d[i]); }
            }
            fprintf(stderr, "[dbg] step %d t=%.3f pred: nan=%zu/%zu min=%.4f max=%.4f\n",
                    s, (double)t, nan, stride * (size_t)N, (double)mn, (double)mx);
        }
        if (!freeze_y) {
            for (size_t i = 0; i < stride; ++i) {
                const float v = do_cfg ? (d[i] + cfg * (d[stride + i] - d[i])) : d[i];
                y[i] += dt * v;
            }
        }
        fprintf(stderr, "\rhymotion: step %d/%d", s + 1, steps);
    }
    fprintf(stderr, "\n");

    // ---- crop, decode, emit ----
    // The reference always denoises train_frames and crops: trajectory[-1][:, :length]
    std::vector<float> cropped((size_t)(frames * p.input_dim));
    std::copy(y.begin(), y.begin() + (size_t)(frames * p.input_dim), cropped.begin());

    auto motion = HYMotion::decode(cropped.data(), frames, mean.data(), std_.data(), (int)p.fps, smooth);
    auto json   = HYMotion::to_clip_json(motion, clip_name);

    std::ofstream o(out_path);
    o << json;
    fprintf(stderr, "hymotion: wrote %s (%lld frames, %lld joints)\n",
            out_path.c_str(), (long long)motion.frames, (long long)motion.joints);
    return 0;
}
