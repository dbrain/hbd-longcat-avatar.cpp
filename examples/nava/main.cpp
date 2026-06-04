// NAVA DiT harness — two modes.
//
//   nava <gguf> [<input_dir>] [<out_dir>]
//       Phase-1 single-forward harness (numeric-diff reference). Loads
//       models/nava-dit-f16.gguf, runs ONE forward of the dual-stream
//       audio-video MMDiT, writes the patched video velocity (+ auto-logged
//       per-block stats). Numeric correctness is validated separately against
//       a PyTorch bf16 reference. (The DiT is VERIFIED — see HANDOFF-nava.md
//       2026-06-03e: all 30 blocks 62-78 dB.)
//
//   nava render --prompt "..." [opts]
//       Phase-2 SILENT text-to-video render. prompt-context -> seeded noise ->
//       Euler flow-match sampling w/ cond+uncond CFG (calling the verified DiT)
//       -> Wan2.2 48ch VAE decode -> frames -> webm. Writes a run dir
//       (clip.webm + meta.json) for the eye-test viewer tools/nava_eyetest_server.py.
//       See docs/nava-phase2-sampler.md.
//       Render opts:
//         --prompt "..."        (currently informational; context comes from --context or dummy)
//         --context <file.bin>  raw umT5 [4096,512] OR post-embed [3072,512] cond context.
//                               (REQUIRED for a real render — dump from NAVA's text_model;
//                                omitted => deterministic DUMMY context, garbage-but-valid output.)
//         --neg-context <file>  uncond context (default: zeros_like(context) — NAVA's MVP path).
//         --steps N             Euler steps (default 10; use 2 for the smoke).
//         --frames N            latent temporal frames (default 13). pixel frames = (N-1)*4+1.
//         --width W --height H  PIXEL resolution (default 832x480). latent = /16; DiT grid = /32.
//         --seed S              noise seed (default 42).
//         --cfg X               video guidance scale (default 3.0).
//         --shift X             flow-match shift (default 5.0).
//         --fps N               webm fps (default 24).
//         --out-name NAME       run dir name under RUNS_DIR (default a timestamp).
//         --runs-dir DIR        base output dir (default /mnt/hdd/nava/cpp-runs).
//         --vae <gguf>          Wan2.2 48ch VAE gguf (default models/longcat-wan-vae-f16.gguf).
//         --cuda                use the CUDA backend (default CPU; CPU is for the smoke only).
//         --label "..."         meta.json label.
//
// CPU concessions (NOT model truth; a CUDA service build runs natively):
//   * LONGCAT_NO_FUSED_ROPE=1 — the fused interleaved-RoPE op (ggml_rope_pe) is
//     CUDA-only; force the portable chain-RoPE path.
//   * modulation/head-mod params allocated F32 (CPU rejects f32(+)f16; the gguf
//     has them F16). Handled inside nava.hpp.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "ggml.h"
#include "model.h"
#include "stable-diffusion.h"
#include "tensor.hpp"
#include "tensor_ggml.hpp"
#include "util.h"

#include "common/media_io.h"

#include "nava.hpp"
#include "rng_philox.hpp"
#include "unipc.hpp"
#include "vae.hpp"
#include "wan.hpp"
#include "ltx_audio_vae.h"
#include "t5.hpp"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

static void log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
}

static bool path_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// Match python's HuggingfaceTokenizer text preprocessing (nava_src .../tokenizers.py):
// the umT5 sentencepiece normalizer applies NFKC, mapping full-width forms to ASCII
// (U+FF01..U+FF5E -> cp-0xFEE0, U+3000 -> space). The cpp t5 unigram tokenizer does NOT
// apply that charsmap, so full-width CJK punctuation ('，' U+FF0C, '：' U+FF1A, ...)
// mis-tokenizes (id 356/619 instead of 275/283), corrupting the whole umT5 context via
// bidirectional attention. Also apply whitespace_clean (collapse \s+ -> ' ', strip).
static std::string nava_normalize_text(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    auto emit = [&](uint32_t cp) {
        if (cp < 0x80) out.push_back((char)cp);
        else if (cp < 0x800) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) { out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
        else { out.push_back((char)(0xF0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
    };
    size_t i = 0, n = in.size();
    while (i < n) {
        unsigned char c = (unsigned char)in[i];
        uint32_t cp; int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else { cp = c; len = 1; }
        for (int k = 1; k < len && i + k < n; ++k) cp = (cp << 6) | ((unsigned char)in[i + k] & 0x3F);
        i += len;
        if (cp >= 0xFF01 && cp <= 0xFF5E) cp -= 0xFEE0;  // full-width forms -> ASCII (NFKC)
        else if (cp == 0x3000) cp = 0x20;                // ideographic space -> space
        emit(cp);
    }
    // whitespace_clean: collapse whitespace runs to a single space, then strip.
    std::string ws;
    ws.reserve(out.size());
    bool prev_sp = false;
    for (char ch : out) {
        bool sp = (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v');
        if (sp) { if (!prev_sp) ws.push_back(' '); prev_sp = true; }
        else { ws.push_back(ch); prev_sp = false; }
    }
    size_t a = ws.find_first_not_of(' '), b = ws.find_last_not_of(' ');
    return (a == std::string::npos) ? std::string() : ws.substr(a, b - a + 1);
}

static void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); ++i) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) {
            if (!cur.empty() && cur != "/") {
                mkdir(cur.c_str(), 0777);
            }
        }
    }
}

static void dump_stats(const char* tag, const sd::Tensor<float>& t) {
    if (t.empty()) {
        printf("%-26s EMPTY\n", tag);
        return;
    }
    double sum = 0.0, sq = 0.0, mn = 1e30, mx = -1e30;
    for (int64_t i = 0; i < t.numel(); ++i) {
        double v = t.data()[i];
        sum += v;
        sq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double mean = sum / (double)t.numel();
    double var  = sq / (double)t.numel() - mean * mean;
    printf("%-26s shape=%s mean=%.5f std=%.5f min=%.4f max=%.4f\n",
           tag, sd::tensor_shape_to_string(t.shape()).c_str(), mean,
           var > 0 ? sqrt(var) : 0.0, mn, mx);
}

static void write_bin(const std::string& path, const sd::Tensor<float>& t, const std::string& name) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("cannot open %s\n", path.c_str()); return; }
    int32_t n_dims = (int32_t)t.dim();
    int32_t len    = (int32_t)name.size();
    int32_t ttype  = (int32_t)GGML_TYPE_F32;
    fwrite(&n_dims, sizeof(int32_t), 1, f);
    fwrite(&len, sizeof(int32_t), 1, f);
    fwrite(&ttype, sizeof(int32_t), 1, f);
    for (int i = 0; i < n_dims; ++i) {
        int32_t d = (int32_t)t.shape()[(size_t)i];
        fwrite(&d, sizeof(int32_t), 1, f);
    }
    fwrite(name.data(), 1, (size_t)len, f);
    fwrite(t.data(), sizeof(float), (size_t)t.numel(), f);
    fclose(f);
    printf("wrote %s\n", path.c_str());
}

// deterministic dummy fill (small magnitudes, sin-based so it's reproducible)
static sd::Tensor<float> dummy(std::vector<int64_t> shape, float scale = 0.02f) {
    sd::Tensor<float> t(shape);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data()[i] = scale * std::sin(0.1f * (float)(i % 997));
    }
    return t;
}

static sd::Tensor<float> load_or_dummy(const std::string& dir,
                                       const std::string& fname,
                                       std::vector<int64_t> dummy_shape) {
    if (!dir.empty()) {
        std::string p = dir + "/" + fname;
        if (path_exists(p)) {
            auto t = sd::load_tensor_from_file_as_tensor<float>(p);
            printf("loaded %s\n", p.c_str());
            return t;
        }
    }
    return dummy(std::move(dummy_shape));
}

// ---------------------------------------------------------------------------
// Phase-1 single-forward harness (unchanged behaviour).
// ---------------------------------------------------------------------------
static int run_single_forward(int argc, char** argv) {
    std::string gguf_path = argv[1];
    std::string in_dir, out_dir = ".";
    int positional = 0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (positional == 0) {
            in_dir = a;
            positional++;
        } else if (positional == 1) {
            out_dir = a;
            positional++;
        }
    }

    setenv("LONGCAT_NO_FUSED_ROPE", "1", /*overwrite=*/0);

    const int W = 16, H = 16, F = 2;  // -> grid h'=8, w'=8, f'=2 -> L_vid=128
    const int Lraw = 32;              // -> L_audio = 32

    auto video    = load_or_dummy(in_dir, "video.bin",    {W, H, F, 48});
    auto audio    = load_or_dummy(in_dir, "audio.bin",    {128, Lraw});
    auto context  = load_or_dummy(in_dir, "context.bin",  {4096, 512});
    auto timestep = load_or_dummy(in_dir, "timestep.bin", {1});
    if (in_dir.empty() || !path_exists(in_dir + "/timestep.bin")) {
        timestep.data()[0] = 500.0f;
    }

    // NAVA_I2V=1: replace the scalar timestep with the PER-TOKEN clean-anchor t
    // (first frame's h'*w' VIDEO tokens = 0, all other video + all audio = t) so the
    // per-token-timestep path can be validated against the PyTorch first_frame_is_clean
    // reference (~/dev/NAVA/nava_dump_i2v_ref.py).
    if (getenv("NAVA_I2V")) {
        // derive the grid from the LOADED video tensor [W,H,F,48], not the smoke
        // fallback consts, so the per-token t is correct at any resolution.
        const int Wv = (int)video.shape()[0], Hv = (int)video.shape()[1], Fv = (int)video.shape()[2];
        const int h_grid = Hv / 2, w_grid = Wv / 2;
        const int64_t L_vid = (int64_t)Fv * h_grid * w_grid;
        const int64_t L_aud = audio.shape()[1];
        const int64_t L_total = L_vid + L_aud;
        const int64_t n_clean = (int64_t)h_grid * w_grid;         // first-frame spatial tokens
        const float tval = timestep.data()[0];
        sd::Tensor<float> t_tok({L_total});
        for (int64_t i = 0; i < L_total; ++i) t_tok.data()[i] = (i < n_clean) ? 0.0f : tval;
        timestep = std::move(t_tok);
        printf("NAVA_I2V: per-token t [L_total=%lld], first %lld tokens = 0, rest = %.1f\n",
               (long long)L_total, (long long)n_clean, tval);
    }

    dump_stats("input video",    video);
    dump_stats("input audio",    audio);
    dump_stats("input context",  context);
    dump_stats("input timestep", timestep);

    ggml_backend_t backend = ggml_backend_cpu_init();
    printf("backend: CPU\n");

    ModelLoader loader;
    if (!loader.init_from_file(gguf_path, "")) {
        printf("failed to init loader from %s\n", gguf_path.c_str());
        return 1;
    }
    const bool head_f32 = getenv("NAVA_HEAD_F32") != nullptr;
    auto& tsm = loader.get_tensor_storage_map();
    for (auto& [name, ts] : tsm) {
        if (ends_with(name, "weight") && ts.type == GGML_TYPE_F16) {
            ts.expected_type = GGML_TYPE_F16;
        }
        // Experiment: upcast the two head Linear weights to F32 to isolate whether
        // the ~44 dB head error is F16-vs-bf16 precision or a structural op bug.
        if (head_f32 && (name == "backbone.head.head.weight"
                         || name == "backbone.head_audio.head.weight")) {
            ts.expected_type = GGML_TYPE_F32;
            printf("NAVA_HEAD_F32: upcasting %s to F32\n", name.c_str());
        }
    }

    auto runner = std::make_shared<NAVA::NavaRunner>(backend, backend, tsm, "backbone");
    runner->alloc_params_buffer();

    std::map<std::string, ggml_tensor*> tensors;
    runner->get_param_tensors(tensors, "backbone");
    printf("expecting %zu param tensors\n", tensors.size());
    if (!loader.load_tensors(tensors)) {
        printf("failed to load NAVA tensors\n");
        return 1;
    }
    printf("NAVA backbone loaded\n");

    int n_threads = 8;
    // NAVA_MASK_MODALITY=1: run the separate intra-modal attention path (align_3d_cfg
    // masking_modality forward) so it can be diffed vs the PyTorch mmask reference.
    if (getenv("NAVA_MASK_MODALITY")) {
        runner->mask_modality = true;
        printf("NAVA_MASK_MODALITY: separate intra-modal self-attention\n");
    }
    int64_t t0 = ggml_time_ms();
    auto vel_video = runner->compute(n_threads, video, audio, context, timestep);
    int64_t t1 = ggml_time_ms();
    printf("forward done in %lld ms\n", (long long)(t1 - t0));

    // VALIDATE compute_va() == compute(): the render uses compute_va (joint
    // video+audio via pad/concat/split); the per-block validation used compute().
    // The video velocity MUST be bit-identical between them — any diff is a
    // render-only bug (slicing/stride in the joint path).
    {
        // pass the same mask_modality the compute() above used, else this compares
        // masked-compute() vs unmasked-compute_va() (compute_va resets the flag).
        auto vv = runner->compute_va(n_threads, video, audio, context, timestep, runner->mask_modality).first;
        dump_stats("compute()    vel_video", vel_video);
        dump_stats("compute_va() vv        ", vv);
        if (!vv.empty() && vv.numel() == vel_video.numel()) {
            double mx = 0, sa = 0;
            for (int64_t i = 0; i < vv.numel(); ++i) {
                double d = std::fabs((double)vv.data()[i] - (double)vel_video.data()[i]);
                if (d > mx) mx = d;
                sa += d;
            }
            printf(">>> compute_va vs compute: maxAbsDiff=%.6g  meanAbsDiff=%.6g  (MUST be ~0)\n",
                   mx, sa / (double)vv.numel());
        } else {
            printf(">>> compute_va vs compute: SHAPE MISMATCH vv.numel=%lld vel.numel=%lld\n",
                   (long long)vv.numel(), (long long)vel_video.numel());
        }
    }

    dump_stats("velocity_video_patched", vel_video);
    if (!vel_video.empty()) {
        write_bin(out_dir + "/nava_velocity_video_patched.bin", vel_video, "velocity_video_patched");
    } else {
        printf("WARNING: empty output\n");
        return 1;
    }
    printf("(audio velocity + per-block hidden states are auto-logged above as debug-tensor stats)\n");

    // --- VALIDATE compute_va() vs compute() -------------------------------
    // The render path uses compute_va() (joint pad+concat+host-split). It must
    // return a video velocity bit-identical to compute()'s. Compare here.
    if (getenv("NAVA_VALIDATE_VA")) {
        printf("\n=== NAVA_VALIDATE_VA: compute() vs compute_va() ===\n");
        auto [vv, va] = runner->compute_va(n_threads, video, audio, context, timestep);
        dump_stats("compute_va vv (video vel)", vv);
        dump_stats("compute_va va (audio vel)", va);
        if (vv.empty()) {
            printf("WARNING: compute_va returned empty vv\n");
            return 1;
        }
        if (vv.shape().size() != 2 || vv.shape()[0] != vel_video.shape()[0]
            || vv.shape()[1] != vel_video.shape()[1]) {
            printf("SHAPE MISMATCH: compute() vv ne[%lld,%lld] vs compute_va() vv ne[%lld,%lld]\n",
                   (long long)vel_video.shape()[0], (long long)vel_video.shape()[1],
                   (long long)vv.shape()[0], (long long)vv.shape()[1]);
            return 1;
        }
        const int64_t n = (int64_t)vel_video.shape()[0] * vel_video.shape()[1];
        double maxabs = 0.0, sumsq_diff = 0.0, sumsq_ref = 0.0;
        int64_t argmax = -1;
        const float* a = vel_video.data();
        const float* b = vv.data();
        for (int64_t i = 0; i < n; ++i) {
            double d = std::fabs((double)a[i] - (double)b[i]);
            if (d > maxabs) { maxabs = d; argmax = i; }
            sumsq_diff += (double)(a[i] - b[i]) * (a[i] - b[i]);
            sumsq_ref  += (double)a[i] * a[i];
        }
        double mse  = sumsq_diff / (double)n;
        double peak = 0.0;
        for (int64_t i = 0; i < n; ++i) peak = std::max(peak, (double)std::fabs(a[i]));
        double psnr = (mse > 0.0) ? 20.0 * std::log10(peak) - 10.0 * std::log10(mse) : 999.0;
        printf("compute_va vv vs compute() vv: N=%lld maxabsdiff=%.6g (at idx %lld: %.6g vs %.6g) "
               "rel_l2=%.6g PSNR=%.2f dB\n",
               (long long)n, maxabs, (long long)argmax,
               argmax >= 0 ? a[argmax] : 0.0f, argmax >= 0 ? b[argmax] : 0.0f,
               std::sqrt(sumsq_diff / (sumsq_ref + 1e-30)), psnr);
        if (maxabs < 1e-4) {
            printf("RESULT: compute_va is CLEAN (bit-identical within tol).\n");
        } else {
            printf("RESULT: compute_va DIFFERS — BUG in joint pad/concat/split path.\n");
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Phase-2 render mode.
// ---------------------------------------------------------------------------

struct RenderOpts {
    std::string gguf      = "models/nava-dit-f16.gguf";
    std::string vae       = "models/longcat-wan-vae-f16.gguf";
    std::string audio_vae = "models/nava-ltx-audio-vae-f16.gguf";  // LTX audio VAE; "" => silent webm
    std::string context;       // cond context .bin (raw umT5 [4096,512] or [3072,512]); empty => dummy
    std::string neg_context;   // VIDEO uncond context; empty => zeros_like(context)
    std::string audio_neg_context;  // AUDIO uncond context; empty => falls back to neg_context
    std::string prompt    = "(informational only — context comes from --context)";
    std::string label;
    std::string out_name;
    std::string runs_dir  = "/mnt/hdd/nava/cpp-runs";
    int steps             = 10;
    int frames            = 13;   // latent temporal frames
    int width             = 832;  // pixel
    int height            = 480;  // pixel
    int fps               = 24;
    uint64_t seed         = 42;
    float cfg             = 3.0f;
    float cfg_align       = 3.0f;   // video_align_guidance_scale (align_3d_cfg)
    float cfg_align_audio = 2.0f;   // audio_align_guidance_scale (align_3d_cfg)
    float shift           = 5.0f;
    bool cuda             = false;
    std::string image;         // I2V: input image; frame-0 clean-latent anchor (empty => T2V)
};

// FlowMatchScheduler (Euler), distilled from NAVA scheduler/flow_match.py.
// sigmas = linspace(sigma_max, sigma_min, N); shift transform; timesteps = sigma*1000.
struct FlowMatchSched {
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    void set_timesteps(int n, float shift) {
        const float sigma_max = 1.0f;
        const float sigma_min = 0.003f / 1.002f;
        sigmas.resize(n);
        for (int i = 0; i < n; ++i) {
            // NAVA's FlowMatchScheduler uses extra_one_step=True:
            // sigmas = linspace(sigma_max, sigma_min, n+1)[:-1]  -> divide by n (NOT n-1),
            // so the steps stay at higher sigma and the FINAL Euler step is one big jump
            // sigma_last -> 0. (Dividing by n-1 reaches sigma_min and crawls the tail,
            // a different denoising trajectory that destabilizes strong CFG guidance.)
            float s = (n == 1) ? sigma_max
                               : sigma_max + (sigma_min - sigma_max) * ((float)i / (float)n);
            // shift transform: s <- shift*s / (1 + (shift-1)*s)
            sigmas[i] = shift * s / (1.0f + (shift - 1.0f) * s);
        }
        timesteps.resize(n);
        for (int i = 0; i < n; ++i) timesteps[i] = sigmas[i] * 1000.0f;
    }
    // sigma_next for step i (0 at the final step).
    float sigma_next(int i) const {
        return (i + 1 < (int)sigmas.size()) ? sigmas[i + 1] : 0.0f;
    }
};

// Unpatchify the DiT head output [192, L_vid] (ggml ne; 192 = (p=1,q=2,r=2,c=48),
// c fastest; L_vid token order f-major then h then w) into a VAE-latent video
// tensor with ggml ne [W_lat, H_lat, F, 48] (W fastest) — exactly what WanVAE
// decode + diffusion_to_vae_latents expect. Mirrors NAVA model_mm.py:
//   u.view(F,H',W',1,2,2,48); einsum('fhwpqrc->cfphqwr') -> [48,F,H'*2,W'*2].
// (here H'=h_grid, W'=w_grid are the DiT conv grid = VAE-latent /2; output H_lat=H'*2.)
static sd::Tensor<float> unpatchify_video(const sd::Tensor<float>& patched,
                                          int f_len, int h_grid, int w_grid) {
    const int pq = 2, pr = 2, c = 48;
    const int64_t L_vid = (int64_t)f_len * h_grid * w_grid;
    // patched ggml ne [192, L_vid]; data ne0 (the 192) fastest.
    // within a token the 192 are ordered (p,q,r,c) with c fastest (matches the
    // .view(...,1,2,2,48) layout: last dim c is fastest).
    const float* src = patched.data();
    const int64_t H_lat = (int64_t)h_grid * pq;
    const int64_t W_lat = (int64_t)w_grid * pr;
    // output ggml ne [W_lat, H_lat, F, 48]; data idx = ((cc*F + ff)*H_lat + yy)*W_lat + xx
    sd::Tensor<float> out({W_lat, H_lat, (int64_t)f_len, (int64_t)c});
    float* dst = out.data();
    for (int ff = 0; ff < f_len; ++ff) {
        for (int hh = 0; hh < h_grid; ++hh) {
            for (int ww = 0; ww < w_grid; ++ww) {
                int64_t tok = ((int64_t)ff * h_grid + hh) * w_grid + ww;
                const float* tbase = src + tok * (int64_t)(pq * pr * c);
                for (int q = 0; q < pq; ++q) {        // sub-row within patch (H)
                    for (int r = 0; r < pr; ++r) {    // sub-col within patch (W)
                        for (int cc = 0; cc < c; ++cc) {
                            // src layout (p=1,q,r,c): index = ((q*pr + r)*c) + cc
                            float v = tbase[((int64_t)q * pr + r) * c + cc];
                            int64_t yy = (int64_t)hh * pq + q;
                            int64_t xx = (int64_t)ww * pr + r;
                            int64_t didx = (((int64_t)cc * f_len + ff) * H_lat + yy) * W_lat + xx;
                            dst[didx] = v;
                        }
                    }
                }
            }
        }
    }
    (void)L_vid;
    return out;
}

static std::string now_stamp() {
    time_t t = time(nullptr);
    struct tm tmv;
    localtime_r(&t, &tmv);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tmv);
    return buf;
}

static int run_render(int argc, char** argv) {
    RenderOpts o;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def) -> std::string {
            if (i + 1 < argc) return argv[++i];
            return def;
        };
        if (a == "--prompt") o.prompt = next("");
        else if (a == "--context") o.context = next("");
        else if (a == "--neg-context") o.neg_context = next("");
        else if (a == "--audio-neg-context") o.audio_neg_context = next("");
        else if (a == "--steps") o.steps = atoi(next("10").c_str());
        else if (a == "--frames") o.frames = atoi(next("13").c_str());
        else if (a == "--width") o.width = atoi(next("832").c_str());
        else if (a == "--height") o.height = atoi(next("480").c_str());
        else if (a == "--fps") o.fps = atoi(next("24").c_str());
        else if (a == "--seed") o.seed = (uint64_t)strtoull(next("42").c_str(), nullptr, 10);
        else if (a == "--image") o.image = next("");
        else if (a == "--cfg") o.cfg = atof(next("3.0").c_str());
        else if (a == "--cfg-align") o.cfg_align = atof(next("3.0").c_str());
        else if (a == "--cfg-align-audio") o.cfg_align_audio = atof(next("2.0").c_str());
        else if (a == "--shift") o.shift = atof(next("5.0").c_str());
        else if (a == "--out-name") o.out_name = next("");
        else if (a == "--runs-dir") o.runs_dir = next("");
        else if (a == "--vae") o.vae = next("");
        else if (a == "--audio-vae") o.audio_vae = next("");
        else if (a == "--no-audio") o.audio_vae = "";
        else if (a == "--gguf") o.gguf = next("");
        else if (a == "--label") o.label = next("");
        else if (a == "--cuda") o.cuda = true;
        else { printf("unknown render arg: %s\n", a.c_str()); return 1; }
    }
    if (o.out_name.empty()) o.out_name = "nava-" + now_stamp();
    if (o.label.empty()) o.label = o.out_name;

    setenv("LONGCAT_NO_FUSED_ROPE", "1", /*overwrite=*/0);

    // Latent grid. VAE spatial downsample = 16; DiT patch [1,2,2] downsamples by
    // another 2 -> conv grid = latent/2 = pixel/32. (See NAVA t2v.py:277-278.)
    const int W_lat = o.width / 16;
    const int H_lat = o.height / 16;
    const int w_grid = W_lat / 2;
    const int h_grid = H_lat / 2;
    const int f_len  = o.frames;
    // Audio stream length. NAVA is a JOINT audio-video MMDiT — every block does
    // joint self-attention over [video ++ audio]. A degenerate audio stream
    // (1 token, near-zero) is OUT of the trained distribution and can poison the
    // joint attention. Size it to the video duration (audio_tokens_per_sec=25,
    // video_fps=24 per nava_run.yaml), min 8, so the joint forward sees a
    // plausible audio stream even though we render silent.
    // PyTorch uses ceil (t2v.py:417: math.ceil(video_duration * audio_tokens_per_sec)),
    // NOT round — match it so the joint audio sequence length is identical.
    const int audio_len = std::max(8, (int)std::ceil(((f_len - 1) * 4 + 1) / 24.0 * 25.0));

    printf("=== NAVA render (SILENT) ===\n");
    printf("prompt: %s\n", o.prompt.c_str());
    printf("steps=%d frames=%d (pixel=%d) res=%dx%d latent=[48,%d,%d,%d] grid=[%d,%d,%d] seed=%llu cfg=%.2f shift=%.2f\n",
           o.steps, f_len, (f_len - 1) * 4 + 1, o.width, o.height, f_len, H_lat, W_lat,
           f_len, h_grid, w_grid, (unsigned long long)o.seed, o.cfg, o.shift);

    // backend
    ggml_backend_t backend = nullptr;
    if (o.cuda) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
        printf("backend: CUDA\n");
#else
        printf("--cuda requested but binary built without CUDA; falling back to CPU\n");
#endif
    }
    if (backend == nullptr) {
        backend = ggml_backend_cpu_init();
        printf("backend: CPU\n");
    }

    int n_threads = 8;
    int64_t wall_t0 = ggml_time_ms();

    // ----- I2V: encode the input image -> frame-0 clean-latent anchor -----
    // --image is a PREPROCESSED RGB bin (ggml ne [W,H,1,3], values [0,1] — produced by
    // tools/nava_prep_image.py so we avoid linking stb here). VAE-encode it (decode_only
    // =false) BEFORE the 12.6GB DiT loads (VRAM: VAE alone fits, then freed). The anchor
    // is spliced at frame 0 and pinned each step via the per-token clean-anchor timestep.
    sd::Tensor<float> anchor_latent;          // [W_lat, H_lat, 1, 48], diffusion convention
    const bool i2v = !o.image.empty();
    if (i2v) {
        if (!path_exists(o.image)) { printf("ERROR: --image %s not found\n", o.image.c_str()); return 1; }
        auto img_rgb = sd::load_tensor_from_file_as_tensor<float>(o.image);
        printf("I2V: loaded image %s shape=%s\n", o.image.c_str(),
               sd::tensor_shape_to_string(img_rgb.shape()).c_str());
        auto encvae = std::make_shared<WAN::WanVAERunner>(
            backend, backend, String2TensorStorage{}, "", /*decode_only=*/false, VERSION_WAN2_2_TI2V);
        {
            ModelLoader vl;
            if (!vl.init_from_file_and_convert_name(o.vae, "vae.")) { printf("I2V VAE loader fail\n"); return 1; }
            encvae->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> vt;
            encvae->get_param_tensors(vt, "first_stage_model");
            if (!vl.load_tensors(vt)) { printf("I2V VAE tensors fail\n"); return 1; }
        }
        sd_tiling_params_t et = {};
        et.enabled = (W_lat > 24 || H_lat > 24);
        et.temporal_tiling = false; et.tile_size_x = 24; et.tile_size_y = 24; et.target_overlap = 0.25f;
        auto mu = encvae->encode(n_threads, img_rgb, et, /*encode_video=*/true, false);
        if (mu.empty()) { printf("ERROR: I2V VAE encode failed\n"); return 1; }
        printf("I2V: encoded mu shape=%s\n", sd::tensor_shape_to_string(mu.shape()).c_str());
        sd::Tensor<float> mu5 = mu;
        mu5.reshape_({(int64_t)W_lat, (int64_t)H_lat, 1, 48, 1});
        anchor_latent = encvae->vae_to_diffusion_latents(mu5);
        anchor_latent.reshape_({(int64_t)W_lat, (int64_t)H_lat, 1, 48});
        encvae.reset();  // free VAE before the DiT loads
        dump_stats("I2V anchor latent (diffusion)", anchor_latent);
    }

    // ----- load DiT -----
    ModelLoader loader;
    if (!loader.init_from_file(o.gguf, "")) {
        printf("failed to init loader from %s\n", o.gguf.c_str());
        return 1;
    }
    auto& tsm = loader.get_tensor_storage_map();
    for (auto& [name, ts] : tsm) {
        if (ends_with(name, "weight") && ts.type == GGML_TYPE_F16) {
            ts.expected_type = GGML_TYPE_F16;
        }
    }
    auto runner = std::make_shared<NAVA::NavaRunner>(backend, backend, tsm, "backbone");
    runner->alloc_params_buffer();
    {
        std::map<std::string, ggml_tensor*> tensors;
        runner->get_param_tensors(tensors, "backbone");
        if (!loader.load_tensors(tensors)) {
            printf("failed to load NAVA tensors\n");
            return 1;
        }
    }
    int64_t load_t1 = ggml_time_ms();
    printf("NAVA backbone loaded (%.2fs)\n", (load_t1 - wall_t0) / 1000.0f);

    // ----- contexts -----
    // cond context: raw umT5 [4096,512] (DiT text_embedding runs) or post-embed
    // [3072,512]. Dummy if not provided (smoke). uncond = zeros_like (NAVA MVP).
    // BUGFIX (the Phase-2 incoherence): load the cond context DIRECTLY by its
    // (typically absolute) path. The old code used load_or_dummy(".", o.context)
    // which builds "./" + "/abs/path" = ".//abs/path" — a RELATIVE path that never
    // exists, so it SILENTLY fell back to the sin-based dummy context. The render
    // then denoised with a dummy prompt embedding (context_embedded std 0.036 vs
    // the real 0.21) → every clip was unconditioned garbage AND the dummy-cond vs
    // real-neg CFG guidance was biased, driving the std runaway. (ctx_neg below was
    // already loaded correctly via load_tensor_from_file_as_tensor.)
    sd::Tensor<float> ctx_pos;
    if (!o.context.empty()) {
        if (!path_exists(o.context)) {
            printf("ERROR: --context %s not found\n", o.context.c_str());
            return 1;
        }
        ctx_pos = sd::load_tensor_from_file_as_tensor<float>(o.context);
        printf("loaded cond context %s (ne0=%lld)\n", o.context.c_str(),
               (long long)ctx_pos.shape()[0]);
    } else {
        printf("NOTE: no --context => DUMMY cond context (garbage-but-valid render).\n");
        ctx_pos = dummy({4096, 512});
    }
    sd::Tensor<float> ctx_neg;
    if (!o.neg_context.empty()) {
        ctx_neg = sd::load_tensor_from_file_as_tensor<float>(o.neg_context);
    } else {
        ctx_neg = sd::Tensor<float>(ctx_pos.shape());  // zeros_like
        for (int64_t i = 0; i < ctx_neg.numel(); ++i) ctx_neg.data()[i] = 0.0f;
    }
    // AUDIO uncond context. PyTorch (pipeline_nava.py negative_prompt_mode=True) uses
    // a SEPARATE audio negative prompt ("机械音…") for the audio stream's CFG uncond,
    // NOT the video neg-prompt. Using the wrong audio uncond corrupts the co-denoised
    // audio velocity, which feeds back through the joint self-attention every step and
    // makes the video diverge. Fall back to the video neg only if not supplied.
    sd::Tensor<float> ctx_audio_neg;
    const bool have_audio_neg = !o.audio_neg_context.empty();
    if (have_audio_neg) {
        ctx_audio_neg = sd::load_tensor_from_file_as_tensor<float>(o.audio_neg_context);
        printf("audio uncond uses SEPARATE audio-neg context: %s\n", o.audio_neg_context.c_str());
    } else {
        printf("NOTE: no --audio-neg-context => audio uncond reuses the VIDEO neg context.\n");
    }

    // ----- seeded noise in VAE-latent space [W_lat, H_lat, F, 48] -----
    PhiloxRNG rng(o.seed);
    sd::Tensor<float> latent({(int64_t)W_lat, (int64_t)H_lat, (int64_t)f_len, 48});
    {
        auto r = rng.randn((uint32_t)latent.numel());
        for (int64_t i = 0; i < latent.numel(); ++i) latent.data()[i] = r[(size_t)i];
    }
    // Audio latent: proper std-1 Gaussian noise (NOT a near-zero dummy), so the
    // joint self-attention sees an in-distribution audio stream. We still render
    // silent (the audio velocity is discarded here; joint denoise is the next step
    // if this proves the joint forward was being poisoned by the degenerate dummy).
    sd::Tensor<float> audio_latent({128, (int64_t)audio_len});
    {
        auto r = rng.randn((uint32_t)audio_latent.numel());
        for (int64_t i = 0; i < audio_latent.numel(); ++i) audio_latent.data()[i] = r[(size_t)i];
    }

    // I2V: token bookkeeping + splice the clean anchor into frame 0. The first
    // h_grid*w_grid VIDEO tokens (frame 0) carry timestep 0 every step (clean anchor)
    // and frame 0's latent is re-pinned to the image after every Euler step.
    const int h_grid_i      = H_lat / 2, w_grid_i = W_lat / 2;
    const int64_t L_vid_i   = (int64_t)f_len * h_grid_i * w_grid_i;
    const int64_t n_clean_i = (int64_t)h_grid_i * w_grid_i;   // frame-0 spatial tokens
    auto splice_anchor = [&]() {
        if (!i2v) return;
        const int64_t W = W_lat, H = H_lat, F = f_len;  // latent ne [W,H,F,48]
        for (int64_t c = 0; c < 48; ++c)
            for (int64_t y = 0; y < H; ++y)
                for (int64_t x = 0; x < W; ++x)
                    latent.data()[x + W * (y + H * (0 + F * c))] =
                        anchor_latent.data()[x + W * (y + H * c)];
    };
    splice_anchor();

    // ----- Euler flow-match sampling (JOINT video+audio denoise) -----
    // NAVA is a joint AV MMDiT: every block self-attends over [video ++ audio].
    // We co-denoise the audio stream in lockstep so the video tokens always attend
    // to an audio stream at the matching noise level (a frozen / degenerate audio
    // stream poisons the joint attention and the video never coheres).
    const float cfg_audio = 2.0f;  // nava_run.yaml audio_guidance_scale
    // align_3d_cfg (nava_640.yaml): a 3rd "masking_modality" forward (separate
    // intra-modal attention) adds an alignment guidance term to BOTH streams. This
    // is what locks audio content to the video on hard prompts — its omission was
    // the audio->noise divergence. Default ON to match python; NAVA_NO_ALIGN_CFG=1
    // reverts to the old 2-way CFG for A/B.
    const float cfg_video_align = o.cfg_align;        // video_align_guidance_scale
    const float cfg_audio_align = o.cfg_align_audio;  // audio_align_guidance_scale
    const bool  align_cfg       = getenv("NAVA_NO_ALIGN_CFG") == nullptr;
    FlowMatchSched sched;
    sched.set_timesteps(o.steps, o.shift);
    // UniPC multistep solver (validated vs PyTorch FlowUniPCMultistepScheduler) is the
    // DEFAULT: it matches python prod (nava_run.yaml scheduler_unipc=true). Its higher-order
    // corrector tames the big low-sigma Euler steps that otherwise leave residual noise/"fuzz"
    // at low step counts (and can run the std away). Opt out with NAVA_EULER=1 (plain Euler).
    const bool use_unipc = getenv("NAVA_EULER") == nullptr;
    // Per-stream sampler: the AUDIO stream can use a DIFFERENT solver than video.
    // UniPC's higher-order corrector cleans up the VIDEO fuzz, but on the audio stream
    // it can run aud_std away (-> the voice diverges to noise on hard prompts). Set
    // NAVA_AUDIO_EULER=1 to keep video on UniPC but denoise audio with plain Euler.
    const bool use_unipc_audio = use_unipc && getenv("NAVA_AUDIO_EULER") == nullptr;
    UniPCSched usched_v, usched_a;
    if (use_unipc) {
        usched_v.set_timesteps(o.steps, o.shift);
        usched_a.set_timesteps(o.steps, o.shift);
    }
    printf("sampler: video=%s  audio=%s\n",
           use_unipc ? "UniPC" : "Euler",
           use_unipc_audio ? "UniPC" : "Euler");

    // Optional per-step trajectory dump for cpp-vs-PyTorch sampler faithfulness probe.
    const char* dump_traj = getenv("NAVA_DUMP_TRAJ");
    if (dump_traj) {
        std::string d = dump_traj;
        write_bin(d + "/vid_noise.bin", latent, "vid_noise");
        write_bin(d + "/aud_noise.bin", audio_latent, "aud_noise");
        printf("NAVA_DUMP_TRAJ active -> %s\n", d.c_str());
    }

    int64_t sample_t0 = ggml_time_ms();
    for (int step = 0; step < o.steps; ++step) {
        const float tval = use_unipc ? usched_v.timesteps[step] : sched.timesteps[step];
        // I2V: per-token clean-anchor timestep (frame-0 video tokens = 0, rest = t).
        // T2V: scalar timestep (uniform-t, broadcast). Validated vs PyTorch (blocks 100-120 dB).
        sd::Tensor<float> ts;
        if (i2v) {
            const int64_t L_total = L_vid_i + audio_len;
            ts = sd::Tensor<float>({L_total});
            for (int64_t i = 0; i < L_total; ++i) ts.data()[i] = (i < n_clean_i) ? 0.0f : tval;
        } else {
            ts = sd::Tensor<float>({1});
            ts.data()[0] = tval;
        }

        // cond + uncond JOINT forwards -> both stream velocities each.
        auto [vv_cond,   va_cond]   = runner->compute_va(n_threads, latent, audio_latent, ctx_pos, ts);
        auto [vv_uncond, va_uncond] = runner->compute_va(n_threads, latent, audio_latent, ctx_neg, ts);
        if (vv_cond.empty() || vv_uncond.empty()) {
            printf("ERROR: DiT forward returned empty at step %d\n", step);
            return 1;
        }
        // AUDIO uncond context. PyTorch's MMDiT merges to a SINGLE shared context for
        // both streams: when spk_embed is None (our case — spk is stubbed), the merge
        // picks context_VID, so the audio stream's uncond uses the VIDEO negative and
        // the audio negative is DISCARDED (model_mm.py:1644, uncond pass spk=None at
        // pipeline_nava.py:491). So by default we reuse va_uncond (video-neg joint pass)
        // for audio — matching python — and SKIP the separate audio-neg forward.
        // NAVA_SEPARATE_AUDIO_NEG=1 restores the old behaviour (correct only once a real
        // speaker embed is wired, where the merge would pick context_audio).
        static const bool separate_audio_neg =
            have_audio_neg && getenv("NAVA_SEPARATE_AUDIO_NEG") != nullptr;
        sd::Tensor<float> va_audio_uncond;
        if (separate_audio_neg) {
            auto [vv_audio_unc, va_audio_unc] =
                runner->compute_va(n_threads, latent, audio_latent, ctx_audio_neg, ts);
            (void)vv_audio_unc;
            va_audio_uncond = std::move(va_audio_unc);
        }
        const sd::Tensor<float>& va_uncond_for_audio =
            separate_audio_neg ? va_audio_uncond : va_uncond;
        float dsig = sched.sigma_next(step) - sched.sigmas[step];

        // align_3d_cfg: 3rd forward with masking_modality=true (separate intra-modal
        // attention, same POSITIVE context) -> alignment guidance anchor for both streams.
        sd::Tensor<float> vv_mmask, va_mmask;
        if (align_cfg) {
            std::tie(vv_mmask, va_mmask) =
                runner->compute_va(n_threads, latent, audio_latent, ctx_pos, ts, /*mask_modality=*/true);
        }

        // --- video: CFG combine (patched) -> unpatchify -> Euler step ---
        // align on : eps = ec + g*(ec-eu) + ga*(ec-emmask)   (cond base, 3-term)
        // align off: eps = eu + g*(ec-eu)                     (legacy 2-term)
        sd::Tensor<float> v_p(vv_cond.shape());
        for (int64_t i = 0; i < v_p.numel(); ++i) {
            float vc = vv_cond.data()[i], vu = vv_uncond.data()[i];
            if (align_cfg && !vv_mmask.empty()) {
                float vm      = vv_mmask.data()[i];
                v_p.data()[i] = vc + o.cfg * (vc - vu) + cfg_video_align * (vc - vm);
            } else {
                v_p.data()[i] = vu + o.cfg * (vc - vu);
            }
        }
        auto v = unpatchify_video(v_p, f_len, h_grid, w_grid);
        if (dump_traj) {
            char fn[80];
            snprintf(fn, sizeof(fn), "/vel_vid_cfg_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, v, "vel_vid_cfg");
            snprintf(fn, sizeof(fn), "/vel_vid_cond_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, vv_cond, "vel_vid_cond");
            snprintf(fn, sizeof(fn), "/vel_vid_uncond_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, vv_uncond, "vel_vid_uncond");
        }
        if (use_unipc) {
            latent = usched_v.step(v, latent);  // x0-predict + multistep corrector
        } else {
            for (int64_t i = 0; i < latent.numel(); ++i) {
                latent.data()[i] += v.data()[i] * dsig;
            }
        }
        // I2V: re-pin frame 0 to the clean image latent (the per-token t=0 makes its
        // velocity ~0, but re-splicing keeps the anchor exact through the trajectory).
        splice_anchor();

        // --- audio: CFG combine -> Euler step (same schedule/shift -> same dsig) ---
        static const bool freeze_audio = getenv("NAVA_FREEZE_AUDIO") != nullptr;
        if (!freeze_audio && !va_cond.empty() && va_cond.numel() == audio_latent.numel()
            && va_uncond_for_audio.numel() == audio_latent.numel()) {
            sd::Tensor<float> va_cfg(audio_latent.shape());
            for (int64_t i = 0; i < audio_latent.numel(); ++i) {
                float ac = va_cond.data()[i], au = va_uncond_for_audio.data()[i];
                if (align_cfg && !va_mmask.empty() && va_mmask.numel() == audio_latent.numel()) {
                    float am         = va_mmask.data()[i];
                    va_cfg.data()[i] = ac + cfg_audio * (ac - au) + cfg_audio_align * (ac - am);
                } else {
                    va_cfg.data()[i] = au + cfg_audio * (ac - au);
                }
            }
            if (dump_traj) {
                char fn[80];
                snprintf(fn, sizeof(fn), "/va_cfg_%02d.bin", step);
                write_bin(std::string(dump_traj) + fn, va_cfg, "va_cfg");
                snprintf(fn, sizeof(fn), "/va_cond_%02d.bin", step);
                write_bin(std::string(dump_traj) + fn, va_cond, "va_cond");
            }
            if (use_unipc_audio) {
                audio_latent = usched_a.step(va_cfg, audio_latent);
            } else {
                for (int64_t i = 0; i < audio_latent.numel(); ++i) {
                    audio_latent.data()[i] += va_cfg.data()[i] * dsig;
                }
            }
            if (dump_traj) {
                char fn[80];
                snprintf(fn, sizeof(fn), "/aud_step_%02d.bin", step);
                write_bin(std::string(dump_traj) + fn, audio_latent, "aud_step");
            }
        }
        double asm_ = 0, asq = 0; int64_t an = audio_latent.numel();
        for (int64_t i = 0; i < an; ++i) { double a = audio_latent.data()[i]; asm_ += a; asq += a*a; }
        double astd = sqrt(asq/(double)an - (asm_/(double)an)*(asm_/(double)an));

        // track latent std per step to see whether the trajectory converges or diverges.
        double sm = 0, sq = 0; int64_t nn = latent.numel();
        for (int64_t i = 0; i < nn; ++i) { double vv = latent.data()[i]; sm += vv; sq += vv * vv; }
        double lmean = sm / (double)nn, lstd = sqrt(sq / (double)nn - lmean * lmean);
        printf("  step %2d/%d  t=%.2f sigma=%.4f dsig=%+.4f  latent mean=%+.3f std=%.3f  aud_std=%.3f\n",
               step + 1, o.steps, sched.timesteps[step], sched.sigmas[step], dsig, lmean, lstd, astd);

        if (dump_traj) {
            char fn[64];
            snprintf(fn, sizeof(fn), "/vid_step_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, latent, "vid_step");
        }
    }
    int64_t sample_t1 = ggml_time_ms();
    float sample_s = (sample_t1 - sample_t0) / 1000.0f;
    dump_stats("final latent", latent);

    // Optional: dump the final video latent [W_lat,H_lat,F,48] so it can be decoded
    // by PyTorch's own wan VAE (cross-decode → isolates a cpp VAE bug from a cpp DiT bug).
    if (const char* lp = getenv("NAVA_DUMP_LATENT")) {
        write_bin(lp, latent, "final_latent");
    }

    // free the DiT before VAE decode to lower peak memory.
    runner.reset();

    // ----- VAE decode -----
    // NAVA latent normalization (RESOLVED — see report): NAVA's LocalVideoVAEAdapter
    // has scaling_factor=1.0/shift_factor=0.0 (no renorm in the pipeline), and the
    // Wan2.2 video VAE applies z = z*std + mean INSIDE decode. The in-tree
    // WanVAERunner mirrors this exactly: diffusion_to_vae_latents = latents*std/1.0 + mean
    // with scale_factor=1.0, and the 48ch mean/std constants are BIT-IDENTICAL to
    // NAVA vae2_2.py (verified). So: latent (diffusion convention) ->
    // diffusion_to_vae_latents -> decode. decode() then maps [-1,1] -> [0,1].
    auto vae = std::make_shared<WAN::WanVAERunner>(
        backend, backend, String2TensorStorage{}, "", /*decode_only=*/true, VERSION_WAN2_2_TI2V);
    {
        ModelLoader vloader;
        if (!vloader.init_from_file_and_convert_name(o.vae, "vae.")) {
            printf("failed to init VAE loader from %s\n", o.vae.c_str());
            return 1;
        }
        vae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> vtensors;
        vae->get_param_tensors(vtensors, "first_stage_model");
        if (!vloader.load_tensors(vtensors)) {
            printf("failed to load VAE tensors\n");
            return 1;
        }
    }
    printf("VAE loaded; decoding...\n");

    sd_tiling_params_t tiling = {};
    // The Wan2.2 16x VAE decode buffer is spatial-bound (~7GB at 24x24 latent on the
    // 3060). Tile spatially at the proven-fit 24x24 latent size so full-bucket
    // (832x480 -> 52x30 latent) fits 12GB. Single-tile no-op when <= 24x24.
    tiling.enabled        = (W_lat > 24 || H_lat > 24);
    tiling.temporal_tiling = false;  // wan VAE is causal-conv3d => temporal already streamed
    tiling.tile_size_x    = 24;
    tiling.tile_size_y    = 24;
    tiling.target_overlap = 0.25f;
    int64_t dec_t0 = ggml_time_ms();
    // WanVAERunner expects a 5D video latent [W,H,T,C,1] (get_latents_mean_std
    // reads channel at ne dim 3 for 5D; a 4D tensor is treated as an IMAGE
    // [W,H,C,1] with channel at dim 2). Our sampling latent is [W,H,T,48] -> add
    // the trailing batch singleton.
    sd::Tensor<float> latent5 = latent;
    latent5.reshape_({(int64_t)W_lat, (int64_t)H_lat, (int64_t)f_len, 48, 1});
    auto latent_vae = vae->diffusion_to_vae_latents(latent5);
    auto rgb = vae->decode(n_threads, latent_vae, tiling, /*decode_video=*/true, false, false);
    int64_t dec_t1 = ggml_time_ms();
    if (rgb.empty()) {
        printf("ERROR: VAE decode failed\n");
        return 1;
    }
    printf("VAE decode %.2fs; rgb shape=%s\n", (dec_t1 - dec_t0) / 1000.0f,
           sd::tensor_shape_to_string(rgb.shape()).c_str());
    dump_stats("decoded rgb [0,1]", rgb);

    // ----- audio: decode the co-denoised audio latent -> waveform (LTX audio VAE) -----
    // audio_latent (ggml ne [128, audio_len]) carries the joint-denoised audio stream.
    // The LTX audio VAE applies its own per_channel_statistics renorm INSIDE decode and
    // the pipeline unscale (scaling_factor=1.0/shift_factor=0.0) is a no-op for NAVA, so
    // we feed the RAW latent. Output: planar waveform ggml ne [n_samples, n_ch] @ 48 kHz
    // (bwe). We interleave it into an sd_audio_t and let the webm muxer Opus-encode it.
    sd_audio_t* audio_track = nullptr;
    sd_audio_t audio_obj{};
    std::vector<float> audio_interleaved;  // must outlive the webm write below
    // Optional: dump the co-denoised audio latent (ggml ne [128, audio_len]) so the
    // cpp decode can be validated against nava_audio_vae_decode_ref.py on the SAME
    // latent. NAVA_DUMP_AUDIO_LATENT=<path> (or =1 -> /tmp/nava_audio_latent.bin).
    if (const char* dl = getenv("NAVA_DUMP_AUDIO_LATENT")) {
        std::string p = (std::string(dl) == "1") ? "/tmp/nava_audio_latent.bin" : dl;
        write_bin(p, audio_latent, "audio_latent");
        printf("dumped co-denoised audio latent -> %s (ne [128, %d])\n", p.c_str(), audio_len);
    }
    if (!o.audio_vae.empty()) {
        vae.reset();  // free the video VAE; the 340 MB audio VAE loads into the freed VRAM
        int64_t a_t0 = ggml_time_ms();
        ModelLoader aloader;
        if (!aloader.init_from_file(o.audio_vae)) {
            printf("WARN: failed to load audio VAE '%s' -> SILENT webm\n", o.audio_vae.c_str());
        } else {
            auto& atsm = aloader.get_tensor_storage_map();
            auto avae  = std::make_shared<LTXV::LTXAudioVAERunner>(backend, backend, atsm, "");
            avae->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> at;
            avae->get_param_tensors(at, "");
            if (!aloader.load_tensors(at)) {
                printf("WARN: audio VAE tensors load failed -> SILENT webm\n");
            } else {
                auto wav     = avae->decode(n_threads, audio_latent);  // ne [n_samples, n_ch]
                int64_t a_t1 = ggml_time_ms();
                if (wav.empty()) {
                    printf("WARN: audio VAE decode returned empty -> SILENT webm\n");
                } else {
                    const auto& ws  = wav.shape();
                    int64_t n_samp  = ws[0];
                    int64_t n_ch    = ws.size() > 1 ? ws[1] : 1;
                    if (n_ch < 1) n_ch = 1;
                    printf("audio VAE decode %.2fs; waveform ne=%s -> %lld samp x %lld ch @ %d Hz\n",
                           (a_t1 - a_t0) / 1000.0f, sd::tensor_shape_to_string(ws).c_str(),
                           (long long)n_samp, (long long)n_ch, avae->config.output_sample_rate());
                    dump_stats("audio waveform", wav);
                    // ggml is planar (ne0=n_samples fastest): data[c*n_samp + i].
                    // sd_audio_t wants interleaved: data[i*n_ch + c].
                    audio_interleaved.resize((size_t)(n_samp * n_ch));
                    const float* wd = wav.data();
                    for (int64_t i = 0; i < n_samp; ++i)
                        for (int64_t c = 0; c < n_ch; ++c)
                            audio_interleaved[(size_t)(i * n_ch + c)] = wd[(size_t)(c * n_samp + i)];
                    audio_obj.sample_rate  = (uint32_t)avae->config.output_sample_rate();
                    audio_obj.channels     = (uint32_t)n_ch;
                    audio_obj.sample_count = (uint64_t)n_samp;
                    audio_obj.data         = audio_interleaved.data();
                    audio_track            = &audio_obj;
                }
            }
        }
    }

    // ----- frames -> webm -----
    // rgb ggml ne [W,H,T,C], already [0,1] (decode applied (x+1)/2). Build sd_image_t per frame.
    int64_t Tpix = rgb.shape()[2];
    std::vector<sd_image_t> images;
    images.reserve((size_t)Tpix);
    for (int64_t fi = 0; fi < Tpix; ++fi) {
        images.push_back(tensor_to_sd_image(rgb, (int)fi));  // converts [0,1] f32 -> u8
    }

    std::string run_dir = o.runs_dir + "/" + o.out_name;
    mkdirs(run_dir);
    std::string webm_path = run_dir + "/clip.webm";

#ifdef SD_USE_WEBM
    int rc = create_webm_from_sd_images(webm_path.c_str(), images.data(), (int)images.size(), o.fps, 90, audio_track);
#else
    // fall back to whatever the build provides (mjpg avi etc.) but keep the name.
    int rc = create_video_from_sd_images(webm_path.c_str(), images.data(), (int)images.size(), o.fps, 90, audio_track);
#endif
    int64_t wall_t1 = ggml_time_ms();
    float wall_s = (wall_t1 - wall_t0) / 1000.0f;

    for (auto& im : images) free(im.data);

    bool ok = (rc == 0);
    printf("webm write rc=%d -> %s\n", rc, webm_path.c_str());

    // ----- meta.json (schema per tools/nava_eyetest_server.py) -----
    {
        std::string mp = run_dir + "/meta.json";
        FILE* mf = fopen(mp.c_str(), "wb");
        if (mf) {
            float s_per_step = o.steps > 0 ? sample_s / o.steps : 0.0f;
            fprintf(mf, "{\n");
            fprintf(mf, "  \"label\": \"%s\",\n", o.label.c_str());
            fprintf(mf, "  \"ok\": %s,\n", ok ? "true" : "false");
            fprintf(mf, "  \"rc\": %d,\n", rc);
            fprintf(mf, "  \"phase\": \"phase2-silent\",\n");
            fprintf(mf, "  \"w\": %d, \"h\": %d,\n", o.width, o.height);
            fprintf(mf, "  \"resolution\": \"%dx%d\",\n", o.width, o.height);
            fprintf(mf, "  \"frames\": %lld,\n", (long long)Tpix);
            fprintf(mf, "  \"latent_frames\": %d,\n", f_len);
            fprintf(mf, "  \"steps\": %d,\n", o.steps);
            fprintf(mf, "  \"seed\": %llu,\n", (unsigned long long)o.seed);
            fprintf(mf, "  \"cfg\": %.3f,\n", o.cfg);
            fprintf(mf, "  \"shift\": %.3f,\n", o.shift);
            fprintf(mf, "  \"wall_s\": %.2f,\n", wall_s);
            fprintf(mf, "  \"load_s\": %.2f,\n", (load_t1 - wall_t0) / 1000.0f);
            fprintf(mf, "  \"s_per_step\": %.3f,\n", s_per_step);
            fprintf(mf, "  \"prompt\": \"%s\",\n", o.prompt.c_str());
            fprintf(mf, "  \"backend\": \"%s\",\n", o.cuda ? "cuda" : "cpu");
            fprintf(mf, "  \"notes\": \"%s\"\n",
                    o.context.empty() ? "DUMMY context (smoke; no real prompt)" : "umT5 context.bin");
            fprintf(mf, "}\n");
            fclose(mf);
            printf("wrote %s\n", mp.c_str());
        }
    }

    printf("=== render done: %s (wall %.2fs, sample %.2fs, %d frames) ===\n",
           run_dir.c_str(), wall_s, sample_s, (int)Tpix);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// UniPC scheduler numerical validation vs PyTorch reference.
//   nava unipc-test <dir>
// Reads raw float32 .bin files written by unipc_ref.py:
//   init.bin [1,64], model_outputs.bin [N,64], traj.bin [N,64] (N=10).
// Runs UniPCSched and prints per-step + overall max-abs-error.
// ---------------------------------------------------------------------------
static std::vector<float> read_f32_bin(const std::string& path, size_t count) {
    std::vector<float> v(count);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("cannot open %s\n", path.c_str());
        return {};
    }
    size_t got = fread(v.data(), sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        printf("short read %s: got %zu want %zu\n", path.c_str(), got, count);
        return {};
    }
    return v;
}

static int run_unipc_test(int argc, char** argv) {
    std::string dir = (argc >= 3) ? argv[2] : "/tmp/unipc_ref";
    const int N = 10;
    const int D = 64;

    auto init = read_f32_bin(dir + "/init.bin", (size_t)D);
    auto mos  = read_f32_bin(dir + "/model_outputs.bin", (size_t)N * D);
    auto traj = read_f32_bin(dir + "/traj.bin", (size_t)N * D);
    if (init.empty() || mos.empty() || traj.empty()) {
        printf("FAIL: could not load reference .bin files from %s\n", dir.c_str());
        return 1;
    }

    UniPCSched sched;
    sched.set_timesteps(N, 5.0f);

    printf("sigmas (cpp): ");
    for (float s : sched.sigmas) printf("%.6f ", s);
    printf("\n");

    sd::Tensor<float> sample({1, (int64_t)D}, init);

    double overall_max = 0.0;
    bool any_bad       = false;
    for (int k = 0; k < N; ++k) {
        std::vector<float> mo_k(mos.begin() + (size_t)k * D, mos.begin() + (size_t)(k + 1) * D);
        sd::Tensor<float> mo({1, (int64_t)D}, mo_k);
        sample = sched.step(mo, sample);

        double step_max = 0.0;
        for (int i = 0; i < D; ++i) {
            float v = sample.data()[i];
            if (std::isnan(v) || std::isinf(v)) any_bad = true;
            double e = std::fabs((double)v - (double)traj[(size_t)k * D + i]);
            if (e > step_max) step_max = e;
        }
        if (step_max > overall_max) overall_max = step_max;
        printf("step %2d  sigma_t=%.6f -> %.6f   max_abs_err=%.3e\n",
               k, sched.sigmas[(size_t)k], sched.sigmas[(size_t)k + 1], step_max);
    }

    printf("\nfinal head (cpp): ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", sample.data()[i]);
    printf("\nfinal head (py):  ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", traj[(size_t)(N - 1) * D + i]);
    printf("\n\nOVERALL MAX ABS ERROR = %.6e\n", overall_max);
    printf("NaN/Inf present: %s\n", any_bad ? "YES" : "no");
    bool pass = (overall_max < 1e-4) && !any_bad;
    printf("GATE (<1e-4): %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    sd_set_log_callback(log_cb, nullptr);

    if (argc >= 2 && std::string(argv[1]) == "render") {
        return run_render(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "unipc-test") {
        return run_unipc_test(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "encode-prompt") {
        // nava encode-prompt <umt5_gguf> <prompt.txt> <out.bin>  — encode a text prompt
        // through umT5-xxl (GPU) into a NAVA context bin [4096,512] (zero-padded), matching
        // dump_three_ctx.py. Makes the cpp nava pipeline self-sufficient (text -> context).
        if (argc < 5) {
            printf("usage: %s encode-prompt <umt5_gguf> <prompt.txt> <out.bin>\n", argv[0]);
            return 1;
        }
        std::string gguf = argv[2], pf = argv[3], out = argv[4];
        std::string text;
        {
            FILE* f = fopen(pf.c_str(), "rb");
            if (!f) { printf("cannot open prompt file %s\n", pf.c_str()); return 1; }
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (n > 0) { text.resize((size_t)n); size_t r = fread(&text[0], 1, (size_t)n, f); text.resize(r); }
            fclose(f);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        }
        // NAVA data-loader caption transform (nava_src/data/t2v.py:391, the
        // use_speech_special_token=false path that nava_640.yaml/prod selects):
        //   text = text.replace("<S>", "<S><extra_id_2>")
        // i.e. insert the umT5 sentinel <extra_id_2> (id 256297) right after each
        // speech-start marker. The pipeline ALWAYS does this before the umT5 encode;
        // omitting it shifts every token from the spoken line onward and the
        // bidirectional umT5 context diverges there (cos ~0.35 past the marker),
        // which the joint DiT faithfully turns into GARBLED SPEECH while the video
        // (barely dependent on those tokens) looks fine. <E> stays literal, matching
        // use_speech_special_token=false. The cpp unigram tokenizer already maps
        // <extra_id_2> -> 256297, so only this textual insertion is needed.
        // The TRAILING SPACE is load-bearing: HF/sentencepiece re-adds the metaspace
        // (▁) to the word AFTER a special token, so "<extra_id_2>We" tokenizes as
        // [<extra_id_2>, ▁We]. The cpp unigram tokenizer does NOT add ▁ after a
        // special token, so without the space "We" tokenizes as bare 7440 instead of
        // ▁We=1136 — one wrong token right at the start of the spoken line, which
        // ripples through the bidirectional context. The inserted space makes the cpp
        // tokenizer emit ▁We, matching HF exactly (0 token diffs). whitespace_clean
        // (in nava_normalize_text below) preserves the single space.
        {
            const std::string from = "<S>", to = "<S><extra_id_2> ";
            for (size_t p = 0; (p = text.find(from, p)) != std::string::npos; p += to.size())
                text.replace(p, from.size(), to);
        }
        // NFKC full-width normalization + whitespace_clean, to match python's tokenizer
        // (the cpp t5 unigram tokenizer skips the sentencepiece NFKC charsmap).
        text = nava_normalize_text(text);
        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        ModelLoader ml;
        if (!ml.init_from_file_and_convert_name(gguf)) { printf("umt5 load fail: %s\n", gguf.c_str()); return 1; }
        auto& tsm           = ml.get_tensor_storage_map();
        const char* T5PREFIX = "text_encoders.t5xxl.transformer";
        auto t5   = std::make_shared<T5Embedder>(backend, backend, tsm, T5PREFIX, /*is_umt5=*/true);
        t5->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        t5->get_param_tensors(tensors, T5PREFIX);
        if (!ml.load_tensors(tensors)) { printf("umt5 tensors load fail\n"); return 1; }
        printf("umt5 loaded; prompt is %zu bytes\n", text.size());
        // No token padding (max_length=0, padding=false) -> raw token length, then we
        // zero-pad the OUTPUT to 512, exactly like dump_three_ctx.py.
        auto tw          = t5->tokenize(text, 0, false);
        auto& tokens     = std::get<0>(tw);
        auto& masks      = std::get<2>(tw);
        if (const char* tp = getenv("NAVA_DUMP_TOKENS")) {
            FILE* tf = fopen(tp, "w");
            if (tf) { for (auto id : tokens) fprintf(tf, "%d\n", (int)id); fclose(tf);
                      printf("dumped %zu token ids -> %s\n", tokens.size(), tp); }
        }
        // NAVA caps umT5 at text_len=512 (model_loading_utils.py): truncate the INPUT
        // to 512 so the encoder doesn't attend past it. (Truncating only the OUTPUT
        // would let the kept embeddings see the dropped tail -> not faithful to python.)
        const size_t kMaxTok = 512;
        if (tokens.size() > kMaxTok) {
            // Match python (HF tokenizer seq_len=512 truncation): keep the first 511
            // content tokens and KEEP the EOS at position 511 — do NOT keep 512 content
            // and drop EOS. umT5 is a bidirectional encoder, so a missing/!=EOS terminator
            // shifts EVERY token's embedding (context cos ~0.57 vs ~0.99 otherwise).
            int eos = tokens.back();  // umT5 EOS (id 1) at the natural sequence end
            printf("tokens=%zu -> truncating to %zu (511 content + EOS %d, NAVA umT5 text_len cap)\n",
                   tokens.size(), kMaxTok, eos);
            tokens.resize(kMaxTok - 1);
            tokens.push_back(eos);
            masks.assign(kMaxTok, 0.0f);  // tokenize already mapped valid->0.0 / pad->-HUGE (t5.hpp:509)
        }
        printf("tokens=%zu\n", tokens.size());
        auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
        auto attn_mask = sd::Tensor<float>::from_vector(masks);
        auto emb       = t5->model.compute(8, input_ids, attn_mask);  // ne [4096, L]
        if (emb.empty()) { printf("umt5 encode returned empty\n"); return 1; }
        int64_t C  = emb.shape()[0];
        int64_t L  = emb.shape().size() > 1 ? emb.shape()[1] : 1;
        int64_t Lc = std::min<int64_t>(L, 512);
        sd::Tensor<float> ctx({C, 512});
        std::fill(ctx.data(), ctx.data() + ctx.numel(), 0.0f);
        for (int64_t tok = 0; tok < Lc; ++tok)
            for (int64_t c = 0; c < C; ++c)
                ctx.data()[c + C * tok] = emb.data()[c + C * tok];
        write_bin(out, ctx, "context");
        dump_stats("umt5 context", ctx);
        printf("encoded -> %s  ne=[%lld,512]  (token L=%lld)\n", out.c_str(), (long long)C, (long long)L);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "ltx-audio-test") {
        // nava ltx-audio-test <gguf> <latent.bin> [out_wave.bin]  — load the audio VAE gguf
        // into the in-tree LTXV decoder + decode on CUDA (CPU hits an F16 assert), dump waveform.
        if (argc < 4) { printf("usage: %s ltx-audio-test <gguf> <latent.bin> [out.bin]\n", argv[0]); return 1; }
        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        ModelLoader ml;
        if (!ml.init_from_file(argv[2])) { printf("audio vae load fail\n"); return 1; }
        auto& tsm = ml.get_tensor_storage_map();
        auto avae = std::make_shared<LTXV::LTXAudioVAERunner>(backend, backend, tsm, "");
        avae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> at;
        avae->get_param_tensors(at, "");
        if (!ml.load_tensors(at)) { printf("audio vae tensors fail\n"); return 1; }
        printf("audio vae loaded (%zu params); config sample_rate=%d\n", at.size(), avae->config.sample_rate);
        auto z = sd::load_tensor_from_file_as_tensor<float>(argv[3]);
        printf("latent shape=%s\n", sd::tensor_shape_to_string(z.shape()).c_str());
        auto wav = avae->decode(8, z);
        if (wav.empty()) { printf("ERROR: audio decode returned empty\n"); return 1; }
        dump_stats("waveform", wav);
        if (argc >= 5) write_bin(argv[4], wav, "waveform");
        return 0;
    }
    if (argc < 2) {
        printf("usage:\n");
        printf("  %s <gguf> [<input_dir>] [<out_dir>]      # phase-1 single forward\n", argv[0]);
        printf("  %s render --prompt \"...\" [opts]           # phase-2 silent render\n", argv[0]);
        printf("  %s unipc-test [<ref_dir>]                 # UniPC scheduler numeric validation\n", argv[0]);
        return 1;
    }
    return run_single_forward(argc, argv);
}
