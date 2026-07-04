// ---------------------------------------------------------------------------
// ShotStream streaming multi-shot T2V driver (Stage A–E scaffold).
//
// Drives the causal AR loop (WAN_SHOTSTREAM::ShotStreamRunner): per shot, prefill
// the global/context cache, then generate 7 chunks of 3 latent frames each with the
// 4-step warped DMD schedule + clean-rewrite KV persistence.
//
// This CLI now wires the two previously-stubbed halves so a REAL text prompt
// produces viewable pixels through the streaming path:
//   * umT5-XXL text encoding (--t5xxl + --prompt) -> [4096,512,1] context, fed to
//     the causal DiT in place of the zeros stub (single caption reused per shot).
//   * Wan2.1 VAE decode (--vae) of each shot latent [104,60,21,16] -> 81 pixel
//     frames -> PNG (f%03d.png) -> mp4 (ffmpeg, 16 fps).
// Both reuse the existing runners (T5Embedder, WAN::WanVAERunner), matching the
// proven examples/s2v/main.cpp wiring. Multi-shot history VAE pixel round-trip and
// the change_rope shot-phase remain follow-ups (see IMPL-NOTES-shotstream-stageA.md).
//
// Run (3060, device 0):
//   sd-shotstream --dit models/shotstream-1.3b-dit-f16.gguf \
//                 --t5xxl models/longcat-umt5-xxl-q8_0.gguf \
//                 --vae   models/longcat-wan-vae-f16.gguf \
//                 -p "a red fox trotting through a snowy pine forest at dawn" \
//                 --shots 1 --W 832 --H 480 --fps 16 --out shotstream_out
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model.h"
#include "stable-diffusion.h"
#include "model/te/t5.hpp"
#include "tensor.hpp"
#include "tensor_ggml.hpp"
#include "model/diffusion/wan.hpp"
#include "model/diffusion/wan_shotstream.hpp"
#include "model/vae/wan_vae.hpp"  // WAN::WanVAERunner (Wan2.1 3D VAE)

static void log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
}

static void dump_stats(const char* tag, const sd::Tensor<float>& t) {
    if (t.empty()) { printf("%-22s EMPTY\n", tag); return; }
    double sum = 0, sq = 0, mn = 1e30, mx = -1e30;
    int64_t nnan = 0;
    for (int64_t i = 0; i < t.numel(); ++i) {
        double v = t.data()[i];
        if (std::isnan(v) || std::isinf(v)) { nnan++; continue; }
        sum += v; sq += v * v;
        if (v < mn) mn = v; if (v > mx) mx = v;
    }
    double mean = sum / (double)t.numel();
    double var  = sq / (double)t.numel() - mean * mean;
    printf("%-22s shape=%s mean=%.5f std=%.5f min=%.4f max=%.4f nnan=%lld\n",
           tag, sd::tensor_shape_to_string(t.shape()).c_str(), mean,
           var > 0 ? sqrt(var) : 0.0, mn, mx, (long long)nnan);
}

static void write_bin(const std::string& path, const sd::Tensor<float>& t, const char* tag) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("WARN: cannot write %s\n", path.c_str()); return; }
    int32_t ndim = (int32_t)t.dim();
    fwrite(&ndim, sizeof(int32_t), 1, f);
    for (int i = 0; i < ndim; i++) { int64_t s = t.shape()[i]; fwrite(&s, sizeof(int64_t), 1, f); }
    fwrite(t.data(), sizeof(float), (size_t)t.numel(), f);
    fclose(f);
    printf("wrote %s [%s] numel=%lld\n", path.c_str(), tag, (long long)t.numel());
}

static bool read_context_bin(const std::string& path, sd::Tensor<float>& out) {
    // Expect [text_dim=4096, text_len=512, 1] float32 (umT5 embedding for one shot).
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t ndim = 0;
    if (fread(&ndim, sizeof(int32_t), 1, f) != 1 || ndim < 2) { fclose(f); return false; }
    std::vector<int64_t> shp(ndim);
    for (int i = 0; i < ndim; i++) { if (fread(&shp[i], sizeof(int64_t), 1, f) != 1) { fclose(f); return false; } }
    int64_t n = 1; for (int i = 0; i < ndim; i++) n *= shp[i];
    out = sd::Tensor<float>(shp);
    size_t got = fread(out.data(), sizeof(float), (size_t)n, f);
    fclose(f);
    return got == (size_t)n;
}

// Wan2.1 (16ch) VAE latent normalization constants (vae2_1.py / WanVAERunner).
// The DiT operates in normalized diffusion space z = (raw - mean)/std; WanVAERunner::
// decode does NOT re-apply mean/std, so convert z -> raw VAE space (z*std+mean) before
// decode. The latent ne order is [W,H,T,C=16] (C outermost, stride W*H*T) — index the
// channel at ne[3], NOT via diffusion_to_vae_latents() which assumes channel at dim 2.
static const float WAN_VAE_MEAN[16] = {
    -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
    0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
static const float WAN_VAE_STD[16] = {
    2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
    3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

static void vae_diffusion_to_mu(sd::Tensor<float>& t) {
    int64_t W = t.shape()[0], H = t.shape()[1], T = t.shape()[2], C = t.shape()[3];
    GGML_ASSERT(C == 16);
    float* d = t.data();
    for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < W * H * T; ++i)
            d[c * W * H * T + i] = d[c * W * H * T + i] * WAN_VAE_STD[c] + WAN_VAE_MEAN[c];
}

// Inverse: raw VAE mu -> diffusion space z = (mu - mean)/std. Used on per-frame history
// encode (VAE returns raw mu; the DiT global cache wants diffusion-space latents).
static void vae_mu_to_diffusion(sd::Tensor<float>& t) {
    int64_t W = t.shape()[0], H = t.shape()[1], T = t.shape()[2], C = t.shape()[3];
    GGML_ASSERT(C == 16);
    float* d = t.data();
    for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < W * H * T; ++i)
            d[c * W * H * T + i] = (d[c * W * H * T + i] - WAN_VAE_MEAN[c]) / WAN_VAE_STD[c];
}

// Extract pixel frame `fi` from an RGB video tensor ne [W,H,T,3] -> ne [W,H,1,3] ([0,1]).
static sd::Tensor<float> extract_frame(const sd::Tensor<float>& rgb, int64_t fi) {
    int64_t W = rgb.shape()[0], H = rgb.shape()[1], T = rgb.shape()[2], C = rgb.shape()[3];
    sd::Tensor<float> f({W, H, 1, C});
    const float* s = rgb.data();
    float* d       = f.data();
    for (int64_t c = 0; c < C; ++c)
        std::memcpy(d + c * W * H, s + (c * T + fi) * W * H, sizeof(float) * (size_t)(W * H));
    return f;
}

// ---- minimal .npy I/O (v1.0, C-order) for the offline oracle-parity dumps ----
static void write_npy(const std::string& path, const float* data, const std::vector<int64_t>& shape) {
    int64_t n = 1;
    std::string dims;
    for (size_t i = 0; i < shape.size(); ++i) { n *= shape[i]; dims += std::to_string(shape[i]); dims += ", "; }
    if (shape.size() == 1) dims = std::to_string(shape[0]) + ",";  // numpy 1-D tuple
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
    size_t pre = 10;  // magic(6)+ver(2)+hlen(2)
    size_t total = pre + hdr.size() + 1;  // +newline
    size_t pad = (64 - (total % 64)) % 64;
    hdr.append(pad, ' '); hdr.push_back('\n');
    uint16_t hlen = (uint16_t)hdr.size();
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("WARN: cannot write %s\n", path.c_str()); return; }
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, sizeof(uint16_t), 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    fwrite(data, sizeof(float), (size_t)n, f);
    fclose(f);
    printf("wrote %s (%lld floats, shape [%s])\n", path.c_str(), (long long)n, dims.c_str());
}

static bool read_npy_f32(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path.c_str()); return false; }
    char magic[8]; if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return false; }
    uint16_t hlen = 0; if (fread(&hlen, sizeof(uint16_t), 1, f) != 1) { fclose(f); return false; }
    std::string hdr(hlen, '\0'); if (fread(&hdr[0], 1, hlen, f) != hlen) { fclose(f); return false; }
    bool f8 = hdr.find("'<f8'") != std::string::npos || hdr.find("\"<f8\"") != std::string::npos;
    // parse shape tuple
    shape.clear();
    size_t sp = hdr.find("'shape':"); size_t lp = hdr.find('(', sp), rp = hdr.find(')', lp);
    std::string tup = hdr.substr(lp + 1, rp - lp - 1);
    size_t pos = 0; int64_t n = 1;
    while (pos < tup.size()) {
        while (pos < tup.size() && (tup[pos] == ' ' || tup[pos] == ',')) pos++;
        if (pos >= tup.size()) break;
        int64_t v = atoll(&tup[pos]); shape.push_back(v); n *= v;
        while (pos < tup.size() && tup[pos] != ',') pos++;
    }
    out.resize((size_t)n);
    if (f8) {
        std::vector<double> tmp((size_t)n);
        size_t got = fread(tmp.data(), sizeof(double), (size_t)n, f);
        for (size_t i = 0; i < (size_t)n; ++i) out[i] = (float)tmp[i];
        fclose(f);
        return got == (size_t)n;
    }
    size_t got = fread(out.data(), sizeof(float), (size_t)n, f);
    fclose(f);
    return got == (size_t)n;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    sd_set_log_callback(log_cb, nullptr);

    std::string dit_path, out_dir = "out", context_path, t5_path, vae_path;
    std::string dump_rope_dir, dump_block0_dir, block0_input_npy, decode_latent_bin;
    std::string prompt = "a red fox trotting through a snowy pine forest at dawn, "
                         "volumetric morning light, cinematic, photorealistic";
    // PER-SHOT PROMPTS: N captions, one per shot ("chain on real-time prompting").
    // Populated by repeatable -p/--prompt or pipe-separated --prompts "p1|p2|p3".
    // Empty → seeded with the single default `prompt` below (byte-identical to `-p X`).
    // GLOBAL caption (scene-level, shared): shot i's umT5 input is `global_prompt + prompt[i]`
    // (PLAIN string concat, NO separator — ref Teacher_Ode_Sample.py:222 / wan/text2video.py:220
    // `global_captions[0][0] + shots_captions[0][i]...`). Empty by default so single-`-p` is
    // byte-identical to today.
    std::vector<std::string> prompts;
    std::string global_prompt;
    int shots = 1, W = 832, H = 480, n_threads = 8, fps = 16;
    int chunks_override = 0;  // >0 => cap frames_per_shot to chunks*num_frame_per_block (short perf runs)
    bool no_vae = false;      // skip VAE decode (DiT-only perf runs)
    bool serve  = false;      // P0: warm interactive stdin loop (one line = one chained shot)
    unsigned seed = 42;
    bool vae_tiling = true, vae_temporal_tiling = true;
    float vae_rel_x = 0.5f, vae_rel_y = 0.5f;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--dit") dit_path = next();
        else if (a == "--dump-rope") dump_rope_dir = next();
        else if (a == "--dump-block0") dump_block0_dir = next();
        else if (a == "--block0-input") block0_input_npy = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--context") context_path = next();
        else if (a == "--t5xxl" || a == "--umt5") t5_path = next();
        else if (a == "--vae") vae_path = next();
        else if (a == "--decode-latent") decode_latent_bin = next();  // VAE-only A/B: decode a saved shotNN_latent.bin
        else if (a == "-p" || a == "--prompt") prompts.push_back(next());  // repeatable: shot k's caption
        else if (a == "--global-prompt" || a == "--global") global_prompt = next();  // scene-level, prepended to each shot
        else if (a == "--prompts") {  // pipe-separated per-shot captions: "p1|p2|p3|p4"
            std::string all = next();
            size_t pos = 0, bar;
            while ((bar = all.find('|', pos)) != std::string::npos) { prompts.push_back(all.substr(pos, bar - pos)); pos = bar + 1; }
            prompts.push_back(all.substr(pos));
        }
        else if (a == "--shots") shots = atoi(next().c_str());
        else if (a == "--chunks") chunks_override = atoi(next().c_str());
        else if (a == "--serve") serve = true;
        else if (a == "--no-vae") no_vae = true;
        else if (a == "--W") W = atoi(next().c_str());
        else if (a == "--H") H = atoi(next().c_str());
        else if (a == "--fps") fps = atoi(next().c_str());
        else if (a == "--seed") seed = (unsigned)atoi(next().c_str());
        else if (a == "--threads") n_threads = atoi(next().c_str());
        else if (a == "--vae-tiling") vae_tiling = true;
        else if (a == "--no-vae-tiling") vae_tiling = false;
        else if (a == "--temporal-tiling") vae_temporal_tiling = true;
        else if (a == "--no-temporal-tiling") vae_temporal_tiling = false;
        else if (a == "--vae-relative-tile-size") {
            std::string v = next();
            size_t xp = v.find('x');
            if (xp != std::string::npos) { vae_rel_x = atof(v.substr(0, xp).c_str()); vae_rel_y = atof(v.substr(xp + 1).c_str()); }
            else { vae_rel_x = vae_rel_y = atof(v.c_str()); }
        }
    }
    // ---- P0.5: OFFLINE RoPE-table parity dump (no model, no backend) ----
    // Emits rope_freqs_chunk_shot0.npy / shot1.npy in the oracle's [4680,64,2] (cos,sin)
    // layout, built by the SAME pe_build_frames the runner uses (t_offset=6, phase θ=1/6
    // for shot 1). Diff vs tools/shotstream_goldens via tools/shotstream_compare.py.
    if (!dump_rope_dir.empty()) {
        { std::string mk = "mkdir -p '" + dump_rope_dir + "'"; (void)system(mk.c_str()); }
        WAN::WanConfig cfg;  // defaults = ShotStream RoPE params: patch{1,2,2}, θ=10000, axes{44,42,42}
        const float theta_shot = 1.0f / 6.0f;
        const int T = 3, H = 60, W = 104;                 // 3 latent frames @ 60x104 -> 4680 tokens
        const int chans = (int)cfg.axes_dim_sum / 2;      // 64 complex channels
        const int stride = chans * 4;                     // 4 floats/channel ([c,-s,s,c])
        for (int shot = 0; shot <= 1; ++shot) {
            // t_offset=6 (condition_start_frame) matches the oracle's start_frame=6.
            auto pe = WAN_SHOTSTREAM::pe_build_frames(cfg, theta_shot, /*phase_enabled=*/true,
                                                      T, H, W, /*t_offset=*/6, /*shot_idx=*/shot);
            int npos = (int)(pe.size() / stride);
            std::vector<float> ri((size_t)npos * chans * 2);
            for (int p = 0; p < npos; ++p)
                for (int c = 0; c < chans; ++c) {
                    float cs = pe[(size_t)p * stride + (size_t)c * 4 + 0];  // cos
                    float sn = pe[(size_t)p * stride + (size_t)c * 4 + 2];  // sin
                    ri[((size_t)p * chans + c) * 2 + 0] = cs;
                    ri[((size_t)p * chans + c) * 2 + 1] = sn;
                }
            char nm[512]; snprintf(nm, sizeof(nm), "%s/rope_freqs_chunk_shot%d.npy", dump_rope_dir.c_str(), shot);
            write_npy(nm, ri.data(), {(int64_t)npos, (int64_t)chans, 2});
        }
        printf("RoPE dump done -> %s\n", dump_rope_dir.c_str());
        return 0;
    }

    if (prompts.empty()) prompts.push_back(prompt);  // default single caption (reused per shot)

    if (dit_path.empty() && decode_latent_bin.empty()) {
        printf("usage: sd-shotstream --dit <gguf> [--t5xxl <umt5.gguf>] [-p <prompt>] [--vae <vae.gguf>]\n");
        printf("                     [--context <umt5.bin>] [--shots N] [--W 832 --H 480] [--fps 16]\n");
        printf("                     [--seed 42] [--out dir] [--vae-relative-tile-size 0.5x0.5]\n");
        return 1;
    }
    if (!out_dir.empty()) { std::string mk = "mkdir -p '" + out_dir + "'"; (void)system(mk.c_str()); }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    bool on_gpu = (backend != nullptr);
    if (!backend) { backend = sd_backend_cpu_init(); printf("backend: CPU\n"); }
    else printf("backend: GPU (%s)\n", ggml_backend_name(backend));

    // The fork's fused-rope op (ggml_rope_pe / ROPE_PE) is CUDA-only and aborts
    // ggml_graph_plan on the CPU backend. On a CPU fallback force the decomposed path.
    if (!on_gpu) setenv("LONGCAT_NO_FUSED_ROPE", "1", /*overwrite=*/0);

    // ------------------------------------------------------------------
    // VAE-ONLY DECODE A/B (--decode-latent shotNN_latent.bin). Loads ONLY the VAE
    // (no umT5, no DiT resident) and decodes a saved diffusion-space latent
    // [W,H,T,16] to pixels + mp4. This is the cheapest possible harness for the VAE
    // decode perf/VRAM levers (conv3d dispatch, tile size, temporal chunk, overlap)
    // — ~decode-time only, no 130s DiT. Uses the same dec_tiling / vae_diffusion_to_mu
    // path as the real shot loop so its timing/peak transfer 1:1.
    // ------------------------------------------------------------------
    if (!decode_latent_bin.empty()) {
        if (vae_path.empty()) { printf("ERROR: --decode-latent needs --vae\n"); return 1; }
        sd::Tensor<float> lat;
        if (!read_context_bin(decode_latent_bin, lat) || lat.dim() != 4 || lat.shape()[3] != 16) {
            printf("ERROR: read latent %s (want [W,H,T,16])\n", decode_latent_bin.c_str()); return 1;
        }
        printf("decode-latent: %s [%lld,%lld,%lld,%lld]\n", decode_latent_bin.c_str(),
               (long long)lat.shape()[0], (long long)lat.shape()[1], (long long)lat.shape()[2], (long long)lat.shape()[3]);
        auto vae1 = std::make_shared<WAN::WanVAERunner>(backend, backend, String2TensorStorage{}, "",
                                                        /*decode_only=*/true, VERSION_WAN2);
        { ModelLoader loader;
          if (!loader.init_from_file_and_convert_name(vae_path, "vae.")) { printf("ERROR: VAE loader init\n"); return 1; }
          vae1->alloc_params_buffer();
          std::map<std::string, ggml_tensor*> tensors; vae1->get_param_tensors(tensors, "first_stage_model");
          if (!loader.load_tensors(tensors)) { printf("ERROR: VAE tensor load\n"); return 1; } }
        vae1->set_flash_attention_enabled(true);
        // GGML_CUDNN_CONV=1: route VAE 2D convs through GGML_OP_CONV_2D (conv2d-cudnn.cu,
        // F32-accumulate implicit-GEMM, im2col-free) instead of ggml_conv_2d's im2col.
        // Mirrors src/stable-diffusion.cpp:1076 for this VAE-only A/B path. Resample.1's
        // F32 input cast (wan_vae.hpp) makes dst F32 too, passing the cuDNN dtype gate.
        if (getenv("GGML_CUDNN_CONV")) vae1->set_conv2d_direct_enabled(true);
        vae1->set_temporal_tiling_enabled(vae_temporal_tiling);
        sd_tiling_params_t dt = {};
        dt.enabled = vae_tiling; dt.temporal_tiling = vae_temporal_tiling;
        const char* ovl = getenv("SHOTSTREAM_VAE_OVERLAP");
        dt.target_overlap = ovl ? (float)atof(ovl) : 0.25f;
        dt.rel_size_x = vae_rel_x; dt.rel_size_y = vae_rel_y;
        vae_diffusion_to_mu(lat);
        lat.unsqueeze_(4);
        int64_t td0 = ggml_time_ms();
        auto rgb = vae1->decode(n_threads, lat, dt, /*decode_video=*/true, false, false);
        int64_t td1 = ggml_time_ms();
        printf("[DECODE-LATENT] rel=%.2fx%.2f overlap=%.3f tchunk=%s cudnn3d=%s: %.1fs\n",
               vae_rel_x, vae_rel_y, dt.target_overlap,
               getenv("LONGCAT_VAE_TEMPORAL_CHUNK") ? getenv("LONGCAT_VAE_TEMPORAL_CHUNK") : "?",
               getenv("GGML_CUDNN_CONV3D") ? getenv("GGML_CUDNN_CONV3D") : "off",
               (td1 - td0) / 1000.0);
        dump_stats("decoded rgb", rgb);
        // Dump a few frames as PNG so tile seams can be eye-checked (--out subdir).
        if (!rgb.empty() && rgb.dim() >= 4) {
            int64_t Wp = rgb.shape()[0], Hp = rgb.shape()[1], Tp = rgb.shape()[2];
            const float* d = rgb.data();
            std::vector<unsigned char> fr((size_t)Wp * Hp * 3);
            const char* dtag = getenv("SHOTSTREAM_DECTAG") ? getenv("SHOTSTREAM_DECTAG") : "dec";
            for (int64_t t : {(int64_t)0, Tp / 2, Tp - 1}) {
                for (int64_t y = 0; y < Hp; ++y)
                    for (int64_t x = 0; x < Wp; ++x)
                        for (int c = 0; c < 3; ++c) {
                            float v = d[(((size_t)c * Tp + t) * Hp + y) * Wp + x];
                            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                            fr[((size_t)y * Wp + x) * 3 + c] = (unsigned char)std::lround(v * 255.0f);
                        }
                char fp[600]; snprintf(fp, sizeof(fp), "%s/%s_f%03lld.png", out_dir.c_str(), dtag, (long long)t);
                stbi_write_png(fp, (int)Wp, (int)Hp, 3, fr.data(), (int)Wp * 3);
                printf("wrote %s\n", fp);
            }
        }
        return 0;
    }

    // ------------------------------------------------------------------
    // 1) TEXT CONDITIONING — umT5-XXL encode of --prompt (single caption reused per
    //    shot). Runs on a dedicated CPU backend (matches examples/s2v), so it never
    //    competes with the DiT/VAE for VRAM. Falls back to a precomputed --context
    //    .bin or a zeros stub (DiT-path smoke only) when no --t5xxl is given.
    // ------------------------------------------------------------------
    // umT5-XXL encoder. In --serve mode it is kept RESIDENT (on a dedicated CPU backend,
    // so it never eats VRAM) and re-invoked per prompt via encode_prompt(); in one-shot
    // mode it is loaded, used once, and freed (original behavior). encode_prompt tokenizes,
    // runs the umT5, and zero-pads to text_len=512 (the Wan T2V DiT does not pad internally).
    std::shared_ptr<T5Embedder> t5;
    auto encode_prompt = [&](const std::string& p) -> sd::Tensor<float> {
        auto tw  = t5->tokenize(p, 512, false);
        auto ids = sd::Tensor<int32_t>::from_vector(std::get<0>(tw));
        auto am  = sd::Tensor<float>::from_vector(std::get<2>(tw));
        sd::Tensor<float> ctx = t5->model.compute(n_threads, ids, am);  // [4096, n_tok, 1]
        const int64_t TEXT_LEN = 512;
        int64_t tok = ctx.shape()[1];
        if (tok < TEXT_LEN) {
            sd::Tensor<float> ctx_pad({ctx.shape()[0], TEXT_LEN, ctx.shape()[2]});
            std::fill(ctx_pad.data(), ctx_pad.data() + ctx_pad.numel(), 0.0f);
            sd::ops::slice_assign(&ctx_pad, 1, 0, tok, ctx);
            ctx = std::move(ctx_pad);
        }
        return ctx;
    };

    // PER-SHOT PROMPTS: one umT5 [4096,512,1] context per caption, encoded up-front. Shot k
    // feeds `contexts[min(k, N-1)]` to BOTH its context prefill and its generation (§4.3
    // multi_caption=False path: shot k uses its OWN caption for prefill + gen). If fewer
    // prompts than shots are supplied the LAST caption is reused (so `-p X` alone stays
    // byte-identical to the old single-caption behavior). `context` keeps a single alias for
    // the serve path (which re-encodes per stdin line) + legacy single-context refs.
    sd::Tensor<float> context;
    std::vector<sd::Tensor<float>> contexts;
    if (!context_path.empty() && read_context_bin(context_path, context)) {
        printf("context(.bin): %s\n", context_path.c_str());
        dump_stats("context(.bin)", context);
        contexts.push_back(context);  // single precomputed context, reused per shot
    } else if (!t5_path.empty()) {
        printf("loading umT5 '%s'%s\n", t5_path.c_str(), serve ? " (RESIDENT for --serve)" : "");
        int64_t t5_t0 = ggml_time_ms();
        ggml_backend_t t5_backend = sd_backend_cpu_init();
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(t5_path, "text_encoders.t5xxl.transformer.")) {
            printf("ERROR: umT5 loader init %s\n", t5_path.c_str()); return 1;
        }
        t5 = std::make_shared<T5Embedder>(t5_backend, t5_backend, loader.get_tensor_storage_map(),
                                          "text_encoders.t5xxl.transformer", /*is_umt5=*/true);
        t5->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        if (!loader.load_tensors(tensors)) { printf("WARN: some umT5 tensors failed to load\n"); }
        printf("[STAGE] umT5 load: %.1fs\n", (ggml_time_ms() - t5_t0) / 1000.0);
        if (!serve) {
            // one-shot / batch: encode EACH per-shot caption now, then release the umT5 host
            // params. umT5 is CPU-backed so N contexts (~4 MB each) never touch VRAM. Shot i's
            // umT5 input = global_prompt + prompt[i] (PLAIN concat, ref Teacher_Ode_Sample.py:222).
            printf("encoding %zu per-shot caption(s)%s:\n", prompts.size(),
                   global_prompt.empty() ? "" : " (global prefix)");
            if (!global_prompt.empty()) printf("  global: \"%s\"\n", global_prompt.c_str());
            for (size_t p = 0; p < prompts.size(); ++p) {
                std::string full = global_prompt + prompts[p];  // global + shot (no separator)
                printf("  caption[%zu]: \"%s\"\n", p, full.c_str());
                auto c = encode_prompt(full);
                char tag[64]; snprintf(tag, sizeof(tag), "context[%zu] (umT5, 512)", p);
                dump_stats(tag, c);
                contexts.push_back(std::move(c));
            }
            context = contexts.front();  // single-context alias for legacy refs
            t5->model.free_params_buffer();
            t5.reset();
        }
        // serve: leave t5 resident; context is encoded per-prompt inside the serve loop.
    } else {
        context = sd::Tensor<float>({(int64_t)4096, (int64_t)512, (int64_t)1});
        for (int64_t i = 0; i < context.numel(); i++) context.data()[i] = 0.0f;
        printf("context: zeros [4096,512,1] (no --t5xxl/--context — DiT path smoke only)\n");
        contexts.push_back(context);
    }
    // Shot k's caption: reuse the last when fewer prompts than shots were supplied.
    auto context_for_shot = [&](int k) -> const sd::Tensor<float>& {
        if (contexts.empty()) return context;
        int idx = (k < (int)contexts.size()) ? k : (int)contexts.size() - 1;
        return contexts[idx];
    };

    // ------------------------------------------------------------------
    // 2) DiT (ShotStream causal streaming). Weights resident by default (2.6 GB f16,
    //    fits the 12 GB card with room for the causal compute buffer); flash/cuDNN
    //    streams the KV-cache scores. See IMPL-NOTES for the offload/flash rationale.
    // ------------------------------------------------------------------
    ggml_backend_t dit_params_backend = backend;
    if (getenv("SHOTSTREAM_NO_OFFLOAD") == nullptr) {
        dit_params_backend = sd_backend_cpu_init();
        printf("DiT weights: CPU-offload (graph-cut segmented compute on %s)\n", ggml_backend_name(backend));
    }

    std::shared_ptr<WAN_SHOTSTREAM::ShotStreamRunner> dit;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(dit_path, "model.diffusion_model.")) {
            printf("ERROR: init DiT loader %s\n", dit_path.c_str()); return 1;
        }
        dit = std::make_shared<WAN_SHOTSTREAM::ShotStreamRunner>(backend, dit_params_backend,
                                                                 loader.get_tensor_storage_map(), "model.diffusion_model");
        dit->set_flash_attention_enabled(getenv("SHOTSTREAM_NO_FLASH") == nullptr);
        dit->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        dit->get_param_tensors(tensors, "model.diffusion_model");
        if (!loader.load_tensors(tensors)) { printf("ERROR: load DiT tensors\n"); return 1; }
        dit->rng.seed(seed);
        printf("DiT loaded: %s (seed=%u)\n", dit->get_desc().c_str(), seed);
    }

    // ---- P1: OFFLINE block-0 self-attn op-parity dump ----
    // Feed the oracle's fixed block0_selfattn_input.npy through block-0's real causal
    // self-attn (shot-0 RoPE, empty caches) and emit block0_{selfattn_out,roped_k,v}.npy.
    if (!dump_block0_dir.empty()) {
        { std::string mk = "mkdir -p '" + dump_block0_dir + "'"; (void)system(mk.c_str()); }
        if (block0_input_npy.empty()) { printf("ERROR: --dump-block0 needs --block0-input <npy>\n"); return 1; }
        std::vector<float> xin; std::vector<int64_t> xshape;
        if (!read_npy_f32(block0_input_npy, xin, xshape)) { printf("ERROR: read %s\n", block0_input_npy.c_str()); return 1; }
        // golden [1,4680,1536] (C-order, dim fastest) -> ggml ne [1536,4680,1] (same memory).
        sd::Tensor<float> x_in({(int64_t)1536, (int64_t)4680, 1});
        GGML_ASSERT((int64_t)xin.size() == x_in.numel());
        std::memcpy(x_in.data(), xin.data(), sizeof(float) * xin.size());
        sd::Tensor<float> so, rk, v;
        if (!dit->dump_block0(n_threads, x_in, so, rk, v)) { printf("ERROR: block0 dump failed\n"); return 1; }
        // so ne[1536,4680,1] -> golden [1,4680,1536]; rk/v ne[128,12,4680] -> golden [1,4680,12,128]
        write_npy(dump_block0_dir + "/block0_selfattn_out.npy", so.data(), {1, 4680, 1536});
        write_npy(dump_block0_dir + "/block0_roped_k.npy", rk.data(), {1, 4680, 12, 128});
        write_npy(dump_block0_dir + "/block0_v.npy", v.data(), {1, 4680, 12, 128});
        printf("block-0 dump done -> %s\n", dump_block0_dir.c_str());
        return 0;
    }

    // Perf: cap the number of chunks per shot (per-chunk DiT + per-frame VAE scale ~linearly,
    // so a short 2-3 chunk run is enough to A/B levers without the full 7-chunk cost).
    if (chunks_override > 0) {
        dit->ss.frames_per_shot = chunks_override * dit->ss.num_frame_per_block;
        printf("[perf] chunks capped to %d (frames_per_shot=%d)\n", chunks_override, dit->ss.frames_per_shot);
    }

    // perf-ideas #8 (QUALITY-RISK): cap the LOCAL attention cache to a sliding window of
    // SHOTSTREAM_LOCAL_WINDOW chunks (default 7 = the shipped no-eviction config = full 21
    // latent frames). Smaller windows shrink L_k for the deep chunks (cuts the 37% attention)
    // but the model then forgets the earliest frames within a shot — motion-eye-test required.
    // The eviction machinery already exists (forward_block window clamp); this just un-pins it.
    if (const char* w = std::getenv("SHOTSTREAM_LOCAL_WINDOW")) {
        int wc = atoi(w);
        if (wc > 0) {
            dit->ss.local_attn_size = wc * dit->ss.num_frame_per_block;
            printf("[perf] LOCAL attention window capped to %d chunks (%d latent frames) — QUALITY-RISK, eye-test\n",
                   wc, dit->ss.local_attn_size);
        }
    }

    // The DiT is a LATENT-space model: --W/--H are PIXEL resolution; the DiT (+ its
    // causal KV caches / RoPE grid) operate on the Wan2.1 VAE latent (8x spatial, 16ch).
    const int VAE_SPATIAL = 8;
    int latW = W / VAE_SPATIAL, latH = H / VAE_SPATIAL;
    printf("resolution: %dx%d px -> %dx%d latent (16ch, /8 VAE)\n", W, H, latW, latH);

    // ------------------------------------------------------------------
    // 3) VAE (Wan2.1 3D VAE, encode+decode). Loaded ALONGSIDE the resident DiT so the
    //    multi-shot loop can round-trip history through PIXELS (§1.2/§4): decode shot k,
    //    then per-frame VAE-encode ≤6 sampled history pixels to build shot k+1's global
    //    cache — the reference path (NOT carrying latents directly). VAE f16 params are
    //    ~0.25 GB; the DiT compute buffer is freed before each decode so peak stays ~9 GB.
    // ------------------------------------------------------------------
    std::shared_ptr<WAN::WanVAERunner> vae;
    const bool have_vae = (!vae_path.empty() && !no_vae);
    // VRAM lever: with SHOTSTREAM_VAE_OFFLOAD=1 the VAE params live on a CPU backend and
    // stream to the GPU only for the encode/decode compute (freed again after). This
    // removes the ~0.24 GB of resident VAE weights from VRAM during the DiT generation
    // phase (the peak) — the VAE is idle there anyway. The per-decode H2D of 0.24 GB is
    // negligible against a ~50 s decode; the tile loop can keep it resident via
    // LONGCAT_VAE_KEEP_RESIDENT=1 (VRAM-neutral within the decode). Off by default.
    ggml_backend_t vae_params_backend = backend;
    static ggml_backend_t vae_cpu_backend = nullptr;
    if (getenv("SHOTSTREAM_VAE_OFFLOAD") != nullptr && on_gpu) {
        vae_cpu_backend = sd_backend_cpu_init();
        vae_params_backend = vae_cpu_backend;
    }
    if (have_vae) {
        printf("loading VAE '%s' (encode+decode)%s\n", vae_path.c_str(),
               vae_params_backend != backend ? " [params CPU-offload]" : "");
        vae = std::make_shared<WAN::WanVAERunner>(backend, vae_params_backend, String2TensorStorage{}, "",
                                                  /*decode_only=*/false, VERSION_WAN2);
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(vae_path, "vae.")) { printf("ERROR: VAE loader init\n"); return 1; }
        vae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        vae->get_param_tensors(tensors, "first_stage_model");
        if (!loader.load_tensors(tensors)) { printf("ERROR: VAE tensor load failed\n"); return 1; }
        vae->set_flash_attention_enabled(true);
        vae->set_temporal_tiling_enabled(vae_temporal_tiling);
        printf("VAE loaded (%zu tensors)\n", tensors.size());
    } else {
        printf("No --vae: chaining needs the VAE (history round-trip). Latents only, no chain.\n");
    }

    sd_tiling_params_t dec_tiling = {};
    dec_tiling.enabled         = vae_tiling;
    dec_tiling.temporal_tiling = vae_temporal_tiling;
    { const char* ovl = getenv("SHOTSTREAM_VAE_OVERLAP"); dec_tiling.target_overlap = ovl ? (float)atof(ovl) : 0.25f; }
    dec_tiling.rel_size_x      = vae_rel_x;
    dec_tiling.rel_size_y      = vae_rel_y;
    sd_tiling_params_t enc_tiling = {};
    enc_tiling.enabled         = true;           // single-frame encode, spatial tiles fit 12 GB
    enc_tiling.temporal_tiling = false;
    enc_tiling.target_overlap  = 0.25f;
    enc_tiling.rel_size_x      = 0.5f;
    enc_tiling.rel_size_y      = 0.5f;

    const int nctx = dit->ss.context_frames;  // 6 history latent frames

    // Write a decoded RGB video [W,H,T,3] -> PNGs -> mp4. Returns 0 on ffmpeg success.
    auto write_video = [&](const sd::Tensor<float>& rgb, const std::string& tag) {
        int64_t Wp = rgb.shape()[0], Hp = rgb.shape()[1], Tp = rgb.shape()[2];
        char framedir[512]; snprintf(framedir, sizeof(framedir), "%s/%s_frames", out_dir.c_str(), tag.c_str());
        { std::string mk = std::string("mkdir -p '") + framedir + "'"; (void)system(mk.c_str()); }
        const float* d = rgb.data();
        std::vector<unsigned char> frame((size_t)Wp * Hp * 3);
        for (int64_t t = 0; t < Tp; ++t) {
            for (int64_t y = 0; y < Hp; ++y)
                for (int64_t x = 0; x < Wp; ++x)
                    for (int c = 0; c < 3; ++c) {
                        float v = d[(((size_t)c * Tp + t) * Hp + y) * Wp + x];
                        v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                        frame[((size_t)y * Wp + x) * 3 + c] = (unsigned char)std::lround(v * 255.0f);
                    }
            char fp[600]; snprintf(fp, sizeof(fp), "%s/f%03lld.png", framedir, (long long)t);
            stbi_write_png(fp, (int)Wp, (int)Hp, 3, frame.data(), (int)Wp * 3);
        }
        char mp4[600]; snprintf(mp4, sizeof(mp4), "%s/%s.mp4", out_dir.c_str(), tag.c_str());
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -y -framerate %d -i '%s/f%%03d.png' -c:v libx264 -pix_fmt yuv420p "
                 "'%s' 2>'%s/ffmpeg_%s.log'",
                 fps, framedir, mp4, out_dir.c_str(), tag.c_str());
        int rc = system(cmd);
        printf("wrote %lld frames + ffmpeg rc=%d -> %s\n", (long long)Tp, rc, mp4);
    };

    // Build shot k>0's global/context latent [latW,latH,nctx,16] by sampling nctx pixel
    // frames evenly from a prior shot's decoded RGB and per-frame VAE-encoding them (the
    // reference pixel round-trip, §1.2/§4.1). Shared by the serve loop and the batch loop.
    auto build_history_ctx = [&](const sd::Tensor<float>& prev_px) -> sd::Tensor<float> {
        int64_t Tp = prev_px.shape()[2];
        sd::Tensor<float> ctx_lat;
        for (int j = 0; j < nctx; ++j) {
            int64_t fi = (nctx == 1) ? 0 : (int64_t)std::lround((double)j * (Tp - 1) / (nctx - 1));
            fi = std::min(std::max<int64_t>(0, fi), Tp - 1);
            auto frame_px = extract_frame(prev_px, fi);          // [W,H,1,3] [0,1]
            if (j == 0) dump_stats("  history frame_px[0]", frame_px);
            auto mu = vae->encode(n_threads, frame_px, enc_tiling, false, false);
            mu.reshape_({(int64_t)latW, (int64_t)latH, 1, 16});  // single-frame -> [W,H,1,16]
            if (j == 0) dump_stats("  history raw mu[0]", mu);
            vae_mu_to_diffusion(mu);                             // raw mu -> diffusion z
            ctx_lat = (j == 0) ? mu : sd::ops::concat(ctx_lat, mu, 2);
        }
        dump_stats("  history ctx_lat (diffusion)", ctx_lat);
        return ctx_lat;
    };

    // Multi-shot history (true-endless chain, §4.1 dynamic_sample_frames): distribute the ≤nctx
    // context frames across ALL prior shots 0..k-1 — base = nctx/k frames each, the remainder
    // going to the LAST shots (e.g. k=3 → [2,2,2]; k=4,nctx=6 → [1,1,2,2]). Within each prior
    // shot the frames are sampled evenly (linspace) over that shot's decoded pixels, and each
    // sampled frame is tagged (out_phases) with its SOURCE-shot index so prefill_context can
    // apply that frame's k·θ RoPE phase (§3.4). For a single prior shot (k=1, the 2-shot chain)
    // this reduces to counts=[nctx] over shot 0 with all-zero phases → the SAME nctx evenly
    // spaced frames and byte-identical RoPE as build_history_ctx. Returns [latW,latH,nctx,16].
    auto build_history_ctx_multi = [&](const std::vector<sd::Tensor<float>>& shots_px, int k,
                                       std::vector<int>& out_phases) -> sd::Tensor<float> {
        std::vector<int> counts(k, nctx / k);
        int rem = nctx % k;
        for (int i = 1; i <= rem; ++i) counts[k - i] += 1;  // remainder → last shots (ref §4.1)
        out_phases.clear();
        sd::Tensor<float> ctx_lat;
        bool first = true;
        for (int s = 0; s < k; ++s) {
            const auto& px = shots_px[s];
            int64_t Ts     = px.shape()[2];
            for (int j = 0; j < counts[s]; ++j) {
                // linspace(0, Ts-1, counts[s]).round() — count==1 → first frame (ref min-index).
                int64_t fi = (counts[s] == 1) ? 0
                                              : (int64_t)std::lround((double)j * (Ts - 1) / (counts[s] - 1));
                fi         = std::min(std::max<int64_t>(0, fi), Ts - 1);
                auto frame_px = extract_frame(px, fi);        // [W,H,1,3] [0,1]
                auto mu       = vae->encode(n_threads, frame_px, enc_tiling, false, false);
                mu.reshape_({(int64_t)latW, (int64_t)latH, 1, 16});
                vae_mu_to_diffusion(mu);                      // raw mu -> diffusion z
                ctx_lat = first ? mu : sd::ops::concat(ctx_lat, mu, 2);
                first   = false;
                out_phases.push_back(s);
            }
        }
        printf("  history: %d prior shots, counts=[", k);
        for (int s = 0; s < k; ++s) printf("%d%s", counts[s], s + 1 < k ? "," : "");
        printf("], per-frame source-shot phases=[");
        for (size_t i = 0; i < out_phases.size(); ++i) printf("%d%s", out_phases[i], i + 1 < out_phases.size() ? "," : "");
        printf("]\n");
        dump_stats("  history ctx_lat (diffusion, multi-shot)", ctx_lat);
        return ctx_lat;
    };

    // ==================================================================
    // P0 — WARM INTERACTIVE SERVING LOOP (--serve)
    //   Models (umT5 + DiT + VAE) are already resident. Loop: read one prompt line from
    //   stdin (one line = one chained shot; ':quit'/EOF exits), continue the rolling
    //   history from the prior shot's decoded pixels (global context cache), generate the
    //   next shot, and hand the latents to a BACKGROUND decode+write worker so the prompt
    //   is accepted again the moment the DiT finishes (P1 pipeline: the 67s VAE decode is
    //   hidden behind user think-time + the next prompt's CPU umT5 encode). The rolling
    //   history persists across prompts, so shot k+1 continues shot k's world.
    //
    //   Single-GPU note: shot k+1's history REQUIRES shot k's decoded pixels, so the decode
    //   and the next shot's GPU work are serialized by that data dependency — the decode
    //   cannot truly co-execute with the next DiT on one card. The realized overlap is
    //   decode(k) ‖ (stdin-wait + umT5 encode of prompt k+1). The user is never BLOCKED by
    //   the decode. SHOTSTREAM_SYNC_DECODE=1 forces the old inline-decode path for A/B.
    // ==================================================================
    if (serve) {
        if (!have_vae) { printf("ERROR: --serve requires --vae (history round-trip + decode)\n"); return 1; }
        if (!t5)       { printf("ERROR: --serve requires --t5xxl (per-prompt text encode)\n"); return 1; }
        const bool sync_decode = (getenv("SHOTSTREAM_SYNC_DECODE") != nullptr);
        printf("rope_shot_phase=%s  condition_start_frame=%d  frames_per_shot=%d  decode=%s\n",
               dit->rope_phase_enabled ? "ON(θ=1/6)" : "off", dit->ss.condition_start_frame,
               dit->ss.frames_per_shot, sync_decode ? "SYNC (inline)" : "PIPELINED (background)");

        // ---- background decode+write worker ----
        std::mutex mtx;
        std::condition_variable cv_job, cv_done;
        sd::Tensor<float> job_lat; int job_shot = -1;
        bool job_pending = false, worker_busy = false, shutdown = false;
        sd::Tensor<float> ready_pixels;  // most-recent decoded shot (next shot's history)
        double last_decode_secs = 0.0;

        std::thread worker([&]() {
            for (;;) {
                sd::Tensor<float> lat; int shot;
                {
                    std::unique_lock<std::mutex> lk(mtx);
                    cv_job.wait(lk, [&] { return job_pending || shutdown; });
                    if (shutdown && !job_pending) break;
                    lat  = std::move(job_lat);
                    shot = job_shot;
                    job_pending = false;
                    worker_busy = true;
                }
                // decode this shot's latents to pixels + write its mp4 (off the prompt path).
                int64_t td0 = ggml_time_ms();
                vae_diffusion_to_mu(lat);   // diffusion -> raw VAE space
                lat.unsqueeze_(4);          // [W,H,T,C,1]
                auto rgb = vae->decode(n_threads, lat, dec_tiling, /*decode_video=*/true, false, false);
                int64_t td1 = ggml_time_ms();
                char tag[64]; snprintf(tag, sizeof(tag), "shot%02d", shot);
                if (!rgb.empty() && rgb.dim() >= 4) write_video(rgb, tag);
                else printf("[WORKER] shot %d decode EMPTY\n", shot);
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    ready_pixels     = std::move(rgb);
                    last_decode_secs = (td1 - td0) / 1000.0;
                    worker_busy      = false;
                    cv_done.notify_all();
                }
                printf("[WORKER] shot %d decode+write done: %.1fs (background)\n", shot, (td1 - td0) / 1000.0);
            }
        });
        auto wait_worker_idle = [&]() {
            std::unique_lock<std::mutex> lk(mtx);
            cv_done.wait(lk, [&] { return !job_pending && !worker_busy; });
        };
        auto enqueue_decode = [&](sd::Tensor<float> lat, int shot) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_done.wait(lk, [&] { return !job_pending && !worker_busy; });  // drain any prior
                job_lat = std::move(lat); job_shot = shot; job_pending = true;
            }
            cv_job.notify_all();
        };

        printf("\n[SERVE] ready — models resident. One prompt line = one chained shot; ':quit' or EOF to exit.\n");
        int shot_idx = 0;
        int64_t serve_t0 = ggml_time_ms();
        int64_t first_shot_ready_ms = 0, first_shot_decoded_ms = 0;
        std::vector<std::string> shot_tags;
        std::string line;
        while (true) {
            printf("\nprompt> "); fflush(stdout);
            if (!std::getline(std::cin, line)) break;  // EOF
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
                line.pop_back();
            size_t bpos = line.find_first_not_of(" \t");
            if (bpos != std::string::npos) line = line.substr(bpos);
            if (line.empty()) continue;
            if (line == ":quit" || line == ":q" || line == ":exit") break;

            int64_t submit_ms = ggml_time_ms();
            printf("\n=== [SERVE] SHOT %d : \"%s\" ===\n", shot_idx, line.c_str());

            // 1) text conditioning — umT5 on CPU (overlaps any in-flight decode of shot k-1).
            //    Scene-level --global-prompt is prepended to each live line (global + shot,
            //    plain concat; empty global = byte-identical to the raw line).
            auto ctx = encode_prompt(global_prompt + line);

            // 2) global/context cache (rolling history persists across prompts).
            if (shot_idx == 0) {
                sd::Tensor<float> zero_ctx({(int64_t)latW, (int64_t)latH, (int64_t)nctx, 16});
                std::fill(zero_ctx.data(), zero_ctx.data() + zero_ctx.numel(), 0.0f);
                dit->prefill_context(n_threads, std::vector<int>(nctx, 0), zero_ctx, ctx);
                dit->free_compute_buffer();
            } else {
                wait_worker_idle();  // ensure shot k-1 decoded (ready_pixels valid) + VAE idle
                auto history_ctx = build_history_ctx(ready_pixels);  // vae->encode (GPU, worker idle)
                printf("[SERVE] history: encoded %d frames from shot %d -> context [%d,%d,%d,16]\n",
                       nctx, shot_idx - 1, latW, latH, nctx);
                // serve keeps only the immediately-prior shot's pixels → single-source history,
                // so all nctx frames carry shot (shot_idx-1)'s uniform phase (per-frame vector of
                // one value). The batch loop below does the full multi-source distribution.
                dit->prefill_context(n_threads, std::vector<int>(nctx, shot_idx - 1), history_ctx, ctx);
                dit->free_compute_buffer();
            }

            // 3) generate this shot (DiT, on the prompt path).
            int64_t tg0 = ggml_time_ms();
            auto lat = dit->run_shot(n_threads, shot_idx, latW, latH, ctx);
            int64_t tg1 = ggml_time_ms();
            if (lat.empty()) { printf("ERROR: shot %d produced no latent\n", shot_idx); break; }
            dit->free_compute_buffer();  // release DiT compute before the VAE decode buffer
            double dit_secs = (tg1 - tg0) / 1000.0;

            // 4) hand latents to the decode worker; the prompt is accepted again NOW.
            char tag[64]; snprintf(tag, sizeof(tag), "shot%02d", shot_idx);
            shot_tags.emplace_back(tag);
            enqueue_decode(std::move(lat), shot_idx);
            if (sync_decode) wait_worker_idle();  // A/B: block on decode like the old serial path
            int64_t ready_ms = ggml_time_ms();
            if (shot_idx == 0) first_shot_ready_ms = ready_ms;
            printf("[SERVE] shot %d: DiT %.1fs | prompt accepted again at +%.1fs from submit%s\n",
                   shot_idx, dit_secs, (ready_ms - submit_ms) / 1000.0,
                   sync_decode ? " (SYNC: blocked on decode)" : " (decode backgrounded)");
            shot_idx++;
        }

        // drain the final decode, then stitch all shots into the eye-test chain.
        printf("\n[SERVE] draining final decode...\n");
        wait_worker_idle();
        if (first_shot_ready_ms) first_shot_decoded_ms = ggml_time_ms();  // approx (only meaningful for 1-shot)
        { std::lock_guard<std::mutex> lk(mtx); shutdown = true; }
        cv_job.notify_all();
        worker.join();

        if (shot_idx >= 2) {
            std::string listp = out_dir + "/serve_concat.txt";
            FILE* lf = fopen(listp.c_str(), "w");
            if (lf) {
                for (auto& t : shot_tags) fprintf(lf, "file '%s.mp4'\n", t.c_str());
                fclose(lf);
                char cmd[1024];
                snprintf(cmd, sizeof(cmd),
                         "ffmpeg -y -f concat -safe 0 -i '%s' -c copy '%s/serve_chain.mp4' 2>'%s/ffmpeg_serve_chain.log'",
                         listp.c_str(), out_dir.c_str(), out_dir.c_str());
                int rc = system(cmd);
                printf("[SERVE] stitched %d shots -> %s/serve_chain.mp4 (ffmpeg rc=%d)\n", shot_idx, out_dir.c_str(), rc);
            }
        }
        printf("[SERVE] done: %d shots, total serve wall %.1fs, time-to-first-shot-ready %.1fs (decode +%.1fs)\n",
               shot_idx, (ggml_time_ms() - serve_t0) / 1000.0,
               first_shot_ready_ms ? (first_shot_ready_ms - serve_t0) / 1000.0 : 0.0,
               (first_shot_decoded_ms && shot_idx == 1) ? (first_shot_decoded_ms - first_shot_ready_ms) / 1000.0 : last_decode_secs);

        dit->free_params_buffer();
        dit->free_compute_buffer();
        dit.reset();
        if (t5) { t5->model.free_params_buffer(); t5.reset(); }
        return 0;
    }

    // ------------------------------------------------------------------
    // MULTI-SHOT STREAMING LOOP (the headline feature). Per shot: build the global cache
    // (zeroed history for shot 0; per-frame re-encode of sampled prior-shot PIXELS for
    // k>0), generate 21 latent frames, decode to pixels, keep pixels as the next shot's
    // history. RoPE θ=1/6 shot-phase decouples shots (generated frames phase k; context
    // frames phase of their SOURCE shot). See REFERENCE §3.4/§4/§6.
    // ------------------------------------------------------------------
    std::vector<sd::Tensor<float>> shot_pixels;  // decoded RGB per shot, [W,H,T,3] in [0,1]
    const bool zero_ctx_shot0 = (getenv("SHOTSTREAM_NO_ZERO_CTX_SHOT0") == nullptr);
    printf("rope_shot_phase=%s  zero_ctx_shot0=%s  condition_start_frame=%d\n",
           dit->rope_phase_enabled ? "ON(θ=1/6)" : "off",
           zero_ctx_shot0 ? "yes" : "no", dit->ss.condition_start_frame);

    printf("per-shot captions: %zu prompt(s) for %d shot(s) (reuse-last when fewer)\n", contexts.size(), shots);
    for (int k = 0; k < shots; ++k) {
        int cap_idx = (k < (int)contexts.size()) ? k : (int)contexts.size() - 1;
        printf("\n=== SHOT %d/%d (caption[%d]) ===\n", k, shots, cap_idx);

        // ---- build the GLOBAL/context cache for this shot ----
        if (k == 0) {
            if (zero_ctx_shot0 && have_vae) {
                // Faithful shot-0 path: prefill the global cache from ZEROED context latents
                // (6 frames, positions 0..5, phase 0) — §6 runs context_insert for k=0 too.
                sd::Tensor<float> zero_ctx({(int64_t)latW, (int64_t)latH, (int64_t)nctx, 16});
                std::fill(zero_ctx.data(), zero_ctx.data() + zero_ctx.numel(), 0.0f);
                dit->prefill_context(n_threads, std::vector<int>(nctx, 0), zero_ctx, context_for_shot(k));
                dit->free_compute_buffer();
            } else {
                dit->reset_global_cache();  // known-good empty-cache path (A/B fallback)
            }
        } else if (getenv("SHOTSTREAM_NO_CONTEXT") != nullptr) {
            // ---- ISOLATION RENDER B: empty global cache (no history conditioning) ----
            // If shot 1 turns clean+photoreal with this on, the bug is in the context/history
            // path (VAE encode / cache), NOT the RoPE phase.
            printf("[ISOLATION B] SHOTSTREAM_NO_CONTEXT: shot %d gets an EMPTY global cache\n", k);
            dit->reset_global_cache();
        } else {
            // ---- history round-trip through PIXELS across ALL prior shots (§1.2/§3.4/§4.1) ----
            // True-endless chain: the ≤6 context frames are distributed over shots 0..k-1, each
            // tagged with its SOURCE-shot index so prefill_context applies that frame's k·θ RoPE
            // phase. (2-shot: k=1 → all 6 from shot 0, phases all 0 = byte-identical to before.)
            std::vector<int> ctx_phases;
            auto ctx_lat = build_history_ctx_multi(shot_pixels, k, ctx_phases);
            printf("history: encoded %d frames across %d prior shots -> context latent [%d,%d,%d,16]\n",
                   (int)ctx_phases.size(), k, latW, latH, nctx);
            // multi_caption=False (§4.3): shot k's context prefill uses shot k's OWN caption.
            // Per-frame multi-caption gather (context frames attend their SOURCE shot's caption)
            // is a follow-up (see IMPL-NOTES "Per-shot prompts" TODO).
            dit->prefill_context(n_threads, ctx_phases, ctx_lat, context_for_shot(k));
            dit->free_compute_buffer();
        }

        // ---- generate this shot (with shot k's OWN caption context) ----
        int64_t t0 = ggml_time_ms();
        auto lat = dit->run_shot(n_threads, k, latW, latH, context_for_shot(k));
        int64_t t1 = ggml_time_ms();
        if (lat.empty()) { printf("ERROR: shot %d produced no latent\n", k); return 1; }
        printf("shot %d: %d latent frames in %.1fs\n", k, (int)lat.shape()[2], (t1 - t0) / 1000.0);
        dump_stats("shot latent", lat);
        { char p[256]; snprintf(p, sizeof(p), "%s/shot%02d_latent.bin", out_dir.c_str(), k); write_bin(p, lat, "shot_latent"); }
        dit->free_compute_buffer();  // release DiT compute before the VAE decode buffer
        // VRAM: this shot's resident KV caches (local ~21f + global ~6f, F16, ~6-8 GB on the
        // 3060) are DEAD during the VAE decode and are fully rebuilt next shot (prefill_context
        // rebuilds the global cache from the decoded-pixel round-trip; run_shot resets the local
        // cache). Freeing them here lets the ~3.5 GB VAE compute buffer fit beside the resident
        // DiT weights on the 12 GB 3060 — the at-the-edge chunks=7 chain OOM'd here otherwise
        // (the resident-KV + VAE-decode buffers together exceed 12 GB). Env opt-out for A/B.
        if (getenv("SHOTSTREAM_KEEP_KV_FOR_DECODE") == nullptr) {
            dit->reset_local_cache();
            dit->reset_global_cache();
        }

        if (!have_vae) { continue; }  // latents only

        // ---- decode this shot to pixels (kept as next shot's history) ----
        auto lat_vae = lat;
        vae_diffusion_to_mu(lat_vae);   // diffusion -> raw VAE space
        lat_vae.unsqueeze_(4);          // [W,H,T,C,1] so WanVAE reads ne[2]=T, ne[3]=C
        int64_t tdec0 = ggml_time_ms();
        auto rgb = vae->decode(n_threads, lat_vae, dec_tiling, /*decode_video=*/true, false, false);
        int64_t tdec1 = ggml_time_ms();
        printf("shot %d decode: %.1fs\n", k, (tdec1 - tdec0) / 1000.0);
        dump_stats("decoded rgb", rgb);
        if (rgb.empty() || rgb.dim() < 4) { printf("ERROR: shot %d decode empty\n", k); return 1; }
        char tag[64]; snprintf(tag, sizeof(tag), "shot%02d", k);
        write_video(rgb, tag);
        shot_pixels.push_back(std::move(rgb));
    }

    // ---- stitch all shots into one continuous chain mp4 (the eye-test artifact) ----
    if (have_vae && (int)shot_pixels.size() >= 2) {
        int64_t Wp = shot_pixels[0].shape()[0], Hp = shot_pixels[0].shape()[1];
        int64_t Ttot = 0; for (auto& s : shot_pixels) Ttot += s.shape()[2];
        sd::Tensor<float> chain({Wp, Hp, Ttot, 3});
        float* cd = chain.data(); int64_t toff = 0;
        for (auto& s : shot_pixels) {
            int64_t Ts = s.shape()[2]; const float* sd_ = s.data();
            for (int c = 0; c < 3; ++c)
                for (int64_t t = 0; t < Ts; ++t)
                    std::memcpy(cd + ((size_t)c * Ttot + (toff + t)) * Wp * Hp,
                                sd_ + ((size_t)c * Ts + t) * Wp * Hp, sizeof(float) * (size_t)(Wp * Hp));
            toff += Ts;
        }
        printf("\n=== stitching %zu shots -> chain (%lld frames) ===\n", shot_pixels.size(), (long long)Ttot);
        char chain_tag[32]; snprintf(chain_tag, sizeof(chain_tag), "chain_%zushot", shot_pixels.size());
        write_video(chain, chain_tag);
    }

    dit->free_params_buffer();
    dit->free_compute_buffer();
    dit.reset();
    printf("\nShotStream done: %d shots.\n", shots);
    return 0;
}
