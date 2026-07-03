// InfiniteTalk (MultiTalk on Wan2.1-I2V-14B) CLI — promptable lip-sync dub.
//
// Wan2.2-I2V-A14B generates the silent promptable shots; THIS tool dubs the user's
// song onto the singing character. Two conditioning modes:
//   --image <png>  I2V: every window anchors to the SAME static image; the model invents
//                  all motion from the audio + prompt (talking-head style).
//   --video <mp4>  V2V (sparse-frame dubbing): every window re-anchors to the SOURCE clip's
//                  frame at the window's start index, so the source's identity / scene /
//                  camera / pose are preserved while the mouth (and in-between motion) are
//                  driven by the audio. This is InfiniteTalk's sparse-frame mechanism —
//                  upstream multitalk.py does `cond_image = extract_specific_frames(
//                  cond_file, audio_start_idx)` per streaming window (ONE source keyframe
//                  per ~3s window; mask=1 at latent-frame 0 only). Use V2V to lip-sync a
//                  Wan-rendered clip while keeping its motion.
// Loads the merged InfiniteTalk DiT (base Wan2.1-I2V-14B + audio graft + lightx2v distill,
// q4_K), chinese-wav2vec2-base, the Wan2.1 16ch VAE, umT5-XXL, and (optionally) CLIP-H/14
// vision, then runs the non-causal motion-frame streaming sampler.
//
// Streaming (multitalk.py generate_infinitetalk): 81-frame (4n+1) windows; per window
//   the anchor cond frame (static image OR source-video frame @ audio_start_idx) is
//   VAE-encoded + masked into a 20ch c_concat and CLIP-H'd into 257 image tokens; the
//   song's per-frame wav2vec stack is windowed (±2) -> AudioProjModel -> 32 tokens/
//   latent-frame; sampling pins the first `motion_lat` latent frames to the carried-over
//   clean motion latents each step. Windows overlap by frame_num-motion_frame (the first
//   motion_frame output frames of windows>0 are dropped).
//
// Usage:
//   sd-infinitetalk --dit <it.gguf> --wav2vec <w2v.gguf> --vae <vae.gguf> --umt5 <umt5.gguf>
//                   [--clip-vision <clip.pth>] (--image <png> | --video <mp4>)
//                   --prompt "<text>" --wav <song.wav>
//                   [--out <dir>] [--frames 81] [--height H] [--width W] [--steps 4]
//                   [--shift 5] [--text-cfg 1] [--audio-cfg 1] [--motion-frame 9]
//                   [--max-windows N] [--fps 25] [--cpu] [--distilled] [--load-only]

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model.h"
#include "stable-diffusion.h"
#include "util.h"  // clip_preprocess
#include "model/te/t5.hpp"
#include "tensor.hpp"
#include "tensor_ggml.hpp"
#include "model/vae/wan_vae.hpp"
#include "conditioning/conditioner.hpp"  // FrozenCLIPVisionEmbedder
#include "infinitetalk.hpp"
#include "wav2vec2.hpp"

static void log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
}

static void dump_stats(const char* tag, const sd::Tensor<float>& t) {
    if (t.empty()) { printf("%-22s EMPTY\n", tag); return; }
    double sum = 0, sq = 0, mn = 1e30, mx = -1e30;
    for (int64_t i = 0; i < t.numel(); ++i) {
        double v = t.data()[i];
        sum += v; sq += v * v;
        if (v < mn) mn = v; if (v > mx) mx = v;
    }
    double mean = sum / (double)t.numel();
    double var  = sq / (double)t.numel() - mean * mean;
    printf("%-22s shape=%s mean=%.5f std=%.5f min=%.4f max=%.4f\n",
           tag, sd::tensor_shape_to_string(t.shape()).c_str(), mean,
           var > 0 ? sqrt(var) : 0.0, mn, mx);
}

// Per-frame stats of a [W,H,T,C] tensor (ne: idx = c*(W*H*T) + t*(W*H) + i). Reveals whether
// generated frames sit in a different distribution than the (pinned) anchor frame 0.
static void dump_per_frame(const char* tag, const sd::Tensor<float>& t) {
    if (t.empty() || t.dim() < 4) { printf("%-18s (empty/ndim<4)\n", tag); return; }
    int64_t W = t.shape()[0], H = t.shape()[1], T = t.shape()[2], C = t.shape()[3];
    int64_t plane = W * H, chan = W * H * T;
    const float* d = t.data();
    for (int64_t tt = 0; tt < T; ++tt) {
        double s = 0, s2 = 0; int64_t n = 0;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t i = 0; i < plane; ++i) { double v = d[c * chan + tt * plane + i]; s += v; s2 += v * v; ++n; }
        double m = s / n, var = s2 / n - m * m;
        printf("  %-16s f%lld: mean=%+.4f std=%.4f\n", tag, (long long)tt, m, var > 0 ? sqrt(var) : 0.0);
    }
}

static std::string opt(int argc, char** argv, const std::string& flag, const std::string& def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return def;
}
static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (flag == argv[i]) return true;
    return false;
}

// Wan2.1 (16ch) VAE latent normalization (vae2_1.py). raw mu <-> diffusion space.
static const float WAN_VAE_MEAN[16] = {
    -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
    0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
static const float WAN_VAE_STD[16] = {
    2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
    3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

// in/out latents ggml ne [W,H,T,16]; per-channel (ne[3]) normalize.
static void vae_mu_to_diffusion(sd::Tensor<float>& t) {
    int64_t W = t.shape()[0], H = t.shape()[1], T = t.shape()[2], C = t.shape()[3];
    GGML_ASSERT(C == 16);
    float* d = t.data();
    for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < W * H * T; ++i)
            d[c * W * H * T + i] = (d[c * W * H * T + i] - WAN_VAE_MEAN[c]) / WAN_VAE_STD[c];
}
static void vae_diffusion_to_mu(sd::Tensor<float>& t) {
    int64_t W = t.shape()[0], H = t.shape()[1], T = t.shape()[2], C = t.shape()[3];
    GGML_ASSERT(C == 16);
    float* d = t.data();
    for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < W * H * T; ++i)
            d[c * W * H * T + i] = d[c * W * H * T + i] * WAN_VAE_STD[c] + WAN_VAE_MEAN[c];
}

// flow-match sigma schedule (shift). sigma 1->0; reparam sigma' = shift*s/(1+(shift-1)*s);
static std::vector<float> flow_sigmas(int steps, float shift) {
    std::vector<float> s(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        float sigma = 1.0f - (float)i / (float)steps;
        s[i] = shift * sigma / (1.0f + (shift - 1.0f) * sigma);
    }
    return s;
}
// lightx2v distilled euler (diffusers FlowMatchEulerDiscreteScheduler.set_timesteps(N), shift).
static std::vector<float> distilled_sigmas(int steps, float shift) {
    const float sigma_max = 1.0f, sigma_min = 0.003f / 1.002f;
    std::vector<float> s(steps + 1);
    for (int i = 0; i < steps; ++i) {
        float sigma = (steps == 1) ? sigma_max
                                   : sigma_max + (sigma_min - sigma_max) * (float)i / (float)(steps - 1);
        s[i] = shift * sigma / (1.0f + (shift - 1.0f) * sigma);
    }
    s[steps] = 0.0f;
    return s;
}

// Load image -> RGB[0,1], short-side resize + center-crop to WxH, ggml ne [W,H,1,3].
static bool load_image(const std::string& path, int W, int H, sd::Tensor<float>& out) {
    int iw, ih, ic;
    unsigned char* img = stbi_load(path.c_str(), &iw, &ih, &ic, 3);
    if (!img) { printf("ERROR: stbi_load %s failed\n", path.c_str()); return false; }
    int target_short = std::min(W, H);
    double scale = (double)target_short / (double)std::min(iw, ih);
    int rw = std::max((int)std::round(iw * scale), W), rh = std::max((int)std::round(ih * scale), H);
    std::vector<unsigned char> resized((size_t)rw * rh * 3);
    stbir_resize_uint8(img, iw, ih, 0, resized.data(), rw, rh, 0, 3);
    stbi_image_free(img);
    int ox = (rw - W) / 2, oy = (rh - H) / 2;
    out = sd::Tensor<float>({(int64_t)W, (int64_t)H, 1, 3});
    float* d = out.data();
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                unsigned char v = resized[((size_t)(oy + y) * rw + (ox + x)) * 3 + c];
                d[((size_t)c * H + y) * W + x] = (float)v / 255.0f;
            }
    return true;
}

// Build full_audio_emb [audio_dim, blocks(12), T_video] from the song: wav2vec2 12-layer
// stack (hidden[1:13]) interpolated 50fps -> video fps.
static sd::Tensor<float> build_full_audio_emb(WAV2VEC2::Wav2Vec2EncoderRunner* w2v, int n_threads,
                                              const std::vector<float>& wav16k, int fps,
                                              const IT::InfiniteTalkConfig& cfg) {
    auto wav_n = WAV2VEC2::normalize_waveform(wav16k);
    sd::Tensor<float> wav_t({(int64_t)wav_n.size(), 1, 1});
    memcpy(wav_t.data(), wav_n.data(), wav_n.size() * sizeof(float));
    auto hs = w2v->compute(n_threads, wav_t);  // [768, T50, 13]
    dump_stats("wav2vec2_hs", hs);
    int64_t d_model = hs.shape()[0], T50 = hs.shape()[1], n_hs = hs.shape()[2];

    int T_video = std::max(1, (int)std::llround((double)T50 / 50.0 * (double)fps));
    int out_len = 0;
    auto interp = WAV2VEC2::interp_50_to_30(hs, /*video_length=*/T_video, out_len);  // [n_hs][out_len][d_model]
    // take the 12 transformer layers (skip hidden 0 = input embeds) into [d_model, 12, T_video].
    int blocks = (int)cfg.blocks_per_token;  // 12
    sd::Tensor<float> full({d_model, (int64_t)blocks, (int64_t)out_len});
    float* fd = full.data();
    for (int f = 0; f < out_len; ++f)
        for (int b = 0; b < blocks; ++b) {
            int l = b + 1;  // hidden[1:13]
            if (l >= n_hs) l = (int)n_hs - 1;
            for (int64_t c = 0; c < d_model; ++c)
                fd[((int64_t)f * blocks + b) * d_model + c] = interp[((size_t)l * out_len + f) * d_model + c];
        }
    printf("full_audio_emb: T50=%lld -> T_video=%d (fps=%d), %lld layers stacked\n",
           (long long)T50, out_len, fps, (long long)blocks);
    return full;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    sd_set_log_callback(log_cb, nullptr);

    if (argc < 2 || has_flag(argc, argv, "--help")) {
        printf("usage: %s --dit <it.gguf> --wav2vec <w2v.gguf> --vae <vae.gguf> --umt5 <umt5.gguf>\n", argv[0]);
        printf("          [--clip-vision <clip.pth>] (--image <png> | --video <mp4>) --prompt \"<text>\" --wav <song.wav>\n");
        printf("          [--out <dir>] [--frames 81] [--height H] [--width W] [--steps 4] [--shift 5]\n");
        printf("          [--text-cfg 1] [--audio-cfg 1] [--motion-frame 9] [--max-windows N] [--fps 25]\n");
        printf("          [--cpu] [--distilled] [--load-only] [--clip-fea <bin>]\n");
        return 1;
    }

    std::string dit_path  = opt(argc, argv, "--dit");
    std::string w2v_path  = opt(argc, argv, "--wav2vec");
    std::string vae_path  = opt(argc, argv, "--vae");
    std::string umt5_path = opt(argc, argv, "--umt5");
    std::string clip_path = opt(argc, argv, "--clip-vision");
    std::string clipfea_bin = opt(argc, argv, "--clip-fea");  // precomputed [1280,257] fallback
    std::string image_path = opt(argc, argv, "--image");
    std::string video_path = opt(argc, argv, "--video");  // V2V source clip (per-window sparse anchor)
    std::string prompt    = opt(argc, argv, "--prompt", "a person singing");
    std::string n_prompt  = opt(argc, argv, "--neg-prompt", "");
    std::string wav_path  = opt(argc, argv, "--wav");
    std::string out_dir   = opt(argc, argv, "--out", "./it_out");

    int   frames = atoi(opt(argc, argv, "--frames", "81").c_str());
    int   height = atoi(opt(argc, argv, "--height", "480").c_str());
    int   width  = atoi(opt(argc, argv, "--width", "480").c_str());
    int   steps  = atoi(opt(argc, argv, "--steps", "4").c_str());
    float shift  = atof(opt(argc, argv, "--shift", "5.0").c_str());
    float text_cfg  = atof(opt(argc, argv, "--text-cfg", "1.0").c_str());
    float audio_cfg = atof(opt(argc, argv, "--audio-cfg", "1.0").c_str());
    int   motion_frame = atoi(opt(argc, argv, "--motion-frame", "9").c_str());
    int   max_windows  = atoi(opt(argc, argv, "--max-windows", "0").c_str());  // 0 = until audio runs out
    int   fps    = atoi(opt(argc, argv, "--fps", "25").c_str());
    bool  cpu    = has_flag(argc, argv, "--cpu");
    bool  load_only = has_flag(argc, argv, "--load-only");
    bool  distilled = has_flag(argc, argv, "--distilled");
    if (distilled) { if (!has_flag(argc, argv, "--shift")) shift = 5.0f; }

    if (cpu) setenv("LONGCAT_NO_FUSED_ROPE", "1", 0);
    if (!out_dir.empty()) { std::string mk = "mkdir -p '" + out_dir + "'"; (void)system(mk.c_str()); }

    IT::InfiniteTalkConfig cfg;
    int lat_h = height / 8, lat_w = width / 8;
    int lat_t = (frames - 1) / cfg.vae_scale + 1;  // 21 for frames=81
    int motion_lat = 1 + (motion_frame - 1) / cfg.vae_scale;  // 3 for motion_frame=9

    ggml_backend_t backend = nullptr;
    ggml_backend_load_all();
    if (!cpu) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (backend) printf("backend: GPU (%s)\n", ggml_backend_name(backend));
    }
    if (!backend) { backend = sd_backend_cpu_init(); printf("backend: CPU\n"); }
    ggml_backend_t audio_backend = sd_backend_cpu_init();
    int n_threads = 8;

    // ---- DiT (CPU-offload streaming on GPU; resident on CPU) ----
    if (dit_path.empty()) { printf("ERROR: --dit required\n"); return 1; }
    ggml_backend_t dit_params_backend = backend;
    bool dit_offload = !cpu && getenv("IT_NO_OFFLOAD") == nullptr;
    if (dit_offload) { dit_params_backend = sd_backend_cpu_init(); printf("DiT weights: CPU-offload\n"); }
    float max_vram_gib = 6.5f;
    if (const char* mv = getenv("IT_MAX_VRAM_GIB")) max_vram_gib = atof(mv);

    printf("loading InfiniteTalk DiT '%s'\n", dit_path.c_str());
    std::shared_ptr<IT::InfiniteTalkRunner> dit;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(dit_path, "model.diffusion_model.")) {
            printf("ERROR: init DiT loader\n"); return 1;
        }
        dit = std::make_shared<IT::InfiniteTalkRunner>(backend, dit_params_backend, loader.get_tensor_storage_map(), "model.diffusion_model");
        dit->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        dit->get_param_tensors(tensors, "model.diffusion_model");
        if (!loader.load_tensors(tensors)) { printf("ERROR: load DiT tensors\n"); return 1; }
        dit->set_flash_attention_enabled(getenv("IT_NO_FLASH") == nullptr);
        if (dit_offload) dit->set_max_graph_vram_bytes((size_t)(max_vram_gib * 1024.0 * 1024.0 * 1024.0));
        printf("DiT loaded (%zu tensors)\n", tensors.size());
    }

    // ---- wav2vec2 (base) ----
    std::shared_ptr<WAV2VEC2::Wav2Vec2EncoderRunner> w2v;
    if (!w2v_path.empty()) {
        printf("loading wav2vec2 '%s'\n", w2v_path.c_str());
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(w2v_path, "")) { printf("ERROR: w2v loader\n"); return 1; }
        w2v = std::make_shared<WAV2VEC2::Wav2Vec2EncoderRunner>(audio_backend, audio_backend, loader.get_tensor_storage_map(),
                                                                "audio_encoder", WAV2VEC2::Wav2Vec2Params::base());
        w2v->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        w2v->get_param_tensors(tensors, "audio_encoder");
        if (!loader.load_tensors(tensors)) { printf("WARN: some wav2vec2 tensors failed\n"); }
        printf("wav2vec2 loaded (%zu tensors)\n", tensors.size());
    }

    // ---- Wan2.1 VAE ----
    std::shared_ptr<WAN::WanVAERunner> vae;
    if (!vae_path.empty()) {
        printf("loading VAE '%s'\n", vae_path.c_str());
        vae = std::make_shared<WAN::WanVAERunner>(backend, backend, String2TensorStorage{}, "", false, VERSION_WAN2);
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(vae_path, "vae.")) { printf("ERROR: VAE loader\n"); return 1; }
        vae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        vae->get_param_tensors(tensors, "first_stage_model");
        if (!loader.load_tensors(tensors)) { printf("ERROR: VAE load\n"); return 1; }
        vae->set_flash_attention_enabled(true);
        printf("VAE loaded (%zu tensors)\n", tensors.size());
    }

    // ---- CLIP-H vision (optional) ----
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;
    if (!clip_path.empty()) {
        printf("loading CLIP-H vision '%s'\n", clip_path.c_str());
        ModelLoader loader;
        // convert_name so the raw open-clip-xlm-roberta .pth names map to the
        // cond_stage_model.transformer.* scheme CLIPVisionModelProjection expects.
        if (!loader.init_from_file_and_convert_name(clip_path, "cond_stage_model.transformer.")) {
            printf("WARN: CLIP loader init failed; clip_fea will be zeros\n");
        } else {
            clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, backend, loader.get_tensor_storage_map());
            clip_vision->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            clip_vision->get_param_tensors(tensors);
            if (!loader.load_tensors(tensors)) { printf("WARN: some CLIP tensors failed\n"); }
            clip_vision->set_flash_attention_enabled(true);
            printf("CLIP-H loaded (%zu tensors)\n", tensors.size());
        }
    }

    // ---- umT5 context (positive + negative) ----
    auto encode_text = [&](const std::string& text) -> sd::Tensor<float> {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(umt5_path, "text_encoders.t5xxl.transformer.")) {
            printf("ERROR: umT5 loader\n"); return {};
        }
        auto t5 = std::make_shared<T5Embedder>(audio_backend, audio_backend, loader.get_tensor_storage_map(), "text_encoders.t5xxl.transformer", true);
        t5->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        loader.load_tensors(tensors);
        auto tw  = t5->tokenize(text, 512, false);
        auto ids = sd::Tensor<int32_t>::from_vector(std::get<0>(tw));
        auto am  = sd::Tensor<float>::from_vector(std::get<2>(tw));
        auto ctx = t5->model.compute(n_threads, ids, am);  // [4096, n_tok, 1]
        int64_t tok = ctx.shape()[1];
        const int64_t TEXT_LEN = 512;
        if (tok < TEXT_LEN) {
            sd::Tensor<float> pad({ctx.shape()[0], TEXT_LEN, ctx.shape()[2]});
            std::fill(pad.data(), pad.data() + pad.numel(), 0.0f);
            sd::ops::slice_assign(&pad, 1, 0, tok, ctx);
            ctx = std::move(pad);
        }
        return ctx;
    };
    sd::Tensor<float> context, context_null;
    if (!umt5_path.empty()) {
        printf("encoding prompt: \"%s\"\n", prompt.c_str());
        context = encode_text(prompt);
        bool need_null = !(text_cfg == 1.0f);
        context_null = need_null ? encode_text(n_prompt.empty() ? " " : n_prompt) : context;
        dump_stats("context", context);
    } else {
        context = sd::Tensor<float>({4096, 512, 1});
        context_null = context;
    }

    if (load_only) { printf("\n--load-only: all GGUFs loaded. exiting.\n"); return 0; }

    // ---- song -> full_audio_emb ----
    sd::Tensor<float> full_audio;
    if (w2v && !wav_path.empty()) {
        std::vector<float> wav;
        if (!LONGCAT_AUDIO::load_wav_16k_mono(wav_path, wav)) { printf("ERROR: load wav\n"); return 1; }
        full_audio = build_full_audio_emb(w2v.get(), n_threads, wav, fps, cfg);
        dump_stats("full_audio_emb", full_audio);
    } else {
        printf("WARN: no audio (need --wav2vec --wav); audio tokens = zeros\n");
        full_audio = sd::Tensor<float>({cfg.audio_dim, (int64_t)cfg.blocks_per_token, (int64_t)(frames * 4)});
    }
    int T_video = (int)full_audio.shape()[2];

    // ---- conditioning anchor source: V2V (--video) or I2V (--image) ----
    // V2V = InfiniteTalk sparse-frame dubbing: each streaming window re-anchors to the
    // SOURCE clip's frame at the window's start index (audio_start_idx). Upstream
    // multitalk.py: `cond_image = extract_specific_frames(cond_file, audio_start_idx)` per
    // window, mask=1 at latent-frame 0 only. The source frames (NOT the generated tail)
    // carry identity/scene/camera; the model regenerates in-between motion from the audio.
    // We replicate that: extract the source clip to PNG frames at the OUTPUT fps once, then
    // re-fetch the anchor per window. (motion-frame continuity still comes from generated
    // frames, exactly as upstream — so one sparse source keyframe per ~3s window.)
    bool v2v = !video_path.empty();
    std::string src_dir;
    int n_src_frames = 0;
    if (v2v) {
        src_dir = out_dir + "/_src";
        { std::string mk = "mkdir -p '" + src_dir + "'"; (void)system(mk.c_str()); }
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -y -i '%s' -vf fps=%d '%s/src_%%05d.png' >%s/ffmpeg_extract.log 2>&1",
                 video_path.c_str(), fps, src_dir.c_str(), out_dir.c_str());
        printf("V2V: extracting source frames @ %dfps from %s\n", fps, video_path.c_str());
        if (system(cmd) != 0) { printf("ERROR: ffmpeg frame-extract failed (see %s/ffmpeg_extract.log)\n", out_dir.c_str()); return 1; }
        for (int i = 1;; ++i) {  // ffmpeg numbers from 1
            char fp[512]; snprintf(fp, sizeof(fp), "%s/src_%05d.png", src_dir.c_str(), i);
            FILE* f = fopen(fp, "rb"); if (!f) break; fclose(f); n_src_frames = i;
        }
        if (n_src_frames == 0) { printf("ERROR: no source frames extracted from %s\n", video_path.c_str()); return 1; }
        printf("V2V: %d source frames extracted (audio timeline = %d frames @ %dfps)\n", n_src_frames, T_video, fps);
        if (n_src_frames < T_video)
            printf("V2V WARN: source clip (%d f) shorter than audio (%d f); holding last source frame past its end\n",
                   n_src_frames, T_video);
    } else if (image_path.empty()) {
        printf("ERROR: --image or --video required\n"); return 1;
    }

    // Fetch the conditioning anchor pixel frame for an absolute output-frame index.
    //   I2V: always the static --image. V2V: source frame @ idx (1-based ffmpeg, clamped to last).
    auto load_anchor = [&](int abs_idx) -> sd::Tensor<float> {
        sd::Tensor<float> px;
        if (v2v) {
            int fi = abs_idx + 1; if (fi < 1) fi = 1; if (fi > n_src_frames) fi = n_src_frames;
            char fp[512]; snprintf(fp, sizeof(fp), "%s/src_%05d.png", src_dir.c_str(), fi);
            if (!load_image(fp, width, height, px)) printf("WARN: load source frame %d failed\n", fi);
        } else {
            load_image(image_path, width, height, px);
        }
        return px;
    };

    sd::Tensor<float> cond_image_px = load_anchor(0);  // [W,H,1,3] [0,1] — window-0 anchor
    if (cond_image_px.empty()) { printf("ERROR: failed to load conditioning anchor\n"); return 1; }

    sd_tiling_params_t tiling = {};
    tiling.enabled = getenv("IT_NO_VAE_TILE") == nullptr;
    // Temporal-tile the multi-frame conditioning encode too: without it the whole
    // window is one buffer (~6.8 GB at 448x448x21) -> OOM above toy resolutions.
    // Env IT_NO_ENCODE_TEMPORAL reverts to the old whole-window encode.
    tiling.temporal_tiling = getenv("IT_NO_ENCODE_TEMPORAL") == nullptr;
    tiling.target_overlap = 0.25f;
    tiling.rel_size_x = getenv("IT_VAE_TILE_REL") ? atof(getenv("IT_VAE_TILE_REL")) : 0.5f;
    tiling.rel_size_y = getenv("IT_VAE_TILE_REL") ? atof(getenv("IT_VAE_TILE_REL")) : 0.5f;

    // VAE-encode a pixel video [W,H,Tpix,3] [0,1] -> diffusion latent [W,H,Tlat,16].
    // The VAE _compute unsqueeze(2)'s a 4D input (wrong for multi-frame), so pass 5D
    // [W,H,T,C,1] (dim==5 skips the unsqueeze). encode args: (..., circular_x, circular_y).
    auto vae_encode_video = [&](const sd::Tensor<float>& px) -> sd::Tensor<float> {
        sd::Tensor<float> in = px;
        in.reshape_({px.shape()[0], px.shape()[1], px.shape()[2], 3, 1});
        auto mu = vae->encode(n_threads, in, tiling, /*circular_x=*/false, /*circular_y=*/false);
        int64_t Wl = mu.shape()[0], Hl = mu.shape()[1];
        int64_t Tlat = (px.shape()[2] - 1) / cfg.vae_scale + 1;
        mu.reshape_({Wl, Hl, Tlat, 16});
        vae_mu_to_diffusion(mu);
        return mu;
    };
    auto vae_encode_frame = [&](const sd::Tensor<float>& frame_px) -> sd::Tensor<float> {
        auto mu = vae->encode(n_threads, frame_px, tiling, /*circular_x=*/false, /*circular_y=*/false);
        int64_t Wl = mu.shape()[0], Hl = mu.shape()[1];
        mu.reshape_({Wl, Hl, 1, 16});
        vae_mu_to_diffusion(mu);
        return mu;
    };

    // CLIP-H of a single frame [W,H,1,3] -> [1280, 257, 1].
    auto clip_of = [&](const sd::Tensor<float>& frame_px) -> sd::Tensor<float> {
        if (clip_vision) {
            // clip_preprocess wants ne [W,H,3,1] (C in ne[2]); our frame is [W,H,1,3]. T=1 so the
            // memory is identical — reshape, don't permute.
            sd::Tensor<float> img = frame_px;
            img.reshape_({frame_px.shape()[0], frame_px.shape()[1], 3, 1});
            auto pv = clip_preprocess(img, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);
            auto o  = clip_vision->compute(n_threads, pv, /*return_pooled=*/false, /*clip_skip=*/-1);  // [1280,257]
            if (!o.empty()) { o.reshape_({1280, 257, 1}); return o; }
        }
        if (!clipfea_bin.empty()) {
            auto o = sd::load_tensor_from_file_as_tensor<float>(clipfea_bin);
            o.reshape_({1280, 257, 1});
            return o;
        }
        printf("WARN: clip_fea = zeros (no --clip-vision/--clip-fea)\n");
        return sd::Tensor<float>({1280, 257, 1});
    };

    // ---- streaming sampler ----
    unsigned seed_val = 1234;
    if (const char* se = getenv("IT_SEED")) seed_val = (unsigned)atoi(se);
    std::mt19937 rng(seed_val);
    std::normal_distribution<float> nd(0.f, 1.f);

    auto sigmas = distilled ? distilled_sigmas(steps, shift) : flow_sigmas(steps, shift);
    {
        char sbuf[256]; int off = snprintf(sbuf, sizeof(sbuf), "sigmas:");
        for (size_t i = 0; i < sigmas.size() && off < (int)sizeof(sbuf) - 12; ++i)
            off += snprintf(sbuf + off, sizeof(sbuf) - off, " %.5f", sigmas[i]);
        printf("%s (steps=%d shift=%.2f text_cfg=%.2f audio_cfg=%.2f)\n", sbuf, steps, shift, text_cfg, audio_cfg);
    }

    std::vector<sd::Tensor<float>> gen_frames;  // each [W,H,1,3]
    int audio_start_idx = 0;
    int cur_motion_frames = 1;  // first window conditions on the single image
    bool is_first = true;
    sd::Tensor<float> cond_frame_px = cond_image_px;  // for motion latent (last frames of prev video)
    int window = 0;
    // IT_LATENT_CARRY: carry the prior window's CLEAN latent tail directly into the next
    // window's motion pin, instead of decode->carry-pixels->re-encode. Avoids the lossy VAE
    // roundtrip that compounds appearance drift over a multi-window chain.
    bool latent_carry = getenv("IT_LATENT_CARRY") != nullptr;
    sd::Tensor<float> carried_motion_latent;

    while (true) {
        if (max_windows > 0 && window >= max_windows) break;
        printf("\n=== window %d (audio_start=%d / %d, motion_frames=%d) ===\n",
               window, audio_start_idx, T_video, cur_motion_frames);

        // V2V sparse-frame dubbing: re-anchor this window to the SOURCE clip's frame at
        // the window start. Updates the c_concat cond latent + CLIP-H tokens (and, on the
        // first window, the pinned motion latent) to the source — preserving the clip's
        // identity/scene/camera. (I2V leaves cond_image_px = the static image.)
        if (v2v) {
            cond_image_px = load_anchor(audio_start_idx);
            if (cond_image_px.empty()) { printf("ERROR: V2V anchor load failed (win %d)\n", window); return 1; }
        }

        // audio embedding for this window (step/CFG-invariant) -> [768,32,lat_t].
        sd::Tensor<float> first_in, vf_in;
        IT::build_audio_proj_inputs(full_audio, audio_start_idx, frames, cfg, first_in, vf_in);
        auto audio_emb = dit->compute_audio_embedding(n_threads, first_in, vf_in);  // [768,32,lat_t]
        sd::Tensor<float> audio_zero = audio_emb;
        std::fill(audio_zero.data(), audio_zero.data() + audio_zero.numel(), 0.0f);
        dump_stats("audio_emb", audio_emb);
        // IT_NO_GRAFT: disable the audio cross-attn graft entirely (empty -> nullptr in the runner).
        // A/B to test whether the graft causes the dark/green generated-frame distribution shift.
        if (getenv("IT_NO_GRAFT")) { audio_emb = sd::Tensor<float>(); printf("IT_NO_GRAFT: audio graft DISABLED\n"); }

        // c_concat: [mask(4) | vae([cond_image, neutral])(16)] -> [W,H,lat_t,20].
        // Blank (non-cond) frames must be NEUTRAL, not black: vae->encode() applies
        // scale_input ([0,1]->[-1,1]), so 0.0 here decodes to pixel -1 (pure black) and
        // poisons the conditioning latent (negative-mean/high-std) -> generated frames
        // drift dark/green. 0.5 -> 0.0 in [-1,1] = neutral, matching the reference
        // (torch.zeros in [-1,1]) and sd-cli's i2v path (full(0.5)).
        sd::Tensor<float> cond_video({(int64_t)width, (int64_t)height, (int64_t)frames, 3});
        std::fill(cond_video.data(), cond_video.data() + cond_video.numel(), 0.5f);
        sd::ops::slice_assign(&cond_video, 2, 0, 1, cond_image_px);  // frame 0 = the image
        auto y = vae_encode_video(cond_video);  // [W,H,lat_t,16] diffusion space
        sd::Tensor<float> c_concat({(int64_t)lat_w, (int64_t)lat_h, (int64_t)lat_t, 20});
        {
            float* d = c_concat.data();
            const float* yd = y.data();
            int64_t plane = (int64_t)lat_w * lat_h;
            // channels 0..3: mask = 1.0 at latent frame 0 else 0.0.
            for (int ch = 0; ch < 4; ++ch)
                for (int t = 0; t < lat_t; ++t)
                    for (int64_t i = 0; i < plane; ++i)
                        d[((int64_t)ch * lat_t + t) * plane + i] = (t == 0) ? 1.0f : 0.0f;
            // channels 4..19: the VAE latent.
            for (int ch = 0; ch < 16; ++ch)
                for (int t = 0; t < lat_t; ++t)
                    for (int64_t i = 0; i < plane; ++i)
                        d[(((int64_t)ch + 4) * lat_t + t) * plane + i] = yd[((int64_t)ch * lat_t + t) * plane + i];
        }

        // clean motion latents (diffusion space) pinned into the first motion_lat frames.
        sd::Tensor<float> motion_latent;
        if (is_first) {
            motion_latent = vae_encode_frame(cond_image_px);
        } else if (latent_carry && !carried_motion_latent.empty()) {
            motion_latent = carried_motion_latent;  // latent-direct: prior window's clean tail, NO VAE roundtrip
        } else {
            motion_latent = vae_encode_video(cond_frame_px);  // pixel roundtrip (decode -> re-encode)
        }
        int cur_motion_lat = is_first ? 1 : motion_lat;

        // clip_fea of the (last) cond frame.
        auto clip_fea = clip_of(cond_image_px);
        // === INSTRUMENTATION (one-pass root-cause) ===
        dump_stats("clip_fea", clip_fea);          // degenerate (~0 std) => CLIP inert
        dump_stats("c_concat(all)", c_concat);
        dump_per_frame("y_vae(c_concat)", y);      // the VAE cond latent fed into c_concat ch4..19
        dump_per_frame("motion_latent", motion_latent);

        // init noise latent [W,H,lat_t,16].
        sd::Tensor<float> latent({(int64_t)lat_w, (int64_t)lat_h, (int64_t)lat_t, 16});
        for (int64_t i = 0; i < latent.numel(); ++i) latent.data()[i] = nd(rng);

        auto pin_motion = [&]() {
            // overwrite latent[:, :, :cur_motion_lat, :] = motion_latent (clean).
            int64_t plane = (int64_t)lat_w * lat_h;
            for (int ch = 0; ch < 16; ++ch)
                for (int t = 0; t < cur_motion_lat; ++t)
                    memcpy(latent.data() + ((int64_t)ch * lat_t + t) * plane,
                           motion_latent.data() + ((int64_t)ch * cur_motion_lat + t) * plane,
                           plane * sizeof(float));
        };

        int64_t t0 = ggml_time_ms();
        for (int i = 0; i < steps; ++i) {
            pin_motion();
            float sigma_t = sigmas[i], sigma_next = sigmas[i + 1];
            auto ts = sd::Tensor<float>::from_vector(std::vector<float>{sigma_t * 1000.f});

            auto v_cond = dit->compute(n_threads, latent, c_concat, ts, context, clip_fea, audio_emb);
            if (v_cond.empty()) { printf("ERROR: DiT forward empty (win %d step %d)\n", window, i); return 1; }

            sd::Tensor<float> v = v_cond;
            bool do_text  = !(text_cfg == 1.0f);
            bool do_audio = !(audio_cfg == 1.0f);
            if (do_text) {
                auto v_dt = dit->compute(n_threads, latent, c_concat, ts, context_null, clip_fea, audio_emb);
                auto v_un = dit->compute(n_threads, latent, c_concat, ts, context_null, clip_fea, audio_zero);
                for (int64_t k = 0; k < v.numel(); ++k)
                    v.data()[k] = v_un.data()[k] + text_cfg * (v_cond.data()[k] - v_dt.data()[k])
                                  + audio_cfg * (v_dt.data()[k] - v_un.data()[k]);
            } else if (do_audio) {
                auto v_da = dit->compute(n_threads, latent, c_concat, ts, context, clip_fea, audio_zero);
                for (int64_t k = 0; k < v.numel(); ++k)
                    v.data()[k] = v_da.data()[k] + audio_cfg * (v_cond.data()[k] - v_da.data()[k]);
            }

            if (i == 0 || i == steps - 1) { char tg[24]; snprintf(tg, sizeof(tg), "v@step%d", i); dump_per_frame(tg, v); }
            float dt = sigma_next - sigma_t;
            for (int64_t k = 0; k < latent.numel(); ++k) latent.data()[k] += v.data()[k] * dt;
            pin_motion();
        }
        printf("window %d sampled in %.1fs\n", window, (ggml_time_ms() - t0) / 1000.0);
        dump_per_frame("final_latent", latent);  // f0 = pinned anchor; f1.. = generated. Shift here = DiT/cond bug

        // decode: [W,H,lat_t,16] diffusion -> mu -> RGB [W,H,Tpix,3].
        auto dec = latent; vae_diffusion_to_mu(dec);
        dec.unsqueeze_(4);  // [W,H,T,C,1] so VAE::_compute skips unsqueeze(2)
        sd_tiling_params_t dt = {};
        dt.enabled = getenv("IT_NO_VAE_TILE") == nullptr;
        dt.temporal_tiling = true; dt.target_overlap = 0.25f; dt.rel_size_x = 0.34f; dt.rel_size_y = 0.34f;
        vae->set_temporal_tiling_enabled(true);
        auto rgb = vae->decode(n_threads, dec, dt, /*decode_video=*/true, false, false);
        if (rgb.empty()) { printf("ERROR: decode empty\n"); return 1; }
        int64_t Wd = rgb.shape()[0], Hd = rgb.shape()[1], Tpix = rgb.shape()[2];

        // drop the first cur_motion_frames pixel frames on windows>0 (overlap).
        int drop = is_first ? 0 : cur_motion_frames;
        for (int64_t t = drop; t < Tpix; ++t) {
            sd::Tensor<float> fr({Wd, Hd, 1, 3});
            const float* d = rgb.data();
            float* o = fr.data();
            for (int c = 0; c < 3; ++c)
                for (int64_t y2 = 0; y2 < Hd; ++y2)
                    for (int64_t x2 = 0; x2 < Wd; ++x2)
                        o[((int64_t)c * Hd + y2) * Wd + x2] = d[(((int64_t)c * Tpix + t) * Hd + y2) * Wd + x2];
            gen_frames.push_back(std::move(fr));
        }
        printf("window %d: kept %lld frames (total %zu)\n", window, (long long)(Tpix - drop), gen_frames.size());

        // next window: carry the last motion_frame pixels as the motion cond frame.
        is_first = false;
        cur_motion_frames = motion_frame;
        {
            int64_t have = (int64_t)gen_frames.size();
            int64_t mf = std::min<int64_t>(motion_frame, have);
            cond_frame_px = sd::Tensor<float>({Wd, Hd, mf, 3});
            for (int64_t t = 0; t < mf; ++t) {
                const float* s = gen_frames[have - mf + t].data();
                float* o = cond_frame_px.data();
                for (int c = 0; c < 3; ++c)
                    for (int64_t i = 0; i < Wd * Hd; ++i)
                        o[((int64_t)c * mf + t) * Wd * Hd + i] = s[(int64_t)c * Wd * Hd + i];
            }
        }
        if (latent_carry) {
            // carry this window's CLEAN final-latent tail (last motion_lat frames) directly.
            int64_t nml = motion_lat, plane = (int64_t)lat_w * lat_h;
            carried_motion_latent = sd::Tensor<float>({(int64_t)lat_w, (int64_t)lat_h, nml, 16});
            for (int ch = 0; ch < 16; ++ch)
                for (int64_t t = 0; t < nml; ++t)
                    memcpy(carried_motion_latent.data() + ((int64_t)ch * nml + t) * plane,
                           latent.data() + ((int64_t)ch * lat_t + (lat_t - nml + t)) * plane,
                           plane * sizeof(float));
        }
        audio_start_idx += (frames - cur_motion_frames);
        window++;
        if (audio_start_idx + frames > T_video) break;
    }

    // ---- write frames + mux ----
    if (!gen_frames.empty()) {
        int64_t W = gen_frames[0].shape()[0], H = gen_frames[0].shape()[1];
        std::vector<unsigned char> buf((size_t)W * H * 3);
        for (size_t t = 0; t < gen_frames.size(); ++t) {
            const float* d = gen_frames[t].data();
            for (int64_t y = 0; y < H; ++y)
                for (int64_t x = 0; x < W; ++x)
                    for (int c = 0; c < 3; ++c) {
                        float v = d[((int64_t)c * H + y) * W + x];
                        v = v < 0 ? 0 : (v > 1 ? 1 : v);
                        buf[((size_t)y * W + x) * 3 + c] = (unsigned char)std::lround(v * 255.f);
                    }
            char fp[512]; snprintf(fp, sizeof(fp), "%s/frame_%04zu.png", out_dir.c_str(), t);
            stbi_write_png(fp, (int)W, (int)H, 3, buf.data(), (int)W * 3);
        }
        printf("wrote %zu PNG frames to %s\n", gen_frames.size(), out_dir.c_str());

        std::string mp4 = out_dir + "/infinitetalk.mp4";
        char cmd[2048];
        if (!wav_path.empty())
            snprintf(cmd, sizeof(cmd),
                     "ffmpeg -y -framerate %d -i '%s/frame_%%04d.png' -i '%s' -c:v libx264 -pix_fmt yuv420p "
                     "-c:a aac -shortest '%s' 2>%s/ffmpeg.log", fps, out_dir.c_str(), wav_path.c_str(), mp4.c_str(), out_dir.c_str());
        else
            snprintf(cmd, sizeof(cmd),
                     "ffmpeg -y -framerate %d -i '%s/frame_%%04d.png' -c:v libx264 -pix_fmt yuv420p '%s' 2>%s/ffmpeg.log",
                     fps, out_dir.c_str(), mp4.c_str(), out_dir.c_str());
        int rc = system(cmd);
        printf("ffmpeg rc=%d -> %s\n", rc, mp4.c_str());
    }

    printf("\nInfiniteTalk render complete. (NOT validated against a torch oracle.)\n");
    return 0;
}
