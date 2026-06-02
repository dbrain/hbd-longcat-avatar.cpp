// Wan2.2-S2V-14B (LiveAvatar) M1 CLI — render-ready harness.
//
// M1 SCOPE: STOCK non-causal model, NO LoRA, drop_motion_frames=True (single clip,
// FramePack motioner SKIPPED), full-step flow-match sampling. Loads the DiT +
// wav2vec2 + casual_audio_encoder GGUFs, runs the audio pipeline, the flow-match
// loop, and the Wan2.1 VAE decode to RGB frames -> PNGs + mp4 (via ffmpeg).
//
// Usage:
//   sd-s2v --dit <dit.gguf> --wav2vec <w2v.gguf> --audioenc <cae.gguf>
//          --vae <vae.gguf> --umt5 <umt5.gguf>
//          --ref-image <png> --prompt "<text>" --wav <audio.wav>
//          [--out <dir>] [--frames N] [--height H] [--width W]
//          [--steps S] [--cfg G] [--shift F] [--fps R] [--cpu] [--load-only]
//
//   --ref-image  : reference face PNG/JPG; resized (short side) + center-cropped to
//                  HxW, VAE-encoded to the single ref latent frame.
//   --prompt     : text prompt; umT5-XXL encoded -> context (padded to 512).
//   --wav        : 16kHz mono wav (resampled). Drives the audio path.
//
// Latent geometry (matches speech2video.py): lat_motion_frames = (73+3)//4 = 19;
//   lat_target = (infer_frames + 3 + 73)//4 - 19. Audio: 30fps bucket cropped to
//   infer_frames, prepend 73 copies of frame-0, CausalAudioEncoder (÷4 temporal),
//   then drop the first 19 latent frames -> per-latent-frame audio tokens.
// Decode (drop_motion, r==0): cat([ref_latent, latents], dim=T), decode, take last
//   infer_frames pixel frames, drop the first 3.

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
#include "t5.hpp"
#include "tensor.hpp"
#include "tensor_ggml.hpp"
#include "wan.hpp"
#include "wan_s2v.hpp"
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

// Write a sd::Tensor<float> in the simple .bin format that
// load_tensor_from_file_as_tensor reads back.
static void write_bin(const std::string& path, const sd::Tensor<float>& t, const std::string& name) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("cannot open %s\n", path.c_str()); return; }
    int32_t n_dims = (int32_t)t.dim();
    int32_t len    = (int32_t)name.size();
    int32_t ttype  = (int32_t)GGML_TYPE_F32;
    fwrite(&n_dims, sizeof(int32_t), 1, f);
    fwrite(&len, sizeof(int32_t), 1, f);
    fwrite(&ttype, sizeof(int32_t), 1, f);
    for (int i = 0; i < n_dims; ++i) { int32_t d = (int32_t)t.shape()[(size_t)i]; fwrite(&d, sizeof(int32_t), 1, f); }
    fwrite(name.data(), 1, (size_t)len, f);
    fwrite(t.data(), sizeof(float), (size_t)t.numel(), f);
    fclose(f);
    printf("wrote %s\n", path.c_str());
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

// Wan2.1 (16ch) VAE latent normalization constants (vae2_1.py / WanVAERunner).
static const float WAN_VAE_MEAN[16] = {
    -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
    0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
static const float WAN_VAE_STD[16] = {
    2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
    3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

// In/out latents are ggml ne [W,H,T,C=16]; per-channel (ne[3]) normalize.
// raw mu -> diffusion: (mu - mean)/std ;  diffusion -> raw: z*std + mean.
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

// flow-match sigma schedule (shift applied). sigma linearly 1->0 over `steps`,
// reparam sigma' = shift*sigma / (1 + (shift-1)*sigma); timestep = sigma*1000.
static std::vector<float> flow_sigmas(int steps, float shift) {
    std::vector<float> s(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        float sigma = 1.0f - (float)i / (float)steps;  // 1 .. 0
        s[i] = shift * sigma / (1.0f + (shift - 1.0f) * sigma);
    }
    return s;
}

// Load image -> RGB[0,1], short-side resize then center-crop to WxH, as ggml ne
// [W,H,1,3] (channel-planar, ne[0] fastest = x). Mirrors speech2video.py's
// Resize(min(H,W)) + CenterCrop((H,W)) + ToTensor (channels-first, [0,1]).
static bool load_ref_image(const std::string& path, int W, int H, sd::Tensor<float>& out) {
    int iw, ih, ic;
    unsigned char* img = stbi_load(path.c_str(), &iw, &ih, &ic, 3);
    if (!img) { printf("ERROR: stbi_load %s failed\n", path.c_str()); return false; }
    // Resize so the short side == min(W,H), preserving aspect (transforms.Resize).
    int target_short = std::min(W, H);
    double scale = (double)target_short / (double)std::min(iw, ih);
    int rw = (int)std::round(iw * scale), rh = (int)std::round(ih * scale);
    rw = std::max(rw, W); rh = std::max(rh, H);
    std::vector<unsigned char> resized((size_t)rw * rh * 3);
    stbir_resize_uint8(img, iw, ih, 0, resized.data(), rw, rh, 0, 3);
    stbi_image_free(img);
    // Center-crop to WxH.
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

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    sd_set_log_callback(log_cb, nullptr);

    if (argc < 2 || has_flag(argc, argv, "--help")) {
        printf("usage: %s --dit <dit.gguf> --wav2vec <w2v.gguf> --audioenc <cae.gguf>\n", argv[0]);
        printf("          --vae <vae.gguf> --umt5 <umt5.gguf>\n");
        printf("          --ref-image <png> --prompt \"<text>\" --wav <audio.wav>\n");
        printf("          [--out <dir>] [--frames N] [--height H] [--width W]\n");
        printf("          [--steps S] [--cfg G] [--shift F] [--fps R] [--cpu] [--load-only]\n");
        return 1;
    }

    std::string dit_path  = opt(argc, argv, "--dit");
    std::string w2v_path  = opt(argc, argv, "--wav2vec");
    std::string cae_path  = opt(argc, argv, "--audioenc");
    std::string vae_path  = opt(argc, argv, "--vae");
    std::string umt5_path = opt(argc, argv, "--umt5");
    std::string ref_img   = opt(argc, argv, "--ref-image");
    std::string ref_path  = opt(argc, argv, "--ref-latent");  // legacy: pre-encoded latent .bin
    std::string prompt    = opt(argc, argv, "--prompt", "a person talking");
    std::string ctx_path  = opt(argc, argv, "--context");     // legacy: pre-encoded umT5 .bin
    std::string wav_path  = opt(argc, argv, "--wav");
    std::string out_dir   = opt(argc, argv, "--out", "./s2v_out");

    int   frames = atoi(opt(argc, argv, "--frames", "49").c_str());
    int   height = atoi(opt(argc, argv, "--height", "512").c_str());
    int   width  = atoi(opt(argc, argv, "--width", "512").c_str());
    int   steps  = atoi(opt(argc, argv, "--steps", "20").c_str());
    float cfg    = atof(opt(argc, argv, "--cfg", "4.5").c_str());
    float shift  = atof(opt(argc, argv, "--shift", "3.0").c_str());
    int   fps    = atoi(opt(argc, argv, "--fps", "16").c_str());
    bool  cpu    = has_flag(argc, argv, "--cpu");
    bool  load_only = has_flag(argc, argv, "--load-only");

    // S2V motion-frame constants (configs/wan_s2v_14B.py: motion_frames=73).
    const int MOTION_FRAMES     = 73;
    const int LAT_MOTION_FRAMES = (MOTION_FRAMES + 3) / 4;  // 19

    ggml_backend_t backend = nullptr;
    ggml_backend_load_all();
    if (!cpu) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (backend) printf("backend: GPU (%s)\n", ggml_backend_name(backend));
        else printf("WARN: no GPU backend available, falling back to CPU\n");
    }
    if (!backend) { backend = ggml_backend_cpu_init(); printf("backend: CPU\n"); }

    // The audio path (wav2vec2 + CausalAudioEncoder) hits a CUDA binbcast contiguity
    // assert on some broadcast ops; run it on a dedicated CPU backend (it is tiny).
    // Override with S2V_AUDIO_ON_GPU=1 to force the audio path onto `backend`.
    ggml_backend_t audio_backend = backend;
    if (getenv("S2V_AUDIO_ON_GPU") == nullptr) {
        audio_backend = ggml_backend_cpu_init();
        printf("audio path backend: CPU (DiT/VAE on %s)\n", ggml_backend_name(backend));
    }

    int n_threads = 8;

    // ---- load DiT ----
    if (dit_path.empty()) { printf("ERROR: --dit required\n"); return 1; }
    printf("loading S2V DiT '%s'\n", dit_path.c_str());
    std::shared_ptr<WAN_S2V::WanS2VRunner> dit;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(dit_path, "model.diffusion_model.")) {
            printf("ERROR: init loader %s\n", dit_path.c_str()); return 1;
        }
        dit = std::make_shared<WAN_S2V::WanS2VRunner>(backend, backend, loader.get_tensor_storage_map(), "model.diffusion_model");
        dit->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        dit->get_param_tensors(tensors, "model.diffusion_model");
        if (!loader.load_tensors(tensors)) { printf("ERROR: load DiT tensors\n"); return 1; }
        // Flash attention is REQUIRED here: the self-attn is over ~13k tokens x 40
        // heads; without FA the materialized scores tensor needs ~28 GB. FA keeps
        // the DiT compute buffer to a few GB. (L_k=13312 % 256 == 0, head_dim=128.)
        dit->set_flash_attention_enabled(getenv("S2V_NO_FLASH") == nullptr);
        printf("S2V DiT loaded (%zu tensors), flash_attn=%d\n", tensors.size(),
               getenv("S2V_NO_FLASH") == nullptr);
    }

    // ---- load wav2vec2 + casual audio encoder ----
    std::shared_ptr<WAV2VEC2::Wav2Vec2EncoderRunner> w2v;
    std::shared_ptr<WAV2VEC2::CausalAudioEncoderRunner> cae;
    if (!w2v_path.empty()) {
        printf("loading wav2vec2 '%s'\n", w2v_path.c_str());
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(w2v_path, "")) { printf("ERROR: init w2v loader\n"); return 1; }
        w2v = std::make_shared<WAV2VEC2::Wav2Vec2EncoderRunner>(audio_backend, audio_backend, loader.get_tensor_storage_map(), "audio_encoder");
        w2v->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        w2v->get_param_tensors(tensors, "audio_encoder");
        if (!loader.load_tensors(tensors)) { printf("WARN: some wav2vec2 tensors failed to load\n"); }
        printf("wav2vec2 loaded (%zu tensors)\n", tensors.size());
    }
    if (!cae_path.empty()) {
        printf("loading casual_audio_encoder '%s'\n", cae_path.c_str());
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(cae_path, "")) { printf("ERROR: init cae loader\n"); return 1; }
        cae = std::make_shared<WAV2VEC2::CausalAudioEncoderRunner>(audio_backend, audio_backend, loader.get_tensor_storage_map(), "model.diffusion_model.casual_audio_encoder");
        cae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        cae->get_param_tensors(tensors, "model.diffusion_model.casual_audio_encoder");
        if (!loader.load_tensors(tensors)) { printf("WARN: some casual_audio_encoder tensors failed to load\n"); }
        printf("casual_audio_encoder loaded (%zu tensors)\n", tensors.size());
    }

    if (load_only) { printf("\n--load-only: all GGUFs loaded successfully. exiting.\n"); return 0; }

    // ---- latent geometry ----
    // lat_target_frames = (infer_frames + 3 + motion_frames)//4 - lat_motion_frames.
    int lat_t = (frames + 3 + MOTION_FRAMES) / 4 - LAT_MOTION_FRAMES;
    int lat_h = height / 8;
    int lat_w = width / 8;
    printf("latent target: [%d, %d, %d, 16]  (frames=%d %dx%d, steps=%d cfg=%.2f shift=%.2f)\n",
           lat_w, lat_h, lat_t, frames, height, width, steps, cfg, shift);

    // ---- VAE (encode ref image + final decode) ----
    std::shared_ptr<WAN::WanVAERunner> vae;
    if (!vae_path.empty()) {
        printf("loading VAE '%s'\n", vae_path.c_str());
        vae = std::make_shared<WAN::WanVAERunner>(backend, backend, String2TensorStorage{}, "", /*decode_only=*/false, VERSION_WAN2);
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(vae_path, "vae.")) { printf("ERROR: VAE loader init\n"); return 1; }
        vae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        vae->get_param_tensors(tensors, "first_stage_model");
        if (!loader.load_tensors(tensors)) { printf("ERROR: VAE tensor load failed\n"); return 1; }
        vae->set_flash_attention_enabled(true);
        printf("VAE loaded (%zu tensors)\n", tensors.size());
    }

    // ---- reference latent ----
    // ref_pixel = resize/crop ref image -> [W,H,1,3] [0,1]; vae.encode internally
    // scales to [-1,1] and returns the diffusion-space latent (mean/std normalized),
    // matching speech2video.py's self.vae.encode(ref_pixel*2-1).
    sd::Tensor<float> ref_latent;
    sd_tiling_params_t tiling = {}; tiling.enabled = false;
    if (!ref_path.empty()) {
        ref_latent = sd::load_tensor_from_file_as_tensor<float>(ref_path);
        dump_stats("ref_latent(.bin)", ref_latent);
    } else if (vae && !ref_img.empty()) {
        sd::Tensor<float> ref_px;
        if (!load_ref_image(ref_img, width, height, ref_px)) return 1;
        dump_stats("ref_pixel [0,1]", ref_px);
        auto mu = vae->encode(n_threads, ref_px, tiling, false, false);
        dump_stats("ref vae mu", mu);
        // Single-frame encode returns ne [W,H,16,1] (C in ne[2], T=1 in ne[3]); the DiT
        // patch_embedding wants [W,H,T=1,C=16]. For T=1 the memory layout is identical,
        // so reshape to the expected ne, THEN per-channel normalize over ne[3].
        ref_latent = mu;
        ref_latent.reshape_({(int64_t)lat_w, (int64_t)lat_h, 1, 16});
        vae_mu_to_diffusion(ref_latent);  // (mu - mean)/std
        dump_stats("ref_latent", ref_latent);

        // VAE roundtrip diagnostic: decode the ref latent straight back to a PNG to
        // isolate VAE-path correctness from the DiT/sampler. (S2V_REF_ROUNDTRIP=1)
        if (getenv("S2V_REF_ROUNDTRIP")) {
            auto rr = mu;  // raw VAE-space single frame
            rr.reshape_({(int64_t)lat_w, (int64_t)lat_h, 1, 16});
            rr.unsqueeze_(4);  // [W,H,T=1,C=16,1] -> skip _compute unsqueeze(2)
            sd_tiling_params_t rt = {}; rt.enabled = false;
            auto rrgb = vae->decode(n_threads, rr, rt, true, false, false);
            dump_stats("ref roundtrip rgb", rrgb);
            if (!rrgb.empty()) {
                int64_t W = rrgb.shape()[0], H = rrgb.shape()[1];
                const float* d = rrgb.data();
                std::vector<unsigned char> fr((size_t)W * H * 3);
                for (int64_t y = 0; y < H; ++y)
                    for (int64_t x2 = 0; x2 < W; ++x2)
                        for (int c = 0; c < 3; ++c) {
                            float v = d[((size_t)c * H + y) * W + x2];
                            v = v < 0 ? 0 : (v > 1 ? 1 : v);
                            fr[((size_t)y * W + x2) * 3 + c] = (unsigned char)std::lround(v * 255.f);
                        }
                std::string rp = out_dir + "/ref_roundtrip.png";
                stbi_write_png(rp.c_str(), (int)W, (int)H, 3, fr.data(), (int)W * 3);
                printf("wrote ref roundtrip -> %s\n", rp.c_str());
            }
            return 0;
        }
    } else {
        ref_latent = sd::Tensor<float>({(int64_t)lat_w, (int64_t)lat_h, 1, 16});
        printf("WARN: no --ref-image/--vae, ref_latent = zeros\n");
    }

    // ---- text context (umT5) ----
    sd::Tensor<float> context;
    if (!ctx_path.empty()) {
        context = sd::load_tensor_from_file_as_tensor<float>(ctx_path);
        dump_stats("context(.bin)", context);
    } else if (!umt5_path.empty()) {
        printf("loading umT5 '%s' and encoding prompt: \"%s\"\n", umt5_path.c_str(), prompt.c_str());
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(umt5_path, "text_encoders.t5xxl.transformer.")) {
            printf("ERROR: umT5 loader init\n"); return 1;
        }
        auto t5 = std::make_shared<T5Embedder>(audio_backend, audio_backend, loader.get_tensor_storage_map(), "text_encoders.t5xxl.transformer", true);
        t5->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        if (!loader.load_tensors(tensors)) { printf("WARN: some umT5 tensors failed to load\n"); }
        // tokenize (no padding -> keep the real token count; the DiT pads to text_len=512).
        auto tw  = t5->tokenize(prompt, 512, false);
        auto ids = sd::Tensor<int32_t>::from_vector(std::get<0>(tw));
        auto am  = sd::Tensor<float>::from_vector(std::get<2>(tw));
        context  = t5->model.compute(n_threads, ids, am);  // [text_dim=4096, n_tok, 1]
        dump_stats("context (umT5)", context);
        // model_s2v.py pads the text context to text_len=512 with zeros BEFORE the
        // text_embedding MLP (the DiT does NOT pad internally). Without this the
        // cross-attn sees only the few real tokens — weak/incorrect text conditioning.
        {
            int64_t tok = context.shape()[1];
            const int64_t TEXT_LEN = 512;
            if (tok < TEXT_LEN) {
                sd::Tensor<float> ctx_pad({context.shape()[0], TEXT_LEN, context.shape()[2]});
                std::fill(ctx_pad.data(), ctx_pad.data() + ctx_pad.numel(), 0.0f);
                sd::ops::slice_assign(&ctx_pad, 1, 0, tok, context);
                context = std::move(ctx_pad);
                dump_stats("context (padded 512)", context);
            }
        }
    } else {
        context = sd::Tensor<float>({4096, 512, 1});
        printf("WARN: no --umt5/--context, context = zeros\n");
    }

    // ---- audio pipeline ----
    sd::Tensor<float> audio_tokens, audio_global;
    if (w2v && cae && !wav_path.empty()) {
        std::vector<float> wav;
        if (!LONGCAT_AUDIO::load_wav_16k_mono(wav_path, wav)) { printf("ERROR: load wav\n"); return 1; }
        auto wav_n = WAV2VEC2::normalize_waveform(wav);
        sd::Tensor<float> wav_t({(int64_t)wav_n.size(), 1, 1});
        memcpy(wav_t.data(), wav_n.data(), wav_n.size() * sizeof(float));
        printf("running wav2vec2 on %zu samples...\n", wav_n.size());
        auto hs = w2v->compute(n_threads, wav_t);  // [d_model, T_enc, 25]
        dump_stats("wav2vec2_hs", hs);

        int out_len = 0;
        auto interp = WAV2VEC2::interp_50_to_30(hs, /*video_length=*/0, out_len);
        int64_t d_model = hs.shape()[0], n_layers = hs.shape()[2];
        int num_repeat = 0;
        // bucket at 30fps; audio_frame_num=out_len, infer_frames=this clip's pixel frames.
        auto bucket = WAV2VEC2::audio_embed_bucket_fps(interp, n_layers, d_model, out_len,
                                                       /*infer_frames=*/frames, fps, num_repeat);
        dump_stats("audio_bucket", bucket);  // [d_model, bucket_num, 25]

        // This clip = first `frames` bucket entries (r==0 -> left_idx=0).
        int64_t bucket_num = bucket.shape()[1];
        int64_t clip_n = std::min<int64_t>(frames, bucket_num);
        auto clip = sd::ops::narrow(bucket, /*dim=*/1, /*start=*/0, /*len=*/clip_n);  // [d_model, frames, 25]

        // model_s2v.py:683: audio_input = cat([audio[...,0:1].repeat(motion_frames[0]), audio]).
        // Prepend MOTION_FRAMES copies of frame-0 along the frame dim (ne[1]).
        auto frame0 = sd::ops::narrow(clip, 1, 0, 1);  // [d_model, 1, 25]
        sd::Tensor<float> prepad({d_model, (int64_t)MOTION_FRAMES, n_layers});
        for (int f = 0; f < MOTION_FRAMES; ++f)
            sd::ops::slice_assign(&prepad, 1, f, f + 1, frame0);
        auto cae_in = sd::ops::concat(prepad, clip, 1);  // [d_model, MOTION_FRAMES+frames, 25]
        dump_stats("cae_in (prepad+clip)", cae_in);

        auto aout = cae->compute(n_threads, cae_in);          // x_local  [dim, 5, F3]
        auto aglob = cae->compute_global(n_threads, cae_in);  // x_global [dim, 1, F3]
        dump_stats("cae x_local", aout);
        dump_stats("cae x_global", aglob);

        // crop [motion_frames[1]:] along the frame dim (ne[2] for both local & global).
        int64_t F3 = aout.shape()[2];
        int64_t crop = std::min<int64_t>(LAT_MOTION_FRAMES, F3);
        int64_t keep = F3 - crop;
        printf("CAE F3=%lld, drop first %lld -> keep %lld (need lat_t=%d)\n",
               (long long)F3, (long long)crop, (long long)keep, lat_t);
        audio_tokens = sd::ops::narrow(aout,  2, crop, keep);
        audio_global = sd::ops::narrow(aglob, 2, crop, keep);
        // Align to exactly lat_t frames (the noisy latent frame count).
        if (audio_tokens.shape()[2] > lat_t) {
            audio_tokens = sd::ops::narrow(audio_tokens, 2, 0, lat_t);
            audio_global = sd::ops::narrow(audio_global, 2, 0, lat_t);
        } else if (audio_tokens.shape()[2] < lat_t) {
            // pad trailing frames with the last available frame.
            int64_t have = audio_tokens.shape()[2];
            sd::Tensor<float> at2({audio_tokens.shape()[0], audio_tokens.shape()[1], (int64_t)lat_t});
            sd::Tensor<float> ag2({audio_global.shape()[0], audio_global.shape()[1], (int64_t)lat_t});
            for (int f = 0; f < lat_t; ++f) {
                int64_t src = std::min<int64_t>(f, have - 1);
                sd::ops::slice_assign(&at2, 2, f, f + 1, sd::ops::narrow(audio_tokens, 2, src, 1));
                sd::ops::slice_assign(&ag2, 2, f, f + 1, sd::ops::narrow(audio_global, 2, src, 1));
            }
            audio_tokens = std::move(at2);
            audio_global = std::move(ag2);
        }
        dump_stats("audio_tokens(cropped)", audio_tokens);
        dump_stats("audio_global(cropped)", audio_global);
    } else {
        printf("WARN: audio path skipped (need --wav2vec --audioenc --wav). audio tokens = zeros.\n");
        audio_tokens = sd::Tensor<float>({5120, 5, (int64_t)lat_t});
        audio_global = sd::Tensor<float>({5120, 1, (int64_t)lat_t});
    }

    // ---- M1 flow-match sampler ----
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    sd::Tensor<float> x({(int64_t)lat_w, (int64_t)lat_h, (int64_t)lat_t, 16});
    for (int64_t i = 0; i < x.numel(); ++i) x.data()[i] = nd(rng);

    sd::Tensor<float> cond_states;  // M1: empty (zeros) -> cond_encoder contributes +0.
    auto sigmas = flow_sigmas(steps, shift);

    printf("\n=== flow-match sampling (%d steps) ===\n", steps);
    int64_t t_sample0 = ggml_time_ms();
    for (int i = 0; i < steps; ++i) {
        float sigma_t = sigmas[i], sigma_next = sigmas[i + 1];
        sd::Tensor<float> ts = sd::Tensor<float>::from_vector(std::vector<float>{sigma_t * 1000.f});

        auto v_cond = dit->compute(n_threads, x, ref_latent, cond_states, ts, context, audio_tokens, audio_global);
        if (v_cond.empty()) { printf("ERROR: DiT forward returned empty at step %d\n", i); return 1; }

        sd::Tensor<float> v = v_cond;
        if (cfg > 1.0f) {
            sd::Tensor<float> audio_zero = audio_tokens;
            for (int64_t k = 0; k < audio_zero.numel(); ++k) audio_zero.data()[k] = 0.f;
            auto v_uncond = dit->compute(n_threads, x, ref_latent, cond_states, ts, context, audio_zero, audio_global);
            if (!v_uncond.empty())
                for (int64_t k = 0; k < v.numel(); ++k)
                    v.data()[k] = v_uncond.data()[k] + cfg * (v_cond.data()[k] - v_uncond.data()[k]);
        }

        float dt = sigma_next - sigma_t;
        for (int64_t k = 0; k < x.numel(); ++k) x.data()[k] += v.data()[k] * dt;

        if (i == 0 || i == steps - 1) { char tag[32]; snprintf(tag, sizeof(tag), "x@step%d", i); dump_stats(tag, x); }
    }
    int64_t t_sample1 = ggml_time_ms();
    dump_stats("final_latent", x);
    printf("sampling: %.1fs total, %.2fs/step (cfg=%.1f -> %d fwd/step)\n",
           (t_sample1 - t_sample0) / 1000.0, (t_sample1 - t_sample0) / 1000.0 / steps,
           cfg, cfg > 1.0f ? 2 : 1);

    {
        std::string lp = out_dir + "/s2v_final_latent.bin";
        write_bin(lp, x, "s2v_final_latent");
    }

    // Free the DiT (~9.7 GB VRAM) before VAE decode — the decode compute buffer
    // needs several GB and won't coexist with the resident DiT on a 12 GB card.
    dit->free_params_buffer();
    dit->free_compute_buffer();
    printf("freed DiT params/compute buffers before VAE decode\n");

    // ---- VAE decode -> frames -> PNG + mp4 ----
    if (vae) {
        // drop_motion, r==0: decode_latents = cat([ref_latents, latents], dim=T).
        // ref_latent + x are diffusion-space; decode() does NOT re-apply mean/std, so
        // convert to raw VAE space (z*std+mean) here, concat along T (ne[2]), decode.
        printf("\ndecoding final latent (cat ref + noisy)...\n");
        auto ref_vae   = ref_latent; vae_diffusion_to_mu(ref_vae);    // [W,H,1,16]
        auto noisy_vae = x;          vae_diffusion_to_mu(noisy_vae);  // [W,H,lat_t,16]
        auto dec_diff  = sd::ops::concat(ref_vae, noisy_vae, 2);      // [W,H,1+lat_t,16]
        dump_stats("decode latent (vae)", dec_diff);
        // VAE::_compute does z.unsqueeze(2) on a 4D input, which mis-shapes a
        // multi-frame [W,H,T,C] video latent (the first conv then reads T as the
        // channel dim -> im2col assert). Pre-add a trailing singleton so dim()==5
        // and the unsqueeze is skipped; WanVAE.decode reads ne[2]=T, ne[3]=C right.
        dec_diff.unsqueeze_(4);  // [W,H,T,C,1]
        // Spatial+temporal VAE tiling: the full-frame single-shot decode compute
        // buffer is ~13 GB (too big for a 12 GB card even with the DiT freed). Tiling
        // decodes in spatial tiles (default 32-latent) with temporal chunking.
        sd_tiling_params_t dec_tiling = {};
        dec_tiling.enabled         = getenv("S2V_NO_VAE_TILE") == nullptr;
        dec_tiling.temporal_tiling = true;
        dec_tiling.target_overlap  = 0.25f;
        dec_tiling.rel_size_x      = 0.5f;  // ~2 tiles across each spatial dim
        dec_tiling.rel_size_y      = 0.5f;
        vae->set_temporal_tiling_enabled(true);
        auto rgb = vae->decode(n_threads, dec_diff, dec_tiling, /*decode_video=*/true, false, false);
        dump_stats("decoded_rgb (all)", rgb);
        if (!rgb.empty() && rgb.dim() >= 4) {
            int64_t W = rgb.shape()[0], H = rgb.shape()[1], T = rgb.shape()[2];
            // image = image[:, :, -infer_frames:]; then image[:, :, 3:]  (drop_motion r==0).
            int64_t start = std::max<int64_t>(0, T - frames);
            auto last = sd::ops::narrow(rgb, 2, start, T - start);
            int64_t T2 = last.shape()[2];
            int64_t drop = std::min<int64_t>(3, T2);
            auto vid = sd::ops::narrow(last, 2, drop, T2 - drop);  // [W,H,Tout,3]
            int64_t Tout = vid.shape()[2];
            dump_stats("video frames", vid);

            // Write each frame as PNG (channel-planar [W,H,T,C] -> interleaved RGB).
            const float* d = vid.data();
            std::vector<unsigned char> frame((size_t)W * H * 3);
            for (int64_t t = 0; t < Tout; ++t) {
                for (int64_t y = 0; y < H; ++y)
                    for (int64_t x2 = 0; x2 < W; ++x2)
                        for (int c = 0; c < 3; ++c) {
                            float val = d[(((size_t)c * Tout + t) * H + y) * W + x2];
                            val = val < 0.f ? 0.f : (val > 1.f ? 1.f : val);
                            frame[((size_t)y * W + x2) * 3 + c] = (unsigned char)std::lround(val * 255.0f);
                        }
                char fp[512];
                snprintf(fp, sizeof(fp), "%s/frame_%04lld.png", out_dir.c_str(), (long long)t);
                stbi_write_png(fp, (int)W, (int)H, 3, frame.data(), (int)W * 3);
            }
            printf("wrote %lld PNG frames to %s\n", (long long)Tout, out_dir.c_str());

            // Mux to mp4 with audio via ffmpeg CLI (best-effort).
            std::string mp4 = out_dir + "/s2v.mp4";
            char cmd[2048];
            if (!wav_path.empty()) {
                snprintf(cmd, sizeof(cmd),
                         "ffmpeg -y -framerate %d -i '%s/frame_%%04d.png' -i '%s' "
                         "-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest '%s' 2>%s/ffmpeg.log",
                         fps, out_dir.c_str(), wav_path.c_str(), mp4.c_str(), out_dir.c_str());
            } else {
                snprintf(cmd, sizeof(cmd),
                         "ffmpeg -y -framerate %d -i '%s/frame_%%04d.png' "
                         "-c:v libx264 -pix_fmt yuv420p '%s' 2>%s/ffmpeg.log",
                         fps, out_dir.c_str(), mp4.c_str(), out_dir.c_str());
            }
            printf("mux: %s\n", cmd);
            int rc = system(cmd);
            printf("ffmpeg rc=%d -> %s\n", rc, mp4.c_str());
        }
    }

    printf("\nM1 forward/sampler completed. (NOT validated against a torch oracle.)\n");
    return 0;
}
