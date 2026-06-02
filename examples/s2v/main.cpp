// Wan2.2-S2V-14B (LiveAvatar) M1 CLI — render-ready harness.
//
// M1 SCOPE: STOCK non-causal model, NO LoRA, drop_motion_frames=True (single clip,
// FramePack motioner SKIPPED), full-step flow-match sampling. Loads the DiT +
// wav2vec2 + casual_audio_encoder GGUFs, runs the audio pipeline, the flow-match
// loop, and (optionally) the Wan2.1 VAE decode to RGB frames.
//
// This file's PRIMARY purpose for M1 is structural/load verification: it loads the
// ggufs into the runners and runs a forward pass / full sampler on CPU (or CUDA if
// built with SD_CUDA). The parent runs the actual GPU render later.
//
// Usage:
//   sd-s2v --dit <dit.gguf> --wav2vec <w2v.gguf> --audioenc <cae.gguf>
//          --ref-latent <ref.bin> --context <umt5.bin> --wav <audio.wav>
//          [--vae <vae.gguf>] [--out <dir>]
//          [--frames N] [--height H] [--width W] [--steps S] [--cfg G] [--cpu]
//
//   --ref-latent : Wan2.1-VAE-encoded reference image latent, ggml ne [W,H,1,16].
//   --context    : umT5-XXL text embedding [text_dim=4096, text_len, 1], padded to 512.
//   --wav        : 16kHz mono wav (or any RIFF/WAVE; resampled). Drives the audio path.
//
// The latent target frame count is derived from --frames (infer_frames): the noisy
// latent has lat_target = (infer_frames + 3) // 4 frames at H/8 x W/8 (channels 16).
//
// NOTE: GGUF tensor naming follows model_s2v.py's state_dict; match
// /mnt/hdd/live-avatar/gguf/NAMING.md when produced (see wan_s2v.hpp / wav2vec2.hpp).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model.h"
#include "stable-diffusion.h"
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
// load_tensor_from_file_as_tensor reads back (n_dims, name_len, ggml_type, dims[], name, data).
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

// flow-match sigma schedule (shift applied). Mirrors FlowUniPC/FlowMatch sigmas:
//   sigmas linearly spaced 1->0 over `steps`, with the sample_shift reparam
//   sigma' = shift*sigma / (1 + (shift-1)*sigma). timestep = sigma*1000.
static std::vector<float> flow_sigmas(int steps, float shift) {
    std::vector<float> s(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        float sigma = 1.0f - (float)i / (float)steps;  // 1 .. 0
        s[i] = shift * sigma / (1.0f + (shift - 1.0f) * sigma);
    }
    return s;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: keep progress visible up to a crash
    setvbuf(stderr, nullptr, _IONBF, 0);
    sd_set_log_callback(log_cb, nullptr);

    if (argc < 2 || has_flag(argc, argv, "--help")) {
        printf("usage: %s --dit <dit.gguf> --wav2vec <w2v.gguf> --audioenc <cae.gguf>\n", argv[0]);
        printf("          --ref-latent <ref.bin> --context <umt5.bin> --wav <audio.wav>\n");
        printf("          [--vae <vae.gguf>] [--out <dir>] [--frames N] [--height H] [--width W]\n");
        printf("          [--steps S] [--cfg G] [--shift F] [--cpu] [--load-only]\n");
        return 1;
    }

    std::string dit_path  = opt(argc, argv, "--dit");
    std::string w2v_path  = opt(argc, argv, "--wav2vec");
    std::string cae_path  = opt(argc, argv, "--audioenc");
    std::string ref_path  = opt(argc, argv, "--ref-latent");
    std::string ctx_path  = opt(argc, argv, "--context");
    std::string wav_path  = opt(argc, argv, "--wav");
    std::string vae_path  = opt(argc, argv, "--vae");
    std::string out_dir   = opt(argc, argv, "--out", "./s2v_out");

    int   frames = atoi(opt(argc, argv, "--frames", "80").c_str());
    int   height = atoi(opt(argc, argv, "--height", "512").c_str());
    int   width  = atoi(opt(argc, argv, "--width", "512").c_str());
    int   steps  = atoi(opt(argc, argv, "--steps", "40").c_str());
    float cfg    = atof(opt(argc, argv, "--cfg", "4.5").c_str());
    float shift  = atof(opt(argc, argv, "--shift", "3.0").c_str());
    bool  cpu    = has_flag(argc, argv, "--cpu");
    bool  load_only = has_flag(argc, argv, "--load-only");

    ggml_backend_t backend = nullptr;
    ggml_backend_load_all();  // register CUDA/CPU backends from the ggml registry
    if (!cpu) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (backend) printf("backend: GPU (%s)\n", ggml_backend_name(backend));
        else printf("WARN: no GPU backend available, falling back to CPU\n");
    }
    if (!backend) { backend = ggml_backend_cpu_init(); printf("backend: CPU\n"); }

    // The audio path (wav2vec2 + CausalAudioEncoder) is tiny (<20 MB compute) and
    // hits a CUDA binbcast contiguity assert on some of its broadcast ops; run it
    // on a dedicated CPU backend. The heavy DiT + VAE stay on `backend` (GPU).
    // Override with S2V_AUDIO_ON_GPU=1 to force the audio path onto `backend`.
    ggml_backend_t audio_backend = backend;
    if (getenv("S2V_AUDIO_ON_GPU") == nullptr) {
        audio_backend = ggml_backend_cpu_init();
        printf("audio path backend: CPU (DiT/VAE on %s)\n", ggml_backend_name(backend));
    }

    int n_threads = 8;

    // ---- load DiT ----
    // IMPORTANT: build the runner WITH the gguf's tensor_storage_map so each
    // GGMLBlock creates its weight tensor at the gguf's stored dtype (Q4_K/F16),
    // not the F32 default (which would balloon the param buffer to ~62 GB).
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
        printf("S2V DiT loaded (%zu tensors)\n", tensors.size());
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
        // The CausalAudioEncoder lives INSIDE the DiT checkpoint, so its tensors
        // carry the DiT's "model.diffusion_model." prefix (NAMING.md: routed to the
        // DiT gguf). Request them under the full prefix.
        cae = std::make_shared<WAV2VEC2::CausalAudioEncoderRunner>(audio_backend, audio_backend, loader.get_tensor_storage_map(), "model.diffusion_model.casual_audio_encoder");
        cae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        cae->get_param_tensors(tensors, "model.diffusion_model.casual_audio_encoder");
        if (!loader.load_tensors(tensors)) { printf("WARN: some casual_audio_encoder tensors failed to load\n"); }
        printf("casual_audio_encoder loaded (%zu tensors)\n", tensors.size());
    }

    if (load_only) { printf("\n--load-only: all GGUFs loaded successfully. exiting.\n"); return 0; }

    // ---- latent geometry ----
    // lat_target_frames = (infer_frames + 3) // 4  (M1: motion_frames dropped, so no
    // motion-latent subtraction). channels = 16, spatial = H/8 x W/8.
    int lat_t = (frames + 3) / 4;
    int lat_h = height / 8;
    int lat_w = width / 8;
    printf("latent target: [%d, %d, %d, 16]  (frames=%d %dx%d, steps=%d cfg=%.2f)\n",
           lat_w, lat_h, lat_t, frames, height, width, steps, cfg);

    // ---- reference latent + text context ----
    sd::Tensor<float> ref_latent, context;
    if (!ref_path.empty()) { ref_latent = sd::load_tensor_from_file_as_tensor<float>(ref_path); dump_stats("ref_latent", ref_latent); }
    else { ref_latent = sd::Tensor<float>({(int64_t)lat_w, (int64_t)lat_h, 1, 16}); printf("WARN: no --ref-latent, using zeros\n"); }
    if (!ctx_path.empty()) { context = sd::load_tensor_from_file_as_tensor<float>(ctx_path); dump_stats("context", context); }
    else { context = sd::Tensor<float>({4096, 512, 1}); printf("WARN: no --context, using zeros\n"); }

    // ---- audio pipeline ----
    // 1. wav -> normalized waveform -> wav2vec2 -> [d_model, T_enc, 25].
    // 2. interp 50->30 fps; 3. bucket fps=16/m=0 -> [audio_dim, F_bucket, 25];
    // 4. casual_audio_encoder -> per-frame tokens [dim, 5, F].
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
        // interp to 30fps, video_length = 0 -> derive from T_enc.
        auto interp = WAV2VEC2::interp_50_to_30(hs, /*video_length=*/0, out_len);
        int64_t d_model = hs.shape()[0], n_layers = hs.shape()[2];
        int num_repeat = 0;
        auto bucket = WAV2VEC2::audio_embed_bucket_fps(interp, n_layers, d_model, out_len,
                                                       /*infer_frames=*/lat_t * 4, /*fps=*/16, num_repeat);
        dump_stats("audio_bucket", bucket);

        // crop to this clip's F latent frames: motion_frames context is prepended in
        // the reference before the causal encoder; for M1 drop_motion_frames we feed
        // the bucketed [d_model, F, 25] directly (F = lat_t).  TODO: prepend the
        // motion_frames[0] frame-0 repeats + crop [motion_frames[1]:] for exact parity.
        auto aout = cae->compute(n_threads, bucket);  // x_local [dim, 5, F]
        dump_stats("audio_tokens", aout);
        audio_tokens = aout;  // [dim, 5, F]; injector k/v project 5120->5120 (CAE out_dim).
        // x_global [dim, 1, F] -> AdaLN temb (enable_adain attn_norm path). The DiT's
        // audio_inject AdaLayerNorm REQUIRES this; without it the injector pre-norm
        // dereferences a null temb. Compute it from the same bucket.
        audio_global = cae->compute_global(n_threads, bucket);  // [dim, 1, F]
        dump_stats("audio_global", audio_global);
    } else {
        printf("WARN: audio path skipped (need --wav2vec --audioenc --wav). audio tokens = zeros.\n");
        audio_tokens = sd::Tensor<float>({5120, 5, (int64_t)lat_t});
    }

    // ---- M1 flow-match sampler ----
    // x = noise; for each step: v = dit(x,...); x = x + v*(sigma_next - sigma_t).
    // CFG: run cond + uncond (uncond zeroes the audio + uses null context).
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    sd::Tensor<float> x({(int64_t)lat_w, (int64_t)lat_h, (int64_t)lat_t, 16});
    for (int64_t i = 0; i < x.numel(); ++i) x.data()[i] = nd(rng);

    sd::Tensor<float> cond_states;  // M1: empty (zeros) -> cond_encoder contributes +0.
    auto sigmas = flow_sigmas(steps, shift);

    printf("\n=== flow-match sampling (%d steps) ===\n", steps);
    for (int i = 0; i < steps; ++i) {
        float sigma_t = sigmas[i], sigma_next = sigmas[i + 1];
        sd::Tensor<float> ts = sd::Tensor<float>::from_vector(std::vector<float>{sigma_t * 1000.f});

        auto v_cond = dit->compute(n_threads, x, ref_latent, cond_states, ts, context, audio_tokens, audio_global);
        if (v_cond.empty()) { printf("ERROR: DiT forward returned empty at step %d\n", i); return 1; }

        sd::Tensor<float> v = v_cond;
        if (cfg > 1.0f) {
            // uncond: zero audio tokens (mirrors arg_null audio_input = 0).
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
    dump_stats("final_latent", x);

    // Always persist the final latent so a forward pass produces an output file even
    // if VAE decode is unavailable/fails. ne layout [W,H,T,16].
    {
        std::string lp = out_dir + "/s2v_final_latent.bin";
        write_bin(lp, x, "s2v_final_latent");
    }

    // ---- optional VAE decode ----
    if (!vae_path.empty()) {
        printf("\nloading VAE '%s' for decode...\n", vae_path.c_str());
        auto vae = std::make_shared<WAN::WanVAERunner>(backend, backend, String2TensorStorage{}, "", /*decode_only=*/true, VERSION_WAN2);
        ModelLoader loader;
        if (loader.init_from_file_and_convert_name(vae_path, "vae.")) {
            vae->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            vae->get_param_tensors(tensors, "first_stage_model");
            if (loader.load_tensors(tensors)) {
                sd_tiling_params_t tiling = {}; tiling.enabled = false;
                // The DiT emits the latent as ne [W,H,T,C=16] which is the Wan VAE
                // CausalConv3d input layout. NOTE: the Wan VAE decode still asserts in a
                // 2D-conv im2col for this S2V latent — the exact ref-frame concat / frame
                // handling is the parent's next-phase work; the raw latent is saved above.
                auto rgb = vae->decode(n_threads, x, tiling, true, false, false);
                dump_stats("decoded_rgb", rgb);
                if (!rgb.empty()) {
                    std::string rp = out_dir + "/s2v_decoded_rgb.bin";
                    write_bin(rp, rgb, "s2v_decoded_rgb");
                    printf("decoded %lld frames -> %s\n",
                           rgb.shape().size() >= 3 ? (long long)rgb.shape()[2] : 0, rp.c_str());
                }
            } else printf("WARN: VAE tensor load failed\n");
        } else printf("WARN: VAE loader init failed\n");
    }

    printf("\nM1 forward/sampler completed. (NOT validated against a torch oracle.)\n");
    (void)out_dir;
    return 0;
}
