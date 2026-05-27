// Standalone Wan VAE encode->decode isolation harness for the LongCat avatar port.
// NOT part of the avatar inference path; it exists only to validate the VAE gguf
// round-trip against a torch oracle. Build target: sd-vae-roundtrip.
//
// Usage:
//   sd-vae-roundtrip <vae.gguf> <input_rgb.bin> <oracle_latent.bin> <out_dir> [--cpu]
//   sd-vae-roundtrip <vae.gguf> <input_rgb.bin> --iters N [--gpu]
//
// Steps (default mode):
//   1. load the Wan2.1 (16ch) VAE gguf,
//   2. encode <input_rgb.bin> ([0,1] RGB, ggml ne [W,H,T,C]) -> latent mu, dump stats + bin,
//   3. decode our-own-latent -> pixels, dump stats + bin,
//   4. decode the torch <oracle_latent.bin> -> pixels, dump stats + bin.
// Comparing (3) vs (4) isolates ENCODE vs DECODE faults.
//
// --iters N mode (DETAIL DECAY probe): encode->decode the input N times, feeding
// each decode output back as the next encode input (exactly the chained-render
// per-seam cond round-trip, in isolation from the DiT). Prints per-iteration
// sharpness (variance-of-Laplacian on luma) so the per-round-trip blur floor is
// quantified — answers STEP 2's "how fast does the VAE round-trip lose detail".

#include <cstdio>
#include <cstring>
#include <string>

#include "ggml.h"
#include "model.h"
#include "stable-diffusion.h"
#include "tensor.hpp"
#include "tensor_ggml.hpp"
#include "wan.hpp"

static void log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
}

static void dump_stats(const char* tag, const sd::Tensor<float>& t) {
    if (t.empty()) {
        printf("%-22s EMPTY\n", tag);
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
    printf("%-22s shape=%s mean=%.5f std=%.5f min=%.4f max=%.4f\n",
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

// variance-of-Laplacian sharpness on the decoded RGB tensor (ne = [W,H,T,C]),
// averaged over frames. Mirrors tools/detail_analysis.py's vlap on luma.
static double sharpness_vlap(const sd::Tensor<float>& rgb) {
    const auto& s = rgb.shape();
    if (s.size() < 4) return 0.0;
    int64_t W = s[0], H = s[1], T = s[2], C = s[3];
    if (W < 3 || H < 3 || C < 3) return 0.0;
    const float* d = rgb.data();
    // index helper: data laid out ne[0] fastest -> idx = ((c*T + t)*H + y)*W + x
    auto at = [&](int64_t x, int64_t y, int64_t t, int64_t c) -> double {
        return (double)d[(((c * T + t) * H + y) * W + x)];
    };
    double sum_vlap = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        // luma plane + Laplacian variance
        double s1 = 0.0, s2 = 0.0;
        int64_t cnt = 0;
        auto luma = [&](int64_t x, int64_t y) {
            return 0.299 * at(x, y, t, 0) + 0.587 * at(x, y, t, 1) + 0.114 * at(x, y, t, 2);
        };
        for (int64_t y = 1; y < H - 1; ++y)
            for (int64_t x = 1; x < W - 1; ++x) {
                double lap = -4.0 * luma(x, y) + luma(x - 1, y) + luma(x + 1, y) + luma(x, y - 1) + luma(x, y + 1);
                s1 += lap;
                s2 += lap * lap;
                ++cnt;
            }
        double mean = s1 / cnt;
        sum_vlap += s2 / cnt - mean * mean;
    }
    // input is [0,1]; scale to 0..255 luma domain to match detail_analysis.py (×255^2)
    return (sum_vlap / T) * (255.0 * 255.0);
}

int main(int argc, char** argv) {
    // --iters mode: sd-vae-roundtrip <vae.gguf> <input_rgb.bin> --iters N
    // (CPU; the round-trip detail loss is a VAE-architecture property.)
    int iters = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--iters" && i + 1 < argc) iters = atoi(argv[i + 1]);
    }

    if (iters <= 0 && argc < 5) {
        printf("usage: %s <vae.gguf> <input_rgb.bin> <oracle_latent.bin> <out_dir> [--cpu]\n", argv[0]);
        printf("   or: %s <vae.gguf> <input_rgb.bin> --iters N\n", argv[0]);
        return 1;
    }
    std::string vae_path = argv[1];
    std::string in_path  = argv[2];

    sd_set_log_callback(log_cb, nullptr);

    if (iters > 0) {
        ggml_backend_t backend = ggml_backend_cpu_init();
        printf("backend: CPU\n");
        auto vae = std::make_shared<WAN::WanVAERunner>(
            backend, backend, String2TensorStorage{}, "", /*decode_only=*/false, VERSION_WAN2);
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(vae_path, "vae.")) {
            printf("failed to init loader from %s\n", vae_path.c_str());
            return 1;
        }
        vae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        vae->get_param_tensors(tensors, "first_stage_model");
        if (!loader.load_tensors(tensors)) {
            printf("failed to load VAE tensors\n");
            return 1;
        }
        int n_threads = 8;
        sd_tiling_params_t tiling = {};
        tiling.enabled = false;

        auto cur = sd::load_tensor_from_file_as_tensor<float>(in_path);
        printf("\n=== VAE round-trip DETAIL DECAY (decode->re-encode loop, %d iters) ===\n", iters);
        double v0 = sharpness_vlap(cur);
        printf("iter  0 (input)        vlap=%9.3f  (100.0%% of input)\n", v0);
        for (int it = 1; it <= iters; ++it) {
            auto mu = vae->encode(n_threads, cur, tiling, false, false);
            if (mu.empty()) { printf("encode failed at iter %d\n", it); break; }
            auto rec = vae->decode(n_threads, mu, tiling, true, false, false);
            if (rec.empty()) { printf("decode failed at iter %d\n", it); break; }
            // clamp to [0,1] (decode can over/undershoot) before feeding back, as the
            // pixel path does when it writes uint8.
            for (int64_t i = 0; i < rec.numel(); ++i) {
                float x = rec.data()[i];
                rec.data()[i] = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
            }
            double v = sharpness_vlap(rec);
            printf("iter %2d (roundtrip)    vlap=%9.3f  (%5.1f%% of input)\n", it, v, 100.0 * v / v0);
            cur = std::move(rec);
        }
        return 0;
    }

    std::string lat_path = argv[3];
    std::string out_dir  = argv[4];
    (void)argc;

    // VAE is light; run on CPU for a deterministic, dependency-free harness.
    ggml_backend_t backend = ggml_backend_cpu_init();
    printf("backend: CPU\n");

    // Wan2.1 16-channel VAE, full (encode+decode).
    auto vae = std::make_shared<WAN::WanVAERunner>(
        backend, backend, String2TensorStorage{}, "", /*decode_only=*/false, VERSION_WAN2);

    ModelLoader loader;
    if (!loader.init_from_file_and_convert_name(vae_path, "vae.")) {
        printf("failed to init loader from %s\n", vae_path.c_str());
        return 1;
    }
    vae->alloc_params_buffer();
    std::map<std::string, ggml_tensor*> tensors;
    vae->get_param_tensors(tensors, "first_stage_model");
    if (!loader.load_tensors(tensors)) {
        printf("failed to load VAE tensors\n");
        return 1;
    }
    printf("VAE loaded (%zu tensors)\n", tensors.size());

    int n_threads = 8;
    sd_tiling_params_t tiling = {};
    tiling.enabled = false;

    // --- encode ---
    auto x = sd::load_tensor_from_file_as_tensor<float>(in_path);
    dump_stats("input (0..1 rgb)", x);
    auto mu = vae->encode(n_threads, x, tiling, false, false);
    dump_stats("sdcpp encode mu", mu);
    write_bin(out_dir + "/sdcpp_latent.bin", mu, "sdcpp_latent");

    // --- decode our own latent ---
    if (!mu.empty()) {
        auto rec = vae->decode(n_threads, mu, tiling, true, false, false);
        dump_stats("decode(sdcpp mu)", rec);
        write_bin(out_dir + "/sdcpp_recon_self.bin", rec, "recon_self");
    }

    // --- decode the torch oracle latent ---
    auto zoracle = sd::load_tensor_from_file_as_tensor<float>(lat_path);
    dump_stats("oracle latent", zoracle);
    auto rec2 = vae->decode(n_threads, zoracle, tiling, true, false, false);
    dump_stats("decode(oracle z)", rec2);
    write_bin(out_dir + "/sdcpp_recon_oracle.bin", rec2, "recon_oracle");

    return 0;
}
