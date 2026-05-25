// Standalone Wan VAE encode->decode isolation harness for the LongCat avatar port.
// NOT part of the avatar inference path; it exists only to validate the VAE gguf
// round-trip against a torch oracle. Build target: sd-vae-roundtrip.
//
// Usage:
//   sd-vae-roundtrip <vae.gguf> <input_rgb.bin> <oracle_latent.bin> <out_dir> [--cpu]
//
// Steps:
//   1. load the Wan2.1 (16ch) VAE gguf,
//   2. encode <input_rgb.bin> ([0,1] RGB, ggml ne [W,H,T,C]) -> latent mu, dump stats + bin,
//   3. decode our-own-latent -> pixels, dump stats + bin,
//   4. decode the torch <oracle_latent.bin> -> pixels, dump stats + bin.
// Comparing (3) vs (4) isolates ENCODE vs DECODE faults.

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

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("usage: %s <vae.gguf> <input_rgb.bin> <oracle_latent.bin> <out_dir> [--cpu]\n", argv[0]);
        return 1;
    }
    std::string vae_path = argv[1];
    std::string in_path  = argv[2];
    std::string lat_path = argv[3];
    std::string out_dir  = argv[4];
    (void)argc;

    sd_set_log_callback(log_cb, nullptr);

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
