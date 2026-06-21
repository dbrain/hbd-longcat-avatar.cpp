// ggml flash_attn_ext baseline at LTX-2.3 DiT self-attention shapes, SAME GPU.
// Replicates the LTX self-attn flash path from src/core/ggml_extend.hpp:
//   q:[d_head, L, H] f32, k:[d_head, L, H] f16, v:[d_head, L, H] f16, mask=null,
//   scale=1/sqrt(d_head), GGML_PREC_F32, no kv-pad (flash_skip_kv_pad, mask==null).
// Built against the repo's prebuilt libggml-cuda.a so the kernel is the EXACT one
// production LTX runs. Times median over iters on the CUDA backend.
//
// Build (inside flux2-dev:builder-cudnn):
//   nvcc -O3 -arch=sm_120 -std=c++17 attn_ggml_baseline.cu -o attn_ggml_baseline \
//     -I/src/ggml/include -L/src/build-cudnn/ggml/src -L/src/build-cudnn/ggml/src/ggml-cuda \
//     -lggml -lggml-base -lggml-cpu -lggml-cuda -lcublas -lcuda
// (link order matters; ggml-cuda before ggml-base)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char** argv) {
    int64_t H = 30, S = 11440, D = 128;
    int iters = 50;
    const char* tag = "ltx_video_selfattn";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        if (!strcmp(argv[i], "--H") && i + 1 < argc) H = atoll(argv[++i]);
        if (!strcmp(argv[i], "--S") && i + 1 < argc) S = atoll(argv[++i]);
        if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoll(argv[++i]);
        if (!strcmp(argv[i], "--tag") && i + 1 < argc) tag = argv[++i];
    }
    const float scale = 1.0f / sqrtf((float)D);
    fprintf(stderr, "=== ggml flash_attn_ext LTX  %s  H=%ld S=%ld D=%ld scale=%.6f ===\n", tag, H, S, D, scale);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "cuda init failed\n"); return 1; }

    // graph context (no_alloc: tensors allocated in a backend buffer via gallocr)
    size_t ctx_sz = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params ip = { ctx_sz, nullptr, true };
    struct ggml_context* ctx = ggml_init(ip);

    // flash_attn_ext expects q [d, n_q, n_head, n_batch], k/v [d, n_kv, n_head, n_batch]
    struct ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, S, H, 1);
    struct ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S, H, 1);
    struct ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S, H, 1);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    struct ggml_tensor* out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_output(out);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc graph failed\n"); return 1; }

    // fill inputs
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    {
        size_t n = (size_t)D * S * H;
        std::vector<float> qf(n);
        std::vector<uint16_t> kh(n), vh(n);
        for (size_t i = 0; i < n; i++) qf[i] = nd(rng) * 0.5f;
        for (size_t i = 0; i < n; i++) kh[i] = ggml_fp32_to_fp16(nd(rng) * 0.5f);
        for (size_t i = 0; i < n; i++) vh[i] = ggml_fp32_to_fp16(nd(rng) * 0.5f);
        ggml_backend_tensor_set(q, qf.data(), 0, n * sizeof(float));
        ggml_backend_tensor_set(k, kh.data(), 0, n * sizeof(uint16_t));
        ggml_backend_tensor_set(v, vh.data(), 0, n * sizeof(uint16_t));
    }

    // warmup
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed (no flash kernel for D=%ld on sm_120?)\n", D); return 1; }
    ggml_backend_synchronize(backend);

    // timing: wall-clock median (backend compute incl. sync)
    std::vector<double> times(iters);
    for (int it = 0; it < iters; it++) {
        ggml_backend_synchronize(backend);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[it] = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
    }
    std::sort(times.begin(), times.end());
    double med_us = times[iters / 2];
    double flops = 2.0 * 1 * H * (2.0 * (double)S * S * D);
    double tflops = flops / (med_us * 1e-6) / 1e12;
    fprintf(stderr, "  median %.1f us  %.1f TFLOP/s\n", med_us, tflops);
    printf("{\"kernel\":\"ggml_flash_attn_ext\",\"tag\":\"%s\",\"H\":%ld,\"S\":%ld,\"D\":%ld,\"us\":%.2f,\"tflops\":%.2f}\n",
           tag, H, S, D, med_us, tflops);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
