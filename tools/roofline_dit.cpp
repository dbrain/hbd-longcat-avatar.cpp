// Roofline microbench for the LongCat-Avatar DiT hot ops, IN ISOLATION (no render).
// Times ggml's Q4_K MMQ matmul at the avatar's hot shapes (M≈10920 tokens) and the
// d=128 flash-attn, computes achieved effective TFLOPS and the arithmetic intensity,
// and reports achieved-% of the applicable roofline (BW roof for the Q4_K weight read,
// fp16-tensor roof for flash-attn) so we can decide compute- vs BW-bound and whether a
// hand kernel has headroom.
//
// Build inside the builder image; links libggml. Run on CUDA (sm_86 / RTX 3060).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// RTX 3060 (GA106, sm_86) published roofs.
static const double BW_GBPS          = 360.0;   // GDDR6 192-bit @ 15 Gbps ≈ 360 GB/s
static const double INT8_TOPS        = 101.0;   // peak INT8 tensor-core TOPS
static const double FP16_TF          = 25.6;    // peak FP16 tensor-core TFLOPS (FP16 acc)
                                                // (datasheet ~12.7 non-tensor; tensor ~25.6, no sparsity)

struct BenchResult {
    double ms;
    double tflops_eff;
    double weight_bytes;     // Q4_K weight bytes read (the BW-heavy term), per call
    double ai_flop_per_byte; // arithmetic intensity vs the weight read
};

// Time op via cudaEvent-equivalent: ggml_backend_graph_compute + synchronize, averaged.
static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int iters) {
    // warmup
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    int64_t t0 = ggml_time_us();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    int64_t t1 = ggml_time_us();
    return (t1 - t0) / 1000.0 / iters;  // ms/iter
}

// Q4_K weight [K(in) x N(out)] (ggml ne0=K, ne1=N) times F32 activation [K x M].
// ggml_mul_mat(W, x) -> [N x M]. M = token count.
static BenchResult bench_q4k_matmul(ggml_backend_t backend, int64_t K, int64_t N, int64_t M,
                                    const char * label, ggml_type wtype = GGML_TYPE_Q4_K) {
    ggml_init_params ip{ (size_t)64 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, N);            // weight (quant or f16)
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);    // F32 activation
    ggml_set_input(W); ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);                          // [N x M]
    ggml_set_output(y);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // fill: random Q4_K weight bytes + random F32 activation. (Values don't affect timing.)
    size_t wbytes = ggml_nbytes(W);
    std::vector<uint8_t> wb(wbytes);
    for (auto & b : wb) b = (uint8_t)(rand() & 0xff);
    ggml_backend_tensor_set(W, wb.data(), 0, wbytes);
    std::vector<float> xv((size_t)K * M);
    for (auto & v : xv) v = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));

    double ms = time_graph(backend, gf, 30);

    double flops   = 2.0 * (double)M * N * K;          // MAC = 2 flops
    double tflops  = flops / (ms / 1000.0) / 1e12;
    double wbytes_f = (double)wbytes;                   // Q4_K weight read (≈ 0.5625 B/elem)
    double ai      = flops / wbytes_f;

    printf("  %-22s K=%-6lld N=%-6lld M=%-6lld | %7.3f ms | %6.2f TFLOPS eff | "
           "Wread %6.1f MiB | AI %6.1f flop/B\n",
           label, (long long)K, (long long)N, (long long)M, ms, tflops,
           wbytes_f / 1048576.0, ai);

    // roof analysis: BW-bound time = weight_bytes / BW; compute-bound (int8 tensor) time = flops / TOPS.
    // (MMQ quantizes activation to Q8_1 and runs int8 tensor cores, so the compute roof is INT8 TOPS.)
    double t_bw_ms   = wbytes_f / (BW_GBPS * 1e9) * 1000.0;
    double t_int8_ms = flops / (INT8_TOPS * 1e12) * 1000.0;
    const char * roof = (t_bw_ms > t_int8_ms) ? "BW" : "INT8-compute";
    double t_roof_ms = (t_bw_ms > t_int8_ms) ? t_bw_ms : t_int8_ms;
    double pct = t_roof_ms / ms * 100.0;
    printf("      roof=%-12s  t_bw=%.3f ms  t_int8=%.3f ms  -> achieved %.1f%% of %s roof\n",
           roof, t_bw_ms, t_int8_ms, pct, roof);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return { ms, tflops, wbytes_f, ai };
}

// Flash-attn: q/k/v [head_dim x n_head x L x 1], k/v F16, mask null.
static void bench_flash_attn(ggml_backend_t backend, int64_t d_head, int64_t n_head, int64_t L,
                             const char * label) {
    ggml_init_params ip{ (size_t)64 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    // ggml_flash_attn_ext expects q [d, n_head, L, batch], k/v [d, n_head, Lk, batch] (k/v F16).
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, L, n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, L, n_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, L, n_head, 1);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    float scale = 1.0f / sqrtf((float)d_head);
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    ggml_set_output(o);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> qv((size_t)d_head * n_head * L);
    for (auto & val : qv) val = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * sizeof(float));
    // k/v F16: fill with small halfs
    std::vector<uint16_t> kf((size_t)d_head * n_head * L, 0x3000);  // ~0.125 in f16
    ggml_backend_tensor_set(k, kf.data(), 0, kf.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(v, kf.data(), 0, kf.size() * sizeof(uint16_t));

    double ms = time_graph(backend, gf, 20);

    // FLOPs: 2 * (QK^T) + 2 * (PV) = 2*L*L*d*n_head + 2*L*L*d*n_head = 4*L*L*d*n_head
    double flops  = 4.0 * (double)L * L * d_head * n_head;
    double tflops = flops / (ms / 1000.0) / 1e12;
    double pct    = tflops / FP16_TF * 100.0;
    printf("  %-22s d=%-3lld n_head=%-3lld L=%-6lld | %7.3f ms | %6.2f TFLOPS eff | "
           "-> achieved %.1f%% of FP16-tensor roof (%.1f TF)\n",
           label, (long long)d_head, (long long)n_head, (long long)L, ms, tflops, pct, FP16_TF);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("no cuda\n"); return 1; }
    srand(1234);

    printf("=== RTX 3060 roofs: BW %.0f GB/s | INT8 %.0f TOPS | FP16-tensor %.1f TFLOPS ===\n",
           BW_GBPS, INT8_TOPS, FP16_TF);

    // M ≈ 10920 tokens at 25f (7 latent frames x 1560 spatial). Noise pass ≈ 9360, full ≈ 10920.
    const int64_t M = 10920;
    printf("\n-- Q4_K MMQ matmuls (the per-block weight matmuls), M=%lld --\n", (long long)M);
    bench_q4k_matmul(backend, 11008, 4096, M, "ffn.w2 [11008->4096]");
    bench_q4k_matmul(backend, 4096, 11008, M, "ffn.w1 [4096->11008]");
    bench_q4k_matmul(backend, 4096, 11008, M, "ffn.w3 [4096->11008]");
    bench_q4k_matmul(backend, 4096, 4096,  M, "qkv/proj [4096->4096]");

    printf("\n-- same shape, alt weight types (MMQ ceiling vs Q4_K-specific deficit) --\n");
    bench_q4k_matmul(backend, 4096, 11008, M, "ffn.w1 Q8_0", GGML_TYPE_Q8_0);
    bench_q4k_matmul(backend, 4096, 11008, M, "ffn.w1 Q4_0", GGML_TYPE_Q4_0);
    bench_q4k_matmul(backend, 4096, 11008, M, "ffn.w1 F16(cuBLAS)", GGML_TYPE_F16);

    printf("\n-- Flash-attn d=128 (self-attn over the noise tokens) --\n");
    bench_flash_attn(backend, 128, 32, 10920, "self-attn full L=10920");
    bench_flash_attn(backend, 128, 32, 9360,  "self-attn noise L=9360");

    return 0;
}
