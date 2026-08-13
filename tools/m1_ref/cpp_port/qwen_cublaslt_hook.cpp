// Exact cuBLASLt dispatch for Qwen3 BF16 o_proj during the diagnostic
// expanded-beam prefix prefill. PyTorch 2.10/cu130 selects algo 21 for this
// 896x2048 by 10280 GEMM; ggml's ordinary cuBLAS path is close but differs
// after the BF16 activation boundary.
#include "ggml.h"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

struct ggml_backend_cuda_context;
extern "C" {
typedef bool (*ggml_cuda_mul_mat_hook_fn)(ggml_backend_cuda_context *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, cudaStream_t);
void ggml_cuda_set_mul_mat_hook(ggml_cuda_mul_mat_hook_fn fn);
}

static cublasLtHandle_t qwen_lt = nullptr;

static bool qwen_o_proj_lt(ggml_backend_cuda_context *, const ggml_tensor * w,
                           const ggml_tensor * x, ggml_tensor * y, cudaStream_t stream) {
    const int64_t n = x->ne[1] * x->ne[2] * x->ne[3];
    if (w->type != GGML_TYPE_BF16 || x->type != GGML_TYPE_BF16 || y->type != GGML_TYPE_F32 ||
        w->ne[0] != 2048 || w->ne[1] != 896 || x->ne[0] != 2048 || y->ne[0] != 896 ||
        n != 10280 || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(y)) return false;
    if (!qwen_lt && cublasLtCreate(&qwen_lt) != CUBLAS_STATUS_SUCCESS) return false;

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr, dd = nullptr;
    cublasLtMatmulAlgo_t algo{};
    cublasStatus_t st = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transa = CUBLAS_OP_T;
    int sm = 28;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm, sizeof(sm));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&ad, CUDA_R_16BF, 2048, 896, 2048);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&bd, CUDA_R_16BF, 2048, n, 2048);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&cd, CUDA_R_32F, 896, n, 896);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&dd, CUDA_R_32F, 896, n, 896);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulAlgoInit(qwen_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F, 21, &algo);
    uint32_t tile = CUBLASLT_MATMUL_TILE_128x256, stages = CUBLASLT_MATMUL_STAGES_32x3;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages));
    const float alpha = 1.0f, beta = 0.0f;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmul(qwen_lt, op, &alpha, w->data, ad, x->data, bd,
        &beta, y->data, cd, y->data, dd, &algo, nullptr, 0, stream);
    if (dd) cublasLtMatrixLayoutDestroy(dd);
    if (cd) cublasLtMatrixLayoutDestroy(cd);
    if (bd) cublasLtMatrixLayoutDestroy(bd);
    if (ad) cublasLtMatrixLayoutDestroy(ad);
    if (op) cublasLtMatmulDescDestroy(op);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    std::fprintf(stderr, "[QWEN_CUBLASLT] exact B20 o_proj N=%lld\n", (long long) n);
    return true;
}

// The expanded HF prefix also chooses fixed cuBLASLt kernels for Q/K/V.  The
// result is BF16 (unlike o_proj's F32 accumulator boundary), so ggml's normal
// GEMM can differ by one BF16 ULP before QK-norm and RoPE amplify it.
static bool qwen_prefix_qkv_lt(ggml_backend_cuda_context *, const ggml_tensor * w,
                               const ggml_tensor * x, ggml_tensor * y, cudaStream_t stream) {
    const int64_t n = x->ne[1] * x->ne[2] * x->ne[3];
    const int in = (int) w->ne[0], out = (int) w->ne[1];
    const bool prefix_qkv = n == 10280 &&
        ((in == 896 && (out == 2048 || out == 1024 || out == 3072 || out == 33036)) ||
         (in == 2048 && out == 896) || (in == 3072 && out == 896));
    // Cached decode is deliberately not intercepted here. Its activations
    // have graph-specific strides/cache history; forcing the standalone Lt
    // kernels changes the beam result. Prefix B20 is the independently
    // replayed contract this hook owns.
    const bool decode_linear = n == 20 && in == 2048 && out == 896 &&
        std::getenv("QWEN_CUBLASLT_DECODE_OPROJ") != nullptr;
    if (w->type != GGML_TYPE_BF16 || x->type != GGML_TYPE_BF16 || y->type != GGML_TYPE_F32 ||
        x->ne[0] != in || y->ne[0] != out || !(prefix_qkv || decode_linear) ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(y)) return false;
    if (!qwen_lt && cublasLtCreate(&qwen_lt) != CUBLAS_STATUS_SUCCESS) return false;

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr, dd = nullptr, f32d = nullptr;
    cublasLtMatrixTransformDesc_t to_f32 = nullptr;
    void * bf16_out = nullptr;
    cublasLtMatmulAlgo_t algo{};
    cublasStatus_t st = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transa = CUBLAS_OP_T;
    int sm = 28;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm, sizeof(sm));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&ad, CUDA_R_16BF, in, out, in);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&bd, CUDA_R_16BF, in, n, in);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&cd, CUDA_R_16BF, out, n, out);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&dd, CUDA_R_16BF, out, n, out);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&f32d, CUDA_R_32F, out, n, out);
    if (st == CUBLAS_STATUS_SUCCESS && cudaMalloc(&bf16_out, (size_t) out * n * sizeof(__nv_bfloat16)) != cudaSuccess) st = CUBLAS_STATUS_ALLOC_FAILED;
    int algo_id = 0;
    uint32_t tile = 0, stages = 0;
    int swizzle = 0;
    if (prefix_qkv) {
        if (in == 896 && out == 2048) {
            algo_id = 6; tile = CUBLASLT_MATMUL_TILE_256x128; stages = CUBLASLT_MATMUL_STAGES_32x3;
        } else if (in == 896 && out == 1024) {
            algo_id = 5; tile = CUBLASLT_MATMUL_TILE_128x128; swizzle = 1;
        } else if (in == 896 && out == 3072) {
            algo_id = 21; tile = CUBLASLT_MATMUL_TILE_256x128; stages = CUBLASLT_MATMUL_STAGES_32x3;
        } else if (in == 3072 && out == 896) {
            algo_id = 21; tile = CUBLASLT_MATMUL_TILE_128x256; stages = CUBLASLT_MATMUL_STAGES_32x3;
        } else if (in == 2048 && out == 896) {
            algo_id = 21; tile = CUBLASLT_MATMUL_TILE_128x256; stages = CUBLASLT_MATMUL_STAGES_32x3;
        } else {
            algo_id = 23; tile = CUBLASLT_MATMUL_TILE_128x256; stages = CUBLASLT_MATMUL_STAGES_32x3;
        }
    } else if (in == 896 && (out == 2048 || out == 1024)) {
        algo_id = 31; tile = CUBLASLT_MATMUL_TILE_64x64; stages = CUBLASLT_MATMUL_STAGES_64x5; swizzle = 1;
    } else if (in == 2048 && out == 896) {
        algo_id = 31; tile = CUBLASLT_MATMUL_TILE_64x64; stages = CUBLASLT_MATMUL_STAGES_64x6;
    } else if (in == 896 && out == 3072) {
        algo_id = 6; tile = CUBLASLT_MATMUL_TILE_128x64; stages = CUBLASLT_MATMUL_STAGES_64x3;
    } else if (in == 3072 && out == 896) {
        algo_id = 31; tile = CUBLASLT_MATMUL_TILE_64x64; stages = CUBLASLT_MATMUL_STAGES_64x5; swizzle = 1;
    } else {
        algo_id = 23; tile = CUBLASLT_MATMUL_TILE_128x64; stages = CUBLASLT_MATMUL_STAGES_64x3;
    }
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulAlgoInit(qwen_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, algo_id, &algo);
    if (st == CUBLAS_STATUS_SUCCESS && tile) st = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
    if (st == CUBLAS_STATUS_SUCCESS && stages) st = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages));
    if (st == CUBLAS_STATUS_SUCCESS && swizzle) st = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle));
    const float alpha = 1.0f, beta = 0.0f;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmul(qwen_lt, op, &alpha, w->data, ad, x->data, bd,
        &beta, bf16_out, cd, bf16_out, dd, &algo, nullptr, 0, stream);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixTransformDescCreate(&to_f32, CUDA_R_32F);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixTransform(qwen_lt, to_f32, &alpha, bf16_out, cd,
        &beta, nullptr, nullptr, y->data, f32d, stream);
    if (to_f32) cublasLtMatrixTransformDescDestroy(to_f32);
    if (bf16_out) cudaFree(bf16_out);
    if (f32d) cublasLtMatrixLayoutDestroy(f32d);
    if (dd) cublasLtMatrixLayoutDestroy(dd);
    if (cd) cublasLtMatrixLayoutDestroy(cd);
    if (bd) cublasLtMatrixLayoutDestroy(bd);
    if (ad) cublasLtMatrixLayoutDestroy(ad);
    if (op) cublasLtMatmulDescDestroy(op);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    std::fprintf(stderr, "[QWEN_CUBLASLT] exact linear %dx%d N=%lld\n", in, out, (long long) n);
    return true;
}

static bool qwen_lt_hook(ggml_backend_cuda_context * ctx, const ggml_tensor * w,
                         const ggml_tensor * x, ggml_tensor * y, cudaStream_t stream) {
    return qwen_prefix_qkv_lt(ctx, w, x, y, stream) || qwen_o_proj_lt(ctx, w, x, y, stream);
}

extern "C" void qwen_cublaslt_hook_enable() {
    if (const char * e = std::getenv("QWEN_CUBLASLT_OPROJ"); e && e[0] != '\0' && e[0] != '0')
        ggml_cuda_set_mul_mat_hook(qwen_lt_hook);
}
