// Replays PyTorch's captured layer-0 Qwen o_proj cuBLASLt call exactly.
#include "../../sparse_spike/npy.hpp"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

static uint16_t fp32_to_bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    // IEEE round-to-nearest-even, matching torch.bfloat16 conversion.
    u += 0x7fffU + ((u >> 16) & 1U);
    return (uint16_t) (u >> 16);
}
static float bf16_to_fp32(uint16_t x) {
    uint32_t u = (uint32_t) x << 16; float r; std::memcpy(&r, &u, sizeof(r)); return r;
}

static void ck(cudaError_t s, const char * what) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
}
static void bk(cublasStatus_t s, const char * what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string((int) s));
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <layer0_f32.npz> <qwen3_weights_dir>\n", argv[0]);
        return 2;
    }
    // npz is intentionally read by a tiny Python export wrapper; this replay consumes its .npy members.
    const std::string dir = argv[1];
    const char * in_name = std::getenv("QWEN_LT_INPUT");
    const char * out_name = std::getenv("QWEN_LT_OUTPUT");
    const char * weight_name = std::getenv("QWEN_LT_WEIGHT");
    if (!in_name) in_name = "attn_pre_o.npy";
    if (!out_name) out_name = "o_proj.npy";
    if (!weight_name) weight_name = "transformer.model.layers.0.self_attn.o_proj.weight.npy";
    NpyArray in = npy_load(dir + "/" + in_name);
    NpyArray want = npy_load(dir + "/" + out_name);
    NpyArray w = npy_load(std::string(argv[2]) + "/" + weight_name);
    if (in.shape.empty() || want.shape.empty() || w.shape.size() != 2)
        throw std::runtime_error("unexpected replay tensor shape");
    const int64_t k = in.shape.back(), m = want.shape.back();
    if (k != w.shape[1] || m != w.shape[0] || in.numel() % k != 0 || want.numel() != in.numel() / k * m)
        throw std::runtime_error("incompatible replay tensor shape");
    const int64_t n = (int64_t) in.numel() / k;

    const bool f32_out = std::getenv("QWEN_LT_F32_OUTPUT") != nullptr;
    std::vector<uint16_t> hb_w(w.numel()), hb_x(in.numel()), hb_y(want.numel());
    std::vector<float> hf_y(f32_out ? want.numel() : 0);
    for (size_t i = 0; i < hb_w.size(); ++i) hb_w[i] = fp32_to_bf16(w.f32()[i]);
    for (size_t i = 0; i < hb_x.size(); ++i) hb_x[i] = fp32_to_bf16(in.f32()[i]);
    uint16_t * db_w = nullptr, * db_x = nullptr;
    void * db_y = nullptr;
    ck(cudaMalloc(&db_w, hb_w.size() * sizeof(*db_w)), "cudaMalloc weight");
    ck(cudaMalloc(&db_x, hb_x.size() * sizeof(*db_x)), "cudaMalloc input");
    ck(cudaMalloc(&db_y, want.numel() * (f32_out ? sizeof(float) : sizeof(uint16_t))), "cudaMalloc output");
    ck(cudaMemcpy(db_w, hb_w.data(), hb_w.size() * sizeof(*db_w), cudaMemcpyHostToDevice), "copy weight");
    ck(cudaMemcpy(db_x, hb_x.data(), hb_x.size() * sizeof(*db_x), cudaMemcpyHostToDevice), "copy input");

    cublasLtHandle_t h = nullptr; bk(cublasLtCreate(&h), "cublasLtCreate");
    cublasLtMatmulDesc_t op = nullptr;
    bk(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "op create");
    cublasOperation_t transa = CUBLAS_OP_T;
    bk(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "set transa");
    int sm = 28;
    bk(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm, sizeof(sm)), "set sm target");
    cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr, dd = nullptr;
    bk(cublasLtMatrixLayoutCreate(&ad, CUDA_R_16BF, k, m, k), "A layout");
    bk(cublasLtMatrixLayoutCreate(&bd, CUDA_R_16BF, k, n, k), "B layout");
    const cudaDataType_t out_type = f32_out ? CUDA_R_32F : CUDA_R_16BF;
    bk(cublasLtMatrixLayoutCreate(&cd, out_type, m, n, m), "C layout");
    bk(cublasLtMatrixLayoutCreate(&dd, out_type, m, n, m), "D layout");
    cublasLtMatmulAlgo_t algo{};
    const int algo_id = (k == 896 && m == 2048) ? 6 : 21;
    bk(cublasLtMatmulAlgoInit(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, out_type, out_type, algo_id, &algo), "algo init");
    uint32_t tile = (k == 896 && m == 2048) ? CUBLASLT_MATMUL_TILE_256x128 : (n == 1 ? CUBLASLT_MATMUL_TILE_64x64 : CUBLASLT_MATMUL_TILE_128x256);
    uint32_t stages = n == 1 ? CUBLASLT_MATMUL_STAGES_32x6 : CUBLASLT_MATMUL_STAGES_32x3;
    bk(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile)), "set tile");
    bk(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages)), "set stages");
    const float alpha = 1.0f, beta = 0.0f;
    bk(cublasLtMatmul(h, op, &alpha, db_w, ad, db_x, bd, &beta, db_y, cd, db_y, dd, &algo, nullptr, 0, 0), "matmul");
    if (f32_out) ck(cudaMemcpy(hf_y.data(), db_y, hf_y.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy output");
    else ck(cudaMemcpy(hb_y.data(), db_y, hb_y.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "copy output");
    double mae = 0, mse = 0, mx = 0;
    for (size_t i = 0; i < hb_y.size(); ++i) {
        const float got = f32_out ? bf16_to_fp32(fp32_to_bf16(hf_y[i])) : bf16_to_fp32(hb_y[i]);
        const double e = (double) got - want.f32()[i];
        mae += std::fabs(e); mse += e * e; mx = std::max(mx, std::fabs(e));
    }
    std::printf("QWEN_CUBLASLT_LINEAR_REPLAY K=%lld M=%lld N=%lld f32out=%d mae=%.9g rmse=%.9g maxabs=%.9g\n", (long long) k, (long long) m, (long long) n, (int) f32_out, mae / want.numel(), std::sqrt(mse / want.numel()), mx);
    cublasLtMatrixLayoutDestroy(dd); cublasLtMatrixLayoutDestroy(cd); cublasLtMatrixLayoutDestroy(bd); cublasLtMatrixLayoutDestroy(ad);
    cublasLtMatmulDescDestroy(op); cublasLtDestroy(h); cudaFree(db_y); cudaFree(db_x); cudaFree(db_w);
}
