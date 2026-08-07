// Standalone, bounded feasibility benchmark for H3 direct ConvRot I8 GEMMs.
// It compares the production cublasGemmEx spelling against cublasLt's best
// cached heuristic for the four real 864x480 H3 projection shapes.  It is not
// linked into the engine or a production path.
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\\n", cudaGetErrorString(e)); return 2; } } while (0)
#define CUBLAS_OK(x) do { cublasStatus_t status_ = (x); if (status_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS: %d\\n", int(status_)); return 2; } } while (0)

struct Shape { const char * name; int m, n, k; };

static float time_gemmex(cublasHandle_t h, const Shape & s, const int8_t * a, const int8_t * b, int32_t * c, cudaStream_t stream) {
    const int32_t alpha = 1, beta = 0;
    for (int i = 0; i < 5; ++i) CUBLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, s.m, s.k, &alpha, a, CUDA_R_8I, s.k, b, CUDA_R_8I, s.k, &beta, c, CUDA_R_32I, s.n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaEvent_t begin, end; CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaEventRecord(begin, stream));
    for (int i = 0; i < 20; ++i) CUBLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, s.m, s.k, &alpha, a, CUDA_R_8I, s.k, b, CUDA_R_8I, s.k, &beta, c, CUDA_R_32I, s.n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_OK(cudaEventRecord(end, stream)); CUDA_OK(cudaEventSynchronize(end)); float ms; CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
    cudaEventDestroy(begin); cudaEventDestroy(end); return ms / 20;
}

static float time_cublaslt(cublasLtHandle_t lt, const Shape & s, const int8_t * a, const int8_t * b, int32_t * c, void * workspace, size_t workspace_size, cudaStream_t stream) {
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t ad, bd, cd;
    CUBLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&ad, CUDA_R_8I, s.k, s.n, s.k));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&bd, CUDA_R_8I, s.k, s.m, s.k));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&cd, CUDA_R_32I, s.n, s.m, s.n));
    cublasLtMatmulPreference_t pref; CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    cublasLtMatmulHeuristicResult_t heuristic{}; int returned = 0;
    CUBLAS_OK(cublasLtMatmulAlgoGetHeuristic(lt, op, ad, bd, cd, cd, pref, 1, &heuristic, &returned));
    if (returned != 1) { fprintf(stderr, "no cublasLt heuristic for %s\\n", s.name); return -1; }
    const int32_t alpha = 1, beta = 0;
    for (int i = 0; i < 5; ++i) CUBLAS_OK(cublasLtMatmul(lt, op, &alpha, a, ad, b, bd, &beta, c, cd, c, cd, &heuristic.algo, workspace, workspace_size, stream));
    cudaEvent_t begin, end; CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaEventRecord(begin, stream));
    for (int i = 0; i < 20; ++i) CUBLAS_OK(cublasLtMatmul(lt, op, &alpha, a, ad, b, bd, &beta, c, cd, c, cd, &heuristic.algo, workspace, workspace_size, stream));
    CUDA_OK(cudaEventRecord(end, stream)); CUDA_OK(cudaEventSynchronize(end)); float ms; CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
    cudaEventDestroy(begin); cudaEventDestroy(end); cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(ad); cublasLtMatrixLayoutDestroy(bd); cublasLtMatrixLayoutDestroy(cd); cublasLtMatmulDescDestroy(op); return ms / 20;
}

int main() {
    const Shape shapes[] = {{"qkv",7105,21504,5376},{"out",7105,5376,7168},{"fc1",7105,28672,5376},{"fc2",7105,5376,14336}};
    cublasHandle_t h; cublasLtHandle_t lt; CUBLAS_OK(cublasCreate(&h)); CUBLAS_OK(cublasLtCreate(&lt)); cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream)); CUBLAS_OK(cublasSetStream(h, stream));
    void * workspace = nullptr; constexpr size_t workspace_size = 64ull << 20; CUDA_OK(cudaMalloc(&workspace, workspace_size));
    for (const Shape & s : shapes) {
        int8_t * a; int8_t * b; int32_t * c0; int32_t * c1;
        CUDA_OK(cudaMalloc(&a, size_t(s.n)*s.k)); CUDA_OK(cudaMalloc(&b, size_t(s.m)*s.k)); CUDA_OK(cudaMalloc(&c0, size_t(s.n)*s.m*sizeof(int32_t))); CUDA_OK(cudaMalloc(&c1, size_t(s.n)*s.m*sizeof(int32_t)));
        CUDA_OK(cudaMemsetAsync(a, 1, size_t(s.n)*s.k, stream)); CUDA_OK(cudaMemsetAsync(b, 1, size_t(s.m)*s.k, stream));
        const float base_ms = time_gemmex(h, s, a, b, c0, stream); const float lt_ms = time_cublaslt(lt, s, a, b, c1, workspace, workspace_size, stream);
        std::vector<int32_t> host0(1024), host1(1024); CUDA_OK(cudaMemcpy(host0.data(), c0, host0.size()*sizeof(int32_t), cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(host1.data(), c1, host1.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
        int max_abs = 0; for (size_t i = 0; i < host0.size(); ++i) max_abs = std::max(max_abs, std::abs(host0[i]-host1[i]));
        printf("%s M=%d N=%d K=%d gemmex_ms=%.3f cublaslt_ms=%.3f change=%.1f%% sample_max_abs=%d\\n", s.name,s.m,s.n,s.k,base_ms,lt_ms,100*(lt_ms/base_ms-1),max_abs);
        cudaFree(a); cudaFree(b); cudaFree(c0); cudaFree(c1);
    }
    cudaFree(workspace); cudaStreamDestroy(stream); cublasLtDestroy(lt); cublasDestroy(h); return 0;
}
