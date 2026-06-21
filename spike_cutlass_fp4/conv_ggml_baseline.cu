// ggml im2col+GEMM baseline microbench, SAME flux2 VAE decoder conv shapes, SAME GPU.
//
// Replicates ggml's conv2d path exactly:
//   (1) im2col_kernel<half>  : [N,IC,IH,IW] f32 -> [N,OH,OW,IC*KH*KW] f16  (3x3 s1 p1)
//       (the SAME kernel source ggml uses; copied verbatim from ggml/src/ggml-cuda/im2col.cu)
//   (2) cuBLAS hgemm          : col(M=N*OH*OW, K=IC*KH*KW) x W(K, OC) -> Y(M, OC) f16
//       (ggml does mul_mat(im2col_out, weight_2d); fp16xfp16->fp16, the nvjet_hhh path)
//
// Times the COMBINED im2col+GEMM us/call (median over ~50 iters, cudaEvents), which
// is what the VAE pays per conv. The im2col write traffic (IC*9 HBM blowup) is the cost
// cuDNN's implicit GEMM avoids -- this baseline makes that explicit.
//
// Build: nvcc -O3 -arch=sm_120 -std=c++17 conv_ggml_baseline.cu -o conv_ggml_baseline -lcublas
// Run pinned to the 5060 Ti.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CBL(x) do{ cublasStatus_t s=(x); if(s){fprintf(stderr,"cuBLAS err %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535
#define CUDA_IM2COL_BLOCK_SIZE 256

// ---- verbatim ggml im2col_kernel (ggml/src/ggml-cuda/im2col.cu) ----
template <typename T>
static __global__ void im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) return;
    const int64_t iic = i / (KH_KW);
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;
    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t in = iz / OH;
            const int64_t ioh = iz - in * OH;
            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t offset_dst = ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;
            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) dst[offset_dst] = 0.0f;
            else { const int64_t offset_src = iic * IC_IH_IW + in * IH_IW; dst[offset_dst] = x[offset_src + iih * IW + iiw]; }
        }
    }
    (void)IC; (void)KH;
}

struct Shape { int Cin, Cout, H, W; const char* tag; };

static void bench(cublasHandle_t cb, const Shape& sh, int iters) {
    const int N = 1, KH = 3, KW = 3, s0 = 1, s1 = 1, p0 = 1, p1 = 1, d0 = 1, d1 = 1;
    const int IC = sh.Cin, OC = sh.Cout, IH = sh.H, IW = sh.W;
    const int OH = IH, OW = IW;                 // s1p1 -> same spatial
    const int64_t M = (int64_t)N * OH * OW;     // rows = output pixels
    const int64_t Kdim = (int64_t)IC * KH * KW; // cols of im2col
    fprintf(stderr, "=== ggml im2col+GEMM  %s  Cin=%d Cout=%d %dx%d  M=%ld K=%ld ===\n",
            sh.tag, IC, OC, IH, IW, M, Kdim);

    size_t xn = (size_t)N * IC * IH * IW;
    size_t coln = (size_t)M * Kdim;             // im2col output
    size_t wn = (size_t)Kdim * OC;              // weight reshaped [K, OC]
    size_t yn = (size_t)M * OC;

    float* dX; __half *dCol, *dW, *dY;
    CK(cudaMalloc(&dX, xn * 4)); CK(cudaMalloc(&dCol, coln * 2));
    CK(cudaMalloc(&dW, wn * 2)); CK(cudaMalloc(&dY, yn * 2));

    // fill X and W
    std::mt19937 rng(99 + IC); std::normal_distribution<float> nd(0, 1);
    { std::vector<float> h(xn); for (auto& v : h) v = nd(rng) * 0.3f; CK(cudaMemcpy(dX, h.data(), xn * 4, cudaMemcpyHostToDevice)); }
    { std::vector<__half> h(wn); for (auto& v : h) v = __float2half(nd(rng) * 0.05f); CK(cudaMemcpy(dW, h.data(), wn * 2, cudaMemcpyHostToDevice)); }

    const int64_t IC_KH_KW = Kdim, N_OH = (int64_t)N * OH, KH_KW = KH * KW;
    // ggml passes IC_IH_IW = src1->nb[2]/4 = per-CHANNEL stride (IH*IW), and
    // IH_IW = src1->nb[3]/4 = per-BATCH stride (IC*IH*IW). The names are ggml's; the
    // kernel does offset_src = iic*IC_IH_IW + in*IH_IW, so IC_IH_IW must be IH*IW.
    const int64_t IC_IH_IW = (int64_t)IH * IW, IH_IW = (int64_t)IC * IH * IW;
    const int64_t num_blocks = (IC_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 grid(num_blocks, std::min<int64_t>(OW, MAX_GRIDDIM_Y), std::min<int64_t>(N_OH, MAX_GRIDDIM_Z));
    int tpb = std::min<int64_t>(IC_KH_KW, CUDA_IM2COL_BLOCK_SIZE);

    auto run_im2col = [&]() {
        im2col_kernel<__half><<<grid, tpb>>>(dX, dCol, IC, IW, IH, OH, OW, KW, KH,
            IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW, s0, s1, p0, p1, d0, d1);
    };
    // GEMM: Y(M x OC) = Col(M x K) * W(K x OC), all row-major.
    // cuBLAS col-major: compute Y^T(OC x M) = W^T(OC x K) * Col^T(K x M)
    //   => C(OC,M) = op_N(W as K x OC col-major = OC outer) ... use:
    //   cublasHgemm(N,N, OC, M, K, W(lda=OC? ), Col(...)) — pick the standard row-major trick:
    //   Treat row-major A(MxK),B(KxN) producing C(MxN): call gemm(N,N, N,M,K, B, N, A, K, C, N).
    // Here A=Col(MxK), B=W(KxOC), N=OC. So: gemm(N,N, OC,M,K, W(ldb=OC), Col(lda=K), Y(ldc=OC)).
    // fp16 IO, fp32 accumulate (matches ggml conv GEMM numerics), via cublasGemmEx.
    float alpha = 1.f, beta = 0.f;
    auto run_gemm = [&]() {
        cublasStatus_t st = cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int)OC, (int)M, (int)Kdim,
                         &alpha, dW, CUDA_R_16F, (int)OC, dCol, CUDA_R_16F, (int)Kdim,
                         &beta, dY, CUDA_R_16F, (int)OC,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st) { fprintf(stderr, "GemmEx status=%d  OC=%d M=%ld K=%ld\n", (int)st, (int)OC, M, Kdim); exit(1); }
    };

    // warmup
    run_im2col();
    cudaError_t ie = cudaDeviceSynchronize();
    if (ie) { fprintf(stderr, "im2col launch err: %s (grid %u,%u,%u tpb %d)\n", cudaGetErrorString(ie), grid.x, grid.y, grid.z, tpb); exit(1); }
    run_gemm(); CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1, em; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&em));
    std::vector<float> t_comb(iters), t_im(iters), t_gemm(iters);
    for (int it = 0; it < iters; it++) {
        CK(cudaEventRecord(e0));
        run_im2col();
        CK(cudaEventRecord(em));
        run_gemm();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float a = 0, b = 0; CK(cudaEventElapsedTime(&a, e0, em)); CK(cudaEventElapsedTime(&b, em, e1));
        t_im[it] = a; t_gemm[it] = b; t_comb[it] = a + b;
    }
    auto med = [&](std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    double mc = med(t_comb), mi = med(t_im), mg = med(t_gemm);
    double flops = 2.0 * N * (double)OC * OH * OW * IC * KH * KW;
    double tflops = flops / (mc * 1e-3) / 1e12;
    // im2col write traffic (dominant): M*K*2 bytes
    double wbytes = (double)coln * 2.0;
    double gbps = (wbytes / 1e9) / (mi * 1e-3);

    fprintf(stderr, "  combined %.1f us (im2col %.1f + gemm %.1f)  %.1f TFLOP/s  im2col-write %.0f GB/s\n",
            mc * 1e3, mi * 1e3, mg * 1e3, tflops, gbps);
    printf("{\"kernel\":\"ggml_im2col_gemm\",\"tag\":\"%s\",\"Cin\":%d,\"Cout\":%d,\"H\":%d,\"W\":%d,"
           "\"comb_us\":%.2f,\"im2col_us\":%.2f,\"gemm_us\":%.2f,\"tflops\":%.2f,\"im2col_gbps\":%.0f}\n",
           sh.tag, IC, OC, IH, IW, mc * 1e3, mi * 1e3, mg * 1e3, tflops, gbps);
    fflush(stdout);

    cudaFree(dX); cudaFree(dCol); cudaFree(dW); cudaFree(dY);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(em);
}

int main(int argc, char** argv) {
    int iters = 50;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    cublasHandle_t cb; CBL(cublasCreate(&cb));
    void* ws = nullptr; CK(cudaMalloc(&ws, 64u * 1024 * 1024)); CBL(cublasSetWorkspace(cb, ws, 64u * 1024 * 1024));
    std::vector<Shape> shapes = {
        {384, 384, 128, 128, "mid/up3 384@128"},
        {384, 384, 256, 256, "up2 384@256"},
        {192, 192, 512, 512, "up1 192@512"},
        {384, 384, 512, 512, "up1in/ups 384@512"},
        {96,  96, 1024, 1024, "up0 96@1024"},
        {192, 96, 1024, 1024, "up0in 192@1024"},
    };
    for (auto& sh : shapes) bench(cb, sh, iters);
    cublasDestroy(cb);
    return 0;
}
