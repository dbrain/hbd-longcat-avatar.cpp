// p3sam_heads_cuda.cu — GPU kernels for the P3-SAM seg/iou heads (see p3sam_heads.hpp host
// reference). The seg/iou heads are the entire CPU cost of segmentation (~30 min / 400 prompts
// over N=100k via the scalar host linear()); cuBLAS collapses the GEMMs and these kernels do the
// glue (build feats_seg / concat / GELU / per-column max / sigmoid-threshold) on resident buffers.
//
// All ops mirror the validated host path: nn.GELU() = exact-erf; cuBLAS fp32 under
// NVIDIA_TF32_OVERRIDE=0; per-column max in float (== host std::max). extern "C" launchers take
// DEVICE pointers; orchestration (p3sam_heads_gpu.hpp) owns malloc + cuBLAS handle.
#include <cstdint>
#include <cuda_runtime.h>

// ---- y[N,Cout] += bias[Cout] (row broadcast) ----
__global__ void p3_biasadd_kernel(float* __restrict__ y, const float* __restrict__ bias, int N, int Cout) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t tot = (size_t)N * Cout;
    if (i < tot) y[i] += bias[i % Cout];
}
extern "C" void p3_launch_biasadd(float* d_y, const float* d_bias, int N, int Cout, cudaStream_t s) {
    const int T = 256; size_t tot = (size_t)N * Cout;
    p3_biasadd_kernel<<<(unsigned)((tot + T - 1) / T), T, 0, s>>>(d_y, d_bias, N, Cout);
}

// ---- exact-erf GELU (== nn.GELU() default) ----
__global__ void p3_gelu_kernel(float* __restrict__ x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)); }
}
extern "C" void p3_launch_gelu(float* d_x, size_t n, cudaStream_t s) {
    const int T = 256; p3_gelu_kernel<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(d_x, n);
}

// ---- fill feats_seg cols 0..511 = feat, 512..514 = pts (once; prompt cols set per-prompt) ----
__global__ void p3_fs_base_kernel(float* __restrict__ fs, const float* __restrict__ feat,
                                  const float* __restrict__ pts, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float* d = fs + (size_t)n * 518;
    const float* f = feat + (size_t)n * 512;
    for (int i = 0; i < 512; i++) d[i] = f[i];
    d[512] = pts[n*3+0]; d[513] = pts[n*3+1]; d[514] = pts[n*3+2];
}
extern "C" void p3_launch_fs_base(float* fs, const float* feat, const float* pts, int N, cudaStream_t s) {
    const int T = 128; p3_fs_base_kernel<<<(N + T - 1) / T, T, 0, s>>>(fs, feat, pts, N);
}

// ---- set the 3 prompt cols (515..517) of feats_seg for the current prompt ----
__global__ void p3_fs_prompt_kernel(float* __restrict__ fs, int N, float p0, float p1, float p2) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float* d = fs + (size_t)n * 518;
    d[515] = p0; d[516] = p1; d[517] = p2;
}
extern "C" void p3_launch_fs_prompt(float* fs, int N, float p0, float p1, float p2, cudaStream_t s) {
    const int T = 128; p3_fs_prompt_kernel<<<(N + T - 1) / T, T, 0, s>>>(fs, N, p0, p1, p2);
}

// ---- scatter a [N,1] column src into column `col` of dst[N,3] ----
__global__ void p3_scatter_col3_kernel(float* __restrict__ dst, const float* __restrict__ src, int N, int col) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    dst[(size_t)n*3 + col] = src[n];
}
extern "C" void p3_launch_scatter_col3(float* dst, const float* src, int N, int col, cudaStream_t s) {
    const int T = 256; p3_scatter_col3_kernel<<<(N + T - 1) / T, T, 0, s>>>(dst, src, N, col);
}

// ---- feats_seg_2[N,521] = [fs(518), mask1(3)] ----
__global__ void p3_build_fs2_kernel(float* __restrict__ fs2, const float* __restrict__ fs,
                                    const float* __restrict__ mask1, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float* d = fs2 + (size_t)n * 521;
    const float* s = fs + (size_t)n * 518;
    for (int i = 0; i < 518; i++) d[i] = s[i];
    d[518] = mask1[n*3+0]; d[519] = mask1[n*3+1]; d[520] = mask1[n*3+2];
}
extern "C" void p3_launch_build_fs2(float* fs2, const float* fs, const float* mask1, int N, cudaStream_t st) {
    const int T = 128; p3_build_fs2_kernel<<<(N + T - 1) / T, T, 0, st>>>(fs2, fs, mask1, N);
}

// ---- feats_seg_3[N,777] = [gmax(256 bcast), fs2(521)] ----
__global__ void p3_build_fs3_kernel(float* __restrict__ fs3, const float* __restrict__ gmax,
                                    const float* __restrict__ fs2, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float* d = fs3 + (size_t)n * 777;
    for (int c = 0; c < 256; c++) d[c] = gmax[c];
    const float* s = fs2 + (size_t)n * 521;
    for (int i = 0; i < 521; i++) d[256 + i] = s[i];
}
extern "C" void p3_launch_build_fs3(float* fs3, const float* gmax, const float* fs2, int N, cudaStream_t st) {
    const int T = 128; p3_build_fs3_kernel<<<(N + T - 1) / T, T, 0, st>>>(fs3, gmax, fs2, N);
}

// ---- feats_iou[N,777] = [gmax(256), fs(518), mask2(3)] ----
__global__ void p3_build_fiou_kernel(float* __restrict__ fiou, const float* __restrict__ gmax,
                                     const float* __restrict__ fs, const float* __restrict__ mask2, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float* d = fiou + (size_t)n * 777;
    for (int c = 0; c < 256; c++) d[c] = gmax[c];
    const float* s = fs + (size_t)n * 518;
    for (int i = 0; i < 518; i++) d[256 + i] = s[i];
    d[256+518+0] = mask2[n*3+0]; d[256+518+1] = mask2[n*3+1]; d[256+518+2] = mask2[n*3+2];
}
extern "C" void p3_launch_build_fiou(float* fiou, const float* gmax, const float* fs,
                                     const float* mask2, int N, cudaStream_t st) {
    const int T = 128; p3_build_fiou_kernel<<<(N + T - 1) / T, T, 0, st>>>(fiou, gmax, fs, mask2, N);
}

// ---- per-column max over N rows: out[C] = max_n x[n,C]. one block per column. ----
template <int BLK>
__global__ void p3_colmax_kernel(const float* __restrict__ x, int N, int C, float* __restrict__ out) {
    int c = blockIdx.x;
    if (c >= C) return;
    __shared__ float sm[BLK];
    float m = -1e30f;
    for (int n = threadIdx.x; n < N; n += BLK) { float v = x[(size_t)n*C + c]; if (v > m) m = v; }
    sm[threadIdx.x] = m; __syncthreads();
    for (int s = BLK/2; s > 0; s >>= 1) { if (threadIdx.x < s && sm[threadIdx.x+s] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x+s]; __syncthreads(); }
    if (threadIdx.x == 0) out[c] = sm[0];
}
extern "C" void p3_launch_colmax(const float* d_x, int N, int C, float* d_out, cudaStream_t s) {
    p3_colmax_kernel<256><<<C, 256, 0, s>>>(d_x, N, C, d_out);
}

// ---- threshold: out_u8[n] = (logits[n*3+ch] > 0) ? 1 : 0 ----
__global__ void p3_threshold_kernel(const float* __restrict__ logits, int N, int ch, uint8_t* __restrict__ out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    out[n] = (logits[(size_t)n*3 + ch] > 0.0f) ? 1 : 0;
}
extern "C" void p3_launch_threshold(const float* d_logits, int N, int ch, uint8_t* d_out, cudaStream_t s) {
    const int T = 256; p3_threshold_kernel<<<(N + T - 1) / T, T, 0, s>>>(d_logits, N, ch, d_out);
}
