// GPU-resident dense ops for the sparse-VAE decode (Phase C #1 lever).
// Mirrors the host svae:: ops (sparse_vae.hpp) so the decode can keep `feats` RESIDENT
// on the GPU across all 4 levels — only subdiv decisions (and the final head/coords) get
// pulled back to host. Numerically matched to the host path: layernorm reduces in double
// (host uses double); silu/add elementwise in float (== host).
//
// extern "C" launchers take DEVICE pointers; the orchestration (svp_gpu.hpp) owns malloc.
#include <cstdint>
#include <cuda_runtime.h>

// ---- LayerNorm over C (population mean/var, double accum), optional affine. one block / row ----
template <int BLK>
__global__ void layernorm_kernel(float* __restrict__ x, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, int N, int C, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    float* r = x + (size_t)row * C;
    __shared__ double s_sum[BLK];
    __shared__ double s_sq[BLK];
    double ls = 0, lq = 0;
    for (int c = threadIdx.x; c < C; c += BLK) { double v = r[c]; ls += v; lq += v * v; }
    s_sum[threadIdx.x] = ls; s_sq[threadIdx.x] = lq;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { s_sum[threadIdx.x] += s_sum[threadIdx.x + s]; s_sq[threadIdx.x] += s_sq[threadIdx.x + s]; }
        __syncthreads();
    }
    __shared__ double mean, inv;
    if (threadIdx.x == 0) {
        mean = s_sum[0] / C;
        double var = s_sq[0] / C - mean * mean;
        inv = 1.0 / sqrt(var + (double)eps);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += BLK) {
        double v = ((double)r[c] - mean) * inv;
        if (gamma) v = v * (double)gamma[c] + (double)beta[c];
        r[c] = (float)v;
    }
}

extern "C" void launch_layernorm(float* d_x, const float* d_gamma, const float* d_beta,
                                 int N, int C, float eps, cudaStream_t stream) {
    constexpr int BLK = 256;
    layernorm_kernel<BLK><<<N, BLK, 0, stream>>>(d_x, d_gamma, d_beta, N, C, eps);
}

// ---- SiLU in place (== host: v/(1+exp(-v)), float) ----
__global__ void silu_kernel(float* __restrict__ x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v / (1.0f + __expf(-v)); }
}
extern "C" void launch_silu(float* d_x, size_t n, cudaStream_t stream) {
    int T = 256; silu_kernel<<<(unsigned)((n + T - 1) / T), T, 0, stream>>>(d_x, n);
}

// ---- a += b (elementwise) ----
__global__ void add_kernel(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
extern "C" void launch_add(float* d_a, const float* d_b, size_t n, cudaStream_t stream) {
    int T = 256; add_kernel<<<(unsigned)((n + T - 1) / T), T, 0, stream>>>(d_a, d_b, n);
}

// ---- bias add: y[n*Cout + o] += bias[o]  (broadcast over rows) ----
__global__ void biasadd_kernel(float* __restrict__ y, const float* __restrict__ bias, int N, int Cout) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t tot = (size_t)N * Cout;
    if (i < tot) y[i] += bias[i % Cout];
}
extern "C" void launch_biasadd(float* d_y, const float* d_bias, int N, int Cout, cudaStream_t stream) {
    int T = 256; size_t tot = (size_t)N * Cout;
    biasadd_kernel<<<(unsigned)((tot + T - 1) / T), T, 0, stream>>>(d_y, d_bias, N, Cout);
}

// ---- gather_children: out[m, c] = src[(parent[m]*8 + child[m]) * per + c] ----
__global__ void gather_children_kernel(const float* __restrict__ src, const int* __restrict__ parent,
                                       const int* __restrict__ child, float* __restrict__ out,
                                       int M, int per) {
    int m = blockIdx.x;
    if (m >= M) return;
    const float* sp = src + ((size_t)parent[m] * 8 + child[m]) * per;
    float* op = out + (size_t)m * per;
    for (int c = threadIdx.x; c < per; c += blockDim.x) op[c] = sp[c];
}
extern "C" void launch_gather_children(const float* d_src, const int* d_parent, const int* d_child,
                                       float* d_out, int M, int per, cudaStream_t stream) {
    int T = per < 256 ? ((per + 31) / 32) * 32 : 256; if (T < 32) T = 32;
    gather_children_kernel<<<M, T, 0, stream>>>(d_src, d_parent, d_child, d_out, M, per);
}

// ---- repeat_interleave along channel: out[m, i*factor + r] = x[m, i] ----
__global__ void repeat_interleave_kernel(const float* __restrict__ x, float* __restrict__ out,
                                         int M, int c, int factor) {
    int m = blockIdx.x;
    if (m >= M) return;
    const float* xr = x + (size_t)m * c;
    float* op = out + (size_t)m * c * factor;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        float v = xr[i];
        for (int rr = 0; rr < factor; rr++) op[i * factor + rr] = v;
    }
}
extern "C" void launch_repeat_interleave(const float* d_x, float* d_out, int M, int c, int factor,
                                         cudaStream_t stream) {
    int T = c < 256 ? ((c + 31) / 32) * 32 : 256; if (T < 32) T = 32;
    repeat_interleave_kernel<<<M, T, 0, stream>>>(d_x, d_out, M, c, factor);
}
