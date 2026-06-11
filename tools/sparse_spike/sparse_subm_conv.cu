// Submanifold sparse 3D conv — CUDA implicit-GEMM (Rung 1: correct, fp32, simple).
//
// Mirrors FlexGEMM's implicit-GEMM dataflow (gather neighbours via the rulebook
// INSIDE the accumulation — no im2col buffer) but in plain CUDA C, one output
// tile per block, fp32 accumulate. NO tensor cores yet (Rung 2) — this rung is
// for correctness + establishing the GPU path; optimize once it's measurable
// against the flex_gemm baseline (golden_model/flexgemm_timing.json).
//
// Validated against the SAME goldens as the CPU ref (incl. neighbor_map.npy),
// so build-and-run in the GPU session is a direct bit/tol check vs out_feats.npy.
//
// Interface is backend-agnostic: launch_subm_conv_{cuda,cpu} take identical args
// so test_subm.cpp can exercise either. Host has no nvcc → the .cu compiles in
// the docker builder; the CPU mirror in sparse_subm_conv_cpu.hpp runs today.
//
// Canonical layouts (match sparse_conv_ref.py / sparse_conv.cpp):
//   feats   f32 [N, Cin]
//   nmap    u32 [N, V]   (V = K^3 ≤ 32), entry = input idx or 0xFFFFFFFF
//   weight  f32 [V, Cin, Cout]
//   bias    f32 [Cout]
//   out     f32 [N, Cout]
#include <cstdint>
#include <cuda_runtime.h>

#define SUBM_SENTINEL 0xFFFFFFFFu

// One block handles a tile of TN output voxels for ALL Cout (threads = Cout-ish).
// Each thread owns a set of output channels for the tile's voxels.
// Reduction = over V kernel taps × Cin, gathering feats[nmap[n,v]] on the fly.
template <int TN>
__global__ void subm_conv_implicit_kernel(
    const float* __restrict__ feats,     // [N, Cin]
    const uint32_t* __restrict__ nmap,   // [N, V]
    const float* __restrict__ weight,    // [V, Cin, Cout]
    const float* __restrict__ bias,      // [Cout] or null
    float* __restrict__ out,             // [N, Cout]
    int N, int Cin, int Cout, int V)
{
    const int n0 = blockIdx.x * TN;                 // first voxel of this tile
    const int co = blockIdx.y * blockDim.x + threadIdx.x;  // output channel
    if (co >= Cout) return;

    float acc[TN];
#pragma unroll
    for (int t = 0; t < TN; t++) acc[t] = 0.0f;

    // K-loop: over kernel taps v, then over Cin. Gather is implicit (via nmap).
    for (int v = 0; v < V; v++) {
        const float* wv = weight + (size_t)v * Cin * Cout;   // [Cin, Cout]
        for (int t = 0; t < TN; t++) {
            int n = n0 + t;
            if (n >= N) continue;
            uint32_t j = nmap[(size_t)n * V + v];
            if (j == SUBM_SENTINEL) continue;
            const float* fj = feats + (size_t)j * Cin;
            float a = 0.0f;
            for (int ci = 0; ci < Cin; ci++)
                a += fj[ci] * wv[(size_t)ci * Cout + co];   // weight[v,ci,co]
            acc[t] += a;
        }
    }
    float b = bias ? bias[co] : 0.0f;
    for (int t = 0; t < TN; t++) {
        int n = n0 + t;
        if (n < N) out[(size_t)n * Cout + co] = acc[t] + b;
    }
}

// Host launcher. Pointers are DEVICE pointers. Caller owns the rulebook (nmap).
extern "C" void launch_subm_conv_cuda(
    const float* d_feats, const uint32_t* d_nmap, const float* d_weight,
    const float* d_bias, float* d_out, int N, int Cin, int Cout, int V,
    cudaStream_t stream)
{
    constexpr int TN = 4;
    constexpr int THREADS = 128;
    dim3 grid((N + TN - 1) / TN, (Cout + THREADS - 1) / THREADS);
    subm_conv_implicit_kernel<TN><<<grid, THREADS, 0, stream>>>(
        d_feats, d_nmap, d_weight, d_bias, d_out, N, Cin, Cout, V);
}
