// CPU mirror of sparse_subm_conv.cu — IDENTICAL math (fp32 accumulate, loop
// v then ci, per output channel) so we can (a) validate the exact kernel
// interface/dataflow today without nvcc, and (b) predict the CUDA kernel's
// accuracy vs the f64 goldens BEFORE running on GPU.
//
// NOTE: fp32 accumulate ≠ the f64 golden bit-exactly — expect ~1e-5 rel error.
// That is the CORRECT expectation for the GPU kernel too (it also accumulates
// fp32; flex_gemm itself runs fp16/tf32, so our fp32 path is *more* accurate).
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

static const uint32_t SUBM_SENTINEL_CPU = 0xFFFFFFFFu;

// Same signature shape as launch_subm_conv_cuda, but host pointers + returns vec.
inline std::vector<float> launch_subm_conv_cpu(
    const float* feats, const uint32_t* nmap, const float* weight,
    const float* bias, int N, int Cin, int Cout, int V)
{
    std::vector<float> out((size_t)N * Cout);
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < Cout; co++) {
            float acc = 0.0f;                       // fp32 accumulate (matches kernel)
            for (int v = 0; v < V; v++) {
                uint32_t j = nmap[(size_t)n * V + v];
                if (j == SUBM_SENTINEL_CPU) continue;
                const float* fj = feats + (size_t)j * Cin;
                const float* wv = weight + (size_t)v * Cin * Cout;  // [Cin,Cout]
                float a = 0.0f;
                for (int ci = 0; ci < Cin; ci++)
                    a += fj[ci] * wv[(size_t)ci * Cout + co];
                acc += a;
            }
            out[(size_t)n * Cout + co] = acc + (bias ? bias[co] : 0.0f);
        }
    }
    return out;
}
