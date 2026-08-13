// Native CUDA parity fixture for torch 2.7 CUDA float exponential_(1) with
// torch.Generator(device="cuda").manual_seed(0).  PyTorch's distribution
// kernel assigns one Philox state per CUDA thread and writes curand_uniform4
// in grid-stride groups.  This test keeps that mapping explicit so the
// sampled SkinTokens beam can consume the same exponential-race stream rather
// than std::mt19937_64.
//
// Build/run (3060 only):
//   ./build.sh philox_exponential_test cuda && CUDA_VISIBLE_DEVICES=GPU-... ./philox_exponential_test
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kN = 1024;
constexpr unsigned long long kSeed = 0;

__global__ void philox_exponential4(float* out, int n, unsigned long long seed,
                                    unsigned long long offset) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    // `lane` is PyTorch's thread index / Philox subsequence.  Note that the
    // four outputs are STRIDED by gridDim.x*blockDim.x, not adjacent.
    curand_init(seed, lane, offset, &state);
    const int stride = blockDim.x * gridDim.x;
    const int rounded = ((n - 1) / (stride * 4) + 1) * stride * 4;
    for (int base = lane; base < rounded; base += stride * 4) {
        const float4 u = curand_uniform4(&state);
        const float values[4] = { -logf(u.x), -logf(u.y), -logf(u.z), -logf(u.w) };
        for (int j = 0; j < 4 && base + stride * j < n; ++j) out[base + stride * j] = values[j];
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

} // namespace

int main() {
    // Captured immediately before this fixture from the installed torch
    // 2.7.0+cu128 on the assigned RTX 3060:
    // torch.empty(1024, device='cuda').exponential_(1, generator=seed0).
    constexpr float first_expected[] = {
        0.918677270f, 0.660333097f, 3.69166756f, 0.0617909096f,
        0.0556669161f, 0.227237836f, 0.879442692f, 0.198137566f,
        1.47384369f, 0.0947640240f, 2.13422132f, 2.58730626f,
    };
    // The next call after any first exponential_ size in [1, 10240] begins at
    // offset 4 on this RTX 3060.  That is PyTorch's reservation unit: one
    // curand_uniform4 call, not the tensor's scalar element count.
    constexpr float second_expected[] = {
        0.0281513054f, 0.234496087f, 0.757065475f, 1.10866296f,
        1.09517968f, 0.972135365f, 0.269208252f, 0.445528269f,
        2.20490146f, 1.02344978f, 1.58235323f, 0.620854616f,
    };

    float* device = nullptr;
    check(cudaMalloc(&device, kN * sizeof(float)), "cudaMalloc");
    constexpr int kThreads = 256;
    // Mirrors DistributionTemplates::calc_execution_policy: block 256; grid
    // clamped to SM_count*(maxThreadsPerSM/256).  kN=1024 therefore uses 4
    // blocks, and reserves counter offset 4 for the following call.
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const int grid = std::min((kN + kThreads - 1) / kThreads,
                              prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / kThreads));
    philox_exponential4<<<grid, kThreads>>>(device, kN, kSeed, 0);
    check(cudaGetLastError(), "philox_exponential4 launch");
    std::vector<float> got(kN);
    check(cudaMemcpy(got.data(), device, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");

    float max_abs = 0.0f;
    for (size_t i = 0; i < sizeof(first_expected) / sizeof(first_expected[0]); ++i) {
        const float d = std::fabs(got[i] - first_expected[i]);
        if (d > max_abs) max_abs = d;
        std::printf("philox_exp[first,%zu] native=%.9g torch=%.9g abs=%.3g\n", i, got[i], first_expected[i], d);
    }
    philox_exponential4<<<grid, kThreads>>>(device, kN, kSeed, 4);
    check(cudaGetLastError(), "philox_exponential4 second launch");
    check(cudaMemcpy(got.data(), device, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy second");
    for (size_t i = 0; i < sizeof(second_expected) / sizeof(second_expected[0]); ++i) {
        const float d = std::fabs(got[i] - second_expected[i]);
        if (d > max_abs) max_abs = d;
        std::printf("philox_exp[next,%zu] native=%.9g torch=%.9g abs=%.3g\n", i, got[i], second_expected[i], d);
    }
    check(cudaFree(device), "cudaFree");

    // Exercise the actual four-wide unroll: on this 28-SM RTX 3060 the grid
    // caps at 168 blocks (stride 43008), so index 43008 is rand.y from state
    // 0 rather than an adjacent output of that state.  Values were captured
    // from torch.empty(65536, device='cuda').exponential_(1, generator=seed0).
    constexpr int kLargeN = 65536;
    constexpr int kProbeIndex[] = {0, 1, 2, 255, 256, 43007, 43008, 43009, 43010, 65535};
    constexpr float kProbeExpected[] = {
        0.918677270f, 0.660333097f, 3.69166756f, 0.358651608f, 2.45374346f,
        3.48908305f, 0.127242342f, 0.0622391477f, 0.338375837f, 1.28703296f,
    };
    float* large_device = nullptr;
    check(cudaMalloc(&large_device, kLargeN * sizeof(float)), "cudaMalloc large");
    const int large_grid = std::min((kLargeN + kThreads - 1) / kThreads,
                                    prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / kThreads));
    philox_exponential4<<<large_grid, kThreads>>>(large_device, kLargeN, kSeed, 0);
    check(cudaGetLastError(), "philox_exponential4 large launch");
    std::vector<float> large(kLargeN);
    check(cudaMemcpy(large.data(), large_device, large.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy large");
    check(cudaFree(large_device), "cudaFree large");
    if (large_grid * kThreads != 43008) {
        std::fprintf(stderr, "FAIL: unexpected RTX 3060 launch stride %d (expected 43008)\n", large_grid * kThreads);
        return 1;
    }
    for (size_t i = 0; i < sizeof(kProbeIndex) / sizeof(kProbeIndex[0]); ++i) {
        const int at = kProbeIndex[i];
        const float d = std::fabs(large[at] - kProbeExpected[i]);
        if (d > max_abs) max_abs = d;
        std::printf("philox_exp[unroll,%d] native=%.9g torch=%.9g abs=%.3g\n", at, large[at], kProbeExpected[i], d);
    }
    // Float transcendental implementations differ by at most a few ulps; the
    // crucial parity invariant is Philox state/lane order, validated here.
    if (max_abs > 2.0e-6f) {
        std::fprintf(stderr, "FAIL: torch CUDA exponential mapping mismatch (max_abs=%.9g)\n", max_abs);
        return 1;
    }
    std::printf("PASS: torch CUDA exponential seed=0: curand_init(seed,thread_idx,offset), curand_uniform4 strided by grid*block; first call offset=0, next offset=4 (max_abs=%.3g)\n", max_abs);
    return 0;
}
