// Native counterpart to PyTorch 2.7 CUDA exponential_(1).  Keep launch and
// Philox offset accounting in lockstep with DistributionTemplates.h.
#include "rig_philox_race.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace {
constexpr int kBlock = 256;
constexpr uint64_t kCurandOffset = 4;

__global__ void exponential_kernel(float* out, int64_t n, uint64_t seed, uint64_t offset) {
    const int64_t lane = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)lane, (unsigned long long)offset, &state);
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t rounded = ((n - 1) / (stride * 4) + 1) * stride * 4;
    for (int64_t base = lane; base < rounded; base += stride * 4) {
        const float4 u = curand_uniform4(&state);
        const float v[4] = { -logf(u.x), -logf(u.y), -logf(u.z), -logf(u.w) };
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (base + stride * j < n) out[base + stride * j] = v[j];
    }
}
}

extern "C" bool rig_torch_philox_exponentials(std::vector<float>& out, uint64_t seed, uint64_t* offset) {
    if (!offset || out.empty()) return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    const uint64_t n = out.size();
    const uint64_t max_grid = (uint64_t)prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / kBlock);
    const uint64_t grid = std::min((n + kBlock - 1) / kBlock, max_grid);
    float* device = nullptr;
    if (cudaMalloc(&device, n * sizeof(float)) != cudaSuccess) return false;
    exponential_kernel<<<(unsigned)grid, kBlock>>>(device, (int64_t)n, seed, *offset);
    const cudaError_t launch = cudaGetLastError();
    const cudaError_t copy = launch == cudaSuccess ? cudaMemcpy(out.data(), device, n * sizeof(float), cudaMemcpyDeviceToHost) : launch;
    cudaFree(device);
    if (copy != cudaSuccess) return false;
    // DistributionTemplates::calc_execution_policy reserves one curand4
    // invocation for every per-thread grid-stride group.
    *offset += ((n - 1) / ((uint64_t)kBlock * grid * 4) + 1) * kCurandOffset;
    return true;
}
