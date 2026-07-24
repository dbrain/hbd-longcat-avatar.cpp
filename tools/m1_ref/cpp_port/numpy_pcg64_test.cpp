// Exact regression fixture for the fixed-seed query sampler used by SkinTokens R1.
#include "mesh_sample.hpp"
#include <cstdio>

int main() {
    const int expected8[] = {6962, 616, 135, 4184, 2520, 2209, 335, 5214};
    const int expected32[] = {
        7075,2723,5487,4393,7979,1314,478,7678,6748,521,4778,1760,5433,8189,1904,1266,
        6834,1942,8071,2066,3418,2716,5623,5971,3191,4405,4056,4246,5770,1074,7406,6969};
    const auto a = rig::numpy_choice_seed0_without_replacement(8192, 8);
    const auto b = rig::numpy_choice_seed0_without_replacement(8192, 2048);
    if (a.size() != 8 || b.size() != 2048) return 1;
    for (int i = 0; i < 8; ++i) if (a[i] != expected8[i]) {
        std::fprintf(stderr, "choice(8192,8) mismatch at %d: got %d expected %d\n", i, a[i], expected8[i]); return 1;
    }
    for (int i = 0; i < 32; ++i) if (b[i] != expected32[i]) {
        std::fprintf(stderr, "choice(8192,2048) mismatch at %d: got %d expected %d\n", i, b[i], expected32[i]); return 1;
    }
    std::puts("numpy PCG64 choice seed=0: PASS");
    return 0;
}
