// Validate p3sam_serialize.hpp against the ser_order_s0 / ser_inverse_s0 goldens
// (4 orders x M0). Build: g++ -O2 -std=c++17 test_p3sam_serialize.cpp -o test_p3sam_serialize
#include "p3sam_serialize.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    std::string G = argc > 1 ? argv[1] : "/mnt/hdd/3d/avatar-shootout/p3sam_goldens";
    auto grid = npy_load(G + "/grid_s0.npy");        // (M,3) int32
    auto gord = npy_load(G + "/ser_order_s0.npy");   // (4,M) int64
    auto ginv = npy_load(G + "/ser_inverse_s0.npy"); // (4,M) int64
    int M = (int)grid.shape[0];
    int O = (int)gord.shape[0];
    printf("M=%d orders=%d\n", M, O);
    const int32_t* g = grid.i32();
    const int64_t* go = reinterpret_cast<const int64_t*>(gord.raw.data());
    const int64_t* gi = reinterpret_cast<const int64_t*>(ginv.raw.data());
    const char* names[4] = {"z", "z-trans", "hilbert", "hilbert-trans"};
    int fails = 0;
    for (int o = 0; o < O; o++) {
        std::vector<int64_t> inv;
        auto ord = p3sam::serialize_order(g, M, o, &inv);
        long mism_o = 0, mism_i = 0;
        for (int n = 0; n < M; n++) {
            if (ord[n] != go[(size_t)o * M + n]) mism_o++;
            if (inv[n] != gi[(size_t)o * M + n]) mism_i++;
        }
        printf("  order %-14s : order_mismatch=%ld inverse_mismatch=%ld  %s\n",
               names[o], mism_o, mism_i, (mism_o == 0 && mism_i == 0) ? "PASS" : "*** FAIL ***");
        if (mism_o || mism_i) fails++;
    }
    printf("%s\n", fails ? "SERIALIZE FAIL" : "SERIALIZE ALL PASS");
    return fails ? 1 : 0;
}
