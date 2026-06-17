// detok_r5.cpp — R5 validation harness (no ggml/GPU). Thin wrapper over the VALIDATED core in
// detok_r5.hpp: parse the skeleton token stream -> joints + parent tree, validate vs the Python
// goldens (detok_joints.npy / detok_parents.npy). Reads tok_spec.txt + output_ids.npy.
//   build: ./build.sh detok_r5   |   run: ./detok_r5 [golden=/tmp/skintokens_e2e]
#include "../../sparse_spike/npy.hpp"
#include "detok_r5.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string g = argc > 1 ? argv[1] : "/tmp/skintokens_e2e";
    detok::Spec s = detok::load_spec(g + "/tok_spec.txt");
    NpyArray oi = npy_load(g + "/output_ids.npy");

    detok::Skeleton sk = detok::detokenize(oi.i64(), oi.numel(), s);
    if (!sk.ok) { printf("[detok_r5] FAIL: %s\n", sk.err.c_str()); return 1; }
    int J = sk.J;

    // validate
    NpyArray gj = npy_load(g + "/detok_joints.npy");   // [J,3]
    NpyArray gp = npy_load(g + "/detok_parents.npy");  // [J]
    int Jg = (int)gj.shape[0];
    printf("[detok_r5] J=%d (golden %d)\n", J, Jg);
    if (J != Jg) { printf("[detok_r5] FAIL joint count\n"); return 1; }
    const float* gjp = gj.f32(); const int64_t* gpp = gp.i64();
    float maxabs = 0; int pbad = 0;
    for (int k = 0; k < J; k++) {
        for (int c = 0; c < 3; c++) maxabs = std::max(maxabs, std::fabs(sk.joints[(size_t)k * 3 + c] - gjp[k * 3 + c]));
        if (sk.parents[k] != (int)gpp[k]) pbad++;
    }
    printf("[detok_r5] joints maxabs=%.3e  parents mismatch=%d/%d\n", maxabs, pbad, J);
    bool ok = (maxabs < 1e-5) && (pbad == 0);
    printf("[detok_r5] %s\n", ok ? "PASS (joints exact + parent tree identical)" : "FAIL");
    return ok ? 0 : 1;
}
