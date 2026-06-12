// NAF conditioner ggml port test. Validates ImageEncoder (vs naf_ie_out) + full hr
// (vs naf_hr_golden) — goldens from naf_capture.py; reference math in naf_ref.py
// (ImageEncoder bit-exact 0.0, na2d 5.3e-5). Build: ./build.sh naf_test [cuda].
#include "naf_graph.hpp"
#include <cmath>

static const char* WDIR = "weights_npy/naf";
static const char* REFS = "refs";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");

    M1Harness H(WDIR, 1024, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t g_ne[4] = {naf::IMG, naf::IMG, 3, 1};      ggml_tensor* guide = H.input("guide", 4, g_ne);
    int64_t v_ne[4] = {naf::SRC, naf::SRC, naf::VC, 1}; ggml_tensor* vin = H.input("lr", 4, v_ne);

    ggml_tensor* ie = nullptr;
    ggml_tensor* hr = naf::build_naf_forward(ctx, H, guide, vin, &ie);
    ggml_set_output(hr);
    ggml_cgraph* gf = new_graph(ctx, 4096);
    ggml_build_forward_expand(gf, hr);
    H.alloc_and_upload(gf);

    H.upload_input_npy(guide, std::string(REFS) + "/naf_guide.npy");
    H.upload_input_npy(vin, std::string(REFS) + "/naf_lr_features.npy");
    H.compute(gf);

    printf("[naf] backend=%s\n", use_cuda ? "cuda" : "cpu");
    // TIGHT: vs true-fp32 oracle (naf_ref.py, tf32 disabled) — the correct apples-to-apples target.
    printf("[naf] vs TRUE-fp32 oracle (tight):\n");
    CmpStats ie_f = compare_to_npy(H, ie, std::string(REFS) + "/naf_ie_fp32.npy", true, "ie_fp32");
    CmpStats hr_f = compare_to_npy(H, hr, std::string(REFS) + "/naf_hr_fp32.npy", true, "hr_fp32");
    // SANITY: vs the captured tf32-CUDA golden (expect ~1e-2 on conv-heavy encoder — tf32 noise).
    printf("[naf] vs tf32-CUDA golden (sanity, ~1e-2 expected):\n");
    compare_to_npy(H, ie, std::string(REFS) + "/naf_ie_out.npy", true, "ie_gold");
    compare_to_npy(H, hr, std::string(REFS) + "/naf_hr_golden.npy", true, "hr_gold");
    bool ok = (use_cuda ? (ie_f.maxabs < 5e-2 && hr_f.maxabs < 5e-2)
                        : (ie_f.maxabs < 1e-3 && hr_f.maxabs < 1e-3));
    printf("[naf] %s (vs fp32 oracle: ie maxabs=%.3e, hr maxabs=%.3e)\n",
           ok ? "PASS" : "FAIL", ie_f.maxabs, hr_f.maxabs);
    return ok ? 0 : 1;
}
