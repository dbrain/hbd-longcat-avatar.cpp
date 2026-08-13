// NAF@1024 (shape_1024 cond): guide@1024 -> avg-pool to 512, lr 64x64 -> hr 512x512.
// Validates the parameterized naf::build_naf_forward(CFG1024) vs the captured naf_hr_1024
// (stage3b_cond_capture.py reproduced the golden cond bit-exactly; naf_hr_1024 carries the
// model's tf32 noise, so expect ~5e-3 like the M2 @512 tf32 comparison — the ImageEncoder is
// bit-exact and the na2d shift-rule is identical, just dil8/block8 instead of dil16/block16).
// Build: ./build.sh naf_1024_test [cuda]   (run cuda w/ NVIDIA_TF32_OVERRIDE=0)
#include "naf_graph.hpp"
#include <cstdio>

static const char* WDIR = "weights_npy/naf";
static const char* REFS = "refs/stage3b";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const naf::Cfg cfg = naf::CFG1024;
    const int IMG_IN = cfg.IMG_IN, SRC = cfg.SRC, OUT = cfg.IMG, VC = naf::VC;

    M1Harness H(WDIR, 2048, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t g_ne[4] = {IMG_IN, IMG_IN, 3, 1};      ggml_tensor* guide = H.input("guide", 4, g_ne);
    int64_t v_ne[4] = {SRC, SRC, VC, 1};           ggml_tensor* vin = H.input("vin", 4, v_ne);
    ggml_tensor* hr = naf::build_naf_forward(ctx, H, guide, vin, nullptr, cfg);
    ggml_set_output(hr);

    ggml_cgraph* gf = new_graph(ctx, 16384);
    ggml_build_forward_expand(gf, hr);
    H.alloc_and_upload(gf);

    // guide = image_1024_chw [3,1024,1024] ([0,1], CHW) -> ggml [W,H,C]
    H.upload_input_npy(guide, std::string(REFS) + "/image_1024_chw.npy");
    // vin = patchmap_1024 [1,64,64,1024] BHWC -> transpose to [C,H,W] for ggml [W,H,C]
    NpyArray pm = npy_load(std::string(REFS) + "/patchmap_1024.npy");  // [1,64,64,1024]
    const float* p = pm.f32();
    std::vector<float> vd((size_t)SRC * SRC * VC);
    for (int hh = 0; hh < SRC; hh++) for (int ww = 0; ww < SRC; ww++) for (int c = 0; c < VC; c++)
        vd[(size_t)c*SRC*SRC + (size_t)hh*SRC + ww] = p[((size_t)hh*SRC + ww)*VC + c];
    H.upload_input_raw(vin, vd);

    H.compute(gf);

    printf("[naf@1024] backend=%s guide=%d->pool%d src=%d out=%d\n", use_cuda?"cuda":"cpu", IMG_IN, OUT, SRC, OUT);
    CmpStats s = compare_to_npy(H, hr, std::string(REFS) + "/naf_hr_1024.npy", true, "hr");
    // tf32-golden comparison: ~5e-3 == correct (M2 lesson). meanabs is the real signal.
    bool ok = s.meanabs < 5e-3;
    printf("[naf@1024] %s (vs tf32 golden: meanabs<5e-3 == correct; maxabs carries tf32 noise)\n", ok?"PASS":"CHECK");
    return ok ? 0 : 1;
}
