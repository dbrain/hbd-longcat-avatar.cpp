// DINOv3 ViT-L/16 @ image 1024 (shape_1024 cond, 64x64 patches) — validates the parameterized
// dino::build_dinov3(CFG1024) patchmap + global vs the captured goldens (refs/stage3b/, from
// stage3b_cond_capture.py which reproduced the golden cond bit-exactly). Same weights as @512;
// only the patch grid / RoPE differ. Build: ./build.sh dinov3_1024_test [cuda]
#include "dinov3_graph.hpp"
#include <cmath>

static const char* WDIR = "weights_npy/dinov3";
static const char* REFS = "refs/stage3b";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const dino::Cfg cfg = dino::CFG1024;
    const int IMG = cfg.IMG, NP = cfg.NP();

    M1Harness H(WDIR, 1024, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t img_ne[4] = {IMG, IMG, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);
    int64_t cs_ne[4] = {dino::HEAD_DIM, 1, NP, 1};
    ggml_tensor* cosT = H.input("rope_cos", 3, cs_ne), *sinT = H.input("rope_sin", 3, cs_ne);
    ggml_tensor *global = nullptr, *patchmap = nullptr;
    dino::build_dinov3(ctx, H, img, cosT, sinT, &global, &patchmap, cfg);

    ggml_cgraph* gf = new_graph(ctx, 16384);
    ggml_build_forward_expand(gf, patchmap);
    ggml_build_forward_expand(gf, global);
    H.alloc_and_upload(gf);

    // load image_1024_chw [3,1024,1024] in [0,1] -> ImageNet-normalize -> upload as [W,H,C]
    NpyArray im = npy_load(std::string(REFS) + "/image_1024_chw.npy");
    std::vector<float> imn(im.f32(), im.f32() + im.numel());
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    const size_t HW = (size_t)IMG * IMG;
    for (int c = 0; c < 3; c++) for (size_t i = 0; i < HW; i++) imn[(size_t)c*HW + i] = (imn[(size_t)c*HW + i] - mean[c]) / sd[c];
    H.upload_input_raw(img, imn);
    std::vector<float> cosb, sinb; dino::rope_cos_sin(cosb, sinb, cfg);
    H.upload_input_raw(cosT, cosb); H.upload_input_raw(sinT, sinb);

    H.compute(gf);

    printf("[dinov3@1024] backend=%s NP=%d\n", use_cuda ? "cuda" : "cpu", NP);
    // DINOv3 has massive-activation outlier channels -> judge by meanabs (maxabs on a 1e5
    // value is ~1e-7 rel; same as the M1 @512 lesson).
    CmpStats g  = compare_to_npy(H, global,   std::string(REFS) + "/global_1024.npy",   true, "global");
    CmpStats pm = compare_to_npy(H, patchmap, std::string(REFS) + "/patchmap_1024.npy", true, "patchmap");
    bool ok = (g.meanabs < 1e-3) && (pm.meanabs < 1e-3);
    printf("[dinov3@1024] %s (global/patchmap meanabs<1e-3; outliers dominate maxabs)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
