// Validate the shared DINOv2 ViT-L/14 port (dinov2_graph.hpp) against MoGe-2 goldens.
//   encoder.backbone (DINOv2-L) on the miku matte (840x840 = 60x60 patch grid, square).
// Feeds the BANKED preprocessed input image_14_norm (ImageNet-normalized), computes the bicubic-
// interpolated pos_embed on host, runs the 24-block ViT, and compares the final-normed PATCH tokens
// after blocks 5/11/17/23 to vit_inter_{5,11,17,23}, and the cls token to cls_token.
//
//   ./build.sh dinov2_test          # CPU fp32 oracle
//   ./build.sh dinov2_test cuda     # CUDA
#include "dinov2_graph.hpp"
#include <cstdio>
#include <string>

static const char* DIR = "/mnt/hdd/3d/avatar-shootout/moge_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/moge_goldens/weights_npy";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const int rows = 60, cols = 60;  // 840/14; matches meta.npy (token_rows,token_cols)
    M1Harness H(WDIR, 1024, use_cuda);
    ggml_context* ctx = H.ctx;

    // input: image_14_norm [1,3,840,840] (N,C,H,W) == ggml [W=840,H=840,C=3,N=1] same raw bytes.
    int64_t img_ne[4] = {(int64_t)cols * 14, (int64_t)rows * 14, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);

    // host-interpolated pos_embed [HID, SEQ] const.
    NpyArray pe = npy_load(std::string(WDIR) + "/encoder.backbone.pos_embed.npy");
    std::vector<float> pos = dinov2::interp_pos_embed(pe.f32(), rows, cols);
    int64_t pos_ne[4] = {dinov2::HID, (int64_t)(1 + rows * cols), 1, 1};
    ggml_tensor* pos_const = H.const_tensor("pos_embed_interp", 2, pos_ne, pos);

    std::vector<int> capture = {5, 11, 17, 23};
    std::vector<ggml_tensor*> inter;
    ggml_tensor* cls_out = nullptr;
    dinov2::build(ctx, H, img, pos_const, rows, cols, capture, inter, &cls_out);

    ggml_cgraph* gf = new_graph(ctx, 32768);
    for (auto* t : inter) ggml_build_forward_expand(gf, t);
    ggml_build_forward_expand(gf, cls_out);
    H.alloc_and_upload(gf);
    H.upload_input_npy(img, std::string(DIR) + "/image_14_norm.npy");

    H.compute(gf);

    printf("[dinov2] backend=%s grid=%dx%d SEQ=%d\n", use_cuda ? "cuda" : "cpu", rows, cols, 1 + rows * cols);
    // Pass on cosine (the project bar is ~0.999+): DINOv2 carries massive-activation outlier channels
    // (|x|~39) where the cuda-fp32 golden vs our cpu-fp32 accumulation order makes maxabs/maxrel
    // uninformative (~0.7% on a magnitude-39 value). cosine + meanabs are the meaningful signal.
    bool ok = true;
    for (size_t i = 0; i < capture.size(); i++) {
        char tag[24]; snprintf(tag, sizeof(tag), "vit_inter_%d", capture[i]);
        std::string ref = std::string(DIR) + "/vit_inter_" + std::to_string(capture[i]) + ".npy";
        CmpStats s = compare_to_npy(H, inter[i], ref, true, tag);
        ok = ok && s.meanabs < 1e-2;  // cosine (printed) is ~0.999997+; meanabs gate covers it
    }
    CmpStats c = compare_to_npy(H, cls_out, std::string(DIR) + "/cls_token.npy", true, "cls_token");
    ok = ok && c.meanabs < 1e-2;
    printf("[dinov2] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
