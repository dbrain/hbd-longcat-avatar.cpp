// Validate the DINOv2-large conditioner port (dinov2::encode, HF scheme) against the UltraShape golden.
// Feeds the banked cond_inputs (ImageNet-normalized 1022^2 = 73x73 patch grid) -> full last_hidden_state
// [1,5330,1024], compares to cond_last_hidden. Proves the shared dinov2_graph.hpp serves UltraShape too.
//   ./build.sh dinov2_cond_test [cuda]      (POS_OFFSET env overrides the pos-embed kludge offset)
#include "dinov2_graph.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/cond_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/conditioner";
static const char* PREFIX = "main_image_encoder.model.";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const int rows = 73, cols = 73;   // 1022/14
    double offset = std::getenv("POS_OFFSET") ? atof(std::getenv("POS_OFFSET")) : 0.1;
    M1Harness H(WDIR, 2048, use_cuda);
    ggml_context* ctx = H.ctx;

    int64_t img_ne[4] = {(int64_t)cols * 14, (int64_t)rows * 14, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);
    NpyArray pe = npy_load(std::string(WDIR) + "/" + PREFIX + "embeddings.position_embeddings.npy");
    std::vector<float> pos = dinov2::interp_pos_embed(pe.f32(), rows, cols, offset);
    int64_t pos_ne[4] = {dinov2::HID, (int64_t)(1 + rows * cols), 1, 1};
    ggml_tensor* pos_const = H.const_tensor("pos_embed_interp", 2, pos_ne, pos);

    ggml_tensor* out = dinov2::encode(ctx, H, img, pos_const, rows, cols, dinov2::HF, PREFIX);

    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, out);
    H.alloc_and_upload(gf);
    H.upload_input_npy(img, std::string(GDIR) + "/cond_inputs.npy");
    H.compute(gf);

    printf("[dinov2_cond] backend=%s grid=%dx%d offset=%.2f SEQ=%d\n",
           use_cuda ? "cuda" : "cpu", rows, cols, offset, 1 + rows * cols);
    CmpStats s = compare_to_npy(H, out, std::string(GDIR) + "/cond_last_hidden.npy", true, "last_hidden");
    bool ok = s.meanabs < 1e-2;  // cosine (printed) is the real signal; DINOv2 massive activations inflate maxabs
    printf("[dinov2_cond] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
