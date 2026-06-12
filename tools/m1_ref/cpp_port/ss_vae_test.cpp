// SS VAE decoder (SparseStructureDecoder) in ggml: z_s [1,8,16,16,16] -> ss_logits
// [1,1,64,64,64] -> occupancy coords [N,4] @ res32.
//
// Mirrors tools/m1_ref/ss_vae_decode.py. Dense 3D-conv VAE: Conv3d(8->512) -> 2x
// ResBlock3d(512) middle -> blocks [512,512,Up512->128,128,128,Up128->32,32,32] ->
// ChannelLayerNorm32 + SiLU + Conv3d(32->1). ResBlock = CLN->SiLU->conv->CLN->SiLU->conv
// + identity skip. Upsample = Conv3d(ch->next*8) + pixel_shuffle_3d(2). All fp32.
//
// Layout: ggml [W,H,D,C] (ne3=channel) throughout (== ggml_conv_3d data convention).
// Validates ss_logits vs refs/ss_logits_fp32.npy (tight) + coords SET vs golden (N=1126).
#include "m1_ggml.hpp"
#include <cmath>
#include <set>
#include <array>

static const char* WDIR = "weights_npy/ss_dec";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";

// ChannelLayerNorm32: LayerNorm over channel (ne3). x [W,H,D,C].
static ggml_tensor* cln(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& pfx) {
    ggml_tensor* xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 3, 0));  // [C,W,H,D]
    xp = ggml_norm(ctx, xp, 1e-5f);
    xp = ggml_mul(ctx, xp, H.weight(pfx + ".weight"));
    xp = ggml_add(ctx, xp, H.weight(pfx + ".bias"));
    return ggml_cont(ctx, ggml_permute(ctx, xp, 3, 0, 1, 2));            // back to [W,H,D,C]
}

static ggml_tensor* conv3d(ggml_context* ctx, M1Harness& H, ggml_tensor* x,
                           const std::string& key, int64_t IC) {
    ggml_tensor* w = H.weight_conv3d(key + ".weight");  // [KW,KH,KD,IC*OC]
    ggml_tensor* b = H.weight(key + ".bias");
    x = ggml_conv_3d(ctx, w, x, IC, 1, 1, 1, 1, 1, 1, 1, 1, 1);  // k3 p1 s1
    b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);
    return ggml_add(ctx, x, b);
}

static ggml_tensor* resblock(ggml_context* ctx, M1Harness& H, ggml_tensor* x,
                             const std::string& pfx, int64_t ch) {
    ggml_tensor* h = cln(ctx, H, x, pfx + ".norm1");
    h = silu_(ctx, h);
    h = conv3d(ctx, H, h, pfx + ".conv1", ch);
    h = cln(ctx, H, h, pfx + ".norm2");
    h = silu_(ctx, h);
    h = conv3d(ctx, H, h, pfx + ".conv2", ch);
    return ggml_add(ctx, h, x);  // identity skip
}

static ggml_tensor* upsample(ggml_context* ctx, M1Harness& H, ggml_tensor* x,
                             const std::string& pfx, int64_t ch, int64_t next) {
    x = conv3d(ctx, H, x, pfx + ".conv", ch);     // ch -> next*8
    return depth_to_space_3d(ctx, x, next, 2, 2);  // pixel_shuffle_3d(2)
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    M1Harness H(WDIR, 256, use_cuda);
    ggml_context* ctx = H.ctx;

    int64_t z_ne[4] = {16, 16, 16, 8};  // z_s [1,8,16,16,16] flat == [W,H,D,C]
    ggml_tensor* z = H.input("z_s", 4, z_ne);

    ggml_tensor* h = conv3d(ctx, H, z, "input_layer", 8);   // [16,16,16,512]
    h = resblock(ctx, H, h, "middle_block.0", 512);
    h = resblock(ctx, H, h, "middle_block.1", 512);
    h = resblock(ctx, H, h, "blocks.0", 512);
    h = resblock(ctx, H, h, "blocks.1", 512);
    h = upsample(ctx, H, h, "blocks.2", 512, 128);          // [32,32,32,128]
    h = resblock(ctx, H, h, "blocks.3", 128);
    h = resblock(ctx, H, h, "blocks.4", 128);
    h = upsample(ctx, H, h, "blocks.5", 128, 32);           // [64,64,64,32]
    h = resblock(ctx, H, h, "blocks.6", 32);
    h = resblock(ctx, H, h, "blocks.7", 32);
    h = cln(ctx, H, h, "out_layer.0");
    h = silu_(ctx, h);
    h = conv3d(ctx, H, h, "out_layer.2", 32);               // [64,64,64,1]
    ggml_tensor* logits = ggml_cont(ctx, h);
    ggml_set_output(logits);

    ggml_cgraph* gf = new_graph(ctx, 8192);
    ggml_build_forward_expand(gf, logits);
    H.alloc_and_upload(gf);
    H.upload_input_npy(z, std::string(GOLD) + "/stage1_ssdec/z_s.npy");
    H.compute(gf);

    printf("[ss_vae] backend=%s\n", use_cuda ? "cuda" : "cpu");
    CmpStats s = compare_to_npy(H, logits, std::string(REFS) + "/ss_logits_fp32.npy", true, "ss_logits");

    // ---- occupancy -> coords (host), compare SET to golden ----
    const int G = 64, G2 = 32;
    std::vector<float> L((size_t)G * G * G);
    ggml_backend_tensor_get(logits, L.data(), 0, L.size() * sizeof(float));
    // ggml [W,H,D] flat == numpy ss_logits[0,0,X,Y,Z] C-order (X slowest). pool 2^3 -> res32.
    auto at = [&](int x, int y, int z) { return L[((size_t)x * G + y) * G + z]; };
    std::set<std::array<int, 3>> mine;
    for (int x = 0; x < G2; x++)
        for (int y = 0; y < G2; y++)
            for (int z = 0; z < G2; z++) {
                bool occ = false;
                for (int ax = 0; ax < 2 && !occ; ax++)
                    for (int ay = 0; ay < 2 && !occ; ay++)
                        for (int az = 0; az < 2 && !occ; az++)
                            if (at(2 * x + ax, 2 * y + ay, 2 * z + az) > 0) occ = true;
                if (occ) mine.insert({x, y, z});
            }
    // load golden coords [N,4] int32, cols (b,x,y,z)
    NpyArray gc = npy_load(std::string(GOLD) + "/stage1_out/coords.npy");
    const int32_t* gci = gc.i32();
    int64_t gn = gc.shape[0];
    std::set<std::array<int, 3>> gold;
    for (int64_t i = 0; i < gn; i++)
        gold.insert({gci[i * 4 + 1], gci[i * 4 + 2], gci[i * 4 + 3]});

    int inter = 0;
    for (auto& c : mine) if (gold.count(c)) inter++;
    printf("  coords mine N=%zu  golden N=%lld  intersection=%d\n",
           mine.size(), (long long)gn, inter);
    bool set_eq = (mine == gold);
    printf("  set-equal=%s\n", set_eq ? "true" : "false");
    bool ok = s.maxabs < 5e-3 && set_eq;
    printf("[ss_vae] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
