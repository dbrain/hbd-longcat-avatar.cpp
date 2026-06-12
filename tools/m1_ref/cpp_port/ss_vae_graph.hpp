// Reusable SS VAE decoder graph (shared by ss_vae_test + sampler/E2E).
// build_ss_vae_decode: z_s [W,H,D,C=8] -> ss_logits [W=64,H=64,D=64,C=1]. See
// ss_vae_test.cpp for the validated reference (mirrors tools/m1_ref/ss_vae_decode.py).
#pragma once
#include "m1_ggml.hpp"
#include <set>
#include <array>

namespace ssvae {

static inline ggml_tensor* cln(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& pfx) {
    ggml_tensor* xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 3, 0));  // [C,W,H,D]
    xp = ggml_norm(ctx, xp, 1e-5f);
    xp = ggml_mul(ctx, xp, H.weight(pfx + ".weight"));
    xp = ggml_add(ctx, xp, H.weight(pfx + ".bias"));
    return ggml_cont(ctx, ggml_permute(ctx, xp, 3, 0, 1, 2));            // [W,H,D,C]
}
static inline ggml_tensor* conv3d(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key, int64_t IC) {
    ggml_tensor* w = H.weight_conv3d(key + ".weight");
    ggml_tensor* b = H.weight(key + ".bias");
    x = ggml_conv_3d(ctx, w, x, IC, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);
    return ggml_add(ctx, x, b);
}
static inline ggml_tensor* resblock(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& pfx, int64_t ch) {
    ggml_tensor* h = cln(ctx, H, x, pfx + ".norm1");
    h = silu_(ctx, h);
    h = conv3d(ctx, H, h, pfx + ".conv1", ch);
    h = cln(ctx, H, h, pfx + ".norm2");
    h = silu_(ctx, h);
    h = conv3d(ctx, H, h, pfx + ".conv2", ch);
    return ggml_add(ctx, h, x);
}
static inline ggml_tensor* upsample(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& pfx, int64_t ch, int64_t next) {
    x = conv3d(ctx, H, x, pfx + ".conv", ch);
    return depth_to_space_3d(ctx, x, next, 2, 2);
}

static inline ggml_tensor* build_ss_vae_decode(ggml_context* ctx, M1Harness& H, ggml_tensor* z) {
    ggml_tensor* h = conv3d(ctx, H, z, "input_layer", 8);
    h = resblock(ctx, H, h, "middle_block.0", 512);
    h = resblock(ctx, H, h, "middle_block.1", 512);
    h = resblock(ctx, H, h, "blocks.0", 512);
    h = resblock(ctx, H, h, "blocks.1", 512);
    h = upsample(ctx, H, h, "blocks.2", 512, 128);
    h = resblock(ctx, H, h, "blocks.3", 128);
    h = resblock(ctx, H, h, "blocks.4", 128);
    h = upsample(ctx, H, h, "blocks.5", 128, 32);
    h = resblock(ctx, H, h, "blocks.6", 32);
    h = resblock(ctx, H, h, "blocks.7", 32);
    h = cln(ctx, H, h, "out_layer.0");
    h = silu_(ctx, h);
    h = conv3d(ctx, H, h, "out_layer.2", 32);  // [64,64,64,1]
    return ggml_cont(ctx, h);
}

// occupancy -> coords set. logits flat [64^3] in ggml [W,H,D] order == numpy [X,Y,Z].
static inline std::set<std::array<int, 3>> logits_to_coords(const std::vector<float>& L) {
    const int G = 64, G2 = 32;
    auto at = [&](int x, int y, int z) { return L[((size_t)x * G + y) * G + z]; };
    std::set<std::array<int, 3>> s;
    for (int x = 0; x < G2; x++)
        for (int y = 0; y < G2; y++)
            for (int z = 0; z < G2; z++) {
                bool occ = false;
                for (int ax = 0; ax < 2 && !occ; ax++)
                    for (int ay = 0; ay < 2 && !occ; ay++)
                        for (int az = 0; az < 2 && !occ; az++)
                            if (at(2 * x + ax, 2 * y + ay, 2 * z + az) > 0) occ = true;
                if (occ) s.insert({x, y, z});
            }
    return s;
}
static inline std::set<std::array<int, 3>> load_golden_coords(const std::string& npy) {
    NpyArray gc = npy_load(npy);
    const int32_t* g = gc.i32();
    std::set<std::array<int, 3>> s;
    for (int64_t i = 0; i < gc.shape[0]; i++) s.insert({g[i * 4 + 1], g[i * 4 + 2], g[i * 4 + 3]});
    return s;
}
}  // namespace ssvae
