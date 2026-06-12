// Reusable DINOv3 ViT-L/16 forward graph (shared by dinov3_test + stage1 E2E + M5 stage3b).
// See dinov3_test.cpp for the validated reference (mirrors tools/m1_ref/dinov3_proj.py).
// Parameterized by image size via dino::Cfg: CFG512 (ss/shape_512, 32x32 patches, validated)
// and CFG1024 (shape_1024, 64x64 patches). Model arch (HID/layers/heads/RoPE-theta) is fixed;
// only the patch grid (IMG/NPH/NPW/NP/SEQ) varies. The @512 path is bit-identical to before.
#pragma once
#include "m1_ggml.hpp"
#include <cmath>

namespace dino {
static const int HID = 1024, N_LAYERS = 24, N_HEADS = 16, HEAD_DIM = 64;
static const int PATCH = 16, N_REG = 4;
static const float ROPE_THETA = 100.0f, LN_EPS = 1e-5f;
static const int N_PREFIX = 1 + N_REG;

struct Cfg {
    int IMG, NPH, NPW;
    int NP()  const { return NPH * NPW; }
    int SEQ() const { return N_PREFIX + NP(); }
};
static const Cfg CFG512{512, 32, 32};
static const Cfg CFG1024{1024, 64, 64};
// backward-compat constants (== CFG512) for callers that wired the @512 path directly.
static const int IMG = 512, NPH = 32, NPW = 32, NP = 32 * 32, SEQ = N_PREFIX + 32 * 32;

static inline void rope_cos_sin(std::vector<float>& cosb, std::vector<float>& sinb, const Cfg& cfg = CFG512) {
    const int NPH = cfg.NPH, NPW = cfg.NPW, NP = cfg.NP();
    cosb.assign((size_t)HEAD_DIM * NP, 0.f); sinb.assign((size_t)HEAD_DIM * NP, 0.f);
    int nf = HEAD_DIM / 4;
    std::vector<float> inv_freq(nf);
    for (int f = 0; f < nf; f++) inv_freq[f] = 1.0f / std::pow(ROPE_THETA, (4.0f / HEAD_DIM) * f);
    for (int i = 0; i < NPH; i++)
        for (int j = 0; j < NPW; j++) {
            int p = i * NPW + j;
            float cy = ((i + 0.5f) / NPH) * 2.f - 1.f, cx = ((j + 0.5f) / NPW) * 2.f - 1.f;
            for (int f = 0; f < nf; f++) {
                float ay = 2.f * (float)M_PI * cy * inv_freq[f], ax = 2.f * (float)M_PI * cx * inv_freq[f];
                size_t base = (size_t)p * HEAD_DIM;
                cosb[base + f] = std::cos(ay); sinb[base + f] = std::sin(ay);
                cosb[base + nf + f] = std::cos(ax); sinb[base + nf + f] = std::sin(ax);
                cosb[base + 32 + f] = std::cos(ay); sinb[base + 32 + f] = std::sin(ay);
                cosb[base + 32 + nf + f] = std::cos(ax); sinb[base + 32 + nf + f] = std::sin(ax);
            }
        }
}

// img [W,H,3,1], cosT/sinT [d,1,NP]. Returns final hidden [ch,SEQ]; sets *global_out [ch,5]
// and *patchmap_out [ch,NPW,NPH] if non-null.
static inline ggml_tensor* build_dinov3(ggml_context* ctx, M1Harness& H, ggml_tensor* img,
        ggml_tensor* cosT, ggml_tensor* sinT, ggml_tensor** global_out, ggml_tensor** patchmap_out,
        const Cfg& cfg = CFG512) {
    const int NPH = cfg.NPH, NPW = cfg.NPW, NP = cfg.NP(), SEQ = cfg.SEQ();
    ggml_tensor* pw = H.weight("embeddings.patch_embeddings.weight");
    ggml_tensor* pb = H.weight("embeddings.patch_embeddings.bias");
    ggml_tensor* cls = H.weight("embeddings.cls_token");
    ggml_tensor* reg = H.weight("embeddings.register_tokens");

    ggml_tensor* conv = ggml_conv_2d(ctx, pw, img, PATCH, PATCH, 0, 0, 1, 1);
    ggml_tensor* emb = ggml_reshape_2d(ctx, conv, (int64_t)NPW * NPH, HID);
    emb = ggml_cont(ctx, ggml_transpose(ctx, emb));
    emb = ggml_add(ctx, emb, pb);
    ggml_tensor* clsr = ggml_reshape_2d(ctx, cls, HID, 1);
    ggml_tensor* regr = ggml_reshape_2d(ctx, reg, HID, N_REG);
    ggml_tensor* x = ggml_concat(ctx, ggml_concat(ctx, clsr, regr, 1), emb, 1);

    for (int li = 0; li < N_LAYERS; li++) {
        std::string p = "layer." + std::to_string(li) + ".";
        ggml_tensor* h = layernorm(ctx, x, H.weight(p + "norm1.weight"), H.weight(p + "norm1.bias"), LN_EPS);
        ggml_tensor* q = lin(ctx, H.weight(p + "attention.q_proj.weight"), H.weight(p + "attention.q_proj.bias"), h);
        ggml_tensor* k = lin(ctx, H.weight(p + "attention.k_proj.weight"), nullptr, h);
        ggml_tensor* v = lin(ctx, H.weight(p + "attention.v_proj.weight"), H.weight(p + "attention.v_proj.bias"), h);
        q = ggml_reshape_3d(ctx, q, HEAD_DIM, N_HEADS, SEQ);
        k = ggml_reshape_3d(ctx, k, HEAD_DIM, N_HEADS, SEQ);
        v = ggml_reshape_3d(ctx, v, HEAD_DIM, N_HEADS, SEQ);
        auto apply_rope = [&](ggml_tensor* t) {
            ggml_tensor* pre = ggml_cont(ctx, ggml_view_3d(ctx, t, HEAD_DIM, N_HEADS, N_PREFIX, t->nb[1], t->nb[2], 0));
            ggml_tensor* pat = ggml_cont(ctx, ggml_view_3d(ctx, t, HEAD_DIM, N_HEADS, NP, t->nb[1], t->nb[2], (size_t)N_PREFIX * t->nb[2]));
            ggml_tensor* rot = ggml_add(ctx, ggml_mul(ctx, pat, cosT), ggml_mul(ctx, rotate_half(ctx, pat), sinT));
            return ggml_concat(ctx, pre, rot, 2);
        };
        q = apply_rope(q); k = apply_rope(k);
        ggml_tensor* attn = attention(ctx, q, k, v, 1.0f / std::sqrt((float)HEAD_DIM));
        attn = lin(ctx, H.weight(p + "attention.o_proj.weight"), H.weight(p + "attention.o_proj.bias"), attn);
        attn = ggml_mul(ctx, attn, H.weight(p + "layer_scale1.lambda1"));
        x = ggml_add(ctx, x, attn);
        ggml_tensor* h2 = layernorm(ctx, x, H.weight(p + "norm2.weight"), H.weight(p + "norm2.bias"), LN_EPS);
        h2 = lin(ctx, H.weight(p + "mlp.up_proj.weight"), H.weight(p + "mlp.up_proj.bias"), h2);
        h2 = gelu_erf_(ctx, h2);
        h2 = lin(ctx, H.weight(p + "mlp.down_proj.weight"), H.weight(p + "mlp.down_proj.bias"), h2);
        h2 = ggml_mul(ctx, h2, H.weight(p + "layer_scale2.lambda1"));
        x = ggml_add(ctx, x, h2);
    }
    x = layernorm(ctx, x, nullptr, nullptr, LN_EPS);
    if (global_out) { *global_out = ggml_cont(ctx, ggml_view_2d(ctx, x, HID, N_PREFIX, x->nb[1], 0)); ggml_set_output(*global_out); }
    if (patchmap_out) {
        ggml_tensor* patch = ggml_cont(ctx, ggml_view_2d(ctx, x, HID, NP, x->nb[1], (size_t)N_PREFIX * x->nb[1]));
        *patchmap_out = ggml_reshape_3d(ctx, patch, HID, NPW, NPH);
        ggml_set_output(*patchmap_out);
    }
    return x;
}
}  // namespace dino
