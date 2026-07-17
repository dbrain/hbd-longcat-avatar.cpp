// Reusable DINOv2 ViT-L/14 forward graph — shared by MoGe-2 (camera/FOV) and UltraShape (conditioner).
//
// Adapted from dinov3_graph.hpp (our DINOv3 ViT-L/16). Same macro shape (HID 1024, 24 blocks, 16 heads,
// head_dim 64, pre-norm ViT, LayerScale, plain erf-GELU MLP, affine final norm) but the DINOv2 deltas
// (see moge_port/MOGE-ARCH.md §7) are baked in:
//   * patch 14 (not 16)            * cls-only, NO register tokens (N_PREFIX = 1)
//   * learned additive pos_embed, bicubic-interpolated to the (rows,cols) grid with the +0.1 "kludge"
//     offset (host C++, F.interpolate-exact) — NOT RoPE
//   * fused qkv (3072,1024)+bias    * LayerNorm eps 1e-6
//   * weight names encoder.backbone.{blocks.{i}.norm1/attn.qkv/attn.proj/mlp.fc1-fc2/ls{1,2}.gamma,
//     patch_embed.proj, cls_token, pos_embed, norm}
//   * affine final norm applied to EACH captured intermediate (get_intermediate_layers, norm=True)
//
// Validated (CPU fp32 oracle) against MoGe goldens vit_inter_{5,11,17,23} (patch tokens, final-normed)
// and cls_token (final-normed cls of block 23).
#pragma once
#include "m1_ggml.hpp"
#include <cmath>
#include <vector>
#include <string>

namespace dinov2 {
static const int HID = 1024, N_LAYERS = 24, N_HEADS = 16, HEAD_DIM = 64;
static const int PATCH = 14, N_PREFIX = 1;       // cls only, no registers
static const float LN_EPS = 1e-6f;
static const int POS_GRID = 37;                  // pos_embed is 1 cls + 37x37 patches (img_size 518 / 14)

// Two state-dict conventions carry the SAME DINOv2-L arch:
//   MOGE  — MoGe-2 checkpoint: prefix "encoder.backbone.", blocks "blocks.{i}.", FUSED qkv
//           ("attn.qkv"), out "attn.proj", layerscale "ls{1,2}.gamma", final "norm", patch "patch_embed.proj".
//   HF    — stock HF Dinov2Model (UltraShape conditioner): prefix "main_image_encoder.model.",
//           blocks "encoder.layer.{i}.", SPLIT q/k/v+bias ("attention.attention.{query,key,value}"),
//           out "attention.output.dense", layerscale "layer_scale{1,2}.lambda1", final "layernorm",
//           patch "embeddings.patch_embeddings.projection", cls "embeddings.cls_token".
enum Scheme { MOGE, HF };
struct Names {
    Scheme scheme; std::string prefix;
    std::string patch_w, patch_b, cls, final_w, final_b;
    std::string blk(int i) const {
        return scheme == MOGE ? prefix + "blocks." + std::to_string(i) + "."
                              : prefix + "encoder.layer." + std::to_string(i) + ".";
    }
};
static inline Names names(Scheme s, const std::string& prefix) {
    Names n; n.scheme = s; n.prefix = prefix;
    if (s == MOGE) {
        n.patch_w = prefix + "patch_embed.proj.weight"; n.patch_b = prefix + "patch_embed.proj.bias";
        n.cls = prefix + "cls_token"; n.final_w = prefix + "norm.weight"; n.final_b = prefix + "norm.bias";
    } else {
        n.patch_w = prefix + "embeddings.patch_embeddings.projection.weight";
        n.patch_b = prefix + "embeddings.patch_embeddings.projection.bias";
        n.cls = prefix + "embeddings.cls_token";
        n.final_w = prefix + "layernorm.weight"; n.final_b = prefix + "layernorm.bias";
    }
    return n;
}

// ---- PyTorch bicubic (align_corners=False, A=-0.75) cubic-conv coefficients ----
static inline void cubic_coeffs(float t, float w[4]) {
    const float A = -0.75f;
    // x1 = t (distance to index floor+0); x2 = 1-t (distance to floor+1)
    float x1 = t;
    w[0] = ((A * (x1 + 1) - 5 * A) * (x1 + 1) + 8 * A) * (x1 + 1) - 4 * A;  // cubic_convolution2(x1+1)
    w[1] = ((A + 2) * x1 - (A + 3)) * x1 * x1 + 1;                          // cubic_convolution1(x1)
    float x2 = 1.0f - t;
    w[2] = ((A + 2) * x2 - (A + 3)) * x2 * x2 + 1;                          // cubic_convolution1(x2)
    w[3] = ((A * (x2 + 1) - 5 * A) * (x2 + 1) + 8 * A) * (x2 + 1) - 4 * A;  // cubic_convolution2(x2+1)
}

// Build a separable 1-D bicubic resample weight matrix W[out][in] for in_size -> out_size, matching
// F.interpolate(scale_factor=s, mode='bicubic', align_corners=False, antialias=False). With a scale_factor
// given, PyTorch uses kernel scale = 1/scale_factor (compute_scales_value), src = scale*(dst+0.5)-0.5,
// and clamps the 4 source taps to [0,in-1] (upsample_get_value_bounded).
static inline std::vector<float> bicubic_weights(int in_size, int out_size, double scale_factor) {
    double scale = 1.0 / scale_factor;  // PyTorch: scales given -> 1/scale_factor
    std::vector<float> W((size_t)out_size * in_size, 0.f);
    for (int o = 0; o < out_size; o++) {
        double src = scale * (o + 0.5) - 0.5;
        double fl = std::floor(src);
        float t = (float)(src - fl);
        float w[4]; cubic_coeffs(t, w);
        int base = (int)fl;
        for (int k = 0; k < 4; k++) {
            int idx = base - 1 + k;
            idx = idx < 0 ? 0 : (idx >= in_size ? in_size - 1 : idx);  // clamp (bounded access)
            W[(size_t)o * in_size + idx] += w[k];
        }
    }
    return W;
}

// Interpolate the learned pos_embed (raw npy [1,1370,1024] = cls + 37x37 patches, row-major) to a
// (rows,cols) patch grid and prepend the cls pos. Returns a flat buffer laid out as ggml [HID, 1+rows*cols]
// (ne0=HID fastest), i.e. token-major / channel-fastest — ready to upload as a const and ggml_add'd to x.
// Reproduces DINOv2 interpolate_pos_encoding: scale_factor = ((cols+0.1)/37, (rows+0.1)/37) bicubic.
// (PyTorch quirk: after reshape(1,37,37,dim).permute(0,3,1,2)->NCHW, the H axis is the patch ROW and is
//  scaled by scale_factor[0]; the code passes (w0,h0) so H uses (w0=cols), W uses (h0=rows). For square
//  grids this is symmetric; we replicate the exact pairing for non-square safety.)
// offset = the DINOv2 "historical kludge" added to the scale_factor numerator (original/MoGe = 0.1);
// some HF Dinov2 builds interpolate with size=(new,new) instead (offset 0.0). If block-0 parity is off,
// flip this.
static inline std::vector<float> interp_pos_embed(const float* pos_embed, int rows, int cols, double offset = 0.1) {
    const int M = POS_GRID;
    // patch grid pos = pos_embed[1:], shape [37*37, HID]; cls pos = pos_embed[0].
    const float* cls_pos = pos_embed;               // [HID]
    const float* grid    = pos_embed + (size_t)HID; // [37*37, HID]
    // PyTorch pairing: H(rows of NCHW)=scale[0] uses cols; W=scale[1] uses rows.
    std::vector<float> Wrow = bicubic_weights(M, rows, (double)(cols + offset) / M);  // rows-axis resample
    std::vector<float> Wcol = bicubic_weights(M, cols, (double)(rows + offset) / M);  // cols-axis resample
    const int SEQ = 1 + rows * cols;
    std::vector<float> out((size_t)SEQ * HID, 0.f);
    // token 0 = cls pos
    for (int c = 0; c < HID; c++) out[(size_t)0 * HID + c] = cls_pos[c];
    // separable: out[oy,ox,:] = sum_iy sum_ix Wrow[oy,iy]*Wcol[ox,ix]*grid[iy*37+ix,:]
    // do it in two passes to keep it O(rows*M*M*HID + ...) cheap; M=37 so direct is fine.
    // pass 1: tmp[oy, ix, c] = sum_iy Wrow[oy,iy] * grid[iy,ix,c]
    std::vector<float> tmp((size_t)rows * M * HID, 0.f);
    for (int oy = 0; oy < rows; oy++) {
        const float* wr = &Wrow[(size_t)oy * M];
        for (int iy = 0; iy < M; iy++) {
            float wv = wr[iy];
            if (wv == 0.f) continue;
            for (int ix = 0; ix < M; ix++) {
                const float* g = grid + ((size_t)iy * M + ix) * HID;
                float* d = &tmp[((size_t)oy * M + ix) * HID];
                for (int c = 0; c < HID; c++) d[c] += wv * g[c];
            }
        }
    }
    // pass 2: out[1 + oy*cols + ox, c] = sum_ix Wcol[ox,ix] * tmp[oy,ix,c]
    for (int oy = 0; oy < rows; oy++) {
        for (int ox = 0; ox < cols; ox++) {
            const float* wc = &Wcol[(size_t)ox * M];
            float* d = &out[((size_t)(1 + oy * cols + ox)) * HID];
            for (int ix = 0; ix < M; ix++) {
                float wv = wc[ix];
                if (wv == 0.f) continue;
                const float* s = &tmp[((size_t)oy * M + ix) * HID];
                for (int c = 0; c < HID; c++) d[c] += wv * s[c];
            }
        }
    }
    return out;
}

// Build the DINOv2 ViT graph.
//   img        : ggml input [cols*14, rows*14, 3, 1] — ALREADY normalized (ImageNet mean/std), RGB.
//   pos_const  : ggml const  [HID, 1+rows*cols]      — from interp_pos_embed() above.
//   capture    : 0-indexed block outputs to tap (e.g. {5,11,17,23}); each is final-normed then patch-split.
//   inter_out  : filled with one tensor per capture entry — the final-normed PATCH tokens [HID, rows*cols].
//   cls_out    : if non-null, set to the final-normed cls token of the LAST captured block [HID, 1].
//   patchmap_out (opt): final-normed patch tokens of the deepest block reshaped [HID, cols, rows].
// Returns the final-normed full hidden of the last requested layer (== the deepest capture, [HID, SEQ]).

// patch-embed + add pos -> [HID, SEQ] token stream. (img already normalized; pos_const = interp_pos_embed.)
static inline ggml_tensor* embed_tokens(ggml_context* ctx, M1Harness& H, const Names& N,
                                        ggml_tensor* img, ggml_tensor* pos_const, int rows, int cols) {
    ggml_tensor* conv = ggml_conv_2d(ctx, H.weight(N.patch_w), img, PATCH, PATCH, 0, 0, 1, 1); // [cols,rows,1024]
    ggml_tensor* emb = ggml_reshape_2d(ctx, conv, (int64_t)cols * rows, HID);                  // [p, ch]
    emb = ggml_cont(ctx, ggml_transpose(ctx, emb));                                            // [ch, p]
    emb = ggml_add(ctx, emb, H.weight(N.patch_b));
    ggml_tensor* clsr = ggml_reshape_2d(ctx, H.weight(N.cls), HID, 1);
    ggml_tensor* x = ggml_concat(ctx, clsr, emb, 1);                                           // [ch, SEQ]
    return ggml_add(ctx, x, pos_const);
}

// one DINOv2 ViT block (pre-norm, fused-or-split qkv, SDPA, layerscale residuals). x:[HID,SEQ].
static inline ggml_tensor* vit_block(ggml_context* ctx, M1Harness& H, const Names& N, int li, ggml_tensor* x, int SEQ) {
    std::string p = N.blk(li);
    ggml_tensor* h = layernorm(ctx, x, H.weight(p + "norm1.weight"), H.weight(p + "norm1.bias"), LN_EPS);
    ggml_tensor *q, *k, *v;
    if (N.scheme == MOGE) {
        ggml_tensor* qkv = lin(ctx, H.weight(p + "attn.qkv.weight"), H.weight(p + "attn.qkv.bias"), h); // [3072,SEQ]
        q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, HID, SEQ, qkv->nb[1], 0));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, HID, SEQ, qkv->nb[1], (size_t)HID * qkv->nb[0]));
        v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, HID, SEQ, qkv->nb[1], (size_t)2 * HID * qkv->nb[0]));
    } else {  // HF split q/k/v, each with bias
        q = lin(ctx, H.weight(p + "attention.attention.query.weight"), H.weight(p + "attention.attention.query.bias"), h);
        k = lin(ctx, H.weight(p + "attention.attention.key.weight"),   H.weight(p + "attention.attention.key.bias"),   h);
        v = lin(ctx, H.weight(p + "attention.attention.value.weight"), H.weight(p + "attention.attention.value.bias"), h);
    }
    q = ggml_reshape_3d(ctx, q, HEAD_DIM, N_HEADS, SEQ);
    k = ggml_reshape_3d(ctx, k, HEAD_DIM, N_HEADS, SEQ);
    v = ggml_reshape_3d(ctx, v, HEAD_DIM, N_HEADS, SEQ);
    ggml_tensor* attn = attention(ctx, q, k, v, 1.0f / std::sqrt((float)HEAD_DIM));  // [HID, SEQ]
    const char* op_w = N.scheme == MOGE ? "attn.proj.weight" : "attention.output.dense.weight";
    const char* op_b = N.scheme == MOGE ? "attn.proj.bias"   : "attention.output.dense.bias";
    const char* ls1  = N.scheme == MOGE ? "ls1.gamma"        : "layer_scale1.lambda1";
    const char* ls2  = N.scheme == MOGE ? "ls2.gamma"        : "layer_scale2.lambda1";
    attn = lin(ctx, H.weight(p + op_w), H.weight(p + op_b), attn);
    attn = ggml_mul(ctx, attn, H.weight(p + ls1));
    x = ggml_add(ctx, x, attn);
    ggml_tensor* h2 = layernorm(ctx, x, H.weight(p + "norm2.weight"), H.weight(p + "norm2.bias"), LN_EPS);
    h2 = lin(ctx, H.weight(p + "mlp.fc1.weight"), H.weight(p + "mlp.fc1.bias"), h2);
    h2 = gelu_erf_(ctx, h2);
    h2 = lin(ctx, H.weight(p + "mlp.fc2.weight"), H.weight(p + "mlp.fc2.bias"), h2);
    h2 = ggml_mul(ctx, h2, H.weight(p + ls2));
    return ggml_add(ctx, x, h2);
}

// MoGe path: capture final-normed PATCH tokens at the given block indices + the deepest cls token.
static inline void build(ggml_context* ctx, M1Harness& H, ggml_tensor* img, ggml_tensor* pos_const,
                         int rows, int cols, const std::vector<int>& capture,
                         std::vector<ggml_tensor*>& inter_out, ggml_tensor** cls_out,
                         const std::string& prefix = "encoder.backbone.") {
    const int NP = rows * cols, SEQ = N_PREFIX + NP;
    Names N = names(MOGE, prefix);
    ggml_tensor* x = embed_tokens(ctx, H, N, img, pos_const, rows, cols);
    inter_out.assign(capture.size(), nullptr);
    int deepest = -1; for (int c : capture) deepest = std::max(deepest, c);
    for (int li = 0; li < N_LAYERS; li++) {
        x = vit_block(ctx, H, N, li, x, SEQ);
        for (size_t ci = 0; ci < capture.size(); ci++) {
            if (capture[ci] != li) continue;
            ggml_tensor* normed = layernorm(ctx, x, H.weight(N.final_w), H.weight(N.final_b), LN_EPS);
            ggml_tensor* patch = ggml_cont(ctx, ggml_view_2d(ctx, normed, HID, NP, normed->nb[1],
                                                             (size_t)N_PREFIX * normed->nb[1]));
            inter_out[ci] = patch; ggml_set_output(patch);
            if (li == deepest && cls_out) {
                *cls_out = ggml_cont(ctx, ggml_view_2d(ctx, normed, HID, N_PREFIX, normed->nb[1], 0));
                ggml_set_output(*cls_out);
            }
        }
    }
}

// Conditioner path (UltraShape / stock HF Dinov2Model): run all 24 blocks + the final affine layernorm,
// return the full last_hidden_state [HID, SEQ] (cls + patches) — the cross-attn KV the DiT consumes.
static inline ggml_tensor* encode(ggml_context* ctx, M1Harness& H, ggml_tensor* img, ggml_tensor* pos_const,
                                  int rows, int cols, Scheme scheme, const std::string& prefix) {
    const int SEQ = N_PREFIX + rows * cols;
    Names N = names(scheme, prefix);
    ggml_tensor* x = embed_tokens(ctx, H, N, img, pos_const, rows, cols);
    for (int li = 0; li < N_LAYERS; li++) x = vit_block(ctx, H, N, li, x, SEQ);
    x = layernorm(ctx, x, H.weight(N.final_w), H.weight(N.final_b), LN_EPS);  // [HID, SEQ]
    ggml_set_output(x);
    return x;
}
}  // namespace dinov2
