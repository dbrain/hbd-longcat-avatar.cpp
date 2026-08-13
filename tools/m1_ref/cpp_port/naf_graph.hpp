// NAF (valeoai/NAF) guide-conditioned feature upsampler — ggml graph.
// Validated math (naf_ref.py: ImageEncoder bit-exact 0.0, na2d 5.3e-5 vs golden):
//   guide[1,3,512,512] + lr DINOv3 patchmap[1,1024,32,32] -> hr[1,1024,512,512].
//   ImageEncoder: 2 conv encoders (k1 spatial + k3 sem, reflect pad; each = initial conv +
//     2x EncBlock[GroupNorm8->SiLU->Conv ×2, NO residual]) -> cat 256ch -> adaptive_pool
//     (identity@512) -> axial RoPE (heads 4, periods 100^(i/16), coords[-1,1] separate-norm).
//   CrossAttention (heads 4): q=ie_out, k=avgpool(ie_out,32²), v=lr; 9×9 dilated(16) neighborhood
//     cross-attn (natten na2d, shift border). Since k/v are block-16 nearest-upsampled from 32²,
//     na2d == per-output-pixel 9×9 over the SOURCE grid centered at clamp(i//16,4,27) (probe-
//     confirmed). Implemented as 81 get_rows taps over the flattened 32² source grid.
//
// ggml layout: guide [W,H,3,1]; all intermediates [W,H,C,1]; conv kernel via weight() = [KW,KH,IC,OC].
#pragma once
#include "m1_ggml.hpp"
#include <cmath>
#include <vector>

namespace naf {
static const int DIM = 256, NHA = 4, HQK = 64, NHR = 4, DR = 64;
static const int KS = 9, RAD = 4, ENC_C = 128;
static const int VC = 1024;  // v / output channels
static const float ROPE_BASE = 100.0f;

// per-config geometry: guide input res, output/target res, lr source grid, block upsample.
// @512 (shape_512): guide 512 -> identity pool -> SRC 32, UP 16.
// @1024 (shape_1024): guide 1024 -> avg-pool to 512 -> SRC 64, UP 8.
struct Cfg { int IMG_IN, IMG, SRC, UP; };
static const Cfg CFG512{512, 512, 32, 16};
static const Cfg CFG1024{1024, 512, 64, 8};

// conv2d with reflect pad (pad=0 for k1, pad=1 for k3). kernel = weight() ([KW,KH,IC,OC]).
static inline ggml_tensor* conv(ggml_context* ctx, M1Harness& H, ggml_tensor* x,
                                const std::string& key, int pad) {
    ggml_tensor* w = H.weight(key + ".weight");
    ggml_tensor* b = H.weight(key + ".bias");
    if (pad > 0) {
        x = ggml_pad_reflect_1d(ctx, x, pad, pad);                 // pad ne0 (W)
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));      // [H,W,C,B]
        x = ggml_pad_reflect_1d(ctx, x, pad, pad);                 // pad ne0 (now H)
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));      // back [W',H',C,B]
    }
    // VRAM lever (D): ggml_conv_2d's im2col takes the KERNEL's dtype. At @1024 the ImageEncoder's
    // 128ch k3 im2col over 1024² is the chain's VRAM PEAK (~4.8GB f32). Under PIXAL3D_FAST, cast the
    // kernel to F16 -> F16 im2col (halves it) + tensor cores; the mul_mat still accumulates in F32.
    // Near-lossless (im2col is data rearrangement; cosine validated). Default path stays f32 bit-exact.
    if (pix_fast_prec()) w = ggml_cast(ctx, w, GGML_TYPE_F16);
    x = ggml_conv_2d(ctx, w, x, 1, 1, 0, 0, 1, 1);                 // pad already applied
    ggml_tensor* bb = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);  // [1,1,OC,1]
    return ggml_add(ctx, x, bb);
}

// GroupNorm(8) over [W,H,C,1] (ggml_group_norm treats ne2 as channels) + per-channel affine.
static inline ggml_tensor* gnorm(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key) {
    x = ggml_group_norm(ctx, x, 8, 1e-5f);
    ggml_tensor* w = ggml_reshape_4d(ctx, H.weight(key + ".weight"), 1, 1, ENC_C, 1);
    ggml_tensor* b = ggml_reshape_4d(ctx, H.weight(key + ".bias"), 1, 1, ENC_C, 1);
    return ggml_add(ctx, ggml_mul(ctx, x, w), b);
}

static inline ggml_tensor* encblock(ggml_context* ctx, M1Harness& H, ggml_tensor* x,
                                    const std::string& p, int pad) {
    ggml_tensor* h = conv(ctx, H, ggml_silu(ctx, gnorm(ctx, H, x, p + ".norm1")), p + ".conv1", pad);
    h = conv(ctx, H, ggml_silu(ctx, gnorm(ctx, H, h, p + ".norm2")), p + ".conv2", pad);
    return h;  // residual=False
}

static inline ggml_tensor* enc(ggml_context* ctx, M1Harness& H, ggml_tensor* x,
                               const std::string& base, int pad) {
    ggml_tensor* h = conv(ctx, H, x, base + ".0", pad);
    h = encblock(ctx, H, h, base + ".1", pad);
    h = encblock(ctx, H, h, base + ".2", pad);
    return h;
}

// [W,H,C,1] -> [d, nh, HW]  (c = head*d + d_idx; hw = h*W + w)
static inline ggml_tensor* to_head(ggml_context* ctx, ggml_tensor* x, int d, int nh) {
    int64_t HW = x->ne[0] * x->ne[1];
    int64_t C = x->ne[2];
    ggml_tensor* f = ggml_reshape_2d(ctx, x, HW, C);        // [HW, C]
    f = ggml_cont(ctx, ggml_transpose(ctx, f));             // [C, HW]
    return ggml_reshape_3d(ctx, f, d, nh, HW);              // [d, nh, HW]
}

// axial RoPE on x [W,H,256,1]. periods loaded from wdir host. COS/SIN persistent consts.
static inline ggml_tensor* rope(ggml_context* ctx, M1Harness& H, ggml_tensor* x, int IMG) {
    // periods [16] (host)
    std::vector<float> pe = H.host_f32("image_encoder.rope.periods");
    const float* per = pe.data();
    const int64_t HW = (int64_t)IMG * IMG;
    std::vector<float> cosb((size_t)DR * HW), sinb((size_t)DR * HW);
    for (int h = 0; h < IMG; h++) {
        float ch = 2.0f * ((h + 0.5f) / IMG) - 1.0f;
        for (int w = 0; w < IMG; w++) {
            float cw = 2.0f * ((w + 0.5f) / IMG) - 1.0f;
            size_t hw = (size_t)h * IMG + w;
            float ang[DR];
            for (int i = 0; i < 16; i++) ang[i] = 2.0f * (float)M_PI * ch / per[i];        // h-freqs
            for (int i = 0; i < 16; i++) ang[16 + i] = 2.0f * (float)M_PI * cw / per[i];   // w-freqs
            for (int i = 0; i < 32; i++) ang[32 + i] = ang[i];                              // tile(2)
            size_t base = hw * DR;
            for (int d = 0; d < DR; d++) { cosb[base + d] = std::cos(ang[d]); sinb[base + d] = std::sin(ang[d]); }
        }
    }
    int64_t cs_ne[4] = {DR, 1, HW, 1};
    ggml_tensor* COS = H.const_tensor("naf_rope_cos", 3, cs_ne, std::move(cosb));
    ggml_tensor* SIN = H.const_tensor("naf_rope_sin", 3, cs_ne, std::move(sinb));
    ggml_tensor* xr = to_head(ctx, x, DR, NHR);             // [64,4,HW]
    ggml_tensor* rot = rotate_half(ctx, xr);               // half-split over ne0
    xr = ggml_add(ctx, ggml_mul(ctx, xr, COS), ggml_mul(ctx, rot, SIN));
    // back to [W,H,C,1]
    ggml_tensor* back = ggml_reshape_2d(ctx, xr, DIM, HW); // [C,HW]
    back = ggml_cont(ctx, ggml_transpose(ctx, back));      // [HW,C]
    return ggml_reshape_4d(ctx, back, IMG, IMG, DIM, 1);
}

// Build the NAF forward. guide [W,H,3,1] input; vin = lr_features [SRC,SRC,1024,1] input.
// Returns hr [W,H,1024,1] == naf_hr_golden [1,1024,512,512] raw. ie_out optional debug.
static inline ggml_tensor* build_naf_forward(ggml_context* ctx, M1Harness& H,
        ggml_tensor* guide, ggml_tensor* vin, ggml_tensor** ie_out = nullptr, const Cfg& cfg = CFG512) {
    const int IMG = cfg.IMG, SRC = cfg.SRC, UP = cfg.UP;
    // ---- ImageEncoder (runs at guide res IMG_IN) ----
    ggml_tensor* sp = enc(ctx, H, guide, "image_encoder.encoder", 0);      // k=1
    ggml_tensor* se = enc(ctx, H, guide, "image_encoder.sem_encoder", 1);  // k=3
    ggml_tensor* x = ggml_concat(ctx, sp, se, 2);                          // [IMG_IN,IMG_IN,256,1]
    // adaptive_avg_pool2d to (IMG,IMG): identity @512, factor-2 avg-pool @1024
    if (cfg.IMG_IN != IMG) { int f = cfg.IMG_IN / IMG; x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, f, f, f, f, 0, 0); }
    x = rope(ctx, H, x, IMG);                                              // [IMG,IMG,256,1]
    if (ie_out) { ggml_tensor* o = ggml_cont(ctx, x); *ie_out = o; ggml_set_output(o); x = o; }

    // ---- CrossAttention / na2d ----
    ggml_tensor* qh = to_head(ctx, x, HQK, NHA);                           // [64,4,HW]
    ggml_tensor* k = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, UP, UP, UP, UP, 0, 0);  // [SRC,SRC,256,1]
    // kf [256, SRC²] (ne0=C, ne1=s=sh*SRC+sw); vf [1024, SRC²]
    ggml_tensor* kf = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, k, SRC * SRC, DIM)));
    ggml_tensor* vf = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, vin, SRC * SRC, VC)));
    const int64_t HW = (int64_t)IMG * IMG;
    const float scale = 1.0f / std::sqrt((float)HQK);

    // host-precompute 81 source-gather index arrays idx_ab[hw] = src_h*SRC + src_w
    auto make_idx = [&](int a, int b) {
        std::vector<int32_t> idx((size_t)HW);
        for (int h = 0; h < IMG; h++) {
            int ci = h / UP; ci = ci < RAD ? RAD : (ci > SRC - 1 - RAD ? SRC - 1 - RAD : ci);
            int sh = ci + (a - RAD);
            for (int w = 0; w < IMG; w++) {
                int cj = w / UP; cj = cj < RAD ? RAD : (cj > SRC - 1 - RAD ? SRC - 1 - RAD : cj);
                int sw = cj + (b - RAD);
                idx[(size_t)h * IMG + w] = sh * SRC + sw;
            }
        }
        return idx;
    };

    // ---- scores: [81,4,HW] ----
    ggml_tensor* scores = nullptr;
    std::vector<ggml_tensor*> idx_t(KS * KS);
    for (int a = 0; a < KS; a++)
        for (int b = 0; b < KS; b++) {
            int t = a * KS + b;
            int64_t ine[4] = {HW, 1, 1, 1};
            char nm[32]; snprintf(nm, sizeof(nm), "naf_idx_%d", t);
            idx_t[t] = H.const_tensor_i32(nm, 1, ine, make_idx(a, b));
            ggml_tensor* kg = ggml_get_rows(ctx, kf, idx_t[t]);            // [256, HW]
            kg = ggml_reshape_3d(ctx, kg, HQK, NHA, HW);                   // [64,4,HW]
            ggml_tensor* s = ggml_sum_rows(ctx, ggml_mul(ctx, qh, kg));    // [1,4,HW]
            s = ggml_scale(ctx, s, scale);
            s = ggml_cont(ctx, ggml_permute(ctx, s, 0, 2, 1, 3));         // [1,HW,4]
            scores = scores ? ggml_concat(ctx, scores, s, 0) : s;         // -> [t+1,HW,4]
        }
    // soft_max over ne0=81. layout [81,HW,4] => CUDA grid (ne01=HW, ne02=4) keeps HW in
    // gridDim.x (limit 2^31); [81,4,HW] would put HW in gridDim.y (limit 65535) -> CUDA abort.
    scores = ggml_soft_max(ctx, scores);

    // ---- weighted V: out [256,4,HW] ----
    ggml_tensor* out = nullptr;
    for (int t = 0; t < KS * KS; t++) {
        ggml_tensor* vg = ggml_get_rows(ctx, vf, idx_t[t]);               // [1024, HW]
        vg = ggml_reshape_3d(ctx, vg, VC / NHA, NHA, HW);                 // [256,4,HW]
        // scores tap t: view [1,HW,4] -> permute to [1,4,HW] to broadcast over vg's ne0=256
        ggml_tensor* w = ggml_view_3d(ctx, scores, 1, HW, NHA,
                                      scores->nb[1], scores->nb[2], (size_t)t * scores->nb[0]);
        w = ggml_cont(ctx, ggml_permute(ctx, w, 0, 2, 1, 3));            // [1,4,HW]
        ggml_tensor* contrib = ggml_mul(ctx, vg, w);                      // w broadcasts over ne0
        out = out ? ggml_add(ctx, out, contrib) : contrib;
    }
    // [256,4,HW] -> [1024,HW] -> [HW,1024] -> [W,H,1024,1]
    out = ggml_reshape_2d(ctx, out, VC, HW);
    out = ggml_cont(ctx, ggml_transpose(ctx, out));                       // [HW,1024]
    return ggml_reshape_4d(ctx, out, IMG, IMG, VC, 1);
}
}  // namespace naf
