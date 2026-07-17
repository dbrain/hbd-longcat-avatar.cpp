// MoGe-2 neck + points_head + mask_head (ConvStack) + UV pyramid, in ggml.
// Mirrors moge/model/{v2.py,modules.py} (see moge_port/MOGE-ARCH.md §3,§4). Consumes the summed
// DINOv2 encoder feature map (output_projections) + the analytic view-plane UV pyramid; produces the
// dense point map / mask the FOV recovery needs.
//
//   ConvStack(neck): 5 stages, res-blocks [0,2,2,2,0], resamplers [convT,convT,convT,bilinear],
//                    output_blocks all Identity -> returns 5 feature maps (60²..960²).
//   ConvStack(points/mask_head): same but res-blocks [0,1,1,1,0]; input_blocks adapt channels;
//                    output_blocks.4 = 1x1 conv -> 3ch (points) / 1ch (mask) at 960².
// ResidualConvBlock norms == "none" -> [ReLU, Conv3x3(replicate), ReLU, Conv3x3(replicate)] + skip.
#pragma once
#include "m1_ggml.hpp"
#include <cmath>
#include <vector>
#include <string>

namespace moge {

// replicate (edge) padding by p on ne0 (W) and ne1 (H) — torch padding_mode="replicate".
static inline ggml_tensor* replicate_pad2d(ggml_context* ctx, ggml_tensor* x, int p) {
    if (p == 0) return x;
    const int64_t W = x->ne[0], Hh = x->ne[1], C = x->ne[2], N = x->ne[3];
    // pad ne0 (W): replicate first/last column p times.
    {
        ggml_tensor* left  = ggml_cont(ctx, ggml_view_4d(ctx, x, 1, Hh, C, N, x->nb[1], x->nb[2], x->nb[3], 0));
        ggml_tensor* right = ggml_cont(ctx, ggml_view_4d(ctx, x, 1, Hh, C, N, x->nb[1], x->nb[2], x->nb[3], (W - 1) * x->nb[0]));
        ggml_tensor* acc = x;
        for (int i = 0; i < p; i++) acc = ggml_concat(ctx, left, acc, 0);
        for (int i = 0; i < p; i++) acc = ggml_concat(ctx, acc, right, 0);
        x = acc;
    }
    // pad ne1 (H): replicate first/last row.
    {
        const int64_t W2 = x->ne[0];
        ggml_tensor* top = ggml_cont(ctx, ggml_view_4d(ctx, x, W2, 1, C, N, x->nb[1], x->nb[2], x->nb[3], 0));
        ggml_tensor* bot = ggml_cont(ctx, ggml_view_4d(ctx, x, W2, 1, C, N, x->nb[1], x->nb[2], x->nb[3], (Hh - 1) * x->nb[1]));
        ggml_tensor* acc = x;
        for (int i = 0; i < p; i++) acc = ggml_concat(ctx, top, acc, 1);
        for (int i = 0; i < p; i++) acc = ggml_concat(ctx, acc, bot, 1);
        x = acc;
    }
    return x;
}

static inline ggml_tensor* add_bias_chw(ggml_context* ctx, ggml_tensor* x, ggml_tensor* b) {
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));  // [1,1,OC,1] broadcasts over W,H
}

// 1x1 conv (== per-pixel linear). kernel npy [OC,IC,1,1] -> ggml [1,1,IC,OC].
static inline ggml_tensor* conv1x1(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key) {
    ggml_tensor* w = H.weight(key + ".weight");
    ggml_tensor* y = ggml_conv_2d(ctx, w, x, 1, 1, 0, 0, 1, 1);
    return add_bias_chw(ctx, y, H.weight(key + ".bias"));
}
// 3x3 conv, replicate padding p=1. kernel npy [OC,IC,3,3] -> ggml [3,3,IC,OC].
static inline ggml_tensor* conv3x3_rep(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key) {
    ggml_tensor* w = H.weight(key + ".weight");
    x = replicate_pad2d(ctx, x, 1);
    ggml_tensor* y = ggml_conv_2d(ctx, w, x, 1, 1, 0, 0, 1, 1);
    return add_bias_chw(ctx, y, H.weight(key + ".bias"));
}
// ConvTranspose2d k2 s2 p0. kernel npy [IC,OC,2,2] -> ggml [2,2,OC,IC] (== conv_transpose_2d_p0 layout).
static inline ggml_tensor* convT2x2(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key) {
    ggml_tensor* w = H.weight(key + ".weight");
    ggml_tensor* y = ggml_conv_transpose_2d_p0(ctx, w, x, 2);
    return add_bias_chw(ctx, y, H.weight(key + ".bias"));
}

// ResidualConvBlock (both norms Identity): relu -> conv3x3 -> relu -> conv3x3, + skip.
static inline ggml_tensor* resblock(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key) {
    ggml_tensor* skip = x;
    ggml_tensor* h = ggml_relu(ctx, x);
    h = conv3x3_rep(ctx, H, h, key + ".layers.2");
    h = ggml_relu(ctx, h);
    h = conv3x3_rep(ctx, H, h, key + ".layers.5");
    return ggml_add(ctx, h, skip);
}

// resampler ×2: convT (idx 0,1,2) or bilinear-upsample (idx 3), each + a refine conv3x3.
static inline ggml_tensor* resampler_convT(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key) {
    x = convT2x2(ctx, H, x, key + ".0");
    return conv3x3_rep(ctx, H, x, key + ".1");
}
static inline ggml_tensor* resampler_bilin(ggml_context* ctx, M1Harness& H, ggml_tensor* x, const std::string& key) {
    x = ggml_interpolate(ctx, x, 2 * x->ne[0], 2 * x->ne[1], x->ne[2], x->ne[3], GGML_SCALE_MODE_BILINEAR);
    return conv3x3_rep(ctx, H, x, key + ".1");
}

// Generic ConvStack forward. in_feats: 5 input maps. num_res: per-stage residual-block count.
// has_out4: whether output_blocks.4 is a real 1x1 conv (heads) vs Identity (neck).
// Returns all 5 out_features (out_features[i] is taken BEFORE resampling, output_blocks applied).
static inline std::vector<ggml_tensor*> convstack(ggml_context* ctx, M1Harness& H, const std::string& prefix,
        const std::vector<ggml_tensor*>& in_feats, const int num_res[5], bool has_out4) {
    std::vector<ggml_tensor*> out(5, nullptr);
    ggml_tensor* x = nullptr;
    for (int i = 0; i < 5; i++) {
        ggml_tensor* feat = conv1x1(ctx, H, in_feats[i], prefix + ".input_blocks." + std::to_string(i));
        x = (i == 0) ? feat : ggml_add(ctx, x, feat);
        for (int r = 0; r < num_res[i]; r++)
            x = resblock(ctx, H, x, prefix + ".res_blocks." + std::to_string(i) + "." + std::to_string(r));
        out[i] = (i == 4 && has_out4) ? conv1x1(ctx, H, x, prefix + ".output_blocks.4") : x;
        if (i < 3)      x = resampler_convT(ctx, H, x, prefix + ".resamplers." + std::to_string(i));
        else if (i == 3) x = resampler_bilin(ctx, H, x, prefix + ".resamplers.3");
    }
    return out;
}

// ---- analytic view-plane UV grid (geometry_torch.py normalized_view_plane_uv), as ggml [W,H,2,1] ----
// channel 0 = u (varies along W/cols), channel 1 = v (varies along H/rows). aspect_ratio = imgW/imgH.
static inline std::vector<float> view_plane_uv(int width, int height, double aspect_ratio) {
    double AR = aspect_ratio;
    double span_x = AR / std::sqrt(1.0 + AR * AR);
    double span_y = 1.0 / std::sqrt(1.0 + AR * AR);
    auto lin = [](double a, double b, int n, int i) {
        return n == 1 ? a : a + (b - a) * (double)i / (double)(n - 1);
    };
    // ggml [W,H,2] layout: idx = ch*W*H + h*W + w. ch0=u(w), ch1=v(h).
    std::vector<float> uv((size_t)2 * width * height);
    for (int h = 0; h < height; h++) {
        double v = lin(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, h);
        for (int w = 0; w < width; w++) {
            double u = lin(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, w);
            uv[(size_t)0 * width * height + (size_t)h * width + w] = (float)u;
            uv[(size_t)1 * width * height + (size_t)h * width + w] = (float)v;
        }
    }
    return uv;
}

}  // namespace moge
