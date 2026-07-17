// Native UltraShape image conditioner host-side glue (preprocess + foreground-mask token gather).
// The DINOv2-large ViT itself is dinov2::encode (validated). This header adds the two host pieces that
// wrap it (ImageProcessorV2 output -> SingleImageEncoder/DinoImageEncoder.forward):
//   (1) preprocess(): value_range [-1,1]->[0,1], torchvision Resize(image_size, bilinear, antialias=True)
//       (PIL separable resampling), CenterCrop (identity for square), ImageNet Normalize.
//   (2) mask_valid_tokens(): mask_transform Resize(NEAREST)+CenterCrop, MaxPool2d(14), keep CLS +
//       foreground patches (maxpooled mask != -1), in token-index order -> indices into [0,SEQ).
// Both validated numerically vs the fp32 oracle goldens (cond_full_goldens): resize maxabs 7e-5,
// token selection EXACT (3970 of 5330). See ultrashape_cond_test.cpp.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace us_cond {

// ImageNet mean/std (DinoImageEncoder).
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

struct Tap { int xmin; std::vector<float> w; };

// PIL/torchvision antialias bilinear (triangle filter) separable resampling weights, in==out per side.
// scale = in/out (>=1 for downscale here); support = 1.0*max(scale,1); triangle((x+0.5-center)/fscale).
static inline std::vector<Tap> pil_bilinear_weights(int in_size, int out_size) {
    double scale  = (double)in_size / out_size;
    double fscale = scale >= 1.0 ? scale : 1.0;
    double support = 1.0 * fscale;            // bilinear filter radius = 1.0
    std::vector<Tap> taps(out_size);
    for (int xx = 0; xx < out_size; xx++) {
        double center = (xx + 0.5) * scale;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > in_size) xmax = in_size;
        std::vector<float> w; double ss = 0.0;
        for (int x = xmin; x < xmax; x++) {
            double t = (x + 0.5 - center) / fscale;
            double v = 1.0 - std::fabs(t); if (v < 0) v = 0;
            w.push_back((float)v); ss += v;
        }
        for (auto& v : w) v = (float)(v / ss);
        taps[xx] = {xmin, std::move(w)};
    }
    return taps;
}

// img: [C,H,W] (W fastest), values in [-1,1]. Returns [C,OUT,OUT] (== ggml [OUT,OUT,C,1]) ImageNet-normed.
static inline std::vector<float> preprocess(const float* img, int C, int H, int W, int OUT,
                                            const float mean[3], const float std_[3]) {
    std::vector<Tap> tx = pil_bilinear_weights(W, OUT);
    std::vector<Tap> ty = pil_bilinear_weights(H, OUT);
    std::vector<double> tmp((size_t)C * H * OUT, 0.0);     // horizontal pass -> [C,H,OUT]
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++) {
            const float* row = img + ((size_t)c * H + y) * W;
            double* d = &tmp[((size_t)c * H + y) * OUT];
            for (int o = 0; o < OUT; o++) {
                const Tap& t = tx[o]; double acc = 0.0;
                for (size_t k = 0; k < t.w.size(); k++) acc += ((double)row[t.xmin + k] + 1.0) * 0.5 * t.w[k];
                d[o] = acc;
            }
        }
    std::vector<float> out((size_t)C * OUT * OUT);          // vertical pass -> [C,OUT,OUT] + norm
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < OUT; oy++) {
            const Tap& t = ty[oy];
            float* d = &out[((size_t)c * OUT + oy) * OUT];
            for (int ox = 0; ox < OUT; ox++) {
                double acc = 0.0;
                for (size_t k = 0; k < t.w.size(); k++)
                    acc += tmp[((size_t)c * H + (t.xmin + k)) * OUT + ox] * t.w[k];
                d[ox] = (float)((acc - mean[c]) / std_[c]);
            }
        }
    return out;
}

// mask: [1,H,W] (W fastest) in {-1,+1}. Returns valid token indices (CLS=0 + foreground patches),
// in increasing token-index order. patch=14, grid = OUT/patch per side; mask_transform = NEAREST resize
// (src=floor(dst*in/out)) + CenterCrop(identity), then MaxPool2d(14,14); valid = maxpool != -1.
static inline std::vector<int> mask_valid_tokens(const float* mask, int H, int W, int OUT,
                                                 int patch, bool use_cls) {
    double sy = (double)H / OUT, sx = (double)W / OUT;
    int G = OUT / patch;                                    // 73
    std::vector<int> idx;
    if (use_cls) idx.push_back(0);
    for (int gy = 0; gy < G; gy++)
        for (int gx = 0; gx < G; gx++) {
            float mx = -1e30f;
            for (int py = 0; py < patch; py++) {
                int oy = gy * patch + py;
                int my = (int)std::floor(oy * sy); if (my > H - 1) my = H - 1;
                for (int px = 0; px < patch; px++) {
                    int ox = gx * patch + px;
                    int mxx = (int)std::floor(ox * sx); if (mxx > W - 1) mxx = W - 1;
                    float v = mask[(size_t)my * W + mxx];
                    if (v > mx) mx = v;
                }
            }
            if (mx != -1.0f) idx.push_back(1 + gy * G + gx);  // token index (CLS at 0)
        }
    return idx;
}

// Gather valid tokens from a [HID, SEQ] (HID fastest) hidden state into [HID, Tc] (HID fastest).
static inline std::vector<float> gather_tokens(const float* hidden, int HID, int SEQ,
                                               const std::vector<int>& valid) {
    std::vector<float> out((size_t)HID * valid.size());
    for (size_t t = 0; t < valid.size(); t++)
        std::copy(hidden + (size_t)valid[t] * HID, hidden + (size_t)(valid[t] + 1) * HID,
                  out.begin() + (size_t)t * HID);
    return out;
}

}  // namespace us_cond
