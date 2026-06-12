// Image loading + PIL-matching Lanczos-3 resize for the pixal3d CLI front-end.
//   load PNG/JPG (stb_image) -> resize to image_size (Lanczos-3, antialiased, PIL algorithm)
//   -> /255 -> CHW float [0,1]. (DINOv3 input is ImageNet-normalized downstream; NAF uses [0,1].)
// PIL parity: matches Image.resize(..., LANCZOS) to ~1/255 (PIL uses 8-bit fixed-point coeffs;
// this keeps float intermediates, slightly more accurate). The preprocessed square matte is the
// expected input (host rembg cut-line); raw photo -> matte/crop is a host pre-step.
#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "../../../thirdparty/stb_image.h"
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

namespace imgio {

struct Image { int w=0, h=0; std::vector<float> rgb; };  // HWC, [0,1]

inline Image load_rgb01(const std::string& path) {
    int w, h, c;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, 3);  // force RGB
    if (!d) throw std::runtime_error("failed to load image: " + path + " (" + stbi_failure_reason() + ")");
    Image im; im.w = w; im.h = h; im.rgb.resize((size_t)w*h*3);
    for (size_t i = 0; i < (size_t)w*h*3; i++) im.rgb[i] = d[i] / 255.0f;
    stbi_image_free(d);
    return im;
}

static inline double sinc(double x) { if (x == 0.0) return 1.0; x *= M_PI; return std::sin(x) / x; }
static inline double lanczos3(double x) { return (x > -3.0 && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0; }

// PIL precompute_coeffs: per output sample, the input window [start, start+width) + weights.
static inline void build_coeffs(int in, int out, std::vector<int>& start, std::vector<int>& width,
                                std::vector<double>& w, int& maxk) {
    const double support0 = 3.0;
    double scale = (double)in / (double)out;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = support0 * filterscale;
    maxk = (int)std::ceil(support) * 2 + 1;
    start.assign(out, 0); width.assign(out, 0); w.assign((size_t)out * maxk, 0.0);
    double ss = 1.0 / filterscale;
    for (int xx = 0; xx < out; xx++) {
        double center = (xx + 0.5) * scale;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > in) xmax = in;
        int k = xmax - xmin;
        double sum = 0;
        for (int x = 0; x < k; x++) { double ww = lanczos3((x + xmin - center + 0.5) * ss); w[(size_t)xx*maxk + x] = ww; sum += ww; }
        if (sum != 0) for (int x = 0; x < k; x++) w[(size_t)xx*maxk + x] /= sum;
        start[xx] = xmin; width[xx] = k;
    }
}

// resize HWC [0,1] -> HWC [0,1] (separable Lanczos-3, horizontal then vertical)
inline Image resize_lanczos3(const Image& src, int dw, int dh) {
    std::vector<int> hs, hw_, vs, vw_; std::vector<double> hwt, vwt; int hk, vk;
    build_coeffs(src.w, dw, hs, hw_, hwt, hk);
    build_coeffs(src.h, dh, vs, vw_, vwt, vk);
    // horizontal: src.w -> dw  (rows = src.h)
    std::vector<float> tmp((size_t)dw * src.h * 3);
    for (int y = 0; y < src.h; y++)
        for (int x = 0; x < dw; x++)
            for (int c = 0; c < 3; c++) {
                double acc = 0;
                for (int t = 0; t < hw_[x]; t++) acc += hwt[(size_t)x*hk + t] * src.rgb[((size_t)y*src.w + (hs[x]+t))*3 + c];
                tmp[((size_t)y*dw + x)*3 + c] = (float)acc;
            }
    // vertical: src.h -> dh
    Image out; out.w = dw; out.h = dh; out.rgb.resize((size_t)dw*dh*3);
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++)
            for (int c = 0; c < 3; c++) {
                double acc = 0;
                for (int t = 0; t < vw_[y]; t++) acc += vwt[(size_t)y*vk + t] * tmp[((size_t)(vs[y]+t)*dw + x)*3 + c];
                float v = (float)acc; v = v < 0 ? 0 : (v > 1 ? 1 : v);
                out.rgb[((size_t)y*dw + x)*3 + c] = v;
            }
    return out;
}

// HWC [0,1] -> CHW [0,1] (the cond-model layout)
inline std::vector<float> to_chw(const Image& im) {
    std::vector<float> chw((size_t)3 * im.h * im.w);
    const size_t HW = (size_t)im.h * im.w;
    for (int y = 0; y < im.h; y++) for (int x = 0; x < im.w; x++) for (int c = 0; c < 3; c++)
        chw[(size_t)c*HW + (size_t)y*im.w + x] = im.rgb[((size_t)y*im.w + x)*3 + c];
    return chw;
}

// load + resize to `size` + CHW [0,1]
inline std::vector<float> load_chw(const std::string& path, int size) {
    Image im = load_rgb01(path);
    Image r = resize_lanczos3(im, size, size);
    return to_chw(r);
}

}  // namespace imgio
