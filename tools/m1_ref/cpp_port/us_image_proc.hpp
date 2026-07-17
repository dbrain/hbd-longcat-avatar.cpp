// Native UltraShape ImageProcessorV2 (ultrashape/preprocessors.py) — the raw-PNG front-end.
// Replaces the banked proc_image/proc_mask boundary: load an RGBA matte -> recenter(0.15) -> resize to
// `size` -> [-1,1] CHW tensors that the conditioner consumes. Faithful host port of the cv2/PIL ops:
//   recenter:  alpha bbox -> crop (excl. last bbox row/col, matching x_max-x_min) -> cv2.INTER_AREA
//              resize to 0.85*size -> center in size^2 zero canvas -> composite RGBA over WHITE.
//   load_image: cv2.resize(image, size, INTER_CUBIC a=-0.75) ; cv2.resize(mask, size, INTER_NEAREST).
//   array_to_tensor: /255*2-1, HWC->CHW.
// cv2 resamplers ported exactly (computeResizeAreaTab; interpolateCubic A=-0.75 + BORDER_REPLICATE;
// nearest = floor(dx*src/dst)). Float intermediates (cv2's 8-bit path uses fixed-point coeffs -> our
// approximation is within ~1 LSB; the conditioner is robust, validate cond_main cosine >0.999).
#pragma once
// Provide the stb_image implementation ONLY if no earlier header already pulled it in (e.g. image_io.hpp,
// when this file is compiled into the same TU as image_to_rig.cpp). Guarding just the `#define` isn't
// enough — the second unconditional `#include` re-emits the impl while STB_IMAGE_IMPLEMENTATION is still
// defined. So we skip the re-include entirely once stb is present, and provide the impl when we're first.
#ifndef STBI_INCLUDE_STB_IMAGE_H
  #ifndef STB_IMAGE_IMPLEMENTATION
  #define STB_IMAGE_IMPLEMENTATION   // first includer provides the impl
  #endif
  #include "../../../thirdparty/stb_image.h"
#endif
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace us_imgproc {

// round-to-nearest with [0,255] saturation (cv2 saturate_cast<uchar>; ties differ by <=1 LSB, benign).
static inline unsigned char sat_u8(double v) {
    long r = std::lround(v);
    return (unsigned char)(r < 0 ? 0 : (r > 255 ? 255 : r));
}

// one axis of cv2 INTER_AREA general resize: per dst index, the contributing (src_index, weight) list.
struct AreaTap { std::vector<int> si; std::vector<double> w; };
static inline std::vector<AreaTap> area_tab(int ssize, int dsize) {
    double scale = (double)ssize / dsize;
    std::vector<AreaTap> tab(dsize);
    for (int dx = 0; dx < dsize; dx++) {
        double fsx1 = dx * scale, fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, (double)ssize - fsx1);
        int sx1 = (int)std::ceil(fsx1), sx2 = (int)std::floor(fsx2);
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);
        AreaTap& t = tab[dx];
        if (sx1 - fsx1 > 1e-3) { t.si.push_back(sx1 - 1); t.w.push_back((sx1 - fsx1) / cellWidth); }
        for (int sx = sx1; sx < sx2; sx++) { t.si.push_back(sx); t.w.push_back(1.0 / cellWidth); }
        if (fsx2 - sx2 > 1e-3) { t.si.push_back(sx2);
            t.w.push_back(std::min(std::min(fsx2 - sx2, 1.0), cellWidth) / cellWidth); }
    }
    return tab;
}

// cv2 INTER_AREA resize, C channels, uint8 in -> uint8 out. (downscale path; separable 2-pass.)
static inline std::vector<unsigned char> resize_area(const unsigned char* src, int sh, int sw, int C,
                                                     int dh, int dw) {
    std::vector<AreaTap> tx = area_tab(sw, dw), ty = area_tab(sh, dh);
    std::vector<double> tmp((size_t)sh * dw * C, 0.0);          // horizontal: [sh, dw, C]
    for (int y = 0; y < sh; y++)
        for (int dx = 0; dx < dw; dx++) {
            const AreaTap& t = tx[dx];
            double* d = &tmp[((size_t)y * dw + dx) * C];
            for (size_t k = 0; k < t.si.size(); k++) {
                const unsigned char* s = src + ((size_t)y * sw + t.si[k]) * C;
                for (int c = 0; c < C; c++) d[c] += s[c] * t.w[k];
            }
        }
    std::vector<unsigned char> out((size_t)dh * dw * C);        // vertical: [dh, dw, C]
    for (int dy = 0; dy < dh; dy++) {
        const AreaTap& t = ty[dy];
        for (int dx = 0; dx < dw; dx++) {
            double acc[8] = {0,0,0,0,0,0,0,0};
            for (size_t k = 0; k < t.si.size(); k++) {
                const double* s = &tmp[((size_t)t.si[k] * dw + dx) * C];
                for (int c = 0; c < C; c++) acc[c] += s[c] * t.w[k];
            }
            unsigned char* o = &out[((size_t)dy * dw + dx) * C];
            for (int c = 0; c < C; c++) o[c] = sat_u8(acc[c]);
        }
    }
    return out;
}

// cv2 interpolateCubic (A=-0.75): 4 taps at sx-1,sx,sx+1,sx+2 for fractional fx in [0,1).
static inline void cubic_coeffs(double fx, double w[4]) {
    const double A = -0.75;
    w[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
    w[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
    w[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
    w[3] = 1.0 - w[0] - w[1] - w[2];
}
// one axis of cv2 INTER_CUBIC resize: per dst index, 4 src indices (BORDER_REPLICATE) + 4 weights.
struct CubicTap { int si[4]; double w[4]; };
static inline std::vector<CubicTap> cubic_tab(int ssize, int dsize) {
    double scale = (double)ssize / dsize;
    std::vector<CubicTap> tab(dsize);
    for (int dx = 0; dx < dsize; dx++) {
        double fx = (dx + 0.5) * scale - 0.5;
        int sx = (int)std::floor(fx); fx -= sx;
        CubicTap& t = tab[dx];
        cubic_coeffs(fx, t.w);
        for (int k = 0; k < 4; k++) {
            int s = sx - 1 + k;
            t.si[k] = s < 0 ? 0 : (s > ssize - 1 ? ssize - 1 : s);
        }
    }
    return tab;
}
// cv2 INTER_CUBIC resize, C channels, uint8 -> uint8, separable 2-pass.
static inline std::vector<unsigned char> resize_cubic(const unsigned char* src, int sh, int sw, int C,
                                                      int dh, int dw) {
    std::vector<CubicTap> tx = cubic_tab(sw, dw), ty = cubic_tab(sh, dh);
    std::vector<double> tmp((size_t)sh * dw * C);              // horizontal: [sh, dw, C]
    for (int y = 0; y < sh; y++)
        for (int dx = 0; dx < dw; dx++) {
            const CubicTap& t = tx[dx];
            double* d = &tmp[((size_t)y * dw + dx) * C];
            for (int c = 0; c < C; c++) {
                double acc = 0;
                for (int k = 0; k < 4; k++) acc += src[((size_t)y * sw + t.si[k]) * C + c] * t.w[k];
                d[c] = acc;
            }
        }
    std::vector<unsigned char> out((size_t)dh * dw * C);       // vertical: [dh, dw, C]
    for (int dy = 0; dy < dh; dy++) {
        const CubicTap& t = ty[dy];
        for (int dx = 0; dx < dw; dx++) {
            unsigned char* o = &out[((size_t)dy * dw + dx) * C];
            for (int c = 0; c < C; c++) {
                double acc = 0;
                for (int k = 0; k < 4; k++) acc += tmp[((size_t)t.si[k] * dw + dx) * C + c] * t.w[k];
                o[c] = sat_u8(acc);
            }
        }
    }
    return out;
}

// cv2 INTER_NEAREST resize, C channels: src index = min(floor(dx*src/dst), src-1).
static inline std::vector<unsigned char> resize_nearest(const unsigned char* src, int sh, int sw, int C,
                                                        int dh, int dw) {
    double sxr = (double)sw / dw, syr = (double)sh / dh;
    std::vector<unsigned char> out((size_t)dh * dw * C);
    for (int dy = 0; dy < dh; dy++) {
        int sy = std::min((int)std::floor(dy * syr), sh - 1);
        for (int dx = 0; dx < dw; dx++) {
            int sx = std::min((int)std::floor(dx * sxr), sw - 1);
            const unsigned char* s = src + ((size_t)sy * sw + sx) * C;
            unsigned char* o = &out[((size_t)dy * dw + dx) * C];
            for (int c = 0; c < C; c++) o[c] = s[c];
        }
    }
    return out;
}

// Result of ImageProcessorV2: proc_image [3,size,size] and proc_mask [1,size,size], both in [-1,1],
// CHW (W fastest) — ready to feed us_cond::preprocess / mask_valid_tokens.
struct Processed { int size; std::vector<float> image; std::vector<float> mask; };

// recenter(image RGBA uint8 [H,W,4], border_ratio) -> result RGB uint8 [S,S,3] + mask uint8 [S,S].
static inline void recenter(const unsigned char* rgba, int H, int W, double border_ratio,
                            std::vector<unsigned char>& out_rgb, std::vector<unsigned char>& out_mask,
                            int& S) {
    S = std::max(H, W);
    // alpha bbox (rows=x, cols=y), matching np.nonzero(mask)
    int x_min = H, x_max = -1, y_min = W, y_max = -1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (rgba[((size_t)y * W + x) * 4 + 3] != 0) {
                if (y < x_min) x_min = y;
                if (y > x_max) x_max = y;
                if (x < y_min) y_min = x;
                if (x > y_max) y_max = x;
            }
    int h = x_max - x_min, w = y_max - y_min;                  // NOTE: not +1 (matches python)
    if (h <= 0 || w <= 0) throw std::runtime_error("recenter: input image is empty");
    int desired = (int)(S * (1.0 - border_ratio));
    double scale = (double)desired / std::max(h, w);
    int h2 = (int)(h * scale), w2 = (int)(w * scale);
    // crop image[x_min:x_max, y_min:y_max] (h rows, w cols, RGBA)
    std::vector<unsigned char> crop((size_t)h * w * 4);
    for (int yy = 0; yy < h; yy++)
        for (int xx = 0; xx < w; xx++) {
            const unsigned char* s = rgba + ((size_t)(x_min + yy) * W + (y_min + xx)) * 4;
            unsigned char* d = &crop[((size_t)yy * w + xx) * 4];
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
    std::vector<unsigned char> rsz = resize_area(crop.data(), h, w, 4, h2, w2);  // -> [h2,w2,4]
    int x2_min = (S - h2) / 2, y2_min = (S - w2) / 2;
    // composite result over white
    out_rgb.assign((size_t)S * S * 3, 255);                   // bg = white where alpha=0
    out_mask.assign((size_t)S * S, 0);
    for (int yy = 0; yy < h2; yy++)
        for (int xx = 0; xx < w2; xx++) {
            const unsigned char* p = &rsz[((size_t)yy * w2 + xx) * 4];
            int oy = x2_min + yy, ox = y2_min + xx;
            double a = p[3] / 255.0;                            // mask = alpha/255
            unsigned char* o = &out_rgb[((size_t)oy * S + ox) * 3];
            for (int c = 0; c < 3; c++) o[c] = sat_u8(p[c] * a + 255.0 * (1.0 - a));
            out_mask[(size_t)oy * S + ox] = sat_u8(a * 255.0);
        }
}

// Full ImageProcessorV2(path, size, border_ratio): raw PNG -> proc_image/proc_mask in [-1,1] CHW.
static inline Processed process(const std::string& path, int size = 1024, double border_ratio = 0.15) {
    int W, H, c;
    unsigned char* d = stbi_load(path.c_str(), &W, &H, &c, 4);  // force RGBA (alpha=255 if absent)
    if (!d) throw std::runtime_error("us_imgproc: failed to load " + path + " (" + stbi_failure_reason() + ")");
    std::vector<unsigned char> rgba(d, d + (size_t)W * H * 4);
    stbi_image_free(d);

    std::vector<unsigned char> rc_rgb, rc_mask; int S;
    recenter(rgba.data(), H, W, border_ratio, rc_rgb, rc_mask, S);

    std::vector<unsigned char> img = resize_cubic(rc_rgb.data(), S, S, 3, size, size);     // [size,size,3]
    std::vector<unsigned char> msk = resize_nearest(rc_mask.data(), S, S, 1, size, size);  // [size,size,1]

    Processed out; out.size = size;
    out.image.resize((size_t)3 * size * size);
    out.mask.resize((size_t)size * size);
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            for (int ch = 0; ch < 3; ch++)
                out.image[(size_t)ch * size * size + (size_t)y * size + x] =
                    img[((size_t)y * size + x) * 3 + ch] / 255.0f * 2.0f - 1.0f;
            out.mask[(size_t)y * size + x] = msk[(size_t)y * size + x] / 255.0f * 2.0f - 1.0f;
        }
    return out;
}

}  // namespace us_imgproc
