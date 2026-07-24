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
#include <queue>

namespace imgio {

struct Image {
    int w=0, h=0;
    std::vector<float> rgb;      // HWC, [0,1], premultiplied onto black when alpha exists
    std::vector<float> alpha;    // HWC alpha sidecar when the caller supplied a real cutout
    bool has_alpha=false;
};

inline Image load_rgb01(const std::string& path) {
    int w, h, c;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, 4);  // retain a real cutout's alpha
    if (!d) throw std::runtime_error("failed to load image: " + path + " (" + stbi_failure_reason() + ")");
    Image im; im.w = w; im.h = h; im.rgb.resize((size_t)w*h*3);
    bool varied_alpha = (c == 2 || c == 4);
    if (varied_alpha) for (int i = 0; i < w*h; i++) if (d[(size_t)i*4+3] == 255) {} else { im.has_alpha = true; break; }
    if (im.has_alpha) im.alpha.resize((size_t)w*h);
    for (int i = 0; i < w*h; i++) {
        const float a = d[(size_t)i*4+3] / 255.0f;
        for (int k = 0; k < 3; k++) im.rgb[(size_t)i*3+k] = (d[(size_t)i*4+k] / 255.0f) * a;
        if (im.has_alpha) im.alpha[(size_t)i] = a;
    }
    stbi_image_free(d);
    return im;
}

// Pixal's preprocess_image crop, bit-matching PIL Image.crop's float-box semantics.
// Python: center=((x0+x1)/2, (y0+y1)/2) [FLOAT], size=int(max(x1-x0,y1-y0)*1.1),
//   box=(cx-size//2, cy-size//2, cx+size//2, cy+size//2), then output.crop(box).
// PIL crop rounds each float box edge to the nearest integer (round-half-to-EVEN) and
// zero-pads outside — verified empirically. Native previously used an INTEGER center
// ((x0+x1)/2), which for an odd (x0+x1) shifts the crop 1px horizontally and injects a
// black edge column vs Python; stage-1 SS is chaotically sensitive to that ~1px frame
// shift (measured ~92%->97% coord overlap). std::nearbyint honours the default
// round-to-nearest-even mode == PIL's banker's rounding.
struct PixCrop { int left, top, w, h; };
static inline PixCrop pixal_square_crop(int x0, int y0, int x1, int y1) {
    const int size = (int)(std::max(x1 - x0, y1 - y0) * 1.1f);  // python int(size*1.1)
    const int hs = size / 2;                                    // python size//2
    const double cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0;    // python float center
    const int left  = (int)std::nearbyint(cx - hs), right  = (int)std::nearbyint(cx + hs);
    const int top   = (int)std::nearbyint(cy - hs), bottom = (int)std::nearbyint(cy + hs);
    return { left, top, std::max(1, right - left), std::max(1, bottom - top) };
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

// ---- PIL-EXACT 8-bit fixed-point Lanczos-3 resampler (parity-critical) ----
// Pixal3D's Python cond front-end resizes UINT8 PIL images with Image.resize(LANCZOS).
// PIL's ImagingResample uses INT coefficients (PRECISION_BITS) and rounds the horizontal
// pass to a UINT8 intermediate BEFORE the vertical pass — NOT float intermediates. A float
// resampler (however "accurate") diverges from PIL by up to ~7/255 per pixel; that ~1e-3
// image delta feeds the (otherwise bit-exact) DINOv3 encoder and chaotically flips ~2-3% of
// the SS occupancy voxels (arms-spread vs Python's arms-down). Matching PIL bit-for-bit is the
// last shape/pose parity gap. Verified: this reproduces Image.resize(LANCZOS) with maxabs=0.
static const int PIL_PREC_BITS = 32 - 8 - 2;   // = 22
static inline void normalize_coeffs_8bpc(const std::vector<double>& w, int out, int maxk,
                                         std::vector<int>& ik) {
    ik.assign((size_t)out * maxk, 0);
    const double scale = (double)(1 << PIL_PREC_BITS);
    for (int xx = 0; xx < out; xx++)
        for (int t = 0; t < maxk; t++) {
            double v = w[(size_t)xx * maxk + t];
            ik[(size_t)xx * maxk + t] = (int)(v < 0 ? (-0.5 + v * scale) : (0.5 + v * scale));
        }
}
static inline unsigned char pil_clip8(long long in) {
    if (in >= ((long long)1 << PIL_PREC_BITS << 8)) return 255;  // >= 256<<PREC
    if (in <= 0) return 0;
    return (unsigned char)(in >> PIL_PREC_BITS);
}
// separable pass over a uint8 plane-interleaved buffer (channels C). Returns uint8.
static inline std::vector<unsigned char> pil_pass_h(const std::vector<unsigned char>& src, int sw, int sh,
        int C, int dw, const std::vector<int>& start, const std::vector<int>& width,
        const std::vector<int>& ik, int maxk) {
    std::vector<unsigned char> out((size_t)dw * sh * C);
    const long long bias = (long long)1 << (PIL_PREC_BITS - 1);
    for (int y = 0; y < sh; y++)
        for (int x = 0; x < dw; x++) {
            const int* kk = &ik[(size_t)x * maxk];
            const int xs = start[x], kw = width[x];
            for (int c = 0; c < C; c++) {
                long long acc = bias;
                for (int t = 0; t < kw; t++) acc += (long long)src[((size_t)y * sw + (xs + t)) * C + c] * kk[t];
                out[((size_t)y * dw + x) * C + c] = pil_clip8(acc);
            }
        }
    return out;
}
static inline std::vector<unsigned char> pil_pass_v(const std::vector<unsigned char>& src, int sw, int sh,
        int C, int dh, const std::vector<int>& start, const std::vector<int>& width,
        const std::vector<int>& ik, int maxk) {
    std::vector<unsigned char> out((size_t)sw * dh * C);
    const long long bias = (long long)1 << (PIL_PREC_BITS - 1);
    for (int y = 0; y < dh; y++) {
        const int* kk = &ik[(size_t)y * maxk];
        const int ys = start[y], kw = width[y];
        for (int x = 0; x < sw; x++)
            for (int c = 0; c < C; c++) {
                long long acc = bias;
                for (int t = 0; t < kw; t++) acc += (long long)src[((size_t)(ys + t) * sw + x) * C + c] * kk[t];
                out[((size_t)y * sw + x) * C + c] = pil_clip8(acc);
            }
    }
    return out;
}
static inline void u8_round(const std::vector<float>& f, std::vector<unsigned char>& u8) {
    u8.resize(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        int v = (int)std::lround(f[i] * 255.0f);
        u8[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}

// resize HWC [0,1] -> HWC [0,1], bit-matching PIL Image.resize(LANCZOS) on a uint8 image.
// The float rgb/alpha are quantized to uint8 (as PIL's uint8 image), resampled in PIL's exact
// 8-bit fixed-point (uint8 intermediate between passes), then returned as k/255 floats — so the
// downstream ImageNet-normalize matches Python's uint8/255 path bit-for-bit.
inline Image resize_lanczos3(const Image& src, int dw, int dh) {
    std::vector<int> hs, hw_, vs, vw_; std::vector<double> hwt, vwt; int hk, vk;
    build_coeffs(src.w, dw, hs, hw_, hwt, hk);
    build_coeffs(src.h, dh, vs, vw_, vwt, vk);
    std::vector<int> hik, vik;
    normalize_coeffs_8bpc(hwt, dw, hk, hik);
    normalize_coeffs_8bpc(vwt, dh, vk, vik);

    // RGB: quantize -> horizontal (uint8) -> vertical (uint8) -> float k/255
    std::vector<unsigned char> rgb_u8; u8_round(src.rgb, rgb_u8);
    std::vector<unsigned char> tmp = pil_pass_h(rgb_u8, src.w, src.h, 3, dw, hs, hw_, hik, hk);
    std::vector<unsigned char> res = pil_pass_v(tmp, dw, src.h, 3, dh, vs, vw_, vik, vk);
    Image out; out.w = dw; out.h = dh; out.rgb.resize((size_t)dw * dh * 3);
    for (size_t i = 0; i < out.rgb.size(); i++) out.rgb[i] = res[i] / 255.0f;

    if (src.has_alpha) {
        std::vector<unsigned char> a_u8; u8_round(src.alpha, a_u8);
        std::vector<unsigned char> at = pil_pass_h(a_u8, src.w, src.h, 1, dw, hs, hw_, hik, hk);
        std::vector<unsigned char> ar = pil_pass_v(at, dw, src.h, 1, dh, vs, vw_, vik, vk);
        out.has_alpha = true; out.alpha.resize((size_t)dw * dh);
        for (size_t i = 0; i < out.alpha.size(); i++) out.alpha[i] = ar[i] / 255.0f;
    }
    return out;
}

// Pixal3D does not feed its DINO conditioner the caller's canvas verbatim.  It
// first constrains the long edge to 1024, asks RMBG for a silhouette, then crops
// a square with 10% context.  The native runbook already normalizes cutouts to
// an opaque black matte, so recover the same silhouette deterministically from
// the border-connected near-black background.  This deliberately preserves
// detached black hair/clothing/pupil pixels: only black connected to the image
// border is considered background.
inline Image pixal_preprocess_black_matte(Image im) {
    const int longest = std::max(im.w, im.h);
    if (longest > 1024) {
        const double scale = 1024.0 / (double)longest;
        const int rw = std::max(1, (int)((double)im.w * scale));  // PIL int()
        const int rh = std::max(1, (int)((double)im.h * scale));
        im = resize_lanczos3(im, rw, rh);
    }

    // An RGBA cutout is the exact model-agnostic contract: this mirrors
    // Pixal's alpha path, avoiding a second background-removal model entirely.
    if (im.has_alpha) {
        int x0=im.w, y0=im.h, x1=-1, y1=-1;
        for (int y=0; y<im.h; y++) for (int x=0; x<im.w; x++) if (im.alpha[(size_t)y*im.w+x] > 0.8f) {
            x0=std::min(x0,x); y0=std::min(y0,y); x1=std::max(x1,x); y1=std::max(y1,y);
        }
        if (x1 >= x0 && y1 >= y0) {
            const PixCrop cr = pixal_square_crop(x0, y0, x1, y1);
            Image out; out.w=cr.w; out.h=cr.h;
            out.rgb.assign((size_t)out.w*out.h*3, 0.0f);
            for (int y=0;y<out.h;y++) for (int x=0;x<out.w;x++) {
                const int sx=cr.left+x, sy=cr.top+y;
                if (sx<0||sy<0||sx>=im.w||sy>=im.h) continue;
                for (int c=0;c<3;c++) out.rgb[((size_t)y*out.w+x)*3+c]=im.rgb[((size_t)sy*im.w+sx)*3+c];
            }
            return out;
        }
    }

    int border = 0, near_black_border = 0;
    auto black = [&](int x, int y) {
        const float* p = &im.rgb[((size_t)y * im.w + x) * 3];
        return std::max({p[0], p[1], p[2]}) <= (8.0f / 255.0f);
    };
    for (int y = 0; y < im.h; y++) for (int x = 0; x < im.w; x++) {
        if (x == 0 || y == 0 || x == im.w - 1 || y == im.h - 1) {
            border++;
            if (black(x, y)) near_black_border++;
        }
    }
    // A non-matte image needs the normal caller-supplied preprocessing route;
    // never invent a luminance mask for it here.
    if (!border || near_black_border * 100 < border * 95) return im;

    std::vector<uint8_t> outside((size_t)im.w * im.h, 0);
    std::queue<int> q;
    auto push = [&](int x, int y) {
        const size_t i = (size_t)y * im.w + x;
        if (!outside[i] && black(x, y)) { outside[i] = 1; q.push(y * im.w + x); }
    };
    for (int x = 0; x < im.w; x++) { push(x, 0); push(x, im.h - 1); }
    for (int y = 1; y + 1 < im.h; y++) { push(0, y); push(im.w - 1, y); }
    while (!q.empty()) {
        const int p = q.front(); q.pop(); const int x = p % im.w, y = p / im.w;
        if (x) push(x - 1, y); if (x + 1 < im.w) push(x + 1, y);
        if (y) push(x, y - 1); if (y + 1 < im.h) push(x, y + 1);
    }
    int x0 = im.w, y0 = im.h, x1 = -1, y1 = -1;
    for (int y = 0; y < im.h; y++) for (int x = 0; x < im.w; x++) if (!outside[(size_t)y * im.w + x]) {
        x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
    }
    if (x1 < x0 || y1 < y0) return im;

    // Pixal's `size = int(max(max-min) * 1.1)` + PIL float-box crop (round-half-to-even,
    // black outside fill) — see pixal_square_crop. A FLOAT center is load-bearing: an
    // integer center shifts the frame 1px for odd (x0+x1) and SS is chaotically sensitive.
    const PixCrop cr = pixal_square_crop(x0, y0, x1, y1);
    Image out; out.w = cr.w; out.h = cr.h;
    out.rgb.assign((size_t)out.w * out.h * 3, 0.0f);  // PIL crop's black outside fill
    for (int y = 0; y < out.h; y++) for (int x = 0; x < out.w; x++) {
        const int sx = cr.left + x, sy = cr.top + y;
        if (sx < 0 || sy < 0 || sx >= im.w || sy >= im.h) continue;
        for (int c = 0; c < 3; c++) out.rgb[((size_t)y * out.w + x) * 3 + c] = im.rgb[((size_t)sy * im.w + sx) * 3 + c];
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

// Model-facing Pixal image loader.  Keep `load_chw` available for deliberate
// raw-image utilities, but geometry must use the same framed matte contract as
// Pixal3D's Python pipeline before its 512/1024 condition-model resizes.
inline std::vector<float> load_pixal_matte_chw(const std::string& path, int size) {
    Image im = pixal_preprocess_black_matte(load_rgb01(path));
    return to_chw(resize_lanczos3(im, size, size));
}

}  // namespace imgio
