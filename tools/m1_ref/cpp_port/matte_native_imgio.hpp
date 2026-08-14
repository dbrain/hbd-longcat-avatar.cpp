// matte_native_imgio.hpp — the imgio::Image face of the native RMBG-2.0 matte.
//
// This is shootout/matte_cpp.sh with the shell, the container and the three intermediate PNGs
// removed. It is header-only and lives in the CALLER's translation unit on purpose: image_io.hpp
// instantiates stb_image, so it must be included exactly once per binary, and every caller of
// this glue (image_to_rig.cpp, make_matte_native.cpp) already includes it.
//
// The three steps are unchanged, and steps 1 and 3 are the same code that was verified
// byte-identical to PIL/numpy (birefnet_prep prescale / compose):
//   1. PIL-exact LANCZOS down to a 1024 long edge          imgio::resize_lanczos3
//   2. RMBG-2.0 (BiRefNet graph) on the 3060               matte_native::rmbg_alpha
//   3. rgb*alpha over black + the soft alpha as A          compose_black(), numpy truncation
//
// A source that already carries a real cutout skips the model entirely, exactly as
// `birefnet_prep prescale --emit-rgba` did (Python's has_alpha branch).
#pragma once

#include "image_io.hpp"
#include "matte_native.hpp"

#include <string>
#include <vector>

namespace matte_native {

struct MatteRGBA {
    int w = 0;
    int h = 0;
    std::vector<unsigned char> rgba; // w*h*4, RGB already composited over black
    bool ran_model = false;          // false when the source already had a cutout
    int src_w = 0;
    int src_h = 0;
};

namespace detail {

// numpy `.astype(np.uint8)` truncates toward zero; do NOT round (birefnet_prep::trunc_u8).
inline unsigned char trunc_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return (unsigned char)(int)v;
}

inline unsigned char round_u8(float v) {
    int x = (int)std::lround(v * 255.0f);
    return (unsigned char)(x < 0 ? 0 : (x > 255 ? 255 : x));
}

// Python: `scale = min(1, 1024/max(size)); img.resize(..., LANCZOS)`, with int() truncation
// on the target extent.
inline imgio::Image prescale_1024(imgio::Image im) {
    const int longest = std::max(im.w, im.h);
    if (longest > 1024) {
        const double scale = 1024.0 / (double)longest;
        const int rw = std::max(1, (int)((double)im.w * scale));
        const int rh = std::max(1, (int)((double)im.h * scale));
        im = imgio::resize_lanczos3(im, rw, rh);
    }
    return im;
}

} // namespace detail

// Matte an already-loaded image. `im` is what imgio::load_rgb01 returns (RGB premultiplied by any
// alpha it carried).
inline MatteRGBA matte_image(imgio::Image im, const Options& opt = Options()) {
    MatteRGBA out;
    out.src_w = im.w;
    out.src_h = im.h;
    im = detail::prescale_1024(std::move(im));
    out.w = im.w;
    out.h = im.h;
    out.rgba.assign((size_t)out.w * out.h * 4, 0);

    if (im.has_alpha) {
        // Source is already a cutout: no model, just repack (birefnet_prep prescale --emit-rgba).
        for (size_t i = 0; i < (size_t)out.w * out.h; i++) {
            for (int k = 0; k < 3; k++) {
                out.rgba[i * 4 + k] = detail::round_u8(im.rgb[i * 3 + k]);
            }
            out.rgba[i * 4 + 3] = detail::round_u8(im.alpha[i]);
        }
        out.ran_model = false;
        return out;
    }

    // The exact bytes the old lane wrote to prescaled.png and handed to the model.
    std::vector<unsigned char> rgb8((size_t)im.w * im.h * 3);
    for (size_t i = 0; i < rgb8.size(); i++) {
        rgb8[i] = detail::round_u8(im.rgb[i]);
    }

    const Mask mask = rmbg_alpha(rgb8.data(), im.w, im.h, opt);
    if (mask.w != im.w || mask.h != im.h) {
        throw std::runtime_error("matte_native: mask extent does not match input");
    }

    for (size_t i = 0; i < (size_t)out.w * out.h; i++) {
        const float a = mask.alpha[i] / 255.0f;
        for (int k = 0; k < 3; k++) {
            out.rgba[i * 4 + k] = detail::trunc_u8((float)rgb8[i * 3 + k] * a);
        }
        out.rgba[i * 4 + 3] = mask.alpha[i];
    }
    out.ran_model = true;
    return out;
}

// Matte straight from a file.
inline MatteRGBA matte_file(const std::string& path, const Options& opt = Options()) {
    return matte_image(imgio::load_rgb01(path), opt);
}

//
// --- pipeline entry point ------------------------------------------------------------------
//
// Everything above reproduces matte_cpp.sh, which mattes UNCONDITIONALLY unless the source
// already carries alpha. The geometry pipeline needs one more guard: an input that is already
// model-facing must keep taking the path it takes today, byte-for-byte. So classify first, with
// make_matte --inspect-input's exact two-ring test, and only run the model on inputs that are not
// already a matte.

enum class InputKind { rgba_cutout, black_matte, framed_opaque, opaque };

inline const char* to_string(InputKind k) {
    switch (k) {
        case InputKind::rgba_cutout: return "rgba-cutout";
        case InputKind::black_matte: return "black-matte";
        case InputKind::framed_opaque: return "framed-opaque";
        default: return "opaque";
    }
}

// make_matte.cpp --inspect-input, on an imgio::Image. TWO rings, not one: the outermost 1px ring
// alone is satisfied by a PICTURE FRAME — a studio render with a white background and a thin black
// border passes it, gets called a black-matte, and the pipeline then eats the entire background as
// subject (this is what happened to toy1_matte.png). A real black matte is black on both rings.
inline InputKind classify(const imgio::Image& im) {
    if (im.has_alpha) {
        bool visible = false;
        for (float a : im.alpha) {
            if (a > 8.0f / 255.0f) { visible = true; break; }
        }
        if (visible) return InputKind::rgba_cutout;
    }
    auto ring_black = [&](int inset, int& count) {
        int black = 0;
        count = 0;
        const int x0 = inset, x1 = im.w - 1 - inset, y0 = inset, y1 = im.h - 1 - inset;
        if (x0 >= x1 || y0 >= y1) return 0;
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                if (x != x0 && x != x1 && y != y0 && y != y1) continue;
                count++;
                const float* p = &im.rgb[((size_t)y * im.w + x) * 3];
                if (std::max({p[0], p[1], p[2]}) <= 8.0f / 255.0f) black++;
            }
        }
        return black;
    };
    int border = 0, inner_count = 0;
    const int black = ring_black(0, border);
    const int inset = std::max(2, (int)(std::min(im.w, im.h) * 0.06f));
    const int inner_black = ring_black(inset, inner_count);
    const int outer_pct = border ? black * 100 / border : 0;
    const int inner_pct = inner_count ? inner_black * 100 / inner_count : 0;
    if (outer_pct >= 95 && inner_pct >= 60) return InputKind::black_matte;
    if (outer_pct >= 95) return InputKind::framed_opaque;
    return InputKind::opaque;
}

// The model-facing framed image, from any input: matte first when the input is not already a
// matte, then apply the unchanged pixal_preprocess_black_matte crop. An rgba-cutout or a real
// black-matte never touches the model, so those inputs behave exactly as they do today.
inline imgio::Image preprocess_black_matte(
    imgio::Image im, const Options& opt = Options(), InputKind* out_kind = nullptr) {
    im = detail::prescale_1024(std::move(im));
    const InputKind kind = classify(im);
    if (out_kind) *out_kind = kind;
    if (kind == InputKind::opaque || kind == InputKind::framed_opaque) {
        const MatteRGBA m = matte_image(im, opt);
        im = imgio::Image();
        im.w = m.w;
        im.h = m.h;
        im.rgb.resize((size_t)m.w * m.h * 3);
        const size_t n = (size_t)m.w * m.h;
        im.has_alpha = true;
        im.alpha.resize(n);
        for (size_t i = 0; i < n; i++) {
            const float a = m.rgba[i * 4 + 3] / 255.0f;
            for (int k = 0; k < 3; k++) im.rgb[i * 3 + k] = (m.rgba[i * 4 + k] / 255.0f) * a;
            im.alpha[i] = a;
        }
    }
    return imgio::pixal_preprocess_black_matte(std::move(im));
}

// Drop-in for imgio::load_pixal_matte_chw(): same framing contract, but an un-matted photo is
// matted by RMBG-2.0 in-process instead of being handed to the border-flood heuristic (or to a
// container). The framed image is cached per path because the caller asks for it at two
// resolutions (512 and 1024) and the model must not run twice.
inline std::vector<float> load_pixal_matte_chw(
    const std::string& path, int size, const Options& opt = Options()) {
    static std::string cached_path;
    static imgio::Image cached;
    if (cached_path != path || cached.w == 0) {
        cached = preprocess_black_matte(imgio::load_rgb01(path), opt);
        cached_path = path;
    }
    return imgio::to_chw(imgio::resize_lanczos3(cached, size, size));
}

// Drop-in for `imgio::load_rgb01(path)` in front of pixal_preprocess_black_matte: returns exactly
// the Image that load_rgb01() would return if the matte had been written to an RGBA PNG and read
// back — including the second premultiply load_rgb01 applies (rgb*alpha^2), so the in-memory and
// via-disk paths are bit-identical.
inline imgio::Image load_rgb01_matted(const std::string& path, const Options& opt = Options()) {
    const MatteRGBA m = matte_file(path, opt);
    imgio::Image im;
    im.w = m.w;
    im.h = m.h;
    im.rgb.resize((size_t)m.w * m.h * 3);
    const size_t n = (size_t)m.w * m.h;
    for (size_t i = 0; i < n; i++) {
        if (m.rgba[i * 4 + 3] != 255) {
            im.has_alpha = true;
            break;
        }
    }
    if (im.has_alpha) im.alpha.resize(n);
    for (size_t i = 0; i < n; i++) {
        const float a = m.rgba[i * 4 + 3] / 255.0f;
        for (int k = 0; k < 3; k++) im.rgb[i * 3 + k] = (m.rgba[i * 4 + k] / 255.0f) * a;
        if (im.has_alpha) im.alpha[i] = a;
    }
    return im;
}

} // namespace matte_native
