// C++ replacement for the non-neural halves of shootout/birefnet_matte.py.
//
// birefnet_matte.py did three things: (1) PIL-LANCZOS the source down to a 1024 long edge,
// (2) run ZhengPeng7/BiRefNet under PyTorch, (3) composite rgb*alpha over black and pack the
// soft alpha into an RGBA that native's has_alpha crop path (image_io.hpp
// pixal_preprocess_black_matte) consumes.  Step (2) is now vision.cpp's GGML BiRefNet
// (vision-cli birefnet, same ZhengPeng7 weights converted to GGUF); this tool owns (1) and (3)
// so the lane carries no torch/transformers import at all.
//
//   birefnet_prep prescale <in.(png|jpg)> <out_rgb.png>
//       Python's `scale = min(1, 1024/max(size)); img.resize(..., LANCZOS)`.  Uses image_io's
//       PIL-exact 8-bit fixed-point Lanczos-3, so the bytes handed to the matting model match
//       what PIL would have produced.  Prints `prescale_size=WxH` and `has_alpha=0|1`.
//       When the source already carries a real (non-opaque) alpha, BiRefNet is not needed:
//       pass --emit-rgba to write the finished RGBA directly (Python's has_alpha branch).
//
//   birefnet_prep compose <rgb.png> <mask.png> <out_rgba.png>
//       RGB = trunc(rgb * alpha)  (numpy `.astype(np.uint8)` truncates; do NOT round),
//       A   = mask.  Byte-exact with birefnet_matte.py given the same mask.
//
// Deps: header-only stb (already vendored) + image_io.hpp.  Build: ./build.sh birefnet_prep
// <algorithm> first: image_io.hpp uses std::max({a,b,c}) without including it itself.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "image_io.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../thirdparty/stb_image_write.h"

using imgio::Image;

static int usage() {
    std::printf(
        "usage: birefnet_prep prescale [--emit-rgba] <in.png> <out.png>\n"
        "       birefnet_prep compose <rgb.png> <mask.png> <out_rgba.png>\n");
    return 1;
}

// Python: `np.clip(x, 0, 255).astype(np.uint8)` -> truncate toward zero.
static inline unsigned char trunc_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return (unsigned char)(int)v;
}

static int cmd_prescale(int argc, char** argv) {
    int a = 0;
    bool emit_rgba = false;
    if (a < argc && !std::strcmp(argv[a], "--emit-rgba")) { emit_rgba = true; a++; }
    if (argc - a < 2) return usage();
    const char* in = argv[a];
    const char* out = argv[a + 1];

    // load_rgb01 returns rgb ALREADY premultiplied by alpha plus the alpha sidecar, which is
    // exactly the `rgb*alpha` product Python composites; for an opaque source alpha==1 so the
    // RGB is untouched.
    Image im = imgio::load_rgb01(in);
    const int longest = std::max(im.w, im.h);
    if (longest > 1024) {
        const double scale = 1024.0 / (double)longest;   // python: min(1.0, 1024/max_size)
        const int rw = std::max(1, (int)((double)im.w * scale));  // python int() truncation
        const int rh = std::max(1, (int)((double)im.h * scale));
        im = imgio::resize_lanczos3(im, rw, rh);
    }

    std::printf("prescale_size=%dx%d\n", im.w, im.h);
    std::printf("has_alpha=%d\n", im.has_alpha ? 1 : 0);

    const bool rgba = emit_rgba && im.has_alpha;
    const int C = rgba ? 4 : 3;
    std::vector<unsigned char> px((size_t)im.w * im.h * C);
    for (size_t i = 0; i < (size_t)im.w * im.h; i++) {
        for (int k = 0; k < 3; k++) {
            px[i * C + k] = (unsigned char)std::lround(im.rgb[i * 3 + k] * 255.0f);
        }
        if (rgba) px[i * C + 3] = (unsigned char)std::lround(im.alpha[i] * 255.0f);
    }
    if (!stbi_write_png(out, im.w, im.h, C, px.data(), im.w * C)) {
        std::fprintf(stderr, "birefnet_prep: write failed %s\n", out);
        return 1;
    }
    std::printf("[birefnet_prep] %s -> %s (%dx%d %s)\n", in, out, im.w, im.h, rgba ? "RGBA" : "RGB");
    return 0;
}

static int cmd_compose(int argc, char** argv) {
    if (argc < 3) return usage();
    const char* rgb_path = argv[0];
    const char* mask_path = argv[1];
    const char* out_path = argv[2];

    int W = 0, H = 0, C = 0;
    unsigned char* rgb = stbi_load(rgb_path, &W, &H, &C, 3);
    if (!rgb) {
        std::fprintf(stderr, "birefnet_prep: load failed %s (%s)\n", rgb_path, stbi_failure_reason());
        return 1;
    }
    int mw = 0, mh = 0, mc = 0;
    unsigned char* mask = stbi_load(mask_path, &mw, &mh, &mc, 1);
    if (!mask) {
        std::fprintf(stderr, "birefnet_prep: load failed %s (%s)\n", mask_path, stbi_failure_reason());
        return 1;
    }
    if (mw != W || mh != H) {
        std::fprintf(stderr, "birefnet_prep: mask %dx%d does not match image %dx%d\n", mw, mh, W, H);
        return 1;
    }

    std::vector<unsigned char> out((size_t)W * H * 4);
    for (size_t i = 0; i < (size_t)W * H; i++) {
        const float alpha = mask[i] / 255.0f;
        for (int k = 0; k < 3; k++) out[i * 4 + k] = trunc_u8(rgb[i * 3 + k] * alpha);
        out[i * 4 + 3] = mask[i];
    }
    stbi_image_free(rgb);
    stbi_image_free(mask);

    if (!stbi_write_png(out_path, W, H, 4, out.data(), W * 4)) {
        std::fprintf(stderr, "birefnet_prep: write failed %s\n", out_path);
        return 1;
    }
    std::printf("[birefnet_prep] %s x %s -> %s (%dx%d RGBA, black-composited)\n",
                rgb_path, mask_path, out_path, W, H);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    if (cmd == "prescale") return cmd_prescale(argc - 2, argv + 2);
    if (cmd == "compose") return cmd_compose(argc - 2, argv + 2);
    return usage();
}
