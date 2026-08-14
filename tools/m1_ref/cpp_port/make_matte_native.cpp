// make_matte_native — RMBG-2.0 background matte, in-process, no container and no service.
//
// Replaces `shootout/matte_cpp.sh in.png out.png`, which launched a docker container per image.
// Same output contract: an RGBA PNG whose RGB is already composited over black (rgb*alpha) and
// whose A is the model's soft matte, at the 1024-long-edge scale — the input
// image_io.hpp's pixal_preprocess_black_matte expects.
//
//   ./build.sh make_matte_native cuda
//   CUDA_VISIBLE_DEVICES=<3060 uuid> ./make_matte_native in.png out_rgba.png
//
// usage: make_matte_native <in.(png|jpg)> <out_rgba.png> [options]
//   --gguf <file>   weights (default RMBG-2.0-F16.gguf; env MATTE_NATIVE_GGUF)
//   --cpu           run on the CPU backend instead of the GPU
//   --repeat <n>    matte the same image n times (shows the amortised per-image cost once the
//                   model is resident — the whole point of running in-process)
//   --quiet         no progress chatter
//   --framed <png>  ALSO write the model-facing framed image — i.e. exactly what image_to_rig
//                   would feed its condition models via matte_native::load_pixal_matte_chw
//                   (classify -> matte if needed -> pixal_preprocess_black_matte crop). Prints
//                   input_kind and the framed extent.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "matte_native_imgio.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../thirdparty/stb_image_write.h"

static int usage() {
    std::printf(
        "usage: make_matte_native <in.png> <out_rgba.png> [--gguf <file>] [--cpu] [--repeat n] "
        "[--quiet]\n");
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string in = argv[1];
    const std::string out = argv[2];

    matte_native::Options opt;
    opt.verbose = true;
    int repeat = 1;
    std::string framed_out;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--framed" && i + 1 < argc) {
            framed_out = argv[++i];
        } else if (a == "--gguf" && i + 1 < argc) {
            opt.gguf = argv[++i];
        } else if (a == "--cpu") {
            opt.gpu = false;
        } else if (a == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (a == "--quiet") {
            opt.verbose = false;
        } else {
            std::fprintf(stderr, "make_matte_native: unknown argument '%s'\n", a.c_str());
            return usage();
        }
    }

    try {
        matte_native::MatteRGBA m;
        double total_ms = 0.0;
        for (int i = 0; i < std::max(1, repeat); i++) {
            const auto t0 = std::chrono::steady_clock::now();
            m = matte_native::matte_file(in, opt);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            total_ms += ms;
            std::printf(
                "iter=%d wall_ms=%.0f load_ms=%.0f compute_ms=%.0f\n", i, ms,
                matte_native::last_load_ms(), matte_native::last_compute_ms());
        }
        std::printf("prescale_size=%dx%d\n", m.w, m.h);
        std::printf("has_alpha=%d\n", m.ran_model ? 0 : 1);
        std::printf("model_run=%d\n", m.ran_model ? 1 : 0);
        std::printf("gguf=%s\n", matte_native::resolve_gguf_path(opt).c_str());
        std::printf("mean_wall_ms=%.0f\n", total_ms / (double)std::max(1, repeat));

        if (!stbi_write_png(out.c_str(), m.w, m.h, 4, m.rgba.data(), m.w * 4)) {
            std::fprintf(stderr, "make_matte_native: write failed %s\n", out.c_str());
            return 1;
        }
        std::printf("[make_matte_native] %s -> %s (%dx%d RGBA, black-composited)\n", in.c_str(),
                    out.c_str(), m.w, m.h);

        if (!framed_out.empty()) {
            matte_native::InputKind kind = matte_native::InputKind::opaque;
            const imgio::Image f =
                matte_native::preprocess_black_matte(imgio::load_rgb01(in), opt, &kind);
            std::vector<unsigned char> px((size_t)f.w * f.h * 3);
            for (size_t i = 0; i < px.size(); i++) {
                px[i] = matte_native::detail::round_u8(f.rgb[i]);
            }
            if (!stbi_write_png(framed_out.c_str(), f.w, f.h, 3, px.data(), f.w * 3)) {
                std::fprintf(stderr, "make_matte_native: write failed %s\n", framed_out.c_str());
                return 1;
            }
            std::printf("input_kind=%s\nframed_size=%dx%d\nframed=%s\n",
                        matte_native::to_string(kind), f.w, f.h, framed_out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "make_matte_native: %s\n", e.what());
        return 1;
    }
    return 0;
}
