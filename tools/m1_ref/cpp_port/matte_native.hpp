// matte_native.hpp — RMBG-2.0 background matting, in-process, no service and no container.
//
// This is the neural half of shootout/matte_cpp.sh (which shelled out to
// `docker run … vision-cli birefnet` once per image) reduced to a function call. The graph is
// vision.cpp's BiRefNet implementation, vendored at thirdparty/visioncpp and compiled against
// THIS repo's ggml, so the matte runs on the same CUDA context, in the same process, under the
// same 3060 flock as the rest of image_to_rig.
//
// WEIGHTS: RMBG-2.0-F16.gguf. RMBG-2.0 and BiRefNet are the same graph — 585 tensors, identical
// names and shapes, differing only numerically — so `birefnet_*` loads either file and the model
// choice is purely which GGUF you point at. Default path below; override with the `gguf` field or
// the MATTE_NATIVE_GGUF env var. Nothing is ever downloaded at runtime.
//
// This header is deliberately C++17-clean and free of both <visp/…> and stb: the implementation
// (matte_native.cpp) is C++20 and lives in libvisioncpp.a, so a C++17 caller such as
// image_to_rig.cpp can include this and link the archive.
//
// For the pixel-level entry point that speaks imgio::Image, include matte_native_imgio.hpp.
#pragma once

#include <string>
#include <vector>

namespace matte_native {

struct Options {
    // Path to the matting GGUF. Empty -> MATTE_NATIVE_GGUF, else the built-in default.
    std::string gguf;
    // Run on the GPU (CUDA_VISIBLE_DEVICES pins the card, as everywhere else in this tree).
    bool gpu = true;
    // Chatter on stderr (model load, resolution, timings).
    bool verbose = false;
};

struct Mask {
    int w = 0;
    int h = 0;
    std::vector<unsigned char> alpha; // w*h, 0..255 soft matte
};

// Built-in weights location (no download, no network): the converted RMBG-2.0 already on disk.
const char* default_gguf_path();

// Resolve Options::gguf -> env MATTE_NATIVE_GGUF -> default_gguf_path().
std::string resolve_gguf_path(const Options& opt);

// Run the matting model on a packed 8-bit RGB buffer (w*h*3, no row padding) and return the soft
// alpha at the SAME extent. The model is loaded once per process and cached: the second call on a
// same-sized image reuses both weights and compute graph.
//
// Callers are expected to have already scaled the image to a 1024 long edge (imgio's PIL-exact
// Lanczos); the model itself works at its native square resolution internally.
Mask rmbg_alpha(const unsigned char* rgb, int w, int h, const Options& opt = Options());

// Drop the cached model and free its VRAM. Optional — process exit does the same.
void unload();

// Timings (milliseconds) of the most recent rmbg_alpha() call. load_ms is 0 on a cache hit.
double last_load_ms();
double last_compute_ms();

} // namespace matte_native
