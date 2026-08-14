// matte_native.cpp — the in-process RMBG-2.0 matte. See matte_native.hpp for the contract.
//
// Compiled at C++20 into thirdparty/visioncpp/build/libvisioncpp.a (build_visioncpp.sh) alongside
// the vendored vision.cpp library, so C++17 callers only ever see matte_native.hpp.
//
// TWO THINGS THAT WILL BITE IF CHANGED:
//
// 1. VISP_FLASH_ATTENTION=0 IS REQUIRED. The Swin backbone runs head_dim=32, for which ggml's
//    CUDA flash-attention has no kernel instance: it does not fall back, it hard-aborts inside
//    fattn.cu (BEST_FATTN_KERNEL_NONE). vision.cpp's HTTP server sets this for itself; vision-cli
//    does not, which is why the old shell lane passed `-e VISP_FLASH_ATTENTION=0` to docker. We
//    set it here with overwrite=0, so an explicit setting from the environment still wins.
//
// 2. This does NOT reproduce the vision service's /remove endpoint. That composites through
//    image_estimate_foreground; our pipeline wants the raw soft alpha, and the caller composites
//    rgb*alpha over black (matte_native_imgio.hpp) because that is what
//    image_io.hpp's pixal_preprocess_black_matte consumes downstream.
#include "matte_native.hpp"

#include "visp/image.h"
#include "visp/ml.h"
#include "visp/vision.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace matte_native {
namespace {

using clock_t_ = std::chrono::steady_clock;

double ms_since(clock_t_::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count();
}

struct cached_model {
    std::string path;
    bool gpu = false;
    std::unique_ptr<visp::backend_device> backend;
    std::unique_ptr<visp::birefnet_model> model;
};

// Deliberately leaked (never destroyed at static-destruction time). ggml's CUDA buffers must be
// freed while the CUDA runtime is still alive; a static destructor runs after libcudart has begun
// tearing down, and ggml_backend_cuda_buffer_free_buffer then hits ggml_cuda_error -> ggml_abort,
// i.e. a successful matte would still exit non-zero. Callers that want the VRAM back before exit
// call unload() explicitly.
cached_model& cache() {
    static cached_model* c = new cached_model();
    return *c;
}

double g_load_ms = 0.0;
double g_compute_ms = 0.0;

} // namespace

const char* default_gguf_path() {
    // Already on disk; the matting service's own weights, converted. Nothing downloads at runtime.
    return "/mnt/hdd/3d/avatar-shootout/_weights/gguf/RMBG-2.0-F16.gguf";
}

std::string resolve_gguf_path(const Options& opt) {
    std::string path;
    if (!opt.gguf.empty()) {
        path = opt.gguf;
    } else if (const char* e = std::getenv("MATTE_NATIVE_GGUF"); e && *e) {
        path = e;
    } else {
        path = default_gguf_path();
    }
    // RMBG-2.0 WON THE MATTE A/B and BiRefNet is not a delivery-lane option. The two are the same
    // 585-tensor graph, so a BiRefNet file LOADS AND WORKS — it would just quietly be the model
    // that lost, with no symptom to notice. That is exactly the class of mistake worth one line of
    // stderr, and a warning rather than a refusal because the standalone A/B CLI shares this code.
    const std::string base = path.substr(path.find_last_of('/') + 1);
    if (base.rfind("RMBG", 0) != 0) {
        std::fprintf(stderr,
                     "matte_native: WARNING — matting weights '%s' are not an RMBG-2.0 file. RMBG-2.0 "
                     "is the chosen delivery matte; anything else here is an A/B, not a default.\n",
                     base.c_str());
    }
    return path;
}

void unload() {
    cached_model& c = cache();
    c.model.reset();
    c.backend.reset();
    c.path.clear();
}

double last_load_ms() { return g_load_ms; }
double last_compute_ms() { return g_compute_ms; }

Mask rmbg_alpha(const unsigned char* rgb, int w, int h, const Options& opt) {
    if (!rgb || w <= 0 || h <= 0) {
        throw std::runtime_error("matte_native: empty input image");
    }
    // Trap 1 above. Must be set before the compute graph is built (ml.cpp reads it into
    // model_build_flags), and harmless to set repeatedly.
    ::setenv("VISP_FLASH_ATTENTION", "0", 0);

    const std::string path = resolve_gguf_path(opt);
    cached_model& c = cache();

    g_load_ms = 0.0;
    if (!c.model || c.path != path || c.gpu != opt.gpu) {
        const auto t0 = clock_t_::now();
        unload();
        c.backend = std::make_unique<visp::backend_device>(
            visp::backend_init(opt.gpu ? visp::backend_type::gpu : visp::backend_type::cpu));
        c.model = std::make_unique<visp::birefnet_model>(
            visp::birefnet_load_model(path.c_str(), *c.backend));
        c.path = path;
        c.gpu = opt.gpu;
        g_load_ms = ms_since(t0);
        if (opt.verbose) {
            std::fprintf(
                stderr, "[matte_native] loaded %s on %s in %.0f ms (native res %d)\n", path.c_str(),
                opt.gpu ? "gpu" : "cpu", g_load_ms, c.model->params.image_size);
        }
    }

    const auto t1 = clock_t_::now();
    visp::image_view input({w, h}, visp::image_format::rgb_u8, rgb);
    visp::image_data out = visp::birefnet_compute(*c.model, input);
    g_compute_ms = ms_since(t1);

    Mask mask;
    mask.w = out.extent[0];
    mask.h = out.extent[1];
    if (out.format != visp::image_format::alpha_u8) {
        throw std::runtime_error("matte_native: unexpected mask format from birefnet_compute");
    }
    mask.alpha.assign(out.data.get(), out.data.get() + size_t(mask.w) * size_t(mask.h));
    if (opt.verbose) {
        std::fprintf(
            stderr, "[matte_native] matte %dx%d in %.0f ms\n", mask.w, mask.h, g_compute_ms);
    }
    return mask;
}

} // namespace matte_native
