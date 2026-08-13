// Independent final-texture recovery audit for a textured GLB.
//
// This is deliberately a delivery verifier, not a model: it samples the final
// base-color atlas through deterministic interior points of GLB triangles and
// compares that value with the same sparse PBR volume used to bake it.  It can
// therefore measure the native xatlas bake and the retained Python/CuMesh
// reference under one contract without invoking Python or a GPU.
//
// Usage:
//   atlas_recovery_audit mesh.glb base_color.png pbr_feats.npy pbr_coords.npy
//       [--samples N] [--v-origin gltf|top]
//
// `gltf` is the normal convention: UV V=0 is the lower texture edge while a
// PNG's row zero is its top.  `top` is retained only to diagnose malformed
// historical GLBs.
#define STB_IMAGE_IMPLEMENTATION
#include "../../../thirdparty/stb_image.h"
#include "glb_reader.hpp"
#include "tex_grid_sample.hpp"
#include "../../sparse_spike/npy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Metric {
    double abs_err = 0.0, sq_err = 0.0;
    size_t channels = 0;
    void add(float got, float want) {
        const double e = (double)got - (double)want;
        abs_err += std::fabs(e);
        sq_err += e * e;
        ++channels;
    }
    void print(const char* label) const {
        const double mae = channels ? abs_err / (double)channels : 0.0;
        const double rmse = channels ? std::sqrt(sq_err / (double)channels) : 0.0;
        const double psnr = rmse > 1e-12 ? std::min(99.0, 20.0 * std::log10(1.0 / rmse)) : 99.0;
        std::printf("%s_samples=%zu\n%s_mae=%.8f\n%s_rmse=%.8f\n%s_psnr_db=%.4f\n",
                    label, channels / 3, label, mae, label, rmse, label, psnr);
    }
};

static inline float clamp01(float x) { return std::max(0.f, std::min(1.f, x)); }
static inline float srgb_to_linear(float x) {
    x = clamp01(x);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

static float bilinear(const unsigned char* px, int w, int h, int c, float u, float v, bool gltf_v) {
    u = clamp01(u); v = clamp01(v);
    if (gltf_v) v = 1.f - v;
    const float x = u * (float)w - .5f;
    const float y = v * (float)h - .5f;
    const int x0 = std::max(0, std::min(w - 1, (int)std::floor(x)));
    const int y0 = std::max(0, std::min(h - 1, (int)std::floor(y)));
    const int x1 = std::min(w - 1, x0 + 1);
    const int y1 = std::min(h - 1, y0 + 1);
    const float tx = x - std::floor(x), ty = y - std::floor(y);
    const auto q = [&](int xx, int yy) { return px[((size_t)yy * w + (size_t)xx) * 4 + c] / 255.f; };
    const float a = q(x0, y0) * (1.f - tx) + q(x1, y0) * tx;
    const float b = q(x0, y1) * (1.f - tx) + q(x1, y1) * tx;
    return a * (1.f - ty) + b * ty;
}

void usage(const char* exe) {
    std::fprintf(stderr,
                 "usage: %s <mesh.glb> <base_color.png> <pbr_feats.npy> <pbr_coords.npy>"
                 " [--samples N] [--v-origin gltf|top]\n", exe);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) { usage(argv[0]); return 2; }
    const char* glb_path = argv[1];
    const char* png_path = argv[2];
    size_t samples = 200000;
    bool gltf_v = true;
    for (int i = 5; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--samples") && i + 1 < argc) {
            const long long n = std::atoll(argv[++i]);
            if (n <= 0) { std::fprintf(stderr, "--samples must be positive\n"); return 2; }
            samples = (size_t)n;
        } else if (!std::strcmp(argv[i], "--v-origin") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "gltf") gltf_v = true;
            else if (value == "top") gltf_v = false;
            else { std::fprintf(stderr, "--v-origin must be gltf or top\n"); return 2; }
        } else { usage(argv[0]); return 2; }
    }

    glb::Mesh mesh;
    if (!glb::read_glb(glb_path, mesh) || mesh.verts.empty() || mesh.uvs.size() != mesh.verts.size() / 3 * 2) {
        std::fprintf(stderr, "need a readable textured GLB with one UV per vertex: %s\n", glb_path);
        return 1;
    }
    int w = 0, h = 0, components = 0;
    unsigned char* png = stbi_load(png_path, &w, &h, &components, 4);
    if (!png) { std::fprintf(stderr, "could not decode %s: %s\n", png_path, stbi_failure_reason()); return 1; }
    NpyArray feats_n = npy_load(argv[3]);
    NpyArray coords_n = npy_load(argv[4]);
    if (feats_n.shape.size() != 2 || feats_n.shape[1] < 3 || coords_n.shape.size() != 2 ||
        coords_n.shape[0] != feats_n.shape[0] || coords_n.shape[1] < 4) {
        std::fprintf(stderr, "expected feats[N,C>=3] and coords[N,4+] with matching N\n");
        stbi_image_free(png); return 1;
    }
    const int n = (int)feats_n.shape[0], channels = (int)feats_n.shape[1];
    texgs::VolIndex vol(coords_n.i32(), n, (int)coords_n.shape[1], 1);
    const size_t vertices = mesh.verts.size() / 3;
    const size_t faces = mesh.faces.size() / 3;
    if (faces == 0) { std::fprintf(stderr, "GLB has no triangles\n"); stbi_image_free(png); return 1; }
    samples = std::min(samples, faces);
    const size_t stride = std::max<size_t>(1, faces / samples);
    Metric raw, srgb;
    std::vector<float> source((size_t)channels);
    for (size_t k = 0, fi = 0; k < samples; ++k, fi = (fi + stride) % faces) {
        const int64_t ia = mesh.faces[fi * 3], ib = mesh.faces[fi * 3 + 1], ic = mesh.faces[fi * 3 + 2];
        if (ia < 0 || ib < 0 || ic < 0 || (size_t)ia >= vertices || (size_t)ib >= vertices || (size_t)ic >= vertices) continue;
        // Centroids avoid sampling exactly on atlas chart gutters/seams.  The
        // same physical surface point feeds the sparse PBR trilinear sampler.
        float p[3], u = 0.f, v = 0.f;
        for (int d = 0; d < 3; ++d) p[d] = (mesh.verts[(size_t)ia * 3 + d] + mesh.verts[(size_t)ib * 3 + d] + mesh.verts[(size_t)ic * 3 + d]) / 3.f;
        for (int d = 0; d < 2; ++d) {
            const float q = (mesh.uvs[(size_t)ia * 2 + d] + mesh.uvs[(size_t)ib * 2 + d] + mesh.uvs[(size_t)ic * 2 + d]) / 3.f;
            if (d == 0) u = q; else v = q;
        }
        texgs::sample_one(vol, feats_n.f32(), channels,
                          (p[0] + .5f) * 1024.f, (p[1] + .5f) * 1024.f, (p[2] + .5f) * 1024.f,
                          source.data());
        for (int c = 0; c < 3; ++c) {
            const float encoded = bilinear(png, w, h, c, u, v, gltf_v);
            raw.add(encoded, source[c]);
            srgb.add(srgb_to_linear(encoded), source[c]);
        }
    }
    std::printf("schema_version=1\nmesh=%s\natlas=%s\natlas_width=%d\natlas_height=%d\nvertices=%zu\nfaces=%zu\n"
                "sample_triangle_centroids=%zu\nuv_v_origin=%s\nreference=PBR sparse volume sampled at GLB triangle centroids\n",
                glb_path, png_path, w, h, vertices, faces, samples, gltf_v ? "gltf" : "top");
    raw.print("stored_raw");
    srgb.print("gltf_srgb_linear");
    stbi_image_free(png);
    return 0;
}
