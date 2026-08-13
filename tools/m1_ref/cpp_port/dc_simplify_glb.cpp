// Native CuMesh QEM simplify probe -- the same simplifier Pixal3D's o_voxel.postprocess.to_glb
// applies to its narrow-band dual-contour output (mesh.simplify(decimation_target)).  The atlas
// path's meshopt/err decimator roughens the DC surface (visible facet ripples); CuMesh's QEM keeps
// it smooth.  This exposes it standalone so the DC delivery surface can be simplified to Python
// parity BEFORE unwrap/bake, instead of relying on texture_rebake_native's --decimate.
//
// Usage: dc_simplify_glb <input.glb> <output.glb> <target_faces> [taubin_iters=0] [lambda=0.5] [mu=0.53]
//
// Optional Taubin (lambda|mu) smoothing runs AFTER the QEM simplify.  Native's coarse decode surface
// is the MC-solid staircase (dihedral p95~79), and DC shrink-wrapping it inherits those grid-scale
// steps (p95~32-44 after simplify), whereas Python's DC input (o_voxel MC decode) is smoother and it
// lands at p95~26 with no post-smooth.  Taubin is a volume-preserving low-pass (unlike plain
// Laplacian it does not shrink) that removes the residual DC staircase without pulling the surface
// off the sparse PBR volume, so the textured surface stays crisp AND reaches Python smoothness.
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include "native_cumesh_bridge.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

// One umbrella (uniform-weight Laplacian) pass with factor `f`: v += f * (mean(neighbours) - v).
static void umbrella_pass(std::vector<float>& v, const std::vector<std::vector<int>>& adj, float f) {
    const size_t n = v.size() / 3;
    std::vector<float> disp(v.size(), 0.f);
    #pragma omp parallel for schedule(dynamic, 8192)
    for (long long i = 0; i < (long long)n; ++i) {
        const auto& nb = adj[(size_t)i];
        if (nb.empty()) continue;
        float c[3] = {0, 0, 0};
        for (int j : nb) { c[0] += v[(size_t)j * 3]; c[1] += v[(size_t)j * 3 + 1]; c[2] += v[(size_t)j * 3 + 2]; }
        const float inv = 1.f / (float)nb.size();
        for (int k = 0; k < 3; ++k) disp[(size_t)i * 3 + k] = f * (c[k] * inv - v[(size_t)i * 3 + k]);
    }
    for (size_t i = 0; i < v.size(); ++i) v[i] += disp[i];
}

static void taubin_smooth(std::vector<float>& verts, const std::vector<int64_t>& faces,
                          int iters, float lambda, float mu) {
    if (iters <= 0) return;
    const size_t nv = verts.size() / 3;
    std::vector<std::vector<int>> adj(nv);
    for (size_t f = 0; f < faces.size(); f += 3) {
        int a = (int)faces[f], b = (int)faces[f + 1], c = (int)faces[f + 2];
        adj[(size_t)a].push_back(b); adj[(size_t)a].push_back(c);
        adj[(size_t)b].push_back(a); adj[(size_t)b].push_back(c);
        adj[(size_t)c].push_back(a); adj[(size_t)c].push_back(b);
    }
    for (auto& nb : adj) { std::sort(nb.begin(), nb.end()); nb.erase(std::unique(nb.begin(), nb.end()), nb.end()); }
    for (int it = 0; it < iters; ++it) {
        umbrella_pass(verts, adj, lambda);   // shrink
        umbrella_pass(verts, adj, -mu);      // un-shrink (mu>lambda) -> band-pass, volume preserving
    }
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 7) {
        std::fprintf(stderr, "usage: %s <input.glb> <output.glb> <target_faces> [taubin_iters=0] [lambda=0.5] [mu=0.53]\n", argv[0]);
        return 2;
    }
    const int target = std::atoi(argv[3]);
    const int taubin_iters = argc >= 5 ? std::atoi(argv[4]) : 0;
    const float lambda = argc >= 6 ? std::strtof(argv[5], nullptr) : 0.5f;
    const float mu = argc >= 7 ? std::strtof(argv[6], nullptr) : 0.53f;
    if (target <= 0) {
        std::fprintf(stderr, "target_faces must be positive\n");
        return 2;
    }
    glb::Mesh in;
    if (!glb::read_glb(argv[1], in)) {
        std::fprintf(stderr, "could not read %s\n", argv[1]);
        return 1;
    }
    std::vector<float> verts;
    std::vector<int64_t> faces;
    try {
        native_cumesh::simplify_to_faces(in.verts, in.faces, target, verts, faces);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native CuMesh simplify failed: %s\n", e.what());
        return 1;
    }
    if (taubin_iters > 0) {
        taubin_smooth(verts, faces, taubin_iters, lambda, mu);
        std::printf("native Taubin smooth: %d iters (lambda=%.3f mu=%.3f)\n", taubin_iters, lambda, mu);
    }
    if (!glb::write_glb(argv[2], verts, faces)) {
        std::fprintf(stderr, "could not write %s\n", argv[2]);
        return 1;
    }
    std::printf("native CuMesh QEM simplify: V %zu -> %zu, F %zu -> %zu, output=%s\n",
                in.verts.size() / 3, verts.size() / 3,
                in.faces.size() / 3, faces.size() / 3, argv[2]);
    return 0;
}
