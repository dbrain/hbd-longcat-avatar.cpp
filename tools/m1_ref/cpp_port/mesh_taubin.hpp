// Feature-preserving Taubin (λ|μ) low-pass smoother — the SMOOTH-COARSE post-step for the
// narrow-band DC remesh. Taubin alternates a shrinking Laplacian pass (+λ) with an inflating pass
// (-μ, |μ|>λ) so it removes high-frequency staircase terracing WITHOUT the volume collapse
// ("mush") of repeated Laplacian smoothing — macro features (eyes/bangs/nose) are kept.
//
// Shared by the mesh_taubin CLI and image_to_rig --dc-remesh so both smooth identically.
#pragma once
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace meshsmooth {

// In-place Taubin low-pass over `verts` using the umbrella (uniform-weight) Laplacian of `faces`.
inline void taubin(std::vector<float>& verts, const std::vector<int64_t>& faces,
                   int iters, float lambda = 0.53f, float mu = -0.53f) {
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    if (V == 0 || F == 0 || iters <= 0) return;

    // Vertex adjacency from the triangle edges. A CSR layout keeps the per-iteration Laplacian
    // cache-friendly on the ~2.8M-vertex DC delivery mesh.
    std::vector<std::unordered_set<int>> adjset(V);
    auto add = [&](int a, int b){ if (a != b) { adjset[a].insert(b); adjset[b].insert(a); } };
    for (size_t f = 0; f < F; ++f) {
        int a = (int)faces[f*3], b = (int)faces[f*3+1], c = (int)faces[f*3+2];
        add(a,b); add(b,c); add(c,a);
    }
    std::vector<int> off(V+1, 0), nbr;
    for (size_t i = 0; i < V; ++i) off[i+1] = off[i] + (int)adjset[i].size();
    nbr.reserve(off[V]);
    for (size_t i = 0; i < V; ++i) for (int j : adjset[i]) nbr.push_back(j);
    std::vector<std::unordered_set<int>>().swap(adjset);

    std::vector<float>& p = verts;
    std::vector<float> lap(V*3);
    auto pass = [&](float w){
        for (size_t i = 0; i < V; ++i) {
            const int b = off[i], e = off[i+1]; const int deg = e - b;
            if (deg == 0) { lap[i*3]=lap[i*3+1]=lap[i*3+2]=0.f; continue; }
            float sx=0,sy=0,sz=0;
            for (int k = b; k < e; ++k) { int j = nbr[k]; sx+=p[j*3]; sy+=p[j*3+1]; sz+=p[j*3+2]; }
            const float inv = 1.f/deg;
            lap[i*3]   = sx*inv - p[i*3];
            lap[i*3+1] = sy*inv - p[i*3+1];
            lap[i*3+2] = sz*inv - p[i*3+2];
        }
        for (size_t k = 0; k < V*3; ++k) p[k] += w * lap[k];
    };
    for (int it = 0; it < iters; ++it) { pass(lambda); pass(mu); }
}

}  // namespace meshsmooth
