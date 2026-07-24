// Feature-preserving Taubin (λ|μ) low-pass smoother for a GLB mesh — the SMOOTH-COARSE-DELIVERY
// post-step for narrow_band_dc_probe. Taubin alternates a shrinking Laplacian pass (+λ) with an
// inflating pass (-μ, |μ|>λ) so it removes high-frequency staircase terracing WITHOUT the volume
// collapse ("mush") of repeated Laplacian smoothing — macro features (eyes/bangs/nose) are kept.
//
// This is a delivery/preview post-process only. It does NOT touch the watertight MC-solid mesh that
// feeds UltraShape/rig; that path (pixal3d_chain.hpp build_mc_remesh) is unchanged.
//
// usage: mesh_taubin in.glb out.glb [iterations=8] [lambda=0.53] [mu=-0.53]
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::fprintf(stderr, "usage: %s in.glb out.glb [iterations=8] [lambda=0.53] [mu=-0.53]\n", argv[0]);
        return 2;
    }
    const int   iters  = argc >= 4 ? std::atoi(argv[3]) : 8;
    const float lambda = argc >= 5 ? std::strtof(argv[4], nullptr) : 0.53f;
    const float mu     = argc >= 6 ? std::strtof(argv[5], nullptr) : -0.53f;

    glb::Mesh m;
    if (!glb::read_glb(argv[1], m)) return 1;
    const size_t V = m.verts.size() / 3, F = m.faces.size() / 3;
    if (V == 0 || F == 0) { std::fprintf(stderr, "empty mesh\n"); return 1; }

    // Uniform-weight (umbrella) vertex adjacency from the triangle edges. A CSR layout keeps the
    // per-iteration Laplacian cache-friendly on the ~2.8M-vertex DC delivery mesh.
    std::vector<std::unordered_set<int>> adjset(V);
    auto add = [&](int a, int b){ if (a != b) { adjset[a].insert(b); adjset[b].insert(a); } };
    for (size_t f = 0; f < F; ++f) {
        int a = (int)m.faces[f*3], b = (int)m.faces[f*3+1], c = (int)m.faces[f*3+2];
        add(a,b); add(b,c); add(c,a);
    }
    std::vector<int> off(V+1, 0), nbr;
    for (size_t i = 0; i < V; ++i) off[i+1] = off[i] + (int)adjset[i].size();
    nbr.reserve(off[V]);
    for (size_t i = 0; i < V; ++i) for (int j : adjset[i]) nbr.push_back(j);
    std::vector<std::unordered_set<int>>().swap(adjset);

    std::vector<float>& p = m.verts;
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

    std::fprintf(stderr, "[taubin] V=%zu F=%zu iters=%d lambda=%.3f mu=%.3f -> %s\n",
                 V, F, iters, lambda, mu, argv[2]);
    if (!glb::write_glb(argv[2], m.verts, m.faces)) return 1;
    return 0;
}
