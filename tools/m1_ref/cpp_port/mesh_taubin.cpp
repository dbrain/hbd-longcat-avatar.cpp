// CLI driver for the feature-preserving Taubin (λ|μ) low-pass smoother. The smoother itself lives
// in mesh_taubin.hpp so image_to_rig --dc-remesh applies exactly the same post-step in-process.
//
// This is a delivery/preview post-process only. It does NOT touch the watertight MC-solid mesh that
// feeds UltraShape/rig on the legacy --clean path (pixal3d_chain.hpp build_mc_remesh) — unchanged.
//
// usage: mesh_taubin in.glb out.glb [iterations=8] [lambda=0.53] [mu=-0.53]
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include "mesh_taubin.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

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

    meshsmooth::taubin(m.verts, m.faces, iters, lambda, mu);

    std::fprintf(stderr, "[taubin] V=%zu F=%zu iters=%d lambda=%.3f mu=%.3f -> %s\n",
                 V, F, iters, lambda, mu, argv[2]);
    if (!glb::write_glb(argv[2], m.verts, m.faces)) return 1;
    return 0;
}
