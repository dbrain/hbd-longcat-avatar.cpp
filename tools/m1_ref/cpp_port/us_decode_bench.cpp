// Decode-only micro-bench for the UltraShape VAE dense decode -- the single most expensive stage in
// the pipeline (2529.3s = 53.2% of a 4751.8s prod run, measured 2026-07-17 on the rebased tree).
//
// WHY THIS IS A FAITHFUL PROXY, not a toy: us_geo_decoder is a FIXED-SHAPE graph. Its shape depends
// only on CHUNK and N (the latent count) -- never on the octree G. G sets only the compute COUNT:
// ceil(G^3/CHUNK). So prod wall = (ms per compute) * ceil(G^3/CHUNK), and both factors are measured
// here at prod shapes (N=8192, width=1024) in seconds instead of a 79-minute pipeline run.
//
// Latents are random: the graph is data-independent for TIMING (fixed shapes, no branches on values).
// This bench answers "how fast" only -- correctness of any decode change must be proven against the
// banked golden via ultrashape_volume_test, NOT here.
//
//   ./build.sh us_decode_bench cuda
//   ./us_decode_bench cuda [--chunk N] [--latents N] [--iters N] [--octree G]
#include "ultrashape_vae.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <ctime>

static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/vae";

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char** argv) {
    bool use_cuda = false;
    int64_t CHUNK = 2048;   // prod default (RefineCfg::chunk, commented "(VRAM lever)")
    int64_t N     = 8192;   // prod num_latents
    int     ITERS = 40;     // timed computes
    int     OCT   = 512;    // only used to extrapolate a prod wall estimate
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "cuda") use_cuda = true;
        else if (a == "--chunk" && i + 1 < argc) CHUNK = atoll(argv[++i]);
        else if (a == "--latents" && i + 1 < argc) N = atoll(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) ITERS = atoi(argv[++i]);
        else if (a == "--octree" && i + 1 < argc) OCT = atoi(argv[++i]);
    }

    UsVaeCfg cfg;
    M1Harness H(WDIR, 4096, use_cuda);

    // latents [width, N] as a persistent const -- exactly how the refine holds tr_cached.
    std::vector<float> latv((size_t)cfg.width * N);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : latv) v = nd(rng);
    int64_t lat_ne[4] = {cfg.width, N, 1, 1};
    ggml_tensor* latents = H.const_tensor("latents", 2, lat_ne, std::move(latv));

    int64_t qe_ne[4] = {cfg.fourier_out(), CHUNK, 1, 1};
    ggml_tensor* query_embed = H.input("query_embed", 2, qe_ne);
    ggml_tensor* occ = us_geo_decoder(H, H.ctx, cfg, query_embed, latents);
    ggml_set_output(occ);
    ggml_cgraph* g = new_graph(H.ctx, 32768);
    ggml_build_forward_expand(g, occ);
    H.alloc_and_upload(g);

    // one chunk of queries; identical every iter (timing is data-independent)
    std::vector<float> q((size_t)CHUNK * 3, 0.f);
    for (auto& v : q) v = nd(rng) * 0.5f;
    std::vector<float> qe = us_fourier_embed(q.data(), CHUNK, cfg);
    H.upload_input_raw(query_embed, qe);

    // --faithful: replicate ultrashape_refine.hpp's ACTUAL per-chunk host work (vector allocs +
    // single-threaded us_fourier_embed + readback), not just the GPU compute. The delta between
    // the two modes IS the host-side serial cost the GPU stalls behind.
    const bool faithful = getenv("US_BENCH_FAITHFUL") != nullptr;
    std::vector<float> grid((size_t)CHUNK * 3, 0.f);
    for (auto& v : grid) v = nd(rng) * 0.5f;

    // warmup (first compute pays cuBLAS handle/workspace init)
    for (int i = 0; i < 3; i++) H.compute(g);

    double t0 = now_s();
    for (int i = 0; i < ITERS; i++) {
        if (faithful) {
            std::vector<float> qchunk((size_t)CHUNK * 3, 0.0f);
            std::copy(grid.begin(), grid.end(), qchunk.begin());
            std::vector<float> qef = us_fourier_embed(qchunk.data(), CHUNK, cfg);
            H.upload_input_raw(query_embed, qef);
            H.compute(g);
            std::vector<float> got(CHUNK);
            ggml_backend_tensor_get(occ, got.data(), 0, got.size() * sizeof(float));
        } else {
            H.upload_input_raw(query_embed, qe);
            H.compute(g);
        }
    }
    double dt = now_s() - t0;

    const double ms = 1000.0 * dt / ITERS;
    const long long G = (long long)OCT + 1;
    const long long Ngrid = G * G * G;
    const long long computes = (Ngrid + CHUNK - 1) / CHUNK;
    const double wall = ms * 1e-3 * (double)computes;

    printf("chunk=%-6lld latents=%-5lld  %8.3f ms/compute  %10.0f queries/s"
           "  ->  G=%lld: %lld computes = %.1fs prod wall\n",
           (long long)CHUNK, (long long)N, ms, CHUNK / (ms * 1e-3),
           G, computes, wall);
    return 0;
}
