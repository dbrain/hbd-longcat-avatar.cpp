// Validate the dense VanillaVolumeDecoder host loop: generate_dense_grid_points -> chunked
// us_geo_decoder -> grid_logits, vs the banked golden. Latents (post vae.forward) are loaded from the
// golden (us_vae_transformer already validated separately). Proves the latents->occupancy-grid bridge.
//   ./build.sh ultrashape_volume_test [cuda]
#include "ultrashape_vae.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/volume_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/vae";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const int OCT = 64;
    const float BNDS = 1.01f;
    const int64_t CHUNK = 16384;
    UsVaeCfg cfg;
    M1Harness H(WDIR, 4096, use_cuda);
    ggml_context* ctx = H.ctx;

    // latents (post-transformer) as a persistent const [width, K]
    NpyArray lt = npy_load(std::string(GDIR) + "/latents_tr.npy");   // [1,K,1024]
    int64_t K = lt.shape[1];
    std::vector<float> latv(lt.f32(), lt.f32() + lt.numel());
    int64_t lat_ne[4] = {cfg.width, K, 1, 1};
    ggml_tensor* latents = H.const_tensor("latents", 2, lat_ne, latv);

    // query_embed input [51, CHUNK]
    int64_t qe_ne[4] = {cfg.fourier_out(), CHUNK, 1, 1};
    ggml_tensor* query_embed = H.input("query_embed", 2, qe_ne);

    ggml_tensor* occ = us_geo_decoder(H, ctx, cfg, query_embed, latents);  // [1, CHUNK]
    ggml_set_output(occ);

    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, occ);
    H.alloc_and_upload(gf);

    int G;
    std::vector<float> grid = us_dense_grid_queries(OCT, BNDS, G);   // [G^3, 3]
    int64_t Ngrid = (int64_t)G * G * G;
    std::vector<float> logits(Ngrid, 0.0f);

    for (int64_t s = 0; s < Ngrid; s += CHUNK) {
        int64_t n = std::min(CHUNK, Ngrid - s);
        // host-fourier this chunk (pad to CHUNK with zeros for the tail)
        std::vector<float> qchunk((size_t)CHUNK * 3, 0.0f);
        std::copy(grid.begin() + s * 3, grid.begin() + (s + n) * 3, qchunk.begin());
        std::vector<float> qe = us_fourier_embed(qchunk.data(), CHUNK, cfg);  // [CHUNK,51]
        H.upload_input_raw(query_embed, qe);
        H.compute(gf);
        std::vector<float> got(CHUNK);
        ggml_backend_tensor_get(occ, got.data(), 0, CHUNK * sizeof(float));
        std::copy(got.begin(), got.begin() + n, logits.begin() + s);
    }

    // compare to golden grid_logits [1,G,G,G]
    NpyArray gl = npy_load(std::string(GDIR) + "/grid_logits.npy");
    double maxabs = 0, sum = 0, dot = 0, ng = 0, nr = 0;
    const float* r = gl.f32();
    for (int64_t i = 0; i < Ngrid; i++) {
        double d = std::fabs((double)logits[i] - r[i]);
        if (d > maxabs) maxabs = d;
        sum += d; dot += (double)logits[i]*r[i]; ng += (double)logits[i]*logits[i]; nr += (double)r[i]*r[i];
    }
    double cosine = dot / (std::sqrt(ng)*std::sqrt(nr) + 1e-30);
    printf("[us_vol] backend=%s G=%d N=%lld  cosine=%.6f maxabs=%.3e meanabs=%.3e\n",
           use_cuda ? "cuda" : "cpu", G, (long long)Ngrid, cosine, maxabs, sum / Ngrid);
    bool ok = (sum / Ngrid) < 1e-3;
    printf("[us_vol] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
