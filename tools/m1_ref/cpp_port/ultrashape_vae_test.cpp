// Validate the UltraShape ShapeVAE decode port (ultrashape_vae.hpp) against banked goldens.
//   vae_in [1,2048,64] --post_kl+16x transformer--> vae_transformer_out [1,2048,1024]
//   geo_queries [1,4096,3] --fourier+query_proj+cross-attn+ln_post+output_proj--> geo_logits [1,4096,1]
// The geo_decoder consumes the transformer output as its latents (matches latents2mesh).
//   ./build.sh ultrashape_vae_test [cuda]
#include "ultrashape_vae.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/vae_dec_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/vae";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    UsVaeCfg cfg;
    M1Harness H(WDIR, 4096, use_cuda);
    ggml_context* ctx = H.ctx;

    // --- inputs ---
    NpyArray lat = npy_load(std::string(GDIR) + "/vae_in.npy");        // [1,K,64]
    int64_t K = lat.shape[1];
    int64_t lat_ne[4] = {cfg.embed_dim, K, 1, 1};
    ggml_tensor* latents = H.input("latents", 2, lat_ne);             // ggml [64, K]

    NpyArray qn = npy_load(std::string(GDIR) + "/geo_queries.npy");   // [1,Nq,3]
    int64_t Nq = qn.shape[1];
    std::vector<float> qemb = us_fourier_embed(qn.f32(), Nq, cfg);    // [Nq, 51] row-major
    int64_t qe_ne[4] = {cfg.fourier_out(), Nq, 1, 1};
    ggml_tensor* query_embed = H.const_tensor("query_embed", 2, qe_ne, qemb);  // ggml [51, Nq]

    // --- graph: transformer then geo_decoder(query, transformer_out) ---
    ggml_tensor* tr  = us_vae_transformer(H, ctx, cfg, latents);      // [1024, K]
    ggml_set_output(tr);
    ggml_tensor* occ = us_geo_decoder(H, ctx, cfg, query_embed, tr);  // [1, Nq]
    ggml_set_output(occ);

    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, tr);
    ggml_build_forward_expand(gf, occ);
    H.alloc_and_upload(gf);
    H.upload_input_npy(latents, std::string(GDIR) + "/vae_in.npy");
    H.compute(gf);

    printf("[us_vae] backend=%s K=%lld Nq=%lld\n", use_cuda ? "cuda" : "cpu", (long long)K, (long long)Nq);
    CmpStats s1 = compare_to_npy(H, tr,  std::string(GDIR) + "/vae_transformer_out.npy", true, "transformer");
    CmpStats s2 = compare_to_npy(H, occ, std::string(GDIR) + "/geo_logits.npy",          true, "geo_logits");
    bool ok = s1.meanabs < 1e-3 && s2.meanabs < 1e-3;
    printf("[us_vae] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
