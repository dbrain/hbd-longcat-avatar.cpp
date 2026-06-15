// vecset_encode.cpp — R1 spike harness for the SkinTokens VecSet mesh-condition encoder
// (HANDOFF-RIGGING-skintokens.md R1: "port the VecSet mesh encoder and validate its condition
// features vs an R0 golden dump to fp32-oracle tol BEFORE committing to the rest").
//
// STATUS: scaffolding that COMPILES and runs the full ggml encoder graph. The two GPU/torch
// dependencies — (a) the R0 golden dump, (b) the GGUF-packed weights — are STOP points marked
// TODO below; provide them and this validates end-to-end. The Fourier embed (host, deterministic)
// is fully implemented here; the host FPS + numpy-PCG64 rng.choice sampling port is a separate
// CPU sub-task (TODO) sidestepped by consuming the golden sampled points for now.
//
//   build: ./build.sh vecset_encode          (CPU)   |   ./build.sh vecset_encode cuda
//   run:   PIXAL3D_GGUF_DIR=<dir-with-skintokens.gguf> ./vecset_encode [golden_dir] [cuda]
//
// Validation target: encoder latents [width=512, Q=512]  ==  golden mesh_encoder.encoder output
// (the features encode_mesh_cond feeds to output_proj). fp32-oracle tol (cosine/maxabs).
#include "vecset_encoder.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

// ---- host Fourier embed (FourierEmbedder, exact). logspace freqs 2^[0..F), include_pi=FALSE,
//      include_input=TRUE.  out per point = [ xyz(3), sin(embed)(3F), cos(embed)(3F) ] then we
//      concat the 3 normal feats -> in_dim. embed[c*F+k] = coord_c * freq_k.  pts row-major [N,3]. ----
static std::vector<float> fourier_embed_with_feats(const float* pts, const float* feats, int64_t N,
                                                   const VecsetCfg& cfg) {
    const int F = cfg.num_freqs;
    std::vector<float> freq(F);
    for (int k = 0; k < F; ++k) {
        float f = std::pow(2.0f, (float) k);          // logspace=True
        if (cfg.include_pi) f *= (float) M_PI;        // include_pi=False here
        freq[k] = f;
    }
    const int in_dim = cfg.in_dim();                  // 3 + 6F + point_feats
    std::vector<float> out((size_t) N * in_dim);
    for (int64_t i = 0; i < N; ++i) {
        const float* p = pts + i * 3;
        float* o = out.data() + i * in_dim;
        int w = 0;
        if (cfg.include_input) for (int c = 0; c < 3; ++c) o[w++] = p[c];   // xyz
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) o[w++] = std::sin(p[c] * freq[k]); // sin
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) o[w++] = std::cos(p[c] * freq[k]); // cos
        if (feats) for (int c = 0; c < cfg.point_feats; ++c) o[w++] = feats[i * 3 + c];            // normals
    }
    return out;
}

// Load an [n,3] npy (f32) into a flat row-major vector; returns rows via out_n.
static bool load_pts(const std::string& path, std::vector<float>& dst, int64_t& out_n) {
    FILE* fp = fopen(path.c_str(), "rb"); if (!fp) return false; fclose(fp);
    NpyArray a = npy_load(path);
    out_n = a.shape.size() >= 2 ? a.shape[a.shape.size() - 2] : a.numel() / 3;
    dst.assign(a.f32(), a.f32() + a.numel());
    return true;
}

int main(int argc, char** argv) {
    std::string golden = (argc > 1 && std::string(argv[1]) != "cuda") ? argv[1] : "refs/skintokens";
    bool use_cuda = (argc > 1 && std::string(argv[argc-1]) == "cuda");

    VecsetCfg cfg;   // dims hard-set from the proven checkpoint (see vecset_encoder.hpp header).
    printf("[vecset] cfg: width=%d heads=%d layers=%d num_freqs=%d include_pi=%d in_dim=%d Q(token_num)=512\n",
           cfg.width, cfg.heads, cfg.n_layers, cfg.num_freqs, (int)cfg.include_pi, cfg.in_dim());

    // ---- R0 GOLDEN INPUTS (STOP POINT: produced by the GPU run; TODO wire the real dump) --------
    // Expected npy in <golden>/:  pc.npy [N,3], feats.npy [N,3] (normals),
    //   sampled_pc.npy [Q,3], sampled_feats.npy [Q,3] (post rng.choice+FPS, seed=0 eval),
    //   latents.npy [Q,512] (mesh_encoder.encoder output, the validation target).
    // The host rng.choice (numpy PCG64) + FPS(ratio=1/4) sampling is a CPU sub-task (TODO):
    //   port it and DROP sampled_pc/sampled_feats inputs (derive them from pc/feats). For the
    //   spike we consume the golden sampled points so R1 isolates the ggml encoder numerics.
    std::vector<float> pc, feats, spc, sfeats; int64_t N = 0, Q = 0, t;
    bool have = load_pts(golden + "/pc.npy", pc, N)
             && load_pts(golden + "/feats.npy", feats, t)
             && load_pts(golden + "/sampled_pc.npy", spc, Q)
             && load_pts(golden + "/sampled_feats.npy", sfeats, t);
    if (!have) {
        printf("[vecset] R0 golden not found under '%s' (need pc/feats/sampled_pc/sampled_feats/latents .npy).\n"
               "         Scaffolding COMPILES; provide the golden dump (GPU stop point) to validate.\n", golden.c_str());
        // Build the graph anyway against a tiny dummy to prove it assembles? No — weights also need
        // the GGUF pack (stop point). Exit cleanly: this is expected until R0 lands.
        return 0;
    }
    printf("[vecset] loaded golden: N=%lld input points, Q=%lld sampled queries\n", (long long)N, (long long)Q);

    std::vector<float> data_embed = fourier_embed_with_feats(pc.data(), feats.data(), N, cfg);     // [in_dim, N]
    std::vector<float> samp_embed = fourier_embed_with_feats(spc.data(), sfeats.data(), Q, cfg);   // [in_dim, Q]

    // ---- ggml graph. Weights via GGUF (PIXAL3D_GGUF_DIR=<dir>/skintokens.gguf) — the R0/pack step
    //      (TODO: pack_gguf.cpp extension that writes mesh_encoder.encoder.* from the .ckpt). --------
    // Weights: GGUF (PIXAL3D_GGUF_DIR) in the real path, OR npy from the golden dir (synthetic
    // CPU parity test, vecset_synth_gen.py). wdir = golden dir; use_gguf overrides when set.
    M1Harness H(golden, 2048, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t dne[4] = {cfg.in_dim(), N, 1, 1};
    int64_t sne[4] = {cfg.in_dim(), Q, 1, 1};
    ggml_tensor* data_in = H.input("data_embed", 2, dne);
    ggml_tensor* samp_in = H.input("sampled_embed", 2, sne);

    ggml_tensor* latents = build_vecset_encoder(H, ctx, cfg, data_in, samp_in);  // [width, Q]
    ggml_set_output(latents);

    ggml_cgraph* gf = new_graph(ctx, 16384);
    ggml_build_forward_expand(gf, latents);
    H.alloc_and_upload(gf);
    H.upload_input_raw(data_in, data_embed);
    H.upload_input_raw(samp_in, samp_embed);
    H.compute(gf);

    printf("[vecset] backend=%s, ran encoder graph -> latents [%lld,%lld]\n",
           use_cuda ? "cuda" : "cpu", (long long)latents->ne[0], (long long)latents->ne[1]);

    // ---- validate vs golden latents (fp32-oracle tol; cosine + maxabs) ----
    std::string lat = golden + "/latents.npy";
    FILE* lf = fopen(lat.c_str(), "rb");
    if (!lf) { printf("[vecset] no latents.npy golden — graph ran; provide it to validate.\n"); return 0; }
    fclose(lf);
    CmpStats s = compare_to_npy(H, latents, lat, true, "enc_latents");
    bool ok = s.maxabs < 2e-3;   // fp32-oracle tol (refine vs the TRUE-fp32 oracle, not bf16 golden)
    printf("[vecset] %s (tol 2e-3 maxabs)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
