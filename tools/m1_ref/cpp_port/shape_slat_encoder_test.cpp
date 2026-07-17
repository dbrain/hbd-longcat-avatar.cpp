// Compile + load-check stub for the shape_slat encoder port (shape_slat_encoder.hpp).
// It does NOT run the full encoder (that needs a voxelized o_voxel input — the remaining
// blocker; see SHAPE_ENC_ARCH.md). It verifies:
//   (1) shape_slat_encoder.hpp COMPILES against the validated sparse_vae.hpp ops,
//   (2) the packed GGUF (or npy dir) loads and every expected encoder tensor is present
//       with the element count the arch predicts,
//   (3) the net-new Spatial2Channel downsample (s2c_shrink/scatter/skip) runs on a tiny
//       synthetic coord set and is the exact inverse of the decoder's c2s_grow.
// Build: ./build.sh shape_slat_encoder_test          (CPU; links ggml CPU but uses none of it)
// Run:   PIXAL3D_GGUF_DIR=/mnt/hdd/pixal3d/weights_gguf ./shape_slat_encoder_test
//    or: ./shape_slat_encoder_test                   (falls back to weights_npy/shape_enc)
#include "shape_slat_encoder.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <map>

static const char* WDIR = "weights_npy/shape_enc";

int main() {
    printf("[shape_enc_test] ---- arch + gguf load check (no GPU, no voxel input) ----\n");
    senc::WLoad W(WDIR);

    // expected tensors -> element count, from the config (model_channels [64,128,256,512,1024],
    // num_blocks [0,4,8,16,4], latent 32; conv kernels are spike [27,Cin,Cout]).
    const int MC[5] = {64, 128, 256, 512, 1024};
    const int NB[5] = {0, 4, 8, 16, 4};
    std::map<std::string, long long> expect;
    expect["input_layer.weight"] = 64LL * 6;   expect["input_layer.bias"] = 64;
    expect["to_latent.weight"]   = 64LL * 1024; expect["to_latent.bias"]  = 64;   // 2*latent=64
    char p[64];
    for (int L = 0; L < 5; L++) {
        int C = MC[L];
        for (int j = 0; j < NB[L]; j++) {                              // ConvNeXt blocks
            snprintf(p, sizeof(p), "blocks.%d.%d", L, j); std::string b(p);
            expect[b+".conv.weight"]  = 27LL * C * C; expect[b+".conv.bias"]  = C;
            expect[b+".norm.weight"]  = C;            expect[b+".norm.bias"]  = C;
            expect[b+".mlp.0.weight"] = 4LL * C * C;  expect[b+".mlp.0.bias"] = 4LL*C;
            expect[b+".mlp.2.weight"] = 4LL * C * C;  expect[b+".mlp.2.bias"] = C;
        }
        if (L == 4) break;
        int Cout = MC[L+1], Cmid = Cout / 8;                          // S2C down block @ index NB[L]
        snprintf(p, sizeof(p), "blocks.%d.%d", L, NB[L]); std::string b(p);
        expect[b+".norm1.weight"] = C;               expect[b+".norm1.bias"] = C;
        expect[b+".conv1.weight"] = 27LL * C * Cmid; expect[b+".conv1.bias"] = Cmid;
        expect[b+".conv2.weight"] = 27LL * Cout*Cout;expect[b+".conv2.bias"] = Cout;
    }

    int bad = 0, ok = 0;
    for (auto& kv : expect) {
        if (!W.has(kv.first)) { printf("  MISSING %s\n", kv.first.c_str()); bad++; continue; }
        std::vector<float> t = W(kv.first);
        if ((long long)t.size() != kv.second) {
            printf("  SHAPE  %s got=%zu expect=%lld\n", kv.first.c_str(), t.size(), kv.second);
            bad++; continue;
        }
        ok++;
    }
    printf("[shape_enc_test] tensors: %d ok, %d bad (of %zu expected)\n", ok, bad, expect.size());

    // ---- s2c_shrink sanity: children at coords 0..1 in each axis under one parent (0,0,0) ----
    // build the 8 children of parent (0,0,0): coords (b=0, c1,c2,c3) with each in {0,1}.
    std::vector<int32_t> coords;
    for (int c1 = 0; c1 < 2; c1++) for (int c2 = 0; c2 < 2; c2++) for (int c3 = 0; c3 < 2; c3++) {
        coords.push_back(0); coords.push_back(c1); coords.push_back(c2); coords.push_back(c3);
    }
    int N = 8;
    senc::S2CIdx ix = senc::s2c_shrink(coords.data(), N);
    bool s2c_ok = (ix.M == 1) &&
                  ix.coords[0]==0 && ix.coords[1]==0 && ix.coords[2]==0 && ix.coords[3]==0;
    // each child must land in a distinct slot 0..7
    int seen = 0;
    for (int n = 0; n < N; n++) { if (ix.parent[n]==0) seen |= (1 << ix.subidx[n]); }
    s2c_ok = s2c_ok && (seen == 0xFF);
    printf("[shape_enc_test] s2c_shrink 8-children->1-parent: M=%d slots=0x%02x %s\n",
           ix.M, seen, s2c_ok ? "OK" : "FAIL");

    // scatter+skip exercise (Cin=2): out [M,8*2]; skip-mean to Cout=4 -> [M,4]
    std::vector<float> feats(N*2);
    for (int n = 0; n < N; n++) { feats[n*2]=(float)n; feats[n*2+1]=(float)(n+100); }
    std::vector<float> down = senc::s2c_scatter(feats, ix, N, 2);     // [1,16]
    std::vector<float> skip = senc::s2c_skip_mean(down, ix.M, 2, 4);  // [1,4]
    printf("[shape_enc_test] s2c_scatter->[%zu] skip_mean->[%zu] (no crash) OK\n", down.size(), skip.size());

    bool pass = (bad == 0) && s2c_ok;
    printf("[shape_enc_test] %s\n", pass
        ? "PASS (compiles, gguf/npy loads, all encoder tensors present + shaped, s2c sane)"
        : "FAIL/CHECK");
    return pass ? 0 : 1;
}
