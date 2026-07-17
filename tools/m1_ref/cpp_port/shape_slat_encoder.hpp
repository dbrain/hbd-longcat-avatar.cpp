// shape_slat = shape_slat_encoder(voxelized_mesh) — the ENCODER half of the FlexiDualGrid
// sparse-VAE (rung 1 native texturing). Mirror of the already-validated DECODER port
// (sparse_vae_pipeline.hpp m4_decode_mesh) run in REVERSE: input_layer -> 5 stages of
// (ConvNeXt blocks + Spatial2Channel DOWN block) -> final non-affine LayerNorm -> to_latent
// (1024 -> 2*32) -> chunk(mean,logvar); shape_slat = mean (sample_posterior=False).
//
// Python source: pixal3d/models/sc_vaes/{fdg_vae.py:FlexiDualGridVaeEncoder,
//   sparse_unet_vae.py:SparseUnetVaeEncoder + SparseResBlockS2C3d + SparseConvNeXtBlock3d}
//   and modules/sparse/spatial/spatial2channel.py:SparseSpatial2Channel.
// Config (shape_enc_next_dc_f16c32_fp16.json): model_channels [64,128,256,512,1024],
//   latent_channels 32, num_blocks [0,4,8,16,4], block SparseConvNeXtBlock3d,
//   down_block SparseResBlockS2C3d.
//
// Reuses sparse_vae.hpp's VALIDATED ops verbatim (subm_conv / layernorm / silu / linear /
// build_nmap / add_inplace). The ONE net-new op vs the decoder is the Spatial2Channel
// downsample (s2c_shrink) — the exact inverse of the decoder's c2s_grow: child voxel
// (c1,c2,c3) -> parent (c1/2,c2/2,c3/2) at subidx = (c1%2)+(c2%2)*2+(c3%2)*4, scattering
// each child's feats into one of 8 slots of the parent (missing slots zero-filled).
//
// STATUS: COMPILES (cpu). NOT yet validated end-to-end — it needs a voxelized input
// (the o_voxel `mesh_to_flexible_dual_grid` CUDA solver output), which is OUT OF SCOPE /
// the remaining blocker. See SHAPE_ENC_ARCH.md for the validation plan.
//
// Weight conventions (extract_shape_enc.py, same as the decoder npy/gguf):
//   conv weight  : spike [V=27, Cin, Cout]   (torch flex_gemm [OC,3,3,3,IC] transposed)
//   Linear weight: torch [out, in]           (y = x @ W^T + b)
//   norm weight  : [C] gamma, [C] beta (affine) or none (non-affine)
//   coords       : int32 [N,4] = (batch, c1, c2, c3)
#pragma once
#include "sparse_vae.hpp"          // svae:: subm_conv / layernorm / silu / linear / build_nmap / ...
#include "gguf_reader.hpp"
#include "../../sparse_spike/npy.hpp"
#include <string>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <algorithm>

namespace senc {

// weight loader: identical to svp::WLoad — per-tensor .npy under <dir>, OR (if PIXAL3D_GGUF_DIR
// set and <gdir>/<base>.gguf present) from the packed GGUF (bit-identical f32 bytes).
struct WLoad {
    std::string dir;
    std::unique_ptr<GgufReader> g;
    explicit WLoad(const std::string& d) : dir(d) {
        if (const char* gdir = std::getenv("PIXAL3D_GGUF_DIR")) {
            std::string base = d.substr(d.find_last_of('/') + 1);
            std::string gpath = std::string(gdir) + "/" + base + ".gguf";
            FILE* f = std::fopen(gpath.c_str(), "rb");
            if (f) { std::fclose(f); g = std::make_unique<GgufReader>(gpath); }
        }
    }
    bool has(const std::string& key) const {
        if (g) return g->has(key);
        FILE* f = std::fopen((dir + "/" + key + ".npy").c_str(), "rb");
        if (f) { std::fclose(f); return true; }
        return false;
    }
    std::vector<float> operator()(const std::string& key) const {
        if (g) return g->tensor_f32(key);
        NpyArray a = npy_load(dir + "/" + key + ".npy");
        return std::vector<float>(a.f32(), a.f32() + a.numel());
    }
};

// ---- SparseSpatial2Channel(factor=2): the DOWN coord/feat merge (mirror of svae::c2s_grow) ----
// Child voxel (c1,c2,c3) -> parent (c1/2, c2/2, c3/2); subidx = (c1%2) + (c2%2)*2 + (c3%2)*4.
// Output parents are the UNIQUE parent coords. Each child scatters its Cin-vector into slot
// `subidx` of its parent's 8-slot block; empty slots are zero. Result feats are [M, 8*Cin]
// (channels grow x8), reshaped per parent. Parent ordering == torch `code.unique()` which sorts
// by the linearized parent code (z-major: c1 is most significant). We reproduce that sort so the
// emitted parent order matches torch exactly.
struct S2CIdx {
    std::vector<int32_t> coords;   // [M,4] parent coords
    std::vector<int>     parent;   // [N] child -> parent index
    std::vector<int>     subidx;   // [N] child -> slot in [0,8)
    int M = 0;
};
inline S2CIdx s2c_shrink(const int32_t* coords, int N) {
    // floor-division by 2 (handles negative coords the torch way: Python // is floor div).
    auto fdiv2 = [](int32_t v) { return v >= 0 ? (v >> 1) : -(((-v) + 1) >> 1); };
    auto fmod2 = [](int32_t v) { return ((v % 2) + 2) % 2; };

    // collect unique parents, sorted by (b, pc1, pc2, pc3) z-major == torch linearized-code order.
    struct PC { int32_t b, c1, c2, c3; };
    std::vector<int64_t> keys(N);
    std::vector<PC> pcs(N);
    for (int i = 0; i < N; i++) {
        int32_t b  = coords[i*4+0];
        int32_t p1 = fdiv2(coords[i*4+1]);
        int32_t p2 = fdiv2(coords[i*4+2]);
        int32_t p3 = fdiv2(coords[i*4+3]);
        pcs[i] = {b, p1, p2, p3};
        keys[i] = svae::coord_key(b, p1, p2, p3);
    }
    // unique sorted parent keys
    std::vector<int64_t> uniq = keys;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::unordered_map<int64_t,int> kidx;
    kidx.reserve(uniq.size() * 2);
    for (int i = 0; i < (int)uniq.size(); i++) kidx[uniq[i]] = i;

    S2CIdx r;
    r.M = (int)uniq.size();
    r.coords.resize((size_t)r.M * 4);
    r.parent.resize(N);
    r.subidx.resize(N);
    // fill parent coords from any child mapping to that key (all share the same parent coord)
    std::vector<char> filled(r.M, 0);
    for (int i = 0; i < N; i++) {
        int pi = kidx[keys[i]];
        r.parent[i] = pi;
        r.subidx[i] = fmod2(coords[i*4+1]) + fmod2(coords[i*4+2]) * 2 + fmod2(coords[i*4+3]) * 4;
        if (!filled[pi]) {
            r.coords[(size_t)pi*4+0] = pcs[i].b;
            r.coords[(size_t)pi*4+1] = pcs[i].c1;
            r.coords[(size_t)pi*4+2] = pcs[i].c2;
            r.coords[(size_t)pi*4+3] = pcs[i].c3;
            filled[pi] = 1;
        }
    }
    return r;
}

// scatter child feats [N, Cin] into parent 8-slot blocks -> [M, 8*Cin] (zero-fill missing slots).
// out[parent*8*Cin + subidx*Cin + c] = feats[n*Cin + c]. (== torch new_feats reshape [M, 8*Cin].)
inline std::vector<float> s2c_scatter(const std::vector<float>& feats, const S2CIdx& ix,
                                      int N, int Cin) {
    std::vector<float> out((size_t)ix.M * 8 * Cin, 0.f);
    for (int n = 0; n < N; n++) {
        const float* fn = feats.data() + (size_t)n * Cin;
        float* op = out.data() + ((size_t)ix.parent[n] * 8 + ix.subidx[n]) * Cin;
        for (int c = 0; c < Cin; c++) op[c] = fn[c];
    }
    return out;
}

// S2C skip_connection: x_down [M, 8*Cin] -> [M, Cout] via reshape [M, Cout, (8*Cin)/Cout].mean(-1).
// Layout of x_down is [Cout-blocks][group] contiguous? torch: feats.reshape(M, Cout, 8*Cin//Cout)
// over the flat 8*Cin channel axis (row-major) => group g of out-channel o = flat[o*K + k], K=8*Cin/Cout.
inline std::vector<float> s2c_skip_mean(const std::vector<float>& xdown, int M, int Cin, int Cout) {
    int flat = 8 * Cin;
    int K = flat / Cout;                       // channels averaged per output channel
    std::vector<float> out((size_t)M * Cout);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* xr = xdown.data() + (size_t)m * flat;
        float* orow = out.data() + (size_t)m * Cout;
        for (int o = 0; o < Cout; o++) {
            float acc = 0.f;
            const float* g = xr + (size_t)o * K;
            for (int k = 0; k < K; k++) acc += g[k];
            orow[o] = acc / (float)K;
        }
    }
    return out;
}

// ---- the encoder forward (mirror of svp::m4_decode_mesh backbone, reversed) ----
//   coords [N,4] @grid_in (e.g. the o_voxel occupancy grid), feats6 [N,6] = the FDG voxel input
//     (vertices.feats-0.5 [3] ++ intersected.feats.float()-0.5 [3]) — the concat is done by the
//     caller (FlexiDualGridVaeEncoder.forward), matching the python wrapper.
//   Returns shape_slat feats [M,32] (== mean) and writes the final coords to out_coords.
// Stages: model_channels MC=[64,128,256,512,1024], num_blocks NB=[0,4,8,16,4]. Stage i runs NB[i]
// ConvNeXt blocks at MC[i] then (if i<4) one S2C down block MC[i]->MC[i+1] (the LAST sub-block of
// the stage's ModuleList, index NB[i]). The decoder names that boundary block blocks.i.NB[i]; the
// encoder names it the SAME way (blocks.i.NB[i]) — confirmed against the safetensors keys.
inline std::vector<float> shape_slat_encode(
        const std::vector<int32_t>& coords_in, const std::vector<float>& feats6,
        const std::string& wdir, bool use_cuda, std::vector<int32_t>* out_coords = nullptr) {
    WLoad W(wdir);
    const int MC[5] = {64, 128, 256, 512, 1024};
    const int NB[5] = {0, 4, 8, 16, 4};
    const float EPS = 1e-6f;

    int N = (int)coords_in.size() / 4;
    std::vector<int32_t> coords = coords_in;
    // input_layer: SparseLinear 6 -> 64
    std::vector<float> feats = svae::linear(feats6, W("input_layer.weight"), W("input_layer.bias"), N, 6, MC[0]);
    int C = MC[0];
    std::vector<uint32_t> nmap = svae::build_nmap(coords.data(), N);

    for (int L = 0; L < 5; L++) {
        char pfx[64];
        // ConvNeXt blocks (NB[L] of them) at channels C. (SparseConvNeXtBlock3d: conv -> norm(affine)
        // -> mlp(Linear,SiLU,Linear) -> residual. IDENTICAL to the decoder's ConvNeXt path.)
        for (int j = 0; j < NB[L]; j++) {
            snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, j); std::string p(pfx);
            std::vector<float> h = svae::subm_conv(feats, nmap, W(p+".conv.weight"), W(p+".conv.bias"), N, C, C, use_cuda);
            svae::layernorm(h, W(p+".norm.weight"), W(p+".norm.bias"), N, C, EPS);
            std::vector<float> mm = svae::linear(h, W(p+".mlp.0.weight"), W(p+".mlp.0.bias"), N, C, 4*C);
            svae::silu_inplace(mm);
            mm = svae::linear(mm, W(p+".mlp.2.weight"), W(p+".mlp.2.bias"), N, 4*C, C);
            svae::add_inplace(feats, mm);
        }
        if (L == 4) break;                         // last stage: no down block
        int Cout = MC[L+1];
        snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, NB[L]); std::string p(pfx);
        // SparseResBlockS2C3d._forward:
        //   h = silu(norm1(x)); h = conv1(h)  [C -> Cout//8]
        //   h = S2C(h); x = S2C(x)            [downsample, channels x8]
        //   h = silu(norm2(h)); h = conv2(h)  [Cout -> Cout]
        //   h = h + skip(x)                    skip = reshape(x_down, Cout, 8*C/Cout).mean(-1)
        std::vector<float> h1 = feats;
        svae::layernorm(h1, W(p+".norm1.weight"), W(p+".norm1.bias"), N, C, EPS);   // affine eps1e-6
        svae::silu_inplace(h1);
        int Cmid = Cout / 8;
        std::vector<float> conv1_out = svae::subm_conv(h1, nmap, W(p+".conv1.weight"), W(p+".conv1.bias"), N, C, Cmid, use_cuda);

        S2CIdx ix = s2c_shrink(coords.data(), N);
        int M = ix.M;
        // S2C(h): [N, Cmid] -> [M, 8*Cmid] = [M, Cout]
        std::vector<float> h_down = s2c_scatter(conv1_out, ix, N, Cmid);   // [M, Cout]
        // S2C(x): [N, C] -> [M, 8*C]; skip = reshape(_, Cout, 8C/Cout).mean(-1) -> [M, Cout]
        std::vector<float> x_down = s2c_scatter(feats, ix, N, C);          // [M, 8*C]
        std::vector<float> skip = s2c_skip_mean(x_down, M, C, Cout);       // [M, Cout]

        svae::layernorm(h_down, std::vector<float>(), std::vector<float>(), M, Cout, EPS);  // norm2 non-affine
        svae::silu_inplace(h_down);
        std::vector<uint32_t> nmap2 = svae::build_nmap(ix.coords.data(), M);
        std::vector<float> conv2_out = svae::subm_conv(h_down, nmap2, W(p+".conv2.weight"), W(p+".conv2.bias"), M, Cout, Cout, use_cuda);
        svae::add_inplace(conv2_out, skip);

        feats = std::move(conv2_out); coords = std::move(ix.coords); nmap = std::move(nmap2);
        N = M; C = Cout;
    }

    // head: final F.layer_norm(non-affine, default eps 1e-5) over last dim, then to_latent 1024->64
    svae::layernorm(feats, std::vector<float>(), std::vector<float>(), N, C, 1e-5f);
    std::vector<float> h64 = svae::linear(feats, W("to_latent.weight"), W("to_latent.bias"), N, C, 64);
    // chunk(2): mean = h[:, 0:32], logvar = h[:, 32:64]; shape_slat = mean (sample_posterior=False)
    std::vector<float> mean((size_t)N * 32);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++)
        for (int c = 0; c < 32; c++) mean[(size_t)n*32 + c] = h64[(size_t)n*64 + c];

    if (out_coords) *out_coords = coords;
    return mean;   // shape_slat feats [N,32]
}

// ---- tex-decoder guide_subs from the mesh's multi-resolution occupancy ----
// The tex decoder (m6_tex_decode, pred_subdiv=false) needs, per up-level, the 8-bit child-occupancy
// of each voxel — the "subdivision" the Python decoder reads from the SparseTensor's channel2spatial
// cache (the encoder's exact idx/subidx). We rebuild it order-independently: at each decoder level we
// look up, for the decoder's CURRENT voxels (in c2s_grow's own parent-major-slot order — NOT the
// encoder's z-major order, which is why a plain coord lookup is required), which of the 8 finer-grid
// children exist in the mesh occupancy. This is exactly what `channel2spatial` reconstructs, with no
// dependence on matching the encoder's voxel ordering.
//
//   occ1024: the voxelizer's grid-1024 occupied coords [N*4] (b,x,y,z).
//   slat_coords64: the encoder's grid-64 output coords [N64*4] (the tex-DiT/decoder input).
// Returns guide_subs[4] (decoder L=0..3: 64->128,128->256,256->512,512->1024), each [N_L*8] uint8
// aligned 1:1 with the decoder's coords at that level. (CPU, order-independent, no ggml.)
inline std::vector<std::vector<uint8_t>> build_guide_subs(
        const std::vector<int32_t>& occ1024, const std::vector<int32_t>& slat_coords64) {
    auto fdiv2 = [](int32_t v) { return v >= 0 ? (v >> 1) : -(((-v) + 1) >> 1); };
    // occupancy sets at grid 128/256/512/1024 (next-finer for decoder L=0..3), keyed by coord_key.
    // occ[3]=grid1024 (from voxelizer), occ[2]=512, occ[1]=256, occ[0]=128 via repeated //2.
    int Nf = (int)occ1024.size() / 4;
    std::vector<std::unordered_set<int64_t>> occ(4);   // occ[L] = children set for decoder level L
    // build grid-1024 down to grid-128 occupancy sets
    std::vector<std::vector<int32_t>> levels(5);       // [0]=1024 [1]=512 [2]=256 [3]=128 [4]=64
    levels[0] = occ1024;
    for (int s = 1; s <= 4; s++) {
        std::unordered_set<int64_t> seen;
        const std::vector<int32_t>& prev = levels[s-1];
        std::vector<int32_t>& cur = levels[s];
        int Np = (int)prev.size() / 4;
        for (int i = 0; i < Np; i++) {
            int32_t b = prev[i*4], p1 = fdiv2(prev[i*4+1]), p2 = fdiv2(prev[i*4+2]), p3 = fdiv2(prev[i*4+3]);
            int64_t k = svae::coord_key(b, p1, p2, p3);
            if (seen.insert(k).second) { cur.push_back(b); cur.push_back(p1); cur.push_back(p2); cur.push_back(p3); }
        }
    }
    // decoder L=0 children grid128 = levels[3], L=1 grid256 = levels[2], L=2 grid512 = levels[1],
    // L=3 grid1024 = levels[0].
    const std::vector<int32_t>* child_lvl[4] = {&levels[3], &levels[2], &levels[1], &levels[0]};
    for (int L = 0; L < 4; L++) {
        occ[L].reserve(child_lvl[L]->size());
        const std::vector<int32_t>& cl = *child_lvl[L];
        for (int i = 0; i < (int)cl.size()/4; i++)
            occ[L].insert(svae::coord_key(cl[i*4], cl[i*4+1], cl[i*4+2], cl[i*4+3]));
    }
    (void)Nf;
    std::vector<std::vector<uint8_t>> subs(4);
    std::vector<int32_t> coords = slat_coords64;        // decoder current coords (grid64 -> ...)
    for (int L = 0; L < 4; L++) {
        int N = (int)coords.size() / 4;
        std::vector<uint8_t>& sub = subs[L];
        sub.assign((size_t)N * 8, 0);
        for (int n = 0; n < N; n++) {
            int32_t b = coords[n*4], c1 = coords[n*4+1], c2 = coords[n*4+2], c3 = coords[n*4+3];
            for (int k = 0; k < 8; k++) {
                int32_t x = 2*c1 + (k & 1), y = 2*c2 + ((k>>1)&1), z = 2*c3 + ((k>>2)&1);
                if (occ[L].count(svae::coord_key(b, x, y, z))) sub[(size_t)n*8 + k] = 1;
            }
        }
        // advance coords to the next level in c2s_grow's order (so subs[L+1] aligns with the decoder)
        svae::C2SIdx ix = svae::c2s_grow(coords.data(), N, sub.data());
        coords = std::move(ix.coords);
    }
    return subs;   // guide_subs[0..3], decoder order
}

}  // namespace senc
