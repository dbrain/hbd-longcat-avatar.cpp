// Callable wrappers around the VALIDATED M3a/M4 host ports (sparse_vae.hpp), for the geometry
// E2E driver. The bodies are lifted verbatim from m3a_upsample.cpp / m4_mesh.cpp (their main()s
// validated bit-exact CPU+CUDA); here they take in-memory inputs and return outputs, no goldens.
#pragma once
#include "sparse_vae.hpp"
#include "gguf_reader.hpp"
#include "../../sparse_spike/npy.hpp"
#include <string>
#include <memory>
#include <cstdlib>
#include <cmath>

namespace svp {

// weight loader: per-tensor .npy under a model dir, OR (if PIXAL3D_GGUF_DIR set and
// <gdir>/<model>.gguf present) from one GGUF — bit-identical (the stored bytes are the npy
// bytes; the sparse-VAE ops treat each weight as a flat fp32 array, so ne layout is irrelevant).
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
    std::vector<float> operator()(const std::string& key) const {
        if (g) return g->tensor_f32(key);
        NpyArray a = npy_load(dir + "/" + key + ".npy");
        return std::vector<float>(a.f32(), a.f32() + a.numel());
    }
};

// M3a: FlexiDualGridVaeDecoder.upsample(lr_slat, x4) -> hr_coords[Nh,4].  (== m3a_upsample.cpp)
//   coords [N,4] @grid32, lr_denorm [N,32] (denormalized lr_slat). Returns hr_coords [Nh,4].
inline std::vector<int32_t> m3a_upsample(const std::vector<int32_t>& coords_in,
        const std::vector<float>& lr_denorm, const std::string& wdir, bool use_cuda) {
    WLoad W(wdir);
    const int MC[5] = {1024, 512, 256, 128, 64};
    const int NB[4] = {4, 16, 8, 4};
    const float EPS = 1e-6f;
    int N = (int)coords_in.size() / 4;
    std::vector<int32_t> coords = coords_in;
    std::vector<float> feats = svae::linear(lr_denorm, W("from_latent.weight"), W("from_latent.bias"), N, 32, 1024);
    int C = 1024;
    std::vector<uint32_t> nmap = svae::build_nmap(coords.data(), N);
    for (int L = 0; L < 4; L++) {
        int Cout = MC[L+1];
        char pfx[64];
        for (int j = 0; j < NB[L]; j++) {
            snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, j); std::string p(pfx);
            std::vector<float> h = svae::subm_conv(feats, nmap, W(p+".conv.weight"), W(p+".conv.bias"), N, C, C, use_cuda);
            svae::layernorm(h, W(p+".norm.weight"), W(p+".norm.bias"), N, C, EPS);
            std::vector<float> m = svae::linear(h, W(p+".mlp.0.weight"), W(p+".mlp.0.bias"), N, C, 4*C);
            svae::silu_inplace(m);
            m = svae::linear(m, W(p+".mlp.2.weight"), W(p+".mlp.2.bias"), N, 4*C, C);
            svae::add_inplace(feats, m);
        }
        snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, NB[L]); std::string p(pfx);
        std::vector<float> subdiv = svae::linear(feats, W(p+".to_subdiv.weight"), W(p+".to_subdiv.bias"), N, C, 8);
        std::vector<uint8_t> sub((size_t)N*8);
        for (size_t i = 0; i < sub.size(); i++) sub[i] = (subdiv[i] > 0.f) ? 1 : 0;
        std::vector<float> h1 = feats;
        svae::layernorm(h1, W(p+".norm1.weight"), W(p+".norm1.bias"), N, C, EPS);
        svae::silu_inplace(h1);
        std::vector<float> conv1_out = svae::subm_conv(h1, nmap, W(p+".conv1.weight"), W(p+".conv1.bias"), N, C, Cout*8, use_cuda);
        svae::C2SIdx ix = svae::c2s_grow(coords.data(), N, sub.data());
        int M = ix.M;
        std::vector<float> h_grown = svae::gather_children(conv1_out, ix, Cout);
        int per = C / 8;
        std::vector<float> skip_small = svae::gather_children(feats, ix, per);
        std::vector<float> skip = svae::repeat_interleave(skip_small, M, per, Cout/per);
        std::vector<float> empty;
        svae::layernorm(h_grown, empty, empty, M, Cout, EPS);
        svae::silu_inplace(h_grown);
        std::vector<uint32_t> nmap2 = svae::build_nmap(ix.coords.data(), M);
        std::vector<float> conv2_out = svae::subm_conv(h_grown, nmap2, W(p+".conv2.weight"), W(p+".conv2.bias"), M, Cout, Cout, use_cuda);
        svae::add_inplace(conv2_out, skip);
        feats = std::move(conv2_out); coords = std::move(ix.coords); nmap = std::move(nmap2); N = M; C = Cout;
    }
    return coords;  // hr_coords [N,4]
}

static inline float sigmoidf(float x){ return 1.f/(1.f+std::exp(-x)); }
static inline float softplusf(float x){ return x>20.f ? x : std::log1p(std::exp(x)); }

// M4: decoder.forward (grid64->grid1024) + FDG head + flexible_dual_grid_to_mesh.  (== m4_mesh.cpp)
//   coords [M,4] @grid64, shape_slat [M,32] (denorm). Returns the triangle mesh.
//   out_subs (optional): the per-level subdiv decisions (4 × [N_L*8] bool) for the tex decoder's
//   guide_subs (pred_subdiv=false reuses the SHAPE decoder's subs).
inline svae::Mesh m4_decode_mesh(const std::vector<int32_t>& coords_in,
        const std::vector<float>& shape_slat, const std::string& wdir, bool use_cuda,
        std::vector<std::vector<uint8_t>>* out_subs = nullptr) {
    WLoad W(wdir);
    const int MC[5] = {1024,512,256,128,64}; const int NB[4] = {4,16,8,4}; const float EPS = 1e-6f;
    int N = (int)coords_in.size() / 4;
    std::vector<int32_t> coords = coords_in;
    std::vector<float> feats = svae::linear(shape_slat, W("from_latent.weight"), W("from_latent.bias"), N, 32, 1024);
    int C = 1024;
    std::vector<uint32_t> nmap = svae::build_nmap(coords.data(), N);
    if (out_subs) out_subs->clear();
    for (int L = 0; L < 4; L++) {
        int Cout = MC[L+1]; char pfx[64];
        for (int j = 0; j < NB[L]; j++) {
            snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, j); std::string p(pfx);
            std::vector<float> h = svae::subm_conv(feats, nmap, W(p+".conv.weight"), W(p+".conv.bias"), N, C, C, use_cuda);
            svae::layernorm(h, W(p+".norm.weight"), W(p+".norm.bias"), N, C, EPS);
            std::vector<float> mm = svae::linear(h, W(p+".mlp.0.weight"), W(p+".mlp.0.bias"), N, C, 4*C);
            svae::silu_inplace(mm);
            mm = svae::linear(mm, W(p+".mlp.2.weight"), W(p+".mlp.2.bias"), N, 4*C, C);
            svae::add_inplace(feats, mm);
        }
        snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, NB[L]); std::string p(pfx);
        std::vector<float> subdiv = svae::linear(feats, W(p+".to_subdiv.weight"), W(p+".to_subdiv.bias"), N, C, 8);
        std::vector<uint8_t> sub((size_t)N*8); for (size_t i=0;i<sub.size();i++) sub[i]=(subdiv[i]>0.f)?1:0;
        if (out_subs) out_subs->push_back(sub);
        std::vector<float> h1 = feats;
        svae::layernorm(h1, W(p+".norm1.weight"), W(p+".norm1.bias"), N, C, EPS);
        svae::silu_inplace(h1);
        std::vector<float> conv1_out = svae::subm_conv(h1, nmap, W(p+".conv1.weight"), W(p+".conv1.bias"), N, C, Cout*8, use_cuda);
        svae::C2SIdx ix = svae::c2s_grow(coords.data(), N, sub.data());
        int M = ix.M;
        std::vector<float> h_grown = svae::gather_children(conv1_out, ix, Cout);
        int per = C/8; std::vector<float> skip_small = svae::gather_children(feats, ix, per);
        std::vector<float> skip = svae::repeat_interleave(skip_small, M, per, Cout/per);
        std::vector<float> empty;
        svae::layernorm(h_grown, empty, empty, M, Cout, EPS);
        svae::silu_inplace(h_grown);
        std::vector<uint32_t> nmap2 = svae::build_nmap(ix.coords.data(), M);
        std::vector<float> conv2_out = svae::subm_conv(h_grown, nmap2, W(p+".conv2.weight"), W(p+".conv2.bias"), M, Cout, Cout, use_cuda);
        svae::add_inplace(conv2_out, skip);
        feats=std::move(conv2_out); coords=std::move(ix.coords); nmap=std::move(nmap2); N=M; C=Cout;
    }
    // head: final F.layer_norm (non-affine, eps 1e-5) + output_layer 64->7 + FDG split
    std::vector<float> empty;
    svae::layernorm(feats, empty, empty, N, 64, 1e-5f);
    std::vector<float> h7 = svae::linear(feats, W("output_layer.weight"), W("output_layer.bias"), N, 64, 7);
    std::vector<float> dual((size_t)N*3), qlerp(N); std::vector<int8_t> inter((size_t)N*3);
    for (int i=0;i<N;i++) {
        for (int d=0;d<3;d++) dual[(size_t)i*3+d] = 2.f*sigmoidf(h7[(size_t)i*7+d]) - 0.5f;
        for (int d=0;d<3;d++) inter[(size_t)i*3+d] = (h7[(size_t)i*7+3+d] > 0.f) ? 1 : 0;
        qlerp[i] = softplusf(h7[(size_t)i*7+6]);
    }
    return svae::flexible_dual_grid_to_mesh(coords.data(), N, dual.data(), inter.data(), qlerp.data(), 1024);
}

// M6: tex decoder (SparseUnetVaeDecoder, out 6, pred_subdiv=false). Reuses the m4 backbone but
// the C2S up-blocks use guide_subs (the SHAPE decoder's per-level subdiv decisions) instead of
// to_subdiv, and the head is output_layer 64->6 (PBR) ·0.5+0.5 (no FDG/mesh).
//   coords [M,4] @grid64, tex_slat [M,32] (denorm), guide_subs[4] each [N_L*8] bool.
//   Returns PBR voxels [N,6] (base_color[0:3]/metallic[3]/roughness[4]/alpha[5]) ·0.5+0.5.
inline std::vector<float> m6_tex_decode(const std::vector<int32_t>& coords_in,
        const std::vector<float>& tex_slat, const std::vector<std::vector<uint8_t>>& guide_subs,
        const std::string& wdir, bool use_cuda, std::vector<int32_t>* out_coords = nullptr) {
    WLoad W(wdir);
    const int MC[5] = {1024,512,256,128,64}; const int NB[4] = {4,16,8,4}; const float EPS = 1e-6f;
    int N = (int)coords_in.size() / 4;
    std::vector<int32_t> coords = coords_in;
    std::vector<float> feats = svae::linear(tex_slat, W("from_latent.weight"), W("from_latent.bias"), N, 32, 1024);
    int C = 1024;
    std::vector<uint32_t> nmap = svae::build_nmap(coords.data(), N);
    for (int L = 0; L < 4; L++) {
        int Cout = MC[L+1]; char pfx[64];
        for (int j = 0; j < NB[L]; j++) {
            snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, j); std::string p(pfx);
            std::vector<float> h = svae::subm_conv(feats, nmap, W(p+".conv.weight"), W(p+".conv.bias"), N, C, C, use_cuda);
            svae::layernorm(h, W(p+".norm.weight"), W(p+".norm.bias"), N, C, EPS);
            std::vector<float> mm = svae::linear(h, W(p+".mlp.0.weight"), W(p+".mlp.0.bias"), N, C, 4*C);
            svae::silu_inplace(mm);
            mm = svae::linear(mm, W(p+".mlp.2.weight"), W(p+".mlp.2.bias"), N, 4*C, C);
            svae::add_inplace(feats, mm);
        }
        snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, NB[L]); std::string p(pfx);
        const std::vector<uint8_t>& sub = guide_subs[L];   // pred_subdiv=false: SHAPE decoder's subs
        std::vector<float> h1 = feats;
        svae::layernorm(h1, W(p+".norm1.weight"), W(p+".norm1.bias"), N, C, EPS);
        svae::silu_inplace(h1);
        std::vector<float> conv1_out = svae::subm_conv(h1, nmap, W(p+".conv1.weight"), W(p+".conv1.bias"), N, C, Cout*8, use_cuda);
        svae::C2SIdx ix = svae::c2s_grow(coords.data(), N, sub.data());
        int M = ix.M;
        std::vector<float> h_grown = svae::gather_children(conv1_out, ix, Cout);
        int per = C/8; std::vector<float> skip_small = svae::gather_children(feats, ix, per);
        std::vector<float> skip = svae::repeat_interleave(skip_small, M, per, Cout/per);
        std::vector<float> empty;
        svae::layernorm(h_grown, empty, empty, M, Cout, EPS);
        svae::silu_inplace(h_grown);
        std::vector<uint32_t> nmap2 = svae::build_nmap(ix.coords.data(), M);
        std::vector<float> conv2_out = svae::subm_conv(h_grown, nmap2, W(p+".conv2.weight"), W(p+".conv2.bias"), M, Cout, Cout, use_cuda);
        svae::add_inplace(conv2_out, skip);
        feats=std::move(conv2_out); coords=std::move(ix.coords); nmap=std::move(nmap2); N=M; C=Cout;
    }
    std::vector<float> empty;
    svae::layernorm(feats, empty, empty, N, 64, 1e-5f);   // final F.layer_norm (non-affine)
    std::vector<float> out6 = svae::linear(feats, W("output_layer.weight"), W("output_layer.bias"), N, 64, 6);
    for (auto& v : out6) v = v * 0.5f + 0.5f;             // decode_tex_slat: ·0.5+0.5
    if (out_coords) *out_coords = coords;
    return out6;  // [N,6] PBR
}

}  // namespace svp
