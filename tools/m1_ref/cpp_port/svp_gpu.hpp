// GPU-resident sparse-VAE decode (Phase C #1 lever). Drop-in replacements for
// svp::m3a_upsample / m4_decode_mesh / m6_tex_decode that keep `feats` RESIDENT on the
// GPU across all 4 levels: dense linear via cuBLAS, layernorm/silu/add/gather via svae_cuda.cu
// kernels, conv via the spike launcher (resident pointers). Only the subdiv decisions and the
// final head are pulled back to host (coord growth + mesh extraction stay host code, unchanged).
//
// Profile (golden input) showed linear = 82% of the host decode (CPU OpenMP GEMM); cuBLAS
// collapses it. Weights upload to the GPU ONCE (cached), not per-call. Numerically matched to
// the validated host path (layernorm double-accum; fp32 cuBLAS under NVIDIA_TF32_OVERRIDE=0).
#pragma once
#ifdef M3A_USE_CUDA
#include "sparse_vae.hpp"          // svae:: host helpers (build_nmap, c2s_grow, mesh, Prof)
#include "sparse_vae_pipeline.hpp" // svp::WLoad
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// kernels (svae_cuda.cu)
extern "C" void launch_layernorm(float*, const float*, const float*, int, int, float, cudaStream_t);
extern "C" void launch_silu(float*, size_t, cudaStream_t);
extern "C" void launch_add(float*, const float*, size_t, cudaStream_t);
extern "C" void launch_biasadd(float*, const float*, int, int, cudaStream_t);
extern "C" void launch_gather_children(const float*, const int*, const int*, float*, int, int, cudaStream_t);
extern "C" void launch_repeat_interleave(const float*, float*, int, int, int, cudaStream_t);
extern "C" void launch_f32_to_f16(const float*, void*, size_t, cudaStream_t);
// conv (sparse_subm_conv.cu)
extern "C" void launch_subm_conv_cuda(const float*, const uint32_t*, const float*, const float*,
                                      float*, int, int, int, int, cudaStream_t);

namespace svpg {

static inline float sigmoidf(float x){ return 1.f/(1.f+std::exp(-x)); }
static inline float softplusf(float x){ return x>20.f ? x : std::log1p(std::exp(x)); }

// device-resident weight cache for one decoder model (upload each tensor once).
struct GpuW {
    svp::WLoad loader;
    cublasHandle_t handle;
    std::unordered_map<std::string, float*> cache;
    std::unordered_map<std::string, void*> cache_f16;   // f16 weight copies (scoped f16-GEMM path)
    explicit GpuW(const std::string& wdir) : loader(wdir) { cublasCreate(&handle); }
    ~GpuW() { for (auto& kv : cache) cudaFree(kv.second); for (auto& kv : cache_f16) cudaFree(kv.second); cublasDestroy(handle); }
    GpuW(const GpuW&) = delete; GpuW& operator=(const GpuW&) = delete;
    float* get(const std::string& key) {
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        std::vector<float> h = loader(key);
        float* p; cudaMalloc(&p, h.size()*4); cudaMemcpy(p, h.data(), h.size()*4, cudaMemcpyHostToDevice);
        cache[key] = p; return p;
    }
    // f16 copy of an f32 weight, converted once on device and cached (for cublasGemmEx). `n` = element count.
    void* get_f16(const std::string& key, size_t n) {
        auto it = cache_f16.find(key);
        if (it != cache_f16.end()) return it->second;
        float* d_f32 = get(key);
        void* p; cudaMalloc(&p, n*2); launch_f32_to_f16(d_f32, p, n, 0);
        cache_f16[key] = p; return p;
    }
};

static inline float* dmalloc(size_t n) { float* p; cudaMalloc(&p, n*4); return p; }
static inline uint32_t* dmalloc_u32(size_t n) { uint32_t* p; cudaMalloc(&p, n*4); return p; }
static inline int* dmalloc_i32(size_t n) { int* p; cudaMalloc(&p, n*4); return p; }

// y[N,Cout] = x[N,Cin] @ W[Cout,Cin]^T + b[Cout].  (row-major; see svp_gpu notes)
// `fast`: run the matmul as an f16-input / f32-accumulate tensor-core GEMM (cublasGemmEx). Weights and
// activations cast to f16, accumulation stays f32 (== the geo-DiT scoped-flash recipe). Bias add is
// f32. ONLY safe for CONTINUOUS output fields (the PBR tex decoder) — never for threshold/occupancy
// decodes (M4 head, SS). Off by default; set true only from m6_tex_decode.
static inline float* linear_gpu(GpuW& W, float* d_x, const std::string& wkey, const std::string& bkey,
                                int N, int Cin, int Cout, bool fast=false) {
    float* d_y = dmalloc((size_t)N*Cout);
    const float alpha=1.f, beta=0.f;
    if (fast) {
        void* dW16 = W.get_f16(wkey, (size_t)Cin*Cout);
        void* dx16; cudaMalloc(&dx16, (size_t)N*Cin*2); launch_f32_to_f16(d_x, dx16, (size_t)N*Cin, 0);
        cublasGemmEx(W.handle, CUBLAS_OP_T, CUBLAS_OP_N, Cout, N, Cin, &alpha,
                     dW16, CUDA_R_16F, Cin, dx16, CUDA_R_16F, Cin, &beta,
                     d_y, CUDA_R_32F, Cout, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaFree(dx16);
    } else {
        cublasSgemm(W.handle, CUBLAS_OP_T, CUBLAS_OP_N, Cout, N, Cin, &alpha,
                    W.get(wkey), Cin, d_x, Cin, &beta, d_y, Cout);
    }
    if (!bkey.empty()) launch_biasadd(d_y, W.get(bkey), N, Cout, 0);
    return d_y;
}

static inline float* conv_gpu(GpuW& W, float* d_feats, const uint32_t* d_nmap,
                              const std::string& wkey, const std::string& bkey,
                              int N, int Cin, int Cout) {
    float* d_out = dmalloc((size_t)N*Cout);
    launch_subm_conv_cuda(d_feats, d_nmap, W.get(wkey), W.get(bkey), d_out, N, Cin, Cout, 27, 0);
    return d_out;
}

// One ConvNeXt + C2S level. d_feats resident [N,C]; returns new d_feats [M,Cout]; updates
// coords/nmap/N/C in place. (Shared by m3a/m4/m6: shape & tex differ only in subdiv source.)
struct LevelIO {
    float* d_feats; int N; int C;
    std::vector<int32_t> coords;   // host coords [N,4]
    uint32_t* d_nmap;              // device nmap [N,27]
};

// run NB ConvNeXt blocks at (N,C) on resident feats. `fast` = f16 MLP GEMMs (continuous fields only).
static inline void run_convnext_blocks(GpuW& W, LevelIO& io, int L, int NB, float EPS, bool fast=false) {
    char pfx[64];
    for (int j=0;j<NB;j++) {
        snprintf(pfx,sizeof(pfx),"blocks.%d.%d",L,j); std::string p(pfx);
        float* h = conv_gpu(W, io.d_feats, io.d_nmap, p+".conv.weight", p+".conv.bias", io.N, io.C, io.C);
        launch_layernorm(h, W.get(p+".norm.weight"), W.get(p+".norm.bias"), io.N, io.C, EPS, 0);
        float* mm = linear_gpu(W, h, p+".mlp.0.weight", p+".mlp.0.bias", io.N, io.C, 4*io.C, fast);
        launch_silu(mm, (size_t)io.N*4*io.C, 0);
        float* mm2 = linear_gpu(W, mm, p+".mlp.2.weight", p+".mlp.2.bias", io.N, 4*io.C, io.C, fast);
        launch_add(io.d_feats, mm2, (size_t)io.N*io.C, 0);
        cudaFree(h); cudaFree(mm); cudaFree(mm2);
    }
}

// the C2S up-block. `sub` is the per-voxel subdiv bool [N*8] (shape: predicted here; tex: guide).
// `p` is the up-block tensor prefix ("blocks.L.NB[L]").
// Returns the grown feats; updates io to (M, Cout) with new coords + nmap. Frees old nmap.
static inline void run_c2s_upblock(GpuW& W, LevelIO& io, const std::string& p, int Cout, float EPS,
                                   const std::vector<uint8_t>& sub) {
    // ---- norm1 + silu on a COPY of feats ----
    int N=io.N, C=io.C;
    float* h1 = dmalloc((size_t)N*C);
    cudaMemcpy(h1, io.d_feats, (size_t)N*C*4, cudaMemcpyDeviceToDevice);
    launch_layernorm(h1, W.get(p+".norm1.weight"), W.get(p+".norm1.bias"), N, C, EPS, 0);
    launch_silu(h1, (size_t)N*C, 0);
    float* conv1_out = conv_gpu(W, h1, io.d_nmap, p+".conv1.weight", p+".conv1.bias", N, C, Cout*8);
    cudaFree(h1);
    // ---- coord growth (host) ----
    svae::C2SIdx ix = svae::c2s_grow(io.coords.data(), N, sub.data());
    int M = ix.M;
    int* d_parent = dmalloc_i32(M); int* d_child = dmalloc_i32(M);
    cudaMemcpy(d_parent, ix.parent.data(), (size_t)M*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_child,  ix.child.data(),  (size_t)M*4, cudaMemcpyHostToDevice);
    // ---- gather children + skip ----
    float* h_grown = dmalloc((size_t)M*Cout);
    launch_gather_children(conv1_out, d_parent, d_child, h_grown, M, Cout, 0);
    cudaFree(conv1_out);
    int per = C/8;
    float* skip_small = dmalloc((size_t)M*per);
    launch_gather_children(io.d_feats, d_parent, d_child, skip_small, M, per, 0);
    float* skip = dmalloc((size_t)M*Cout);
    launch_repeat_interleave(skip_small, skip, M, per, Cout/per, 0);
    cudaFree(skip_small); cudaFree(d_parent); cudaFree(d_child);
    // ---- layernorm(non-affine) + silu + conv2 + skip ----
    launch_layernorm(h_grown, nullptr, nullptr, M, Cout, EPS, 0);
    launch_silu(h_grown, (size_t)M*Cout, 0);
    uint32_t* d_nmap2 = dmalloc_u32((size_t)M*27);
    { std::vector<uint32_t> nm2 = svae::build_nmap(ix.coords.data(), M);
      cudaMemcpy(d_nmap2, nm2.data(), (size_t)M*27*4, cudaMemcpyHostToDevice); }
    float* conv2_out = conv_gpu(W, h_grown, d_nmap2, p+".conv2.weight", p+".conv2.bias", M, Cout, Cout);
    cudaFree(h_grown);
    launch_add(conv2_out, skip, (size_t)M*Cout, 0);
    cudaFree(skip);
    // ---- commit ----
    cudaFree(io.d_feats); cudaFree(io.d_nmap);
    io.d_feats = conv2_out; io.coords = std::move(ix.coords); io.d_nmap = d_nmap2; io.N = M; io.C = Cout;
}

// pull subdiv decisions: subdiv = linear(feats, to_subdiv) -> host sub bool [N*8].
// `p` is the up-block tensor prefix ("blocks.L.NB[L]").
static inline std::vector<uint8_t> compute_subdiv(GpuW& W, float* d_feats, int N, int C, const std::string& p) {
    float* d_sub = linear_gpu(W, d_feats, p+".to_subdiv.weight", p+".to_subdiv.bias", N, C, 8);
    std::vector<float> sub_logits((size_t)N*8);
    cudaMemcpy(sub_logits.data(), d_sub, (size_t)N*8*4, cudaMemcpyDeviceToHost);
    cudaFree(d_sub);
    std::vector<uint8_t> sub((size_t)N*8);
    for (size_t i=0;i<sub.size();i++) sub[i] = (sub_logits[i] > 0.f) ? 1 : 0;
    return sub;
}

// ===================== M4: decoder.forward + FDG head + mesh =====================
inline svae::Mesh m4_decode_mesh(const std::vector<int32_t>& coords_in,
        const std::vector<float>& shape_slat, const std::string& wdir,
        std::vector<std::vector<uint8_t>>* out_subs = nullptr, bool close_surface = false,
        std::vector<int32_t>* out_coords1024 = nullptr, int resolution = 1024) {
    GpuW W(wdir);
    const int MC[5]={1024,512,256,128,64}; const int NB[4]={4,16,8,4}; const float EPS=1e-6f;
    int N=(int)coords_in.size()/4;
    LevelIO io; io.coords = coords_in; io.N=N; io.C=1024;
    // from_latent
    float* d_slat = dmalloc((size_t)N*32);
    cudaMemcpy(d_slat, shape_slat.data(), (size_t)N*32*4, cudaMemcpyHostToDevice);
    io.d_feats = linear_gpu(W, d_slat, "from_latent.weight", "from_latent.bias", N, 32, 1024);
    cudaFree(d_slat);
    { std::vector<uint32_t> nm = svae::build_nmap(io.coords.data(), N);
      io.d_nmap = dmalloc_u32((size_t)N*27);
      cudaMemcpy(io.d_nmap, nm.data(), (size_t)N*27*4, cudaMemcpyHostToDevice); }
    if (out_subs) out_subs->clear();
    for (int L=0;L<4;L++) {
        int Cout=MC[L+1];
        std::string up = std::string("blocks.")+std::to_string(L)+"."+std::to_string(NB[L]);
        run_convnext_blocks(W, io, L, NB[L], EPS);
        std::vector<uint8_t> sub = compute_subdiv(W, io.d_feats, io.N, io.C, up);
        if (out_subs) out_subs->push_back(sub);
        run_c2s_upblock(W, io, up, Cout, EPS, sub);
    }
    // head: final layernorm(non-affine, 1e-5) + output_layer 64->7
    launch_layernorm(io.d_feats, nullptr, nullptr, io.N, 64, 1e-5f, 0);
    float* d_h7 = linear_gpu(W, io.d_feats, "output_layer.weight", "output_layer.bias", io.N, 64, 7);
    std::vector<float> h7((size_t)io.N*7);
    cudaMemcpy(h7.data(), d_h7, (size_t)io.N*7*4, cudaMemcpyDeviceToHost);
    cudaFree(d_h7); cudaFree(io.d_feats); cudaFree(io.d_nmap);
    int Nf=io.N;
    std::vector<float> dual((size_t)Nf*3), qlerp(Nf); std::vector<int8_t> inter((size_t)Nf*3);
    for (int i=0;i<Nf;i++) {
        for (int d=0;d<3;d++) dual[(size_t)i*3+d]=2.f*sigmoidf(h7[(size_t)i*7+d])-0.5f;
        for (int d=0;d<3;d++) inter[(size_t)i*3+d]=(h7[(size_t)i*7+3+d]>0.f)?1:0;
        qlerp[i]=softplusf(h7[(size_t)i*7+6]);
    }
    // PIXAL3D_DEBUG_DECODE: diagnose the 1536 "0-face" bug — is `intersected` all-zero because the
    // logits are NaN (overflow/garbage), all-negative (degenerate/OOD decode), or fine (meshing bug)?
    if (std::getenv("PIXAL3D_DEBUG_DECODE")) {
        long nz=0, nan=0; float mn=1e30f, mx=-1e30f;
        for (int i=0;i<Nf;i++) for (int d=0;d<3;d++){ float v=h7[(size_t)i*7+3+d];
            if (std::isnan(v)) nan++; if (v>0.f) nz++; if(v<mn)mn=v; if(v>mx)mx=v; }
        fprintf(stderr,"[decode-dbg] Nf=%d intersected_pos=%ld nan=%ld intersected_logit[min=%.4f max=%.4f]\n",
                Nf, nz, nan, mn, mx);
    }
    if (out_coords1024) *out_coords1024 = io.coords;   // A1: grid-`resolution` occupancy for the marching-tet remesh
    double t_msh = svae::now_s();
    svae::Mesh out_mesh = svae::flexible_dual_grid_to_mesh(io.coords.data(), Nf, dual.data(), inter.data(), qlerp.data(), resolution, close_surface);
    if (std::getenv("PIXAL3D_TIME_MESHER")) fprintf(stderr, "[7m] FDG mesher (N=%d -> %d v / %d f): %.1fs\n", Nf, out_mesh.N, out_mesh.F, svae::now_s()-t_msh);
    return out_mesh;
}

// ===================== M3a: upsample -> hr_coords =====================
inline std::vector<int32_t> m3a_upsample(const std::vector<int32_t>& coords_in,
        const std::vector<float>& lr_denorm, const std::string& wdir) {
    GpuW W(wdir);
    const int MC[5]={1024,512,256,128,64}; const int NB[4]={4,16,8,4}; const float EPS=1e-6f;
    int N=(int)coords_in.size()/4;
    LevelIO io; io.coords=coords_in; io.N=N; io.C=1024;
    float* d_lr = dmalloc((size_t)N*32);
    cudaMemcpy(d_lr, lr_denorm.data(), (size_t)N*32*4, cudaMemcpyHostToDevice);
    io.d_feats = linear_gpu(W, d_lr, "from_latent.weight", "from_latent.bias", N, 32, 1024);
    cudaFree(d_lr);
    { std::vector<uint32_t> nm = svae::build_nmap(io.coords.data(), N);
      io.d_nmap = dmalloc_u32((size_t)N*27);
      cudaMemcpy(io.d_nmap, nm.data(), (size_t)N*27*4, cudaMemcpyHostToDevice); }
    for (int L=0;L<4;L++) {
        int Cout=MC[L+1];
        std::string up = std::string("blocks.")+std::to_string(L)+"."+std::to_string(NB[L]);
        run_convnext_blocks(W, io, L, NB[L], EPS);
        std::vector<uint8_t> sub = compute_subdiv(W, io.d_feats, io.N, io.C, up);
        run_c2s_upblock(W, io, up, Cout, EPS, sub);
    }
    cudaFree(io.d_feats); cudaFree(io.d_nmap);
    return io.coords;
}

// ===================== M6: tex decoder (out 6, guide_subs) =====================
inline std::vector<float> m6_tex_decode(const std::vector<int32_t>& coords_in,
        const std::vector<float>& tex_slat, const std::vector<std::vector<uint8_t>>& guide_subs,
        const std::string& wdir, std::vector<int32_t>* out_coords = nullptr) {
    GpuW W(wdir);
    // Scoped f16-input/f32-accumulate GEMMs for the ConvNeXt MLPs + head. PBR is a CONTINUOUS field
    // (not a threshold like M4-occupancy/SS), so f16 rounding shifts colour sub-quantisation, not
    // topology. Opt-in via PIXAL3D_TEX_F16=1 pending an owner textured A/B. The subm convs stay f32.
    const bool f16 = std::getenv("PIXAL3D_TEX_F16") != nullptr;
    const int MC[5]={1024,512,256,128,64}; const int NB[4]={4,16,8,4}; const float EPS=1e-6f;
    int N=(int)coords_in.size()/4;
    LevelIO io; io.coords=coords_in; io.N=N; io.C=1024;
    float* d_slat = dmalloc((size_t)N*32);
    cudaMemcpy(d_slat, tex_slat.data(), (size_t)N*32*4, cudaMemcpyHostToDevice);
    io.d_feats = linear_gpu(W, d_slat, "from_latent.weight", "from_latent.bias", N, 32, 1024, f16);
    cudaFree(d_slat);
    { std::vector<uint32_t> nm = svae::build_nmap(io.coords.data(), N);
      io.d_nmap = dmalloc_u32((size_t)N*27);
      cudaMemcpy(io.d_nmap, nm.data(), (size_t)N*27*4, cudaMemcpyHostToDevice); }
    for (int L=0;L<4;L++) {
        int Cout=MC[L+1];
        std::string up = std::string("blocks.")+std::to_string(L)+"."+std::to_string(NB[L]);
        run_convnext_blocks(W, io, L, NB[L], EPS, f16);
        run_c2s_upblock(W, io, up, Cout, EPS, guide_subs[L]);   // pred_subdiv=false: SHAPE subs
    }
    launch_layernorm(io.d_feats, nullptr, nullptr, io.N, 64, 1e-5f, 0);
    float* d_out6 = linear_gpu(W, io.d_feats, "output_layer.weight", "output_layer.bias", io.N, 64, 6, f16);
    std::vector<float> out6((size_t)io.N*6);
    cudaMemcpy(out6.data(), d_out6, (size_t)io.N*6*4, cudaMemcpyDeviceToHost);
    cudaFree(d_out6); cudaFree(io.d_feats); cudaFree(io.d_nmap);
    for (auto& v : out6) v = v*0.5f+0.5f;
    if (out_coords) *out_coords = io.coords;
    return out6;
}

} // namespace svpg
#endif // M3A_USE_CUDA
