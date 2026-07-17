// p3sam_heads_gpu.hpp — GPU (cuBLAS + p3sam_heads_cuda.cu kernels) port of run_heads.
// Drop-in for the host run_heads() inner loop in p3sam_postprocess.hpp: keeps feats_N[N,512]
// and points[N,3] RESIDENT on the GPU, runs the seg/iou MLP heads per prompt via cuBLAS GEMMs,
// and returns the SAME HeadOut{seg2[K,N,3], iou[K,3]} the host produces (seg1 is unused by
// postprocess so it is left empty). Numerically matched to the validated host path:
//   nn.GELU() exact-erf; cuBLAS fp32 under NVIDIA_TF32_OVERRIDE=0; per-column max in float.
// The boolean part masks (stage2 logit > 0) and best-iou channel are identical to host up to
// fp32 GEMM rounding (near-zero logits may flip — absorbed by the NMS/part-IoU, like host).
#pragma once
#ifdef P3SAM_USE_CUDA
#include "p3sam_heads.hpp"        // p3sam::Weights, HeadOut
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

extern "C" void p3_launch_biasadd(float*, const float*, int, int, cudaStream_t);
extern "C" void p3_launch_gelu(float*, size_t, cudaStream_t);
extern "C" void p3_launch_fs_base(float*, const float*, const float*, int, cudaStream_t);
extern "C" void p3_launch_fs_prompt(float*, int, float, float, float, cudaStream_t);
extern "C" void p3_launch_scatter_col3(float*, const float*, int, int, cudaStream_t);
extern "C" void p3_launch_build_fs2(float*, const float*, const float*, int, cudaStream_t);
extern "C" void p3_launch_build_fs3(float*, const float*, const float*, int, cudaStream_t);
extern "C" void p3_launch_build_fiou(float*, const float*, const float*, const float*, int, cudaStream_t);
extern "C" void p3_launch_colmax(const float*, int, int, float*, cudaStream_t);
extern "C" void p3_launch_threshold(const float*, int, int, uint8_t*, cudaStream_t);

namespace p3gpu {

// device-resident weight cache for the heads (upload each tensor once; keep host shapes).
struct GpuW {
    p3sam::Weights host;
    cublasHandle_t handle;
    std::unordered_map<std::string, float*> dev;
    explicit GpuW(const std::string& dir) : host(dir) { cublasCreate(&handle); }
    ~GpuW() { for (auto& kv : dev) cudaFree(kv.second); cublasDestroy(handle); }
    GpuW(const GpuW&) = delete; GpuW& operator=(const GpuW&) = delete;
    float* get(const std::string& key) {
        auto it = dev.find(key);
        if (it != dev.end()) return it->second;
        const NpyArray& a = host.get(key);
        float* p; cudaMalloc(&p, a.numel()*4);
        cudaMemcpy(p, a.f32(), a.numel()*4, cudaMemcpyHostToDevice);
        dev[key] = p; return p;
    }
    int rows(const std::string& key) { return (int)host.get(key).shape[0]; }
};

static inline float* dmalloc(size_t n) { float* p; cudaMalloc(&p, n*4); return p; }

// y[N,Cout] = x[N,Cin] @ W[Cout,Cin]^T + b[Cout]   (row-major; mirrors svp_gpu linear_gpu)
static inline float* linear_gpu(GpuW& W, float* d_x, const std::string& wk, const std::string& bk,
                                int N, int Cin, int Cout) {
    float* d_W = W.get(wk);
    float* d_y = dmalloc((size_t)N*Cout);
    const float alpha=1.f, beta=0.f;
    cublasSgemm(W.handle, CUBLAS_OP_T, CUBLAS_OP_N, Cout, N, Cin, &alpha,
                d_W, Cin, d_x, Cin, &beta, d_y, Cout);
    p3_launch_biasadd(d_y, W.get(bk), N, Cout, 0);
    return d_y;
}

// 3-layer MLP (Linear,GELU,Linear,GELU,Linear) with keys <name>.{0,2,4}. returns [N,H4] device.
static inline float* gpu_mlp3(GpuW& W, const std::string& name, float* d_in, int N, int I) {
    int H0 = W.rows(name + ".0.weight");
    float* a = linear_gpu(W, d_in, name + ".0.weight", name + ".0.bias", N, I, H0);
    p3_launch_gelu(a, (size_t)N*H0, 0);
    int H2 = W.rows(name + ".2.weight");
    float* c = linear_gpu(W, a, name + ".2.weight", name + ".2.bias", N, H0, H2);
    p3_launch_gelu(c, (size_t)N*H2, 0);
    int H4 = W.rows(name + ".4.weight");
    float* e = linear_gpu(W, c, name + ".4.weight", name + ".4.bias", N, H2, H4);
    cudaFree(a); cudaFree(c);
    return e;
}

// resident context: uploads feats_N[N,512] + points[N,3] once, runs heads per prompt.
struct HeadsGpu {
    GpuW W;
    int N;
    float *d_feat=nullptr, *d_pts=nullptr, *d_fs=nullptr;
    float *d_fs2=nullptr, *d_fs3=nullptr, *d_fiou=nullptr;
    float *d_mask1=nullptr, *d_mask2=nullptr, *d_gmax=nullptr, *d_imax=nullptr;

    HeadsGpu(const std::string& dir, const float* feats_N, const float* points, int N_)
        : W(dir), N(N_) {
        d_feat = dmalloc((size_t)N*512); cudaMemcpy(d_feat, feats_N, (size_t)N*512*4, cudaMemcpyHostToDevice);
        d_pts  = dmalloc((size_t)N*3);   cudaMemcpy(d_pts,  points,  (size_t)N*3*4,   cudaMemcpyHostToDevice);
        d_fs   = dmalloc((size_t)N*518);
        d_fs2  = dmalloc((size_t)N*521);
        d_fs3  = dmalloc((size_t)N*777);
        d_fiou = dmalloc((size_t)N*777);
        d_mask1= dmalloc((size_t)N*3);
        d_mask2= dmalloc((size_t)N*3);
        d_gmax = dmalloc(256);
        d_imax = dmalloc(256);
        p3_launch_fs_base(d_fs, d_feat, d_pts, N, 0);   // fill cols 0..514 once
    }
    ~HeadsGpu() {
        for (float* p : {d_feat,d_pts,d_fs,d_fs2,d_fs3,d_fiou,d_mask1,d_mask2,d_gmax,d_imax}) cudaFree(p);
    }
    HeadsGpu(const HeadsGpu&) = delete; HeadsGpu& operator=(const HeadsGpu&) = delete;

    // run K prompts -> HeadOut{seg2[K,N,3] logits, iou[K,3] sigmoided}. seg1 left empty.
    p3sam::HeadOut run(const float* prompts, int K) {
        p3sam::HeadOut R; R.N = N; R.K = K;
        R.seg2.resize((size_t)K * N * 3);
        R.iou.resize((size_t)K * 3);
        for (int k = 0; k < K; k++) {
            float p0=prompts[k*3+0], p1=prompts[k*3+1], p2=prompts[k*3+2];
            p3_launch_fs_prompt(d_fs, N, p0, p1, p2, 0);
            // stage1: mask1[N,3]
            for (int j = 0; j < 3; j++) {
                float* m = gpu_mlp3(W, "seg_mlp_" + std::to_string(j+1), d_fs, N, 518);  // [N,1]
                p3_launch_scatter_col3(d_mask1, m, N, j, 0); cudaFree(m);
            }
            p3_launch_build_fs2(d_fs2, d_fs, d_mask1, N, 0);
            // g -> gmax[256]
            float* gv = gpu_mlp3(W, "seg_s2_mlp_g", d_fs2, N, 521);   // [N,256]
            p3_launch_colmax(gv, N, 256, d_gmax, 0); cudaFree(gv);
            p3_launch_build_fs3(d_fs3, d_gmax, d_fs2, N, 0);
            // stage2: mask2[N,3]
            for (int j = 0; j < 3; j++) {
                float* s = gpu_mlp3(W, "seg_s2_mlp_" + std::to_string(j+1), d_fs3, N, 777);  // [N,1]
                p3_launch_scatter_col3(d_mask2, s, N, j, 0); cudaFree(s);
            }
            // iou -> imax[256] -> [1,3] sigmoid
            p3_launch_build_fiou(d_fiou, d_gmax, d_fs, d_mask2, N, 0);
            float* iv = gpu_mlp3(W, "iou_mlp", d_fiou, N, 777);   // [N,256]
            p3_launch_colmax(iv, N, 256, d_imax, 0); cudaFree(iv);
            float* io = gpu_mlp3(W, "iou_mlp_out", d_imax, 1, 256);   // [1,3]
            // download stage2 logits + iou for this prompt
            cudaMemcpy(R.seg2.data() + (size_t)k*N*3, d_mask2, (size_t)N*3*4, cudaMemcpyDeviceToHost);
            float io3[3]; cudaMemcpy(io3, io, 3*4, cudaMemcpyDeviceToHost); cudaFree(io);
            for (int c = 0; c < 3; c++) R.iou[(size_t)k*3+c] = 1.0f / (1.0f + std::exp(-io3[c]));
        }
        return R;
    }
};

} // namespace p3gpu
#endif // P3SAM_USE_CUDA
