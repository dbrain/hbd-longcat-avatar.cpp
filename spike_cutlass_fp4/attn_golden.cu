// Op-golden for cuDNN fused SDPA (flash attention forward) at the flux2-klein-9b
// DiT self-attention shape: B=1, H=24, S_q=S_k=4608, D=128, NO causal mask,
// softmax scale = 1/sqrt(128), fp16 in/out, fp32 accumulation.
//
// Mirrors kernel_golden.cu style: warmup + median timing over ~50 iters with
// cudaEvents, fp32 cuBLAS reference (batched QK^T -> softmax -> *V) for cosine,
// emits JSON {kernel:"cudnn_sdpa", B,H,S,D, cosine, us, tflops}.
//
// Build (inside flux2-dev:builder-cudnn):
//   nvcc -O3 -arch=sm_120 -std=c++17 attn_golden.cu -o attn_golden \
//     -I/opt/cudnn-frontend/include -I/usr/include/x86_64-linux-gnu \
//     -lcudnn -lcublas
//
// Run pinned to the 5060 Ti via CUDA_VISIBLE_DEVICES=GPU-bd93...

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn_frontend.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>

namespace fe = cudnn_frontend;

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CBL(x) do{ cublasStatus_t s=(x); if(s){fprintf(stderr,"cuBLAS err %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4

int main(int argc, char** argv) {
    const int64_t B = 1, H = 24, S = 4608, D = 128;
    const float scale = 1.0f / sqrtf((float)D);
    int iters = 50;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);

    fprintf(stderr, "=== cuDNN SDPA golden  B=%ld H=%ld S=%ld D=%ld scale=%.6f ===\n",
            B, H, S, D, scale);

    // ---- host random inputs (fp16) ----
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    size_t qkv_n = (size_t)B * H * S * D;
    std::vector<__half> hQ(qkv_n), hK(qkv_n), hV(qkv_n);
    // scale inputs modestly so softmax of fp32 ref is well-conditioned (~unit-variance scores)
    for (size_t i = 0; i < qkv_n; i++) hQ[i] = __float2half(nd(rng) * 0.5f);
    for (size_t i = 0; i < qkv_n; i++) hK[i] = __float2half(nd(rng) * 0.5f);
    for (size_t i = 0; i < qkv_n; i++) hV[i] = __float2half(nd(rng) * 0.5f);

    __half *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, qkv_n * 2)); CK(cudaMalloc(&dK, qkv_n * 2));
    CK(cudaMalloc(&dV, qkv_n * 2)); CK(cudaMalloc(&dO, qkv_n * 2));
    CK(cudaMemcpy(dQ, hQ.data(), qkv_n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK.data(), qkv_n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV.data(), qkv_n * 2, cudaMemcpyHostToDevice));

    // ---- build cuDNN SDPA forward graph ----
    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) { fprintf(stderr, "cudnnCreate failed\n"); return 1; }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID)
                 .set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID)
                 .set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID)
                 .set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}));

    auto sdpa_opts = fe::graph::SDPA_attributes().set_name("flash_attention")
                         .set_generate_stats(false)   // inference: no LSE stats
                         .set_attn_scale(scale);
    // no causal mask, no padding mask -> full bidirectional attention

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_opts);
    O->set_output(true).set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}).set_uid(O_UID);

    if (!graph->validate().is_good())                       { fprintf(stderr, "validate failed\n"); return 1; }
    if (!graph->build_operation_graph(handle).is_good())    { fprintf(stderr, "build_operation_graph failed\n"); return 1; }
    if (!graph->create_execution_plans({fe::HeurMode_t::A}).is_good()) { fprintf(stderr, "create_execution_plans failed\n"); return 1; }
    if (!graph->check_support(handle).is_good())            { fprintf(stderr, "check_support failed (no SDPA engine for sm_120?)\n"); return 1; }
    if (!graph->build_plans(handle).is_good())              { fprintf(stderr, "build_plans failed\n"); return 1; }

    int64_t ws = 0;
    if (!graph->get_workspace_size(ws).is_good()) { fprintf(stderr, "get_workspace_size failed\n"); return 1; }
    void* dWs = nullptr; if (ws > 0) CK(cudaMalloc(&dWs, ws));
    fprintf(stderr, "  cuDNN plan built. workspace=%ld bytes\n", ws);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vpack = {
        {Q_UID, dQ}, {K_UID, dK}, {V_UID, dV}, {O_UID, dO}};

    // warmup + correctness
    if (!graph->execute(handle, vpack, dWs).is_good()) { fprintf(stderr, "execute failed\n"); return 1; }
    CK(cudaDeviceSynchronize());

    // ---- fp32 reference: per-head QK^T -> softmax -> *V via cuBLAS ----
    // We validate a subset of heads (all 24 heads, full S would be ~24*4608*4608*4 = 4GB scores;
    // validate head 0 + head H-1 fully -> two independent 4608x4608 softmaxes is plenty).
    cublasHandle_t cb; CBL(cublasCreate(&cb));
    auto half2float_vec = [](const __half* p, size_t n, std::vector<float>& out){
        out.resize(n); for (size_t i = 0; i < n; i++) out[i] = __half2float(p[i]);
    };
    // pull O back to host (fp16)
    std::vector<__half> hO(qkv_n);
    CK(cudaMemcpy(hO.data(), dO, qkv_n * 2, cudaMemcpyDeviceToHost));

    double dot = 0, na = 0, nb = 0, maxabs = 0;
    std::vector<int> check_heads = {0, (int)H/2, (int)H-1};
    // fp32 Q,K,V for the checked heads on device
    for (int h : check_heads) {
        size_t off = (size_t)h * S * D;
        std::vector<float> Qf, Kf, Vf;
        half2float_vec(hQ.data() + off, S * D, Qf);
        half2float_vec(hK.data() + off, S * D, Kf);
        half2float_vec(hV.data() + off, S * D, Vf);
        float *dQf, *dKf, *dVf, *dScores, *dRef;
        CK(cudaMalloc(&dQf, S * D * 4)); CK(cudaMalloc(&dKf, S * D * 4));
        CK(cudaMalloc(&dVf, S * D * 4)); CK(cudaMalloc(&dScores, (size_t)S * S * 4));
        CK(cudaMalloc(&dRef, S * D * 4));
        CK(cudaMemcpy(dQf, Qf.data(), S * D * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dKf, Kf.data(), S * D * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dVf, Vf.data(), S * D * 4, cudaMemcpyHostToDevice));
        float one = 1.f, zero = 0.f;
        const int Si = (int)S, Di = (int)D;
        // scores[s_q, s_k] = scale * Q[s_q,:] . K[s_k,:]
        // row-major Q (SxD), K (SxD). Want S(SxS) = Q * K^T, scaled.
        // cuBLAS column-major: compute S^T(col) = K(col-as-DxS) ... use:
        // C(SxS row) = Q(SxD) * K^T. In col-major: treat row-major MxN as col-major NxM.
        // C^T = K * Q^T : (S x S). C^T[s_k,s_q]. We'll just compute and softmax over s_k.
        // Simplest: gemm with op to get scores_rowmajor[s_q*S + s_k].
        // col-major: A=Q^T? Let's compute scoresT = K (S x D) %*% Q^T -> (S x S) col-major = scores row-major.
        CBL(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, Si, Si, Di, &scale,
                        dKf, Di, dQf, Di, &zero, dScores, Si));
        // dScores is col-major SxS: element (i,j) at i + j*S = scores[s_q=j, s_k=i].
        // softmax over s_k (i index) for each s_q (j column). Do on host for the checked head.
        std::vector<float> hScores((size_t)S * S);
        CK(cudaMemcpy(hScores.data(), dScores, (size_t)S * S * 4, cudaMemcpyDeviceToHost));
        // softmax each column j over i
        for (int64_t j = 0; j < S; j++) {
            float* col = &hScores[(size_t)j * S];
            float mx = -1e30f; for (int64_t i = 0; i < S; i++) mx = fmaxf(mx, col[i]);
            float sum = 0; for (int64_t i = 0; i < S; i++) { col[i] = expf(col[i] - mx); sum += col[i]; }
            float inv = 1.f / sum; for (int64_t i = 0; i < S; i++) col[i] *= inv;
        }
        CK(cudaMemcpy(dScores, hScores.data(), (size_t)S * S * 4, cudaMemcpyHostToDevice));
        // ref[s_q, d] = sum_s_k P[s_q,s_k] * V[s_k,d]
        // dScores col-major (i=s_k, j=s_q) == P^T... element (s_k, s_q). V is row-major SxD.
        // out_rowmajor[s_q, d] = sum_sk P[s_q,sk] V[sk,d].
        // col-major: out^T (D x S) = V^T(DxS) %*% P  where P col-major (s_k,s_q).
        // C(DxS col) = V(SxD, op_T -> DxS) * Pmat(SxS, op_N): C[d,s_q]=sum_sk V[sk,d]*P[sk,s_q]. Yes.
        // V row-major (S x D) == col-major (D x S) with lda=D. out[d,sq]=sum_sk V[d,sk]*P[sk,sq].
        CBL(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, Di, Si, Si, &one,
                        dVf, Di, dScores, Si, &zero, dRef, Di));
        // dRef col-major DxS: element (d, s_q) at d + s_q*D == ref row-major [s_q, d]. matches O layout!
        std::vector<float> hRef((size_t)S * D);
        CK(cudaMemcpy(hRef.data(), dRef, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
        // compare to hO for this head
        const __half* oh = hO.data() + off;
        for (size_t k = 0; k < (size_t)S * D; k++) {
            double r = hRef[k], g = __half2float(oh[k]);
            dot += r * g; na += r * r; nb += g * g; maxabs = fmax(maxabs, fabs(r - g));
        }
        cudaFree(dQf); cudaFree(dKf); cudaFree(dVf); cudaFree(dScores); cudaFree(dRef);
    }
    double cosine = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    cublasDestroy(cb);

    // ---- timing: median over iters ----
    std::vector<float> times(iters);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int it = 0; it < iters; it++) {
        CK(cudaEventRecord(e0));
        if (!graph->execute(handle, vpack, dWs).is_good()) { fprintf(stderr, "execute(timed) failed\n"); return 1; }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); times[it] = ms;
    }
    std::sort(times.begin(), times.end());
    double med_ms = times[iters / 2];
    double med_us = med_ms * 1e3;
    // attention FLOPs = 2 * B*H*(2*S^2*D)  (QK^T + AV)
    double flops = 2.0 * B * H * (2.0 * (double)S * S * D);
    double tflops = flops / (med_ms * 1e-3) / 1e12;

    fprintf(stderr, "  cosine=%.6f max|err|=%.5f  median %.1f us  %.1f TFLOP/s  (heads checked: 0,%ld,%ld)\n",
            cosine, maxabs, med_us, tflops, H/2, H-1);

    printf("{\"kernel\":\"cudnn_sdpa\",\"B\":%ld,\"H\":%ld,\"S\":%ld,\"D\":%ld,"
           "\"cosine\":%.6f,\"maxabs\":%.6f,\"us\":%.2f,\"tflops\":%.2f}\n",
           B, H, S, D, cosine, maxabs, med_us, tflops);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); if (dWs) cudaFree(dWs);
    cudnnDestroy(handle);
    return (cosine >= 0.99) ? 0 : 1;
}
