// Op-golden for cuDNN fused SDPA at the LTX-2.3 22B DiT SELF-attention shapes.
// Parameterized clone of attn_golden.cu (flux2). NO causal mask, gqa=1, fp16 IO,
// fp32 accum, scale=1/sqrt(D).
//
// LTX-2.3 video DiT self-attn (attn1): num_heads=30, head_dim=128.
//   Video tokens L = latent_T * latent_H * latent_W. For 704x1280 x 97 frames,
//   VAE scale {8,32,32}, causal temporal: latent_T=(97-1)/8+1=13, latent_H=704/32=22,
//   latent_W=1280/32=40 -> L = 13*22*40 = 11440  (>> flux2's 4608).
// LTX-2.3 audio DiT self-attn (audio_attn1): num_heads=32, head_dim=64; short seq.
//
// Run: --H 30 --S 11440 --D 128   (video)   |   --H 32 --S ... --D 64  (audio)
//
// Build (inside flux2-dev:builder-cudnn):
//   nvcc -O3 -arch=sm_120 -std=c++17 attn_ltx_golden.cu -o attn_ltx_golden \
//     -I/opt/cudnn-frontend/include -lcudnn -lcublas

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn_frontend.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    int64_t B = 1, H = 30, S = 11440, D = 128;
    int iters = 50;
    const char* tag = "ltx_video_selfattn";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        if (!strcmp(argv[i], "--H") && i + 1 < argc) H = atoll(argv[++i]);
        if (!strcmp(argv[i], "--S") && i + 1 < argc) S = atoll(argv[++i]);
        if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoll(argv[++i]);
        if (!strcmp(argv[i], "--tag") && i + 1 < argc) tag = argv[++i];
    }
    const float scale = 1.0f / sqrtf((float)D);
    fprintf(stderr, "=== cuDNN SDPA LTX  %s  B=%ld H=%ld S=%ld D=%ld scale=%.6f ===\n", tag, B, H, S, D, scale);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    size_t qkv_n = (size_t)B * H * S * D;
    std::vector<__half> hQ(qkv_n), hK(qkv_n), hV(qkv_n);
    for (size_t i = 0; i < qkv_n; i++) hQ[i] = __float2half(nd(rng) * 0.5f);
    for (size_t i = 0; i < qkv_n; i++) hK[i] = __float2half(nd(rng) * 0.5f);
    for (size_t i = 0; i < qkv_n; i++) hV[i] = __float2half(nd(rng) * 0.5f);

    __half *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, qkv_n * 2)); CK(cudaMalloc(&dK, qkv_n * 2));
    CK(cudaMalloc(&dV, qkv_n * 2)); CK(cudaMalloc(&dO, qkv_n * 2));
    CK(cudaMemcpy(dQ, hQ.data(), qkv_n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hK.data(), qkv_n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV.data(), qkv_n * 2, cudaMemcpyHostToDevice));

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
                         .set_generate_stats(false).set_attn_scale(scale);
    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_opts);
    O->set_output(true).set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}).set_uid(O_UID);

    if (!graph->validate().is_good())                       { fprintf(stderr, "validate failed\n"); return 1; }
    if (!graph->build_operation_graph(handle).is_good())    { fprintf(stderr, "build_operation_graph failed\n"); return 1; }
    if (!graph->create_execution_plans({fe::HeurMode_t::A}).is_good()) { fprintf(stderr, "create_execution_plans failed\n"); return 1; }
    if (!graph->check_support(handle).is_good())            { fprintf(stderr, "check_support failed (no SDPA engine for D=%ld on sm_120?)\n", D); return 1; }
    if (!graph->build_plans(handle).is_good())              { fprintf(stderr, "build_plans failed\n"); return 1; }

    int64_t ws = 0;
    if (!graph->get_workspace_size(ws).is_good()) { fprintf(stderr, "get_workspace_size failed\n"); return 1; }
    void* dWs = nullptr; if (ws > 0) CK(cudaMalloc(&dWs, ws));
    fprintf(stderr, "  cuDNN plan built. workspace=%ld bytes\n", ws);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vpack = {{Q_UID, dQ}, {K_UID, dK}, {V_UID, dV}, {O_UID, dO}};
    if (!graph->execute(handle, vpack, dWs).is_good()) { fprintf(stderr, "execute failed\n"); return 1; }
    CK(cudaDeviceSynchronize());

    // fp32 reference on a subset of heads (S^2*4 bytes per head)
    cublasHandle_t cbh; CBL(cublasCreate(&cbh));
    auto half2float_vec = [](const __half* p, size_t n, std::vector<float>& out){ out.resize(n); for (size_t i=0;i<n;i++) out[i]=__half2float(p[i]); };
    std::vector<__half> hO(qkv_n);
    CK(cudaMemcpy(hO.data(), dO, qkv_n * 2, cudaMemcpyDeviceToHost));
    double dot=0,na=0,nb=0,maxabs=0;
    std::vector<int> check_heads = {0, (int)H/2, (int)H-1};
    for (int h : check_heads) {
        size_t off = (size_t)h * S * D;
        std::vector<float> Qf, Kf, Vf;
        half2float_vec(hQ.data()+off, S*D, Qf); half2float_vec(hK.data()+off, S*D, Kf); half2float_vec(hV.data()+off, S*D, Vf);
        float *dQf,*dKf,*dVf,*dScores,*dRef;
        CK(cudaMalloc(&dQf, S*D*4)); CK(cudaMalloc(&dKf, S*D*4)); CK(cudaMalloc(&dVf, S*D*4));
        CK(cudaMalloc(&dScores, (size_t)S*S*4)); CK(cudaMalloc(&dRef, S*D*4));
        CK(cudaMemcpy(dQf, Qf.data(), S*D*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dKf, Kf.data(), S*D*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dVf, Vf.data(), S*D*4, cudaMemcpyHostToDevice));
        float one=1.f, zero=0.f; const int Si=(int)S, Di=(int)D;
        CBL(cublasSgemm(cbh, CUBLAS_OP_T, CUBLAS_OP_N, Si, Si, Di, &scale, dKf, Di, dQf, Di, &zero, dScores, Si));
        std::vector<float> hScores((size_t)S*S);
        CK(cudaMemcpy(hScores.data(), dScores, (size_t)S*S*4, cudaMemcpyDeviceToHost));
        for (int64_t j=0;j<S;j++) {
            float* col=&hScores[(size_t)j*S];
            float mx=-1e30f; for (int64_t i=0;i<S;i++) mx=fmaxf(mx,col[i]);
            float sum=0; for (int64_t i=0;i<S;i++){ col[i]=expf(col[i]-mx); sum+=col[i]; }
            float inv=1.f/sum; for (int64_t i=0;i<S;i++) col[i]*=inv;
        }
        CK(cudaMemcpy(dScores, hScores.data(), (size_t)S*S*4, cudaMemcpyHostToDevice));
        CBL(cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Di, Si, Si, &one, dVf, Di, dScores, Si, &zero, dRef, Di));
        std::vector<float> hRef((size_t)S*D);
        CK(cudaMemcpy(hRef.data(), dRef, (size_t)S*D*4, cudaMemcpyDeviceToHost));
        const __half* oh = hO.data()+off;
        for (size_t k=0;k<(size_t)S*D;k++){ double r=hRef[k], g=__half2float(oh[k]); dot+=r*g; na+=r*r; nb+=g*g; maxabs=fmax(maxabs,fabs(r-g)); }
        cudaFree(dQf); cudaFree(dKf); cudaFree(dVf); cudaFree(dScores); cudaFree(dRef);
    }
    double cosine = dot/(sqrt(na)*sqrt(nb)+1e-30);
    cublasDestroy(cbh);

    std::vector<float> times(iters);
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int it=0; it<iters; it++) {
        CK(cudaEventRecord(e0));
        if (!graph->execute(handle, vpack, dWs).is_good()) { fprintf(stderr, "execute(timed) failed\n"); return 1; }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms=0; CK(cudaEventElapsedTime(&ms, e0, e1)); times[it]=ms;
    }
    std::sort(times.begin(), times.end());
    double med_us = times[iters/2]*1e3;
    double flops = 2.0 * B * H * (2.0 * (double)S * S * D);
    double tflops = flops / (times[iters/2]*1e-3) / 1e12;
    fprintf(stderr, "  cosine=%.6f max|err|=%.5f  median %.1f us  %.1f TFLOP/s\n", cosine, maxabs, med_us, tflops);
    printf("{\"kernel\":\"cudnn_sdpa\",\"tag\":\"%s\",\"B\":%ld,\"H\":%ld,\"S\":%ld,\"D\":%ld,\"cosine\":%.6f,\"maxabs\":%.6f,\"us\":%.2f,\"tflops\":%.2f}\n",
           tag, B, H, S, D, cosine, maxabs, med_us, tflops);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); if (dWs) cudaFree(dWs);
    cudnnDestroy(handle);
    return (cosine >= 0.99) ? 0 : 1;
}
