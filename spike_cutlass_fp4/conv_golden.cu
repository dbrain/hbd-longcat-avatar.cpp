// Op-golden for cuDNN conv-fprop (3x3, stride1, pad1, fp16 IO / fp32 accum) at the
// flux2-klein-9b AutoencoderKL DECODER conv shapes (1024^2 output, "small" decoder ch=96).
//
// Goal: decisively measure whether a borrowed cuDNN implicit-GEMM convolution beats
// ggml's current im2col(materialize)+cuBLAS-GEMM conv path on the heavy VAE convs.
// The im2col path pays a large memory-blowup (writes IC*9 expansion to HBM) that the
// cuDNN implicit GEMM avoids -- that is the win to quantify.
//
// Mirrors attn_golden.cu style: cudnn-frontend graph build, warmup + median timing over
// ~50 iters with cudaEvents, fp32 DIRECT-conv reference (CUDA kernel) for cosine, emits
// JSON {kernel:"cudnn_conv", Cin,Cout,H,W, cosine, us, tflops}.
//
// Layout: cuDNN prefers NHWC for tensor-op fp16 conv. We feed NHWC activations and
// KRSC weights (cudnn-frontend conv_fprop convention: image [N,C,H,W] dims with NHWC
// strides; weight [K,C,R,S] dims with KRSC strides).
//
// Build (inside flux2-dev:builder-cudnn):
//   nvcc -O3 -arch=sm_120 -std=c++17 conv_golden.cu -o conv_golden \
//     -I/opt/cudnn-frontend/include -lcudnn -lcublas
//
// Run pinned to the 5060 Ti via CUDA_VISIBLE_DEVICES=GPU-bd93...

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

#define X_UID 1
#define W_UID 2
#define Y_UID 3

// ---- fp32 reference: direct 3x3 s1 p1 convolution, NHWC in/out, KRSC weights ----
// out[n,oh,ow,k] = sum_{c,r,s} in[n, oh+r-1, ow+s-1, c] * w[k, r, s, c]
__global__ void ref_conv3x3_s1p1(const float* __restrict__ x, const float* __restrict__ w,
                                 float* __restrict__ y, int N, int H, int W, int Cin, int Cout) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)N * H * W * Cout;
    if (idx >= total) return;
    int k  = idx % Cout;
    long t = idx / Cout;
    int ow = t % W; t /= W;
    int oh = t % H; t /= H;
    int n  = t;
    float acc = 0.f;
    for (int r = 0; r < 3; r++) {
        int ih = oh + r - 1;
        if (ih < 0 || ih >= H) continue;
        for (int s = 0; s < 3; s++) {
            int iw = ow + s - 1;
            if (iw < 0 || iw >= W) continue;
            const float* xp = x + (((long)n * H + ih) * W + iw) * Cin;
            const float* wp = w + (((long)k * 3 + r) * 3 + s) * Cin;
            for (int c = 0; c < Cin; c++) acc += xp[c] * wp[c];
        }
    }
    y[idx] = acc;
}

struct Shape { int Cin, Cout, H, W; const char* tag; };

static int bench_shape(cudnnHandle_t handle, const Shape& sh, int iters) {
    const int N = 1, R = 3, S = 3, pad = 1, stride = 1, dil = 1;
    const int Cin = sh.Cin, Cout = sh.Cout, H = sh.H, W = sh.W;
    fprintf(stderr, "=== cuDNN conv golden  %s  Cin=%d Cout=%d %dx%d  (3x3 s1 p1, NHWC fp16) ===\n",
            sh.tag, Cin, Cout, H, W);

    size_t xn = (size_t)N * H * W * Cin;       // NHWC
    size_t wn = (size_t)Cout * R * S * Cin;    // KRSC
    size_t yn = (size_t)N * H * W * Cout;      // NHWC

    // host random inputs (modest scale so fp32 ref is well conditioned)
    std::mt19937 rng(1234 + sh.Cin * 131 + sh.H);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> hX(xn), hW(wn);
    std::vector<float>  fX(xn), fW(wn);
    for (size_t i = 0; i < xn; i++) { float v = nd(rng) * 0.3f; fX[i] = v; hX[i] = __float2half(v); }
    for (size_t i = 0; i < wn; i++) { float v = nd(rng) * (0.1f / sqrtf((float)Cin)); fW[i] = v; hW[i] = __float2half(v); }

    __half *dX, *dW, *dY;
    CK(cudaMalloc(&dX, xn * 2)); CK(cudaMalloc(&dW, wn * 2)); CK(cudaMalloc(&dY, yn * 2));
    CK(cudaMemcpy(dX, hX.data(), xn * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, hW.data(), wn * 2, cudaMemcpyHostToDevice));

    // ---- build cuDNN conv-fprop graph ----
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // X: image dims [N,C,H,W] with NHWC strides
    auto X = graph->tensor(fe::graph::Tensor_attributes().set_name("X").set_uid(X_UID)
                 .set_dim({N, Cin, H, W})
                 .set_stride({(int64_t)H * W * Cin, 1, (int64_t)W * Cin, (int64_t)Cin}));
    // W: filter dims [K,C,R,S] with KRSC strides (k outermost, then r,s,c contiguous)
    auto Wt = graph->tensor(fe::graph::Tensor_attributes().set_name("W").set_uid(W_UID)
                 .set_dim({Cout, Cin, R, S})
                 .set_stride({(int64_t)R * S * Cin, 1, (int64_t)S * Cin, (int64_t)Cin}));

    auto conv_opts = fe::graph::Conv_fprop_attributes().set_name("conv3x3")
                         .set_padding({pad, pad}).set_stride({stride, stride}).set_dilation({dil, dil});
    auto Y = graph->conv_fprop(X, Wt, conv_opts);
    Y->set_output(true).set_dim({N, Cout, H, W})
     .set_stride({(int64_t)H * W * Cout, 1, (int64_t)W * Cout, (int64_t)Cout}).set_uid(Y_UID);

    if (!graph->validate().is_good())                    { fprintf(stderr, "validate failed\n"); return 1; }
    if (!graph->build_operation_graph(handle).is_good()) { fprintf(stderr, "build_operation_graph failed\n"); return 1; }
    if (!graph->create_execution_plans({fe::HeurMode_t::A}).is_good()) { fprintf(stderr, "create_execution_plans failed\n"); return 1; }
    if (!graph->check_support(handle).is_good())          { fprintf(stderr, "check_support failed (no conv engine for sm_120?)\n"); return 1; }
    if (!graph->build_plans(handle).is_good())            { fprintf(stderr, "build_plans failed\n"); return 1; }

    int64_t ws = 0;
    if (!graph->get_workspace_size(ws).is_good()) { fprintf(stderr, "get_workspace_size failed\n"); return 1; }
    void* dWs = nullptr; if (ws > 0) CK(cudaMalloc(&dWs, ws));
    fprintf(stderr, "  cuDNN plan built. workspace=%ld bytes\n", ws);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vpack = {
        {X_UID, dX}, {W_UID, dW}, {Y_UID, dY}};

    // warmup + correctness
    if (!graph->execute(handle, vpack, dWs).is_good()) { fprintf(stderr, "execute failed\n"); return 1; }
    CK(cudaDeviceSynchronize());

    // ---- fp32 direct-conv reference on device ----
    float *dXf, *dWf, *dRef;
    CK(cudaMalloc(&dXf, xn * 4)); CK(cudaMalloc(&dWf, wn * 4)); CK(cudaMalloc(&dRef, yn * 4));
    CK(cudaMemcpy(dXf, fX.data(), xn * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWf, fW.data(), wn * 4, cudaMemcpyHostToDevice));
    {
        long total = (long)N * H * W * Cout;
        int tpb = 256; long blk = (total + tpb - 1) / tpb;
        ref_conv3x3_s1p1<<<(unsigned)blk, tpb>>>(dXf, dWf, dRef, N, H, W, Cin, Cout);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    }
    std::vector<__half> hY(yn); std::vector<float> hRef(yn);
    CK(cudaMemcpy(hY.data(), dY, yn * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hRef.data(), dRef, yn * 4, cudaMemcpyDeviceToHost));

    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (size_t i = 0; i < yn; i++) {
        double r = hRef[i], g = __half2float(hY[i]);
        dot += r * g; na += r * r; nb += g * g; maxabs = fmax(maxabs, fabs(r - g));
    }
    double cosine = dot / (sqrt(na) * sqrt(nb) + 1e-30);

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
    // conv FLOPs = 2 * N * Cout * H * W * Cin * R * S
    double flops = 2.0 * N * (double)Cout * H * W * Cin * R * S;
    double tflops = flops / (med_ms * 1e-3) / 1e12;

    fprintf(stderr, "  cosine=%.6f max|err|=%.5f  median %.1f us  %.1f TFLOP/s\n",
            cosine, maxabs, med_us, tflops);

    printf("{\"kernel\":\"cudnn_conv\",\"tag\":\"%s\",\"Cin\":%d,\"Cout\":%d,\"H\":%d,\"W\":%d,"
           "\"cosine\":%.6f,\"maxabs\":%.6f,\"us\":%.2f,\"tflops\":%.2f,\"ws_bytes\":%ld}\n",
           sh.tag, Cin, Cout, H, W, cosine, maxabs, med_us, tflops, ws);
    fflush(stdout);

    cudaFree(dX); cudaFree(dW); cudaFree(dY); if (dWs) cudaFree(dWs);
    cudaFree(dXf); cudaFree(dWf); cudaFree(dRef);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (cosine >= 0.99) ? 0 : 2;
}

int main(int argc, char** argv) {
    int iters = 50;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) { fprintf(stderr, "cudnnCreate failed\n"); return 1; }

    // flux2 "small" decoder ch=96, ch_mult={1,2,4,4}, block_in=384, z=32, 1024^2 out.
    // Heavy DISTINCT 3x3 s1 p1 convs (from nsys im2col instances, gridX->IC confirmed):
    std::vector<Shape> shapes = {
        {384, 384, 128, 128, "mid/up3 384@128"},   // im2col ~1.36ms x10
        {384, 384, 256, 256, "up2 384@256"},        // im2col ~5.49ms x7
        {192, 192, 512, 512, "up1 192@512"},        // im2col ~11.08ms x5
        {384, 384, 512, 512, "up1in/ups 384@512"},  // im2col ~21.93ms x2 (heaviest mid-res)
        {96,  96, 1024, 1024, "up0 96@1024"},       // im2col ~23.18ms x6
        {192, 96, 1024, 1024, "up0in 192@1024"},    // im2col ~44.35ms x2 (heaviest)
    };

    int bad = 0;
    for (auto& sh : shapes) {
        int rc = bench_shape(handle, sh, iters);
        if (rc == 1) { fprintf(stderr, "  [BLOCKED] %s\n", sh.tag); bad |= 1; }
        else if (rc == 2) { fprintf(stderr, "  [LOW-COSINE] %s\n", sh.tag); bad |= 2; }
    }
    cudnnDestroy(handle);
    return bad ? 1 : 0;
}
