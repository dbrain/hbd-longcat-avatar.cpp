// Op-golden for cuDNN 3D conv-fprop (3x3x3, stride1, spatial-pad1, depth-pad0
// causal-prepadded; fp16 NDHWC / fp32 accum) at the LTX-2.3 22B VideoVAE DECODER
// conv3d shapes (704x1280 video target). Mirrors conv_golden.cu (the 2D flux2 spike).
//
// Why this is the BIG lever: ggml decodes LTX VAE via im2col_3d (materialize the
// IC*27 column blowup to HBM) + cuBLAS GEMM. cuDNN's implicit 3D GEMM avoids the
// materialization entirely. The flux2 *2D* spike already saw 2.5-5.8x; the 3D
// im2col writes 27x (vs 9x) so the blowup is worse -> expect a bigger win.
//
// NOTE: the ggml baseline here is NOT the naive scattered gather -- the LTX VAE
// im2col_3d path is the lap-23 shared-memory HALO-TILED kernel (HAS_PAD), tuned for
// exactly these p0=p1=1 shapes. conv3d_ggml_baseline.cu replicates that tiled kernel
// VERBATIM so the comparison is honest (cuDNN vs the optimized ggml path, not a
// strawman).
//
// LTX-2.3 22B decoder ladder (from ltx-2.3-22b-distilled_video_vae.safetensors):
//   latent 22x40 (704/32 x 1280/32). 3 spatial upsamples -> conv_out grid 176x320,
//   then patch_size=4 unpatchify -> 704x1280. Channel/res tiers:
//     blk0  res x2   1024ch @ 22x40    (heaviest channels, latent res)
//     blk2  res x2    512ch @ 44x80
//     blk4  res x4    512ch @ 88x160   (8 convs)
//     blk6  res x6    256ch @ 176x320  (12 convs, high res)
//     blk8  res x4    128ch @ 176x320  (8 convs, highest res)
//   ID (temporal) per conv = tile_T + 2 causal pad; tiled decode uses small T.
//   We bench ID=3 (per-frame tiled, the common case) for each tier.
//
// Build (inside flux2-dev:builder-cudnn):
//   nvcc -O3 -arch=sm_120 -std=c++17 conv3d_golden.cu -o conv3d_golden \
//     -I/opt/cudnn-frontend/include -lcudnn -lcublas

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

// ---- fp32 reference: direct 3x3x3 conv, NDHWC in/out, KCRS-3d weights.
// Padding: spatial p1=1 (H,W), depth p0=0 (causal pre-pad => OD = ID-2). ----
// out[n,od,oh,ow,k] = sum_{c,kd,kh,kw} in[n, od+kd, oh+kh-1, ow+kw-1, c] * w[k,kd,kh,kw,c]
__global__ void ref_conv3d(const float* __restrict__ x, const float* __restrict__ w,
                           float* __restrict__ y, int N, int ID, int IH, int IW,
                           int Cin, int Cout, int OD, int OH, int OW) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)N * OD * OH * OW * Cout;
    if (idx >= total) return;
    int k  = idx % Cout;
    long t = idx / Cout;
    int ow = t % OW; t /= OW;
    int oh = t % OH; t /= OH;
    int od = t % OD; t /= OD;
    int n  = t;
    float acc = 0.f;
    for (int kd = 0; kd < 3; kd++) {
        int id = od + kd;                 // depth pad 0 (causal pre-pad)
        if (id < 0 || id >= ID) continue;
        for (int kh = 0; kh < 3; kh++) {
            int ih = oh + kh - 1;          // spatial pad 1
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < 3; kw++) {
                int iw = ow + kw - 1;
                if (iw < 0 || iw >= IW) continue;
                const float* xp = x + ((((long)n * ID + id) * IH + ih) * IW + iw) * Cin;
                const float* wp = w + ((((long)k * 3 + kd) * 3 + kh) * 3 + kw) * Cin;
                for (int c = 0; c < Cin; c++) acc += xp[c] * wp[c];
            }
        }
    }
    y[idx] = acc;
}

struct Shape { int Cin, Cout, ID, H, W; const char* tag; };

static int bench_shape(cudnnHandle_t handle, const Shape& sh, int iters) {
    const int N = 1, KD = 3, KH = 3, KW = 3;
    const int p_d = 0, p_h = 1, p_w = 1;          // causal depth pad external, spatial pad 1
    const int Cin = sh.Cin, Cout = sh.Cout, ID = sh.ID, H = sh.H, W = sh.W;
    const int OD = ID - KD + 1 + 2 * p_d;          // = ID-2
    const int OH = H, OW = W;                       // spatial s1p1 -> same
    fprintf(stderr, "=== cuDNN conv3d  %s  Cin=%d Cout=%d ID=%d %dx%d -> OD=%d (3x3x3 s1 pH=1 pW=1 pD=0, NDHWC fp16) ===\n",
            sh.tag, Cin, Cout, ID, H, W, OD);
    if (OD <= 0) { fprintf(stderr, "  ID too small\n"); return 1; }

    size_t xn = (size_t)N * ID * H * W * Cin;            // NDHWC
    size_t wn = (size_t)Cout * KD * KH * KW * Cin;       // K,KD,KH,KW,C
    size_t yn = (size_t)N * OD * OH * OW * Cout;         // NDHWC

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

    // ---- build cuDNN 3D conv-fprop graph ----
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // X: dims [N,C,D,H,W] with NDHWC strides
    auto X = graph->tensor(fe::graph::Tensor_attributes().set_name("X").set_uid(X_UID)
                 .set_dim({N, Cin, ID, H, W})
                 .set_stride({(int64_t)ID * H * W * Cin, 1,
                              (int64_t)H * W * Cin, (int64_t)W * Cin, (int64_t)Cin}));
    // W: dims [K,C,KD,KH,KW] with K,KD,KH,KW,C (=KRS-3d) strides
    auto Wt = graph->tensor(fe::graph::Tensor_attributes().set_name("W").set_uid(W_UID)
                 .set_dim({Cout, Cin, KD, KH, KW})
                 .set_stride({(int64_t)KD * KH * KW * Cin, 1,
                              (int64_t)KH * KW * Cin, (int64_t)KW * Cin, (int64_t)Cin}));

    auto conv_opts = fe::graph::Conv_fprop_attributes().set_name("conv3x3x3")
                         .set_padding({p_d, p_h, p_w})
                         .set_stride({1, 1, 1})
                         .set_dilation({1, 1, 1});
    auto Y = graph->conv_fprop(X, Wt, conv_opts);
    Y->set_output(true).set_dim({N, Cout, OD, OH, OW})
     .set_stride({(int64_t)OD * OH * OW * Cout, 1,
                  (int64_t)OH * OW * Cout, (int64_t)OW * Cout, (int64_t)Cout}).set_uid(Y_UID);

    if (!graph->validate().is_good())                    { fprintf(stderr, "validate failed\n"); return 1; }
    if (!graph->build_operation_graph(handle).is_good()) { fprintf(stderr, "build_operation_graph failed\n"); return 1; }
    if (!graph->create_execution_plans({fe::HeurMode_t::A}).is_good()) { fprintf(stderr, "create_execution_plans failed\n"); return 1; }
    if (!graph->check_support(handle).is_good())          { fprintf(stderr, "check_support FAILED (no 3D conv engine for sm_120?)\n"); return 1; }
    if (!graph->build_plans(handle).is_good())            { fprintf(stderr, "build_plans failed\n"); return 1; }

    int64_t ws = 0;
    if (!graph->get_workspace_size(ws).is_good()) { fprintf(stderr, "get_workspace_size failed\n"); return 1; }
    void* dWs = nullptr; if (ws > 0) CK(cudaMalloc(&dWs, ws));
    fprintf(stderr, "  cuDNN plan built. workspace=%ld bytes\n", ws);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vpack = {
        {X_UID, dX}, {W_UID, dW}, {Y_UID, dY}};

    if (!graph->execute(handle, vpack, dWs).is_good()) { fprintf(stderr, "execute failed\n"); return 1; }
    CK(cudaDeviceSynchronize());

    // ---- fp32 direct-conv reference ----
    float *dXf, *dWf, *dRef;
    CK(cudaMalloc(&dXf, xn * 4)); CK(cudaMalloc(&dWf, wn * 4)); CK(cudaMalloc(&dRef, yn * 4));
    CK(cudaMemcpy(dXf, fX.data(), xn * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWf, fW.data(), wn * 4, cudaMemcpyHostToDevice));
    {
        long total = (long)N * OD * OH * OW * Cout;
        int tpb = 256; long blk = (total + tpb - 1) / tpb;
        ref_conv3d<<<(unsigned)blk, tpb>>>(dXf, dWf, dRef, N, ID, H, W, Cin, Cout, OD, OH, OW);
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

    // ---- timing ----
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
    double flops = 2.0 * N * (double)Cout * OD * OH * OW * Cin * KD * KH * KW;
    double tflops = flops / (med_ms * 1e-3) / 1e12;

    fprintf(stderr, "  cosine=%.6f max|err|=%.5f  median %.1f us  %.1f TFLOP/s  ws=%ld\n",
            cosine, maxabs, med_us, tflops, ws);

    printf("{\"kernel\":\"cudnn_conv3d\",\"tag\":\"%s\",\"Cin\":%d,\"Cout\":%d,\"ID\":%d,\"H\":%d,\"W\":%d,"
           "\"OD\":%d,\"cosine\":%.6f,\"maxabs\":%.6f,\"us\":%.2f,\"tflops\":%.2f,\"ws_bytes\":%ld}\n",
           sh.tag, Cin, Cout, ID, H, W, OD, cosine, maxabs, med_us, tflops, ws);
    fflush(stdout);

    cudaFree(dX); cudaFree(dW); cudaFree(dY); if (dWs) cudaFree(dWs);
    cudaFree(dXf); cudaFree(dWf); cudaFree(dRef);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (cosine >= 0.99) ? 0 : 2;
}

int main(int argc, char** argv) {
    int iters = 50, ID = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        if (!strcmp(argv[i], "--id")    && i + 1 < argc) ID    = atoi(argv[++i]);
    }
    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) { fprintf(stderr, "cudnnCreate failed\n"); return 1; }

    // LTX-2.3 22B decoder conv3d tiers (Cin,Cout,ID,H,W). ID set via --id (default 3).
    std::vector<Shape> shapes = {
        {1024, 1024, ID,  22,  40, "blk0 res 1024@22x40"},
        { 512,  512, ID,  44,  80, "blk2 res 512@44x80"},
        { 512,  512, ID,  88, 160, "blk4 res 512@88x160"},
        { 256,  256, ID, 176, 320, "blk6 res 256@176x320"},
        { 128,  128, ID, 176, 320, "blk8 res 128@176x320"},
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
