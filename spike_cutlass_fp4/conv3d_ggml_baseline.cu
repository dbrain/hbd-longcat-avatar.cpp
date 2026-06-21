// ggml im2col_3d + cuBLAS-GEMM baseline, SAME LTX-2.3 VAE decoder conv3d shapes, SAME GPU.
//
// Replicates ggml's conv3d path EXACTLY:
//   (1) the lap-23 shared-memory HALO-TILED im2col_3d kernel (HAS_PAD path) used for
//       the LTX VAE p0=p1=1 shapes -- copied VERBATIM from ggml/src/ggml-cuda/im2col.cu
//       (+ init_fastdiv_values / fast_div_modulo from common.cuh). This is NOT the naive
//       scattered gather: it is the optimized production kernel, so the baseline is honest.
//       Falls back to the fastdiv kernel on shapes the tiled path can't cover (also verbatim).
//   (2) cuBLAS hgemm: col(M=N*OD*OH*OW, K=IC*KD*KH*KW) x W(K, OC) -> Y(M, OC), fp16/fp32 accum
//       (ggml does mul_mat(im2col_out, weight_3d_reshaped)).
//
// Times the COMBINED im2col_3d + GEMM us/call (median, cudaEvents). The im2col write
// blowup is IC*27 elements to HBM -- the cost cuDNN's implicit 3D GEMM avoids.
//
// Build: nvcc -O3 -arch=sm_120 -std=c++17 conv3d_ggml_baseline.cu -o conv3d_ggml_baseline -lcublas

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CBL(x) do{ cublasStatus_t s=(x); if(s){fprintf(stderr,"cuBLAS err %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535
#define CUDA_IM2COL_BLOCK_SIZE 256
#define MIN(a,b) ((a)<(b)?(a):(b))

// ===== fastdiv helpers (verbatim from ggml/src/ggml-cuda/common.cuh) =====
static uint3 init_fastdiv_values(uint64_t d_64) {
    uint32_t d = (uint32_t)d_64;
    uint32_t L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) L++;
    uint32_t mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}
static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fv) {
    const uint32_t hi = __umulhi(n, fv.x);
    return (hi + n) >> fv.y;
}
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fv) {
    const uint32_t div_val = fastdiv(n, fv);
    const uint32_t mod_val = n - div_val * fv.z;
    return make_uint2(div_val, mod_val);
}

// ===== fastdiv (universal fallback) im2col_3d kernel (verbatim) =====
template <typename T>
static __global__ void im2col_3d_kernel(
        const float * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2,
        uint3 fd_kdkhkw, uint3 fd_khkw, uint3 fd_kw, uint3 fd_odoh, uint3 fd_oh) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KD_KH_KW) return;
    const uint2 dm_ic = fast_div_modulo((uint32_t) i,    fd_kdkhkw);
    const uint2 dm_kd = fast_div_modulo(dm_ic.y,         fd_khkw);
    const uint2 dm_kh = fast_div_modulo(dm_kd.y,         fd_kw);
    const int64_t iic = dm_ic.x;
    const int64_t ikd = dm_kd.x;
    const int64_t ikh = dm_kh.x;
    const int64_t ikw = dm_kh.y;
    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
            const uint2 dm_n  = fast_div_modulo((uint32_t) iz, fd_odoh);
            const uint2 dm_d  = fast_div_modulo(dm_n.y,        fd_oh);
            const int64_t in  = dm_n.x;
            const int64_t iod = dm_d.x;
            const int64_t ioh = dm_d.y;
            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t iid = iod * s2 + ikd * d2 - p2;
            const int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;
            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
                dst[offset_dst] = src[offset_src];
            }
        }
    }
    (void)N;(void)OC;(void)OH_OW;(void)OD;(void)OW;(void)KD;(void)KH;
    (void)ID_IH_IW;(void)IH_IW;(void)IC_ID_IH_IW;(void)OW_KD_KH_KW;(void)KD_KH_KW;(void)KH_KW;(void)KW;
}

// ===== lap-23 shared-memory halo-tiled im2col_3d kernel (verbatim) =====
template <typename T, int VEC, bool HAS_PAD>
static __global__ void im2col_3d_tiled_kernel(
        const float * __restrict__ src, T * __restrict__ dst,
        int IC, int ID, int IH, int IW,
        int KD, int KH, int KW, int OD, int OH, int OW,
        int CB, int TOH, int TOW, int p0, int p1,
        int KH_KW, int KD_KH_KW,
        int64_t IC_KD_KH_KW, int64_t OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW,
        int num_cblocks, int n_oh_tiles, int n_ow_tiles,
        uint3 fd_kdkhkw, uint3 fd_khkw, uint3 fd_kw, uint3 fd_tow, uint3 fd_towtoh,
        uint3 fd_ncblocks, uint3 fd_nohtiles) {
    extern __shared__ char smem_raw[];
    float * smem = (float *) smem_raw;
    const int SH = TOH + KH - 1;
    const int SW = TOW + KW - 1;
    const int DHW = ID * SH * SW;
    const int bz = blockIdx.z;
    const int in = fastdiv((uint32_t) bz, fd_ncblocks);
    const int cb = bz - in * num_cblocks;
    const int cb0 = cb * CB;
    const int CBeff = min(CB, IC - cb0);
    const int oh0 = blockIdx.y * TOH;
    const int ow0 = blockIdx.x * TOW;
    const int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const int64_t src_chan_base = (int64_t)(in * IC + cb0) * ID * IH * IW;
    const int smem_elems = CBeff * DHW;
    for (int e = tid; e < smem_elems; e += nthreads) {
        int t = e;
        const int sc  = t % SW;  t /= SW;
        const int sr  = t % SH;  t /= SH;
        const int iid = t % ID;  t /= ID;
        const int icl = t;
        int iih, iiw;
        bool in_bounds;
        if constexpr (HAS_PAD) {
            iih = oh0 + sr - p1;
            iiw = ow0 + sc - p0;
            in_bounds = (iih >= 0 && iih < IH && iiw >= 0 && iiw < IW);
        } else {
            iih = oh0 + sr;
            iiw = ow0 + sc;
            in_bounds = (iih < IH && iiw < IW);
        }
        float v = 0.0f;
        if (in_bounds) {
            v = src[src_chan_base + (int64_t)icl * ID * IH * IW + (int64_t)iid * IH * IW + (int64_t)iih * IW + iiw];
        }
        smem[(int64_t)icl * DHW + ((int64_t)iid * SH + sr) * SW + sc] = v;
    }
    __syncthreads();
    const int c0 = threadIdx.x * VEC;
    int s_icl[VEC], s_ikd[VEC], s_ikh[VEC], s_ikw[VEC];
    bool any_active = false;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        const uint2 d_icl = fast_div_modulo((uint32_t)(c0 + j), fd_kdkhkw);
        const uint2 d_ikd = fast_div_modulo(d_icl.y,            fd_khkw);
        const uint2 d_ikh = fast_div_modulo(d_ikd.y,            fd_kw);
        s_icl[j] = d_icl.x; s_ikd[j] = d_ikd.x; s_ikh[j] = d_ikh.x; s_ikw[j] = d_ikh.y;
        any_active |= (d_icl.x < CBeff);
    }
    const int64_t col = (int64_t) cb0 * KD_KH_KW + c0;
    const int n_out = OD * TOH * TOW;
    for (int o = threadIdx.y; o < n_out; o += blockDim.y) {
        const uint2 d_iod = fast_div_modulo((uint32_t) o, fd_towtoh);
        const uint2 d_loh = fast_div_modulo(d_iod.y,      fd_tow);
        const int iod = d_iod.x;
        const int loh = d_loh.x;
        const int low = d_loh.y;
        const int oh  = oh0 + loh;
        const int ow  = ow0 + low;
        if (!any_active || oh >= OH || ow >= OW) continue;
        const int64_t base = (int64_t) in * OD_OH_OW_IC_KD_KH_KW
                           + (int64_t) iod * OH_OW_IC_KD_KH_KW
                           + (int64_t) oh  * OW_IC_KD_KH_KW
                           + (int64_t) ow  * IC_KD_KH_KW
                           + col;
        float v[VEC];
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            v[j] = (s_icl[j] < CBeff)
                 ? smem[(int64_t)s_icl[j] * DHW + ((int64_t)(iod + s_ikd[j]) * SH + (loh + s_ikh[j])) * SW + (low + s_ikw[j])]
                 : 0.0f;
        }
        if constexpr (VEC == 2 && sizeof(T) == 2) {
            __half2 h2 = __floats2half2_rn(v[0], v[1]);
            *reinterpret_cast<__half2 *>(&dst[base]) = h2;
        } else {
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                if (s_icl[j] < CBeff) dst[base + j] = v[j];
            }
        }
    }
    (void)n_oh_tiles;(void)n_ow_tiles;(void)fd_nohtiles;
}

// host launcher for tiled fast path (verbatim logic), returns false if uncovered
template <typename T>
static bool im2col_3d_tiled_try(const float * src, T * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {
    if (!(s0 == 1 && s1 == 1 && s2 == 1 && d0 == 1 && d1 == 1 && d2 == 1 && p2 == 0)) return false;
    const bool has_pad = (p0 != 0 || p1 != 0);
    int CB = 8, TOH = 8, TOW = 8, P = 4;
    if (const char * t = getenv("LONGCAT_IM2COL_TILE")) sscanf(t, "%d,%d,%d,%d", &CB, &TOH, &TOW, &P);
    const int KD_KH_KW = (int)(KD * KH * KW);
    const int cols = CB * KD_KH_KW;
    const int VEC = (sizeof(T) == 2 && (cols % 2) == 0 && (IC % CB) == 0 &&
                     !getenv("LONGCAT_IM2COL_NOVEC2")) ? 2 : 1;
    const int tpx = cols / VEC;
    if (tpx > 1024 || (int64_t) tpx * P > 1024) return false;
    auto smem_for = [&](int toh, int tow) {
        const int sh = toh + (int) KH - 1, sw = tow + (int) KW - 1;
        return (size_t) CB * ID * sh * sw * sizeof(float);
    };
    if (has_pad) {
        while ((TOH > 1 || TOW > 1) && smem_for(TOH, TOW) > 48 * 1024) {
            if (TOW >= TOH && TOW > 1) TOW--; else if (TOH > 1) TOH--; else break;
        }
    }
    const int SH = TOH + (int) KH - 1;
    const int SW = TOW + (int) KW - 1;
    const size_t smem_bytes = (size_t) CB * ID * SH * SW * sizeof(float);
    if (smem_bytes > 48 * 1024) return false;
    const int num_cblocks = (int)((IC + CB - 1) / CB);
    const int n_oh_tiles  = (int)((OH + TOH - 1) / TOH);
    const int n_ow_tiles  = (int)((OW + TOW - 1) / TOW);
    const int64_t gz = N * num_cblocks;
    if (n_ow_tiles > MAX_GRIDDIM_Y || n_oh_tiles > MAX_GRIDDIM_Y || gz > MAX_GRIDDIM_Z) return false;
    const int64_t IC_KD_KH_KW          = IC * KD * KH * KW;
    const int64_t OW_IC_KD_KH_KW       = OW * IC_KD_KH_KW;
    const int64_t OH_OW_IC_KD_KH_KW    = OH * OW_IC_KD_KH_KW;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD * OH_OW_IC_KD_KH_KW;
    const uint3 fd_kdkhkw  = init_fastdiv_values((uint64_t) KD_KH_KW);
    const uint3 fd_khkw    = init_fastdiv_values((uint64_t)(KH * KW));
    const uint3 fd_kw      = init_fastdiv_values((uint64_t) KW);
    const uint3 fd_tow     = init_fastdiv_values((uint64_t) TOW);
    const uint3 fd_towtoh  = init_fastdiv_values((uint64_t)(TOW * TOH));
    const uint3 fd_ncblocks= init_fastdiv_values((uint64_t) num_cblocks);
    const uint3 fd_nohtiles= init_fastdiv_values((uint64_t) n_oh_tiles);
    dim3 grid(n_ow_tiles, n_oh_tiles, (unsigned) gz);
    dim3 block(tpx, P, 1);
#define LC_TILED_LAUNCH(VECN, HASPAD) \
    im2col_3d_tiled_kernel<T, VECN, HASPAD><<<grid, block, smem_bytes, stream>>>( \
        src, dst, (int)IC, (int)ID, (int)IH, (int)IW, \
        (int)KD, (int)KH, (int)KW, (int)OD, (int)OH, (int)OW, \
        CB, TOH, TOW, p0, p1, (int)(KH * KW), KD_KH_KW, \
        IC_KD_KH_KW, OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW, \
        num_cblocks, n_oh_tiles, n_ow_tiles, \
        fd_kdkhkw, fd_khkw, fd_kw, fd_tow, fd_towtoh, fd_ncblocks, fd_nohtiles)
    if (has_pad) {
        if (VEC == 2) { LC_TILED_LAUNCH(2, true); } else { LC_TILED_LAUNCH(1, true); }
    } else {
        if (VEC == 2) { LC_TILED_LAUNCH(2, false); } else { LC_TILED_LAUNCH(1, false); }
    }
#undef LC_TILED_LAUNCH
    return true;
}

struct Shape { int Cin, Cout, ID, H, W; const char* tag; };

static void bench(cublasHandle_t cb, const Shape& sh, int iters) {
    const int N = 1, KD = 3, KH = 3, KW = 3;
    const int s0=1,s1=1,s2=1, p0=1,p1=1,p2=0, d0=1,d1=1,d2=1;
    const int IC = sh.Cin, OC = sh.Cout, ID = sh.ID, IH = sh.H, IW = sh.W;
    const int OD = ID - KD + 1;            // depth pad 0
    const int OH = IH, OW = IW;            // spatial s1p1
    if (OD <= 0) { fprintf(stderr, "  ID too small for %s\n", sh.tag); return; }
    const int64_t M    = (int64_t)N * OD * OH * OW;
    const int64_t Kdim = (int64_t)IC * KD * KH * KW;
    fprintf(stderr, "=== ggml im2col_3d+GEMM  %s  Cin=%d Cout=%d ID=%d %dx%d  M=%ld K=%ld ===\n",
            sh.tag, IC, OC, ID, IH, IW, M, Kdim);

    size_t xn = (size_t)N * IC * ID * IH * IW;
    size_t coln = (size_t)M * Kdim;
    size_t wn = (size_t)Kdim * OC;
    size_t yn = (size_t)M * OC;

    float* dX; __half *dCol, *dW, *dY;
    CK(cudaMalloc(&dX, xn * 4)); CK(cudaMalloc(&dCol, coln * 2));
    CK(cudaMalloc(&dW, wn * 2)); CK(cudaMalloc(&dY, yn * 2));
    std::mt19937 rng(99 + IC); std::normal_distribution<float> nd(0, 1);
    { std::vector<float> h(xn); for (auto& v : h) v = nd(rng) * 0.3f; CK(cudaMemcpy(dX, h.data(), xn * 4, cudaMemcpyHostToDevice)); }
    { std::vector<__half> h(wn); for (auto& v : h) v = __float2half(nd(rng) * 0.05f); CK(cudaMemcpy(dW, h.data(), wn * 2, cudaMemcpyHostToDevice)); }

    // ggml src strides (contiguous NCDHW): stride_q=ID*IH*IW(per-chan), z=IH*IW, y=IW, x=1
    const int64_t stride_q = (int64_t)ID*IH*IW, stride_z=(int64_t)IH*IW, stride_y=(int64_t)IW, stride_x=1;
    // fastdiv-fallback launch params (verbatim from im2col_3d_cuda)
    const int64_t OH_OW=(int64_t)OH*OW, KD_KH_KW=(int64_t)KD*KH*KW, ID_IH_IW=(int64_t)ID*IH*IW;
    const int64_t KH_KW=(int64_t)KH*KW, IH_IW=(int64_t)IH*IW, IC_KD_KH_KW=(int64_t)IC*KD*KH*KW;
    const int64_t OW_KD_KH_KW=(int64_t)OW*KD*KH*KW, N_OD_OH=(int64_t)N*OD*OH, OD_OH=(int64_t)OD*OH;
    const int64_t IC_ID_IH_IW=(int64_t)IC*ID*IH*IW;
    const int64_t OD_OH_OW_IC_KD_KH_KW=(int64_t)OD*OH*OW*IC*KD*KH*KW;
    const int64_t OH_OW_IC_KD_KH_KW=(int64_t)OH*OW*IC*KD*KH*KW, OW_IC_KD_KH_KW=(int64_t)OW*IC*KD*KH*KW;
    const int64_t num_blocks = (IC_KD_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OD_OH, MAX_GRIDDIM_Z));
    const uint3 fd_kdkhkw=init_fastdiv_values(KD_KH_KW), fd_khkw=init_fastdiv_values(KH_KW);
    const uint3 fd_kw=init_fastdiv_values(KW), fd_odoh=init_fastdiv_values(OD_OH), fd_oh=init_fastdiv_values(OH);

    bool used_tiled_probe = false;
    auto run_im2col = [&]() {
        bool ut = im2col_3d_tiled_try<__half>(dX, dCol, N, IC, ID, IH, IW, KD, KH, KW, OD, OH, OW,
                                              s0,s1,s2,p0,p1,p2,d0,d1,d2, 0);
        used_tiled_probe = ut;
        if (!ut) {
            im2col_3d_kernel<__half><<<block_nums, MIN(IC_KD_KH_KW, CUDA_IM2COL_BLOCK_SIZE)>>>(
                dX, dCol, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                OH_OW, KD_KH_KW, ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW,
                IC_KD_KH_KW, OW_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                stride_q, stride_z, stride_y, stride_x, s0,s1,s2,p0,p1,p2,d0,d1,d2,
                fd_kdkhkw, fd_khkw, fd_kw, fd_odoh, fd_oh);
        }
    };
    float alpha = 1.f, beta = 0.f;
    auto run_gemm = [&]() {
        cublasStatus_t st = cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int)OC, (int)M, (int)Kdim,
                         &alpha, dW, CUDA_R_16F, (int)OC, dCol, CUDA_R_16F, (int)Kdim,
                         &beta, dY, CUDA_R_16F, (int)OC, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st) { fprintf(stderr, "GemmEx status=%d  OC=%d M=%ld K=%ld\n", (int)st, (int)OC, M, Kdim); exit(1); }
    };

    run_im2col();
    cudaError_t ie = cudaDeviceSynchronize();
    if (ie) { fprintf(stderr, "im2col launch err: %s\n", cudaGetErrorString(ie)); exit(1); }
    run_gemm(); CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1, em; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&em));
    std::vector<float> t_comb(iters), t_im(iters), t_gemm(iters);
    for (int it = 0; it < iters; it++) {
        CK(cudaEventRecord(e0));
        run_im2col();
        CK(cudaEventRecord(em));
        run_gemm();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float a=0,b=0; CK(cudaEventElapsedTime(&a, e0, em)); CK(cudaEventElapsedTime(&b, em, e1));
        t_im[it]=a; t_gemm[it]=b; t_comb[it]=a+b;
    }
    auto med = [&](std::vector<float>& v){ std::sort(v.begin(), v.end()); return v[v.size()/2]; };
    double mc=med(t_comb), mi=med(t_im), mg=med(t_gemm);
    double flops = 2.0 * N * (double)OC * OD * OH * OW * IC * KD * KH * KW;
    double tflops = flops / (mc * 1e-3) / 1e12;
    double wbytes = (double)coln * 2.0;
    double gbps = (wbytes / 1e9) / (mi * 1e-3);

    fprintf(stderr, "  combined %.1f us (im2col %.1f [%s] + gemm %.1f)  %.1f TFLOP/s  im2col-write %.0f GB/s\n",
            mc*1e3, mi*1e3, used_tiled_probe ? "TILED":"fastd", mg*1e3, tflops, gbps);
    printf("{\"kernel\":\"ggml_im2col3d_gemm\",\"tag\":\"%s\",\"Cin\":%d,\"Cout\":%d,\"ID\":%d,\"H\":%d,\"W\":%d,"
           "\"path\":\"%s\",\"comb_us\":%.2f,\"im2col_us\":%.2f,\"gemm_us\":%.2f,\"tflops\":%.2f,\"im2col_gbps\":%.0f}\n",
           sh.tag, IC, OC, ID, IH, IW, used_tiled_probe?"tiled":"fastdiv", mc*1e3, mi*1e3, mg*1e3, tflops, gbps);
    fflush(stdout);

    cudaFree(dX); cudaFree(dCol); cudaFree(dW); cudaFree(dY);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(em);
}

int main(int argc, char** argv) {
    int iters = 50, ID = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        if (!strcmp(argv[i], "--id")    && i + 1 < argc) ID    = atoi(argv[++i]);
    }
    cublasHandle_t cb; CBL(cublasCreate(&cb));
    void* ws = nullptr; CK(cudaMalloc(&ws, 64u*1024*1024)); CBL(cublasSetWorkspace(cb, ws, 64u*1024*1024));
    std::vector<Shape> shapes = {
        {1024, 1024, ID,  22,  40, "blk0 res 1024@22x40"},
        { 512,  512, ID,  44,  80, "blk2 res 512@44x80"},
        { 512,  512, ID,  88, 160, "blk4 res 512@88x160"},
        { 256,  256, ID, 176, 320, "blk6 res 256@176x320"},
        { 128,  128, ID, 176, 320, "blk8 res 128@176x320"},
    };
    for (auto& sh : shapes) bench(cb, sh, iters);
    cublasDestroy(cb);
    return 0;
}
