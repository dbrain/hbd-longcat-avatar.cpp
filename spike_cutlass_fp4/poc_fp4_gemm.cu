// Standalone cuBLASLt NVFP4 blockscaled GEMM PoC for sm_120 (RTX 5060 Ti).
// Proves: (a) cuBLASLt FP4 GEMM builds+runs on CUDA 12.9, (b) correctness vs fp32 ref,
// (c) throughput (TFLOPs) on DiT-representative shapes.
//
// Layout follows comfy-kitchen cublas_gemm_nvfp4.cu: A=weight (N x K), B=activation (M x K),
// both E2M1 packed, per-16 UE4M3 block scales in SWIZZLE_32_4_4 layout, alpha = global scale.
// TN GEMM: D[M,N] = (B[M,K] * scale_b) @ (A[N,K] * scale_a)^T  (row-major torch view).

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{ cublasStatus_t s=(x); if(s){printf("cuBLAS err %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

static const float FP4_MAX = 6.0f;      // E2M1 max
static const float FP8E4M3_MAX = 448.0f;

// E2M1 nibble encode of a value already scaled into [-6,6]
static uint8_t enc_e2m1(float v){
  static const float lv[8]={0,0.5f,1,1.5f,2,3,4,6};
  float a=fabsf(v); int best=0; float bd=1e30f;
  for(int i=0;i<8;i++){float d=fabsf(a-lv[i]); if(d<bd){bd=d;best=i;}}
  uint8_t n=best; if(v<0) n|=0x8; return n; // sign bit high
}
static float dec_e2m1(uint8_t n){
  static const float lv[8]={0,0.5f,1,1.5f,2,3,4,6};
  float a=lv[n&0x7]; return (n&0x8)?-a:a;
}

// SWIZZLE_32_4_4 offset (from comfy float_utils.cuh scale_factor_swizzled_offset)
static size_t swz(size_t row_idx, size_t col_idx, uint32_t col_length){
  const uint32_t R=128,RC=32,CC=4;
  size_t rb=row_idx/R, rem=row_idx%R, d4=rem/RC, d3=rem%RC;
  size_t cbg=col_idx/CC, d5=col_idx%CC;
  size_t cbg_cnt=(col_length+CC-1)/CC;
  return ((rb*cbg_cnt+cbg)*RC+d3)*16 + d4*CC + d5;
}

// Quantize one row-major matrix (rows x cols, cols%16==0) to NVFP4.
// Returns packed E2M1 (rows*cols/2 bytes, row-major, 2 vals/byte low=even),
// swizzled UE4M3 block scales, and chosen per-tensor global (decode) scale.
struct Q { std::vector<uint8_t> data; std::vector<uint8_t> scales; float global; std::vector<float> ref; };
static Q quantize(const std::vector<float>& in, int rows, int cols){
  Q q; int nb=cols/16;
  float amax=0; for(float v:in) amax=fmaxf(amax,fabsf(v));
  float global = amax/(FP4_MAX*FP8E4M3_MAX); if(global<=0) global=1e-8f;
  q.global=global;
  size_t rb_p=((rows+127)/128)*128, cb_p=((nb+3)/4)*4;
  q.scales.assign(rb_p*cb_p, 0);
  q.data.assign((size_t)rows*cols/2, 0);
  q.ref.assign((size_t)rows*cols,0);
  for(int r=0;r<rows;r++){
    for(int b=0;b<nb;b++){
      float ba=0; for(int j=0;j<16;j++) ba=fmaxf(ba,fabsf(in[(size_t)r*cols+b*16+j]));
      float decode = ba/FP4_MAX/global;
      decode=fminf(decode,FP8E4M3_MAX);
      __nv_fp8_e4m3 sf((float)decode); float decode_q=(float)sf;
      q.scales[swz(r,b,nb)] = *reinterpret_cast<uint8_t*>(&sf);
      float enc = (decode_q*global>0)? 1.0f/(decode_q*global) : 0.0f;
      for(int j=0;j<16;j++){
        size_t idx=(size_t)r*cols+b*16+j;
        uint8_t n=enc_e2m1(in[idx]*enc);
        float dq=dec_e2m1(n)*decode_q*global; q.ref[idx]=dq;
        size_t byte=idx/2; if(idx&1) q.data[byte]|=(n<<4); else q.data[byte]|=n;
      }
    }
  }
  return q;
}

int main(int argc,char**argv){
  int M=atoi(argv[1]), K=atoi(argv[2]), N=atoi(argv[3]);
  int iters = argc>4?atoi(argv[4]):200;
  printf("=== FP4 GEMM  M=%d K=%d N=%d (torch row-major: A[M,K]@W[N,K]^T=D[M,N]) ===\n",M,K,N);

  std::mt19937 rng(1234); std::normal_distribution<float> nd(0,1);
  std::vector<float> Wf((size_t)N*K), Af((size_t)M*K);
  for(auto&v:Wf) v=nd(rng)*0.05f;
  for(auto&v:Af) v=nd(rng)*0.5f;

  Q qW=quantize(Wf,N,K);   // weight  -> A in cublas impl
  Q qA=quantize(Af,M,K);   // activation -> B in cublas impl
  float alpha = qW.global*qA.global;

  // device buffers
  uint8_t *dW,*dA,*dWs,*dAs; void* dD; float* dAlpha;
  CK(cudaMalloc(&dW,qW.data.size())); CK(cudaMalloc(&dA,qA.data.size()));
  CK(cudaMalloc(&dWs,qW.scales.size())); CK(cudaMalloc(&dAs,qA.scales.size()));
  CK(cudaMalloc(&dD,(size_t)M*N*2)); CK(cudaMalloc(&dAlpha,4));
  CK(cudaMemcpy(dW,qW.data.data(),qW.data.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dA,qA.data.data(),qA.data.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dWs,qW.scales.data(),qW.scales.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dAs,qA.scales.data(),qA.scales.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dAlpha,&alpha,4,cudaMemcpyHostToDevice));
  float zero=0; float* dBeta; CK(cudaMalloc(&dBeta,4)); CK(cudaMemcpy(dBeta,&zero,4,cudaMemcpyHostToDevice));
  void* ws; size_t wsz=32*1024*1024; CK(cudaMalloc(&ws,wsz));

  cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
  // cublas column-major: m=N, n=M, k=K. lda=ldb=K (packed /2 elems? cuBLAS k in elements). ldc=ldd=N.
  int m=N, n=M, k=K;
  cublasLtMatmulDesc_t op; CB(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm)));
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm)));
  cublasOperation_t T=CUBLAS_OP_T,Nn=CUBLAS_OP_N;
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&T,sizeof(T)));
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nn,sizeof(Nn)));
  void* wsp=(void*)dWs; void* asp=(void*)dAs;
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&wsp,sizeof(wsp)));
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&asp,sizeof(asp)));
  cublasDataType_t st=CUDA_R_32F;
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_SCALE_TYPE,&st,sizeof(st)));
  cublasLtPointerMode_t pm=CUBLASLT_POINTER_MODE_DEVICE;
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_POINTER_MODE,&pm,sizeof(pm)));

  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
  // A: weight, transa=T => stored as k x m, ld=k
  CB(cublasLtMatrixLayoutCreate(&Ad,CUDA_R_4F_E2M1,k,m,k));
  CB(cublasLtMatrixLayoutCreate(&Bd,CUDA_R_4F_E2M1,k,n,k));
  CB(cublasLtMatrixLayoutCreate(&Cd,CUDA_R_16F,m,n,m));
  CB(cublasLtMatrixLayoutCreate(&Dd,CUDA_R_16F,m,n,m));

  cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
  CB(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)));
  cublasLtMatmulHeuristicResult_t hr={}; int got=0;
  cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Dd,pref,1,&hr,&got);
  if(hs!=CUBLAS_STATUS_SUCCESS||got==0){printf("NO ALGO (status=%d got=%d)\n",(int)hs,got);return 2;}
  printf("heuristic OK, algos=%d\n",got);

  auto run=[&](){ CB(cublasLtMatmul(lt,op,dAlpha,dW,Ad,dA,Bd,dBeta,dD,Cd,dD,Dd,&hr.algo,ws,wsz,0)); };
  run(); CK(cudaDeviceSynchronize());

  // correctness vs fp32 ref (use quantized weights/acts refs => isolates GEMM math)
  std::vector<__half> hD((size_t)M*N); CK(cudaMemcpy(hD.data(),dD,(size_t)M*N*2,cudaMemcpyDeviceToHost));
  double dot=0,na=0,nb=0; double maxabs=0;
  // ref: D[i,j] = sum_k Aref[i,k]*Wref[j,k]  (subset of rows: GEMM math identical across rows)
  int Mc = M<8?M:8;
  for(int i=0;i<Mc;i++) for(int j=0;j<N;j++){
    double acc=0; for(int kk=0;kk<K;kk++) acc+=(double)qA.ref[(size_t)i*K+kk]*(double)qW.ref[(size_t)j*K+kk];
    float got=__half2float(hD[(size_t)i*N+j]);
    dot+=acc*got; na+=acc*acc; nb+=(double)got*got; maxabs=fmax(maxabs,fabs(acc-got));
    if(i==0&&j<3) printf("  D[0,%d] ref=%.4f got=%.4f\n",j,acc,got);
  }
  double cos=dot/(sqrt(na)*sqrt(nb)+1e-30);
  printf("correctness: cosine=%.6f  max|err|=%.4f\n",cos,maxabs);

  // throughput
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for(int i=0;i<iters;i++) run();
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
  double flops=2.0*M*N*K;
  printf("THROUGHPUT: %.3f ms/gemm   %.1f TFLOP/s\n",ms,flops/(ms*1e-3)/1e12);
  return 0;
}
