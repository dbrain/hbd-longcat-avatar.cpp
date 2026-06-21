// Tier-1 op-golden runner for the NVFP4 FP4 GEMM kernel (Phase 0 of the fast-FP4
// kernel work). Generalizes poc_fp4_gemm.cu into a multi-shape harness that emits
// machine-readable JSON {kernel, shape, M,K,N, cosine, tflops, us, pass} so the
// bench harness can gate regressions.
//
//   kernel_golden                         # run the default DiT-representative set
//   kernel_golden M K N [M K N ...]       # run explicit shapes
//   kernel_golden --iters 300 ...         # override timing iters (default 200)
//
// GATE: cosine >= 0.999 vs an fp32 matmul of the SAME dequantized weights/acts
// (isolates GEMM math from quantization error). Exit code 0 iff every shape
// passes; non-zero otherwise. Human-readable trace -> stderr; JSON -> stdout.
//
// Layout follows comfy-kitchen cublas_gemm_nvfp4.cu / poc_fp4_gemm.cu:
// A=weight (N x K), B=activation (M x K), both E2M1 packed, per-16 UE4M3 block
// scales in SWIZZLE_32_4_4 layout, alpha = global_w * global_a.

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <random>

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{ cublasStatus_t s=(x); if(s){fprintf(stderr,"cuBLAS err %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

static const float FP4_MAX = 6.0f;       // E2M1 max
static const float FP8E4M3_MAX = 448.0f;
static const double GATE = 0.999;        // cosine gate per HANDOFF

static uint8_t enc_e2m1(float v){
  static const float lv[8]={0,0.5f,1,1.5f,2,3,4,6};
  float a=fabsf(v); int best=0; float bd=1e30f;
  for(int i=0;i<8;i++){float d=fabsf(a-lv[i]); if(d<bd){bd=d;best=i;}}
  uint8_t n=best; if(v<0) n|=0x8; return n;
}
static float dec_e2m1(uint8_t n){
  static const float lv[8]={0,0.5f,1,1.5f,2,3,4,6};
  float a=lv[n&0x7]; return (n&0x8)?-a:a;
}

// SWIZZLE_32_4_4 offset (comfy float_utils.cuh scale_factor_swizzled_offset)
static size_t swz(size_t row_idx, size_t col_idx, uint32_t col_length){
  const uint32_t R=128,RC=32,CC=4;
  size_t rb=row_idx/R, rem=row_idx%R, d4=rem/RC, d3=rem%RC;
  size_t cbg=col_idx/CC, d5=col_idx%CC;
  size_t cbg_cnt=(col_length+CC-1)/CC;
  return ((rb*cbg_cnt+cbg)*RC+d3)*16 + d4*CC + d5;
}

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

struct Res { int M,K,N; double cosine; double maxabs; double tflops; double us; bool pass; };

static Res run_shape(cublasLtHandle_t lt, int M, int K, int N, int iters){
  std::mt19937 rng(1234); std::normal_distribution<float> nd(0,1);
  std::vector<float> Wf((size_t)N*K), Af((size_t)M*K);
  for(auto&v:Wf) v=nd(rng)*0.05f;
  for(auto&v:Af) v=nd(rng)*0.5f;

  Q qW=quantize(Wf,N,K);
  Q qA=quantize(Af,M,K);
  float alpha = qW.global*qA.global;

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
  CB(cublasLtMatrixLayoutCreate(&Ad,CUDA_R_4F_E2M1,k,m,k));
  CB(cublasLtMatrixLayoutCreate(&Bd,CUDA_R_4F_E2M1,k,n,k));
  CB(cublasLtMatrixLayoutCreate(&Cd,CUDA_R_16F,m,n,m));
  CB(cublasLtMatrixLayoutCreate(&Dd,CUDA_R_16F,m,n,m));

  cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
  CB(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)));
  cublasLtMatmulHeuristicResult_t hr={}; int got=0;
  cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Dd,pref,1,&hr,&got);
  if(hs!=CUBLAS_STATUS_SUCCESS||got==0){fprintf(stderr,"NO ALGO M=%d K=%d N=%d (status=%d got=%d)\n",M,K,N,(int)hs,got);exit(2);}

  auto runk=[&](){ CB(cublasLtMatmul(lt,op,dAlpha,dW,Ad,dA,Bd,dBeta,dD,Cd,dD,Dd,&hr.algo,ws,wsz,0)); };
  runk(); CK(cudaDeviceSynchronize());

  std::vector<__half> hD((size_t)M*N); CK(cudaMemcpy(hD.data(),dD,(size_t)M*N*2,cudaMemcpyDeviceToHost));
  double dot=0,na=0,nb=0,maxabs=0;
  int Mc = M<8?M:8;  // each output is an independent dot-product; 8 rows x N = plenty of samples
  for(int i=0;i<Mc;i++) for(int j=0;j<N;j++){
    double acc=0; for(int kk=0;kk<K;kk++) acc+=(double)qA.ref[(size_t)i*K+kk]*(double)qW.ref[(size_t)j*K+kk];
    float got=__half2float(hD[(size_t)i*N+j]);
    dot+=acc*got; na+=acc*acc; nb+=(double)got*got; maxabs=fmax(maxabs,fabs(acc-got));
  }
  double cos=dot/(sqrt(na)*sqrt(nb)+1e-30);

  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for(int i=0;i<iters;i++) runk();
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
  double flops=2.0*M*N*K;

  CB(cublasLtMatmulPreferenceDestroy(pref));
  CB(cublasLtMatrixLayoutDestroy(Ad)); CB(cublasLtMatrixLayoutDestroy(Bd));
  CB(cublasLtMatrixLayoutDestroy(Cd)); CB(cublasLtMatrixLayoutDestroy(Dd));
  CB(cublasLtMatmulDescDestroy(op));
  cudaFree(dW);cudaFree(dA);cudaFree(dWs);cudaFree(dAs);cudaFree(dD);cudaFree(dAlpha);cudaFree(dBeta);cudaFree(ws);

  Res r; r.M=M;r.K=K;r.N=N;r.cosine=cos;r.maxabs=maxabs;
  r.tflops=flops/(ms*1e-3)/1e12; r.us=ms*1e3; r.pass=(cos>=GATE);
  fprintf(stderr,"  M=%-5d K=%-6d N=%-6d  cosine=%.6f max|err|=%.4f  %.1f TFLOP/s  %.1f us  %s\n",
          M,K,N,cos,maxabs,r.tflops,r.us, r.pass?"PASS":"FAIL");
  return r;
}

int main(int argc,char**argv){
  int iters=200;
  std::vector<int> nums;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--iters") && i+1<argc){ iters=atoi(argv[++i]); continue; }
    nums.push_back(atoi(argv[i]));
  }
  // DiT-representative GEMM shapes for FLUX.2-Klein-9b @ 1024^2 (M = ~4608 tokens,
  // K/N from the cited ncu breakdown: K14336 N4096 + qkv/proj/ffn families).
  std::vector<std::array<int,3>> shapes;
  if(nums.size()>=3){
    for(size_t i=0;i+2<nums.size();i+=3) shapes.push_back({nums[i],nums[i+1],nums[i+2]});
  } else {
    shapes = {
      {512, 14336, 4096},   // cited reference shape
      {4608, 4096, 4096},   // attn proj
      {4608, 4096, 12288},  // qkv
      {4608, 4096, 14336},  // ffn up (gate+up family)
      {4608, 14336, 4096},  // ffn down
      {4608, 12288, 4096},  // attn out
    };
  }
  fprintf(stderr,"=== Tier-1 FP4 GEMM goldens (gate cosine>=%.3f) ===\n",GATE);
  cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
  std::vector<Res> results;
  bool all_pass=true;
  for(auto&s:shapes){ Res r=run_shape(lt,s[0],s[1],s[2],iters); results.push_back(r); all_pass &= r.pass; }
  cublasLtDestroy(lt);

  // JSON to stdout
  printf("[\n");
  for(size_t i=0;i<results.size();i++){
    Res&r=results[i];
    printf("  {\"kernel\":\"cublaslt_nvfp4\",\"M\":%d,\"K\":%d,\"N\":%d,\"cosine\":%.6f,\"maxabs\":%.6f,\"tflops\":%.2f,\"us\":%.2f,\"pass\":%s}%s\n",
           r.M,r.K,r.N,r.cosine,r.maxabs,r.tflops,r.us, r.pass?"true":"false", i+1<results.size()?",":"");
  }
  printf("]\n");
  fprintf(stderr,"=== %s ===\n", all_pass?"ALL PASS":"SOME FAILED");
  return all_pass?0:1;
}
