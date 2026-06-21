// Convention-proof for the Phase-1 ggml->cuBLASLt NVFP4 GEMM integration.
//
// Proves that ggml's `block_nvfp4` bytes, after a (nibble-reorder + scale-swizzle)
// repack, feed cuBLASLt's CUDA_R_4F_E2M1 / VEC16_UE4M3 GEMM at **alpha=1.0** and
// reproduce ggml's own dequantized matmul to cosine ~1.0.
//
// Why alpha=1.0: ggml stores E2M1 nibbles decoded via kvalues_mxfp4={0,1,2,3,4,6,8,12}
// (= 2x standard E2M1) and UE4M3 scales decoded via ggml_ue4m3_to_fp32 = std_e4m3/2.
// ggml value = (std_e4m3(d)/2) * (2*std_e2m1[q]) = std_e4m3(d)*std_e2m1[q] — exactly
// cuBLASLt's native interpretation. The two 2x factors cancel.
//
// WEIGHT path: ggml-quantize -> repack (this is what the real load-time hook does).
// ACTIVATION path: quantize directly into cuBLASLt layout (what the matmul-time
// activation quant kernel does). Both at alpha=1.0.
//
//   nvfp4_repack_golden [M K N ...]   (default DiT shapes)

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

static const double GATE = 0.999;

// kvalues_mxfp4 = 2x standard E2M1 (matches ggml-common.h:1116)
static const float KV[16]={0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};
// standard E2M1 magnitudes (what cuBLASLt decodes a nibble to)
static const float E2M1_STD[8]={0,0.5f,1,1.5f,2,3,4,6};

// standard E2M1 nibble encode (sign high bit), matches cuBLASLt + PoC enc
static uint8_t enc_e2m1_std(float v){
  float a=fabsf(v); int best=0; float bd=1e30f;
  for(int i=0;i<8;i++){float d=fabsf(a-E2M1_STD[i]); if(d<bd){bd=d;best=i;}}
  uint8_t n=best; if(v<0) n|=0x8; return n;
}
// ggml best_index over kvalues_mxfp4 (16 entries, signed) given decode scale d
static uint8_t best_index_mxfp4(float x, float d){
  int best=0; float bd=fabsf(KV[0]*d - x);
  for(int i=1;i<16;i++){float e=fabsf(KV[i]*d - x); if(e<bd){bd=e;best=i;}}
  return (uint8_t)best;
}
// ggml decode of a UE4M3 byte = standard e4m3 value / 2
static float ggml_ue4m3_to_fp32(uint8_t x){
  if(x==0||x==0x7F) return 0.0f;
  __nv_fp8_e4m3 f; *reinterpret_cast<uint8_t*>(&f)=x;
  return (float)f * 0.5f;
}

// SWIZZLE_32_4_4 offset (comfy float_utils.cuh) over (rows x col_length) scale grid
static size_t swz(size_t row_idx, size_t col_idx, uint32_t col_length){
  const uint32_t R=128,RC=32,CC=4;
  size_t rb=row_idx/R, rem=row_idx%R, d4=rem/RC, d3=rem%RC;
  size_t cbg=col_idx/CC, d5=col_idx%CC;
  size_t cbg_cnt=(col_length+CC-1)/CC;
  return ((rb*cbg_cnt+cbg)*RC+d3)*16 + d4*CC + d5;
}

// -------- WEIGHT: ggml block_nvfp4 layout (what's in the gguf / on device) --------
struct GgmlW { std::vector<uint8_t> d; std::vector<uint8_t> qs; std::vector<float> ref; };
static GgmlW quant_ggml(const std::vector<float>& in,int rows,int cols){
  GgmlW w; int nblk=cols/64;          // 64-elem blocks along K
  w.d.assign((size_t)rows*nblk*4,0);
  w.qs.assign((size_t)rows*nblk*32,0);
  w.ref.assign((size_t)rows*cols,0);
  for(int r=0;r<rows;r++) for(int b=0;b<nblk;b++) for(int s=0;s<4;s++){
    const float* xb = &in[(size_t)r*cols + b*64 + s*16];
    float amax=0; for(int j=0;j<16;j++) amax=fmaxf(amax,fabsf(xb[j]));
    __nv_fp8_e4m3 sf((float)(amax/6.0f)); uint8_t ue=*reinterpret_cast<uint8_t*>(&sf);
    w.d[((size_t)r*nblk+b)*4+s]=ue;
    float d = ggml_ue4m3_to_fp32(ue);   // = std_e4m3/2
    for(int j=0;j<8;j++){
      uint8_t x0=best_index_mxfp4(xb[j],   d);
      uint8_t x1=best_index_mxfp4(xb[8+j], d);
      w.qs[((size_t)r*nblk+b)*32 + s*8 + j] = x0 | (x1<<4);
      w.ref[(size_t)r*cols + b*64 + s*16 + j]   = KV[x0]*d;
      w.ref[(size_t)r*cols + b*64 + s*16 + 8+j] = KV[x1]*d;
    }
  }
  return w;
}
// repack ggml weight -> cuBLASLt (consecutive nibbles + swizzled std-e4m3 scales)
static void repack_w(const GgmlW& w,int rows,int cols,
                     std::vector<uint8_t>& data,std::vector<uint8_t>& scales){
  int nblk=cols/64, nsub=cols/16;
  data.assign((size_t)rows*cols/2,0);
  size_t rb_p=((rows+127)/128)*128, cb_p=((nsub+3)/4)*4;
  scales.assign(rb_p*cb_p,0);
  for(int r=0;r<rows;r++){
    // scales: ggml d byte for sub-index ss -> swizzled slot. ggml byte IS std-e4m3.
    for(int ss=0;ss<nsub;ss++){ int b=ss/4,s=ss%4; scales[swz(r,ss,nsub)] = w.d[((size_t)r*nblk+b)*4+s]; }
    // nibbles: ggml (j,j+8) interleave within each 16-sub -> consecutive along K
    for(int e=0;e<cols;e++){
      int b=e/64, s=(e%64)/16, local=e%16;
      uint8_t qbyte = w.qs[((size_t)r*nblk+b)*32 + s*8 + (local%8)];
      uint8_t nib = (local<8)? (qbyte&0xF) : (qbyte>>4);
      size_t oidx=(size_t)r*cols+e, byte=oidx/2;
      if(oidx&1) data[byte]|=(nib<<4); else data[byte]|=nib;
    }
  }
}

// -------- ACTIVATION: quantize directly into cuBLASLt layout (std e4m3, alpha=1) --------
struct Qa { std::vector<uint8_t> data; std::vector<uint8_t> scales; std::vector<float> ref; };
static Qa quant_act(const std::vector<float>& in,int rows,int cols){
  Qa q; int nsub=cols/16;
  q.data.assign((size_t)rows*cols/2,0);
  size_t rb_p=((rows+127)/128)*128, cb_p=((nsub+3)/4)*4;
  q.scales.assign(rb_p*cb_p,0);
  q.ref.assign((size_t)rows*cols,0);
  for(int r=0;r<rows;r++) for(int ss=0;ss<nsub;ss++){
    const float* xb=&in[(size_t)r*cols+ss*16];
    float amax=0; for(int j=0;j<16;j++) amax=fmaxf(amax,fabsf(xb[j]));
    __nv_fp8_e4m3 sf((float)(amax/6.0f)); uint8_t ue=*reinterpret_cast<uint8_t*>(&sf);
    float scale=(float)sf;                       // standard e4m3 decode (no /2)
    q.scales[swz(r,ss,nsub)]=ue;
    float enc = scale>0 ? 1.0f/scale : 0.0f;
    for(int j=0;j<16;j++){
      size_t idx=(size_t)r*cols+ss*16+j;
      uint8_t n=enc_e2m1_std(xb[j]*enc);
      q.ref[idx]=E2M1_STD[n&0x7]*((n&0x8)?-1:1)*scale;
      size_t byte=idx/2; if(idx&1) q.data[byte]|=(n<<4); else q.data[byte]|=n;
    }
  }
  return q;
}

static bool run(cublasLtHandle_t lt,int M,int K,int N){
  std::mt19937 rng(1234); std::normal_distribution<float> nd(0,1);
  std::vector<float> Wf((size_t)N*K),Af((size_t)M*K);
  for(auto&v:Wf) v=nd(rng)*0.05f;
  for(auto&v:Af) v=nd(rng)*0.5f;
  GgmlW gw=quant_ggml(Wf,N,K);
  std::vector<uint8_t> Wdata,Wsc; repack_w(gw,N,K,Wdata,Wsc);
  Qa qa=quant_act(Af,M,K);
  float alpha=1.0f;

  uint8_t *dW,*dA,*dWs,*dAs; void* dD; float *dAlpha,*dBeta;
  CK(cudaMalloc(&dW,Wdata.size())); CK(cudaMalloc(&dA,qa.data.size()));
  CK(cudaMalloc(&dWs,Wsc.size())); CK(cudaMalloc(&dAs,qa.scales.size()));
  CK(cudaMalloc(&dD,(size_t)M*N*2)); CK(cudaMalloc(&dAlpha,4)); CK(cudaMalloc(&dBeta,4));
  CK(cudaMemcpy(dW,Wdata.data(),Wdata.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dA,qa.data.data(),qa.data.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dWs,Wsc.data(),Wsc.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dAs,qa.scales.data(),qa.scales.size(),cudaMemcpyHostToDevice));
  float zero=0; CK(cudaMemcpy(dAlpha,&alpha,4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dBeta,&zero,4,cudaMemcpyHostToDevice));
  void* ws; size_t wsz=32*1024*1024; CK(cudaMalloc(&ws,wsz));

  int m=N,n=M,k=K;
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
  CB(cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Dd,pref,1,&hr,&got));
  if(!got){fprintf(stderr,"no algo\n");return false;}
  CB(cublasLtMatmul(lt,op,dAlpha,dW,Ad,dA,Bd,dBeta,dD,Cd,dD,Dd,&hr.algo,ws,wsz,0));
  CK(cudaDeviceSynchronize());

  std::vector<__half> hD((size_t)M*N); CK(cudaMemcpy(hD.data(),dD,(size_t)M*N*2,cudaMemcpyDeviceToHost));
  double dot=0,na=0,nb=0,maxabs=0; int Mc=M<8?M:8;
  for(int i=0;i<Mc;i++) for(int j=0;j<N;j++){
    double acc=0; for(int kk=0;kk<K;kk++) acc+=(double)qa.ref[(size_t)i*K+kk]*(double)gw.ref[(size_t)j*K+kk];
    float got=__half2float(hD[(size_t)i*N+j]);
    dot+=acc*got; na+=acc*acc; nb+=(double)got*got; maxabs=fmax(maxabs,fabs(acc-got));
  }
  double cos=dot/(sqrt(na)*sqrt(nb)+1e-30);
  bool pass=cos>=GATE;
  fprintf(stderr,"  M=%-5d K=%-6d N=%-6d  alpha=1.0  cosine=%.6f  max|err|=%.4f  %s\n",M,K,N,cos,maxabs,pass?"PASS":"FAIL");
  cudaFree(dW);cudaFree(dA);cudaFree(dWs);cudaFree(dAs);cudaFree(dD);cudaFree(dAlpha);cudaFree(dBeta);cudaFree(ws);
  return pass;
}

int main(int argc,char**argv){
  std::vector<int> nums; for(int i=1;i<argc;i++) nums.push_back(atoi(argv[i]));
  std::vector<std::array<int,3>> shapes;
  if(nums.size()>=3){ for(size_t i=0;i+2<nums.size();i+=3) shapes.push_back({nums[i],nums[i+1],nums[i+2]}); }
  else shapes={{512,14336,4096},{4608,4096,4096},{4608,4096,12288},{4608,14336,4096}};
  fprintf(stderr,"=== ggml->cuBLASLt NVFP4 repack convention proof (gate cosine>=%.3f, alpha=1.0) ===\n",GATE);
  cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
  bool all=true; for(auto&s:shapes) all &= run(lt,s[0],s[1],s[2]);
  cublasLtDestroy(lt);
  fprintf(stderr,"=== %s ===\n", all?"ALL PASS — convention proven, integrate with alpha=1.0":"SOME FAILED");
  return all?0:1;
}
