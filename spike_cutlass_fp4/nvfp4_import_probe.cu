// Decisive isolation: run the SAME repack->cuBLASLt FP4 GEMM on (1) self-quantized random
// weights and (2) the ACTUAL imported gguf block_nvfp4 bytes, same activation. If (1) passes
// and (2) overflows, the bug is the imported bytes (not the GEMM/shape/activation/convention).
// Also decodes the imported bytes both ways (ggml kvalues*ue4m3/2 vs std e2m1*std e4m3) to
// confirm per-element equivalence, and dumps the scale/value ranges.
//
//   nvfp4_import_probe <w.bin> <K> <N>     (w.bin = raw block_nvfp4, K=in, N=out)
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=(x);if(s){fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

static const float KV[16]={0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};       // ggml kvalues_mxfp4 (2x std)
static const float E2M1_STD[8]={0,0.5f,1,1.5f,2,3,4,6};                       // cuBLASLt nibble decode
static uint8_t enc_e2m1_std(float v){float a=fabsf(v);int b=0;float bd=1e30f;for(int i=0;i<8;i++){float d=fabsf(a-E2M1_STD[i]);if(d<bd){bd=d;b=i;}}uint8_t n=b;if(v<0)n|=0x8;return n;}
static float ue4m3_to_fp32(uint8_t x){if(x==0||x==0x7F)return 0.f;__nv_fp8_e4m3 f;*reinterpret_cast<uint8_t*>(&f)=x;return (float)f*0.5f;}   // ggml decode
static float stde4m3(uint8_t x){__nv_fp8_e4m3 f;*reinterpret_cast<uint8_t*>(&f)=x;return (float)f;}                                        // cuBLASLt decode
static size_t swz(size_t row,size_t col,uint32_t cl){const uint32_t R=128,RC=32,CC=4;size_t rb=row/R,rem=row%R,d4=rem/RC,d3=rem%RC,cbg=col/CC,d5=col%CC,cc=(cl+CC-1)/CC;return ((rb*cc+cbg)*RC+d3)*16+d4*CC+d5;}

struct W { std::vector<uint8_t> d,qs; std::vector<float> ref; };

// ggml block_nvfp4 decode -> ref[N*K]; d/qs filled in golden's per-(r,nblk) layout.
static W load_blocks(const uint8_t* raw,int N,int K){
  W w; int nblk=K/64; w.d.assign((size_t)N*nblk*4,0); w.qs.assign((size_t)N*nblk*32,0); w.ref.assign((size_t)N*K,0);
  const uint8_t* p=raw;
  for(int r=0;r<N;r++) for(int b=0;b<nblk;b++){
    uint8_t* d=&w.d[((size_t)r*nblk+b)*4]; uint8_t* qs=&w.qs[((size_t)r*nblk+b)*32];
    memcpy(d,p,4); p+=4; memcpy(qs,p,32); p+=32;
    for(int s=0;s<4;s++){ float dd=ue4m3_to_fp32(d[s]);
      for(int j=0;j<8;j++){ uint8_t qb=qs[s*8+j]; uint8_t n0=qb&0xF,n1=qb>>4;
        w.ref[(size_t)r*K + b*64 + s*16 + j]   = KV[n0]*dd;
        w.ref[(size_t)r*K + b*64 + s*16 + 8+j] = KV[n1]*dd; } }
  }
  return w;
}
static W quant_rand(int N,int K){
  std::mt19937 rng(7);std::normal_distribution<float> nd(0,1);W w;int nblk=K/64;
  w.d.assign((size_t)N*nblk*4,0);w.qs.assign((size_t)N*nblk*32,0);w.ref.assign((size_t)N*K,0);
  for(int r=0;r<N;r++)for(int b=0;b<nblk;b++)for(int s=0;s<4;s++){
    float xb[16],amax=0;for(int j=0;j<16;j++){xb[j]=nd(rng)*0.05f;amax=fmaxf(amax,fabsf(xb[j]));}
    __nv_fp8_e4m3 sf(amax/6.f);uint8_t ue=*reinterpret_cast<uint8_t*>(&sf);w.d[((size_t)r*nblk+b)*4+s]=ue;
    float dd=ue4m3_to_fp32(ue);
    for(int j=0;j<8;j++){int i0=0;float b0=1e30f;for(int q=0;q<16;q++){float e=fabsf(KV[q]*dd-xb[j]);if(e<b0){b0=e;i0=q;}}
      int i1=0;float b1=1e30f;for(int q=0;q<16;q++){float e=fabsf(KV[q]*dd-xb[8+j]);if(e<b1){b1=e;i1=q;}}
      w.qs[((size_t)r*nblk+b)*32+s*8+j]=(i0&0xF)|((i1&0xF)<<4);
      w.ref[(size_t)r*K+b*64+s*16+j]=KV[i0]*dd; w.ref[(size_t)r*K+b*64+s*16+8+j]=KV[i1]*dd;}
  }
  return w;
}
static void repack(const W&w,int N,int K,std::vector<uint8_t>&data,std::vector<uint8_t>&sc){
  int nblk=K/64,nsub=K/16; data.assign((size_t)N*K/2,0);
  size_t rb=((N+127)/128)*128,cb=((nsub+3)/4)*4; sc.assign(rb*cb,0);
  for(int r=0;r<N;r++){
    for(int ss=0;ss<nsub;ss++){int b=ss/4,s=ss%4; sc[swz(r,ss,nsub)]=w.d[((size_t)r*nblk+b)*4+s];}
    for(int e=0;e<K;e++){int b=e/64,s=(e%64)/16,loc=e%16; uint8_t qb=w.qs[((size_t)r*nblk+b)*32+s*8+(loc%8)];
      uint8_t nib=(loc<8)?(qb&0xF):(qb>>4); size_t o=(size_t)r*K+e; if(o&1)data[o/2]|=(nib<<4); else data[o/2]|=nib;}
  }
}
// activation: quant random values (magnitude `mag`) into cuBLASLt layout (std), alpha=1.
// stores the std-e4m3 scale byte verbatim (what the per-tensor cuBLASLt scale pointer reads).
struct A{std::vector<uint8_t>data,sc;std::vector<float>ref;};
static A quant_act_rand(int M,int K,float mag,unsigned seed){
  std::mt19937 rng(seed);std::normal_distribution<float> nd(0,1);
  A q;int nsub=K/16;q.data.assign((size_t)M*K/2,0);
  size_t rb=((M+127)/128)*128,cb=((nsub+3)/4)*4;q.sc.assign(rb*cb,0);q.ref.assign((size_t)M*K,0);
  std::vector<float> X((size_t)M*K); for(auto&v:X) v=nd(rng)*mag;
  for(int r=0;r<M;r++)for(int ss=0;ss<nsub;ss++){
    const float* xb=&X[(size_t)r*K+ss*16]; float amax=0; for(int j=0;j<16;j++) amax=fmaxf(amax,fabsf(xb[j]));
    __nv_fp8_e4m3 sf(amax/6.f);uint8_t ue=*reinterpret_cast<uint8_t*>(&sf);
    float scale=(float)sf;q.sc[swz(r,ss,nsub)]=ue;float enc=scale>0?1.f/scale:0.f;
    for(int j=0;j<16;j++){uint8_t n=enc_e2m1_std(xb[j]*enc);q.ref[(size_t)r*K+ss*16+j]=E2M1_STD[n&7]*((n&8)?-1:1)*scale;
      size_t o=(size_t)r*K+ss*16+j;if(o&1)q.data[o/2]|=(n<<4);else q.data[o/2]|=n;}}
  return q;}

static void run(cublasLtHandle_t lt,const char*tag,const W&w,const A&a,int M,int K,int N){
  // per-element decode equivalence check (ggml vs std), on the repacked bytes
  std::vector<uint8_t> Wd,Ws; repack(w,N,K,Wd,Ws);
  double maxd=0; int nbad=0; float smin=1e30f,smax=0,vmax=0;
  for(int r=0;r<2;r++)for(int e=0;e<K;e++){
    int ss=e/16; uint8_t sb=Ws[swz(r,ss,K/16)]; float sc=stde4m3(sb); smin=fminf(smin,sc>0?sc:smin);smax=fmaxf(smax,sc);
    size_t o=(size_t)r*K+e; uint8_t nib=(o&1)?(Wd[o/2]>>4):(Wd[o/2]&0xF); float v=E2M1_STD[nib&7]*((nib&8)?-1:1)*sc;
    vmax=fmaxf(vmax,fabsf(v)); double d=fabs(v-w.ref[(size_t)r*K+e]); if(d>maxd)maxd=d; if(d>1e-3)nbad++;
  }
  fprintf(stderr,"[%s] decode-equiv max|cublas-ggml|=%.5f badcnt(>1e-3)=%d  scale[min=%.4g max=%.4g] |val|max=%.4g\n",
          tag,maxd,nbad,smin,smax,vmax);
  // full cuBLASLt GEMM with the provided activation, compare to CPU dot
  float alpha=1,beta=0;
  uint8_t*dW,*dA,*dWs,*dAs;void*dD;
  CK(cudaMalloc(&dW,Wd.size()));CK(cudaMalloc(&dA,a.data.size()));CK(cudaMalloc(&dWs,Ws.size()));CK(cudaMalloc(&dAs,a.sc.size()));CK(cudaMalloc(&dD,(size_t)M*N*2));
  CK(cudaMemcpy(dW,Wd.data(),Wd.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dA,a.data.data(),a.data.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dWs,Ws.data(),Ws.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dAs,a.sc.data(),a.sc.size(),cudaMemcpyHostToDevice));
  void*ws;size_t wsz=32<<20;CK(cudaMalloc(&ws,wsz));
  int m=N,n=M,k=K;cublasLtMatmulDesc_t op;CB(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm)));CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm)));
  cublasOperation_t T=CUBLAS_OP_T,Nn=CUBLAS_OP_N;CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&T,sizeof(T)));CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nn,sizeof(Nn)));
  void*wsp=dWs,*asp=dAs;CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&wsp,sizeof(wsp)));CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&asp,sizeof(asp)));
  cublasDataType_t st=CUDA_R_32F;CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_SCALE_TYPE,&st,sizeof(st)));
  cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;CB(cublasLtMatrixLayoutCreate(&Ad,CUDA_R_4F_E2M1,k,m,k));CB(cublasLtMatrixLayoutCreate(&Bd,CUDA_R_4F_E2M1,k,n,k));CB(cublasLtMatrixLayoutCreate(&Cd,CUDA_R_16F,m,n,m));CB(cublasLtMatrixLayoutCreate(&Dd,CUDA_R_16F,m,n,m));
  cublasLtMatmulPreference_t pref;CB(cublasLtMatmulPreferenceCreate(&pref));CB(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)));
  cublasLtMatmulHeuristicResult_t hr={};int got=0;CB(cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Dd,pref,1,&hr,&got));if(!got){fprintf(stderr,"no algo\n");return;}
  CB(cublasLtMatmul(lt,op,&alpha,dW,Ad,dA,Bd,&beta,dD,Cd,dD,Dd,&hr.algo,ws,wsz,0));CK(cudaDeviceSynchronize());
  std::vector<__half>hD((size_t)M*N);CK(cudaMemcpy(hD.data(),dD,(size_t)M*N*2,cudaMemcpyDeviceToHost));
  double dot=0,na=0,nb=0,gmax=0;int nf=0;
  for(int i=0;i<1;i++)for(int j=0;j<N;j++){double acc=0;for(int kk=0;kk<K;kk++)acc+=(double)a.ref[(size_t)i*K+kk]*w.ref[(size_t)j*K+kk];
    float g=__half2float(hD[(size_t)i*N+j]);if(!isfinite(g))nf++;gmax=fmax(gmax,fabs(g));dot+=acc*g;na+=acc*acc;nb+=(double)g*g;}
  double cos=dot/(sqrt(na)*sqrt(nb)+1e-30);
  fprintf(stderr,"[%s] GEMM(ones) cosine=%.6f  |out|max=%.4g  nonfinite=%d\n",tag,cos,gmax,nf);
  cudaFree(dW);cudaFree(dA);cudaFree(dWs);cudaFree(dAs);cudaFree(dD);cudaFree(ws);
}

int main(int argc,char**argv){
  const char*wf=argv[1];int K=atoi(argv[2]),N=atoi(argv[3]),M=8;
  cublasLtHandle_t lt;CB(cublasLtCreate(&lt));
  std::vector<uint8_t>raw((size_t)N*(K/64)*36);FILE*f=fopen(wf,"rb");fread(raw.data(),1,raw.size(),f);fclose(f);
  W wi=load_blocks(raw.data(),N,K);
  // sweep activation magnitude on the imported weight: where does the cuBLASLt path break?
  float mags[6]={0.5f,5.f,50.f,200.f,1000.f,5000.f};
  for(int i=0;i<6;i++){char tag[64];snprintf(tag,64,"imported act_mag=%.0f",mags[i]);
    run(lt,tag,wi,quant_act_rand(M,K,mags[i],100+i),M,K,N);}
  cublasLtDestroy(lt);return 0;
}
