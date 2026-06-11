// Rung-1.5 DIAGNOSTIC: explicit im2col gather + cuBLAS tf32 GEMM.
// Goal: prove tensor cores close the gap to flex_gemm, get the speed CEILING,
// and measure the im2col VRAM cost (the reason this is a diagnostic, not the
// final design — the implicit Rung-2 kernel must fuse the gather to drop it).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -ccbin <g++> bench_cublas.cu -o bench_cublas -lcublas
// Run:   LD_LIBRARY_PATH=<toolchain>/lib ./bench_cublas golden_model
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "npy.hpp"

static int64_t ckey(int32_t b,int32_t z,int32_t y,int32_t x){auto m=[](int32_t v){return (int64_t)(v&0xFFFFF);};return (((((int64_t)b<<20)|m(z))<<20|m(y))<<20)|m(x);}
static std::vector<uint32_t> build_nmap(const int32_t* c,int N,int K){
  int cc=(K-1)/2,V=K*K*K; std::unordered_map<int64_t,int> idx; idx.reserve(N*2);
  for(int i=0;i<N;i++) idx[ckey(c[i*4],c[i*4+1],c[i*4+2],c[i*4+3])]=i;
  std::vector<uint32_t> nm((size_t)N*V,0xFFFFFFFFu);
  for(int i=0;i<N;i++){int32_t b=c[i*4],z=c[i*4+1],y=c[i*4+2],x=c[i*4+3];int v=0;
    for(int kz=0;kz<K;kz++)for(int ky=0;ky<K;ky++)for(int kx=0;kx<K;kx++,v++){auto it=idx.find(ckey(b,z+kz-cc,y+ky-cc,x+kx-cc));if(it!=idx.end())nm[(size_t)i*V+v]=it->second;}}
  return nm;
}
__global__ void im2col_gather(const float* feats,const uint32_t* nmap,float* im,int N,int Cin,int V){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x, tot=(size_t)N*V*Cin; if(idx>=tot)return;
  int ci=idx%Cin; int v=(idx/Cin)%V; size_t n=idx/((size_t)V*Cin);
  uint32_t j=nmap[n*V+v];
  im[n*((size_t)V*Cin)+(size_t)v*Cin+ci]=(j==0xFFFFFFFFu)?0.0f:feats[(size_t)j*Cin+ci];
}
__global__ void bias_add(float* out,const float* bias,size_t NCo,int Cout){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=NCo)return; out[idx]+=bias[idx%Cout];
}
static double baseline_min_ms(const std::string& root,int Cin,int Cout,int N){
  std::ifstream f(root+"/flexgemm_timing.json"); if(!f)return -1;
  std::string s((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
  std::string key="subm_K333_Ci"+std::to_string(Cin)+"_Co"+std::to_string(Cout)+"_N"+std::to_string(N);
  auto p=s.find('"'+key+'"'); if(p==std::string::npos)return -1;
  p=s.find("\"min_ms\"",p); if(p==std::string::npos)return -1; p=s.find(':',p)+1; return std::atof(s.c_str()+p);
}

int main(int argc,char** argv){
  std::string root=argc>1?argv[1]:"golden_model";
  cublasHandle_t h; cublasCreate(&h); cublasSetMathMode(h,CUBLAS_TF32_TENSOR_OP_MATH);
  DIR* d=opendir(root.c_str()); if(!d){printf("no %s\n",root.c_str());return 2;}
  std::vector<std::string> cases;
  for(dirent* e;(e=readdir(d));){std::string n=e->d_name; if(n[0]=='.')continue; if(std::ifstream(root+"/"+n+"/out_feats.npy"))cases.push_back(n);}
  closedir(d); std::sort(cases.begin(),cases.end());
  printf("bench_cublas (im2col + cuBLAS tf32): %zu cases\n",cases.size());
  for(auto& c:cases){
    std::string dir=root+"/"+c;
    auto coords=npy_load(dir+"/in_coords.npy"); auto feats=npy_load(dir+"/in_feats.npy");
    auto weight=npy_load(dir+"/weight.npy"); auto bias=npy_load(dir+"/bias.npy"); auto gold=npy_load(dir+"/out_feats.npy");
    int N=(int)coords.shape[0],Cin=(int)feats.shape[1],Cout=(int)weight.shape[2],V=(int)weight.shape[0],K=(V==27?3:(V==1?1:(int)(std::cbrt((double)V)+0.5)));
    if((int)gold.shape[0]!=N){printf("  %-28s SKIP non-subm\n",c.c_str());continue;}
    auto nm=build_nmap(coords.i32(),N,K);
    int Kdim=V*Cin;
    float *df,*dw,*db,*dim,*dout; uint32_t* dn;
    cudaMalloc(&df,(size_t)N*Cin*4); cudaMalloc(&dn,(size_t)N*V*4); cudaMalloc(&dw,(size_t)Kdim*Cout*4);
    cudaMalloc(&db,(size_t)Cout*4); cudaMalloc(&dim,(size_t)N*Kdim*4); cudaMalloc(&dout,(size_t)N*Cout*4);
    double im2col_gb=(double)N*Kdim*4/1e9;
    cudaMemcpy(df,feats.f32(),(size_t)N*Cin*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dn,nm.data(),(size_t)N*V*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dw,weight.f32(),(size_t)Kdim*Cout*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,bias.f32(),(size_t)Cout*4,cudaMemcpyHostToDevice);
    float alpha=1.f,beta=0.f;
    auto run=[&](){
      size_t tot=(size_t)N*V*Cin; im2col_gather<<<(tot+255)/256,256>>>(df,dn,dim,N,Cin,V);
      // row-major C[N,Cout]=im[N,Kdim]*W[Kdim,Cout]  -> col-major trick
      cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,Cout,N,Kdim,&alpha,dw,CUDA_R_32F,Cout,dim,CUDA_R_32F,Kdim,&beta,dout,CUDA_R_32F,Cout,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT);
      bias_add<<<((size_t)N*Cout+255)/256,256>>>(dout,db,(size_t)N*Cout,Cout);
    };
    for(int i=0;i<3;i++)run(); cudaDeviceSynchronize();
    cudaEvent_t a,b2; cudaEventCreate(&a); cudaEventCreate(&b2); int it=30; cudaEventRecord(a);
    for(int i=0;i<it;i++)run(); cudaEventRecord(b2); cudaEventSynchronize(b2);
    float t; cudaEventElapsedTime(&t,a,b2); double ms=t/it;
    std::vector<float> out((size_t)N*Cout); cudaMemcpy(out.data(),dout,(size_t)N*Cout*4,cudaMemcpyDeviceToHost);
    double maxabs=0,gmax=0; for(size_t i=0;i<out.size();i++){maxabs=std::max(maxabs,(double)std::fabs(out[i]-gold.f32()[i]));gmax=std::max(gmax,(double)std::fabs(gold.f32()[i]));}
    double maxrel=gmax>0?maxabs/gmax:maxabs; double base=baseline_min_ms(root,Cin,Cout,N);
    printf("  %-28s N=%-7d Ci=%-4d Co=%-4d maxrel=%.2e  ours=%.3fms flexgemm=%.3fms (%.2fx)  im2col=%.2fGB\n",
           c.c_str(),N,Cin,Cout,maxrel,ms,base,base>0?base/ms:0.0,im2col_gb);
    cudaFree(df);cudaFree(dn);cudaFree(dw);cudaFree(db);cudaFree(dim);cudaFree(dout);
  }
  cublasDestroy(h); return 0;
}
