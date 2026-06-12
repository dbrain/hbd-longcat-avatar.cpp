// M4 decode PROFILER — runs the exact m4 decode loop on golden input, wrapping each op
// category with timers to get a precise per-category breakdown of the ~110s host decode.
// Build: ./build.sh m4_profile cuda    Run: SVAE_PROF=1 ./m4_profile cuda
#include "sparse_vae.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <string>

static const char* WDIR = "weights_npy/shape_dec";
static const char* GOLD = "../../sparse_spike/golden_stages";
static std::vector<float> W(const std::string& k){ NpyArray a=npy_load(std::string(WDIR)+"/"+k+".npy"); return std::vector<float>(a.f32(),a.f32()+a.numel()); }

struct T { double nmap=0, linear=0, layernorm=0, silu=0, add=0, conv=0, c2s=0, gather=0, repeat=0, subdiv=0, head=0; } t;
static inline double tic(){ return svae::now_s(); }

int main(int argc,char**argv){
    bool use_cuda = (argc>1 && std::string(argv[1])=="cuda");
    svae::prof().on = (getenv("SVAE_PROF")!=nullptr);
    printf("[prof] backend=%s prof=%d\n", use_cuda?"cuda":"cpu", (int)svae::prof().on);
    const int MC[5]={1024,512,256,128,64}; const int NB[4]={4,16,8,4}; const float EPS=1e-6f;

    NpyArray lcN=npy_load(std::string(GOLD)+"/stage3b_out/shape_slat_coords.npy");
    NpyArray lfN=npy_load(std::string(GOLD)+"/stage3b_out/shape_slat_feats.npy");
    int N=(int)lcN.shape[0];
    std::vector<int32_t> coords(lcN.i32(), lcN.i32()+(size_t)N*4);
    std::vector<float> slat(lfN.f32(), lfN.f32()+(size_t)N*32);
    printf("[prof] N=%d\n", N);

    double T0=tic(), s;
    s=tic(); std::vector<float> feats=svae::linear(slat,W("from_latent.weight"),W("from_latent.bias"),N,32,1024); t.linear+=tic()-s;
    int C=1024;
    s=tic(); std::vector<uint32_t> nmap=svae::build_nmap(coords.data(),N); t.nmap+=tic()-s;

    for(int L=0;L<4;L++){
        int Cout=MC[L+1]; char pfx[64];
        printf("[prof] level %d C=%d Cout=%d N=%d\n",L,C,Cout,N);
        for(int j=0;j<NB[L];j++){
            snprintf(pfx,sizeof(pfx),"blocks.%d.%d",L,j); std::string p(pfx);
            s=tic(); std::vector<float> h=svae::subm_conv(feats,nmap,W(p+".conv.weight"),W(p+".conv.bias"),N,C,C,use_cuda); t.conv+=tic()-s;
            s=tic(); svae::layernorm(h,W(p+".norm.weight"),W(p+".norm.bias"),N,C,EPS); t.layernorm+=tic()-s;
            s=tic(); std::vector<float> mm=svae::linear(h,W(p+".mlp.0.weight"),W(p+".mlp.0.bias"),N,C,4*C); t.linear+=tic()-s;
            s=tic(); svae::silu_inplace(mm); t.silu+=tic()-s;
            s=tic(); mm=svae::linear(mm,W(p+".mlp.2.weight"),W(p+".mlp.2.bias"),N,4*C,C); t.linear+=tic()-s;
            s=tic(); svae::add_inplace(feats,mm); t.add+=tic()-s;
        }
        snprintf(pfx,sizeof(pfx),"blocks.%d.%d",L,NB[L]); std::string p(pfx);
        s=tic(); std::vector<float> subdiv=svae::linear(feats,W(p+".to_subdiv.weight"),W(p+".to_subdiv.bias"),N,C,8); t.subdiv+=tic()-s;
        std::vector<uint8_t> sub((size_t)N*8); for(size_t i=0;i<sub.size();i++) sub[i]=(subdiv[i]>0.f)?1:0;
        std::vector<float> h1=feats;
        s=tic(); svae::layernorm(h1,W(p+".norm1.weight"),W(p+".norm1.bias"),N,C,EPS); t.layernorm+=tic()-s;
        s=tic(); svae::silu_inplace(h1); t.silu+=tic()-s;
        s=tic(); std::vector<float> conv1_out=svae::subm_conv(h1,nmap,W(p+".conv1.weight"),W(p+".conv1.bias"),N,C,Cout*8,use_cuda); t.conv+=tic()-s;
        s=tic(); svae::C2SIdx ix=svae::c2s_grow(coords.data(),N,sub.data()); t.c2s+=tic()-s;
        int M=ix.M;
        s=tic(); std::vector<float> h_grown=svae::gather_children(conv1_out,ix,Cout); t.gather+=tic()-s;
        int per=C/8;
        s=tic(); std::vector<float> skip_small=svae::gather_children(feats,ix,per); t.gather+=tic()-s;
        s=tic(); std::vector<float> skip=svae::repeat_interleave(skip_small,M,per,Cout/per); t.repeat+=tic()-s;
        std::vector<float> empty;
        s=tic(); svae::layernorm(h_grown,empty,empty,M,Cout,EPS); t.layernorm+=tic()-s;
        s=tic(); svae::silu_inplace(h_grown); t.silu+=tic()-s;
        s=tic(); std::vector<uint32_t> nmap2=svae::build_nmap(ix.coords.data(),M); t.nmap+=tic()-s;
        s=tic(); std::vector<float> conv2_out=svae::subm_conv(h_grown,nmap2,W(p+".conv2.weight"),W(p+".conv2.bias"),M,Cout,Cout,use_cuda); t.conv+=tic()-s;
        s=tic(); svae::add_inplace(conv2_out,skip); t.add+=tic()-s;
        feats=std::move(conv2_out); coords=std::move(ix.coords); nmap=std::move(nmap2); N=M; C=Cout;
    }
    std::vector<float> empty;
    s=tic(); svae::layernorm(feats,empty,empty,N,64,1e-5f); t.layernorm+=tic()-s;
    s=tic(); std::vector<float> h7=svae::linear(feats,W("output_layer.weight"),W("output_layer.bias"),N,64,7); t.head+=tic()-s;
    double total=tic()-T0;

    printf("\n=== M4 decode profile (N_final=%d, total=%.2fs) ===\n", N, total);
    auto pr=[&](const char*nm,double v){ printf("  %-12s %8.2fs  %5.1f%%\n", nm, v, 100*v/total); };
    pr("build_nmap",t.nmap); pr("conv(total)",t.conv); pr("linear",t.linear); pr("subdiv",t.subdiv);
    pr("layernorm",t.layernorm); pr("silu",t.silu); pr("add",t.add);
    pr("c2s_grow",t.c2s); pr("gather",t.gather); pr("repeat",t.repeat); pr("head",t.head);
    if(svae::prof().on){ svae::Prof&P=svae::prof();
        printf("\n  conv breakdown (%d calls): malloc=%.2fs h2d=%.2fs kernel=%.2fs d2h=%.2fs free=%.2fs\n",
            P.conv_calls,P.conv_malloc,P.conv_h2d,P.conv_kernel,P.conv_d2h,P.conv_free); }
    return 0;
}
