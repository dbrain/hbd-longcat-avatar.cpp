// Validate the GPU-resident M6 tex decode (svpg::m6_tex_decode) vs the fp32 oracle.
// Same golden inputs/checks as m6_tex_decode_test.cpp. Build: ./build.sh m6_gpu_test cuda
#include "svp_gpu.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <set>
#include <array>
#include <chrono>

static const char* WDIR = "weights_npy/tex_dec";
static const char* GOLD = "../../sparse_spike/golden_stages";
static const char* REFS = "refs/stage4";
typedef std::array<int32_t,4> C4;
static std::set<C4> cset(const int32_t* c, int N){ std::set<C4> s; for(int i=0;i<N;i++) s.insert({c[i*4],c[i*4+1],c[i*4+2],c[i*4+3]}); return s; }
static double now_s(){ return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()*1e-9; }

int main(){
    NpyArray tcN = npy_load(std::string(GOLD)+"/stage4_out/tex_slat_coords.npy");
    NpyArray tfN = npy_load(std::string(GOLD)+"/stage4_out/tex_slat_feats.npy");
    int M=(int)tcN.shape[0];
    std::vector<int32_t> coords(tcN.i32(), tcN.i32()+(size_t)M*4);
    std::vector<float> tex_slat(tfN.f32(), tfN.f32()+(size_t)M*32);
    printf("[m6_gpu] tex_slat M=%d\n", M);

    std::vector<std::vector<uint8_t>> guide(4);
    for (int L=0;L<4;L++){
        NpyArray sf = npy_load(std::string(GOLD)+"/stage5_mesh/sub"+std::to_string(L)+"_feats.npy");
        guide[L].resize(sf.numel());
        for (int64_t i=0;i<sf.numel();i++) guide[L][i] = (sf.f32()[i] > 0.f) ? 1 : 0;
    }

    std::vector<int32_t> out_coords;
    double t0=now_s();
    std::vector<float> pbr = svpg::m6_tex_decode(coords, tex_slat, guide, WDIR, &out_coords);
    double dt=now_s()-t0;
    int N=(int)pbr.size()/6;
    printf("[m6_gpu] decode: %.2fs   PBR voxels N=%d\n", dt, N);

    NpyArray opN = npy_load(std::string(REFS)+"/tex_pbr.npy");
    NpyArray ocN = npy_load(std::string(REFS)+"/tex_out_coords.npy");
    int oN=(int)opN.shape[0];
    auto sm=cset(out_coords.data(),N), so=cset(ocN.i32(),(int)ocN.shape[0]);
    int inter=0; for(auto&c:sm) if(so.count(c)) inter++;
    printf("[m6_gpu] out_coords: mine=%d oracle=%d IoU=%.6f\n", N, oN, (double)inter/(sm.size()+so.size()-inter));
    bool ok=false;
    if (N==oN){
        const float* o=opN.f32(); double ma=0,sum=0; size_t worst=0;
        for (size_t i=0;i<pbr.size();i++){ double d=std::fabs((double)pbr[i]-o[i]); if(d>ma){ma=d;worst=i;} sum+=d; }
        printf("[m6_gpu] PBR vs oracle: maxabs=%.3e meanabs=%.3e worst@%zu got=%.5f ref=%.5f\n",
               ma, sum/pbr.size(), worst, pbr[worst], o[worst]);
        ok = ma < 5e-2;
    } else printf("[m6_gpu] COUNT MISMATCH\n");
    printf("[m6_gpu] %s\n", ok?"PASS":"FAIL/CHECK");
    return ok?0:1;
}
