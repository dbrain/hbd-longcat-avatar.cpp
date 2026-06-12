// Validate the GPU-resident M3a upsample (svpg::m3a_upsample) vs the fp32 oracle hr_coords
// (set-equal). Same golden input as m3a_upsample.cpp. Build: ./build.sh m3a_gpu_test cuda
#include "svp_gpu.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <set>
#include <array>
#include <chrono>

static const char* WDIR = "weights_npy/shape_dec";
static const char* REFS = "refs/stage3a";
static const char* GOLD = "../../sparse_spike/golden_stages";
typedef std::array<int32_t,4> C4;
static std::set<C4> cset(const int32_t* c, int N){ std::set<C4> s; for(int i=0;i<N;i++) s.insert({c[i*4],c[i*4+1],c[i*4+2],c[i*4+3]}); return s; }
static double now_s(){ return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()*1e-9; }

int main(){
    NpyArray lcN=npy_load(std::string(GOLD)+"/stage2_out/lr_slat_coords.npy");
    NpyArray lfN=npy_load(std::string(GOLD)+"/stage2_out/lr_slat_feats.npy");
    int N=(int)lcN.shape[0];
    std::vector<int32_t> coords(lcN.i32(), lcN.i32()+(size_t)N*4);
    std::vector<float> lr_feats(lfN.f32(), lfN.f32()+(size_t)N*32);
    printf("[m3a_gpu] lr_slat N=%d\n", N);

    double t0=now_s();
    std::vector<int32_t> hr = svpg::m3a_upsample(coords, lr_feats, WDIR);
    double dt=now_s()-t0;
    int Nh=(int)hr.size()/4;
    printf("[m3a_gpu] upsample: %.2fs  hr_coords N=%d\n", dt, Nh);

    NpyArray oN=npy_load(std::string(REFS)+"/hr_coords_fp32.npy");
    auto so=cset(oN.i32(),(int)oN.shape[0]), sm=cset(hr.data(),Nh);
    int inter=0; for(auto&c:sm) if(so.count(c)) inter++;
    int uni=(int)sm.size()+(int)so.size()-inter;
    double iou=(double)inter/uni;
    printf("[m3a_gpu] vs FP32 ORACLE: N mine=%d oracle=%d IoU=%.6f (mine-only=%d oracle-only=%d)\n",
           Nh, (int)so.size(), iou, (int)sm.size()-inter, (int)so.size()-inter);
    bool pass = iou > 0.999;
    printf("[m3a_gpu] %s (IoU=%.6f)\n", pass?"PASS":"FAIL", iou);
    return pass?0:1;
}
