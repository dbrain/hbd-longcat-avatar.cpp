// Validate the GPU-resident M4 decode (svpg::m4_decode_mesh) vs the fp32 oracle + the
// host path. Same golden input + checks as m4_mesh.cpp; reports verts maxabs / faces delta
// / IoU, plus wall time (the perf win). Build: ./build.sh m4_gpu_test cuda
#include "svp_gpu.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>

static const char* WDIR = "weights_npy/shape_dec";
static const char* REFS = "refs/stage5";
static const char* GOLD = "../../sparse_spike/golden_stages";
static std::vector<float> R(const std::string& k){ NpyArray a=npy_load(std::string(REFS)+"/"+k+".npy"); return std::vector<float>(a.f32(),a.f32()+a.numel()); }
static double now_s(){ return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()*1e-9; }

int main(){
    NpyArray lcN=npy_load(std::string(GOLD)+"/stage3b_out/shape_slat_coords.npy");
    NpyArray lfN=npy_load(std::string(GOLD)+"/stage3b_out/shape_slat_feats.npy");
    int N=(int)lcN.shape[0];
    std::vector<int32_t> coords(lcN.i32(), lcN.i32()+(size_t)N*4);
    std::vector<float> slat(lfN.f32(), lfN.f32()+(size_t)N*32);
    printf("[m4_gpu] HR shape_slat N=%d C=32\n", N);

    double t0=now_s();
    svae::Mesh mesh = svpg::m4_decode_mesh(coords, slat, WDIR);
    double dt=now_s()-t0;
    printf("[m4_gpu] decode+mesh: %.2fs   verts=%d faces=%d\n", dt, mesh.N, mesh.F);

    NpyArray ovN=npy_load(std::string(REFS)+"/oracle_verts.npy");
    NpyArray ofN=npy_load(std::string(REFS)+"/oracle_faces.npy");
    int oV=(int)ovN.shape[0], oF=(int)ofN.shape[0];
    printf("[m4_gpu] oracle: verts=%d faces=%d\n", oV, oF);

    bool vok = ((int)mesh.verts.size()==oV*3);
    double vma=0; if(vok){ const float* ov=ovN.f32(); for(size_t i=0;i<mesh.verts.size();i++) vma=std::max(vma,(double)std::fabs(mesh.verts[i]-ov[i])); }
    printf("[m4_gpu] verts %s maxabs=%.3e\n", vok?"size-OK":"SIZE-MISMATCH", vma);

    bool fok=((int)mesh.faces.size()==oF*3);
    int64_t fdiff=0; if(fok){ const int64_t* of=(const int64_t*)ofN.raw.data(); for(size_t i=0;i<mesh.faces.size();i++) if(mesh.faces[i]!=of[i]) fdiff++; }
    double ffrac=(double)std::llabs((long long)mesh.F-oF)/oF;
    printf("[m4_gpu] faces mine=%d oracle=%d delta=%.4f%% (elem-diff=%lld)\n", mesh.F, oF, 100.0*ffrac, (long long)fdiff);

    bool verts_ok = vok && vma < 1e-4;
    bool faces_close = ffrac < 1e-3;
    bool pass = verts_ok && faces_close;
    printf("\n[m4_gpu] %s  (verts maxabs=%.3e <1e-4=%d ; faces delta %.4f%% <0.1%%=%d)\n",
           pass?"PASS":"FAIL", vma, verts_ok, 100.0*ffrac, faces_close);
    return pass?0:1;
}
