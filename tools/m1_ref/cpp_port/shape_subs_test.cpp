// Validate senc::build_guide_subs — the tex-decoder guide_subs rebuilt from the mesh's multi-res
// occupancy. Structural (CPU, no ggml/GPU): take the voxelizer's grid-1024 occupied coords, downsample
// to grid-64 (the tex-DiT/decoder input lattice), build guide_subs, then GROW grid-64 back up through
// the 4 levels via the same c2s_grow the decoder uses, and confirm the reconstructed grid-1024 coord
// SET equals the voxelizer's grid-1024 set (IoU 1.0). This proves the decoder, fed these subs, lands on
// exactly the mesh's lattice — independent of the encoder's z-major vs c2s_grow parent-major ordering.
//   ./build.sh shape_subs_test   (uses golden_voxel/coords.npy = the real o_voxel grid-1024 output)
#include "shape_slat_encoder.hpp"
#include "sparse_vae.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>

static std::vector<int32_t> downsample(const std::vector<int32_t>& c, int times) {
    auto fdiv2 = [](int32_t v){ return v>=0 ? (v>>1) : -(((-v)+1)>>1); };
    std::vector<int32_t> cur = c;
    for (int t=0;t<times;t++){
        std::unordered_set<int64_t> seen; std::vector<int32_t> nx;
        for (int i=0;i<(int)cur.size()/4;i++){
            int32_t b=cur[i*4],x=fdiv2(cur[i*4+1]),y=fdiv2(cur[i*4+2]),z=fdiv2(cur[i*4+3]);
            int64_t k=svae::coord_key(b,x,y,z);
            if(seen.insert(k).second){ nx.push_back(b);nx.push_back(x);nx.push_back(y);nx.push_back(z); }
        }
        cur=std::move(nx);
    }
    return cur;
}
static std::unordered_set<int64_t> keyset(const std::vector<int32_t>& c){
    std::unordered_set<int64_t> s; for(int i=0;i<(int)c.size()/4;i++) s.insert(svae::coord_key(c[i*4],c[i*4+1],c[i*4+2],c[i*4+3])); return s;
}

int main(int argc, char** argv){
    std::string gvox = argc>1?argv[1]:"/mnt/hdd/pixal3d_tex/golden_voxel";
    std::string g69  = "/mnt/hdd/pixal3d_tex/golden_69k_1024";

    NpyArray gc = npy_load(gvox + "/coords.npy");   // [N,3] int32 grid-1024
    int N = (int)gc.shape[0]; const int32_t* p = (const int32_t*)gc.raw.data();
    std::vector<int32_t> occ1024((size_t)N*4);
    for (int i=0;i<N;i++){ occ1024[i*4]=0; occ1024[i*4+1]=p[i*3]; occ1024[i*4+2]=p[i*3+1]; occ1024[i*4+3]=p[i*3+2]; }
    printf("[subs] voxelizer grid-1024 voxels: %d\n", N);

    // grid-64 lattice = the encoder output = the tex decoder input
    std::vector<int32_t> coords64 = downsample(occ1024, 4);
    printf("[subs] grid-64 voxels: %d\n", (int)coords64.size()/4);

    // cross-check grid-64 set vs golden_69k coords
    NpyArray g6 = npy_load(g69 + "/shape_slat_coords.npy");  // [K,4]
    int K=(int)g6.shape[0]; const int32_t* gp=(const int32_t*)g6.raw.data();
    std::vector<int32_t> g64((size_t)K*4); for(int i=0;i<K;i++){g64[i*4]=gp[i*4];g64[i*4+1]=gp[i*4+1];g64[i*4+2]=gp[i*4+2];g64[i*4+3]=gp[i*4+3];}
    { auto a=keyset(coords64), b=keyset(g64); long inter=0; for(auto k:a) if(b.count(k))inter++;
      printf("[subs] grid-64 vs golden_69k: mine=%zu golden=%zu inter=%ld IoU=%.4f\n",
             a.size(),b.size(),inter,(double)inter/(a.size()+b.size()-inter)); }

    // build guide_subs and grow back up
    std::vector<std::vector<uint8_t>> subs = senc::build_guide_subs(occ1024, coords64);
    std::vector<int32_t> coords = coords64;
    const char* RES[4] = {"128","256","512","1024"};
    for (int L=0;L<4;L++){
        int n=(int)coords.size()/4; long bits=0; for(auto v:subs[L]) bits+=v;
        svae::C2SIdx ix = svae::c2s_grow(coords.data(), n, subs[L].data());
        printf("[subs] L=%d grow ->grid%-4s : %d voxels (%ld subdiv bits set)\n", L, RES[L], ix.M, bits);
        coords = std::move(ix.coords);
    }
    // final grid-1024 set vs voxelizer
    auto a=keyset(coords), b=keyset(occ1024); long inter=0; for(auto k:a) if(b.count(k))inter++;
    double iou=(double)inter/(a.size()+b.size()-inter);
    printf("[subs] reconstructed grid-1024: %zu (voxelizer %zu) inter=%ld IoU=%.4f\n", a.size(),b.size(),inter,iou);
    bool pass = iou > 0.999;
    printf("[subs] %s (grid-1024 reconstruction IoU=%.4f)\n", pass?"PASS":"FAIL", iou);
    return pass?0:1;
}
