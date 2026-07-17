// voxelizer_e2e.cpp — END-TO-END validation: voxelizer -> shape_slat_encoder -> compare to
// golden_69k (the load-bearing metric). Runs the encoder on TWO inputs:
//   (A) the captured o_voxel GOLDEN voxelizer output  -> must reproduce golden_69k (validates the
//       whole data path + my encoder feats6 derivation)
//   (B) the C++ vox::mesh_to_flexible_dual_grid output -> measures how my voxelizer's dual/inter
//       approximation perturbs the final shape_slat.
// CPU only (senc uses the OpenMP subm_conv path when use_cuda=false).
//   build: ./build.sh voxelizer_e2e ; run: PIXAL3D_GGUF_DIR=/mnt/hdd/pixal3d/weights_gguf ./voxelizer_e2e
#include "voxelizer.hpp"
#include "shape_slat_encoder.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>

static double cmp_feats(const std::vector<int32_t>& coords, const std::vector<float>& feats,
                        const std::string& gdir) {
    NpyArray gc = npy_load(gdir + "/shape_slat_coords.npy");   // [K,4]
    NpyArray gf = npy_load(gdir + "/shape_slat_feats.npy");    // [K,32]
    int K = (int)gc.shape[0]; const int32_t* g = (const int32_t*)gc.raw.data();
    const float* gff = gf.f32();
    int N = (int)coords.size()/4;
    std::unordered_map<int64_t,int> gm;
    for (int i=0;i<K;i++) gm.emplace(vox::cell_key(g[i*4+1],g[i*4+2],g[i*4+3]), i);
    long matched=0; double mx=0,sm=0; long cnt=0;
    for (int i=0;i<N;i++){
        auto it=gm.find(vox::cell_key(coords[i*4+1],coords[i*4+2],coords[i*4+3]));
        if(it==gm.end()) continue; matched++;
        int gj=it->second;
        for(int c=0;c<32;c++){ double e=std::fabs((double)feats[(size_t)i*32+c]-(double)gff[(size_t)gj*32+c]); mx=std::max(mx,e); sm+=e; cnt++; }
    }
    printf("  coords %d (golden %d) matched %ld  feats maxabs %.5f meanabs %.5f\n",
           N,K,matched,mx,sm/(cnt?cnt:1));
    return mx;
}

int main(int argc, char** argv){
    std::string gvox = "/mnt/hdd/pixal3d_tex/golden_voxel";
    std::string g69  = "/mnt/hdd/pixal3d_tex/golden_69k_1024";
    std::string wdir = "/mnt/hdd/pixal3d/weights_npy/shape_enc";
    int grid = 1024;
#ifdef M3A_USE_CUDA
    const bool USE_CUDA = true;
#else
    const bool USE_CUDA = false;
#endif
    printf("[voxelizer_e2e] encoder backend = %s\n", USE_CUDA ? "cuda" : "cpu");

    // ---- (A) encoder on the GOLDEN voxelizer output ----
    NpyArray gc = npy_load(gvox+"/coords.npy");        // [M,3] i4
    NpyArray gd = npy_load(gvox+"/dual_vertices.npy"); // [M,3] f4 (global normalized)
    NpyArray gi = npy_load(gvox+"/intersected.npy");   // [M,3] i1
    int M=(int)gc.shape[0];
    const int32_t* gcp=(const int32_t*)gc.raw.data();
    const float* gdp=gd.f32();
    const int8_t* gip=(const int8_t*)gi.raw.data();
    std::vector<int32_t> A_coords((size_t)M*4);
    std::vector<float> A_feats6((size_t)M*6);
    for(int i=0;i<M;i++){
        A_coords[i*4+0]=0; A_coords[i*4+1]=gcp[i*3]; A_coords[i*4+2]=gcp[i*3+1]; A_coords[i*4+3]=gcp[i*3+2];
        for(int a=0;a<3;a++){
            // encoder vertices.feats = dual*resolution - voxel_indices ; then forward subtracts 0.5
            double off = (double)gdp[i*3+a]*grid - (double)gcp[i*3+a];
            A_feats6[(size_t)i*6+a] = (float)(off - 0.5);
            A_feats6[(size_t)i*6+3+a] = (float)((double)gip[i*3+a] - 0.5);
        }
    }
    printf("[A] encoder on GOLDEN voxelizer output (M=%d):\n",M);
    std::vector<int32_t> A_out;
    std::vector<float> A_slat = senc::shape_slat_encode(A_coords, A_feats6, wdir, USE_CUDA, &A_out);
    cmp_feats(A_out, A_slat, g69);

    // ---- (B) encoder on the C++ voxelizer output ----
    NpyArray vp = npy_load(gvox+"/verts_pre.npy"); NpyArray fc = npy_load(gvox+"/faces.npy");
    int V=(int)vp.shape[0], F=(int)fc.shape[0];
    std::vector<int32_t> faces((size_t)F*3);
    { const int32_t* p=(const int32_t*)fc.raw.data(); for(size_t i=0;i<(size_t)F*3;i++) faces[i]=p[i]; }
    int gs[3]={grid,grid,grid}; float amn[3]={-0.5f,-0.5f,-0.5f}, amx[3]={0.5f,0.5f,0.5f};
    vox::VoxelOut o = vox::mesh_to_flexible_dual_grid(vp.f32(),V,faces.data(),F,gs,amn,amx,1.0f,0.2f,1e-2f);
    std::vector<int32_t> B_coords((size_t)o.N*4);
    std::vector<float> B_feats6((size_t)o.N*6);
    for(int i=0;i<o.N;i++){
        B_coords[i*4+0]=0; B_coords[i*4+1]=o.coords[i*3]; B_coords[i*4+2]=o.coords[i*3+1]; B_coords[i*4+3]=o.coords[i*3+2];
        for(int a=0;a<3;a++){ B_feats6[(size_t)i*6+a]=(float)(o.dual[i*3+a]-0.5); B_feats6[(size_t)i*6+3+a]=(float)(o.intersected[i*3+a]-0.5); }
    }
    printf("[B] encoder on C++ voxelizer output (N=%d):\n",o.N);
    std::vector<int32_t> B_out;
    std::vector<float> B_slat = senc::shape_slat_encode(B_coords, B_feats6, wdir, USE_CUDA, &B_out);
    cmp_feats(B_out, B_slat, g69);
    (void)argc;(void)argv;
    return 0;
}
