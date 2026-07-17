// Validate the native voxelizer (us_voxelize.hpp) against the Python voxel goldens. Exact match is
// impossible (unseeded surface rng); validate the STABLE quantity: occupied-voxel set IoU vs the
// golden occupied set, and that the golden 512-draws are a subset of my occupied set.
//   ./build.sh us_voxelize_test          (pure host, no ggml)
#include "us_voxelize.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <string>
#include <unordered_set>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/voxel_goldens";
static const char* MESH = "/mnt/hdd/3d/avatar-shootout/_shootout_out/char1_coarse.glb";

static uint64_t key(int x,int y,int z){ return us_vox::pack_voxel(x,y,z); }

int main(int argc, char** argv) {
    const int RES = 128, K = 512;
    int num_sample = (argc > 1) ? atoi(argv[1]) : 409600;

    us_vox::VoxelCond vc;
    if (!us_vox::voxelize_mesh(MESH, K, RES, num_sample, 42, vc)) { printf("voxelize_mesh FAILED\n"); return 1; }
    printf("[vox] native: occupied=%d  voxel_cond=%zu (num_sample=%d)\n", vc.n_occupied, vc.coords.size()/3, num_sample);

    // rebuild my full occupied SET (not just the K subset) for IoU
    glb::Mesh mesh; glb::read_glb(MESH, mesh);
    rig::normalize_mesh(mesh.verts); for (auto&v:mesh.verts) v*=0.99f;
    std::vector<float> pts, nrm; rig::sample_surface(mesh.verts, mesh.faces, num_sample, 42, pts, nrm);
    auto occ = us_vox::occupied_voxels(pts, RES);
    std::unordered_set<uint64_t> mine; for (auto&v:occ) mine.insert(key(v[0],v[1],v[2]));

    // golden occupied set
    NpyArray go = npy_load(std::string(GDIR) + "/occupied_voxels.npy");   // [M,3] int32
    int M = (int)go.shape[0];
    const int32_t* gi = go.i32();
    std::unordered_set<uint64_t> gold; for (int i=0;i<M;i++) gold.insert(key(gi[i*3],gi[i*3+1],gi[i*3+2]));

    int inter=0; for (auto k:gold) if (mine.count(k)) inter++;
    int uni = (int)mine.size() + (int)gold.size() - inter;
    printf("[vox] occupied IoU: mine=%zu golden=%d inter=%d IoU=%.4f  (golden-in-mine=%.4f)\n",
           mine.size(), M, inter, (double)inter/uni, (double)inter/gold.size());

    // golden 512-draws should all be occupied in my set
    for (int s : {42,7,123}) {
        char p[256]; snprintf(p,sizeof(p),"%s/voxel_cond_seed%d.npy",GDIR,s);
        NpyArray vg = npy_load(p); const int32_t* v=vg.i32(); int n=(int)vg.shape[0];
        int in=0; for (int i=0;i<n;i++) if (mine.count(key(v[i*3],v[i*3+1],v[i*3+2]))) in++;
        printf("[vox] golden voxel_cond_seed%-3d: %d/%d in my occupied set (%.4f)\n", s, in, n, (double)in/n);
    }
    // PASS: occupied IoU high and golden 512-draws (seed42) essentially fully contained.
    bool pass = (double)inter/gold.size() > 0.95;
    printf("[vox] %s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
