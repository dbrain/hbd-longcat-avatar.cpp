// coarse_obj — produce a SMALL MANIFOLD mesh for QuadriFlow from the cached occupancy, no GPU.
// marching_cubes_solid (coarse-grid solid MC, manifold+watertight+orient-consistent per FINDINGS-15)
// at a chosen stride → ~10^5 faces → Taubin smooth → OBJ. This is the QuadriFlow-tractable manifold
// input the R1 ladder needs (the raw marching-tet is 8M v = too big; the QEM soup is non-manifold).
//   build: ./build.sh coarse_obj
//   run:   ./coarse_obj <stride=6> <blur=2> <smooth=6> <out.obj>
#include "remesh.hpp"
#include "sparse_vae.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    int stride  = argc > 1 ? atoi(argv[1]) : 6;
    int blur    = argc > 2 ? atoi(argv[2]) : 2;
    int smooth  = argc > 3 ? atoi(argv[3]) : 6;
    const char* out = argc > 4 ? argv[4] : "/tmp/miku_coarse.obj";

    NpyArray cN = npy_load("refs/stage5/head_coords.npy");   // i32 [N,4], grid 1024
    int N = (int)cN.shape[0];
    printf("[coarse] occupancy N=%d @ grid1024, stride=%d blur=%d smooth=%d\n", N, stride, blur, smooth);

    svae::Mesh m = svae::marching_cubes_solid(cN.i32(), N, 1024, stride, blur, 0.5f);
    if (smooth > 0) svae::taubin_smooth(m, smooth);
    int64_t b, nm; svae::mesh_topology_stats(m, b, nm);
    printf("[coarse] verts=%d faces=%d  boundary=%lld nonmanifold=%lld  %s\n",
           m.N, m.F, (long long)b, (long long)nm, (b==0 && nm==0) ? "(CLEAN MANIFOLD)" : "(NOT clean)");

    FILE* o = fopen(out, "w"); if (!o) { printf("out open failed\n"); return 1; }
    for (size_t i = 0; i < m.verts.size(); i += 3)
        fprintf(o, "v %g %g %g\n", m.verts[i], m.verts[i+1], m.verts[i+2]);
    for (size_t i = 0; i < m.faces.size(); i += 3)
        fprintf(o, "f %lld %lld %lld\n", (long long)m.faces[i]+1, (long long)m.faces[i+1]+1, (long long)m.faces[i+2]+1);
    fclose(o);
    printf("[coarse] wrote %s\n", out);
    return 0;
}
