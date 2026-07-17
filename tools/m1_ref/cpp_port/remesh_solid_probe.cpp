// remesh_solid_probe — CPU-ONLY in-grid probe for marching_cubes_solid()'s flood-fill seal.
//
// Answers the question every "is the mesh double-walled?" investigation actually needs, on the
// REAL grid the production path uses (not a 256^3 proxy of the shipped mesh): does the exterior
// flood-fill LEAK through the coarse occupancy shell, i.e. is solid/wall ~= 1.00?
//
// Loads a --dump-geo/_geocache coords1024.bin (8-byte length prefix, then int32 [N,4] = b,x,y,z on
// the grid-`res` lattice) and runs svae::marching_cubes_solid exactly as pixal3d_chain does, with
// REMESH_SEAL_VERBOSE forced on so the seal metrics + silhouette projections print.
//
//   build: /usr/bin/g++ -O2 -std=c++17 -fopenmp remesh_solid_probe.cpp -o remesh_solid_probe -lm
//   run:   REMESH_CLOSE_R=0 ./remesh_solid_probe <coords1024.bin> [res=1536] [stride=2] [blur=0]
//
// NO GPU, NO CUDA — pure host. Sweep REMESH_CLOSE_R to pick the radius that seals.
#include "remesh.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <coords1024.bin> [res=1536] [stride=2] [blur=0]\n", argv[0]);
        printf("  env: REMESH_CLOSE_R (seal radius, 0=legacy leaky), REMESH_SEAL_STRICT\n");
        return 1;
    }
    const char* path = argv[1];
    const int res    = argc > 2 ? atoi(argv[2]) : 1536;
    const int stride = argc > 3 ? atoi(argv[3]) : 2;
    const int blur   = argc > 4 ? atoi(argv[4]) : 0;
    setenv("REMESH_SEAL_VERBOSE", "1", 0);   // don't clobber an explicit setting

    FILE* f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); return 1; }
    uint64_t nbytes = 0;
    if (fread(&nbytes, sizeof(uint64_t), 1, f) != 1) { printf("FAIL: short header\n"); return 1; }
    if (nbytes % (4 * sizeof(int32_t))) { printf("FAIL: %llu bytes is not a whole [N,4] i32\n",
                                                 (unsigned long long)nbytes); return 1; }
    const int N = (int)(nbytes / (4 * sizeof(int32_t)));
    std::vector<int32_t> coords((size_t)N * 4);
    if (fread(coords.data(), 1, nbytes, f) != nbytes) { printf("FAIL: short read\n"); return 1; }
    fclose(f);

    int mx[3] = {-1,-1,-1}, mn[3] = {1<<30,1<<30,1<<30};
    for (int i = 0; i < N; i++) for (int c = 0; c < 3; c++) {
        int v = coords[(size_t)i*4 + 1 + c];
        if (v > mx[c]) mx[c] = v; if (v < mn[c]) mn[c] = v;
    }
    const char* cr = getenv("REMESH_CLOSE_R");
    printf("[probe] %s\n[probe] N=%d occupied fine voxels, extent x[%d..%d] y[%d..%d] z[%d..%d]\n",
           path, N, mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);
    printf("[probe] res=%d stride=%d -> COARSE GRID %d^3 | blur=%d | REMESH_CLOSE_R=%s\n",
           res, stride, res / stride, blur, cr ? cr : "(default)");
    if (mx[0] >= res || mx[1] >= res || mx[2] >= res)
        printf("[probe] !! WARNING: a coord >= res -- wrong --res for this cache?\n");

    double t0 = now_sec();
    svae::Mesh m = svae::marching_cubes_solid(coords.data(), N, res, stride, blur, 0.5f);
    double t1 = now_sec();
    int64_t b = 0, nm = 0;
    svae::mesh_topology_stats(m, b, nm);
    printf("[probe] MC-solid: verts=%d faces=%d boundary=%lld nonmanifold=%lld  (%.1fs)\n",
           m.N, m.F, (long long)b, (long long)nm, t1 - t0);
    printf("[probe] %s\n", (b == 0 && nm == 0) ? "(clean manifold -- NB: passes even when leaked!)"
                                               : "(NOT clean)");

    // ---- WHAT DOES THE SURFACE BOUND: a BODY, or just WALL MATERIAL?
    // Signed volume via the divergence theorem. For a closed oriented mesh this equals the
    // parity/"voxel-in-out" volume BY DEFINITION (they are the same integral), so comparing those
    // two proves nothing. The discriminating comparison is against the grid's own occupancy:
    //   solid-cell volume  = what the flood fill decided is INSIDE  (a filled body)
    //   seed-cell volume   = the shell material alone                (a double wall would match this)
    double vol = 0.0;
    const int64_t F = (int64_t)m.faces.size() / 3;
    #pragma omp parallel for reduction(+:vol) schedule(static)
    for (int64_t t = 0; t < F; t++) {
        const int64_t i0 = m.faces[t*3], i1 = m.faces[t*3+1], i2 = m.faces[t*3+2];
        const float *p0 = &m.verts[i0*3], *p1 = &m.verts[i1*3], *p2 = &m.verts[i2*3];
        vol += ((double)p0[0]*((double)p1[1]*p2[2] - (double)p1[2]*p2[1])
              - (double)p0[1]*((double)p1[0]*p2[2] - (double)p1[2]*p2[0])
              + (double)p0[2]*((double)p1[0]*p2[1] - (double)p1[1]*p2[0])) / 6.0;
    }
    const double cellv = 1.0 / ((double)(res/stride)*(res/stride)*(res/stride));
    printf("[probe] signed volume of the emitted surface = %.6f\n", vol);
    printf("[probe]   (1 coarse cell = %.4g; compare against the seal's seed/solid cell counts above:\n"
           "[probe]    ~= solid*cellv  -> the surface bounds the FILLED BODY  (single skin)\n"
           "[probe]    ~= seed*cellv   -> the surface bounds only WALL MATERIAL (DOUBLE-WALLED))\n",
           cellv);
    return 0;
}
