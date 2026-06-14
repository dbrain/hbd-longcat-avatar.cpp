// ply_decimate_obj — read a binary-LE PLY (float xyz verts + uchar-count int tri faces), meshopt-simplify
// (non-sloppy, manifold-preserving) to a target triangle count, write OBJ for QuadriFlow. R1 plumbing:
// the marching-tet manifold mesh (~8M tris) is too big for QF; decimate to ~200k keeping manifoldness.
//   build: g++ -O2 -std=c++17 ply_decimate_obj.cpp ../../../thirdparty/meshoptimizer/{simplifier,...}.cpp
//   run:   ./ply_decimate_obj in.ply out.obj <target_tris>
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: ply_decimate_obj in.ply out.obj <target_tris>\n"); return 1; }
    size_t target_tris = (size_t)atol(argv[3]);
    FILE* f = fopen(argv[1], "rb"); if (!f) { printf("open failed\n"); return 1; }
    // --- parse ascii header ---
    std::string hdr; int c; size_t nv = 0, nf = 0;
    while ((c = fgetc(f)) != EOF) {
        hdr.push_back((char)c);
        if (hdr.size() >= 11 && hdr.compare(hdr.size()-11, 11, "end_header\n") == 0) break;
    }
    { size_t p = hdr.find("element vertex"); if (p!=std::string::npos) nv = strtoul(hdr.c_str()+p+14, nullptr, 10);
      size_t q = hdr.find("element face");   if (q!=std::string::npos) nf = strtoul(hdr.c_str()+q+12, nullptr, 10); }
    if (hdr.find("binary_little_endian") == std::string::npos) { printf("not binary-LE PLY\n"); return 1; }
    printf("[ply] %zu verts, %zu faces\n", nv, nf);

    std::vector<float> verts(nv * 3);
    if (fread(verts.data(), 4, nv*3, f) != nv*3) { printf("vert read short\n"); return 1; }
    std::vector<uint32_t> idx; idx.reserve(nf * 3);
    for (size_t i = 0; i < nf; i++) {
        uint8_t cnt; if (fread(&cnt,1,1,f)!=1){printf("face read short\n");return 1;}
        int32_t v[8]; if (fread(v,4,cnt,f)!=cnt){printf("face idx short\n");return 1;}
        for (int k = 1; k+1 < cnt; k++) { idx.push_back(v[0]); idx.push_back(v[k]); idx.push_back(v[k+1]); }
    }
    fclose(f);
    printf("[ply] %zu triangles loaded\n", idx.size()/3);

    // manifold-preserving simplify: LockBorder (watertight → no border anyway), NOT sloppy.
    std::vector<uint32_t> out(idx.size());
    float err = 0;
    float target_error = argc > 4 ? (float)atof(argv[4]) : 1e-2f;  // raise (e.g. 1.0) to force past the
    // voxel-lattice quality-stall (FINDINGS-15); QF re-meshes anyway so intermediate error is fine.
    size_t n = meshopt_simplify(out.data(), idx.data(), idx.size(), verts.data(), nv, 12,
                                target_tris*3, target_error, meshopt_SimplifyLockBorder, &err);
    out.resize(n);
    printf("[simplify] %zu -> %zu tris (err=%.4f)\n", idx.size()/3, n/3, err);

    // weld unused verts out (QuadriFlow + smaller OBJ). Remap to used verts only.
    std::vector<uint32_t> remap(nv, UINT32_MAX);
    std::vector<float> nverts; nverts.reserve(nv*3/4);
    for (uint32_t& vi : out) {
        if (remap[vi] == UINT32_MAX) { remap[vi] = (uint32_t)(nverts.size()/3);
            nverts.push_back(verts[vi*3]); nverts.push_back(verts[vi*3+1]); nverts.push_back(verts[vi*3+2]); }
        vi = remap[vi];
    }
    FILE* o = fopen(argv[2], "w"); if (!o) { printf("out open failed\n"); return 1; }
    for (size_t i = 0; i < nverts.size(); i += 3) fprintf(o, "v %g %g %g\n", nverts[i], nverts[i+1], nverts[i+2]);
    for (size_t i = 0; i < out.size(); i += 3) fprintf(o, "f %u %u %u\n", out[i]+1, out[i+1]+1, out[i+2]+1);
    fclose(o);
    printf("[obj] wrote %s: %zu verts, %zu tris\n", argv[2], nverts.size()/3, out.size()/3);
    return 0;
}
