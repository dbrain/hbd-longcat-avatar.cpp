// M4 mesh-extractor isolation test — feeds the ORACLE's exact head inputs (refs/stage5/head_*)
// to svae::flexible_dual_grid_to_mesh and compares verts/faces to the o_voxel oracle output
// (refs/stage5/oracle_{verts,faces}). Same inputs + deterministic geometry => must be BIT-EXACT.
// Runs in seconds (no decoder). Build: ./build.sh m4_mesh_only
#include "sparse_vae.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

static const char* REFS = "refs/stage5";

int main() {
    auto P = [](const std::string& k){ return std::string(REFS) + "/" + k + ".npy"; };
    NpyArray cN = npy_load(P("head_coords"));        // i32 [N,4]
    NpyArray dN = npy_load(P("head_dual_vertices")); // f32 [N,3]
    NpyArray iN = npy_load(P("head_intersected"));   // i8  [N,3]
    NpyArray qN = npy_load(P("head_quad_lerp"));     // f32 [N,1]
    int N = (int)cN.shape[0];
    printf("[m4-mesh] N=%d (head inputs from fp32 oracle)\n", N);
    std::vector<int8_t> inter(iN.i8(), iN.i8() + (size_t)N*3);

    svae::Mesh mesh = svae::flexible_dual_grid_to_mesh(cN.i32(), N, dN.f32(), inter.data(), qN.f32(), 1024);
    printf("[m4-mesh] mine: verts=%d faces=%d\n", mesh.N, mesh.F);

    NpyArray ovN = npy_load(P("oracle_verts"));      // f32 [V,3]
    NpyArray ofN = npy_load(P("oracle_faces"));      // i64 [F,3]
    int oV = (int)ovN.shape[0], oF = (int)ofN.shape[0];
    printf("[m4-mesh] oracle: verts=%d faces=%d\n", oV, oF);

    bool vsz = ((int)mesh.verts.size() == oV*3);
    double vma = 0; size_t vworst = 0;
    if (vsz) { const float* ov = ovN.f32();
        for (size_t i = 0; i < mesh.verts.size(); i++) { double d = std::fabs((double)mesh.verts[i]-ov[i]); if (d>vma){vma=d;vworst=i;} } }
    printf("[m4-mesh] verts %s maxabs=%.3e @%zu\n", vsz?"size-OK":"SIZE-MISMATCH", vma, vworst);

    bool fsz = ((int)mesh.faces.size() == oF*3);
    int64_t fdiff = 0, fworst = -1;
    if (fsz) { const int64_t* of = ofN.i64();
        for (size_t i = 0; i < mesh.faces.size(); i++) if (mesh.faces[i] != of[i]) { fdiff++; if (fworst<0) fworst=(int64_t)i; } }
    printf("[m4-mesh] faces %s elementwise-diff=%lld%s\n", fsz?"count-OK":"COUNT-MISMATCH",
           (long long)fdiff, fworst>=0 ? "" : " (or count mismatch)");
    if (fworst >= 0) printf("           first diff @%lld: mine=%lld oracle=%lld\n",
                            (long long)fworst, (long long)mesh.faces[fworst], (long long)ofN.i64()[fworst]);

    bool ok = vsz && fsz && vma < 1e-6 && fdiff == 0;
    printf("[m4-mesh] %s (mesh extractor vs o_voxel: verts maxabs=%.3e, faces-diff=%lld)\n",
           ok ? "PASS (BIT-EXACT)" : "FAIL", vma, (long long)fdiff);

    // emit the untextured mesh as a binary PLY (tangible geometry artifact)
    const char* out = "miku_geometry.ply";
    if (svae::write_ply(out, mesh.verts, mesh.faces))
        printf("[m4-mesh] wrote %s (%d verts, %d faces) — loadable in MeshLab/Blender\n", out, mesh.N, mesh.F);
    return ok ? 0 : 1;
}
