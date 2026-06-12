// Standalone validation of the Marching-Tetrahedra remesh (A1). Loads the saved Miku grid-1024
// occupancy (refs/stage5/head_coords.npy, the M4 decoder's ~1.5M voxels) and the raw dual-grid
// extractor output, then checks the remesh is MANIFOLD + WATERTIGHT (boundary==0, nonmanifold==0)
// where the raw dual-grid mesh is not. Writes miku_remesh.ply for an eyeball. Runs in seconds (no
// decoder). Build: ./build.sh remesh_test
#include "remesh.hpp"
#include "sparse_vae.hpp"
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"
#include "../../../thirdparty/xatlas.h"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

static const char* REFS = "refs/stage5";

// decimate (meshopt quality collapse, sloppy fallback only if quality stalls) + report which path,
// then xatlas-unwrap the decimated mesh and report the chart count + atlas utilization. This is the
// downstream payoff of a manifold mesh: quality collapse hits target + charts collapse to tens.
static void decimate_and_unwrap(const char* tag, const std::vector<float>& v,
                                const std::vector<int64_t>& f, size_t target_faces) {
    const size_t Vin=v.size()/3;
    std::vector<uint32_t> idx(f.size()); for (size_t i=0;i<idx.size();i++) idx[i]=(uint32_t)f[i];
    float dec_err = std::getenv("DEC_ERR") ? atof(std::getenv("DEC_ERR")) : 1.0f;
    float smult   = std::getenv("DEC_SMULT") ? atof(std::getenv("DEC_SMULT")) : 1.25f;
    // Quality quadric collapse. Regularize evens triangle shapes -> coherent normals -> chart
    // collapse. (Prune is NOT used: as a flag it dissolves handles bounded by target_error, which
    // at the error needed for aggressive collapse nukes the whole mesh.)
    unsigned opts = 0;
    if (std::getenv("DEC_REG")) opts |= meshopt_SimplifyRegularize;
    if (std::getenv("DEC_PRUNE")) opts |= meshopt_SimplifyPrune;
    if (std::getenv("DEC_PERM")) opts |= meshopt_SimplifyPermissive;
    std::vector<uint32_t> dst(idx.size()); float err=0.f;
    size_t nic = meshopt_simplify(dst.data(), idx.data(), idx.size(), v.data(), Vin, 3*sizeof(float),
                                  target_faces*3, dec_err, opts, &err);
    size_t quality_ic = nic;   // quality result BEFORE any sloppy fallback
    bool sloppy=false;
    if (nic > (size_t)(target_faces*3*smult)) {
        float serr=0.f;
        size_t sic = meshopt_simplifySloppy(dst.data(), idx.data(), idx.size(), v.data(), Vin, 3*sizeof(float),
                                            target_faces*3, 1.0f, &serr);
        if (sic>0){ nic=sic; err=serr; sloppy=true; }
    }
    dst.resize(nic);
    printf("[remesh] %s quality->%zu faces (target %zu, err=%.4f)%s\n", tag, quality_ic/3, target_faces, err,
           sloppy?" [would SLOPPY]":" [QUALITY OK]");
    if (std::getenv("DEC_NOUNWRAP")) return;   // fast diagnostic: skip the slow xatlas unwrap
    // compact verts
    std::vector<int> remap(Vin,-1); std::vector<float> vo; std::vector<uint32_t> fo; fo.reserve(nic);
    for (uint32_t oi:dst){ if(remap[oi]<0){ remap[oi]=(int)(vo.size()/3);
        vo.push_back(v[(size_t)oi*3]); vo.push_back(v[(size_t)oi*3+1]); vo.push_back(v[(size_t)oi*3+2]); } fo.push_back(remap[oi]); }
    // POST_SMOOTH: Taubin-smooth the LOW-POLY decimated mesh (cheap — only ~target verts) so the
    // sloppy-decimated facet normals become coherent -> xatlas charts collapse + render isn't blobby.
    int ps = std::getenv("POST_SMOOTH") ? atoi(std::getenv("POST_SMOOTH")) : 0;
    if (ps > 0) {
        svae::Mesh dm; dm.verts = vo; dm.faces.assign(fo.begin(), fo.end()); dm.N=(int)(vo.size()/3); dm.F=(int)(fo.size()/3);
        svae::taubin_smooth(dm, ps);
        vo = dm.verts; printf("[remesh] %s post-decimate Taubin %d on %zu f\n", tag, ps, fo.size()/3);
    }
    printf("[remesh] %s decimate%s: quality->%zu, final %zu->%zu faces (target %zu, err=%.4f)\n",
           tag, sloppy?" (SLOPPY fallback)":" (QUALITY)", quality_ic/3, f.size()/3, fo.size()/3, target_faces, err);

    xatlas::Atlas* atlas = xatlas::Create();
    xatlas::MeshDecl md; md.vertexCount=(uint32_t)(vo.size()/3); md.vertexPositionData=vo.data();
    md.vertexPositionStride=3*sizeof(float); md.indexCount=(uint32_t)fo.size();
    md.indexData=fo.data(); md.indexFormat=xatlas::IndexFormat::UInt32;
    xatlas::AddMesh(atlas, md); xatlas::AddMeshJoin(atlas);
    xatlas::ChartOptions co;
    co.maxCost              = std::getenv("ATL_MAXCOST")  ? atof(std::getenv("ATL_MAXCOST"))  : 16.0f;
    co.maxIterations        = std::getenv("ATL_ITERS")    ? (uint32_t)atoi(std::getenv("ATL_ITERS")) : 1;
    co.normalDeviationWeight= std::getenv("ATL_NORMW")    ? atof(std::getenv("ATL_NORMW"))    : 2.0f;
    co.straightnessWeight   = std::getenv("ATL_STRAIGHTW")? atof(std::getenv("ATL_STRAIGHTW")): 6.0f;
    co.roundnessWeight      = std::getenv("ATL_ROUNDW")   ? atof(std::getenv("ATL_ROUNDW"))   : 0.01f;
    co.normalSeamWeight     = std::getenv("ATL_SEAMW")    ? atof(std::getenv("ATL_SEAMW"))    : 4.0f;
    xatlas::PackOptions po; po.resolution=2048; po.padding=4; po.bilinear=true; po.blockAlign=true; po.createImage=false;
    double tc=svae::now_s(); xatlas::ComputeCharts(atlas, co);
    double tp=svae::now_s(); xatlas::PackCharts(atlas, po); double te=svae::now_s();
    printf("[remesh] %s atlas: %ux%u  charts=%u sub-atlases=%u  (charts %.1fs, pack %.1fs)\n",
           tag, atlas->width, atlas->height, atlas->chartCount, atlas->atlasCount, tp-tc, te-tp);
    fflush(stdout);
    xatlas::Destroy(atlas);
}

int main() {
    auto P = [](const std::string& k){ return std::string(REFS) + "/" + k + ".npy"; };
    NpyArray cN = npy_load(P("head_coords"));        // i32 [N,4]
    int N = (int)cN.shape[0];
    printf("[remesh] occupancy: N=%d voxels @ grid1024\n", N);

    // raw dual-grid extractor (the validated bit-exact M4 mesh) — for the before/after topology delta
    {
        NpyArray dN = npy_load(P("head_dual_vertices"));
        NpyArray iN = npy_load(P("head_intersected"));
        NpyArray qN = npy_load(P("head_quad_lerp"));
        std::vector<int8_t> inter(iN.i8(), iN.i8()+(size_t)N*3);
        svae::Mesh raw = svae::flexible_dual_grid_to_mesh(cN.i32(), N, dN.f32(), inter.data(), qN.f32(), 1024);
        int64_t b,nm; svae::mesh_topology_stats(raw, b, nm);
        printf("[remesh] RAW dual-grid: verts=%d faces=%d  boundary=%lld nonmanifold=%lld\n",
               raw.N, raw.F, (long long)b, (long long)nm);
        // RAW unwrap already measured: quality stalls -> SLOPPY, 30091 charts, 4336x4344. (slow; skip)
    }

    // ====================================================================================
    // SMOOTHED-FIELD path (the tight-atlas fix). Build a box-blurred occupancy field, march on it
    // with iso=0.5 + linear edge interp → smooth low-curvature surface → quadric-decimatable →
    // tight xatlas. Env: REMESH_BLUR (radius, default 1=3³), DEC_TARGET (decimate target faces),
    // POST_SMOOTH (Taubin iters on the low-poly mesh). This is the candidate that replaces the
    // binary-MT + post-Taubin path.
    // COARSE-GRID MC: march a downsampled occupancy field directly → low-poly smooth watertight mesh,
    // NO decimation stall. Env: REMESH_STRIDE (fine voxels/coarse cell), REMESH_BLUR (coarse blur),
    // POST_SMOOTH (Taubin iters), DEC_TARGET (0 = no decimation, unwrap as-is).
    if (std::getenv("REMESH_COARSE")) {
        int stride = std::getenv("REMESH_STRIDE") ? atoi(std::getenv("REMESH_STRIDE")) : 4;
        int blur   = std::getenv("REMESH_BLUR") ? atoi(std::getenv("REMESH_BLUR")) : 0;
        bool solid = std::getenv("REMESH_SOLID") != nullptr;
        size_t target = std::getenv("DEC_TARGET") ? (size_t)atoll(std::getenv("DEC_TARGET")) : 0;
        double tc = svae::now_s();
        svae::Mesh cm = solid ? svae::marching_cubes_solid(cN.i32(), N, 1024, stride, blur, 0.5f)
                              : svae::marching_cubes_coarse(cN.i32(), N, 1024, stride, blur, 0.5f);
        double dtc = svae::now_s()-tc;
        int64_t cb,cnm; svae::mesh_topology_stats(cm, cb, cnm);
        printf("[remesh] COARSE-MC stride=%d blur=%d: grid=%d verts=%d faces=%d  boundary=%lld nonmanifold=%lld  (%.2fs)\n",
               stride, blur, 1024/stride, cm.N, cm.F, (long long)cb, (long long)cnm, dtc);
        fflush(stdout);
        int ps = std::getenv("POST_SMOOTH") ? atoi(std::getenv("POST_SMOOTH")) : 0;
        if (ps > 0) { double ts=svae::now_s(); svae::taubin_smooth(cm, ps);
            printf("[remesh] COARSE Taubin %d (%.2fs)\n", ps, svae::now_s()-ts); fflush(stdout); }
        svae::write_ply("miku_remesh_coarse.ply", cm.verts, cm.faces);
        char tag[48]; snprintf(tag, sizeof(tag), "COARSE-s%d", stride);
        decimate_and_unwrap(tag, cm.verts, cm.faces, target ? target : cm.F);
        printf("[remesh] wrote miku_remesh_coarse.ply\n");
        return (cb==0 && cnm==0) ? 0 : 1;
    }

    if (std::getenv("REMESH_FIELD")) {
        int blur = std::getenv("REMESH_BLUR") ? atoi(std::getenv("REMESH_BLUR")) : 1;
        size_t target = std::getenv("DEC_TARGET") ? (size_t)atoll(std::getenv("DEC_TARGET")) : 150000;
        double tf = svae::now_s();
        svae::Mesh fm = svae::marching_tetrahedra_field(cN.i32(), N, 1024, blur, 0.5f);
        double dtf = svae::now_s()-tf;
        int64_t fb,fnm; svae::mesh_topology_stats(fm, fb, fnm);
        printf("[remesh] FIELD-MT blur=%d: verts=%d faces=%d  boundary=%lld nonmanifold=%lld  (%.2fs)\n",
               blur, fm.N, fm.F, (long long)fb, (long long)fnm, dtf);
        fflush(stdout);
        char tag[48]; snprintf(tag, sizeof(tag), "FIELD-b%d", blur);
        decimate_and_unwrap(tag, fm.verts, fm.faces, target);
        svae::write_ply("miku_remesh_field.ply", fm.verts, fm.faces);
        printf("[remesh] wrote miku_remesh_field.ply\n");
        return (fb==0 && fnm==0) ? 0 : 1;
    }

    double t0 = svae::now_s();
    svae::Mesh m = svae::marching_tetrahedra(cN.i32(), N, 1024);
    double dt = svae::now_s()-t0;
    int64_t b,nm; svae::mesh_topology_stats(m, b, nm);
    printf("[remesh] MARCHING-TET: verts=%d faces=%d  boundary=%lld nonmanifold=%lld  (%.2fs)\n",
           m.N, m.F, (long long)b, (long long)nm, dt);

    bool ok = (b==0 && nm==0);
    printf("[remesh] %s — clean manifold watertight = %s\n",
           ok?"PASS":"FAIL", ok?"YES":"NO");
    // (raw-remesh unwrap measured once: SLOPPY, 48305 charts, 5196x5192 — worse than dual-grid's
    // 30091; the diagonal facet-noise defeats chart merging -> smoothing is the fix. Skip; slow.)

    // Taubin low-pass kills the staircase facet noise -> coherent normals -> chart collapse.
    int iters = std::getenv("SMOOTH_ITERS") ? atoi(std::getenv("SMOOTH_ITERS")) : 6;
    {
        svae::Mesh s = m;
        double ts = svae::now_s();
        svae::taubin_smooth(s, iters);
        printf("[remesh] --- Taubin %d iters (%.1fs) ---\n", iters, svae::now_s()-ts); fflush(stdout);
        char tag[32]; snprintf(tag, sizeof(tag), "SMOOTH%d", iters);
        decimate_and_unwrap(tag, s.verts, s.faces, 150000);
        svae::write_ply("miku_remesh_smooth.ply", s.verts, s.faces);
    }

    const char* out = "miku_remesh.ply";
    if (svae::write_ply(out, m.verts, m.faces))
        printf("[remesh] wrote %s (%d verts, %d faces)\n", out, m.N, m.F);
    return ok?0:1;
}
