// Standalone validation of the Marching-Tetrahedra remesh (A1). Loads the saved Miku grid-1024
// occupancy (refs/stage5/head_coords.npy, the M4 decoder's ~1.5M voxels) and the raw dual-grid
// extractor output, then checks the remesh is MANIFOLD + WATERTIGHT (boundary==0, nonmanifold==0)
// where the raw dual-grid mesh is not. Writes miku_remesh.ply for an eyeball. Runs in seconds (no
// decoder). Build: ./build.sh remesh_test
#include "remesh.hpp"
#include "qem.hpp"
#include "tex_atlas.hpp"
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
    // DUAL_QEM (lap-17 P0 crispness): decimate the CRISP golden dual-grid mesh DIRECTLY with QEM,
    // never re-march a coarse grid (that's what rounds/blocks the features). Source = the watertight
    // dual-grid mesh (close_surface=true synthesises the frontier corners so boundary==0). orient
    // it (consistent winding unblocks meshopt QUALITY), build per-vertex normals, then
    // meshopt_simplifyWithAttributes (QUALITY QEM, NOT Sloppy) -> keeps silhouette + sharp edges
    // (heel spikes, fingers) while collapsing flat regions. Env: DEC_TARGET (faces, default 200k),
    // DEC_ERR (target_error, default 0.05 relative), NORMW (normal attribute weight, default 0.5),
    // LOCKB (lock borders), DEC_SLOPPYOFF (never fall to sloppy). Writes miku_dual_qem.ply.
    if (std::getenv("REMESH_DUAL")) {
        NpyArray dN = npy_load(P("head_dual_vertices"));
        NpyArray iN = npy_load(P("head_intersected"));
        NpyArray qN = npy_load(P("head_quad_lerp"));
        std::vector<int8_t> inter(iN.i8(), iN.i8()+(size_t)N*3);
        double tb = svae::now_s();
        svae::Mesh m = svae::flexible_dual_grid_to_mesh(cN.i32(), N, dN.f32(), inter.data(), qN.f32(),
                                                        1024, /*close_surface=*/true);
        int64_t b0,nm0; svae::mesh_topology_stats(m, b0, nm0);
        printf("[remesh] DUAL(close): verts=%d faces=%d boundary=%lld nonmanifold=%lld (%.2fs)\n",
               m.N, m.F, (long long)b0, (long long)nm0, svae::now_s()-tb);
        fflush(stdout);
        if (std::getenv("REMESH_MANIFOLD")) {
            double tm=svae::now_s(); int64_t add=svae::make_manifold(m);
            int64_t b1,nm1; svae::mesh_topology_stats(m, b1, nm1);
            printf("[remesh] make_manifold: +%lld verts -> verts=%d faces=%d boundary=%lld nonmanifold=%lld (%.2fs)\n",
                   (long long)add, m.N, m.F, (long long)b1, (long long)nm1, svae::now_s()-tm);
            fflush(stdout);
        }
        if (std::getenv("FILLH")) {
            double tf=svae::now_s(); svae::fill_holes(m, 0, true);
            printf("[remesh] fill_holes (%.2fs)\n", svae::now_s()-tf); fflush(stdout);
        }
        // QEM_PURE: our own non-manifold-tolerant Garland-Heckbert QEM (qem.hpp) — collapses the
        // SHARP dual mesh straight to target, no meshopt non-manifold stall, no sloppy noise.
        if (std::getenv("QEM_PURE")) {
            int target = std::getenv("DEC_TARGET") ? atoi(std::getenv("DEC_TARGET")) : 200000;
            double aggr = std::getenv("QEM_AGGR") ? atof(std::getenv("QEM_AGGR")) : 7.0;
            double tq=svae::now_s();
            svae::Mesh om = qem::qem_simplify(m, target, aggr);
            printf("[remesh] QEM pure -> %d verts %d faces (target %d, aggr=%.1f, %.2fs)\n",
                   om.N, om.F, target, aggr, svae::now_s()-tq);
            if (std::getenv("QEM_MANIFOLD")) {
                double tm=svae::now_s(); int64_t add=svae::make_manifold(om);
                int64_t b,nm; svae::mesh_topology_stats(om,b,nm);
                printf("[remesh] QEM make_manifold: +%lld v -> %d v %d f boundary=%lld nm=%lld (%.2fs)\n",
                       (long long)add, om.N, om.F, (long long)b, (long long)nm, svae::now_s()-tm);
            }
            svae::orient_consistent(om);
            svae::write_ply("miku_dual_qem.ply", om.verts, om.faces);
            printf("[remesh] wrote miku_dual_qem.ply\n");
            // offline atlas-speed test: PRECLUSTER (AddUvMesh, pack-only) vs xatlas ComputeCharts
            if (std::getenv("QEM_ATLAS_PRECLUSTER")) {
                std::vector<float> uv; std::vector<int> uv2o; std::vector<uint32_t> uvf, fm; int ncl=0;
                float cdeg = std::getenv("ATL_CONE") ? atof(std::getenv("ATL_CONE")) : 62.f;
                double tp=svae::now_s();
                texatlas::precluster_charts(om.verts, om.faces, std::cos(cdeg*3.14159265f/180.f), uv, uv2o, uvf, fm, ncl);
                printf("[remesh] PRECLUSTER: %d charts (cone %.0f°, %.2fs)\n", ncl, cdeg, svae::now_s()-tp);
                xatlas::Atlas* at=xatlas::Create();
                xatlas::UvMeshDecl um; um.vertexCount=(uint32_t)uv2o.size(); um.vertexUvData=uv.data();
                um.vertexStride=2*sizeof(float); um.indexCount=(uint32_t)uvf.size(); um.indexData=uvf.data();
                um.indexFormat=xatlas::IndexFormat::UInt32; um.faceMaterialData=fm.data();
                xatlas::AddUvMesh(at, um);
                double tc=svae::now_s(); xatlas::ComputeCharts(at);
                double tk=svae::now_s(); xatlas::PackOptions po; po.resolution=2048; po.padding=4; po.bilinear=true; po.blockAlign=true; po.createImage=false;
                xatlas::PackCharts(at, po); double te=svae::now_s();
                printf("[remesh] PRECLUSTER atlas: %ux%u charts=%u (compute %.2fs, pack %.2fs)\n",
                       at->width, at->height, at->chartCount, tk-tc, te-tk);
                xatlas::Destroy(at);
            } else if (!std::getenv("DEC_NOUNWRAP")) decimate_and_unwrap("QEM", om.verts, om.faces, om.F);
            return 0;
        }
        double to = svae::now_s(); svae::orient_consistent(m);
        printf("[remesh] orient_consistent (%.2fs)\n", svae::now_s()-to); fflush(stdout);

        const size_t Vin = m.verts.size()/3;
        // per-vertex area-weighted normals (attribute that locks sharp features for QEM)
        std::vector<float> nrm(Vin*3, 0.f);
        for (size_t t=0;t<m.faces.size();t+=3){
            int64_t i0=m.faces[t],i1=m.faces[t+1],i2=m.faces[t+2];
            const float*A=&m.verts[i0*3],*B=&m.verts[i1*3],*C=&m.verts[i2*3];
            float e1[3]={B[0]-A[0],B[1]-A[1],B[2]-A[2]}, e2[3]={C[0]-A[0],C[1]-A[1],C[2]-A[2]};
            float fn[3]={e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
            for(int64_t v:{i0,i1,i2}) for(int d=0;d<3;d++) nrm[v*3+d]+=fn[d];
        }
        for (size_t i=0;i<Vin;i++){ float*n=&nrm[i*3]; float L=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if(L>1e-12f){n[0]/=L;n[1]/=L;n[2]/=L;} }

        std::vector<uint32_t> idx(m.faces.begin(), m.faces.end());
        size_t target = std::getenv("DEC_TARGET") ? (size_t)atoll(std::getenv("DEC_TARGET")) : 200000;
        float derr = std::getenv("DEC_ERR") ? atof(std::getenv("DEC_ERR")) : 0.05f;
        float normw = std::getenv("NORMW") ? atof(std::getenv("NORMW")) : 0.5f;
        unsigned opts = 0;
        if (std::getenv("LOCKB")) opts |= meshopt_SimplifyLockBorder;
        if (std::getenv("DEC_REG")) opts |= meshopt_SimplifyRegularize;
        float aw[3]={normw,normw,normw};
        std::vector<uint32_t> dst(idx.size()); float err=0.f;
        double td=svae::now_s();
        // PASS 1 — QEM QUALITY (feature-preserving). On the non-manifold dual mesh this stalls at the
        // locked-skeleton floor (~1.08M) but it's CLEAN: sharp heels/fingers intact, coherent normals.
        size_t nic = meshopt_simplifyWithAttributes(dst.data(), idx.data(), idx.size(),
            m.verts.data(), Vin, 3*sizeof(float), nrm.data(), 3*sizeof(float), aw, 3,
            /*vertex_lock=*/nullptr, target*3, derr, opts, &err);
        bool sloppy=false;
        printf("[remesh] DUAL QEM-attr -> %zu faces (target %zu, err=%.5f, %.2fs)%s\n",
               nic/3, target, err, svae::now_s()-td, (nic > target*3*5/4)?" [STALLED]":" [OK]");
        dst.resize(nic);
        // compact the QUALITY result
        std::vector<int> remap(Vin,-1); std::vector<float> vo; std::vector<int64_t> fo; fo.reserve(nic);
        std::vector<float> no; no.reserve(nic);   // carry normals for a possible 2nd attr pass
        for (uint32_t oi:dst){ if(remap[oi]<0){ remap[oi]=(int)(vo.size()/3);
            vo.push_back(m.verts[(size_t)oi*3]); vo.push_back(m.verts[(size_t)oi*3+1]); vo.push_back(m.verts[(size_t)oi*3+2]);
            no.push_back(nrm[(size_t)oi*3]); no.push_back(nrm[(size_t)oi*3+1]); no.push_back(nrm[(size_t)oi*3+2]); }
            fo.push_back(remap[oi]); }
        // PASS 2 — if still over target, decimate the CLEAN quality intermediate further. Default uses
        // QEM-attr again with permissive (often breaks past the stall now that the mesh is far smaller +
        // re-welded); SLOPPY2=1 forces sloppy (faster, slightly noisier). Feeding the clean intermediate
        // (not the raw 3.25M soup) makes the 2nd pass far less destructive than sloppy-from-raw.
        if (!std::getenv("DEC_SLOPPYOFF") && fo.size() > target*3*5/4) {
            size_t V2=vo.size()/3; std::vector<uint32_t> i2(fo.begin(),fo.end()), d2(i2.size()); float e2=0.f;
            size_t n2; double ts=svae::now_s();
            if (std::getenv("SLOPPY2")) {
                n2 = meshopt_simplifySloppy(d2.data(), i2.data(), i2.size(), vo.data(), V2,
                                            3*sizeof(float), target*3, 1.0f, &e2); sloppy=true;
            } else {
                n2 = meshopt_simplifyWithAttributes(d2.data(), i2.data(), i2.size(), vo.data(), V2,
                    3*sizeof(float), no.data(), 3*sizeof(float), aw, 3, nullptr, target*3, 1.0f,
                    opts|meshopt_SimplifyPermissive, &e2);
                if (n2 > target*3*5/4) {   // still stuck → sloppy on the clean intermediate
                    n2 = meshopt_simplifySloppy(d2.data(), i2.data(), i2.size(), vo.data(), V2,
                                                3*sizeof(float), target*3, 1.0f, &e2); sloppy=true; }
            }
            d2.resize(n2);
            std::vector<int> rm2(V2,-1); std::vector<float> v2; std::vector<int64_t> f2; f2.reserve(n2);
            for(uint32_t oi:d2){ if(rm2[oi]<0){ rm2[oi]=(int)(v2.size()/3);
                v2.push_back(vo[(size_t)oi*3]); v2.push_back(vo[(size_t)oi*3+1]); v2.push_back(vo[(size_t)oi*3+2]); }
                f2.push_back(rm2[oi]); }
            printf("[remesh] DUAL pass2 %s -> %zu faces (err=%.4f, %.2fs)\n",
                   sloppy?"SLOPPY":"QEM-perm", f2.size()/3, e2, svae::now_s()-ts);
            vo.swap(v2); fo.swap(f2);
        }
        printf("[remesh] DUAL final: %zu verts %zu faces %s\n", vo.size()/3, fo.size()/3,
               sloppy?"(pass2 SLOPPY)":"(QUALITY)");
        svae::write_ply("miku_dual_qem.ply", vo, fo);
        printf("[remesh] wrote miku_dual_qem.ply\n");
        // optional unwrap timing
        if (!std::getenv("DEC_NOUNWRAP")) {
            svae::Mesh om; om.verts=vo; om.faces=fo; om.N=(int)(vo.size()/3); om.F=(int)(fo.size()/3);
            decimate_and_unwrap("DUAL", om.verts, om.faces, om.F);
        }
        return 0;
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
