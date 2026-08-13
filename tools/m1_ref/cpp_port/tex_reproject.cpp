// Offline lap-18 BVH-reproject bake on the ALIGNED dump (PIXAL3D_DUMP_BAKE=1):
//   dump_mesh_*  = QEM mesh (the bake target / UV atlas)
//   dump_dense_* = the smooth DENSE outer-shell mesh (the colour source-of-truth)
//   dump_pbr_*   = the per-voxel 6-ch PBR volume
// Colours the dense verts (grid_sample, 6-ch, on-shell → exact), then bakes a UV PBR texture for the
// QEM mesh by snapping each texel onto the dense mesh (closest-pt-on-tri + front-face reject). Iterate
// params in seconds, NO GPU. Render vs pyref_front.png.
//   ./build.sh tex_reproject cuda && ./tex_reproject 4096 out.glb   # v6 parity defaults baked in
// The v6 FINDINGS-A flag-set (native cumesh 500k + pyref xatlas + RP_OFF + FBR0 + RASTER_SS2 + TELEA +
// topo-normals + keep-atlas-size + fill-bg) is now the DEFAULT (see main()); any env overrides a single
// flag. For a game LOD: `CUMESH_TARGET=70000 TEX_FINAL_SIZE=1024 ./tex_reproject 4096 lod.glb`.
// Env: ATL_CONE, ATL_PAD, RP_NCELL, RP_MAXRING, RP_FRONTDOT, TEX_INPAINT_ITERS, VCOLOR_FB.
#include "tex_atlas.hpp"
#include "tex_grid_sample.hpp"
#include "qem.hpp"               // feature-preserving QEM decimation (the production bake-target path)
#ifdef TEXATLAS_NATIVE_CUMESH
#include "native_cumesh_bridge.hpp"
#endif
#include "glb_textured.hpp"
#include "glb_writer.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static std::vector<uint8_t> rd(const char* p){ FILE* f=fopen(p,"rb"); if(!f){printf("missing %s\n",p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint8_t> b(n); size_t r=fread(b.data(),1,n,f); (void)r; fclose(f); return b; }

int main(int argc,char**argv){
    int TS=(argc>1)?atoi(argv[1]):2048;
    const char* OUT=(argc>2)?argv[2]:"miku_reproject_tex.glb";

    // v6 PARITY DEFAULTS (FINDINGS-A) — the diagnosed flag-set that reaches Python parity. Baked in as
    // defaults so the finalized command is just `./tex_reproject 4096 out.glb`. setenv(...,0) = the
    // env still wins (override any single flag), and the shared texatlas::bake() is left untouched so
    // the pixal3d --tex production path is NOT changed by this. TEX_KEEP_ATLAS_SIZE is auto-skipped
    // when a LOD downsample (TEX_FINAL_SIZE) is requested, matching the FINDINGS-A game-LOD recipe.
    {
        auto def=[](const char*k,const char*v){ setenv(k,v,0); }; // 0 = do NOT overwrite a user-set value
        def("ATL_NATIVE_CUMESH","1"); def("CUMESH_TARGET","500000"); def("ATL_PYREF_XATLAS","1");
        def("RP_OFF","1");            def("TEX_FBR","0");           def("TEX_RASTER_SS","2");
        def("TEX_TELEA_INPAINT","1"); def("TEX_TELEA_RADIUS","4");  def("TEX_INPAINT_ITERS","16");
        def("TEX_TOPO_NORMALS","1");  def("TEX_FILL_BACKGROUND","1");
        if (!getenv("TEX_FINAL_SIZE")) def("TEX_KEEP_ATLAS_SIZE","1"); // full-res parity = no downsample
        printf("[rp] v6 parity defaults active (set any env to override; TEX_FINAL_SIZE=<N> for a LOD)\n");
    }
    size_t NVq,NFq,NP; { FILE* f=fopen("dump_bake.txt","r"); if(!f){printf("no dump_bake.txt — run pixal3d PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 first\n");return 1;}
        if(fscanf(f,"%zu %zu %zu",&NVq,&NFq,&NP)!=3){fclose(f);return 1;} fclose(f); }
    size_t NVd,NFd; { FILE* f=fopen("dump_dense.txt","r"); if(!f){printf("no dump_dense.txt — rebuild the dump with the dense-mesh dump enabled\n");return 1;}
        if(fscanf(f,"%zu %zu",&NVd,&NFd)!=2){fclose(f);return 1;} fclose(f); }

    auto qvb=rd("dump_mesh_v.bin"); auto qfb=rd("dump_mesh_f.bin");
    auto dvb=rd("dump_dense_v.bin"); auto dfb=rd("dump_dense_f.bin");
    auto pf=rd("dump_pbr_f.bin"); auto pc=rd("dump_pbr_c.bin");
    std::vector<float>   qverts((float*)qvb.data(),(float*)qvb.data()+NVq*3);
    std::vector<int64_t> qfaces((int64_t*)qfb.data(),(int64_t*)qfb.data()+NFq*3);
    std::vector<float>   dverts((float*)dvb.data(),(float*)dvb.data()+NVd*3);
    std::vector<int64_t> dfaces((int64_t*)dfb.data(),(int64_t*)dfb.data()+NFd*3);
    std::vector<float>   pbr((float*)pf.data(),(float*)pf.data()+NP*6);
    std::vector<int32_t> coords((int32_t*)pc.data(),(int32_t*)pc.data()+NP*4);
    printf("[rp] QEM %zu v / %zu f | DENSE %zu v / %zu f | PBR %zu voxels | TS=%d\n", NVq,NFq,NVd,NFd,NP,TS);

    // RP_CANON_TO_DENSE: the AUTOMATED single-coordinate-frame fix. When the bake TARGET (dump_mesh =
    // qverts) comes from a different frame than the colour source (dump_dense = dverts) — e.g. the final
    // 69k routed through UltraShape (which re-emits in its own marching-cubes box_v=1.0 frame) — its verts
    // don't sit on the dense shell, so the closest-point snap misses (was 97.8% inpaint). UltraShape's
    // transform is empirically a pure CENTER + UNIFORM-SCALE (no rotation: per-axis bbox-extent ratios
    // match to <2%, axis assignment preserved), so we reconcile it by matching the qverts bbox-centre +
    // uniform scale onto the dverts bbox — computed from BOTH meshes at runtime, zero hand-typed numbers.
    // This carries the pipeline into the dump frame by construction; the bake (target+source now coincident)
    // needs no alignment. Off by default so the native pixal3d --tex path (target already in dump frame) is
    // unchanged.
    if (getenv("RP_CANON_TO_DENSE") && NVq && NVd) {
        auto bbox=[](const std::vector<float>& v, size_t n, float* lo, float* hi){
            for (int k=0;k<3;k++){ lo[k]= 1e30f; hi[k]=-1e30f; }
            for (size_t i=0;i<n;i++) for (int k=0;k<3;k++){ float x=v[i*3+k]; if(x<lo[k])lo[k]=x; if(x>hi[k])hi[k]=x; } };
        float qlo[3],qhi[3],dlo[3],dhi[3];
        bbox(qverts,NVq,qlo,qhi); bbox(dverts,NVd,dlo,dhi);
        float qc[3],dc[3],qe[3],de[3];
        for (int k=0;k<3;k++){ qc[k]=0.5f*(qlo[k]+qhi[k]); dc[k]=0.5f*(dlo[k]+dhi[k]); qe[k]=qhi[k]-qlo[k]; de[k]=dhi[k]-dlo[k]; }
        // uniform scale = ratio of bbox DIAGONAL lengths (robust to the small per-axis refine drift).
        double qd=std::sqrt((double)qe[0]*qe[0]+(double)qe[1]*qe[1]+(double)qe[2]*qe[2]);
        double dd=std::sqrt((double)de[0]*de[0]+(double)de[1]*de[1]+(double)de[2]*de[2]);
        float s=(qd>1e-12)?(float)(dd/qd):1.f;
        for (size_t i=0;i<NVq;i++) for (int k=0;k<3;k++) qverts[i*3+k]=(qverts[i*3+k]-qc[k])*s+dc[k];
        printf("[rp] RP_CANON_TO_DENSE: target -> dense frame  (scale %.4f, per-axis ext ratio [%.3f %.3f %.3f])\n",
               s, de[0]/qe[0], de[1]/qe[1], de[2]/qe[2]);
    }

    // colour the DENSE verts: 6-ch grid_sample at each vertex (on the shell → trilinear exact; small
    // fallback for any sub-voxel-off vert). This is the full-saturation, smooth source-of-truth.
    int fbr = getenv("VCOLOR_FB")?atoi(getenv("VCOLOR_FB")):12;
    std::vector<float> dattr(NVd*6, 0.f);
    double t0=texatlas::_now();
    // dense vert i IS pbr voxel i (1:1, NVd==NP) — DIRECT_ATTR uses the exact per-voxel PBR (no
    // trilinear, so no cross-shell bleed between the thin twin-tail front/back). Default trilinear
    // (grid_sample) smooths but can blend across the ~2-voxel tail gap.
    if (getenv("DIRECT_ATTR") && NVd==NP) {
        dattr.assign(pbr.begin(), pbr.end());
        printf("[rp] dense attr = per-voxel PBR direct (1:1, %.2fs)\n", texatlas::_now()-t0);
    } else {
        texgs::VolIndex vol(coords.data(),(int)NP,4,1);
        #pragma omp parallel for schedule(dynamic,4096)
        for (size_t i=0;i<NVd;i++){
            float q0=(dverts[i*3]+0.5f)*1024.f, q1=(dverts[i*3+1]+0.5f)*1024.f, q2=(dverts[i*3+2]+0.5f)*1024.f;
            texgs::sample_one(vol, pbr.data(), 6, q0,q1,q2, &dattr[i*6], fbr);
        }
        printf("[rp] coloured dense verts via grid_sample (%.2fs)\n", texatlas::_now()-t0);
    }

    // DENSE_VCOLOR: write the DENSE mesh with per-vertex base_color (the smooth on-shell ceiling — no
    // facets at 1.5M verts, exact colours, no UV needed). Diagnostic of the colour source-of-truth.
    if (getenv("DENSE_VCOLOR")) {
        std::vector<float> col(NVd*3);
        for (size_t i=0;i<NVd;i++) for (int c=0;c<3;c++){ float v=dattr[i*6+c]; col[i*3+c]=v<0?0:(v>1?1:v); }
        glb::write_glb(OUT, dverts, dfaces, &col);
        printf("[rp] wrote %s (DENSE per-vertex colour, %zu v / %zu f)\n", OUT, NVd, NFd);
        return 0;
    }
    // QEM_VCOLOR=N: feature-preserving QEM-decimate the DENSE mesh to N faces, CARRYING the per-vertex
    // colour through the collapses (qem in_col/out_col) — smooth per-vertex colour on a web-sized mesh,
    // no UV atlas (so no chart streaks/seams). The lap-17 vcolor path but at a higher budget so facets
    // are small enough to stay smooth.
    if (getenv("QEM_VCOLOR")) {
        int N=atoi(getenv("QEM_VCOLOR"));
        std::vector<float> col(NVd*3);
        for (size_t i=0;i<NVd;i++) for (int c=0;c<3;c++){ float v=dattr[i*6+c]; col[i*3+c]=v<0?0:(v>1?1:v); }
        svae::Mesh dm; dm.verts=dverts; dm.faces=dfaces; dm.N=(int)NVd; dm.F=(int)NFd;
        std::vector<float> ocol; double tq=texatlas::_now();
        svae::Mesh qm=qem::qem_simplify(dm, N, getenv("QEM_AGGR")?atof(getenv("QEM_AGGR")):7.0, &col, &ocol);
        glb::write_glb(OUT, qm.verts, qm.faces, &ocol);
        printf("[rp] wrote %s (QEM+colour-carry %zu->%zu f, %.2fs)\n", OUT, NFd, qm.faces.size()/3, texatlas::_now()-tq);
        return 0;
    }
    bool precl=getenv("NO_PRECL")?false:true;     // NO_PRECL=1 → real xatlas ComputeCharts unwrap
    float cone=getenv("ATL_CONE")?atof(getenv("ATL_CONE")):55.f;
    int pad=getenv("ATL_PAD")?atoi(getenv("ATL_PAD")):4;
    // DIAGNOSTIC: bake target = the DENSE mesh decimated to TEX_MESH_FACES (tests whether the blocky/
    // streaky texture is from the coarse 200k QEM facets). Reproject source stays the full dense mesh.
    int texmf=getenv("TEX_MESH_FACES")?atoi(getenv("TEX_MESH_FACES")):0;
    int qemt =getenv("QEM_TARGET")?atoi(getenv("QEM_TARGET")):0;
    int cmt =getenv("CUMESH_TARGET")?atoi(getenv("CUMESH_TARGET")):0;
    const std::vector<float>*  tv=&qverts; const std::vector<int64_t>* tf=&qfaces; int decim=0;
    std::vector<float> qv2; std::vector<int64_t> qf2;
    if (getenv("DENSE_TARGET")){
        // best-look path: bake on the full DENSE mesh with a real conformal xatlas unwrap (use NO_PRECL).
        tv=&dverts; tf=&dfaces; printf("[rp] bake target = full DENSE mesh %zu f (use NO_PRECL=1 for real unwrap)\n", NFd);
#ifdef TEXATLAS_NATIVE_CUMESH
    } else if (cmt>0){
        double tq=texatlas::_now();
        native_cumesh::simplify_to_faces(dverts, dfaces, cmt, qv2, qf2,
            getenv("CUMESH_THRESH") ? (float)atof(getenv("CUMESH_THRESH")) : 1e-8f,
            getenv("CUMESH_LAMBDA_EDGE") ? (float)atof(getenv("CUMESH_LAMBDA_EDGE")) : 1e-2f,
            getenv("CUMESH_LAMBDA_SKINNY") ? (float)atof(getenv("CUMESH_LAMBDA_SKINNY")) : 1e-3f);
        tv=&qv2; tf=&qf2;
        printf("[rp] bake target = native CuMesh simplify(dense) %zu -> %zu faces (%.2fs)\n", NFd, qf2.size()/3, texatlas::_now()-tq);
#endif
    } else if (qemt>0){
        // PRODUCTION path: feature-preserving QEM-decimate the DENSE mesh to QEM_TARGET faces.
        svae::Mesh dm; dm.verts=dverts; dm.faces=dfaces; dm.N=(int)NVd; dm.F=(int)NFd;
        double tq=texatlas::_now();
        double aggr=getenv("QEM_AGGR")?atof(getenv("QEM_AGGR")):7.0;
        svae::Mesh qm=qem::qem_simplify(dm, qemt, aggr);
        qv2=qm.verts; qf2=qm.faces; tv=&qv2; tf=&qf2;
        printf("[rp] bake target = QEM(dense) %zu -> %zu faces (aggr %.1f, %.2fs)\n", NFd, qf2.size()/3, aggr, texatlas::_now()-tq);
    } else if (texmf>0){ tv=&dverts; tf=&dfaces; decim=texmf; printf("[rp] bake target = DENSE sloppy-decimated to %d faces\n", texmf); }
    bool reproject = getenv("RP_OFF") ? false : true;
    // RP_CANON_TO_DENSE bakes onto a DECIMATED, canonicalised target (the 69k) whose verts no longer sit
    // on the sparse PBR shell voxels — direct grid_sample then misses most texels. Force the closest-point
    // reproject (snap each texel onto the dense shell → read its colour), the path built for exactly this.
    if (getenv("RP_CANON_TO_DENSE")) reproject = true;
    int bake_fbr = getenv("TEX_FBR") ? atoi(getenv("TEX_FBR")) : 16;
    texatlas::BakedTexture bt=texatlas::bake(*tv,*tf,pbr,coords,1024,TS,decim,pad,true,bake_fbr,
                                             precl,cone, &dverts,&dfaces,&dattr,reproject);
    glb::write_glb_textured(OUT, bt.verts,bt.normals,bt.uvs,bt.faces,bt.base_color,bt.metal_rough,bt.tw,bt.th);
    printf("[rp] wrote %s (atlas %dx%d, %d charts)\n", OUT, bt.tw,bt.th,bt.chart_count);
    return 0;
}
