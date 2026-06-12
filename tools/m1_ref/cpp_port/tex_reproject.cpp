// Offline lap-18 BVH-reproject bake on the ALIGNED dump (PIXAL3D_DUMP_BAKE=1):
//   dump_mesh_*  = QEM mesh (the bake target / UV atlas)
//   dump_dense_* = the smooth DENSE outer-shell mesh (the colour source-of-truth)
//   dump_pbr_*   = the per-voxel 6-ch PBR volume
// Colours the dense verts (grid_sample, 6-ch, on-shell → exact), then bakes a UV PBR texture for the
// QEM mesh by snapping each texel onto the dense mesh (closest-pt-on-tri + front-face reject). Iterate
// params in seconds, NO GPU. Render vs pyref_front.png.
//   ./build.sh tex_reproject && ./tex_reproject [texsize]
// Env: ATL_CONE, ATL_PAD, RP_NCELL, RP_MAXRING, RP_FRONTDOT, TEX_INPAINT_ITERS, VCOLOR_FB.
#include "tex_atlas.hpp"
#include "tex_grid_sample.hpp"
#include "qem.hpp"               // feature-preserving QEM decimation (the production bake-target path)
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
    const std::vector<float>*  tv=&qverts; const std::vector<int64_t>* tf=&qfaces; int decim=0;
    std::vector<float> qv2; std::vector<int64_t> qf2;
    if (getenv("DENSE_TARGET")){
        // best-look path: bake on the full DENSE mesh with a real conformal xatlas unwrap (use NO_PRECL).
        tv=&dverts; tf=&dfaces; printf("[rp] bake target = full DENSE mesh %zu f (use NO_PRECL=1 for real unwrap)\n", NFd);
    } else if (qemt>0){
        // PRODUCTION path: feature-preserving QEM-decimate the DENSE mesh to QEM_TARGET faces.
        svae::Mesh dm; dm.verts=dverts; dm.faces=dfaces; dm.N=(int)NVd; dm.F=(int)NFd;
        double tq=texatlas::_now();
        double aggr=getenv("QEM_AGGR")?atof(getenv("QEM_AGGR")):7.0;
        svae::Mesh qm=qem::qem_simplify(dm, qemt, aggr);
        qv2=qm.verts; qf2=qm.faces; tv=&qv2; tf=&qf2;
        printf("[rp] bake target = QEM(dense) %zu -> %zu faces (aggr %.1f, %.2fs)\n", NFd, qf2.size()/3, aggr, texatlas::_now()-tq);
    } else if (texmf>0){ tv=&dverts; tf=&dfaces; decim=texmf; printf("[rp] bake target = DENSE sloppy-decimated to %d faces\n", texmf); }
    texatlas::BakedTexture bt=texatlas::bake(*tv,*tf,pbr,coords,1024,TS,decim,pad,true,/*fbr*/16,
                                             precl,cone, &dverts,&dfaces,&dattr,/*reproject*/true);
    glb::write_glb_textured(OUT, bt.verts,bt.normals,bt.uvs,bt.faces,bt.base_color,bt.metal_rough,bt.tw,bt.th);
    printf("[rp] wrote %s (atlas %dx%d, %d charts)\n", OUT, bt.tw,bt.th,bt.chart_count);
    return 0;
}
