// Offline bake on the ALIGNED dump from `pixal3d ... PIXAL3D_DUMP_BAKE=1` (mesh + PBR volume from the
// SAME run, so no stage4/stage5 misalignment). Iterate atlas/bake params in seconds, no GPU.
//   ./build.sh tex_bake_dump && ./tex_bake_dump [texsize]
// Env: PRECLUSTER, ATL_CONE, ATL_PAD, SAMPLE_FALLBACK_R, TEX_INPAINT_ITERS, VCOLOR (per-vertex instead).
#include "tex_atlas.hpp"
#include "tex_grid_sample.hpp"
#include "glb_textured.hpp"
#include "glb_writer.hpp"
#include "sparse_vae.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static std::vector<uint8_t> rd(const char* p){ FILE* f=fopen(p,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint8_t> b(n); size_t r=fread(b.data(),1,n,f); (void)r; fclose(f); return b; }

int main(int argc,char**argv){
    int TS=(argc>1)?atoi(argv[1]):2048;
    size_t NV,NF,NP; { FILE* f=fopen("dump_bake.txt","r"); if(!f){printf("no dump_bake.txt — run pixal3d PIXAL3D_DUMP_BAKE=1 first\n");return 1;}
        if(fscanf(f,"%zu %zu %zu",&NV,&NF,&NP)!=3){fclose(f);return 1;} fclose(f); }
    auto vb=rd("dump_mesh_v.bin"); auto fb=rd("dump_mesh_f.bin"); auto pf=rd("dump_pbr_f.bin"); auto pc=rd("dump_pbr_c.bin");
    std::vector<float> verts((float*)vb.data(),(float*)vb.data()+NV*3);
    std::vector<int64_t> faces((int64_t*)fb.data(),(int64_t*)fb.data()+NF*3);
    std::vector<float> pbr((float*)pf.data(),(float*)pf.data()+NP*6);
    std::vector<int32_t> coords((int32_t*)pc.data(),(int32_t*)pc.data()+NP*4);
    printf("[dump] mesh %zu v / %zu f, pbr %zu voxels, TS=%d\n", NV, NF, NP, TS);
    int fbr=getenv("SAMPLE_FALLBACK_R")?atoi(getenv("SAMPLE_FALLBACK_R")):4;

    if (getenv("VCOLOR")) {
        // per-vertex grid_sample (no UV) — clean diagnostic of the volume sampling itself
        texgs::VolIndex vol(coords.data(),(int)NP,4,1);
        std::vector<float> vcol(NV*3,0.f);
        #pragma omp parallel for schedule(dynamic,1024)
        for (size_t i=0;i<NV;i++){ float P[6]={0};
            float q0=(verts[i*3]+0.5f)*1024, q1=(verts[i*3+1]+0.5f)*1024, q2=(verts[i*3+2]+0.5f)*1024;
            texgs::sample_one(vol, pbr.data(), 6, q0,q1,q2, P, fbr);
            for(int c=0;c<3;c++) vcol[i*3+c]=P[c]<0?0:(P[c]>1?1:P[c]); }
        glb::write_glb("miku_dump_vcolor.glb", verts, faces, &vcol);
        printf("[dump] wrote miku_dump_vcolor.glb (per-vertex, fb_r=%d)\n", fbr);
        return 0;
    }
    bool precl=getenv("PRECLUSTER")!=nullptr;
    float cone=getenv("ATL_CONE")?atof(getenv("ATL_CONE")):55.f;
    int pad=getenv("ATL_PAD")?atoi(getenv("ATL_PAD")):4;
    texatlas::BakedTexture bt=texatlas::bake(verts,faces,pbr,coords,1024,TS,0,pad,true,fbr,precl,cone);
    glb::write_glb_textured("miku_dump_tex.glb", bt.verts,bt.normals,bt.uvs,bt.faces,bt.base_color,bt.metal_rough,bt.tw,bt.th);
    printf("[dump] wrote miku_dump_tex.glb (atlas %dx%d, %d charts)\n", bt.tw,bt.th,bt.chart_count);
    return 0;
}
