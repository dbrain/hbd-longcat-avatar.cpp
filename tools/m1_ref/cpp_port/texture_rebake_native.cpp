// texture_rebake_native -- CPU-only native re-atlas of an already decoded PBR volume.
//
// Lets the runbook A/B chart-safe cleanup, UV settings, and atlas size without repeating DINO,
// texture diffusion, or M6 decode.  Input dumps are emitted by texture_mesh_native --dump-dir.
// No Python is used at runtime.
#include "tex_atlas.hpp"
#include "glb_reader.hpp"
#include "glb_textured.hpp"
#include "../../sparse_spike/npy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void usage() {
    std::printf("usage: texture_rebake_native --mesh refined.glb --pbr-dir native-dump --out out.glb\\n"
                "                             [--resolution 512|1024] [--texsize N] [--decimate faces]\\n"
                "                             [--unwrap production|reference] [--atlas-out base_color.png]\\n");
}

// Matches texture_mesh_native's Trellis texturing frame exactly.
static void preprocess_mesh(std::vector<float>& v) {
    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (size_t i=0;i<v.size();i+=3) for (int c=0;c<3;c++) { mn[c]=std::min(mn[c],v[i+c]); mx[c]=std::max(mx[c],v[i+c]); }
    float center[3], extent=0.f;
    for (int c=0;c<3;c++) { center[c]=.5f*(mn[c]+mx[c]); extent=std::max(extent,mx[c]-mn[c]); }
    float scale=.99999f/std::max(extent,1e-12f);
    for (size_t i=0;i<v.size();i+=3) { float x=(v[i]-center[0])*scale, y=(v[i+1]-center[1])*scale, z=(v[i+2]-center[2])*scale;
        v[i]=x; v[i+1]=-z; v[i+2]=y; }
}
static void restore_mesh_frame(std::vector<float>& v) {
    for (size_t i=0;i<v.size();i+=3) { float y=v[i+1],z=v[i+2]; v[i+1]=z; v[i+2]=-y; }
}

int main(int argc,char**argv) {
    std::string mesh_path,pdir,out,atlas_out,unwrap="reference"; int resolution=512,texsize=2048,decimate=0;
    for (int i=1;i<argc;i++) { std::string a=argv[i];
        if(a=="--mesh"&&i+1<argc) mesh_path=argv[++i];
        else if(a=="--pbr-dir"&&i+1<argc) pdir=argv[++i];
        else if(a=="--out"&&i+1<argc) out=argv[++i];
        else if(a=="--resolution"&&i+1<argc) resolution=atoi(argv[++i]);
        else if(a=="--texsize"&&i+1<argc) texsize=atoi(argv[++i]);
        else if(a=="--decimate"&&i+1<argc) decimate=atoi(argv[++i]);
        else if(a=="--unwrap"&&i+1<argc) unwrap=argv[++i];
        else if(a=="--atlas-out"&&i+1<argc) atlas_out=argv[++i];
        else { usage(); return 2; }
    }
    if(mesh_path.empty()||pdir.empty()||out.empty()||(resolution!=512&&resolution!=1024)||texsize<=0 ||
       (unwrap!="production" && unwrap!="reference")) { usage(); return 2; }
    if (atlas_out.empty()) atlas_out=out+".atlas.png";
    try {
        glb::Mesh mesh; if(!glb::read_glb(mesh_path.c_str(),mesh)) throw std::runtime_error("cannot read mesh: "+mesh_path);
        NpyArray pf=npy_load(pdir+"/native_pbr_feats.npy"), pc=npy_load(pdir+"/native_pbr_coords.npy");
        if(pf.descr!="<f4" || pf.shape.size()!=2 || pf.shape[1]!=6 || pc.shape.size()!=2 || pc.shape[1]!=4)
            throw std::runtime_error("invalid native PBR dump layout");
        int N=(int)pf.shape[0]; if((int)pc.shape[0]!=N) throw std::runtime_error("PBR feature/coordinate count mismatch");
        std::vector<float> pbr(pf.f32(),pf.f32()+(size_t)N*6);
        std::vector<int32_t> coords((size_t)N*4);
        if(pc.descr=="<i4") std::memcpy(coords.data(),pc.i32(),coords.size()*sizeof(int32_t));
        else if(pc.descr=="<f4") for(size_t i=0;i<coords.size();i++) coords[i]=(int32_t)std::lround(pc.f32()[i]);
        else throw std::runtime_error("PBR coordinates must be int32 or float32");
        if (const char* f=std::getenv("TEX_PBR_OUTLIER")) {
            float threshold=(float)atof(f);
            size_t changed=texatlas::filter_pbr_rgb_outliers(pbr,coords,threshold);
            std::printf("[native-rebake] PBR RGB outlier cleanup threshold=%.3f: %zu / %d voxels\n",threshold,changed,N);
        }
        preprocess_mesh(mesh.verts);
        // `reference` removes the native pre-cluster stage and gives xatlas the same chart
        // parameters used by Pixal3D's CuMesh unwrap.  It intentionally retains the native
        // rasterizer and PBR sampling: this is an isolated, CPU-only unwrap/bake A/B.
        const bool reference_unwrap = unwrap=="reference";
        if (reference_unwrap) {
            setenv("ATL_PYREF_XATLAS", "1", 1);
            std::printf("[native-rebake] reference unwrap: direct mesh xatlas, Pixal3D chart settings\\n");
        }
        texatlas::BakedTexture baked=texatlas::bake(mesh.verts,mesh.faces,pbr,coords,resolution,texsize,decimate,
                                                    reference_unwrap ? 0 : 4,true,8,!reference_unwrap);
        restore_mesh_frame(baked.verts); restore_mesh_frame(baked.normals);
        if(!glb::write_glb_textured(out.c_str(),baked.verts,baked.normals,baked.uvs,baked.faces,
                                    baked.base_color,baked.metal_rough,baked.tw,baked.th))
            throw std::runtime_error("could not write output GLB");
        if(!stbi_write_png(atlas_out.c_str(),baked.tw,baked.th,4,baked.base_color.data(),baked.tw*4))
            throw std::runtime_error("could not write baseColor atlas: "+atlas_out);
        std::printf("[native-rebake] DONE: %s (%d charts, %dx%d atlas)\\n",out.c_str(),baked.chart_count,baked.tw,baked.th);
        std::printf("[native-rebake] baseColor atlas: %s\\n",atlas_out.c_str());
    } catch(const std::exception&e) { std::fprintf(stderr,"FAIL: %s\\n",e.what()); return 1; }
    return 0;
}
