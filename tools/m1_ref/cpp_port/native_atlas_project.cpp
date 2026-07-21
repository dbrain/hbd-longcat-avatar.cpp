// native_atlas_project — CPU-only observed-image overlay for a *native* GLB atlas.
//
// Unlike image_to_rig --tex-project-overlay this starts from the selected native
// base-color atlas, so unobserved texels retain native generated material. It writes
// a replacement base-color PNG only; glb_repack's GLB_REPACK_BASE_COLOR_PNG path
// then rebuilds the GLB while preserving its metallic/roughness texture.
#include "glb_reader.hpp"
#include "tex_project.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void usage() {
    std::printf("usage: native_atlas_project --mesh native.glb --front image.png --out-atlas atlas.png\\n"
                "       [--camera camera_provenance.txt] [--cam radians --dist units --scale value]\\n"
                "       [--back image.png] [--view yaw_deg image.png]... [--debug-dir dir] [--stats out.txt]\\n");
}
static float read_camera_value(const std::string& path, const char* key, float fallback) {
    std::ifstream f(path); std::string line, prefix=std::string(key)+"=";
    while (std::getline(f,line)) if (line.rfind(prefix,0)==0) return std::strtof(line.c_str()+prefix.size(),nullptr);
    return fallback;
}
static bool write_stats(const std::string& path, const texproj::Stats& s) {
    std::ofstream f(path,std::ios::trunc); if (!f) return false;
    f << "schema_version=1\n"
      << "front_painted_percent=" << s.front_pct << '\n'
      << "back_painted_percent=" << s.back_pct << '\n'
      << "unobserved_hole_percent_before_native_fallback=" << s.hole_pct << '\n'
      << "covered_texels=" << s.covered << '\n'
      << "unobserved_texels_retained_from_native_base=" << s.n_base << '\n'
      << "telea_fallback_texels=" << s.n_telea << '\n'
      << "seam_texels=" << s.n_seam << '\n'
      << "seam_mean_absdiff_255=" << s.seam_mean_absdiff << '\n';
    for (size_t i=0;i<s.view_yaw.size();i++)
        f << "view yaw=" << s.view_yaw[i] << " painted_percent=" << s.view_pct[i]
          << " align_scale=" << s.view_scale[i] << " align_tx=" << s.view_tx[i]
          << " align_ty=" << s.view_ty[i] << '\n';
    return (bool)f;
}
int main(int argc,char**argv) {
    std::string mesh,front,out_atlas,back,camera,debug,stats;
    texproj::Cfg cfg; cfg.preserve_base_for_holes=true; cfg.verbose=true;
    bool cam_set=false, dist_set=false, scale_set=false;
    for(int i=1;i<argc;i++) { std::string a=argv[i];
        auto next=[&]()->const char* { if(++i>=argc) throw std::runtime_error("missing value for "+a); return argv[i]; };
        if(a=="--mesh") mesh=next(); else if(a=="--front") front=next(); else if(a=="--out-atlas") out_atlas=next();
        else if(a=="--back") back=next(); else if(a=="--camera") camera=next(); else if(a=="--debug-dir") debug=next(); else if(a=="--stats") stats=next();
        else if(a=="--cam") { cfg.cam=std::strtof(next(),nullptr); cam_set=true; }
        else if(a=="--dist") { cfg.dist=std::strtof(next(),nullptr); dist_set=true; }
        else if(a=="--scale") { cfg.ms=std::strtof(next(),nullptr); scale_set=true; }
        else if(a=="--view") { float yaw=std::strtof(next(),nullptr); std::string img=next(); cfg.views.push_back({yaw,img}); }
        else { usage(); return 2; }
    }
    if(mesh.empty()||front.empty()||out_atlas.empty()) { usage(); return 2; }
    try {
        if(!camera.empty()) {
            if(!cam_set) cfg.cam=read_camera_value(camera,"cam_angle_x_rad",cfg.cam);
            if(!dist_set) cfg.dist=read_camera_value(camera,"camera_distance",cfg.dist);
            if(!scale_set) cfg.ms=read_camera_value(camera,"mesh_scale",cfg.ms);
        }
        glb::Mesh m; if(!glb::read_glb(mesh.c_str(),m)) throw std::runtime_error("cannot read native GLB mesh");
        if(m.uvs.size()!=m.verts.size()/3*2 || m.normals.size()!=m.verts.size())
            throw std::runtime_error("native GLB must have full POSITION/NORMAL/TEXCOORD_0");
        std::vector<uint8_t> png; std::string mime;
        if(!glb::read_glb_basecolor_png(mesh.c_str(),png,mime)) throw std::runtime_error("cannot extract native baseColor image");
        int w=0,h=0,c=0; uint8_t* px=stbi_load_from_memory(png.data(),(int)png.size(),&w,&h,&c,4);
        if(!px || w<1 || h<1) throw std::runtime_error("cannot decode native baseColor image");
        texatlas::BakedTexture bt; bt.tw=w; bt.th=h; bt.verts=std::move(m.verts); bt.normals=std::move(m.normals);
        bt.uvs=std::move(m.uvs); bt.faces.reserve(m.faces.size());
        for(int64_t f:m.faces) { if(f<0 || f>0xffffffffLL) throw std::runtime_error("native GLB has invalid index"); bt.faces.push_back((uint32_t)f); }
        bt.base_color.assign(px,px+(size_t)w*h*4); stbi_image_free(px);
        // texproj only modifies base_color, but retain a correctly sized material buffer so
        // BakedTexture remains a valid carrier for future direct GLB writing.
        bt.metal_rough.assign((size_t)w*h*3,0);
        cfg.front_img=front; cfg.back_img=back; cfg.debug_dir=debug;
        texproj::Stats s;
        if(!texproj::project_onto(bt,cfg,&s)) throw std::runtime_error("native atlas projection failed");
        if(!stbi_write_png(out_atlas.c_str(),w,h,4,bt.base_color.data(),w*4)) throw std::runtime_error("cannot write projected native atlas");
        if(!stats.empty() && !write_stats(stats,s)) throw std::runtime_error("cannot write projection stats");
        std::printf("[native-atlas-project] DONE atlas=%s %dx%d native-base-fallback=%d telea=%d seam=%.3f/255\n",
                    out_atlas.c_str(),w,h,s.n_base,s.n_telea,s.seam_mean_absdiff);
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
    return 0;
}
