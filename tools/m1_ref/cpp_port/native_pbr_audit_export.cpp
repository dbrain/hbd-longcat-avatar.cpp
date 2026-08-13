// native_pbr_audit_export -- export an exact native bake input for the Python reference audit.
//
// This is deliberately NOT a production-stage dependency.  It makes a same-mesh, same-cleaned-PBR
// fixture so the reference can be used as an offline oracle for a native atlas comparison.
//
// usage:
//   native_pbr_audit_export <refined.glb> <native_high_texture_dump> <out-dir> [decimate=300000]
// out-dir receives the binary filenames expected by bake_uv.py: dump_mesh_{v,f}.bin,
// dump_pbr_{f,c}.bin, plus audit_grid_res.txt.
#include "glb_reader.hpp"
#include "tex_atlas.hpp"
#include "../../sparse_spike/npy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void preprocess_mesh(std::vector<float>& v) {
    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (size_t i=0;i<v.size();i+=3) for (int c=0;c<3;c++) { mn[c]=std::min(mn[c],v[i+c]); mx[c]=std::max(mx[c],v[i+c]); }
    float center[3], extent=0.f;
    for (int c=0;c<3;c++) { center[c]=.5f*(mn[c]+mx[c]); extent=std::max(extent,mx[c]-mn[c]); }
    float scale=.99999f/std::max(extent,1e-12f);
    for (size_t i=0;i<v.size();i+=3) {
        float x=(v[i]-center[0])*scale, y=(v[i+1]-center[1])*scale, z=(v[i+2]-center[2])*scale;
        v[i]=x; v[i+1]=-z; v[i+2]=y;
    }
}

template<class T> static void write_binary(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path,std::ios::binary|std::ios::trunc);
    if (!f || (!v.empty() && !f.write(reinterpret_cast<const char*>(v.data()),(std::streamsize)(v.size()*sizeof(T)))))
        throw std::runtime_error("cannot write "+path);
}

int main(int argc,char**argv) {
    if (argc<4 || argc>5) {
        std::fprintf(stderr,"usage: native_pbr_audit_export <refined.glb> <native-dump> <out-dir> [decimate=300000]\n");
        return 2;
    }
    const std::string mesh_path=argv[1], pdir=argv[2], outdir=argv[3];
    const int decimate=argc==5 ? std::atoi(argv[4]) : 300000;
    if (decimate<=0) { std::fprintf(stderr,"decimate must be positive\n"); return 2; }
    try {
        glb::Mesh mesh;
        if (!glb::read_glb(mesh_path.c_str(),mesh)) throw std::runtime_error("cannot read "+mesh_path);
        preprocess_mesh(mesh.verts);
        std::vector<float> dv; std::vector<int64_t> df;
        texatlas::decimate(mesh.verts,mesh.faces,(size_t)decimate,dv,df,true);
        NpyArray pf=npy_load(pdir+"/native_pbr_feats.npy"), pc=npy_load(pdir+"/native_pbr_coords.npy");
        if (pf.descr!="<f4" || pf.shape.size()!=2 || pf.shape[1]!=6 || pc.shape.size()!=2 || pc.shape[1]!=4 || pf.shape[0]!=pc.shape[0])
            throw std::runtime_error("invalid native PBR dump");
        const size_t n=(size_t)pf.shape[0];
        std::vector<float> pbr(pf.f32(),pf.f32()+n*6);
        std::vector<int32_t> coords(n*4);
        if (pc.descr=="<i4") std::memcpy(coords.data(),pc.i32(),coords.size()*sizeof(int32_t));
        else if (pc.descr=="<f4") for(size_t i=0;i<coords.size();i++) coords[i]=(int32_t)std::lround(pc.f32()[i]);
        else throw std::runtime_error("PBR coordinates must be int32 or float32");
        std::filesystem::create_directories(outdir);
        write_binary(outdir+"/dump_mesh_v.bin",dv);
        write_binary(outdir+"/dump_mesh_f.bin",df);
        write_binary(outdir+"/dump_pbr_f.bin",pbr);
        write_binary(outdir+"/dump_pbr_c.bin",coords);
        std::ofstream meta(outdir+"/audit_grid_res.txt",std::ios::trunc);
        if (!meta) throw std::runtime_error("cannot write audit grid metadata");
        meta << 512 << '\n'; // Native 512 texture flow/M6 decoder uses the 512 lattice.
        std::printf("[native-pbr-audit] mesh %zu v / %zu f, PBR %zu voxels -> %s (grid=512)\n",
                    dv.size()/3,df.size()/3,n,outdir.c_str());
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
    return 0;
}
