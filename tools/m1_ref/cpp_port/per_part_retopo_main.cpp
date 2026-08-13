// per_part_retopo_main — CPU fast-iter harness for seam-aware per-part IM retopo.
//   per_part_retopo <verts.npy f32[V,3]> <faces.npy i64[F,3]> <face_ids.npy i64[F]> <out.glb>
// Reads a mesh + P3-SAM per-face labels straight from npy (no GPU, no glb decode) so the seam/weld/
// fill algorithm can be tuned in a <10s loop on the goldens Miku. Env knobs:
//   PPR_SNAP  (default 3.0)  snap_radius_frac  (0 = no snapping)
//   PPR_WELD  (default 0.5)  weld_eps_frac     (0 = no weld)
//   PPR_FILL  (default 1)    fill holes (0 = off, to SEE the raw seam gaps)
//   IM_BIN                   instant_meshes_batch path (else im_retopo.hpp default)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../../sparse_spike/npy.hpp"   // npy_load -> f32
#include "per_part_retopo.hpp"
#include "glb_reader.hpp"
#include "glb_writer.hpp"

static bool ends_with(const std::string& s, const char* suf) {
    std::string t(suf); return s.size()>=t.size() && s.compare(s.size()-t.size(), t.size(), t)==0;
}

int main(int argc, char** argv) {
    // two forms:
    //   per_part_retopo <mesh.glb> <face_ids.npy> <out.glb>
    //   per_part_retopo <verts.npy> <faces.npy> <face_ids.npy> <out.glb>
    glb::Mesh in;
    std::vector<int64_t> fid;
    const char* out_path = nullptr;
    if (argc >= 4 && ends_with(argv[1], ".glb")) {
        if (!glb::read_glb(argv[1], in)) { std::fprintf(stderr, "read glb failed\n"); return 1; }
        std::printf("glb: %lld v / %lld f\n", (long long)in.verts.size()/3, (long long)in.faces.size()/3);
        if (!ppd::read_npy_int(argv[2], fid)) { std::fprintf(stderr, "read face_ids failed\n"); return 1; }
        out_path = argv[3];
    } else if (argc >= 5) {
        NpyArray va = npy_load(argv[1]);
        in.verts.assign(va.f32(), va.f32() + va.numel());
        std::vector<int64_t> faces;
        if (!ppd::read_npy_int(argv[2], faces)) { std::fprintf(stderr, "read faces failed\n"); return 1; }
        in.faces = faces;
        if (!ppd::read_npy_int(argv[3], fid)) { std::fprintf(stderr, "read face_ids failed\n"); return 1; }
        out_path = argv[4];
    } else {
        std::fprintf(stderr, "usage: per_part_retopo <mesh.glb> <face_ids.npy> <out.glb>\n"
                             "   or: per_part_retopo <verts.npy> <faces.npy> <face_ids.npy> <out.glb>\n");
        return 1;
    }
    std::printf("face_ids = %zu\n", fid.size());

    ppr::RetopoCfg cfg;
    if (const char* e = std::getenv("PPR_SNAP")) cfg.snap_radius_frac = std::atof(e);
    if (const char* e = std::getenv("PPR_WELD")) cfg.weld_eps_frac    = std::atof(e);
    if (const char* e = std::getenv("PPR_FILL")) cfg.fill_holes       = std::atoi(e) != 0;

    glb::Mesh out; std::vector<ppd::PartReport> rep;
    if (!ppr::per_part_im_retopo(in, fid, cfg, out, rep)) { std::fprintf(stderr, "retopo failed\n"); return 1; }

    if (!glb::write_glb(out_path, out.verts, out.faces)) { std::fprintf(stderr, "write %s failed\n", out_path); return 1; }
    // also emit an .obj sibling (for obj_fill_holes / render tools)
    { std::string op(out_path); size_t d=op.find_last_of('.'); if(d!=std::string::npos) op=op.substr(0,d); op+=".obj";
      FILE* f=std::fopen(op.c_str(),"w");
      if(f){ size_t nv=out.verts.size()/3, nf=out.faces.size()/3;
        for(size_t i=0;i<nv;i++) std::fprintf(f,"v %.9g %.9g %.9g\n",out.verts[i*3],out.verts[i*3+1],out.verts[i*3+2]);
        for(size_t t=0;t<nf;t++) std::fprintf(f,"f %lld %lld %lld\n",(long long)out.faces[t*3]+1,(long long)out.faces[t*3+1]+1,(long long)out.faces[t*3+2]+1);
        std::fclose(f); std::printf("wrote %s\n", op.c_str()); } }
    std::printf("wrote %s (%lld v / %lld f)\n", out_path, (long long)out.verts.size()/3, (long long)out.faces.size()/3);
    return 0;
}
