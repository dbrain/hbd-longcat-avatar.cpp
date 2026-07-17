// TRELLIS-2 native UV-atlas bake (Stage-2 texturing, step 3): bake the PBR volume onto the oracle's
// mesh_pp and write a textured GLB to compare vs tex_goldens/oracle_textured.glb.
//   xatlas unwrap -> raster -> trilinear grid_sample (q=(pos+0.5)*1024, matches the oracle's
//   flex_gemm grid_sample_3d) -> baseColor[0,1,2]+alpha[5] / metalRough(R=0,G=rough[4],B=metal[3]).
//   Y/Z swap (y=z, z=-y) at the end to match the oracle's GLB frame.
//   PBR source: tex_goldens/pbr_{feats,coords}.npy by default (isolates the bake); env PBR_DIR overrides
//   (e.g. a native-decoded dump) — files <dir>/pbr_feats.npy + <dir>/pbr_coords.npy.
//   ./build.sh tex_bake_trellis && ./tex_bake_trellis [texture_size] [out.glb]
#include "tex_atlas.hpp"
#include "glb_textured.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* G = "/mnt/hdd/3d/avatar-shootout/tex_goldens";

int main(int argc, char** argv) {
    int TS = (argc>1)? atoi(argv[1]) : 2048;
    std::string out = (argc>2)? argv[2] : "/mnt/hdd/3d/avatar-shootout/tex_goldens/native_textured.glb";
    std::string pdir = getenv("PBR_DIR") ? getenv("PBR_DIR") : G;

    NpyArray vN = npy_load(std::string(G)+"/mesh_pp_verts.npy");   // [V,3] f64
    NpyArray fN = npy_load(std::string(G)+"/mesh_pp_faces.npy");   // [F,3] i32
    NpyArray pfN= npy_load(pdir+"/pbr_feats.npy");                 // [N,6] f32
    NpyArray pcN= npy_load(pdir+"/pbr_coords.npy");                // [N,4] f32 (golden) or i32
    int V=(int)vN.shape[0], F=(int)fN.shape[0], N=(int)pfN.shape[0];
    printf("[bake] mesh_pp V=%d F=%d, PBR N=%d, TS=%d, PBR_DIR=%s\n", V, F, N, TS, pdir.c_str());

    std::vector<float> verts((size_t)V*3);
    if (vN.descr=="<f8") for (size_t i=0;i<(size_t)V*3;i++) verts[i]=(float)vN.f64()[i];
    else                 for (size_t i=0;i<(size_t)V*3;i++) verts[i]=vN.f32()[i];
    std::vector<int64_t> faces((size_t)F*3);
    if (fN.descr=="<i4") for (size_t i=0;i<(size_t)F*3;i++) faces[i]=fN.i32()[i];
    else                 for (size_t i=0;i<(size_t)F*3;i++) faces[i]=fN.i64()[i];
    std::vector<float> pbr(pfN.f32(), pfN.f32()+(size_t)N*6);
    std::vector<int32_t> coords((size_t)N*4);    // coords saved as float32 (golden) or int32
    if (pcN.descr=="<i4") for (size_t i=0;i<(size_t)N*4;i++) coords[i]=pcN.i32()[i];
    else                  for (size_t i=0;i<(size_t)N*4;i++) coords[i]=(int32_t)std::lround(pcN.f32()[i]);

    int pad = getenv("ATL_PAD") ? atoi(getenv("ATL_PAD")) : 4;
    float cone = getenv("ATL_CONE") ? atof(getenv("ATL_CONE")) : 40.f;
    texatlas::BakedTexture bt = texatlas::bake(verts, faces, pbr, coords, /*grid_res*/1024, TS, /*deci*/0,
                                               pad, /*verbose*/true, /*fb_r*/0, /*precluster*/false, cone);
    printf("[bake] atlas %dx%d, %d charts, %d out-verts\n", bt.tw, bt.th, bt.chart_count, (int)bt.verts.size()/3);

    // Y/Z swap to match the oracle GLB frame (vertices[:,1],vertices[:,2] = vertices[:,2],-vertices[:,1])
    if (!getenv("NO_YZ_SWAP")) {
        auto swap=[](std::vector<float>& a){ for (size_t i=0;i+3<=a.size(); i+=3){ float y=a[i+1],z=a[i+2]; a[i+1]=z; a[i+2]=-y; } };
        swap(bt.verts); swap(bt.normals);
    }

    stbi_write_png("/mnt/hdd/3d/avatar-shootout/tex_goldens/native_base_color.png", bt.tw, bt.th, 4, bt.base_color.data(), bt.tw*4);
    bool ok = glb::write_glb_textured(out.c_str(), bt.verts, bt.normals, bt.uvs, bt.faces,
                                      bt.base_color, bt.metal_rough, bt.tw, bt.th);
    printf("[bake] wrote %s (%s) + native_base_color.png\n", out.c_str(), ok?"ok":"FAIL");
    return ok?0:1;
}
