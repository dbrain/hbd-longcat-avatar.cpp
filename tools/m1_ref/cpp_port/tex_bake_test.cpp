// UV-atlas PBR bake validation on the golden mesh + golden PBR volume (no chain re-run).
//   xatlas unwrap -> rasterize -> grid_sample the per-voxel 6-ch PBR -> baseColor/metalRough atlas
//   -> textured GLB.  Writes debug PNGs + a textured .glb for visual validation.
//   ./build.sh tex_bake_test && ./tex_bake_test [texture_size]
#include "tex_atlas.hpp"
#include "glb_textured.hpp"
#include "sparse_vae.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>

static const char* GOLD = "../../sparse_spike/golden_stages/stage5_mesh";
static const char* REFS = "refs/stage4";
static double now(){ using namespace std::chrono; return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    int TS = (argc>1)? atoi(argv[1]) : 2048;
    int DECI = (argc>2)? atoi(argv[2]) : 0;   // decimate target faces (0=off)
    // A controlled bake must not overwrite the historical root-level oracle.
    const std::string outdir = getenv("OUT_DIR") ? getenv("OUT_DIR") : ".";
    NpyArray vN = npy_load(std::string(GOLD)+"/vertices.npy");        // [V,3] f32
    NpyArray fN = npy_load(std::string(GOLD)+"/faces.npy");           // [F,3] i64
    NpyArray pfN= npy_load(std::string(REFS)+"/tex_pbr.npy");         // [N,6] f32
    NpyArray pcN= npy_load(std::string(REFS)+"/tex_out_coords.npy");  // [N,4] i32
    int V=(int)vN.shape[0], F=(int)fN.shape[0], N=(int)pfN.shape[0];
    printf("[bake] mesh V=%d F=%d, PBR volume N=%d, texture_size=%d\n", V, F, N, TS);

    std::vector<float> verts(vN.f32(), vN.f32()+(size_t)V*3);
    std::vector<int64_t> faces(fN.i64(), fN.i64()+(size_t)F*3);
    // MESH_PLY=path → bake the golden PBR volume onto an arbitrary remesh output instead of the
    // golden mesh (same world frame). Lets us texture+render the coarse-solid remesh for QA.
    if (getenv("MESH_PLY")) {
        std::vector<float> pv; std::vector<int64_t> pf;
        if (!svae::read_ply(getenv("MESH_PLY"), pv, pf)) { printf("[bake] FAILED to read %s\n", getenv("MESH_PLY")); return 1; }
        verts = pv; faces = pf; V=(int)(pv.size()/3); F=(int)(pf.size()/3);
        printf("[bake] using MESH_PLY %s: V=%d F=%d\n", getenv("MESH_PLY"), V, F);
    }
    std::vector<float> pbr(pfN.f32(), pfN.f32()+(size_t)N*6);
    std::vector<int32_t> coords(pcN.i32(), pcN.i32()+(size_t)N*4);

    // A1: optional watertight hole-fill before unwrap (FILL_HOLES=1). Measures the chart collapse.
    if (getenv("FILL_HOLES")) {
        svae::Mesh m; m.verts = verts; m.faces = faces; m.N = V; m.F = F;
        double tf = now();
        int64_t b0 = svae::boundary_edge_count(m);
        int loops = svae::fill_holes(m, 4, true);
        int64_t b1 = svae::boundary_edge_count(m);
        printf("[fill] boundary edges %lld -> %lld, loops=%d, faces %d -> %d (%.1fs)\n",
               (long long)b0, (long long)b1, loops, F, m.F, now()-tf);
        faces = m.faces;
    }

    double t0=now();
    int fbr = getenv("SAMPLE_FALLBACK_R") ? atoi(getenv("SAMPLE_FALLBACK_R")) : 0;
    bool precl = getenv("PRECLUSTER") != nullptr;
    float cone = getenv("ATL_CONE") ? atof(getenv("ATL_CONE")) : 40.f;
    int pad = getenv("ATL_PAD") ? atoi(getenv("ATL_PAD")) : 4;
    texatlas::BakedTexture bt = texatlas::bake(verts, faces, pbr, coords, /*grid_res*/1024, TS, DECI,
                                               pad, /*verbose*/true, fbr, precl, cone);
    printf("[bake] done in %.1fs  (atlas %dx%d, %d out-verts)\n", now()-t0, bt.tw, bt.th, (int)bt.verts.size()/3);

    // debug PNGs
    glb::encode_png(bt.base_color.data(), bt.tw, bt.th, 4);  // (warm)
    const std::string base_png = outdir + "/tex_base_color.png";
    const std::string metal_png = outdir + "/tex_metal_rough.png";
    const std::string glb_out = outdir + "/miku_uvatlas_native.glb";
    stbi_write_png(base_png.c_str(), bt.tw, bt.th, 4, bt.base_color.data(), bt.tw*4);
    stbi_write_png(metal_png.c_str(), bt.tw, bt.th, 3, bt.metal_rough.data(), bt.tw*3);
    bool ok = glb::write_glb_textured(glb_out.c_str(), bt.verts, bt.normals, bt.uvs, bt.faces,
                                      bt.base_color, bt.metal_rough, bt.tw, bt.th);
    printf("[bake] wrote %s (%s) + %s + %s\n", glb_out.c_str(), ok?"ok":"FAIL", base_png.c_str(), metal_png.c_str());
    return ok?0:1;
}
