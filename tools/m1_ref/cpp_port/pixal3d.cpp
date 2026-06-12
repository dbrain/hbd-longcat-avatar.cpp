// pixal3d — the geometry CLI (Phase A): load a Pixal3D/TRELLIS.2 GGUF + an image, emit a GLB.
//   pixal3d --model <gguf_dir> --image <png> --out <glb> [--fov <deg>] [--cam <ang_rad> <dist> <scale>]
//           [--cpu] [--ply]
// The image is the preprocessed square matte (host rembg cut-line); raw photo -> matte is a host
// pre-step. Camera defaults to the miku cam.json (fov 42deg / dist 1.302 / scale 1.0); --fov or
// --cam override. Weights load from <gguf_dir>/<model>.gguf (set via PIXAL3D_GGUF_DIR). Output:
// untextured glTF 2.0 .glb (web-ready: normals + double-sided). M6 adds textures.
#include "pixal3d_chain.hpp"
#include "image_io.hpp"
#include "glb_writer.hpp"
#include "tex_atlas.hpp"
#include "glb_textured.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const float DEF_CAM = 0.7332379387484828f, DEF_DIST = 1.3021559715270996f, DEF_MS = 1.0f;

static void usage() {
    printf("usage: pixal3d --model <gguf_dir> --image <png> --out <glb>\n"
           "               [--fov <deg>] [--cam <ang_rad> <dist> <scale>] [--mesh-scale <s>]\n"
           "               [--tex] [--vcolor] [--texsize <N>] [--decimate <faces>]\n"
           "               [--remesh] [--no-watertight] [--cpu] [--ply] [--fast]\n"
           "               [--seed <N>] [--guidance <G>] [--steps <N>]\n"
           "               [--{ss,shape,tex}-guidance <G>] [--{ss,shape,tex}-rescale <R>]\n"
           "               [--{ss,shape,tex}-rescale-t <T>] [--{ss,shape,tex}-steps <N>]\n"
           "       --tex          : texture branch -> UV-atlas PBR (baseColor+metallicRoughness) GLB\n"
           "       --vcolor       : interim per-vertex COLOR_0 instead of the UV-atlas bake\n"
           "       --texsize <N>  : atlas resolution (default 2048)\n"
           "       --decimate <F> : downmesh to ~F faces (default 150000; 0=off=full mesh).\n"
           "                        Game assets want a low budget, e.g. --decimate 40000.\n"
           "       --remesh       : proper marching-tet MANIFOLD watertight remesh (no flaps; unblocks\n"
           "                        quality decimation + a tight atlas). Supersedes the --watertight hack.\n"
           "       --no-watertight: skip the interim dual-grid hole-fill (ignored when --remesh).\n"
           "       --seed <N>     : noise seed (default 42; varies the generation).\n"
           "       --guidance <G> : shorthand for --ss-guidance + --shape-guidance = \"how close to the\n"
           "                        image\" (CFG strength; default 7.5; higher = more faithful).\n"
           "       --steps <N>    : shorthand for all three --*-steps (default 12; more = finer/slower).\n"
           "       --<stage>-guidance/-rescale/-rescale-t/-steps : per-stage sampler knobs\n"
           "                        (stage = ss | shape | tex; defaults = inference.py run_inference).\n");
}

static std::vector<float> load_norm(const std::string& model, const char* which) {
    // shape_slat normalization (32-dim config const): prefer <model>/, fall back to refs/
    for (std::string p : {model + "/shape_slat_norm_" + which + ".npy",
                          std::string("refs/shape_slat_norm_") + which + ".npy"}) {
        FILE* f = fopen(p.c_str(), "rb");
        if (f) { fclose(f); NpyArray a = npy_load(p); return std::vector<float>(a.f32(), a.f32()+a.numel()); }
    }
    throw std::runtime_error(std::string("shape_slat_norm_") + which + " not found in model dir or refs/");
}

int main(int argc, char** argv) {
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);  // fp32 matmul (correctness-first); perf phase relaxes
    std::string model, image, out;
    float cam = DEF_CAM, dist = DEF_DIST, ms = DEF_MS;
    bool use_cuda = true, write_ply = false, textured = false, fast = false, vcolor = false, watertight = true, remesh = false;
    int texsize = 2048, decimate = -1;   // -1 = auto (remesh→90k tight/fast atlas, else 150k). dual-grid
                                          // mesh holes -> xatlas charts/boundary; keep
                                              // face count tractable (~100s unwrap). --decimate to tune.
    pix::ChainInput in;   // sampler conditioning parsed straight into its defaults (== inference.py)
    auto next = [&](int& i){ return (float)std::atof(argv[++i]); };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i+1 < argc) model = argv[++i];
        else if (a == "--image" && i+1 < argc) image = argv[++i];
        else if (a == "--out" && i+1 < argc) out = argv[++i];
        else if (a == "--fov" && i+1 < argc) cam = std::atof(argv[++i]) * (float)M_PI / 180.0f;
        else if (a == "--cam" && i+3 < argc) { cam = std::atof(argv[++i]); dist = std::atof(argv[++i]); ms = std::atof(argv[++i]); }
        else if (a == "--mesh-scale" && i+1 < argc) ms = std::atof(argv[++i]);
        else if (a == "--tex") textured = true;
        else if (a == "--vcolor") { textured = true; vcolor = true; }   // interim per-vertex COLOR_0
        else if (a == "--texsize" && i+1 < argc) texsize = std::atoi(argv[++i]);
        else if (a == "--decimate" && i+1 < argc) decimate = std::atoi(argv[++i]);
        else if (a == "--remesh") remesh = true;
        else if (a == "--no-watertight") watertight = false;
        else if (a == "--cpu") use_cuda = false;
        else if (a == "--ply") write_ply = true;
        else if (a == "--fast") fast = true;   // Phase C perf: f16 tensor-core DiT path (use w/ weights_gguf_f16)
        // ---- conditioning (Trellis.2 sampler knobs; defaults already set in ChainInput) ----
        else if (a == "--seed" && i+1 < argc) in.seed = std::atoi(argv[++i]);
        else if (a == "--guidance" && i+1 < argc) { float g=next(i); in.ss.guidance=g; in.shape.guidance=g; }
        else if (a == "--steps" && i+1 < argc) { int s=std::atoi(argv[++i]); in.ss.steps=in.shape.steps=in.tex.steps=s; }
        else if (a == "--ss-guidance"   && i+1 < argc) in.ss.guidance    = next(i);
        else if (a == "--ss-rescale"    && i+1 < argc) in.ss.rescale     = next(i);
        else if (a == "--ss-rescale-t"  && i+1 < argc) in.ss.rescale_t   = next(i);
        else if (a == "--ss-steps"      && i+1 < argc) in.ss.steps       = std::atoi(argv[++i]);
        else if (a == "--shape-guidance"&& i+1 < argc) in.shape.guidance = next(i);
        else if (a == "--shape-rescale" && i+1 < argc) in.shape.rescale  = next(i);
        else if (a == "--shape-rescale-t"&&i+1 < argc) in.shape.rescale_t= next(i);
        else if (a == "--shape-steps"   && i+1 < argc) in.shape.steps    = std::atoi(argv[++i]);
        else if (a == "--tex-guidance"  && i+1 < argc) in.tex.guidance   = next(i);
        else if (a == "--tex-rescale"   && i+1 < argc) in.tex.rescale    = next(i);
        else if (a == "--tex-rescale-t" && i+1 < argc) in.tex.rescale_t  = next(i);
        else if (a == "--tex-steps"     && i+1 < argc) in.tex.steps      = std::atoi(argv[++i]);
        else { printf("unknown/incomplete arg: %s\n", a.c_str()); usage(); return 1; }
    }
    // --fast = the validated near-lossless perf config (Phase C LAP 2/3): f16 DiT weights via fp16
    // tensor cores + f16 attention, with fp32 accumulation. Keeps NVIDIA_TF32_OVERRIDE=0 (f32 stages
    // stay bit-exact — the SS-DiT occupancy is sensitive). Use with --model weights_gguf_f16.
    if (fast) { setenv("PIXAL3D_FAST", "1", 1); setenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "1", 1); }
    if (model.empty() || image.empty() || out.empty()) { usage(); return 1; }

    setenv("PIXAL3D_GGUF_DIR", model.c_str(), 1);  // harness + WLoad load <model>/<m>.gguf
    printf("==== pixal3d ====\n  model: %s\n  image: %s\n  out:   %s\n  cam:   fov=%.4frad dist=%.4f scale=%.2f  backend=%s\n",
           model.c_str(), image.c_str(), out.c_str(), cam, dist, ms, use_cuda ? "cuda" : "cpu");

    try {
        in.img512_raw = imgio::load_chw(image, 512);
        in.img1024_raw = imgio::load_chw(image, 1024);
    } catch (const std::exception& e) { printf("image load failed: %s\n", e.what()); return 1; }
    in.cam = cam; in.dist = dist; in.ms = ms; in.use_cuda = use_cuda; in.verbose = true; in.textured = textured;
    in.watertight = watertight; in.remesh = remesh;
    // resolve auto decimate: the remesh (stride4) path quality-decimates to 150k to preserve
    // chin/finger detail (unwrap ~28s @ ~49% util); dual-grid stays 150k. Lower --decimate for a
    // smaller/faster atlas if detail isn't critical.
    if (decimate < 0) decimate = 150000;
    in.norm_mean = load_norm(model, "mean");
    in.norm_std  = load_norm(model, "std");
    if (textured) {
        // tex_slat normalization (config consts) — <model>/tex_slat_norm_* or refs/ fallback
        auto ld = [&](const char* w){ for (std::string p : {model+"/tex_slat_norm_"+w+".npy", std::string("refs/tex_slat_norm_")+w+".npy"}) {
            FILE* f=fopen(p.c_str(),"rb"); if(f){fclose(f); NpyArray a=npy_load(p); return std::vector<float>(a.f32(),a.f32()+a.numel());} }
            throw std::runtime_error(std::string("tex_slat_norm_")+w+" not found"); };
        in.tex_mean = ld("mean"); in.tex_std = ld("std");
    }

    pix::ChainStats st;
    std::vector<float> vcolors, pbr_feats;
    std::vector<int32_t> pbr_coords;
    // --remesh textured → PER-VERTEX colour by default: the QEM mesh's UV-atlas bake samples the teal
    // model INTERIOR at folded charts ("teal splattered everywhere"); per-vertex grid_sample at the
    // surface vertices is clean. UV-atlas remains opt-in (PIXAL3D_FORCE_UVATLAS) for non-remesh meshes
    // or once a manifold remesh lands. (Per-vertex loses the metallic/roughness maps — fine for now.)
    if (remesh && textured && !vcolor && !std::getenv("PIXAL3D_FORCE_UVATLAS")) vcolor = true;
    const bool uvatlas = textured && !vcolor;
    svae::Mesh mesh = pix::run_geometry(in, &st,
        (textured && vcolor) ? &vcolors : nullptr,
        uvatlas ? &pbr_feats : nullptr,
        uvatlas ? &pbr_coords : nullptr);

    // PIXAL3D_DUMP_BAKE: dump the ALIGNED bake inputs (decimated mesh + the PBR volume from the SAME
    // run) so the atlas/bake can be iterated offline (tex_bake_test) without misaligned golden data.
    if (uvatlas && std::getenv("PIXAL3D_DUMP_BAKE")) {
        auto sv=[](const char* p, const void* d, size_t n){ FILE* f=fopen(p,"wb"); if(f){fwrite(d,1,n,f);fclose(f);} };
        sv("dump_mesh_v.bin", mesh.verts.data(), mesh.verts.size()*4);
        sv("dump_mesh_f.bin", mesh.faces.data(), mesh.faces.size()*8);
        sv("dump_pbr_f.bin",  pbr_feats.data(),  pbr_feats.size()*4);
        sv("dump_pbr_c.bin",  pbr_coords.data(), pbr_coords.size()*4);
        FILE* m=fopen("dump_bake.txt","w"); if(m){ fprintf(m,"%zu %zu %zu\n", mesh.verts.size()/3, mesh.faces.size()/3, pbr_feats.size()/6); fclose(m);}
        printf("  [dump] bake inputs -> dump_*.bin (mesh %zu v / %zu f, pbr %zu)\n", mesh.verts.size()/3, mesh.faces.size()/3, pbr_feats.size()/6);
    }
    bool ok;
    if (uvatlas) {
        // UV-atlas PBR bake: xatlas unwrap (decimated) -> rasterize -> grid_sample the per-voxel
        // PBR volume -> baseColor + metallicRoughness atlases -> textured glTF.
        double tb = pix::now_s();
        // --remesh now QEM-decimates the dual-grid mesh IN THE CHAIN (feature-preserving, clean), so
        // the bake must NOT re-decimate (that would re-run meshopt sloppy → undo the clean QEM). Pass
        // deci=0 for remesh. The QEM vertices are the dual-grid QEF positions (≈ on the PBR shell), so
        // grid_sample mostly hits; a moderate nearest-voxel fallback still covers QEM's small in/out
        // displacement in flat/concave regions (skirt/underarm) so no black texels. Non-remesh path
        // keeps the auto decimate (dual mesh is full-res there).
        int fb_r = remesh ? 16 : 0;
        int bake_deci = remesh ? 0 : decimate;
        // remesh path: the QEM mesh has ~50k non-manifold edges that make xatlas ComputeCharts
        // segmentation hang (minutes); use our normal-cone PRE-CLUSTER + AddUvMesh (pack-only) instead.
        bool precluster = remesh && !std::getenv("PIXAL3D_NO_PRECLUSTER");
        texatlas::BakedTexture bt = texatlas::bake(mesh.verts, mesh.faces, pbr_feats, pbr_coords,
                                                   /*grid_res*/1024, texsize, bake_deci, 4, true, fb_r,
                                                   precluster);
        printf("  [tex] UV-atlas bake: %dx%d, %d charts, %d out-verts (%.1fs)\n",
               bt.tw, bt.th, bt.chart_count, (int)bt.verts.size()/3, pix::now_s()-tb);
        ok = glb::write_glb_textured(out.c_str(), bt.verts, bt.normals, bt.uvs, bt.faces,
                                     bt.base_color, bt.metal_rough, bt.tw, bt.th);
    } else if (!vcolor && decimate > 0 && (int)mesh.faces.size()/3 > decimate) {
        // plain GLB with downmesh: decimate to the configurable face budget (game assets). --vcolor
        // keeps the full mesh (its COLOR_0 is zipped 1:1 to the verts; decimation would break it).
        std::vector<float> dv; std::vector<int64_t> df;
        texatlas::decimate(mesh.verts, mesh.faces, (size_t)decimate, dv, df);
        ok = glb::write_glb(out.c_str(), dv, df, nullptr);
        mesh.verts = dv; mesh.faces = df; mesh.N = (int)dv.size()/3; mesh.F = (int)df.size()/3;
    } else {
        ok = glb::write_glb(out.c_str(), mesh.verts, mesh.faces, vcolor ? &vcolors : nullptr);
    }
    if (!ok) { printf("glb write failed: %s\n", out.c_str()); return 1; }
    if (write_ply) { std::string ply = out.substr(0, out.find_last_of('.')) + ".ply";
                     svae::write_ply(ply.c_str(), mesh.verts, mesh.faces); printf("  wrote %s\n", ply.c_str()); }
    printf("==== DONE  N1=%d M=%d verts=%d faces=%d  %.1fs  -> %s ====\n",
           st.N1, st.M, mesh.N, mesh.F, st.secs, out.c_str());
    return 0;
}
