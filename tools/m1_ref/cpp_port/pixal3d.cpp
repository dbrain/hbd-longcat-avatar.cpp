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
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const float DEF_CAM = 0.7332379387484828f, DEF_DIST = 1.3021559715270996f, DEF_MS = 1.0f;

static void usage() {
    printf("usage: pixal3d --model <gguf_dir> --image <png> --out <glb>\n"
           "               [--fov <deg>] [--cam <ang_rad> <dist> <scale>] [--tex] [--cpu] [--ply]\n"
           "       --tex : also run the texture branch (per-vertex base_color)\n");
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
    bool use_cuda = true, write_ply = false, textured = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i+1 < argc) model = argv[++i];
        else if (a == "--image" && i+1 < argc) image = argv[++i];
        else if (a == "--out" && i+1 < argc) out = argv[++i];
        else if (a == "--fov" && i+1 < argc) cam = std::atof(argv[++i]) * (float)M_PI / 180.0f;
        else if (a == "--cam" && i+3 < argc) { cam = std::atof(argv[++i]); dist = std::atof(argv[++i]); ms = std::atof(argv[++i]); }
        else if (a == "--tex") textured = true;
        else if (a == "--cpu") use_cuda = false;
        else if (a == "--ply") write_ply = true;
        else { printf("unknown/incomplete arg: %s\n", a.c_str()); usage(); return 1; }
    }
    if (model.empty() || image.empty() || out.empty()) { usage(); return 1; }

    setenv("PIXAL3D_GGUF_DIR", model.c_str(), 1);  // harness + WLoad load <model>/<m>.gguf
    printf("==== pixal3d ====\n  model: %s\n  image: %s\n  out:   %s\n  cam:   fov=%.4frad dist=%.4f scale=%.2f  backend=%s\n",
           model.c_str(), image.c_str(), out.c_str(), cam, dist, ms, use_cuda ? "cuda" : "cpu");

    pix::ChainInput in;
    try {
        in.img512_raw = imgio::load_chw(image, 512);
        in.img1024_raw = imgio::load_chw(image, 1024);
    } catch (const std::exception& e) { printf("image load failed: %s\n", e.what()); return 1; }
    in.cam = cam; in.dist = dist; in.ms = ms; in.use_cuda = use_cuda; in.verbose = true; in.textured = textured;
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
    std::vector<float> vcolors;
    svae::Mesh mesh = pix::run_geometry(in, &st, textured ? &vcolors : nullptr);

    if (!glb::write_glb(out.c_str(), mesh.verts, mesh.faces, textured ? &vcolors : nullptr)) { printf("glb write failed: %s\n", out.c_str()); return 1; }
    if (write_ply) { std::string ply = out.substr(0, out.find_last_of('.')) + ".ply";
                     svae::write_ply(ply.c_str(), mesh.verts, mesh.faces); printf("  wrote %s\n", ply.c_str()); }
    printf("==== DONE  N1=%d M=%d verts=%d faces=%d  %.1fs  -> %s ====\n",
           st.N1, st.M, mesh.N, mesh.F, st.secs, out.c_str());
    return 0;
}
