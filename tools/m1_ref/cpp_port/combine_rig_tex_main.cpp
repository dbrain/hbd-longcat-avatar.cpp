// FINAL-ASSET GLUE — merge a TEXTURED source mesh with a SAMPLED rig into one textured+skinned GLB.
//
//   combine_rig_tex <textured.glb> <rig_dir> <out.glb>
//        [--sampled verts.npy] [--skin skin_pred.npy] [--joints gen_joints.npy] [--parents gen_parents.npy] [--k N]
//
// Positional:
//   textured.glb    a TEXTURED source mesh (POSITION/NORMAL/TEXCOORD_0 + an embedded baseColor PNG),
//                   read via glb_reader.hpp (verts/faces/uvs) + glb::read_glb_basecolor_png (image bytes).
//   rig_dir         directory holding the SAMPLED rig npys. Default file names:
//                     vertices.npy   [Ns,3]  sampled point positions
//                     skin_pred.npy  [Ns,J]  per-sample skin-weight rows
//                     gen_joints.npy [J,3]   world joints
//                     gen_parents.npy[J]     int64 parents (root=-1)
//   out.glb         output: a TEXTURED + SKINNED GLB (write_rigged_textured_glb).
//
// Pipeline: kNN-transfer the sampled skin weights onto EVERY textured-mesh vertex
// (rig_transfer.hpp), then write the combined asset carrying the source UV + baseColor image.
//
// Pure CPU. Header-only deps: glb_reader.hpp, glb_rigged_textured.hpp (+ glb_rigged/glb_writer),
// rig_transfer.hpp. No ggml, no CUDA, no torch. Build: ./build.sh combine_rig_tex_main
#include "glb_reader.hpp"
#include "glb_rigged_textured.hpp"
#include "glb_writer.hpp"
#include "rig_transfer.hpp"
#include "rig_bone_names.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

// --------------------------------------------------------------------------
// Tiny .npy reader ('<f4' C-order float32 / '<i8' C-order int64). Mirrors rig_transfer_main.cpp.
// --------------------------------------------------------------------------
static bool npy_header(FILE* f, std::string& dtype, std::vector<int64_t>& shape, size_t& data_off) {
    unsigned char magic[8];
    if (std::fread(magic, 1, 8, f) != 8) return false;
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    int ver_major = magic[6];
    uint32_t hlen;
    if (ver_major == 1) {
        uint16_t h16; if (std::fread(&h16, 2, 1, f) != 1) return false; hlen = h16; data_off = 10 + hlen;
    } else {
        uint32_t h32; if (std::fread(&h32, 4, 1, f) != 1) return false; hlen = h32; data_off = 12 + hlen;
    }
    std::string hdr(hlen, '\0');
    if (std::fread(&hdr[0], 1, hlen, f) != hlen) return false;
    auto dpos = hdr.find("'descr'");
    if (dpos == std::string::npos) return false;
    auto q1 = hdr.find('\'', hdr.find(':', dpos) + 1);
    auto q2 = hdr.find('\'', q1 + 1);
    dtype = hdr.substr(q1 + 1, q2 - q1 - 1);
    auto spos = hdr.find("'shape'");
    auto p1 = hdr.find('(', spos);
    auto p2 = hdr.find(')', p1);
    std::string sh = hdr.substr(p1 + 1, p2 - p1 - 1);
    shape.clear();
    size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) i++;
        if (i >= sh.size()) break;
        char* end = nullptr;
        long v = std::strtol(sh.c_str() + i, &end, 10);
        if (end == sh.c_str() + i) break;
        shape.push_back((int64_t)v);
        i = (size_t)(end - sh.c_str());
    }
    return true;
}
static bool load_npy_f32(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) { std::fprintf(stderr, "npy: cannot open %s\n", path.c_str()); return false; }
    std::string dtype; size_t off;
    if (!npy_header(f, dtype, shape, off)) { std::fclose(f); std::fprintf(stderr, "npy: bad header %s\n", path.c_str()); return false; }
    if (dtype != "<f4") { std::fprintf(stderr, "npy %s: dtype %s != <f4\n", path.c_str(), dtype.c_str()); std::fclose(f); return false; }
    size_t n = 1; for (int64_t d : shape) n *= (size_t)d;
    out.resize(n);
    std::fseek(f, (long)off, SEEK_SET);
    bool ok = std::fread(out.data(), 4, n, f) == n;
    std::fclose(f);
    return ok;
}
static bool load_npy_i64(const std::string& path, std::vector<int64_t>& out, std::vector<int64_t>& shape) {
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) { std::fprintf(stderr, "npy: cannot open %s\n", path.c_str()); return false; }
    std::string dtype; size_t off;
    if (!npy_header(f, dtype, shape, off)) { std::fclose(f); std::fprintf(stderr, "npy: bad header %s\n", path.c_str()); return false; }
    if (dtype != "<i8") { std::fprintf(stderr, "npy %s: dtype %s != <i8\n", path.c_str(), dtype.c_str()); std::fclose(f); return false; }
    size_t n = 1; for (int64_t d : shape) n *= (size_t)d;
    out.resize(n);
    std::fseek(f, (long)off, SEEK_SET);
    bool ok = std::fread(out.data(), 8, n, f) == n;
    std::fclose(f);
    return ok;
}
static long file_size(const char* p) {
    FILE* f = std::fopen(p, "rb"); if (!f) return -1;
    std::fseek(f, 0, SEEK_END); long s = std::ftell(f); std::fclose(f); return s;
}
static void print_bbox(const char* tag, const rig::BBox& b) {
    std::printf("  %s bbox: [%.4g,%.4g,%.4g] .. [%.4g,%.4g,%.4g]  diag=%.4g\n",
                tag, b.mn[0], b.mn[1], b.mn[2], b.mx[0], b.mx[1], b.mx[2], b.diag());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <textured.glb> <rig_dir> <out.glb>\n"
            "          [--sampled verts.npy] [--skin skin_pred.npy]\n"
            "          [--joints gen_joints.npy] [--parents gen_parents.npy] [--k N]\n",
            argv[0]);
        return 2;
    }
    const std::string tex_path = argv[1];
    const std::string dir      = argv[2];
    const std::string out_path = argv[3];

    std::string verts_npy   = dir + "/vertices.npy";
    std::string skin_npy    = dir + "/skin_pred.npy";
    std::string joints_npy  = dir + "/gen_joints.npy";
    std::string parents_npy = dir + "/gen_parents.npy";
    int k = 4;

    for (int a = 4; a <= argc - 1; a++) {
        std::string f = argv[a];
        auto next = [&]() -> std::string { return (a + 1 < argc) ? std::string(argv[++a]) : std::string(); };
        if      (f == "--sampled") verts_npy   = next();
        else if (f == "--verts")   verts_npy   = next();
        else if (f == "--skin")    skin_npy    = next();
        else if (f == "--joints")  joints_npy  = next();
        else if (f == "--parents") parents_npy = next();
        else if (f == "--k")       k = std::atoi(next().c_str());
        else std::fprintf(stderr, "warn: ignoring unknown arg '%s'\n", f.c_str());
    }
    if (k < 1) k = 1;

    std::printf("== combine rig + texture ==\n");
    std::printf("  textured src : %s\n", tex_path.c_str());
    std::printf("  rig          : verts=%s skin=%s joints=%s parents=%s\n",
                verts_npy.c_str(), skin_npy.c_str(), joints_npy.c_str(), parents_npy.c_str());
    std::printf("  k            : %d\n", k);

    // --- load the sampled rig ---
    std::vector<float>   src_pts, src_w, joints;
    std::vector<int64_t> parents_i64, sh;
    if (!load_npy_f32(verts_npy, src_pts, sh)) return 1;
    if (sh.size() != 2 || sh[1] != 3) { std::fprintf(stderr, "verts npy shape not [Ns,3]\n"); return 1; }
    int64_t Ns = sh[0];
    if (!load_npy_f32(skin_npy, src_w, sh)) return 1;
    if (sh.size() != 2 || sh[0] != Ns) { std::fprintf(stderr, "skin npy shape not [Ns,J] (Ns mismatch)\n"); return 1; }
    int J = (int)sh[1];
    if (!load_npy_f32(joints_npy, joints, sh)) return 1;
    if (sh.size() != 2 || sh[1] != 3) { std::fprintf(stderr, "joints npy shape not [J,3]\n"); return 1; }
    int Jj = (int)sh[0];
    if (Jj != J) { std::fprintf(stderr, "joint count mismatch: skin J=%d vs joints J=%d\n", J, Jj); return 1; }
    if (!load_npy_i64(parents_npy, parents_i64, sh)) return 1;
    if ((int)parents_i64.size() != J) { std::fprintf(stderr, "parents count %zu != J=%d\n", parents_i64.size(), J); return 1; }
    std::vector<int> parents(parents_i64.begin(), parents_i64.end());

    // --- load the TEXTURED full mesh (geometry + uvs) ---
    glb::Mesh mesh;
    if (!glb::read_glb(tex_path.c_str(), mesh)) { std::fprintf(stderr, "failed to read %s\n", tex_path.c_str()); return 1; }
    int64_t Nd = (int64_t)(mesh.verts.size() / 3);
    int64_t F  = (int64_t)(mesh.faces.size() / 3);
    bool have_nrm = mesh.normals.size() == mesh.verts.size();
    bool have_uv  = mesh.uvs.size() == (size_t)Nd * 2;
    if (!have_uv) {
        std::fprintf(stderr, "ERROR: source glb has no TEXCOORD_0 (uvs) — cannot carry a texture.\n");
        return 1;
    }

    // --- extract the source baseColor image bytes ---
    std::vector<uint8_t> tex_png;
    std::string tex_mime;
    bool have_tex = glb::read_glb_basecolor_png(tex_path.c_str(), tex_png, tex_mime);
    if (!have_tex) {
        std::fprintf(stderr, "ERROR: no embedded baseColor image found in %s — cannot build a textured asset.\n", tex_path.c_str());
        return 1;
    }

    std::printf("  Nd=%lld  F=%lld  J=%d  Ns=%lld  (normals=%s, uvs=yes)\n",
                (long long)Nd, (long long)F, J, (long long)Ns, have_nrm ? "yes" : "computed");
    std::printf("  texture found: YES  mime=%s  bytes=%zu\n", tex_mime.c_str(), tex_png.size());

    // --- NORMALIZE the source mesh into the rig's frame (mesh_sample.hpp normalize_mesh: center +
    //     1/max|v-c|), so the rig (joints/skin sampled in normalized [-1,1] space) and the full mesh
    //     share ONE frame for the kNN transfer AND the output (joints already normalized). The texture
    //     UV is per-vertex and frame-independent. We write the mesh in this normalized frame so the
    //     skeleton lines up with the geometry. ---
    {
        double c[3] = {0,0,0}, mn[3]={1e30,1e30,1e30}, mx[3]={-1e30,-1e30,-1e30};
        for (int64_t i=0;i<Nd;i++) for(int d=0;d<3;d++){ float v=mesh.verts[i*3+d]; mn[d]=std::min(mn[d],(double)v); mx[d]=std::max(mx[d],(double)v);}
        for(int d=0;d<3;d++) c[d]=(mn[d]+mx[d])*0.5;
        double s=0; for (int64_t i=0;i<Nd;i++) for(int d=0;d<3;d++) s=std::max(s,std::fabs((double)mesh.verts[i*3+d]-c[d]));
        double inv = s>0 ? 1.0/s : 1.0;
        for (int64_t i=0;i<Nd;i++) for(int d=0;d<3;d++) mesh.verts[i*3+d]=(float)(((double)mesh.verts[i*3+d]-c[d])*inv);
        std::printf("  normalized mesh into rig frame (center [%.3g,%.3g,%.3g] scale %.4g)\n", c[0],c[1],c[2],inv);
    }

    // --- transfer skin weights sampled -> full textured mesh ---
    std::vector<float> dst_w;
    rig::BBox sbb, dbb;
    auto t0 = std::chrono::steady_clock::now();
    rig::transfer_skin(src_pts, src_w, J, mesh.verts, dst_w, k, &sbb, &dbb);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    print_bbox("rig", sbb);
    print_bbox("mesh", dbb);
    std::printf("  transfer time: %.3f s\n", secs);

    // --- skin sanity: mean over verts of the max single-joint weight ---
    double sum_maxw = 0.0;
    #pragma omp parallel for reduction(+:sum_maxw) schedule(static)
    for (long long v = 0; v < (long long)Nd; v++) {
        const float* row = &dst_w[(size_t)v * J];
        float mw = 0.f;
        for (int j = 0; j < J; j++) if (row[j] > mw) mw = row[j];
        sum_maxw += mw;
    }
    double mean_maxw = Nd > 0 ? sum_maxw / (double)Nd : 0.0;
    std::printf("  skin sanity: mean(max single-joint weight) = %.4f\n", mean_maxw);
    {
        float ddiag = std::max(1e-9f, dbb.diag());
        float scale_ratio = sbb.diag() / ddiag;
        std::printf("  frame check: scale ratio (rig/mesh) = %.3f%s\n", scale_ratio,
                    (scale_ratio < 0.5f || scale_ratio > 2.0f)
                      ? "  (NOTE: stand-in rig — geometric mismatch expected)" : "");
    }

    // --- derive standard humanoid names from the generated rest skeleton ---
    // SkinTokens joint indices are mesh-dependent, so an index table would be
    // a silent retargeting bug.  The naming pass uses the tree + rest pose and
    // falsifies its own anatomical/chirality assumptions before we promise a
    // Mixamo/Hymotion hand-off.
    rig::NameOpts name_opts;
    if (const char* facing = std::getenv("RIG_BONE_FACING")) {
        if (std::strcmp(facing, "+z") == 0) name_opts.facing_override = +1;
        else if (std::strcmp(facing, "-z") == 0) name_opts.facing_override = -1;
        else { std::fprintf(stderr, "bad RIG_BONE_FACING '%s' (expected +z or -z)\n", facing); return 2; }
    }
    rig::BoneNaming named = rig::name_bones(joints, parents, name_opts);
    if (!named.ok) { std::fprintf(stderr, "bone naming failed: %s\n", named.fail_reason.c_str()); return 1; }
    if (rig::falsify_bone_names(joints, parents, named, true) != 0) {
        std::fprintf(stderr, "bone naming falsifier failed; refusing ambiguous retarget asset\n"); return 1;
    }

    // --- write the combined TEXTURED + RIGGED glb ---
    bool ok = glb::write_rigged_textured_glb(
        out_path.c_str(), mesh.verts, mesh.faces, mesh.uvs, joints, parents, dst_w,
        have_nrm ? &mesh.normals : nullptr,
        tex_png.data(), tex_png.size(), tex_mime.c_str(), &named.names);
    if (!ok) { std::fprintf(stderr, "write_rigged_textured_glb failed\n"); return 1; }
    long osz = file_size(out_path.c_str());
    std::printf("  wrote %s (%ld bytes, %.1f MB)\n", out_path.c_str(), osz, osz / 1e6);
    std::printf("OK\n");
    return 0;
}
