// R7 CLI — transfer the SAMPLED rig (joints+parents+skin weights) onto a high-res target
// mesh and write a rigged GLB.
//
//   rig_transfer <full_mesh.glb> <sampled_dir | npy...> <out_rigged.glb> [overrides]
//
// Positional:
//   full_mesh.glb   high-res target mesh (e.g. native_v7.glb), read via glb_reader.hpp.
//   sampled_dir     directory holding the SAMPLED rig npys. Default file names:
//                     vertices.npy   [Ns,3]  sampled point positions
//                     skin_pred.npy  [Ns,J]  per-sample skin-weight rows
//                     detok_joints.npy  [J,3] world joints
//                     detok_parents.npy [J]   int64 parents (root=-1)
//   out_rigged.glb  output rigged full-mesh GLB (write_rigged_glb).
//
// Optional --flag overrides (so the same tool serves golden + live e2e output):
//   --verts <f.npy>    --skin <f.npy>    --joints <f.npy>    --parents <i64.npy>    --k <int>
//   --normalize-target  apply mesh_sample_main's normalized rig frame to the target before transfer
//   (e.g. --joints <dir>/gen_joints.npy --parents <dir>/gen_parents.npy for live output)
//
// Pure CPU. Header-only deps: glb_reader.hpp, glb_rigged.hpp (+ glb_writer.hpp), rig_transfer.hpp.
// No ggml, no CUDA, no torch. Build: ./build.sh rig_transfer_main
#include "glb_reader.hpp"
#include "glb_rigged.hpp"
#include "glb_writer.hpp"
#include "rig_transfer.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

// --------------------------------------------------------------------------
// Tiny .npy reader (mirrors glb_rigged_test.cpp): '<f4' C-order float32 + '<i8'
// C-order int64 only. Returns flat data + shape.
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
    std::printf("  %s bbox: [%.4g,%.4g,%.4g] .. [%.4g,%.4g,%.4g]  ext=(%.4g,%.4g,%.4g) diag=%.4g\n",
                tag, b.mn[0], b.mn[1], b.mn[2], b.mx[0], b.mx[1], b.mx[2],
                b.ext(0), b.ext(1), b.ext(2), b.diag());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <full_mesh.glb> <sampled_dir> <out_rigged.glb>\n"
            "          [--verts f.npy] [--skin f.npy] [--joints f.npy] [--parents i64.npy] [--k N] [--normalize-target]\n",
            argv[0]);
        return 2;
    }
    const std::string mesh_path = argv[1];
    const std::string dir       = argv[2];
    const std::string out_path  = argv[3];

    // defaults relative to the sampled dir
    std::string verts_npy   = dir + "/vertices.npy";
    std::string skin_npy    = dir + "/skin_pred.npy";
    std::string joints_npy  = dir + "/detok_joints.npy";
    std::string parents_npy = dir + "/detok_parents.npy";
    int k = 4;
    bool normalize_target = false;

    for (int a = 4; a < argc; a++) {
        std::string f = argv[a];
        auto next = [&]() -> std::string { return (a + 1 < argc) ? std::string(argv[++a]) : std::string(); };
        if      (f == "--verts")   verts_npy   = next();
        else if (f == "--skin")    skin_npy    = next();
        else if (f == "--joints")  joints_npy  = next();
        else if (f == "--parents") parents_npy = next();
        else if (f == "--k")       k = std::atoi(next().c_str());
        else if (f == "--normalize-target") normalize_target = true;
        else std::fprintf(stderr, "warn: ignoring unknown arg '%s'\n", f.c_str());
    }
    if (k < 1) k = 1;

    std::printf("== R7 rig transfer ==\n");
    std::printf("  full mesh : %s\n", mesh_path.c_str());
    std::printf("  sampled   : verts=%s skin=%s joints=%s parents=%s\n",
                verts_npy.c_str(), skin_npy.c_str(), joints_npy.c_str(), parents_npy.c_str());
    std::printf("  k         : %d\n", k);

    // --- load the sampled rig ---
    std::vector<float>   src_pts, src_w, joints;
    std::vector<int64_t> parents_i64;
    std::vector<int64_t> sh;
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

    // --- load the full mesh ---
    glb::Mesh mesh;
    if (!glb::read_glb(mesh_path.c_str(), mesh)) { std::fprintf(stderr, "failed to read %s\n", mesh_path.c_str()); return 1; }
    int64_t Nd = (int64_t)(mesh.verts.size() / 3);
    int64_t F  = (int64_t)(mesh.faces.size() / 3);
    bool have_nrm = mesh.normals.size() == mesh.verts.size();

    std::printf("  Ns=%lld  Nd=%lld  J=%d  (full mesh F=%lld, normals=%s)\n",
                (long long)Ns, (long long)Nd, J, (long long)F, have_nrm ? "yes" : "computed");

    // mesh_sample_main always supplies points in its bbox-midpoint/global-max
    // normalized frame.  A source GLB may retain authoring units or a scene
    // transform, so an explicit transfer diagnostic needs this same transform
    // before kNN. Keep it opt-in: callers with already-normalized targets do
    // not silently alter their requested output frame.
    if (normalize_target) {
        double mn[3] = {1e30,1e30,1e30}, mx[3] = {-1e30,-1e30,-1e30}, c[3];
        for (int64_t i = 0; i < Nd; ++i) for (int d = 0; d < 3; ++d) {
            const double v = mesh.verts[(size_t)i * 3 + d];
            mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v);
        }
        for (int d = 0; d < 3; ++d) c[d] = 0.5 * (mn[d] + mx[d]);
        double radius = 0.0;
        for (int64_t i = 0; i < Nd; ++i) for (int d = 0; d < 3; ++d)
            radius = std::max(radius, std::fabs((double)mesh.verts[(size_t)i * 3 + d] - c[d]));
        const double inv = radius > 0.0 ? 1.0 / radius : 1.0;
        for (int64_t i = 0; i < Nd; ++i) for (int d = 0; d < 3; ++d) {
            float& value = mesh.verts[(size_t)i * 3 + d];
            value = (float)(((double)value - c[d]) * inv);
        }
        std::printf("  normalized target into sampled rig frame (center [%.3g,%.3g,%.3g] scale %.4g)\n",
                    c[0], c[1], c[2], inv);
    }

    // --- transfer ---
    std::vector<float> dst_w;
    rig::BBox sbb, dbb;
    auto t0 = std::chrono::steady_clock::now();
    rig::transfer_skin(src_pts, src_w, J, mesh.verts, dst_w, k, &sbb, &dbb);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    print_bbox("src", sbb);
    print_bbox("dst", dbb);
    std::printf("  transfer time: %.3f s  (%.2f Mverts/s)\n", secs, (double)Nd / 1e6 / std::max(1e-9, secs));

    // --- frame-mismatch heuristic: compare bbox centers (relative to dst diag) + extent ratio ---
    {
        float ddiag = std::max(1e-9f, dbb.diag());
        float center_off = 0.f;
        for (int d = 0; d < 3; d++) {
            float sc = 0.5f * (sbb.mn[d] + sbb.mx[d]);
            float dc = 0.5f * (dbb.mn[d] + dbb.mx[d]);
            float e = (sc - dc); center_off += e * e;
        }
        center_off = std::sqrt(center_off) / ddiag;
        float scale_ratio = sbb.diag() / ddiag;
        std::printf("  frame check: center offset = %.3f * dst-diag, scale ratio (src/dst) = %.3f\n",
                    center_off, scale_ratio);
        bool mismatch = center_off > 0.25f || scale_ratio < 0.5f || scale_ratio > 2.0f;
        if (mismatch) {
            std::printf("  *** WARNING: src/dst bounding boxes are POORLY ALIGNED — the sampled rig and the\n"
                        "      full mesh appear to be in different frames/scales. NN transfer will be SMEARED.\n"
                        "      Pick a full mesh matched to the sampled point cloud's frame, or align first.\n");
        } else {
            std::printf("  frames look aligned (transfer should be coherent).\n");
        }
    }

    // --- sanity stat: mean over dst verts of the max single-joint weight ---
    double sum_maxw = 0.0;
    #pragma omp parallel for reduction(+:sum_maxw) schedule(static)
    for (long long v = 0; v < (long long)Nd; v++) {
        const float* row = &dst_w[(size_t)v * J];
        float mx = 0.f;
        for (int j = 0; j < J; j++) if (row[j] > mx) mx = row[j];
        sum_maxw += mx;
    }
    double mean_maxw = Nd > 0 ? sum_maxw / (double)Nd : 0.0;
    std::printf("  sanity: mean(max single-joint weight) over dst = %.4f  (higher = more coherent)\n", mean_maxw);

    // --- write the rigged full-mesh GLB ---
    bool ok = glb::write_rigged_glb(out_path.c_str(), mesh.verts, mesh.faces, joints, parents, dst_w,
                                    have_nrm ? &mesh.normals : nullptr);
    if (!ok) { std::fprintf(stderr, "write_rigged_glb failed\n"); return 1; }
    long osz = file_size(out_path.c_str());
    std::printf("  wrote %s (%ld bytes, %.1f MB)\n", out_path.c_str(), osz, osz / 1e6);
    std::printf("OK\n");
    return 0;
}
