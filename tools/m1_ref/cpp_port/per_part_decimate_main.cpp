// per_part_decimate_main — CLI for the C++ port of shootout/per_part_decimate.py.
//   per_part_decimate <mesh.glb> <face_ids.npy int[F]> <out.glb> [dump_dir]
// Reads a GLB (glb_reader) + per-FACE part labels (npy int32/int64), splits the mesh by label,
// decimates each part to a region-adaptive budget by shelling out to ./obj_decimate (same binary +
// LockBorder the python uses), recombines, and writes out.glb (glb_writer). Optional dump_dir writes
// dump_mesh_v.bin (f32) + dump_mesh_f.bin (i64) + dump_bake.txt (NV NF NP), matching the python.
//   build: ./build.sh per_part_decimate_main   (CPU g++ -O2 -std=c++17 -fopenmp)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "per_part_decimate.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: per_part_decimate <mesh.glb> <face_ids.npy> <out.glb> [dump_dir]\n");
        return 1;
    }
    const char* mesh_path = argv[1];
    const char* fid_path  = argv[2];
    const char* out_path  = argv[3];
    const char* dump_dir  = argc > 4 ? argv[4] : nullptr;

    // locate obj_decimate next to this binary (argv[0]'s dir), else cwd.
    std::string self = argv[0];
    std::string dir = ".";
    size_t slash = self.find_last_of('/');
    if (slash != std::string::npos) dir = self.substr(0, slash);
    std::string obj_decimate = dir + "/obj_decimate";

    glb::Mesh in;
    if (!glb::read_glb(mesh_path, in)) { std::fprintf(stderr, "failed to read mesh %s\n", mesh_path); return 1; }
    const int64_t F = (int64_t)in.faces.size() / 3;
    std::printf("F_in = %lld\n", (long long)F);

    std::vector<int64_t> fid;
    if (!ppd::read_npy_int(fid_path, fid)) { std::fprintf(stderr, "failed to read face_ids %s\n", fid_path); return 1; }
    std::printf("face_ids = %zu\n", fid.size());

    glb::Mesh out;
    std::vector<ppd::PartReport> reports;
    if (!ppd::per_part_decimate(in, fid, obj_decimate, out, reports)) {
        std::fprintf(stderr, "per_part_decimate failed\n"); return 1;
    }
    std::printf("n_parts = %zu\n", reports.size());

    const int64_t Fout = (int64_t)out.faces.size() / 3;
    std::printf("TOTAL %lld -> %lld faces  (%.1f%%)  -> %s\n",
                (long long)F, (long long)Fout, 100.0 * (double)Fout / (double)F, out_path);

    if (!glb::write_glb(out_path, out.verts, out.faces)) {
        std::fprintf(stderr, "failed to write %s\n", out_path); return 1;
    }

    if (dump_dir) {
        if (!ppd::dump_mesh(dump_dir, out)) { std::fprintf(stderr, "dump failed\n"); return 1; }
    }
    return 0;
}
