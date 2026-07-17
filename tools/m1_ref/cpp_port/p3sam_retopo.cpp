// p3sam_retopo — standalone native finger-preserving retopo: dense GLB ->
// P3-SAM part segmentation (native C++/ggml-CPU) -> per-part region-adaptive
// decimation (ppd::per_part_decimate, hands kept dense) -> decimated GLB.
// Also writes a face-colored segmentation GLB for inspection.
//
// Build: g++ -O3 -fopenmp -std=c++17 -march=native -ffast-math p3sam_retopo.cpp -o p3sam_retopo
// Run:   ./p3sam_retopo <dense.glb> <out_decimated.glb> [weights_dir] [obj_decimate_path] [seg_out.glb]
#include "p3sam_segment.hpp"
#include "per_part_decimate.hpp"
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <ctime>

static double now_s() { return (double)clock() / CLOCKS_PER_SEC; }

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <dense.glb> <out_decimated.glb> [weights_dir] [obj_decimate] [seg_out.glb]\n", argv[0]);
        return 1;
    }
    std::string in = argv[1], out = argv[2];
    std::string wdir = argc > 3 ? argv[3] : "/mnt/hdd/3d/avatar-shootout/p3sam_goldens/weights_npy";
    std::string objdec = argc > 4 ? argv[4] : "./obj_decimate";
    std::string segout = argc > 5 ? argv[5] : "";

    glb::Mesh m;
    if (!glb::read_glb(in.c_str(), m)) { printf("FAIL: read %s\n", in.c_str()); return 1; }
    int V = (int)m.verts.size() / 3, F = (int)m.faces.size() / 3;
    printf("[retopo] mesh V=%d F=%d\n", V, F);

    double t0 = now_s();
    auto face_ids = p3sam::segment_mesh(m.verts.data(), V, m.faces.data(), F, wdir, 42);
    printf("[retopo] segment_mesh: %zu faces labelled, wall(cpu-approx)=%.0fs\n", face_ids.size(), now_s() - t0);
    // part histogram
    std::unordered_map<int64_t, int64_t> hist;
    for (auto l : face_ids) hist[l]++;
    printf("[retopo] %zu parts:", hist.size());
    for (auto& kv : hist) printf(" %lld:%lld", (long long)kv.first, (long long)kv.second);
    printf("\n");

    // optional: write face-colored segmentation GLB (per-part random color)
    if (!segout.empty()) {
        std::vector<float> sv; std::vector<int64_t> sf; std::vector<float> vc;
        // duplicate verts per face so each face gets a flat color (inspection only)
        std::unordered_map<int64_t, std::array<float,3>> cmap;
        auto col = [&](int64_t l) -> std::array<float,3> {
            auto it = cmap.find(l); if (it != cmap.end()) return it->second;
            uint64_t h = (uint64_t)(l * 2654435761u + 1013904223u);
            std::array<float,3> c = { (40+h%200)/255.f, (40+(h>>8)%200)/255.f, (40+(h>>16)%200)/255.f };
            cmap[l] = c; return c;
        };
        sv.reserve((size_t)F*9); sf.reserve((size_t)F*3); vc.reserve((size_t)F*9);
        for (int f = 0; f < F; f++) {
            auto c = col(face_ids[f]);
            for (int j = 0; j < 3; j++) {
                int64_t vi = m.faces[f*3+j];
                int64_t nv = (int64_t)sv.size()/3;
                sv.push_back(m.verts[vi*3]); sv.push_back(m.verts[vi*3+1]); sv.push_back(m.verts[vi*3+2]);
                vc.push_back(c[0]); vc.push_back(c[1]); vc.push_back(c[2]);
                sf.push_back(nv);
            }
        }
        glb::write_glb(segout.c_str(), sv, sf, &vc);
        printf("[retopo] wrote segmentation -> %s\n", segout.c_str());
    }

    glb::Mesh dec; std::vector<ppd::PartReport> rep;
    if (!ppd::per_part_decimate(m, face_ids, objdec, dec, rep)) { printf("FAIL: per_part_decimate\n"); return 1; }
    int Fo = (int)dec.faces.size() / 3;
    printf("[retopo] decimated F=%d (from %d)\n", Fo, F);
    if (!glb::write_glb(out.c_str(), dec.verts, dec.faces)) { printf("FAIL: write %s\n", out.c_str()); return 1; }
    printf("[retopo] DONE -> %s\n", out.c_str());
    return 0;
}
