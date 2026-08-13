// Native P3-SAM inference-view builder.
//
// Drops explicitly selected P3-SAM face labels from a geometry GLB, compacts
// the retained surface, and writes an ordinary native GLB for mesh sampling.
// It deliberately makes no semantic claim about the selected regions; callers
// retain the label IDs as provenance and SkinTokens still generates the rig.
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include "../../sparse_spike/npy.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <p3sam.glb> <face_ids.npy> <out.glb> --drop <label> [label ...]\n", argv[0]);
        return 2;
    }
    const std::string mesh_path = argv[1], labels_path = argv[2], out_path = argv[3];
    if (std::string(argv[4]) != "--drop") { std::fprintf(stderr, "expected --drop\n"); return 2; }
    std::unordered_set<int64_t> drop;
    for (int i = 5; i < argc; ++i) {
        char* end = nullptr; const long long v = std::strtoll(argv[i], &end, 10);
        if (!end || *end) { std::fprintf(stderr, "invalid label: %s\n", argv[i]); return 2; }
        drop.insert((int64_t)v);
    }
    try {
        glb::Mesh in;
        if (!glb::read_glb(mesh_path.c_str(), in) || in.faces.empty() || in.verts.empty()) {
            std::fprintf(stderr, "cannot read triangle mesh: %s\n", mesh_path.c_str()); return 1;
        }
        const NpyArray ids = npy_load(labels_path);
        const size_t faces = in.faces.size() / 3;
        if (ids.descr != "<i8" || ids.shape.size() != 1 || (size_t)ids.numel() != faces) {
            std::fprintf(stderr, "face ids must be int64 [%zu]: %s\n", faces, labels_path.c_str()); return 1;
        }
        const int64_t* labels = ids.i64();
        std::vector<int64_t> remap(in.verts.size() / 3, -1), out_faces;
        std::vector<float> out_verts;
        out_faces.reserve(in.faces.size()); out_verts.reserve(in.verts.size());
        size_t removed = 0;
        for (size_t f = 0; f < faces; ++f) {
            if (drop.count(labels[f])) { ++removed; continue; }
            for (int k = 0; k < 3; ++k) {
                const int64_t old = in.faces[f * 3 + k];
                if (old < 0 || (size_t)old >= remap.size()) throw std::runtime_error("mesh index out of range");
                if (remap[(size_t)old] < 0) {
                    remap[(size_t)old] = (int64_t)(out_verts.size() / 3);
                    out_verts.insert(out_verts.end(), &in.verts[(size_t)old * 3], &in.verts[(size_t)old * 3 + 3]);
                }
                out_faces.push_back(remap[(size_t)old]);
            }
        }
        if (out_faces.empty() || removed == 0 || removed == faces) {
            std::fprintf(stderr, "drop labels must remove a non-empty proper subset of %zu faces\n", faces); return 1;
        }
        if (!glb::write_glb(out_path.c_str(), out_verts, out_faces)) {
            std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1;
        }
        std::printf("[p3sam-mask-native] dropped labels=%zu faces=%zu/%zu vertices=%zu -> %zu: %s\n",
                    drop.size(), removed, faces, in.verts.size() / 3, out_verts.size() / 3, out_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
    return 0;
}
