// Native CuMesh feature-preserving cleanup probe.
//
// Usage: cumesh_clean_glb <input.glb> <output.glb> [max_hole_perimeter=0.03]
// This is deliberately a geometry-only diagnostic.  It proves whether the native cleanup can retain
// the sharp Pixal dual-grid surface while satisfying the same topology gate as the production runner.
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include "native_cumesh_bridge.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s <input.glb> <output.glb> [max_hole_perimeter=0.03]\n", argv[0]);
        return 2;
    }
    const float perimeter = argc == 4 ? std::strtof(argv[3], nullptr) : 3e-2f;
    glb::Mesh in;
    if (!glb::read_glb(argv[1], in)) {
        std::fprintf(stderr, "could not read %s\n", argv[1]);
        return 1;
    }
    std::vector<float> verts;
    std::vector<int64_t> faces;
    native_cumesh::CleanReport report;
    try {
        native_cumesh::clean_feature_preserving(in.verts, in.faces, verts, faces, &report, perimeter);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native CuMesh cleanup failed: %s\n", e.what());
        return 1;
    }
    if (!glb::write_glb(argv[2], verts, faces)) {
        std::fprintf(stderr, "could not write %s\n", argv[2]);
        return 1;
    }
    std::printf("native CuMesh feature cleanup: V %d -> %d, F %d -> %d, output=%s\n",
                report.input_vertices, report.output_vertices,
                report.input_faces, report.output_faces, argv[2]);
    return 0;
}
