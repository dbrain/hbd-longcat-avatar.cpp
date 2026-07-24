// mesh_exact_clean_glb — apply the production exact-only cleanup to a GLB.
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include "mesh_exact_clean.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <input.glb> <output.glb>\n", argv[0]);
        return 2;
    }
    glb::Mesh mesh;
    if (!glb::read_glb(argv[1], mesh)) {
        std::fprintf(stderr, "could not read %s\n", argv[1]);
        return 1;
    }
    const mesh_exact_clean::Report r = mesh_exact_clean::clean(mesh.verts, mesh.faces);
    if (!glb::write_glb(argv[2], mesh.verts, mesh.faces)) {
        std::fprintf(stderr, "could not write %s\n", argv[2]);
        return 1;
    }
    std::printf("exact mesh cleanup: V %lld -> %lld (welded %lld), F %lld -> %lld (degenerate %lld, duplicate %lld)\n",
                (long long)r.input_vertices, (long long)r.output_vertices, (long long)r.welded_vertices,
                (long long)r.input_faces, (long long)r.output_faces,
                (long long)r.dropped_degenerate_faces, (long long)r.dropped_duplicate_faces);
    return 0;
}
