// CPU self-test for glb_reader.hpp: build a unit cube in memory, write it via glb::write_glb,
// read it back via glb::read_glb, assert verts/faces round-trip exactly.
//   ./glb_reader_test            # self-test, prints OK
//   ./glb_reader_test file.glb   # + smoke-test a real .glb (prints V/F/uv/bbox)
#include "glb_writer.hpp"
#include "glb_reader.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } } while (0)

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[2], "--extract-basecolor") == 0) {
        std::vector<uint8_t> png;
        std::string mime;
        if (!glb::read_glb_basecolor_png(argv[1], png, mime)) {
            std::fprintf(stderr, "extract: no embedded baseColor texture in '%s'\n", argv[1]);
            return 2;
        }
        FILE* f = std::fopen(argv[3], "wb");
        if (!f || std::fwrite(png.data(), 1, png.size(), f) != png.size()) {
            if (f) std::fclose(f);
            std::fprintf(stderr, "extract: cannot write '%s'\n", argv[3]);
            return 2;
        }
        std::fclose(f);
        std::printf("extract: wrote %zu bytes (%s) -> %s\n", png.size(), mime.c_str(), argv[3]);
        return 0;
    }
    // ---- unit cube: 8 verts, 12 triangles ----
    std::vector<float> verts = {
        0,0,0,  1,0,0,  1,1,0,  0,1,0,   // z=0 face
        0,0,1,  1,0,1,  1,1,1,  0,1,1,   // z=1 face
    };
    std::vector<int64_t> faces = {
        0,1,2,  0,2,3,    // bottom (z=0)
        4,6,5,  4,7,6,    // top (z=1)
        0,4,5,  0,5,1,    // front
        1,5,6,  1,6,2,    // right
        2,6,7,  2,7,3,    // back
        3,7,4,  3,4,0,    // left
    };
    CHECK(verts.size() == 24, "cube verts size");
    CHECK(faces.size() == 36, "cube faces size");

    const char* tmp = "/tmp/glb_reader_test_cube.glb";
    bool wok = glb::write_glb(tmp, verts, faces);
    CHECK(wok, "write_glb returned true");

    glb::Mesh m;
    bool rok = glb::read_glb(tmp, m);
    CHECK(rok, "read_glb returned true");

    // round-trip verts (positions must be bit-identical)
    CHECK(m.verts.size() == verts.size(), "verts count round-trip");
    if (m.verts.size() == verts.size()) {
        bool eq = true;
        for (size_t i = 0; i < verts.size(); i++)
            if (std::memcmp(&m.verts[i], &verts[i], sizeof(float)) != 0) { eq = false; break; }
        CHECK(eq, "verts bit-identical");
    }

    // round-trip faces (indices identical)
    CHECK(m.faces.size() == faces.size(), "faces count round-trip");
    if (m.faces.size() == faces.size()) {
        bool eq = true;
        for (size_t i = 0; i < faces.size(); i++)
            if (m.faces[i] != faces[i]) { eq = false; break; }
        CHECK(eq, "faces identical");
    }

    // writer bakes per-vertex normals -> reader should recover them
    CHECK(m.normals.size() == verts.size(), "normals recovered (NORMAL attribute)");

    // Node TRS is part of glTF placement, not mesh geometry. Verify the helper used by the
    // reader before exercising an externally transformed GLB in the optional smoke below.
    {
        glb::detail::JVal node; node.type = glb::detail::JVal::Obj;
        auto vec = [](std::initializer_list<float> values) {
            glb::detail::JVal a; a.type = glb::detail::JVal::Arr;
            for (float v : values) { glb::detail::JVal n; n.type = glb::detail::JVal::Num; n.num = v; a.arr.push_back(n); }
            return a;
        };
        node.obj["translation"] = vec({4.f, -2.f, 1.f});
        node.obj["scale"] = vec({2.f, 3.f, 4.f});
        glb::detail::Mat4 tr;
        CHECK(glb::detail::node_local_matrix(node, tr), "node TRS parsed");
        float p[3] = {1, 1, 1}, got[3]; glb::detail::mat4_point(tr, p, got);
        CHECK(std::fabs(got[0] - 6.f) < 1e-6f && std::fabs(got[1] - 1.f) < 1e-6f && std::fabs(got[2] - 5.f) < 1e-6f,
              "node TRS applied to POSITION");
        float n[3] = {0, 1, 0}, ngot[3];
        CHECK(glb::detail::mat4_normal(tr, n, ngot) && std::fabs(ngot[1] - 1.f) < 1e-6f,
              "node inverse-transpose applied to NORMAL");
        glb::detail::JVal rotated; rotated.type = glb::detail::JVal::Obj;
        rotated.obj["rotation"] = vec({0.f, 0.f, 0.70710678f, 0.70710678f});
        rotated.obj["scale"] = vec({2.f, 3.f, 4.f});
        glb::detail::Mat4 rot;
        CHECK(glb::detail::node_local_matrix(rotated, rot), "rotated node TRS parsed");
        float xnormal[3] = {1, 0, 0}, rotated_normal[3];
        CHECK(glb::detail::mat4_normal(rot, xnormal, rotated_normal) &&
              std::fabs(rotated_normal[0]) < 1e-5f && std::fabs(rotated_normal[1] - 1.f) < 1e-5f,
              "inverse-transpose preserves rotated normal direction");
    }

    std::printf("self-test: V=%zu F=%zu normals=%zu uvs=%zu\n",
                m.verts.size() / 3, m.faces.size() / 3, m.normals.size() / 3, m.uvs.size() / 2);

    // ---- optional real-file smoke test ----
    if (argc > 1) {
        glb::Mesh r;
        if (!glb::read_glb(argv[1], r)) {
            std::fprintf(stderr, "smoke: read_glb('%s') FAILED\n", argv[1]);
            return 2;
        }
        float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        size_t V = r.verts.size() / 3;
        for (size_t i = 0; i < V; i++)
            for (int d = 0; d < 3; d++) {
                float v = r.verts[i * 3 + d];
                if (v < mn[d]) mn[d] = v;
                if (v > mx[d]) mx[d] = v;
            }
        std::printf("smoke '%s': V=%zu F=%zu has_uv=%d has_normals=%d bbox=[%.4g,%.4g,%.4g]..[%.4g,%.4g,%.4g]\n",
                    argv[1], V, r.faces.size() / 3, r.uvs.empty() ? 0 : 1, r.normals.empty() ? 0 : 1,
                    mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
    }

    if (g_fail) { std::fprintf(stderr, "%d CHECK(s) FAILED\n", g_fail); return 1; }
    std::printf("OK\n");
    return 0;
}
