// im_retopo.hpp — RETOPO via Instant Meshes (thirdparty/instant-meshes, instant_meshes_batch).
// Robust organic quad retopology: clean field-aligned flow on characters where quadwild's crossfield
// tears flat organic regions (quadwild was validated on a hard-edged toy robot). Triangulated on output
// for the tri-only downstream (bake/rig/glb). CPU-only, no VRAM, <1s on a 400k mesh (~45x quadwild).
//
// KEY (see [[project_image_to_rig_retopo_findings]]):
//   -A 0 (UNIFORM)  -> watertight, clean, but same density everywhere (thin fingers can under-resolve).
//   -A >0 (ADAPTIVE)-> keeps thin-feature density BUT leaves holes (density transitions tear gaps).
// We default to UNIFORM (watertight); per-part density is handled upstream (P3-SAM part split feeding
// im_retopo per part at a part-appropriate target). Feed a VERBATIM mesh — no dedup/merge (trimesh-style
// GLB merges corrupt topology into non-manifold edges that become IM holes).
#pragma once
#include "sparse_vae.hpp"   // svae::Mesh { std::vector<float> verts; std::vector<int64_t> faces; int N,F; }
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace imretopo {

struct ImCfg {
    // default: thirdparty binary relative to tools/m1_ref/cpp_port; IM_BIN env overrides.
    std::string bin = "../../../thirdparty/instant-meshes/build_headless/instant_meshes_batch";
    std::string tmp = "/tmp/im_retopo";
    int   target_verts = 0;     // -v (0 = IM default density); see IM_TARGET_VERTS env
    float adaptivity   = 0.0f;  // -A (0 = uniform = watertight; >0 = adaptive but holey)
    float crease_deg   = 30.f;  // -c (feature crease angle)
    bool  verbose      = true;
};

// verbatim svae tri Mesh -> Wavefront OBJ (v + f only, 1-based, EXACT — no dedup).
static inline bool write_obj_verbatim(const std::string& path, const svae::Mesh& m) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    const size_t nv = m.verts.size() / 3, nf = m.faces.size() / 3;
    for (size_t i = 0; i < nv; i++)
        std::fprintf(f, "v %.9g %.9g %.9g\n", m.verts[i*3], m.verts[i*3+1], m.verts[i*3+2]);
    for (size_t t = 0; t < nf; t++)
        std::fprintf(f, "f %lld %lld %lld\n",
                     (long long)(m.faces[t*3]+1), (long long)(m.faces[t*3+1]+1), (long long)(m.faces[t*3+2]+1));
    std::fclose(f);
    return true;
}

// Read an OBJ that may contain quads (or tris/ngons); triangulate (fan) into svae::Mesh.
static inline bool read_obj_tri(const std::string& path, svae::Mesh& out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    out.verts.clear(); out.faces.clear();
    char line[8192];
    std::vector<int> poly;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z; if (std::sscanf(line+2, "%f %f %f", &x, &y, &z) == 3) { out.verts.push_back(x); out.verts.push_back(y); out.verts.push_back(z); }
        } else if (line[0] == 'f' && line[1] == ' ') {
            poly.clear();
            const char* p = line + 2;
            while (*p) {
                while (*p == ' ' || *p == '\t') p++;
                if (*p == '\0' || *p == '\n') break;
                long idx = std::strtol(p, (char**)&p, 10);   // 1-based; skip /vt/vn
                while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
                if (idx != 0) poly.push_back((int)(idx > 0 ? idx - 1 : idx));
            }
            for (size_t k = 1; k + 1 < poly.size(); k++) {   // fan
                out.faces.push_back(poly[0]); out.faces.push_back(poly[k]); out.faces.push_back(poly[k+1]);
            }
        }
    }
    std::fclose(f);
    out.N = (int)(out.verts.size() / 3);
    out.F = (int)(out.faces.size() / 3);
    return out.N > 0 && out.F > 0;
}

// Retopo `in` (tri) via Instant Meshes -> `out` (triangulated quad mesh). false on any failure.
static inline bool im_retopo(const svae::Mesh& in, const ImCfg& cfg, svae::Mesh& out) {
    ::mkdir(cfg.tmp.c_str(), 0755);
    const std::string oin = cfg.tmp + "/im_in.obj", oout = cfg.tmp + "/im_out.obj";
    if (!write_obj_verbatim(oin, in)) { std::printf("[im-retopo] cannot write %s\n", oin.c_str()); return false; }
    ::remove(oout.c_str());

    const char* env_bin = std::getenv("IM_BIN");
    std::string bin = env_bin ? env_bin : cfg.bin;
    const char* env_tv = std::getenv("IM_TARGET_VERTS");
    int tv = env_tv ? std::atoi(env_tv) : cfg.target_verts;
    const char* env_a = std::getenv("IM_ADAPTIVITY");   // instant_meshes_batch also reads this; pass -A too
    float adapt = env_a ? (float)std::atof(env_a) : cfg.adaptivity;

    char cmd[2048];
    int n = std::snprintf(cmd, sizeof cmd, "'%s' -i '%s' -o '%s' -A %.3f -c %.1f -D",
                          bin.c_str(), oin.c_str(), oout.c_str(), adapt, cfg.crease_deg);
    if (tv > 0) n += std::snprintf(cmd + n, sizeof cmd - n, " -v %d", tv);
    if (!cfg.verbose) std::snprintf(cmd + n, sizeof cmd - n, " >/dev/null 2>&1");
    if (int rc = std::system(cmd)) { std::printf("[im-retopo] instant_meshes_batch rc=%d\n", rc); return false; }

    if (!read_obj_tri(oout, out)) { std::printf("[im-retopo] cannot read IM output %s\n", oout.c_str()); return false; }
    if (cfg.verbose) std::printf("[im-retopo] %d v / %d f -> %d v / %d f (A=%.2f)\n", in.N, in.F, out.N, out.F, adapt);
    return true;
}

}  // namespace imretopo
