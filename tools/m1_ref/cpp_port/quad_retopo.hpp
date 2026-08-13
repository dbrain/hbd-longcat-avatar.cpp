// quad_retopo.hpp — RUNG 2: quad retopology stage via quadwild-bimdf (Gurobi-free Bi-MDF/libSatsuma).
// First cut = shell out to the two validated CLIs (QUADWILD-INTEGRATION-RECIPE.md §5a): a CLI-only tool
// that round-trips OBJ + intermediate field/patch files, so shell-out is the honest first integration.
// NO linking / NO build.sh change — just this header + std::system(). CPU-only, zero VRAM.
//
//   tri svae::Mesh  --write OBJ-->  quadwild (remesh+crossfield+trace)  -->  quad_from_patches (Bi-MDF)
//     --> *_quadrangulation_smooth.obj (100% quads)  --read+triangulate-->  clean field-aligned tri Mesh
//
// Feed it the UltraShape REFINED mesh (~15k–160k v ok; the 974k COARSE mesh stalls patch-tracing — never
// use coarse). Output ~6k–60k quads (density scales with input; tune via basic_setup.txt scaleFact later).
#pragma once
#include "sparse_vae.hpp"          // svae::Mesh
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sys/stat.h>

namespace qr {

struct QuadCfg {
    std::string repo = "/home/dbrain/dev/quadwild-bimdf";   // built binaries in build/Build/bin/
    std::string tmp  = "/tmp/qr_stage";                     // scratch dir for OBJ round-trip
    std::string prep_cfg = "config/prep_config/basic_setup.txt";        // relative to repo
    std::string flow_cfg = "config/main_config/flow_noalign_lemon.txt";  // relative to repo (Gurobi-free)
    bool verbose = true;
};

// svae tri Mesh -> Wavefront OBJ (1-based faces). Only v + f lines (quadwild ignores the rest).
static inline bool write_obj(const std::string& path, const std::vector<float>& v, const std::vector<int64_t>& f) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    for (size_t i = 0; i + 2 < v.size(); i += 3) fprintf(fp, "v %.7g %.7g %.7g\n", v[i], v[i+1], v[i+2]);
    for (size_t i = 0; i + 2 < f.size(); i += 3)
        fprintf(fp, "f %lld %lld %lld\n", (long long)f[i]+1, (long long)f[i+1]+1, (long long)f[i+2]+1);
    fclose(fp);
    return true;
}

// Read an OBJ that may contain quads (or tris). Returns verts + per-face 0-based index lists (any arity).
static inline bool read_obj_poly(const std::string& path, std::vector<float>& v, std::vector<std::vector<int>>& polys) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    v.clear(); polys.clear();
    char* line = nullptr; size_t n = 0; ssize_t r;
    while ((r = getline(&line, &n, fp)) != -1) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z; if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) { v.push_back(x); v.push_back(y); v.push_back(z); }
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::vector<int> face; const char* p = line + 1;
            while (*p) {
                while (*p == ' ' || *p == '\t') p++;
                if (*p < '0' || *p > '9') { if (*p == '\n' || *p == 0) break; p++; continue; }
                long idx = strtol(p, (char**)&p, 10);          // 1-based vertex index
                face.push_back((int)idx - 1);
                while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;   // skip /vt/vn
            }
            if (face.size() >= 3) polys.push_back(std::move(face));
        }
    }
    free(line); fclose(fp);
    return !v.empty() && !polys.empty();
}

// Run quadwild on `in` (tri). `out` = the field-aligned quad mesh TRIANGULATED (fan) for the tri-only
// downstream (bake/rig/glb are triangle pipelines). `out_quads` (optional) keeps the raw quad polygons.
// Returns false on any stage failure (missing repo/binary, CLI nonzero, unreadable output).
static inline bool quad_retopo(const svae::Mesh& in, const QuadCfg& cfg, svae::Mesh& out,
                               std::vector<std::vector<int>>* out_quads = nullptr) {
    mkdir(cfg.tmp.c_str(), 0755);
    const std::string in_obj  = cfg.tmp + "/qr_in.obj";
    const std::string rem_p0  = cfg.tmp + "/qr_in_rem_p0.obj";
    const std::string smooth  = cfg.tmp + "/qr_in_rem_p0_0_quadrangulation_smooth.obj";
    const std::string stats   = cfg.tmp + "/qr_stats.json";
    ::remove(rem_p0.c_str()); ::remove(smooth.c_str());        // stale-guard

    if (!write_obj(in_obj, in.verts, in.faces)) { printf("[quad] write OBJ failed: %s\n", in_obj.c_str()); return false; }

    // STEP 1: remesh + cross-field + patch tracing ('2' = stop after tracing). CWD must be the repo
    // (relative config paths); input/output are absolute (written next to the input, i.e. in cfg.tmp).
    std::string c1 = "cd '" + cfg.repo + "' && ./build/Build/bin/quadwild '" + in_obj + "' 2 '" + cfg.prep_cfg + "'"
                   + (cfg.verbose ? "" : " >/dev/null 2>&1");
    if (int rc = std::system(c1.c_str())) { printf("[quad] quadwild step1 rc=%d\n", rc); return false; }
    // STEP 2: Bi-MDF quantization + quad extraction -> *_quadrangulation_smooth.obj
    std::string c2 = "cd '" + cfg.repo + "' && ./build/Build/bin/quad_from_patches '" + rem_p0 + "' 0 '"
                   + cfg.flow_cfg + "' '" + stats + "'" + (cfg.verbose ? "" : " >/dev/null 2>&1");
    if (int rc = std::system(c2.c_str())) { printf("[quad] quad_from_patches step2 rc=%d\n", rc); return false; }

    std::vector<float> qv; std::vector<std::vector<int>> polys;
    if (!read_obj_poly(smooth, qv, polys)) { printf("[quad] cannot read quad output: %s\n", smooth.c_str()); return false; }
    if (out_quads) *out_quads = polys;

    out.verts = std::move(qv);
    out.faces.clear();
    int nquad = 0, ntri = 0, nother = 0;
    for (auto& f : polys) {
        if (f.size() == 4) { // quad -> two tris (0,1,2)+(0,2,3)
            out.faces.push_back(f[0]); out.faces.push_back(f[1]); out.faces.push_back(f[2]);
            out.faces.push_back(f[0]); out.faces.push_back(f[2]); out.faces.push_back(f[3]); nquad++;
        } else if (f.size() == 3) {
            out.faces.push_back(f[0]); out.faces.push_back(f[1]); out.faces.push_back(f[2]); ntri++;
        } else { // ngon fan
            for (size_t k = 1; k + 1 < f.size(); k++) { out.faces.push_back(f[0]); out.faces.push_back(f[k]); out.faces.push_back(f[k+1]); }
            nother++;
        }
    }
    out.N = (int)(out.verts.size()/3);
    out.F = (int)(out.faces.size()/3);
    if (cfg.verbose)
        printf("[quad] quadwild: %zu polys (%d quad / %d tri / %d ngon) -> %d v / %d tri faces\n",
               polys.size(), nquad, ntri, nother, out.N, out.F);
    return true;
}

}  // namespace qr
