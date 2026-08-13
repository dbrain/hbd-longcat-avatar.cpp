// per_part_decimate.hpp — C++ port of shootout/per_part_decimate.py: per-part region-adaptive
// decimation of a P3-SAM-segmented mesh. Split the mesh by per-FACE part label, decimate EACH part
// to a region-importance budget (hands keep ~full, hair/torso crush hard), recombine into one mesh.
//
// The decimation itself is the SAME C++ obj_decimate binary the python shells out to (meshoptimizer
// feature-aware QEM with meshopt_SimplifyLockBorder — part cut-boundaries are pinned so the parts
// recombine seamlessly without cracks). We shell out per part (write part .obj, run ./obj_decimate,
// read result back) rather than calling qem.hpp in-process, because:
//   (1) the python reference uses meshoptimizer (NOT qem.hpp's sp4cerat scheme) — to match its
//       per-part face_out counts we must use the same algorithm/binary, and
//   (2) meshopt_SimplifyLockBorder gives the border-lock that keeps part boundaries seamless; the
//       obj_decimate binary already wires that flag (qem.hpp's border classification differs).
// The python only does split + recombine I/O; this header reproduces exactly that glue.
//
// Region table (KEEP fraction) + heuristic ported verbatim from tier_keep() in the python:
//   HAND 0.80, HAIR 0.025, FOOT 0.12, HEAD 0.15, LIMB 0.10, BODY 0.05.
// target_faces = max(400, round(part_faces * keep)); only decimated if part_faces > target*1.05.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include "glb_reader.hpp"
#include "glb_writer.hpp"

namespace ppd {

// minimal OBJ reader (v / f only; triangulates fans; handles negative + v/vt/vn indices).
inline bool read_obj(const char* path, glb::Mesh& out);

// ----------------------------------------------------------------------------
// self-contained .npy reader for a 1-D int array (face_ids[F]). Handles int32/int64
// (and uint variants), little-endian, C-contiguous. Returns false on anything else.
// ----------------------------------------------------------------------------
inline bool read_npy_int(const char* path, std::vector<int64_t>& out) {
    out.clear();
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "read_npy_int: cannot open '%s'\n", path); return false; }
    unsigned char magic[6];
    if (std::fread(magic, 1, 6, f) != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "read_npy_int: bad magic\n"); std::fclose(f); return false;
    }
    unsigned char ver[2];
    if (std::fread(ver, 1, 2, f) != 2) { std::fclose(f); return false; }
    uint32_t header_len = 0;
    if (ver[0] == 1) {                       // v1.0: 2-byte little-endian header length
        unsigned char hl[2];
        if (std::fread(hl, 1, 2, f) != 2) { std::fclose(f); return false; }
        header_len = (uint32_t)hl[0] | ((uint32_t)hl[1] << 8);
    } else {                                 // v2.0/3.0: 4-byte little-endian header length
        unsigned char hl[4];
        if (std::fread(hl, 1, 4, f) != 4) { std::fclose(f); return false; }
        header_len = (uint32_t)hl[0] | ((uint32_t)hl[1] << 8) |
                     ((uint32_t)hl[2] << 16) | ((uint32_t)hl[3] << 24);
    }
    std::string hdr(header_len, '\0');
    if (std::fread(&hdr[0], 1, header_len, f) != header_len) { std::fclose(f); return false; }

    // parse descr (e.g. '<i8','<i4','<u4','|i1') and shape from the dict-literal header.
    auto find_val = [&](const char* key) -> std::string {
        size_t k = hdr.find(key);
        if (k == std::string::npos) return std::string();
        k = hdr.find(':', k);
        if (k == std::string::npos) return std::string();
        size_t q1 = hdr.find('\'', k);
        if (q1 == std::string::npos) return std::string();
        size_t q2 = hdr.find('\'', q1 + 1);
        if (q2 == std::string::npos) return std::string();
        return hdr.substr(q1 + 1, q2 - q1 - 1);
    };
    std::string descr = find_val("descr");
    if (descr.empty()) { std::fprintf(stderr, "read_npy_int: no descr\n"); std::fclose(f); return false; }

    bool fortran = hdr.find("'fortran_order': True") != std::string::npos;
    if (fortran) { std::fprintf(stderr, "read_npy_int: fortran_order unsupported\n"); std::fclose(f); return false; }

    // count = product of shape dims
    size_t sp = hdr.find("'shape'");
    size_t lp = hdr.find('(', sp), rp = hdr.find(')', lp);
    if (sp == std::string::npos || lp == std::string::npos || rp == std::string::npos) {
        std::fprintf(stderr, "read_npy_int: no shape\n"); std::fclose(f); return false;
    }
    std::string shp = hdr.substr(lp + 1, rp - lp - 1);
    int64_t count = 1; bool any = false;
    { const char* s = shp.c_str();
      while (*s) {
        while (*s == ' ' || *s == ',') s++;
        if (!*s || *s == ')') break;
        char* e; long d = std::strtol(s, &e, 10);
        if (e == s) break;
        count *= d; any = true; s = e;
      } }
    if (!any) count = 1;  // 0-d scalar

    // element type
    char endian = descr[0];           // '<','>','|','='
    char kind   = descr.size() > 1 ? descr[1] : 'i';
    int  esize  = descr.size() > 2 ? std::atoi(descr.c_str() + 2) : 8;
    if (endian == '>') { std::fprintf(stderr, "read_npy_int: big-endian unsupported\n"); std::fclose(f); return false; }
    if (kind != 'i' && kind != 'u') { std::fprintf(stderr, "read_npy_int: descr '%s' not integer\n", descr.c_str()); std::fclose(f); return false; }

    std::vector<unsigned char> raw((size_t)count * (size_t)esize);
    if (count > 0 && std::fread(raw.data(), 1, raw.size(), f) != raw.size()) {
        std::fprintf(stderr, "read_npy_int: short data read\n"); std::fclose(f); return false;
    }
    std::fclose(f);

    out.resize((size_t)count);
    const bool sgn = (kind == 'i');
    for (int64_t i = 0; i < count; i++) {
        const unsigned char* p = raw.data() + (size_t)i * esize;
        int64_t v = 0;
        switch (esize) {
            case 1: v = sgn ? (int64_t)(int8_t)p[0] : (int64_t)p[0]; break;
            case 2: { uint16_t u; std::memcpy(&u, p, 2); v = sgn ? (int64_t)(int16_t)u : (int64_t)u; break; }
            case 4: { uint32_t u; std::memcpy(&u, p, 4); v = sgn ? (int64_t)(int32_t)u : (int64_t)u; break; }
            case 8: { uint64_t u; std::memcpy(&u, p, 8); v = (int64_t)u; break; }
            default: std::fprintf(stderr, "read_npy_int: esize %d unsupported\n", esize); return false;
        }
        out[(size_t)i] = v;
    }
    return true;
}

// ----------------------------------------------------------------------------
// region table + heuristic — verbatim port of tier_keep() (per_part_decimate.py:24-33).
//   FCN = normalized face centroids ((FC - c)/s, c=(min+max)/2, s=max|V-c|), Y-up.
//   cen = mean FCN of the part's faces; xm = max |FCN.x| over the part.
// ----------------------------------------------------------------------------
struct Tier { const char* name; double keep; };
inline Tier tier_keep(int64_t n_part, int64_t total,
                       double cen_x, double cen_y, double cen_z, double xm) {
    (void)cen_x; (void)cen_z;
    double frac = (double)n_part / (double)total;
    // fingers: small part, far in X, mid in Y -> KEEP dense
    if (n_part < 0.025 * total && xm > 0.50 && cen_y > -0.35 && cen_y < 0.35) return {"HAND", 0.80};
    if (frac > 0.12)                                                          return {"HAIR", 0.025};
    if (cen_y < -0.70)                                                        return {"FOOT", 0.12};
    if (cen_y > 0.55)                                                         return {"HEAD", 0.15};
    if (xm > 0.42 && cen_y > 0.05)                                            return {"LIMB", 0.10};
    return {"BODY", 0.05};
}

struct PartReport {
    int64_t     label;
    std::string region;
    int64_t     face_in;
    int64_t     face_out;
};

// Run the per-part decimation. obj_decimate_path = path to the obj_decimate binary.
// Returns true on success; fills `out_mesh` (verts f32, faces i64) and `reports`.
inline bool per_part_decimate(const glb::Mesh& in, const std::vector<int64_t>& fid,
                              const std::string& obj_decimate_path,
                              glb::Mesh& out_mesh, std::vector<PartReport>& reports) {
    const int64_t V = (int64_t)in.verts.size() / 3;
    const int64_t F = (int64_t)in.faces.size() / 3;
    if ((int64_t)fid.size() != F) {
        std::fprintf(stderr, "per_part_decimate: face_ids count %zu != F %lld\n", fid.size(), (long long)F);
        return false;
    }
    if (F == 0) { std::fprintf(stderr, "per_part_decimate: empty mesh\n"); return false; }

    // --- normalized face centroids (FCN) ---
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int64_t v = 0; v < V; v++)
        for (int d = 0; d < 3; d++) {
            float x = in.verts[v * 3 + d];
            if (x < mn[d]) mn[d] = x;
            if (x > mx[d]) mx[d] = x;
        }
    double c[3] = {(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5};
    // s = max over all verts of max-component |V-c| (np.abs(V-c).max() is the global max abs entry)
    double s = 0.0;
    for (int64_t v = 0; v < V; v++)
        for (int d = 0; d < 3; d++)
            s = std::max(s, std::fabs((double)in.verts[v * 3 + d] - c[d]));
    if (s <= 0.0) s = 1.0;

    // face centroid (normalized) for each face
    std::vector<double> fcn(F * 3);
    for (int64_t f = 0; f < F; f++) {
        int64_t a = in.faces[f * 3], b = in.faces[f * 3 + 1], d2 = in.faces[f * 3 + 2];
        for (int d = 0; d < 3; d++) {
            double fc = ((double)in.verts[a * 3 + d] + (double)in.verts[b * 3 + d] +
                         (double)in.verts[d2 * 3 + d]) / 3.0;
            fcn[f * 3 + d] = (fc - c[d]) / s;
        }
    }

    // --- group faces by label (preserve first-seen order; we re-sort by size below) ---
    std::vector<int64_t> labels;
    std::unordered_map<int64_t, std::vector<int64_t>> face_of;  // label -> face indices
    face_of.reserve(64);
    for (int64_t f = 0; f < F; f++) {
        auto it = face_of.find(fid[f]);
        if (it == face_of.end()) { labels.push_back(fid[f]); face_of[fid[f]].push_back(f); }
        else it->second.push_back(f);
    }
    // sort parts by descending face count (matches python: sorted(... key=-count)).
    // tie-break by label ascending for determinism.
    std::sort(labels.begin(), labels.end(), [&](int64_t a, int64_t b) {
        size_t na = face_of[a].size(), nb = face_of[b].size();
        if (na != nb) return na > nb;
        return a < b;
    });

    std::printf("%5s %5s %8s -> %7s\n", "part", "tier", "in_f", "out_f");

    out_mesh = glb::Mesh{};
    reports.clear();

    char tmpl[] = "/tmp/ppd_XXXXXX";
    std::string tdir = mkdtemp(tmpl) ? std::string(tmpl) : std::string("/tmp/ppd_fallback");
    if (tdir.empty()) tdir = "/tmp";
    const std::string oin = tdir + "/i.obj", oout = tdir + "/o.obj";

    for (int64_t lbl : labels) {
        const std::vector<int64_t>& flist = face_of[lbl];
        const int64_t nf = (int64_t)flist.size();

        // extract sub-mesh: compact the referenced vertices to a local index set
        std::unordered_map<int64_t, int> remap;
        remap.reserve(nf * 2);
        std::vector<int64_t> vsub_idx;        // local->global vertex index
        std::vector<int64_t> fsub2;           // local triangle indices (3*nf)
        fsub2.reserve(nf * 3);
        for (int64_t f : flist) {
            for (int j = 0; j < 3; j++) {
                int64_t g = in.faces[f * 3 + j];
                auto it = remap.find(g);
                int local;
                if (it == remap.end()) { local = (int)vsub_idx.size(); remap.emplace(g, local); vsub_idx.push_back(g); }
                else local = it->second;
                fsub2.push_back(local);
            }
        }

        // tier: cen = mean FCN over part faces; xm = max |FCN.x| over part faces
        double cen[3] = {0, 0, 0}, xm = 0.0;
        for (int64_t f : flist) {
            for (int d = 0; d < 3; d++) cen[d] += fcn[f * 3 + d];
            xm = std::max(xm, std::fabs(fcn[f * 3 + 0]));
        }
        for (int d = 0; d < 3; d++) cen[d] /= (double)nf;
        Tier tier = tier_keep(nf, F, cen[0], cen[1], cen[2], xm);

        int64_t target = std::max((int64_t)400, (int64_t)llround((double)nf * tier.keep));

        // write part OBJ
        {
            FILE* fo = std::fopen(oin.c_str(), "w");
            if (!fo) { std::fprintf(stderr, "per_part_decimate: cannot write %s\n", oin.c_str()); return false; }
            for (int64_t gi : vsub_idx)
                std::fprintf(fo, "v %.9g %.9g %.9g\n", in.verts[gi * 3], in.verts[gi * 3 + 1], in.verts[gi * 3 + 2]);
            for (int64_t t = 0; t < nf; t++)
                std::fprintf(fo, "f %lld %lld %lld\n",
                             (long long)(fsub2[t * 3] + 1), (long long)(fsub2[t * 3 + 1] + 1),
                             (long long)(fsub2[t * 3 + 2] + 1));
            std::fclose(fo);
        }

        glb::Mesh part;
        if ((double)nf > (double)target * 1.05) {
            // shell out to obj_decimate (meshoptimizer QEM + LockBorder)
            std::string cmd = "'" + obj_decimate_path + "' '" + oin + "' '" + oout + "' " +
                              std::to_string((long long)target) + " >/dev/null 2>&1";
            int rc = std::system(cmd.c_str());
            if (rc != 0) {
                std::fprintf(stderr, "per_part_decimate: obj_decimate failed (rc=%d) for label %lld\n",
                             rc, (long long)lbl);
                return false;
            }
            // read OBJ back
            if (!read_obj(oout.c_str(), part)) {
                std::fprintf(stderr, "per_part_decimate: cannot read decimated %s\n", oout.c_str());
                return false;
            }
        } else {
            // already small: keep the part as-is (verbatim python branch)
            part.verts.resize(vsub_idx.size() * 3);
            for (size_t i = 0; i < vsub_idx.size(); i++) {
                int64_t gi = vsub_idx[i];
                part.verts[i * 3]     = in.verts[gi * 3];
                part.verts[i * 3 + 1] = in.verts[gi * 3 + 1];
                part.verts[i * 3 + 2] = in.verts[gi * 3 + 2];
            }
            part.faces.assign(fsub2.begin(), fsub2.end());
        }

        // recombine: concat verts + offset faces
        int64_t base_v = (int64_t)(out_mesh.verts.size() / 3);
        out_mesh.verts.insert(out_mesh.verts.end(), part.verts.begin(), part.verts.end());
        for (int64_t fi : part.faces) out_mesh.faces.push_back(base_v + fi);

        int64_t fout = (int64_t)part.faces.size() / 3;
        std::printf("%5lld %5s %8lld -> %7lld\n", (long long)lbl, tier.name, (long long)nf, (long long)fout);
        reports.push_back({lbl, tier.name, nf, fout});
    }

    return true;
}

inline bool read_obj_impl(const char* path, glb::Mesh& out) {
    out = glb::Mesh{};
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char line[4096];
    std::vector<float>& V = out.verts;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z; if (std::sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) { V.push_back(x); V.push_back(y); V.push_back(z); }
        } else if (line[0] == 'f' && line[1] == ' ') {
            int64_t v[64]; int n = 0; const char* s = line + 2;
            while (*s && n < 64) {
                while (*s == ' ') s++;
                if (!*s || *s == '\n' || *s == '\r') break;
                char* e; long a = std::strtol(s, &e, 10);
                if (e == s) break;
                if (a < 0) a = (long)(V.size() / 3) + a + 1;   // relative index
                v[n++] = (int64_t)a - 1;
                s = e;
                while (*s && *s != ' ' && *s != '\n' && *s != '\r') s++;  // skip /vt/vn
            }
            for (int k = 1; k + 1 < n; k++) { out.faces.push_back(v[0]); out.faces.push_back(v[k]); out.faces.push_back(v[k + 1]); }
        }
    }
    std::fclose(f);
    return true;
}
inline bool read_obj(const char* path, glb::Mesh& out) { return read_obj_impl(path, out); }

// dump_mesh bins matching the python: dump_mesh_v.bin (f32 V*3), dump_mesh_f.bin (i64 F*3),
// dump_bake.txt = "NV NF NP" (NP preserved from an existing dump_bake.txt, else 0).
inline bool dump_mesh(const std::string& dir, const glb::Mesh& m) {
    const int64_t V = (int64_t)m.verts.size() / 3, F = (int64_t)m.faces.size() / 3;
    std::string vp = dir + "/dump_mesh_v.bin", fp = dir + "/dump_mesh_f.bin", bp = dir + "/dump_bake.txt";
    FILE* fv = std::fopen(vp.c_str(), "wb");
    if (!fv) { std::fprintf(stderr, "dump_mesh: cannot write %s\n", vp.c_str()); return false; }
    std::fwrite(m.verts.data(), 4, m.verts.size(), fv); std::fclose(fv);
    FILE* ff = std::fopen(fp.c_str(), "wb");
    if (!ff) { std::fprintf(stderr, "dump_mesh: cannot write %s\n", fp.c_str()); return false; }
    std::fwrite(m.faces.data(), 8, m.faces.size(), ff); std::fclose(ff);
    // preserve NP (3rd field) from an existing dump_bake.txt if present
    int64_t np_voxels = 0;
    if (FILE* rb = std::fopen(bp.c_str(), "r")) {
        long long a, b, cc;
        if (std::fscanf(rb, "%lld %lld %lld", &a, &b, &cc) == 3) np_voxels = cc;
        std::fclose(rb);
    }
    FILE* fb = std::fopen(bp.c_str(), "w");
    if (!fb) { std::fprintf(stderr, "dump_mesh: cannot write %s\n", bp.c_str()); return false; }
    std::fprintf(fb, "%lld %lld %lld\n", (long long)V, (long long)F, (long long)np_voxels);
    std::fclose(fb);
    std::printf("[dump] bake target -> %s/dump_mesh_*.bin (%lld v / %lld f), dump_bake.txt updated\n",
                dir.c_str(), (long long)V, (long long)F);
    return true;
}

} // namespace ppd
