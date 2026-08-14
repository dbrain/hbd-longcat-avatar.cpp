// rig_glb_skin_io.hpp — load a rigged GLB into the pose gate's `rigqc::SkinnedRig`, and write one
// back with ONLY its JOINTS_0/WEIGHTS_0 bytes replaced.
//
// WHY THIS EXISTS
// ---------------
// The loader was born inline in rig_pose_gate_main.cpp. `rig_weight_cleanup` needs exactly the same
// view of a rigged GLB — same "first node with both a mesh and a skin" rule, same primitive
// concatenation, same joint-space collapse — because the whole point of the cleanup is that the
// instrument which judges it (rig_pose_gate.hpp) and the tool which edits it must agree on what the
// asset *is*. Two copies of that rule would drift. So it lives here and both include it.
//
// THE WRITER IS A BYTE PATCH, NOT A RE-EXPORT.
// `write_patched_glb` copies the original file verbatim and overwrites the JOINTS_0 and WEIGHTS_0
// accessor payloads in place. Vertex count, accessor layout, bufferView offsets, JSON, textures,
// materials, animations, extensions: all bit-identical to the input. That is deliberate — a rigged
// GLB out of this pipeline carries a baked 2048px texture atlas and a 26k-chart UV layout, and a
// re-export is a chance to lose some of it. A cleanup that also silently re-encodes the asset is not
// a cleanup you can offer to run on somebody's finished character.
//
// Header-only, CPU, no ggml.
#pragma once

#include "glb_reader.hpp"     // JSON parser + component-size helpers
#include "rig_pose_gate.hpp"  // rigqc::SkinnedRig, M4
#include "rig_weight_cleanup.hpp"  // the cleanup itself, for the one-call facade at the bottom

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rigio {

using glb::detail::JVal;

// One skinned primitive's place in the concatenated vertex array, plus the accessors whose bytes the
// patcher has to rewrite.
struct PrimRef {
    int     joints_acc  = -1;
    int     weights_acc = -1;
    int64_t first_vertex = 0;
    int64_t count        = 0;
};

struct SkinnedGlb {
    std::vector<uint8_t> raw;        // the whole file, untouched
    JVal                 root;       // parsed JSON chunk
    size_t               bin_off = 0;// offset of the BIN chunk's DATA inside `raw`
    size_t               bin_len = 0;
    std::vector<PrimRef> prims;      // in concatenation order
    rigqc::SkinnedRig    rig;
    std::vector<float>   joint_pos;  // J*3 — joint world rest positions, in the mesh's own space
};

namespace detail {

inline bool read_accessor(const SkinnedGlb& g, int idx, std::vector<double>& out, int& ncomp,
                          int64_t& count) {
    const JVal* accs = g.root.find("accessors");
    const JVal* bvs  = g.root.find("bufferViews");
    if (!accs || !accs->is_arr() || idx < 0 || idx >= (int)accs->arr.size()) return false;
    const JVal& a = accs->arr[(size_t)idx];
    const JVal* bvi = a.find("bufferView");
    if (!bvi || !bvs || !bvs->is_arr()) return false;
    const JVal& bv = bvs->arr[(size_t)bvi->as_int()];
    const int ct = (int)a.find("componentType")->as_int();
    const int cs = glb::detail::comp_size(ct);
    ncomp = glb::detail::type_count(a.find("type")->str);
    count = a.find("count")->as_int();
    if (cs == 0 || ncomp == 0) return false;
    size_t base = 0, stride = 0;
    if (const JVal* o = bv.find("byteOffset")) base += (size_t)o->as_int();
    if (const JVal* o = a.find("byteOffset"))  base += (size_t)o->as_int();
    if (const JVal* s = bv.find("byteStride")) stride = (size_t)s->as_int();
    if (stride == 0) stride = (size_t)cs * (size_t)ncomp;
    if (base + (size_t)(count - 1) * stride + (size_t)cs * ncomp > g.bin_len) return false;
    out.resize((size_t)count * ncomp);
    for (int64_t i = 0; i < count; ++i) {
        const uint8_t* p = &g.raw[g.bin_off + base + (size_t)i * stride];
        for (int c = 0; c < ncomp; ++c) {
            const uint8_t* q = p + (size_t)c * cs;
            double v = 0;
            switch (ct) {
                case 5120: { int8_t   x; std::memcpy(&x, q, 1); v = x; break; }
                case 5121: { uint8_t  x; std::memcpy(&x, q, 1); v = x; break; }
                case 5122: { int16_t  x; std::memcpy(&x, q, 2); v = x; break; }
                case 5123: { uint16_t x; std::memcpy(&x, q, 2); v = x; break; }
                case 5125: { uint32_t x; std::memcpy(&x, q, 4); v = x; break; }
                case 5126: { float    x; std::memcpy(&x, q, 4); v = x; break; }
                default: return false;
            }
            out[(size_t)i * ncomp + c] = v;
        }
    }
    return true;
}

// Write `values` (count*ncomp doubles) back through accessor `idx`, in its EXISTING component type
// and layout. Refuses if the shape does not match what is already there — the cleanup never changes
// how many vertices or how many influences an asset has, so a mismatch means a bug, not a case to
// handle.
inline bool write_accessor(SkinnedGlb& g, int idx, const std::vector<double>& values, int want_ncomp,
                           int64_t want_count, std::string& err) {
    const JVal* accs = g.root.find("accessors");
    const JVal* bvs  = g.root.find("bufferViews");
    if (!accs || !accs->is_arr() || idx < 0 || idx >= (int)accs->arr.size()) { err = "bad accessor"; return false; }
    const JVal& a = accs->arr[(size_t)idx];
    const JVal* bvi = a.find("bufferView");
    if (!bvi || !bvs || !bvs->is_arr()) { err = "accessor has no bufferView"; return false; }
    const JVal& bv = bvs->arr[(size_t)bvi->as_int()];
    const int ct = (int)a.find("componentType")->as_int();
    const int cs = glb::detail::comp_size(ct);
    const int ncomp = glb::detail::type_count(a.find("type")->str);
    const int64_t count = a.find("count")->as_int();
    if (cs == 0 || ncomp != want_ncomp || count != want_count) { err = "accessor shape mismatch"; return false; }
    if (values.size() != (size_t)count * (size_t)ncomp) { err = "value count mismatch"; return false; }
    size_t base = 0, stride = 0;
    if (const JVal* o = bv.find("byteOffset")) base += (size_t)o->as_int();
    if (const JVal* o = a.find("byteOffset"))  base += (size_t)o->as_int();
    if (const JVal* s = bv.find("byteStride")) stride = (size_t)s->as_int();
    if (stride == 0) stride = (size_t)cs * (size_t)ncomp;
    if (base + (size_t)(count - 1) * stride + (size_t)cs * ncomp > g.bin_len) { err = "accessor out of range"; return false; }
    for (int64_t i = 0; i < count; ++i) {
        uint8_t* p = &g.raw[g.bin_off + base + (size_t)i * stride];
        for (int c = 0; c < ncomp; ++c) {
            uint8_t* q = p + (size_t)c * cs;
            const double v = values[(size_t)i * ncomp + c];
            switch (ct) {
                case 5120: { const int8_t   x = (int8_t)  std::lround(v); std::memcpy(q, &x, 1); break; }
                case 5121: { const uint8_t  x = (uint8_t) std::lround(v); std::memcpy(q, &x, 1); break; }
                case 5122: { const int16_t  x = (int16_t) std::lround(v); std::memcpy(q, &x, 2); break; }
                case 5123: { const uint16_t x = (uint16_t)std::lround(v); std::memcpy(q, &x, 2); break; }
                case 5125: { const uint32_t x = (uint32_t)std::lround(v); std::memcpy(q, &x, 4); break; }
                case 5126: { const float    x = (float)v;                std::memcpy(q, &x, 4); break; }
                default: err = "unsupported componentType"; return false;
            }
        }
    }
    return true;
}

// Affine 4x4 inverse (row-major, bottom row 0001). Used only to cross-check the skeleton against the
// inverse bind matrices; falls back to identity on a singular upper-left block.
inline bool affine_inverse(const rigqc::M4& m, rigqc::M4& out) {
    const double a = m[0], b = m[1], c = m[2];
    const double d = m[4], e = m[5], f = m[6];
    const double g = m[8], h = m[9], i = m[10];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::fabs(det) < 1e-20) return false;
    const double id = 1.0 / det;
    double r[9];
    r[0] = (e * i - f * h) * id; r[1] = (c * h - b * i) * id; r[2] = (b * f - c * e) * id;
    r[3] = (f * g - d * i) * id; r[4] = (a * i - c * g) * id; r[5] = (c * d - a * f) * id;
    r[6] = (d * h - e * g) * id; r[7] = (b * g - a * h) * id; r[8] = (a * e - b * d) * id;
    const double tx = m[3], ty = m[7], tz = m[11];
    out = rigqc::m4_identity();
    for (int rr = 0; rr < 3; ++rr) for (int cc = 0; cc < 3; ++cc) out[(size_t)rr * 4 + cc] = (float)r[rr * 3 + cc];
    out[3]  = (float)(-(r[0] * tx + r[1] * ty + r[2] * tz));
    out[7]  = (float)(-(r[3] * tx + r[4] * ty + r[5] * tz));
    out[11] = (float)(-(r[6] * tx + r[7] * ty + r[8] * tz));
    return true;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Load. Mirrors rig_pose_smoke.mesh_and_skin (and rig_pose_gate_main's loader) exactly: pick the
// first node carrying BOTH a mesh and a skin whose primitive has POSITION/JOINTS_0/WEIGHTS_0 +
// indices, then concatenate every identity-transform primitive sharing that same skin.
// ---------------------------------------------------------------------------
inline bool load_skinned_glb(const char* path, SkinnedGlb& out, std::string& err) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { err = std::string("cannot open ") + path; return false; }
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    if (n <= 12) { std::fclose(f); err = "file too small to be a GLB"; return false; }
    out.raw.resize((size_t)n);
    if (std::fread(out.raw.data(), 1, (size_t)n, f) != (size_t)n) { std::fclose(f); err = "short read"; return false; }
    std::fclose(f);

    uint32_t magic, ver;
    std::memcpy(&magic, &out.raw[0], 4); std::memcpy(&ver, &out.raw[4], 4);
    if (magic != 0x46546C67u || ver != 2) { err = "not a GLB v2"; return false; }
    size_t off = 12;
    bool have_json = false;
    while (off + 8 <= out.raw.size()) {
        uint32_t clen, ctype;
        std::memcpy(&clen, &out.raw[off], 4); std::memcpy(&ctype, &out.raw[off + 4], 4);
        off += 8;
        if (off + clen > out.raw.size()) break;
        if (ctype == 0x4E4F534Au) {
            glb::detail::JParser p((const char*)&out.raw[off], clen);
            if (!p.parse(out.root)) { err = "bad JSON chunk"; return false; }
            have_json = true;
        } else if (ctype == 0x004E4942u) {
            out.bin_off = off; out.bin_len = clen;
        }
        off += clen + ((4 - (clen & 3)) & 3);
    }
    if (!have_json) { err = "no JSON chunk"; return false; }

    const JVal* nodes  = out.root.find("nodes");
    const JVal* meshes = out.root.find("meshes");
    const JVal* skins  = out.root.find("skins");
    if (!nodes || !meshes || !skins || !nodes->is_arr() || !skins->is_arr()) { err = "no nodes/meshes/skins"; return false; }

    auto usable_prim = [&](const JVal& prim) {
        const JVal* at = prim.find("attributes");
        return at && at->find("POSITION") && at->find("JOINTS_0") && at->find("WEIGHTS_0") &&
               prim.find("indices");
    };

    int skin_i = -1;
    for (size_t ni = 0; ni < nodes->arr.size() && skin_i < 0; ++ni) {
        const JVal& nd = nodes->arr[ni];
        if (!nd.find("mesh") || !nd.find("skin")) continue;
        const JVal* prims = meshes->arr[(size_t)nd.find("mesh")->as_int()].find("primitives");
        if (!prims || !prims->is_arr()) continue;
        for (const JVal& p : prims->arr) if (usable_prim(p)) { skin_i = (int)nd.find("skin")->as_int(); break; }
    }
    if (skin_i < 0) { err = "no indexed skinned primitive"; return false; }

    rigqc::SkinnedRig& R = out.rig;
    R = rigqc::SkinnedRig{};
    int32_t vbase = 0;
    for (size_t ni = 0; ni < nodes->arr.size(); ++ni) {
        const JVal& nd = nodes->arr[ni];
        if (!nd.find("skin") || (int)nd.find("skin")->as_int() != skin_i || !nd.find("mesh")) continue;
        if (nd.find("matrix") || nd.find("translation") || nd.find("rotation") || nd.find("scale")) {
            err = "skinned mesh node carries a transform — unsupported (as in the Python)"; return false;
        }
        const JVal* prims = meshes->arr[(size_t)nd.find("mesh")->as_int()].find("primitives");
        for (const JVal& p : prims->arr) {
            if (!usable_prim(p)) continue;
            const JVal* at = p.find("attributes");
            std::vector<double> pos, jj, ww, idx;
            int nc; int64_t cnt;
            if (!detail::read_accessor(out, (int)at->find("POSITION")->as_int(), pos, nc, cnt) || nc != 3) { err = "bad POSITION"; return false; }
            const int64_t nv = cnt;
            for (int64_t i = 0; i < nv * 3; ++i) R.vertices.push_back((float)pos[(size_t)i]);
            if (!detail::read_accessor(out, (int)at->find("JOINTS_0")->as_int(), jj, nc, cnt) || nc != 4) { err = "bad JOINTS_0"; return false; }
            for (int64_t i = 0; i < nv * 4; ++i) R.jidx.push_back((int32_t)jj[(size_t)i]);
            if (!detail::read_accessor(out, (int)at->find("WEIGHTS_0")->as_int(), ww, nc, cnt) || nc != 4) { err = "bad WEIGHTS_0"; return false; }
            for (int64_t i = 0; i < nv * 4; ++i) R.jw.push_back((float)ww[(size_t)i]);
            if (!detail::read_accessor(out, (int)p.find("indices")->as_int(), idx, nc, cnt) || nc != 1) { err = "bad indices"; return false; }
            for (int64_t i = 0; i < cnt; ++i) R.faces.push_back((int32_t)idx[(size_t)i] + vbase);
            PrimRef pr;
            pr.joints_acc  = (int)at->find("JOINTS_0")->as_int();
            pr.weights_acc = (int)at->find("WEIGHTS_0")->as_int();
            pr.first_vertex = vbase;
            pr.count = nv;
            out.prims.push_back(pr);
            vbase += (int32_t)nv;
        }
    }

    const JVal& sk = skins->arr[(size_t)skin_i];
    const JVal* jl = sk.find("joints");
    if (!jl || !jl->is_arr()) { err = "skin has no joints"; return false; }
    const int J = (int)jl->arr.size();
    std::vector<int> jnode((size_t)J);
    for (int j = 0; j < J; ++j) jnode[(size_t)j] = (int)jl->arr[(size_t)j].as_int();

    std::vector<int> nparent(nodes->arr.size(), -1);
    for (size_t ni = 0; ni < nodes->arr.size(); ++ni)
        if (const JVal* ch = nodes->arr[ni].find("children"))
            for (const JVal& c : ch->arr) nparent[(size_t)c.as_int()] = (int)ni;
    std::vector<int> node_to_joint(nodes->arr.size(), -1);
    for (int j = 0; j < J; ++j) node_to_joint[(size_t)jnode[(size_t)j]] = j;

    auto node_local = [&](int nd_i) {
        rigqc::M4 m = rigqc::m4_identity();
        const JVal& nd = nodes->arr[(size_t)nd_i];
        if (const JVal* mm = nd.find("matrix")) {
            for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r)
                m[(size_t)r * 4 + c] = (float)mm->arr[(size_t)(c * 4 + r)].num;
        } else if (const JVal* t = nd.find("translation")) {
            m[3] = (float)t->arr[0].num; m[7] = (float)t->arr[1].num; m[11] = (float)t->arr[2].num;
        }
        return m;
    };

    R.parent.assign((size_t)J, -1);
    R.local.assign((size_t)J, rigqc::m4_identity());
    for (int j = 0; j < J; ++j) {
        rigqc::M4 acc = node_local(jnode[(size_t)j]);
        int cur = nparent[(size_t)jnode[(size_t)j]];
        while (cur >= 0 && node_to_joint[(size_t)cur] < 0) {
            acc = rigqc::m4_mul(node_local(cur), acc);
            cur = nparent[(size_t)cur];
        }
        R.parent[(size_t)j] = (cur >= 0) ? node_to_joint[(size_t)cur] : -1;
        R.local[(size_t)j] = acc;
    }

    R.ibm.assign((size_t)J, rigqc::m4_identity());
    if (const JVal* ibmi = sk.find("inverseBindMatrices")) {
        std::vector<double> ibm; int nc; int64_t cnt;
        if (!detail::read_accessor(out, (int)ibmi->as_int(), ibm, nc, cnt) || nc != 16 || cnt != J) {
            err = "bad inverseBindMatrices"; return false;
        }
        for (int j = 0; j < J; ++j)
            for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r)
                R.ibm[(size_t)j][(size_t)r * 4 + c] = (float)ibm[(size_t)j * 16 + (size_t)(c * 4 + r)];
    }

    R.names.assign((size_t)J, std::string());
    for (int j = 0; j < J; ++j)
        if (const JVal* nm = nodes->arr[(size_t)jnode[(size_t)j]].find("name")) R.names[(size_t)j] = nm->str;

    // Joint world rest positions, in the space the POSITION accessor lives in. LBS uses
    // M_j = global_j * ibm_j, so `global_j`'s translation IS the joint's bind position in mesh space
    // whenever the rig is in bind pose (global_j * ibm_j == I). We take it from `global`, and the
    // caller can cross-check against inverse(ibm) via `joint_bind_disagreement()`.
    std::vector<rigqc::M4> gl;
    rigqc::detail::global_transforms(R, R.local, gl);
    out.joint_pos.assign((size_t)J * 3, 0.f);
    for (int j = 0; j < J; ++j) {
        out.joint_pos[(size_t)j * 3 + 0] = gl[(size_t)j][3];
        out.joint_pos[(size_t)j * 3 + 1] = gl[(size_t)j][7];
        out.joint_pos[(size_t)j * 3 + 2] = gl[(size_t)j][11];
    }
    return true;
}

// Largest distance between a joint's position taken from the node tree and the one implied by its
// inverse bind matrix, as a fraction of the mesh bbox diagonal. Should be ~0 for a bind-pose rig; a
// big number means the skeleton and the mesh are not in the same space and every distance the
// cleanup measures would be meaningless. The caller checks it and refuses.
inline double joint_bind_disagreement(const SkinnedGlb& g) {
    const rigqc::SkinnedRig& R = g.rig;
    const int J = R.J(), V = R.V();
    if (J == 0 || V == 0) return INFINITY;
    float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < V; ++i) for (int a = 0; a < 3; ++a) {
        const float x = R.vertices[(size_t)i * 3 + a];
        mn[a] = std::min(mn[a], x); mx[a] = std::max(mx[a], x);
    }
    double diag = 0;
    for (int a = 0; a < 3; ++a) { const double d = mx[a] - mn[a]; diag += d * d; }
    diag = std::sqrt(diag);
    if (!(diag > 0)) return INFINITY;
    double worst = 0;
    for (int j = 0; j < J; ++j) {
        rigqc::M4 inv;
        if (!detail::affine_inverse(R.ibm[(size_t)j], inv)) continue;
        const double dx = inv[3]  - g.joint_pos[(size_t)j * 3 + 0];
        const double dy = inv[7]  - g.joint_pos[(size_t)j * 3 + 1];
        const double dz = inv[11] - g.joint_pos[(size_t)j * 3 + 2];
        worst = std::max(worst, std::sqrt(dx * dx + dy * dy + dz * dz) / diag);
    }
    return worst;
}

// Write the file back out with new skin. `jidx`/`jw` are V*4 in the loader's concatenated order.
inline bool write_patched_glb(SkinnedGlb& g, const std::vector<int32_t>& jidx,
                              const std::vector<float>& jw, const char* path, std::string& err) {
    const int V = g.rig.V();
    if ((int)jidx.size() != V * 4 || (int)jw.size() != V * 4) { err = "skin array size mismatch"; return false; }
    for (const PrimRef& pr : g.prims) {
        std::vector<double> ji((size_t)pr.count * 4), wi((size_t)pr.count * 4);
        for (int64_t i = 0; i < pr.count * 4; ++i) {
            ji[(size_t)i] = (double)jidx[(size_t)(pr.first_vertex * 4 + i)];
            wi[(size_t)i] = (double)jw[(size_t)(pr.first_vertex * 4 + i)];
        }
        if (!detail::write_accessor(g, pr.joints_acc,  ji, 4, pr.count, err)) return false;
        if (!detail::write_accessor(g, pr.weights_acc, wi, 4, pr.count, err)) return false;
    }
    FILE* f = std::fopen(path, "wb");
    if (!f) { err = std::string("cannot write ") + path; return false; }
    const bool okw = std::fwrite(g.raw.data(), 1, g.raw.size(), f) == g.raw.size();
    std::fclose(f);
    if (!okw) { err = "short write"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// ONE CALL: load a rigged GLB, run the verified cleanup, write it back. This is the whole surface
// the pipeline needs, and it is deliberately failure-tolerant — a cleanup that cannot run is a
// missed improvement, never a failed request. `summary` always ends up with a one-line verdict.
// `out_path == in_path` rewrites in place. Returns false only if nothing was written.
// ---------------------------------------------------------------------------
inline bool clean_rigged_glb(const char* in_path, const char* out_path,
                             const rigclean::Options& opt, std::string& summary,
                             std::string& err, double tolerance = 1.001) {
    SkinnedGlb g;
    if (!load_skinned_glb(in_path, g, err)) { summary = "skin cleanup skipped: " + err; return false; }
    const double disagree = joint_bind_disagreement(g);
    if (!(disagree < 0.02)) {
        char b[192];
        std::snprintf(b, sizeof(b), "skin cleanup skipped: rig is not in bind pose (joint/IBM "
                                    "disagreement %.4f of the diagonal)", disagree);
        summary = b; err = summary; return false;
    }
    rigqc::SkinnedRig R = g.rig;
    rigclean::VerifiedResult vr;
    if (!rigclean::clean_skin_weights_verified(R, g.joint_pos, opt, vr, tolerance)) {
        summary = "skin cleanup skipped: malformed rig arrays"; err = summary; return false;
    }
    char b[512];
    std::snprintf(b, sizeof(b),
        "skin cleanup rung %d (%s): %ld/%d vertices changed (%ld material), %ld far influences "
        "dropped, %ld smoothed; pose gate %.3f->%.3f default, %.3f->%.3f all-influential",
        vr.rung, vr.rung_name, vr.after.vertices_changed, R.V(),
        vr.after.vertices_changed_material, vr.after.influences_dropped, vr.after.tension_verts,
        vr.before_default, vr.after_default, vr.before_allinf, vr.after_allinf);
    summary = b;
    if (vr.rung == 0 && std::string(in_path) == std::string(out_path)) return true;  // nothing to write
    if (!write_patched_glb(g, R.jidx, R.jw, out_path, err)) {
        summary += " (WRITE FAILED: " + err + ")";
        return false;
    }
    return true;
}

}  // namespace rigio
