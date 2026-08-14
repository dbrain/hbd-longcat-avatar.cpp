// rig_pose_gate.hpp — NATIVE port of rig_pose_smoke.py's `--pose-gate` verdict (and of
// rig_weight_health.py), so the deformation gate can run PER REQUEST inside the library instead of
// only in a wrapper script that an API caller need not invoke.
//
// WHY THIS EXISTS
// ---------------
// Across the whole 2026-08 rig campaign the pose gate is the ONLY instrument that judged every case
// correctly: good rigs land at 2.2-4.1, broken ones at 9.2-46.7, and its `moved` term is what
// catches a WEIGHTLESS rig — whose stretch columns read a deceptively perfect 1.000/1.000/1.000
// because a rig that deforms nothing also stretches nothing. `rig_score` gives gilly's broken solid
// rig 0.588 against 0.685 for the good one (no signal), and `rig_weight_health` passes the
// weightless rig outright. So the retry selector has to key on THIS number, and this number has to
// be available without shelling out to Python + numpy + pyrender.
//
// WHAT IS PORTED, AND WHAT IS NOT
// ------------------------------
// Ported: `mesh_and_skin`'s data model, `global_transforms`, `deform` (the glTF COLUMN-VECTOR LBS
// convention — see the self-test), `generic_joint_masses_and_centres`, `edge_component_ids`, the
// stretch/`moved` statistics, the all-influential audit and the per-component audit, and both gate
// thresholds (5.0 single-pose / 6.0 all-influential). NOT ported: the pyrender tiles. This is the
// verdict, not the picture; the Python remains the tool you run when you want to LOOK.
//
// NUMERICAL FIDELITY. Geometry is carried in float32 exactly as numpy carries it, and the quantile
// is numpy's default 'linear' interpolation (index = q*(n-1), lerp between neighbours), so the
// printed line can be diffed against the Python's character for character. `pose_gate_line()`
// reproduces the Python's format string verbatim — including its quirk of printing the
// all-influential joint tag AFTER the component-worst number.
//
// Header-only, no ggml, no CUDA, no glTF parsing: callers hand it arrays. `pose_gate_from_rig()`
// builds those arrays from exactly what `write_rigged_textured_glb` is about to write (same
// `rig_local_and_ibm`, same `rig_topk4`), so the gate judges the asset that ships, not a proxy.
#pragma once

#include "glb_rigged.hpp"   // rig_local_and_ibm, rig_topk4 — the SHIPPED skeleton/skin encoding
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace rigqc {

// ROW-major 4x4: m[r*4+c]. A point is transformed as M @ p (glTF's column-vector convention).
// NOTE this is the TRANSPOSE of the layout glTF stores on disk, which is what the Python does too
// (`.reshape(-1,4,4).transpose(0,2,1)` for the IBMs, `.reshape(4,4).T` for a node matrix).
using M4 = std::array<float, 16>;

inline M4 m4_identity() { return {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; }

inline M4 m4_mul(const M4& a, const M4& b) {
    M4 o{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += a[r * 4 + k] * b[k * 4 + c];
            o[r * 4 + c] = s;
        }
    return o;
}

// rig_pose_smoke.rotation_z: cos/sin computed in double, stored float32 (so Rz(-90) carries numpy's
// 6.12e-17 cosine, not a hand-zeroed one).
inline M4 m4_rotation_z(double degrees) {
    const double a = degrees * M_PI / 180.0;
    const float c = (float)std::cos(a), s = (float)std::sin(a);
    M4 o = m4_identity();
    o[0] = c; o[1] = -s;
    o[4] = s; o[5] = c;
    return o;
}

// ---------------------------------------------------------------------------
// The rig, as the gate needs it. `local`/`ibm` are per JOINT (not per glTF node): `local[j]` is the
// transform of joint j relative to its joint-parent, and for a root joint it is that joint's full
// world transform. A glTF whose skeleton has intermediate non-joint nodes collapses onto this
// without loss, because the pose only ever POST-multiplies one joint's own local matrix.
// ---------------------------------------------------------------------------
struct SkinnedRig {
    std::vector<float>       vertices;   // V*3, the POSITION accessor (rest, pre-LBS)
    std::vector<int32_t>     faces;      // F*3
    std::vector<int32_t>     jidx;       // V*4, JOINTS_0
    std::vector<float>       jw;         // V*4, WEIGHTS_0
    std::vector<M4>          local;      // J
    std::vector<M4>          ibm;        // J
    std::vector<int32_t>     parent;     // J, joint-space, -1 = root
    std::vector<std::string> names;      // J, or empty
    int V() const { return (int)(vertices.size() / 3); }
    int F() const { return (int)(faces.size() / 3); }
    int J() const { return (int)parent.size(); }
};

struct PoseGateOpts {
    enum Mode {
        Default = 0,            // named Mixamo arms if present, else the heaviest non-root joint
        GenericAllInfluential,  // audit EVERY joint holding >= 8% of peak mass  (the campaign gate)
        GenericExtremity,       // the distal-weighted diagnostic
        GenericJoint            // one requested joint
    };
    Mode mode = Default;
    int  requested_joint = -1;
};

struct PoseGateResult {
    bool   ok   = false;        // the gate ran (false => `err` says why it could not)
    bool   pass = false;
    std::string err;
    std::string pose_label;
    int    chosen_joint = -1;
    int    n_audit = 0;
    double moved = 0, p99 = 0, p995 = 0, p999 = 0;
    double over5 = 0, over10 = 0, max_stretch = 0, max_disp = 0;
    double worst_audit_p999 = 0;     int worst_audit_joint = -1;
    double worst_component_p999 = 0; int worst_component_joint = -1; long worst_component_id = -1;
    double gate_limit = 0;
    // The single number to rank draws by: the worse of the two audited p999s. It is what the gate
    // thresholds, and it is comparable across modes only within one mode.
    double worst() const { return std::max(worst_audit_p999, worst_component_p999); }
};

struct WeightHealth {
    int    J = 0, V = 0;
    int    influential = 0;     // joints holding >= 8% of the peak joint's mass
    int    mass_1pct = 0;
    int    dominant_1pct = 0;
    double single_share = 0;
    bool   pass = false;        // influential >= min_influential
    bool   review = false;      // single_share > review threshold (a FLAG, never a failure)
};

namespace detail {

// numpy's default quantile ('linear'): index = q*(n-1), lerp between floor and ceil neighbours.
// Selection, not a full sort — the audit loop calls this once per influential joint.
inline double quantile_inplace(std::vector<float>& v, double q) {
    const size_t n = v.size();
    if (n == 0) return INFINITY;
    if (n == 1) return (double)v[0];
    const double pos = q * (double)(n - 1);
    size_t lo = (size_t)std::floor(pos);
    if (lo >= n - 1) lo = n - 2;
    const double frac = pos - (double)lo;
    std::nth_element(v.begin(), v.begin() + (long)lo, v.end());
    const double a = (double)v[lo];
    const double b = (double)*std::min_element(v.begin() + (long)lo + 1, v.end());
    return a + frac * (b - a);
}

inline double max_of(const std::vector<float>& v) {
    if (v.empty()) return INFINITY;
    return (double)*std::max_element(v.begin(), v.end());
}

inline double frac_over(const std::vector<float>& v, float t) {
    if (v.empty()) return 1.0;
    size_t c = 0;
    for (float x : v) if (x > t) ++c;
    return (double)c / (double)v.size();
}

// rig_pose_smoke.global_transforms, reduced to joint space (see SkinnedRig).
inline void global_transforms(const SkinnedRig& R, const std::vector<M4>& local, std::vector<M4>& out) {
    const int J = R.J();
    out.assign((size_t)J, m4_identity());
    std::vector<char> done((size_t)J, 0);
    // parents-before-children; the forest is shallow, so a bounded sweep is simpler than a sort.
    for (int pass = 0; pass < J + 1; ++pass) {
        bool progressed = false, pending = false;
        for (int j = 0; j < J; ++j) {
            if (done[(size_t)j]) continue;
            const int p = R.parent[(size_t)j];
            if (p < 0)                { out[(size_t)j] = local[(size_t)j]; }
            else if (done[(size_t)p]) { out[(size_t)j] = m4_mul(out[(size_t)p], local[(size_t)j]); }
            else { pending = true; continue; }
            done[(size_t)j] = 1; progressed = true;
        }
        if (!pending) return;
        if (!progressed) return;   // cyclic parent tree: leave the rest identity (validated upstream)
    }
}

// rig_pose_smoke.deform — glTF LBS in the COLUMN-VECTOR convention. The row-vector form silently
// preserves a translation-only rest pose while rotating an ANIMATED joint about the model origin
// instead of the joint, which made the gate reject valid rigs; keep this explicit.
inline void deform(const SkinnedRig& R, const std::vector<M4>& global_, std::vector<float>& out) {
    const int V = R.V(), J = R.J();
    std::vector<M4> m((size_t)J);
    for (int j = 0; j < J; ++j) m[(size_t)j] = m4_mul(global_[(size_t)j], R.ibm[(size_t)j]);
    out.assign((size_t)V * 3, 0.f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long vi = 0; vi < (long long)V; ++vi) {
        const float* p = &R.vertices[(size_t)vi * 3];
        float ox = 0.f, oy = 0.f, oz = 0.f;
        for (int k = 0; k < 4; ++k) {
            const int j = R.jidx[(size_t)vi * 4 + k];
            if (j < 0 || j >= J) continue;
            const float* mm = m[(size_t)j].data();
            const float w = R.jw[(size_t)vi * 4 + k];
            ox += w * (mm[0] * p[0] + mm[1] * p[1] + mm[2]  * p[2] + mm[3]);
            oy += w * (mm[4] * p[0] + mm[5] * p[1] + mm[6]  * p[2] + mm[7]);
            oz += w * (mm[8] * p[0] + mm[9] * p[1] + mm[10] * p[2] + mm[11]);
        }
        out[(size_t)vi * 3 + 0] = ox;
        out[(size_t)vi * 3 + 1] = oy;
        out[(size_t)vi * 3 + 2] = oz;
    }
}

// rig_pose_smoke.generic_joint_masses_and_centres
inline void joint_mass_and_centres(const SkinnedRig& R, std::vector<double>& mass,
                                   std::vector<double>& centres) {
    const int V = R.V(), J = R.J();
    mass.assign((size_t)J, 0.0);
    centres.assign((size_t)J * 3, 0.0);
    for (int k = 0; k < 4; ++k)
        for (int vi = 0; vi < V; ++vi) {
            const int j = R.jidx[(size_t)vi * 4 + k];
            if (j < 0 || j >= J) continue;
            const double w = (double)R.jw[(size_t)vi * 4 + k];
            mass[(size_t)j] += w;
            for (int ax = 0; ax < 3; ++ax)
                centres[(size_t)j * 3 + ax] += w * (double)R.vertices[(size_t)vi * 3 + ax];
        }
    for (int j = 0; j < J; ++j)
        for (int ax = 0; ax < 3; ++ax)
            centres[(size_t)j * 3 + ax] /= std::max(mass[(size_t)j], 1e-12);
}

// rig_pose_smoke.edge_component_ids — label faces by POSITION-WELDED surface piece. glTF duplicates
// a position at every UV chart boundary; without the weld every chart becomes a false "component"
// and a clean textured character fails the per-component audit.
inline void face_component_ids(const std::vector<float>& rest_v, const std::vector<int32_t>& faces,
                               std::vector<int32_t>& out) {
    const int V = (int)(rest_v.size() / 3), F = (int)(faces.size() / 3);
    std::vector<int32_t> uf((size_t)V);
    for (int i = 0; i < V; ++i) uf[(size_t)i] = i;
    std::function<int(int)> find = [&](int v) {
        while (uf[(size_t)v] != v) { uf[(size_t)v] = uf[(size_t)uf[(size_t)v]]; v = uf[(size_t)v]; }
        return v;
    };
    float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < V; ++i)
        for (int a = 0; a < 3; ++a) {
            const float x = rest_v[(size_t)i * 3 + a];
            mn[a] = std::min(mn[a], x); mx[a] = std::max(mx[a], x);
        }
    float extf = 0.f;
    for (int a = 0; a < 3; ++a) extf = std::max(extf, mx[a] - mn[a]);
    const double quantum = std::max((double)extf * 1e-6, 1e-8);

    struct Key { int64_t k[3]; int32_t v; };
    std::vector<Key> keys((size_t)V);
    for (int i = 0; i < V; ++i) {
        for (int a = 0; a < 3; ++a)
            keys[(size_t)i].k[a] = (int64_t)std::nearbyint((double)rest_v[(size_t)i * 3 + a] / quantum);
        keys[(size_t)i].v = i;
    }
    std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) {
        if (a.k[0] != b.k[0]) return a.k[0] < b.k[0];
        if (a.k[1] != b.k[1]) return a.k[1] < b.k[1];
        if (a.k[2] != b.k[2]) return a.k[2] < b.k[2];
        return a.v < b.v;
    });
    for (size_t i = 1; i < keys.size(); ++i) {
        const Key& a = keys[i - 1];
        const Key& b = keys[i];
        if (a.k[0] != b.k[0] || a.k[1] != b.k[1] || a.k[2] != b.k[2]) continue;
        const int ra = find(a.v), rb = find(b.v);
        if (ra != rb) uf[(size_t)rb] = ra;
    }
    for (int f = 0; f < F; ++f) {
        const int root = find(faces[(size_t)f * 3 + 0]);
        for (int t = 1; t < 3; ++t) {
            const int other = find(faces[(size_t)f * 3 + t]);
            if (other != root) uf[(size_t)other] = root;
        }
    }
    out.assign((size_t)F, 0);
    for (int f = 0; f < F; ++f) out[(size_t)f] = find(faces[(size_t)f * 3 + 0]);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------
inline PoseGateResult run_pose_gate(const SkinnedRig& R, const PoseGateOpts& opt = PoseGateOpts{}) {
    using namespace detail;
    PoseGateResult G;
    const int V = R.V(), F = R.F(), J = R.J();
    if (V == 0 || F == 0 || J == 0) { G.err = "empty mesh or skeleton"; return G; }
    if ((int)R.jidx.size() != V * 4 || (int)R.jw.size() != V * 4) { G.err = "skin shape mismatch"; return G; }
    if ((int)R.local.size() != J || (int)R.ibm.size() != J) { G.err = "skeleton shape mismatch"; return G; }

    std::vector<M4> rest_g;
    global_transforms(R, R.local, rest_g);

    std::vector<double> mass, centres;
    joint_mass_and_centres(R, mass, centres);

    // candidates = non-root joints (a root cannot articulate anything relative to its parent).
    std::vector<int> candidates;
    for (int j = 0; j < J; ++j) if (R.parent[(size_t)j] >= 0) candidates.push_back(j);
    if (candidates.empty()) { G.err = "no non-root skinned joint"; return G; }

    double peak = 0.0;
    for (int j : candidates) peak = std::max(peak, mass[(size_t)j]);

    std::vector<int> influential;
    for (int j : candidates) if (mass[(size_t)j] >= 0.08 * peak) influential.push_back(j);

    std::vector<M4> posed_local = R.local;
    std::vector<int> audit_joints;
    char label[256];

    int left_arm = -1, right_arm = -1;
    if ((int)R.names.size() == J)
        for (int j = 0; j < J; ++j) {
            if (R.names[(size_t)j] == "mixamorig:LeftArm")  left_arm = j;
            if (R.names[(size_t)j] == "mixamorig:RightArm") right_arm = j;
        }

    if (opt.mode == PoseGateOpts::Default && left_arm >= 0 && right_arm >= 0) {
        posed_local[(size_t)left_arm]  = m4_mul(posed_local[(size_t)left_arm],  m4_rotation_z(-90));
        posed_local[(size_t)right_arm] = m4_mul(posed_local[(size_t)right_arm], m4_rotation_z(+90));
        std::snprintf(label, sizeof(label), "arms raised");
    } else {
        int chosen = -1;
        if (opt.mode == PoseGateOpts::GenericJoint) {
            for (int j : candidates) if (j == opt.requested_joint) chosen = j;
            if (chosen < 0) { G.err = "requested joint is not a non-root skinned joint"; return G; }
            std::snprintf(label, sizeof(label), "generic requested joint %02d rotated", chosen);
        } else if (opt.mode == PoseGateOpts::GenericExtremity) {
            float cmn[3] = {INFINITY, INFINITY, INFINITY}, cmx[3] = {-INFINITY, -INFINITY, -INFINITY};
            for (int i = 0; i < V; ++i)
                for (int a = 0; a < 3; ++a) {
                    const float x = R.vertices[(size_t)i * 3 + a];
                    cmn[a] = std::min(cmn[a], x); cmx[a] = std::max(cmx[a], x);
                }
            double mc[3], scale = 1e-6;
            for (int a = 0; a < 3; ++a) { mc[a] = ((double)cmn[a] + (double)cmx[a]) * 0.5;
                                          scale = std::max(scale, (double)(cmx[a] - cmn[a])); }
            double best = -1;
            for (int j : influential) {
                double d = 0;
                for (int a = 0; a < 3; ++a) { const double t = centres[(size_t)j * 3 + a] - mc[a]; d += t * t; }
                const double score = std::sqrt(d) / scale *
                                     (0.25 + 0.75 * std::sqrt(mass[(size_t)j] / std::max(peak, 1e-12)));
                if (score > best) { best = score; chosen = j; }
            }
            std::snprintf(label, sizeof(label), "generic distal joint %02d rotated", chosen);
        } else if (opt.mode == PoseGateOpts::GenericAllInfluential) {
            double best = -1;
            for (int j : influential) if (mass[(size_t)j] > best) { best = mass[(size_t)j]; chosen = j; }
            audit_joints = influential;
            std::snprintf(label, sizeof(label),
                          "generic materially weighted joint %02d rotated; all %d influential joints audited",
                          chosen, (int)audit_joints.size());
        } else {
            double best = -1;
            for (int j : candidates) if (mass[(size_t)j] > best) { best = mass[(size_t)j]; chosen = j; }
            std::snprintf(label, sizeof(label), "generic joint %02d rotated", chosen);
        }
        if (chosen < 0) { G.err = "no articulable joint"; return G; }
        posed_local[(size_t)chosen] = m4_mul(posed_local[(size_t)chosen], m4_rotation_z(45));
        G.chosen_joint = chosen;
    }
    G.pose_label = label;
    G.n_audit = (int)audit_joints.size();

    std::vector<M4> posed_g;
    global_transforms(R, posed_local, posed_g);

    std::vector<float> rest_v, pose_v;
    deform(R, rest_g, rest_v);
    deform(R, posed_g, pose_v);

    float rmn[3] = {INFINITY, INFINITY, INFINITY}, rmx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < V; ++i)
        for (int a = 0; a < 3; ++a) {
            const float x = rest_v[(size_t)i * 3 + a];
            rmn[a] = std::min(rmn[a], x); rmx[a] = std::max(rmx[a], x);
        }
    float dx = rmx[0] - rmn[0], dy = rmx[1] - rmn[1], dz = rmx[2] - rmn[2];
    const double mesh_diag = (double)std::sqrt(dx * dx + dy * dy + dz * dz);

    const float move_thresh = (float)std::max(1e-6, mesh_diag * 0.02);
    size_t moved_n = 0;
    double max_disp = 0;
    for (int i = 0; i < V; ++i) {
        const float ax = pose_v[(size_t)i * 3 + 0] - rest_v[(size_t)i * 3 + 0];
        const float ay = pose_v[(size_t)i * 3 + 1] - rest_v[(size_t)i * 3 + 1];
        const float az = pose_v[(size_t)i * 3 + 2] - rest_v[(size_t)i * 3 + 2];
        const float d = std::sqrt(ax * ax + ay * ay + az * az);
        if (d > move_thresh) ++moved_n;
        max_disp = std::max(max_disp, (double)d);
    }
    G.moved = (double)moved_n / (double)V;
    G.max_disp = max_disp;

    // edges, in the Python's order: all [0,1], then all [1,2], then all [2,0].
    const size_t E = (size_t)F * 3;
    std::vector<int32_t> ea(E), eb(E);
    {
        static const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
        for (int b = 0; b < 3; ++b)
            for (int f = 0; f < F; ++f) {
                ea[(size_t)b * F + f] = R.faces[(size_t)f * 3 + pairs[b][0]];
                eb[(size_t)b * F + f] = R.faces[(size_t)f * 3 + pairs[b][1]];
            }
    }
    auto edge_len = [&](const std::vector<float>& X, size_t e) {
        const float* p = &X[(size_t)ea[e] * 3];
        const float* q = &X[(size_t)eb[e] * 3];
        const float u = p[0] - q[0], v = p[1] - q[1], w = p[2] - q[2];
        return std::sqrt(u * u + v * v + w * w);
    };

    const float len_floor = (float)(mesh_diag * 1e-4);
    std::vector<size_t> valid;       // indices into the edge arrays
    std::vector<float>  rest_len;    // rest length of each VALID edge
    valid.reserve(E); rest_len.reserve(E);
    for (size_t e = 0; e < E; ++e) {
        const float rl = edge_len(rest_v, e);
        if (rl >= len_floor) { valid.push_back(e); rest_len.push_back(rl); }
    }

    auto stretch_of = [&](const std::vector<float>& X, std::vector<float>& out) {
        out.resize(valid.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < (long long)valid.size(); ++i)
            out[(size_t)i] = edge_len(X, valid[(size_t)i]) / std::max(rest_len[(size_t)i], 1e-9f);
    };

    std::vector<float> stretch;
    stretch_of(pose_v, stretch);
    {
        std::vector<float> tmp = stretch;
        G.p99  = quantile_inplace(tmp, 0.99);  tmp = stretch;
        G.p995 = quantile_inplace(tmp, 0.995); tmp = stretch;
        G.p999 = quantile_inplace(tmp, 0.999);
        G.max_stretch = max_of(stretch);
        G.over5  = frac_over(stretch, 5.0f);
        G.over10 = frac_over(stretch, 10.0f);
    }
    G.worst_audit_p999     = G.p999;
    G.worst_component_p999 = G.p999;

    if (!audit_joints.empty()) {
        // Components, on the WELDED rest surface. Tiny numeric fragments (< 30 edges) are ignored;
        // everything else gets its own p999 so a small shredded prop cannot hide inside a global one.
        std::vector<int32_t> fcomp;
        face_component_ids(rest_v, R.faces, fcomp);
        std::vector<int32_t> vcomp(valid.size());
        for (size_t i = 0; i < valid.size(); ++i) vcomp[i] = fcomp[valid[i] % (size_t)F];
        std::vector<int32_t> uniq = vcomp;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        std::vector<std::vector<int32_t>> members(uniq.size());
        for (size_t i = 0; i < vcomp.size(); ++i) {
            const size_t ci = (size_t)(std::lower_bound(uniq.begin(), uniq.end(), vcomp[i]) - uniq.begin());
            members[ci].push_back((int32_t)i);
        }

        struct AuditOut { double p999; double comp_p999; long comp_id; };
        std::vector<AuditOut> outs(audit_joints.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (long long ai = 0; ai < (long long)audit_joints.size(); ++ai) {
            const int aj = audit_joints[(size_t)ai];
            std::vector<M4> al = R.local;
            al[(size_t)aj] = m4_mul(al[(size_t)aj], m4_rotation_z(45));
            std::vector<M4> ag;
            global_transforms(R, al, ag);
            std::vector<float> av;
            deform(R, ag, av);
            std::vector<float> ast;
            ast.resize(valid.size());
            for (size_t i = 0; i < valid.size(); ++i)
                ast[i] = edge_len(av, valid[i]) / std::max(rest_len[i], 1e-9f);
            std::vector<float> tmp = ast;
            outs[(size_t)ai].p999 = quantile_inplace(tmp, 0.999);
            double best = -INFINITY; long best_id = -1;
            std::vector<float> cbuf;
            for (size_t ci = 0; ci < uniq.size(); ++ci) {
                if (members[ci].size() < 30) continue;
                cbuf.resize(members[ci].size());
                for (size_t t = 0; t < members[ci].size(); ++t) cbuf[t] = ast[(size_t)members[ci][t]];
                const double q = quantile_inplace(cbuf, 0.999);
                if (q > best) { best = q; best_id = (long)uniq[ci]; }
            }
            outs[(size_t)ai].comp_p999 = best;
            outs[(size_t)ai].comp_id   = best_id;
        }
        // Reduce in audit order with strict >, so the first joint to reach a value keeps the tag —
        // matching the Python's sequential loop exactly.
        for (size_t ai = 0; ai < audit_joints.size(); ++ai) {
            if (outs[ai].p999 > G.worst_audit_p999) {
                G.worst_audit_p999 = outs[ai].p999;
                G.worst_audit_joint = audit_joints[ai];
            }
            if (outs[ai].comp_id >= 0 && outs[ai].comp_p999 > G.worst_component_p999) {
                G.worst_component_p999  = outs[ai].comp_p999;
                G.worst_component_joint = audit_joints[ai];
                G.worst_component_id    = outs[ai].comp_id;
            }
        }
    }

    // The all-influential sweep necessarily includes joints at narrow wing/feather transitions, so
    // it publishes at a slightly wider threshold than the single named humanoid-arm diagnostic.
    G.gate_limit = audit_joints.empty() ? 5.0 : 6.0;
    G.pass = !(G.moved < 0.010 || G.worst_audit_p999 > G.gate_limit ||
               G.worst_component_p999 > G.gate_limit);
    G.ok = true;
    return G;
}

// The Python's `pose gate:` line, character for character (its trailing-space quirks included), so a
// port regression shows up as a plain text diff.
inline std::string pose_gate_line(const PoseGateResult& G) {
    char b[1024];
    std::string s = "pose gate: pose='" + G.pose_label + "' ";
    std::snprintf(b, sizeof(b),
                  "moved(>2%%diag)=%.3f p99/p995/p999_stretch=%.3f/%.3f/%.3f "
                  "over5/over10=%.5f/%.5f max_stretch=%.3f all-influential-worst=%.3f "
                  "component-worst=%.3f",
                  G.moved, G.p99, G.p995, G.p999, G.over5, G.over10, G.max_stretch,
                  G.worst_audit_p999, G.worst_component_p999);
    s += b;
    if (G.worst_audit_joint >= 0) { std::snprintf(b, sizeof(b), "@joint%02d ", G.worst_audit_joint); s += b; }
    else s += " ";
    if (G.worst_component_joint >= 0) {
        std::snprintf(b, sizeof(b), "component=%ld@joint%02d ", G.worst_component_id, G.worst_component_joint);
        s += b;
    } else s += " ";
    std::snprintf(b, sizeof(b), "(gate: moved>=0.010, p999<=%.3f)", G.gate_limit);
    s += b;
    return s;
}

// ---------------------------------------------------------------------------
// rig_weight_health.py, ported. Cheap (one pass over the skin), and it catches a rig whose
// articulation is nominal — a 50-bone skeleton with all its mass on one joint poses "within
// tolerance" precisely because the pose gate only exercises joints above 8% of peak.
// It does NOT catch the weightless rig: an all-zeros skin field is top-4 encoded as joint 0 = 1.0
// for every vertex, i.e. exactly 1 influential joint, which trips `influential < 4` — verify that
// claim, do not assume it either way.
// ---------------------------------------------------------------------------
inline WeightHealth run_weight_health(const SkinnedRig& R, int min_influential = 4,
                                      double review_single_share = 0.35) {
    WeightHealth H;
    const int V = R.V(), J = R.J();
    H.V = V; H.J = J;
    if (V == 0 || J == 0) return H;
    std::vector<double> mass, centres;
    detail::joint_mass_and_centres(R, mass, centres);
    double total = 0, peak = 0;
    for (int j = 0; j < J; ++j) { total += mass[(size_t)j]; peak = std::max(peak, mass[(size_t)j]); }
    total = std::max(total, 1e-12);
    std::vector<double> dominant((size_t)J, 0.0);
    for (int vi = 0; vi < V; ++vi) {
        int bestk = 0;
        for (int k = 1; k < 4; ++k)
            if (R.jw[(size_t)vi * 4 + k] > R.jw[(size_t)vi * 4 + bestk]) bestk = k;   // numpy argmax: first max wins
        const int j = R.jidx[(size_t)vi * 4 + bestk];
        if (j >= 0 && j < J) dominant[(size_t)j] += 1.0;
    }
    for (int j = 0; j < J; ++j) {
        if (mass[(size_t)j] >= 0.08 * peak) H.influential++;
        if (mass[(size_t)j] / total >= 0.01) H.mass_1pct++;
        if (dominant[(size_t)j] / std::max(V, 1) >= 0.01) H.dominant_1pct++;
        H.single_share = std::max(H.single_share, mass[(size_t)j] / total);
    }
    H.pass   = H.influential >= min_influential;
    H.review = H.single_share > review_single_share;
    return H;
}

// ---------------------------------------------------------------------------
// Build the gate's view of a rig from exactly what the pipeline is about to WRITE: the same
// `rig_local_and_ibm` skeleton encoding and the same `rig_topk4` skin encoding the GLB writer uses.
// `skin_weights` is V*J row-major (the dense transferred field), `joints` J*3 world rest positions.
// ---------------------------------------------------------------------------
inline bool pose_gate_from_rig(const std::vector<float>& verts,          // V*3
                               const std::vector<int64_t>& faces,        // F*3
                               const std::vector<float>& joints,         // J*3 world
                               const std::vector<int>& parents,          // J
                               const std::vector<float>& skin_weights,   // V*J
                               const std::vector<std::string>& names,    // J or empty
                               SkinnedRig& out) {
    const uint32_t V = (uint32_t)(verts.size() / 3);
    const uint32_t J = (uint32_t)(joints.size() / 3);
    if (V == 0 || J == 0 || parents.size() != J) return false;
    if (skin_weights.size() != (size_t)V * J) return false;
    std::vector<float> locals, ibm;
    if (!glb::rig_local_and_ibm(joints, parents, locals, ibm)) return false;
    std::vector<uint16_t> j4; std::vector<float> w4;
    glb::rig_topk4(skin_weights, V, J, j4, w4);

    out = SkinnedRig{};
    out.vertices = verts;
    out.faces.resize(faces.size());
    for (size_t i = 0; i < faces.size(); ++i) out.faces[i] = (int32_t)faces[i];
    out.jidx.resize(j4.size());
    for (size_t i = 0; i < j4.size(); ++i) out.jidx[i] = (int32_t)j4[i];
    out.jw = w4;
    out.parent.assign(parents.begin(), parents.end());
    out.local.assign(J, m4_identity());
    out.ibm.assign(J, m4_identity());
    for (uint32_t j = 0; j < J; ++j) {
        // node local = translation only (glb_rigged.hpp), row-major here.
        out.local[j][3]  = locals[(size_t)j * 3 + 0];
        out.local[j][7]  = locals[(size_t)j * 3 + 1];
        out.local[j][11] = locals[(size_t)j * 3 + 2];
        // ibm is stored COLUMN-major by the writer; transpose into this header's row-major form.
        const float* m = &ibm[(size_t)j * 16];
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out.ibm[j][(size_t)r * 4 + c] = m[c * 4 + r];
    }
    if (names.size() == (size_t)J) out.names = names;
    return true;
}

}  // namespace rigqc
