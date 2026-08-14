// rig_exercise.hpp — bake a "skeleton exercise" clip into a rigged GLB, natively.
//
// The C++ port of rig_exercise_anim.py, which was the LAST Python on the delivery path (called by
// shootout/final_e2e_dc_rig.sh and shootout/final_lod_tiers.sh).  A still pose gate proves that
// ONE joint moves; this walks every materially weighted joint in turn, each swinging about an axis
// perpendicular to its own bone direction and overlapping slightly into the next, so the whole rig
// can be eye-tested in motion in any glTF viewer.  Bad weights are obvious here in a way they are
// not in a still: a joint that drags unrelated geometry, an appendage nothing moves, a limb that
// rubber-bands.
//
// WHAT IS PORTED, AND WHAT IS DELIBERATELY NOT
//   Ported: the joint SELECTION (rig_pose_smoke.mesh_and_skin + generic_joint_masses_and_centres,
//   mass >= 8% of the heaviest, roots dropped, cap 40), the bone-direction/axis rule, the 5-key
//   swing, the slot/stride timing, and the accessor/animation writing.
//   NOT ported: rig_weight_health.py and rig_pose_smoke.py themselves.  Those are fail-closed
//   PUBLISH gates that never run inside the API; rewriting a working gate only risks one that
//   wrongly passes.  This file reproduces the gate's notion of "materially weighted" by porting
//   the two functions the clip needs, so the clip still exercises exactly the joints the gate
//   audits — but the gate stays the Python one.
//
// PARITY CAVEAT, stated rather than hidden: numpy's argsort is introsort, not a stable sort, so
// two joints with EXACTLY equal mass could order differently here (this uses a stable sort with an
// index tiebreak).  Equal float masses do not occur on a transferred skin field, and the clip is
// an eye-test artifact either way — but it is an assumption, not a law.
#pragma once

#include "motion_glb_anim.hpp"   // mret::write_glb, jnumv/jarrv/jobjv, the JSON writer
#include "motion_retarget.hpp"   // mret::Rig / load_rig / glb_split, Q4/V3 maths

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rigex {

using mret::Q4;
using mret::V3;

struct Options {
    double degrees = 25.0;   // swing amplitude, +/- about the rest pose
    double slot = 1.2;       // seconds one joint's full swing occupies
    double stride = 0.6;     // seconds between the START of consecutive joints (so they overlap)
    int    max_joints = 40;
    double mass_frac = 0.08; // keep joints with >= this fraction of the heaviest joint's mass
};

struct Result {
    int    joints = 0;       // joints exercised
    double duration = 0;     // clip length in seconds
    int    skin_joints = 0;  // joints in the selected skin
    int    vertices = 0;     // skinned vertices considered
};

namespace detail {

using J = glb::detail::JVal;

// One resolved accessor, widened to double.  Interleaved accessors are REFUSED rather than
// silently mis-read — the same refusal rig_pose_smoke.accessor() makes.
struct Acc {
    int64_t count = 0, cols = 0;
    std::vector<double> v;
    bool ok = false;
};

inline Acc read_accessor(const J& g, const std::vector<uint8_t>& bin, int64_t idx, std::string& err) {
    Acc a;
    const J* accs = g.find("accessors");
    const J* bvs = g.find("bufferViews");
    if (!accs || !bvs || idx < 0 || idx >= (int64_t)accs->arr.size()) { err = "bad accessor index"; return a; }
    const J& acc = accs->arr[(size_t)idx];
    const J* bvj = acc.find("bufferView");
    if (!bvj) { err = "sparse accessor unsupported"; return a; }
    const J& bv = bvs->arr[(size_t)bvj->as_int()];
    const int ct = (int)acc.find("componentType")->as_int();
    const std::string type = acc.find("type")->str;
    a.count = acc.find("count")->as_int();
    a.cols = glb::detail::type_count(type);
    const int csz = glb::detail::comp_size(ct);
    if (!a.cols || !csz) { err = "unsupported accessor type"; return a; }
    const int64_t elem = a.cols * csz;
    const int64_t stride = bv.find("byteStride") ? bv.find("byteStride")->as_int() : elem;
    if (stride != elem) { err = "interleaved accessor unsupported"; return a; }
    int64_t off = (bv.find("byteOffset") ? bv.find("byteOffset")->as_int() : 0) +
                  (acc.find("byteOffset") ? acc.find("byteOffset")->as_int() : 0);
    if (off + a.count * elem > (int64_t)bin.size()) { err = "accessor overruns BIN"; return a; }
    a.v.resize((size_t)(a.count * a.cols));
    const uint8_t* p = bin.data() + off;
    for (int64_t i = 0; i < a.count * a.cols; i++) {
        const uint8_t* q = p + i * csz;
        switch (ct) {
            case 5120: { int8_t x;   std::memcpy(&x, q, 1); a.v[(size_t)i] = x; break; }
            case 5121: { uint8_t x;  std::memcpy(&x, q, 1); a.v[(size_t)i] = x; break; }
            case 5122: { int16_t x;  std::memcpy(&x, q, 2); a.v[(size_t)i] = x; break; }
            case 5123: { uint16_t x; std::memcpy(&x, q, 2); a.v[(size_t)i] = x; break; }
            case 5125: { uint32_t x; std::memcpy(&x, q, 4); a.v[(size_t)i] = x; break; }
            case 5126: { float x;    std::memcpy(&x, q, 4); a.v[(size_t)i] = x; break; }
            default: err = "unsupported componentType"; return a;
        }
    }
    // glTF allows a NORMALIZED integer WEIGHTS_0; the generated rigs are f32, but a source-retained
    // rig may not be, and reading 65535 as a weight would put every joint's mass in one bone.
    if (acc.find("normalized") && acc.find("normalized")->b) {
        double denom = 0;
        switch (ct) { case 5121: denom = 255.0; break; case 5123: denom = 65535.0; break;
                      case 5120: denom = 127.0; break; case 5122: denom = 32767.0; break; }
        if (denom > 0) for (auto& x : a.v) x = std::max(-1.0, x / denom);
    }
    a.ok = true;
    return a;
}

}  // namespace detail

// Port of rig_pose_smoke.mesh_and_skin + generic_joint_masses_and_centres, reduced to what the
// exercise clip needs: the per-joint influence mass and its weighted vertex centroid.
struct SkinStats {
    std::vector<int>    skin_nodes;    // skin.joints, i.e. joint slot -> glTF node
    std::vector<double> mass;          // per joint slot
    std::vector<V3>     centres;       // per joint slot
    int                 vertices = 0;
    bool                ok = false;
};

inline SkinStats skin_stats(const glb::detail::JVal& g, const std::vector<uint8_t>& bin,
                            std::string& err) {
    using J = glb::detail::JVal;
    SkinStats S;
    const J* nodes = g.find("nodes");
    const J* meshes = g.find("meshes");
    const J* skins = g.find("skins");
    if (!nodes || !meshes || !skins || !skins->is_arr() || skins->arr.empty()) {
        err = "no skin, nothing to exercise";
        return S;
    }
    // "First usable skinned primitive", not "node zero": a source asset can carry a non-skinned
    // accessory mesh before its actual skinned character mesh in node order.
    auto usable = [&](const J& prim) {
        const J* at = prim.find("attributes");
        return at && at->find("POSITION") && at->find("JOINTS_0") && at->find("WEIGHTS_0") &&
               prim.find("indices");
    };
    int skin_i = -1;
    for (size_t n = 0; n < nodes->arr.size() && skin_i < 0; n++) {
        const J& nd = nodes->arr[n];
        if (!nd.find("mesh") || !nd.find("skin")) continue;
        const J* prims = meshes->arr[(size_t)nd.find("mesh")->as_int()].find("primitives");
        if (!prims) continue;
        for (const auto& pr : prims->arr)
            if (usable(pr)) { skin_i = (int)nd.find("skin")->as_int(); break; }
    }
    if (skin_i < 0) { err = "no indexed skinned primitive with POSITION/JOINTS_0/WEIGHTS_0"; return S; }

    // Every identity-transform primitive sharing that skin, so a valid-looking tiny accessory
    // cannot mask a bad body skin.  A transformed multi-node source needs a scene-aware renderer;
    // refuse rather than silently apply its node transform with the wrong skin convention.
    std::vector<const J*> cands;
    for (size_t n = 0; n < nodes->arr.size(); n++) {
        const J& nd = nodes->arr[n];
        if (!nd.find("skin") || (int)nd.find("skin")->as_int() != skin_i || !nd.find("mesh")) continue;
        for (const char* k : {"matrix", "translation", "rotation", "scale"})
            if (nd.find(k)) { err = "multi-primitive source needs identity skinned mesh-node transforms"; return S; }
        const J* prims = meshes->arr[(size_t)nd.find("mesh")->as_int()].find("primitives");
        if (!prims) continue;
        for (const auto& pr : prims->arr) if (usable(pr)) cands.push_back(&pr);
    }
    if (cands.empty()) { err = "selected skin has no usable primitives"; return S; }

    const J* sj = skins->arr[(size_t)skin_i].find("joints");
    if (!sj) { err = "skin has no joints"; return S; }
    for (const auto& v : sj->arr) S.skin_nodes.push_back((int)v.as_int());
    const size_t JN = S.skin_nodes.size();
    S.mass.assign(JN, 0.0);
    std::vector<double> wpos(JN * 3, 0.0);

    for (const J* pr : cands) {
        const J* at = pr->find("attributes");
        detail::Acc P = detail::read_accessor(g, bin, at->find("POSITION")->as_int(), err);
        detail::Acc Jo = detail::read_accessor(g, bin, at->find("JOINTS_0")->as_int(), err);
        detail::Acc W = detail::read_accessor(g, bin, at->find("WEIGHTS_0")->as_int(), err);
        if (!P.ok || !Jo.ok || !W.ok) return S;
        if (P.count != Jo.count || P.count != W.count) { err = "attribute counts differ"; return S; }
        S.vertices += (int)P.count;
        for (int64_t v = 0; v < P.count; v++)
            for (int s = 0; s < 4; s++) {
                const size_t id = (size_t)Jo.v[(size_t)(v * Jo.cols + s)];
                const double w = W.v[(size_t)(v * W.cols + s)];
                if (id >= JN) { err = "JOINTS_0 index out of range"; return S; }
                S.mass[id] += w;
                for (int ax = 0; ax < 3; ax++) wpos[id * 3 + ax] += w * P.v[(size_t)(v * 3 + ax)];
            }
    }
    S.centres.assign(JN, V3{});
    for (size_t j = 0; j < JN; j++) {
        const double m = std::max(S.mass[j], 1e-12);
        S.centres[j] = V3{wpos[j * 3] / m, wpos[j * 3 + 1] / m, wpos[j * 3 + 2] / m};
    }
    S.ok = true;
    return S;
}

// Bake the exercise clip into `src_glb` and write `out_glb`.
inline bool bake_exercise(const char* src_glb, const char* out_glb, const Options& opt, Result& R,
                          std::string& err) {
    using J = glb::detail::JVal;
    std::vector<uint8_t> file, bin;
    J g;
    if (!mret::glb_split(src_glb, file, g, bin, err)) return false;
    SkinStats S = skin_stats(g, bin, err);
    if (!S.ok) return false;
    R.skin_joints = (int)S.skin_nodes.size();
    R.vertices = S.vertices;

    const J* nodes = g.find("nodes");
    std::vector<int> parent(nodes->arr.size(), -1);
    for (size_t n = 0; n < nodes->arr.size(); n++)
        if (const J* c = nodes->arr[n].find("children"))
            for (const auto& cv : c->arr) parent[(size_t)cv.as_int()] = (int)n;
    std::vector<int> node_to_slot(nodes->arr.size(), -1);
    for (size_t j = 0; j < S.skin_nodes.size(); j++) node_to_slot[(size_t)S.skin_nodes[j]] = (int)j;

    // order = joints by descending influence mass, keeping only the materially weighted ones,
    // capped, with SKELETON ROOTS dropped (rotating the root moves the whole character, which
    // tells you nothing about the weights).
    double mmax = 0;
    for (double m : S.mass) mmax = std::max(mmax, m);
    std::vector<int> order;
    for (size_t j = 0; j < S.mass.size(); j++) order.push_back((int)j);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return S.mass[(size_t)a] > S.mass[(size_t)b]; });
    {
        std::vector<int> keep;
        for (int j : order) {
            if (S.mass[(size_t)j] < opt.mass_frac * mmax) continue;
            if ((int)keep.size() >= opt.max_joints) break;
            keep.push_back(j);
        }
        order.swap(keep);
    }
    {
        std::vector<int> keep;
        for (int j : order) {
            const int pn = parent[(size_t)S.skin_nodes[(size_t)j]];
            const bool is_root = pn < 0 || node_to_slot[(size_t)pn] < 0;
            if (!is_root) keep.push_back(j);
        }
        order.swap(keep);
    }
    if (order.empty()) { err = "no non-root influential joint"; return false; }

    // ---- append the animation ----
    J* bufferViews = const_cast<J*>(g.find("bufferViews"));
    if (!bufferViews) { g.obj["bufferViews"] = mret::jarrv(); bufferViews = &g.obj["bufferViews"]; }
    J* accessors = const_cast<J*>(g.find("accessors"));
    if (!accessors) { g.obj["accessors"] = mret::jarrv(); accessors = &g.obj["accessors"]; }
    if (!g.find("animations")) g.obj["animations"] = mret::jarrv();

    auto add_view = [&](const void* data, size_t nbytes) {
        while (bin.size() % 4) bin.push_back(0);
        const size_t off = bin.size();
        const uint8_t* p = (const uint8_t*)data;
        bin.insert(bin.end(), p, p + nbytes);
        J bv = mret::jobjv();
        bv.obj["buffer"] = mret::jnumv(0);
        bv.obj["byteOffset"] = mret::jnumv((double)off);
        bv.obj["byteLength"] = mret::jnumv((double)nbytes);
        bufferViews->arr.push_back(bv);
        return (int)bufferViews->arr.size() - 1;
    };

    J samplers = mret::jarrv(), channels = mret::jarrv();
    for (size_t slot_i = 0; slot_i < order.size(); slot_i++) {
        const int j = order[slot_i];
        const int node_i = S.skin_nodes[(size_t)j];
        const J& node = nodes->arr[(size_t)node_i];

        // Bone direction: mean child offset, else the offset to this joint's own influence
        // centroid.  Swing about an axis PERPENDICULAR to it, so the motion is maximally visible
        // rather than a twist about the bone (which a viewer cannot see at all).
        V3 d{};
        int nk = 0;
        if (const J* ch = node.find("children"))
            for (const auto& cv : ch->arr) {
                const J& kid = nodes->arr[(size_t)cv.as_int()];
                V3 t{};
                if (const J* tr = kid.find("translation"))
                    t = V3{tr->arr[0].as_double(), tr->arr[1].as_double(), tr->arr[2].as_double()};
                d = d + t;
                nk++;
            }
        if (nk) d = d * (1.0 / nk);
        V3 self_t{};
        if (const J* tr = node.find("translation"))
            self_t = V3{tr->arr[0].as_double(), tr->arr[1].as_double(), tr->arr[2].as_double()};
        if (mret::norm(d) < 1e-6) d = S.centres[(size_t)j] - self_t;
        if (mret::norm(d) < 1e-6) d = V3{0, 1, 0};
        const V3 ref = std::fabs(d.z / std::max(mret::norm(d), 1e-9)) < 0.9 ? V3{0, 0, 1} : V3{1, 0, 0};
        const V3 axis = mret::cross(d, ref);

        Q4 base{0, 0, 0, 1};
        if (const J* r = node.find("rotation"))
            base = Q4{r->arr[0].as_double(), r->arr[1].as_double(), r->arr[2].as_double(), r->arr[3].as_double()};

        const double ang = opt.degrees * M_PI / 180.0;
        const double t0 = (double)slot_i * opt.stride;
        const float times[5] = {(float)t0, (float)(t0 + opt.slot * 0.25), (float)(t0 + opt.slot * 0.5),
                                (float)(t0 + opt.slot * 0.75), (float)(t0 + opt.slot)};
        // axis-angle about `axis`, applied on the RIGHT of the node's own rest rotation, exactly as
        // quat_mul(base, quat_from_axis_angle(axis, a)) does.
        auto aa = [&](double a) {
            const double n = mret::norm(axis);
            if (n < 1e-9) return Q4{0, 0, 0, 1};
            const V3 u = axis * (1.0 / n);
            const double s = std::sin(a * 0.5);
            return Q4{u.x * s, u.y * s, u.z * s, std::cos(a * 0.5)};
        };
        const Q4 q[5] = {mret::q_mul(base, aa(0.0)), mret::q_mul(base, aa(ang)), mret::q_mul(base, aa(0.0)),
                         mret::q_mul(base, aa(-ang)), mret::q_mul(base, aa(0.0))};
        float qf[20];
        for (int k = 0; k < 5; k++) {
            qf[k * 4 + 0] = (float)q[k].x; qf[k * 4 + 1] = (float)q[k].y;
            qf[k * 4 + 2] = (float)q[k].z; qf[k * 4 + 3] = (float)q[k].w;
        }

        const int tv = add_view(times, sizeof(times));
        J ta = mret::jobjv();
        ta.obj["bufferView"] = mret::jnumv(tv);
        ta.obj["componentType"] = mret::jnumv(5126);
        ta.obj["count"] = mret::jnumv(5);
        ta.obj["type"] = mret::jstrv("SCALAR");
        { J mn = mret::jarrv(); mn.arr.push_back(mret::jnumv(times[0])); ta.obj["min"] = mn;
          J mx = mret::jarrv(); mx.arr.push_back(mret::jnumv(times[4])); ta.obj["max"] = mx; }
        accessors->arr.push_back(ta);
        const int t_acc = (int)accessors->arr.size() - 1;

        const int qv = add_view(qf, sizeof(qf));
        J qa = mret::jobjv();
        qa.obj["bufferView"] = mret::jnumv(qv);
        qa.obj["componentType"] = mret::jnumv(5126);
        qa.obj["count"] = mret::jnumv(5);
        qa.obj["type"] = mret::jstrv("VEC4");
        accessors->arr.push_back(qa);

        J s = mret::jobjv();
        s.obj["input"] = mret::jnumv(t_acc);
        s.obj["output"] = mret::jnumv((double)accessors->arr.size() - 1);
        s.obj["interpolation"] = mret::jstrv("LINEAR");
        samplers.arr.push_back(s);
        J tgt = mret::jobjv();
        tgt.obj["node"] = mret::jnumv(node_i);
        tgt.obj["path"] = mret::jstrv("rotation");
        J ch = mret::jobjv();
        ch.obj["sampler"] = mret::jnumv((double)samplers.arr.size() - 1);
        ch.obj["target"] = tgt;
        channels.arr.push_back(ch);
    }

    J an = mret::jobjv();
    an.obj["name"] = mret::jstrv("skeleton-exercise");
    an.obj["samplers"] = samplers;
    an.obj["channels"] = channels;
    g.obj["animations"].arr.push_back(an);

    // The Python writer pads the blob to 4 and then rewrites buffers[0].byteLength to the PADDED
    // length, so match that: a byteLength short of the real chunk makes strict loaders reject the
    // last accessor.
    while (bin.size() % 4) bin.push_back(0);
    J* bufs = const_cast<J*>(g.find("buffers"));
    if (bufs && bufs->is_arr() && !bufs->arr.empty()) bufs->arr[0].obj["byteLength"] = mret::jnumv((double)bin.size());

    R.joints = (int)order.size();
    R.duration = (double)(order.size() - 1) * opt.stride + opt.slot;
    return mret::write_glb(out_glb, g, bin, err);
}

}  // namespace rigex
