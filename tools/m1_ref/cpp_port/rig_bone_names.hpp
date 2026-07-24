// rig_bone_names.hpp — name a SkinTokens skeleton to a standard humanoid convention.
//
// WHY: SkinTokens emits unnamed `bone_0..bone_N` joints, and J is MESH-DEPENDENT, so
// nothing can be mapped by index. Naming is derived from STRUCTURE ONLY: the parent tree
// plus the joints' rest-pose 3D positions. Input is exactly rig::RigResult{joints,parents}.
//
// Header-only, no ggml/CUDA/json deps. Pure function + a falsifier.
//
// ---------------------------------------------------------------------------------------
// PROVENANCE / WHAT IS BORROWED
//   The spine/leg/arm topology walk follows /mnt/hdd/3d/avatar-shootout/bonemap.py, with
//   the two fixes from puppy-eyetest/anim/tools/bonemap_v2.py (spine = highest-climbing
//   root child; legs = the OTHER descending children). Stock bonemap.py CRASHES on the
//   soldier rig; its "verified" status in MIXAMO_REPORT.md was gilly-only.
//   The finger classifier follows ComfyUI-SkinToken's _assign_finger_names.
//
// ---------------------------------------------------------------------------------------
// CHIRALITY — THE LOAD-BEARING ASSUMPTION. READ THIS.
//
//   A bare skeleton (points + tree) does NOT determine left from right. A mirrored
//   skeleton is not a rotation of the original, so "which side is left" is only
//   recoverable from a FACING cue. The one facing cue present in a bare humanoid
//   skeleton is the FEET: toes lie forward of ankles.
//
//     forward = sign( sum over both feet of (toe.z - ankle.z) )      [+1 => faces +Z]
//     right   = up x forward = yhat x (f * zhat) = f * xhat          [so right_sign = f]
//
//   MEASURED, NOT ASSUMED: this is NOT a constant of the SkinTokens output.
//     soldier1536_rigged.glb : faces +Z  (char right = +X)
//     gilly_rigged.glb       : faces -Z  (char right = -X)
//   Both were confirmed independently by rendering the mesh from +Z and -Z and looking
//   at which side the face is on. The toe cue agreed with the render on both.
//
//   => bonemap.py's hardcoded `char-left = -X` is WRONG for gilly, and the gilly map in
//      MIXAMO_REPORT.md is consequently L/R-MIRRORED. That is why this file derives
//      facing per-character instead of hardcoding it.
//
//   Override: RIG_BONE_FACING=+z|-z  (or NameOpts::facing_override).
// ---------------------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace rig {

// Axis convention of the glTF rest space these rigs live in.
enum { AX_LAT = 0, AX_UP = 1, AX_FWD = 2 };

// ---- canonical internal slots = SMPL-22 ordering (bonemap.py / retarget_delta.py) -------
// 0 pelvis 1/2 L/R_hip 3 spine1 4/5 L/R_knee 6 spine2 7/8 L/R_ankle 9 spine3
// 10/11 L/R_foot 12 neck 13/14 L/R_collar 15 head 16/17 L/R_shoulder 18/19 L/R_elbow
// 20/21 L/R_wrist
enum { SMPL_N = 22 };

static const char* kSmplNames[SMPL_N] = {
    "pelvis",   "L_hip",      "R_hip",      "spine1",   "L_knee",   "R_knee",
    "spine2",   "L_ankle",    "R_ankle",    "spine3",   "L_foot",   "R_foot",
    "neck",     "L_collar",   "R_collar",   "head",     "L_shoulder","R_shoulder",
    "L_elbow",  "R_elbow",    "L_wrist",    "R_wrist"};

// SMPL-22 slot -> Mixamo bone name (mirrors rename_to_mixamo.py's SMPL_TO_MIXAMO).
// Mixamo has no collar/clavicle distinct from Shoulder, so collar -> *Shoulder and
// SMPL's shoulder -> *Arm, which is the standard Mixamo humanoid arm chain.
static const char* kMixamoNames[SMPL_N] = {
    "mixamorig:Hips",        "mixamorig:LeftUpLeg",   "mixamorig:RightUpLeg",
    "mixamorig:Spine",       "mixamorig:LeftLeg",     "mixamorig:RightLeg",
    "mixamorig:Spine1",      "mixamorig:LeftFoot",    "mixamorig:RightFoot",
    "mixamorig:Spine2",      "mixamorig:LeftToeBase", "mixamorig:RightToeBase",
    "mixamorig:Neck",        "mixamorig:LeftShoulder","mixamorig:RightShoulder",
    "mixamorig:Head",        "mixamorig:LeftArm",     "mixamorig:RightArm",
    "mixamorig:LeftForeArm", "mixamorig:RightForeArm","mixamorig:LeftHand",
    "mixamorig:RightHand"};

// L/R slot pairs, for the symmetry falsifier. {left_slot, right_slot}
static const int kLRPairs[][2] = {{1,2},{4,5},{7,8},{10,11},{13,14},{16,17},{18,19},{20,21}};
enum { LR_PAIR_N = 8 };

enum class NameStyle { Mixamo, SmplH };

struct NameOpts {
    NameStyle style = NameStyle::Mixamo;
    int  facing_override = 0;    // 0 = auto-derive; +1 = faces +Z; -1 = faces -Z
    bool name_fingers = true;
    bool verbose = true;
};

struct BoneNaming {
    std::vector<std::string> names;   // [J] final emitted names
    std::vector<int>         smpl;    // [J] SMPL-22 slot or -1
    int   facing = 0;                 // +1 faces +Z, -1 faces -Z
    int   right_sign = 0;             // sign of X that is the character's RIGHT
    float facing_margin = 0.f;        // |toe-ankle .z| summed, / body height (PRIMARY cue)
    // Second, INDEPENDENT chirality cue: the head joint sits forward of the neck.
    // Different joints, different anatomy from the toe cue -- so it can contradict it.
    // WEAK: measured at only 1.2% (soldier) / 1.6% (gilly) of body height, vs the toe cue's
    // 24.2% / 9.6%. Validated on n=2 characters. Used to cross-check, never to decide.
    float cue_head_margin = 0.f;      // |head.z - neck.z| / height
    int   cue_head_sign = 0;          // +1 / -1 / 0 if unavailable
    bool  facing_from_override = false;
    int   named_core = 0;             // unique joints assigned among 22 slots
    int   named_fingers = 0;
    int   n_extra = 0;                // joints given stable non-semantic skintokens:Extra_NNN names
    bool  ok = false;                 // topology parse succeeded
    std::vector<std::string> notes;   // non-fatal observations
    std::string fail_reason;          // set when ok == false
};

// Non-humanoid rigs must not be forced through the Mixamo topology parser:
// doing so turns a valid bird, dragon, or creature skeleton into a false
// "missing arms" error.  TokenRig does not emit semantic joint labels, so the
// only honest universal contract is a stable, unambiguous joint namespace.
// The source index is retained in every emitted name; semantic labels remain
// reserved for a separately validated creature-specific mapper.
struct GenericBoneNaming {
    std::vector<std::string> names;
    int root = -1;
    int max_fan = 0;
    bool ok = false;
    std::string fail_reason;
};

namespace bn_detail {

struct Tree {
    int J = 0;
    std::vector<std::vector<int>> ch;
    std::vector<int> par;
    int root = -1;
    std::vector<float> up_max, up_min, lat_max;   // subtree extents
};

inline float jat(const std::vector<float>& j, int i, int ax) { return j[(size_t)i * 3 + ax]; }

inline void subtree_extents(Tree& t, const std::vector<float>& J, int b) {
    float u = jat(J, b, AX_UP), d = jat(J, b, AX_UP), l = std::fabs(jat(J, b, AX_LAT));
    for (int c : t.ch[b]) {
        subtree_extents(t, J, c);
        u = std::max(u, t.up_max[c]);
        d = std::min(d, t.up_min[c]);
        l = std::max(l, t.lat_max[c]);
    }
    t.up_max[b] = u; t.up_min[b] = d; t.lat_max[b] = l;
}

inline bool build_tree(Tree& t, const std::vector<float>& J, const std::vector<int>& par,
                       std::string& err, int min_joints = 10) {
    t.J = (int)par.size();
    if (t.J < min_joints) {
        err = "skeleton too small (<" + std::to_string(min_joints) + " joints)";
        return false;
    }
    if ((int)(J.size() / 3) != t.J) { err = "joints/parents length mismatch"; return false; }
    t.par = par;
    t.ch.assign(t.J, {});
    t.root = -1;
    int roots = 0;
    for (int i = 0; i < t.J; i++) {
        int p = par[i];
        if (p == -1) { t.root = i; roots++; }
        else if (p < -1) { err = "parent index below -1"; return false; }
        else if (p >= t.J) { err = "parent index out of range"; return false; }
        else t.ch[p].push_back(i);
    }
    if (roots != 1) {
        err = roots == 0 ? "no root joint (none with parent == -1)" : "multiple root joints";
        return false;
    }
    // A parent array can contain an otherwise invisible cycle or a disconnected
    // island.  Do not name or export a partial hierarchy as a valid creature.
    std::vector<unsigned char> seen((size_t)t.J, 0);
    std::vector<int> todo{t.root};
    int reached = 0;
    while (!todo.empty()) {
        int b = todo.back();
        todo.pop_back();
        if (seen[b]) { err = "parent graph is not a tree"; return false; }
        seen[b] = 1;
        reached++;
        for (int c : t.ch[b]) todo.push_back(c);
    }
    if (reached != t.J) { err = "parent graph is disconnected or cyclic"; return false; }
    t.up_max.assign(t.J, 0); t.up_min.assign(t.J, 0); t.lat_max.assign(t.J, 0);
    subtree_extents(t, J, t.root);
    return true;
}

// follow `n` joints from `start`, each step taking the child that maximises `key`.
template <typename KeyFn>
inline std::vector<int> follow(const Tree& t, int start, int n, KeyFn key) {
    std::vector<int> out{start};
    int cur = start;
    for (int k = 1; k < n; k++) {
        if (t.ch[cur].empty()) break;
        int best = t.ch[cur][0];
        float bs = key(best);
        for (int c : t.ch[cur]) { float s = key(c); if (s > bs) { bs = s; best = c; } }
        cur = best;
        out.push_back(cur);
    }
    return out;
}

// linear branch walk used for fingers: keep taking the single longest-reaching child.
inline std::vector<int> linear_branch(const Tree& t, int start) {
    std::vector<int> out{start};
    int cur = start;
    while (!t.ch[cur].empty()) {
        int best = t.ch[cur][0];
        float bs = -1e30f;
        for (int c : t.ch[cur]) {
            float s = t.lat_max[c] + (t.up_max[c] - t.up_min[c]);
            if (s > bs) { bs = s; best = c; }
        }
        cur = best;
        out.push_back(cur);
        if ((int)out.size() > 8) break;
    }
    return out;
}

inline void norm3(float* v) {
    float n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n > 1e-9f) { v[0]/=n; v[1]/=n; v[2]/=n; }
}

// printf-style into a std::string (a lambda cannot take C varargs, so this is a free fn).
inline std::string bn_fmt(const char* fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    return std::string(msg);
}

}  // namespace bn_detail

// -----------------------------------------------------------------------------------------
// name_bones — the whole thing.
// -----------------------------------------------------------------------------------------
inline GenericBoneNaming name_generic_bones(const std::vector<float>& joints,
                                            const std::vector<int>& parents) {
    using namespace bn_detail;
    GenericBoneNaming R;
    const int J = (int)parents.size();
    if (J < 2) { R.fail_reason = "need at least two joints"; return R; }
    if ((int)(joints.size() / 3) != J) { R.fail_reason = "joints/parents length mismatch"; return R; }
    for (float joint : joints) {
        if (!std::isfinite(joint)) { R.fail_reason = "non-finite joint coordinate"; return R; }
    }
    Tree tree;
    if (!build_tree(tree, joints, parents, R.fail_reason, 2)) return R;
    // A syntactically valid parent tree is not enough for an arbitrary
    // creature.  The autoregressive decoder can occasionally keep emitting
    // parent tokens after its quantised joint coordinate has stopped moving.
    // That makes a 100+-joint, zero-length chain which passes a root/fan-only
    // check but cannot be a useful skinning skeleton.  Do not use a depth
    // limit here: a real snake or tail can legitimately be deep.  Instead
    // reject the geometry failure itself, while retaining any long chain that
    // actually advances through 3-D space.
    std::vector<std::array<float, 3>> unique_points;
    unique_points.reserve((size_t)J);
    float lo[3] = {joints[0], joints[1], joints[2]};
    float hi[3] = {lo[0], lo[1], lo[2]};
    for (int j = 0; j < J; ++j) {
        const float x = joints[(size_t)j * 3 + 0];
        const float y = joints[(size_t)j * 3 + 1];
        const float z = joints[(size_t)j * 3 + 2];
        unique_points.push_back({x, y, z});
        lo[0] = std::min(lo[0], x); lo[1] = std::min(lo[1], y); lo[2] = std::min(lo[2], z);
        hi[0] = std::max(hi[0], x); hi[1] = std::max(hi[1], y); hi[2] = std::max(hi[2], z);
    }
    std::sort(unique_points.begin(), unique_points.end());
    unique_points.erase(std::unique(unique_points.begin(), unique_points.end()), unique_points.end());
    if ((int)unique_points.size() * 2 < J) {
        R.fail_reason = "more than half of generic joints collapse onto duplicate positions";
        return R;
    }
    const float diag = std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0]) +
                                 (hi[1]-lo[1])*(hi[1]-lo[1]) +
                                 (hi[2]-lo[2])*(hi[2]-lo[2]));
    const float zero_eps2 = std::max(1e-16f, diag * diag * 1e-10f);
    int collapsed_edges = 0;
    for (int j = 0; j < J; ++j) {
        const int p = parents[j];
        if (p < 0) continue;
        const float dx = joints[(size_t)j * 3 + 0] - joints[(size_t)p * 3 + 0];
        const float dy = joints[(size_t)j * 3 + 1] - joints[(size_t)p * 3 + 1];
        const float dz = joints[(size_t)j * 3 + 2] - joints[(size_t)p * 3 + 2];
        collapsed_edges += dx*dx + dy*dy + dz*dz <= zero_eps2;
    }
    if (collapsed_edges * 4 > J - 1) {
        R.fail_reason = "more than one quarter of generic parent edges have zero length";
        return R;
    }
    R.root = tree.root;
    for (const auto& children : tree.ch) R.max_fan = std::max(R.max_fan, (int)children.size());
    R.names.resize(J);
    for (int j = 0; j < J; ++j) {
        char name[64];
        if (j == R.root) std::snprintf(name, sizeof(name), "skintokens:Root_%03d", j);
        else std::snprintf(name, sizeof(name), "skintokens:Joint_%03d", j);
        R.names[j] = name;
    }
    R.ok = true;
    return R;
}

// Broad-creature guard, not an anatomy prior: permit an octopus/spider-like
// trunk while still rejecting a collapsed star tree.  At least eight direct
// branches are allowed; bigger rigs scale linearly at 20% of their joints.
inline int generic_max_fan_limit(int J) {
    return std::max(8, (J + 4) / 5);
}

// The generic decoder sometimes emits a geometrically useful joint cloud but
// attaches a large fraction of it directly to one token parent.  This is not
// an anatomy decision: the stable generic namespace has no limb semantics to
// infer.  Keep the decoded tree intact except for the overflow children of
// the pathological star, which are attached to their closest retained sibling
// with remaining branch capacity.  This is a deterministic native topology
// normalization rather than a permissive fan gate or an upstream/Python
// replacement.
//
// The trigger is an actual size-aware fan violation, not a relaxed gate:
// callers must still run the real GLB pose gate on the resulting skin.
inline bool normalize_generic_parent_fan(const std::vector<float>& joints,
                                         std::vector<int>& parents,
                                         std::string* detail = nullptr) {
    const int J = (int)parents.size();
    if (J < 2 || (int)joints.size() != J * 3) return false;
    GenericBoneNaming before = name_generic_bones(joints, parents);
    if (!before.ok) return false;
    const int limit = generic_max_fan_limit(J);
    if (before.max_fan <= limit) return false;

    std::vector<int> out = parents;
    for (int iteration = 0; iteration < J; ++iteration) {
        std::vector<std::vector<int>> children((size_t)J);
        for (int j = 0; j < J; ++j) if (out[(size_t)j] >= 0) children[(size_t)out[(size_t)j]].push_back(j);
        int star = -1;
        for (int j = 0; j < J; ++j) if ((int)children[(size_t)j].size() > limit) { star = j; break; }
        if (star < 0) break;
        auto& direct = children[(size_t)star];
        std::sort(direct.begin(), direct.end(), [&](int a, int b) {
            const float adx=joints[(size_t)a*3]-joints[(size_t)star*3], ady=joints[(size_t)a*3+1]-joints[(size_t)star*3+1], adz=joints[(size_t)a*3+2]-joints[(size_t)star*3+2];
            const float bdx=joints[(size_t)b*3]-joints[(size_t)star*3], bdy=joints[(size_t)b*3+1]-joints[(size_t)star*3+1], bdz=joints[(size_t)b*3+2]-joints[(size_t)star*3+2];
            const float da=adx*adx+ady*ady+adz*adz, db=bdx*bdx+bdy*bdy+bdz*bdz;
            return da != db ? da < db : a < b;
        });
        std::vector<int> anchors(direct.begin(), direct.begin() + limit);
        for (size_t n = (size_t)limit; n < direct.size(); ++n) {
            const int child = direct[n];
            int best = -1; float best_d2 = std::numeric_limits<float>::infinity();
            for (int candidate : anchors) {
                if ((int)children[(size_t)candidate].size() >= limit) continue;
                const float dx=joints[(size_t)child*3]-joints[(size_t)candidate*3], dy=joints[(size_t)child*3+1]-joints[(size_t)candidate*3+1], dz=joints[(size_t)child*3+2]-joints[(size_t)candidate*3+2];
                const float d2=dx*dx+dy*dy+dz*dz;
                if (d2 < best_d2 || (d2 == best_d2 && candidate < best)) { best_d2=d2; best=candidate; }
            }
            if (best < 0) return false;
            out[(size_t)child] = best;
            children[(size_t)best].push_back(child);
        }
    }
    GenericBoneNaming after = name_generic_bones(joints, out);
    if (!after.ok || after.max_fan > limit) return false;
    parents.swap(out);
    if (detail) *detail = bn_detail::bn_fmt("generic local fan normalization: maxfan=%d->%d/%d, J=%d",
                                             before.max_fan, after.max_fan, limit, J);
    return true;
}

inline BoneNaming name_bones(const std::vector<float>& joints,
                             const std::vector<int>&   parents,
                             const NameOpts&           opt = NameOpts{}) {
    using namespace bn_detail;
    BoneNaming R;
    const int J = (int)parents.size();
    R.names.assign(J, "");
    R.smpl.assign(J, -1);
    if ((int)(joints.size() / 3) != J) { R.fail_reason = "joints/parents length mismatch"; return R; }

    Tree t;
    if (!build_tree(t, joints, parents, R.fail_reason)) return R;

    std::vector<int> slot_of(SMPL_N, -1);   // SMPL slot -> joint index

    // ---- 1. spine start = root child whose subtree CLIMBS HIGHEST (bonemap_v2 FIX-1).
    // Stock bonemap.py took "whatever is left after removing legs", which mis-fires when
    // hanging hands make the spine subtree dip below the hips (the soldier: fingertips at
    // y=-0.309 vs hips y=-0.238) and out-reach the legs laterally (0.637 vs 0.199).
    const std::vector<int>& rc = t.ch[t.root];
    if (rc.empty()) { R.fail_reason = "root joint has no children"; return R; }
    int spine_start = rc[0];
    for (int c : rc) if (t.up_max[c] > t.up_max[spine_start]) spine_start = c;

    // ---- 2. legs = the OTHER root children that descend below the hips (FIX-2).
    std::vector<int> legs;
    for (int c : rc)
        if (c != spine_start && t.up_min[c] < jat(joints, t.root, AX_UP) - 1e-3f) legs.push_back(c);
    std::sort(legs.begin(), legs.end(), [&](int a, int b) { return t.lat_max[a] > t.lat_max[b]; });
    // SkinTokens sometimes inserts a one-bone pelvis connector: the root has
    // one descending child, and that child (not the root) fans into the two
    // legs.  Treating this as a malformed humanoid discarded otherwise sound
    // rigs.  Look for the *shallowest* bilateral descending fan below root;
    // that preference selects the pelvis connector before the much deeper
    // shoulder/arm fan (whose hands can also descend below the hips).
    if (legs.size() < 2) {
        std::vector<int> depth(J, -1), q{t.root}; depth[t.root] = 0;
        for (size_t qi=0; qi<q.size(); qi++)
            for (int c : t.ch[q[qi]]) { depth[c]=depth[q[qi]]+1; q.push_back(c); }
        int hub = -1;
        std::vector<int> hub_legs;
        for (int n=0; n<J; n++) {
            if (n == t.root || t.ch[n].size() < 2) continue;
            std::vector<int> cand;
            bool neg=false, pos=false;
            for (int c : t.ch[n]) {
                if (t.up_min[c] >= jat(joints, t.root, AX_UP) - 1e-3f) continue;
                cand.push_back(c);
                neg = neg || jat(joints,c,AX_LAT) < 0.f;
                pos = pos || jat(joints,c,AX_LAT) > 0.f;
            }
            if (!neg || !pos || cand.size() < 2) continue;
            std::sort(cand.begin(), cand.end(), [&](int a, int b) { return t.lat_max[a] > t.lat_max[b]; });
            if (hub < 0 || depth[n] < depth[hub]) { hub=n; hub_legs.assign(cand.begin(), cand.begin()+2); }
        }
        if (hub >= 0) {
            legs = std::move(hub_legs);
            char b[160];
            std::snprintf(b, sizeof(b), "legs found below root via pelvis connector joint %d", hub);
            R.notes.push_back(b);
        }
    }
    if (legs.size() < 2) {
        char b[160];
        std::snprintf(b, sizeof(b), "expected 2 leg chains under the root, found %d", (int)legs.size());
        R.fail_reason = b;
        return R;
    }
    if (legs.size() > 2) {
        char b[160];
        std::snprintf(b, sizeof(b), "%d descending root-children; kept the 2 with widest lateral reach",
                      (int)legs.size());
        R.notes.push_back(b);
        legs.resize(2);
    }

    // ---- 3. walk the spine up-chain; chest = first joint with lateral children on BOTH
    // sides of the midline. (Sign-agnostic: this only needs the two arms to straddle x=0.)
    std::vector<int> up_chain;
    int chest = -1, armA = -1, armB = -1;
    for (int cur = spine_start; cur >= 0;) {
        up_chain.push_back(cur);
        if (t.ch[cur].empty()) break;
        int up_kid = t.ch[cur][0];
        for (int c : t.ch[cur]) if (t.up_max[c] > t.up_max[up_kid]) up_kid = c;
        if (chest < 0) {
            int neg = -1, pos = -1;
            for (int c : t.ch[cur]) {
                if (c == up_kid) continue;
                float x = jat(joints, c, AX_LAT);
                if (x < 0 && (neg < 0 || t.lat_max[c] > t.lat_max[neg])) neg = c;
                if (x > 0 && (pos < 0 || t.lat_max[c] > t.lat_max[pos])) pos = c;
            }
            if (neg >= 0 && pos >= 0) { chest = cur; armA = neg; armB = pos; }  // armA:-X armB:+X
        }
        cur = up_kid;
    }
    if (chest < 0) { R.fail_reason = "no chest joint with arm children on both sides of the midline"; return R; }

    // ---- 4. FACING. Identify the foot joints first (sign-agnostic: legs were picked by
    // descent + lateral reach), then read the toe-ahead-of-ankle cue off them.
    auto leg_chain = [&](int s) {
        return follow(t, s, 4, [&](int b) { return -t.up_min[b]; });   // keep descending
    };
    std::vector<int> lgA = leg_chain(legs[0]), lgB = leg_chain(legs[1]);

    float fwd_sum = 0.f;
    int   fwd_votes[2] = {0, 0};
    int   nfeet = 0;
    for (const std::vector<int>* lg : {&lgA, &lgB}) {
        if (lg->size() < 4) continue;
        float d = jat(joints, (*lg)[3], AX_FWD) - jat(joints, (*lg)[2], AX_FWD);  // toe - ankle
        fwd_sum += d;
        if (d > 0) fwd_votes[0]++; else if (d < 0) fwd_votes[1]++;
        nfeet++;
    }
    float height = t.up_max[t.root] - t.up_min[t.root];
    for (int i = 0; i < J; i++) {
        height = std::max(height, jat(joints, i, AX_UP) - t.up_min[t.root]);
    }
    if (height < 1e-6f) height = 1.f;
    R.facing_margin = std::fabs(fwd_sum) / height;

    if (opt.facing_override != 0) {
        R.facing = opt.facing_override > 0 ? +1 : -1;
        R.facing_from_override = true;
    } else if (const char* e = std::getenv("RIG_BONE_FACING")) {
        if (e[0] == '-') R.facing = -1; else R.facing = +1;
        R.facing_from_override = true;
    } else {
        if (nfeet == 0) { R.fail_reason = "no 4-joint leg chain: cannot derive facing"; return R; }
        R.facing = (fwd_sum >= 0.f) ? +1 : -1;
        if (fwd_votes[0] && fwd_votes[1])
            R.notes.push_back("feet DISAGREE on facing (toe-vs-ankle sign differs L/R) — chirality unreliable");
        if (R.facing_margin < 0.03f)
            R.notes.push_back("facing margin < 3% of body height — chirality is weakly determined");
    }
    // right = up x forward = yhat x (f*zhat) = f*xhat
    R.right_sign = R.facing;

    // ---- 5. now that chirality is known, bind the arms/legs to sides.
    // armA sits at -X, armB at +X.
    int Rarm = (R.right_sign > 0) ? armB : armA;
    int Larm = (R.right_sign > 0) ? armA : armB;
    const std::vector<int>* Rleg = nullptr; const std::vector<int>* Lleg = nullptr;
    {
        float xa = jat(joints, legs[0], AX_LAT);
        bool a_is_right = (xa * (float)R.right_sign) > 0.f;
        Rleg = a_is_right ? &lgA : &lgB;
        Lleg = a_is_right ? &lgB : &lgA;
    }

    // ---- 6. spine / neck / head.
    size_t ci = 0;
    while (ci < up_chain.size() && up_chain[ci] != chest) ci++;
    std::vector<int> spine_nodes(up_chain.begin(), up_chain.begin() + (ci + 1));
    std::vector<int> neckhead(up_chain.begin() + (ci + 1),
                              up_chain.begin() + std::min(up_chain.size(), ci + 3));

    slot_of[0] = t.root;                                  // pelvis
    {   // assign spine3 at the chest and walk down: spine3, spine2, spine1
        const int sp_slots[3] = {9, 6, 3};
        int k = 0;
        for (int i = (int)spine_nodes.size() - 1; i >= 0 && k < 3; i--, k++)
            slot_of[sp_slots[k]] = spine_nodes[i];
    }
    slot_of[9] = chest;
    if (neckhead.size() > 0) slot_of[12] = neckhead[0];
    if (neckhead.size() > 1) slot_of[15] = neckhead[1];

    // ---- 7. arm + leg chains -> slots.
    auto arm_chain = [&](int s) { return follow(t, s, 4, [&](int b) { return t.lat_max[b]; }); };
    std::vector<int> La = arm_chain(Larm), Ra = arm_chain(Rarm);
    const int La_slots[4] = {13, 16, 18, 20};
    const int Ra_slots[4] = {14, 17, 19, 21};
    for (size_t i = 0; i < La.size() && i < 4; i++) slot_of[La_slots[i]] = La[i];
    for (size_t i = 0; i < Ra.size() && i < 4; i++) slot_of[Ra_slots[i]] = Ra[i];
    const int Ll_slots[4] = {1, 4, 7, 10};
    const int Rl_slots[4] = {2, 5, 8, 11};
    for (size_t i = 0; i < Lleg->size() && i < 4; i++) slot_of[Ll_slots[i]] = (*Lleg)[i];
    for (size_t i = 0; i < Rleg->size() && i < 4; i++) slot_of[Rl_slots[i]] = (*Rleg)[i];

    // ---- 7b. second, independent chirality cue: head forward of neck.
    // Computed here (not at step 4) because it needs the neck/head slots. It never feeds
    // the decision -- falsify_bone_names() cross-checks the final facing against it, so a
    // wrong facing (auto OR overridden) has something that can contradict it.
    if (slot_of[12] >= 0 && slot_of[15] >= 0) {
        float d = jat(joints, slot_of[15], AX_FWD) - jat(joints, slot_of[12], AX_FWD);
        R.cue_head_margin = std::fabs(d) / height;
        R.cue_head_sign = (d > 0) ? +1 : (d < 0 ? -1 : 0);
    }

    // ---- 8. emit core names.
    const char** tbl = (opt.style == NameStyle::Mixamo) ? kMixamoNames : kSmplNames;
    for (int s = 0; s < SMPL_N; s++) {
        if (slot_of[s] < 0) continue;
        const int joint = slot_of[s];
        // A topology walker may visit the same joint through two short or
        // malformed chains. Counting both semantic slots made a 14-joint
        // skeleton appear to satisfy a 20/22 gate, while the later assignment
        // silently overwrote its first name. A Mixamo core requires one
        // distinct transform per slot; preserve the first label only and make
        // the collision an explicit structural deficit.
        if (R.smpl[joint] >= 0) {
            R.notes.push_back(bn_fmt("semantic slot collision: joint %d requested for %s and %s",
                                     joint, tbl[R.smpl[joint]], tbl[s]));
            continue;
        }
        R.names[joint] = tbl[s];
        R.smpl[joint] = s;
        R.named_core++;
    }

    // ---- 9. fingers (Mixamo only; SMPL-H's hand joints are a different 30-joint scheme
    // we do not attempt to fill, so SmplH style leaves fingers as Extra).
    if (opt.name_fingers && opt.style == NameStyle::Mixamo) {
        static const char* kFing[5] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
        for (int side = 0; side < 2; side++) {
            int hand = slot_of[side == 0 ? 20 : 21];
            if (hand < 0 || t.ch[hand].empty()) continue;
            const char* sname = (side == 0) ? "Left" : "Right";

            std::vector<std::vector<int>> br;
            for (int c : t.ch[hand]) br.push_back(linear_branch(t, c));
            if (br.size() < 2) continue;
            if (br.size() > 5) {
                std::sort(br.begin(), br.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
                    return a.size() > b.size();
                });
                br.resize(5);
            }
            const int NB = (int)br.size();

            // tip direction of each branch, from the hand joint
            std::vector<std::array<float,3>> dir(NB);
            for (int i = 0; i < NB; i++) {
                int tip = br[i].back();
                float v[3] = {jat(joints,tip,0)-jat(joints,hand,0),
                              jat(joints,tip,1)-jat(joints,hand,1),
                              jat(joints,tip,2)-jat(joints,hand,2)};
                norm3(v);
                dir[i] = {v[0], v[1], v[2]};
            }
            // thumb = branch least aligned with the others (upstream's rule)
            int thumb = 0;
            if (NB >= 3) {
                float worst = 1e30f;
                for (int i = 0; i < NB; i++) {
                    float s = 0;
                    for (int k = 0; k < NB; k++) {
                        if (k == i) continue;
                        s += dir[i][0]*dir[k][0] + dir[i][1]*dir[k][1] + dir[i][2]*dir[k][2];
                    }
                    s /= (float)std::max(NB - 1, 1);
                    if (s < worst) { worst = s; thumb = i; }
                }
            }
            // order the remaining branches along the palm's lateral axis
            float fwd[3] = {0,0,0};
            for (int i = 0; i < NB; i++) { if (i==thumb) continue;
                fwd[0]+=dir[i][0]; fwd[1]+=dir[i][1]; fwd[2]+=dir[i][2]; }
            norm3(fwd);
            // lateral axis = widest-spread direction of the finger bases, orthogonal to fwd.
            // Use the character's own UP as the seed, projected off fwd (fingers fan across
            // the palm, which for a humanoid hand is roughly the up/forward plane's normal).
            float lat[3] = {0,0,0};
            {
                float seed[3] = {0, 0, 0};
                seed[AX_FWD] = (float)R.facing;   // across-palm ~ body-forward for a hanging hand
                float d = seed[0]*fwd[0] + seed[1]*fwd[1] + seed[2]*fwd[2];
                lat[0] = seed[0] - d*fwd[0]; lat[1] = seed[1] - d*fwd[1]; lat[2] = seed[2] - d*fwd[2];
                norm3(lat);
                if (std::fabs(lat[0])+std::fabs(lat[1])+std::fabs(lat[2]) < 1e-6f) { lat[AX_UP] = 1.f; }
            }
            std::vector<std::pair<float,int>> ord;
            for (int i = 0; i < NB; i++) {
                if (i == thumb) continue;
                int base = br[i][0];
                float b[3] = {jat(joints,base,0)-jat(joints,hand,0),
                              jat(joints,base,1)-jat(joints,hand,1),
                              jat(joints,base,2)-jat(joints,hand,2)};
                float d = b[0]*fwd[0]+b[1]*fwd[1]+b[2]*fwd[2];
                float p[3] = {b[0]-d*fwd[0], b[1]-d*fwd[1], b[2]-d*fwd[2]};
                ord.push_back({p[0]*lat[0]+p[1]*lat[1]+p[2]*lat[2], i});
            }
            std::sort(ord.begin(), ord.end());
            // thumb side decides which end is the index finger
            {
                int base = br[thumb][0];
                float b[3] = {jat(joints,base,0)-jat(joints,hand,0),
                              jat(joints,base,1)-jat(joints,hand,1),
                              jat(joints,base,2)-jat(joints,hand,2)};
                float d = b[0]*fwd[0]+b[1]*fwd[1]+b[2]*fwd[2];
                float p[3] = {b[0]-d*fwd[0], b[1]-d*fwd[1], b[2]-d*fwd[2]};
                float ts = p[0]*lat[0]+p[1]*lat[1]+p[2]*lat[2];
                float mean = 0; for (auto& o : ord) mean += o.first;
                if (!ord.empty()) mean /= (float)ord.size();
                if (ts >= mean) std::reverse(ord.begin(), ord.end());
            }
            // emit: thumb + up to 4 others as Index/Middle/Ring/Pinky, 3 segments each
            auto emit = [&](const std::vector<int>& branch, const char* fname) {
                for (size_t k = 0; k < branch.size() && k < 3; k++) {
                    char b[96];
                    std::snprintf(b, sizeof(b), "mixamorig:%s%s%d", sname, fname, (int)k + 1);
                    R.names[branch[k]] = b;
                    R.named_fingers++;
                }
            };
            emit(br[thumb], kFing[0]);
            for (size_t k = 0; k < ord.size() && k < 4; k++) emit(br[ord[k].second], kFing[k + 1]);
        }
    }

    // ---- 10. Anything left over is retained under an explicit non-semantic namespace. It is
    // crucial not to put these in the `mixamorig:` namespace: they may be wings, hair, props, or a
    // malformed branch, and only the validated core is a Mixamo contract. Source-index names remain
    // stable across a later semantic appendage mapper.
    for (int i = 0; i < J; i++) {
        if (!R.names[i].empty()) continue;
        char b[64];
        std::snprintf(b, sizeof(b), "skintokens:Extra_%03d", i);
        R.n_extra++;
        R.names[i] = b;
    }
    R.ok = true;
    return R;
}

// SkinTokens sometimes emits a semantically complete torso with only two
// spine segments: Hips -> Spine1 -> Spine2. A direct Mixamo hand-off needs
// the intervening `Spine` node too. Insert one *only* when the naming pass has
// unambiguously identified short-torso topology (the exact 21/22 form, or
// the exact 20/22 form whose only other missing slot is Head). The new joint sits
// halfway from Hips to the existing Spine1 and becomes its parent; its skin
// column is zero because it is a transform-only intermediary. Child
// inverse-bind matrices retain the rest pose, so animation of this node still
// propagates through the hierarchy.
inline bool synthesize_missing_mixamo_spine(std::vector<float>& joints,
                                             std::vector<int>& parents,
                                             std::vector<float>& skin_weights,
                                             int64_t skin_rows,
                                             std::string* detail = nullptr) {
    const int old_j = (int)(joints.size() / 3);
    if (old_j <= 0 || (int)parents.size() != old_j || skin_rows < 0 ||
        skin_weights.size() != (size_t)skin_rows * (size_t)old_j) return false;
    BoneNaming before = name_bones(joints, parents);
    if (!before.ok || (before.named_core != 21 && before.named_core != 20)) return false;
    int hips = -1, spine = -1, spine1 = -1, spine2 = -1;
    for (int i = 0; i < old_j; i++) {
        if (before.smpl[i] == 0) hips = i;
        else if (before.smpl[i] == 3) spine = i;
        else if (before.smpl[i] == 6) spine1 = i;
        else if (before.smpl[i] == 9) spine2 = i;
    }
    // Do not invent anatomy for a general partial skeleton. This narrowly
    // handles the observed short-torso form, with an already validated Hips /
    // Spine1 / Spine2 chain and no existing Spine assignment.  A missing Head
    // may coexist and is handled separately by the geometry-supported repair
    // below; hands, toes, limbs, or any other missing semantic node reject.
    for (int s = 0; s < SMPL_N; s++) {
        bool found = false;
        for (int i = 0; i < old_j; i++) if (before.smpl[i] == s) { found = true; break; }
        if (!found && s != 3 && s != 15) return false;
    }
    if (spine >= 0 || hips < 0 || spine1 < 0 || spine2 < 0 || parents[spine1] != hips)
        return false;

    const int inserted = old_j;
    for (int d = 0; d < 3; d++)
        joints.push_back(0.5f * (joints[(size_t)hips * 3 + d] + joints[(size_t)spine1 * 3 + d]));
    parents.push_back(hips);
    parents[spine1] = inserted;
    std::vector<float> expanded((size_t)skin_rows * (size_t)(old_j + 1), 0.f);
    for (int64_t r = 0; r < skin_rows; r++) {
        std::memcpy(&expanded[(size_t)r * (old_j + 1)],
                    &skin_weights[(size_t)r * old_j], (size_t)old_j * sizeof(float));
    }
    skin_weights.swap(expanded);
    BoneNaming after = name_bones(joints, parents);
    if (!after.ok || after.named_core != before.named_core + 1) return false;
    if (detail) *detail = "inserted transform-only Mixamo Spine between Hips and Spine1";
    return true;
}

// A small number of otherwise complete SkinTokens rigs end at Neck: they have
// the complete bilateral arm/leg topology and a short torso, but omit both the
// structural `Spine` and the terminal `Head` joint.  Unlike a generic
// name-filling fallback, this repair is allowed only for that exact 20/22
// semantic skeleton.  The new Head is anchored in the actual upper sampled
// geometry and takes skin influence only from Neck in that upper-head band, so
// a Head animation moves head geometry rather than merely adding a dead node.
//
// It deliberately does NOT repair missing hands, toes, limbs, or an ambiguous
// torso. Those cases remain rejected: inventing several endpoint bones would
// make a name check pass without a defensible skinning contract.
inline bool synthesize_missing_mixamo_head(std::vector<float>& joints,
                                           std::vector<int>& parents,
                                           std::vector<float>& skin_weights,
                                           const std::vector<float>& sampled_points,
                                           int64_t skin_rows,
                                           std::string* detail = nullptr) {
    const int old_j = (int)(joints.size() / 3);
    if (old_j <= 0 || (int)parents.size() != old_j || skin_rows <= 0 ||
        sampled_points.size() != (size_t)skin_rows * 3 ||
        skin_weights.size() != (size_t)skin_rows * (size_t)old_j) return false;
    BoneNaming before = name_bones(joints, parents);
    if (!before.ok || before.named_core != 21) return false;

    int neck = -1, hips = -1;
    bool has_head = false;
    for (int i = 0; i < old_j; i++) {
        if (before.smpl[i] == 0) hips = i;
        if (before.smpl[i] == 12) neck = i;
        if (before.smpl[i] == 15) has_head = true;
    }
    if (hips < 0 || neck < 0 || has_head) return false;

    // This is intentionally exact: after the validated Spine insertion, the
    // only absent semantic endpoint must be Head.  Do not promote a skeleton
    // that happens to have 21 names but is missing a hand/toe instead.
    bool missing_only_head = true;
    for (int s = 0; s < SMPL_N; s++) {
        bool found = false;
        for (int i = 0; i < old_j; i++) if (before.smpl[i] == s) { found = true; break; }
        if (!found && s != 15) { missing_only_head = false; break; }
    }
    if (!missing_only_head) return false;

    float ymin = sampled_points[1], ymax = sampled_points[1];
    for (int64_t r = 0; r < skin_rows; r++) {
        float y = sampled_points[(size_t)r * 3 + 1];
        ymin = std::min(ymin, y); ymax = std::max(ymax, y);
    }
    const float height = ymax - ymin;
    const float neck_y = joints[(size_t)neck * 3 + 1];
    if (height < 1e-5f || ymax - neck_y < 0.14f * height) return false;

    // Estimate the rest Head location from real points in the upper 22% of
    // the character. This is more defensible than extrapolating a torso vector
    // through a stylised head/hair silhouette.
    const float head_band = ymax - 0.22f * height;
    double sx = 0, sy = 0; int64_t nhead = 0;
    for (int64_t r = 0; r < skin_rows; r++) {
        const float* p = &sampled_points[(size_t)r * 3];
        if (p[1] < head_band) continue;
        sx += p[0]; sy += p[1]; ++nhead;
    }
    if (nhead < std::max<int64_t>(8, skin_rows / 200)) return false;

    const int inserted = old_j;
    joints.push_back((float)(sx / nhead));
    joints.push_back((float)(sy / nhead));
    // A generated endpoint cannot be used as an independent front/back
    // chirality witness. Keep its depth aligned with Neck, while retaining the
    // sampled upper-head x/y placement and head-region skin mass. The
    // falsifier will report this cue as unavailable rather than treating a
    // stylised hair/back centroid as evidence against the toe-derived facing.
    joints.push_back(joints[(size_t)neck * 3 + 2]);
    parents.push_back(neck);

    std::vector<float> expanded((size_t)skin_rows * (size_t)(old_j + 1), 0.f);
    const float skin_begin = neck_y + 0.08f * height;
    const float skin_end = neck_y + 0.22f * height;
    double moved_mass = 0.0;
    int64_t moved_rows = 0;
    for (int64_t r = 0; r < skin_rows; r++) {
        const float* in = &skin_weights[(size_t)r * old_j];
        float* out = &expanded[(size_t)r * (old_j + 1)];
        std::memcpy(out, in, (size_t)old_j * sizeof(float));
        const float y = sampled_points[(size_t)r * 3 + 1];
        const float t = std::max(0.f, std::min(0.85f, (y - skin_begin) / std::max(1e-6f, skin_end - skin_begin) * 0.85f));
        const float moved = out[neck] * t;
        out[neck] -= moved;
        out[inserted] = moved;
        if (moved > 1e-5f) { moved_mass += moved; ++moved_rows; }
    }
    // A Head node without any actual skin would be exactly the misleading
    // repair this gate is meant to prevent.
    if (moved_rows < std::max<int64_t>(8, skin_rows / 200) || moved_mass < 0.5) return false;
    skin_weights.swap(expanded);
    BoneNaming after = name_bones(joints, parents);
    if (!after.ok || after.named_core != 22) return false;
    if (detail) {
        char b[256];
        std::snprintf(b, sizeof(b), "inserted geometry-supported Mixamo Head under Neck; moved %.3f Neck skin mass across %lld upper-head samples",
                      moved_mass, (long long)moved_rows);
        *detail = b;
    }
    return true;
}

// An otherwise complete humanoid can omit exactly one terminal ToeBase.  This
// is not a general limb repair: it is permitted only when every other Mixamo
// slot, both feet, and the opposite toe are already structurally identified.
// The inserted toe mirrors the opposite foot-to-toe rest vector and receives
// real skin mass only from points that are closer to the inferred toe than to
// its parent foot.  The original one-toe facing cue is preserved by the
// caller's normal naming pass; this helper never invents a torso or limb.
inline bool synthesize_missing_mixamo_toe(std::vector<float>& joints,
                                          std::vector<int>& parents,
                                          std::vector<float>& skin_weights,
                                          const std::vector<float>& sampled_points,
                                          int64_t skin_rows,
                                          std::string* detail = nullptr) {
    const int old_j = (int)(joints.size() / 3);
    if (old_j <= 0 || (int)parents.size() != old_j || skin_rows <= 0 ||
        sampled_points.size() != (size_t)skin_rows * 3 ||
        skin_weights.size() != (size_t)skin_rows * (size_t)old_j) return false;
    BoneNaming before = name_bones(joints, parents);
    if (!before.ok || before.named_core != 21) return false;
    int slot[SMPL_N]; std::fill(slot, slot + SMPL_N, -1);
    for (int j = 0; j < old_j; ++j) if (before.smpl[j] >= 0) slot[before.smpl[j]] = j;
    int missing = -1;
    for (int s = 0; s < SMPL_N; ++s) if (slot[s] < 0) { if (missing >= 0) return false; missing = s; }
    if (missing != 10 && missing != 11) return false;
    const int other_toe_slot = missing == 10 ? 11 : 10;
    const int own_foot_slot = missing == 10 ? 7 : 8;
    const int other_foot_slot = missing == 10 ? 8 : 7;
    if (slot[own_foot_slot] < 0 || slot[other_foot_slot] < 0 || slot[other_toe_slot] < 0 ||
        parents[slot[other_toe_slot]] != slot[other_foot_slot]) return false;
    const int own_foot = slot[own_foot_slot], other_foot = slot[other_foot_slot], other_toe = slot[other_toe_slot];
    float d[3];
    for (int k=0;k<3;k++) d[k] = joints[(size_t)other_toe*3+k] - joints[(size_t)other_foot*3+k];
    d[0] = -d[0]; // bilateral reflection around the skeleton centreline
    const float dlen = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    if (!(dlen > 1e-4f)) return false;
    const int inserted = old_j;
    for (int k=0;k<3;k++) joints.push_back(joints[(size_t)own_foot*3+k] + d[k]);
    parents.push_back(own_foot);
    std::vector<float> expanded((size_t)skin_rows*(size_t)(old_j+1),0.f);
    double moved_mass=0.; int64_t moved_rows=0;
    for (int64_t r=0;r<skin_rows;r++) {
        const float* in=&skin_weights[(size_t)r*old_j]; float* out=&expanded[(size_t)r*(old_j+1)];
        std::memcpy(out,in,(size_t)old_j*sizeof(float));
        const float* p=&sampled_points[(size_t)r*3];
        float df=0, dt=0;
        for (int k=0;k<3;k++) { float a=p[k]-joints[(size_t)own_foot*3+k], b=p[k]-joints[(size_t)inserted*3+k]; df+=a*a; dt+=b*b; }
        const float share = std::max(0.f,std::min(.80f,(df-dt)/(df+dt+1e-6f)*.80f));
        const float moved=out[own_foot]*share; out[own_foot]-=moved; out[inserted]=moved;
        if (moved>1e-5f) { moved_mass+=moved; ++moved_rows; }
    }
    if (moved_rows < std::max<int64_t>(8,skin_rows/500) || moved_mass < .10) return false;
    skin_weights.swap(expanded);
    BoneNaming after=name_bones(joints,parents);
    if (!after.ok || after.named_core!=22) return false;
    if (detail) {
        char b[256]; std::snprintf(b,sizeof(b),"inserted skin-supported Mixamo %s under %s; moved %.3f Foot skin mass across %lld samples",
            kMixamoNames[missing],kMixamoNames[own_foot_slot],moved_mass,(long long)moved_rows); *detail=b;
    }
    return true;
}

// Exact bilateral counterpart recovery for the observed Soldier form: the
// complete left Shoulder->Arm->ForeArm->Hand chain exists, the right Shoulder
// exists, and precisely the three right-chain slots are absent.  Rest joints
// are reflected around the named hip centre. Skin support is copied from the
// nearest mirrored sampled point, never fabricated from a label or Python.
inline bool synthesize_missing_mixamo_right_arm(std::vector<float>& joints,
                                                std::vector<int>& parents,
                                                std::vector<float>& skin_weights,
                                                const std::vector<float>& sampled_points,
                                                int64_t skin_rows,
                                                std::string* detail=nullptr) {
    const int old_j=(int)(joints.size()/3);
    if(old_j<=0 || (int)parents.size()!=old_j || skin_rows<=0 || sampled_points.size()!=(size_t)skin_rows*3 ||
       skin_weights.size()!=(size_t)skin_rows*(size_t)old_j) return false;
    BoneNaming before=name_bones(joints,parents); if(!before.ok || before.named_core!=19) return false;
    int slot[SMPL_N]; std::fill(slot,slot+SMPL_N,-1);
    for(int j=0;j<old_j;j++) if(before.smpl[j]>=0) slot[before.smpl[j]]=j;
    for(int s=0;s<SMPL_N;s++) if(slot[s]<0 && s!=17 && s!=19 && s!=21) return false;
    if(slot[17]>=0||slot[19]>=0||slot[21]>=0 || slot[1]<0||slot[2]<0||slot[13]<0||slot[14]<0||
       slot[16]<0||slot[18]<0||slot[20]<0 || parents[slot[16]]!=slot[13] ||
       parents[slot[18]]!=slot[16] || parents[slot[20]]!=slot[18]) return false;
    const float cx=.5f*(joints[(size_t)slot[1]*3]+joints[(size_t)slot[2]*3]);
    const int rarm=old_j, rfore=old_j+1, rhand=old_j+2;
    for(int src: {slot[16],slot[18],slot[20]}) {
        joints.push_back(2.f*cx-joints[(size_t)src*3]);
        joints.push_back(joints[(size_t)src*3+1]); joints.push_back(joints[(size_t)src*3+2]);
    }
    parents.push_back(slot[14]); parents.push_back(rarm); parents.push_back(rfore);
    std::vector<float> out((size_t)skin_rows*(size_t)(old_j+3),0.f);
    double moved=0.; int64_t rows=0;
    const float right_sign = joints[(size_t)slot[14]*3] >= cx ? 1.f : -1.f;
    for(int64_t r=0;r<skin_rows;r++) {
        const float* p=&sampled_points[(size_t)r*3]; float* dst=&out[(size_t)r*(old_j+3)];
        const float* srcrow=&skin_weights[(size_t)r*old_j]; std::memcpy(dst,srcrow,(size_t)old_j*sizeof(float));
        if ((p[0]-cx)*right_sign <= .02f) continue;
        float best=1e30f; int64_t qbest=-1;
        const float mx=2.f*cx-p[0];
        for(int64_t q=0;q<skin_rows;q++) { const float* qv=&sampled_points[(size_t)q*3];
            float dx=qv[0]-mx,dy=qv[1]-p[1],dz=qv[2]-p[2],d=dx*dx+dy*dy+dz*dz;
            if(d<best){best=d;qbest=q;}
        }
        const float* mirror=&skin_weights[(size_t)qbest*old_j];
        float a=mirror[slot[16]], b=mirror[slot[18]], c=mirror[slot[20]], mass=a+b+c;
        if(mass<.03f) continue;
        mass=std::min(.85f,mass); for(int j=0;j<old_j;j++) dst[j]*=(1.f-mass);
        float norm=std::max(1e-6f,a+b+c); dst[rarm]=mass*a/norm; dst[rfore]=mass*b/norm; dst[rhand]=mass*c/norm;
        moved+=mass; ++rows;
    }
    if(rows<std::max<int64_t>(32,skin_rows/100) || moved<2.) return false;
    skin_weights.swap(out); BoneNaming after=name_bones(joints,parents);
    if(!after.ok||after.named_core!=22) return false;
    if(detail){char b[256];std::snprintf(b,sizeof(b),"inserted mirrored skin-supported Mixamo RightArm/RightForeArm/RightHand; moved %.3f skin mass across %lld mirrored samples",moved,(long long)rows);*detail=b;}
    return true;
}

// Correct only a complete named core whose generated head sits at/below Neck
// and whose right collar is a clear non-mirrored outlier.  The head anchor is
// re-estimated from actual upper-body samples; its skin support is retained
// and supplemented from Neck in that same band. This is deliberately not a
// partial-skeleton repair.
inline bool normalize_malformed_mixamo_head_and_collar(std::vector<float>& joints,
                                                        std::vector<int>& parents,
                                                        std::vector<float>& w,
                                                        const std::vector<float>& pts,
                                                        int64_t rows,
                                                        std::string* detail=nullptr) {
    const int J=(int)(joints.size()/3); if(J<=0||(int)parents.size()!=J||pts.size()!=(size_t)rows*3||w.size()!=(size_t)rows*(size_t)J) return false;
    // The mapper can deliberately report an anatomy falsifier failure for
    // this exact malformed-head case.  We need the complete semantic mapping
    // in order to correct it; requiring `ok` here would make that correction
    // unreachable, while the post-repair falsifier remains mandatory.
    BoneNaming n=name_bones(joints,parents); if(n.named_core!=22) return false;
    int s[SMPL_N];std::fill(s,s+SMPL_N,-1);for(int j=0;j<J;j++)if(n.smpl[j]>=0)s[n.smpl[j]]=j;
    if(s[1]<0||s[2]<0||s[12]<0||s[13]<0||s[14]<0||s[15]<0)return false;
    float ymin=pts[1],ymax=pts[1];for(int64_t r=0;r<rows;r++){float y=pts[(size_t)r*3+1];ymin=std::min(ymin,y);ymax=std::max(ymax,y);}float h=ymax-ymin;
    bool bad_head=joints[(size_t)s[15]*3+1] <= joints[(size_t)s[12]*3+1]+.03f*h;
    float cx=.5f*(joints[(size_t)s[1]*3]+joints[(size_t)s[2]*3]);
    float dx=joints[(size_t)s[14]*3]-(2*cx-joints[(size_t)s[13]*3]);
    float dy=joints[(size_t)s[14]*3+1]-joints[(size_t)s[13]*3+1];
    float dz=joints[(size_t)s[14]*3+2]-joints[(size_t)s[13]*3+2];
    bool bad_collar=std::sqrt(dx*dx+dy*dy+dz*dz)>.08f*h;
    if(!bad_head&&!bad_collar)return false;
    if(bad_collar){joints[(size_t)s[14]*3]=2*cx-joints[(size_t)s[13]*3];joints[(size_t)s[14]*3+1]=joints[(size_t)s[13]*3+1];joints[(size_t)s[14]*3+2]=joints[(size_t)s[13]*3+2];}
    int64_t count=0;double sx=0,sy=0;float band=ymax-.22f*h;
    for(int64_t r=0;r<rows;r++){const float* p=&pts[(size_t)r*3];if(p[1]<band)continue;sx+=p[0];sy+=p[1];count++;float take=.55f*w[(size_t)r*J+s[12]];w[(size_t)r*J+s[12]]-=take;w[(size_t)r*J+s[15]]+=take;}
    // A malformed decoded Head can have no learned Head/Neck mass at all
    // (its rows were assigned to the stray low branch). Geometry still gives
    // us an unambiguous rest anchor; preserve the existing native skin field
    // in that case and let the written GLB pose gate decide publication.
    if(bad_head && count<std::max<int64_t>(8,rows/200)) return false;
    if(bad_head){joints[(size_t)s[15]*3]=(float)(sx/count);joints[(size_t)s[15]*3+1]=(float)(sy/count);joints[(size_t)s[15]*3+2]=joints[(size_t)s[12]*3+2];parents[s[15]]=s[12];}
    // The caller immediately remaps and runs the complete hard falsifier on
    // the edited skeleton. Do not suppress the repair merely because this
    // intermediate mapper still carries the pre-repair anatomy verdict.
    BoneNaming after=name_bones(joints,parents); (void)after;
    if(detail){char b[256];std::snprintf(b,sizeof(b),"normalized malformed %s%s from native sampled geometry",bad_head?"Head":"",bad_collar?(bad_head?" and right collar":"right collar"):"");*detail=b;}return true;
}

// -----------------------------------------------------------------------------------------
// FALSIFIER — tests the naming against geometry the naming did NOT use, and FAILS.
//
// Deliberately NOT "did we name 22/22" — that passes just as happily on a mirrored or
// scrambled skeleton. Each check below can actually come out false on a wrong answer.
//
// Returns the number of FAILED checks (0 == clean). Prints the evidence either way.
// -----------------------------------------------------------------------------------------
inline int falsify_bone_names(const std::vector<float>& joints,
                              const std::vector<int>&   parents,
                              const BoneNaming&         R,
                              bool                      print = true) {
    using namespace bn_detail;
    const int J = (int)parents.size();
    int fails = 0;
    auto slot = [&](int s) { for (int i = 0; i < J; i++) if (R.smpl[i] == s) return i; return -1; };
    auto X = [&](int i) { return jat(joints, i, AX_LAT); };
    auto Y = [&](int i) { return jat(joints, i, AX_UP); };
    auto Z = [&](int i) { return jat(joints, i, AX_FWD); };
    auto dist = [&](int a, int b) {
        float dx=X(a)-X(b), dy=Y(a)-Y(b), dz=Z(a)-Z(b);
        return std::sqrt(dx*dx+dy*dy+dz*dz);
    };
    // Two categories, deliberately separated:
    //   CHECK — NAMING CORRECTNESS. "is this label on the right joint?" A wrong answer
    //           makes these false. These are hard FAILs and set the exit code.
    //   ANAT  — ANATOMY PLAUSIBILITY. "is this character human-proportioned?" That is a
    //           property of the CHARACTER, not of the naming: a stylised character can be
    //           named perfectly and still be non-human (gilly's shin is 1.7x its thigh).
    //           So these WARN and print their numbers, but do not fail the naming.
    //           They still matter — they predict how well a human-proportioned AMASS/Mixamo
    //           clip will retarget onto this rig.
    int warns = 0;
    auto CHECK = [&](bool pass, const char* tag, const std::string& msg) {
        if (!pass) fails++;
        if (print) std::printf("  [%s] %-26s %s\n", pass ? "PASS" : "FAIL", tag, msg.c_str());
    };
    auto ANAT = [&](bool pass, const char* tag, const std::string& msg) {
        if (!pass) warns++;
        if (print) std::printf("  [%s] %-26s %s\n", pass ? "ok  " : "WARN", tag, msg.c_str());
    };

    if (print) {
        std::printf("\n=== bone-naming falsifier ===\n");
        std::printf("  facing=%+dZ (%s)  char-right=%+dX  facing_margin=%.1f%% of height\n",
                    R.facing, R.facing_from_override ? "OVERRIDE" : "derived from toe-vs-ankle",
                    R.right_sign, R.facing_margin * 100.f);
        std::printf("  named core=%d/22  fingers=%d  extra=%d  of J=%d\n",
                    R.named_core, R.named_fingers, R.n_extra, J);
    }
    if (!R.ok) {
        if (print) std::printf("  [FAIL] topology                  %s\n", R.fail_reason.c_str());
        return fails + 1;
    }

    // body scale for tolerances
    float ymin = 1e30f, ymax = -1e30f;
    for (int i = 0; i < J; i++) { ymin = std::min(ymin, Y(i)); ymax = std::max(ymax, Y(i)); }
    const float H = std::max(ymax - ymin, 1e-6f);

    // --- C0: CHIRALITY. The one check that guards LEFT-vs-RIGHT.
    //
    // Read this before trusting the others: every L/R check below (C3/C4) compares against
    // R.right_sign, which the NAMING also used -- so they are CIRCULAR w.r.t. a mirror.
    // Flip the facing and the labels and the expectations flip together: C3/C4 still pass.
    // (Verified: forcing gilly to the wrong facing passes C1-C8 with 22/22 named.)
    // So the ONLY thing standing between us and a silently mirrored rig is this check:
    // does the final facing agree with a cue the facing decision did NOT come from?
    //
    // Calibration: FAIL only on ACTIVE CONTRADICTION (both cues have real magnitude and
    // disagree). Absence of corroboration is NOT evidence of error, so it WARNs instead --
    // a check that fires on silence is a check that gets ignored. The head cue's real
    // magnitude is ~1.2-1.6% of height (measured, n=2); below ~1% it is noise, so we
    // refuse to draw any conclusion from it rather than pretend it corroborates.
    {
        const float kHeadCueFloor = 0.01f;   // 1% of body height
        if (R.cue_head_sign == 0 || R.cue_head_margin < kHeadCueFloor) {
            ANAT(false, "chirality-cross-check",
                 bn_fmt("UNCORROBORATED: head cue %.2f%% < %.0f%% floor -> facing rests on the toe "
                        "cue ALONE (%.1f%%). A MIRROR HERE WOULD BE UNDETECTABLE.",
                        R.cue_head_margin * 100.f, kHeadCueFloor * 100.f, R.facing_margin * 100.f));
        } else {
            CHECK(R.cue_head_sign == R.facing, "chirality-cross-check",
                  bn_fmt("toe cue %+dZ (%.1f%%) vs independent head-vs-neck cue %+dZ (%.2f%%)%s",
                         R.facing, R.facing_margin * 100.f, R.cue_head_sign, R.cue_head_margin * 100.f,
                         R.facing_from_override ? " [facing was OVERRIDDEN]" : ""));
        }
    }

    // --- C1: the Head's subtree contains the highest joint in the skeleton.
    // Never used by the naming (which only compares ROOT-CHILD subtree maxima).
    {
        int h = slot(15);
        if (h < 0) { CHECK(false, "head-present", "no Head named"); }
        else {
            // max Y over head's subtree
            std::vector<std::vector<int>> ch(J);
            for (int i = 0; i < J; i++) if (parents[i] >= 0) ch[parents[i]].push_back(i);
            float hmax = -1e30f; std::vector<int> st{h};
            while (!st.empty()) { int x = st.back(); st.pop_back();
                hmax = std::max(hmax, Y(x)); for (int c : ch[x]) st.push_back(c); }
            CHECK(hmax >= ymax - 1e-4f, "head-is-highest",
                  bn_fmt("head subtree maxY=%+.4f vs skeleton maxY=%+.4f", hmax, ymax));
        }
    }

    // --- C2: Hips -> Spine.. -> Neck -> Head is monotonically INCREASING in height.
    // Naming only ever compared subtree extents, never the chain's own monotonicity.
    {
        const int chain[] = {0, 3, 6, 9, 12, 15};
        float prev = -1e30f; bool mono = true; std::string trail;
        for (int s : chain) {
            int i = slot(s); if (i < 0) continue;
            char b[64]; std::snprintf(b, sizeof(b), "%s=%+.3f ", kSmplNames[s], Y(i));
            trail += b;
            if (Y(i) < prev - 1e-4f) mono = false;
            prev = Y(i);
        }
        CHECK(mono, "spine-monotonic-up", trail);
    }

    // --- C3: every L/R pair straddles the midline with the correct sign.
    // THE mirror check. The chain-follower ranks children by |x| (sign-blind), so a
    // joint's x SIGN this far down the chain is not something the naming constrained.
    {
        int bad = 0; std::string ev;
        for (int p = 0; p < LR_PAIR_N; p++) {
            int li = slot(kLRPairs[p][0]), ri = slot(kLRPairs[p][1]);
            if (li < 0 || ri < 0) continue;
            // char-left is the side opposite right_sign
            bool lok = (X(li) * (float)R.right_sign) <= 0.f;
            bool rok = (X(ri) * (float)R.right_sign) >= 0.f;
            if (!lok || !rok) {
                bad++;
                char b[160];
                std::snprintf(b, sizeof(b), "%s.x=%+.3f/%s.x=%+.3f ",
                              kSmplNames[kLRPairs[p][0]], X(li), kSmplNames[kLRPairs[p][1]], X(ri));
                ev += b;
            }
        }
        CHECK(bad == 0, "L/R-on-correct-sides",
              bn_fmt("%d pair(s) on the wrong side of x=0 %s", bad, ev.c_str()));
    }

    // --- C4: bilateral symmetry. Each L/R pair should mirror about x=0.
    // The two sides are named by INDEPENDENT chain walks, so agreement is real evidence.
    {
        float worst = 0; const char* wname = "-"; int bad = 0;
        for (int p = 0; p < LR_PAIR_N; p++) {
            int li = slot(kLRPairs[p][0]), ri = slot(kLRPairs[p][1]);
            if (li < 0 || ri < 0) continue;
            float ex = std::fabs(X(li) + X(ri)) / H;          // mirrored x should cancel
            float ey = std::fabs(Y(li) - Y(ri)) / H;          // same height
            float e = std::max(ex, ey);
            if (e > worst) { worst = e; wname = kSmplNames[kLRPairs[p][0]]; }
            if (e > 0.08f) bad++;
        }
        CHECK(bad == 0, "bilateral-symmetry",
              bn_fmt("worst pair %s off by %.1f%% of height (tol 8%%)", wname, worst * 100.f));
    }

    // --- C5: the feet are the lowest joints in the skeleton.
    // Legs were picked by "descends below the hips" — never by being the global minimum.
    {
        int lf = slot(10), rf = slot(11), la = slot(7), ra = slot(8);
        float foot_lo = 1e30f;
        for (int i : {lf, rf, la, ra}) if (i >= 0) foot_lo = std::min(foot_lo, Y(i));
        CHECK(foot_lo <= ymin + 0.02f * H, "feet-are-lowest",
              bn_fmt("lowest foot joint y=%+.4f vs skeleton minY=%+.4f", foot_lo, ymin));
    }

    // --- C6: limb proportions. The naming never looks at bone LENGTHS at all, so these
    // are fully independent evidence. But a non-human ratio means the CHARACTER is
    // stylised, not that the naming is wrong -> ANAT (warn), not CHECK (fail).
    {
        int lh=slot(1), lk=slot(4), la=slot(7);
        int ls=slot(16), le=slot(18), lw=slot(20);
        if (lh>=0&&lk>=0&&la>=0&&ls>=0&&le>=0&&lw>=0) {
            float femur=dist(lh,lk), tibia=dist(lk,la);
            float hum=dist(ls,le), fore=dist(le,lw);
            ANAT(femur > hum * 0.85f, "femur-vs-humerus",
                  bn_fmt("femur=%.3f humerus=%.3f (ratio %.2f, human ~1.4)", femur, hum, femur/std::max(hum,1e-6f)));
            ANAT(femur > tibia * 0.75f && femur < tibia * 1.8f, "femur-vs-tibia",
                  bn_fmt("femur=%.3f tibia=%.3f (ratio %.2f, human ~1.2)", femur, tibia, femur/std::max(tibia,1e-6f)));
            ANAT(hum > fore * 0.6f && hum < fore * 1.8f, "humerus-vs-forearm",
                  bn_fmt("humerus=%.3f forearm=%.3f (ratio %.2f, human ~1.1)", hum, fore, hum/std::max(fore,1e-6f)));
        } else {
            ANAT(false, "limb-proportions", "missing a limb joint; cannot measure");
        }
    }

    // --- C7: the pelvis sits between the two hips laterally.
    {
        int p=slot(0), lh=slot(1), rh=slot(2);
        if (p>=0&&lh>=0&&rh>=0) {
            float lo=std::min(X(lh),X(rh)), hi=std::max(X(lh),X(rh));
            CHECK(X(p) >= lo-0.02f*H && X(p) <= hi+0.02f*H, "pelvis-between-hips",
                  bn_fmt("pelvis.x=%+.4f in [%.4f,%.4f]", X(p), lo, hi));
        }
    }

    // --- C8: hands are further from the midline than elbows, elbows than shoulders.
    // A chain walked in the wrong order would invert this. Naming ranked by subtree |x|
    // extent, not by the joint's own |x|.
    {
        for (int side = 0; side < 2; side++) {
            int sh=slot(side?17:16), el=slot(side?19:18), wr=slot(side?21:20);
            if (sh<0||el<0||wr<0) continue;
            bool ok = std::fabs(X(wr)) > std::fabs(X(el)) && std::fabs(X(el)) > std::fabs(X(sh));
            CHECK(ok, side ? "R-arm-extends-outward" : "L-arm-extends-outward",
                  bn_fmt("|x| shoulder=%.3f elbow=%.3f wrist=%.3f", std::fabs(X(sh)), std::fabs(X(el)), std::fabs(X(wr))));
        }
    }

    if (print) {
        for (const auto& n : R.notes) std::printf("  [NOTE] %s\n", n.c_str());
        std::printf("  ---- %d naming check(s) FAILED, %d anatomy warning(s) ----\n", fails, warns);
    }
    return fails;
}

}  // namespace rig
