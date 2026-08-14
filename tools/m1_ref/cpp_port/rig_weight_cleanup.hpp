// rig_weight_cleanup.hpp — remove the "spike" artefact from an automatically skinned character.
//
// ===========================================================================================
// THE DEFECT
// ===========================================================================================
// On char1 (V=247,727, J=63, bbox diagonal 2.437) the distance from each vertex to the joint it is
// MOST strongly bound to reads:
//
//     p50=0.045   p99=0.202   p999=0.212   max=0.512      (fractions of the bbox diagonal)
//     vertices bound >0.25 diag from their joint:  97
//     vertices bound >0.35 diag from their joint:  97
//     vertices bound >0.50 diag from their joint:  97
//
// The count does not move between 0.25, 0.35 and 0.50. That is not the tail of a smooth
// distribution — it is a cleanly separated outlier population of 97 vertices (0.039% of the mesh)
// bound to a joint more than half a body away. When that joint rotates they are flung across the
// character, and what the owner sees is "spears hanging off the arms". It is also why the pose gate
// reads 25.877 with a max_stretch of 214x while the median vertex is completely fine: p999 over
// 1.5M edges is the worst ~1500 edges, and ~100 spiked vertices own about that many.
//
// ===========================================================================================
// THE RULE, AND WHY IT IS NOT A SPATIAL CUTOFF
// ===========================================================================================
// There is prior art to avoid. SkinTokens' `--use_postprocess` applies a hard spatial cutoff: any
// influence beyond a fixed radius dies. It does fix leaks, and it also collapses legitimate blends —
// on the same assets it took shoulder influences from 2.35 to 1.20 and rigid seam edges from 129 to
// 2032. Trading spikes for rigid seams is not a win. A cleanup that touches 5% of the mesh to fix
// 0.04% of it has done more harm than the defect.
//
// So the rule here is a CONJUNCTION of a relative and an absolute test, and an influence must fail
// BOTH to be touched:
//
//     d(v,j) = distance from vertex v to joint j's INFLUENCE REGION — the union of the bone segment
//              from j's parent to j and the segments from j to each of its children, not the
//              distance to the joint point. A vertex on the middle of an upper arm is far from both
//              its endpoints and near neither joint, but it is ON the bone.
//
//     dnear(v) = min over ALL joints of d(v,j). This is the offset from the skeleton that this
//              vertex cannot avoid — large on a thick torso, tiny on a finger. It makes the test
//              self-calibrating to body thickness instead of assuming a radius.
//
//     influence (v,j) is an OUTLIER  <=>  d(v,j) > far_ratio * dnear(v)
//                                    AND  d(v,j) > far_frac_diag * bbox_diagonal
//
// The relative term is what a hard cutoff lacks: a shoulder vertex sits near where the spine bone
// and the arm bone meet, so BOTH of its influences have d ~ dnear and the ratio is ~1 — untouchable
// at any far_ratio. The absolute term is the guard the relative term lacks: a vertex lying almost
// exactly on a bone has dnear ~ 0, which would make every other influence "infinitely far" in ratio
// terms. Requiring a real, body-scaled distance as well means the rule can only ever fire on an
// influence that is BOTH disproportionate for this vertex AND far in absolute terms. Both defaults
// (far_ratio 4.0, far_frac_diag 0.12 diag) sit in a gap the measured distributions leave empty.
//
// ===========================================================================================
// WHAT HAPPENS TO AN OUTLIER — AND WHY THE REPAIR IS NOT A SNAP TO THE NEAREST BONE
// ===========================================================================================
//  1. Drop the outlier influence, renormalise the vertex's remaining weights. Most spiked vertices
//     keep a perfectly good secondary influence and this alone fixes them.
//  2. If dropping leaves the vertex with no weight at all, do NOT snap it to the nearest bone —
//     that is precisely the move that manufactures rigid seams. Instead INPAINT it from its mesh
//     neighbours: a Jacobi wave over the position-welded surface graph, taking the mean weight
//     vector of the neighbours that are already clean. A clean neighbour carries a proper two-bone
//     blend, so the repaired vertex inherits a blend, not a hard assignment. Then a couple of
//     restricted Laplacian rounds over ONLY the changed vertices, to feather the patch into the
//     skin around it.
//  3. If a whole surface island is empty — a detached prop legitimately bound to one distant bone —
//     inpainting has nothing to draw from. Restore that island's ORIGINAL weights untouched and
//     count it. The cleanup never leaves a vertex worse than it found it.
//
// Vertices whose influences all pass the test are bit-for-bit untouched.
//
// ===========================================================================================
// THE SECOND DEFECT: A WEIGHT FIELD THAT VARIES FASTER THAN THE MESH CAN CARRY
// ===========================================================================================
// char1's spike is a distance outlier. gilly's is NOT, and no distance rule will ever find it. Its
// worst edge under the arms-raised pose stretches 209x, and its two endpoints read:
//
//     v20732  [j05 w=.245 d=.318][j01 w=.408 d=.352][j16 w=.169 d=.447][j03 w=.177 d=.300]
//     v20735  [j05 w=.243 d=.318][j01 w=.407 d=.351][j02 w=.171 d=.306][j03 w=.179 d=.300]
//                                             ^^^^ the mirror joint, on the neighbouring vertex
//     rest edge length = 0.00141 = 5.2e-4 of the diagonal
//
// Two ADJACENT vertices, 0.05% of a body apart, one bound 17% to j16 and the other 17% to its
// left/right mirror j02. Both joints are a plausible distance away — d/dnear is 1.5, nowhere near
// any outlier threshold — so the field is not "wrong" anywhere, it is DISCONTINUOUS. When the two
// mirrored joints swing opposite ways, a 0.0014-long edge is asked to absorb the difference.
//
// The invariant that catches this is dimensional, and it is the natural one: a skin weight field has
// to be Lipschitz with respect to the surface, at the scale the bones' lever arms set. Define, for
// a welded edge (a,b) of rest length e,
//
//     tension(a,b) = SUM_j |w_a(j) - w_b(j)| * max(d(a,bone_j), d(b,bone_j))  /  e
//
// which bounds the differential displacement of the two endpoints per radian of joint rotation,
// divided by the length that has to absorb it. A 45-degree (0.785 rad) audit rotation therefore
// stretches that edge by at most about 1 + 0.785 * tension. Tension is a property of the REST pose
// and the weights alone — no pose is chosen, nothing is fitted to the gate.
//
// Where tension exceeds `max_tension` the field is locally smoothed — and ONLY there. On a rig whose
// field is already smooth the pass finds no edges and does nothing at all, which is what keeps it
// from becoming the blanket smoothing that flattens blends. It also self-terminates: smoothing
// reduces |w_a - w_b|, so the over-tension set shrinks each round.
//
// The mesh is UV-chart split (26,932 charts on char1), so index adjacency is severed at every chart
// boundary and a raw neighbour walk would leak nothing across a seam. Everything here runs on
// POSITION-WELDED representatives, the same weld rig_pose_gate.hpp's component labeller uses.
//
// Header-only, CPU, no ggml, OpenMP-parallel where it matters. ~0.5 s on a 250k-vertex character.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "rig_pose_gate.hpp"   // the SHIPPED gate — the cleanup verifies itself against it

namespace rigclean {

struct Options {
    // An influence must fail BOTH tests to be dropped. See the header comment.
    float far_ratio     = 4.0f;   // d(v,j) > far_ratio * dnear(v)
    float far_frac_diag = 0.12f;  // AND d(v,j) > far_frac_diag * bbox diagonal
    // Influences at or below this are left alone by rule 1. They are numerically negligible as
    // WEIGHTS (a 0.001 share of a vertex's motion), but they are not negligible as a
    // DISCONTINUITY on a 5e-4-diagonal edge — so rule 2, which measures exactly that, is what
    // handles them. Leaving them to rule 2 is what takes rule 1's footprint on miku from 54% of
    // the mesh to 0.2%.
    float min_weight    = 0.01f;
    int   inpaint_rounds = 64;    // Jacobi waves available to refill emptied vertices
    int   smooth_rounds  = 2;     // restricted Laplacian passes over the CHANGED vertices only
    float smooth_blend   = 0.5f;  // per pass: (1-b)*own + b*mean(neighbours)

    // Pass 2 — the Lipschitz repair. `max_tension` is in units of (displacement per radian) per
    // (rest edge length): a 45-degree audit rotation stretches an edge by at most ~1 + 0.785*T.
    // 0 disables the pass entirely.
    bool  tension_enabled = true; // turn the pass off WITHOUT changing what `max_tension` reports
    float max_tension    = 4.0f;
    int   tension_rounds = 24;    // smoothing waves; the over-tension set shrinks every round
    float tension_blend  = 0.6f;

    bool  far_enabled    = true;  // likewise for rule 1
    bool  enabled        = true;
};

// Everything the report needs. Distances are fractions of the mesh bbox diagonal.
struct Stats {
    int    V = 0, J = 0;
    double diag = 0;

    long   influences_dropped   = 0;  // individual (vertex, joint) influences zeroed
    long   vertices_changed     = 0;  // vertices whose top-4 skin differs at all afterwards
    // ... of which the change was MATERIAL (L1 >= 0.05). The bare `vertices_changed` count is
    // dominated by 0.001-weight influences on distant joints: correct to remove, invisible to move.
    long   vertices_changed_material = 0;
    double dropped_weight_total = 0;  // sum of every weight removed, in units of "whole vertices"
    long   vertices_emptied     = 0;  // lost every influence to the rule
    long   vertices_inpainted   = 0;  // ... and were refilled from clean neighbours
    long   vertices_restored    = 0;  // ... and could not be, so kept their original weights
    double max_dropped_weight   = 0;  // the largest single weight the rule removed
    double max_weight_delta     = 0;  // largest per-vertex L1 change in the weight vector

    // vertex -> DOMINANT joint distance, as the brief measures it (to the joint POINT).
    double dom_p50 = 0, dom_p99 = 0, dom_p999 = 0, dom_max = 0;
    long   dom_over25 = 0, dom_over35 = 0, dom_over50 = 0;
    // vertex -> dominant joint's BONE, which is what the rule actually thresholds.
    double bone_p50 = 0, bone_p99 = 0, bone_p999 = 0, bone_max = 0;

    // Blend-preservation instruments — the guard against trading spikes for rigid seams.
    double mean_influences   = 0;  // mean count of influences >= 0.02 per vertex
    double blended_frac      = 0;  // fraction of vertices whose 2nd-largest weight >= 0.10
    long   rigid_seam_edges  = 0;  // welded edges whose endpoints share < 0.05 of common influence

    // Edge tension: differential displacement per radian, per unit of rest edge length.
    double tension_p50 = 0, tension_p99 = 0, tension_p999 = 0, tension_max = 0;
    long   tension_over  = 0;      // welded edges above the configured max_tension
    long   tension_verts = 0;      // vertices smoothed by the Lipschitz pass
};

namespace detail {

struct Weld {
    std::vector<int32_t>              rep;    // V -> representative vertex id
    std::vector<std::vector<int32_t>> nbr;    // rep -> welded neighbour reps (only rep slots filled)
    std::vector<std::vector<int32_t>> members;// rep -> every vertex sharing that position
};

// Position weld + welded adjacency. Same quantisation as rig_pose_gate.hpp's component labeller:
// 1e-6 of the largest bbox extent, which merges UV-chart duplicates and nothing else.
inline void build_weld(const std::vector<float>& verts, const std::vector<int32_t>& faces, Weld& w) {
    const int V = (int)(verts.size() / 3);
    std::vector<int32_t> uf((size_t)V);
    for (int i = 0; i < V; ++i) uf[(size_t)i] = i;
    std::function<int32_t(int32_t)> find = [&](int32_t v) {
        while (uf[(size_t)v] != v) { uf[(size_t)v] = uf[(size_t)uf[(size_t)v]]; v = uf[(size_t)v]; }
        return v;
    };
    float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < V; ++i) for (int a = 0; a < 3; ++a) {
        const float x = verts[(size_t)i * 3 + a];
        mn[a] = std::min(mn[a], x); mx[a] = std::max(mx[a], x);
    }
    float extf = 0.f;
    for (int a = 0; a < 3; ++a) extf = std::max(extf, mx[a] - mn[a]);
    const double quantum = std::max((double)extf * 1e-6, 1e-8);

    struct Key { int64_t k[3]; int32_t v; };
    std::vector<Key> keys((size_t)V);
    for (int i = 0; i < V; ++i) {
        for (int a = 0; a < 3; ++a)
            keys[(size_t)i].k[a] = (int64_t)std::nearbyint((double)verts[(size_t)i * 3 + a] / quantum);
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
        const int32_t ra = find(a.v), rb = find(b.v);
        if (ra != rb) uf[(size_t)rb] = ra;
    }
    w.rep.assign((size_t)V, 0);
    for (int i = 0; i < V; ++i) w.rep[(size_t)i] = find((int32_t)i);
    w.members.assign((size_t)V, {});
    for (int i = 0; i < V; ++i) w.members[(size_t)w.rep[(size_t)i]].push_back((int32_t)i);

    w.nbr.assign((size_t)V, {});
    const size_t F = faces.size() / 3;
    for (size_t f = 0; f < F; ++f) {
        const int32_t t[3] = {w.rep[(size_t)faces[f * 3 + 0]],
                              w.rep[(size_t)faces[f * 3 + 1]],
                              w.rep[(size_t)faces[f * 3 + 2]]};
        for (int e = 0; e < 3; ++e) {
            const int32_t a = t[e], b = t[(e + 1) % 3];
            if (a == b) continue;
            w.nbr[(size_t)a].push_back(b);
            w.nbr[(size_t)b].push_back(a);
        }
    }
    for (int i = 0; i < V; ++i) {
        auto& r = w.nbr[(size_t)i];
        if (r.empty()) continue;
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
    }
}

inline double point_seg_dist2(const float* p, const float* a, const float* b) {
    const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    const double den = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    double t = 0.0;
    if (den > 1e-24) {
        t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / den;
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
    }
    const double d[3] = {ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]};
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
}

// Joint j's influence region: every bone segment incident on j (to its parent, and to each child),
// plus j itself so an isolated joint still has a region.
struct BoneRegions {
    std::vector<float>   pts;    // flat list of segment endpoints, 6 floats per segment
    std::vector<int32_t> begin;  // J+1 offsets into `pts` in units of segments
};

inline void build_regions(const std::vector<float>& joint_pos, const std::vector<int32_t>& parent,
                          BoneRegions& R) {
    const int J = (int)parent.size();
    std::vector<std::vector<int32_t>> child((size_t)J);
    for (int j = 0; j < J; ++j) {
        const int p = parent[(size_t)j];
        if (p >= 0 && p < J) child[(size_t)p].push_back(j);
    }
    R.begin.assign((size_t)J + 1, 0);
    R.pts.clear();
    for (int j = 0; j < J; ++j) {
        R.begin[(size_t)j] = (int32_t)(R.pts.size() / 6);
        const float* pj = &joint_pos[(size_t)j * 3];
        auto push = [&](const float* a, const float* b) {
            R.pts.insert(R.pts.end(), {a[0], a[1], a[2], b[0], b[1], b[2]});
        };
        const int p = parent[(size_t)j];
        if (p >= 0 && p < J) push(&joint_pos[(size_t)p * 3], pj);
        for (int32_t c : child[(size_t)j]) push(pj, &joint_pos[(size_t)c * 3]);
        if ((int32_t)(R.pts.size() / 6) == R.begin[(size_t)j]) push(pj, pj);
    }
    R.begin[(size_t)J] = (int32_t)(R.pts.size() / 6);
}

inline double region_dist(const BoneRegions& R, int j, const float* p) {
    double best = INFINITY;
    for (int32_t s = R.begin[(size_t)j]; s < R.begin[(size_t)j + 1]; ++s)
        best = std::min(best, point_seg_dist2(p, &R.pts[(size_t)s * 6], &R.pts[(size_t)s * 6 + 3]));
    return std::sqrt(best);
}

inline double quantile_sorted(std::vector<double>& v, double q) {
    if (v.empty()) return 0;
    if (v.size() == 1) return v[0];
    std::sort(v.begin(), v.end());
    const double pos = q * (double)(v.size() - 1);
    size_t lo = (size_t)std::floor(pos);
    if (lo >= v.size() - 1) lo = v.size() - 2;
    return v[lo] + (pos - (double)lo) * (v[lo + 1] - v[lo]);
}

// Sort a vertex's 4 influences by descending weight (keeps the encoding canonical, which is what
// the top-k4 writer produces), then renormalise.
inline void canonicalise(int32_t* ji, float* w) {
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
            if (w[b] > w[a]) { std::swap(w[a], w[b]); std::swap(ji[a], ji[b]); }
    double s = 0;
    for (int k = 0; k < 4; ++k) { if (!(w[k] > 0)) { w[k] = 0.f; ji[k] = 0; } s += w[k]; }
    if (s > 1e-12) for (int k = 0; k < 4; ++k) w[k] = (float)(w[k] / s);
}

// Collapse a (joint -> weight) accumulator down to the 4 heaviest, normalised.
inline void topk4_from_pairs(std::vector<std::pair<int32_t, double>>& acc, int32_t* ji, float* w) {
    std::sort(acc.begin(), acc.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    double s = 0;
    const size_t n = std::min<size_t>(4, acc.size());
    for (size_t k = 0; k < n; ++k) s += acc[k].second;
    for (int k = 0; k < 4; ++k) { ji[k] = 0; w[k] = 0.f; }
    if (s <= 1e-12) return;
    for (size_t k = 0; k < n; ++k) { ji[k] = acc[k].first; w[k] = (float)(acc[k].second / s); }
}

inline void accumulate(std::vector<std::pair<int32_t, double>>& acc, const int32_t* ji,
                       const float* w, double scale) {
    for (int k = 0; k < 4; ++k) {
        if (!(w[k] > 0)) continue;
        bool found = false;
        for (auto& e : acc) if (e.first == ji[k]) { e.second += (double)w[k] * scale; found = true; break; }
        if (!found) acc.emplace_back(ji[k], (double)w[k] * scale);
    }
}

// tension(a,b) = SUM_j |w_a(j) - w_b(j)| * max(d(a,bone_j), d(b,bone_j)) / rest_len(a,b).
// An upper bound on how much differential displacement one radian of joint rotation can force
// across this edge, divided by the length that has to absorb it. Rest-pose only — no pose is
// chosen and nothing is fitted to the gate.
inline double edge_tension_wv(const std::vector<float>& verts, const BoneRegions& regions,
                              int32_t a, const int32_t* ja, const float* wa_,
                              int32_t b, const int32_t* jb, const float* wb_, double len_floor) {
    const float* pa = &verts[(size_t)a * 3];
    const float* pb = &verts[(size_t)b * 3];
    const double dx = pa[0] - pb[0], dy = pa[1] - pb[1], dz = pa[2] - pb[2];
    const double e = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(e > len_floor)) return 0.0;   // a degenerate edge carries no information
    int32_t js[8]; double wa[8], wb[8]; int n = 0;
    auto put = [&](int32_t j, double w, bool side_a) {
        for (int i = 0; i < n; ++i) if (js[i] == j) { (side_a ? wa[i] : wb[i]) += w; return; }
        if (n >= 8) return;
        js[n] = j; wa[n] = 0; wb[n] = 0; (side_a ? wa[n] : wb[n]) = w; ++n;
    };
    for (int k = 0; k < 4; ++k) {
        if (wa_[k] > 0) put(ja[k], (double)wa_[k], true);
        if (wb_[k] > 0) put(jb[k], (double)wb_[k], false);
    }
    double acc = 0;
    for (int i = 0; i < n; ++i) {
        const double dw = std::fabs(wa[i] - wb[i]);
        if (dw < 1e-6) continue;
        acc += dw * std::max(region_dist(regions, js[i], pa), region_dist(regions, js[i], pb));
    }
    return acc / e;
}

inline double edge_tension(const std::vector<float>& verts, const std::vector<int32_t>& jidx,
                           const std::vector<float>& jw, const BoneRegions& regions,
                           int32_t a, int32_t b, double len_floor) {
    return edge_tension_wv(verts, regions, a, &jidx[(size_t)a * 4], &jw[(size_t)a * 4],
                           b, &jidx[(size_t)b * 4], &jw[(size_t)b * 4], len_floor);
}

// The blend-preservation instruments, measured on whatever skin is handed in.
inline void measure_blends(const std::vector<int32_t>& jidx, const std::vector<float>& jw,
                           const Weld& weld, int V, Stats& st) {
    double inf_sum = 0; long blended = 0;
    for (int v = 0; v < V; ++v) {
        int c = 0; float first = 0, second = 0;
        for (int k = 0; k < 4; ++k) {
            const float x = jw[(size_t)v * 4 + k];
            if (x >= 0.02f) ++c;
            if (x > first) { second = first; first = x; }
            else if (x > second) second = x;
        }
        inf_sum += c;
        if (second >= 0.10f) ++blended;
    }
    st.mean_influences = V ? inf_sum / V : 0;
    st.blended_frac    = V ? (double)blended / V : 0;

    long seams = 0;
    for (int a = 0; a < V; ++a) {
        if (weld.rep[(size_t)a] != a) continue;
        for (int32_t b : weld.nbr[(size_t)a]) {
            if (b <= a) continue;
            double shared = 0;
            for (int ka = 0; ka < 4; ++ka) {
                const float wa = jw[(size_t)a * 4 + ka];
                if (!(wa > 0)) continue;
                for (int kb = 0; kb < 4; ++kb)
                    if (jidx[(size_t)b * 4 + kb] == jidx[(size_t)a * 4 + ka])
                        shared += std::min(wa, jw[(size_t)b * 4 + kb]);
            }
            if (shared < 0.05) ++seams;
        }
    }
    st.rigid_seam_edges = seams;
}

// Tension over every welded edge. `len_floor` matches the pose gate's own edge-length floor
// (diag * 1e-4) so the two instruments look at the same edges.
inline void measure_tension(const std::vector<float>& verts, const std::vector<int32_t>& jidx,
                            const std::vector<float>& jw, const BoneRegions& regions,
                            const Weld& weld, int V, double len_floor, double max_tension,
                            Stats& st) {
    std::vector<int32_t> ea, eb;
    ea.reserve((size_t)V * 3); eb.reserve((size_t)V * 3);
    for (int a = 0; a < V; ++a) {
        if (weld.rep[(size_t)a] != a) continue;
        for (int32_t b : weld.nbr[(size_t)a]) if (b > a) { ea.push_back((int32_t)a); eb.push_back(b); }
    }
    std::vector<double> t(ea.size(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < (long long)ea.size(); ++i)
        t[(size_t)i] = edge_tension(verts, jidx, jw, regions, ea[(size_t)i], eb[(size_t)i], len_floor);
    st.tension_over = 0;
    for (double x : t) if (x > max_tension) ++st.tension_over;
    if (t.empty()) return;
    st.tension_max = *std::max_element(t.begin(), t.end());
    std::vector<double> tmp = t;
    st.tension_p50 = quantile_sorted(tmp, 0.50);  tmp = t;
    st.tension_p99 = quantile_sorted(tmp, 0.99);  tmp = t;
    st.tension_p999 = quantile_sorted(tmp, 0.999);
}

// vertex -> dominant joint distances, both to the joint POINT (the brief's measure) and to the
// joint's bone region (what the rule thresholds).
inline void measure_distance_distribution(const std::vector<float>& verts,
                                          const std::vector<int32_t>& jidx,
                                          const std::vector<float>& jw,
                                          const std::vector<float>& joint_pos,
                                          const BoneRegions& regions, int V, int J, double diag,
                                          Stats& st) {
    std::vector<double> dom((size_t)V), bone((size_t)V);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long v = 0; v < (long long)V; ++v) {
        int bk = 0;
        for (int k = 1; k < 4; ++k)
            if (jw[(size_t)v * 4 + k] > jw[(size_t)v * 4 + bk]) bk = k;
        const int j = jidx[(size_t)v * 4 + bk];
        const float* p = &verts[(size_t)v * 3];
        if (j < 0 || j >= J) { dom[(size_t)v] = 0; bone[(size_t)v] = 0; continue; }
        const double dx = p[0] - joint_pos[(size_t)j * 3 + 0];
        const double dy = p[1] - joint_pos[(size_t)j * 3 + 1];
        const double dz = p[2] - joint_pos[(size_t)j * 3 + 2];
        dom[(size_t)v]  = std::sqrt(dx * dx + dy * dy + dz * dz) / diag;
        bone[(size_t)v] = region_dist(regions, j, p) / diag;
    }
    st.dom_over25 = st.dom_over35 = st.dom_over50 = 0;
    for (double d : dom) {
        if (d > 0.25) ++st.dom_over25;
        if (d > 0.35) ++st.dom_over35;
        if (d > 0.50) ++st.dom_over50;
    }
    std::vector<double> t = dom;
    st.dom_p50 = quantile_sorted(t, 0.50);   t = dom;
    st.dom_p99 = quantile_sorted(t, 0.99);   t = dom;
    st.dom_p999 = quantile_sorted(t, 0.999); t = dom;
    st.dom_max = *std::max_element(dom.begin(), dom.end());
    t = bone;
    st.bone_p50 = quantile_sorted(t, 0.50);  t = bone;
    st.bone_p99 = quantile_sorted(t, 0.99);  t = bone;
    st.bone_p999 = quantile_sorted(t, 0.999);
    st.bone_max = *std::max_element(bone.begin(), bone.end());
}

}  // namespace detail

// ---------------------------------------------------------------------------
// The cleanup. `jidx` / `jw` are V*4 in and out; every other array is read-only.
// `joint_pos` must be joint REST positions in the same space as `verts`.
// `before` / `after` are filled with the measurement set (pass nullptr to skip either).
// Returns false only on malformed input.
// ---------------------------------------------------------------------------
inline bool clean_skin_weights(const std::vector<float>& verts,        // V*3
                               const std::vector<int32_t>& faces,      // F*3
                               const std::vector<float>& joint_pos,    // J*3, rest, mesh space
                               const std::vector<int32_t>& parent,     // J
                               std::vector<int32_t>& jidx,             // V*4  (in/out)
                               std::vector<float>& jw,                 // V*4  (in/out)
                               const Options& opt,
                               Stats* before, Stats* after) {
    const int V = (int)(verts.size() / 3);
    const int J = (int)parent.size();
    if (V == 0 || J == 0 || joint_pos.size() != (size_t)J * 3) return false;
    if (jidx.size() != (size_t)V * 4 || jw.size() != (size_t)V * 4) return false;

    float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < V; ++i) for (int a = 0; a < 3; ++a) {
        const float x = verts[(size_t)i * 3 + a];
        mn[a] = std::min(mn[a], x); mx[a] = std::max(mx[a], x);
    }
    double diag = 0;
    for (int a = 0; a < 3; ++a) { const double d = mx[a] - mn[a]; diag += d * d; }
    diag = std::sqrt(diag);
    if (!(diag > 0)) return false;

    detail::BoneRegions regions;
    detail::build_regions(joint_pos, parent, regions);
    detail::Weld weld;
    detail::build_weld(verts, faces, weld);

    const std::vector<int32_t> jidx0 = jidx;
    const std::vector<float>   jw0   = jw;

    if (before) {
        *before = Stats{};
        before->V = V; before->J = J; before->diag = diag;
        detail::measure_distance_distribution(verts, jidx, jw, joint_pos, regions, V, J, diag, *before);
        detail::measure_blends(jidx, jw, weld, V, *before);
        detail::measure_tension(verts, jidx, jw, regions, weld, V, diag * 1e-4, opt.max_tension, *before);
    }
    if (!opt.enabled) {
        if (after) { *after = before ? *before : Stats{}; }
        return true;
    }

    // ---- 1. per-vertex outlier test ------------------------------------------------------------
    const double abs_floor = (double)opt.far_frac_diag * diag;
    std::vector<uint8_t> emptied((size_t)V, 0), touched((size_t)V, 0);
    long dropped = 0; double max_dropped_w = 0, dropped_total = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : dropped, dropped_total) reduction(max : max_dropped_w)
#endif
    for (long long vv = 0; vv < (long long)V; ++vv) {
        const size_t v = (size_t)vv;
        const float* p = &verts[v * 3];
        double dnear = INFINITY;
        for (int j = 0; j < J; ++j) dnear = std::min(dnear, detail::region_dist(regions, j, p));
        const double limit = std::max((double)opt.far_ratio * dnear, abs_floor);

        bool any = false;
        for (int k = 0; k < 4; ++k) {
            const float w = jw[v * 4 + k];
            if (!(w > opt.min_weight)) continue;
            const int j = jidx[v * 4 + k];
            if (j < 0 || j >= J) continue;
            any = true;
            if (opt.far_enabled && detail::region_dist(regions, j, p) > limit) {
                jw[v * 4 + k] = 0.f;
                ++dropped;
                dropped_total += (double)w;
                max_dropped_w = std::max(max_dropped_w, (double)w);
                touched[v] = 1;
            }
        }
        if (!any) continue;   // a vertex with no usable influence at all is left exactly as found
        double s = 0;
        for (int k = 0; k < 4; ++k) s += jw[v * 4 + k];
        if (s <= 1e-9) { if (touched[v]) emptied[v] = 1; }
        else if (touched[v]) {
            for (int k = 0; k < 4; ++k) jw[v * 4 + k] = (float)(jw[v * 4 + k] / s);
            detail::canonicalise(&jidx[v * 4], &jw[v * 4]);
        }
    }

    // ---- 2. inpaint the emptied vertices from clean welded neighbours ---------------------------
    // Emptiness is a property of the welded POSITION, not of a chart copy, so drive the wave over
    // representatives and write the result back to every duplicate.
    std::vector<int32_t> empty_reps;
    for (int v = 0; v < V; ++v)
        if (emptied[(size_t)v] && weld.rep[(size_t)v] == v) empty_reps.push_back((int32_t)v);
    // A duplicate whose representative is not itself empty (possible only if the two copies carried
    // different weights) is refilled from the representative directly.
    long n_emptied = 0;
    for (int v = 0; v < V; ++v) if (emptied[(size_t)v]) ++n_emptied;

    std::vector<uint8_t> still_empty((size_t)V, 0);
    for (int32_t r : empty_reps) still_empty[(size_t)r] = 1;

    long inpainted_reps = 0;
    for (int round = 0; round < opt.inpaint_rounds; ++round) {
        std::vector<int32_t> filled_now;
        std::vector<std::array<float, 4>> new_w;
        std::vector<std::array<int32_t, 4>> new_j;
        for (int32_t r : empty_reps) {
            if (!still_empty[(size_t)r]) continue;
            std::vector<std::pair<int32_t, double>> acc;
            for (int32_t nb : weld.nbr[(size_t)r]) {
                if (still_empty[(size_t)nb]) continue;
                detail::accumulate(acc, &jidx[(size_t)nb * 4], &jw[(size_t)nb * 4], 1.0);
            }
            if (acc.empty()) continue;
            std::array<int32_t, 4> ji{}; std::array<float, 4> w{};
            detail::topk4_from_pairs(acc, ji.data(), w.data());
            if (!(w[0] > 0)) continue;
            filled_now.push_back(r); new_j.push_back(ji); new_w.push_back(w);
        }
        if (filled_now.empty()) break;
        for (size_t i = 0; i < filled_now.size(); ++i) {
            const int32_t r = filled_now[i];
            for (int k = 0; k < 4; ++k) { jidx[(size_t)r * 4 + k] = new_j[i][k]; jw[(size_t)r * 4 + k] = new_w[i][k]; }
            still_empty[(size_t)r] = 0;
            ++inpainted_reps;
        }
    }

    // ---- 3. an island nothing could reach keeps its ORIGINAL weights --------------------------
    long restored = 0;
    for (int32_t r : empty_reps)
        if (still_empty[(size_t)r]) {
            for (int k = 0; k < 4; ++k) {
                jidx[(size_t)r * 4 + k] = jidx0[(size_t)r * 4 + k];
                jw[(size_t)r * 4 + k]   = jw0[(size_t)r * 4 + k];
            }
            touched[(size_t)r] = 0;
            ++restored;
        }

    // Propagate every representative's decision to its position duplicates, so a chart seam cannot
    // end up with two different skins for one physical point.
    for (int32_t r : empty_reps)
        for (int32_t m : weld.members[(size_t)r]) {
            if (m == r) continue;
            for (int k = 0; k < 4; ++k) {
                jidx[(size_t)m * 4 + k] = jidx[(size_t)r * 4 + k];
                jw[(size_t)m * 4 + k]   = jw[(size_t)r * 4 + k];
            }
            touched[(size_t)m] = touched[(size_t)r];
        }
    // Any vertex still empty (a duplicate of a non-empty rep never gets here) falls back too.
    for (int v = 0; v < V; ++v) {
        double s = 0;
        for (int k = 0; k < 4; ++k) s += jw[(size_t)v * 4 + k];
        if (s > 1e-9) continue;
        for (int k = 0; k < 4; ++k) {
            jidx[(size_t)v * 4 + k] = jidx0[(size_t)v * 4 + k];
            jw[(size_t)v * 4 + k]   = jw0[(size_t)v * 4 + k];
        }
        touched[(size_t)v] = 0;
    }

    // ---- 4. feather the patch — restricted to vertices this pass already changed ---------------
    if (opt.smooth_rounds > 0) {
        std::vector<int32_t> changed_reps;
        for (int v = 0; v < V; ++v) if (touched[(size_t)v] && weld.rep[(size_t)v] == v) changed_reps.push_back((int32_t)v);
        for (int round = 0; round < opt.smooth_rounds; ++round) {
            std::vector<std::array<int32_t, 4>> nj(changed_reps.size());
            std::vector<std::array<float, 4>>   nw(changed_reps.size());
            for (size_t i = 0; i < changed_reps.size(); ++i) {
                const int32_t r = changed_reps[i];
                std::vector<std::pair<int32_t, double>> acc;
                detail::accumulate(acc, &jidx[(size_t)r * 4], &jw[(size_t)r * 4], 1.0 - opt.smooth_blend);
                const auto& nb = weld.nbr[(size_t)r];
                if (!nb.empty()) {
                    const double s = (double)opt.smooth_blend / (double)nb.size();
                    for (int32_t b : nb) detail::accumulate(acc, &jidx[(size_t)b * 4], &jw[(size_t)b * 4], s);
                }
                detail::topk4_from_pairs(acc, nj[i].data(), nw[i].data());
            }
            for (size_t i = 0; i < changed_reps.size(); ++i) {
                const int32_t r = changed_reps[i];
                if (!(nw[i][0] > 0)) continue;
                for (int32_t m : weld.members[(size_t)r])
                    for (int k = 0; k < 4; ++k) {
                        jidx[(size_t)m * 4 + k] = nj[i][k];
                        jw[(size_t)m * 4 + k]   = nw[i][k];
                    }
            }
        }
    }

    // ---- 4b. the Lipschitz repair — smooth ONLY where the field outruns the surface ------------
    // Rebuilt from scratch each round because smoothing one edge's endpoints changes its
    // neighbours' tension too. The set shrinks monotonically in practice (averaging can only reduce
    // |w_a - w_b| along the edge that triggered it), so the loop terminates well before the cap; if
    // it does not, it stops anyway and the remaining edges are reported, not hidden.
    long tension_verts = 0;
    if (opt.tension_enabled && opt.max_tension > 0 && opt.tension_rounds > 0) {
        const double len_floor = diag * 1e-4;
        std::vector<uint8_t> hot((size_t)V, 0);
        std::vector<uint8_t> ever_hot((size_t)V, 0);
        for (int round = 0; round < opt.tension_rounds; ++round) {
            std::fill(hot.begin(), hot.end(), 0);
            long n_hot = 0;
            for (int a = 0; a < V; ++a) {
                if (weld.rep[(size_t)a] != a) continue;
                for (int32_t b : weld.nbr[(size_t)a]) {
                    if (b <= a) continue;
                    if (detail::edge_tension(verts, jidx, jw, regions, (int32_t)a, b, len_floor) > opt.max_tension) {
                        if (!hot[(size_t)a]) { hot[(size_t)a] = 1; ++n_hot; }
                        if (!hot[(size_t)b]) { hot[(size_t)b] = 1; ++n_hot; }
                    }
                }
            }
            if (n_hot == 0) break;
            std::vector<int32_t> reps;
            reps.reserve((size_t)n_hot);
            for (int v = 0; v < V; ++v) if (hot[(size_t)v]) reps.push_back((int32_t)v);
            std::vector<std::array<int32_t, 4>> nj(reps.size());
            std::vector<std::array<float, 4>>   nw(reps.size());
            std::vector<uint8_t> accept(reps.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (long long i = 0; i < (long long)reps.size(); ++i) {
                const int32_t r = reps[(size_t)i];
                std::vector<std::pair<int32_t, double>> acc;
                detail::accumulate(acc, &jidx[(size_t)r * 4], &jw[(size_t)r * 4], 1.0 - opt.tension_blend);
                const auto& nb = weld.nbr[(size_t)r];
                if (!nb.empty()) {
                    const double s = (double)opt.tension_blend / (double)nb.size();
                    for (int32_t b : nb) detail::accumulate(acc, &jidx[(size_t)b * 4], &jw[(size_t)b * 4], s);
                }
                detail::topk4_from_pairs(acc, nj[(size_t)i].data(), nw[(size_t)i].data());
                if (!(nw[(size_t)i][0] > 0)) continue;
                // ACCEPT ONLY A STRICT LOCAL IMPROVEMENT. Averaging normally lowers |w_a - w_b|,
                // but the top-4 truncation that follows it can hand a distant joint a larger share
                // than it had, and a smoothed vertex sitting next to unsmoothed ones can push the
                // discontinuity one ring outwards instead of removing it. Measuring the vertex's
                // worst incident tension before and after, and keeping the update only when it
                // falls, makes the pass monotone: on a rig with nothing to fix it cannot fire, and
                // on a rig with something to fix it cannot overshoot into making it worse.
                double t_old = 0, t_new = 0;
                for (int32_t b : nb) {
                    t_old = std::max(t_old, detail::edge_tension(verts, jidx, jw, regions, r, b, len_floor));
                    t_new = std::max(t_new, detail::edge_tension_wv(
                        verts, regions, r, nj[(size_t)i].data(), nw[(size_t)i].data(),
                        b, &jidx[(size_t)b * 4], &jw[(size_t)b * 4], len_floor));
                }
                if (t_new < t_old) accept[(size_t)i] = 1;
            }
            bool any_accepted = false;
            for (size_t i = 0; i < reps.size(); ++i) {
                const int32_t r = reps[i];
                if (!accept[i]) continue;
                any_accepted = true;
                ever_hot[(size_t)r] = 1;
                for (int32_t m : weld.members[(size_t)r]) {
                    for (int k = 0; k < 4; ++k) {
                        jidx[(size_t)m * 4 + k] = nj[i][k];
                        jw[(size_t)m * 4 + k]   = nw[i][k];
                    }
                    touched[(size_t)m] = 1;
                }
            }
            if (!any_accepted) break;   // nothing left this pass can improve
        }
        for (int v = 0; v < V; ++v) if (ever_hot[(size_t)v]) tension_verts += (long)weld.members[(size_t)v].size();
    }

    // ---- 5. book-keeping ----------------------------------------------------------------------
    long changed = 0, changed_material = 0; double max_delta = 0;
    for (int v = 0; v < V; ++v) {
        double l1 = 0;
        bool differs = false;
        for (int k = 0; k < 4; ++k) {
            if (jidx[(size_t)v * 4 + k] != jidx0[(size_t)v * 4 + k] ||
                jw[(size_t)v * 4 + k]   != jw0[(size_t)v * 4 + k]) differs = true;
        }
        if (!differs) continue;
        ++changed;
        // L1 over the union of joints, computed as a small merge.
        std::vector<std::pair<int32_t, double>> a, b;
        detail::accumulate(a, &jidx0[(size_t)v * 4], &jw0[(size_t)v * 4], 1.0);
        detail::accumulate(b, &jidx[(size_t)v * 4],  &jw[(size_t)v * 4],  1.0);
        for (auto& ea : a) {
            double bv = 0;
            for (auto& eb : b) if (eb.first == ea.first) { bv = eb.second; break; }
            l1 += std::fabs(ea.second - bv);
        }
        for (auto& eb : b) {
            bool seen = false;
            for (auto& ea : a) if (ea.first == eb.first) { seen = true; break; }
            if (!seen) l1 += std::fabs(eb.second);
        }
        max_delta = std::max(max_delta, l1);
        if (l1 >= 0.05) ++changed_material;
    }

    if (after) {
        *after = Stats{};
        after->V = V; after->J = J; after->diag = diag;
        after->influences_dropped = dropped;
        after->dropped_weight_total = dropped_total;
        after->vertices_changed   = changed;
        after->vertices_changed_material = changed_material;
        after->vertices_emptied   = n_emptied;
        after->vertices_restored  = restored;
        after->vertices_inpainted = n_emptied - restored;
        after->max_dropped_weight = max_dropped_w;
        after->max_weight_delta   = max_delta;
        detail::measure_distance_distribution(verts, jidx, jw, joint_pos, regions, V, J, diag, *after);
        detail::measure_blends(jidx, jw, weld, V, *after);
        detail::measure_tension(verts, jidx, jw, regions, weld, V, diag * 1e-4, opt.max_tension, *after);
        after->tension_verts = tension_verts;
    }
    (void)inpainted_reps;
    return true;
}

// One line per side, for logs and the eye page's provenance box.
inline std::string stats_line(const char* tag, const Stats& s) {
    char b[1024];
    std::snprintf(b, sizeof(b),
        "%s V=%d J=%d diag=%.4f | dom_dist p50/p99/p999/max=%.4f/%.4f/%.4f/%.4f "
        ">0.25/0.35/0.50=%ld/%ld/%ld | bone_dist p50/p99/p999/max=%.4f/%.4f/%.4f/%.4f | "
        "mean_influences=%.3f blended=%.4f rigid_seam_edges=%ld | "
        "tension p50/p99/p999/max=%.2f/%.2f/%.2f/%.2f over=%ld",
        tag, s.V, s.J, s.diag, s.dom_p50, s.dom_p99, s.dom_p999, s.dom_max,
        s.dom_over25, s.dom_over35, s.dom_over50,
        s.bone_p50, s.bone_p99, s.bone_p999, s.bone_max,
        s.mean_influences, s.blended_frac, s.rigid_seam_edges,
        s.tension_p50, s.tension_p99, s.tension_p999, s.tension_max, s.tension_over);
    return b;
}

// ---------------------------------------------------------------------------
// THE VERIFIED LADDER — "offer an improvement, never impose a regression".
//
// Both rules are principled, and both are still HEURISTICS: rule 2's guard proves that each edit
// lowers a local UPPER BOUND on stretch, which is not the same as lowering the stretch a particular
// audit pose actually produces. On miku — a rig that was already good — the Lipschitz pass lowers
// the single-pose gate (1.979 -> 1.889) while raising the all-influential audit (3.623 -> 4.109).
// Small, and still far inside the limit, but it is a regression on the control, and a cleanup you
// are offering to run on somebody's finished character is not allowed to gamble with that.
//
// So the cleanup does not trust itself. It builds candidates from the strongest rung down, runs the
// SHIPPED pose gate on each — both modes, the exact-parity native port — and keeps the strongest
// candidate that worsens neither number by more than `tolerance`. If none does, the asset is
// returned untouched. That is a measured guarantee rather than an argued one, and it costs about
// 0.2 s per gate evaluation.
//
//   rung 3 : far-outlier removal + Lipschitz repair
//   rung 2 : Lipschitz repair only
//   rung 1 : far-outlier removal only
//   rung 0 : nothing
// ---------------------------------------------------------------------------
struct VerifiedResult {
    int         rung = 0;
    const char* rung_name = "unchanged";
    double      before_default = 0, after_default = 0;   // pose gate, shipped default mode
    double      before_allinf  = 0, after_allinf  = 0;   // pose gate, all-influential mode
    Stats       before, after;                            // `after` describes the ACCEPTED rung
    std::string note;
};

// `R` is edited in place (jidx/jw only) to the accepted rung. `joint_pos` is J*3 rest positions in
// the mesh's space. Returns false only on malformed input.
inline bool clean_skin_weights_verified(rigqc::SkinnedRig& R, const std::vector<float>& joint_pos,
                                        const Options& opt, VerifiedResult& out,
                                        double tolerance = 1.001) {
    const std::vector<int32_t> jidx0 = R.jidx;
    const std::vector<float>   jw0   = R.jw;

    auto gate_pair = [&](const rigqc::SkinnedRig& X, double& def_, double& all_) {
        rigqc::PoseGateOpts d;
        rigqc::PoseGateOpts a; a.mode = rigqc::PoseGateOpts::GenericAllInfluential;
        const rigqc::PoseGateResult Gd = rigqc::run_pose_gate(X, d);
        const rigqc::PoseGateResult Ga = rigqc::run_pose_gate(X, a);
        def_ = Gd.ok ? Gd.worst() : 0.0;
        all_ = Ga.ok ? Ga.worst() : 0.0;
    };
    gate_pair(R, out.before_default, out.before_allinf);
    out.after_default = out.before_default;
    out.after_allinf  = out.before_allinf;

    // Three candidates, evaluated and SCORED rather than taken in order. Scoring matters because
    // the strongest candidate is not always the best one: on gilly the far-outlier rule rewrites
    // 12.5% of the mesh and buys nothing the Lipschitz pass had not already bought, so the cheaper
    // candidate is the one to ship. Score = the worse of the two gate ratios; ties inside 1% go to
    // whichever candidate touched fewer vertices.
    struct Cand { const char* name; int rung; bool far_on; bool tension_on; };
    static const Cand cands[3] = {
        {"far-outlier + Lipschitz", 3, true,  true },
        {"Lipschitz only",          2, false, true },
        {"far-outlier only",        1, true,  false},
    };
    double best_rank = 1e9;
    long   best_changed = 0;
    bool   have_best = false;
    std::vector<int32_t> best_j; std::vector<float> best_w; Stats best_a; int best_i = -1;
    double best_gd = 0, best_ga = 0;

    for (int ci = 0; ci < 3; ++ci) {
        Options o = opt;
        o.far_enabled     = cands[ci].far_on;
        o.tension_enabled = cands[ci].tension_on;
        std::vector<int32_t> ji = jidx0;
        std::vector<float>   jwv = jw0;
        Stats b, a;
        if (!clean_skin_weights(R.vertices, R.faces, joint_pos, R.parent, ji, jwv, o, &b, &a))
            return false;
        if (ci == 0) out.before = b;
        rigqc::SkinnedRig X = R;
        X.jidx = ji; X.jw = jwv;
        double gd = 0, ga = 0;
        gate_pair(X, gd, ga);
        const double rd = out.before_default > 0 ? gd / out.before_default : 1.0;
        const double ra = out.before_allinf  > 0 ? ga / out.before_allinf  : 1.0;
        // ELIGIBILITY is the worse of the two ratios — no candidate may regress either number.
        // RANKING is their geometric mean, so a candidate that clearly improves one gate and leaves
        // the other alone beats one that changes nothing.
        const double score = std::max(rd, ra);
        const double rank  = std::sqrt(std::max(rd, 1e-6) * std::max(ra, 1e-6));
        char note[256];
        std::snprintf(note, sizeof(note), "%s: %.3f/%.3f (worst x%.3f, rank %.3f, %ld verts)%s; ",
                      cands[ci].name, gd, ga, score, rank, a.vertices_changed,
                      score <= tolerance ? "" : " REJECTED");
        out.note += note;
        if (score > tolerance) continue;
        const bool better = !have_best || rank < best_rank * 0.99 ||
                            (rank < best_rank * 1.01 && a.vertices_changed < best_changed);
        if (better) {
            have_best = true;
            best_rank = rank;
            best_changed = a.vertices_changed; best_j = ji; best_w = jwv; best_a = a;
            best_i = ci; best_gd = gd; best_ga = ga;
        }
    }
    if (have_best) {
        R.jidx = best_j; R.jw = best_w;
        out.rung = cands[best_i].rung;
        out.rung_name = cands[best_i].name;
        out.after = best_a;
        out.after_default = best_gd;
        out.after_allinf  = best_ga;
        return true;
    }
    out.note += "kept the original skin";
    // `after` describes rung 0: no change.
    out.after = out.before;
    out.after.influences_dropped = out.after.vertices_changed = out.after.vertices_changed_material = 0;
    out.after.vertices_emptied = out.after.vertices_inpainted = out.after.vertices_restored = 0;
    out.after.tension_verts = 0;
    out.after.dropped_weight_total = out.after.max_dropped_weight = out.after.max_weight_delta = 0;
    return true;
}

}  // namespace rigclean
