#!/usr/bin/env bash
# build_instant_meshes.sh — bootstrap + build a HEADLESS Instant Meshes batch CLI (field-aligned quad
# retopo, NO GUI/OpenGL/nanogui). QuadriFlow OOM-blows on the dense AI mesh's non-manifold/holey
# topology; Instant Meshes ingests the dense MANIFOLD source directly and emits clean field-aligned
# topology that preserves fingers. Standalone (reads .ply/.obj, writes .ply/.obj) — compiler/ABI
# independent of pixal3d. Same bootstrap-on-demand pattern as build_quadriflow.sh.
#   ./build_instant_meshes.sh           # bootstrap classic TBB 2020 + clone + patch + build
#   QF=$(./build_instant_meshes.sh -p)  # print the binary path (build if needed)
# Deps via pixi (no root, conda-forge): classic TBB 2020.2 (Instant Meshes uses the removed tbb::task /
# task_scheduler_init low-level API → oneTBB won't compile it) + eigen, into the `retopo-deps` env.
# Instant Meshes BSD-style (commercial OK, Modo ships it); TBB Apache-2.0; rply MIT; Eigen MPL2.
#
# CURVATURE-ADAPTIVE RETOPO (HANDOFF-D/E Track 1): wjakob/instant-meshes is UNIFORM (one global lattice
# spacing) so a fixed quad budget webs thin/high-curvature parts (fingers) before the flat torso needs
# that density. We add a curvature-driven per-vertex SIZING FIELD behind one knob (IM_ADAPTIVITY env or
# -A <float>; default 0 = the uniform path, bit-for-bit). The src/ tree is a fresh upstream clone
# (untracked/bootstrapped), so the change is carried as the patch heredoc below and applied here
# (idempotent) after clone — same pattern as the QuadriFlow field-math.hpp patch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IM="$HERE/../../../thirdparty/instant-meshes"
BIN="$IM/build_headless/instant_meshes_batch"
IM_REPO="${IM_REPO:-https://github.com/wjakob/instant-meshes.git}"
DEPS_ENV="${DEPS_ENV:-$HOME/.pixi/envs/retopo-deps}"

[ "${1:-}" = "-p" ] && { [ -x "$BIN" ] || "$0" >&2; echo "$BIN"; exit 0; }

# 1) classic TBB 2020 + eigen (no root)
if [ ! -f "$DEPS_ENV/include/tbb/task_scheduler_init.h" ] \
   || [ ! -f "$DEPS_ENV/include/eigen3/Eigen/Core" ]; then
  echo ">> installing classic tbb 2020.2 + eigen into pixi env retopo-deps"
  pixi global install --environment retopo-deps "tbb-devel==2020.2" "tbb==2020.2" eigen
fi

# 2) clone Instant Meshes (recursive — needs ext/{pcg32,half,dset,pss,rply}) if absent
if [ ! -f "$IM/src/batch.cpp" ]; then
  echo ">> cloning Instant Meshes (recursive)"
  git clone --recursive "$IM_REPO" "$IM"
fi

# 2b) apply the curvature-adaptive sizing-field patch (idempotent: skip if already applied). The patch
#     adds compute_sizing_field()+adaptivity to batch.{h,cpp}, a per-vertex/level sizing field to
#     hierarchy.{h,cpp}, and routes it through the position optimizer (field.cpp) + extraction collapse
#     and snap threshold (extract.cpp). See HANDOFF-E. Edit the heredoc to evolve the lever.
PATCH="$IM/im_adaptive.patch"
cat > "$PATCH" <<'IM_ADAPTIVE_PATCH_EOF'
diff --git a/src/batch.cpp b/src/batch.cpp
--- a/src/batch.cpp
+++ b/src/batch.cpp
@@ -21,12 +21,132 @@
 #include "normal.h"
 #include "extract.h"
 #include "bvh.h"
+#include <cmath>
+
+/* Curvature-driven sizing field for adaptive retopo (NAVA, the one open geometry
+   lever for carving — not normal-faking — clean low-poly fingers).
+
+   wjakob/instant-meshes is UNIFORM: one global lattice spacing, so a fixed quad
+   budget spreads evenly and thin/high-curvature parts (fingers) web together
+   before the flat torso ever needs that density. This computes a per-vertex
+   target edge length s_i so the lattice is finer where curvature is high and
+   coarser where it is low. The field is BUDGET-CONSERVING: the area-weighted
+   mean relative density is renormalized to 1, so at a fixed -f the same total
+   quad count is redistributed (torso pays for fingers), not inflated.
+
+   Density is realized downstream by feeding s_i (and per-edge means) to the
+   position-field optimizer and the extraction collapse, which already
+   parameterize on (scale, inv_scale). Returns empty when adaptivity <= 0. */
+static VectorXf compute_sizing_field(const MatrixXf &V, const MatrixXf &N,
+                                     const VectorXf &A, const AdjacencyMatrix adj,
+                                     Float base_scale, float adaptivity) {
+    const uint32_t n = (uint32_t) V.cols();
+    VectorXf curv(n); curv.setZero();
+
+    /* 1. raw curvature proxy: mean over the 1-ring of (1 - n_i . n_j) in [0,2].
+          Cheap, robust, and exactly the normal-variation that webs thin parts. */
+    for (uint32_t i = 0; i < n; ++i) {
+        Vector3f ni = N.col(i);
+        Float acc = 0; int deg = 0;
+        for (Link *l = adj[i]; l != adj[i+1]; ++l) {
+            if (l->weight == 0) continue;
+            acc += (Float)1 - ni.dot(N.col(l->id));
+            ++deg;
+        }
+        curv[i] = deg > 0 ? acc / deg : (Float)0;
+    }
+
+    /* 2. smooth the field a few 1-ring Jacobi passes (raw curvature is speckly;
+          unsmoothed it produces isolated dense islands that snap back out). */
+    for (int it = 0; it < 3; ++it) {
+        VectorXf s(n);
+        for (uint32_t i = 0; i < n; ++i) {
+            Float acc = curv[i]; int cnt = 1;
+            for (Link *l = adj[i]; l != adj[i+1]; ++l) {
+                if (l->weight == 0) continue;
+                acc += curv[l->id]; ++cnt;
+            }
+            s[i] = acc / cnt;
+        }
+        curv.swap(s);
+    }
+
+    /* 3. robust area-weighted standardization: keep only curvature that stands
+          out above the surface mean (fingers/edges), normalized by its spread. */
+    double Asum = 0, mean = 0;
+    for (uint32_t i = 0; i < n; ++i) { Asum += A[i]; mean += (double)A[i] * curv[i]; }
+    mean /= std::max(Asum, 1e-12);
+    double var = 0;
+    for (uint32_t i = 0; i < n; ++i) { double d = curv[i] - mean; var += (double)A[i]*d*d; }
+    var /= std::max(Asum, 1e-12);
+    double sd = std::sqrt(std::max(var, 1e-20));
+
+    /* 4. relative density multiplier: high standardized curvature -> denser.
+          GAMMA = aggressiveness; adaptivity in [0,1] scales it (0 => uniform). */
+    const float GAMMA = 6.0f;
+    VectorXf d(n);
+    for (uint32_t i = 0; i < n; ++i) {
+        double cn = (curv[i] - mean) / (2.0 * sd);
+        cn = std::min(1.0, std::max(0.0, cn));
+        d[i] = (float)(1.0 + (double)adaptivity * GAMMA * cn);
+    }
+
+    /* 5. clamp density ratio (<=3x finer / 2x coarser) -> per-vertex edge length:
+          density ~ 1/s^2  =>  s_i = base / sqrt(d_i). */
+    const float DMAX = 9.0f, DMIN = 0.25f;
+    VectorXf S(n);
+    for (uint32_t i = 0; i < n; ++i) {
+        float di = std::min(DMAX, std::max(DMIN, d[i]));
+        S[i] = base_scale / std::sqrt(di);
+    }
+
+    /* 6. GRADATION LIMITING (the load-bearing robustness step). Instant Meshes'
+          position field assumes a locally-consistent lattice; abrupt s jumps make
+          neighbouring lattices incommensurate -> spurious singularities, inflated
+          vertex counts, and degenerate faces that break pure-quad subdivision.
+          Bound the field's local rate of change so neighbours never differ by
+          more than GRAD: repeatedly lower any vertex above GRAD*neighbour (the
+          standard min-propagation sweep) until stable. Only lowers (-> finer). */
+    const float GRAD = 1.5f;
+    for (int it = 0; it < 20; ++it) {
+        bool changed = false;
+        for (uint32_t i = 0; i < n; ++i) {
+            float si = S[i];
+            for (Link *l = adj[i]; l != adj[i+1]; ++l) {
+                if (l->weight == 0) continue;
+                float cap = GRAD * S[l->id];
+                if (si > cap) { si = cap; changed = true; }
+            }
+            S[i] = si;
+        }
+        if (!changed) break;
+    }
+
+    /* 7. conserve the budget with a single global scalar (preserves the
+          gradation-limited ratios): scale s uniformly so the total lattice-cell
+          count Sum(A_i / s_i^2) matches the uniform path's Sum(A_i / base^2).
+          => same total quad count at a fixed -f; density is moved, not added. */
+    double cur = 0;
+    for (uint32_t i = 0; i < n; ++i) cur += (double)A[i] / ((double)S[i]*S[i]);
+    double target = Asum / ((double)base_scale*base_scale);
+    if (cur > 1e-12 && target > 1e-12) {
+        float f = (float) std::sqrt(cur / target);
+        for (uint32_t i = 0; i < n; ++i) S[i] *= f;
+    }
+
+    Float smin = S[0], smax = S[0];
+    for (uint32_t i = 0; i < n; ++i) { smin = std::min(smin, S[i]); smax = std::max(smax, S[i]); }
+    cout << "Adaptive sizing field: edge length in [" << smin << ", " << smax
+         << "] (base " << base_scale << ", ratio " << (smax/smin)
+         << "), curvature mean " << mean << " sd " << sd << endl;
+    return S;
+}

 void batch_process(const std::string &input, const std::string &output,
                    int rosy, int posy, Float scale, int face_count,
                    int vertex_count, Float creaseAngle, bool extrinsic,
                    bool align_to_boundaries, int smooth_iter, int knn_points,
-                   bool pure_quad, bool deterministic) {
+                   bool pure_quad, bool deterministic, float adaptivity) {
     cout << endl;
     cout << "Running in batch mode:" << endl;
     cout << "   Input file             = " << input << endl;
@@ -42,6 +162,7 @@ void batch_process(const std::string &input, const std::string &output,
     cout << "   Align to boundaries    = " << (align_to_boundaries ? "yes" : "no") << endl;
     cout << "   kNN points             = " << knn_points << " (only applies to point clouds)"<< endl;
     cout << "   Fully deterministic    = " << (deterministic ? "yes" : "no") << endl;
+    cout << "   Adaptivity (curvature) = " << (adaptivity > 0 ? std::to_string(adaptivity) : std::string("disabled (uniform)")) << endl;
     if (posy == 4)
         cout << "   Output mode            = " << (pure_quad ? "pure quad mesh" : "quad-dominant mesh") << endl;
     cout << endl;
@@ -126,6 +247,16 @@ void batch_process(const std::string &input, const std::string &output,
         mRes.setE2E(std::move(E2E));
     }

+    /* Adaptive retopo (optional, default off): install a curvature-driven
+       per-vertex sizing field so the quad budget concentrates where curvature
+       is. MUST run before the adjacency/areas are moved into mRes. */
+    if (adaptivity > 0.0f) {
+        cout << "Computing curvature-adaptive sizing field (adaptivity="
+             << adaptivity << ") .." << endl;
+        VectorXf S0 = compute_sizing_field(V, N, A, adj, scale, adaptivity);
+        mRes.setS(std::move(S0));
+    }
+
     /* Build multi-resolution hierarrchy */
     mRes.setAdj(std::move(adj));
     mRes.setF(std::move(F));
diff --git a/src/batch.h b/src/batch.h
--- a/src/batch.h
+++ b/src/batch.h
@@ -19,4 +19,5 @@ extern void batch_process(const std::string &input, const std::string &output,
                           int rosy, int posy, Float scale, int face_count,
                           int vertex_count, Float creaseAngle, bool extrinsic,
                           bool align_to_boundaries, int smooth_iter,
-                          int knn_points, bool dominant, bool deterministic);
+                          int knn_points, bool dominant, bool deterministic,
+                          float adaptivity = 0.0f);
diff --git a/src/extract.cpp b/src/extract.cpp
--- a/src/extract.cpp
+++ b/src/extract.cpp
@@ -37,6 +37,12 @@ extract_graph(const MultiResolutionHierarchy &mRes, bool extrinsic, int rosy, in
               bool snap_vertices) {

     Float scale = mRes.scale(), inv_scale = 1 / scale;
+    /* Adaptive retopo: per-edge lattice spacing from the sizing field decides
+       which vertices share a lattice cell (collapse) -> this is what actually
+       redistributes density. Empty field => uniform (global scale), unchanged. */
+    const bool adaptive = mRes.hasSizingField();
+    const VectorXf &S = adaptive ? mRes.S() : mRes.A(); // S unused when !adaptive
+    VectorXf S_new; // per-extracted-vertex local edge length (adaptive); filled during clustering

     auto compat_orientation = rosy == 2 ? compat_orientation_extrinsic_2 :
         (rosy == 4 ? compat_orientation_extrinsic_4 : compat_orientation_extrinsic_6);
@@ -72,10 +78,12 @@ extract_graph(const MultiResolutionHierarchy &mRes, bool extrinsic, int rosy, in
                             Q.col(i), N.col(i), Q.col(j), N.col(j));

                     Float error = 0;
+                    const Float s_e = adaptive ? 0.5f * (S[i] + S[j]) : scale;
+                    const Float is_e = 1.0f / s_e;
                     std::pair<Vector2i, Vector2i> shift = compat_position(
                             V.col(i), N.col(i), Q_rot.first, O.col(i),
                             V.col(j), N.col(j), Q_rot.second, O.col(j),
-                            scale, inv_scale, &error);
+                            s_e, is_e, &error);

                     Vector2i absDiff = (shift.first-shift.second).cwiseAbs();

@@ -269,6 +277,7 @@ extract_graph(const MultiResolutionHierarchy &mRes, bool extrinsic, int rosy, in
         {
             Eigen::VectorXf cluster_weight(nVertices);
             cluster_weight.setZero();
+            if (adaptive) { S_new.resize(nVertices); S_new.setZero(); }

             tbb::blocked_range<uint32_t> range(0u, V.cols(), GRAIN_SIZE);
             tbb::spin_mutex mutex;
@@ -291,6 +300,8 @@ extract_graph(const MultiResolutionHierarchy &mRes, bool extrinsic, int rosy, in
                         atomicAdd(&N_new.coeffRef(k, j), N(k, i)*weight);
                     }
                     atomicAdd(&cluster_weight[j], weight);
+                    if (adaptive)
+                        atomicAdd(&S_new[j], S[i]*weight);
                 }
             };

@@ -306,6 +317,8 @@ extract_graph(const MultiResolutionHierarchy &mRes, bool extrinsic, int rosy, in
                 }
                 O_new.col(i) /= cluster_weight[i];
                 N_new.col(i).normalize();
+                if (adaptive)
+                    S_new[i] /= cluster_weight[i];
             }

             cout << "done. (took " << timeString(timer.reset()) << ")" << endl;
@@ -345,7 +358,8 @@ extract_graph(const MultiResolutionHierarchy &mRes, bool extrinsic, int rosy, in
                             if (a > std::max(b, c)) {
                                 Float s = 0.5f * (a+b+c);
                                 Float height = 2*std::sqrt(s*(s-a)*(s-b)*(s-c))/a;
-                                if (height < thresh)
+                                Float thr = adaptive ? 0.3f * S_new[i_id] : thresh;
+                                if (height < thr)
                                     candidates.push_back(std::make_tuple(height, i_id, j_id, k_id));
                             }
                         }
@@ -375,8 +389,9 @@ extract_graph(const MultiResolutionHierarchy &mRes, bool extrinsic, int rosy, in
                     Float height = 2*std::sqrt(s*(s-a)*(s-b)*(s-c))/a;
                     if (height != std::get<0>(t))
                         continue;
-                    if ((p_i-p_j).norm() < thresh || (p_i-p_k).norm() < thresh) {
-                        uint32_t merge_id = (p_i-p_j).norm() < thresh ? j : k;
+                    Float thr = adaptive ? 0.3f * S_new[i] : thresh;
+                    if ((p_i-p_j).norm() < thr || (p_i-p_k).norm() < thr) {
+                        uint32_t merge_id = (p_i-p_j).norm() < thr ? j : k;
                         O_new.col(i) = (O_new.col(i) + O_new.col(merge_id)) * 0.5f;
                         N_new.col(i) = (N_new.col(i) + N_new.col(merge_id)) * 0.5f;
                         std::set<uint32_t> adj_updated;
diff --git a/src/field.cpp b/src/field.cpp
--- a/src/field.cpp
+++ b/src/field.cpp
@@ -901,6 +901,12 @@ template <typename CompatFunctor, typename RoundFunctor> static inline Float opt
     const AdjacencyMatrix &adj = mRes.adj(level);
     const MatrixXf &N = mRes.N(level), &Q = mRes.Q(level), &V = mRes.V(level);
     const Float scale = mRes.scale(), inv_scale = 1.0f / scale;
+    /* Adaptive retopo: a per-vertex sizing field replaces the global lattice
+       spacing. The lattice-math functors already parameterize on (scale,
+       inv_scale), so we just feed them per-vertex / per-edge values. Empty
+       field => the original uniform path, bit-for-bit. */
+    const bool adaptive = mRes.hasSizingField();
+    const VectorXf &S = adaptive ? mRes.S(level) : mRes.A(level); // S unused when !adaptive
     const std::vector<uint32_t> *phase = nullptr;
     const MatrixXf &CQ = mRes.CQ(level);
     const MatrixXf &CO = mRes.CO(level);
@@ -912,6 +918,7 @@ template <typename CompatFunctor, typename RoundFunctor> static inline Float opt
             const uint32_t i = (*phase)[phaseIdx];
             const Vector3f n_i = N.col(i), v_i = V.col(i);
             Vector3f q_i = Q.col(i);
+            const Float s_i = adaptive ? S[i] : scale, is_i = 1.0f / s_i;

             Vector3f sum = O.col(i);
             Float weight_sum = 0.0f;
@@ -928,13 +935,15 @@ template <typename CompatFunctor, typename RoundFunctor> static inline Float opt

                 const Vector3f n_j = N.col(j), v_j = V.col(j);
                 Vector3f q_j = Q.col(j), o_j = O.col(j);
+                const Float s_e = adaptive ? 0.5f * (s_i + S[j]) : scale;
+                const Float is_e = 1.0f / s_e;

                 #if 1
                     q_j.normalize();
                 #endif

                 std::pair<Vector3f, Vector3f> value = compat_functor(
-                    v_i, n_i, q_i, sum, v_j, n_j, q_j, o_j, scale, inv_scale);
+                    v_i, n_i, q_i, sum, v_j, n_j, q_j, o_j, s_e, is_e);

                 sum = value.first*weight_sum + value.second*weight;
                 weight_sum += weight;
@@ -955,7 +964,7 @@ template <typename CompatFunctor, typename RoundFunctor> static inline Float opt
             }

             if (weight_sum > 0)
-                O.col(i) = round_functor(sum, q_i, n_i, v_i, scale, inv_scale);
+                O.col(i) = round_functor(sum, q_i, n_i, v_i, s_i, is_i);
         }
     };

@@ -985,8 +994,9 @@ template <typename CompatFunctor, typename RoundFunctor> static inline Float opt
                     q_j.normalize();
                 #endif
                 const Vector3f t_j = n_j.cross(q_j);
+                const Float s_e = adaptive ? 0.5f * (S[i] + S[j]) : scale;

-                sum += o_j + scale * (
+                sum += o_j + s_e * (
                       q_j * link->ivar[1].translate_u
                     + t_j * link->ivar[1].translate_v
                     - q_i * link->ivar[0].translate_u
diff --git a/src/hierarchy.cpp b/src/hierarchy.cpp
--- a/src/hierarchy.cpp
+++ b/src/hierarchy.cpp
@@ -393,6 +393,7 @@ MultiResolutionHierarchy::MultiResolutionHierarchy() {
     if (sizeof(Link) != 12)
         throw std::runtime_error("Adjacency matrix entries are not packed! Investigate compiler settings.");
     mA.reserve(MAX_DEPTH+1);
+    mS.reserve(MAX_DEPTH+1);
     mV.reserve(MAX_DEPTH+1);
     mN.reserve(MAX_DEPTH+1);
     mQ.reserve(MAX_DEPTH+1);
@@ -443,6 +444,26 @@ void MultiResolutionHierarchy::build(bool deterministic, const ProgressCallback
         else
             generate_graph_coloring(adj_p, V_p.cols(), phases_p, progress);

+        /* Downsample the per-vertex sizing field (adaptive retopo) alongside the
+           geometry: each coarse vertex inherits the area-weighted mean of its
+           (<=2) children's local target edge length. Only when a field is set. */
+        if (!mS.empty()) {
+            const VectorXf &S_prev = mS[i];
+            const VectorXf &A_prev = mA[i];
+            VectorXf S_p(V_p.cols());
+            for (uint32_t k = 0; k < (uint32_t) V_p.cols(); ++k) {
+                uint32_t a = toUpper(0, k), b = toUpper(1, k);
+                if (b == INVALID) {
+                    S_p[k] = S_prev[a];
+                } else {
+                    Float wa = A_prev[a], wb = A_prev[b], w = wa + wb;
+                    S_p[k] = (w > 0) ? (wa * S_prev[a] + wb * S_prev[b]) / w
+                                     : Float(0.5) * (S_prev[a] + S_prev[b]);
+                }
+            }
+            mS.push_back(std::move(S_p));
+        }
+
         mTotalSize += V_p.cols();
         mPhases.push_back(std::move(phases_p));
         mAdj.push_back(std::move(adj_p));
diff --git a/src/hierarchy.h b/src/hierarchy.h
--- a/src/hierarchy.h
+++ b/src/hierarchy.h
@@ -49,6 +49,10 @@ public:
     inline const MatrixXf &V(int level = 0) const { return mV[level]; }
     inline const MatrixXf &N(int level = 0) const { return mN[level]; }
     inline const VectorXf &A(int level = 0) const { return mA[level]; }
+    /* Per-vertex sizing field (local target edge length). Empty unless adaptive
+       retopo was requested; when present, has one entry per hierarchy level. */
+    inline const VectorXf &S(int level = 0) const { return mS[level]; }
+    inline bool hasSizingField() const { return !mS.empty() && mS[0].size() > 0; }
     inline const MatrixXu &toUpper(int level) const { return mToUpper[level]; }
     inline const VectorXu &toLower(int level) const { return mToLower[level]; }
     inline const MatrixXf &Q(int level = 0) const { return mQ[level]; }
@@ -71,6 +75,9 @@ public:
     inline void setV(MatrixXf &&V) { mV.clear(); mV.push_back(std::move(V)); }
     inline void setN(MatrixXf &&N) { mN.clear(); mN.push_back(std::move(N)); }
     inline void setA(MatrixXf &&A) { mA.clear(); mA.push_back(std::move(A)); }
+    /* Install the level-0 sizing field (per-vertex target edge length). build()
+       downsamples it (area-weighted) to all coarser levels. */
+    inline void setS(VectorXf &&S) { mS.clear(); mS.push_back(std::move(S)); }
     inline void setAdj(AdjacencyMatrix &&adj) { mAdj.clear(); mAdj.push_back(std::move(adj)); }

     inline uint32_t size(int level = 0) const { return mV[level].cols(); }
@@ -114,6 +121,7 @@ public:
     std::vector<MatrixXf> mV;
     std::vector<MatrixXf> mN;
     std::vector<VectorXf> mA;
+    std::vector<VectorXf> mS;   // per-level per-vertex sizing field (adaptive retopo; empty = uniform)
     std::vector<VectorXu> mToLower;
     std::vector<MatrixXu> mToUpper;
     std::vector<MatrixXf> mO;
IM_ADAPTIVE_PATCH_EOF

if git -C "$IM" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo ">> adaptive sizing-field patch already applied"
elif git -C "$IM" apply --check "$PATCH" >/dev/null 2>&1; then
  echo ">> applying adaptive sizing-field patch"
  git -C "$IM" apply "$PATCH"
else
  echo "!! adaptive patch does not apply cleanly (upstream drift?) — building UNIFORM-only" >&2
fi

# 3) compile the GUI-free core + headless main. g++14 promotes -Wchanges-meaning to an error inside
#    TBB's own task.h → -Wno-changes-meaning; -w silences the deprecation spam.
mkdir -p "$IM/build_headless"
CXX="${CXX:-g++}"
INC="-I $IM/src -I $IM/ext/pcg32 -I $IM/ext/half -I $IM/ext/dset -I $IM/ext/pss -I $IM/ext/rply \
     -I $DEPS_ENV/include -I $DEPS_ENV/include/eigen3"
CXXFLAGS="-O3 -std=gnu++17 -fopenmp -w -fpermissive -Wno-changes-meaning $INC"
CORE="meshio normal adjacency meshstats hierarchy extract field bvh subdivide reorder batch \
      smoothcurve cleanup dedge serializer"

echo ">> compiling rply.c"
gcc -O2 -c "$IM/ext/rply/rply.c" -o "$IM/build_headless/rply.o"
OBJS="$IM/build_headless/rply.o"
for f in $CORE; do
  echo ">> compiling $f.cpp"
  $CXX $CXXFLAGS -c "$IM/src/$f.cpp" -o "$IM/build_headless/$f.o"
  OBJS="$OBJS $IM/build_headless/$f.o"
done
echo ">> compiling im_batch_main.cpp + link"
$CXX $CXXFLAGS -c "$HERE/im_batch_main.cpp" -o "$IM/build_headless/im_batch_main.o"
OBJS="$OBJS $IM/build_headless/im_batch_main.o"
$CXX -fopenmp -o "$BIN" $OBJS -L "$DEPS_ENV/lib" -ltbb -Wl,-rpath,"$DEPS_ENV/lib"
echo ">> built $BIN"
