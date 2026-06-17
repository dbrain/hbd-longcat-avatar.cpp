// mesh_sample.hpp — mesh PREPROCESSING for the auto-rig pipeline (PURE CPU, header-only).
//
// Ports the host front-end from capture_skintokens_e2e.py / capture_vecset_r0.py so an ARBITRARY
// mesh can be rigged (not just the banked giraffe). Three stages:
//
//   (1) normalize_mesh: center on bbox-midpoint, scale so |v-c|max == 1  ->  fits in [-1,1]^3.
//       Matches:  c=(v.max(0)+v.min(0))/2 ; mesh.vertices = (v-c)/abs(v-c).max()
//
//   (2) sample_surface: AREA-WEIGHTED random surface sampling of N points + per-point normals.
//       Matches:  pts,fid = trimesh.sample.sample_surface(mesh, N) ; nrm = mesh.face_normals[fid]
//       - face chosen with prob proportional to triangle area (cumulative-area + binary search),
//       - uniform barycentric point via the sqrt trick (u=1-sqrt(r1), v=sqrt(r1)*r2),
//       - out_normal = that face's GEOMETRIC (cross-product) unit normal  == trimesh face_normals.
//       NOTE: bit-matching numpy's RNG is NOT a goal; this is a correct area-weighted sampler.
//
//   (3) fps: greedy farthest-point sampling N->M from a fixed start index (deterministic).
//       Matches src/model/utils.py::fps(..., random_start=False) -> farthest starts at 0.
//
//   (4) prep_mesh_for_rig: load GLB (glb_reader) -> normalize -> sample_surface(N) -> derive the
//       R1 query sampled_pc[M]. The query selection EXACTLY mirrors capture_vecset_r0.py:
//          rng = np.random.default_rng(seed=0)
//          ind = rng.choice(N, M*4, replace = M*4 > N)        # 8192 -> 2048, replace=False
//          fps(pc[ind], ratio=1/4, random_start=False)        # 2048 -> 512, start index 0
//       i.e. random-choice N->(M*4), then FPS (M*4)->M from index 0. We replicate the SHAPE of this
//       op (choose 4M, FPS down to M). The np.random.default_rng(seed=0) choice is replaced by a
//       std::mt19937 partial Fisher-Yates draw of 4M distinct indices — so the resulting query set
//       (and therefore the rig) will NOT bit-match the python banked giraffe, but is a valid query.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include "glb_reader.hpp"

namespace rig {

// ---------------------------------------------------------------------------
// (1) normalize_mesh — in place. verts is V*3 (xyz interleaved).
// c=(max+min)/2 per axis; scale = 1 / max_over_all(|v-c|); v = (v-c)*scale.
// ---------------------------------------------------------------------------
inline void normalize_mesh(std::vector<float>& verts) {
    const size_t V = verts.size() / 3;
    if (V == 0) return;
    double mn[3] = { 1e300, 1e300, 1e300 };
    double mx[3] = { -1e300, -1e300, -1e300 };
    for (size_t i = 0; i < V; ++i)
        for (int c = 0; c < 3; ++c) {
            double x = verts[i * 3 + c];
            if (x < mn[c]) mn[c] = x;
            if (x > mx[c]) mx[c] = x;
        }
    double c[3] = { (mx[0] + mn[0]) / 2, (mx[1] + mn[1]) / 2, (mx[2] + mn[2]) / 2 };
    double amax = 0.0;
    for (size_t i = 0; i < V; ++i)
        for (int k = 0; k < 3; ++k) {
            double a = std::fabs((double)verts[i * 3 + k] - c[k]);
            if (a > amax) amax = a;
        }
    double scale = amax > 0 ? 1.0 / amax : 1.0;
    for (size_t i = 0; i < V; ++i)
        for (int k = 0; k < 3; ++k)
            verts[i * 3 + k] = (float)(((double)verts[i * 3 + k] - c[k]) * scale);
}

// Geometric unit normal of a triangle (a,b,c) via cross product; matches trimesh face_normals.
// Degenerate (zero-area) faces -> (0,0,0) (trimesh leaves them ~nan but those faces have ~0 area
// so are never sampled in practice).
inline void tri_normal(const float* a, const float* b, const float* c, float out[3]) {
    double e1[3] = { (double)b[0] - a[0], (double)b[1] - a[1], (double)b[2] - a[2] };
    double e2[3] = { (double)c[0] - a[0], (double)c[1] - a[1], (double)c[2] - a[2] };
    double n[3] = {
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0]
    };
    double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len > 1e-20) { out[0] = (float)(n[0] / len); out[1] = (float)(n[1] / len); out[2] = (float)(n[2] / len); }
    else { out[0] = out[1] = out[2] = 0.f; }
}

// Twice the triangle area = |cross(b-a, c-a)| (we only need it as a relative weight, so the /2 is
// irrelevant for the cumulative distribution).
inline double tri_double_area(const float* a, const float* b, const float* c) {
    double e1[3] = { (double)b[0] - a[0], (double)b[1] - a[1], (double)b[2] - a[2] };
    double e2[3] = { (double)c[0] - a[0], (double)c[1] - a[1], (double)c[2] - a[2] };
    double n[3] = {
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0]
    };
    return std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

// ---------------------------------------------------------------------------
// (2) sample_surface — area-weighted N surface points + face-normals.
// verts: V*3, faces: F*3 (int64). out_pts/out_normals must hold N*3 floats.
// ---------------------------------------------------------------------------
inline bool sample_surface(const std::vector<float>& verts, const std::vector<int64_t>& faces,
                           int N, uint64_t seed, std::vector<float>& out_pts,
                           std::vector<float>& out_normals) {
    const size_t F = faces.size() / 3;
    const size_t V = verts.size() / 3;
    if (F == 0 || V == 0 || N <= 0) return false;
    out_pts.assign((size_t)N * 3, 0.f);
    out_normals.assign((size_t)N * 3, 0.f);

    // cumulative area distribution over faces
    std::vector<double> cum(F);
    double total = 0.0;
    for (size_t fi = 0; fi < F; ++fi) {
        int64_t i0 = faces[fi * 3 + 0], i1 = faces[fi * 3 + 1], i2 = faces[fi * 3 + 2];
        double da = 0.0;
        if (i0 >= 0 && i1 >= 0 && i2 >= 0 && (size_t)i0 < V && (size_t)i1 < V && (size_t)i2 < V)
            da = tri_double_area(&verts[i0 * 3], &verts[i1 * 3], &verts[i2 * 3]);
        total += da;
        cum[fi] = total;
    }
    if (total <= 0.0) return false;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int s = 0; s < N; ++s) {
        // pick a face with prob proportional to area
        double r = U(rng) * total;
        size_t fi = (size_t)(std::lower_bound(cum.begin(), cum.end(), r) - cum.begin());
        if (fi >= F) fi = F - 1;
        int64_t i0 = faces[fi * 3 + 0], i1 = faces[fi * 3 + 1], i2 = faces[fi * 3 + 2];
        const float* a = &verts[i0 * 3];
        const float* b = &verts[i1 * 3];
        const float* c = &verts[i2 * 3];
        // uniform barycentric point (sqrt trick)
        double r1 = U(rng), r2 = U(rng);
        double sr1 = std::sqrt(r1);
        double w0 = 1.0 - sr1;           // weight of a
        double w1 = sr1 * (1.0 - r2);    // weight of b
        double w2 = sr1 * r2;            // weight of c
        for (int k = 0; k < 3; ++k)
            out_pts[(size_t)s * 3 + k] = (float)(w0 * a[k] + w1 * b[k] + w2 * c[k]);
        float nn[3]; tri_normal(a, b, c, nn);
        out_normals[(size_t)s * 3 + 0] = nn[0];
        out_normals[(size_t)s * 3 + 1] = nn[1];
        out_normals[(size_t)s * 3 + 2] = nn[2];
    }
    return true;
}

// ---------------------------------------------------------------------------
// (3) fps — greedy farthest-point sampling. pts is Npts*3; returns M indices into pts.
// Deterministic given start_idx (matches utils.py fps with random_start=False -> start 0).
// ---------------------------------------------------------------------------
inline std::vector<int> fps(const std::vector<float>& pts, int M, int start_idx) {
    const int Npts = (int)(pts.size() / 3);
    std::vector<int> out;
    if (Npts <= 0 || M <= 0) return out;
    if (M > Npts) M = Npts;
    out.reserve(M);
    std::vector<double> dist((size_t)Npts, std::numeric_limits<double>::infinity());
    int farthest = std::max(0, std::min(start_idx, Npts - 1));
    for (int i = 0; i < M; ++i) {
        out.push_back(farthest);
        const float* cpt = &pts[(size_t)farthest * 3];
        double best = -1.0; int besti = farthest;
        for (int p = 0; p < Npts; ++p) {
            const float* q = &pts[(size_t)p * 3];
            double dx = (double)q[0] - cpt[0], dy = (double)q[1] - cpt[1], dz = (double)q[2] - cpt[2];
            double d = dx * dx + dy * dy + dz * dz;
            if (d < dist[p]) dist[p] = d;
            if (dist[p] > best) { best = dist[p]; besti = p; }
        }
        farthest = besti;
    }
    return out;
}

// Gather M selected points from pts (Npts*3) into out (M*3).
inline void gather(const std::vector<float>& pts, const std::vector<int>& idx, std::vector<float>& out) {
    out.assign(idx.size() * 3, 0.f);
    for (size_t i = 0; i < idx.size(); ++i)
        for (int k = 0; k < 3; ++k)
            out[i * 3 + k] = pts[(size_t)idx[i] * 3 + k];
}

// ---------------------------------------------------------------------------
// (4) prep_mesh_for_rig — full host front-end for a GLB.
// ---------------------------------------------------------------------------
struct PrepResult {
    std::vector<float> vertices;    // N*3  (area-weighted surface samples)
    std::vector<float> normals;     // N*3  (per-sample face normals)
    std::vector<float> sampled_pc;  // M*3  (R1 query: choice N->4M then FPS 4M->M from idx 0)
    int N = 0;
    int M = 0;
    bool ok = false;
};

inline PrepResult prep_mesh_for_rig(const char* glb_path, int N = 8192, int M = 512, uint64_t seed = 0) {
    PrepResult R;
    glb::Mesh mesh;
    if (!glb::read_glb(glb_path, mesh)) {
        std::fprintf(stderr, "prep_mesh_for_rig: read_glb failed for '%s'\n", glb_path);
        return R;
    }
    if (mesh.verts.empty() || mesh.faces.empty()) {
        std::fprintf(stderr, "prep_mesh_for_rig: empty mesh (V=%zu F=%zu)\n",
                     mesh.verts.size() / 3, mesh.faces.size() / 3);
        return R;
    }
    normalize_mesh(mesh.verts);
    if (!sample_surface(mesh.verts, mesh.faces, N, seed, R.vertices, R.normals)) {
        std::fprintf(stderr, "prep_mesh_for_rig: sample_surface failed\n");
        return R;
    }
    R.N = N;
    R.M = M;

    // --- R1 query selection, mirroring capture_vecset_r0.py exactly in SHAPE ---
    // ind = choice(N, M*4, replace = M*4 > N)  -> here 8192 -> 2048 distinct (replace=False).
    int K = M * 4;
    std::vector<float> pre;          // K*3 candidate subset
    std::mt19937_64 rng(seed);
    if (K <= N) {
        // distinct draw of K indices (partial Fisher-Yates over 0..N-1)
        std::vector<int> perm(N);
        for (int i = 0; i < N; ++i) perm[i] = i;
        for (int i = 0; i < K; ++i) {
            std::uniform_int_distribution<int> D(i, N - 1);
            int j = D(rng);
            std::swap(perm[i], perm[j]);
        }
        pre.resize((size_t)K * 3);
        for (int i = 0; i < K; ++i)
            for (int k = 0; k < 3; ++k)
                pre[(size_t)i * 3 + k] = R.vertices[(size_t)perm[i] * 3 + k];
    } else {
        // with replacement (K > N): pad by sampling with replacement
        pre.resize((size_t)K * 3);
        std::uniform_int_distribution<int> D(0, N - 1);
        for (int i = 0; i < K; ++i) {
            int j = D(rng);
            for (int k = 0; k < 3; ++k) pre[(size_t)i * 3 + k] = R.vertices[(size_t)j * 3 + k];
        }
    }
    // fps(pre, ratio=1/4) -> K/4 == M points, start index 0.
    std::vector<int> fidx = fps(pre, M, 0);
    gather(pre, fidx, R.sampled_pc);
    R.M = (int)fidx.size();
    R.ok = true;
    return R;
}

}  // namespace rig
