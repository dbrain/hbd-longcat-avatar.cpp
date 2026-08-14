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
//       i.e. random-choice N->(M*4), then FPS (M*4)->M from index 0.  The fixed-seed path below
//       is an exact small port of NumPy 1.26's PCG64 + Floyd choice implementation, rather than a
//       statistically-similar std::mt19937 substitute.  This is material: these points are learned
//       encoder queries, so changing them changes the generated skeleton.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include "glb_reader.hpp"

namespace rig {

// NumPy's `np.random.default_rng(0)` state after SeedSequence initialization.
// The production SkinTokens capture fixes this seed to zero.  Keeping this narrowly-scoped avoids
// silently claiming NumPy SeedSequence parity for arbitrary seeds while providing exact parity for
// the only sequence used by the rig path.  PCG64 is XSL-RR 128/64; next32 has NumPy's low-half then
// cached high-half ordering (numpy/random/src/pcg64/pcg64.h, v1.26.4).
class NumpyPcg64Seed0 {
public:
    NumpyPcg64Seed0()
        : state_(((__uint128_t)0x1aa1b5345996452dULL << 64) | 0x09585eb7a69561e3ULL),
          inc_(((__uint128_t)0x418ddadb3af71a82ULL << 64) | 0x588133bc447873a9ULL) {}

    uint64_t next64() {
        static const __uint128_t mult = ((__uint128_t)2549297995355413924ULL << 64) |
                                        4865540595714422341ULL;
        state_ = state_ * mult + inc_;
        const uint64_t hi = (uint64_t)(state_ >> 64);
        const uint64_t lo = (uint64_t)state_;
        const unsigned rot = (unsigned)(hi >> 58);
        const uint64_t x = hi ^ lo;
        return (x >> rot) | (x << ((-rot) & 63));
    }

    uint32_t next32() {
        if (has_uint32_) { has_uint32_ = false; return uinteger_; }
        const uint64_t v = next64();
        has_uint32_ = true;
        uinteger_ = (uint32_t)(v >> 32);
        return (uint32_t)v;
    }

    // Equivalent to NumPy's random_bounded_uint64(bitgen, 0, inclusive_max, 0, 0)
    // for the int64 ranges used by Generator.choice here (all <= uint32 max).
    uint64_t bounded_inclusive(uint64_t inclusive_max) {
        if (inclusive_max == 0) return 0;
        if (inclusive_max == 0xffffffffULL) return next32();
        const uint32_t range = (uint32_t)inclusive_max;
        const uint32_t range_excl = range + 1;
        uint64_t product = (uint64_t)next32() * range_excl;
        uint32_t leftover = (uint32_t)product;
        if (leftover < range_excl) {
            const uint32_t threshold = (UINT32_MAX - range) % range_excl;
            while (leftover < threshold) {
                product = (uint64_t)next32() * range_excl;
                leftover = (uint32_t)product;
            }
        }
        return product >> 32;
    }

private:
    __uint128_t state_, inc_;
    bool has_uint32_ = false;
    uint32_t uinteger_ = 0;
};

// Exact `np.random.default_rng(0).choice(population, count, replace=False)` for the range used by
// SkinTokens. NumPy 1.26 chooses Floyd's algorithm when population <= 10000, then shuffles output.
inline std::vector<int> numpy_choice_seed0_without_replacement(int population, int count) {
    std::vector<int> out;
    // Generator.choice changes to NumPy's tail-shuffle heuristic above this threshold.  The
    // SkinTokens contract is exactly 8192 -> 2048, so fail closed rather than return a subtly
    // non-NumPy sequence for an unsupported caller.
    if (population < 0 || population > 10000 || count < 0 || count > population) return out;
    out.resize((size_t)count);
    if (count == 0) return out;
    NumpyPcg64Seed0 rng;
    uint64_t target = (uint64_t)(1.2 * (double)count); // exact C cast used by NumPy's Cython code
    uint64_t mask = 1;
    while (mask < target) mask = (mask << 1) | 1;       // _gen_mask: all ones above target
    std::vector<uint64_t> hash_set((size_t)mask + 1, UINT64_MAX);
    for (int j = population - count; j < population; ++j) {
        const uint64_t value = rng.bounded_inclusive((uint64_t)j);
        uint64_t loc = value & mask;
        while (hash_set[(size_t)loc] != UINT64_MAX && hash_set[(size_t)loc] != value)
            loc = (loc + 1) & mask;
        if (hash_set[(size_t)loc] == UINT64_MAX) {
            hash_set[(size_t)loc] = value;
            out[(size_t)(j - population + count)] = (int)value;
        } else {
            loc = (uint64_t)j & mask;
            while (hash_set[(size_t)loc] != UINT64_MAX) loc = (loc + 1) & mask;
            hash_set[(size_t)loc] = (uint64_t)j;
            out[(size_t)(j - population + count)] = j;
        }
    }
    // NumPy _shuffle_int(bitgen, count, 1, data): descending Fisher-Yates with the same bounded draw.
    for (int i = count - 1; i >= 1; --i) std::swap(out[(size_t)i], out[(size_t)rng.bounded_inclusive((uint64_t)i)]);
    return out;
}

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
    std::vector<float> vertices;      // N*3  (area-weighted surface samples)
    std::vector<float> normals;       // N*3  (per-sample face normals)
    std::vector<float> sampled_pc;    // M*3  (R1 query: choice N->4M then FPS 4M->M from idx 0)
    std::vector<float> sampled_feats; // M*3  (the per-sample normals AT the query points; R1 feats)
    int N = 0;
    int M = 0;
    bool ok = false;
};

// The SkeletonTokens encoder has no semantic distinction between an arm and a
// long ponytail/cape/weapon.  These are deliberately *candidate* rig guides,
// not geometry edits: the full mesh is still retained for the final skin
// transfer, and callers must use the anatomy gate before accepting a result.
// Coordinates are canonicalized by the image-to-rig path (Y up, X lateral).
enum class RigGuideProxy { Full, CentralCore, SideBand };

inline const char* rig_guide_proxy_name(RigGuideProxy p) {
    switch (p) {
        case RigGuideProxy::Full:        return "full";
        case RigGuideProxy::CentralCore: return "central-core";
        case RigGuideProxy::SideBand:    return "side-band";
    }
    return "unknown";
}

inline bool parse_rig_guide_proxy(const std::string& text, RigGuideProxy& out) {
    if (text == "full") { out = RigGuideProxy::Full; return true; }
    if (text == "central-core") { out = RigGuideProxy::CentralCore; return true; }
    if (text == "side-band") { out = RigGuideProxy::SideBand; return true; }
    return false;
}

inline bool keep_rig_guide_point(const float* p, RigGuideProxy proxy) {
    const float x = std::fabs(p[0]);
    if (proxy == RigGuideProxy::Full) return true;
    // A central body guide removes long lateral accessories entirely.  It is
    // intentionally conservative: legs and torso remain, while the full-guide
    // candidate still covers wide poses and non-humanoids.
    if (proxy == RigGuideProxy::CentralCore) return x <= 0.50f;
    // Keep the near-body region everywhere.  Far-lateral samples survive only
    // in the shoulder/hip band where a normal A/T-pose arm belongs; this breaks
    // the long top-to-bottom traces that side hair, capes and weapons otherwise
    // present to a point-cloud skeleton model as giant limbs.
    return x <= 0.42f || (p[1] >= -0.32f && p[1] <= 0.32f);
}

// In-memory variant: same logic as prep_mesh_for_rig but on verts/faces already in RAM (the inline
// image->rig API feeds the freshly-generated mesh straight in, no GLB round-trip). `verts` is copied
// (it is normalized in place); `faces` is read-only.
inline PrepResult prep_mesh_for_rig_inmem(std::vector<float> verts, const std::vector<int64_t>& faces,
                                          int N = 8192, int M = 512, uint64_t seed = 0,
                                          RigGuideProxy proxy = RigGuideProxy::Full) {
    PrepResult R;
    if (verts.empty() || faces.empty()) {
        std::fprintf(stderr, "prep_mesh_for_rig_inmem: empty mesh (V=%zu F=%zu)\n",
                     verts.size() / 3, faces.size() / 3);
        return R;
    }
    normalize_mesh(verts);
    // Rejection sample the guide instead of deleting vertices from the delivery
    // mesh.  This preserves area-weighted surface sampling conditional on the
    // proxy predicate and leaves the exact original mesh available for texture
    // and skin transfer.
    const int oversample = proxy == RigGuideProxy::Full ? N : N * 4;
    std::vector<float> sampled_pts, sampled_nrm;
    if (!sample_surface(verts, faces, oversample, seed, sampled_pts, sampled_nrm)) {
        std::fprintf(stderr, "prep_mesh_for_rig: sample_surface failed\n");
        return R;
    }
    R.vertices.clear(); R.normals.clear();
    R.vertices.reserve((size_t)N * 3); R.normals.reserve((size_t)N * 3);
    for (int i = 0; i < oversample && (int)(R.vertices.size() / 3) < N; ++i) {
        const float* p = &sampled_pts[(size_t)i * 3];
        if (!keep_rig_guide_point(p, proxy)) continue;
        R.vertices.insert(R.vertices.end(), p, p + 3);
        const float* n = &sampled_nrm[(size_t)i * 3];
        R.normals.insert(R.normals.end(), n, n + 3);
    }
    if ((int)(R.vertices.size() / 3) != N) {
        std::fprintf(stderr, "prep_mesh_for_rig: proxy '%s' retained only %zu/%d samples\n",
                     rig_guide_proxy_name(proxy), R.vertices.size() / 3, N);
        return R;
    }
    R.N = N;
    R.M = M;

    // --- R1 query selection, mirroring capture_vecset_r0.py exactly ---
    // ind = default_rng(0).choice(N, M*4, replace = M*4 > N), then FPS from index zero.
    int K = M * 4;
    std::vector<float> pre;          // K*3 candidate subset
    std::vector<int>   pre_idx(K);   // original surface-sample index of each candidate (-> normals)
    if (K <= N) {
        // seed 0 is the SHIPPED path: the exact NumPy PCG64 choice, bit-for-bit with
        // capture_vecset_r0.py. A non-zero seed is only ever used to draw an INDEPENDENT
        // conditioning cloud from the same mesh (the rig retry loop) — there is no Python
        // reference to match there, so a plain shuffle is the right generator. This used to
        // hard-fail, which made "sample the same mesh again" impossible.
        std::vector<int> perm;
        if (seed == 0) {
            perm = numpy_choice_seed0_without_replacement(N, K);
        } else {
            perm.resize(N);
            for (int i = 0; i < N; ++i) perm[i] = i;
            std::mt19937_64 qrng(seed * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull);
            for (int i = N - 1; i >= 1; --i) {
                std::uniform_int_distribution<int> D(0, i);
                std::swap(perm[(size_t)i], perm[(size_t)D(qrng)]);
            }
            perm.resize(K);
        }
        if ((int)perm.size() != K) return R;
        pre.resize((size_t)K * 3);
        for (int i = 0; i < K; ++i) {
            pre_idx[i] = perm[i];
            for (int k = 0; k < 3; ++k)
                pre[(size_t)i * 3 + k] = R.vertices[(size_t)perm[i] * 3 + k];
        }
    } else {
        // with replacement (K > N): pad by sampling with replacement
        pre.resize((size_t)K * 3);
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> D(0, N - 1);
        for (int i = 0; i < K; ++i) {
            int j = D(rng);
            pre_idx[i] = j;
            for (int k = 0; k < 3; ++k) pre[(size_t)i * 3 + k] = R.vertices[(size_t)j * 3 + k];
        }
    }
    // fps(pre, ratio=1/4) -> K/4 == M points, start index 0.
    std::vector<int> fidx = fps(pre, M, 0);
    gather(pre, fidx, R.sampled_pc);
    // sampled_feats = the per-sample NORMALS at exactly those query points (R1 VecSet `sampled_feats`),
    // gathered from THIS mesh — not borrowed from a banked giraffe.
    R.sampled_feats.assign(fidx.size() * 3, 0.f);
    for (size_t i = 0; i < fidx.size(); ++i) {
        int orig = pre_idx[(size_t)fidx[i]];
        for (int k = 0; k < 3; ++k)
            R.sampled_feats[i * 3 + k] = R.normals[(size_t)orig * 3 + k];
    }
    R.M = (int)fidx.size();
    R.ok = true;
    return R;
}

inline PrepResult prep_mesh_for_rig(const char* glb_path, int N = 8192, int M = 512, uint64_t seed = 0,
                                    RigGuideProxy proxy = RigGuideProxy::Full) {
    glb::Mesh mesh;
    if (!glb::read_glb(glb_path, mesh)) {
        std::fprintf(stderr, "prep_mesh_for_rig: read_glb failed for '%s'\n", glb_path);
        return PrepResult{};
    }
    return prep_mesh_for_rig_inmem(std::move(mesh.verts), mesh.faces, N, M, seed, proxy);
}

}  // namespace rig
