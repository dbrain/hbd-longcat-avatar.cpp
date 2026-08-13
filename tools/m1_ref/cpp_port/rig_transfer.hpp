// R7 — skin-weight TRANSFER from a SAMPLED point set onto a high-res target mesh.
//
// The auto-rigger computes a skeleton (joints[J,3] + parents[J]) and per-vertex skin
// weights for a SMALL sampled point set (~8192 pts). To rig the actual million-vertex
// avatar mesh (from pixal3d) we must transfer those skin-weight rows onto every target
// vertex. This is the last glue step before a usable rigged GLB.
//
// Method: for each target (dst) vertex, find its k nearest sampled (src) points
// (Euclidean), inverse-distance-weighted blend (w_i = 1/(d_i+eps)) of their skin-weight
// rows -> the dst skin-weight row. The full [Nd,J] blended rows are produced verbatim;
// the downstream GLB writer (glb_rigged.hpp rig_topk4) selects top-4 + renormalizes.
//
// Acceleration: a uniform voxel grid over the (small) src point set + an expanding-ring
// (shell-by-shell) cell search per dst vertex, so the 8192 x 1.1M query is fast on CPU.
// The per-dst loop is OpenMP-parallelized. Header-only, namespace rig, no ggml/CUDA.
//
// Frame-mismatch defence: NN transfer is scale-relative within each cloud, so it still
// "works" if the src/dst bounding boxes differ — but a wildly different frame smears the
// result. transfer_skin reports both bboxes (via the out-params) so the caller can flag a
// mismatch; the grid is sized from the SRC bbox but the ring search expands unboundedly,
// so an off-frame dst vertex still resolves to its true nearest src points (just slowly /
// meaninglessly). We also expose bbox extents so the CLI can warn.
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <limits>

namespace rig {

struct BBox {
    float mn[3] = { std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max() };
    float mx[3] = { -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max() };
    void add(const float* p) {
        for (int d = 0; d < 3; d++) { if (p[d] < mn[d]) mn[d] = p[d]; if (p[d] > mx[d]) mx[d] = p[d]; }
    }
    float ext(int d) const { return mx[d] - mn[d]; }
    float diag() const {
        float s = 0; for (int d = 0; d < 3; d++) { float e = ext(d); s += e * e; } return std::sqrt(s);
    }
};

inline BBox bbox_of(const std::vector<float>& pts) {
    BBox b;
    size_t n = pts.size() / 3;
    for (size_t i = 0; i < n; i++) b.add(&pts[i * 3]);
    return b;
}

// Uniform voxel grid over a point set, for nearest-k queries via an expanding cell ring.
struct VoxelGrid {
    BBox  bb;
    float origin[3] = {0, 0, 0};
    float cell = 1.f;          // cell size (cube)
    int   dim[3]   = {1, 1, 1};
    std::vector<int>            cell_start;  // size = ncells+1 (CSR-style)
    std::vector<int>            point_idx;   // size = Ns, point indices grouped by cell
    const std::vector<float>*   pts = nullptr;

    int ncells() const { return dim[0] * dim[1] * dim[2]; }
    inline int clampi(int v, int lo, int hi) const { return v < lo ? lo : (v > hi ? hi : v); }

    inline void cell_of(const float* p, int& cx, int& cy, int& cz) const {
        cx = clampi((int)std::floor((p[0] - origin[0]) / cell), 0, dim[0] - 1);
        cy = clampi((int)std::floor((p[1] - origin[1]) / cell), 0, dim[1] - 1);
        cz = clampi((int)std::floor((p[2] - origin[2]) / cell), 0, dim[2] - 1);
    }
    inline int lin(int cx, int cy, int cz) const { return (cz * dim[1] + cy) * dim[0] + cx; }

    // Build a grid with ~target_per_cell points per occupied cell on average.
    void build(const std::vector<float>& src_pts, int target_per_cell = 1) {
        pts = &src_pts;
        size_t Ns = src_pts.size() / 3;
        bb = bbox_of(src_pts);
        for (int d = 0; d < 3; d++) origin[d] = bb.mn[d];
        // cell size from cube-root density; guard degenerate extents.
        float vol = 1.f;
        int   nz_dims = 0;
        for (int d = 0; d < 3; d++) { float e = bb.ext(d); if (e > 0) { vol *= e; nz_dims++; } }
        int n_cells_target = std::max(1, (int)(Ns / std::max(1, target_per_cell)));
        if (nz_dims == 0 || vol <= 0) {
            cell = 1.f;
        } else {
            // average extent -> per-axis cell count ~ cbrt(n_cells_target)
            float avg_ext = 0; int c = 0;
            for (int d = 0; d < 3; d++) { float e = bb.ext(d); if (e > 0) { avg_ext += e; c++; } }
            avg_ext /= std::max(1, c);
            float per_axis = std::cbrt((float)n_cells_target);
            cell = avg_ext / std::max(1.f, per_axis);
            if (cell <= 0) cell = 1.f;
        }
        for (int d = 0; d < 3; d++) {
            int n = (int)std::floor(bb.ext(d) / cell) + 1;
            dim[d] = std::max(1, n);
        }
        // CSR fill: count, prefix-sum, scatter.
        std::vector<int> counts(ncells(), 0);
        std::vector<int> cidx(Ns);
        for (size_t i = 0; i < Ns; i++) {
            int cx, cy, cz; cell_of(&src_pts[i * 3], cx, cy, cz);
            int c = lin(cx, cy, cz);
            cidx[i] = c; counts[c]++;
        }
        cell_start.assign(ncells() + 1, 0);
        for (int c = 0; c < ncells(); c++) cell_start[c + 1] = cell_start[c] + counts[c];
        point_idx.assign(Ns, 0);
        std::vector<int> cursor(cell_start.begin(), cell_start.end() - 1);
        for (size_t i = 0; i < Ns; i++) point_idx[cursor[cidx[i]]++] = (int)i;
    }

    // Find up to k nearest src points to query q. Fills out_idx/out_d2 (sorted ascending by d2),
    // returns the count found (<=k). Expanding-ring shell search; stops once the ring's minimum
    // possible distance exceeds the current k-th best (correctness for any frame).
    int query_knn(const float* q, int k, int* out_idx, float* out_d2) const {
        int qx, qy, qz; cell_of(q, qx, qy, qz);
        // running top-k (small k; simple insertion into a max-on-top array)
        int   cnt = 0;
        float worst = std::numeric_limits<float>::max();
        auto consider = [&](int pi) {
            const float* p = &(*pts)[(size_t)pi * 3];
            float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
            float d2 = dx * dx + dy * dy + dz * dz;
            if (cnt < k) {
                // insertion sort into ascending list
                int j = cnt - 1;
                while (j >= 0 && out_d2[j] > d2) { out_d2[j + 1] = out_d2[j]; out_idx[j + 1] = out_idx[j]; j--; }
                out_d2[j + 1] = d2; out_idx[j + 1] = pi; cnt++;
                if (cnt == k) worst = out_d2[k - 1];
            } else if (d2 < worst) {
                int j = k - 2;
                while (j >= 0 && out_d2[j] > d2) { out_d2[j + 1] = out_d2[j]; out_idx[j + 1] = out_idx[j]; j--; }
                out_d2[j + 1] = d2; out_idx[j + 1] = pi;
                worst = out_d2[k - 1];
            }
        };
        int max_ring = 0;
        for (int d = 0; d < 3; d++) max_ring = std::max(max_ring, dim[d]);
        for (int ring = 0; ring <= max_ring; ring++) {
            // when we already have k, stop if the nearest possible point in this ring is farther
            // than the current worst (ring-1 fully covers radius (ring-1)*cell around the query cell).
            if (cnt >= k && ring > 0) {
                float ring_min = (float)(ring - 1) * cell;
                if (ring_min * ring_min > worst) break;
            }
            int x0 = qx - ring, x1 = qx + ring;
            int y0 = qy - ring, y1 = qy + ring;
            int z0 = qz - ring, z1 = qz + ring;
            for (int cz = z0; cz <= z1; cz++) {
                if (cz < 0 || cz >= dim[2]) continue;
                bool z_shell = (cz == z0 || cz == z1);
                for (int cy = y0; cy <= y1; cy++) {
                    if (cy < 0 || cy >= dim[1]) continue;
                    bool y_shell = (cy == y0 || cy == y1);
                    for (int cx = x0; cx <= x1; cx++) {
                        if (cx < 0 || cx >= dim[0]) continue;
                        // only visit the OUTER shell of the ring (interior was visited earlier)
                        bool x_shell = (cx == x0 || cx == x1);
                        if (ring > 0 && !x_shell && !y_shell && !z_shell) continue;
                        int c = lin(cx, cy, cz);
                        for (int s = cell_start[c]; s < cell_start[c + 1]; s++) consider(point_idx[s]);
                    }
                }
            }
        }
        return cnt;
    }
};

// Transfer skin weights from sampled points (src) to a target mesh (dst).
//   src_pts  : Ns*3   sampled point positions
//   src_w    : Ns*J   per-sample skin-weight rows (row-major)
//   J        : joint count
//   dst_verts: Nd*3   target mesh vertices
//   dst_w    : out, Nd*J blended skin-weight rows (row-major)
//   k        : nearest neighbours to blend (default 4)
// Inverse-distance weighting: contribution_i = 1/(d_i + eps). Writes full rows; downstream
// top-4 selection + renorm happens in the GLB writer. Reports src/dst bboxes via out-params.
inline void transfer_skin(const std::vector<float>& src_pts,
                          const std::vector<float>& src_w,
                          int J,
                          const std::vector<float>& dst_verts,
                          std::vector<float>&       dst_w,
                          int k = 4,
                          BBox* src_bbox_out = nullptr,
                          BBox* dst_bbox_out = nullptr) {
    const size_t Ns = src_pts.size() / 3;
    const size_t Nd = dst_verts.size() / 3;
    dst_w.assign(Nd * (size_t)J, 0.f);
    if (Ns == 0 || Nd == 0 || J <= 0) return;

    VoxelGrid grid;
    grid.build(src_pts, /*target_per_cell=*/1);
    BBox sbb = grid.bb;
    BBox dbb = bbox_of(dst_verts);
    if (src_bbox_out) *src_bbox_out = sbb;
    if (dst_bbox_out) *dst_bbox_out = dbb;

    int kk = std::min(k, (int)Ns);
    // eps scaled to the src cloud so 1/(d+eps) is well-behaved regardless of units.
    float eps = 1e-6f * std::max(1e-6f, sbb.diag());

    #pragma omp parallel
    {
        std::vector<int>   nidx(kk);
        std::vector<float> nd2(kk);
        std::vector<float> iw(kk);
        #pragma omp for schedule(dynamic, 4096)
        for (long long vi = 0; vi < (long long)Nd; vi++) {
            const float* q = &dst_verts[(size_t)vi * 3];
            int found = grid.query_knn(q, kk, nidx.data(), nd2.data());
            float* outrow = &dst_w[(size_t)vi * J];
            if (found == 0) continue;  // (only if Ns==0, already handled)
            float wsum = 0.f;
            for (int n = 0; n < found; n++) {
                float d = std::sqrt(nd2[n]);
                float w = 1.f / (d + eps);
                iw[n] = w;
                wsum += w;
            }
            if (wsum <= 0.f) { wsum = 1.f; iw[0] = 1.f; }
            for (int n = 0; n < found; n++) {
                float wn = iw[n] / wsum;
                const float* srow = &src_w[(size_t)nidx[n] * J];
                for (int j = 0; j < J; j++) outrow[j] += wn * srow[j];
            }
        }
    }
}

}  // namespace rig
