// Dense inside/outside oracle on the grid-`resolution` lattice, built from the O-Voxel occupancy.
//
// WHY THIS EXISTS
// ---------------
// nbdc::remesh dual-contours the level set {udf == eps} of an UNSIGNED distance field (that is what
// Pixal3D's postprocess does, and it is the "parity" remesh).  An unsigned field has TWO level sets
// at every eps: one eps OUTSIDE the input surface and one eps INSIDE it.  So the DC output is not a
// surface at all -- it is a closed SHRINK-WRAP ENVELOPE about two voxels thick, an outward wall plus
// an inward wall joined around every rim.  Measured on the delivered miku/gilly meshes: total
// enclosed signed volume 0.0020 == surface_area * 2*eps (not the model's volume), every edge has
// exactly 2 or 4 incident faces, and ~200 of the "components" are pieces of the buried inward wall.
// Where the model is thicker than 2*eps that inward wall is buried inside the body: invisible,
// interpenetrating, and the source of every non-manifold pinch (the two walls meet where a feature
// is thinner than 2*eps).
//
// The cure is to SIGN the field so the inward level set never exists.  Signing needs an
// inside/outside oracle -- and the input O-Voxel dual-grid mesh cannot provide one: its winding is
// random (measured signed-volume ratio -0.04, only 44% of faces outward) and it has ~58k boundary
// edges, so neither a pseudonormal test nor ray parity works on it.  The occupancy voxel set that
// the same decoder emits (svp*::m4_decode_mesh's out_coords1024) CAN: flood-fill its exterior and
// what is left is the body.
//
// ACCURACY BUDGET (why `seal` and `erode` are what they are)
// ----------------------------------------------------------
// The sign only has to be right where udf > eps == 1 voxel, because at udf <= eps the signed and
// unsigned fields agree by construction (see nbdc::remesh).  So this field must
//   (a) contain every point deeper than 1 voxel inside the model  -- else a residual inward wall
//       re-appears at the depth where the flag stops, and
//   (b) not extend more than 1 voxel outside it                   -- else the outward wall is
//       dragged inward there.
// That is a two-voxel-wide window for the boundary, which is comfortable.  The raw occupancy is a
// ~1-voxel-thick shell straddling the surface and is NOT 6-connected watertight (measured
// solid/wall = 1.03 without a seal: the exterior fill leaks straight through it), so we dilate by
// `seal` before filling and erode by `seal + 1` after, landing the boundary about half a voxel
// inside the surface.  A feature thinner than ~2*(seal+1) voxels ends up with no interior at all,
// which is the RIGHT answer: a thin sheet (cloth, a ribbon, hair) has no inside to bury, and its
// envelope is kept whole rather than being punched full of holes.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace solidfield {

// 1 = inside the model.  Queried per grid vertex by nbdc::remesh, so `inside` must be cheap and
// must answer "outside" for anything beyond the stored box (the box is the occupancy bbox padded).
struct SolidField {
    std::vector<uint8_t> v;
    int lo[3] = {0, 0, 0};
    int dim[3] = {0, 0, 0};
    bool empty() const { return v.empty(); }
    inline bool inside(int x, int y, int z) const {
        x -= lo[0]; y -= lo[1]; z -= lo[2];
        if (x < 0 || y < 0 || z < 0 || x >= dim[0] || y >= dim[1] || z >= dim[2]) return false;
        return v[((size_t)x * dim[1] + y) * dim[2] + z] != 0;
    }
};

namespace detail {

// Separable box dilate/erode by `r` (a (2r+1)^3 structuring element).  Three 1-D passes, O(N*r).
// Erosion runs the same sweep on the complement so one routine covers both.
inline void box_morph(std::vector<uint8_t>& g, int nx, int ny, int nz, int r, bool dilate) {
    if (r <= 0) return;
    if (!dilate) for (auto& c : g) c = (uint8_t)(1 - c);
    std::vector<uint8_t> line, out;
    // One axis pass: dilate `line` by r in 1-D via a forward and a backward reach counter.
    // `run` is "cells of reach still owed"; a set cell recharges it to r, so the r cells that
    // follow are covered and the (r+1)-th is not.
    auto pass = [&](int n) {
        out.assign(n, 0);
        int run = -1;
        for (int i = 0; i < n; i++) {
            if (line[i]) run = r; else if (run > 0) --run; else run = -1;
            if (run >= 0) out[i] = 1;
        }
        run = -1;
        for (int i = n - 1; i >= 0; i--) {
            if (line[i]) run = r; else if (run > 0) --run; else run = -1;
            if (run >= 0) out[i] = 1;
        }
    };
    const size_t sx = (size_t)ny * nz, sy = (size_t)nz;
    line.resize(nx);
    for (int y = 0; y < ny; y++) for (int z = 0; z < nz; z++) {
        for (int x = 0; x < nx; x++) line[x] = g[(size_t)x * sx + (size_t)y * sy + z];
        pass(nx);
        for (int x = 0; x < nx; x++) g[(size_t)x * sx + (size_t)y * sy + z] = out[x];
    }
    line.resize(ny);
    for (int x = 0; x < nx; x++) for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) line[y] = g[(size_t)x * sx + (size_t)y * sy + z];
        pass(ny);
        for (int y = 0; y < ny; y++) g[(size_t)x * sx + (size_t)y * sy + z] = out[y];
    }
    line.resize(nz);
    for (int x = 0; x < nx; x++) for (int y = 0; y < ny; y++) {
        uint8_t* col = &g[(size_t)x * sx + (size_t)y * sy];
        for (int z = 0; z < nz; z++) line[z] = col[z];
        pass(nz);
        for (int z = 0; z < nz; z++) col[z] = out[z];
    }
    if (!dilate) for (auto& c : g) c = (uint8_t)(1 - c);
}

// SPAN flood fill of the exterior (6-connected) from cell (0,0,0), which the padding guarantees is
// empty.  Span-based rather than per-cell: a per-cell stack on a full-resolution grid (a quarter of
// a billion cells at res 1024, three times that at 1536) can hold the whole exterior at once, which
// is gigabytes.  Spans bound the stack by the frontier area instead.
inline void flood_exterior(const std::vector<uint8_t>& wall, std::vector<uint8_t>& ext,
                           int nx, int ny, int nz) {
    ext.assign(wall.size(), 0);
    const size_t sx = (size_t)ny * nz, sy = (size_t)nz;
    struct Span { int x, y, z0, z1; };
    std::vector<Span> stack;
    auto span_at = [&](int x, int y, int z) {
        const uint8_t* w = &wall[(size_t)x * sx + (size_t)y * sy];
        uint8_t* e = &ext[(size_t)x * sx + (size_t)y * sy];
        if (w[z] || e[z]) return;
        int z0 = z, z1 = z;
        while (z0 > 0 && !w[z0 - 1] && !e[z0 - 1]) --z0;
        while (z1 + 1 < nz && !w[z1 + 1] && !e[z1 + 1]) ++z1;
        for (int k = z0; k <= z1; k++) e[k] = 1;
        stack.push_back({x, y, z0, z1});
    };
    span_at(0, 0, 0);
    while (!stack.empty()) {
        const Span s = stack.back(); stack.pop_back();
        const int nb[4][2] = {{s.x - 1, s.y}, {s.x + 1, s.y}, {s.x, s.y - 1}, {s.x, s.y + 1}};
        for (const auto& p : nb) {
            if (p[0] < 0 || p[0] >= nx || p[1] < 0 || p[1] >= ny) continue;
            const uint8_t* w = &wall[(size_t)p[0] * sx + (size_t)p[1] * sy];
            const uint8_t* e = &ext[(size_t)p[0] * sx + (size_t)p[1] * sy];
            for (int z = s.z0; z <= s.z1; z++) if (!w[z] && !e[z]) span_at(p[0], p[1], z);
        }
    }
}

// 3-D MAJORITY (median) filter over a (2r+1)^3 box, `passes` times.  The occupancy is a voxel set,
// so the interior it bounds has a voxel STAIRCASE for a boundary.  Where the model is thick that
// does not matter -- the sign only decides which side of a surface that is a voxel away.  Where it
// is THIN (miku's twintails, a skirt hem) the flipped region is only a voxel or two across, so its
// own boundary becomes part of the contoured surface, and the staircase shows up as a fine ripple on
// something the eye is looking straight at.  Median-filtering the field first takes that ripple out
// without moving the boundary anywhere it was not already.  Box sums are separable, so this is O(N)
// per pass regardless of r.
inline void majority_smooth(std::vector<uint8_t>& g, int nx, int ny, int nz, int r, int passes) {
    if (r <= 0 || passes <= 0) return;
    const size_t n = g.size(), sx = (size_t)ny * nz, sy = (size_t)nz;
    std::vector<uint16_t> a(n), b(n);
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < n; ++i) a[i] = g[i];
        // X
        for (int y = 0; y < ny; y++) for (int z = 0; z < nz; z++) {
            int acc = 0;
            for (int x = 0; x <= r && x < nx; x++) acc += a[(size_t)x * sx + (size_t)y * sy + z];
            for (int x = 0; x < nx; x++) {
                b[(size_t)x * sx + (size_t)y * sy + z] = (uint16_t)acc;
                const int add = x + r + 1, sub = x - r;
                if (add < nx) acc += a[(size_t)add * sx + (size_t)y * sy + z];
                if (sub >= 0) acc -= a[(size_t)sub * sx + (size_t)y * sy + z];
            }
        }
        // Y
        for (int x = 0; x < nx; x++) for (int z = 0; z < nz; z++) {
            int acc = 0;
            for (int y = 0; y <= r && y < ny; y++) acc += b[(size_t)x * sx + (size_t)y * sy + z];
            for (int y = 0; y < ny; y++) {
                a[(size_t)x * sx + (size_t)y * sy + z] = (uint16_t)acc;
                const int add = y + r + 1, sub = y - r;
                if (add < ny) acc += b[(size_t)x * sx + (size_t)add * sy + z];
                if (sub >= 0) acc -= b[(size_t)x * sx + (size_t)sub * sy + z];
            }
        }
        // Z + threshold.  The window is clipped at the array border, so compare against the number
        // of cells actually summed rather than a constant.
        const int W = 2 * r + 1;
        for (int x = 0; x < nx; x++) for (int y = 0; y < ny; y++) {
            const uint16_t* col = &a[(size_t)x * sx + (size_t)y * sy];
            uint8_t* out = &g[(size_t)x * sx + (size_t)y * sy];
            int acc = 0;
            for (int z = 0; z <= r && z < nz; z++) acc += col[z];
            for (int z = 0; z < nz; z++) {
                const int cx = std::min(x + r, nx - 1) - std::max(x - r, 0) + 1;
                const int cy = std::min(y + r, ny - 1) - std::max(y - r, 0) + 1;
                const int cz = std::min(z + r, nz - 1) - std::max(z - r, 0) + 1;
                out[z] = (uint8_t)(2 * acc > cx * cy * cz ? 1 : 0);
                const int add = z + r + 1, sub = z - r;
                if (add < nz) acc += col[add];
                if (sub >= 0) acc -= col[sub];
            }
        }
        (void)W;
    }
}

}  // namespace detail

// coords = [n,4] (batch,x,y,z) occupancy on the grid-`resolution` lattice.
// Returns an empty field (and prints why) if the input cannot produce a sealed solid, so callers
// can fall back to the unsigned behaviour rather than silently contouring a garbage sign.
inline SolidField build(const int32_t* coords, int n, int resolution, int seal, int erode,
                        bool verbose) {
    SolidField out;
    if (!coords || n <= 0 || resolution <= 0) return out;
    int mn[3] = {resolution, resolution, resolution}, mx[3] = {-1, -1, -1};
    for (int i = 0; i < n; i++) for (int d = 0; d < 3; d++) {
        const int c = coords[(size_t)i * 4 + 1 + d];
        if (c < mn[d]) mn[d] = c;
        if (c > mx[d]) mx[d] = c;
    }
    if (mx[0] < mn[0]) return out;
    // Pad must clear the dilation so the corner seed is guaranteed empty, plus a voxel of slack for
    // the grid-vertex (as opposed to cell) lookups nbdc makes.
    const int pad = 2 + (seal > 0 ? seal : 0);
    for (int d = 0; d < 3; d++) { out.lo[d] = mn[d] - pad; out.dim[d] = (mx[d] - mn[d]) + 1 + 2 * pad; }
    const size_t ncell = (size_t)out.dim[0] * out.dim[1] * out.dim[2];
    std::vector<uint8_t> occ(ncell, 0);
    const size_t sx = (size_t)out.dim[1] * out.dim[2], sy = (size_t)out.dim[2];
    for (int i = 0; i < n; i++)
        occ[(size_t)(coords[(size_t)i * 4 + 1] - out.lo[0]) * sx +
            (size_t)(coords[(size_t)i * 4 + 2] - out.lo[1]) * sy +
            (size_t)(coords[(size_t)i * 4 + 3] - out.lo[2])] = 1;
    int64_t wall = 0; for (size_t i = 0; i < ncell; i++) wall += occ[i];

    std::vector<uint8_t> sealed = occ;
    detail::box_morph(sealed, out.dim[0], out.dim[1], out.dim[2], seal, /*dilate=*/true);
    std::vector<uint8_t> ext;
    detail::flood_exterior(sealed, ext, out.dim[0], out.dim[1], out.dim[2]);
    std::vector<uint8_t>().swap(sealed);
    out.v.resize(ncell);
    for (size_t i = 0; i < ncell; i++) out.v[i] = (uint8_t)(ext[i] ? 0 : 1);
    std::vector<uint8_t>().swap(ext);
    int64_t filled = 0; for (size_t i = 0; i < ncell; i++) filled += out.v[i];
    // Pull the boundary back to ~half a voxel INSIDE the surface: the fill above stopped at the
    // dilated shell's outer face, i.e. seal + ~0.5 voxels outside it.
    detail::box_morph(out.v, out.dim[0], out.dim[1], out.dim[2], erode, /*dilate=*/false);
    // OFF by default: it is a small, real improvement at the DC stage (miku chi -10 -> -1,
    // non-manifold edges 255 -> 221 at r=1/p=2) but it has not been measured end to end, and the
    // ripple it was written to remove turned out to be render moire rather than geometry -- the
    // twintail surface is equally smooth before and after at magnification, and the delivered mesh's
    // area-weighted mean dihedral angle IMPROVES (8.94 deg -> 8.14 deg) without it. Left as a knob.
    const int mr = std::getenv("NBDC_SOLID_MEDIAN_R") ? atoi(std::getenv("NBDC_SOLID_MEDIAN_R")) : 1;
    const int mp = std::getenv("NBDC_SOLID_MEDIAN_P") ? atoi(std::getenv("NBDC_SOLID_MEDIAN_P")) : 0;
    detail::majority_smooth(out.v, out.dim[0], out.dim[1], out.dim[2], mr, mp);
    int64_t solid = 0; for (size_t i = 0; i < ncell; i++) solid += out.v[i];
    const double ratio = wall ? (double)filled / (double)wall : 0.0;
    if (verbose)
        std::fprintf(stderr,
                     "[solid] occupancy=%lld voxels -> filled=%lld (fill/wall %.2f) -> interior=%lld"
                     " (seal=%d erode=%d, grid %dx%dx%d, %.0f MB)\n",
                     (long long)wall, (long long)filled, ratio, (long long)solid, seal, erode,
                     out.dim[0], out.dim[1], out.dim[2], (double)ncell / 1e6);
    // The one number that sees a leak.  A filled body is many times its own wall; a leak pins the
    // ratio near 1.0 and the "interior" collapses onto the shell, which would make the signed field
    // WORSE than the unsigned one (spurious flips right at the surface).  Refuse instead.
    if (ratio < 1.2 || solid <= 0) {
        std::fprintf(stderr,
                     "[solid] *** exterior flood fill LEAKED through the occupancy shell "
                     "(fill/wall %.2f, interior %lld) -- refusing to sign the field; raise seal ***\n",
                     ratio, (long long)solid);
        out.v.clear();
    }
    return out;
}

}  // namespace solidfield
