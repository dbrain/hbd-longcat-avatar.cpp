// Sparse-volume trilinear grid_sample (host port of flex_gemm GridSample3dTorch._trilinear).
// This is the one net-new MATH op for the UV-atlas texture bake: given a sparse voxel volume
// (feats [N,C] at integer coords [N,3], on a grid of side `gs`), sample C-channel attributes at
// arbitrary continuous query points `q` in grid-index space (q = (pos+0.5)*resolution).
//
// Algorithm (verbatim from grid_sample_torch.py:_trilinear) — voxel at integer coord c occupies
// the cube centred at c+0.5; the 8 trilinear neighbours are int(q ± 0.5) per axis (int() truncates
// toward zero, matching torch .int()); each present neighbour gets weight prod(1 - |c+0.5 - q|);
// missing neighbours contribute 0; the result is the present-weighted average (renormalised by the
// sum of present weights, NOT attenuated). Validated bit-exact vs flex_gemm on the real PBR volume.
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <unordered_map>

namespace texgs {

// injective lattice key for coords in [-1, 4094] (corners of q in [0, gs] stay in range)
static inline int64_t key3(int x, int y, int z) {
    return ((int64_t)(x + 1) * 4096 + (int64_t)(y + 1)) * 4096 + (int64_t)(z + 1);
}

// Hashmap over the sparse coords: (x,y,z) -> feat index. coords are [N,*stride] with the xyz at
// offset `xoff` (4-wide [b,x,y,z] -> xoff=1; 3-wide [x,y,z] -> xoff=0).
struct VolIndex {
    std::unordered_map<int64_t,int> map;
    // Nearest-voxel fallback is used by the atlas baker when a decimated surface lies just
    // outside the sparse PBR shell.  Probing every lattice point in an 8-voxel cube costs up to
    // 4,913 hash lookups per texel.  Keep 4^3-cell buckets so the exact same search considers
    // only occupied candidates in the requested neighbourhood instead.
    std::unordered_map<int64_t,std::vector<int>> near_buckets;
    const int32_t* coords = nullptr;
    int stride = 0, xoff = 0;
    VolIndex(const int32_t* coords_in, int N, int stride_in, int xoff_in)
        : coords(coords_in), stride(stride_in), xoff(xoff_in) {
        map.reserve((size_t)N * 2);
        near_buckets.reserve((size_t)N / 2);
        for (int i = 0; i < N; i++) {
            const int x=coords[(size_t)i*stride+xoff], y=coords[(size_t)i*stride+xoff+1], z=coords[(size_t)i*stride+xoff+2];
            map[key3(x,y,z)] = i;
            near_buckets[key3(x >> 2, y >> 2, z >> 2)].push_back(i);
        }
    }
    inline int find(int x, int y, int z) const {
        auto it = map.find(key3(x,y,z));
        return it == map.end() ? -1 : it->second;
    }
    // Exact nearest occupied voxel to (qx,qy,qz) within Chebyshev radius Rmax.  The old expanding
    // lattice-shell implementation repeated a sparse-map lookup for every empty coordinate; the
    // coarse buckets retain the same radius/Euclidean contract while visiting only occupied cells.
    // Used by the REMESH texture bake: the remeshed surface sits a few voxels off the sparse PBR
    // shell, so the 8 trilinear neighbours are all empty → fall back to the nearest shell voxel.
    inline int find_nearest(float qx, float qy, float qz, int Rmax) const {
        int bx=(int)std::lround(qx-0.5f), by=(int)std::lround(qy-0.5f), bz=(int)std::lround(qz-0.5f);
        auto floor4=[](int v) { return v >= 0 ? v/4 : -(((-v)+3)/4); };
        const int xmin=bx-Rmax, xmax=bx+Rmax, ymin=by-Rmax, ymax=by+Rmax, zmin=bz-Rmax, zmax=bz+Rmax;
        int best=-1; float bestd=1e30f;
        for (int ux=floor4(xmin); ux<=floor4(xmax); ux++)
            for (int uy=floor4(ymin); uy<=floor4(ymax); uy++)
                for (int uz=floor4(zmin); uz<=floor4(zmax); uz++) {
                    auto it=near_buckets.find(key3(ux,uy,uz));
                    if (it==near_buckets.end()) continue;
                    for (int idx : it->second) {
                        const int x=coords[(size_t)idx*stride+xoff], y=coords[(size_t)idx*stride+xoff+1], z=coords[(size_t)idx*stride+xoff+2];
                        if (x<xmin || x>xmax || y<ymin || y>ymax || z<zmin || z>zmax) continue;
                        float dx=x+0.5f-qx, dy=y+0.5f-qy, dz=z+0.5f-qz;
                        float d=dx*dx+dy*dy+dz*dz;
                        if (d<bestd){ bestd=d; best=idx; }
                    }
                }
        return best;
    }
};

// Sample one query point q (grid-index space) -> out[C]. feats is [N,C] row-major.
static inline void sample_one(const VolIndex& vol, const float* feats, int C,
                              float qx, float qy, float qz, float* out, int fallback_r = 0) {
    for (int c = 0; c < C; c++) out[c] = 0.f;
    float tw = 0.f;
    // 8 corners: (dx,dy,dz) in {-0.5,+0.5}^3, in the torch corner order
    static const float OFF[8][3] = {
        {-0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f, 0.5f}, {-0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f, 0.5f},
        { 0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f, 0.5f}, { 0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f, 0.5f},
    };
    for (int k = 0; k < 8; k++) {
        int cx = (int)(qx + OFF[k][0]);   // (int) truncates toward zero == torch .int()
        int cy = (int)(qy + OFF[k][1]);
        int cz = (int)(qz + OFF[k][2]);
        int idx = vol.find(cx, cy, cz);
        if (idx < 0) continue;            // missing neighbour -> weight 0
        float w = (1.f - std::fabs((float)cx + 0.5f - qx))
                * (1.f - std::fabs((float)cy + 0.5f - qy))
                * (1.f - std::fabs((float)cz + 0.5f - qz));
        const float* f = &feats[(size_t)idx * C];
        for (int c = 0; c < C; c++) out[c] += w * f[c];
        tw += w;
    }
    if (tw > 1e-12f) { float inv = 1.f / tw; for (int c = 0; c < C; c++) out[c] *= inv; return; }
    // trilinear missed (remesh surface off the sparse shell) → nearest-voxel fallback
    if (fallback_r > 0) {
        int idx = vol.find_nearest(qx, qy, qz, fallback_r);
        if (idx >= 0) { const float* f=&feats[(size_t)idx*C]; for (int c=0;c<C;c++) out[c]=f[c]; return; }
    }
    for (int c = 0; c < C; c++) out[c] = 0.f;
}

// Like sample_one but WITHOUT the nearest-voxel fallback, and returns the total present trilinear
// weight tw in [0,1] alongside the (renormalised) value. tw is a direct measure of how well the
// query sits inside the occupied shell: tw~=1 => all 8 corners present => full support => a smooth,
// stable read; tw small => the point is in the shell FRINGE, only a corner or two present, and the
// renormalised value flips between neighbouring texels as different corner subsets appear -> the
// speckle grain. Used by the normal-march projector to find the on-shell point (max tw) along the
// texel normal, which is where the volume field is clean.
static inline float sample_one_weighted(const VolIndex& vol, const float* feats, int C,
                                        float qx, float qy, float qz, float* out) {
    for (int c = 0; c < C; c++) out[c] = 0.f;
    float tw = 0.f;
    static const float OFF[8][3] = {
        {-0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f, 0.5f}, {-0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f, 0.5f},
        { 0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f, 0.5f}, { 0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f, 0.5f},
    };
    for (int k = 0; k < 8; k++) {
        int cx = (int)(qx + OFF[k][0]);
        int cy = (int)(qy + OFF[k][1]);
        int cz = (int)(qz + OFF[k][2]);
        int idx = vol.find(cx, cy, cz);
        if (idx < 0) continue;
        float w = (1.f - std::fabs((float)cx + 0.5f - qx))
                * (1.f - std::fabs((float)cy + 0.5f - qy))
                * (1.f - std::fabs((float)cz + 0.5f - qz));
        const float* f = &feats[(size_t)idx * C];
        for (int c = 0; c < C; c++) out[c] += w * f[c];
        tw += w;
    }
    if (tw > 1e-12f) { float inv = 1.f / tw; for (int c = 0; c < C; c++) out[c] *= inv; }
    return tw;
}

// NORMAL-MARCH projection: the sparse PBR volume was decoded on the COARSE grid-`gs` iso-surface;
// the refined delivery mesh sits a few voxels off that shell along its own normal, so a direct read
// at the texel lands in the shell fringe (partial support -> speckle) or misses entirely. Marching
// the query along +/- the texel normal `n` (unit, in world space == grid space since q=(pos+0.5)*gs)
// and taking the point of MAXIMUM trilinear support snaps onto the volume iso-surface WITHOUT any
// lateral slide (unlike closest-point-on-mesh, which smears thin features across material
// boundaries). Returns the tw of the chosen point (0 if nothing on the shell was found within
// `range` voxels). Searches closest-to-surface first and stops early once full support is reached,
// so faithful geometry wins ties and the common case is cheap.
static inline float sample_shell_normal_march(const VolIndex& vol, const float* feats, int C,
                                              float qx, float qy, float qz,
                                              float nx, float ny, float nz,
                                              float range, float step, float* out) {
    // SUPPORT-WEIGHTED ACCUMULATION (not hard argmax). A hard argmax over the march ("take the
    // single most-supported point") is spatially UNSTABLE on a fine mesh: adjacent texels have
    // slightly different normals, so their argmax flips to a different on-shell voxel and the atlas
    // streaks. Instead integrate the trilinear reads along the normal weighted by their own support
    // AND a Gaussian centred on the refined surface (t=0). The support weight concentrates the
    // integral on the shell crossing (off-shell points contribute ~0); the Gaussian prefers the
    // NEAREST crossing (so a thin feature whose +n and -n both hit a wall does not blend the far
    // wall) and, crucially, makes the result a SMOOTH function of the query -> no per-texel flip,
    // no streaks. Averaging is ALONG the normal (through the thin shell, where colour barely
    // varies), never laterally, so material boundaries stay crisp. Returns the accumulated support
    // weight as a coverage measure (0 => nothing on the shell along the normal).
    const float sigma = std::getenv("RP_NM_SIGMA") ? (float)atof(std::getenv("RP_NM_SIGMA")) : 3.0f;
    const float inv2s2 = 1.f / (2.f * sigma * sigma);
    float acc[8] = {0,0,0,0,0,0,0,0};   // C<=6 for PBR
    float wsum = 0.f, twsum = 0.f;
    float tmp[8];
    for (float a = 0.f; a <= range + 1e-6f; a += step) {
        for (int s = 0; s < (a == 0.f ? 1 : 2); s++) {
            float t = (s == 0) ? a : -a;
            float tw = sample_one_weighted(vol, feats, C, qx + t*nx, qy + t*ny, qz + t*nz, tmp);
            if (tw <= 1e-6f) continue;
            float w = tw * std::exp(-(t*t) * inv2s2);
            for (int c = 0; c < C; c++) acc[c] += w * tmp[c];
            wsum += w; twsum += tw;
        }
    }
    if (wsum > 1e-9f) { float inv = 1.f / wsum; for (int c = 0; c < C; c++) out[c] = acc[c] * inv; }
    else { for (int c = 0; c < C; c++) out[c] = 0.f; return 0.f; }
    // coverage = peak support reached; twsum/count is noisy, so report the best single-point support
    // approximately via wsum normalised by the Gaussian mass near 0. Simpler: return min(1, twsum).
    return twsum > 1.f ? 1.f : twsum;
}

// Batch: query[Q,3] (grid-index space) -> out[Q,C]. OpenMP over queries.
static inline void grid_sample_trilinear(const float* feats, const int32_t* coords, int N, int coord_stride,
                                         int coord_xoff, int C, const float* query, int Q, float* out) {
    VolIndex vol(coords, N, coord_stride, coord_xoff);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Q; i++)
        sample_one(vol, feats, C, query[(size_t)i*3+0], query[(size_t)i*3+1], query[(size_t)i*3+2], &out[(size_t)i*C]);
}

}  // namespace texgs
