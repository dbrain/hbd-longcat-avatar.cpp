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
    VolIndex(const int32_t* coords, int N, int stride, int xoff) {
        map.reserve((size_t)N * 2);
        for (int i = 0; i < N; i++)
            map[key3(coords[(size_t)i*stride+xoff], coords[(size_t)i*stride+xoff+1], coords[(size_t)i*stride+xoff+2])] = i;
    }
    inline int find(int x, int y, int z) const {
        auto it = map.find(key3(x,y,z));
        return it == map.end() ? -1 : it->second;
    }
    // nearest occupied voxel to (qx,qy,qz) within Chebyshev radius Rmax (expanding-shell search,
    // returns the Euclidean-nearest within the first non-empty shell + one extra shell). -1 if none.
    // Used by the REMESH texture bake: the remeshed surface sits a few voxels off the sparse PBR
    // shell, so the 8 trilinear neighbours are all empty → fall back to the nearest shell voxel.
    inline int find_nearest(float qx, float qy, float qz, int Rmax) const {
        int bx=(int)std::lround(qx-0.5f), by=(int)std::lround(qy-0.5f), bz=(int)std::lround(qz-0.5f);
        int best=-1; float bestd=1e30f; int found_r=-1;
        for (int r=0; r<=Rmax; r++) {
            if (found_r>=0 && r>found_r+1) break;     // one shell past the first hit is enough
            // iterate the Chebyshev shell of radius r (faces of the r-box)
            for (int dx=-r; dx<=r; dx++) for (int dy=-r; dy<=r; dy++) for (int dz=-r; dz<=r; dz++) {
                if (std::max(std::max(std::abs(dx),std::abs(dy)),std::abs(dz)) != r) continue;
                int idx = find(bx+dx, by+dy, bz+dz);
                if (idx<0) continue;
                float cx=bx+dx+0.5f, cy=by+dy+0.5f, cz=bz+dz+0.5f;
                float d=(cx-qx)*(cx-qx)+(cy-qy)*(cy-qy)+(cz-qz)*(cz-qz);
                if (d<bestd){ bestd=d; best=idx; }
            }
            if (best>=0 && found_r<0) found_r=r;
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

// Batch: query[Q,3] (grid-index space) -> out[Q,C]. OpenMP over queries.
static inline void grid_sample_trilinear(const float* feats, const int32_t* coords, int N, int coord_stride,
                                         int coord_xoff, int C, const float* query, int Q, float* out) {
    VolIndex vol(coords, N, coord_stride, coord_xoff);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Q; i++)
        sample_one(vol, feats, C, query[(size_t)i*3+0], query[(size_t)i*3+1], query[(size_t)i*3+2], &out[(size_t)i*C]);
}

}  // namespace texgs
