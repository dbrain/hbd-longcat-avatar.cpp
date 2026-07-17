// Native UltraShape voxel_cond: coarse mesh -> normalized surface point cloud -> res-128 occupied
// voxels -> a num_latents subset (positions for the RefineDiT 3D-RoPE / latent placement).
// Mirrors SharpEdgeSurfaceLoader(normalize_scale=0.99) + voxelize_from_point (utils/voxelize.py):
//   norm = (pc+1)/2 ; idx = clamp(floor(norm*RES), 0, RES-1) ; unique ; shuffle ; take K (wrap if <K).
// The Python surface loader uses an UNSEEDED rng, so voxel_cond is not bit-reproducible; the OCCUPIED
// SET is the stable quantity (validated by IoU/containment in us_voxelize_test.cpp). normalize_mesh here
// = rig::normalize_mesh (longest axis -> [-1,1]) then *0.99 (UltraShape's scale=0.99 shrink).
#pragma once
#include "glb_reader.hpp"
#include "mesh_sample.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <array>
#include <random>
#include <algorithm>
#include <unordered_set>

namespace us_vox {

static inline uint64_t pack_voxel(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) | (uint64_t)(uint32_t)z;
}

// Occupied res^3 voxels of the (already normalized to [-s,s]) point cloud, in first-seen order.
static inline std::vector<std::array<int,3>> occupied_voxels(const std::vector<float>& pc, int res) {
    std::vector<std::array<int,3>> occ;
    std::unordered_set<uint64_t> seen;
    occ.reserve(1 << 14);
    const size_t N = pc.size() / 3;
    for (size_t i = 0; i < N; i++) {
        int v[3];
        for (int c = 0; c < 3; c++) {
            float nrm = (pc[i*3+c] + 1.0f) * 0.5f;
            int idx = (int)std::floor(nrm * res);
            if (idx < 0) idx = 0;
            if (idx > res-1) idx = res-1;
            v[c] = idx;
        }
        if (seen.insert(pack_voxel(v[0],v[1],v[2])).second) occ.push_back({v[0],v[1],v[2]});
    }
    return occ;
}

struct VoxelCond {
    std::vector<int>   coords;   // [K*3] grid indices (== voxel_cond)
    std::vector<float> centers;  // [K*3] cell centers in [-1,1]: (g+0.5)*2/res - 1
    int n_occupied = 0;
};

// In-memory core: (already-owned) coarse verts[V*3] + int64 faces[F*3] -> voxel_cond [K,3].
// Same recipe as voxelize_mesh (normalize longest axis -> [-1,1], *0.99, area-sample, res occupancy,
// seeded shuffle take-K) but with NO GLB round-trip — for the inline image_to_rig driver where the
// coarse mesh is already in RAM from pixal3d. verts is COPIED (normalized in place on the copy).
static inline bool voxelize_mesh_inmem(std::vector<float> verts, const std::vector<int64_t>& faces,
                                       int K, int res, int num_sample, uint64_t seed, VoxelCond& out) {
    rig::normalize_mesh(verts);                      // longest axis -> [-1,1]  (on our copy)
    for (auto& v : verts) v *= 0.99f;                       // UltraShape normalize_scale=0.99
    std::vector<float> pts, nrm;
    if (!rig::sample_surface(verts, faces, num_sample, seed, pts, nrm)) return false;
    auto occ = occupied_voxels(pts, res);
    out.n_occupied = (int)occ.size();
    if (occ.empty()) return false;
    std::vector<int> perm(occ.size());
    for (size_t i = 0; i < perm.size(); i++) perm[i] = (int)i;
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ull);
    std::shuffle(perm.begin(), perm.end(), rng);
    out.coords.resize((size_t)K*3);
    out.centers.resize((size_t)K*3);
    for (int k = 0; k < K; k++) {
        const auto& v = occ[perm[k % occ.size()]];
        for (int c = 0; c < 3; c++) {
            out.coords[k*3+c]  = v[c];
            out.centers[k*3+c] = (v[c] + 0.5f) * 2.0f / res - 1.0f;
        }
    }
    return true;
}

// Full native path: coarse mesh GLB -> voxel_cond [K,3]. num_sample surface points (area-weighted),
// res-128 occupancy, then a seeded shuffle picks K (wrap-with-replacement if fewer than K occupied).
static inline bool voxelize_mesh(const char* glb_path, int K, int res, int num_sample,
                                 uint64_t seed, VoxelCond& out) {
    glb::Mesh mesh;
    if (!glb::read_glb(glb_path, mesh)) return false;
    rig::normalize_mesh(mesh.verts);                 // longest axis -> [-1,1]
    for (auto& v : mesh.verts) v *= 0.99f;                   // UltraShape normalize_scale=0.99
    std::vector<float> pts, nrm;
    if (!rig::sample_surface(mesh.verts, mesh.faces, num_sample, seed, pts, nrm)) return false;
    auto occ = occupied_voxels(pts, res);
    out.n_occupied = (int)occ.size();
    // shuffle (deterministic from seed) and take K
    std::vector<int> perm(occ.size());
    for (size_t i = 0; i < perm.size(); i++) perm[i] = (int)i;
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ull);
    std::shuffle(perm.begin(), perm.end(), rng);
    out.coords.resize((size_t)K*3);
    out.centers.resize((size_t)K*3);
    for (int k = 0; k < K; k++) {
        const auto& v = occ[perm[k % occ.size()]];
        for (int c = 0; c < 3; c++) {
            out.coords[k*3+c]  = v[c];
            out.centers[k*3+c] = (v[c] + 0.5f) * 2.0f / res - 1.0f;
        }
    }
    return true;
}

}  // namespace us_vox
