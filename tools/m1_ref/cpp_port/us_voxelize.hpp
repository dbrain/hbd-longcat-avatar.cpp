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

// Sharp-edge point sampling (USR_SHARP_EDGES). Mirrors UltraShape's sharp_sample_pointcloud
// (surface_loaders.py): a vertex is "sharp" when its (area-weighted) vertex normal diverges
// from an incident face normal (min dot < thr, ~0.985 = ~10deg); edges with BOTH ends sharp are
// sampled proportional to length. The reference feeds HALF its 409600 surface points this way so
// crease voxels (finger gaps, straps) register in the res-128 occupancy the area-sampler misses.
// Points are returned in the SAME (already-normalized) frame as the verts passed in. Empty out ->
// no sharp edges found (caller keeps the uniform samples only).
static inline void sample_sharp_edges(const std::vector<float>& verts,
                                      const std::vector<int64_t>& faces,
                                      int num, uint64_t seed, float thr,
                                      std::vector<float>& out_pts) {
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    out_pts.clear();
    if (V == 0 || F == 0 || num <= 0) return;
    std::vector<float>  fn(F * 3, 0.f);   // face normals
    std::vector<double> vn(V * 3, 0.0);   // area-weighted vertex normals (accum)
    for (size_t f = 0; f < F; ++f) {
        int64_t i0 = faces[f*3], i1 = faces[f*3+1], i2 = faces[f*3+2];
        if (i0<0||i1<0||i2<0||(size_t)i0>=V||(size_t)i1>=V||(size_t)i2>=V) continue;
        const float *a=&verts[i0*3], *b=&verts[i1*3], *c=&verts[i2*3];
        float n[3]; rig::tri_normal(a,b,c,n); fn[f*3]=n[0]; fn[f*3+1]=n[1]; fn[f*3+2]=n[2];
        double area = rig::tri_double_area(a,b,c);
        for (int k=0;k<3;k++){ vn[i0*3+k]+=n[k]*area; vn[i1*3+k]+=n[k]*area; vn[i2*3+k]+=n[k]*area; }
    }
    for (size_t v=0; v<V; ++v){
        double l=std::sqrt(vn[v*3]*vn[v*3]+vn[v*3+1]*vn[v*3+1]+vn[v*3+2]*vn[v*3+2]);
        if (l>1e-12){ vn[v*3]/=l; vn[v*3+1]/=l; vn[v*3+2]/=l; }
    }
    std::vector<double> vn2(V, 1.0);      // min dot(vertex_normal, incident face_normal)
    for (size_t f=0; f<F; ++f){
        int64_t idx[3]={faces[f*3],faces[f*3+1],faces[f*3+2]};
        for (int j=0;j<3;j++){ int64_t v=idx[j]; if (v<0||(size_t)v>=V) continue;
            double d=vn[v*3]*fn[f*3]+vn[v*3+1]*fn[f*3+1]+vn[v*3+2]*fn[f*3+2];
            if (d<vn2[v]) vn2[v]=d; }
    }
    std::vector<std::pair<int64_t,int64_t>> edges; std::vector<double> cum; double tot=0;
    auto add_edge=[&](int64_t a, int64_t b){
        if (vn2[a]<thr && vn2[b]<thr){
            const float *pa=&verts[a*3], *pb=&verts[b*3];
            double dx=pb[0]-pa[0], dy=pb[1]-pa[1], dz=pb[2]-pa[2];
            double len=std::sqrt(dx*dx+dy*dy+dz*dz);
            if (len>0){ edges.push_back({a,b}); tot+=len; cum.push_back(tot); } }
    };
    for (size_t f=0; f<F; ++f){ int64_t i0=faces[f*3],i1=faces[f*3+1],i2=faces[f*3+2];
        if (i0<0||i1<0||i2<0||(size_t)i0>=V||(size_t)i1>=V||(size_t)i2>=V) continue;
        add_edge(i0,i1); add_edge(i1,i2); add_edge(i2,i0); }
    if (edges.empty() || tot<=0) return;
    std::mt19937_64 rng(seed ^ 0xABCDEF1234567ull);
    std::uniform_real_distribution<double> U(0.0,1.0);
    out_pts.resize((size_t)num*3);
    for (int s=0;s<num;++s){
        double r=U(rng)*tot;
        size_t e=(size_t)(std::lower_bound(cum.begin(),cum.end(),r)-cum.begin());
        if (e>=edges.size()) e=edges.size()-1;
        double t=U(rng);
        const float *pa=&verts[edges[e].first*3], *pb=&verts[edges[e].second*3];
        for (int k=0;k<3;k++) out_pts[(size_t)s*3+k]=(float)(t*pa[k]+(1.0-t)*pb[k]);
    }
}

// Sample `num_sample` surface points into `pts`, honouring USR_SHARP_EDGES (half uniform + half
// sharp-edge, the reference 204800/204800 split). verts must already be in the normalized frame.
static inline bool sample_surface_maybe_sharp(const std::vector<float>& verts,
                                              const std::vector<int64_t>& faces,
                                              int num_sample, uint64_t seed,
                                              std::vector<float>& pts) {
    static const bool sharp = std::getenv("USR_SHARP_EDGES") != nullptr;
    std::vector<float> nrm;
    if (!sharp) return rig::sample_surface(verts, faces, num_sample, seed, pts, nrm);
    int nu = num_sample / 2, ns = num_sample - nu;
    if (!rig::sample_surface(verts, faces, nu, seed, pts, nrm)) return false;
    std::vector<float> sp;
    sample_sharp_edges(verts, faces, ns, seed, 0.985f, sp);
    if (sp.empty()) {                       // no creases: top up with uniform so counts match
        std::vector<float> extra, en;
        if (rig::sample_surface(verts, faces, ns, seed ^ 0x5555ull, extra, en))
            pts.insert(pts.end(), extra.begin(), extra.end());
    } else {
        pts.insert(pts.end(), sp.begin(), sp.end());
    }
    return true;
}

// In-memory core: (already-owned) coarse verts[V*3] + int64 faces[F*3] -> voxel_cond [K,3].
// Same recipe as voxelize_mesh (normalize longest axis -> [-1,1], *0.99, area-sample, res occupancy,
// seeded shuffle take-K) but with NO GLB round-trip — for the inline image_to_rig driver where the
// coarse mesh is already in RAM from pixal3d. verts is COPIED (normalized in place on the copy).
static inline bool voxelize_mesh_inmem(std::vector<float> verts, const std::vector<int64_t>& faces,
                                       int K, int res, int num_sample, uint64_t seed, VoxelCond& out) {
    rig::normalize_mesh(verts);                      // longest axis -> [-1,1]  (on our copy)
    for (auto& v : verts) v *= 0.99f;                       // UltraShape normalize_scale=0.99
    std::vector<float> pts;
    if (!sample_surface_maybe_sharp(verts, faces, num_sample, seed, pts)) return false;
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
    std::vector<float> pts;
    if (!sample_surface_maybe_sharp(mesh.verts, mesh.faces, num_sample, seed, pts)) return false;
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
