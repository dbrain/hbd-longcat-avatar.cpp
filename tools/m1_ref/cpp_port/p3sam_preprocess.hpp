// p3sam_preprocess.hpp — native C++ port of the P3-SAM input transform
// (auto_mask_no_postprocess.py: sample_surface + normalize_pc + get_feat's
//  model.transform = sonata.transform.default()).
//
// Pipeline (matching XPart/partgen/models/sonata/transform.py default()):
//   1. sample_surface(verts,faces,N)  -> area-weighted barycentric points +
//      per-point face normal (the sampled face's normal).
//   2. normalize_pc                   -> center/scale to ~unit cube.
//   3. CenterShift(apply_z=True)      -> shift x,y to center, z to min.
//   4. GridSample(0.005, fnv, train)  -> voxel-dedup to representatives.
//   5. NormalizeColor                 -> color (ones) / 255.
//   6. Collect feat = cat(coord, color, normal).
//
// NOTE: train-mode GridSample picks a RANDOM representative per voxel via
// np.random.randint; we deterministically pick the FIRST point in sorted-key
// order. This is reproducible but will NOT bit-match the golden tf_grid/tf_feat
// (which used the RNG). Downstream segmentation is robust to this.
#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

struct PreOut {
    std::vector<int32_t> grid;     // [M0,3]
    std::vector<float>   feat;     // [M0,9]  (coord3,color3,normal3)
    std::vector<int64_t> inverse;  // [Nin]
    std::vector<float>   points;   // [Nin,3] normalized point cloud
    std::vector<float>   normals;  // [Nin,3]
    std::vector<float>   points_org; // [Nin,3] pre-normalization sampled points
    int M0 = 0, Nin = 0;
};

namespace p3sam {

// Area-weighted barycentric surface sampling (trimesh.sample.sample_surface style).
// Fills points[N,3], normals[N,3] (sampled face normal, normalized).
inline void sample_surface(const float* verts, int V, const int64_t* faces, int F,
                           int N, uint64_t seed,
                           std::vector<float>& points, std::vector<float>& normals) {
    (void)V;
    // per-face areas + face normals
    std::vector<double> area(F);
    std::vector<float>  fn((size_t)F * 3);
    double total = 0.0;
    for (int f = 0; f < F; f++) {
        int64_t a = faces[(size_t)f*3+0], b = faces[(size_t)f*3+1], c = faces[(size_t)f*3+2];
        const float* v0 = verts + (size_t)a*3;
        const float* v1 = verts + (size_t)b*3;
        const float* v2 = verts + (size_t)c*3;
        double e1x=v1[0]-v0[0], e1y=v1[1]-v0[1], e1z=v1[2]-v0[2];
        double e2x=v2[0]-v0[0], e2y=v2[1]-v0[1], e2z=v2[2]-v0[2];
        double nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
        double len = std::sqrt(nx*nx+ny*ny+nz*nz);
        area[f] = 0.5*len;
        total += area[f];
        if (len > 0) { nx/=len; ny/=len; nz/=len; }
        fn[(size_t)f*3+0]=(float)nx; fn[(size_t)f*3+1]=(float)ny; fn[(size_t)f*3+2]=(float)nz;
    }
    // cumulative CDF over area
    std::vector<double> cdf(F);
    double acc = 0.0;
    for (int f = 0; f < F; f++) { acc += area[f]; cdf[f] = acc; }
    double inv_total = (total > 0) ? 1.0/total : 0.0;

    points.resize((size_t)N*3);
    normals.resize((size_t)N*3);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int i = 0; i < N; i++) {
        // pick face by area-weighted CDF
        double r = U(rng) * total;
        int lo = (int)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
        if (lo >= F) lo = F-1;
        // barycentric (trimesh: reflect across diagonal)
        double r1 = U(rng), r2 = U(rng);
        if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }
        int64_t a = faces[(size_t)lo*3+0], b = faces[(size_t)lo*3+1], c = faces[(size_t)lo*3+2];
        const float* v0 = verts + (size_t)a*3;
        const float* v1 = verts + (size_t)b*3;
        const float* v2 = verts + (size_t)c*3;
        for (int d = 0; d < 3; d++) {
            double p = v0[d] + r1*(v1[d]-v0[d]) + r2*(v2[d]-v0[d]);
            points[(size_t)i*3+d] = (float)p;
        }
        normals[(size_t)i*3+0]=fn[(size_t)lo*3+0];
        normals[(size_t)i*3+1]=fn[(size_t)lo*3+1];
        normals[(size_t)i*3+2]=fn[(size_t)lo*3+2];
    }
    (void)inv_total;
}

// normalize_pc: center=(max+min)/2, scale=max|(max-min)/2|, pc=(pc-center)/(scale+1e-10)
inline void normalize_pc(std::vector<float>& pc, int N) {
    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (int i=0;i<N;i++) for (int d=0;d<3;d++){ float v=pc[(size_t)i*3+d]; mn[d]=std::min(mn[d],v); mx[d]=std::max(mx[d],v); }
    float center[3], scale=0.f;
    for (int d=0;d<3;d++){ center[d]=(mx[d]+mn[d])*0.5f; scale=std::max(scale, std::fabs((mx[d]-mn[d])*0.5f)); }
    float inv = 1.0f/(scale + 1e-10f);
    for (int i=0;i<N;i++) for (int d=0;d<3;d++) pc[(size_t)i*3+d]=(pc[(size_t)i*3+d]-center[d])*inv;
}

// FNV64-1A on a [.,3] int grid (uint64 wraparound semantics, matches numpy uint64).
inline uint64_t fnv3(int32_t gx, int32_t gy, int32_t gz) {
    uint64_t h = 14695981039346656037ULL;
    uint64_t g[3] = { (uint64_t)(uint32_t)gx | ((gx<0)? 0xFFFFFFFF00000000ULL:0ULL),
                      (uint64_t)(uint32_t)gy | ((gy<0)? 0xFFFFFFFF00000000ULL:0ULL),
                      (uint64_t)(uint32_t)gz | ((gz<0)? 0xFFFFFFFF00000000ULL:0ULL) };
    // grid is always >=0 after grid-=min, so sign-extension is moot, but keep faithful.
    for (int j=0;j<3;j++){ h *= 1099511628211ULL; h ^= g[j]; }
    return h;
}

inline PreOut preprocess(const float* verts, int V, const int64_t* faces, int F,
                         uint64_t seed = 42) {
    const int N = 100000;
    PreOut R; R.Nin = N;
    // 1. sample surface
    sample_surface(verts, V, faces, F, N, seed, R.points, R.normals);
    R.points_org = R.points;          // org = pre-normalization copy
    // 2. normalize
    normalize_pc(R.points, N);
    // 3. CenterShift(apply_z=True) on a working coord copy (model.transform mutates coord)
    std::vector<float> coord = R.points;
    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (int i=0;i<N;i++) for (int d=0;d<3;d++){ float v=coord[(size_t)i*3+d]; mn[d]=std::min(mn[d],v); mx[d]=std::max(mx[d],v); }
    float shift[3] = { (mn[0]+mx[0])*0.5f, (mn[1]+mx[1])*0.5f, mn[2] };
    for (int i=0;i<N;i++) for (int d=0;d<3;d++) coord[(size_t)i*3+d]-=shift[d];

    // 4. GridSample 0.005
    const float gs = 0.005f;
    std::vector<int32_t> gridc((size_t)N*3);
    int32_t gmin[3]={INT32_MAX,INT32_MAX,INT32_MAX};
    for (int i=0;i<N;i++) for (int d=0;d<3;d++){
        float sc = coord[(size_t)i*3+d]/gs;
        int32_t fl = (int32_t)std::floor(sc);
        gridc[(size_t)i*3+d]=fl;
        gmin[d]=std::min(gmin[d],fl);
    }
    for (int i=0;i<N;i++) for (int d=0;d<3;d++) gridc[(size_t)i*3+d]-=gmin[d];

    // key per point
    std::vector<uint64_t> key(N);
    for (int i=0;i<N;i++)
        key[i]=fnv3(gridc[(size_t)i*3+0],gridc[(size_t)i*3+1],gridc[(size_t)i*3+2]);

    // argsort by key (stable, matches np.argsort default quicksort for unique mapping
    // we only need: group by key, first-in-sorted-order representative).
    std::vector<int> order(N);
    for (int i=0;i<N;i++) order[i]=i;
    std::sort(order.begin(), order.end(), [&](int a,int b){
        if (key[a]!=key[b]) return key[a]<key[b];
        return a<b;
    });
    // assign unique-voxel ids in sorted-key order; representative = first point of each group.
    R.inverse.assign(N, 0);
    std::vector<int> reps;            // representative point index per unique voxel
    reps.reserve(80000);
    int64_t uid = -1;
    uint64_t prev = 0; bool first = true;
    for (int s = 0; s < N; s++) {
        int p = order[s];
        if (first || key[p] != prev) { uid++; reps.push_back(p); prev = key[p]; first = false; }
        R.inverse[p] = uid;
    }
    int M0 = (int)reps.size();
    R.M0 = M0;
    // 5/6. grid_coord + feat for representatives.
    R.grid.resize((size_t)M0*3);
    R.feat.resize((size_t)M0*9);
    const float color = 1.0f/255.0f;
    for (int m=0;m<M0;m++){
        int p = reps[m];
        for (int d=0;d<3;d++) R.grid[(size_t)m*3+d]=gridc[(size_t)p*3+d];
        float* fd = R.feat.data()+(size_t)m*9;
        fd[0]=coord[(size_t)p*3+0]; fd[1]=coord[(size_t)p*3+1]; fd[2]=coord[(size_t)p*3+2];
        fd[3]=color; fd[4]=color; fd[5]=color;
        fd[6]=R.normals[(size_t)p*3+0]; fd[7]=R.normals[(size_t)p*3+1]; fd[8]=R.normals[(size_t)p*3+2];
    }
    return R;
}

} // namespace p3sam
