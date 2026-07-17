// p3sam_postprocess.hpp — native C++ port of mesh_sam postprocess
// (auto_mask_no_postprocess.py L481-736). Given backbone feats_N[N,512],
// normalized points[N,3], points_org[N,3], and the mesh: FPS-sample prompts,
// run the seg/iou heads, NMS-cluster the masks, merge, patch, and project to
// a per-face label via nearest-neighbour vote (uniform spatial hash grid).
//
// FPS and the GridSample representative draw differ from the Python RNG, so the
// partition is *sensible* but not bit-identical to the golden.
#pragma once
#include "p3sam_heads.hpp"        // p3sam::Weights, run_heads, HeadOut
#ifdef P3SAM_USE_CUDA
#include "p3sam_heads_gpu.hpp"    // p3gpu::HeadsGpu (cuBLAS heads)
#endif
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace p3sam {

// ---- Farthest point sampling (greedy, start idx 0) ----
inline std::vector<int> fps(const float* pts, int N, int K) {
    std::vector<int> sel; sel.reserve(K);
    std::vector<float> dist(N, 1e30f);
    int cur = 0;
    for (int k = 0; k < K; k++) {
        sel.push_back(cur);
        const float* c = pts + (size_t)cur*3;
        float best = -1.f; int bidx = 0;
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            float lbest = -1.f; int lidx = 0;
            #ifdef _OPENMP
            #pragma omp for schedule(static) nowait
            #endif
            for (int i = 0; i < N; i++) {
                const float* p = pts + (size_t)i*3;
                float dx=p[0]-c[0], dy=p[1]-c[1], dz=p[2]-c[2];
                float d = dx*dx+dy*dy+dz*dz;
                if (d < dist[i]) dist[i] = d;
                if (dist[i] > lbest) { lbest = dist[i]; lidx = i; }
            }
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            { if (lbest > best) { best = lbest; bidx = lidx; } }
        }
        cur = bidx;
    }
    return sel;
}

// ---- mask IoU helpers over boolean masks stored as uint8 [N] ----
inline double cal_iou(const uint8_t* a, const uint8_t* b, int N) {
    long inter=0, uni=0;
    for (int i=0;i<N;i++){ int x=a[i], y=b[i]; inter += (x&y); uni += (x|y); }
    return uni>0 ? (double)inter/(double)uni : 0.0;
}
inline double cal_single_iou(const uint8_t* a, const uint8_t* b, int N) {
    long inter=0, sa=0;
    for (int i=0;i<N;i++){ int x=a[i], y=b[i]; inter += (x&y); sa += x; }
    return sa>0 ? (double)inter/(double)sa : 0.0;
}
// 3D AABB IoU of two point subsets selected by masks a,b over points[N,3].
inline double cal_bbox_iou(const float* pts, const uint8_t* a, const uint8_t* b, int N) {
    float amn[3]={1e30f,1e30f,1e30f}, amx[3]={-1e30f,-1e30f,-1e30f};
    float bmn[3]={1e30f,1e30f,1e30f}, bmx[3]={-1e30f,-1e30f,-1e30f};
    bool ha=false, hb=false;
    for (int i=0;i<N;i++){
        const float* p=pts+(size_t)i*3;
        if (a[i]){ ha=true; for(int d=0;d<3;d++){amn[d]=std::min(amn[d],p[d]);amx[d]=std::max(amx[d],p[d]);} }
        if (b[i]){ hb=true; for(int d=0;d<3;d++){bmn[d]=std::min(bmn[d],p[d]);bmx[d]=std::max(bmx[d],p[d]);} }
    }
    if (!ha || !hb) return 0.0;
    double ixmin=std::max(amn[0],bmn[0]), iymin=std::max(amn[1],bmn[1]), izmin=std::max(amn[2],bmn[2]);
    double ixmax=std::min(amx[0],bmx[0]), iymax=std::min(amx[1],bmx[1]), izmax=std::min(amx[2],bmx[2]);
    if (ixmin>=ixmax || iymin>=iymax || izmin>=izmax) return 0.0;
    double iv=(ixmax-ixmin)*(iymax-iymin)*(izmax-izmin);
    double v1=(double)(amx[0]-amn[0])*(amx[1]-amn[1])*(amx[2]-amn[2]);
    double v2=(double)(bmx[0]-bmn[0])*(bmx[1]-bmn[1])*(bmx[2]-bmn[2]);
    double uni=v1+v2-iv;
    return uni>0 ? iv/uni : 0.0;
}

inline std::vector<int64_t> postprocess(
        const float* feats_N, int N, const float* points, const float* points_org,
        const float* verts, int V, const int64_t* faces, int F,
        Weights& W, uint64_t seed = 42) {
    (void)V; (void)seed;
    // prompt_num=400 is the P3-SAM default. The seg/iou heads are the entire CPU
    // cost (~200s CPU / prompt over N=100k via the validated scalar linear()), so
    // P3SAM_PROMPTS lets a validation run use fewer prompts; production = 400.
    int prompt_num = 400;
    if (const char* e = std::getenv("P3SAM_PROMPTS")) { int v = std::atoi(e); if (v > 0) prompt_num = v; }
    const int bs = 32;

    // 1. FPS prompts from normalized points.
    std::vector<int> fps_idx = fps(points, N, prompt_num);
    std::vector<float> prompts((size_t)prompt_num*3);
    for (int k=0;k<prompt_num;k++)
        for (int d=0;d<3;d++) prompts[(size_t)k*3+d]=points[(size_t)fps_idx[k]*3+d];

    const bool verbose = std::getenv("P3SAM_VERBOSE") != nullptr;
    // 2. run heads in batches -> per-prompt boolean mask (best iou channel) + iou.
    std::vector<uint8_t> mask_res((size_t)prompt_num*N);  // [K,N] row-major per prompt
    std::vector<float>   iou_res(prompt_num);
    double th0 = (double)clock()/CLOCKS_PER_SEC;
#ifdef P3SAM_USE_CUDA
    p3gpu::HeadsGpu hg(W.dir, feats_N, points, N);   // upload feats_N+points once (resident)
#endif
    for (int start=0; start<prompt_num; start+=bs) {
        int kb = std::min(bs, prompt_num-start);
#ifdef P3SAM_USE_CUDA
        HeadOut H = hg.run(prompts.data()+(size_t)start*3, kb);
#else
        HeadOut H = run_heads(W, feats_N, points, N, prompts.data()+(size_t)start*3, kb);
#endif
        if (verbose) std::fprintf(stderr, "  [heads] %d/%d prompts (%.0fs cpu)\n",
                     start+kb, prompt_num, (double)clock()/CLOCKS_PER_SEC - th0);
        // H.seg2 is [kb,N,3] logits; H.iou is [kb,3] (already sigmoided).
        for (int j=0;j<kb;j++) {
            const float* io = H.iou.data()+(size_t)j*3;
            int ch = 0; float bv = io[0];
            if (io[1] > bv){ bv=io[1]; ch=1; }
            if (io[2] > bv){ bv=io[2]; ch=2; }
            uint8_t* mr = mask_res.data()+(size_t)(start+j)*N;
            const float* s2 = H.seg2.data()+(size_t)j*N*3;
            for (int n=0;n<N;n++) {
                float logit = s2[(size_t)n*3+ch];
                mr[n] = (logit > 0.0f) ? 1 : 0;   // sigmoid(x)>0.5 <=> x>0
            }
            iou_res[start+j] = bv;
        }
    }

    // 3. sort prompts by iou desc -> mask_sorted order.
    std::vector<int> ord(prompt_num);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a,int b){ return iou_res[a] > iou_res[b]; });
    // mask_sorted[s] = mask_res[ord[s]]; index masks by sorted position s in [0,prompt_num).
    auto MS = [&](int s)->const uint8_t* { return mask_res.data()+(size_t)ord[s]*N; };

    // 4. NMS: greedy clustering in sorted (iou desc) order. cluster keys = sorted index s.
    //    For each i, compare to existing cluster reps; if cal_iou>0.9 join first match else new rep.
    std::vector<int> cluster_keys;            // representative sorted-indices, insertion order
    std::vector<int> cluster_size;            // member count per cluster
    cluster_keys.reserve(prompt_num);
    for (int i=0;i<prompt_num;i++) {
        const uint8_t* mi = MS(i);
        bool joined=false;
        for (size_t c=0;c<cluster_keys.size();c++) {
            if (cal_iou(mi, MS(cluster_keys[c]), N) > 0.9) { cluster_size[c]++; joined=true; break; }
        }
        if (!joined) { cluster_keys.push_back(i); cluster_size.push_back(1); }
    }

    // 5. filter clusters with > 2 members.
    std::vector<int> filt;
    for (size_t c=0;c<cluster_keys.size();c++)
        if (cluster_size[c] > 2) filt.push_back(cluster_keys[c]);

    // 6. merge again by bbox-iou>0.5 (j into i, i<j over filtered order).
    int fn = (int)filt.size();
    std::vector<char> is_union(fn, 0);
    std::vector<int> cluster2;                 // surviving sorted-index reps (cluster2 keys)
    for (int i=0;i<fn;i++) {
        if (is_union[i]) continue;
        cluster2.push_back(filt[i]);
        const uint8_t* mi = MS(filt[i]);
        for (int j=i+1;j<fn;j++) {
            if (is_union[j]) continue;
            if (cal_bbox_iou(points, MS(filt[j]), mi, N) > 0.5) is_union[j]=1;
        }
    }

    // 7. no_mask = points covered by no cluster2 mask.
    std::vector<uint8_t> no_mask(N, 1);
    for (int c : cluster2) { const uint8_t* m = MS(c); for (int n=0;n<N;n++) if (m[n]) no_mask[n]=0; }

    // 8. patch: for masks NOT yet a cluster, if cal_single_iou(mask,no_mask)>0.7 add + clear.
    //    iterate over ALL sorted masks i in order (python: range(len(mask_sorted))).
    {
        std::vector<char> in_c2(prompt_num, 0);
        for (int c : cluster2) in_c2[c] = 1;
        for (int i=0;i<prompt_num;i++) {
            if (in_c2[i]) continue;
            const uint8_t* m = MS(i);
            if (cal_single_iou(m, no_mask.data(), N) > 0.7) {
                cluster2.push_back(i);
                for (int n=0;n<N;n++) if (m[n]) no_mask[n]=0;
            }
        }
    }

    // 9. final clusters sorted by area desc; assign result_mask in that order
    //    (later/smaller overwrite earlier/larger — faithful to python loop).
    std::vector<std::pair<int,long>> areas;  // (sorted-index key, area)
    for (int c : cluster2) {
        const uint8_t* m = MS(c); long a=0; for (int n=0;n<N;n++) a+=m[n];
        areas.push_back({c,a});
    }
    std::stable_sort(areas.begin(), areas.end(),
                     [](const std::pair<int,long>& x, const std::pair<int,long>& y){ return x.second > y.second; });

    std::vector<int64_t> result_mask(N, -1);
    for (auto& pr : areas) {
        int c = pr.first;
        const uint8_t* m = MS(c);
        for (int n=0;n<N;n++) if (m[n]) result_mask[n] = c;   // cluster id = sorted-index key
    }

    if (verbose) std::fprintf(stderr, "  [clusters] final=%zu parts; starting per-face NN vote over %d faces\n",
                 areas.size(), F);

    // 10. per-face label via NN of 10 barycentric samples to labeled points_org.
    //     Build labeled set (result_mask>=0) and a uniform spatial hash grid over it.
    std::vector<int> valid;
    for (int n=0;n<N;n++) if (result_mask[n]>=0) valid.push_back(n);
    int nv = (int)valid.size();

    std::vector<int64_t> face_ids(F, -2);  // default -2 (python argmax(bincount)-2 of all -2 -> -2)
    if (nv == 0) return face_ids;

    // grid bounds over labeled points_org
    float bmn[3]={1e30f,1e30f,1e30f}, bmx[3]={-1e30f,-1e30f,-1e30f};
    for (int vi=0;vi<nv;vi++){ const float* p=points_org+(size_t)valid[vi]*3;
        for(int d=0;d<3;d++){bmn[d]=std::min(bmn[d],p[d]);bmx[d]=std::max(bmx[d],p[d]);} }
    float ext = std::max(std::max(bmx[0]-bmn[0],bmx[1]-bmn[1]),bmx[2]-bmn[2]);
    if (ext<=0) ext=1.f;
    // ~ aim for a few points per cell: cells ≈ nv -> res = cbrt(nv)
    int res = std::max(1, (int)std::cbrt((double)nv));
    float cell = ext / res;
    if (cell<=0) cell = 1.f;
    auto cidx = [&](float x,float y,float z)->int64_t {
        int gx=(int)std::floor((x-bmn[0])/cell), gy=(int)std::floor((y-bmn[1])/cell), gz=(int)std::floor((z-bmn[2])/cell);
        gx=std::min(std::max(gx,0),res); gy=std::min(std::max(gy,0),res); gz=std::min(std::max(gz,0),res);
        return ((int64_t)gx*(res+1)+gy)*(res+1)+gz;
    };
    std::unordered_map<int64_t,std::vector<int>> grid_map;
    grid_map.reserve(nv*2);
    for (int vi=0;vi<nv;vi++){ const float* p=points_org+(size_t)valid[vi]*3;
        grid_map[cidx(p[0],p[1],p[2])].push_back(vi); }

    const int pre_face = 10;
    std::mt19937_64 base_rng(seed);  // (face sampling RNG; python uses global np.random)
    (void)base_rng;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ (uint64_t)
        #ifdef _OPENMP
            omp_get_thread_num()
        #else
            0
        #endif
        );
        std::uniform_real_distribution<double> U(0.0,1.0);
        std::vector<int64_t> votes(64);
        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 1024)
        #endif
        for (int f=0; f<F; f++) {
            int64_t a=faces[(size_t)f*3+0], b=faces[(size_t)f*3+1], c=faces[(size_t)f*3+2];
            const float* v0=verts+(size_t)a*3; const float* v1=verts+(size_t)b*3; const float* v2=verts+(size_t)c*3;
            // per-sample label; majority vote of (label+2) like np.bincount.
            // We accumulate labels and pick the mode (cluster ids are sparse -> map).
            std::unordered_map<int64_t,int> tally;
            for (int s=0;s<pre_face;s++) {
                double u=std::sqrt(U(rng)), vv=U(rng);
                double w0=1.0-u, w1=u*(1.0-vv), w2=u*vv;
                float px=(float)(w0*v0[0]+w1*v1[0]+w2*v2[0]);
                float py=(float)(w0*v0[1]+w1*v1[1]+w2*v2[1]);
                float pz=(float)(w0*v0[2]+w1*v1[2]+w2*v2[2]);
                // NN over labeled points: search the cell + 1-ring neighbours.
                int gx=(int)std::floor((px-bmn[0])/cell), gy=(int)std::floor((py-bmn[1])/cell), gz=(int)std::floor((pz-bmn[2])/cell);
                gx=std::min(std::max(gx,0),res); gy=std::min(std::max(gy,0),res); gz=std::min(std::max(gz,0),res);
                float bestd=1e30f; int bestvi=-1;
                int ring=1;
                while (bestvi<0 && ring<=res+1) {
                    for (int dx=-ring;dx<=ring;dx++) for (int dy=-ring;dy<=ring;dy++) for (int dz=-ring;dz<=ring;dz++) {
                        // only scan the shell at distance==ring after first pass to avoid rescans
                        if (ring>1 && std::max(std::max(std::abs(dx),std::abs(dy)),std::abs(dz))!=ring) continue;
                        int cx=gx+dx, cy=gy+dy, cz=gz+dz;
                        if (cx<0||cy<0||cz<0||cx>res||cy>res||cz>res) continue;
                        int64_t key=((int64_t)cx*(res+1)+cy)*(res+1)+cz;
                        auto it=grid_map.find(key);
                        if (it==grid_map.end()) continue;
                        for (int vi : it->second) {
                            const float* p=points_org+(size_t)valid[vi]*3;
                            float ddx=p[0]-px, ddy=p[1]-py, ddz=p[2]-pz;
                            float dd=ddx*ddx+ddy*ddy+ddz*ddz;
                            if (dd<bestd){ bestd=dd; bestvi=vi; }
                        }
                    }
                    ring++;
                }
                if (bestvi<0) {
                    // fallback brute force (degenerate)
                    for (int vi=0;vi<nv;vi++){ const float* p=points_org+(size_t)valid[vi]*3;
                        float ddx=p[0]-px,ddy=p[1]-py,ddz=p[2]-pz; float dd=ddx*ddx+ddy*ddy+ddz*ddz;
                        if(dd<bestd){bestd=dd;bestvi=vi;} }
                }
                int64_t lab = result_mask[valid[bestvi]];
                tally[lab]++;
            }
            // majority vote (ties -> lowest (label+2) like argmax of bincount = first max)
            int64_t best_lab=-2; int best_cnt=-1;
            // emulate np.argmax(np.bincount(labels+2)): lowest index wins ties.
            // iterate candidate labels sorted ascending by (label+2)=label+2.
            std::vector<std::pair<int64_t,int>> cand(tally.begin(), tally.end());
            std::sort(cand.begin(), cand.end(),
                      [](const std::pair<int64_t,int>& x, const std::pair<int64_t,int>& y){ return x.first < y.first; });
            for (auto& kv : cand) if (kv.second > best_cnt) { best_cnt=kv.second; best_lab=kv.first; }
            face_ids[f] = best_lab;
        }
    }
    return face_ids;
}

} // namespace p3sam
