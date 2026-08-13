// test_p3sam_segment.cpp — two modes:
//   golden : run ONLY postprocess on banked feats_N -> face_ids; report
//            #parts, sorted part sizes, and best-match mean part-IoU vs
//            goldens/face_ids.npy.
//   full   : run full segment_mesh(verts,faces) -> face_ids; report #parts,
//            sizes, timing.
#include <pmmintrin.h>   // _MM_SET_FLUSH_ZERO / DENORMALS_ZERO
#include <xmmintrin.h>
#include "../../sparse_spike/npy.hpp"
#include "p3sam_segment.hpp"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <map>
#include <vector>
#include <algorithm>
#include <string>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/p3sam_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/p3sam_goldens/weights_npy";

static void part_summary(const std::vector<int64_t>& ids, const char* tag) {
    std::map<int64_t,long> cnt;
    for (auto v : ids) cnt[v]++;
    std::vector<std::pair<int64_t,long>> v(cnt.begin(), cnt.end());
    std::sort(v.begin(), v.end(), [](auto&a,auto&b){return a.second>b.second;});
    int nparts = 0; for (auto&p:v) if (p.first>=0) nparts++;
    printf("[%s] %d labeled parts (%zu unique incl -1/-2), sizes(desc): ", tag, nparts, v.size());
    int shown=0;
    for (auto&p:v){ printf("%lld:%ld ", (long long)p.first, p.second); if(++shown>=25){printf("...");break;} }
    printf("\n");
}

// best-match mean part-IoU: for each golden part, the best-overlapping pred part
// IoU; averaged weighted by golden part size.
static double mean_part_iou(const std::vector<int64_t>& gold, const std::vector<int64_t>& pred) {
    int F = (int)gold.size();
    // collect parts
    std::map<int64_t,long> gsz, psz;
    for (int i=0;i<F;i++){ if(gold[i]>=0) gsz[gold[i]]++; if(pred[i]>=0) psz[pred[i]]++; }
    // overlap counts gpart x ppart
    std::map<std::pair<int64_t,int64_t>,long> ov;
    for (int i=0;i<F;i++){ if(gold[i]>=0 && pred[i]>=0) ov[{gold[i],pred[i]}]++; }
    double wsum=0, isum=0;
    for (auto& g : gsz) {
        int64_t gp=g.first; long ga=g.second;
        double best=0;
        for (auto& p : psz) {
            int64_t pp=p.first;
            auto it=ov.find({gp,pp});
            if (it==ov.end()) continue;
            long inter=it->second;
            long uni=ga + p.second - inter;
            double iou = uni>0 ? (double)inter/uni : 0.0;
            if (iou>best) best=iou;
        }
        isum += best*ga; wsum += ga;
    }
    return wsum>0 ? isum/wsum : 0.0;
}

int main(int argc, char** argv) {
    // Flush denormals to zero: the heads' GELU/sigmoid (std::erf/std::exp) hit
    // huge denormal-FP slowdowns over 100k x 400 prompts otherwise. FTZ/DAZ
    // matches GPU behaviour and does not affect the boolean masks (>0 test).
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    #ifdef _OPENMP
    #pragma omp parallel
    {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
    #endif

    std::string mode = argc>1 ? argv[1] : "golden";

    if (mode == "golden") {
        NpyArray feats = npy_load(std::string(GDIR)+"/feats_N.npy");
        NpyArray pts   = npy_load(std::string(GDIR)+"/points.npy");
        NpyArray ptso  = npy_load(std::string(GDIR)+"/points_org.npy");
        NpyArray verts = npy_load(std::string(GDIR)+"/verts.npy");
        NpyArray faces = npy_load(std::string(GDIR)+"/faces.npy");
        NpyArray gold  = npy_load(std::string(GDIR)+"/face_ids.npy");
        int N = (int)pts.shape[0];
        int V = (int)verts.shape[0];
        int F = (int)faces.shape[0];
        printf("golden mode: N=%d V=%d F=%d feats=%lldx%lld\n", N, V, F,
               (long long)feats.shape[0], (long long)feats.shape[1]);

        p3sam::Weights W(WDIR);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int64_t> face_ids = p3sam::postprocess(
            feats.f32(), N, pts.f32(), ptso.f32(),
            verts.f32(), V, faces.i64(), F, W, 42);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1-t0).count();

        part_summary(face_ids, "pred");
        std::vector<int64_t> gv(gold.i64(), gold.i64()+gold.numel());
        part_summary(gv, "gold");
        double miou = mean_part_iou(gv, face_ids);
        printf("postprocess %.2fs | mean best-match part-IoU vs golden = %.4f\n", sec, miou);
        return 0;
    }

    if (mode == "full") {
        NpyArray verts = npy_load(std::string(GDIR)+"/verts.npy");
        NpyArray faces = npy_load(std::string(GDIR)+"/faces.npy");
        int V = (int)verts.shape[0];
        int F = (int)faces.shape[0];
        printf("full mode: V=%d F=%d\n", V, F);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int64_t> face_ids =
            p3sam::segment_mesh(verts.f32(), V, faces.i64(), F, WDIR, 42);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1-t0).count();
        part_summary(face_ids, "pred");
        printf("segment_mesh %.2fs\n", sec);
        return 0;
    }

    fprintf(stderr, "usage: %s [golden|full]\n", argv[0]);
    return 1;
}
