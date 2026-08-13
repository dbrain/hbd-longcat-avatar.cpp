// p3sam_segment.hpp — full native P3-SAM entry: mesh -> per-face part ids.
//   preprocess (sample/normalize/grid) -> run_sonata (banked, validated)
//   -> postprocess (heads + NMS + face vote).
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include "p3sam_heads.hpp"
#include "p3sam_sonata.hpp"
#include "p3sam_preprocess.hpp"
#include "p3sam_postprocess.hpp"

namespace p3sam {

inline std::vector<int64_t> segment_mesh(const float* verts, int V,
                                         const int64_t* faces, int F,
                                         const std::string& weights_dir,
                                         uint64_t seed = 42) {
    // P3SAM_TIMING=1 -> per-phase wall clock (preprocess / Sonata encoder / postprocess).
    const bool tm = std::getenv("P3SAM_TIMING") != nullptr;
    auto now = []{ return std::chrono::steady_clock::now(); };
    auto ms  = [](auto a, auto b){ return std::chrono::duration<double>(b-a).count(); };
    auto t0 = now();
    Weights W(weights_dir);
    auto t1 = now();
    PreOut pre = preprocess(verts, V, faces, F, seed);
    auto t2 = now();
    std::vector<float> feats_N = run_sonata(W, pre.grid.data(), pre.feat.data(), pre.M0,
                                            pre.inverse.data(), pre.Nin, nullptr);
    auto t3 = now();
    auto out = postprocess(feats_N.data(), pre.Nin, pre.points.data(), pre.points_org.data(),
                           verts, V, faces, F, W, seed);
    auto t4 = now();
    if (tm) std::fprintf(stderr,
        "[p3sam-timing] weights %.2fs | preprocess %.2fs | sonata(CPU encoder) %.2fs | postprocess(heads+vote) %.2fs | TOTAL %.2fs\n",
        ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4), ms(t0,t4));
    return out;
}

} // namespace p3sam
