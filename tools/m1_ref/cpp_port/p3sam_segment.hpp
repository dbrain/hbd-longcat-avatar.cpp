// p3sam_segment.hpp — full native P3-SAM entry: mesh -> per-face part ids.
//   preprocess (sample/normalize/grid) -> run_sonata (banked, validated)
//   -> postprocess (heads + NMS + face vote).
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "p3sam_heads.hpp"
#include "p3sam_sonata.hpp"
#include "p3sam_preprocess.hpp"
#include "p3sam_postprocess.hpp"

namespace p3sam {

inline std::vector<int64_t> segment_mesh(const float* verts, int V,
                                         const int64_t* faces, int F,
                                         const std::string& weights_dir,
                                         uint64_t seed = 42) {
    Weights W(weights_dir);
    PreOut pre = preprocess(verts, V, faces, F, seed);
    std::vector<float> feats_N = run_sonata(W, pre.grid.data(), pre.feat.data(), pre.M0,
                                            pre.inverse.data(), pre.Nin, nullptr);
    return postprocess(feats_N.data(), pre.Nin, pre.points.data(), pre.points_org.data(),
                       verts, V, faces, F, W, seed);
}

} // namespace p3sam
