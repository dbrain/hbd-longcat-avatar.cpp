// Narrow-band UDF dual contouring (Pixal3D / CuMesh `remesh_narrow_band_dc` equivalent), as a
// linkable entry point rather than a probe-only main.
//
// This is the PARITY remesh: run it on the raw O-Voxel dual-grid decoder mesh (image_to_rig
// --no-clean) and the result is Python's smooth watertight coarse surface. Running it on the
// MC-solid binary-occupancy mesh instead keeps the staircase — see memory
// project_image_to_rig_coarse_parity_ovoxel_dc.
//
// The implementation lives in narrow_band_dc_cuda.cu (native BVH + CUDA UDF queries, no
// Python/Torch/CuBVH). Host callers only need this declaration, so a plain g++ TU
// (image_to_rig.cpp) can link it alongside the nvcc-compiled object.
#pragma once
#include <cstdint>
#include <vector>

#include "solid_field.hpp"

namespace nbdc {

// Dual-contour `in_*` onto the narrow band of its own unsigned distance field.
//   resolution : final grid. Must be in [32, 2047] (11-bit sparse cell keys, and grid vertices
//                reach `resolution` inclusive -- 2048 would alias) and must halve down to a grid
//                of <= 128 for the coarse level of the refinement ladder. Every multiple of 16 in
//                range qualifies: 1024 -> ladder starts at 32, 1536 -> 48. NOT power-of-two only.
//   band       : narrow-band width in voxels (Pixal3D postprocess default 1)
//   interior   : OPTIONAL inside/outside oracle (solid_field.hpp).  nullptr == the historical
//                UNSIGNED behaviour, bit-for-bit -- do not change that, res-1024 parity depends on
//                it.  When supplied, the field is SIGNED before contouring, which is the difference
//                between delivering a two-walled shrink-wrap ENVELOPE and delivering a solid:
//
//                  f(x) = udf(x) - eps        outside the model, and wherever udf(x) <= eps
//                       = -(udf(x) - eps)     inside the model with udf(x) > eps
//
//                The two agree exactly whenever udf <= eps, so every zero crossing of the OUTWARD
//                wall keeps the values it had and the wall is unmoved; the only crossings that
//                disappear are the ones that produced the INWARD wall, because the deep interior
//                stops being positive.  Cells that held crossings from both walls no longer average
//                the two, so the pinches that made those cells non-manifold go away as well.
// Returns false (leaving outputs untouched) on invalid input or a CUDA failure; the reason is
// printed to stderr. Caller keeps the input mesh — the DC output is a NEW surface, the input is
// still the volume-aligned shell the texture bake needs.
bool remesh(const std::vector<float>& in_verts, const std::vector<int64_t>& in_faces,
            int resolution, float band,
            std::vector<float>& out_verts, std::vector<int64_t>& out_faces,
            bool verbose = true,
            const solidfield::SolidField* interior = nullptr);

}  // namespace nbdc
