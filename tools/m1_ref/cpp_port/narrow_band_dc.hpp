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

namespace nbdc {

// Dual-contour `in_*` onto the narrow band of its own unsigned distance field.
//   resolution : final grid (power of two >= 32; 1024 = the Pixal3D generation lattice)
//   band       : narrow-band width in voxels (Pixal3D postprocess default 1)
// Returns false (leaving outputs untouched) on invalid input or a CUDA failure; the reason is
// printed to stderr. Caller keeps the input mesh — the DC output is a NEW surface, the input is
// still the volume-aligned shell the texture bake needs.
bool remesh(const std::vector<float>& in_verts, const std::vector<int64_t>& in_faces,
            int resolution, float band,
            std::vector<float>& out_verts, std::vector<int64_t>& out_faces,
            bool verbose = true);

}  // namespace nbdc
