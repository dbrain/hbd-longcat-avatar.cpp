#pragma once

// CUDA companion for the data-parallel portion of native atlas baking.  Chart
// creation remains a bounded CPU/CuMesh decision; this stage turns the packed
// UV triangles into the position/normal/mask buffers consumed by the existing
// PBR sampler.  Keeping the API in terms of host vectors makes the CPU path a
// straightforward fallback and avoids giving the atlas writer CUDA ownership.

#include <cstdint>
#include <string>
#include <vector>

namespace texatlas_cuda {

struct RasterStats {
    double upload_seconds = 0.0;
    double kernel_seconds = 0.0;
    double download_seconds = 0.0;
};

// `faces` is uint32 triangle indices, `uv_px` is pixel-space UV [V,2], and
// vertices/normals are [V,3].  The output layout matches tex_atlas.hpp:
// position/normal are [W*H,3], mask is [W*H].  Surface-chart ownership stays
// host-side because it is a chart-local postprocess tag, not a material value.
bool raster_uv(const std::vector<float>& vertices,
               const std::vector<float>& normals,
               const std::vector<uint32_t>& faces,
               const std::vector<float>& uv_px,
               int width, int height, int supersample,
               std::vector<float>& position,
               std::vector<float>& normal,
               std::vector<uint8_t>& mask,
               RasterStats* stats,
               std::string* error);

} // namespace texatlas_cuda
