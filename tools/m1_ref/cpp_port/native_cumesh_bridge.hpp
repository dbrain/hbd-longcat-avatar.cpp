#pragma once

#include <cstdint>
#include <vector>

namespace native_cumesh {

struct ClusterMesh {
    std::vector<float> verts;
    std::vector<uint32_t> faces;
    std::vector<int> vmap;
};

// Feature-preserving cleanup for a decoded Pixal/O-Voxel dual-grid surface.  Unlike the production
// occupancy marching-cubes fallback, this never resamples the surface: it removes degenerates and
// duplicate faces, splits non-manifold fans, fills only small boundary loops, and makes winding
// consistent.  It is a native CUDA implementation used for the clay-only postprocess A/B.
struct CleanReport {
    int input_vertices = 0;
    int input_faces = 0;
    int output_vertices = 0;
    int output_faces = 0;
};

void clean_feature_preserving(const std::vector<float>& vertices,
                              const std::vector<int64_t>& faces,
                              std::vector<float>& out_vertices,
                              std::vector<int64_t>& out_faces,
                              CleanReport* report = nullptr,
                              float max_hole_perimeter = 3e-2f);

std::vector<ClusterMesh> compute_clusters(const std::vector<float>& vertices,
                                          const std::vector<int64_t>& faces,
                                          float threshold_cone_half_angle_rad,
                                          int refine_iterations,
                                          int global_iterations,
                                          float smooth_strength,
                                          float area_penalty_weight,
                                          float perimeter_area_ratio_weight);

void simplify_to_faces(const std::vector<float>& vertices,
                       const std::vector<int64_t>& faces,
                       int target_num_faces,
                       std::vector<float>& out_vertices,
                       std::vector<int64_t>& out_faces,
                       float thresh = 1e-8f,
                       float lambda_edge_length = 1e-2f,
                       float lambda_skinny = 1e-3f);

} // namespace native_cumesh
