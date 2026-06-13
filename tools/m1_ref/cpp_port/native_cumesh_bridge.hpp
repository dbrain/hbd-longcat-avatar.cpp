#pragma once

#include <cstdint>
#include <vector>

namespace native_cumesh {

struct ClusterMesh {
    std::vector<float> verts;
    std::vector<uint32_t> faces;
    std::vector<int> vmap;
};

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
