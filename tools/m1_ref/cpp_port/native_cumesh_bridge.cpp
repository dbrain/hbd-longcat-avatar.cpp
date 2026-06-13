#include "native_cumesh_bridge.hpp"

#include "../../../thirdparty/cumesh_native/src/cumesh.h"

#include <cstdio>
#include <stdexcept>
#include <tuple>

namespace native_cumesh {

std::vector<ClusterMesh> compute_clusters(const std::vector<float>& vertices,
                                          const std::vector<int64_t>& faces,
                                          float threshold_cone_half_angle_rad,
                                          int refine_iterations,
                                          int global_iterations,
                                          float smooth_strength,
                                          float area_penalty_weight,
                                          float perimeter_area_ratio_weight) {
    if (vertices.empty() || faces.empty()) return {};
    if (vertices.size() % 3 || faces.size() % 3) {
        throw std::runtime_error("native_cumesh::compute_clusters received malformed mesh arrays");
    }

    cumesh::CuMesh mesh;
    mesh.init_host(vertices.data(), vertices.size() / 3, faces.data(), faces.size() / 3);
    mesh.remove_degenerate_faces(1e-24f, 1e-12f);
    mesh.compute_charts(threshold_cone_half_angle_rad, refine_iterations, global_iterations,
                        smooth_strength, area_penalty_weight, perimeter_area_ratio_weight);

    std::vector<float> clean_vertices;
    std::vector<int64_t> clean_faces;
    mesh.read_host(clean_vertices, clean_faces);
    cumesh::HostAtlasCharts charts = mesh.read_atlas_charts_host();

    std::vector<ClusterMesh> out((size_t)charts.num_charts);
    for (int c = 0; c < charts.num_charts; c++) {
        const int v0 = charts.chart_vertex_offset[(size_t)c];
        const int v1 = charts.chart_vertex_offset[(size_t)c + 1];
        const int f0 = charts.chart_face_offset[(size_t)c];
        const int f1 = charts.chart_face_offset[(size_t)c + 1];
        ClusterMesh& cm = out[(size_t)c];
        cm.vmap.reserve((size_t)(v1 - v0));
        cm.verts.reserve((size_t)(v1 - v0) * 3);
        for (int i = v0; i < v1; i++) {
            int ov = charts.chart_vertex_map[(size_t)i];
            cm.vmap.push_back(ov);
            cm.verts.push_back(clean_vertices[(size_t)ov * 3 + 0]);
            cm.verts.push_back(clean_vertices[(size_t)ov * 3 + 1]);
            cm.verts.push_back(clean_vertices[(size_t)ov * 3 + 2]);
        }
        cm.faces.reserve((size_t)(f1 - f0) * 3);
        for (int i = f0; i < f1; i++) {
            cm.faces.push_back((uint32_t)(charts.chart_faces[(size_t)i * 3 + 0] - v0));
            cm.faces.push_back((uint32_t)(charts.chart_faces[(size_t)i * 3 + 1] - v0));
            cm.faces.push_back((uint32_t)(charts.chart_faces[(size_t)i * 3 + 2] - v0));
        }
    }
    return out;
}

void simplify_to_faces(const std::vector<float>& vertices,
                       const std::vector<int64_t>& faces,
                       int target_num_faces,
                       std::vector<float>& out_vertices,
                       std::vector<int64_t>& out_faces,
                       float thresh,
                       float lambda_edge_length,
                       float lambda_skinny) {
    if (target_num_faces <= 0) {
        throw std::runtime_error("native_cumesh::simplify_to_faces target must be positive");
    }
    cumesh::CuMesh mesh;
    mesh.init_host(vertices.data(), vertices.size() / 3, faces.data(), faces.size() / 3);
    int num_faces = mesh.num_faces();
    if (num_faces > target_num_faces) {
        while (true) {
            int new_num_vertices = 0;
            int new_num_faces = 0;
            std::tie(new_num_vertices, new_num_faces) =
                mesh.simplify_step(lambda_edge_length, lambda_skinny, thresh, false);
            (void)new_num_vertices;
            if (new_num_faces <= target_num_faces) break;
            int del_num_faces = num_faces - new_num_faces;
            if (num_faces > 0 && (float)del_num_faces / (float)num_faces < 1e-2f) thresh *= 10.f;
            if (new_num_faces >= num_faces) thresh *= 10.f;
            num_faces = new_num_faces;
        }
    }
    mesh.read_host(out_vertices, out_faces);
}

} // namespace native_cumesh
