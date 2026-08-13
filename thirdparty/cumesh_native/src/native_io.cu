#include "cumesh.h"

namespace cumesh {

void CuMesh::init_host(const float* vertices_host, size_t num_vertices, const int64_t* faces_host, size_t num_faces) {
    std::vector<int> faces_i32(num_faces * 3);
    for (size_t i = 0; i < num_faces * 3; i++) faces_i32[i] = (int)faces_host[i];
    vertices.resize(num_vertices);
    faces.resize(num_faces);
    CUDA_CHECK(cudaMemcpy2D(vertices.ptr, sizeof(float3), vertices_host, sizeof(float) * 3,
        sizeof(float) * 3, num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(faces.ptr, sizeof(int3), faces_i32.data(), sizeof(int) * 3,
        sizeof(int) * 3, num_faces, cudaMemcpyHostToDevice));
}

void CuMesh::read_host(std::vector<float>& vertices_host, std::vector<int64_t>& faces_host) {
    vertices_host.resize(vertices.size * 3);
    std::vector<int> faces_i32(faces.size * 3);
    CUDA_CHECK(cudaMemcpy2D(vertices_host.data(), sizeof(float) * 3, vertices.ptr, sizeof(float3),
        sizeof(float) * 3, vertices.size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(faces_i32.data(), sizeof(int) * 3, faces.ptr, sizeof(int3),
        sizeof(int) * 3, faces.size, cudaMemcpyDeviceToHost));
    faces_host.resize(faces.size * 3);
    for (size_t i = 0; i < faces_i32.size(); i++) faces_host[i] = faces_i32[i];
}

HostAtlasCharts CuMesh::read_atlas_charts_host() {
    HostAtlasCharts out;
    out.num_charts = atlas_num_charts;
    out.chart_ids.resize(faces.size);
    out.chart_vertex_map.resize(atlas_chart_vertex_map.size);
    out.chart_faces.resize(atlas_chart_faces.size * 3);
    out.chart_vertex_offset.resize(atlas_chart_vertex_offset.size);
    out.chart_face_offset.resize(atlas_chart_faces_offset.size);
    if (!out.chart_ids.empty()) CUDA_CHECK(cudaMemcpy(out.chart_ids.data(), atlas_chart_ids.ptr,
        out.chart_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    if (!out.chart_vertex_map.empty()) CUDA_CHECK(cudaMemcpy(out.chart_vertex_map.data(), atlas_chart_vertex_map.ptr,
        out.chart_vertex_map.size() * sizeof(int), cudaMemcpyDeviceToHost));
    if (!out.chart_faces.empty()) CUDA_CHECK(cudaMemcpy2D(out.chart_faces.data(), sizeof(int) * 3,
        atlas_chart_faces.ptr, sizeof(int3), sizeof(int) * 3, atlas_chart_faces.size, cudaMemcpyDeviceToHost));
    if (!out.chart_vertex_offset.empty()) CUDA_CHECK(cudaMemcpy(out.chart_vertex_offset.data(), atlas_chart_vertex_offset.ptr,
        out.chart_vertex_offset.size() * sizeof(int), cudaMemcpyDeviceToHost));
    if (!out.chart_face_offset.empty()) CUDA_CHECK(cudaMemcpy(out.chart_face_offset.data(), atlas_chart_faces_offset.ptr,
        out.chart_face_offset.size() * sizeof(int), cudaMemcpyDeviceToHost));
    return out;
}

} // namespace cumesh
