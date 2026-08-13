// Minimal glTF 2.0 binary (.glb) writer for a single untextured triangle mesh.
//   positions [V,3] f32 + indices [F,3] (uint32) -> one buffer, three bufferViews
//   (POSITION, NORMAL, indices), one mesh/primitive (mode 4), a double-sided PBR
//   material, one node, one scene. Web-ready (loads directly in <model-viewer> / three.js):
//   per-vertex normals are baked (else flat per-face shading) and the material is
//   doubleSided (the O-Voxel dual-grid mesh has inconsistent winding -> backface culling
//   would punch holes). Untextured (Phase A); M6 adds UV + a baseColor/metallic-roughness map.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cfloat>

namespace glb {

// area-weighted per-vertex normals (robust to the dual-grid winding; doubleSided handles flips)
inline std::vector<float> vertex_normals(const std::vector<float>& verts, const std::vector<int64_t>& faces) {
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    std::vector<float> n(V * 3, 0.f);
    for (size_t f = 0; f < F; f++) {
        int64_t a = faces[f*3], b = faces[f*3+1], c = faces[f*3+2];
        const float *pa = &verts[a*3], *pb = &verts[b*3], *pc = &verts[c*3];
        float e1[3] = {pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]};
        float e2[3] = {pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]};
        float fn[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        for (int64_t v : {a, b, c}) for (int d = 0; d < 3; d++) n[v*3+d] += fn[d];  // weight = 2*area
    }
    for (size_t v = 0; v < V; v++) {
        float* p = &n[v*3];
        float L = std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        if (L > 1e-20f) { p[0]/=L; p[1]/=L; p[2]/=L; } else { p[0]=0; p[1]=0; p[2]=1; }
    }
    return n;
}

// vcolors (optional): per-vertex base_color RGB [V*3] -> a COLOR_0 (VEC3 float) attribute (M6).
inline bool write_glb(const char* path, const std::vector<float>& verts,
                      const std::vector<int64_t>& faces, const std::vector<float>* vcolors = nullptr) {
    const uint32_t V = (uint32_t)(verts.size() / 3);
    const uint32_t F3 = (uint32_t)faces.size();            // index count = F*3
    const bool COL = (vcolors && vcolors->size() == (size_t)V*3);
    const uint32_t POS_BYTES = V * 3u * 4u, NRM_BYTES = V * 3u * 4u;
    const uint32_t COL_BYTES = COL ? V * 3u * 4u : 0u;
    const uint32_t IDX_BYTES = F3 * 4u;                    // uint32 indices
    const uint32_t NRM_OFF = POS_BYTES;
    const uint32_t COL_OFF = POS_BYTES + NRM_BYTES;
    const uint32_t IDX_OFF = POS_BYTES + NRM_BYTES + COL_BYTES;

    std::vector<float> nrm = vertex_normals(verts, faces);

    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < V; i++)
        for (int d = 0; d < 3; d++) {
            float v = verts[(size_t)i*3 + d];
            if (v < mn[d]) mn[d] = v;
            if (v > mx[d]) mx[d] = v;
        }

    const uint32_t BIN_LEN = POS_BYTES + NRM_BYTES + COL_BYTES + IDX_BYTES;

    char json[2560]; int jl;
    if (COL) {
        jl = snprintf(json, sizeof(json),
            "{\"asset\":{\"version\":\"2.0\",\"generator\":\"pixal3d-cpp\"},"
            "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
            "\"materials\":[{\"name\":\"tex\",\"doubleSided\":true,"
            "\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],\"metallicFactor\":0.0,\"roughnessFactor\":0.9}}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"COLOR_0\":2},"
            "\"indices\":3,\"material\":0,\"mode\":4}]}],"
            "\"buffers\":[{\"byteLength\":%u}],"
            "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963}],"
            "\"accessors\":["
            "{\"bufferView\":0,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\","
            "\"min\":[%.8g,%.8g,%.8g],\"max\":[%.8g,%.8g,%.8g]},"
            "{\"bufferView\":1,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\"},"
            "{\"bufferView\":2,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\"},"
            "{\"bufferView\":3,\"componentType\":5125,\"count\":%u,\"type\":\"SCALAR\"}]}",
            BIN_LEN, POS_BYTES, NRM_OFF, NRM_BYTES, COL_OFF, COL_BYTES, IDX_OFF, IDX_BYTES, V,
            mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], V, V, F3);
    } else {
        jl = snprintf(json, sizeof(json),
            "{\"asset\":{\"version\":\"2.0\",\"generator\":\"pixal3d-cpp\"},"
            "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
            "\"materials\":[{\"name\":\"geom\",\"doubleSided\":true,"
            "\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.74,0.77,0.82,1.0],\"metallicFactor\":0.05,\"roughnessFactor\":0.75}}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1},\"indices\":2,\"material\":0,\"mode\":4}]}],"
            "\"buffers\":[{\"byteLength\":%u}],"
            "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963}],"
            "\"accessors\":["
            "{\"bufferView\":0,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\","
            "\"min\":[%.8g,%.8g,%.8g],\"max\":[%.8g,%.8g,%.8g]},"
            "{\"bufferView\":1,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\"},"
            "{\"bufferView\":2,\"componentType\":5125,\"count\":%u,\"type\":\"SCALAR\"}]}",
            BIN_LEN, POS_BYTES, NRM_OFF, NRM_BYTES, IDX_OFF, IDX_BYTES, V,
            mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], V, F3);
    }
    if (jl < 0 || jl >= (int)sizeof(json)) return false;

    uint32_t json_len = (uint32_t)jl;
    uint32_t json_pad = (4 - (json_len & 3)) & 3;
    uint32_t json_chunk = json_len + json_pad;
    uint32_t bin_pad = (4 - (BIN_LEN & 3)) & 3;            // 0 (already aligned)
    uint32_t bin_chunk = BIN_LEN + bin_pad;
    uint32_t total = 12 + 8 + json_chunk + 8 + bin_chunk;

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    w32(0x46546C67u); w32(2u); w32(total);                 // header
    w32(json_chunk); w32(0x4E4F534Au);                     // "JSON"
    std::fwrite(json, 1, json_len, f);
    for (uint32_t i = 0; i < json_pad; i++) std::fputc(' ', f);
    w32(bin_chunk); w32(0x004E4942u);                      // "BIN\0"
    std::fwrite(verts.data(), 4, (size_t)V*3, f);
    std::fwrite(nrm.data(), 4, (size_t)V*3, f);
    if (COL) std::fwrite(vcolors->data(), 4, (size_t)V*3, f);
    std::vector<uint32_t> idx(F3);
    for (uint32_t i = 0; i < F3; i++) idx[i] = (uint32_t)faces[i];
    std::fwrite(idx.data(), 4, F3, f);
    for (uint32_t i = 0; i < bin_pad; i++) std::fputc(0, f);
    std::fclose(f);
    return true;
}

}  // namespace glb
