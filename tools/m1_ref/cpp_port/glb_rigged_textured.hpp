// glTF 2.0 binary (.glb) writer for a TEXTURED + RIGGED mesh — the final-asset rung of the
// auto-rigging pipeline. It is the union of glb_rigged.hpp (skin/skeleton/IBM + JOINTS_0/
// WEIGHTS_0) and glb_textured.hpp (TEXCOORD_0 + an embedded baseColor image + a PBR material
// with baseColorTexture). The output is a single skinned mesh node that ALSO carries a UV-mapped
// baseColor texture, i.e. a web-ready textured avatar with a bound skeleton.
//
// Reuses glb_rigged.hpp's rig_local_and_ibm() + rig_topk4() (top-4 LBS encode, world->local
// joint translations, column-major inverse-bind matrices) verbatim — the rig half is byte-for-byte
// identical to write_rigged_glb. The texture half embeds the supplied image bytes (PNG or JPEG,
// taken AS-IS — no re-encode) into a bufferView and wires textures[0]/images[0]/samplers[0] +
// materials[0].pbrMetallicRoughness.baseColorTexture, matching glb_textured.hpp's conventions.
//
// JSON is assembled with snprintf fragments into a std::string (like glb_rigged.hpp), since the
// joint-node tree dwarfs any fixed buffer. Header-only, namespace glb, no ggml/CUDA, no nlohmann
// (the image bytes are carried raw, so no stb re-encode is needed).
//
// BIN accessor layout:
//   0 POSITION  VEC3 f32
//   1 NORMAL    VEC3 f32
//   2 TEXCOORD_0 VEC2 f32
//   3 JOINTS_0  VEC4 u16
//   4 WEIGHTS_0 VEC4 f32
//   5 indices   SCALAR u32
//   6 IBM       MAT4 f32 (count J)
// bufferView 7 = the embedded baseColor image bytes (no target).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cfloat>
#include "glb_rigged.hpp"   // reuse rig_local_and_ibm(), rig_topk4(), glb::vertex_normals()

namespace glb {

// verts V*3, faces F*3, uvs V*2 (glTF convention), joints J*3 (world), parents J (root=-1),
// skin_weights V*J (row-major), normals optional (computed if null/wrong size). tex_png/tex_png_len
// = the baseColor image bytes (carried verbatim), mime e.g. "image/png" / "image/jpeg". Returns
// false on malformed forest, size mismatch, or a missing texture.
inline bool write_rigged_textured_glb(const char* path,
                                      const std::vector<float>&    verts,        // V*3
                                      const std::vector<int64_t>&  faces,        // F*3
                                      const std::vector<float>&    uvs,          // V*2
                                      const std::vector<float>&    joints,       // J*3 world
                                      const std::vector<int>&      parents,      // J, root=-1
                                      const std::vector<float>&    skin_weights, // V*J row-major
                                      const std::vector<float>*    normals,
                                      const uint8_t*               tex_png,
                                      size_t                       tex_png_len,
                                      const char*                  mime = "image/png") {
    const uint32_t V  = (uint32_t)(verts.size() / 3);
    const uint32_t F3 = (uint32_t)faces.size();   // index count = F*3
    const uint32_t J  = (uint32_t)(joints.size() / 3);
    if (V == 0 || J == 0) return false;
    if (parents.size() != J) return false;
    if (skin_weights.size() != (size_t)V * J) return false;
    if (uvs.size() != (size_t)V * 2) return false;
    if (!tex_png || tex_png_len == 0) return false;

    // normals: use supplied, else compute
    std::vector<float> nrm_local;
    const std::vector<float>* nrm = normals;
    if (!nrm || nrm->size() != (size_t)V * 3) {
        nrm_local = vertex_normals(verts, faces);
        nrm = &nrm_local;
    }

    // skeleton: local translations + inverse-bind matrices (shared with glb_rigged.hpp)
    std::vector<float> locals, ibm;
    if (!rig_local_and_ibm(joints, parents, locals, ibm)) return false;

    // LBS top-4 encode (shared with glb_rigged.hpp)
    std::vector<uint16_t> joints4;
    std::vector<float>    weights4;
    rig_topk4(skin_weights, V, J, joints4, weights4);

    // POSITION bounds
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < V; i++)
        for (int d = 0; d < 3; d++) {
            float v = verts[(size_t)i * 3 + d];
            if (v < mn[d]) mn[d] = v;
            if (v > mx[d]) mx[d] = v;
        }

    // ---- BIN layout (4-byte aligned segments) ----
    auto pad4 = [](uint32_t n) { return (4 - (n & 3)) & 3; };
    const uint32_t POS = V * 3u * 4u;
    const uint32_t NRM = V * 3u * 4u;
    const uint32_t UV  = V * 2u * 4u;
    const uint32_t JNT = V * 4u * 2u;    // u16
    const uint32_t WGT = V * 4u * 4u;    // f32
    const uint32_t IDX = F3 * 4u;        // u32
    const uint32_t IBM = J * 16u * 4u;   // f32
    const uint32_t TEX = (uint32_t)tex_png_len;
    uint32_t off = 0;
    auto alloc = [&](uint32_t len) { uint32_t o = off; off += len + pad4(len); return o; };
    const uint32_t POS_OFF = alloc(POS);
    const uint32_t NRM_OFF = alloc(NRM);
    const uint32_t UV_OFF  = alloc(UV);
    const uint32_t JNT_OFF = alloc(JNT);
    const uint32_t WGT_OFF = alloc(WGT);
    const uint32_t IDX_OFF = alloc(IDX);
    const uint32_t IBM_OFF = alloc(IBM);
    const uint32_t TEX_OFF = alloc(TEX);
    const uint32_t BIN_LEN = off;

    // ---- build node tree from parents ----
    std::vector<std::vector<int>> children(J);
    int root_joint = -1;
    for (uint32_t j = 0; j < J; j++) {
        int p = parents[j];
        if (p >= 0) children[p].push_back((int)j);
        else if (root_joint < 0) root_joint = (int)j;
    }
    if (root_joint < 0) root_joint = 0;

    // ---- assemble JSON ----
    char buf[1024];
    std::string js;
    js.reserve((size_t)J * 160 + 4096);

    js += "{\"asset\":{\"version\":\"2.0\",\"generator\":\"pixal3d-cpp\"},";
    js += "\"scene\":0,";

    // scenes: root joint nodes + mesh node (index J)
    js += "\"scenes\":[{\"nodes\":[";
    {
        bool first = true;
        for (uint32_t j = 0; j < J; j++) {
            if (parents[j] != -1) continue;
            snprintf(buf, sizeof(buf), "%s%u", first ? "" : ",", j);
            js += buf; first = false;
        }
        snprintf(buf, sizeof(buf), "%s%u", first ? "" : ",", J);
        js += buf;
    }
    js += "]}],";

    // nodes
    js += "\"nodes\":[";
    for (uint32_t j = 0; j < J; j++) {
        int n = snprintf(buf, sizeof(buf),
            "{\"name\":\"bone_%u\",\"translation\":[%.8g,%.8g,%.8g]",
            j, locals[j*3+0], locals[j*3+1], locals[j*3+2]);
        if (n < 0 || n >= (int)sizeof(buf)) return false;
        js += buf;
        if (!children[j].empty()) {
            js += ",\"children\":[";
            for (size_t c = 0; c < children[j].size(); c++) {
                snprintf(buf, sizeof(buf), "%s%d", c ? "," : "", children[j][c]);
                js += buf;
            }
            js += "]";
        }
        js += "}";
        js += ",";
    }
    js += "{\"name\":\"rigged_mesh\",\"mesh\":0,\"skin\":0}";  // mesh node (index J)
    js += "],";

    // skins
    js += "\"skins\":[{";
    snprintf(buf, sizeof(buf), "\"inverseBindMatrices\":6,\"skeleton\":%d,\"joints\":[", root_joint);
    js += buf;
    for (uint32_t j = 0; j < J; j++) { snprintf(buf, sizeof(buf), "%s%u", j ? "," : "", j); js += buf; }
    js += "]}],";

    // material: PBR with baseColorTexture (texture 0), doubleSided (like glb_textured.hpp)
    js += "\"materials\":[{\"name\":\"rigged_pbr\",\"doubleSided\":true,\"alphaMode\":\"OPAQUE\","
          "\"pbrMetallicRoughness\":{"
          "\"baseColorTexture\":{\"index\":0,\"texCoord\":0},"
          "\"baseColorFactor\":[1,1,1,1],"
          "\"metallicFactor\":0.0,\"roughnessFactor\":1.0}}],";

    // samplers / textures / images
    js += "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,\"wrapS\":33071,\"wrapT\":33071}],";
    js += "\"textures\":[{\"source\":0,\"sampler\":0}],";
    snprintf(buf, sizeof(buf), "\"images\":[{\"bufferView\":7,\"mimeType\":\"%s\"}],", mime);
    js += buf;

    // mesh
    js += "\"meshes\":[{\"primitives\":[{\"attributes\":{"
          "\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2,\"JOINTS_0\":3,\"WEIGHTS_0\":4},"
          "\"indices\":5,\"material\":0,\"mode\":4}]}],";

    // buffers / bufferViews
    snprintf(buf, sizeof(buf), "\"buffers\":[{\"byteLength\":%u}],", BIN_LEN); js += buf;
    js += "\"bufferViews\":[";
    snprintf(buf, sizeof(buf),
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // POSITION
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // NORMAL
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // TEXCOORD_0
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // JOINTS_0
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // WEIGHTS_0
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"   // indices
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u},"                    // IBM (no target)
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}",                    // image (no target)
        POS_OFF, POS, NRM_OFF, NRM, UV_OFF, UV, JNT_OFF, JNT, WGT_OFF, WGT,
        IDX_OFF, IDX, IBM_OFF, IBM, TEX_OFF, TEX);
    js += buf;
    js += "],";

    // accessors
    js += "\"accessors\":[";
    snprintf(buf, sizeof(buf),
        "{\"bufferView\":0,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\","
        "\"min\":[%.8g,%.8g,%.8g],\"max\":[%.8g,%.8g,%.8g]},",
        V, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
    js += buf;
    snprintf(buf, sizeof(buf),
        "{\"bufferView\":1,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\"},"   // NORMAL
        "{\"bufferView\":2,\"componentType\":5126,\"count\":%u,\"type\":\"VEC2\"},"   // TEXCOORD_0
        "{\"bufferView\":3,\"componentType\":5123,\"count\":%u,\"type\":\"VEC4\"},"   // JOINTS_0 u16
        "{\"bufferView\":4,\"componentType\":5126,\"count\":%u,\"type\":\"VEC4\"},"   // WEIGHTS_0 f32
        "{\"bufferView\":5,\"componentType\":5125,\"count\":%u,\"type\":\"SCALAR\"}," // indices u32
        "{\"bufferView\":6,\"componentType\":5126,\"count\":%u,\"type\":\"MAT4\"}",   // IBM
        V, V, V, V, F3, J);
    js += buf;
    js += "]}";

    // ---- write GLB ----
    uint32_t json_len   = (uint32_t)js.size();
    uint32_t json_pad   = pad4(json_len);
    uint32_t json_chunk = json_len + json_pad;
    uint32_t bin_pad    = pad4(BIN_LEN);
    uint32_t bin_chunk  = BIN_LEN + bin_pad;
    uint32_t total      = 12 + 8 + json_chunk + 8 + bin_chunk;

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto wz  = [&](uint32_t n) { for (uint32_t i = 0; i < n; i++) std::fputc(0, f); };
    w32(0x46546C67u); w32(2u); w32(total);                 // header
    w32(json_chunk); w32(0x4E4F534Au);                     // "JSON"
    std::fwrite(js.data(), 1, json_len, f);
    for (uint32_t i = 0; i < json_pad; i++) std::fputc(' ', f);
    w32(bin_chunk); w32(0x004E4942u);                      // "BIN\0"
    std::fwrite(verts.data(),   4, (size_t)V * 3, f); wz(pad4(POS));
    std::fwrite(nrm->data(),    4, (size_t)V * 3, f); wz(pad4(NRM));
    std::fwrite(uvs.data(),     4, (size_t)V * 2, f); wz(pad4(UV));
    std::fwrite(joints4.data(), 2, (size_t)V * 4, f); wz(pad4(JNT));
    std::fwrite(weights4.data(),4, (size_t)V * 4, f); wz(pad4(WGT));
    {
        std::vector<uint32_t> idx(F3);
        for (uint32_t i = 0; i < F3; i++) idx[i] = (uint32_t)faces[i];
        std::fwrite(idx.data(), 4, F3, f); wz(pad4(IDX));
    }
    std::fwrite(ibm.data(), 4, (size_t)J * 16, f); wz(pad4(IBM));
    std::fwrite(tex_png, 1, tex_png_len, f); wz(pad4(TEX));
    for (uint32_t i = 0; i < bin_pad; i++) std::fputc(0, f);
    std::fclose(f);
    return true;
}

}  // namespace glb
