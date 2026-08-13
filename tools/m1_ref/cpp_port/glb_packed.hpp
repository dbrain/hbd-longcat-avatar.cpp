// glb_packed.hpp — fully in-process COMPRESSED textured GLB writer (the native gltfpack equivalent).
// Emits: KHR_mesh_quantization (int16 pos / int8 normal / uint16 uv) + EXT_meshopt_compression
// (meshopt-encoded vertex+index streams) + KHR_texture_basisu (KTX2 baseColor + metallicRoughness).
// All native C++: meshoptimizer (already vendored) + basis_universal (ktx2_encode.hpp). No external
// gltfpack/toktx/node. Produces the same shippable asset gltfpack -cc -tc/-tu does, but from the
// in-memory atlas, so pixal3d --tex --pack writes the small GLB directly (no second process, no reparse).
//
// Layout (matches gltfpack): buffer 0 = the GLB BIN chunk holding [ktx0|ktx1|comp streams]; buffer 1
// is a non-stored EXT_meshopt_compression.fallback buffer (the decode target the loader synthesizes).
#pragma once
#include "../../../thirdparty/json.hpp"
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"
#include "ktx2_encode.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace glb {

// verts/normals [V*3], uvs [V*2] (glTF convention), faces uint32 [F*3]. base_color RGBA8 [tw*th*4],
// metal_rough RGB8 [tw*th*3]. uastc=true → near-lossless UASTC+Zstd textures (hero); false → ETC1S
// (small). etc1s_quality [1,255] (ignored for uastc). Returns false on any encode failure.
inline bool write_glb_textured_packed(const char* path,
        const std::vector<float>& verts, const std::vector<float>& normals,
        const std::vector<float>& uvs, const std::vector<uint32_t>& faces,
        const std::vector<uint8_t>& base_color, const std::vector<uint8_t>& metal_rough,
        int tw, int th, bool uastc, int etc1s_quality = 192, int threads = 0,
        const std::vector<uint8_t>* normal_map = nullptr /* RGB tw*th*3, tangent-space */) {
    using json = nlohmann::json;
    const uint32_t V = (uint32_t)(verts.size() / 3), F3 = (uint32_t)faces.size();

    // ---- 1. quantize (KHR_mesh_quantization): pos int16 snorm in a unit cube about the bbox center,
    //         mapped back by the node translation+scale; normal int8 snorm; uv uint16 unorm. ----
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < V; i++)
        for (int d = 0; d < 3; d++) { float v = verts[(size_t)i*3+d]; mn[d]=std::min(mn[d],v); mx[d]=std::max(mx[d],v); }
    float center[3] = {(mn[0]+mx[0])*0.5f, (mn[1]+mx[1])*0.5f, (mn[2]+mx[2])*0.5f};
    float radius = std::max({mx[0]-center[0], mx[1]-center[1], mx[2]-center[2], 1e-6f});

    // strided streams (meshopt requires vertex_size % 4 == 0): pos int16[3]+pad=8, nrm int8[3]+pad=4, uv uint16[2]=4
    std::vector<uint8_t> pos_raw((size_t)V*8, 0), nrm_raw((size_t)V*4, 0), uv_raw((size_t)V*4, 0);
    std::vector<float> nrm_f((size_t)V*4);   // xyzw input for the octahedral filter (w unused → 0)
    // POSITION accessor min/max MUST be the RAW QUANTIZED INTEGERS for an integer componentType (SHORT),
    // even with normalized:true (glTF spec; floats here = ARRAY_TYPE_MISMATCH → loaders reject the mesh).
    int qmn[3] = {32767, 32767, 32767}, qmx[3] = {-32768, -32768, -32768};
    for (uint32_t i = 0; i < V; i++) {
        int16_t* p = (int16_t*)&pos_raw[(size_t)i*8];
        uint16_t* u = (uint16_t*)&uv_raw[(size_t)i*4];
        for (int d = 0; d < 3; d++) {
            float v = std::max(-1.f, std::min(1.f, (verts[(size_t)i*3+d]-center[d])/radius));
            int qv = meshopt_quantizeSnorm(v, 16);
            p[d] = (int16_t)qv;
            qmn[d] = std::min(qmn[d], qv); qmx[d] = std::max(qmx[d], qv);
            nrm_f[(size_t)i*4+d] = std::max(-1.f, std::min(1.f, normals[(size_t)i*3+d]));
        }
        nrm_f[(size_t)i*4+3] = 0.f;
        for (int d = 0; d < 2; d++)
            u[d] = (uint16_t)meshopt_quantizeUnorm(std::max(0.f, std::min(1.f, uvs[(size_t)i*2+d])), 16);
    }
    // octahedral normal filter: int8 X/Y oct → ~24% smaller geometry than raw int8 xyz (matches gltfpack).
    // Decoder runs meshopt_decodeFilterOct (filter "OCTAHEDRAL") to reconstruct the int8 VEC3 normal.
    meshopt_encodeFilterOct(nrm_raw.data(), V, 4, 8, nrm_f.data());

    // ---- 2. meshopt-compress each stream (mode ATTRIBUTES) + the index buffer (mode TRIANGLES) ----
    // Force vertex codec VERSION 0: the vendored meshoptimizer defaults to v1 (a recent/experimental
    // codec, header 0xa1) which older decoders (model-viewer 3.5.0's bundled MeshoptDecoder, three.js)
    // CANNOT decode → the mesh fails to load and nothing renders. v0 (0xa0) is universally supported.
    meshopt_encodeVertexVersion(0);
    auto enc_vtx = [&](const std::vector<uint8_t>& raw, size_t stride) {
        std::vector<uint8_t> out(meshopt_encodeVertexBufferBound(V, stride));
        out.resize(meshopt_encodeVertexBuffer(out.data(), out.size(), raw.data(), V, stride));
        return out;
    };
    std::vector<uint8_t> comp_pos = enc_vtx(pos_raw, 8), comp_nrm = enc_vtx(nrm_raw, 4), comp_uv = enc_vtx(uv_raw, 4);
    std::vector<uint8_t> comp_idx(meshopt_encodeIndexBufferBound(F3, V));
    comp_idx.resize(meshopt_encodeIndexBuffer(comp_idx.data(), comp_idx.size(), faces.data(), F3));

    // ---- 3. KTX2 textures (KHR_texture_basisu). baseColor=sRGB, metallicRoughness=linear ----
    std::vector<uint8_t> ktx0 = ktx2enc::encode(base_color.data(), tw, th, uastc, etc1s_quality, /*srgb*/true,  /*comps*/4, threads);
    std::vector<uint8_t> ktx1 = ktx2enc::encode(metal_rough.data(), tw, th, uastc, etc1s_quality, /*srgb*/false, /*comps*/3, threads);
    if (ktx0.empty() || ktx1.empty()) { std::fprintf(stderr, "[glb_packed] KTX2 encode failed\n"); return false; }
    std::vector<uint8_t> ktxn;   // optional tangent-space normal map (linear)
    if (normal_map && !normal_map->empty())
        ktxn = ktx2enc::encode(normal_map->data(), tw, th, uastc, etc1s_quality, /*srgb*/false, /*comps*/3, threads);
    const bool hasN = !ktxn.empty();

    // ---- 4. lay out buffer 0 (BIN): [ktx0|ktx1|comp_pos|comp_nrm|comp_uv|comp_idx], 4-aligned ----
    auto pad4 = [](uint32_t n) { return (4 - (n & 3)) & 3; };
    uint32_t off = 0; auto alloc = [&](uint32_t len) { uint32_t o = off; off += len + pad4(len); return o; };
    const uint32_t KTX0_OFF = alloc((uint32_t)ktx0.size()), KTX1_OFF = alloc((uint32_t)ktx1.size());
    const uint32_t CPOS_OFF = alloc((uint32_t)comp_pos.size()), CNRM_OFF = alloc((uint32_t)comp_nrm.size());
    const uint32_t CUV_OFF  = alloc((uint32_t)comp_uv.size()),  CIDX_OFF = alloc((uint32_t)comp_idx.size());
    const uint32_t KTXN_OFF = hasN ? alloc((uint32_t)ktxn.size()) : 0;   // normal image appended (bufferView 6)
    const uint32_t BIN_LEN = off;

    // fallback buffer 1 (not stored): uncompressed strided sizes the loader decodes into
    const uint32_t FB_POS = V*8, FB_NRM = V*4, FB_UV = V*4, FB_IDX = F3*4;
    const uint32_t FB_POS_OFF = 0, FB_NRM_OFF = FB_POS, FB_UV_OFF = FB_POS+FB_NRM, FB_IDX_OFF = FB_POS+FB_NRM+FB_UV;
    const uint32_t FB_LEN = FB_POS + FB_NRM + FB_UV + FB_IDX;

    // NOTE: `count` is REQUIRED by EXT_meshopt_compression (elements: vertices for ATTRIBUTES, indices
    // for TRIANGLES). Omitting it = the meshopt decoder can't decode → mesh doesn't render (and the
    // glTF-Validator won't flag it, since it treats meshopt as an unsupported extension).
    auto moc = [](uint32_t buf, uint32_t boff, uint32_t blen, uint32_t stride, uint32_t count, const char* mode, const char* filter) {
        json e = {{"buffer", buf}, {"byteOffset", boff}, {"byteLength", blen}, {"byteStride", stride}, {"mode", mode}, {"count", count}};
        if (filter) e["filter"] = filter;
        return json{{"EXT_meshopt_compression", e}};
    };

    json j;
    j["asset"] = {{"version", "2.0"}, {"generator", "pixal3d-cpp (in-process meshopt+ktx2)"}};
    j["extensionsUsed"] = {"KHR_mesh_quantization", "EXT_meshopt_compression", "KHR_texture_basisu"};
    j["extensionsRequired"] = {"KHR_mesh_quantization", "EXT_meshopt_compression", "KHR_texture_basisu"};
    j["scene"] = 0; j["scenes"] = {{{"nodes", {0}}}};
    j["nodes"] = {{{"mesh", 0}, {"translation", {center[0], center[1], center[2]}}, {"scale", {radius, radius, radius}}}};
    j["materials"] = {{{"name", "pbr"}, {"doubleSided", true}, {"alphaMode", "OPAQUE"},
        {"pbrMetallicRoughness", {{"baseColorTexture", {{"index", 0}, {"texCoord", 0}}}, {"baseColorFactor", {1,1,1,1}},
            {"metallicRoughnessTexture", {{"index", 1}, {"texCoord", 0}}}, {"metallicFactor", 1.0}, {"roughnessFactor", 1.0}}}}};
    j["meshes"] = {{{"primitives", {{{"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
        {"indices", 3}, {"material", 0}, {"mode", 4}}}}}};
    j["samplers"] = {{{"magFilter", 9729}, {"minFilter", 9987}, {"wrapS", 33071}, {"wrapT", 33071}}};  // mipmapped
    j["textures"] = {{{"sampler", 0}, {"extensions", {{"KHR_texture_basisu", {{"source", 0}}}}}},
                     {{"sampler", 0}, {"extensions", {{"KHR_texture_basisu", {{"source", 1}}}}}}};
    j["images"] = {{{"bufferView", 0}, {"mimeType", "image/ktx2"}}, {{"bufferView", 1}, {"mimeType", "image/ktx2"}}};
    j["buffers"] = {{{"byteLength", BIN_LEN}},
                    {{"byteLength", FB_LEN}, {"extensions", {{"EXT_meshopt_compression", {{"fallback", true}}}}}}};
    j["bufferViews"] = {
        {{"buffer", 0}, {"byteOffset", KTX0_OFF}, {"byteLength", (uint32_t)ktx0.size()}},
        {{"buffer", 0}, {"byteOffset", KTX1_OFF}, {"byteLength", (uint32_t)ktx1.size()}},
        {{"buffer", 1}, {"byteOffset", FB_POS_OFF}, {"byteLength", FB_POS}, {"byteStride", 8}, {"target", 34962},
            {"extensions", moc(0, CPOS_OFF, (uint32_t)comp_pos.size(), 8, V, "ATTRIBUTES", nullptr)}},
        {{"buffer", 1}, {"byteOffset", FB_NRM_OFF}, {"byteLength", FB_NRM}, {"byteStride", 4}, {"target", 34962},
            {"extensions", moc(0, CNRM_OFF, (uint32_t)comp_nrm.size(), 4, V, "ATTRIBUTES", "OCTAHEDRAL")}},
        {{"buffer", 1}, {"byteOffset", FB_UV_OFF}, {"byteLength", FB_UV}, {"byteStride", 4}, {"target", 34962},
            {"extensions", moc(0, CUV_OFF, (uint32_t)comp_uv.size(), 4, V, "ATTRIBUTES", nullptr)}},
        {{"buffer", 1}, {"byteOffset", FB_IDX_OFF}, {"byteLength", FB_IDX}, {"target", 34963},
            {"extensions", moc(0, CIDX_OFF, (uint32_t)comp_idx.size(), 4, F3, "TRIANGLES", nullptr)}}};
    j["accessors"] = {
        {{"bufferView", 2}, {"componentType", 5122}, {"count", V}, {"type", "VEC3"}, {"normalized", true},
            {"min", {qmn[0], qmn[1], qmn[2]}}, {"max", {qmx[0], qmx[1], qmx[2]}}},
        {{"bufferView", 3}, {"componentType", 5120}, {"count", V}, {"type", "VEC3"}, {"normalized", true}},
        {{"bufferView", 4}, {"componentType", 5123}, {"count", V}, {"type", "VEC2"}, {"normalized", true}},
        {{"bufferView", 5}, {"componentType", 5125}, {"count", F3}, {"type", "SCALAR"}}};

    if (hasN) {   // append normal map as bufferView 6 / image 2 / texture 2 → material.normalTexture
        j["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", KTXN_OFF}, {"byteLength", (uint32_t)ktxn.size()}});
        j["images"].push_back({{"bufferView", 6}, {"mimeType", "image/ktx2"}});
        j["textures"].push_back({{"sampler", 0}, {"extensions", {{"KHR_texture_basisu", {{"source", 2}}}}}});
        j["materials"][0]["normalTexture"] = {{"index", 2}, {"texCoord", 0}};
    }

    std::string js = j.dump();
    uint32_t json_len = (uint32_t)js.size(), json_pad = pad4(json_len), json_chunk = json_len + json_pad;
    uint32_t bin_pad = pad4(BIN_LEN), bin_chunk = BIN_LEN + bin_pad;
    uint32_t total = 12 + 8 + json_chunk + 8 + bin_chunk;

    FILE* f = std::fopen(path, "wb"); if (!f) return false;
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto wblock = [&](const std::vector<uint8_t>& b) { std::fwrite(b.data(), 1, b.size(), f);
        for (uint32_t i = 0, n = pad4((uint32_t)b.size()); i < n; i++) std::fputc(0, f); };
    w32(0x46546C67u); w32(2u); w32(total);
    w32(json_chunk); w32(0x4E4F534Au);  // "JSON"
    std::fwrite(js.data(), 1, json_len, f); for (uint32_t i = 0; i < json_pad; i++) std::fputc(' ', f);
    w32(bin_chunk); w32(0x004E4942u);    // "BIN\0"
    wblock(ktx0); wblock(ktx1); wblock(comp_pos); wblock(comp_nrm); wblock(comp_uv); wblock(comp_idx);
    if (hasN) wblock(ktxn);
    for (uint32_t i = 0; i < bin_pad; i++) std::fputc(0, f);
    std::fclose(f);
    return true;
}

}  // namespace glb
