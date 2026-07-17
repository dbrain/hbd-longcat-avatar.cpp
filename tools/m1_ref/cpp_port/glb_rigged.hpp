// glTF 2.0 binary (.glb) writer for a SKINNED / RIGGED mesh — the "R6" rung of the
// auto-rigging pipeline. Emits POSITION (+NORMAL), indices (uint32), per-vertex
// JOINTS_0 (VEC4 u16) + WEIGHTS_0 (VEC4 f32), a node-per-joint skeleton (local
// translations derived from world joints via the `parents` forest), and skins[0]
// with inverseBindMatrices (column-major MAT4, translation-only bind).
//
// Mirrors glb_writer.hpp's conventions: header-only, namespace glb, GLB chunk layout
// (12-byte header magic 0x46546C67 / version 2 / total, JSON chunk 0x4E4F534A
// space-padded, BIN chunk 0x004E4942 zero-padded), verts std::vector<float> V*3,
// faces std::vector<int64_t> F*3. The JSON is far larger than glb_writer's fixed
// snprintf buffer (60 joint nodes), so it is assembled into a std::string with
// snprintf'd fragments (each fragment bounds-checked).
//
// glTF skinning (linear blend skinning) semantics encoded here:
//   * node translations are LOCAL: local = world_joint - world_parent (root = world).
//   * inverseBindMatrices[j] = inverse of joint j's world bind transform. Bind is
//     translation-only, so inverseBind = translate(-world_joint_pos), stored as 16
//     floats COLUMN-MAJOR.
//   * JOINTS_0/WEIGHTS_0 hold the TOP-4 weights per vertex (largest magnitude),
//     renormalized to sum 1.0; all-zero rows fall back to joint 0 weight 1.0.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cfloat>
#include "glb_writer.hpp"   // reuse glb::vertex_normals()

namespace glb {

// Compute world->local joint translations + per-joint inverse-bind matrices (column-major
// MAT4). Returns false on a malformed forest (parent index out of range or a cycle).
// locals: J*3 (filled), ibm: J*16 (filled, column-major). worlds passed back for asserts.
inline bool rig_local_and_ibm(const std::vector<float>& joints,    // J*3 world
                              const std::vector<int>&   parents,   // J, root=-1
                              std::vector<float>&        locals,    // out J*3
                              std::vector<float>&        ibm) {     // out J*16
    const size_t J = joints.size() / 3;
    if (parents.size() != J) return false;
    // validate forest: parent in [-1, J), no self-parent; topological reachability to a root
    // (detect cycles by walking up to J steps).
    for (size_t j = 0; j < J; j++) {
        int p = parents[j];
        if (p == (int)j) return false;
        if (p < -1 || p >= (int)J) return false;
        // walk to root, bounded by J to catch cycles
        int cur = (int)j, steps = 0;
        while (cur != -1) {
            cur = parents[cur];
            if (cur < -1 || cur >= (int)J) return false;
            if (++steps > (int)J) return false;  // cycle
        }
    }
    locals.assign(J * 3, 0.f);
    ibm.assign(J * 16, 0.f);
    for (size_t j = 0; j < J; j++) {
        int p = parents[j];
        for (int d = 0; d < 3; d++) {
            float w = joints[j * 3 + d];
            float pw = (p >= 0) ? joints[(size_t)p * 3 + d] : 0.f;
            locals[j * 3 + d] = w - pw;
        }
        // inverseBind = translate(-world_pos), column-major 4x4:
        //   [ 1 0 0 0,  0 1 0 0,  0 0 1 0,  -x -y -z 1 ]  (translation is the last column,
        //   which in column-major layout occupies indices 12,13,14,15).
        float* m = &ibm[j * 16];
        m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
        m[12] = -joints[j * 3 + 0];
        m[13] = -joints[j * 3 + 1];
        m[14] = -joints[j * 3 + 2];
    }
    return true;
}

// Top-4 LBS encode. skin_weights is V*J row-major (vertex-major). Fills joints4 (V*4 u16)
// and weights4 (V*4 f32), top-4 largest per vertex renormalized to sum 1.0 (all-zero -> joint0=1).
inline void rig_topk4(const std::vector<float>& skin_weights, uint32_t V, uint32_t J,
                      std::vector<uint16_t>& joints4, std::vector<float>& weights4) {
    joints4.assign((size_t)V * 4, 0);
    weights4.assign((size_t)V * 4, 0.f);
    for (uint32_t v = 0; v < V; v++) {
        const float* row = &skin_weights[(size_t)v * J];
        // find top-4 indices by weight (simple selection; J small ~60)
        int    bi[4] = {-1, -1, -1, -1};
        float  bw[4] = {0.f, 0.f, 0.f, 0.f};
        for (uint32_t k = 0; k < J; k++) {
            float w = row[k];
            if (w <= 0.f) continue;
            // insert into the running top-4
            int slot = -1; float minw = bw[0]; int mins = 0;
            for (int s = 0; s < 4; s++) {
                if (bi[s] == -1) { slot = s; break; }
                if (bw[s] < minw) { minw = bw[s]; mins = s; }
            }
            if (slot >= 0) { bi[slot] = (int)k; bw[slot] = w; }
            else if (w > bw[mins]) { bi[mins] = (int)k; bw[mins] = w; }
        }
        float sum = 0.f;
        for (int s = 0; s < 4; s++) if (bi[s] >= 0) sum += bw[s];
        if (sum <= 0.f) {
            // all-zero row -> fully weight joint 0
            joints4[(size_t)v * 4 + 0] = 0;
            weights4[(size_t)v * 4 + 0] = 1.f;
        } else {
            for (int s = 0; s < 4; s++) {
                int idx = bi[s] >= 0 ? bi[s] : 0;
                float w = bi[s] >= 0 ? bw[s] / sum : 0.f;
                joints4[(size_t)v * 4 + s] = (uint16_t)idx;
                weights4[(size_t)v * 4 + s] = w;
            }
        }
    }
}

// Escape a bone name for embedding in the glTF JSON chunk. Standard names (mixamorig:Hips)
// need nothing, but never trust a name we did not generate to be JSON-safe.
inline std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

// joint_names: optional [J] names for the joint nodes. nullptr (or wrong size) => "bone_<j>".
inline bool write_rigged_glb(const char* path,
                             const std::vector<float>&   verts,        // V*3
                             const std::vector<int64_t>& faces,        // F*3
                             const std::vector<float>&   joints,       // J*3 world
                             const std::vector<int>&     parents,      // J, root=-1
                             const std::vector<float>&   skin_weights, // V*J row-major
                             const std::vector<float>*   normals = nullptr,
                             const std::vector<std::string>* joint_names = nullptr) {
    const uint32_t V  = (uint32_t)(verts.size() / 3);
    const uint32_t F3 = (uint32_t)faces.size();   // index count = F*3
    const uint32_t J  = (uint32_t)(joints.size() / 3);
    if (V == 0 || J == 0) return false;
    if (parents.size() != J) return false;
    if (skin_weights.size() != (size_t)V * J) return false;

    // normals: use supplied, else compute (degenerate if no faces -> falls back to +Z)
    std::vector<float> nrm_local;
    const std::vector<float>* nrm = normals;
    if (!nrm || nrm->size() != (size_t)V * 3) {
        nrm_local = vertex_normals(verts, faces);
        nrm = &nrm_local;
    }

    // skeleton: local translations + inverse-bind matrices
    std::vector<float> locals, ibm;
    if (!rig_local_and_ibm(joints, parents, locals, ibm)) return false;

    // LBS top-4 encode
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
    // accessor 0 POSITION  VEC3 f32
    // accessor 1 NORMAL    VEC3 f32
    // accessor 2 JOINTS_0  VEC4 u16
    // accessor 3 WEIGHTS_0 VEC4 f32
    // accessor 4 indices   SCALAR u32
    // accessor 5 IBM       MAT4 f32 (count J)
    auto pad4 = [](uint32_t n) { return (4 - (n & 3)) & 3; };
    const uint32_t POS = V * 3u * 4u;
    const uint32_t NRM = V * 3u * 4u;
    const uint32_t JNT = V * 4u * 2u;    // u16
    const uint32_t WGT = V * 4u * 4u;    // f32
    const uint32_t IDX = F3 * 4u;        // u32
    const uint32_t IBM = J * 16u * 4u;   // f32
    uint32_t off = 0;
    auto alloc = [&](uint32_t len) { uint32_t o = off; off += len + pad4(len); return o; };
    const uint32_t POS_OFF = alloc(POS);
    const uint32_t NRM_OFF = alloc(NRM);
    const uint32_t JNT_OFF = alloc(JNT);
    const uint32_t WGT_OFF = alloc(WGT);
    const uint32_t IDX_OFF = alloc(IDX);
    const uint32_t IBM_OFF = alloc(IBM);
    const uint32_t BIN_LEN = off;

    // ---- build node tree from parents ----
    // node indexing: nodes[0..J-1] = joints, node[J] = the skinned mesh node.
    // scene root nodes = all joint roots (parent==-1) + the mesh node.
    std::vector<std::vector<int>> children(J);
    int root_joint = -1;
    for (uint32_t j = 0; j < J; j++) {
        int p = parents[j];
        if (p >= 0) children[p].push_back((int)j);
        else if (root_joint < 0) root_joint = (int)j;
    }
    if (root_joint < 0) root_joint = 0;  // skin.skeleton must point somewhere valid

    // ---- assemble JSON into a std::string ----
    char buf[1024];
    std::string js;
    js.reserve((size_t)J * 160 + 4096);

    js += "{\"asset\":{\"version\":\"2.0\",\"generator\":\"pixal3d-cpp\"},";
    js += "\"scene\":0,";

    // scenes: root joint nodes + mesh node
    js += "\"scenes\":[{\"nodes\":[";
    {
        bool first = true;
        for (uint32_t j = 0; j < J; j++) {
            if (parents[j] != -1) continue;
            snprintf(buf, sizeof(buf), "%s%u", first ? "" : ",", j);
            js += buf; first = false;
        }
        snprintf(buf, sizeof(buf), "%s%u", first ? "" : ",", J);  // mesh node
        js += buf;
    }
    js += "]}],";

    // nodes
    js += "\"nodes\":[";
    const bool have_names = joint_names && joint_names->size() == J;
    for (uint32_t j = 0; j < J; j++) {
        std::string jname = have_names ? json_escape((*joint_names)[j])
                                       : ("bone_" + std::to_string(j));
        js += "{\"name\":\"" + jname + "\",";
        int n = snprintf(buf, sizeof(buf), "\"translation\":[%.8g,%.8g,%.8g]",
                         locals[j*3+0], locals[j*3+1], locals[j*3+2]);
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
    // mesh node (index J), references skin 0
    js += "{\"name\":\"rigged_mesh\",\"mesh\":0,\"skin\":0}";
    js += "],";

    // skins
    js += "\"skins\":[{";
    snprintf(buf, sizeof(buf), "\"inverseBindMatrices\":5,\"skeleton\":%d,\"joints\":[", root_joint);
    js += buf;
    for (uint32_t j = 0; j < J; j++) { snprintf(buf, sizeof(buf), "%s%u", j ? "," : "", j); js += buf; }
    js += "]}],";

    // material (doubleSided, like the existing writer)
    js += "\"materials\":[{\"name\":\"rigged\",\"doubleSided\":true,"
          "\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.74,0.77,0.82,1.0],"
          "\"metallicFactor\":0.05,\"roughnessFactor\":0.75}}],";

    // mesh
    js += "\"meshes\":[{\"primitives\":[{\"attributes\":{"
          "\"POSITION\":0,\"NORMAL\":1,\"JOINTS_0\":2,\"WEIGHTS_0\":3},"
          "\"indices\":4,\"material\":0,\"mode\":4}]}],";

    // buffers / bufferViews
    snprintf(buf, sizeof(buf), "\"buffers\":[{\"byteLength\":%u}],", BIN_LEN); js += buf;
    js += "\"bufferViews\":[";
    snprintf(buf, sizeof(buf),
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // POSITION
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // NORMAL
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // JOINTS_0
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"   // WEIGHTS_0
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"   // indices
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}",                    // IBM (no target)
        POS_OFF, POS, NRM_OFF, NRM, JNT_OFF, JNT, WGT_OFF, WGT, IDX_OFF, IDX, IBM_OFF, IBM);
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
        "{\"bufferView\":2,\"componentType\":5123,\"count\":%u,\"type\":\"VEC4\"},"   // JOINTS_0 u16
        "{\"bufferView\":3,\"componentType\":5126,\"count\":%u,\"type\":\"VEC4\"},"   // WEIGHTS_0 f32
        "{\"bufferView\":4,\"componentType\":5125,\"count\":%u,\"type\":\"SCALAR\"}," // indices u32
        "{\"bufferView\":5,\"componentType\":5126,\"count\":%u,\"type\":\"MAT4\"}",   // IBM
        V, V, V, F3, J);
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
    // POSITION
    std::fwrite(verts.data(), 4, (size_t)V * 3, f); wz(pad4(POS));
    // NORMAL
    std::fwrite(nrm->data(), 4, (size_t)V * 3, f); wz(pad4(NRM));
    // JOINTS_0 (u16)
    std::fwrite(joints4.data(), 2, (size_t)V * 4, f); wz(pad4(JNT));
    // WEIGHTS_0 (f32)
    std::fwrite(weights4.data(), 4, (size_t)V * 4, f); wz(pad4(WGT));
    // indices (u32)
    {
        std::vector<uint32_t> idx(F3);
        for (uint32_t i = 0; i < F3; i++) idx[i] = (uint32_t)faces[i];
        std::fwrite(idx.data(), 4, F3, f); wz(pad4(IDX));
    }
    // IBM (f32, column-major)
    std::fwrite(ibm.data(), 4, (size_t)J * 16, f); wz(pad4(IBM));
    for (uint32_t i = 0; i < bin_pad; i++) std::fputc(0, f);
    std::fclose(f);
    return true;
}

}  // namespace glb
