#include "../../../thirdparty/json.hpp"
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../../thirdparty/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../thirdparty/stb_image_write.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef REPACK_INPROC_BUILD
#include "glb_packed.hpp"   // in-process meshopt+KTX2 writer (validate the native packer at full scale)
#endif

using json = nlohmann::json;

static std::vector<uint8_t> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "open failed: %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> out((size_t)n);
    if (std::fread(out.data(), 1, out.size(), f) != out.size()) { std::fprintf(stderr, "read failed\n"); std::exit(1); }
    std::fclose(f);
    return out;
}

static uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

static void png_collect(void* ctx, void* data, int size) {
    auto* v = (std::vector<uint8_t>*)ctx;
    v->insert(v->end(), (uint8_t*)data, (uint8_t*)data + size);
}

static std::vector<uint8_t> encode_png(const uint8_t* data, int w, int h, int comp) {
    std::vector<uint8_t> out;
    stbi_write_png_to_func(png_collect, &out, w, h, comp, data, w * comp);
    return out;
}

static std::vector<uint8_t> resize_u8(const uint8_t* src, int sw, int sh, int dw, int dh, int C) {
    std::vector<uint8_t> dst((size_t)dw * dh * C);
    for (int y = 0; y < dh; y++) {
        float sy = ((y + 0.5f) * sh / (float)dh) - 0.5f;
        int y0 = std::max(0, std::min(sh - 1, (int)std::floor(sy)));
        int y1 = std::max(0, std::min(sh - 1, y0 + 1));
        float fy = sy - std::floor(sy);
        for (int x = 0; x < dw; x++) {
            float sx = ((x + 0.5f) * sw / (float)dw) - 0.5f;
            int x0 = std::max(0, std::min(sw - 1, (int)std::floor(sx)));
            int x1 = std::max(0, std::min(sw - 1, x0 + 1));
            float fx = sx - std::floor(sx);
            for (int c = 0; c < C; c++) {
                float a = src[((size_t)y0 * sw + x0) * C + c] * (1.f - fx) + src[((size_t)y0 * sw + x1) * C + c] * fx;
                float b = src[((size_t)y1 * sw + x0) * C + c] * (1.f - fx) + src[((size_t)y1 * sw + x1) * C + c] * fx;
                float v = a * (1.f - fy) + b * fy;
                dst[((size_t)y * dw + x) * C + c] = (uint8_t)std::max(0.f, std::min(255.f, v + 0.5f));
            }
        }
    }
    return dst;
}

static std::vector<uint8_t> maybe_resize_png(const std::vector<uint8_t>& png, int max_dim, int comp) {
    if (max_dim <= 0) return png;
    int w = 0, h = 0, c = 0;
    uint8_t* img = stbi_load_from_memory(png.data(), (int)png.size(), &w, &h, &c, comp);
    if (!img) return png;
    int nw = w, nh = h;
    if (std::max(w, h) > max_dim) {
        if (w >= h) { nw = max_dim; nh = std::max(1, (int)std::lround(h * (double)max_dim / w)); }
        else { nh = max_dim; nw = std::max(1, (int)std::lround(w * (double)max_dim / h)); }
    }
    std::vector<uint8_t> out = (nw == w && nh == h) ? encode_png(img, w, h, comp)
                                                     : encode_png(resize_u8(img, w, h, nw, nh, comp).data(), nw, nh, comp);
    stbi_image_free(img);
    return out;
}

template<typename T>
static const T* accessor_ptr(const json& j, const std::vector<uint8_t>& bin, int ai) {
    const json& a = j["accessors"][ai];
    const json& bv = j["bufferViews"][(int)a["bufferView"]];
    size_t off = (size_t)bv.value("byteOffset", 0) + (size_t)a.value("byteOffset", 0);
    return (const T*)(bin.data() + off);
}

static std::vector<uint32_t> read_indices(const json& j, const std::vector<uint8_t>& bin, int ai) {
    const json& a = j["accessors"][ai];
    int ct = a["componentType"];
    size_t n = a["count"];
    std::vector<uint32_t> out(n);
    if (ct == 5125) {
        const uint32_t* p = accessor_ptr<uint32_t>(j, bin, ai);
        out.assign(p, p + n);
    } else if (ct == 5123) {
        const uint16_t* p = accessor_ptr<uint16_t>(j, bin, ai);
        for (size_t i = 0; i < n; i++) out[i] = p[i];
    } else {
        std::fprintf(stderr, "unsupported index componentType %d\n", ct);
        std::exit(1);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: glb_repack in.glb out.glb [texture_max_dim=0] [target_faces=0]\n");
        return 1;
    }
    int tex_max = argc > 3 ? std::atoi(argv[3]) : 0;
    int target_faces = argc > 4 ? std::atoi(argv[4]) : 0;
    std::vector<uint8_t> glb = read_file(argv[1]);
    if (glb.size() < 28 || rd32(glb.data()) != 0x46546C67u) { std::fprintf(stderr, "not a GLB\n"); return 1; }
    uint32_t json_len = rd32(glb.data() + 12), json_type = rd32(glb.data() + 16);
    if (json_type != 0x4E4F534Au) { std::fprintf(stderr, "missing JSON chunk\n"); return 1; }
    json j = json::parse((const char*)glb.data() + 20, (const char*)glb.data() + 20 + json_len);
    size_t bin_hdr = 20 + ((json_len + 3) & ~3u);
    uint32_t bin_len = rd32(glb.data() + bin_hdr), bin_type = rd32(glb.data() + bin_hdr + 4);
    if (bin_type != 0x004E4942u) { std::fprintf(stderr, "missing BIN chunk\n"); return 1; }
    std::vector<uint8_t> bin(glb.begin() + bin_hdr + 8, glb.begin() + bin_hdr + 8 + bin_len);

    const json& prim = j["meshes"][0]["primitives"][0];
    int ai_pos = prim["attributes"]["POSITION"];
    int ai_uv = prim["attributes"]["TEXCOORD_0"];
    int ai_nrm = prim["attributes"]["NORMAL"];
    int ai_idx = prim["indices"];
    uint32_t V = j["accessors"][ai_pos]["count"];
    uint32_t F3 = j["accessors"][ai_idx]["count"];
    const float* pos = accessor_ptr<float>(j, bin, ai_pos);
    const float* uv = accessor_ptr<float>(j, bin, ai_uv);
    const float* nrm = accessor_ptr<float>(j, bin, ai_nrm);
    std::vector<float> posv(pos, pos + (size_t)V * 3);
    std::vector<float> uvv(uv, uv + (size_t)V * 2);
    std::vector<float> nrmv(nrm, nrm + (size_t)V * 3);
    std::vector<uint32_t> idx = read_indices(j, bin, ai_idx);

    if (target_faces > 0 && (uint32_t)target_faces * 3 < F3) {
        std::vector<float> attrs((size_t)V * 5);
        for (uint32_t i = 0; i < V; i++) {
            attrs[(size_t)i * 5 + 0] = nrmv[(size_t)i * 3 + 0] * 0.25f;
            attrs[(size_t)i * 5 + 1] = nrmv[(size_t)i * 3 + 1] * 0.25f;
            attrs[(size_t)i * 5 + 2] = nrmv[(size_t)i * 3 + 2] * 0.25f;
            attrs[(size_t)i * 5 + 3] = uvv[(size_t)i * 2 + 0] * 8.0f;
            attrs[(size_t)i * 5 + 4] = uvv[(size_t)i * 2 + 1] * 8.0f;
        }
        float weights[5] = {0.25f, 0.25f, 0.25f, 8.0f, 8.0f};
        std::vector<uint32_t> simp(idx.size());
        float err = 0.f;
        unsigned int opts = std::getenv("REPACK_PERMISSIVE") ? meshopt_SimplifyPermissive : meshopt_SimplifyLockBorder;
        float target_error = std::getenv("REPACK_ERR") ? (float)std::atof(std::getenv("REPACK_ERR")) : 0.05f;
        size_t new_ic = meshopt_simplifyWithAttributes(simp.data(), idx.data(), idx.size(),
            posv.data(), V, 3 * sizeof(float), attrs.data(), 5 * sizeof(float), weights, 5,
            nullptr, (size_t)target_faces * 3, target_error, opts, &err);
        if (new_ic == 0) { std::fprintf(stderr, "simplify failed\n"); return 1; }
        simp.resize(new_ic);

        std::vector<uint32_t> remap(V, UINT32_MAX);
        std::vector<float> npos, nnrm, nuv;
        npos.reserve(posv.size()); nnrm.reserve(nrmv.size()); nuv.reserve(uvv.size());
        for (uint32_t& ii : simp) {
            uint32_t ni = remap[ii];
            if (ni == UINT32_MAX) {
                ni = (uint32_t)(npos.size() / 3);
                remap[ii] = ni;
                npos.insert(npos.end(), &posv[(size_t)ii * 3], &posv[(size_t)ii * 3 + 3]);
                nnrm.insert(nnrm.end(), &nrmv[(size_t)ii * 3], &nrmv[(size_t)ii * 3 + 3]);
                nuv.insert(nuv.end(), &uvv[(size_t)ii * 2], &uvv[(size_t)ii * 2 + 2]);
            }
            ii = ni;
        }
        posv.swap(npos); nrmv.swap(nnrm); uvv.swap(nuv); idx.swap(simp);
        V = (uint32_t)(posv.size() / 3);
        F3 = (uint32_t)idx.size();
        std::printf("[repack] simplify: target=%d actual=%u faces verts=%u err=%.4f\n", target_faces, F3 / 3, V, err);
    }

    std::vector<uint8_t> img0, img1;
    auto image_bytes = [&](int ii) {
        const json& im = j["images"][ii];
        const json& bv = j["bufferViews"][(int)im["bufferView"]];
        size_t off = (size_t)bv.value("byteOffset", 0), len = (size_t)bv["byteLength"];
        return std::vector<uint8_t>(bin.begin() + off, bin.begin() + off + len);
    };
    img0 = maybe_resize_png(image_bytes(0), tex_max, 4);
    img1 = maybe_resize_png(image_bytes(1), tex_max, 3);

#ifdef REPACK_INPROC_BUILD
    // REPACK_INPROC=hero|small: decode the PNG atlases to pixels and write a fully in-process compressed
    // GLB (meshopt geo + KTX2 tex) — validates the native packer at real scale, no GPU. Compare size/look
    // vs the gltfpack output.
    if (const char* mode = std::getenv("REPACK_INPROC")) {
        int w0,h0,c0, w1,h1,c1;
        uint8_t* p0 = stbi_load_from_memory(img0.data(), (int)img0.size(), &w0,&h0,&c0, 4);
        uint8_t* p1 = stbi_load_from_memory(img1.data(), (int)img1.size(), &w1,&h1,&c1, 3);
        if (!p0 || !p1) { std::fprintf(stderr, "[inproc] PNG decode failed\n"); return 1; }
        std::vector<uint8_t> base(p0, p0 + (size_t)w0*h0*4), mr(p1, p1 + (size_t)w1*h1*3);
        stbi_image_free(p0); stbi_image_free(p1);
        bool uastc = std::strcmp(mode, "small") != 0;
        bool ok = glb::write_glb_textured_packed(argv[2], posv, nrmv, uvv, idx, base, mr, w0, h0, uastc, 192,
                                                 (int)std::thread::hardware_concurrency());
        std::printf("[inproc] %s -> %s (%s: meshopt+KTX2, %ux%u v, %ux%u tex)\n",
                    argv[1], argv[2], uastc?"hero/UASTC":"small/ETC1S", V, F3/3, w0, h0);
        return ok ? 0 : 1;
    }
#endif

    std::vector<int16_t> qpos((size_t)V * 3);
    std::vector<int8_t> qnrm((size_t)V * 3);
    std::vector<uint16_t> quv((size_t)V * 2);
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < V; i++) for (int d = 0; d < 3; d++) { float v = posv[(size_t)i * 3 + d]; mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v); }
    float center[3] = {(mn[0] + mx[0]) * 0.5f, (mn[1] + mx[1]) * 0.5f, (mn[2] + mx[2]) * 0.5f};
    float radius = std::max({mx[0] - center[0], mx[1] - center[1], mx[2] - center[2], 1e-6f});
    float qmn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, qmx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < V; i++) {
        for (int d = 0; d < 3; d++) {
            float v = std::max(-1.f, std::min(1.f, (posv[(size_t)i * 3 + d] - center[d]) / radius));
            qpos[(size_t)i * 3 + d] = (int16_t)meshopt_quantizeSnorm(v, 16);
            qmn[d] = std::min(qmn[d], v); qmx[d] = std::max(qmx[d], v);
        }
        for (int d = 0; d < 3; d++) qnrm[(size_t)i * 3 + d] = (int8_t)meshopt_quantizeSnorm(std::max(-1.f, std::min(1.f, nrmv[(size_t)i * 3 + d])), 8);
        for (int d = 0; d < 2; d++) quv[(size_t)i * 2 + d] = (uint16_t)meshopt_quantizeUnorm(std::max(0.f, std::min(1.f, uvv[(size_t)i * 2 + d])), 16);
    }

    auto pad4 = [](uint32_t n) { return (4 - (n & 3)) & 3; };
    const uint32_t POS = V * 3 * 2, NRM = V * 3, UV = V * 2 * 2, IDX = F3 * 4;
    const uint32_t P0 = (uint32_t)img0.size(), P1 = (uint32_t)img1.size();
    uint32_t off = 0;
    auto alloc = [&](uint32_t len) { uint32_t o = off; off += len + pad4(len); return o; };
    uint32_t POS_OFF = alloc(POS), NRM_OFF = alloc(NRM), UV_OFF = alloc(UV), IDX_OFF = alloc(IDX), P0_OFF = alloc(P0), P1_OFF = alloc(P1);
    uint32_t BIN_LEN = off;

    json o;
    o["asset"] = {{"version", "2.0"}, {"generator", "pixal3d-cpp-repack"}};
    o["extensionsUsed"] = {"KHR_mesh_quantization"};
    o["scene"] = 0; o["scenes"] = {{{"nodes", {0}}}};
    o["nodes"] = {{{"mesh", 0}, {"translation", {center[0], center[1], center[2]}}, {"scale", {radius, radius, radius}}}};
    o["materials"] = {{{"name", "pbr"}, {"doubleSided", true}, {"alphaMode", "OPAQUE"}, {"pbrMetallicRoughness", {
        {"baseColorTexture", {{"index", 0}, {"texCoord", 0}}},
        {"baseColorFactor", {1, 1, 1, 1}},
        {"metallicRoughnessTexture", {{"index", 1}, {"texCoord", 0}}},
        {"metallicFactor", 1.0}, {"roughnessFactor", 1.0}}}}};
    o["meshes"] = {{{"primitives", {{{"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}}, {"indices", 3}, {"material", 0}, {"mode", 4}}}}}};
    o["samplers"] = {{{"magFilter", 9729}, {"minFilter", 9987}, {"wrapS", 10497}, {"wrapT", 10497}}};
    o["textures"] = {{{"source", 0}, {"sampler", 0}}, {{"source", 1}, {"sampler", 0}}};
    o["images"] = {{{"bufferView", 4}, {"mimeType", "image/png"}}, {{"bufferView", 5}, {"mimeType", "image/png"}}};
    o["buffers"] = {{{"byteLength", BIN_LEN}}};
    o["bufferViews"] = {{{"buffer", 0}, {"byteOffset", POS_OFF}, {"byteLength", POS}, {"target", 34962}},
        {{"buffer", 0}, {"byteOffset", NRM_OFF}, {"byteLength", NRM}, {"target", 34962}},
        {{"buffer", 0}, {"byteOffset", UV_OFF}, {"byteLength", UV}, {"target", 34962}},
        {{"buffer", 0}, {"byteOffset", IDX_OFF}, {"byteLength", IDX}, {"target", 34963}},
        {{"buffer", 0}, {"byteOffset", P0_OFF}, {"byteLength", P0}},
        {{"buffer", 0}, {"byteOffset", P1_OFF}, {"byteLength", P1}}};
    o["accessors"] = {{{"bufferView", 0}, {"componentType", 5122}, {"count", V}, {"type", "VEC3"}, {"normalized", true}, {"min", {qmn[0], qmn[1], qmn[2]}}, {"max", {qmx[0], qmx[1], qmx[2]}}},
        {{"bufferView", 1}, {"componentType", 5120}, {"count", V}, {"type", "VEC3"}, {"normalized", true}},
        {{"bufferView", 2}, {"componentType", 5123}, {"count", V}, {"type", "VEC2"}, {"normalized", true}},
        {{"bufferView", 3}, {"componentType", 5125}, {"count", F3}, {"type", "SCALAR"}}};

    std::string js = o.dump();
    uint32_t json_len2 = (uint32_t)js.size(), json_pad = pad4(json_len2), json_chunk = json_len2 + json_pad;
    uint32_t bin_pad = pad4(BIN_LEN), bin_chunk = BIN_LEN + bin_pad;
    uint32_t total = 12 + 8 + json_chunk + 8 + bin_chunk;
    FILE* f = std::fopen(argv[2], "wb");
    if (!f) { std::fprintf(stderr, "write failed: %s\n", argv[2]); return 1; }
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto wz = [&](uint32_t n) { for (uint32_t i = 0; i < n; i++) std::fputc(0, f); };
    w32(0x46546C67u); w32(2u); w32(total);
    w32(json_chunk); w32(0x4E4F534Au); std::fwrite(js.data(), 1, json_len2, f); for (uint32_t i = 0; i < json_pad; i++) std::fputc(' ', f);
    w32(bin_chunk); w32(0x004E4942u);
    std::fwrite(qpos.data(), 1, POS, f); wz(pad4(POS));
    std::fwrite(qnrm.data(), 1, NRM, f); wz(pad4(NRM));
    std::fwrite(quv.data(), 1, UV, f); wz(pad4(UV));
    std::fwrite(idx.data(), 1, IDX, f); wz(pad4(IDX));
    std::fwrite(img0.data(), 1, P0, f); wz(pad4(P0));
    std::fwrite(img1.data(), 1, P1, f); wz(pad4(P1));
    wz(bin_pad);
    std::fclose(f);
    std::printf("[repack] %s -> %s | %u verts / %u tris | tex_max=%d | %.2f MB\n",
        argv[1], argv[2], V, F3 / 3, tex_max, total / 1048576.0);
    return 0;
}
