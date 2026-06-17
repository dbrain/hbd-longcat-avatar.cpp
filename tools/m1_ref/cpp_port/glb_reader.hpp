// Minimal glTF 2.0 binary (.glb) READER — mirror of glb_writer.hpp's data conventions:
//   mesh = std::vector<float> verts (V*3, interleaved xyz) + std::vector<int64_t> faces (F*3).
//   Optional uvs (V*2, TEXCOORD_0) + normals (V*3).
// Header-only, no external deps (vendors a tiny self-contained JSON parser). CPU only.
//
// Scope: the single embedded GLB binary buffer (buffer 0 = the BIN chunk). Supports
//   POSITION (VEC3 f32), indices (SCALAR u8/u16/u32), TEXCOORD_0 (VEC2 f32), NORMAL (VEC3 f32).
//   Handles accessor byteOffset + bufferView byteStride (tight or interleaved). Concatenates all
//   primitives across all meshes (offsetting indices). Returns false on malformed input (no crash).
//   External/base64 buffer URIs are out of scope -> false with a clear stderr message.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

namespace glb {

struct Mesh {
    std::vector<float>   verts;    // V*3
    std::vector<int64_t> faces;    // F*3
    std::vector<float>   uvs;      // V*2, empty if absent
    std::vector<float>   normals;  // V*3, empty if absent
};

// ----------------------------------------------------------------------------
// Tiny self-contained JSON parser. Supports objects, arrays, strings, numbers,
// true/false/null. Enough for a glTF JSON chunk. Numbers stored as double.
// ----------------------------------------------------------------------------
namespace detail {

struct JVal {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool                       b = false;
    double                     num = 0.0;
    std::string                str;
    std::vector<JVal>          arr;
    std::map<std::string, JVal> obj;

    bool is_obj() const { return type == Obj; }
    bool is_arr() const { return type == Arr; }
    bool is_num() const { return type == Num; }
    bool is_str() const { return type == Str; }

    // object member access (returns nullptr if missing / not an object)
    const JVal* find(const char* key) const {
        if (type != Obj) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
    // integer with default
    int64_t as_int(int64_t dflt = 0) const { return type == Num ? (int64_t)num : dflt; }
    double  as_double(double dflt = 0.0) const { return type == Num ? num : dflt; }
};

class JParser {
public:
    JParser(const char* p, size_t n) : s_(p), n_(n), i_(0) {}
    bool parse(JVal& out) {
        skip_ws();
        if (!value(out)) return false;
        skip_ws();
        return true;  // trailing content tolerated (glTF JSON chunk may be space-padded)
    }
private:
    const char* s_;
    size_t      n_, i_;

    void skip_ws() {
        while (i_ < n_) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i_++;
            else break;
        }
    }
    bool eof() const { return i_ >= n_; }
    char peek() const { return s_[i_]; }

    bool value(JVal& v) {
        skip_ws();
        if (eof()) return false;
        char c = peek();
        if (c == '{') return object(v);
        if (c == '[') return array(v);
        if (c == '"') { v.type = JVal::Str; return string(v.str); }
        if (c == 't' || c == 'f') return boolean(v);
        if (c == 'n') return null(v);
        if (c == '-' || (c >= '0' && c <= '9')) return number(v);
        return false;
    }

    bool object(JVal& v) {
        v.type = JVal::Obj;
        i_++;  // '{'
        skip_ws();
        if (!eof() && peek() == '}') { i_++; return true; }
        while (true) {
            skip_ws();
            if (eof() || peek() != '"') return false;
            std::string key;
            if (!string(key)) return false;
            skip_ws();
            if (eof() || peek() != ':') return false;
            i_++;  // ':'
            JVal child;
            if (!value(child)) return false;
            v.obj.emplace(std::move(key), std::move(child));
            skip_ws();
            if (eof()) return false;
            char c = peek();
            if (c == ',') { i_++; continue; }
            if (c == '}') { i_++; return true; }
            return false;
        }
    }

    bool array(JVal& v) {
        v.type = JVal::Arr;
        i_++;  // '['
        skip_ws();
        if (!eof() && peek() == ']') { i_++; return true; }
        while (true) {
            JVal child;
            if (!value(child)) return false;
            v.arr.push_back(std::move(child));
            skip_ws();
            if (eof()) return false;
            char c = peek();
            if (c == ',') { i_++; continue; }
            if (c == ']') { i_++; return true; }
            return false;
        }
    }

    bool string(std::string& out) {
        if (eof() || peek() != '"') return false;
        i_++;  // opening quote
        out.clear();
        while (!eof()) {
            char c = s_[i_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (eof()) return false;
                char e = s_[i_++];
                switch (e) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        if (i_ + 4 > n_) return false;
                        unsigned cp = 0;
                        for (int k = 0; k < 4; k++) {
                            char h = s_[i_++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            else return false;
                        }
                        // minimal UTF-8 encode (BMP only; surrogate pairs not needed for glTF keys)
                        if (cp < 0x80) out.push_back((char)cp);
                        else if (cp < 0x800) {
                            out.push_back((char)(0xC0 | (cp >> 6)));
                            out.push_back((char)(0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back((char)(0xE0 | (cp >> 12)));
                            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back((char)(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default: return false;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;  // unterminated
    }

    bool number(JVal& v) {
        size_t start = i_;
        if (!eof() && peek() == '-') i_++;
        while (!eof()) {
            char c = peek();
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') i_++;
            else break;
        }
        if (i_ == start) return false;
        std::string tok(s_ + start, i_ - start);
        v.type = JVal::Num;
        v.num = std::strtod(tok.c_str(), nullptr);
        return true;
    }

    bool boolean(JVal& v) {
        if (i_ + 4 <= n_ && std::strncmp(s_ + i_, "true", 4) == 0) {
            v.type = JVal::Bool; v.b = true; i_ += 4; return true;
        }
        if (i_ + 5 <= n_ && std::strncmp(s_ + i_, "false", 5) == 0) {
            v.type = JVal::Bool; v.b = false; i_ += 5; return true;
        }
        return false;
    }

    bool null(JVal& v) {
        if (i_ + 4 <= n_ && std::strncmp(s_ + i_, "null", 4) == 0) {
            v.type = JVal::Null; i_ += 4; return true;
        }
        return false;
    }
};

// glTF accessor componentType byte sizes
inline int comp_size(int ct) {
    switch (ct) {
        case 5120: case 5121: return 1;  // byte / unsigned byte
        case 5122: case 5123: return 2;  // short / unsigned short
        case 5125: case 5126: return 4;  // unsigned int / float
        default: return 0;
    }
}
inline int type_count(const std::string& t) {
    if (t == "SCALAR") return 1;
    if (t == "VEC2")   return 2;
    if (t == "VEC3")   return 3;
    if (t == "VEC4")   return 4;
    if (t == "MAT2")   return 4;
    if (t == "MAT3")   return 9;
    if (t == "MAT4")   return 16;
    return 0;
}

// Resolved accessor view into the BIN blob.
struct AccessorView {
    const uint8_t* base = nullptr;  // pointer to first element (bufferView.byteOffset + accessor.byteOffset)
    int64_t        count = 0;       // element count
    int            ncomp = 0;       // components per element (from type)
    int            comp_ct = 0;     // componentType
    int            csize = 0;       // bytes per component
    int            stride = 0;      // byte stride between consecutive elements
    bool valid() const { return base != nullptr && ncomp > 0 && csize > 0 && count >= 0; }
};

}  // namespace detail

// ----------------------------------------------------------------------------
// read_glb
// ----------------------------------------------------------------------------
inline bool read_glb(const char* path, Mesh& out) {
    using namespace detail;
    out = Mesh{};

    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "glb::read_glb: cannot open '%s'\n", path); return false; }
    std::fseek(f, 0, SEEK_END);
    long fsz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsz < 12) { std::fprintf(stderr, "glb::read_glb: file too small\n"); std::fclose(f); return false; }
    std::vector<uint8_t> buf((size_t)fsz);
    if (std::fread(buf.data(), 1, (size_t)fsz, f) != (size_t)fsz) {
        std::fprintf(stderr, "glb::read_glb: short read\n"); std::fclose(f); return false;
    }
    std::fclose(f);

    auto rd32 = [&](size_t off) -> uint32_t {
        uint32_t v; std::memcpy(&v, buf.data() + off, 4); return v;
    };

    // ---- 12-byte header ----
    if (rd32(0) != 0x46546C67u) { std::fprintf(stderr, "glb::read_glb: bad magic\n"); return false; }
    if (rd32(4) != 2u)          { std::fprintf(stderr, "glb::read_glb: unsupported version %u\n", rd32(4)); return false; }
    uint32_t total = rd32(8);
    if (total > (uint32_t)fsz) { std::fprintf(stderr, "glb::read_glb: declared length exceeds file\n"); return false; }

    // ---- chunks ----
    const char*  json_ptr = nullptr; size_t json_len = 0;
    const uint8_t* bin_ptr = nullptr; size_t bin_len = 0;
    size_t off = 12;
    while (off + 8 <= (size_t)total) {
        uint32_t clen = rd32(off);
        uint32_t ctype = rd32(off + 4);
        size_t cstart = off + 8;
        if (cstart + clen > (size_t)total) { std::fprintf(stderr, "glb::read_glb: chunk overruns file\n"); return false; }
        if (ctype == 0x4E4F534Au) {          // "JSON"
            json_ptr = (const char*)(buf.data() + cstart);
            json_len = clen;
        } else if (ctype == 0x004E4942u) {   // "BIN\0"
            bin_ptr = buf.data() + cstart;
            bin_len = clen;
        }
        off = cstart + clen;
    }
    if (!json_ptr) { std::fprintf(stderr, "glb::read_glb: no JSON chunk\n"); return false; }

    // ---- parse JSON ----
    JVal root;
    if (!JParser(json_ptr, json_len).parse(root) || !root.is_obj()) {
        std::fprintf(stderr, "glb::read_glb: JSON parse failed\n"); return false;
    }

    const JVal* buffers   = root.find("buffers");
    const JVal* bufviews  = root.find("bufferViews");
    const JVal* accessors = root.find("accessors");
    const JVal* meshes    = root.find("meshes");
    if (!meshes || !meshes->is_arr() || !accessors || !accessors->is_arr() ||
        !bufviews || !bufviews->is_arr()) {
        std::fprintf(stderr, "glb::read_glb: missing meshes/accessors/bufferViews\n"); return false;
    }

    // buffer 0 must be the embedded BIN chunk (no uri).
    if (buffers && buffers->is_arr() && !buffers->arr.empty()) {
        const JVal& b0 = buffers->arr[0];
        if (b0.find("uri")) {
            std::fprintf(stderr, "glb::read_glb: buffer 0 has a 'uri' (external/base64) — out of scope\n");
            return false;
        }
    }
    if (!bin_ptr) { std::fprintf(stderr, "glb::read_glb: no BIN chunk\n"); return false; }

    // Resolve an accessor index -> a view into the BIN blob. Returns valid()==false on failure.
    auto resolve = [&](int64_t acc_idx) -> AccessorView {
        AccessorView av;
        if (acc_idx < 0 || acc_idx >= (int64_t)accessors->arr.size()) return av;
        const JVal& acc = accessors->arr[(size_t)acc_idx];
        const JVal* bvj = acc.find("bufferView");
        if (!bvj) return av;  // sparse / no bufferView — unsupported
        int64_t bv_idx = bvj->as_int(-1);
        if (bv_idx < 0 || bv_idx >= (int64_t)bufviews->arr.size()) return av;
        const JVal& bv = bufviews->arr[(size_t)bv_idx];

        int64_t buf_idx = 0;
        if (const JVal* bj = bv.find("buffer")) buf_idx = bj->as_int(0);
        if (buf_idx != 0) return av;  // only embedded buffer 0 supported

        int64_t bv_off = bv.find("byteOffset") ? bv.find("byteOffset")->as_int(0) : 0;
        int64_t bv_len = bv.find("byteLength") ? bv.find("byteLength")->as_int(0) : 0;
        int64_t bv_stride = bv.find("byteStride") ? bv.find("byteStride")->as_int(0) : 0;

        int64_t acc_off = acc.find("byteOffset") ? acc.find("byteOffset")->as_int(0) : 0;
        int64_t count = acc.find("count") ? acc.find("count")->as_int(0) : 0;
        int comp_ct = acc.find("componentType") ? (int)acc.find("componentType")->as_int(0) : 0;
        const JVal* tj = acc.find("type");
        std::string type = (tj && tj->is_str()) ? tj->str : std::string();

        int ncomp = type_count(type);
        int csize = comp_size(comp_ct);
        if (ncomp == 0 || csize == 0 || count < 0) return av;

        int elem_size = ncomp * csize;
        int stride = bv_stride > 0 ? (int)bv_stride : elem_size;

        int64_t start = bv_off + acc_off;
        // bounds: last element must fit within the bufferView (and the BIN blob).
        if (count > 0) {
            int64_t last = start + (count - 1) * (int64_t)stride + elem_size;
            int64_t bv_end = bv_off + (bv_len > 0 ? bv_len : last - bv_off);
            if (last > bv_end) return av;
            if ((size_t)last > bin_len) return av;
        }
        av.base = bin_ptr + start;
        av.count = count;
        av.ncomp = ncomp;
        av.comp_ct = comp_ct;
        av.csize = csize;
        av.stride = stride;
        return av;
    };

    // Read a strided index element as int64 (u8/u16/u32, or signed variants).
    auto read_index = [](const AccessorView& av, int64_t i) -> int64_t {
        const uint8_t* p = av.base + i * (int64_t)av.stride;
        switch (av.comp_ct) {
            case 5121: return (int64_t)(*(const uint8_t*)p);
            case 5123: { uint16_t v; std::memcpy(&v, p, 2); return (int64_t)v; }
            case 5125: { uint32_t v; std::memcpy(&v, p, 4); return (int64_t)v; }
            case 5120: return (int64_t)(*(const int8_t*)p);
            case 5122: { int16_t v; std::memcpy(&v, p, 2); return (int64_t)v; }
            default:   return -1;
        }
    };
    // Read a strided float component.
    auto read_float = [](const AccessorView& av, int64_t i, int c) -> float {
        const uint8_t* p = av.base + i * (int64_t)av.stride + (int64_t)c * av.csize;
        float v; std::memcpy(&v, p, 4); return v;
    };

    // ---- iterate meshes -> primitives, concatenating ----
    bool any_prim = false;
    for (const JVal& mesh : meshes->arr) {
        const JVal* prims = mesh.find("primitives");
        if (!prims || !prims->is_arr()) continue;
        for (const JVal& prim : prims->arr) {
            const JVal* attrs = prim.find("attributes");
            if (!attrs || !attrs->is_obj()) continue;
            const JVal* posj = attrs->find("POSITION");
            if (!posj) continue;  // need positions

            AccessorView pos = resolve(posj->as_int(-1));
            if (!pos.valid() || pos.comp_ct != 5126 || pos.ncomp != 3) {
                std::fprintf(stderr, "glb::read_glb: POSITION accessor invalid/unsupported\n");
                return false;
            }
            int64_t base_v = (int64_t)(out.verts.size() / 3);
            int64_t V = pos.count;

            // POSITION
            for (int64_t v = 0; v < V; v++)
                for (int c = 0; c < 3; c++) out.verts.push_back(read_float(pos, v, c));

            // NORMAL (optional). Pad with previous behavior: keep arrays index-aligned.
            const JVal* nrmj = attrs->find("NORMAL");
            bool have_nrm = false;
            if (nrmj) {
                AccessorView nv = resolve(nrmj->as_int(-1));
                if (nv.valid() && nv.comp_ct == 5126 && nv.ncomp == 3 && nv.count == V) {
                    have_nrm = true;
                    for (int64_t v = 0; v < V; v++)
                        for (int c = 0; c < 3; c++) out.normals.push_back(read_float(nv, v, c));
                }
            }
            if (!have_nrm && !out.normals.empty()) {
                // keep alignment if a prior primitive had normals
                out.normals.insert(out.normals.end(), (size_t)V * 3, 0.f);
            }

            // TEXCOORD_0 (optional, VEC2 f32 only for the float fast-path)
            const JVal* uvj = attrs->find("TEXCOORD_0");
            bool have_uv = false;
            if (uvj) {
                AccessorView uv = resolve(uvj->as_int(-1));
                if (uv.valid() && uv.comp_ct == 5126 && uv.ncomp == 2 && uv.count == V) {
                    have_uv = true;
                    for (int64_t v = 0; v < V; v++)
                        for (int c = 0; c < 2; c++) out.uvs.push_back(read_float(uv, v, c));
                }
            }
            if (!have_uv && !out.uvs.empty()) {
                out.uvs.insert(out.uvs.end(), (size_t)V * 2, 0.f);
            }

            // indices (optional — if absent, glTF implies a sequential non-indexed draw)
            const JVal* idxj = prim.find("indices");
            if (idxj) {
                AccessorView iv = resolve(idxj->as_int(-1));
                if (!iv.valid() || iv.ncomp != 1) {
                    std::fprintf(stderr, "glb::read_glb: indices accessor invalid\n");
                    return false;
                }
                for (int64_t i = 0; i < iv.count; i++) {
                    int64_t idx = read_index(iv, i);
                    if (idx < 0) { std::fprintf(stderr, "glb::read_glb: bad index componentType\n"); return false; }
                    out.faces.push_back(base_v + idx);
                }
            } else {
                // non-indexed: implicit 0,1,2,... over the vertices of this primitive
                for (int64_t i = 0; i < V; i++) out.faces.push_back(base_v + i);
            }
            any_prim = true;
        }
    }

    if (!any_prim) { std::fprintf(stderr, "glb::read_glb: no primitive with POSITION found\n"); return false; }
    // discard partial normal/uv arrays that never got real data
    bool nrm_real = out.normals.size() == out.verts.size();
    if (!nrm_real) out.normals.clear();
    // uvs are V*2; valid only if fully sized
    if (out.uvs.size() != (out.verts.size() / 3) * 2) out.uvs.clear();
    (void)nrm_real;
    return true;
}

// ----------------------------------------------------------------------------
// read_glb_basecolor_png
//   Extract the EMBEDDED baseColor texture image bytes (PNG/JPEG) + mime type from a GLB:
//     materials[0].pbrMetallicRoughness.baseColorTexture.index
//       -> textures[texIdx].source
//       -> images[imgIdx].bufferView  (embedded in the BIN chunk)
//   Fills out_png (raw image bytes) + out_mime (e.g. "image/png"). Returns true iff a
//   baseColor image was found and is embedded via a bufferView in buffer 0. Returns false
//   (out_png cleared) when there is no material/texture, or the image uses an external/uri
//   source (out of scope — the caller can then fall back to a flat material).
//
//   Header-only, no external deps (reuses the tiny JSON parser above). CPU only.
// ----------------------------------------------------------------------------
inline bool read_glb_basecolor_png(const char* path,
                                   std::vector<uint8_t>& out_png,
                                   std::string&          out_mime) {
    using namespace detail;
    out_png.clear();
    out_mime.clear();

    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "glb::read_glb_basecolor_png: cannot open '%s'\n", path); return false; }
    std::fseek(f, 0, SEEK_END);
    long fsz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsz < 12) { std::fclose(f); return false; }
    std::vector<uint8_t> buf((size_t)fsz);
    if (std::fread(buf.data(), 1, (size_t)fsz, f) != (size_t)fsz) { std::fclose(f); return false; }
    std::fclose(f);

    auto rd32 = [&](size_t off) -> uint32_t { uint32_t v; std::memcpy(&v, buf.data() + off, 4); return v; };
    if (rd32(0) != 0x46546C67u || rd32(4) != 2u) return false;
    uint32_t total = rd32(8);
    if (total > (uint32_t)fsz) return false;

    const char*    json_ptr = nullptr; size_t json_len = 0;
    const uint8_t* bin_ptr = nullptr;  size_t bin_len  = 0;
    size_t off = 12;
    while (off + 8 <= (size_t)total) {
        uint32_t clen = rd32(off), ctype = rd32(off + 4);
        size_t cstart = off + 8;
        if (cstart + clen > (size_t)total) return false;
        if (ctype == 0x4E4F534Au) { json_ptr = (const char*)(buf.data() + cstart); json_len = clen; }
        else if (ctype == 0x004E4942u) { bin_ptr = buf.data() + cstart; bin_len = clen; }
        off = cstart + clen;
    }
    if (!json_ptr) return false;

    JVal root;
    if (!JParser(json_ptr, json_len).parse(root) || !root.is_obj()) return false;

    const JVal* materials = root.find("materials");
    const JVal* textures  = root.find("textures");
    const JVal* images    = root.find("images");
    if (!materials || !materials->is_arr() || materials->arr.empty()) return false;
    if (!images || !images->is_arr() || images->arr.empty()) return false;

    // materials[0].pbrMetallicRoughness.baseColorTexture.index -> texture index
    const JVal& m0 = materials->arr[0];
    const JVal* pbr = m0.find("pbrMetallicRoughness");
    if (!pbr || !pbr->is_obj()) return false;
    const JVal* bct = pbr->find("baseColorTexture");
    if (!bct || !bct->is_obj()) return false;
    const JVal* tij = bct->find("index");
    if (!tij) return false;
    int64_t tex_idx = tij->as_int(-1);

    // texture index -> source image index
    int64_t img_idx = tex_idx;  // some assets omit a textures[] and index images directly; prefer textures[] when present
    if (textures && textures->is_arr() && tex_idx >= 0 && tex_idx < (int64_t)textures->arr.size()) {
        const JVal* srcj = textures->arr[(size_t)tex_idx].find("source");
        if (srcj) img_idx = srcj->as_int(-1);
    }
    if (img_idx < 0 || img_idx >= (int64_t)images->arr.size()) return false;

    const JVal& img = images->arr[(size_t)img_idx];
    // mime type (default to image/png if absent)
    const JVal* mimej = img.find("mimeType");
    out_mime = (mimej && mimej->is_str()) ? mimej->str : std::string("image/png");

    // external URI is out of scope
    if (img.find("uri")) {
        std::fprintf(stderr, "glb::read_glb_basecolor_png: baseColor image has a 'uri' (external) — out of scope\n");
        return false;
    }
    const JVal* bvj = img.find("bufferView");
    if (!bvj) return false;
    int64_t bv_idx = bvj->as_int(-1);

    const JVal* bufviews = root.find("bufferViews");
    if (!bufviews || !bufviews->is_arr() || bv_idx < 0 || bv_idx >= (int64_t)bufviews->arr.size()) return false;
    const JVal& bv = bufviews->arr[(size_t)bv_idx];

    int64_t buf_idx = bv.find("buffer") ? bv.find("buffer")->as_int(0) : 0;
    if (buf_idx != 0 || !bin_ptr) return false;  // only the embedded BIN buffer
    int64_t bv_off = bv.find("byteOffset") ? bv.find("byteOffset")->as_int(0) : 0;
    int64_t bv_len = bv.find("byteLength") ? bv.find("byteLength")->as_int(0) : 0;
    if (bv_len <= 0 || bv_off < 0 || (size_t)(bv_off + bv_len) > bin_len) return false;

    out_png.assign(bin_ptr + bv_off, bin_ptr + bv_off + bv_len);
    return true;
}

}  // namespace glb
