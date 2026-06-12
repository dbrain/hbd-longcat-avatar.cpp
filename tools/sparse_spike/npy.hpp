// Minimal .npy reader (little-endian, C-order) — just enough for the spike.
// Supports descr '<f4' (float32), '<f8' (float64), '<i4' (int32).
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyArray {
    std::vector<int64_t> shape;
    std::string descr;            // e.g. "<f4"
    std::vector<uint8_t> raw;     // raw bytes

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    // typed views (assumes caller knows dtype)
    const float*   f32() const { return reinterpret_cast<const float*>(raw.data()); }
    const double*  f64() const { return reinterpret_cast<const double*>(raw.data()); }
    const int32_t* i32() const { return reinterpret_cast<const int32_t*>(raw.data()); }
    const int64_t* i64() const { return reinterpret_cast<const int64_t*>(raw.data()); }
    const int8_t*  i8()  const { return reinterpret_cast<const int8_t*>(raw.data()); }
};

// element size for a numpy descr (little-endian / byte types).
static inline size_t npy_elem_size(const std::string& d) {
    if (d == "<f8" || d == "<i8" || d == "<u8") return 8;
    if (d == "<f4" || d == "<i4" || d == "<u4") return 4;
    if (d == "<f2" || d == "<i2" || d == "<u2") return 2;
    if (d == "|i1" || d == "|u1" || d == "|b1") return 1;
    return 4;  // default (legacy)
}

inline NpyArray npy_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy: bad magic " + path);
    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl = 0; f.read(reinterpret_cast<char*>(&hl), 2); header_len = hl;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    NpyArray a;
    // descr
    auto dpos = header.find("'descr'");
    auto q1 = header.find('\'', dpos + 7);
    auto q2 = header.find('\'', q1 + 1);
    a.descr = header.substr(q1 + 1, q2 - q1 - 1);
    if (header.find("'fortran_order': True") != std::string::npos)
        throw std::runtime_error("npy: fortran_order unsupported " + path);
    // shape
    auto spos = header.find("'shape'");
    auto lp = header.find('(', spos);
    auto rp = header.find(')', lp);
    std::string shp = header.substr(lp + 1, rp - lp - 1);
    size_t i = 0;
    while (i < shp.size()) {
        while (i < shp.size() && !isdigit(shp[i])) i++;
        if (i >= shp.size()) break;
        int64_t v = 0;
        while (i < shp.size() && isdigit(shp[i])) { v = v * 10 + (shp[i] - '0'); i++; }
        a.shape.push_back(v);
    }
    if (a.shape.empty()) a.shape.push_back(1);

    size_t elem = npy_elem_size(a.descr);
    a.raw.resize((size_t)a.numel() * elem);
    f.read(reinterpret_cast<char*>(a.raw.data()), a.raw.size());
    if (!f) throw std::runtime_error("npy: short read " + path);
    return a;
}
