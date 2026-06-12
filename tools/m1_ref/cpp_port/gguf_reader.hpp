// Minimal dependency-free GGUF reader (mmap) for the standalone sparse-VAE host port (m3a/m4),
// which must NOT link ggml. Reads fp32 tensors by name. Handles GGUF v3; generically skips kv
// metadata (our packer writes none). Used by svp::WLoad's GGUF path.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

struct GgufReader {
    int fd = -1;
    const uint8_t* base = nullptr;
    size_t fsize = 0;
    size_t data_off = 0;          // start of tensor-data section (aligned)
    struct Info { std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };
    std::unordered_map<std::string, Info> tensors;

    explicit GgufReader(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("gguf open failed: " + path);
        struct stat st; fstat(fd, &st); fsize = st.st_size;
        base = (const uint8_t*)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) throw std::runtime_error("gguf mmap failed: " + path);
        parse();
    }
    ~GgufReader() { if (base && base != MAP_FAILED) munmap((void*)base, fsize); if (fd >= 0) ::close(fd); }
    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    size_t p = 0;
    template<typename T> T rd() { T v; memcpy(&v, base + p, sizeof(T)); p += sizeof(T); return v; }
    std::string rdstr() { uint64_t n = rd<uint64_t>(); std::string s((const char*)(base + p), n); p += n; return s; }

    // skip one metadata value of the given gguf value-type
    void skip_val(uint32_t vt) {
        switch (vt) {
            case 0: case 1: case 7: p += 1; break;          // u8/i8/bool
            case 2: case 3: p += 2; break;                  // u16/i16
            case 4: case 5: case 6: p += 4; break;          // u32/i32/f32
            case 10: case 11: case 12: p += 8; break;       // u64/i64/f64
            case 8: { uint64_t n = rd<uint64_t>(); p += n; break; }   // string
            case 9: {                                       // array
                uint32_t et = rd<uint32_t>(); uint64_t cnt = rd<uint64_t>();
                for (uint64_t i = 0; i < cnt; i++) skip_val(et);
                break;
            }
            default: throw std::runtime_error("gguf: bad value type");
        }
    }

    void parse() {
        p = 0;
        uint32_t magic = rd<uint32_t>(); if (magic != 0x46554747u) throw std::runtime_error("not a GGUF file");
        uint32_t ver = rd<uint32_t>(); (void)ver;
        uint64_t n_tensors = rd<uint64_t>();
        uint64_t n_kv = rd<uint64_t>();
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rdstr(); uint32_t vt = rd<uint32_t>();
            if (key == "general.alignment" && vt == 4) { alignment = rd<uint32_t>(); }
            else skip_val(vt);
        }
        for (uint64_t i = 0; i < n_tensors; i++) {
            std::string name = rdstr();
            uint32_t nd = rd<uint32_t>();
            Info info; info.dims.resize(nd);
            for (uint32_t d = 0; d < nd; d++) info.dims[d] = rd<uint64_t>();
            info.type = rd<uint32_t>();
            info.offset = rd<uint64_t>();
            tensors[name] = std::move(info);
        }
        // data section start = current pos rounded up to alignment
        data_off = (p + alignment - 1) / alignment * alignment;
    }

    bool has(const std::string& name) const { return tensors.count(name) != 0; }

    // raw fp32 data for a tensor (copy). type 0 == GGML_TYPE_F32.
    std::vector<float> tensor_f32(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("gguf tensor not found: " + name);
        const Info& in = it->second;
        if (in.type != 0) throw std::runtime_error("gguf tensor not f32: " + name);
        uint64_t n = 1; for (uint64_t d : in.dims) n *= d;
        const float* src = (const float*)(base + data_off + in.offset);
        return std::vector<float>(src, src + n);
    }
};
