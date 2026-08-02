#include "nvfp4_import.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "binary_io.h"
#include "core/util.h"
#include "ggml.h"
#include "json.hpp"

namespace {

// block_nvfp4 (ggml/src/ggml-common.h): uint8_t d[4]; uint8_t qs[32];
constexpr int64_t NVFP4_BLCK      = 64;
constexpr int64_t NVFP4_SUB       = 16;
constexpr int64_t NVFP4_SUBS      = NVFP4_BLCK / NVFP4_SUB;  // 4
constexpr size_t NVFP4_BLOCK_SIZE = NVFP4_SUBS + NVFP4_BLCK / 2;

constexpr size_t ST_HEADER_SIZE_LEN = 8;

struct RawEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0;  // absolute file offset
    uint64_t nbytes = 0;
};

// The shared safetensors reader drops U8 tensors, which is where the packed nibbles live,
// so recover them with an independent header parse.  Only reached once a
// weight_scale_2 / weight_global_scale entry has already proven this is such a checkpoint.
bool read_raw_header(const std::string& file_path, std::map<std::string, RawEntry>& out) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    uint8_t len_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)len_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        return false;
    }
    const size_t header_size = model_io::read_u64(len_buf);
    std::vector<char> header_buf(header_size + 1, '\0');
    file.read(header_buf.data(), (std::streamsize)header_size);
    if (!file) {
        return false;
    }
    nlohmann::json header;
    try {
        header = nlohmann::json::parse(header_buf.data());
    } catch (const std::exception&) {
        return false;
    }
    const uint64_t data_start = ST_HEADER_SIZE_LEN + header_size;
    for (auto& item : header.items()) {
        if (item.key() == "__metadata__" || !item.value().is_object()) {
            continue;
        }
        const nlohmann::json& info = item.value();
        if (!info.contains("dtype") || !info.contains("shape") || !info.contains("data_offsets")) {
            continue;
        }
        RawEntry e;
        e.dtype = info["dtype"].get<std::string>();
        for (const auto& d : info["shape"]) {
            e.shape.push_back(d.get<int64_t>());
        }
        const uint64_t begin = info["data_offsets"][0].get<uint64_t>();
        const uint64_t end   = info["data_offsets"][1].get<uint64_t>();
        if (end < begin) {
            return false;
        }
        e.offset      = data_start + begin;
        e.nbytes      = end - begin;
        out[item.key()] = std::move(e);
    }
    return true;
}

bool read_f32_at(const std::string& file_path, uint64_t offset, float* out) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.seekg((std::streamoff)offset);
    file.read((char*)out, sizeof(float));
    return (bool)file;
}

bool strip_suffix(const std::string& name, const char* suffix, std::string* base) {
    const size_t n = strlen(suffix);
    if (name.size() <= n || name.compare(name.size() - n, n, suffix) != 0) {
        return false;
    }
    *base = name.substr(0, name.size() - n);
    return true;
}

}  // namespace

void nvfp4_import_assemble_blocks(const uint8_t* packed,
                                  const uint8_t* scales,
                                  uint8_t* dst,
                                  int64_t in_features,
                                  int64_t out_features) {
    const int64_t nsub = in_features / NVFP4_SUB;
    const int64_t nblk = in_features / NVFP4_BLCK;

    for (int64_t r = 0; r < out_features; ++r) {
        const uint8_t* prow = packed + (size_t)r * (size_t)(in_features / 2);
        const uint8_t* srow = scales + (size_t)r * (size_t)nsub;
        uint8_t* drow       = dst + (size_t)r * (size_t)nblk * NVFP4_BLOCK_SIZE;

        for (int64_t b = 0; b < nblk; ++b) {
            uint8_t* blk = drow + (size_t)b * NVFP4_BLOCK_SIZE;
            uint8_t* qs  = blk + NVFP4_SUBS;
            for (int64_t s = 0; s < NVFP4_SUBS; ++s) {
                const int64_t ss = b * NVFP4_SUBS + s;
                blk[s]           = srow[ss];  // FP8 E4M3 group scale, verbatim

                // Source: 8 bytes holding elements 0..15 of this group as (2j low, 2j+1
                // high).  Destination: 8 bytes holding (j low, j+8 high).  Elements j and
                // j+8 always share a parity, so both come from the same nibble half.
                const uint8_t* od = prow + (size_t)ss * (NVFP4_SUB / 2);
                uint8_t* qsub     = qs + s * (NVFP4_SUB / 2);
                for (int j = 0; j < NVFP4_SUB / 2; ++j) {
                    const uint8_t lo_byte = od[j >> 1];
                    const uint8_t hi_byte = od[(j >> 1) + 4];
                    const uint8_t n0      = (j & 1) ? (uint8_t)(lo_byte >> 4) : (uint8_t)(lo_byte & 0x0F);
                    const uint8_t n1      = (j & 1) ? (uint8_t)(hi_byte >> 4) : (uint8_t)(hi_byte & 0x0F);
                    qsub[j]               = (uint8_t)(n0 | (n1 << 4));
                }
            }
        }
    }
}

bool nvfp4_import_materialize(const TensorStorage& tensor_storage,
                              size_t dst_nbytes,
                              const nvfp4_read_fn& read_at,
                              std::vector<uint8_t>& scratch,
                              std::vector<uint8_t>& out) {
    if (tensor_storage.is_inline_f32) {
        if (dst_nbytes != sizeof(float)) {
            LOG_ERROR("nvfp4 import: '%s' expects a 4-byte f32 destination, got %zu",
                      tensor_storage.name.c_str(),
                      dst_nbytes);
            return false;
        }
        out.resize(sizeof(float));
        memcpy(out.data(), &tensor_storage.inline_f32, sizeof(float));
        return true;
    }

    const int64_t in_features  = tensor_storage.ne[0];
    const int64_t out_features = tensor_storage.ne[1];
    const size_t packed_bytes  = (size_t)(in_features / 2) * (size_t)out_features;
    const size_t scale_bytes   = (size_t)(in_features / NVFP4_SUB) * (size_t)out_features;
    const size_t block_bytes   = (size_t)(in_features / NVFP4_BLCK) * (size_t)out_features * NVFP4_BLOCK_SIZE;

    if (dst_nbytes != block_bytes) {
        LOG_ERROR("nvfp4 import: '%s' destination is %zu bytes, expected %zu",
                  tensor_storage.name.c_str(),
                  dst_nbytes,
                  block_bytes);
        return false;
    }

    scratch.resize(packed_bytes + scale_bytes);
    if (!read_at((char*)scratch.data(), packed_bytes, tensor_storage.offset)) {
        return false;
    }
    if (!read_at((char*)scratch.data() + packed_bytes, scale_bytes, tensor_storage.nvfp4_scale_offset)) {
        return false;
    }

    out.resize(block_bytes);
    nvfp4_import_assemble_blocks(scratch.data(), scratch.data() + packed_bytes, out.data(), in_features, out_features);
    return true;
}

bool nvfp4_import_rewrite_safetensors(const std::string& file_path,
                                      std::vector<TensorStorage>& tensor_storages) {
    // Cheap bail: the per-tensor global is F32, so the shared reader has already surfaced
    // it if this is an external NVFP4 export.
    bool maybe_external = false;
    for (const TensorStorage& ts : tensor_storages) {
        std::string base;
        if (strip_suffix(ts.name, ".weight_scale_2", &base) ||
            strip_suffix(ts.name, ".weight_global_scale", &base)) {
            maybe_external = true;
            break;
        }
    }
    if (!maybe_external) {
        return true;
    }

    if (ggml_blck_size(GGML_TYPE_NVFP4) != NVFP4_BLCK ||
        ggml_type_size(GGML_TYPE_NVFP4) != NVFP4_BLOCK_SIZE) {
        LOG_ERROR("nvfp4 import: ggml block_nvfp4 layout changed (blck=%d size=%zu); refusing to import",
                  ggml_blck_size(GGML_TYPE_NVFP4),
                  ggml_type_size(GGML_TYPE_NVFP4));
        return false;
    }

    std::map<std::string, RawEntry> raw;
    if (!read_raw_header(file_path, raw)) {
        LOG_ERROR("nvfp4 import: failed to re-read safetensors header of '%s'", file_path.c_str());
        return false;
    }

    // base -> (global scalar tensor name, packed tensor name)
    std::map<std::string, std::pair<std::string, std::string>> linears;
    for (const auto& [name, entry] : raw) {
        std::string base;
        if (strip_suffix(name, ".weight_scale_2", &base)) {
            linears[base] = {name, base + ".weight"};
        } else if (strip_suffix(name, ".weight_global_scale", &base)) {
            linears[base] = {name, base + ".weight_packed"};
        }
    }
    if (linears.empty()) {
        return true;
    }

    std::map<std::string, TensorStorage> imported;  // new/replacement entries by name
    std::vector<std::string> drop;                  // source-only names to remove

    for (const auto& [base, names] : linears) {
        const std::string& global_name = names.first;
        const std::string& packed_name = names.second;
        const std::string scale_name   = base + ".weight_scale";

        if (raw.count(base + ".pre_quant_scale") != 0) {
            LOG_ERROR(
                "nvfp4 import: '%s' carries pre_quant_scale (ModelOpt AWQ input smoothing), which this "
                "engine cannot apply; importing it would silently mis-scale every activation",
                base.c_str());
            return false;
        }

        auto packed_it = raw.find(packed_name);
        auto scale_it  = raw.find(scale_name);
        auto global_it = raw.find(global_name);
        if (packed_it == raw.end() || scale_it == raw.end() || global_it == raw.end()) {
            LOG_ERROR("nvfp4 import: '%s' is missing one of {%s, %s, %s}",
                      base.c_str(),
                      packed_name.c_str(),
                      scale_name.c_str(),
                      global_name.c_str());
            return false;
        }

        const RawEntry& packed = packed_it->second;
        const RawEntry& scale  = scale_it->second;
        const RawEntry& global = global_it->second;

        if (packed.dtype != "U8" || packed.shape.size() != 2) {
            LOG_ERROR("nvfp4 import: '%s' is %s rank-%zu, expected U8 rank-2",
                      packed_name.c_str(),
                      packed.dtype.c_str(),
                      packed.shape.size());
            return false;
        }
        if (scale.dtype != "F8_E4M3" || scale.shape.size() != 2) {
            LOG_ERROR("nvfp4 import: '%s' is %s rank-%zu, expected F8_E4M3 rank-2",
                      scale_name.c_str(),
                      scale.dtype.c_str(),
                      scale.shape.size());
            return false;
        }
        if (global.dtype != "F32" || global.nbytes != sizeof(float)) {
            LOG_ERROR("nvfp4 import: '%s' is not an F32 scalar", global_name.c_str());
            return false;
        }

        const int64_t out_features = packed.shape[0];
        const int64_t in_features  = packed.shape[1] * 2;

        if (in_features % NVFP4_BLCK != 0) {
            // ggml's block is 64 wide with four 16-wide sub-scales.  A source row that is a
            // multiple of 16 but not of 64 has a valid group scale for every group yet no
            // whole block to put them in, and cannot be represented without repadding.
            LOG_ERROR("nvfp4 import: '%s' has in_features=%" PRId64 ", not a multiple of %" PRId64,
                      packed_name.c_str(),
                      in_features,
                      NVFP4_BLCK);
            return false;
        }
        if (scale.shape[0] != out_features || scale.shape[1] != in_features / NVFP4_SUB) {
            LOG_ERROR("nvfp4 import: '%s' shape [%" PRId64 ", %" PRId64 "] does not match a group size of %" PRId64,
                      scale_name.c_str(),
                      scale.shape[0],
                      scale.shape[1],
                      NVFP4_SUB);
            return false;
        }
        if (packed.nbytes != (uint64_t)(in_features / 2) * (uint64_t)out_features ||
            scale.nbytes != (uint64_t)(in_features / NVFP4_SUB) * (uint64_t)out_features) {
            LOG_ERROR("nvfp4 import: byte count does not match the declared shape for '%s'", base.c_str());
            return false;
        }

        float wglobal = 0.0f;
        if (!read_f32_at(file_path, global.offset, &wglobal)) {
            LOG_ERROR("nvfp4 import: failed to read '%s'", global_name.c_str());
            return false;
        }
        if (!(wglobal > 0.0f) || !std::isfinite(wglobal)) {
            LOG_ERROR("nvfp4 import: '%s' = %g is not a usable positive scale", global_name.c_str(), wglobal);
            return false;
        }

        const int64_t ne[SD_MAX_DIMS] = {in_features, out_features, 1, 1, 1};
        TensorStorage w(base + ".weight", GGML_TYPE_NVFP4, ne, 2, 0, packed.offset);
        w.is_nvfp4_import     = true;
        w.nvfp4_scale_offset  = scale.offset;
        imported[w.name]      = w;

        const int64_t ne1[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
        TensorStorage g(base + ".weight.wglobal", GGML_TYPE_F32, ne1, 1, 0, 0);
        g.is_inline_f32  = true;
        g.inline_f32     = wglobal;
        imported[g.name] = g;

        // These three are fully consumed above and must not survive into the tensor map.
        // weight_scale in particular COLLIDES BY NAME with Linear's unrelated
        // per-output-channel F32 `weight_scale` feature (ggml_extend.hpp): any Linear
        // constructed with allow_weight_scale would bind this [out, in/16] FP8 tensor to an
        // [out] F32 parameter.  input_scale has no consumer at all -- the NVFP4 GEMM
        // quantises activations dynamically per call.
        drop.push_back(scale_name);
        drop.push_back(global_name);
        drop.push_back(base + ".input_scale");
    }

    std::set<std::string> drop_set(drop.begin(), drop.end());
    std::vector<TensorStorage> result;
    result.reserve(tensor_storages.size() + imported.size());
    for (TensorStorage& ts : tensor_storages) {
        if (drop_set.count(ts.name) != 0) {
            continue;
        }
        auto it = imported.find(ts.name);
        if (it != imported.end()) {
            continue;  // replaced below
        }
        result.push_back(std::move(ts));
    }
    for (auto& [name, ts] : imported) {
        result.push_back(ts);
    }
    tensor_storages = std::move(result);

    LOG_INFO("nvfp4 import: %zu externally-quantised NVFP4 linears in '%s'", linears.size(), file_path.c_str());
    return true;
}
