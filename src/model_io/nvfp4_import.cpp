#include "nvfp4_import.h"

#include <algorithm>
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

// cuBLAS block-scaling-factor tiling, as emitted by comfy-kitchen's to_blocked() /
// scale_factor_swizzled_offset(): a (rows, cols) scale plane is zero-padded to
// (roundup(rows, 128), roundup(cols, 4)) and then reordered so a tensor-core MMA can read
// its scales contiguously.  128-row base blocks, 4-group columns, (32, 4, 4) inner order.
constexpr int64_t SWZ_ROWS_PER_BASE_BLOCK = 128;
constexpr int64_t SWZ_ROWS_PER_COL        = 32;
constexpr int64_t SWZ_COLS_PER_COL        = 4;

inline int64_t roundup_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Byte offset of the scale for output row `o`, 16-group `g`, in a plane whose padded group
// count is `n_col_blocks` * 4.  Verbatim port of comfy-kitchen's C++/CUDA helper.
inline size_t swizzled_scale_offset(int64_t o, int64_t g, int64_t n_col_blocks) {
    const int64_t rb  = o / SWZ_ROWS_PER_BASE_BLOCK;
    const int64_t rem = o % SWZ_ROWS_PER_BASE_BLOCK;
    const int64_t d4  = rem / SWZ_ROWS_PER_COL;
    const int64_t d3  = rem % SWZ_ROWS_PER_COL;
    const int64_t cbg = g / SWZ_COLS_PER_COL;
    const int64_t d5  = g % SWZ_COLS_PER_COL;
    return (size_t)(((rb * n_col_blocks + cbg) * SWZ_ROWS_PER_COL + d3) * 16 + d4 * SWZ_COLS_PER_COL + d5);
}

// Total bytes of a ComfyUI scale plane for an [out, in] weight.
inline size_t comfy_scale_plane_bytes(int64_t in_features, int64_t out_features) {
    const int64_t padded_rows = roundup_div(out_features, SWZ_ROWS_PER_BASE_BLOCK) * SWZ_ROWS_PER_BASE_BLOCK;
    const int64_t padded_cols = roundup_div(in_features / NVFP4_SUB, SWZ_COLS_PER_COL) * SWZ_COLS_PER_COL;
    return (size_t)padded_rows * (size_t)padded_cols;
}

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

namespace {

// `src_hi_first`  : element 2j is in the HIGH nibble of each source byte (ComfyUI) rather
//                   than the LOW one (ModelOpt / llm-compressor).
// `src_swizzled`  : `scales` is a cuBLAS blocked plane rather than [out, in/16] row-major.
void assemble_blocks_impl(const uint8_t* packed,
                          const uint8_t* scales,
                          uint8_t* dst,
                          int64_t in_features,
                          int64_t out_features,
                          bool src_hi_first,
                          bool src_swizzled) {
    const int64_t nsub         = in_features / NVFP4_SUB;
    const int64_t nblk         = in_features / NVFP4_BLCK;
    const int64_t n_col_blocks = roundup_div(nsub, SWZ_COLS_PER_COL);

    for (int64_t r = 0; r < out_features; ++r) {
        const uint8_t* prow = packed + (size_t)r * (size_t)(in_features / 2);
        const uint8_t* srow = scales + (size_t)r * (size_t)nsub;  // row-major only
        uint8_t* drow       = dst + (size_t)r * (size_t)nblk * NVFP4_BLOCK_SIZE;

        for (int64_t b = 0; b < nblk; ++b) {
            uint8_t* blk = drow + (size_t)b * NVFP4_BLOCK_SIZE;
            uint8_t* qs  = blk + NVFP4_SUBS;
            for (int64_t s = 0; s < NVFP4_SUBS; ++s) {
                const int64_t ss = b * NVFP4_SUBS + s;
                // FP8 E4M3 group scale, verbatim -- only its ADDRESS differs by convention.
                blk[s] = src_swizzled ? scales[swizzled_scale_offset(r, ss, n_col_blocks)] : srow[ss];

                // Source: 8 bytes holding elements 0..15 of this group, as (2j low, 2j+1
                // high) or, when src_hi_first, (2j high, 2j+1 low).  Destination: 8 bytes
                // holding (j low, j+8 high).  Elements j and j+8 always share a parity, so
                // both come from the same nibble half.
                const uint8_t* od = prow + (size_t)ss * (NVFP4_SUB / 2);
                uint8_t* qsub     = qs + s * (NVFP4_SUB / 2);
                for (int j = 0; j < NVFP4_SUB / 2; ++j) {
                    const uint8_t lo_byte = od[j >> 1];
                    const uint8_t hi_byte = od[(j >> 1) + 4];
                    const bool take_high  = ((j & 1) != 0) != src_hi_first;
                    const uint8_t n0      = take_high ? (uint8_t)(lo_byte >> 4) : (uint8_t)(lo_byte & 0x0F);
                    const uint8_t n1      = take_high ? (uint8_t)(hi_byte >> 4) : (uint8_t)(hi_byte & 0x0F);
                    qsub[j]               = (uint8_t)(n0 | (n1 << 4));
                }
            }
        }
    }
}

}  // namespace

void nvfp4_import_assemble_blocks(const uint8_t* packed,
                                  const uint8_t* scales,
                                  uint8_t* dst,
                                  int64_t in_features,
                                  int64_t out_features) {
    assemble_blocks_impl(packed, scales, dst, in_features, out_features, /*src_hi_first=*/false, /*src_swizzled=*/false);
}

void nvfp4_import_assemble_blocks_comfy(const uint8_t* packed,
                                        const uint8_t* scales,
                                        uint8_t* dst,
                                        int64_t in_features,
                                        int64_t out_features) {
    assemble_blocks_impl(packed, scales, dst, in_features, out_features, /*src_hi_first=*/true, /*src_swizzled=*/true);
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
    const size_t scale_bytes   = tensor_storage.nvfp4_comfy_layout
                                     ? comfy_scale_plane_bytes(in_features, out_features)
                                     : (size_t)(in_features / NVFP4_SUB) * (size_t)out_features;
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
    if (tensor_storage.nvfp4_comfy_layout) {
        nvfp4_import_assemble_blocks_comfy(scratch.data(),
                                           scratch.data() + packed_bytes,
                                           out.data(),
                                           in_features,
                                           out_features);
    } else {
        nvfp4_import_assemble_blocks(scratch.data(),
                                     scratch.data() + packed_bytes,
                                     out.data(),
                                     in_features,
                                     out_features);
    }
    return true;
}

namespace {

constexpr const char* COMFY_QUANT_SUFFIX = ".comfy_quant";

// Replace / delete entries in `tensor_storages` according to an import's decisions.
void splice_tensor_storages(std::vector<TensorStorage>& tensor_storages,
                            const std::map<std::string, TensorStorage>& imported,
                            const std::vector<std::string>& drop) {
    std::set<std::string> drop_set(drop.begin(), drop.end());
    std::vector<TensorStorage> result;
    result.reserve(tensor_storages.size() + imported.size());
    for (TensorStorage& ts : tensor_storages) {
        if (drop_set.count(ts.name) != 0) {
            continue;
        }
        if (imported.find(ts.name) != imported.end()) {
            continue;  // replaced below
        }
        result.push_back(std::move(ts));
    }
    for (const auto& [name, ts] : imported) {
        result.push_back(ts);
    }
    tensor_storages = std::move(result);
}

// Materialise the shared block-layout guard once for both branches.
bool ggml_nvfp4_layout_ok() {
    if (ggml_blck_size(GGML_TYPE_NVFP4) != NVFP4_BLCK || ggml_type_size(GGML_TYPE_NVFP4) != NVFP4_BLOCK_SIZE) {
        LOG_ERROR("nvfp4 import: ggml block_nvfp4 layout changed (blck=%d size=%zu); refusing to import",
                  ggml_blck_size(GGML_TYPE_NVFP4),
                  ggml_type_size(GGML_TYPE_NVFP4));
        return false;
    }
    return true;
}

// Import every supported ComfyUI-quantised linear. NVFP4 weights are reblocked and
// receive a `.wglobal` sidecar. Scaled FP8 weights stay on the ordinary safetensors
// path, with their scalar folded while the reader expands FP8 to F16.
bool rewrite_comfy_nvfp4(const std::string& file_path, std::vector<TensorStorage>& tensor_storages) {
    if (!ggml_nvfp4_layout_ok()) {
        return false;
    }

    std::map<std::string, RawEntry> raw;
    if (!read_raw_header(file_path, raw)) {
        LOG_ERROR("comfy_quant import: failed to re-read safetensors header of '%s'", file_path.c_str());
        return false;
    }

    std::vector<std::string> bases;
    for (const auto& [name, entry] : raw) {
        std::string base;
        if (strip_suffix(name, COMFY_QUANT_SUFFIX, &base)) {
            bases.push_back(base);
        }
    }
    if (bases.empty()) {
        return true;
    }

    std::map<std::string, TensorStorage> imported;
    std::vector<std::string> drop;
    size_t nvfp4_count = 0;
    size_t float8_count = 0;

    for (const std::string& base : bases) {
        const std::string packed_name = base + ".weight";
        const std::string scale_name  = base + ".weight_scale";
        const std::string global_name = base + ".weight_scale_2";
        const std::string quant_name  = base + COMFY_QUANT_SUFFIX;

        auto packed_it = raw.find(packed_name);
        auto scale_it  = raw.find(scale_name);
        auto global_it = raw.find(global_name);
        auto quant_it  = raw.find(quant_name);
        if (quant_it == raw.end()) {
            LOG_ERROR("comfy_quant import: missing '%s'", quant_name.c_str());
            return false;
        }
        std::ifstream quant_file(file_path, std::ios::binary);
        std::string quant_blob((size_t)quant_it->second.nbytes, '\0');
        quant_file.seekg((std::streamoff)quant_it->second.offset);
        quant_file.read(&quant_blob[0], (std::streamsize)quant_blob.size());
        nlohmann::json quant_config;
        try {
            quant_config = nlohmann::json::parse(quant_blob);
        } catch (const std::exception&) {
            LOG_ERROR("comfy_quant import: malformed '%s'", quant_name.c_str());
            return false;
        }
        const std::string format = quant_config["format"].get<std::string>();

        if (format == "float8_e4m3fn") {
            if (packed_it == raw.end() || scale_it == raw.end()) {
                LOG_ERROR("comfy_quant import: '%s' is missing one of {%s, %s}",
                          base.c_str(),
                          packed_name.c_str(),
                          scale_name.c_str());
                return false;
            }
            const RawEntry& packed = packed_it->second;
            const RawEntry& scale  = scale_it->second;
            if (packed.dtype != "F8_E4M3" || packed.shape.size() != 2 ||
                scale.dtype != "F32" || scale.nbytes != sizeof(float) || !scale.shape.empty()) {
                LOG_ERROR("comfy_quant import: malformed scaled-FP8 family '%s'", base.c_str());
                return false;
            }
            float tensor_scale = 0.0f;
            if (!read_f32_at(file_path, scale.offset, &tensor_scale) ||
                !(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
                LOG_ERROR("comfy_quant import: '%s' is not a usable positive F32 scale", scale_name.c_str());
                return false;
            }
            auto storage_it = std::find_if(tensor_storages.begin(), tensor_storages.end(),
                                           [&](const TensorStorage& storage) {
                                               return storage.name == packed_name;
                                           });
            if (storage_it == tensor_storages.end() || !storage_it->is_f8_e4m3) {
                LOG_ERROR("comfy_quant import: FP8 weight '%s' was not read as F8_E4M3", packed_name.c_str());
                return false;
            }
            storage_it->f8_scale = tensor_scale;
            drop.push_back(scale_name);
            drop.push_back(quant_name);
            ++float8_count;
            continue;
        }

        if (packed_it == raw.end() || scale_it == raw.end() || global_it == raw.end()) {
            LOG_ERROR("comfy_quant import: '%s' says format nvfp4 but is missing one of {%s, %s, %s}",
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
            LOG_ERROR("comfy_quant import: '%s' is %s rank-%zu, expected U8 rank-2",
                      packed_name.c_str(),
                      packed.dtype.c_str(),
                      packed.shape.size());
            return false;
        }
        if (scale.dtype != "F8_E4M3" || scale.shape.size() != 2) {
            LOG_ERROR("comfy_quant import: '%s' is %s rank-%zu, expected F8_E4M3 rank-2",
                      scale_name.c_str(),
                      scale.dtype.c_str(),
                      scale.shape.size());
            return false;
        }
        if (global.dtype != "F32" || global.nbytes != sizeof(float)) {
            LOG_ERROR("comfy_quant import: '%s' is not an F32 scalar", global_name.c_str());
            return false;
        }

        const int64_t out_features = packed.shape[0];
        const int64_t in_features  = packed.shape[1] * 2;

        if (in_features % NVFP4_BLCK != 0) {
            // comfy-kitchen pads to a multiple of 16 and does NOT record the pre-pad shape,
            // so a 16-but-not-64 row cannot be re-blocked or un-padded.  See the ModelOpt
            // branch for the same reasoning.
            LOG_ERROR("comfy_quant import: '%s' has in_features=%" PRId64 ", not a multiple of %" PRId64,
                      packed_name.c_str(),
                      in_features,
                      NVFP4_BLCK);
            return false;
        }
        if (packed.nbytes != (uint64_t)(in_features / 2) * (uint64_t)out_features) {
            LOG_ERROR("comfy_quant import: byte count does not match the declared shape for '%s'",
                      packed_name.c_str());
            return false;
        }

        // 🔴 The scale plane is the cuBLAS blocked swizzle, so its shape is the PADDED one:
        // [roundup(out, 128), roundup(in/16, 4)].  A ModelOpt export of the same weight
        // would be [out, in/16] row-major and, for the shapes typical of a DiT, the two are
        // the same size -- which is exactly why this must be checked, not inferred.
        const int64_t padded_rows = roundup_div(out_features, SWZ_ROWS_PER_BASE_BLOCK) * SWZ_ROWS_PER_BASE_BLOCK;
        const int64_t padded_cols = roundup_div(in_features / NVFP4_SUB, SWZ_COLS_PER_COL) * SWZ_COLS_PER_COL;
        if (scale.shape[0] != padded_rows || scale.shape[1] != padded_cols) {
            LOG_ERROR("comfy_quant import: '%s' is [%" PRId64 ", %" PRId64 "], expected the blocked-swizzle shape [%" PRId64
                      ", %" PRId64 "] for a [%" PRId64 ", %" PRId64 "] weight",
                      scale_name.c_str(),
                      scale.shape[0],
                      scale.shape[1],
                      padded_rows,
                      padded_cols,
                      out_features,
                      in_features);
            return false;
        }
        if (scale.nbytes != (uint64_t)padded_rows * (uint64_t)padded_cols) {
            LOG_ERROR("comfy_quant import: byte count does not match the declared shape for '%s'", scale_name.c_str());
            return false;
        }

        float wglobal = 0.0f;
        if (!read_f32_at(file_path, global.offset, &wglobal)) {
            LOG_ERROR("comfy_quant import: failed to read '%s'", global_name.c_str());
            return false;
        }
        if (!(wglobal > 0.0f) || !std::isfinite(wglobal)) {
            LOG_ERROR("comfy_quant import: '%s' = %g is not a usable positive scale", global_name.c_str(), wglobal);
            return false;
        }

        const int64_t ne[SD_MAX_DIMS] = {in_features, out_features, 1, 1, 1};
        TensorStorage w(packed_name, GGML_TYPE_NVFP4, ne, 2, 0, packed.offset);
        w.is_nvfp4_import    = true;
        w.nvfp4_comfy_layout = true;
        w.nvfp4_scale_offset = scale.offset;
        imported[w.name]     = w;

        const int64_t ne1[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
        TensorStorage g(base + ".weight.wglobal", GGML_TYPE_F32, ne1, 1, 0, 0);
        g.is_inline_f32  = true;
        g.inline_f32     = wglobal;
        imported[g.name] = g;

        // Consumed above.  weight_scale must not survive: it collides by name with Linear's
        // unrelated per-output-channel F32 `weight_scale` parameter (ggml_extend.hpp).
        // comfy_quant is U8 and the shared safetensors reader already skips it, but drop it
        // explicitly so this does not depend on that.
        drop.push_back(scale_name);
        drop.push_back(global_name);
        drop.push_back(quant_name);
        ++nvfp4_count;
    }

    splice_tensor_storages(tensor_storages, imported, drop);
    LOG_INFO("comfy_quant import: %zu NVFP4 + %zu scaled-FP8 ComfyUI linears in '%s'",
             nvfp4_count,
             float8_count,
             file_path.c_str());
    return true;
}

bool rewrite_comfy_int8_convrot(const std::string& file_path,
                                std::vector<TensorStorage>& tensor_storages) {
    std::map<std::string, RawEntry> raw;
    if (!read_raw_header(file_path, raw)) {
        LOG_ERROR("comfy int8_convrot import: failed to re-read '%s'", file_path.c_str());
        return false;
    }

    std::map<std::string, TensorStorage> imported;
    std::vector<std::string> drop;
    size_t count = 0;
    for (const auto& [name, quant] : raw) {
        std::string base;
        if (!strip_suffix(name, COMFY_QUANT_SUFFIX, &base)) {
            continue;
        }
        const std::string weight_name = base + ".weight";
        const std::string scale_name  = base + ".weight_scale";
        const auto wit = raw.find(weight_name);
        const auto sit = raw.find(scale_name);
        if (wit == raw.end() || sit == raw.end()) {
            LOG_ERROR("comfy int8_convrot import: '%s' is missing weight or weight_scale", base.c_str());
            return false;
        }
        const RawEntry& weight = wit->second;
        const RawEntry& scale  = sit->second;
        if (weight.dtype != "I8" || weight.shape.size() != 2 ||
            weight.shape[1] % 256 != 0 ||
            scale.dtype != "F32" || scale.shape.size() != 2 ||
            scale.shape[0] != weight.shape[0] || scale.shape[1] != 1) {
            LOG_ERROR("comfy int8_convrot import: malformed family '%s' (expected I8[out,in], "
                      "in%%256=0, F32[out,1])", base.c_str());
            return false;
        }
        if (weight.nbytes != static_cast<uint64_t>(weight.shape[0]) *
                                static_cast<uint64_t>(weight.shape[1]) ||
            scale.nbytes != static_cast<uint64_t>(weight.shape[0]) * sizeof(float)) {
            LOG_ERROR("comfy int8_convrot import: byte count mismatch in '%s'", base.c_str());
            return false;
        }

        // Keep both payloads literally.  Only normalize [out,1] to a 1-D ggml
        // broadcast vector; this changes tensor metadata, not bytes.
        auto tsit = std::find_if(tensor_storages.begin(), tensor_storages.end(),
                                 [&](const TensorStorage& ts) { return ts.name == scale_name; });
        if (tsit == tensor_storages.end()) {
            LOG_ERROR("comfy int8_convrot import: shared reader omitted '%s'", scale_name.c_str());
            return false;
        }
        TensorStorage row_scale = *tsit;
        row_scale.n_dims = 1;
        row_scale.ne[0]  = weight.shape[0];
        row_scale.ne[1]  = row_scale.ne[2] = row_scale.ne[3] = row_scale.ne[4] = 1;
        imported[scale_name] = row_scale;
        drop.push_back(name);
        ++count;
    }
    splice_tensor_storages(tensor_storages, imported, drop);
    LOG_INFO("comfy int8_convrot import: %zu direct I8 linears in '%s'", count, file_path.c_str());
    return count != 0;
}

}  // namespace

ComfyQuantScan comfy_quant_scan_safetensors(const std::string& file_path, std::string* detail) {
    const auto fail = [detail](ComfyQuantScan r, const std::string& msg) {
        if (detail != nullptr) {
            *detail = msg;
        }
        return r;
    };

    std::map<std::string, RawEntry> raw;
    if (!read_raw_header(file_path, raw)) {
        // Not necessarily a comfy file at all -- let the ordinary reader report on it.
        return ComfyQuantScan::None;
    }

    std::vector<std::pair<std::string, RawEntry>> blobs;
    for (const auto& [name, entry] : raw) {
        std::string base;
        if (strip_suffix(name, COMFY_QUANT_SUFFIX, &base)) {
            blobs.emplace_back(name, entry);
        }
    }
    if (blobs.empty()) {
        return ComfyQuantScan::None;
    }

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return fail(ComfyQuantScan::Malformed, "comfy_quant: failed to open '" + file_path + "'");
    }

    size_t nvfp4_count = 0;
    size_t float8_count = 0;
    size_t int8_convrot_count = 0;
    for (const auto& [name, entry] : blobs) {
        if (entry.dtype != "U8" || entry.nbytes == 0 || entry.nbytes > 4096) {
            return fail(ComfyQuantScan::Malformed,
                        "comfy_quant: '" + name + "' is " + entry.dtype + " of " + std::to_string(entry.nbytes) +
                            " bytes; expected a small U8 JSON blob");
        }
        std::string blob((size_t)entry.nbytes, '\0');
        file.seekg((std::streamoff)entry.offset);
        file.read(&blob[0], (std::streamsize)entry.nbytes);
        if (!file) {
            return fail(ComfyQuantScan::Malformed, "comfy_quant: failed to read '" + name + "'");
        }

        nlohmann::json cfg;
        try {
            cfg = nlohmann::json::parse(blob);
        } catch (const std::exception&) {
            return fail(ComfyQuantScan::Malformed, "comfy_quant: '" + name + "' is not JSON: " + blob);
        }
        if (!cfg.is_object() || !cfg.contains("format") || !cfg["format"].is_string()) {
            return fail(ComfyQuantScan::Malformed, "comfy_quant: '" + name + "' has no string `format`: " + blob);
        }

        const std::string format = cfg["format"].get<std::string>();
        const bool int8_convrot =
            format == "int8_tensorwise" &&
            cfg.value("convrot", false) &&
            cfg.value("convrot_groupsize", 0) == 256;
        if (format != "nvfp4" && format != "float8_e4m3fn" && !int8_convrot) {
            return fail(ComfyQuantScan::Unsupported,
                        "comfy_quant: '" + name + "' declares format '" + format +
                            "', which this engine cannot load: " + blob +
                            "  (supported: nvfp4, float8_e4m3fn, and exact "
                            "int8_tensorwise+convrot_groupsize=256)");
        }
        // An unrecognised MODIFIER on an nvfp4 layer (e.g. a rotation, an activation
        // pre-scale) would change the maths while leaving every shape and dtype intact.
        for (const auto& item : cfg.items()) {
            if (item.key() != "format" &&
                !(int8_convrot && (item.key() == "convrot" || item.key() == "convrot_groupsize"))) {
                return fail(ComfyQuantScan::Unsupported,
                            "comfy_quant: '" + name + "' carries the unrecognised key '" + item.key() +
                                "' on an nvfp4 layer, which would silently change the dequantisation: " + blob);
            }
        }
        if (format == "nvfp4") {
            ++nvfp4_count;
        } else if (format == "float8_e4m3fn") {
            ++float8_count;
        } else {
            ++int8_convrot_count;
        }
    }

    if (int8_convrot_count != 0) {
        if (nvfp4_count != 0 || float8_count != 0 || int8_convrot_count != blobs.size()) {
            return fail(ComfyQuantScan::Unsupported,
                        "comfy_quant: mixed int8_convrot and other quant formats are not supported");
        }
        return ComfyQuantScan::Int8Convrot;
    }
    if (nvfp4_count != 0 && float8_count != 0) {
        return ComfyQuantScan::Nvfp4Float8;
    }
    return nvfp4_count != 0 ? ComfyQuantScan::Nvfp4 : ComfyQuantScan::Float8;
}

bool nvfp4_import_rewrite_safetensors(const std::string& file_path,
                                      std::vector<TensorStorage>& tensor_storages,
                                      ComfyQuantScan comfy) {
    // 🔴 A ComfyUI nvfp4 export uses ModelOpt's EXACT tensor names with a different byte
    // order (see the header).  Dispatch on comfy_quant first, or the branch below matches
    // it on names and silently imports a wrong model.
    if (comfy == ComfyQuantScan::Nvfp4 ||
        comfy == ComfyQuantScan::Float8 ||
        comfy == ComfyQuantScan::Nvfp4Float8) {
        return rewrite_comfy_nvfp4(file_path, tensor_storages);
    }
    if (comfy == ComfyQuantScan::Int8Convrot) {
        return rewrite_comfy_int8_convrot(file_path, tensor_storages);
    }
    if (comfy != ComfyQuantScan::None) {
        LOG_ERROR("nvfp4 import: refusing '%s' -- its comfy_quant blobs were not accepted", file_path.c_str());
        return false;
    }

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

    splice_tensor_storages(tensor_storages, imported, drop);

    LOG_INFO("nvfp4 import: %zu externally-quantised NVFP4 linears in '%s'", linears.size(), file_path.c_str());
    return true;
}
