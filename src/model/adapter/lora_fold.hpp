#ifndef __SD_MODEL_ADAPTER_LORA_FOLD_HPP__
#define __SD_MODEL_ADAPTER_LORA_FOLD_HPP__

// Fold a LoRA delta straight into the PARAMS-BACKEND copy of a weight, once, so the
// per-step adapter branch disappears entirely.
//
// WHY THIS IS NOT THE `apply_loras_immediately` PATH IN lora.hpp
// -------------------------------------------------------------
// That path builds a ggml graph (`cast_f32 -> add -> cpy back`) on the COMPUTE backend.
// Two independent reasons it cannot work here:
//   1. it runs AFTER staging, and a compute staging block is freed and rebuilt every
//      graph, which resets `applied_lora_epoch` -- so under offload the merge would be
//      redone continuously instead of once;
//   2. CUDA has no `GGML_OP_CPY` with an NVFP4 destination at all (see the CPY case in
//      ggml/src/ggml-cuda/ggml-cuda.cu::supports_op), and `ggml_ext_cast_f32` dequantises
//      an NVFP4 weight WITHOUT its `.wglobal`, so the magnitudes would be wrong even if
//      the copy existed.
// This runs on the host copy, before staging, and writes NVFP4 block bytes directly.
//
// WHY THE ROUNDING IS STOCHASTIC -- THIS IS THE WHOLE TRICK
// ---------------------------------------------------------
// MEASURED on nvfp4-CLEAN.gguf + ltx2.3-audio-reactive-v2 @1.5, against the bf16 base:
// the base's own nvfp4 error is 0.094 relative, while the LoRA delta is only 0.015-0.047
// relative. The delta is 2-6x SMALLER than the quantisation noise it has to survive, so
// round-to-nearest lands almost none of it -- it pulls straight back to the code the
// weight already sat on. Projection of the applied delta onto the intended one:
//
//     frozen grid + round-to-nearest (route A)     0.062
//     re-derived grid + round-to-nearest           0.078   <- re-deriving scales buys ~0
//     re-derived grid + STOCHASTIC                 0.948
//     frozen grid + STOCHASTIC                     0.944
//     route B (merge in bf16, quantise once)       0.995   <- offline only; needs bf16
//
// So the grid barely matters and the ROUNDING RULE is everything. Keeping the grid frozen
// means `.wglobal` never changes, so the process-global NVFP4 weight-global registry and
// the cuBLASLt alpha fold stay valid and untouched.
//
// The cost is honest and is NOT zero: weight error goes 0.094 -> 0.106, i.e. ~13% more
// incoherent noise than a route-B pre-folded gguf. Whether that is visible in a render is
// NOT settled here.
//
// Rounding is stochastic but DETERMINISTIC: the random stream is a counter-based hash of
// (tensor name, element index), so two runs of the same fold produce identical bytes and
// renders stay bit-reproducible.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"

namespace sd_lora_fold {

// One LoRA's contribution to one weight, already dequantised to f32 on the host.
// down is [in, rank] and up is [rank, rows] in ggml (ne[0]-fastest) layout, matching how
// LoraModel::get_out_diff feeds them to ggml_ext_linear.
//
// row_begin/rows exist because a LoRA may target a PACKED projection in segments: the
// runtime path finds `lora.<name>.lora_up`, `...lora_up.1`, `...lora_up.2` and concatenates
// their outputs along dim 0 (see get_out_diff's ggml_concat). Each such module therefore
// owns a consecutive slice of the output features, not the whole weight.
struct ModuleDelta {
    const float* down = nullptr;
    const float* up   = nullptr;
    const float* mid  = nullptr;  // optional [rank, rank], chained between down and up
    int64_t in        = 0;
    int64_t row_begin = 0;
    int64_t rows      = 0;
    int64_t rank      = 0;
    float scale       = 1.0f;  // alpha/rank (or an explicit .scale) times the multiplier
};

// --- NVFP4 block format, mirrored from ggml/src/ggml-common.h so this header does not
// --- have to reach into ggml's private headers. 64 values per block, one UE4M3 scale per
// --- 16-value sub-block, E2M1 nibbles packed j / j+8 into one byte.
static constexpr int kNvfp4Block    = 64;
static constexpr int kNvfp4Sub      = 16;
static constexpr int kNvfp4BlockSz  = 36;  // 4 scale bytes + 32 nibble bytes
// kvalues_fp4, i.e. TWICE the E2M1 magnitudes, paired with a ue4m3 decode that returns
// HALF the true scale -- the product is the true value. Keeping ggml's convention exactly
// is what makes these bytes byte-compatible with dequantize_row_nvfp4.
static constexpr float kE2m1Mag[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};

inline float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const float raw = (exp == 0) ? std::ldexp((float)man, -9)
                                 : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

// 256-entry decode table. The scalar version calls ldexp, which is a libm call and is not
// inlined, and it runs once per 16-value sub-block -- a million times per 4096x4096 tensor.
struct Ue4m3Table {
    float v[256];
    Ue4m3Table() {
        for (int i = 0; i < 256; ++i) {
            v[i] = ue4m3_to_fp32((uint8_t)i);
        }
    }
};
inline const Ue4m3Table& ue4m3_table() {
    static const Ue4m3Table t;
    return t;
}

// splitmix64 over (seed, index): a counter-based stream, so the fold is order-independent
// and reproducible without carrying RNG state across threads.
inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline float uniform01(uint64_t seed, uint64_t index) {
    // 24 bits is ample: it only has to resolve a rounding probability.
    return (float)(mix64(seed ^ mix64(index)) >> 40) * (1.0f / 16777216.0f);
}

inline uint64_t seed_of(const std::string& name) {
    uint64_t h = 1469598103934665603ull;  // FNV-1a
    for (unsigned char c : name) {
        h = (h ^ c) * 1099511628211ull;
    }
    return h;
}

// Which E2M1 code brackets a magnitude from below, indexed by (int)mag for mag in [0, 12).
// Replaces a 7-iteration search whose branches are unpredictable; that search was the
// single most expensive thing in the requantise loop.
static constexpr uint8_t kE2m1LowerByInt[12] = {0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6};
// 1 / (kE2m1Mag[lo+1] - kE2m1Mag[lo]), indexed by lo. lo == 7 is the saturated case.
static constexpr float kE2m1InvGap[8] = {1.f, 1.f, 1.f, 1.f, 0.5f, 0.5f, 0.25f, 0.f};

// Round |v| (expressed in E2M1 magnitude units) to a code 0..7, stochastically between the
// two bracketing table entries so the result is unbiased in expectation.
inline uint8_t e2m1_code_stochastic(float mag, float u) {
    if (!(mag > 0.0f)) {
        return 0;
    }
    if (mag >= 12.0f) {
        return 7;  // saturated: nothing above to round up to
    }
    const int lo  = kE2m1LowerByInt[(int)mag];
    const float p = (mag - kE2m1Mag[lo]) * kE2m1InvGap[lo];
    return (uint8_t)(lo + (u < p ? 1 : 0));
}

// How many output rows one worker computes at a time. The point is `lora_down`, which is
// rank*in floats (1 MB at rank 64, in 4096): computing a SINGLE row at a time re-streams
// the whole of it per row, which for a 4096-row tensor is ~4 GB of L3 traffic and made the
// fold L3-bound (measured 81 ms/tensor => ~90 s for a 1344-module adapter). Blocking the
// rows drops that by the block factor, at the cost of a `kRowBlock * in` f32 accumulator
// (256 KB at 16x4096) that has to stay in L2.
static constexpr int64_t kRowBlock = 16;

// Accumulate sum_l scale_l * (up_l @ down_l) for output rows [o0, o0+nrows) into
// `acc` [nrows][in]. Innermost loop is a contiguous axpy over `in`, which is what lets the
// compiler vectorise it.
inline void accumulate_rows(float* acc,
                            int64_t in,
                            const std::vector<ModuleDelta>& deltas,
                            int64_t o0,
                            int64_t nrows) {
    std::memset(acc, 0, sizeof(float) * (size_t)in * (size_t)nrows);
    std::vector<float> coeff_block;
    for (const ModuleDelta& d : deltas) {
        const int64_t begin = std::max(o0, d.row_begin);
        const int64_t end   = std::min(o0 + nrows, d.row_begin + d.rows);
        if (begin >= end) {
            continue;  // this module owns a different slice of the packed output
        }
        // Effective rank-wide coefficients per row: up, or up @ mid when a lora_mid exists.
        const float* coeff  = nullptr;
        int64_t coeff_pitch = d.rank;
        if (d.mid == nullptr) {
            coeff = d.up + (size_t)(begin - d.row_begin) * (size_t)d.rank;
        } else {
            coeff_block.assign((size_t)(end - begin) * (size_t)d.rank, 0.0f);
            for (int64_t r = begin; r < end; ++r) {
                const float* up_row = d.up + (size_t)(r - d.row_begin) * (size_t)d.rank;
                float* dst          = coeff_block.data() + (size_t)(r - begin) * (size_t)d.rank;
                for (int64_t k = 0; k < d.rank; ++k) {
                    const float c = up_row[k];
                    if (c == 0.0f) {
                        continue;
                    }
                    const float* mrow = d.mid + (size_t)k * (size_t)d.rank;
                    for (int64_t j = 0; j < d.rank; ++j) {
                        dst[j] += c * mrow[j];
                    }
                }
            }
            coeff = coeff_block.data();
        }
        for (int64_t k = 0; k < d.rank; ++k) {
            const float* drow = d.down + (size_t)k * (size_t)in;
            for (int64_t r = begin; r < end; ++r) {
                const float c = d.scale * coeff[(size_t)(r - begin) * (size_t)coeff_pitch + k];
                if (c == 0.0f) {
                    continue;
                }
                float* arow = acc + (size_t)(r - o0) * (size_t)in;
                for (int64_t i = 0; i < in; ++i) {
                    arow[i] += c * drow[i];
                }
            }
        }
    }
}

// Fold into one NVFP4 row (nb blocks of 64), keeping every UE4M3 scale byte as stored.
// `delta` is in TRUE units; `inv_wglobal` converts it into the block domain the nibbles
// live in, because the runtime multiplies this weight's output by `.wglobal`.
inline void fold_row_nvfp4(uint8_t* row, int64_t nb, const float* delta, float inv_wglobal, uint64_t seed, int64_t o) {
    const float* ue4m3 = ue4m3_table().v;
    const uint64_t row_seed = seed ^ mix64((uint64_t)o);
    for (int64_t b = 0; b < nb; ++b) {
        uint8_t* blk       = row + b * kNvfp4BlockSz;
        const uint8_t* sc  = blk;
        uint8_t* qs        = blk + 4;
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3[sc[s]];
            if (!(d > 0.0f)) {
                continue;  // a zero scale block stays zero; nothing is representable in it
            }
            const float inv_d  = 1.0f / d;
            const int64_t base = b * kNvfp4Block + s * kNvfp4Sub;
            for (int j = 0; j < 8; ++j) {
                const uint8_t packed = qs[s * 8 + j];
                // One hash serves both nibbles of this byte: 24 bits each out of a 64-bit
                // mix, which is far more resolution than a rounding probability needs and
                // halves the RNG cost, the second-biggest item in this loop.
                const uint64_t h = mix64(row_seed ^ (uint64_t)(base + j));
                uint8_t out_lo = 0, out_hi = 0;
                for (int half = 0; half < 2; ++half) {
                    const uint8_t code = (packed >> (half * 4)) & 0x0F;
                    const int64_t idx  = base + j + half * 8;
                    // current value, block domain
                    const float cur = kE2m1Mag[code & 0x7] * d * ((code & 0x8) ? -1.0f : 1.0f);
                    const float tgt = cur + delta[idx] * inv_wglobal;
                    const float mag = std::fabs(tgt) * inv_d;
                    const uint32_t bits = (uint32_t)((half == 0 ? h : (h >> 32)) & 0xFFFFFFu);
                    const float u       = (float)bits * (1.0f / 16777216.0f);
                    uint8_t nc          = e2m1_code_stochastic(mag, u);
                    if (nc != 0 && tgt < 0.0f) {
                        nc |= 0x8;
                    }
                    if (half == 0) {
                        out_lo = nc;
                    } else {
                        out_hi = nc;
                    }
                }
                qs[s * 8 + j] = (uint8_t)(out_lo | (out_hi << 4));
            }
        }
    }
}

// Returns false if `w`'s type is one this fold does not handle, leaving it untouched.
inline bool fold_into_tensor(ggml_tensor* w,
                             void* host_data,
                             float wglobal,
                             const std::vector<ModuleDelta>& deltas,
                             const std::string& name,
                             int n_threads) {
    if (w == nullptr || host_data == nullptr || deltas.empty()) {
        return false;
    }
    if (ggml_n_dims(w) != 2) {
        return false;
    }
    const int64_t in  = w->ne[0];
    const int64_t out = w->ne[1];
    for (const ModuleDelta& d : deltas) {
        if (d.in != in || d.rows <= 0 || d.row_begin < 0 || d.row_begin + d.rows > out) {
            return false;
        }
    }
    if (w->type == GGML_TYPE_NVFP4 && in % kNvfp4Block != 0) {
        return false;
    }
    if (w->type != GGML_TYPE_NVFP4 && w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 &&
        w->type != GGML_TYPE_BF16) {
        return false;
    }

    const uint64_t seed  = seed_of(name);
    const float inv_wg   = (wglobal != 0.0f) ? 1.0f / wglobal : 1.0f;
    const int64_t nb     = in / kNvfp4Block;
    const int nthreads   = std::max(1, n_threads);

    auto worker = [&](int64_t row_begin, int64_t row_end) {
        std::vector<float> acc((size_t)in * (size_t)kRowBlock);
        for (int64_t o0 = row_begin; o0 < row_end; o0 += kRowBlock) {
            const int64_t nrows = std::min(kRowBlock, row_end - o0);
            accumulate_rows(acc.data(), in, deltas, o0, nrows);
            for (int64_t r = 0; r < nrows; ++r) {
                const int64_t o  = o0 + r;
                const float* row = acc.data() + (size_t)r * (size_t)in;
                switch (w->type) {
                    case GGML_TYPE_NVFP4:
                        fold_row_nvfp4((uint8_t*)host_data + (size_t)o * (size_t)nb * kNvfp4BlockSz,
                                       nb, row, inv_wg, seed, o);
                        break;
                    case GGML_TYPE_F32: {
                        float* dst = (float*)host_data + (size_t)o * (size_t)in;
                        for (int64_t i = 0; i < in; ++i) {
                            dst[i] += row[i];
                        }
                        break;
                    }
                    case GGML_TYPE_F16: {
                        ggml_fp16_t* dst = (ggml_fp16_t*)host_data + (size_t)o * (size_t)in;
                        for (int64_t i = 0; i < in; ++i) {
                            dst[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(dst[i]) + row[i]);
                        }
                        break;
                    }
                    case GGML_TYPE_BF16: {
                        ggml_bf16_t* dst = (ggml_bf16_t*)host_data + (size_t)o * (size_t)in;
                        for (int64_t i = 0; i < in; ++i) {
                            dst[i] = ggml_fp32_to_bf16(ggml_bf16_to_fp32(dst[i]) + row[i]);
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }
    };

    if (nthreads == 1 || out < nthreads) {
        worker(0, out);
        return true;
    }
    std::vector<std::thread> pool;
    pool.reserve((size_t)nthreads);
    const int64_t chunk = ((out + nthreads - 1) / nthreads + kRowBlock - 1) / kRowBlock * kRowBlock;
    for (int t = 0; t < nthreads; ++t) {
        const int64_t begin = (int64_t)t * chunk;
        const int64_t end   = std::min(out, begin + chunk);
        if (begin >= end) {
            break;
        }
        pool.emplace_back(worker, begin, end);
    }
    for (auto& th : pool) {
        th.join();
    }
    return true;
}

}  // namespace sd_lora_fold

#endif  // __SD_MODEL_ADAPTER_LORA_FOLD_HPP__
