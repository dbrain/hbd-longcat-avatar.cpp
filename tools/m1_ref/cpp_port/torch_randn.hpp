// Self-contained port of torch's CPU randn (the cross-stage seed-42 noise for the E2E chain).
//
// The Pixal3D pipeline seeds ONCE (torch.manual_seed(42)) then draws, in order, on the CPU
// global generator: stage1 randn(1,8,16,16,16); stage2 randn(N1,32); stage3b randn(M,32) —
// where N1,M are the COMPUTED coord counts, so the noise must be generated in-stream after
// each count is known. This reproduces torch's exact CPU RNG so the C++ chain matches the
// python pipeline's seed-42 output (no torch dependency in the driver).
//
// torch 2.6 CPU randn(FLOAT32, size>=16, contiguous) path (DistributionTemplates.h
// normal_kernel -> normal_fill / normal_fill_AVX2):
//   1. fill data[0..size) with uniform<float>: data[i] = (random()&0xFFFFFF) * 2^-24   (one uint32 each)
//   2. for each 16-block: normal_fill_16:  for j in 0..8:
//        u1 = 1-data[j]; u2 = data[j+8]; radius=sqrt(-2*log(u1)); theta=2*pi*u2 (pi as double);
//        data[j]=radius*cos(theta);  data[j+8]=radius*sin(theta)
//   3. if size%16: redraw last 16 uniforms (overlap) + normal_fill_16 them (+16 uint32 draws)
// MT19937 = at::mt19937_engine (NOT std::mt19937). Validated bit-exact (randn_test.cpp) vs
// refs/{noise_seed42,stage2_noise,stage3b_noise}.npy (one continuous seed-42 stream).
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>

namespace trandn {

// ---- at::mt19937_engine (torch's MT19937, NOT std::mt19937) ----
struct MT19937 {
    static constexpr int N = 624, M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu, UMASK = 0x80000000u, LMASK = 0x7fffffffu;
    uint32_t state_[N];
    int left_;
    int next_;

    explicit MT19937(uint64_t seed = 5489) { do_seed(seed); }

    void do_seed(uint64_t seed) {
        state_[0] = (uint32_t)(seed & 0xffffffffu);
        for (int i = 1; i < N; i++)
            state_[i] = (uint32_t)(1812433253u * (state_[i-1] ^ (state_[i-1] >> 30)) + (uint32_t)i);
        left_ = 1;
        next_ = 0;
    }

    static inline uint32_t mix_bits(uint32_t u, uint32_t v) { return (u & UMASK) | (v & LMASK); }
    static inline uint32_t twist(uint32_t u, uint32_t v) { return (mix_bits(u, v) >> 1) ^ ((v & 1u) ? MATRIX_A : 0u); }

    void next_state() {
        left_ = N;
        next_ = 0;
        int pi = 0;
        for (int j = N - M + 1; --j; pi++) state_[pi] = state_[pi + M] ^ twist(state_[pi], state_[pi + 1]);
        for (int j = M; --j; pi++)         state_[pi] = state_[pi + (M - N)] ^ twist(state_[pi], state_[pi + 1]);
        state_[pi] = state_[pi + (M - N)] ^ twist(state_[pi], state_[0]);
    }

    inline uint32_t operator()() {
        if ((--left_) <= 0) next_state();
        uint32_t y = state_[next_++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }
};

// uniform_real_distribution<float>: (random() & ((1<<24)-1)) * 2^-24  in [0,1).
static inline float uniform_float(MT19937& eng) {
    uint32_t v = eng();
    constexpr uint32_t MASK = (1u << 24) - 1u;
    constexpr float DIVISOR = 1.0f / (float)(1u << 24);
    return (float)(v & MASK) * DIVISOR;
}

// normal_fill_16 (scalar, == torch's): in-place Box-Muller over a 16-float block.
static inline void normal_fill_16(float* data) {
    for (int j = 0; j < 8; j++) {
        float u1 = 1.0f - data[j];       // [0,1)->(0,1]
        float u2 = data[j + 8];
        float radius = std::sqrt(-2.0f * std::log(u1));
        float theta = (float)(2.0 * M_PI * (double)u2);   // pi as double (matches torch)
        data[j]     = radius * std::cos(theta);
        data[j + 8] = radius * std::sin(theta);
    }
}

// Generator holding the continuous MT19937 stream. randn(size) reproduces torch's float path.
struct Generator {
    MT19937 eng;
    explicit Generator(uint64_t seed) : eng(seed) {}

    void fill_randn(float* data, int64_t size) {
        // size>=16 contiguous float path. (size<16 would use the double per-element path —
        // never hit by our noise tensors; assert away from it.)
        for (int64_t i = 0; i < size; i++) data[i] = uniform_float(eng);
        int64_t i = 0;
        for (; i < size - 15; i += 16) normal_fill_16(data + i);
        if (size % 16 != 0) {
            float* tail = data + size - 16;
            for (int k = 0; k < 16; k++) tail[k] = uniform_float(eng);
            normal_fill_16(tail);
        }
    }
    std::vector<float> randn(int64_t size) {
        std::vector<float> v((size_t)size);
        fill_randn(v.data(), size);
        return v;
    }
};

}  // namespace trandn
