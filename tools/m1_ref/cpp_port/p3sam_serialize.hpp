// p3sam_serialize.hpp — native C++ port of sonata Point.serialization (structure.py +
// serialization/{z_order,hilbert}.py). Produces, for each of the 4 orders
// ("z","z-trans","hilbert","hilbert-trans"), the serialized_order (argsort of the
// space-filling-curve code) and serialized_inverse, EXACTLY matching the torch path
// (shuffle_orders=False, single batch). Validated bit-exact vs the ser_order_s0 golden.
//
// depth = bit_length(grid_coord.max()+1)  (structure.py:85). batch==0 so the batch<<48
// term vanishes. Codes are unique (grid_coord unique per voxel, both curves bijective),
// so argsort is unambiguous; we use a stable sort for determinism.
#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

namespace p3sam {

inline int bit_length(uint64_t n) { int b = 0; while (n) { n >>= 1; b++; } return b; }

// z-order (Morton) key: bit i of x->3i+2, y->3i+1, z->3i+0  (z_order.py xyz2key)
inline uint64_t z_encode(uint32_t x, uint32_t y, uint32_t z, int depth) {
    uint64_t key = 0;
    for (int i = 0; i < depth; i++) {
        key |= (uint64_t)((x >> i) & 1u) << (3 * i + 2);
        key |= (uint64_t)((y >> i) & 1u) << (3 * i + 1);
        key |= (uint64_t)((z >> i) & 1u) << (3 * i + 0);
    }
    return key;
}

// Hilbert key (Skilling), faithful port of hilbert.py encode(num_dims=3, num_bits=depth).
// gray[d][b]: bit b (MSB-first) of coord d. Transform in place, flatten interleaved
// S[bit*3+dim], prefix-xor (gray2binary), pack MSB-first -> code.
inline uint64_t hilbert_encode(uint32_t cx, uint32_t cy, uint32_t cz, int num_bits) {
    const int D = 3;
    uint32_t c[3] = {cx, cy, cz};
    // gray[d][b], b=0..num_bits-1, index 0 = MSB
    std::vector<std::array<uint8_t, 64>> g(D);
    for (int d = 0; d < D; d++)
        for (int b = 0; b < num_bits; b++)
            g[d][b] = (uint8_t)((c[d] >> (num_bits - 1 - b)) & 1u);
    for (int bit = 0; bit < num_bits; bit++) {
        for (int dim = 0; dim < D; dim++) {
            uint8_t mask = g[dim][bit];
            if (mask)
                for (int j = bit + 1; j < num_bits; j++) g[0][j] ^= 1;
            for (int j = bit + 1; j < num_bits; j++) {
                uint8_t tf = (uint8_t)((!mask) & (g[0][j] ^ g[dim][j]));
                g[dim][j] ^= tf; g[0][j] ^= tf;
            }
        }
    }
    // flatten interleaved S[bit*D+dim], then prefix-xor (gray2binary), pack MSB-first
    const int L = num_bits * D;
    uint64_t code = 0, acc = 0;
    for (int bit = 0; bit < num_bits; bit++)
        for (int dim = 0; dim < D; dim++) {
            int i = bit * D + dim;
            acc ^= g[dim][bit];                    // prefix xor (MSB-first)
            if (acc) code |= (uint64_t)1 << (L - 1 - i);
        }
    return code;
}

// order_id: 0=z 1=z-trans 2=hilbert 3=hilbert-trans
inline std::vector<int64_t> serialize_order(const int32_t* grid, int N, int order_id,
                                            std::vector<int64_t>* inverse_out = nullptr) {
    int32_t mx = 0;
    for (int i = 0; i < N * 3; i++) mx = std::max(mx, grid[i]);
    int depth = bit_length((uint64_t)mx + 1);
    std::vector<uint64_t> code((size_t)N);
    for (int n = 0; n < N; n++) {
        uint32_t x = grid[n * 3 + 0], y = grid[n * 3 + 1], z = grid[n * 3 + 2];
        switch (order_id) {
            case 0: code[n] = z_encode(x, y, z, depth); break;          // z
            case 1: code[n] = z_encode(y, x, z, depth); break;          // z-trans  [1,0,2]
            case 2: code[n] = hilbert_encode(x, y, z, depth); break;    // hilbert
            case 3: code[n] = hilbert_encode(y, x, z, depth); break;    // hilbert-trans
        }
    }
    std::vector<int64_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) { return code[a] < code[b]; });
    if (inverse_out) {
        inverse_out->assign(N, 0);
        for (int64_t i = 0; i < N; i++) (*inverse_out)[order[i]] = i;
    }
    return order;
}

} // namespace p3sam
