// Standalone check of the MiniMax-H3 q/k head-channel permutation (ggml-free, no model needed).
//
//   g++ -std=c++17 -I src -O2 -Wall -Wextra -Werror -o /tmp/h3_qk tests/minimax_h3_qk_permute_test.cpp
//   /tmp/h3_qk
//
// The rope identity itself is proven numerically in tools/ref_qk_permute.py; this covers the index
// arithmetic and the byte-level rewrite the GGUF converter performs, which that script does not see.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "model/diffusion/minimax_h3_qk_permute.hpp"

using MiniMaxH3::build_qk_head_permutation;
using MiniMaxH3::permute_head_rows;
using MiniMaxH3::qk_permutation_is_applicable;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        failures++;
        printf("  FAIL  %s\n", what);
    }
}

// The mapping the DiT author specified, spelled out literally for the shipping geometry.
static std::vector<int64_t> expected_128_96() {
    std::vector<int64_t> p(128);
    for (int64_t i = 0; i < 48; i++) {
        p[i] = i;  // new[0..47] = old[0..47]
    }
    for (int64_t i = 0; i < 16; i++) {
        p[48 + i] = 96 + i;  // new[48..63] = old[96..111]
    }
    for (int64_t i = 0; i < 48; i++) {
        p[64 + i] = 48 + i;  // new[64..111] = old[48..95]
    }
    for (int64_t i = 0; i < 16; i++) {
        p[112 + i] = 112 + i;  // new[112..127] = old[112..127]
    }
    return p;
}

static void test_shipping_permutation() {
    const std::vector<int64_t> got = build_qk_head_permutation(128, 96);
    check(got == expected_128_96(), "head_dim=128 rot_dim=96 matches the specified mapping");
}

static void test_bijection_and_pairing() {
    const int64_t head_dims[] = {64, 128, 192, 256};
    const int64_t freq_lens[] = {8, 16, 24, 32};
    for (int64_t head_dim : head_dims) {
        for (int64_t freq_len : freq_lens) {
            const int64_t rot_dim = 2 * 3 * freq_len;
            if (!qk_permutation_is_applicable(head_dim, rot_dim)) {
                continue;
            }
            const std::vector<int64_t> perm = build_qk_head_permutation(head_dim, rot_dim);
            check(static_cast<int64_t>(perm.size()) == head_dim, "permutation has head_dim entries");

            std::vector<int> seen(static_cast<size_t>(head_dim), 0);
            for (int64_t v : perm) {
                check(v >= 0 && v < head_dim, "permutation entry in range");
                seen[static_cast<size_t>(v)]++;
            }
            for (int c : seen) {
                check(c == 1, "permutation is a bijection");
            }

            // The load-bearing property: full-width split-half over the PERMUTED head must pair the
            // same original channels H3 pairs, and must pair each pass-through channel with another
            // pass-through channel (so an identity rotation leaves both alone).
            const int64_t half = head_dim / 2;
            for (int64_t j = 0; j < half; j++) {
                const int64_t lo = perm[static_cast<size_t>(j)];
                const int64_t hi = perm[static_cast<size_t>(j + half)];
                if (j < rot_dim / 2) {
                    check(lo == j && hi == j + rot_dim / 2, "rotated pair (c, c + rot_dim/2) preserved");
                } else {
                    check(lo >= rot_dim && hi >= rot_dim, "tail pair holds only pass-through channels");
                }
            }
        }
    }
    check(build_qk_head_permutation(128, 130).empty(), "rot_dim > head_dim is rejected");
    check(build_qk_head_permutation(0, 0).empty(), "degenerate geometry is rejected");

    // rot_dim == head_dim degenerates to the identity, which is what the converter relies on to
    // decide there is nothing to rewrite.
    const std::vector<int64_t> ident = build_qk_head_permutation(96, 96);
    bool is_identity                 = static_cast<int64_t>(ident.size()) == 96;
    for (size_t i = 0; is_identity && i < ident.size(); i++) {
        is_identity = ident[i] == static_cast<int64_t>(i);
    }
    check(is_identity, "rot_dim == head_dim is the identity permutation");
}

// Rebuild the converter's rewrite of a fused [in, 3*inner] weight and confirm it moves whole rows,
// leaves v alone, and is reversible.
static void test_qkv_row_rewrite() {
    const int64_t head_dim = 128;
    const int64_t rot_dim  = 96;
    const int64_t heads    = 3;
    const int64_t inner    = heads * head_dim;
    const size_t row_bytes = 7;  // deliberately not a power of two, and not an element size

    const std::vector<int64_t> perm = build_qk_head_permutation(head_dim, rot_dim);

    std::vector<uint8_t> data(static_cast<size_t>(3 * inner) * row_bytes);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<uint8_t>((i * 31 + 7) & 0xff);
    }
    const std::vector<uint8_t> original = data;

    std::vector<uint8_t> scratch;
    for (int64_t third = 0; third < 2; third++) {
        permute_head_rows(data.data() + static_cast<size_t>(third * inner) * row_bytes,
                          row_bytes,
                          heads,
                          perm,
                          scratch);
    }

    auto row = [&](const std::vector<uint8_t>& buf, int64_t r) { return buf.data() + static_cast<size_t>(r) * row_bytes; };

    bool rows_ok = true;
    for (int64_t third = 0; third < 3; third++) {
        for (int64_t h = 0; h < heads; h++) {
            for (int64_t c = 0; c < head_dim; c++) {
                const int64_t dst = third * inner + h * head_dim + c;
                const int64_t src = third < 2 ? third * inner + h * head_dim + perm[static_cast<size_t>(c)] : dst;
                for (size_t b = 0; b < row_bytes; b++) {
                    rows_ok = rows_ok && row(data, dst)[b] == row(original, src)[b];
                }
            }
        }
    }
    check(rows_ok, "qkv rewrite permutes q/k rows per head and leaves v untouched");

    // Applying the inverse permutation must restore the original bytes exactly.
    std::vector<int64_t> inverse(static_cast<size_t>(head_dim));
    for (int64_t i = 0; i < head_dim; i++) {
        inverse[static_cast<size_t>(perm[static_cast<size_t>(i)])] = i;
    }
    for (int64_t third = 0; third < 2; third++) {
        permute_head_rows(data.data() + static_cast<size_t>(third * inner) * row_bytes,
                          row_bytes,
                          heads,
                          inverse,
                          scratch);
    }
    check(data == original, "the rewrite is exactly invertible");
}

// The 1-D q_norm / k_norm case: one "row" is one element.
static void test_norm_element_rewrite() {
    const int64_t head_dim          = 128;
    const std::vector<int64_t> perm = build_qk_head_permutation(head_dim, 96);

    std::vector<float> w(static_cast<size_t>(head_dim));
    for (int64_t i = 0; i < head_dim; i++) {
        w[static_cast<size_t>(i)] = static_cast<float>(i) * 0.25f - 3.f;
    }
    const std::vector<float> original = w;

    std::vector<uint8_t> scratch;
    permute_head_rows(reinterpret_cast<uint8_t*>(w.data()), sizeof(float), 1, perm, scratch);

    bool ok = true;
    for (int64_t i = 0; i < head_dim; i++) {
        ok = ok && w[static_cast<size_t>(i)] == original[static_cast<size_t>(perm[static_cast<size_t>(i)])];
    }
    check(ok, "norm rewrite is new[i] = old[perm[i]]");
}

int main() {
    printf("minimax-h3 q/k permutation\n");
    test_shipping_permutation();
    test_bijection_and_pairing();
    test_qkv_row_rewrite();
    test_norm_element_rewrite();
    if (failures == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL (%d)\n", failures);
    return 1;
}
