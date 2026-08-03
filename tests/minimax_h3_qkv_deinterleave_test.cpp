// MiniMax-H3 fused-QKV DE-INTERLEAVE: internal checks, plus a JSON dump for an element-for-element
// diff against the PR's own `reorder_interleaved_qkv` running under real torch.
//
// Header-only -- no ggml, no CUDA, no weights:
//
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror -I src
//         -o /tmp/h3_qkv_deint tests/minimax_h3_qkv_deinterleave_test.cpp
//   /tmp/h3_qkv_deint > /tmp/cpp_qkv_deinterleave.json          # internal checks -> stderr
//   /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
//         ~/handoffs/longcat-avatar.cpp/minimax-h3/tools/ref_qkv_deinterleave_torch.py
//         /tmp/cpp_qkv_deinterleave.json > /tmp/ref_qkv_deinterleave.json
//   python3 ~/handoffs/longcat-avatar.cpp/minimax-h3/tools/diff_qkv_deinterleave.py
//         /tmp/ref_qkv_deinterleave.json /tmp/cpp_qkv_deinterleave.json
//
// The reference side extracts `reorder_interleaved_qkv` VERBATIM BY AST NAME out of
// scripts/convert_minimax_h3_to_diffusers.py and execs it -- nothing is retyped, so a misreading of
// the reference cannot be carried into both sides.  Same method as W16/W24/W27.  The INPUTS travel
// in this file's JSON, so both sides operate on identical numbers with no shared generator.
//
// What is covered:
//   deinterleave        the rewrite itself, against the reference function
//   ordered             de-interleave THEN the q/k RoPE head-channel permutation -- the shipping
//                       composition, against reference(X) with the same permutation applied after
//   reversed            the WRONG order.  The diff tool requires it to DIFFER from `ordered`, so the
//                       ordering constraint is proven rather than asserted in a comment
//   q_norm_invariance   the claim that a per-head-dim gain vector needs no de-interleave, checked
//                       against the REFERENCE function rather than against our own reimplementation
//   probe               the layout detector's two F statistics, cross-checked against the validated
//                       Python implementation on shared fixtures

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model/diffusion/minimax_h3_qk_permute.hpp"
#include "model/diffusion/minimax_h3_qkv_layout_probe.hpp"

using MiniMaxH3::build_qk_head_permutation;
using MiniMaxH3::deinterleave_qkv_rows;
using MiniMaxH3::deinterleave_source_block;
using MiniMaxH3::permute_head_rows;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        failures++;
        fprintf(stderr, "  FAIL  %s\n", what);
    }
}

// -------------------------------------------------------------------------------------------------
// helpers
// -------------------------------------------------------------------------------------------------

struct Matrix {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<double> v;

    double& at(int64_t r, int64_t c) { return v[static_cast<size_t>(r * cols + c)]; }
    double at(int64_t r, int64_t c) const { return v[static_cast<size_t>(r * cols + c)]; }
};

static Matrix make_matrix(int64_t rows, int64_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.v.assign(static_cast<size_t>(rows * cols), 0.0);
    return m;
}

// Every element individually identifiable, so a misplaced row cannot hide behind a plausible value.
static Matrix indexed_matrix(int64_t rows, int64_t cols) {
    Matrix m = make_matrix(rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            m.at(r, c) = static_cast<double>(r) * 1000.0 + static_cast<double>(c);
        }
    }
    return m;
}

// A row here is `cols` doubles, so the byte-level helpers operate on real row strides.
static size_t row_bytes_of(const Matrix& m) {
    return static_cast<size_t>(m.cols) * sizeof(double);
}

static uint8_t* bytes_of(Matrix& m) {
    return reinterpret_cast<uint8_t*>(m.v.data());
}

static Matrix apply_deinterleave(const Matrix& in, int64_t heads, int64_t head_dim) {
    Matrix out = in;
    std::vector<uint8_t> scratch, visited;
    deinterleave_qkv_rows(bytes_of(out), row_bytes_of(out), heads, head_dim, scratch, visited);
    return out;
}

// The shipping q/k head-channel rewrite: q and k thirds only, v untouched.
static Matrix apply_rope_permute(const Matrix& in, int64_t heads, int64_t head_dim, int64_t rot_dim) {
    Matrix out                      = in;
    const std::vector<int64_t> perm = build_qk_head_permutation(head_dim, rot_dim);
    if (perm.empty()) {
        return out;
    }
    const int64_t inner = heads * head_dim;
    std::vector<uint8_t> scratch;
    for (int64_t third = 0; third < 2; third++) {
        permute_head_rows(bytes_of(out) + static_cast<size_t>(third * inner) * row_bytes_of(out),
                          row_bytes_of(out),
                          heads,
                          perm,
                          scratch);
    }
    return out;
}

static void emit_matrix(const char* key, const Matrix& m, bool last) {
    printf("\"%s\":[", key);
    for (size_t i = 0; i < m.v.size(); i++) {
        printf("%s%.17g", i ? "," : "", m.v[i]);
    }
    printf("]%s", last ? "" : ",");
}

static void emit_ints(const char* key, const std::vector<int64_t>& xs, bool last) {
    printf("\"%s\":[", key);
    for (size_t i = 0; i < xs.size(); i++) {
        printf("%s%" PRId64, i ? "," : "", xs[i]);
    }
    printf("]%s", last ? "" : ",");
}

// -------------------------------------------------------------------------------------------------
// internal checks -- properties that hold without any reference
// -------------------------------------------------------------------------------------------------

// A straightforward out-of-place gather, written the obvious way, as the oracle for the
// cycle-following in-place version that ships.
static Matrix naive_deinterleave(const Matrix& in, int64_t heads, int64_t head_dim) {
    Matrix out = make_matrix(in.rows, in.cols);
    for (int64_t blk = 0; blk < 3 * heads; blk++) {
        const int64_t src = deinterleave_source_block(blk, heads);
        for (int64_t i = 0; i < head_dim; i++) {
            for (int64_t c = 0; c < in.cols; c++) {
                out.at(blk * head_dim + i, c) = in.at(src * head_dim + i, c);
            }
        }
    }
    return out;
}

static void test_internal() {
    const int64_t heads_cases[]    = {1, 2, 3, 4, 5, 7, 8, 16, 56};
    const int64_t head_dim_cases[] = {1, 2, 3, 8, 128};

    for (int64_t heads : heads_cases) {
        for (int64_t head_dim : head_dim_cases) {
            if (heads * head_dim > 4096) {
                continue;
            }
            const Matrix in = indexed_matrix(3 * heads * head_dim, 2);

            const Matrix got  = apply_deinterleave(in, heads, head_dim);
            const Matrix want = naive_deinterleave(in, heads, head_dim);
            check(got.v == want.v, "cycle-following de-interleave == out-of-place gather");

            // The source-block map must be a bijection over the 3*heads blocks.
            std::vector<int> seen(static_cast<size_t>(3 * heads), 0);
            for (int64_t b = 0; b < 3 * heads; b++) {
                const int64_t s = deinterleave_source_block(b, heads);
                check(s >= 0 && s < 3 * heads, "source block in range");
                seen[static_cast<size_t>(s)]++;
            }
            for (int c : seen) {
                check(c == 1, "source block map is a bijection");
            }

            // Head h's q/k/v must land at blocks h, heads+h and 2*heads+h.
            bool placed = true;
            for (int64_t h = 0; h < heads; h++) {
                for (int64_t t = 0; t < 3; t++) {
                    for (int64_t i = 0; i < head_dim; i++) {
                        const int64_t src = (3 * h + t) * head_dim + i;
                        const int64_t dst = (t * heads + h) * head_dim + i;
                        placed            = placed && got.at(dst, 0) == in.at(src, 0);
                    }
                }
            }
            check(placed, "head h's q/k/v land at blocks h / heads+h / 2*heads+h");
        }
    }

    // Rows are never split: every output row is some whole input row.
    const Matrix in  = indexed_matrix(3 * 5 * 4, 3);
    const Matrix got = apply_deinterleave(in, 5, 4);
    bool whole_rows  = true;
    for (int64_t r = 0; r < got.rows; r++) {
        const int64_t src = static_cast<int64_t>(got.at(r, 0)) / 1000;
        for (int64_t c = 0; c < got.cols; c++) {
            whole_rows = whole_rows && got.at(r, c) == in.at(src, c);
        }
    }
    check(whole_rows, "every output row is a whole input row");

    // The de-interleave never moves a channel WITHIN a head block -- which is exactly why
    // q_norm/k_norm need no rewrite.  Checked here against our own code; checked against the
    // REFERENCE function in the torch harness.
    bool channels_fixed = true;
    for (int64_t r = 0; r < got.rows; r++) {
        const int64_t src = static_cast<int64_t>(got.at(r, 0)) / 1000;
        channels_fixed    = channels_fixed && (src % 4) == (r % 4);
    }
    check(channels_fixed, "channel index within a head block is invariant");

    // Order matters: the two compositions must differ, or the ordering constraint would be vacuous.
    const int64_t heads = 4, head_dim = 8, rot_dim = 6;
    const Matrix x        = indexed_matrix(3 * heads * head_dim, 2);
    const Matrix ordered  = apply_rope_permute(apply_deinterleave(x, heads, head_dim), heads, head_dim, rot_dim);
    const Matrix reversed = apply_deinterleave(apply_rope_permute(x, heads, head_dim, rot_dim), heads, head_dim);
    check(ordered.v != reversed.v, "de-interleave then rope-permute != rope-permute then de-interleave");
}

// -------------------------------------------------------------------------------------------------
// probe cross-check fixture
// -------------------------------------------------------------------------------------------------

// A q/k/v-like fixture: per-third scale, per-third column emphasis, per-head jitter, plus noise.
// Deterministic (a plain LCG) but the actual numbers travel in the JSON, so the Python side never
// has to reproduce this.
static Matrix probe_fixture(int64_t heads, int64_t head_dim, int64_t cols) {
    Matrix m       = make_matrix(3 * heads * head_dim, cols);
    uint64_t state = 0x2026080300000001ull;
    auto next      = [&]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((state >> 33) & 0x7fffff) / 8388608.0 - 0.5;
    };
    const double third_scale[3] = {1.0, 0.45, 2.1};
    for (int64_t t = 0; t < 3; t++) {
        for (int64_t h = 0; h < heads; h++) {
            const double jitter = 1.0 + 0.15 * next();
            for (int64_t i = 0; i < head_dim; i++) {
                const int64_t r = (t * heads + h) * head_dim + i;
                for (int64_t c = 0; c < cols; c++) {
                    // each third leans on a different part of the input axis
                    const double pos   = static_cast<double>(c) / static_cast<double>(cols);
                    const double shape = 1.0 + 0.8 * ((t == 0) ? pos : (t == 1) ? (1.0 - pos)
                                                                                : (1.0 - 2.0 * (pos - 0.5) * (pos - 0.5)));
                    m.at(r, c)         = third_scale[t] * jitter * shape * (1.0 + 0.5 * next());
                }
            }
        }
    }
    return m;
}

// The inverse of the de-interleave: [q_all; k_all; v_all] -> per-head interleaved.
static Matrix reinterleave(const Matrix& in, int64_t heads, int64_t head_dim) {
    Matrix out = make_matrix(in.rows, in.cols);
    for (int64_t blk = 0; blk < 3 * heads; blk++) {
        const int64_t dst = deinterleave_source_block(blk, heads);
        for (int64_t i = 0; i < head_dim; i++) {
            for (int64_t c = 0; c < in.cols; c++) {
                out.at(dst * head_dim + i, c) = in.at(blk * head_dim + i, c);
            }
        }
    }
    return out;
}

struct ProbeCase {
    const char* name;
    Matrix m;
    int64_t heads;
    int64_t head_dim;
    MiniMaxH3::QKVLayoutReport report;
};

static const char* verdict_name(MiniMaxH3::QKVLayout l) {
    switch (l) {
        case MiniMaxH3::QKVLayout::Contiguous:
            return "contiguous";
        case MiniMaxH3::QKVLayout::Interleaved:
            return "interleaved";
        default:
            return "ambiguous";
    }
}

static void emit_probe_case(ProbeCase& pc, bool last) {
    std::vector<float> f(pc.m.v.begin(), pc.m.v.end());
    pc.report = MiniMaxH3::probe_qkv_layout(f.data(), pc.m.rows, pc.m.cols, pc.heads, pc.head_dim);
    check(pc.report.valid, "probe ran");

    printf("{\"name\":\"%s\",\"heads\":%" PRId64 ",\"head_dim\":%" PRId64 ",\"cols\":%" PRId64 ",",
           pc.name,
           pc.heads,
           pc.head_dim,
           pc.m.cols);
    printf("\"verdict\":\"%s\",", verdict_name(pc.report.layout));
    printf("\"norm_f\":{\"contiguous\":%.17g,\"interleaved\":%.17g},", pc.report.norm.f_contiguous, pc.report.norm.f_interleaved);
    printf("\"profile_f\":{\"contiguous\":%.17g,\"interleaved\":%.17g},",
           pc.report.profile.f_contiguous,
           pc.report.profile.f_interleaved);
    // f32 round-tripped, so the Python side probes exactly the numbers the C++ probed.
    printf("\"w\":[");
    for (size_t i = 0; i < f.size(); i++) {
        printf("%s%.9g", i ? "," : "", static_cast<double>(f[i]));
    }
    printf("]}%s", last ? "" : ",");
}

// -------------------------------------------------------------------------------------------------

struct Case {
    const char* name;
    int64_t heads;
    int64_t head_dim;
    int64_t cols;
    int64_t rot_dim;  // 0 = no rope stage for this case
};

// Kept small enough that the JSON stays readable, except for one case at the real H3 geometry.
static const Case CASES[] = {
    {"min", 1, 1, 2, 0},
    {"tiny", 2, 2, 3, 0},
    {"refiner_like", 2, 6, 4, 0},
    {"small", 4, 3, 5, 0},
    {"roped_8", 4, 8, 2, 6},
    {"roped_16", 3, 16, 2, 12},
    {"h3_geometry", 56, 128, 1, 96},
};

int main() {
    fprintf(stderr, "minimax-h3 fused-qkv de-interleave\n");
    test_internal();

    printf("{\"cases\":[");
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        const Case& c   = CASES[i];
        const Matrix in = indexed_matrix(3 * c.heads * c.head_dim, c.cols);

        const Matrix deint = apply_deinterleave(in, c.heads, c.head_dim);
        printf("{\"name\":\"%s\",\"heads\":%" PRId64 ",\"head_dim\":%" PRId64 ",\"cols\":%" PRId64
               ",\"rot_dim\":%" PRId64 ",",
               c.name,
               c.heads,
               c.head_dim,
               c.cols,
               c.rot_dim);
        emit_matrix("input", in, false);
        emit_matrix("deinterleave", deint, false);
        if (c.rot_dim > 0) {
            emit_ints("perm", build_qk_head_permutation(c.head_dim, c.rot_dim), false);
            emit_matrix("ordered", apply_rope_permute(deint, c.heads, c.head_dim, c.rot_dim), false);
            emit_matrix("reversed",
                        apply_deinterleave(apply_rope_permute(in, c.heads, c.head_dim, c.rot_dim), c.heads, c.head_dim),
                        false);
        }

        // q_norm invariance fixture: every row carries its channel index within its head block, so
        // the reference's output must carry the identical pattern.  The torch harness asserts that
        // property on `reorder_interleaved_qkv` ITSELF -- this side just supplies the matrices.
        Matrix chan = make_matrix(in.rows, 1);
        for (int64_t r = 0; r < in.rows; r++) {
            chan.at(r, 0) = static_cast<double>(r % c.head_dim);
        }
        emit_matrix("q_norm_in", chan, false);
        emit_matrix("q_norm_out", apply_deinterleave(chan, c.heads, c.head_dim), true);
        printf("}%s", i + 1 == sizeof(CASES) / sizeof(CASES[0]) ? "" : ",");
    }
    printf("],\"probe\":[");

    ProbeCase pcs[] = {
        {"fixture_contiguous", probe_fixture(16, 8, 32), 16, 8, {}},
        {"fixture_interleaved", reinterleave(probe_fixture(16, 8, 32), 16, 8), 16, 8, {}},
    };
    for (size_t i = 0; i < sizeof(pcs) / sizeof(pcs[0]); i++) {
        emit_probe_case(pcs[i], i + 1 == sizeof(pcs) / sizeof(pcs[0]));
    }
    printf("]}\n");

    // The probe must call the two fixtures the way they were built, or the C++ twin of the
    // validated detector is not actually working.
    check(pcs[0].report.layout == MiniMaxH3::QKVLayout::Contiguous, "probe calls the contiguous fixture contiguous");
    check(pcs[1].report.layout == MiniMaxH3::QKVLayout::Interleaved, "probe calls the interleaved fixture interleaved");

    if (failures == 0) {
        fprintf(stderr, "internal checks PASS\n");
        return 0;
    }
    fprintf(stderr, "internal checks FAIL (%d)\n", failures);
    return 1;
}
