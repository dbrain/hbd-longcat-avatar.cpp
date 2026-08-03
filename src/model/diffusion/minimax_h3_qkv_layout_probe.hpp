#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_QKV_LAYOUT_PROBE_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_QKV_LAYOUT_PROBE_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

// Data-driven "is this fused QKV weight per-head INTERLEAVED or CONTIGUOUS?" probe.
//
// Included by the GGUF converter and by tests only -- NOT by the DiT graph.  The runtime never
// guesses a layout; it consumes whatever the converter produced.  This exists so the converter can
// state loudly what the numbers say while still refusing to act on them by itself.
//
// -------------------------------------------------------------------------------------------------
// The statistic (a line-for-line twin of tools/h3_qkv_layout_probe.py -- keep them in step)
//
// Cut the weight into `3 * heads` blocks of `head_dim` rows.  Each hypothesis reads a block index
// `c` as a (third, head) pair and they disagree only about which coordinate is which:
//
//     CONTIGUOUS   third = c / heads,  head = c % heads
//     INTERLEAVED  third = c % 3,      head = c / 3
//
// Both are the same balanced 3 x heads factorial design over the same blocks, so one scorer does
// both.  Two per-block features:
//
//     norm     scalar     mean row L2 norm.  q/k/v sit at different weight scales in practice.
//     profile  64-vector  the block's energy over the INPUT axis, binned and L2-normalised.  q, k
//                         and v read the same residual stream but weight its columns differently.
//
// Scoring is a TWO-WAY ANOVA on the `third` main effect with the `head` main effect removed:
//
//     F_third = (SS_third / (3 - 1)) / (SS_resid / ((3 - 1) * (heads - 1)))
//
// The two-way form is load-bearing.  A one-way "group by third" F fires on the WRONG hypothesis
// whenever the weight has head-INDEX structure, and trained attention weights have plenty: reading
// an interleaved tensor with the contiguous grouping is precisely "cut the heads into three
// consecutive blocks", which is significant on its own.  Measured on Qwen3-VL, the one-way version
// returned confident false positives on a fixture with no q/k/v structure at all.
//
// Confidence is a permutation test -- relabel the blocks at random into the same 3 x heads grid and
// count how often chance reaches this F.  A verdict is issued only when all four of these hold: the
// two features AGREE on a winner, at least one is significant for the winner, NEITHER is significant
// for the loser, and one of them separates the hypotheses by at least QKV_PROBE_MIN_F_RATIO.
// Anything else is Unknown, and Unknown is a real answer: a detector that always commits cannot
// fail, and a detector that cannot fail is worthless.
//
// The thresholds were set against the exchangeable null rather than chosen for taste.  At p <= 0.01
// that null committed on ~5% of trials (four looks at alpha = 0.01, with correlated features);
// p <= 0.002 takes it to 0/64, and costs the real fused-qkv fixtures nothing because their p sits at
// the permutation floor of 2.5e-4.
//
// -------------------------------------------------------------------------------------------------
// Validation, on real trained weights, no H3 required
//
// tools/h3_qkv_layout_probe.py --selftest builds fixtures out of Qwen3-VL and feeds each one BOTH
// ways round, so half of every batch is the layout the detector must NOT return:
//
//     vision    model.visual.blocks.N.attn.qkv.weight -- a genuinely fused MHA qkv whose layout is
//               pinned by transformers itself (qkv(x).reshape(seq, 3, heads, -1)), i.e. contiguous
//     text      GQA q/k/v re-fused into a fair MHA shape (first n_kv q heads, all n_kv k/v heads)
//     nullperm  three head-blocks of q_proj alone, shuffled -- exchangeable, so a true null
//
// 256 trials at --limit 16: ZERO wrong verdicts anywhere.  vision 64/64 exactly right; text 56/64
// right with 8 abstentions and nothing wrong; the true null abstained 64/64.

namespace MiniMaxH3 {

    enum class QKVLayout {
        Unknown,     // the evidence does not separate the two
        Contiguous,  // [q_all; k_all; v_all] -- what the engine's split_qkv wants
        Interleaved  // [head0: q k v, head1: q k v, ...] -- what a raw MiniMax shard holds
    };

    struct QKVLayoutFeatureScore {
        double f_contiguous  = 0.0;
        double f_interleaved = 0.0;
        double p_contiguous  = 1.0;
        double p_interleaved = 1.0;

        QKVLayout winner() const { return f_contiguous >= f_interleaved ? QKVLayout::Contiguous : QKVLayout::Interleaved; }
        double margin() const {
            const double lo = std::min(f_contiguous, f_interleaved);
            const double hi = std::max(f_contiguous, f_interleaved);
            return lo <= 0.0 ? std::numeric_limits<double>::infinity() : hi / lo;
        }
    };

    struct QKVLayoutReport {
        bool valid       = false;
        QKVLayout layout = QKVLayout::Unknown;
        QKVLayoutFeatureScore norm;
        QKVLayoutFeatureScore profile;
    };

    inline const char* qkv_layout_name(QKVLayout l) {
        switch (l) {
            case QKVLayout::Contiguous:
                return "contiguous [q_all; k_all; v_all]";
            case QKVLayout::Interleaved:
                return "per-head interleaved [h0: q k v, h1: q k v, ...]";
            default:
                return "AMBIGUOUS";
        }
    }

    // Must match tools/h3_qkv_layout_probe.py.
    constexpr int64_t QKV_PROBE_BINS       = 64;
    constexpr int64_t QKV_PROBE_SHUFFLES   = 4000;
    constexpr double QKV_PROBE_P_THRESHOLD = 0.002;
    constexpr double QKV_PROBE_MIN_F_RATIO = 4.0;
    constexpr uint32_t QKV_PROBE_SEED      = 20260803u;

    // Per-block features.  `data` is row-major [rows][cols] float, rows = 3 * heads * head_dim.
    // `norm_out` is [3*heads]; `profile_out` is [3*heads][bins] row-major.
    inline bool qkv_layout_features(const float* data,
                                    int64_t rows,
                                    int64_t cols,
                                    int64_t heads,
                                    int64_t head_dim,
                                    std::vector<double>& norm_out,
                                    std::vector<double>& profile_out,
                                    int64_t& bins_out) {
        if (data == nullptr || heads <= 1 || head_dim <= 0 || cols <= 0 || rows != 3 * heads * head_dim) {
            return false;
        }
        const int64_t blocks = 3 * heads;
        const int64_t bins   = std::min<int64_t>(QKV_PROBE_BINS, cols);
        bins_out             = bins;
        norm_out.assign(static_cast<size_t>(blocks), 0.0);
        profile_out.assign(static_cast<size_t>(blocks * bins), 0.0);

        std::vector<int64_t> edge(static_cast<size_t>(bins + 1));
        for (int64_t b = 0; b <= bins; b++) {
            edge[static_cast<size_t>(b)] = (cols * b) / bins;
        }

        for (int64_t blk = 0; blk < blocks; blk++) {
            double norm_sum = 0.0;
            double* prof    = profile_out.data() + static_cast<size_t>(blk * bins);
            for (int64_t r = blk * head_dim; r < (blk + 1) * head_dim; r++) {
                const float* row = data + static_cast<size_t>(r) * static_cast<size_t>(cols);
                double row_sq    = 0.0;
                for (int64_t b = 0; b < bins; b++) {
                    double bin_sq = 0.0;
                    for (int64_t c = edge[static_cast<size_t>(b)]; c < edge[static_cast<size_t>(b + 1)]; c++) {
                        const double v = static_cast<double>(row[c]);
                        bin_sq += v * v;
                    }
                    prof[b] += bin_sq;
                    row_sq += bin_sq;
                }
                norm_sum += std::sqrt(row_sq);
            }
            norm_out[static_cast<size_t>(blk)] = norm_sum / static_cast<double>(head_dim);

            double len = 0.0;
            for (int64_t b = 0; b < bins; b++) {
                prof[b] = std::sqrt(prof[b]);
                len += prof[b] * prof[b];
            }
            len = std::sqrt(len) + 1e-30;
            for (int64_t b = 0; b < bins; b++) {
                prof[b] /= len;
            }
        }
        return true;
    }

    // Two-way ANOVA F for the `third` main effect, head main effect removed.  `order` maps grid cell
    // (third * heads + head) to a block index, so one routine scores both hypotheses and the
    // permutation null.  Scalar and vector features alike: the sums of squares are traces and the
    // per-dimension degrees of freedom cancel in the ratio.
    inline double qkv_two_way_f(const double* x, int64_t dim, int64_t heads, const int64_t* order) {
        const int64_t n = 3 * heads;
        const size_t d  = static_cast<size_t>(dim);
        auto cell       = [&](int64_t t, int64_t h) { return x + static_cast<size_t>(order[t * heads + h]) * d; };

        std::vector<double> grand(d, 0.0), m_third(3 * d, 0.0), m_head(static_cast<size_t>(heads) * d, 0.0);
        for (int64_t t = 0; t < 3; t++) {
            for (int64_t h = 0; h < heads; h++) {
                const double* v = cell(t, h);
                for (size_t k = 0; k < d; k++) {
                    grand[k] += v[k];
                    m_third[static_cast<size_t>(t) * d + k] += v[k];
                    m_head[static_cast<size_t>(h) * d + k] += v[k];
                }
            }
        }
        for (size_t k = 0; k < d; k++) {
            grand[k] /= static_cast<double>(n);
        }
        for (int64_t t = 0; t < 3; t++) {
            for (size_t k = 0; k < d; k++) {
                m_third[static_cast<size_t>(t) * d + k] /= static_cast<double>(heads);
            }
        }
        for (int64_t h = 0; h < heads; h++) {
            for (size_t k = 0; k < d; k++) {
                m_head[static_cast<size_t>(h) * d + k] /= 3.0;
            }
        }

        double ss_third = 0.0, ss_head = 0.0, ss_total = 0.0;
        for (int64_t t = 0; t < 3; t++) {
            for (size_t k = 0; k < d; k++) {
                const double e = m_third[static_cast<size_t>(t) * d + k] - grand[k];
                ss_third += e * e;
            }
        }
        ss_third *= static_cast<double>(heads);
        for (int64_t h = 0; h < heads; h++) {
            for (size_t k = 0; k < d; k++) {
                const double e = m_head[static_cast<size_t>(h) * d + k] - grand[k];
                ss_head += e * e;
            }
        }
        ss_head *= 3.0;
        for (int64_t t = 0; t < 3; t++) {
            for (int64_t h = 0; h < heads; h++) {
                const double* v = cell(t, h);
                for (size_t k = 0; k < d; k++) {
                    const double e = v[k] - grand[k];
                    ss_total += e * e;
                }
            }
        }
        const double ss_resid = std::max(ss_total - ss_third - ss_head, 0.0);
        if (ss_resid <= 0.0) {
            return ss_third > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
        }
        const double df_third = 2.0;
        const double df_resid = 2.0 * static_cast<double>(heads - 1);
        return (ss_third / df_third) / (ss_resid / df_resid);
    }

    inline QKVLayoutFeatureScore qkv_score_feature(const std::vector<double>& x, int64_t dim, int64_t heads) {
        const int64_t n = 3 * heads;
        std::vector<int64_t> contiguous(static_cast<size_t>(n)), interleaved(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            // cell i == (third = i / heads, head = i % heads)
            contiguous[static_cast<size_t>(i)]  = i;
            interleaved[static_cast<size_t>(i)] = 3 * (i % heads) + (i / heads);
        }

        QKVLayoutFeatureScore s;
        s.f_contiguous  = qkv_two_way_f(x.data(), dim, heads, contiguous.data());
        s.f_interleaved = qkv_two_way_f(x.data(), dim, heads, interleaved.data());

        std::mt19937 rng(QKV_PROBE_SEED);
        std::vector<int64_t> shuffled = contiguous;
        int64_t ge_c = 0, ge_i = 0;
        for (int64_t it = 0; it < QKV_PROBE_SHUFFLES; it++) {
            std::shuffle(shuffled.begin(), shuffled.end(), rng);
            const double f = qkv_two_way_f(x.data(), dim, heads, shuffled.data());
            ge_c += (f >= s.f_contiguous) ? 1 : 0;
            ge_i += (f >= s.f_interleaved) ? 1 : 0;
        }
        s.p_contiguous  = static_cast<double>(ge_c + 1) / static_cast<double>(QKV_PROBE_SHUFFLES + 1);
        s.p_interleaved = static_cast<double>(ge_i + 1) / static_cast<double>(QKV_PROBE_SHUFFLES + 1);
        return s;
    }

    inline QKVLayoutReport probe_qkv_layout(const float* data, int64_t rows, int64_t cols, int64_t heads, int64_t head_dim) {
        QKVLayoutReport report;
        std::vector<double> norm, profile;
        int64_t bins = 0;
        if (!qkv_layout_features(data, rows, cols, heads, head_dim, norm, profile, bins)) {
            return report;
        }
        report.valid   = true;
        report.norm    = qkv_score_feature(norm, 1, heads);
        report.profile = qkv_score_feature(profile, bins, heads);

        const QKVLayout wn = report.norm.winner();
        if (wn != report.profile.winner()) {
            return report;  // features disagree -> Unknown
        }
        const bool loser_is_contiguous = wn == QKVLayout::Interleaved;
        const double p_win_norm        = wn == QKVLayout::Contiguous ? report.norm.p_contiguous : report.norm.p_interleaved;
        const double p_win_prof        = wn == QKVLayout::Contiguous ? report.profile.p_contiguous : report.profile.p_interleaved;
        const double p_lose_norm       = loser_is_contiguous ? report.norm.p_contiguous : report.norm.p_interleaved;
        const double p_lose_prof       = loser_is_contiguous ? report.profile.p_contiguous : report.profile.p_interleaved;

        const bool winner_sig = std::min(p_win_norm, p_win_prof) <= QKV_PROBE_P_THRESHOLD;
        const bool loser_null = p_lose_norm > QKV_PROBE_P_THRESHOLD && p_lose_prof > QKV_PROBE_P_THRESHOLD;
        const bool separated  = std::max(report.norm.margin(), report.profile.margin()) >= QKV_PROBE_MIN_F_RATIO;
        if (winner_sig && loser_null && separated) {
            report.layout = wn;
        }
        return report;
    }

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_QKV_LAYOUT_PROBE_HPP__
