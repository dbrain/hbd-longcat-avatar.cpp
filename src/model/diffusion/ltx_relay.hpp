#ifndef __SD_MODEL_DIFFUSION_LTX_RELAY_HPP__
#define __SD_MODEL_DIFFUSION_LTX_RELAY_HPP__

// Prompt Relay — inference-time temporal control for multi-event video
// generation (arXiv 2604.10030, Chen/Huang/Liu 2026-04-11), adapted to the
// LTX-AV joint transformer.
//
//   Attn(phi(z_t), psi(P)) = softmax(QK^T/sqrt(d) - C(Q,K)) V
//   C(i,j) = strength * ReLU(|t(i) - m_s| - w)^2 / (2 sigma^2)   for key j in prompt piece s
//   sigma  = (L - w) / sqrt(2 ln(1/eps))
//   w      = L - (2 latent frames)                               (the paper's ablated best)
//
// Everything here is in SECONDS, not frame indices. The video and audio
// cross-attention streams have different query->time mappings (and the video
// mapping additionally shifts when base temporal windowing splits a shot into
// tiles), so the caller hands us an explicit per-query-frame time table for
// each stream and this file never has to re-derive a coordinate system.
//
// The paper's own stated limitation is that persistent elements drift across
// segments because each segment only attends to its local prompt; their fix is
// a global prompt with zero penalty. That is piece 0 here, and a non-empty
// global piece is a hard requirement rather than a style rule: it is the only
// thing guaranteeing every query token keeps some unpenalised key, so the
// masked softmax can never degenerate.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "ggml.h"

namespace sd {
namespace ltx_relay {

    struct Beat {
        // Beat midpoint, in seconds on the rendered segment timeline.
        float mid = 0.f;
        // Half-span of the beat's Voronoi cell, in seconds. This is the paper's
        // L: the distance from the midpoint at which the penalty reaches eps.
        float half_span = 0.f;
        // Flat-top half-width in seconds; the penalty is exactly zero inside
        // [mid - window, mid + window]. Resolved by the caller.
        float window = 0.f;
        // Multiplier on the penalty. One is the paper.
        float strength = 1.f;
    };

    struct Plan {
        std::vector<Beat> beats;
        // Per prompt token (pre-connector, i.e. indexed exactly like the
        // conditioner's context sequence): the beat that token belongs to, or
        // -1 for the global anchor. Key positions past the end of this vector
        // — the connector's learnable registers — are treated as global.
        std::vector<int32_t> token_beat;
        // Query-frame time tables, in seconds, for THIS window/tile only.
        std::vector<float> video_frame_time;
        std::vector<float> audio_frame_time;
        float eps       = 0.01f;
        float audio_eps = 0.01f;
        // Reference-implementation sigma, in SECONDS, held constant instead of
        // the paper's (L - w)/sqrt(2 ln(1/eps)). The relay everyone actually
        // renders with (WhatDreamsCost-ComfyUI/prompt_relay.py) uses
        // sigma = 1/ln(1/eps) in LATENT FRAMES, independent of the beat's cell,
        // which at its default eps=0.001 is 0.1448 latent frames -- roughly a
        // quarter of what the paper form yields here, so ~21x the penalty at the
        // same distance. Zero keeps the paper form.
        float sigma_fixed = 0.f;
        // C grows quadratically with distance from the beat, and the mask is
        // F16 (max 65504): a long windowed shot can sit far enough from a beat
        // to overflow to -inf. exp(-60) is already zero in F32, so clamping
        // here is free and keeps every value finite.
        float max_cost = 60.f;
        // Bumped by the caller whenever the mask contents change (per tile, per
        // shot). The runner caches the materialised mask against this.
        uint64_t revision = 0;

        bool has_video() const {
            return !beats.empty() && !token_beat.empty() && !video_frame_time.empty();
        }
        bool has_audio() const {
            return !beats.empty() && !token_beat.empty() && !audio_frame_time.empty() && audio_eps >= 0.f;
        }
    };

    // sigma from the paper: at |t - mid| == L the penalty is exactly ln(1/eps),
    // so exp(-C) == eps at the edge of the beat's cell. A degenerate span
    // (window >= half_span) would divide by zero; floor it at one millisecond,
    // which makes the crossfade a hard edge rather than a NaN.
    inline float beat_sigma(const Beat& beat, float eps, float sigma_fixed = 0.f) {
        if (sigma_fixed > 0.f) {
            return sigma_fixed;
        }
        const float clamped_eps = std::min(std::max(eps, 1e-6f), 0.5f);
        const float reach       = std::max(beat.half_span - beat.window, 1e-3f);
        return reach / std::sqrt(2.f * std::log(1.f / clamped_eps));
    }

    // Content fingerprint for the materialised-mask cache. The runner caches a
    // mask against {revision, L_q, L_k} and outlives any one shot, so a
    // per-shot counter ALIASES: on a uniform chain every shot's first window
    // carries revision 1 and the same L_q/L_k, and shot 2 onwards then renders
    // with shot 1's beat times. Hash what actually decides the mask instead --
    // equal content shares an entry, different content cannot.
    inline uint64_t plan_fingerprint(const Plan& plan) {
        uint64_t hash = 1469598103934665603ull;  // FNV-1a
        auto mix      = [&hash](const void* data, size_t bytes) {
            const auto* raw = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < bytes; ++i) {
                hash ^= raw[i];
                hash *= 1099511628211ull;
            }
        };
        // Member by member, never the struct: padding bytes are indeterminate.
        for (const Beat& beat : plan.beats) {
            mix(&beat.mid, sizeof(beat.mid));
            mix(&beat.half_span, sizeof(beat.half_span));
            mix(&beat.window, sizeof(beat.window));
            mix(&beat.strength, sizeof(beat.strength));
        }
        if (!plan.token_beat.empty()) {
            mix(plan.token_beat.data(), plan.token_beat.size() * sizeof(int32_t));
        }
        if (!plan.video_frame_time.empty()) {
            mix(plan.video_frame_time.data(), plan.video_frame_time.size() * sizeof(float));
        }
        if (!plan.audio_frame_time.empty()) {
            mix(plan.audio_frame_time.data(), plan.audio_frame_time.size() * sizeof(float));
        }
        mix(&plan.eps, sizeof(plan.eps));
        mix(&plan.audio_eps, sizeof(plan.audio_eps));
        mix(&plan.max_cost, sizeof(plan.max_cost));
        mix(&plan.sigma_fixed, sizeof(plan.sigma_fixed));
        // Zero reads as "no plan cached yet" in the runner's key; keep it out.
        return hash == 0 ? 1ull : hash;
    }

    // Penalty for one query time against one beat. Non-negative.
    inline float beat_cost(const Beat& beat, float sigma, float query_time, float max_cost) {
        const float distance = std::fabs(query_time - beat.mid) - beat.window;
        if (distance <= 0.f) {
            return 0.f;
        }
        const float cost = beat.strength * distance * distance / (2.f * sigma * sigma);
        return std::min(cost, max_cost);
    }

    // LTX_RELAY_ISOLATE=1 turns on connector piece isolation: the registers are
    // partitioned across the pieces and the connector's self-attention is masked
    // block-diagonally, so a register run carries one beat instead of a blend of
    // all of them. Off leaves the graph exactly as it was.
    inline bool isolate_enabled() {
        static const bool value = [] {
            const char* raw = std::getenv("LTX_RELAY_ISOLATE");
            return raw != nullptr && raw[0] == '1';
        }();
        return value;
    }

    // Beat ownership for the POST-connector key axis.
    //
    // GPU-proven 2026-07-28: suppressing the register block does not free the
    // prompt, it destroys it (the cream-wall prompt rendered two people kissing).
    // The registers are not padding diluting the prompt -- they ARE the channel
    // the DiT reads the prompt through. The connector's self-attention is what
    // fills them, and the prompt tokens are only its input.
    //
    // So the fix is not to remove them but to make them ATTRIBUTABLE: give each
    // piece its own contiguous run of registers. Combined with the isolation mask
    // below, register run k is filled from the global anchor plus beat k alone --
    // which is just an ordinary prompt, so each run stays in distribution -- and
    // relay can then penalise it. Costs nothing: same key count, same graph, one
    // extra CPU-side mask.
    //
    // Registers are handed out in contiguous blocks rather than round-robin so
    // each group keeps an unbroken span of connector RoPE positions.
    inline void build_connector_key_beat(const std::vector<int32_t>& token_beat,
                                         size_t beat_count,
                                         int64_t length,
                                         std::vector<int32_t>& out) {
        out.assign(static_cast<size_t>(std::max<int64_t>(0, length)), -1);
        const int64_t mapped = std::min<int64_t>(length, static_cast<int64_t>(token_beat.size()));
        for (int64_t i = 0; i < mapped; ++i) {
            out[static_cast<size_t>(i)] = token_beat[static_cast<size_t>(i)];
        }
        const int64_t registers = length - mapped;
        if (registers <= 0 || beat_count == 0) {
            return;
        }
        // Group 0 is the global anchor: it keeps a register share of its own, so
        // a query far from every beat still has unpenalised summary keys and the
        // masked softmax cannot degenerate.
        const int64_t groups = static_cast<int64_t>(beat_count) + 1;
        for (int64_t i = 0; i < registers; ++i) {
            const int64_t group             = i * groups / registers;
            out[static_cast<size_t>(mapped + i)] = group == 0 ? -1 : static_cast<int32_t>(group - 1);
        }
    }

    // Materialise the connector's [L, L] F16 self-attention bias for piece
    // ISOLATION.
    //
    // Penalising a beat's keys downstream only biases; it cannot isolate, because
    // the connector's two self-attention layers run unmasked over every piece at
    // once, so by the time the DiT sees a key its CONTENT is already a blend of
    // all the beats. This is what makes a relay mask on a blended key axis feel
    // like one flat prompt no matter how hard you push the penalty.
    //
    // The fix is upstream of the blend: let a beat's tokens attend only to
    // themselves and to the global anchor. The global piece stays visible to
    // everyone (it is the shot's setting, and it is what keeps every row of the
    // softmax non-degenerate), and any key past the token map -- the learnable
    // registers -- stays visible too, since it belongs to no piece.
    inline void build_connector_isolation_mask_f16(const std::vector<int32_t>& key_beat,
                                                   int64_t length,
                                                   std::vector<ggml_fp16_t>& out) {
        const std::vector<int32_t>& token_beat = key_beat;
        // -1e4 rather than the relay's -60: this is a hard partition, not a decay,
        // and it still sits well inside F16 range (max 65504).
        const ggml_fp16_t blocked = ggml_fp32_to_fp16(-1e4f);
        const ggml_fp16_t open    = ggml_fp32_to_fp16(0.f);
        out.assign(static_cast<size_t>(length * length), open);
        const int64_t mapped = std::min<int64_t>(length, static_cast<int64_t>(token_beat.size()));
        for (int64_t query = 0; query < mapped; ++query) {
            const int32_t q_piece = token_beat[static_cast<size_t>(query)];
            if (q_piece < 0) {
                continue;  // the global anchor sees everything
            }
            for (int64_t key = 0; key < mapped; ++key) {
                const int32_t k_piece = token_beat[static_cast<size_t>(key)];
                if (k_piece >= 0 && k_piece != q_piece) {
                    out[static_cast<size_t>(query * length + key)] = blocked;
                }
            }
        }
    }

    // Materialise a [L_k, L_q] F16 additive attention bias, contiguous with
    // ne0 == L_k, shaped so that ggml_ext_attention_ext's repeat/cast fallback
    // never fires. `tokens_per_frame` is the number of query tokens that share
    // one entry of `frame_time` (W_lat * H_lat for video, 1 for audio).
    //
    // Rows are identical within a frame, so this fills one row per frame and
    // copies it across the frame's tokens.
    inline void build_mask_f16(const Plan& plan,
                               const std::vector<int32_t>& key_beat,
                               const std::vector<float>& frame_time,
                               int64_t tokens_per_frame,
                               int64_t L_k,
                               float eps,
                               std::vector<ggml_fp16_t>& out) {
        const int64_t frames = static_cast<int64_t>(frame_time.size());
        const int64_t L_q    = frames * tokens_per_frame;
        out.assign(static_cast<size_t>(L_q * L_k), ggml_fp32_to_fp16(0.f));
        if (L_q <= 0 || L_k <= 0 || plan.beats.empty()) {
            return;
        }

        std::vector<float> sigmas(plan.beats.size());
        for (size_t beat = 0; beat < plan.beats.size(); ++beat) {
            sigmas[beat] = beat_sigma(plan.beats[beat], eps, plan.sigma_fixed);
        }

        const int64_t token_keys = std::min<int64_t>(L_k, static_cast<int64_t>(key_beat.size()));
        std::vector<ggml_fp16_t> row(static_cast<size_t>(L_k), ggml_fp32_to_fp16(0.f));
        std::vector<ggml_fp16_t> beat_bias(plan.beats.size());

        for (int64_t frame = 0; frame < frames; ++frame) {
            const float query_time = frame_time[static_cast<size_t>(frame)];
            for (size_t beat = 0; beat < plan.beats.size(); ++beat) {
                beat_bias[beat] = ggml_fp32_to_fp16(-beat_cost(plan.beats[beat], sigmas[beat], query_time, plan.max_cost));
            }
            for (int64_t key = 0; key < token_keys; ++key) {
                const int32_t beat = key_beat[static_cast<size_t>(key)];
                row[static_cast<size_t>(key)] =
                    (beat < 0 || static_cast<size_t>(beat) >= plan.beats.size())
                        ? ggml_fp32_to_fp16(0.f)
                        : beat_bias[static_cast<size_t>(beat)];
            }
            for (int64_t token = 0; token < tokens_per_frame; ++token) {
                const int64_t query = frame * tokens_per_frame + token;
                std::copy(row.begin(), row.end(), out.begin() + static_cast<size_t>(query * L_k));
            }
        }
    }

    // ---------------------------------------------------------------------
    // Temporal VIDEO SELF-attention bias (LTX_RELAY_SELF_MASK). EXPERIMENTAL.
    // ---------------------------------------------------------------------
    //
    // The relay mask above biases attn2 -- TEXT cross-attention -- only. Every
    // DiT block runs video self-attention (attn1) over the whole flattened
    // sequence FIRST and completely unmasked, so the ordering per block is
    //     attn1 (unmasked, all frames <-> all frames) -> attn2 (relay-masked).
    // Once any query near a beat has picked the beat's content up through the
    // text keys, the NEXT block's unmasked self-attention transports that
    // representation to frame-0 tokens, and no text mask can remove a signal
    // that is already resident in the video tokens. That bypass is the best
    // explanation for relay's ~50% seed-dependent hit rate on beats in the
    // first half of a clip: a late beat has few blocks left to leak backwards
    // through, an early beat has all of them.
    //
    // What this closes, and nothing more: a query whose time is BEFORE a beat's
    // protected window attending to a key whose time is INSIDE or AFTER it.
    //   * Same-frame and spatial attention: untouched (t(k) == t(q) is never
    //     penalised), so per-frame structure is unaffected.
    //   * Ordinary past->future attention: untouched, because the penalty is
    //     asymmetric by construction -- a LATER query keeps full sight of every
    //     earlier key, which is what carries subject persistence and continuity.
    //   * A beat's own window and everything after it: untouched.
    // Deliberately NOT a symmetric distance penalty and NOT a hard block on
    // non-local attention: either one severs the long-range coupling the model
    // uses to keep a subject the same subject across a shot.
    //
    // Magnitude reuses the text mask's own cost curve, evaluated at the QUERY's
    // time against the beat being crossed, so "how hard is this query pushed off
    // the beat's prompt" and "how hard is it pushed off the beat's pixels" agree
    // by construction, and the whole thing decays to zero as the query
    // approaches the window instead of introducing a hard edge in time.

    // Strength multiplier. Absent, empty or <= 0 is OFF -- the caller then
    // allocates nothing and the graph is byte-identical to a build without this
    // feature. "1" is the natural unit (exactly the text mask's penalty);
    // anything else scales it, and the result is clamped to Plan::max_cost so a
    // large value saturates instead of overflowing F16.
    //
    // What "1" actually costs, at the shipped defaults (LTX_RELAY_MAX_L=1.0s,
    // 24fps => latent_secs=0.333, so window=0.333 and sigma=0.220), for a beat at
    // 4.0s of a 5s shot:
    //     query 3.5s -> exp(-0.29) = 0.75   barely touched
    //     query 3.0s -> exp(-4.6)  = 0.01   the eps floor, by construction
    //     query 0.0s -> clamped    = 0      total, the frame-0 leak
    // i.e. a query within half a second of the beat keeps its view across it and
    // only a query more than ~1.3s early loses it outright. That shape is the
    // point; if the far end proves too blunt (early frames drifting away from
    // late ones, since a frame-0 token can no longer see the shot's end at all),
    // the knob to reach for is a FRACTION -- 0.02 turns the frame-0 block into
    // ~6% throughput -- not a larger number.
    inline float self_mask_strength() {
        static const float value = [] {
            const char* raw = std::getenv("LTX_RELAY_SELF_MASK");
            if (raw == nullptr || raw[0] == '\0') {
                return 0.f;
            }
            const float parsed = std::strtof(raw, nullptr);
            return parsed > 0.f ? parsed : 0.f;
        }();
        return value;
    }

    // Refusal threshold for the DENSE [L, L] F16 mask, in MiB.
    //
    // ggml gives no cheaper representation: ggml_flash_attn_ext asserts the mask
    // is F16 AND ggml_is_contiguous (ggml.c:5537-5538) and broadcasts only ne2/ne3
    // (head/batch, ggml.c:5541-5542) -- never the query axis, which
    // ggml_ext_attention_ext already materialises by hand for that reason
    // (ggml_extend.hpp:1497-1509). So the bias, whose information content is only
    // [n_latent_frames, n_latent_frames], has to be expanded to [L_k, L_q] before
    // the kernel sees it. Measured shapes:
    //     768x448x121  -> 16 x 24 x 14 =  5376 tokens ->  55 MiB
    //     1280x704x121 -> 16 x 40 x 22 = 14080 tokens -> 378 MiB
    // and ggml_ext_attention_ext's unconditional ggml_cast doubles the live peak.
    // 378 MiB x2 against a ~2.5 GiB compute buffer is a real allocation decision,
    // not a rounding error, so it is the operator's to make.
    inline int64_t self_mask_max_mib() {
        static const int64_t value = [] {
            const char* raw = std::getenv("LTX_RELAY_SELF_MASK_MAX_MIB");
            if (raw == nullptr || raw[0] == '\0') {
                return static_cast<int64_t>(128);
            }
            const long parsed = std::strtol(raw, nullptr, 10);
            return parsed > 0 ? static_cast<int64_t>(parsed) : static_cast<int64_t>(128);
        }();
        return value;
    }

    // LTX_RELAY_SELF_MASK_FOLD -- the SPATIAL FOLD, which is what makes the
    // self-mask affordable at production resolution. Unset / 0 keeps the dense
    // [L, L] expansion above and the graph is unchanged.
    //
    // The bias depends only on (query frame, key frame), so every one of the
    // W_lat*H_lat query tokens inside a frame carries an IDENTICAL mask row.
    // ggml_flash_attn_ext will not broadcast the query axis (ggml.c:5541-5542
    // broadcast ne2/ne3 only), but it WILL broadcast the mask over ne2 and ne3 --
    // and the query axis of Q is ours to choose. Video tokens are frame-major
    // (patchify_video, ltxv.hpp:1790 -- token == frame*W*H + spatial), so
    //     Q [d_head, n_head, tokens_per_frame, frames]      (contiguous)
    // can be permuted to
    //     Q'[d_head, frames*m, tokens_per_frame/m, n_head]
    // for any m dividing tokens_per_frame, at exactly the cost of the ggml_cont
    // the attention wrapper already performs. The mask then only has to be
    // [L_k, frames*m] -- the spatial axis has moved onto the broadcast head axis
    // and off the axis that has to be materialised.
    //
    // m is a PERFORMANCE knob, not a correctness one; every m gives bit-identical
    // mask content. It exists because the CUDA kernel selector tiles the query
    // axis in blocks of 64 (fattn.cu:31-37 picks ncols1 = 64/ncols2, and with an
    // unpadded K -- L_k is never a multiple of FATTN_KQ_STRIDE=256 here -- the GQA
    // path is off, so ncols2 == 1 and ncols1 == 64). Keeping frames*m >= 64 leaves
    // the launch geometry, and therefore the K/V streaming traffic, identical to
    // the unfolded attention: same total blocks, same K reads, but a mask that
    // fits in L2 instead of being streamed from HBM.
    //
    //     1920x1088x121  F=16 tpf=2040  m=4 -> 64 rows ->   3.98 MiB  (was 2.03 GiB)
    //     1280x704x121   F=16 tpf= 880  m=4 -> 64 rows ->   1.72 MiB  (was  378 MiB)
    //      768x448x121   F=16 tpf= 336  m=4 -> 64 rows ->   0.66 MiB  (was   55 MiB)
    //
    // 1 is the value to set: the caller raises it to the smallest divisor of
    // tokens_per_frame that reaches 64 rows.
    inline int64_t self_mask_fold() {
        static const int64_t value = [] {
            const char* raw = std::getenv("LTX_RELAY_SELF_MASK_FOLD");
            if (raw == nullptr || raw[0] == '\0') {
                return static_cast<int64_t>(0);
            }
            const long parsed = std::strtol(raw, nullptr, 10);
            return parsed > 0 ? static_cast<int64_t>(parsed) : static_cast<int64_t>(0);
        }();
        return value;
    }

    // Materialise the COMPACT [F_k, F_q] frame-pair bias in F32. The F16 builder below
    // is a straight cast of this; the SEGMENT splitter needs the exact float values so
    // its "is this column constant" test cannot be fooled by F16 rounding collapsing two
    // genuinely different biases onto the same half.
    inline void build_self_frame_bias_f32(const Plan& plan,
                                          float strength,
                                          std::vector<float>& out) {
        const int64_t frames = static_cast<int64_t>(plan.video_frame_time.size());
        out.assign(static_cast<size_t>(std::max<int64_t>(0, frames * frames)), 0.f);
        if (frames <= 0 || plan.beats.empty() || strength <= 0.f) {
            return;
        }

        std::vector<float> sigmas(plan.beats.size());
        for (size_t beat = 0; beat < plan.beats.size(); ++beat) {
            sigmas[beat] = beat_sigma(plan.beats[beat], plan.eps, plan.sigma_fixed);
        }

        for (int64_t query = 0; query < frames; ++query) {
            const float query_time = plan.video_frame_time[static_cast<size_t>(query)];
            for (int64_t key = 0; key < frames; ++key) {
                const float key_time = plan.video_frame_time[static_cast<size_t>(key)];
                // Past and same-frame keys are never penalised. This is the whole
                // asymmetry: only the future->past direction is a leak.
                if (key_time <= query_time) {
                    continue;
                }
                // The strongest beat boundary lying between the two, rather than a
                // sum: with several beats a sum would stack unrelated penalties on
                // the same frame-0 row until every distant key is equally dead,
                // which is a hard block by another name.
                float cost = 0.f;
                for (size_t beat = 0; beat < plan.beats.size(); ++beat) {
                    const float gate = plan.beats[beat].mid - plan.beats[beat].window;
                    if (!(query_time < gate && gate <= key_time)) {
                        continue;  // no protected window opens between q and k
                    }
                    cost = std::max(cost, beat_cost(plan.beats[beat], sigmas[beat], query_time, plan.max_cost));
                }
                if (cost <= 0.f) {
                    continue;
                }
                out[static_cast<size_t>(query * frames + key)] = -std::min(strength * cost, plan.max_cost);
            }
        }
    }

    // ---------------------------------------------------------------------
    // MASK-FREE formulation: LTX_RELAY_SEGMENT_MERGE.
    // ---------------------------------------------------------------------
    //
    // A mask is what forces the fast kernels to decline. Measured on the deployed
    // service at 1920x1088, no beats, sampling time:
    //     SA3 + cuDNN (prod default)   216.1s
    //     cuDNN only  (SA3 off)        309.3s
    //     any mask -> MMA fallback     535.8s
    // so a mask costs +148%, and teaching cuDNN alone to take an additive bias would
    // only ever reach 309s. The full 216s needs the mask to be GONE, not cheaper.
    //
    // It can be. For a FIXED query, b(q, k) above depends on the key only through
    // which beat gates satisfy gate <= t(k) -- the inner loop's only k-dependence is
    // that one comparison. So b(q, .) is PIECEWISE CONSTANT in the key frame, with
    // breakpoints only at gates. Split the KEY axis at those breakpoints into
    // contiguous segments, run ordinary UNMASKED attention on each, and merge:
    //
    //     O = sum_s w_s O_s,   w_s = exp(LSE_s + b_s - LSE_total)
    //     LSE_total = log sum_s exp(LSE_s + b_s)
    //
    // Exact (softmax over a partitioned key axis is exactly this recombination), at
    // exactly the unmasked FLOP count (the segments partition the keys, they do not
    // duplicate them), with mask == nullptr on every call so cuDNN and SA3 stay
    // eligible. The per-segment LSE comes from ggml_flash_attn_ext_lse.
    //
    // The cuts are derived from the materialised bias matrix rather than from the gate
    // arithmetic, and the constancy is ASSERTED, so this cannot silently drift out of
    // agreement with build_self_frame_bias_f32 if the cost curve is ever changed.

    // Off (unset / 0) leaves the graph exactly as it was. On, and viable, this REPLACES
    // the LTX_RELAY_SELF_MASK dense/folded mask -- which stays as the reference oracle.
    inline bool segment_merge_enabled() {
        static const bool value = [] {
            const char* raw = std::getenv("LTX_RELAY_SEGMENT_MERGE");
            return raw != nullptr && raw[0] == '1';
        }();
        return value;
    }

    // Refusal ceiling on the number of key segments. Every segment is one more
    // full-size attention output held live for the merge (510 MiB F32 at
    // 1920x1088x121), so this is a VRAM decision, not a correctness one. Two beats
    // give three segments plus one for the TASS reference keys.
    inline int64_t segment_merge_max() {
        static const int64_t value = [] {
            const char* raw = std::getenv("LTX_RELAY_SEGMENT_MAX");
            if (raw == nullptr || raw[0] == '\0') {
                return static_cast<int64_t>(6);
            }
            const long parsed = std::strtol(raw, nullptr, 10);
            return parsed > 0 ? static_cast<int64_t>(parsed) : static_cast<int64_t>(6);
        }();
        return value;
    }

    // Split the KEY frame axis wherever the bias changes.
    //
    //   cuts[i] is the first KEY FRAME of segment i; cuts[0] is always 0 and the
    //   segments run to the next cut (the last to F_k).
    //   bias is [n_seg][F_q], SEGMENT-major: bias[s * F_q + q] is the constant additive
    //   bias every key in segment s carries for query frame q.
    //
    // Returns false when the bias is identically zero (nothing to do -- the caller
    // should just run plain attention) or when the split needs more than
    // `max_segments` pieces (the caller falls back to the mask).
    inline bool build_self_segment_bias(const Plan& plan,
                                        float strength,
                                        int64_t max_segments,
                                        std::vector<int64_t>& cuts,
                                        std::vector<float>& bias) {
        cuts.clear();
        bias.clear();

        std::vector<float> dense;
        build_self_frame_bias_f32(plan, strength, dense);
        const int64_t frames = static_cast<int64_t>(plan.video_frame_time.size());
        if (frames <= 0 || dense.empty()) {
            return false;
        }

        auto column_differs = [&](int64_t a, int64_t b) {
            for (int64_t query = 0; query < frames; ++query) {
                if (dense[static_cast<size_t>(query * frames + a)] !=
                    dense[static_cast<size_t>(query * frames + b)]) {
                    return true;
                }
            }
            return false;
        };

        cuts.push_back(0);
        for (int64_t key = 1; key < frames; ++key) {
            if (column_differs(key, key - 1)) {
                cuts.push_back(key);
                if (static_cast<int64_t>(cuts.size()) > max_segments) {
                    cuts.clear();
                    return false;
                }
            }
        }
        if (cuts.size() <= 1) {
            return false;  // bias is uniform over keys => it is a no-op inside softmax
        }

        bias.assign(cuts.size() * static_cast<size_t>(frames), 0.f);
        for (size_t seg = 0; seg < cuts.size(); ++seg) {
            const int64_t key = cuts[seg];
            for (int64_t query = 0; query < frames; ++query) {
                bias[seg * static_cast<size_t>(frames) + static_cast<size_t>(query)] =
                    dense[static_cast<size_t>(query * frames + key)];
            }
        }
        return true;
    }

    // Materialise the COMPACT [F_k, F_q] F16 frame-pair bias, contiguous with
    // ne0 == key frame and ne1 == query frame -- i.e. laid out exactly like the
    // dense mask it expands into, just one entry per frame PAIR instead of per
    // token pair.
    //
    // The penalty depends only on the query's frame and the key's frame, never on
    // spatial position, so the dense mask is this matrix tensored with an
    // all-ones [tokens_per_frame, tokens_per_frame] block. At 1280x704x121 that
    // is 512 bytes of real content standing in for 378 MiB, which is what makes
    // it affordable to upload once and expand on the GPU instead of pushing the
    // dense mask across PCIe for every graph-cut segment.
    inline void build_self_frame_bias_f16(const Plan& plan,
                                          float strength,
                                          std::vector<ggml_fp16_t>& out) {
        // One source of truth for the bias values: the mask path and the mask-free
        // segment path must be comparable to the last bit, and the only way to
        // guarantee that is for the segment splitter and this builder to read the
        // SAME numbers rather than two copies of the same loop.
        std::vector<float> dense;
        build_self_frame_bias_f32(plan, strength, dense);
        out.resize(dense.size());
        for (size_t i = 0; i < dense.size(); ++i) {
            out[i] = ggml_fp32_to_fp16(dense[i]);
        }
    }

}  // namespace ltx_relay
}  // namespace sd

#endif  // __SD_MODEL_DIFFUSION_LTX_RELAY_HPP__
