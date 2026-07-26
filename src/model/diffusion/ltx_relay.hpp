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
    inline float beat_sigma(const Beat& beat, float eps) {
        const float clamped_eps = std::min(std::max(eps, 1e-6f), 0.5f);
        const float reach       = std::max(beat.half_span - beat.window, 1e-3f);
        return reach / std::sqrt(2.f * std::log(1.f / clamped_eps));
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

    // Materialise a [L_k, L_q] F16 additive attention bias, contiguous with
    // ne0 == L_k, shaped so that ggml_ext_attention_ext's repeat/cast fallback
    // never fires. `tokens_per_frame` is the number of query tokens that share
    // one entry of `frame_time` (W_lat * H_lat for video, 1 for audio).
    //
    // Rows are identical within a frame, so this fills one row per frame and
    // copies it across the frame's tokens.
    inline void build_mask_f16(const Plan& plan,
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
            sigmas[beat] = beat_sigma(plan.beats[beat], eps);
        }

        const int64_t token_keys = std::min<int64_t>(L_k, static_cast<int64_t>(plan.token_beat.size()));
        std::vector<ggml_fp16_t> row(static_cast<size_t>(L_k), ggml_fp32_to_fp16(0.f));
        std::vector<ggml_fp16_t> beat_bias(plan.beats.size());

        for (int64_t frame = 0; frame < frames; ++frame) {
            const float query_time = frame_time[static_cast<size_t>(frame)];
            for (size_t beat = 0; beat < plan.beats.size(); ++beat) {
                beat_bias[beat] = ggml_fp32_to_fp16(-beat_cost(plan.beats[beat], sigmas[beat], query_time, plan.max_cost));
            }
            for (int64_t key = 0; key < token_keys; ++key) {
                const int32_t beat = plan.token_beat[static_cast<size_t>(key)];
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

}  // namespace ltx_relay
}  // namespace sd

#endif  // __SD_MODEL_DIFFUSION_LTX_RELAY_HPP__
