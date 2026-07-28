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

}  // namespace ltx_relay
}  // namespace sd

#endif  // __SD_MODEL_DIFFUSION_LTX_RELAY_HPP__
