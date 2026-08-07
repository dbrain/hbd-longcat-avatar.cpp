#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_SCHED_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_SCHED_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ggml.h"
#include "model/diffusion/minimax_h3_layout.hpp"

// MiniMax-H3 denoises video and audio on TWO DIFFERENT shifted flow schedules inside ONE forward
// pass.  The sampler only ever sees the VIDEO sigma; everything the audio stream needs is derived
// from it in closed form here.  This header owns that derivation plus the per-row AdaLN bookkeeping
// that follows from it, and is deliberately ggml-free so it can be diffed against the Python.
//
// Ported from ComfyUI PR #15224, comfy/ldm/minimax/model.py:
//   * `time_shift_sigma` / `time_shift_slope`  (lines 35-49)
//   * the `unique_t` / `t_row` / `mod_segments` construction in `_forward` (lines 526-559)
//   * `VISUAL_COND_TIMESTEP` / `AUDIO_COND_TIMESTEP` (lines 31-32)
//
// Two things here are load-bearing and easy to lose:
//
//   * the audio velocity must be scaled by d(sigma_a)/d(sigma_v).  A stock sampler integrates the
//     flat ODE dX/dsigma_v; multiplying the audio branch by the slope is what makes that integral
//     equal the audio stream's true ODE on ITS schedule.  Drop it and the audio simply denoises at
//     the wrong rate -- no crash, no NaN, just mush.
//     The engine uses the SECANT over the sampler step by default, matching the separate audio
//     scheduler formulation in diffusers PR #14355.  The instantaneous derivative remains an
//     explicit compatibility opt-out (`MINIMAX_H3_AUDIO_SECANT=0`).  Both values live on
//     TimestepPlan so diagnostics can report the selected one.
//   * the distinct timesteps are known ANALYTICALLY (there are at most four), so the AdaLN table is
//     a handful of rows and every packed segment is uniform in (timestep, modality).  That is what
//     lets the DiT modulate by contiguous slices instead of a per-row gather.
namespace MiniMaxH3 {

    // comfy/ldm/minimax/model.py:31-32.  Conditioning rows are pinned NEAR t=1 (i.e. sigma ~ 0,
    // "clean"), not AT it, for the visual stream -- 0.999 is the trained value, not a rounding.
    static const float VISUAL_COND_TIMESTEP = 0.999f;
    static const float AUDIO_COND_TIMESTEP  = 1.0f;

    // One row per (distinct timestep, modality) pair of the block AdaLN table.
    // comfy/ldm/minimax/model.py:544 -- `MINIMAX_H3_MODALITY_NUM` in the diffusers port.
    static const int MODALITY_NUM = 3;

    // comfy/ldm/minimax/model.py:35-38.  Invert `sigma = s*b / (1 + (s-1)*b)` back to the unshifted
    // base grid, then re-apply the other stream's shift.
    inline float time_shift_sigma(float sigma, float from_shift, float to_shift) {
        const float base = sigma / (from_shift + sigma * (1.0f - from_shift));
        return to_shift * base / (1.0f + (to_shift - 1.0f) * base);
    }

    // comfy/ldm/minimax/model.py:41-49.  d(sigma_to)/d(sigma_from) at the same base-grid point.
    inline float time_shift_slope(float sigma, float from_shift, float to_shift) {
        const float base = sigma / (from_shift + sigma * (1.0f - from_shift));
        const float num  = 1.0f + (from_shift - 1.0f) * base;
        const float den  = 1.0f + (to_shift - 1.0f) * base;
        return (to_shift * num * num) / (from_shift * den * den);
    }

    // A half-open run of packed rows that all take the SAME AdaLN row.
    // `row` is already `timestep_index * MODALITY_NUM + modality_tag`.
    struct ModSegment {
        int64_t start = 0;
        int64_t stop  = 0;  // exclusive
        int row       = 0;

        int64_t rows() const { return stop - start; }
    };

    // One of the two TARGET streams.  `timestep_index` indexes `unique_t` directly (NOT multiplied
    // by MODALITY_NUM): the final layer's AdaLN has modalities=1, so its table is one row per
    // timestep.  comfy/ldm/minimax/model.py:634-635.
    struct StreamSegment {
        int64_t start      = 0;
        int64_t stop       = 0;  // exclusive
        int timestep_index = 0;
        bool present       = false;

        int64_t rows() const { return stop - start; }
    };

    struct TimestepPlan {
        float sigma_v = 0.f;
        float sigma_a = 0.f;
        // The factor the AUDIO velocity must be multiplied by before it is handed back to a sampler
        // integrating on the VIDEO schedule.  This is the value the graph actually uses; the two
        // candidates it can hold are kept alongside so a caller can log which one won.
        float slope_a = 1.f;

        // d(sigma_a)/d(sigma_v) at this point -- comfy's `time_shift_slope`, the CONTINUOUS form.
        float slope_a_instant = 1.f;
        // (sigma_a - sigma_a_next) / (sigma_v - sigma_v_next) -- the DISCRETE form, which is what
        // makes one Euler step of size dsigma_v move the audio stream by exactly dsigma_a.  Equal
        // to `slope_a_instant` only in the limit of infinitely many steps.  Valid (and used) only
        // when the sampler supplied sigma_v_next; NaN-free by construction, see build().
        float slope_a_secant = 1.f;
        bool secant_valid    = false;
        // The next step's sigmas, as supplied / derived.  < 0 means "the sampler did not say".
        float sigma_v_next = -1.f;
        float sigma_a_next = -1.f;

        // Ascending, deduplicated -- Python's `sorted(set(...))`.  At most four entries:
        // {t_video, t_audio} plus the visual and audio conditioning pins when those rows exist.
        std::vector<float> unique_t;

        // Covers the whole packed sequence, in packed order.
        std::vector<ModSegment> mod_segments;

        StreamSegment video;
        StreamSegment audio;

        int64_t seq_len = 0;
    };

    namespace detail {
        // Python `set` semantics: exact float equality, no tolerance.  These values come out of the
        // same float32 arithmetic on both sides, so an epsilon here would MERGE rows the reference
        // keeps apart (e.g. a visual_cond_noise_aug of 0.999 against a t_v that has just crossed
        // it), which silently reassigns whole segments to the wrong modulation.
        inline int t_row_of(const std::vector<float>& unique_t, float t) {
            for (size_t i = 0; i < unique_t.size(); i++) {
                if (unique_t[i] == t) {
                    return static_cast<int>(i);
                }
            }
            // Unreachable: every value asked for was inserted by build() below.
            GGML_ASSERT(false && "minimax-h3: timestep missing from the unique table");
            return 0;
        }

        inline void insert_unique(std::vector<float>& values, float t) {
            if (std::find(values.begin(), values.end(), t) == values.end()) {
                values.push_back(t);
            }
        }
    }  // namespace detail

    // `sigma_v` is the sampler's video sigma in [0, 1] (i.e. `timestep / 1000` under this engine's
    // DiscreteFlowDenoiser, whose sigma_to_t is `sigma * 1000` -- the same convention comfy's
    // model_base uses).  `text_token_tags`, when non-null, is one modality tag per TEXT row: the
    // Qwen3-VL presentation interleaves vision pads that carry the VIDEO modality, so the text span
    // is not uniform and has to be split into tag runs.
    inline TimestepPlan build_timestep_plan(const PackedLayout& layout,
                                            float sigma_v,
                                            float shift_v,
                                            float shift_a,
                                            float visual_cond_noise_aug,
                                            float audio_cond_noise_aug,
                                            const std::vector<int>* text_token_tags,
                                            float sigma_v_next = -1.f,
                                            bool use_secant    = true) {
        TimestepPlan plan;
        plan.seq_len = layout.seq_len;

        // comfy/ldm/minimax/model.py:529 -- clamped away from zero before anything divides by it.
        plan.sigma_v         = std::max(sigma_v, 1e-6f);
        plan.sigma_a         = time_shift_sigma(plan.sigma_v, shift_v, shift_a);
        plan.slope_a_instant = time_shift_slope(plan.sigma_v, shift_v, shift_a);
        plan.slope_a         = plan.slope_a_instant;

        // ---- discrete (secant) audio slope ---------------------------------------------------
        //
        // The sampler steps EVERYTHING by dsigma_v = sigma_v_next - sigma_v.  comfy scales the
        // audio velocity by the instantaneous d(sigma_a)/d(sigma_v), which is only exact as the
        // step size goes to zero; diffusers PR #14355 (`MiniMaxH3LoopSchedulerStep`, denoise.py)
        // instead runs a SECOND scheduler for audio and steps it by sigma_a - sigma_a_next.  The
        // two agree iff the audio velocity is multiplied by the SECANT
        //
        //     (sigma_a - sigma_a_next) / (sigma_v - sigma_v_next)
        //
        // because then  secant * dsigma_v == sigma_a_next - sigma_a == dsigma_a  exactly.
        //
        // sigma_v_next == 0 on the final step is fine and needs no special case: time_shift_sigma
        // maps 0 -> 0 in closed form (base = 0 / (shift_v + 0) = 0, then shift_a * 0 / 1 = 0), so
        // the secant stays finite.  What IS guarded is a degenerate/backwards schedule entry --
        // sigma_v_next >= sigma_v would divide by ~0 -- in which case the instantaneous slope is
        // kept and `secant_valid` stays false so the caller can say so.
        if (sigma_v_next >= 0.f) {
            plan.sigma_v_next = sigma_v_next;
            plan.sigma_a_next = time_shift_sigma(sigma_v_next, shift_v, shift_a);
            const float dv    = plan.sigma_v - sigma_v_next;
            if (dv > 1e-9f) {
                plan.slope_a_secant = (plan.sigma_a - plan.sigma_a_next) / dv;
                plan.secant_valid   = std::isfinite(plan.slope_a_secant);
            }
        }
        if (use_secant && plan.secant_valid) {
            plan.slope_a = plan.slope_a_secant;
        }

        const float t_v = 1.0f - plan.sigma_v;
        const float t_a = 1.0f - plan.sigma_a;

        bool has_vis_cond = false;
        bool has_aud_cond = false;
        for (const Segment& seg : layout.segments) {
            if (seg.kind == SegmentKind::Cond || seg.kind == SegmentKind::RefImage) {
                has_vis_cond = true;
            } else if (seg.kind == SegmentKind::RefAudio) {
                has_aud_cond = true;
            }
        }

        // comfy/ldm/minimax/model.py:538-540.  Conditioning rows are pinned at the LATER of their
        // own noise-augmentation level and the current t, so a conditioning row never ends up
        // NOISIER than the target it is conditioning.
        const float t_cond_vis = std::max(t_v, visual_cond_noise_aug);
        const float t_cond_aud = std::max(t_a, audio_cond_noise_aug);

        // Insertion order does not matter -- the table is sorted below, exactly as `sorted(set(...))`.
        detail::insert_unique(plan.unique_t, t_v);
        detail::insert_unique(plan.unique_t, t_a);
        if (has_vis_cond) {
            detail::insert_unique(plan.unique_t, t_cond_vis);
        }
        if (has_aud_cond) {
            detail::insert_unique(plan.unique_t, t_cond_aud);
        }
        std::sort(plan.unique_t.begin(), plan.unique_t.end());

        auto kind_timestep = [&](SegmentKind kind) -> float {
            switch (kind) {
                case SegmentKind::Text:
                case SegmentKind::Video:
                    return t_v;
                case SegmentKind::Audio:
                    return t_a;
                case SegmentKind::Cond:
                case SegmentKind::RefImage:
                    return t_cond_vis;
                case SegmentKind::RefAudio:
                    return t_cond_aud;
            }
            return t_v;
        };

        plan.mod_segments.reserve(layout.segments.size() + 4);
        for (const Segment& seg : layout.segments) {
            const int row_base = detail::t_row_of(plan.unique_t, kind_timestep(seg.kind)) * MODALITY_NUM;

            if (seg.kind == SegmentKind::Text && text_token_tags != nullptr && !text_token_tags->empty()) {
                // comfy/ldm/minimax/model.py:550-557.  Split the presentation span into runs of
                // equal tag.  Each comparison is against the run's FIRST element, which is what
                // makes a run a run rather than a pairwise-difference walk.
                const std::vector<int>& tags = *text_token_tags;
                const int64_t n              = seg.rows();
                GGML_ASSERT(static_cast<int64_t>(tags.size()) == n &&
                            "minimax-h3: text_token_tags must have one entry per text row");
                int64_t run_start = 0;
                for (int64_t i = 1; i <= n; i++) {
                    if (i == n || tags[static_cast<size_t>(i)] != tags[static_cast<size_t>(run_start)]) {
                        // clamp(min=0): the reference tags padding rows -1 and clamps so they do not
                        // index backwards.  We never emit padding rows, but keep the clamp so a
                        // caller that does cannot corrupt the table.
                        const int tag = std::max(0, tags[static_cast<size_t>(run_start)]);
                        plan.mod_segments.push_back({seg.start + run_start, seg.start + i, row_base + tag});
                        run_start = i;
                    }
                }
                continue;
            }

            plan.mod_segments.push_back({seg.start, seg.stop, row_base + static_cast<int>(segment_modality(seg.kind))});

            if (seg.kind == SegmentKind::Video) {
                plan.video = {seg.start, seg.stop, detail::t_row_of(plan.unique_t, t_v), true};
            } else if (seg.kind == SegmentKind::Audio) {
                plan.audio = {seg.start, seg.stop, detail::t_row_of(plan.unique_t, t_a), true};
            }
        }

        return plan;
    }

    // ------------------------------------------------------------------------------------------
    // Timestep embedding, CPU side
    // ------------------------------------------------------------------------------------------
    //
    // Both forms below produce inputs for at most FOUR rows, so they run on the CPU in double
    // precision and are uploaded as a tiny tensor.  Doing them in-graph would buy nothing and would
    // tie the port to whichever cos/sin ordering ggml_ext_timestep_embedding happens to use --
    // MiniMax-H3 emits COS BEFORE SIN and consumes t UNSCALED in [0, 1], which is not the common
    // convention in this tree.

    // comfy/ldm/minimax/model.py:127-133.  Returns [freq_dim, M] in ggml order (ne0 = freq_dim).
    inline std::vector<float> timestep_sinusoid(const std::vector<float>& unique_t, int64_t freq_dim) {
        GGML_ASSERT(freq_dim % 2 == 0);
        const int64_t half = freq_dim / 2;
        const size_t m     = unique_t.size();
        std::vector<float> out(static_cast<size_t>(freq_dim) * m, 0.f);
        for (size_t r = 0; r < m; r++) {
            for (int64_t i = 0; i < half; i++) {
                const double freq                                                      = std::exp(-std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half));
                const double arg                                                       = static_cast<double>(unique_t[r]) * freq;
                out[r * static_cast<size_t>(freq_dim) + static_cast<size_t>(i)]        = static_cast<float>(std::cos(arg));
                out[r * static_cast<size_t>(freq_dim) + static_cast<size_t>(half + i)] = static_cast<float>(std::sin(arg));
            }
        }
        return out;
    }

    // comfy/ldm/minimax/model.py:604-607 -- the PRUNED curve-table form.  `t_emb` is a linear blend
    // of two rows of `adaln_t_table [grid, k]`:
    //     pos  = clamp(t, 0, 1) * (grid - 1)
    //     i0   = clamp(floor(pos), max=grid - 2)
    //     out  = lerp(table[i0], table[i0 + 1], pos - i0)
    // The max-clamp on i0 is what keeps t == 1.0 on the LAST interval (frac == 1) instead of reading
    // one row past the end of the table.  There is deliberately no min-clamp: pos >= 0 already.
    inline void curve_table_lookup(const std::vector<float>& unique_t,
                                   int64_t grid,
                                   std::vector<int32_t>& out_i0,
                                   std::vector<int32_t>& out_i1,
                                   std::vector<float>& out_frac) {
        GGML_ASSERT(grid >= 2 && "minimax-h3: adaln_t_table needs at least two rows to interpolate");
        out_i0.clear();
        out_i1.clear();
        out_frac.clear();
        out_i0.reserve(unique_t.size());
        out_i1.reserve(unique_t.size());
        out_frac.reserve(unique_t.size());
        for (float t : unique_t) {
            const double clamped = std::min(1.0, std::max(0.0, static_cast<double>(t)));
            const double pos     = clamped * static_cast<double>(grid - 1);
            int64_t i0           = static_cast<int64_t>(std::floor(pos));
            i0                   = std::min<int64_t>(i0, grid - 2);
            out_i0.push_back(static_cast<int32_t>(i0));
            out_i1.push_back(static_cast<int32_t>(i0 + 1));
            out_frac.push_back(static_cast<float>(pos - static_cast<double>(i0)));
        }
    }

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_SCHED_HPP__
