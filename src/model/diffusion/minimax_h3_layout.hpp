#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_LAYOUT_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_LAYOUT_HPP__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// MiniMax-H3 packs text, conditioning, audio and video into ONE flat sequence and runs full
// self-attention over it -- there is no cross-attention anywhere in the model.  This header owns
// the shape of that sequence: row ordering, the (t, h, w) rotary grid, and the per-row bookkeeping
// the DiT needs to scatter embeddings in and pull velocities back out.
//
// It is deliberately free of ggml: the layout is pure integer/double bookkeeping, so it can be
// diffed row-for-row against the Python reference without a backend.  See
// handoffs/longcat-avatar.cpp/minimax-h3/tools/ref_layout.py and tests/minimax_h3_layout_test.cpp.
//
// Ported from ComfyUI PR #15224, comfy/ldm/minimax/model.py (`PackedLayout` and the `_*_grid`
// helpers).  Two things here are easy to get subtly wrong and impossible to notice by eye:
//
//   * the spatial grid is AREA-NORMALISED, not a plain arange -- each axis is
//     linspace((1 - r) / 2, (1 + r) / 2, n, endpoint=False) * 32 with r = dim / sqrt(h * w).
//     It therefore depends on the aspect ratio of the canvas, not just its size.
//   * the temporal spans are NON-UNIFORM -- FRAME_PER_TOKEN cycles (1, 4, 4, 4, 4) every five
//     tokens, scaled by 5/3, and the grid is the exclusive cumulative sum from a per-block origin.
namespace MiniMaxH3 {

    // comfy/ldm/minimax/model.py:29-30
    static const int FRAME_PER_TOKEN[5] = {1, 4, 4, 4, 4};
    static const double FRAME_RESCALE   = 5.0 / 3.0;

    // H3 is a FIXED-24-fps model: the 17n+5 pixel-frame grid, the 4x temporal video VAE and the
    // 40 Hz audio latent rate are all defined AT 24 fps. A request at another fps is not a slower
    // render, it is a different (and wrong) audio length.
    //
    // ★ ONE implementation, deliberately. `examples/server/minimax_h3_wire.h` forwards to these
    // rather than keeping a second copy: the HTTP route QUOTES this geometry in its 202 and the
    // core BUILDS the packed sequence from it. The two disagreeing does not crash -- the layout
    // simply describes a different sequence than the one the VAEs produced.
    static const int FPS              = 24;
    static const int AUDIO_LATENT_FPS = 40;

    // Snap UP to the model's grid: `while n % 17 != 5: n += 1`, floored at the shortest legal
    // clip. Up rather than down is the reference behaviour and the safe direction -- a caller
    // asking for 5 s gets at least 5 s.
    inline int align_frame_count(int frames) {
        int aligned = std::max(5, frames);
        while (aligned % 17 != 5) {
            aligned++;
        }
        return aligned;
    }

    inline bool frame_count_is_aligned(int frames) {
        return frames >= 5 && frames % 17 == 5;
    }

    // Video latent frames for an ALIGNED pixel frame count. NOT `1 + (n-1)/4`: FRAME_PER_TOKEN
    // cycles (1, 4, 4, 4, 4), so 5 latent tokens span exactly 17 pixel frames -- 124 frames is
    // latent_t 37, not 31. An earlier revision of the cost table used ceil(frames/4) and was
    // ~20% low on sequence length.
    inline int video_latent_t(int aligned_frames) {
        return aligned_frames <= 5 ? 2 : ((aligned_frames - 5) / 17) * 5 + 2;
    }

    // Audio latent frames: the clip's duration at the audio VAE's 40 Hz latent rate, rounded.
    inline int audio_latent_t(int aligned_frames) {
        const double duration = static_cast<double>(aligned_frames) / FPS;
        return static_cast<int>(std::lround(duration * AUDIO_LATENT_FPS));
    }

    // Row modality, as consumed by the DiT's AdaLN table (`timestep_index * 3 + tag`).
    // comfy/ldm/minimax/model.py:544
    enum class Modality : int {
        Video = 0,
        Text  = 1,
        Audio = 2,
    };

    enum class SegmentKind {
        Text,      // the Qwen3-VL presentation, including its vision-pad runs
        Cond,      // fl2va keyframe rows, sharing the target spatial grid
        RefImage,  // ref2va image / video reference rows
        RefAudio,  // ref2va audio reference rows
        Audio,     // the target audio rows (always second to last)
        Video,     // the target video rows (always last)
    };

    inline const char* segment_kind_name(SegmentKind kind) {
        switch (kind) {
            case SegmentKind::Text:
                return "text";
            case SegmentKind::Cond:
                return "cond";
            case SegmentKind::RefImage:
                return "ref_img";
            case SegmentKind::RefAudio:
                return "ref_audio";
            case SegmentKind::Audio:
                return "audio";
            case SegmentKind::Video:
                return "video";
        }
        return "?";
    }

    // The packed sequence is uniform per segment in (modality, timestep class), which is what lets
    // the DiT modulate it with contiguous slices instead of a gather.  The one exception is the
    // text span, whose tags are resolved at forward time from the presentation's vision-pad runs.
    inline Modality segment_modality(SegmentKind kind) {
        switch (kind) {
            case SegmentKind::Text:
                return Modality::Text;
            case SegmentKind::Audio:
            case SegmentKind::RefAudio:
                return Modality::Audio;
            case SegmentKind::Video:
            case SegmentKind::Cond:
            case SegmentKind::RefImage:
                return Modality::Video;
        }
        return Modality::Video;
    }

    struct Segment {
        int64_t start    = 0;
        int64_t stop     = 0;  // exclusive
        SegmentKind kind = SegmentKind::Text;

        int64_t rows() const { return stop - start; }
    };

    // fl2va conditioning.  Only the first and last pixel frames are anchorable; anything else is
    // rejected rather than silently placed, matching the reference.
    struct Keyframe {
        int64_t resolved_frame_index = 0;
    };

    enum class RefKind {
        Image,
        Audio,
        Video,       // video rows only
        VideoAudio,  // audio rows packed immediately before the video rows, sharing the origin
    };

    struct Reference {
        RefKind kind        = RefKind::Image;
        int64_t latent_h    = 0;
        int64_t latent_w    = 0;
        int64_t latent_t    = 0;
        int64_t ref_audio_t = 0;
    };

    struct LayoutRequest {
        int64_t text_len = 0;
        int64_t latent_t = 0;
        int64_t latent_h = 0;
        int64_t latent_w = 0;
        int64_t audio_t  = 0;
        // `frame_count` is the PIXEL frame count, needed only to recognise a last-frame keyframe.
        int64_t frame_count = -1;
        std::vector<Keyframe> keyframes;
        std::vector<Reference> refs;
    };

    // ---------------------------------------------------------------------------------------
    // grid helpers
    // ---------------------------------------------------------------------------------------

    // comfy/ldm/minimax/model.py:80-84.  Equivalent to
    //   linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32
    inline std::vector<double> axis_from_sqrt_area(int64_t dim, int64_t patch, double sqrt_area) {
        const double ratio = static_cast<double>(dim) / sqrt_area;
        const int64_t n    = dim / patch;
        std::vector<double> axis(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            axis[static_cast<size_t>(i)] = (static_cast<double>(i) * (ratio / static_cast<double>(n)) + (1.0 - ratio) / 2.0) * 32.0;
        }
        return axis;
    }

    // One latent frame's 2x2-patch rows, in h-major order (numpy meshgrid indexing="ij").
    // Also returns the w axis, whose extremes pin the stereo audio rows.
    inline void frame_grid(int64_t latent_h,
                           int64_t latent_w,
                           std::vector<std::array<double, 2>>& out_rows,
                           std::vector<double>& out_w_axis) {
        const double area        = std::sqrt(static_cast<double>(latent_h) * static_cast<double>(latent_w));
        std::vector<double> ax_h = axis_from_sqrt_area(latent_h, 2, area);
        out_w_axis               = axis_from_sqrt_area(latent_w, 2, area);

        out_rows.clear();
        out_rows.reserve(ax_h.size() * out_w_axis.size());
        for (size_t ih = 0; ih < ax_h.size(); ih++) {
            for (size_t iw = 0; iw < out_w_axis.size(); iw++) {
                out_rows.push_back({ax_h[ih], out_w_axis[iw]});
            }
        }
    }

    inline std::vector<double> video_t_spans(int64_t n) {
        std::vector<double> spans(static_cast<size_t>(n));
        for (int64_t k = 0; k < n; k++) {
            spans[static_cast<size_t>(k)] = FRAME_RESCALE * static_cast<double>(FRAME_PER_TOKEN[k % 5]);
        }
        return spans;
    }

    inline double video_t_span_sum(int64_t n) {
        double total = 0.0;
        for (double span : video_t_spans(n)) {
            total += span;
        }
        return total;
    }

    // origin + exclusive cumulative sum of the spans.  n == 1 yields just {origin}.
    inline std::vector<double> video_t_grid(int64_t n, double origin) {
        std::vector<double> spans = video_t_spans(n);
        std::vector<double> grid(static_cast<size_t>(n));
        double acc = 0.0;
        for (int64_t i = 0; i < n; i++) {
            grid[static_cast<size_t>(i)] = origin + acc;
            acc += spans[static_cast<size_t>(i)];
        }
        return grid;
    }

    // ---------------------------------------------------------------------------------------

    struct PackedLayout {
        int64_t seq_len = 0;

        // (t, h, w) rotary coordinate of every row.  Kept in double because the grid is built from
        // ratios and cumulative sums; it is cast to float only when the rope table is materialised.
        std::vector<std::array<double, 3>> position_ids;

        // Positions of every video-family row (cond / ref_img / video) in packed order, and whether
        // each is a TARGET row (true) or a CONDITIONING row (false).  Same for audio.  These are
        // what let the caller interleave conditioning rows and denoised rows into one embed buffer.
        std::vector<int64_t> img_pos;
        std::vector<uint8_t> img_update;
        std::vector<int64_t> audio_pos;
        std::vector<uint8_t> audio_update;

        std::vector<Segment> segments;

        // (text_len, latent_t, latent_h, latent_w, audio_t) -- the layout is rebuilt when this
        // changes, and reused across every step of a sampling run when it does not.
        std::array<int64_t, 5> signature = {0, 0, 0, 0, 0};

        bool matches(const LayoutRequest& req) const {
            return signature[0] == req.text_len && signature[1] == req.latent_t &&
                   signature[2] == req.latent_h && signature[3] == req.latent_w &&
                   signature[4] == req.audio_t;
        }

        static PackedLayout build(const LayoutRequest& req);
    };

    inline PackedLayout PackedLayout::build(const LayoutRequest& req) {
        PackedLayout layout;

        std::vector<std::array<double, 2>> frame;
        std::vector<double> w_axis;
        frame_grid(req.latent_h, req.latent_w, frame, w_axis);
        const int64_t frame_rows = static_cast<int64_t>(frame.size());

        // Rows are emitted in packed order throughout, so `row` doubles as the running offset and
        // `position_ids.size()` always equals it.
        int64_t row = 0;

        auto push_segment = [&](SegmentKind kind, int64_t n) {
            layout.segments.push_back({row, row + n, kind});
        };

        // --- text ------------------------------------------------------------------------------
        push_segment(SegmentKind::Text, req.text_len);
        for (int64_t i = 0; i < req.text_len; i++) {
            layout.position_ids.push_back({static_cast<double>(i), 0.0, 0.0});
        }
        row += req.text_len;

        // `cursor` is the temporal origin the next block is laid down at.  Without references it
        // stays at text_len, which is where the target audio and video both start.
        double cursor = static_cast<double>(req.text_len);

        // --- fl2va keyframes -------------------------------------------------------------------
        // These sit right after the text and share the TARGET spatial grid.  They do not move the
        // cursor: the target streams still start at text_len.
        for (const Keyframe& kf : req.keyframes) {
            double cond_t;
            if (kf.resolved_frame_index == 0) {
                cond_t = static_cast<double>(req.text_len);
            } else if (req.frame_count > 0 && kf.resolved_frame_index == req.frame_count - 1) {
                cond_t = static_cast<double>(req.text_len) + video_t_span_sum(req.latent_t) - FRAME_RESCALE;
            } else {
                throw std::runtime_error("minimax-h3: only first/last keyframe anchors are supported, got frame index " +
                                         std::to_string(kf.resolved_frame_index));
            }
            push_segment(SegmentKind::Cond, frame_rows);
            for (int64_t j = 0; j < frame_rows; j++) {
                layout.position_ids.push_back({cond_t, frame[static_cast<size_t>(j)][0], frame[static_cast<size_t>(j)][1]});
                layout.img_pos.push_back(row + j);
                layout.img_update.push_back(0);
            }
            row += frame_rows;
        }

        const double target_audio_w_low  = w_axis.front();
        const double target_audio_w_high = w_axis.back();

        // Stereo audio rows are CHANNEL-MAJOR: all of channel 0, then all of channel 1.  `h` stays
        // zero and `w` is pinned to the two extremes of the spatial grid, one per channel.
        // Interleaving these instead would not error -- it would just destroy the stereo field.
        auto emit_audio_rows = [&](SegmentKind kind, double origin, int64_t t, double w_low, double w_high) {
            push_segment(kind, t * 2);
            for (int64_t ch = 0; ch < 2; ch++) {
                const double w = (ch == 0) ? w_low : w_high;
                for (int64_t i = 0; i < t; i++) {
                    layout.position_ids.push_back({origin + static_cast<double>(i), 0.0, w});
                }
            }
            for (int64_t i = 0; i < t * 2; i++) {
                layout.audio_pos.push_back(row + i);
                layout.audio_update.push_back(kind == SegmentKind::Audio ? 1 : 0);
            }
            row += t * 2;
        };

        auto emit_video_rows = [&](SegmentKind kind,
                                   double origin,
                                   int64_t vt,
                                   const std::vector<std::array<double, 2>>& grid) {
            const int64_t rows_per_frame = static_cast<int64_t>(grid.size());
            const int64_t n              = vt * rows_per_frame;
            push_segment(kind, n);
            std::vector<double> t_grid = video_t_grid(vt, origin);
            for (int64_t i = 0; i < vt; i++) {
                for (int64_t j = 0; j < rows_per_frame; j++) {
                    layout.position_ids.push_back({t_grid[static_cast<size_t>(i)],
                                                   grid[static_cast<size_t>(j)][0],
                                                   grid[static_cast<size_t>(j)][1]});
                }
            }
            for (int64_t i = 0; i < n; i++) {
                layout.img_pos.push_back(row + i);
                layout.img_update.push_back(kind == SegmentKind::Video ? 1 : 0);
            }
            row += n;
        };

        // --- ref2va reference blocks -----------------------------------------------------------
        if (!req.refs.empty()) {
            cursor = static_cast<double>(req.text_len);
            for (const Reference& ref : req.refs) {
                if (ref.kind == RefKind::Image) {
                    std::vector<std::array<double, 2>> r_frame;
                    std::vector<double> r_w_axis;
                    frame_grid(ref.latent_h, ref.latent_w, r_frame, r_w_axis);
                    // A still occupies a single temporal slot regardless of its spatial size.
                    push_segment(SegmentKind::RefImage, static_cast<int64_t>(r_frame.size()));
                    for (size_t j = 0; j < r_frame.size(); j++) {
                        layout.position_ids.push_back({cursor, r_frame[j][0], r_frame[j][1]});
                        layout.img_pos.push_back(row + static_cast<int64_t>(j));
                        layout.img_update.push_back(0);
                    }
                    row += static_cast<int64_t>(r_frame.size());
                    cursor += 1.0;
                } else if (ref.kind == RefKind::Audio) {
                    // The `> 0` guard skips the rows but the cursor still advances -- a zero-length
                    // audio reference is a no-op in rows and a no-op in time, and both must agree.
                    if (ref.ref_audio_t > 0) {
                        emit_audio_rows(SegmentKind::RefAudio, cursor, ref.ref_audio_t, target_audio_w_low, target_audio_w_high);
                    }
                    cursor += static_cast<double>(ref.ref_audio_t);
                } else {
                    // Video / video+audio: the block's audio rows pack immediately BEFORE its video
                    // rows, both anchored at the same cursor origin, and the cursor then advances by
                    // whichever stream is longer.
                    std::vector<std::array<double, 2>> r_frame;
                    std::vector<double> r_w_axis;
                    frame_grid(ref.latent_h, ref.latent_w, r_frame, r_w_axis);
                    if (ref.ref_audio_t > 0) {
                        emit_audio_rows(SegmentKind::RefAudio, cursor, ref.ref_audio_t, r_w_axis.front(), r_w_axis.back());
                    }
                    emit_video_rows(SegmentKind::RefImage, cursor, ref.latent_t, r_frame);
                    cursor += std::max(static_cast<double>(ref.ref_audio_t), video_t_span_sum(ref.latent_t));
                }
            }
        }

        // --- the two target streams, always last and always in this order ----------------------
        emit_audio_rows(SegmentKind::Audio, cursor, req.audio_t, target_audio_w_low, target_audio_w_high);
        emit_video_rows(SegmentKind::Video, cursor, req.latent_t, frame);

        layout.seq_len   = row;
        layout.signature = {req.text_len, req.latent_t, req.latent_h, req.latent_w, req.audio_t};
        return layout;
    }

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_LAYOUT_HPP__
