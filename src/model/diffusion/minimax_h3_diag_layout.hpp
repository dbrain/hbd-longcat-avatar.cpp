#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_LAYOUT_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_LAYOUT_HPP__

#include <string>
#include <vector>

#include "model/diffusion/minimax_h3_diag.hpp"
#include "model/diffusion/minimax_h3_layout.hpp"
#include "model/diffusion/minimax_h3_sched.hpp"

// The layout-aware half of the diagnostic bundle: geometry, the packed row map, and the two
// sigma schedules.  Kept out of minimax_h3_diag.hpp so that header stays includable from
// anywhere, and out of minimax_h3.hpp so the DiT keeps its own diff small.
//
// ★ GEOMETRY IS WHERE A WRONG SHIFT HIDES.  A row-offset bug does not crash and does not NaN --
// it reads the right tensor at the wrong place and renders something plausible.  The only way to
// see one is to write the whole row map down and check it against arithmetic derived a second,
// independent way, which is what emit_geometry() does.
namespace MiniMaxH3Diag {

    struct GeometryFacts {
        // model
        int64_t hidden_size       = 0;
        int64_t num_layers        = 0;
        int64_t video_patch_dim   = 0;
        int64_t audio_latents_dim = 0;
        int64_t latents_dim       = 0;
        int64_t attention_inner_dim = 0;
        int64_t ffn_hidden_size     = 0;
        int patch_t               = 1;
        int patch_h               = 2;
        int patch_w               = 2;
        bool adaln_curve_form     = false;
        int64_t time_embed_dim    = 0;
        bool dit_f16              = false;

        // as the DiT received it
        int64_t orig_latent_w = 0;  // BEFORE pad_to_patch_size
        int64_t orig_latent_h = 0;
        int64_t latent_w      = 0;  // AFTER pad_to_patch_size -- what the layout describes
        int64_t latent_h      = 0;
        int64_t latent_t      = 0;
        int64_t audio_t       = 0;
        int64_t text_len      = 0;

        int vae_scale_factor = 16;

        // Length-triggered forward switches.  These are copied from the actual graph
        // configuration rather than duplicated constants in the diagnostic code.
        int64_t mlp_chunk_rows = 0;
        int64_t graph_capacity = 0;
    };

    inline std::string segment_json(const MiniMaxH3::Segment& seg, int64_t voff, int64_t aoff) {
        // `voff` / `aoff` are the running offsets into the concatenated video / audio embed
        // buffers at the moment this segment is consumed -- exactly the two counters
        // MiniMaxH3Model::forward walks.  For t2va `aoff` is 0 at the target audio segment
        // because there are no reference audio rows before it; a NON-zero value there when no
        // ref2va audio was requested is a bug, and this is where it would be visible.
        return std::string("{\"kind\":") + jstr(MiniMaxH3::segment_kind_name(seg.kind)) +
               ",\"modality\":" + jint(static_cast<int64_t>(MiniMaxH3::segment_modality(seg.kind))) +
               ",\"start\":" + jint(seg.start) +
               ",\"stop\":" + jint(seg.stop) +
               ",\"rows\":" + jint(seg.rows()) +
               ",\"voff\":" + jint(voff) +
               ",\"aoff\":" + jint(aoff) + "}";
    }

    // Writes the "geometry" section and raises a warning for every violated invariant.
    inline void emit_geometry(const MiniMaxH3::PackedLayout& layout, const GeometryFacts& g) {
        if (!on()) {
            return;
        }
        Recorder& r = rec();

        // --- derived arithmetic, computed a SECOND way ----------------------------------------
        // latent_t = 5n + 2 comes from frames = 17n + 5; invert it and re-derive everything the
        // pipeline derived independently, so the two can be compared inside the artifact.
        const int64_t cycles         = layout.seq_len > 0 && g.latent_t >= 2 ? (g.latent_t - 2) / 5 : 0;
        const int64_t frames_aligned = cycles * 17 + 5;
        const int64_t latent_t_check = MiniMaxH3::video_latent_t(static_cast<int>(frames_aligned));
        const int64_t audio_t_check  = MiniMaxH3::audio_latent_t(static_cast<int>(frames_aligned));

        const int64_t grid_h        = g.latent_h / g.patch_h;
        const int64_t grid_w        = g.latent_w / g.patch_w;
        const int64_t rows_per_frame = grid_h * grid_w;

        const bool mlp_chunking_active = g.mlp_chunk_rows > 0 && layout.seq_len > g.mlp_chunk_rows;
        const int64_t mlp_chunks =
            mlp_chunking_active ? (layout.seq_len + g.mlp_chunk_rows - 1) / g.mlp_chunk_rows : 1;
        const int64_t mlp_rows_per_chunk =
            mlp_chunking_active ? (layout.seq_len + mlp_chunks - 1) / mlp_chunks : layout.seq_len;

        // --- the row map ----------------------------------------------------------------------
        std::vector<std::string> segs;
        segs.reserve(layout.segments.size());
        int64_t voff = 0, aoff = 0;
        int64_t row_sum = 0;
        int64_t target_video_rows = 0, target_audio_rows = 0;
        int64_t cond_video_rows = 0, cond_audio_rows = 0, text_rows = 0;
        int64_t target_audio_start = -1, target_audio_stop = -1;
        for (const MiniMaxH3::Segment& seg : layout.segments) {
            segs.push_back(segment_json(seg, voff, aoff));
            row_sum += seg.rows();
            switch (seg.kind) {
                case MiniMaxH3::SegmentKind::Text:
                    text_rows += seg.rows();
                    break;
                case MiniMaxH3::SegmentKind::Video:
                    target_video_rows += seg.rows();
                    voff += seg.rows();
                    break;
                case MiniMaxH3::SegmentKind::Cond:
                case MiniMaxH3::SegmentKind::RefImage:
                    cond_video_rows += seg.rows();
                    voff += seg.rows();
                    break;
                case MiniMaxH3::SegmentKind::Audio:
                    target_audio_rows += seg.rows();
                    target_audio_start = seg.start;
                    target_audio_stop  = seg.stop;
                    aoff += seg.rows();
                    break;
                case MiniMaxH3::SegmentKind::RefAudio:
                    cond_audio_rows += seg.rows();
                    aoff += seg.rows();
                    break;
            }
        }

        // --- invariants -----------------------------------------------------------------------
        std::vector<std::string> inv;
        auto check = [&](const char* name, bool ok, int64_t got, int64_t want) {
            inv.push_back("{\"name\":" + jstr(name) + ",\"ok\":" + jbool(ok) +
                          ",\"got\":" + jint(got) + ",\"expected\":" + jint(want) + "}");
            if (!ok) {
                r.warn(std::string("layout invariant FAILED: ") + name + " got " +
                       std::to_string(got) + ", expected " + std::to_string(want));
            }
        };
        check("segment_rows_sum_equals_seq_len", row_sum == layout.seq_len, row_sum, layout.seq_len);
        check("position_ids_equals_seq_len",
              static_cast<int64_t>(layout.position_ids.size()) == layout.seq_len,
              static_cast<int64_t>(layout.position_ids.size()),
              layout.seq_len);
        check("img_pos_equals_video_family_rows",
              static_cast<int64_t>(layout.img_pos.size()) == target_video_rows + cond_video_rows,
              static_cast<int64_t>(layout.img_pos.size()),
              target_video_rows + cond_video_rows);
        check("audio_pos_equals_audio_family_rows",
              static_cast<int64_t>(layout.audio_pos.size()) == target_audio_rows + cond_audio_rows,
              static_cast<int64_t>(layout.audio_pos.size()),
              target_audio_rows + cond_audio_rows);
        check("target_video_rows_equals_latent_t_times_grid",
              target_video_rows == g.latent_t * rows_per_frame,
              target_video_rows,
              g.latent_t * rows_per_frame);
        // Stereo audio: 2 rows per audio latent frame, CHANNEL-MAJOR.
        check("target_audio_rows_equals_audio_t_times_stereo",
              target_audio_rows == g.audio_t * 2,
              target_audio_rows,
              g.audio_t * 2);
        check("text_rows_equals_text_len", text_rows == g.text_len, text_rows, g.text_len);
        check("latent_t_matches_17n5_grid", latent_t_check == g.latent_t, g.latent_t, latent_t_check);
        check("audio_t_matches_40hz_rule", audio_t_check == g.audio_t, g.audio_t, audio_t_check);
        check("latent_h_multiple_of_patch", g.latent_h % g.patch_h == 0, g.latent_h % g.patch_h, 0);
        check("latent_w_multiple_of_patch", g.latent_w % g.patch_w == 0, g.latent_w % g.patch_w, 0);

        if (g.orig_latent_h != g.latent_h || g.orig_latent_w != g.latent_w) {
            r.warn("the latent was PATCH-PADDED (" + std::to_string(g.orig_latent_w) + "x" +
                   std::to_string(g.orig_latent_h) + " -> " + std::to_string(g.latent_w) + "x" +
                   std::to_string(g.latent_h) + "); the decoded frame carries the pad");
        }

        // --- position-id extents, per modality -------------------------------------------------
        // The rotary grid is area-normalised, so its numbers are NOT a plain arange and cannot be
        // eyeballed from the shape.  Recording the extents makes a wrong grid visible without a
        // render: t must run from text_len upward, h/w must straddle 16 (= 32/2).
        auto axis_extent = [&](int axis, int64_t start, int64_t stop, double& lo, double& hi) {
            lo = 0.0;
            hi = 0.0;
            bool first = true;
            for (int64_t i = start; i < stop && i < static_cast<int64_t>(layout.position_ids.size()); i++) {
                const double v = layout.position_ids[static_cast<size_t>(i)][axis];
                if (first) {
                    lo = hi = v;
                    first   = false;
                } else {
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
            }
        };
        std::vector<std::string> extents;
        for (const MiniMaxH3::Segment& seg : layout.segments) {
            double t0, t1, h0, h1, w0, w1;
            axis_extent(0, seg.start, seg.stop, t0, t1);
            axis_extent(1, seg.start, seg.stop, h0, h1);
            axis_extent(2, seg.start, seg.stop, w0, w1);
            extents.push_back(std::string("{\"kind\":") + jstr(MiniMaxH3::segment_kind_name(seg.kind)) +
                              ",\"start\":" + jint(seg.start) +
                              ",\"t\":[" + jnum(t0) + "," + jnum(t1) + "]" +
                              ",\"h\":[" + jnum(h0) + "," + jnum(h1) + "]" +
                              ",\"w\":[" + jnum(w0) + "," + jnum(w1) + "]}");
        }

        std::string js = "{";
        js += "\"seq_len\":" + jint(layout.seq_len);
        js += ",\"hidden_size\":" + jint(g.hidden_size);
        js += ",\"num_layers\":" + jint(g.num_layers);
        js += ",\"dit_residual_stream\":" + jstr(g.dit_f16 ? "f16" : "f32");
        js += ",\"adaln_form\":" + jstr(g.adaln_curve_form ? "curve-table" : "time-embedder");
        js += ",\"time_embed_dim\":" + jint(g.time_embed_dim);
        js += ",\"text_len\":" + jint(g.text_len);
        js += ",\"latent\":{\"w\":" + jint(g.latent_w) + ",\"h\":" + jint(g.latent_h) +
              ",\"t\":" + jint(g.latent_t) + ",\"c\":" + jint(g.latents_dim) +
              ",\"orig_w\":" + jint(g.orig_latent_w) + ",\"orig_h\":" + jint(g.orig_latent_h) + "}";
        js += ",\"pixels\":{\"w\":" + jint(g.latent_w * g.vae_scale_factor) +
              ",\"h\":" + jint(g.latent_h * g.vae_scale_factor) +
              ",\"vae_scale_factor\":" + jint(g.vae_scale_factor) + "}";
        js += ",\"patch\":{\"t\":" + jint(g.patch_t) + ",\"h\":" + jint(g.patch_h) + ",\"w\":" + jint(g.patch_w) +
              ",\"grid_h\":" + jint(grid_h) + ",\"grid_w\":" + jint(grid_w) +
              ",\"rows_per_frame\":" + jint(rows_per_frame) +
              ",\"video_patch_dim\":" + jint(g.video_patch_dim) + "}";
        js += ",\"audio\":{\"latent_t\":" + jint(g.audio_t) +
              ",\"stereo\":2,\"latents_dim\":" + jint(g.audio_latents_dim) +
              ",\"rows\":" + jint(target_audio_rows) + "}";
        // The three pieces of arithmetic the whole clip length rests on, written out so a reader
        // never has to re-derive them by hand (we already lost a day to exactly that).
        js += ",\"arithmetic\":{";
        js += "\"frames_aligned\":" + jint(frames_aligned);
        js += ",\"frames_rule\":" + jstr("frames = 17n + 5 (align UP: while n % 17 != 5: n++)");
        js += ",\"latent_t_rule\":" + jstr("latent_t = 5n + 2 = ((frames - 5) / 17) * 5 + 2");
        js += ",\"latent_t_from_frames\":" + jint(latent_t_check);
        js += ",\"audio_t_rule\":" + jstr("audio_t = round(frames / 24 * 40)");
        js += ",\"audio_t_from_frames\":" + jint(audio_t_check);
        js += ",\"fps\":" + jint(MiniMaxH3::FPS);
        js += ",\"audio_latent_fps\":" + jint(MiniMaxH3::AUDIO_LATENT_FPS);
        js += "}";
        js += ",\"rows\":{\"text\":" + jint(text_rows) +
              ",\"cond_video\":" + jint(cond_video_rows) +
              ",\"cond_audio\":" + jint(cond_audio_rows) +
              ",\"target_video\":" + jint(target_video_rows) +
              ",\"target_audio\":" + jint(target_audio_rows) +
              ",\"sum\":" + jint(row_sum) + "}";
        // This is the exact structural switch crossed by the two known cells:
        // 7,349 rows is a single MLP span, while 12,445 rows takes the chunked graph.
        // It is deliberately not a warning: chunking is intended to be exact.  The fields make
        // the switch and the audio rows' relationship to every boundary explicit in one artifact.
        const int64_t audio_first_chunk =
            target_audio_start >= 0 ? target_audio_start / std::max<int64_t>(mlp_rows_per_chunk, 1) : -1;
        const int64_t audio_last_chunk =
            target_audio_stop > 0 ? (target_audio_stop - 1) / std::max<int64_t>(mlp_rows_per_chunk, 1) : -1;
        js += ",\"length_switches\":{";
        js += "\"mlp_chunk_limit\":" + jint(g.mlp_chunk_rows);
        js += ",\"mlp_chunking_active\":" + jbool(mlp_chunking_active);
        js += ",\"mlp_chunks\":" + jint(mlp_chunks);
        js += ",\"mlp_rows_per_chunk\":" + jint(mlp_rows_per_chunk);
        js += ",\"audio_first_chunk\":" + jint(audio_first_chunk);
        js += ",\"audio_last_chunk\":" + jint(audio_last_chunk);
        js += ",\"audio_crosses_mlp_boundary\":" + jbool(audio_first_chunk != audio_last_chunk);
        js += ",\"single_span_predicate\":" + jbool(!mlp_chunking_active);
        js += ",\"graph_capacity\":" + jint(g.graph_capacity);
        js += "}";

        // Byte/element maxima for the full-width tensors created by one block.  At the broken
        // cell these prove or disprove the common signed-32-bit-offset hypothesis without a GPU
        // dump.  MLP fc1 uses the actual balanced chunk span when chunking is active.
        const int64_t elem_bytes = g.dit_f16 ? 2 : 4;
        const int64_t residual_elements = g.hidden_size * layout.seq_len;
        const int64_t qkv_elements      = (3 * g.attention_inner_dim) * layout.seq_len;
        const int64_t mlp_fc1_width     = 2 * g.ffn_hidden_size;
        const int64_t mlp_fc1_elements  = mlp_fc1_width * mlp_rows_per_chunk;
        js += ",\"index_bounds\":{";
        js += "\"element_bytes\":" + jint(elem_bytes);
        js += ",\"residual_elements\":" + jint(residual_elements);
        js += ",\"residual_bytes\":" + jint(residual_elements * elem_bytes);
        js += ",\"qkv_elements_upper_bound\":" + jint(qkv_elements);
        js += ",\"qkv_bytes_upper_bound\":" + jint(qkv_elements * elem_bytes);
        js += ",\"mlp_fc1_span_rows\":" + jint(mlp_rows_per_chunk);
        js += ",\"mlp_fc1_elements\":" + jint(mlp_fc1_elements);
        js += ",\"mlp_fc1_bytes\":" + jint(mlp_fc1_elements * elem_bytes);
        js += ",\"all_element_counts_fit_u32\":" +
              jbool(static_cast<uint64_t>(std::max({residual_elements, qkv_elements, mlp_fc1_elements})) <=
                    static_cast<uint64_t>(UINT32_MAX));
        js += "}";
        js += ",\"segments\":" + jarray(segs);
        js += ",\"position_id_extents\":" + jarray(extents);
        js += ",\"invariants\":" + jarray(inv);
        js += "}";
        r.set_section("geometry", js);
    }

    // The FULL video sigma list and the audio sigmas derived from it, plus both candidate audio
    // slopes at every step.
    //
    // ★ We had a real bug here (instantaneous slope where the secant was needed) and the only way
    // it was found was hand-deriving these numbers off a log.  Never again: the artifact carries
    // the whole table, including the slope the run actually USED.
    inline void emit_schedule(const std::vector<float>& sigmas,
                              float shift_v,
                              float shift_a,
                              bool use_secant,
                              float visual_cond_noise_aug,
                              float audio_cond_noise_aug) {
        if (!on()) {
            return;
        }
        std::vector<std::string> steps;
        steps.reserve(sigmas.size());
        for (size_t i = 0; i < sigmas.size(); i++) {
            const float sv      = sigmas[i];
            const bool has_next = i + 1 < sigmas.size();
            const float sv_next = has_next ? sigmas[i + 1] : -1.f;
            const float sa      = MiniMaxH3::time_shift_sigma(std::max(sv, 1e-6f), shift_v, shift_a);
            const float slope_i = MiniMaxH3::time_shift_slope(std::max(sv, 1e-6f), shift_v, shift_a);
            float sa_next       = -1.f;
            float secant        = slope_i;
            bool secant_valid   = false;
            if (has_next) {
                sa_next        = MiniMaxH3::time_shift_sigma(sv_next, shift_v, shift_a);
                const float dv = std::max(sv, 1e-6f) - sv_next;
                if (dv > 1e-9f) {
                    secant       = (sa - sa_next) / dv;
                    secant_valid = std::isfinite(secant);
                }
            }
            const float used = (use_secant && secant_valid) ? secant : slope_i;
            steps.push_back("{\"i\":" + jint(static_cast<int64_t>(i)) +
                            ",\"sigma_v\":" + jnum(sv) +
                            ",\"sigma_v_next\":" + jnum(sv_next) +
                            ",\"sigma_a\":" + jnum(sa) +
                            ",\"sigma_a_next\":" + jnum(sa_next) +
                            ",\"t_v\":" + jnum(1.0 - sv) +
                            ",\"t_a\":" + jnum(1.0 - sa) +
                            ",\"slope_a_instant\":" + jnum(slope_i) +
                            ",\"slope_a_secant\":" + jnum(secant) +
                            ",\"secant_valid\":" + jbool(secant_valid) +
                            ",\"slope_a_used\":" + jnum(used) + "}");
        }
        std::string js = "{";
        js += "\"sigma_shift_video\":" + jnum(shift_v);
        js += ",\"sigma_shift_audio\":" + jnum(shift_a);
        js += ",\"audio_velocity_scale\":" + jstr(use_secant ? "secant" : "instantaneous");
        js += ",\"visual_cond_noise_aug\":" + jnum(visual_cond_noise_aug);
        js += ",\"audio_cond_noise_aug\":" + jnum(audio_cond_noise_aug);
        js += ",\"n_sigmas\":" + jint(static_cast<int64_t>(sigmas.size()));
        js += ",\"steps\":" + jarray(steps);
        js += "}";
        rec().set_section("schedule", js);
    }

    // The AdaLN row assignment the DiT will actually use, for THIS step.  `mod_segments` is what
    // decides which of the (at most four) timestep rows and which of the three modality rows every
    // packed row is modulated by; a mis-assignment here silently modulates the audio rows with the
    // video row's scale/shift and looks like "the audio branch is broken".
    inline std::string timestep_plan_json(const MiniMaxH3::TimestepPlan& plan) {
        std::vector<std::string> uniq;
        uniq.reserve(plan.unique_t.size());
        for (float t : plan.unique_t) {
            uniq.push_back(jnum(t));
        }
        std::vector<std::string> mods;
        mods.reserve(plan.mod_segments.size());
        for (const MiniMaxH3::ModSegment& m : plan.mod_segments) {
            mods.push_back("{\"start\":" + jint(m.start) + ",\"stop\":" + jint(m.stop) +
                           ",\"rows\":" + jint(m.rows()) +
                           ",\"adaln_row\":" + jint(m.row) +
                           ",\"timestep_index\":" + jint(m.row / MiniMaxH3::MODALITY_NUM) +
                           ",\"modality\":" + jint(m.row % MiniMaxH3::MODALITY_NUM) + "}");
        }
        return std::string("{\"sigma_v\":") + jnum(plan.sigma_v) +
               ",\"sigma_a\":" + jnum(plan.sigma_a) +
               ",\"sigma_v_next\":" + jnum(plan.sigma_v_next) +
               ",\"sigma_a_next\":" + jnum(plan.sigma_a_next) +
               ",\"slope_a\":" + jnum(plan.slope_a) +
               ",\"slope_a_instant\":" + jnum(plan.slope_a_instant) +
               ",\"slope_a_secant\":" + jnum(plan.slope_a_secant) +
               ",\"secant_valid\":" + jbool(plan.secant_valid) +
               ",\"unique_t\":" + jarray(uniq) +
               ",\"video_stream\":{\"start\":" + jint(plan.video.start) + ",\"stop\":" + jint(plan.video.stop) +
               ",\"timestep_index\":" + jint(plan.video.timestep_index) + ",\"present\":" + jbool(plan.video.present) + "}" +
               ",\"audio_stream\":{\"start\":" + jint(plan.audio.start) + ",\"stop\":" + jint(plan.audio.stop) +
               ",\"timestep_index\":" + jint(plan.audio.timestep_index) + ",\"present\":" + jbool(plan.audio.present) + "}" +
               ",\"mod_segments\":" + jarray(mods) + "}";
    }

}  // namespace MiniMaxH3Diag

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_LAYOUT_HPP__
