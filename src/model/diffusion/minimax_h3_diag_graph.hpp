#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_GRAPH_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_GRAPH_HPP__

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/ggml_graph_cut.h"
#include "model/diffusion/minimax_h3_diag_layout.hpp"

// The IN-GRAPH half of the diagnostic bundle: AdaLN modulation values and per-block residual
// stream health, split by modality.  Enabled by MINIMAX_H3_DIAG_GRAPH=1 on top of
// MINIMAX_H3_DIAG=<prefix>.
//
// ★ WHY THIS IS OPT-IN AND THE REST IS NOT.  Everything else in the bundle is arithmetic on
// tensors the pipeline already holds in host RAM.  This part adds graph OUTPUTS, and the H3 DiT
// is executed in graph-cut segments -- only cut-marked tensors are materialised per segment, so
// the captures have to be marked into the SAME cut group as the block they belong to.  That is a
// (small) perturbation of how the graph is partitioned, so it does not default on.
//
// What it costs when it is on:
//   * AdaLN: the six [hidden, modalities, M] chunks are already computed and already contiguous.
//     Marking them output only pins them -- 6 x 5376 x 3 x M x 4 B ~= 1.5 MB per block at M = 4,
//     ~77 MB over 50 blocks, read back once per step.  No extra arithmetic at all.
//   * Residual: a STRIDED ROW SUBSAMPLE, MINIMAX_H3_DIAG_ROWS rows per modality per block
//     (default 8).  That reads 8/30000 of the stream, not the stream -- a full-window reduction
//     would be ~64 GB of extra traffic per step and is not worth it to answer "did this block
//     collapse".  It does mean a single non-finite element can be MISSED here; the full-tensor
//     non-finite check lives on the velocity in the sampler, which sees every element.
namespace MiniMaxH3Diag {

    // The six chunks of the block AdaLN projection, in the checkpoint's fused order.
    inline const char* mod_name(size_t k) {
        static const char* const names[6] = {"shift_msa", "scale_msa", "gate_msa",
                                             "shift_mlp", "scale_mlp", "gate_mlp"};
        return k < 6 ? names[k] : "?";
    }

    inline std::string block_cut_group(int64_t block_index) {
        // MUST match MiniMaxH3Model::forward's own mark_graph_cut group for this block.
        return "minimax_h3.blocks." + std::to_string(block_index);
    }

    // How many rows the subsample takes and where from.  Pure arithmetic, shared by the capture
    // (graph side) and the readback (host side) so the JSON can say which rows it measured.
    struct RowSample {
        int64_t start  = 0;
        int64_t stride = 1;
        int64_t count  = 0;
    };

    inline RowSample plan_row_sample(int64_t start, int64_t rows, int rows_wanted) {
        RowSample s;
        if (rows <= 0 || rows_wanted <= 0) {
            return s;
        }
        const int64_t want = std::min<int64_t>(rows_wanted, rows);
        s.start            = start;
        s.stride           = std::max<int64_t>(1, rows / want);
        s.count            = std::min<int64_t>(want, 1 + (rows - 1) / s.stride);
        return s;
    }

    // Called from MiniMaxH3Model::forward, once per block, AFTER the block has produced `h` and
    // BEFORE `h` is itself cut-marked -- so every capture node sits inside this block's node
    // range in graph order, which is what keeps the segment planner's view consistent.
    inline void capture_block(ggml_context* gctx,
                              ggml_cgraph* gf,
                              int64_t block_index,
                              ggml_tensor* h,
                              const std::vector<ggml_tensor*>& mods,
                              const std::vector<ggml_tensor*>& branch,
                              const MiniMaxH3::StreamSegment& video,
                              const MiniMaxH3::StreamSegment& audio,
                              int rows_wanted) {
        if (gctx == nullptr || gf == nullptr || h == nullptr) {
            return;
        }
        const std::string group = block_cut_group(block_index);

        for (size_t k = 0; k < mods.size(); k++) {
            if (mods[k] == nullptr) {
                continue;
            }
            ggml_set_output(mods[k]);
            sd::ggml_graph_cut::mark_graph_cut(mods[k], group, std::string("diag.mod.") + std::to_string(k));
            ggml_build_forward_expand(gf, mods[k]);
        }

        // `src` is a [hidden, seq] tensor in packed order (the residual stream, or one branch's
        // pre-gate output -- all three share the layout, so all three take the same slice).
        auto sample = [&](ggml_tensor* src, const MiniMaxH3::StreamSegment& seg, const std::string& label) {
            if (src == nullptr || !seg.present || src->ne[1] < seg.stop) {
                return;
            }
            const RowSample rs = plan_row_sample(seg.start, seg.rows(), rows_wanted);
            if (rs.count <= 0) {
                return;
            }
            ggml_tensor* view = ggml_view_2d(gctx,
                                             src,
                                             src->ne[0],
                                             rs.count,
                                             src->nb[1] * static_cast<size_t>(rs.stride),
                                             src->nb[1] * static_cast<size_t>(rs.start));
            ggml_tensor* out = ggml_cont(gctx, view);
            if (out->type != GGML_TYPE_F32) {
                // The stream is F16 under MINIMAX_H3_DIT_F16.  Cast AFTER the cont so ggml_cast
                // only ever sees a contiguous source.
                out = ggml_cast(gctx, out, GGML_TYPE_F32);
            }
            ggml_set_output(out);
            sd::ggml_graph_cut::mark_graph_cut(out, group, label);
            ggml_build_forward_expand(gf, out);
        };
        sample(h, video, "diag.res.video");
        sample(h, audio, "diag.res.audio");
        if (branch.size() >= 2) {
            sample(branch[0], video, "diag.attn.video");
            sample(branch[0], audio, "diag.attn.audio");
            sample(branch[1], video, "diag.mlp.video");
            sample(branch[1], audio, "diag.mlp.audio");
        }
    }

    // ---------------------------------------------------------------------------------------
    // Readback
    // ---------------------------------------------------------------------------------------

    // Accumulates one step's captures, then emits them.  Lives on the runner; scan() is driven
    // from GGMLRunner's segment_readback_hook_, which fires once per graph-cut segment on the
    // segmented path and once on the plain path -- so a block's tensors are only ever visible in
    // the call that computed them, and each one has to be picked up when it appears.
    class GraphReadback {
    public:
        void reset() {
            blocks_.clear();
        }

        // Scans one (segment) graph for this render's diag tensors and copies them to host.
        void scan(ggml_cgraph* gf) {
            if (gf == nullptr) {
                return;
            }
            static const std::string prefix = std::string(sd::ggml_graph_cut::GGML_RUNNER_CUT_PREFIX) + "minimax_h3.blocks.";
            const int n = ggml_graph_n_nodes(gf);
            for (int i = 0; i < n; i++) {
                ggml_tensor* t = ggml_graph_node(gf, i);
                if (t == nullptr || t->name[0] == '\0') {
                    continue;
                }
                const std::string name = t->name;
                if (name.compare(0, prefix.size(), prefix) != 0) {
                    continue;
                }
                const size_t bar = name.find('|');
                if (bar == std::string::npos) {
                    continue;
                }
                // ⚠️ EXACT tail match, deliberately. ggml names a view after its parent with a
                // " (view)" / " (reshaped)" suffix, so h3_mod_row's [hidden, 1] slices of a
                // captured AdaLN chunk all carry the cut name too. A prefix match would read one
                // of those instead of the chunk -- same name, different shape, silently wrong
                // statistics.
                const std::string tail = name.substr(bar + 1);
                if (tail.compare(0, 5, "diag.") != 0 || tail.find(' ') != std::string::npos) {
                    continue;
                }
                const int block = atoi(name.c_str() + prefix.size());
                // A view carries its parent's buffer; a cut output is a real node, but be
                // defensive -- an unallocated tensor here means the segment did not compute it.
                ggml_backend_buffer_t buf = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
                if (buf == nullptr) {
                    continue;
                }
                if (t->type != GGML_TYPE_F32) {
                    // Should be unreachable: the AdaLN chunks are F32 by construction (t_emb is)
                    // and the residual subsample is cast on the way out.  Say so rather than
                    // returning a bundle that is quietly missing half its blocks.
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        rec().warn(std::string("diag capture '") + tail + "' is " + ggml_type_name(t->type) +
                                   ", not F32 -- skipped");
                    }
                    continue;
                }
                const int64_t elems = ggml_nelements(t);
                if (elems <= 0) {
                    continue;
                }
                std::vector<float> host(static_cast<size_t>(elems));
                ggml_backend_tensor_get(t, host.data(), 0, ggml_nbytes(t));

                Block& b = blocks_[block];
                if (tail.size() == 10 && tail.compare(0, 9, "diag.mod.") == 0 &&
                    tail[9] >= '0' && tail[9] <= '5') {
                    const size_t k = static_cast<size_t>(tail[9] - '0');
                    {
                        b.mods[k]       = std::move(host);
                        b.mod_hidden    = t->ne[0];
                        b.mod_modality  = t->ne[1];
                        b.mod_timesteps = t->ne[2];
                    }
                } else if (tail == "diag.res.video") {
                    b.res_video      = std::move(host);
                    b.res_video_rows = t->ne[1];
                } else if (tail == "diag.res.audio") {
                    b.res_audio      = std::move(host);
                    b.res_audio_rows = t->ne[1];
                } else if (tail == "diag.attn.video") {
                    b.attn_video = std::move(host);
                } else if (tail == "diag.attn.audio") {
                    b.attn_audio = std::move(host);
                } else if (tail == "diag.mlp.video") {
                    b.mlp_video = std::move(host);
                } else if (tail == "diag.mlp.audio") {
                    b.mlp_audio = std::move(host);
                }
            }
        }

        bool empty() const { return blocks_.empty(); }

        // Emits one "blocks" array element per captured block, plus the automatic anomaly flags.
        void emit(int step, const MiniMaxH3::TimestepPlan& plan) {
            if (!on() || blocks_.empty()) {
                return;
            }
            Recorder& r = rec();
            double prev_video_rms = -1.0;
            double prev_audio_rms = -1.0;
            int prev_block        = -1;

            for (const auto& kv : blocks_) {
                const int block = kv.first;
                const Block& b  = kv.second;

                std::string js = "{\"step\":" + jint(step) + ",\"block\":" + jint(block);

                // ---- residual stream health, per modality ---------------------------------
                Stats vs, as;
                if (!b.res_video.empty()) {
                    vs = stats_of(b.res_video.data(), static_cast<int64_t>(b.res_video.size()));
                    js += ",\"residual_video\":" + jstats(vs) + ",\"residual_video_rows\":" + jint(b.res_video_rows);
                }
                if (!b.res_audio.empty()) {
                    as = stats_of(b.res_audio.data(), static_cast<int64_t>(b.res_audio.size()));
                    js += ",\"residual_audio\":" + jstats(as) + ",\"residual_audio_rows\":" + jint(b.res_audio_rows);
                }
                if (vs.n > 0 && as.n > 0 && vs.rms > 0.0) {
                    js += ",\"residual_audio_over_video_rms\":" + jnum(as.rms / vs.rms);
                }

                // Pre-gate branch outputs.  Together with the gate values in "adaln" below this
                // says WHICH HALF of a block a modality lost: a healthy attn_out with a zero
                // gate_msa is a modulation problem, a dead attn_out is an attention problem.
                auto branch = [&](const char* key, const std::vector<float>& v, const std::vector<float>& a) {
                    if (v.empty() && a.empty()) {
                        return;
                    }
                    js += std::string(",") + jstr(key) + ":{";
                    js += "\"video\":" + (v.empty() ? std::string("null")
                                                    : jstats(stats_of(v.data(), static_cast<int64_t>(v.size()))));
                    js += ",\"audio\":" + (a.empty() ? std::string("null")
                                                     : jstats(stats_of(a.data(), static_cast<int64_t>(a.size()))));
                    js += "}";
                };
                branch("attn_out", b.attn_video, b.attn_audio);
                branch("mlp_out", b.mlp_video, b.mlp_audio);

                // ---- AdaLN modulation ----------------------------------------------------
                if (b.mod_hidden > 0) {
                    js += ",\"adaln\":{";
                    bool first_mod = true;
                    for (size_t k = 0; k < 6; k++) {
                        if (b.mods[k].empty()) {
                            continue;
                        }
                        if (!first_mod) {
                            js += ",";
                        }
                        first_mod = false;
                        js += jstr(mod_name(k)) + ":[";
                        bool first_row = true;
                        for (int64_t m = 0; m < b.mod_timesteps; m++) {
                            for (int64_t tag = 0; tag < b.mod_modality; tag++) {
                                const size_t off = static_cast<size_t>((m * b.mod_modality + tag) * b.mod_hidden);
                                if (off + static_cast<size_t>(b.mod_hidden) > b.mods[k].size()) {
                                    continue;
                                }
                                const Stats s = stats_of(b.mods[k].data() + off, b.mod_hidden);
                                if (!first_row) {
                                    js += ",";
                                }
                                first_row = false;
                                js += "{\"t\":" + jint(m) + ",\"modality\":" + jint(tag) + ",\"s\":" + jstats(s) + "}";
                                check_mod_anomaly(r, block, k, m, tag, s);
                            }
                        }
                        js += "]";
                    }
                    js += "}";

                    // ---- the comparison the audio hunt actually needs -----------------------
                    // Same block, same forward pass: the modulation the VIDEO rows get against
                    // the modulation the AUDIO rows get.  Codex's standing top pick for the audio
                    // defect was "pathological AdaLN modulation values for audio row 2"; this is
                    // the pair of numbers that settles it, and it is now in every bundle.
                    js += ",\"adaln_target_rows\":" + target_rows_json(b, plan);
                }

                js += "}";
                r.append("blocks", js);

                // ---- automatic flags -----------------------------------------------------
                const Thresholds& th = thresholds();
                if (vs.nonfinite > 0) {
                    r.warn("block " + std::to_string(block) + ": NON-FINITE values in the VIDEO residual stream");
                }
                if (as.nonfinite > 0) {
                    r.warn("block " + std::to_string(block) + ": NON-FINITE values in the AUDIO residual stream");
                }
                if (prev_block == block - 1) {
                    if (prev_video_rms > 0.0 && vs.rms > 0.0) {
                        const double ratio = vs.rms / prev_video_rms;
                        if (ratio > th.block_rms_ratio_warn || ratio < 1.0 / th.block_rms_ratio_warn) {
                            r.warn("VIDEO residual RMS jumped " + jnum(ratio, 3) + "x between blocks " +
                                   std::to_string(prev_block) + " and " + std::to_string(block));
                        }
                    }
                    if (prev_audio_rms > 0.0 && as.rms > 0.0) {
                        const double ratio = as.rms / prev_audio_rms;
                        if (ratio > th.block_rms_ratio_warn || ratio < 1.0 / th.block_rms_ratio_warn) {
                            r.warn("AUDIO residual RMS jumped " + jnum(ratio, 3) + "x between blocks " +
                                   std::to_string(prev_block) + " and " + std::to_string(block));
                        }
                    }
                }
                prev_block     = block;
                prev_video_rms = vs.rms;
                prev_audio_rms = as.rms;
            }
            blocks_.clear();
        }

    private:
        struct Block {
            std::vector<float> mods[6];
            int64_t mod_hidden    = 0;
            int64_t mod_modality  = 0;
            int64_t mod_timesteps = 0;
            std::vector<float> res_video;
            std::vector<float> res_audio;
            int64_t res_video_rows = 0;
            int64_t res_audio_rows = 0;
            // Pre-gate branch outputs, same rows as the residual subsample above.
            std::vector<float> attn_video;
            std::vector<float> attn_audio;
            std::vector<float> mlp_video;
            std::vector<float> mlp_audio;
        };

        // The AdaLN row a stream is modulated by is `timestep_index * MODALITY_NUM + tag`, and
        // the chunk's (ne1 = modality, ne2 = timestep) ordering is built so the flat row index and
        // the ggml offset agree.  Pull exactly the two rows the target streams read.
        static std::string target_rows_json(const Block& b, const MiniMaxH3::TimestepPlan& plan) {
            auto row = [&](size_t k, int64_t m, int64_t tag) -> Stats {
                if (b.mods[k].empty() || m < 0 || m >= b.mod_timesteps || tag < 0 || tag >= b.mod_modality) {
                    return Stats{};
                }
                const size_t off = static_cast<size_t>((m * b.mod_modality + tag) * b.mod_hidden);
                if (off + static_cast<size_t>(b.mod_hidden) > b.mods[k].size()) {
                    return Stats{};
                }
                return stats_of(b.mods[k].data() + off, b.mod_hidden);
            };
            std::string js = "{";
            for (size_t k = 0; k < 6; k++) {
                if (k != 0) {
                    js += ",";
                }
                const Stats v = row(k,
                                    plan.video.present ? plan.video.timestep_index : 0,
                                    static_cast<int64_t>(MiniMaxH3::Modality::Video));
                const Stats a = row(k,
                                    plan.audio.present ? plan.audio.timestep_index : 0,
                                    static_cast<int64_t>(MiniMaxH3::Modality::Audio));
                js += jstr(mod_name(k)) + ":{\"video\":" + jstats(v) + ",\"audio\":" + jstats(a) +
                      ",\"audio_over_video_rms\":" + jnum(v.rms > 0.0 ? a.rms / v.rms : NAN) + "}";
            }
            js += "}";
            return js;
        }

        static void check_mod_anomaly(Recorder& r, int block, size_t k, int64_t m, int64_t tag, const Stats& s) {
            const std::string where = "block " + std::to_string(block) + " " + mod_name(k) +
                                      " (t=" + std::to_string(m) + ", modality=" + std::to_string(tag) + ")";
            if (s.nonfinite > 0) {
                r.warn(where + ": NON-FINITE AdaLN modulation");
                return;
            }
            if (s.n == 0) {
                return;
            }
            const bool is_gate  = (k == 2 || k == 5);
            const bool is_scale = (k == 1 || k == 4);
            if (is_gate && s.absmax < 1e-6) {
                r.warn(where + ": gate is IDENTICALLY ZERO -- this modality's branch output is discarded");
            }
            if (is_scale) {
                // The modulation is `h * (1 + scale) + shift`, so it is `1 + scale` that has to
                // be sane, not `scale`.
                const double effective = 1.0 + s.mean;
                if (std::fabs(effective) < 0.01) {
                    r.warn(where + ": effective scale (1 + scale) is " + jnum(effective, 3) +
                           " -- the normalised stream is being annihilated");
                }
                if (std::fabs(effective) > 100.0) {
                    r.warn(where + ": effective scale (1 + scale) is " + jnum(effective, 3));
                }
            }
            if (s.absmax > 1e4) {
                r.warn(where + ": AdaLN |max| = " + jnum(s.absmax, 4));
            }
        }

        std::map<int, Block> blocks_;
    };

}  // namespace MiniMaxH3Diag

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_DIAG_GRAPH_HPP__
