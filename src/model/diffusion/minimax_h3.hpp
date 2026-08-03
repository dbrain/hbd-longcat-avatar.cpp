#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__

#include <inttypes.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/ggml_graph_cut.h"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/minimax_h3_diag_graph.hpp"
#include "model/diffusion/minimax_h3_layout.hpp"
#include "model/diffusion/minimax_h3_qk_permute.hpp"
#include "model/diffusion/minimax_h3_sched.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// MiniMax-H3 DiT.
//
// ONE packed 1-D sequence -- [text | conditioning | audio | video] -- and FULL SELF-ATTENTION over
// it.  There is no cross-attention anywhere in this model, and batch size is 1 only.  Modality
// behaviour comes from exactly three things: the two input patch projections, a per-row AdaLN
// modality tag, and the two output heads.
//
// Ported from ComfyUI PR #15224 (comfy/ldm/minimax/model.py), cross-checked against diffusers PR
// #14355 (src/diffusers/models/transformers/transformer_minimax_h3.py).  Comfy is the port target:
// it keeps the packed sequence in CONTIGUOUS SEGMENTS and modulates by slices, which maps onto ggml
// views.  diffusers scatters with `index_copy` and per-row index tensors -- do NOT port that shape,
// it is a gather per block per modulation, four times over ~31.5k rows.
//
// ⚠️ inner_dim (heads * head_dim = 7168) is LARGER than hidden_size (5376).  Any code here that
// assumes `inner == hidden` is silently wrong, not loudly wrong.
//
// Layout / RoPE grid / segment tables live in minimax_h3_layout.hpp (verified bit-exact against the
// Python over 16 cases).  The dual-schedule sigma map and the AdaLN row bookkeeping live in
// minimax_h3_sched.hpp.  This file is the graph.
//
// ---------------------------------------------------------------------------------------------
// NVFP4 / F16 residual stream -- same reasoning as krea2.hpp, and it matters MORE here.
//
// Production sequence is 33,051 rows (1280x704, 124 frames).  At that length the per-block
// activation working set in F32 is:
//     residual stream  [5376  x 33051]  =  711 MB
//     fused qkv        [21504 x 33051]  = 2843 MB
//     fused fc1        [28672 x 33051]  = 3791 MB
// The SwiGLU intermediate alone does not fit a 16 GB card in F32, and lowering a VRAM cap does not
// help -- a graph cut splits the BLOCK sequence, but every activation inside a segment still spans
// the full sequence.  Two levers reach these tensors.  MINIMAX_H3_MLP_CHUNK divides the fc1 line
// (and the two SwiGLU temporaries behind it) by the chunk count, exactly, at the cost of one extra
// residual-width concat per block -- see the comment on h3_mlp_chunk_rows().  And halving the
// element width touches all three, which is exactly the situation KREA2_DIT_F16 was built for.
// MINIMAX_H3_DIT_F16 mirrors it verbatim,
// including the two-sided gate: the device must serve an NVFP4 mul_mat with an F16 activation AND an
// F16 destination, and every Linear ON THE STREAM must be NVFP4, because a single non-NVFP4 weight
// is an F16 src1 that ggml_backend_cuda_device_supports_op() REJECTS -- and a rejected DiT Linear is
// not slow, it is silently dropped to the CPU backend.
//
// The block AdaLN projection is deliberately NOT part of that check: it consumes `t_emb` (at most
// four rows, always F32), never the residual stream.  See H3DiTBlock::linears_all_nvfp4().
namespace MiniMaxH3 {

    constexpr int MINIMAX_H3_GRAPH_SIZE = 65536;

    // Value-honouring env read: `=0` must DISABLE.  A presence-only test would ignore anyone
    // bisecting with an explicit 0.
    __STATIC_INLINE__ bool h3_env_flag(const char* name, bool dflt) {
        const char* e = getenv(name);
        if (e == nullptr || e[0] == '\0') {
            return dflt;
        }
        return atoi(e) != 0;
    }

    __STATIC_INLINE__ int64_t h3_env_i64(const char* name, int64_t dflt) {
        const char* e = getenv(name);
        if (e == nullptr || e[0] == '\0') {
            return dflt;
        }
        return atoll(e);
    }

    __STATIC_INLINE__ bool h3_dit_f16_env() {
        static int v = -1;
        if (v < 0) {
            v = h3_env_flag("MINIMAX_H3_DIT_F16", false) ? 1 : 0;
        }
        return v == 1;
    }

    // Device/backend half of the gate.  True only for CUDA + GGML_NVFP4_CUBLASLT=1 + Blackwell; on
    // an sm86 card this is false and the well-tested F32 stream runs unchanged.  A weight adapter
    // that cannot take an F16 activation refuses outright (its delta chain runs against adapter
    // tensors of unknown type, and an F32 adapter weight against an F16 x is the supports_op
    // rejection above).
    __STATIC_INLINE__ bool h3_dit_f16_device_ok(GGMLRunnerContext* ctx) {
        if (ctx == nullptr) {
            return false;
        }
        if (ctx->weight_adapter != nullptr && !ctx->weight_adapter->supports_f16_activation()) {
            return false;
        }
        return ggml_cuda_nvfp4_f16_dst_available(ctx->backend);
    }

    // FNV-1a over eight independent lanes -- see krea2_bulk_hash.  Used to key the RoPE table cache
    // on the packed position grid, which is ~758 KB at production length.
    __STATIC_INLINE__ uint64_t h3_bulk_hash(const void* data, size_t bytes) {
        constexpr int LANES = 8;
        uint64_t h[LANES];
        for (int i = 0; i < LANES; ++i) {
            h[i] = 1469598103934665603ull + static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull;
        }
        const auto* p      = static_cast<const unsigned char*>(data);
        const size_t words = bytes / 8;
        size_t i           = 0;
        for (; i + LANES <= words; i += LANES) {
            for (int l = 0; l < LANES; ++l) {
                uint64_t v;
                std::memcpy(&v, p + (i + static_cast<size_t>(l)) * 8, sizeof(v));
                h[l] = (h[l] ^ v) * 1099511628211ull;
            }
        }
        uint64_t out = 1469598103934665603ull;
        auto mix     = [&out](const void* d, size_t n) {
            const auto* q = static_cast<const unsigned char*>(d);
            for (size_t j = 0; j < n; ++j) {
                out = (out ^ q[j]) * 1099511628211ull;
            }
        };
        mix(&bytes, sizeof(bytes));
        for (int l = 0; l < LANES; ++l) {
            mix(&h[l], sizeof(h[l]));
        }
        mix(p + i * 8, bytes - i * 8);
        // Zero is the "nothing cached yet" sentinel, so never return it.
        return out == 0 ? 1 : out;
    }

    // ---------------------------------------------------------------------------------------
    // Config
    // ---------------------------------------------------------------------------------------

    struct MiniMaxH3Config {
        // Defaults are the reference signature, comfy/ldm/minimax/model.py:413-419.  Everything
        // that CAN be read off a tensor shape is read off a tensor shape in detect_from_weights().
        int64_t hidden_size              = 5376;
        int64_t num_layers               = 50;
        int64_t token_refiner_num_layers = 2;
        int64_t num_attention_heads      = 56;
        int64_t attention_head_dim       = 128;
        int64_t ffn_hidden_size          = 14336;
        int64_t latents_dim              = 24;
        int64_t audio_latents_dim        = 32;
        int64_t text_dim                 = 5120;

        // Full time-embedder form only.
        int64_t timestep_input_dim     = 256;
        int64_t time_embed_hidden_size = 5376;
        // In the full form this is the time embedder's output width (2688).  In the PRUNED
        // curve-table form it is `k`, the rank of the precomputed time-embedding curve basis --
        // comfy's model_detection.py sets both from the same config key.
        int64_t time_embed_dim = 2688;
        // > 0 selects the pruned curve-table form (`adaln_t_table [grid, k]`, no time embedder,
        // no SiLU before the AdaLN projection).  0 selects the full form.
        int64_t adaln_curve_grid = 0;

        int64_t rope_inv_freq_len = 16;

        // patch_size (1, 2, 2).  HARDCODED: comfy's own detector hardcodes it too --
        // model_detection.py derives `latents_dim = video_patch_proj.weight.shape[1] // 4` with the
        // literal comment "patch 1x2x2" -- so there is nothing in the tensor map to read it from.
        int patch_t = 1;
        int patch_h = 2;
        int patch_w = 2;

        // comfy/ldm/minimax/model.py:417-418.  Not derivable from shapes.
        float rope_theta        = 10000.f;
        float norm_eps          = 1e-5f;
        float qk_norm_eps       = 1e-5f;
        float final_norm_eps    = 1e-5f;
        float sigma_shift_video = 12.0f;
        float sigma_shift_audio = 3.0f;

        // The checkpoint is MIXED PRECISION: video_patch_proj, audio_patch_proj, the time embedder
        // and both output heads are float32 islands while the block stack is the model dtype
        // (diffusers `_keep_in_fp32_modules`, transformer_minimax_h3.py:441-449; comfy passes
        // `dtype=torch.float32` explicitly at model.py:435/436/441/284/285).  Declaring those
        // Linears F32 makes the property hold regardless of what a converter decided.  Flip to
        // false if a released GGUF stores them at another type and the loader refuses the
        // conversion -- and say so, because it is a numerical change, not a plumbing one.
        bool force_fp32_islands = true;

        // True when the checkpoint's q/k head channels are already permuted for full-width
        // split-half RoPE -- see minimax_h3_qk_permute.hpp.  Driven ONLY by the presence of the
        // marker tensor the converter stamps in; nothing about a permuted weight is detectable from
        // its shape or its statistics, so this can never be guessed.
        bool qk_rope_permuted = false;

        // True when the converter de-interleaved the fused qkv rows into [q_all; k_all; v_all].
        // The graph does NOT branch on this -- split_qkv only knows the contiguous layout and there
        // is nothing else it could do.  It exists so the loader has a home for the marker tensor and
        // so the log records the file's provenance, which is the only trace of a decision that is
        // otherwise invisible in the weights.  Absence means "unknown", not "interleaved".
        bool qkv_deinterleaved = false;

        int64_t inner_dim() const { return num_attention_heads * attention_head_dim; }
        int64_t video_patch_dim() const { return latents_dim * patch_t * patch_h * patch_w; }
        int64_t patch_volume() const { return patch_t * patch_h * patch_w; }
        bool use_adaln_curves() const { return adaln_curve_grid > 0; }

        // 2 * 3 * rope_inv_freq_len of the head's channels are rotated; the rest pass through
        // untouched.  96 of 128 at the reference sizes.
        int64_t rope_rotary_dim() const { return 2 * 3 * rope_inv_freq_len; }

        // The width apply_rope is actually called on.  On a permuted checkpoint that is the WHOLE
        // head: the pass-through channels sit in the tail pairs, which the widened rotation table
        // leaves alone (identity rotation).  H3Attention keys off `rot_dim == head_dim` to pick the
        // one-call path, so this is the only thing that has to change.
        int64_t rope_apply_dim() const { return qk_rope_permuted ? attention_head_dim : rope_rotary_dim(); }

        // Rotation-table width in (cos, sin) pairs.
        int64_t rope_pe_pairs() const { return rope_apply_dim() / 2; }

        static int64_t count_blocks(const String2TensorStorage& tensor_storage_map,
                                    const std::string& prefix,
                                    const std::string& block_prefix) {
            int64_t count           = 0;
            std::string full_prefix = prefix.empty() ? block_prefix : prefix + "." + block_prefix;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, full_prefix)) {
                    continue;
                }
                std::string tail = name.substr(full_prefix.size());
                size_t dot       = tail.find('.');
                if (dot == std::string::npos) {
                    continue;
                }
                int block_index = std::atoi(tail.substr(0, dot).c_str());
                count           = std::max<int64_t>(count, block_index + 1);
            }
            return count;
        }

        // Transcribed from comfy/model_detection.py's MiniMax-H3 branch, one line for one line.
        // Note the torch/ggml transposition: a torch Linear weight of shape [out, in] arrives here
        // as ne = [in, out], so every `shape[0]` below is `ne[1]` and every `shape[1]` is `ne[0]`.
        static MiniMaxH3Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                   const std::string& prefix) {
            MiniMaxH3Config config;

            auto qualified = [&prefix](const std::string& name) {
                return prefix.empty() ? name : prefix + "." + name;
            };
            auto find = [&](const std::string& name) -> const TensorStorage* {
                auto it = tensor_storage_map.find(qualified(name));
                return it == tensor_storage_map.end() ? nullptr : &it->second;
            };

            config.num_layers               = std::max<int64_t>(1, count_blocks(tensor_storage_map, prefix, "blocks."));
            config.token_refiner_num_layers = count_blocks(tensor_storage_map, prefix, "token_refiner.blocks.");

            // `blocks.` also prefixes `token_refiner.blocks.`, but count_blocks anchors on the
            // start of the name so the refiner's keys never reach the DiT counter.
            if (const TensorStorage* vp = find("video_patch_proj.weight")) {
                if (vp->n_dims == 2) {
                    config.hidden_size = vp->ne[1];
                    config.latents_dim = vp->ne[0] / config.patch_volume();
                }
            }
            if (const TensorStorage* ap = find("audio_patch_proj.weight")) {
                if (ap->n_dims == 2) {
                    config.audio_latents_dim = ap->ne[0];
                }
            }
            if (const TensorStorage* qn = find("blocks.0.attn.q_norm.weight")) {
                if (qn->n_dims == 1) {
                    config.attention_head_dim = qn->ne[0];
                }
            }
            if (const TensorStorage* qkv = find("blocks.0.attn.qkv_proj.weight")) {
                if (qkv->n_dims == 2 && config.attention_head_dim > 0) {
                    config.num_attention_heads = qkv->ne[1] / (3 * config.attention_head_dim);
                }
            }
            if (const TensorStorage* fc1 = find("blocks.0.mlp.fc1.weight")) {
                if (fc1->n_dims == 2) {
                    config.ffn_hidden_size = fc1->ne[1] / 2;
                }
            }
            if (const TensorStorage* cp = find("condition_proj.weight")) {
                if (cp->n_dims == 2) {
                    config.text_dim = cp->ne[0];
                }
            }
            // The two AdaLN forms are told apart by the PRESENCE of the curve table, exactly as
            // upstream does it -- not by a size heuristic.
            if (const TensorStorage* table = find("adaln_t_table")) {
                if (table->n_dims == 2) {
                    config.adaln_curve_grid = table->ne[1];  // torch [grid, k] -> ggml [k, grid]
                    config.time_embed_dim   = table->ne[0];
                }
                // INFO, not DEBUG, and for the same reason the qkv/RoPE markers are: this single
                // tensor's presence swaps the modulation for the WHOLE model onto a different
                // path, and both paths run happily on the other one's file -- a curve checkpoint
                // read as a full one would try to load a [k, N] weight into a [t_dim, N] Linear,
                // but a mis-sized TABLE would just produce wrong modulation, quietly.  Say which
                // form was chosen so the log, not a render, is the record of it.
                LOG_INFO("minimax_h3: AdaLN = PRUNED curve table (rank k=%" PRId64 ", grid=%" PRId64
                         "); the time embedder is replaced by a lerp of two table rows",
                         config.time_embed_dim,
                         config.adaln_curve_grid);
            } else {
                if (const TensorStorage* pin = find("time_embedder.proj_in.weight")) {
                    if (pin->n_dims == 2) {
                        config.timestep_input_dim     = pin->ne[0];
                        config.time_embed_hidden_size = pin->ne[1];
                    }
                }
                if (const TensorStorage* pout = find("time_embedder.proj_out.weight")) {
                    if (pout->n_dims == 2) {
                        config.time_embed_dim = pout->ne[1];
                    }
                }
                LOG_INFO("minimax_h3: AdaLN = full time-embedder form (t_dim=%" PRId64
                         "). This file carries the UNFACTORISED modulation projection "
                         "(~13.0B params, 39%% of the DiT); tools/h3_prune_adaln.py rewrites it "
                         "to the rank-8 curve table.",
                         config.time_embed_dim);
            }
            if (const TensorStorage* inv = find("rope.inv_freq")) {
                if (inv->n_dims == 1) {
                    config.rope_inv_freq_len = inv->ne[0];
                }
            }

            GGML_ASSERT(config.rope_rotary_dim() <= config.attention_head_dim &&
                        "minimax-h3: rotary width exceeds the attention head dim");

            config.qk_rope_permuted = find(qk_permuted_marker_name()) != nullptr;
            if (config.qk_rope_permuted &&
                !qk_permutation_is_applicable(config.attention_head_dim, config.rope_rotary_dim())) {
                // The marker says the weights were rewritten for a geometry this build cannot
                // reproduce.  Running the legacy path on permuted weights is silently wrong, so
                // refuse instead.
                LOG_ERROR("minimax_h3: '%s' is present but head_dim=%" PRId64 " / rot_dim=%" PRId64
                          " admits no q/k permutation",
                          qk_permuted_marker_name(),
                          config.attention_head_dim,
                          config.rope_rotary_dim());
                GGML_ABORT("minimax-h3: inconsistent q/k rope permutation marker");
            }
            if (!config.qk_rope_permuted) {
                LOG_INFO(
                    "minimax_h3: q/k head channels are NOT permuted; RoPE runs the "
                    "slice/rope/concat path. Re-convert this checkpoint to drop a full "
                    "[%" PRId64 " x seq x %" PRId64 "] copy of q AND k per block per step.",
                    config.attention_head_dim,
                    config.num_attention_heads);
            }

            config.qkv_deinterleaved = find(qkv_deinterleaved_marker_name()) != nullptr;
            if (config.qkv_deinterleaved) {
                LOG_INFO("minimax_h3: fused qkv rows were de-interleaved at conversion time");
            } else {
                // Not an error: a checkpoint that was never interleaved carries no marker either.
                // But a RAW MiniMax shard is interleaved, and split_qkv on one is silently garbage,
                // so say so rather than leaving the operator with nothing to check.
                LOG_INFO(
                    "minimax_h3: no '%s' marker -- the fused qkv rows are assumed to be "
                    "[q_all; k_all; v_all] already. If this came from a RAW MiniMax checkpoint it is "
                    "per-head interleaved and the attention is WRONG; re-convert with "
                    "MINIMAX_H3_QKV_DEINTERLEAVE=1.",
                    qkv_deinterleaved_marker_name());
            }

            LOG_DEBUG("minimax_h3: layers=%" PRId64 " refiner=%" PRId64 " hidden=%" PRId64 " heads=%" PRId64
                      " head_dim=%" PRId64 " inner=%" PRId64 " ffn=%" PRId64 " latents=%" PRId64 " audio=%" PRId64
                      " text=%" PRId64 " t_dim=%" PRId64 " adaln=%s rope_freqs=%" PRId64,
                      config.num_layers,
                      config.token_refiner_num_layers,
                      config.hidden_size,
                      config.num_attention_heads,
                      config.attention_head_dim,
                      config.inner_dim(),
                      config.ffn_hidden_size,
                      config.latents_dim,
                      config.audio_latents_dim,
                      config.text_dim,
                      config.time_embed_dim,
                      config.use_adaln_curves() ? "curve-table" : "time-embedder",
                      config.rope_inv_freq_len);
            return config;
        }
    };

    // ---------------------------------------------------------------------------------------
    // Segment-wise modulation helpers
    // ---------------------------------------------------------------------------------------
    //
    // The whole reason the port targets comfy: every packed segment is uniform in (timestep,
    // modality), so a modulation is a slice of the residual stream multiplied by ONE broadcast row
    // of the AdaLN table.  diffusers' `index_select(0, adaln_indices)` would instead materialise a
    // full [hidden x seq] scale AND shift per modulation -- four per block, fifty blocks, 31.5k rows.
    //
    // ⚠️ PERF LEFT ON THE TABLE.  ggml has no scatter, so the per-segment results have to be
    // ggml_concat'ed back into one buffer instead of being written in place the way the reference
    // does (`h[a:b].mul_().add_()`).  That is ONE extra full-sequence write per modulation, i.e.
    // four per block.  A fused "modulate by segment table" CUDA op (segments are at most a handful,
    // and the table is at most 12 rows) would remove all four and is the obvious first win of the
    // Blackwell tuning phase.  Everything below is written so that op can drop straight in.

    // One row of a [hidden, modalities, M] AdaLN chunk, as a [hidden, 1] broadcastable view.
    __STATIC_INLINE__ ggml_tensor* h3_mod_row(ggml_context* ctx, ggml_tensor* table, int row) {
        GGML_ASSERT(ggml_is_contiguous(table));
        const int64_t hidden = table->ne[0];
        const int64_t rows   = table->ne[1] * table->ne[2] * table->ne[3];
        GGML_ASSERT(row >= 0 && static_cast<int64_t>(row) < rows);
        return ggml_view_2d(ctx,
                            table,
                            hidden,
                            1,
                            table->nb[1],
                            static_cast<size_t>(row) * static_cast<size_t>(hidden) * ggml_type_size(table->type));
    }

    // A contiguous [hidden, n] window of the packed sequence.  `x` is contiguous, so slicing along
    // the token axis yields a contiguous block -- no cont, no copy.
    __STATIC_INLINE__ ggml_tensor* h3_seq_window(ggml_context* ctx, ggml_tensor* x, int64_t start, int64_t stop) {
        GGML_ASSERT(ggml_is_contiguous(x));
        GGML_ASSERT(start >= 0 && stop > start && stop <= x->ne[1]);
        return ggml_view_2d(ctx, x, x->ne[0], stop - start, x->nb[1], static_cast<size_t>(start) * x->nb[1]);
    }

    // Concatenate along the token axis, PAIRWISE rather than as a left fold.  Same node count, same
    // result, but a left fold rewrites the accumulated prefix at every step -- n(n+1)/2 rows of
    // traffic instead of n*ceil(log2 n).  At production length one full-sequence copy is ~0.7 GB in
    // F32, and the MLP chunking below concatenates on every block, so the difference is real.
    __STATIC_INLINE__ ggml_tensor* h3_concat_seq(ggml_context* ctx, const std::vector<ggml_tensor*>& parts) {
        GGML_ASSERT(!parts.empty());
        std::vector<ggml_tensor*> level = parts;
        while (level.size() > 1) {
            std::vector<ggml_tensor*> next;
            next.reserve((level.size() + 1) / 2);
            for (size_t i = 0; i < level.size(); i += 2) {
                next.push_back(i + 1 < level.size() ? ggml_concat(ctx, level[i], level[i + 1], 1) : level[i]);
            }
            level = std::move(next);
        }
        return level[0];
    }

    // h[a:b] = h[a:b] * (1 + scale[row]) + shift[row], for every segment.
    // comfy/ldm/minimax/model.py:212-216 (`_mod_scale_shift`).
    __STATIC_INLINE__ ggml_tensor* h3_mod_scale_shift(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* shift,
                                                      ggml_tensor* scale,
                                                      const std::vector<ModSegment>& segments) {
        std::vector<ggml_tensor*> parts;
        parts.reserve(segments.size());
        for (const ModSegment& seg : segments) {
            ggml_tensor* piece = h3_seq_window(ctx, x, seg.start, seg.stop);
            ggml_tensor* s     = h3_mod_row(ctx, scale, seg.row);
            ggml_tensor* h     = h3_mod_row(ctx, shift, seg.row);
            // x * (1 + scale) written as x + x*scale, matching Flux::modulate: it avoids
            // materialising a `1 + scale` tensor and keeps the dst type equal to x's, which is
            // what lets an F16 stream take F32 modulation rows.
            piece = ggml_add(ctx, piece, ggml_mul(ctx, piece, s));
            piece = ggml_add(ctx, piece, h);
            parts.push_back(piece);
        }
        ggml_tensor* out = h3_concat_seq(ctx, parts);
        // The segment table must TILE the sequence exactly -- the reference modulates in place, so a
        // gap there would silently leave rows unmodulated, whereas here it shortens the stream.
        // Assert rather than rely on a downstream shape mismatch to notice.
        GGML_ASSERT(out->ne[1] == x->ne[1] && "minimax-h3: mod segments do not tile the packed sequence");
        return out;
    }

    // x[a:b] += gate[row] * other[a:b], for every segment.
    // comfy/ldm/minimax/model.py:219-223 (`_mod_gate`).
    __STATIC_INLINE__ ggml_tensor* h3_mod_gate(ggml_context* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* other,
                                               ggml_tensor* gate,
                                               const std::vector<ModSegment>& segments) {
        GGML_ASSERT(other->ne[1] == x->ne[1]);
        std::vector<ggml_tensor*> parts;
        parts.reserve(segments.size());
        for (const ModSegment& seg : segments) {
            ggml_tensor* xs = h3_seq_window(ctx, x, seg.start, seg.stop);
            ggml_tensor* os = h3_seq_window(ctx, other, seg.start, seg.stop);
            ggml_tensor* g  = h3_mod_row(ctx, gate, seg.row);
            parts.push_back(ggml_add(ctx, xs, ggml_mul(ctx, os, g)));
        }
        ggml_tensor* out = h3_concat_seq(ctx, parts);
        GGML_ASSERT(out->ne[1] == x->ne[1] && "minimax-h3: mod segments do not tile the packed sequence");
        return out;
    }

    // ---------------------------------------------------------------------------------------
    // Attention
    // ---------------------------------------------------------------------------------------

    // Partial split-half RoPE: rotate the leading `rot_dim` head channels, pass the rest through.
    //
    // MiniMax-H3 rotates 2*3*16 = 96 of 128 channels with the SPLIT-HALF convention taken over the
    // ROTARY SUB-BLOCK: the pair is (c, c + rot_dim/2) for c < rot_dim/2, i.e. (c, c+48), and
    // channels 96..127 are untouched.  Rope::apply_rope's non-interleaved mode pairs
    // (c, c + head_dim/2) = (c, c+64) over the WHOLE head, which is a different pairing -- feeding
    // an UNPERMUTED head to it would rotate the wrong pairs and produce plausible-looking garbage.
    // Hence the slice / rope / concat below.
    //
    // ⚠️ This is the SLOW path, kept only for checkpoints converted before the q/k head-channel
    // permutation existed.  Per call it costs two slice conts, an extra permute cont for the
    // pass-through, and a full [head_dim x L x heads] concat -- ~530M elements of avoidable traffic
    // for q and again for k, in every block of every step.  A checkpoint carrying
    // `rope.qk_permuted` takes `rot_dim == head_dim` below and lands in the one-call fast path
    // instead; see minimax_h3_qk_permute.hpp for why that is an exact identity rather than a
    // trade, and tools/ref_qk_permute.py for the numbers.
    __STATIC_INLINE__ ggml_tensor* h3_rope_partial(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* pe,
                                                   int64_t rot_dim) {
        const int64_t head_dim = x->ne[0];
        const int64_t heads    = x->ne[1];
        const int64_t L        = x->ne[2];
        const int64_t N        = x->ne[3];
        GGML_ASSERT(rot_dim > 0 && rot_dim <= head_dim && rot_dim % 2 == 0);

        if (rot_dim == head_dim) {
            return Rope::apply_rope(ctx, x, pe, /*rope_interleaved=*/false);  // [head_dim, L, heads*N]
        }

        ggml_tensor* x_rot  = ggml_ext_slice(ctx, x, 0, 0, rot_dim);         // [rot_dim, heads, L, N]
        ggml_tensor* x_pass = ggml_ext_slice(ctx, x, 0, rot_dim, head_dim);  // [head_dim-rot, heads, L, N]

        x_rot = Rope::apply_rope(ctx, x_rot, pe, /*rope_interleaved=*/false);  // [rot_dim, L, heads*N]

        // Match apply_rope's output layout before concatenating: it permutes the head axis behind
        // the token axis and folds heads into ne[2].
        x_pass = ggml_ext_cont(ctx, ggml_permute(ctx, x_pass, 0, 2, 1, 3));  // [pass, L, heads, N]
        x_pass = ggml_reshape_3d(ctx, x_pass, head_dim - rot_dim, L, heads * N);

        return ggml_concat(ctx, x_rot, x_pass, 0);  // [head_dim, L, heads*N]
    }

    // Fused-QKV self attention with per-head RMSNorm on q and k.
    // comfy/ldm/minimax/model.py:145-182.  Both projections are bias-free.
    class H3Attention : public GGMLBlock {
    protected:
        int64_t hidden;
        int64_t heads;
        int64_t head_dim;
        int64_t rot_dim;

    public:
        H3Attention(int64_t hidden,
                    int64_t heads,
                    int64_t head_dim,
                    int64_t rot_dim,
                    float qk_norm_eps)
            : hidden(hidden),
              heads(heads),
              head_dim(head_dim),
              rot_dim(rot_dim) {
            const int64_t inner = heads * head_dim;
            // ⚠️ inner != hidden.  qkv_proj is [hidden -> 3*inner] and out_proj is [inner -> hidden].
            blocks["qkv_proj"] = std::make_shared<Linear>(hidden, inner * 3, false);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(head_dim, qk_norm_eps);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(head_dim, qk_norm_eps);
            blocks["out_proj"] = std::make_shared<Linear>(inner, hidden, false);
        }

        // x: [hidden, L, N].  pe == nullptr is the TOKEN REFINER's attention, which has no rotary
        // embedding at all (comfy RefinerBlock; diffusers passes rotary_emb=None).
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            const int64_t L = x->ne[1];
            const int64_t N = x->ne[2];
            GGML_ASSERT(N == 1 && "minimax-h3 is batch-size-1 only");

            // This consumes [q_all; k_all; v_all] along the output axis -- comfy splits with
            // `.split(heads*head_dim, dim=-1)`, contiguously.
            //
            // ⚠️ That is NOT the on-disk layout of a raw MiniMax-H3 shard, and an earlier version of
            // this comment got it wrong.  Raw shards are per-head INTERLEAVED
            // (`[h0: q k v, h1: q k v, ...]`); the reference un-interleaves at LOAD time
            // (`_reorder_grouped_qkv_to_qkv`) and the diffusers converter does it on the way in
            // (`reorder_interleaved_qkv`).  So `[q_all; k_all; v_all]` is an in-memory layout only,
            // and reference SPELLING does not imply it -- the raw file has reference spelling AND
            // raw interleave.  The GGUF converter is what normalises it
            // (MINIMAX_H3_QKV_DEINTERLEAVE, stamping `attn.qkv_deinterleaved`); fed an interleaved
            // file this split produces garbage with no error anywhere.
            ggml_tensor* qkv = qkv_proj->forward(ctx, x);  // [3*inner, L, N]
            auto split       = split_qkv(ctx->ggml_ctx, qkv);
            ggml_tensor* q   = ggml_reshape_4d(ctx->ggml_ctx, split[0], head_dim, heads, L, N);
            ggml_tensor* k   = ggml_reshape_4d(ctx->ggml_ctx, split[1], head_dim, heads, L, N);
            ggml_tensor* v   = ggml_reshape_4d(ctx->ggml_ctx, split[2], head_dim, heads, L, N);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            if (pe != nullptr) {
                q = h3_rope_partial(ctx->ggml_ctx, q, pe, rot_dim);  // [head_dim, L, heads*N]
                k = h3_rope_partial(ctx->ggml_ctx, k, pe, rot_dim);
            } else {
                q = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));
                q = ggml_reshape_3d(ctx->ggml_ctx, q, head_dim, L, heads * N);
                k = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));
                k = ggml_reshape_3d(ctx->ggml_ctx, k, head_dim, L, heads * N);
            }

            // Same Q-must-be-F32 rule as krea2_rope_attention: the native CUDA flash kernels assert
            // Q->type == GGML_TYPE_F32 (fattn-common.cuh), and the kernel selector never inspects
            // Q->type, so an F16 Q would abort mid-render rather than fall back.
            if (q->type != GGML_TYPE_F32) {
                q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_F32);
            }
            const bool kv_prescaled_f16 = k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;

            // NO MASK, ever.  The packed sequence is a single attention document -- we never emit
            // padding rows (the reference only pads to a multiple of 64 for its FlashAttention
            // wrapper and then splits the tail into its own document).  Keeping the mask null is
            // what keeps the unmasked flash fast path available.
            ggml_tensor* out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                      ctx->backend,
                                                      q,
                                                      k,
                                                      v,
                                                      heads,
                                                      /*mask=*/nullptr,
                                                      /*skip_reshape=*/true,
                                                      ctx->flash_attn_enabled,
                                                      /*kv_scale=*/1.0f,
                                                      kv_prescaled_f16,
                                                      /*f16_out_ok=*/x->type == GGML_TYPE_F16);  // [inner, L, N]

            // out_proj takes an F16 activation only when its weight is NVFP4; that is exactly the
            // condition the caller's stream gate already checked, so no re-normalisation here.
            return out_proj->forward(ctx, out);  // [hidden, L, N]
        }

        // See H3DiTBlock::linears_all_nvfp4().
        bool linears_all_nvfp4() {
            for (const char* name : {"qkv_proj", "out_proj"}) {
                auto linear = std::dynamic_pointer_cast<Linear>(blocks[name]);
                if (linear == nullptr || linear->get_weight() == nullptr ||
                    linear->get_weight()->type != GGML_TYPE_NVFP4) {
                    return false;
                }
            }
            return true;
        }
    };

    // ---------------------------------------------------------------------------------------
    // SwiGLU MLP
    // ---------------------------------------------------------------------------------------

    // MINIMAX_H3_MLP_CHUNK -- maximum sequence rows per SwiGLU MLP pass.  0 disables chunking.
    //
    // The MLP is POINTWISE in the sequence axis, so splitting it is exact: no seams, no overlap, no
    // approximation.  What it buys is the peak, which is the largest single allocation in the model:
    // fc1 emits [2 * ffn_hidden_size, seq] = [28672, seq], and ggml_ext_silu_act then holds a
    // [ffn, seq] cont of the gate plus a [ffn, seq] product alongside it.  At 1280x704 / 124 frames
    // (seq = 33,051) that is 1.9 + 0.9 + 0.9 = 3.8 GB in F16, 7.6 GB in F32 -- for ONE block,
    // against a card that is also holding an ~11 GB DiT.  Chunking divides all three by the chunk
    // count (5 at the default, so 0.76 / 1.5 GB).  `mark_graph_cut` does not help here: it splits
    // the BLOCK sequence, and every activation inside one block still spans the whole sequence.
    //
    // What it costs:
    //   * the closing concat, because ggml has no scatter.  h3_concat_seq is a balanced tree, so
    //     that is ~ceil(log2 n) full-width [hidden, seq] passes -- at the default and production
    //     length ~2.6x 0.71 GB F32 per block.  Real, but it buys a 5x cut in a 7.6 GB peak;
    //   * graph nodes: ~7 per chunk instead of ~4 for the whole MLP.  At the default and production
    //     length that is ~39 nodes/block, ~2k over 50 blocks, against MINIMAX_H3_GRAPH_SIZE = 65536.
    //     A small chunk at a long duration would blow that (512 rows at 15 s is ~187 chunks,
    //     ~1.3k nodes/block, ~65k for the MLPs alone), so MINIMAX_H3_MLP_MAX_CHUNKS caps the COUNT
    //     rather than only the rows: the node cost is then bounded no matter what the env says and
    //     no matter how long the clip is, and MINIMAX_H3_GRAPH_SIZE does not have to move;
    //   * shorter GEMMs.  8192 rows is still far past the point where the fc1/fc2 matmuls stop
    //     caring, so the default costs nothing measurable, but 512 would.
    //
    // The default is deliberately above a short preview (a 6,311-row sequence bypasses entirely),
    // and a 832x480/124f preview at 14,921 rows splits into just two.  Only production lengths pay
    // more than one concat.
    constexpr int64_t MINIMAX_H3_MLP_CHUNK_DEFAULT = 8192;
    constexpr int64_t MINIMAX_H3_MLP_CHUNK_MIN     = 512;
    constexpr int64_t MINIMAX_H3_MLP_MAX_CHUNKS    = 64;

    __STATIC_INLINE__ int64_t h3_mlp_chunk_rows() {
        static const int64_t v = []() -> int64_t {
            const int64_t e = h3_env_i64("MINIMAX_H3_MLP_CHUNK", MINIMAX_H3_MLP_CHUNK_DEFAULT);
            if (e <= 0) {
                return 0;
            }
            return std::max<int64_t>(e, MINIMAX_H3_MLP_CHUNK_MIN);
        }();
        return v;
    }

    // comfy/ldm/minimax/model.py:185-192.  fc1 emits a FUSED [gate; up] -- gate FIRST, which is the
    // opposite of diffusers' SwiGLU (the converter swaps the halves; we consume the reference
    // spelling, so no swap here).  Both projections are bias-free.
    class H3MLP : public UnaryBlock {
    public:
        H3MLP(int64_t hidden, int64_t ffn) {
            blocks["fc1"] = std::make_shared<Linear>(hidden, ffn * 2, false);
            blocks["fc2"] = std::make_shared<Linear>(ffn, hidden, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            const int64_t seq   = x->ne[1];
            const int64_t chunk = h3_mlp_chunk_rows();
            // h3_seq_window is a 2-D view, so anything with a real batch or a strided input goes
            // down the single-pass path.  N is 1 in this model, so that is never the hot case.
            if (chunk <= 0 || seq <= chunk || x->ne[2] != 1 || x->ne[3] != 1 || !ggml_is_contiguous(x)) {
                return forward_span(ctx, x);
            }

            // Balance the pieces rather than leaving a ragged tail: ceil-divide twice.  The count is
            // capped first so the node budget cannot depend on the sequence length.
            const int64_t n_chunks = std::min(MINIMAX_H3_MLP_MAX_CHUNKS, (seq + chunk - 1) / chunk);
            const int64_t rows     = (seq + n_chunks - 1) / n_chunks;

            std::vector<ggml_tensor*> parts;
            parts.reserve(static_cast<size_t>(n_chunks));
            for (int64_t start = 0; start < seq; start += rows) {
                const int64_t stop = std::min(seq, start + rows);
                parts.push_back(forward_span(ctx, h3_seq_window(ctx->ggml_ctx, x, start, stop)));
            }
            ggml_tensor* out = h3_concat_seq(ctx->ggml_ctx, parts);
            GGML_ASSERT(out->ne[1] == seq);
            return out;
        }

        bool linears_all_nvfp4() {
            for (const char* name : {"fc1", "fc2"}) {
                auto linear = std::dynamic_pointer_cast<Linear>(blocks[name]);
                if (linear == nullptr || linear->get_weight() == nullptr ||
                    linear->get_weight()->type != GGML_TYPE_NVFP4) {
                    return false;
                }
            }
            return true;
        }

    private:
        // One contiguous span of the sequence, start to finish.
        ggml_tensor* forward_span(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
            auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

            x = fc1->forward(ctx, x);
            // gate_first=true: chunk 0 is the SiLU'd gate, chunk 1 is the multiplicand, and the
            // helper emits `mul(up, silu(gate))` with the SiLU as src[1] -- the operand order that
            // lets ggml fuse {UNARY(SILU), MUL} into ggml_cuda_op_unary_mul instead of paying a
            // separate full pass over [ffn x seq].  See the KreaSwiGLU comment for why the order is
            // load-bearing rather than cosmetic.
            x = ggml_ext_silu_act(ctx->ggml_ctx, x, /*gate_first=*/true);
            return fc2->forward(ctx, x);
        }
    };

    // ---------------------------------------------------------------------------------------
    // AdaLN projection
    // ---------------------------------------------------------------------------------------

    // comfy/ldm/minimax/model.py:195-209.
    //
    // [M, t_dim] -> `expand` tensors of [hidden, modalities, M].  The row layout is
    // [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...], which is what `timestep_index * 3 + tag` addresses;
    // in ggml that is exactly the (ne1 = modality, ne2 = timestep) ordering below, so the flat row
    // index and the ggml offset agree by construction.
    //
    // `apply_silu` is FALSE in the pruned curve-table form: there the projection consumes
    // interpolated coordinates of a precomputed time-embedding curve, not a raw timestep embedding,
    // and the activation has already been baked into the basis.
    class H3AdalnProj : public GGMLBlock {
    protected:
        int64_t t_dim;
        int64_t hidden;
        int expand;
        int modalities;
        bool apply_silu;

    public:
        H3AdalnProj(int64_t t_dim, int64_t hidden, int expand, int modalities, bool apply_silu)
            : t_dim(t_dim),
              hidden(hidden),
              expand(expand),
              modalities(modalities),
              apply_silu(apply_silu) {
            blocks["linear"] = std::make_shared<Linear>(t_dim, expand * hidden * modalities, true);
        }

        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* t_emb) {
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            ggml_tensor* x = t_emb;
            if (apply_silu) {
                // The SiLU runs at t_emb's OWN precision (F32) and only its result is handed to the
                // projection.  diffusers documents why this must not be "simplified" by rounding
                // temb first: every block reads the same t_emb, so a rounding applied before the
                // activation biases every block's modulation identically at every sampling step and
                // accumulates coherently along the denoising trajectory.
                x = ggml_silu(ctx->ggml_ctx, x);
            }
            x = linear->forward(ctx, x);  // [expand*hidden*modalities, M]

            const int64_t m = x->ne[1];
            x               = ggml_reshape_3d(ctx->ggml_ctx, x, expand * hidden, modalities, m);
            // cont=true: the chunks are consumed as broadcast rows, and a strided [hidden, ...]
            // view of a [expand*hidden, ...] tensor is not contiguous.  These are tiny --
            // 6 * 5376 * 3 * M floats, well under a megabyte at M <= 4.
            return ggml_ext_chunk(ctx->ggml_ctx, x, expand, 0, /*cont=*/true);
        }
    };

    // ---------------------------------------------------------------------------------------
    // Token refiner (text stream)
    // ---------------------------------------------------------------------------------------

    // comfy/ldm/minimax/model.py:226-237.  Plain pre-norm block: no AdaLN, no rotary embedding.
    class H3RefinerBlock : public UnaryBlock {
    public:
        H3RefinerBlock(int64_t hidden, int64_t heads, int64_t head_dim, int64_t ffn, float norm_eps, float qk_norm_eps) {
            blocks["norm1"] = std::make_shared<RMSNorm>(hidden, norm_eps);
            blocks["norm2"] = std::make_shared<RMSNorm>(hidden, norm_eps);
            blocks["attn"]  = std::make_shared<H3Attention>(hidden, heads, head_dim, /*rot_dim=*/0, qk_norm_eps);
            blocks["mlp"]   = std::make_shared<H3MLP>(hidden, ffn);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<H3Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<H3MLP>(blocks["mlp"]);

            x = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, norm1->forward(ctx, x), /*pe=*/nullptr));
            x = ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, norm2->forward(ctx, x)));
            return x;
        }
    };

    // comfy/ldm/minimax/model.py:240-252.
    class H3TokenRefiner : public UnaryBlock {
    protected:
        int64_t num_layers;

    public:
        H3TokenRefiner(int64_t num_layers, int64_t hidden, int64_t heads, int64_t head_dim, int64_t ffn, float norm_eps, float qk_norm_eps, float final_norm_eps)
            : num_layers(num_layers) {
            for (int64_t i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] =
                    std::make_shared<H3RefinerBlock>(hidden, heads, head_dim, ffn, norm_eps, qk_norm_eps);
            }
            blocks["final_norm"] = std::make_shared<RMSNorm>(hidden, final_norm_eps);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            for (int64_t i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<H3RefinerBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
            }
            auto final_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm"]);
            return final_norm->forward(ctx, x);
        }
    };

    // ---------------------------------------------------------------------------------------
    // DiT block
    // ---------------------------------------------------------------------------------------

    // comfy/ldm/minimax/model.py:255-272.
    class H3DiTBlock : public GGMLBlock {
    public:
        H3DiTBlock(const MiniMaxH3Config& config, bool apply_silu) {
            blocks["norm1"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]  = std::make_shared<H3Attention>(config.hidden_size,
                                                            config.num_attention_heads,
                                                            config.attention_head_dim,
                                                            config.rope_apply_dim(),
                                                            config.qk_norm_eps);
            blocks["mlp"]   = std::make_shared<H3MLP>(config.hidden_size, config.ffn_hidden_size);
            blocks["adaln_proj"] =
                std::make_shared<H3AdalnProj>(config.time_embed_dim, config.hidden_size, 6, MODALITY_NUM, apply_silu);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* t_emb,
                             const std::vector<ModSegment>& mod_segments,
                             ggml_tensor* pe,
                             // Diagnostics only.  When non-null, receives the six AdaLN chunks
                             // this block just produced so the caller can mark them as graph
                             // outputs.  Observation only: the tensors are the ones the block
                             // already uses, not copies, so nothing about the numerics changes
                             // and nothing extra is computed.
                             std::vector<ggml_tensor*>* out_mods = nullptr,
                             // Diagnostics only.  Receives {attention output, MLP output} BEFORE
                             // the gate, which is what separates "attention lost this modality"
                             // from "the MLP did" without a second render.
                             std::vector<ggml_tensor*>* out_branch = nullptr) {
            auto norm1      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn       = std::dynamic_pointer_cast<H3Attention>(blocks["attn"]);
            auto mlp        = std::dynamic_pointer_cast<H3MLP>(blocks["mlp"]);
            auto adaln_proj = std::dynamic_pointer_cast<H3AdalnProj>(blocks["adaln_proj"]);

            // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp -- diffusers' order,
            // which is the order the checkpoint's fused projection stores them in.
            auto mods = adaln_proj->forward(ctx, t_emb);
            GGML_ASSERT(mods.size() == 6);
            if (out_mods != nullptr) {
                *out_mods = mods;
            }

            ggml_tensor* h        = h3_mod_scale_shift(ctx->ggml_ctx, norm1->forward(ctx, x), mods[0], mods[1], mod_segments);
            ggml_tensor* attn_out = attn->forward(ctx, h, pe);
            x                     = h3_mod_gate(ctx->ggml_ctx, x, attn_out, mods[2], mod_segments);

            h                    = h3_mod_scale_shift(ctx->ggml_ctx, norm2->forward(ctx, x), mods[3], mods[4], mod_segments);
            ggml_tensor* mlp_out = mlp->forward(ctx, h);
            x                    = h3_mod_gate(ctx->ggml_ctx, x, mlp_out, mods[5], mod_segments);
            if (out_branch != nullptr) {
                out_branch->push_back(attn_out);
                out_branch->push_back(mlp_out);
            }
            return x;
        }

        // True iff every Linear ON THE RESIDUAL STREAM carries an NVFP4 weight.
        //
        // `adaln_proj.linear` is deliberately NOT asked about: it consumes `t_emb` (at most four
        // rows, always F32) and never the stream, so its type has no bearing on whether an F16
        // activation is servable.  In the pruned curve-table form the reference even stores it at
        // float32 on purpose.  norm1/norm2 are F32 parameter tensors consumed by rms_norm, which
        // takes an F16 activation against an F32 operand.
        bool linears_all_nvfp4() {
            auto attn = std::dynamic_pointer_cast<H3Attention>(blocks["attn"]);
            auto mlp  = std::dynamic_pointer_cast<H3MLP>(blocks["mlp"]);
            return attn != nullptr && mlp != nullptr && attn->linears_all_nvfp4() && mlp->linears_all_nvfp4();
        }
    };

    // ---------------------------------------------------------------------------------------
    // Final layer
    // ---------------------------------------------------------------------------------------

    // comfy/ldm/minimax/model.py:275-294.
    //
    // One shared RMSNorm, an AdaLN with expand=2 / modalities=1 (shift then scale -- the order LTX2
    // and Wan also use in their output layers), and then TWO heads applied to DIFFERENT ROW RANGES.
    // Comfy slices first and runs each head on its own stream; diffusers runs both heads over every
    // row and index_selects afterwards.  Comfy's is a strict win here -- at production length the
    // video head over the text/reference rows would be pure waste.
    class H3FinalLayer : public GGMLBlock {
    public:
        H3FinalLayer(const MiniMaxH3Config& config, bool apply_silu) {
            blocks["norm"]       = std::make_shared<RMSNorm>(config.hidden_size, config.final_norm_eps);
            blocks["adaln_proj"] = std::make_shared<H3AdalnProj>(config.time_embed_dim, config.hidden_size, 2, 1, apply_silu);
            // fp32 islands -- see MiniMaxH3Config::force_fp32_islands.
            blocks["video_out"] = std::make_shared<Linear>(config.hidden_size, config.video_patch_dim(), true, config.force_fp32_islands);
            blocks["audio_out"] = std::make_shared<Linear>(config.hidden_size, config.audio_latents_dim, true, config.force_fp32_islands);
        }

        // Returns {video_rows, audio_rows}; either may be null when its stream is absent.
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* t_emb,
                                                      const StreamSegment& video,
                                                      const StreamSegment& audio) {
            auto norm       = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto adaln_proj = std::dynamic_pointer_cast<H3AdalnProj>(blocks["adaln_proj"]);
            auto video_out  = std::dynamic_pointer_cast<Linear>(blocks["video_out"]);
            auto audio_out  = std::dynamic_pointer_cast<Linear>(blocks["audio_out"]);

            auto mods = adaln_proj->forward(ctx, t_emb);
            GGML_ASSERT(mods.size() == 2);
            ggml_tensor* shift = mods[0];
            ggml_tensor* scale = mods[1];

            auto head = [&](const StreamSegment& seg, const std::shared_ptr<Linear>& proj) -> ggml_tensor* {
                if (!seg.present) {
                    return nullptr;
                }
                // Slice FIRST.  Everything after this point is row-wise, so the norm, the
                // modulation, the F32 cast and the head all run on the target rows only.
                ggml_tensor* rows = ggml_ext_slice(ctx->ggml_ctx, x, 1, seg.start, seg.stop);
                rows              = norm->forward(ctx, rows);
                // The AdaLN table here has modalities=1, so the row index is the plain timestep
                // index -- NOT timestep*3 + tag.
                ggml_tensor* s = h3_mod_row(ctx->ggml_ctx, scale, seg.timestep_index);
                ggml_tensor* h = h3_mod_row(ctx->ggml_ctx, shift, seg.timestep_index);
                rows           = ggml_add(ctx->ggml_ctx, rows, ggml_mul(ctx->ggml_ctx, rows, s));
                rows           = ggml_add(ctx->ggml_ctx, rows, h);
                // Leave whatever the block stream was: the heads are the checkpoint's fp32 island
                // and an F16 activation against an F32 weight is a supports_op rejection, i.e. the
                // head would silently move to the CPU backend.
                if (rows->type != GGML_TYPE_F32) {
                    rows = ggml_cast(ctx->ggml_ctx, rows, GGML_TYPE_F32);
                }
                return proj->forward(ctx, rows);
            };

            return {head(video, video_out), head(audio, audio_out)};
        }
    };

    // ---------------------------------------------------------------------------------------
    // Time embedder (full form only)
    // ---------------------------------------------------------------------------------------

    // comfy/ldm/minimax/model.py:120-133.  The sinusoid itself is built on the CPU (see
    // MiniMaxH3::timestep_sinusoid) because it is at most four rows and its convention -- COS
    // BEFORE SIN, t consumed UNSCALED in [0, 1] -- is not this tree's default.
    class H3TimeEmbedder : public UnaryBlock {
    public:
        H3TimeEmbedder(int64_t freq_dim, int64_t hidden, int64_t out, bool force_f32) {
            blocks["proj_in"]  = std::make_shared<Linear>(freq_dim, hidden, true, force_f32);
            blocks["proj_out"] = std::make_shared<Linear>(hidden, out, true, force_f32);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            x             = proj_in->forward(ctx, x);
            x             = ggml_silu(ctx->ggml_ctx, x);
            return proj_out->forward(ctx, x);
        }
    };

    // ---------------------------------------------------------------------------------------
    // The model
    // ---------------------------------------------------------------------------------------

    class MiniMaxH3Model : public GGMLBlock {
    protected:
        MiniMaxH3Config config;

        // The pruned curve-table form ships `adaln_t_table [grid, k]` (ggml ne = [k, grid]) INSTEAD
        // of a time embedder, and `rope.inv_freq` is a plain buffer in both forms.  Both are
        // declared so the loader finds a home for every tensor in the file -- a tensor declared but
        // absent is fatal, and an undeclared tensor present in the file is silently dropped, which
        // would turn a missing curve table into a wrong-but-running model.
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            GGML_UNUSED(prefix);
            if (config.use_adaln_curves()) {
                params["adaln_t_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.time_embed_dim, config.adaln_curve_grid);
            }
            // NOT consumed by the graph: the rotary table is built on the CPU from `rope_theta` and
            // `rope_inv_freq_len`, exactly as diffusers COMPUTES this buffer rather than loading it
            // (transformer_minimax_h3.py registers it with persistent=False).  Declared anyway
            // because comfy ships it as a persistent buffer, so a reference-spelled checkpoint
            // carries it and the loader must be told where to put its 16 floats.
            params["rope.inv_freq"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.rope_inv_freq_len);
            // Also not consumed by the graph -- its PRESENCE is the whole signal, and the config has
            // already read it.  Declared for the same reason as rope.inv_freq: the loader must have
            // a home for every tensor in the file.
            if (config.qk_rope_permuted) {
                params[qk_permuted_marker_name()] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            }
            if (config.qkv_deinterleaved) {
                params[qkv_deinterleaved_marker_name()] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            }
        }

    public:
        MiniMaxH3Model() = default;

        explicit MiniMaxH3Model(MiniMaxH3Config config)
            : config(std::move(config)) {
            const MiniMaxH3Config& c = this->config;
            // apply_silu is FALSE exactly when the curve table replaces the time embedder --
            // comfy/ldm/minimax/model.py:431.
            const bool apply_silu = !c.use_adaln_curves();

            // fp32 islands.  The video patch projection is [latents*prod(patch) -> hidden] and the
            // audio one is [audio_latents -> hidden]; both are bias-ON.
            blocks["video_patch_proj"] = std::make_shared<Linear>(c.video_patch_dim(), c.hidden_size, true, c.force_fp32_islands);
            blocks["audio_patch_proj"] = std::make_shared<Linear>(c.audio_latents_dim, c.hidden_size, true, c.force_fp32_islands);
            // condition_proj and the refiner run at the MODEL dtype, not fp32.
            blocks["condition_proj"] = std::make_shared<Linear>(c.text_dim, c.hidden_size, true);

            if (!c.use_adaln_curves()) {
                blocks["time_embedder"] = std::make_shared<H3TimeEmbedder>(c.timestep_input_dim,
                                                                           c.time_embed_hidden_size,
                                                                           c.time_embed_dim,
                                                                           c.force_fp32_islands);
            }

            blocks["token_refiner"] = std::make_shared<H3TokenRefiner>(c.token_refiner_num_layers,
                                                                       c.hidden_size,
                                                                       c.num_attention_heads,
                                                                       c.attention_head_dim,
                                                                       c.ffn_hidden_size,
                                                                       c.norm_eps,
                                                                       c.qk_norm_eps,
                                                                       c.final_norm_eps);

            for (int64_t i = 0; i < c.num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<H3DiTBlock>(c, apply_silu);
            }
            blocks["final_layer"] = std::make_shared<H3FinalLayer>(c, apply_silu);
        }

        const MiniMaxH3Config& get_config() const { return config; }

        // Weight half of the F16 stream gate -- see the file header.  All-or-nothing: one
        // non-NVFP4 weight inside `blocks.*` is an F16 src1 that supports_op refuses, and
        // ggml_backend_sched then runs that Linear on the CPU.
        bool blocks_accept_f16() {
            for (int64_t i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<H3DiTBlock>(blocks["blocks." + std::to_string(i)]);
                if (block == nullptr || !block->linears_all_nvfp4()) {
                    return false;
                }
            }
            return true;
        }

        // condition_proj -> token_refiner.  comfy/ldm/minimax/model.py:455-459
        // (`preprocess_text_embeds`): a pure function of the conditioning, so upstream hoists it out
        // of the sampling loop via extra_conds and hands the DiT an already-refined [hidden, L]
        // tensor.  Same contract here -- forward() detects a pre-refined context by its width.
        ggml_tensor* forward_text(GGMLRunnerContext* ctx, ggml_tensor* context) {
            GGML_ASSERT(context != nullptr);
            if (context->ne[0] == config.hidden_size) {
                return context;
            }
            auto condition_proj = std::dynamic_pointer_cast<Linear>(blocks["condition_proj"]);
            auto token_refiner  = std::dynamic_pointer_cast<H3TokenRefiner>(blocks["token_refiner"]);
            return token_refiner->forward(ctx, condition_proj->forward(ctx, context));
        }

        // t_emb for the (at most four) distinct timesteps.
        //   full form : `t_freq` is the CPU-built sinusoid [freq_dim, M]; run the time embedder.
        //   curve form: `curve_i0` / `curve_i1` are I32 [M] row indices into `adaln_t_table` and
        //               `curve_frac` is F32 [1, M]; the embedding is a straight lerp of two rows.
        ggml_tensor* forward_time(GGMLRunnerContext* ctx,
                                  ggml_tensor* t_freq,
                                  ggml_tensor* curve_i0,
                                  ggml_tensor* curve_i1,
                                  ggml_tensor* curve_frac) {
            if (!config.use_adaln_curves()) {
                GGML_ASSERT(t_freq != nullptr);
                auto time_embedder = std::dynamic_pointer_cast<H3TimeEmbedder>(blocks["time_embedder"]);
                return time_embedder->forward(ctx, t_freq);
            }

            GGML_ASSERT(curve_i0 != nullptr && curve_i1 != nullptr && curve_frac != nullptr);
            ggml_tensor* table = params["adaln_t_table"];
            ggml_tensor* lo    = ggml_get_rows(ctx->ggml_ctx, table, curve_i0);  // [k, M]
            ggml_tensor* hi    = ggml_get_rows(ctx->ggml_ctx, table, curve_i1);  // [k, M]
            ggml_tensor* delta = ggml_sub(ctx->ggml_ctx, hi, lo);
            // curve_frac broadcasts over the k axis (ne0 == 1).
            return ggml_add(ctx->ggml_ctx, lo, ggml_mul(ctx->ggml_ctx, delta, curve_frac));
        }

        // The packed forward.
        //
        //   video_rows : [video_patch_dim, n_video_rows] F32, in PACKED ORDER -- conditioning rows
        //                first (keyframes then references, matching PackedLayout::img_pos) and the
        //                target rows last.  This is `all_video_rows` in the reference, already
        //                interleaved: the reference scatters by a boolean mask, but the layout puts
        //                every conditioning row BEFORE every target row, so a concatenation is the
        //                same thing without the gather.
        //   audio_rows : [audio_latents_dim, n_audio_rows] F32, same rule.
        //   text_embed : [hidden, text_len] (already through forward_text).
        //   pe         : [2, 2, config.rope_pe_pairs(), seq_len] F32.
        //
        // Returns {video velocity rows, audio velocity rows} for the TARGET segments only, both
        // F32, both still in row form (unpatchify / unpack is the runner's job).
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* video_rows,
                                                      ggml_tensor* audio_rows,
                                                      ggml_tensor* text_embed,
                                                      ggml_tensor* t_emb,
                                                      ggml_tensor* pe,
                                                      const PackedLayout& layout,
                                                      const TimestepPlan& plan,
                                                      bool dit_f16,
                                                      // Diagnostics only (MINIMAX_H3_DIAG_GRAPH=1).
                                                      // Non-null enables the per-block AdaLN /
                                                      // residual capture, which needs the graph to
                                                      // expand its extra output nodes onto.
                                                      ggml_cgraph* diag_gf = nullptr) {
            auto video_patch_proj = std::dynamic_pointer_cast<Linear>(blocks["video_patch_proj"]);
            auto audio_patch_proj = std::dynamic_pointer_cast<Linear>(blocks["audio_patch_proj"]);
            auto final_layer      = std::dynamic_pointer_cast<H3FinalLayer>(blocks["final_layer"]);

            const ggml_type stream_type = dit_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

            ggml_tensor* video_embed = video_rows != nullptr ? video_patch_proj->forward(ctx, video_rows) : nullptr;
            ggml_tensor* audio_embed = audio_rows != nullptr ? audio_patch_proj->forward(ctx, audio_rows) : nullptr;

            auto to_stream = [&](ggml_tensor* t) -> ggml_tensor* {
                if (t != nullptr && t->type != stream_type) {
                    t = ggml_cast(ctx->ggml_ctx, t, stream_type);
                }
                return t;
            };
            // Cast each part BEFORE the concat, never the assembled sequence: at production length
            // the packed buffer is 679 MB in F32, and casting after assembly would materialise it
            // at full width first.  ggml_concat also requires src0->type == src1->type == dst->type.
            video_embed = to_stream(video_embed);
            audio_embed = to_stream(audio_embed);
            text_embed  = to_stream(text_embed);

            // Assemble by SLICES.  The segment table is contiguous and in packed order, and both
            // embed buffers are in packed order too, so this is a concatenation -- no scatter.
            std::vector<ggml_tensor*> parts;
            parts.reserve(layout.segments.size());
            int64_t voff = 0;
            int64_t aoff = 0;
            for (const Segment& seg : layout.segments) {
                const int64_t n = seg.rows();
                if (n == 0) {
                    continue;
                }
                switch (seg.kind) {
                    case SegmentKind::Text:
                        GGML_ASSERT(text_embed != nullptr && text_embed->ne[1] == n);
                        parts.push_back(text_embed);
                        break;
                    case SegmentKind::Cond:
                    case SegmentKind::RefImage:
                    case SegmentKind::Video:
                        GGML_ASSERT(video_embed != nullptr && voff + n <= video_embed->ne[1]);
                        parts.push_back(h3_seq_window(ctx->ggml_ctx, video_embed, voff, voff + n));
                        voff += n;
                        break;
                    case SegmentKind::Audio:
                    case SegmentKind::RefAudio:
                        GGML_ASSERT(audio_embed != nullptr && aoff + n <= audio_embed->ne[1]);
                        parts.push_back(h3_seq_window(ctx->ggml_ctx, audio_embed, aoff, aoff + n));
                        aoff += n;
                        break;
                }
            }
            ggml_tensor* h = h3_concat_seq(ctx->ggml_ctx, parts);
            GGML_ASSERT(h->ne[1] == layout.seq_len);
            if (MiniMaxH3Diag::on()) {
                auto tensor_layout_json = [](const ggml_tensor* t) {
                    if (t == nullptr) {
                        return std::string("null");
                    }
                    return std::string("{\"type\":") + MiniMaxH3Diag::jstr(ggml_type_name(t->type)) +
                           ",\"ne\":[" + MiniMaxH3Diag::jint(t->ne[0]) + "," +
                           MiniMaxH3Diag::jint(t->ne[1]) + "," + MiniMaxH3Diag::jint(t->ne[2]) + "," +
                           MiniMaxH3Diag::jint(t->ne[3]) + "]" +
                           ",\"nb\":[" + MiniMaxH3Diag::jint(static_cast<int64_t>(t->nb[0])) + "," +
                           MiniMaxH3Diag::jint(static_cast<int64_t>(t->nb[1])) + "," +
                           MiniMaxH3Diag::jint(static_cast<int64_t>(t->nb[2])) + "," +
                           MiniMaxH3Diag::jint(static_cast<int64_t>(t->nb[3])) + "]" +
                           ",\"contiguous\":" + MiniMaxH3Diag::jbool(ggml_is_contiguous(t)) + "}";
                };
                const int64_t audio_byte_start =
                    plan.audio.present ? static_cast<int64_t>(static_cast<size_t>(plan.audio.start) * h->nb[1]) : -1;
                const int64_t audio_byte_stop =
                    plan.audio.present ? static_cast<int64_t>(static_cast<size_t>(plan.audio.stop) * h->nb[1]) : -1;
                MiniMaxH3Diag::rec().set_section(
                    "forward_tensor_layout",
                    "{\"packed\":" + tensor_layout_json(h) +
                        ",\"video_embed\":" + tensor_layout_json(video_embed) +
                        ",\"audio_embed\":" + tensor_layout_json(audio_embed) +
                        ",\"rope\":" + tensor_layout_json(pe) +
                        ",\"audio_byte_start\":" + MiniMaxH3Diag::jint(audio_byte_start) +
                        ",\"audio_byte_stop\":" + MiniMaxH3Diag::jint(audio_byte_stop) +
                        ",\"audio_view_fits_packed_bytes\":" +
                        MiniMaxH3Diag::jbool(!plan.audio.present ||
                                             (audio_byte_start >= 0 && audio_byte_stop >= audio_byte_start &&
                                              static_cast<uint64_t>(audio_byte_stop) <=
                                                  static_cast<uint64_t>(ggml_nbytes(h)))) +
                        "}");
            }

            const bool diag_capture        = diag_gf != nullptr && MiniMaxH3Diag::graph_on();
            const bool diag_branch_capture = diag_capture && MiniMaxH3Diag::rec().graph_branch();
            std::vector<ggml_tensor*> diag_mods;
            std::vector<ggml_tensor*> diag_branch;
            for (int64_t i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<H3DiTBlock>(blocks["blocks." + std::to_string(i)]);
                diag_mods.clear();
                diag_branch.clear();
                h = block->forward(ctx,
                                   h,
                                   t_emb,
                                   plan.mod_segments,
                                   pe,
                                   diag_capture ? &diag_mods : nullptr,
                                   diag_branch_capture ? &diag_branch : nullptr);
                if (diag_capture) {
                    // BEFORE the cut below, deliberately: every capture node then sits inside this
                    // block's own node range in graph order, so marking them into this block's cut
                    // group does not straddle a segment boundary.
                    MiniMaxH3Diag::capture_block(ctx->ggml_ctx,
                                                 diag_gf,
                                                 i,
                                                 h,
                                                 diag_mods,
                                                 diag_branch,
                                                 plan.video,
                                                 plan.audio,
                                                 MiniMaxH3Diag::rec().graph_rows());
                }
                // Per-block cut point.  Mandatory rather than nice-to-have at this size: the
                // activation working set of a single block is multiple gigabytes at production
                // length, so the graph has to be executable in block-sized pieces.
                //
                // Keep the LAST block and the final heads in one segment.  If block 49 is cut,
                // the planner hoists only the tiny audio-row slice across the boundary while it
                // leaves the video head on the producer side.  At long sequence lengths that
                // asymmetric cached slice is corrupted (audio velocity becomes orthogonal while
                // video remains bit-close); whether it happens used to depend on weight type and
                // --max-vram deciding if the final segment happened to merge back.  There is no
                // memory benefit to that last cut: final-layer scratch is smaller than block 49's
                // already-live peak.  Making the dependency structural removes the cliff.
                if (i + 1 < config.num_layers) {
                    sd::ggml_graph_cut::mark_graph_cut(h,
                                                      "minimax_h3.blocks." + std::to_string(i),
                                                      "hidden_states");
                }
            }

            return final_layer->forward(ctx, h, t_emb, plan.video, plan.audio);
        }
    };

    // ---------------------------------------------------------------------------------------
    // Rotary table
    // ---------------------------------------------------------------------------------------

    // [S, 3] positions -> the [2, 2, pairs, S] pe tensor Rope::apply_rope consumes.
    //
    // This is Rope::embed_nd(ids, 1, theta, {2*inv_len, 2*inv_len, 2*inv_len}) specialised: identical
    // arithmetic (Rope::rope's omega[i] = theta^(-2i/dim) with dim = 2*inv_len is exactly
    // 1/theta^(i/inv_len), the reference's inv_freq), but written straight into one flat buffer.
    // embed_nd allocates a std::vector<float> PER POSITION, which at 33,051 rows is ~33k heap
    // allocations on the critical path of every sampler step; the cache in the runner makes that
    // once per render instead of ten times, and this makes the once cheap.
    //
    // `pairs` may exceed the 3 * inv_freq_len rotated pairs: on a permuted checkpoint apply_rope
    // runs at full head width, and the trailing pairs -- which carry H3's pass-through channels --
    // have to be the IDENTITY rotation so those channels come out untouched.
    __STATIC_INLINE__ std::vector<float> gen_minimax_h3_pe(const std::vector<std::array<double, 3>>& positions,
                                                           int64_t inv_freq_len,
                                                           float theta,
                                                           int64_t pairs) {
        const int64_t rotated_pairs = 3 * inv_freq_len;
        GGML_ASSERT(pairs >= rotated_pairs);
        const size_t n = positions.size();
        std::vector<float> pe(n * static_cast<size_t>(pairs) * 4, 0.f);
        for (size_t s = 0; s < n; s++) {
            for (int64_t p = rotated_pairs; p < pairs; p++) {
                const size_t base = (s * static_cast<size_t>(pairs) + static_cast<size_t>(p)) * 4;
                pe[base + 0]      = 1.f;  // [cos, -sin, sin, cos] = [1, 0, 0, 1]
                pe[base + 3]      = 1.f;
            }
        }

        std::vector<double> inv_freq(static_cast<size_t>(inv_freq_len));
        for (int64_t i = 0; i < inv_freq_len; i++) {
            inv_freq[static_cast<size_t>(i)] =
                1.0 / std::pow(static_cast<double>(theta), static_cast<double>(i) / static_cast<double>(inv_freq_len));
        }

        for (size_t s = 0; s < n; s++) {
            const double* axis = positions[s].data();  // (t, h, w)
            for (int64_t a = 0; a < 3; a++) {
                for (int64_t i = 0; i < inv_freq_len; i++) {
                    const double angle = axis[a] * inv_freq[static_cast<size_t>(i)];
                    const float c      = static_cast<float>(std::cos(angle));
                    const float sn     = static_cast<float>(std::sin(angle));
                    // Rope::rope()'s per-pair quadruple: [cos, -sin, sin, cos].
                    const size_t base = (s * static_cast<size_t>(pairs) + static_cast<size_t>(a * inv_freq_len + i)) * 4;
                    pe[base + 0]      = c;
                    pe[base + 1]      = -sn;
                    pe[base + 2]      = sn;
                    pe[base + 3]      = c;
                }
            }
        }
        return pe;
    }

    // ---------------------------------------------------------------------------------------
    // Runner
    // ---------------------------------------------------------------------------------------

    // Everything H3-specific that DiffusionParams cannot carry yet.
    //
    // `MiniMaxH3DiffusionExtra` (model/diffusion/model.hpp) now exists and carries the one field
    // the sampler has per call today -- the per-row modality tags -- which compute(DiffusionParams)
    // copies in below.  The rest (layout geometry, packed audio latents, conditioning rows) is
    // still set directly on this member by whoever drives the pipeline, because none of it is
    // produced by the generic sampler yet.  Anything moved onto the extra should be DELETED from
    // here rather than mirrored, so there is never a question of which one wins.
    struct MiniMaxH3Request {
        // Describes the packed sequence.  latent_h / latent_w must be the PATCH-PADDED latent
        // dimensions -- the DiT pads x up to the 2x2 patch and the layout has to agree.
        LayoutRequest layout;

        // [T_audio, 2, audio_latents_dim, 1] -- torch [B, C, ch, T] read back to front.
        sd::Tensor<float> audio_x;

        // Already-patchified / already-packed CONDITIONING rows, in packed order, with any noise
        // augmentation already applied.  Left to the caller on purpose: the reference's condition
        // noise is drawn from a CPU RNG that intentionally restarts the same stream per condition
        // (comfy _cond_video_rows / _cond_audio_rows), which is pipeline policy, not DiT graph.
        sd::Tensor<float> cond_video_rows;  // [video_patch_dim, n]
        sd::Tensor<float> cond_audio_rows;  // [audio_latents_dim, n]

        // One modality tag per TEXT row (0 video, 1 text, 2 audio).  Empty means "the whole text
        // span is text" -- correct for t2va, wrong for a presentation carrying vision pads.
        std::vector<int> text_token_tags;

        float visual_cond_noise_aug = VISUAL_COND_TIMESTEP;
        float audio_cond_noise_aug  = AUDIO_COND_TIMESTEP;

        // Per-request overrides of config.sigma_shift_*; <= 0 keeps the config value.
        float sigma_shift_video = -1.f;
        float sigma_shift_audio = -1.f;

        // The NEXT step's video sigma, in the same [0, 1] units as `timesteps / timestep_scale`.
        // Set per step by the sampler (see MiniMaxH3DiffusionExtra::sigma_next); < 0 means "the
        // sampler did not say", which forces the instantaneous-slope behaviour regardless of
        // MINIMAX_H3_AUDIO_SECANT.  Only the AUDIO velocity scale reads it.
        float sigma_v_next = -1.f;

        // The engine's DiscreteFlowDenoiser hands the DiT `sigma * 1000` (sigma_to_t), which is the
        // same convention comfy's model_base uses.  Kept as a named constant so a future scheduler
        // that passes raw sigma has one place to say so.
        float timestep_scale = 1000.f;
    };

    struct MiniMaxH3Runner : public DiffusionModelRunner {
        MiniMaxH3Config config;
        MiniMaxH3Model model;
        MiniMaxH3Request request;

        // Rotary table cache.  gen_minimax_h3_pe() is pure in the packed position grid, which does
        // not change within a render; keyed on the grid BYTES so a hit is byte-identical to a
        // rebuild by construction.  Same fix, same keying pattern, as the LTX and krea2 pe caches.
        std::vector<float> pe_vec;
        uint64_t pe_key = 0;

        // CPU buffers backing graph inputs.  set_backend_tensor_data records a POINTER and copies
        // after graph allocation, so these have to outlive build_graph -- hence members.
        std::vector<float> t_freq_vec;
        std::vector<int32_t> curve_i0_vec;
        std::vector<int32_t> curve_i1_vec;
        std::vector<float> curve_frac_vec;
        sd::Tensor<float> video_rows_cache;
        sd::Tensor<float> audio_rows_cache;

        PackedLayout layout;
        TimestepPlan plan;
        // PackedLayout::matches() only compares the 5-tuple signature (text_len, t, h, w, audio_t),
        // which is all upstream keys on -- but the keyframe and reference blocks change the layout
        // WITHOUT changing that tuple, so a ref2va request following a t2va request of the same
        // shape would silently reuse the wrong row ordering.  Hash them separately.
        uint64_t layout_extra_key = 0;

        // MINIMAX_H3_DIAG_GRAPH=1 only.  Collects the per-block AdaLN / residual captures out of
        // the (possibly segmented) graph after compute; inert and allocation-free when off.
        MiniMaxH3Diag::GraphReadback diag_readback;

        MiniMaxH3Runner(ggml_backend_t backend,
                        const String2TensorStorage& tensor_storage_map      = {},
                        const std::string prefix                            = "",
                        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(MiniMaxH3Config::detect_from_weights(tensor_storage_map, prefix)) {
            model = MiniMaxH3Model(config);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "minimax_h3";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        void runner_done() override {
            pe_key = 0;
            pe_vec.clear();
            GGMLRunner::runner_done();
        }

        // ---------------------------------------------------------------------------------
        // Audio pack / unpack.  comfy/ldm/minimax/model.py:69-77.
        //
        // ⚠️ Stereo audio rows are CHANNEL-MAJOR: all of channel 0, then all of channel 1.
        // Interleaving them will not error -- it will just destroy the stereo field, which is the
        // same class of bug as the LTX `seg_*.audio` interleave.
        // ---------------------------------------------------------------------------------

        // ax: [T, ch, C, 1] -> [C, ch*T]
        ggml_tensor* pack_audio(ggml_context* ctx, ggml_tensor* ax) const {
            const int64_t t  = ax->ne[0];
            const int64_t ch = ax->ne[1];
            const int64_t c  = ax->ne[2];
            ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, ax, 1, 2, 0, 3));  // [C, T, ch, 1]
            return ggml_reshape_2d(ctx, out, c, ch * t);
        }

        // rows: [C, ch*T] -> [T, ch, C, 1]
        ggml_tensor* unpack_audio(ggml_context* ctx, ggml_tensor* rows, int64_t ch) const {
            const int64_t c = rows->ne[0];
            const int64_t t = rows->ne[1] / ch;
            GGML_ASSERT(t * ch == rows->ne[1]);
            ggml_tensor* out = ggml_reshape_3d(ctx, rows, c, t, ch);                // [C, T, ch]
            out              = ggml_cont(ctx, ggml_permute(ctx, out, 2, 0, 1, 3));  // [T, ch, C]
            return ggml_reshape_4d(ctx, out, t, ch, c, 1);
        }

        // Pack the audio velocity into trailing channels of the video latent so the runner can hand
        // the sampler ONE tensor, exactly as the LTX-AV runner does.  The caller splits it back with
        // the same arithmetic.
        ggml_tensor* merge_av(ggml_context* ctx, ggml_tensor* vx, ggml_tensor* ax) const {
            if (ax == nullptr || ggml_nelements(ax) == 0) {
                return vx;
            }
            const int64_t width   = vx->ne[0];
            const int64_t height  = vx->ne[1];
            const int64_t frames  = vx->ne[2];
            const int64_t divisor = width * height * frames;
            const int64_t values  = ggml_nelements(ax);
            const int64_t pad     = (divisor - (values % divisor)) % divisor;

            ggml_tensor* flat = ggml_reshape_4d(ctx, ggml_cont(ctx, ax), values, 1, 1, 1);
            if (pad > 0) {
                flat = ggml_ext_pad(ctx, flat, static_cast<int>(pad), 0, 0, 0);
            }
            flat = ggml_reshape_4d(ctx, flat, width, height, frames, (values + pad) / divisor);
            return ggml_concat(ctx, vx, flat, 3);
        }

        // Everything PackedLayout::build reads that the 5-tuple signature does NOT cover.
        static uint64_t layout_extra_hash(const LayoutRequest& req) {
            std::vector<int64_t> fields;
            fields.reserve(1 + req.keyframes.size() + req.refs.size() * 5);
            fields.push_back(req.frame_count);
            for (const Keyframe& kf : req.keyframes) {
                fields.push_back(kf.resolved_frame_index);
            }
            for (const Reference& ref : req.refs) {
                fields.push_back(static_cast<int64_t>(ref.kind));
                fields.push_back(ref.latent_h);
                fields.push_back(ref.latent_w);
                fields.push_back(ref.latent_t);
                fields.push_back(ref.ref_audio_t);
            }
            return h3_bulk_hash(fields.data(), fields.size() * sizeof(int64_t));
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf = new_graph_custom(MINIMAX_H3_GRAPH_SIZE);

            GGML_ASSERT(!context_tensor.empty() && "minimax-h3 needs conditioning");
            GGML_ASSERT(!timesteps_tensor.empty());
            GGML_ASSERT(x_tensor.dim() >= 4);
            GGML_ASSERT(x_tensor.shape()[3] == config.latents_dim);

            ggml_tensor* x       = make_input(x_tensor);
            ggml_tensor* context = make_input(context_tensor);
            ggml_tensor* ax      = make_optional_input(request.audio_x);

            // ---- patch padding ---------------------------------------------------------------
            const int64_t orig_w = x->ne[0];
            const int64_t orig_h = x->ne[1];
            const int64_t t_len  = x->ne[2];
            auto runner_ctx      = get_context();
            x                    = DiT::pad_to_patch_size(&runner_ctx, x, config.patch_h, config.patch_w);
            const int64_t lat_w  = x->ne[0];
            const int64_t lat_h  = x->ne[1];

            // ---- layout ----------------------------------------------------------------------
            LayoutRequest layout_request = request.layout;
            layout_request.text_len      = context->ne[1];
            layout_request.latent_t      = t_len;
            layout_request.latent_h      = lat_h;
            layout_request.latent_w      = lat_w;
            layout_request.audio_t       = ax != nullptr ? ax->ne[0] : 0;
            const uint64_t extra_key     = layout_extra_hash(layout_request);
            if (!layout.matches(layout_request) || layout.seq_len == 0 || extra_key != layout_extra_key) {
                layout           = PackedLayout::build(layout_request);
                layout_extra_key = extra_key;
            }

            // ---- schedule --------------------------------------------------------------------
            const float shift_v = request.sigma_shift_video > 0.f ? request.sigma_shift_video : config.sigma_shift_video;
            const float shift_a = request.sigma_shift_audio > 0.f ? request.sigma_shift_audio : config.sigma_shift_audio;
            const float sigma_v = timesteps_tensor.data()[0] / request.timestep_scale;
            // MINIMAX_H3_AUDIO_SECANT=1 -- scale the audio velocity by the DISCRETE
            // (sigma_a - sigma_a_next)/(sigma_v - sigma_v_next) instead of comfy's instantaneous
            // d(sigma_a)/d(sigma_v).  That makes one Euler step on the VIDEO schedule advance the
            // audio stream by exactly the step diffusers' second (audio) scheduler would take.
            // Default OFF: this changes every audio latent, so it is an A/B, not a silent repair.
            static const bool audio_secant = h3_env_flag("MINIMAX_H3_AUDIO_SECANT", false);
            plan                           = build_timestep_plan(layout,
                                                                 sigma_v,
                                                                 shift_v,
                                                                 shift_a,
                                                                 request.visual_cond_noise_aug,
                                                                 request.audio_cond_noise_aug,
                                                                 request.text_token_tags.empty() ? nullptr : &request.text_token_tags,
                                                                 request.sigma_v_next,
                                                                 audio_secant);
            {
                // One line per step, not once per run: the whole point of the A/B is watching the
                // two candidates diverge as sigma falls, and a run-once log hides exactly that.
                static bool logged_mode = false;
                if (!logged_mode) {
                    logged_mode = true;
                    LOG_INFO("minimax_h3: audio velocity scale = %s",
                             audio_secant ? "SECANT, discrete d(sigma_a)/d(sigma_v) (MINIMAX_H3_AUDIO_SECANT=1)"
                                          : "INSTANTANEOUS slope (comfy default; set MINIMAX_H3_AUDIO_SECANT=1 to A/B)");
                }
                LOG_DEBUG("minimax_h3: sigma_v %.6f -> %.6f  sigma_a %.6f -> %.6f  slope inst %.4f  secant %.4f%s  using %.4f",
                          plan.sigma_v,
                          plan.sigma_v_next,
                          plan.sigma_a,
                          plan.sigma_a_next,
                          plan.slope_a_instant,
                          plan.slope_a_secant,
                          plan.secant_valid ? "" : " (INVALID: sampler gave no next sigma)",
                          plan.slope_a);
                if (audio_secant && !plan.secant_valid) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        LOG_WARN("minimax_h3: MINIMAX_H3_AUDIO_SECANT=1 but no next sigma reached the DiT; "
                                 "falling back to the instantaneous slope (the A/B is a NO-OP).");
                    }
                }
            }

            // ---- rotary table ----------------------------------------------------------------
            uint64_t want_pe_key               = h3_bulk_hash(layout.position_ids.data(),
                                                              layout.position_ids.size() * sizeof(std::array<double, 3>));
            const int64_t pe_pairs             = config.rope_pe_pairs();
            want_pe_key                        = want_pe_key ^ (static_cast<uint64_t>(config.rope_inv_freq_len) * 0x9E3779B97F4A7C15ull);
            want_pe_key                        = want_pe_key ^ (static_cast<uint64_t>(pe_pairs) * 0xC2B2AE3D27D4EB4Full);
            static const bool pe_cache_enabled = h3_env_flag("MINIMAX_H3_PE_CACHE", true);
            if (!pe_cache_enabled || pe_key != want_pe_key || pe_vec.empty()) {
                const int64_t t0 = ggml_time_ms();
                pe_vec           = gen_minimax_h3_pe(layout.position_ids, config.rope_inv_freq_len, config.rope_theta, pe_pairs);
                LOG_DEBUG("minimax_h3: rope table rebuilt for %" PRId64 " rows in %" PRId64 " ms",
                          layout.seq_len,
                          ggml_time_ms() - t0);
            }
            pe_key          = want_pe_key;
            ggml_tensor* pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, pe_pairs, layout.seq_len);
            set_backend_tensor_data(pe, pe_vec.data());

            // ---- timestep embedding inputs ---------------------------------------------------
            const int64_t m_ts      = static_cast<int64_t>(plan.unique_t.size());
            ggml_tensor* t_freq     = nullptr;
            ggml_tensor* curve_i0   = nullptr;
            ggml_tensor* curve_i1   = nullptr;
            ggml_tensor* curve_frac = nullptr;
            if (config.use_adaln_curves()) {
                curve_table_lookup(plan.unique_t, config.adaln_curve_grid, curve_i0_vec, curve_i1_vec, curve_frac_vec);
                curve_i0 = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, m_ts);
                set_backend_tensor_data(curve_i0, curve_i0_vec.data());
                curve_i1 = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, m_ts);
                set_backend_tensor_data(curve_i1, curve_i1_vec.data());
                curve_frac = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 1, m_ts);
                set_backend_tensor_data(curve_frac, curve_frac_vec.data());
            } else {
                t_freq_vec = timestep_sinusoid(plan.unique_t, config.timestep_input_dim);
                t_freq     = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, config.timestep_input_dim, m_ts);
                set_backend_tensor_data(t_freq, t_freq_vec.data());
            }

            // ---- conditioning rows -----------------------------------------------------------
            ggml_tensor* cond_video = make_optional_input(request.cond_video_rows);
            ggml_tensor* cond_audio = make_optional_input(request.cond_audio_rows);

            // ---- target rows -----------------------------------------------------------------
            // patchify_3d with patch_last=true reproduces comfy's patchify_video exactly: the row
            // order is (t, h, w) major and the column order is C-major within the patch,
            // ((c*pt + r)*ph + p)*pw + q.
            ggml_tensor* video_rows = DiT::patchify_3d(compute_ctx, x, config.patch_t, config.patch_h, config.patch_w, 1, true);
            video_rows              = ggml_reshape_2d(compute_ctx, video_rows, config.video_patch_dim(), video_rows->ne[1]);
            if (cond_video != nullptr) {
                // Conditioning rows come FIRST in packed order (keyframes, then references), so the
                // reference's masked scatter is a plain concatenation here.
                video_rows = ggml_concat(compute_ctx, cond_video, video_rows, 1);
            }

            ggml_tensor* audio_rows = nullptr;
            if (ax != nullptr) {
                audio_rows = pack_audio(compute_ctx, ax);
                if (cond_audio != nullptr) {
                    audio_rows = ggml_concat(compute_ctx, cond_audio, audio_rows, 1);
                }
            } else if (cond_audio != nullptr) {
                audio_rows = cond_audio;
            }

            // ---- F16 stream decision ---------------------------------------------------------
            const bool dit_f16 = h3_dit_f16_env() && h3_dit_f16_device_ok(&runner_ctx) && model.blocks_accept_f16();
            {
                static int logged  = -1;
                const int decision = dit_f16 ? 1 : 0;
                if (logged != decision) {
                    logged = decision;
                    LOG_INFO("minimax_h3: DiT residual stream = %s", dit_f16 ? "F16 (MINIMAX_H3_DIT_F16)" : "F32");
                }
            }

            // ---- diagnostics -----------------------------------------------------------------
            // Written EVERY step (the layout is stable within a render, but the plan is not, and a
            // bundle that only describes step 1 cannot be diffed against itself).  Pure host
            // arithmetic over structures that already exist -- see minimax_h3_diag.hpp.
            if (MiniMaxH3Diag::on()) {
                MiniMaxH3Diag::GeometryFacts g;
                g.hidden_size       = config.hidden_size;
                g.num_layers        = config.num_layers;
                g.video_patch_dim   = config.video_patch_dim();
                g.audio_latents_dim = config.audio_latents_dim;
                g.latents_dim       = config.latents_dim;
                g.attention_inner_dim = config.inner_dim();
                g.ffn_hidden_size     = config.ffn_hidden_size;
                g.patch_t           = config.patch_t;
                g.patch_h           = config.patch_h;
                g.patch_w           = config.patch_w;
                g.adaln_curve_form  = config.use_adaln_curves();
                g.time_embed_dim    = config.time_embed_dim;
                g.dit_f16           = dit_f16;
                g.orig_latent_w     = orig_w;
                g.orig_latent_h     = orig_h;
                g.latent_w          = lat_w;
                g.latent_h          = lat_h;
                g.latent_t          = t_len;
                g.audio_t           = layout_request.audio_t;
                g.text_len          = layout_request.text_len;
                g.mlp_chunk_rows    = h3_mlp_chunk_rows();
                g.graph_capacity    = MINIMAX_H3_GRAPH_SIZE;
                MiniMaxH3Diag::emit_geometry(layout, g);
                MiniMaxH3Diag::rec().append(
                    "timestep_plan",
                    "{\"step\":" + MiniMaxH3Diag::jint(MiniMaxH3Diag::rec().step()) +
                        ",\"plan\":" + MiniMaxH3Diag::timestep_plan_json(plan) + "}");
            }

            // ---- forward ---------------------------------------------------------------------
            ggml_tensor* text_embed = model.forward_text(&runner_ctx, context);
            ggml_tensor* t_emb      = model.forward_time(&runner_ctx, t_freq, curve_i0, curve_i1, curve_frac);

            auto out = model.forward(&runner_ctx,
                                     video_rows,
                                     audio_rows,
                                     text_embed,
                                     t_emb,
                                     pe,
                                     layout,
                                     plan,
                                     dit_f16,
                                     MiniMaxH3Diag::graph_on() ? gf : nullptr);

            // ---- back to latent shape --------------------------------------------------------
            GGML_ASSERT(out.first != nullptr && "minimax-h3: the packed layout has no target video segment");
            ggml_tensor* video_out = DiT::unpatchify_3d(compute_ctx,
                                                        out.first,
                                                        t_len,
                                                        lat_h / config.patch_h,
                                                        lat_w / config.patch_w,
                                                        config.patch_t,
                                                        config.patch_h,
                                                        config.patch_w,
                                                        true);  // [lat_w, lat_h, t_len, C]
            // Crop back only when the patch padding actually added rows/columns: ggml_ext_slice
            // conts, so an unconditional call would copy the whole latent twice for nothing on the
            // (usual) already-aligned case.
            if (orig_h != lat_h) {
                video_out = ggml_ext_slice(compute_ctx, video_out, 1, 0, orig_h);
            }
            if (orig_w != lat_w) {
                video_out = ggml_ext_slice(compute_ctx, video_out, 0, 0, orig_w);
            }

            // ⚠️ SIGN AND SCALE ARE PART OF THE MODEL, not of the sampler.
            //
            // comfy/ldm/minimax/model.py:642-646 returns `[-video, -slope_a * audio]` and the model
            // is registered as ModelType.FLOW, i.e. `denoised = x - sigma * out`.  This engine's
            // DiscreteFlowDenoiser has exactly that scaling (c_skip = 1, c_out = -sigma), so the
            // negation belongs here and the audio slope with it: scaling the audio velocity by
            // d(sigma_a)/d(sigma_v) is what makes the flat ODE the sampler integrates on the VIDEO
            // schedule equal the audio stream's true ODE on its own.  `plan.slope_a` is that
            // factor; whether it holds comfy's instantaneous derivative or diffusers' per-step
            // secant is decided above by MINIMAX_H3_AUDIO_SECANT.  The VIDEO branch is untouched
            // by that switch -- it is on the schedule the sampler is already integrating.
            video_out = ggml_scale(compute_ctx, video_out, -1.0f);

            ggml_tensor* audio_out = nullptr;
            if (out.second != nullptr) {
                audio_out = unpack_audio(compute_ctx, out.second, 2);
                audio_out = ggml_scale(compute_ctx, audio_out, -plan.slope_a);
            }

            ggml_tensor* merged = merge_av(compute_ctx, video_out, audio_out);
            ggml_build_forward_expand(gf, merged);
            if (MiniMaxH3Diag::on()) {
                const int64_t nodes = ggml_graph_n_nodes(gf);
                const int64_t leafs = sd::ggml_graph_cut::leaf_count(gf);
                MiniMaxH3Diag::rec().append(
                    "graph_build",
                    "{\"step\":" + MiniMaxH3Diag::jint(MiniMaxH3Diag::rec().step()) +
                        ",\"nodes\":" + MiniMaxH3Diag::jint(nodes) +
                        ",\"leafs\":" + MiniMaxH3Diag::jint(leafs) +
                        ",\"capacity\":" + MiniMaxH3Diag::jint(MINIMAX_H3_GRAPH_SIZE) +
                        ",\"capacity_headroom\":" + MiniMaxH3Diag::jint(MINIMAX_H3_GRAPH_SIZE - nodes) +
                        ",\"within_capacity\":" + MiniMaxH3Diag::jbool(nodes <= MINIMAX_H3_GRAPH_SIZE) + "}");
            }
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context);
            };
            // MINIMAX_H3_DIAG_GRAPH=1.  The hook fires once per graph-cut segment on the segmented
            // path and once on the plain one; a block's captures are only visible in the call that
            // computed them, so scan() picks each one up as it appears and emit() runs after the
            // whole graph is done.
            const bool diag_graph = MiniMaxH3Diag::graph_on();
            const bool diag_on    = MiniMaxH3Diag::on();
            int64_t previous_segment_nodes = -1;
            int64_t previous_segment_leafs = -1;
            uint64_t previous_segment_signature = 0;
            if (diag_on) {
                graph_cut_plan_hook_ = [](const GraphCutPlan& graph_plan) {
                    std::vector<std::string> segments;
                    segments.reserve(graph_plan.segments.size());
                    for (const GraphCutSegment& seg : graph_plan.segments) {
                        segments.push_back(
                            "{\"name\":" + MiniMaxH3Diag::jstr(seg.group_name) +
                            ",\"nodes\":" +
                            MiniMaxH3Diag::jint(static_cast<int64_t>(seg.internal_node_indices.size())) +
                            ",\"outputs\":" +
                            MiniMaxH3Diag::jint(static_cast<int64_t>(seg.output_node_indices.size())) +
                            ",\"compute_bytes\":" +
                            MiniMaxH3Diag::jint(static_cast<int64_t>(seg.compute_buffer_size)) +
                            ",\"previous_cut_bytes\":" +
                            MiniMaxH3Diag::jint(static_cast<int64_t>(seg.input_previous_cut_bytes)) +
                            ",\"external_bytes\":" +
                            MiniMaxH3Diag::jint(static_cast<int64_t>(seg.input_external_bytes)) +
                            ",\"param_bytes\":" +
                            MiniMaxH3Diag::jint(static_cast<int64_t>(seg.input_param_bytes)) + "}");
                    }
                    MiniMaxH3Diag::rec().append(
                        "graph_cut_plan",
                        "{\"step\":" + MiniMaxH3Diag::jint(MiniMaxH3Diag::rec().step()) +
                            ",\"valid\":" + MiniMaxH3Diag::jbool(graph_plan.valid) +
                            ",\"has_cuts\":" + MiniMaxH3Diag::jbool(graph_plan.has_cuts) +
                            ",\"nodes\":" + MiniMaxH3Diag::jint(graph_plan.n_nodes) +
                            ",\"leafs\":" + MiniMaxH3Diag::jint(graph_plan.n_leafs) +
                            ",\"segments\":" + MiniMaxH3Diag::jarray(segments) + "}");
                };
            }
            if (diag_on) {
                diag_readback.reset();
                segment_readback_hook_ = [this,
                                          diag_graph,
                                          &previous_segment_nodes,
                                          &previous_segment_leafs,
                                          &previous_segment_signature](ggml_cgraph* g) {
                    if (diag_graph) {
                        diag_readback.scan(g);
                    }

                    // The shared gallocr may reuse a prior segment's positional address plan when
                    // node/leaf counts and allocation sizes happen to match
                    // (ggml_extend.hpp:2590; ggml-alloc.c:1023).  Hash the structure that check does
                    // NOT compare: op/type/shape and source positions.  A count match with a hash
                    // mismatch is a concrete allocator-reuse CANDIDATE, not an audio symptom.
                    // It is not proof: ggml_gallocr_needs_realloc() can still re-plan because a
                    // positional tensor grew beyond the prior allocation.
                    uint64_t signature = 1469598103934665603ull;
                    auto mix = [&signature](uint64_t value) {
                        for (int i = 0; i < 8; ++i) {
                            signature ^= static_cast<unsigned char>(value >> (i * 8));
                            signature *= 1099511628211ull;
                        }
                    };
                    const int64_t nodes = ggml_graph_n_nodes(g);
                    const int64_t leafs = sd::ggml_graph_cut::leaf_count(g);
                    for (int64_t i = 0; i < nodes; ++i) {
                        const ggml_tensor* node = ggml_graph_node(g, static_cast<int>(i));
                        mix(static_cast<uint64_t>(node->op));
                        mix(static_cast<uint64_t>(node->type));
                        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                            mix(static_cast<uint64_t>(node->ne[d]));
                            mix(static_cast<uint64_t>(node->nb[d]));
                        }
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            int64_t source_index = -1;
                            if (node->src[s] != nullptr) {
                                for (int64_t j = 0; j < nodes; ++j) {
                                    if (ggml_graph_node(g, static_cast<int>(j)) == node->src[s]) {
                                        source_index = j;
                                        break;
                                    }
                                }
                            }
                            mix(static_cast<uint64_t>(source_index));
                        }
                    }
                    const bool same_counts = nodes == previous_segment_nodes && leafs == previous_segment_leafs;
                    const bool reuse_candidate =
                        same_counts && previous_segment_signature != 0 && signature != previous_segment_signature;
                    MiniMaxH3Diag::rec().append(
                        "graph_segment_execution",
                        "{\"step\":" + MiniMaxH3Diag::jint(MiniMaxH3Diag::rec().step()) +
                            ",\"nodes\":" + MiniMaxH3Diag::jint(nodes) +
                            ",\"leafs\":" + MiniMaxH3Diag::jint(leafs) +
                            ",\"structure_hash\":" + MiniMaxH3Diag::jstr(std::to_string(signature)) +
                            ",\"same_counts_as_previous\":" + MiniMaxH3Diag::jbool(same_counts) +
                            ",\"allocator_reuse_candidate\":" + MiniMaxH3Diag::jbool(reuse_candidate) + "}");
                    if (reuse_candidate) {
                        MiniMaxH3Diag::rec().warn(
                            "graph-cut consecutive segments have equal node/leaf counts but different "
                            "op/shape/source topology; shared gallocr positional-plan reuse is a candidate "
                            "(not proof: a size increase can force replanning; try "
                            "SD_GRAPH_CUT_RESERVE_PER_SEGMENT=1)");
                    }
                    previous_segment_nodes     = nodes;
                    previous_segment_leafs     = leafs;
                    previous_segment_signature = signature;
                };
            }
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false),
                                                          x.dim());
            if (diag_on) {
                segment_readback_hook_ = nullptr;
            }
            if (diag_graph) {
                if (diag_readback.empty()) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        MiniMaxH3Diag::rec().warn(
                            "MINIMAX_H3_DIAG_GRAPH=1 captured NOTHING -- the per-block AdaLN / residual "
                            "tensors were not materialised by this execution path");
                    }
                }
                diag_readback.emit(MiniMaxH3Diag::rec().step(), plan);
                MiniMaxH3Diag::rec().flush();
            }
            if (diag_on) {
                graph_cut_plan_hook_ = nullptr;
            }
            return result;
        }

        // The modality tags arrive per call on MiniMaxH3DiffusionExtra; the remaining H3-specific
        // inputs still ride on `request` (see the note there).  The tags are copied in on EVERY
        // call that carries the extra, including when it carries none, so a previous render's tag
        // vector can never leak into this one.
        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            if (const auto* extra = std::get_if<MiniMaxH3DiffusionExtra>(&diffusion_params.extra)) {
                request.text_token_tags.clear();
                if (extra->text_token_tags != nullptr && !extra->text_token_tags->empty()) {
                    const std::vector<int32_t>& tags = extra->text_token_tags->values();
                    request.text_token_tags.assign(tags.begin(), tags.end());
                }
                // Reset on EVERY call carrying the extra, exactly as the tags are: a stale next
                // sigma would silently scale one step's audio velocity by the previous step's
                // secant, which is a much harder thing to see than an outright missing value.
                request.sigma_v_next = -1.f;
                if (extra->sigma_next != nullptr && !extra->sigma_next->empty()) {
                    request.sigma_v_next = extra->sigma_next->data()[0];
                }
            }
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context));
        }
    };

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__
