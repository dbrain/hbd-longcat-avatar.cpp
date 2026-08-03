#include <inttypes.h>
#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "core/util.h"
#include "model/diffusion/minimax_h3_qk_permute.hpp"
#include "model/diffusion/minimax_h3_qkv_layout_probe.hpp"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/streaming_writer.h"
#include "model_loader.h"

// How a tensor's row / head-channel axis is rewritten on the way out.  See the H3 section below.
enum class QKPermuteMode {
    None,
    QKVRows,         // 2-D [in, 3*inner] in a DiT block: de-interleave, then permute q/k head channels
    RefinerQKVRows,  // 2-D [in, 3*inner] in the token refiner: de-interleave only (it has no rotary)
    NormElems        // 1-D [head_dim]: permute the channels themselves
};

struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
    QKPermuteMode qk_permute = QKPermuteMode::None;
};

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

/*=================================== MiniMax-H3 fused-QKV row rewrites ================================*/

// TWO independent rewrites of the H3 DiT's fused attention weights, in a fixed order.
//
// 1. DE-INTERLEAVE.  A raw MiniMax-H3 shard stores `attn.qkv_proj.weight` per-head interleaved
//    (`[h0: q k v, h1: q k v, ...]`); the reference un-interleaves at load time and the official
//    diffusers converter does it on the way in.  Our engine's `split_qkv` wants
//    `[q_all; k_all; v_all]`, so a raw shard has to be normalised here or the attention is garbage
//    with no error anywhere.
// 2. ROPE PERMUTATION.  Rewrite the head-channel axis of the q/k projections so the runtime can rope
//    the full head in one call instead of slicing, roping and concatenating.
//
// Both are whole-row moves, so both are exact for every ggml type including a quantised one.  The
// maths, the proof that (2) is an exact identity, and the numeric verification all live in
// model/diffusion/minimax_h3_qk_permute.hpp.
//
// ⚠️ ORDER MATTERS AND IS NOT NEGOTIABLE: (1) then (2).  The RoPE permutation is defined on
// `[q_all; k_all; v_all]` -- it walks the first two thirds of the row axis as `heads` consecutive
// head blocks.  Run on an interleaved tensor it would rewrite head0's q/k/v as if they were the q of
// heads 0/1/2, and then BOTH transforms are wrong together, which is the hardest kind of wrong to
// see.  h3_apply_permute is the single place that sequences them.
//
// Scope differs between the two, and it follows the reference rather than symmetry:
//   * de-interleave covers `<prefix>blocks.<N>.attn.qkv_proj.weight` AND
//     `<prefix>token_refiner.blocks.<N>.attn.qkv_proj.weight`, because the diffusers shard streamer
//     applies `reorder_interleaved_qkv` to every key ending `.attn.qkv_proj.weight` (:765);
//   * the RoPE permutation covers the DiT blocks only -- the refiner has no rotary embedding at all,
//     so permuting it would be churn that buys nothing.
// `q_norm` / `k_norm` take the RoPE permutation and are INVARIANT under the de-interleave (which
// only ever moves whole head blocks, never channels inside one).  Nothing else is touched.
struct H3QKPermutePlan {
    bool active       = false;  // anything at all to do
    bool deinterleave = false;  // raw per-head-interleaved qkv -> [q_all; k_all; v_all]
    bool rope_permute = false;  // rewrite q/k head channels for full-width split-half RoPE
    bool refuse       = false;  // an H3 DiT was found but the layout question was not answered
    // The operator ANSWERED the layout question -- with either 0 or 1. Distinct from
    // `deinterleave`, which is only the 1 case. A `0` answer performs no work but is still
    // knowledge worth recording, so the marker gets stamped either way: it means "these rows are
    // [q_all; k_all; v_all]", which a correct `0` also establishes. Without this, a `0` conversion
    // produced a file with the RoPE marker and no qkv marker -- indistinguishable from a file made
    // by a build that predates the de-interleave, so the second conversion stage fired the
    // "attention is wrong and no re-conversion can repair it" warning on a perfectly good model.
    bool layout_declared = false;
    std::string prefix;         // includes the trailing '.', e.g. "model.diffusion_model."
    int64_t head_dim = 0;
    int64_t rot_dim  = 0;
    int64_t heads    = 0;       // > 0 once an H3 fused qkv was found, and the layout probe can run
    std::vector<int64_t> perm;  // out[i] = in[perm[i]]
};

// <prefix><middle>blocks.<digits>.attn.<leaf>
static bool h3_block_attn_leaf(const std::string& name,
                               const std::string& prefix,
                               const char* middle,
                               const char* leaf) {
    const std::string head = prefix + middle + "blocks.";
    if (!starts_with(name, head)) {
        return false;
    }
    size_t i = head.size();
    if (i >= name.size() || !isdigit(static_cast<unsigned char>(name[i]))) {
        return false;
    }
    while (i < name.size() && isdigit(static_cast<unsigned char>(name[i]))) {
        i++;
    }
    return name.compare(i, std::string::npos, std::string(".attn.") + leaf) == 0;
}

// MINIMAX_H3_QKV_DEINTERLEAVE -- the ONLY thing that decides whether the de-interleave runs.
//
// Why an explicit switch and not the probe's verdict: getting this wrong is silent in BOTH
// directions.  De-interleaving an already-contiguous file and failing to de-interleave a raw shard
// produce the same symptom -- a model that loads, renders, and is subtly wrong -- so a statistical
// guess is not an acceptable basis for the decision, however good the statistic is.  The probe's job
// is to tell the operator what the numbers say; the operator's job is to decide.
//
// An env var rather than a CLI flag because `convert()` / `convert_with_components()` are the public
// conversion API in include/, four callers wide, and this is a one-checkpoint-family concern that
// should not widen a public signature.  Same idiom as MINIMAX_H3_MLP_CHUNK on the runtime side.
//
// Returns: 1 = de-interleave, 0 = do not, -1 = not set.
static int h3_env_deinterleave() {
    const char* v = getenv("MINIMAX_H3_QKV_DEINTERLEAVE");
    if (v == nullptr || v[0] == '\0') {
        return -1;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "1" || s == "on" || s == "yes" || s == "true") {
        return 1;
    }
    if (s == "0" || s == "off" || s == "no" || s == "false") {
        return 0;
    }
    LOG_WARN("MINIMAX_H3_QKV_DEINTERLEAVE='%s' is not a recognised value, treating it as unset", v);
    return -1;
}

// Run the data-driven layout probe on blocks.0's fused qkv and say -- loudly -- what it found.
// ADVISORY ONLY: nothing in this file branches on the result.  See model/diffusion/
// minimax_h3_qkv_layout_probe.hpp for the statistic and tools/h3_qkv_layout_probe.py for the
// negative-control validation (256 Qwen3-VL trials, zero wrong verdicts).
//
// ⚠️ MUST be called AFTER h3_add_markers.  It reads a tensor, which latches
// ModelLoader::process_model_files, and that call SNAPSHOTS the tensor storage map into per-file
// tensor lists -- a marker inserted afterwards would never be loaded and the conversion would fail
// on it.  Cost: one F32 copy of the fused qkv (462 MB at the shipping geometry) plus a couple of
// seconds, once, on the planning thread.  Worth it: it cross-checks the operator's declaration on
// every single conversion instead of only when someone remembers to ask.
static void h3_log_layout_probe(ModelLoader& model_loader,
                                const std::string& prefix,
                                int64_t heads,
                                int64_t head_dim) {
    // `heads` is the sentinel, not the prefix: heads > 0 means an H3 fused qkv was found, and the
    // prefix is legitimately empty for a `-m` load (see h3_plan_qk_permutation).  heads == 1 is
    // skipped because the two layouts are then identical and the statistic has nothing to separate.
    if (heads <= 1 || head_dim <= 0) {
        return;
    }
    const std::string name = prefix + "blocks.0.attn.qkv_proj.weight";
    auto it                = model_loader.get_tensor_storage_map().find(name);
    if (it == model_loader.get_tensor_storage_map().end()) {
        return;
    }
    const int64_t cols = it->second.ne[0];
    const int64_t rows = it->second.ne[1];

    // One tensor at F32: 462 MB at the shipping geometry, held only for the duration of the probe
    // and only on the single thread that plans the conversion.
    std::vector<float> data;
    if (!model_loader.load_float_tensor(name, data) ||
        data.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
        LOG_WARN("minimax-h3: could not read '%s' for the qkv layout probe", name.c_str());
        return;
    }
    const MiniMaxH3::QKVLayoutReport r = MiniMaxH3::probe_qkv_layout(data.data(), rows, cols, heads, head_dim);
    if (!r.valid) {
        LOG_WARN("minimax-h3: qkv layout probe could not run on '%s'", name.c_str());
        return;
    }
    LOG_INFO("minimax-h3: qkv layout probe on '%s' (heads=%" PRId64 ", head_dim=%" PRId64 ")",
             name.c_str(),
             heads,
             head_dim);
    LOG_INFO("minimax-h3:   row-norm  F contiguous=%.3f (p=%.4f)  interleaved=%.3f (p=%.4f)",
             r.norm.f_contiguous,
             r.norm.p_contiguous,
             r.norm.f_interleaved,
             r.norm.p_interleaved);
    LOG_INFO("minimax-h3:   profile   F contiguous=%.3f (p=%.4f)  interleaved=%.3f (p=%.4f)",
             r.profile.f_contiguous,
             r.profile.p_contiguous,
             r.profile.f_interleaved,
             r.profile.p_interleaved);
    LOG_INFO("minimax-h3:   VERDICT: %s  (advisory only -- MINIMAX_H3_QKV_DEINTERLEAVE decides)",
             MiniMaxH3::qkv_layout_name(r.layout));
}

static H3QKPermutePlan h3_plan_qk_permutation(ModelLoader& model_loader) {
    H3QKPermutePlan plan;
    const String2TensorStorage& map = model_loader.get_tensor_storage_map();

    const std::string rope_marker = MiniMaxH3::qk_permuted_marker_name();
    const std::string qkv_marker  = MiniMaxH3::qkv_deinterleaved_marker_name();
    const std::string anchor      = "video_patch_proj.weight";
    // ⚠️ The prefix is legitimately EMPTY when the checkpoint is loaded with `-m` rather than
    // `--diffusion-model`: the DiT's own tensor names arrive verbatim, so the anchor is the WHOLE
    // name.  This used to require `name.size() > anchor.size()` and `!prefix.empty()`, which meant
    // an H3 DiT converted via `-m` was never detected at all -- the de-interleave, the RoPE
    // permutation AND the refusal guard below all silently did nothing, and the conversion
    // "succeeded" into a file whose attention is wrong.  MEASURED on the stub DiT: `-m` converted
    // happily with the gate unset, `--diffusion-model` refused as designed.  Detection must key on
    // the anchor PAIR, which is what get_sd_version() uses, and not on there being a prefix.
    bool found = false;
    std::string prefix;
    for (const auto& [name, _] : map) {
        if (name.size() >= anchor.size() && ends_with(name, anchor)) {
            const std::string candidate = name.substr(0, name.size() - anchor.size());
            if (map.count(candidate + "audio_patch_proj.weight") != 0) {
                prefix = candidate;
                found  = true;
                break;
            }
        }
    }
    if (!found) {
        return plan;
    }

    auto find = [&](const std::string& n) -> const TensorStorage* {
        auto it = map.find(prefix + n);
        return it == map.end() ? nullptr : &it->second;
    };
    const TensorStorage* q_norm = find("blocks.0.attn.q_norm.weight");
    const TensorStorage* qkv    = find("blocks.0.attn.qkv_proj.weight");
    if (q_norm == nullptr || q_norm->n_dims != 1 || qkv == nullptr || qkv->n_dims != 2) {
        LOG_WARN("minimax-h3: no blocks.0.attn q/k weights found, passing the attention weights through");
        return plan;
    }
    plan.head_dim = q_norm->ne[0];
    if (plan.head_dim <= 0 || qkv->ne[1] % (3 * plan.head_dim) != 0) {
        LOG_WARN("minimax-h3: qkv_proj out dim %" PRId64 " is not 3 * heads * %" PRId64 ", passing it through",
                 qkv->ne[1],
                 plan.head_dim);
        return plan;
    }
    plan.prefix = prefix;
    plan.heads  = qkv->ne[1] / (3 * plan.head_dim);

    const bool rope_done = map.count(prefix + rope_marker) != 0;
    const bool qkv_done  = map.count(prefix + qkv_marker) != 0;

    // ---- decide the de-interleave -------------------------------------------------------------
    if (qkv_done) {
        LOG_INFO("minimax-h3: fused qkv is already de-interleaved (marker present), passing it through");
    } else if (rope_done) {
        // The RoPE permutation has already been baked in, so the row axis is no longer in either of
        // the two layouts the de-interleave is defined on -- running it now would be wrong, and so
        // would leaving it if the source was ever a raw shard.  Neither can be fixed from here.
        LOG_WARN(
            "minimax-h3: '%s' is present but '%s' is NOT. This file was produced by a build that "
            "had no fused-qkv de-interleave. If it came from a RAW MiniMax checkpoint its attention "
            "is wrong and no re-conversion can repair it -- convert again from the original "
            "checkpoint with MINIMAX_H3_QKV_DEINTERLEAVE=1.",
            rope_marker.c_str(),
            qkv_marker.c_str());
    } else {
        const int env = h3_env_deinterleave();
        if (env < 0) {
            // The caller logs the probe verdict and then this message -- see export_loaded_model.
            plan.refuse = true;
            return plan;
        }
        plan.deinterleave    = env == 1;
        plan.layout_declared = true;
        LOG_INFO("minimax-h3: MINIMAX_H3_QKV_DEINTERLEAVE=%d -> fused qkv rows will %s",
                 env,
                 plan.deinterleave ? "be de-interleaved into [q_all; k_all; v_all]" : "be left as they are");
    }

    // ---- decide the RoPE head-channel permutation ---------------------------------------------
    if (rope_done) {
        LOG_INFO("minimax-h3: q/k head channels are already permuted, passing them through");
    } else {
        // rope.inv_freq is a persistent buffer in the reference spelling; diffusers computes it
        // instead, so fall back to the reference's 16 rather than refusing.
        const TensorStorage* inv = find("rope.inv_freq");
        plan.rot_dim             = 2 * 3 * ((inv != nullptr && inv->n_dims == 1) ? inv->ne[0] : 16);

        if (!MiniMaxH3::qk_permutation_is_applicable(plan.head_dim, plan.rot_dim)) {
            LOG_WARN("minimax-h3: head_dim %" PRId64 " / rot_dim %" PRId64 " admits no q/k permutation, skipping",
                     plan.head_dim,
                     plan.rot_dim);
        } else if (plan.rot_dim == plan.head_dim) {
            // Split-half over the whole head is already H3's own pairing; there is nothing to
            // rewrite and no slice/concat to remove.
        } else {
            plan.perm         = MiniMaxH3::build_qk_head_permutation(plan.head_dim, plan.rot_dim);
            plan.rope_permute = true;
            LOG_INFO("minimax-h3: permuting q/k head channels for full-width split-half RoPE (head_dim=%" PRId64
                     ", rot_dim=%" PRId64 ")",
                     plan.head_dim,
                     plan.rot_dim);
        }
    }

    // `layout_declared` counts: a `0` answer transforms nothing but still has a marker to write,
    // and `h3_marker_name` (which keeps the markers in the export) is gated on `active`.
    plan.active = plan.deinterleave || plan.rope_permute || plan.layout_declared;
    return plan;
}

// Add the markers describing what was done.  Written through the tensor storage map rather than
// appended to the export list so that load_tensor() -- which resolves by name against the map --
// can find them.
static void h3_add_markers(ModelLoader& model_loader, const H3QKPermutePlan& plan) {
    String2TensorStorage& map     = model_loader.get_tensor_storage_map();
    const std::string anchor      = plan.prefix + "video_patch_proj.weight";
    const int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};

    auto add = [&](const std::string& leaf) {
        TensorStorage marker(plan.prefix + leaf, GGML_TYPE_F32, ne, 1, map.at(anchor).file_index, 0);
        marker.is_inline_f32 = true;
        marker.inline_f32    = 1.0f;
        map[marker.name]     = marker;
    };
    if (plan.deinterleave || plan.layout_declared) {
        add(MiniMaxH3::qkv_deinterleaved_marker_name());
    }
    if (plan.rope_permute) {
        add(MiniMaxH3::qk_permuted_marker_name());
    }
}

static bool h3_marker_name(const std::string& name, const H3QKPermutePlan& plan) {
    return plan.active && (name == plan.prefix + MiniMaxH3::qk_permuted_marker_name() ||
                           name == plan.prefix + MiniMaxH3::qkv_deinterleaved_marker_name());
}

static QKPermuteMode h3_permute_mode(const std::string& name, const H3QKPermutePlan& plan) {
    if (!plan.active) {
        return QKPermuteMode::None;
    }
    if (h3_block_attn_leaf(name, plan.prefix, "", "qkv_proj.weight")) {
        return QKPermuteMode::QKVRows;
    }
    if (plan.deinterleave && h3_block_attn_leaf(name, plan.prefix, "token_refiner.", "qkv_proj.weight")) {
        // The refiner has no rotary embedding, so it takes the de-interleave and nothing else.
        return QKPermuteMode::RefinerQKVRows;
    }
    if (plan.rope_permute && (h3_block_attn_leaf(name, plan.prefix, "", "q_norm.weight") ||
                              h3_block_attn_leaf(name, plan.prefix, "", "k_norm.weight"))) {
        // Layout-invariant under the de-interleave -- see minimax_h3_qk_permute.hpp -- so the RoPE
        // permutation is the only thing that ever reaches them.
        return QKPermuteMode::NormElems;
    }
    return QKPermuteMode::None;
}

// Apply the row rewrites to already-typed export bytes.
//
// Every QKV mode is a pure row shuffle along the OUTPUT axis, so all of them are exact for every
// type including a quantised one: a row is `ne[0]` elements, which is a whole number of quant
// blocks, and the quantiser works per row -- quantise-then-shuffle and shuffle-then-quantise are the
// same bytes.  NormElems moves individual elements, which is only meaningful when a "block" is one
// element; collect_tensors_for_export keeps those two tiny 1-D tensors at their source type for
// exactly that reason (and because 4-bit-quantising a per-channel RMSNorm gain was never a good
// idea).
//
// ⚠️ This function is the ONE place the two rewrites are sequenced, and the order below
// (de-interleave, THEN the q/k head-channel permutation) is the whole reason it is one function.
static bool h3_apply_permute(const TensorExportInfo& info, const H3QKPermutePlan& plan, std::vector<uint8_t>& data) {
    if (info.qk_permute == QKPermuteMode::None) {
        return true;
    }
    const int64_t head_dim           = plan.head_dim;
    const std::vector<int64_t>& perm = plan.perm;

    std::vector<uint8_t> scratch;

    if (info.qk_permute == QKPermuteMode::NormElems) {
        const size_t esz = ggml_type_size(info.type);
        if (ggml_blck_size(info.type) != 1 || info.storage.ne[0] != head_dim ||
            data.size() != static_cast<size_t>(head_dim) * esz) {
            LOG_ERROR("minimax-h3: cannot permute '%s' as %s", info.storage.name.c_str(), ggml_type_name(info.type));
            return false;
        }
        // One "row" is one element here, so the same helper does the job.
        MiniMaxH3::permute_head_rows(data.data(), esz, 1, perm, scratch);
        return true;
    }

    const size_t row_bytes = ggml_row_size(info.type, info.storage.ne[0]);
    const int64_t rows     = info.storage.nelements() / info.storage.ne[0];
    const int64_t inner    = rows / 3;
    if (row_bytes == 0 || rows % 3 != 0 || head_dim <= 0 || inner % head_dim != 0 ||
        data.size() != static_cast<size_t>(rows) * row_bytes) {
        LOG_ERROR("minimax-h3: cannot rewrite '%s' (%" PRId64 " rows of %zu bytes)",
                  info.storage.name.c_str(),
                  rows,
                  row_bytes);
        return false;
    }
    const int64_t heads = inner / head_dim;

    // 1. De-interleave FIRST.  Everything below indexes q/k/v as contiguous thirds, which is only
    //    true once this has run.
    if (plan.deinterleave) {
        std::vector<uint8_t> visited;
        MiniMaxH3::deinterleave_qkv_rows(data.data(), row_bytes, heads, head_dim, scratch, visited);
    }

    // 2. Then the RoPE head-channel permutation, q and k only; v keeps its channel order because
    //    out_proj consumes it.  Never on the token refiner, which has no rotary embedding.
    if (plan.rope_permute && info.qk_permute == QKPermuteMode::QKVRows) {
        for (int64_t third = 0; third < 2; third++) {
            MiniMaxH3::permute_head_rows(data.data() + static_cast<size_t>(third * inner) * row_bytes,
                                         row_bytes,
                                         heads,
                                         perm,
                                         scratch);
        }
    }
    return true;
}

/*=====================================================================================================*/

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       const H3QKPermutePlan& h3_plan,
                                       std::vector<TensorExportInfo>& tensors) {
    tensors.clear();
    tensors.reserve(model_loader.get_tensor_storage_map().size());
    for (const auto& kv : model_loader.get_tensor_storage_map()) {
        const TensorStorage& tensor_storage = kv.second;
        TensorExportInfo info;
        info.storage    = tensor_storage;
        info.type       = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);
        info.qk_permute = h3_permute_mode(tensor_storage.name, h3_plan);
        if (info.qk_permute == QKPermuteMode::NormElems || h3_marker_name(tensor_storage.name, h3_plan)) {
            // Both are per-element, not per-row: a quantised destination would make the permutation
            // meaningless for the norms and the 4-byte markers unreadable.
            info.type = tensor_storage.type;
        }
        tensors.push_back(std::move(info));
    }
    LOG_INFO("collected %zu tensors for export", tensors.size());
    return true;
}

static size_t export_tensor_nbytes(const TensorExportInfo& info) {
    TensorStorage output_storage = info.storage;
    output_storage.type          = info.type;
    return static_cast<size_t>(output_storage.nbytes());
}

static TensorWritePlan tensor_write_plan_from_export_info(const TensorExportInfo& info) {
    TensorWritePlan plan;
    plan.name   = info.storage.name;
    plan.type   = info.type;
    plan.n_dims = info.storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        plan.ne[i] = info.storage.ne[i];
    }
    return plan;
}

static std::vector<TensorWritePlan> tensor_write_plans_from_export_infos(const std::vector<TensorExportInfo>& tensors) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        plans.push_back(tensor_write_plan_from_export_info(info));
    }
    return plans;
}

static bool preallocate_output_file(const std::string& output_path, uint64_t file_size, std::string* error) {
    if (file_size == 0) {
        return true;
    }

    std::fstream file(output_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open output file '" + output_path + "' for preallocation";
        }
        return false;
    }

    // This portable fallback sets the final file size. A platform-specific
    // posix_fallocate/ftruncate path can replace it later.
    file.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
    file.put('\0');
    file.flush();
    if (!file) {
        if (error != nullptr) {
            *error = "failed to preallocate output file '" + output_path + "'";
        }
        return false;
    }
    return true;
}

static bool load_tensor_for_export(ModelLoader& model_loader, const H3QKPermutePlan& h3_plan, TensorExportJob& job) {
    size_t mem_size = 1 * 1024 * 1024;
    mem_size += ggml_tensor_overhead();
    TensorStorage output_storage = job.info.storage;
    output_storage.type          = job.info.type;
    mem_size += static_cast<size_t>(output_storage.nbytes());

    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
    if (ggml_ctx == nullptr) {
        job.error = "ggml_init failed for tensor '" + job.info.storage.name + "'";
        return false;
    }

    ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, job.info.type, job.info.storage.n_dims, job.info.storage.ne);
    if (tensor == nullptr) {
        ggml_free(ggml_ctx);
        job.error = "ggml_new_tensor failed for tensor '" + job.info.storage.name + "'";
        return false;
    }
    ggml_set_name(tensor, job.info.storage.name.c_str());

    const size_t tensor_nbytes = ggml_nbytes(tensor);
    if (tensor_nbytes > 0 && !model_loader.load_tensor(job.info.storage, tensor)) {
        ggml_free(ggml_ctx);
        job.error = "failed to load tensor '" + job.info.storage.name + "'";
        return false;
    }

    job.data.resize(tensor_nbytes);
    if (tensor_nbytes > 0) {
        memcpy(job.data.data(), tensor->data, tensor_nbytes);
    }
    ggml_free(ggml_ctx);

    if (!h3_apply_permute(job.info, h3_plan, job.data)) {
        job.error = "failed to permute q/k head channels for '" + job.info.storage.name + "'";
        return false;
    }
    return true;
}

static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::string& output_path,
                               const std::vector<TensorExportInfo>& tensors,
                               const StreamingModelWriter& writer,
                               const H3QKPermutePlan& h3_plan,
                               int n_threads,
                               std::string* error) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    n_threads = std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time       = ggml_time_ms();
    uint64_t bytes_written   = 0;
    size_t tensors_written   = 0;
    size_t next_tensor_index = 0;
    bool failed              = false;
    std::string failure;

    const size_t memory_budget = 1024ull * 1024ull * 1024ull;
    size_t reserved_bytes      = 0;

    std::mutex work_mutex;
    std::mutex progress_mutex;
    std::condition_variable memory_cv;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    auto reserve_memory = [&](size_t bytes) -> bool {
        std::unique_lock<std::mutex> lock(work_mutex);
        memory_cv.wait(lock, [&]() {
            return failed || reserved_bytes == 0 || reserved_bytes + bytes <= memory_budget;
        });
        if (failed) {
            return false;
        }
        reserved_bytes += bytes;
        return true;
    };

    auto release_memory = [&](size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            reserved_bytes -= std::min(reserved_bytes, bytes);
        }
        memory_cv.notify_all();
    };

    auto fail = [&](const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            if (!failed) {
                failed  = true;
                failure = message;
            }
        }
        memory_cv.notify_all();
    };

    for (int worker = 0; worker < n_threads; worker++) {
        workers.emplace_back([&]() {
            std::fstream output_file(output_path, std::ios::binary | std::ios::in | std::ios::out);
            if (!output_file.is_open()) {
                fail("failed to open output file '" + output_path + "' for tensor writing");
                return;
            }

            while (true) {
                size_t tensor_index = 0;
                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    if (failed || next_tensor_index >= tensors.size()) {
                        return;
                    }
                    tensor_index = next_tensor_index++;
                }

                const size_t tensor_bytes = export_tensor_nbytes(tensors[tensor_index]);
                if (!reserve_memory(tensor_bytes)) {
                    return;
                }

                TensorExportJob job;
                job.info = tensors[tensor_index];
                try {
                    job.success = load_tensor_for_export(model_loader, h3_plan, job);
                } catch (const std::exception& e) {
                    job.error   = e.what();
                    job.success = false;
                }

                if (!job.success) {
                    release_memory(tensor_bytes);
                    fail(job.error.empty() ? "streaming conversion failed" : job.error);
                    return;
                }

                std::string write_error;
                if (!writer.write_tensor(output_file,
                                         tensor_index,
                                         job.data.empty() ? nullptr : job.data.data(),
                                         job.data.size(),
                                         &write_error)) {
                    release_memory(tensor_bytes);
                    fail(write_error.empty() ? "streaming conversion write failed" : write_error);
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    bytes_written += job.data.size();
                    tensors_written++;
                    float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
                    pretty_bytes_progress(static_cast<int>(tensors_written),
                                          static_cast<int>(tensors.size()),
                                          bytes_written,
                                          elapsed_seconds);
                }
                release_memory(tensor_bytes);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    printf("\n");
    if (failed) {
        if (error != nullptr) {
            *error = failure;
        }
        return false;
    }
    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_model_file_streaming(ModelLoader& model_loader,
                                       const std::string& output_path,
                                       const std::vector<TensorExportInfo>& tensors,
                                       StreamingModelWriter& writer,
                                       const H3QKPermutePlan& h3_plan,
                                       int n_threads,
                                       std::string* error) {
    std::vector<TensorWritePlan> plans = tensor_write_plans_from_export_infos(tensors);
    if (!writer.write_metadata(output_path, plans, error)) {
        return false;
    }
    if (!preallocate_output_file(output_path, writer.file_size(), error)) {
        return false;
    }
    model_loader.process_model_files(false, false);
    return stream_tensor_data(model_loader, output_path, tensors, writer, h3_plan, n_threads, error);
}

static bool init_convert_path(ModelLoader& model_loader, const char* path, const char* prefix, bool& loaded_any) {
    if (path == nullptr || strlen(path) == 0) {
        return true;
    }
    if (!model_loader.init_from_file(path, prefix)) {
        LOG_ERROR("init model loader from file failed: '%s'", path);
        return false;
    }
    loaded_any = true;
    return true;
}

static bool export_loaded_model(ModelLoader& model_loader,
                                const char* output_path,
                                sd_type_t output_type,
                                const char* tensor_type_rules,
                                int n_threads) {
    ggml_type type             = sd_type_to_ggml_type(output_type);
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    // MiniMax-H3 only, and only for the rewrites the input has not already had; every other model
    // gets an inactive plan and an untouched export path.
    // The plan reads no tensor data, so the markers can be inserted before anything latches the
    // loader's per-file tensor lists.  h3_log_layout_probe DOES read one, hence the strict order:
    // decide, stamp, then probe.
    H3QKPermutePlan h3_plan = h3_plan_qk_permutation(model_loader);
    if (h3_plan.refuse) {
        h3_log_layout_probe(model_loader, h3_plan.prefix, h3_plan.heads, h3_plan.head_dim);
        LOG_ERROR(
            "minimax-h3: refusing to convert -- the fused qkv row order has not been declared.\n"
            "  A RAW MiniMax-H3 checkpoint stores 'attn.qkv_proj.weight' PER-HEAD INTERLEAVED; this\n"
            "  engine needs [q_all; k_all; v_all]. Reference key spelling does not distinguish the\n"
            "  two, and both mistakes are SILENT -- the model loads and renders either way.\n"
            "  Set MINIMAX_H3_QKV_DEINTERLEAVE=1 for a raw checkpoint (the normal case), or\n"
            "  MINIMAX_H3_QKV_DEINTERLEAVE=0 for one already in [q_all; k_all; v_all] order.\n"
            "  The probe verdict logged above is evidence, not a decision.");
        return false;
    }
    if (h3_plan.active) {
        h3_add_markers(model_loader, h3_plan);
    }
    h3_log_layout_probe(model_loader, h3_plan.prefix, h3_plan.heads, h3_plan.head_dim);

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, h3_plan, tensors);
    std::string error;
    if (success) {
        std::unique_ptr<StreamingModelWriter> writer;
        if (output_is_safetensors) {
            writer = std::make_unique<SafetensorsStreamingWriter>();
        } else {
            writer = std::make_unique<GGUFStreamingWriter>();
        }
        success = write_model_file_streaming(model_loader, output_path, tensors, *writer, h3_plan, n_threads, &error);
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}

bool convert_with_components(const char* model_path,
                             const char* clip_l_path,
                             const char* clip_g_path,
                             const char* t5xxl_path,
                             const char* diffusion_model_path,
                             const char* vae_path,
                             const char* output_path,
                             sd_type_t output_type,
                             const char* tensor_type_rules,
                             bool convert_name,
                             int n_threads) {
    ModelLoader model_loader;
    bool loaded_any = false;

    if (!init_convert_path(model_loader, model_path, "", loaded_any) ||
        !init_convert_path(model_loader, clip_l_path, "text_encoders.clip_l.transformer.", loaded_any) ||
        !init_convert_path(model_loader, clip_g_path, "text_encoders.clip_g.transformer.", loaded_any) ||
        !init_convert_path(model_loader, t5xxl_path, "text_encoders.t5xxl.transformer.", loaded_any) ||
        !init_convert_path(model_loader, diffusion_model_path, "model.diffusion_model.", loaded_any) ||
        !init_convert_path(model_loader, vae_path, "vae.", loaded_any)) {
        return false;
    }

    if (!loaded_any) {
        LOG_ERROR("no input model path provided for convert");
        return false;
    }

    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    return export_loaded_model(model_loader, output_path, output_type, tensor_type_rules, n_threads);
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    return convert_with_components(input_path,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   vae_path,
                                   output_path,
                                   output_type,
                                   tensor_type_rules,
                                   convert_name,
                                   0);
}
