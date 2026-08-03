#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_QK_PERMUTE_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_QK_PERMUTE_HPP__

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// MiniMax-H3 q/k head-channel permutation -- shared by the GGUF converter (src/convert.cpp) and by
// the DiT graph (minimax_h3.hpp).  Deliberately ggml-free so the converter does not have to pull in
// the graph, in the same spirit as minimax_h3_layout.hpp.
//
// ---------------------------------------------------------------------------------------------
// WHAT
//
// H3 rotates the leading `rot_dim = 2 * 3 * rope_inv_freq_len` channels of every head (96 of 128 at
// the shipping sizes) with the split-half convention taken over the ROTARY SUB-BLOCK: the pair is
// (c, c + rot_dim/2), and channels [rot_dim, head_dim) pass through untouched
// (diffusers `_apply_rotary_emb`, comfy `rms_rope_split_half` with `rot_dim = 2 * table_pairs`).
//
// ggml's Rope::apply_rope / ggml_rope_pe_ni pair (c, c + head_dim/2) over the WHOLE head.  That is a
// DIFFERENT pairing, so the graph would otherwise have to slice the head, rope the slice, and concat
// the pass-through back on -- a full [head_dim x seq x heads] copy for both q and k, every block,
// every step.
//
// Permuting the head-channel axis of the q and k halves of `qkv_proj.weight`, and of
// `q_norm.weight` / `k_norm.weight`, makes full-width split-half reproduce H3's pairing exactly:
//
//     new[c]                  = old[c]                        c in [0, rot_dim/2)
//     new[head_dim/2 + c]     = old[rot_dim/2 + c]
//     new[rot_dim/2 + j]      = old[rot_dim + j]              j in [0, (head_dim - rot_dim)/2)
//     new[head_dim/2 + rot_dim/2 + j] = old[rot_dim + (head_dim - rot_dim)/2 + j]
//
// At head_dim 128 / rot_dim 96 that is
//     new[0..47] = old[0..47],    new[64..111]  = old[48..95],
//     new[48..63] = old[96..111], new[112..127] = old[112..127].
// The pass-through channels land in pairs [rot_dim/2, head_dim/2), which the widened rotation table
// leaves alone because its tail pairs are the IDENTITY rotation (cos = 1, sin = 0).
//
// ---------------------------------------------------------------------------------------------
// WHY IT IS FREE, AND NOT AN APPROXIMATION
//
// q.k is a dot product over the head axis, so any permutation applied to q and k TOGETHER leaves
// every attention score unchanged.  `v` and `out_proj` are not touched, so the block output is
// unchanged as well.  RMSNorm is permutation-equivariant (its scale is the rms over the whole head,
// which is permutation-invariant), which is why permuting `q_norm` / `k_norm` alongside is both
// necessary and sufficient.
//
// Verified numerically in `tools/ref_qk_permute.py`, including against the diffusers PR source run
// verbatim under torch:
//     permuted engine path vs today's slice/rope/concat path   max|diff| = 1.4e-14 (rel 4.2e-16)
//     both paths vs the reference                              max|diff| = 8.4e-08, identical
// (the 8.4e-08 floor is the reference's own float32 angle computation, not the permutation).
//
// ---------------------------------------------------------------------------------------------
// COMPATIBILITY
//
// The converter stamps `<prefix>rope.qk_permuted` into the output file.  A checkpoint WITHOUT that
// marker is assumed unpermuted and runs the original slice/rope/concat path, so an old GGUF keeps
// working; it just does not get the win.  Detection is by marker presence only -- there is nothing
// in a permuted tensor's shape or statistics that distinguishes it, so guessing is not an option and
// a wrong guess here is silent, not loud.
//
// ⚠️ The reverse direction is NOT protected: a build that predates this file would not declare the
// marker, an undeclared tensor is silently dropped by the loader, and it would then run the
// slice/rope/concat path on permuted weights -- wrong, quietly.  That is acceptable only because
// the H3 port has never shipped, so no such build exists outside this tree.  If a permuted GGUF is
// ever published, the marker has to become something an old build REFUSES rather than ignores.

// =================================================================================================
// SECOND, INDEPENDENT REWRITE IN THIS FILE: the fused-QKV DE-INTERLEAVE
//
// The RAW MiniMax-H3 checkpoint does NOT store `qkv_proj.weight` as [q_all; k_all; v_all].  It
// stores it PER-HEAD INTERLEAVED:
//
//     [head0: q(head_dim) k(head_dim) v(head_dim), head1: q k v, ...]
//
// The reference implementation un-interleaves at LOAD time (`_reorder_grouped_qkv_to_qkv`), and the
// official diffusers converter does the same on the way in (`reorder_interleaved_qkv`,
// scripts/convert_minimax_h3_to_diffusers.py:110, applied by the shard streamer at :765 to EVERY
// key ending `.attn.qkv_proj.weight` -- token_refiner blocks included).  So [q_all; k_all; v_all] is
// an IN-MEMORY layout that no file on disk ever holds, and reference SPELLING says nothing about
// which of the two a given file carries.
//
// The engine's `split_qkv` assumes contiguous.  Fed a raw shard it produces garbage silently: every
// shape matches, nothing errors, the attention is just wrong.
//
// ORDERING, and it is not negotiable: DE-INTERLEAVE FIRST, then the q/k RoPE head-channel
// permutation above.  That permutation is defined on [q_all; k_all; v_all] -- it walks the first two
// thirds of the row axis and treats each as `heads` consecutive head blocks.  Applied to an
// interleaved tensor it would rewrite head0's q/k/v as if they were the q of heads 0/1/2, so both
// transforms would then be wrong TOGETHER, which is exactly the failure that is hardest to see.
//
// Like the RoPE permutation this moves whole rows and never reaches inside one, so it is exact for
// every ggml type, quantised included.
//
// `q_norm.weight` / `k_norm.weight` are NOT touched, and that is a claim about the reference, not an
// assumption: `reorder_interleaved_qkv` reshapes to (heads, 3, head_dim, ...) and only ever moves
// whole (head, third) blocks, so the channel order INSIDE a block is untouched -- a per-head-dim
// gain vector is invariant under it.  Proven against the reference function itself in
// tests/minimax_h3_qkv_deinterleave_test.cpp (case `q_norm_invariance`).

namespace MiniMaxH3 {

    // Presence of this tensor (under the diffusion-model prefix) means the q/k head channels of
    // every `blocks.N.attn` are already permuted for full-width split-half RoPE.  Value unused.
    inline const char* qk_permuted_marker_name() {
        return "rope.qk_permuted";
    }

    // Presence of this tensor means this converter has already de-interleaved every fused
    // `attn.qkv_proj.weight` in the file, so the rows are [q_all; k_all; v_all].  Value unused.
    //
    // Presence-only, exactly like the RoPE marker and for the same reason: absence does NOT mean
    // "interleaved", it means "unknown" -- a file could have been de-interleaved by some other tool.
    // Which is why the converter refuses to guess; see src/convert.cpp.
    inline const char* qkv_deinterleaved_marker_name() {
        return "attn.qkv_deinterleaved";
    }

    // Source row-BLOCK index for each destination row-block, at head_dim-row granularity.
    // Destination block `t * heads + h` (the [q_all; k_all; v_all] order) comes from source block
    // `3 * h + t` (the per-head-interleaved order).  This is a block transpose of an
    // (heads x 3) grid into a (3 x heads) one.
    inline int64_t deinterleave_source_block(int64_t dst_block, int64_t heads) {
        const int64_t t = dst_block / heads;
        const int64_t h = dst_block % heads;
        return 3 * h + t;
    }

    // In-place de-interleave of `3 * heads` blocks of `head_dim` rows each, `row_bytes` wide.
    //
    // Cycle-following rather than gather-into-a-copy: the scratch buffer is ONE block
    // (head_dim * row_bytes) instead of the whole tensor, which at the shipping geometry is 4.5 MB
    // instead of 231 MB per concurrent converter thread.
    inline void deinterleave_qkv_rows(uint8_t* data,
                                      size_t row_bytes,
                                      int64_t heads,
                                      int64_t head_dim,
                                      std::vector<uint8_t>& scratch,
                                      std::vector<uint8_t>& visited) {
        const int64_t n_blocks   = 3 * heads;
        const size_t block_bytes = static_cast<size_t>(head_dim) * row_bytes;
        scratch.resize(block_bytes);
        visited.assign(static_cast<size_t>(n_blocks), 0);

        auto block = [&](int64_t i) { return data + static_cast<size_t>(i) * block_bytes; };

        for (int64_t start = 0; start < n_blocks; start++) {
            if (visited[static_cast<size_t>(start)]) {
                continue;
            }
            // out[i] = in[src(i)].  Walk the cycle, holding the one block the cycle overwrites last.
            std::memcpy(scratch.data(), block(start), block_bytes);
            int64_t i = start;
            while (true) {
                visited[static_cast<size_t>(i)] = 1;
                const int64_t j                 = deinterleave_source_block(i, heads);
                if (j == start) {
                    std::memcpy(block(i), scratch.data(), block_bytes);
                    break;
                }
                std::memcpy(block(i), block(j), block_bytes);
                i = j;
            }
        }
    }

    inline bool qk_permutation_is_applicable(int64_t head_dim, int64_t rot_dim) {
        return head_dim > 0 && rot_dim > 0 && rot_dim <= head_dim &&
               head_dim % 2 == 0 && rot_dim % 2 == 0 && (head_dim - rot_dim) % 2 == 0;
    }

    // `out[i] = in[perm[i]]` over one head's channels.  Identity when rot_dim == head_dim.
    inline std::vector<int64_t> build_qk_head_permutation(int64_t head_dim, int64_t rot_dim) {
        std::vector<int64_t> perm;
        if (!qk_permutation_is_applicable(head_dim, rot_dim)) {
            return perm;
        }
        const int64_t half      = head_dim / 2;
        const int64_t rot_half  = rot_dim / 2;
        const int64_t pass_half = (head_dim - rot_dim) / 2;
        perm.resize(static_cast<size_t>(head_dim));
        for (int64_t c = 0; c < rot_half; c++) {
            perm[static_cast<size_t>(c)]        = c;
            perm[static_cast<size_t>(half + c)] = rot_half + c;
        }
        for (int64_t j = 0; j < pass_half; j++) {
            perm[static_cast<size_t>(rot_half + j)]        = rot_dim + j;
            perm[static_cast<size_t>(half + rot_half + j)] = rot_dim + pass_half + j;
        }
        return perm;
    }

    // Apply `perm` to `n_heads` consecutive groups of `perm.size()` rows, each `row_bytes` wide.
    //
    // A row here is one OUTPUT channel of a Linear weight, i.e. `ne[0]` elements laid out
    // contiguously.  Because a permutation of whole rows never reaches inside one, this is exact for
    // every ggml type, quantised included: `ne[0]` is a whole number of quant blocks and the
    // quantiser works per row, so permuting before or after quantisation gives the same bytes.
    inline void permute_head_rows(uint8_t* data,
                                  size_t row_bytes,
                                  int64_t n_heads,
                                  const std::vector<int64_t>& perm,
                                  std::vector<uint8_t>& scratch) {
        const size_t head_dim = perm.size();
        scratch.resize(head_dim * row_bytes);
        for (int64_t h = 0; h < n_heads; h++) {
            uint8_t* base = data + static_cast<size_t>(h) * head_dim * row_bytes;
            std::memcpy(scratch.data(), base, scratch.size());
            for (size_t c = 0; c < head_dim; c++) {
                std::memcpy(base + c * row_bytes,
                            scratch.data() + static_cast<size_t>(perm[c]) * row_bytes,
                            row_bytes);
            }
        }
    }

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_QK_PERMUTE_HPP__
