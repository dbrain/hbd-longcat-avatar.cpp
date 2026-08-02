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

namespace MiniMaxH3 {

    // Presence of this tensor (under the diffusion-model prefix) means the q/k head channels of
    // every `blocks.N.attn` are already permuted for full-width split-half RoPE.  Value unused.
    inline const char* qk_permuted_marker_name() {
        return "rope.qk_permuted";
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
