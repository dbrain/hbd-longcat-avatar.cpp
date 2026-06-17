// rig_fsq.hpp — header-only port of the SkinTokens skin-VAE FSQ (Finite Scalar
// Quantization) index->code path.  PURE CPU.  namespace rig.
//
// Ports src/model/skin_vae/autoencoders/FSQ.py :
//
//   indices_to_codes(indices)            (~L139)
//     = project_out( _indices_to_codes(indices) )
//
//   _indices_to_codes(indices)           (~L122)
//     level_indices = indices_to_level_indices(indices)   # mixed-radix digits
//     codes         = _scale_and_shift_inverse(level_indices)
//
//   indices_to_level_indices(indices)    (~L133)
//     codes_non_centered = (indices // _basis) % _levels   # per-dim digit
//
//   _scale_and_shift_inverse(zhat)       (~L118)
//     half_width = _levels // 2
//     return (zhat - half_width) / half_width              # symmetric range
//
// REAL config (from grpo_1400.ckpt -> pretrained_vae skin_vae_2_10_32768):
//   FSQ_dict = { levels: [8,8,8,8,8], dim: 512 }
//   codebook_dim = len(levels) = 5
//   codebook_size = prod(levels) = 8^5 = 32768  (== vae_vocab)
//   _basis = cumprod([1] + levels[:-1]) = [1, 8, 64, 512, 4096]
//   half_width = levels // 2 = [4,4,4,4,4]
//   has_projections == (dim != codebook_dim) -> TRUE:
//     project_in  : Linear(512 -> 5)   (encode side, unused here)
//     project_out : Linear(5 -> 512)   (applied by indices_to_codes)
//
// With levels=8 (even), a level digit li in {0..7} maps to (li-4)/4 in
// {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75} — asymmetric (no +1.0).
//
// indices_to_codes is what tokenrig.decode() calls:
//   indices = skin_token - tokenizer.vocab_size   (offset 267)
//   z = vae.model.FSQ.indices_to_codes(indices)    # [N, 512]
// then z is reshaped to [b, tokens_per_skin, 512] and fed to the VAE decoder.
#pragma once
#include <cstdint>
#include <vector>
#include <cassert>

namespace rig {

// FSQ for the SkinTokens skin-VAE.  Defaults == the real checkpoint config.
struct Fsq {
    // levels[d] : number of quantization levels per code dim
    std::vector<int> levels = {8, 8, 8, 8, 8};
    // _basis[d] = cumprod([1] + levels[:-1])
    std::vector<int64_t> basis;          // computed in init()
    int codebook_dim = 5;                // == levels.size()
    int64_t codebook_size = 32768;       // == prod(levels)

    // project_out : Linear(codebook_dim -> out_dim), applied by indices_to_codes.
    // Row-major torch weight has shape [out_dim, codebook_dim] (== ggml ne
    // [codebook_dim, out_dim]).  Stored here flattened row-major [out_dim*codebook_dim].
    int out_dim = 512;
    bool has_projections = true;
    std::vector<float> proj_out_w;       // [out_dim * codebook_dim]
    std::vector<float> proj_out_b;       // [out_dim]

    void init() {
        codebook_dim = (int)levels.size();
        basis.assign(codebook_dim, 1);
        int64_t acc = 1;
        for (int d = 0; d < codebook_dim; ++d) {
            basis[d] = acc;              // cumprod([1] + levels[:-1])
            acc *= levels[d];
        }
        codebook_size = acc;             // prod(levels)
    }

    // --- _indices_to_codes : index -> [codebook_dim] symmetric code (pre-project) ---
    // out_codes must have room for codebook_dim floats.
    inline void index_to_code(int64_t index, float* out_codes) const {
        for (int d = 0; d < codebook_dim; ++d) {
            // mixed-radix digit:  (index // basis[d]) % levels[d]
            int64_t li = (index / basis[d]) % (int64_t)levels[d];
            // _scale_and_shift_inverse: (li - half_width) / half_width
            int half_width = levels[d] / 2;     // integer div, matches _levels // 2
            out_codes[d] = (float)((double)(li - half_width) / (double)half_width);
        }
    }

    // --- project_out : [codebook_dim] -> [out_dim] (Linear with bias) ---
    inline void project_out(const float* code, float* out) const {
        for (int o = 0; o < out_dim; ++o) {
            double acc = proj_out_b.empty() ? 0.0 : (double)proj_out_b[o];
            const float* wr = &proj_out_w[(size_t)o * codebook_dim];
            for (int d = 0; d < codebook_dim; ++d) acc += (double)wr[d] * (double)code[d];
            out[o] = (float)acc;
        }
    }

    // --- indices_to_codes(indices) for a flat [N] index list ---------------
    // Returns [N * codebook_dim] when project=false (the raw symmetric codes),
    // or [N * out_dim] when project=true (the full FSQ.indices_to_codes output).
    std::vector<float> indices_to_codes(const std::vector<int64_t>& indices,
                                        bool project = true) const {
        const int N = (int)indices.size();
        if (!project || !has_projections) {
            std::vector<float> out((size_t)N * codebook_dim);
            for (int i = 0; i < N; ++i) index_to_code(indices[i], &out[(size_t)i * codebook_dim]);
            return out;
        }
        assert(!proj_out_w.empty() && "project_out weights not loaded");
        std::vector<float> out((size_t)N * out_dim);
        std::vector<float> code(codebook_dim);
        for (int i = 0; i < N; ++i) {
            index_to_code(indices[i], code.data());
            project_out(code.data(), &out[(size_t)i * out_dim]);
        }
        return out;
    }
};

} // namespace rig
