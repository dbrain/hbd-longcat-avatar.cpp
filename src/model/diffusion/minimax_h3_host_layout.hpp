#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_HOST_LAYOUT_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_HOST_LAYOUT_HPP__

#include <algorithm>
#include <cstdint>

#include "core/tensor.hpp"
#include "ggml.h"

// Host-side twins of the four MiniMax-H3 layout permutations that the DiT does with ggml ops.
//
// They live in their own header, free of every runner and every backend, so
// tests/minimax_h3_permutation_test.cpp can compare THE SHIPPING CODE -- not a copy of it --
// against the PR's reference implementation under real torch.  A retyped twin in a test proves
// nothing; see tools/ref_permute_torch.py.
//
// ⚠️ TWO audio layouts exist and they are NOT the same tensor:
//   * VAE layout      {T, latent_channels, stereo}  -- what MiniMaxH3AudioVAERunner encodes/decodes
//                                                      (torch [S, C, T], stereo carried as batch)
//   * request layout  {T, stereo, latent_channels}  -- what MiniMaxH3Request::audio_x and the
//                                                      runner's pack_audio() read
//                                                      (torch [B, C, S, T], comfy's audio-VAE
//                                                      encode() output after its permute(0,2,1,3))
// They differ by one transposition, they have identical element counts, and swapping them does
// not error -- it silently smears the stereo field across the latent channels.  Convert only
// through minimax_h3_swap_audio_axes().

// {T, C, S} -> {T, S, C}, and its own inverse. permute() materialises, so the result is a plain
// contiguous tensor either way.
inline sd::Tensor<float> minimax_h3_swap_audio_axes(const sd::Tensor<float>& audio) {
    if (audio.empty()) {
        return {};
    }
    GGML_ASSERT(audio.dim() == 3 || audio.dim() == 4);
    if (audio.dim() == 4) {
        GGML_ASSERT(audio.shape()[3] == 1);
        return audio.permute({0, 2, 1, 3});
    }
    return audio.permute({0, 2, 1});
}

// Patchify a video latent into H3 conditioning rows.
//
// Host-side twin of DiT::patchify_3d(x, pt=1, ph=2, pw=2, patch_last=true), which is what the
// runner applies to the TARGET latent -- conditioning rows have to be in the identical row and
// column order or the DiT reads a keyframe as a differently-scanned image and still renders.
//   in : [W, H, T, C(, 1)] diffusion latent
//   out: [C*4, T*(H/2)*(W/2)] rows, (t, h, w) major, column ((c*2 + p)*2 + q)
inline sd::Tensor<float> minimax_h3_patchify_cond_rows(const sd::Tensor<float>& latent) {
    if (latent.empty()) {
        return {};
    }
    GGML_ASSERT(latent.dim() == 4 || latent.dim() == 5);
    if (latent.dim() == 5) {
        GGML_ASSERT(latent.shape()[4] == 1);
    }
    const int64_t W = latent.shape()[0];
    const int64_t H = latent.shape()[1];
    const int64_t T = latent.shape()[2];
    const int64_t C = latent.shape()[3];
    // The 2x2 patch is not negotiable: an odd latent axis would have to be padded, and the
    // layout would then have to be told the PADDED size or it describes a different grid.
    GGML_ASSERT(W % 2 == 0 && H % 2 == 0);

    const int64_t patch_dim = C * 4;
    const int64_t rows      = T * (H / 2) * (W / 2);
    sd::Tensor<float> out({patch_dim, rows});
    const float* src = latent.data();
    float* dst       = out.data();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t hb = 0; hb < H / 2; ++hb) {
            for (int64_t wb = 0; wb < W / 2; ++wb) {
                const int64_t row = (t * (H / 2) + hb) * (W / 2) + wb;
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t p = 0; p < 2; ++p) {
                        for (int64_t q = 0; q < 2; ++q) {
                            const int64_t col = (c * 2 + p) * 2 + q;
                            dst[row * patch_dim + col] =
                                src[((c * T + t) * H + (hb * 2 + p)) * W + (wb * 2 + q)];
                        }
                    }
                }
            }
        }
    }
    return out;
}

// Host-side twin of MiniMaxH3Runner::pack_audio().
//   in : {T, stereo, C} request-layout audio latent
//   out: [C, stereo*T] rows -- CHANNEL-MAJOR (all of channel 0, then all of channel 1).
// Interleaving instead does not error, it destroys the stereo field; same class of bug as the
// LTX `seg_*.audio` interleave.
inline sd::Tensor<float> minimax_h3_pack_audio_cond_rows(const sd::Tensor<float>& audio) {
    if (audio.empty()) {
        return {};
    }
    GGML_ASSERT(audio.dim() >= 3);
    const int64_t T = audio.shape()[0];
    const int64_t S = audio.shape()[1];
    const int64_t C = audio.shape()[2];
    sd::Tensor<float> out({C, S * T});
    const float* src = audio.data();
    float* dst       = out.data();
    for (int64_t s = 0; s < S; ++s) {
        for (int64_t t = 0; t < T; ++t) {
            const int64_t row = s * T + t;
            for (int64_t c = 0; c < C; ++c) {
                dst[row * C + c] = src[c * (T * S) + s * T + t];
            }
        }
    }
    return out;
}

// Peel this step's target audio latent back out of the packed [W, H, T, 24 + extra] tensor the
// sampler walks. Mirror image of pack_ltxav_audio_and_video_latents' flat append.
inline sd::Tensor<float> minimax_h3_unpack_audio_latent(const sd::Tensor<float>& packed,
                                                        int audio_t,
                                                        int video_channels,
                                                        int audio_latent_channels) {
    if (packed.empty() || audio_t <= 0) {
        return {};
    }
    GGML_ASSERT(packed.dim() == 4 || packed.dim() == 5);
    const int64_t spatial = packed.shape()[0] * packed.shape()[1] * packed.shape()[2];
    const int64_t total   = packed.shape()[3];
    const int64_t values  = static_cast<int64_t>(audio_t) * 2 * audio_latent_channels;
    if (total <= video_channels || (total - video_channels) * spatial < values) {
        return {};
    }
    sd::Tensor<float> audio({audio_t, 2, audio_latent_channels, 1});
    std::copy_n(packed.data() + static_cast<size_t>(video_channels) * static_cast<size_t>(spatial),
                static_cast<size_t>(values),
                audio.data());
    return audio;
}

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_HOST_LAYOUT_HPP__
