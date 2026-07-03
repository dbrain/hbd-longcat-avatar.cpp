#ifndef __VACE_HPP__
#define __VACE_HPP__

#include <cmath>
#include <cstring>
#include <vector>

#include "ggml_extend.hpp"
#include "model/vae/wan_vae.hpp"

// ---------------------------------------------------------------------------
// Wan-VACE control-tensor construction (the only NEW input the Wan DiT needs for
// VACE — the DiT branch itself is already in wan.hpp: VaceWanAttentionBlock +
// vace_patch_embedding(vace_in_dim=96 -> dim) + vace_context/vace_strength threaded
// through WanRunner::compute, auto-detected from `vace_blocks.*` in the GGUF).
//
// Mirrors the official Wan pipeline (wan/vace.py / WanVace.prepare_source):
//   vace_encode_frames: inactive = frames*(1-mask), reactive = frames*mask;
//     VAE-encode EACH -> [16,T_lat,H_lat,W_lat] -> concat channel = 32ch.
//   vace_encode_masks: pixel mask [1,T,H,W] -> space-to-depth by the VAE spatial
//     stride (8x8 -> 64 channels) -> temporal-resample to T_lat = 64ch.
//   vace_latent: concat([frames_latent(32), mask_latent(64)]) = 96ch.
//
// ASSUMPTIONS to verify vs wan/vace.py before the first real render:
//   * inactive=*(1-mask) / reactive=*mask sign (mask: 1 = generate/reactive, 0 = keep).
//   * mask temporal downsample uses nearest (reference: F.interpolate nearest-exact).
//   * VAE latents normalized to diffusion space (mu-mean)/std (our codebase convention;
//     VACE may feed raw VAE space — flip vae_mu_to_diffusion if the oracle disagrees).
//   * no ref-image prepend here (continuation uses the context frames themselves; add a
//     ref slot along T if doing reference-guided VACE).
namespace VACE {

    // VAE-encode a pixel video [W,H,T,3] in [0,1] -> diffusion latent [W,H,T_lat,16].
    // (5D input [W,H,T,3,1] so the VAE _compute does NOT unsqueeze the 4D path; per-channel
    // (mu-mean)/std with the caller's Wan VAE constants.)
    static inline sd::Tensor<float> vae_encode_video(WAN::WanVAERunner* vae, int n_threads,
                                                     const sd::Tensor<float>& px,
                                                     const sd_tiling_params_t& tiling,
                                                     int vae_scale,
                                                     const float* vae_mean, const float* vae_std) {
        sd::Tensor<float> in = px;
        in.reshape_({px.shape()[0], px.shape()[1], px.shape()[2], 3, 1});
        auto mu = vae->encode(n_threads, in, tiling, /*circular_x=*/false, /*circular_y=*/false);
        int64_t Wl = mu.shape()[0], Hl = mu.shape()[1];
        int64_t Tlat = (px.shape()[2] - 1) / vae_scale + 1;
        mu.reshape_({Wl, Hl, Tlat, 16});
        float* d = mu.data();
        int64_t plane = Wl * Hl * Tlat;
        for (int64_t c = 0; c < 16; ++c)
            for (int64_t i = 0; i < plane; ++i)
                d[c * plane + i] = (d[c * plane + i] - vae_mean[c]) / vae_std[c];
        return mu;  // [Wl,Hl,Tlat,16] diffusion space
    }

    // Build the 96ch VACE control latent for one window.
    //   control_px: [W,H,T,3] in [0,1] — context (kept) frames hold real content; the
    //               to-generate frames hold a gray (0.5) placeholder.
    //   mask:       [W,H,T,1] — 1 = generate (reactive), 0 = keep (inactive).
    //   vae_scale=4 (temporal), vae_spatial=8.
    //   Returns vace_context [W_lat,H_lat,T_lat,96] = inactive_lat(16) ++ reactive_lat(16) ++ mask_s2d(64),
    //   ready to feed WanRunner::compute(..., vace_context, vace_strength).
    static inline sd::Tensor<float> build_vace_context(WAN::WanVAERunner* vae, int n_threads,
                                                       const sd::Tensor<float>& control_px,
                                                       const sd::Tensor<float>& mask,
                                                       const sd_tiling_params_t& tiling,
                                                       int vae_scale, int vae_spatial,
                                                       const float* vae_mean, const float* vae_std) {
        int64_t W = control_px.shape()[0], H = control_px.shape()[1], T = control_px.shape()[2];
        int64_t Wl = W / vae_spatial, Hl = H / vae_spatial;
        int64_t Tlat = (T - 1) / vae_scale + 1;
        int64_t plane = W * H;

        // inactive = control*(1-mask); reactive = control*mask  (mask broadcast over channel).
        sd::Tensor<float> inactive({W, H, T, 3}), reactive({W, H, T, 3});
        {
            const float* c = control_px.data();
            const float* m = mask.data();
            float* ia = inactive.data();
            float* re = reactive.data();
            for (int64_t ch = 0; ch < 3; ++ch)
                for (int64_t t = 0; t < T; ++t)
                    for (int64_t i = 0; i < plane; ++i) {
                        int64_t ci = (ch * T + t) * plane + i;
                        float mv   = m[t * plane + i];  // mask [W,H,T,1]
                        ia[ci]     = c[ci] * (1.0f - mv);
                        re[ci]     = c[ci] * mv;
                    }
        }
        auto ia_lat = vae_encode_video(vae, n_threads, inactive, tiling, vae_scale, vae_mean, vae_std);  // [Wl,Hl,Tlat,16]
        auto re_lat = vae_encode_video(vae, n_threads, reactive, tiling, vae_scale, vae_mean, vae_std);  // [Wl,Hl,Tlat,16]

        // mask space-to-depth: pixel (xl*8+sx, yl*8+sy) -> channel sy*8+sx; temporal nearest T->T_lat.
        int64_t plane_l = Wl * Hl;
        int64_t n_ch    = (int64_t)vae_spatial * vae_spatial;  // 64
        sd::Tensor<float> ms({Wl, Hl, Tlat, n_ch});
        {
            const float* m = mask.data();
            float* d       = ms.data();
            for (int64_t tl = 0; tl < Tlat; ++tl) {
                int64_t ts = Tlat == 1 ? 0 : (int64_t)std::llround((double)tl * (T - 1) / (Tlat - 1));
                if (ts < 0) ts = 0;
                if (ts >= T) ts = T - 1;
                for (int sy = 0; sy < vae_spatial; ++sy)
                    for (int sx = 0; sx < vae_spatial; ++sx) {
                        int64_t ch = (int64_t)sy * vae_spatial + sx;
                        for (int64_t yl = 0; yl < Hl; ++yl)
                            for (int64_t xl = 0; xl < Wl; ++xl) {
                                int64_t xp = xl * vae_spatial + sx;
                                int64_t yp = yl * vae_spatial + sy;
                                d[(ch * Tlat + tl) * plane_l + yl * Wl + xl] = m[ts * plane + yp * W + xp];
                            }
                    }
            }
        }

        // concat along channel (c-slowest layout): [ia(16) | re(16) | ms(64)] = 96ch.
        int64_t plt = plane_l * Tlat;
        sd::Tensor<float> vc({Wl, Hl, Tlat, 96});
        memcpy(vc.data() + 0 * 16 * plt, ia_lat.data(), (size_t)16 * plt * sizeof(float));
        memcpy(vc.data() + 1 * 16 * plt, re_lat.data(), (size_t)16 * plt * sizeof(float));
        memcpy(vc.data() + 2 * 16 * plt, ms.data(), (size_t)n_ch * plt * sizeof(float));
        return vc;  // [Wl,Hl,Tlat,96]
    }

    // Convenience: a continuation control video + mask. The first `n_ctx` pixel frames are
    // the carried-over context (real content), the remaining (frame_num - n_ctx) are gray
    // placeholders to be generated. mask = 0 on context, 1 on the rest.
    static inline void make_continuation_source(const sd::Tensor<float>& ctx_frames /*[W,H,n_ctx,3]*/,
                                                int frame_num, float gray,
                                                sd::Tensor<float>& control_px, sd::Tensor<float>& mask) {
        int64_t W = ctx_frames.shape()[0], H = ctx_frames.shape()[1];
        int64_t n_ctx = ctx_frames.shape()[2];
        int64_t plane = W * H;
        control_px = sd::Tensor<float>({W, H, (int64_t)frame_num, 3});
        mask       = sd::Tensor<float>({W, H, (int64_t)frame_num, 1});
        std::fill(control_px.data(), control_px.data() + control_px.numel(), gray);
        float* cp = control_px.data();
        const float* cf = ctx_frames.data();
        for (int64_t ch = 0; ch < 3; ++ch)
            for (int64_t t = 0; t < n_ctx; ++t)
                memcpy(cp + (ch * frame_num + t) * plane, cf + (ch * n_ctx + t) * plane, plane * sizeof(float));
        float* m = mask.data();
        for (int64_t t = 0; t < frame_num; ++t) {
            float mv = (t < n_ctx) ? 0.0f : 1.0f;
            for (int64_t i = 0; i < plane; ++i) m[t * plane + i] = mv;
        }
    }

}  // namespace VACE

#endif  // __VACE_HPP__
