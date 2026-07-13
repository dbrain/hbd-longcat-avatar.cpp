#ifndef __SD_MODEL_VAE_WAN_VAE_HPP__
#define __SD_MODEL_VAE_WAN_VAE_HPP__

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "model/common/block.hpp"
#include "model/vae/vae.hpp"
#include "model_loader.h"

namespace WAN {

    constexpr int CACHE_T = 2;

    // WAN_VAE_F16 (default OFF; F32 byte-identical when unset): run the VAE DECODE activation
    // stream in F16. The Wan VAE weights are already F16; only the activations were F32. Two
    // payoffs: (1) it halves the chunk-1+ temporal-streaming peak (the 4x temporal upsample's
    // ~14GB F32 intermediate -> ~7GB) so 1x1 zero-seam decode fits <=11.5GB, and (2) less HBM
    // traffic = a faster VAE decode. Mirrors WAN_DIT_F16 (wan.hpp): cast the residual stream
    // to F16 at the decode entry, back to F32 before the output, with F32 islands only where
    // an op's ggml dtype would force a re-widen or an unsafe mixed-dtype op:
    //   * CausalConv3d (cuDNN): F16-native — conv3d-cudnn.cu accepts F16 src/dst, ggml.c
    //     ggml_conv_3d_direct emits an F16 result for an F16 input (the heavy conv-boundary
    //     activations, incl. the temporal-upsample intermediate, are the win).
    //   * RMS_norm / SiLU / upscale / pad / concat / residual-add: F16-native (supports_op).
    //   * Conv2d (im2col + ggml_mul_mat): mul_mat hardcodes an F32 dst, so 1x1/3x3 convs
    //     (attention to_qkv/proj, resample.1) emit F32 — cast back to F16 right after.
    //   * AttentionBlock (middle/bottleneck, small): a whole-block F32 island (cast in/out)
    //     so its F32 Conv2d outputs and the `add(proj, identity)` never form the unsafe
    //     binbcast combo add(F32 src0, F16 src1); attention numerics stay exactly as prod.
    // Encode is untouched (stays F32, spatially tiled).
    inline bool wan_vae_f16_enabled() {
        static const bool on = (std::getenv("WAN_VAE_F16") != nullptr);
        return on;
    }

    // WAN_VAE_HEAD_F32 (default ON; set WAN_VAE_HEAD_F32=0 to reproduce the grid): keep the final
    // decoder conv `head.2` on the cuDNN implicit-GEMM path (GGML_CUDNN_CONV3D) but request an
    // F32-IO plan so cuDNN writes an fp32 output instead of fp16.
    //
    // The bug it fixes: cuDNN's conv3d output Y is normally fp16 (conv3d-cudnn.cu: y_ndhwc is a
    // ggml_cuda_pool_alloc<half>), so the conv output is fp16-quantized regardless of the ggml
    // dst dtype. head.2 emits 12 channels (wan2.2) of slightly different magnitude, and decode
    // does unpatchify(out, 2) — a space-to-depth that keys each output pixel's RGB to a specific
    // one of those 12 channels by (x%2, y%2). The per-channel fp16 quantization steps therefore
    // land as a faint ~2px screen-door grid on photoreal content.
    //
    // The fix routes head.2 (only) through a full F32-IO cuDNN plan (fp32 X/W/Y, fp32 accumulate,
    // fp32 store), so the output is never fp16-quantized and the per-channel grid is gone. (A
    // mixed HALF-in/FLOAT-out plan was silently rejected by cuDNN's heuristic for this 3D shape,
    // reverting to fp16; full fp32 IO is the standard, broadly-supported conv.) It stays on the
    // implicit-GEMM engine: there is NO im2col IC*27 column
    // materialization. (Routing head.2 to im2col OOMs — 256 input channels at ~640x352 => a
    // [256*27, 640*352] ~3GB buffer, the exact blowup cuDNN conv3d exists to avoid.) Only head.2's
    // tiny 12-out-channel F32 output buffer grows, a bounded cost well under the decode peak
    // (dominated by the temporal-upsample intermediate), so the ~94s decode is preserved.
    //
    // No effect when GGML_CUDNN_CONV3D is off (that path is already F32 im2col). The fix also
    // forces head.2's ggml output tensor to F32 so it holds under WAN_VAE_F16 (F16-activation
    // decode) too. WAN_VAE_HEAD_F32=0 puts head.2 back on the cuDNN fp16 plan to A/B the grid.
    inline bool wan_vae_head_f32_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("WAN_VAE_HEAD_F32");
            return e == nullptr || atoi(e) != 0;  // default ON; only "0" disables
        }();
        return on;
    }

    // WAN_VAE_RESAMPLE_TSPLIT (default OFF; byte-identical committed graph when unset): run each
    // upsample/downsample `resample.1` Conv2d one temporal frame at a time instead of as a single
    // ne[3]-batched conv. resample.1 is per-frame independent, so this is numerically identical
    // but bounds the live im2col column buffer (the whole-frame decode's peak transient, ~1.4 GB
    // F32 at the final 480x832 upsample) to a single frame — cutting the compute-buffer peak by
    // ~(T-1)/T of the im2col at no precision cost. Correct+fast alternative to conv2d-direct
    // (which overflows in F16 and hits the naive kernel in this build).
    inline bool wan_vae_resample_tsplit_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("WAN_VAE_RESAMPLE_TSPLIT");
            return e != nullptr && atoi(e) != 0;  // default OFF; "1" enables
        }();
        return on;
    }

    // WAN_VAE_RMS_CF (default OFF; VALUE-identical, committed graph byte-identical when unset):
    // cut the #1 decode data-movement cost — the plain F16->F16 `cpy_scalar` copies (nsys: 25.7%
    // of the whole-frame decode). Their dominant source is RMS_norm: it normalizes over the
    // channel dim, which in the [W,H,T,C] activation layout is ne[3]. ggml_rms_norm only reduces
    // over ne[0], so RMS_norm does permute(C->dim0)+CONT, rms, mul(gamma), then permute-back+CONT
    // — TWO ggml_cont materializations per call. There are ~30 RMS_norm per decoded latent frame
    // (every ResidualBlock has 2, plus the heads), so at ~21 frames that permute-back CONT alone
    // is ~600 of the ~1800 cpy_scalar instances.
    //
    // The permute-back CONT (call it CONT#2) is redundant: RMS_norm is ALWAYS followed by SiLU
    // (residual.1/residual.4, head.1), an elementwise op whose result is independent of memory
    // layout, and SiLU is ALWAYS followed by a CausalConv3d whose first op (ggml_pad / ggml_concat
    // for the causal cache) reads its src BY STRIDES (pad.cu / concat.cu non-contiguous path) and
    // writes a fresh contiguous buffer. So we can keep the RMS_norm result in its channels-first
    // [C,W,H,T] contiguous layout, run SiLU there (identical values), then hand the conv a
    // zero-copy permuted [W,H,T,C] VIEW — the conv's pad/concat materializes it exactly as CONT#2
    // used to, but ONCE instead of twice. No new kernels, no dtype change: the numbers are
    // bit-identical to the gate-off path; only ~one cont per RMS->SiLU->conv chain disappears.
    //
    // Scoped to the DECODE phase (g_ext_vae_phase_encode==false) so the F32/tiled encode graph is
    // literally untouched. RMS_norm feeding an AttentionBlock (norm) is left on the plain path
    // (its consumer is a cont'd permute, not SiLU), so attention is unaffected.
    inline bool wan_vae_rms_cf_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("WAN_VAE_RMS_CF");
            return e != nullptr && atoi(e) != 0;  // default OFF; "1" enables
        }();
        return on;
    }

    // WAN_VAE_SLICE_NOCOPY (default OFF; committed graph byte-identical when unset): the per-frame
    // decode loop slices one latent frame `in = slice(x, dim=2, i, i+1)` and ggml_ext_slice conts
    // it by default. That cont is redundant: `in`'s only consumer is decoder conv1, a CausalConv3d
    // whose first op (ggml_pad when historyless, else ggml_concat with the causal cache) reads its
    // src by strides and writes contiguous. Passing the strided view (cont=false) lets that pad/
    // concat do the single materialization. Values identical; saves one small cont per frame.
    inline bool wan_vae_slice_nocopy_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("WAN_VAE_SLICE_NOCOPY");
            return e != nullptr && atoi(e) != 0;  // default OFF; "1" enables
        }();
        return on;
    }

    // WAN_VAE_RMS_KERNEL (default OFF; committed graph byte-identical when unset): the real fix for
    // the RMS_norm data-movement cost. RMS_norm normalizes over the CHANNEL dim, which is ne[3] in
    // the [W,H,T,C] activation layout; ggml_rms_norm only reduces over ne[0], so the committed path
    // does permute(C->ne0)+CONT, rms, mul(gamma), permute-back+CONT — 2 conts + a separate mul per
    // RMS_norm (~30/decoded-frame, the #1 `cpy_scalar` source). This routes RMS_norm to the custom
    // ggml_rms_norm_channels op (ggml-cuda/norm.cu): ONE coalesced kernel that reduces over ne[3]
    // and folds gamma in place, reading [W,H,T,C] natively. Both conts AND the mul disappear —
    // unlike WAN_VAE_RMS_CF (which only relocated CONT#2 into SiLU), this removes the work. Formula-
    // identical (float reduction, invisible at 8-bit). Decode-only; AttentionBlock RMS and the
    // F32/tiled encode stay on the plain ggml_rms_norm. Takes precedence over WAN_VAE_RMS_CF.
    inline bool wan_vae_rms_kernel_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("WAN_VAE_RMS_KERNEL");
            return e != nullptr && atoi(e) != 0;  // default OFF; "1" enables
        }();
        return on;
    }

    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> padding;
        std::tuple<int, int, int> dilation;
        bool bias;
        bool force_f32;  // WAN_VAE_HEAD_F32: force this conv onto the clean F32 im2col path

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["weight"] = ggml_new_tensor_4d(ctx,
                                                  GGML_TYPE_F16,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride   = {1, 1, 1},
                     std::tuple<int, int, int> padding  = {0, 0, 0},
                     std::tuple<int, int, int> dilation = {1, 1, 1},
                     bool bias                          = true,
                     bool force_f32                     = false)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(std::move(kernel_size)),
              stride(std::move(stride)),
              padding(std::move(padding)),
              dilation(std::move(dilation)),
              bias(bias),
              force_f32(force_f32) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* cache_x = nullptr) {
            // x: [N*IC, ID, IH, IW]
            // result: x: [N*OC, ID, IH, IW]
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = nullptr;
            if (bias) {
                b = params["bias"];
            }

            int lp0 = std::get<2>(padding);
            int rp0 = std::get<2>(padding);
            int lp1 = std::get<1>(padding);
            int rp1 = std::get<1>(padding);
            int lp2 = 2 * std::get<0>(padding);
            int rp2 = 0;

            if (cache_x != nullptr && lp2 > 0) {
                x = ggml_concat(ctx->ggml_ctx, cache_x, x, 2);
                lp2 -= (int)cache_x->ne[2];
            }

            x = ggml_ext_pad_ext(ctx->ggml_ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            // Convs tagged force_f32 (head.2 under WAN_VAE_HEAD_F32) stay on the cuDNN
            // implicit-GEMM op (no im2col IC*27 blowup) but request an F32-IO plan so cuDNN writes
            // fp32, not fp16 -> the per-channel fp16 steps don't become an unpatchify grid.
            bool cudnn_hi_prec = force_f32 && wan_vae_head_f32_enabled();
            return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                    std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                    0, 0, 0,
                                    std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                    /*force_prec_f32=*/false, /*cudnn_hi_prec=*/cudnn_hi_prec);
        }
    };

    class RMS_norm : public UnaryBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            auto iter       = tensor_storage_map.find(prefix + "gamma");
            if (iter != tensor_storage_map.end()) {
                params["gamma"] = ggml_new_tensor(ctx, wtype, iter->second.n_dims, &iter->second.ne[0]);
            } else {
                params["gamma"] = ggml_new_tensor_1d(ctx, wtype, dim);
            }
        }

    public:
        RMS_norm(int64_t dim)
            : dim(dim) {}

        // Channels-first result [N*IC, IW, IH, ID] i.e. ggml ne=[C, W, H, T], contiguous. This is
        // exactly `forward()` minus the final permute-back+CONT. rms_norm/mul are unchanged, so the
        // VALUES are identical to forward()'s output re-laid-out; only the layout differs. See
        // WAN_VAE_RMS_CF: the caller runs the (layout-agnostic) SiLU here, then hands the conv the
        // zero-copy view from channels_first_to_whtc_view() so the conv's pad/concat does the one
        // materialization that forward()'s CONT would have done.
        ggml_tensor* forward_channels_first(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["gamma"];
            w              = ggml_reshape_1d(ctx->ggml_ctx, w, ggml_nelements(w));
            auto h         = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));  // [C, W, H, T]
            h              = ggml_rms_norm(ctx->ggml_ctx, h, 1e-12f);
            h              = ggml_mul(ctx->ggml_ctx, h, w);
            return h;  // [C, W, H, T] contiguous (channels-first)
        }

        // Zero-copy inverse of forward_channels_first's leading permute: [C,W,H,T] -> a strided
        // [W,H,T,C] view (no cont). The downstream CausalConv3d's pad/concat reads it by strides.
        static ggml_tensor* channels_first_to_whtc_view(GGMLRunnerContext* ctx, ggml_tensor* h) {
            return ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0);
        }

        // WAN_VAE_RMS_KERNEL: single-op channels-last RMS over ne[3] with gamma folded in — no
        // transpose, no cont, no separate mul. Returns [W,H,T,C] contiguous, same as forward().
        ggml_tensor* forward_channels_kernel(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["gamma"];  // [C], F32
            w              = ggml_reshape_1d(ctx->ggml_ctx, w, ggml_nelements(w));
            return ggml_rms_norm_channels(ctx->ggml_ctx, x, w, 1e-12f);  // normalize over ne[3], fold gamma
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // x: [N*IC, ID, IH, IW], IC == dim
            // assert N == 1

            auto h = forward_channels_first(ctx, x);
            h      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));

            return h;
        }
    };

    class Resample : public GGMLBlock {
    protected:
        int64_t dim;
        std::string mode;

    public:
        Resample(int64_t dim, const std::string& mode, bool wan2_2 = false)
            : dim(dim), mode(mode) {
            if (mode == "upsample2d") {
                if (wan2_2) {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {1, 1}, {1, 1}));
                } else {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                }
            } else if (mode == "upsample3d") {
                if (wan2_2) {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {1, 1}, {1, 1}));
                } else {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                }
                blocks["time_conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim * 2, {3, 1, 1}, {1, 1, 1}, {1, 0, 0}));
            } else if (mode == "downsample2d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
            } else if (mode == "downsample3d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
                blocks["time_conv"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim, {3, 1, 1}, {2, 1, 1}, {0, 0, 0}));
            } else if (mode == "none") {
                // nn.Identity()
            } else {
                GGML_ASSERT(false && "invalid mode");
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            int64_t c = x->ne[3] / b;
            int64_t t = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            // WAN_VAE_F16 (decode/upsample): the heavy ops here (time_conv CausalConv3d, the
            // 2x spatial ggml_upscale, the temporal-doubling cont) stay F16; only resample.1
            // (Conv2d -> ggml_mul_mat) re-widens to F32. Remember the stream dtype so we can
            // restore F16 before returning (the caller's residual add(x, shortcut) needs both
            // operands F16). No-op on the F32 encode/downsample path.
            const bool stream_f16 = (x->type == GGML_TYPE_F16);

            if (mode == "upsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    feat_idx += 1;
                    if (chunk_idx == 0) {
                        // feat_cache[idx] == nullptr, pass
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {  // chunk_idx >= 2
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }
                        if (chunk_idx == 1 && cache_x->ne[2] < 2) {  // Rep
                            cache_x = ggml_pad_ext(ctx->ggml_ctx, cache_x, 0, 0, 0, 0, (int)cache_x->ne[2], 0, 0, 0);
                            // aka cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device),cache_x],dim=2)
                        }
                        if (chunk_idx == 1) {
                            x = time_conv->forward(ctx, x);
                        } else {
                            x = time_conv->forward(ctx, x, feat_cache[idx]);
                        }
                        feat_cache[idx] = cache_x;
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w * h, t, c, 2);                                   // (2, c, t, h*w)
                        x               = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 3, 1, 2));  // (c, t, 2, h*w)
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, 2 * t, c);                                   // (c, t*2, h, w)
                    }
                }
            }

            t = x->ne[2];
            if (mode != "none") {
                auto resample_1 = std::dynamic_pointer_cast<Conv2d>(blocks["resample.1"]);

                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)
                if (mode == "upsample2d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "upsample3d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "downsample2d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                } else if (mode == "downsample3d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                }
                // resample.1 (Conv2d) is the whole-frame decode's peak transient. It goes through
                // ggml_conv_2d = im2col (asserts F32 src1, im2col.cu:87) + mul_mat (F32), so under
                // WAN_VAE_F16 the input is widened to F32 and the im2col column buffer (IC*9 x
                // spatial x the temporal-batch ne[3]) is the single largest F32 buffer in the
                // decode — ~1.4 GB at the final 480x832 upsample where ne[3] is the full ~4-frame
                // temporal batch. resample.1 is a 2D conv (batched over ne[3], per-frame
                // INDEPENDENT), so looping it one temporal frame at a time is numerically IDENTICAL
                // yet caps the live im2col to a SINGLE frame — gallocr reuses that scratch across
                // the loop, cutting the peak by ~(T-1)/T of the im2col (~1 GB at the last upsample)
                // with no precision change. WAN_VAE_RESAMPLE_TSPLIT gates it (default OFF ->
                // committed graph byte-identical). The per-frame F32 widen keeps the F32-accumulate
                // im2col (no F16 conv overflow).
                if (wan_vae_resample_tsplit_enabled() && x->ne[3] > 1) {
                    const int64_t T = x->ne[3];
                    ggml_tensor* acc = nullptr;
                    for (int64_t k = 0; k < T; k++) {
                        ggml_tensor* xf = ggml_ext_slice(ctx->ggml_ctx, x, 3, k, k + 1);  // [w,h,c,1]
                        if (stream_f16 && xf->type == GGML_TYPE_F16) {
                            xf = ggml_cast(ctx->ggml_ctx, xf, GGML_TYPE_F32);  // widen + make contiguous
                        } else if (xf->view_src != nullptr || !ggml_is_contiguous(xf)) {
                            xf = ggml_ext_cont(ctx->ggml_ctx, xf);
                        }
                        xf  = resample_1->forward(ctx, xf);
                        acc = (acc == nullptr) ? xf : ggml_concat(ctx->ggml_ctx, acc, xf, 3);
                    }
                    x = acc;
                } else {
                    if (stream_f16 && x->type == GGML_TYPE_F16) {
                        x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
                    }
                    x = resample_1->forward(ctx, x);
                }
                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)
            }

            if (mode == "downsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    if (feat_cache[idx] == nullptr) {
                        feat_cache[idx] = x;
                        feat_idx += 1;
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x    = ggml_ext_slice(ctx->ggml_ctx, x, 2, -1, x->ne[2]);
                        x               = ggml_concat(ctx->ggml_ctx,
                                                      ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                      x,
                                                      2);
                        x               = time_conv->forward(ctx, x);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                }
            }

            // Restore the F16 residual stream (resample.1 emitted F32 via mul_mat).
            if (stream_f16 && x->type != GGML_TYPE_F16) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F16);
            }
            return x;
        }
    };

    class AvgDown3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int factor_t;
        int factor_s;
        int factor;
        int64_t group_size;

    public:
        AvgDown3D(int64_t in_channels, int64_t out_channels, int factor_t, int factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(in_channels * factor % out_channels == 0);
            group_size = in_channels * factor / out_channels;
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t B = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            int pad_t = (factor_t - T % factor_t) % factor_t;

            x = ggml_pad_ext(ctx->ggml_ctx, x, 0, 0, 0, 0, pad_t, 0, 0, 0);
            T = x->ne[2];

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, factor_t, T / factor_t, C);                                                  // [C, T/factor_t, factor_t, H*W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C, factor_t, T/factor_t, H*W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, factor_s, (H / factor_s) * (T / factor_t), factor_t * C);                        // [C*factor_t, T/factor_t*H/factor_s, factor_s, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C*factor_t, factor_s, T/factor_t*H/factor_s, W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s, W / factor_s, (H / factor_s) * (T / factor_t), factor_s * factor_t * C);  // [C*factor_t*factor_s, T/factor_t*H/factor_s, W/factor_s, factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));                                       // [C*factor_t*factor_s, factor_s, T/factor_t*H/factor_s, W/factor_s]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, (W / factor_s) * (H / factor_s) * (T / factor_t), group_size, out_channels);        // [out_channels, group_size, T/factor_t*H/factor_s*W/factor_s]

            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [out_channels, T/factor_t*H/factor_s*W/factor_s, group_size]
            x = ggml_mean(ctx->ggml_ctx, x);                                                     // [out_channels, T/factor_t*H/factor_s*W/factor_s, 1]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W / factor_s, H / factor_s, T / factor_t, out_channels);
            return x;
        }
    };

    class DupUp3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t factor_t;
        int64_t factor_s;
        int64_t factor;
        int64_t repeats;

    public:
        DupUp3D(int64_t in_channels, int64_t out_channels, int64_t factor_t, int64_t factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(out_channels * factor % in_channels == 0);
            repeats = out_channels * factor / in_channels;
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool first_chunk = false,
                             int64_t B        = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            auto x_ = x;
            for (int64_t i = 1; i < repeats; i++) {
                x = ggml_concat(ctx->ggml_ctx, x, x_, 2);
            }

            C = out_channels;

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, factor_s, factor_s * factor_t * C);  // [C*factor_t*factor_s, factor_s, T*H, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [C*factor_t*factor_s, T*H, W, factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, H * T, factor_s, factor_t * C);  // [C*factor_t, factor_s, T*H, W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C*factor_t, T*H, factor_s, W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W * factor_s * H, T, factor_t, C);  // [C, factor_t, T, H*factor_s*W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C, T, factor_t, H*factor_s*W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, factor_s * H, factor_t * T, C);  // [C, T*factor_t, H*factor_s, W*factor_s]

            if (first_chunk) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 2, factor_t - 1, x->ne[2]);
            }

            return x;
        }
    };

    class ResidualBlock : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t out_dim;

    public:
        ResidualBlock(int64_t in_dim, int64_t out_dim)
            : in_dim(in_dim), out_dim(out_dim) {
            blocks["residual.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(in_dim));
            // residual.1 is nn.SiLU()
            blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["residual.3"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // residual.4 is nn.SiLU()
            // residual.5 is nn.Dropout()
            blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            if (in_dim != out_dim) {
                blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {1, 1, 1}));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* h = x;
            if (in_dim != out_dim) {
                auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["shortcut"]);

                h = shortcut->forward(ctx, x);
            }

            // WAN_VAE_RMS_CF (decode only): keep the RMS_norm result channels-first [C,W,H,T]
            // (drops RMS's permute-back CONT), then let the SiLU that ALWAYS follows absorb the
            // transpose — SiLU reads the zero-copy permuted [W,H,T,C] view and writes it back
            // CONTIGUOUS (ggml-cuda unary strided path). The conv then gets a normal contiguous
            // [W,H,T,C] tensor, so the causal-cache ggml_concat (which requires dim-0-contiguous
            // inputs) and the pad both work. Net: one fewer cont per RMS->SiLU->conv chain, and
            // ggml_concat is never handed a non-contiguous tensor. Value-identical to plain path.
            // `x_is_cf` == x is currently channels-first [C,W,H,T].
            // WAN_VAE_RMS_KERNEL (primary) folds the whole RMS into one channels-last op -> x stays
            // [W,H,T,C] contiguous, no cf machinery. WAN_VAE_RMS_CF (fallback) is the older SiLU-
            // absorb path. Both decode-only.
            const bool rms_kernel = wan_vae_rms_kernel_enabled() && !g_ext_vae_phase_encode;
            const bool rms_cf     = !rms_kernel && wan_vae_rms_cf_enabled() && !g_ext_vae_phase_encode;
            bool x_is_cf          = false;

            for (int i = 0; i < 7; i++) {
                if (i == 0 || i == 3) {  // RMS_norm
                    auto layer = std::dynamic_pointer_cast<RMS_norm>(blocks["residual." + std::to_string(i)]);
                    if (rms_kernel) {
                        x = layer->forward_channels_kernel(ctx, x);  // [W,H,T,C] contiguous, single op
                    } else if (rms_cf) {
                        x        = layer->forward_channels_first(ctx, x);  // [C,W,H,T] contiguous, no permute-back CONT
                        x_is_cf  = true;
                    } else {
                        x = layer->forward(ctx, x);
                    }
                } else if (i == 2 || i == 6) {  // CausalConv3d
                    auto layer = std::dynamic_pointer_cast<CausalConv3d>(blocks["residual." + std::to_string(i)]);

                    // x is already contiguous [W,H,T,C] here (SiLU de-transposed it), so the cache
                    // concat/pad below get contiguous inputs.
                    if (feat_cache.size() > 0) {
                        int idx      = feat_idx;
                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }

                        x               = layer->forward(ctx, x, feat_cache[idx]);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                } else if (i == 1 || i == 4) {
                    // SiLU absorbs the RMS permute-back: read the [C,W,H,T] channels-first result as
                    // a zero-copy [W,H,T,C] view, write it CONTIGUOUS. Plain elementwise otherwise.
                    if (x_is_cf) {
                        x       = RMS_norm::channels_first_to_whtc_view(ctx, x);  // strided [W,H,T,C] view
                        x_is_cf = false;
                    }
                    x = ggml_silu(ctx->ggml_ctx, x);  // strided-in -> contiguous-out (ggml-cuda unary)
                } else {  // i == 5
                    // nn.Dropout(), ignore
                }
            }

            x = ggml_add(ctx->ggml_ctx, x, h);
            return x;
        }
    };

    class Down_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool down_flag;

    public:
        Down_ResidualBlock(int64_t in_dim,
                           int64_t out_dim,
                           int mult,
                           bool temperal_downsample = false,
                           bool down_flag           = false)
            : mult(mult), down_flag(down_flag) {
            blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new AvgDown3D(in_dim, out_dim, temperal_downsample ? 2 : 1, down_flag ? 2 : 1));

            int i = 0;
            for (; i < mult; i++) {
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                     = out_dim;
            }
            if (down_flag) {
                std::string mode                           = temperal_downsample ? "downsample3d" : "downsample2d";
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode, true));
                i++;
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* x_copy = x;

            auto avg_shortcut = std::dynamic_pointer_cast<AvgDown3D>(blocks["avg_shortcut"]);

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (down_flag) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
            }

            auto shortcut = avg_shortcut->forward(ctx, x_copy, b);

            x = ggml_add(ctx->ggml_ctx, x, shortcut);

            return x;
        }
    };

    class Up_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool up_flag;

    public:
        Up_ResidualBlock(int64_t in_dim,
                         int64_t out_dim,
                         int mult,
                         bool temperal_upsample = false,
                         bool up_flag           = false)
            : mult(mult), up_flag(up_flag) {
            if (up_flag) {
                blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new DupUp3D(in_dim, out_dim, temperal_upsample ? 2 : 1, up_flag ? 2 : 1));
            }

            int i = 0;
            for (; i < mult; i++) {
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                   = out_dim;
            }
            if (up_flag) {
                std::string mode                         = temperal_upsample ? "upsample3d" : "upsample2d";
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode, true));
                i++;
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* x_copy = x;

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (up_flag) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);

                auto avg_shortcut = std::dynamic_pointer_cast<DupUp3D>(blocks["avg_shortcut"]);
                auto shortcut     = avg_shortcut->forward(ctx, x_copy, chunk_idx == 0, b);

                x = ggml_add(ctx->ggml_ctx, x, shortcut);
            }

            return x;
        }
    };

    class AttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

    public:
        AttentionBlock(int64_t dim)
            : dim(dim) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new RMS_norm(dim));
            blocks["to_qkv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim * 3, {1, 1}));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {1, 1}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto norm   = std::dynamic_pointer_cast<RMS_norm>(blocks["norm"]);
            auto to_qkv = std::dynamic_pointer_cast<Conv2d>(blocks["to_qkv"]);
            auto proj   = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

            // WAN_VAE_F16: whole-block F32 island. The block sits at the decoder bottleneck
            // (smallest spatial res -> cheap) and its Conv2d (to_qkv/proj) emit F32 anyway, so
            // running it in F32 keeps attention numerics exactly as prod AND avoids the unsafe
            // add(F32 proj-out, F16 identity) mixed-dtype broadcast. Cast back to F16 on exit.
            const bool island_f32 = (x->type == GGML_TYPE_F16);
            if (island_f32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }

            auto identity = x;

            x = norm->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)

            const int64_t n = x->ne[3];
            const int64_t c = x->ne[2];
            const int64_t h = x->ne[1];
            const int64_t w = x->ne[0];

            auto qkv     = to_qkv->forward(ctx, x);
            auto qkv_vec = split_image_qkv(ctx->ggml_ctx, qkv);

            auto q = qkv_vec[0];
            q      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 2, 0, 1, 3));  // [t, h, w, c]
            q      = ggml_reshape_3d(ctx->ggml_ctx, q, c, h * w, n);                                      // [t, h * w, c]

            auto k = qkv_vec[1];
            k      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 2, 0, 1, 3));  // [t, h, w, c]
            k      = ggml_reshape_3d(ctx->ggml_ctx, k, c, h * w, n);                                      // [t, h * w, c]

            auto v = qkv_vec[2];
            v      = ggml_reshape_3d(ctx->ggml_ctx, v, h * w, c, n);  // [t, c, h * w]

            v = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, v, 1, 0, 2, 3));                            // [t, h * w, c]
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);  // [t, h * w, c]

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [t, c, h * w]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, c, n);                             // [t, c, h, w]

            x = proj->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)

            x = ggml_add(ctx->ggml_ctx, x, identity);
            // Restore the F16 residual stream for the next decoder block.
            if (island_f32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F16);
            }
            return x;
        }
    };

    class Encoder3d : public GGMLBlock {
    protected:
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_downsample;

    public:
        Encoder3d(int64_t dim                           = 128,
                  int64_t z_dim                         = 4,
                  std::vector<int> dim_mult             = {1, 2, 4, 4},
                  int num_res_blocks                    = 2,
                  std::vector<bool> temperal_downsample = {false, true, true},
                  bool wan2_2                           = false)
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_downsample(temperal_downsample),
              wan2_2(wan2_2) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }

            if (wan2_2) {
                blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(12, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            } else {
                blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(3, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            }

            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                if (wan2_2) {
                    bool t_down_flag = i < temperal_downsample.size() ? temperal_downsample[i] : false;
                    auto block       = std::shared_ptr<GGMLBlock>(new Down_ResidualBlock(in_dim,
                                                                                         out_dim,
                                                                                         num_res_blocks,
                                                                                         t_down_flag,
                                                                                         i != dim_mult.size() - 1));

                    blocks["downsamples." + std::to_string(index++)] = block;
                } else {
                    for (int j = 0; j < num_res_blocks; j++) {
                        auto block                                       = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                        blocks["downsamples." + std::to_string(index++)] = block;
                        in_dim                                           = out_dim;
                    }

                    if (i != dim_mult.size() - 1) {
                        std::string mode                                 = temperal_downsample[i] ? "downsample3d" : "downsample2d";
                        auto block                                       = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                        blocks["downsamples." + std::to_string(index++)] = block;
                    }
                }
            }

            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(out_dim));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));

            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, z_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.prelude", "x");

            // downsamples
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                if (wan2_2) {
                    auto layer = std::dynamic_pointer_cast<Down_ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                } else {
                    for (int j = 0; j < num_res_blocks; j++) {
                        auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                    }

                    if (i != dim_mult.size() - 1) {
                        auto layer = std::dynamic_pointer_cast<Resample>(blocks["downsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                    }
                }
                // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.down." + std::to_string(i), "x");
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.mid", "x");

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class Decoder3d : public GGMLBlock {
    protected:
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_upsample;

    public:
        Decoder3d(int64_t dim                         = 128,
                  int64_t z_dim                       = 4,
                  std::vector<int> dim_mult           = {1, 2, 4, 4},
                  int num_res_blocks                  = 2,
                  std::vector<bool> temperal_upsample = {true, true, false},
                  bool wan2_2                         = false)
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_upsample(temperal_upsample),
              wan2_2(wan2_2) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }

            // init block
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            // middle blocks
            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(dims[0]));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));

            // upsample blocks
            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                if (wan2_2) {
                    bool t_up_flag = i < temperal_upsample.size() ? temperal_upsample[i] : false;
                    auto block     = std::shared_ptr<GGMLBlock>(new Up_ResidualBlock(in_dim,
                                                                                     out_dim,
                                                                                     num_res_blocks + 1,
                                                                                     t_up_flag,
                                                                                     i != dim_mult.size() - 1));

                    blocks["upsamples." + std::to_string(index++)] = block;
                } else {
                    if (i == 1 || i == 2 || i == 3) {
                        in_dim = in_dim / 2;
                    }
                    for (int j = 0; j < num_res_blocks + 1; j++) {
                        auto block                                     = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                        blocks["upsamples." + std::to_string(index++)] = block;
                        in_dim                                         = out_dim;
                    }

                    if (i != dim_mult.size() - 1) {
                        std::string mode                               = temperal_upsample[i] ? "upsample3d" : "upsample2d";
                        auto block                                     = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                        blocks["upsamples." + std::to_string(index++)] = block;
                    }
                }
            }

            // output blocks
            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            if (wan2_2) {
                // force_f32=true: this final 12-channel conv feeds unpatchify(2); request an
                // F32-IO cuDNN plan (fp32 Y, not fp16) so the per-channel fp16 steps don't grid.
                blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 12, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true, true));

            } else {
                blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 3, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true, true));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.prelude", "x");

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.mid", "x");

            // upsamples
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                if (wan2_2) {
                    auto layer = std::dynamic_pointer_cast<Up_ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                } else {
                    for (int j = 0; j < num_res_blocks + 1; j++) {
                        auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                    }

                    if (i != dim_mult.size() - 1) {
                        auto layer = std::dynamic_pointer_cast<Resample>(blocks["upsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                    }
                }
                // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.up." + std::to_string(i), "x");
            }

            // head
            // WAN_VAE_RMS_CF: same channels-first RMS->SiLU->conv fusion as ResidualBlock — head.0
            // stays channels-first, SiLU reads the zero-copy [W,H,T,C] view and writes it CONTIGUOUS
            // (absorbing the permute-back), so head.2 gets a contiguous tensor. Value-identical.
            const bool rms_kernel = wan_vae_rms_kernel_enabled() && !g_ext_vae_phase_encode;
            const bool rms_cf     = !rms_kernel && wan_vae_rms_cf_enabled() && !g_ext_vae_phase_encode;
            if (rms_kernel) {
                x = head_0->forward_channels_kernel(ctx, x);  // [W,H,T,C] contiguous, single op
                x = ggml_silu(ctx->ggml_ctx, x);
            } else if (rms_cf) {
                x = head_0->forward_channels_first(ctx, x);         // [C,W,H,T] contiguous
                x = RMS_norm::channels_first_to_whtc_view(ctx, x);  // zero-copy [W,H,T,C] view
                x = ggml_silu(ctx->ggml_ctx, x);                    // strided-in -> contiguous-out
            } else {
                x = head_0->forward(ctx, x);
                x = ggml_silu(ctx->ggml_ctx, x);
            }
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class WanVAE : public GGMLBlock {
    public:
        bool wan2_2                           = false;
        bool decode_only                      = true;
        int64_t dim                           = 96;
        int64_t dec_dim                       = 96;
        int64_t z_dim                         = 16;
        std::vector<int> dim_mult             = {1, 2, 4, 4};
        int num_res_blocks                    = 2;
        std::vector<bool> temperal_upsample   = {true, true, false};
        std::vector<bool> temperal_downsample = {false, true, true};

        int _conv_num = 33;
        int _conv_idx = 0;
        std::vector<ggml_tensor*> _feat_map;
        int _enc_conv_num = 28;
        int _enc_conv_idx = 0;
        std::vector<ggml_tensor*> _enc_feat_map;

        void clear_cache() {
            _conv_idx     = 0;
            _feat_map     = std::vector<ggml_tensor*>(_conv_num, nullptr);
            _enc_conv_idx = 0;
            _enc_feat_map = std::vector<ggml_tensor*>(_enc_conv_num, nullptr);
        }

    public:
        // dec_dim_override: narrower decoder base width for a channel-pruned decoder
        // (e.g. lightx2v LightVAE = the official Wan2.1 VAE with dec base dim 96->24).
        // -1 = keep the full-width default for this version (byte-identical to the
        // official VAE). Only the DECODER width changes; the encoder width (`dim`) is
        // untouched so the encoder latent space stays identical to the official VAE
        // (decode-only weight swap). Detected from the gguf by WanVAERunner.
        WanVAE(bool decode_only = true, bool wan2_2 = false, int64_t dec_dim_override = -1)
            : decode_only(decode_only), wan2_2(wan2_2) {
            // attn_scales is always []
            if (wan2_2) {
                dim     = 160;
                dec_dim = 256;
                z_dim   = 48;

                _conv_num     = 34;
                _enc_conv_num = 26;
            }
            if (dec_dim_override > 0) {
                dec_dim = dec_dim_override;
            }
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample, wan2_2));
                blocks["conv1"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim * 2, z_dim * 2, {1, 1, 1}));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, temperal_upsample, wan2_2));
            blocks["conv2"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, z_dim, {1, 1, 1}));
        }

        static ggml_tensor* patchify(ggml_context* ctx,
                                     ggml_tensor* x,
                                     int64_t patch_size,
                                     int64_t b = 1) {
            // x: [b*c, f, h*q, w*r]
            // return: [b*c*r*q, f, h, w]
            if (patch_size == 1) {
                return x;
            }
            int64_t r = patch_size;
            int64_t q = patch_size;
            int64_t c = x->ne[3] / b;
            int64_t f = x->ne[2];
            int64_t h = x->ne[1] / q;
            int64_t w = x->ne[0] / r;

            x = ggml_reshape_4d(ctx, x, r * w, q, h, f * c * b);                 // [b*c*f, h, q, w*r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c*f, q, h, w*r]
            x = ggml_reshape_4d(ctx, x, r, w, h * q, f * c * b);                 // [b*c*f, q*h, w, r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));  // [b*c*f, r, q*h, w]
            x = ggml_reshape_4d(ctx, x, w * h, q * r, f, c * b);                 // [b*c, f, r*q, h*w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c, r*q, f, h*w]
            x = ggml_reshape_4d(ctx, x, w, h, f, q * r * c * b);                 // [b*c*r*q, f, h, w]

            return x;
        }

        static ggml_tensor* unpatchify(ggml_context* ctx,
                                       ggml_tensor* x,
                                       int64_t patch_size,
                                       int64_t b = 1) {
            // x: [b*c*r*q, f, h, w]
            // return: [b*c, f, h*q, w*r]
            if (patch_size == 1) {
                return x;
            }
            int64_t r = patch_size;
            int64_t q = patch_size;
            int64_t c = x->ne[3] / b / q / r;
            int64_t f = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            x = ggml_reshape_4d(ctx, x, w * h, f, q * r, c * b);                 // [b*c, r*q, f, h*w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c, f, r*q, h*w]
            x = ggml_reshape_4d(ctx, x, w, h * q, r, f * c * b);                 // [b*c*f, r, q*h, w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [b*c*f, q*h, w, r]
            x = ggml_reshape_4d(ctx, x, r * w, h, q, f * c * b);                 // [b*c*f, q, h, w*r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c*f, h, q, w*r]
            x = ggml_reshape_4d(ctx, x, r * w, q * h, f, c * b);                 // [b*c, f, h*q, w*r]
            return x;
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx,
                            ggml_tensor* x,
                            int64_t b = 1) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            clear_cache();

            if (wan2_2) {
                x = patchify(ctx->ggml_ctx, x, 2, b);
            }
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encode.prelude", "x");

            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            int64_t t     = x->ne[2];
            int64_t iter_ = 1 + (t - 1) / 4;
            ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _enc_conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);  // [b*c, 1, h, w]
                    out     = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                } else {
                    auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1 + 4 * (i - 1), 1 + 4 * i);  // [b*c, 4, h, w]
                    auto out_ = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                    out       = ggml_concat(ctx->ggml_ctx, out, out_, 2);
                }
            }
            out     = conv1->forward(ctx, out);
            auto mu = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3)[0];
            // sd::ggml_graph_cut::mark_graph_cut(mu, "wan_vae.encode.final", "mu");
            clear_cache();
            return mu;
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx,
                            ggml_tensor* z,
                            int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            clear_cache();

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            // WAN_VAE_F16: cast the latent to F16 at the decode entry so the whole decoder
            // activation stream runs F16 (conv2 onward); the unpatchified output is cast back
            // to F32 below. Default (gate off): z stays F32, byte-identical.
            const bool dec_f16 = wan_vae_f16_enabled();
            if (dec_f16 && z->type == GGML_TYPE_F32) {
                z = ggml_cast(ctx->ggml_ctx, z, GGML_TYPE_F16);
            }

            int64_t iter_ = z->ne[2];
            auto x        = conv2->forward(ctx, z);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decode.prelude", "x");
            ggml_tensor* out = nullptr;
            for (int i = 0; i < iter_; i++) {
                _conv_idx = 0;
                auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1, /*cont=*/!wan_vae_slice_nocopy_enabled());  // [b*c, 1, h, w]
                auto out_ = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
                // WAN_VAE_F16 (wan2.1 only): head.2 is force_f32, so each decoded frame comes
                // back F32; the accumulator therefore grows as a large F32 buffer (all decoded
                // pixel frames held live through the whole per-frame loop). Cast each frame to
                // F16 before accumulating so the resident output buffer is halved during the
                // loop's memory peak; the final cast below restores F32 for host read-back. Only
                // for wan2.1 (direct RGB head, no unpatchify): the wan2.2 path must keep its 12ch
                // head output F32 through unpatchify or the per-channel fp16 grid returns.
                if (dec_f16 && !wan2_2 && out_->type != GGML_TYPE_F16) {
                    out_ = ggml_cast(ctx->ggml_ctx, out_, GGML_TYPE_F16);
                }
                out = (out == nullptr) ? out_ : ggml_concat(ctx->ggml_ctx, out, out_, 2);
            }
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            // WAN_VAE_F16: bring the decoded pixels back to F32 for the host read-back.
            if (dec_f16 && out->type != GGML_TYPE_F32) {
                out = ggml_cast(ctx->ggml_ctx, out, GGML_TYPE_F32);
            }
            // sd::ggml_graph_cut::mark_graph_cut(out, "wan_vae.decode.final", "out");
            clear_cache();
            return out;
        }

        ggml_tensor* decode_partial(GGMLRunnerContext* ctx,
                                    ggml_tensor* z,
                                    int i,
                                    int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto x = conv2->forward(ctx, z);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decode_partial.prelude", "x");
            auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1, /*cont=*/!wan_vae_slice_nocopy_enabled());  // [b*c, 1, h, w]
            _conv_idx = 0;
            auto out  = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            // sd::ggml_graph_cut::mark_graph_cut(out, "wan_vae.decode_partial.final", "out");
            return out;
        }

        // Streaming/cached decode of a RANGE of latent frames [frame_base, frame_base+z->ne[2]).
        // This is decode()'s per-frame loop, but (a) restricted to the frames present in
        // `z` (a temporal slice of the full latent) and (b) WITHOUT the clear_cache() at
        // either end, so the causal _feat_map is carried in/out by the caller across
        // separate compute() passes (temporal streaming). conv2 is a {1,1,1} CausalConv3d
        // (pointwise in time) and unpatchify is per-frame independent, so decoding the
        // latent in temporal chunks is numerically identical to decode() of the whole
        // latent in one graph — only the peak activation memory is bounded (to one
        // chunk's worth of full-spatial-res decoder activations).
        //
        // `frame_base` is the GLOBAL latent-frame index of z's first frame: the decoder's
        // CausalConv3d branches on chunk_idx (==0 / ==1 / >=2), so each frame MUST be fed
        // its global index, not a per-chunk-local one, for the cache logic to match.
        ggml_tensor* decode_chunk(GGMLRunnerContext* ctx,
                                  ggml_tensor* z,
                                  int frame_base,
                                  int64_t b = 1) {
            // z: [b*c, t_chunk, h, w]
            GGML_ASSERT(b == 1);

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            // WAN_VAE_F16: F16 activation stream for this chunk. The cross-graph causal cache
            // (_feat_map, persisted by persist_feat_map) is then F16 too — consistent across
            // chunks since every chunk reloads & feeds it back in the same F16 dtype.
            const bool dec_f16 = wan_vae_f16_enabled();
            if (dec_f16 && z->type == GGML_TYPE_F32) {
                z = ggml_cast(ctx->ggml_ctx, z, GGML_TYPE_F16);
            }

            auto x          = conv2->forward(ctx, z);  // {1,1,1} pointwise -> per-frame independent
            int64_t n_chunk = x->ne[2];
            ggml_tensor* out = nullptr;
            for (int64_t k = 0; k < n_chunk; k++) {
                _conv_idx = 0;
                auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, k, k + 1, /*cont=*/!wan_vae_slice_nocopy_enabled());  // [b*c, 1, h, w]
                int gi    = frame_base + static_cast<int>(k);
                auto out_ = decoder->forward(ctx, in, b, _feat_map, _conv_idx, gi);
                out       = (out == nullptr) ? out_ : ggml_concat(ctx->ggml_ctx, out, out_, 2);
            }
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            // WAN_VAE_F16: back to F32 for host read-back / output concat.
            if (dec_f16 && out->type != GGML_TYPE_F32) {
                out = ggml_cast(ctx->ggml_ctx, out, GGML_TYPE_F32);
            }
            return out;
        }

        // Streaming/cached encode of ONE encoder chunk, mirror of decode_partial.
        // Unlike encode() (which clear_cache()'s at both ends), this keeps the
        // causal encoder history in _enc_feat_map so a chunk encoded after priming
        // chunks carries P-frame temporal state instead of resetting to an I-frame.
        // x: [b*c, t_chunk, h, w]  (t_chunk = 1 for the first chunk, up to 4 later).
        // `i` is the chunk index (0 = first / history-less). Returns this chunk's
        // mu (vae latent). Caller must NOT clear_cache between chunks.
        ggml_tensor* encode_partial(GGMLRunnerContext* ctx,
                                    ggml_tensor* x,
                                    int i,
                                    int64_t b = 1) {
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            if (wan2_2) {
                x = patchify(ctx->ggml_ctx, x, 2, b);
            }

            _enc_conv_idx = 0;
            auto out      = encoder->forward(ctx, x, b, _enc_feat_map, _enc_conv_idx, i);
            // conv1 here is a {1,1,1} CausalConv3d (pointwise in time) so applying it
            // per-chunk is identical to applying it on the concatenated sequence in
            // encode(); the channel-chunk likewise selects mu (first half).
            out     = conv1->forward(ctx, out);
            auto mu = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3)[0];
            return mu;
        }

        // In-graph streaming/cached encode (ONE forward graph — no cross-graph cache).
        // The cross-graph build_graph_partial path is disabled in this codebase (see the
        // "chunk 1 result is weird" note in WanVAERunner::_compute); carrying _enc_feat_map
        // across compute() calls corrupts the chunk-1 result. Instead we build all chunks
        // into a single graph exactly the way encode() does, so the cache is just ordinary
        // intra-graph data dependencies.
        // x: [b*c, K, h, w]. The leading K-1 frames are the priming tail (chunk 0 = 1 frame,
        // then 4-frame chunks); the FINAL frame is encoded as a standalone 1-frame chunk WITH
        // the primed cache. Returns ONLY the target frame's mu (vae latent [W_lat,H_lat,1,48]).
        // K==1 -> a plain history-less I-encode (== encode() of a single frame, bit-identical).
        ggml_tensor* encode_tail(GGMLRunnerContext* ctx,
                                 ggml_tensor* x,
                                 int64_t b = 1) {
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            clear_cache();

            if (wan2_2) {
                x = patchify(ctx->ggml_ctx, x, 2, b);
            }
            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            // The encoder only tolerates chunk sizes 1 (chunk 0) and 4 (the regime encode()
            // ever feeds); a 3-frame chunk crashes Down_ResidualBlock's avg-shortcut add.
            // Front-pad the priming tail (repeat the oldest frame = a "static-before-clip"
            // causal assumption, identical in spirit to the history-less I-frame) so that
            // (n_prime - 1) is divisible by 4, i.e. priming = chunk0(1) + k*chunk(4).
            int64_t n_prime = x->ne[2] - 1;
            if (n_prime >= 1) {
                int64_t padded = n_prime;
                while ((padded - 1) % 4 != 0) padded++;
                for (int64_t p = n_prime; p < padded; p++) {
                    auto first = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                    x          = ggml_concat(ctx->ggml_ctx, first, x, 2);  // prepend oldest
                }
                n_prime = padded;
            }
            const int64_t t = x->ne[2];
            int chunk_i     = 0;

            // priming chunks over frames [0, n_prime): fills _enc_feat_map in-graph.
            if (n_prime >= 1) {
                {
                    _enc_conv_idx = 0;
                    auto in       = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);  // chunk 0: 1 frame
                    encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, chunk_i++);
                }
                for (int64_t s = 1; s < n_prime; s += 4) {
                    _enc_conv_idx = 0;
                    auto in       = ggml_ext_slice(ctx->ggml_ctx, x, 2, s, std::min<int64_t>(n_prime, s + 4));
                    encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, chunk_i++);
                }
            }

            // target: the final frame as a standalone 1-frame chunk with the primed cache.
            _enc_conv_idx = 0;
            auto in       = ggml_ext_slice(ctx->ggml_ctx, x, 2, t - 1, t);
            auto out      = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, chunk_i);
            out           = conv1->forward(ctx, out);
            auto mu       = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3)[0];
            clear_cache();
            return mu;
        }
    };

    struct WanVAERunner : public VAE {
        float scale_factor = 1.0f;
        bool decode_only   = true;
        WanVAE ae;

        // Decoder base width from the gguf: the final-stage decoder RMS_norm
        // (decoder.head.0.gamma) is a 1-D [dec_dim] tensor, so its element count IS the
        // decoder base dim. Matched by suffix so it's independent of the load prefix.
        // Returns -1 when absent -> WanVAE keeps the version default. For the official
        // VAEs this returns the same value as the built-in default (96 wan2.1 / 256
        // wan2.2-ti2v), so existing models build byte-identically; a channel-pruned
        // LightVAE gguf (gamma=[24]) builds the narrower decoder instead.
        static int64_t detect_dec_dim(const String2TensorStorage& tensor_storage_map) {
            static const std::string suffix = "decoder.head.0.gamma";
            for (const auto& kv : tensor_storage_map) {
                const std::string& k = kv.first;
                if (k.size() >= suffix.size() &&
                    k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    int64_t n = 1;
                    for (int i = 0; i < kv.second.n_dims; i++) {
                        n *= kv.second.ne[i];
                    }
                    return n;
                }
            }
            return -1;
        }

        WanVAERunner(ggml_backend_t backend,
                     ggml_backend_t params_backend,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     bool decode_only                               = false,
                     SDVersion version                              = VERSION_WAN2)
            : decode_only(decode_only),
              ae(decode_only, version == VERSION_WAN2_2_TI2V, detect_dec_dim(tensor_storage_map)),
              VAE(version, backend, params_backend) {
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "wan_vae";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
            ae.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
            SD_UNUSED(rng);
            return vae_output;
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> get_latents_mean_std(const sd::Tensor<float>& latents) {
            int channel_dim = latents.dim() == 5 ? 3 : 2;
            std::vector<int64_t> stats_shape(static_cast<size_t>(latents.dim()), 1);
            if (latents.shape()[channel_dim] == 16) {  // Wan2.1 VAE
                stats_shape[static_cast<size_t>(channel_dim)] = 16;

                auto mean_tensor = sd::Tensor<float>::from_vector({-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                                                   0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f});
                mean_tensor.reshape_(stats_shape);
                auto std_tensor = sd::Tensor<float>::from_vector({2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                                                  3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f});
                std_tensor.reshape_(stats_shape);
                return {std::move(mean_tensor), std::move(std_tensor)};
            }
            if (latents.shape()[channel_dim] == 48) {  // Wan2.2 VAE
                stats_shape[static_cast<size_t>(channel_dim)] = 48;

                auto mean_tensor = sd::Tensor<float>::from_vector({-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f,
                                                                   -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f,
                                                                   -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f,
                                                                   -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f,
                                                                   -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f,
                                                                   0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f});
                mean_tensor.reshape_(stats_shape);
                auto std_tensor = sd::Tensor<float>::from_vector({0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f,
                                                                  0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
                                                                  0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f,
                                                                  0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
                                                                  0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f,
                                                                  0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f});
                std_tensor.reshape_(stats_shape);
                return {std::move(mean_tensor), std::move(std_tensor)};
            }
            GGML_ABORT("unexpected latent channel dimension %lld for version %d",
                       (long long)latents.shape()[channel_dim],
                       version);
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents);
            return (latents * std_tensor) / scale_factor + mean_tensor;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents);
            return ((latents - mean_tensor) * scale_factor) / std_tensor;
        }

        int get_encoder_output_channels(int input_channels) {
            return static_cast<int>(ae.z_dim);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
            ggml_cgraph* gf = new_graph_custom(10240 * z_tensor.shape()[2]);
            ggml_tensor* z  = make_input(z_tensor);

            auto runner_ctx = get_context();

            // Tell ggml_ext_conv_3d which VAE phase this graph is, so GGML_CUDNN_CONV3D=encode
            // routes only the encode's CausalConv3d to the low-VRAM cuDNN direct conv (the
            // encode sets the peak) while the decode stays on the fast im2col path.
            g_ext_vae_phase_encode = !decode_graph;

            ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);

            ggml_build_forward_expand(gf, out);

            if (std::getenv("NAVA_VAE_OP_HIST") != nullptr) {
                std::map<int, int> hist;
                std::map<int, int64_t> work;
                int nconcat   = 0;
                int n_nodes   = ggml_graph_n_nodes(gf);
                for (int ni = 0; ni < n_nodes; ni++) {
                    ggml_tensor* n = ggml_graph_node(gf, ni);
                    hist[(int)n->op]++;
                    work[(int)n->op] += ggml_nelements(n);
                    if (n->op == GGML_OP_CONCAT && nconcat < 8) {
                        printf("  CONCAT[%d] dim=%d ne=[%lld,%lld,%lld,%lld]\n", nconcat,
                               n->op_params[0], (long long)n->ne[0], (long long)n->ne[1],
                               (long long)n->ne[2], (long long)n->ne[3]);
                        nconcat++;
                    }
                }
                printf("=== VAE graph op histogram (n_nodes=%d) ===\n", n_nodes);
                for (auto& kv : hist)
                    printf("  %-22s count=%7d  out_elems=%lld\n", ggml_op_name((ggml_op)kv.first),
                           kv.second, (long long)work[kv.first]);
            }

            return gf;
        }

        ggml_cgraph* build_graph_partial(const sd::Tensor<float>& z_tensor, bool decode_graph, int i) {
            ggml_cgraph* gf = new_graph_custom(20480);

            ae.clear_cache();

            // Decoder uses _feat_map ("feat_idx:*"); encoder uses _enc_feat_map
            // ("enc_feat_idx:*"). Reload whichever applies from the cross-graph cache.
            std::vector<ggml_tensor*>& fmap = decode_graph ? ae._feat_map : ae._enc_feat_map;
            const std::string cpfx          = decode_graph ? "feat_idx:" : "enc_feat_idx:";
            for (size_t feat_idx = 0; feat_idx < fmap.size(); feat_idx++) {
                fmap[feat_idx] = get_cache_tensor_by_name(cpfx + std::to_string(feat_idx));
            }

            ggml_tensor* z = make_input(z_tensor);

            auto runner_ctx = get_context();

            ggml_tensor* out = decode_graph ? ae.decode_partial(&runner_ctx, z, i)
                                            : ae.encode_partial(&runner_ctx, z, i);

            for (size_t feat_idx = 0; feat_idx < fmap.size(); feat_idx++) {
                ggml_tensor* feat_cache = fmap[feat_idx];
                if (feat_cache != nullptr) {
                    cache(cpfx + std::to_string(feat_idx), feat_cache);
                    ggml_build_forward_expand(gf, feat_cache);
                }
            }

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        // Streaming/cached encode for clip-continuity experiments. Encodes `frames`
        // [W,H,K,3,1] chunk-by-chunk through encode_partial, carrying _enc_feat_map
        // across chunks (the encoder's causal history) rather than clearing it. The
        // leading K-1 frames are the priming tail (chunk 0 = 1 frame, then 4-frame
        // chunks, mirroring encode()'s chunking); the FINAL frame is then encoded as
        // a standalone 1-frame chunk WITH the primed cache. Returns the final frame's
        // mu (vae latent, ne [W_lat,H_lat,1,48]). With K==1 this degenerates to a
        // fresh history-less I-encode == the M=1 baseline anchor.
        sd::Tensor<float> encode_streaming(const int n_threads, const sd::Tensor<float>& frames) {
            // Single-graph in-graph cache (encode_tail) — NOT the cross-graph
            // build_graph_partial path, which is disabled/buggy for chunk>=1.
            // Mirror the encode() wrapper's [0,1] -> [-1,1] input scaling (scale_input),
            // which encode_streaming bypasses by calling _compute directly.
            sd::Tensor<float> input = frames;
            if (scale_input) {
                scale_tensor_to_minus1_1(&input);
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(10240 * input.shape()[2]);
                ggml_tensor* x  = make_input(input);
                auto runner_ctx = get_context();
                ggml_tensor* out = ae.encode_tail(&runner_ctx, x);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto mu = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, true),
                                                      input.dim());
            free_compute_buffer();
            return mu;
        }

        // --- Temporal streaming (full spatial resolution, bounded VRAM) ------------------
        //
        // 1x1 spatial tiling (zero spatial seams) OOMs because a full-frame x all-frames
        // decode allocates one giant activation buffer (~15.7GB). Instead we stream the
        // CAUSAL VAE over temporal chunks of latent (decode) / pixel (encode) frames at
        // FULL spatial resolution, carrying the causal feat_cache across compute() passes
        // via the runner's persistent (non-gallocr) cache buffer. Chunk boundaries are
        // seamless because the cache reproduces the exact causal context decode()/encode()
        // would have had in a single graph. Env-gated, OFF by default.
        //
        // LONGCAT_VAE_TEMPORAL_CHUNK=N : N>=1 enables decode streaming with N latent frames
        //                            per compute pass (encode streams its natural 1+4k groups).
        //                            0/unset = disabled (committed full-frame path untouched).
        //                            (LONGCAT_ prefix forwards it.)
        static int wan_vae_temporal_chunk() {
            static const int n = [] {
                const char* s = getenv("LONGCAT_VAE_TEMPORAL_CHUNK");
                if (s == nullptr || s[0] == '\0')
                    return 0;
                int v = atoi(s);
                return v < 0 ? 0 : v;
            }();
            return n;
        }

        static std::string wan_dec_feat_name(size_t i) { return "wan_dec_feat:" + std::to_string(i); }
        static std::string wan_enc_feat_name(size_t i) { return "wan_enc_feat:" + std::to_string(i); }

        // Persist the per-conv causal feat_cache into the runner's cross-graph cache buffer.
        //
        // GALLOCR/CACHE FIX (root cause of the disabled build_graph_partial "chunk 1 weird"
        // bug): WAN::CausalConv3d stores its cache slot as a ggml_ext_slice VIEW into this
        // graph's activations (e.g. wan_vae.hpp `cache_x = ggml_ext_slice(...)`). The old
        // path did `cache(name, view); ggml_build_forward_expand(gf, view);`. cache() conts
        // a view INTO A NEW tensor stored in cache_tensor_map, but the loop expanded the
        // VIEW, so the cont node was never wired into the cgraph and never computed -->
        // copy_cache_tensors_to_cache_buffer() persisted UNINITIALIZED memory. Chunk 0's
        // OUTPUT was still correct (it reads no cache), but its STORED cache was garbage, so
        // chunk 1 (which reads it) came out "weird". The LTX streaming path avoids this by
        // storing ggml_cont(...) in the cache slot up-front. We fix it at the persist
        // boundary instead (keeping the committed conv forward byte-identical): cont the
        // view OURSELVES, then cache AND expand the SAME cont node so it is computed before
        // the cache copy reads it.
        void persist_feat_map(ggml_cgraph* gf,
                              std::vector<ggml_tensor*>& fmap,
                              const std::function<std::string(size_t)>& name_of) {
            for (size_t fi = 0; fi < fmap.size(); fi++) {
                ggml_tensor* fc = fmap[fi];
                if (fc == nullptr) {
                    continue;
                }
                if (fc->view_src != nullptr || !ggml_is_contiguous(fc)) {
                    fc = ggml_cont(compute_ctx, fc);
                }
                cache(name_of(fi), fc);
                ggml_build_forward_expand(gf, fc);
            }
        }

        ggml_cgraph* build_graph_temporal_decode_chunk(const sd::Tensor<float>& z_chunk, int frame_base) {
            ggml_cgraph* gf = new_graph_custom(10240 * std::max<int64_t>(1, z_chunk.shape()[2]) + 4096);

            // Reload the decoder's causal cache from the persistent cross-graph buffer
            // (null on the first chunk -> history-less I-frame, == decode()'s chunk_idx 0).
            for (size_t fi = 0; fi < ae._feat_map.size(); fi++) {
                ae._feat_map[fi] = get_cache_tensor_by_name(wan_dec_feat_name(fi));
            }

            ggml_tensor* z         = make_input(z_chunk);
            auto runner_ctx        = get_context();
            g_ext_vae_phase_encode = false;
            ggml_tensor* out       = ae.decode_chunk(&runner_ctx, z, frame_base);

            persist_feat_map(gf, ae._feat_map, &wan_dec_feat_name);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        ggml_cgraph* build_graph_temporal_encode_chunk(const sd::Tensor<float>& x_chunk, int group_idx) {
            ggml_cgraph* gf = new_graph_custom(10240 * std::max<int64_t>(1, x_chunk.shape()[2]) + 4096);

            for (size_t fi = 0; fi < ae._enc_feat_map.size(); fi++) {
                ae._enc_feat_map[fi] = get_cache_tensor_by_name(wan_enc_feat_name(fi));
            }

            ggml_tensor* x         = make_input(x_chunk);
            auto runner_ctx        = get_context();
            g_ext_vae_phase_encode = true;
            // encode_partial(): patchify + encoder->forward(group_idx) + conv1 ({1,1,1}) +
            // channel-chunk -> this group's mu. Carries _enc_feat_map (no clear_cache).
            ggml_tensor* mu        = ae.encode_partial(&runner_ctx, x, group_idx);

            persist_feat_map(gf, ae._enc_feat_map, &wan_enc_feat_name);
            ggml_build_forward_expand(gf, mu);
            return gf;
        }

        sd::Tensor<float> decode_temporal_streaming(const int n_threads,
                                                    const sd::Tensor<float>& z,
                                                    int chunk_frames) {
            const int64_t total = z.shape()[2];  // latent frames
            const size_t expected_dim = static_cast<size_t>(z.dim());

            free_cache_ctx_and_buffer();
            cache_tensor_map.clear();
            ae.clear_cache();  // sizes _feat_map to _conv_num nullptrs

            LOG_DEBUG("wan_vae temporal-streaming decode: %lld latent frames, chunk=%d",
                      (long long)total, chunk_frames);

            sd::Tensor<float> output;
            for (int64_t start = 0; start < total; start += chunk_frames) {
                const int64_t end = std::min<int64_t>(total, start + chunk_frames);
                auto z_chunk      = sd::ops::slice(z, 2, start, end);
                auto get_graph    = [&]() -> ggml_cgraph* {
                    return build_graph_temporal_decode_chunk(z_chunk, static_cast<int>(start));
                };
                auto chunk = restore_trailing_singleton_dims(
                    GGMLRunner::compute<float>(get_graph, n_threads, true), expected_dim);
                if (chunk.empty()) {
                    free_cache_ctx_and_buffer();
                    cache_tensor_map.clear();
                    return {};
                }
                output = output.empty() ? std::move(chunk) : sd::ops::concat(output, chunk, 2);
            }

            free_cache_ctx_and_buffer();
            cache_tensor_map.clear();
            return output;
        }

        sd::Tensor<float> encode_temporal_streaming(const int n_threads,
                                                    const sd::Tensor<float>& x) {
            const int64_t t_pix       = x.shape()[2];  // pixel frames
            const int64_t iter_       = 1 + (t_pix - 1) / 4;
            const size_t expected_dim = static_cast<size_t>(x.dim());

            free_cache_ctx_and_buffer();
            cache_tensor_map.clear();
            ae.clear_cache();  // sizes _enc_feat_map to _enc_conv_num nullptrs

            LOG_DEBUG("wan_vae temporal-streaming encode: %lld pixel frames, %lld groups",
                      (long long)t_pix, (long long)iter_);

            sd::Tensor<float> output;
            for (int64_t i = 0; i < iter_; i++) {
                // encode()'s chunking: group 0 = pixel frame [0,1), group i = [1+4(i-1), 1+4i).
                const int64_t fs = (i == 0) ? 0 : 1 + 4 * (i - 1);
                const int64_t fe = (i == 0) ? 1 : std::min<int64_t>(t_pix, 1 + 4 * i);
                auto x_chunk     = sd::ops::slice(x, 2, fs, fe);
                auto get_graph   = [&]() -> ggml_cgraph* {
                    return build_graph_temporal_encode_chunk(x_chunk, static_cast<int>(i));
                };
                auto mu = restore_trailing_singleton_dims(
                    GGMLRunner::compute<float>(get_graph, n_threads, true), expected_dim);
                if (mu.empty()) {
                    free_cache_ctx_and_buffer();
                    cache_tensor_map.clear();
                    return {};
                }
                output = output.empty() ? std::move(mu) : sd::ops::concat(output, mu, 2);
            }

            free_cache_ctx_and_buffer();
            cache_tensor_map.clear();
            return output;
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            sd::Tensor<float> input;
            if (z.dim() == 4) {
                input = z.unsqueeze(2);
            }
            const sd::Tensor<float>& src = input.empty() ? z : input;
            const size_t out_dim         = src.dim();

            // Temporal streaming (full spatial res, bounded VRAM). Only for genuine video
            // (more than one frame on the temporal axis); single frames use the plain path.
            const int chunk = wan_vae_temporal_chunk();
            LOG_DEBUG("wan_vae _compute %s: z.dim=%zu src.dim=%zu src.shape=[%lld,%lld,%lld,%lld,%lld] chunk=%d",
                      decode_graph ? "decode" : "encode", (size_t)z.dim(), out_dim,
                      (long long)(src.dim() > 0 ? src.shape()[0] : -1), (long long)(src.dim() > 1 ? src.shape()[1] : -1),
                      (long long)(src.dim() > 2 ? src.shape()[2] : -1), (long long)(src.dim() > 3 ? src.shape()[3] : -1),
                      (long long)(src.dim() > 4 ? src.shape()[4] : -1), chunk);
            // Temporal streaming applies to BOTH decode and encode now (chunk = the env
            // LONGCAT_VAE_TEMPORAL_CHUNK; for encode its non-zero value just ENABLES the path —
            // encode_temporal_streaming self-determines its natural 1+4k pixel groups). It
            // bounds the temporal axis so a 21-81 frame encode no longer runs the cuDNN
            // CONV_3D / im2col(IC*27) intermediate over ALL frames at once (the OOM at e.g.
            // [312,536,21] / 1280x704). Both directions use the SAME persist_feat_map GALLOCR
            // fix (build_graph_temporal_{decode,encode}_chunk), so the causal feat_cache threads
            // across compute() passes correctly -> output is numerically ~ monolithic, NOT
            // seam-producing. (The "chunk-1 corrupts" warning is about the OLD, never-called
            // build_graph_partial, not this path.)
            //
            // SPATIAL bounding for the encode (the author's original "4-frame groups blow up at
            // full spatial" concern) comes for FREE from composition, exactly as for decode: when
            // spatial tiling is enabled (LONGCAT_VAE_ENCODE_REL_TILE / vae_tiling_params), the
            // encode() wrapper (vae.hpp:131) spatial-tiles via tiled_compute -> _compute(tile),
            // so this temporal stream runs PER SPATIAL TILE (<=4 frames x one tile). No inner
            // spatial tiling of the chunk is needed (it would double-tile). Encode at 1280x704
            // therefore requires spatial tiling to be on (the failing IT/vid_gen paths already
            // have it firing); without it a 4-frame full-spatial group can still OOM at high res.
            if (chunk >= 1 && src.dim() == 5 && src.shape()[2] > 1) {
                sd::Tensor<float> result = decode_graph
                                               ? decode_temporal_streaming(n_threads, src, chunk)
                                               : encode_temporal_streaming(n_threads, src);
                if (!result.empty()) {
                    result = restore_trailing_singleton_dims(std::move(result), out_dim);
                    if (z.dim() == 4) {
                        result.squeeze_(2);
                    }
                    return result;
                }
                LOG_WARN("wan_vae temporal streaming produced no output; falling back to full-frame path");
            }

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(src, decode_graph);
            };
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, true),
                                                          out_dim);
            if (!result.empty() && z.dim() == 4) {
                result.squeeze_(2);
            }
            return result;
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            if (true) {
                // cpu f32, pass
                // cpu f16, pass
                // cuda f16, pass
                // cuda f32, pass
                auto z = sd::load_tensor_from_file_as_tensor<float>("wan_vae_z.bin");
                print_sd_tensor(z);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = _compute(8, z, true);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("decode test done in %ldms", t1 - t0);
            }
        };

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend            = sd_backend_cpu_init();
            ggml_type model_data_type         = GGML_TYPE_F16;
            std::shared_ptr<WanVAERunner> vae = std::make_shared<WanVAERunner>(backend, backend, String2TensorStorage{}, "", false, VERSION_WAN2_2_TI2V);
            {
                LOG_INFO("loading from '%s'", file_path.c_str());

                if (!vae->alloc_params_buffer()) {
                    LOG_ERROR("vae buffer allocation failed");
                    return;
                }
                std::map<std::string, ggml_tensor*> tensors;
                vae->get_param_tensors(tensors, "first_stage_model");

                ModelLoader model_loader;
                if (!model_loader.init_from_file_and_convert_name(file_path, "vae.")) {
                    LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                    return;
                }

                bool success = model_loader.load_tensors(tensors);

                if (!success) {
                    LOG_ERROR("load tensors from model loader failed");
                    return;
                }

                LOG_INFO("vae model loaded");
            }
            vae->test();
        }
    };

}  // namespace WAN

#endif  // __SD_MODEL_VAE_WAN_VAE_HPP__
