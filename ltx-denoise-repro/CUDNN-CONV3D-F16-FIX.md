# LTX-2.3 22B VAE F16 decode — im2col_3d F16-activation crash fix

## Symptom

```
/src/ggml/src/ggml-cuda/im2col.cu:548  GGML_ASSERT(src1->type == GGML_TYPE_F32) failed
```
Fires on the LTX-2.3 22B VideoVAE decode with `LTX_VAE_DECODE_F16=1` (+ `GGML_CUDNN_CONV3D=…`).
Blocks the F16 decode lever (commit 46fc043) that halves the VAE decode feature-map buffers.
Works on the 0.9.8 2B, crashes on the 22B.

## Which conv was "rejected" and why (root cause)

The assert is in `ggml_cuda_op_im2col_3d` (`GGML_OP_IM2COL_3D`), i.e. a conv that took the
**im2col + mul_mat** path, NOT the cuDNN `GGML_OP_CONV_3D` path. The routing lives in
`src/core/ggml_extend.hpp::ggml_ext_conv_3d`:

- `force_prec_f32` conv  → `ggml_im2col_3d(w, xin, …, w->type)` + PREC_F32 mul_mat   (im2col path)
- `ext_cudnn_conv3d_active()` **and** stride/dilation all 1 → `ggml_conv_3d_direct` (`GGML_OP_CONV_3D`, cuDNN)
- else → `ggml_conv_3d` (im2col path)

Every LTX VAE conv is a `CausalConv3d` = stride 1 / dilation 1 / pad `{0,k/2,k/2}` (F16 weight).
So a conv lands on the **im2col path** whenever cuDNN routing is *not* active for it:

1. `ext_cudnn_conv3d_active()` is false for the phase — e.g. the documented **encode-only cuDNN
   recipe** `GGML_CUDNN_CONV3D=encode` (ggml_extend.hpp:1294-1305: "the win is encode-only …
   decode stays on the fast im2col path"), or a stale/decode phase flag; or
2. the **`LTX_VAE_HEAD_F32=1` head** (`conv_out`, ltx_vae.hpp:997-999) which sets
   `force_prec_f32` and *always* takes the `ggml_im2col_3d` branch regardless of cuDNN.

The crux is dtype, not shape: `ggml_conv_3d` / the `force_prec_f32` branch build the im2col dst as
**`w->type` = F16**, and — critically — under `LTX_VAE_DECODE_F16` the **activation `src1` is F16**.
`ggml_cuda_op_im2col_3d` unconditionally did `src1_d = (const float*)src1->data` and
`GGML_ASSERT(src1->type == F32)`. So *any* conv routed to im2col with an F16 activation reads
garbage and trips the assert. This is a latent defect in the im2col_3d CUDA op, independent of the
cuDNN matcher — the cuDNN conv3d matcher does **not** reject these convs (a stride-1 3×3×3 conv is
exactly the profile it accepts); the reject in `conv3d-cudnn.cu` was never the failure point, because
convs that route to `GGML_OP_CONV_3D` *abort* on reject (ggml-cuda.cu:3327), they do not fall to
im2col. The "0 cuDNN mentions" observation is the phase/`force_prec_f32` routing sending convs to
im2col in the first place, not a matcher rejection.

## Fix (Goal 2 — the correct, minimal one; Goal 1 not needed)

`ggml/src/ggml-cuda/im2col.cu`: make the `im2col_3d` CUDA op read an **F16 or F32** `src1`.

- Added `im2col_load_src(const {float,half}*, i)` → widens the read to `float` (identity for F32).
- Templated the source dtype through `im2col_3d_kernel`, the smem-tiled `im2col_3d_tiled_kernel`,
  `im2col_3d_tiled_try`, and `im2col_3d_cuda` as `<SrcT, T>` (T = im2col dst dtype, unchanged).
- `ggml_cuda_op_im2col_3d`: assert now allows `src1` F16|F32 and dispatches the 2×2
  (src F16/F32 × dst F16/F32) instantiations.

The **F32 activation path is byte-identical**: with `SrcT=float`, `im2col_load_src` is the identity
and every launch resolves to the pre-existing `<float, …>` instantiation. Pure data movement, so the
F16 read is bit-exact vs the value the F32 stream would have carried.

Goal 1 (relaxing the cuDNN matcher) was **not** applied: the matcher already accepts these convs,
and forcing everything onto cuDNN is the *opposite* of the intended encode-only-cuDNN /
im2col-decode recipe. The right fix is to make the im2col path F16-capable, which also makes that
recipe compatible with `LTX_VAE_DECODE_F16` (a real VRAM win the Wan path already relies on).

## VRAM / perf effect

- No new buffers; no cuDNN workspace involved on the im2col path. **`GGML_CUDNN_CONV3D_WS_MB`
  tuning is NOT required** for this fix (the cap only bites when convs actually route to
  `GGML_OP_CONV_3D`).
- The im2col column tensor was *already* F16 (dst = `w->type`); this fix additionally lets its
  **input read** be F16, so the F16 decode lever's halving of the big `[W,H,f,C]` activation /
  feature-map buffers now holds end-to-end on the 22B (the point of commit 46fc043).
- Relative sizing unchanged: cuDNN implicit-GEMM (`GGML_OP_CONV_3D`) still avoids the IC·27 column
  materialization and is the lighter option when a conv routes there; the im2col path materializes
  the F16 column tensor. This fix only makes the im2col fallback *correct* under F16 — pick cuDNN
  routing (`GGML_CUDNN_CONV3D=decode`/`1`) if the column tensor is the VRAM ceiling.

## Files touched
- `ggml/src/ggml-cuda/im2col.cu` (only). `conv3d-cudnn.cu` unchanged.
