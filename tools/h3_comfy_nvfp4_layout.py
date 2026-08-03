#!/usr/bin/env python3
"""NVFP4 byte-layout conversions between ggml `block_nvfp4` and ComfyUI/comfy-kitchen.

WHY THIS EXISTS
---------------
We want to run OUR weights inside ComfyUI.  That is the one experiment that separates
"our conversion is wrong" from "our engine is wrong": no amount of comparing our OUTPUT
against comfy's output can do it.  ComfyUI has no GGUF support, so our GGUF has to be
re-expressed in the layout comfy's own `nvfp4` loader reads.

Both sides store the SAME NUMBERS -- 4-bit E2M1 codes and per-16 E4M3 group scales -- and
neither re-quantises anything here.  This module is a pure permutation of bytes.  It is
the exact inverse of `src/model_io/nvfp4_import.cpp`
(`nvfp4_import_assemble_blocks_comfy`), which reads comfy -> ggml; read that file's header
comment first, it documents the format and was verified numerically.

THE THREE THINGS THAT DIFFER, AND WHY EACH IS DANGEROUS
-------------------------------------------------------
1. GROUP-SCALE ADDRESS.  ggml stores the four E4M3 sub-block scales inline at the front of
   each 64-element block, i.e. row-major [out, in/16].  comfy stores one plane in the
   cuBLAS "blocked" swizzle, zero-padded to (roundup(out,128), roundup(in/16,4)) and
   reordered (rb, cbg, 32, 4, 4) -- `to_blocked()` in comfy_kitchen/float_utils.py:292.
   Getting this wrong is SILENT: for a real layer, reading the scales row-major instead of
   de-swizzled gave a per-output-row L2-norm correlation of 0.013 against the bf16 the
   file was quantised from.  No error, no NaN, no shape mismatch -- just a wrong model.

2. NIBBLE ORDER.  ggml packs a 16-element group as (element j in the LOW nibble of byte j,
   element j+8 in the HIGH nibble) -- split halves.  comfy packs consecutive pairs with
   `hi_first=True`: element 2m in the HIGH nibble of byte m, element 2m+1 in the LOW.
   That default is the same in all three comfy-kitchen backends (eager
   quantization.py:120, triton quantization.py:335, cuda dlpack_bindings.cpp:759).

   ⚠️ NOTE THE TRAP: nibble order is INVISIBLE to every norm-, mean- or std-based check,
   because it only permutes elements WITHIN a 16-group, which no such statistic sees.  It
   is also invisible to the GEMM when it is self-consistent -- swapping k=2m with k=2m+1
   in BOTH operands leaves the dot product unchanged, which is why comfy can pick either
   convention internally.  It is NOT invisible when we write the weights and comfy quantises
   the ACTIVATIONS: comfy's activation packer is hi_first, so a lo_first weight file pairs
   every input channel with its neighbour.  The only way to check it is elementwise against
   an independent copy of the same weights -- see `h3_verify_comfy_layout.py`.

   ⚠️ Our own cuBLASLt repack (ggml/src/ggml-cuda/nvfp4-cublaslt.cu, repack_weight_kernel)
   emits the OPPOSITE order (element 2m in the LOW nibble).  That is fine there because it
   packs the activations the same way.  Do not copy it here.

3. PER-TENSOR GLOBAL.  comfy always carries `weight_scale_2` (F32 scalar) and dequantises
   w = E2M1 * E4M3(group) * scale_2.  Our GGUF's NVFP4 is FLAT -- the group scales are the
   whole story and there is no `.wglobal` sidecar -- so we emit scale_2 = 1.0, which is
   exact.  (ggml's block_nvfp4 decodes E2M1 via kvalues_mxfp4 = 2x standard and E4M3 via
   /2; the two factors cancel, so the stored bytes ARE standard-e4m3 * standard-e2m1 --
   see the convention note at the top of nvfp4-cublaslt.cu.)
"""
import numpy as np

QK_NVFP4 = 64           # elements per ggml block
QK_NVFP4_SUB = 16       # elements per E4M3 group scale
NVFP4_SUBS = QK_NVFP4 // QK_NVFP4_SUB          # 4 group scales per block
NVFP4_BLOCK_BYTES = NVFP4_SUBS + QK_NVFP4 // 2  # 4 + 32 = 36

# cuBLAS block-scaling-factor tiling constants (comfy_kitchen to_blocked / our swz_off).
SWZ_ROWS_PER_BASE_BLOCK = 128
SWZ_ROWS_PER_COL = 32
SWZ_COLS_PER_COL = 4


def _ceil_div(a, b):
    return (a + b - 1) // b


def scale_plane_shape(in_features, out_features):
    """Shape of comfy's `weight_scale` plane for an [out, in] weight."""
    n_sub = in_features // QK_NVFP4_SUB
    rows = _ceil_div(out_features, SWZ_ROWS_PER_BASE_BLOCK) * SWZ_ROWS_PER_BASE_BLOCK
    cols = _ceil_div(n_sub, SWZ_COLS_PER_COL) * SWZ_COLS_PER_COL
    return rows, cols


# ------------------------------------------------------------------ scale-plane swizzle
def to_blocked(scales):
    """[out, n_sub] row-major E4M3 bytes -> comfy's padded, swizzled plane.

    Verbatim port of comfy_kitchen.float_utils.to_blocked(flatten=False).
    """
    rows, cols = scales.shape
    n_rb = _ceil_div(rows, 128)
    n_cb = _ceil_div(cols, 4)
    padded_rows, padded_cols = n_rb * 128, n_cb * 4
    if (rows, cols) != (padded_rows, padded_cols):
        padded = np.zeros((padded_rows, padded_cols), dtype=scales.dtype)
        padded[:rows, :cols] = scales
    else:
        padded = scales
    blocks = padded.reshape(n_rb, 128, n_cb, 4).transpose(0, 2, 1, 3)
    rearranged = np.ascontiguousarray(blocks).reshape(-1, 4, 32, 4).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(rearranged).reshape(padded_rows, padded_cols)


def from_blocked(blocked, num_rows, num_cols):
    """comfy's swizzled plane -> [num_rows, num_cols] row-major E4M3 bytes."""
    n_rb = _ceil_div(num_rows, 128)
    n_cb = _ceil_div(num_cols, 4)
    padded_rows, padded_cols = n_rb * 128, n_cb * 4
    s = blocked.reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
    s = np.ascontiguousarray(s).reshape(n_rb, n_cb, 128, 4).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(s).reshape(padded_rows, padded_cols)[:num_rows, :num_cols]


# ------------------------------------------------------------------ ggml <-> comfy bytes
def ggml_to_comfy(blocks, in_features, out_features):
    """`block_nvfp4` bytes -> (packed [out, in/2] U8, scale plane U8, row-major scales).

    `blocks` is the raw GGUF tensor payload: out * (in/64) blocks of 36 bytes.
    """
    n_blk = in_features // QK_NVFP4
    n_sub = in_features // QK_NVFP4_SUB
    b = np.asarray(blocks, dtype=np.uint8).reshape(out_features, n_blk, NVFP4_BLOCK_BYTES)

    row_major_scales = np.ascontiguousarray(b[:, :, :NVFP4_SUBS]).reshape(out_features, n_sub)

    # ggml qs: 8 bytes per 16-group; byte j holds element j (low) and element j+8 (high).
    qs = np.ascontiguousarray(b[:, :, NVFP4_SUBS:]).reshape(out_features, n_sub, 8)
    lo = qs & 0x0F          # elements 0..7  of the group
    hi = qs >> 4            # elements 8..15 of the group
    elems = np.concatenate([lo, hi], axis=-1)   # [out, n_sub, 16], code per element

    # comfy: byte m holds element 2m in the HIGH nibble, element 2m+1 in the LOW.
    packed = ((elems[:, :, 0::2] << 4) | elems[:, :, 1::2]).reshape(out_features, in_features // 2)
    return packed, to_blocked(row_major_scales), row_major_scales


def comfy_to_ggml(packed, scale_plane, in_features, out_features):
    """Inverse of ggml_to_comfy: -> flat `block_nvfp4` bytes.

    Transcribed from assemble_blocks_impl(src_hi_first=true, src_swizzled=true) in
    src/model_io/nvfp4_import.cpp.
    """
    n_blk = in_features // QK_NVFP4
    n_sub = in_features // QK_NVFP4_SUB

    row_major_scales = from_blocked(np.asarray(scale_plane, dtype=np.uint8), out_features, n_sub)

    p = np.asarray(packed, dtype=np.uint8).reshape(out_features, n_sub, 8)
    elems = np.empty((out_features, n_sub, 16), dtype=np.uint8)
    elems[:, :, 0::2] = p >> 4
    elems[:, :, 1::2] = p & 0x0F

    qs = (elems[:, :, 0:8] | (elems[:, :, 8:16] << 4)).reshape(out_features, n_blk, 32)
    out = np.empty((out_features, n_blk, NVFP4_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, :NVFP4_SUBS] = row_major_scales.reshape(out_features, n_blk, NVFP4_SUBS)
    out[:, :, NVFP4_SUBS:] = qs
    return out.reshape(-1)


# ------------------------------------------------------------------------ dequantisation
def _e4m3_lut():
    """256-entry decode table for float8_e4m3fn (max 448, no inf, 0x7F/0xFF = NaN)."""
    b = np.arange(256, dtype=np.uint8)
    sign = np.where(b >> 7 == 1, -1.0, 1.0)
    exp = ((b >> 3) & 0x0F).astype(np.int32)
    man = (b & 0x07).astype(np.float64)
    val = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1.0 + man / 8.0) * 2.0 ** (exp - 7))
    val = sign * val
    val[(exp == 15) & ((b & 0x07) == 7)] = np.nan
    return val.astype(np.float32)


E4M3_LUT = _e4m3_lut()
E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def dequantize_comfy(packed, scale_plane, scale_2, in_features, out_features, hi_first=True,
                     swizzled=True):
    """comfy NVFP4 triple -> float32 [out, in].  The two flags exist so the verifier can
    demonstrate what the WRONG conventions produce."""
    n_sub = in_features // QK_NVFP4_SUB
    sp = np.asarray(scale_plane, dtype=np.uint8)
    scales = from_blocked(sp, out_features, n_sub) if swizzled else sp.reshape(out_features, n_sub)

    p = np.asarray(packed, dtype=np.uint8).reshape(out_features, n_sub, 8)
    elems = np.empty((out_features, n_sub, 16), dtype=np.uint8)
    if hi_first:
        elems[:, :, 0::2] = p >> 4
        elems[:, :, 1::2] = p & 0x0F
    else:
        elems[:, :, 0::2] = p & 0x0F
        elems[:, :, 1::2] = p >> 4

    w = E2M1_LUT[elems] * (E4M3_LUT[scales][:, :, None] * np.float32(scale_2))
    return w.reshape(out_features, in_features)


def dequantize_ggml(blocks, in_features, out_features):
    """`block_nvfp4` bytes -> float32 [out, in], via the comfy path (round-trip exact)."""
    packed, plane, _ = ggml_to_comfy(blocks, in_features, out_features)
    return dequantize_comfy(packed, plane, 1.0, in_features, out_features)
