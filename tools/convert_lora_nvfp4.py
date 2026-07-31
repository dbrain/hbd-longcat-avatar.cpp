#!/usr/bin/env python3
"""Convert a LoRA (safetensors, F16/BF16/F32) -> an **NVFP4** gguf for RUNTIME use.

Sibling of tools/convert_lora_q8.py. Same container, same names, same dim order — only the
tensor type changes. Read that file first; only the NVFP4-specific reasoning is repeated here.

WHY NVFP4 AND NOT Q8_0 (perf + VRAM, on a Blackwell FP4 base)
-------------------------------------------------------------
A Q8_0 adapter tensor is src0 of a MUL_MAT whose src1 is the F32 residual stream, so
ggml_cuda_mul_mat() falls through to MMQ (ggml-cuda.cu:1900 is skipped for a non-NVFP4 src0)
and quantize_mmq_q8_1_cuda re-quantises the WHOLE activation on every call, every step.
An NVFP4 src0 is taken by the cuBLASLt blockscaled FP4 GEMM instead (ggml-cuda.cu:1898-1904),
which accepts an F32 **or** F16 src1 (nvfp4-cublaslt.cu:439-440) and runs on FP4 tensor cores.
It also halves the adapter on disk and in VRAM (4.5 vs 8.5 bits/elem incl. scales).

TWO PRECONDITIONS, BOTH CHECKED HERE — a violation is a SLOW ADAPTER, NOT AN ERROR
-----------------------------------------------------------------------------------
The FP4 GEMM's shared bail list (ggml-cuda.cu ggml_cuda_nvfp4_cublaslt_shapes_ok,
mirrored at nvfp4-cublaslt.cu:438-453) rejects, and the node silently falls back to
MMQ/dequant-cuBLAS at F32 — i.e. slower than the Q8_0 adapter it replaced, with 4-bit
weights. So this tool refuses to emit such a tensor as NVFP4 unless told what to do:

  1. **K % 64 == 0**, K = ggml ne[0] = the LAST safetensors dim. For a rank-r LoRA the `up`
     GEMM has K = r, so **r must be a multiple of 64**. This is exactly why an NVFP4 adapter
     was rejected for LTX at rank 32/64 — at rank 256 (krea2 identity-edit) it passes.
  2. **2-D, contiguous, no batch dims.** LoRA A/B factors always are; conv2d LoRAs are not.

`--on-unaligned` decides: `fail` (default, loudest), or `q8_0`/`f16` to hold just those
tensors at the old type. A mixed adapter is legal but you MUST know it is mixed, so it is
never the default and every held tensor is listed by name.

FLAT, NOT UNFOLDED — NO `.wglobal` SIDECARS
-------------------------------------------
The quantiser is tools/build_folded_nvfp4.py:quant_nvfp4_unfolded(W, flat=True): per 64
elements, 4 ue4m3 sub-block scale bytes + 32 packed e2m1 nibble bytes (block_nvfp4,
ggml-common.h:221-227). FLAT means the block scale is absolute and the per-tensor global is
1.0. That matches krea2's FLAT base (zero sidecars), and an unregistered name reads back as
exactly 1.0 anyway (nvfp4-cublaslt.cu:243-247). Emitting UNFOLDED would need a
`.wglobal` companion tensor AND a registration path the LoRA loader does not have — every
matmul that missed cuBLASLt would then be wrong by a per-tensor constant.

GAIN CORRECTION (--gain-correct, ON BY DEFAULT) — MEASURED, NOT ASSUMED
-----------------------------------------------------------------------
Plain RTN into the e2m1 grid does not just add noise, it SHRINKS: everything under a quarter
of a block's step rounds to zero, so the reconstruction is systematically short. Measured on
the krea2 identity-edit adapter, 5 modules: per-tensor projection 0.9645, and because BOTH
factors shrink, B@A lands at **projection 0.9256 with cosine 0.9985** — i.e. almost no
direction error, an 7.4% MAGNITUDE loss. An adapter silently running at 0.93x strength is
exactly the kind of thing that gets misread as "fp4 broke the LoRA".

RTN is scale-equivariant, so that deficit is a pure gain and can be cancelled BEFORE
quantising: scale the tensor by 1/p, quantise that, and the stored block scales absorb it.
Costs nothing at run time, adds no tensors, needs no engine change. Measured: projection
0.9256 -> **0.9986**, relative error 0.0903 -> **0.0539**, cosine unchanged. Typical pre-gain
1.037. `--no-gain-correct` reproduces the literal ggml-RTN baseline.

What does NOT work, so nobody re-tries it: refitting the stored block scale by least squares
(<q,w>/<q,q>, codes frozen) and a joint scale-multiplier grid search both leave relerr,
projection and cosine unchanged to 4 decimals — the ue4m3 scale grid is 12.5% coarse, so a
~4% correction cannot be expressed per block. The correction has to be applied to the VALUES
before quantisation, which is what this does.

THIS IS NOT THE FOLD THAT LOST THE MOLE
---------------------------------------
That was round-to-nearest of a *small delta* onto the base's already-frozen fp4 grid
(~0.36 projection). Here the LoRA FACTORS are quantised, with scales derived from their own
values — the same operation Q8_0 already does at 0.0022 relative error, just at 4 bits.
It will cost more. `tools/verify_lora_q8.py --gguf <this>` measures how much; only a
behavioural assay decides whether that matters.

Usage:
  convert_lora_nvfp4.py <in.safetensors> <out.gguf> [--on-unaligned fail|q8_0|f16]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_folded_nvfp4 import quant_nvfp4_unfolded          # noqa: E402
from convert_lora_q8 import st_open, bf16_to_f32, quantize_q8_0  # noqa: E402
from verify_lora_q8 import dequant_nvfp4                    # noqa: E402

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_NVFP4 = 40

QK8_0 = 32
QK_NVFP4 = 64          # ggml-common.h:221
NVFP4_BLOCK_BYTES = 4 + 32   # 4 ue4m3 sub-block scales + 32 packed nibbles
ALIGN = 32

TYPE_NAME = {GGML_TYPE_F32: 'F32', GGML_TYPE_F16: 'F16',
             GGML_TYPE_Q8_0: 'Q8_0', GGML_TYPE_NVFP4: 'NVFP4'}


def nbytes_of(tt, ne):
    n = int(np.prod(ne))
    if tt == GGML_TYPE_NVFP4:
        return (n // QK_NVFP4) * NVFP4_BLOCK_BYTES
    if tt == GGML_TYPE_Q8_0:
        return (n // QK8_0) * (2 + QK8_0)
    if tt == GGML_TYPE_F16:
        return n * 2
    return n * 4


def quant_nvfp4_gain_corrected(W, iters=1):
    """FLAT NVFP4 bytes for W, with the RTN shrink cancelled. Returns (bytes, gain).

    rec(g*W) ~= g*p0*W because RTN is scale-equivariant, so one measurement pass fixes g:
    g <- g / (<rec(g*W), W> / <W, W>). The gain lives entirely in the stored block scales —
    no sidecar, no extra tensor, nothing for the engine to know about. Verified against the
    verifier's own dequantiser, which is also what proves the nibble packing round-trips.
    """
    ne = [W.shape[1], W.shape[0]]
    denom = float((W * W).sum())
    g = 1.0
    if denom > 0:
        for _ in range(iters):
            data, _ = quant_nvfp4_unfolded(W * g, flat=True)
            p = float((dequant_nvfp4(data, ne) * W).sum()) / denom
            if not np.isfinite(p) or p <= 0:
                g = 1.0
                break
            g = g / p
    data, wg = quant_nvfp4_unfolded(W * g, flat=True)
    assert wg == 1.0
    return data, g


def fp4_eligible(ne):
    """Mirrors ggml_cuda_nvfp4_cublaslt_shapes_ok()'s weight-side conditions."""
    return len(ne) == 2 and ne[0] >= QK_NVFP4 and ne[0] % QK_NVFP4 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--on-unaligned', choices=['fail', 'q8_0', 'f16'], default='fail',
                    help='what to do with a tensor whose ggml ne[0] is not a multiple of 64 '
                         '(the FP4 GEMM would refuse it and fall back to a SLOWER path than '
                         'Q8_0). Default: refuse to write the file at all.')
    ap.add_argument('--no-gain-correct', dest='gain_correct', action='store_false', default=True,
                    help='emit the literal ggml-RTN quantisation, which lands ~7%% SHORT on '
                         'the reconstructed B@A (projection 0.926). Only for A/B against the '
                         'corrected default.')
    ap.add_argument('--gain-iters', type=int, default=1,
                    help='measurement passes for the gain (1 is enough; RTN is '
                         'scale-equivariant so the first-order fix converges immediately)')
    a = ap.parse_args()

    fs_, base, h = st_open(a.src)
    names = list(h.keys())

    # ---- ne[0] % 64 AUDIT (the whole reason this tool can exist at rank 256) -------------
    plan, unaligned = [], []
    for name in names:
        shp = h[name]['shape']
        ne = list(reversed(shp))          # safetensors row-major [d0,d1] -> ggml ne [d1,d0]
        if fp4_eligible(ne):
            plan.append((name, ne, GGML_TYPE_NVFP4))
        else:
            unaligned.append((name, shp, ne))
            plan.append((name, ne, None))     # resolved below

    print(f"--- ne[0] % 64 AUDIT over {len(names)} tensors ---")
    ranks = sorted({ne[0] for _, ne, _ in plan})
    print(f"  distinct ne[0]: {ranks}")
    print(f"  FP4-eligible (2-D, ne[0] % 64 == 0): {len(names) - len(unaligned)}/{len(names)}")
    if unaligned:
        print(f"  !!! {len(unaligned)} tensor(s) NOT FP4-eligible — the cuBLASLt FP4 GEMM "
              f"would REFUSE these and fall back to MMQ at F32:")
        for name, shp, ne in unaligned[:20]:
            print(f"      {name}  st{shp} -> ne{ne}  (ne[0] % 64 = {ne[0] % QK_NVFP4})")
        if len(unaligned) > 20:
            print(f"      ... and {len(unaligned) - 20} more")
        if a.on_unaligned == 'fail':
            sys.exit("REFUSING to write a silently-mixed adapter. Re-run with "
                     "--on-unaligned q8_0 (or f16) to hold exactly those tensors at the old "
                     "type, and expect NO FP4 speedup on the modules they belong to.")
        held = GGML_TYPE_Q8_0 if a.on_unaligned == 'q8_0' else GGML_TYPE_F16
        print(f"  --on-unaligned {a.on_unaligned}: holding those "
              f"{len(unaligned)} tensor(s) at {TYPE_NAME[held]}. THE ADAPTER IS MIXED.")
    else:
        print("  ALL tensors FP4-eligible — no fallback dtype needed.")

    resolved = []
    for name, ne, tt in plan:
        if tt is None:
            if len(ne) == 2 and ne[0] % QK8_0 == 0 and a.on_unaligned == 'q8_0':
                tt = GGML_TYPE_Q8_0
            elif a.on_unaligned == 'f16' or len(ne) != 2:
                tt = GGML_TYPE_F16
            else:
                tt = GGML_TYPE_F16   # not even Q8_0-blockable
                print(f"  note: {name} ne{ne} is not Q8_0-blockable either -> F16")
        resolved.append((name, ne, tt))
    plan = resolved

    counts = {}
    for _, _, tt in plan:
        counts[TYPE_NAME[tt]] = counts.get(TYPE_NAME[tt], 0) + 1
    print("plan: " + " + ".join(f"{v} {k}" for k, v in sorted(counts.items()))
          + f" = {len(plan)} tensors")

    # ---- gguf header (nkv = 0, byte-compatible with convert_lora_q8.py's output) ---------
    o = open(a.out, 'wb')
    o.write(b'GGUF')
    o.write(np.uint32(3).tobytes())
    o.write(np.uint64(len(plan)).tobytes())
    o.write(np.uint64(0).tobytes())          # nkv = 0
    off = 0
    for name, ne, tt in plan:
        nb_ = name.encode()
        o.write(np.uint64(len(nb_)).tobytes()); o.write(nb_)
        o.write(np.uint32(len(ne)).tobytes())
        for d_ in ne:
            o.write(np.uint64(d_).tobytes())
        o.write(np.uint32(tt).tobytes()); o.write(np.uint64(off).tobytes())
        nb = nbytes_of(tt, ne)
        off += nb + ((-nb) % ALIGN)
    cur = o.tell(); o.write(b'\x00' * ((-cur) % ALIGN))

    gains = []
    for name, ne, tt in plan:
        m = h[name]; a0, b0 = m['data_offsets']
        fs_.seek(base + a0)
        raw = np.frombuffer(fs_.read(b0 - a0), dtype=np.uint8)
        st_dtype = m['dtype']
        if st_dtype == 'BF16':
            x = bf16_to_f32(raw)
        elif st_dtype in ('F16', 'FP16'):
            x = raw.view(np.float16).astype(np.float32)
        elif st_dtype in ('F32', 'FP32'):
            x = raw.view(np.float32)
        else:
            raise SystemExit(f"{name}: unsupported safetensors dtype {st_dtype}")
        x = x.reshape(m['shape'])
        if tt == GGML_TYPE_NVFP4:
            # float64 in: the block scale is derived from an amax and then divides the row;
            # doing that in f32 costs a needless rounding before the 4-bit rounding that
            # actually matters. flat=True => wg == 1.0, no `.wglobal` sibling emitted.
            x64 = np.asarray(x, dtype=np.float64)
            if a.gain_correct:
                data, g = quant_nvfp4_gain_corrected(x64, a.gain_iters)
                gains.append(g)
            else:
                data, wg = quant_nvfp4_unfolded(x64, flat=True)
                assert wg == 1.0, f"{name}: FLAT quantiser returned wglobal {wg}"
        elif tt == GGML_TYPE_Q8_0:
            data = quantize_q8_0(np.asarray(x, dtype=np.float32))
        elif tt == GGML_TYPE_F16:
            data = np.ascontiguousarray(x, dtype=np.float16).tobytes()
        else:
            data = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        nb = nbytes_of(tt, ne)
        assert len(data) == nb, f"{name}: {len(data)} vs {nb}"
        o.write(data); o.write(b'\x00' * ((-len(data)) % ALIGN))
    o.close()
    if gains:
        gm = np.array(gains)
        print(f"gain correction: {len(gains)} tensor(s) pre-scaled, mean {gm.mean():.4f} "
              f"(min {gm.min():.4f}, max {gm.max():.4f}) — cancels the RTN shrink; a gain "
              f"far from ~1.04 on some tensor means it quantises unusually badly, LOOK at it")
    elif not a.gain_correct:
        print("gain correction: DISABLED (--no-gain-correct) — expect B@A ~7% SHORT")
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes, "
          f"{os.path.getsize(a.out)/os.path.getsize(a.src)*100:.1f}% of source)")
    print("NEXT: tools/verify_lora_q8.py --src <src> --gguf <out>  (it reads NVFP4 too), "
          "then a BEHAVIOURAL assay — projection does not predict whether the adapter "
          "still does its job.")


if __name__ == '__main__':
    main()
