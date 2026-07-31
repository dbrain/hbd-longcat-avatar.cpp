#!/usr/bin/env python3
"""Convert an LTX-2.3 LoRA (engine-native BF16 safetensors) -> a Q8_0 gguf for RUNTIME use.

WHY Q8_0 AND NOT A FOLD
-----------------------
Folding a LoRA into an nvfp4 base is lossy by construction: round-to-nearest discards every
sub-half-step component, so only ~36% of the intended delta survives (tools/verify_fold.py C3).
The runtime path is exact. This tool exists so the runtime path is also CHEAP.

WHY Q8_0 AND NOT MXFP8
----------------------
Within a few percent on size (8.5 vs 8.25 bits/elem) but ~0.4% relative error vs e4m3's 3
mantissa bits, and MXFP8 is Blackwell-only — the 3060 (sm86) has no FP8 path, so an MXFP8
adapter would device-gate a feature that currently runs anywhere.

WHY THIS ACTUALLY SAVES VRAM (verified in-source, not assumed)
--------------------------------------------------------------
LoraModel::load_from_file() creates each tensor with `ts.type` — the ON-DISK type, verbatim —
and MultiLoraAdapter takes the get_out_diff() output-delta path for quantised base weights,
where lora_up/lora_down are consumed AS-IS (src/model/adapter/lora.hpp; only is_conv2d forces
a cast to F16). So a Q8_0 adapter stays Q8_0 in VRAM. It is NOT dequantised to F32 at load.

NAMING
------
Tensor names are emitted UNCHANGED (`diffusion_model.<stem>.lora_{A,B}.weight`).
src/name_conversion.cpp maps `.lora_A.weight` -> `.weight.lora_down` and `.lora_B.weight` ->
`.weight.lora_up` on the NAME STRING, independent of container, so a gguf with the same names
resolves exactly as the safetensors did.

NOTE ON PRECEDENCE: SDGenerationParams::extract_and_remove_lora tries extensions in the order
{".gguf", ".safetensors", ...}, so writing <name>.gguf next to <name>.safetensors makes the
gguf WIN for `<lora:<name>:mult>`. That is the intent — but it is a silent switch, so deploy
deliberately and keep the .safetensors as the archival original.

Usage:
  convert_lora_q8.py <in.safetensors> <out.gguf>
"""
import sys, json, struct, argparse
import numpy as np

GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_NVFP4 = 40
QK8_0 = 32
QK_NVFP4 = 64          # ggml's NVFP4 block: 4 e4m3 sub-scales + 32 packed nibble bytes
ALIGN = 32

# WHY --dtype f16 EXISTS (perf, not size):
# ggml_ext_linear's fast path is self-gated on `w->type == GGML_TYPE_NVFP4` (ggml_extend.hpp:1039),
# so the BASE Linear runs cuBLASLt FP4 on tensor cores with an F16 dst, while a Q8_0 LoRA falls to
# ggml's generic quantised mul_mat at F32 dst — no tensor cores. At 1920x1088 the LoRA is only
# ~3.1% of the FLOPs (rank 64, M=38760, K=N=4096) yet costs ~23.6% of sampling: a ~7.6x throughput
# gap. F16 src0 goes through cuBLAS GEMM instead, which does use tensor cores. Costs ~2x the VRAM
# of Q8_0 (still ~half of BF16 for a 2-byte type... identical, in fact — F16 and BF16 are both
# 2 bytes, so this is purely Q8_0's 2x saving given back in exchange for the faster path).
#
# ⚠️ MEASURED 2026-07-30 on the krea2 edit shape (1024^2, one reference, 10 steps, 5060 Ti):
# **--dtype f16 IS SLOWER, AT EVERY RANK.** Sampling with the identity-edit adapter:
#     rank 256   Q8_0 40.36 s   F16 43.65 s   (+3.3)
#     rank 128   Q8_0 39.74 s   F16 42.73 s   (+3.0)
#     rank  64   Q8_0 39.73 s   F16 42.36 s   (+2.6)
# The prediction above is not what the hardware does, and the penalty is nearly INDEPENDENT of
# adapter size (r64-F16 is 457 MB and still loses to r256-Q8's 971 MB), so it is not weight
# bandwidth either. Mechanism: an F16 src0 pushes the lora_up GEMM off ggml's MMQ int8 path onto
# a cuBLAS HGEMM that reads the weight at 16 bits instead of 8.5, and the down GEMM was already
# on cuBLAS in both dtypes (ggml-cuda.cu bails to mul_mat_cublas for any non-F32 src1, which an
# F16 residual stream always is). Keep the flag for A/Bs; do not ship it.
# See ~/handoffs/docker/krea2/RESULTS-runtime-lora-perf.md.
#
# WHY --dtype nvfp4 EXISTS
# ------------------------
# NVFP4 is the one src0 type that changes the DISPATCH rather than just the operand width:
# ggml_cuda_mul_mat() routes it to the cuBLASLt blockscaled-FP4 GEMM ahead of every F32-only
# gate, and it is the only weight type ggml_ext_linear will emit an F16 matmul DESTINATION for
# — which is what would let a LoRA'd Krea2 keep its F16 residual stream instead of being forced
# back to F32 (krea2.hpp's krea2_dit_f16_device_ok bails on any attached weight adapter).
#
# FLAT convention only (wg == 1, no `.wglobal` sibling), matching krea2's own FLAT base. An
# UNFOLDED adapter is only correct if every one of its matmuls provably reaches cuBLASLt —
# anything that falls back to MMQ or dequant multiplies by 1.0 instead of the wglobal and comes
# out silently mis-scaled — and a runtime LoRA's shapes are not under our control. Flat is exact
# under every fallback. Requires ne[0] % 64 == 0 on BOTH factors, i.e. rank % 64 == 0: fine at
# rank 64/128/256, impossible at rank 32.


def st_open(p):
    f = open(p, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return f, 8 + n, h


def bf16_to_f32(u8):
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def quantize_q8_0(x):
    """x: [rows, k] float32, k % 32 == 0 -> packed block_q8_0 bytes.
    Mirrors ggml quantize_row_q8_0_ref: d = amax/127, qs = round(x/d)."""
    rows, k = x.shape
    nb = k // QK8_0
    b = x.reshape(rows * nb, QK8_0).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    idd = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0))
    # roundf = round-half-AWAY-from-zero. np.round is half-to-EVEN, so it cannot be used.
    # Must be trunc(), NOT floor(): floor(-1.8 - 0.5) = -3 pushes every negative one LSB away
    # from zero, which shows up as a systematic ~+1.2% gain on the reconstructed B@A delta.
    q = np.trunc(b * idd[:, None] + np.where(b >= 0, 0.5, -0.5)).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    out = np.empty((rows * nb, 2 + QK8_0), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


E2M1_MAX, E4M3_MAX, E4M3_MIN_SUB = 6.0, 448.0, 2.0 ** -9


def _e4m3_dec_byte(u8):
    u = u8.astype(np.int32); e = (u >> 3) & 0xF; m = u & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float64), -9), np.ldexp(1.0 + m / 8.0, e - 7))
    return np.where((u & 0x7F) == 0, 0.0, v)


_PB = np.arange(128, dtype=np.uint8); _PV = _e4m3_dec_byte(_PB)
_ORD = np.argsort(_PV); _SV = _PV[_ORD]; _SB = _PB[_ORD]
_LUTM = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e4m3_enc_pos(x):
    x = np.minimum(np.asarray(x, dtype=np.float64), E4M3_MAX)
    i = np.clip(np.searchsorted(_SV, x), 0, len(_SV) - 1); lo = np.clip(i - 1, 0, len(_SV) - 1)
    return _SB[np.where(np.abs(_SV[i] - x) < np.abs(_SV[lo] - x), i, lo)].astype(np.uint8)


def _e2m1_nearest(t):
    s = (t < 0).astype(np.uint8); m = np.abs(t)
    i = np.clip(np.searchsorted(_LUTM, m), 0, 7); lo = np.clip(i - 1, 0, 7)
    return (s << 3) | np.where(np.abs(_LUTM[i] - m) < np.abs(_LUTM[lo] - m), i, lo).astype(np.uint8)


def quantize_nvfp4_flat(W):
    """W [rows, k] -> packed ggml block_nvfp4 bytes, FLAT convention (wglobal == 1).

    Mirrors tools/build_folded_nvfp4.py's quant_nvfp4_unfolded(flat=True); kept here rather than
    imported so this converter stays a single self-contained file like the rest of tools/.
    Per-16 e4m3 scale = ue4m3(amax_sub / 6) in ABSOLUTE units, then e2m1 nibbles against the
    decoded scale (not the ideal one) so the reconstruction matches what the kernel will do.
    """
    rows, k = W.shape
    W = np.asarray(W, dtype=np.float64)
    b16 = W.reshape(rows, k // 16, 16)
    sc = np.clip(np.abs(b16).max(axis=2) / E2M1_MAX, E4M3_MIN_SUB, E4M3_MAX)
    byte = _e4m3_enc_pos(sc)
    eff = np.repeat(_e4m3_dec_byte(byte), 16, axis=1)
    code = _e2m1_nearest(np.where(eff > 0, W / eff, 0.0))
    nblk = k // QK_NVFP4
    oc = code.reshape(rows, nblk, QK_NVFP4)
    qs = np.empty((rows, nblk, 32), dtype=np.uint8)
    for s in range(4):                                   # low nibble = first 8 of each 16
        sub = oc[:, :, s * 16:s * 16 + 16]
        qs[:, :, s * 8:s * 8 + 8] = sub[:, :, 0:8] | (sub[:, :, 8:16] << 4)
    db = byte.reshape(rows, nblk, 4).astype(np.uint8)
    return np.concatenate([db, qs], axis=2).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--dtype', choices=['q8_0', 'f16', 'nvfp4'], default='q8_0',
                    help='q8_0 = smallest sane default (0.4%% error); f16 = 2x the size and '
                         'MEASURABLY SLOWER (see the header); nvfp4 = half of q8_0 and the only '
                         'type that reaches the cuBLASLt FP4 GEMM, needs rank %% 64 == 0')
    a = ap.parse_args()

    fs_, base, h = st_open(a.src)
    names = list(h.keys())

    plan = []
    n_q8 = n_f32 = 0
    for name in names:
        shp = h[name]['shape']
        # safetensors is row-major [d0, d1, ...]; ggml ne is reversed, so ne[0] = last dim
        ne = list(reversed(shp))
        if a.dtype == 'f16':
            plan.append((name, ne, GGML_TYPE_F16)); n_q8 += 1
        elif a.dtype == 'nvfp4':
            # ne[0] % 64 is both the block-packing requirement AND the cuBLASLt FP4 gate's
            # K % 64. A tensor that fails it would silently fall back to F32 and cost more
            # than it saves, so refuse rather than emit a mixed adapter.
            if len(ne) != 2 or ne[0] % QK_NVFP4 != 0:
                raise SystemExit(f"{name}: ne[0]={ne[0]} is not a multiple of {QK_NVFP4} — "
                                 f"this adapter cannot be NVFP4 (rank must be a multiple of 64)")
            plan.append((name, ne, GGML_TYPE_NVFP4)); n_q8 += 1
        elif len(ne) == 2 and ne[0] % QK8_0 == 0:
            plan.append((name, ne, GGML_TYPE_Q8_0)); n_q8 += 1
        else:
            plan.append((name, ne, 0)); n_f32 += 1   # F32 fallback (never hit for these LoRAs)
    print(f"plan: {n_q8} {a.dtype.upper()} + {n_f32} F32 = {len(plan)} tensors")

    def nbytes_of(tt, ne):
        n = int(np.prod(ne))
        if tt == GGML_TYPE_Q8_0:
            return (n // QK8_0) * (2 + QK8_0)
        if tt == GGML_TYPE_NVFP4:
            return (n // QK_NVFP4) * (4 + 32)
        if tt == GGML_TYPE_F16:
            return n * 2
        return n * 4

    o = open(a.out, 'wb')
    o.write(b'GGUF')
    o.write(struct.pack('<I', 3))
    o.write(struct.pack('<Q', len(plan)))
    o.write(struct.pack('<Q', 0))          # nkv = 0 (matches import_ltx_nvfp4.py's ggufs)
    off = 0
    for name, ne, tt in plan:
        nb_ = name.encode()
        o.write(struct.pack('<Q', len(nb_))); o.write(nb_)
        o.write(struct.pack('<I', len(ne)))
        for d_ in ne:
            o.write(struct.pack('<Q', d_))
        o.write(struct.pack('<I', tt)); o.write(struct.pack('<Q', off))
        nb = nbytes_of(tt, ne)
        off += nb + ((-nb) % ALIGN)
    cur = o.tell(); o.write(b'\x00' * ((-cur) % ALIGN))

    for name, ne, tt in plan:
        m = h[name]; a0, b0 = m['data_offsets']
        fs_.seek(base + a0)
        raw = np.frombuffer(fs_.read(b0 - a0), dtype=np.uint8)
        # BF16 / F16 / F32 sources. F16 is not hypothetical: ai-toolkit writes the
        # krea2_edit adapters in F16, and view(float32) on those halves the tensor and
        # reinterprets pairs of halves as garbage floats — a silently wrong adapter.
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
        if tt == GGML_TYPE_Q8_0:
            data = quantize_q8_0(x)
        elif tt == GGML_TYPE_NVFP4:
            data = quantize_nvfp4_flat(x.reshape(-1, x.shape[-1]))
        elif tt == GGML_TYPE_F16:
            data = np.ascontiguousarray(x, dtype=np.float16).tobytes()
        else:
            data = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        nb = nbytes_of(tt, ne)
        assert len(data) == nb, f"{name}: {len(data)} vs {nb}"
        o.write(data); o.write(b'\x00' * ((-len(data)) % ALIGN))
    o.close()
    import os
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes, "
          f"{os.path.getsize(a.out)/os.path.getsize(a.src)*100:.1f}% of source)")


if __name__ == '__main__':
    main()
