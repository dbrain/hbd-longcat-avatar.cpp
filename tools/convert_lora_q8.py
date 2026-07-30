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
QK8_0 = 32
ALIGN = 32

# WHY --dtype f16 EXISTS (perf, not size):
# ggml_ext_linear's fast path is self-gated on `w->type == GGML_TYPE_NVFP4` (ggml_extend.hpp:1039),
# so the BASE Linear runs cuBLASLt FP4 on tensor cores with an F16 dst, while a Q8_0 LoRA falls to
# ggml's generic quantised mul_mat at F32 dst — no tensor cores. At 1920x1088 the LoRA is only
# ~3.1% of the FLOPs (rank 64, M=38760, K=N=4096) yet costs ~23.6% of sampling: a ~7.6x throughput
# gap. F16 src0 goes through cuBLAS GEMM instead, which does use tensor cores. Costs ~2x the VRAM
# of Q8_0 (still ~half of BF16 for a 2-byte type... identical, in fact — F16 and BF16 are both
# 2 bytes, so this is purely Q8_0's 2x saving given back in exchange for the faster path).


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--dtype', choices=['q8_0', 'f16'], default='q8_0',
                    help='q8_0 = smallest (0.4%% error); f16 = 2x the size but lands on a '
                         'tensor-core GEMM path instead of ggml quantised mul_mat')
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
        elif len(ne) == 2 and ne[0] % QK8_0 == 0:
            plan.append((name, ne, GGML_TYPE_Q8_0)); n_q8 += 1
        else:
            plan.append((name, ne, 0)); n_f32 += 1   # F32 fallback (never hit for these LoRAs)
    print(f"plan: {n_q8} {a.dtype.upper()} + {n_f32} F32 = {len(plan)} tensors")

    def nbytes_of(tt, ne):
        n = int(np.prod(ne))
        if tt == GGML_TYPE_Q8_0:
            return (n // QK8_0) * (2 + QK8_0)
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
