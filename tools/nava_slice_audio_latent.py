#!/usr/bin/env python3
"""Slice an encoded audio latent [128, T] into per-segment [128, seg_len] bins.

Drives multi-segment continuation from ONE continuous external audio stream:
encode the whole wav once (no chunk-boundary edge effects), then hand each render
segment its contiguous slice so speech flows unbroken across the chained clip.

Bin format mirrors the C++ write_bin: int32 n_dims, int32 name_len, int32 ttype(F32=0),
n_dims*int32 shape, name bytes, then numel float32 in token-major layout (shape[0]=128
channels contiguous per token).

Usage: nava_slice_audio_latent.py in_latent.bin out_prefix seg_len n_segs [overlap]
"""
import sys, struct, numpy as np

inp, prefix, seg_len, n_segs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
overlap = int(sys.argv[5]) if len(sys.argv) > 5 else 0

with open(inp, "rb") as f:
    n_dims, name_len, ttype = struct.unpack("<iii", f.read(12))
    shape = struct.unpack("<" + "i" * n_dims, f.read(4 * n_dims))
    f.read(name_len)
    data = np.frombuffer(f.read(), dtype=np.float32)
assert n_dims == 2 and shape[0] == 128, f"expected [128,T], got {shape}"
C, T = shape
data = data.reshape(T, C)  # token-major: data[t, c]
print(f"loaded latent {shape} ({T} tokens, {T/25:.2f}s @25tok/s)")

for k in range(n_segs):
    start = k * (seg_len - overlap)
    end = start + seg_len
    if end > T:
        # clamp: pad by repeating the last token (keeps shape exact)
        seg = data[start:T]
        pad = np.repeat(data[T - 1:T], end - T, axis=0)
        seg = np.concatenate([seg, pad], axis=0)
        print(f"  seg{k}: tokens [{start}:{T}] + {end-T} pad")
    else:
        seg = data[start:end]
        print(f"  seg{k}: tokens [{start}:{end}]")
    out = f"{prefix}{k}.bin"
    name = b"audio_latent"
    with open(out, "wb") as g:
        g.write(struct.pack("<iii", 2, len(name), 0))
        g.write(struct.pack("<ii", 128, seg_len))
        g.write(name)
        g.write(seg.astype(np.float32).tobytes())
    print(f"    -> {out}  [128,{seg_len}]")
