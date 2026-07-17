#!/usr/bin/env python3
"""Extract shape_enc_next_dc safetensors -> per-tensor fp32 .npy for the C++ sparse-VAE
encoder port (shape_slat_encoder.hpp), then those get packed to one GGUF via pack_gguf.

CPU/numpy only (no torch, no GPU). Mirrors the layout conventions the DECODER npy dir
(weights_npy/shape_dec) already uses, validated bit-exact in sparse_vae.hpp:
  - sparse conv weight: torch flex_gemm [OC,3,3,3,IC]  ->  spike [V=27, IC, OC]
        (reshape [OC,27,IC] z-major kd*9+kh*3+kw, then transpose to [27,IC,OC])
  - Linear / norm / bias: kept verbatim torch layout ([out,in] for Linear, [C] for 1D).
fp16 stored weights are upcast to fp32 (the host port is fp32 throughout).
"""
import os, sys, json, struct
import numpy as np

SRC = "/mnt/hdd/pixal3d_tex/trellis2_4b/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_npy", "shape_enc")

_DTYPE = {"F16": np.float16, "F32": np.float32, "BF16": None, "F64": np.float64}

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for k, meta in hdr.items():
        if k == "__metadata__":
            continue
        dt = meta["dtype"]
        if dt == "BF16":
            raw = np.frombuffer(blob[meta["data_offsets"][0]:meta["data_offsets"][1]], dtype=np.uint16)
            a = (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
        else:
            npdt = _DTYPE[dt]
            a = np.frombuffer(blob[meta["data_offsets"][0]:meta["data_offsets"][1]], dtype=npdt).reshape(meta["shape"])
        out[k] = np.ascontiguousarray(a.astype(np.float32))
    return out

def to_spike_conv(w):
    # torch flex_gemm [OC, kd, kh, kw, IC] -> spike [V=27, IC, OC]
    OC, kd, kh, kw, IC = w.shape
    assert (kd, kh, kw) == (3, 3, 3), w.shape
    w = w.reshape(OC, kd * kh * kw, IC)   # [OC, 27, IC]  (z-major kd*9+kh*3+kw)
    return np.ascontiguousarray(w.transpose(1, 2, 0))   # [27, IC, OC]

def main():
    os.makedirs(OUT, exist_ok=True)
    tensors = load_safetensors(SRC)
    print(f"[shape_enc] {len(tensors)} tensors from {SRC}")
    n_conv = 0
    for k in sorted(tensors):
        a = tensors[k]
        # sparse conv kernels are the only 5D tensors; the spike layout transpose applies to
        # any *.conv.weight / *.conv1.weight / *.conv2.weight (5D). Linears stay [out,in].
        if a.ndim == 5:
            a = to_spike_conv(a)
            n_conv += 1
        np.save(os.path.join(OUT, k + ".npy"), np.ascontiguousarray(a, dtype=np.float32))
    print(f"[shape_enc] wrote {len(tensors)} .npy ({n_conv} conv kernels transposed to spike [27,IC,OC]) -> {OUT}")

if __name__ == "__main__":
    main()
