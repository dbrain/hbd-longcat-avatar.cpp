#!/usr/bin/env python3
"""Convert NAVA's umT5 encoder (.pth, Wan naming, bf16) -> sd.cpp t5xxl gguf.
This is the CORRECT umT5 for NAVA (Wan2.2 models_t5_umt5-xxl-enc-bf16.pth) — NOT
longcat's text_encoder (different weights). Maps Wan names -> HF UMT5 names that the
cpp T5 loader expects (same as tools/convert_umt5.py output), then quantizes like it.
Usage: convert_nava_umt5_pth.py <in.pth> <out.gguf> <f16|q8_0>
"""
import sys, re, time, numpy as np, torch
import gguf
from gguf import GGMLQuantizationType as QT
from gguf import quants

INP, OUT, DT = sys.argv[1], sys.argv[2], sys.argv[3]
QMAP = {"f16": QT.F16, "q8_0": QT.Q8_0}
big_qt = QMAP[DT]
PREFIX = "text_encoders.t5xxl.transformer."

def map_name(k):
    if k == "token_embedding.weight": return "shared.weight"
    if k == "norm.weight":            return "encoder.final_layer_norm.weight"
    m = re.match(r"blocks\.(\d+)\.(.+)", k)
    if not m: return None
    n, rest = m.group(1), m.group(2)
    b = f"encoder.block.{n}.layer"
    tbl = {
        "norm1.weight":                   f"{b}.0.layer_norm.weight",
        "attn.q.weight":                  f"{b}.0.SelfAttention.q.weight",
        "attn.k.weight":                  f"{b}.0.SelfAttention.k.weight",
        "attn.v.weight":                  f"{b}.0.SelfAttention.v.weight",
        "attn.o.weight":                  f"{b}.0.SelfAttention.o.weight",
        "pos_embedding.embedding.weight": f"{b}.0.SelfAttention.relative_attention_bias.weight",
        "norm2.weight":                   f"{b}.1.layer_norm.weight",
        "ffn.gate.0.weight":              f"{b}.1.DenseReluDense.wi_0.weight",  # gelu-activated
        "ffn.fc1.weight":                 f"{b}.1.DenseReluDense.wi_1.weight",  # linear
        "ffn.fc2.weight":                 f"{b}.1.DenseReluDense.wo.weight",
    }
    return tbl.get(rest)

sd = torch.load(INP, map_location="cpu", weights_only=False)
writer = gguf.GGUFWriter(OUT, arch="umt5-xxl", use_temp_file=True)
writer.add_description("NAVA Wan2.2 umT5-XXL encoder (-> sd.cpp t5xxl names)")
t0 = time.time(); nq = nc = 0; total = 0; unmapped = []
for k in sorted(sd.keys()):
    hf = map_name(k)
    if hf is None:
        unmapped.append(k); continue
    arr = sd[k].float().numpy()
    total += arr.size
    out_name = PREFIX + hf
    is_relbias = hf.endswith("relative_attention_bias.weight")
    if big_qt != QT.F16 and arr.ndim == 2 and not is_relbias and arr.shape[-1] % 32 == 0:
        writer.add_tensor(out_name, quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), big_qt), raw_dtype=big_qt)
        nq += 1
    else:
        writer.add_tensor(out_name, np.ascontiguousarray(arr, dtype=np.float16)); nc += 1
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
import os
print(f"wrote {OUT}  tensors: {nq} {DT} + {nc} f16 = {nq+nc}  params {total/1e9:.2f}B  {os.path.getsize(OUT)/1e9:.2f}GB  {time.time()-t0:.1f}s")
if unmapped: print("UNMAPPED (check!):", unmapped)
