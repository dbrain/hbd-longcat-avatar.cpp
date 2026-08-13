#!/usr/bin/env python3
# Extract the TRELLIS-2 CROSS-MODE tex DiT (slat_flow_imgshape2tex_dit_1_3B_1024_bf16) to per-tensor
# fp32 .npy for the native ggml port. NOTE: this is a DIFFERENT model from pixal3d's own proj-mode
# `slat_flow_imgshape2tex_1024` (cross_attn.cross_attn_block.* + proj_linear) — this one is the ORACLE
# the tex_goldens were captured from (cross_attn.to_q/to_kv directly, full image-token cross-attn).
import os, sys, numpy as np
from safetensors import safe_open

ST  = sys.argv[1] if len(sys.argv) > 1 else "/mnt/hdd/pixal3d_tex/trellis2_4b/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_npy", "trellis2_tex_1024")
os.makedirs(OUT, exist_ok=True)

f = safe_open(ST, "pt")
keys = list(f.keys())
print(f"[tex-dit] {len(keys)} tensors {ST} -> {OUT}", flush=True)
for i, k in enumerate(keys):
    a = f.get_tensor(k).float().cpu().numpy().astype(np.float32)
    a = np.ascontiguousarray(a)
    np.save(os.path.join(OUT, k + ".npy"), a)
    if i % 80 == 0:
        print(f"  {i}/{len(keys)} {k} {tuple(a.shape)}", flush=True)
print("[tex-dit] DONE", flush=True)
