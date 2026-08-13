#!/usr/bin/env python3
"""Dump every tensor in p3sam.safetensors to fp32 .npy (sanitised names) +
a manifest, for the native C++ port to mmap. CPU-only (no torch/GPU)."""
import os, sys, json, numpy as np
from safetensors import safe_open

CKPT = os.environ.get("P3SAM_CKPT", "/work/P3-SAM/weights/p3sam/p3sam.safetensors")
OUT  = os.environ.get("P3SAM_W", "/wout")
os.makedirs(OUT, exist_ok=True)
man = {}
with safe_open(CKPT, framework="numpy") as f:
    keys = list(f.keys())
    for k in keys:
        t = f.get_tensor(k)
        a = np.ascontiguousarray(t.astype(np.float32))
        name = k.replace(".", "__")
        np.save(os.path.join(OUT, name + ".npy"), a)
        man[k] = {"file": name + ".npy", "shape": list(a.shape), "dtype": "float32"}
print(f"dumped {len(keys)} tensors -> {OUT}", flush=True)
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=0)
# quick structural summary
import collections
pref = collections.Counter(k.split(".")[0] for k in keys)
print("top-level prefixes:", dict(pref))
# list head-mlp keys
for k in keys:
    if not k.startswith("sonata."):
        print("  HEAD", k, man[k]["shape"])
