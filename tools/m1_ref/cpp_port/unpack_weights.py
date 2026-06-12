#!/usr/bin/env python3
"""Unpack the m1_ref weight npz archives into per-tensor .npy files for the C++
port to load via tools/sparse_spike/npy.hpp. Keys preserved verbatim as filenames
(dots are fine). fp32 throughout. One-time prep; lazy per-key so memory stays small.

rope_phases (ss_flow) stays [4096,64,2] real/imag — the C++ graph reconstructs the
complex apply from it.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "..", "weights")
OUT = os.path.join(HERE, "weights_npy")

MODELS = ["ss_dec", "dinov3", "ss_flow"]


def unpack(name):
    src = os.path.join(WDIR, f"{name}.npz")
    dst = os.path.join(OUT, name)
    os.makedirs(dst, exist_ok=True)
    z = np.load(src)
    keys = list(z.files)
    print(f"[{name}] {len(keys)} tensors -> {dst}", flush=True)
    for i, k in enumerate(keys):
        a = np.asarray(z[k], dtype=np.float32)
        np.save(os.path.join(dst, k + ".npy"), a)
        if i % 100 == 0:
            print(f"  [{name}] {i}/{len(keys)} {k} {a.shape}", flush=True)
    z.close()
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or MODELS
    for m in which:
        unpack(m)
    print("ALL DONE")
