#!/usr/bin/env python3
"""Compare two NAVA audio latent .bin files.

The expected layout is sd.cpp tensor-bin format with ggml-ne dims [128, T],
feature fastest. This reports whole-tensor stats plus per-time-step drift so
sampler/model changes can be judged without rendering audio.
"""
import argparse
import os

import numpy as np


def read_bin(path):
    with open(path, "rb") as f:
        n_dims, name_len, typ = np.fromfile(f, np.int32, 3)
        dims = [int(x) for x in np.fromfile(f, np.int32, int(n_dims))]
        name = f.read(int(name_len)).decode("utf-8", errors="replace")
        data = np.fromfile(f, np.float32)
    if typ != 0:
        raise ValueError(f"{path}: expected f32 type 0, got {typ}")
    if data.size != int(np.prod(dims)):
        raise ValueError(f"{path}: payload has {data.size} floats, dims imply {np.prod(dims)}")
    return name, dims, data.astype(np.float64)


def corrcoef(a, b):
    aa = a - a.mean()
    bb = b - b.mean()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den else 0.0


def describe(label, dims, x):
    print(
        f"{label:>10s} dims={dims} mean={x.mean():+.6f} std={x.std():.6f} "
        f"min={x.min():+.4f} max={x.max():+.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="reference latent, e.g. Python final audio latent")
    ap.add_argument("candidate", help="candidate latent, e.g. C++ final audio latent")
    ap.add_argument("--top", type=int, default=8, help="number of worst time steps to print")
    args = ap.parse_args()

    r_name, r_dims, ref = read_bin(args.reference)
    c_name, c_dims, cand = read_bin(args.candidate)
    if r_dims != c_dims:
        raise ValueError(f"dims differ: {r_dims} vs {c_dims}")
    if len(r_dims) != 2 or r_dims[0] != 128:
        raise ValueError(f"expected audio latent dims [128, T], got {r_dims}")

    print(f"reference: {args.reference} ({r_name})")
    print(f"candidate: {args.candidate} ({c_name})")
    describe("reference", r_dims, ref)
    describe("candidate", c_dims, cand)

    diff = cand - ref
    rel = np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12)
    print(
        f"overall  corr={corrcoef(ref, cand):.6f} "
        f"rmsdiff={np.sqrt(np.mean(diff * diff)):.6f} "
        f"reldiff={rel:.6f} std_ratio={cand.std() / (ref.std() + 1e-12):.6f}"
    )

    t = r_dims[1]
    ref_t = ref.reshape(t, 128)
    cand_t = cand.reshape(t, 128)
    diff_t = cand_t - ref_t
    step_rms = np.sqrt(np.mean(diff_t * diff_t, axis=1))
    step_corr = np.array([corrcoef(ref_t[i], cand_t[i]) for i in range(t)])

    worst = np.argsort(-step_rms)[: max(0, args.top)]
    print(f"per-step worst by rmsdiff (top {len(worst)}):")
    for i in worst:
        print(
            f"  step={int(i):02d} rmsdiff={step_rms[i]:.6f} "
            f"corr={step_corr[i]:.6f} ref_std={ref_t[i].std():.6f} cand_std={cand_t[i].std():.6f}"
        )

    print("per-step rmsdiff:")
    print("  " + " ".join(f"{v:.3f}" for v in step_rms))


if __name__ == "__main__":
    main()
