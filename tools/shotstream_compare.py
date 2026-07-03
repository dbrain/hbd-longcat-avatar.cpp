#!/usr/bin/env python3
# ruff: noqa
"""
Compare a C++ ShotStream dump against the reference goldens.

Usage:
    shotstream_compare.py <cpp_dump_dir> [golden_dir]

Loads every *.npy that exists in BOTH dirs (matched by basename) and prints
cosine similarity + max-abs-err + relative L2 per tensor, with pass/fail.
Robust to missing files (reports them, does not crash). Exit code:
    0 = all matched tensors pass, 1 = any fail, 2 = nothing to compare.

The C++ side should emit .npy (or .npz) files whose basenames match the golden
names (see MANIFEST.json in the golden dir), e.g. block0_selfattn_out.npy.
Complex golden tables (rope_freqs_chunk_*) are stored as [...,2] (real,imag).
"""
import os
import sys
import glob
import numpy as np

# per-tensor thresholds (cosine >= COS_MIN AND max_abs_err <= tolerance below).
COS_MIN = 0.9999
DEFAULT_ATOL = 2e-3          # absolute tolerance on max-abs-err (f32 kernels)
# tensors that are exact constants -> tighter tolerance
EXACT = {
    "warped_timesteps": 1e-3, "warped_sigmas": 1e-6,
    "scheduler_sigmas_grid": 1e-9, "scheduler_timesteps_grid": 1e-6,
    "rope_theta": 1e-12, "rope_freqs_dynamic_angle": 1e-9,
    "rope_freqs_chunk_shot0": 1e-9, "rope_freqs_chunk_shot1": 1e-9,
    "chunk_input_noise_latent": 1e-6,
}


def load_npy(path):
    a = np.load(path, allow_pickle=False)
    return np.asarray(a)


def stats(ref, got):
    ref = ref.astype(np.float64).ravel()
    got = got.astype(np.float64).ravel()
    n = min(ref.size, got.size)
    shape_ok = ref.size == got.size
    ref, got = ref[:n], got[:n]
    denom = (np.linalg.norm(ref) * np.linalg.norm(got))
    cos = float(np.dot(ref, got) / denom) if denom > 0 else (1.0 if np.allclose(ref, got) else 0.0)
    mae = float(np.max(np.abs(ref - got))) if n else float("nan")
    rl2 = float(np.linalg.norm(ref - got) / (np.linalg.norm(ref) + 1e-30))
    return cos, mae, rl2, shape_ok


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cpp_dir = sys.argv[1]
    gold_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "shotstream_goldens")

    golds = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(gold_dir, "*.npy"))}
    if not golds:
        print(f"No goldens in {gold_dir}")
        sys.exit(2)
    cpps = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(cpp_dir, "*.npy"))}

    print(f"golden dir : {gold_dir} ({len(golds)} tensors)")
    print(f"cpp dir    : {cpp_dir} ({len(cpps)} tensors)")
    print(f"thresholds : cosine >= {COS_MIN}, max_abs_err <= per-tensor tol\n")
    hdr = f"{'tensor':34s} {'cos':>10s} {'max_abs_err':>13s} {'rel_l2':>10s}  result"
    print(hdr); print("-" * len(hdr))

    npass = nfail = 0
    missing = []
    for name in sorted(golds):
        if name not in cpps:
            missing.append(name)
            print(f"{name:34s} {'—':>10s} {'—':>13s} {'—':>10s}  MISSING (cpp dump not found)")
            continue
        try:
            ref, got = load_npy(golds[name]), load_npy(cpps[name])
        except Exception as e:
            print(f"{name:34s}  ERROR loading: {e}")
            nfail += 1
            continue
        cos, mae, rl2, shape_ok = stats(ref, got)
        atol = EXACT.get(name, DEFAULT_ATOL)
        ok = shape_ok and cos >= COS_MIN and mae <= atol
        flag = "PASS" if ok else "FAIL"
        extra = "" if shape_ok else f"  [SHAPE {ref.shape} vs {got.shape}]"
        print(f"{name:34s} {cos:10.6f} {mae:13.3e} {rl2:10.3e}  {flag} (tol {atol:.0e}){extra}")
        npass += ok
        nfail += (not ok)

    print("-" * len(hdr))
    extra_cpp = sorted(set(cpps) - set(golds))
    if extra_cpp:
        print(f"cpp-only tensors (no golden): {', '.join(extra_cpp)}")
    print(f"\n{npass} pass, {nfail} fail, {len(missing)} missing "
          f"(of {len(golds)} goldens; {len(cpps)} cpp dumps present)")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
