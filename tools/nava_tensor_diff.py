#!/usr/bin/env python3
"""
Per-tensor diff between the PyTorch reference (ref_tensors.npz) and the C++ nava
harness outputs (a dir of <name>.bin raw float32, with a manifest.txt of shapes).
Reports max-abs-err, mean-abs-err, rel-err, and PSNR per key; flags the FIRST block
where divergence exceeds threshold (that localizes the bug to one block's math).

Usage:
  python3 tools/nava_tensor_diff.py ref_tensors.npz <cpp_out_dir> [--psnr 40]
cpp_out_dir holds <name>.bin (+ optional manifest.txt "<name> f32 d0 d1 ...").
Matches keys by name; shapes must match (after squeeze of size-1 dims).
"""
import sys, os, argparse
import numpy as np


def _norm_key(name):
    # cpp LONGCAT_DUMP_DIR taps are named "double_blocks.3" / "single_blocks.7";
    # the reference npz uses "double_block_3" / "single_block_7". Normalize.
    import re
    m = re.match(r"(double|single)_blocks\.(\d+)$", name)
    if m:
        return f"{m.group(1)}_block_{m.group(2)}"
    return name


def _read_longcat_dump(path):
    """LONGCAT_DUMP_DIR .bin: int64 ndim, ndim*int64 dims (ggml ne-order,
    ne[0] fastest), then f32 data. Returns array reshaped to numpy
    (reversed dims so ne[0] becomes the last/fastest axis)."""
    with open(path, "rb") as f:
        ndim = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        dims = np.fromfile(f, dtype=np.int64, count=ndim).tolist()  # ne-order
        data = np.fromfile(f, dtype=np.float32)
    np_shape = tuple(int(d) for d in reversed(dims))  # ne[0] fastest -> last axis
    try:
        data = data.reshape(np_shape)
    except Exception:
        pass
    return data


def load_bin_dir(d):
    out = {}
    man = os.path.join(d, "manifest.txt")
    shapes = {}
    if os.path.isfile(man):
        for line in open(man):
            p = line.split()
            if len(p) >= 3 and p[1] == "f32":
                shapes[p[0]] = tuple(int(x) for x in p[2:])
    for fn in os.listdir(d):
        if not fn.endswith(".bin"):
            continue
        name = _norm_key(fn[:-4])
        path = os.path.join(d, fn)
        # Detect the LONGCAT_DUMP_DIR header (int64 ndim in [1..5]); else raw.
        with open(path, "rb") as f:
            head = np.fromfile(f, dtype=np.int64, count=1)
        if head.size == 1 and 1 <= head[0] <= 5:
            a = _read_longcat_dump(path)
        else:
            a = np.fromfile(path, dtype=np.float32)
            if name in shapes:
                try:
                    a = a.reshape(shapes[name])
                except Exception:
                    pass
        out[name] = a
    return out


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    peak = max(np.abs(a).max(), np.abs(b).max(), 1e-8)
    return 20 * np.log10(peak) - 10 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_npz")
    ap.add_argument("cpp_dir")
    ap.add_argument("--psnr", type=float, default=40.0, help="pass threshold (dB)")
    args = ap.parse_args()

    ref = np.load(args.ref_npz, allow_pickle=True)
    cpp = load_bin_dir(args.cpp_dir)

    # order: inputs, then double_block_0..9, single_block_0..19, heads
    def sortkey(k):
        import re
        m = re.search(r"(double|single)_block_(\d+)", k)
        if m:
            return (1 if m.group(1) == "double" else 2, int(m.group(2)))
        if "head" in k:
            return (3, 0)
        return (0, 0)

    keys = sorted(set(ref.files) & set(cpp.keys()), key=sortkey)
    only_ref = set(ref.files) - set(cpp.keys())
    only_cpp = set(cpp.keys()) - set(ref.files)

    print(f"{'key':32s} {'shape':18s} {'maxAbs':>10s} {'meanAbs':>10s} {'PSNR(dB)':>9s}  status")
    print("-" * 90)
    first_fail = None
    for k in keys:
        a = np.asarray(ref[k]).astype(np.float64).squeeze()
        b = np.asarray(cpp[k]).astype(np.float64).squeeze()
        if a.shape != b.shape:
            print(f"{k:32s} SHAPE MISMATCH ref{a.shape} vs cpp{b.shape}")
            if first_fail is None:
                first_fail = k
            continue
        mx = np.abs(a - b).max()
        mn = np.abs(a - b).mean()
        p = psnr(a, b)
        ok = p >= args.psnr
        if not ok and first_fail is None:
            first_fail = k
        print(f"{k:32s} {str(a.shape):18s} {mx:10.4g} {mn:10.4g} {p:9.2f}  {'ok' if ok else 'FAIL'}")

    if only_ref:
        print(f"\n[only in ref, not emitted by cpp]: {sorted(only_ref)}")
    if only_cpp:
        print(f"[only in cpp]: {sorted(only_cpp)}")
    print()
    if first_fail:
        print(f">>> FIRST divergence at: {first_fail}  (bug is in/before this block's math)")
    else:
        print(f">>> ALL matched keys pass PSNR>={args.psnr} dB ✅")


if __name__ == "__main__":
    main()
