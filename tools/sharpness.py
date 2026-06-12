#!/usr/bin/env python3
"""Per-frame sharpness (variance of Laplacian) across one or more clip dirs. A multi-segment
chain that re-encodes pixels each seam loses high-freq detail -> declining sharpness over
windows; latent-direct carry should stay flatter. Pass dirs to compare side by side."""
import sys, glob, os
import numpy as np
from PIL import Image

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def conv2d(a, k):
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(a, k.shape)
    return (w * k).sum((-1, -2))


def sharp(d):
    paths = sorted(glob.glob(os.path.join(d, "*.png")), key=lambda p: (len(p), p))
    vals = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("L")).astype(np.float32)
        vals.append(float(conv2d(a, LAP).var()))
    return vals


def main():
    dirs = sys.argv[1:]
    series = {d: sharp(d) for d in dirs}
    n = max(len(v) for v in series.values())
    print("frame  " + "  ".join(f"{os.path.basename(d):>16}" for d in dirs))
    for i in range(n):
        row = "  ".join(f"{(series[d][i] if i < len(series[d]) else float('nan')):16.1f}" for d in dirs)
        print(f" {i:3d}   {row}")
    print("\nmean sharpness:")
    for d in dirs:
        v = np.array(series[d])
        # split into thirds (≈ windows) to show drift
        t = np.array_split(v, 3)
        print(f"  {os.path.basename(d):>18}: overall={v.mean():8.1f}  "
              f"thirds={t[0].mean():.0f} / {t[1].mean():.0f} / {t[2].mean():.0f}")


main()
