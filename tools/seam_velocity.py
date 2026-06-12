#!/usr/bin/env python3
"""Seam-velocity test for segment continuation. Per-frame motion magnitude (mean |Δ|)
and horizontal motion-centroid across a stitched 2-segment clip. If velocity is
preserved, motion magnitude + centroid drift are CONTINUOUS across the seam; a reset
shows a freeze (dip) or jump at the seam frame."""
import sys, glob, os
import numpy as np
from PIL import Image


def main():
    d = sys.argv[1]
    seam = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    imgs = [np.asarray(Image.open(p).convert("L")).astype(np.float32) for p in paths]
    H, W = imgs[0].shape
    xs = np.arange(W)
    mot, cx = [], []
    for a, b in zip(imgs[:-1], imgs[1:]):
        dmap = np.abs(b - a)
        m = dmap.mean()
        col = dmap.sum(0)
        c = (col * xs).sum() / (col.sum() + 1e-8) / W  # 0..1 horizontal motion centroid
        mot.append(m); cx.append(c)
    mot, cx = np.array(mot), np.array(cx)
    print(f"{d}: {len(imgs)} frames, seam at stitched frame {seam}")
    print("  trans:  motion  cx     bar")
    for i, (m, c) in enumerate(zip(mot, cx)):
        mark = "  <== SEAM" if seam >= 0 and i == seam - 1 else ""
        bar = "#" * int(m / (mot.max() + 1e-8) * 30)
        print(f"   {i:3d}->{i+1:<3d} {m:6.2f}  {c:.2f}  {bar}{mark}")
    if seam > 2 and seam < len(mot) - 2:
        within = np.concatenate([mot[max(0, seam - 5):seam - 1], mot[seam + 1:seam + 5]])
        at = mot[seam - 1:seam + 1].mean()
        print(f"\n  motion within-segment mean={within.mean():.2f} std={within.std():.2f}; "
              f"AT seam={at:.2f}  (ratio {at/(within.mean()+1e-8):.2f}x)")
        print(f"  centroid step at seam = {abs(cx[seam]-cx[seam-1]):.3f} "
              f"vs typical {np.abs(np.diff(cx)).mean():.3f}")
        print("  (ratio ~1 + small centroid step = velocity continuous; "
              "dip/spike or big centroid jump = reset)")


main()
