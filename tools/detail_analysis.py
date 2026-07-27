#!/usr/bin/env python3
"""Perceptual DETAIL analysis for chained (continuation) avatar clips.

The watercolour/garble accumulation owner reports is a DETAIL-LOSS that luma /
seam-delta metrics MISS. This tool measures per-frame sharpness so the
softening shows up as a SHARPNESS-DECAY curve segment-over-segment.

Metrics (per frame):
  - vlap   : variance of the Laplacian (the standard "is it blurry" sharpness
             measure; lower = softer/blurrier). Computed on luma.
  - hfen   : high-frequency energy fraction = (mean |grad|) normalised, a second
             independent sharpness witness (edge density).
  - tenen  : Tenengrad (mean Sobel gradient magnitude squared) — a third witness.

Usage:
  detail_analysis.py <clip.webm> [--boundaries f1 f2 ...] [--seg-len N]
                     [--plot out.png] [--csv out.csv] [--ref-seg0]

  --boundaries  frame indices that start a new stitched segment (the seams).
  --seg-len     if given, auto-derive boundaries every seg-len frames.
  --plot        write a sharpness-curve PNG (vlap + hfen across the clip,
                seam markers, per-segment means + a linear decay-rate fit).
  --csv         dump per-frame metrics.
  --ref-seg0    also report each segment's mean sharpness as a FRACTION of
                segment 0's mean (the accumulation ratio; <1 = detail dying).

A watercolour-accumulating clip shows vlap/hfen DECAYING segment-over-segment
(seg1 mean < seg0 mean < ...). A clean chain stays roughly FLAT.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def extract(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         os.path.join(outdir, "f%04d.png")],
        check=True,
    )
    files = sorted(os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith(".png"))
    return [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) for f in files]


def luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def laplacian_var(y):
    # 4-neighbour discrete Laplacian, variance over interior pixels.
    lap = (-4.0 * y[1:-1, 1:-1]
           + y[:-2, 1:-1] + y[2:, 1:-1]
           + y[1:-1, :-2] + y[1:-1, 2:])
    return float(lap.var())


def hf_energy(y):
    # mean absolute first-difference gradient (edge density), in 0..255 luma.
    gx = np.abs(np.diff(y, axis=1))
    gy = np.abs(np.diff(y, axis=0))
    return float((gx.mean() + gy.mean()) * 0.5)


def tenengrad(y):
    # Sobel gradient magnitude squared, mean.
    sx = (y[:-2, 2:] + 2 * y[1:-1, 2:] + y[2:, 2:]
          - y[:-2, :-2] - 2 * y[1:-1, :-2] - y[2:, :-2])
    sy = (y[2:, :-2] + 2 * y[2:, 1:-1] + y[2:, 2:]
          - y[:-2, :-2] - 2 * y[:-2, 1:-1] - y[:-2, 2:])
    return float((sx * sx + sy * sy).mean())


def segments_from(n, boundaries):
    """Return list of (start, end) frame ranges given seam boundaries."""
    bs = sorted(set(b for b in boundaries if 0 < b < n))
    segs = []
    start = 0
    for b in bs:
        segs.append((start, b))
        start = b
    segs.append((start, n))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--boundaries", type=int, nargs="*", default=[])
    ap.add_argument("--seg-len", type=int, default=0)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ref-seg0", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        frames = extract(args.clip, d)
    n = len(frames)
    if n == 0:
        print("no frames")
        sys.exit(1)

    ys = [luma(f) for f in frames]
    vlap = np.array([laplacian_var(y) for y in ys])
    hfen = np.array([hf_energy(y) for y in ys])
    tene = np.array([tenengrad(y) for y in ys])

    boundaries = list(args.boundaries)
    if args.seg_len > 0:
        boundaries = list(range(args.seg_len, n, args.seg_len))
    segs = segments_from(n, boundaries)

    print(f"clip {args.clip}: {n} frames, {len(segs)} segment(s) "
          f"(boundaries {sorted(set(b for b in boundaries if 0 < b < n))})")
    print(f"{'metric':<8} {'whole-mean':>11} {'whole-std':>10}")
    for name, arr in (("vlap", vlap), ("hfen", hfen), ("tenen", tene)):
        print(f"{name:<8} {arr.mean():>11.3f} {arr.std():>10.3f}")

    print("\nPER-SEGMENT sharpness means (the decay witness):")
    print(f"{'seg':>4} {'frames':>9} {'vlap':>10} {'hfen':>8} {'tenen':>10}"
          + ("  vlap/seg0  hfen/seg0" if args.ref_seg0 else ""))
    seg_vlap, seg_hfen = [], []
    for i, (s, e) in enumerate(segs):
        v = float(vlap[s:e].mean())
        h = float(hfen[s:e].mean())
        t = float(tene[s:e].mean())
        seg_vlap.append(v)
        seg_hfen.append(h)
        extra = ""
        if args.ref_seg0:
            extra = f"  {v / seg_vlap[0]:>9.3f}  {h / seg_hfen[0]:>8.3f}"
        print(f"{i:>4} {f'{s}-{e-1}':>9} {v:>10.3f} {h:>8.3f} {t:>10.3f}{extra}")

    # decay verdict: linear fit of per-segment vlap vs segment index
    if len(seg_vlap) >= 2:
        idx = np.arange(len(seg_vlap))
        slope_v = float(np.polyfit(idx, seg_vlap, 1)[0])
        slope_h = float(np.polyfit(idx, seg_hfen, 1)[0])
        pct_v = 100.0 * slope_v / seg_vlap[0] if seg_vlap[0] else 0.0
        pct_h = 100.0 * slope_h / seg_hfen[0] if seg_hfen[0] else 0.0
        last_v = 100.0 * seg_vlap[-1] / seg_vlap[0] if seg_vlap[0] else 0.0
        last_h = 100.0 * seg_hfen[-1] / seg_hfen[0] if seg_hfen[0] else 0.0
        print("\nDECAY VERDICT (per-segment linear fit):")
        print(f"  vlap slope = {slope_v:+.3f}/seg ({pct_v:+.1f}%/seg of seg0); "
              f"last seg = {last_v:.1f}% of seg0")
        print(f"  hfen slope = {slope_h:+.3f}/seg ({pct_h:+.1f}%/seg of seg0); "
              f"last seg = {last_h:.1f}% of seg0")
        flat = abs(pct_v) < 5.0 and abs(pct_h) < 5.0
        print(f"  -> {'FLAT (detail retained)' if flat else 'DECAYING (detail loss accumulating)'}")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("frame,vlap,hfen,tenen\n")
            for i in range(n):
                fh.write(f"{i},{vlap[i]:.4f},{hfen[i]:.4f},{tene[i]:.4f}\n")
        print(f"\nwrote {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        x = np.arange(n)
        ax1.plot(x, vlap, color="#2277dd", lw=1.2, label="vlap (Laplacian var)")
        ax1.set_ylabel("vlap (sharpness)")
        ax1.set_title(f"detail / sharpness across chained clip — {os.path.basename(args.clip)}")
        ax2.plot(x, hfen, color="#dd5522", lw=1.2, label="hfen (edge density)")
        ax2.set_ylabel("hfen (edge density)")
        ax2.set_xlabel("frame")
        for ax, segmeans in ((ax1, seg_vlap), (ax2, seg_hfen)):
            for b in sorted(set(b for b in boundaries if 0 < b < n)):
                ax.axvline(b, color="#888", ls="--", lw=0.8)
            for (s, e), m in zip(segs, segmeans):
                ax.hlines(m, s, e - 1, color="#11aa44", lw=2.0, alpha=0.8)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
