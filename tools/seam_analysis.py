#!/usr/bin/env python3
"""Seam-quality analysis for chained (continuation) avatar clips.

Usage: seam_analysis.py <clip.webm> <boundary_frame> [<boundary_frame> ...]

Reports the per-frame |delta| to the previous frame (mean abs pixel diff, 0-255)
across the whole clip, then compares the delta AT each segment boundary (the
stitch point) to the median within-segment delta. A clean seam has a boundary
delta close to the within-segment motion; a visible jump/flicker spikes it.

boundary_frame = the index of the FIRST frame of segment K+1 in the stitched
clip (i.e. the delta reported for that index is the seam delta).
"""
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


def main():
    clip = sys.argv[1]
    boundaries = set(int(x) for x in sys.argv[2:])
    with tempfile.TemporaryDirectory() as d:
        frames = extract(clip, d)
    n = len(frames)
    print(f"clip {clip}: {n} frames")
    deltas = [0.0]
    for i in range(1, n):
        deltas.append(float(np.mean(np.abs(frames[i] - frames[i - 1]))))

    within = [deltas[i] for i in range(1, n) if i not in boundaries]
    med = float(np.median(within)) if within else 0.0
    p90 = float(np.percentile(within, 90)) if within else 0.0
    mx = float(np.max(within)) if within else 0.0
    print(f"within-segment frame-to-frame |delta|: median={med:.3f} p90={p90:.3f} max={mx:.3f}")

    print("per-frame delta (* = segment boundary / stitch point):")
    for i in range(1, n):
        mark = "  <== SEAM" if i in boundaries else ""
        flag = ""
        if i in boundaries:
            ratio = deltas[i] / med if med > 0 else 0.0
            flag = f"  ratio-vs-median={ratio:.2f}x"
        if i in boundaries or deltas[i] > p90:
            print(f"  f{i:04d}: {deltas[i]:7.3f}{mark}{flag}")

    print("\nSEAM VERDICT:")
    for b in sorted(boundaries):
        if b < 1 or b >= n:
            print(f"  boundary {b}: out of range")
            continue
        ratio = deltas[b] / med if med > 0 else 0.0
        verdict = "SEAMLESS" if ratio < 1.5 else ("acceptable" if ratio < 3.0 else "VISIBLE JUMP")
        print(f"  seam at f{b}: delta={deltas[b]:.3f} vs within-median {med:.3f} -> {ratio:.2f}x  [{verdict}]")


if __name__ == "__main__":
    main()
