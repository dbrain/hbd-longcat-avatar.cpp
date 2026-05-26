#!/usr/bin/env python3
"""Per-frame exposure / chroma drift analysis for chained avatar clips.

Usage:
  exposure_analysis.py <clip.webm> --new-per-seg N --cond-decoded C [--n-seg S]

Reports per-frame RGB mean (luma proxy + chroma) across the stitched clip and,
for each segment, the mean RGB of its COND/overlap region (the re-rendered tail
of the prior segment, held fixed via the denoise mask) vs its GENERATED region
(the fresh 8-step-DMD frames). The cond->gen brightness STEP inside each segment
is the residual the seam fix could not kill with a uniform offset.

Stitched layout (segment 0 keeps all seg_frames; segments 1+ drop their leading
cond_decoded re-render, contributing new_per_seg frames each):
  seg0: [0 .. seg_frames-1]                      (cond region = none for seg0)
  segK: new_per_seg frames; the FIRST cond_decoded of segK's own render were
        dropped, so in the STITCHED clip the seam frame is the first generated
        frame of segK. We approximate the within-segment cond vs gen split by
        re-deriving from the segment render, but in the stitched clip the cond
        frames of seg>0 are dropped — so here we report the per-segment-boundary
        STEP (last frame of segK vs first frame of seg K+1) and the global
        per-frame RGB trend (which exposes the compounding drift directly).
"""
import argparse
import os
import subprocess
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
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--seg-frames", type=int, default=33)
    ap.add_argument("--new-per-seg", type=int, default=20)
    ap.add_argument("--cond-decoded", type=int, default=13)
    ap.add_argument("--n-seg", type=int, default=3)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        frames = extract(args.clip, d)
    n = len(frames)
    print(f"clip {args.clip}: {n} frames")

    # global per-frame RGB mean
    rgb = np.array([f.reshape(-1, 3).mean(axis=0) for f in frames])  # [n,3]
    luma = rgb.mean(axis=1)

    # segment boundaries in the stitched clip
    bounds = [0]
    for k in range(1, args.n_seg):
        bounds.append(args.seg_frames + (k - 1) * args.new_per_seg)
    bounds.append(n)
    print(f"segment boundaries (stitched first-frame indices): {bounds[:-1]}")

    print("\nper-segment mean RGB + luma (full segment) and its first/last frame:")
    for k in range(args.n_seg):
        s, e = bounds[k], bounds[k + 1]
        seg_rgb = rgb[s:e].mean(axis=0)
        print(f"  seg{k} [{s:3d}..{e-1:3d}]  meanRGB=({seg_rgb[0]:6.2f},{seg_rgb[1]:6.2f},{seg_rgb[2]:6.2f}) luma={seg_rgb.mean():6.2f}"
              f"   first f{s} RGB=({rgb[s][0]:.1f},{rgb[s][1]:.1f},{rgb[s][2]:.1f})"
              f"   last f{e-1} RGB=({rgb[e-1][0]:.1f},{rgb[e-1][1]:.1f},{rgb[e-1][2]:.1f})")

    print("\nseam STEP (last frame of segK vs first frame of seg K+1):")
    for k in range(1, args.n_seg):
        b = bounds[k]
        d_rgb = rgb[b] - rgb[b - 1]
        print(f"  seam{k} @f{b}: ΔRGB=({d_rgb[0]:+.2f},{d_rgb[1]:+.2f},{d_rgb[2]:+.2f}) Δluma={d_rgb.mean():+.2f}")

    print("\nglobal luma trend (first 3, last 3 of each segment):")
    for k in range(args.n_seg):
        s, e = bounds[k], bounds[k + 1]
        head = ",".join(f"{luma[i]:.1f}" for i in range(s, min(s + 3, e)))
        tail = ",".join(f"{luma[i]:.1f}" for i in range(max(e - 3, s), e))
        print(f"  seg{k}: head[{head}] ... tail[{tail}]")


if __name__ == "__main__":
    main()
