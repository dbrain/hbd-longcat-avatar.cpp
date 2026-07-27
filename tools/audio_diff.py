#!/usr/bin/env python3
"""Compare two avatar clips: per-frame lower-face-region diff (audio vs no-audio).

Usage: audio_diff.py <noaudio.webm> <audio.webm>
Extracts frames via ffmpeg, computes mean-abs pixel diff over the whole frame and
over the lower-face region (bottom-center, where the mouth lives in a portrait),
and reports per-frame temporal variation so we can see whether audio drives motion
in the mouth area over time.
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
    na_path, a_path = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as td:
        na = extract(na_path, os.path.join(td, "na"))
        a = extract(a_path, os.path.join(td, "a"))
    n = min(len(na), len(a))
    print(f"frames: no-audio={len(na)} audio={len(a)} compared={n}")
    if n == 0:
        return
    H, W, _ = na[0].shape
    # lower-face region: a portrait headshot has the face in the upper-middle;
    # the mouth sits roughly at 55-75% height, center 30-70% width.
    y0, y1 = int(0.55 * H), int(0.78 * H)
    x0, x1 = int(0.30 * W), int(0.70 * W)
    print(f"frame {W}x{H}; lower-face ROI y[{y0}:{y1}] x[{x0}:{x1}]")

    full_diffs, roi_diffs = [], []
    # temporal: how much each clip's ROI changes frame-to-frame (motion proxy)
    na_motion, a_motion = [], []
    for i in range(n):
        fd = np.abs(na[i] - a[i]).mean()
        rd = np.abs(na[i][y0:y1, x0:x1] - a[i][y0:y1, x0:x1]).mean()
        full_diffs.append(fd)
        roi_diffs.append(rd)
        if i > 0:
            na_motion.append(np.abs(na[i][y0:y1, x0:x1] - na[i - 1][y0:y1, x0:x1]).mean())
            a_motion.append(np.abs(a[i][y0:y1, x0:x1] - a[i - 1][y0:y1, x0:x1]).mean())

    print("\nper-frame |audio - noaudio| (0-255 scale):")
    for i in range(n):
        print(f"  f{i:02d}: full={full_diffs[i]:7.3f}  lower-face={roi_diffs[i]:7.3f}")
    print(f"\nMEAN full-frame diff:   {np.mean(full_diffs):.4f}")
    print(f"MEAN lower-face diff:   {np.mean(roi_diffs):.4f}")
    print(f"MAX  lower-face diff:   {np.max(roi_diffs):.4f}  (frame {int(np.argmax(roi_diffs))})")
    if na_motion:
        print(f"\nROI temporal motion (mean frame-to-frame |Δ| in lower face):")
        print(f"  no-audio: {np.mean(na_motion):.4f}   audio: {np.mean(a_motion):.4f}")
        print(f"  ratio audio/no-audio: {np.mean(a_motion) / max(1e-6, np.mean(na_motion)):.3f}")


if __name__ == "__main__":
    main()
