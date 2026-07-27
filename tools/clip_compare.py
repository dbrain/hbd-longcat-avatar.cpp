#!/usr/bin/env python3
"""Compare two avatar clips frame-by-frame and report coherence.

Usage: clip_compare.py <ref.webm> <test.webm>

Extracts frames via ffmpeg and reports, per frame:
  - PSNR(ref, test) so a perf lever that changes the *decode device* (CPU vs GPU
    VAE) or kernel can be checked for visual equivalence vs a known-good clip;
  - ac_lag16 = long-range spatial autocorrelation of the test frame (natural
    images ~0.8+, white noise ~0). Used as the coherence gate: frame 0 should
    track the portrait; generated frames must stay structured, not collapse to
    noise.

If only one clip is given, just report ac_lag16 per frame (coherence gate).
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


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


def ac_lag(frame, lag=16):
    g = frame.mean(axis=2)
    g = g - g.mean()
    denom = float(np.sum(g * g))
    if denom <= 1e-9:
        return 0.0
    a = g[:, :-lag]
    b = g[:, lag:]
    return float(np.sum(a * b) / denom)


def main():
    if len(sys.argv) < 2:
        print("usage: clip_compare.py <ref.webm> [test.webm]")
        sys.exit(1)
    with tempfile.TemporaryDirectory() as td:
        ref = extract(sys.argv[1], os.path.join(td, "ref"))
        test = None
        if len(sys.argv) >= 3:
            test = extract(sys.argv[2], os.path.join(td, "test"))
        n = len(ref) if test is None else min(len(ref), len(test))
        print(f"frames: ref={len(ref)}" + (f" test={len(test)}" if test else ""))
        psnrs = []
        for i in range(n):
            acr = ac_lag(ref[i])
            if test is not None:
                p = psnr(ref[i], test[i])
                psnrs.append(p)
                act = ac_lag(test[i])
                print(f"  f{i:02d}: PSNR(ref,test)={p:6.2f}dB  ac16 ref={acr:.3f} test={act:.3f}")
            else:
                print(f"  f{i:02d}: ac16={acr:.3f}")
        if psnrs:
            print(f"mean PSNR={np.mean(psnrs):.2f}dB min={np.min(psnrs):.2f}dB")


if __name__ == "__main__":
    main()
