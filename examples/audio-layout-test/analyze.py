#!/usr/bin/env python3
"""Decode clip.avi (lossless PCM16) and report per-channel dominant frequency.

The harness fed the writer a PLANAR buffer: left=440 Hz, right=880 Hz, constant
over the whole 2 s. A correct writer preserves that. The planar-as-interleaved
bug collapses both channels to the same content and doubles the pitch, with the
dominant frequency JUMPING at the midpoint (first half from the 440 region read
2x = ~880, second half from the 880 region read 2x = ~1760).

Verdict is computed, not eyeballed:
  PASS  -> L~440 & R~880, both constant across halves, channels decorrelated
  FAIL  -> channels collapsed (corr~1) and/or pitch doubled / jumps at midpoint
"""
import subprocess
import sys

import numpy as np


def decode(path, sr=48000):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    a = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    return a[:, 0], a[:, 1], sr


def dom_freq(x, sr):
    if len(x) < 16 or np.max(np.abs(x)) < 1e-4:
        return 0.0
    w = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(w))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(f[np.argmax(mag)])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "clip.avi"
    L, R, sr = decode(path)
    n = len(L)
    h = n // 2
    fL, fR = dom_freq(L, sr), dom_freq(R, sr)
    fL1, fL2 = dom_freq(L[:h], sr), dom_freq(L[h:], sr)
    fR1, fR2 = dom_freq(R[:h], sr), dom_freq(R[h:], sr)
    if np.std(L) > 1e-4 and np.std(R) > 1e-4:
        corr = float(np.corrcoef(L, R)[0, 1])
    else:
        corr = float("nan")

    print(f"file:        {path}  ({n} frames @ {sr} Hz, {n/sr:.2f}s)")
    print(f"L dominant:  whole={fL:6.0f}Hz  firsthalf={fL1:6.0f}Hz  secondhalf={fL2:6.0f}Hz")
    print(f"R dominant:  whole={fR:6.0f}Hz  firsthalf={fR1:6.0f}Hz  secondhalf={fR2:6.0f}Hz")
    print(f"L/R corr:    {corr:.3f}   (expect ~0 when stereo preserved, ~1 when collapsed)")

    def near(a, b, tol=40):
        return abs(a - b) <= tol

    stereo_ok = abs(corr) < 0.5 if corr == corr else False
    left_ok = near(fL1, 440) and near(fL2, 440)
    right_ok = near(fR1, 880) and near(fR2, 880)
    verdict = "PASS (channels intact: L=440 R=880, constant)" if (stereo_ok and left_ok and right_ok) \
        else "FAIL (planar buffer scrambled by interleaved read)"
    print(f"VERDICT:     {verdict}")
    return 0 if (stereo_ok and left_ok and right_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
