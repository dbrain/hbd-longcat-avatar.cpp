#!/usr/bin/env python3
"""Compare two rendered NAVA WAV clips for loudness and tone.

This is intentionally lightweight: it reads PCM WAV, resamples to 16 kHz with
linear interpolation for analysis, averages channels to mono, aligns by waveform
correlation, then reports RMS/peak, spectral centroid, and coarse band energy.
"""
import argparse
import wave

import numpy as np


BANDS = [(80, 300), (300, 800), (800, 1500), (1500, 3000), (3000, 5000), (5000, 7600)]


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        sw = w.getsampwidth()
        data = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {sw}")
    return sr, x.reshape(-1, ch)


def resample_linear(x, sr, target=16000):
    if sr == target:
        return x.copy()
    n = int(round(len(x) * target / sr))
    pos = np.arange(n, dtype=np.float64) * sr / target
    i = np.floor(pos).astype(np.int64)
    frac = pos - i
    i1 = np.minimum(i + 1, len(x) - 1)
    i = np.minimum(i, len(x) - 1)
    return x[i] * (1.0 - frac[:, None]) + x[i1] * frac[:, None]


def mono16(path):
    sr, x = read_wav(path)
    y = resample_linear(x, sr, 16000).mean(axis=1)
    return sr, x.shape, y


def corrcoef(a, b):
    aa = a - a.mean()
    bb = b - b.mean()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den else 0.0


def slices(a, b, lag):
    if lag < 0:
        aa = a[-lag:]
        bb = b[: len(aa)]
    elif lag > 0:
        bb = b[lag:]
        aa = a[: len(bb)]
    else:
        aa = a
        bb = b
    n = min(len(aa), len(bb))
    return aa[:n], bb[:n]


def best_lag(a, b, max_lag):
    best = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = slices(a, b, lag)
        c = corrcoef(aa, bb)
        if c > best[0]:
            best = (c, lag)
    return best


def stats(x, sr=16000):
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win)) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total = spec.sum()
    centroid = float((freqs * spec).sum() / total) if total else 0.0
    bands = []
    for lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        bands.append(float(np.sqrt(np.mean(spec[mask] ** 2))) if np.any(mask) else 0.0)
    return {
        "rms": float(np.sqrt(np.mean(x * x))),
        "peak": float(np.max(np.abs(x))) if len(x) else 0.0,
        "centroid": centroid,
        "bands": bands,
    }


def print_stats(label, st):
    band_str = " ".join(f"{lo}-{hi}={v:.8f}" for (lo, hi), v in zip(BANDS, st["bands"]))
    print(
        f"{label:>10s} rms={st['rms']:.6f} peak={st['peak']:.6f} "
        f"centroid={st['centroid']:.1f} {band_str}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--max-lag", type=int, default=3000, help="max alignment lag at 16 kHz samples")
    args = ap.parse_args()

    ref_sr, ref_shape, ref = mono16(args.reference)
    cand_sr, cand_shape, cand = mono16(args.candidate)
    corr, lag = best_lag(ref, cand, args.max_lag)
    ref_a, cand_a = slices(ref, cand, lag)

    print(f"reference: {args.reference} src_sr={ref_sr} shape={ref_shape} mono16={len(ref)}")
    print(f"candidate: {args.candidate} src_sr={cand_sr} shape={cand_shape} mono16={len(cand)}")
    print(f"alignment: corr={corr:.6f} lag={lag} samples @16k aligned_samples={len(ref_a)}")

    ref_st = stats(ref_a)
    cand_st = stats(cand_a)
    print_stats("reference", ref_st)
    print_stats("candidate", cand_st)

    print(
        f"ratios candidate/reference: rms={cand_st['rms'] / (ref_st['rms'] + 1e-12):.6f} "
        f"peak={cand_st['peak'] / (ref_st['peak'] + 1e-12):.6f} "
        f"centroid={cand_st['centroid'] / (ref_st['centroid'] + 1e-12):.6f}"
    )
    print(
        "band ratios candidate/reference: "
        + " ".join(f"{v / (r + 1e-12):.3f}" for r, v in zip(ref_st["bands"], cand_st["bands"]))
    )


if __name__ == "__main__":
    main()
