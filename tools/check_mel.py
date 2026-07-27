#!/usr/bin/env python3
"""Validate the C++ WhisperMel algorithm against HF WhisperFeatureExtractor.

Re-implements the C++ mel math in numpy (same constants/order) and compares to
transformers.WhisperFeatureExtractor on the test wav. Reports max abs diff.
"""
import sys

import numpy as np
from transformers import WhisperFeatureExtractor

SR, NFFT, HOP, NMELS = 16000, 400, 160, 128


def load_wav(path):
    import wave
    w = wave.open(path, "rb")
    n = w.getnframes()
    raw = w.readframes(n)
    ch = w.getnchannels()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    return a


def hz_to_mel(f):
    f_sp = 200.0 / 3.0
    mels = f / f_sp
    min_lhz = 1000.0
    min_lmel = min_lhz / f_sp
    logstep = np.log(6.4) / 27.0
    if f >= min_lhz:
        mels = min_lmel + np.log(f / min_lhz) / logstep
    return mels


def mel_to_hz(mel):
    f_sp = 200.0 / 3.0
    freq = f_sp * mel
    min_lhz = 1000.0
    min_lmel = min_lhz / f_sp
    logstep = np.log(6.4) / 27.0
    if mel >= min_lmel:
        freq = min_lhz * np.exp(logstep * (mel - min_lmel))
    return freq


def mel_filters():
    n_freqs = NFFT // 2 + 1
    fftfreqs = np.arange(n_freqs) * SR / NFFT
    mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(SR / 2.0)
    mel_pts = np.array([mel_to_hz(mel_min + (mel_max - mel_min) * i / (NMELS + 1)) for i in range(NMELS + 2)])
    fdiff = np.diff(mel_pts)
    fb = np.zeros((NMELS, n_freqs))
    for m in range(NMELS):
        for k in range(n_freqs):
            lower = (fftfreqs[k] - mel_pts[m]) / fdiff[m]
            upper = (mel_pts[m + 2] - fftfreqs[k]) / fdiff[m + 1]
            val = max(0.0, min(lower, upper))
            enorm = 2.0 / (mel_pts[m + 2] - mel_pts[m])
            fb[m, k] = val * enorm
    return fb


def cpp_logmel(wav):
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)
    fb = mel_filters()
    pad = NFFT // 2
    padded = np.concatenate([wav[1:pad + 1][::-1], wav, wav[-pad - 1:-1][::-1]])
    n_frames = 1 + (len(padded) - NFFT) // HOP - 1
    n_freqs = NFFT // 2 + 1
    out = np.zeros((NMELS, n_frames))
    for f in range(n_frames):
        frame = padded[f * HOP: f * HOP + NFFT] * hann
        spec = np.fft.rfft(frame, n=NFFT)
        power = (spec.real ** 2 + spec.imag ** 2)
        out[:, f] = fb @ power
    out = np.log10(np.maximum(out, 1e-10))
    out = np.maximum(out, out.max() - 8.0)
    out = (out + 4.0) / 4.0
    return out


def main():
    wav = load_wav(sys.argv[1])
    mine = cpp_logmel(wav)
    fe = WhisperFeatureExtractor(feature_size=128)
    hf = fe(wav, sampling_rate=SR, return_tensors="np").input_features[0]  # [128, 3000]
    # HF pads/truncates to 3000 frames; compare the overlap
    n = min(mine.shape[1], hf.shape[1])
    d = np.abs(mine[:, :n] - hf[:, :n])
    print(f"mine {mine.shape}  hf {hf.shape}  compared {n} frames")
    print(f"max abs diff: {d.max():.5f}   mean abs diff: {d.mean():.6f}")
    print(f"mine[:3,0]={mine[:3,0]}  hf[:3,0]={hf[:3,0]}")


if __name__ == "__main__":
    main()
