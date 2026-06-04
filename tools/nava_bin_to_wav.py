#!/usr/bin/env python3
"""Convert a NAVA waveform .bin (nava_audio_vae_decode_ref.py format: int32
[n_dims,name_len,type], int32 dims (ne order), name, f32 payload Fortran-flattened
ne0-fastest) to a WAV. ne = [n_samples, n_channels]. Usage: <in.bin> <out.wav> [sr]"""
import sys, numpy as np, wave

inp, outp = sys.argv[1], sys.argv[2]
sr = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
with open(inp, "rb") as f:
    n_dims, name_len, t = np.fromfile(f, np.int32, 3)
    dims = [int(x) for x in np.fromfile(f, np.int32, int(n_dims))]
    name = f.read(int(name_len))
    data = np.fromfile(f, np.float32)
x = np.reshape(data, dims, order="F")        # ne order -> [n_samples, n_channels]
x = np.squeeze(x)
if x.ndim == 1:
    x = x[:, None]
n_samp, n_ch = x.shape
x = np.clip(x, -1.0, 1.0)
pcm = (x * 32767.0).astype(np.int16)         # [n_samp, n_ch] C-flatten = interleaved
with wave.open(outp, "wb") as w:
    w.setnchannels(n_ch); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(pcm.tobytes())
print(f"wrote {outp}  {n_samp} samp x {n_ch} ch @ {sr}Hz  ({n_samp/sr:.2f}s)")
