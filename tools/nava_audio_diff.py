#!/usr/bin/env python3
"""Focused AUDIO-stream forward diff: python ref npz vs cpp LONGCAT_DUMP_DIR bins.
Isolates the audio token slice (L_vid:L_total) of the joint block outputs and the
audio velocity head, which the video-centric nava_tensor_diff.py blends away."""
import sys, os, numpy as np

NPZ = sys.argv[1]
CPPDIR = sys.argv[2]

def read_cpp(path):
    with open(path, "rb") as f:
        ndim = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        dims = np.fromfile(f, dtype=np.int64, count=ndim).tolist()
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(tuple(int(d) for d in reversed(dims)))  # ne0 fastest -> last

def metrics(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        return None
    mse = np.mean((a-b)**2)
    peak = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    psnr = float('inf') if mse==0 else 20*np.log10(peak)-10*np.log10(mse)
    cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
    return psnr, cos, float(np.abs(a-b).max())

ref = np.load(NPZ)
DIM = 3072
# L_aud from input_audio (min axis = 128 channels, other = L_aud)
ia = np.squeeze(ref["input_audio"])           # (L_aud,128) or (128,L_aud)
L_aud = int(min(ia.shape)) if 128 in ia.shape and ia.shape[0]!=ia.shape[1] else \
        (ia.shape[0] if ia.shape[1]==128 else ia.shape[1])
# L_total = the block's token axis (the one that isn't DIM and isn't 1)
_b0 = np.squeeze(ref["double_block_0"])       # (L_total, dim)
L_total = int([s for s in _b0.shape if s != DIM][0])
L_vid = L_total - L_aud
print(f"npz keys: {sorted(ref.keys())}")
print(f"derived: L_total={L_total} L_vid={L_vid} L_aud={L_aud} (dim={DIM})\n")

def py_block_audio(key):
    a = np.squeeze(ref[key])          # -> (L_total, dim)
    if a.ndim != 2: return None
    if a.shape[0] < a.shape[1]:       # (L_total, dim)
        return a[L_vid:L_vid+L_aud, :]
    return a[:, L_vid:L_vid+L_aud].T  # (dim, L_total) -> slice cols

def cpp_block_audio(name):
    p = os.path.join(CPPDIR, name+".bin")
    if not os.path.exists(p): return None
    a = np.squeeze(read_cpp(p))       # ne[dim,L_total] -> np (L_total, dim)
    if a.ndim != 2: return None
    if a.shape[0] < a.shape[1]:
        return a[L_vid:L_vid+L_aud, :]
    return a[:, L_vid:L_vid+L_aud].T

print("=== JOINT BLOCK OUTPUTS — AUDIO TOKEN SLICE (psnr dB / cosine / maxabs) ===")
for i in range(10):
    pa, ca = py_block_audio(f"double_block_{i}"), cpp_block_audio(f"double_blocks.{i}")
    if pa is None or ca is None:
        print(f"double_block_{i}: missing (py={pa is not None} cpp={ca is not None})"); continue
    m = metrics(pa, ca)
    print(f"double_block_{i:2d}  {('shape mismatch '+str(pa.shape)+' vs '+str(ca.shape)) if m is None else f'psnr={m[0]:7.2f}  cos={m[1]:.5f}  maxabs={m[2]:.4g}'}")
for i in range(20):
    pa, ca = py_block_audio(f"single_block_{i}"), cpp_block_audio(f"single_blocks.{i}")
    if pa is None or ca is None:
        print(f"single_block_{i}: missing"); continue
    m = metrics(pa, ca)
    print(f"single_block_{i:2d}  {('shape mismatch '+str(pa.shape)+' vs '+str(ca.shape)) if m is None else f'psnr={m[0]:7.2f}  cos={m[1]:.5f}  maxabs={m[2]:.4g}'}")

print("\n=== AUDIO VELOCITY (py vs cpp velocity_audio), both orientations ===")
_vk = "velocity_audio" if "velocity_audio" in ref else ("head_audio" if "head_audio" in ref else None)
pv = np.squeeze(ref[_vk]) if _vk else None
cv = read_cpp(os.path.join(CPPDIR, "velocity_audio.bin"))
cv = np.squeeze(cv)
print(f"py head_audio shape={None if pv is None else pv.shape}  cpp velocity_audio shape={cv.shape}")
if pv is not None:
    for label, pp in [("as-is", pv), ("transposed", pv.T)]:
        m = metrics(pp, cv)
        print(f"  {label:10s}: {'shape mismatch' if m is None else f'psnr={m[0]:7.2f}  cos={m[1]:.5f}  maxabs={m[2]:.4g}'}")
