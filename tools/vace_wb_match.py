#!/usr/bin/env python3
"""L1 fix: per-channel white-balance match of seg2 frames to the seg1 tail, in the
OUTPUT/pixel domain (the VACE-Fun adapter drifts warm->neutral across the seam; the
conditioning-domain AGC can't fix it). Re-stitches seg1 + WB-matched seg2.

Usage: vace_wb_match.py <seg1_dir> <seg2_dir> <out_stitch_dir> [ramp_frames]
  ramp_frames>0: correction is full at the seam and eases to 0 over ramp_frames
                 (avoids over-warming the naturally-recovered tail). 0 = constant match.
Reference = seg1's LAST frame channel means. K=5 seg2 head frames overlap seg1 (dropped).
"""
import sys, os, glob, numpy as np
from PIL import Image

seg1_dir, seg2_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
ramp = int(sys.argv[4]) if len(sys.argv) > 4 else 0
K = 5
os.makedirs(out_dir, exist_ok=True)

seg1 = sorted(glob.glob(os.path.join(seg1_dir, "*.png")))
seg2 = sorted(glob.glob(os.path.join(seg2_dir, "f*.png")))
ref = np.asarray(Image.open(seg1[-1]).convert("RGB"), np.float64)
ref_mean = ref.reshape(-1, 3).mean(0)                       # target per-channel mean
print(f"ref (seg1 tail) channel means R,G,B = {ref_mean.round(2)}  R-G={ref_mean[0]-ref_mean[1]:.2f}")

i = 0
for f in seg1:                                              # seg1 verbatim
    Image.open(f).save(os.path.join(out_dir, f"s{i:03d}.png")); i += 1
for j, f in enumerate(seg2[K:]):                            # seg2 (drop K overlap), WB-matched
    im = np.asarray(Image.open(f).convert("RGB"), np.float64)
    m = im.reshape(-1, 3).mean(0)
    gain = np.where(m > 1e-3, ref_mean / m, 1.0)
    if ramp > 0:
        w = max(0.0, 1.0 - j / ramp)                        # full at seam, eases out
        gain = 1.0 + (gain - 1.0) * w
    out = np.clip(im * gain, 0, 255).astype(np.uint8)
    if j < 3 or j % 6 == 0:
        print(f"  seg2 f{j:02d} mean {m.round(1)} gain {gain.round(3)} -> R-G {(out.reshape(-1,3).mean(0)[0]-out.reshape(-1,3).mean(0)[1]):.2f}")
    Image.fromarray(out).save(os.path.join(out_dir, f"s{i:03d}.png")); i += 1
print(f"wrote {i} frames to {out_dir}")
