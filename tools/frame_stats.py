#!/usr/bin/env python3
import sys, glob, os
import numpy as np
from PIL import Image

def stats(d):
    paths = sorted(glob.glob(os.path.join(d, "*.png")),
                   key=lambda p: (len(p), p))
    if not paths:
        print(f"no PNGs in {d}"); return
    print(f"{d}: {len(paths)} frames")
    prev = None
    for i, p in enumerate(paths):
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
        r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
        bright = a.mean()
        d_ = "" if prev is None else f" d|Δ|={np.abs(a - prev).mean():5.1f}"
        gtint = g - (r + b) / 2
        print(f"  f{i:02d} {os.path.basename(p):20s} bright={bright:6.1f} "
              f"R={r:5.1f} G={g:5.1f} B={b:5.1f} Gtint={gtint:+5.1f}{d_}")
        prev = a

for d in sys.argv[1:]:
    stats(d)
