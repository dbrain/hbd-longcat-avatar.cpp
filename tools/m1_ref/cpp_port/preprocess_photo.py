#!/usr/bin/env python3
# Raw-photo front-end for the pixal3d CLI / future upload-PNG API.
#   raw photo (any bg / aspect)  ->  rembg (BiRefNet RMBG-2.0) if no usable alpha  ->  crop to
#   subject bbox + square  ->  premultiply RGB on black  ->  matte PNG ready for `pixal3d --image`.
# Mirrors Pixal3D's Trellis2*.preprocess_image exactly (the model expects a centered subject on a
# black background — its training renders). The C++ CLI resizes this matte to 512/1024 internally.
#
#   /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python preprocess_photo.py in.png matte.png
# Then:  pixal3d --model weights_gguf_f16 --image matte.png --out model.glb --tex --fast
#
# Camera: pixal3d defaults to a sensible FOV (the miku cam); pass --fov to pixal3d for distorted
# shots (Python auto-estimates FOV via MoGe-2 — a separate host model, not ported; --fov is the
# documented bypass).  This host step is the API ingestion cut-line (keep the C++ on the ggml path).
import os, sys, argparse, numpy as np
from PIL import Image

PIXAL3D_ROOT = "/mnt/hdd/3d/avatar-shootout/Pixal3D"
sys.path.insert(0, PIXAL3D_ROOT)

def preprocess(inp: str, out: str, bg=(0, 0, 0)):
    img = Image.open(inp)
    has_alpha = False
    if img.mode == 'RGBA':
        a = np.array(img)[:, :, 3]
        if not np.all(a == 255):
            has_alpha = True
    # downscale huge inputs (matches preprocess_image: max side 1024 before rembg)
    max_size = max(img.size)
    if max_size > 1024:
        s = 1024 / max_size
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
    if has_alpha:
        rgba = img
    else:
        from pixal3d.pipelines.rembg import BiRefNet
        print("[prep] removing background (BiRefNet RMBG-2.0)...", flush=True)
        m = BiRefNet("briaai/RMBG-2.0"); m.cuda()
        rgba = m(img.convert('RGB'))
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0.8 * 255)
    if len(xs) == 0:
        raise SystemExit("[prep] no foreground found (alpha all ~0) — is the subject visible?")
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1)
    crop = (int(cx - size // 2), int(cy - size // 2), int(cx + size // 2), int(cy + size // 2))
    rgba = rgba.crop(crop)                                   # square, centered on subject
    f = np.array(rgba).astype(np.float32) / 255.0
    rgb = f[:, :, :3] * f[:, :, 3:4] + np.array(bg, np.float32) / 255.0 * (1 - f[:, :, 3:4])
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out)
    print(f"[prep] {inp} -> {out}  (matte {rgba.size[0]}x{rgba.size[1]}, subject {bbox})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Raw photo -> pixal3d matte (rembg + crop, on black).")
    ap.add_argument("input"); ap.add_argument("output")
    a = ap.parse_args()
    preprocess(a.input, a.output)
