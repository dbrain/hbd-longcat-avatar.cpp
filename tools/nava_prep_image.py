#!/usr/bin/env python3
"""Preprocess an image into the cpp nava `--image` RGB bin: ggml ne [W,H,1,3], [0,1].
Usage: nava_prep_image.py <in_image> <W> <H> <out.bin>"""
import sys, numpy as np
try:
    from PIL import Image
    def load(p): return np.asarray(Image.open(p).convert("RGB"))
except Exception:
    import imageio.v2 as iio
    def load(p):
        a = iio.imread(p)
        return a[..., :3] if a.ndim == 3 else np.stack([a]*3, -1)

def main():
    inp, W, H, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    img = load(inp)                                  # [h,w,3] uint8
    from PIL import Image
    img = np.asarray(Image.fromarray(img).resize((W, H), Image.LANCZOS))  # [H,W,3]
    a = (img.astype(np.float32) / 255.0)             # [H,W,3] in [0,1]
    a = np.transpose(a, (2, 0, 1))[:, None, :, :]    # [C=3, T=1, H, W] C-order == ggml ne [W,H,1,3]
    with open(out, "wb") as f:
        np.array([4], np.int32).tofile(f); np.array([5], np.int32).tofile(f)  # n_dims, name_len
        np.array([0], np.int32).tofile(f)                                     # type f32
        np.array([W, H, 1, 3], np.int32).tofile(f); f.write(b"image")
        a.astype(np.float32).ravel().tofile(f)
    print(f"wrote {out}  ne[{W},{H},1,3]  mean={a.mean():.4f} std={a.std():.4f}")

if __name__ == "__main__":
    main()
