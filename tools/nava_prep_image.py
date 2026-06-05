#!/usr/bin/env python3
"""Preprocess an image into the cpp nava `--image` RGB bin: ggml ne [W,H,1,3], [0,1].
Usage: nava_prep_image.py <in_image> <W> <H> <out.bin> [--stretch]

Default matches Python NAVA's LocalVideoVAEAdapter image path
(nava_src/vae/local_video_vae.py:_resize_center_crop): aspect-PRESERVING LANCZOS
resize (scale = max so the image fills the target) followed by a center crop to
(W,H). The legacy stretch-to-(W,H) behavior (which distorts when the source
aspect != W/H, e.g. peter.png 1447x711 -> 896x448) is available via --stretch.
"""
import sys, numpy as np
try:
    from PIL import Image
    def load(p): return np.asarray(Image.open(p).convert("RGB"))
except Exception:
    import imageio.v2 as iio
    def load(p):
        a = iio.imread(p)
        return a[..., :3] if a.ndim == 3 else np.stack([a]*3, -1)

def resize_center_crop(img, W, H):
    """Aspect-preserving LANCZOS resize (fill) + center crop to (W,H).
    Mirrors nava_src/vae/local_video_vae.py:_resize_center_crop exactly."""
    from PIL import Image
    pil = Image.fromarray(img)
    src_w, src_h = pil.size                          # PIL size is (w,h)
    scale = max(W / src_w, H / src_h)
    resize_w = int(round(src_w * scale))
    resize_h = int(round(src_h * scale))
    pil = pil.resize((resize_w, resize_h), Image.LANCZOS)
    left = (resize_w - W) // 2
    top  = (resize_h - H) // 2
    pil = pil.crop((left, top, left + W, top + H))
    return np.asarray(pil)                            # [H,W,3]

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stretch = "--stretch" in sys.argv[1:]
    inp, W, H, out = args[0], int(args[1]), int(args[2]), args[3]
    img = load(inp)                                  # [h,w,3] uint8
    from PIL import Image
    if stretch:
        img = np.asarray(Image.fromarray(img).resize((W, H), Image.LANCZOS))  # legacy distort
    else:
        img = resize_center_crop(img, W, H)          # Python-parity [H,W,3]
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
