#!/usr/bin/env python3
"""Build single-frame NAVA continuation anchors that preserve motion direction.

NAVA's trained i2v input is one clean frame, so feeding multiple clean frames is
out-of-distribution. This tool stays in-distribution: it converts the last few
decoded frames of a prior segment into one plausible "next instant" image, then
that image can be passed to `nava render --image`.

Usage:
  tools/nava_motion_anchor.py --out-dir anchors tail/f001.png tail/f002.png tail/f003.png
  nava render ... --image anchors/linear.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path, arr):
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "RGB")
    img.save(path)


def smooth_delta(delta, radius):
    if radius <= 0:
        return delta
    chans = []
    for c in range(delta.shape[2]):
        plane = Image.fromarray(np.clip((delta[:, :, c] + 1.0) * 127.5, 0, 255).astype(np.uint8), "L")
        plane = plane.filter(ImageFilter.GaussianBlur(radius=radius))
        chans.append(np.asarray(plane, dtype=np.float32) / 127.5 - 1.0)
    return np.stack(chans, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+", help="tail frames in temporal order")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gain", type=float, default=0.85, help="linear velocity extrapolation gain")
    ap.add_argument("--accel", type=float, default=0.35, help="second-order acceleration gain")
    ap.add_argument("--blur-radius", type=float, default=1.2, help="smooth flow proxy before applying it")
    ap.add_argument("--smear", type=float, default=0.35, help="blend extrapolated image with last frame")
    args = ap.parse_args()

    if len(args.frames) < 2:
        raise SystemExit("need at least two frames")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_rgb(p) for p in args.frames]
    shape = frames[-1].shape
    if any(f.shape != shape for f in frames):
        raise SystemExit("all frames must have the same dimensions")

    prev = frames[-2]
    last = frames[-1]
    delta = smooth_delta(last - prev, args.blur_radius)

    linear = last + args.gain * delta
    save_rgb(out_dir / "linear.png", linear)

    if len(frames) >= 3:
        prev2 = frames[-3]
        delta_prev = smooth_delta(prev - prev2, args.blur_radius)
        accel = delta - delta_prev
        second = last + args.gain * delta + args.accel * accel
        save_rgb(out_dir / "accel.png", second)
    else:
        second = linear
        save_rgb(out_dir / "accel.png", second)

    smear = (1.0 - args.smear) * linear + args.smear * last
    save_rgb(out_dir / "smear.png", smear)

    save_rgb(out_dir / "last.png", last)
    print(f"wrote {out_dir}/last.png linear.png accel.png smear.png")


if __name__ == "__main__":
    main()
