#!/usr/bin/env python3
# Production neural soft-matte for the native image_to_rig preprocessing lane.
#
# WHY THIS EXISTS
#   Pixal3D's Python preprocess_image (pixal3d/pipelines/pixal3d_image_to_3d.py) removes the
#   background with ZhengPeng7/BiRefNet (the public MIT original that briaai/RMBG-2.0 is remapped
#   to) and composites the subject over black.  The native C++ front-end has no BiRefNet; its
#   geometric border-flood matte picks a *different* silhouette, and the chaotically sensitive
#   SS-DiT amplifies that into a stage-1 coord collapse (63.6% vs Python).  The kobbler matting
#   service (docker/matting) runs a DIFFERENT set of weights (RMBG-2.0 GGUF), which only reaches
#   ~91.7%.  Matching Python's stage-1 within noise requires Python's *exact* BiRefNet weights.
#
#   This helper produces exactly what native's has_alpha crop path (image_io.hpp
#   pixal_preprocess_black_matte) expects: an RGBA whose RGB is already composited over black
#   (rgb*alpha) and whose alpha is BiRefNet's soft mask.  native then crops by alpha>0.8 and
#   Lanczos-resizes to 512/1024 using its PIL-parity resampler.  Feeding this (WITHOUT
#   --image-model-ready) reproduces Python's preprocess_image up to the crop, which native owns.
#
#   The output is byte-identical in alpha to Pixal3D's rembg, and RGB matches Python's
#   rgb*a + black*(1-a) composite (crop-then-composite == composite-then-crop for a black bg).
#
# Usage:
#   birefnet_matte.py <input.(png|jpg)> <out_rgba.png> [--device cuda|cpu]
#
# Runs on the reserved RTX 3060 when --device cuda (the caller holds the 3060 flock).
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# The exact model Pixal3D uses (briaai/RMBG-2.0 -> ZhengPeng7/BiRefNet remap; MIT).
MODEL_NAME = "ZhengPeng7/BiRefNet"

_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def birefnet_alpha(rgb_img: Image.Image, device: str) -> Image.Image:
    """Return an L-mode soft alpha for rgb_img, exactly as Pixal3D's rembg BiRefNet does."""
    model = AutoModelForImageSegmentation.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval().to(device)
    if device == "cuda":
        # BiRefNet ships fp32; keep fp32 for parity with the Python reference.
        pass
    x = _TRANSFORM(rgb_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[-1].sigmoid().cpu()[0].squeeze()
    return transforms.ToPILImage()(pred).resize(rgb_img.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()

    img = Image.open(a.inp)
    # Pixal3D preprocess_image: if the input already carries a real (non-opaque) alpha, use it
    # directly; otherwise scale to max 1024 (LANCZOS) and run BiRefNet.
    has_alpha = False
    if img.mode == "RGBA":
        al = np.array(img)[:, :, 3]
        if not np.all(al == 255):
            has_alpha = True
    max_size = max(img.size)
    scale = min(1.0, 1024.0 / max_size)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    if has_alpha:
        rgba = img.convert("RGBA")
        alpha = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0
        rgb = np.array(rgba.convert("RGB")).astype(np.float32)
    else:
        rgb_img = img.convert("RGB")
        mask = birefnet_alpha(rgb_img, a.device)
        alpha = np.array(mask).astype(np.float32) / 255.0
        rgb = np.array(rgb_img).astype(np.float32)

    # Composite over black (Python's rgb*a + bg*(1-a), bg=0) and keep the soft alpha for the
    # native alpha-bbox crop.  This is native's has_alpha contract input.
    comp = np.clip(rgb * alpha[:, :, None], 0, 255).astype(np.uint8)
    out = np.dstack([comp, np.clip(alpha * 255.0, 0, 255).astype(np.uint8)])
    Image.fromarray(out, "RGBA").save(a.out)
    print(f"[birefnet_matte] {a.inp} -> {a.out} ({out.shape[1]}x{out.shape[0]} RGBA, "
          f"model={MODEL_NAME}, device={a.device})")


if __name__ == "__main__":
    main()
