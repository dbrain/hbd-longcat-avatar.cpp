#!/usr/bin/env python3
"""Torch oracle for the LongCat Wan VAE round-trip. Encodes the test PNG to a
latent (mode, no sampling), decodes it back, saves reconstruction + dumps the
latent .npy and intermediate tensors for diffing against sd.cpp."""
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image

VAE_DIR = "/mnt/hdd/longcat/base/vae"
PNG = "models/_testinputs/girl_480x832.png"
OUT_DIR = "models/_testinputs"

torch.manual_seed(0)
vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
vae.eval()

img = Image.open(PNG).convert("RGB")
W, H = img.size
print(f"input image {W}x{H}")
arr = np.asarray(img, dtype=np.float32) / 255.0          # [H,W,3] in [0,1]
arr = arr * 2.0 - 1.0                                     # [-1,1]
# AutoencoderKLWan wants [B, C, T, H, W]
x = torch.from_numpy(arr).permute(2, 0, 1)[None, :, None]  # [1,3,1,H,W]
print("x", tuple(x.shape), float(x.mean()), float(x.std()))

with torch.no_grad():
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()                                  # deterministic latent (mean)
    print("latent z", tuple(z.shape), "mean", float(z.mean()), "std", float(z.std()))

    # Apply diffusers Wan latent normalization (what the DiT actually consumes).
    lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    z_norm = (z - lm) / ls
    print("z_norm", "mean", float(z_norm.mean()), "std", float(z_norm.std()))

    # Decode the RAW latent (no normalization) — this is what VAE.decode consumes.
    rec = vae.decode(z).sample                            # [1,3,1,H,W]
    print("rec", tuple(rec.shape), "mean", float(rec.mean()), "std", float(rec.std()))

# raw latent dump (unnormalized) — sd.cpp WanVAE.encode returns this `mu`
np.save(f"{OUT_DIR}/oracle_latent.npy", z[0].cpu().numpy())      # [16,1,h,w]
np.save(f"{OUT_DIR}/oracle_latent_norm.npy", z_norm[0].cpu().numpy())

rec_img = rec[0, :, 0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()  # [H,W,3]
rec_img = ((rec_img + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
Image.fromarray(rec_img).save(f"{OUT_DIR}/oracle_recon.png")
print(f"saved {OUT_DIR}/oracle_recon.png  recon px mean {rec_img.mean():.1f} std {rec_img.std():.1f}")
