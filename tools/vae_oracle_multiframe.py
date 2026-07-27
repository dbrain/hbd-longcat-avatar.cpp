#!/usr/bin/env python3
"""Multi-frame torch oracle for the LongCat Wan VAE temporal-decode path.

Replicates the test PNG into a 25-frame video, encodes to a 7-latent-frame
tensor (4x temporal compression), decodes back, and dumps:
  - oracle_latent_mf.bin   : the raw 7-frame latent (ggml ne [w,h,T,C]) for sd.cpp decode
  - oracle_recon_mf.npy     : torch's 25-frame reconstruction (for comparison)
Also prints per-latent-frame stats so we can compare against sd.cpp.
"""
import struct
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image

VAE_DIR = "/mnt/hdd/longcat/base/vae"
PNG = "models/_testinputs/girl_480x832.png"
OUT_DIR = "models/_testinputs"
N_FRAMES = 25


def write_bin(path, ne, flat, name):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack("<i", len(name)))
        f.write(struct.pack("<i", 0))  # GGML_TYPE_F32
        for d in ne:
            f.write(struct.pack("<i", int(d)))
        f.write(name.encode())
        f.write(flat.astype("<f4").tobytes())
    print(f"wrote {path}  ne={ne}  numel={flat.size}")


torch.manual_seed(0)
vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
vae.eval()

img = Image.open(PNG).convert("RGB")
W, H = img.size
arr = np.asarray(img, dtype=np.float32) / 255.0
arr = arr * 2.0 - 1.0
# [1,3,T,H,W] -- replicate the portrait across T frames.
x1 = torch.from_numpy(arr).permute(2, 0, 1)[None, :, None]  # [1,3,1,H,W]
x = x1.repeat(1, 1, N_FRAMES, 1, 1)
print("x", tuple(x.shape))

with torch.no_grad():
    z = vae.encode(x).latent_dist.mode()  # [1,16,Tlat,h,w]
    print("latent z", tuple(z.shape), "mean", float(z.mean()), "std", float(z.std()))
    Tlat = z.shape[2]
    for t in range(Tlat):
        zt = z[0, :, t]
        print(f"  latent frame {t}: mean {float(zt.mean()):+.5f} std {float(zt.std()):.5f}")
    rec = vae.decode(z).sample  # [1,3,Tvid,H,W]
    print("rec", tuple(rec.shape), "mean", float(rec.mean()), "std", float(rec.std()))

# dump latent in ggml ne [w,h,T,C]; numpy z[0] is [C,Tlat,h,w] (w fastest C-contig).
z0 = z[0].cpu().numpy()  # [C,Tlat,h,w]
C, Tlat, h, w = z0.shape
flat = np.ascontiguousarray(z0).reshape(-1)
write_bin(f"{OUT_DIR}/oracle_latent_mf.bin", [w, h, Tlat, C], flat, "oracle_latent_mf")
# 5D variant [w,h,T,C,B=1] matching the production sampler latent layout.
write_bin(f"{OUT_DIR}/oracle_latent_mf5.bin", [w, h, Tlat, C, 1], flat, "oracle_latent_mf5")

# save torch recon for frame-0 PSNR comparison
rec0 = rec[0, :, 0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
rec0 = ((rec0 + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
Image.fromarray(rec0).save(f"{OUT_DIR}/oracle_recon_mf_f0.png")
np.save(f"{OUT_DIR}/oracle_recon_mf.npy", rec.cpu().numpy())
print(f"saved oracle_recon_mf_f0.png  px mean {rec0.mean():.1f} std {rec0.std():.1f}")
print(f"recon frames = {rec.shape[2]}")
