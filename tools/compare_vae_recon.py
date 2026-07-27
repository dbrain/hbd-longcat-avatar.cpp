#!/usr/bin/env python3
"""Read sd.cpp recon .bin outputs (ggml ne [W,H,C,T], values [0,1]), save PNGs,
and compute PSNR/MSE vs the torch oracle reconstruction + the input image."""
import struct
import numpy as np
from PIL import Image

OUT_DIR = "models/_testinputs"


def read_bin(path):
    with open(path, "rb") as f:
        n_dims = struct.unpack("<i", f.read(4))[0]
        name_len = struct.unpack("<i", f.read(4))[0]
        ttype = struct.unpack("<i", f.read(4))[0]
        ne = [struct.unpack("<i", f.read(4))[0] for _ in range(n_dims)]
        f.read(name_len)
        data = np.frombuffer(f.read(), dtype="<f4")
    assert ttype == 0
    return ne, data


def recon_to_img(path):
    ne, data = read_bin(path)  # ne = [W,H,C,T] ; memory: W fastest
    W, H, C, T = (ne + [1, 1, 1, 1])[:4]
    # reshape into [T,C,H,W] (C order: slowest after T). memory order is W,H,C,T.
    arr = data.reshape(T, C, H, W)  # ne[3]=T slowest .. ne[0]=W fastest
    img = arr[0].transpose(1, 2, 0)  # [H,W,C]
    return (np.clip(img, 0, 1) * 255.0).round().astype(np.uint8)


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf"), 0.0
    return 10.0 * np.log10(255.0 ** 2 / mse), mse


inp = np.asarray(Image.open(f"{OUT_DIR}/girl_480x832.png").convert("RGB"))
oracle = np.asarray(Image.open(f"{OUT_DIR}/oracle_recon.png").convert("RGB"))

for tag in ["sdcpp_recon_self", "sdcpp_recon_oracle"]:
    img = recon_to_img(f"{OUT_DIR}/{tag}.bin")
    Image.fromarray(img).save(f"{OUT_DIR}/{tag}.png")
    p_o, m_o = psnr(img, oracle)
    p_i, m_i = psnr(img, inp)
    print(f"{tag}: px mean={img.mean():.1f} std={img.std():.1f}  "
          f"PSNR vs torch-oracle={p_o:.2f}dB (MSE {m_o:.1f})  "
          f"PSNR vs input={p_i:.2f}dB (MSE {m_i:.1f})  -> saved {tag}.png")

# also report torch oracle vs input as the achievable ceiling
p_ci, m_ci = psnr(oracle, inp)
print(f"[ceiling] torch-oracle vs input: PSNR={p_ci:.2f}dB (MSE {m_ci:.1f})")
