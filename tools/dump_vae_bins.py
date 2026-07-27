#!/usr/bin/env python3
"""Write the test PNG (as a [0,1] RGB [W,H,T,C] tensor) and the torch oracle
latent into the sd.cpp .bin tensor format for the standalone VAE roundtrip test.

sd.cpp tensor file format (little-endian):
  i32 n_dims
  i32 name_len
  i32 ttype          (0 == GGML_TYPE_F32)
  i32 dims[n_dims]   (ggml ne order: shape[0] is innermost/fastest, i.e. W)
  char name[name_len]
  f32 data[...]      (contiguous, ne[0] fastest)
"""
import struct
import numpy as np
from PIL import Image

PNG = "models/_testinputs/girl_480x832.png"
OUT_DIR = "models/_testinputs"


def write_bin(path, arr_whc_order, name):
    # arr already laid out so that axis 0 varies fastest in memory (ggml ne[0]).
    # We pass a python list of ne dims and a flat f32 buffer (ne[0]-fastest).
    ne, flat = arr_whc_order
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack("<i", len(name)))
        f.write(struct.pack("<i", 0))  # GGML_TYPE_F32
        for d in ne:
            f.write(struct.pack("<i", int(d)))
        f.write(name.encode())
        f.write(flat.astype("<f4").tobytes())
    print(f"wrote {path}  ne={ne}  numel={flat.size}")


# --- input image: VAE wants ggml shape [W,H,T,C], values in [0,1] ---
img = Image.open(PNG).convert("RGB")
W, H = img.size
arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
# Build a buffer in ggml ne order [W,H,T,C]: ne[0]=W fastest, then H, then T=1, then C=3.
# numpy: index as [c, t, h, w] (C-contiguous => w fastest). That gives ne[0]=w.
buf = np.transpose(arr, (2, 0, 1))[:, None, :, :]  # [C, T=1, H, W]
flat = np.ascontiguousarray(buf).reshape(-1)        # w fastest, then h, t, c
write_bin(f"{OUT_DIR}/vae_input.bin", ([W, H, 1, 3], flat), "vae_input")

# --- oracle latent: ggml shape [w,h,T,C] = [60,104,1,16] ---
z = np.load(f"{OUT_DIR}/oracle_latent.npy")  # [16,1,104,60] (C,T,h,w)
C, T, h, w = z.shape
flatz = np.ascontiguousarray(z).reshape(-1)  # w fastest, then h, t, c
write_bin(f"{OUT_DIR}/oracle_latent.bin", ([w, h, T, C], flatz), "oracle_latent")
print("latent torch stats: mean", float(z.mean()), "std", float(z.std()))
