#!/usr/bin/env python3
"""
GPU-free validation of the DINOv3 2D-axial RoPE (the one non-vanilla ViT detail) —
numpy port vs the verbatim torch logic from transformers DINOv3ViTRopePositionEmbedding
+ apply_rotary_pos_emb. No model weights needed (RoPE depends only on config + H,W).

Run: CUDA_VISIBLE_DEVICES="" <Pixal3D>/.venv/bin/python test_dinov3_rope.py
"""
import math, os, sys
import numpy as np
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch

# ---- config (camenduru/dinov3-vitl16) ----
HID, HEADS, PATCH, BASE = 1024, 16, 16, 100.0
HEAD_DIM = HID // HEADS  # 64


# ---- VERBATIM torch (from modeling_dinov3_vit.py) ----
def torch_rope(img_h, img_w):
    nph, npw = img_h // PATCH, img_w // PATCH
    inv_freq = 1 / BASE ** torch.arange(0, 1, 4 / HEAD_DIM, dtype=torch.float32)  # (head_dim/4,)
    coords_h = torch.arange(0.5, nph, dtype=torch.float32) / nph
    coords_w = torch.arange(0.5, npw, dtype=torch.float32) / npw
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0                                   # [P,2] (y,x) in [-1,1]
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]  # [P,2,hd/4]
    angles = angles.flatten(1, 2).tile(2)                         # [P,hd]
    return torch.cos(angles), torch.sin(angles)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def torch_apply(q, cos, sin):  # q [P,hd] patch tokens
    return q * cos + rotate_half(q) * sin


# ---- numpy port (candidate for C++) ----
def np_rope(img_h, img_w):
    nph, npw = img_h // PATCH, img_w // PATCH
    inv_freq = 1.0 / BASE ** np.arange(0, 1, 4 / HEAD_DIM, dtype=np.float32)
    ch = (np.arange(nph, dtype=np.float32) + 0.5) / nph
    cw = (np.arange(npw, dtype=np.float32) + 0.5) / npw
    yy, xx = np.meshgrid(ch, cw, indexing='ij')
    coords = np.stack([yy, xx], -1).reshape(-1, 2) * 2.0 - 1.0    # [P,2]
    angles = 2 * np.pi * coords[:, :, None] * inv_freq[None, None, :]  # [P,2,hd/4]
    angles = angles.reshape(coords.shape[0], -1)                  # flatten(1,2) -> [P,hd/2]
    angles = np.concatenate([angles, angles], -1)                # tile(2) -> [P,hd]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def np_apply(q, cos, sin):
    h = q.shape[-1] // 2
    rot = np.concatenate([-q[..., h:], q[..., :h]], -1)
    return q * cos + rot * sin


def main():
    print(f"[dinov3_rope] head_dim={HEAD_DIM} base={BASE}, cuda devices={torch.cuda.device_count()}")
    ok = True
    for (H, W) in [(512, 512), (1024, 1024), (224, 224)]:
        tc, ts = torch_rope(H, W)
        nc, ns = np_rope(H, W)
        dc = float(np.abs(tc.numpy() - nc).max()); ds = float(np.abs(ts.numpy() - ns).max())
        # apply on a random q
        P = tc.shape[0]
        g = torch.Generator().manual_seed(H)
        q = torch.randn(P, HEAD_DIM, generator=g)
        ta = torch_apply(q, tc, ts).numpy()
        na = np_apply(q.numpy(), nc, ns)
        da = float(np.abs(ta - na).max())
        cas = dc < 1e-6 and ds < 1e-6 and da < 1e-5
        ok &= cas
        print(f"  {H}x{W} P={P}: cos {dc:.2e} sin {ds:.2e} apply {da:.2e} {'PASS' if cas else 'FAIL'}")
    print("[dinov3_rope]", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
