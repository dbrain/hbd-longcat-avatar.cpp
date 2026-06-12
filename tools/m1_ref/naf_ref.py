#!/usr/bin/env python3
"""
NAF reference (torch functional + my own RoPE/na2d) validated vs the captured goldens
(naf_capture.py). Proves the exact op sequence to build in ggml. NOT the NAF nn.Module —
every op is spelled out explicitly so it maps 1:1 to the C++ port.

Validates:
  (1) ImageEncoder  -> naf_ie_out.npy   (2 conv encoders + GroupNorm/SiLU + adaptive_pool + axial RoPE)
  (2) na2d (shift)  -> naf_hr_golden    (using golden ie_out as q/k, isolates the attention)
  (3) FULL NAF      -> naf_hr_golden    (my ie_out -> na2d, end to end)

na2d uses the natten "shift" border rule (probe-confirmed @2.5e-6): per dilation phase the
9x9 window slides inward to stay in-bounds. For NAF (dil 16, k/v block-16 nearest-upsampled
from 32x32) this == per-output-pixel 9x9 over the SOURCE grid centered at clamp(i//16,4,27).

GPU run:
  /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python naf_ref.py
"""
import os, math
import numpy as np
import torch
import torch.nn.functional as F

# TRUE fp32 (no tf32) so this is the apples-to-apples oracle for the ggml CPU/fp32 port.
# The captured golden (naf_hr_golden) ran the NAF module on CUDA with tf32 ON (default),
# so it differs from true fp32 by ~1e-2 on the conv-heavy encoder — exactly like the
# bf16/tf32 goldens elsewhere. ggml (true fp32) must match THIS, not the tf32 golden.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "cpp_port", "refs")
WEIGHTS = os.path.join(HERE, "weights")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

NH_ATTN = 4
NH_ROPE = 4
DIM = 256
HEAD_QK = DIM // NH_ATTN     # 64
KS = 9
R = KS // 2                  # 4


def load():
    z = np.load(os.path.join(WEIGHTS, "naf.npz"))
    W = {k: torch.from_numpy(z[k].astype(np.float32)).to(DEV) for k in z.files}
    guide = torch.from_numpy(np.load(os.path.join(REFS, "naf_guide.npy"))).to(DEV)
    lr = torch.from_numpy(np.load(os.path.join(REFS, "naf_lr_features.npy"))).to(DEV)
    return W, guide, lr


def conv2d(x, W, key, pad):
    w = W[key + ".weight"]; b = W[key + ".bias"]
    if pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, w, b)


def gn(x, W, key):
    return F.group_norm(x, 8, W[key + ".weight"], W[key + ".bias"], eps=1e-5)


def encblock(x, W, p, pad):
    h = conv2d(F.silu(gn(x, W, p + ".norm1")), W, p + ".conv1", pad)
    h = conv2d(F.silu(gn(h, W, p + ".norm2")), W, p + ".conv2", pad)
    return h  # residual=False -> no skip


def enc(x, W, base, pad):
    h = conv2d(x, W, base + ".0", pad)        # initial conv
    h = encblock(h, W, base + ".1", pad)
    h = encblock(h, W, base + ".2", pad)
    return h


def rope(x, periods):
    # x [1,256,H,W]; heads_rope=4 -> d=64; periods[16]. axial, coords in [-1,1], eval (no rescale).
    B, C, H, Wd = x.shape
    d = C // NH_ROPE
    xr = x.reshape(B, NH_ROPE, d, H, Wd).permute(0, 1, 3, 4, 2).reshape(B, NH_ROPE, H * Wd, d)
    ch = torch.arange(0.5, H, device=DEV, dtype=torch.float32) / H
    cw = torch.arange(0.5, Wd, device=DEV, dtype=torch.float32) / Wd
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).reshape(-1, 2)  # [HW,2]
    coords = 2.0 * coords - 1.0
    angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # [HW,2,16]
    angles = angles.flatten(1, 2)                                       # [HW,32]
    angles = torch.cat([angles, angles], dim=-1)                        # tile(2) [HW,64]
    cos = torch.cos(angles); sin = torch.sin(angles)                    # [HW,64]
    x1, x2 = xr.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    xr = xr * cos + rot * sin
    out = xr.reshape(B, NH_ROPE, H, Wd, d).permute(0, 1, 4, 2, 3).reshape(B, C, H, Wd)
    return out


def image_encoder(W, guide):
    sp = enc(guide, W, "image_encoder.encoder", 0)       # k=1, pad=0
    se = enc(guide, W, "image_encoder.sem_encoder", 1)   # k=3, pad=1
    x = torch.cat([sp, se], dim=1)                        # [1,256,512,512]
    x = F.adaptive_avg_pool2d(x, (guide.shape[-2], guide.shape[-1]))  # identity @512
    x = rope(x, W["image_encoder.rope.periods"])
    return x


def na2d_naf(q, k_src, v_src):
    # q [1,256,H,W]; k_src [1,256,S,S]; v_src [1,1024,S,S]; shift-border na2d over source grid
    B, C, H, Wd = q.shape
    S = k_src.shape[-1]
    dv = v_src.shape[1] // NH_ATTN  # 256
    scale = HEAD_QK ** -0.5
    qh = q.reshape(B, NH_ATTN, HEAD_QK, H, Wd).permute(0, 3, 4, 1, 2)[0]   # [H,W,4,64]
    kh = k_src.reshape(B, NH_ATTN, HEAD_QK, S, S).permute(0, 3, 4, 1, 2)[0]  # [S,S,4,64]
    vh = v_src.reshape(B, NH_ATTN, dv, S, S).permute(0, 3, 4, 1, 2)[0]       # [S,S,4,256]
    up = H // S  # 16
    ci = torch.clamp(torch.arange(H, device=DEV) // up, R, S - 1 - R)        # [H]
    sr = ci[:, None] + (torch.arange(KS, device=DEV) - R)[None, :]           # [H,KS] rows
    scores = torch.empty(H, Wd, NH_ATTN, KS * KS, device=DEV)
    for a in range(KS):
        for b in range(KS):
            ki = kh[sr[:, a][:, None], sr[:, b][None, :]]   # [H,W,4,64]
            scores[:, :, :, a * KS + b] = (qh * ki).sum(-1) * scale
    w = scores.softmax(-1)
    out = torch.zeros(H, Wd, NH_ATTN, dv, device=DEV)
    for a in range(KS):
        for b in range(KS):
            vi = vh[sr[:, a][:, None], sr[:, b][None, :]]   # [H,W,4,256]
            out += w[:, :, :, a * KS + b:a * KS + b + 1] * vi
    out = out.permute(2, 3, 0, 1).reshape(1, NH_ATTN * dv, H, Wd)            # b (n d) h w
    return out


def cmp(tag, got, gold):
    got = got.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(got) else got
    d = np.abs(got - gold)
    a = got.reshape(-1); g = gold.reshape(-1)
    cos = float(a @ g / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-9))
    print(f"  [{tag}] maxabs={d.max():.3e} meanabs={d.mean():.3e} cosine={cos:.6f}")
    return d.max()


def main():
    torch.set_grad_enabled(False)
    W, guide, lr = load()
    ie_gold = np.load(os.path.join(REFS, "naf_ie_out.npy"))
    hr_gold = np.load(os.path.join(REFS, "naf_hr_golden.npy"))

    # (1) ImageEncoder
    print("[naf_ref] (1) ImageEncoder (true fp32) vs tf32 golden:")
    ie = image_encoder(W, guide)
    cmp("ie_out", ie, ie_gold)

    ie_gold_t = torch.from_numpy(ie_gold).to(DEV)

    # (2) na2d isolated (golden ie_out as q, golden-derived k/v)
    print("[naf_ref] (2) na2d (golden ie_out) vs hr golden:")
    k = F.adaptive_avg_pool2d(ie_gold_t, lr.shape[-2:])
    hr2 = na2d_naf(ie_gold_t, k, lr)
    cmp("hr(na2d,goldie)", hr2, hr_gold)

    # (3) FULL NAF (my ie_out, true fp32) vs tf32 golden
    print("[naf_ref] (3) FULL NAF (true fp32) vs tf32 golden:")
    k2 = F.adaptive_avg_pool2d(ie, lr.shape[-2:])
    hr3 = na2d_naf(ie, k2, lr)
    cmp("hr(full)", hr3, hr_gold)

    # save the TRUE-fp32 reference for the ggml port to validate against (tight)
    np.save(os.path.join(REFS, "naf_ie_fp32.npy"), ie.detach().cpu().numpy().astype(np.float32))
    np.save(os.path.join(REFS, "naf_hr_fp32.npy"), hr3.detach().cpu().numpy().astype(np.float32))
    print("[naf_ref] saved naf_ie_fp32.npy + naf_hr_fp32.npy (true-fp32 oracle for ggml).")
    print("[naf_ref] done.")


if __name__ == "__main__":
    main()
