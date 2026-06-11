#!/usr/bin/env python3
"""
NumPy fp32 CPU reference of the Pixal3D (TRELLIS.2) Sparse-Structure VAE decoder.

Mirrors pixal3d/models/sparse_structure_vae.py::SparseStructureDecoder.forward
exactly (config ss_dec_conv3d_16l8_fp16.json):
    out_channels=1, latent_channels=8, num_res_blocks=2, num_res_blocks_middle=2,
    channels=[512,128,32], norm_type="layer" (ChannelLayerNorm32).

Everything is fp32 (the Python torso runs fp16; goldens carry fp16 noise so the
ss_logits match is tolerance-only, but the occupancy coords match exactly).

No torch in the implementation.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W_NPZ = os.path.join(HERE, "weights", "ss_dec.npz")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
Z_S = os.path.join(GOLD, "stage1_ssdec", "z_s.npy")
SS_LOGITS = os.path.join(GOLD, "stage1_ssdec", "ss_logits.npy")
COORDS = os.path.join(GOLD, "stage1_out", "coords.npy")

_W = {k: np.asarray(v, dtype=np.float32) for k, v in np.load(W_NPZ).items()}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def silu(x):
    return x / (1.0 + np.exp(-x))


def conv3d(x, w, b, pad=1):
    """
    x: [B, Cin, D, H, W]  (we use (D,H,W) generically as the 3 spatial dims)
    w: [Cout, Cin, kd, kh, kw]  (torch correlation layout)
    b: [Cout]
    Direct fp32 correlation with zero padding `pad`, stride 1. im2col.
    """
    B, Cin, D, H, Wd = x.shape
    Cout, Cin_, kd, kh, kw = w.shape
    assert Cin == Cin_
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad), (pad, pad)))
    Do, Ho, Wo = D, H, Wd  # stride 1, padding 'same' for k odd
    # Build im2col patches: [B, Cin*kd*kh*kw, Do*Ho*Wo]
    cols = np.empty((B, Cin * kd * kh * kw, Do * Ho * Wo), dtype=np.float32)
    idx = 0
    for cd in range(kd):
        for ch in range(kh):
            for cw in range(kw):
                patch = xp[:, :, cd:cd + Do, ch:ch + Ho, cw:cw + Wo]  # [B,Cin,Do,Ho,Wo]
                cols[:, idx * Cin:(idx + 1) * Cin, :] = patch.reshape(B, Cin, -1)
                idx += 1
    # weight reorder to match col ordering: cols block order is (cd,ch,cw) outer, Cin inner
    w_re = np.transpose(w, (0, 2, 3, 4, 1)).reshape(Cout, kd * kh * kw * Cin)
    out = np.einsum("ok,bkn->bon", w_re, cols).astype(np.float32)
    out = out + b[None, :, None]
    return out.reshape(B, Cout, Do, Ho, Wo)


def channel_layer_norm(x, weight, bias, eps=1e-5):
    """
    ChannelLayerNorm32: LayerNorm over the channel dim (dim=1).
    Implementation: permute channels to last, LayerNorm over last axis, permute back.
    x: [B, C, D, H, W]
    """
    # move channel to last: [B, D, H, W, C]
    xt = np.transpose(x, (0, 2, 3, 4, 1)).astype(np.float32)
    mean = xt.mean(axis=-1, keepdims=True)
    var = xt.var(axis=-1, keepdims=True)  # biased (population) variance, like torch LayerNorm
    xn = (xt - mean) / np.sqrt(var + eps)
    xn = xn * weight[None, None, None, None, :] + bias[None, None, None, None, :]
    return np.transpose(xn, (0, 4, 1, 2, 3)).astype(np.float32)


def pixel_shuffle_3d(x, s=2):
    """
    Mirror pixal3d.modules.spatial.pixel_shuffle_3d.
    x: [B, C, H, W, D]; C_ = C // s^3
    reshape (B, C_, s, s, s, H, W, D); permute(0,1,5,2,6,3,7,4);
    reshape (B, C_, H*s, W*s, D*s)
    """
    B, C, H, W, D = x.shape
    C_ = C // (s ** 3)
    x = x.reshape(B, C_, s, s, s, H, W, D)
    x = np.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
    x = x.reshape(B, C_, H * s, W * s, D * s)
    return np.ascontiguousarray(x)


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------
def res_block(x, prefix):
    w = _W
    h = channel_layer_norm(x, w[f"{prefix}.norm1.weight"], w[f"{prefix}.norm1.bias"])
    h = silu(h)
    h = conv3d(h, w[f"{prefix}.conv1.weight"], w[f"{prefix}.conv1.bias"], pad=1)
    h = channel_layer_norm(h, w[f"{prefix}.norm2.weight"], w[f"{prefix}.norm2.bias"])
    h = silu(h)
    h = conv3d(h, w[f"{prefix}.conv2.weight"], w[f"{prefix}.conv2.bias"], pad=1)
    # skip_connection is Identity (in==out for every stage in this config)
    return h + x


def upsample_block(x, prefix):
    w = _W
    x = conv3d(x, w[f"{prefix}.conv.weight"], w[f"{prefix}.conv.bias"], pad=1)
    return pixel_shuffle_3d(x, 2)


# ---------------------------------------------------------------------------
# decoder forward
# ---------------------------------------------------------------------------
def decode(z_s):
    w = _W
    z_s = np.asarray(z_s, dtype=np.float32)
    h = conv3d(z_s, w["input_layer.weight"], w["input_layer.bias"], pad=1)

    # middle_block: 2 ResBlocks @512
    h = res_block(h, "middle_block.0")
    h = res_block(h, "middle_block.1")

    # blocks flat list:
    #   0,1  ResBlock(512)
    #   2    Upsample(512->128)
    #   3,4  ResBlock(128)
    #   5    Upsample(128->32)
    #   6,7  ResBlock(32)
    h = res_block(h, "blocks.0")
    h = res_block(h, "blocks.1")
    h = upsample_block(h, "blocks.2")
    h = res_block(h, "blocks.3")
    h = res_block(h, "blocks.4")
    h = upsample_block(h, "blocks.5")
    h = res_block(h, "blocks.6")
    h = res_block(h, "blocks.7")

    # out_layer: ChannelLayerNorm32, SiLU, Conv3d(32->1)
    h = channel_layer_norm(h, w["out_layer.0.weight"], w["out_layer.0.bias"])
    h = silu(h)
    h = conv3d(h, w["out_layer.2.weight"], w["out_layer.2.bias"], pad=1)
    return h.astype(np.float32)


def to_coords(ss_logits):
    """
    Occupancy post-process (sample_sparse_structure):
        decoded = ss_logits > 0                         # [1,1,64,64,64]
        ratio = 64//32 = 2 -> maxpool3d(k2,s2) > 0.5    # [1,1,32,32,32]
        coords = argwhere(decoded)[:, [0,2,3,4]]        # (b, x,y,z)
    """
    decoded = ss_logits > 0  # bool [1,1,64,64,64]
    B, C, X, Y, Z = decoded.shape
    # maxpool k2 s2 pad0 over bool->float == any-True in each 2x2x2 block
    f = decoded.astype(np.float32)
    f = f.reshape(B, C, X // 2, 2, Y // 2, 2, Z // 2, 2)
    pooled = f.max(axis=(3, 5, 7))  # [B,C,32,32,32]
    pooled = pooled > 0.5
    idx = np.argwhere(pooled)  # C-order, columns (dim0..dim4)=(b,c,x,y,z)
    coords = idx[:, [0, 2, 3, 4]].astype(np.int32)
    return coords


# ---------------------------------------------------------------------------
def _rows_set(a):
    return set(map(tuple, a.tolist()))


if __name__ == "__main__":
    z_s = np.load(Z_S)
    gold_logits = np.load(SS_LOGITS)
    gold_coords = np.load(COORDS)

    print(f"z_s {z_s.shape} {z_s.dtype}")
    logits = decode(z_s)
    print(f"ss_logits out {logits.shape} {logits.dtype}")

    err = np.abs(logits - gold_logits)
    maxabs = err.max()
    p999 = np.percentile(err, 99.9)
    med = np.median(err)
    # relative error (guard small denom)
    denom = np.maximum(np.abs(gold_logits), 1e-3)
    maxrel = (err / denom).max()
    # sign agreement near threshold (within |logit|<1.0 of zero in gold)
    sign_agree = (np.sign(logits) == np.sign(gold_logits)).mean()
    near = np.abs(gold_logits) < 1.0
    sign_agree_near = (np.sign(logits[near]) == np.sign(gold_logits[near])).mean()
    print(f"  maxabs   = {maxabs:.4e}")
    print(f"  median   = {med:.4e}")
    print(f"  p99.9    = {p999:.4e}")
    print(f"  maxrel   = {maxrel:.4e}")
    print(f"  sign-agree all      = {sign_agree*100:.4f}%")
    print(f"  sign-agree near-0   = {sign_agree_near*100:.4f}% (|gold|<1.0)")

    coords = to_coords(logits)
    gset = _rows_set(gold_coords)
    cset = _rows_set(coords)
    print(f"coords mine N={coords.shape[0]}  gold N={gold_coords.shape[0]}")
    same_n = coords.shape[0] == gold_coords.shape[0]
    set_eq = cset == gset
    print(f"  same N      = {same_n}")
    print(f"  set-equal   = {set_eq}")
    if not set_eq:
        miss = gset - cset
        extra = cset - gset
        print(f"  missing (in gold, not mine): {len(miss)}")
        print(f"  extra   (in mine, not gold): {len(extra)}")
        if miss:
            print("   sample missing:", list(miss)[:8])
        if extra:
            print("   sample extra:  ", list(extra)[:8])

    ok = same_n and set_eq
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
