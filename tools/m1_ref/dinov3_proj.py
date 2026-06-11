#!/usr/bin/env python3
"""
NUMPY (fp32, CPU) reference for the Pixal3D Stage-1 image conditioner:
  DINOv3 ViT-L/16 (camenduru/dinov3-vitl16) + view-aligned ProjGrid.

  image -> (z_global [1,5,1024], z_proj [1,4096,1024])

Mirrors `DinoV3ProjFeatureExtractor.extract_features` + `forward`
(pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:344-566) and the
verbatim HF DINOv3 forward (transformers/models/dinov3_vit/modeling_dinov3_vit.py).

The IMPL is pure numpy. torch/PIL are used ONLY for image load+LANCZOS resize (to
match the PIL Image.LANCZOS the extractor uses) and an OPTIONAL torch cross-check.

Key correctness details (all load-bearing, verified against the source):
  - RoPE 2D-axial applied to PATCH TOKENS ONLY (first 5 = cls+reg excluded).
  - k_proj has NO bias; q/v/o have bias.
  - LayerScale: per-channel mul by learned lambda1 (init 1.0, USE the weights).
  - Final norm = PLAIN unweighted F.layer_norm(h,[1024]) (eps 1e-5), NOT model.norm.
  - patch embed flatten(2).transpose -> token p = row*32+col; spatial reshape [B,32,32,1024].
  - gelu = EXACT erf variant (HF ACT2FN["gelu"] == GELUActivation), NOT tanh.
  - attention scaling = head_dim^-0.5 = 64^-0.5; softmax in fp32.
  - ImageNet normalize mean[.485,.456,.406] std[.229,.224,.225] after /255.

Self-validating __main__ compares z_global and z_proj vs the captured goldens.
"""
import json
import os
import sys
import math
import numpy as np

# reuse the ALREADY-VALIDATED proj grid_sample + camera unproject
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proj_cond"))
from proj_cond_ref import proj_grid_forward  # noqa: E402

# ---- config (camenduru/dinov3-vitl16, VERIFIED) ----
HID = 1024
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = HID // N_HEADS  # 64
PATCH = 16
N_REG = 4
INTERMEDIATE = 4096
ROPE_THETA = 100.0
LN_EPS = 1e-5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NPZ = os.path.join(_HERE, "weights", "dinov3.npz")


# =============================================================================
# numpy primitives
# =============================================================================
def layer_norm(x, weight=None, bias=None, eps=LN_EPS):
    """LayerNorm over last dim. weight/bias=None -> plain (unweighted) norm."""
    x = x.astype(np.float32)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased (matches torch)
    xn = (x - mu) / np.sqrt(var + eps)
    if weight is not None:
        xn = xn * weight
    if bias is not None:
        xn = xn + bias
    return xn.astype(np.float32)


def linear(x, weight, bias=None):
    """y = x @ W.T + b. weight stored [out,in] (torch nn.Linear)."""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y.astype(np.float32)


def gelu_erf(x):
    """Exact erf GELU == HF GELUActivation == torch nn.GELU() default."""
    x = x.astype(np.float64)
    out = x * 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))
    return out.astype(np.float32)


def erf_vec(x):
    # math.erf is scalar; use the vectorized numpy-friendly Abramowitz/Stegun?
    # No -- need fp accuracy. Use scipy if present, else a high-accuracy poly.
    try:
        from scipy.special import erf  # type: ignore
        return erf(x)
    except Exception:
        # vectorize math.erf (accurate to fp64); slow but only used in MLP act.
        return np.vectorize(math.erf, otypes=[np.float64])(x)


def softmax(x, axis=-1):
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


# =============================================================================
# RoPE (validated math from test_dinov3_rope.py)
# =============================================================================
def rope_cos_sin(img_h, img_w):
    nph, npw = img_h // PATCH, img_w // PATCH
    inv_freq = 1.0 / ROPE_THETA ** np.arange(0, 1, 4.0 / HEAD_DIM, dtype=np.float32)  # [16]
    ch = (np.arange(nph, dtype=np.float32) + 0.5) / nph
    cw = (np.arange(npw, dtype=np.float32) + 0.5) / npw
    yy, xx = np.meshgrid(ch, cw, indexing="ij")
    coords = np.stack([yy, xx], -1).reshape(-1, 2) * 2.0 - 1.0  # [P,2]
    angles = 2 * np.pi * coords[:, :, None] * inv_freq[None, None, :]  # [P,2,16]
    angles = angles.reshape(coords.shape[0], -1)  # flatten(1,2) -> [P,32]
    angles = np.concatenate([angles, angles], -1)  # tile(2) -> [P,64]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def apply_rope_patches(t, cos, sin, n_prefix):
    """t [H, P+prefix, head_dim]; apply RoPE to patch tokens only."""
    prefix = t[:, :n_prefix, :]
    patches = t[:, n_prefix:, :]
    # cos/sin [Ppatch, head_dim] -> broadcast over head axis
    patches = patches * cos[None] + rotate_half(patches) * sin[None]
    return np.concatenate([prefix, patches], axis=1)


# =============================================================================
# weights
# =============================================================================
class Weights:
    def __init__(self, npz_path=WEIGHTS_NPZ):
        z = np.load(npz_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files}

    def __getitem__(self, k):
        return self.w[k]


# =============================================================================
# forward
# =============================================================================
def embeddings(image_chw, W):
    """image_chw [3,H,W] (normalized). Returns [1, 5+P, 1024]."""
    C, H, Wd = image_chw.shape
    nph, npw = H // PATCH, Wd // PATCH
    pw = W["embeddings.patch_embeddings.weight"]  # [1024,3,16,16]
    pb = W["embeddings.patch_embeddings.bias"]    # [1024]
    # Conv2d k16 s16 == unfold non-overlapping patches @ matmul.
    # patch (r,c): image_chw[:, r*16:(r+1)*16, c*16:(c+1)*16] -> flatten [3*16*16]
    # token order = flatten(2).transpose -> p = r*npw + c
    patches = image_chw.reshape(C, nph, PATCH, npw, PATCH)  # [3,nph,16,npw,16]
    patches = patches.transpose(1, 3, 0, 2, 4)              # [nph,npw,3,16,16]
    patches = patches.reshape(nph * npw, C * PATCH * PATCH)  # [P, 3*16*16]
    kern = pw.reshape(HID, C * PATCH * PATCH)               # [1024, 3*16*16]
    emb = patches @ kern.T + pb                             # [P,1024]
    emb = emb[None].astype(np.float32)                      # [1,P,1024]
    cls = W["embeddings.cls_token"]                         # [1,1,1024]
    reg = W["embeddings.register_tokens"]                   # [1,4,1024]
    return np.concatenate([cls, reg, emb], axis=1)          # [1,5+P,1024]


def attention(x, W, li, cos, sin, n_prefix):
    """x [1,S,1024]. Standard MHA with patch-only RoPE."""
    S = x.shape[1]
    p = f"layer.{li}.attention."
    q = linear(x[0], W[p + "q_proj.weight"], W[p + "q_proj.bias"])  # [S,1024]
    k = linear(x[0], W[p + "k_proj.weight"], None)                  # NO bias
    v = linear(x[0], W[p + "v_proj.weight"], W[p + "v_proj.bias"])
    # [S,1024] -> [heads,S,head_dim]
    q = q.reshape(S, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = k.reshape(S, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = v.reshape(S, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    q = apply_rope_patches(q, cos, sin, n_prefix)
    k = apply_rope_patches(k, cos, sin, n_prefix)
    scaling = HEAD_DIM ** -0.5
    scores = (q @ k.transpose(0, 2, 1)) * scaling  # [heads,S,S]
    attn = softmax(scores, axis=-1)
    out = attn @ v                                  # [heads,S,head_dim]
    out = out.transpose(1, 0, 2).reshape(S, HID)    # [S,1024]
    out = linear(out, W[p + "o_proj.weight"], W[p + "o_proj.bias"])
    return out[None]


def mlp(x, W, li):
    p = f"layer.{li}.mlp."
    h = linear(x, W[p + "up_proj.weight"], W[p + "up_proj.bias"])
    h = gelu_erf(h)
    h = linear(h, W[p + "down_proj.weight"], W[p + "down_proj.bias"])
    return h


def block(x, W, li, cos, sin, n_prefix):
    p = f"layer.{li}."
    # attn sub-block
    h = layer_norm(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
    h = attention(h, W, li, cos, sin, n_prefix)
    h = h * W[p + "layer_scale1.lambda1"]
    x = x + h
    # mlp sub-block
    h = layer_norm(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
    h = mlp(h, W, li)
    h = h * W[p + "layer_scale2.lambda1"]
    x = x + h
    return x.astype(np.float32)


def extract_features(image_chw, W):
    """Returns plain-normed hidden states [1, 5+P, 1024]."""
    H, Wd = image_chw.shape[1], image_chw.shape[2]
    x = embeddings(image_chw, W)
    cos, sin = rope_cos_sin(H, Wd)
    n_prefix = 1 + N_REG
    for li in range(N_LAYERS):
        x = block(x, W, li, cos, sin, n_prefix)
    # final PLAIN unweighted layer_norm (NOT model.norm)
    x = layer_norm(x, None, None, eps=LN_EPS)
    return x


# =============================================================================
# preprocess (torch/PIL allowed for load+resize only)
# =============================================================================
def load_preprocess(image_path, image_size=512):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD        # ImageNet normalize
    return arr.transpose(2, 0, 1).astype(np.float32)  # [3,H,W]


# =============================================================================
# top-level extract
# =============================================================================
def extract(image_path, cam, image_size=512, grid_resolution=16, weights=None):
    """image -> (z_global [1,5,1024], z_proj [1,R^3,1024], z_patchmap [1,h,w,1024])."""
    W = weights if weights is not None else Weights()
    image_chw = load_preprocess(image_path, image_size)
    z = extract_features(image_chw, W)  # [1,5+P,1024]
    z_cls = z[:, 0:1]
    z_reg = z[:, 1:1 + N_REG]
    z_patch = z[:, 1 + N_REG:]  # [1,P,1024]
    pn = image_size // PATCH
    z_patchmap = z_patch.reshape(1, pn, pn, HID)  # [1,h,w,1024]
    z_global = np.concatenate([z_cls, z_reg], axis=1)  # [1,5,1024]
    z_proj = proj_grid_forward(
        z_patchmap[0], cam["camera_angle_x"], cam["distance"], cam["mesh_scale"],
        grid_resolution=grid_resolution, image_resolution=image_size,
    )[None]  # [1,R^3,1024]
    return z_global.astype(np.float32), z_proj.astype(np.float32), z_patchmap


# =============================================================================
# self-validation
# =============================================================================
def _stats(name, got, ref):
    d = np.abs(got - ref)
    denom = np.abs(ref) + 1e-6
    maxabs = float(d.max())
    meanabs = float(d.mean())
    maxrel = float((d / denom).max())
    print(f"  {name:10s} shape={got.shape}  maxabs={maxabs:.3e}  "
          f"meanabs={meanabs:.3e}  maxrel={maxrel:.3e}")
    return maxabs


def main():
    base = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
    img = os.path.join(base, "pre", "preprocessed.png")
    cam = json.load(open(os.path.join(base, "cam.json")))
    g_ref = np.load(os.path.join(base, "stage1_cond", "global.npy"))
    p_ref = np.load(os.path.join(base, "stage1_cond", "proj.npy"))

    print(f"[dinov3_proj] image={img}")
    print(f"[dinov3_proj] cam={cam}")
    z_global, z_proj, _ = extract(img, cam)

    print("[dinov3_proj] vs goldens:")
    m_g = _stats("z_global", z_global, g_ref)
    m_p = _stats("z_proj", z_proj, p_ref)

    tol = 1e-3
    ok = m_g < tol and m_p < tol
    print(f"[dinov3_proj] {'PASS' if ok else 'FAIL'} (tol={tol:.0e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
