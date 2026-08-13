"""
NumPy (fp32, CPU) reference of the Pixal3D Sparse-Structure flow DiT
(SparseStructureFlowModel) + the FlowEulerGuidanceIntervalSampler.

Deliverable: pure-numpy `model_forward` and `sample`. Torch is NOT used here.
Validation (torch oracle + golden compare) lives in the __main__ block, which
only uses torch to (a) generate the seed-42 noise and (b) run the real model as
an independent CPU cross-check.

Math mirrored exactly from:
  pixal3d/models/sparse_structure_flow.py        (forward, TimestepEmbedder)
  pixal3d/modules/transformer/modulated.py       (ModulatedTransformerCrossBlock, share_mod)
  pixal3d/modules/attention/modules.py           (MultiHeadAttention, MultiHeadRMSNorm)
  pixal3d/modules/attention/proj_attention.py    (ProjectAttention)
  pixal3d/modules/attention/rope.py              (complex interleaved RoPE)
  pixal3d/modules/transformer/blocks.py          (FeedForwardNet: GELU tanh, hidden 8192)
  pipelines/samplers/flow_euler.py + mixins      (CFG + interval + std-rescale)

Config: model_channels 1536, num_blocks 30, num_heads 12 (head_dim 128),
mlp hidden 8192, share_mod, qk_rms_norm (self+cross), image_attn_mode proj.
"""

import os
import json
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

# ---- config (verified) ----
MODEL_CHANNELS = 1536
NUM_BLOCKS = 30
NUM_HEADS = 12
HEAD_DIM = MODEL_CHANNELS // NUM_HEADS  # 128
RESOLUTION = 16
IN_CHANNELS = 8
SEQ = RESOLUTION ** 3  # 4096

# ============================================================
# primitive ops (all fp32)
# ============================================================

def linear(x, w, b=None):
    # x: [..., in], w: [out, in] (torch convention), b: [out]
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def silu(x):
    return x / (1.0 + np.exp(-x))


_GELU_C = np.float32(np.sqrt(2.0 / np.pi))


def gelu_tanh(x):
    # nn.GELU(approximate="tanh"); avoid x**3 (slow elementwise pow)
    inner = _GELU_C * (x + np.float32(0.044715) * (x * x * x))
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))


def layer_norm(x, eps=1e-6, weight=None, bias=None):
    # normalize over last dim. F.layer_norm uses eps in default (1e-5) but the
    # LayerNorm32 modules use eps=1e-6; the final F.layer_norm(h, h.shape[-1:])
    # uses default 1e-5. Caller picks eps.
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # population variance (torch layer_norm)
    y = (x - mu) / np.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def softmax(x, axis=-1):
    # in-place to avoid extra ~800MB transients on the [H,L,L] attn matrix
    m = x.max(axis=axis, keepdims=True)
    x -= m
    np.exp(x, out=x)
    s = x.sum(axis=axis, keepdims=True)
    x /= s
    return x


def multihead_rms_norm(x, gamma):
    # x: [B, L, H, D]; gamma: [H, D]; scale = sqrt(D)
    # F.normalize(x.float(), dim=-1) * gamma * sqrt(D)
    D = x.shape[-1]
    norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))  # L2 over last dim
    # torch F.normalize uses eps=1e-12 added to the norm denominator
    xn = x / np.maximum(norm, 1e-12)
    return xn * gamma * np.sqrt(D)


def apply_rope(x, cos, sin):
    # x: [B, L, H, D]; cos/sin: [1, L, 1, D/2] (real/imag of saved phases)
    # view_as_complex on reshape(...,-1,2) -> interleaved (even=real, odd=imag)
    # complex multiply (a+bi)(c+di) = (ac-bd) + (ad+bc)i, done in fp32 reals.
    B, L, H, D = x.shape
    xc = x.reshape(B, L, H, D // 2, 2)
    a = xc[..., 0]                                    # [B,L,H,D/2] real parts
    b = xc[..., 1]                                    # imag parts
    out = np.empty((B, L, H, D // 2, 2), dtype=np.float32)
    out[..., 0] = a * cos - b * sin
    out[..., 1] = a * sin + b * cos
    return out.reshape(B, L, H, D)


def sdpa(q, k, v):
    # q,k,v: [B, L, H, D] -> [B, H, L, D]
    q = np.ascontiguousarray(q.transpose(0, 2, 1, 3))
    k = np.ascontiguousarray(k.transpose(0, 2, 1, 3))
    v = np.ascontiguousarray(v.transpose(0, 2, 1, 3))
    scale = np.float32(1.0 / np.sqrt(q.shape[-1]))
    attn = (q @ k.transpose(0, 1, 3, 2))   # [B, H, L, L]
    attn *= scale
    attn = softmax(attn, axis=-1)
    out = attn @ v                       # [B, H, L, D]
    return out.transpose(0, 2, 1, 3)     # [B, L, H, D]


# ============================================================
# weight container
# ============================================================

class Weights:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        # precompute complex rope phases [4096, 64]
        rp = self.w["rope_phases"]  # [4096, 64, 2]
        self.rope_phases = (rp[..., 0] + 1j * rp[..., 1]).astype(np.complex64)
        # real/imag broadcast-ready [1, L, 1, D/2] for fast apply_rope
        self.rope_cos = np.ascontiguousarray(rp[..., 0])[None, :, None, :].astype(np.float32)
        self.rope_sin = np.ascontiguousarray(rp[..., 1])[None, :, None, :].astype(np.float32)

    def __getitem__(self, k):
        return self.w[k]


# ============================================================
# DiT pieces
# ============================================================

def timestep_embedding(t, dim=256, max_period=10000.0):
    # t: [B]; returns [B, dim]
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half)
    args = t[:, None].astype(np.float32) * freqs[None]
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1).astype(np.float32)


def t_embedder(W, t):
    emb = timestep_embedding(t, 256)
    emb = linear(emb, W["t_embedder.mlp.0.weight"], W["t_embedder.mlp.0.bias"])
    emb = silu(emb)
    emb = linear(emb, W["t_embedder.mlp.2.weight"], W["t_embedder.mlp.2.bias"])
    return emb  # [B, 1536]


def self_attn(W, prefix, h, rope_phases=None):
    B, L, C = h.shape
    qkv = linear(h, W[prefix + "to_qkv.weight"], W[prefix + "to_qkv.bias"])
    qkv = qkv.reshape(B, L, 3, NUM_HEADS, HEAD_DIM)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]
    q = multihead_rms_norm(q, W[prefix + "q_rms_norm.gamma"])
    k = multihead_rms_norm(k, W[prefix + "k_rms_norm.gamma"])
    q = apply_rope(q, W.rope_cos, W.rope_sin)
    k = apply_rope(k, W.rope_cos, W.rope_sin)
    out = sdpa(q, k, v)  # [B, L, H, D]
    out = out.reshape(B, L, C)
    out = linear(out, W[prefix + "to_out.weight"], W[prefix + "to_out.bias"])
    return out


def cross_attn_global(W, prefix, h, global_ctx):
    # MultiHeadAttention type=cross, qk_rms_norm_cross
    B, L, C = h.shape
    Lkv = global_ctx.shape[1]
    q = linear(h, W[prefix + "to_q.weight"], W[prefix + "to_q.bias"])
    kv = linear(global_ctx, W[prefix + "to_kv.weight"], W[prefix + "to_kv.bias"])
    q = q.reshape(B, L, NUM_HEADS, HEAD_DIM)
    kv = kv.reshape(B, Lkv, 2, NUM_HEADS, HEAD_DIM)
    k = kv[:, :, 0]
    v = kv[:, :, 1]
    q = multihead_rms_norm(q, W[prefix + "q_rms_norm.gamma"])
    k = multihead_rms_norm(k, W[prefix + "k_rms_norm.gamma"])
    out = sdpa(q, k, v)
    out = out.reshape(B, L, C)
    out = linear(out, W[prefix + "to_out.weight"], W[prefix + "to_out.bias"])
    return out


def proj_attention(W, bp, h, global_ctx, proj_ctx):
    # ProjectAttention: global_out + proj_linear(proj)
    cab = bp + "cross_attn.cross_attn_block."
    global_out = cross_attn_global(W, cab, h, global_ctx)
    proj_out = linear(proj_ctx, W[bp + "cross_attn.proj_linear.weight"],
                      W[bp + "cross_attn.proj_linear.bias"])
    return proj_out + global_out


def mlp(W, bp, h):
    h = linear(h, W[bp + "mlp.mlp.0.weight"], W[bp + "mlp.mlp.0.bias"])
    h = gelu_tanh(h)
    h = linear(h, W[bp + "mlp.mlp.2.weight"], W[bp + "mlp.mlp.2.bias"])
    return h


def block_forward(W, i, x, t_emb_mod, global_ctx, proj_ctx, rope_phases):
    bp = f"blocks.{i}."
    # share_mod: per-block learned modulation bias + shared t_emb -> chunk 6
    mod6 = W[bp + "modulation"] + t_emb_mod  # [B, 9216]
    chunks = np.split(mod6, 6, axis=1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
    # add token axis
    shift_msa = shift_msa[:, None, :]
    scale_msa = scale_msa[:, None, :]
    gate_msa = gate_msa[:, None, :]
    shift_mlp = shift_mlp[:, None, :]
    scale_mlp = scale_mlp[:, None, :]
    gate_mlp = gate_mlp[:, None, :]

    # norm1 non-affine
    h = layer_norm(x, eps=1e-6)
    h = h * (1 + scale_msa) + shift_msa
    h = self_attn(W, bp + "self_attn.", h, rope_phases)
    h = h * gate_msa
    x = x + h

    # norm2 AFFINE
    h = layer_norm(x, eps=1e-6, weight=W[bp + "norm2.weight"], bias=W[bp + "norm2.bias"])
    h = proj_attention(W, bp, h, global_ctx, proj_ctx)
    x = x + h

    # norm3 non-affine
    h = layer_norm(x, eps=1e-6)
    h = h * (1 + scale_mlp) + shift_mlp
    h = mlp(W, bp, h)
    h = h * gate_mlp
    x = x + h
    return x


def model_forward(W, x, t_scaled, cond, neg_cond_unused=None):
    """
    x: [1, 8, 16, 16, 16]
    t_scaled: [1] timestep already multiplied by 1000 (model receives t*1000)
    cond: dict {'global': [1,5,1024], 'proj': [1,4096,1024]}
    returns v-prediction [1, 8, 16, 16, 16]
    """
    x = np.asarray(x, dtype=np.float32)
    B = x.shape[0]
    # flatten: x.view(B,C,-1).permute(0,2,1)
    h = x.reshape(B, IN_CHANNELS, -1).transpose(0, 2, 1)  # [B, 4096, 8]
    h = linear(h, W["input_layer.weight"], W["input_layer.bias"])  # [B,4096,1536]

    t_emb = t_embedder(W, np.asarray(t_scaled, dtype=np.float32))  # [B,1536]
    # share_mod global adaLN: SiLU + Linear(1536->9216)
    t_emb = silu(t_emb)
    t_emb_mod = linear(t_emb, W["adaLN_modulation.1.weight"], W["adaLN_modulation.1.bias"])  # [B,9216]

    global_ctx = np.asarray(cond["global"], dtype=np.float32)
    proj_ctx = np.asarray(cond["proj"], dtype=np.float32)

    for i in range(NUM_BLOCKS):
        h = block_forward(W, i, h, t_emb_mod, global_ctx, proj_ctx, W.rope_phases)

    # final F.layer_norm non-affine (default eps 1e-5)
    h = layer_norm(h, eps=1e-5)
    h = linear(h, W["out_layer.weight"], W["out_layer.bias"])  # [B,4096,8]
    h = h.transpose(0, 2, 1).reshape(B, IN_CHANNELS, RESOLUTION, RESOLUTION, RESOLUTION)
    return h


# ============================================================
# FlowEulerGuidanceIntervalSampler
# ============================================================

class FlowSampler:
    def __init__(self, W, sigma_min=1e-5):
        self.W = W
        self.sigma_min = sigma_min

    def _v_to_xstart_eps(self, x_t, t, v):
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _raw_forward(self, x_t, t, cond):
        t_scaled = np.array([1000.0 * t] * x_t.shape[0], dtype=np.float32)
        return model_forward(self.W, x_t, t_scaled, cond)

    def _cfg_forward(self, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale):
        if guidance_strength == 1:
            return self._raw_forward(x_t, t, cond)
        if guidance_strength == 0:
            return self._raw_forward(x_t, t, neg_cond)
        pred_pos = self._raw_forward(x_t, t, cond)
        pred_neg = self._raw_forward(x_t, t, neg_cond)
        pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        if guidance_rescale > 0:
            x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
            x_0_cfg = self._pred_to_xstart(x_t, t, pred)
            axes = tuple(range(1, x_0_pos.ndim))
            std_pos = x_0_pos.std(axis=axes, keepdims=True, ddof=0)
            std_cfg = x_0_cfg.std(axis=axes, keepdims=True, ddof=0)
            x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
            x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
            pred = self._xstart_to_pred(x_t, t, x_0)
        return pred

    def _interval_forward(self, x_t, t, cond, neg_cond, guidance_strength,
                          guidance_rescale, guidance_interval):
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self._cfg_forward(x_t, t, cond, neg_cond, guidance_strength, guidance_rescale)
        else:
            # guidance_strength=1 -> plain cond forward (rescale not applied)
            return self._cfg_forward(x_t, t, cond, neg_cond, 1, guidance_rescale)

    def sample(self, noise, cond, neg_cond, steps=12, rescale_t=5.0,
               guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=(0.6, 1.0), verbose=False):
        sample = np.asarray(noise, dtype=np.float32)
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        for si, (t, t_prev) in enumerate(t_pairs):
            pred_v = self._interval_forward(sample, t, cond, neg_cond,
                                            guidance_strength, guidance_rescale,
                                            guidance_interval)
            sample = sample - (t - t_prev) * pred_v
            if verbose:
                print(f"  step {si+1}/{steps} t={t:.4f} done", flush=True)
        return sample


def sample(noise, cond, neg_cond, W=None, verbose=False):
    if W is None:
        W = Weights(os.path.join(WEIGHTS_DIR, "ss_flow.npz"))
    s = FlowSampler(W, sigma_min=1e-5)
    return s.sample(noise, cond, neg_cond, steps=12, rescale_t=5.0,
                    guidance_strength=7.5, guidance_rescale=0.7,
                    guidance_interval=(0.6, 1.0), verbose=verbose)


# ============================================================
# Validation
# ============================================================

def _load_cond():
    base = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages/stage1_cond"
    g = np.load(os.path.join(base, "global.npy")).astype(np.float32)
    p = np.load(os.path.join(base, "proj.npy")).astype(np.float32)
    cond = {"global": g, "proj": p}
    neg = {"global": np.zeros_like(g), "proj": np.zeros_like(p)}
    return cond, neg


def _build_torch_model():
    """Instantiate the real SparseStructureFlowModel on CPU and load weights."""
    import sys
    os.environ.setdefault("ATTN_BACKEND", "sdpa")  # flash_attn not installed; use torch sdpa
    sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
    import torch
    from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
    # Skip the expensive xavier init of 1.3B params — we overwrite every weight
    # from the npz immediately after construction. Saves time + peak RAM.
    SparseStructureFlowModel.initialize_weights = lambda self: None
    m = SparseStructureFlowModel(
        resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
        out_channels=8, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="proj", dtype="float32",
    )
    # load weights from npz
    z = np.load(os.path.join(WEIGHTS_DIR, "ss_flow.npz"))
    sd = m.state_dict()
    loaded = 0
    for k in sd.keys():
        if k == "rope_phases":
            # saved as [4096,64,2]; model buffer is complex [4096,64]
            rp = z["rope_phases"]
            sd[k] = torch.view_as_complex(torch.from_numpy(rp.copy()))
            loaded += 1
            continue
        if k in z.files:
            sd[k] = torch.from_numpy(z[k].astype(np.float32))
            loaded += 1
        else:
            print("MISSING in npz:", k, tuple(sd[k].shape))
    m.load_state_dict(sd, strict=True)
    m.eval()
    print(f"torch model: loaded {loaded}/{len(sd)} tensors")
    return m, torch


def main():
    np.set_printoptions(precision=6, suppress=True)
    W = Weights(os.path.join(WEIGHTS_DIR, "ss_flow.npz"))
    cond, neg = _load_cond()
    print("cond global", cond["global"].shape, "proj", cond["proj"].shape)

    # ---------- Validation 1: cross-check vs real torch model, single forward ----------
    try:
        m, torch = _build_torch_model()
        rng = np.random.RandomState(0)
        x_fixed = rng.randn(1, 8, 16, 16, 16).astype(np.float32)
        t_fixed = np.array([537.0], dtype=np.float32)  # arbitrary t*1000

        my_out = model_forward(W, x_fixed, t_fixed, cond)

        with torch.no_grad():
            tx = torch.from_numpy(x_fixed)
            tt = torch.from_numpy(t_fixed)
            tcond = {"global": torch.from_numpy(cond["global"]),
                     "proj": torch.from_numpy(cond["proj"])}
            torch_out = m(tx, tt, tcond).numpy()

        cc_maxabs = np.max(np.abs(my_out - torch_out))
        cc_median = np.median(np.abs(my_out - torch_out))
        print(f"\n[CROSS-CHECK] numpy vs real torch model (single forward):", flush=True)
        print(f"  maxabs  = {cc_maxabs:.3e}   median = {cc_median:.3e}", flush=True)
        print(f"  target  maxabs < ~1e-3 ->", "PASS" if cc_maxabs < 1e-3 else "FAIL", flush=True)
        cross_ok = cc_maxabs < 1e-3
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[CROSS-CHECK] could not instantiate torch model:", e)
        cross_ok = None

    # ---------- noise (seed 42, CPU) ----------
    import torch as _torch
    _torch.manual_seed(42)
    noise = _torch.randn(1, 8, 16, 16, 16).numpy().astype(np.float32)

    # ---------- Validation 2: E2E vs golden ----------
    print("\n[E2E] running 12-step sampler...", flush=True)
    z_s = sample(noise, cond, neg, W=W, verbose=True)
    golden = np.load("/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/"
                     "golden_stages/stage1_ssdec/z_s.npy").astype(np.float32)
    e2e_maxabs = np.max(np.abs(z_s - golden))
    e2e_median = np.median(np.abs(z_s - golden))
    print(f"\n[E2E] z_s vs golden:")
    print(f"  z_s shape {z_s.shape}  golden {golden.shape}")
    print(f"  maxabs = {e2e_maxabs:.3e}   median = {e2e_median:.3e}")
    print(f"  targets: maxabs < ~5e-2 ->", "PASS" if e2e_maxabs < 5e-2 else "FAIL",
          f"| median < ~1e-2 ->", "PASS" if e2e_median < 1e-2 else "FAIL")

    # occupancy agreement (the real M1 signal)
    occ_my = (z_s > 0)
    occ_gold = (golden > 0)
    # (z_s occupancy isn't the decoder threshold, but a quick sanity signal)
    print(f"  sign-agreement on z_s elements: "
          f"{np.mean(occ_my == occ_gold)*100:.2f}%")

    e2e_ok = (e2e_maxabs < 5e-2) and (e2e_median < 1e-2)
    if cross_ok and not e2e_ok:
        print("\nNOTE: cross-check PASSED but E2E FAILED -> block math is correct, "
              "suspect noise reproduction.")
    print("\nRESULT:", "ALL PASS" if (cross_ok and e2e_ok) else "SEE ABOVE")


if __name__ == "__main__":
    main()
