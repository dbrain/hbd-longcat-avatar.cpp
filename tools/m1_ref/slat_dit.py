"""
NumPy (fp32, CPU) reference of the Pixal3D Shape-SLat LR flow DiT (M2):
  ElasticSLatFlowModel / SLatFlowModel.forward (structured_latent_flow.py).

For batch=1 the "full varlen sparse attention" is plain attention over the N voxel
tokens (single sequence), so the block math is IDENTICAL to the SS dense DiT
(ss_dit.py, validated vs torch @1.5e-5). The ONLY new pieces vs ss_dit.py:
  - tokens = N voxels (no grid flatten); RoPE phases from coords[:,1:] (3D sparse RoPE)
  - proj_in_channels = 2048 (proj_linear 2048->1536), in/out channels = 32.

This file REUSES ss_dit.py's validated primitives and just swaps the rope + dims.

Sparse 3D RoPE (SparseRotaryPositionEmbedder, rope.py):
  head_dim 128, dim 3, freq_dim = 128//2//3 = 21, freqs = 1/10000^(arange(21)/21).
  phases[n] = outer(coords[n] (3 vals), freqs) -> [3,21] -> flatten [63], pad to 64
  with angle 0. interleaved-complex rotate (same as ss_dit apply_rope).

Config (slat_flow_512): model_ch 1536, 30 blocks, 12 heads (hd 128), mlp 8192,
share_mod, qk_rms_norm (self+cross), proj_in 2048, in/out 32, pe_mode rope.
"""
import os
import numpy as np

import ss_dit as sd  # reuse validated primitives (linear, silu, gelu_tanh, layer_norm, ...)

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
MODEL_CHANNELS = 1536
NUM_BLOCKS = 30
NUM_HEADS = 12
HEAD_DIM = 128
IN_CHANNELS = 32
PROJ_IN = 2048
ROPE_THETA = 10000.0
ROPE_DIM = 3


# ---- sparse 3D rope: per-voxel complex phases [N, 64] (cos, sin) ----
def sparse_rope_cos_sin(coords_xyz):
    """coords_xyz [N,3] int -> cos,sin [N,64] for interleaved-complex rope."""
    N = coords_xyz.shape[0]
    freq_dim = HEAD_DIM // 2 // ROPE_DIM  # 21
    freqs = np.arange(freq_dim, dtype=np.float32) / freq_dim
    freqs = 1.0 / (ROPE_THETA ** freqs)                      # rope_freq=(1,10000)
    # outer(coords.reshape(-1), freqs) -> [N*3, 21] -> [N, 63]
    flat = coords_xyz.reshape(-1).astype(np.float32)
    phases = np.outer(flat, freqs).reshape(N, ROPE_DIM * freq_dim)  # [N,63]
    pad = HEAD_DIM // 2 - phases.shape[1]                     # 64-63 = 1
    if pad > 0:
        phases = np.concatenate([phases, np.zeros((N, pad), np.float32)], axis=1)  # [N,64]
    return np.cos(phases).astype(np.float32), np.sin(phases).astype(np.float32)


def apply_rope_sparse(x, cos, sin):
    """x [B,N,H,D]; cos/sin [N,D/2]. interleaved complex multiply (matches rope.py)."""
    B, N, H, D = x.shape
    xc = x.reshape(B, N, H, D // 2, 2)
    a = xc[..., 0]; b = xc[..., 1]                            # [B,N,H,D/2]
    c = cos[None, :, None, :]; s = sin[None, :, None, :]      # [1,N,1,D/2]
    out = np.empty_like(xc)
    out[..., 0] = a * c - b * s
    out[..., 1] = a * s + b * c
    return out.reshape(B, N, H, D)


class Weights(sd.Weights):
    def __init__(self, npz_path, coords_xyz):
        z = np.load(npz_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        cos, sin = sparse_rope_cos_sin(coords_xyz)
        self.rope_cos = cos                                   # [N,64]
        self.rope_sin = sin


# ---- block (mirror ss_dit.block_forward; rope is sparse) ----
def self_attn(W, prefix, h):
    B, L, Cdim = h.shape
    qkv = sd.linear(h, W[prefix + "to_qkv.weight"], W[prefix + "to_qkv.bias"]).reshape(B, L, 3, NUM_HEADS, HEAD_DIM)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = sd.multihead_rms_norm(q, W[prefix + "q_rms_norm.gamma"])
    k = sd.multihead_rms_norm(k, W[prefix + "k_rms_norm.gamma"])
    q = apply_rope_sparse(q, W.rope_cos, W.rope_sin)
    k = apply_rope_sparse(k, W.rope_cos, W.rope_sin)
    out = sd.sdpa(q, k, v).reshape(B, L, Cdim)
    return sd.linear(out, W[prefix + "to_out.weight"], W[prefix + "to_out.bias"])


def block_forward(W, i, x, t_emb_mod, global_ctx, proj_ctx):
    bp = f"blocks.{i}."
    mod6 = W[bp + "modulation"] + t_emb_mod
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = [c[:, None, :] for c in np.split(mod6, 6, axis=1)]
    h = sd.layer_norm(x, eps=1e-6) * (1 + sc_msa) + sh_msa
    h = self_attn(W, bp + "self_attn.", h)
    x = x + h * g_msa
    h = sd.layer_norm(x, eps=1e-6, weight=W[bp + "norm2.weight"], bias=W[bp + "norm2.bias"])
    # ProjectAttention: cross_attn(global) + proj_linear(proj[N,2048])
    cab = bp + "cross_attn.cross_attn_block."
    global_out = sd.cross_attn_global(W, cab, h, global_ctx)
    proj_out = sd.linear(proj_ctx, W[bp + "cross_attn.proj_linear.weight"], W[bp + "cross_attn.proj_linear.bias"])
    x = x + proj_out + global_out
    h = sd.layer_norm(x, eps=1e-6) * (1 + sc_mlp) + sh_mlp
    h = sd.mlp(W, bp, h)
    x = x + h * g_mlp
    return x


def model_forward(W, x_feats, t_scaled, cond):
    """x_feats [N,32]; t_scaled [1] (=1000*t); cond {global[1,5,1024], proj[N,2048]}."""
    h = x_feats[None].astype(np.float32)                      # [1,N,32]
    h = sd.linear(h, W["input_layer.weight"], W["input_layer.bias"])  # [1,N,1536]
    t_emb = sd.t_embedder(W, np.asarray(t_scaled, np.float32))
    t_emb = sd.silu(t_emb)
    t_emb_mod = sd.linear(t_emb, W["adaLN_modulation.1.weight"], W["adaLN_modulation.1.bias"])
    g = np.asarray(cond["global"], np.float32)
    p = np.asarray(cond["proj"], np.float32)[None]            # [1,N,2048]
    for i in range(NUM_BLOCKS):
        h = block_forward(W, i, h, t_emb_mod, g, p)
    h = sd.layer_norm(h, eps=1e-5)
    h = sd.linear(h, W["out_layer.weight"], W["out_layer.bias"])  # [1,N,32]
    return h[0]                                                # [N,32]


# ============================================================
def _build_torch_model(coords):
    """Try to instantiate the real ElasticSLatFlowModel on CPU for a single-forward
    cross-check. Returns (model, torch, sp) or None if the sparse infra needs CUDA."""
    import sys
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
    import torch
    try:
        from pixal3d.models.structured_latent_flow import ElasticSLatFlowModel
        import pixal3d.modules.sparse as sp
        ElasticSLatFlowModel.initialize_weights = lambda self: None
        m = ElasticSLatFlowModel(
            resolution=32, in_channels=32, out_channels=32, model_channels=1536,
            cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
            pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
            image_attn_mode="proj", proj_in_channels=2048, dtype="float32",
        )
        z = np.load(os.path.join(WEIGHTS_DIR, "slat_flow_512.npz"))
        sdct = m.state_dict()
        for k in sdct.keys():
            if k in z.files:
                sdct[k] = torch.from_numpy(z[k].astype(np.float32))
        m.load_state_dict(sdct, strict=True)
        m.eval()
        return m, torch, sp
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[slat_dit] torch sparse model unavailable on CPU:", e)
        return None


def main():
    import sys
    GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
    coords = np.load(os.path.join(GOLD, "stage2_cond", "proj_coords.npy"))  # [N,4]
    coords_xyz = coords[:, 1:].astype(np.float32)
    N = coords.shape[0]
    g = np.load(os.path.join(GOLD, "stage2_cond", "global.npy")).astype(np.float32)
    proj = np.load(os.path.join(GOLD, "stage2_cond", "proj_feats.npy")).astype(np.float32)  # [N,2048]
    cond = {"global": g, "proj": proj}

    W = Weights(os.path.join(WEIGHTS_DIR, "slat_flow_512.npz"), coords_xyz)
    rng = np.random.RandomState(0)
    x = rng.randn(N, 32).astype(np.float32)
    t = np.array([537.0], np.float32)

    my = model_forward(W, x, t, cond)
    print(f"[slat_dit] forward: x[{x.shape}] -> v[{my.shape}]")

    # save single-forward reference for the C++ port
    REFS = os.path.join(os.path.dirname(__file__), "cpp_port", "refs")
    os.makedirs(REFS, exist_ok=True)
    np.save(os.path.join(REFS, "slat_dit_x.npy"), x)
    np.save(os.path.join(REFS, "slat_dit_t.npy"), t)
    np.save(os.path.join(REFS, "slat_dit_v.npy"), my)
    np.save(os.path.join(REFS, "slat_coords.npy"), coords.astype(np.int32))
    print(f"[slat_dit] saved slat_dit_{{x,t,v}}.npy + slat_coords.npy to refs/")

    # cross-check vs real torch model (if sparse infra runs on CPU)
    built = _build_torch_model(coords)
    if built is not None:
        m, torch, sp = built
        with torch.no_grad():
            xt = sp.SparseTensor(feats=torch.from_numpy(x), coords=torch.from_numpy(coords.astype(np.int32)))
            pt = sp.SparseTensor(feats=torch.from_numpy(proj), coords=torch.from_numpy(coords.astype(np.int32)))
            tcond = {"global": torch.from_numpy(g), "proj": pt}
            out = m(xt, torch.from_numpy(t), tcond)
            tv = out.feats.numpy()
        ma = np.abs(my - tv).max(); md = np.median(np.abs(my - tv))
        print(f"[CROSS-CHECK] numpy vs torch SLat: maxabs={ma:.3e} median={md:.3e} -> "
              f"{'PASS' if ma < 1e-3 else 'FAIL'}")
    else:
        print("[slat_dit] (no torch cross-check — numpy ref reuses ss_dit primitives "
              "validated @1.5e-5; sparse-rope validated separately)")


if __name__ == "__main__":
    main()
