#!/usr/bin/env python
# capture_vecset_r0.py — R0 golden capture for the VecSet mesh-condition encoder (real path):
# REAL trained weights from the proven checkpoint + the REAL rng.choice(seed=0)+FPS sampling +
# GPU fp32 (the TRUE-fp32 oracle, NVIDIA_TF32_OVERRIDE=0 — NOT the bf16/flash prod path). Dumps the
# encoder weights (npy) + pc/feats/sampled_pc/sampled_feats/latents so vecset_encode validates the
# C++/ggml graph end-to-end against the real model. Run from the SkinTokens repo:
#   cd /mnt/hdd/3d/avatar-shootout/SkinTokens && NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=$PWD \
#       .venv/bin/python <thisdir>/capture_vecset_r0.py <outdir> [mesh.glb]
import os, sys
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import numpy as np, torch, trimesh
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); np.random.seed(0)

OUT  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vecset_r0"
MESH = sys.argv[2] if len(sys.argv) > 2 else "examples/giraffe.glb"
CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

from src.model.michelangelo.models.modules.embedder import FourierEmbedder
from src.model.michelangelo.models.tsal.sal_perceiver import CrossAttentionEncoder
from src.model.utils import fps

# exact checkpoint config (model_config.mesh_encoder)
fe = FourierEmbedder(num_freqs=8, include_pi=False)
enc = CrossAttentionEncoder(
    device=dev, dtype=torch.float32, num_latents=256, fourier_embedder=fe,
    point_feats=3, width=512, heads=8, layers=8, init_scale=0.25*(1.0/512)**0.5,
    qkv_bias=False, flash=False, use_ln_post=True, query_method=False,
    use_full_input=True, token_num=512, no_query=True,
).to(dev).eval()

# load the REAL encoder weights from the checkpoint (strip the mesh_encoder.encoder. prefix)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
pre = "mesh_encoder.encoder."
enc_sd = {k[len(pre):]: v.float() for k, v in sd.items() if k.startswith(pre)}
missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
print(f"[r0] loaded {len(enc_sd)} encoder weights (missing={len(missing)} unexpected={len(unexpected)})")
assert not missing, f"missing weights: {missing[:5]}"

# input mesh -> normalize to [-1,1]^3 -> sample N surface points + normals (the host front-end)
scene = trimesh.load(MESH, process=False)
mesh = trimesh.util.concatenate([g for g in scene.geometry.values()]) if hasattr(scene, "geometry") else scene
v = np.asarray(mesh.vertices, np.float64); c = (v.max(0) + v.min(0)) / 2
mesh.vertices = (v - c) / (np.abs(v - c).max())          # center + scale to [-1,1]^3
N = 8192
pts, fid = trimesh.sample.sample_surface(mesh, N)
nrm = mesh.face_normals[fid]
pc    = torch.tensor(np.asarray(pts), dtype=torch.float32, device=dev)[None]   # [1,N,3]
feats = torch.tensor(np.asarray(nrm), dtype=torch.float32, device=dev)[None]   # [1,N,3]
print(f"[r0] mesh={MESH} sampled N={N} bbox after norm: {pc[0].min(0).values.tolist()}..{pc[0].max(0).values.tolist()}")

with torch.no_grad():
    # replicate CrossAttentionEncoder._forward (query_method=False, use_full_input=True), capturing the
    # FPS-sampled set. seed=0 + random_start=False => deterministic, so this == the real _forward.
    token_num = 512
    rng = np.random.default_rng(seed=0)
    ind = rng.choice(pc.shape[1], token_num * 4, replace=token_num * 4 > pc.shape[1])
    pre_pc, pre_feats = pc[:, ind, :], feats[:, ind, :]
    Bn, Nn, D = pre_pc.shape; C = pre_feats.shape[-1]
    pos, pos_feats = pre_pc.view(Bn * Nn, D), pre_feats.view(Bn * Nn, C)
    batch = torch.repeat_interleave(torch.arange(Bn, device=dev), Nn)
    idx = fps(pos, batch, ratio=1.0 / 4, random_start=False)
    sampled_pc    = pos[idx].view(Bn, -1, 3)
    sampled_feats = pos_feats[idx].view(Bn, -1, C)
    data         = enc.input_proj(torch.cat([fe(pc), feats], dim=-1))
    sampled_data = enc.input_proj(torch.cat([fe(sampled_pc), sampled_feats], dim=-1))
    lat = enc.cross_attn(sampled_data, data)
    lat = enc.self_attn(lat)
    if enc.ln_post is not None:
        lat = enc.ln_post(lat)                                       # [1, Q, 512]

def save(n, t): np.save(os.path.join(OUT, n + ".npy"), np.ascontiguousarray(np.asarray(t)))
save("pc",            pc[0].cpu().numpy())
save("feats",         feats[0].cpu().numpy())
save("sampled_pc",    sampled_pc[0].cpu().numpy())
save("sampled_feats", sampled_feats[0].cpu().numpy())
save("latents",       lat[0].cpu().numpy())
for k, v in enc.state_dict().items():
    save("mesh_encoder.encoder." + k, v.float().cpu().numpy())
print(f"[r0] sampled_pc {tuple(sampled_pc[0].shape)}  latents {tuple(lat[0].shape)} "
      f"mean {lat.mean():.5f} std {lat.std():.5f} -> {OUT}")
