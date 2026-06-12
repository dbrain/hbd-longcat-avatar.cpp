#!/usr/bin/env python3
"""
Definitive fp32 closure for M2 Stage-2: run the REAL Pixal3D shape-SLat LR flow model
(ElasticSLatFlowModel, slat_flow_512) through the REAL FlowEulerGuidanceIntervalSampler
on CPU in fp32, with the GOLDEN stage-2 cond (DINOv3+NAF over the golden 1126 coords) and
the seed-42-reproduced stage-2 noise, then denormalize -> lr_slat[N,32].

This is the tight fp32 oracle the C++ M2 sampler validates against (~1e-3), and it also
confirms the NOISE reproduction by comparing the denormalized fp32 lr_slat to the bf16
golden (stage2_out/lr_slat_feats, ~1e-2 like the stage-1 z_s).

Noise reproduction (run() sets torch.manual_seed(42) ONCE, then the CPU global RNG flows):
  stage1: noise = randn(1,8,16,16,16)         (32768 draws)   [get_proj_cond_ss does NOT draw — proven in M1]
  stage2: noise = randn(N, in_channels=32)    (N*32 draws)     [continues the SAME RNG]
We replay that exact sequence on CPU. If the denorm lr_slat matches the bf16 golden at
~1e-2, the noise (and the no-RNG-in-between assumption through get_proj_cond_shape/NAF)
is confirmed; otherwise we fall back to capturing the real noise via a golden-hook replay.

CPU only, fp32. Run:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python torch_stage2_ref.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
sys.path.insert(0, HERE)

import torch
import json


def build_model(coords):
    from pixal3d.models.structured_latent_flow import ElasticSLatFlowModel
    ElasticSLatFlowModel.initialize_weights = lambda self: None  # skip 1.3B init
    m = ElasticSLatFlowModel(
        resolution=32, in_channels=32, out_channels=32, model_channels=1536,
        cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="proj", proj_in_channels=2048, dtype="float32",
    )
    z = np.load(os.path.join(WEIGHTS_DIR, "slat_flow_512.npz"))
    sd = m.state_dict()
    for k in sd.keys():
        if k in z.files:
            sd[k] = torch.from_numpy(z[k].astype(np.float32))
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def main():
    torch.set_grad_enabled(False)
    from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
    import pixal3d.modules.sparse as sp

    # golden cond + coords (the cond was computed over the golden 1126 coords)
    coords = np.load(os.path.join(GOLD, "stage2_cond", "proj_coords.npy")).astype(np.int32)  # [N,4]
    N = coords.shape[0]
    g = np.load(os.path.join(GOLD, "stage2_cond", "global.npy")).astype(np.float32)   # [1,5,1024]
    proj = np.load(os.path.join(GOLD, "stage2_cond", "proj_feats.npy")).astype(np.float32)  # [N,2048]
    print(f"[torch_s2] N={N} coords{coords.shape} global{g.shape} proj{proj.shape}", flush=True)

    m = build_model(coords)

    # cond / neg_cond (proj is a SparseTensor; neg is zeros_like)
    ct = torch.from_numpy(coords)
    proj_st = sp.SparseTensor(feats=torch.from_numpy(proj), coords=ct)
    cond = {"global": torch.from_numpy(g), "proj": proj_st}
    neg = {"global": torch.zeros_like(cond["global"]),
           "proj": sp.SparseTensor(feats=torch.zeros_like(proj_st.feats), coords=ct)}

    # ---- reproduce the seed-42 stage-2 noise (continues RNG after stage-1's draw) ----
    torch.manual_seed(42)
    _stage1_noise = torch.randn(1, 8, 16, 16, 16)   # discard (consumes 32768 RNG values)
    noise_feats = torch.randn(N, m.in_channels)      # stage-2 noise [N,32]
    np.save(os.path.join(HERE, "cpp_port", "refs", "stage2_noise.npy"),
            noise_feats.numpy().astype(np.float32))
    noise = sp.SparseTensor(feats=noise_feats, coords=ct)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    print("[torch_s2] sampling (real ElasticSLatFlowModel + real sampler, fp32, 12 steps)...", flush=True)
    out = sampler.sample(
        m, noise, cond, neg_cond=neg, steps=12, rescale_t=3.0,
        guidance_strength=7.5, guidance_rescale=0.5, guidance_interval=(0.6, 1.0),
        verbose=True,
    )
    lr_slat_raw = out.samples.feats.numpy().astype(np.float32)   # pre-denorm [N,32]

    # denorm: slat = slat*std + mean (32-dim shape_slat_normalization)
    pj = json.load(open(os.path.join(GOLD, "configs", "pipeline.json")))
    norm = pj["args"]["shape_slat_normalization"]
    std = np.asarray(norm["std"], np.float32)[None]
    mean = np.asarray(norm["mean"], np.float32)[None]
    lr_slat = lr_slat_raw * std + mean

    refs = os.path.join(HERE, "cpp_port", "refs")
    np.save(os.path.join(refs, "torch_lr_slat_fp32.npy"), lr_slat_raw)
    np.save(os.path.join(refs, "torch_lr_slat_denorm_fp32.npy"), lr_slat)
    print(f"[torch_s2] saved stage2_noise / torch_lr_slat_fp32 / torch_lr_slat_denorm_fp32 "
          f"({lr_slat.shape})", flush=True)

    # compare denorm fp32 vs the bf16 golden (confirms noise + pipeline)
    gold = np.load(os.path.join(GOLD, "stage2_out", "lr_slat_feats.npy")).astype(np.float32)
    d = np.abs(lr_slat - gold)
    rel = d / (np.abs(gold) + 1e-3)
    print(f"\n[torch_s2] lr_slat(fp32 torch, denorm) vs golden(bf16 torso):")
    print(f"           maxabs={d.max():.3e} meanabs={d.mean():.3e} median={np.median(d):.3e} "
          f"meanrel={rel.mean():.3e}")
    print(f"           golden range [{gold.min():.3f},{gold.max():.3f}]  mine [{lr_slat.min():.3f},{lr_slat.max():.3f}]")
    # cosine over the whole tensor (scale/shift robust sanity)
    a = lr_slat.reshape(-1); b = gold.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"           cosine(flat)={cos:.6f}")
    if d.mean() < 0.3:
        print("[torch_s2] NOISE CONFIRMED (bf16-level agreement) — fp32 oracle is the C++ target.")
    else:
        print("[torch_s2] WARNING: large divergence — RNG may drift in get_proj_cond_shape/NAF; "
              "capture real noise via golden-hook replay.")


if __name__ == "__main__":
    main()
