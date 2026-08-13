#!/usr/bin/env python3
"""
Definitive fp32 closure for M3b Stage-3b: real ElasticSLatFlowModel(1024, grid 64) +
real FlowEulerGuidanceIntervalSampler in fp32, golden stage3b cond (DINOv3@1024+NAF over
the golden HR coords) + the seed-42-reproduced stage-3b noise -> denorm -> shape_slat[M,32].

Arch is IDENTICAL to stage-2 (M2) — only resolution (64) + weights (slat_flow_1024) differ;
resolution only sizes the unused-at-inference PE, coords drive the sparse RoPE.

Noise reproduction (single global seed-42 CPU RNG flows through all stages, nothing else
draws): stage1 randn(1,8,16,16,16); stage2 randn(N=1126,32); stage3b randn(M=4734,32).
(get_proj_cond_*, NAF, the upsample decoder, token-budget loop draw NO RNG — proven for M2.)

CPU only, fp32. Run:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 \
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python torch_stage3b_ref.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")

import torch


def build_model():
    from pixal3d.models.structured_latent_flow import ElasticSLatFlowModel
    ElasticSLatFlowModel.initialize_weights = lambda self: None
    m = ElasticSLatFlowModel(
        resolution=64, in_channels=32, out_channels=32, model_channels=1536,
        cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="proj", proj_in_channels=2048, dtype="float32",
    )
    z = np.load(os.path.join(WEIGHTS_DIR, "slat_flow_1024.npz"))
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

    coords = np.load(os.path.join(GOLD, "stage3b_cond", "proj_coords.npy")).astype(np.int32)  # [M,4]
    M = coords.shape[0]
    g = np.load(os.path.join(GOLD, "stage3b_cond", "global.npy")).astype(np.float32)
    proj = np.load(os.path.join(GOLD, "stage3b_cond", "proj_feats.npy")).astype(np.float32)   # [M,2048]
    N2 = int(np.load(os.path.join(GOLD, "stage2_cond", "proj_coords.npy")).shape[0])          # 1126
    print(f"[torch_s3b] M={M} N_stage2={N2} coords{coords.shape} proj{proj.shape}", flush=True)

    m = build_model()
    ct = torch.from_numpy(coords)
    proj_st = sp.SparseTensor(feats=torch.from_numpy(proj), coords=ct)
    cond = {"global": torch.from_numpy(g), "proj": proj_st}
    neg = {"global": torch.zeros_like(cond["global"]),
           "proj": sp.SparseTensor(feats=torch.zeros_like(proj_st.feats), coords=ct)}

    # reproduce seed-42 stage-3b noise = 3rd draw
    torch.manual_seed(42)
    _ = torch.randn(1, 8, 16, 16, 16)   # stage1
    _ = torch.randn(N2, 32)             # stage2
    noise_feats = torch.randn(M, m.in_channels)  # stage3b [M,32]
    np.save(os.path.join(HERE, "cpp_port", "refs", "stage3b_noise.npy"),
            noise_feats.numpy().astype(np.float32))
    noise = sp.SparseTensor(feats=noise_feats, coords=ct)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    print("[torch_s3b] sampling (real model + real sampler, fp32, 12 steps)...", flush=True)
    out = sampler.sample(
        m, noise, cond, neg_cond=neg, steps=12, rescale_t=3.0,
        guidance_strength=7.5, guidance_rescale=0.5, guidance_interval=(0.6, 1.0), verbose=True,
    )
    hr_raw = out.samples.feats.numpy().astype(np.float32)

    pj = json.load(open(os.path.join(GOLD, "configs", "pipeline.json")))
    norm = pj["args"]["shape_slat_normalization"]
    std = np.asarray(norm["std"], np.float32)[None]; mean = np.asarray(norm["mean"], np.float32)[None]
    hr = hr_raw * std + mean

    refs = os.path.join(HERE, "cpp_port", "refs")
    np.save(os.path.join(refs, "torch_shape_slat_fp32.npy"), hr_raw)
    np.save(os.path.join(refs, "torch_shape_slat_denorm_fp32.npy"), hr)
    print(f"[torch_s3b] saved stage3b_noise + torch_shape_slat_{{fp32,denorm_fp32}} ({hr.shape})", flush=True)

    gold = np.load(os.path.join(GOLD, "stage3b_out", "shape_slat_feats.npy")).astype(np.float32)
    d = np.abs(hr - gold)
    a = hr.reshape(-1); b = gold.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"\n[torch_s3b] shape_slat(fp32 denorm) vs golden(bf16): maxabs={d.max():.3e} "
          f"meanabs={d.mean():.3e} median={np.median(d):.3e} cosine={cos:.6f}")
    print(f"           golden range [{gold.min():.3f},{gold.max():.3f}] mine [{hr.min():.3f},{hr.max():.3f}]")
    print("[torch_s3b] NOISE CONFIRMED" if d.mean() < 0.4 else "[torch_s3b] WARNING: divergence")


if __name__ == "__main__":
    main()
