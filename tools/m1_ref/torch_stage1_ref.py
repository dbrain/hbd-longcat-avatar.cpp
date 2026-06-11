#!/usr/bin/env python3
"""
Definitive fp32 closure for M1 Stage-1: run the REAL Pixal3D SS flow model through the
REAL FlowEulerGuidanceIntervalSampler on CPU in fp32 (seed-42 noise, golden cond), then
decode to coords. This independently validates the SAMPLER loop (which the per-block
cross-check did NOT cover) and tells us whether the fp32 path lands on:
  - my numpy coords (1120)  -> the golden's 1126 is purely its bf16 torso; my impl correct.
  - the golden coords (1126) -> my numpy pipeline has a real bug.

CPU only, fp32. Run:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 \
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python torch_stage1_ref.py
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
import ss_vae_decode  # reuse the validated numpy decoder for coords


def build_model():
    from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
    SparseStructureFlowModel.initialize_weights = lambda self: None  # skip 1.3B xavier
    m = SparseStructureFlowModel(
        resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
        out_channels=8, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="proj", dtype="float32",
    )
    z = np.load(os.path.join(WEIGHTS_DIR, "ss_flow.npz"))
    sd = m.state_dict()
    for k in sd.keys():
        if k == "rope_phases":
            sd[k] = torch.view_as_complex(torch.from_numpy(z["rope_phases"].copy()))
        else:
            sd[k] = torch.from_numpy(z[k].astype(np.float32))
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def main():
    torch.set_grad_enabled(False)
    from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

    m = build_model()
    g = np.load(os.path.join(GOLD, "stage1_cond", "global.npy")).astype(np.float32)
    p = np.load(os.path.join(GOLD, "stage1_cond", "proj.npy")).astype(np.float32)
    cond = {"global": torch.from_numpy(g), "proj": torch.from_numpy(p)}
    neg = {"global": torch.zeros_like(cond["global"]), "proj": torch.zeros_like(cond["proj"])}

    torch.manual_seed(42)
    noise = torch.randn(1, 8, 16, 16, 16)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    print("[torch_ref] sampling (real model + real sampler, fp32, 12 steps)...", flush=True)
    out = sampler.sample(
        m, noise, cond, neg_cond=neg, steps=12, rescale_t=5.0,
        guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=(0.6, 1.0),
        verbose=True,
    )
    z_s = out.samples.numpy().astype(np.float32)

    z_s_gold = np.load(os.path.join(GOLD, "stage1_ssdec", "z_s.npy")).astype(np.float32)
    print(f"\n[torch_ref] z_s(fp32 torch) vs golden(bf16 torso): "
          f"maxabs={np.abs(z_s-z_s_gold).max():.3e} median={np.median(np.abs(z_s-z_s_gold)):.3e}")

    # decode -> coords (numpy validated decoder)
    coords = ss_vae_decode.to_coords(ss_vae_decode.decode(z_s))
    gold = np.load(os.path.join(GOLD, "stage1_out", "coords.npy")).astype(np.int32)
    cset, gset = set(map(tuple, coords.tolist())), set(map(tuple, gold.tolist()))
    inter, union = len(cset & gset), len(cset | gset)
    print(f"[torch_ref] coords fp32-torch N={coords.shape[0]} vs golden N={gold.shape[0]}  "
          f"set_equal={cset==gset}  IoU={inter/union:.4f}")
    print("\nINTERPRETATION: if fp32-torch coords match the numpy E2E (~1120) and NOT the "
          "golden 1126, the golden/numpy diff is the bf16 torso (numpy impl correct). "
          "If fp32-torch matches golden 1126, the numpy pipeline has a bug.")


if __name__ == "__main__":
    main()
