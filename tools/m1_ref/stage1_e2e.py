#!/usr/bin/env python3
"""
M1 Stage-1 END-TO-END (numpy fp32, CPU): preprocessed image -> occupancy coords,
chaining the three validated reference modules with NO golden tensors as inputs
(only the preprocessed image + camera scalars). Validates the final coords against
the golden as a set — the "functionally complete Stage-1" milestone.

  image -> [dinov3_proj] (z_global, z_proj)
        -> [ss_dit + FlowEuler sampler] z_s          (noise = seed-42, like the pipeline)
        -> [ss_vae_decode] ss_logits -> coords

Also reports per-boundary error vs the goldens (z_global/z_proj/z_s) as diagnostics, so
if the final coords diverge we can see WHERE (cond mismatch vs noise-repro vs decode).

Run: CUDA_VISIBLE_DEVICES="" /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python stage1_e2e.py
"""
import os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"

import dinov3_proj
import ss_dit
import ss_vae_decode


def _t(msg, t0):
    print(f"  [{time.time()-t0:6.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    cam = json.load(open(os.path.join(GOLD, "cam.json")))
    img = os.path.join(GOLD, "pre", "preprocessed.png")
    print(f"[stage1_e2e] image={img}  cam={cam}", flush=True)

    # --- Stage 1a: DINOv3 + proj conditioning ---
    z_global, z_proj, _ = dinov3_proj.extract(img, cam)
    _t(f"dinov3+proj -> global{z_global.shape} proj{z_proj.shape}", t0)
    # diagnostic vs golden cond
    g_gold = np.load(os.path.join(GOLD, "stage1_cond", "global.npy"))
    p_gold = np.load(os.path.join(GOLD, "stage1_cond", "proj.npy"))
    print(f"    cond vs golden: global maxabs={np.abs(z_global-g_gold).max():.2e}  "
          f"proj maxabs={np.abs(z_proj-p_gold).max():.2e}", flush=True)

    cond = {"global": z_global, "proj": z_proj}
    neg = {"global": np.zeros_like(z_global), "proj": np.zeros_like(z_proj)}

    # --- noise: reproduce the pipeline's seed-42 CPU noise ---
    import torch
    torch.manual_seed(42)
    noise = torch.randn(1, 8, 16, 16, 16).numpy().astype(np.float32)

    # --- Stage 1b: SS DiT + FlowEuler sampler (12 steps, CFG+interval+rescale) ---
    print("[stage1_e2e] running 12-step sampler (naive numpy, ~minutes)...", flush=True)
    z_s = ss_dit.sample(noise, cond, neg, verbose=True)
    _t(f"ss_dit -> z_s{z_s.shape}", t0)
    z_s_gold = np.load(os.path.join(GOLD, "stage1_ssdec", "z_s.npy"))
    print(f"    z_s vs golden: maxabs={np.abs(z_s-z_s_gold).max():.2e}  "
          f"median={np.median(np.abs(z_s-z_s_gold)):.2e}", flush=True)

    # --- Stage 1c: SS VAE decode -> occupancy coords ---
    ss_logits = ss_vae_decode.decode(z_s)
    coords = ss_vae_decode.to_coords(ss_logits)
    _t(f"ss_vae_decode -> coords{coords.shape}", t0)

    # --- validate coords vs golden (the milestone signal) ---
    coords_gold = np.load(os.path.join(GOLD, "stage1_out", "coords.npy")).astype(np.int32)
    my_set = set(map(tuple, coords.tolist()))
    gold_set = set(map(tuple, coords_gold.tolist()))
    inter = len(my_set & gold_set)
    union = len(my_set | gold_set)
    iou = inter / union if union else 0.0
    same_n = coords.shape[0] == coords_gold.shape[0]
    set_eq = my_set == gold_set
    print(f"\n[stage1_e2e] COORDS  mine N={coords.shape[0]}  golden N={coords_gold.shape[0]}")
    print(f"    same_N={same_n}  set_equal={set_eq}  IoU={iou:.4f}  "
          f"(inter={inter} union={union})")
    print(f"\nRESULT: {'PASS — image->coords reproduces golden' if set_eq else 'SEE ABOVE'}")
    _t("done", t0)


if __name__ == "__main__":
    main()
