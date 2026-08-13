#!/usr/bin/env python3
"""
NAF golden capture (Stage-2 conditioner, the heavy net-new piece).

NAF (valeoai/NAF) guide-conditioned feature upsampler: takes the DINOv3 LR patch
feature map [1,1024,32,32] + an RGB guide [1,3,512,512] -> HR feature map
[1,1024,512,512] via 9x9 dilated neighborhood (local) cross-attention (natten na2d).

KEY: NAF's lr_features == the ALREADY-VALIDATED stage-1 DINOv3 patchmap (same
dinov3-vitl16 model, same 512px image), so we capture the NAF golden standalone
WITHOUT re-running the pipeline. The guide is the 512px LANCZOS resize of the
preprocessed image / 255 (exactly what DinoV3ProjFeatureExtractor passes as
image_for_naf, before ImageNet normalization).

Captures (refs/):
  naf_guide.npy        [1,3,512,512]   guide image, [0,1]
  naf_lr_features.npy  [1,1024,32,32]  DINOv3 lr features (== dino_patchmap.permute)
  naf_hr_golden.npy    [1,1024,512,512] NAF output (1GB, gitignored)
  naf_ie_out.npy       [1,256,512,512] ImageEncoder output (post-RoPE) for debugging
  naf_k_resized.npy    [1,512,512,4,64] / naf_v_resized / naf_q  (na2d inputs) for debug
weights/ naf.npz + cpp_port refs naf_keys.json (state_dict for the ggml port).

GPU run (natten na2d needs CUDA). Run AFTER any CPU oracle finishes:
  /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python naf_capture.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
REFS = os.path.join(HERE, "cpp_port", "refs")
WEIGHTS = os.path.join(HERE, "weights")
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")

import torch
import torch.nn.functional as F
from PIL import Image


def main():
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[naf] device={dev}", flush=True)

    # ---- guide image: 512px LANCZOS resize of preprocessed, /255, CHW (image_for_naf) ----
    pre = Image.open(os.path.join(GOLD, "pre", "preprocessed.png")).convert("RGB")
    pre512 = pre.resize((512, 512), Image.LANCZOS)
    guide = np.array(pre512).astype(np.float32) / 255.0          # [512,512,3]
    guide = torch.from_numpy(guide).permute(2, 0, 1)[None].to(dev)  # [1,3,512,512]
    np.save(os.path.join(REFS, "naf_guide.npy"), guide.cpu().numpy())

    # ---- lr_features: validated dino_patchmap [1,32,32,1024] -> [1,1024,32,32] ----
    patchmap = np.load(os.path.join(REFS, "dino_patchmap.npy")).astype(np.float32)  # [1,32,32,1024]
    lr_features = torch.from_numpy(patchmap).permute(0, 3, 1, 2).contiguous().to(dev)  # [1,1024,32,32]
    np.save(os.path.join(REFS, "naf_lr_features.npy"), lr_features.cpu().numpy())
    print(f"[naf] guide{tuple(guide.shape)} lr_features{tuple(lr_features.shape)}", flush=True)

    # ---- load NAF (torch.hub, pretrained) ----
    naf = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device=dev, trust_repo=True)
    naf.eval().requires_grad_(False)
    naf.to(dev)

    # dump state_dict for the ggml port
    sd = naf.state_dict()
    npz = {k: v.detach().cpu().float().numpy() for k, v in sd.items()}
    os.makedirs(WEIGHTS, exist_ok=True)
    np.savez(os.path.join(WEIGHTS, "naf.npz"), **npz)
    keys = {k: list(v.shape) for k, v in npz.items()}
    json.dump(keys, open(os.path.join(REFS, "naf_keys.json"), "w"), indent=1)
    print(f"[naf] saved {len(keys)} weights -> naf.npz; keys:")
    for k, s in keys.items():
        print(f"        {k}: {s}")

    # ---- intermediates: hook ImageEncoder + CrossAttention ----
    inter = {}
    ie = naf.image_encoder
    _ie_fwd = ie.forward
    def ie_fwd(x, output_size):
        out = _ie_fwd(x, output_size)
        inter["ie_out"] = out.detach().cpu().numpy()
        return out
    ie.forward = ie_fwd

    up = naf.upsampler
    _up_fwd = up.forward
    def up_fwd(q, k, v, image=None, return_weights=False, **kw):
        # replicate the internal resize to dump na2d inputs
        hq, wq = q.shape[-2:]; hk, wk = k.shape[-2:]
        dil = (hq // hk, wq // wk)
        qr = q.reshape(q.shape[0], up.num_heads, -1, hq, wq)  # debug only
        inter["q_pre"] = q.detach().cpu().numpy()
        inter["k_pre"] = k.detach().cpu().numpy()
        inter["v_pre"] = v.detach().cpu().numpy()
        inter["dilation"] = list(dil)
        out = _up_fwd(q, k, v, image=image, return_weights=return_weights, **kw)
        return out
    up.forward = up_fwd

    target = (512, 512)
    print(f"[naf] running NAF -> target {target} ...", flush=True)
    hr = naf(guide, lr_features, target)   # [1,1024,512,512]
    hr_np = hr.detach().cpu().float().numpy()
    np.save(os.path.join(REFS, "naf_hr_golden.npy"), hr_np)
    print(f"[naf] hr_features {hr_np.shape} range [{hr_np.min():.4f},{hr_np.max():.4f}] "
          f"mean {hr_np.mean():.4f}", flush=True)

    for k, v in inter.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(REFS, f"naf_{k}.npy"), v)
            print(f"[naf] inter naf_{k} {v.shape}")
        else:
            print(f"[naf] inter {k} = {v}")

    # ---- also validate against golden proj_feats[:,1024:] via proj-gather (sanity) ----
    # (the DiT actually consumes proj_grid(hr) gathered at the 1126 voxel coords)
    print("[naf] (proj-gather cross-check deferred to the C++/numpy port; hr golden saved)")


if __name__ == "__main__":
    main()
