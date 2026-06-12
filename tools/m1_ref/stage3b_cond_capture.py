#!/usr/bin/env python3
"""
M5 stage3b cond capture: shape_1024 DinoV3ProjFeatureExtractor pieces (DINOv3@1024 +
NAF@1024 + proj_grid64) for the C++ port to validate against. Mirrors get_proj_cond_shape
for shape_1024 (image_size 1024, grid64, naf_target 512).

Dumps to cpp_port/refs/stage3b/:
  image_1024_chw  [3,1024,1024]  (resized [0,1], the DINOv3-pre-normalize / NAF guide)
  patchmap_1024   [1,64,64,1024] (z_patchtokens_spatial)
  naf_hr_1024     [1,1024,512,512]
  global_1024     [1,5,1024]
  proj_lr_1024 / proj_hr_1024 at the golden stage3b coords (for branch-split validation)
Validates the gathered proj == golden stage3b_cond/proj_feats.

Run (GPU; DINOv3@1024 + NAF need CUDA):
  /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python stage3b_cond_capture.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROUT = os.path.join(HERE, "cpp_port", "refs", "stage3b")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
IMG = "/mnt/hdd/3d/avatar-shootout/assets/miku.png"
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
os.environ.setdefault("ATTN_BACKEND", "sdpa")


def main():
    os.makedirs(ROUT, exist_ok=True)
    import torch, torch.nn.functional as F
    from PIL import Image
    torch.set_grad_enabled(False)
    from inference import IMAGE_COND_CONFIGS, build_image_cond_model

    cam = json.load(open(os.path.join(GOLD, "cam.json")))
    print("[s3b] cam.json:", cam, flush=True)
    ax = float(cam.get("camera_angle_x", cam.get("cam", 0.7332379387484828)))
    dist = float(cam.get("distance", cam.get("dist", 1.3021559715270996)))
    ms = float(cam.get("mesh_scale", 1.0))

    model = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"]).cuda()
    print("[s3b] shape_1024 model: image_size", model.image_size, "grid", model.grid_resolution,
          "naf_target", model.naf_target_size, flush=True)

    # preprocess exactly like the model forward (PIL list path)
    pil = Image.open(IMG).convert("RGB")
    # NOTE: pipeline preprocess_image (rembg+crop) runs BEFORE this; the golden used the
    # preprocessed image. Use the golden preprocessed image if present, else raw.
    pre_img = os.path.join(GOLD, "pre", "preprocessed.png")  # the exact golden cond input
    print("[s3b] pre image:", pre_img, flush=True)
    pil = Image.open(pre_img).convert("RGB")

    sz = model.image_size  # 1024
    pil = pil.resize((sz, sz), Image.LANCZOS)
    img01 = torch.from_numpy(np.array(pil).astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()  # [1,3,1024,1024]
    np.save(os.path.join(ROUT, "image_1024_chw.npy"), img01[0].cpu().numpy().astype(np.float32))

    image_for_naf = img01.clone()
    image_norm = model.transform(img01)

    z = model.extract_features(image_norm)               # [1, seq, 1024]
    num_reg = getattr(model.model.config, "num_register_tokens", 4)
    z_cls = z[:, 0:1]; z_reg = z[:, 1:1+num_reg]; z_patch = z[:, 1+num_reg:]
    pn = model.patch_number  # 64
    patchmap = z_patch.reshape(1, pn, pn, -1)            # [1,64,64,1024]
    z_global = torch.cat([z_cls, z_reg], dim=1)          # [1,5,1024]
    np.save(os.path.join(ROUT, "patchmap_1024.npy"), patchmap.cpu().numpy().astype(np.float32))
    np.save(os.path.join(ROUT, "global_1024.npy"), z_global.cpu().numpy().astype(np.float32))
    print(f"[s3b] patchmap {tuple(patchmap.shape)} global {tuple(z_global.shape)}", flush=True)

    camA = torch.tensor([ax]).cuda(); distT = torch.tensor([dist]).cuda(); msT = torch.tensor([ms]).cuda()
    proj_lr = model.proj_grid(patchmap, camA, distT, msT, None)   # [1, grid^3, 1024]
    model._load_naf()
    lr_bchw = patchmap.permute(0, 3, 1, 2)               # [1,1024,64,64]
    hr = model.naf_model(image_for_naf, lr_bchw, model.naf_target_size)  # [1,1024,512,512]
    np.save(os.path.join(ROUT, "naf_hr_1024.npy"), hr.cpu().numpy().astype(np.float32))
    proj_hr = model.proj_grid(hr, camA, distT, msT, None, BHWC=False)
    z_proj = torch.cat([proj_lr, proj_hr], dim=-1)        # [1, grid^3, 2048]
    print(f"[s3b] naf_hr {tuple(hr.shape)} z_proj {tuple(z_proj.shape)}", flush=True)

    # gather at golden stage3b coords
    coords = np.load(os.path.join(GOLD, "stage3b_cond", "proj_coords.npy")).astype(np.int64)
    gr = model.grid_resolution
    zg = z_proj.reshape(1, gr, gr, gr, -1)
    ct = torch.from_numpy(coords).cuda()
    proj_sparse = zg[ct[:, 0], ct[:, 1], ct[:, 2], ct[:, 3]].cpu().numpy().astype(np.float32)  # [N,2048]
    # split branches at the golden coords for branch-localised validation
    np.save(os.path.join(ROUT, "proj_lr_at_coords.npy"), proj_sparse[:, :1024])
    np.save(os.path.join(ROUT, "proj_hr_at_coords.npy"), proj_sparse[:, 1024:])

    gold = np.load(os.path.join(GOLD, "stage3b_cond", "proj_feats.npy")).astype(np.float32)
    d = np.abs(proj_sparse - gold)
    a = proj_sparse.reshape(-1); b = gold.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    dlr = np.abs(proj_sparse[:, :1024] - gold[:, :1024]).max()
    dhr = np.abs(proj_sparse[:, 1024:] - gold[:, 1024:]).max()
    print(f"\n[s3b] z_proj@coords vs golden proj_feats[{coords.shape[0]},2048]: maxabs={d.max():.3e} "
          f"meanabs={d.mean():.3e} cosine={cos:.6f}  (lr-maxabs={dlr:.3e} hr-maxabs={dhr:.3e})", flush=True)
    print("[s3b] DONE", flush=True)


if __name__ == "__main__":
    main()
