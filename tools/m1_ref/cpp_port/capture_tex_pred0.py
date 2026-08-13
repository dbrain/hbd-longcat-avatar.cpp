#!/usr/bin/env python3
# Cheap single-forward (step-0) golden for the cross-mode tex DiT, from BANKED goldens (no voxelizer /
# dinov3 / 12-step sampling). Loads only the tex flow model, forces it to cuda, runs ONE forward with
# the banked noise + shape_norm + cond -> tex_pred0_feats.npy. Isolates the DiT math for C++ parity.
import os, sys
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0"); os.environ.setdefault("ATTN_BACKEND", "sdpa")
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
import numpy as np, torch
from pixal3d.modules import sparse as sp
from pixal3d.pipelines import Trellis2TexturingPipeline

G = "/mnt/hdd/3d/avatar-shootout/tex_goldens"
RT, STEPS = 3.0, 12
pipe = Trellis2TexturingPipeline.from_pretrained("/mnt/hdd/pixal3d_tex/trellis2_4b", "_texturing_pipeline_local.json")
tex_model = pipe.models['tex_slat_flow_model_1024'].to("cuda").eval()

coords = torch.from_numpy(np.load(f"{G}/shape_slat_coords.npy")).int().cuda()      # [N,4]
noise  = torch.from_numpy(np.load(f"{G}/tex_noise.npy")).float().cuda()            # [N,32]
snorm  = torch.from_numpy(np.load(f"{G}/shape_norm_feats.npy")).float().cuda()     # [N,32]
cond   = torch.from_numpy(np.load(f"{G}/cond_cond.npy")).float().cuda()            # [1,Ntok,1024]

x_st     = sp.SparseTensor(feats=noise, coords=coords)
snorm_st = sp.SparseTensor(feats=snorm, coords=coords)
t_seq = np.linspace(1, 0, STEPS + 1); t_seq = RT * t_seq / (1 + (RT - 1) * t_seq)
tt = torch.tensor([1000.0 * float(t_seq[0])], device="cuda", dtype=torch.float32)
with torch.no_grad():
    pred0 = tex_model(x_st, tt, cond, concat_cond=snorm_st)
out = pred0.feats.detach().float().cpu().numpy()
np.save(f"{G}/tex_pred0_feats.npy", np.ascontiguousarray(out))
print(f"[pred0] t0={float(t_seq[0]):.6f} shape {out.shape} mean {out.mean():.5f} std {out.std():.5f} -> {G}/tex_pred0_feats.npy", flush=True)
