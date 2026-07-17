#!/usr/bin/env python3
"""Capture the production auto-mask face_ids (fp32 oracle) using the REAL P3SAM
class (has forward), for end-to-end partition validation of the native port."""
import os, sys, time, random
import numpy as np, torch, trimesh
sys.path.append("..")
sys.path.append("/work/XPart/partgen")
MESH = os.environ.get("P3SAM_MESH", "/cpp/native_v6_defaults.glb")
CKPT = os.environ.get("P3SAM_CKPT", "/work/P3-SAM/weights/p3sam/p3sam.safetensors")
SONATA = os.environ.get("P3SAM_SONATA", "/root/sonata/facebook/sonata/sonata.pth")
OUT = os.environ.get("P3SAM_OUT", "/out")
SEED = 42
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
from models import sonata
from models.sonata.model import SerializedAttention, GridPooling
_o = sonata.load
def _pl(name="sonata", repo_id=None, download_root=None, custom_config=None, ckpt_only=False):
    cfg={"enable_flash":False,"upcast_attention":True,"upcast_softmax":True,"shuffle_orders":False,"drop_path":0.0}
    if custom_config: cfg.update(custom_config)
    return _o(SONATA, repo_id=repo_id, download_root=download_root, custom_config=cfg, ckpt_only=ckpt_only)
sonata.load = _pl
from demo.auto_mask_no_postprocess import P3SAM, mesh_sam, clean_mesh
seed_all()
model = P3SAM(); model.load_state_dict(CKPT); model.eval().cuda().float()
for m in model.modules():
    if isinstance(m, SerializedAttention):
        m.enable_flash=False; m.upcast_attention=True; m.upcast_softmax=True
        if not hasattr(m,"patch_size_max"): m.patch_size_max=1024
        m.patch_size=0
        if not isinstance(m.attn_drop,torch.nn.Module): m.attn_drop=torch.nn.Dropout(0.0)
    if isinstance(m, GridPooling): m.shuffle_orders=False
model.sonata.shuffle_orders=False
mesh = trimesh.load(MESH, force="mesh", process=False)
mesh = clean_mesh(mesh)
mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
seed_all()
# single GPU: pass model directly as the "parallel" model (DataParallel just copies+OOMs).
# Also wrap get_mask path with empty_cache via a tiny monkeypatch on the module fn.
import demo.auto_mask_no_postprocess as AM
_get_mask = AM.get_mask
def get_mask_freed(*a, **k):
    r = _get_mask(*a, **k); torch.cuda.empty_cache(); return r
AM.get_mask = get_mask_freed
t0=time.time()
aabb, face_ids, _ = mesh_sam((model, model), mesh, save_path=OUT, point_num=100000,
                             prompt_num=400, save_mid_res=False, show_info=True,
                             seed=SEED, clean_mesh_flag=False, prompt_bs=4)
print(f"mesh_sam {time.time()-t0:.1f}s parts={len(np.unique(face_ids))}",flush=True)
np.save(os.path.join(OUT,"face_ids.npy"), np.asarray(face_ids,np.int64))
print("DONE faces", np.unique(face_ids).tolist()[:20], flush=True)
