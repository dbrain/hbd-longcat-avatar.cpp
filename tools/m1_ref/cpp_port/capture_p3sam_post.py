#!/usr/bin/env python3
"""Capture P3-SAM heads (on a memory-safe point subset) + the full auto-mask
face_ids (production output), reusing the banked fp32 backbone feats.  Splits
out of capture_p3sam_goldens.py so the 100k x 32 head tensor never OOMs."""
import os, sys, json, time, random
import numpy as np, torch, trimesh
sys.path.append("..")
sys.path.append("/work/XPart/partgen")

MESH   = os.environ.get("P3SAM_MESH", "/cpp/native_v6_defaults.glb")
CKPT   = os.environ.get("P3SAM_CKPT", "/work/P3-SAM/weights/p3sam/p3sam.safetensors")
SONATA = os.environ.get("P3SAM_SONATA", "/root/sonata/facebook/sonata/sonata.pth")
OUT    = os.environ.get("P3SAM_OUT", "/out")
SEED, POINT_NUM, PROMPT_NUM = 42, 100000, 400
NSUB, KSUB = 20000, 8     # head-golden subset

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def sv(name, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy() if arr.dtype.is_floating_point else arr.detach().cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    print(f"  saved {name:20s} {np.asarray(arr).shape} {np.asarray(arr).dtype}", flush=True)

from model import build_P3SAM, load_state_dict
from models import sonata
from models.sonata.model import SerializedAttention, GridPooling
class P3SAM(torch.nn.Module):
    def __init__(self): super().__init__(); build_P3SAM(self)
_orig=sonata.load
def _pl(name="sonata", repo_id=None, download_root=None, custom_config=None, ckpt_only=False):
    cfg={"enable_flash":False,"upcast_attention":True,"upcast_softmax":True,"shuffle_orders":False,"drop_path":0.0}
    if custom_config: cfg.update(custom_config)
    return _orig(SONATA, repo_id=repo_id, download_root=download_root, custom_config=cfg, ckpt_only=ckpt_only)
sonata.load=_pl
seed_all(); model=P3SAM(); load_state_dict(model,ckpt_path=CKPT); model.eval().cuda().float()
for m in model.modules():
    if isinstance(m, SerializedAttention):
        m.enable_flash=False; m.upcast_attention=True; m.upcast_softmax=True
        if not hasattr(m,"patch_size_max"): m.patch_size_max=1024
        m.patch_size=0
        if not isinstance(m.attn_drop,torch.nn.Module): m.attn_drop=torch.nn.Dropout(0.0)
    if isinstance(m, GridPooling): m.shuffle_orders=False
model.sonata.shuffle_orders=False

# ---- heads golden on a banked-feats subset --------------------------------
feats_N = np.load(os.path.join(OUT,"feats_N.npy"))     # (100000,512)
points  = np.load(os.path.join(OUT,"points.npy"))       # (100000,3)
import fpsample
seed_all(); fps_idx = fpsample.fps_sampling(points, PROMPT_NUM)
prompts = points[fps_idx]
sub = np.arange(NSUB)
f_sub = torch.from_numpy(feats_N[sub]).cuda()
p_sub = points[sub].astype(np.float32)
pr = prompts[:KSUB].astype(np.float32)
sv("hd_feats_sub", feats_N[sub]); sv("hd_points_sub", p_sub); sv("hd_prompt_sub", pr)

@torch.no_grad()
def heads(feats, points, prompt):
    N=points.shape[0]; K=prompt.shape[0]
    f=feats.unsqueeze(1).repeat(1,K,1)
    p=torch.from_numpy(points).float().cuda().unsqueeze(1).repeat(1,K,1)
    pc=torch.from_numpy(prompt).float().cuda().unsqueeze(0).repeat(N,1,1)
    fs=torch.cat([f,p,pc],dim=-1)            # [N,K,518]
    m1=model.seg_mlp_1(fs).squeeze(-1); m2=model.seg_mlp_2(fs).squeeze(-1); m3=model.seg_mlp_3(fs).squeeze(-1)
    pm=torch.stack([m1,m2,m3],dim=-1)        # [N,K,3] stage1 logits
    seg1=pm.permute(1,0,2).contiguous()
    fs2=torch.cat([fs,pm],dim=-1)
    g=model.seg_s2_mlp_g(fs2); g=torch.max(g,dim=0).values; g=g.unsqueeze(0).repeat(N,1,1)
    fs3=torch.cat([g,fs2],dim=-1)
    s1=model.seg_s2_mlp_1(fs3).squeeze(-1); s2=model.seg_s2_mlp_2(fs3).squeeze(-1); s3=model.seg_s2_mlp_3(fs3).squeeze(-1)
    pm2=torch.stack([s1,s2,s3],dim=-1)
    seg2=pm2.permute(1,0,2).contiguous()
    fi=torch.cat([g,fs,pm2],dim=-1); fi=model.iou_mlp(fi); fi=torch.max(fi,dim=0).values
    iou=torch.sigmoid(model.iou_mlp_out(fi))
    return seg1,seg2,iou
seed_all(); seg1,seg2,iou=heads(f_sub,p_sub,pr)
sv("hd_seg1_logits",seg1); sv("hd_seg2_logits",seg2); sv("hd_iou",iou)
print("heads captured",flush=True)
del f_sub; torch.cuda.empty_cache()

# ---- full auto-mask -> face_ids (production output) -----------------------
from demo.auto_mask_no_postprocess import mesh_sam, clean_mesh
mesh = trimesh.load(MESH, force="mesh", process=False)
mesh = clean_mesh(mesh)
mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
seed_all()
mp = torch.nn.DataParallel(model)
t0=time.time()
aabb, face_ids, _ = mesh_sam((model, mp), mesh, save_path=OUT, point_num=POINT_NUM,
                             prompt_num=PROMPT_NUM, save_mid_res=False, show_info=True,
                             seed=SEED, clean_mesh_flag=False, prompt_bs=8)
print(f"mesh_sam {time.time()-t0:.1f}s parts={len(np.unique(face_ids))}",flush=True)
sv("face_ids", np.asarray(face_ids,np.int64))
print("DONE post goldens",flush=True)
