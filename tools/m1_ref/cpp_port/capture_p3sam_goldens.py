#!/usr/bin/env python3
"""
Capture P3-SAM goldens for the native ggml port.

Runs the real P3-SAM (Sonata PTv3 encoder + seg/iou heads + auto-mask postprocess)
on a fixed dense mesh, fully SEEDED and with a deterministic fp32 NON-flash oracle
(enable_flash=False, upcast_attention/softmax=True, shuffle_orders=False), and banks
every stage boundary to $OUT so the C++/ggml port can be validated cosine/mean-abs
against an fp32 reference (per project methodology: validate vs fp32 oracle, never a
flash/bf16 golden).

Banked (all .npy in OUT):
  input:   points.npy (Nin,3 normalised)  points_org.npy  normals.npy (Nin,3)
           verts.npy faces.npy  (the cleaned mesh actually fed)
  transform: tf_coord.npy (M0,3)  tf_grid_coord.npy (M0,3 int)  tf_feat.npy (M0,9)
             tf_inverse.npy (Nin,)  # GridSample inverse  points->voxels
  sonata:  ser_order_s0.npy (2,M0) ser_inverse_s0.npy (2,M0)  # stage0 serialization
           feat_s0.npy (M0,48) feat_s1.npy (M1,96) feat_s2.npy (M2,192)
           feat_s3.npy (M3,384) feat_s4.npy (M4,512)
           pool_inv_s1.npy (M0,) pool_inv_s2.npy (M1,) pool_inv_s3.npy (M2,) pool_inv_s4.npy (M3,)
           grid_s0..s4.npy  (stage grid_coord, int)
           feat_1232.npy (M0,1232)  mlp_512.npy (M0,512)  feats_N.npy (Nin,512)
  heads:   hd_prompt.npy (K,3)  hd_points.npy (Nin,3)  hd_feats.npy (Nin,512)
           hd_seg1_logits.npy (K,Nin,3)  hd_seg2_logits.npy (K,Nin,3)  hd_iou.npy (K,3)
  final:   result_mask.npy (Nin,) point->cluster-id   face_ids.npy (F,)  per-face label
           color_map.npy (ids,3)
"""
import os, sys, json, time, random
import numpy as np
import torch
import trimesh

sys.path.append("..")                       # P3-SAM/  (build_P3SAM, load_state_dict)
sys.path.append("/work/XPart/partgen")      # sonata

MESH   = os.environ.get("P3SAM_MESH", "/work/native_v6_defaults.glb")
CKPT   = os.environ.get("P3SAM_CKPT", "/work/weights/p3sam/p3sam.safetensors")
SONATA = os.environ.get("P3SAM_SONATA", "/root/sonata/facebook/sonata/sonata.pth")
OUT    = os.environ.get("P3SAM_OUT", "/out")
SEED   = 42
POINT_NUM  = 100000
PROMPT_NUM = 400
os.makedirs(OUT, exist_ok=True)

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sv(name, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy() if arr.dtype.is_floating_point else arr.detach().cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    print(f"  saved {name:20s} {np.asarray(arr).shape} {np.asarray(arr).dtype}", flush=True)

# ----------------------------------------------------------------------------
# Build model, force the deterministic fp32 oracle.
# ----------------------------------------------------------------------------
from model import build_P3SAM, load_state_dict
from models import sonata
from models.sonata.model import SerializedAttention, GridPooling, PointTransformerV3

class P3SAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        build_P3SAM(self)

# build_P3SAM calls sonata.load(...) which fetches from HF; monkeypatch to local fp32 oracle
_orig_load = sonata.load
def _patched_load(name="sonata", repo_id=None, download_root=None, custom_config=None, ckpt_only=False):
    cfg_override = {
        "enable_flash": False,          # fp32 path
        "upcast_attention": True,
        "upcast_softmax": True,
        "shuffle_orders": False,        # deterministic serialization order
        "drop_path": 0.0,
    }
    if custom_config: cfg_override.update(custom_config)
    return _orig_load(SONATA, repo_id=repo_id, download_root=download_root,
                      custom_config=cfg_override, ckpt_only=ckpt_only)
sonata.load = _patched_load

print("building P3-SAM (fp32 non-flash oracle) ...", flush=True)
seed_all()
model = P3SAM()
load_state_dict(model, ckpt_path=CKPT)
model.eval().cuda()
model.float()

# defensive: ensure every attention block is in deterministic non-flash fp32 mode
for m in model.modules():
    if isinstance(m, SerializedAttention):
        m.enable_flash = False
        m.upcast_attention = True
        m.upcast_softmax = True
        if not hasattr(m, "patch_size_max"):
            m.patch_size_max = 1024
        m.patch_size = 0
        if not isinstance(m.attn_drop, torch.nn.Module):
            m.attn_drop = torch.nn.Dropout(0.0)
    if isinstance(m, GridPooling):
        m.shuffle_orders = False
model.sonata.shuffle_orders = False

# ----------------------------------------------------------------------------
# Sample points (seeded) + normals.
# ----------------------------------------------------------------------------
from demo.auto_mask_no_postprocess import normalize_pc, clean_mesh  # noqa
print("loading + sampling mesh:", MESH, flush=True)
mesh = trimesh.load(MESH, force="mesh", process=False)
mesh = clean_mesh(mesh)
mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
print(f"  mesh V={len(mesh.vertices)} F={len(mesh.faces)}", flush=True)
sv("verts", np.asarray(mesh.vertices, np.float32))
sv("faces", np.asarray(mesh.faces, np.int64))

seed_all()
_points, face_idx = trimesh.sample.sample_surface(mesh, POINT_NUM, seed=SEED)
_points_org = _points.copy().astype(np.float32)
_points = normalize_pc(_points).astype(np.float32)
normals = np.asarray(mesh.face_normals[face_idx], np.float32)
sv("points", _points); sv("points_org", _points_org); sv("normals", normals)

# ----------------------------------------------------------------------------
# Sonata forward, hooking each encoder stage to bank intermediates.
# ----------------------------------------------------------------------------
stage_cap = {}
def mk_hook(s):
    def hook(mod, inp, out):
        # out is a Point
        stage_cap[s] = {
            "feat": out.feat.detach().cpu().float().numpy(),
            "grid_coord": out.grid_coord.detach().cpu().numpy().astype(np.int32),
        }
        if "pooling_inverse" in out.keys():
            stage_cap[s]["pool_inv"] = out.pooling_inverse.detach().cpu().numpy().astype(np.int64)
        if s == 0 and "serialized_order" in out.keys():
            stage_cap[s]["ser_order"] = out.serialized_order.detach().cpu().numpy().astype(np.int64)
            stage_cap[s]["ser_inverse"] = out.serialized_inverse.detach().cpu().numpy().astype(np.int64)
    return hook

hooks = []
for s in range(5):
    hooks.append(model.sonata.enc._modules[f"enc{s}"].register_forward_hook(mk_hook(s)))

@torch.no_grad()
def get_feat_capture(points, normals):
    data_dict = {
        "coord": points, "normal": normals,
        "color": np.ones_like(points),
        "batch": np.zeros(points.shape[0], dtype=np.int64),
    }
    data_dict = model.transform(data_dict)
    # bank the transform output (post GridSample) BEFORE moving to cuda
    sv("tf_coord", data_dict["coord"])
    sv("tf_grid_coord", torch.as_tensor(data_dict["grid_coord"]).int())
    sv("tf_feat", data_dict["feat"])
    sv("tf_inverse", torch.as_tensor(data_dict["inverse"]).long())
    for k in data_dict:
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].cuda()
    point = model.sonata(data_dict)
    # unpool concat (matches auto_mask_no_postprocess.get_feat)
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    feat_1232 = point.feat
    sv("feat_1232", feat_1232)
    sv("tf_inverse_final", point.inverse.detach().cpu().numpy().astype(np.int64))
    feat = model.mlp(feat_1232)
    sv("mlp_512", feat)
    feat = feat[point.inverse]
    sv("feats_N", feat)
    return feat

seed_all()
t0 = time.time()
feats = get_feat_capture(_points, normals)
print(f"sonata+mlp forward {time.time()-t0:.1f}s feats {tuple(feats.shape)}", flush=True)
for s in range(5):
    c = stage_cap[s]
    sv(f"feat_s{s}", c["feat"]); sv(f"grid_s{s}", c["grid_coord"])
    if "pool_inv" in c: sv(f"pool_inv_s{s}", c["pool_inv"])
    if "ser_order" in c: sv("ser_order_s0", c["ser_order"]); sv("ser_inverse_s0", c["ser_inverse"])
for h in hooks: h.remove()

# ----------------------------------------------------------------------------
# Heads: run the forward on first 32 FPS prompts, bank logits/iou.
# ----------------------------------------------------------------------------
import fpsample
seed_all()
fps_idx = fpsample.fps_sampling(_points, PROMPT_NUM)
prompts = _points[fps_idx]            # (400,3)
K = 32
prompt_b = prompts[:K]
sv("hd_prompt", prompt_b.astype(np.float32))
sv("hd_points", _points)
sv("hd_feats", feats)

@torch.no_grad()
def heads_capture(feats, points, point_prompt, iters=1):
    N = points.shape[0]; Kp = point_prompt.shape[0]
    f = feats.unsqueeze(1).repeat(1, Kp, 1).cuda()                       # [N,K,512]
    p = torch.from_numpy(points).float().cuda().unsqueeze(1).repeat(1, Kp, 1)
    pc = torch.from_numpy(point_prompt).float().cuda().unsqueeze(0).repeat(N, 1, 1)
    f = f.transpose(0, 1); p = p.transpose(0, 1); pc = pc.transpose(0, 1)  # [K,N,*]
    # replicate P3SAM.forward, banking logits
    f = f.transpose(0, 1); p = p.transpose(0, 1); pc = pc.transpose(0, 1)  # [N,K,*]
    feats_seg = torch.cat([f, p, pc], dim=-1)
    m1 = model.seg_mlp_1(feats_seg).squeeze(-1)
    m2 = model.seg_mlp_2(feats_seg).squeeze(-1)
    m3 = model.seg_mlp_3(feats_seg).squeeze(-1)
    pred_mask = torch.stack([m1, m2, m3], dim=-1)            # [N,K,3] stage1 logits
    seg1 = pred_mask.permute(1, 0, 2).contiguous()          # [K,N,3]
    for _ in range(iters):
        feats_seg_2 = torch.cat([feats_seg, pred_mask], dim=-1)
        g = model.seg_s2_mlp_g(feats_seg_2)
        g = torch.max(g, dim=0).values
        g = g.unsqueeze(0).repeat(N, 1, 1)
        feats_seg_3 = torch.cat([g, feats_seg_2], dim=-1)
        s1 = model.seg_s2_mlp_1(feats_seg_3).squeeze(-1)
        s2 = model.seg_s2_mlp_2(feats_seg_3).squeeze(-1)
        s3 = model.seg_s2_mlp_3(feats_seg_3).squeeze(-1)
        pred_mask = torch.stack([s1, s2, s3], dim=-1)
    seg2 = pred_mask.permute(1, 0, 2).contiguous()          # [K,N,3] stage2 logits
    feats_iou = torch.cat([g, feats_seg, pred_mask], dim=-1)
    feats_iou = model.iou_mlp(feats_iou)
    feats_iou = torch.max(feats_iou, dim=0).values
    pred_iou = torch.sigmoid(model.iou_mlp_out(feats_iou))  # [K,3]
    return seg1, seg2, pred_iou

seed_all()
seg1, seg2, iou = heads_capture(feats, _points, prompt_b)
sv("hd_seg1_logits", seg1); sv("hd_seg2_logits", seg2); sv("hd_iou", iou)
print("heads captured", flush=True)

# ----------------------------------------------------------------------------
# Full auto-mask postprocess -> result_mask + face_ids (the production output).
# ----------------------------------------------------------------------------
from demo.auto_mask_no_postprocess import mesh_sam
seed_all()
model_parallel = torch.nn.DataParallel(model)
t0 = time.time()
aabb, face_ids, _ = mesh_sam((model, model_parallel), mesh,
                             save_path=OUT, point_num=POINT_NUM, prompt_num=PROMPT_NUM,
                             save_mid_res=False, show_info=True, seed=SEED,
                             clean_mesh_flag=False)
print(f"mesh_sam {time.time()-t0:.1f}s  n_parts={len(np.unique(face_ids))}", flush=True)
sv("face_ids", np.asarray(face_ids, np.int64))
print("DONE. goldens in", OUT, flush=True)
