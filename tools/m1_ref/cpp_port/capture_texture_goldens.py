#!/usr/bin/env python3
# Texture ORACLE + intermediate goldens for the native TRELLIS-2 texturing wiring (Stage 2).
# Runs Trellis2TexturingPipeline stage-by-stage on (mesh, image), banking each boundary so the C++
# native wiring can validate per-substage (methodology: fp32-ish oracle, cosine 0.999+):
#   preprocessed image, shape_slat (feats/coords), tex_slat (feats/coords), pbr_voxel (feats/coords),
#   + the final textured GLB (the bar to beat) and a UV base-color PNG dump.
# Input mesh = miku_lowpoly.glb (V=39141 F=69404 = the golden_69k / golden_voxel source mesh), so the
# banked shape_slat is directly comparable to /mnt/hdd/pixal3d_tex/golden_69k_1024.
#   ATTN_BACKEND=sdpa <Pixal3D venv>/python capture_texture_goldens.py [mesh.glb] [img.png] [outdir] [res] [texsize] [seed]
import os, sys
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
PIXAL3D_ROOT = "/mnt/hdd/3d/avatar-shootout/Pixal3D"
TRELLIS2_DIR = "/mnt/hdd/pixal3d_tex/trellis2_4b"
sys.path.insert(0, PIXAL3D_ROOT)
import trimesh, numpy as np, torch
from PIL import Image

MESH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/hdd/3d/avatar-shootout/_shootout_out/miku_lowpoly.glb"
IMG  = sys.argv[2] if len(sys.argv) > 2 else "/mnt/hdd/3d/avatar-shootout/_shootout_out/miku_clean_pose.png"
OUT  = sys.argv[3] if len(sys.argv) > 3 else "/mnt/hdd/3d/avatar-shootout/tex_goldens"
RES  = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
TEX  = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 42
os.makedirs(OUT, exist_ok=True)

def save(n, a):
    if isinstance(a, torch.Tensor): a = a.detach().float().cpu().numpy()
    a = np.ascontiguousarray(np.asarray(a))
    np.save(os.path.join(OUT, n + ".npy"), a); print(f"  [golden] {n:18s} {tuple(a.shape)} {a.dtype}", flush=True)

from pixal3d.pipelines import Trellis2TexturingPipeline
print(f"[tex] loading pipeline ({TRELLIS2_DIR}) res={RES} tex={TEX} seed={SEED}", flush=True)
pipe = Trellis2TexturingPipeline.from_pretrained(TRELLIS2_DIR, "_texturing_pipeline_local.json")
pipe.to("cuda")

scene = trimesh.load(MESH, process=False)
mesh = trimesh.util.concatenate([g for g in scene.geometry.values()]) if hasattr(scene, "geometry") else scene
mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
img = Image.open(IMG)
print(f"[tex] mesh V={len(mesh.vertices)} F={len(mesh.faces)} img={img.size}", flush=True)

with torch.no_grad():
    image = pipe.preprocess_image(img)
    save("proc_image_rgb", np.asarray(image.convert("RGB")))
    image.save(os.path.join(OUT, "proc_image.png"))
    mesh_pp = pipe.preprocess_mesh(mesh)
    save("mesh_pp_verts", mesh_pp.vertices); save("mesh_pp_faces", mesh_pp.faces.astype(np.int32))
    torch.manual_seed(SEED)
    cond = pipe.get_cond([image], 512) if RES == 512 else pipe.get_cond([image], 1024)
    # cond is a dict of tensors (image cond); bank the main cond tensor for validation
    for k, v in cond.items():
        if isinstance(v, torch.Tensor): save(f"cond_{k}", v)

    shape_slat = pipe.encode_shape_slat(mesh_pp, RES)
    save("shape_slat_feats", shape_slat.feats); save("shape_slat_coords", shape_slat.coords.int())

    tex_model = pipe.models['tex_slat_flow_model_512'] if RES == 512 else pipe.models['tex_slat_flow_model_1024']
    # bank the EXACT init noise drawn inside sample_tex_slat (the (M, in_ch-32) randn) so the C++ port
    # can validate the tex-DiT in ISOLATION (identical noise -> identical 12-step trajectory).
    _real_randn = torch.randn; _cap = {}
    def _rec(*a, **k):
        o = _real_randn(*a, **k)
        if o.dim() == 2 and o.shape[1] == 32: _cap['noise'] = o.detach().float().cpu().numpy().copy()
        return o
    torch.randn = _rec
    tex_slat = pipe.sample_tex_slat(cond, tex_model, shape_slat, {})
    torch.randn = _real_randn
    save("tex_slat_feats", tex_slat.feats); save("tex_slat_coords", tex_slat.coords.int())
    assert 'noise' in _cap, "did not capture (M,32) noise"
    save("tex_noise", _cap['noise'])
    # also bank the normalized shape_slat (concat_cond) the C++ must reproduce, and a step-0 pred_v
    std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(shape_slat.device)
    mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(shape_slat.device)
    shape_norm = (shape_slat - mean) / std
    save("shape_norm_feats", shape_norm.feats)
    # (step-0 pred_v golden is captured separately by capture_tex_pred0.py — the model is low_vram-moved
    # to CPU after sampling here, so a direct forward would device-mismatch.)

    pbr = pipe.decode_tex_slat(tex_slat)
    save("pbr_feats", pbr.feats); save("pbr_coords", pbr.coords.int())

    out_mesh = pipe.postprocess_mesh(mesh_pp, pbr, RES, TEX)
    glb = os.path.join(OUT, "oracle_textured.glb")
    out_mesh.export(glb); print(f"[tex] wrote {glb}", flush=True)

print("[done] ->", OUT, flush=True)
