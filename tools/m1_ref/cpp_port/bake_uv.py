#!/usr/bin/env python3
# lap-18 PRODUCTION texture bake — conformal UV PBR via cumesh (GPU xatlas).
#
# Why a Python sidecar: the smooth, streak-free texture needs a real CONFORMAL UV unwrap. C++ xatlas
# (ComputeCharts) is pathologically slow on these meshes (>20 min on the 3.1M dense mesh; 173 s + 37 k
# fragmented charts on the 200 k QEM → seams). cumesh wraps a GPU xatlas and unwraps the QEM mesh in
# ~5 s with clean, non-folding charts. The C++ precluster atlas folds → teal-interior + sliver streaks.
# So the bake is a GPU-Python post-step on the C++ chain's dump (same shape as Python's pyref bake).
#
# Pipeline (== Trellis2 postprocess): cumesh.uv_unwrap → nvdiffrast rasterize → flex_gemm grid_sample
# the 6-ch PBR volume at the per-texel 3D surface position → cv2 inpaint gutter → baseColor (sRGB) +
# metallicRoughness PBR glTF. The QEM mesh's raster positions sit on the (≈on-shell) crisp surface and
# the unwrap does NOT fold, so grid_sample never reads the teal interior — no snap/reproject needed.
#
#   bake_uv.py --dump <dir> --mesh qem|dense --texsize 2048 --out miku_qem_uv.glb
# dump dir holds dump_{mesh,dense}_{v,f}.bin + dump_pbr_{f,c}.bin (PIXAL3D_DUMP_BAKE=1 output).
import os, sys, time, argparse, numpy as np, torch, cv2
from PIL import Image
import trimesh, trimesh.visual, cumesh
import nvdiffrast.torch as dr
from flex_gemm.ops.grid_sample import grid_sample_3d

RES = 1024
LAYOUT = {'base_color': slice(0,3), 'metallic': slice(3,4), 'roughness': slice(4,5), 'alpha': slice(5,6)}

ap = argparse.ArgumentParser()
ap.add_argument('--dump', default='.')
ap.add_argument('--mesh', default='qem', choices=['qem','dense'])
ap.add_argument('--texsize', type=int, default=2048)
ap.add_argument('--out', default='out_uv.glb')
ap.add_argument('--inpaint', type=int, default=3)   # cv2 TELEA radius for the gutter
ap.add_argument('--ssaa', type=int, default=1)       # supersample: bake at texsize*ssaa, area-downsample → AA/smooth
ap.add_argument('--blur', type=float, default=0.0)   # gaussian sigma on baseColor (after inpaint) — gentle smoothing
ap.add_argument('--decimate', type=int, default=0)   # cumesh.simplify the DENSE mesh to N faces (finer silhouette than the 200k QEM)
a = ap.parse_args()
TS = a.texsize * a.ssaa
vbase = 'dump_mesh' if a.mesh == 'qem' else 'dump_dense'
v = np.fromfile(os.path.join(a.dump, f'{vbase}_v.bin'), dtype=np.float32).reshape(-1,3)
f = np.fromfile(os.path.join(a.dump, f'{vbase}_f.bin'), dtype=np.int64).reshape(-1,3)
feats = np.fromfile(os.path.join(a.dump, 'dump_pbr_f.bin'), dtype=np.float32).reshape(-1,6)
coords = np.fromfile(os.path.join(a.dump, 'dump_pbr_c.bin'), dtype=np.int32).reshape(-1,4)
print(f"[bake] mesh={a.mesh} V={v.shape[0]} F={f.shape[0]} vol N={feats.shape[0]} TS={TS}")

vt = torch.from_numpy(v).float().cuda(); ft = torch.from_numpy(f.astype(np.int32)).int().cuda()
t0 = time.time()
cm = cumesh.CuMesh(); cm.init(vt, ft)
if a.decimate > 0:
    cm.simplify(a.decimate)
    print(f"[bake] cumesh simplify -> {a.decimate} faces target ({time.time()-t0:.1f}s)")
vt, ft, uvt, vmap = cm.uv_unwrap(return_vmaps=True)
vt = vt.cuda(); ft = ft.cuda(); uvt = uvt.cuda()
print(f"[bake] cumesh unwrap: {vt.shape[0]} verts / {ft.shape[0]} tris ({time.time()-t0:.1f}s)")
verts = vt.cpu().numpy(); faces = ft.cpu().numpy(); uvs = uvt.cpu().numpy()
# smooth vertex normals on the (possibly simplified) UNWRAPPED mesh
normals = trimesh.Trimesh(vertices=verts, faces=faces, process=False).vertex_normals

ctx = dr.RasterizeCudaContext()
uvr = torch.cat([uvt*2-1, torch.zeros_like(uvt[:,:1]), torch.ones_like(uvt[:,:1])], -1).unsqueeze(0)
rast, _ = dr.rasterize(ctx, uvr, ft, resolution=[TS, TS]); mask = rast[0,...,3] > 0
pos = dr.interpolate(vt.unsqueeze(0), rast, ft)[0][0]
feats_t = torch.from_numpy(feats).float().cuda(); coords_t = torch.from_numpy(coords).int().cuda()
attrs = torch.zeros(TS, TS, 6, device='cuda')
attrs[mask] = grid_sample_3d(feats_t, coords_t, shape=torch.Size([1,6,RES,RES,RES]),
                             grid=((pos[mask]+0.5)*RES).reshape(1,-1,3), mode='trilinear')
m = mask.cpu().numpy()
def chan(sl, r):
    img = np.clip(attrs[..., sl].cpu().numpy()*255, 0, 255).astype(np.uint8)
    return cv2.inpaint(img, (~m).astype(np.uint8), r, cv2.INPAINT_TELEA)
base = chan(LAYOUT['base_color'], a.inpaint)
metal = chan(LAYOUT['metallic'], 1)[..., None]; rough = chan(LAYOUT['roughness'], 1)[..., None]
alpha = chan(LAYOUT['alpha'], 1)[..., None]
# gentle smoothing: optional gaussian blur (gutters are inpainted so no dark bleed), then
# supersample-downsample to the target size (area-average AA across charts/texels).
if a.blur > 0:
    base = cv2.GaussianBlur(base, (0,0), a.blur)
if a.ssaa > 1:
    ts = a.texsize
    base = cv2.resize(base, (ts,ts), interpolation=cv2.INTER_AREA)
    metal = cv2.resize(metal, (ts,ts), interpolation=cv2.INTER_AREA)[...,None]
    rough = cv2.resize(rough, (ts,ts), interpolation=cv2.INTER_AREA)[...,None]
    alpha = cv2.resize(alpha, (ts,ts), interpolation=cv2.INTER_AREA)[...,None]
mat = trimesh.visual.material.PBRMaterial(
    baseColorTexture=Image.fromarray(np.concatenate([base, alpha], -1)),
    baseColorFactor=np.array([255,255,255,255], np.uint8),
    metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metal), rough, metal], -1)),
    metallicFactor=1.0, roughnessFactor=1.0, alphaMode='OPAQUE', doubleSided=True)
uvs2 = uvs.copy(); uvs2[:,1] = 1 - uvs2[:,1]
out = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False,
                      visual=trimesh.visual.TextureVisuals(uv=uvs2, material=mat))
out.export(a.out)
print(f"[bake] wrote {a.out} ({faces.shape[0]} tris, {a.texsize}² atlas, ssaa={a.ssaa} blur={a.blur})")
