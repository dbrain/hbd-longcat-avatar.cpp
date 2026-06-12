#!/usr/bin/env python3
# Python parity reference for the UV-atlas bake: runs Trellis2TexturingPipeline.postprocess_mesh's
# core (cumesh.uv_unwrap + nvdiffrast rasterize + flex_gemm grid_sample_3d + cv2.inpaint + PBR
# material) on the SAME golden mesh + golden PBR volume the C++ tex_bake_test uses — so the two GLBs
# are directly comparable (same surface, same volume; only the UV layout / raster differ). This is
# the no-remesh path (what to_glb reduces to without remeshing). GPU required.
#   /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python tex_bake_python_ref.py [texture_size]
import os, sys, numpy as np, torch, cv2
from PIL import Image
import trimesh, trimesh.visual
import cumesh
import nvdiffrast.torch as dr
from flex_gemm.ops.grid_sample import grid_sample_3d
HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "refs", "stage4")
GOLD = os.path.join(HERE, "..", "..", "sparse_spike", "golden_stages", "stage5_mesh")
RES = 1024
TS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
LAYOUT = {'base_color': slice(0,3), 'metallic': slice(3,4), 'roughness': slice(4,5), 'alpha': slice(5,6)}

verts_np  = np.load(os.path.join(GOLD, "vertices.npy"))           # [V,3] f32  in [-0.5,0.5]
faces_np  = np.load(os.path.join(GOLD, "faces.npy"))              # [F,3] i64
feats_np  = np.load(os.path.join(REFS, "tex_pbr.npy"))            # [N,6] f32
coords_np = np.load(os.path.join(REFS, "tex_out_coords.npy"))     # [N,4] i32
print(f"[pyref] mesh V={verts_np.shape[0]} F={faces_np.shape[0]} volume N={feats_np.shape[0]} TS={TS}")

vertices_torch = torch.from_numpy(verts_np).float().cuda()
faces_torch = torch.from_numpy(faces_np.astype(np.int32)).int().cuda()
mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
normals = mesh.vertex_normals

# --- uv unwrap (xatlas via cumesh) ---
cm = cumesh.CuMesh(); cm.init(vertices_torch, faces_torch)
vertices_torch, faces_torch, uvs_torch, vmap = cm.uv_unwrap(return_vmaps=True)
vertices_torch = vertices_torch.cuda(); faces_torch = faces_torch.cuda(); uvs_torch = uvs_torch.cuda()
vertices = vertices_torch.cpu().numpy(); faces = faces_torch.cpu().numpy(); uvs = uvs_torch.cpu().numpy()
normals = normals[vmap.cpu().numpy()]
print(f"[pyref] unwrapped: {vertices.shape[0]} verts / {faces.shape[0]} tris")

# --- rasterize ---
ctx = dr.RasterizeCudaContext()
uvr = torch.cat([uvs_torch*2-1, torch.zeros_like(uvs_torch[:,:1]), torch.ones_like(uvs_torch[:,:1])], dim=-1).unsqueeze(0)
rast, _ = dr.rasterize(ctx, uvr, faces_torch, resolution=[TS, TS])
mask = rast[0, ..., 3] > 0
pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]

feats = torch.from_numpy(feats_np).float().cuda()
coords = torch.from_numpy(coords_np).int().cuda()
attrs = torch.zeros(TS, TS, feats.shape[1], device='cuda')
attrs[mask] = grid_sample_3d(feats, coords, shape=torch.Size([1, feats.shape[1], RES, RES, RES]),
                             grid=((pos[mask] + 0.5) * RES).reshape(1, -1, 3), mode='trilinear')

m = mask.cpu().numpy()
base = np.clip(attrs[..., LAYOUT['base_color']].cpu().numpy()*255, 0, 255).astype(np.uint8)
metal = np.clip(attrs[..., LAYOUT['metallic']].cpu().numpy()*255, 0, 255).astype(np.uint8)
rough = np.clip(attrs[..., LAYOUT['roughness']].cpu().numpy()*255, 0, 255).astype(np.uint8)
alpha = np.clip(attrs[..., LAYOUT['alpha']].cpu().numpy()*255, 0, 255).astype(np.uint8)
mi = (~m).astype(np.uint8)
base = cv2.inpaint(base, mi, 3, cv2.INPAINT_TELEA)
metal = cv2.inpaint(metal, mi, 1, cv2.INPAINT_TELEA)[..., None]
rough = cv2.inpaint(rough, mi, 1, cv2.INPAINT_TELEA)[..., None]
alpha = cv2.inpaint(alpha, mi, 1, cv2.INPAINT_TELEA)[..., None]

Image.fromarray(np.concatenate([base, alpha], -1)).save(os.path.join(HERE, "pyref_base_color.png"))
Image.fromarray(np.concatenate([np.zeros_like(metal), rough, metal], -1)).save(os.path.join(HERE, "pyref_metal_rough.png"))

material = trimesh.visual.material.PBRMaterial(
    baseColorTexture=Image.fromarray(np.concatenate([base, alpha], -1)),
    baseColorFactor=np.array([255,255,255,255], dtype=np.uint8),
    metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metal), rough, metal], -1)),
    metallicFactor=1.0, roughnessFactor=1.0, alphaMode='OPAQUE', doubleSided=True)
uvs2 = uvs.copy(); uvs2[:,1] = 1 - uvs2[:,1]
out = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False,
                      visual=trimesh.visual.TextureVisuals(uv=uvs2, material=material))
out.export(os.path.join(HERE, "miku_uvatlas_pyref.glb"))
print(f"[pyref] wrote miku_uvatlas_pyref.glb + pyref_base_color.png + pyref_metal_rough.png")
print(f"[pyref] base_color valid-region mean = {base[m].mean(0) if m.any() else None}")
