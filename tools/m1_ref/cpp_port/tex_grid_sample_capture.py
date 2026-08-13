#!/usr/bin/env python3
# Capture a grid_sample_3d oracle for the UV-atlas texture bake validation.
#  - real PBR volume: refs/stage4/tex_pbr.npy  [N,6]  +  tex_out_coords.npy [N,4]  (grid 1024)
#  - query = (golden mesh vertices + 0.5) * 1024   (the surface points the atlas samples)
#  - oracle = flex_gemm.ops.grid_sample.grid_sample_3d(feats, coords, grid, 'trilinear')
# Also cross-checks a pure-numpy reimplementation of the algorithm vs flex_gemm on a subset, so we
# trust the C++ port's target. Run from the Pixal3D venv with the GPU visible:
#   /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python tex_grid_sample_capture.py
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "refs", "stage4")
GOLD = os.path.join(HERE, "..", "..", "sparse_spike", "golden_stages")
RES = 1024

feats_np  = np.load(os.path.join(REFS, "tex_pbr.npy"))         # [N,6] f32
coords_np = np.load(os.path.join(REFS, "tex_out_coords.npy"))  # [N,4] i32
verts_np  = np.load(os.path.join(GOLD, "stage5_mesh", "vertices.npy"))  # [V,3] f32 in [-0.5,0.5]
N, C = feats_np.shape
V = verts_np.shape[0]
print(f"[cap] volume N={N} C={C}, mesh V={V}")

query_np = (verts_np + 0.5) * RES                              # [V,3] grid-index space
np.save(os.path.join(REFS, "tex_gs_query.npy"), query_np.astype(np.float32))

# ---- flex_gemm oracle (GPU) ----
from flex_gemm.ops.grid_sample import grid_sample_3d
feats  = torch.from_numpy(feats_np).float().cuda()
coords = torch.from_numpy(coords_np).int().cuda()
grid   = torch.from_numpy(query_np).float().cuda().reshape(1, -1, 3)
shape  = torch.Size([1, C, RES, RES, RES])
out = grid_sample_3d(feats, coords, shape=shape, grid=grid, mode="trilinear")  # [1,V,C]
oracle = out[0].cpu().numpy().astype(np.float32)
np.save(os.path.join(REFS, "tex_gs_oracle.npy"), oracle)
print(f"[cap] oracle saved [{oracle.shape}], range [{oracle.min():.4f},{oracle.max():.4f}]")

# ---- pure-numpy cross-check on a subset (validates the algorithm we port to C++) ----
SUB = min(20000, V)
sub = query_np[:SUB]
index = {}
cx = coords_np[:, 1]; cy = coords_np[:, 2]; cz = coords_np[:, 3]
for i in range(N):
    index[(int(cx[i]), int(cy[i]), int(cz[i]))] = i
OFF = [(-.5,-.5,-.5),(-.5,-.5,.5),(-.5,.5,-.5),(-.5,.5,.5),(.5,-.5,-.5),(.5,-.5,.5),(.5,.5,-.5),(.5,.5,.5)]
ref = np.zeros((SUB, C), np.float32)
for j in range(SUB):
    q = sub[j]; acc = np.zeros(C, np.float32); tw = 0.0
    for (dx,dy,dz) in OFF:
        ix = int(q[0]+dx); iy = int(q[1]+dy); iz = int(q[2]+dz)  # int() truncates toward zero
        k = (ix,iy,iz)
        if k in index:
            w = (1-abs(ix+0.5-q[0]))*(1-abs(iy+0.5-q[1]))*(1-abs(iz+0.5-q[2]))
            acc += np.float32(w) * feats_np[index[k]]; tw += w
    ref[j] = acc/tw if tw > 1e-12 else 0.0
d = np.abs(ref - oracle[:SUB])
print(f"[cap] numpy-ref vs flex_gemm (subset {SUB}): maxabs={d.max():.3e} meanabs={d.mean():.3e}")
print("[cap] DONE")
