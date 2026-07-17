#!/usr/bin/env python3
# Symmetric surface (point-cloud) chamfer between two GLB/PLY meshes — quality A/B for precision levers.
#   _mesh_cmp.py <a.glb> <b.glb>
# Samples points on each surface, kNN nearest-distance to the other's samples (scipy cKDTree, no rtree),
# reports as a fraction of the bbox diagonal (scale-free). ~0 => geometrically identical.
import sys, numpy as np, trimesh
from scipy.spatial import cKDTree

def asmesh(p):
    return trimesh.load(p, force='mesh')

a = asmesh(sys.argv[1]); b = asmesh(sys.argv[2])
N = 300000
diag = float(np.linalg.norm(a.bounds[1] - a.bounds[0]))
pa = a.sample(N); pb = b.sample(N)
ta = cKDTree(pa); tb = cKDTree(pb)
dab, _ = tb.query(pa)   # a-samples -> nearest b-sample
dba, _ = ta.query(pb)   # b-samples -> nearest a-sample
cham = (dab.mean() + dba.mean()) / 2.0
print(f"A: {len(a.vertices)} v / {len(a.faces)} f   B: {len(b.vertices)} v / {len(b.faces)} f")
print(f"bbox diag (A) = {diag:.5f}")
print(f"mean A->B = {dab.mean():.6f} ({100*dab.mean()/diag:.4f}% diag)  p95 {np.percentile(dab,95):.6f}  max {dab.max():.6f}")
print(f"mean B->A = {dba.mean():.6f} ({100*dba.mean()/diag:.4f}% diag)  p95 {np.percentile(dba,95):.6f}  max {dba.max():.6f}")
print(f"symmetric chamfer = {cham:.6f}  ({100*cham/diag:.4f}% of bbox diag)")
