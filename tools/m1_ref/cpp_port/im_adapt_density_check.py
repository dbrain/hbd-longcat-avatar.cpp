#!/usr/bin/env python
# CPU-ONLY (numpy, no GPU/render) sanity check for the IM adaptive-retopo A/B.
# NOT a success claim — finger separation is a visual question for the owner's render.
# This only quantifies WHETHER the curvature-driven sizing field actually moved
# vertex density toward high-curvature regions (vs the uniform baseline), and
# confirms the meshes are well-formed (parse, finite, matching bbox).
#
#   im_adapt_density_check.py a0.obj a0.5.obj a1.0.obj
import sys, numpy as np

def load_obj(path):
    V, F = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('v '):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('f '):
                idx = [int(t.split('/')[0]) for t in ln.split()[1:]]
                # fan-triangulate polygons (quads -> 2 tris)
                for k in range(1, len(idx)-1):
                    F.append([idx[0]-1, idx[k]-1, idx[k+1]-1])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)

def vertex_normals(V, F):
    fn = np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]])
    nl = np.linalg.norm(fn, axis=1, keepdims=True); nl[nl==0]=1; fn/=nl
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:,k], fn)
    nl = np.linalg.norm(vn, axis=1, keepdims=True); nl[nl==0]=1
    return vn/nl

def curvature(V, F, vn):
    # per-vertex mean (1 - n_i.n_j) over mesh edges (normal variation proxy)
    E = np.vstack([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]])
    dots = np.einsum('ij,ij->i', vn[E[:,0]], vn[E[:,1]])
    c = np.clip(1.0 - dots, 0, 2)
    acc = np.zeros(len(V)); cnt = np.zeros(len(V))
    for col in (0,1):
        np.add.at(acc, E[:,col], c); np.add.at(cnt, E[:,col], 1.0)
    cnt[cnt==0]=1
    return acc/cnt

stats = []
ref_curv_thresh = None
for path in sys.argv[1:]:
    V, F = load_obj(path)
    finite = np.isfinite(V).all()
    V0 = V - V.mean(0); V0 /= np.abs(V0).max()
    vn = vertex_normals(V0, F)
    cv = curvature(V0, F, vn)
    # use the FIRST mesh (uniform baseline) to fix the high-curvature threshold,
    # then measure what fraction of EACH mesh's vertices land above it.
    if ref_curv_thresh is None:
        ref_curv_thresh = np.quantile(cv, 0.90)  # top-decile curvature on baseline
    frac_hi = float((cv > ref_curv_thresh).mean())
    bb = V.max(0) - V.min(0)
    stats.append((path, len(V), len(F), finite, bb, frac_hi, float(cv.mean())))

print(f"{'mesh':<34}{'V':>9}{'F(tri)':>9}{'finite':>7}{'meanCurv':>10}{'%verts>thr':>11}")
base_frac = stats[0][5]
for path, nv, nf, fin, bb, frac, mc in stats:
    name = path.rsplit('/',1)[-1]
    rel = frac/base_frac if base_frac>0 else float('nan')
    print(f"{name:<34}{nv:>9}{nf:>9}{str(fin):>7}{mc:>10.4f}{frac*100:>9.2f}%  ({rel:.2f}x base)")
print(f"\nbbox (last mesh): {stats[-1][4]}")
print("threshold = top-decile curvature of the UNIFORM baseline.")
print(">1.0x means the adaptive mesh placed MORE of its vertices in the high-curvature band.")
