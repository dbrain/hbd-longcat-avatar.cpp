#!/usr/bin/env python
# Per-part region-adaptive decimation: split a P3-SAM-segmented mesh by part, decimate EACH part to a
# budget set by its importance (hands keep ~full, hair/torso crush hard), via the C++ obj_decimate
# (meshopt feature-aware QEM, LockBorder -> part cut-boundaries pinned so parts recombine seamlessly).
# python only does split + recombine I/O; the decimation is the C++ QEM.
#   per_part_decimate.py <mesh.glb> <face_ids.npy> <out.glb> [dump_mesh_dir]
# [dump_mesh_dir] (optional): also write the final mesh as dump_mesh_v.bin (float32) + dump_mesh_f.bin
# (int64) there, the format tex_reproject reads as its bake target. The texture is then baked onto THIS
# exact mesh via `RP_CANON_TO_DENSE=1 CUMESH_TARGET=0 tex_reproject` (auto-aligned to the dump frame).
import os, sys, subprocess, tempfile
import numpy as np, trimesh
HERE = os.path.dirname(os.path.abspath(__file__))
OBJ_DECIMATE = os.path.join(HERE, '..', 'obj_decimate')   # cpp_port/obj_decimate

mesh_path, fid_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
dump_dir = sys.argv[4] if len(sys.argv) > 4 else None
m = trimesh.load(mesh_path, process=False)
m = trimesh.util.concatenate(list(m.geometry.values())) if isinstance(m, trimesh.Scene) else m
fid = np.load(fid_path)
V = np.asarray(m.vertices, float); F = np.asarray(m.faces); FC = V[F].mean(1)
c = (V.min(0)+V.max(0))/2; s = np.abs(V-c).max(); FCN = (FC-c)/s   # normalized face centroids, Y-up
TOTAL = len(F)

def tier_keep(part):
    sel = fid == part; n = sel.sum(); frac = n/TOTAL
    cen = FCN[sel].mean(0); xm = np.abs(FCN[sel][:,0]).max()
    # region-adaptive importance -> fraction of faces to KEEP
    if n < 0.025*TOTAL and xm > 0.50 and -0.35 < cen[1] < 0.35:  return 'HAND', 0.80   # fingers: keep
    if frac > 0.12:                                              return 'HAIR', 0.025  # twintails: crush
    if cen[1] < -0.70:                                           return 'FOOT', 0.12
    if cen[1] > 0.55:                                            return 'HEAD', 0.15
    if xm > 0.42 and cen[1] > 0.05:                              return 'LIMB', 0.10   # forearm/sleeve
    return 'BODY', 0.05                                                                # torso/skirt/legs

parts = sorted(np.unique(fid), key=lambda p:-(fid==p).sum())
pieces = []
print(f"{'part':>5} {'tier':>5} {'in_f':>8} -> {'out_f':>7}")
for p in parts:
    sel = fid == p
    fsub = F[sel]
    used = np.unique(fsub); remap = {v:i for i,v in enumerate(used)}
    vsub = V[used]; fsub2 = np.vectorize(remap.get)(fsub)
    tier, keep = tier_keep(p)
    target = max(400, int(round(len(fsub2)*keep)))
    with tempfile.TemporaryDirectory() as td:
        oin, oout = os.path.join(td,'i.obj'), os.path.join(td,'o.obj')
        with open(oin,'w') as f:
            for v in vsub: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for t in fsub2: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
        if len(fsub2) > target*1.05:
            subprocess.run([OBJ_DECIMATE, oin, oout, str(target)], capture_output=True, check=True)
            dm = trimesh.load(oout, process=False)
        else:
            dm = trimesh.Trimesh(vertices=vsub, faces=fsub2, process=False)   # already small: keep
    pieces.append(dm)
    print(f"{p:>5} {tier:>5} {len(fsub2):>8} -> {len(dm.faces):>7}")

out = trimesh.util.concatenate(pieces)
out.export(out_path)
print(f"TOTAL {TOTAL} -> {len(out.faces)} faces  ({100*len(out.faces)/TOTAL:.1f}%)  -> {out_path}")

if dump_dir:
    ov = np.ascontiguousarray(out.vertices, dtype=np.float32)
    of = np.ascontiguousarray(out.faces,    dtype=np.int64)
    ov.tofile(os.path.join(dump_dir, 'dump_mesh_v.bin'))
    of.tofile(os.path.join(dump_dir, 'dump_mesh_f.bin'))
    # tex_reproject reads NVq/NFq from dump_bake.txt's first two fields (NP stays from the --tex run).
    bk = os.path.join(dump_dir, 'dump_bake.txt')
    np_voxels = 0
    if os.path.exists(bk):
        try: np_voxels = int(open(bk).read().split()[2])
        except Exception: np_voxels = 0
    with open(bk, 'w') as f:
        f.write(f"{len(ov)} {len(of)} {np_voxels}\n")
    print(f"[dump] bake target -> {dump_dir}/dump_mesh_*.bin ({len(ov)} v / {len(of)} f), dump_bake.txt updated")
