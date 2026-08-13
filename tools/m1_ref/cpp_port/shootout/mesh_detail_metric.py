#!/usr/bin/env python3
"""How much SURFACE RELIEF does a mesh actually carry, at a scale-independent measure?

Used to settle "does UltraShape refine add detail the O-Voxel-DC parity mesh does not already
have?" Face count answers nothing (a dense mesh can be smooth), and a flat-lit clay render
answers nothing either, so measure the geometry:

  dihedral   angle between adjacent face normals. The distribution's upper tail is creases,
             folds and hard edges — the things a normal map would carry.
  roughness  per-vertex displacement from a Taubin-smoothed copy of the SAME mesh, as a
             fraction of the bounding-box diagonal. This is literally "what a low-pass filter
             removes" = the high-frequency relief, and it is invariant to tessellation density
             in a way face count is not.

Both are reported at fixed percentiles so two meshes at different densities compare fairly.

usage: mesh_detail_metric.py <mesh.glb> [more.glb ...]
"""

from __future__ import annotations

import sys

import numpy as np
import trimesh


def metrics(path: str) -> dict:
    scene = trimesh.load(path, process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))

    adj = mesh.face_adjacency_angles  # radians between adjacent face normals
    dih = np.degrees(adj)

    # One explicit Laplacian (umbrella) pass as the low-pass. Not Taubin's shrink/inflate pair:
    # we want what a smoother REMOVES, and a single pass isolates that without the volume
    # correction confusing the residual.
    nbr_sum = np.zeros_like(v)
    deg = np.zeros(len(v))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        i, j = f[:, a], f[:, b]
        np.add.at(nbr_sum, i, v[j]); np.add.at(deg, i, 1)
        np.add.at(nbr_sum, j, v[i]); np.add.at(deg, j, 1)
    deg[deg == 0] = 1
    smoothed = nbr_sum / deg[:, None]
    resid = np.linalg.norm(v - smoothed, axis=1) / diag

    return {
        "path": path.split("/")[-1],
        "V": len(v), "F": len(f),
        "dihedral_p50": float(np.percentile(dih, 50)),
        "dihedral_p95": float(np.percentile(dih, 95)),
        "dihedral_p99": float(np.percentile(dih, 99)),
        "rough_p50_ppm": float(np.percentile(resid, 50) * 1e6),
        "rough_p95_ppm": float(np.percentile(resid, 95) * 1e6),
        "rough_p99_ppm": float(np.percentile(resid, 99) * 1e6),
        "edge_len_p50_ppm": float(np.percentile(mesh.edges_unique_length, 50) / diag * 1e6),
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <mesh.glb> [more.glb ...]")
    rows = [metrics(p) for p in sys.argv[1:]]
    hdr = ("mesh", "V", "F", "dih50", "dih95", "dih99", "rgh50", "rgh95", "rgh99", "edge50")
    print(f"{hdr[0]:<26}{hdr[1]:>10}{hdr[2]:>10}" + "".join(f"{h:>9}" for h in hdr[3:]))
    for r in rows:
        print(f"{r['path']:<26}{r['V']:>10}{r['F']:>10}"
              f"{r['dihedral_p50']:>9.2f}{r['dihedral_p95']:>9.2f}{r['dihedral_p99']:>9.2f}"
              f"{r['rough_p50_ppm']:>9.0f}{r['rough_p95_ppm']:>9.0f}{r['rough_p99_ppm']:>9.0f}"
              f"{r['edge_len_p50_ppm']:>9.0f}")
    print("\ndih* = degrees between adjacent faces (creases live in the tail)")
    print("rgh* = per-vertex displacement from a one-pass Laplacian, in ppm of bbox diagonal")
    print("edge50 = median edge length in ppm of bbox diagonal (tessellation scale)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
