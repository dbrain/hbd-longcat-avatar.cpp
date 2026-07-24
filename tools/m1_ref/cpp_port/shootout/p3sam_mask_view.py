#!/usr/bin/env python3
"""Create an exact P3-SAM face-label view of a mesh for rigging diagnostics.

P3-SAM labels are geometric regions, not semantics.  This tool therefore
requires the caller to state the labels to remove and records no implicit
"hair" policy.  Use it only to test a candidate rigging view; final skinning
must still target the untouched textured mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("face_ids", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--drop", nargs="+", required=True, type=np.int64)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    loaded = trimesh.load(args.mesh, process=False)
    if not isinstance(loaded, trimesh.Scene) or len(loaded.geometry) != 1:
        raise SystemExit("expected exactly one mesh geometry")
    mesh = next(iter(loaded.geometry.values()))
    labels = np.load(args.face_ids)
    if labels.shape != (len(mesh.faces),):
        raise SystemExit(f"face-label shape {labels.shape} does not match {len(mesh.faces)} faces")
    keep = ~np.isin(labels, np.asarray(args.drop, dtype=np.int64))
    faces = np.asarray(mesh.faces)[keep]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    body = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[used], faces=inverse.reshape(-1, 3), process=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body.export(args.out)
    removed = int((~keep).sum())
    print(f"[p3sam-view] labels dropped={args.drop} faces {len(mesh.faces)} -> {len(faces)} "
          f"(removed={removed}), vertices {len(mesh.vertices)} -> {len(body.vertices)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
