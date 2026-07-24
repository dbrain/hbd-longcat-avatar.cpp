#!/usr/bin/env python3
"""Capture the official o_voxel encoder boundary for one immutable fixed mesh.

This performs no model inference, sampling, baking, or geometry generation.  It only
records the tensors passed to FlexiDualGridVaeEncoder so the native voxelizer and
encoder can be compared at the actual framework boundary.
"""
import argparse
import hashlib
import json
from pathlib import Path

PIXAL3D_ROOT = Path("/mnt/hdd/3d/avatar-shootout/Pixal3D")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not args.mesh.is_file():
        raise SystemExit(f"missing fixed mesh: {args.mesh}")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite immutable diagnostic: {args.out}")

    import numpy as np
    import torch
    import trimesh
    import sys
    sys.path.insert(0, str(PIXAL3D_ROOT))
    import o_voxel

    scene = trimesh.load(args.mesh, process=False)
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) if hasattr(scene, "geometry") else scene
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    vertices = mesh.vertices
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = (vertices - center) * (0.99999 / (vertices.max(axis=0) - vertices.min(axis=0)).max())
    tmp = vertices[:, 1].copy()
    vertices[:, 1] = -vertices[:, 2]
    vertices[:, 2] = tmp
    if not (np.all(vertices >= -0.5) and np.all(vertices <= 0.5)):
        raise RuntimeError("official preprocess_mesh bounds check failed")

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        torch.from_numpy(vertices).float(), torch.from_numpy(mesh.faces).long(),
        grid_size=1024, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=True,
    )
    coords = np.concatenate([np.zeros((len(voxel_indices), 1), dtype=np.int32),
                             voxel_indices.numpy().astype(np.int32, copy=False)], axis=1)
    encoder_feats = np.concatenate([
        (dual_vertices * 1024 - voxel_indices).numpy(),
        intersected.float().numpy(),
    ], axis=1).astype(np.float32, copy=False)
    feats6 = encoder_feats - 0.5
    args.out.mkdir(parents=True)
    np.save(args.out / "python_voxel_coords.npy", np.ascontiguousarray(coords.astype("<i4", copy=False)))
    np.save(args.out / "python_voxel_feats6.npy", np.ascontiguousarray(feats6.astype("<f4", copy=False)))
    np.save(args.out / "python_voxel_encoder_feats.npy", np.ascontiguousarray(encoder_feats.astype("<f4", copy=False)))
    np.save(args.out / "python_voxel_dual_absolute.npy", np.ascontiguousarray(dual_vertices.numpy().astype("<f4", copy=False)))
    np.save(args.out / "python_voxel_intersected.npy", np.ascontiguousarray(intersected.numpy().astype("<i1", copy=False)))
    np.save(args.out / "python_preprocessed_vertices.npy", np.ascontiguousarray(vertices.astype("<f4", copy=False)))
    np.save(args.out / "python_preprocessed_faces.npy", np.ascontiguousarray(mesh.faces.astype("<i8", copy=False)))
    manifest = {"purpose": "official-Python o_voxel / FlexiDualGridVaeEncoder boundary capture",
                "mesh": str(args.mesh), "mesh_sha256": digest(args.mesh),
                "voxel_count": int(len(coords)), "resolution": 1024,
                "contract": "preprocess_mesh then mesh_to_flexible_dual_grid; no model inference or geometry run"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
