#!/usr/bin/env python3
"""Rank symmetric P3-SAM face-label pairs as *rigging-view* candidates.

This intentionally does not call a pair hair, wings, or limbs.  A caller may
try the resulting views only after the full-mesh rig has failed its anatomy
gate.  A pair is ranked from purely geometric evidence and must still earn a
valid skeleton plus an attachment/pose check before it can affect delivery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def one_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, process=False)
    if not isinstance(scene, trimesh.Scene) or len(scene.geometry) != 1:
        raise ValueError(f"expected one mesh geometry in {path}")
    return next(iter(scene.geometry.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("face_ids", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--min-face-fraction", type=float, default=0.03)
    parser.add_argument("--min-vertical-span", type=float, default=0.25)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    mesh = one_mesh(args.mesh)
    labels = np.load(args.face_ids)
    if labels.shape != (len(mesh.faces),):
        raise SystemExit(f"face-label shape {labels.shape} does not match {len(mesh.faces)} faces")
    vertices = np.asarray(mesh.vertices)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    ext = hi - lo
    # The pipeline canonicalises upright characters.  Retain an orientation-
    # fallback for other meshes rather than assuming Y blindly.
    vertical = int(np.argmax(ext))
    transverse = [axis for axis in range(3) if axis != vertical]
    lateral = max(transverse, key=lambda axis: ext[axis])
    depth = next(axis for axis in transverse if axis != lateral)
    centre = (lo + hi) * 0.5
    stats = []
    for label in np.unique(labels):
        faces = np.asarray(mesh.faces)[labels == label]
        used = np.unique(faces.reshape(-1))
        points = vertices[used]
        p_lo, p_hi = points.min(axis=0), points.max(axis=0)
        stats.append({
            "label": int(label),
            "faces": int(len(faces)),
            "fraction": float(len(faces) / len(mesh.faces)),
            "centre": ((p_lo + p_hi) * 0.5).tolist(),
            "extent": (p_hi - p_lo).tolist(),
        })
    candidate_stats = [
        s for s in stats
        if s["fraction"] >= args.min_face_fraction and s["extent"][vertical] >= args.min_vertical_span * ext[vertical]
    ]
    candidates = []
    for i, left in enumerate(candidate_stats):
        for right in candidate_stats[i + 1:]:
            lx = left["centre"][lateral] - centre[lateral]
            rx = right["centre"][lateral] - centre[lateral]
            if lx * rx >= 0:
                continue
            mirror_lat = abs(abs(lx) - abs(rx)) / max(ext[lateral], 1e-8)
            mirror_up = abs(left["centre"][vertical] - right["centre"][vertical]) / max(ext[vertical], 1e-8)
            mirror_depth = abs(left["centre"][depth] - right["centre"][depth]) / max(ext[depth], 1e-8)
            span_delta = abs(left["extent"][vertical] - right["extent"][vertical]) / max(ext[vertical], 1e-8)
            if mirror_lat > 0.18 or mirror_up > 0.25 or mirror_depth > 0.25 or span_delta > 0.25:
                continue
            area = left["fraction"] + right["fraction"]
            symmetry = 1.0 - (mirror_lat + mirror_up + mirror_depth + span_delta) / 4.0
            vertical_span = (left["extent"][vertical] + right["extent"][vertical]) / (2.0 * ext[vertical])
            # Large, tall, mirrored regions are useful *candidates*.  The
            # score never classifies their semantics or grants promotion.
            score = 0.55 * area + 0.30 * vertical_span + 0.15 * symmetry
            candidates.append({
                "labels": sorted([left["label"], right["label"]]),
                "score": round(float(score), 6),
                "area_fraction": round(float(area), 6),
                "vertical_span": round(float(vertical_span), 6),
                "symmetry": round(float(symmetry), 6),
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    result = {
        "mesh": str(args.mesh), "face_ids": str(args.face_ids), "face_count": int(len(mesh.faces)),
        "axes": {"vertical": vertical, "lateral": lateral, "depth": depth},
        "thresholds": {"min_face_fraction": args.min_face_fraction, "min_vertical_span": args.min_vertical_span},
        "regions": stats, "candidates": candidates,
        "promotion_rule": "Try only after full-mesh structural rejection; require valid skeleton, full-mesh attachment check and pose smoke. Do not infer semantic type from this ranking.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[p3sam-candidates] regions={len(stats)} candidates={len(candidates)} axes={result['axes']} -> {args.out}")
    for candidate in candidates[:8]:
        print("  labels={labels} score={score:.3f} area={area_fraction:.3f} span={vertical_span:.3f} sym={symmetry:.3f}".format(**candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
