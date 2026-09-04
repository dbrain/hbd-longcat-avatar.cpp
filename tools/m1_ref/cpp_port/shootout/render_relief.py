#!/usr/bin/env python3
"""Hard-lit clay render for judging SURFACE RELIEF (not silhouette).

The existing _render_clay.py renders with heavy ambient, which flattens everything into a
silhouette — useless for "does this mesh carry more detail than that one". This lights the
subject with a single strong directional key and almost no ambient, so relief reads as shading,
and emits a full-body view plus a tight head crop where anime-face detail either survives or does
not.

usage: render_relief.py <out.png> <label=mesh.glb> [label=mesh.glb ...]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw

W, H = 520, 700


def load_normalized(path: str) -> trimesh.Trimesh:
    scene = trimesh.load(path, process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    v = np.asarray(mesh.vertices, dtype=float)
    v = v - (v.min(0) + v.max(0)) * 0.5
    v = v / max(float((v.max(0) - v.min(0)).max()), 1e-9)   # unit longest axis, centred
    mesh.vertices = v
    # NORMALS AS COLOUR, not a lit clay. A lit render depends on getting light intensity and
    # material right (get it wrong and everything saturates to one flat grey — which is exactly
    # what happened with the clay renders). Painting the vertex normal into RGB makes relief
    # visible unconditionally: flat regions are one colour, every crease and bump is a colour
    # break, and two meshes are comparable because the mapping has no free parameters.
    n = np.asarray(mesh.vertex_normals, dtype=float)
    rgb = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=np.column_stack([rgb, np.full(len(rgb), 255, np.uint8)]))
    return mesh


def render(mesh: trimesh.Trimesh, centre, radius: float, yaw_deg: float) -> Image.Image:
    # Full ambient + no key light: we want the baked normal colour itself, unmodulated.
    scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.07, 1.0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    yaw = np.radians(yaw_deg)
    eye = np.array(centre, dtype=float) + radius * np.array([np.sin(yaw) * 0.55, 0.30, np.cos(yaw)])
    fwd = np.array(centre, dtype=float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 5, aspectRatio=W / H), pose=pose)
    colour, _ = pyrender.OffscreenRenderer(W, H).render(
        scene, flags=pyrender.RenderFlags.FLAT)   # FLAT = show vertex colour, skip lighting
    return Image.fromarray(colour)


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <out.png> <label=mesh.glb> ...")
    out = sys.argv[1]
    args = sys.argv[2:]
    # Front-facing yaw is PER SUBJECT (miku 180, soldier 0), so a fixed camera renders some
    # subjects from behind and the A/B silently compares a back to a back.
    yaw_deg = 180.0
    if "--yaw" in args:
        i = args.index("--yaw")
        yaw_deg = float(args[i + 1])
        del args[i:i + 2]
    entries = [a.split("=", 1) for a in args]

    cols = []
    for label, path in entries:
        mesh = load_normalized(path)
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        body = render(mesh, [0, 0, 0], 1.5, yaw_deg)
        # Torso/upper-body crop: for a standing figure the very top of the bbox is hair, so aim a
        # little lower — that band holds the face, collar and chest folds, i.e. the detail that
        # either survives a remesh or does not.
        head_y = lo[1] + 0.78 * (hi[1] - lo[1])
        head = render(mesh, [0, head_y, 0], 0.42, yaw_deg)
        col = Image.new("RGB", (W, H * 2), (12, 12, 16))
        col.paste(body, (0, 0)); col.paste(head, (0, H))
        d = ImageDraw.Draw(col)
        d.text((8, 6), f"{label}  V={len(mesh.vertices)} F={len(mesh.faces)}", fill=(255, 220, 90))
        d.text((8, H + 6), f"{label} — head", fill=(255, 220, 90))
        cols.append(col)

    sheet = Image.new("RGB", (W * len(cols), H * 2), (12, 12, 16))
    for i, col in enumerate(cols):
        sheet.paste(col, (i * W, 0))
    sheet.save(out)
    print(f"wrote {out} ({len(cols)} meshes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
