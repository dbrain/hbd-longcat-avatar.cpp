#!/usr/bin/env python
# render_tex_variants.py — "make textures betterer" comparison. Loads a TEXTURED glb, extracts its
# baseColor atlas, applies a few PRINCIPLED color transforms, and renders each variant from the same
# views into one labeled montage for owner judgment (no winner declared here).
#   render_tex_variants.py in.glb out.png
# Transforms:
#   baseline    : as-baked (current pipeline output)
#   srgb        : treat baked values as LINEAR albedo, encode to sRGB (the colorspace-fix hypothesis
#                 for "too dark": glTF baseColorTexture is sRGB-decoded by viewers; if the model emits
#                 linear, straight-write is double-dark). Brightens midtones.
#   gamma0.80   : pow(c, 0.80) mild perceptual brighten (no colorspace claim)
#   bright1.25  : 1.25x gain (clamped) — pure exposure
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image, ImageDraw

path = sys.argv[1] if len(sys.argv) > 1 else "_i2r_partretopo.glb"
out  = sys.argv[2] if len(sys.argv) > 2 else "_tex_variants.png"

def load_mesh(p):
    s = trimesh.load(p, process=False)
    if isinstance(s, trimesh.Scene):
        m = list(s.geometry.values())[0]
        for name in s.graph.nodes_geometry:
            T, gname = s.graph[name]
            if gname == list(s.geometry.keys())[0]:
                m = m.copy(); m.apply_transform(T); break
    else:
        m = s
    v = np.asarray(m.vertices, float); v -= v.mean(0); v /= np.abs(v).max(); m.vertices = v
    return m

base = load_mesh(path)
tex = base.visual.material.baseColorTexture
if tex is None:
    print("no baseColorTexture; aborting"); sys.exit(1)
arr0 = np.asarray(tex.convert("RGBA"), np.float32) / 255.0
print("baseColor atlas", arr0.shape, "valid-mean RGB(0-255)",
      (arr0[...,:3][arr0[...,3] > 0.5].mean(0) * 255).round(1) if (arr0[...,3] > 0.5).any() else "n/a")

def lin_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.clip(c, 0, 1), 1/2.4) - 0.055)

def xform(arr, kind):
    a = arr.copy(); rgb = a[..., :3]
    if   kind == "baseline":  pass
    elif kind == "srgb":      rgb = lin_to_srgb(rgb)
    elif kind == "gamma0.80": rgb = np.power(np.clip(rgb,0,1), 0.80)
    elif kind == "bright1.25":rgb = np.clip(rgb * 1.25, 0, 1)
    a[..., :3] = np.clip(rgb, 0, 1); return a

VARIANTS = ["baseline", "srgb", "gamma0.80", "bright1.25"]
YAWS = (0, 35)

def render_variant(arr, kind):
    a = xform(arr, kind)
    img = Image.fromarray((a * 255 + 0.5).astype(np.uint8), "RGBA")
    m = base.copy()
    m.visual = trimesh.visual.TextureVisuals(uv=base.visual.uv,
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=img,
                                                     metallicFactor=0.0, roughnessFactor=1.0))
    pm = pyrender.Mesh.from_trimesh(m, smooth=False)
    panels = []
    for yaw_deg in YAWS:
        scene = pyrender.Scene(bg_color=[0.1,0.1,0.12,1.0], ambient_light=[0.35,0.35,0.35])
        scene.add(pm)
        cam = pyrender.PerspectiveCamera(yfov=np.pi/3.5)
        yaw = np.radians(yaw_deg); cx, cz = 2.4*np.sin(yaw), 2.4*np.cos(yaw); cy = 0.4
        eye = np.array([cx, cy, cz]); tgt = np.array([0,0,0]); up = np.array([0,1,0.])
        f = (tgt-eye); f/=np.linalg.norm(f); s=np.cross(f,up); s/=np.linalg.norm(s); u2=np.cross(s,f)
        pose = np.eye(4); pose[:3,0]=s; pose[:3,1]=u2; pose[:3,2]=-f; pose[:3,3]=eye
        scene.add(cam, pose=pose)
        scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=4.0), pose=pose)
        scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.0),
                  pose=np.array([[1,0,0,2],[0,1,0,3],[0,0,1,2],[0,0,0,1.0]]))
        r = pyrender.OffscreenRenderer(560, 720); color, _ = r.render(scene); r.delete()
        panels.append(color)
    row = np.concatenate(panels, axis=1)
    im = Image.fromarray(row); d = ImageDraw.Draw(im)
    d.rectangle([0,0,260,28], fill=(0,0,0)); d.text((6,6), kind, fill=(255,255,0))
    return np.asarray(im)

rows = [render_variant(arr0, k) for k in VARIANTS]
Image.fromarray(np.concatenate(rows, axis=0)).save(out)
print("wrote", out)
