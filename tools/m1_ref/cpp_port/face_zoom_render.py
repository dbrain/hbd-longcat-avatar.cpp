#!/usr/bin/env python
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw

paths = sys.argv[1:-1]
out = sys.argv[-1]
YAW = float(os.environ.get("FACE_YAW", "0"))
TARGET_Y = float(os.environ.get("FACE_TARGET_Y", "0.30"))
FOV_DEG = float(os.environ.get("FACE_FOV_DEG", "18"))
if len(paths) < 1:
    print("usage: face_zoom_render.py a.glb [b.glb c.glb] out.png")
    raise SystemExit(1)

def load_mesh(path):
    s = trimesh.load(path, process=False)
    if isinstance(s, trimesh.Scene):
        meshes = []
        for name in s.graph.nodes_geometry:
            T, gname = s.graph[name]
            m = s.geometry[gname].copy()
            m.apply_transform(T)
            meshes.append(m)
        m = trimesh.util.concatenate(meshes)
    else:
        m = s
    m.vertices = m.vertices - m.vertices.mean(0)
    m.vertices = m.vertices / np.abs(m.vertices).max()
    return m

def look_at(eye, target):
    up = np.array([0, 1, 0.0])
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    pose = np.eye(4)
    pose[:3, 0] = s
    pose[:3, 1] = u
    pose[:3, 2] = -f
    pose[:3, 3] = eye
    return pose

def render(path, yaw_deg=0):
    m = load_mesh(path)
    pm = pyrender.Mesh.from_trimesh(m, smooth=True)
    scene = pyrender.Scene(bg_color=[0.04, 0.045, 0.05, 1], ambient_light=[0.75, 0.75, 0.75])
    scene.add(pm)
    yaw = np.radians(yaw_deg)
    target = np.array([0.0, TARGET_Y, 0.0])
    eye = target + np.array([1.10*np.sin(yaw), 0.02, 1.10*np.cos(yaw)])
    cam = pyrender.PerspectiveCamera(yfov=np.radians(FOV_DEG))
    pose = look_at(eye, target)
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.0), pose=pose)
    for d in [(1,1,1),(-1,1,1),(0,1,-1)]:
        dd=np.array(d,float); dd/=np.linalg.norm(dd)
        p=look_at(target-dd, target)
        scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=0.9), pose=p)
    r = pyrender.OffscreenRenderer(640, 640)
    color, _ = r.render(scene)
    r.delete()
    return Image.fromarray(color)

imgs = []
for p in paths:
    im = render(p, YAW)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0,0,639,24], fill=(5,5,8))
    draw.text((8,5), os.path.basename(p), fill=(230,230,230))
    imgs.append(im)

canvas = Image.new("RGB", (640*len(imgs), 640), (0,0,0))
for i, im in enumerate(imgs):
    canvas.paste(im, (640*i, 0))
canvas.save(out)
print("wrote", out)
