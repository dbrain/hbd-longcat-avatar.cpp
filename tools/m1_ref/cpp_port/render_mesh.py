#!/usr/bin/env python
# Render a .ply mesh headless (EGL offscreen) from a few angles -> PNG montage.
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "miku_geometry_e2e.ply"
out  = sys.argv[2] if len(sys.argv) > 2 else "miku_geometry_e2e_render.png"

m = trimesh.load(path, process=False)
print("loaded", path, "verts", len(m.vertices), "faces", len(m.faces))
m.vertices -= m.vertices.mean(0)
scale = 1.0 / np.abs(m.vertices).max()
m.vertices *= scale

mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.75,0.78,0.82,1.0],
                                         metallicFactor=0.1, roughnessFactor=0.7,
                                         doubleSided=True)
pm = pyrender.Mesh.from_trimesh(m, material=mat, smooth=False)

def render_view(yaw_deg):
    scene = pyrender.Scene(bg_color=[0.1,0.1,0.12,1.0], ambient_light=[0.25,0.25,0.25])
    scene.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3.5)
    yaw = np.radians(yaw_deg)
    # camera orbit around Y, looking at origin, slight elevation
    cx, cz = 2.4*np.sin(yaw), 2.4*np.cos(yaw)
    cy = 0.4
    eye = np.array([cx, cy, cz]); tgt = np.array([0,0,0]); up = np.array([0,1,0])
    f = (tgt-eye); f/=np.linalg.norm(f); s=np.cross(f,up); s/=np.linalg.norm(s); u2=np.cross(s,f)
    pose = np.eye(4); pose[:3,0]=s; pose[:3,1]=u2; pose[:3,2]=-f; pose[:3,3]=eye
    scene.add(cam, pose=pose)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=4.0)
    scene.add(light, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=1.5),
              pose=np.array([[1,0,0,2],[0,1,0,3],[0,0,1,2],[0,0,0,1.0]]))
    r = pyrender.OffscreenRenderer(640, 720)
    color, _ = r.render(scene)
    r.delete()
    return color

imgs = [render_view(a) for a in (0, 35, 90)]
montage = np.concatenate(imgs, axis=1)
Image.fromarray(montage).save(out)
print("wrote", out, montage.shape)
