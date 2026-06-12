#!/usr/bin/env python
# Render a TEXTURED .glb headless (EGL) using its embedded PBR texture + UVs (no material override),
# from a few angles -> PNG montage. For validating the UV-atlas bake visually.
#   render_textured.py in.glb out.png
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "miku_uvatlas.glb"
out  = sys.argv[2] if len(sys.argv) > 2 else "miku_uvatlas_render.png"

scene_or_mesh = trimesh.load(path, process=False)
if isinstance(scene_or_mesh, trimesh.Scene):
    geoms = list(scene_or_mesh.geometry.values())
    m = geoms[0]
    # apply the scene transform so orientation matches
    if scene_or_mesh.graph is not None:
        for name in scene_or_mesh.graph.nodes_geometry:
            T, gname = scene_or_mesh.graph[name]
            if gname == list(scene_or_mesh.geometry.keys())[0]:
                m = m.copy(); m.apply_transform(T); break
else:
    m = scene_or_mesh
print("loaded", path, "verts", len(m.vertices), "faces", len(m.faces), "visual", type(m.visual).__name__)

m.vertices = m.vertices - m.vertices.mean(0)
m.vertices = m.vertices / np.abs(m.vertices).max()
pm = pyrender.Mesh.from_trimesh(m, smooth=False)   # uses the trimesh TextureVisuals/material

def render_view(yaw_deg):
    scene = pyrender.Scene(bg_color=[0.1,0.1,0.12,1.0], ambient_light=[0.35,0.35,0.35])
    scene.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3.5)
    yaw = np.radians(yaw_deg)
    cx, cz = 2.4*np.sin(yaw), 2.4*np.cos(yaw); cy = 0.4
    eye = np.array([cx, cy, cz]); tgt = np.array([0,0,0]); up = np.array([0,1,0])
    f = (tgt-eye); f/=np.linalg.norm(f); s=np.cross(f,up); s/=np.linalg.norm(s); u2=np.cross(s,f)
    pose = np.eye(4); pose[:3,0]=s; pose[:3,1]=u2; pose[:3,2]=-f; pose[:3,3]=eye
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=4.0), pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.0),
              pose=np.array([[1,0,0,2],[0,1,0,3],[0,0,1,2],[0,0,0,1.0]]))
    r = pyrender.OffscreenRenderer(640, 720)
    color, _ = r.render(scene); r.delete()
    return color

imgs = [render_view(a) for a in (0, 35, 90, 180)]
Image.fromarray(np.concatenate(imgs, axis=1)).save(out)
print("wrote", out)
