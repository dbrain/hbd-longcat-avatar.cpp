#!/usr/bin/env python3
# Robust multi-view renderer: handles trimesh Scene (multi-geom GLB) + textured materials.
#   _val_render.py <in.glb|ply> <out.png> [textured|geom]
import sys, numpy as np, trimesh, pyrender
from PIL import Image

path = sys.argv[1]; out = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "auto"

loaded = trimesh.load(path)
if isinstance(loaded, trimesh.Scene):
    scene_tm = loaded
    geoms = list(loaded.geometry.values())
    nv = sum(len(g.vertices) for g in geoms); nf = sum(len(g.faces) for g in geoms)
else:
    scene_tm = trimesh.Scene(loaded); geoms = [loaded]
    nv = len(loaded.vertices); nf = len(loaded.faces)
print(f"loaded {path} verts {nv} faces {nf} geoms {len(geoms)}")

# center + scale to unit
bounds = scene_tm.bounds
center = bounds.mean(axis=0); extent = (bounds[1]-bounds[0]).max()
scl = 1.6/extent

def render_view(yaw):
    sc = pyrender.Scene(bg_color=[0.1,0.1,0.11,1.0], ambient_light=[0.35,0.35,0.35])
    for g in geoms:
        gg = g.copy()
        gg.apply_translation(-center); gg.apply_scale(scl)
        smooth = mode != "geom"
        try:
            m = pyrender.Mesh.from_trimesh(gg, smooth=smooth)
        except Exception:
            m = pyrender.Mesh.from_trimesh(gg, smooth=False)
        sc.add(m)
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    import math
    cp = np.eye(4)
    r = 2.4; a = math.radians(yaw)
    cp[:3,3] = [r*math.sin(a), 0.15, r*math.cos(a)]
    # look-at
    fwd = -cp[:3,3]/np.linalg.norm(cp[:3,3]); up=np.array([0,1,0.0])
    rt = np.cross(up,fwd); rt/=np.linalg.norm(rt); up2=np.cross(fwd,rt)
    cp[:3,0]=rt; cp[:3,1]=up2; cp[:3,2]=-fwd
    sc.add(cam, pose=cp)
    for dl in ([2,2,2],[-2,1,1],[0,2,-2]):
        lp=np.eye(4); lp[:3,3]=dl
        sc.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=lp)
    r2 = pyrender.OffscreenRenderer(640, 960)
    color,_ = r2.render(sc); r2.delete()
    return color

views = [render_view(y) for y in (0, 90, 180)]
mont = np.concatenate(views, axis=1)
Image.fromarray(mont).save(out)
print("wrote", out, mont.shape)
