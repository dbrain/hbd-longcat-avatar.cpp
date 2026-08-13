#!/usr/bin/env python
# Crispness-revealing geometry render. Unlike render_mesh.py (light ON the camera axis -> flat
# silhouette), this lights OFF-axis (upper-front-left) so surface orientation shades -> sharp edges,
# heel spikes, finger gaps and the chin/neck step all read. Also emits zoomed detail crops of the
# feet and the hands/arms region (the features the owner cares about). Geometry-only (grey clay).
#   render_geo_detail.py <mesh.ply> <out_prefix>
# Writes <out_prefix>_body.png (3 views) and <out_prefix>_detail.png (feet + hand zoom crops).
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image

path = sys.argv[1]
outp = sys.argv[2] if len(sys.argv) > 2 else "geo_detail"

m = trimesh.load(path, process=False)
print("loaded", path, "verts", len(m.vertices), "faces", len(m.faces))
m.vertices -= m.vertices.mean(0)
scale = 1.0 / np.abs(m.vertices).max()
m.vertices *= scale

mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.72,0.74,0.78,1.0],
                                         metallicFactor=0.0, roughnessFactor=0.55,
                                         doubleSided=True)
pm = pyrender.Mesh.from_trimesh(m, material=mat, smooth=False)

def render(yaw_deg, target, dist, elev, W, H, fov=np.pi/4.0):
    scene = pyrender.Scene(bg_color=[0.06,0.06,0.07,1.0], ambient_light=[0.12,0.12,0.13])
    scene.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=fov)
    yaw = np.radians(yaw_deg)
    tgt = np.array(target, float)
    cx, cz = dist*np.sin(yaw), dist*np.cos(yaw)
    eye = tgt + np.array([cx, elev, cz])
    up = np.array([0,1,0.0])
    f = (tgt-eye); f/=np.linalg.norm(f); s=np.cross(f,up); s/=np.linalg.norm(s); u2=np.cross(s,f)
    pose = np.eye(4); pose[:3,0]=s; pose[:3,1]=u2; pose[:3,2]=-f; pose[:3,3]=eye
    scene.add(cam, pose=pose)
    # KEY: off-axis key light (upper-front-left) reveals surface orientation -> sharp features read.
    keypose = np.eye(4); keypose[:3,3]=[-2.0, 2.6, 2.2]
    kd=np.array([-2.0,2.6,2.2]); kd/=np.linalg.norm(kd)
    kf=-kd; ks=np.cross(kf,up); ks/=np.linalg.norm(ks); ku=np.cross(ks,kf)
    keypose[:3,0]=ks; keypose[:3,1]=ku; keypose[:3,2]=-kf
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.2), pose=keypose)
    # weak fill from the right so the shadow side isn't black
    fillpose=np.eye(4); fd=np.array([2.0,0.5,1.5]); fd/=np.linalg.norm(fd)
    ff=-fd; fs=np.cross(ff,up); fs/=np.linalg.norm(fs); fu=np.cross(fs,ff)
    fillpose[:3,0]=fs; fillpose[:3,1]=fu; fillpose[:3,2]=-ff
    scene.add(pyrender.DirectionalLight(color=[0.7,0.75,0.85], intensity=1.1), pose=fillpose)
    r = pyrender.OffscreenRenderer(W, H)
    color, _ = r.render(scene); r.delete()
    return color

# whole body, 3 yaws
body = [render(a, [0,0,0], 2.4, 0.35, 560, 760) for a in (0, 40, 90)]
Image.fromarray(np.concatenate(body, axis=1)).save(outp+"_body.png")
print("wrote", outp+"_body.png")

# detail: feet (low y) and hands (mid, towards the sides). Use the mesh bbox to aim.
lo = m.vertices.min(0); hi = m.vertices.max(0)
feet_t  = [0.0, lo[1]+0.12*(hi[1]-lo[1]), 0.0]   # bottom of the model
hand_t  = [0.0, lo[1]+0.45*(hi[1]-lo[1]), 0.0]   # ~hip/hand height
feet = [render(a, feet_t, 0.95, 0.18, 620, 620, fov=np.pi/5) for a in (15, 60)]
hand = [render(a, hand_t, 1.0, 0.10, 620, 620, fov=np.pi/5) for a in (15, 60)]
top = np.concatenate(feet, axis=1)
bot = np.concatenate(hand, axis=1)
Image.fromarray(np.concatenate([top, bot], axis=0)).save(outp+"_detail.png")
print("wrote", outp+"_detail.png", "(top row=feet, bottom row=hands/arms)")
