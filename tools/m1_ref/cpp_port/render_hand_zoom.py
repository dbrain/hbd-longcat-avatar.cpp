#!/usr/bin/env python
# Tight hand close-ups — the hip-height detail crop in render_geo_detail was too loose to judge finger
# separation. This zooms in on the hand region (sides, ~hip height) from several offsets/yaws so at
# least one frame frames a hand cleanly. Geometry-only grey clay (off-axis key light reveals gaps).
#   render_hand_zoom.py <mesh> <out_prefix>   -> <out_prefix>.png (grid of hand close-ups)
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image

path = sys.argv[1]; outp = sys.argv[2] if len(sys.argv) > 2 else "hand_zoom"
m = trimesh.load(path, process=False)
print("loaded", path, "verts", len(m.vertices), "faces", len(m.faces))
m.vertices -= m.vertices.mean(0)
m.vertices *= 1.0 / np.abs(m.vertices).max()
lo = m.vertices.min(0); hi = m.vertices.max(0); span = hi - lo

mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.72,0.74,0.78,1.0],
        metallicFactor=0.0, roughnessFactor=0.55, doubleSided=True)
pm = pyrender.Mesh.from_trimesh(m, material=mat, smooth=False)

def render(target, yaw_deg, dist, W, H, fov=np.pi/6):
    sc = pyrender.Scene(bg_color=[0.06,0.06,0.07,1.0], ambient_light=[0.13,0.13,0.14]); sc.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=fov); yaw = np.radians(yaw_deg)
    tgt = np.array(target, float); up = np.array([0,1,0.0])
    eye = tgt + np.array([dist*np.sin(yaw), 0.06*dist, dist*np.cos(yaw)])
    f = (tgt-eye); f/=np.linalg.norm(f); s=np.cross(f,up); s/=np.linalg.norm(s); u2=np.cross(s,f)
    pose=np.eye(4); pose[:3,0]=s; pose[:3,1]=u2; pose[:3,2]=-f; pose[:3,3]=eye; sc.add(cam,pose=pose)
    kp=np.eye(4); kd=np.array([-2.0,2.4,2.4]); kp[:3,3]=kd; kd=kd/np.linalg.norm(kd)
    kf=-kd; ks=np.cross(kf,up); ks/=np.linalg.norm(ks); ku=np.cross(ks,kf)
    kp[:3,0]=ks; kp[:3,1]=ku; kp[:3,2]=-kf
    sc.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.4), pose=kp)
    fp=np.eye(4); fd=np.array([2.0,0.3,1.5]); fp[:3,3]=fd; fd=fd/np.linalg.norm(fd)
    ff=-fd; fs=np.cross(ff,up); fs/=np.linalg.norm(fs); fu=np.cross(fs,ff)
    fp[:3,0]=fs; fp[:3,1]=fu; fp[:3,2]=-ff
    sc.add(pyrender.DirectionalLight(color=[0.7,0.75,0.85], intensity=1.2), pose=fp)
    r=pyrender.OffscreenRenderer(W,H); col,_=r.render(sc); r.delete(); return col

# Hands hang ~hip height in this pose. Sweep both sides (x = +/- fraction of half-span) at hip y,
# several yaws, tight dist. One frame should frame a hand with finger gaps if they exist.
hipy = lo[1] + 0.40*span[1]
hx   = 0.42*max(span[0],span[2])/2*2   # toward the side where a hand hangs
tiles=[]
for sx in (+hx*0.5, -hx*0.5):
    for yaw in (10, 45, 80):
        tiles.append(render([sx, hipy, 0.0], yaw, 0.55, 360, 440))
# 2 rows (left/right hand) x 3 yaws
rows=[np.concatenate(tiles[i*3:(i+1)*3],axis=1) for i in range(2)]
Image.fromarray(np.concatenate(rows,axis=0)).save(outp+".png")
print("wrote", outp+".png", "(row0 = +x side, row1 = -x side; yaws 10/45/80)")
