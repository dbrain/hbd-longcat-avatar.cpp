#!/usr/bin/env python
# render_hands_auto.py — auto-locate the hands and render a tight uniform-vs-adaptive comparison.
# Heuristic: in a standing figure the hands are the lowest-Y clusters of the LEFT/RIGHT arm columns
# (max |x| extremity near mid-low height, excluding the hair mass). We find the two side extremities,
# then render a tight clay crop of each from a few yaws, uniform (a0) beside adaptive (a1.0).
#   render_hands_auto.py d10   -> /tmp/im_adapt/HANDS_d10.png
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image, ImageDraw

G = sys.argv[1] if len(sys.argv) > 1 else "d10"
DIR = "/tmp/im_adapt"

def load_norm(path):
    m = trimesh.load(path, process=False)
    v = np.asarray(m.vertices, float); v -= v.mean(0); v /= np.abs(v).max()
    m.vertices = v
    return m

# locate hands on the UNIFORM mesh (same normalization for both, so the target transfers).
mref = load_norm(f"{DIR}/{G}_a0_f60k.obj")
V = np.asarray(mref.vertices)
# arm/hand band: below the shoulders, above the feet; pick the side extremities.
# robust: among verts with y in the lower-mid band, take the ones with the largest |x| on each side.
ymin, ymax = V[:,1].min(), V[:,1].max()
band = (V[:,1] > ymin + 0.30*(ymax-ymin)) & (V[:,1] < ymin + 0.62*(ymax-ymin))
Vb = V[band]
targets = []
for side in (+1, -1):
    sb = Vb[np.sign(Vb[:,0]) == side]
    if len(sb) < 50: continue
    # hand = the extreme-|x| tail on this side, then its centroid (z toward +z front)
    thr = np.quantile(np.abs(sb[:,0]), 0.985)
    tip = sb[np.abs(sb[:,0]) >= thr]
    targets.append(tip.mean(0))
print(f"[{G}] hand targets:", [np.round(t,3).tolist() for t in targets])

mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.72,0.74,0.78,1.0],
        metallicFactor=0.0, roughnessFactor=0.55, doubleSided=True)
def mesh_pm(path): return pyrender.Mesh.from_trimesh(load_norm(path), material=mat, smooth=False)

def render(pm, target, yaw_deg, dist, W, H, fov=np.pi/7):
    sc = pyrender.Scene(bg_color=[0.06,0.06,0.07,1.0], ambient_light=[0.13,0.13,0.14]); sc.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=fov); yaw = np.radians(yaw_deg)
    tgt = np.array(target,float); up=np.array([0,1,0.0])
    eye = tgt + np.array([dist*np.sin(yaw), 0.05*dist, dist*np.cos(yaw)])
    f=(tgt-eye); f/=np.linalg.norm(f); s=np.cross(f,up); s/=np.linalg.norm(s); u2=np.cross(s,f)
    pose=np.eye(4); pose[:3,0]=s; pose[:3,1]=u2; pose[:3,2]=-f; pose[:3,3]=eye; sc.add(cam,pose=pose)
    for d,inten in (([-2,2.4,2.4],3.4),([2,0.3,1.5],1.2)):
        kp=np.eye(4); kd=np.array(d,float); kp[:3,3]=kd; kd/=np.linalg.norm(kd)
        kf=-kd; ks=np.cross(kf,up); ks/=np.linalg.norm(ks); ku=np.cross(ks,kf)
        kp[:3,0]=ks; kp[:3,1]=ku; kp[:3,2]=-kf
        sc.add(pyrender.DirectionalLight(color=[1,1,1], intensity=inten), pose=kp)
    r=pyrender.OffscreenRenderer(W,H); col,_=r.render(sc); r.delete(); return Image.fromarray(col)

pm0 = mesh_pm(f"{DIR}/{G}_a0_f60k.obj")
pm1 = mesh_pm(f"{DIR}/{G}_a1.0_f60k.obj")
rows = []
for ti, tgt in enumerate(targets):
    for yaw in (20, 60, 110):
        a = render(pm0, tgt, yaw, 0.32, 360, 480)
        b = render(pm1, tgt, yaw, 0.32, 360, 480)
        def lab(im,t):
            d=ImageDraw.Draw(im); d.rectangle([0,0,im.width,16],fill=(20,20,24)); d.text((4,3),t,fill=(235,235,240)); return im
        pair = Image.new("RGB",(a.width+b.width,a.height),(20,20,24))
        pair.paste(lab(a,f"UNIFORM hand{ti} yaw{yaw}"),(0,0)); pair.paste(lab(b,f"ADAPT1.0 hand{ti} yaw{yaw}"),(a.width,0))
        rows.append(pair)
W=max(r.width for r in rows); Hh=sum(r.height for r in rows)
out=Image.new("RGB",(W,Hh),(20,20,24)); y=0
for r in rows: out.paste(r,(0,y)); y+=r.height
out.save(f"{DIR}/HANDS_{G}.png"); print("wrote", f"{DIR}/HANDS_{G}.png", out.size)
