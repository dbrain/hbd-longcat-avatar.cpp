#!/usr/bin/env python
# Finger-survival bake-off: render several meshes' HAND region from a front-quarter camera (NOT the
# auto-locator that kept framing Miku's twintails). Each mesh is normalized identically; we render a
# full front view + tight crops of the lower-left/right (hip-height hands) so finger separation is
# judgeable. Grey clay, off-axis key light to reveal webbing/gaps/shards.
#   render_fingers_cmp.py OUT.png  LABEL1=mesh1  LABEL2=mesh2  ...
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image, ImageDraw

args = sys.argv[1:]
OUT = args[0]; pairs = args[1:]

def load(path):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    v = np.asarray(m.vertices, float); v -= (v.min(0)+v.max(0))/2.0; v /= np.abs(v).max()
    m.vertices = v
    return m

def cam_pose(yaw_deg, pitch_deg, dist, target):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    eye = target + dist*np.array([np.cos(p)*np.sin(y), np.sin(p), np.cos(p)*np.cos(y)])
    fwd = (target-eye); fwd/=np.linalg.norm(fwd)
    right = np.cross(fwd,[0,1,0.]); right/=np.linalg.norm(right)
    up = np.cross(right,fwd)
    M = np.eye(4); M[:3,0]=right; M[:3,1]=up; M[:3,2]=-fwd; M[:3,3]=eye
    return M

def render(m, yaw, pitch, dist, target, W, H, fov=np.pi/7):
    sc = pyrender.Scene(bg_color=[0.06,0.06,0.07,1.0], ambient_light=[0.18,0.18,0.2])
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.74,0.76,0.80,1.0],
            metallicFactor=0.0, roughnessFactor=0.5, doubleSided=True)
    sc.add(pyrender.Mesh.from_trimesh(m, material=mat, smooth=False))
    sc.add(pyrender.PerspectiveCamera(yfov=fov, aspectRatio=W/H), pose=cam_pose(yaw,pitch,dist,target))
    key = pyrender.DirectionalLight(color=[1,1,1], intensity=4.0)
    sc.add(key, pose=cam_pose(yaw+40, pitch+25, dist, target))
    sc.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.0), pose=cam_pose(yaw-50, pitch+10, dist, target))
    r = pyrender.OffscreenRenderer(W, H)
    col,_ = r.render(sc); r.delete()
    return Image.fromarray(col)

# views: full front, + two hand crops (front-quarter, aimed at hip-height sides where hands hang).
# target y≈ -0.15..-0.25 (hip), x≈ ±0.35 (side). tuned for a standing figure normalized to [-1,1].
VIEWS = [
    ("full",   0,   5, 2.6, np.array([0,0,0.0]),       560, 760),
    ("L-hand", 30,  0, 0.95, np.array([ 0.33,-0.18,0.1]), 420, 520),
    ("R-hand",-30,  0, 0.95, np.array([-0.33,-0.18,0.1]), 420, 520),
]
rows = []
for label, mesh in (p.split("=",1) for p in pairs):
    m = load(mesh)
    nf = len(m.faces)
    tiles = [render(m, y,pi,d,t,W,H) for (_,y,pi,d,t,W,H) in VIEWS]
    h = max(t.height for t in tiles); tiles=[t.resize((int(t.width*h/t.height),h)) for t in tiles]
    row = Image.new("RGB", (sum(t.width for t in tiles)+200, h), (15,15,17))
    x=200
    for t in tiles: row.paste(t,(x,0)); x+=t.width
    d=ImageDraw.Draw(row); d.text((8,8), label, fill=(240,240,120)); d.text((8,28), f"{nf//1000}k faces", fill=(180,180,180))
    rows.append(row); print(f"rendered {label}: {mesh} ({nf} faces)")
W=max(r.width for r in rows)
cv=Image.new("RGB",(W,sum(r.height for r in rows)+4*len(rows)),(0,0,0)); y=0
for r in rows: cv.paste(r,(0,y)); y+=r.height+4
cv.save(OUT); print("WROTE", OUT)
