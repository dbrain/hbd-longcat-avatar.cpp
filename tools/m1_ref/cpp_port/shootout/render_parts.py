#!/usr/bin/env python
# Render a P3-SAM part-segmented mesh preserving its per-part vertex colors, from several angles +
# tight hand crops, so we can judge whether HANDS isolate as their own part(s).
#   render_parts.py <colored.glb> <out.png>
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image, ImageDraw

path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "parts.png"
sc_in = trimesh.load(path, process=False)
m = trimesh.util.concatenate([g for g in sc_in.geometry.values()]) if isinstance(sc_in, trimesh.Scene) else sc_in
v = np.asarray(m.vertices, float); c = (v.min(0)+v.max(0))/2; v -= c; v /= np.abs(v).max(); m.vertices = v
# count distinct part colors
vc = np.asarray(m.visual.vertex_colors)[:, :3] if hasattr(m.visual, "vertex_colors") else None
nparts = len(np.unique(vc.reshape(-1,3), axis=0)) if vc is not None else -1
print(f"loaded {path}: verts={len(m.vertices)} faces={len(m.faces)} distinct_colors={nparts}")
pm = pyrender.Mesh.from_trimesh(m, smooth=False)

def cam_pose(yaw, pitch, dist, tgt):
    y,p=np.radians(yaw),np.radians(pitch); eye=tgt+dist*np.array([np.cos(p)*np.sin(y),np.sin(p),np.cos(p)*np.cos(y)])
    f=tgt-eye; f/=np.linalg.norm(f); r=np.cross(f,[0,1,0.]); r/=np.linalg.norm(r); u=np.cross(r,f)
    M=np.eye(4); M[:3,0]=r; M[:3,1]=u; M[:3,2]=-f; M[:3,3]=eye; return M

def render(yaw,pitch,dist,tgt,W,H,fov=np.pi/6):
    s=pyrender.Scene(bg_color=[0.07,0.07,0.08,1],ambient_light=[0.5,0.5,0.5]); s.add(pm)
    s.add(pyrender.PerspectiveCamera(yfov=fov,aspectRatio=W/H),pose=cam_pose(yaw,pitch,dist,tgt))
    s.add(pyrender.DirectionalLight(intensity=3),pose=cam_pose(yaw+30,pitch+20,dist,tgt))
    r=pyrender.OffscreenRenderer(W,H); col,_=r.render(s); r.delete(); return Image.fromarray(col)

# 6 axis-aligned faces (orientation-agnostic; mesh up-axis unknown) at a dist that fits the ~2-unit
# normalized figure (view span 2*dist*tan(fov/2) must exceed ~2.0 -> dist >= ~4).
D=4.6
views=[("+Z",0,0,D,[0,0,0],460,560),("-Z",180,0,D,[0,0,0],460,560),
       ("+X",90,0,D,[0,0,0],460,560),("-X",-90,0,D,[0,0,0],460,560),
       ("+Y_top",0,89,D,[0,0,0],460,560),("-Y_bot",0,-89,D,[0,0,0],460,560)]
tiles=[]
for name,y,p,d,t,W,H in views:
    im=render(y,p,d,np.array(t,float),W,H); dr=ImageDraw.Draw(im); dr.text((6,6),name,fill=(255,255,0)); tiles.append(im)
h=max(t.height for t in tiles); tiles=[t.resize((int(t.width*h/t.height),h)) for t in tiles]
cv=Image.new("RGB",(sum(t.width for t in tiles),h),(0,0,0)); x=0
for t in tiles: cv.paste(t,(x,0)); x+=t.width
cv.save(out); print("WROTE",out)
