#!/usr/bin/env python
# Close-up textured render of the torso/skirt region (where seam-bleed shows). 2 yaws.
#   render_tex_zoom.py <glb> <out.png>
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image
path=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else "tex_zoom.png"
m=trimesh.load(path, process=False)
g=trimesh.util.concatenate(tuple(m.geometry.values())) if hasattr(m,'geometry') else m
print("loaded",path,"faces",len(g.faces))
g.vertices -= g.vertices.mean(0); s=1.0/np.abs(g.vertices).max(); g.vertices*=s
pm=pyrender.Mesh.from_trimesh(g, smooth=False)
lo=g.vertices.min(0); hi=g.vertices.max(0)
tgt=[0, lo[1]+0.50*(hi[1]-lo[1]), 0]   # hips/skirt height
def render(yaw):
    sc=pyrender.Scene(bg_color=[0.07,0.07,0.08,1], ambient_light=[0.4,0.4,0.4]); sc.add(pm)
    cam=pyrender.PerspectiveCamera(yfov=np.pi/6); y=np.radians(yaw); t=np.array(tgt,float)
    eye=t+np.array([0.9*np.sin(y),0.05,0.9*np.cos(y)]); up=np.array([0,1,0.])
    f=(t-eye); f/=np.linalg.norm(f); sx=np.cross(f,up); sx/=np.linalg.norm(sx); u=np.cross(sx,f)
    P=np.eye(4); P[:3,0]=sx; P[:3,1]=u; P[:3,2]=-f; P[:3,3]=eye; sc.add(cam,pose=P)
    sc.add(pyrender.DirectionalLight(intensity=3), pose=P)
    r=pyrender.OffscreenRenderer(640,640); c,_=r.render(sc); r.delete(); return c
Image.fromarray(np.concatenate([render(20),render(70)],axis=1)).save(out); print("wrote",out)
