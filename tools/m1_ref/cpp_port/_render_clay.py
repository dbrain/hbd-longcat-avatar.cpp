#!/usr/bin/env python
# Robust clay render of a (possibly Scene) GLB: full front + tight zoom on each hand region.
# _render_clay.py in.glb out.png
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image, ImageDraw

path = sys.argv[1]; out = sys.argv[2]
s = trimesh.load(path, process=False)
m = s.dump(concatenate=True) if isinstance(s, trimesh.Scene) else s
v = np.asarray(m.vertices, float); c = v.mean(0); v -= c; sc = np.abs(v).max(); v /= sc; m.vertices = v
print("loaded", path, "verts", len(m.vertices), "faces", len(m.faces))
m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=[200,200,205,255])
pm = pyrender.Mesh.from_trimesh(m, smooth=True)

def render(eye, tgt, W=560, H=720, fov=np.pi/4):
    scene = pyrender.Scene(bg_color=[0.08,0.08,0.10,1.0], ambient_light=[0.3,0.3,0.3])
    scene.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=fov)
    eye=np.array(eye,float); tgt=np.array(tgt,float); up=np.array([0,1,0.])
    f=(tgt-eye); f/=np.linalg.norm(f); r=np.cross(f,up); r/=np.linalg.norm(r); u2=np.cross(r,f)
    pose=np.eye(4); pose[:3,0]=r; pose[:3,1]=u2; pose[:3,2]=-f; pose[:3,3]=eye
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=4.0), pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.5),
              pose=np.array([[1,0,0,2],[0,1,0,3],[0,0,1,2],[0,0,0,1.]]))
    rr=pyrender.OffscreenRenderer(W,H); col,_=rr.render(scene); rr.delete(); return col

# Hands: lowest-Y points among the extreme +/-X columns (exclude hair via |z| filter)
def hand_center(sign):
    sel = (np.sign(v[:,0])==sign) & (np.abs(v[:,0])>0.45) & (np.abs(v[:,2])<0.35)
    if sel.sum()<20: sel=(np.sign(v[:,0])==sign)&(np.abs(v[:,0])>0.4)
    pts=v[sel]; lo=pts[pts[:,1] < np.percentile(pts[:,1],25)]
    return lo.mean(0) if len(lo) else pts.mean(0)

full = render([0,0.1,2.6],[0,0,0])
hands=[]
for sign,name in ((-1,"L"),(1,"R")):
    hc = hand_center(sign)
    eye = hc + np.array([0,0,0.9])
    img = render(eye, hc, W=400, H=520, fov=np.pi/5)
    im=Image.fromarray(img); d=ImageDraw.Draw(d:=im); ImageDraw.Draw(im).text((8,8),f"hand {name} zoom",fill=(255,255,0))
    hands.append(np.asarray(im))
fim=Image.fromarray(full); ImageDraw.Draw(fim).text((8,8),"decimated full",fill=(255,255,0))
# stack: full beside the two hand zooms
hcat=np.concatenate(hands,axis=0)
fpad=np.asarray(fim);
# match heights
import numpy as _np
target_h=hcat.shape[0]
from PIL import Image as _I
fr=_np.asarray(_I.fromarray(fpad).resize((int(fpad.shape[1]*target_h/fpad.shape[0]),target_h)))
Image.fromarray(np.concatenate([fr,hcat],axis=1)).save(out)
print("wrote",out)
