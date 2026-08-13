#!/usr/bin/env python
# Side-by-side render: C++/ggml E2E mesh (top) vs Python fp16 golden (bottom), 3 angles.
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image, ImageDraw

def load_ply(path):
    with open(path,'rb') as f: data=f.read()
    he=data.index(b'end_header\n')+len(b'end_header\n')
    head=data[:he].decode('ascii','replace')
    nv=int([l for l in head.splitlines() if l.startswith('element vertex')][0].split()[-1])
    v=np.frombuffer(data,dtype='<f4',count=nv*3,offset=he).reshape(nv,3).astype(np.float64)
    # faces
    off=he+nv*12; faces=[]
    fbuf=np.frombuffer(data,dtype=np.uint8,offset=off)
    # list uchar int: 1 + 3*4 = 13 bytes/face
    nf=(len(fbuf))//13
    rec=fbuf[:nf*13].reshape(nf,13)
    fi=rec[:,1:].copy().view('<i4').reshape(nf,3)
    return trimesh.Trimesh(vertices=v, faces=fi, process=False)

mine = load_ply("miku_geometry_e2e.ply")
gv=np.load("../../sparse_spike/golden_stages/stage5_mesh/vertices.npy").astype(np.float64)
gf=np.load("../../sparse_spike/golden_stages/stage5_mesh/faces.npy").astype(np.int64)
gold = trimesh.Trimesh(vertices=gv, faces=gf, process=False)

# shared transform (center each by mean, scale by mine's extent) so scale is comparable
c = mine.vertices.mean(0); scale = 1.0/np.abs(mine.vertices - c).max()
def prep(m, col):
    mm = m.copy(); mm.vertices = (mm.vertices - mm.vertices.mean(0)) * scale
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=col, metallicFactor=0.05,
                                             roughnessFactor=0.7, doubleSided=True)
    return pyrender.Mesh.from_trimesh(mm, material=mat, smooth=True)

pm_mine = prep(mine, [0.74,0.77,0.82,1])
pm_gold = prep(gold, [0.82,0.75,0.70,1])

def render(pm, yaw_deg):
    sc = pyrender.Scene(bg_color=[0.06,0.06,0.08,1], ambient_light=[0.3,0.3,0.3]); sc.add(pm)
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3.6)
    yaw=np.radians(yaw_deg); eye=np.array([2.4*np.sin(yaw),0.35,2.4*np.cos(yaw)])
    up=np.array([0,1,0]); f=-eye/np.linalg.norm(eye); s=np.cross(f,up); s/=np.linalg.norm(s); u=np.cross(s,f)
    pose=np.eye(4); pose[:3,0]=s; pose[:3,1]=u; pose[:3,2]=-f; pose[:3,3]=eye
    sc.add(cam,pose=pose); sc.add(pyrender.DirectionalLight(color=[1,1,1],intensity=4.5),pose=pose)
    r=pyrender.OffscreenRenderer(560,680); color,_=r.render(sc); r.delete(); return color

angles=[0,40,90]
top=np.concatenate([render(pm_mine,a) for a in angles],axis=1)
bot=np.concatenate([render(pm_gold,a) for a in angles],axis=1)
full=np.concatenate([top,bot],axis=0)
img=Image.fromarray(full); d=ImageDraw.Draw(img)
d.text((12,10), "C++/ggml E2E (mine, fp32)", fill=(180,200,230))
d.text((12, top.shape[0]+10), "Python golden (fp16)", fill=(230,200,180))
img.save("miku_compare_render.png"); print("wrote miku_compare_render.png", full.shape)
