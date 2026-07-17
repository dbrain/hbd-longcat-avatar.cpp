#!/usr/bin/env python3
# Render a textured+rigged GLB: row1 = textured surface (3 yaws), row2 = wireframe-ish skeleton joints
# over the silhouette. Supersampled (EGL has no MSAA). Usage: _render_rigged.py <glb> <out.png>
import os, sys; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
from PIL import Image

glb = sys.argv[1]; out = sys.argv[2]
sc_in = trimesh.load(glb, process=False)
geos = list(sc_in.geometry.values()) if hasattr(sc_in, "geometry") else [sc_in]
mesh = geos[0]
# normalize to unit
v = mesh.vertices.copy(); c = v.mean(0); v -= c; s = 1.0/np.abs(v).max(); mesh.vertices = v*s
SS = 2; W, H = 360*SS, 560*SS

def cam_pose(yaw, dist=2.4, y=0.1):
    yr = np.radians(yaw); eye = np.array([dist*np.sin(yr), y, dist*np.cos(yr)]); up=np.array([0,1,0.])
    f=-eye/np.linalg.norm(eye); r=np.cross(f,up); r/=np.linalg.norm(r); u=np.cross(r,f)
    P=np.eye(4); P[:3,0]=r; P[:3,1]=u; P[:3,2]=-f; P[:3,3]=eye; return P

def render_tex(yaw):
    sc=pyrender.Scene(bg_color=[0.08,0.08,0.09,1], ambient_light=[0.35]*3)
    pm=pyrender.Mesh.from_trimesh(mesh, smooth=True)
    for p in pm.primitives: p.material.doubleSided=True
    sc.add(pm); P=cam_pose(yaw)
    sc.add(pyrender.PerspectiveCamera(yfov=np.pi/3.2), pose=P)
    sc.add(pyrender.DirectionalLight(intensity=3.0), pose=P)
    r=pyrender.OffscreenRenderer(W,H); col,_=r.render(sc); r.delete(); return col

# skeleton: pull joints/IBM from the glTF skin if present
def render_skel(yaw):
    sc=pyrender.Scene(bg_color=[0.08,0.08,0.09,1], ambient_light=[0.6]*3)
    gray=mesh.copy(); gray.visual=trimesh.visual.ColorVisuals(gray, vertex_colors=[120,120,130,120])
    pm=pyrender.Mesh.from_trimesh(gray, smooth=True)
    for p in pm.primitives: p.material.doubleSided=True; p.material.alphaMode='BLEND'
    sc.add(pm)
    # joints from dump (normalized rig frame == our mesh frame after normalize)
    try:
        J=np.load('/tmp/skintokens_e2e/gen_joints.npy'); Jp=np.load('/tmp/skintokens_e2e/gen_parents.npy')
        # joints are in the rig's normalized frame; our mesh got re-normalized to ~same. scale match best-effort.
        sm=pyrender.Mesh.from_trimesh(trimesh.creation.uv_sphere(radius=0.02), poses=[np.eye(4)]*0) if False else None
        sph=trimesh.creation.uv_sphere(radius=0.018); sph.visual.vertex_colors=[255,80,80,255]
        poses=[np.eye(4) for _ in range(len(J))]
        for i,jp in enumerate(J): poses[i][:3,3]=jp
        sc.add(pyrender.Mesh.from_trimesh(sph, poses=np.array(poses)))
        # bones as thin cylinders
        segs=[]
        for i,par in enumerate(Jp):
            if par>=0 and par<len(J): segs.append((J[par],J[i]))
        if segs:
            segs=np.array(segs);
            bl=pyrender.Mesh.from_trimesh(_bones(segs), smooth=False)
            sc.add(bl)
    except Exception as e:
        print('skel overlay skipped:', e)
    P=cam_pose(yaw); sc.add(pyrender.PerspectiveCamera(yfov=np.pi/3.2), pose=P)
    sc.add(pyrender.DirectionalLight(intensity=3.0), pose=P)
    r=pyrender.OffscreenRenderer(W,H); col,_=r.render(sc); r.delete(); return col

def _bones(segs):
    mlist=[]
    for a,b in segs:
        d=b-a; L=np.linalg.norm(d)
        if L<1e-6: continue
        cyl=trimesh.creation.cylinder(radius=0.006, height=L, sections=6)
        cyl.visual.vertex_colors=[90,160,255,255]
        z=np.array([0,0,1.]); ax=np.cross(z,d/L); s=np.linalg.norm(ax); cth=np.dot(z,d/L)
        T=np.eye(4)
        if s>1e-6:
            ax/=s; K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
            T[:3,:3]=np.eye(3)+s*K+(1-cth)*K@K
        T[:3,3]=(a+b)/2; cyl.apply_transform(T); mlist.append(cyl)
    return trimesh.util.concatenate(mlist) if mlist else trimesh.Trimesh()

top=np.concatenate([render_tex(0), render_tex(35), render_tex(180)],1)
bot=np.concatenate([render_skel(0), render_skel(35), render_skel(180)],1)
im=np.concatenate([top,bot],0)
Image.fromarray(im).resize((360*3,560*2), Image.LANCZOS).save(out)
print("wrote", out, "verts", len(mesh.vertices))
