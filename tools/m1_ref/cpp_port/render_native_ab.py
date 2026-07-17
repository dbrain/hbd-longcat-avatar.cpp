import os,sys; os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import numpy as np,trimesh,pyrender; from PIL import Image,ImageDraw
def load(p):
    g=trimesh.load(p,process=False)
    g=trimesh.util.concatenate(tuple(g.geometry.values())) if hasattr(g,'geometry') else g
    g.vertices=g.vertices-g.vertices.mean(0); g.vertices=g.vertices*(1.0/np.abs(g.vertices).max())
    return g
def R(g,yaw):
    pm=pyrender.Mesh.from_trimesh(g,smooth=False)
    sc=pyrender.Scene(bg_color=[0.07,0.07,0.08,1],ambient_light=[0.2,0.2,0.2]);sc.add(pm)
    c=pyrender.PerspectiveCamera(yfov=np.pi/3.2);y=np.radians(yaw)
    eye=np.array([2.3*np.sin(y),0.25,2.3*np.cos(y)]);up=np.array([0,1,0.])
    f=-eye/np.linalg.norm(eye);s=np.cross(f,up);s/=np.linalg.norm(s);u=np.cross(s,f)
    P=np.eye(4);P[:3,0]=s;P[:3,1]=u;P[:3,2]=-f;P[:3,3]=eye;sc.add(c,pose=P)
    sc.add(pyrender.DirectionalLight(intensity=2.4),pose=P)
    r=pyrender.OffscreenRenderer(420,620);col,_=r.render(sc);r.delete();return col
SRC="/mnt/hdd/3d/avatar-shootout/_shootout_out"
items=[("coarse input",f"{SRC}/char1_coarse.glb"),
       ("banked-ref (=golden)",f"{SRC}/us_e2e_native.glb"),
       ("FULL NATIVE",f"{SRC}/us_e2e_FULLNATIVE.glb")]
cols=[]
for label,p in items:
    g=load(p); strip=np.concatenate([R(g,0),R(g,25)],1)
    im=Image.fromarray(strip); d=ImageDraw.Draw(im)
    d.text((8,8),f"{label}  V={len(g.vertices)}",fill=(255,255,80)); cols.append(np.array(im))
Image.fromarray(np.concatenate(cols,0)).save(sys.argv[1]);print("wrote",sys.argv[1])
