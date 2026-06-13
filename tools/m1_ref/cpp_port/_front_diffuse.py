import os,sys; os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import numpy as np,trimesh,pyrender; from PIL import Image
g=trimesh.load(sys.argv[1],process=False)
g=trimesh.util.concatenate(tuple(g.geometry.values())) if hasattr(g,'geometry') else g
g.vertices-=g.vertices.mean(0); g.vertices*=1.0/np.abs(g.vertices).max()
pm=pyrender.Mesh.from_trimesh(g,smooth=False)
for p in pm.primitives:  # kill specular: full rough, zero metal
    p.material.roughnessFactor=1.0; p.material.metallicFactor=0.0; p.material.metallicRoughnessTexture=None
def R(yaw):
    sc=pyrender.Scene(bg_color=[0.07,0.07,0.08,1],ambient_light=[0.18,0.18,0.18]);sc.add(pm)
    c=pyrender.PerspectiveCamera(yfov=np.pi/3.2);y=np.radians(yaw)
    eye=np.array([2.3*np.sin(y),0.25,2.3*np.cos(y)]);up=np.array([0,1,0.])
    f=-eye/np.linalg.norm(eye);s=np.cross(f,up);s/=np.linalg.norm(s);u=np.cross(s,f)
    P=np.eye(4);P[:3,0]=s;P[:3,1]=u;P[:3,2]=-f;P[:3,3]=eye;sc.add(c,pose=P)
    sc.add(pyrender.DirectionalLight(intensity=2.2),pose=P)
    r=pyrender.OffscreenRenderer(520,720);col,_=r.render(sc);r.delete();return col
Image.fromarray(np.concatenate([R(0),R(20)],1)).save(sys.argv[2]);print("wrote",sys.argv[2])
