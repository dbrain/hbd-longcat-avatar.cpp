import os,sys; os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import numpy as np,trimesh,pyrender; from PIL import Image
g=trimesh.load(sys.argv[1],process=False)
g=trimesh.util.concatenate(tuple(g.geometry.values())) if hasattr(g,'geometry') else g
g.vertices-=g.vertices.mean(0); g.vertices*=1.0/np.abs(g.vertices).max()
pm=pyrender.Mesh.from_trimesh(g,smooth=True)
lo=g.vertices.min(0); hi=g.vertices.max(0)
# target the lower hair tail (wide teal sweep), front-ish view
tgt=np.array([0.35*(hi[0]-lo[0]), lo[1]+0.30*(hi[1]-lo[1]), 0.0])
def R(yaw):
    sc=pyrender.Scene(bg_color=[0.07,0.07,0.08,1],ambient_light=[0.35,0.35,0.35]);sc.add(pm)
    c=pyrender.PerspectiveCamera(yfov=np.pi/6);y=np.radians(yaw)
    eye=tgt+np.array([0.8*np.sin(y),0.05,0.8*np.cos(y)]);up=np.array([0,1,0.])
    f=tgt-eye;f/=np.linalg.norm(f);s=np.cross(f,up);s/=np.linalg.norm(s);u=np.cross(s,f)
    P=np.eye(4);P[:3,0]=s;P[:3,1]=u;P[:3,2]=-f;P[:3,3]=eye;sc.add(c,pose=P)
    sc.add(pyrender.DirectionalLight(intensity=2.6),pose=P)
    r=pyrender.OffscreenRenderer(640,640);col,_=r.render(sc);r.delete();return col
Image.fromarray(np.concatenate([R(8),R(-25)],1)).save(sys.argv[2]);print("wrote",sys.argv[2])
