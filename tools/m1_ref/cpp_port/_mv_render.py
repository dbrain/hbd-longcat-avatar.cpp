# Bright, even, IBL-ish render approximating model-viewer "neutral" environment (the owner's compare.html view).
import os,sys; os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import numpy as np,trimesh,pyrender; from PIL import Image
g=trimesh.load(sys.argv[1],process=False)
g=trimesh.util.concatenate(tuple(g.geometry.values())) if hasattr(g,'geometry') else g
g.vertices-=g.vertices.mean(0); g.vertices*=1.0/np.abs(g.vertices).max()
pm=pyrender.Mesh.from_trimesh(g,smooth=True)
def R(yaw):
    sc=pyrender.Scene(bg_color=[0.05,0.06,0.07,1],ambient_light=[0.55,0.55,0.55]);sc.add(pm)
    c=pyrender.PerspectiveCamera(yfov=np.pi/3.2);y=np.radians(yaw)
    eye=np.array([2.3*np.sin(y),0.25,2.3*np.cos(y)]);up=np.array([0,1,0.])
    f=-eye/np.linalg.norm(eye);s=np.cross(f,up);s/=np.linalg.norm(s);u=np.cross(s,f)
    P=np.eye(4);P[:3,0]=s;P[:3,1]=u;P[:3,2]=-f;P[:3,3]=eye;sc.add(c,pose=P)
    # even fill from several directions (approx neutral IBL)
    for d in [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0.3),(0,-1,0.3)]:
        dl=pyrender.DirectionalLight(intensity=1.2); M=np.eye(4)
        dd=np.array(d,float); dd/=np.linalg.norm(dd)
        zaxis=-dd; xaxis=np.cross([0,1,0.001],zaxis); xaxis/=np.linalg.norm(xaxis); yaxis=np.cross(zaxis,xaxis)
        M[:3,0]=xaxis;M[:3,1]=yaxis;M[:3,2]=zaxis; sc.add(dl,pose=M)
    r=pyrender.OffscreenRenderer(520,720);col,_=r.render(sc);r.delete();return col
Image.fromarray(np.concatenate([R(0),R(20)],1)).save(sys.argv[2]);print("wrote",sys.argv[2])
