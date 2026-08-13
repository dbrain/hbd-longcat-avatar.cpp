# Offscreen render of a textured GLB, approximating model-viewer's "neutral" IBL — for quick
# colour/shape checks. (Real crack/quality judge is compare.html in model-viewer; pyrender hides UV seams.)
#   <venv>/python _mv_render.py <in.glb> <out.png>
#   venv = /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python  (system python3 lacks trimesh/pyrender)
# Brightness knobs (env):  AMB=ambient (default 0.55)   LINT=directional intensity (default 1.2)
#                          UNLIT=1 -> flat/unlit (high ambient, no directional = pure baseColor, like the source art)
#                          RES=WxH (default 520x720)    YAWS=comma list of degrees (default 0,20)
import os,sys; os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import numpy as np,trimesh,pyrender; from PIL import Image
AMB  = float(os.environ.get("AMB","0.55"))
LINT = float(os.environ.get("LINT","1.2"))
UNLIT= os.environ.get("UNLIT")
if UNLIT: AMB = max(AMB, 1.6)                       # flat: crank ambient, drop directional
W,H  = (int(x) for x in os.environ.get("RES","520x720").split("x"))
YAWS = [float(y) for y in os.environ.get("YAWS","0,20").split(",")]
g=trimesh.load(sys.argv[1],process=False)
g=trimesh.util.concatenate(tuple(g.geometry.values())) if hasattr(g,'geometry') else g
g.vertices-=g.vertices.mean(0); g.vertices*=1.0/np.abs(g.vertices).max()
pm=pyrender.Mesh.from_trimesh(g,smooth=True)
def R(yaw):
    sc=pyrender.Scene(bg_color=[0.05,0.06,0.07,1],ambient_light=[AMB,AMB,AMB]); sc.add(pm)
    c=pyrender.PerspectiveCamera(yfov=np.pi/3.2); y=np.radians(yaw)
    eye=np.array([2.3*np.sin(y),0.25,2.3*np.cos(y)]); up=np.array([0,1,0.])
    f=-eye/np.linalg.norm(eye); s=np.cross(f,up); s/=np.linalg.norm(s); u=np.cross(s,f)
    P=np.eye(4); P[:3,0]=s; P[:3,1]=u; P[:3,2]=-f; P[:3,3]=eye; sc.add(c,pose=P)
    if not UNLIT:                                   # even fill from several directions (approx neutral IBL)
        for d in [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0.3),(0,-1,0.3)]:
            dl=pyrender.DirectionalLight(intensity=LINT); M=np.eye(4)
            dd=np.array(d,float); dd/=np.linalg.norm(dd)
            z=-dd; x=np.cross([0,1,0.001],z); x/=np.linalg.norm(x); yy=np.cross(z,x)
            M[:3,0]=x; M[:3,1]=yy; M[:3,2]=z; sc.add(dl,pose=M)
    r=pyrender.OffscreenRenderer(W,H); col,_=r.render(sc); r.delete(); return col
Image.fromarray(np.concatenate([R(y) for y in YAWS],1)).save(sys.argv[2])
print(f"wrote {sys.argv[2]}  (AMB={AMB} LINT={LINT} UNLIT={bool(UNLIT)} RES={W}x{H} YAWS={YAWS})")
