import struct,json,numpy as np,os,sys
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import trimesh,pyrender; from PIL import Image
d=open(sys.argv[1],'rb').read(); off=12; ch={}
while off<len(d):
    ln,t=struct.unpack_from('<II',d,off); off+=8; ch[t]=d[off:off+ln]; off+=ln
js=json.loads(ch[0x4E4F534A]); b=ch[0x004E4942]
def acc(i):
    a=js['accessors'][i];bv=js['bufferViews'][a['bufferView']];o=bv.get('byteOffset',0)+a.get('byteOffset',0)
    nc={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];dt={5126:np.float32,5125:np.uint32,5123:np.uint16}[a['componentType']]
    return np.frombuffer(b,dt,a['count']*nc,o).reshape(a['count'],nc)
p=js['meshes'][0]['primitives'][0]
V=acc(p['attributes']['POSITION']).astype(float);F=acc(p['indices']).reshape(-1,3);C=acc(p['attributes']['COLOR_0'])[:,:3]
g=trimesh.Trimesh(vertices=V,faces=F,vertex_colors=(np.clip(C,0,1)*255).astype(np.uint8),process=False)
g.vertices-=g.vertices.mean(0); g.vertices*=1.0/np.abs(g.vertices).max()
pm=pyrender.Mesh.from_trimesh(g,smooth=True)
def R(yaw):
    sc=pyrender.Scene(bg_color=[0.05,0.06,0.07,1],ambient_light=[0.75,0.75,0.75]);sc.add(pm)
    c=pyrender.PerspectiveCamera(yfov=np.pi/3.2);y=np.radians(yaw)
    eye=np.array([2.3*np.sin(y),0.25,2.3*np.cos(y)]);up=np.array([0,1,0.])
    f=-eye/np.linalg.norm(eye);s=np.cross(f,up);s/=np.linalg.norm(s);u=np.cross(s,f)
    P=np.eye(4);P[:3,0]=s;P[:3,1]=u;P[:3,2]=-f;P[:3,3]=eye;sc.add(c,pose=P)
    for dd in [(0,0,1),(1,0,0.3),(-1,0,0.3),(0,1,0.5)]:
        dl=pyrender.DirectionalLight(intensity=1.0); M=np.eye(4); v=np.array(dd,float);v/=np.linalg.norm(v)
        z=-v; x=np.cross([0,1,0.001],z);x/=np.linalg.norm(x);yy=np.cross(z,x); M[:3,0]=x;M[:3,1]=yy;M[:3,2]=z; sc.add(dl,pose=M)
    r=pyrender.OffscreenRenderer(520,720);col,_=r.render(sc);r.delete();return col
Image.fromarray(np.concatenate([R(0),R(20)],1)).save(sys.argv[2]);print("wrote",sys.argv[2])
