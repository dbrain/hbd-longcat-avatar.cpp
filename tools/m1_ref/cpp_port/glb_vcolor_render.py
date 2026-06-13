import struct,json,numpy as np,os,sys,trimesh
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
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
trimesh.util.attach_to_log() if False else None
import importlib; fr=importlib.import_module('front_render')  # noqa
m=trimesh.Trimesh(vertices=V,faces=F,vertex_colors=(np.clip(C,0,1)*255).astype(np.uint8),process=False)
m.export('_tmp_vc.glb')
import subprocess; subprocess.run([sys.executable,'front_render.py','_tmp_vc.glb',sys.argv[2]])
