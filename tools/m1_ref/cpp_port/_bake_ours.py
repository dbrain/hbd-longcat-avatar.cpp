# Bake OUR dump's dense mesh + PBR with pyref's WORKING GPU pipeline (cumesh unwrap + nvdiffrast raster
# + flex_gemm grid_sample). Fast conformal UV (GPU xatlas) where C++ xatlas timed out.
import os,sys,time,numpy as np,torch,cv2
from PIL import Image
import trimesh, trimesh.visual, cumesh
import nvdiffrast.torch as dr
from flex_gemm.ops.grid_sample import grid_sample_3d
HERE=os.path.dirname(os.path.abspath(__file__)); RES=1024
TS=int(sys.argv[1]) if len(sys.argv)>1 else 2048
OUT=sys.argv[2] if len(sys.argv)>2 else "miku_ours_uv.glb"
LAYOUT={'base_color':slice(0,3),'metallic':slice(3,4),'roughness':slice(4,5),'alpha':slice(5,6)}
v=np.fromfile("dump_dense_v.bin",dtype=np.float32).reshape(-1,3)
f=np.fromfile("dump_dense_f.bin",dtype=np.int64).reshape(-1,3)
feats=np.fromfile("dump_pbr_f.bin",dtype=np.float32).reshape(-1,6)
coords=np.fromfile("dump_pbr_c.bin",dtype=np.int32).reshape(-1,4)
print(f"[ours] dense V={v.shape[0]} F={f.shape[0]} vol N={feats.shape[0]} TS={TS}")
vt=torch.from_numpy(v).float().cuda(); ft=torch.from_numpy(f.astype(np.int32)).int().cuda()
mesh=trimesh.Trimesh(vertices=v,faces=f,process=False); normals=mesh.vertex_normals
t0=time.time(); cm=cumesh.CuMesh(); cm.init(vt,ft)
vt,ft,uvt,vmap=cm.uv_unwrap(return_vmaps=True)
vt=vt.cuda();ft=ft.cuda();uvt=uvt.cuda()
print(f"[ours] cumesh unwrap: {vt.shape[0]} verts / {ft.shape[0]} tris ({time.time()-t0:.1f}s)")
verts=vt.cpu().numpy(); faces=ft.cpu().numpy(); uvs=uvt.cpu().numpy(); normals=normals[vmap.cpu().numpy()]
ctx=dr.RasterizeCudaContext()
uvr=torch.cat([uvt*2-1,torch.zeros_like(uvt[:,:1]),torch.ones_like(uvt[:,:1])],-1).unsqueeze(0)
rast,_=dr.rasterize(ctx,uvr,ft,resolution=[TS,TS]); mask=rast[0,...,3]>0
pos=dr.interpolate(vt.unsqueeze(0),rast,ft)[0][0]
feats_t=torch.from_numpy(feats).float().cuda(); coords_t=torch.from_numpy(coords).int().cuda()
attrs=torch.zeros(TS,TS,6,device='cuda')
attrs[mask]=grid_sample_3d(feats_t,coords_t,shape=torch.Size([1,6,RES,RES,RES]),grid=((pos[mask]+0.5)*RES).reshape(1,-1,3),mode='trilinear')
m=mask.cpu().numpy()
base=np.clip(attrs[...,LAYOUT['base_color']].cpu().numpy()*255,0,255).astype(np.uint8)
metal=np.clip(attrs[...,LAYOUT['metallic']].cpu().numpy()*255,0,255).astype(np.uint8)
rough=np.clip(attrs[...,LAYOUT['roughness']].cpu().numpy()*255,0,255).astype(np.uint8)
alpha=np.clip(attrs[...,LAYOUT['alpha']].cpu().numpy()*255,0,255).astype(np.uint8)
mi=(~m).astype(np.uint8)
base=cv2.inpaint(base,mi,3,cv2.INPAINT_TELEA); metal=cv2.inpaint(metal,mi,1,cv2.INPAINT_TELEA)[...,None]
rough=cv2.inpaint(rough,mi,1,cv2.INPAINT_TELEA)[...,None]; alpha=cv2.inpaint(alpha,mi,1,cv2.INPAINT_TELEA)[...,None]
mat=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(np.concatenate([base,alpha],-1)),
    baseColorFactor=np.array([255,255,255,255],np.uint8),
    metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metal),rough,metal],-1)),
    metallicFactor=1.0,roughnessFactor=1.0,alphaMode='OPAQUE',doubleSided=True)
uvs2=uvs.copy(); uvs2[:,1]=1-uvs2[:,1]
out=trimesh.Trimesh(vertices=verts,faces=faces,vertex_normals=normals,process=False,
    visual=trimesh.visual.TextureVisuals(uv=uvs2,material=mat))
out.export(os.path.join(HERE,OUT)); print(f"[ours] wrote {OUT} (valid mean {base[m].mean(0)})")
