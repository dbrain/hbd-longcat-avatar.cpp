#!/usr/bin/env python3
# Bank the tex DECODER's ACTUAL per-up-level subdivision (the encoder's channel2spatial cache that
# decode_tex_slat consumes) to debug build_guide_subs. Hooks SparseChannel2Spatial.forward and records,
# per call, the COARSE input coords + the 8-bit child occupancy reconstructed from (idx, subidx).
# Output: tex_goldens/dec_sub{L}_coords.npy [Nc,4], dec_sub{L}_bits.npy [Nc,8] (uint8), L=0..3.
import os, sys
os.environ.setdefault("NVIDIA_TF32_OVERRIDE","0"); os.environ.setdefault("ATTN_BACKEND","sdpa")
sys.path.insert(0,"/mnt/hdd/3d/avatar-shootout/Pixal3D")
import numpy as np, torch, trimesh
from PIL import Image
from pixal3d.pipelines import Trellis2TexturingPipeline
from pixal3d.modules.sparse.spatial import spatial2channel as s2c

G="/mnt/hdd/3d/avatar-shootout/tex_goldens"
MESH="/mnt/hdd/3d/avatar-shootout/_shootout_out/miku_lowpoly.glb"
IMG="/mnt/hdd/3d/avatar-shootout/_shootout_out/miku_clean_pose.png"
pipe=Trellis2TexturingPipeline.from_pretrained("/mnt/hdd/pixal3d_tex/trellis2_4b","_texturing_pipeline_local.json"); pipe.to("cuda")
scene=trimesh.load(MESH,process=False)
m=trimesh.util.concatenate([g for g in scene.geometry.values()]) if hasattr(scene,'geometry') else scene
m=trimesh.Trimesh(vertices=np.asarray(m.vertices),faces=np.asarray(m.faces),process=False)

rec=[]
_orig=s2c.SparseChannel2Spatial.forward
def hook(self, x, subdivision=None):
    cache=x.get_spatial_cache(f'channel2spatial_{self.factor}')
    coarse=x.coords.detach().cpu().numpy().copy()
    if cache is not None:
        _,idx,subidx=cache
        idx=idx.detach().cpu().numpy(); subidx=subidx.detach().cpu().numpy()
        bits=np.zeros((coarse.shape[0],8),np.uint8); bits[idx,subidx]=1
        rec.append((coarse,bits,'cache'))
    else:
        rec.append((coarse,None,'nocache'))
    return _orig(self,x,subdivision)
s2c.SparseChannel2Spatial.forward=hook

with torch.no_grad():
    img=pipe.preprocess_image(Image.open(IMG))
    torch.manual_seed(42)
    cond=pipe.get_cond([img],1024)
    shape_slat=pipe.encode_shape_slat(pipe.preprocess_mesh(m),1024)
    tex_slat=pipe.sample_tex_slat(cond,pipe.models['tex_slat_flow_model_1024'],shape_slat,{})
    rec.clear()  # only keep the DECODER's calls
    pbr=pipe.decode_tex_slat(tex_slat)

print(f"[subs] decoder made {len(rec)} C2S calls")
L=0
for coarse,bits,tag in rec:
    print(f"  call {L}: coarse N={coarse.shape[0]} {tag}", flush=True)
    if bits is not None:
        np.save(f"{G}/dec_sub{L}_coords.npy", np.ascontiguousarray(coarse.astype(np.int32)))
        np.save(f"{G}/dec_sub{L}_bits.npy", np.ascontiguousarray(bits))
    L+=1
print("[done]", flush=True)
