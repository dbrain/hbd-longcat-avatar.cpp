#!/usr/bin/env python
# "make textures betterer" — 2D comparison (no GL). Extract the baseColor atlas from a textured GLB,
# apply principled color transforms, write each as a variant atlas PNG + a variant GLB (for the owner's
# browser viewer), and a labeled side-by-side montage with valid-region mean RGB. Owner judges.
#   tex_variants_2d.py in.glb out_montage.png
import sys, numpy as np, trimesh
from PIL import Image, ImageDraw

path = sys.argv[1]; out = sys.argv[2] if len(sys.argv)>2 else "_tex_variants_2d.png"
s = trimesh.load(path, process=False)
m = s.to_geometry() if isinstance(s, trimesh.Scene) else s
tex = m.visual.material.baseColorTexture
arr0 = np.asarray(tex.convert("RGBA"), np.float32)/255.0
valid = arr0[...,3] > 0.5

def lin_to_srgb(c): return np.where(c<=0.0031308, 12.92*c, 1.055*np.power(np.clip(c,0,1),1/2.4)-0.055)
def xform(a, kind):
    a=a.copy(); rgb=a[...,:3]
    if   kind=="baseline": pass
    elif kind=="srgb":     rgb=lin_to_srgb(rgb)
    elif kind=="gamma0.80":rgb=np.power(np.clip(rgb,0,1),0.80)
    elif kind=="bright1.30":rgb=np.clip(rgb*1.30,0,1)
    a[...,:3]=np.clip(rgb,0,1); return a

VARIANTS=["baseline","srgb","gamma0.80","bright1.30"]
panels=[]
for k in VARIANTS:
    a=xform(arr0,k)
    mean=(a[...,:3][valid].mean(0)*255).round(1)
    img=Image.fromarray((a*255+0.5).astype(np.uint8),"RGBA").convert("RGB")
    # variant GLB for the browser viewer
    mm=m.copy()
    mm.visual=trimesh.visual.TextureVisuals(uv=m.visual.uv,
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray((a*255+0.5).astype(np.uint8),"RGBA"),
                                                     metallicFactor=0.0, roughnessFactor=1.0))
    gname=f"_texvar_{k}.glb"; mm.export(gname)
    # panel: shrink atlas to 512 for the montage
    p=img.resize((512,512)); d=ImageDraw.Draw(p)
    d.rectangle([0,0,512,40],fill=(0,0,0)); d.text((6,4),f"{k}",fill=(255,255,0))
    d.text((6,22),f"valid-mean RGB {tuple(mean)}",fill=(180,255,180))
    panels.append(np.asarray(p)); print(k,"mean",mean,"->",gname)
Image.fromarray(np.concatenate(panels,axis=1)).save(out)
print("wrote",out)
