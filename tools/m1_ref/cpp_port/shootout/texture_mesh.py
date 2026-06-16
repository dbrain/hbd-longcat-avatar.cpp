#!/usr/bin/env python
# Native mesh texturing (NOT reproject): texture a PROVIDED mesh from a reference image via the pixal3d
# repo's Trellis2TexturingPipeline. The mesh is voxelized + encoded (shape_slat_encoder) so the texture
# flow generates PBR at the mesh's OWN voxels -> bake is 1:1, no closest-point snap, no thin-feature bleed.
# Python ONLY because this wraps the unported encoder/flow (the C++ end goal ports just the shape encoder
# onto the existing C++ tex DiT/dec/bake). Run inside the pixal3d-tex Docker image (PYTHONPATH=pixal3d repo).
#   python texture_mesh.py --weights <dir w/ texturing_pipeline.json+ckpts> --mesh m.glb --image rgba.png --out o.glb
# Notes: feed an RGBA image (preprocess skips rembg -> no RMBG-2.0 download). --dino redirects the gated
# facebook/dinov3 to a local mirror dir. --golden dumps shape_slat/tex_slat for the C++ encoder port.
import argparse, json, os
import numpy as np, torch, trimesh
from PIL import Image
from pixal3d.pipelines import Trellis2TexturingPipeline

# pure inference: disable autograd globally so the sparse U-Net encoder doesn't retain activations for a
# backward that never comes (that retention is what OOMs the shape encoder at resolution 1024 on a 12GB 3060).
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True, help='dir containing texturing_pipeline.json + ckpts/')
ap.add_argument('--mesh', required=True)
ap.add_argument('--image', required=True, help='RGBA reference (skips rembg)')
ap.add_argument('--out', required=True)
ap.add_argument('--resolution', type=int, default=1024, choices=[512, 1024])
ap.add_argument('--texture_size', type=int, default=2048)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--dino', default=None, help='local DINOv3 dir (redirects gated facebook/dinov3-vitl16)')
ap.add_argument('--golden', default=None, help='dir to dump shape_slat/tex_slat goldens (for C++ port)')
args = ap.parse_args()

# redirect the image-cond model to a local DINOv3 mirror (facebook/dinov3-vitl16 is gated)
config_file = 'texturing_pipeline.json'
if args.dino:
    cfg = json.load(open(os.path.join(args.weights, config_file)))
    cfg['args']['image_cond_model']['args']['model_name'] = args.dino
    config_file = '_texturing_pipeline_local.json'
    json.dump(cfg, open(os.path.join(args.weights, config_file), 'w'))

print(f'[tex] loading Trellis2TexturingPipeline from {args.weights}', flush=True)
pipe = Trellis2TexturingPipeline.from_pretrained(args.weights, config_file=config_file)
pipe.low_vram = True            # 3060: keep modules on CPU, page in per stage
pipe.to('cuda')

m = trimesh.load(args.mesh, process=False)
m = trimesh.util.concatenate(list(m.geometry.values())) if isinstance(m, trimesh.Scene) else m
print(f'[tex] mesh {args.mesh}: {len(m.vertices)} v / {len(m.faces)} f', flush=True)
img = Image.open(args.image)
print(f'[tex] image {args.image}: {img.size} {img.mode} (RGBA skips rembg)', flush=True)

# optional goldens for the C++ encoder port: dump the encoded shape_slat (the new model's output)
if args.golden:
    os.makedirs(args.golden, exist_ok=True)
    img_p = pipe.preprocess_image(img); mesh_p = pipe.preprocess_mesh(m)
    torch.manual_seed(args.seed)
    ss = pipe.encode_shape_slat(mesh_p, args.resolution)
    np.save(os.path.join(args.golden, 'shape_slat_feats.npy'), ss.feats.detach().float().cpu().numpy())
    np.save(os.path.join(args.golden, 'shape_slat_coords.npy'), ss.coords.detach().int().cpu().numpy())
    print(f'[tex][golden] shape_slat: {tuple(ss.feats.shape)} feats / {tuple(ss.coords.shape)} coords -> {args.golden}', flush=True)

out = pipe.run(m, img, seed=args.seed, resolution=args.resolution, texture_size=args.texture_size)
out.export(args.out)
print(f'[tex] wrote {args.out}', flush=True)
