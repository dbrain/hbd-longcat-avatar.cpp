#!/usr/bin/env python3
# texture_mesh.py — the "direct texture path": retexture an EXISTING mesh (e.g. the UltraShape-refined
# pixal3d mesh) from a reference image, via TRELLIS.2-4B Trellis2TexturingPipeline. Host step (like the
# UltraShape docker + the rembg/MoGe cut-lines) — the native C++ equivalent (voxelizer + shape_slat_encoder
# + tex-DiT/decode/bake) is partly ported (encoder+voxelizer validated; tex-DiT wiring on an external mesh
# is the remaining native work, see VOXELIZER_PORT.md / SHAPE_ENC_ARCH.md).
#
#   texture_mesh.py <mesh.glb> <image.png> <out.glb> [resolution=512] [texsize=1024] [seed=42]
#
# resolution 512 fits the 3060 (low_vram); 1024 = finer voxel lattice / more VRAM. The pipeline:
# preprocess_image (rembg+crop) -> preprocess_mesh ([-0.5,0.5]^3 + y/z swap) -> encode_shape_slat
# (o_voxel dual-grid + shape encoder) -> sample tex_slat DiT (image+shape cond) -> decode 6-ch PBR voxel
# -> reproject/bake onto the mesh's UV atlas. Output: textured trimesh (baseColor + metallicRoughness).
import os, sys
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
PIXAL3D_ROOT = "/mnt/hdd/3d/avatar-shootout/Pixal3D"
TRELLIS2_DIR = "/mnt/hdd/pixal3d_tex/trellis2_4b"
sys.path.insert(0, PIXAL3D_ROOT)
import trimesh, numpy as np
from PIL import Image

def main():
    mesh_path = sys.argv[1]; image_path = sys.argv[2]; out_path = sys.argv[3]
    resolution = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    texsize   = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    seed      = int(sys.argv[6]) if len(sys.argv) > 6 else 42

    from pixal3d.pipelines import Trellis2TexturingPipeline
    print(f"[tex] loading Trellis2TexturingPipeline ({TRELLIS2_DIR}) ...", flush=True)
    pipe = Trellis2TexturingPipeline.from_pretrained(TRELLIS2_DIR, "_texturing_pipeline_local.json")
    pipe.to("cuda")

    scene = trimesh.load(mesh_path, process=False)
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()]) if hasattr(scene, "geometry") else scene
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    img = Image.open(image_path)
    print(f"[tex] mesh V={len(mesh.vertices)} F={len(mesh.faces)}  img={img.size}  res={resolution} texsize={texsize} seed={seed}", flush=True)

    out_mesh = pipe.run(mesh, img, seed=seed, resolution=resolution, texture_size=texsize, preprocess_image=True)
    out_mesh.export(out_path)
    print(f"[tex] wrote {out_path}", flush=True)

if __name__ == "__main__":
    main()
