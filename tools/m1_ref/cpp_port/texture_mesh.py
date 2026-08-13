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

    # Production never calls this Python bridge.  When native/C++ parity needs diagnosing, retain
    # each reference-stage tensor under an explicit opt-in directory so the mismatch can be located
    # (image conditioning, shape encode, texture flow, decoder, or UV bake) rather than guessed from
    # a final render.  The normal `pipe.run` path remains byte-for-byte untouched.
    audit_dir = os.environ.get("TEXTURE_AUDIT_DUMP", "")
    if audit_dir:
        import torch
        torch.set_grad_enabled(False)
        os.makedirs(audit_dir, exist_ok=True)
        proc_img = pipe.preprocess_image(img)
        proc_img.save(os.path.join(audit_dir, "python_proc_image.png"))
        proc_mesh = pipe.preprocess_mesh(mesh)
        np.save(os.path.join(audit_dir, "python_mesh_vertices.npy"), proc_mesh.vertices.astype(np.float32))
        np.save(os.path.join(audit_dir, "python_mesh_faces.npy"), proc_mesh.faces.astype(np.int64))
        torch.manual_seed(seed)
        cond = pipe.get_cond([proc_img], 512 if resolution == 512 else 1024)
        np.save(os.path.join(audit_dir, "python_cond.npy"), cond["cond"].detach().float().cpu().numpy())
        shape_slat = pipe.encode_shape_slat(proc_mesh, resolution)
        np.save(os.path.join(audit_dir, "python_shape_coords.npy"), shape_slat.coords.detach().cpu().numpy())
        np.save(os.path.join(audit_dir, "python_shape_feats.npy"), shape_slat.feats.detach().float().cpu().numpy())
        tex_model = pipe.models['tex_slat_flow_model_512'] if resolution == 512 else pipe.models['tex_slat_flow_model_1024']
        tex_slat = pipe.sample_tex_slat(cond, tex_model, shape_slat)
        np.save(os.path.join(audit_dir, "python_tex_coords.npy"), tex_slat.coords.detach().cpu().numpy())
        np.save(os.path.join(audit_dir, "python_tex_feats.npy"), tex_slat.feats.detach().float().cpu().numpy())
        # The audit keeps CPU copies above, so release the encoder/conditioning tensors before
        # bringing the decoder onto the 12 GB 3060. This changes only diagnostic residency, not
        # the values passed into decode_tex_slat.
        del cond, shape_slat, tex_model
        torch.cuda.empty_cache()
        pbr_voxel = pipe.decode_tex_slat(tex_slat)
        np.save(os.path.join(audit_dir, "python_pbr_coords.npy"), pbr_voxel.coords.detach().cpu().numpy())
        np.save(os.path.join(audit_dir, "python_pbr_feats.npy"), pbr_voxel.feats.detach().float().cpu().numpy())
        out_mesh = pipe.postprocess_mesh(proc_mesh, pbr_voxel, resolution, texsize)
    else:
        out_mesh = pipe.run(mesh, img, seed=seed, resolution=resolution, texture_size=texsize, preprocess_image=True)
    out_mesh.export(out_path)
    print(f"[tex] wrote {out_path}", flush=True)

if __name__ == "__main__":
    main()
