#!/usr/bin/env python3
"""Official-Python M6 material diagnostic on a fixed supplied mesh.

The generic Trellis mesh encoder supplies the selected mesh's shape latent; Pixal3D's
official projection-conditioned texture sampler and decoder then generate the PBR
volume.  This deliberately isolates the M6 material branch from image-to-shape
sampling.  It is diagnostic-only: output is raw PBR, never an observed-image atlas.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PIXAL3D_ROOT = Path("/mnt/hdd/3d/avatar-shootout/Pixal3D")
TEX_WEIGHTS = Path("/mnt/hdd/pixal3d_tex/trellis2_4b")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def camera(path: Path) -> dict:
    fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {"camera_angle_x": float(fields["cam_angle_x_rad"]),
            "distance": float(fields["camera_distance"]),
            "mesh_scale": float(fields["mesh_scale"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path,
                    help="the frozen model-facing frame; it is not re-cropped")
    ap.add_argument("--camera", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    if any(not p.is_file() for p in (args.mesh, args.image, args.camera)):
        raise SystemExit("mesh, image, and camera must be files")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite immutable diagnostic: {args.out}")
    if not 0 <= args.seed <= 2_147_483_647:
        raise SystemExit("seed must be a non-negative 32-bit integer")

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PIXAL3D_TEX_NAF_TARGET", "512")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    sys.path.insert(0, str(PIXAL3D_ROOT))
    os.chdir(PIXAL3D_ROOT)
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines import Trellis2TexturingPipeline
    import inference

    args.out.mkdir(parents=True)
    cam = camera(args.camera)
    manifest = {
        "schema_version": 1,
        "purpose": "fixed-mesh official-Python projection-conditioned Pixal M6 material diagnostic",
        "mesh": str(args.mesh), "mesh_sha256": digest(args.mesh),
        "model_input": str(args.image), "model_input_sha256": digest(args.image),
        "camera": cam, "camera_sha256": digest(args.camera), "seed": args.seed,
        "shape_contract": "official Trellis2 mesh encoder on fixed mesh; Pixal M6 sampler/decoder; no image-to-shape run",
        "bake_contract": "raw PBR is rebaked separately with native_texture_rebake.sh; no image projection",
    }
    (args.out / "fixed_mesh_m6_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    scene = trimesh.load(args.mesh, process=False)
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) if hasattr(scene, "geometry") else scene
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    # Stage A: official mesh encoder.  Release it before the M6 projection model enters GPU memory.
    encoder = Trellis2TexturingPipeline.from_pretrained(str(TEX_WEIGHTS), "_texturing_pipeline_local.json")
    encoder.to("cuda")
    with torch.no_grad():
        shape = encoder.encode_shape_slat(encoder.preprocess_mesh(mesh), 1024)
        shape_coords = shape.coords.detach().cpu().int().numpy()
        shape_feats = shape.feats.detach().float().cpu().numpy()
    del shape, encoder
    torch.cuda.empty_cache()

    # Stage B: official Pixal M6 projection condition, flow, and decoder.  Sampling starts from
    # the supplied seed exactly once, matching the fixed-mesh native M6 diagnostic's texture seed.
    pipe = inference.init_pipeline("TencentARC/Pixal3D", device="cuda", low_vram=True)
    shape = SparseTensor(feats=torch.from_numpy(shape_feats).to("cuda"),
                         coords=torch.from_numpy(shape_coords).to("cuda"))
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        _, subs = pipe.decode_shape_slat(shape, 1024)
        cond = pipe.get_proj_cond_shape(pipe.image_cond_model_tex_1024, [image], shape.coords,
                                        camera_angle_x=cam["camera_angle_x"], distance=cam["distance"],
                                        mesh_scale=cam["mesh_scale"], grid_resolution_override=64)
        torch.manual_seed(args.seed)
        # Bank the exact M6 sampler boundary.  The native fixed-input oracle consumes
        # these tensors read-only, so it can distinguish a mesh-encoder deviation from
        # a projection conditioner/M6 flow deviation without another geometry run.
        real_randn, captured_noise = torch.randn, {}
        def capture_randn(*a, **kw):
            value = real_randn(*a, **kw)
            if value.ndim == 2 and value.shape == (shape.coords.shape[0], 32):
                captured_noise["value"] = value.detach().float().cpu().numpy().copy()
            return value
        torch.randn = capture_randn
        tex = pipe.sample_tex_slat(cond, pipe.models["tex_slat_flow_model_1024"], shape,
                                   {"steps": 12, "guidance_strength": 1.0,
                                    "guidance_rescale": 0.0, "rescale_t": 3.0})
        torch.randn = real_randn
        if "value" not in captured_noise:
            raise RuntimeError("failed to capture official M6 texture noise")
        pbr = pipe.decode_tex_slat(tex, subs)
        coords = np.ascontiguousarray(pbr.coords.detach().cpu().int().numpy().astype("<i4", copy=False))
        feats = np.ascontiguousarray(pbr.feats.detach().float().cpu().numpy().astype("<f4", copy=False))
    if coords.ndim != 2 or coords.shape[1] != 4 or feats.ndim != 2 or feats.shape[1] != 6 or len(coords) != len(feats):
        raise RuntimeError(f"invalid M6 PBR result: coords={coords.shape}, feats={feats.shape}")
    raw = args.out / "raw_pbr"; raw.mkdir()
    np.save(raw / "python_pbr_coords.npy", coords); np.save(raw / "python_pbr_feats.npy", feats)
    # The existing rebaker's --pbr-dir input is deliberately used: it applies the same fixed-mesh
    # coordinate transform and UV recipe as native texture_mesh_native's arbitrary-mesh diagnostic.
    rebake = args.out / "rebake_source"; rebake.mkdir()
    np.save(rebake / "native_pbr_coords.npy", coords); np.save(rebake / "native_pbr_feats.npy", feats)
    np.save(rebake / "native_shape_slat_coords.npy", shape_coords)
    np.save(rebake / "native_shape_slat_feats.npy", shape_feats)
    oracle = args.out / "m6_sampler_oracle"; oracle.mkdir()
    np.save(oracle / "shape_slat_coords.npy", shape_coords)
    np.save(oracle / "shape_slat_feats.npy", shape_feats)
    np.save(oracle / "cond_global.npy", cond["cond"]["global"].detach().float().cpu().numpy())
    np.save(oracle / "cond_proj.npy", cond["cond"]["proj"].feats.detach().float().cpu().numpy())
    np.save(oracle / "tex_noise.npy", captured_noise["value"])
    np.save(oracle / "tex_slat.npy", tex.feats.detach().float().cpu().numpy())
    metrics = {"shape_slat_voxels": int(len(shape_coords)), "pbr_voxels": int(len(coords)),
               "pbr_coords_sha256": digest(rebake / "native_pbr_coords.npy"),
               "pbr_feats_sha256": digest(rebake / "native_pbr_feats.npy")}
    (args.out / "fixed_mesh_m6_raw_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
