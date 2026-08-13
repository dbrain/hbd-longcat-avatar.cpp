#!/usr/bin/env python3
"""Capture Pixal3D's official projection-conditioned M6 material boundary.

This is deliberately a diagnostic, not a replacement production texture path.  It
uses the official Python image pipeline with a recorded model-facing image, camera,
and seed, retains every M6 boundary, and writes its decoded PBR volume in the native
image-to-rig cache format.  The latter can then be CPU-rebaked onto a *separately
selected, unchanged* native refined mesh, so a final atlas comparison does not mix
material generation with UV/LOD variance.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PIXAL3D_ROOT = Path("/mnt/hdd/3d/avatar-shootout/Pixal3D")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def camera_contract(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    required = ("cam_angle_x_rad", "camera_distance", "mesh_scale")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"camera contract lacks {', '.join(missing)}: {path}")
    return {
        "camera_angle_x": float(values["cam_angle_x_rad"]),
        "distance": float(values["camera_distance"]),
        "mesh_scale": float(values["mesh_scale"]),
    }


def write_cache_blob(path: Path, array) -> None:
    data = array.tobytes(order="C")
    # image_to_rig's cache uses native size_t (eight bytes on the supported host).
    path.write_bytes(struct.pack("=Q", len(data)) + data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path,
                    help="already model-facing RGB frame; do not preprocess again")
    ap.add_argument("--camera", required=True, type=Path,
                    help="selected geometry cache camera_provenance.txt")
    ap.add_argument("--out", required=True, type=Path,
                    help="new immutable diagnostic directory")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--model", default="TencentARC/Pixal3D")
    args = ap.parse_args()

    if not args.image.is_file() or not args.camera.is_file():
        raise SystemExit("image and camera contract must exist")
    if args.seed < 0 or args.seed > 2_147_483_647:
        raise SystemExit("seed must be a non-negative 32-bit integer")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite immutable diagnostic: {args.out}")
    args.out.mkdir(parents=True)

    # These settings mirror the retained Python 1024 reference; the outer shell
    # pins CUDA to physical PCI GPU 0 and serialises it with the native lane.
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PIXAL3D_TEX_NAF_TARGET", "512")
    os.environ.setdefault("PIXAL3D_MAX_TOKENS", "49152")
    os.environ.setdefault("PIXAL3D_HR_RES_FLOOR", "1024")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    sys.path.insert(0, str(PIXAL3D_ROOT))
    sys.path.insert(0, str(ROOT / "tools" / "sparse_spike"))
    os.chdir(PIXAL3D_ROOT)

    import numpy as np
    import torch
    from PIL import Image
    import golden_stage_hook
    import inference

    camera = camera_contract(args.camera)
    metadata = {
        "schema_version": 1,
        "purpose": "official Python Pixal projection-conditioned M6 material diagnostic; no delivery geometry is written or modified",
        "input": str(args.image),
        "input_sha256": sha256(args.image),
        "camera_contract": str(args.camera),
        "camera_contract_sha256": sha256(args.camera),
        "camera": camera,
        "seed": args.seed,
        "pipeline_type": "1024_cascade",
        "preprocess_image": False,
        "texture_sampler": {"steps": 12, "guidance_strength": 1.0,
                            "guidance_rescale": 0.0, "rescale_t": 3.0},
        "attention_backend": os.environ["ATTN_BACKEND"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "note": "PBR cache is for a controlled CPU rebake on the frozen selected mesh; it is not an image projection.",
    }
    (args.out / "reference_material_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    shutil.copy2(args.image, args.out / "model_input.png")
    shutil.copy2(args.camera, args.out / "camera_provenance.txt")

    # The hook records the official Python boundaries as NPY files before any
    # final GLB/UV export can obscure them.
    golden_stage_hook.install(out_dir=str(args.out / "stages"))
    pipe = inference.init_pipeline(args.model, device="cuda", low_vram=True)
    image = Image.open(args.image).convert("RGB")
    torch.manual_seed(args.seed)
    shape_params = {"steps": 12, "guidance_strength": 7.5,
                    "guidance_rescale": 0.5, "rescale_t": 3.0}
    ss_params = {"steps": 12, "guidance_strength": 7.5,
                 "guidance_rescale": 0.7, "rescale_t": 5.0}
    tex_params = {"steps": 12, "guidance_strength": 1.0,
                  "guidance_rescale": 0.0, "rescale_t": 3.0}
    with torch.no_grad():
        pipe.run(image, camera_params=camera, seed=args.seed,
                 sparse_structure_sampler_params=ss_params,
                 shape_slat_sampler_params=shape_params,
                 tex_slat_sampler_params=tex_params,
                 preprocess_image=False, return_latent=True,
                 pipeline_type="1024_cascade", max_num_tokens=49152)

    pbr_dir = args.out / "stages" / "stage4b_pbr"
    coords_path, feats_path = pbr_dir / "pbr_coords.npy", pbr_dir / "pbr_feats.npy"
    if not coords_path.is_file() or not feats_path.is_file():
        raise RuntimeError("official M6 capture did not emit decoded PBR boundary")
    coords = np.ascontiguousarray(np.load(coords_path).astype("<i4", copy=False))
    feats = np.ascontiguousarray(np.load(feats_path).astype("<f4", copy=False))
    if coords.ndim != 2 or coords.shape[1] != 4 or feats.ndim != 2 or feats.shape[1] != 6:
        raise RuntimeError(f"invalid official PBR layout: coords={coords.shape}, feats={feats.shape}")
    if len(coords) != len(feats) or not len(coords):
        raise RuntimeError("official PBR coordinate/feature count mismatch")

    raw = args.out / "raw_pbr"
    raw.mkdir()
    np.save(raw / "python_pbr_coords.npy", coords)
    np.save(raw / "python_pbr_feats.npy", feats)
    # This is intentionally the same read-only cache ABI used by image_to_rig,
    # permitting the existing CPU-only rebaker to use precisely the same atlas
    # settings as the native material comparison.
    cache = args.out / "pbr_cache"
    cache.mkdir()
    write_cache_blob(cache / "pbr_coords.bin", coords)
    write_cache_blob(cache / "pbr_feats.bin", feats)
    (cache / "resolution.bin").write_bytes(struct.pack("=Q", 4) + struct.pack("=i", 1024))
    metrics = {
        "pbr_voxels": int(len(coords)),
        "pbr_coords_sha256": sha256(cache / "pbr_coords.bin"),
        "pbr_feats_sha256": sha256(cache / "pbr_feats.bin"),
        "raw_coords_sha256": sha256(raw / "python_pbr_coords.npy"),
        "raw_feats_sha256": sha256(raw / "python_pbr_feats.npy"),
    }
    (args.out / "python_m6_raw_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
