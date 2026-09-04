#!/usr/bin/env python3
"""Compare native Pixal stage blobs against a hooked Python reference run.

This is a diagnostic gate, not part of the production pipeline.  It compares sparse
latents/PBR by integer coordinate, so differences in sparse ordering cannot masquerade
as material damage.  A low overlap localises a structural decode mismatch; high overlap
with poor values localises sampling/weight/precision drift before atlas baking.
"""
import argparse
import json
from pathlib import Path

import numpy as np


# Current `texture_mesh_native --dump-dir` layout.  The reference directory is the direct
# `capture_texture_goldens.py` output, so this tool is usable immediately after a native run
# without staging or renaming blobs.
STAGES = (
    ("shape_slat", "native_shape_slat_coords.npy", "native_shape_slat_feats.npy",
     "shape_slat_coords.npy", "shape_slat_feats.npy", 32),
    ("tex_slat", "native_shape_slat_coords.npy", "native_tex_slat_feats.npy",
     "tex_slat_coords.npy", "tex_slat_feats.npy", 32),
    ("pbr", "native_pbr_coords.npy", "native_pbr_feats.npy",
     "pbr_coords.npy", "pbr_feats.npy", 6),
)


def raw(path: Path, dtype, channels: int) -> np.ndarray:
    """Read a native opt-in stage blob and fail loudly on a truncated dump."""
    a = np.fromfile(path, dtype=dtype)
    if a.size % channels:
        raise ValueError(f"{path}: {a.size} scalars is not divisible by {channels}")
    return a.reshape(-1, channels)


def dense_compare(native: np.ndarray, reference: np.ndarray) -> dict:
    # Python stage hooks retain the batch dimension while the C++ boundary stores
    # the single generated sample directly.  Do not turn an otherwise exact
    # global-condition comparison into a spurious shape mismatch.
    if reference.ndim == native.ndim + 1 and reference.shape[0] == 1:
        reference = reference[0]
    if native.shape != reference.shape:
        return {"status": "shape_mismatch", "native_shape": list(native.shape),
                "reference_shape": list(reference.shape)}
    a = native.astype(np.float64, copy=False)
    b = reference.astype(np.float64, copy=False)
    d = a - b
    return {
        "status": "compared", "shape": list(a.shape),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))),
        "cosine": float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
    }


def compare_same_image_native_dump(native: Path, reference: Path) -> dict:
    """Compare image-to-rig's raw F32 boundaries to a hooked Pixal3D run.

    The native stage dump deliberately uses raw binaries so production never pays NPY
    serialization.  This adapter keeps the oracle read-only and avoids a fragile
    manual conversion step in the runbook.
    """
    report = {"layout": "native-image-to-rig-stage-dump"}
    ncond = native / "native_cond_coords_i32.bin"
    # A deliberately stopped stage-1 capture has no later M3b condition dump,
    # but is still a valid (and much cheaper) oracle input.
    if not ncond.is_file() and not (native / "native_stage1_global_f32.bin").is_file():
        # This is a texture-mesh dump (NPY stages), not an image-to-rig raw
        # binary dump.  Return an empty sentinel so main selects the NPY path.
        return {}

    # Stage 1 is the causally first geometry boundary.  These blobs are
    # optional because they are substantial and exist only for a focused
    # native/Python investigation; production runs never write them.
    stage1_specs = (
        ("global", "native_stage1_global_f32.bin", "stage1_cond/global.npy", 5 * 1024),
        ("projection", "native_stage1_proj_f32.bin", "stage1_cond/proj.npy", 4096 * 1024),
        ("noise", "native_stage1_noise_f32.bin", "stage1_noise/noise.npy", 8 * 16 * 16 * 16),
        ("z_s", "native_stage1_z_s_f32.bin", "stage1_ssdec/z_s.npy", 8 * 16 * 16 * 16),
        ("decoder_logits", "native_stage1_ss_logits_f32.bin", "stage1_ssdec/ss_logits.npy", 64 * 64 * 64),
    )
    stage1 = {}
    for name, native_name, ref_name, width in stage1_specs:
        npth, rpth = native / native_name, reference / ref_name
        if not npth.is_file() or not rpth.is_file():
            stage1[name] = {"status": "missing", "files": [str(p) for p in (npth, rpth) if not p.is_file()]}
            continue
        n = raw(npth, "<f4", width).reshape(-1)
        r = np.load(rpth).astype(np.float32, copy=False).reshape(-1)
        stage1[name] = dense_compare(n, r)
    ncoords, rcoords = native / "native_stage1_coords_i32.bin", reference / "stage1_out/coords.npy"
    if ncoords.is_file() and rcoords.is_file():
        nc = raw(ncoords, "<i4", 4)
        rc = np.load(rcoords).astype(np.int32, copy=False).reshape(-1, 4)
        nk, rk = keyed(nc), keyed(rc)
        common = int(np.intersect1d(nk, rk, assume_unique=False).size)
        stage1["coords"] = {
            "status": "compared", "native_voxels": len(nk), "reference_voxels": len(rk),
            "common_voxels": common, "native_coverage": common / len(nk) if len(nk) else 0.0,
            "reference_coverage": common / len(rk) if len(rk) else 0.0,
        }
    else:
        stage1["coords"] = {"status": "missing", "files": [str(p) for p in (ncoords, rcoords) if not p.is_file()]}
    report["stage1"] = stage1

    # Exact texture conditioning: same final shape lattice, 5 global DINO tokens,
    # and the per-voxel projection features consumed by the projection texture DiT.
    if ncond.is_file():
      try:
        nc = raw(ncond, "<i4", 4)
        npj = raw(native / "native_cond_proj_f32.bin", "<f4", 2048)
        rc = np.load(reference / "stage4_cond/proj_coords.npy").astype(np.int32, copy=False).reshape(-1, 4)
        rpj = np.load(reference / "stage4_cond/proj_feats.npy").astype(np.float32, copy=False).reshape(-1, 2048)
        report["projection_condition"] = {"status": "compared", **compare("projection_condition", nc, npj, rc, rpj)}
        ng = raw(native / "native_cond_global_f32.bin", "<f4", 1024)
        rg = np.load(reference / "stage4_cond/global.npy").astype(np.float32, copy=False)
        report["projection_global"] = dense_compare(ng, rg)
      except (OSError, ValueError, FileNotFoundError) as e:
        report["projection_condition"] = {"status": "missing_or_invalid", "detail": str(e)}

    specs = (
        ("shape_slat", "native_shape_coords_i32.bin", "native_shape_slat_f32.bin",
         "stage3b_out/shape_slat_coords.npy", "stage3b_out/shape_slat_feats.npy", 32),
        ("tex_slat", "native_shape_coords_i32.bin", "native_tex_slat_f32.bin",
         "stage4_out/tex_slat_coords.npy", "stage4_out/tex_slat_feats.npy", 32),
        ("pbr", "native_pbr_coords_i32.bin", "native_pbr_feats_f32.bin",
         "stage4b_pbr/pbr_coords.npy", "stage4b_pbr/pbr_feats.npy", 6),
    )
    for name, nc_name, nf_name, rc_name, rf_name, channels in specs:
        try:
            nc = raw(native / nc_name, "<i4", 4)
            nf = raw(native / nf_name, "<f4", channels)
            rc = np.load(reference / rc_name).astype(np.int32, copy=False).reshape(-1, 4)
            rf = np.load(reference / rf_name).astype(np.float32, copy=False).reshape(-1, channels)
            report[name] = {"status": "compared", **compare(name, nc, nf, rc, rf)}
        except (OSError, ValueError, FileNotFoundError) as e:
            report[name] = {"status": "missing_or_invalid", "detail": str(e)}
    return report


def keyed(coords: np.ndarray) -> np.ndarray:
    """A compact, sortable view of [N,4] int32 coordinates."""
    a = np.ascontiguousarray(coords.astype("<i4", copy=False))
    return a.view(np.dtype((np.void, a.dtype.itemsize * 4))).reshape(-1)


def compare(name, native_coords, native_feats, ref_coords, ref_feats):
    nk, rk = keyed(native_coords), keyed(ref_coords)
    _, ni, ri = np.intersect1d(nk, rk, assume_unique=False, return_indices=True)
    n_native, n_ref, n_common = len(nk), len(rk), len(ni)
    result = {
        "native_voxels": n_native,
        "reference_voxels": n_ref,
        "common_voxels": n_common,
        "native_coverage": n_common / n_native if n_native else 0.0,
        "reference_coverage": n_common / n_ref if n_ref else 0.0,
    }
    if not n_common:
        return result
    a = native_feats[ni].astype(np.float64, copy=False)
    b = ref_feats[ri].astype(np.float64, copy=False)
    delta = a - b
    result.update({
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "cosine": float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
    })
    # A 2,048-entry diagnostic is neither readable nor useful in the report;
    # retain per-channel detail for compact latent/PBR tensors only.
    if a.shape[1] <= 64:
        result["per_channel_mae"] = [float(x) for x in np.mean(np.abs(delta), axis=0)]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--shape-reference", type=Path,
                    help="optional directory containing only shape_slat_coords/feats.npy")
    ap.add_argument("--shape-reference-coords", type=Path)
    ap.add_argument("--shape-reference-feats", type=Path)
    ap.add_argument("--out", type=Path, help="optional JSON report path")
    args = ap.parse_args()

    report = compare_same_image_native_dump(args.native, args.reference)
    if report.get("layout") == "native-image-to-rig-stage-dump":
        payload = json.dumps(report, indent=2, sort_keys=True)
        print(payload)
        if args.out:
            args.out.write_text(payload + "\n")
        return

    report = {}
    for name, nc, nf, rc, rf, channels in STAGES:
        nc, nf = args.native / nc, args.native / nf
        if name == "shape_slat" and args.shape_reference_coords and args.shape_reference_feats:
            rc, rf = args.shape_reference_coords, args.shape_reference_feats
        else:
            ref_base = args.shape_reference if name == "shape_slat" and args.shape_reference else args.reference
            rc, rf = ref_base / rc, ref_base / rf
        missing = [str(p) for p in (nc, nf, rc, rf) if not p.is_file()]
        if missing:
            report[name] = {"status": "missing", "files": missing}
            continue
        native_coords = np.load(nc).astype(np.int32, copy=False).reshape(-1, 4)
        native_feats = np.load(nf).astype(np.float32, copy=False).reshape(-1, channels)
        ref_coords = np.load(rc).astype(np.int32, copy=False).reshape(-1, 4)
        ref_feats = np.load(rf).astype(np.float32, copy=False).reshape(-1, channels)
        if len(native_coords) != len(native_feats) or len(ref_coords) != len(ref_feats):
            raise ValueError(f"{name}: coordinate/feature count mismatch")
        report[name] = {"status": "compared", **compare(name, native_coords, native_feats, ref_coords, ref_feats)}

    # Generic arbitrary-mesh texturing uses full DINO image-token cross attention, rather than
    # Pixal's camera-projection conditioning.  Report that input separately: if it drifts, later
    # sampler differences are expected; if it is close, inspect encoder/flow/decoder next.
    nc = args.native / "native_tex_cross_cond.npy"
    rc = args.reference / "cond_cond.npy"
    missing = [str(p) for p in (nc, rc) if not p.is_file()]
    if missing:
        report["conditioning"] = {"status": "missing", "files": missing}
    else:
        native = np.load(nc).astype(np.float64, copy=False)
        reference = np.load(rc).astype(np.float64, copy=False)
        if native.shape != reference.shape:
            report["conditioning"] = {"status": "shape_mismatch", "native_shape": list(native.shape),
                                       "reference_shape": list(reference.shape)}
        else:
            delta = native - reference
            report["conditioning"] = {
                "status": "compared",
                "shape": list(native.shape),
                "mae": float(np.mean(np.abs(delta))),
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "max_abs": float(np.max(np.abs(delta))),
                "cosine": float(np.sum(native * reference) /
                                (np.linalg.norm(native) * np.linalg.norm(reference) + 1e-30)),
            }

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.out:
        args.out.write_text(payload + "\n")


if __name__ == "__main__":
    main()
