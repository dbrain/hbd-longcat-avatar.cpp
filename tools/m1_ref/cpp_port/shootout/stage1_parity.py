#!/usr/bin/env python3
# Stage-1 (sparse-structure) parity: native image_to_rig dumps vs the pinned Python reference.
# Metrics mirror the stage1-*-parity.json methodology: coord overlap (native/reference coverage)
# and z_s cosine.  These are the acceptance numbers for the production matte fix.
import argparse, json
from pathlib import Path
import numpy as np


def load_native_coords(d: Path):
    b = np.fromfile(d / "native_stage1_coords_i32.bin", dtype="<i4").reshape(-1, 4)
    return b[:, 1:]  # drop batch col -> (z,y,x) or (x,y,z); consistent with reference cols[-3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, type=Path, help="dir with native_stage1_*.bin")
    ap.add_argument("--ref", required=True, type=Path, help="python_pinned_reference dir")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    nc = load_native_coords(a.native)
    pc = np.load(a.ref / "stage1_out" / "coords.npy")[:, -3:].astype(np.int32)
    ns = set(map(tuple, nc.tolist()))
    ps = set(map(tuple, pc.tolist()))
    common = len(ns & ps)
    rep = {
        "native_voxels": len(ns),
        "reference_voxels": len(ps),
        "common_voxels": common,
        "native_coverage": common / len(ns) if ns else 0.0,
        "reference_coverage": common / len(ps) if ps else 0.0,
    }

    nzf = a.native / "native_stage1_z_s_f32.bin"
    if nzf.exists():
        nz = np.fromfile(nzf, dtype="<f4").astype(np.float64)
        pz = np.load(a.ref / "stage1_ssdec" / "z_s.npy").astype(np.float64).ravel()
        if nz.size == pz.size:
            cz, cp = nz, pz
            rep["z_s"] = {
                "cosine": float(np.sum(cz * cp) / (np.linalg.norm(cz) * np.linalg.norm(cp) + 1e-30)),
                "mae": float(np.mean(np.abs(cz - cp))),
                "rmse": float(np.sqrt(np.mean((cz - cp) ** 2))),
                "shape": int(nz.size),
            }
        else:
            rep["z_s"] = {"status": f"size mismatch native={nz.size} ref={pz.size}"}
    print(json.dumps(rep, indent=2, sort_keys=True))
    if a.out:
        a.out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
