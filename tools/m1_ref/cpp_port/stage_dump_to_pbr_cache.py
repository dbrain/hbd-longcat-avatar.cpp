#!/usr/bin/env python3
"""Convert an opt-in raw image_to_rig stage dump into the native PBR cache ABI.

This is diagnostic-only.  It enables a CPU atlas inspection of a no-refine
geometry trace without rerunning image generation; normal image_to_rig runs
write this cache directly after refinement.
"""
import argparse
import struct
from pathlib import Path

import numpy as np


def write_blob(path: Path, array: np.ndarray) -> None:
    data = np.ascontiguousarray(array).tobytes(order='C')
    path.write_bytes(struct.pack('=Q', len(data)) + data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage-dump', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--resolution', required=True, type=int)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f'refusing to overwrite {args.out}')
    if args.resolution not in (512, 1024):
        raise SystemExit('resolution must be 512 or 1024')
    coords_path = args.stage_dump / 'native_pbr_coords_i32.bin'
    feats_path = args.stage_dump / 'native_pbr_feats_f32.bin'
    coords = np.fromfile(coords_path, dtype='<i4')
    feats = np.fromfile(feats_path, dtype='<f4')
    if not coords.size or coords.size % 4 or not feats.size or feats.size % 6:
        raise SystemExit('invalid raw PBR dump shape')
    coords = coords.reshape(-1, 4)
    feats = feats.reshape(-1, 6)
    if len(coords) != len(feats):
        raise SystemExit('PBR coordinate/feature count mismatch')
    args.out.mkdir(parents=True)
    write_blob(args.out / 'pbr_coords.bin', coords)
    write_blob(args.out / 'pbr_feats.bin', feats)
    write_blob(args.out / 'resolution.bin', np.asarray([args.resolution], dtype='<i4'))
    print(f'cached {len(coords)} PBR voxels -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
