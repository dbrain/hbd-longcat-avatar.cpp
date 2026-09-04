#!/usr/bin/env python3
"""Capture only Pixal's first image-conditioner boundary for native parity work.

The input must already be the model-facing RGB frame.  This intentionally does
not call preprocess_image or any diffusion flow, so it is a short, immutable
diagnostic that isolates DINOv3 and the camera projection from sampling.
"""
import argparse
import json
import os
import sys
from pathlib import Path


PIXAL3D_ROOT = Path('/mnt/hdd/3d/avatar-shootout/Pixal3D')
HERE = Path(__file__).resolve().parent


def camera_contract(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key] = value
    required = ('cam_angle_x_rad', 'camera_distance', 'mesh_scale')
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f'camera contract lacks {", ".join(missing)}: {path}')
    return {
        'camera_angle_x': float(values['cam_angle_x_rad']),
        'distance': float(values['camera_distance']),
        'mesh_scale': float(values['mesh_scale']),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, type=Path,
                    help='already model-facing RGB frame; no preprocessing occurs')
    ap.add_argument('--camera', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path,
                    help='new diagnostic directory; refuses overwrite')
    ap.add_argument('--model', default='TencentARC/Pixal3D')
    args = ap.parse_args()
    if not args.image.is_file() or not args.camera.is_file():
        raise SystemExit('image and camera must exist')
    if args.out.exists():
        raise SystemExit(f'refusing to overwrite immutable diagnostic: {args.out}')
    args.out.mkdir(parents=True)

    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')
    os.environ['PIXAL3D_DINO_BOUNDARY_TRACE'] = '1'
    sys.path.insert(0, str(PIXAL3D_ROOT))
    sys.path.insert(0, str(HERE))
    os.chdir(PIXAL3D_ROOT)

    from PIL import Image
    import golden_stage_hook
    import inference

    camera = camera_contract(args.camera)
    golden_stage_hook.install(out_dir=str(args.out / 'stages'))
    pipe = inference.init_pipeline(args.model, device='cuda', low_vram=True)
    image = Image.open(args.image).convert('RGB')
    result = pipe.get_proj_cond_ss([image], **camera)
    global_shape = list(result['cond']['global'].shape)
    projection_shape = list(result['cond']['proj'].shape)
    (args.out / 'manifest.json').write_text(json.dumps({
        'purpose': 'Pixal stage-1 DINO/projection boundary diagnostic',
        'image': str(args.image), 'camera': camera,
        'global_shape': global_shape, 'projection_shape': projection_shape,
    }, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'global_shape': global_shape, 'projection_shape': projection_shape,
                      'out': str(args.out)}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
