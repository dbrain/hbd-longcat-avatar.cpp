#!/usr/bin/env python3
"""
GPU run-once: capture real Pixal3D submanifold-conv layer goldens, then early-exit.

Run from the spike tools dir with the Pixal3D venv:
  cd <Pixal3D>; source .venv/bin/activate
  ATTN_BACKEND=sdpa python <spike>/tools/sparse_spike/golden_dump_runner.py \
      --image <img> --out <spike>/tools/sparse_spike/golden_model --stop 20

Installs golden_hook (flex_gemm backend), runs run_inference, and stops as soon as
`--stop` distinct layer shapes are captured (so we don't pay the full decode).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # golden_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pixal3d', default='/mnt/hdd/3d/avatar-shootout/Pixal3D')
    ap.add_argument('--image', default='/mnt/hdd/3d/avatar-shootout/assets/miku.png')
    ap.add_argument('--out', default=os.path.join(HERE, 'golden_model'))
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--stop', type=int, default=20)
    args = ap.parse_args()

    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('PIXAL3D_TEX_NAF_TARGET', '512')

    sys.path.insert(0, args.pixal3d)
    os.chdir(args.pixal3d)

    import golden_hook
    golden_hook.install(out_dir=args.out, max_per_key=1, stop_after=args.stop)

    import inference
    try:
        inference.run_inference(
            image_path=args.image,
            output_path=os.path.join(args.out, '_decode.glb'),
            seed=42, manual_fov=-1.0,
            model_path=inference.MODEL_PATH,
            low_vram=True, resolution=args.resolution,
        )
        print('[runner] inference finished without hitting stop_after — '
              'captured all distinct shapes that appeared')
    except golden_hook._StopCapture:
        print(f'[runner] early-exit after capturing {args.stop} distinct layer shapes')

    print('[runner] goldens in', args.out)


if __name__ == '__main__':
    main()
