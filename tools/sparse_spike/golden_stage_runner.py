#!/usr/bin/env python3
"""
M0 GPU run-once: capture Pixal3D *stage-boundary* goldens for the C++/ggml port.

Runs ONE full Pixal3D decode with golden_stage_hook installed -> dumps every stage
boundary (preprocessed image, SS cond, z_s, coords, lr_slat, hr_coords, shape_slat,
tex_slat, mesh verts/faces) + per-stage peak VRAM + the model configs.

Coordinate GPU with the owner first (the 3060 is shared). Run:
  cd /mnt/hdd/3d/avatar-shootout/Pixal3D && source .venv/bin/activate
  ATTN_BACKEND=sdpa python \
      /home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stage_runner.py \
      --image /mnt/hdd/3d/avatar-shootout/assets/miku.png --resolution 1024

Notes
-----
* Uses low_vram=True (sequential per-stage offload) so peak VRAM == max(stage), the
  mode the C++ port must replicate. The vram.json this writes is the per-stage budget.
* Goldens are saved incrementally (each stage writes immediately) and metadata via
  atexit, so even if the final GLB export (o_voxel remesh, PHASE-2) errors, the
  geometry goldens are already on disk.
* seed fixed to 42 (run() reseeds at the same point Python does), manual_fov=-1 so
  the camera matches a normal decode. Pass --fov to pin a manual FOV if MoGe is
  unavailable / you want determinism across machines.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # golden_stage_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pixal3d', default='/mnt/hdd/3d/avatar-shootout/Pixal3D')
    ap.add_argument('--model', default=None,
                    help='Pixal3D checkpoint directory or repo; pin a local HF snapshot for an exact oracle')
    ap.add_argument('--image', default='/mnt/hdd/3d/avatar-shootout/assets/miku.png')
    ap.add_argument('--out', default=os.path.join(HERE, 'golden_stages'))
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fov', type=float, default=-1.0)
    args = ap.parse_args()

    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('PIXAL3D_TEX_NAF_TARGET', '512')

    sys.path.insert(0, args.pixal3d)
    os.chdir(args.pixal3d)

    import golden_stage_hook
    golden_stage_hook.install(out_dir=args.out)

    import inference
    glb_out = os.path.join(args.out, '_stage_decode.glb')
    try:
        inference.run_inference(
            image_path=args.image,
            output_path=glb_out,
            seed=args.seed, manual_fov=args.fov,
            model_path=args.model or inference.MODEL_PATH,
            low_vram=True, resolution=args.resolution,
        )
        print('[stage_runner] full decode + GLB export finished')
    except golden_stage_hook.Stage1TraceComplete as e:
        print('[stage_runner]', e)
    except Exception as e:
        import traceback
        print('[stage_runner] decode raised (geometry goldens already saved):', e)
        traceback.print_exc()

    print('[stage_runner] stage goldens in', args.out)


if __name__ == '__main__':
    main()
