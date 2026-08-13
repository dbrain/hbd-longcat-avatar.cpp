#!/usr/bin/env bash
# Official-Python fixed-mesh M6 diagnostic, safely pinned to PCI GPU 0 / RTX 3060.
set -euo pipefail
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH="${1:?usage: pixal_fixed_mesh_m6_reference_run.sh <mesh.glb> <model-input.png> <camera.txt> <new-output-dir> <seed>}"
IMAGE="${2:?need model input}"; CAMERA="${3:?need camera}"; OUT="${4:?need output}"; SEED="${5:?need seed}"
PY="/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python3"
[[ -f "$MESH" && -f "$IMAGE" && -f "$CAMERA" && -x "$PY" ]] || { echo "missing fixed-mesh reference input" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite immutable diagnostic: $OUT" >&2; exit 2; }
export ATTN_BACKEND=sdpa NVIDIA_TF32_OVERRIDE=0 PIXAL3D_TEX_NAF_TARGET=512
"$CP/shootout/run_on_3060.sh" nice -n "${IMAGE_TO_RIG_NICE_LEVEL:-10}" ionice -c 2 -n 7 \
  "$PY" "$CP/shootout/capture_fixed_mesh_pixal_m6_reference.py" \
  --mesh "$MESH" --image "$IMAGE" --camera "$CAMERA" --out "$OUT" --seed "$SEED"
