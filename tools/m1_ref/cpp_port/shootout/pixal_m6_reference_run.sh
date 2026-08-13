#!/usr/bin/env bash
# Run the official Pixal projection-conditioned M6 material capture on PCI GPU 0 only.
# The selected delivery mesh is deliberately not an input: this command writes raw material
# boundaries only.  Bake its pbr_cache onto that unchanged mesh with native_texture_rebake.sh.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:?usage: pixal_m6_reference_run.sh <model-input.png> <camera_provenance.txt> <new-output-dir> <seed>}"
CAMERA="${2:?need camera provenance}"
OUT="${3:?need new immutable output directory}"
SEED="${4:?need seed}"
PY="/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python3"

[[ -f "$IMAGE" && -f "$CAMERA" ]] || { echo "missing image or camera contract" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "missing official Pixal3D Python: $PY" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite immutable diagnostic: $OUT" >&2; exit 2; }
export ATTN_BACKEND=sdpa NVIDIA_TF32_OVERRIDE=0 PIXAL3D_TEX_NAF_TARGET=512
"$CP/shootout/run_on_3060.sh" nice -n "${IMAGE_TO_RIG_NICE_LEVEL:-10}" ionice -c 2 -n 7 \
  "$PY" "$CP/shootout/capture_pixal_m6_reference.py" \
  --image "$IMAGE" --camera "$CAMERA" --out "$OUT" --seed "$SEED"
