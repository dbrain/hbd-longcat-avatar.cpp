#!/usr/bin/env bash
# Native Trellis-2 retexture entrypoint for an already-clean/refined GLB.
#
# This is the production texture rung: no Python or Docker at runtime.  It
# keeps the native 3060 lane serialized with image_to_rig_sane.sh and accepts
# any image/mesh pair in the same coordinate/camera frame.
#
# Usage:
#   native_texture_run.sh <refined.glb> <source_rgba_or_black_matte.png> <out.glb> [texture_mesh_native args...]
#
# Common examples:
#   native_texture_run.sh refined.glb input.png high_native.glb --resolution 512 --texsize 2048 --decimate 300000
#   native_texture_run.sh refined.glb input.png med_native.glb  --resolution 512 --texsize 1024 --decimate 150000
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH="${1:?usage: native_texture_run.sh <refined.glb> <image.png> <out.glb> [args...]}"
IMAGE="${2:?need image}"
OUT="${3:?need output glb}"
shift 3

BIN="$CP/texture_mesh_native"
WEIGHTS="${PIXAL3D_GGUF_DIR:-/mnt/hdd/pixal3d/weights_gguf}"
OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"

[[ -x "$BIN" ]] || { echo "missing $BIN; build it first: cd $CP && ./build.sh texture_mesh_native cuda" >&2; exit 2; }
[[ -f "$MESH" ]] || { echo "missing mesh: $MESH" >&2; exit 2; }
[[ -f "$IMAGE" ]] || { echo "missing image: $IMAGE" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$OUT_ROOT"

# PCI order is stable: physical GPU 0 is the reserved 3060.  Do not remove
# CUDA_DEVICE_ORDER: a bare CVD=0 has historically selected the owner's 5060.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: PCI GPU 0 is '$GPU_NAME', expected RTX 3060" >&2; exit 1; }

exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
flock -n 9 || { echo "another image-to-rig job owns the 3060" >&2; exit 75; }

echo "== native texture: $(basename "$MESH") + $(basename "$IMAGE") -> $OUT =="
"$BIN" --model "$WEIGHTS" --mesh "$MESH" --image "$IMAGE" --out "$OUT" "$@"
