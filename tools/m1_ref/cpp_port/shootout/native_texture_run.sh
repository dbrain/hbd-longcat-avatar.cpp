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
STATUS_FILE="${OUT}.run-status.txt"
STARTED_AT="$(date -Is)"
START_SECONDS=$SECONDS

# PCI order is stable: physical GPU 0 is the reserved 3060.  Do not remove
# CUDA_DEVICE_ORDER: a bare CVD=0 has historically selected the owner's 5060.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: PCI GPU 0 is '$GPU_NAME', expected RTX 3060" >&2; exit 1; }

exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
flock -n 9 || { echo "another image-to-rig job owns the 3060" >&2; exit 75; }

write_status() {
  local state="$1" rc="${2:-0}"
  shift 2
  # The child appends its artifact outcome after it has actually written outputs.  Preserve that
  # authoritative record when the launcher refreshes or finalizes its own status fields.
  local artifact_lines=""
  if [[ -f "$STATUS_FILE" ]]; then artifact_lines="$(sed -n '/^artifact_/p' "$STATUS_FILE")"; fi
  {
    printf 'launcher_state=%s\n' "$state"
    printf 'started_at=%s\n' "$STARTED_AT"
    printf 'updated_at=%s\n' "$(date -Is)"
    printf 'elapsed_seconds=%s\n' "$((SECONDS-START_SECONDS))"
    printf 'exit_code=%s\n' "$rc"
    printf 'gpu=PCI GPU 0 / %s\n' "$GPU_NAME"
    printf 'mesh=%s\nimage=%s\nout=%s\n' "$MESH" "$IMAGE" "$OUT"
    printf 'args='
    printf ' %q' "$@"
    printf '\n'
    [[ -z "$artifact_lines" ]] || printf '%s\n' "$artifact_lines"
  } >"$STATUS_FILE"
}
finish_status() {
  local rc=$?
  if (( rc == 0 )); then write_status succeeded 0 "$@"; else write_status failed "$rc" "$@"; fi
}
write_status running 0 "$@"
trap 'finish_status "$@"' EXIT

echo "== native texture: $(basename "$MESH") + $(basename "$IMAGE") -> $OUT =="
"$BIN" --model "$WEIGHTS" --mesh "$MESH" --image "$IMAGE" --out "$OUT" --status-file "$STATUS_FILE" "$@" &
CHILD_PID=$!
# Long direct unwraps are CPU-bound after native inference.  Refresh the status artifact while
# the child runs so operators never have to guess whether the 3060 lane or chart solve is alive.
while kill -0 "$CHILD_PID" 2>/dev/null; do
  write_status running 0 "$@"
  sleep 10
done
set +e
wait "$CHILD_PID"
CHILD_RC=$?
set -e
exit "$CHILD_RC"
