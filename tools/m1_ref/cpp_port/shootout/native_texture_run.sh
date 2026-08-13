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
#   native_texture_run.sh refined.glb input.png high_native.glb --resolution 1024 --texsize 2048 --decimate 300000
#   native_texture_run.sh refined.glb input.png med_native.glb  --resolution 1024 --texsize 1024 --decimate 150000
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH="${1:?usage: native_texture_run.sh <refined.glb> <image.png> <out.glb> [args...]}"
IMAGE="${2:?need image}"
OUT="${3:?need output glb}"
shift 3

BIN="$CP/texture_mesh_native"
# Match the geometry runner: local NVMe F16 control bundle while the clean recipe is
# being verified.  Q8 remains the intended VRAM/performance target and is selected only
# through an explicitly labelled PIXAL3D_GGUF_DIR A/B after this control path passes.
WEIGHTS="${PIXAL3D_GGUF_DIR:-/home/dbrain/models/3d/geo_f16_native}"
OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"

# The executable loads its checked-in NPY encoder/decoder fallbacks by relative
# path.  A caller may invoke this runner from any directory, so anchor the child
# at cpp_port rather than silently failing after it has reserved the 3060.
cd "$CP"

[[ -x "$BIN" ]] || { echo "missing $BIN; build it first: cd $CP && ./build.sh texture_mesh_native cuda" >&2; exit 2; }
[[ -f "$MESH" ]] || { echo "missing mesh: $MESH" >&2; exit 2; }
[[ -f "$IMAGE" ]] || { echo "missing image: $IMAGE" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$OUT_ROOT"
STATUS_FILE="${OUT}.run-status.txt"
STAGE_FILE="${OUT}.stage-log.txt"
QC_FILE="${OUT}.texture-qc.txt"
STARTED_AT="$(date -Is)"
START_SECONDS=$SECONDS
STATUS_INITIALIZED=0
# Keep measured resource evidence beside the artifact.  `texture_mesh_native`
# has both CUDA inference and CPU atlas stages; a single end-of-run snapshot is
# misleading because the CUDA graph has normally been released before encode.
CPU_UTIL_PEAK=0
GPU_VRAM_PEAK_MIB=0
GPU_VRAM_LAST_MIB=0

sample_resources() {
  local cpu vram
  cpu="$(ps -p "$CHILD_PID" -o %cpu= 2>/dev/null | awk '{printf "%d", $1 + 0}' || true)"
  [[ "$cpu" =~ ^[0-9]+$ ]] || cpu=0
  if (( cpu > CPU_UTIL_PEAK )); then CPU_UTIL_PEAK=$cpu; fi
  vram="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_3060_UUID" 2>/dev/null | awk 'NR==1 {print int($1)}' || true)"
  [[ "$vram" =~ ^[0-9]+$ ]] || vram=0
  GPU_VRAM_LAST_MIB=$vram
  if (( vram > GPU_VRAM_PEAK_MIB )); then GPU_VRAM_PEAK_MIB=$vram; fi
}

# The executable writes QC only after it has written both GLB and atlas. Preserve
# a previous sidecar for audit rather than letting a live rerun appear to inherit
# a prior success in its status file.
if [[ -e "$QC_FILE" ]]; then
  mv -f "$QC_FILE" "${QC_FILE}.prior-$(date +%Y%m%dT%H%M%S%z)"
fi

# A visual result is only useful if its material recipe can be reconstructed later.  Compute these
# once (rather than in the ten-second status refresh) and retain them alongside the live PID/status.
MESH_SHA256="$(sha256sum "$MESH" | awk '{print $1}')"
IMAGE_SHA256="$(sha256sum "$IMAGE" | awk '{print $1}')"
BIN_SHA256="$(sha256sum "$BIN" | awk '{print $1}')"
SOURCE_REVISION="$(git -C "$CP" rev-parse --verify HEAD 2>/dev/null || printf unknown)"
TEXTURE_RESOLUTION=1024
UNWRAP_MODE=reference
ENCODER_DECIMATE=0
TEXTURE_MODEL=generic-cross
for ((arg_i=1; arg_i<=$#; arg_i++)); do
  case "${!arg_i}" in
    --resolution)
      ((arg_i += 1)); TEXTURE_RESOLUTION="${!arg_i}";;
    --unwrap)
      ((arg_i += 1)); UNWRAP_MODE="${!arg_i}";;
    --encoder-decimate)
      ((arg_i += 1)); ENCODER_DECIMATE="${!arg_i}";;
    --texture-model)
      ((arg_i += 1)); TEXTURE_MODEL="${!arg_i}";;
  esac
done

# Use the 3060 UUID directly. A bare CUDA ordinal has historically selected
# the owner's busy 5060 after a host/device-order change.
GPU_3060_UUID="${IMAGE_TO_RIG_GPU_3060_UUID:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {uuid=$1} END {print uuid}')}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_3060_UUID" 2>/dev/null | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_3060_UUID' is '$GPU_NAME', expected the reserved RTX 3060" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_3060_UUID"

exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
flock -n 9 || { echo "another image-to-rig job owns the 3060" >&2; exit 75; }

write_status() {
  local state="$1" rc="${2:-0}"
  shift 2
  # The child appends its artifact outcome after it has actually written outputs. Preserve that
  # authoritative record when this launcher's status refreshes or finalizes its own fields.
  local artifact_lines=""
  local stage_line=""
  local qc_lines=""
  # The child appends completion truth while this launcher refreshes the status every ten seconds.
  # Retain that truth only after THIS invocation has written its initial record: otherwise a rerun
  # over the same output inherits a stale `artifact_state=succeeded` from the prior child while the
  # new inference is still live.
  if [[ "$STATUS_INITIALIZED" == 1 && -f "$STATUS_FILE" ]]; then
    artifact_lines="$(sed -n '/^artifact_/p' "$STATUS_FILE")"
  fi
  if [[ -f "$STAGE_FILE" ]]; then
    stage_line="$(tail -n 1 "$STAGE_FILE" || true)"
  fi
  if [[ -s "$QC_FILE" ]]; then
    qc_lines="$(sed -n '/^sampling_verdict=/p;/^source_missing_fraction_before_repair=/p;/^unresolved_surface_texels_after_chart_repair=/p' "$QC_FILE")"
  fi
  # Atlas and sampling controls are intentionally environment-overridable for
  # native A/Bs. Record the effective overrides: without this, a retained GLB
  # cannot distinguish the promoted parity recipe from a cone/median test.
  local texture_controls
  texture_controls="$(env | LC_ALL=C sort | awk -F= '$1 ~ /^(ATL|TEX)_/ && $1 != "TEX_STAGE_LOG" {print}' | paste -sd ';' -)"
  {
    printf 'launcher_state=%s\n' "$state"
    printf 'started_at=%s\n' "$STARTED_AT"
    printf 'updated_at=%s\n' "$(date -Is)"
    printf 'elapsed_seconds=%s\n' "$((SECONDS-START_SECONDS))"
    printf 'exit_code=%s\n' "$rc"
    printf 'gpu=PCI GPU 0 / %s\n' "$GPU_NAME"
    printf 'gpu_uuid=%s\n' "$GPU_3060_UUID"
    printf 'gpu_vram_mib_last=%s\ngpu_vram_mib_peak=%s\n' "$GPU_VRAM_LAST_MIB" "$GPU_VRAM_PEAK_MIB"
    printf 'cpu_util_percent_peak=%s\n' "$CPU_UTIL_PEAK"
    printf 'scheduler=nice=%s; ionice=best-effort:7\n' "${IMAGE_TO_RIG_NICE_LEVEL:-10}"
    printf 'mesh=%s\nimage=%s\nout=%s\n' "$MESH" "$IMAGE" "$OUT"
    printf 'mesh_sha256=%s\nimage_sha256=%s\nbinary_sha256=%s\nsource_revision=%s\n' \
      "$MESH_SHA256" "$IMAGE_SHA256" "$BIN_SHA256" "$SOURCE_REVISION"
    printf 'texture_model=%s; native C++ only\n' "$TEXTURE_MODEL"
    printf 'texture_lattice=%s\n' "$TEXTURE_RESOLUTION"
    printf 'encoder_decimate_faces=%s\n' "$ENCODER_DECIMATE"
    printf 'unwrap_mode=%s\n' "$UNWRAP_MODE"
    printf 'texture_controls=%s\n' "${texture_controls:-default}"
    [[ -z "$stage_line" ]] || printf 'native_stage=%s\n' "$stage_line"
    [[ -z "$qc_lines" ]] || printf '%s\n' "$qc_lines"
    if [[ "$UNWRAP_MODE" == reference ]]; then
      printf 'clean_material_contract=adaptive xatlas parity charts (direct clean mesh / conformal local islands for high curvature); 2x raster; topology normals; full gutter repair; PBR RGB outlier default 0.10 unless explicitly overridden\n'
    fi
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
STATUS_INITIALIZED=1
trap 'finish_status "$@"' EXIT

echo "== native texture: $(basename "$MESH") + $(basename "$IMAGE") -> $OUT =="
# Sparse texture decode, xatlas, and inpainting include CPU-heavy phases.  Keep the complete
# 3060-owned child niced/ioniced; CUDA selection above remains physical PCI GPU 0.
nice -n "${IMAGE_TO_RIG_NICE_LEVEL:-10}" ionice -c 2 -n 7 \
  "$BIN" --model "$WEIGHTS" --mesh "$MESH" --image "$IMAGE" --out "$OUT" --status-file "$STATUS_FILE" "$@" &
CHILD_PID=$!
# Long direct unwraps are CPU-bound after native inference.  Refresh the status artifact while
# the child runs so operators never have to guess whether the 3060 lane or chart solve is alive.
while kill -0 "$CHILD_PID" 2>/dev/null; do
  sample_resources
  write_status running 0 "$@"
  sleep 10
done
set +e
wait "$CHILD_PID"
CHILD_RC=$?
set -e
exit "$CHILD_RC"
