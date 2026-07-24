#!/usr/bin/env bash
# CPU-only native texture rebake from a saved native PBR volume.
#
# Use after a native inference run to A/B atlas resolution or conservative cleanup without
# rerunning DINO / texture diffusion. This intentionally holds no GPU lock: it performs only
# xatlas, CPU rasterisation, and native GLB writing.
#
# Usage:
#   native_texture_rebake.sh <refined.glb> <native-dump-dir|image_to_rig-stage-dir> <out.glb> [texture_rebake_native args...]
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH="${1:?usage: native_texture_rebake.sh <refined.glb> <native-dump-dir> <out.glb> [args...]}"
PBR_SOURCE="${2:?need native PBR dump or image_to_rig stage directory}"
OUT="${3:?need output glb}"
shift 3

BIN="$CP/texture_rebake_native"
[[ -x "$BIN" ]] || { echo "missing $BIN; build it first: cd $CP && ./build.sh texture_rebake_native" >&2; exit 2; }
NATIVE_ATLAS_CUDA="${NATIVE_ATLAS_CUDA:-0}"
[[ "$NATIVE_ATLAS_CUDA" == 0 || "$NATIVE_ATLAS_CUDA" == 1 ]] || { echo "NATIVE_ATLAS_CUDA must be 0 or 1" >&2; exit 2; }
[[ -f "$MESH" ]] || { echo "missing mesh: $MESH" >&2; exit 2; }
PBR_ARGS=()
PBR_FEATS=""
PBR_COORDS=""
if [[ -f "$PBR_SOURCE/native_pbr_feats.npy" && -f "$PBR_SOURCE/native_pbr_coords.npy" ]]; then
  PBR_ARGS=(--pbr-dir "$PBR_SOURCE")
  PBR_FEATS="$PBR_SOURCE/native_pbr_feats.npy"
  PBR_COORDS="$PBR_SOURCE/native_pbr_coords.npy"
elif [[ -f "$PBR_SOURCE/pbr_feats.bin" && -f "$PBR_SOURCE/pbr_coords.bin" ]]; then
  # Generated in image_to_rig's Pixal flow. These are size-prefixed raw arrays in the exact
  # refined-mesh frame, and avoid re-encoding an arbitrary mesh just to make texture latents.
  PBR_ARGS=(--pbr-cache "$PBR_SOURCE")
  PBR_FEATS="$PBR_SOURCE/pbr_feats.bin"
  PBR_COORDS="$PBR_SOURCE/pbr_coords.bin"
else
  echo "missing native PBR dump or image_to_rig PBR cache in $PBR_SOURCE" >&2; exit 2;
fi
mkdir -p "$(dirname "$OUT")"

# A rebake does not occupy the 3060, but it can still spend time in a difficult
# xatlas solve.  Give it the same inspectable, append-only phase record as the
# inference rung so the delivery manifest can say which adaptive chart route
# actually ran rather than guessing from the requested unwrap mode.
STATUS_FILE="${OUT}.run-status.txt"
STAGE_FILE="${TEX_STAGE_LOG:-${OUT}.stage-log.txt}"
QC_FILE="${OUT}.texture-qc.txt"
export TEX_STAGE_LOG="$STAGE_FILE"
: >"$STAGE_FILE"
STARTED_AT="$(date -Is)"
START_SECONDS=$SECONDS
CPU_UTIL_PEAK=0
GPU_VRAM_PEAK_MIB=0
GPU_VRAM_LAST_MIB=0
GPU_NAME=""
GPU_3060_UUID=""
if [[ "$NATIVE_ATLAS_CUDA" == 1 ]]; then
  GPU_3060_UUID="${IMAGE_TO_RIG_GPU_3060_UUID:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {uuid=$1} END {print uuid}')}"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_3060_UUID" 2>/dev/null | head -1)"
  [[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_3060_UUID' is '$GPU_NAME', expected RTX 3060" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPU_3060_UUID"
  # CUDA rebakes default to the bounded native CuMesh chart precluster. A
  # caller can still set ATL_NATIVE_CUMESH=0 for an explicitly labelled CPU
  # comparison, but never reaches the global direct-chart route by default.
  export ATL_NATIVE_CUMESH="${ATL_NATIVE_CUMESH:-1}"
  OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"
  mkdir -p "$OUT_ROOT"
  exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
  flock -n 9 || { echo "another image-to-rig job owns the 3060" >&2; exit 75; }
fi

sample_resources() {
  local cpu vram
  cpu="$(ps -p "$CHILD_PID" -o %cpu= 2>/dev/null | awk '{printf "%d", $1 + 0}' || true)"
  [[ "$cpu" =~ ^[0-9]+$ ]] || cpu=0
  if (( cpu > CPU_UTIL_PEAK )); then CPU_UTIL_PEAK=$cpu; fi
  if [[ "$NATIVE_ATLAS_CUDA" == 1 ]]; then
    vram="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_3060_UUID" 2>/dev/null | awk 'NR==1 {print int($1)}' || true)"
    [[ "$vram" =~ ^[0-9]+$ ]] || vram=0
    GPU_VRAM_LAST_MIB=$vram
    if (( vram > GPU_VRAM_PEAK_MIB )); then GPU_VRAM_PEAK_MIB=$vram; fi
  fi
}
MESH_SHA256="$(sha256sum "$MESH" | awk '{print $1}')"
DUMP_FEATS_SHA256="$(sha256sum "$PBR_FEATS" | awk '{print $1}')"
DUMP_COORDS_SHA256="$(sha256sum "$PBR_COORDS" | awk '{print $1}')"
BINARY_SHA256="$(sha256sum "$BIN" | awk '{print $1}')"
SOURCE_REVISION="$(git -C "$CP" rev-parse --verify HEAD 2>/dev/null || printf unknown)"
if [[ -e "$QC_FILE" ]]; then
  mv -f "$QC_FILE" "${QC_FILE}.prior-$(date +%Y%m%dT%H%M%S%z)"
fi

write_status() {
  local state="$1" rc="${2:-0}" stage_line="" qc_lines=""
  shift 2
  [[ -f "$STAGE_FILE" ]] && stage_line="$(tail -n 1 "$STAGE_FILE" || true)"
  if [[ -s "$QC_FILE" ]]; then
    qc_lines="$(sed -n '/^sampling_verdict=/p;/^source_missing_fraction_before_repair=/p;/^unresolved_surface_texels_after_chart_repair=/p' "$QC_FILE")"
  fi
  # Keep CPU A/Bs reproducible. The chosen atlas policy often lives in ATL_ /
  # TEX_ environment controls rather than the positional command line.
  local texture_controls
  texture_controls="$(env | LC_ALL=C sort | awk -F= '$1 ~ /^(ATL|TEX)_/ && $1 != "TEX_STAGE_LOG" {print}' | paste -sd ';' -)"
  {
    printf 'launcher_state=%s\n' "$state"
    printf 'started_at=%s\nupdated_at=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
      "$STARTED_AT" "$(date -Is)" "$((SECONDS-START_SECONDS))" "$rc"
    if [[ "$NATIVE_ATLAS_CUDA" == 1 ]]; then
      printf 'execution=GPU native atlas raster; bounded native chart route\n'
      printf 'gpu=PCI GPU 0 / %s\ngpu_uuid=%s\n' "$GPU_NAME" "$GPU_3060_UUID"
    else
      printf 'execution=CPU-only native re-atlas; no GPU reserved\n'
    fi
    printf 'gpu_vram_mib_last=%s\ngpu_vram_mib_peak=%s\ncpu_util_percent_peak=%s\n' "$GPU_VRAM_LAST_MIB" "$GPU_VRAM_PEAK_MIB" "$CPU_UTIL_PEAK"
    printf 'mesh=%s\npbr_source=%s\nout=%s\nstage_log=%s\n' "$MESH" "$PBR_SOURCE" "$OUT" "$STAGE_FILE"
    printf 'mesh_sha256=%s\npbr_feats_sha256=%s\npbr_coords_sha256=%s\nbinary_sha256=%s\nsource_revision=%s\n' \
      "$MESH_SHA256" "$DUMP_FEATS_SHA256" "$DUMP_COORDS_SHA256" "$BINARY_SHA256" "$SOURCE_REVISION"
    printf 'texture_controls=%s\n' "${texture_controls:-default}"
    [[ -z "$stage_line" ]] || printf 'native_stage=%s\n' "$stage_line"
    [[ -z "$qc_lines" ]] || printf '%s\n' "$qc_lines"
    [[ "$state" != succeeded ]] || printf 'artifact_state=succeeded\n'
    [[ "$state" != failed ]] || printf 'artifact_state=failed\n'
    printf 'args='
    printf ' %q' "$@"
    printf '\n'
  } >"$STATUS_FILE"
}

write_status running 0 "$@"
echo "== niced native $([[ "$NATIVE_ATLAS_CUDA" == 1 ]] && echo CUDA || echo CPU) texture rebake: $(basename "$MESH") -> $OUT =="
# This remains CPU-only, but xatlas can be long-running.  Keep it below interactive and GPU
# pipeline work by default; callers can still override the numeric nice level deliberately.
nice -n "${IMAGE_TO_RIG_NICE_LEVEL:-10}" ionice -c 2 -n 7 \
  "$BIN" --mesh "$MESH" "${PBR_ARGS[@]}" --out "$OUT" "$@" &
CHILD_PID=$!
while kill -0 "$CHILD_PID" 2>/dev/null; do
  sample_resources
  sleep 5
  write_status running 0 "$@"
done
set +e
wait "$CHILD_PID"
CHILD_RC=$?
set -e
if (( CHILD_RC != 0 )); then
  write_status failed "$CHILD_RC" "$@"
  exit "$CHILD_RC"
fi
if [[ ! -s "$OUT" ]]; then
  echo "rebake produced no GLB: $OUT" >&2
  write_status failed 1 "$@"
  exit 1
fi
write_status succeeded 0 "$@"
