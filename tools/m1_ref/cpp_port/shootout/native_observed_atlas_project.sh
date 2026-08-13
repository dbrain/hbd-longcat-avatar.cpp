#!/usr/bin/env bash
# Repack a native high GLB with real observed-image detail while retaining the
# authoritative native atlas everywhere no source view actually sees the mesh.
#
# Usage:
#   native_observed_atlas_project.sh <native-high.glb> <camera-provenance.txt> <out-dir> \
#       --front /absolute/front.png [--back /absolute/back.png] [--view yaw /absolute/view.png]...
#       [--rigged-source /absolute/hymotion_rigged.glb]
#
# This stage is intentionally CPU-only. It neither loads a generative model
# nor reserves a GPU. The source high asset remains the production default;
# this writes a separately named, verifiable observed-view candidate.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIGH="${1:?usage: native_observed_atlas_project.sh <native-high.glb> <camera-provenance.txt> <out-dir> --front image [--back image] [--view yaw image]...}"
CAMERA="${2:?need camera provenance from the geometry cache}"
OUT="${3:?need output directory}"
shift 3

[[ -s "$HIGH" ]] || { echo "missing native high GLB: $HIGH" >&2; exit 2; }
[[ -s "$CAMERA" ]] || { echo "missing camera provenance: $CAMERA" >&2; exit 2; }
for key in cam_angle_x_rad camera_distance mesh_scale; do
  value="$(awk -F= -v key="$key" '$1==key {print $2; exit}' "$CAMERA")"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "camera provenance lacks numeric $key: $CAMERA" >&2; exit 2; }
done
for bin in native_atlas_project glb_repack mesh_topo; do
  [[ -x "$CP/$bin" ]] || { echo "missing executable: $CP/$bin (build it first)" >&2; exit 2; }
done

FRONT=""
BACK=""
RIGGED_SOURCE=""
VIEW_ARGS=()
VIEW_RECORDS=()
declare -A SEEN_YAW=()
normalise_yaw() {
  awk -v yaw="$1" 'BEGIN {
    if (yaw !~ /^[-+]?[0-9]*([.][0-9]+)?$/) exit 1
    y = yaw + 0; y = y - 360 * int(y / 360); if (y < 0) y += 360; if (y >= 360) y -= 360
    printf "%.6g", y
  }'
}
add_view() {
  local raw_yaw="$1" image="$2" yaw
  yaw="$(normalise_yaw "$raw_yaw")" || { echo "invalid view yaw: $raw_yaw" >&2; exit 2; }
  [[ "$image" = /* && -s "$image" ]] || { echo "view image must be an existing absolute path: $image" >&2; exit 2; }
  [[ -z "${SEEN_YAW[$yaw]:-}" ]] || { echo "duplicate observed-view yaw: $yaw" >&2; exit 2; }
  SEEN_YAW[$yaw]=1
  case "$yaw" in
    0) FRONT="$image";;
    180) BACK="$image";;
    *) VIEW_ARGS+=(--view "$yaw" "$image");;
  esac
  VIEW_RECORDS+=("$yaw|$image")
}
while (( $# )); do
  case "$1" in
    --front) (( $# >= 2 )) || { echo "--front needs a path" >&2; exit 2; }; add_view 0 "$2"; shift 2;;
    --back) (( $# >= 2 )) || { echo "--back needs a path" >&2; exit 2; }; add_view 180 "$2"; shift 2;;
    --view) (( $# >= 3 )) || { echo "--view needs yaw and path" >&2; exit 2; }; add_view "$2" "$3"; shift 3;;
    --rigged-source) (( $# >= 2 )) || { echo "--rigged-source needs a path" >&2; exit 2; }; RIGGED_SOURCE="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$FRONT" ]] || { echo "a yaw=0 front image is required" >&2; exit 2; }
(( ${#VIEW_RECORDS[@]} <= 8 )) || { echo "at most eight observed views are supported" >&2; exit 2; }
[[ -z "$RIGGED_SOURCE" || -s "$RIGGED_SOURCE" ]] || { echo "missing rigged source: $RIGGED_SOURCE" >&2; exit 2; }

mkdir -p "$OUT"
BASE="${HIGH##*/}"; BASE="${BASE%.glb}"
ATLAS="$OUT/${BASE}_observed_projected_ab_atlas.png"
GLB="$OUT/${BASE}_observed_projected_ab.glb"
STATS="$OUT/${BASE}_observed_projected_ab.projection-qc.txt"
LOG="$OUT/${BASE}_observed_projected_ab.projection.log"
SRC="$OUT/${BASE}_observed_projected_ab.projection-source.txt"
DEBUG="$OUT/${BASE}_observed_projected_ab_debug"
TMP_GLB="$GLB.tmp.$$"
TMP_RIGGED="$OUT/${BASE}_observed_projected_ab_rigged.glb.tmp.$$"
trap 'rm -f "$TMP_GLB" "$TMP_RIGGED"' EXIT

ARGS=(--mesh "$HIGH" --front "$FRONT" --camera "$CAMERA" --out-atlas "$ATLAS" --stats "$STATS" --debug-dir "$DEBUG")
[[ -z "$BACK" ]] || ARGS+=(--back "$BACK")
ARGS+=("${VIEW_ARGS[@]}")
"$CP/native_atlas_project" "${ARGS[@]}" 2>&1 | tee "$LOG"
GLB_REPACK_UNQUANTIZED=1 GLB_REPACK_BASE_COLOR_PNG="$ATLAS" "$CP/glb_repack" "$HIGH" "$TMP_GLB" 2>&1 | tee -a "$LOG"
mv -f "$TMP_GLB" "$GLB"
TOPO="$($CP/mesh_topo "$GLB")"
[[ "$TOPO" =~ open=([0-9]+) ]] && (( BASH_REMATCH[1] == 0 )) || { echo "REJECT: projected native candidate is not closed: $TOPO" >&2; exit 1; }
RIGGED_GLB=""
RIGGED_TOPO=""
if [[ -n "$RIGGED_SOURCE" ]]; then
  # A native texture stage may only publish an animation-ready projected copy
  # when the supplied hand-off is already a complete exact-name Mixamo rig.
  for node in Hips LeftUpLeg RightUpLeg Spine LeftLeg RightLeg Spine1 LeftFoot RightFoot Spine2 LeftToeBase RightToeBase Neck LeftShoulder RightShoulder Head LeftArm RightArm LeftForeArm RightForeArm LeftHand RightHand; do
    grep -a -F -q "\"name\":\"mixamorig:$node\"" "$RIGGED_SOURCE" || { echo "rigged source lacks exact Mixamo node: $node" >&2; exit 1; }
  done
  RIGGED_GLB="$OUT/${BASE}_observed_projected_ab_rigged.glb"
  GLB_REPACK_PRESERVE_CONTAINER=1 GLB_REPACK_BASE_COLOR_PNG="$ATLAS" "$CP/glb_repack" "$RIGGED_SOURCE" "$TMP_RIGGED" 2>&1 | tee -a "$LOG"
  mv -f "$TMP_RIGGED" "$RIGGED_GLB"
  RIGGED_TOPO="$($CP/mesh_topo "$RIGGED_GLB")"
  [[ "$RIGGED_TOPO" =~ open=([0-9]+) ]] && (( BASH_REMATCH[1] == 0 )) || { echo "REJECT: projected rigged candidate is not closed: $RIGGED_TOPO" >&2; exit 1; }
fi

{
  printf 'schema_version=1\n'
  printf 'mode=native-base observed-image overlay A/B; source native high remains production default\n'
  printf 'execution=CPU-only; no model inference and no GPU reserved\n'
  printf 'source_native_high=%s sha256=%s\n' "$HIGH" "$(sha256sum "$HIGH" | awk '{print $1}')"
  printf 'camera_provenance=%s sha256=%s\n' "$CAMERA" "$(sha256sum "$CAMERA" | awk '{print $1}')"
  printf 'camera_count=%s\n' "${#VIEW_RECORDS[@]}"
  for record in "${VIEW_RECORDS[@]}"; do
    yaw="${record%%|*}"; image="${record#*|}"
    printf 'view yaw=%s path=%s sha256=%s\n' "$yaw" "$image" "$(sha256sum "$image" | awk '{print $1}')"
  done
  printf 'candidate_glb=%s sha256=%s\n' "${GLB##*/}" "$(sha256sum "$GLB" | awk '{print $1}')"
  printf 'candidate_atlas=%s sha256=%s\n' "${ATLAS##*/}" "$(sha256sum "$ATLAS" | awk '{print $1}')"
  printf 'projection_qc=%s sha256=%s\n' "${STATS##*/}" "$(sha256sum "$STATS" | awk '{print $1}')"
  printf 'projection_log=%s sha256=%s\n' "${LOG##*/}" "$(sha256sum "$LOG" | awk '{print $1}')"
  printf 'topology=%s\n' "$TOPO"
  if [[ -n "$RIGGED_GLB" ]]; then
    printf 'rigged_source=%s sha256=%s\n' "$RIGGED_SOURCE" "$(sha256sum "$RIGGED_SOURCE" | awk '{print $1}')"
    printf 'rigged_candidate_glb=%s sha256=%s\n' "${RIGGED_GLB##*/}" "$(sha256sum "$RIGGED_GLB" | awk '{print $1}')"
    printf 'rigged_candidate_topology=%s\n' "$RIGGED_TOPO"
    printf 'rigged_container=preserved skins/nodes/animations; only embedded baseColor image replaced\n'
  fi
  printf 'promotion_result=not-promoted; visual A/B and native-observed gate required\n'
} >"$SRC.tmp"
mv -f "$SRC.tmp" "$SRC"
echo "DONE native observed-image A/B: $GLB"
