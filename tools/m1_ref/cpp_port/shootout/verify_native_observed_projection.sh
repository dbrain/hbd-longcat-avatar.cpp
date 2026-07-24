#!/usr/bin/env bash
# Read-only gate for a native-base observed-image texture candidate.
# A one- to three-view candidate is a valid, explicitly unpromotable A/B: it
# can improve observed detail while retaining native material in blind areas.
# The 4--8-view / circular-coverage rule applies only when the caller asks to
# validate a turntable promotion candidate.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLB="${1:?usage: verify_native_observed_projection.sh <candidate.glb>}"
shift
REQUIRE_TURNAROUND=0
if (( $# > 0 )); then
  [[ "$1" == "--require-turnaround" && $# == 1 ]] || {
    echo "usage: verify_native_observed_projection.sh <candidate.glb> [--require-turnaround]" >&2; exit 2;
  }
  REQUIRE_TURNAROUND=1
fi
BASE="${GLB%.glb}"
SRC="${BASE}.projection-source.txt"
QC="${BASE}.projection-qc.txt"
[[ -s "$GLB" && -s "$SRC" && -s "$QC" ]] || { echo "REJECT: native observed candidate needs GLB, projection-source, and projection-qc" >&2; exit 1; }

value() { awk -F= -v key="$1" '$1==key {print substr($0,length(key)+2); exit}' "$QC"; }
[[ "$(value schema_version)" == 1 ]] || { echo "REJECT: unsupported projection QC schema" >&2; exit 1; }
TELEA="$(value telea_fallback_texels)"
[[ "$TELEA" =~ ^[0-9]+$ ]] && (( TELEA == 0 )) || { echo "REJECT: native observed candidate used $TELEA Telea fallback texels" >&2; exit 1; }
NATIVE_BASE="$(value unobserved_texels_retained_from_native_base)"
HOLES="$(value unobserved_hole_percent_before_native_fallback)"
[[ "$NATIVE_BASE" =~ ^[0-9]+$ && "$HOLES" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "REJECT: candidate lacks native-base fallback accounting" >&2; exit 1;
}
# A fully observed turntable legitimately has no holes. Otherwise project_onto
# must have preserved every hole from the native base instead of inventing atlas
# colour; the emitted integer count is the authoritative proof of that route.
if awk -v holes="$HOLES" 'BEGIN { exit !(holes > 0) }'; then
  (( NATIVE_BASE > 0 )) || { echo "REJECT: unobserved texels were not retained from the native base" >&2; exit 1; }
fi
SEAM="$(value seam_mean_absdiff_255)"
[[ "$SEAM" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "REJECT: projection QC lacks seam metric" >&2; exit 1; }
MAX_SEAM="${NATIVE_OBSERVED_MAX_SEAM_ABSDIFF_255:-32}"
awk -v got="$SEAM" -v max="$MAX_SEAM" 'BEGIN { exit !(got <= max) }' || { echo "REJECT: observed seam $SEAM/255 > $MAX_SEAM/255" >&2; exit 1; }

SRC_HIGH="$(sed -n 's/^source_native_high=\([^ ]*\) sha256=.*/\1/p' "$SRC")"
SRC_HASH="$(sed -n 's/^source_native_high=.* sha256=\([0-9a-f]*\)$/\1/p' "$SRC")"
[[ -s "$SRC_HIGH" && "$SRC_HASH" =~ ^[0-9a-f]{64}$ ]] || { echo "REJECT: missing native high provenance" >&2; exit 1; }
[[ "$(sha256sum "$SRC_HIGH" | awk '{print $1}')" == "$SRC_HASH" ]] || { echo "REJECT: source native high hash changed" >&2; exit 1; }
"$CP/shootout/verify_native_texture_asset.sh" "$SRC_HIGH" >/dev/null
CAMERA="$(sed -n 's/^camera_provenance=\([^ ]*\) sha256=.*/\1/p' "$SRC")"
CAMERA_HASH="$(sed -n 's/^camera_provenance=.* sha256=\([0-9a-f]*\)$/\1/p' "$SRC")"
[[ -s "$CAMERA" && "$CAMERA_HASH" =~ ^[0-9a-f]{64}$ ]] || { echo "REJECT: missing camera provenance" >&2; exit 1; }
[[ "$(sha256sum "$CAMERA" | awk '{print $1}')" == "$CAMERA_HASH" ]] || { echo "REJECT: camera provenance changed" >&2; exit 1; }
for key in cam_angle_x_rad camera_distance mesh_scale; do
  value="$(awk -F= -v key="$key" '$1==key {print $2; exit}' "$CAMERA")"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "REJECT: camera provenance lacks numeric $key" >&2; exit 1; }
done

LOG_NAME="$(sed -n 's/^projection_log=\([^ ]*\) sha256=.*/\1/p' "$SRC")"
LOG_HASH="$(sed -n 's/^projection_log=.* sha256=\([0-9a-f]*\)$/\1/p' "$SRC")"
LOG="$(dirname "$SRC")/$LOG_NAME"
[[ -s "$LOG" && "$LOG_HASH" =~ ^[0-9a-f]{64}$ ]] || { echo "REJECT: missing projection log provenance" >&2; exit 1; }
[[ "$(sha256sum "$LOG" | awk '{print $1}')" == "$LOG_HASH" ]] || { echo "REJECT: projection log changed" >&2; exit 1; }
# The projector explicitly measures whether rejected source samples are on the
# grazing silhouette (expected) or inside the subject (a bad fit/mask). A
# candidate with the latter warning is not clean enough for an eye-test A/B,
# much less promotion: it would bake background/noise into visible material.
if rg -q '\[WARN: rejected texels are NOT grazing — mask/fit suspect' "$LOG"; then
  echo "REJECT: projection mask/fit suspect; inspect source framing or camera provenance: $LOG" >&2
  exit 1
fi

CAMERAS="$(awk -F= '$1=="camera_count" {print $2; exit}' "$SRC")"
[[ "$CAMERAS" =~ ^[1-8]$ ]] || { echo "REJECT: invalid camera count" >&2; exit 1; }
declare -A seen=()
VIEWS=0
while IFS= read -r line; do
  [[ "$line" == view\ * ]] || continue
  yaw="$(sed -n 's/^view yaw=\([^ ]*\).*/\1/p' <<<"$line")"
  path="$(sed -n 's/.* path=\(.*\) sha256=.*/\1/p' <<<"$line")"
  expected="$(sed -n 's/.* sha256=\([0-9a-f]*\)$/\1/p' <<<"$line")"
  [[ -n "$yaw" && "$path" = /* && "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "REJECT: malformed view record" >&2; exit 1; }
  [[ -z "${seen[$yaw]:-}" ]] || { echo "REJECT: duplicate view yaw $yaw" >&2; exit 1; }
  seen[$yaw]=1; ((VIEWS += 1))
  [[ -s "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || { echo "REJECT: view source changed: $path" >&2; exit 1; }
done <"$SRC"
(( VIEWS == CAMERAS )) || { echo "REJECT: expected $CAMERAS views, found $VIEWS" >&2; exit 1; }
# A nominal four-camera set is not a turnaround if all yaws are clustered at
# the front. The projection can only improve genuinely observed regions, so a
# *promotion* requires every circular gap to be bounded. Ordinary 0/90/180/270
# and 45-degree eight-view turntables pass; 0/10/20/30 does not. A one-front or
# front/back A/B deliberately does not meet this bar, but remains useful and
# valid as long as its blind texels retain the native generated base.
MAX_YAW_GAP="$(sed -n 's/^view yaw=\([^ ]*\).*/\1/p' "$SRC" | LC_ALL=C sort -n | awk '
  NR==1 { first=$1; prev=$1; next }
  { gap=$1-prev; if (gap>max) max=gap; prev=$1 }
  END { if (NR) { gap=first+360-prev; if (gap>max) max=gap; printf "%.6f", max } }
')"
[[ "$MAX_YAW_GAP" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "REJECT: could not determine observed-view yaw coverage" >&2; exit 1; }
MAX_ALLOWED_YAW_GAP="${NATIVE_OBSERVED_MAX_YAW_GAP_DEGREES:-135}"
[[ "$MAX_ALLOWED_YAW_GAP" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "NATIVE_OBSERVED_MAX_YAW_GAP_DEGREES must be numeric" >&2; exit 2; }
if (( REQUIRE_TURNAROUND )); then
  (( CAMERAS >= 4 )) || { echo "REJECT: promotion needs a real 4--8-view turnaround, not $CAMERAS view(s)" >&2; exit 1; }
  awk -v got="$MAX_YAW_GAP" -v max="$MAX_ALLOWED_YAW_GAP" 'BEGIN { exit !(got <= max) }' || {
    echo "REJECT: observed yaw coverage has a ${MAX_YAW_GAP}° gap (> ${MAX_ALLOWED_YAW_GAP}°): use a real 4--8-view turnaround" >&2
    exit 1
  }
fi
MIN_VIEW="${NATIVE_OBSERVED_MIN_VIEW_COVERAGE_PCT:-0.5}"
PAINTED=0
while IFS= read -r pct; do
  awk -v got="$pct" -v min="$MIN_VIEW" 'BEGIN { exit !(got >= min) }' || { echo "REJECT: view paint $pct% < $MIN_VIEW%" >&2; exit 1; }
  ((PAINTED += 1))
done < <(sed -n 's/^view yaw=.* painted_percent=\([^ ]*\).*/\1/p' "$QC")
(( PAINTED == CAMERAS )) || { echo "REJECT: per-view paint evidence missing ($PAINTED/$CAMERAS)" >&2; exit 1; }
TOPO="$($CP/mesh_topo "$GLB")"
[[ "$TOPO" =~ open=([0-9]+) ]] && (( BASH_REMATCH[1] == 0 )) || { echo "REJECT: candidate topology invalid: $TOPO" >&2; exit 1; }
RIGGED_GLB="$(sed -n 's/^rigged_candidate_glb=\([^ ]*\) sha256=.*/\1/p' "$SRC")"
if [[ -n "$RIGGED_GLB" ]]; then
  RIGGED_HASH="$(sed -n 's/^rigged_candidate_glb=.* sha256=\([0-9a-f]*\)$/\1/p' "$SRC")"
  RIGGED_GLB="$(dirname "$SRC")/$RIGGED_GLB"
  [[ -s "$RIGGED_GLB" && "$RIGGED_HASH" =~ ^[0-9a-f]{64}$ ]] || { echo "REJECT: missing rigged observed candidate" >&2; exit 1; }
  [[ "$(sha256sum "$RIGGED_GLB" | awk '{print $1}')" == "$RIGGED_HASH" ]] || { echo "REJECT: rigged observed candidate hash changed" >&2; exit 1; }
  RIGGED_TOPO="$($CP/mesh_topo "$RIGGED_GLB")"
  [[ "$RIGGED_TOPO" =~ open=([0-9]+) ]] && (( BASH_REMATCH[1] == 0 )) || { echo "REJECT: rigged candidate topology invalid: $RIGGED_TOPO" >&2; exit 1; }
  for node in Hips LeftUpLeg RightUpLeg Spine LeftLeg RightLeg Spine1 LeftFoot RightFoot Spine2 LeftToeBase RightToeBase Neck LeftShoulder RightShoulder Head LeftArm RightArm LeftForeArm RightForeArm LeftHand RightHand; do
    grep -a -F -q "\"name\":\"mixamorig:$node\"" "$RIGGED_GLB" || { echo "REJECT: rigged candidate lacks exact Mixamo node: $node" >&2; exit 1; }
  done
  RIG_SCORE="$($CP/rig_score "$RIGGED_GLB" 55 2>&1 || true)"
  [[ "$RIG_SCORE" =~ maxfan=([0-9]+) ]] && (( BASH_REMATCH[1] <= 7 )) || { echo "REJECT: rigged candidate fanout invalid: $RIG_SCORE" >&2; exit 1; }
  [[ "$RIG_SCORE" =~ TOTAL=([0-9.]+) ]] && awk -v total="${BASH_REMATCH[1]}" 'BEGIN { exit !(total >= 0.50) }' || {
    echo "REJECT: rigged candidate score invalid: $RIG_SCORE" >&2; exit 1;
  }
fi
printf 'VERIFIED native observed projection candidate: views=%s max_yaw_gap=%sdeg turnaround_required=%s native-base=%s telea=0 seam=%s/255\n' \
  "$CAMERAS" "$MAX_YAW_GAP" "$REQUIRE_TURNAROUND" "$NATIVE_BASE" "$SEAM"
