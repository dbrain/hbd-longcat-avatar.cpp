#!/usr/bin/env bash
# Read-only gate for a native-base observed-image texture candidate.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLB="${1:?usage: verify_native_observed_projection.sh <candidate.glb>}"
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
MIN_VIEW="${NATIVE_OBSERVED_MIN_VIEW_COVERAGE_PCT:-0.5}"
PAINTED=0
while IFS= read -r pct; do
  awk -v got="$pct" -v min="$MIN_VIEW" 'BEGIN { exit !(got >= min) }' || { echo "REJECT: view paint $pct% < $MIN_VIEW%" >&2; exit 1; }
  ((PAINTED += 1))
done < <(sed -n 's/^view yaw=.* painted_percent=\([^ ]*\).*/\1/p' "$QC")
(( PAINTED == CAMERAS )) || { echo "REJECT: per-view paint evidence missing ($PAINTED/$CAMERAS)" >&2; exit 1; }
TOPO="$($CP/mesh_topo "$GLB")"
[[ "$TOPO" =~ open=([0-9]+) ]] && (( BASH_REMATCH[1] == 0 )) || { echo "REJECT: candidate topology invalid: $TOPO" >&2; exit 1; }
printf 'VERIFIED native observed projection candidate: views=%s native-base=%s telea=0 seam=%s/255\n' "$CAMERAS" "$NATIVE_BASE" "$SEAM"
