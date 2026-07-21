#!/usr/bin/env bash
# Read-only evidence gate for a 4--8-view observed-image hybrid projection.
#
# Usage:
#   verify_observed_projection.sh <delivery-dir>
#
# A one-front-image overlay remains a useful labelled A/B. It must not become
# the native production texture merely because it wrote a plausible GLB. This
# gate establishes a real, consistent 4--8-view candidate: it checks the
# recorded source hashes, unique yaw coverage, per-view paint evidence,
# silhouette alignment, and the fact that overlay holes retained the native
# base rather than falling through to atlas-space colour invention.
set -euo pipefail

OUT="${1:?usage: verify_observed_projection.sh <delivery-dir>}"
SRC="$OUT/projection_source.txt"
LOG="$OUT/projection.log"
GLB="$OUT/high_hybrid_projected.glb"
[[ -s "$SRC" && -s "$LOG" && -s "$GLB" ]] || {
  echo "REJECT: projection needs projection_source.txt, projection.log, and high_hybrid_projected.glb in $OUT" >&2; exit 1;
}

value() { awk -F= -v key="$1" '$1==key {print substr($0,length(key)+2); exit}' "$SRC"; }
CAMERAS="$(value camera_count)"
[[ "$CAMERAS" =~ ^[4-8]$ ]] || { echo "REJECT: validated observed projection needs 4--8 cameras (got '${CAMERAS:-missing}')" >&2; exit 1; }
[[ "$(value mode)" == 'observed-view hybrid A/B; native_high_textured.glb remains production default' ]] || {
  echo "REJECT: projection provenance has unexpected mode: $SRC" >&2; exit 1;
}

declare -A seen=()
views=0
while IFS= read -r line; do
  [[ "$line" == view\ * ]] || continue
  yaw="$(sed -n 's/^view yaw=\([^ ]*\).*/\1/p' <<<"$line")"
  path="$(sed -n 's/.* path=\(.*\) sha256=.*/\1/p' <<<"$line")"
  expected="$(sed -n 's/.* sha256=\([0-9a-f]*\) origin=.*/\1/p' <<<"$line")"
  [[ -n "$yaw" && -n "$path" && "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "REJECT: malformed view provenance: $line" >&2; exit 1; }
  [[ -z "${seen[$yaw]:-}" ]] || { echo "REJECT: duplicate yaw in projection provenance: $yaw" >&2; exit 1; }
  seen[$yaw]=1; ((views += 1))
  [[ -s "$path" ]] || { echo "REJECT: observed projection source is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || { echo "REJECT: observed view hash changed at yaw $yaw: $path" >&2; exit 1; }
done <"$SRC"
(( views == CAMERAS )) || { echo "REJECT: camera_count=$CAMERAS but provenance records $views views" >&2; exit 1; }

# These diagnostics are emitted by the real projection implementation. Require
# every submitted camera to contribute a meaningful fraction, so an incorrect
# yaw/crop cannot silently pass as a nominal 4-view input set.
MIN_VIEW_PCT="${PROJECTION_MIN_VIEW_COVERAGE_PCT:-0.5}"
painted=0
while IFS= read -r pct; do
  awk -v got="$pct" -v min="$MIN_VIEW_PCT" 'BEGIN { exit !(got >= min) }' || {
    echo "REJECT: one observed view painted only ${pct}% (<${MIN_VIEW_PCT}%) of covered texels: $LOG" >&2; exit 1;
  }
  ((painted += 1))
done < <(sed -n 's/.* painted *\([0-9.]*\)% of covered.*/\1/p' "$LOG")
(( painted == CAMERAS )) || { echo "REJECT: projection log has $painted per-view paint reports, expected $CAMERAS" >&2; exit 1; }

aligns="$(rg -c '^\[texproj\] view [0-9]+ align: scale=' "$LOG" || true)"
(( aligns == CAMERAS )) || { echo "REJECT: every observed view must have a fitted silhouette alignment ($aligns/$CAMERAS): $LOG" >&2; exit 1; }

# Overlay mode should preserve the native material in genuinely unobserved
# texels. Any Telea fallback means a non-observed atlas neighbourhood invented
# colour and therefore cannot be promoted as an observed-view result.
telea="$(sed -n 's/.* \([0-9][0-9]*\) telea-fallback.*/\1/p' "$LOG" | tail -1)"
[[ "$telea" =~ ^[0-9]+$ ]] || { echo "REJECT: projection log lacks hole-fill accounting: $LOG" >&2; exit 1; }
(( telea == 0 )) || { echo "REJECT: observed projection used $telea atlas-space Telea fallback texels: $LOG" >&2; exit 1; }

MAX_SEAM="${PROJECTION_MAX_SEAM_ABSDIFF_255:-32}"
seam="$(sed -n 's/.*|viewA-viewB| = \([0-9.]*\)\/255.*/\1/p' "$LOG" | tail -1)"
if [[ -n "$seam" ]]; then
  awk -v got="$seam" -v max="$MAX_SEAM" 'BEGIN { exit !(got <= max) }' || {
    echo "REJECT: observed-view seam disagreement ${seam}/255 > ${MAX_SEAM}/255: $LOG" >&2; exit 1;
  }
fi

printf 'VERIFIED observed projection candidate: cameras=%s views_painted=%s min_view_pct=%s telea=0 seam=%s\n' \
  "$CAMERAS" "$painted" "$MIN_VIEW_PCT" "${seam:-none}"
