#!/usr/bin/env bash
# Verify one completed native texture asset without rerunning model inference.
#
# Usage:
#   verify_native_texture_asset.sh <textured.glb> [--execution gpu|cpu]
#
# This is deliberately independent of native_image_to_rig.sh so an eye-test
# artifact or a copied delivery can be checked later. It validates the exact
# companions a production native bake must retain: atlas, phase log, live/final
# status, texture sampling QC, and the position-welded topology invariant. It
# also proves the external audit atlas is the byte-identical PNG embedded by
# the GLB, so a stale sidecar cannot make a different exported material pass.
#
# Schema-3 QC also measures how much the final encoded atlas changed the
# decoded native PBR material. A bake can have no holes but still be washed out
# or compressed, so this is a hard production gate. Defaults are deliberately
# below the observed clean 512/2048 range (35.82--41.15 dB, MAE
# 0.00188--0.00492): they catch material loss without making a subject-specific
# promise. Labelled A/Bs may override the two environment variables.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLB="${1:?usage: verify_native_texture_asset.sh <textured.glb> [--execution gpu|cpu]}"
shift
EXPECTED_EXECUTION=""
MIN_RECOVERY_PSNR_DB="${NATIVE_TEXTURE_MIN_RECOVERY_PSNR_DB:-30}"
MAX_RECOVERY_MAE_LINEAR="${NATIVE_TEXTURE_MAX_RECOVERY_MAE_LINEAR:-0.020}"
if (( $# > 0 )); then
  [[ "$1" == "--execution" && $# == 2 ]] || { echo "usage: verify_native_texture_asset.sh <textured.glb> [--execution gpu|cpu]" >&2; exit 2; }
  EXPECTED_EXECUTION="$2"
  [[ "$EXPECTED_EXECUTION" == gpu || "$EXPECTED_EXECUTION" == cpu ]] || { echo "--execution must be gpu or cpu" >&2; exit 2; }
fi

[[ -s "$GLB" ]] || { echo "missing textured GLB: $GLB" >&2; exit 1; }
BASE="${GLB%.glb}"
ATLAS="${BASE}_atlas.png"
STATUS="${GLB}.run-status.txt"
STAGE="${GLB}.stage-log.txt"
QC="${GLB}.texture-qc.txt"
for sidecar in "$ATLAS" "$STATUS" "$STAGE" "$QC"; do
  [[ -s "$sidecar" ]] || { echo "missing required native sidecar: $sidecar" >&2; exit 1; }
done

EXTRACTOR="$CP/glb_reader_test"
[[ -x "$EXTRACTOR" ]] || { echo "missing GLB baseColor extractor: $EXTRACTOR" >&2; exit 1; }
EMBEDDED_ATLAS="$(mktemp "${TMPDIR:-/tmp}/native-texture-atlas.XXXXXX.png")"
cleanup_embedded_atlas() { rm -f "$EMBEDDED_ATLAS"; }
trap cleanup_embedded_atlas EXIT
"$EXTRACTOR" "$GLB" --extract-basecolor "$EMBEDDED_ATLAS" >/dev/null
ATLAS_SHA256="$(sha256sum "$ATLAS" | awk '{print $1}')"
EMBEDDED_ATLAS_SHA256="$(sha256sum "$EMBEDDED_ATLAS" | awk '{print $1}')"
[[ "$ATLAS_SHA256" == "$EMBEDDED_ATLAS_SHA256" ]] || {
  echo "REJECT: sidecar atlas does not match the PNG embedded by the GLB: $ATLAS" >&2; exit 1;
}

TOPO="$($CP/mesh_topo "$GLB")" || { echo "could not inspect topology: $GLB" >&2; exit 1; }
[[ "$TOPO" =~ open=([0-9]+) ]] || { echo "topology report lacks open-edge count: $GLB" >&2; exit 1; }
(( BASH_REMATCH[1] == 0 )) || { echo "REJECT: textured asset has open edges: $TOPO" >&2; exit 1; }

status_value() { awk -F= -v key="$1" '$1==key {print substr($0,length(key)+2); exit}' "$STATUS"; }
qc_value() { awk -F= -v key="$1" '$1==key {print substr($0,length(key)+2); exit}' "$QC"; }

[[ "$(status_value launcher_state)" == succeeded ]] || { echo "native launcher did not succeed: $STATUS" >&2; exit 1; }
[[ "$(status_value artifact_state)" == succeeded ]] || { echo "native artifact write did not succeed: $STATUS" >&2; exit 1; }
grep -q 'stage=atlas_encode_complete' "$STAGE" || { echo "native stage log lacks atlas encode completion: $STAGE" >&2; exit 1; }

UNRESOLVED="$(qc_value unresolved_surface_texels_after_chart_repair)"
[[ "$UNRESOLVED" =~ ^[0-9]+$ ]] || { echo "texture QC lacks unresolved-surface count: $QC" >&2; exit 1; }
(( UNRESOLVED == 0 )) || { echo "REJECT: $UNRESOLVED unresolved surface texels: $QC" >&2; exit 1; }
[[ "$(qc_value sampling_verdict)" == complete-after-chart-repair ]] || { echo "texture QC did not complete chart repair: $QC" >&2; exit 1; }

SCHEMA="$(qc_value schema_version)"
RECOVERY_TEXELS="$(qc_value source_recovery_texels)"
RECOVERY_MAE="$(qc_value source_recovery_mae_linear)"
RECOVERY_PSNR="$(qc_value source_recovery_psnr_db)"
[[ "$SCHEMA" =~ ^[0-9]+$ ]] && (( SCHEMA >= 3 )) || {
  echo "REJECT: texture QC lacks schema-3 final-atlas recovery evidence: $QC" >&2; exit 1;
}
[[ "$RECOVERY_TEXELS" =~ ^[1-9][0-9]*$ ]] || {
  echo "REJECT: texture QC lacks sampled final-atlas recovery texels: $QC" >&2; exit 1;
}
[[ "$RECOVERY_MAE" =~ ^[0-9]+([.][0-9]+)?$ && "$RECOVERY_PSNR" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "REJECT: texture QC has invalid final-atlas recovery metrics: $QC" >&2; exit 1;
}
awk -v psnr="$RECOVERY_PSNR" -v min="$MIN_RECOVERY_PSNR_DB" 'BEGIN { exit !(psnr >= min) }' || {
  echo "REJECT: final-atlas recovery PSNR ${RECOVERY_PSNR} dB < ${MIN_RECOVERY_PSNR_DB} dB: $QC" >&2; exit 1;
}
awk -v mae="$RECOVERY_MAE" -v max="$MAX_RECOVERY_MAE_LINEAR" 'BEGIN { exit !(mae <= max) }' || {
  echo "REJECT: final-atlas recovery MAE ${RECOVERY_MAE} > ${MAX_RECOVERY_MAE_LINEAR}: $QC" >&2; exit 1;
}

case "$EXPECTED_EXECUTION" in
  gpu)
    grep -q '^gpu=PCI GPU 0 / .*RTX 3060' "$STATUS" || { echo "GPU texture asset was not pinned to PCI GPU 0 / RTX 3060: $STATUS" >&2; exit 1; }
    ;;
  cpu)
    grep -q '^execution=CPU-only native re-atlas; no GPU reserved$' "$STATUS" || { echo "rebake is not recorded as CPU-only: $STATUS" >&2; exit 1; }
    ;;
esac

printf 'VERIFIED native texture asset\n'
printf 'glb=%s\natlas=%s\natlas_sha256=%s\ntopology=%s\n' "$GLB" "$ATLAS" "$ATLAS_SHA256" "$TOPO"
printf 'sampling_verdict=%s\nsource_missing_fraction_before_repair=%s\n' \
  "$(qc_value sampling_verdict)" "$(qc_value source_missing_fraction_before_repair)"
printf 'final_atlas_recovery=texels=%s mae_linear=%s psnr_db=%s (gate: mae<=%s psnr>=%s)\n' \
  "$RECOVERY_TEXELS" "$RECOVERY_MAE" "$RECOVERY_PSNR" "$MAX_RECOVERY_MAE_LINEAR" "$MIN_RECOVERY_PSNR_DB"
