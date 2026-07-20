#!/usr/bin/env bash
# Verify one completed native texture asset without rerunning model inference.
#
# Usage:
#   verify_native_texture_asset.sh <textured.glb> [--execution gpu|cpu]
#
# This is deliberately independent of native_image_to_rig.sh so an eye-test
# artifact or a copied delivery can be checked later. It validates the exact
# companions a production native bake must retain: atlas, phase log, live/final
# status, texture sampling QC, and the position-welded topology invariant.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLB="${1:?usage: verify_native_texture_asset.sh <textured.glb> [--execution gpu|cpu]}"
shift
EXPECTED_EXECUTION=""
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

case "$EXPECTED_EXECUTION" in
  gpu)
    grep -q '^gpu=PCI GPU 0 / .*RTX 3060' "$STATUS" || { echo "GPU texture asset was not pinned to PCI GPU 0 / RTX 3060: $STATUS" >&2; exit 1; }
    ;;
  cpu)
    grep -q '^execution=CPU-only native re-atlas; no GPU reserved$' "$STATUS" || { echo "rebake is not recorded as CPU-only: $STATUS" >&2; exit 1; }
    ;;
esac

printf 'VERIFIED native texture asset\n'
printf 'glb=%s\natlas=%s\ntopology=%s\n' "$GLB" "$ATLAS" "$TOPO"
printf 'sampling_verdict=%s\nsource_missing_fraction_before_repair=%s\n' \
  "$(qc_value sampling_verdict)" "$(qc_value source_missing_fraction_before_repair)"
