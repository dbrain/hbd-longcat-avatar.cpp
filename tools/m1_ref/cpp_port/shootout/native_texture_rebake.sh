#!/usr/bin/env bash
# CPU-only native texture rebake from a saved texture_mesh_native PBR dump.
#
# Use after a native inference run to A/B atlas resolution or conservative cleanup without
# rerunning DINO / texture diffusion. This intentionally holds no GPU lock: it performs only
# xatlas, CPU rasterisation, and native GLB writing.
#
# Usage:
#   native_texture_rebake.sh <refined.glb> <native-dump-dir> <out.glb> [texture_rebake_native args...]
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH="${1:?usage: native_texture_rebake.sh <refined.glb> <native-dump-dir> <out.glb> [args...]}"
DUMP="${2:?need native PBR dump directory}"
OUT="${3:?need output glb}"
shift 3

BIN="$CP/texture_rebake_native"
[[ -x "$BIN" ]] || { echo "missing $BIN; build it first: cd $CP && ./build.sh texture_rebake_native" >&2; exit 2; }
[[ -f "$MESH" ]] || { echo "missing mesh: $MESH" >&2; exit 2; }
[[ -f "$DUMP/native_pbr_feats.npy" && -f "$DUMP/native_pbr_coords.npy" ]] || {
  echo "missing native PBR dump in $DUMP (run native_texture_run.sh with --dump-dir first)" >&2; exit 2;
}
mkdir -p "$(dirname "$OUT")"
"$BIN" --mesh "$MESH" --pbr-dir "$DUMP" --out "$OUT" "$@"
[[ -s "$OUT" ]] || { echo "rebake produced no GLB: $OUT" >&2; exit 1; }
