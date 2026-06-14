#!/usr/bin/env bash
# pack_glb.sh — compress a pixal3d GLB to a shippable asset (geometry + textures), NO quality loss
# in the geometry and (with UASTC) near-lossless textures. Pure CPU post-step, no GPU.
#
#   ./pack_glb.sh in.glb out.glb [hero|small]
#     hero  (default) : UASTC -tq 10  — near-lossless textures, larger (hero/inspection asset)
#     small           : ETC1S -tc     — supercompressed, much smaller (LOD / bandwidth-bound)
#
# Applies: KHR_mesh_quantization + EXT_meshopt_compression (-cc) and KTX2/BasisU (-tc/-tu).
# gltfpack ships a static linux binary with BasisU baked in (no separate toktx/basisu needed):
#   meshoptimizer releases -> gltfpack-ubuntu.zip ; installed at ~/.local/bin/gltfpack.
# Verify the result loads in model-viewer (it supports both extensions).
set -euo pipefail
GP="${GLTFPACK:-$HOME/.local/bin/gltfpack}"
in="${1:?usage: pack_glb.sh in.glb out.glb [hero|small]}"
out="${2:?usage: pack_glb.sh in.glb out.glb [hero|small]}"
mode="${3:-hero}"
case "$mode" in
  hero)  tex=(-tu -tq 10) ;;          # UASTC, near-lossless
  small) tex=(-tc) ;;                  # ETC1S, supercompressed
  *) echo "unknown mode: $mode (hero|small)" >&2; exit 2 ;;
esac
"$GP" -i "$in" -o "$out" -cc -tj "$(nproc)" "${tex[@]}"
printf '%s (%s)  %.1f MB -> %.1f MB\n' "$out" "$mode" \
  "$(echo "scale=1;$(stat -c%s "$in")/1048576"|bc)" \
  "$(echo "scale=1;$(stat -c%s "$out")/1048576"|bc)"
