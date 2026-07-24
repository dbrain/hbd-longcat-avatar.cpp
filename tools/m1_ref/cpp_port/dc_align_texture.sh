#!/usr/bin/env bash
# DC-aligned texture delivery -- brings Pixal3D's narrow-band dual-contour remesh into native's
# texture path so the delivered/textured surface is (a) Python-smooth and (b) sits ON the sparse PBR
# volume's decode surface, fixing BOTH the geometric blockiness and the face/boundary texture dirt.
#
# Mechanism (mirrors o_voxel.postprocess.to_glb with remesh=True, band=1):
#   1. narrow_band_dc_probe : narrow-band UDF dual contouring of coarse.glb (the decode surface the
#                             sparse PBR volume was generated on) -> smooth watertight surface ON the
#                             volume iso-surface.  This is the native equivalent of Python's
#                             cumesh.remeshing.remesh_narrow_band_dc (both use mass-point DC).
#   2. dc_simplify_glb      : CuMesh QEM simplify to the delivery budget + native Taubin low-pass to
#                             erase the residual MC-solid staircase the DC inherits from coarse.glb
#                             (Python's DC input is smoother, so it needs no post-smooth; native's
#                             coarse is the stride-2 MC-solid, dihedral p95~79, hence Taubin).
#   3. texture_rebake_native --bake volume-trilinear : unwrap + volume-direct trilinear sample.  The
#                             DC surface sits on the volume, so ~0% of texels fall back to the shell
#                             guard -> no cross-surface projection slide -> crisp clothing boundaries.
#
# Usage: dc_align_texture.sh <pbr_cache_dir> <out.glb> [target_faces=945000] [taubin_iters=5] [texsize=4096] [dc_res=1024]
#   pbr_cache_dir must contain: coarse.glb, pbr_feats.bin, pbr_coords.bin, resolution.bin
set -euo pipefail

CACHE="${1:?pbr_cache_dir}"
OUT="${2:?out.glb}"
TARGET="${3:-945000}"
TAUBIN="${4:-5}"
TEXSIZE="${5:-4096}"
DCRES="${6:-1024}"

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(dirname "$OUT")"
STEM="$(basename "$OUT" .glb)"
: "${CUDA_VISIBLE_DEVICES:=GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6}"  # 3060 only
export CUDA_VISIBLE_DEVICES

echo ">> [1/3] narrow-band DC remesh of $CACHE/coarse.glb @${DCRES}"
"$HERE/narrow_band_dc_probe" "$CACHE/coarse.glb" "$WORK/${STEM}.dc.glb" "$DCRES" 1

echo ">> [2/3] QEM simplify -> ${TARGET}f + Taubin x${TAUBIN}"
"$HERE/dc_simplify_glb" "$WORK/${STEM}.dc.glb" "$WORK/${STEM}.dcsimp.glb" "$TARGET" "$TAUBIN" 0.5 0.53

echo ">> [3/3] volume-trilinear bake"
"$HERE/texture_rebake_native" --mesh "$WORK/${STEM}.dcsimp.glb" --pbr-cache "$CACHE" \
  --bake volume-trilinear --decimate 0 --texsize "$TEXSIZE" \
  --out "$OUT" --atlas-out "${OUT%.glb}_atlas.png"

echo ">> DONE: $OUT"
