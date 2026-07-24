#!/usr/bin/env bash
# O-Voxel-DC + DIRECT-bake parity delivery (2026-07-25).
#
# Python-parity coarse: instead of MC-soliding the binary occupancy (staircase), keep the
# smooth O-Voxel dual-grid decoder mesh (--no-clean) and dual-contour-remesh THAT (Python's
# o_voxel.postprocess.to_glb remesh=True equivalent), then texture it DIRECT (no reproject
# slide). See ~/handoffs/tools/m1_ref/cpp_port/QUALITY-FIXES-MANIFEST-2026-07-24.md and memory
# project_image_to_rig_coarse_parity_ovoxel_dc.
#
# usage: native_ovoxel_dc_parity.sh <input_image_or_matte.png> <outdir> [texsize=2048] [band=1] [taubin=2]
set -euo pipefail

IMG="${1:?input image/matte png}"
OUT="${2:?output dir}"
TEXSIZE="${3:-2048}"
BAND="${4:-1}"
TAUBIN="${5:-2}"

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${GEO_MODEL_DIR:-/home/dbrain/models/3d/geo_f16_native}"
RES="${PIXAL_RESOLUTION:-1024}"
LOCK="/mnt/hdd/3d/avatar-shootout/_shootout_out/.3060-image-to-rig.lock"
# image_to_rig is 3060-ONLY. CUDA enumerates fastest-first so index 0 = 5060 Ti; pin by UUID.
GPU3060="${GPU_3060_UUID:-GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6}"

mkdir -p "$OUT"
cd "$CPP_DIR"

echo "[1/4] geometry decode (--no-clean keeps the O-Voxel mesh) + PBR dump  [3060]"
CUDA_VISIBLE_DEVICES="$GPU3060" nice -n 10 ionice -c2 -n7 \
  flock -w 2400 "$LOCK" ./image_to_rig \
    --model "$MODEL" --image "$IMG" \
    --no-clean --no-refine --dump-geo "$OUT" \
    --out "$OUT/_decode_probe.glb" 2>&1 | tee "$OUT/01_decode.log"
test -f "$OUT/coarse.glb"     || { echo "FAIL: no O-Voxel coarse.glb staged"; exit 1; }
test -f "$OUT/pbr_feats.bin"  || { echo "FAIL: --dump-geo did not cache pbr_feats.bin"; exit 1; }
# --dump-geo does NOT write resolution.bin; texture_rebake reads it from --pbr-cache. Stamp the
# 12-byte size-prefixed int32 tag (int64 size=4, int32 value=RES) it expects.
python3 -c "import struct; open('$OUT/resolution.bin','wb').write(struct.pack('<qi',4,$RES))"

echo "[2/4] DC-remesh the O-Voxel mesh (Python remesh_narrow_band_dc equiv, band=$BAND)  [3060]"
CUDA_VISIBLE_DEVICES="$GPU3060" flock -w 1200 "$LOCK" \
  ./narrow_band_dc_probe "$OUT/coarse.glb" "$OUT/ovoxel_dc.glb" "$RES" "$BAND" 2>&1 | tee "$OUT/02_dc.log"

echo "[3/4] light feature-preserving Taubin (x$TAUBIN)"
./mesh_taubin "$OUT/ovoxel_dc.glb" "$OUT/ovoxel_dc_taubin.glb" "$TAUBIN" 2>&1 | tee "$OUT/03_taubin.log"

echo "[4/4] DIRECT volume bake (RP_VOLUME_DIRECT=1, no reproject slide)  [CPU]"
# ATL_BASECOLOR_SRGB=0 = RAW baseColor like Python's o_voxel.to_glb (np.clip(attrs*255), no OETF).
# The sRGB OETF lifts shadows -> darks look washed / less saturated; raw keeps Python-dark darks.
CUDA_VISIBLE_DEVICES="" RP_VOLUME_DIRECT=1 ATL_BASECOLOR_SRGB=0 \
  ./texture_rebake_native \
    --mesh "$OUT/ovoxel_dc_taubin.glb" \
    --pbr-cache "$OUT" --resolution "$RES" \
    --bake volume-trilinear --texsize "$TEXSIZE" --decimate 220000 \
    --out "$OUT/parity.glb" 2>&1 | tee "$OUT/04_bake.log"

test -f "$OUT/parity.glb" || { echo "FAIL: no parity.glb produced"; exit 1; }
echo "DONE -> $OUT/parity.glb  (geometry: O-Voxel DC; texture: direct volume-trilinear)"
