#!/usr/bin/env bash
# FAST remesh iteration — NO E2E, NO GPU DiT. Runs the remesh offline on the saved Miku occupancy
# (refs/stage5/head_coords.npy) and renders the GEOMETRY so you can eyeball shape/corners/spikes/
# fingers in seconds. This is the loop for tuning the remesh (stride/blur/smooth) without the ~5min
# chain. Texture needs the chain's PBR (use the full E2E for that); for SHAPE crispness this is enough.
#
#   ./iter_remesh.sh <stride> <blur> <smooth>        e.g. ./iter_remesh.sh 3 1 1
#   ./iter_remesh.sh 2 1 0                            # crisper (grid 512), no Taubin
# Output: m_s<stride>b<blur>m<smooth>.ply + geo_<tag>.png (4-view geometry montage).
# Compare against the crisp target: miku_golden_geo.png (the full 3.25M-tri O-Voxel mesh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
S="${1:-3}"; B="${2:-1}"; SM="${3:-1}"; TAG="s${S}b${B}m${SM}"
PY=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
[ -x remesh_test ] || ./build.sh remesh_test
echo ">> remesh stride=$S blur=$B smooth=$SM"
REMESH_COARSE=1 REMESH_SOLID=1 REMESH_STRIDE="$S" REMESH_BLUR="$B" POST_SMOOTH="$SM" DEC_NOUNWRAP=1 \
    ./remesh_test 2>&1 | grep -iE "COARSE-MC|Taubin"
cp miku_remesh_coarse.ply "m_${TAG}.ply"
"$PY" render_mesh.py "m_${TAG}.ply" "geo_${TAG}.png" 2>&1 | grep -i loaded
echo ">> wrote geo_${TAG}.png (compare vs miku_golden_geo.png)"
