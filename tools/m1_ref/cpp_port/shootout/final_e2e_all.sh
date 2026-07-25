#!/usr/bin/env bash
# Run the whole model table through final_e2e_dc_rig.sh, one at a time.
#
# SEQUENTIAL BY DESIGN: image_to_rig peaks ~8 GB and the box is shared. The flock inside the
# driver already serialises, but running them from one loop keeps the queue visible and lets a
# single Ctrl-C stop the sweep rather than leaving a fan of queued jobs behind the lock.
#
# usage: final_e2e_all.sh <outroot> [model ...]      (default: every model in the table)
set -euo pipefail

OUTROOT="${1:?output root}"; shift || true
CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOOT=/mnt/hdd/3d/avatar-shootout/_shootout_out

declare -A INPUT=(
  [miku]="$SHOOT/v8_production_e2e_20260724/input_birefnet_rgba.png"
  [soldier]="$SHOOT/soldier_matte.png"
  [gilly]="$SHOOT/runbook_image_to_rig/gilly/gilly_matte.png"
  [char1]="$SHOOT/char1_matte.png"
  [toy1]="$SHOOT/toy1_matte.png"
  [toy2]="$SHOOT/toy2_matte.png"
)
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(miku soldier gilly char1 toy1 toy2)

mkdir -p "$OUTROOT"
for m in "${MODELS[@]}"; do
  img="${INPUT[$m]:-}"
  [ -n "$img" ] && [ -f "$img" ] || { echo "== $m: SKIP (no input: ${img:-unset})"; continue; }
  if [ -f "$OUTROOT/$m/rigged.glb" ]; then echo "== $m: already done, skipping"; continue; fi
  echo "===================== $m ====================="
  if bash "$CPP_DIR/shootout/final_e2e_dc_rig.sh" "$img" "$OUTROOT/$m"; then
    grep -E '^wall_total_s|^peak_mib' "$OUTROOT/$m/metrics.txt" || true
  else
    echo "== $m: FAILED (see $OUTROOT/$m/01_e2e.log)"
  fi
done

echo "===================== summary ====================="
for m in "${MODELS[@]}"; do
  f="$OUTROOT/$m/metrics.txt"
  [ -f "$f" ] || { printf '%-9s no run\n' "$m"; continue; }
  printf '%-9s %s  %s  %s\n' "$m" \
    "$(grep -o 'wall_total_s=[0-9]*' "$f" || echo wall=?)" \
    "$(grep -o 'peak_mib=[0-9]*' "$f" || echo peak=?)" \
    "$(grep -o 'TOTAL=[0-9.]*' "$f" | head -1 || echo score=?)"
done
