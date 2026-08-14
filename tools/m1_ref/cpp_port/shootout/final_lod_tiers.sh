#!/usr/bin/env bash
# Hero -> game-asset ladder from ONE geometry run.
#
# Every tier re-enters at the bake from the stage cache (--from-refined), so the ~5 min diffusion
# and the DC remesh are paid ONCE and each additional tier costs only its bake + rig (~30-60s).
# Every tier is a decimation of the SAME parity mesh and carries the SAME parity texture, plus a
# tangent-space normal map baked from the pre-decimation dense surface — that is what lets the
# 15k-face game tier still read as the hero asset.
#
# usage: final_lod_tiers.sh <image> <stage_dir> <outdir> [tier ...]
#   tier spec: name:faces:atlas   (default ladder below)
set -euo pipefail

# NB: no apostrophes in a ${x:?msg} message — bash parses quoting inside it even within double
# quotes, and one contraction here turned the whole script into "unexpected EOF".
IMG="${1:?input image - must be the same source image the stage cache was built from}"
STAGE="${2:?stage dir from a previous --dc-remesh run}"
OUT="${3:?output dir}"
shift 3 || true

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIERS=("$@")
[ ${#TIERS[@]} -gt 0 ] || TIERS=(hero:220000:2048 high:100000:2048 medium:40000:1024 game:15000:1024)

test -f "$STAGE/refined.glb" || { echo "FAIL: $STAGE has no refined.glb (not a --dc-remesh stage dir)"; exit 1; }
mkdir -p "$OUT"

# ONE skeleton for the whole ladder. The first tier generates and caches it; the rest load it and
# transfer the same skin field onto their own mesh. Delete this dir to re-rig from scratch.
RIG_CACHE_DIR="$OUT/rig_cache"
mkdir -p "$RIG_CACHE_DIR"

for spec in "${TIERS[@]}"; do
  IFS=: read -r name faces atlas <<< "$spec"
  echo "===== tier $name: ${faces} faces / ${atlas}px atlas ====="
  RESUME_STAGE="$STAGE" RIG_CACHE="$RIG_CACHE_DIR" SKIP_ANIM=1 \
    bash "$CPP_DIR/shootout/final_e2e_dc_rig.sh" "$IMG" "$OUT/$name" "$atlas" "$faces" \
    || { echo "  tier $name FAILED"; continue; }
done

# One exercise clip on the hero. Because every tier now shares the skeleton, this clip is valid
# for the whole ladder — which is the point of caching the rig.
if [ -f "$OUT/hero/rigged.glb" ] && [ -z "${SKIP_ANIM:-}" ]; then
  # native since 2026-08-14 — see the note in final_e2e_dc_rig.sh
  "$CPP_DIR/motion_retarget" --exercise "$OUT/hero/rigged.glb" "$OUT/hero/rigged.anim.glb" \
    > "$OUT/hero/anim.log" 2>&1 || echo "  WARN: exercise animation failed (see $OUT/hero/anim.log)"
fi

echo "===================== tier ladder ====================="
printf '%-8s %10s %10s %8s %10s %s\n' tier verts faces atlas normalmap rig
for spec in "${TIERS[@]}"; do
  IFS=: read -r name faces atlas <<< "$spec"
  log="$OUT/$name/01_e2e.log"
  [ -f "$log" ] || { printf '%-8s %s\n' "$name" "no run"; continue; }
  vf=$(grep -oE 'verts=[0-9]+ faces=[0-9]+' "$log" | tail -1)
  nm=$(grep -c 'normal map baked' "$log" || true)
  rig=$(grep -oE 'J=[0-9]+ +roots' "$OUT/$name/qc_rig_score.txt" 2>/dev/null | head -1 || echo "?")
  hl=$(grep -oE 'weight health: [A-Z]+' "$OUT/$name/qc_weight_health.txt" 2>/dev/null || echo "?")
  printf '%-8s %s  atlas=%s  nmap=%s  %s %s\n' "$name" "$vf" "$atlas" \
    "$([ "$nm" -gt 0 ] && echo yes || echo NO)" "$rig" "$hl"
done
