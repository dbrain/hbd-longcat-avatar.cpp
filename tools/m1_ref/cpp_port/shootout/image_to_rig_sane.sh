#!/usr/bin/env bash
# Locked, repeatable image -> clean textured mesh -> Hymotion-ready rig shootout.
#
# Usage:
#   image_to_rig_sane.sh <miku|gilly|soldier> [all|high|medium|low]
#
# `all` emits the high/medium/low clean meshes, three high-detail texture A/Bs,
# and selects a valid Hymotion rig by scoring high -> medium -> low candidates.
# This decision is based on rig quality, never on the model name.
# All GPU work is deliberately bound to the physical RTX 3060 (PCI bus order).
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"
MODEL="${1:?model: miku, gilly, or soldier}"
LEVEL="${2:-all}"

case "$MODEL" in
  miku)
    IMAGE=/mnt/hdd/3d/avatar-shootout/_shootout_out/miku_prod1536_matte.png
    CACHE=/mnt/hdd/3d/avatar-shootout/_shootout_out/ab_part_retopo_tex/stage_adaptive
    ;;
  gilly)
    IMAGE=/mnt/hdd/3d/avatar-shootout/assets/gilly.png
    CACHE="$OUT_ROOT/gilly/cache"
    ;;
  soldier)
    IMAGE=/mnt/hdd/3d/avatar-shootout/_shootout_out/inline_soldier
    IMAGE="$IMAGE/input.png"
    CACHE=/mnt/hdd/3d/avatar-shootout/_shootout_out/inline_soldier
    ;;
  *) echo "unknown model '$MODEL' (expected miku, gilly, or soldier)" >&2; exit 2 ;;
esac
case "$LEVEL" in all|high|medium|low) ;; *) echo "unknown level '$LEVEL'" >&2; exit 2 ;; esac

# CUDA normally enumerates by a fast-but-unstable order.  PCI order is stable here,
# where index 0 is the 3060 and index 1 is the owner's busy 5060 Ti.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)"
if [[ "$GPU_NAME" != *"RTX 3060"* ]]; then
  echo "refusing to run: PCI GPU 0 is '$GPU_NAME', not the reserved RTX 3060" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/$MODEL"
OUT_DIR="$OUT_ROOT/$MODEL"
ln -sfn "$IMAGE" "$OUT_DIR/input.png"

# One global lock is intentional: do not let two agents quietly collide on the 3060.
exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
flock -n 9 || { echo "another image_to_rig job owns the 3060 lock" >&2; exit 1; }

BASE=("$CP/image_to_rig" --model /home/dbrain/models/3d/geo --image "$IMAGE" --moge
      --no-quad --tex-dit cross --tex-volume-direct --tex-fallback-r 8)

ensure_cache() {
  if [[ -f "$CACHE/refined.glb" && -f "$CACHE/pbr_feats.bin" && -f "$CACHE/pbr_coords.bin" ]]; then
    return
  fi
  echo "== $MODEL: creating deterministic refined cache on the RTX 3060 =="
  mkdir -p "$CACHE"
  "${BASE[@]}" --no-rig --stage-dir "$CACHE" --out "$CACHE/cache_textured.glb"
}

run_level() {
  local name="$1" faces="$2" texsize="$3" rig="$4" stem="${5:-$1}"
  local out="$OUT_DIR/${stem}_$([[ "$rig" == 1 ]] && echo rigged || echo textured).glb"
  local rig_args=(--no-rig)
  local bone_args=()
  [[ "$rig" == 1 ]] && rig_args=()
  # Canonicalized image-to-rig meshes use +Z forward. The automatic facing finder
  # cannot reliably infer that from generated leg topology, so pin the convention.
  [[ "$rig" == 1 ]] && bone_args=(--bone-facing +z)
  echo "== $MODEL: $name (${faces} target faces, ${texsize}px atlas, rig=$rig) =="
  "${BASE[@]}" --from-refined "$CACHE" --stage-dir "$OUT_DIR" --decimate "$faces" --texsize "$texsize" \
    "${rig_args[@]}" "${bone_args[@]}" --out "$out"
  "$CP/mesh_topo" "$out"
  if [[ "$rig" == 1 && -x "$CP/rig_score" ]]; then
    "$CP/rig_score" "$out" 55 || true
  fi
}

rig_ok() {
  local file="$1" report fan total
  report="$("$CP/rig_score" "$file" 55 2>&1 || true)"
  printf '%s\n' "$report"
  [[ "$report" =~ maxfan=([0-9]+) ]] || return 1; fan="${BASH_REMATCH[1]}"
  [[ "$report" =~ TOTAL=([0-9.]+) ]] || return 1; total="${BASH_REMATCH[1]}"
  (( fan <= 6 )) && awk "BEGIN { exit !($total >= 0.50) }" \
    && strings "$file" | grep -q 'mixamorig:Hips'
}

SELECTED_RIG=""
try_rig() {
  local name="$1" faces="$2" texsize="$3"
  local candidate="$OUT_DIR/rig_candidate_${name}_rigged.glb"
  run_level "$name" "$faces" "$texsize" 1 "rig_candidate_$name"
  if rig_ok "$candidate"; then
    cp -f "$candidate" "$OUT_DIR/hymotion_rigged.glb"
    SELECTED_RIG="$name"
    return 0
  fi
  echo "== $MODEL: rejected $name rig candidate; trying the next conditioning rung ==" >&2
  return 1
}

run_texture_variants() {
  # These are comparison artifacts, not alternate production assets.  The explicit
  # cross DiT in BASE is the all-round production choice (never inherit the binary's
  # mutable default); front projection is the detail ceiling and exposes its
  # unobserved-back limitation; tex-dit=proj is the second generative texture model
  # with identical geometry/UVs.
  echo "== $MODEL: high-resolution texture variants =="
  "${BASE[@]}" --from-refined "$CACHE" --stage-dir "$OUT_DIR" --decimate 300000 --texsize 2048 \
    --no-rig --tex-dit proj --out "$OUT_DIR/high_generative_proj.glb"
  "$CP/mesh_topo" "$OUT_DIR/high_generative_proj.glb"
  "${BASE[@]}" --from-refined "$CACHE" --stage-dir "$OUT_DIR" --decimate 300000 --texsize 2048 \
    --no-rig --tex-project --out "$OUT_DIR/high_front_projected.glb"
  "$CP/mesh_topo" "$OUT_DIR/high_front_projected.glb"
}

write_manifest() {
  local rig_label="Hymotion-ready · ${SELECTED_RIG:-unvalidated} rig"
  local rig_note="selected by maxfan ≤ 6, rig_score ≥ 0.50, and a Mixamo Hips root; source candidate retained for audit"
  cat >"$OUT_DIR/stages.json" <<JSON
{"subject":"$MODEL · locked image-to-rig runbook","input":"input.png","stages":[
 {"file":"hymotion_rigged.glb","label":"$rig_label","note":"$rig_note"},
 {"file":"high_textured.glb","label":"HIGH · textured","note":"300k target faces · 2048 atlas"},
 {"file":"medium_textured.glb","label":"MEDIUM · textured","note":"150k target faces · 1024 atlas"},
 {"file":"low_textured.glb","label":"LOW · textured","note":"50k target faces · 1024 atlas; QEM only, no quad retopo"},
 {"file":"high_generative_proj.glb","label":"Texture A/B · generative proj","note":"same high mesh; Pixal3D texture DiT"},
 {"file":"high_front_projected.glb","label":"Texture A/B · front projection","note":"same high mesh; sharp observed front, synthesized/unobserved back"}
]}
JSON
}

ensure_cache
if [[ "$LEVEL" == all || "$LEVEL" == high ]]; then
  run_level high 300000 2048 0
  run_texture_variants
  try_rig high 300000 2048 || true
fi
if [[ "$LEVEL" == all || "$LEVEL" == medium ]]; then
  run_level medium 150000 1024 0
  [[ -n "$SELECTED_RIG" ]] || try_rig medium 150000 1024 || true
fi
if [[ "$LEVEL" == all || "$LEVEL" == low ]]; then
  run_level low 50000 1024 0
  [[ -n "$SELECTED_RIG" ]] || try_rig low 50000 1024 || true
fi
[[ -n "$SELECTED_RIG" ]] || { echo "FAIL: no rig candidate passed the quality gate" >&2; exit 1; }
write_manifest
echo "== DONE: $OUT_DIR =="
