#!/usr/bin/env bash
# Reproducible native texture + rig hand-off for ANY cleaned image-to-mesh pair.
#
# Usage:
#   native_image_to_rig.sh <refined.glb> <same-frame-rgba-or-black-matte.png> <out-dir> [label]
#
# Produces three independently baked native Trellis assets (high/medium/low),
# then deterministically rigs the highest quality candidate that passes the
# structural gate.  The source image and refined mesh must share a frame; a
# front-only image is deliberately not treated as a 360-degree observation.
# All model inference is bound to physical PCI GPU 0 (the RTX 3060).
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFINED="${1:?usage: native_image_to_rig.sh <refined.glb> <image.png> <out-dir> [label]}"
IMAGE="${2:?need same-frame source image}"
OUT="${3:?need output directory}"
LABEL="${4:-$(basename "$OUT")}" 

[[ -f "$REFINED" ]] || { echo "missing refined mesh: $REFINED" >&2; exit 2; }
[[ -f "$IMAGE" ]] || { echo "missing source image: $IMAGE" >&2; exit 2; }
for bin in native_texture_run.sh mesh_sample_main rig_texture_chain.sh rig_score mesh_topo; do
  [[ -x "$CP/shootout/$bin" || -x "$CP/$bin" ]] || { echo "missing executable: $bin" >&2; exit 2; }
done

SKIN_GGUF="${SKIN_VAE_GGUF:-/home/dbrain/models/3d/rig/skin_vae_gguf}"
QWEN3_W="${QWEN3_W:-/home/dbrain/models/3d/rig/qwen3_w}"
R1W_SRC="${R1W_SRC:-/home/dbrain/models/3d/rig/r1w_real}"
for d in "$SKIN_GGUF" "$QWEN3_W" "$R1W_SRC"; do
  [[ -d "$d" ]] || { echo "missing rig weights: $d" >&2; exit 2; }
done

mkdir -p "$OUT"
[[ "$REFINED" == "$OUT/refined_geometry.glb" ]] || ln -sfn "$REFINED" "$OUT/refined_geometry.glb"
ln -sfn "$IMAGE" "$OUT/input.png"
# The refined-mesh stage canonicalizes characters with +Z forward.  Pin this
# convention for the structural Mixamo-name mapper so left/right does not
# depend on a weak toe cue from a stylised or partially reconstructed foot.
export RIG_BONE_FACING="${RIG_BONE_FACING:-+z}"
# 512/2048 is the clean, economical default.  1024/4096 is an explicitly
# requested hero-detail tier: it takes materially longer on the reserved 3060
# and must still pass the visual gate before publication.
NATIVE_HIGH_RESOLUTION="${NATIVE_HIGH_RESOLUTION:-512}"
NATIVE_HIGH_ATLAS="${NATIVE_HIGH_ATLAS:-2048}"
[[ "$NATIVE_HIGH_RESOLUTION" == 512 || "$NATIVE_HIGH_RESOLUTION" == 1024 ]] || {
  echo "NATIVE_HIGH_RESOLUTION must be 512 or 1024" >&2; exit 2;
}

run_texture_level() {
  local name="$1" faces="$2" resolution="$3" atlas="$4"
  local out="$OUT/native_${name}_textured.glb"
  echo "== $LABEL: native $name (${faces} faces, ${atlas}px atlas) =="
  "$CP/shootout/native_texture_run.sh" "$REFINED" "$IMAGE" "$out" \
    --resolution "$resolution" --texsize "$atlas" --decimate "$faces"
  "$CP/mesh_topo" "$out"
}

rig_ok() {
  local file="$1" report fan total
  report="$("$CP/rig_score" "$file" 55 2>&1 || true)"
  printf '%s\n' "$report"
  [[ "$report" =~ maxfan=([0-9]+) ]] || return 1; fan="${BASH_REMATCH[1]}"
  [[ "$report" =~ TOTAL=([0-9.]+) ]] || return 1; total="${BASH_REMATCH[1]}"
  (( fan <= 7 )) && awk "BEGIN { exit !($total >= 0.50) }" \
    && grep -a -q 'mixamorig:Hips' "$file"
}

try_rig() {
  local name="$1"
  local mesh="$OUT/native_${name}_textured.glb"
  local samples="$OUT/rig_inputs_${name}"
  local candidate="$OUT/rig_candidate_native_${name}.glb"
  echo "== $LABEL: deterministic rig candidate from native $name =="
  mkdir -p "$samples"
  "$CP/mesh_sample_main" "$mesh" "$samples"
  if ! PIXAL3D_GGUF_DIR="$SKIN_GGUF" R1W_SRC="$R1W_SRC" \
    "$CP/rig_texture_chain.sh" "$samples" "$mesh" "$QWEN3_W" "$candidate" 20; then
    return 1
  fi
  # A lower LOD may yield the cleanest skeleton.  Its sampled skin weights are
  # still transferred onto the high native texture asset, so animation never
  # inherits a low-resolution atlas merely because it won the rig gate.
  if [[ "$name" != high ]]; then
    "$CP/combine_rig_tex_main" "$OUT/native_high_textured.glb" /tmp/skintokens_e2e "$candidate" \
      --sampled "$samples/vertices.npy" --skin /tmp/skintokens_e2e/gen_skin_pred.npy \
      --joints /tmp/skintokens_e2e/gen_joints.npy --parents /tmp/skintokens_e2e/gen_parents.npy
  fi
  "$CP/mesh_topo" "$candidate"
  if rig_ok "$candidate"; then
    cp -f "$candidate" "$OUT/hymotion_rigged.glb"
    SELECTED_RIG="$name"
    return 0
  fi
  return 1
}

run_texture_level high   300000 "$NATIVE_HIGH_RESOLUTION" "$NATIVE_HIGH_ATLAS"
run_texture_level medium 150000 512 1024
run_texture_level low     50000 512 1024

SELECTED_RIG=""
for level in high medium low; do
  try_rig "$level" && break || echo "== $LABEL: rejected native $level rig; trying next LOD ==" >&2
done
[[ -n "$SELECTED_RIG" ]] || { echo "FAIL: no native rig candidate passed quality gate" >&2; exit 1; }

cat >"$OUT/stages.json" <<JSON
{"subject":"$LABEL · native refined-mesh image-to-rig runbook","input":"input.png","stages":[
 {"file":"hymotion_rigged.glb","label":"Hymotion-ready · native $SELECTED_RIG rig","note":"native Trellis texture; deterministic beam rig; maxfan ≤ 7 and rig_score ≥ 0.50"},
 {"file":"native_high_textured.glb","label":"HIGH · native textured","note":"300k target faces · ${NATIVE_HIGH_RESOLUTION}px model · ${NATIVE_HIGH_ATLAS} atlas"},
 {"file":"native_medium_textured.glb","label":"MEDIUM · native textured","note":"150k target faces · 1024 atlas"},
 {"file":"native_low_textured.glb","label":"LOW · native textured","note":"50k target faces · 1024 atlas"}
]}
JSON
echo "== DONE: $OUT (native $SELECTED_RIG rig selected) =="
