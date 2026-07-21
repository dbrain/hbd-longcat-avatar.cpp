#!/usr/bin/env bash
# Reproducible native texture + rig hand-off for ANY cleaned image-to-mesh pair.
#
# Usage:
#   native_image_to_rig.sh <refined.glb> <same-frame-rgba-or-black-matte.png> <out-dir> [label]
#
# Produces one authoritative native Trellis material, then CPU-rebakes it onto
# high/medium/low meshes.  This keeps every LOD's appearance consistent and
# avoids spending the reserved 3060 three times on the same source image.
# It then deterministically rigs the highest quality candidate that passes the
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
for bin in native_texture_run.sh native_texture_rebake.sh verify_native_texture_asset.sh texture_rebake_native mesh_sample_main rig_texture_chain.sh rig_score mesh_topo; do
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
NATIVE_UNWRAP="${NATIVE_UNWRAP:-reference}"
[[ "$NATIVE_HIGH_RESOLUTION" == 512 || "$NATIVE_HIGH_RESOLUTION" == 1024 ]] || {
  echo "NATIVE_HIGH_RESOLUTION must be 512 or 1024" >&2; exit 2;
}
[[ "$NATIVE_UNWRAP" == reference || "$NATIVE_UNWRAP" == production ]] || {
  echo "NATIVE_UNWRAP must be reference or production" >&2; exit 2;
}

# A texture can look plausible while being unusable in animation if a later LOD bake introduced a
# boundary. `mesh_topo` position-welds split GLB vertices, so `open=0` is the meaningful invariant
# rather than raw index adjacency. Preserve its full report in the delivery manifest too.
assert_closed_mesh() {
  local file="$1" report open
  report="$("$CP/mesh_topo" "$file")" || { echo "could not inspect topology: $file" >&2; return 1; }
  printf '%s\n' "$report"
  [[ "$report" =~ open=([0-9]+) ]] || { echo "topology report lacks open-edge count: $file" >&2; return 1; }
  open="${BASH_REMATCH[1]}"
  (( open == 0 )) || { echo "REJECT: texture LOD has $open open edges: $file" >&2; return 1; }
}

# A filled PNG alone is insufficient proof of a usable material: retain and gate
# the native bake's original-surface sampling report.  Chart-local repair may
# fix sparse misses, but no covered texel may remain unresolved before the
# cosmetic whole-atlas background dilation.
assert_texture_qc() {
  local file="$1" qc unresolved
  qc="${file}.texture-qc.txt"
  [[ -s "$qc" ]] || { echo "missing native texture QC: $qc" >&2; return 1; }
  unresolved="$(awk -F= '$1=="unresolved_surface_texels_after_chart_repair" {print $2; exit}' "$qc")"
  [[ "$unresolved" =~ ^[0-9]+$ ]] || { echo "texture QC lacks unresolved-surface count: $qc" >&2; return 1; }
  (( unresolved == 0 )) || { echo "REJECT: texture bake leaves $unresolved unresolved surface texels: $qc" >&2; return 1; }
  printf 'texture QC: %s\n' "$(tr '\n' ' ' <"$qc")"
}

# One durable hand-off record, written only after all three texture assets passed the structural
# gate. It intentionally names the high texture as production even if medium/low supplied the best
# skeleton: try_rig transfers that accepted skin onto high, so Hymotion does not silently inherit a
# lower-resolution atlas.
write_texture_delivery_manifest() {
  local rig_state="$1" level="${2:-none}" name file atlas status stage route stage_sha qc qc_sha qc_verdict qc_missing topo
  # `reference` is a material recipe, not a promise that all meshes can use a
  # single global xatlas solve.  Record the measured route from the stage log:
  # direct for clean topology; conformal local islands where sharp/high-curvature
  # geometry would otherwise make the global solve pathological.
  atlas_route_for() {
    local stage_file="$1"
    if [[ ! -f "$stage_file" ]]; then
      printf 'unknown-no-stage-log'
    elif grep -q 'stage=atlas_unwrap_islands_begin' "$stage_file"; then
      printf 'adaptive-conformal-local-islands'
    elif grep -q 'stage=atlas_unwrap_direct_begin' "$stage_file"; then
      printf 'direct-parity-charts'
    else
      printf 'unknown-incomplete-stage-log'
    fi
  }
  {
    printf 'schema_version=2\n'
    printf 'production_texture=native_high_textured.glb\n'
    printf 'native_unwrap=%s\n' "$NATIVE_UNWRAP"
    printf 'high_model_lattice=%s\nhigh_atlas_px=%s\n' "$NATIVE_HIGH_RESOLUTION" "$NATIVE_HIGH_ATLAS"
    printf 'lod_material_contract=medium/low are CPU rebakes of native_high_texture_dump; no second texture inference; adaptive xatlas uses direct parity charts for clean topology or conformal local islands for high curvature\n'
    printf 'topology_gate=position-welded open=0 for every texture LOD\n'
    printf 'texture_qc_gate=zero unresolved rasterised surface texels after chart-local repair; source-missing count is retained for eye-test audit\n'
    printf 'rig_state=%s\nselected_rig=%s\n' "$rig_state" "$level"
    for name in high medium low; do
      file="$OUT/native_${name}_textured.glb"
      atlas="$OUT/native_${name}_textured_atlas.png"
      status="${file}.run-status.txt"
      stage="${file}.stage-log.txt"
      route="$(atlas_route_for "$stage")"
      stage_sha="missing"
      [[ ! -f "$stage" ]] || stage_sha="$(sha256sum "$stage" | awk '{print $1}')"
      qc="${file}.texture-qc.txt"
      qc_sha="missing"; qc_verdict="missing"; qc_missing="missing"
      if [[ -f "$qc" ]]; then
        qc_sha="$(sha256sum "$qc" | awk '{print $1}')"
        qc_verdict="$(awk -F= '$1=="sampling_verdict" {print $2; exit}' "$qc")"
        qc_missing="$(awk -F= '$1=="source_missing_fraction_before_repair" {print $2; exit}' "$qc")"
      fi
      topo="$("$CP/mesh_topo" "$file")"
      printf 'lod=%s file=%s sha256=%s atlas=%s atlas_sha256=%s atlas_route=%s stage_log=%s stage_log_sha256=%s texture_qc=%s texture_qc_sha256=%s sampling_verdict=%s source_missing_fraction_before_repair=%s topology="%s"\n' \
        "$name" "$(basename "$file")" "$(sha256sum "$file" | awk '{print $1}')" \
        "$(basename "$atlas")" "$(sha256sum "$atlas" | awk '{print $1}')" "$route" \
        "$(basename "$stage")" "$stage_sha" "$(basename "$qc")" "$qc_sha" "$qc_verdict" "$qc_missing" "$topo"
      [[ ! -f "$status" ]] || printf 'lod=%s native_texture_status=%s\n' "$name" "$(basename "$status")"
    done
    [[ "$rig_state" != succeeded ]] || printf 'hymotion_rigged=hymotion_rigged.glb sha256=%s\n' "$(sha256sum "$OUT/hymotion_rigged.glb" | awk '{print $1}')"
  } >"$OUT/texture_delivery.txt"
}

run_native_high() {
  local out="$OUT/native_high_textured.glb"
  local atlas_out="$OUT/native_high_textured_atlas.png"
  local dump="$OUT/native_high_texture_dump"
  echo "== $LABEL: native high (300000 faces, ${NATIVE_HIGH_ATLAS}px atlas, one 3060 inference) =="
  "$CP/shootout/native_texture_run.sh" "$REFINED" "$IMAGE" "$out" \
    --resolution "$NATIVE_HIGH_RESOLUTION" --texsize "$NATIVE_HIGH_ATLAS" --decimate 300000 --unwrap "$NATIVE_UNWRAP" \
    --atlas-out "$atlas_out" --dump-dir "$dump"
  [[ -s "$atlas_out" ]] || { echo "native high did not produce its baseColor atlas: $atlas_out" >&2; return 1; }
  [[ -s "$dump/native_pbr_feats.npy" && -s "$dump/native_pbr_coords.npy" ]] || {
    echo "native high did not retain the authoritative PBR volume" >&2; return 1;
  }
  assert_closed_mesh "$out"
  assert_texture_qc "$out"
  "$CP/shootout/verify_native_texture_asset.sh" "$out" --execution gpu
}

run_rebaked_lod() {
  local name="$1" faces="$2" atlas="$3"
  local out="$OUT/native_${name}_textured.glb"
  local atlas_out="$OUT/native_${name}_textured_atlas.png"
  echo "== $LABEL: native $name (${faces} faces, ${atlas}px atlas, CPU rebake of high material) =="
  "$CP/shootout/native_texture_rebake.sh" "$REFINED" "$OUT/native_high_texture_dump" "$out" \
    --resolution "$NATIVE_HIGH_RESOLUTION" --texsize "$atlas" --decimate "$faces" --unwrap "$NATIVE_UNWRAP" \
    --atlas-out "$atlas_out"
  [[ -s "$atlas_out" ]] || { echo "native $name did not produce its baseColor atlas: $atlas_out" >&2; return 1; }
  assert_closed_mesh "$out"
  assert_texture_qc "$out"
  "$CP/shootout/verify_native_texture_asset.sh" "$out" --execution cpu
}

mixamo_core_ok() {
  local file="$1" n
  local core=(Hips LeftUpLeg RightUpLeg Spine LeftLeg RightLeg Spine1 LeftFoot RightFoot Spine2
              LeftToeBase RightToeBase Neck LeftShoulder RightShoulder Head LeftArm RightArm
              LeftForeArm RightForeArm LeftHand RightHand)
  for n in "${core[@]}"; do grep -a -q "mixamorig:$n" "$file" || return 1; done
}

rig_ok() {
  local file="$1" report fan total
  report="$("$CP/rig_score" "$file" 55 2>&1 || true)"
  printf '%s\n' "$report"
  [[ "$report" =~ maxfan=([0-9]+) ]] || return 1; fan="${BASH_REMATCH[1]}"
  [[ "$report" =~ TOTAL=([0-9.]+) ]] || return 1; total="${BASH_REMATCH[1]}"
  (( fan <= 7 )) && awk "BEGIN { exit !($total >= 0.50) }" \
    && mixamo_core_ok "$file"
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

run_native_high
run_rebaked_lod medium 150000 1024
run_rebaked_lod low     50000 1024

SELECTED_RIG=""
for level in high medium low; do
  try_rig "$level" && break || echo "== $LABEL: rejected native $level rig; trying next LOD ==" >&2
done

# Texture delivery is independent of learned-skeleton acceptance.  Preserve a
# small, authoritative manifest even when the structural gate correctly
# refuses every candidate: the eye-test page can still compare the successful
# high/medium/low materials, and a rerun has an unambiguous reason for not
# publishing a Hymotion asset.
if [[ -z "$SELECTED_RIG" ]]; then
  write_texture_delivery_manifest rejected
  cat >"$OUT/run-status.txt" <<EOF
texture_lods=succeeded
rig_state=rejected
rig_gate=maxfan <= 7; rig_score >= 0.50; full 22-bone Mixamo core; bone-naming falsifier
rig_logs=rig_candidate_native_{high,medium,low}.rig.log
published_hymotion_rig=none
EOF
  cat >"$OUT/stages.json" <<JSON
{"subject":"$LABEL · native textured run (rig rejected)","input":"input.png","stages":[
 {"file":"native_high_textured.glb","label":"HIGH · native textured","note":"300k target faces · ${NATIVE_HIGH_RESOLUTION}px model · ${NATIVE_HIGH_ATLAS} atlas · ${NATIVE_UNWRAP} unwrap · authoritative native material"},
 {"file":"native_medium_textured.glb","label":"MEDIUM · native textured","note":"150k target faces · 1024 atlas · CPU rebake of the high native material"},
 {"file":"native_low_textured.glb","label":"LOW · native textured","note":"50k target faces · 1024 atlas · CPU rebake of the high native material"}
]}
JSON
  echo "FAIL: texture LODs succeeded, but no native rig candidate passed the structural gate; see $OUT/run-status.txt" >&2
  exit 1
fi

cat >"$OUT/run-status.txt" <<EOF
texture_lods=succeeded
rig_state=succeeded
selected_rig=$SELECTED_RIG
rig_gate=maxfan <= 7; rig_score >= 0.50; full 22-bone Mixamo core; bone-naming falsifier
published_hymotion_rig=hymotion_rigged.glb
EOF

write_texture_delivery_manifest succeeded "$SELECTED_RIG"

cat >"$OUT/stages.json" <<JSON
{"subject":"$LABEL · native refined-mesh image-to-rig runbook","input":"input.png","stages":[
 {"file":"hymotion_rigged.glb","label":"Hymotion-ready · native $SELECTED_RIG rig","note":"native Trellis texture; deterministic beam rig; maxfan ≤ 7, rig_score ≥ 0.50, and full 22-bone Mixamo core"},
 {"file":"native_high_textured.glb","label":"HIGH · native textured","note":"300k target faces · ${NATIVE_HIGH_RESOLUTION}px model · ${NATIVE_HIGH_ATLAS} atlas · ${NATIVE_UNWRAP} unwrap · authoritative native material"},
 {"file":"native_medium_textured.glb","label":"MEDIUM · native textured","note":"150k target faces · 1024 atlas · CPU rebake of the high native material"},
 {"file":"native_low_textured.glb","label":"LOW · native textured","note":"50k target faces · 1024 atlas · CPU rebake of the high native material"}
]}
JSON
echo "== DONE: $OUT (native $SELECTED_RIG rig selected) =="
