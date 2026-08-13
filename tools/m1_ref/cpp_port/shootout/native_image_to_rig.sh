#!/usr/bin/env bash
# Reproducible native texture + rig hand-off for ANY cleaned image-to-mesh pair.
#
# Usage:
#   native_image_to_rig.sh <refined.glb> <same-frame-rgba-or-black-matte.png> <out-dir> [label]
#
# Produces one authoritative native Pixal projection-conditioned PBR material, then CPU-rebakes it onto
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
ROOT_MIN_FREE_GIB="${IMAGE_TO_RIG_MIN_ROOT_FREE_GIB:-20}"
[[ "$ROOT_MIN_FREE_GIB" =~ ^[0-9]+$ ]] || { echo "IMAGE_TO_RIG_MIN_ROOT_FREE_GIB must be a non-negative integer" >&2; exit 2; }
root_free_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
(( root_free_kib >= ROOT_MIN_FREE_GIB * 1024 * 1024 )) || {
  echo "refusing image-to-rig run: / has only $(( root_free_kib / 1024 / 1024 )) GiB free; require ${ROOT_MIN_FREE_GIB} GiB. Keep Docker builder cache and all generated artifacts off /." >&2
  exit 75
}

[[ -f "$REFINED" ]] || { echo "missing refined mesh: $REFINED" >&2; exit 2; }
[[ -f "$IMAGE" ]] || { echo "missing source image: $IMAGE" >&2; exit 2; }
for bin in native_texture_run.sh native_texture_rebake.sh verify_native_texture_asset.sh verify_native_texture_delivery.sh texture_rebake_native mesh_sample_main rig_texture_chain.sh rig_score mesh_topo glb_reader_test; do
  [[ -x "$CP/shootout/$bin" || -x "$CP/$bin" ]] || { echo "missing executable: $bin" >&2; exit 2; }
done

SKIN_GGUF="${SKIN_VAE_GGUF:-/home/dbrain/models/3d/rig/skin_vae_gguf}"
QWEN3_W="${QWEN3_W:-/home/dbrain/models/3d/rig/qwen3_w}"
R1W_SRC="${R1W_SRC:-/home/dbrain/models/3d/rig/r1w_real}"
RIG_POSE_GATE_PYTHON="${RIG_POSE_GATE_PYTHON:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"
RIG_PROFILE="${RIG_PROFILE:-humanoid}"
RIG_RECOVER_APPENDAGE_TIPS="${RIG_RECOVER_APPENDAGE_TIPS:-0}"
# Production never delegates skeleton or weight generation to the upstream
# Python decoder.  Keep the old helper as an explicitly labelled historical
# diagnostic, but make attempts to enable it fail closed here.
RIG_OFFICIAL_SAMPLED_FALLBACK="${RIG_OFFICIAL_SAMPLED_FALLBACK:-0}"
[[ "$RIG_RECOVER_APPENDAGE_TIPS" == 0 || "$RIG_RECOVER_APPENDAGE_TIPS" == 1 ]] || {
  echo "RIG_RECOVER_APPENDAGE_TIPS must be 0 or 1" >&2; exit 2;
}
[[ "$RIG_OFFICIAL_SAMPLED_FALLBACK" == 0 ]] || {
  echo "RIG_OFFICIAL_SAMPLED_FALLBACK is disabled for production: native SkinTokens must generate the published skeleton and weights" >&2; exit 2;
}
case "$RIG_PROFILE" in
  humanoid)
    PUBLISHED_RIG="hymotion_rigged.glb"
    if [[ "$RIG_RECOVER_APPENDAGE_TIPS" == 1 ]]; then
      RIG_GATE="validated mirrored appendage tips; profile-aware maxfan allowance; rig_score >= 0.50; full 22-bone Mixamo core; bone-naming falsifier"
    else
    RIG_GATE="maxfan <= 7; rig_score >= 0.50; full 22-bone Mixamo core; bone-naming falsifier; real-GLB arm pose stretch p999 <= 5.0"
    fi
    RIG_LABEL="Hymotion-ready"
    ;;
  generic)
    PUBLISHED_RIG="generic_rigged.glb"
    RIG_GATE="maxfan <= max(8, ceil(joints/5)); rig_score >= 0.50; one valid generic root; stable skintokens joint namespace; real-GLB 45-degree stress of every materially weighted joint, worst p999 stretch <= 6.0"
    RIG_LABEL="generic creature rig"
    ;;
  *) echo "RIG_PROFILE must be humanoid or generic (got $RIG_PROFILE)" >&2; exit 2 ;;
esac
[[ "$RIG_RECOVER_APPENDAGE_TIPS" == 0 || "$RIG_PROFILE" == humanoid ]] || {
  echo "RIG_RECOVER_APPENDAGE_TIPS requires RIG_PROFILE=humanoid" >&2; exit 2;
}
for d in "$SKIN_GGUF" "$QWEN3_W" "$R1W_SRC"; do
  [[ -d "$d" ]] || { echo "missing rig weights: $d" >&2; exit 2; }
done
[[ -x "$RIG_POSE_GATE_PYTHON" ]] || { echo "missing pose-gate Python: $RIG_POSE_GATE_PYTHON" >&2; exit 2; }

# A full image→rig generation retains its projection-conditioned Pixal PBR
# volume while the geometry/material model state is live, then rebakes it here.
# Starting a fresh M6 pass from only an arbitrary refined mesh is a separate
# diagnostic route and currently fails the frozen/visual texture gates.  Do
# not let a missing cache silently turn a normal delivery into that experiment.
# Validate this before creating output aliases, so a rejected request cannot
# look like a partial delivery.
NATIVE_GENERATED_PBR_CACHE="${NATIVE_GENERATED_PBR_CACHE:-}"
# The sparse PBR volume is a set of integer voxel indices on a grid-N lattice, where N is the Pixal
# HR cascade resolution the geometry stage ran at (IMAGE_TO_RIG_GEO_RESOLUTION: 1024 or 1536).  The
# rebake maps mesh positions onto those indices with (p+0.5)*N, so N is a property of the cache and
# must be read from it -- assuming 1024 for a grid-1536 cache shrinks every sample by 2/3 and bakes
# an almost entirely unpainted atlas.  image_to_rig stages it as a size-prefixed int32 in
# `resolution.bin`; schema-2 manifests also record it so the value is covered by cache validation.
PBR_CACHE_LATTICE=""
read_pbr_lattice_bin() {
  local f="$1/resolution.bin" v
  [[ -s "$f" ]] || return 1
  v="$(od -An -tu4 -j8 -N4 "$f" 2>/dev/null | tr -d ' \n')"
  [[ "$v" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s' "$v"
}
if [[ -n "$NATIVE_GENERATED_PBR_CACHE" ]]; then
  if [[ -s "$NATIVE_GENERATED_PBR_CACHE/pbr_feats.bin" && -s "$NATIVE_GENERATED_PBR_CACHE/pbr_coords.bin" ]]; then
    PBR_CACHE_FEATS="$NATIVE_GENERATED_PBR_CACHE/pbr_feats.bin"
    PBR_CACHE_COORDS="$NATIVE_GENERATED_PBR_CACHE/pbr_coords.bin"
  elif [[ -s "$NATIVE_GENERATED_PBR_CACHE/native_pbr_feats.npy" && -s "$NATIVE_GENERATED_PBR_CACHE/native_pbr_coords.npy" ]]; then
    PBR_CACHE_FEATS="$NATIVE_GENERATED_PBR_CACHE/native_pbr_feats.npy"
    PBR_CACHE_COORDS="$NATIVE_GENERATED_PBR_CACHE/native_pbr_coords.npy"
  else
    echo "missing generated Pixal PBR cache in $NATIVE_GENERATED_PBR_CACHE" >&2; exit 2;
  fi
  PBR_CACHE_MANIFEST="$NATIVE_GENERATED_PBR_CACHE/pbr_cache_manifest.txt"
  if [[ ! -s "$PBR_CACHE_MANIFEST" ]]; then
    if [[ "${NATIVE_PBR_CACHE_ALLOW_LEGACY_UNVERIFIED:-0}" != 1 ]]; then
      echo "refusing unproven generated Pixal PBR cache: missing $PBR_CACHE_MANIFEST (set NATIVE_PBR_CACHE_ALLOW_LEGACY_UNVERIFIED=1 only for a labelled diagnostic replay)" >&2
      exit 2
    fi
    echo "WARNING: using legacy unproven Pixal PBR cache by explicit diagnostic override: $NATIVE_GENERATED_PBR_CACHE" >&2
  else
    cache_manifest_value() { awk -F= -v key="$1" '$1==key {print substr($0,length(key)+2); exit}' "$PBR_CACHE_MANIFEST"; }
    PBR_CACHE_SCHEMA="$(cache_manifest_value schema_version)"
    # schema 2 adds pbr_lattice_resolution.  Schema 1 remains readable: its lattice comes from
    # resolution.bin, which image_to_rig has always written next to the volume.
    [[ "$PBR_CACHE_SCHEMA" == 1 || "$PBR_CACHE_SCHEMA" == 2 ]] || { echo "invalid PBR cache manifest schema: $PBR_CACHE_MANIFEST" >&2; exit 2; }
    if [[ "$PBR_CACHE_SCHEMA" == 2 ]]; then
      PBR_CACHE_LATTICE="$(cache_manifest_value pbr_lattice_resolution)"
      [[ "$PBR_CACHE_LATTICE" =~ ^[1-9][0-9]*$ ]] || {
        echo "schema-2 PBR cache manifest has no usable pbr_lattice_resolution: $PBR_CACHE_MANIFEST" >&2; exit 2;
      }
    fi
    [[ "$(cache_manifest_value model_image_sha256)" == "$(sha256sum "$IMAGE" | awk '{print $1}')" ]] || {
      echo "PBR cache model-image mismatch: refusing cache from another image frame" >&2; exit 2;
    }
    [[ "$(cache_manifest_value refined_mesh_sha256)" == "$(sha256sum "$REFINED" | awk '{print $1}')" ]] || {
      echo "PBR cache refined-mesh mismatch: refusing cache from another geometry run" >&2; exit 2;
    }
    [[ "$(cache_manifest_value pbr_feats_sha256)" == "$(sha256sum "$PBR_CACHE_FEATS" | awk '{print $1}')" ]] || {
      echo "PBR cache feature hash mismatch: refusing modified cache" >&2; exit 2;
    }
    [[ "$(cache_manifest_value pbr_coords_sha256)" == "$(sha256sum "$PBR_CACHE_COORDS" | awk '{print $1}')" ]] || {
      echo "PBR cache coordinate hash mismatch: refusing modified cache" >&2; exit 2;
    }
  fi
  PBR_CACHE_LATTICE_BIN="$(read_pbr_lattice_bin "$NATIVE_GENERATED_PBR_CACHE" || true)"
  if [[ -n "$PBR_CACHE_LATTICE" && -n "$PBR_CACHE_LATTICE_BIN" && "$PBR_CACHE_LATTICE" != "$PBR_CACHE_LATTICE_BIN" ]]; then
    echo "PBR cache lattice disagreement: manifest says $PBR_CACHE_LATTICE, resolution.bin says $PBR_CACHE_LATTICE_BIN" >&2; exit 2;
  fi
  [[ -n "$PBR_CACHE_LATTICE" ]] || PBR_CACHE_LATTICE="$PBR_CACHE_LATTICE_BIN"
  [[ -n "$PBR_CACHE_LATTICE" ]] || {
    echo "generated Pixal PBR cache $NATIVE_GENERATED_PBR_CACHE records no lattice resolution (no schema-2 manifest field and no resolution.bin); refusing to guess the voxel coordinate space" >&2
    exit 2
  }
fi
if [[ -z "$NATIVE_GENERATED_PBR_CACHE" && "${NATIVE_TEXTURE_ALLOW_UNVALIDATED_DIRECT_M6:-0}" != 1 ]]; then
  echo "refusing unvalidated standalone M6 texture inference: provide NATIVE_GENERATED_PBR_CACHE from native_image_to_rig_from_image.sh (or set NATIVE_TEXTURE_ALLOW_UNVALIDATED_DIRECT_M6=1 for a diagnostic-only run)" >&2
  exit 2
fi

mkdir -p "$OUT"
[[ "$REFINED" == "$OUT/refined_geometry.glb" ]] || ln -sfn "$REFINED" "$OUT/refined_geometry.glb"
ln -sfn "$IMAGE" "$OUT/input.png"
# The refined-mesh stage canonicalizes characters with +Z forward.  Pin this
# convention for the structural Mixamo-name mapper so left/right does not
# depend on a weak toe cue from a stylised or partially reconstructed foot.
export RIG_BONE_FACING="${RIG_BONE_FACING:-+z}"
# The standalone native texture *inference* checkpoint is Pixal's 1024 projection model.
# There is no installed 512 checkpoint, so accepting 512 here would be a false-quality fallback.
# This governs the direct-M6 route only (native_texture_run.sh); a rebake of a generated cache uses
# that cache's own lattice instead -- see NATIVE_REBAKE_RESOLUTION below.
NATIVE_HIGH_RESOLUTION="${NATIVE_HIGH_RESOLUTION:-1024}"
# These are delivery density tiers, not four separate generative materials.
# The retained 1024-model PBR volume is decoded once, then rebaked at the
# appropriate mesh and atlas density. `hero` intentionally retains the full
# refined surface (zero means no decimation); it is the asset for a close-up
# render, not the mesh selected for animation. The fixed-volume control
# measured 39.16 dB at 4K and 42.88 dB at 8K, both with zero unresolved
# surface texels.
NATIVE_HERO_FACES="${NATIVE_HERO_FACES:-0}"
NATIVE_HERO_ATLAS="${NATIVE_HERO_ATLAS:-8192}"
# Full-resolution meshes are deliberately charted with the bounded production
# route by default. Callers can request reference charts explicitly, but the
# direct global xatlas route is not an acceptable surprise for a hero export.
NATIVE_HERO_UNWRAP="${NATIVE_HERO_UNWRAP:-production}"
NATIVE_HIGH_ATLAS="${NATIVE_HIGH_ATLAS:-8192}"
NATIVE_MEDIUM_ATLAS="${NATIVE_MEDIUM_ATLAS:-2048}"
NATIVE_LOW_ATLAS="${NATIVE_LOW_ATLAS:-1024}"
# The bounded production chart route is the delivery default for every tier.
# `reference` remains an explicit parity/quality A/B, never an accidental
# 30-minute global-chart solve on an otherwise ordinary character.
NATIVE_UNWRAP="${NATIVE_UNWRAP:-production}"
# Cache-rebake sampling mode for every delivered LOD (texture_rebake_native --bake).
#   volume-trilinear (default) - Python to_glb parity: volume-direct sampling at the refined-mesh
#                                texel with crisp boundaries, coarse-shell guard, RAW baseColor (no
#                                sRGB OETF), TELEA inpaint + background fill. This is the delivery
#                                material and the fix set validated in texture_rebake_native.cpp.
#   projection                 - the older projection bake; retained as an explicit A/B knob only.
NATIVE_BAKE_MODE="${NATIVE_BAKE_MODE:-volume-trilinear}"
[[ "$NATIVE_BAKE_MODE" == volume-trilinear || "$NATIVE_BAKE_MODE" == projection ]] || {
  echo "NATIVE_BAKE_MODE must be volume-trilinear or projection" >&2; exit 2;
}
[[ "$NATIVE_HIGH_RESOLUTION" == 1024 ]] || {
  echo "NATIVE_HIGH_RESOLUTION must be 1024: the only authoritative native Pixal texture checkpoint is 1024" >&2; exit 2;
}
# Every rebake (high + LOD tiers) samples the retained volume, so it must run on the volume's own
# lattice.  With a generated cache that is the cache's recorded resolution, which tracks
# IMAGE_TO_RIG_GEO_RESOLUTION; without one the only rebake source is this run's own 1024 M6 dump.
NATIVE_REBAKE_RESOLUTION="${PBR_CACHE_LATTICE:-$NATIVE_HIGH_RESOLUTION}"
[[ "$NATIVE_REBAKE_RESOLUTION" =~ ^[1-9][0-9]*$ ]] && (( NATIVE_REBAKE_RESOLUTION % 16 == 0 )) || {
  echo "PBR rebake lattice must be a positive multiple of 16 (got '$NATIVE_REBAKE_RESOLUTION')" >&2; exit 2;
}
[[ "$NATIVE_UNWRAP" == reference || "$NATIVE_UNWRAP" == production ]] || {
  echo "NATIVE_UNWRAP must be reference or production" >&2; exit 2;
}
[[ "$NATIVE_HERO_UNWRAP" == reference || "$NATIVE_HERO_UNWRAP" == production ]] || {
  echo "NATIVE_HERO_UNWRAP must be reference or production" >&2; exit 2;
}
[[ "$NATIVE_HERO_FACES" =~ ^[0-9]+$ ]] || {
  echo "NATIVE_HERO_FACES must be a non-negative face budget (0 retains the complete refined mesh)" >&2; exit 2;
}
for atlas_env in NATIVE_HERO_ATLAS NATIVE_HIGH_ATLAS NATIVE_MEDIUM_ATLAS NATIVE_LOW_ATLAS; do
  atlas_value="${!atlas_env}"
  [[ "$atlas_value" =~ ^(1024|2048|4096|8192)$ ]] || {
    echo "$atlas_env must be one of 1024, 2048, 4096, or 8192" >&2; exit 2;
  }
done
(( NATIVE_HERO_ATLAS >= NATIVE_HIGH_ATLAS && NATIVE_HIGH_ATLAS >= NATIVE_MEDIUM_ATLAS && NATIVE_MEDIUM_ATLAS >= NATIVE_LOW_ATLAS )) || {
  echo "atlas tiers must be non-increasing: hero >= high >= medium >= low" >&2; exit 2;
}
NATIVE_TEXTURE_CAMERA_ARGS=()
if [[ -n "${NATIVE_TEXTURE_CAMERA_FILE:-}" ]]; then
  [[ -s "$NATIVE_TEXTURE_CAMERA_FILE" ]] || { echo "missing texture camera provenance: $NATIVE_TEXTURE_CAMERA_FILE" >&2; exit 2; }
  if [[ -n "$NATIVE_GENERATED_PBR_CACHE" && -s "${PBR_CACHE_MANIFEST:-}" ]]; then
    manifest_camera_sha="$(awk -F= '$1=="camera_provenance_sha256" {print substr($0,length($1)+2); exit}' "$PBR_CACHE_MANIFEST")"
    [[ -n "$manifest_camera_sha" && "$manifest_camera_sha" == "$(sha256sum "$NATIVE_TEXTURE_CAMERA_FILE" | awk '{print $1}')" ]] || {
      echo "PBR cache camera-provenance mismatch: refusing cache from another camera contract" >&2; exit 2;
    }
  fi
  NATIVE_TEXTURE_CAMERA_ARGS=(--camera "$NATIVE_TEXTURE_CAMERA_FILE")
fi
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
# cosmetic whole-atlas background dilation. The final-atlas recovery evidence
# is then enforced by verify_native_texture_asset.sh, so a no-hole-but-washed-
# out atlas cannot be promoted.
assert_texture_qc() {
  local file="$1" qc unresolved
  qc="${file}.texture-qc.txt"
  [[ -s "$qc" ]] || { echo "missing native texture QC: $qc" >&2; return 1; }
  unresolved="$(awk -F= '$1=="unresolved_surface_texels_after_chart_repair" {print $2; exit}' "$qc")"
  [[ "$unresolved" =~ ^[0-9]+$ ]] || { echo "texture QC lacks unresolved-surface count: $qc" >&2; return 1; }
  (( unresolved == 0 )) || { echo "REJECT: texture bake leaves $unresolved unresolved surface texels: $qc" >&2; return 1; }
  printf 'texture QC: %s\n' "$(tr '\n' ' ' <"$qc")"
}

# One durable hand-off record, written only after every texture asset passed the structural gate.
# Hero is the un-decimated visual source; high remains the animation source even if medium/low
# supplied the best skeleton, so Hymotion never silently inherits a lower-resolution atlas.
write_texture_delivery_manifest() {
  local rig_state="$1" level="${2:-none}" name file atlas status stage route stage_sha qc qc_sha qc_verdict qc_missing qc_schema qc_recovery_texels qc_mae qc_rmse qc_psnr topo manifest_tmp stage_metrics elapsed cpu_peak gpu_peak gpu_last execution
  manifest_tmp="$OUT/.texture_delivery.txt.$$"
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
  stage_metrics_for() {
    local stage_file="$1" previous_t="" previous_stage="" line t stage metric
    [[ -f "$stage_file" ]] || { printf 'missing-stage-log'; return; }
    while IFS= read -r line; do
      [[ "$line" =~ ^t=([0-9]+)[[:space:]]stage=([^[:space:]]+)$ ]] || continue
      t="${BASH_REMATCH[1]}"; stage="${BASH_REMATCH[2]}"
      if [[ -n "$previous_t" ]]; then
        case "$stage" in
          atlas_unwrap_complete) metric=atlas_unwrap ;;
          *_complete) metric="${stage%_complete}" ;;
          *) metric="transition_${previous_stage}_to_${stage}" ;;
        esac
        printf '%s:%ss,' "$metric" "$((t - previous_t))"
      fi
      previous_t="$t"; previous_stage="$stage"
    done <"$stage_file"
    [[ -n "$previous_t" ]] || printf 'no-parseable-stages'
  }
  status_value() {
    local status_file="$1" key="$2"
    [[ -f "$status_file" ]] || return 0
    awk -F= -v key="$key" '$1==key {print substr($0,length(key)+2); exit}' "$status_file"
  }
  {
    printf 'schema_version=4\n'
    printf 'production_texture=native_hero_textured.glb\n'
    printf 'rig_texture=native_high_textured.glb\n'
    printf 'native_unwrap=%s\nhero_unwrap=%s\n' "$NATIVE_UNWRAP" "$NATIVE_HERO_UNWRAP"
    printf 'high_model_lattice=%s\nhero_face_budget=%s\nhero_atlas_px=%s\nhigh_atlas_px=%s\nmedium_atlas_px=%s\nlow_atlas_px=%s\n' \
      "$NATIVE_HIGH_RESOLUTION" "$NATIVE_HERO_FACES" "$NATIVE_HERO_ATLAS" "$NATIVE_HIGH_ATLAS" "$NATIVE_MEDIUM_ATLAS" "$NATIVE_LOW_ATLAS"
    # The lattice every tier actually sampled the retained PBR volume on.  It tracks the geometry
    # stage's IMAGE_TO_RIG_GEO_RESOLUTION, which is not necessarily the M6 texture-model lattice.
    printf 'pbr_lattice_resolution=%s\n' "$NATIVE_REBAKE_RESOLUTION"
    if [[ -n "$NATIVE_GENERATED_PBR_CACHE" ]]; then
      printf 'material_source=native Pixal projection-conditioned PBR cache generated during the 3060 geometry stage: %s\n' "$NATIVE_GENERATED_PBR_CACHE"
      printf 'lod_material_contract=hero/high/medium/low are CPU rebakes of that one native Pixal PBR cache; hero retains the refined mesh unless explicitly budgeted; no inverse mesh encoder and no second texture inference\n'
    else
      printf 'material_source=texture_mesh_native PBR dump: native_high_texture_dump\n'
      printf 'lod_material_contract=hero/medium/low are CPU rebakes of native_high_texture_dump; hero retains the refined mesh unless explicitly budgeted; no second texture inference\n'
    fi
    printf 'topology_gate=position-welded open=0 for every texture LOD\n'
    printf 'texture_qc_gate=zero unresolved rasterised surface texels after chart-local repair; schema-3 final-atlas recovery MAE<=%s and PSNR>=%s dB; source-missing count is retained for eye-test audit\n' \
      "${NATIVE_TEXTURE_MAX_RECOVERY_MAE_LINEAR:-0.020}" "${NATIVE_TEXTURE_MIN_RECOVERY_PSNR_DB:-30}"
    printf 'rig_state=%s\nselected_rig=%s\n' "$rig_state" "$level"
    for name in hero high medium low; do
      file="$OUT/native_${name}_textured.glb"
      atlas="$OUT/native_${name}_textured_atlas.png"
      status="${file}.run-status.txt"
      stage="${file}.stage-log.txt"
      route="$(atlas_route_for "$stage")"
      stage_sha="missing"
      [[ ! -f "$stage" ]] || stage_sha="$(sha256sum "$stage" | awk '{print $1}')"
      qc="${file}.texture-qc.txt"
      qc_sha="missing"; qc_verdict="missing"; qc_missing="missing"
      qc_schema="missing"; qc_recovery_texels="missing"; qc_mae="missing"; qc_rmse="missing"; qc_psnr="missing"
      if [[ -f "$qc" ]]; then
        qc_sha="$(sha256sum "$qc" | awk '{print $1}')"
        qc_verdict="$(awk -F= '$1=="sampling_verdict" {print $2; exit}' "$qc")"
        qc_missing="$(awk -F= '$1=="source_missing_fraction_before_repair" {print $2; exit}' "$qc")"
        qc_schema="$(awk -F= '$1=="schema_version" {print $2; exit}' "$qc")"
        qc_recovery_texels="$(awk -F= '$1=="source_recovery_texels" {print $2; exit}' "$qc")"
        qc_mae="$(awk -F= '$1=="source_recovery_mae_linear" {print $2; exit}' "$qc")"
        qc_rmse="$(awk -F= '$1=="source_recovery_rmse_linear" {print $2; exit}' "$qc")"
        qc_psnr="$(awk -F= '$1=="source_recovery_psnr_db" {print $2; exit}' "$qc")"
      fi
      topo="$("$CP/mesh_topo" "$file")"
      stage_metrics="$(stage_metrics_for "$stage")"
      elapsed="$(status_value "$status" elapsed_seconds)"; cpu_peak="$(status_value "$status" cpu_util_percent_peak)"
      gpu_peak="$(status_value "$status" gpu_vram_mib_peak)"; gpu_last="$(status_value "$status" gpu_vram_mib_last)"
      execution="$(status_value "$status" execution)"
      printf 'lod=%s file=%s sha256=%s atlas=%s atlas_sha256=%s atlas_route=%s stage_log=%s stage_log_sha256=%s stage_wall_seconds="%s" elapsed_seconds=%s cpu_util_percent_peak=%s gpu_vram_mib_peak=%s gpu_vram_mib_last=%s execution="%s" texture_qc=%s texture_qc_sha256=%s qc_schema=%s sampling_verdict=%s source_missing_fraction_before_repair=%s source_recovery_texels=%s source_recovery_mae_linear=%s source_recovery_rmse_linear=%s source_recovery_psnr_db=%s topology="%s"\n' \
        "$name" "$(basename "$file")" "$(sha256sum "$file" | awk '{print $1}')" \
        "$(basename "$atlas")" "$(sha256sum "$atlas" | awk '{print $1}')" "$route" \
        "$(basename "$stage")" "$stage_sha" "$stage_metrics" "${elapsed:-unknown}" "${cpu_peak:-unknown}" "${gpu_peak:-unknown}" "${gpu_last:-unknown}" "${execution:-unknown}" "$(basename "$qc")" "$qc_sha" "$qc_schema" "$qc_verdict" "$qc_missing" \
        "$qc_recovery_texels" "$qc_mae" "$qc_rmse" "$qc_psnr" "$topo"
      [[ ! -f "$status" ]] || printf 'lod=%s native_texture_status=%s\n' "$name" "$(basename "$status")"
    done
    [[ "$rig_state" != succeeded ]] || printf 'published_rig=%s sha256=%s\n' "$PUBLISHED_RIG" "$(sha256sum "$OUT/$PUBLISHED_RIG" | awk '{print $1}')"
  } >"$manifest_tmp"
  mv -f "$manifest_tmp" "$OUT/texture_delivery.txt"
}

run_native_high() {
  local out="$OUT/native_high_textured.glb"
  local atlas_out="$OUT/native_high_textured_atlas.png"
  local dump="$OUT/native_high_texture_dump"
  if [[ -n "$NATIVE_GENERATED_PBR_CACHE" ]]; then
    echo "== $LABEL: native high (300000 faces, ${NATIVE_HIGH_ATLAS}px atlas, CPU rebake of 3060-generated grid-${NATIVE_REBAKE_RESOLUTION} Pixal PBR cache, ${NATIVE_BAKE_MODE} bake) =="
    "$CP/shootout/native_texture_rebake.sh" "$REFINED" "$NATIVE_GENERATED_PBR_CACHE" "$out" \
      --resolution "$NATIVE_REBAKE_RESOLUTION" --texsize "$NATIVE_HIGH_ATLAS" --decimate 300000 --unwrap "$NATIVE_UNWRAP" \
      --bake "$NATIVE_BAKE_MODE" --atlas-out "$atlas_out"
  else
    echo "== $LABEL: native high (300000 faces, ${NATIVE_HIGH_ATLAS}px atlas, one 3060 inference) =="
    "$CP/shootout/native_texture_run.sh" "$REFINED" "$IMAGE" "$out" \
      --resolution "$NATIVE_HIGH_RESOLUTION" --texsize "$NATIVE_HIGH_ATLAS" --decimate 300000 --unwrap "$NATIVE_UNWRAP" \
      --atlas-out "$atlas_out" --dump-dir "$dump" "${NATIVE_TEXTURE_CAMERA_ARGS[@]}"
  fi
  [[ -s "$atlas_out" ]] || { echo "native high did not produce its baseColor atlas: $atlas_out" >&2; return 1; }
  if [[ -z "$NATIVE_GENERATED_PBR_CACHE" ]]; then
    [[ -s "$dump/native_pbr_feats.npy" && -s "$dump/native_pbr_coords.npy" ]] || {
      echo "native high did not retain the authoritative PBR volume" >&2; return 1;
    }
  fi
  assert_closed_mesh "$out"
  assert_texture_qc "$out"
  "$CP/shootout/verify_native_texture_asset.sh" "$out" --execution "$([[ -n "$NATIVE_GENERATED_PBR_CACHE" ]] && printf cpu || printf gpu)"
}

run_rebaked_lod() {
  local name="$1" faces="$2" atlas="$3" unwrap="$4" tier_note="$5"
  local out="$OUT/native_${name}_textured.glb"
  local atlas_out="$OUT/native_${name}_textured_atlas.png"
  echo "== $LABEL: native $name (${tier_note}, ${atlas}px atlas, CPU rebake of retained grid-${NATIVE_REBAKE_RESOLUTION} native material, ${NATIVE_BAKE_MODE} bake) =="
  "$CP/shootout/native_texture_rebake.sh" "$REFINED" "${NATIVE_GENERATED_PBR_CACHE:-$OUT/native_high_texture_dump}" "$out" \
    --resolution "$NATIVE_REBAKE_RESOLUTION" --texsize "$atlas" --decimate "$faces" --unwrap "$unwrap" \
    --bake "$NATIVE_BAKE_MODE" --atlas-out "$atlas_out"
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
  # Match the entire JSON node name.  A substring test makes `Spine` falsely
  # pass when a malformed rig contains only `Spine1`/`Spine2`.
  for n in "${core[@]}"; do
    grep -a -F -q "\"name\":\"mixamorig:$n\"" "$file" || return 1
  done
}

generic_core_ok() {
  local file="$1"
  grep -a -F -q '"name":"skintokens:Root_' "$file" \
    && grep -a -F -q '"name":"skintokens:Joint_' "$file"
}

rig_ok() {
  local file="$1" report joints fan fan_limit total
  report="$("$CP/rig_score" "$file" 55 2>&1 || true)"
  printf '%s\n' "$report"
  [[ "$report" =~ J=([0-9]+) ]] || return 1; joints="${BASH_REMATCH[1]}"
  [[ "$report" =~ maxfan=([0-9]+) ]] || return 1; fan="${BASH_REMATCH[1]}"
  [[ "$report" =~ TOTAL=([0-9.]+) ]] || return 1; total="${BASH_REMATCH[1]}"
  if [[ "$RIG_PROFILE" == humanoid ]]; then
    if [[ "$RIG_RECOVER_APPENDAGE_TIPS" == 1 ]]; then
      [[ "$report" =~ maxfan=[0-9]+/([0-9]+) ]] || return 1; fan_limit="${BASH_REMATCH[1]}"
      [[ "$report" =~ recovered_appendage_tips=([0-9]+) ]] || return 1
      (( BASH_REMATCH[1] >= 2 )) || return 1
    else
      fan_limit=7
    fi
  else
    fan_limit=$(( (joints + 4) / 5 )); (( fan_limit < 8 )) && fan_limit=8
    # `rig_score` verifies that every skin joint uses the complete stable
    # Root_/Joint_ namespace, rather than merely finding one of each string.
    # Keep the delivery gate tied to that stronger scorer contract.
    [[ "$report" == *"generic_namespace=yes"* ]] || return 1
  fi
  (( fan <= fan_limit )) && awk "BEGIN { exit !($total >= 0.50) }" || return 1
  if [[ "$RIG_PROFILE" == humanoid ]]; then mixamo_core_ok "$file"; else generic_core_ok "$file"; fi
}

# A failed LOD is expected while selecting the first publishable native rig,
# but its reason must remain visible to an interactive caller.  The complete
# transcript is retained beside the candidate; this extracts the decisive
# stage/gate lines without making a user open every log after the run.
emit_rig_failure_summary() {
  # Two statements deliberately: bash expands every word of `local` before the
  # assignments take effect, so a single `local name=$1 log=...${name}...` reads
  # `name` while it is still unset and dies under `set -u`. That aborted the
  # `|| { ... }` block of the LOD loop before it could continue, so a rejected
  # high-tier rig silently ended the run instead of falling back to medium/low.
  local name="$1"
  local log="$OUT/rig_candidate_native_${name}.rig.log"
  echo "== $LABEL: native $name rig failure summary ==" >&2
  if [[ -s "$log" ]]; then
    echo "   log: $log" >&2
    rg -i '\[STAGE R[1-5]\].*(FAIL|rejected|missing|falsifier)|\bFAIL:|failed pose gate|validation failed|quality gate|maxfan=' "$log" \
      | tail -n 24 >&2 || true
  else
    echo "   no rig transcript was written (failure occurred before SkinTokens started)" >&2
  fi
}

try_rig() {
  local name="$1"
  local mesh="$OUT/native_${name}_textured.glb"
  local samples="$OUT/rig_inputs_${name}"
  local candidate_base="$OUT/rig_candidate_native_${name}"
  local candidate="$candidate_base.learned.glb"
  local rig_data="/tmp/skintokens_e2e"
  echo "== $LABEL: deterministic rig candidate from native $name =="
  mkdir -p "$samples"
  "$CP/mesh_sample_main" "$mesh" "$samples"
  if ! RIG_PROFILE="$RIG_PROFILE" RIG_RECOVER_APPENDAGE_TIPS="$RIG_RECOVER_APPENDAGE_TIPS" PIXAL3D_GGUF_DIR="$SKIN_GGUF" R1W_SRC="$R1W_SRC" \
    "$CP/rig_texture_chain.sh" "$samples" "$mesh" "$QWEN3_W" "$candidate" 20; then
    # R4's learned weights can be wrong even when the inferred generic tree
    # itself is clean (for example, long hair is often assigned to an arm or
    # torso). A labelled, deterministic joint-proximity fallback tests that
    # hypothesis against the same real-GLB pose gate. It is generic-only: the
    # humanoid route must retain its learned semantic skin field rather than
    # silently trading it for anonymous proximity weights.
    if [[ "$RIG_PROFILE" != generic ]]; then
      return 1
    fi
    echo "== $LABEL: learned generic skin rejected; retrying labelled geometric-skin fallback ==" >&2
    candidate="$candidate_base.geometric.glb"
    if ! RIG_PROFILE=generic RIG_SKIN_MODE=geometric PIXAL3D_GGUF_DIR="$SKIN_GGUF" R1W_SRC="$R1W_SRC" \
      "$CP/rig_texture_chain.sh" "$samples" "$mesh" "$QWEN3_W" "$candidate" 20; then
      # A structural tree failure cannot be corrected by substituting an
      # upstream sampled decoder.  Retain the native failure evidence and do
      # not publish a generic rig until the deterministic route passes.
      return 1
    fi
  fi
  # A lower LOD may yield the cleanest skeleton.  Its sampled skin weights are
  # still transferred onto the high native texture asset, so animation never
  # inherits a low-resolution atlas merely because it won the rig gate.
  if [[ "$name" != high ]]; then
    local transferred="$candidate_base.high-transfer.glb"
    COMBINE_ARGS=(--profile "$RIG_PROFILE")
    [[ "$RIG_RECOVER_APPENDAGE_TIPS" == 1 ]] && COMBINE_ARGS+=(--recover-appendage-tips)
    "$CP/combine_rig_tex_main" "$OUT/native_high_textured.glb" "$rig_data" "$transferred" \
      --sampled "$rig_data/vertices.npy" --skin "$rig_data/gen_skin_pred.npy" \
      --joints "$rig_data/gen_joints.npy" --parents "$rig_data/gen_parents.npy" \
      "${COMBINE_ARGS[@]}"
    # A skin transferred from a lower LOD onto the final high-texture mesh is
    # a new delivery artifact, so it must clear the same actual-LBS gate.  Do
    # not reuse a lower-LOD success as evidence for a different vertex set.
    local pose_args=(--show-skeleton --pose-gate)
    [[ "$RIG_PROFILE" == generic ]] && pose_args=(--generic-all-influential "${pose_args[@]}")
    if ! "$RIG_POSE_GATE_PYTHON" "$CP/rig_pose_smoke.py" "$transferred" "${transferred%.glb}.pose-gate.png" "${pose_args[@]}"; then
      if [[ "$RIG_PROFILE" != generic ]]; then
        return 1
      fi
      # Python is validation-only in the production path.  It must not edit
      # skin weights after the native decoder has run.
      return 1
    else
      candidate="$transferred"
    fi
  fi
  # Keep every attempted skeleton on a stable, level-labelled path for the
  # clay-stage eye test.  A failed structural gate is still useful geometry
  # evidence; this alias never promotes it to a Hymotion hand-off.
  ln -sfn "$candidate" "$OUT/rig_candidate_${name}.glb"
  "$CP/mesh_topo" "$candidate"
  if rig_ok "$candidate"; then
    cp -f "$candidate" "$OUT/$PUBLISHED_RIG"
    SELECTED_RIG="$name"
    return 0
  fi
  echo "FAIL: native $name candidate failed the $RIG_PROFILE publication gate (see rig_score above)" >&2
  return 1
}

run_native_high
run_rebaked_lod hero   "$NATIVE_HERO_FACES" "$NATIVE_HERO_ATLAS"   "$NATIVE_HERO_UNWRAP" "full refined mesh (face budget ${NATIVE_HERO_FACES})"
run_rebaked_lod medium 150000              "$NATIVE_MEDIUM_ATLAS" "$NATIVE_UNWRAP" "150000 faces"
run_rebaked_lod low     50000               "$NATIVE_LOW_ATLAS"    "$NATIVE_UNWRAP" "50000 faces"

SELECTED_RIG=""
for level in high medium low; do
  try_rig "$level" && break || {
    emit_rig_failure_summary "$level"
    echo "== $LABEL: rejected native $level rig; trying next LOD ==" >&2
  }
done

# Texture delivery is independent of learned-skeleton acceptance.  Preserve a
# small, authoritative manifest even when the structural gate correctly
# refuses every candidate: the eye-test page can still compare the successful
# high/medium/low materials, and a rerun has an unambiguous reason for not
# publishing a Hymotion asset.
if [[ -z "$SELECTED_RIG" ]]; then
  write_texture_delivery_manifest rejected
  "$CP/shootout/verify_native_texture_delivery.sh" "$OUT"
  cat >"$OUT/run-status.txt" <<EOF
texture_lods=succeeded
rig_state=rejected
rig_profile=$RIG_PROFILE
rig_gate=$RIG_GATE
rig_logs=rig_candidate_native_{high,medium,low}.rig.log
published_rig=none
EOF
  cat >"$OUT/stages.json" <<JSON
{"subject":"$LABEL · native Pixal textured run (rig rejected)","input":"input.png","stages":[
 {"file":"native_hero_textured.glb","label":"HERO · full refined mesh","note":"${NATIVE_HERO_FACES} face budget (0 retains every refined face) · ${NATIVE_HERO_ATLAS} atlas · ${NATIVE_HERO_UNWRAP} unwrap · visual asset, not auto-rigged"},
 {"file":"native_high_textured.glb","label":"HIGH · native Pixal material","note":"300k target faces · ${NATIVE_HIGH_RESOLUTION}px model · ${NATIVE_HIGH_ATLAS} atlas · ${NATIVE_UNWRAP} unwrap · visual gate pending"},
 {"file":"native_medium_textured.glb","label":"MEDIUM · native Pixal material","note":"150k target faces · ${NATIVE_MEDIUM_ATLAS} atlas · CPU rebake of the high native material"},
 {"file":"native_low_textured.glb","label":"LOW · native Pixal material","note":"50k target faces · ${NATIVE_LOW_ATLAS} atlas · CPU rebake of the high native material"}
]}
JSON
  echo "FAIL: texture LODs succeeded, but no native rig candidate passed the structural gate; see $OUT/run-status.txt" >&2
  exit 1
fi

cat >"$OUT/run-status.txt" <<EOF
texture_lods=succeeded
rig_state=succeeded
selected_rig=$SELECTED_RIG
rig_profile=$RIG_PROFILE
rig_gate=$RIG_GATE
published_rig=$PUBLISHED_RIG
EOF

write_texture_delivery_manifest succeeded "$SELECTED_RIG"
"$CP/shootout/verify_native_texture_delivery.sh" "$OUT"

cat >"$OUT/stages.json" <<JSON
{"subject":"$LABEL · native refined-mesh image-to-rig runbook","input":"input.png","stages":[
 {"file":"$PUBLISHED_RIG","label":"$RIG_LABEL · native $SELECTED_RIG rig","note":"native Pixal texture; deterministic beam rig; $RIG_GATE"},
 {"file":"native_hero_textured.glb","label":"HERO · full refined mesh","note":"${NATIVE_HERO_FACES} face budget (0 retains every refined face) · ${NATIVE_HERO_ATLAS} atlas · ${NATIVE_HERO_UNWRAP} unwrap · visual asset, not auto-rigged"},
 {"file":"native_high_textured.glb","label":"HIGH · native Pixal material","note":"300k target faces · ${NATIVE_HIGH_RESOLUTION}px model · ${NATIVE_HIGH_ATLAS} atlas · ${NATIVE_UNWRAP} unwrap"},
 {"file":"native_medium_textured.glb","label":"MEDIUM · native Pixal material","note":"150k target faces · ${NATIVE_MEDIUM_ATLAS} atlas · CPU rebake of the high native material"},
 {"file":"native_low_textured.glb","label":"LOW · native Pixal material","note":"50k target faces · ${NATIVE_LOW_ATLAS} atlas · CPU rebake of the high native material"}
]}
JSON
echo "== DONE: $OUT (native $SELECTED_RIG rig selected) =="
