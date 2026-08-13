#!/usr/bin/env bash
# ONE-CALL image -> Python-parity textured -> RIGGED -> animated GLB, with perf capture.
#
# This is the productionized form of shootout/native_ovoxel_dc_parity.sh: image_to_rig --dc-remesh
# does the O-Voxel narrow-band DC + direct volume bake IN-PROCESS and then runs the existing
# SkinTokens rig on that parity mesh, so the whole delivery is a single binary call. The wrapper
# adds only the things a binary should not own: the 3060 lock, VRAM sampling, the rig QC gates,
# and the exercise-animation clip.
#
# usage: final_e2e_dc_rig.sh <matte.png> <outdir> [texsize=2048] [decimate=220000]
#   env: RESUME_STAGE=<dir>  re-enter at the bake from a previous run's stage dir (no diffusion, no DC)
#        SKIP_ANIM=1         skip the exercise-animation clip
# out: <outdir>/rigged.glb (+ .anim.glb), stage/ intermediates, metrics.txt, vram.csv
set -euo pipefail

IMG="${1:?input matte png}"
OUT="${2:?output dir}"
TEXSIZE="${3:-2048}"
DECIMATE="${4:-220000}"

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${GEO_MODEL_DIR:-/home/dbrain/models/3d/geo_f16_native}"
LOCK="/mnt/hdd/3d/avatar-shootout/_shootout_out/.3060-image-to-rig.lock"
# image_to_rig is 3060-ONLY. CUDA enumerates fastest-first so index 0 = 5060 Ti; pin by UUID.
GPU3060="${GPU_3060_UUID:-GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6}"
QC_PY="${RIG_POSE_GATE_PYTHON:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"

mkdir -p "$OUT/stage"
cd "$CPP_DIR"

# ---- [0/3] MATTE GUARD ---------------------------------------------------------------------
# This driver used to assume its input was already a model-facing matte, and a file merely NAMED
# *_matte.png was trusted. toy1_matte.png is a framed studio render with a WHITE background, so
# 63% of the image was consumed as subject and the framing was baked into the mesh. Classify the
# input and matte it with BiRefNet when it is not already a matte.
#
# BiRefNet (ZhengPeng7's exact weights) is the production matter: it is what Python's
# preprocess_image uses, and native's geometric border-flood picks a different silhouette that the
# chaotically sensitive SS-DiT amplifies into a stage-1 collapse. The kobbler RMBG-2.0 service is a
# labelled A/B only (~91.7% stage-1 agreement), not the default.
#   MATTE=skip  trust the input verbatim (for an already-prepared frame)
if [ "${MATTE:-auto}" != "skip" ]; then
  KIND="$(./make_matte --inspect-input "$IMG" | awk -F= '$1=="input_kind" {print $2}')"
  echo "[0/3] input classified: ${KIND:-unknown}"
  case "$KIND" in
    black-matte|rgba-cutout)
      echo "      already model-facing; no matting needed" ;;
    *)
      MATTED="$OUT/input_birefnet_rgba.png"
      echo "      NOT a matte -> BiRefNet neural matte [3060]"
      CUDA_VISIBLE_DEVICES="$GPU3060" flock -w 3600 "$LOCK" \
        "$QC_PY" shootout/birefnet_matte.py "$IMG" "$MATTED" --device cuda 2>&1 | tee "$OUT/00_matte.log"
      test -s "$MATTED" || { echo "FAIL: BiRefNet produced no cutout"; exit 1; }
      IMG="$MATTED" ;;
  esac
fi

RESUME_ARGS=()
if [ -n "${RESUME_STAGE:-}" ]; then
  RESUME_ARGS=(--from-refined "$RESUME_STAGE")
  echo "[resume] bake+rig only, from $RESUME_STAGE (no diffusion, no DC)"
fi
# RIG_CACHE: every tier of one asset must come out with the SAME skeleton, so the first run
# writes it and the rest load it. Without this each tier gets its own skeleton and no single clip
# plays across the ladder.
if [ -n "${RIG_CACHE:-}" ]; then
  RESUME_ARGS+=(--rig-cache "$RIG_CACHE")
fi

echo "[1/3] image_to_rig --dc-remesh (geometry -> DC parity mesh -> bake -> rig)  [3060]"
SAMPLER=$(bash "$CPP_DIR/shootout/gpu_sample.sh" start "$OUT/vram.csv")
trap 'bash "$CPP_DIR/shootout/gpu_sample.sh" stop "$SAMPLER" 2>/dev/null || true' EXIT
T0=$(date +%s)
set +e
CUDA_VISIBLE_DEVICES="$GPU3060" nice -n 10 ionice -c2 -n7 \
  flock -w 3600 "$LOCK" ./image_to_rig \
    --model "$MODEL" --image "$IMG" \
    --dc-remesh --texsize "$TEXSIZE" --decimate "$DECIMATE" \
    --stage-dir "$OUT/stage" "${RESUME_ARGS[@]}" \
    --out "$OUT/rigged.glb" 2>&1 | tee "$OUT/01_e2e.log"
RC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
bash "$CPP_DIR/shootout/gpu_sample.sh" stop "$SAMPLER"; trap - EXIT
[ "$RC" -eq 0 ] || { echo "FAIL: image_to_rig exited $RC"; exit 1; }
test -f "$OUT/rigged.glb" || { echo "FAIL: no rigged.glb"; exit 1; }

echo "[2/3] rig QC gates (skeleton score / weight health / pose gate)"
./rig_score "$OUT/rigged.glb" > "$OUT/qc_rig_score.txt" 2>&1 || true
"$QC_PY" rig_weight_health.py "$OUT/rigged.glb" > "$OUT/qc_weight_health.txt" 2>&1 || true
"$QC_PY" rig_pose_smoke.py "$OUT/rigged.glb" "$OUT/qc_pose_gate.png" --pose-gate \
  > "$OUT/qc_pose_gate.txt" 2>&1 || true

if [ -z "${SKIP_ANIM:-}" ]; then
  echo "[3/3] exercise animation (every materially weighted joint swings in turn)"
  "$QC_PY" rig_exercise_anim.py "$OUT/rigged.glb" "$OUT/rigged.anim.glb" \
    > "$OUT/04_anim.log" 2>&1 || echo "  WARN: exercise animation failed (see 04_anim.log)"
fi

# ---- metrics: the per-stage times the binary printed + the sampled VRAM peak ----
{
  echo "input=$IMG"
  echo "texsize=$TEXSIZE decimate=$DECIMATE"
  echo "wall_total_s=$((T1-T0))"
  bash "$CPP_DIR/shootout/gpu_sample.sh" peak "$OUT/vram.csv"
  grep -E '^\s*\[(1/4|1a0/4|2/4|3/4)\]|^\[[0-9]+\] |^==== DONE' "$OUT/01_e2e.log" || true
  echo "-- rig score --";      cat "$OUT/qc_rig_score.txt" 2>/dev/null | tail -20
  echo "-- weight health --";  tail -6 "$OUT/qc_weight_health.txt" 2>/dev/null
  echo "-- pose gate --";      tail -6 "$OUT/qc_pose_gate.txt" 2>/dev/null
} > "$OUT/metrics.txt"

echo "DONE -> $OUT/rigged.glb"
sed -n '1,6p' "$OUT/metrics.txt"
