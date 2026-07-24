#!/usr/bin/env bash
# Stage-explorer track for MESH-SOURCED subjects (creatures, quadrupeds,
# tail/wing body plans). These have no input image: they enter the pipeline at
# the rig stage from an authored source mesh, so their stage list is
# source mesh -> FPS samples -> rig -> skeleton -> exercise.
#
# This is the limitations sweep. A subject whose rig gate REJECTS it is kept and
# recorded, not dropped: the rejection is the finding. The raw pre-gate candidate
# is retained next to it so the failure can be looked at.
#
#   stage_explorer_mesh.sh <out-root> [subject ...]
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
F=/mnt/hdd/3d/avatar-shootout/_shootout_out/fixtures
OUT_ROOT="${1:?usage: stage_explorer_mesh.sh <out-root> [subject ...]}"
shift || true
SUBJECTS=("$@")
[[ ${#SUBJECTS[@]} -gt 0 ]] || SUBJECTS=(moth fairy winged_imp fallenangel angelriggy tira winged_bird giraffe)

GPU="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {print $1; exit}')"
[[ -n "$GPU" ]] || { echo "no RTX 3060 found" >&2; exit 1; }

source_for() {
  case "$1" in
    moth)        echo "$F/articulationxl_moth_source.glb" ;;
    fairy)       echo "$F/articulationxl_fairy_source.glb" ;;
    winged_imp)  echo "$F/articulationxl_winged_imp_source.glb" ;;
    fallenangel) echo "$F/articulationxl_fallenangel_source.glb" ;;
    angelriggy)  echo "$F/articulationxl_angelriggy_source.glb" ;;
    tira)        echo "$F/skintokens_tira.glb" ;;
    winged_bird) echo "$F/skintokens_winged_bird.glb" ;;
    giraffe)     echo "/mnt/hdd/3d/avatar-shootout/SkinTokens/examples/giraffe.glb" ;;
    *) echo "unknown subject '$1'" >&2; return 2 ;;
  esac
}

mkdir -p "$OUT_ROOT"

for subject in "${SUBJECTS[@]}"; do
  src="$(source_for "$subject")"
  out="$OUT_ROOT/$subject"
  log="$OUT_ROOT/$subject.e2e.log"
  vram="$OUT_ROOT/$subject.vram.csv"
  [[ -f "$src" ]] || { echo "[mesh] $subject: missing source $src"; continue; }
  echo "=== $subject: $src"
  mkdir -p "$out/rig_inputs"
  cp -f "$src" "$out/source_mesh.glb"

  ( while true; do
      printf '%s,%s\n' "$(date +%s.%N)" \
        "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 0)"
      sleep 2
    done ) > "$vram" &
  sampler=$!
  trap 'kill '"$sampler"' 2>/dev/null || true' EXIT

  start=$(date +%s)
  set +e
  {
    echo "== [1] FPS rig samples =="
    s1=$(date +%s)
    "$CP/mesh_sample_main" "$src" "$out/rig_inputs"
    echo "[1] mesh_sample ($(( $(date +%s) - s1 )).0s)"

    echo "== [2] native SkinTokens rig (generic profile, beams=20) =="
    s2=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU" \
    RIG_PROFILE=generic RIG_SKIN_MODE=learned-smooth RIG_SKIN_SMOOTH_ROUNDS=16 \
    RIG_ALLOW_FLAT_BASECOLOR=1 \
    PIXAL3D_GGUF_DIR=/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf \
    R1W_SRC=/mnt/hdd/3d/avatar-shootout/rig_audit/r1w_real \
      "$CP/rig_texture_chain.sh" "$out/rig_inputs" "$src" \
        /home/dbrain/models/3d/rig/qwen3_w "$out/generic_rigged.glb" 20
    rig_rc=$?
    echo "[2] rig chain ($(( $(date +%s) - s2 )).0s) exit=$rig_rc"
  } > "$log" 2>&1
  rc=$?
  set -e
  end=$(date +%s)

  kill "$sampler" 2>/dev/null || true
  trap - EXIT
  peak=$(awk -F, 'BEGIN{m=0} {if ($2+0>m) m=$2+0} END{print m}' "$vram" 2>/dev/null || echo 0)
  gate_rc=$(awk -F'exit=' '/^\[2\] rig chain/ {print $2}' "$log" | tail -1)

  printf 'subject=%s\nkind=mesh\nexit=%s\nwall_seconds=%d\npeak_vram_mib=%s\nsource=%s\ngpu=%s\n' \
    "$subject" "${gate_rc:-$rc}" "$((end-start))" "$peak" "$src" "$GPU" \
    > "$OUT_ROOT/$subject.timing.txt"
  echo "[mesh] $subject exit=${gate_rc:-$rc} wall=$((end-start))s peak_vram=${peak}MiB rig=$([[ -f "$out/generic_rigged.glb" ]] && echo published || echo REJECTED)"
done

echo "[mesh] done -> $OUT_ROOT"
