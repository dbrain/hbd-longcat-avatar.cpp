#!/usr/bin/env bash
# Fresh full image->textured LODs->rig e2e for every image-driven subject, with
# per-stage wall time and peak VRAM, retaining EVERY intermediate so each stage
# can be eye-tested independently.
#
#   stage_explorer_e2e.sh <out-root> [subject ...]
#
# Native only. No Python fallback exists in this path: RIG_OFFICIAL_SAMPLED_FALLBACK
# and RIG_COMPONENT_BRANCH_REPAIR both fail closed, and the only Python invoked is
# the read-only EGL pose-gate renderer.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/mnt/hdd/3d/avatar-shootout/_shootout_out
OUT_ROOT="${1:?usage: stage_explorer_e2e.sh <out-root> [subject ...]}"
shift || true
SUBJECTS=("$@")
[[ ${#SUBJECTS[@]} -gt 0 ]] || SUBJECTS=(miku gilly soldier)

GPU="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {print $1; exit}')"
[[ -n "$GPU" ]] || { echo "no RTX 3060 found" >&2; exit 1; }
echo "[stage-explorer] 3060 = $GPU"

image_for() {
  case "$1" in
    miku)    echo "$S/miku_prod1536_matte.png" ;;
    gilly)   echo "$S/runbook_image_to_rig/gilly/gilly_matte.png" ;;
    soldier) echo "$S/inline_soldier/input.png" ;;
    *) echo "unknown subject '$1'" >&2; return 2 ;;
  esac
}

mkdir -p "$OUT_ROOT"

for subject in "${SUBJECTS[@]}"; do
  image="$(image_for "$subject")"
  out="$OUT_ROOT/$subject"
  log="$OUT_ROOT/$subject.e2e.log"
  vram="$OUT_ROOT/$subject.vram.csv"
  echo "=== $subject: $image -> $out"
  mkdir -p "$out"

  # Sample VRAM for the whole subject run; the per-stage split comes from the
  # timestamps the wrapper prints into the log.
  ( while true; do
      printf '%s,%s\n' "$(date +%s.%N)" \
        "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 0)"
      sleep 2
    done ) > "$vram" &
  sampler=$!
  trap 'kill '"$sampler"' 2>/dev/null || true' EXIT

  start=$(date +%s)
  set +e
  IMAGE_TO_RIG_OUT_ROOT="$OUT_ROOT" \
  IMAGE_TO_RIG_CLAY_VERDICT=approved \
  NATIVE_HERO_FACES=0 NATIVE_HERO_ATLAS=8192 \
  RIG_PROFILE=humanoid \
    "$CP/shootout/native_image_to_rig_from_image.sh" "$image" "$out" "$subject" \
    > "$log" 2>&1
  rc=$?
  set -e
  end=$(date +%s)

  kill "$sampler" 2>/dev/null || true
  trap - EXIT
  peak=$(awk -F, 'BEGIN{m=0} {if ($2+0>m) m=$2+0} END{print m}' "$vram" 2>/dev/null || echo 0)

  printf 'subject=%s\nexit=%d\nwall_seconds=%d\npeak_vram_mib=%s\nimage=%s\ngpu=%s\n' \
    "$subject" "$rc" "$((end-start))" "$peak" "$image" "$GPU" > "$OUT_ROOT/$subject.timing.txt"
  echo "[stage-explorer] $subject exit=$rc wall=$((end-start))s peak_vram=${peak}MiB"
done

echo "[stage-explorer] done -> $OUT_ROOT"
