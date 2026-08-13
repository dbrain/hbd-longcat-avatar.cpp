#!/usr/bin/env bash
# After stage_explorer_e2e.sh: produce the two eye-test-only artifacts the
# pipeline does not emit itself, under canonical names the page expects.
#
#   hymotion_rigged.pose-gate.png/.txt  skeleton + rest-vs-posed deformation evidence
#   hymotion_rigged.exercise.glb        every influential joint swings in turn
#
# Both are derived from the published rig; neither modifies it.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${RIG_POSE_GATE_PYTHON:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"
ROOT="${1:?usage: stage_explorer_postprocess.sh <out-root>}"
GPU="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {print $1; exit}')"

for dir in "$ROOT"/*/; do
  subject="$(basename "$dir")"
  # Mesh-sourced (creature) subjects publish generic_rigged.glb; image-driven
  # subjects publish hymotion_rigged.glb.
  if [[ -f "$dir/source_mesh.glb" ]]; then stem=generic_rigged; else stem=hymotion_rigged; fi
  rig="$dir/$stem.glb"
  [[ -f "$rig" ]] || { echo "[post] $subject: NO PUBLISHED RIG — the gate rejected it (this is a recorded limitation, not a skip)"; continue; }

  echo "[post] $subject: pose gate"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$CP/rig_pose_smoke.py" "$rig" \
    "$dir/$stem.pose-gate.png" --generic-all-influential --show-skeleton --pose-gate \
    | tee "$dir/$stem.pose-gate.txt" | tail -1 || true

  echo "[post] $subject: weight health"
  "$PY" "$CP/rig_weight_health.py" "$rig" | tee "$dir/$stem.weight-health.txt" | tail -2 || true

  echo "[post] $subject: exercise clip"
  "$PY" "$CP/rig_exercise_anim.py" "$rig" "$dir/$stem.exercise.glb" || true
done

echo "[post] manifest"
"$PY" "$CP/shootout/stage_explorer_manifest.py" "$ROOT" > "$ROOT/stages.json"
echo "[post] wrote $ROOT/stages.json"
