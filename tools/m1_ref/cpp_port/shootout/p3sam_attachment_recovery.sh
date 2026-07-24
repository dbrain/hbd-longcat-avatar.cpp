#!/usr/bin/env bash
# Fail-closed P3-SAM attachment recovery after a *recorded* full-mesh
# SkinTokens structural rejection.  This does not name regions (hair, wings,
# limbs, etc.) and never publishes a Hymotion hand-off.  It creates one
# reviewable recovery candidate plus an arm-raise pose smoke image.
#
# Usage:
#   p3sam_attachment_recovery.sh <full-rig-rejection.log> <p3sam-mesh.glb>
#       <p3sam-face-ids.npy> <delivery-textured.glb> <qwen3-weights> <out-dir>
#       [candidate-index=0]
#
# P3-SAM segmentation is deliberately a separate preceding stage: its labels
# are geometric regions, and this runner retains both its mesh and label file
# in the output provenance.  A candidate may be reviewed, but must not be
# promoted automatically: a wing, tail, third arm, or weapon can also be a
# large symmetric pair.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_LOG="${1:?full-rig-rejection.log}"; P3_MESH="${2:?p3sam mesh}"; FACE_IDS="${3:?P3-SAM face ids}"
DELIVERY="${4:?full textured delivery GLB}"; QW="${5:?qwen3 weights}"; OUT="${6:?output dir}"
INDEX="${7:-0}"
RIG_SAMPLE_SEED="${RIG_SAMPLE_SEED:-42}"
RIG_BODY_SKIN_MODE="${RIG_BODY_SKIN_MODE:-learned-smooth}"
RIG_SKIN_SMOOTH_ROUNDS="${RIG_SKIN_SMOOTH_ROUNDS:-32}"

[[ -f "$FULL_LOG" && -f "$P3_MESH" && -f "$FACE_IDS" && -f "$DELIVERY" ]] || {
  echo "missing full-rig log, P3-SAM evidence, or delivery mesh" >&2; exit 2;
}
[[ "$INDEX" =~ ^[0-9]+$ ]] || { echo "candidate index must be a non-negative integer" >&2; exit 2; }
[[ "$RIG_SAMPLE_SEED" =~ ^[0-9]+$ ]] || { echo "RIG_SAMPLE_SEED must be a non-negative integer" >&2; exit 2; }
[[ "$RIG_BODY_SKIN_MODE" == learned || "$RIG_BODY_SKIN_MODE" == learned-smooth ]] || {
  echo "RIG_BODY_SKIN_MODE must be learned or learned-smooth" >&2; exit 2;
}
[[ "$RIG_SKIN_SMOOTH_ROUNDS" =~ ^[0-9]+$ ]] && (( RIG_SKIN_SMOOTH_ROUNDS >= 1 && RIG_SKIN_SMOOTH_ROUNDS <= 64 )) || {
  echo "RIG_SKIN_SMOOTH_ROUNDS must be an integer in [1,64]" >&2; exit 2;
}
[[ -d "$QW" ]] || { echo "missing Qwen rig weights: $QW" >&2; exit 2; }
: "${PIXAL3D_GGUF_DIR:?set PIXAL3D_GGUF_DIR}"
: "${R1W_SRC:=/home/dbrain/models/3d/rig/r1w_real}"
[[ -d "$R1W_SRC" ]] || { echo "missing real R1 weights: $R1W_SRC" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to reuse output directory: $OUT" >&2; exit 2; }
rg -qi 'structural candidate .* rejected|structural gate|rig_state=rejected' "$FULL_LOG" || {
  echo "refusing attachment recovery without a recorded full-mesh structural rejection: $FULL_LOG" >&2; exit 2;
}
for f in mesh_sample_main rig_texture_chain.sh rig_score p3sam_mask_view_native; do
  [[ -x "$CP/$f" ]] || { echo "missing executable: $CP/$f" >&2; exit 2; }
done

PY3D="${PYTHON3D:-/mnt/hdd/3d/avatar-shootout/SkinTokens/.venv/bin/python}"
POSE_PY="${POSE_PYTHON:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"
[[ -x "$PY3D" && -x "$POSE_PY" ]] || { echo "missing configured Python environment" >&2; exit 2; }

mkdir -p "$OUT"
STATUS="$OUT/recovery-status.txt"
# Keep failed recoveries inspectable.  The caller supplies this runner only
# after a recorded structural rejection, so an early exit is useful evidence,
# not disposable noise.  The final asset remains experimental regardless of
# this status file; production entry points do not invoke this helper.
on_recovery_error() {
  local rc=$?
  {
    echo "state=rejected"
    echo "exit_code=$rc"
    echo "failed_line=${BASH_LINENO[0]:-unknown}"
    echo "full_mesh_rejection_log=$FULL_LOG"
    echo "p3sam_mesh=$P3_MESH"
    echo "p3sam_face_ids=$FACE_IDS"
    echo "candidate_index=$INDEX"
    echo "failure_log=$OUT/recovery.log"
    echo "promotion_rule=Never publish a rejected or experimental recovery candidate."
  } >"$STATUS"
  exit "$rc"
}
trap on_recovery_error ERR
exec > >(tee "$OUT/recovery.log") 2>&1
RANK="$OUT/attachment_candidates.json"
"$PY3D" "$CP/shootout/p3sam_attachment_candidates.py" "$P3_MESH" "$FACE_IDS" "$RANK"
LABELS="$($PY3D - "$RANK" "$INDEX" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
index = int(sys.argv[2])
try:
    labels = data["candidates"][index]["labels"]
except IndexError:
    raise SystemExit(f"candidate index {index} is unavailable")
if len(labels) != 2 or not all(isinstance(x, int) for x in labels):
    raise SystemExit("candidate must be one explicit pair of integer P3-SAM labels")
print(*labels)
PY
)"
read -r -a LABEL_ARRAY <<<"$LABELS"

BODY="$OUT/inference_view.glb"
# Native mesh filtering keeps this recovery path honest about where its
# skeleton input comes from. Python remains only for P3-SAM diagnostic label
# ranking and the historical post-transfer attachment experiment; it does not
# generate the SkinTokens skeleton or sampled weights.
"$CP/p3sam_mask_view_native" "$P3_MESH" "$FACE_IDS" "$BODY" --drop "${LABEL_ARRAY[@]}"
SAMPLES="$OUT/rig_inputs"
mkdir -p "$SAMPLES"
"$CP/mesh_sample_main" "$BODY" "$SAMPLES" 8192 512 "$RIG_SAMPLE_SEED"
FULL_TRANSFER="$OUT/full_mesh_skin_transfer.glb"
# `rig_texture_chain` deliberately permits conditioning from the body view
# while transferring its predicted weights to the untouched textured delivery
# mesh.  Do not use BODY here: P3-SAM's view is geometry-only and has no UVs.
# The delivery mesh is deliberately changed by the attachment pass below.
# Defer the chain's gate, then apply the authoritative actual-GLB gate to the
# post-attachment asset.  This keeps a pre-attachment smoke from certifying a
# different asset than the one exposed for review.
R1W_SRC="$R1W_SRC" RIG_SKIN_MODE="$RIG_BODY_SKIN_MODE" RIG_SKIN_SMOOTH_ROUNDS="$RIG_SKIN_SMOOTH_ROUNDS" RIG_ALLOW_TRANSFER_MESH_MISMATCH=1 RIG_DEFER_POSE_GATE=1 \
  "$CP/rig_texture_chain.sh" "$SAMPLES" "$DELIVERY" "$QW" "$FULL_TRANSFER" 20
FULL_RIGGED="$OUT/attachment_recovery_experimental.glb"
"$PY3D" "$CP/shootout/p3sam_rigid_attach.py" "$FULL_TRANSFER" "$P3_MESH" "$FACE_IDS" "$FULL_RIGGED" --labels "${LABEL_ARRAY[@]}"
"$CP/rig_score" "$FULL_RIGGED" 50 | tee "$OUT/rig-score.txt"
PYOPENGL_PLATFORM=egl "$POSE_PY" "$CP/rig_pose_smoke.py" "$FULL_RIGGED" "$OUT/arm_raise_pose_smoke.png" --show-skeleton --pose-gate

cat >"$OUT/recovery-status.txt" <<EOF
state=experimental-review-required-passed-real-deformation-gate
full_mesh_rejection_log=$FULL_LOG
p3sam_mesh=$P3_MESH
p3sam_face_ids=$FACE_IDS
candidate_index=$INDEX
candidate_labels=${LABEL_ARRAY[*]}
rig_sample_seed=$RIG_SAMPLE_SEED
body_skin_mode=$RIG_BODY_SKIN_MODE
skin_smooth_rounds=$RIG_SKIN_SMOOTH_ROUNDS
delivery_mesh=$DELIVERY
inference_view=$BODY
full_mesh_skin_transfer=$FULL_TRANSFER
attachment_candidate=$FULL_RIGGED
pose_smoke=arm_raise_pose_smoke.png
promotion_rule=Do not publish automatically. The final post-attachment GLB passed the real arm-raise deformation gate, but review appendage semantics and the pose smoke before any creature-specific promotion; direct full-mesh SkinTokens remains the normal route for winged and multi-limbed models.
EOF
echo "[p3sam-recovery] REVIEW REQUIRED: $FULL_RIGGED and $OUT/arm_raise_pose_smoke.png"
