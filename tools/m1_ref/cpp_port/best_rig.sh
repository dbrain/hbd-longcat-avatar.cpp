#!/usr/bin/env bash
# best_rig.sh — best-of-N auto-rig: the C++ sampled beam is clean ~6/10 seeds but occasionally emits a
# head-fan / runaway (fp-forward drift vs Python's bf16; exact match is infeasible). rig_score now scores
# the fan defect (maxfan>6 -> TOTAL 0), so generating N seeds and keeping the highest TOTAL reliably yields
# a clean Python-range rig. Saves the winning rig GLB + its R4/R5 dump (joints/parents/skin) for combine.
#
# Usage: best_rig.sh <rig_in_dir> <qwen3_w_dir> <out_rig.glb> <out_dump_dir> [N=6] [extra e2e args...]
# Env: PIXAL3D_GGUF_DIR must be set.
set -euo pipefail
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIG_IN="${1:?rig_in_dir}"; QW="${2:?qwen3_w_dir}"; OUT_GLB="${3:?out_rig.glb}"; OUT_DUMP="${4:?out_dump_dir}"
N="${5:-6}"; shift $(( $#<5 ? $# : 5 )) || true; EXTRA="${*:-}"
: "${PIXAL3D_GGUF_DIR:?set PIXAL3D_GGUF_DIR}"
DUMP=/tmp/skintokens_e2e
mkdir -p "$OUT_DUMP" "$(dirname "$OUT_GLB")"
best_total="-1"; best_seed=""; best_J=""
echo "== best-of-$N auto-rig (fan-aware selection) =="
for s in $(seq 1 "$N"); do
  G="/tmp/best_rig_s$s.glb"
  "$CP/skintokens_e2e" "$RIG_IN" "$QW" "$G" cuda beam prec=fp32 seed="$s" r1=banked cond=banked \
      beams=10 maxnew=2048 temp=1.5 topk=10 $EXTRA >/tmp/best_rig_s$s.log 2>&1 || { echo "  seed $s FAILED"; continue; }
  line=$("$CP/rig_score" "$G" 55 2>/dev/null | head -1)
  T=$(echo "$line" | grep -oP 'TOTAL=\K[0-9.]+'); J=$(echo "$line" | grep -oP 'J=\K[0-9]+'); MF=$(echo "$line" | grep -oP 'maxfan=\K[0-9]+')
  echo "  seed $s: J=${J:-?} maxfan=${MF:-?} TOTAL=${T:-0}"
  if awk "BEGIN{exit !(${T:-0} > $best_total)}"; then
    best_total="$T"; best_seed="$s"; best_J="$J"
    cp "$G" "$OUT_GLB"; cp "$DUMP"/gen_joints.npy "$DUMP"/gen_parents.npy "$DUMP"/gen_skin_pred.npy "$OUT_DUMP"/ 2>/dev/null || true
  fi
done
echo "== BEST: seed=$best_seed J=$best_J TOTAL=$best_total -> $OUT_GLB (dump $OUT_DUMP) =="
[ -n "$best_seed" ]
