#!/usr/bin/env bash
# run_and_log.sh — render ONE VACE segment via iter_seg2.sh, capture peak VRAM + wall +
# per-module VRAM breakdown, and append a results row to the eye-test page + a TSV log.
# Stops the hand-editing of ltx_dub_next.html (handoff ask).
#
# Usage:  bash run_and_log.sh <tag> "<prompt>" [CHECK="quality note"] [KEY=VAL ...]
#   Any KEY=VAL after the prompt is exported into iter_seg2's environment
#   (FR, W, H, MAXV, WAN_ROPE_F16, VTILE, LONGCAT_NO_OFFLOAD_PIPELINING, ...).
#   CHECK= sets the quality-check line shown on the page (optional).
# The stitched clip cont_<tag>.mp4 is written to the eye-test out/ dir by iter_seg2.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"
OUT="$REPO/tools/lightx2v-test/out"; PAGE="$OUT/ltx_dub_next.html"; TSV="$OUT/runlog.tsv"
TAG="${1:?tag}"; PROMPT="${2:?prompt}"; shift 2
CHECK=""; ENVKV=()
for a in "$@"; do
  case "$a" in CHECK=*) CHECK="${a#CHECK=}";; *=*) ENVKV+=("$a");; esac
done

sampf="/tmp/vram_${TAG}.log"; : > "$sampf"
{ while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null; sleep 0.3; done ; } > "$sampf" &
SAMP=$!
t0=$(date +%s)
env "${ENVKV[@]}" LONGCAT_VRAM_BREAKDOWN=1 bash ./iter_seg2.sh "$PROMPT" "$TAG" >/dev/null 2>&1
rc=$?
t1=$(date +%s); kill $SAMP 2>/dev/null
LOG="cont_$TAG/log"; WALL=$((t1-t0))
PEAK=$(sort -n "$sampf" 2>/dev/null | tail -1)
FRAMES=$(ls "cont_$TAG"/*.png 2>/dev/null | wc -l)
OOM=$(grep -aciE "out of memory|GGML_ASSERT.*cache" "$LOG" 2>/dev/null)
# per-module max driver_used + its compute_buf
BRK=$(grep -a "\[VRAM\]" "$LOG" 2>/dev/null | sed -E 's/.*\] ([A-Za-z0-9_.-]+) reserve: driver_used=([0-9]+) MB.*compute_buf=([0-9]+) MB.*/\1 \2 \3/' \
  | awk '{if($2+0>m[$1]){m[$1]=$2+0;b[$1]=$3}} END{for(k in m) printf "%s=%d(buf %d) ",k,m[k],b[k]}')
FR=$(grep -aoE "video-frames [0-9]+" "$LOG" | head -1 | awk '{print $2}')
RES=$(grep -aoE "generate_video [0-9]+x[0-9]+" "$LOG" | head -1 | awk '{print $2}')
FLAGS=$(printf '%s ' "${ENVKV[@]}")
STATUS="ok"; [ "$OOM" -gt 0 ] && STATUS="OOM"; [ "$FRAMES" -lt "${FR:-1}" ] && STATUS="incomplete"

printf '%s\t%s\t%s\tpeak=%sMiB\twall=%ss\tframes=%s/%s\t%s\t%s\t%s\n' \
  "$(date +%F_%T)" "$TAG" "$STATUS" "${PEAK:-?}" "$WALL" "$FRAMES" "${FR:-?}" "$RES" "$FLAGS" "$BRK" | tee -a "$TSV"

# insert an HTML row (newest on top) before the marker
PEAKGB=$(awk -v p="${PEAK:-0}" 'BEGIN{printf "%.2f", p/1024}')
peakclass="ok"; awk -v p="${PEAK:-0}" 'BEGIN{exit !(p>11776)}' && peakclass="bad"
[ "$STATUS" != ok ] && peakclass="bad"
row=$(cat <<HTML
<div class="rl">
  <video src="cont_${TAG}.mp4" controls loop muted></video>
  <div>
    <h3>${TAG} — ${RES:-?} · ${FR:-?}f · <span class="${peakclass}">peak ${PEAK:-?} MiB (${PEAKGB} GB)</span> · ${STATUS}</h3>
    <p class="meta">wall ${WALL}s · frames ${FRAMES}/${FR:-?} · flags: ${FLAGS}
per-module VRAM: ${BRK}</p>
    <p style="font-size:13px;margin:0">$( [ -n "$CHECK" ] && echo "<span class=q>✔ $CHECK</span>" || echo "prompt: ${PROMPT}" )</p>
  </div>
</div>
HTML
)
if grep -q "RUNLOG-INSERT" "$PAGE" 2>/dev/null; then
  tmp=$(mktemp)
  awk -v row="$row" '/RUNLOG-INSERT/{print; print row; next} {print}' "$PAGE" > "$tmp" && mv "$tmp" "$PAGE"
  echo "[run_and_log] appended row for $TAG to $PAGE"
else
  echo "[run_and_log] WARN: no RUNLOG-INSERT marker in $PAGE; TSV updated only"
fi
echo "[run_and_log] $TAG: peak=${PEAK}MiB wall=${WALL}s frames=${FRAMES}/${FR} status=$STATUS"
