#!/usr/bin/env bash
# One config at the REAL prod workload: 4s @ 48fps = 193 frames, 1280x704, captures peak VRAM (5060 Ti).
set -uo pipefail
cd /home/dbrain/dev/longcat-avatar.cpp; mkdir -p ltx_ab_out
LBL="$1"; EENV="$2"; MV="${3:-10.5}"
PEAK=/tmp/peak_$LBL; echo 0 > "$PEAK"
( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null); m=$(cat "$PEAK"); [ "${u:-0}" -gt "${m:-0}" ] && echo "$u" > "$PEAK"; sleep 1; done ) & POLL=$!
LABEL="$LBL" EXTRA_ENV="$EENV" MAX_VRAM="$MV" FRAMES=193 W=1280 H=704 FPS=48 STEPS=8 SEED=42 bash ltx_ab.sh > "ltx_ab_out/run_$LBL.log" 2>&1
kill $POLL 2>/dev/null
echo "RESULT $LBL: $(grep -E 'RESULT\|' ltx_ab_out/run_$LBL.log | tail -1)"
echo "PEAK_VRAM_MiB $LBL: $(cat $PEAK)  (max-vram=$MV)"
echo "OOM? $(grep -c 'out of memory' ltx_ab_out/run_$LBL.log)   webm: $(ls -la ltx_ab_out/$LBL.webm 2>/dev/null | awk '{print $5}')"
