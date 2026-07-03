#!/usr/bin/env bash
# probe_vram.sh — fast VRAM probe: launch a 65f@1280 config, wait until the DiT vace-block
# reserves (the ceiling, buf~6268) have printed, capture the max driver_used, then KILL.
# ~3 min per config vs a ~15 min full render. Usage: bash probe_vram.sh <tag> [KEY=VAL ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TAG="${1:?tag}"; shift
ENVKV=("$@")
sampf="/tmp/vram_probe_${TAG}.log"; : > "$sampf"
{ while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null; sleep 0.3; done ; } > "$sampf" &
SAMP=$!
env "${ENVKV[@]}" LONGCAT_VRAM_BREAKDOWN=1 WAN_ROPE_F16=1 FR=65 W=1280 H=704 \
  bash ./iter_seg2.sh "the same person keeps dancing energetically on the neon-lit city street at night, cinematic" "probe_$TAG" >/dev/null 2>&1 &
LOG="cont_probe_$TAG/log"
# wait until at least 3 vace-block (buf 62xx) reserves have printed, or OOM, or timeout
for i in $(seq 1 100); do
  sleep 6
  n=$(grep -ac "compute_buf=62[0-9][0-9] MB" "$LOG" 2>/dev/null); n=${n:-0}; n=$(echo $n | tr -d "\n" | head -c4)
  if grep -qa "out of memory" "$LOG" 2>/dev/null; then echo "[$TAG] OOM"; break; fi
  if [ "${n:-0}" -ge 3 ]; then break; fi
done
kill $SAMP 2>/dev/null
# kill the render container/process
pid=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1)
cn=$(docker ps --format '{{.Names}} {{.Image}}' | awk '/longcat-avatar-dev:builder/{print $1; exit}')
[ -n "$cn" ] && docker kill "$cn" >/dev/null 2>&1
[ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
sleep 2
echo "=== [$TAG] flags: ${ENVKV[*]} ==="
echo "vace-block DiT reserves (driver / buf / params):"
grep -a "\[VRAM\] Wan2.x-VACE.*reserve" "$LOG" 2>/dev/null | grep -a "compute_buf=62" \
  | sed -E 's/.*driver_used=([0-9]+).*compute_buf=([0-9]+).*(partial=[0-9]+ prefetched=[0-9]+ pool=[0-9]+\([0-9]+\)).*/  driver=\1 buf=\2 \3/' | sort -u
maxd=$(grep -a "\[VRAM\] Wan2.x-VACE.*reserve" "$LOG" 2>/dev/null | sed -E 's/.*driver_used=([0-9]+).*/\1/' | sort -n | tail -1)
echo "  MAX DiT driver_used = ${maxd:-?} MiB   (target <= 11776)   smi-peak=$(sort -n "$sampf"|tail -1) MiB"
