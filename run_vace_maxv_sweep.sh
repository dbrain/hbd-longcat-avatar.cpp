#!/usr/bin/env bash
# Lever #2: DiT --max-vram re-sweep for the 9.87GB VACE-FUN expert.
# maxv6 was tuned for the smaller i2v expert (8.15GB) -> VACE got 42->9 graph cuts.
# Single fresh segment per budget (gray-fast ON so the cheap encode doesn't mask DiT).
# Captures: graph-cut segment count, DiT sampling wall, per-step s/it, peak VRAM (smi poll).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/maxvsweep"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=480; H=832; FR="${FR:-21}"; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf
VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P1="A young man with tousled dark brown hair sings into the camera, warm amber neon at dusk, cinematic, medium shot."

poll_vram(){ # $1 = logfile; writes max MiB to $2 on exit
  local out=$1; local peak=0
  while kill -0 "$3" 2>/dev/null; do
    local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -gt "$peak" ] && peak=$u
    sleep 0.5
  done
  echo "$peak" > "$out"
}

for MAXV in 6 7 8 9; do
  D="$OUT/mv$MAXV"; rm -rf "$D"; mkdir -p "$D/seg1"
  echo "=== max-vram=$MAXV ==="; t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/maxvsweep/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler \
    --steps 2 --high-noise-steps 2 --flow-shift 7 -W $W -H $H --video-frames $FR \
    --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --vae-relative-tile-size 0.25x0.25 --temporal-tiling -s $SEED \
    --diffusion-model $VL --high-noise-diffusion-model $VH \
    --init-img $DR/char.png -p "$P1" -o /src/perf_out/maxvsweep/mv$MAXV/f%03d.png -v > "$D/run.log" 2>&1 &
  RUNPID=$!
  mkdir -p "$OUT/gcache"
  ( poll_vram "$D/peak.txt" x $RUNPID ) &
  POLLPID=$!
  wait $RUNPID; rc=$?
  wait $POLLPID 2>/dev/null
  wall=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N)-$t0}")
  cuts=$(grep -oE "merged [0-9]+ segments -> [0-9]+ segments" "$D/run.log" | grep VACE -A0 | tail -1)
  cuts=$(grep "Wan2.x-VACE-14B graph cut max_vram" "$D/run.log" | grep -oE "merged [0-9]+ segments -> [0-9]+ segments" | tail -1)
  dit=$(grep -E "sampling completed, taking|sampling\(high noise\) completed" "$D/run.log" | grep -oE "[0-9.]+s" | tr '\n' ' ')
  gen=$(grep "generate_video completed" "$D/run.log" | grep -oE "[0-9.]+s" | tail -1)
  peak=$(cat "$D/peak.txt" 2>/dev/null)
  echo "  rc=$rc wall=${wall}s  DiT=[$dit] gen=$gen  cuts=[$cuts]  peakVRAM=${peak}MiB"
done
echo "=== sweep done ==="
