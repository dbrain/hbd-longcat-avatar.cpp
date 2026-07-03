#!/usr/bin/env bash
# Lever L5: background prefetch thread (LONGCAT_OFFLOAD_PREFETCH_THREAD=1) to overlap the
# 5.34s/step-pair pageable-mmap HtoD weight stream under DiT compute (the measured 14% idle stall).
# Full 4-step seg, prefetch OFF (baseline) vs ON. Measure DiT hi/lo + gen + peak VRAM + frame bit-exact.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/prefetch"; mkdir -p "$OUT/gcache"
M=/models; DR=/models/_drive; W=480; H=832; FR=21; MAXV=7.3; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P1="A young man with tousled dark brown hair sings into the camera, warm amber neon at dusk, cinematic, medium shot."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.3; done; echo "$p">"$2"; }
run(){ local tag=$1; shift; local D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/prefetch/gcache "$@" "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
    --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P1" \
    -o /src/perf_out/prefetch/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  local hi=$(grep "sampling(high noise) completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local lo=$(grep -E "stable-diffusion.cpp:[0-9]+ +- sampling completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local gen=$(grep "generate_video completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local pf=$(grep -c "prefetch thread ON" "$D/run.log")
  printf "  %-10s rc=%s  DiT_hi=%-7s DiT_lo=%-7s gen=%-8s peak=%sMiB prefetchON=%s wall=%.1fs\n" \
    "$tag" "$rc" "${hi:-?}" "${lo:-?}" "${gen:-FAIL}" "$(cat $D/peak.txt 2>/dev/null)" "$pf" "$(awk "BEGIN{print $(date +%s.%N)-$t0}")"
}
echo "=== prefetch-thread A/B (480x832 FR=21 4-step maxv7.3 ov0.25) ==="
run off
run on  -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1
echo "=== frame bit-exact off vs on ==="
d=0; n=0; for f in "$OUT"/off/f*.png; do bn=$(basename "$f"); n=$((n+1)); cmp -s "$f" "$OUT/on/$bn" || { echo "DIFF $bn"; d=$((d+1)); }; done
echo "=== $n frames, $d differ ==="
