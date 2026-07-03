#!/usr/bin/env bash
# Lever #9: DiT step-count quality/perf A/B. Distill DMD = high-noise (structure) + low-noise (detail).
# Baseline = 2+2 = 4 steps. Each step dropped ≈ -25% of that expert's DiT. Render full segs, measure DiT
# + eyeball quality. Uses the new bests: maxv7.3 + --vae-tile-overlap 0.25 (decode -40%).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/steps"; mkdir -p "$OUT/gcache"
M=/models; DR=/models/_drive; W=480; H=832; FR=21; MAXV=7.3; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P1="A young man with tousled dark brown hair sings into the camera, warm amber neon at dusk, cinematic, medium shot."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
# $1=tag $2=high_steps $3=low_steps
run(){ local tag=$1 hs=$2 ls=$3; local D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/steps/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --steps $ls --high-noise-steps $hs \
    --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P1" \
    -o /src/perf_out/steps/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  local hi=$(grep "sampling(high noise) completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local lo=$(grep -E "^\[INFO \] stable-diffusion.cpp.*sampling completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local gen=$(grep "generate_video completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  printf "  %-10s (hi=%s lo=%s) rc=%s  DiT_hi=%-7s DiT_lo=%-7s gen=%-8s peak=%sMiB wall=%.1fs\n" \
    "$tag" "$hs" "$ls" "$rc" "${hi:-?}" "${lo:-?}" "${gen:-FAIL}" "$(cat $D/peak.txt 2>/dev/null)" "$(awk "BEGIN{print $(date +%s.%N)-$t0}")"
}
echo "=== DiT step-count A/B (480x832 FR=21, maxv7.3, ov0.25) ==="
run s4_2x2 2 2   # baseline 4-step
run s3_2x1 2 1   # drop a low-noise (detail) step
run s3_1x2 1 2   # drop a high-noise (structure) step
run s2_1x1 1 1   # 2-step (aggressive)
echo "=== done. eyeball perf_out/steps/<tag>/f000.png (+ mid frame) vs s4_2x2 ==="
