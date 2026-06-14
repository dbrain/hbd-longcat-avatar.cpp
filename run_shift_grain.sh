#!/usr/bin/env bash
# FREE speckle lever test: does LOWER shift (same 4 steps) reduce the "dotty" grain vs paying +17% for 2+4?
# Neon street (the dotty shot), seed 42. shift 7/5/3 @ 4-step (free) + 2+4@shift7 (paid) reference.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/shift_grain"; rm -rf "$OUT"; mkdir -p "$OUT/gcache"
M=/models; W=1280; H=704; FR=13; MAXV=7.3; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A man walks down a rain-slick neon street at night singing to the camera, reflections shimmering, breath visible in the cold air. Slow steadicam tracking shot. Saturated cyan and magenta neon, cinematic, volumetric light, shallow depth of field, high detail."
# 4-step grids at shift S (linear: t/1000 -> shift*r/(1+(shift-1)r)) for trained [1000,750,500,250]:
sig4(){ awk -v S=$1 'BEGIN{n=split("1000 750 500 250",t," "); o=""; for(i=1;i<=n;i++){r=t[i]/1000; s=S*r/(1+(S-1)*r); o=o sprintf("%.5f,",s)} print o "0.0"}'; }
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
run(){ local tag=$1; shift; local D="$OUT/$tag"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/shift_grain/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler \
    "$@" -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/shift_grain/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  ffmpeg -y -framerate 16 -i "$D/f%03d.png" -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p "$OUT/${tag}.mp4" -loglevel error 2>/dev/null
  ffmpeg -y -i "$D/f006.png" -vf "crop=500:360:380:300,scale=1000:720:flags=neighbor" "$OUT/${tag}_zoom.png" -loglevel error 2>/dev/null || true
  echo "  $tag rc=$rc $(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1) $(awk "BEGIN{printf \"%.0fs\",$(date +%s.%N)-$t0}")"
}
echo "=== shift-for-grain (neon, seed $SEED) ==="
run sh7_4step --sigmas "$(sig4 7)" --high-noise-steps 2 --steps 2   # current default (dotty)
run sh5_4step --sigmas "$(sig4 5)" --high-noise-steps 2 --steps 2   # FREE: lower shift
run sh3_4step --sigmas "$(sig4 3)" --high-noise-steps 2 --steps 2   # FREE: lowest shift (user's pick)
run sh7_2h4l  --sigmas "1.0,0.95455,0.875,0.70,0.50,0.31633,0.0" --high-noise-steps 2 --steps 4  # PAID reference
echo "=== zoom strip ==="
ffmpeg -y -i "$OUT/sh7_4step_zoom.png" -i "$OUT/sh5_4step_zoom.png" -i "$OUT/sh3_4step_zoom.png" -i "$OUT/sh7_2h4l_zoom.png" -filter_complex "hstack=inputs=4" "$OUT/STRIP_grain.png" -loglevel error 2>/dev/null || true
echo "=== done: STRIP_grain.png (sh7-4 | sh5-4 | sh3-4 | sh7-2+4paid) ==="
