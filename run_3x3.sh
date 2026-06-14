#!/usr/bin/env bash
# 3+3 test (6 steps, same +17% as 2+4 but balanced): coherence (seed123 car) + grain (neon street).
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/test3x3"; rm -rf "$OUT"; mkdir -p "$OUT/gcache"
M=/models; W=1280; H=704; FR=13; MAXV=7.3
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
run(){ local tag=$1 seed=$2 P=$3; local D="$OUT/$tag"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e WAN_DISTILL_SIGMAS=1 -e VACE_GRAY_CACHE_DIR=/src/perf_out/test3x3/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --high-noise-steps 3 --steps 3 \
    -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $seed --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/test3x3/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  echo "  $tag rc=$rc $(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1) $(grep -oE 'DMD distilled schedule:[^(]*' "$D/run.log"|tail -1)"
}
CAR="A vintage car rolls to a stop outside a neon-lit corner bar at dusk, headlights sweeping the wet asphalt, tyres easing to a halt. Slow tracking shot alongside the car. Warm amber and magenta neon reflecting on the damp street, cinematic, volumetric light, shallow depth of field, high detail."
NEON="A man walks down a rain-slick neon street at night singing to the camera, reflections shimmering, breath visible in the cold air. Slow steadicam tracking shot. Saturated cyan and magenta neon, cinematic, volumetric light, shallow depth of field, high detail."
echo "=== 3+3 test ==="
run car_3h3l 123 "$CAR"
run neon_3h3l 42 "$NEON"
# coherence: 3x3 car vs base(2h2l) vs 4h2l
ffmpeg -y -i perf_out/coherence/base_2h2l_mid.png -i "$OUT/car_3h3l/f006.png" -i perf_out/coherence/more_4h2l_mid.png -filter_complex "hstack=inputs=3,scale=1500:-1" "$OUT/CAR_2h2l_3h3l_4h2l.png" -loglevel error 2>/dev/null || true
# grain: 3x3 neon zoom vs base vs 2h4l
ffmpeg -y -i "$OUT/neon_3h3l/f006.png" -vf "crop=500:360:380:300,scale=1000:720:flags=neighbor" "$OUT/neon_3h3l_zoom.png" -loglevel error 2>/dev/null || true
ffmpeg -y -i perf_out/speckle_neon/base_2h2l_zoom.png -i "$OUT/neon_3h3l_zoom.png" -i perf_out/speckle_neon/add_2h4l_zoom.png -filter_complex "hstack=inputs=3,scale=1500:-1" "$OUT/NEON_2h2l_3h3l_2h4l.png" -loglevel error 2>/dev/null || true
echo "=== done: CAR_2h2l_3h3l_4h2l.png (coherence) + NEON_2h2l_3h3l_2h4l.png (grain) ==="
