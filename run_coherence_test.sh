#!/usr/bin/env bash
# COHERENCE test: does MORE HIGH-noise steps fix the "two-halves car" (structure failure)? Reproduce the
# bad montage shot (shot1 prompt, seed 123) at base 2+2 vs 4high+2low vs 4high+4low. High grids bisect the
# high-noise region [1000..500] (off the trained 2-anchor pattern, via --sigmas). shift7 linear sigmas.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/coherence"; rm -rf "$OUT"; mkdir -p "$OUT/gcache"
M=/models; W=1280; H=704; FR=13; MAXV=7.3; SEED=123
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A vintage car rolls to a stop outside a neon-lit corner bar at dusk, headlights sweeping the wet asphalt, tyres easing to a halt. Slow tracking shot alongside the car. Warm amber and magenta neon reflecting on the damp street, cinematic, volumetric light, shallow depth of field, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
run(){ local tag=$1; shift; local D="$OUT/$tag"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/coherence/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler \
    "$@" -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/coherence/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  ffmpeg -y -framerate 16 -i "$D/f%03d.png" -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p "$OUT/${tag}.mp4" -loglevel error 2>/dev/null
  cp "$D/f006.png" "$OUT/${tag}_mid.png" 2>/dev/null
  echo "  $tag rc=$rc $(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1) $(awk "BEGIN{printf \"%.0fs\",$(date +%s.%N)-$t0}")"
}
echo "=== coherence: two-halves car, seed $SEED ==="
run base_2h2l --sigmas "1.0,0.95455,0.875,0.70,0.0"                                  --high-noise-steps 2 --steps 2
run more_4h2l --sigmas "1.0,0.98000,0.95455,0.92105,0.875,0.70,0.0"                  --high-noise-steps 4 --steps 2
run more_4h4l --sigmas "1.0,0.98000,0.95455,0.92105,0.875,0.808,0.70,0.50,0.0"       --high-noise-steps 4 --steps 4
echo "=== strip ==="; ffmpeg -y -i "$OUT/base_2h2l_mid.png" -i "$OUT/more_4h2l_mid.png" -i "$OUT/more_4h4l_mid.png" -filter_complex "hstack=inputs=3,scale=1400:-1" "$OUT/STRIP_coherence.png" -loglevel error 2>/dev/null || true
echo "=== done: STRIP_coherence.png (2h2l | 4h2l | 4h4l), seed 123 ==="
