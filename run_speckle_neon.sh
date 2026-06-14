#!/usr/bin/env bash
# Faithful speckle confirm: the NEON-STREET shot that showed dots, base(2+2) vs 2+4 low-noise. seed 42.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/speckle_neon"; rm -rf "$OUT"; mkdir -p "$OUT/gcache"
M=/models; W=1280; H=704; FR=13; MAXV=7.3; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A man walks down a rain-slick neon street at night singing to the camera, reflections shimmering, breath visible in the cold air. Slow steadicam tracking shot. Saturated cyan and magenta neon, cinematic, volumetric light, shallow depth of field, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
run(){ local tag=$1; shift; local D="$OUT/$tag"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/speckle_neon/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler \
    "$@" -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/speckle_neon/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  ffmpeg -y -framerate 16 -i "$D/f%03d.png" -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p -movflags +faststart "$OUT/${tag}.mp4" -loglevel error 2>/dev/null
  ffmpeg -y -i "$D/f006.png" -vf "crop=500:360:380:300,scale=1000:720:flags=neighbor" "$OUT/${tag}_zoom.png" -loglevel error 2>/dev/null || true
  echo "  $tag rc=$rc $(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1) peak=$(cat $D/peak.txt 2>/dev/null)MiB $(awk "BEGIN{printf \"%.0fs\",$(date +%s.%N)-$t0}")"
}
echo "=== neon-street speckle A/B (seed $SEED) ==="
run base_2h2l --sigmas "1.0,0.95455,0.875,0.70,0.0"                  --high-noise-steps 2 --steps 2
run add_2h4l  --sigmas "1.0,0.95455,0.875,0.70,0.50,0.31633,0.0"     --high-noise-steps 2 --steps 4
echo "=== done: perf_out/speckle_neon/{base_2h2l,add_2h4l}.mp4 + _zoom.png ==="
