#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/offprof"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=1280; H=704; FR="${FR:-13}"; MAXV="${MAXV:-7.3}"; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A young man sings into the camera, neon bar at dusk, cinematic."
D="$OUT/mv${MAXV}_fr${FR}"; rm -rf "$D"; mkdir -p "$D"
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  -e LONGCAT_OFFLOAD_PROFILE=1 -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
  --sampling-method euler --high-noise-sampling-method euler --steps 1 --high-noise-steps 1 \
  --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
  -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P" \
  -o /src/perf_out/offprof/mv${MAXV}_fr${FR}/f%03d.png -v > "$D/run.log" 2>&1
echo "=== OFFLOAD_PROFILE ==="; grep -E "OFFLOAD_PROFILE|graph cut.*merged" "$D/run.log"
echo "=== DiT ==="; grep -iE "sampling.*completed" "$D/run.log"
