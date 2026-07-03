#!/usr/bin/env bash
# Phase-2 SMOKE TEST: ONE t2v segment @1280x704 4-step DMD, NO --init-img (no anime-girl bleed).
# Eyeball the frames: clean? no anime girl? face quality? Gates the long quality run.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/t2v_smoke"; rm -rf "$OUT"; mkdir -p "$OUT"
M=/models; W=1280; H=704; FR="${FR:-13}"; MAXV=7.3; SEED=42; FPS=16
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A man swings a vintage car door open, steps out and sings toward the camera, gesturing to the beat. Handheld medium shot with a slow push-in. Warm amber neon glow from a bar sign at dusk, rain-slick reflective street, cinematic, volumetric light, shallow depth of field, high detail."
t0=$(date +%s.%N)
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
  --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
  --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
  -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
  -o /src/perf_out/t2v_smoke/f%03d.png -v > "$OUT/run.log" 2>&1
rc=$?
echo "rc=$rc wall=$(awk "BEGIN{printf \"%.0f\",$(date +%s.%N)-$t0}")s gen=$(grep -oE 'generate_video completed in [0-9.]+s' "$OUT/run.log"|tail -1)"
ls "$OUT"/f*.png 2>/dev/null | wc -l | xargs echo "frames:"
grep -iE "out of memory|CUDA error" "$OUT/run.log" | tail -2 || true
# stitch to mp4 for motion check
docker run --rm -v "$REPO:/src" -w /src "$BUILDER" bash -lc "ffmpeg -y -framerate $FPS -i /src/perf_out/t2v_smoke/f%03d.png -c:v libx264 -pix_fmt yuv420p /src/perf_out/t2v_smoke/smoke.mp4 -loglevel error" 2>/dev/null && echo "mp4: $OUT/smoke.mp4"
