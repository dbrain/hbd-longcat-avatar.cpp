#!/usr/bin/env bash
# A14B retry with VAE spatial tiling (the init-image encode OOM'd without it at 480x832).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"
OUT="$REPO/mvp_out"; mkdir -p "$OUT/a14b"
W=480; H=832; FR=21; SHIFT=7; M=/models; DR=/models/_drive
BASE=(docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src)

sample_vram() { local f="$1"; echo 0 > "$f"
  while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$u" ] && { m=$(cat "$f"); [ "$u" -gt "$m" ] && echo "$u" > "$f"; }; sleep 0.3; done; }

vf="$OUT/.vram_a14b"; sample_vram "$vf" & sp=$!
t0=$(date +%s.%N)
PA="A young man with tousled dark brown hair and light stubble in a faded blue denim jacket over a white t-shirt sings energetically into the camera, bobbing his head and mouthing the words to an upbeat rock song, shoulders swaying with the beat. Locked static medium shot. Warm amber neon glows behind him outside a small corner bar at dusk, the damp street reflecting the lights, soft bokeh, cinematic, volumetric light, high detail."
"${BASE[@]}" "$BUILDER" /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
  --high-noise-diffusion-model $M/wan22-i2v-a14b-high-q4_k.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
  -p "$PA" --cfg-scale 1 --high-noise-cfg-scale 1 \
  --sampling-method euler --high-noise-sampling-method euler \
  --steps 2 --high-noise-steps 2 --flow-shift $SHIFT \
  -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram 4.5 \
  --vae-tiling --vae-relative-tile-size 4x4 \
  --init-img $DR/char.png -o /src/mvp_out/a14b/f%03d.png -v > "$OUT/a14b.log" 2>&1
rc=$?; t1=$(date +%s.%N)
kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"); peak=$(cat "$vf"); rm -f "$vf"
# rewrite a14b row in results.csv
grep -v '^a14b,' "$OUT/results.csv" > "$OUT/results.csv.tmp" 2>/dev/null; mv "$OUT/results.csv.tmp" "$OUT/results.csv"
echo "a14b,$wall,$peak,$rc" >> "$OUT/results.csv"
echo ">> a14b: ${wall}s  peak ${peak}MiB  rc=$rc"
if [ -n "$(ls "$OUT/a14b"/f*.png 2>/dev/null)" ]; then
  ffmpeg -y -framerate 16 -pattern_type glob -i "$OUT/a14b/*.png" -c:v libx264 -pix_fmt yuv420p "$OUT/a14b.mp4" -loglevel error
fi
cat "$OUT/results.csv"
