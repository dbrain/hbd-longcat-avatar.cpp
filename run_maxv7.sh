#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=480; H=832; FR="${FR:-21}"; MAXV="${MAXV:-7}"; LABEL="${LABEL:-maxv7}"
rm -rf "$OUT/$LABEL.frames"; mkdir -p "$OUT/$LABEL.frames"
PA="A young man with tousled dark brown hair and light stubble in a faded blue denim jacket sings energetically into the camera, bobbing his head to an upbeat rock song. Locked static medium shot. Warm amber neon glows behind him outside a small corner bar at dusk, cinematic, high detail."
vf="$OUT/.vram_$LABEL"; echo 0 > "$vf"
( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ] && { m=$(cat "$vf"); [ "$u" -gt "$m" ] && echo "$u">"$vf"; }; sleep 0.3; done ) & sp=$!
t0=$(date +%s.%N)
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
  --high-noise-diffusion-model $M/wan22-i2v-a14b-high-q4_k.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
  -p "$PA" --cfg-scale 1 --high-noise-cfg-scale 1 \
  --sampling-method euler --high-noise-sampling-method euler \
  --steps 2 --high-noise-steps 2 --flow-shift 7 \
  -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
  --vae-tiling --vae-relative-tile-size 4x4 \
  --init-img $DR/char.png -o /src/perf_out/$LABEL.frames/f%03d.png -v > "$OUT/$LABEL.log" 2>&1
rc=$?; t1=$(date +%s.%N); kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"); peak=$(cat "$vf"); rm -f "$vf"
echo ">> $LABEL maxv=$MAXV FR=$FR: wall=${wall}s peak=${peak}MiB rc=$rc"
grep -E "get_learned_condition completed|sampling\(high noise\) completed|sampling completed|decode_first_stage completed|generate_video completed|encode_first_stage completed" "$OUT/$LABEL.log"
