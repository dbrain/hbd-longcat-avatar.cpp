#!/usr/bin/env bash
# Wan2.2-S2V smoke test: ref-image + text-prompt + audio -> directed lip-synced video.
# Base test (NO pose yet) to validate the model runs + lip-syncs at 480x832 on the 3060.
# audioenc = the s2v dit gguf itself (casual_audio_encoder bundled under model.diffusion_model.*).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/s2v"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=${W:-480}; H=${H:-832}; FR=${FR:-21}; LABEL=${LABEL:-s2v}; MAXV=${MAXV:-6.5}
S2V=$M/wan-s2v-14b-dit-dmd-q4_k.gguf
rm -rf "$OUT/$LABEL"; mkdir -p "$OUT/$LABEL"
PI="A young man with tousled dark brown hair and a blue denim jacket sings energetically into the camera, warm amber neon at dusk, cinematic, medium shot."
vf="$OUT/.vram_$LABEL"; echo 0 > "$vf"
( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ] && { m=$(cat "$vf"); [ "$u" -gt "$m" ] && echo "$u">"$vf"; }; sleep 0.3; done ) & sp=$!
t0=$(date +%s.%N)
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src -e S2V_MAX_VRAM_GIB=$MAXV "$BUILDER" \
  stdbuf -oL -eL /src/build/bin/sd-s2v \
  --dit $S2V --audioenc $S2V --wav2vec $M/wav2vec2-xlsr53-f16.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --umt5 $M/longcat-umt5-xxl-q8_0.gguf \
  --ref-image $DR/char.png --wav $DR/song_16k.wav --prompt "$PI" \
  --frames $FR --height $H --width $W --distilled --steps 4 \
  --out /src/perf_out/s2v/$LABEL > "$OUT/$LABEL.log" 2>&1
rc=$?; t1=$(date +%s.%N); kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"); peak=$(cat "$vf"); rm -f "$vf"
echo ">> $LABEL ${W}x${H} ${FR}f: wall=${wall}s peak=${peak}MiB rc=$rc frames=$(ls "$OUT/$LABEL"/*.png 2>/dev/null|wc -l)"
grep -iE "completed, taking|generate|out of memory|ERROR|casual_audio|cond_encoder|lip" "$OUT/$LABEL.log" 2>/dev/null | grep -ivE "Container|WARNING" | tail -10
if [ -n "$(ls "$OUT/$LABEL"/*.png 2>/dev/null)" ]; then
  ffmpeg -y -framerate 16 -pattern_type glob -i "$OUT/$LABEL/*.png" -i "$DR/song_16k.wav" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$OUT/$LABEL.mp4" -loglevel error 2>/dev/null && echo "mp4: $OUT/$LABEL.mp4"
fi
