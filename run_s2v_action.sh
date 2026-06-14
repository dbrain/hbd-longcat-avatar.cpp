#!/usr/bin/env bash
# Probe S2V's TEXT-DRIVEN ACTION RANGE: does it render locomotion/scene-action from the prompt,
# or just sing in place? Wider frame + explicit action prompts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/s2v"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=${W:-832}; H=${H:-480}; FR=${FR:-49}; LABEL=${LABEL:?}; PROMPT=${PROMPT:?}
S2V=$M/wan-s2v-14b-dit-dmd-q4_k.gguf
rm -rf "$OUT/$LABEL"; mkdir -p "$OUT/$LABEL"
t0=$(date +%s.%N)
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src -e S2V_MAX_VRAM_GIB=6.5 "$BUILDER" \
  stdbuf -oL -eL /src/build/bin/sd-s2v \
  --dit $S2V --audioenc $S2V --wav2vec $M/wav2vec2-xlsr53-f16.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --umt5 $M/longcat-umt5-xxl-q8_0.gguf \
  --ref-image $DR/char.png --wav $DR/song_16k.wav --prompt "$PROMPT" \
  --frames $FR --height $H --width $W --distilled --steps 4 \
  --out /src/perf_out/s2v/$LABEL > "$OUT/$LABEL.log" 2>&1
rc=$?; wall=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N)-$t0}")
echo ">> $LABEL ${W}x${H} ${FR}f: wall=${wall}s rc=$rc frames=$(ls "$OUT/$LABEL"/*.png 2>/dev/null|wc -l)"
ls "$OUT/$LABEL"/*.png >/dev/null 2>&1 && ffmpeg -y -framerate 16 -pattern_type glob -i "$OUT/$LABEL/*.png" "$OUT/$LABEL.mp4" -loglevel error 2>/dev/null
