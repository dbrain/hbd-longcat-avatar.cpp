#!/usr/bin/env bash
# Quality-drop ladder render helper. ONE GPU job at a time.
# Usage: ladder_render.sh <name> <dit_gguf_basename> <frames> <gpute|cpute> [extra sd-cli args...]
#   - gpute  = GPU text-encode (drop --clip-on-cpu)
#   - cpute  = CPU text-encode (--clip-on-cpu, BEST-matching)
# Output webm -> models/_perf/<name>.webm ; VRAM samples -> models/_perf/<name>.vram
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILDER_IMG="${BUILDER_IMG:-longcat-avatar-dev:builder}"

NAME="$1"; DIT="$2"; FRAMES="$3"; TE="$4"; shift 4
EXTRA=("$@")

CONT="longcat-ladder-$NAME"
OUT="models/_perf/${NAME}.webm"
VRAM="models/_perf/${NAME}.vram"

docker rm -f "$CONT" >/dev/null 2>&1 || true

TE_FLAG="--clip-on-cpu"
[ "$TE" = "gpute" ] && TE_FLAG=""

# offload for >85 frames (resident ceiling for Q4_K); resident otherwise
OFFLOAD=""
if [ "$FRAMES" -gt 81 ]; then OFFLOAD="--offload-to-cpu"; fi

# VRAM sampler in background on host
: > "$REPO_DIR/$VRAM"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$REPO_DIR/$VRAM" 2>/dev/null || true; sleep 0.5; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

echo ">> rendering $NAME : dit=$DIT frames=$FRAMES te=$TE offload='${OFFLOAD}' extra='${EXTRA[*]}'"
START=$(date +%s)
docker run --rm --name "$CONT" --gpus all \
  -v "$REPO_DIR:/src" -v "$REPO_DIR/models:/models" -w /src \
  "$BUILDER_IMG" /src/build/bin/sd-cli \
    -M vid_gen -m "models/$DIT" \
    --t5xxl models/longcat-umt5-xxl-q8_0.gguf \
    --vae models/longcat-wan-vae-f16.gguf \
    --audio-vae models/longcat-whisper-v3-encoder-f16.gguf \
    --init-img models/_testinputs/girl_480x832.png \
    --audio models/_testinputs/speech_16k.wav \
    -p "a person talking" --cfg-scale 1.0 --video-frames "$FRAMES" -W 480 -H 832 \
    --steps 8 --diffusion-fa --seed 42 $TE_FLAG --max-vram 9 $OFFLOAD \
    "${EXTRA[@]}" \
    -o "$OUT"
END=$(date +%s)
kill $SAMPLER 2>/dev/null || true
PEAK=$(sort -n "$REPO_DIR/$VRAM" | tail -1)
echo ">> DONE $NAME : wall=$((END-START))s peak_vram=${PEAK}MiB out=$OUT"
