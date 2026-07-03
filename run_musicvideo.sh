#!/usr/bin/env bash
# FINALE + full-scale validation of the deep-dive fixes: 6-shot music video, chunks=7 (full 81f/shot),
# host KV, fast 4-tile VAE, deep-dive fixes ON (default). Same singer across 6 shots (char-consistency
# stress). Device 0 (3060) only. Proves chunks=7 fits at scale + gives the long fun clip.
set -uo pipefail
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
BUILDER=longcat-avatar-dev:builder-cudnn-ff
TAG=musicvid; OUT="$SRC/shotstream_out/$TAG"; rm -rf "$OUT"; mkdir -p "$OUT"
GLOBAL="a music video, a charismatic young male singer with a leather jacket, neon-lit nighttime city street, cinematic, photorealistic, energetic performance"
P1="the singer pulls up to a bar in his car, a glowing pink neon bar sign overhead, wet pavement reflecting the lights at night"
P2="the singer gets out of the car and pauses on the sidewalk, singing enthusiastically with his mouth wide open and dancing under the neon sign"
P3="the singer pushes the door open and steps inside the bar, the room is empty, dim warm lighting, rows of empty stools"
P4="the singer sings for a moment then sits down on a stool at the wooden bar counter"
P5="the singer grabs a bag of crisps and stuffs a handful into his mouth, cheeks bulging, crumbs on his face"
P6="the singer sings loudly with his mouth full of crisps, crisp crumbs spraying out everywhere, comedic and chaotic"
declare -a ENVV=( SHOTSTREAM_KV_HOST=1 SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1 LONGCAT_FFN_TILE_TOKENS=4096
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1
  GGML_CUDA_RMS_MOD_FUSE=1 GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0 SHOTSTREAM_VAE_OVERLAP=0.15 )
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done

vf="$OUT/.vram"; echo 0 > "$vf"
( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null|head -1)
    [ -n "$u" ] && { m=$(cat "$vf"); [ "$u" -gt "$m" ] && echo "$u">"$vf"; }; sleep 0.4; done ) & SP=$!

t0=$(date +%s)
docker run --rm --gpus '"device=0"' "${ENV_FLAGS[@]}" -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
  /src/build/bin/sd-shotstream \
  --dit /models/shotstream-1.3b-dit-f16.gguf --t5xxl /models/longcat-umt5-xxl-q8_0.gguf --vae /models/longcat-wan-vae-f16.gguf \
  --shots 6 --chunks 7 --W 832 --H 480 --fps 16 --seed 42 \
  --global-prompt "$GLOBAL" -p "$P1" -p "$P2" -p "$P3" -p "$P4" -p "$P5" -p "$P6" \
  --vae-relative-tile-size 0.57x0.57 --out /src/shotstream_out/$TAG > "$OUT/run.log" 2>&1
rc=$?; t1=$(date +%s)
kill $SP 2>/dev/null; peak=$(cat "$vf" 2>/dev/null); rm -f "$vf"

cd "$OUT"
if ls shot0*.mp4 >/dev/null 2>&1; then
  : > list.txt; for f in $(ls shot0*.mp4 | sort); do echo "file '$f'" >> list.txt; done
  docker run --rm -v "$OUT:/o" -w /o --entrypoint ffmpeg "$BUILDER" -v error -f concat -safe 0 -i /o/list.txt -c copy /o/music_video.mp4 2>/dev/null
fi
echo "=== MUSICVID DONE rc=$rc wall=$((t1-t0))s peak=${peak}MiB (chunks=7 full-81f, deep-dive fixes ON) ==="
echo ">> per-shot:"; grep -aoE '=== SHOT [0-9]+/[0-9]+|decode graph completed, taking [0-9.]+s|generate_video completed in [0-9.]+s' "$OUT/run.log" 2>/dev/null | tail -30
echo ">> outputs:"; ls -la "$OUT"/*.mp4 2>/dev/null
echo ">> errors?"; grep -aiE 'out of memory|alloc.*failed|CUDA error|nan|ERROR' "$OUT/run.log" 2>/dev/null | head -5 || echo "  (none)"
