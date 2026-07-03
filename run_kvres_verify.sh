#!/usr/bin/env bash
# KV-residency verify + folded per-shot-prompt test: full 81f (chunks=7) 4-shot fox story,
# resident KV mode (default), fast 4-tile VAE. Proves: chunks=7 fits <11.5GB, chaining still
# correct, global+per-shot prompts adhere. Device 0 (3060) only.
set -uo pipefail
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
BUILDER=longcat-avatar-dev:builder-cudnn-ff
TAG=kvres; OUT="$SRC/shotstream_out/$TAG"; rm -rf "$OUT"; mkdir -p "$OUT"
GLOBAL="a red fox in a snowy pine forest at dawn, cinematic, photorealistic, golden volumetric light"
P1="the fox trots alertly across open snow between the tall pines"
P2="the fox stops and digs into the snow with its front paws, nose down"
P3="the fox leaps and pounces, front paws up, snow spraying around it"
P4="the fox curls up and rests in the snow beneath a pine tree, tail over its nose"
declare -a ENVV=( "SHOTSTREAM_KV_HOST=${SHOTSTREAM_KV_HOST:-1}"
  SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1 LONGCAT_FFN_TILE_TOKENS=4096
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1
  GGML_CUDA_RMS_MOD_FUSE=1 GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0
  SHOTSTREAM_VAE_OVERLAP=0.15 )
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done

vf="$OUT/.vram"; echo 0 > "$vf"
( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null|head -1)
    [ -n "$u" ] && { m=$(cat "$vf"); [ "$u" -gt "$m" ] && echo "$u">"$vf"; }; sleep 0.4; done ) & SP=$!

t0=$(date +%s)
docker run --rm --gpus '"device=0"' "${ENV_FLAGS[@]}" -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
  /src/build/bin/sd-shotstream \
  --dit /models/shotstream-1.3b-dit-f16.gguf --t5xxl /models/longcat-umt5-xxl-q8_0.gguf --vae /models/longcat-wan-vae-f16.gguf \
  --shots 4 --chunks "${CHUNKS:-6}" --W 832 --H 480 --fps 16 --seed 42 \
  --global-prompt "$GLOBAL" -p "$P1" -p "$P2" -p "$P3" -p "$P4" \
  --vae-relative-tile-size 0.57x0.57 --out /src/shotstream_out/$TAG > "$OUT/run.log" 2>&1
rc=$?; t1=$(date +%s)
kill $SP 2>/dev/null; peak=$(cat "$vf" 2>/dev/null); rm -f "$vf"

# stitch per-shot mp4s -> chain_kvres.mp4 (fall back to muxing frames if needed)
cd "$OUT"
if ls shot0*.mp4 >/dev/null 2>&1; then
  : > list.txt; for f in $(ls shot0*.mp4 | sort); do echo "file '$f'" >> list.txt; done
  docker run --rm -v "$OUT:/o" -w /o --entrypoint ffmpeg "$BUILDER" -v error -f concat -safe 0 -i /o/list.txt -c copy /o/chain_kvres.mp4 2>/dev/null
fi
echo "=== KVRES DONE rc=$rc wall=$((t1-t0))s peak=${peak}MiB (<11500 target) ==="
echo ">> per-shot timing:"
grep -aoE '=== SHOT [0-9]+/[0-9]+|CHUNK\] shot [0-9]+ chunk [0-9]+/[0-9]+ .*|decode.*taking [0-9.]+s|generate_video completed in [0-9.]+s|prefill' "$OUT/run.log" 2>/dev/null | tail -40
echo ">> outputs:"; ls -la "$OUT"/*.mp4 2>/dev/null
echo ">> any OOM/error?"; grep -aiE 'out of memory|alloc.*failed|CUDA error|nan|ERROR' "$OUT/run.log" 2>/dev/null | head -5 || echo "  (none)"
