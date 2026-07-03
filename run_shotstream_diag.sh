#!/usr/bin/env bash
# ShotStream 2-shot BATCH diagnostic on the RTX 3060 (device 0).
# Short shots (CHUNKS) for cheap isolation renders. TAG names the output subdir.
# Pass extra env via EXTRA_ENV="SHOTSTREAM_NO_CONTEXT=1 ..." to toggle isolation paths.
set -uo pipefail
BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}"
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
TAG="${TAG:-diag}"; OUT="shotstream_out/$TAG"; mkdir -p "$SRC/$OUT"

DIT=/models/shotstream-1.3b-dit-f16.gguf
VAE=/models/longcat-wan-vae-f16.gguf
UMT5=/models/longcat-umt5-xxl-q8_0.gguf
W=${W:-832}; H=${H:-480}; FPS=${FPS:-16}; SEED=${SEED:-42}; SHOTS=${SHOTS:-2}
CHUNKS=${CHUNKS:-3}
PROMPT="${PROMPT:-a red fox trotting through a snowy pine forest at dawn, volumetric morning light, cinematic, photorealistic}"

declare -a ENVV=( SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1 LONGCAT_FFN_TILE_TOKENS=4096
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1
  GGML_CUDA_RMS_MOD_FUSE=1 GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0 )
for kv in ${EXTRA_ENV:-}; do ENVV+=( "$kv" ); done
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done

echo ">> ShotStream DIAG batch device=0  tag=$TAG shots=$SHOTS chunks=$CHUNKS  extra='${EXTRA_ENV:-}'  $(date +%T)"
docker run --rm --gpus '"device=0"' "${ENV_FLAGS[@]}" \
  -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
  /src/build/bin/sd-shotstream \
    --dit "$DIT" --t5xxl "$UMT5" --vae "$VAE" \
    --shots "$SHOTS" --chunks "$CHUNKS" \
    --W "$W" --H "$H" --fps "$FPS" --seed "$SEED" \
    -p "$PROMPT" \
    --vae-relative-tile-size 0.5x0.5 \
    --out "/src/$OUT" 2>&1 | tee "$SRC/$OUT/run.log"
echo ">> done. outputs in $SRC/$OUT"
