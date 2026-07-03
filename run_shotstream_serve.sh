#!/usr/bin/env bash
# ShotStream WARM INTERACTIVE SERVING LOOP (--serve) on the RTX 3060 (device 0).
#
# Loads umT5 + DiT + VAE ONCE (resident), then reads prompts from stdin — one line = one
# chained shot continuing the rolling history — writing shotNN.mp4 per shot and a stitched
# serve_chain.mp4 at exit. Shot N's 67s VAE decode is pipelined onto a background worker
# (P1) so the prompt is accepted again the moment the DiT finishes.
#
# Usage:
#   ./run_shotstream_serve.sh                      # interactive (type prompts)
#   CHUNKS=3 TAG=eyetest ./run_shotstream_serve.sh <<'EOF'
#   a red fox trotting through a snowy pine forest at dawn
#   the fox pauses and looks toward the camera, snow drifting
#   the fox bounds away between the pines, low winter sun
#   :quit
#   EOF
set -uo pipefail
BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}"     # -ff = ffmpeg present
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
TAG="${TAG:-serve}"; OUT="shotstream_out/$TAG"; mkdir -p "$SRC/$OUT"

DIT=/models/shotstream-1.3b-dit-f16.gguf
VAE=/models/longcat-wan-vae-f16.gguf
UMT5=/models/longcat-umt5-xxl-q8_0.gguf
W=${W:-832}; H=${H:-480}; FPS=${FPS:-16}; SEED=${SEED:-42}
CHUNKS_ARG=(); [ -n "${CHUNKS:-}" ] && CHUNKS_ARG=(--chunks "$CHUNKS")

# Best Ampere recipe (IMPL-NOTES): resident weights + native FA2 + cuDNN attn + generic
# fusions + FFN tiling + monolithic VAE temporal chunk. NO WAN_DIT_F16 / *_F16 levers.
declare -a ENVV=( SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1 LONGCAT_FFN_TILE_TOKENS=4096
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1
  GGML_CUDA_RMS_MOD_FUSE=1 GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0 )
[ -n "${SHOTSTREAM_SYNC_DECODE:-}" ] && ENVV+=( "SHOTSTREAM_SYNC_DECODE=${SHOTSTREAM_SYNC_DECODE}" )
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done

echo ">> ShotStream --serve on device=0 (3060)  tag=$TAG chunks=${CHUNKS:-7}  $(date +%T)"
docker run --rm -i --gpus '"device=0"' "${ENV_FLAGS[@]}" \
  -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
  /src/build/bin/sd-shotstream --serve \
    --dit "$DIT" --t5xxl "$UMT5" --vae "$VAE" \
    --W "$W" --H "$H" --fps "$FPS" --seed "$SEED" \
    --vae-relative-tile-size 0.5x0.5 "${CHUNKS_ARG[@]}" \
    --out "/src/$OUT" 2>&1 | tee "$SRC/$OUT/serve.log"
echo ">> done. outputs in $SRC/$OUT"
