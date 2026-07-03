#!/usr/bin/env bash
# VAE-only decode A/B on the RTX 3060 (device 0). Decodes a saved full-shot latent
# (81 px frames) with different VAE env/tiling, capturing decode time + peak VRAM.
# Cheap harness: no DiT, no umT5 — pure VAE decode (~60s each).
set -uo pipefail
BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}"
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
VAE=/models/longcat-wan-vae-f16.gguf
LAT=${LAT:-/src/shotstream_out/shot00_latent.bin}
OUT="shotstream_out/vae_ab"; mkdir -p "$SRC/$OUT"

BASEFUSE="GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1 GGML_CUDA_RMS_MOD_FUSE=1"

run_one() {
  local tag="$1"; shift
  local rel="$1"; shift
  local extra="$*"
  local env_flags=()
  for kv in $BASEFUSE $extra; do env_flags+=(-e "$kv"); done
  echo "=================================================================="
  echo ">> VAE A/B [$tag] rel=$rel extra='$extra'  $(date +%T)"
  # background VRAM sampler on device 0
  ( while true; do nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 0.3; done ) > "$SRC/$OUT/vram_$tag.txt" &
  local samp=$!
  docker run --rm --gpus '"device=0"' "${env_flags[@]}" \
    -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
    /src/build/bin/sd-shotstream --vae "$VAE" --decode-latent "$LAT" \
      --W 832 --H 480 --vae-relative-tile-size "$rel" --out "/src/$OUT" \
      2>&1 | grep -E "DECODE-LATENT|decode-latent|Tile size|s/it|taking|compute buffer size|cudaMalloc|failed|decoded rgb|VAE Tile" | tail -8
  kill $samp 2>/dev/null; wait $samp 2>/dev/null
  local peak=$(sort -n "$SRC/$OUT/vram_$tag.txt" | tail -1)
  echo ">> [$tag] PEAK VRAM = ${peak} MiB"
}

# 1) current prod: cuDNN conv3d ON, 0.5x0.5, overlap 0.25, temporal monolithic
run_one base       0.5x0.5 "GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0"
# 2) cuDNN conv3d OFF (im2col path) — is cuDNN actually helping on Ampere?
run_one nocudnn    0.5x0.5 "LONGCAT_VAE_TEMPORAL_CHUNK=0"
# 3) zero overlap (no recompute) — how much is overlap costing?
run_one ovl0       0.5x0.5 "GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0 SHOTSTREAM_VAE_OVERLAP=0.0"
# 4) fewer/bigger tiles (0.6) with DiT absent -> headroom
run_one rel06      0.6x0.6 "GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0"
# 5) temporal streaming chunk=1
run_one tchunk1    0.5x0.5 "GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=1"
echo ">> VAE A/B done."
