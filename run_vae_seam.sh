#!/usr/bin/env bash
# VAE decode seam+time check: decode the saved full-shot latent with candidate tile
# configs, dumping frames (f000/f040/f080) per tag for visual seam inspection.
set -uo pipefail
BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}"
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
VAE=/models/longcat-wan-vae-f16.gguf
LAT=/src/shotstream_out/shot00_latent.bin
OUT="shotstream_out/vae_seam"; mkdir -p "$SRC/$OUT"
BASEFUSE="GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1 GGML_CUDA_RMS_MOD_FUSE=1"

run_one() {
  local tag="$1"; local rel="$2"; local ovl="$3"
  local env_flags=()
  for kv in $BASEFUSE GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0 SHOTSTREAM_VAE_OVERLAP=$ovl SHOTSTREAM_DECTAG=$tag; do
    env_flags+=(-e "$kv"); done
  ( while true; do nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 0.3; done ) > "$SRC/$OUT/vram_$tag.txt" &
  local samp=$!
  docker run --rm --gpus '"device=0"' "${env_flags[@]}" \
    -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
    /src/build/bin/sd-shotstream --vae "$VAE" --decode-latent "$LAT" \
      --W 832 --H 480 --vae-relative-tile-size "$rel" --out "/src/$OUT" > "$SRC/$OUT/log_$tag.txt" 2>&1
  kill $samp 2>/dev/null; wait $samp 2>/dev/null
  local peak=$(sort -n "$SRC/$OUT/vram_$tag.txt" | tail -1)
  local dl=$(grep -oE "\[DECODE-LATENT\].*" "$SRC/$OUT/log_$tag.txt" | head -1)
  local nt=$(grep -oE "num tiles : [0-9]+, [0-9]+" "$SRC/$OUT/log_$tag.txt" | head -1)
  echo ">> [$tag] rel=$rel ovl=$ovl  PEAK=${peak}MiB  $nt  $dl"
}
run_one base   0.5x0.5 0.25
run_one ovl0   0.5x0.5 0.0
run_one ovl012 0.5x0.5 0.125
run_one strip  1.0x0.5 0.15
echo ">> seam check done. frames in $SRC/$OUT/"
