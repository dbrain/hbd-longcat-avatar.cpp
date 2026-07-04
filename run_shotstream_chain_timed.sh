#!/usr/bin/env bash
# Final ShotStream continued-chain timing run on the RTX 3060 (device 0).
# FULL 7-chunk (81-frame) shots. Records per-shot DiT / decode / prefill timing and
# a timestamped per-shot peak-VRAM trace. SHOTS shots for the timing table; the
# eye-test clip is stitched from the FIRST 2 shots only (shot 3+ has a known
# separate per-frame-phase visual bug — timing still valid).
set -uo pipefail
BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}"
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
TAG="${TAG:-chain_timed}"; OUT="shotstream_out/$TAG"; mkdir -p "$SRC/$OUT"
DIT=/models/shotstream-1.3b-dit-f16.gguf
VAE=/models/longcat-wan-vae-f16.gguf
UMT5=/models/longcat-umt5-xxl-q8_0.gguf
W=832; H=480; FPS=16; SEED=42; SHOTS=${SHOTS:-3}; CHUNKS="${CHUNKS:-0}"
CHUNK_FLAG=(); [ "$CHUNKS" != "0" ] && CHUNK_FLAG=(--chunks "$CHUNKS")
PROMPT="a red fox trotting through a snowy pine forest at dawn, volumetric morning light, cinematic, photorealistic"

# Fast recipe: base fusion + cuDNN attn/conv3d + FFN tile + resident DiT + K/V-skip,
# + FAST VAE tiling (0.57x0.57 / overlap 0.15 = 4 tiles, seam-safe, ~39s vs 67s).
# NOTE: VAE param CPU-offload (SHOTSTREAM_VAE_OFFLOAD) is a NET NEGATIVE here — the
# per-encode/decode param streaming leaves ~1 GB of GPU pool residue that pushes the
# next shot's chunk-6 8.55 GB compute-buffer alloc over the edge (hard cudaMalloc OOM).
# VAE weights resident (0.24 GB) is the robust choice.
# Chain uses the CHAIN-SAFE base VAE tiling (0.5x0.5 / overlap 0.25 = small 4.4 GB
# decode buffer) so the at-the-edge chunks=7 chain completes. The FAST VAE tiling
# (0.57x0.57 / overlap 0.15 = 38.7 s, measured in the decode A/B) uses a 5.5 GB
# buffer whose extra fragmentation OOMs shot-1 chunk-6 — deployable in-chain only
# once the DiT compute buffer is reduced (KV residency, next lever). Override via
# VAE_REL / SHOTSTREAM_VAE_OVERLAP.
VAE_REL="${VAE_REL:-0.5x0.5}"
declare -a ENVV=( SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1 LONGCAT_FFN_TILE_TOKENS=4096
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1
  GGML_CUDA_RMS_MOD_FUSE=1 GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0
  "SHOTSTREAM_VAE_OVERLAP=${SHOTSTREAM_VAE_OVERLAP:-0.25}" )
for kv in ${EXTRA_ENV:-}; do ENVV+=( "$kv" ); done
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done

# timestamped VRAM sampler (ms since epoch, MiB used) so peaks attribute per shot
( while true; do printf '%s %s\n' "$(date +%s.%N)" "$(nvidia-smi -i "${DEVICE:-0}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; sleep 0.25; done ) > "$SRC/$OUT/vram.trace" &
SAMP=$!
DEVICE="${DEVICE:-0}"
echo ">> CHAIN TIMED device=$DEVICE shots=$SHOTS FULL-7-chunk  $(date +%T)"
docker run --rm --gpus "\"device=$DEVICE\"" "${ENV_FLAGS[@]}" \
  -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
  /src/build/bin/sd-shotstream \
    --dit "$DIT" --t5xxl "$UMT5" --vae "$VAE" \
    --shots "$SHOTS" "${CHUNK_FLAG[@]}" --W "$W" --H "$H" --fps "$FPS" --seed "$SEED" \
    -p "$PROMPT" --vae-relative-tile-size "$VAE_REL" \
    --out "/src/$OUT" 2>&1 | tee "$SRC/$OUT/run.log"
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
echo ">> overall peak VRAM: $(awk '{print $2}' "$SRC/$OUT/vram.trace" | sort -n | tail -1) MiB"
# 2-shot eye-test clip (shots 0+1 only)
if [ -f "$SRC/$OUT/shot00.mp4" ] && [ -f "$SRC/$OUT/shot01.mp4" ]; then
  printf "file 'shot00.mp4'\nfile 'shot01.mp4'\n" > "$SRC/$OUT/eyetest2.txt"
  docker run --rm -v "$SRC:/src" -w /src "$BUILDER" bash -lc \
    "ffmpeg -y -f concat -safe 0 -i '/src/$OUT/eyetest2.txt' -c copy '/src/$OUT/perf_chain.mp4' 2>/dev/null" || true
  echo ">> 2-shot eye-test clip -> $OUT/perf_chain.mp4"
fi
echo ">> done -> $SRC/$OUT"
