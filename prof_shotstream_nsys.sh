#!/usr/bin/env bash
# nsys timeline of a SHORT ShotStream render (1 shot, CHUNKS chunks) on the RTX 3060 (dev0).
# Folds profiling INTO a real render (owner rule). Captures DiT + VAE CUDA kernels, imports
# the qdstrm in-container (libdw1), and prints the per-kernel GPU-time summary.
# Baseline = owner's host-KV mode (SHOTSTREAM_KV_HOST=1) + the fast recipe env.
set -uo pipefail
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
BUILDER=longcat-avatar-dev:builder-cudnn-ff
NVER=2026.2.0
NSYS=/opt/nvidia/nsight-compute/$NVER/host/target-linux-x64/nsys
IMP=/opt/nvidia/nsight-compute/$NVER/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter
TAG="${TAG:-nsys}"; OUT="shotstream_out/$TAG"; mkdir -p "$SRC/$OUT"
CHUNKS="${CHUNKS:-3}"; SHOTS="${SHOTS:-1}"; VAEFLAG=""; [ "${NO_VAE:-0}" = "1" ] && VAEFLAG="--no-vae"
PROMPT="a red fox trotting through a snowy pine forest at dawn, cinematic, photorealistic"

declare -a ENVV=( "SHOTSTREAM_KV_HOST=${SHOTSTREAM_KV_HOST:-1}" SHOTSTREAM_PROFILE=1
  SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1 LONGCAT_FFN_TILE_TOKENS=4096
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1
  GGML_CUDA_RMS_MOD_FUSE=1 GGML_CUDNN_CONV3D=1 LONGCAT_VAE_TEMPORAL_CHUNK=0
  "SHOTSTREAM_VAE_OVERLAP=${SHOTSTREAM_VAE_OVERLAP:-0.25}" )
for kv in ${EXTRA_ENV:-}; do ENVV+=( "$kv" ); done
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done
rm -f "$SRC/$OUT/${TAG}.nsys-rep" "$SRC/$OUT/${TAG}.qdstrm"

echo ">> nsys profile: shots=$SHOTS chunks=$CHUNKS KV_HOST=${SHOTSTREAM_KV_HOST:-1} $(date +%T)"
docker run --rm --gpus '"device=0"' --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
  "${ENV_FLAGS[@]}" -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" bash -lc '
    apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libdw1 libelf1 >/dev/null 2>&1 || true
    '"$NSYS"' profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite true \
      -o /src/'"$OUT"'/'"$TAG"' \
      /src/build/bin/sd-shotstream \
        --dit /models/shotstream-1.3b-dit-f16.gguf --t5xxl /models/longcat-umt5-xxl-q8_0.gguf \
        --vae /models/longcat-wan-vae-f16.gguf \
        --shots '"$SHOTS"' --chunks '"$CHUNKS"' --W 832 --H 480 --fps 16 --seed 42 \
        -p "'"$PROMPT"'" --vae-relative-tile-size 0.57x0.57 '"$VAEFLAG"' \
        --out /src/'"$OUT"' 2>&1
    echo "=== import qdstrm ==="
    if [ -f /src/'"$OUT"'/'"$TAG"'.qdstrm ]; then '"$IMP"' --input-file /src/'"$OUT"'/'"$TAG"'.qdstrm; fi
    echo "=== cuda_gpu_kern_sum ==="
    '"$NSYS"' stats --report cuda_gpu_kern_sum --format table /src/'"$OUT"'/'"$TAG"'.nsys-rep 2>/dev/null | head -50
    echo "=== cuda_api_sum (launch/CPU) ==="
    '"$NSYS"' stats --report cuda_api_sum --format table /src/'"$OUT"'/'"$TAG"'.nsys-rep 2>/dev/null | head -20
  ' 2>&1 | tee "$SRC/$OUT/run.log"
echo ">> done -> $SRC/$OUT/${TAG}.nsys-rep"
