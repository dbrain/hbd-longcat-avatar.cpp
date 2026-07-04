#!/usr/bin/env bash
# nsys timeline of a SHORT ShotStream render (1 shot, CHUNKS chunks) on the RTX 3060 (dev0).
# Folds profiling INTO a real render (owner rule). Captures DiT + VAE CUDA kernels and prints
# the per-kernel GPU-time summary + the CUDA-API (launch/CPU) summary.
#
# nsys FIX (2026-07-04): the builder image ships only Nsight COMPUTE (ncu); the nsys bundled
# with it (2026.2) has a BROKEN QdstrmImporter (can't convert .qdstrm -> .nsys-rep: "unable to
# retrieve importer version"), which is why every prior nsys run left an unreadable .qdstrm.
# Fix = install the STANDALONE Nsight Systems (apt: cuda-nsight-systems-13-0, from the CUDA repo
# already in the base image); its nsys writes a readable .nsys-rep directly, no importer needed.
# ~30s one-time apt per fresh container (or bake cuda-nsight-systems-13-0 into Dockerfile.cudnn
# to make it instant + permanent).
set -uo pipefail
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
BUILDER=longcat-avatar-dev:builder-cudnn-ff
TAG="${TAG:-nsys}"; OUT="shotstream_out/$TAG"; mkdir -p "$SRC/$OUT"
CHUNKS="${CHUNKS:-3}"; SHOTS="${SHOTS:-1}"; VAEFLAG=""; [ "${NO_VAE:-0}" = "1" ] && VAEFLAG="--no-vae"
# Whole-frame cuDNN decode by default (NO_VAE_TILE=1); set NO_VAE_TILE=0 for the tiled path.
TILEFLAG="--vae-relative-tile-size 0.57x0.57"; [ "${NO_VAE_TILE:-1}" = "1" ] && TILEFLAG="--no-vae-tiling"
PROMPT="a red fox trotting through a snowy pine forest at dawn, cinematic, photorealistic"

declare -a ENVV=( "SHOTSTREAM_KV_HOST=${SHOTSTREAM_KV_HOST:-1}" SHOTSTREAM_PROFILE=1
  SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1 LONGCAT_FFN_TILE_TOKENS=4096
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1
  GGML_CUDA_RMS_MOD_FUSE=1 GGML_CUDNN_CONV3D=1 GGML_CUDNN_CONV=1 WAN_VAE_F16=1
  LONGCAT_VAE_TEMPORAL_CHUNK=0
  "SHOTSTREAM_VAE_OVERLAP=${SHOTSTREAM_VAE_OVERLAP:-0.25}" )
for kv in ${EXTRA_ENV:-}; do ENVV+=( "$kv" ); done
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done
rm -f "$SRC/$OUT/${TAG}.nsys-rep" "$SRC/$OUT/${TAG}.qdstrm"

echo ">> nsys profile: shots=$SHOTS chunks=$CHUNKS KV_HOST=${SHOTSTREAM_KV_HOST:-1} $(date +%T)"
docker run --rm --gpus '"device=0"' --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
  "${ENV_FLAGS[@]}" -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" bash -lc '
    # standalone Nsight Systems (its nsys writes a readable .nsys-rep; install if not baked in)
    NSYS=$(ls /opt/nvidia/nsight-systems/*/target-linux-x64/nsys 2>/dev/null | head -1)
    if [ -z "$NSYS" ]; then
      apt-get update -qq >/dev/null 2>&1
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nsight-systems-13-0 >/dev/null 2>&1
      NSYS=$(ls /opt/nvidia/nsight-systems/*/target-linux-x64/nsys 2>/dev/null | head -1)
    fi
    [ -z "$NSYS" ] && { echo "ERROR: standalone nsys unavailable"; exit 1; }
    echo "nsys: $NSYS ($("$NSYS" --version 2>&1 | head -1))"
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
    "$NSYS" profile --trace=cuda,cudnn,cublas --sample=none --cpuctxsw=none --force-overwrite true \
      -o /src/'"$OUT"'/'"$TAG"' \
      /src/build/bin/sd-shotstream \
        --dit /models/shotstream-1.3b-dit-f16.gguf --t5xxl /models/longcat-umt5-xxl-q8_0.gguf \
        --vae /models/longcat-wan-vae-f16.gguf \
        --shots '"$SHOTS"' --chunks '"$CHUNKS"' --W 832 --H 480 --fps 16 --seed 42 \
        -p "'"$PROMPT"'" '"$TILEFLAG"' '"$VAEFLAG"' \
        --out /src/'"$OUT"' 2>&1
    # --force-export=true: regenerate the .sqlite even if a stale one exists (else stats errors)
    echo "=== cuda_gpu_kern_sum (GPU kernel time) ==="
    "$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format table /src/'"$OUT"'/'"$TAG"'.nsys-rep 2>/dev/null | head -50
    echo "=== cuda_api_sum (CPU-side launch/sync time) ==="
    "$NSYS" stats --force-export=true --report cuda_api_sum --format table /src/'"$OUT"'/'"$TAG"'.nsys-rep 2>/dev/null | head -20
  ' 2>&1 | tee "$SRC/$OUT/run.log"
echo ">> done -> $SRC/$OUT/${TAG}.nsys-rep"
