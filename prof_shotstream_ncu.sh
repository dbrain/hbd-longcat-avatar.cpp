#!/usr/bin/env bash
# ncu roofline of the hottest DiT kernel (causal self-attn flash_attn_ext_f16) on the 3060.
# Runs a 1-shot chunks=1 --no-vae render; ncu instruments only launches [SKIP, SKIP+COUNT)
# so it lands on big-L_k self-attn instances (the rest run un-profiled). KERNEL/ SKIP/ COUNT
# overridable. Needs --cap-add SYS_ADMIN for HW counters (ref_ncu_docker_syadmin).
set -uo pipefail
SRC=/home/dbrain/dev/longcat-avatar-shotstream
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models
BUILDER=longcat-avatar-dev:builder-cudnn-ff
NCU=/usr/local/cuda/bin/ncu
TAG="${TAG:-ncu}"; OUT="shotstream_out/$TAG"; mkdir -p "$SRC/$OUT"
KERNEL="${KERNEL:-flash_attn_ext_f16}"; SKIP="${SKIP:-60}"; COUNT="${COUNT:-6}"
SECTIONS="${SECTIONS:---section SpeedOfLight --section SpeedOfLight_RooflineChart --section Occupancy --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section LaunchStats}"

declare -a ENVV=( SHOTSTREAM_KV_HOST=1 SHOTSTREAM_NO_OFFLOAD=1 GGML_CUDNN_ATTN=1
  LONGCAT_FFN_TILE_TOKENS=4096 GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1
  GGML_CUDA_BIAS_RMS_FUSE=1 GGML_CUDA_RMS_MOD_FUSE=1 )
for kv in ${EXTRA_ENV:-}; do ENVV+=( "$kv" ); done
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done

echo ">> ncu kernel=$KERNEL skip=$SKIP count=$COUNT $(date +%T)"
docker run --rm --gpus '"device=0"' --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
  "${ENV_FLAGS[@]}" -v "$SRC:/src" -v "$MODELS:/models" -w /src "$BUILDER" \
  "$NCU" --target-processes all --kernel-name "regex:$KERNEL" \
      --launch-skip "$SKIP" --launch-count "$COUNT" $SECTIONS \
      --csv --log-file "/src/$OUT/${TAG}.csv" \
  /src/build/bin/sd-shotstream \
    --dit /models/shotstream-1.3b-dit-f16.gguf --t5xxl /models/longcat-umt5-xxl-q8_0.gguf \
    --vae /models/longcat-wan-vae-f16.gguf --no-vae \
    --shots "${SHOTS:-1}" --chunks "${CHUNKS:-1}" --W 832 --H 480 --fps 16 --seed 42 \
    -p "a red fox trotting through a snowy pine forest at dawn, cinematic, photorealistic" \
    --out "/src/$OUT" > "$SRC/$OUT/run.log" 2>&1
echo ">> rc=$? -> $OUT/${TAG}.csv ($(wc -l < "$SRC/$OUT/${TAG}.csv" 2>/dev/null) lines)"
