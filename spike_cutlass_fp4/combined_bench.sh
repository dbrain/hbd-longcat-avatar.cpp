#!/usr/bin/env bash
# Combined ALL-ON FP4 bench: cuBLASLt FP4 GEMM + cuDNN attention.
# Foreground only. One GPU job at a time, pinned to 5060 Ti.
set -uo pipefail

REPO=/home/dbrain/dev/flux2.cpp
SSD=/mnt/ssd/models/flux2
IMG=flux2-dev:builder-cudnn
GPU=GPU-bd93e020-65d1-1a5c-ad6c-57c9f655cf45
BIN=/src/build-cudnn/bin/sd-cli
OUT="$REPO/spike_cutlass_fp4/combined_out"
MODELS="$REPO/models"

VAE=full_encoder_small_decoder.safetensors
ENC=Qwen3-8B-UD-Q4_K_XL.gguf
NVFP4=flux-2-klein-9b-OFFICIAL-fp4.gguf
STEPS=4; W=1024; H=1024; CFG=1; SEED=42
PROMPT="a photograph of a red fox in a snowy forest, golden hour"

mkdir -p "$OUT"

# $1=label $2=cublaslt(0/1) $3=cudnn(0/1)
run() {
  local L="$1" CB="$2" CD="$3"
  local png="/out/${L}.png" rl="$OUT/log_${L}.txt"
  docker run --rm --gpus all \
    -e "CUDA_VISIBLE_DEVICES=$GPU" \
    -e "GGML_NVFP4_CUBLASLT=$CB" -e "GGML_CUDNN_ATTN=$CD" \
    -v "$REPO:/src" -v "$MODELS:/models" -v "$SSD:/models_ssd" -v "$OUT:/out" -w /src \
    "$IMG" "$BIN" \
    --diffusion-model "/models_ssd/$NVFP4" --vae "/models/vae/$VAE" --llm "/models/text_encoders/$ENC" \
    --diffusion-fa --mmap \
    -p "$PROMPT" --steps "$STEPS" -W "$W" -H "$H" --cfg-scale "$CFG" --seed "$SEED" -o "$png" -v 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g;s/\r/\n/g' > "$rl"
  parse "$L" "$CB" "$CD" "$rl"
}

parse() {
  local L="$1" CB="$2" CD="$3" rl="$4"
  local te=$(grep -oE 'get_learned_condition completed, taking [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  local samp=$(grep -oE 'sampling completed, taking [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  local vae=$(grep -oE 'decode_first_stage completed, taking [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  # per-step s/it: list all 4, and steady (step 4)
  local sits=$(grep -oE '[0-9]+/[0-9]+ - [0-9.]+s/it' "$rl"|grep -oE '[0-9.]+s/it'|grep -oE '[0-9.]+'|tr '\n' ' ')
  local sit4=$(echo $sits|awk '{print $NF}')
  local pflux=$(grep -oE 'flux compute buffer size: [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  local pvae=$(grep -oE 'vae compute buffer size: [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  local pparams=$(grep -oE 'total params memory size = [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  local total=$(awk -v a=${te:-0} -v b=${samp:-0} -v c=${vae:-0} 'BEGIN{printf "%.2f", a+b+c}')
  echo "RESULT|$L|cublaslt=$CB|cudnn=$CD|te=${te}|sampling=${samp}|vae=${vae}|total_compute=${total}|sit_all=[${sits}]|sit_step4=${sit4}|params_MB=${pparams}|flux_buf_MB=${pflux}|vae_buf_MB=${pvae}"
}

CFGLABEL="$1"
case "$CFGLABEL" in
  allon)    run "$2" 1 1 ;;
  alloff)   run "$2" 0 0 ;;
  cublaslt) run "$2" 1 0 ;;
  cudnn)    run "$2" 0 1 ;;
esac
