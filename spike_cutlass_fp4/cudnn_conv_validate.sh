#!/usr/bin/env bash
# cuDNN conv2d A/B validation for the flux2 VAE decode (5060 Ti, FP4 recipe).
# Bank a latent once, render im2col baseline (GGML_CUDNN_CONV unset) and cuDNN-conv
# (GGML_CUDNN_CONV=1) RNG-matched, plus a fresh cuDNN render. Foreground; one GPU job.
set -uo pipefail

REPO=/home/dbrain/dev/flux2.cpp
SSD=/mnt/ssd/models/flux2
IMG=flux2-dev:builder-cudnn
GPU=GPU-bd93e020-65d1-1a5c-ad6c-57c9f655cf45
BIN=/src/build-cudnn/bin/sd-cli
OUT="$REPO/spike_cutlass_fp4/conv_cudnn_out"
MODELS="$REPO/models"

VAE=full_encoder_small_decoder.safetensors
ENC=Qwen3-8B-UD-Q4_K_XL.gguf
NVFP4=flux-2-klein-9b-OFFICIAL-fp4.gguf
STEPS=4; W=1024; H=1024; CFG=1; SEED=42
LATENT=/models_ssd/_cudnn_conv_init_latent.bin
PROMPT="a photograph of a red fox in a snowy forest, golden hour"

mkdir -p "$OUT"

run() {  # $1=label $2=conv(0/1) $3=latent_mode(save/init/none) $4=nsys(0/1)
  local L="$1" CD="$2" LM="$3" NS="${4:-0}"
  local png="/out/${L}.png" rl="$OUT/log_${L}.txt"
  local latenv=()
  case "$LM" in
    save) latenv=( -e "FLUX2_SAVE_LATENT=$LATENT" ) ;;
    init) latenv=( -e "FLUX2_INIT_LATENT=$LATENT" ) ;;
    none) latenv=() ;;
  esac
  local convenv=()
  if [ "$CD" = "1" ]; then convenv=( -e "GGML_CUDNN_CONV=1" -e "GGML_CUDNN_TRACE=1" ); fi
  local cmd=( "$BIN" )
  if [ "$NS" = "1" ]; then
    cmd=( /usr/local/bin/nsys profile --trace=cuda --force-overwrite=true -o "/out/nsys_${L}" "$BIN" )
  fi
  docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -e "CUDA_VISIBLE_DEVICES=$GPU" "${convenv[@]}" "${latenv[@]}" \
    -v "$REPO:/src" -v "$MODELS:/models" -v "$SSD:/models_ssd" -v "$OUT:/out" -w /src \
    "$IMG" "${cmd[@]}" \
    --diffusion-model "/models_ssd/$NVFP4" --vae "/models/vae/$VAE" --llm "/models/text_encoders/$ENC" \
    --diffusion-fa --mmap \
    -p "$PROMPT" --steps "$STEPS" -W "$W" -H "$H" --cfg-scale "$CFG" --seed "$SEED" -o "$png" -v 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g;s/\r/\n/g' > "$rl"
  local sit=$(grep -oE '[0-9]+/[0-9]+ - [0-9.]+s/it' "$rl" | tail -1 | grep -oE '[0-9.]+s/it' | grep -oE '[0-9.]+')
  local samp=$(grep -oE 'sampling completed, taking [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  local vae=$(grep -oiE 'decod[a-z]* (completed|finished|image).*[0-9.]+s|computing vae .*[0-9.]+s' "$rl"|tail -1)
  echo "  [$L] conv=$CD warm-s/it=$sit sampling=${samp}s  $vae -> $png"
}

MODE="${1:-all}"
case "$MODE" in
  all)
    echo "==> bank latent"; run bank 0 save 0; rm -f "$OUT/bank.png"
    echo "==> baseline im2col (conv off, replay)"; run im2col 0 init 0
    echo "==> cuDNN conv (replay, RNG-matched)"; run cudnnconv 1 init 0
    echo "==> cuDNN conv fresh (no replay)"; run cudnnconv_fresh 1 none 0
    ;;
  nsys_off) run im2col_nsys 0 init 1 ;;
  nsys_on)  run cudnnconv_nsys 1 init 1 ;;
  bank)     run bank 0 save 0; rm -f "$OUT/bank.png" ;;
esac
echo "DONE"
