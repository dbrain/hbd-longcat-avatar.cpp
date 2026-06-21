#!/usr/bin/env bash
# cuDNN SDPA flash-attn A/B validation for flux2 (5060 Ti, FP4 recipe).
# Bank a latent once, render MMA baseline (GGML_CUDNN_ATTN=0) and cuDNN (=1) RNG-matched,
# plus a fresh (non-replayed) cuDNN render. Foreground; one GPU job at a time.
set -uo pipefail

REPO=/home/dbrain/dev/flux2.cpp
SSD=/mnt/ssd/models/flux2
IMG=flux2-dev:builder-cudnn
GPU=GPU-bd93e020-65d1-1a5c-ad6c-57c9f655cf45
BIN=/src/build-cudnn/bin/sd-cli
OUT="$REPO/spike_cutlass_fp4/cudnn_out"
MODELS="$REPO/models"

VAE=full_encoder_small_decoder.safetensors
ENC=Qwen3-8B-UD-Q4_K_XL.gguf
NVFP4=flux-2-klein-9b-OFFICIAL-fp4.gguf
STEPS=4; W=1024; H=1024; CFG=1; SEED=42
LATENT=/models_ssd/_cudnn_init_latent.bin
PROMPT="a photograph of a red fox in a snowy forest, golden hour"

mkdir -p "$OUT"

run() {  # $1=label $2=cudnn(0/1) $3=latent_mode(save/init/none) $4=nsys(0/1)
  local L="$1" CD="$2" LM="$3" NS="${4:-0}"
  local png="/out/${L}.png" rl="$OUT/log_${L}.txt"
  local latenv=()
  case "$LM" in
    save) latenv=( -e "FLUX2_SAVE_LATENT=$LATENT" ) ;;
    init) latenv=( -e "FLUX2_INIT_LATENT=$LATENT" ) ;;
    none) latenv=() ;;
  esac
  local cmd=( "$BIN" )
  if [ "$NS" = "1" ]; then
    cmd=( /usr/local/bin/nsys profile --trace=cuda --force-overwrite=true -o "/out/nsys_${L}" "$BIN" )
  fi
  docker run --rm --gpus all \
    -e "CUDA_VISIBLE_DEVICES=$GPU" -e "GGML_CUDNN_ATTN=$CD" "${latenv[@]}" \
    -v "$REPO:/src" -v "$MODELS:/models" -v "$SSD:/models_ssd" -v "$OUT:/out" -w /src \
    "$IMG" "${cmd[@]}" \
    --diffusion-model "/models_ssd/$NVFP4" --vae "/models/vae/$VAE" --llm "/models/text_encoders/$ENC" \
    --diffusion-fa --mmap \
    -p "$PROMPT" --steps "$STEPS" -W "$W" -H "$H" --cfg-scale "$CFG" --seed "$SEED" -o "$png" -v 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g;s/\r/\n/g' > "$rl"
  local sit=$(grep -oE '[0-9]+/[0-9]+ - [0-9.]+s/it' "$rl" | tail -1 | grep -oE '[0-9.]+s/it' | grep -oE '[0-9.]+')
  local samp=$(grep -oE 'sampling completed, taking [0-9.]+' "$rl"|tail -1|grep -oE '[0-9.]+$')
  echo "  [$L] cudnn=$CD warm-s/it=$sit sampling=${samp}s -> $png"
}

echo "==> bank latent"
run bank 0 save 0
rm -f "$OUT/bank.png"
echo "==> baseline MMA (cudnn off, replay)"
run mma 0 init 0
echo "==> cuDNN (replay, RNG-matched)"
run cudnn 1 init 0
echo "==> cuDNN fresh (no replay)"
run cudnn_fresh 1 none 0
echo "DONE"
