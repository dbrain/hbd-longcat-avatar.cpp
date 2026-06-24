#!/usr/bin/env bash
# ltx_ab.sh — ONE-SHOT LTX-2.3-22b video render (longcat-avatar.cpp / sd-cli) for the
# cuDNN-borrow A/B on the 5060 Ti. Foreground, single GPU job at a time.
#
# longcat-avatar.cpp is the prod LTX renderer (kobbler-ltx-video-1 runs the sd-server
# build of this same tree). Here we drive the in-process sd-cli single-segment render
# (no --ltx-chain-segments -> generate_video). Recipe mirrors the prod ENTRYPOINT:
# distilled cfg-free 8-step euler/simple, --offload-to-cpu --mmap --diffusion-fa,
# 4x4 + temporal_tile_frames=4 VAE tiling, --max-vram.
#
# A/B borrow flags (pass via EXTRA_ENV, space-separated KEY=VAL; default = prod, all off):
#   GGML_CUDNN_ATTN=1     cuDNN fused SDPA (LTX self-attn attn1; cross-attn keeps MMA)
#   GGML_CUDNN_CONV3D=1   cuDNN implicit-GEMM 3D conv (LTX VAE CausalConv3d, im2col_3d-free)
#   GGML_NVFP4_CUBLASLT=1 FP4 cuBLASLt GEMM (only for an FP4 gguf, not Q4_K)
#   GGML_CUDNN_TRACE=1    print which cuDNN kernels engage (first 8 calls each)
set -uo pipefail

REPO=/home/dbrain/dev/longcat-avatar.cpp
IMG=flux2-dev:builder-cudnn
BIN=/src/build-cudnn/bin/sd-cli
GPU="${GPU:-GPU-bd93e020-65d1-1a5c-ad6c-57c9f655cf45}"   # RTX 5060 Ti
MODELS="${MODELS:-/home/dbrain/dev/longcat-avatar.cpp/models}"

DIFFUSION_GGUF="${DIFFUSION_GGUF:-ltx2/distilled-1.1/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf}"
VIDEO_VAE="${VIDEO_VAE:-ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors}"
AUDIO_VAE="${AUDIO_VAE:-ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf}"
LLM="${LLM:-ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf}"
CONNECTORS="${CONNECTORS:-ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors}"

SEED="${SEED:-42}"
FRAMES="${FRAMES:-97}"
W="${W:-1280}"
H="${H:-704}"
STEPS="${STEPS:-8}"
CFG="${CFG:-1.0}"
FPS="${FPS:-24}"
MAX_VRAM="${MAX_VRAM:-10.5}"
PROMPT="${PROMPT:-a red fox trotting through a snowy pine forest at golden hour, cinematic, shallow depth of field}"

OUT="${OUT:-$REPO/ltx_ab_out}"
LABEL="${LABEL:-ltx_ab}"
mkdir -p "$OUT"

# Prod recipe env (from kobbler docker-compose.yml ltx-video service). These materially
# affect VRAM peak (esp. LONGCAT_FFN_TILE_TOKENS — tiles the FFN compute buffer) + perf.
# Without them a 192f render peaks ~14GB; with them it matches the 12GB-card prod path.
PROD_ENV=(
  -e LONGCAT_NO_OFFLOAD_PIPELINING=1
  -e LONGCAT_VAE_KEEP_RESIDENT=1
  -e LONGCAT_SHARED_RESIDENT=1
  -e LONGCAT_FFN_TILE_TOKENS=4096
  -e LONGCAT_ENCODE_MAX_VRAM=9.5
  -e LONGCAT_DIT_NO_MMAP=0
)
ENV_FLAGS=("${PROD_ENV[@]}")
for kv in ${EXTRA_ENV:-}; do ENV_FLAGS+=(-e "$kv"); done

# Optional init image (i2v). INIT_IMG = host path; mounted at /init.png.
# CHAIN = ltx-chain-segments count (empty/0 = single generate_video, no chain).
IMG_FLAGS=()
if [ -n "${INIT_IMG:-}" ]; then IMG_FLAGS+=(-v "$INIT_IMG:/init.png:ro"); fi
CLI_IMG=(); [ -n "${INIT_IMG:-}" ] && CLI_IMG=(--init-img /init.png)
CLI_CHAIN=(); [ -n "${CHAIN:-}" ] && [ "${CHAIN}" != "0" ] && CLI_CHAIN=(--ltx-chain-segments "$CHAIN")

WEBM="/out/${LABEL}.webm"
RLOG="$OUT/log_${LABEL}.txt"

echo "=== LTX render: label=$LABEL frames=$FRAMES ${W}x${H} steps=$STEPS seed=$SEED max_vram=$MAX_VRAM ==="
echo "=== gguf=$DIFFUSION_GGUF  EXTRA_ENV='${EXTRA_ENV:-}' ==="

docker run --rm --gpus all \
  -e "CUDA_VISIBLE_DEVICES=$GPU" \
  "${ENV_FLAGS[@]}" \
  "${IMG_FLAGS[@]}" \
  -v "$REPO:/src" \
  -v "$MODELS:/models:ro" \
  -v /mnt/ssd/models:/mnt/ssd/models:ro \
  -v "$OUT:/out" \
  -w /src \
  "$IMG" "$BIN" \
    --mode vid_gen \
    --diffusion-model       "/models/$DIFFUSION_GGUF" \
    --vae                   "/models/$VIDEO_VAE" \
    --audio-vae             "/models/$AUDIO_VAE" \
    --llm                   "/models/$LLM" \
    --embeddings-connectors "/models/$CONNECTORS" \
    -p "$PROMPT" \
    "${CLI_IMG[@]}" "${CLI_CHAIN[@]}" \
    --cfg-scale "$CFG" --sampling-method euler --steps "$STEPS" --scheduler simple \
    -W "$W" -H "$H" --video-frames "$FRAMES" --fps "$FPS" --seed "$SEED" \
    --offload-to-cpu --mmap --diffusion-fa --max-vram "$MAX_VRAM" \
    --vae-tiling --vae-relative-tile-size "${VAE_REL_TILE:-4x4}" --temporal-tiling \
    --extra-tiling-args "temporal_tile_frames=${VAE_TEMPORAL_FRAMES:-4},temporal_tile_overlap=1" \
    -o "$WEBM" -v 2>&1 \
  | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g;s/\r/\n/g' | tee "$RLOG"
EXIT=${PIPESTATUS[0]}

echo ""
echo "=== TIMING (label=$LABEL) ==="
te=$(grep -oE 'get_learned_condition completed, taking [0-9.]+' "$RLOG" | tail -1 | grep -oE '[0-9.]+$')
samp=$(grep -oE 'sampling completed, taking [0-9.]+' "$RLOG" | tail -1 | grep -oE '[0-9.]+$')
vae=$(grep -oE 'decode_first_stage completed, taking [0-9.]+' "$RLOG" | tail -1 | grep -oE '[0-9.]+$')
wall=$(grep -oE 'generate_video completed in [0-9.]+' "$RLOG" | tail -1 | grep -oE '[0-9.]+$')
sits=$(grep -oE '[0-9]+/[0-9]+ - [0-9.]+s/it' "$RLOG" | grep -oE '[0-9.]+s/it' | grep -oE '[0-9.]+' | tr '\n' ' ')
sit_last=$(echo $sits | awk '{print $NF}')
cudnn_attn=$(grep -c 'cudnn' "$RLOG" 2>/dev/null)
echo "RESULT|$LABEL|env='${EXTRA_ENV:-}'|frames=$FRAMES|${W}x${H}|steps=$STEPS|te=${te:-NA}|sampling=${samp:-NA}|vae=${vae:-NA}|wall=${wall:-NA}|sit_last=${sit_last:-NA}|cudnn_log_lines=${cudnn_attn}"
echo "log:   $RLOG"
echo "video: $OUT/${LABEL}.webm"
echo "EXIT=$EXIT"
exit "$EXIT"
