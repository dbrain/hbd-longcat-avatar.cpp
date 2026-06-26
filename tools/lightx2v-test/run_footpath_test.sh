#!/bin/bash
# ============================================================================
# LightX2V Wan2.2-NVFP4-Sparse T2V test — "footpath / man walking toward camera"
# 1280x704, ~5s (81 frames @16fps), 4-step NVFP4 + SLA sparse attention.
# Runs the official LightX2V docker image on the RTX 5060 Ti (16GB, Blackwell).
#
# Prereqs (all prepared by the setup session, see README.md):
#   - docker image  lightx2v/lightx2v:26052801-cu130-5090   (docker images | grep lightx2v)
#   - models under  ../../models/lightx2v/{nvfp4-sparse,wan22-base-t2v}/
#   - this dir mounted read-only for the config
#
# Usage:   ./run_footpath_test.sh            # auto-picks the 5060 Ti
#          GPU_NAME=3060 ./run_footpath_test.sh   # force a different card by name
#          PROMPT="..." ./run_footpath_test.sh    # override the prompt
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$(cd "$HERE/../../models/lightx2v" && pwd)"
# The docker image is a runtime base (cuda13 + torch/sage/flash/triton) but does
# NOT contain the lightx2v package itself — we mount the cloned repo as the source.
LIGHTX2V_REPO="$(cd "$HERE/../LightX2V" && pwd)"
OUT="$HERE/out"
mkdir -p "$OUT"

IMAGE="lightx2v/lightx2v:26052801-cu130-5090"
CONFIG_IN_CTR="/cfg/config_t2v_1280x704_footpath.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="footpath_t2v_1280x704_${STAMP}.mp4"

# --- pick the GPU by name (default: the 16GB Blackwell 5060 Ti) ---------------
GPU_NAME="${GPU_NAME:-5060}"
GPU_IDX="$(nvidia-smi --query-gpu=index,name --format=csv,noheader \
            | awk -F', ' -v n="$GPU_NAME" 'index($2,n){print $1; exit}')"
if [ -z "${GPU_IDX:-}" ]; then
  echo "ERROR: no GPU matching name '$GPU_NAME'. Available:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv; exit 1
fi
echo ">> Using GPU index $GPU_IDX ($(nvidia-smi -i "$GPU_IDX" --query-gpu=name --format=csv,noheader))"

# --- prompt ------------------------------------------------------------------
PROMPT="${PROMPT:-A static eye-level camera faces a long straight footpath in a city park. In the far distance a man walks steadily toward the camera, growing larger as he approaches over the course of the shot, then walks past close to the camera and out of frame to one side. Soft overcast daylight, realistic skin and clothing, natural walking gait, gentle ambient motion in background trees, photorealistic, cinematic, shallow depth of field, 35mm look.}"
# LightX2V's shipped Wan-distill negative prompt (known-good for these models):
NEG="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# --- sanity: models present --------------------------------------------------
for f in "nvfp4-sparse/Wan2.2-T2V-A14B_NVFP4_Sparse_high.safetensors" \
         "nvfp4-sparse/Wan2.2-T2V-A14B_NVFP4_Sparse_low.safetensors" \
         "wan22-base-t2v/models_t5_umt5-xxl-enc-bf16.pth" \
         "wan22-base-t2v/Wan2.1_VAE.pth" \
         "wan22-base-t2v/google/umt5-xxl/spiece.model"; do
  [ -e "$MODELS/$f" ] || { echo "ERROR: missing model file: $MODELS/$f  (download not complete?)"; exit 1; }
done
[ -f "$LIGHTX2V_REPO/scripts/base/base.sh" ] || { echo "ERROR: LightX2V clone missing at $LIGHTX2V_REPO (git clone https://github.com/ModelTC/LightX2V)"; exit 1; }

echo ">> Output -> $OUT/$RESULT"
echo ">> Launching container…"

# The image provides the compiled deps (torch cu130, sageattention, spas_sage_attn,
# flash_attn, triton); we mount the LightX2V repo as the package source. base.sh
# REQUIRES lightx2v_path + model_path set before sourcing (else it exit 1's), and it
# sets PYTHONPATH, BF16 dtype, expandable_segments alloc, profiling level 2.
docker run --rm \
  --gpus "device=${GPU_IDX}" \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$MODELS":/models:ro \
  -v "$LIGHTX2V_REPO":/lightx2v \
  -v "$HERE":/cfg:ro \
  -v "$OUT":/out \
  "$IMAGE" \
  bash -lc '
    set -e
    export lightx2v_path=/lightx2v
    export model_path=/models/wan22-base-t2v
    source /lightx2v/scripts/base/base.sh
    cd /lightx2v
    python -m lightx2v.infer \
      --model_cls wan2.2_moe_distill \
      --task t2v \
      --model_path "$model_path" \
      --config_json '"$CONFIG_IN_CTR"' \
      --prompt "'"$PROMPT"'" \
      --negative_prompt "'"$NEG"'" \
      --save_result_path /out/'"$RESULT"'
  ' 2>&1 | tee "$OUT/${RESULT%.mp4}.log"

echo ">> DONE. Result: $OUT/$RESULT"
ls -lh "$OUT/$RESULT" 2>/dev/null || echo ">> (no output file — check the log above)"
