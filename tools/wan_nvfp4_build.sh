#!/usr/bin/env bash
# Build a Wan2.2-I2V-A14B expert NVFP4 gguf via the imatrix path.
#   ./tools/wan_nvfp4_build.sh <high|low>
#
# Stages (per expert):
#   1. bf16 safetensors -> F16 gguf            (host python/uv; skipped if present)
#   2. single-expert imatrix collection         (GPU; loads the F16 in the --diffusion-model
#      slot so collector keys strip to FLAT names that match the flat gguf — the dual-expert
#      run would only match the LOW leg, since imx_strip_dit_prefix doesn't strip high_noise.)
#   3. F16 -> NVFP4 quantize w/ imatrix + Wan skip-list  (CPU; sd-cli -M convert)
#
# Quantize from F16 ONLY (never q4_K). Skip-list keeps modulation/time_projection/head out of
# FP4 (added to the F32 time-embed -> must stay addable); norm*/*embedding*/.bias/.scale are
# auto-skipped by tensor_should_be_converted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"; M="$REPO/models"
BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn}"
LEG="${1:?usage: wan_nvfp4_build.sh <high|low>}"

SRC="$M/dl/distill_models/${LEG}_noise_model/distill_model.safetensors"
F16="$M/wan22-i2v-a14b-${LEG}-f16.gguf"
IMX="$M/wan-dit-${LEG}.imatrix"
NVFP4="$M/wan22-i2v-a14b-${LEG}-nvfp4.gguf"
VAE="$M/longcat-wan-vae-f16.gguf"; UMT5="$M/longcat-umt5-xxl-q8_0.gguf"; IMG="$M/flux_neon_seed7.png"

# calibration render: prod res (per-column act distribution faithful), fewer frames for speed
CW=${CW:-1280}; CH=${CH:-704}; CFR=${CFR:-33}
DOCK=(docker run --rm --gpus '"device=1"' -v "$REPO:/src" -v "$M:/models" -w /src)

echo "== [$LEG] STAGE 1: bf16 -> F16 gguf =="
if [ -s "$F16" ]; then echo "  $F16 exists, skip"; else
  [ -s "$SRC" ] || { echo "MISSING src $SRC" >&2; exit 1; }
  uv run --with numpy --with gguf python3 tools/convert_wan_dit.py --src "$SRC" --out "$F16" || exit 1
fi
ls -la "$F16"

echo "== [$LEG] STAGE 2: single-expert imatrix ($CW x $CH x $CFR) =="
if [ -s "$IMX" ]; then echo "  $IMX exists, skip"; else
  "${DOCK[@]}" "$BUILDER" /src/build/bin/sd-cli -M vid_gen \
    --diffusion-model "/models/$(basename "$F16")" \
    --vae "/models/$(basename "$VAE")" --t5xxl "/models/$(basename "$UMT5")" \
    --cfg-scale 1 --sampling-method euler_a --eta 1.0 \
    --steps 4 --flow-shift 5 -W "$CW" -H "$CH" --video-frames "$CFR" \
    --diffusion-fa --offload-to-cpu --mmap --max-vram 14 \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.5x0.5 \
    --init-img "/models/$(basename "$IMG")" \
    -p "A person dancing with big head and body movements, neon-lit, photorealistic, cinematic." \
    -s 42 --collect-imatrix "/models/$(basename "$IMX")" \
    -o /src/wan_dance_out/calib_${LEG}.png -v 2>&1 | tee "$REPO/wan_dance_out/imatrix_${LEG}.log"
  echo "  imatrix tensors written:"; grep -i "imatrix: wrote\|wrote .* tensors" "$REPO/wan_dance_out/imatrix_${LEG}.log" | tail -2
fi
[ -s "$IMX" ] || { echo "imatrix collection FAILED for $LEG" >&2; exit 1; }
ls -la "$IMX"

echo "== [$LEG] STAGE 3: F16 -> NVFP4 (imatrix + skip-list) =="
"${DOCK[@]}" "$BUILDER" /src/build/bin/sd-cli -M convert \
  -m "/models/$(basename "$F16")" --type nvfp4 \
  --imatrix "/models/$(basename "$IMX")" \
  --tensor-type-rules 'modulation=f32,time_projection=f32,head\.=f32' \
  -o "/models/$(basename "$NVFP4")" -v 2>&1 | tee "$REPO/wan_dance_out/quant_${LEG}.log"
rc=${PIPESTATUS[0]}
echo "  quantize rc=$rc"; ls -la "$NVFP4" 2>/dev/null
exit $rc
