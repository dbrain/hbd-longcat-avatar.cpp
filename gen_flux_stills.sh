#!/usr/bin/env bash
# Generate FLUX.2-Klein start images for Wan i2v via flux2.cpp sd-cli (flux2-dev builder).
# (The prod :8095 server only exposes a thin worker-iso API; the dev sd-cli is the clean path.)
# Writes 1280x704 PNGs into ti2v5b_out/ so the eye-test page (gen_ti2v5b_eyetest.sh) shows them.
set -uo pipefail
FREPO="$HOME/dev/flux2.cpp"; BUILDER="flux2-dev:builder"
OUTDIR="$HOME/dev/longcat-avatar-wan22/ti2v5b_out"; mkdir -p "$OUTDIR"
UNET="${UNET:-flux-2-klein-base-9b-Q4_K_M.gguf}"
VAE="${VAE:-full_encoder_small_decoder.safetensors}"
ENC="${ENC:-Qwen3-8B-UD-Q4_K_XL.gguf}"
W="${W:-1280}"; H="${H:-704}"; STEPS="${STEPS:-28}"; CFG="${CFG:-4}"
TAG="${TAG:-neon_distant}"; SEEDS="${SEEDS:-42 7 123}"
PROMPT="${PROMPT:-A cinematic wide shot of a young man with dark stubble and tousled brown hair, denim jacket over a white t-shirt, standing in the middle of a rain-slick neon-lit city street at night, about fifteen meters from camera, full body visible, facing the camera. Glowing amber, cyan and magenta neon signs line both sides of the narrow street, their colors reflecting on the wet asphalt. Thin mist drifts through the air, soft volumetric light, shallow depth of field, photorealistic, sharp focus, cinematic color grading, high detail, film grain.}"

for seed in $SEEDS; do
  L="flux_${TAG}_seed${seed}"
  echo ">> $L  (${W}x${H} steps=$STEPS cfg=$CFG)"
  docker run --rm --gpus all -v "$FREPO:/src" -v "$FREPO/models:/models" -v "$OUTDIR:/out" -w /src "$BUILDER" \
    /src/build/bin/sd-cli \
    --diffusion-model "/models/unet/$UNET" \
    --vae "/models/vae/$VAE" \
    --llm "/models/text_encoders/$ENC" \
    --offload-to-cpu --mmap --diffusion-fa --max-vram 7.5 \
    -p "$PROMPT" --steps "$STEPS" -W "$W" -H "$H" --cfg-scale "$CFG" --seed "$seed" \
    -o "/out/$L.png" -v > "$OUTDIR/$L.fluxlog" 2>&1
  rc=$?
  if [ -f "$OUTDIR/$L.png" ]; then echo "   wrote $L.png ($(du -k "$OUTDIR/$L.png"|cut -f1) KB) rc=$rc"; else
    echo "   FAILED rc=$rc"; tail -6 "$OUTDIR/$L.fluxlog"; fi
done
echo "FLUX STILLS DONE"
