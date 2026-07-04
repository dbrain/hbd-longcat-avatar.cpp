#!/usr/bin/env bash
# gen_i2v_stills.sh — flux.2 start frames for the i2v ablation (busy/distant scenes).
# GPU step (run when the card is free). Reuses the flux2.cpp Klein-9B dev sd-cli
# (same as gen_flux_stills.sh). Writes 1280x704 PNGs into models/ltx2/_inputs/ (so the
# ablation --init-img finds them) AND onto the eye-test page for a look before we animate.
#
#   bash gen_i2v_stills.sh            # both scenes, 2 seeds each
# The neon still (flux_neon_seed7) is kept for scenario 1 (close "make him dance crazy" motion
# test); these add the DISTANT/BUSY start frames scenario 2 (crowd) + 3 (crossing) need.
set -uo pipefail
FREPO="$HOME/dev/flux2.cpp"; BUILDER="flux2-dev:builder"
INP="$HOME/dev/longcat-avatar.cpp/models/ltx2/_inputs"; EYE="$HOME/dev/longcat-avatar-wan22/perf_out/ltx_denoise"
mkdir -p "$INP" "$EYE"
UNET="${UNET:-flux-2-klein-base-9b-Q4_K_M.gguf}"; VAE="${VAE:-full_encoder_small_decoder.safetensors}"; ENC="${ENC:-Qwen3-8B-UD-Q4_K_XL.gguf}"
W=1280; H=704; STEPS="${STEPS:-28}"; CFG="${CFG:-4}"; SEEDS="${SEEDS:-42 7}"

# preflight: don't fight the owner's GPU work (tolerate kobbler /app/api)
busy="$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$1 !~ /\/app\/api/ && $2+0 > 300')"
[ -n "$busy" ] && { echo "[abort] GPU busy — refusing:"; echo "$busy"; exit 9; }

# scene key -> flux image prompt (describe the FIRST frame / starting pose)
S3="A cinematic wide locked-off shot of a busy daytime city street, eye level from the far kerb, 35mm lens. A woman in a red raincoat and blue jeans stands at the edge of a zebra crossing on the left, about to step off the kerb, small in the wide frame at a middle distance, a canvas bag on her shoulder. Cool overcast daylight, wet pavement with pale reflections, blurred traffic waiting at the light, a few pedestrians at the frame edges, muted colours, photorealistic, sharp focus on the woman, shallow depth on the background, cinematic colour grading, high detail, film grain."
S2="A cinematic medium-wide shot of a crowded outdoor night concert, 50mm lens, warm stage light from behind. A young woman with long dark hair in a fringed denim jacket stands at the centre of the frame facing forward, mid-distance, full body visible, about to dance. Behind her a dense crowd of silhouetted onlookers kept soft by shallow depth of field. Amber and gold rim light, cooler blue background wash, dust motes in the beams, photorealistic, sharp focus on the woman, cinematic colour grading, film grain, high detail."

gen() { local key="$1" prompt="$2"; for seed in $SEEDS; do
  local L="i2v_start_${key}_seed${seed}"
  echo ">> $L (${W}x${H} steps=$STEPS cfg=$CFG)"
  docker run --rm --gpus all -v "$FREPO:/src" -v "$FREPO/models:/models" -v "$INP:/out" -w /src "$BUILDER" \
    /src/build/bin/sd-cli --diffusion-model "/models/unet/$UNET" --vae "/models/vae/$VAE" --llm "/models/text_encoders/$ENC" \
    --offload-to-cpu --mmap --diffusion-fa --max-vram 7.5 \
    -p "$prompt" --steps "$STEPS" -W "$W" -H "$H" --cfg-scale "$CFG" --seed "$seed" \
    -o "/out/$L.png" -v > "$INP/$L.fluxlog" 2>&1
  if [ -f "$INP/$L.png" ]; then cp -f "$INP/$L.png" "$EYE/$L.png"; echo "   wrote $L.png"; else echo "   FAILED"; tail -5 "$INP/$L.fluxlog"; fi
done; }

gen crossing "$S3"      # scenario 3 start (distant, locked-off crossing)
gen crowd    "$S2"      # scenario 2 start (dancer centre, crowd behind)
echo "STILLS DONE -> models/ltx2/_inputs/i2v_start_*  (+ eye-test page). Pick one, pass as INIT= to run_ablation.sh."
