#!/usr/bin/env bash
###############################################################################
# run_parity_nvfp4.sh — closest nvfp4 match to Comfy's REAL scenario-3 t2v graph
#   (comfy-docker/wf_comfy_s3_t2v.json), WITHOUT switching to dev-fp8.
#
# Matches Comfy node-for-node on everything our engine can express:
#   model   = nvfp4-CLEAN-dev050.gguf   (== dev + distill-lora@0.5, folded)  ✓ exact
#   base    = 960x544  (Comfy EmptyLTXVLatentVideo)                          ✓ exact
#   x2      = ltx-2.3-spatial-upscaler-x2-1.1  -> final 1920x1088            ✓ exact
#   base σ  = 8-step 1.0..0.0  (Comfy ManualSigmas 4984)                     ✓ exact
#   refine σ= 3-step 0.85,0.725,0.421875,0.0  (Comfy ManualSigmas 4985)      ✓ exact
#   cfg     = 1.0 ; no NAG ; no detailer  (Comfy graph has none)             ✓ exact
#   frames  = 121 @ 24fps  (Comfy PrimitiveInt 4988 / PrimitiveFloat 4989)   ✓ exact
# UNMATCHABLE on our engine (documented deltas — see PARITY-RECIPE.md):
#   sampler : Comfy base=euler_ancestral_cfg_pp / refine=euler_cfg_pp;
#             ours = plain euler (no cfg_pp / ancestral variants). cfg=1 so
#             cfg_pp is near-inert; ancestral base-noise is the only real gap.
#   decode  : Comfy = spatial 2x2 tiles overlap6, NO temporal tiling.
#             --> VAE_DECODE flags come from TILING-FIX.md (placeholder below).
#
# Env overrides:
#   RES=parity|speed   parity(default)=960x544->1920x1088 (true comfy res, ~2.3x pixels)
#                      speed          =640x352->1280x704  (our current res, speed-neutral)
#   FR=121 FPS=24  SEED=42  DEVICE=1  MAXV=11  DIT=nvfp4-CLEAN-dev050.gguf
#   PROMPT=... (default = scenario-3 t2v, byte-identical to the comfy graph)
#
# CPU/analysis note: this script is prepped for GPU day; do not run during a bench.
###############################################################################
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WT=/home/dbrain/dev/longcat-avatar-ltxdenoise
MAIN=/home/dbrain/dev/longcat-avatar.cpp; LTX2="$MAIN/models/ltx2"
BUILDER="longcat-avatar-dev:builder-cudnn-ff"; BIN=/src/build-cudnn/bin/sd-cli
UPDIR=/ltx2/latent_upscale_models; UPSCALER="${UPSCALER:-ltx-2.3-spatial-upscaler-x2-1.1}"
OUTROOT="$WT/_ablation_out"; mkdir -p "$OUTROOT"

RES="${RES:-parity}"
case "$RES" in
  parity) WBASE=960; HBASE=544 ;;   # -> x2 -> 1920x1088 (Comfy)
  mid)    WBASE=832; HBASE=480 ;;   # -> x2 -> 1664x960 (middle)
  speed)  WBASE=640; HBASE=352 ;;   # -> x2 -> 1280x704  (our current)
  x15)    WBASE=864; HBASE=480 ;;   # -> x1.5 -> 1296x720 (~720p, but 1.84x the DiT base pixels of speed -> better faces; set UPSCALER=ltx-2.3-spatial-upscaler-x1.5-1.0)
  x15hi)  WBASE=1280; HBASE=704 ;;  # -> x1.5 -> 1920x1056 (~1080p from a 1280 base = 1.73x parity's base pixels -> best faces; latent-clean 40x22 -> 60x33; set UPSCALER=...x1.5...)
  *) echo "RES must be parity|mid|speed|x15|x15hi"; exit 2 ;;
esac
FR="${FR:-121}"; FPS="${FPS:-24}"; SEED="${SEED:-42}"; DEVICE="${DEVICE:-1}"; MAXV="${MAXV:-7}"   # LOCKED: MAXV=7 (11 inflates offload partial+prefetch buffers ~8.7GB -> 15GB)
DIT="${DIT:-nvfp4-CLEAN-dev050.gguf}"
TBF="${TBF:-3}"; TBO="${TBO:-2}"; VWT="${VWT:-16}"; VHT="${VHT:-8}"; DF16="${DF16:-1}"   # VAE decode tuning levers
# LOCKED VAE recipe (2026-07-06): 2x2 feathered SPATIAL tiling, overlap 4, F16 decode + offload eviction
# = clean (owner-judged seam-free), fastest (283s), fits 11164MB <=11.5GB cap. Set SPATIAL_TILES="" for old temporal path.
SPATIAL_TILES="${SPATIAL_TILES-2x2}"; SPATIAL_OVERLAP="${SPATIAL_OVERLAP:-4}"
if [ "${NATIVE_FA:-0}" = 1 ]; then CUDNN_ATTN_FLAGS=""; else CUDNN_ATTN_FLAGS="-e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1"; fi   # NATIVE_FA=1 -> deterministic MMA flash-attn

# --- schedules: EXACT from wf_comfy_s3_t2v.json ---
BASE_SIGMAS="1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0"   # node 4984, 8 steps
BASE_STEPS=8
REFINE_SIGMAS="0.85,0.725,0.421875,0.0"                                       # node 4985, 3 steps (0.4219->0.421875)
REFINE_STEPS=3

# --- prompt: scenario-3 t2v, verbatim from the comfy graph (CLIPTextEncode 2483) ---
PROMPT="${PROMPT:-Locked-off wide shot of a pedestrian crossing on a busy daytime city street, shot on a 35mm lens at eye level from the far kerb; the camera does not move. Cool overcast daylight flattens the scene and wet pavement holds pale reflections of the surrounding buildings. A woman in a red raincoat and jeans walks briskly from the left of the frame across the zebra crossing to the right, her gait natural and purposeful, a canvas bag swinging at her shoulder. Behind her, blurred traffic waits at the light and a few other pedestrians drift at the edges of the frame. She stays at a middle distance, small in the wide composition, never dominating the frame. Ambient city sound of idling engines, distant chatter, and footsteps on the crossing.}"

TAG="${TAG:-parity_nvfp4_${RES}}"; od="$OUTROOT/$TAG"; rm -rf "$od"; mkdir -p "$od"
[ -f "$LTX2/$DIT" ] || { echo "[abort] model missing: $LTX2/$DIT (fold dev050 first)"; exit 3; }

PRODENV=( -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks -e LTX_DIT_F16=1
  -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  ${CUDNN_ATTN_FLAGS} -e GGML_CUDNN_CONV3D=1
  -e LONGCAT_NO_OFFLOAD_PIPELINING=0 -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_PREFETCH_POOL=1
  -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=0 -e LONGCAT_FFN_TILE_TOKENS=4096 -e LONGCAT_ENCODE_MAX_VRAM=6.5 -e LONGCAT_DIT_NO_MMAP=0
  -e LTXAV_END_RENDER_RECLAIM=1 -e LTXAV_CHAIN_POOL_TRIM=1 -e LONGCAT_VRAM_BREAKDOWN=1
  -e LTXAV_VAE_LAZY=1 -e LTXAV_DIT_FREE_DURING_DECODE=1   # LOCKED: evict VAE during sampling + DiT during decode (needs KEEP_RESIDENT=0)
  -e LTX_VAE_HEAD_F32=1 -e LTX_VAE_CONV3D_WTILES=$VWT -e LTX_VAE_CONV3D_HTILES=$VHT )
# SPATIAL_TILES=NxM -> comfy-style feathered spatial tiling (all frames, NO temporal chop); else temporal-blend.
if [ -n "${SPATIAL_TILES:-}" ]; then
  PRODENV+=( -e LTX_VAE_SPATIAL_TILES=$SPATIAL_TILES -e LTX_VAE_SPATIAL_OVERLAP=${SPATIAL_OVERLAP:-6} )
else
  PRODENV+=( -e LTX_VAE_TEMPORAL_BLEND=1 -e LTX_VAE_TBLEND_FRAMES=$TBF -e LTX_VAE_TBLEND_OVERLAP=$TBO )
fi
[ "${DF16:-1}" = 1 ] && PRODENV+=(-e LTX_VAE_DECODE_F16=1)   # presence-gated: omit entirely to force F32 decode
[ -n "${XE:-}" ] && PRODENV+=($XE)   # extra -e passthrough for determinism/ablation probes

if [ "${NOHIRES:-0}" = 1 ]; then HIRES=""; else HIRES="--hires --hires-upscaler $UPSCALER --hires-upscalers-dir $UPDIR --hires-steps $REFINE_STEPS --hires-sigmas $REFINE_SIGMAS"; fi

# ===========================================================================
# {{ VAE_DECODE_FLAGS }} — SUPPLIED BY TILING-FIX.md (separate agent).
#   Comfy target = spatial 2x2 tiles, overlap 6, NO temporal tiling.
#   Current prod (temporal_tile_frames=4) is the coherence-risk to replace.
#   Leave the line below until TILING-FIX lands its exact flags, then swap it.
# SPATIAL_TILES routes through _compute directly (feathered spatial path), so --vae-tiling MUST be off.
if [ -n "${SPATIAL_TILES:-}" ]; then VAE_DECODE=""; else VAE_DECODE="--vae-tiling --vae-relative-tile-size 1x1"; fi
# ===========================================================================

echo "=== PARITY nvfp4 ($RES): $DIT ${WBASE}x${HBASE} -> x2  ${FR}f@${FPS}  base=${BASE_STEPS}st refine=${REFINE_STEPS}st ==="
docker run --rm --gpus "\"device=$DEVICE\"" "${PRODENV[@]}" -e "LTX_CUSTOM_SIGMAS=$BASE_SIGMAS" \
  -v "$WT:/src" -v "$LTX2:/ltx2" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src "$BUILDER" \
  stdbuf -oL -eL "$BIN" -M vid_gen \
  --diffusion-model /ltx2/$DIT \
  --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors \
  --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
  --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf \
  --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  --lora-model-dir /ltx2/loras \
  -p "$PROMPT" \
  -W "$WBASE" -H "$HBASE" --video-frames "$FR" --fps "$FPS" \
  --sampling-method ${SAMP:-euler} --steps "$BASE_STEPS" --cfg-scale 1.0 --diffusion-fa \
  $VAE_DECODE \
  $HIRES \
  --offload-to-cpu --mmap --max-vram "$MAXV" -s "$SEED" -v \
  -o /src/_ablation_out/$TAG/out.webm > "$od/log" 2>&1
rc=$?
echo "  rc=$rc  log: $od/log"
[ -f "$od/out.webm" ] && echo "  -> $od/out.webm" || { echo "  [FAIL] no webm"; grep -aiE "out of memory|error|abort|assert" "$od/log"|grep -aviE 0x|tail -3; }
