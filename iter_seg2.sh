#!/usr/bin/env bash
# FAST ITERATION: VACE second segment continuing from the BANKED seg1 (no seg1 re-render).
# Usage: bash iter_seg2.sh "<seg2 prompt>" [tag] [VACE_STRENGTH]
# Reads cont_bank/seg1.bin (latent) + cont_tail/ (control frames). Stitches seg1+seg2 -> a clip.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"; M="$REPO/models"
B="longcat-avatar-dev:builder-cudnn"
VAE="/models/longcat-wan-vae-f16.gguf"; T5="/models/longcat-umt5-xxl-q8_0.gguf"
W=${W:-1280}; H=${H:-704}; FR=${FR:-25}; K=5; SEED=42; OUTDIR="$REPO/tools/lightx2v-test/out"
P2="${1:-the same person keeps dancing energetically on the neon-lit city street at night, cinematic}"
TAG="${2:-seg2}"; STR="${3:-1.0}"; AGC="${4:-1}"; AGCT="${5:-0.65}"   # AGC=1 de-drifts contrast (FINDINGS-L12)
REF="${6:-/models/_drive/flux_neon_seed7.png}"   # VACE reference image = identity anchor (the seg1 init). "none" to disable.
# QUANT: nvfp4 (default, −26s/seg sampling) or q4_k via QUANT=q4_k env
QUANT="${QUANT:-nvfp4}"
VL=/models/wan22-vace-fun-a14b-low-distillT2V-$QUANT.gguf; VH=/models/wan22-vace-fun-a14b-high-distillT2V-$QUANT.gguf
# Canonical Wan2.2-VACE nvfp4 PRODUCTION env. See WAN-VACE-PRODUCTION.md for the full VRAM story.
# WAN_DIT_F16 (F16 residual stream) is now VACE-safe (wan.hpp: main-stream upcast at the vace add +
# c->F16 cast) — it was i2v-only before. LONGCAT_VAE_TEMPORAL_CHUNK bounds the VAE decode buffer
# (the ceiling: 10.9GB->0.7GB) — the fix --temporal-tiling should trigger but doesn't for Wan.
CUDNN_ATTN_E="-e GGML_CUDNN_ATTN=1"; [ "${CUDNN_OFF:-0}" = 1 ] && CUDNN_ATTN_E=""   # A/B: cuDNN attn scratch scales w/ tokens
FP8_FFN_E="-e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=blocks."; [ "${FP8_OFF:-0}" = 1 ] && FP8_FFN_E=""
# PROD fit levers baked ON (see WAN-VACE-PRODUCTION.md): WAN_DIT_F16 = F16 residual stream (now
# VACE-safe, halves the DiT activation buffer); LONGCAT_VAE_TEMPORAL_CHUNK = the VAE temporal
# streaming that --temporal-tiling should trigger (10.9GB->0.7GB decode buffer). VTILE 0.25x0.25.
NVENV=""; [ "$QUANT" = nvfp4 ] && NVENV="-e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 $FP8_FFN_E $CUDNN_ATTN_E -e WAN_DIT_F16=1 -e LONGCAT_VAE_TEMPORAL_CHUNK=${LONGCAT_VAE_TEMPORAL_CHUNK:-1} -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1"
[ -s cont_bank/seg1.bin ] || { echo "ERROR: cont_bank/seg1.bin missing — run bank_seg1.sh first"; exit 1; }
[ -d cont_tail ] || { echo "ERROR: cont_tail/ missing — run bank_seg1.sh first"; exit 1; }
# Forward any GGML_/LONGCAT_/WAN_/LTX_ debug toggle set in the parent shell into the container
# (same pattern as run_wan22_i2v_nvfp4.sh) — for diagnostics like LONGCAT_VRAM_BREAKDOWN=1.
FWD=""; for v in $(compgen -e | grep -E '^(GGML_|LONGCAT_|WAN_|LTX_)' || true); do
  case " $NVENV " in *" -e $v="*) continue;; esac; FWD="$FWD -e $v=${!v}"; done
od="$REPO/cont_$TAG"; rm -rf "$od"; mkdir -p "$od"
echo "=== SEG2 VACE continuation [$TAG] str=$STR :: $P2 ==="
docker run --rm --gpus '"device=1"' -e WAN_DISTILL_SIGMAS=1 -e WAN_DISTILL_SHIFT=5 \
  -e VACE_SKIP_BLOCKS=0 -e VACE_CONT_FRAMES=$K -e VACE_CONT_LATENT=/src/cont_bank/seg1.bin -e VACE_STRENGTH=$STR -e VACE_STRENGTH_TAIL=${VACE_STRENGTH_TAIL:-0.2} -e VACE_STRENGTH_ANCHOR_FRAMES=${VACE_STRENGTH_ANCHOR_FRAMES:-2} \
  -e VACE_CONT_AGC=$AGC -e VACE_CONT_AGC_TARGET=$AGCT -e LONGCAT_FFN_TILE_TOKENS=${FFNTILE:-4096} ${WAN_VAE_F16:+-e WAN_VAE_F16=$WAN_VAE_F16} $NVENV $FWD \
  -v "$REPO:/src" -v "$M:/models" -w /src "$B" \
  stdbuf -oL -eL /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model "$VL" --high-noise-diffusion-model "$VH" --vae "$VAE" --t5xxl "$T5" --control-video /src/cont_tail -p "$P2" \
  $([ "$REF" != "none" ] && echo "--init-img $REF") \
  --sampling-method euler_a --high-noise-sampling-method euler_a --eta ${ETA:-1.0} \
  --steps 2 --high-noise-steps 2 --cfg-scale 1.0 --high-noise-cfg-scale 1.0 --flow-shift 5 -s $SEED \
  -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram ${MAXV:-10} \
  --vae-tiling --temporal-tiling --vae-relative-tile-size ${VTILE:-0.25x0.25} -o "/src/cont_$TAG/f%03d.png" -v > "$od/log" 2>&1
n=$(ls "$od"/*.png 2>/dev/null|wc -l); inj=$(grep -aciE "VACE_CONT_LATENT|VACE continuation|cont latent" "$od/log" 2>/dev/null)
echo "SEG2: $n/$FR frames | latent-injection-loglines=$inj"
if [ "$n" -ge "$FR" ]; then
  st="$REPO/cont_stitch_$TAG"; rm -rf "$st"; mkdir -p "$st"; i=0
  for f in $(ls cont_seg1/f*.png|sort); do cp "$f" "$(printf $st/s%03d.png $i)"; i=$((i+1)); done
  for f in $(ls "$od"/f*.png|sort|tail -n +$((K+1))); do cp "$f" "$(printf $st/s%03d.png $i)"; i=$((i+1)); done
  ffmpeg -y -framerate 16 -pattern_type glob -i "$st/*.png" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUTDIR/cont_stitch_$TAG.mp4" -loglevel error 2>/dev/null
  ffmpeg -y -framerate 16 -pattern_type glob -i "$od/*.png" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUTDIR/cont_$TAG.mp4" -loglevel error 2>/dev/null
  cp "$od/$(ls "$od"|sort|tail -1)" "/tmp/cont_${TAG}_LAST.png"
  echo "DONE: cont_stitch_$TAG.mp4 (seam at frame $FR) + cont_$TAG.mp4 (seg2 only)"
else grep -aiE "error|memory|fail|mismatch|cont" "$od/log"|grep -av 0x|tail -5; fi
