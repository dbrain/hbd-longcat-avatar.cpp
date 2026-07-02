#!/usr/bin/env bash
# chain_long.sh — the GOAL: a clean LONG clip = seg1 (native i2v) + N chained 4s (65f) VACE
# segments at 1280×704, ≤11.5GB, one story prompt per segment. Uses the 2026-07-02 FIT recipe
# (F16-rope + gray-suffix encode + no-offload-pipelining @ max-vram 3) so each 65f segment fits.
# Roll-forward: each segment saves its latent (VACE_SAVE_LATENT); the next segment injects it
# (VACE_CONT_LATENT) + its kept pixel tail as --control-video, dropping the degraded terminal
# frame (VACE_CONT_LATENT_DROP_TAIL). Colour continuity via tools/vace_colour_lock.py (per-scene
# relative mean/std match on the OUTPUT frames — NOT per-segment AGC, which pulses).
#
# Requires bank_seg1.sh already run. Prompts: pass a story as PROMPTS env (newline-separated) or
# edit STORY below. Output: tools/lightx2v-test/out/cont_<OUTTAG>.mp4 (colour-locked stitch).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"; M="$REPO/models"
B="longcat-avatar-dev:builder-cudnn"
VAE="/models/longcat-wan-vae-f16.gguf"; T5="/models/longcat-umt5-xxl-q8_0.gguf"
W=${W:-1280}; H=${H:-704}; FR=${FR:-65}; K=5; SEED=42; OUTDIR="$REPO/tools/lightx2v-test/out"
DISCARD=${DISCARD:-4}; DROPLAT=${DROPLAT:-1}          # drop 1 degraded latent frame (4 px) per segment
TAIL=${VACE_STRENGTH_TAIL:-}                          # empty = scalar VACE_STRENGTH (continuation striping is a non-issue)
ANCHOR=${VACE_STRENGTH_ANCHOR_FRAMES:-2}
AGC=${AGC:-0}; AGCT=${AGCT:-0.65}                     # AGC OFF by default — colour-lock (post) handles continuity, no pulse
MAXV=${MAXV:-3}; VTILE=${VTILE:-0.25x0.25}
GPU_FREE_GATE=${GPU_FREE_GATE:-4200}; RETRIES=${RETRIES:-6}
QUANT="${QUANT:-nvfp4}"; REF="${REF:-/models/_drive/flux_neon_seed7.png}"
OUTTAG="${OUTTAG:-story_1280}"; CONTRAST=${CONTRAST:-0.85}
VL=/models/wan22-vace-fun-a14b-low-distillT2V-$QUANT.gguf; VH=/models/wan22-vace-fun-a14b-high-distillT2V-$QUANT.gguf
# FIT recipe env (2026-07-02): F16-rope + no-offload-pipelining. gray-suffix encode is default-on.
NVENV="-e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=blocks. -e GGML_CUDNN_ATTN=1 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1 -e WAN_DIT_F16=1 -e WAN_ROPE_F16=1 -e LONGCAT_VAE_TEMPORAL_CHUNK=1 -e LONGCAT_NO_OFFLOAD_PIPELINING=1 -e LONGCAT_FFN_TILE_TOKENS=4096"

[ -s cont_bank/seg1.bin ] || { echo "ERROR: cont_bank/seg1.bin missing — run bank_seg1.sh"; exit 1; }
[ -d cont_tail ] && [ -d cont_seg1 ] || { echo "ERROR: cont_tail/ or cont_seg1/ missing — run bank_seg1.sh"; exit 1; }

# A short 3-beat story (one prompt per 4s segment). Override with PROMPTS env (newline-separated).
STORY_DEFAULT="the same person walks forward slowly along the neon-lit city street at night, looking around, cinematic, photorealistic
the same person stops and turns to face the camera on the neon-lit city street at night, gentle motion, cinematic, photorealistic
the same person raises a hand and waves at the camera on the neon-lit city street at night, warm smile, cinematic, photorealistic"
mapfile -t PROMPTS < <(printf '%s\n' "${PROMPTS:-$STORY_DEFAULT}")
N=${N:-${#PROMPTS[@]}}
echo "chain_long: $N segments x ${FR}f @ ${W}x${H}, FIT recipe (mv$MAXV, no-pipeline, F16-rope, gray-suffix)."

wait_gpu () { local w=0; while :; do local u; u=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null||echo 0)
  [ "${u:-99999}" -lt "$GPU_FREE_GATE" ] && break; [ $((w%60)) -eq 0 ] && echo "  ...waiting GPU-1 (${u}MiB>=$GPU_FREE_GATE)"; sleep 15; w=$((w+15)); done; }

# Per-segment OUTPUT dirs (for the colour-lock post-pass) — seg1 first, then each VACE segment.
SEGDIRS=("$REPO/cont_seg1")
PRIOR_LATENT="/src/cont_bank/seg1.bin"; PRIOR_TAIL_DIR="$REPO/cont_tail"; PRIOR_DROP=0

for ((i=0; i<N; i++)); do
  P="${PROMPTS[$i]}"; SEG=$((i+2)); TAG="s$(printf %02d $SEG)"
  od="$REPO/cont_$OUTTAG-$TAG"; rm -rf "$od"; mkdir -p "$od"; odc="/src/cont_$OUTTAG-$TAG"
  LATOUT="/src/cont_bank/$OUTTAG-$TAG.bin"
  echo "=== SEG$SEG [$TAG] $((i+1))/$N drop_prior=$PRIOR_DROP :: ${P:0:64}... ==="
  n=0
  for ((try=1; try<=RETRIES; try++)); do
    wait_gpu; rm -f "$od"/f*.png
    docker run --rm --gpus '"device=1"' -e WAN_DISTILL_SIGMAS=1 -e WAN_DISTILL_SHIFT=5 \
      -e VACE_CONT_FRAMES=$K -e VACE_CONT_LATENT="$PRIOR_LATENT" -e VACE_CONT_LATENT_DROP_TAIL=$PRIOR_DROP \
      -e VACE_STRENGTH_TAIL="$TAIL" -e VACE_STRENGTH_ANCHOR_FRAMES=$ANCHOR \
      -e VACE_CONT_AGC=$AGC -e VACE_CONT_AGC_TARGET=$AGCT -e VACE_SAVE_LATENT="$LATOUT" $NVENV \
      -v "$REPO:/src" -v "$M:/models" -v "$PRIOR_TAIL_DIR:/priortail:ro" -w /src "$B" \
      stdbuf -oL -eL /src/build/bin/sd-cli -M vid_gen \
      --diffusion-model "$VL" --high-noise-diffusion-model "$VH" --vae "$VAE" --t5xxl "$T5" --control-video /priortail -p "$P" \
      --init-img "$REF" --sampling-method euler_a --high-noise-sampling-method euler_a --eta ${ETA:-1.0} \
      --steps 2 --high-noise-steps 2 --cfg-scale 1.0 --high-noise-cfg-scale 1.0 --flow-shift 5 -s $SEED \
      -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
      --vae-tiling --temporal-tiling --vae-relative-tile-size $VTILE -o "$odc/f%03d.png" -v > "$od/log" 2>&1
    n=$(ls "$od"/f*.png 2>/dev/null|wc -l); [ "$n" -ge "$FR" ] && break
    if grep -qaiE "out of memory|CUDA error|alloc.*failed" "$od/log"; then echo "  SEG$SEG try$try OOM — retry"; sleep 20; continue; fi
    echo "SEG$SEG FAILED (non-OOM): $n/$FR"; grep -aiE "error|fail|mismatch|not implemented" "$od/log"|grep -av 0x|tail -6; exit 2
  done
  [ "$n" -lt "$FR" ] && { echo "SEG$SEG FAILED after $RETRIES: $n/$FR"; exit 2; }
  keep=$((n - DISCARD)); echo "SEG$SEG: $n frames, keep $keep."
  SEGDIRS+=("$od")
  # rebuild next window's pixel tail = last K of KEPT frames; roll the latent + drop the degraded tail.
  ntdir="$REPO/cont_${OUTTAG}_tail-$TAG"; rm -rf "$ntdir"; mkdir -p "$ntdir"
  j=0; for f in $(ls "$od"/f*.png|sort|head -n $keep|tail -n $K); do cp "$f" "$(printf $ntdir/c%03d.png $j)"; j=$((j+1)); done
  PRIOR_LATENT="$LATOUT"; PRIOR_TAIL_DIR="$ntdir"; PRIOR_DROP=$DROPLAT
done

# Colour-lock: one scene (the whole story is one continuous scene here). seg1 = reference (verbatim);
# each VACE segment matched to the previous corrected tail. skip-head=K drops the overlap frames.
echo "=== colour-lock ($N+1 segment dirs, contrast=$CONTRAST) ==="
CL="$REPO/cont_${OUTTAG}_colourlocked"; rm -rf "$CL"
CLARGS=(); for d in "${SEGDIRS[@]}"; do CLARGS+=(--seg "$d"); done
python3 tools/vace_colour_lock.py --out "$CL" --contrast "$CONTRAST" --tail $K --skip-head $K "${CLARGS[@]}" \
  || { echo "colour-lock failed — stitching raw instead"; CL=""; }

# Stitch: colour-locked frames if available, else raw seg dirs (seg1 full + each VACE seg minus overlap head + discard tail).
ST="$REPO/cont_stitch_$OUTTAG"; rm -rf "$ST"; mkdir -p "$ST"; gi=0
if [ -n "$CL" ] && [ -d "$CL" ]; then
  for f in $(ls "$CL"/*.png 2>/dev/null|sort); do cp "$f" "$(printf $ST/s%05d.png $gi)"; gi=$((gi+1)); done
else
  for f in $(ls cont_seg1/f*.png|sort); do cp "$f" "$(printf $ST/s%05d.png $gi)"; gi=$((gi+1)); done
  for ((i=0;i<N;i++)); do SEG=$((i+2)); od="$REPO/cont_$OUTTAG-s$(printf %02d $SEG)"; keep=$(( $(ls "$od"/f*.png|wc -l) - DISCARD ))
    for f in $(ls "$od"/f*.png|sort|head -n $keep|tail -n +$((K+1))); do cp "$f" "$(printf $ST/s%05d.png $gi)"; gi=$((gi+1)); done; done
fi
ffmpeg -y -framerate 16 -pattern_type glob -i "$ST/*.png" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUTDIR/cont_$OUTTAG.mp4" -loglevel error 2>/dev/null
echo "DONE: $OUTDIR/cont_$OUTTAG.mp4 ($gi frames, ~$(echo "scale=1;$gi/16"|bc)s @16fps)"
