#!/usr/bin/env bash
# RUN ONCE: generate segment 1 from the CLEAN native Wan2.2 i2v, and BANK everything seg2 needs:
#   - seg1.bin      : the diffusion latent (WAN_SAVE_LATENT) for latent-injected continuation
#   - cont_seg1/    : seg1 pixel frames (for stitching + the pixel tail)
#   - cont_tail/    : last K frames (the --control-video for VACE seg2)
# After this, iter_seg2.sh re-runs seg2 in ~100s with NO seg1 re-render.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"; M="$REPO/models"
B="longcat-avatar-dev:builder-cudnn"; IMG="/models/_drive/flux_neon_seed7.png"
VAE="/models/longcat-wan-vae-f16.gguf"; T5="/models/longcat-umt5-xxl-q8_0.gguf"
W=1280; H=704; FR=25; K=5; SEED=42; OUTDIR="$REPO/tools/lightx2v-test/out"
P1="A person dancing energetically on a neon-lit city street at night, cinematic, photorealistic"
IL=/models/wan22-i2v-a14b-low-q4_k.gguf; IH=/models/wan22-i2v-a14b-high-q4_k.gguf
od="$REPO/cont_seg1"; rm -rf "$od" "$REPO/cont_tail" "$REPO/cont_bank/seg1.bin"; mkdir -p "$od" "$REPO/cont_tail" "$REPO/cont_bank"
echo "=== SEG1 (native i2v, clean) + bank latent ==="
docker run --rm --gpus '"device=1"' -e LONGCAT_FFN_TILE_TOKENS=4096 -e WAN_DISTILL_SIGMAS=1 -e WAN_DISTILL_SHIFT=5 \
  -e WAN_SAVE_LATENT=/src/cont_bank/seg1.bin -v "$REPO:/src" -v "$M:/models" -w /src "$B" \
  stdbuf -oL -eL /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model "$IL" --high-noise-diffusion-model "$IH" --vae "$VAE" --t5xxl "$T5" --init-img "$IMG" -p "$P1" \
  --sampling-method euler_a --high-noise-sampling-method euler_a --eta 1.0 \
  --steps 2 --high-noise-steps 2 --cfg-scale 1.0 --high-noise-cfg-scale 1.0 --flow-shift 5 -s $SEED \
  -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram 12 \
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.5x0.5 -o "/src/cont_seg1/f%03d.png" -v > "$od/log" 2>&1
n=$(ls "$od"/*.png 2>/dev/null|wc -l); lat=$(ls -l cont_bank/seg1.bin 2>/dev/null|awk '{print $5}')
echo "SEG1: $n/$FR frames | latent=${lat:-MISSING}B"
if [ "$n" -ge "$FR" ]; then
  i=0; for f in $(ls "$od"/f*.png|sort|tail -n $K); do cp "$f" "$(printf $REPO/cont_tail/c%03d.png $i)"; i=$((i+1)); done
  ffmpeg -y -framerate 16 -pattern_type glob -i "$od/*.png" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUTDIR/cont_seg1.mp4" -loglevel error 2>/dev/null
  echo "BANKED: cont_bank/seg1.bin + cont_tail/ ($K frames) + cont_seg1.mp4. Now run: bash iter_seg2.sh \"<prompt>\""
else grep -aiE "error|memory|fail|SAVE_LATENT" "$od/log"|grep -av 0x|tail -4; fi
