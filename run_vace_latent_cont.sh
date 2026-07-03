#!/bin/bash
# CLEAN A/B: VACE continuation, pixel re-encode vs latent backdoor, from the SAME seg1.
# One seg1 (banks its diffusion latent), then TWO seg2 renders sharing the same tail + seed:
#   seg2-pixel : kept context = re-encoded seg1 tail pixels (lossy VAE roundtrip)
#   seg2-latent: kept context = seg1's exact diffusion latents injected (VACE_CONT_LATENT)
# Only the kept-frame source differs, so any seam/quality delta isolates the roundtrip cost.
set -u
cd /home/dbrain/dev/wan22-infinitetalk
export LONGCAT_NO_FUSED_ROPE=1
ts(){ date +%H:%M:%S; }
mkffmpeg(){ ffmpeg -y -framerate ${2:-12} -pattern_type glob -i "$1/*.png" -c:v libx264 -pix_fmt yuv420p "$1/clip.mp4" 2>/dev/null; }

# wait for the main overnight batch to clear the CPU/RAM (each sd-cli loads ~20GB)
while pgrep -f run_overnight.sh >/dev/null 2>&1; do sleep 60; done
sleep 10

VL=models/wan22-vace-fun-a14b-low-q4_k.gguf
VH=models/wan22-vace-fun-a14b-high-q4_k.gguf
VAE=models/longcat-wan-vae-f16.gguf
UMT5=models/longcat-umt5-xxl-q8_0.gguf
W=256; H=256; FR=25; K=5; SEED=42
COMMON="--vae $VAE --t5xxl $UMT5 --cfg-scale 3.5 --high-noise-cfg-scale 3.5 \
  --sampling-method euler --high-noise-sampling-method euler \
  --steps 4 --high-noise-steps 4 --flow-shift 5 -W $W -H $H --video-frames $FR --diffusion-fa -s $SEED"
P1="a man in a leather jacket walking forward down an empty city street at dusk, steadicam tracking shot, cinematic"
P2="the man continues walking forward down the city street at dusk, steadicam tracking shot, cinematic"
LAT=it_vcl_seg1.bin

echo "[$(ts)] === SEG1 (free gen, seed $SEED, banking latent) ==="
rm -rf it_vcl_seg1 it_vcl_tail it_vcl_seg2p it_vcl_seg2l it_vcl_stitch_pixel it_vcl_stitch_latent "$LAT"
mkdir -p it_vcl_seg1
VACE_SAVE_LATENT=$LAT ./build/bin/sd-cli -M vid_gen --diffusion-model $VL --high-noise-diffusion-model $VH $COMMON \
  -p "$P1" -o "it_vcl_seg1/f%03d.png" -v > logs/vcl-seg1.log 2>&1
echo "[$(ts)] SEG1 rc=$? latent=$(ls -l "$LAT" 2>/dev/null | awk '{print $5}')B"

mkdir -p it_vcl_tail
n=0; for f in $(ls it_vcl_seg1/f*.png 2>/dev/null | sort | tail -n $K); do cp "$f" "$(printf it_vcl_tail/c%03d.png $n)"; n=$((n+1)); done

echo "[$(ts)] === SEG2-PIXEL (re-encode tail, seed $SEED) ==="
mkdir -p it_vcl_seg2p
VACE_CONT_FRAMES=$K ./build/bin/sd-cli -M vid_gen --diffusion-model $VL --high-noise-diffusion-model $VH $COMMON \
  --control-video it_vcl_tail -p "$P2" -o "it_vcl_seg2p/f%03d.png" -v > logs/vcl-seg2-pixel.log 2>&1
echo "[$(ts)] SEG2-PIXEL rc=$?"

echo "[$(ts)] === SEG2-LATENT (inject seg1 latent, seed $SEED) ==="
mkdir -p it_vcl_seg2l
VACE_CONT_FRAMES=$K VACE_CONT_LATENT=$LAT ./build/bin/sd-cli -M vid_gen --diffusion-model $VL --high-noise-diffusion-model $VH $COMMON \
  --control-video it_vcl_tail -p "$P2" -o "it_vcl_seg2l/f%03d.png" -v > logs/vcl-seg2-latent.log 2>&1
echo "[$(ts)] SEG2-LATENT rc=$?"
grep -E "VACE_CONT_LATENT|VACE continuation" logs/vcl-seg2-latent.log || echo "[WARN] no injection log line"

# two stitches: seg1 + each seg2 after the K kept overlap
for variant in pixel latent; do
  src=it_vcl_seg2p; [ "$variant" = latent ] && src=it_vcl_seg2l
  out=it_vcl_stitch_$variant; mkdir -p "$out"; i=0
  for f in $(ls it_vcl_seg1/f*.png 2>/dev/null | sort); do cp "$f" "$(printf $out/s%03d.png $i)"; i=$((i+1)); done
  SEAM=$i
  for f in $(ls $src/f*.png 2>/dev/null | sort | tail -n +$((K+1))); do cp "$f" "$(printf $out/s%03d.png $i)"; i=$((i+1)); done
  echo "seam=$SEAM frames=$i variant=$variant" > "$out/SEAM.txt"
  mkffmpeg "$out" 12
done
echo "[$(ts)] === A/B DONE: it_vcl_stitch_pixel vs it_vcl_stitch_latent (seam at frame $FR) ==="
