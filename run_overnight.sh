#!/bin/bash
# Overnight eye-test clip generation. Produces clips + objective measurements for the
# user's morning review. NO self-grading.
cd /home/dbrain/dev/wan22-infinitetalk
export LONGCAT_NO_FUSED_ROPE=1
VL=models/wan22-vace-fun-a14b-low-q4_k.gguf
VH=models/wan22-vace-fun-a14b-high-q4_k.gguf
VAE=models/longcat-wan-vae-f16.gguf
UMT5=models/longcat-umt5-xxl-q8_0.gguf
ITDIT=models/infinitetalk-14b-q4_k.gguf
W2V=models/chinese-wav2vec2-base-f16.gguf
ts(){ date +%H:%M:%S; }
mkffmpeg(){ ffmpeg -y -framerate ${2:-12} -pattern_type glob -i "$1/*.png" -c:v libx264 -pix_fmt yuv420p "$1/clip.mp4" 2>/dev/null; }

# ============ 1) VACE-Fun 2-segment velocity-continuation test ============
W=256; H=256; FR=25; K=5
COMMON="--vae $VAE --t5xxl $UMT5 --cfg-scale 3.5 --high-noise-cfg-scale 3.5 \
  --sampling-method euler --high-noise-sampling-method euler \
  --steps 4 --high-noise-steps 4 --flow-shift 5 -W $W -H $H --video-frames $FR --diffusion-fa"

echo "[$(ts)] === VACE SEG1 (free gen, motion prompt) ==="
rm -rf it_vc_seg1 it_vc_tail it_vc_seg2 it_vc_stitch; mkdir -p it_vc_seg1
./build/bin/sd-cli -M vid_gen --diffusion-model $VL --high-noise-diffusion-model $VH $COMMON \
  -p "a man in a leather jacket walking forward down an empty city street at dusk, steadicam tracking shot, cinematic" \
  -o "it_vc_seg1/f%03d.png" -v > logs/vc-seg1.log 2>&1
echo "[$(ts)] VACE SEG1 rc=$?"

mkdir -p it_vc_tail
n=0; for f in $(ls it_vc_seg1/f*.png 2>/dev/null | sort | tail -n $K); do cp "$f" "$(printf it_vc_tail/c%03d.png $n)"; n=$((n+1)); done
echo "[$(ts)] tail frames for continuation: $(ls it_vc_tail | wc -l)"

echo "[$(ts)] === VACE SEG2 (continuation from tail, K=$K kept) ==="
mkdir -p it_vc_seg2
VACE_CONT_FRAMES=$K ./build/bin/sd-cli -M vid_gen --diffusion-model $VL --high-noise-diffusion-model $VH $COMMON \
  --control-video it_vc_tail \
  -p "the man continues walking forward down the city street at dusk, steadicam tracking shot, cinematic" \
  -o "it_vc_seg2/f%03d.png" -v > logs/vc-seg2.log 2>&1
echo "[$(ts)] VACE SEG2 rc=$?"

# stitch: seg1 (all) + seg2 frames after the K kept overlap
mkdir -p it_vc_stitch; i=0
for f in $(ls it_vc_seg1/f*.png 2>/dev/null | sort); do cp "$f" "$(printf it_vc_stitch/s%03d.png $i)"; i=$((i+1)); done
SEAM=$i
for f in $(ls it_vc_seg2/f*.png 2>/dev/null | sort | tail -n +$((K+1))); do cp "$f" "$(printf it_vc_stitch/s%03d.png $i)"; i=$((i+1)); done
echo "seam=$SEAM frames=$i" > it_vc_stitch/SEAM.txt
mkffmpeg it_vc_seg1 12; mkffmpeg it_vc_seg2 12; mkffmpeg it_vc_stitch 12
echo "[$(ts)] === VACE stitched: seam at frame $SEAM, total $i ==="

# ============ 2) InfiniteTalk 2-window streaming (velocity carry in audio path) ============
echo "[$(ts)] === INFINITETALK 2-window streaming, full JFK, 320px ==="
IT_SEED=42 ./build/bin/sd-infinitetalk --dit $ITDIT --wav2vec $W2V --vae $VAE --umt5 $UMT5 \
  --image models/lenna_face.jpg --wav models/jfk_speech.wav --prompt "a woman talking, close up portrait, frontal" \
  --frames 45 --height 320 --width 320 --steps 4 --shift 5 --motion-frame 9 --max-windows 2 \
  --fps 25 --distilled --cpu --out it_stream2 > logs/it-stream2.log 2>&1
echo "[$(ts)] INFINITETALK stream rc=$?"

# ============ 3) Fold Wan2.2 MoE distill onto VACE-Fun -> crisp 4-step clip ============
echo "[$(ts)] === DISTILL FOLD onto VACE-Fun (crisp 4-step attempt) ==="
DISTILL=models/dl/wan22-i2v-a14b-moe-distill/loras
SRC=models/dl/wan22-vace-fun-a14b
ok=1
for E in low high; do
  if [ -f models/wan22-vace-fun-a14b-${E}-distill-q4_k.gguf ]; then echo "  ${E} distill exists"; continue; fi
  uv run --with numpy --with gguf python3 tools/convert_wan_dit.py \
     --src $SRC/${E}_noise_model/diffusion_pytorch_model.safetensors \
     --lora $DISTILL/${E}_noise_model_rank64.safetensors \
     --out models/wan22-vace-fun-a14b-${E}-distill-f16.gguf >> logs/vace-distill.log 2>&1 || { echo "[$(ts)] DISTILL_CONVERT_FAIL ${E}"; ok=0; break; }
  ./build/bin/sd-cli -M convert -m models/wan22-vace-fun-a14b-${E}-distill-f16.gguf \
     -o models/wan22-vace-fun-a14b-${E}-distill-q4_k.gguf --type q4_K -v >> logs/vace-distill.log 2>&1 || { echo "[$(ts)] DISTILL_QUANT_FAIL ${E}"; ok=0; break; }
  rm -f models/wan22-vace-fun-a14b-${E}-distill-f16.gguf
done
if [ "$ok" = "1" ] && [ -f models/wan22-vace-fun-a14b-high-distill-q4_k.gguf ]; then
  echo "[$(ts)] === VACE distilled crisp render (4-step, cfg 1) ==="
  rm -rf it_vc_distill; mkdir -p it_vc_distill
  ./build/bin/sd-cli -M vid_gen \
    --diffusion-model models/wan22-vace-fun-a14b-low-distill-q4_k.gguf \
    --high-noise-diffusion-model models/wan22-vace-fun-a14b-high-distill-q4_k.gguf \
    --vae $VAE --t5xxl $UMT5 -p "a man in a leather jacket walking down a neon-lit city street at night, cinematic, tracking shot" \
    --cfg-scale 1 --high-noise-cfg-scale 1 --sampling-method euler --high-noise-sampling-method euler \
    --steps 2 --high-noise-steps 2 --flow-shift 5 -W 256 -H 256 --video-frames 25 --diffusion-fa \
    -o "it_vc_distill/f%03d.png" -v > logs/vc-distill-render.log 2>&1
  echo "[$(ts)] VACE distilled render rc=$?"
  mkffmpeg it_vc_distill 12
fi
df -h . | tail -1
echo "[$(ts)] === ALL OVERNIGHT RENDERS DONE ==="
