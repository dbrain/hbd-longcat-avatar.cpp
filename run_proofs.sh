#!/bin/bash
cd /home/dbrain/dev/wan22-infinitetalk
export LONGCAT_NO_FUSED_ROPE=1   # ROPE_PE is CUDA-only; CPU needs the decomposed rope. sd-cli does NOT set this itself.
DIT=models/infinitetalk-14b-q4_k.gguf; W2V=models/chinese-wav2vec2-base-f16.gguf
VAE=models/longcat-wan-vae-f16.gguf; UMT5=models/longcat-umt5-xxl-q8_0.gguf
IMG=models/ref_singer.png; WAV=models/dwarf_3s.wav
CLIP=models/dl/wan21-i2v-14b-480p/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
SRC=models/dl/wan22-vace-fun-a14b
ts(){ date +%H:%M:%S; }

echo "[$(ts)] >>> RUNG 3: A14B MoE (independent motion discriminator)"
./build/bin/sd-cli -M vid_gen \
  --diffusion-model models/wan22-i2v-a14b-low-q4_k.gguf \
  --high-noise-diffusion-model models/wan22-i2v-a14b-high-q4_k.gguf \
  --vae $VAE --t5xxl $UMT5 -p "a person singing, portrait, detailed" \
  --cfg-scale 1 --high-noise-cfg-scale 1 --sampling-method euler --high-noise-sampling-method euler \
  --steps 2 --high-noise-steps 2 --flow-shift 5 -W 256 -H 256 --video-frames 13 --diffusion-fa \
  --init-img $IMG -o it_a14b/clip.png -v > logs/a14b-mvp.log 2>&1
echo "[$(ts)] >>> RUNG3_DONE rc=$?"

echo "[$(ts)] >>> RUNG 2: InfiniteTalk + CLIP-H"
IT_SEED=42 ./build/bin/sd-infinitetalk --dit $DIT --wav2vec $W2V --vae $VAE --umt5 $UMT5 \
  --clip-vision "$CLIP" --image $IMG --wav $WAV --prompt "a person singing, close up portrait" \
  --frames 21 --height 256 --width 256 --steps 4 --motion-frame 1 --max-windows 1 \
  --distilled --cpu --out it_mvp_clip > logs/it-mvp-clip.log 2>&1
echo "[$(ts)] >>> RUNG2_DONE rc=$?"

echo "[$(ts)] >>> RUNG 4: InfiniteTalk streaming (2 windows)"
IT_SEED=42 ./build/bin/sd-infinitetalk --dit $DIT --wav2vec $W2V --vae $VAE --umt5 $UMT5 \
  --image $IMG --wav $WAV --prompt "a person singing, close up portrait" \
  --frames 21 --height 256 --width 256 --steps 4 --motion-frame 5 --max-windows 2 \
  --distilled --cpu --out it_mvp_stream > logs/it-mvp-stream.log 2>&1
echo "[$(ts)] >>> RUNG4_DONE rc=$?"

echo "[$(ts)] >>> RUNG 5a: convert VACE-Fun experts"
for E in low high; do
  [ -f models/wan22-vace-fun-a14b-${E}-q4_k.gguf ] && { echo "  ${E} exists"; continue; }
  uv run --with numpy --with gguf python3 tools/convert_wan_dit.py \
     --src $SRC/${E}_noise_model/diffusion_pytorch_model.safetensors \
     --out models/wan22-vace-fun-a14b-${E}-f16.gguf >> logs/vace-mvp.log 2>&1 || { echo "[$(ts)] >>> VACE_CONVERT_FAIL ${E}"; break; }
  ./build/bin/sd-cli -M convert -m models/wan22-vace-fun-a14b-${E}-f16.gguf \
     -o models/wan22-vace-fun-a14b-${E}-q4_k.gguf --type q4_K -v >> logs/vace-mvp.log 2>&1 || { echo "[$(ts)] >>> VACE_QUANT_FAIL ${E}"; break; }
  rm -f models/wan22-vace-fun-a14b-${E}-f16.gguf
done
echo "[$(ts)] >>> RUNG 5b: VACE render"
if [ -f models/wan22-vace-fun-a14b-high-q4_k.gguf ]; then
./build/bin/sd-cli -M vid_gen \
  --diffusion-model models/wan22-vace-fun-a14b-low-q4_k.gguf \
  --high-noise-diffusion-model models/wan22-vace-fun-a14b-high-q4_k.gguf \
  --vae $VAE --t5xxl $UMT5 -p "a person singing on a stage, portrait, detailed" \
  --cfg-scale 3.5 --high-noise-cfg-scale 3.5 --sampling-method euler --high-noise-sampling-method euler \
  --steps 2 --high-noise-steps 2 --flow-shift 5 -W 256 -H 256 --video-frames 5 --diffusion-fa \
  -o it_vace/clip.png -v >> logs/vace-render.log 2>&1
echo "[$(ts)] >>> RUNG5_DONE rc=$?"
else echo "[$(ts)] >>> RUNG5_SKIPPED (no vace gguf)"; fi
echo "[$(ts)] === ALL_PROOFS_DONE ==="
