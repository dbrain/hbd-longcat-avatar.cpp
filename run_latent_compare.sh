#!/bin/bash
# Pixel-carry vs latent-carry continuation over 3 windows (same seed). Shows whether the
# VAE decode->re-encode roundtrip degrades appearance over a multi-segment chain.
cd /home/dbrain/dev/wan22-infinitetalk
export LONGCAT_NO_FUSED_ROPE=1
ts(){ date +%H:%M:%S; }

# wait for the main overnight batch to clear the CPU
while pgrep -f run_overnight.sh >/dev/null 2>&1; do sleep 60; done
sleep 10

run() {  # $1 = out dir, plus env already set
  IT_SEED=42 ./build/bin/sd-infinitetalk \
    --dit models/infinitetalk-14b-q4_k.gguf --wav2vec models/chinese-wav2vec2-base-f16.gguf \
    --vae models/longcat-wan-vae-f16.gguf --umt5 models/longcat-umt5-xxl-q8_0.gguf \
    --image models/lenna_face.jpg --wav models/jfk_speech.wav \
    --prompt "a woman talking, close up portrait, frontal" \
    --frames 33 --height 256 --width 256 --steps 4 --shift 5 --motion-frame 9 --max-windows 3 \
    --fps 25 --distilled --cpu --out "$1"
}

echo "[$(ts)] === PIXEL-carry (baseline, 3 windows) ==="
run it_carry_pixel > logs/carry-pixel.log 2>&1
echo "[$(ts)] PIXEL rc=$? frames=$(ls it_carry_pixel/*.png 2>/dev/null | wc -l)"

echo "[$(ts)] === LATENT-carry (3 windows, no VAE roundtrip) ==="
IT_LATENT_CARRY=1 run it_carry_latent > logs/carry-latent.log 2>&1
echo "[$(ts)] LATENT rc=$? frames=$(ls it_carry_latent/*.png 2>/dev/null | wc -l)"

echo "[$(ts)] === CARRY COMPARE DONE (seams at ~frame 33, 57; window=33f motion-frame=9) ==="
