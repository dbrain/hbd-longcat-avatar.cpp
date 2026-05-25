#!/usr/bin/env bash
# 93f RESIDENT probe (no --offload-to-cpu), steps=1, to test if FFN tiling lets the
# monolithic compute buffer + resident weights fit 12 GB.
set -x
/src/build/bin/sd-cli -M vid_gen \
  -m /models/longcat-avatar-1.5-dit-dmd-q4_k.gguf \
  --t5xxl /models/longcat-umt5-xxl-q8_0.gguf \
  --vae /models/longcat-wan-vae-f16.gguf \
  --audio-vae /models/longcat-whisper-v3-encoder-f16.gguf \
  --init-img /models/_testinputs/girl_480x832.png \
  --audio /models/_testinputs/speech_16k.wav \
  -p "a person talking" --cfg-scale 1.0 --video-frames 93 -W 480 -H 832 \
  --steps 1 --diffusion-fa --seed 42 --clip-on-cpu --max-vram 11 \
  -o /models/_perf/probe93_resident.webm
