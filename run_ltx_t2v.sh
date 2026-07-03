#!/usr/bin/env bash
# LTX-2.3 t2v — generate a SILENT source video (frames dir) for relip experiments.
# Same prod engine/recipe as run_ltx_relip.sh, minus the relip control-video/drive-audio.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"
BUILDER="longcat-avatar-dev:builder-cudnn-ff"; LTXSRC="/home/dbrain/dev/longcat-avatar.cpp"; LTX2="$LTXSRC/models/ltx2"
W=${W:-1280}; H=${H:-704}; FR=${FR:-97}; FPS=${FPS:-24}; MAXV=${MAXV:-7.5}; STEPS=${STEPS:-8}; SEED=${SEED:-42}
TAG=${TAG:-ltxt2v}; OUT="$REPO/$TAG"; rm -rf "$OUT"; mkdir -p "$OUT"
PROMPT=${PROMPT:-"a person on a city street, photorealistic, cinematic"}
DIT=${DIT:-nvfp4-CLEAN.gguf}
PRODENV=(
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks -e LTX_DIT_F16=1
  -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1
  -e LONGCAT_NO_OFFLOAD_PIPELINING=0 -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_PREFETCH_POOL=1
  -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=1 -e LONGCAT_FFN_TILE_TOKENS=4096 -e LONGCAT_ENCODE_MAX_VRAM=6.5 -e LONGCAT_DIT_NO_MMAP=0
  -e "LTX_CUSTOM_SIGMAS=${LTX_CUSTOM_SIGMAS:-1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0}"
)
echo "=== LTX t2v: $TAG ${FR}f @ ${W}x${H} ==="
docker run --rm --gpus "\"device=1\"" "${PRODENV[@]}" \
  -v "$LTXSRC:/ltxsrc" -v "$REPO:/src" -v "$LTX2:/ltx2" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src "$BUILDER" \
  stdbuf -oL -eL /ltxsrc/build/bin/sd-cli -M vid_gen \
  --diffusion-model /ltx2/$DIT --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors \
  --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
  --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  -p "$PROMPT" -W $W -H $H --video-frames $FR --fps $FPS --sampling-method euler --steps $STEPS --cfg-scale 1.0 --diffusion-fa \
  --vae-tiling --vae-relative-tile-size 1x1 --temporal-tiling --extra-tiling-args temporal_tile_frames=4,temporal_tile_overlap=1 \
  --offload-to-cpu --mmap --max-vram $MAXV -s $SEED -v \
  -o /src/$TAG/f%03d.png > "$OUT/log" 2>&1
n=$(ls "$OUT"/*.png 2>/dev/null|wc -l); echo "  -> $n/$FR frames"
[ "$n" -ge 1 ] && ffmpeg -y -framerate 24 -pattern_type glob -i "$OUT/*.png" -c:v libx264 -crf 16 -pix_fmt yuv420p "tools/lightx2v-test/out/$TAG.mp4" -loglevel error 2>/dev/null && echo "  -> $TAG.mp4"
[ "$n" -lt "$FR" ] && grep -aiE "out of memory|error|fail" "$OUT/log"|grep -aviE 0x|tail -3
