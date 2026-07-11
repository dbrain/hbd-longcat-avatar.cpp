#!/usr/bin/env bash
set -euo pipefail
WT=${WT:-/home/dbrain/dev/longcat-avatar-ltxdenoise}
LTX2=${LTX2:-/home/dbrain/dev/longcat-avatar.cpp/models/ltx2}
SRC=${SRC:-/home/dbrain/dev/longcat-avatar-wan22/perf_out/prod_eyetest/lipdub_source.webm}
SONG=${SONG:-$LTX2/_drive/music_video/song.wav}
OUTDIR=${OUTDIR:-$WT/ltx-denoise-repro/_relip_current}
BIN=${BIN:-/src/build-sa3/bin/sd-cli}; GPU=${GPU:-1}; STEPS=${STEPS:-8}; REFSTEPS=${REFSTEPS:-3}; MAXV=${MAXV:-8}; FR=${FR:-97}
# Relip conditioning budget.  These locked production defaults retain the
# first 11 consecutive reference-latent frames at half spatial resolution.
# Override them only for an explicitly labelled A/B run.
RELIP_REF_DOWNSCALE=${LTXAV_RELIP_REF_DOWNSCALE:-2}
RELIP_REF_TSTRIDE=${LTXAV_RELIP_REF_TSTRIDE:-1}
RELIP_REF_MAX_TFRAMES=${LTXAV_RELIP_REF_MAX_TFRAMES:-11}
REFINE_MAXV=${LTXAV_REFINE_MAX_VRAM:-8.5}
VAE_SPATIAL_TILES=${LTX_VAE_SPATIAL_TILES:-2x2}
RELIP_BASE_RESIDENT_CAP=${LTXAV_RELIP_BASE_RESIDENT_CAP:-1}
SHARED_RESIDENT_MAX_MB=${LONGCAT_SHARED_RESIDENT_MAX_MB:-3000}
FREE_TE_PARAMS=${LTXAV_FREE_TE_PARAMS:-1}
OFFLOAD_PREFETCH_THREAD=${LONGCAT_OFFLOAD_PREFETCH_THREAD:-0}
NO_OFFLOAD_PIPELINING=${LONGCAT_NO_OFFLOAD_PIPELINING:-1}
# Keep the Relip repro on the same locked attention policy as production.
SA3_ENABLED=${GGML_LTX_SA3:-1}; SA3_POLICY=${GGML_LTX_SA3_POLICY:-first}
SECONDS=$(awk "BEGIN { printf \"%.3f\", $FR / 24 }")
mkdir -p "$OUTDIR/control"
docker run --rm -v "$(dirname "$SRC"):/in:ro" -v "$OUTDIR/control:/out" --entrypoint ffmpeg linuxserver/ffmpeg -y -v error -i "/in/$(basename "$SRC")" -frames:v "$FR" /out/%05d.png
docker run --rm -v "$(dirname "$SONG"):/in:ro" -v "$OUTDIR:/out" --entrypoint ffmpeg linuxserver/ffmpeg -y -v error -i "/in/$(basename "$SONG")" -t "$SECONDS" -ac 1 -ar 16000 /out/drive.wav
: > "$OUTDIR/vram.log"
(while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" >> "$OUTDIR/vram.log" 2>/dev/null; sleep 1; done) & p=$!
t0=$(date +%s)
set +e
docker run --rm --gpus "\"device=$GPU\"" \
 -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1 -e LTX_DIT_F16=1 \
 -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks \
 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1 \
 -e GGML_LTX_SA3="$SA3_ENABLED" -e GGML_LTX_SA3_POLICY="$SA3_POLICY" -e GGML_LTX_SA3_DELTA_F16=1 -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=0 -e LONGCAT_FFN_TILE_TOKENS=8192 -e LONGCAT_ENCODE_MAX_VRAM=6.5 \
 -e LONGCAT_NO_PREFETCH_POOL=1 -e LONGCAT_OFFLOAD_PREFETCH_THREAD="$OFFLOAD_PREFETCH_THREAD" -e LONGCAT_NO_OFFLOAD_PIPELINING="$NO_OFFLOAD_PIPELINING" -e LONGCAT_DIT_NO_MMAP=0 \
 -e LTXAV_VAE_LAZY=1 -e LTXAV_DIT_FREE_DURING_DECODE=1 -e LONGCAT_VRAM_BREAKDOWN=1 -e LTX_VAE_HEAD_F32=1 -e LTX_VAE_CONV3D_WTILES=16 -e LTX_VAE_CONV3D_HTILES=8 -e LTX_VAE_DECODE_F16=1 \
 -e LTX_VAE_SPATIAL_TILES="$VAE_SPATIAL_TILES" -e LTX_VAE_SPATIAL_OVERLAP=4 -e LTX_CUSTOM_SIGMAS=1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0 \
 -e LTXAV_CHAIN_OVERLAP_DROP=24 -e LTXAV_SKIP_AUDIO_DECODE=1 -e LONGCAT_ATTN_TILES=2 -e LONGCAT_PERSIST_GRAPH_INPUTS=1 \
 -e LTXAV_RELIP_TWOSTAGE=1 -e LTXAV_RELIP_ENCODE_TILE=0.25 \
 -e LTXAV_RELIP_REF_DOWNSCALE="$RELIP_REF_DOWNSCALE" -e LTXAV_RELIP_REF_TSTRIDE="$RELIP_REF_TSTRIDE" \
 -e LTXAV_RELIP_REF_MAX_TFRAMES="$RELIP_REF_MAX_TFRAMES" \
 -e LTXAV_RELIP_BASE_RESIDENT_CAP="$RELIP_BASE_RESIDENT_CAP" \
 -e LTXAV_FREE_TE_PARAMS="$FREE_TE_PARAMS" \
 ${SHARED_RESIDENT_MAX_MB:+-e LONGCAT_SHARED_RESIDENT_MAX_MB="$SHARED_RESIDENT_MAX_MB"} \
 ${REFINE_MAXV:+-e LTXAV_REFINE_MAX_VRAM="$REFINE_MAXV"} \
 -v "$WT:/src" -v "$LTX2:/ltx2" -v "$OUTDIR:/work" -w /src longcat-avatar-dev:builder-cudnn-ff \
 bash -lc "stdbuf -oL -eL $BIN -M vid_gen --diffusion-model /ltx2/nvfp4-CLEAN-lipdub.gguf --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors --lora-model-dir /ltx2/loras --control-video /work/control --drive-audio /work/drive.wav -p 'A person singing on a neon-lit city street at night, cinematic, photorealistic <lora:ltx-2.3-22b-ic-lora-lipdub-0.9:1.0>' -s 42 -W 1280 -H 704 --video-frames $FR --fps 24 --sampling-method euler_a --steps $STEPS --cfg-scale 1.0 --diffusion-fa --hires --hires-upscalers-dir /ltx2/latent_upscale_models --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-steps $REFSTEPS --vae-tiling --vae-relative-tile-size 1x1 --offload-to-cpu --mmap --max-vram $MAXV -v -o /work/relip.webm" 2>&1 | tee "$OUTDIR/render.log"
rc=${PIPESTATUS[0]}; set -e; kill "$p" 2>/dev/null || true
peak_driver=$(perl -ne 'while(/driver_used=(\d+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$OUTDIR/render.log")
peak_cudnn=$(perl -ne 'while(/used ([0-9.]+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$OUTDIR/render.log")
peak_smi=$(sort -n "$OUTDIR/vram.log" | tail -1)
echo "rc=$rc wall=$(( $(date +%s)-t0 ))s ref_downscale=$RELIP_REF_DOWNSCALE ref_tstride=$RELIP_REF_TSTRIDE ref_max_tframes=$RELIP_REF_MAX_TFRAMES free_te_params=$FREE_TE_PARAMS relip_base_resident_cap=$RELIP_BASE_RESIDENT_CAP shared_resident_max_mb=${SHARED_RESIDENT_MAX_MB:-same} offload_prefetch_thread=$OFFLOAD_PREFETCH_THREAD no_offload_pipelining=$NO_OFFLOAD_PIPELINING refine_maxv=${REFINE_MAXV:-same} vae_spatial_tiles=$VAE_SPATIAL_TILES peak_smi=${peak_smi:-0}MiB peak_driver=${peak_driver}MB peak_cudnn=${peak_cudnn}MB" | tee "$OUTDIR/metrics.txt"
if [ "$rc" -eq 0 ]; then
 docker run --rm -v "$OUTDIR:/work" -v "$(dirname "$SONG"):/in:ro" --entrypoint ffmpeg linuxserver/ffmpeg -y -v error -i /work/relip.webm -i "/in/$(basename "$SONG")" -map 0:v:0 -map 1:a:0 -c:v copy -c:a libopus -shortest /work/relip_song.webm
fi
exit "$rc"
