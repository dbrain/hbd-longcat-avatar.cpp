#!/usr/bin/env bash
# Deep nsys profile of the VACE DiT at the PROD shape (1280x704x65f) with the FULL fit stack
# (F16-rope, cuDNN attn, nvfp4 cublasLt, FP8 FFN, gray-suffix). 1 high + 1 low step = every kernel
# type. Extracts the GPU-kernel time summary so we can see EXACTLY where the 700s/clip goes
# (attention vs matmul vs conv/VAE vs glue) at large frames. Quality is locked; this is the
# no-compromise speed dive.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"; M="$REPO/models"
B="longcat-avatar-dev:builder-cudnn"
NSYS=/opt/nvidia/nsight-compute/2026.2.0/host/target-linux-x64/nsys
OUT="$REPO/perf_out"; mkdir -p "$OUT/gcache"
VAE="/models/longcat-wan-vae-f16.gguf"; T5="/models/longcat-umt5-xxl-q8_0.gguf"
VL=/models/wan22-vace-fun-a14b-low-distillT2V-nvfp4.gguf; VH=/models/wan22-vace-fun-a14b-high-distillT2V-nvfp4.gguf
W=${W:-1280}; H=${H:-704}; FR=${FR:-65}; LABEL=${LABEL:-vace1280}
P="the same person keeps dancing energetically on the neon-lit city street at night, cinematic"
ENV="-e WAN_DISTILL_SIGMAS=1 -e WAN_DISTILL_SHIFT=5 -e VACE_CONT_FRAMES=5 -e VACE_CONT_LATENT=/src/cont_bank/seg1.bin \
 -e VACE_CONT_AGC=1 -e VACE_CONT_AGC_TARGET=0.65 -e LONGCAT_FFN_TILE_TOKENS=4096 \
 -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=blocks. \
 -e GGML_CUDNN_ATTN=1 -e WAN_DIT_F16=1 -e WAN_ROPE_F16=1 -e GGML_CUDNN_ATTN_F16_OUT=1 \
 -e LONGCAT_VAE_TEMPORAL_CHUNK=1 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 \
 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1 -e VACE_GRAY_CACHE_DIR=/src/perf_out/gcache"
rm -f "$OUT/${LABEL}.nsys-rep" "$OUT/${LABEL}.sqlite"
echo "profiling $LABEL @ ${W}x${H}x${FR} (1 high + 1 low step)..."
docker run --rm --gpus '"device=1"' --cap-add SYS_ADMIN -e TMPDIR=/src/perf_out $ENV -v "$REPO:/src" -v "$M:/models" -w /src "$B" \
  "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite --force-overwrite true -o "/src/perf_out/${LABEL}" \
  /src/build/bin/sd-cli -M vid_gen --diffusion-model "$VL" --high-noise-diffusion-model "$VH" --vae "$VAE" --t5xxl "$T5" \
    --control-video /src/cont_tail -p "$P" --init-img /models/_drive/flux_neon_seed7.png \
    --sampling-method euler_a --high-noise-sampling-method euler_a --eta 1.0 \
    --steps 1 --high-noise-steps 1 --cfg-scale 1 --high-noise-cfg-scale 1 --flow-shift 5 -s 42 \
    -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram 10 \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 -o "/src/perf_out/${LABEL}.frames/f%03d.png" -v \
    > "$OUT/${LABEL}.run.log" 2>&1
echo "profile rc=$?"
echo "=== GPU KERNEL TIME SUMMARY (where the compute goes) ==="
docker run --rm -v "$REPO:/src" -w /src "$B" bash -lc \
  "$NSYS stats --report cuda_gpu_kern_sum --format table --force-export=true /src/perf_out/${LABEL}.nsys-rep 2>/dev/null | head -40" \
  2>&1 | grep -vE "^==|CUDA Version|Container|deep-learn|NGC|copy of|accept|docs.nvidia|^WARNING|Use the|^$|Processing|Generating|Exporting|SQLite|^ +\* "
