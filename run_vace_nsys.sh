#!/usr/bin/env bash
# Deep nsys profile of the VACE DiT (confirm kernel split + fusion firing + vace-block overhead).
# 1 high + 1 low step (both experts' kernels), FR=21 production tokens, maxv7.3, gray-cached encode.
# Extracts a GPU-kernel time summary so we can declare the floor metric-by-metric.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out"; mkdir -p "$OUT/nsys_gcache"
NSYS=/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys
M=/models; DR=/models/_drive; W=480; H=832; FR=21; LABEL=vace_dit_nsys
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P1="A young man with tousled dark brown hair sings into the camera, warm amber neon at dusk, cinematic, medium shot."
rm -f "$OUT/${LABEL}.nsys-rep" "$OUT/${LABEL}.sqlite" "$OUT/${LABEL}.qdstrm"
docker run --rm --gpus all --cap-add SYS_ADMIN -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  -e VACE_GRAY_CACHE_DIR=/src/perf_out/nsys_gcache "$BUILDER" \
  "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite true -o "/src/perf_out/${LABEL}" \
  /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --steps 1 --high-noise-steps 1 \
    --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram 7.3 \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s 42 --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P1" \
    -o /src/perf_out/${LABEL}.frames/f%03d.png -v > "$OUT/${LABEL}.run.log" 2>&1
echo "profile rc=$? ; extracting kernel summary..."
docker run --rm -v "$REPO:/src" -w /src "$BUILDER" bash -lc "
  $NSYS stats --report cuda_gpu_kern_sum --format table,csv --output -,/src/perf_out/${LABEL}.kern /src/perf_out/${LABEL}.nsys-rep 2>/dev/null | head -45
" 2>&1 | grep -vE "^==|CUDA Version|Container|deep-learn|NGC|copy of|accept the terms|docs.nvidia|^WARNING|^$"
echo "=== NORM-FUSE-DBG / fusion firing (from run log) ==="
grep -cE "NORM-FUSE|fused" "$OUT/${LABEL}.run.log" 2>/dev/null || true
grep -E "execute_graph timing" "$OUT/${LABEL}.run.log" 2>/dev/null | grep -i VACE | head -3
