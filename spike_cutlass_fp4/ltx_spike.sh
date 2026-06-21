#!/usr/bin/env bash
# Phase B LTX spike: cuDNN SDPA (long video seq) + cuDNN conv3d vs ggml im2col_3d.
# Foreground, pinned to the 5060 Ti. One docker run builds + runs all benches.
set -uo pipefail
REPO=/home/dbrain/dev/flux2.cpp
IMG=flux2-dev:builder-cudnn
GPU=GPU-bd93e020-65d1-1a5c-ad6c-57c9f655cf45
SPK=/src/spike_cutlass_fp4
OUT="$REPO/spike_cutlass_fp4/ltx_out"
mkdir -p "$OUT"

docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -e "CUDA_VISIBLE_DEVICES=$GPU" \
  -v "$REPO:/src" -v "$OUT:/out" -w "$SPK" \
  "$IMG" bash -c '
set -e
NVCC="nvcc -O3 -arch=sm_120 -std=c++17"
echo "=== building ==="
$NVCC attn_ltx_golden.cu     -o /out/attn_ltx_golden     -I/opt/cudnn-frontend/include -lcudnn -lcublas -lnvrtc
$NVCC conv3d_golden.cu       -o /out/conv3d_golden       -I/opt/cudnn-frontend/include -lcudnn -lcublas -lnvrtc
$NVCC conv3d_ggml_baseline.cu -o /out/conv3d_ggml_baseline -lcublas
$NVCC attn_ggml_baseline.cu  -o /out/attn_ggml_baseline  -I/src/ggml/include \
   -L/src/build-cudnn/ggml/src -L/src/build-cudnn/ggml/src/ggml-cuda \
   -lggml -lggml-cuda -lggml-base -lggml-cpu -lcublas -lcublasLt -lcuda -lcudnn -lnvrtc -lgomp -lcudart \
   || echo "GGML_ATTN_BUILD_FAILED"
echo "=== SPIKE 1: cuDNN SDPA (LTX self-attn) ==="
echo "--- video D=128 H=30 S=11440 ---"
/out/attn_ltx_golden --H 30 --S 11440 --D 128 --tag ltx_video_S11440 | tee /out/r_attn_video.json
echo "--- video flux2-ref D=128 H=24 S=4608 (sanity vs attn.json 5616us) ---"
/out/attn_ltx_golden --H 24 --S 4608  --D 128 --tag flux2ref_S4608   | tee /out/r_attn_flux2ref.json
echo "--- audio D=64 H=32 S=4096 (representative) ---"
/out/attn_ltx_golden --H 32 --S 4096  --D 64  --tag ltx_audio_S4096  | tee /out/r_attn_audio.json
echo "=== SPIKE 1 ggml flash baseline ==="
LD_LIBRARY_PATH=/src/build-cudnn/ggml/src:/src/build-cudnn/ggml/src/ggml-cuda \
  /out/attn_ggml_baseline --H 30 --S 11440 --D 128 --tag ltx_video_S11440 | tee /out/r_attn_ggml_video.json || echo "ggml video attn run failed"
LD_LIBRARY_PATH=/src/build-cudnn/ggml/src:/src/build-cudnn/ggml/src/ggml-cuda \
  /out/attn_ggml_baseline --H 24 --S 4608 --D 128 --tag flux2ref_S4608 | tee /out/r_attn_ggml_flux2ref.json || echo "ggml flux2ref attn run failed"
LD_LIBRARY_PATH=/src/build-cudnn/ggml/src:/src/build-cudnn/ggml/src/ggml-cuda \
  /out/attn_ggml_baseline --H 32 --S 4096 --D 64 --tag ltx_audio_S4096 | tee /out/r_attn_ggml_audio.json || echo "ggml audio attn run failed"
echo "=== SPIKE 2: cuDNN conv3d (LTX VAE, ID=3) ==="
/out/conv3d_golden --id 3 | tee /out/r_conv3d_cudnn_id3.json
echo "=== SPIKE 2: ggml im2col_3d+GEMM (LTX VAE, ID=3) ==="
/out/conv3d_ggml_baseline --id 3 | tee /out/r_conv3d_ggml_id3.json
echo "=== SPIKE 2 sensitivity: ID=9 ==="
/out/conv3d_golden --id 9 | tee /out/r_conv3d_cudnn_id9.json
/out/conv3d_ggml_baseline --id 9 | tee /out/r_conv3d_ggml_id9.json
echo "=== DONE ==="
' 2>&1 | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g;s/\r/\n/g' | tee "$OUT/run.log"
echo "EXIT=${PIPESTATUS[0]}"
