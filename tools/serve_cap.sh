#!/usr/bin/env bash
# Launch flux2-server with an optional GGML_MMQ_X_CAP for the occupancy sweep.
# Usage: serve_cap.sh [CAP]   (CAP empty/0 = upstream mmq_x=128)
set -euo pipefail
REPO_DIR="/home/dbrain/dev/flux2.cpp"
CAP="${1:-0}"
MINLEN="${2:-0}"   # FLUX2_TEXT_MINLEN; 0 = default 512
docker rm -f flux2-iter >/dev/null 2>&1 || true
docker run -d --rm --gpus all --name flux2-iter -p 8095:8080 \
  -e GGML_MMQ_X_CAP="$CAP" \
  -e FLUX2_TEXT_MINLEN="$MINLEN" \
  -v "$REPO_DIR:/src" -v "$REPO_DIR/models:/models:ro" -w /src \
  flux2-dev:builder \
  /src/build/bin/flux2-server \
    --diffusion-model /models/unet/flux-2-klein-base-9b-Q4_K_M.gguf \
    --vae /models/vae/full_encoder_small_decoder.safetensors \
    --llm /models/text_encoders/Qwen3-8B-Q4_K_M.gguf \
    --offload-to-cpu --mmap --diffusion-fa -v \
    -l 0.0.0.0 --listen-port 8080 >/dev/null
echo "started flux2-iter with GGML_MMQ_X_CAP=$CAP"
for i in $(seq 1 60); do curl -s -m 2 http://localhost:8095/health >/dev/null 2>&1 && { echo ready; exit 0; }; sleep 2; done
echo "TIMEOUT waiting for ready" >&2; exit 1
