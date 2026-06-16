#!/usr/bin/env bash
# Native mesh texturing (docker image `pixal3d-tex`): texture a provided mesh from a reference image via
# the pixal3d Trellis2TexturingPipeline (TRELLIS.2-4B weights — the release that ships the shape ENCODER).
# Reuses the unported encoder/flow as a golden source for the eventual C++ encoder port. NEEDS GPU.
#   ./pixal3d_tex_run.sh <mesh.glb> <image_rgba.png> <out.glb> [resolution=1024] [texsize=2048]
# Weights at $WROOT (download once via the block below). Big stuff lives on /mnt/hdd (not the SSD).
set -uo pipefail
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # cpp_port (holds shootout/texture_mesh.py)
PIXAL3D_REPO=/mnt/hdd/3d/avatar-shootout/Pixal3D          # the TRELLIS.2-fork python pipeline code
WROOT=/mnt/hdd/pixal3d_tex                                # weights (TRELLIS.2-4B set + dino mirror)
WTEX="$WROOT/trellis2_4b"; WDINO="$WROOT/dinov3_local"

MESH="${1:?usage: pixal3d_tex_run.sh <mesh.glb> <image_rgba.png> <out.glb> [res] [texsize]}"
IMG="${2:?need reference RGBA image}"
OUT="${3:?need out.glb}"
RES="${4:-1024}"; TEXSIZE="${5:-2048}"
GOLDEN="${GOLDEN_DIR:-}"

[ -f "$WTEX/texturing_pipeline.json" ] || { echo "ERR: weights missing at $WTEX — run the download (see header of this script / task)"; exit 1; }

echo "=== pixal3d-tex: $MESH + $IMG -> $OUT  (res $RES, texsize $TEXSIZE) ==="
docker run --rm --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e HF_HOME="$WROOT/hf_cache" \
  -e ATTN_BACKEND=sdpa \
  -v /mnt/hdd:/mnt/hdd -v "$CP":/driver \
  pixal3d-tex bash -c "cd $PIXAL3D_REPO && PYTHONPATH=$PIXAL3D_REPO python3 /driver/shootout/texture_mesh.py \
      --weights '$WTEX' --mesh '$MESH' --image '$IMG' --out '$OUT' \
      --resolution $RES --texture_size $TEXSIZE --dino '$WDINO' ${GOLDEN:+--golden '$GOLDEN'}" \
  2>&1 | grep -ivE "FutureWarning|custom_bwd|def backward|NVIDIA Driver|Container image|NGC-DL|UserWarning|warnings.warn"
echo "=== done -> $OUT ==="
