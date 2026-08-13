#!/usr/bin/env bash
# Native mesh texturing (docker image `pixal3d-tex`): texture a provided mesh from a reference image via
# the pixal3d Trellis2TexturingPipeline (TRELLIS.2-4B weights — the release that ships the shape ENCODER).
# Reuses the unported encoder/flow as a golden source for the eventual C++ encoder port. NEEDS GPU.
#   ./pixal3d_tex_run.sh <mesh.glb> <image_rgba.png> <out.glb> [resolution=1024] [texsize=2048]
# Weights at $WROOT (download once via the block below). Big stuff lives on /mnt/hdd (not the SSD).
set -euo pipefail
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # cpp_port (holds shootout/texture_mesh.py)
PIXAL3D_REPO=/mnt/hdd/3d/avatar-shootout/Pixal3D          # the TRELLIS.2-fork python pipeline code
WROOT=/mnt/hdd/pixal3d_tex                                # weights (TRELLIS.2-4B set + dino mirror)
WTEX="$WROOT/trellis2_4b"; WDINO="$WROOT/dinov3_local"
# Resolve the reserved 3060 by UUID for Docker and local CUDA alike; a numeric
# ordinal can select the busy 5060 Ti after device-order changes.
GPU_DEVICE="${PIXAL3D_GPU_DEVICE:-${IMAGE_TO_RIG_GPU_3060_UUID:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {uuid=$1} END {print uuid}')}}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_DEVICE" 2>/dev/null | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_DEVICE' is '$GPU_NAME', expected the reserved RTX 3060" >&2; exit 1; }

MESH="${1:?usage: pixal3d_tex_run.sh <mesh.glb> <image_rgba.png> <out.glb> [res] [texsize]}"
IMG="${2:?need reference RGBA image}"
OUT="${3:?need out.glb}"
RES="${4:-1024}"; TEXSIZE="${5:-2048}"
GOLDEN="${GOLDEN_DIR:-}"

[ -f "$WTEX/texturing_pipeline.json" ] || { echo "ERR: weights missing at $WTEX — run the download (see header of this script / task)"; exit 1; }

echo "=== pixal3d-tex: $MESH + $IMG -> $OUT  (res $RES, texsize $TEXSIZE, GPU $GPU_DEVICE) ==="
if docker image inspect pixal3d-tex >/dev/null 2>&1; then
  docker run --rm --gpus "device=${GPU_DEVICE}" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e HF_HOME="$WROOT/hf_cache" \
    -e ATTN_BACKEND=sdpa \
    -v /mnt/hdd:/mnt/hdd -v "$CP":/driver \
    pixal3d-tex bash -c "cd $PIXAL3D_REPO && PYTHONPATH=$PIXAL3D_REPO nice -n ${IMAGE_TO_RIG_NICE_LEVEL:-10} python3 /driver/shootout/texture_mesh.py \
        --weights '$WTEX' --mesh '$MESH' --image '$IMG' --out '$OUT' \
        --resolution $RES --texture_size $TEXSIZE --dino '$WDINO' ${GOLDEN:+--golden '$GOLDEN'}"
else
  # The reference environment is also installed locally on the render host.  Using it is
  # intentional: do not silently pull an unknown image, and preserve the same GPU isolation.
  PYTHON="$PIXAL3D_REPO/.venv/bin/python3"
  [[ -x "$PYTHON" ]] || { echo "ERR: missing reference Python at $PYTHON" >&2; exit 2; }
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="$WROOT/hf_cache" ATTN_BACKEND=sdpa \
    PYTHONPATH="$PIXAL3D_REPO" nice -n "${IMAGE_TO_RIG_NICE_LEVEL:-10}" ionice -c 2 -n 7 "$PYTHON" "$CP/shootout/texture_mesh.py" \
      --weights "$WTEX" --mesh "$MESH" --image "$IMG" --out "$OUT" \
      --resolution "$RES" --texture_size "$TEXSIZE" --dino "$WDINO" ${GOLDEN:+--golden "$GOLDEN"}
fi
[[ -s "$OUT" ]] || { echo "ERR: reference runner exited without GLB: $OUT" >&2; exit 1; }
echo "=== done -> $OUT ==="
