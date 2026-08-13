#!/usr/bin/env bash
# UltraShape refine (docker image `ultrashape`): image + coarse mesh -> refined, watertight (holes
# filled), detail-added mesh. NEEDS GPU. 3060: --low_vram + reduced num_latents/chunk_size (README tips).
# All-in-one ckpt (vae+dit+conditioner in ultrashape_v1.pt). Output is HEAVIER than input (densifies).
#   ./ultrashape_run.sh <coarse_mesh.glb> <image_matte.png> [outname]
set -uo pipefail
ROOT=/mnt/hdd/3d/avatar-shootout
MESH="${1:-$ROOT/_shootout_out/miku_try.glb}"
IMG="${2:-$ROOT/_shootout_out/miku_try_matte.png}"
NAME="${3:-$(basename "$MESH" | sed 's/\.[^.]*$//')}"
OUT="$ROOT/_shootout_out/ultrashape_$NAME"; mkdir -p "$OUT"
MESH_C="/work/${MESH#"$ROOT/"}"; IMG_C="/work/${IMG#"$ROOT/"}"
NUM_LATENTS="${NUM_LATENTS:-8192}"; CHUNK="${CHUNK:-2048}"; STEPS="${STEPS:-50}"; OCTREE="${OCTREE:-512}"
# Dual-GPU box: pin to the 3060 (sm_86) by UUID. This image's PyTorch only carries sm_50..sm_90 kernels,
# so --gpus all exposes the 5060 Ti (sm_120) and PyTorch dies "no kernel image". Override w/ GPU_UUID.
GPU_UUID="${GPU_UUID:-GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_UUID" 2>/dev/null | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_UUID' is '$GPU_NAME', expected the reserved RTX 3060" >&2; exit 1; }
echo "=== UltraShape refine: $MESH (img $IMG) -> $OUT  (low_vram, num_latents=$NUM_LATENTS chunk=$CHUNK octree=$OCTREE steps=$STEPS) ==="
docker run --rm --gpus "device=$GPU_UUID" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$ROOT":/work ultrashape bash -c "cd /work/UltraShape-1.0 && \
    PYTHONPATH=/work/UltraShape-1.0 python3 scripts/infer_dit_refine.py \
      --config configs/infer_dit_refine.yaml \
      --ckpt /work/_weights/ultrashape/ultrashape_v1.pt \
      --image '$IMG_C' --mesh '$MESH_C' \
      --output_dir '/work/_shootout_out/ultrashape_$NAME' \
      --low_vram --num_latents $NUM_LATENTS --chunk_size $CHUNK --octree_res $OCTREE --steps $STEPS" \
  2>&1 | grep -ivE "NVIDIA Driver|Container image|NGC-DL|FutureWarning|custom_bwd|def backward"
echo "=== done -> $OUT ==="; ls -la "$OUT" 2>/dev/null | head
