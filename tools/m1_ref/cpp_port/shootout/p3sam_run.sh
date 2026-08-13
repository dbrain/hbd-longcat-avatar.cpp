#!/usr/bin/env bash
# Run P3-SAM (docker image `p3sam`) part-segmentation on a mesh. NEEDS GPU (--gpus all).
# Segments the fused mesh into parts → colored part GLBs + per-face part labels.
#   ./p3sam_run.sh [mesh] [outname]
# Default mesh = the Pixal3D Miku. Sonata backbone weights persist in _sonata_cache.
set -euo pipefail
ROOT=/mnt/hdd/3d/avatar-shootout
MESH="${1:-$ROOT/_shootout_out/miku_res1024.glb}"
NAME="${2:-$(basename "$MESH" | sed 's/\.[^.]*$//')}"
OUT="$ROOT/_shootout_out/p3sam_$NAME"
MESH_C="/work/${MESH#"$ROOT/"}"; [ "$MESH_C" = "/work/$MESH" ] && MESH_C="$MESH"

# 3060 12GB: seg-head tensor = [point_num, prompt_bs, 518] fp32. Default prompt_bs=32 OOMs (~6.6GB);
# 8 keeps it ~1.6GB (just more prompt iterations, no quality loss). Tune via PROMPT_BS / POINT_NUM env.
PROMPT_BS="${PROMPT_BS:-8}"; POINT_NUM="${POINT_NUM:-100000}"
# Pin to the 3060 (sm_86): the image's PyTorch has no sm_120 kernels for the 5060 Ti.
GPU_UUID="${GPU_UUID:-GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6}"
P3SAM_IMAGE="${P3SAM_IMAGE:-p3sam}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_UUID" 2>/dev/null | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_UUID' is '$GPU_NAME', expected the reserved RTX 3060" >&2; exit 1; }
[[ -f "$MESH" ]] || { echo "missing input mesh: $MESH" >&2; exit 2; }
docker image inspect "$P3SAM_IMAGE" >/dev/null 2>&1 || {
  echo "missing P3-SAM container image '$P3SAM_IMAGE'; build the checked-in Dockerfile or set P3SAM_IMAGE" >&2
  exit 2
}
mkdir -p "$OUT" "$ROOT/_sonata_cache"
echo "=== P3-SAM segment: $MESH -> $OUT  (prompt_bs=$PROMPT_BS point_num=$POINT_NUM) ==="
docker run --rm --gpus "device=$GPU_UUID" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$ROOT":/work \
  -v "$ROOT/_sonata_cache":/root/sonata \
  "$P3SAM_IMAGE" bash -c "cd /work/Hunyuan3D-Part/P3-SAM/demo && \
    PYTHONPATH=/work/Hunyuan3D-Part/P3-SAM:/work/Hunyuan3D-Part/XPart/partgen \
    python3 auto_mask.py \
      --ckpt_path /work/Hunyuan3D-Part/P3-SAM/weights/p3sam/p3sam.safetensors \
      --mesh_path '$MESH_C' \
      --output_path '/work/_shootout_out/p3sam_$NAME' \
      --prompt_bs $PROMPT_BS --point_num $POINT_NUM \
      --parallel 0" 2>&1 | grep -ivE "FutureWarning|custom_bwd|def backward|NVIDIA Driver|Container image|NGC-DL"
echo "=== done -> $OUT (colored part GLB + labels) ==="
[[ -s "$OUT/auto_mask_mesh_final.glb" && -s "$OUT/auto_mask_mesh_final_face_ids.npy" ]] || {
  echo "P3-SAM did not produce final mesh/face-label evidence in $OUT" >&2
  exit 1
}
ls -la "$OUT" | head
