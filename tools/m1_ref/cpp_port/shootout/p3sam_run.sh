#!/usr/bin/env bash
# Run P3-SAM (docker image `p3sam`) part-segmentation on a mesh. NEEDS GPU (--gpus all).
# Segments the fused mesh into parts → colored GLB + per-face part labels → we isolate the HAND part
# and feed it to a per-part retopo backend (MeshFlow/FastMesh) at full budget. (HANDOFF-H §0.)
#   ./p3sam_run.sh [mesh] [outname]
# Default mesh = the pixal3d Miku. Sonata backbone weights persist in _sonata_cache (downloaded once).
set -uo pipefail
ROOT=/mnt/hdd/3d/avatar-shootout
MESH="${1:-$ROOT/_shootout_out/miku_res1024.glb}"          # any .glb/.obj; P3-SAM samples a point cloud
NAME="${2:-$(basename "$MESH" | sed 's/\.[^.]*$//')}"
OUT="$ROOT/_shootout_out/p3sam_$NAME"
mkdir -p "$OUT" "$ROOT/_sonata_cache"
# map host mesh path -> container /work path
MESH_C="/work/${MESH#"$ROOT/"}"; [ "$MESH_C" = "/work/$MESH" ] && MESH_C="$MESH"   # if outside ROOT, pass as-is

# 3060 12GB: seg-head tensor = [point_num, prompt_bs, 518] fp32. Default prompt_bs=32 OOMs (~6.6GB);
# 8 keeps it ~1.6GB (just more prompt iterations, no quality loss). Tune via PROMPT_BS / POINT_NUM env.
PROMPT_BS="${PROMPT_BS:-8}"; POINT_NUM="${POINT_NUM:-100000}"
echo "=== P3-SAM segment: $MESH -> $OUT  (prompt_bs=$PROMPT_BS point_num=$POINT_NUM) ==="
docker run --rm --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$ROOT":/work \
  -v "$ROOT/_sonata_cache":/root/sonata \
  p3sam bash -c "cd /work/Hunyuan3D-Part/P3-SAM/demo && \
    PYTHONPATH=/work/Hunyuan3D-Part/P3-SAM:/work/Hunyuan3D-Part/XPart/partgen \
    python3 auto_mask.py \
      --ckpt_path /work/Hunyuan3D-Part/P3-SAM/weights/p3sam/p3sam.safetensors \
      --mesh_path '$MESH_C' \
      --output_path '/work/_shootout_out/p3sam_$NAME' \
      --prompt_bs $PROMPT_BS --point_num $POINT_NUM \
      --parallel 0" 2>&1 | grep -ivE "FutureWarning|custom_bwd|def backward|NVIDIA Driver|Container image|NGC-DL"
echo "=== done -> $OUT (colored part GLB + labels) ==="
ls -la "$OUT" 2>/dev/null | head