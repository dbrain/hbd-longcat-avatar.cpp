#!/usr/bin/env bash
cd "$(dirname "$0")"
# tight VRAM peak poll (0.3s) during the E2E
( max=0; while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1);
   [ -n "$u" ] && [ "$u" -gt "$max" ] && { max=$u; echo "VRAMPEAK $max MiB"; }; sleep 0.3; done ) > _turtle_vram.log 2>&1 &
POLL=$!
NVIDIA_TF32_OVERRIDE=0 PIXAL3D_FLASH=1 PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 \
  ./pixal3d --model /mnt/hdd/pixal3d/weights_gguf_f16 --image prep_test_matte.png \
    --out turtle_uvdump.glb --remesh --tex --fast
kill $POLL 2>/dev/null
echo "=== VRAM PEAK ==="; tail -1 _turtle_vram.log
