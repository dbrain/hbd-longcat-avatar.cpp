#!/usr/bin/env bash
# Fetch StdGEN + UltraShape weights to /mnt/hdd (NOT baked into the docker image). Run on the host
# (needs `huggingface-cli`/`hf` + `wget`) OR inside the image: `conda run -n ultrashape bash fetch_weights.sh`.
# Heavy: UltraShape ckpt ~7.4GB, StdGEN repo several GB, SAM ViT-H 2.4GB. Slower on /mnt/hdd — fine.
set -uo pipefail
W=/mnt/hdd/3d/avatar-shootout/_weights
mkdir -p "$W/stdgen/ckpt" "$W/ultrashape/checkpoints"
HF="${HF:-huggingface-cli}"; command -v "$HF" >/dev/null || HF="hf"

echo "=== UltraShape (infinith/UltraShape -> $W/ultrashape) ==="
$HF download infinith/UltraShape --local-dir "$W/ultrashape" || echo "!! UltraShape fetch failed"

echo "=== StdGEN (hyz317/StdGEN -> $W/stdgen/ckpt) ==="
$HF download hyz317/StdGEN --local-dir "$W/stdgen/ckpt" || echo "!! StdGEN fetch failed"

echo "=== SAM ViT-H (StdGEN canonicalize dep) -> $W/stdgen/ckpt/sam_vit_h_4b8939.pth ==="
SAM="$W/stdgen/ckpt/sam_vit_h_4b8939.pth"
[ -f "$SAM" ] || wget -q --show-progress -O "$SAM" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth || echo "!! SAM fetch failed"

echo "=== done. Layout ==="
du -sh "$W"/* 2>/dev/null
echo "NOTE: UltraShape refine wants checkpoints/ultrashape_v1.pt — verify the downloaded filename:"
ls -la "$W/ultrashape" 2>/dev/null | grep -iE "\.pt$|\.safetensors$|\.ckpt$" || true
