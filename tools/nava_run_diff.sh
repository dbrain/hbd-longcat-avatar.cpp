#!/bin/bash
# Phase-1 cpp DiT validation driver:
#   1. npz -> input .bin (harness format)
#   2. run the cpp nava harness on the real inputs (CPU), dumping per-block taps
#   3. diff per-block vs the PyTorch reference
set -euo pipefail
REPO=/home/dbrain/dev/longcat-avatar.cpp
NPZ=${NPZ:-/mnt/hdd/nava/cpp-runs/_ref/ref_tensors.npz}
INDIR=${INDIR:-/mnt/hdd/nava/cpp-runs/_ref/bin}
OUTDIR=${OUTDIR:-/mnt/hdd/nava/cpp-runs/_ref/cpp_out}
GGUF=${GGUF:-$REPO/models/nava-dit-f16.gguf}
PSNR=${PSNR:-40}

mkdir -p "$INDIR" "$OUTDIR"
cd "$REPO"

echo "=== [1/3] npz -> input bins ==="
python3 tools/nava_npz_to_bin.py "$NPZ" "$INDIR"

echo "=== [2/3] cpp harness forward (per-block taps -> $OUTDIR) ==="
LONGCAT_DUMP_DIR="$OUTDIR" ./build-nava/bin/nava "$GGUF" "$INDIR" "$OUTDIR"

echo "=== [3/3] diff vs reference ==="
python3 tools/nava_tensor_diff.py "$NPZ" "$OUTDIR" --psnr "$PSNR"
