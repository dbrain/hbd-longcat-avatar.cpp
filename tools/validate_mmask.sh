#!/bin/bash
# One-shot: convert the python mmask reference -> cpp bins, run the cpp oracle in
# masking_modality mode, and diff the cpp separate-attention forward vs PyTorch.
# High dB (blocks 60+, head ~45, like the joint gate) => cpp mmask path is correct.
set -euo pipefail
cd /home/dbrain/dev/longcat-avatar.cpp
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
NPZ=/mnt/hdd/nava/cpp-runs/_ref_mmask/ref_tensors.npz
BIN=/mnt/hdd/nava/cpp-runs/_ref_mmask/bin
OUT=/mnt/hdd/nava/cpp-runs/_ref_mmask/cpp_out
mkdir -p "$BIN" "$OUT"
echo "=== npz -> bins ==="
python3 tools/nava_npz_to_bin.py "$NPZ" "$BIN"
echo "=== cpp oracle (NAVA_MASK_MODALITY=1, f16) ==="
NAVA_MASK_MODALITY=1 LONGCAT_DUMP_DIR="$OUT" ./build-nava/bin/nava models/nava-dit-f16.gguf "$BIN" "$OUT" 2>&1 | grep -iE "MASK_MODALITY|velocity_audio shape|forward done" || true
echo "=== AUDIO-slice diff: cpp mmask vs python mmask ==="
python3 tools/nava_audio_diff.py "$NPZ" "$OUT"
