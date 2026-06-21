#!/usr/bin/env bash
# Build the LTX cuDNN-borrow eye-test page from the A/B webms. Copies clips, extracts a
# representative frame from each, builds a side-by-side montage, computes PSNR vs baseline,
# and writes ltx.html into the :8099 server dir.
set -uo pipefail
SRC=/home/dbrain/dev/longcat-avatar.cpp/ltx_ab_out
DST=/home/dbrain/dev/bench/fp4/out
FFM="docker run --rm -v $SRC:/s -v $DST:/d linuxserver/ffmpeg"
FRAME="${FRAME:-48}"
mkdir -p "$DST"

labels=(ab_baseline ab_attn ab_conv3d ab_all)
for l in "${labels[@]}"; do
  [ -f "$SRC/$l.webm" ] || { echo "missing $l.webm"; continue; }
  cp "$SRC/$l.webm" "$DST/ltx_$l.webm"
  $FFM -y -i "/s/$l.webm" -vf "select=eq(n\,$FRAME)" -frames:v 1 "/d/ltx_${l}_f.png" >/dev/null 2>&1
done

# PSNR of each vs baseline (same seed 42 -> RNG-matched)
psnr() { # webm
  $FFM -i "/s/ab_baseline.webm" -i "/s/$1.webm" -lavfi psnr -f null - 2>&1 | grep -oE "average:[0-9.]+|average:inf" | tail -1
}
echo "PSNR vs baseline:"
for l in ab_attn ab_conv3d ab_all; do echo "  $l: $(psnr $l.webm)"; done

# side-by-side montage of the representative frame (baseline | conv3d | attn | all)
$FFM -y -i "/d/ltx_ab_baseline_f.png" -i "/d/ltx_ab_conv3d_f.png" -i "/d/ltx_ab_attn_f.png" -i "/d/ltx_ab_all_f.png" \
  -filter_complex "[0][1][2][3]hstack=4" "/d/ltx_montage.png" >/dev/null 2>&1
echo "wrote $DST/ltx_montage.png + ltx_*.webm + frames"
