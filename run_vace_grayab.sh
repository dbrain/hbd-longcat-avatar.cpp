#!/usr/bin/env bash
# Lever #1 A/B: gray-latent fast path vs the old per-call VAE encode.
# Runs the 2-seg continuation chain TWICE (baseline VACE_NO_GRAY_FAST=1 vs optimized
# gray-fast + disk cache) into separate dirs, then byte-compares every output frame.
# Proves bit-exactness + measures the VAE-encode wall saved per segment.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/grayab"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=480; H=832; FR="${FR:-21}"; MAXV="${MAXV:-6}"; K="${K:-5}"; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf
VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
COMMON=(--vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1
  --sampling-method euler --high-noise-sampling-method euler
  --steps 2 --high-noise-steps 2 --flow-shift 7 -W $W -H $H --video-frames $FR
  --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV
  --vae-tiling --vae-relative-tile-size 0.25x0.25 --temporal-tiling -s $SEED
  --diffusion-model $VL --high-noise-diffusion-model $VH)
P1="A young man with tousled dark brown hair sings into the camera, warm amber neon at dusk, cinematic, medium shot."
P2="The same young man keeps singing, nodding to the beat, warm amber neon at dusk, cinematic, medium shot."
run(){ docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src "${ENVV[@]}" "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen "${COMMON[@]}" "$@"; }

# $1 = tag (base|opt); sets gray-fast env accordingly
do_chain(){
  local tag=$1; local D="$OUT/$tag"
  rm -rf "$D"; mkdir -p "$D/seg1" "$D/seg2" "$D/tail" "$D/gcache"
  local extra=()
  if [ "$tag" = base ]; then extra=(-e VACE_NO_GRAY_FAST=1)
  else extra=(-e VACE_GRAY_CACHE_DIR=/src/perf_out/grayab/$tag/gcache); fi

  echo "=== [$tag] SEG1 ==="; local t0=$(date +%s.%N)
  ENVV=(-e VACE_SAVE_LATENT=/src/perf_out/grayab/$tag/seg1.bin "${extra[@]}")
  run --init-img $DR/char.png -p "$P1" -o /src/perf_out/grayab/$tag/seg1/f%03d.png -v > "$D/seg1.log" 2>&1
  echo "  [$tag] seg1 rc=$? wall=$(awk "BEGIN{printf \"%.1fs\", $(date +%s.%N)-$t0}")"
  grep -E "encode_first_stage completed|gray-latent|generate_video completed|vae encode graph completed" "$D/seg1.log"

  local n=0; for f in $(ls "$D"/seg1/f*.png 2>/dev/null|sort|tail -n $K); do cp "$f" "$(printf "$D/tail/c%03d.png" $n)"; n=$((n+1)); done

  echo "=== [$tag] SEG2 (continuation) ==="; local t1=$(date +%s.%N)
  ENVV=(-e VACE_CONT_FRAMES=$K -e VACE_CONT_LATENT=/src/perf_out/grayab/$tag/seg1.bin "${extra[@]}")
  run --control-video /src/perf_out/grayab/$tag/tail -p "$P2" -o /src/perf_out/grayab/$tag/seg2/f%03d.png -v > "$D/seg2.log" 2>&1
  echo "  [$tag] seg2 rc=$? wall=$(awk "BEGIN{printf \"%.1fs\", $(date +%s.%N)-$t1}")"
  grep -E "encode_first_stage completed|gray-latent|VACE_CONT_LATENT|generate_video completed|vae encode graph completed" "$D/seg2.log"
}

do_chain base
do_chain opt

echo; echo "=== BIT-EXACT FRAME COMPARE (base vs opt) ==="
diffs=0; total=0
for seg in seg1 seg2; do
  for f in "$OUT"/base/$seg/f*.png; do
    bn=$(basename "$f"); total=$((total+1))
    if ! cmp -s "$f" "$OUT/opt/$seg/$bn"; then echo "  DIFF: $seg/$bn"; diffs=$((diffs+1)); fi
  done
done
echo "=== $total frames compared, $diffs differ ==="
