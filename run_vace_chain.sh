#!/usr/bin/env bash
# VACE 2-segment continuation test at 480x832 maxv6 (the fast resident config).
# seg1: free i2v gen from char.png, banks its diffusion latent (VACE_SAVE_LATENT).
# seg2: continues from seg1's tail via the latent backdoor (VACE_CONT_LATENT) + control-video tail.
# Validates: VACE model loads, seam smoothness, identity/motion continuity, per-seg wall+VRAM.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/vace"; mkdir -p "$OUT"
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

rm -rf "$OUT"/seg1 "$OUT"/seg2l "$OUT"/tail "$OUT"/seg1.bin; mkdir -p "$OUT"/seg1 "$OUT"/seg2l "$OUT"/tail
echo "=== SEG1 (free i2v, bank latent) ==="; t0=$(date +%s.%N)
ENVV=(-e VACE_SAVE_LATENT=/src/perf_out/vace/seg1.bin)
run --init-img $DR/char.png -p "$P1" -o /src/perf_out/vace/seg1/f%03d.png -v > "$OUT/seg1.log" 2>&1
echo "  seg1 rc=$? $(awk "BEGIN{printf \"%.1fs\", $(date +%s.%N)-$t0}") latent=$(ls -l $OUT/seg1.bin 2>/dev/null|awk '{print $5}')B"
grep -E "VACE|get_desc|generate_video completed" "$OUT/seg1.log" | tail -3

# tail = last K frames of seg1 as control context
n=0; for f in $(ls "$OUT"/seg1/f*.png 2>/dev/null|sort|tail -n $K); do cp "$f" "$(printf "$OUT/tail/c%03d.png" $n)"; n=$((n+1)); done

echo "=== SEG2 (continue via latent backdoor) ==="; t1=$(date +%s.%N)
ENVV=(-e VACE_CONT_FRAMES=$K -e VACE_CONT_LATENT=/src/perf_out/vace/seg1.bin)
run --control-video /src/perf_out/vace/tail -p "$P2" -o /src/perf_out/vace/seg2l/f%03d.png -v > "$OUT/seg2l.log" 2>&1
echo "  seg2 rc=$? $(awk "BEGIN{printf \"%.1fs\", $(date +%s.%N)-$t1}")"
grep -E "VACE continuation|VACE_CONT_LATENT|generate_video completed" "$OUT/seg2l.log" | tail -4
# stitch seg1 + seg2 (drop seg2's K overlap head)
mkdir -p "$OUT"/stitch; i=0
for f in $(ls "$OUT"/seg1/f*.png|sort); do cp "$f" "$(printf "$OUT/stitch/s%03d.png" $i)"; i=$((i+1)); done
for f in $(ls "$OUT"/seg2l/f*.png|sort|tail -n +$((K+1))); do cp "$f" "$(printf "$OUT/stitch/s%03d.png" $i)"; i=$((i+1)); done
docker run --rm -v "$REPO:/src" -w /src "$BUILDER" bash -lc "ffmpeg -y -framerate 16 -i /src/perf_out/vace/stitch/s%03d.png -c:v libx264 -pix_fmt yuv420p /src/perf_out/vace/chain.mp4 -loglevel error" 2>/dev/null
echo "=== stitched $i frames -> perf_out/vace/chain.mp4 ==="
