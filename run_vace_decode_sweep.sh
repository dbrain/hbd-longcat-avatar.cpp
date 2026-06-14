#!/usr/bin/env bash
# Lever #3: VAE decode tiling sweep — decode-only (VACE_DECODE_LATENT loads a banked
# 6-frame latent, SKIPS the 84s DiT) so each config costs ~30-40s not ~111s.
# Baseline = overlap 0.5 / rel 0.25 = 42 tiles / ~27s. The avatar's lap-21 0.25-overlap
# win was never applied to VACE; the DiT expert is freed before decode so we have VRAM
# headroom for bigger/fewer tiles. Sweep overlap + tile size; measure decode wall + peak VRAM.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/decsweep"; mkdir -p "$OUT/gcache"
M=/models; DR=/models/_drive; W=480; H=832; FR=21; MAXV=7.3; SEED=42  # maxv7.3 = LTX-matched cap (DiT flat 6/7/8)
LAT="${LAT:-/src/perf_out/grayab/opt/seg1.bin}"
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf
VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P1="A young man with tousled dark brown hair sings into the camera, warm amber neon at dusk, cinematic, medium shot."

poll(){ local peak=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$peak" ]&&peak=$u; sleep 0.3; done; echo "$peak">"$2"; }

# $1=tag $2=tiling args...
dec(){
  local tag=$1; shift; local D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_DECODE_LATENT="$LAT" -e VACE_GRAY_CACHE_DIR=/src/perf_out/decsweep/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
    --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P1" \
    -o /src/perf_out/decsweep/$tag/f%03d.png -v "$@" > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!
  wait $pid; local rc=$?; wait $pp 2>/dev/null
  local dec=$(grep "decode_first_stage completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local tiles=$(grep "processing.*tiles" "$D/run.log"|tail -1|grep -oE "[0-9]+ tiles")
  local peak=$(cat "$D/peak.txt" 2>/dev/null)
  printf "  %-22s rc=%s  decode=%-7s tiles=%-10s peakVRAM=%sMiB\n" "$tag" "$rc" "${dec:-FAIL}" "${tiles:-?}" "$peak"
}

echo "=== VAE decode tiling sweep (decode-only, 480x832, 6 latent frames) ==="
dec base_ov50_r25   --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.5
dec ov25_r25        --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25
dec ov25_r50        --vae-tiling --temporal-tiling --vae-relative-tile-size 0.5x0.5   --vae-tile-overlap 0.25
dec ov25_r50_notmp  --vae-tiling                   --vae-relative-tile-size 0.5x0.5   --vae-tile-overlap 0.25
dec ov25_r100_notmp --vae-tiling                   --vae-relative-tile-size 1.0x1.0   --vae-tile-overlap 0.25
dec notiling                                                                                                  # full-frame decode if it fits
echo "=== done. eyeball: perf_out/decsweep/<tag>/f000.png ==="
