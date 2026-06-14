#!/usr/bin/env bash
# PRINCIPLED FIX TEST: drive the lightx2v distill with its DOCUMENTED schedule via --sigmas.
# Smoke baseline used the generic DiscreteScheduler (t=[999,666,333,0] = wasted step, wrong grid).
# lightx2v doc: 4-step t=[1000,750,500,250], shift 5, cfg 1. We inject exact sigmas. Same seed/prompt.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/sigfix"; mkdir -p "$OUT"
M=/models; W=1280; H=704; FR=13; MAXV=7.3; SEED=42; FPS=16
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A man swings a vintage car door open, steps out and sings toward the camera, gesturing to the beat. Handheld medium shot with a slow push-in. Warm amber neon glow from a bar sign at dusk, rain-slick reflective street, cinematic, volumetric light, shallow depth of field, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
# $1=tag $2=sigmas
run(){ local tag=$1 sig=$2; local D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
    --sigmas "$sig" -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/sigfix/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  local gen=$(grep "generate_video completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  cp "$D/f006.png" "$OUT/${tag}_mid.png" 2>/dev/null
  ffmpeg -y -i "$D/f006.png" -vf 'crop=320:320:440:90,scale=640:640:flags=lanczos' "$OUT/${tag}_face.png" -loglevel error 2>/dev/null || true
  echo "  $tag rc=$rc gen=${gen:-FAIL} sigmas=[$sig] $(grep -oE 'switching from high noise model at step [0-9]+' $D/run.log|tail -1)"
}
echo "=== SIGMA-FIX TEST 1280x704 t2v (seed 42) ==="
run C_grid_sh7  "1.0,0.9545,0.875,0.699,0.0"   # proper grid, shift 7 (vs smoke wrong grid same shift)
run B_lx2v_sh5  "1.0,0.9375,0.8333,0.625,0.0"  # exact lightx2v documented (shift 5)
echo "=== compare perf_out/sigfix/{C_grid_sh7,B_lx2v_sh5}_{mid,face}.png vs perf_out/t2v_smoke/f006.png ==="
