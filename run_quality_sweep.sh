#!/usr/bin/env bash
# QUALITY DISAMBIGUATION: is the murk from under-stepping or wrong flow-shift (NOT the quant)?
# Same seed/prompt as the t2v smoke (1280x704 FR=13, t2v, no init-img). Vary steps + shift. Eyeball mid-frame.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/qual"; mkdir -p "$OUT"
M=/models; W=1280; H=704; FR=13; MAXV=7.3; SEED=42; FPS=16
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A man swings a vintage car door open, steps out and sings toward the camera, gesturing to the beat. Handheld medium shot with a slow push-in. Warm amber neon glow from a bar sign at dusk, rain-slick reflective street, cinematic, volumetric light, shallow depth of field, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
# $1=tag $2=high_steps $3=low_steps $4=shift
run(){ local tag=$1 hs=$2 ls=$3 sh=$4; local D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --steps $ls --high-noise-steps $hs \
    --flow-shift $sh -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/qual/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  local gen=$(grep "generate_video completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  # zoom the face region of the mid frame for sharpness read
  cp "$D/f006.png" "$OUT/${tag}_mid.png" 2>/dev/null
  ffmpeg -y -i "$D/f006.png" -vf 'crop=320:320:440:90,scale=640:640:flags=lanczos' "$OUT/${tag}_face.png" -loglevel error 2>/dev/null || true
  printf "  %-10s hs=%s ls=%s shift=%s rc=%s gen=%-8s peak=%sMiB wall=%.0fs\n" \
    "$tag" "$hs" "$ls" "$sh" "$rc" "${gen:-FAIL}" "$(cat $D/peak.txt 2>/dev/null)" "$(awk "BEGIN{print $(date +%s.%N)-$t0}")"
}
echo "=== QUALITY SWEEP 1280x704 t2v (seed 42) ==="
run s8_sh7  4 4 7    # 2x steps, same shift -> tests under-stepping
run s4_sh5  2 2 5    # baseline steps, lower shift -> tests shift
run s8_sh5  4 4 5    # more steps + lower shift
run s16_sh7 8 8 7    # ceiling steps -> if ~= baseline, distill is fine
echo "=== done. compare perf_out/qual/{baseline=t2v_smoke, s8_sh7,s4_sh5,s8_sh5,s16_sh7}_{mid,face}.png ==="
