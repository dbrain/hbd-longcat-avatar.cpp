#!/usr/bin/env bash
# FR ceiling probe @1280x704 t2v 3+3: what's the max frames-per-segment that FITS today (before buffer-shrink)?
# Reports fit/OOM + peak VRAM + DiT compute-buffer reserve + output frame count per FR.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/frceiling"; rm -rf "$OUT"; mkdir -p "$OUT/gcache"
M=/models; W=1280; H=704; MAXV=7.3; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A man sings into the camera on a neon street at dusk, cinematic, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
run(){ local fr=$1; local D="$OUT/fr$fr"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e WAN_DISTILL_SIGMAS=1 -e LONGCAT_VRAM_BREAKDOWN=1 -e VACE_GRAY_CACHE_DIR=/src/perf_out/frceiling/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --high-noise-steps 3 --steps 3 \
    -W $W -H $H --video-frames $fr --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/frceiling/fr$fr/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  local oom=$(grep -ciE "out of memory|CUDA error" "$D/run.log"); local stat="FIT"; [ "$rc" != 0 ] && stat="rc=$rc"; [ "$oom" != 0 ] && stat="OOM"
  local buf=$(grep -oE "VACE-14B compute buffer size: [0-9.]+ MB" "$D/run.log"|grep -oE "[0-9.]+ MB"|sort -gr|head -1)
  local nf=$(ls "$D"/f*.png 2>/dev/null|wc -l); local gen=$(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1)
  echo "  FR=$fr -> $stat | out_frames=$nf (${gen:-no gen}) | maxDiTbuf=${buf:-?} | peak=$(cat $D/peak.txt 2>/dev/null)MiB | $(awk "BEGIN{printf \"%.0fs\",$(date +%s.%N)-$t0}")"
}
echo "=== FR ceiling @1280x704 t2v 3+3 (latent frames=(FR-1)/4+1) ==="
for fr in 17 21 25; do run $fr; done
echo "=== done (FR=13 known FIT @ ~9.4GB peak; LTX does ~90f/seg) ==="
