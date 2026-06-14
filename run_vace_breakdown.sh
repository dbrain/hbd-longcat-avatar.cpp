#!/usr/bin/env bash
# Phase-1 DIAGNOSTIC: dump the 1280x704 FR=13 DiT compute-buffer composition + VRAM breakdown.
# Gray cache is pre-warmed (perf_out/final1280/gcache). LONGCAT_VRAM_BREAKDOWN=1 logs per-phase
# driver_used + compute_buf + param working-set. Single segment, full -v log kept.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/breakdown"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=1280; H=704; FR="${FR:-13}"; MAXV="${MAXV:-7.3}"; SEED=42; FPS=16
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A young man with tousled dark brown hair and light stubble in a faded blue denim jacket over a white t-shirt sings energetically into the camera, bobbing his head, mouthing the lyrics, shoulders swaying to an upbeat rock beat. Static locked-off medium shot. Warm amber neon glows behind him outside a corner bar at dusk, the damp street reflecting the lights in soft bokeh. Cinematic, volumetric lighting, shallow depth of field, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.3; done; echo "$p">"$2"; }
D="$OUT/fr${FR}_mv${MAXV}"; rm -rf "$D"; mkdir -p "$D"; t0=$(date +%s.%N)
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  -e LONGCAT_VRAM_BREAKDOWN=1 -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
  --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
  --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
  -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P" \
  -o /src/perf_out/breakdown/fr${FR}_mv${MAXV}/f%03d.png -v > "$D/run.log" 2>&1 &
pid=$!; poll $pid "$D/peak.txt" & pp=$!; wait $pid; rc=$?; wait $pp 2>/dev/null
echo "=== rc=$rc peak=$(cat $D/peak.txt 2>/dev/null)MiB wall=$(awk "BEGIN{printf \"%.1f\",$(date +%s.%N)-$t0}")s ==="
echo "--- [VRAM] breakdown lines ---"; grep -E "\[VRAM\]" "$D/run.log"
echo "--- compute buffer sizes ---"; grep -iE "compute buffer size" "$D/run.log"
echo "--- DiT timing ---"; grep -E "sampling.*completed|generate_video completed|encode_first_stage completed|decode_first_stage completed" "$D/run.log" | tail -8
echo "--- graph cut / segments ---"; grep -iE "segment|graph.?cut|n_segments" "$D/run.log" | tail -5
grep -iE "out of memory|CUDA error" "$D/run.log" | tail -3 || true
echo "LOG: $D/run.log"
