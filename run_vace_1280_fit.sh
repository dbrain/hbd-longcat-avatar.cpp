#!/usr/bin/env bash
# 1280x704 VRAM-fit retry: the per-block DiT compute buffer is ~8.5GB at FR=21, so keep max-vram LOW
# (stream weights) to leave room. Gray cache already warm from the prior run. Try maxv2.5; one segment.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/final1280"; mkdir -p "$OUT/gcache"
M=/models; DR=/models/_drive; W=1280; H=704; FR="${FR:-21}"; MAXV="${MAXV:-2.5}"; SEED=42; FPS=16
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A young man with tousled dark brown hair and light stubble in a faded blue denim jacket over a white t-shirt sings energetically into the camera, bobbing his head, mouthing the lyrics, shoulders swaying to an upbeat rock beat. Static locked-off medium shot. Warm amber neon glows behind him outside a corner bar at dusk, the damp street reflecting the lights in soft bokeh. Cinematic, volumetric lighting, shallow depth of field, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.3; done; echo "$p">"$2"; }
D="$OUT/fit_mv${MAXV}_fr${FR}"; rm -rf "$D"; mkdir -p "$D"; t0=$(date +%s.%N)
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
  --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
  --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
  -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P" \
  -o /src/perf_out/final1280/fit_mv${MAXV}_fr${FR}/f%03d.png -v > "$D/run.log" 2>&1 &
pid=$!; poll $pid "$D/peak.txt" & pp=$!; wait $pid; rc=$?; wait $pp 2>/dev/null
enc=$(grep "encode_first_stage completed" "$D/run.log"|grep -oE "[0-9.]+ ms"|tail -1)
hi=$(grep "sampling(high noise) completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
lo=$(grep -E "stable-diffusion.cpp:[0-9]+ +- sampling completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
dec=$(grep "decode_first_stage completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
gen=$(grep "generate_video completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
maxbuf=$(grep "VACE-14B compute buffer size" "$D/run.log"|grep -oE "[0-9.]+ MB"|sort -gr|head -1)
echo "mv$MAXV fr$FR rc=$rc enc=${enc:-?} DiT_hi=${hi:-?} DiT_lo=${lo:-?} dec=${dec:-?} gen=${gen:-FAIL} maxDiTbuf=${maxbuf:-?} peak=$(cat $D/peak.txt 2>/dev/null)MiB wall=$(awk "BEGIN{printf \"%.1f\",$(date +%s.%N)-$t0}")s"
grep -iE "out of memory|CUDA error" "$D/run.log" | tail -2 || true
if [ "$rc" = 0 ]; then
  docker run --rm -v "$REPO:/src" -w /src "$BUILDER" bash -lc "ffmpeg -y -framerate $FPS -i /src/perf_out/final1280/fit_mv${MAXV}_fr${FR}/f%03d.png -c:v libx264 -pix_fmt yuv420p /src/perf_out/final1280/final_1280x704.mp4 -loglevel error" 2>/dev/null
  awk "BEGIN{g=${gen:-0}+0; sv=$FR/$FPS; if(g>0) printf \"THROUGHPUT: %.0f render-sec/sec-of-video (LTX-2.3=111)\n\", g/sv}"
fi
