#!/usr/bin/env bash
# SPECKLE test: does MORE low-noise cleanup kill the "dit dotty" grain? Hold 2 HIGH steps (trained
# structure phase), ladder LOW-noise steps by appending finer trained-anchor tail sigmas (shift7 linear:
# t1000=1.0 t750=0.95455 t500=0.875 t250=0.70 t125=0.50 t62=0.31633 t31=0.18297). Dark prompt to reveal dots.
# Plus one re-derived 8-step hedge. Same seed. Zoom shadows + strip for comparison.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/speckle"; rm -rf "$OUT"; mkdir -p "$OUT/gcache"
M=/models; W=1280; H=704; FR=13; MAXV=7.3; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A man stands in a dark dim room lit by a single weak warm lamp, deep soft shadows, smooth dark walls and floor, faint volumetric haze, moody cinematic, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
# $1=tag $2=extra args
run(){ local tag=$1; shift; local D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/speckle/gcache "${ENVV[@]}" "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler \
    "$@" -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
    -o /src/perf_out/speckle/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  cp "$D/f006.png" "$OUT/${tag}_mid.png" 2>/dev/null
  # zoom a dark region (lower-left shadow) at true pixels
  ffmpeg -y -i "$D/f006.png" -vf "crop=420:300:60:380,scale=840:600:flags=neighbor" "$OUT/${tag}_dark.png" -loglevel error 2>/dev/null || true
  local gen=$(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1)
  echo "  $tag rc=$rc $gen peak=$(cat $D/peak.txt 2>/dev/null)MiB $(awk "BEGIN{printf \"%.0fs\",$(date +%s.%N)-$t0}")"
}
echo "=== SPECKLE ladder (2 high + N low), seed $SEED, dark prompt ==="
ENVV=();                run base_2h2l --sigmas "1.0,0.95455,0.875,0.70,0.0"                              --high-noise-steps 2 --steps 2
ENVV=();                run add_2h3l  --sigmas "1.0,0.95455,0.875,0.70,0.50,0.0"                         --high-noise-steps 2 --steps 3
ENVV=();                run add_2h4l  --sigmas "1.0,0.95455,0.875,0.70,0.50,0.31633,0.0"                 --high-noise-steps 2 --steps 4
ENVV=();                run add_2h5l  --sigmas "1.0,0.95455,0.875,0.70,0.50,0.31633,0.18297,0.0"         --high-noise-steps 2 --steps 5
ENVV=(-e WAN_DISTILL_SIGMAS=1); run redrive8 --high-noise-steps 4 --steps 4
echo "=== building comparison strips (mid + dark-zoom) ==="
ffmpeg -y $(for t in base_2h2l add_2h3l add_2h4l add_2h5l redrive8; do echo -n "-i $OUT/${t}_mid.png "; done) -filter_complex "scale=380:-1,hstack=inputs=5" "$OUT/STRIP_mid.png" -loglevel error 2>/dev/null || true
ffmpeg -y $(for t in base_2h2l add_2h3l add_2h4l add_2h5l redrive8; do echo -n "-i $OUT/${t}_dark.png "; done) -filter_complex "hstack=inputs=5" "$OUT/STRIP_dark.png" -loglevel error 2>/dev/null || true
echo "=== done. strips: perf_out/speckle/STRIP_{mid,dark}.png (order: base|+1low|+2low|+3low|redrive8) ==="
