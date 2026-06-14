#!/usr/bin/env bash
# QUALITY PASS (not a benchmark): full LTX music-video shot list as Wan2.2 t2v, 1280x704, chained
# segments per scene (tests intra-scene continuity), stitched to ONE watchable mp4 the user can judge
# (faces fuzzy/warping? motion natural? do segments connect?). t2v = NO --init-img (no anime-face bleed).
# Steps = 4 (DMD distill native). Config from pre-flight (maxv/FR). Run while out — long.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/musicvideo"; mkdir -p "$OUT/gcache"
M=/models; W=1280; H=704; FR="${FR:-13}"; MAXV="${MAXV:-7.3}"; SEED=42; FPS=16; K="${K:-5}"
SEGS_PER_SCENE="${SEGS_PER_SCENE:-2}"   # chained segments per scene (2 => ~1.3s/scene; 3 => ~2s)
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
# LTX shot list ported to Wan2.2 t2v (Scene+Subject -> Motion[concrete verbs] -> Camera -> Lighting -> Mood -> Quality)
SCENES=(
"A vintage car drives up and rolls to a stop outside a neon-lit corner bar at dusk, headlights sweeping across the wet asphalt, tyres easing to a halt. Slow tracking shot alongside the car. Warm amber and magenta neon reflecting on the damp street, cinematic, volumetric light, shallow depth of field, high detail."
"A man swings the car door open, steps out and sings toward the camera, gesturing to the beat, the door swinging shut behind him. Handheld medium shot with a slow push-in. Warm amber neon glow from the bar sign, rain-slick reflective street, cinematic, volumetric light, high detail."
"A man pushes open the door of a dim empty bar and walks inside past rows of glowing bottles, dust drifting in a shaft of light. Camera dollies in behind him through the doorway. Moody warm tungsten light, deep shadows, cinematic, volumetric light, high detail."
"A man sits on a stool at an empty bar counter singing energetically while eating chips from a bowl, crumbs flying from his mouth on the beat, shoulders bouncing. Static medium shot. Warm tungsten light over the bar, glowing bottles behind him, cinematic, shallow depth of field, high detail."
)
# 3+3 schedule (FINDINGS-L8h): WAN_DISTILL_SIGMAS=1 (added per-call via WD env below) + 3 high + 3 low steps,
# no --flow-shift (the wired grid sets shift 7). Override via HSTEPS/LSTEPS. Applies to seg0 AND continuations.
COMMON=(--vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1
  --sampling-method euler --high-noise-sampling-method euler --high-noise-steps ${HSTEPS:-3} --steps ${LSTEPS:-3}
  -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 -s $SEED
  --diffusion-model $VL --high-noise-diffusion-model $VH)
WD=(-e WAN_DISTILL_SIGMAS=1)   # prepend to every ENVV so the fixed DMD grid is used
gen(){ docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src "${WD[@]}" "${ENVV[@]}" "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen "${COMMON[@]}" "$@"; }

rm -rf "$OUT/stitch"; mkdir -p "$OUT/stitch"; GIDX=0; t_all=$(date +%s.%N)
for si in "${!SCENES[@]}"; do
  P="${SCENES[$si]}"; SD="$OUT/scene$si"; rm -rf "$SD"; mkdir -p "$SD"
  echo "=== SCENE $si: ${P:0:60}... ==="
  for seg in $(seq 0 $((SEGS_PER_SCENE-1))); do
    mkdir -p "$SD/seg$seg"
    t0=$(date +%s.%N)
    if [ "$seg" = 0 ]; then
      # free t2v (no init, no control), bank tail latent
      ENVV=(-e VACE_GRAY_CACHE_DIR=/src/perf_out/musicvideo/gcache -e VACE_SAVE_LATENT=/src/perf_out/musicvideo/scene$si/seg0.bin)
      gen -p "$P" -o /src/perf_out/musicvideo/scene$si/seg$seg/f%03d.png -v > "$SD/seg$seg.log" 2>&1
    else
      # continuation from prior seg tail (intra-scene continuity)
      prev=$((seg-1)); mkdir -p "$SD/tail$seg"; n=0
      for f in $(ls "$SD/seg$prev"/f*.png 2>/dev/null|sort|tail -n $K); do cp "$f" "$(printf "$SD/tail$seg/c%03d.png" $n)"; n=$((n+1)); done
      ENVV=(-e VACE_GRAY_CACHE_DIR=/src/perf_out/musicvideo/gcache -e VACE_CONT_FRAMES=$K
            -e VACE_CONT_LATENT=/src/perf_out/musicvideo/scene$si/seg$prev.bin
            -e VACE_SAVE_LATENT=/src/perf_out/musicvideo/scene$si/seg$seg.bin)
      gen --control-video /src/perf_out/musicvideo/scene$si/tail$seg -p "$P" -o /src/perf_out/musicvideo/scene$si/seg$seg/f%03d.png -v > "$SD/seg$seg.log" 2>&1
    fi
    rc=$?; echo "  scene$si seg$seg rc=$rc $(awk "BEGIN{printf \"%.0fs\", $(date +%s.%N)-$t0}") $(grep -oE 'generate_video completed in [0-9.]+s' "$SD/seg$seg.log"|tail -1)"
    # stitch: seg0 all frames, continuations drop K-overlap head
    if [ "$seg" = 0 ]; then start=1; else start=$((K+1)); fi
    for f in $(ls "$SD/seg$seg"/f*.png 2>/dev/null|sort|tail -n +$start); do cp "$f" "$(printf "$OUT/stitch/g%04d.png" $GIDX)"; GIDX=$((GIDX+1)); done
  done
done
echo "=== stitched $GIDX frames; muxing (host ffmpeg) ==="
ffmpeg -y -framerate $FPS -i "$OUT/stitch/g%04d.png" -c:v libx264 -pix_fmt yuv420p "$OUT/musicvideo_1280x704.mp4" -loglevel error 2>/dev/null && \
  echo "DONE: $OUT/musicvideo_1280x704.mp4 ($GIDX frames, ${FPS}fps = $(awk "BEGIN{printf \"%.1f\", $GIDX/$FPS}")s) total $(awk "BEGIN{printf \"%.0f\", $(date +%s.%N)-$t_all}")s"
