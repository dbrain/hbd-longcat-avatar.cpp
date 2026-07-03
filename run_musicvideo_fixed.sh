#!/usr/bin/env bash
# 27s QUALITY MUSIC VIDEO — t2v hard-cuts @1280x704, FIXED lightx2v sigma grid (FINDINGS-L8).
# NO --init-img (no anime bleed). Distinct shots incl a face close-up to stress-test faces.
# Each take = 125s; ~32 takes -> ~27s @16fps -> ~65min. Harness-track this (run_in_background).
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/mv27"; rm -rf "$OUT"; mkdir -p "$OUT/stitch"
M=/models; W=1280; H=704; FR=13; MAXV=7.3; FPS=16
# Schedule fix is now wired into sd-cli: WAN_DISTILL_SIGMAS=1 auto-uses the proper lightx2v DMD grid
# at fixed shift 7 (FINDINGS-L8/L8c). No manual --sigmas needed. Override shift via WAN_DISTILL_SHIFT.
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
# Distinct shots (Subject -> concrete motion verbs -> camera -> lighting -> mood -> quality). [0]=face close-up stress test.
SHOTS=(
"Tight close-up of a man with dark stubble singing into a vintage microphone, eyes half-closed, mouthing lyrics, sweat glistening, slight head bob. Shallow depth of field. Warm amber stage light, dark background, cinematic, high detail, sharp focus on the face."
"A vintage car rolls to a stop outside a neon-lit corner bar at dusk, headlights sweeping the wet asphalt, tyres easing to a halt. Slow tracking shot alongside the car. Warm amber and magenta neon reflecting on the damp street, cinematic, volumetric light, shallow depth of field, high detail."
"A man swings the car door open, steps out and sings toward the camera, gesturing to the beat, the door swinging shut behind him. Handheld medium shot, slow push-in. Warm amber neon glow from the bar sign, rain-slick reflective street, cinematic, volumetric light, high detail."
"A man pushes open the door of a dim empty bar and walks inside past rows of glowing bottles, dust drifting in a shaft of light. Camera dollies in behind him through the doorway. Moody warm tungsten light, deep shadows, cinematic, volumetric light, high detail."
"A man sits on a stool at an empty bar counter singing energetically, shoulders bouncing, tapping the counter to the beat. Static medium shot. Warm tungsten light over the bar, glowing bottles behind him, cinematic, shallow depth of field, high detail."
"Medium close-up of a woman with curly hair singing backup, swaying side to side, snapping her fingers, smiling. Soft rim light, blurred neon bokeh behind. Cinematic, shallow depth of field, high detail."
"A drummer plays an energetic beat in a smoky bar, arms blurring with motion, cymbals flashing under a spotlight. Dynamic medium shot. Moody warm light, volumetric haze, cinematic, high detail."
"A man walks down a rain-slick neon street at night singing to the camera, reflections shimmering, breath visible in the cold air. Slow steadicam tracking shot. Saturated cyan and magenta neon, cinematic, volumetric light, shallow depth of field, high detail."
)
SEEDS=(42 7 123 2024)   # 8 shots x 4 seeds = 32 takes ~= 27s
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.5; done; echo "$p">"$2"; }
GIDX=0; t_all=$(date +%s.%N); NTAKE=0
for si in "${!SHOTS[@]}"; do
  P="${SHOTS[$si]}"
  for seed in "${SEEDS[@]}"; do
    D="$OUT/s${si}_seed${seed}"; mkdir -p "$D"; t0=$(date +%s.%N)
    docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
      -e WAN_DISTILL_SIGMAS=1 -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
      /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
      --sampling-method euler --high-noise-sampling-method euler --high-noise-steps ${HSTEPS:-3} --steps ${LSTEPS:-3} \
      -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
      --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
      -s $seed --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
      -o /src/perf_out/mv27/s${si}_seed${seed}/f%03d.png -v > "$D/run.log" 2>&1
    rc=$?; gen=$(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1)
    nf=$(ls "$D"/f*.png 2>/dev/null|wc -l); NTAKE=$((NTAKE+1))
    echo "  take $NTAKE/32 shot$si seed$seed rc=$rc frames=$nf $gen $(awk "BEGIN{printf \"%.0fs\",$(date +%s.%N)-$t0}")"
    for f in $(ls "$D"/f*.png 2>/dev/null|sort); do cp "$f" "$(printf "$OUT/stitch/g%04d.png" $GIDX)"; GIDX=$((GIDX+1)); done
  done
done
echo "=== stitched $GIDX frames (~$(awk "BEGIN{printf \"%.1f\",$GIDX/$FPS}")s); muxing ==="
ffmpeg -y -framerate $FPS -i "$OUT/stitch/g%04d.png" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/musicvideo_27s.mp4" -loglevel error 2>/dev/null \
  && echo "DONE: $OUT/musicvideo_27s.mp4 ($GIDX frames @${FPS}fps) total $(awk "BEGIN{printf \"%.0fmin\",($(date +%s.%N)-$t_all)/60}")" \
  || echo "MUX FAILED"
