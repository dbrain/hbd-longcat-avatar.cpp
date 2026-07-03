#!/usr/bin/env bash
# MVP "see it all work" on the 3060: one Wan2.2-I2V-A14B scene shot + one InfiniteTalk
# lip-sync dub, both at a 480p portrait bucket (~LTX-comparable), distilled 4-step.
# Captures wall-time + peak VRAM (sampled from the host while the container renders).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"
BUILDER="longcat-avatar-dev:builder"
OUT="$REPO/mvp_out"; rm -rf "$OUT"; mkdir -p "$OUT"
RESULTS="$OUT/results.csv"; echo "render,wall_s,peak_vram_mib,rc" > "$RESULTS"

W=480; H=832; FR=81; SHIFT=7      # LTX-leaning portrait bucket (char.png 512x896 -> Wan 480p)
M=/models; DR=/models/_drive

# base docker args (CUDA libs live in the builder image; host has no CUDA)
BASE=(docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src)

sample_vram() { local f="$1"; echo 0 > "$f"
  while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$u" ] && { m=$(cat "$f"); [ "$u" -gt "$m" ] && echo "$u" > "$f"; }; sleep 0.3; done; }

run_timed() { # $1=name  $2..=full command line (docker ... image binary args)
  local name="$1"; shift
  local vf="$OUT/.vram_$name"; sample_vram "$vf" & local sp=$!
  local t0=$(date +%s.%N)
  "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  local t1=$(date +%s.%N)
  kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
  local wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  local peak=$(cat "$vf" 2>/dev/null); rm -f "$vf"
  echo "$name,$wall,$peak,$rc" >> "$RESULTS"
  echo ">> $name: ${wall}s  peak ${peak}MiB  rc=$rc"
}

# ============ 1) Wan2.2-I2V-A14B single scene shot ============
mkdir -p "$OUT/a14b"
PA="A young man with tousled dark brown hair and light stubble in a faded blue denim jacket over a white t-shirt sings energetically into the camera, bobbing his head and mouthing the words to an upbeat rock song, shoulders swaying with the beat. Locked static medium shot. Warm amber neon glows behind him outside a small corner bar at dusk, the damp street reflecting the lights, soft bokeh, cinematic, volumetric light, high detail."
run_timed a14b "${BASE[@]}" "$BUILDER" /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
  --high-noise-diffusion-model $M/wan22-i2v-a14b-high-q4_k.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
  -p "$PA" --cfg-scale 1 --high-noise-cfg-scale 1 \
  --sampling-method euler --high-noise-sampling-method euler \
  --steps 2 --high-noise-steps 2 --flow-shift $SHIFT \
  -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram 7.3 \
  --init-img $DR/char.png -o /src/mvp_out/a14b/f%03d.png -v
[ -n "$(ls "$OUT/a14b"/f*.png 2>/dev/null)" ] && \
  ffmpeg -y -framerate 16 -pattern_type glob -i "$OUT/a14b/*.png" -c:v libx264 -pix_fmt yuv420p "$OUT/a14b.mp4" -loglevel error

# ============ 2) InfiniteTalk lip-sync dub ============
PI="A young man with dark hair and a blue denim jacket singing, close-up portrait, warm amber neon bar lighting at dusk, cinematic, high detail."
run_timed infinitetalk "${BASE[@]}" -e IT_MAX_VRAM_GIB=7.3 "$BUILDER" /src/build/bin/sd-infinitetalk \
  --dit $M/infinitetalk-14b-q4_k.gguf --wav2vec $M/chinese-wav2vec2-base-f16.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --umt5 $M/longcat-umt5-xxl-q8_0.gguf \
  --image $DR/char.png --wav $DR/song_16k.wav --prompt "$PI" \
  --frames $FR --height $H --width $W --steps 4 --shift $SHIFT \
  --motion-frame 1 --max-windows 1 --distilled --out /src/mvp_out/it
[ -n "$(ls "$OUT/it"/*.png 2>/dev/null)" ] && \
  ffmpeg -y -framerate 25 -pattern_type glob -i "$OUT/it/*.png" -i "$DR/song_16k.wav" \
    -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$OUT/it.mp4" -loglevel error

echo "=== DONE ==="; cat "$RESULTS"
