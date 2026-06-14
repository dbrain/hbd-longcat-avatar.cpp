#!/usr/bin/env bash
# InfiniteTalk retry: scaled to fit 12GB (81f OOM'd->segfault). Line-buffered log via stdbuf
# so a crash leaves a trail. NOTE: sd-infinitetalk loads its 10.7GB DiT into ANONYMOUS CPU RAM
# (no mmap) -> ~17GB hard RSS; watch host memory vs prod.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"
OUT="$REPO/mvp_out"; mkdir -p "$OUT/it"
W=${W:-480}; H=${H:-832}; FR=${FR:-21}; SHIFT=${SHIFT:-7}; MAXV=${MAXV:-5.0}
M=/models; DR=/models/_drive
BASE=(docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src -e IT_MAX_VRAM_GIB=$MAXV)

sample_vram() { local f="$1"; echo 0 > "$f"
  while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$u" ] && { m=$(cat "$f"); [ "$u" -gt "$m" ] && echo "$u" > "$f"; }; sleep 0.3; done; }

vf="$OUT/.vram_it"; sample_vram "$vf" & sp=$!
t0=$(date +%s.%N)
PI="A young man with dark hair and a blue denim jacket singing, close-up portrait, warm amber neon bar lighting at dusk, cinematic, high detail."
"${BASE[@]}" "$BUILDER" stdbuf -oL -eL /src/build/bin/sd-infinitetalk \
  --dit $M/infinitetalk-14b-q4_k.gguf --wav2vec $M/chinese-wav2vec2-base-f16.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --umt5 $M/longcat-umt5-xxl-q8_0.gguf \
  --image $DR/char.png --wav $DR/song_16k.wav --prompt "$PI" \
  --frames $FR --height $H --width $W --steps 4 --shift $SHIFT \
  --motion-frame 1 --max-windows 1 --distilled --out /src/mvp_out/it > "$OUT/it.log" 2>&1
rc=$?; t1=$(date +%s.%N)
kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"); peak=$(cat "$vf"); rm -f "$vf"
grep -v '^infinitetalk,' "$OUT/results.csv" > "$OUT/results.csv.tmp" 2>/dev/null; mv "$OUT/results.csv.tmp" "$OUT/results.csv"
echo "infinitetalk,$wall,$peak,$rc" >> "$OUT/results.csv"
echo ">> infinitetalk ${W}x${H} ${FR}f: ${wall}s  peak ${peak}MiB  rc=$rc"
if [ -n "$(ls "$OUT/it"/*.png 2>/dev/null)" ]; then
  ffmpeg -y -framerate 25 -pattern_type glob -i "$OUT/it/*.png" -i "$DR/song_16k.wav" \
    -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$OUT/it.mp4" -loglevel error
fi
cat "$OUT/results.csv"
