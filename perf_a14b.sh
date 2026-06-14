#!/usr/bin/env bash
# A14B perf sweep harness. Env knobs:
#   LABEL  (required) - tag for this run
#   FR=21             - video frames
#   OFFLOAD=1|0       - --offload-to-cpu (1) vs resident DiT (0)
#   TEMPORAL=0|1      - --temporal-tiling on VAE
#   MAXV=4.5          - --max-vram (only matters when OFFLOAD=1)
#   TILE=4x4          - --vae-relative-tile-size
# Logs peak VRAM, wall, DiT s/it, VAE encode/decode buffer to perf_out/<label>.log + perf_out/sweep.csv
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"
OUT="$REPO/perf_out"; mkdir -p "$OUT"
LABEL="${LABEL:?need LABEL}"; FR="${FR:-21}"; OFFLOAD="${OFFLOAD:-1}"; TEMPORAL="${TEMPORAL:-0}"; MAXV="${MAXV:-4.5}"; TILE="${TILE:-4x4}"
SINGLE="${SINGLE:-0}"; PINNED="${PINNED:-0}"
W=480; H=832; SHIFT=7; M=/models; DR=/models/_drive
FRAMEDIR="$OUT/$LABEL.frames"; rm -rf "$FRAMEDIR"; mkdir -p "$FRAMEDIR"

flags=(--diffusion-fa --vae-tiling --vae-relative-tile-size "$TILE")
# single-expert (resident-friendly) omits the high-noise expert
[ "$SINGLE" = 0 ] && flags+=(--high-noise-diffusion-model $M/wan22-i2v-a14b-high-q4_k.gguf --high-noise-cfg-scale 1 --high-noise-sampling-method euler --high-noise-steps 2)
[ "$OFFLOAD" = 1 ] && flags+=(--offload-to-cpu --mmap --max-vram "$MAXV")
[ "$TEMPORAL" = 1 ] && flags+=(--temporal-tiling)
[ "${CLIPCPU:-0}" = 1 ] && flags+=(--clip-on-cpu)
[ "${VAECPU:-0}" = 1 ] && flags+=(--vae-on-cpu)
denv=(); [ "$PINNED" = 1 ] && denv+=(-e LONGCAT_DIT_NO_MMAP=1)
# arbitrary env passthrough, e.g. GENV="GGML_CUDA_FORCE_MMQ=1 GGML_CUDA_FORCE_CUBLAS=1"
for kv in ${GENV:-}; do denv+=(-e "$kv"); done
# step count: MoE splits 2 high + 2 low; single expert runs all 4 itself
STEPS=2; [ "$SINGLE" = 1 ] && STEPS=4

vf="$OUT/.vram_$LABEL"; echo 0 > "$vf"
( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1)
    [ -n "$u" ] && { m=$(cat "$vf"); [ "$u" -gt "$m" ] && echo "$u">"$vf"; }; sleep 0.3; done ) & sp=$!
t0=$(date +%s.%N)
PA="A young man with tousled dark brown hair and light stubble in a faded blue denim jacket sings energetically into the camera, bobbing his head to an upbeat rock song. Locked static medium shot. Warm amber neon glows behind him outside a small corner bar at dusk, cinematic, high detail."
docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src "${denv[@]}" "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
  --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
  -p "$PA" --cfg-scale 1 \
  --sampling-method euler \
  --steps $STEPS --flow-shift $SHIFT -W $W -H $H --video-frames $FR \
  "${flags[@]}" --init-img $DR/char.png -o "/src/perf_out/$LABEL.frames/f%03d.png" -v > "$OUT/$LABEL.log" 2>&1
rc=$?; t1=$(date +%s.%N)
kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"); peak=$(cat "$vf"); rm -f "$vf"
# parse DiT s/it (median of the s/it tokens) + VAE buffers
sit=$(grep -oE "[0-9.]+s/it" "$OUT/$LABEL.log" | grep -oE "[0-9.]+" | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+0]}')
encbuf=$(grep -A0 "wan_vae compute buffer" "$OUT/$LABEL.log" | grep -oE "[0-9.]+ MB" | head -1 | grep -oE "[0-9.]+")
decbuf=$(grep "wan_vae compute buffer" "$OUT/$LABEL.log" | tail -1 | grep -oE "[0-9.]+ MB" | grep -oE "[0-9.]+")
ditbuf=$(grep "Wan2.2-I2V-14B compute buffer" "$OUT/$LABEL.log" | grep -oE "[0-9.]+ MB" | sort -rn | head -1 | grep -oE "[0-9.]+")
[ -f "$OUT/sweep.csv" ] || echo "label,fr,offload,temporal,tile,wall_s,peak_mib,dit_s_it,dit_buf_mib,vae_enc_mib,vae_dec_mib,rc" > "$OUT/sweep.csv"
echo "$LABEL,$FR,$OFFLOAD,$TEMPORAL,$TILE,$wall,$peak,${sit:-?},${ditbuf:-?},${encbuf:-?},${decbuf:-?},$rc" >> "$OUT/sweep.csv"
echo ">> $LABEL: wall=${wall}s peak=${peak}MiB dit=${sit:-?}s/it ditbuf=${ditbuf}MB vaeENC=${encbuf}MB vaeDEC=${decbuf}MB rc=$rc"
column -t -s, "$OUT/sweep.csv"
