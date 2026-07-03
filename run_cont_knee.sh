#!/usr/bin/env bash
# A2 throughput-knee + B3 seam eye-test: ONE scene rendered as a CHAINED continuation at 3+3, 1280x704.
# seg0 = t2v (banks tail latent), segN = VACE continuation from prior tail. Measures continuation seg
# time (control-path: real-tail control-encode + 8 vace_layers + DiT + decode) and preserves all frames
# for seam inspection. Parametric: FR (13/17/21), SEGS, K overlap. Gray cache pre-warmed in OUT/gcache.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"
FR="${FR:-13}"; SEGS="${SEGS:-3}"; K="${K:-5}"; MAXV="${MAXV:-7.3}"; SEED="${SEED:-42}"; W=1280; H=704; FPS=16
AGC="${AGC:-0}"; AGC_TARGET="${AGC_TARGET:-0.65}"   # FINDINGS-L12: AGC=1 breaks the continuation contrast ratchet
TAG="${TAG:-}"; [ -n "$TAG" ] && TAG="_$TAG"
OUT="$REPO/perf_out/contknee/fr${FR}${TAG}"; rm -rf "$OUT"; mkdir -p "$OUT"
GC="$REPO/perf_out/contknee/gcache"; mkdir -p "$GC"
M=/models
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="A vintage car drives up and rolls to a stop outside a neon-lit corner bar at dusk, headlights sweeping across the wet asphalt, tyres easing to a halt. Slow tracking shot alongside the car. Warm amber and magenta neon reflecting on the damp street, cinematic, volumetric light, shallow depth of field, high detail."
COMMON=(--vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1
  --sampling-method euler --high-noise-sampling-method euler --high-noise-steps ${HSTEPS:-3} --steps ${LSTEPS:-3}
  -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 -s $SEED
  --diffusion-model $VL --high-noise-diffusion-model $VH)
WD=(-e WAN_DISTILL_SIGMAS=1)
gen(){ docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src "${WD[@]}" "${ENVV[@]}" "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen "${COMMON[@]}" "$@"; }

echo "=== CONT-KNEE FR=$FR SEGS=$SEGS K=$K maxv=$MAXV 3+3 1280x704 ==="
rm -rf "$OUT/stitch"; mkdir -p "$OUT/stitch"; GIDX=0; t_all=$(date +%s.%N)
for seg in $(seq 0 $((SEGS-1))); do
  mkdir -p "$OUT/seg$seg"; t0=$(date +%s.%N)
  C="/src/perf_out/contknee/fr${FR}${TAG}"   # in-container path, matches host $OUT
  if [ "$seg" = 0 ]; then
    ENVV=(-e VACE_GRAY_CACHE_DIR=/src/perf_out/contknee/gcache -e VACE_SAVE_LATENT=$C/seg0.bin)
    gen -p "$P" -o $C/seg$seg/f%03d.png -v > "$OUT/seg$seg.log" 2>&1
  else
    prev=$((seg-1)); mkdir -p "$OUT/tail$seg"; n=0
    for f in $(ls "$OUT/seg$prev"/f*.png 2>/dev/null|sort|tail -n $K); do cp "$f" "$(printf "$OUT/tail$seg/c%03d.png" $n)"; n=$((n+1)); done
    # NOLATENT=1 => pure pixel-based VACE extension/"outpainting" (no raw-latent backdoor):
    # the K tail PNGs get VAE-encoded fresh (decode->re-encode roundtrip launders OOD latent drift).
    ENVV=(-e VACE_GRAY_CACHE_DIR=/src/perf_out/contknee/gcache -e VACE_CONT_FRAMES=$K
          -e VACE_CONT_AGC=$AGC -e VACE_CONT_AGC_TARGET=$AGC_TARGET
          -e VACE_SAVE_LATENT=$C/seg$seg.bin)
    [ "${NOLATENT:-0}" = "1" ] || ENVV+=(-e VACE_CONT_LATENT=$C/seg$prev.bin)
    gen --control-video $C/tail$seg -p "$P" -o $C/seg$seg/f%03d.png -v > "$OUT/seg$seg.log" 2>&1
  fi
  rc=$?
  wall=$(awk "BEGIN{printf \"%.0f\", $(date +%s.%N)-$t0}")
  gv=$(grep -oE 'generate_video completed in [0-9.]+s' "$OUT/seg$seg.log"|tail -1)
  # per-seg latent-std (the ratchet metric): max per-frame std from the predecode log
  pstd=$(grep -oE 'predecode latent\] frame [0-9]+ mean=[-0-9.]+ std=[0-9.]+' "$OUT/seg$seg.log" | grep -oE 'std=[0-9.]+' | sed 's/std=//' | sort -rn | head -1)
  agcline=$(grep -oE 'VACE_CONT_AGC: carried-tail std [0-9.]+ -> [0-9.]+ .gain [0-9.]+' "$OUT/seg$seg.log" | tail -1)
  echo "  seg$seg rc=$rc wall=${wall}s  $gv  maxFrameStd=${pstd:-?}  ${agcline}"
  # stitch
  if [ "$seg" = 0 ]; then start=1; else start=$((K+1)); fi
  for f in $(ls "$OUT/seg$seg"/f*.png 2>/dev/null|sort|tail -n +$start); do cp "$f" "$(printf "$OUT/stitch/g%04d.png" $GIDX)"; GIDX=$((GIDX+1)); done
done
ffmpeg -y -framerate $FPS -i "$OUT/stitch/g%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 16 "$OUT/chain_fr${FR}${TAG}.mp4" -loglevel error 2>/dev/null
echo "=== DONE FR=$FR AGC=$AGC${TAG:+ tag=$TAG}: $GIDX frames stitched, total $(awk "BEGIN{printf \"%.0f\", $(date +%s.%N)-$t_all}")s -> $OUT/chain_fr${FR}${TAG}.mp4 ==="
