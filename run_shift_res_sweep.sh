#!/usr/bin/env bash
# SHIFT x RESOLUTION calibration sweep — find best flow-shift per resolution for the lightx2v
# Wan2.2 distill, to calibrate wan_distill_res_shift() anchors (FINDINGS-L8b). For each res, render
# the face close-up at shifts {3,5,7,9,11} using the proper DMD grid (build_longcat_dmd_sigmas via
# --sigmas, computed in awk). Hold MoE split 2+2 (switches at 0.875 boundary). FR=5 (fast). seed 42.
# Eyeball perf_out/shiftsweep/<res>_sh<shift>_mid.png for sharpness/detail; pick the knee per res.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/shiftsweep"; rm -rf "$OUT"; mkdir -p "$OUT/gcache"
M=/models; FR=5; MAXV=7.3; SEED=42
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
P="Tight close-up of a man with dark stubble singing into a vintage microphone, eyes half-closed, mouthing lyrics, sweat glistening. Shallow depth of field. Warm amber stage light, dark background, cinematic, high detail, sharp focus on the face."
RES=("768x432" "640x640" "1280x720")   # ~420p, square, 720p
SHIFTS=(3 5 7 9 11)
# DMD 4-step grid for a given shift (raw seq for steps=4: 1.0,0.74975,0.4995,0.24925), terminal 0.
sigmas_for(){ awk -v S=$1 'BEGIN{n=split("1.0 0.74975 0.4995 0.24925",r," "); o=""; for(i=1;i<=n;i++){s=S*r[i]/(1+(S-1)*r[i]); o=o sprintf("%.5f,",s)} print o "0.0"}'; }
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.4; done; echo "$p">"$2"; }
echo "=== SHIFT x RES sweep (FR=$FR, seed $SEED, face close-up) ==="
for res in "${RES[@]}"; do
  W=${res%x*}; H=${res#*x}
  for sh in "${SHIFTS[@]}"; do
    SIG=$(sigmas_for $sh); tag="${W}x${H}_sh${sh}"; D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"; t0=$(date +%s.%N)
    docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
      -e VACE_GRAY_CACHE_DIR=/src/perf_out/shiftsweep/gcache "$BUILDER" \
      /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
      --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
      --sigmas "$SIG" -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
      --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
      -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH -p "$P" \
      -o /src/perf_out/shiftsweep/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
    pid=$!; poll $pid "$D/peak.txt" & pp=$!; wait $pid; rc=$?; wait $pp 2>/dev/null
    mid=$(ls "$D"/f*.png 2>/dev/null | sed -n '2p'); [ -z "$mid" ] && mid=$(ls "$D"/f*.png 2>/dev/null|head -1)
    [ -n "$mid" ] && cp "$mid" "$OUT/${tag}_mid.png"
    gen=$(grep -oE 'generate_video completed in [0-9.]+s' "$D/run.log"|tail -1)
    echo "  $tag rc=$rc sigmas=[$SIG] $gen peak=$(cat $D/peak.txt 2>/dev/null)MiB $(awk "BEGIN{printf \"%.0fs\",$(date +%s.%N)-$t0}")"
    grep -iE "out of memory|CUDA error" "$D/run.log" | tail -1 || true
  done
done
echo "=== done. eyeball perf_out/shiftsweep/{768x432,640x640,1280x720}_sh{3,5,7,9,11}_mid.png ==="
