#!/usr/bin/env bash
# Per-kernel duration decomposition of a full generate (single-pass, cheap).
# Buckets gpu__time_duration.sum by demangled kernel name -> what % of the DiT step
# is the Q4_K matmul vs FA-attention vs norm/rope/copy/elementwise.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=480; H=832; FR="${FR:-13}"
LABEL="${LABEL:-decomp}"; CSV="/src/perf_out/${LABEL}.ncu.csv"
PA="A young man with tousled dark brown hair sings into the camera. Locked static medium shot. Warm amber neon, cinematic."
# single low expert, resident, temporal-tiled VAE — mirrors the basic-ncu config
docker run --rm --gpus all --cap-add SYS_ADMIN -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  "$BUILDER" \
  ncu --metrics gpu__time_duration.sum --csv --target-processes all \
      --kernel-name-base demangled --replay-mode kernel \
      --log-file "$CSV" \
  /src/build/bin/sd-cli -M vid_gen \
    --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
    --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
    -p "$PA" --cfg-scale 1 --sampling-method euler --steps 2 --flow-shift 7 \
    -W $W -H $H --video-frames $FR \
    --diffusion-fa --clip-on-cpu --vae-tiling --vae-relative-tile-size 0.25x0.25 --temporal-tiling \
    --init-img $DR/char.png -o /src/perf_out/${LABEL}.frames/f%03d.png -v \
  > "$OUT/${LABEL}.run.log" 2>&1
echo "rc=$? csv=$OUT/${LABEL}.ncu.csv log=$OUT/${LABEL}.run.log"
wc -l "$OUT/${LABEL}.ncu.csv" 2>/dev/null
