#!/usr/bin/env bash
# Deep --set full profile of the hot Q4_K DiT matmul. Minimal launch count (GPU-economy):
# capture just the first 1 step's worth of mul_mat_q launches with all sections.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=480; H=832; FR="${FR:-13}"
LABEL="${LABEL:-matmul_full}"; CNT="${CNT:-3}"
PA="A young man sings into the camera. Static medium shot. Warm neon, cinematic."
# -c CNT: profile only first CNT matched launches. -k regex:mul_mat_q : only the Q4_K matmul.
# --set full : all sections (WarpState/Scheduler/Instruction/Memory/SASS).
docker run --rm --gpus all --cap-add SYS_ADMIN -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  "$BUILDER" \
  ncu --set full -k "regex:mul_mat_q" -c "$CNT" --target-processes all \
      --kernel-name-base demangled --log-file "/src/perf_out/${LABEL}.log" \
  /src/build/bin/sd-cli -M vid_gen \
    --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
    --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
    -p "$PA" --cfg-scale 1 --sampling-method euler --steps 1 --flow-shift 7 \
    -W $W -H $H --video-frames $FR \
    --diffusion-fa --clip-on-cpu --vae-tiling --vae-relative-tile-size 0.25x0.25 --temporal-tiling \
    --init-img $DR/char.png -o /src/perf_out/${LABEL}.frames/f%03d.png -v \
  > "$OUT/${LABEL}.run.log" 2>&1
echo "rc=$? log=$OUT/${LABEL}.log"
