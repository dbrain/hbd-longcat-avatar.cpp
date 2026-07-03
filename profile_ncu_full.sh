#!/usr/bin/env bash
# ncu --set full on BOTH the hot Q4_K matmul and flash_attn, capturing ~one DiT block in context.
# Minimal: single low expert, FR=13, 1 step. SYS_ADMIN for HW counters.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out"; mkdir -p "$OUT"
M=/models; DR=/models/_drive; W=480; H=832; FR="${FR:-13}"; LABEL="${LABEL:-ncu_full}"; CNT="${CNT:-14}"
PA="A young man sings into the camera. Static medium shot. Warm neon, cinematic."
docker run --rm --gpus all --cap-add SYS_ADMIN -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  "$BUILDER" \
  ncu --set full -k "regex:flash_attn_ext_f16|mul_mat_q" -c "$CNT" --target-processes all \
      --kernel-name-base demangled --replay-mode kernel --log-file "/src/perf_out/${LABEL}.log" \
  /src/build/bin/sd-cli -M vid_gen \
    --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
    --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
    -p "$PA" --cfg-scale 1 --sampling-method euler --steps 1 --flow-shift 7 \
    -W $W -H $H --video-frames $FR \
    --diffusion-fa --clip-on-cpu --vae-tiling --vae-relative-tile-size 0.25x0.25 --temporal-tiling \
    --init-img $DR/char.png -o /src/perf_out/${LABEL}.frames/f%03d.png -v \
  > "$OUT/${LABEL}.run.log" 2>&1
echo "rc=$? log=$OUT/${LABEL}.log"
grep -c "Section:" "$OUT/${LABEL}.log" 2>/dev/null
