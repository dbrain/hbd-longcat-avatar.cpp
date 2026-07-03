#!/usr/bin/env bash
# nsys timeline of the DiT step: kernel sequence, gaps (launch-overhead / serialization),
# GPU-busy vs wall. Single low expert, FR=13, 1 step, resident, clip-on-cpu (minimal GPU).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out"; mkdir -p "$OUT"
NSYS=/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys
M=/models; DR=/models/_drive; W=480; H=832; FR="${FR:-13}"; LABEL="${LABEL:-dit_nsys}"
PA="A young man sings into the camera. Static medium shot. Warm neon, cinematic."
rm -f "$OUT/${LABEL}.nsys-rep" "$OUT/${LABEL}.sqlite"
docker run --rm --gpus all --cap-add SYS_ADMIN -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
  "$BUILDER" \
  "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite true \
      -o "/src/perf_out/${LABEL}" \
  /src/build/bin/sd-cli -M vid_gen \
    --diffusion-model $M/wan22-i2v-a14b-low-q4_k.gguf \
    --vae $M/longcat-wan-vae-f16.gguf --t5xxl $M/longcat-umt5-xxl-q8_0.gguf \
    -p "$PA" --cfg-scale 1 --sampling-method euler --steps 1 --flow-shift 7 \
    -W $W -H $H --video-frames $FR \
    --diffusion-fa --clip-on-cpu --vae-tiling --vae-relative-tile-size 0.25x0.25 --temporal-tiling \
    --init-img $DR/char.png -o /src/perf_out/${LABEL}.frames/f%03d.png -v \
  > "$OUT/${LABEL}.run.log" 2>&1
echo "rc=$? rep=$OUT/${LABEL}.nsys-rep"
ls -la "$OUT/${LABEL}.nsys-rep" 2>/dev/null
