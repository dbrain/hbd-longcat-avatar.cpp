#!/usr/bin/env bash
# ShotStream (KlingTeam, Wan2.1-T2V-1.3B distilled) SMOKE TEST on the RTX 3060 (device 0).
#
# IMPORTANT: this runs the ShotStream weights through sd-cli's STANDARD Wan T2V sampler
# (bidirectional denoise), NOT ShotStream's causal next-shot + dual-cache streaming loop
# (which isn't ported yet). So this validates the chain end-to-end — convert -> load dense
# Wan2.1-1.3B -> umT5 -> Wan-VAE -> frames on the 3060 — but the OUTPUT is not true
# multi-shot ShotStream behavior. Plumbing check only.
#
# Reuses the wan22 worktree's sd-cli binary (byte-identical: same commit + same ggml pin)
# to skip a redundant ~15min CUDA build. Models come from longcat-avatar.cpp/models.
set -uo pipefail
BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}"   # -ff = ffmpeg present (for the mp4)
BIN_SRC=/home/dbrain/dev/longcat-avatar-wan22                # has /build/bin/sd-cli (same code)
MODELS=/home/dbrain/dev/longcat-avatar.cpp/models            # shotstream DiT + umt5 + wan-vae
OUTREPO=/home/dbrain/dev/longcat-avatar-shotstream
TAG="${TAG:-smoke}"; OUT="$OUTREPO/shotstream_out/$TAG"; mkdir -p "$OUT/frames"

DIT=/models/shotstream-1.3b-dit-f16.gguf
VAE=/models/longcat-wan-vae-f16.gguf
UMT5=/models/longcat-umt5-xxl-q8_0.gguf

W=${W:-832}; H=${H:-480}; FR=${FR:-33}; STEPS=${STEPS:-4}
CFG=${CFG:-1}; SHIFT=${SHIFT:-5}; MAXV=${MAXV:-10}; SEED=${SEED:-42}
PROMPT="${PROMPT:-A cinematic shot of a red fox trotting through a snowy forest at golden hour, soft light, photorealistic.}"

# Ampere-safe env: cuDNN attention + the generic ggml fusions + FFN tiling + distilled DMD
# sigma grid. Deliberately NOT setting WAN_DIT_F16 / CUDNN_F16 (those are Blackwell sm120 levers).
declare -a ENVV=( LONGCAT_FFN_TILE_TOKENS=4096 GGML_CUDNN_ATTN=1
  GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1 GGML_CUDA_RMS_MOD_FUSE=1 )
[ "${DISTILL_SIGMAS_OFF:-0}" = "1" ] || ENVV+=( WAN_DISTILL_SIGMAS=1 "WAN_DISTILL_SHIFT=${SHIFT}" )
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done

BASE=(docker run --rm --gpus '"device=0"' "${ENV_FLAGS[@]}"
  -v "$BIN_SRC:/src" -v "$MODELS:/models" -v "$OUTREPO:/out" -w /src)

echo ">> ShotStream smoke: ${W}x${H} ${FR}f ${STEPS}step cfg${CFG} shift${SHIFT} maxv${MAXV} on device=0 (3060)"
t0=$(date +%s.%N)
"${BASE[@]}" "$BUILDER" /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model "$DIT" --vae "$VAE" --t5xxl "$UMT5" \
  --cfg-scale "$CFG" --sampling-method euler --eta 0 \
  --steps "$STEPS" --flow-shift "$SHIFT" \
  -W "$W" -H "$H" --video-frames "$FR" --diffusion-fa --offload-to-cpu --mmap --max-vram "$MAXV" \
  --vae-tiling --temporal-tiling --vae-relative-tile-size 0.5x0.5 \
  -p "$PROMPT" -s "$SEED" \
  -o "/out/shotstream_out/$TAG/frames/f%03d.png" -v 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}; t1=$(date +%s.%N)
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
echo ">> [$TAG] rc=$rc wall=${wall}s"
if ls "$OUT/frames"/f*.png >/dev/null 2>&1; then
  ffmpeg -y -framerate 16 -pattern_type glob -i "$OUT/frames/*.png" \
    -c:v libx264 -pix_fmt yuv420p "$OUT/shotstream_${TAG}.mp4" -loglevel error 2>/dev/null \
    && echo ">> wrote $OUT/shotstream_${TAG}.mp4 ($(ls "$OUT/frames"/f*.png | wc -l) frames)"
else
  echo ">> NO FRAMES produced — see $OUT/run.log"
fi
