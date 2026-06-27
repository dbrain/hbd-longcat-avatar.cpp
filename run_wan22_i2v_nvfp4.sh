#!/usr/bin/env bash
# Wan2.2-14B i2v "dancing" on NVFP4 (Blackwell FP4/FP8 tensor cores) — fast path.
# A/B target = the q4_K clip from run_wan22_i2v_dance.sh. Samples peak VRAM + wall.
#
#   TAG=nvfp4_base ./run_wan22_i2v_nvfp4.sh
#   TAG=nvfp4_stoch DOTS_ENV="GGML_NVFP4_ACT_STOCHASTIC=1" ./run_wan22_i2v_nvfp4.sh
#
# Set QK=1 to run the q4_K baseline through the SAME cuDNN binary + env (apples-to-apples).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="${BUILDER:-longcat-avatar-dev:builder-cudnn}"
TAG="${TAG:-nvfp4_base}"
OUT="$REPO/wan_dance_out/$TAG"; mkdir -p "$OUT/frames"
M=/models
if [ "${QK:-0}" = "1" ]; then
  VL=$M/wan22-i2v-a14b-low-q4_k.gguf;    VH=$M/wan22-i2v-a14b-high-q4_k.gguf
else
  VL=$M/wan22-i2v-a14b-low-nvfp4.gguf;   VH=$M/wan22-i2v-a14b-high-nvfp4.gguf
fi
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf; IMG=$M/flux_neon_seed7.png

W=${W:-1280}; H=${H:-704}; FR=${FR:-81}; SHIFT=${SHIFT:-5}
HSTEPS=${HSTEPS:-2}; LSTEPS=${LSTEPS:-2}; CFG=${CFG:-1}; MAXV=${MAXV:-14}; SEED=${SEED:-42}
PROMPT="${PROMPT:-The person dances energetically with big, expressive head and upper-body movements, hair and clothing swaying with the motion, dynamic and lively, neon-lit, photorealistic, cinematic.}"

# Fast GEMM + cuDNN-attn + fusion env stack (generic at ggml mul_mat; fires on NVFP4 weights).
# DOTS_ENV adds/overrides act-quant knobs for the dots experiments (space-separated KEY=VAL).
declare -a ENVV=( "GGML_NVFP4_CUBLASLT=${NVFP4_CUBLASLT:-1}" "GGML_NVFP4_QUANT_TWOLEVEL=${TWOLEVEL:-1}" LONGCAT_FFN_TILE_TOKENS=4096 )
[ "${FP8_OFF:-0}" = "1" ]   || ENVV+=( GGML_FP8_FFN=1 "GGML_FP8_LAYERS=${FP8_LAYERS:-blocks.}" )
[ "${CUDNN_OFF:-0}" = "1" ] || ENVV+=( GGML_CUDNN_ATTN=1 GGML_CUDNN_ATTN_F16_OUT=1 )
[ "${FUSE_OFF:-0}" = "1" ]  || ENVV+=( GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1 GGML_CUDA_RMS_MOD_FUSE=1 )
for kv in ${DOTS_ENV:-}; do ENVV+=("$kv"); done
ENV_FLAGS=(); for e in "${ENVV[@]}"; do ENV_FLAGS+=(-e "$e"); done
# Forward any GGML_*/LONGCAT_*/LTX_* debug toggles set in the parent shell (DBG, TRACE,
# NANCHECK, ACT_* knobs, etc.) into the container without baking them into the stack.
for v in $(compgen -e | grep -E '^(GGML_|LONGCAT_|LTX_)' || true); do
  case " ${ENVV[*]} " in *" $v="*) continue;; esac   # don't double-set stack vars
  ENV_FLAGS+=(-e "$v=${!v}")
done

BASE=(docker run --rm --gpus '"device=1"' "${ENV_FLAGS[@]}" -v "$REPO:/src" -v "$REPO/models:/models" -w /src)

sample_vram() { local f="$1"; echo 0 > "$f"
  while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
    [ -n "$u" ] && { m=$(cat "$f"); [ "$u" -gt "$m" ] && echo "$u" > "$f"; }; sleep 0.3; done; }
vf="$OUT/.vram"; sample_vram "$vf" & sp=$!
t0=$(date +%s.%N)
"${BASE[@]}" "$BUILDER" /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model "$VL" --high-noise-diffusion-model "$VH" \
  --vae "$VAE" --t5xxl "$UMT5" \
  --cfg-scale "$CFG" --high-noise-cfg-scale "$CFG" \
  --sampling-method euler_a --high-noise-sampling-method euler_a --eta 1.0 \
  --steps "$LSTEPS" --high-noise-steps "$HSTEPS" --flow-shift "$SHIFT" \
  -W "$W" -H "$H" --video-frames "$FR" --diffusion-fa --offload-to-cpu --mmap --max-vram "$MAXV" \
  --vae-tiling --temporal-tiling --vae-relative-tile-size "${VAE_TILE:-0.5x0.5}" \
  --init-img "$IMG" -p "$PROMPT" -s "$SEED" \
  -o /src/wan_dance_out/$TAG/frames/f%03d.png -v 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}; t1=$(date +%s.%N)
kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"); peak=$(cat "$vf" 2>/dev/null); rm -f "$vf"
echo ">> [$TAG] wan22 i2v ${W}x${H} ${FR}f $((HSTEPS+LSTEPS))step: ${wall}s  peak ${peak}MiB  rc=$rc"
if [ -n "$(ls "$OUT/frames"/f*.png 2>/dev/null)" ]; then
  ffmpeg -y -framerate 16 -pattern_type glob -i "$OUT/frames/*.png" \
    -c:v libx264 -pix_fmt yuv420p "$OUT/wan22_${TAG}.mp4" -loglevel error
  echo ">> $OUT/wan22_${TAG}.mp4"
fi
