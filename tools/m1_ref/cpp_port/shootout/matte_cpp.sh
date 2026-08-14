#!/usr/bin/env bash
# Torch-free neural matte lane: the C++/GGML replacement for shootout/birefnet_matte.py.
#
# SUPERSEDED for RMBG-2.0 by ../make_matte_native, which runs the same graph IN-PROCESS (vendored
# vision.cpp at thirdparty/visioncpp, this repo's ggml) instead of launching a container per image:
#     ./make_matte_native in.png out_rgba.png            # identical output contract
# Byte-identical prescale and compose; the mask differs only by the ggml version the container
# pinned (alpha MAE 0.07/255, and the alpha>0.8 crop box is unchanged on soldier/toy1/toy2).
# This script stays as the A/B harness — it is the only lane that can still run BiRefNet's
# ZhengPeng7 weights next to RMBG-2.0 through the shipped vision-server image.
#
# Same output contract as birefnet_matte.py — an RGBA whose RGB is already composited over black
# (rgb*alpha) and whose alpha is the model's soft mask, at the 1024-long-edge scale — so native's
# has_alpha crop path (image_io.hpp pixal_preprocess_black_matte) consumes it unchanged.
#
# Three steps, no Python:
#   1. birefnet_prep prescale   PIL-exact LANCZOS down to a 1024 long edge  (verified byte-identical
#                               to PIL on toy1: maxabs 0 over 1024x1024x3)
#   2. vision-cli birefnet      vision.cpp's GGML BiRefNet graph on the 3060
#   3. birefnet_prep compose    rgb*alpha over black + soft alpha -> RGBA
#
# WHICH WEIGHTS: both candidates are the SAME architecture (585 tensors, identical names and
# shapes; every tensor differs numerically — RMBG-2.0 is BiRefNet retrained/remapped).
#   MATTE_MODEL=birefnet  ZhengPeng7/BiRefNet, the weights Pixal3D's Python preprocess_image uses
#   MATTE_MODEL=rmbg      briaai/RMBG-2.0, what the kobbler vision service serves
# Neither is wired into final_e2e_dc_rig.sh yet — this is the A/B harness.
#
# usage: matte_cpp.sh <in.(png|jpg)> <out_rgba.png> [workdir]
#   env: MATTE_MODEL=birefnet|rmbg     (default birefnet)
#        MATTE_GGUF_DIR=<dir>          (default /mnt/hdd/3d/avatar-shootout/_weights/gguf)
#        MATTE_IMAGE=<docker image>    (default kobbler-vision-server:latest)
#        MATTE_NO_LOCK=1               (caller already holds the 3060 flock)
set -euo pipefail

IN="${1:?input image}"
OUT="${2:?output rgba png}"
WORK="${3:-$(dirname "$OUT")/.matte_cpp_$(basename "${OUT%.png}")}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF_DIR="${MATTE_GGUF_DIR:-/mnt/hdd/3d/avatar-shootout/_weights/gguf}"
IMAGE="${MATTE_IMAGE:-kobbler-vision-server:latest}"
LOCK="/mnt/hdd/3d/avatar-shootout/_shootout_out/.3060-image-to-rig.lock"
GPU3060="${GPU_3060_UUID:-GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6}"

case "${MATTE_MODEL:-birefnet}" in
  birefnet) GGUF=BiRefNet-ZhengPeng7-F16.gguf ;;
  rmbg)     GGUF=RMBG-2.0-F16.gguf ;;
  *) echo "matte_cpp: unknown MATTE_MODEL=${MATTE_MODEL}" >&2; exit 1 ;;
esac
test -f "$GGUF_DIR/$GGUF" || { echo "matte_cpp: missing $GGUF_DIR/$GGUF" >&2; exit 1; }

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"

# Step 1 also reports has_alpha: a source that already carries a real cutout is Python's has_alpha
# branch and needs no model at all.
PRE="$("$HERE/birefnet_prep" prescale --emit-rgba "$IN" "$WORK/prescaled.png")"
echo "$PRE"
if echo "$PRE" | grep -q '^has_alpha=1$'; then
  cp "$WORK/prescaled.png" "$OUT"
  echo "[matte_cpp] source already carries alpha; no matting model run"
  exit 0
fi

# Step 2. VISP_FLASH_ATTENTION=0 is REQUIRED: Swin's head_dim=32 has no CUDA flash-attn instance
# and ggml aborts in fattn.cu ("fatal error", BEST_FATTN_KERNEL_NONE). vision-server sets this
# itself (worker.cpp); vision-cli does not, so the CLI lane must.
CLI=(docker run --rm --gpus "\"device=$GPU3060\"" -e VISP_FLASH_ATTENTION=0
     -v "$GGUF_DIR:/gguf:ro" -v "$WORK:/work"
     --entrypoint /usr/local/bin/vision-cli "$IMAGE"
     birefnet -m "/gguf/$GGUF" -i /work/prescaled.png -o /work/mask.png -b gpu)
if [ -n "${MATTE_NO_LOCK:-}" ]; then
  "${CLI[@]}"
else
  flock -w 3600 "$LOCK" "${CLI[@]}"
fi

# Step 3.
"$HERE/birefnet_prep" compose "$WORK/prescaled.png" "$WORK/mask.png" "$OUT"
echo "[matte_cpp] model=${MATTE_MODEL:-birefnet} gguf=$GGUF -> $OUT"
