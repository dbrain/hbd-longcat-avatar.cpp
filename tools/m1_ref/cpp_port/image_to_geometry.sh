#!/usr/bin/env bash
# image_to_geometry.sh — ONE command: raw image.png -> geometry .glb. No manual matte/camera.
#
#   image.png
#     -> (1) bg-removal / matte   : RMBG-2.0 vision.cpp service (MATTING_URL) -> RGBA subject cutout
#     -> (2) make_matte (C++)      : crop to subject + square + premultiply on black -> pixal3d matte
#     -> (3) MoGe-2 camera (host)  : estimate_camera.py FOV+distance (Ruicheng/moge-2-vitl, MIT)
#     -> (4) pixal3d (C++/ggml)    : geometry generation with the estimated --cam
#   => geometry .glb
#
# Every stage is a committed, reproducible tool (host service or native binary) — no hand stitching.
# The matte service + MoGe both idle-unload; estimate_camera execs pixal3d (torch frees before ggml
# starts) so torch and ggml never share the GPU.
#
# Usage:
#   image_to_geometry.sh <input_image> <out.glb> [-- <extra pixal3d args...>]
#     e.g.  image_to_geometry.sh photo.png model.glb -- --tex --fast --decimate 40000
#
# Env (defaults shown):
#   MATTING_URL=http://localhost:18898         RMBG-2.0 matting-server base
#   PIXAL3D_GGUF_DIR=/mnt/hdd/pixal3d/weights_gguf_f16   geometry GGUFs (--model arg)
#   PIXAL3D_MODEL=weights_gguf_f16             model dir name under PIXAL3D_GGUF_DIR's parent
#   MOGE_PY=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python   host venv (PIL+torch+MoGe)
#   MATTE_MARGIN=0.05                          subject border each side (controls MoGe framing/FOV)
#   FOV=<deg>                                  bypass MoGe; use a fixed FOV (controlled input)
#   WORKDIR=<dir>                              keep intermediates (default: a temp dir, cleaned)
set -euo pipefail
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IN="${1:?input image}"; OUT="${2:?out.glb}"; shift 2 || true
# everything after a literal `--` is forwarded verbatim to pixal3d
EXTRA=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA=("$@"); fi

MATTING_URL="${MATTING_URL:-http://localhost:18898}"
PIXAL3D_GGUF_DIR="${PIXAL3D_GGUF_DIR:-/mnt/hdd/pixal3d/weights_gguf_f16}"
MODEL_DIR="${PIXAL3D_MODEL:-$PIXAL3D_GGUF_DIR}"
MOGE_PY="${MOGE_PY:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"
MATTE_MARGIN="${MATTE_MARGIN:-0.05}"
export PIXAL3D_GGUF_DIR

WORKDIR="${WORKDIR:-$(mktemp -d /tmp/img2geo.XXXXXX)}"
KEEP_WORK=0; [[ -n "${WORKDIR_KEEP:-}" ]] && KEEP_WORK=1
mkdir -p "$WORKDIR"
RGB="$WORKDIR/rgb.png"; CUT="$WORKDIR/cutout.png"; MATTE="$WORKDIR/matte.png"
cleanup(){ [[ $KEEP_WORK -eq 0 && "$WORKDIR" == /tmp/img2geo.* ]] && rm -rf "$WORKDIR" || true; }
trap cleanup EXIT

echo "== [1/4] bg-removal (RMBG-2.0 @ $MATTING_URL) =="
# Normalise any input (format/alpha) to clean RGB the stb-based service can decode.
"$MOGE_PY" -c "
import sys; from PIL import Image
Image.open('$IN').convert('RGB').save('$RGB')
print('  normalised', '$IN', '->', Image.open('$RGB').size)
"
code=$(curl -s -X POST "$MATTING_URL/remove?bg_mode=alpha" -F "images=@$RGB" -o "$CUT" -w "%{http_code}")
[[ "$code" == "200" ]] || { echo "matting service HTTP $code: $(head -c 200 "$CUT")"; exit 1; }
echo "  cutout -> $CUT"

echo "== [2/4] make_matte (square, black-bg, premultiplied; margin $MATTE_MARGIN) =="
"$CP/make_matte" "$CUT" "$MATTE" "$MATTE_MARGIN" 8

echo "== [3/4] MoGe-2 camera + [4/4] pixal3d geometry =="
mkdir -p "$(dirname "$OUT")"
if [[ -n "${FOV:-}" ]]; then
  echo "  (fixed --fov $FOV, MoGe bypassed)"
  "$CP/pixal3d" --model "$MODEL_DIR" --image "$MATTE" --out "$OUT" --fov "$FOV" "${EXTRA[@]}"
else
  # estimate_camera.py --run estimates the camera then exec's pixal3d with the right --cam in-process
  # (torch frees + empty_cache before the exec, so ggml has the GPU to itself).
  "$MOGE_PY" "$CP/estimate_camera.py" "$MATTE" --run \
      --pixal3d "$CP/pixal3d" --model "$MODEL_DIR" --out "$OUT" "${EXTRA[@]}"
fi

echo "== DONE -> $OUT =="
