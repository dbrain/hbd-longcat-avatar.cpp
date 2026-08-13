#!/usr/bin/env bash
# image2rig.sh — ONE command: raw image.png -> textured + rigged (skinned) GLB.
# Every stage is a committed C++/ggml binary or a thin host service. No manual stitching, no
# placeholder giraffe goldens, no hand-managed temp dirs.
#
#   image.png
#     -> image_to_geometry.sh --tex   : RMBG matte + MoGe-2 camera + pixal3d -> TEXTURED dense mesh.glb
#     -> mesh_sample_main             : normalize + area-sample 8192 pts + normals + FPS query[512]
#                                       (+ fresh sampled_feats — NO banked giraffe normals)
#     -> skintokens_e2e (beam)        : R1 VecSet(real) -> R3 Qwen3 AR (HF-official beam-sample:
#                                       beams=10, temp=1.0, top_k=5, top_p=0.95, rep=2.0, bf16,
#                                       max_len=2048) -> R5 detok -> R4 skin decode -> rig + dump
#     -> combine_rig_tex_main         : kNN skin-transfer sampled->full textured mesh + merge
#   => textured + skinned GLB (POSITION/NORMAL/TEXCOORD_0/JOINTS_0/WEIGHTS_0 + skeleton + baseColor)
#
# RIG decode = the documented Python-equivalent (skintokens_e2e DEFAULTS: beam-sample do_sample=True,
# beams=10, temp=1.0, top_k=5, top_p=0.95, repetition_penalty=2.0, bf16, max_length=2048). This is the
# model's native regime. It is STOCHASTIC (seed-dependent) — SEED env picks the draw; best-of-N over
# seeds (best_rig.sh, scored by rig_score) is the reliability lever when a draw runs away.
#
# Usage:
#   image2rig.sh <input_image> <out.glb> [-- <extra pixal3d args...>]
#     e.g.  image2rig.sh char.png char_rigged.glb -- --texsize 1024 --decimate 120000
#
# Env (defaults shown):
#   PIXAL3D_GGUF_DIR=/mnt/hdd/pixal3d/weights_gguf_f16        geometry GGUFs
#   SKIN_VAE_GGUF=/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf
#   QWEN3_W=/mnt/hdd/3d/avatar-shootout/rig_audit/inputs/qwen3_w
#   R1W_SRC=/mnt/hdd/3d/avatar-shootout/rig_audit/r1w_real    vecset encoder + output_proj weights
#   BEAMS=20   TEXSIZE=1024   DECIMATE=150000
#   FOV=<deg>  bypass MoGe (controlled input)
#   WORKDIR=<dir>  keep intermediates (default temp dir, cleaned unless WORKDIR_KEEP=1)
set -euo pipefail

echo "image2rig.sh is retired for production: it bypasses the native real-R1/R4 provenance and real-LBS publication gates." >&2
echo "Use shootout/native_image_to_rig.sh, which must finish through rig_texture_chain.sh." >&2
exit 64

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="${1:?input image}"; OUT="${2:?out.glb}"; shift 2 || true
EXTRA=(); if [[ "${1:-}" == "--" ]]; then shift; EXTRA=("$@"); fi

GEO_GGUF="${PIXAL3D_GGUF_DIR:-/home/dbrain/models/3d/geo}"   # M3b+tex Q8_0 (near-lossless, −111s); F16 set = weights_gguf_f16
SKIN_VAE_GGUF="${SKIN_VAE_GGUF:-/home/dbrain/models/3d/rig/skin_vae_gguf}"
QWEN3_W="${QWEN3_W:-/home/dbrain/models/3d/rig/qwen3_w}"
R1W_SRC="${R1W_SRC:-/home/dbrain/models/3d/rig/r1w_real}"
SEED="${SEED:-0}"; TEXSIZE="${TEXSIZE:-1024}"; DECIMATE="${DECIMATE:-150000}"

WORKDIR="${WORKDIR:-$(mktemp -d /tmp/image2rig.XXXXXX)}"; mkdir -p "$WORKDIR"
cleanup(){ [[ -z "${WORKDIR_KEEP:-}" && "$WORKDIR" == /tmp/image2rig.* ]] && rm -rf "$WORKDIR" || true; }
trap cleanup EXIT
MESH="$WORKDIR/mesh.glb"; SAMP="$WORKDIR/samp"; R1W="$WORKDIR/r1w"; RIG="$WORKDIR/rig.glb"
DUMP=/tmp/skintokens_e2e   # skintokens_e2e hardcodes its dump dir here (gen_joints/parents/skin)
mkdir -p "$SAMP" "$R1W" "$DUMP" "$(dirname "$OUT")"

echo "############ [1/4] image -> TEXTURED geometry ############"
PIXAL3D_GGUF_DIR="$GEO_GGUF" "$CP/image_to_geometry.sh" "$IN" "$MESH" -- \
    --tex --texsize "$TEXSIZE" --decimate "$DECIMATE" "${EXTRA[@]}"

echo "############ [2/4] mesh -> rig cond points (native) ############"
"$CP/mesh_sample_main" "$MESH" "$SAMP"

echo "############ [3/4] auto-rig (R1 real, HF-official beam-sample, seed=$SEED) ############"
# per-run R1 weight dir: shared vecset/output_proj weights + THIS mesh's fresh query points/feats.
for f in "$R1W_SRC"/*.npy; do ln -s "$f" "$R1W/" 2>/dev/null || true; done
cp --remove-destination "$SAMP/sampled_pc.npy" "$SAMP/sampled_feats.npy" "$R1W/"   # override giraffe
# All sampler knobs left at the binary's DEFAULTS (= HF official: beams=10 temp=1.0 top_k=5 top_p=0.95
# rep=2.0 bf16 max_length=2048). Only seed + the real-mesh cond source are set here.
PIXAL3D_GGUF_DIR="$SKIN_VAE_GGUF" "$CP/skintokens_e2e" "$SAMP" "$QWEN3_W" "$RIG" \
    cuda beam prec=bf16 r1=real cond=real r1w="$R1W" seed="$SEED" \
  | grep -iE 'STAGE|J=|joints|DONE|FAIL|WARN|PERF|cond_latents' || true
"$CP/rig_score" "$RIG" 55 2>/dev/null | grep -iE 'maxfan|TOTAL' | head -1 || true

echo "############ [4/4] combine textured mesh + rig -> textured+skinned GLB ############"
"$CP/combine_rig_tex_main" "$MESH" "$DUMP" "$OUT" \
    --sampled "$SAMP/vertices.npy" \
    --skin    "$DUMP/gen_skin_pred.npy" \
    --joints  "$DUMP/gen_joints.npy" \
    --parents "$DUMP/gen_parents.npy"

echo "############ DONE -> $OUT ############"
ls -la "$OUT"
