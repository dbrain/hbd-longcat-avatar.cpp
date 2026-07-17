#!/usr/bin/env bash
# rig_texture_chain.sh — ONE reproducible chain: textured mesh + sampled-point rig inputs
#   -> skintokens_e2e (rig: R1->R3->R5->R4)  -> combine_rig_tex_main (kNN skin-transfer + merge)
#   -> a single TEXTURED + SKINNED .glb (the avatar deliverable). No manual stitching.
#
# RIG = DETERMINISTIC beam (do_sample=False), beams=20. This is the PRINCIPLED decode, not a selection
# hack: sampling (do_sample=True, temp=1.5) is what produces the occasional head-fan/runaway — it's chaos
# amplifying the (unavoidable) ggml-vs-PyTorch float drift in the forward. Turning sampling OFF removes the
# chaos entirely; the deterministic top-beam decode is structurally clean every time. beams=20 (not 10)
# because the forward drift pushes Python's winning hypothesis just outside the top-10 — at top-20 the C++
# recovers it: on gilly the parent tree is BIT-IDENTICAL to Python's deterministic J56 rig (maxfan 5),
# 6.6 GB VRAM, ~17 s, fully reproducible (no seed lottery, no fan-rejection). See FINDINGS-rig-forward-parity.md.
#
# Usage:
#   rig_texture_chain.sh <rig_input_dir> <textured_mesh.glb> <qwen3_w_dir> <out.glb> [beams=20]
#
#   <rig_input_dir>     dir with vertices.npy [Ns,3] + normals.npy [Ns,3] (FPS-sampled cond points; banked).
#   <textured_mesh.glb> full-res TEXTURED mesh to skin (POSITION/NORMAL/TEXCOORD_0 + baseColor).
#   <qwen3_w_dir>       rigger qwen3 weight dir.
#   <out.glb>           output textured+skinned GLB.
#   [beams]             beam width (default 20; wider recovers the richer Python-matching skeleton).
#
# Env: PIXAL3D_GGUF_DIR must point at the skin-VAE GGUFs (skin_vae_gguf).
set -euo pipefail
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIG_IN="${1:?rig_input_dir}"; TEX_GLB="${2:?textured_mesh.glb}"; QW="${3:?qwen3_w_dir}"; OUT="${4:?out.glb}"
BEAMS="${5:-20}"
: "${PIXAL3D_GGUF_DIR:?set PIXAL3D_GGUF_DIR to the skin_vae_gguf dir}"
RIG_GLB="$(dirname "$OUT")/_rig_only.glb"
DUMP=/tmp/skintokens_e2e

echo "== [1/2] DETERMINISTIC auto-rig (beam nosample, beams=$BEAMS, fp32) =="
"$CP/skintokens_e2e" "$RIG_IN" "$QW" "$RIG_GLB" cuda beam nosample prec=fp32 \
    r1=banked cond=banked beams="$BEAMS" maxnew=2048 \
  | grep -iE 'DONE|J=|STAGE R4|STAGE R5|WARN' || true
"$CP/rig_score" "$RIG_GLB" 55 2>/dev/null | grep -iE 'maxfan|TOTAL' | head -1 || true

echo "== [2/2] combine textured mesh + rig -> textured+skinned GLB =="
"$CP/combine_rig_tex_main" "$TEX_GLB" "$DUMP" "$OUT" \
    --sampled "$RIG_IN/vertices.npy" \
    --skin    "$DUMP/gen_skin_pred.npy" \
    --joints  "$DUMP/gen_joints.npy" \
    --parents "$DUMP/gen_parents.npy"

echo "== DONE -> $OUT =="
