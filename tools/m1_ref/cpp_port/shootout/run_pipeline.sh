#!/usr/bin/env bash
# ONE reproducible driver: image.png -> textured, riggable low-poly (fingers intact) GLB, fully automatic.
#   image ─► [matte] ─► pixal3d geom ─► pixal3d --tex (PBR dump) ─► UltraShape (clean/watertight)
#         ─► P3-SAM (segment, hands isolate) ─► per-part region-adaptive QEM (69k, fingers)
#         ─► canonical tex_reproject bake (auto single-frame) ─► <name>_lowpoly_tex.glb
# Every stage is a real call (no manual steps, no hand alignment). Python only wraps the unported models
# (UltraShape, P3-SAM) + mesh-file I/O; all geometry/decimate/normalize/bake compute is C++ (pixal3d port).
# GPU-heavy stages (pixal3d ×2, UltraShape, P3-SAM); coordinate with the owner's bench before launching.
#   ./shootout/run_pipeline.sh <image.png> [name]
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # tools/m1_ref/cpp_port
ROOT=/mnt/hdd/3d/avatar-shootout
OUT="$ROOT/_shootout_out"; mkdir -p "$OUT"
PY="$ROOT/Pixal3D/.venv/bin/python"                       # trimesh/numpy/PIL (mesh + image I/O only)
TEXSIZE="${TEXSIZE:-2048}"

IMG="${1:?usage: run_pipeline.sh <image.png> [name]}"
NAME="${2:-$(basename "$IMG" | sed 's/\.[^.]*$//')}"
MATTE="$OUT/${NAME}_matte.png"; MATTE_BASE="$(basename "$MATTE" | sed 's/\.[^.]*$//')"
GEOM="$OUT/${NAME}.glb"
TEXGLB="$OUT/${NAME}_tex.glb"
REFINED="$OUT/ultrashape_${NAME}/${MATTE_BASE}_refined.glb"
P3DIR="$OUT/p3sam_${NAME}_refined"
FINAL="$P3DIR/auto_mask_mesh_final.glb"
FIDS="$P3DIR/auto_mask_mesh_final_face_ids.npy"
LOWPOLY="$OUT/${NAME}_lowpoly.glb"
LOWPOLY_TEX="$OUT/${NAME}_lowpoly_tex.glb"

step(){ echo; echo "######## $* ########"; }

step "0/6 matte  ($IMG -> $MATTE)"
( cd "$CP" && ./make_matte "$IMG" "$MATTE" )

step "1/6 pixal3d geometry  (-> $GEOM)"
( cd "$CP" && ./pixal3d --model weights_gguf --image "$MATTE" --out "$GEOM" --ply )

step "1b/6 pixal3d --tex  (PBR + dump_*.bin into $CP)  (-> $TEXGLB)"
( cd "$CP" && PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 ./pixal3d --model weights_gguf \
    --image "$MATTE" --out "$TEXGLB" --tex --texsize "$TEXSIZE" )
# Snapshot the fresh DENSE mesh (the --tex run dumps it as dump_mesh_*; dump_dense is gated on the
# remesh path) -> dump_dense_*, the bake's colour source. Must happen BEFORE step 4 overwrites dump_mesh
# with the 69k. Keeps colour-source + PBR + target all from THIS run (the stale-dump bug the old probe hit).
( cd "$CP" && cp -f dump_mesh_v.bin dump_dense_v.bin && cp -f dump_mesh_f.bin dump_dense_f.bin \
    && awk '{print $1, $2}' dump_bake.txt > dump_dense.txt )

step "2/6 UltraShape refine  (-> $REFINED)"
( cd "$CP" && ./shootout/ultrashape_run.sh "$GEOM" "$MATTE" "$NAME" )
[ -f "$REFINED" ] || { echo "ERR: UltraShape output not found: $REFINED"; exit 1; }

step "3/6 P3-SAM segment  (-> $FINAL)"
( cd "$CP" && POINT_NUM="${POINT_NUM:-100000}" PROMPT_BS="${PROMPT_BS:-2}" \
    ./shootout/p3sam_run.sh "$REFINED" "${NAME}_refined" )
[ -f "$FINAL" ] && [ -f "$FIDS" ] || { echo "ERR: P3-SAM output not found ($FINAL / $FIDS)"; exit 1; }

step "4/6 per-part region-adaptive decimate  (-> $LOWPOLY  + dump_mesh bins)"
# dump_dir=$CP: the final 69k is also written as dump_mesh_{v,f}.bin where tex_reproject reads it.
"$PY" "$CP/shootout/per_part_decimate.py" "$FINAL" "$FIDS" "$LOWPOLY" "$CP"

step "5/6 canonical texture bake  (auto single-frame; -> $LOWPOLY_TEX)"
# RP_CANON_TO_DENSE=1: auto center+uniform-scale the 69k (UltraShape frame) onto dump_dense (dump frame).
# CUMESH_TARGET=0: bake onto the PROVIDED dump_mesh (the 69k), don't rebuild a target from the dense.
( cd "$CP" && RP_CANON_TO_DENSE=1 CUMESH_TARGET=0 TEX_FINAL_SIZE="$TEXSIZE" \
    ./tex_reproject "$TEXSIZE" "$LOWPOLY_TEX" )

echo
echo "######## DONE ########"
echo "  textured riggable low-poly: $LOWPOLY_TEX"
echo "  (view: copy to \$CP and add/refresh a compare.html entry; hard-reload Ctrl-Shift-R)"
