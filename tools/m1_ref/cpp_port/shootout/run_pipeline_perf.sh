#!/usr/bin/env bash
# INSTRUMENTED production-param run of the proven pipeline (run_pipeline.sh) with per-stage
# wall-time + peak-VRAM capture, for the 2026-06-20 param-audit + perf run.
#   - pixal3d runs at PRODUCTION --resolution 1536 (upstream standard mode; the headline param gap).
#   - UltraShape / P3-SAM already default to production params in their sub-scripts.
#   - Every stage is wrapped by timed_stage: starts a background nvidia-smi poller, runs the stage,
#     records wall-time + peak GPU memory into $PERFDIR/stages.tsv (stage<TAB>params<TAB>wall_s<TAB>peak_vram_mib).
#   - Full per-stage stdout/stderr captured to $PERFDIR/<stage>.log.
# Same stage commands as run_pipeline.sh — DO NOT diverge the logic; only instrumentation + 1536 added.
#   ./shootout/run_pipeline_perf.sh <image.png> [name]
# Env: RES (pixal3d resolution, default 1536), TEXSIZE (default 2048), PROMPT_BS (default 8),
#      POINT_NUM (default 100000), PERFDIR (default $OUT/_perf_<name>_<RES>).
set -uo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # tools/m1_ref/cpp_port
ROOT=/mnt/hdd/3d/avatar-shootout
OUT="$ROOT/_shootout_out"; mkdir -p "$OUT"
PY="$ROOT/Pixal3D/.venv/bin/python"
TEXSIZE="${TEXSIZE:-2048}"
RES="${RES:-1536}"
PROMPT_BS="${PROMPT_BS:-8}"
POINT_NUM="${POINT_NUM:-100000}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0

IMG="${1:?usage: run_pipeline_perf.sh <image.png> [name]}"
NAME="${2:-$(basename "$IMG" | sed 's/\.[^.]*$//')}"
PERFDIR="${PERFDIR:-$OUT/_perf_${NAME}_${RES}}"; mkdir -p "$PERFDIR"
TSV="$PERFDIR/stages.tsv"
# START = first stage index to run (resume support): 0 matte,1 geom,2 tex,3 ultrashape,4 p3sam,5 decimate,6 bake
START="${START:-0}"
sr() { [ "$START" -le "$1" ]; }   # should-run stage $1
if [ "$START" = "0" ]; then : > "$TSV"; printf 'stage\tparams\twall_s\tpeak_vram_mib\n' >> "$TSV"; fi

MATTE="$OUT/${NAME}_matte.png"; MATTE_BASE="$(basename "$MATTE" | sed 's/\.[^.]*$//')"
GEOM="$OUT/${NAME}.glb"
TEXGLB="$OUT/${NAME}_tex.glb"
REFINED="$OUT/ultrashape_${NAME}/${MATTE_BASE}_refined.glb"
P3DIR="$OUT/p3sam_${NAME}_refined"
FINAL="$P3DIR/auto_mask_mesh_final.glb"
FIDS="$P3DIR/auto_mask_mesh_final_face_ids.npy"
LOWPOLY="$OUT/${NAME}_lowpoly.glb"
LOWPOLY_TEX="$OUT/${NAME}_lowpoly_tex.glb"

# timed_stage <stage-id> <params-desc> -- <command...>
timed_stage() {
  local id="$1"; local params="$2"; shift 2
  [ "$1" = "--" ] && shift
  local vlog="$PERFDIR/vram_${id}.csv"; local slog="$PERFDIR/${id}.log"
  echo; echo "######## [$id] $params ########"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; sleep 0.2; done ) > "$vlog" 2>/dev/null &
  local SMP=$! S=$(date +%s)
  ( "$@" ) 2>&1 | tee "$slog"
  local rc=${PIPESTATUS[0]}
  kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
  local W=$(( $(date +%s) - S ))
  local PK=$(sort -n "$vlog" 2>/dev/null | tail -1)
  printf '%s\t%s\t%s\t%s\n' "$id" "$params" "$W" "${PK:-NA}" >> "$TSV"
  echo "[$id] rc=$rc wall=${W}s peakVRAM=${PK:-NA}MiB"
  return $rc
}

echo "=== INSTRUMENTED PRODUCTION RUN  image=$IMG name=$NAME RES=$RES TEXSIZE=$TEXSIZE ==="
echo "=== perfdir=$PERFDIR ==="

sr 0 && timed_stage "0_matte" "birefnet matte" -- \
  bash -c "cd '$CP' && ./make_matte '$IMG' '$MATTE'"

# MERGED geom+tex (Tier-0.1 dedup): ONE pixal3d run emits the textured GLB + PBR dump AND, via
# --geom-out, the decimated geometry GLB that UltraShape consumes — from a SINGLE geometry diffusion.
# Replaces the old separate `--ply` geom pass (stage 1) + `--tex` pass (stage 2), which re-ran the
# entire ~427s geom diffusion twice. The --geom-out GLB is bit-identical to the old --ply output
# (same `mesh`, same decimate budget). Stage index 2 retired; resume indices 3-6 unchanged.
if sr 1; then
timed_stage "1_pixal3d_geomtex" "res=$RES --tex --geom-out texsize=$TEXSIZE" -- \
  bash -c "cd '$CP' && PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 ./pixal3d --model weights_gguf \
    --image '$MATTE' --out '$TEXGLB' --geom-out '$GEOM' --tex --resolution $RES --texsize $TEXSIZE"
# snapshot dense colour source (same as run_pipeline.sh)
( cd "$CP" && cp -f dump_mesh_v.bin dump_dense_v.bin && cp -f dump_mesh_f.bin dump_dense_f.bin \
    && awk '{print $1, $2}' dump_bake.txt > dump_dense.txt ) || echo "WARN: dense-dump snapshot failed"
fi

sr 3 && { timed_stage "3_ultrashape" "octree=512 latents=8192 chunk=2048 steps=50 low_vram" -- \
  bash -c "cd '$CP' && ./shootout/ultrashape_run.sh '$GEOM' '$MATTE' '$NAME'"
[ -f "$REFINED" ] || echo "ERR: UltraShape output not found: $REFINED"; }

sr 4 && { timed_stage "4_p3sam" "point_num=$POINT_NUM prompt_bs=$PROMPT_BS" -- \
  bash -c "cd '$CP' && POINT_NUM='$POINT_NUM' PROMPT_BS='$PROMPT_BS' ./shootout/p3sam_run.sh '$REFINED' '${NAME}_refined'"
[ -f "$FINAL" ] && [ -f "$FIDS" ] || echo "ERR: P3-SAM output not found ($FINAL / $FIDS)"; }

sr 5 && timed_stage "5_decimate" "per-part region-adaptive QEM" -- \
  bash -c "'$PY' '$CP/shootout/per_part_decimate.py' '$FINAL' '$FIDS' '$LOWPOLY' '$CP'"

sr 6 && timed_stage "6_tex_bake" "TEX_FINAL_SIZE=$TEXSIZE canon_to_dense" -- \
  bash -c "cd '$CP' && RP_CANON_TO_DENSE=1 CUMESH_TARGET=0 TEX_FINAL_SIZE='$TEXSIZE' ./tex_reproject '$TEXSIZE' '$LOWPOLY_TEX'"

echo
echo "######## DONE ########"
echo "  textured riggable low-poly: $LOWPOLY_TEX"
echo "=== PERF TABLE ($TSV) ==="
column -t -s $'\t' "$TSV"
