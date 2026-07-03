#!/usr/bin/env bash
# render_shot.sh — DIRECTOR UX for Wan2.2-VACE video.
# =====================================================================================
# You give ONE prompt + a target duration (SHOT_FRAMES). The engine renders that as a
# chain of overlapping SHORTER windows (WINDOW frames each) via VACE continuation,
# colour-locks + overlap-trims them, and emits ONE mp4 trimmed to ~SHOT_FRAMES. No manual
# per-segment scripting — the window count N, the roll-forward latent, the pixel tail and
# the stitch are all derived automatically.
#
# This is a self-contained fold-in of THREE existing scripts (nothing new invented):
#   * bank_seg1.sh          -> window 1 (i2v with --init-img, or t2v), WAN_SAVE_LATENT + tail bank
#   * iter_seg2.sh          -> the single VACE continuation reference (env shape)
#   * chain_long.sh         -> the N-window roll-forward loop + colour-lock/overlap-trim stitch
#   * run_clean_face_t2v.sh -> the clean t2v base window (VACE_STRENGTH=0)
#   * run_skipblock0_test.sh-> the VACE_SKIP_BLOCKS=0 grid/flash fix + 0.5x0.5 VTILE + nvfp4 stack
#
# Baked-in fixes (every render, per the brief):
#   -e VACE_SKIP_BLOCKS=0            (Kijai block-0 flash/grid fix; wan.hpp:1107)
#   -e GGML_CUDNN_CONV3D=1           (VAE head.2 conv3d grid/screen-door fix; wan_vae.hpp:42)
#   -e VACE_STRENGTH_TAIL=<TAIL>     (per-frame striping ramp, default 0.2; wan.hpp:1135) *
#   -e VACE_STRENGTH_ANCHOR_FRAMES=2 (leading full-strength anchor frames; wan.hpp:1140)   *
#   -e LONGCAT_VAE_TEMPORAL_CHUNK=0  (monolithic VAE decode = the start-flash fix, cleaner+faster)
#     * = continuation windows only. On the clean t2v base window VACE_STRENGTH=0 (no control
#         residual) so the ramp is intentionally OMITTED there — enabling a 0.2 tail floor there
#         would re-add gray-control residual to a window that must stay pure base-t2v
#         (wan.hpp:1132 gates the ramp on `c != nullptr`; sd.cpp:4089-90 documents STRENGTH=0).
#   + full nvfp4 prod stack, --max-vram 3, --vae-relative-tile-size 0.5x0.5, euler_a, cfg 1,
#     flow-shift 5, 2+2 steps, distill sigmas/shift.
#
# NO GPU here — DRY_RUN=1 prints the plan + every docker command WITHOUT running. The main
# thread runs it for real on device 1.
# =====================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"; M="$REPO/models"
B="longcat-avatar-dev:builder-cudnn"
VAE="/models/longcat-wan-vae-f16.gguf"; T5="/models/longcat-umt5-xxl-q8_0.gguf"
OUTDIR="$REPO/tools/lightx2v-test/out"; mkdir -p "$OUTDIR"

# ---- Director inputs (flags via env) -------------------------------------------------
PROMPT="${PROMPT:-A person dancing energetically on a neon-lit city street at night, cinematic, photorealistic}"
SHOT_FRAMES="${SHOT_FRAMES:-81}"     # target OUTPUT length (frames). 4n+1 recommended (81=4*20+1).
INIT_IMG="${INIT_IMG:-}"             # set => i2v window-1 (+ VACE identity anchor); unset => t2v.
WINDOW="${WINDOW:-41}"               # internal render window length (4n+1; 41=4*10+1).
K="${K:-5}"                          # pixel overlap frames carried between windows (=> (K-1)/4+1 latent).
SEED="${SEED:-42}"
W="${W:-1280}"; H="${H:-704}"
QUANT="${QUANT:-nvfp4}"              # nvfp4 (prod) or q4_k. Slots into i2v AND vace model names.
TAIL="${VACE_STRENGTH_TAIL:-0.2}"    # striping-ramp tail floor (continuation windows only).
ANCHOR="${VACE_STRENGTH_ANCHOR_FRAMES:-2}"
DISCARD="${DISCARD:-4}"              # degraded terminal PIXEL frames dropped from each continuation.
DROPLAT="${DROPLAT:-1}"             # degraded terminal LATENT frames dropped when rolling forward.
MAXV="${MAXV:-3}"; VTILE="${VTILE:-0.5x0.5}"
CONTRAST="${CONTRAST:-0.85}"         # colour-lock std-match strength.
OUTTAG="${OUTTAG:-shot}"
GPU="${GPU:-1}"                      # device 1 (all sibling scripts target it).
DRY_RUN="${DRY_RUN:-0}"
GPU_FREE_GATE="${GPU_FREE_GATE:-4200}"; RETRIES="${RETRIES:-6}"; ETA="${ETA:-1.0}"

# ---- Models (QUANT slots into both families) -----------------------------------------
IL="/models/wan22-i2v-a14b-low-$QUANT.gguf";  IH="/models/wan22-i2v-a14b-high-$QUANT.gguf"
VL="/models/wan22-vace-fun-a14b-low-distillT2V-$QUANT.gguf"; VH="/models/wan22-vace-fun-a14b-high-distillT2V-$QUANT.gguf"

# ---- nvfp4 PRODUCTION env stack (mirrors run_skipblock0_test.sh + chain_long NVENV, plus the
#      CONV3D grid fix and CHUNK=0 start-flash fix required by the brief) -----------------
NVENV=(
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1
  -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=blocks.
  -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_CONV3D=1
  -e WAN_DIT_F16=1 -e WAN_ROPE_F16=1
  -e LONGCAT_VAE_TEMPORAL_CHUNK=0 -e LONGCAT_FFN_TILE_TOKENS=4096
  -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1
  -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  -e WAN_DISTILL_SIGMAS=1 -e WAN_DISTILL_SHIFT=5
)
GPUS="\"device=$GPU\""              # literal: docker gets  --gpus "device=1"  (as the sibling scripts do)

# Common sd-cli sampler tail shared by every window (euler_a, cfg1, flow-shift5, 2+2 steps).
sampler_args() {
  printf '%s\0' \
    --sampling-method euler_a --high-noise-sampling-method euler_a --eta "$ETA" \
    --steps 2 --high-noise-steps 2 --cfg-scale 1.0 --high-noise-cfg-scale 1.0 --flow-shift 5 -s "$SEED" \
    -W "$W" -H "$H" --video-frames "$WINDOW" --diffusion-fa --offload-to-cpu --mmap --max-vram "$MAXV" \
    --vae-tiling --temporal-tiling --vae-relative-tile-size "$VTILE"
}

# ---- INIT_IMG mode + mount -----------------------------------------------------------
if [ -n "$INIT_IMG" ]; then
  MODE="i2v"
  IIABS="$(readlink -f "$INIT_IMG")"
  [ -f "$IIABS" ] || { echo "ERROR: INIT_IMG not found: $INIT_IMG"; exit 1; }
  IIDIR="$(dirname "$IIABS")"; IIBASE="$(basename "$IIABS")"
  INIT_MNT=(-v "$IIDIR:/initdir:ro"); INIT_ARG="/initdir/$IIBASE"
else
  MODE="t2v"; INIT_MNT=(); INIT_ARG=""
fi

# ---- Window math ---------------------------------------------------------------------
# Each continuation contributes conservatively (WINDOW - K - DISCARD) NEW displayed frames:
#   - K head frames are the overlap with the prior window (colour-lock --skip-head K drops them,
#     mirroring chain_long.sh:84 `--skip-head $K` and the raw-stitch `tail -n +$((K+1))` at :94),
#   - DISCARD tail frames are the degraded terminal frames (chain_long.sh:71 `keep=n-DISCARD`, :93).
# Window 1 (scene start) contributes its full WINDOW frames (colour-lock keeps it verbatim,
# vace_colour_lock.py:64). We size N so the CONSERVATIVE total >= SHOT_FRAMES, then trim the
# real (slightly longer) stitch down to exactly SHOT_FRAMES — over-producing is safe, and the
# trim removes the LAST window's degraded tail.
CONT_NEW=$(( WINDOW - K - DISCARD ))
[ "$CONT_NEW" -gt 0 ] || { echo "ERROR: WINDOW($WINDOW) - K($K) - DISCARD($DISCARD) = $CONT_NEW <= 0; raise WINDOW."; exit 1; }
if [ "$SHOT_FRAMES" -le "$WINDOW" ]; then N=1
else N=$(( 1 + (SHOT_FRAMES - WINDOW + CONT_NEW - 1) / CONT_NEW )); fi
CONS_TOTAL=$(( WINDOW + (N - 1) * CONT_NEW ))          # guaranteed >= SHOT_FRAMES
REAL_TOTAL=$(( WINDOW + (N - 1) * (WINDOW - K) ))      # actual colour-lock output (before trim)

warn41() { local f=$1 n=$2; [ $(( (f - 1) % 4 )) -eq 0 ] || echo "  NOTE: $n=$f is not 4n+1; the engine will align it (LOG_WARN)."; }

echo "======================================================================================"
echo "render_shot: mode=$MODE  prompt=\"${PROMPT:0:60}...\""
echo "  target SHOT_FRAMES=$SHOT_FRAMES  WINDOW=$WINDOW  K=$K  DISCARD=$DISCARD  DROPLAT=$DROPLAT  quant=$QUANT"
echo "  => N=$N windows  (window1=$WINDOW f full; each continuation ~+$CONT_NEW new f)"
echo "     conservative total=$CONS_TOTAL f (>= target); real stitch=$REAL_TOTAL f -> trim to $SHOT_FRAMES"
echo "  res=${W}x${H}  seed=$SEED  max-vram=$MAXV  vtile=$VTILE  tail-ramp=$TAIL@anchor$ANCHOR  device=$GPU"
warn41 "$SHOT_FRAMES" SHOT_FRAMES; warn41 "$WINDOW" WINDOW
echo "======================================================================================"

# ---- workspace -----------------------------------------------------------------------
WORKREL="render_$OUTTAG"; WORK="$REPO/$WORKREL"
if [ "$DRY_RUN" != 1 ]; then rm -rf "$WORK"; mkdir -p "$WORK/bank"; fi

# ---- helpers -------------------------------------------------------------------------
# pretty-print a docker argv array for the dry-run trace (newline before each -e / -v / stdbuf)
pp_cmd() { local a; for a in "$@"; do case "$a" in -e|-v|stdbuf) printf '\n      ';; esac; printf '%q ' "$a"; done; printf '\n'; }

wait_gpu() {
  [ "$DRY_RUN" = 1 ] && return 0
  local w=0 u
  while :; do u=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    [ "${u:-99999}" -lt "$GPU_FREE_GATE" ] && break
    [ $((w % 60)) -eq 0 ] && echo "  ...waiting GPU-$GPU (${u}MiB >= $GPU_FREE_GATE)"; sleep 15; w=$((w + 15)); done
}

# background nvidia-smi peak sampler (reports whole-GPU used MiB incl. other tenants on the shared card)
vram_start() { VPF=$(mktemp); echo 0 >"$VPF"
  ( while :; do u=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null||echo 0)
      m=$(cat "$VPF" 2>/dev/null||echo 0); [ "${u:-0}" -gt "${m:-0}" ] && echo "${u:-0}" >"$VPF"; sleep 1; done ) & VPID=$!; }
vram_stop() { kill "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null; PEAK=$(cat "$VPF" 2>/dev/null||echo 0); rm -f "$VPF"; }

# run (or echo) one window's docker command; retries on OOM; returns via globals FRAMES_OUT
declare -a WALLS=() PEAKS=()
run_window() {  # $1=label  $2=host_out_dir  $3=logfile  ; command already in global CMD[]
  local label="$1" od="$2" log="$3"
  if [ "$DRY_RUN" = 1 ]; then
    echo "--- [$label] docker command (DRY_RUN — not executed) ---"; pp_cmd "${CMD[@]}"; echo
    WALLS+=("dry"); PEAKS+=("dry"); return 0
  fi
  mkdir -p "$od"
  local try n t0 wall
  for ((try=1; try<=RETRIES; try++)); do
    wait_gpu; rm -f "$od"/f*.png
    vram_start; t0=$SECONDS
    "${CMD[@]}" >"$log" 2>&1
    wall=$((SECONDS - t0)); vram_stop
    n=$(ls "$od"/f*.png 2>/dev/null | wc -l)
    if [ "$n" -ge "$WINDOW" ]; then WALLS+=("$wall"); PEAKS+=("$PEAK"); echo "  [$label] OK $n/$WINDOW f  wall=${wall}s  peakVRAM=${PEAK}MiB"; return 0; fi
    if grep -qaiE "out of memory|CUDA error|alloc.*failed" "$log"; then echo "  [$label] try$try OOM ($n/$WINDOW) — retry"; sleep 20; continue; fi
    echo "  [$label] FAILED non-OOM: $n/$WINDOW"; grep -aiE "error|fail|mismatch|not implemented" "$log"|grep -av 0x|tail -6; exit 2
  done
  echo "  [$label] FAILED after $RETRIES retries"; exit 2
}

# =====================================================================================
# WINDOW 1  (fold-in of bank_seg1.sh / run_clean_face_t2v.sh) — auto, no manual step.
# =====================================================================================
mapfile -d '' SAMP < <(sampler_args)
w1od="$WORK/w01"; w1odc="/src/$WORKREL/w01"; w1log="$w1od/log"
W1_LAT="/src/$WORKREL/bank/w01.bin"                                   # WAN_SAVE_LATENT target (== seg1.bin role)
if [ "$MODE" = i2v ]; then
  # bank_seg1.sh logic: i2v models + --init-img, bank the diffusion latent via WAN_SAVE_LATENT.
  CMD=(docker run --rm --gpus "$GPUS"
    -e WAN_SAVE_LATENT="$W1_LAT" -e VACE_SKIP_BLOCKS=0 "${NVENV[@]}"
    -v "$REPO:/src" -v "$M:/models" "${INIT_MNT[@]}" -w /src "$B"
    stdbuf -oL -eL /src/build/bin/sd-cli -M vid_gen
    --diffusion-model "$IL" --high-noise-diffusion-model "$IH" --vae "$VAE" --t5xxl "$T5"
    --init-img "$INIT_ARG" -p "$PROMPT" "${SAMP[@]}" -o "$w1odc/f%03d.png" -v)
else
  # run_clean_face_t2v.sh logic: VACE models + VACE_STRENGTH=0 (clean base t2v, no control/init).
  # NOTE: VACE_STRENGTH_TAIL/ANCHOR are deliberately NOT set here (see header) so the base stays pure.
  CMD=(docker run --rm --gpus "$GPUS"
    -e WAN_SAVE_LATENT="$W1_LAT" -e VACE_STRENGTH=0 -e VACE_SKIP_BLOCKS=0 "${NVENV[@]}"
    -v "$REPO:/src" -v "$M:/models" -w /src "$B"
    stdbuf -oL -eL /src/build/bin/sd-cli -M vid_gen
    --diffusion-model "$VL" --high-noise-diffusion-model "$VH" --vae "$VAE" --t5xxl "$T5"
    -p "$PROMPT" "${SAMP[@]}" -o "$w1odc/f%03d.png" -v)
fi
echo "=== WINDOW 1/$N [$MODE base] :: ${PROMPT:0:56}... ==="
run_window "w01" "$w1od" "$w1log"

# bank window-1 pixel tail = last K frames (bank_seg1.sh:27). Roll-forward state init (chain_long.sh:45).
w1tail="$WORK/tail-w01"
if [ "$DRY_RUN" = 1 ]; then
  echo "  [bank w01] tail = last $K frames of $w1od -> $w1tail   (latent -> $W1_LAT)"; echo
else
  rm -rf "$w1tail"; mkdir -p "$w1tail"; j=0
  for f in $(ls "$w1od"/f*.png 2>/dev/null | sort | tail -n "$K"); do cp "$f" "$(printf "$w1tail/c%03d.png" "$j")"; j=$((j+1)); done
fi
SEGDIRS=("$w1od")
PRIOR_LATENT="$W1_LAT"; PRIOR_TAIL_DIR="$w1tail"; PRIOR_DROP=0        # chain_long.sh:45 (window-2 drop=0)

# =====================================================================================
# WINDOWS 2..N  (fold-in of chain_long.sh loop) — SAME PROMPT reused every window.
# =====================================================================================
for ((win=2; win<=N; win++)); do
  wtag="w$(printf '%02d' "$win")"; wod="$WORK/$wtag"; wodc="/src/$WORKREL/$wtag"; wlog="$wod/log"
  LATOUT="/src/$WORKREL/bank/$wtag.bin"                               # VACE_SAVE_LATENT for this window
  echo "=== WINDOW $win/$N [VACE cont, drop_prior=$PRIOR_DROP] :: ${PROMPT:0:56}... ==="
  PRIOR_TAIL_C="/priortail"
  # chain_long.sh:55-65 continuation. --init-img only in i2v mode (identity anchor); none for t2v.
  CMD=(docker run --rm --gpus "$GPUS"
    -e VACE_CONT_FRAMES="$K" -e VACE_CONT_LATENT="$PRIOR_LATENT" -e VACE_CONT_LATENT_DROP_TAIL="$PRIOR_DROP"
    -e VACE_STRENGTH_TAIL="$TAIL" -e VACE_STRENGTH_ANCHOR_FRAMES="$ANCHOR"
    -e VACE_SKIP_BLOCKS=0 -e VACE_CONT_AGC=0 -e VACE_SAVE_LATENT="$LATOUT" "${NVENV[@]}"
    -v "$REPO:/src" -v "$M:/models" -v "$PRIOR_TAIL_DIR:$PRIOR_TAIL_C:ro" "${INIT_MNT[@]}" -w /src "$B"
    stdbuf -oL -eL /src/build/bin/sd-cli -M vid_gen
    --diffusion-model "$VL" --high-noise-diffusion-model "$VH" --vae "$VAE" --t5xxl "$T5"
    --control-video "$PRIOR_TAIL_C" -p "$PROMPT")
  [ "$MODE" = i2v ] && CMD+=(--init-img "$INIT_ARG")
  CMD+=("${SAMP[@]}" -o "$wodc/f%03d.png" -v)
  run_window "$wtag" "$wod" "$wlog"

  SEGDIRS+=("$wod")
  # roll forward: next tail = last K of KEPT frames (chain_long.sh:71-76); drop degraded latent tail.
  ntdir="$WORK/tail-$wtag"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [bank $wtag] keep=$((WINDOW-DISCARD)); tail = last $K of kept -> $ntdir ; next latent -> $LATOUT ; next drop=$DROPLAT"; echo
  else
    n=$(ls "$wod"/f*.png 2>/dev/null | wc -l); keep=$((n - DISCARD))
    rm -rf "$ntdir"; mkdir -p "$ntdir"; j=0
    for f in $(ls "$wod"/f*.png | sort | head -n "$keep" | tail -n "$K"); do cp "$f" "$(printf "$ntdir/c%03d.png" "$j")"; j=$((j+1)); done
  fi
  PRIOR_LATENT="$LATOUT"; PRIOR_TAIL_DIR="$ntdir"; PRIOR_DROP="$DROPLAT"
done

# =====================================================================================
# STITCH  (fold-in of chain_long.sh:79-97) — colour-lock + overlap-trim, then trim to SHOT_FRAMES.
# =====================================================================================
CL="$WORK/colourlocked"; ST="$WORK/stitch"; OUT_MP4="$OUTDIR/${OUTTAG}.mp4"
CLARGS=(); for d in "${SEGDIRS[@]}"; do CLARGS+=(--seg "$d"); done   # first --seg = implicit scene start
CL_CMD=(python3 tools/vace_colour_lock.py --out "$CL" --contrast "$CONTRAST" --tail "$K" --skip-head "$K" "${CLARGS[@]}")

echo "=== STITCH: colour-lock ($N segment dirs, contrast=$CONTRAST, skip-head=$K) then trim to $SHOT_FRAMES ==="
if [ "$DRY_RUN" = 1 ]; then
  echo "  colour-lock:"; pp_cmd "${CL_CMD[@]}"
  echo "  then: copy first $SHOT_FRAMES of $CL/*.png -> $ST/ , ffmpeg @16fps -> $OUT_MP4"
  echo
  echo "======================================================================================"
  echo "DRY_RUN complete: $N windows planned, env + roll-forward + stitch above. No GPU used."
  echo "======================================================================================"
  exit 0
fi

rm -rf "$CL" "$ST"; mkdir -p "$ST"
"${CL_CMD[@]}" || { echo "colour-lock failed"; exit 3; }
gi=0
for f in $(ls "$CL"/*.png 2>/dev/null | sort | head -n "$SHOT_FRAMES"); do cp "$f" "$(printf "$ST/s%05d.png" "$gi")"; gi=$((gi + 1)); done
[ "$gi" -lt "$SHOT_FRAMES" ] && echo "  WARN: only $gi frames available (< $SHOT_FRAMES target)"
ffmpeg -y -framerate 16 -pattern_type glob -i "$ST/*.png" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUT_MP4" -loglevel error 2>/dev/null

# ---- summary -------------------------------------------------------------------------
echo "======================================================================================"
echo "DONE: $OUT_MP4  ($gi frames, ~$(echo "scale=1;$gi/16" | bc)s @16fps)  frames: $ST"
TOTWALL=0
for ((i=0; i<${#WALLS[@]}; i++)); do
  w="${WALLS[$i]}"; p="${PEAKS[$i]}"
  printf "  window %2d: wall=%ss  peakVRAM=%sMiB\n" "$((i+1))" "$w" "$p"
  [[ "$w" =~ ^[0-9]+$ ]] && TOTWALL=$((TOTWALL + w))
done
echo "  total render wall: ${TOTWALL}s ($(echo "scale=1;$TOTWALL/60" | bc) min)  over $N windows"
echo "======================================================================================"
