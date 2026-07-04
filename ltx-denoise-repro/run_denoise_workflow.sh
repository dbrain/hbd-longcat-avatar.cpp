#!/usr/bin/env bash
###############################################################################
# run_denoise_workflow.sh — reproduce the "Denoise-AI" LTX-2.3 compressed
# workflow on OUR ggml/nvfp4 sd-cli stack.
#
# ---------------------------------------------------------------------------
# WHAT THE WORKFLOW DOES (reverse-engineered — see ../_CONTEXT.md + ./workflows)
# ---------------------------------------------------------------------------
# The Korean creator's ComfyUI graph produces clean high-motion faces where our
# single-pass distilled LTX goes mushy ("LTX poison"). Its trick is a
# LOW-RES-BASE -> SPATIAL x2 UPSCALE -> HIRES-REFINE ladder driven by a
# PARTIAL-strength distillation LoRA (not the full baked distill), plus NAG
# negative guidance and a detailer LoRA on the hires pass:
#
#   S2 base    544x960   sigmas 0.725,0.421875,0.0   dev + distill@0.65   cfg1
#     -- LTXVLatentUpsampler + ltx-2.3-spatial-upscaler-x2-1.1 -> 1088x1920 --
#   S4 refine  1088x1920 sigmas 0.421875,0.0          dev + distill@0.8
#                                                     + detailer@0.8       cfg1
#   fps=24  frames=121  sampler=euler  cfg=1 everywhere (guidance via NAG)
#
# (S1/S3 "preview" taps are inspection-only and dropped here — S2 base + the x2
#  upscale + S4 refine are the load-bearing stages.)
#
# ---------------------------------------------------------------------------
# HOW WE MAP IT ONTO sd-cli (the cheap win — see _CONTEXT.md "key insight")
# ---------------------------------------------------------------------------
# `models/ltx2/nvfp4-CLEAN.gguf` == dev + distill-lora@1.0 (that IS "distilled").
# So dev + distill@0.65 == distilled-1.1 + distill-lora@(0.65-1.0=-0.35).
# tools/fold_distill_lora.py (produced separately, by another agent) folds the
# distill LoRA at a chosen NEGATIVE strength onto nvfp4-CLEAN.gguf in place,
# yielding e.g. `nvfp4-CLEAN-dev065.gguf` == "dev + distill@0.65" — SAME nvfp4
# format / param count / kernels => SAME VRAM + per-step speed as prod, only
# less distillation baked in. The base+upscale ladder is what lets the weaker
# few-step distillation still converge.
#
#   base pass   : sd-cli -M vid_gen  <folded gguf>  --sigmas <base>   ...
#   x2 + refine : same call with  --hires --hires-upscaler <spatial-x2>
#                 --hires-sigmas <refine> --hires-steps <n>
#
# ---------------------------------------------------------------------------
# KNOWN GAPS vs the exact workflow (best current-CLI approximation inline):
#   * TODO(cpp:per-phase-lora) — workflow runs distill@0.65 on base but
#     distill@0.8 (+detailer@0.7/0.8) on the hires pass. Our --hires REUSES the
#     same model+LoRA. Approximation: ONE folded strength baked in the gguf for
#     both passes (default = 0.65). Detailer LoRA, if enabled, is appended to
#     the prompt and therefore ALSO leaks onto the base pass (can't scope yet).
#   * TODO(cpp:base-sigmas) — `--sigmas` exists in the parser but is not yet
#     confirmed wired to the LTX base pass; prod actually drives the base
#     schedule via the LTX_CUSTOM_SIGMAS env var. We pass BOTH so whichever is
#     live wins. Collapse to just --sigmas once it's plumbed.
#   * TODO(cpp:NAG) — NAG (normalized attention guidance) is not implemented;
#     the fork has CFG negative-prompt guidance only. The workflow runs cfg=1
#     with guidance done ENTIRELY by NAG, so with NAG missing we run cfg=1 and
#     rely on the folded partial-distill + ladder alone (closest current match).
#     A NEG prompt + cfg>1 is offered as a coarse stand-in but is NOT NAG.
#
# ---------------------------------------------------------------------------
# USAGE
#   ./run_denoise_workflow.sh <mode> [KEY=VAL ...]      (or MODE=<mode> env)
#   modes:  i2v_audio | t2v_audio | t2v_genaudio | chain
#     i2v_audio    : --init-img INIT + supplied audio (--drive-audio WAV)
#     t2v_audio    : no init image  + supplied audio (--drive-audio WAV)
#     t2v_genaudio : no init image, LTX generates its own audio natively
#     chain        : N segments, same characters (last-frame -> next --init-img;
#                    optional per-segment audio). See CHAIN section.
#   Every knob below is overridable as an env var, e.g.:
#     ./run_denoise_workflow.sh i2v_audio W=640 H=352 FPS=48 FR=193 TAG=busytest
#
#   *** THIS SCRIPT IS RUNNABLE-SHAPED BUT NOT RUN HERE (CPU-only work-ahead).***
#   *** No GPU / no render is invoked by drafting it; execute later on the box.***
###############################################################################
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."; REPO="$PWD"     # repo root (worktree)

# ---- fixed infra (mirrors run_ltx_t2v.sh / run_ltx_relip.sh) ---------------
BUILDER=${BUILDER:-longcat-avatar-dev:builder-cudnn-ff} # ffmpeg-in-image builder
LTXSRC=${LTXSRC:-/home/dbrain/dev/longcat-avatar.cpp}   # sibling checkout: sd-cli + models
LTX2="$LTXSRC/models/ltx2"                              # prod LTX-2.3 weights
GPU=${GPU:-1}                                           # --gpus device=N (dev card)
OUTDIR="$REPO/tools/lightx2v-test/out"; mkdir -p "$OUTDIR"  # eye-test docroot mp4s land here

# ===========================================================================
# KNOBS  (defaults = the Denoise-AI workflow's documented values)
# ===========================================================================
# First positional is the mode UNLESS it's a KEY=VAL (then fall back to $MODE env).
if [ "${1:-}" ] && [[ "$1" != *=* ]]; then MODE="$1"; shift; else MODE=${MODE:-i2v_audio}; fi
# absorb any remaining KEY=VAL args into the environment (so `mode W=640 ...` works)
for a in "$@"; do case "$a" in *=*) export "$a";; esac; done

# --- Model selection --------------------------------------------------------
# The folded "dev + distill@LORA_STRENGTH" gguf produced by fold_distill_lora.py.
# Does NOT need to exist yet — preflight only WARNS if absent (falls back below).
DIT=${DIT:-nvfp4-CLEAN-dev065.gguf}                    # == dev + distill@0.65 (fold @ -0.35)
DIT_FALLBACK=${DIT_FALLBACK:-nvfp4-CLEAN.gguf}         # == dev + distill@1.0 (prod, always present)
LORA_STRENGTH=${LORA_STRENGTH:-0.65}                   # doc only: the strength baked into DIT
# Optional hires detailer LoRA (LTX-2 19B — dim-compat with our 22B base UNKNOWN,
# left OFF by default; see _CONTEXT.md). Runtime LoRA via prompt <lora:name:str>.
DETAILER_LORA=${DETAILER_LORA:-}                        # e.g. ltx-2-19b-ic-lora-detailer:0.8 (empty=off)

# --- Base stage -------------------------------------------------------------
# Workflow low-res base is 544x960 PORTRAIT. For our landscape A/B ("busy clip")
# pass W=640 H=352 (or the GPU-phase target W=1280 H=704). x2 upscale doubles it.
W=${W:-544}; H=${H:-960}                                # low-res base resolution
FR=${FR:-121}                                          # video-frames (workflow=121; A/B uses 193)
FPS=${FPS:-24}                                         # workflow fps=24; expose 48 for the A/B
STEPS=${STEPS:-6}                                      # base sampler steps (S2 schedule = ~3 pts; euler few-step)
CFG=${CFG:-1.0}                                        # cfg=1 everywhere (NAG replaces CFG in the workflow)
SEED=${SEED:-42}
SAMPLER=${SAMPLER:-euler}                              # workflow sampler = euler
SCHED=${SCHED:-ltx2}                                   # LTX sigma scheduler (base for --sigmas fallback)
# TODO(cpp:base-sigmas) — S2 base ManualSigmas. Passed via --sigmas AND env.
BASE_SIGMAS=${BASE_SIGMAS:-0.725,0.421875,0.0}

# --- Upscale + hires refine -------------------------------------------------
HIRES=${HIRES:-1}                                      # 1 = run the x2 upscale + refine ladder
HIRES_UPSCALER=${HIRES_UPSCALER:-ltx-2.3-spatial-upscaler-x2-1.1}   # spatial x2 latent upsampler
HIRES_UPSCALERS_DIR=${HIRES_UPSCALERS_DIR:-/ltx2/latent_upscale_models}
HIRES_STEPS=${HIRES_STEPS:-4}                          # S4 refine steps (0 = reuse --steps)
# S3/S4 refine ManualSigmas. Default 2-pt; alt 3-pt "0.725,0.421875,0.0".
HIRES_SIGMAS=${HIRES_SIGMAS:-0.421875,0.0}

# --- Guidance (NAG stand-in) ------------------------------------------------
# TODO(cpp:NAG) — real NAG(scale14, a0.35, t2.5) is unimplemented. Default = the
# workflow's cfg=1 (guidance-off) which, with the partial-distill fold + ladder,
# is the closest faithful match. NEG+CFG>1 is a coarse CFG stand-in ONLY.
NEG=${NEG:-}                                           # negative prompt (only bites if CFG>1)

# --- Prompt / inputs --------------------------------------------------------
PROMPT=${PROMPT:-"a person on a city street at night, distant, walking, cinematic, photorealistic"}
INIT=${INIT:-models/_drive/char.png}                   # i2v init image (repo-relative)
WAV=${WAV:-models/_drive/song_vocals_16k.wav}          # supplied drive audio (16kHz mono)
MUXWAV=${MUXWAV:-$WAV}                                  # audio to mux back for the eye-test mp4
STRENGTH=${STRENGTH:-}                                 # i2v denoise strength (empty=model default)
TAG=${TAG:-denoise_$MODE}

# --- Chain (segment continuation) knobs -------------------------------------
CHAIN_N=${CHAIN_N:-3}                                  # number of chained segments
# One prompt per segment (newline-separated). Fewer than CHAIN_N reuses the last.
CHAIN_PROMPTS=${CHAIN_PROMPTS:-"the same person walks forward along the neon-lit street, cinematic, photorealistic
the same person stops and turns to face the camera, gentle motion, cinematic, photorealistic
the same person raises a hand and waves, warm smile, cinematic, photorealistic"}
CHAIN_AUDIO_DIR=${CHAIN_AUDIO_DIR:-}                   # dir of aud_<seg>.wav slices (empty = silent/gen chain)

# ---- prod compute/offload recipe (verbatim from the running LTX container) --
# nvfp4 cuBLASLt + FP8 FFN + F16 residual + cuDNN attn/conv3d + fusion glue +
# offload-overlap + shared/VAE residency. LTX_CUSTOM_SIGMAS = base-pass schedule
# (see TODO(cpp:base-sigmas)); here we drive it from BASE_SIGMAS.
PRODENV=(
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1
  -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks -e LTX_DIT_F16=1
  -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1
  -e LONGCAT_NO_OFFLOAD_PIPELINING=${NO_OVERLAP:-0} -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_PREFETCH_POOL=1
  -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=1
  -e LONGCAT_FFN_TILE_TOKENS=4096 -e LONGCAT_ENCODE_MAX_VRAM=6.5 -e LONGCAT_DIT_NO_MMAP=0
  -e LTXAV_END_RENDER_RECLAIM=1 -e LTXAV_CHAIN_POOL_TRIM=1
  -e "LTX_CUSTOM_SIGMAS=${LTX_CUSTOM_SIGMAS:-$BASE_SIGMAS}"   # TODO(cpp:base-sigmas)
)
MAXV=${MAXV:-9.5}                                       # --max-vram (offload knee; ≤11.5GB target)
VTILE=${VTILE:-1x1}                                    # VAE spatial tile (1x1 = full-frame, zero seams)
TT=${TT:-4}; TOV=${TOV:-1}                             # VAE temporal-tile frames / overlap (prod=4/1)

# ===========================================================================
# PREFLIGHT
# ===========================================================================
preflight() {
  echo "== preflight =="
  # -- model presence (WARN only; folded gguf is produced by another agent) --
  if [ ! -f "$LTX2/$DIT" ]; then
    echo "  WARN: folded model $LTX2/$DIT not found (produced by fold_distill_lora.py)."
    echo "        -> falling back to $DIT_FALLBACK (== dev+distill@1.0, full distill, no partial fold)."
    DIT="$DIT_FALLBACK"
  fi
  [ -f "$LTX2/$DIT" ]         || echo "  WARN: $LTX2/$DIT missing too — render will fail."
  [ -f "$LTX2/latent_upscale_models/${HIRES_UPSCALER}.safetensors" ] \
      || echo "  WARN: spatial upscaler ${HIRES_UPSCALER}.safetensors not in latent_upscale_models/."
  [ "$MODE" = i2v_audio ] && [ ! -f "$REPO/$INIT" ] && echo "  WARN: init image $INIT missing."
  case "$MODE" in i2v_audio|t2v_audio) [ ! -f "$REPO/$WAV" ] && echo "  WARN: drive audio $WAV missing.";; esac

  # -- GPU-idle / one-job-at-a-time gate (COMMENTED — enable on the box later) --
  #    Prevents stomping a foreground render; mirrors chain_long.sh wait_gpu().
  # local gate=${GPU_FREE_GATE:-4200}
  # while :; do u=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null||echo 0)
  #   [ "${u:-99999}" -lt "$gate" ] && break; echo "  ...waiting GPU-$GPU (${u}MiB>=$gate)"; sleep 15; done

  echo "  model=$DIT  mode=$MODE  ${W}x${H} base -> hires=$HIRES  ${FR}f @ ${FPS}fps"
  echo "  base_sigmas=$BASE_SIGMAS  hires_sigmas=$HIRES_SIGMAS  lora_strength(baked)=$LORA_STRENGTH"
  [ -n "$DETAILER_LORA" ] && echo "  NOTE(cpp:per-phase-lora): detailer <$DETAILER_LORA> leaks onto BASE too (can't scope to hires)."
  echo
}

# ===========================================================================
# COMMON sd-cli ARG BUILDER
#   $1 = init image path inside container (or "" for t2v)
#   $2 = drive-audio path inside container (or "" for genaudio/silent)
#   $3 = output path inside container (webm file, or f%03d.png for frames)
# Assembles the base+hires ladder identically for every mode.
# ===========================================================================
build_cmd() {
  local init="$1" audio="$2" out="$3"
  local -n _CMD=CMD; CMD=()

  # detailer LoRA is a runtime prompt tag (TODO(cpp:per-phase-lora): base+hires both)
  local prompt="$PROMPT"
  [ -n "$DETAILER_LORA" ] && prompt="$PROMPT <lora:${DETAILER_LORA}>"

  CMD=( -M vid_gen
    --diffusion-model       "/ltx2/$DIT"
    --vae                   "/ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors"
    --audio-vae             "/ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf"
    --llm                   "/ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf"
    --embeddings-connectors "/ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
    --lora-model-dir        "/ltx2/loras"
    -p "$prompt"
    -W "$W" -H "$H" --video-frames "$FR" --fps "$FPS"
    --sampling-method "$SAMPLER" --scheduler "$SCHED" --steps "$STEPS"
    --sigmas "$BASE_SIGMAS"                 # TODO(cpp:base-sigmas) — also driven via LTX_CUSTOM_SIGMAS env
    --cfg-scale "$CFG" --diffusion-fa
    -s "$SEED"
    --vae-tiling --vae-relative-tile-size "$VTILE"
    --offload-to-cpu --mmap --max-vram "$MAXV" -v
  )
  # TODO(cpp:NAG) — no NAG flag exists; NEG only bites when CFG>1 (coarse CFG stand-in, not NAG).
  [ -n "$NEG" ] && CMD+=( -n "$NEG" )
  [ "$TT" != "0" ] && CMD+=( --temporal-tiling --extra-tiling-args "temporal_tile_frames=$TT,temporal_tile_overlap=$TOV" )

  # base -> spatial x2 upscale -> hires refine ladder
  if [ "$HIRES" = "1" ]; then
    CMD+=( --hires
           --hires-upscalers-dir "$HIRES_UPSCALERS_DIR"
           --hires-upscaler      "$HIRES_UPSCALER"
           --hires-sigmas        "$HIRES_SIGMAS" )
    [ "$HIRES_STEPS" != "0" ] && CMD+=( --hires-steps "$HIRES_STEPS" )
  fi

  [ -n "$init"  ] && CMD+=( --init-img "$init" )
  [ -n "$init" ] && [ -n "$STRENGTH" ] && CMD+=( --strength "$STRENGTH" )
  [ -n "$audio" ] && CMD+=( --drive-audio "$audio" )   # supplied-audio flag for LTX-2.3
  CMD+=( -o "$out" )
}

# ===========================================================================
# RENDER WRAPPER — docker run + peak-VRAM sampler + wall timing.
#   $1 = log file, $2.. = sd-cli args (from CMD). Extra `-v mount` pairs via EXTRA_MOUNTS.
# ===========================================================================
EXTRA_MOUNTS=()
render() {
  local log="$1"; shift
  local vf; vf="$(mktemp)"; echo 0 >"$vf"
  ( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
      [ -n "$u" ] && [ "$u" -eq "$u" ] 2>/dev/null && { m=$(cat "$vf"); [ "$u" -gt "$m" ] && echo "$u">"$vf"; }; sleep 0.4; done ) & local sp=$!
  local t0; t0=$(date +%s.%N)
  docker run --rm --gpus "\"device=$GPU\"" \
    "${PRODENV[@]}" \
    "${EXTRA_MOUNTS[@]}" \
    -v "$LTXSRC:/ltxsrc" -v "$REPO:/src" -v "$LTX2:/ltx2" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src "$BUILDER" \
    stdbuf -oL -eL /ltxsrc/build/bin/sd-cli "$@" > "$log" 2>&1
  local rc=$?; local t1; t1=$(date +%s.%N); kill "$sp" 2>/dev/null
  local wall peak; wall=$(awk "BEGIN{printf \"%.1f\",$t1-$t0}"); peak=$(cat "$vf"); rm -f "$vf"
  echo ">> ${TAG} ${W}x${H} ${FR}f: ${wall}s peak ${peak}MiB rc=$rc"
  return $rc
}

# ===========================================================================
# EYE-TEST SURFACING — webm -> mp4 (KEEP audio), copy into the eye-test docroot,
# regen the index page. Reuses the run_ltx_relip.sh mux + gen_eyetest.sh regen.
#   $1 = produced webm ; $2 = optional audio to mux (empty = keep embedded audio)
# ===========================================================================
surface_eyetest() {
  local webm="$1" mux="${2:-}"
  local mp4="$OUTDIR/${TAG}.mp4"
  [ -f "$webm" ] || { echo "  no webm ($webm) — render failed, nothing to surface"; return 1; }
  if [ -n "$mux" ] && [ -f "$mux" ]; then
    # supplied-audio path: replace/attach the clean vocal track
    ffmpeg -y -i "$webm" -i "$mux" -map 0:v -map 1:a -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest \
      "$mp4" -loglevel error 2>/dev/null
  else
    # gen-audio path: transcode carrying whatever audio LTX baked into the webm
    ffmpeg -y -i "$webm" -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac \
      "$mp4" -loglevel error 2>/dev/null
  fi
  echo "  surfaced -> $mp4"
  regen_eyetest_index
}

# Rebuild a simple index.html listing every mp4 in the eye-test dir + (re)serve.
# (Pattern lifted from gen_eyetest.sh; PORT 8077 = the owner's desktop LAN page.)
regen_eyetest_index() {
  local PORT=${EYETEST_PORT:-8077}
  {
    echo '<!doctype html><meta charset=utf-8><title>LTX Denoise-AI workflow — eye test</title>'
    echo '<style>body{background:#111;color:#ddd;font:14px/1.5 system-ui;margin:24px}'
    echo 'h1{font-size:20px}.grid{display:flex;gap:18px;flex-wrap:wrap}'
    echo 'figure{margin:0;background:#000;padding:8px;border:1px solid #333;border-radius:6px}'
    echo 'video{max-height:520px;display:block}figcaption{color:#aaa;margin-top:6px;font-size:12px;max-width:340px}</style>'
    echo '<h1>LTX Denoise-AI compressed workflow — base → spatial x2 → hires refine</h1>'
    echo '<div class=grid>'
    local f
    for f in "$OUTDIR"/*.mp4; do [ -f "$f" ] || continue
      local b; b=$(basename "$f")
      printf '<figure><video src="%s" controls loop muted playsinline></video><figcaption>%s</figcaption></figure>\n' "$b" "$b"
    done
    echo '</div>'
  } > "$OUTDIR/index.html"
  echo "  wrote $OUTDIR/index.html"
  # (re)serve — commented so drafting/regen never spawns a server unattended.
  # pkill -f "http.server $PORT" 2>/dev/null || true
  # ( cd "$OUTDIR" && nohup python3 -m http.server "$PORT" --bind 0.0.0.0 >/dev/null 2>&1 & )
  # echo "  serving http://$(hostname -I | awk '{print $1}'):$PORT/"
}

# ===========================================================================
# MODES
# ===========================================================================
run_single() {  # $1 init(container path or "")  $2 audio(container path or "")
  local init="$1" audio="$2"
  local wdir="$REPO/mvp_out/$TAG"; mkdir -p "$wdir"
  local webm="/src/mvp_out/$TAG/out.webm"
  build_cmd "$init" "$audio" "$webm"
  render "$wdir/render.log" "${CMD[@]}"
  # supplied-audio modes mux MUXWAV; gen-audio keeps the embedded track.
  local mux=""; [ -n "$audio" ] && mux="$REPO/$MUXWAV"
  surface_eyetest "$wdir/out.webm" "$mux"
}

case "$MODE" in
  i2v_audio)     preflight; run_single "/src/$INIT" "/src/$WAV" ;;
  t2v_audio)     preflight; run_single ""            "/src/$WAV" ;;
  t2v_genaudio)  preflight; run_single ""            ""          ;;   # LTX generates audio natively (no --drive-audio)

  chain)
    # ------------------------------------------------------------------------
    # SEGMENT CONTINUATION — keep the same characters across N clips.
    #
    # Default approach: LAST-FRAME -> NEXT-INIT handoff (always expressible on
    # today's CLI). seg0 = i2v from INIT; extract its final frame as a PNG and
    # feed it as --init-img to seg1; repeat. Each segment runs the full base+
    # hires ladder. Per-segment audio via CHAIN_AUDIO_DIR/aud_<seg>.wav.
    #
    # NATIVE ALTERNATIVE (in-process, no per-seg reload, latent-tail continuity —
    # cleaner than pixel handoff): sd-cli has
    #     --ltx-chain-segments N  --ltx-chain-prompts <file>  --ltx-chain-audio-dir <dir>
    #     [--cont-latent <tail.bin> --cont-anchor <anchor.bin> --cont-latent-frames K]
    # Prefer that once you want latent-level continuity; wired as CHAIN_NATIVE=1.
    # ------------------------------------------------------------------------
    preflight
    mapfile -t CP < <(printf '%s\n' "$CHAIN_PROMPTS")
    prev_init="/src/$INIT"
    for ((i=0; i<CHAIN_N; i++)); do
      idx=$((i < ${#CP[@]} ? i : ${#CP[@]}-1)); PROMPT="${CP[$idx]}"
      TAG="denoise_chain_s$(printf %02d $i)"
      wdir="$REPO/mvp_out/$TAG"; mkdir -p "$wdir"
      # per-segment audio slice if provided (else silent/gen)
      seg_audio=""
      if [ -n "$CHAIN_AUDIO_DIR" ] && [ -f "$REPO/$CHAIN_AUDIO_DIR/aud_${i}.wav" ]; then
        seg_audio="/src/$CHAIN_AUDIO_DIR/aud_${i}.wav"
      fi
      echo "=== chain seg $((i+1))/$CHAIN_N :: ${PROMPT:0:60}... init=$prev_init ==="
      # emit FRAMES (not webm) so we can grab the last frame for the next init
      build_cmd "$prev_init" "$seg_audio" "/src/mvp_out/$TAG/f%03d.png"
      render "$wdir/render.log" "${CMD[@]}"
      # last-frame -> next-init handoff
      last=$(ls "$wdir"/f*.png 2>/dev/null | sort | tail -1)
      if [ -n "$last" ]; then
        cp "$last" "$wdir/last.png"; prev_init="/src/mvp_out/$TAG/last.png"
      else
        echo "  seg $i produced no frames — aborting chain"; break
      fi
      # stitch this segment to mp4 + surface (each seg is a row; final stitch TODO)
      ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$wdir/f*.png" \
        -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUTDIR/${TAG}.mp4" -loglevel error 2>/dev/null
    done
    regen_eyetest_index
    echo "chain done ($CHAIN_N segments). Concatenate $OUTDIR/denoise_chain_s*.mp4 for the full clip."
    ;;

  *) echo "unknown mode '$MODE' (want: i2v_audio | t2v_audio | t2v_genaudio | chain)"; exit 2 ;;
esac
