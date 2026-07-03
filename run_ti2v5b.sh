#!/usr/bin/env bash
# Wan2.2-TI2V-5B Turbo — RESIDENT few-step render (LTX-2.3 replacement candidate).
# Single dense 5B model (NO high/low-noise split), 48-channel Wan2.2 VAE, umT5.
# The bet: 5B fits 12GB fully resident (no --offload-to-cpu) -> fast + clean at 4-5 steps.
#
# Recipe (Kiijoku/civitai "wan-damme" consensus, ComfyUI-tuned, translated to our knobs):
#   4 steps, cfg 1, euler, scheduler simple, flow-shift ~8 (usable 6-18), 1280x704 / 704x1280.
#   Turbo distill is BAKED INTO the GGUF -> no LoRA needed for this path (School A direct).
#   "LatentMultiply 0.8" (oversaturation fix) has no flag here; only matters if output blows out.
#
# Usage:
#   ./run_ti2v5b.sh                              # PRESET=turbo, t2v, 4 steps, 1280x704, 69f, resident
#   PRESET=combo MODE=i2v ./run_ti2v5b.sh        # owner's validated base+2-LoRA combo (I2V = its strong suit)
#   PRESET=combo ./run_ti2v5b.sh                 # combo t2v
#   PRESET=schoolA ./run_ti2v5b.sh               # Turbo + negative Turbo-LoRA, 5 steps
#   MODE=i2v INIT=models/_drive/char.png ./run_ti2v5b.sh
#   FR=121 ./run_ti2v5b.sh                       # any knob overridable: STEPS SHIFT W H FR CFG SAMPLER SCHEDULER LORAS
# Fallback if output murks/blurs (wrong few-step schedule, same bug class as LTX simple!=trained):
#   try SCHEDULER=normal/beta, bump STEPS=5/6, or SDENV='-e WAN_DISTILL_SIGMAS=1' (DMD grid).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; M=/models
OUT="$REPO/ti2v5b_out"; mkdir -p "$OUT"

MODE="${MODE:-t2v}"

# ─── PRESET (recipe schools, researched 2026-06-15) ─────────────────────────────
# All share: cfg 1.0 · euler · scheduler simple · flow-shift 8 (flat 6–18, opt 8–12).
# Distill is I2V-biased (Kijai: "trained on I2V data", weaker T2V at 4 steps) -> I2V is its strong suit.
#   turbo   (default) Turbo GGUF, distill BAKED IN, NO lora, 4 steps, cfg1 shift8.   <- fastest
#   combo             BASE GGUF + FastWan(0.5) + Turbo(0.5) lora, 4 steps, cfg1.      <- owner's combo
#   schoolA           Turbo GGUF + Turbo lora at -0.2, 5 steps, cfg1.
#   base              BASE GGUF, NO lora, 50 steps, cfg 5.0, shift 5.0 (Wan OFFICIAL full-quality).
#                     The non-distilled quality CEILING — real CFG + many steps (slow). Sweep STEPS=20/30/50.
PRESET="${PRESET:-turbo}"
TURBO_LORA="Wan22_TI2V_5B_Turbo_lora_rank_64_fp16"
FAST_LORA="Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16"
case "$PRESET" in
  turbo)   _MODEL=wan22-ti2v-5b-turbo-q8_0.gguf; _STEPS=4;  _CFG=1; _SHIFT=8; _LORAS="" ;;
  combo)   _MODEL=wan22-ti2v-5b-base-q8_0.gguf;  _STEPS=4;  _CFG=1; _SHIFT=8; _LORAS="<lora:${FAST_LORA}:0.5><lora:${TURBO_LORA}:0.5>" ;;
  schoolA) _MODEL=wan22-ti2v-5b-turbo-q8_0.gguf; _STEPS=5;  _CFG=1; _SHIFT=8; _LORAS="<lora:${TURBO_LORA}:-0.2>" ;;
  base)    _MODEL=wan22-ti2v-5b-base-q8_0.gguf;  _STEPS=50; _CFG=5; _SHIFT=5; _LORAS="" ;;
  fast)    _MODEL=wan22-ti2v-5b-base-q8_0.gguf;  _STEPS=4;  _CFG=1; _SHIFT=8; _LORAS="<lora:${FAST_LORA}:${FW:-0.5}>" ;;          # FastWan ONLY (no Turbo) — isolate the motion/blur LoRA
  fastturbo) _MODEL=wan22-ti2v-5b-base-q8_0.gguf; _STEPS=4; _CFG=1; _SHIFT=8; _LORAS="<lora:${FAST_LORA}:${FW:-0.5}><lora:${TURBO_LORA}:${TW:-0.5}>" ;; # like combo but FW/TW strengths overridable
  *) echo "unknown PRESET=$PRESET (turbo|combo|schoolA|base|fast|fastturbo)"; exit 2 ;;
esac

# NOTE: use the REPACKED ggufs (tools/repack_ti2v5b_patch_embed.py) — the raw Kiijoku/QuantStack
# *-TI2V-5B-*.gguf store patch_embedding 5D, which our 4D-max loader can't read.
MODEL="${MODEL:-$_MODEL}"
VAE="${VAE:-wan2.2-vae-48ch-f16.gguf}"
T5="${T5:-longcat-umt5-xxl-q8_0.gguf}"
LORAS="${LORAS:-$_LORAS}"                        # <lora:STEM:mult> tags; resolved from --lora-model-dir
STEPS="${STEPS:-$_STEPS}"; CFG="${CFG:-$_CFG}"; SHIFT="${SHIFT:-$_SHIFT}"
SAMPLER="${SAMPLER:-euler}"; SCHEDULER="${SCHEDULER:-simple}"
W="${W:-1280}"; H="${H:-704}"; FR="${FR:-69}"; FPS="${FPS:-24}"; SEED="${SEED:-42}"
# umT5 (6GB q8) + DiT (5.15GB) + VAE (1.4GB) all-resident OOMs 12GB. CLIP_ON_CPU=1 runs the text
# encoder on CPU once/clip (frees ~6GB GPU) so the DiT stays FULLY RESIDENT — the whole 5B bet.
CLIP_ON_CPU="${CLIP_ON_CPU:-1}"
INIT="${INIT:-models/_drive/char.png}"          # only used when MODE=i2v
STRENGTH="${STRENGTH:-1.0}"
LABEL="${LABEL:-ti2v5b_${PRESET}_${MODE}_$(basename "$MODEL" .gguf)_s${STEPS}_sh${SHIFT}_${W}x${H}_${FR}f}"
SDENV="${SDENV:-}"                               # e.g. SDENV='-e WAN_DISTILL_SIGMAS=1'
PROMPT="${PROMPT:-A young man with tousled dark brown hair and light stubble in a faded blue denim jacket over a white t-shirt sings energetically into the camera, bobbing his head to an upbeat rock song, shoulders swaying. Locked static medium shot. Warm amber neon glows behind him outside a corner bar at dusk, damp street reflecting the lights, soft bokeh, cinematic, volumetric light, high detail.}"

FRAMES="$OUT/$LABEL.frames"; rm -rf "$FRAMES"; mkdir -p "$FRAMES"

# I2V init-image (TI2V-5B has an explicit IMG2VID path)
INIT_ARGS=()
if [ "$MODE" = "i2v" ]; then INIT_ARGS=(--init-img "/models/_drive/$(basename "$INIT")" --strength "$STRENGTH"); fi

# LoRA wiring: tags go in the prompt (<lora:STEM:mult>), resolved from --lora-model-dir.
# Our fork maps the Kijai format (diffusion_model. prefix, lora_down/up, diff/diff_b) — verified.
LORA_ARGS=()
if [ -n "$LORAS" ]; then LORA_ARGS=(--lora-model-dir "$M"); PROMPT="$PROMPT $LORAS"; fi

# Text encoder on CPU (frees ~6GB GPU so DiT stays resident)
CLIP_ARGS=()
if [ "$CLIP_ON_CPU" = "1" ]; then CLIP_ARGS=(--clip-on-cpu); fi

# Custom DMD/distill sigma schedule (overrides --scheduler/--steps). FastWan FullAttn trained
# grid = DMD timesteps 1000,757,522 -> flow sigmas 1.0,0.757,0.522 + terminal 0. Set SIGMAS to use.
SIG_ARGS=()
if [ -n "${SIGMAS:-}" ]; then SIG_ARGS=(--sigmas "$SIGMAS"); fi

# peak-VRAM sampler
vf="$OUT/.vram_$LABEL"; echo 0 > "$vf"
( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); \
    [ -n "$u" ] && { m=$(cat "$vf"); [ "$u" -gt "$m" ] && echo "$u">"$vf"; }; sleep 0.3; done ) & sp=$!

echo ">> $LABEL"
echo ">> preset=$PRESET model=$MODEL steps=$STEPS cfg=$CFG shift=$SHIFT sampler=$SAMPLER sched=$SCHEDULER ${W}x${H} ${FR}f RESIDENT ${LORAS:+loras:$LORAS} ${SDENV:+env:$SDENV}"
t0=$(date +%s.%N)
docker run --rm --gpus all $SDENV -v "$REPO:/src" -v "$REPO/models:/models" -w /src "$BUILDER" \
  /src/build/bin/sd-cli -M vid_gen \
  --diffusion-model $M/$MODEL \
  --vae $M/$VAE --t5xxl $M/$T5 \
  "${LORA_ARGS[@]}" "${CLIP_ARGS[@]}" "${SIG_ARGS[@]}" \
  -p "$PROMPT" --cfg-scale $CFG -s $SEED \
  --sampling-method $SAMPLER --scheduler $SCHEDULER \
  --steps $STEPS --flow-shift $SHIFT \
  -W $W -H $H --video-frames $FR --diffusion-fa --mmap \
  --vae-tiling --temporal-tiling --vae-relative-tile-size ${VAE_TILE:-0.25x0.25} --vae-tile-overlap ${VAE_OVERLAP:-0.25} \
  "${INIT_ARGS[@]}" \
  -o /src/ti2v5b_out/$LABEL.frames/f%03d.png -v > "$OUT/$LABEL.log" 2>&1
rc=$?; t1=$(date +%s.%N); kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"); peak=$(cat "$vf"); rm -f "$vf"
secvid=$(awk "BEGIN{printf \"%.2f\", $FR/$FPS}")
rps=$(awk "BEGIN{printf \"%.1f\", $wall/($FR/$FPS)}")
echo ">> $LABEL: wall=${wall}s peak=${peak}MiB rc=$rc  (${secvid}s video -> ${rps}s render/sec-video)"
# sidecar for the eye-test page
printf 'preset=%s model=%s mode=%s steps=%s cfg=%s shift=%s sampler=%s sched=%s res=%sx%s frames=%s seed=%s\nwall=%ss peak=%sMiB rc=%s render_per_svid=%ss%s\n' \
  "$PRESET" "$MODEL" "$MODE" "$STEPS" "$CFG" "$SHIFT" "$SAMPLER" "$SCHEDULER" "$W" "$H" "$FR" "$SEED" \
  "$wall" "$peak" "$rc" "$rps" "${LORAS:+ loras=$LORAS}" > "$OUT/$LABEL.meta"
grep -E "Wan2.2-TI2V-5B|get_learned_condition completed|sampling completed|decode_first_stage completed|generate_video completed|out of memory|error" "$OUT/$LABEL.log" | tail -10
if ls "$FRAMES"/f*.png >/dev/null 2>&1; then
  ffmpeg -y -framerate $FPS -pattern_type glob -i "$FRAMES/*.png" -c:v libx264 -pix_fmt yuv420p "$OUT/$LABEL.mp4" -loglevel error \
    && echo ">> wrote $OUT/$LABEL.mp4"
fi
