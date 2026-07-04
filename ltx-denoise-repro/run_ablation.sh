#!/usr/bin/env bash
###############################################################################
# run_ablation.sh — the ONE turnkey runner for the LTX-Denoise ablation.
#
#   bash run_ablation.sh <scenario:1|2|3> <mode:t2v|i2v> ["rows"]
#     scenario : 1=walk-toward-cam  2=dance-in-crowd  3=static-cam crossing (see PROMPTS.md)
#     mode     : t2v (generate scene) | i2v (animate INIT still)
#     rows     : space list, default "0 1 2 3".  0 baseline · 1 fold · 2 +ladder
#                · 3 +hires-lora/detailer · 4 +NAG · 5 24fps(winner)
#   env: FR=193 FPS=48 SEED=42 DEVICE=1 MAXV=11
#        AUDIO=none|<wav>   (default: t2v=none[native gen] ; set to a wav for --drive-audio)
#        INIT=<png>         (i2v start still; default = flux_neon_seed7 for scenario 1)
#        NAGMODEL / WINMODEL to pin the model for rows 4 / 5.
#
# Runs each row through the SAME prompt/seed (only model+flags change), emits an AV
# webm, muxes to mp4 (keeps audio), drops it on the eye-test page, and rebuilds an
# ablation results table. ONE GPU job at a time (flock + idle preflight).
#
# Binary = the WORKTREE build (NAG + --hires-lora): /src/build-cudnn/bin/sd-cli, run in
# the prod runtime image longcat-avatar-dev:builder-cudnn-ff. Models from the main
# checkout's models/ltx2 (folded ggufs produced by tools/fold_distill_lora.py).
###############################################################################
set -uo pipefail

SCN="${1:?scenario 1|2|3}"; MODE="${2:?t2v|i2v}"; ROWS="${3:-0 1 2 3}"
FR="${FR:-193}"; FPS="${FPS:-48}"; SEED="${SEED:-42}"; DEVICE="${DEVICE:-1}"; MAXV="${MAXV:-11}"

WT=/home/dbrain/dev/longcat-avatar-ltxdenoise          # worktree (binary lives here)
MAIN=/home/dbrain/dev/longcat-avatar.cpp; LTX2="$MAIN/models/ltx2"
BUILDER="longcat-avatar-dev:builder-cudnn-ff"; BIN=/src/build-cudnn/bin/sd-cli
EYE=/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise; CLIPS="$EYE/clips"
OUTROOT="$WT/_ablation_out"; mkdir -p "$CLIPS" "$OUTROOT"
UPDIR=/ltx2/latent_upscale_models; UPSCALER=ltx-2.3-spatial-upscaler-x2-1.1
LADDER_SIGMAS="1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0"   # base pass (full schedule; low-res)
FULL_SIGMAS="1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0"     # prod single-pass default
HIRES_SIGMAS="0.725,0.421875,0.0"; NEG_DEFAULT="blurry, mushy warped face, distorted anatomy, extra limbs, low quality, jpeg artifacts, text, watermark, flicker"

# ---- prompts (verbatim from PROMPTS.md; keep byte-identical across rows) ----
case "$SCN:$MODE" in
  1:t2v) PROMPT="Wide cinematic shot on a rain-slicked city street at night, shot on a 35mm lens with a shallow depth of field. Neon signage in magenta and cyan reflects across the wet asphalt while a faint mist softens the distant traffic lights. A man in his early thirties in a dark wool coat over a grey hoodie starts far down the block as a small silhouette and walks steadily toward the camera, his stride even and unhurried, hands loose at his sides. As he closes the distance his features resolve into focus — a short beard, tired eyes, breath fogging faintly in the cold air — until his face nearly fills the frame. The camera holds a slow, locked eyeline and lets him come to it rather than moving to meet him. Ambient city sound of tyres hissing on wet road, a low neon hum, and footsteps growing louder." ;;
  1:i2v) PROMPT="The figure begins to walk steadily toward the camera, closing the distance with an even, unhurried stride, hands loose at his sides. As he approaches, his face resolves into sharper focus and his expression softens into a faint, tired half-smile, breath fogging in the cold air. The camera stays locked on his eyeline, holding still as he comes to it, while neon reflections slide across the wet street behind him. Ambient sound of tyres on wet asphalt and footsteps growing louder." ;;
  2:t2v) PROMPT="Medium-wide shot at a crowded outdoor night concert, shot on a 50mm lens with warm stage light spilling from behind. A young woman with long dark hair in a fringed denim jacket dances at the centre of the frame, spinning on one heel, stepping side to side and throwing her arms up to the rhythm, her movements fluid and full-bodied. Behind her a dense crowd of silhouetted onlookers sways and claps, kept soft and slightly out of focus by the shallow depth of field so the eye stays on her. Amber and gold light rims her figure while cooler blues wash the background and dust motes drift through the beams. The camera holds an almost-static frame with only the faintest push-in, letting her motion carry the shot. Ambient sound of a live crowd, rhythmic music, and cheering." ;;
  2:i2v) PROMPT="The woman breaks into an energetic dance, spinning on one heel, stepping side to side and throwing her arms up to the beat, her movements fluid and full-bodied. The crowd behind her sways and claps, kept soft by the shallow depth of field. Warm stage light rims her as she moves and the camera holds an almost-static frame with the faintest push-in. Ambient live-crowd noise, rhythmic music, and cheering." ;;
  3:t2v) PROMPT="Locked-off wide shot of a pedestrian crossing on a busy daytime city street, shot on a 35mm lens at eye level from the far kerb; the camera does not move. Cool overcast daylight flattens the scene and wet pavement holds pale reflections of the surrounding buildings. A woman in a red raincoat and jeans walks briskly from the left of the frame across the zebra crossing to the right, her gait natural and purposeful, a canvas bag swinging at her shoulder. Behind her, blurred traffic waits at the light and a few other pedestrians drift at the edges of the frame. She stays at a middle distance, small in the wide composition, never dominating the frame, her figure crisp against the muted street. Ambient city sound of idling engines, distant chatter, and the rhythmic tap of her footsteps on the crossing." ;;
  3:i2v) PROMPT="The woman steps off the kerb and walks briskly across the zebra crossing from left to right, her gait natural and purposeful, a canvas bag swinging at her shoulder. The camera stays completely locked-off and does not move while blurred traffic waits behind her. She holds a middle distance, small and crisp in the wide frame. Ambient city sound of idling engines, distant chatter, and footsteps tapping on the crossing." ;;
  *) echo "bad scenario:mode '$SCN:$MODE'"; exit 2 ;;
esac

# ---- i2v start still (host path -> /ltx2 mount) ----
INIT_HOST="${INIT:-$LTX2/_inputs/flux_neon_seed7.png}"
if [ "$MODE" = i2v ]; then
  [ -f "$INIT_HOST" ] || { echo "[abort] i2v INIT still not found: $INIT_HOST (flux one first — see runbook)"; exit 3; }
  INIT_ARG="--init-img /ltx2/_inputs/$(basename "$INIT_HOST")"
  # ensure it's under _inputs so the /ltx2 mount sees it
  case "$INIT_HOST" in "$LTX2/_inputs/"*) : ;; *) cp -f "$INIT_HOST" "$LTX2/_inputs/"; INIT_ARG="--init-img /ltx2/_inputs/$(basename "$INIT_HOST")";; esac
else INIT_ARG=""; fi

# ---- supplied audio -> 16k mono wav for --drive-audio (default: none = native gen audio) ----
AUDIO="${AUDIO:-none}"; DRIVE_ARG=""
if [ "$AUDIO" != none ]; then
  [ -f "$AUDIO" ] || { echo "[abort] AUDIO not found: $AUDIO"; exit 3; }
  a16="$LTX2/_drive/$(basename "${AUDIO%.*}")_16k.wav"
  ffmpeg -y -i "$AUDIO" -ac 1 -ar 16000 "$a16" -loglevel error && echo "[audio] 16k mono -> $a16"
  DRIVE_ARG="--drive-audio /ltx2/_drive/$(basename "$a16")"
fi

# ---- one-job-at-a-time + GPU-idle preflight (owner's work wins; tolerate kobbler /app/api) ----
exec 9>/tmp/ltx_ablation.lock; flock -n 9 || { echo "[abort] another ablation holds the lock"; exit 9; }
busy="$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$1 !~ /\/app\/api/ && $2+0 > 300')"
[ -n "$busy" ] && { echo "[abort] GPU busy (owner's work?) — refusing:"; echo "$busy"|sed 's/^/    /'; exit 9; }
echo "[preflight] GPU idle — proceeding. scenario=$SCN mode=$MODE rows=[$ROWS]"

PRODENV=( -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks -e LTX_DIT_F16=1
  -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1
  -e LONGCAT_NO_OFFLOAD_PIPELINING=0 -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_PREFETCH_POOL=1
  -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=1 -e LONGCAT_FFN_TILE_TOKENS=4096 -e LONGCAT_ENCODE_MAX_VRAM=6.5 -e LONGCAT_DIT_NO_MMAP=0
  -e LTXAV_END_RENDER_RECLAIM=1 -e LTXAV_CHAIN_POOL_TRIM=1 )

render() { # render TAG MODEL WBASE HBASE STEPS SIGMAS "HIRES" "HIRESLORA" "NAG" FPS_OVR
  local TAG="$1" MODEL="$2" W="$3" H="$4" STEPS="$5" SIGMAS="$6" HIRES="$7" HLORA="$8" NAG="$9" FPSR="${10:-$FPS}"
  local od="$OUTROOT/$TAG"; rm -rf "$od"; mkdir -p "$od"
  [ -f "$LTX2/$MODEL" ] || { echo "  [skip $TAG] model missing: $MODEL (fold it first)"; return 0; }
  echo "=== ROW $TAG :: $MODEL ${W}x${H}${HIRES:+ ->x2} ${FR}f@${FPSR} steps=$STEPS ==="
  local vf="$od/.vram"; echo 0 >"$vf"
  ( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE" 2>/dev/null); [ -n "$u" ]&&[ "$u" -eq "$u" ] 2>/dev/null&&{ m=$(cat "$vf"); [ "$u" -gt "$m" ]&&echo "$u">"$vf"; }; sleep 0.4; done ) & local sp=$!
  local t0=$(date +%s.%N)
  docker run --rm --gpus "\"device=$DEVICE\"" "${PRODENV[@]}" -e "LTX_CUSTOM_SIGMAS=$SIGMAS" \
    -v "$WT:/src" -v "$LTX2:/ltx2" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src "$BUILDER" \
    stdbuf -oL -eL "$BIN" -M vid_gen \
    --diffusion-model /ltx2/$MODEL \
    --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors \
    --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
    --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf \
    --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
    --lora-model-dir /ltx2/loras \
    -p "$PROMPT" $NAG ${NEGP:+-n "$NEGP"} $INIT_ARG $DRIVE_ARG \
    -W "$W" -H "$H" --video-frames "$FR" --fps "$FPSR" \
    --sampling-method euler --steps "$STEPS" --cfg-scale 1.0 --diffusion-fa \
    --vae-tiling --vae-relative-tile-size 1x1 --temporal-tiling --extra-tiling-args temporal_tile_frames=4,temporal_tile_overlap=1 \
    $HIRES $HLORA \
    --offload-to-cpu --mmap --max-vram "$MAXV" -s "$SEED" -v \
    -o /src/_ablation_out/$TAG/out.webm > "$od/log" 2>&1
  local rc=$?; local t1=$(date +%s.%N); kill "$sp" 2>/dev/null
  local wall=$(awk "BEGIN{printf \"%.1f\",$t1-$t0}"); local peak=$(cat "$vf" 2>/dev/null); rm -f "$vf"
  if [ -f "$od/out.webm" ]; then
    ffmpeg -y -i "$od/out.webm" -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac "$CLIPS/$TAG.mp4" -loglevel error 2>/dev/null && echo "  -> $CLIPS/$TAG.mp4  (${wall}s, peak ${peak}MiB, rc=$rc)"
  else echo "  [FAIL $TAG] no webm (rc=$rc):"; grep -aiE "out of memory|error|abort|assert" "$od/log"|grep -aviE 0x|tail -3; fi
  echo "$TAG,$MODEL,${W}x${H}$([ -n "$HIRES" ]&&echo '->x2'),$FR,$FPSR,$STEPS,$wall,$peak,$rc" >> "$EYE/results.csv"
}

LAD="--hires --hires-upscaler $UPSCALER --hires-upscalers-dir $UPDIR --hires-steps 2 --hires-sigmas $HIRES_SIGMAS"
# detailer-only by default: the rank-384 distill LoRA (7.6 GB) runtime-applied on the hires pass
# OOMs the 16 GB card (dev065 already bakes 0.65). Override HLORA3 to add it if you fold headroom.
HLORA3="${HLORA3:---hires-lora ltx-2-19b-ic-lora-detailer:0.8}"
NAG4="--nag-scale 13 --nag-alpha 0.35 --nag-tau 2.5 --nag-until-sigma 0.9"   # neg prompt passed via NEGP (proper quoting)

echo "scenario,mode=$SCN,$MODE seed=$SEED"
# only reset the results table on a fresh batch (row 0 present); otherwise append/refresh
case " $ROWS " in *" 0 "*) : > "$EYE/results.csv" ;; esac
for R in $ROWS; do case "$R" in
  0) render "s${SCN}_${MODE}_r0_baseline"  nvfp4-CLEAN.gguf         1280 704 8 "$FULL_SIGMAS"   ""     ""        "" ;;
  1) render "s${SCN}_${MODE}_r1_fold050"   nvfp4-CLEAN-dev050.gguf  1280 704 8 "$FULL_SIGMAS"   ""     ""        "" ;;
  2) render "s${SCN}_${MODE}_r2_ladder"    nvfp4-CLEAN-dev050.gguf   640 352 8 "$LADDER_SIGMAS" "$LAD" ""        "" ;;
  3) render "s${SCN}_${MODE}_r3_hireslora" nvfp4-CLEAN-dev065.gguf   640 352 8 "$LADDER_SIGMAS" "$LAD" "$HLORA3" "" ;;
  4) NEGP="$NEG_DEFAULT" render "s${SCN}_${MODE}_r4_nag" "${NAGMODEL:-nvfp4-CLEAN-dev050.gguf}" 640 352 8 "$LADDER_SIGMAS" "$LAD" "" "$NAG4" ;;
  5) render "s${SCN}_${MODE}_r5_24fps"     "${WINMODEL:-nvfp4-CLEAN-dev050.gguf}" 640 352 8 "$LADDER_SIGMAS" "$LAD" "" "" 24 ;;  # FR override below
  *) echo "unknown row $R";; esac
done

# 24fps row uses half the frames for the same ~4s duration (the "50% less work" test)
# (handled by passing FR=97 FPS=24 when you run:  FR=97 bash run_ablation.sh $SCN $MODE 5)

# ---- rebuild the ablation results page (auto; turnkey) ----
LBL=( ["0"]="r0 baseline (current prod nvfp4-CLEAN, single pass)" ["1"]="r1 fold (dev+distill@0.5, single pass — isolates de-distillation)"
      ["2"]="r2 +ladder (low-res base -> spatial x2 -> refine)" ["3"]="r3 +hires-lora (distill@0.8 + detailer on refine)"
      ["4"]="r4 +NAG (scale13, until-sigma0.9)" ["5"]="r5 24fps winner (half the frames, ~same 4s)" )
{
  echo "<!doctype html><meta charset=utf-8><title>LTX-Denoise ablation — s${SCN} ${MODE}</title>"
  echo "<style>body{background:#111;color:#ddd;font:14px/1.6 system-ui,sans-serif;margin:24px;max-width:1200px}h1{font-size:20px}h2{font-size:14px;color:#9cf}table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #333;padding:4px 8px}th{background:#1c1c1c}.ok{color:#7d7}.bad{color:#f77}.grid{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;background:#000;padding:8px;border:1px solid #333;border-radius:6px}video{max-height:420px;display:block;border-radius:4px}figcaption{color:#aaa;margin-top:6px;font-size:12px;max-width:360px}.p{color:#888;font-size:12px;max-width:1100px}</style>"
  echo "<h1>LTX-Denoise ablation — scenario ${SCN}, ${MODE}, seed ${SEED}</h1>"
  echo "<p class=p><b>Prompt (fixed across all rows):</b> ${PROMPT}</p>"
  echo "<h2>timing &amp; VRAM</h2><table><tr><th>row</th><th>model</th><th>res</th><th>frames</th><th>fps</th><th>steps</th><th>wall</th><th>peak</th><th>status</th></tr>"
  while IFS=, read -r tag model res fr fps steps wall peak rc; do
    st="ok"; [ "${rc:-1}" != 0 ] && st="FAIL"
    echo "<tr><td>${tag##*_r}</td><td>${model%.gguf}</td><td>$res</td><td>$fr</td><td>$fps</td><td>$steps</td><td>${wall}s</td><td>${peak}MiB</td><td class=\"$([ $st = ok ]&&echo ok||echo bad)\">$st</td></tr>"
  done < "$EYE/results.csv"
  echo "</table><h2>clips (play in motion — mush shows in motion, not stills; 🔊 = has audio)</h2><div class=grid>"
  for f in $(cd "$CLIPS" && ls s${SCN}_${MODE}_r*.mp4 2>/dev/null | sort); do
    R=$(echo "$f" | sed -E 's/.*_r([0-9]+)_.*/\1/')
    echo "<figure><video src=\"clips/$f\" controls loop playsinline></video><figcaption>${LBL[$R]:-$f}</figcaption></figure>"
  done
  echo "</div>"
} > "$EYE/ablation.html"
echo "@@@ ablation done. clips: $CLIPS  |  page: http://10.0.0.208:8077/ltx_denoise/ablation.html @@@"
