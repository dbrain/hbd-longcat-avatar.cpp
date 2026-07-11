#!/usr/bin/env bash
# run_weights_sweep.sh — single-segment LTX-2.3 production-upscale comparison.
#
# Produces/updates:
#   http://10.0.0.208:8077/ltx_denoise/weights_imatrix_upscale.html
#
# Matrix:
#   prompts: PROMPTS.md t2v scenarios 1/2/3, plus flux_neon_seed7 i2v energetic dance
#   DiT: nvfp4-imatrix-dev050
#   rows: 960x544 -> production x2 and x1.5 latent upscale/refine, seeds 7/42/123.
# Defaults intentionally mirror run_singing_clip.sh's locked working set.
set -uo pipefail

WT=${WT:-/home/dbrain/dev/longcat-avatar-ltxdenoise}
LTX2=${LTX2:-/home/dbrain/dev/longcat-avatar.cpp/models/ltx2}
EYE=${EYE:-/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise}
BUILDER=${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}
FFMPEG=${FFMPEG:-linuxserver/ffmpeg}
BIN=${BIN:-/src/build-sa3/bin/sd-cli}
GPU=${GPU:-1}
MAXV=${MAXV:-9}
TIMEOUT=${TIMEOUT:-500}
SA3_MODE=${SA3_MODE:-native}
SA3_DELTA_F16=${GGML_LTX_SA3_DELTA_F16:-1}
SWEEP_LABEL=${SWEEP_LABEL:-Locked SA3 defaults}

FR=${FR:-97}
FPS=${FPS:-24}
STEPS=${STEPS:-8}
REFSTEPS=${REFSTEPS:-3}
CFG=${CFG:-1.0}
SEED_LIST=${SEED_LIST:-${SEEDS:-7 42 123}}
read -r -a SEEDS <<< "$SEED_LIST"
SCENARIO_LIST=${SCENARIO_LIST:-s1_t2v s2_t2v s3_t2v neon_dance_i2v}
read -r -a SCENARIOS <<< "$SCENARIO_LIST"

OUTROOT=${OUTROOT:-$WT/ltx-denoise-repro/_weights_sweep_imatrix_upscale}
CLIP_SUBDIR=${CLIP_SUBDIR:-sweep_weights_imatrix_upscale}
CLIPDIR=$EYE/$CLIP_SUBDIR
MANIFEST=${MANIFEST:-$EYE/weights_imatrix_upscale.tsv}
PAGE=${PAGE:-$EYE/weights_imatrix_upscale.html}
mkdir -p "$OUTROOT" "$CLIPDIR" "$EYE"
touch "$MANIFEST"

DIT_LIST=${DIT_LIST:-nvfp4-imatrix-dev050.gguf}
VARIANT_LIST=${VARIANT_LIST:-upscale_x2 upscale_x15}
read -r -a DITS <<< "$DIT_LIST"
read -r -a VARIANTS <<< "$VARIANT_LIST"
export DIT_LIST VARIANT_LIST SEED_LIST SCENARIO_LIST SWEEP_LABEL

BASE_SIGMAS="1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0"
REFINE_SIGMAS="0.909375,0.725,0.421875,0.0"
UPDIR=/ltx2/latent_upscale_models

PROMPT_s1_t2v="Wide cinematic shot on a rain-slicked city street at night, shot on a 35mm lens with a shallow depth of field. Neon signage in magenta and cyan reflects across the wet asphalt while a faint mist softens the distant traffic lights. A man in his early thirties in a dark wool coat over a grey hoodie starts far down the block as a small silhouette and walks steadily toward the camera, his stride even and unhurried, hands loose at his sides. As he closes the distance his features resolve into focus - a short beard, tired eyes, breath fogging faintly in the cold air - until his face nearly fills the frame. The camera holds a slow, locked eyeline and lets him come to it rather than moving to meet him. Ambient city sound of tyres hissing on wet road, a low neon hum, and footsteps growing louder."
PROMPT_s2_t2v="Medium-wide shot at a crowded outdoor night concert, shot on a 50mm lens with warm stage light spilling from behind. A young woman with long dark hair in a fringed denim jacket dances at the centre of the frame, spinning on one heel, stepping side to side and throwing her arms up to the rhythm, her movements fluid and full-bodied. Behind her a dense crowd of silhouetted onlookers sways and claps, kept soft and slightly out of focus by the shallow depth of field so the eye stays on her. Amber and gold light rims her figure while cooler blues wash the background and dust motes drift through the beams. The camera holds an almost-static frame with only the faintest push-in, letting her motion carry the shot. Ambient sound of a live crowd, rhythmic music, and cheering."
PROMPT_s3_t2v="Locked-off wide shot of a pedestrian crossing on a busy daytime city street, shot on a 35mm lens at eye level from the far kerb; the camera does not move. Cool overcast daylight flattens the scene and wet pavement holds pale reflections of the surrounding buildings. A woman in a red raincoat and jeans walks briskly from the left of the frame across the zebra crossing to the right, her gait natural and purposeful, a canvas bag swinging at her shoulder. Behind her, blurred traffic waits at the light and a few other pedestrians drift at the edges of the frame. She stays at a middle distance, small in the wide composition, never dominating the frame, her figure crisp against the muted street. Ambient city sound of idling engines, distant chatter, and the rhythmic tap of her footsteps on the crossing."
PROMPT_neon_dance_i2v="Preserve the figure, clothing, lighting, wet neon street, and camera angle from the starting image. The figure explodes into a super energetic dance in place, bouncing on the balls of his feet, snapping his shoulders, throwing both arms upward, then swinging them across his body as he pivots and steps side to side with fast rhythmic footwork. His coat and hoodie move with the motion while his face remains recognisable and expressive under the magenta and cyan neon light. The camera stays locked off and steady, letting the dance carry the motion, while reflections ripple across the wet street behind him. Ambient sound of heavy bass from a nearby club, wet footsteps, and city traffic."

prompt_text() {
  case "$1" in
    s1_t2v) printf '%s' "$PROMPT_s1_t2v" ;;
    s2_t2v) printf '%s' "$PROMPT_s2_t2v" ;;
    s3_t2v) printf '%s' "$PROMPT_s3_t2v" ;;
    neon_dance_i2v) printf '%s' "$PROMPT_neon_dance_i2v" ;;
  esac
}

scenario_title() {
  case "$1" in
    s1_t2v) echo "1 - Person walking toward camera (t2v)" ;;
    s2_t2v) echo "2 - Person dancing in front of a crowd (t2v)" ;;
    s3_t2v) echo "3 - Static camera crossing (t2v)" ;;
    neon_dance_i2v) echo "4 - flux_neon_seed7 energetic dance (i2v)" ;;
  esac
}

scenario_mode() {
  case "$1" in
    *_i2v) echo "i2v" ;;
    *) echo "t2v" ;;
  esac
}

slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/\.gguf$//;s/[^a-z0-9]+/_/g;s/^_|_$//g'
}

write_page() {
  python3 - "$MANIFEST" "$PAGE" <<'PY'
import csv, html, os, sys, pathlib
manifest = pathlib.Path(sys.argv[1])
page = pathlib.Path(sys.argv[2])
all_scenarios = [
    ("s1_t2v", "1 - Person walking toward camera (t2v)"),
    ("s2_t2v", "2 - Person dancing in front of a crowd (t2v)"),
    ("s3_t2v", "3 - Static camera crossing (t2v)"),
    ("neon_dance_i2v", "4 - flux_neon_seed7 energetic dance (i2v)"),
]
scenario_names = set(os.environ["SCENARIO_LIST"].split())
scenarios = [(key, title) for key, title in all_scenarios if key in scenario_names]
dits = os.environ["DIT_LIST"].split()
variants = os.environ["VARIANT_LIST"].split()
seeds = os.environ["SEED_LIST"].split()
prompts = {
    "s1_t2v": "Wide cinematic shot on a rain-slicked city street at night, shot on a 35mm lens with a shallow depth of field. Neon signage in magenta and cyan reflects across the wet asphalt while a faint mist softens the distant traffic lights. A man in his early thirties in a dark wool coat over a grey hoodie starts far down the block as a small silhouette and walks steadily toward the camera, his stride even and unhurried, hands loose at his sides. As he closes the distance his features resolve into focus - a short beard, tired eyes, breath fogging faintly in the cold air - until his face nearly fills the frame. The camera holds a slow, locked eyeline and lets him come to it rather than moving to meet him. Ambient city sound of tyres hissing on wet road, a low neon hum, and footsteps growing louder.",
    "s2_t2v": "Medium-wide shot at a crowded outdoor night concert, shot on a 50mm lens with warm stage light spilling from behind. A young woman with long dark hair in a fringed denim jacket dances at the centre of the frame, spinning on one heel, stepping side to side and throwing her arms up to the rhythm, her movements fluid and full-bodied. Behind her a dense crowd of silhouetted onlookers sways and claps, kept soft and slightly out of focus by the shallow depth of field so the eye stays on her. Amber and gold light rims her figure while cooler blues wash the background and dust motes drift through the beams. The camera holds an almost-static frame with only the faintest push-in, letting her motion carry the shot. Ambient sound of a live crowd, rhythmic music, and cheering.",
    "s3_t2v": "Locked-off wide shot of a pedestrian crossing on a busy daytime city street, shot on a 35mm lens at eye level from the far kerb; the camera does not move. Cool overcast daylight flattens the scene and wet pavement holds pale reflections of the surrounding buildings. A woman in a red raincoat and jeans walks briskly from the left of the frame across the zebra crossing to the right, her gait natural and purposeful, a canvas bag swinging at her shoulder. Behind her, blurred traffic waits at the light and a few other pedestrians drift at the edges of the frame. She stays at a middle distance, small in the wide composition, never dominating the frame, her figure crisp against the muted street. Ambient city sound of idling engines, distant chatter, and the rhythmic tap of her footsteps on the crossing.",
    "neon_dance_i2v": "Preserve the figure, clothing, lighting, wet neon street, and camera angle from the starting image. The figure explodes into a super energetic dance in place, bouncing on the balls of his feet, snapping his shoulders, throwing both arms upward, then swinging them across his body as he pivots and steps side to side with fast rhythmic footwork. His coat and hoodie move with the motion while his face remains recognisable and expressive under the magenta and cyan neon light. The camera stays locked off and steady, letting the dance carry the motion, while reflections ripple across the wet street behind him. Ambient sound of heavy bass from a nearby club, wet footsteps, and city traffic.",
}
rows = {}
if manifest.exists():
    with manifest.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[(r["scenario"], r["dit"], r["variant"], r["seed"])] = r
def cell(r):
    if not r:
        return '<div class="slot pending">pending</div>'
    status = r.get("status", "")
    if status != "OK":
        return f'<div class="slot fail"><b>{html.escape(status)}</b><br><span>{html.escape(r.get("note",""))}</span><br><span>{html.escape(r.get("wall_s",""))}s · peak {html.escape(r.get("peak_driver_mb",""))} MB</span></div>'
    src = html.escape(r["clip"])
    return (
        f'<div class="slot"><video src="{src}" controls loop playsinline></video>'
        f'<div><b>seed {html.escape(r["seed"])}</b> · {html.escape(r["wall_s"])}s · '
        f'driver {html.escape(r["peak_driver_mb"])} MB · cuDNN {html.escape(r["peak_cudnn_mb"])} MB</div></div>'
    )
def variant_row(scn, dit, variant, seeds):
    label = {
        "upscale_x2": "960x544 → x2 refine",
        "upscale_x15": "960x544 → x1.5 refine",
    }[variant]
    tds = [f'<th>{label}</th>']
    for seed in seeds:
        tds.append(f'<td>{cell(rows.get((scn, dit, variant, str(seed))))}</td>')
    return '<tr>' + ''.join(tds) + '</tr>'
out = ["""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LTX-2.3 weights sweep</title>
<style>
body{background:#0d0f12;color:#e6e8ea;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;max-width:1500px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:19px;margin:32px 0 10px;color:#7cc4ff;border-bottom:1px solid #23262b;padding-bottom:6px}
h3{font-size:15px;margin:22px 0 8px;color:#d6d8da}.sub{color:#9aa0a6;margin:0 0 14px}.prompt{background:#15181c;border:1px solid #23262b;border-radius:8px;padding:10px 12px;color:#b8bec6;font-size:13px}
table{border-collapse:collapse;width:100%;margin:8px 0 20px;table-layout:fixed}th,td{border:1px solid #23262b;padding:8px;vertical-align:top}th{background:#181c20;color:#9fb0bd;width:150px}
td{width:31%}.slot{font-size:12px;color:#aeb4bb}.slot video{width:100%;background:#000;border-radius:5px;display:block;margin-bottom:6px}.pending{color:#c9b06a}.fail{color:#e08a8a;background:#1f1214;border-radius:6px;padding:8px}
.note{background:#151a14;border:1px solid #2a3a22;border-radius:8px;padding:12px 16px;margin:14px 0;font-size:14px}.tag{display:inline-block;background:#233;color:#7cc4ff;border-radius:4px;padding:1px 7px;font-size:12px;margin-left:6px}
code{background:#1e2126;padding:1px 5px;border-radius:3px;font-size:13px}
</style></head><body>"""]
out.append(f'<h1>{html.escape(os.environ["SWEEP_LABEL"])} <span class="tag">97f @ 24fps · single segment · no continuation</span></h1>')
out.append('<p class="sub">Only <code>nvfp4-imatrix-dev050.gguf</code>. Compare the locked production 960x544 latent-upscale/refine flow at x2 and x1.5 across every seed.</p>')
out.append('<div class="note">Shared locked defaults from <code>run_singing_clip.sh</code>: SA3 CLI, max VRAM 9, 8 base steps, 3 refine steps, cfg 1.0, Euler ancestral, FFN tiles 8192, and LTX custom base/refine sigmas <code>1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0</code> / <code>0.909375,0.725,0.421875,0.0</code>.</div>')
for scn, title in scenarios:
    out.append(f'<h2>{html.escape(title)}</h2>')
    out.append(f'<div class="prompt">{html.escape(prompts[scn])}</div>')
    for dit in dits:
        out.append(f'<h3>{html.escape(dit)}</h3>')
        out.append('<table>')
        out.append('<tr><th>variant</th>' + ''.join(f'<th>seed {html.escape(seed)}</th>' for seed in seeds) + '</tr>')
        for variant in variants:
            out.append(variant_row(scn, dit, variant, seeds))
        out.append('</table>')
out.append('</body></html>')
page.write_text('\n'.join(out))
PY
}

record_result() {
  local scenario="$1" dit="$2" variant="$3" seed="$4" status="$5" wall="$6" peak_driver="$7" peak_cudnn="$8" clip="$9" note="${10:-}"
  local tmp="$MANIFEST.tmp"
  if ! head -1 "$MANIFEST" 2>/dev/null | grep -q '^scenario	dit	variant	seed	'; then
    printf "scenario\tdit\tvariant\tseed\tstatus\twall_s\tpeak_driver_mb\tpeak_cudnn_mb\tclip\tnote\n" > "$tmp"
  else
    head -1 "$MANIFEST" > "$tmp"
  fi
  awk -F'\t' -v OFS='\t' -v s="$scenario" -v d="$dit" -v v="$variant" -v seed="$seed" 'NR>1 && !($1==s && $2==d && $3==v && $4==seed)' "$MANIFEST" >> "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$scenario" "$dit" "$variant" "$seed" "$status" "$wall" "$peak_driver" "$peak_cudnn" "$clip" "$note" >> "$tmp"
  mv "$tmp" "$MANIFEST"
  write_page
}

has_done() {
  local scenario="$1" dit="$2" variant="$3" seed="$4"
  awk -F'\t' -v s="$scenario" -v d="$dit" -v v="$variant" -v seed="$seed" 'NR>1 && $1==s && $2==d && $3==v && $4==seed && ($5=="OK" || $5 ~ /^FAIL/){found=1} END{exit !found}' "$MANIFEST"
}

render_one() {
  local scenario="$1" dit="$2" variant="$3" seed="$4"
  has_done "$scenario" "$dit" "$variant" "$seed" && { echo "[skip] $scenario $dit $variant seed=$seed"; return 0; }
  [ -f "$LTX2/$dit" ] || { record_result "$scenario" "$dit" "$variant" "$seed" "FAIL" 0 0 0 "" "missing model"; return 0; }

  local mode w h hires_args init_arg="" variant_label outname tag od log vramf prompt
  mode=$(scenario_mode "$scenario")
  prompt=$(prompt_text "$scenario")
  tag="$(slug "${scenario}_${dit}_${variant}_seed${seed}")"
  od="$OUTROOT/$tag"; log="$od/log"; vramf="$od/vram.log"
  rm -rf "$od"; mkdir -p "$od"
  outname="${tag}.webm"

  case "$variant" in
    upscale_x2) w=960; h=544; hires_args="--hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-upscalers-dir $UPDIR --hires-steps $REFSTEPS --hires-denoising-strength 0.7 --hires-sigmas $REFINE_SIGMAS" ;;
    upscale_x15) w=960; h=544; hires_args="--hires --hires-upscaler ltx-2.3-spatial-upscaler-x1.5-1.0 --hires-upscalers-dir $UPDIR --hires-steps $REFSTEPS --hires-denoising-strength 0.7 --hires-sigmas $REFINE_SIGMAS" ;;
    *) echo "bad variant $variant"; return 2 ;;
  esac

  if [ "$mode" = i2v ]; then
    local init_name="flux_neon_seed7.png"
    local init="$LTX2/_inputs/$init_name"
    [ -f "$init" ] || { record_result "$scenario" "$dit" "$variant" "$seed" "FAIL" 0 0 0 "" "missing $init_name"; return 0; }
    init_arg="--init-img /ltx2/_inputs/$init_name"
  fi

  echo "=== $scenario | $dit | $variant | seed=$seed | ${w}x${h} | $(date '+%F %T') ==="
  : > "$vramf"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null >> "$vramf"; sleep 2; done ) &
  local spid=$!
  local t0 t1 rc wall
  local sa3_args=() sa3_policy_args=()
  case "$SA3_MODE" in
    off) sa3_args+=( -e GGML_LTX_SA3=0 ) ;;
    native) sa3_args+=( -e GGML_LTX_SA3=1 ) ;;
    default) ;;
    *) echo "bad SA3_MODE $SA3_MODE (expected native, off, or default)"; return 2 ;;
  esac
  if [ -n "${GGML_LTX_SA3_POLICY:-}" ]; then
    sa3_policy_args+=( -e "GGML_LTX_SA3_POLICY=$GGML_LTX_SA3_POLICY" )
  fi
  t0=$(date +%s)
  timeout "$TIMEOUT" docker run --rm --gpus "\"device=$GPU\"" \
    -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1 -e LTX_DIT_F16=1 \
    -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks \
    -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1 \
    -e GGML_LTX_SA3_DELTA_F16="$SA3_DELTA_F16" -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=0 -e LONGCAT_FFN_TILE_TOKENS=8192 -e LONGCAT_ENCODE_MAX_VRAM=6.5 \
    -e LONGCAT_NO_PREFETCH_POOL=1 -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_OFFLOAD_PIPELINING=0 -e LONGCAT_DIT_NO_MMAP=0 \
    -e LTXAV_VAE_LAZY=1 -e LTXAV_DIT_FREE_DURING_DECODE=1 -e LONGCAT_VRAM_BREAKDOWN=1 \
    -e LTX_VAE_HEAD_F32=1 -e LTX_VAE_CONV3D_WTILES=16 -e LTX_VAE_CONV3D_HTILES=8 -e LTX_VAE_DECODE_F16=1 \
    -e LTX_VAE_SPATIAL_TILES=2x2 -e LTX_VAE_SPATIAL_OVERLAP=4 -e LTX_CUSTOM_SIGMAS="$BASE_SIGMAS" \
    -e LTXAV_CHAIN_OVERLAP_DROP=24 -e LTXAV_SKIP_AUDIO_DECODE=1 -e LONGCAT_ATTN_TILES=2 -e LONGCAT_PERSIST_GRAPH_INPUTS=1 -e PROMPT="$prompt" \
    "${sa3_policy_args[@]}" \
    "${sa3_args[@]}" \
    -v "$WT:/src" -v "$LTX2:/ltx2" -v "$od:/work" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src \
    "$BUILDER" bash -lc "stdbuf -oL -eL $BIN -M vid_gen --diffusion-model /ltx2/$dit \
      --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors \
      --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
      --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf \
      --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
      --lora-model-dir /ltx2/loras \
      -p \"\$PROMPT\" $init_arg \
      -W $w -H $h --video-frames $FR --fps $FPS --steps $STEPS --cfg-scale $CFG --sampling-method euler_a --diffusion-fa \
      $hires_args --offload-to-cpu --mmap --max-vram $MAXV -s $seed -v -o /work/out.webm" > "$log" 2>&1
  rc=$?
  t1=$(date +%s); wall=$((t1-t0))
  kill "$spid" 2>/dev/null

  local peak_driver peak_cudnn clip rel note status
  peak_driver=$(perl -ne 'while(/driver_used=(\d+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$log")
  if [ "$peak_driver" = 0 ]; then peak_driver=$(sort -n "$vramf" 2>/dev/null | tail -1); fi
  peak_cudnn=$(perl -ne 'while(/used ([0-9.]+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$log")
  note=""
  if [ "$rc" = 124 ]; then
    status="FAIL"; note="timeout ${TIMEOUT}s"
  elif [ -s "$od/out.webm" ]; then
    cp -f "$od/out.webm" "$CLIPDIR/$outname"
    rel="$CLIP_SUBDIR/$outname"
    status="OK"; clip="$rel"
  else
    status="FAIL"
    note=$(grep -aoiE "out of memory|cudaMalloc failed|cuda error|latent spatial upscale failed|assert|error" "$log" | tail -1)
    [ -n "$note" ] || note="no output rc=$rc"
  fi
  record_result "$scenario" "$dit" "$variant" "$seed" "$status" "$wall" "${peak_driver:-0}" "${peak_cudnn:-0}" "${clip:-}" "$note"
  echo ">>> $status $scenario $dit $variant seed=$seed wall=${wall}s driver=${peak_driver:-0}MB cudnn=${peak_cudnn:-0}MB ${note:+note=$note}"
}

write_page

for scenario in "${SCENARIOS[@]}"; do
  echo "######## $(scenario_title "$scenario") ########"
  for dit in "${DITS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      for seed in "${SEEDS[@]}"; do render_one "$scenario" "$dit" "$variant" "$seed"; done
    done
  done
done

write_page
echo "[done] $PAGE"
