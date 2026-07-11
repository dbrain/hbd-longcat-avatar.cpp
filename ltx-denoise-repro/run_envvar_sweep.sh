#!/usr/bin/env bash
# Focused one-knob env sweep for the static-crossing LTX-2.3 quality path.
set -uo pipefail

WT=${WT:-/home/dbrain/dev/longcat-avatar-ltxdenoise}
LTX2=${LTX2:-/home/dbrain/dev/longcat-avatar.cpp/models/ltx2}
EYE=${EYE:-/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise}
BUILDER=${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}
BIN=/src/build-cudnn/bin/sd-cli
GPU=${GPU:-1}
TIMEOUT=${TIMEOUT:-700}

DIT=${DIT:-nvfp4-imatrix-dev050.gguf}
SEED=${SEED:-42}
MAXV=${MAXV:-7}
LONGCAT_NO_PREFETCH_POOL_ENV=${LONGCAT_NO_PREFETCH_POOL_ENV-1}
LONGCAT_FFN_TILE_TOKENS_ENV=${LONGCAT_FFN_TILE_TOKENS_ENV-4096}
LONGCAT_SHARED_RESIDENT_VALUE=${LONGCAT_SHARED_RESIDENT_VALUE-1}
LTXAV_VAE_LAZY_VALUE=${LTXAV_VAE_LAZY_VALUE-1}
LTXAV_DIT_FREE_VALUE=${LTXAV_DIT_FREE_VALUE-1}
FR=97
FPS=24
STEPS=8
REFSTEPS=3
BASE_SIGMAS="1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0"
REFINE_SIGMAS="0.909375,0.725,0.421875,0.0"
UPSCALER="ltx-2.3-spatial-upscaler-x2-1.1"
UPDIR=/ltx2/latent_upscale_models
PROMPT=${PROMPT_TEXT:-"Locked-off wide shot of a pedestrian crossing on a busy daytime city street, shot on a 35mm lens at eye level from the far kerb; the camera does not move. Cool overcast daylight flattens the scene and wet pavement holds pale reflections of the surrounding buildings. A woman in a red raincoat and jeans walks briskly from the left of the frame across the zebra crossing to the right, her gait natural and purposeful, a canvas bag swinging at her shoulder. Behind her, blurred traffic waits at the light and a few other pedestrians drift at the edges of the frame. She stays at a middle distance, small in the wide composition, never dominating the frame, her figure crisp against the muted street. Ambient city sound of idling engines, distant chatter, and the rhythmic tap of her footsteps on the crossing."}
PROMPT_SUMMARY=${PROMPT_SUMMARY:-"Locked-off wide shot of a pedestrian crossing on a busy daytime city street... woman in a red raincoat crosses left to right; camera locked, overcast wet pavement."}

OUTROOT=${OUTROOT:-$WT/ltx-denoise-repro/_envvar_sweep}
CLIPDIR=$EYE/envvar_sweep
MANIFEST=$EYE/envvar_sweep.tsv
PAGE=$EYE/envvar_sweep.html
mkdir -p "$OUTROOT" "$CLIPDIR" "$EYE"

slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/\.gguf$//;s/[^a-z0-9]+/_/g;s/^_|_$//g'
}

write_page() {
  python3 - "$MANIFEST" "$PAGE" <<'PY'
import csv, html, pathlib, sys
manifest = pathlib.Path(sys.argv[1])
page = pathlib.Path(sys.argv[2])
rows = []
if manifest.exists():
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
def num(v):
    try: return float(v)
    except Exception: return 0.0
rows.sort(key=lambda r: (r.get("group",""), num(r.get("sort","")), r.get("label","")))
out = ["""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LTX env-var performance sweep</title>
<style>
body{background:#0d0f12;color:#e6e8ea;font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;max-width:1500px}
h1{font-size:24px;margin:0 0 6px}.sub{color:#a8b0b8;margin:0 0 16px}.prompt{background:#15181c;border:1px solid #272b31;border-radius:8px;padding:10px 12px;color:#bac1c9;font-size:13px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin-top:16px}th,td{border:1px solid #272b31;padding:7px;vertical-align:top}th{background:#181c20;color:#aeb8c2;text-align:left}td video{width:260px;max-width:100%;background:#000;border-radius:5px;display:block}
.ok{color:#8ee39a}.bad{color:#f09898}.muted{color:#9aa2aa}code{background:#20242a;padding:1px 4px;border-radius:3px}
</style></head><body>"""]
out.append("<h1>LTX env-var performance sweep</h1>")
out.append('<p class="sub">Single 97f segment, static crossing prompt, <code>nvfp4-imatrix-dev050.gguf</code>, quality path: 960x544 base -> x2 latent upscale/refine -> 1920x1088 output. Each row changes one lever from the baseline unless noted.</p>')
out.append('<div class="prompt">Static crossing baseline prompt by default. Some rows use prompt variants; see each row label/env.</div>')
out.append("<table><tr><th>group</th><th>label</th><th>status</th><th>wall</th><th>sampling</th><th>refine</th><th>decode</th><th>driver peak</th><th>cuDNN peak</th><th>clip</th><th>note</th></tr>")
for r in rows:
    cls = "ok" if r.get("status") == "OK" else "bad"
    clip = r.get("clip","")
    media = f'<video src="{html.escape(clip)}" controls loop playsinline></video>' if clip else ""
    out.append("<tr>"
        f"<td>{html.escape(r.get('group',''))}</td>"
        f"<td><b>{html.escape(r.get('label',''))}</b><br><span class=muted>{html.escape(r.get('env',''))}</span></td>"
        f"<td class={cls}>{html.escape(r.get('status',''))}</td>"
        f"<td>{html.escape(r.get('wall_s',''))}s</td>"
        f"<td>{html.escape(r.get('base_sample_s',''))}s</td>"
        f"<td>{html.escape(r.get('refine_sample_s',''))}s</td>"
        f"<td>{html.escape(r.get('decode_s',''))}s</td>"
        f"<td>{html.escape(r.get('peak_driver_mb',''))} MB</td>"
        f"<td>{html.escape(r.get('peak_cudnn_mb',''))} MB</td>"
        f"<td>{media}</td>"
        f"<td>{html.escape(r.get('note',''))}</td></tr>")
out.append("</table></body></html>")
page.write_text("\n".join(out))
PY
}

record_result() {
  local group="$1" sort="$2" label="$3" envdesc="$4" status="$5" wall="$6" base_s="$7" refine_s="$8" decode_s="$9" peak_driver="${10}" peak_cudnn="${11}" clip="${12}" note="${13:-}"
  local tmp="$MANIFEST.tmp"
  printf "group\tsort\tlabel\tenv\tstatus\twall_s\tbase_sample_s\trefine_sample_s\tdecode_s\tpeak_driver_mb\tpeak_cudnn_mb\tclip\tnote\n" > "$tmp"
  if [ -s "$MANIFEST" ]; then
    awk -F'\t' -v OFS='\t' -v g="$group" -v l="$label" 'NR>1 && !($1==g && $3==l)' "$MANIFEST" >> "$tmp"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$group" "$sort" "$label" "$envdesc" "$status" "$wall" "$base_s" "$refine_s" "$decode_s" "$peak_driver" "$peak_cudnn" "$clip" "$note" >> "$tmp"
  mv "$tmp" "$MANIFEST"
  write_page
}

done_row() {
  local group="$1" label="$2"
  awk -F'\t' -v g="$group" -v l="$label" 'NR>1 && $1==g && $3==l && ($5=="OK" || $5 ~ /^FAIL/){found=1} END{exit !found}' "$MANIFEST" 2>/dev/null
}

extract_last_float() {
  local pattern="$1" file="$2"
  perl -0777 -ne "while (/$pattern/g) { \$m=\$1 } END { print defined(\$m) ? \$m : q{} }" "$file"
}

render_one() {
  local group="$1" sort="$2" label="$3" envdesc="$4"; shift 4
  done_row "$group" "$label" && { echo "[skip] $group / $label"; return 0; }
  local tag od log vramf outname
  tag="$(slug "${group}_${label}")"
  od="$OUTROOT/$tag"; log="$od/log"; vramf="$od/vram.log"; outname="${tag}.webm"
  rm -rf "$od"; mkdir -p "$od"
  echo "=== $group | $label | $envdesc | $(date '+%F %T') ==="
  : > "$vramf"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null >> "$vramf"; sleep 1; done ) &
  local spid=$!
  local t0 t1 rc wall
  t0=$(date +%s)
  timeout "$TIMEOUT" docker run --rm --gpus "\"device=$GPU\"" \
    -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1 -e LTX_DIT_F16=1 \
    -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks \
    -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1 \
    ${LONGCAT_SHARED_RESIDENT_VALUE:+-e LONGCAT_SHARED_RESIDENT="$LONGCAT_SHARED_RESIDENT_VALUE"} -e LONGCAT_VAE_KEEP_RESIDENT=0 -e LONGCAT_ENCODE_MAX_VRAM=6.5 \
    -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_OFFLOAD_PIPELINING=0 -e LONGCAT_DIT_NO_MMAP=0 \
    ${LTXAV_VAE_LAZY_VALUE:+-e LTXAV_VAE_LAZY="$LTXAV_VAE_LAZY_VALUE"} ${LTXAV_DIT_FREE_VALUE:+-e LTXAV_DIT_FREE_DURING_DECODE="$LTXAV_DIT_FREE_VALUE"} -e LONGCAT_VRAM_BREAKDOWN=1 \
    -e LTX_VAE_HEAD_F32=1 -e LTX_VAE_CONV3D_WTILES=16 -e LTX_VAE_CONV3D_HTILES=8 -e LTX_VAE_DECODE_F16=1 \
    -e LTX_VAE_SPATIAL_TILES=2x2 -e LTX_VAE_SPATIAL_OVERLAP=4 -e LTX_CUSTOM_SIGMAS="$BASE_SIGMAS" \
    ${LONGCAT_NO_PREFETCH_POOL_ENV:+-e LONGCAT_NO_PREFETCH_POOL="$LONGCAT_NO_PREFETCH_POOL_ENV"} \
    ${LONGCAT_FFN_TILE_TOKENS_ENV:+-e LONGCAT_FFN_TILE_TOKENS="$LONGCAT_FFN_TILE_TOKENS_ENV"} \
    "$@" \
    -e PROMPT="$PROMPT" \
    -v "$WT:/src" -v "$LTX2:/ltx2" -v "$od:/work" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src \
    "$BUILDER" bash -lc "stdbuf -oL -eL $BIN -M vid_gen --diffusion-model /ltx2/$DIT \
      --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors \
      --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
      --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf \
      --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
      --lora-model-dir /ltx2/loras -p \"\$PROMPT\" \
      -W 960 -H 544 --video-frames $FR --fps $FPS --steps $STEPS --cfg-scale 1.0 --sampling-method euler --diffusion-fa \
      --hires --hires-upscaler $UPSCALER --hires-upscalers-dir $UPDIR --hires-steps $REFSTEPS --hires-sigmas $REFINE_SIGMAS \
      --offload-to-cpu --mmap --max-vram $MAXV -s $SEED -v -o /work/out.webm" > "$log" 2>&1
  rc=$?
  t1=$(date +%s); wall=$((t1-t0))
  kill "$spid" 2>/dev/null

  local peak_driver peak_cudnn base_s refine_s decode_s clip status note
  peak_driver=$(perl -ne 'while(/driver_used=(\d+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$log")
  [ "$peak_driver" = 0 ] && peak_driver=$(sort -n "$vramf" 2>/dev/null | tail -1)
  peak_cudnn=$(perl -ne 'while(/used ([0-9.]+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$log")
  base_s=$(extract_last_float 'sampling completed, taking ([0-9.]+)s' "$log")
  refine_s=$(extract_last_float 'sampling\(latent upscale\) completed, taking ([0-9.]+)s' "$log")
  decode_s=$(extract_last_float 'decode_first_stage completed, taking ([0-9.]+)s' "$log")
  clip=""
  note=""
  if [ "$rc" = 124 ]; then
    status=FAIL; note="timeout ${TIMEOUT}s"
  elif [ -s "$od/out.webm" ]; then
    cp -f "$od/out.webm" "$CLIPDIR/$outname"
    clip="envvar_sweep/$outname"
    status=OK
  else
    status=FAIL
    note=$(grep -aoiE "out of memory|cudaMalloc failed|cuda error|assert|error" "$log" | tail -1)
    [ -n "$note" ] || note="no output rc=$rc"
  fi
  record_result "$group" "$sort" "$label" "$envdesc" "$status" "$wall" "${base_s:-}" "${refine_s:-}" "${decode_s:-}" "${peak_driver:-0}" "${peak_cudnn:-0}" "$clip" "$note"
  echo ">>> $status $group/$label wall=${wall}s base=${base_s:-?}s refine=${refine_s:-?}s decode=${decode_s:-?}s driver=${peak_driver:-0}MB cudnn=${peak_cudnn:-0}MB ${note:+note=$note}"
}

write_page

case "${SWEEP:-maxv}" in
  maxv)
    for v in ${MAXV_VALUES:-7 8 9 10 11}; do
      MAXV=$v render_one "max-vram" "$v" "maxv-$v" "--max-vram $v"
    done
    ;;
  prefetch)
    LONGCAT_NO_PREFETCH_POOL_ENV=1 render_one "prefetch" 1 "no-prefetch-pool-on" "LONGCAT_NO_PREFETCH_POOL=1"
    LONGCAT_NO_PREFETCH_POOL_ENV= render_one "prefetch" 2 "prefetch-pool-unset" "LONGCAT_NO_PREFETCH_POOL unset"
    ;;
  ffn)
    for v in ${FFN_VALUES:-0 2048 4096 8192}; do
      LONGCAT_FFN_TILE_TOKENS_ENV=$v render_one "ffn-tile" "$v" "ffn-$v" "LONGCAT_FFN_TILE_TOKENS=$v"
    done
    ;;
  memory)
    LTXAV_VAE_LAZY_VALUE=1 LTXAV_DIT_FREE_VALUE=1 render_one "memory" 1 "lazy+ditfree-on" "LTXAV_VAE_LAZY=1 LTXAV_DIT_FREE_DURING_DECODE=1"
    LTXAV_VAE_LAZY_VALUE= LTXAV_DIT_FREE_VALUE=1 render_one "memory" 2 "vae-lazy-unset" "LTXAV_VAE_LAZY unset"
    LTXAV_VAE_LAZY_VALUE=1 LTXAV_DIT_FREE_VALUE= render_one "memory" 3 "ditfree-unset" "LTXAV_DIT_FREE_DURING_DECODE unset"
    ;;
  chainlow)
    render_one "chain-low" 1 "refine-context-0-single" "LTX_REFINE_CONTEXT_FRAMES=0" -e LTX_REFINE_CONTEXT_FRAMES=0
    render_one "chain-low" 2 "refine-const-seed-single" "LTX_REFINE_CONST_SEED=1" -e LTX_REFINE_CONST_SEED=1
    ;;
  fixvram)
    MAXV=9 render_one "fix-vram" 10 "qtile-8160" "LTX_ATTN_QTILE=8160" -e LTX_ATTN_QTILE=8160
    MAXV=9 render_one "fix-vram" 20 "conv3d-bucket-d16" "GGML_CUDNN_CONV3D_BUCKET_D=16" -e GGML_CUDNN_CONV3D_BUCKET_D=16
    MAXV=9 render_one "fix-vram" 30 "end-render-reclaim" "LTXAV_END_RENDER_RECLAIM=1" -e LTXAV_END_RENDER_RECLAIM=1
    MAXV=9 render_one "fix-vram" 40 "chain-pool-trim" "LTXAV_CHAIN_POOL_TRIM=1" -e LTXAV_CHAIN_POOL_TRIM=1
    MAXV=9 render_one "fix-vram" 50 "reclaim-plus-trim" "LTXAV_END_RENDER_RECLAIM=1 LTXAV_CHAIN_POOL_TRIM=1" -e LTXAV_END_RENDER_RECLAIM=1 -e LTXAV_CHAIN_POOL_TRIM=1
    MAXV=9 render_one "fix-vram" 60 "attn-bucket-default" "GGML_CUDNN_ATTN_BUCKET=1" -e GGML_CUDNN_ATTN_BUCKET=1
    MAXV=9 render_one "fix-vram" 70 "attn-buckets-custom" "GGML_CUDNN_ATTN_BUCKETS=160,1024,8160,16320,32640,38760" -e GGML_CUDNN_ATTN_BUCKETS=160,1024,8160,16320,32640,38760
    ;;
  visualflags)
    WALK_PROMPT="Wide cinematic shot on a rain-slicked city street at night, shot on a 35mm lens with a shallow depth of field. Neon signage in magenta and cyan reflects across the wet asphalt while a faint mist softens the distant traffic lights. A man in his early thirties in a dark wool coat over a grey hoodie starts far down the block as a small silhouette and walks steadily toward the camera, his stride even and unhurried, hands loose at his sides. As he closes the distance his features resolve into focus - a short beard, tired eyes, breath fogging faintly in the cold air - until his face nearly fills the frame. The camera holds a slow, locked eyeline and lets him come to it rather than moving to meet him. Ambient city sound of tyres hissing on wet road, a low neon hum, and footsteps growing louder."
    for s in 7 42; do
      PROMPT="$WALK_PROMPT" SEED=$s MAXV=9 render_one "visual-walk" "$((100 + s))" "baseline-seed$s" "walking prompt; baseline MAXV=9"
    done
    for s in 7 42; do
      PROMPT="$WALK_PROMPT" SEED=$s MAXV=9 render_one "visual-walk" "$((200 + s))" "qtile-seed$s" "walking prompt; LTX_ATTN_QTILE=8160" -e LTX_ATTN_QTILE=8160
    done
    for s in 7 42; do
      PROMPT="$WALK_PROMPT" SEED=$s MAXV=9 render_one "visual-walk" "$((300 + s))" "convbucket-seed$s" "walking prompt; GGML_CUDNN_CONV3D_BUCKET_D=16" -e GGML_CUDNN_CONV3D_BUCKET_D=16
    done
    PROMPT="$WALK_PROMPT" SEED=7 MAXV=9 render_one "visual-walk" 407 "both-seed7" "walking prompt; LTX_ATTN_QTILE=8160 GGML_CUDNN_CONV3D_BUCKET_D=16" -e LTX_ATTN_QTILE=8160 -e GGML_CUDNN_CONV3D_BUCKET_D=16
    PROMPT="$WALK_PROMPT" SEED=42 MAXV=9 render_one "visual-walk" 4421 "both-seed42-repeat-a" "walking prompt; duplicate determinism A; both flags" -e LTX_ATTN_QTILE=8160 -e GGML_CUDNN_CONV3D_BUCKET_D=16
    PROMPT="$WALK_PROMPT" SEED=42 MAXV=9 render_one "visual-walk" 4422 "both-seed42-repeat-b" "walking prompt; duplicate determinism B; both flags" -e LTX_ATTN_QTILE=8160 -e GGML_CUDNN_CONV3D_BUCKET_D=16
    ;;
  all)
    SWEEP=maxv "$0"
    SWEEP=ffn "$0"
    SWEEP=prefetch "$0"
    SWEEP=chainlow "$0"
    ;;
  *)
    echo "unknown SWEEP=${SWEEP}" >&2
    exit 2
    ;;
esac

write_page
echo "[done] $PAGE"
