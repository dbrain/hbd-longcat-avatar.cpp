#!/usr/bin/env bash
# sweep_render.sh TAG DIT LORA(0|1) [MAXV] [RES] [EXTRA_XE]
# Renders s1/s2/s3 at 1920x1088 (RES=parity), captures wall + peak VRAM, surfaces clips + manifest.
set -u
cd /home/dbrain/dev/longcat-avatar-ltxdenoise/ltx-denoise-repro
TAG="$1"; DIT="$2"; LORA="${3:-0}"; MAXV="${4:-9}"; RES="${5:-parity}"; EXTRA_XE="${6:-}"
CLIPS=/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/clips
MAN=/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/sweep_manifest.tsv
LORATAG='<lora:ltx-2.3-22b-distilled-lora-384-1.1:0.5>'
python3 -c "import json;p=json.load(open('/tmp/claude-1000/-home-dbrain-dev-kobbler/9fc768a3-d859-4b32-82f6-ec146566f6ed/scratchpad/prompts.json'));[open(f'/tmp/prompt_s{s}.txt','w').write(t) for s,t in p.items()]"
for s in 1 2 3; do
  PT="$(cat /tmp/prompt_s$s.txt)"
  [ "$LORA" = 1 ] && PT="${LORATAG}${PT}"
  echo "######### $TAG s$s  (DIT=$DIT MAXV=$MAXV RES=$RES) $(date +%H:%M:%S) #########"
  # peak-VRAM sampler
  SMPF=/tmp/vram_${TAG}_s$s; : > "$SMPF"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null >> "$SMPF"; sleep 2; done ) & SPID=$!
  W=/home/dbrain/dev/longcat-avatar-ltxdenoise/_ablation_out/sweep_${TAG}_s$s/out.webm
  t0=$(date +%s)
  for att in 1 2 3 4; do
    # wait for prod (kobbler-ltx-video) to release VRAM — need ~14.5GB free
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null || echo 0)" -gt 14500 ] 2>/dev/null; do echo "  (waiting for GPU: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i 1) free)"; sleep 25; done
    DIT="$DIT" RES="$RES" MAXV="$MAXV" TBF=3 VWT=16 VHT=8 SAMP=euler TAG="sweep_${TAG}_s$s" \
      XE="$EXTRA_XE" PROMPT="$PT" bash run_parity_nvfp4.sh > /tmp/sweep_${TAG}_s$s.log 2>&1
    [ -s "$W" ] && break
    if grep -aqiE "out of memory|cudaMalloc failed" /home/dbrain/dev/longcat-avatar-ltxdenoise/_ablation_out/sweep_${TAG}_s$s/log 2>/dev/null; then
      echo "  (OOM on attempt $att — prod contention, waiting 40s + retry)"; sleep 40; continue
    fi
    break   # non-OOM failure: don't retry
  done
  t1=$(date +%s); wall=$((t1-t0)); kill $SPID 2>/dev/null
  peak=$(sort -n "$SMPF" 2>/dev/null | tail -1)
  if [ -s "$W" ]; then
    ffmpeg -y -i "$W" -c:v libx264 -crf 17 -pix_fmt yuv420p -c:a aac "$CLIPS/SWEEP_${TAG}_s$s.mp4" -loglevel error 2>/dev/null
    res=$(ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$W" 2>/dev/null|head -1)
    [ -f "$MAN" ] && grep -v "^${TAG}	s$s	" "$MAN" > "$MAN.t" 2>/dev/null && mv "$MAN.t" "$MAN"
    printf "%s\ts%s\t%ss\t%sMiB\t%s\tOK\n" "$TAG" "$s" "$wall" "$peak" "$res" >> "$MAN"
    echo ">>> $TAG s$s OK  ${wall}s  peak=${peak}MiB  $res -> SWEEP_${TAG}_s$s.mp4"
  else
    err=$(grep -aoiE "out of memory|cuda error|latent spatial upscale failed|assert" /home/dbrain/dev/longcat-avatar-ltxdenoise/_ablation_out/sweep_${TAG}_s$s/log 2>/dev/null|tail -1)
    printf "%s\ts%s\t%ss\t%sMiB\t-\tFAIL:%s\n" "$TAG" "$s" "$wall" "$peak" "$err" >> "$MAN"
    echo ">>> $TAG s$s FAIL (${wall}s) — $err"
  fi
done
echo "@@@ SWEEP $TAG DONE @@@"
