#!/usr/bin/env bash
# fid_run.sh LABEL "one-line description" [ENV=val ...]
# Renders ONE fidelity experiment (1280x704, seed 42) with the given env overrides,
# saves to a uniquely-named clip, and appends to the manifest. Never overwrites another experiment.
set -u
cd /home/dbrain/dev/longcat-avatar-ltxdenoise/ltx-denoise-repro
LABEL="$1"; DESC="$2"; shift 2
CLIPS=/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/clips
MAN=/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/fid_manifest.tsv
OUT=/home/dbrain/dev/longcat-avatar-ltxdenoise/_ablation_out/parity_nvfp4_speed/out.webm
echo "########## FID $LABEL — $DESC  ($*) ##########"
t0=$(date +%s)
# defaults = the 135s baseline; caller overrides one variable
env RES=speed MAXV=9 TBF=4 VWT=4 VHT=2 SAMP=euler DF16=1 "$@" bash run_parity_nvfp4.sh > /tmp/fid_$LABEL.log 2>&1
rc=$?; t1=$(date +%s); wall=$((t1-t0))
if [ -s "$OUT" ]; then
  ffmpeg -y -i "$OUT" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac "$CLIPS/FID_$LABEL.mp4" -loglevel error 2>/dev/null
  res=$(ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$OUT" 2>/dev/null | head -1)
  # strip any prior row for this label, then append
  [ -f "$MAN" ] && grep -v "^$LABEL	" "$MAN" > "$MAN.tmp" 2>/dev/null && mv "$MAN.tmp" "$MAN"
  printf "%s\t%s\t%s\t%ss\t%s\n" "$LABEL" "$DESC" "$*" "$wall" "$res" >> "$MAN"
  echo ">>> FID_$LABEL.mp4  ${wall}s  $res"
else
  echo ">>> $LABEL FAILED (${wall}s) — $(grep -aoiE 'out of memory|cuda error' /tmp/fid_$LABEL.log|head -1)"
fi
