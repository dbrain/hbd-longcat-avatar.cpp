#!/usr/bin/env bash
# B3 seam eye-test helper: for a contknee chain, build a side-by-side strip of the JOIN frames
# (last K frames of seg(n-1) tail vs first frames of seg n) so subject/motion carry + grain/brightness
# jump at the boundary can be eyeballed. Also makes a hard-cut comparison (seg(n-1) end |cut| seg n start
# WITHOUT continuation overlap-drop) implicitly via the raw frames. Outputs PNG strips to OUT/seams/.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FR="${FR:-13}"; K="${K:-5}"
OUT="$PWD/perf_out/contknee/fr$FR"; S="$OUT/seams"; rm -rf "$S"; mkdir -p "$S"
command -v montage >/dev/null 2>&1 || { echo "need imagemagick montage"; }
for seg in 1 2; do
  prev=$((seg-1))
  [ -d "$OUT/seg$seg" ] || continue
  # The continuation seg's first K frames are the carried/overlap region; frame K is the first NEW frame.
  # Show seg(prev) last 3 frames, then seg(seg) frames 0..K+1 to inspect the join.
  pl=$(ls "$OUT/seg$prev"/f*.png 2>/dev/null | sort | tail -3)
  cl=$(ls "$OUT/seg$seg"/f*.png 2>/dev/null | sort | head -n $((K+3)))
  imgs="$pl $cl"
  if command -v montage >/dev/null 2>&1; then
    montage $imgs -tile $(( $(echo $imgs|wc -w) ))x1 -geometry 256x+2+2 -background black \
      -title "seam seg$prev->seg$seg (last3 prev | first $((K+3)) cont; frame $K = first NEW)" \
      "$S/seam_${prev}_${seg}.png" 2>/dev/null && echo "wrote $S/seam_${prev}_${seg}.png"
  fi
  # brightness/grain jump metric across the join: mean luma per frame around the boundary
  echo "  luma around seam seg$prev->seg$seg (prev tail then cont head):"
  for f in $pl $cl; do
    m=$(ffmpeg -hide_banner -loglevel error -i "$f" -vf "format=gray,signalstats" -f null - 2>&1 | grep -oE 'YAVG:[0-9.]+' | head -1)
    printf "    %s %s\n" "$(basename "$f")" "$m"
  done
done
ls -la "$S" 2>/dev/null
