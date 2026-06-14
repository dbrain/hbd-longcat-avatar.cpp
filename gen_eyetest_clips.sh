#!/usr/bin/env bash
# Build a self-contained eye-test page for ALL clips produced this session (27s montage, shift sweep,
# smoke, sigma-fix), with gen-time + peak-VRAM per clip for later comparison. Re-runnable as renders land.
# Host-side only (ffmpeg + python http.server). Serve: PORT=8098 then http://10.0.0.208:8098/
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD/perf_out"; EY="$ROOT/eyetest_clips"; FPS=16; mkdir -p "$EY/media"
HOSTIP=$(hostname -I 2>/dev/null | awk '{print $1}'); PORT="${PORT:-8098}"

# metadata helpers (read a render dir's run.log + peak.txt)
meta(){ # $1=dir -> echoes "gen|peak|res|fr|steps|sigmas|rc"
  local d="$1" log="$1/run.log"
  local gen=$(grep -oE 'generate_video completed in [0-9.]+s' "$log" 2>/dev/null|grep -oE '[0-9.]+s'|tail -1)
  local peak=$(cat "$d/peak.txt" 2>/dev/null); local pg="-"; [ -n "$peak" ] && pg=$(awk "BEGIN{printf \"%.1f\",$peak/1024}")
  local res=$(grep -oE 'generate_video [0-9]+x[0-9]+x[0-9]+' "$log" 2>/dev/null|tail -1|grep -oE '[0-9]+x[0-9]+x[0-9]+')
  local sig=$(grep -oE 'custom_sigmas: \[[^]]*\]' "$log" 2>/dev/null|tail -1|sed 's/custom_sigmas: //')
  [ -z "$sig" ] && sig=$(grep -oE 'res-coupled shift=[0-9.]+' "$log" 2>/dev/null|tail -1)
  local rc=$(grep -ciE 'out of memory|CUDA error' "$log" 2>/dev/null); local st="ok"; [ "${rc:-0}" != "0" ] && st="ERR"
  echo "${gen:--}|${pg}|${res:--}|${sig:--}|${st}"
}
# build an mp4 from a frame dir if missing
mkmp4(){ local d="$1" out="$2"; local nf=$(ls "$d"/f*.png 2>/dev/null|wc -l)
  [ "$nf" -ge 1 ] || return 1
  [ -f "$out" ] && [ "$out" -nt "$d/run.log" ] && return 0
  ffmpeg -y -framerate $FPS -pattern_type glob -i "$d/f*.png" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$out" -loglevel error 2>/dev/null
}

fig(){ local src="$1" cap="$2"; echo "<figure><video src=\"$src\" controls loop muted playsinline></video><figcaption>$cap</figcaption></figure>"; }
figimg(){ local src="$1" cap="$2"; echo "<figure><img src=\"$src\"><figcaption>$cap</figcaption></figure>"; }

H="$EY/index.html"
{
cat <<HEAD
<!doctype html><meta charset=utf-8><title>Wan2.2-VACE eye test — all clips</title>
<style>
 body{background:#0e0e0e;color:#ddd;font:14px/1.5 system-ui,sans-serif;margin:20px;max-width:1500px}
 h1{font-size:21px} h2{font-size:16px;color:#9cf;margin-top:30px;border-bottom:1px solid #333;padding-bottom:4px}
 table{border-collapse:collapse;margin:10px 0;font-size:12px} td,th{border:1px solid #2c2c2c;padding:4px 9px;text-align:left}
 th{background:#1b1b1b} .ok{color:#7d7} .ERR{color:#f77}
 .grid{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
 figure{margin:0;background:#000;padding:6px;border:1px solid #2c2c2c;border-radius:6px}
 video,img{max-width:340px;max-height:380px;display:block;border-radius:4px}
 figcaption{color:#bbb;margin-top:5px;font-size:11px;max-width:340px}
 .big video,.big img{max-width:900px;max-height:560px}
 code{background:#222;padding:1px 5px;border-radius:3px} .note{color:#999;font-size:12px}
 .sw td:first-child{color:#9cf}
</style>
<h1>Wan2.2-VACE-FUN-A14B (distill, RTX 3060) — eye test, all clips</h1>
<p class=note>Generated $(date '+%Y-%m-%d %H:%M'). Murk root-cause = sampler grid (FINDINGS-L8): generic DiscreteScheduler gave a wasted step on the wrong grid; fix = proper lightx2v DMD grid via <code>--sigmas</code> / <code>WAN_DISTILL_SIGMAS</code>. All clips below are the FIXED schedule unless labelled BEFORE.</p>
HEAD

# --- 27s montage ---
echo "<h2>27s music-video montage (1280×704, fixed sigma grid shift 7, t2v hard-cuts)</h2>"
if [ -f "$ROOT/mv27/musicvideo_27s.mp4" ]; then
  cp "$ROOT/mv27/musicvideo_27s.mp4" "$EY/media/montage.mp4"
  echo "<div class='grid big'>"; fig "media/montage.mp4" "Full 27s montage — 8 shots × 4 seeds, hard cut"; echo "</div>"
else
  echo "<p class=note>⏳ montage still rendering ($(ls $ROOT/mv27/stitch/g*.png 2>/dev/null|wc -l)/~416 frames) — re-run this script when done.</p>"
fi
# per-shot representative take (seed 42) as mini-clips + table
echo "<table class=sw><tr><th>take</th><th>res</th><th>gen</th><th>sigmas/shift</th><th>status</th></tr>"
for d in $(ls -d "$ROOT"/mv27/s*/ 2>/dev/null | sort); do
  [ -f "$d/run.log" ] || continue; tag=$(basename "$d"); IFS='|' read gen pg res sig st <<<"$(meta "$d")"
  echo "<tr><td>$tag</td><td>$res</td><td>$gen</td><td>${sig}</td><td class=$st>$st</td></tr>"
done
echo "</table>"
echo "<div class=grid>"
for d in $(ls -d "$ROOT"/mv27/s*_seed42/ 2>/dev/null | sort); do
  tag=$(basename "$d"); mkmp4 "$d" "$EY/media/$tag.mp4" && fig "media/$tag.mp4" "$tag"
done
echo "</div>"

# --- shift x res sweep ---
echo "<h2>Shift × resolution sweep (face close-up, FR=5) — sharpness/timing/VRAM per config</h2>"
if ls "$ROOT"/shiftsweep/*x*_sh*/run.log >/dev/null 2>&1; then
  echo "<table class=sw><tr><th>config</th><th>res(latent)</th><th>gen</th><th>peak VRAM</th><th>sigmas</th><th>status</th></tr>"
  for d in $(ls -d "$ROOT"/shiftsweep/*x*_sh*/ 2>/dev/null | sort); do
    [ -f "$d/run.log" ] || continue; tag=$(basename "$d"); IFS='|' read gen pg res sig st <<<"$(meta "$d")"
    echo "<tr><td>$tag</td><td>$res</td><td>$gen</td><td>${pg} GiB</td><td>${sig}</td><td class=$st>$st</td></tr>"
  done
  echo "</table><div class=grid>"
  for d in $(ls -d "$ROOT"/shiftsweep/*x*_sh*/ 2>/dev/null | sort); do
    tag=$(basename "$d"); mkmp4 "$d" "$EY/media/sw_$tag.mp4" && fig "media/sw_$tag.mp4" "$tag"
  done
  echo "</div>"
else
  echo "<p class=note>⏳ shift sweep not run yet (queued after the montage).</p>"
fi

# --- sigma-fix before/after + smoke ---
echo "<h2>Sigma-fix before/after (root-cause proof) + smoke</h2><div class=grid>"
for p in mid_before_after face_before_after shift_mid_7v5 shift_face_7v5; do cp "$ROOT/$p.png" "$EY/media/$p.png" 2>/dev/null || true; done
[ -f "$EY/media/mid_before_after.png" ] && figimg "media/mid_before_after.png" "mid-frame: TOP=before(discrete grid) BOTTOM=after(proper grid), same seed/shift"
[ -f "$EY/media/face_before_after.png" ] && figimg "media/face_before_after.png" "face: LEFT=before RIGHT=after"
[ -f "$EY/media/shift_mid_7v5.png" ] && figimg "media/shift_mid_7v5.png" "shift 7 (L) vs 5 (R), proper grid"
mkmp4 "$ROOT/t2v_smoke" "$EY/media/smoke.mp4" && fig "media/smoke.mp4" "t2v smoke (shift7 fixed)"
echo "</div>"
} > "$H"

echo "wrote $H"
# (re)serve
if ! pgrep -f "http.server $PORT" >/dev/null 2>&1; then
  ( cd "$EY" && nohup python3 -m http.server "$PORT" --bind 0.0.0.0 >/tmp/eyetest_clips_http.log 2>&1 & )
  sleep 1
fi
echo "SERVING: http://${HOSTIP:-10.0.0.208}:$PORT/"
