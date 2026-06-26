#!/usr/bin/env bash
# Build + serve an eye-test page for the Wan2.2-TI2V-5B Turbo clips in ti2v5b_out/.
# Host-side only (python http.server). Re-run anytime to pick up new clips.
#   ./gen_ti2v5b_eyetest.sh           -> http://<host>:8099/
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="$PWD/ti2v5b_out"; PORT="${PORT:-8099}"
HOSTIP=$(hostname -I 2>/dev/null | awk '{print $1}'); HOSTIP="${HOSTIP:-10.0.0.208}"
mkdir -p "$OUT"

# shot index -> short prompt label (mirror run_ti2v5b_shots.sh order)
declare -a SHOTLBL=(
"shot0 face close-up (singing into mic)"
"shot1 car rolls to bar (tracking, neon)"
"shot2 car door, steps out singing (push-in)"
"shot3 pushes door, walks into bar (dolly)"
"shot4 sits at bar counter singing (static)"
"shot5 backup singer, curly hair (rim light)"
"shot6 drummer, motion blur (smoky bar)"
"shot7 rain-slick neon street walk (steadicam)"
)

HTML="$OUT/index.html"
{
echo '<!doctype html><meta charset=utf-8><title>Wan2.2-TI2V-5B Turbo — eye test</title>'
echo '<style>body{background:#111;color:#ddd;font:14px system-ui;margin:18px}'
echo 'h1{font-size:18px} .grid{display:flex;flex-wrap:wrap;gap:14px}'
echo 'figure{margin:0;background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:8px;width:560px}'
echo 'video{width:100%;border-radius:5px;background:#000} figcaption{margin-top:6px;font-size:12px;line-height:1.45;color:#bbb}'
echo 'b{color:#fff} .m{color:#8c8;font-family:monospace}</style>'
echo "<h1>Wan2.2-TI2V-5B Turbo — t2v eval vs LTX-2.3 prompts <span class=m>($(date '+%Y-%m-%d %H:%M'))</span></h1>"
echo "<p>1280×704 · 4 steps · cfg 1 · euler · simple · shift 8. Resident DiT, umT5 on CPU. LTX-2.3 baseline ≈ 111 render-s/s-video.</p>"
# start images (flux.2 stills for i2v)
if ls "$OUT"/flux_*.png >/dev/null 2>&1; then
  echo '<h2 style="font-size:15px;color:#9cf">flux.2 start images (i2v base)</h2><div class=grid>'
  for png in $(ls -t "$OUT"/flux_*.png 2>/dev/null); do
    b=$(basename "$png")
    echo "<figure style='width:420px'><img src=\"$b\" style='width:100%;border-radius:5px'><figcaption><span class=m>$b</span></figcaption></figure>"
  done
  echo '</div>'
fi
echo '<h2 style="font-size:15px;color:#9cf">video clips</h2>'
echo '<div class=grid>'
for mp4 in $(ls -t "$OUT"/shot*.mp4 2>/dev/null); do
  base=$(basename "$mp4" .mp4)
  si=$(echo "$base" | sed -n 's/^shot\([0-9]\+\)_.*/\1/p')
  lbl="${SHOTLBL[$si]:-shot$si}"
  meta="$OUT/$base.meta"
  if [ -f "$meta" ]; then info=$(tr '\n' ' ' < "$meta"); else
    g=$(grep -oE 'generate_video completed in [0-9.]+s' "$OUT/$base.log" 2>/dev/null | tail -1)
    info="(no .meta) $g"
  fi
  # LTX clips carry native audio -> don't mute them
  case "$base" in *ltx*) MUTE="" ;; *) MUTE="muted" ;; esac
  echo "<figure><video src=\"$base.mp4\" controls loop $MUTE playsinline></video>"
  echo "<figcaption><b>$lbl</b>$( [ -z "$MUTE" ] && echo ' 🔊' )<br><span class=m>$base</span><br>$info</figcaption></figure>"
done
echo '</div>'
} > "$HTML"

# (re)serve
if ! pgrep -f "http.server $PORT" >/dev/null 2>&1; then
  ( cd "$OUT" && nohup python3 -m http.server "$PORT" --bind 0.0.0.0 >/tmp/ti2v5b_eyetest_http.log 2>&1 & )
  sleep 0.5
fi
n=$(ls "$OUT"/shot*.mp4 2>/dev/null | wc -l)
echo "SERVING $n clips: http://${HOSTIP}:$PORT/"
