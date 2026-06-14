#!/usr/bin/env bash
# Build mvp_out/index.html from results.csv + the rendered mp4s, and (re)serve it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="$PWD/mvp_out"; PORT="${PORT:-8097}"
cp models/_drive/char.png "$OUT/char.png" 2>/dev/null || true

row() { # name label res frames steps
  local n="$1" label="$2" res="$3" fr="$4" st="$5"
  local line=$(grep "^$n," "$OUT/results.csv" 2>/dev/null | tail -1)
  local wall=$(echo "$line" | cut -d, -f2); local peak=$(echo "$line" | cut -d, -f3); local rc=$(echo "$line" | cut -d, -f4)
  local vg="-"; [ -n "${peak:-}" ] && [ "$peak" != "" ] && vg=$(awk "BEGIN{printf \"%.1f\", $peak/1024}")
  local stat="ok"; [ "${rc:-1}" != "0" ] && stat="FAIL(rc=$rc)"
  echo "<tr><td>$label</td><td>$res</td><td>$fr</td><td>$st</td><td>${wall:-–}s</td><td>${vg} GiB</td><td class=\"$([ "$stat" = ok ] && echo ok || echo bad)\">$stat</td></tr>"
}

vid() { local f="$1" cap="$2"; if [ -f "$OUT/$f" ]; then
  echo "<figure><video src=\"$f\" controls loop muted playsinline></video><figcaption>$cap</figcaption></figure>"
 else echo "<figure class=missing><div>no clip — render failed</div><figcaption>$cap</figcaption></figure>"; fi; }

cat > "$OUT/index.html" <<HTML
<!doctype html><meta charset=utf-8><title>Wan2.2 + InfiniteTalk — MVP eye test</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.5 system-ui,sans-serif;margin:24px;max-width:1100px}
 h1{font-size:20px} h2{font-size:15px;color:#9cf;margin-top:28px}
 table{border-collapse:collapse;margin:12px 0;font-size:13px} td,th{border:1px solid #333;padding:6px 10px;text-align:left}
 th{background:#1c1c1c} .ok{color:#7d7} .bad{color:#f77}
 .grid{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
 figure{margin:0;background:#000;padding:8px;border:1px solid #333;border-radius:6px}
 video{max-height:520px;display:block;border-radius:4px} figcaption{color:#aaa;margin-top:6px;font-size:12px;max-width:320px}
 .missing div{width:300px;height:420px;display:grid;place-items:center;color:#f77;border:1px dashed #533}
 img.ref{max-height:300px;border:1px solid #333;border-radius:6px} code{background:#222;padding:1px 5px;border-radius:3px}
 .note{color:#999;font-size:12px}
</style>
<h1>Wan2.2 + InfiniteTalk — MVP "see it all work" (RTX 3060)</h1>
<p class=note>Branch <code>wan22-infinitetalk</code>, CUDA build, distilled 4-step, 480p portrait bucket (flow-shift 7), LTX-leaning VRAM params (offload + mmap + max-vram 7.3). Reference image is the same char.png used in the LTX music video.</p>

<h2>Timing &amp; VRAM</h2>
<table>
<tr><th>stage</th><th>res</th><th>frames</th><th>steps</th><th>wall</th><th>peak VRAM</th><th>status</th></tr>
$(row a14b "Wan2.2-I2V-A14B (scene-gen)" "480×832" 21 "2+2")
$(row infinitetalk "InfiniteTalk (lip-sync)" "256×256" 13 4)
</table>
<p class=note>peak VRAM = whole-GPU sample (the InfiniteTalk figure may include concurrent prod gemma/ace spikes; A14B's 6.5 GB is its own clean run). LTX-2.3 reference (fully tuned): 720p/97f/4s ≈ 239s. These are <b>untuned</b> CUDA first-light.</p>

<h2>The 12 GB wall (these are the LTX problems again)</h2>
<table>
<tr><th>attempt</th><th>result</th><th>cause → LTX fix that already solved this class</th></tr>
<tr><td>A14B 480×832 <b>81f</b></td><td class=bad>OOM @ 11.9 GB (DiT sampling)</td><td>DiT activations over ~131k-token latent; <code>--max-vram</code> caps only weight-staging → <b>FINDINGS-01 + 03</b></td></tr>
<tr><td>A14B 480×832 init-encode</td><td class=ok>fixed by <code>--vae-tiling</code></td><td>VAE encode buffer → <b>FINDINGS-05</b> (tiling ladder)</td></tr>
<tr><td>InfiniteTalk 480×832 81f</td><td class=bad>segfault @ 10.9 GB</td><td>same activation wall + unhandled CUDA-OOM</td></tr>
<tr><td>InfiniteTalk 448² / 480×832 21f</td><td class=bad>OOM: 6.8 GB VAE-encode buffer</td><td>conditioning encode sets <code>temporal_tiling=false</code> → <b>FINDINGS-11</b> (needs temporal tiling)</td></tr>
<tr><td>A14B 480×832 21f / InfiniteTalk 256² 13f</td><td class=ok>complete</td><td>small enough to fit untuned</td></tr>
</table>
<p class=note>Verdict: pipeline + CUDA build are sound and Wan2.2 quality is clean. The blockers are the same VRAM/offload/tiling plumbing we already solved for LTX-2.3 — porting work, not new research. Next: port offload-overlap + VAE temporal-tiling + residency levers onto the Wan DiT &amp; InfiniteTalk encode, then it scales to real-length clips.</p>

<h2>Clips</h2>
<div class=grid>
$(vid a14b.mp4 "Wan2.2-I2V-A14B — directed scene shot from char.png (no audio). Tests scene quality + motion + identity.")
$(vid it.mp4 "InfiniteTalk — lip-sync dub of the song onto char.png (audio embedded). Tests the audio-driven mouth.")
</div>

<h2>Reference</h2>
<div class=grid>
<figure><img class=ref src="char.png"><figcaption>char.png — identity input (same as LTX)</figcaption></figure>
</div>
HTML
echo "wrote $OUT/index.html"
# (re)serve
docker rm -f wan22-eyetest >/dev/null 2>&1 || true
pkill -f "http.server $PORT" 2>/dev/null || true
( cd "$OUT" && nohup python3 -m http.server "$PORT" --bind 0.0.0.0 >/dev/null 2>&1 & )
echo "serving http://localhost:$PORT/  (and http://$(hostname -I | awk '{print $1}'):$PORT/ )"
