#!/usr/bin/env python3
"""
NAVA cpp-port eye-test viewer (live HTTP server).

The owner is on a headless box driving via the Claude CLI, so rendered clips +
their VRAM/host-RAM/timing are only visible through a web page. This serves one,
LIVE (re-scans the runs dir on every page load — no static regen step).

Layout convention (each render writes one run dir):
    RUNS_DIR/<run_name>/
        clip.webm | clip.mp4        the rendered clip (optional; may be missing on fail)
        meta.json                   { label, ok, rc, phase, resolution|w|h, frames,
                                      steps, wall_s, load_s, s_per_step,
                                      vram_peak_mib, vram_idle_mib, vram_series:[...],
                                      host_ram_peak_mib, host_ram_series:[...],
                                      psnr|psnr99, notes, args, ts }
        (any field optional; the page degrades gracefully)

A helper `navarun_meta()` for the C++/shell harness to emit meta.json lives at the
bottom as a copy-paste reference.

Run:
    python3 tools/nava_eyetest_server.py            # serves :8097 over /mnt/hdd/nava/cpp-runs
    RUNS_DIR=/path PORT=8097 python3 tools/nava_eyetest_server.py
Then open http://10.0.0.208:8097  (manual refresh; auto-refresh is a toggle, default OFF).
"""
import json, os, html, subprocess, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUNS_DIR = os.environ.get("RUNS_DIR", "/mnt/hdd/nava/cpp-runs")
PORT = int(os.environ.get("PORT", "8097"))
VRAM_BUDGET_MIB = int(os.environ.get("VRAM_BUDGET_MIB", "7680"))  # 7.5 GB slab line


def _dur(path):
    try:
        o = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=20)
        return float(o.stdout.strip())
    except Exception:
        return None


def scan_runs():
    runs = []
    if not os.path.isdir(RUNS_DIR):
        return runs
    for name in os.listdir(RUNS_DIR):
        d = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(d):
            continue
        meta = {}
        mp = os.path.join(d, "meta.json")
        if os.path.isfile(mp):
            try:
                meta = json.load(open(mp))
            except Exception as e:
                meta = {"_meta_error": str(e)}
        clip = None
        for cand in ("clip.webm", "clip.mp4", "out.webm", "out.mp4"):
            if os.path.isfile(os.path.join(d, cand)):
                clip = cand
                break
        if clip is None:
            for f in sorted(os.listdir(d)):
                if f.endswith((".webm", ".mp4")):
                    clip = f
                    break
        mtime = os.path.getmtime(mp) if os.path.isfile(mp) else os.path.getmtime(d)
        runs.append({"name": name, "dir": d, "clip": clip, "meta": meta, "mtime": mtime})
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def sparkline_svg(series, budget=None, w=360, h=48):
    if not series:
        return ""
    vals = [v for v in series if isinstance(v, (int, float))]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = max(hi - lo, 1.0)
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = (i / max(n - 1, 1)) * (w - 2) + 1
        y = h - 2 - ((v - lo) / rng) * (h - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    budget_line = ""
    if budget and lo <= budget <= hi:
        by = h - 2 - ((budget - lo) / rng) * (h - 4)
        budget_line = (f'<line x1="0" y1="{by:.1f}" x2="{w}" y2="{by:.1f}" '
                       f'stroke="#e55" stroke-dasharray="4 3" stroke-width="1"/>')
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="#4af" stroke-width="1.5" points="{poly}"/>'
            f'{budget_line}</svg>')


def fmt(v, suffix="", nd=None):
    if v is None:
        return "—"
    if isinstance(v, float) and nd is not None:
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def render_card(run):
    m = run["meta"]
    name = run["name"]
    label = m.get("label", name)
    ok = m.get("ok")
    rc = m.get("rc")
    status = ("✅ ok" if ok or rc == 0 else
              (f"❌ rc={rc}" if rc not in (None, 0) else "⚠️"))
    res = m.get("resolution") or (f'{m.get("w")}×{m.get("h")}' if m.get("w") else "—")
    s_per_step = m.get("s_per_step")
    if s_per_step is None and m.get("wall_s") and m.get("steps"):
        try:
            s_per_step = m["wall_s"] / m["steps"]
        except Exception:
            s_per_step = None
    vram_peak = m.get("vram_peak_mib")
    vram_idle = m.get("vram_idle_mib")
    net_vram = (vram_peak - vram_idle) if (isinstance(vram_peak, (int, float))
                                           and isinstance(vram_idle, (int, float))) else None

    clip = run["clip"]
    if clip:
        cdur = _dur(os.path.join(run["dir"], clip))
        src = "/clip/" + urllib.parse.quote(name) + "/" + urllib.parse.quote(clip)
        video = (f'<video src="{html.escape(src)}" controls loop muted '
                 f'preload="metadata"></video>')
        if cdur:
            label += f" · {cdur:.1f}s"
    else:
        video = '<div class="noclip">no clip — failed or pending</div>'

    # The owner judges timing and seams on a filmstrip, not by scrubbing, so show it inline
    # under the clip whenever the render harness wrote one.
    strip = m.get("filmstrip") or "filmstrip.png"
    if os.path.isfile(os.path.join(run["dir"], strip)):
        ssrc = "/clip/" + urllib.parse.quote(name) + "/" + urllib.parse.quote(strip)
        video += (f'<a href="{html.escape(ssrc)}" target="_blank">'
                  f'<img class="strip" src="{html.escape(ssrc)}" loading="lazy"></a>')

    rows = [
        ("status", status),
        ("phase", fmt(m.get("phase"))),
        ("res", res),
        ("frames", fmt(m.get("frames"))),
        ("steps", fmt(m.get("steps"))),
        ("s/step", fmt(s_per_step, "s", 1)),
        ("wall", fmt(m.get("wall_s"), "s", 0) if isinstance(m.get("wall_s"), float)
            else fmt(m.get("wall_s"), "s")),
        ("load", fmt(m.get("load_s"), "s", 0) if isinstance(m.get("load_s"), float)
            else fmt(m.get("load_s"), "s")),
        ("VRAM peak", fmt(vram_peak, " MiB")),
        ("VRAM net", fmt(net_vram, " MiB")),
        ("host RAM peak", fmt(m.get("host_ram_peak_mib"), " MiB")),
        ("PSNR", fmt(m.get("psnr99") or m.get("psnr"))),
    ]
    tbl = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                  for k, v in rows)

    spark_v = sparkline_svg(m.get("vram_series"), VRAM_BUDGET_MIB)
    spark_r = sparkline_svg(m.get("host_ram_series"))
    sparks = ""
    if spark_v:
        sparks += f'<div class="sparkbox"><span>VRAM MiB (— = {VRAM_BUDGET_MIB} slab)</span>{spark_v}</div>'
    if spark_r:
        sparks += f'<div class="sparkbox"><span>host RAM MiB</span>{spark_r}</div>'

    notes = m.get("notes")
    notes_h = f'<div class="notes">{html.escape(str(notes))}</div>' if notes else ""
    args = m.get("args")
    args_h = (f'<details><summary>args</summary><pre>{html.escape(str(args))}</pre></details>'
              if args else "")
    if m.get("_meta_error"):
        notes_h += f'<div class="notes err">meta.json error: {html.escape(m["_meta_error"])}</div>'

    return (f'<div class="card"><h3>{html.escape(label)}</h3>'
            f'<div class="body"><div class="vid">{video}{sparks}</div>'
            f'<table class="meta">{tbl}</table></div>{notes_h}{args_h}</div>')


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>MiniMax-H3 eye-test</title>
<style>
 body{{background:#111;color:#ddd;font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:16px}}
 h1{{font-size:18px;margin:0 0 4px}}
 .bar{{position:sticky;top:0;background:#111;padding:8px 0 12px;border-bottom:1px solid #333;z-index:10;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
 .bar a{{color:#4af;text-decoration:none}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px;margin-top:16px}}
 .card{{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px}}
 .card h3{{margin:0 0 8px;font-size:14px;color:#fff}}
 .body{{display:flex;gap:12px;align-items:flex-start;min-width:0}}
 .vid{{flex:0 0 280px}}
 .metacol{{flex:1;min-width:0;overflow:hidden}}
 video{{width:280px;border-radius:4px;background:#000}}
 .noclip{{width:280px;height:160px;display:flex;align-items:center;justify-content:center;color:#888;background:#000;border-radius:4px}}
 img.strip{{width:280px;margin-top:6px;border-radius:4px;background:#000;display:block}}
 table.meta{{flex:1;min-width:0;border-collapse:collapse;font-size:12px;table-layout:fixed;width:100%}}
 table.meta td{{padding:1px 6px;border-bottom:1px solid #262626;vertical-align:top}}
 table.meta td:first-child{{color:#999;white-space:nowrap}}
 table.meta td:last-child{{color:#eee;text-align:right;font-variant-numeric:tabular-nums}}
 .sparkbox{{margin-top:8px;font-size:10px;color:#888;max-width:100%;overflow:hidden}}
 .sparkbox span{{display:block}}
 svg.spark{{background:#0d0d0d;border-radius:3px;margin-top:2px;width:100%;height:48px;display:block}}
 .notes{{margin-top:8px;color:#bbb;font-size:12px;white-space:pre-wrap}}
 .notes.err{{color:#e88}}
 details{{margin-top:6px}} summary{{cursor:pointer;color:#888;font-size:11px}}
 pre{{background:#0d0d0d;padding:8px;border-radius:4px;overflow:auto;font-size:11px;color:#9c9}}
 .empty{{color:#888;margin-top:40px;text-align:center}}
</style></head><body>
<div class="bar">
 <h1>NAVA cpp eye-test</h1>
 <span>{count} runs · {runs_dir}</span>
 <a href="/">↻ refresh</a>
 <label><input type="checkbox" id="auto"> auto-refresh</label>
 <input type="number" id="ivl" value="10" min="2" style="width:48px;background:#222;color:#ddd;border:1px solid #444;border-radius:3px"> s
</div>
{body}
<script>
 const cb=document.getElementById('auto'), iv=document.getElementById('ivl');
 cb.checked = localStorage.getItem('nava_auto')==='1';
 iv.value = localStorage.getItem('nava_ivl')||'10';
 let t=null;
 function arm(){{ if(t)clearInterval(t); t=null;
   if(cb.checked){{ t=setInterval(()=>location.reload(), Math.max(2,+iv.value)*1000); }} }}
 cb.onchange=()=>{{ localStorage.setItem('nava_auto', cb.checked?'1':'0'); arm(); }};
 iv.onchange=()=>{{ localStorage.setItem('nava_ivl', iv.value); arm(); }};
 arm();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urllib.parse.unquote(self.path.split("?", 1)[0])
        if path == "/" or path == "/index.html":
            runs = scan_runs()
            if runs:
                body = '<div class="grid">' + "".join(render_card(r) for r in runs) + "</div>"
            else:
                body = (f'<div class="empty">no runs yet in <code>{html.escape(RUNS_DIR)}</code>'
                        f'<br>each render should write <code>&lt;name&gt;/meta.json</code> '
                        f'(+ clip.webm)</div>')
            page = PAGE.format(count=len(runs), runs_dir=html.escape(RUNS_DIR), body=body)
            self._send(200, page.encode(), "text/html; charset=utf-8")
            return
        if path.startswith("/clip/"):
            parts = path[len("/clip/"):].split("/", 1)
            if len(parts) == 2:
                run, fn = parts
                fp = os.path.realpath(os.path.join(RUNS_DIR, run, fn))
                if fp.startswith(os.path.realpath(RUNS_DIR)) and os.path.isfile(fp):
                    self._send_file(fp)
                    return
        self._send(404, b"not found", "text/plain")

    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, fp):
        ctype = ("video/webm" if fp.endswith(".webm") else
                 "image/png" if fp.endswith(".png") else
                 "image/jpeg" if fp.endswith((".jpg", ".jpeg")) else "video/mp4")
        sz = os.path.getsize(fp)
        # minimal range support so <video> seeking works
        rng = self.headers.get("Range")
        with open(fp, "rb") as f:
            if rng and rng.startswith("bytes="):
                s, _, e = rng[6:].partition("-")
                start = int(s) if s else 0
                end = int(e) if e else sz - 1
                end = min(end, sz - 1)
                f.seek(start)
                data = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{sz}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(sz))
                self.end_headers()
                self.wfile.write(f.read())


if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)
    print(f"NAVA eye-test viewer: http://0.0.0.0:{PORT}  (runs: {RUNS_DIR})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
