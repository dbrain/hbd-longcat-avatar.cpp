#!/usr/bin/env python3
"""flux2.cpp gallery — interactive gen + comparison grid over the resident server.

Drives the resident flux2-server (sdcpp async API) via flux_client: submit →
sample peak VRAM → poll → scrape per-stage timings → save image + meta.json +
stamp the build sha. Every gen is keyed (prompt|seed|steps|cfg|WxH); the FIRST
gen for a key becomes the golden, and later gens score PSNR vs that golden — so
a regression ("we broke the image") is visible at a glance. In-flight Cancel
button hits the server's cooperative-abort.

  python3 tools/gallery_server.py            # serve on 0.0.0.0:8096

The resident server hosts ONE model (whatever iter.sh serve loaded); set
FLUX2_SERVER_MODEL to label it. GPU is single — gens serialize behind a lock.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import flux_client as fc

REPO = Path(__file__).resolve().parent.parent
GALLERY = REPO / "gallery"
GOLDENS = GALLERY / "goldens"
GALLERY.mkdir(exist_ok=True)
GOLDENS.mkdir(exist_ok=True)
PORT = 8096
SERVER_BASE = os.environ.get("FLUX2_SERVER", fc.DEFAULT_BASE)
SERVER_MODEL = os.environ.get("FLUX2_SERVER_MODEL", "klein-base-9b-q4 / enc-q4")

_gen_lock = threading.Lock()
_current = {"job_id": None}

KSTYLE = ("full body standing figure, a single isolated character floating centered on a "
          "completely flat empty solid cream-white backdrop, smooth clean blank background, "
          "even flat front lighting. Clean digital painterly illustration, warm earthy muted "
          "palette, soft hand-painted round-brush shading, brown-toned soft outlines, slight "
          "oil-painting texture. Classic tabletop board-game character art in the manner of "
          "Klemens Franz or Paul Bonner.")
CIV = {
    "caveman": "a stone-age caveman hunter wearing rugged fur clothing, holding a wooden spear",
    "roman": "a Roman legionary soldier with a rectangular scutum shield, a gladius short sword and a plumed helmet",
    "knight": "a medieval knight in full plate armour holding a longsword and a kite shield",
    "musketeer": "a 17th-century musketeer with a flintlock musket, a wide-brimmed feathered hat and a long coat",
    "modern": "a modern infantry soldier in camouflage tactical gear with an assault rifle and a combat helmet",
}
PRESETS = {f"civ:{k}": f"{v}, {KSTYLE}" for k, v in CIV.items()}


def git_sha(d: Path):
    try:
        sha = subprocess.check_output(["git", "-C", str(d), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(d), "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return "?", False


def gen_key(p):
    raw = f"{p['prompt']}|{p['seed']}|{p['steps']}|{p['cfg']}|{p['width']}x{p['height']}|{SERVER_MODEL}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def psnr(a_path: Path, b_bytes: bytes) -> float | None:
    try:
        import io
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(io.BytesIO(b_bytes)).convert("RGB"), dtype=np.float64)
        if a.shape != b.shape:
            return None
        mse = float(np.mean((a - b) ** 2))
        if mse == 0:
            return 999.0
        return float(round(10 * np.log10((255.0 ** 2) / mse), 2))  # float() so meta JSON-serializable (np.float64 -> np.bool_ in is_golden)
    except Exception:
        return None


def run_gen(params: dict) -> dict:
    with _gen_lock:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        p = {
            "prompt": params.get("prompt", "").strip(),
            "seed": int(params.get("seed") or 42),
            "steps": int(params.get("steps") or 8),
            "cfg": float(params.get("cfg") or 5.0),
            "width": int(params.get("width") or 512),
            "height": int(params.get("height") or 512),
            "sampler": params.get("sampler") or "euler",
            "label": params.get("label", "").strip(),
            "vae_tiling": params.get("vae_tiling") == "1",
        }
        body = fc.build_body(p["prompt"], width=p["width"], height=p["height"], seed=p["seed"],
                             steps=p["steps"], cfg=p["cfg"], sampler=p["sampler"], vae_tiling=p["vae_tiling"])
        r = fc.run(body, base=SERVER_BASE, on_submit=lambda jid: _current.update(job_id=jid))
        _current["job_id"] = None

        ok = r["status"] == "completed" and r["image_bytes"]
        out_name = f"{ts}.png"
        key = gen_key(p)
        pnsr_val = None
        if ok:
            (GALLERY / out_name).write_bytes(r["image_bytes"])
            golden = GOLDENS / f"{key}.png"
            if golden.exists():
                pnsr_val = psnr(golden, r["image_bytes"])
            else:
                golden.write_bytes(r["image_bytes"])
                pnsr_val = 999.0  # this gen IS the golden
        sha, dirty = git_sha(REPO)
        ggml_sha, _ = git_sha(REPO / "ggml")
        meta = {
            "id": ts, "ts": datetime.now().strftime("%Y-%m-%d %H:%M"), "ok": bool(ok),
            "image": out_name if ok else None, "key": key, "psnr": pnsr_val,
            "is_golden": pnsr_val == 999.0,
            "model": SERVER_MODEL, **p,
            "status": r["status"],
            "wall_s": r["wall_s"], "cond_s": r["cond_s"], "dit_s": r["dit_s"],
            "dit_s_per_step": r["dit_s_per_step"], "vae_s": r["vae_s"], "overhead_s": r["overhead_s"],
            "vram_idle_mib": r["vram_idle_mib"], "vram_peak_mib": r["vram_peak_mib"],
            "vram_service_mib": r["vram_service_mib"],
            "sha": sha, "ggml_sha": ggml_sha, "dirty": dirty,
        }
        if not ok:
            meta["error"] = str(r.get("error"))
        (GALLERY / f"{ts}.meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def load_gallery():
    items = []
    for mf in sorted(GALLERY.glob("*.meta.json"), reverse=True):
        try:
            items.append(json.loads(mf.read_text()))
        except Exception:
            pass
    return items


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>flux2 gallery</title>
<style>
 body{{font:14px/1.4 system-ui,sans-serif;margin:0;background:#15171a;color:#e8e8e8}}
 header{{padding:14px 18px;background:#1d2024;border-bottom:1px solid #2c3036;position:sticky;top:0;z-index:5}}
 h1{{font-size:15px;margin:0 0 10px;color:#8fd}}
 form{{display:grid;grid-template-columns:repeat(auto-fit,minmax(80px,1fr));gap:8px;align-items:end}}
 label{{display:block;font-size:11px;color:#9aa;margin-bottom:2px}}
 input,select,textarea{{width:100%;box-sizing:border-box;background:#0f1113;color:#e8e8e8;border:1px solid #333;border-radius:5px;padding:6px;font:13px system-ui}}
 textarea{{height:60px;resize:vertical;font-size:12px}}
 button{{background:#2a6;color:#04130c;border:0;border-radius:5px;padding:8px 14px;font-weight:600;cursor:pointer}}
 button.sec{{background:#334;color:#cde}} button.warn{{background:#c44;color:#fff}}
 .presets{{margin:8px 0 0}} .presets button{{background:#26303a;color:#bdf;font-weight:500;padding:5px 9px;margin:2px;font-size:12px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(265px,1fr));gap:14px;padding:16px}}
 .card{{background:#1d2024;border:1px solid #2c3036;border-radius:8px;overflow:hidden}}
 .card img{{width:100%;display:block;background:#000}}
 .meta{{padding:8px 10px;font-size:12px}} .meta .p{{color:#cde;margin-bottom:6px;max-height:34px;overflow:hidden}}
 .row{{display:flex;flex-wrap:wrap;gap:4px 10px;color:#9aa;font-size:11px;margin-top:4px}} .b{{color:#fff}}
 .tag{{display:inline-block;background:#26303a;border-radius:3px;padding:1px 6px;font-size:11px;color:#bdf}}
 .gold{{color:#fd4}} .ok{{color:#6e6}} .bad{{color:#f77}} .dirty{{color:#fb4}}
 .err{{color:#f88;font-size:11px;white-space:pre-wrap;max-height:90px;overflow:auto}}
 #overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9;align-items:center;justify-content:center;flex-direction:column;gap:14px}}
 #overlay.on{{display:flex}} .spin{{width:46px;height:46px;border:5px solid #333;border-top-color:#2a6;border-radius:50%;animation:s 1s linear infinite}}
 @keyframes s{{to{{transform:rotate(360deg)}}}}
</style></head><body>
<header>
 <h1>flux2.cpp gallery &mdash; model <span class=tag>{model}</span> build <span class=tag>{sha}{dirtymark}</span> ggml <span class=tag>{ggml}</span></h1>
 <form id=f method=post action=/gen>
  <div style=grid-column:1/-1><label>prompt</label><textarea name=prompt placeholder="subject, style...">{lastprompt}</textarea></div>
  <div><label>seed</label><input name=seed value=42></div>
  <div><label>W</label><input name=width value=512></div>
  <div><label>H</label><input name=height value=512></div>
  <div><label>steps</label><input name=steps value=8></div>
  <div><label>cfg</label><input name=cfg value=5.0></div>
  <div><label>sampler</label><select name=sampler><option>euler</option><option>euler_a</option><option>dpm++2m</option></select></div>
  <div><label>note</label><input name=label placeholder="lap label"></div>
  <div><label>vae-tile <input type=checkbox name=vae_tiling value=1 style=width:auto></label></div>
  <div><button type=submit>Generate</button></div>
 </form>
 <div class=presets>
  <b style=color:#9aa;font-size:11px>civ:</b> {civbtns}
  <button class=sec type=button onclick="runAll()">run all civ</button>
 </div>
</header>
<div class=grid>{cards}</div>
<div id=overlay><div class=spin></div><p style=color:#cde id=ovtext>rendering on GPU&hellip;</p>
 <button class=warn type=button onclick="doCancel()">Cancel render</button></div>
<script>
 const PRESETS={presets_json};
 const ta=document.querySelector('[name=prompt]');
 function setp(k){{ta.value=PRESETS[k];}}
 document.getElementById('f').addEventListener('submit',()=>document.getElementById('overlay').classList.add('on'));
 async function doCancel(){{ document.getElementById('ovtext').textContent='cancelling...'; await fetch('/cancel',{{method:'POST'}}); }}
 async function runAll(){{
   document.getElementById('overlay').classList.add('on');
   for(const k of Object.keys(PRESETS)){{
     document.getElementById('ovtext').textContent='rendering '+k+'...';
     const fd=new URLSearchParams(); fd.set('prompt',PRESETS[k]); fd.set('seed','42');
     fd.set('width','512'); fd.set('height','512'); fd.set('steps','8'); fd.set('cfg','5.0');
     fd.set('sampler','euler'); fd.set('label',k);
     await fetch('/gen',{{method:'POST',body:fd}});
   }}
   location.reload();
 }}
</script></body></html>"""


def card(m):
    if not m.get("ok"):
        return (f"<div class=card><div class=meta><div class=p>{html.escape(m.get('prompt','')[:120])}</div>"
                f"<div class=row><span class=bad>{html.escape(m.get('status','FAILED'))}</span> seed {m.get('seed')}</div>"
                f"<div class=err>{html.escape((m.get('error') or '')[-500:])}</div></div></div>")
    ps = m.get("psnr")
    if m.get("is_golden"):
        pstr = "<span class=gold>★ golden</span>"
    elif ps is None:
        pstr = "<span class=row>psnr n/a</span>"
    else:
        cls = "ok" if ps >= 35 else "bad"
        pstr = f"<span class={cls}>PSNR {ps} dB</span>"
    return f"""<div class=card><img src="/img/{m['image']}" loading=lazy><div class=meta>
  <div class=p>{html.escape(m.get('label') or m.get('prompt','')[:120])}</div>
  <div class=row><span class=tag>{m.get('width')}&times;{m.get('height')}</span>
   <span>seed <span class=b>{m.get('seed')}</span></span><span>{m.get('steps')}st/cfg{m.get('cfg')}</span>{pstr}</div>
  <div class=row><span>wall <span class=b>{m.get('wall_s')}s</span></span>
   <span>dit <span class=b>{m.get('dit_s')}s</span> ({m.get('dit_s_per_step')}/st)</span>
   <span>vae {m.get('vae_s')}s</span><span>cond {m.get('cond_s')}s</span></div>
  <div class=row><span>VRAM peak <span class=b>{m.get('vram_peak_mib')} MiB</span></span>
   <span>svc {m.get('vram_service_mib')}</span></div>
  <div class=row><span class={'dirty' if m.get('dirty') else ''}>build {m.get('sha')}{'*' if m.get('dirty') else ''}</span>
   <span>ggml {m.get('ggml_sha')}</span><span>{m.get('ts')}</span></div>
 </div></div>"""


def render_page(lastprompt=""):
    sha, dirty = git_sha(REPO)
    ggml_sha, _ = git_sha(REPO / "ggml")
    cards = "".join(card(m) for m in load_gallery()) or "<p style=padding:20px;color:#9aa>no gens yet</p>"
    civbtns = " ".join(f"<button class=sec type=button onclick=\"setp('civ:{k}')\">{k}</button>" for k in CIV)
    return PAGE.format(model=SERVER_MODEL, sha=sha, dirtymark="*" if dirty else "", ggml=ggml_sha,
                       lastprompt=html.escape(lastprompt), cards=cards, civbtns=civbtns,
                       presets_json=json.dumps(PRESETS))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", ""):
            self._send(200, render_page())
        elif path.startswith("/img/"):
            f = GALLERY / Path(path[5:]).name
            self._send(200, f.read_bytes(), "image/png") if f.exists() else self._send(404, "no")
        else:
            self._send(404, "no")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/cancel":
            jid = _current.get("job_id")
            if jid:
                try:
                    fc.cancel(jid, SERVER_BASE)
                except Exception:
                    pass
            self._send(200, "ok")
            return
        if path != "/gen":
            self._send(404, "no")
            return
        n = int(self.headers.get("Content-Length", 0))
        params = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode()).items()}
        try:
            run_gen(params)
        except Exception as e:
            self._send(500, f"<pre>gen error: {html.escape(str(e))}</pre>")
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main():
    print(f"flux2 gallery → http://0.0.0.0:{PORT}  (server={SERVER_BASE}, model={SERVER_MODEL})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
