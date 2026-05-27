#!/usr/bin/env python3
"""
Tiny zero-dependency webserver to EYEBALL generated avatar clips.

The original PyTorch agent's Gradio app (proof_app.py :7860) needs a venv that's
been deleted, so this is the lightweight replacement: serves a directory of clips
with an auto-refreshing index that embeds the newest .webm/.mp4 in a <video> tag.

    python3 tools/serve_clips.py [--dir models] [--port 8011]

Then open http://10.0.0.208:8011/ (binds 0.0.0.0). The page auto-reloads every
10s and shows the most-recent clip on top, so you can leave it open while a run
renders and just watch it appear.
"""
import argparse, html, http.server, os, socketserver, urllib.parse
from pathlib import Path

VID_EXT = (".webm", ".mp4", ".avi", ".webp", ".gif")


def build_index(root: Path) -> bytes:
    clips = sorted(
        (p for p in root.rglob("*") if p.suffix.lower() in VID_EXT),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    rows = []
    for p in clips:
        rel = urllib.parse.quote(str(p.relative_to(root)))
        mtime = __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime(p.stat().st_mtime))
        size = p.stat().st_size / 1e6
        tag = (f'<video src="/{rel}" controls autoplay loop muted '
               f'style="max-width:480px;border:1px solid #444"></video>'
               if p.suffix.lower() != ".gif" else f'<img src="/{rel}" style="max-width:480px">')
        rows.append(f'<div style="margin:18px 0"><div style="font:13px monospace;color:#aaa">'
                    f'{html.escape(str(p.relative_to(root)))} &middot; {mtime} &middot; {size:.2f} MB</div>{tag}</div>')
    body = "\n".join(rows) or '<p style="color:#aaa">no clips yet — render one and it will appear here.</p>'
    return (f'<!doctype html><meta charset=utf-8>'
            f'<title>LongCat-Avatar clips</title>'
            f'<body style="background:#111;color:#ddd;font-family:system-ui;margin:24px">'
            f'<h2>LongCat-Video-Avatar — clips</h2>'
            f'<p style="color:#777;font:12px monospace">manual reload only (auto-refresh disabled)</p>'
            f'{body}</body>').encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    root = Path(args.dir).resolve()

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = build_index(root)
                self.send_response(200); self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page))); self.end_headers()
                self.wfile.write(page); return
            super().do_GET()
        def log_message(self, *a):  # quiet
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", args.port), H) as httpd:
        print(f"serving {root} at http://0.0.0.0:{args.port}/  (http://10.0.0.208:{args.port}/)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
