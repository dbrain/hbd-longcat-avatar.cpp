#!/usr/bin/env python3
"""Build the image->rigged gallery: one rotatable model-viewer row per character.

Reads the final_e2e_dc_rig.sh run directories, links their artifacts into the puppy-eyetest
inline-3d tree (which already vendors model-viewer 4.3.1) and writes a single page:

    input image | Python Pixal3D ref | native rigged | native rig in motion

FRONT YAW VARIES PER MODEL (miku 180, soldier 0) — the per-model azimuth below is the initial
camera only; every tile is drag-rotatable, so a wrong guess is cosmetic, not a claim.

usage: final_gallery.py --runs <dir with <model>/rigged.glb> [--out <inline-3d dir>]
                        [--page final-e2e.html]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil

SHOOT = "/mnt/hdd/3d/avatar-shootout/_shootout_out"

# model -> (input image, Python reference GLB or None, initial azimuth deg, note)
MODELS: dict[str, tuple[str, str | None, int, str]] = {
    "miku": (f"{SHOOT}/v8_production_e2e_20260724/input_birefnet_rgba.png",
             f"{SHOOT}/puppy-eyetest/python_pixal1024_sdpa.glb", 180, "canonical standing figure"),
    "soldier": (f"{SHOOT}/soldier_matte.png",
                f"{SHOOT}/runbook_image_to_rig/soldier_raw_full_e2e_probe/python_texture_oracle_sdpa.glb",
                0, "bearskin busby = large rigid prop"),
    "gilly": (f"{SHOOT}/runbook_image_to_rig/gilly/gilly_matte.png",
              f"{SHOOT}/runbook_image_to_rig/gilly_modelready_r4_generic_audit/gilly_python_r4_generic_audit.glb",
              0, "generic creature namespace"),
    "char1": (f"{SHOOT}/char1_matte.png", None, 0, "no Python ref"),
    "toy1": (f"{SHOOT}/toy1_matte.png", None, 0, "no Python ref"),
    "toy2": (f"{SHOOT}/toy2_matte.png", None, 0, "no Python ref"),
}

CSS = """
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#15151c;color:#d8dbe8;font:14px system-ui,sans-serif}
header{padding:12px 16px;background:#0e0e14;border-bottom:1px solid #292938;position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0;font-weight:600}
.sub{font-size:12px;color:#9ba2b8;margin-top:3px;max-width:100ch}
section{padding:10px 12px;border-bottom:1px solid #21212c}
section>h2{margin:0 0 8px;font-size:14px;font-weight:600}
section>h2 em{font-style:normal;font-weight:400;font-size:11px;color:#8c94ab;margin-left:8px}
.row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.tile{background:#0e0e14;border:1px solid #292938;border-radius:8px;overflow:hidden;display:grid;grid-template-rows:auto 320px}
.tile.ours{outline:2px solid #5aba7a}
.tile>h3{margin:0;padding:7px 10px;font-size:12px;font-weight:600;border-bottom:1px solid #292938}
.tile>h3 em{display:block;font-style:normal;font-weight:400;font-size:10.5px;color:#8c94ab;margin-top:2px}
model-viewer{width:100%;height:100%;background:radial-gradient(circle at 50% 40%,#3a3c4d,#15151c 68%)}
img.src{width:100%;height:100%;object-fit:contain;background:#000}
.metrics{font:11px ui-monospace,monospace;color:#8c94ab;padding:6px 10px;white-space:pre-wrap}
.miss{display:grid;place-items:center;color:#5c637a;font-size:12px;height:100%}
@media(max-width:1100px){.row{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def tile(title: str, sub: str, body: str, ours: bool = False) -> str:
    cls = "tile ours" if ours else "tile"
    return (f'<div class="{cls}"><h3>{html.escape(title)}<em>{html.escape(sub)}</em></h3>'
            f'{body}</div>')


def viewer(src: str, azimuth: int, autoplay: bool = False) -> str:
    ap = ' autoplay animation-name="rig-exercise"' if autoplay else ""
    return (f'<model-viewer src="{src}" camera-controls touch-action="pan-y" '
            f'shadow-intensity="0.6" exposure="1.0" '
            f'camera-orbit="{azimuth}deg 85deg 2.2m"{ap}></model-viewer>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir containing <model>/rigged.glb")
    ap.add_argument("--out", default="/home/dbrain/dev/puppy-eyetest/inline-3d")
    ap.add_argument("--page", default="final-e2e.html")
    args = ap.parse_args()

    assets = os.path.join(args.out, "final_e2e")
    os.makedirs(assets, exist_ok=True)

    sections = []
    for model, (src_img, pyref, azimuth, note) in MODELS.items():
        run = os.path.join(args.runs, model)
        rigged = os.path.join(run, "rigged.glb")
        if not os.path.exists(rigged):
            continue
        dst = os.path.join(assets, model)
        os.makedirs(dst, exist_ok=True)

        def link(src: str, name: str) -> str | None:
            """Copy artifacts in — the page is served from a different tree, and a symlink to
            /mnt/hdd would 404 through the static server's root check."""
            if not src or not os.path.exists(src):
                return None
            target = os.path.join(dst, name)
            if not os.path.exists(target) or os.path.getmtime(src) > os.path.getmtime(target):
                shutil.copyfile(src, target)
            return f"./final_e2e/{model}/{name}"

        u_img = link(src_img, "input.png")
        u_py = link(pyref, "python_ref.glb")
        u_rig = link(rigged, "rigged.glb")
        u_anim = link(os.path.join(run, "rigged.anim.glb"), "rigged_anim.glb")

        metrics = ""
        mpath = os.path.join(run, "metrics.txt")
        if os.path.exists(mpath):
            with open(mpath) as fh:
                keep = [ln for ln in fh.read().splitlines()
                        if ln.startswith(("wall_total_s", "peak_mib", "texsize"))
                        or "rig_score" in ln or "joints" in ln or "influential" in ln]
            metrics = f'<div class="metrics">{html.escape(" | ".join(keep[:6]))}</div>'

        tiles = [
            tile("input", "matte fed to the pipeline",
                 f'<img class="src" src="{u_img}">' if u_img else '<div class="miss">n/a</div>'),
            tile("Python Pixal3D", "reference target",
                 viewer(u_py, azimuth) if u_py else '<div class="miss">no Python ref</div>'),
            tile("native rigged", "O-Voxel DC + direct bake + SkinTokens rig",
                 viewer(u_rig, azimuth), ours=True),
            tile("native, in motion", "every weighted joint swings in turn",
                 viewer(u_anim, azimuth, autoplay=True) if u_anim
                 else '<div class="miss">no animation</div>', ours=True),
        ]
        sections.append(f'<section><h2>{html.escape(model)}<em>{html.escape(note)}</em></h2>'
                        f'<div class="row">{"".join(tiles)}</div>{metrics}</section>')

    page = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>image -> rigged - final e2e</title>'
        '<script type="module" src="./vendor/model-viewer-4.3.1.min.js"></script>'
        f'<style>{CSS}</style></head><body><header>'
        '<h1>image &rarr; textured &rarr; rigged &rarr; animated &mdash; final e2e</h1>'
        '<div class="sub">One <code>image_to_rig --dc-remesh</code> call per row: smooth O-Voxel '
        'dual-contour geometry, direct volume bake (raw baseColor), SkinTokens skeleton + skin. '
        'Drag any tile to rotate. The fourth tile plays the exercise clip &mdash; watch joints for '
        'tearing, drag, or a limb nothing moves.</div></header>'
        + "".join(sections) + "</body></html>")

    out_page = os.path.join(args.out, args.page)
    with open(out_page, "w") as fh:
        fh.write(page)
    print(f"wrote {out_page} ({len(sections)} models)")
    print(json.dumps({"models": len(sections), "assets": assets}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
