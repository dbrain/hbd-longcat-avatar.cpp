#!/usr/bin/env python3
"""Lap A/B bench: run the canonical golden gen, report timings/VRAM/md5/PSNR.

Usage: python3 lap_bench.py [label] [--n N] [--prompt P] [--steps S]
Default = canonical golden (knight, seed42, 8step, cfg5, 512) md5 6c0a783425ea.
Also POSTs to gallery :8096 /gen so it lands as a card (unless --no-gallery).
"""
import argparse, hashlib, io, json, statistics, sys, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import flux_client as fc

GOLDEN_MD5 = "6c0a783425ea"
GOLDEN_PNG = __file__.rsplit("/", 2)[0] + "/gallery/goldens/3d52ea6ecf84.png"

def psnr_vs_golden(img_bytes):
    try:
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(GOLDEN_PNG).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"), dtype=np.float64)
        if a.shape != b.shape:
            return None
        mse = float(np.mean((a - b) ** 2))
        return 999.0 if mse == 0 else round(10 * np.log10((255.0 ** 2) / mse), 2)
    except Exception as e:
        return f"err:{e}"

ap = argparse.ArgumentParser()
ap.add_argument("label", nargs="?", default="lap")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--prompt", default="a knight in shining armor standing in a misty forest, cinematic")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--w", type=int, default=512)
ap.add_argument("--h", type=int, default=512)
a = ap.parse_args()

body = fc.build_body(a.prompt, width=a.w, height=a.h, seed=a.seed, steps=a.steps, cfg=a.cfg)
runs = []
md5 = None
for i in range(a.n):
    r = fc.run(body)
    if r["status"] != "completed":
        print(f"  run {i}: FAILED status={r['status']} err={r.get('error')}"); continue
    m = hashlib.md5(r["image_bytes"]).hexdigest()[:12]
    md5 = m
    runs.append(r)
    print(f"  run {i}: wall={r['wall_s']}s cond={r['cond_s']}s dit={r['dit_s']}s "
          f"({r['dit_s_per_step']}/step) vae={r['vae_s']}s peak={r['vram_peak_mib']}MiB md5={m}")

if runs:
    def med(k):
        vals=[x[k] for x in runs if x[k] is not None]
        return round(statistics.median(vals),3) if vals else None
    print(f"\n[{a.label}] n={len(runs)} median: wall={med('wall_s')}s cond={med('cond_s')}s "
          f"dit={med('dit_s')}s ({med('dit_s_per_step')}/step) vae={med('vae_s')}s "
          f"peak={med('vram_peak_mib')}MiB")
    p = psnr_vs_golden(runs[-1]["image_bytes"])
    print(f"  md5={md5}  golden={'MATCH (bit-exact)' if md5==GOLDEN_MD5 else 'DIFFERS'}  PSNR_vs_golden={p} dB")
