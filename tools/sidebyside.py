#!/usr/bin/env python3
"""Build a labelled side-by-side (hstack) comparison mp4 from N webm clips.
Usage: sidebyside.py <out.mp4> <clip1>::<label1> <clip2>::<label2> [...]
Audio taken from the first clip. Labels rendered as a PIL banner (no system TTF needed).
"""
import os, sys, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont

PANEL_W, PANEL_H = 480, 832
BANNER_H = 56

def main():
    out = sys.argv[1]
    specs = sys.argv[2:]
    clips, labels = [], []
    for s in specs:
        clip, label = s.split("::", 1)
        clips.append(clip); labels.append(label)
    n = len(clips)
    total_w = PANEL_W * n

    # build banner PNG
    banner = Image.new("RGBA", (total_w, BANNER_H), (0, 0, 0, 160))
    draw = ImageDraw.Draw(banner)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        # scale default bitmap font up by drawing to a temp and resizing if needed
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        x = i * PANEL_W + (PANEL_W - tw) // 2
        y = (BANNER_H - th) // 2 - bbox[1]
        draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)
    with tempfile.TemporaryDirectory() as td:
        bpath = os.path.join(td, "banner.png")
        banner.save(bpath)
        inputs = []
        for c in clips:
            inputs += ["-i", c]
        inputs += ["-i", bpath]
        # scale each panel, hstack, overlay banner at top
        parts = []
        for i in range(n):
            parts.append(f"[{i}:v]scale={PANEL_W}:{PANEL_H}[p{i}];")
        stack_in = "".join(f"[p{i}]" for i in range(n))
        flt = "".join(parts) + f"{stack_in}hstack=inputs={n}[stacked];[stacked][{n}:v]overlay=0:0[out]"
        cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + [
            "-filter_complex", flt, "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", out]
        subprocess.run(cmd, check=True)
    print(f">> wrote {out}")
    subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=p=0", out])

if __name__ == "__main__":
    main()
