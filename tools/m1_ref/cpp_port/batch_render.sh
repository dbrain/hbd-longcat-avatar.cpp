#!/usr/bin/env bash
# batch_render.sh — all the queued GPU (pyrender/EGL) verification renders in one shot, to run when the
# GPU is free again. Geometry + textured + tight hand close-ups for the d9 vs d10 retopo candidates,
# plus comparison montages. NO heavy GPU compute — just offscreen rasterization.
set -e
cd "$(dirname "$0")"
VENV=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
Q(){ "$VENV" "$@" 2>&1 | grep -vE "warn|Warning|GLFW|libGL|deprecat" | tail -2; }

echo "### geometry (body + hip detail)"
Q render_geo_detail.py /tmp/im_d10_filled.obj      /tmp/geo_d10
Q render_geo_detail.py /tmp/im_mani_d9_filled2.obj /tmp/geo_d9
Q render_geo_detail.py /tmp/qem_d10_120k_f.obj     /tmp/geo_cheap   # ManifoldPlus->IM->QEM 120k (cheap+fingers candidate)

echo "### tight hand close-ups (the finger judge)"
Q render_hand_zoom.py /tmp/im_d10_filled.obj      /tmp/hand_d10
Q render_hand_zoom.py /tmp/im_mani_d9_filled2.obj /tmp/hand_d9
Q render_hand_zoom.py /tmp/qem_d10_120k_f.obj     /tmp/hand_cheap

echo "### textured (insp sidecars)"
[ -f retopo_d10_2048.glb.insp.glb ]      && YAWS=0,40,90 RES=440x620 Q _mv_render.py retopo_d10_2048.glb.insp.glb      /tmp/tex_d10.png
[ -f retopo_manifold_2048.glb.insp.glb ] && YAWS=0,40,90 RES=440x620 Q _mv_render.py retopo_manifold_2048.glb.insp.glb /tmp/tex_d9.png
[ -f retopo_cheap_120k.glb.insp.glb ]    && YAWS=0,40,90 RES=440x620 Q _mv_render.py retopo_cheap_120k.glb.insp.glb    /tmp/tex_cheap.png

echo "### montages"
"$VENV" - <<'PY'
from PIL import Image, ImageDraw
def label(im,txt):
    bar=Image.new("RGB",(im.width,24),(20,20,24)); d=ImageDraw.Draw(bar); d.text((6,5),txt,fill=(235,235,240))
    o=Image.new("RGB",(im.width,im.height+24),(20,20,24)); o.paste(bar,(0,0)); o.paste(im,(0,24)); return o
def vstack(paths_labels,out):
    ims=[label(Image.open(p).convert("RGB"),t) for p,t in paths_labels]
    W=max(i.width for i in ims); H=sum(i.height for i in ims)
    c=Image.new("RGB",(W,H),(20,20,24)); y=0
    for i in ims: c.paste(i,(0,y)); y+=i.height
    c.save(out); print("wrote",out,c.size)
vstack([("/tmp/hand_d9.png","d9 (512^3 manifold) -> IM 100k hands"),
        ("/tmp/hand_d10.png","d10 (1024^3 manifold) -> IM 150k hands")], "/tmp/CMP_hands_d9_d10.png")
vstack([("/tmp/geo_d10_body.png","d10->IM 150k geometry (fingers candidate)")], "/tmp/CMP_geo_d10.png")
import os
if os.path.exists("/tmp/tex_d10.png"):
    vstack([("/tmp/tex_d10.png","NEW d10->IM 150k textured + normal (fingers candidate)"),
            ("/tmp/tex_d9.png","d9->IM 100k textured + normal (prior)")], "/tmp/CMP_tex_d9_d10.png")
PY
echo "### DONE — view /tmp/CMP_hands_d9_d10.png, /tmp/CMP_geo_d10.png, /tmp/CMP_tex_d9_d10.png"
