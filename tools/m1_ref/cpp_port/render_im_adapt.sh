#!/usr/bin/env bash
# render_im_adapt.sh — QUEUED GPU (pyrender/EGL) verification renders for the IM curvature-adaptive
# retopo A/B (Track 1 of HANDOFF-D). CPU produced the 6 variants in /tmp/im_adapt/; this is the
# RENDER-AND-LOOK step (owner's GPU job). NO heavy GPU compute — offscreen rasterization only.
#
# The judge: at a FIXED -f budget (final face counts ~equal), do fingers SEPARATE at adaptivity
# 0.5/1.0 where the uniform path (0.0) webs them, with torso density visibly dropping to pay for it?
#   run when the GPU is free:  ./render_im_adapt.sh
set -e
cd "$(dirname "$0")"
VENV=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
OUT=/tmp/im_adapt
Q(){ "$VENV" "$@" 2>&1 | grep -vE "warn|Warning|GLFW|libGL|deprecat" | tail -2; }

for G in d9 d10; do
  echo "### $G — tight hand close-ups (the finger judge)"
  for A in a0 a0.5 a1.0; do
    Q render_hand_zoom.py  $OUT/${G}_${A}_f60k.obj  $OUT/hand_${G}_${A}
  done
  echo "### $G — body + detail geometry (uniform vs full-adaptive)"
  Q render_geo_detail.py  $OUT/${G}_a0_f60k.obj    $OUT/geo_${G}_a0
  Q render_geo_detail.py  $OUT/${G}_a1.0_f60k.obj  $OUT/geo_${G}_a1
done

echo "### montages (adaptivity 0 / 0.5 / 1.0 side by side, same -f budget)"
"$VENV" - <<'PY'
from PIL import Image, ImageDraw
import os
def label(im,txt):
    bar=Image.new("RGB",(im.width,24),(20,20,24)); d=ImageDraw.Draw(bar); d.text((6,5),txt,fill=(235,235,240))
    o=Image.new("RGB",(im.width,im.height+24),(20,20,24)); o.paste(bar,(0,0)); o.paste(im,(0,24)); return o
def hstack(paths_labels,out):
    ims=[label(Image.open(p).convert("RGB"),t) for p,t in paths_labels if os.path.exists(p)]
    if not ims: print("skip",out); return
    H=max(i.height for i in ims); W=sum(i.width for i in ims)
    c=Image.new("RGB",(W,H),(20,20,24)); x=0
    for i in ims: c.paste(i,(x,0)); x+=i.width
    c.save(out); print("wrote",out,c.size)
for G in ("d9","d10"):
    hstack([(f"/tmp/im_adapt/hand_{G}_a0.png","adaptivity 0 (uniform)"),
            (f"/tmp/im_adapt/hand_{G}_a0.5.png","adaptivity 0.5"),
            (f"/tmp/im_adapt/hand_{G}_a1.0.png","adaptivity 1.0")],
           f"/tmp/im_adapt/CMP_hands_{G}.png")
    hstack([(f"/tmp/im_adapt/geo_{G}_a0_body.png","uniform body"),
            (f"/tmp/im_adapt/geo_{G}_a1_body.png","adaptive 1.0 body (torso should be coarser)")],
           f"/tmp/im_adapt/CMP_geo_{G}.png")
PY
echo "### DONE — view /tmp/im_adapt/CMP_hands_d9.png, CMP_hands_d10.png, CMP_geo_d9.png, CMP_geo_d10.png"
echo "###        (also copy onto compare.html as a new 'IM adaptive' tab if useful)"
