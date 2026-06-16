#!/usr/bin/env bash
# 1536-vs-1024 head-to-head for the C++ pixal3d port (--resolution support).
# Generates the 1536 mesh (the new finer-lattice path) and renders hand close-ups + a full-body
# silhouette montage for BOTH resolutions, so finger separation / staircasing can be eyeballed.
#   ./run_1536_compare.sh [input.png]
# Outputs land in $OUT (on /mnt/hdd). 1024 is assumed already generated (regression baseline);
# pass REGEN=1 to (re)generate it too.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"   # cpp_port (binary + weights live here)
cd "$HERE"
PY=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python    # has trimesh + pyrender (EGL)
OUT=/mnt/hdd/3d/avatar-shootout/_shootout_out
IMG="${1:-prep_test_matte.png}"
mkdir -p "$OUT"

vram() { sort -n "$1" 2>/dev/null | tail -1; }
run_res() {  # $1 = resolution
  local R="$1"; local GLB="$OUT/miku_res${R}.glb"
  if [ "$R" = "1024" ] && [ "${REGEN:-0}" = "0" ] && [ -f "$GLB" ]; then
    echo "[gen $R] exists, skip (REGEN=1 to force)"; return; fi
  echo "=== generate $R ==="
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$OUT/vram_${R}.csv" 2>/dev/null &
  local SMP=$! S=$(date +%s)
  ./pixal3d --model weights_gguf --image "$IMG" --out "$GLB" --resolution "$R" --ply 2>&1 | tee "$OUT/gen_${R}.log" | grep -iE "\[[0-9]\]|grid|res|verts|faces|GLB|error|assert"
  kill $SMP 2>/dev/null
  echo "[gen $R] wall=$(( $(date +%s) - S ))s peakVRAM=$(vram "$OUT/vram_${R}.csv")MiB  -> $GLB"
}

run_res 1024
run_res 1536

echo "=== render hand close-ups + silhouette (both resolutions) ==="
for R in 1024 1536; do
  PLY="$OUT/miku_res${R}.ply"   # PLY = single clean mesh (GLB loads as a multi-geom trimesh Scene)
  [ -f "$PLY" ] || { echo "missing $PLY, skip render"; continue; }
  "$PY" render_hand_zoom.py "$PLY" "$OUT/hands_res${R}" 2>&1 | grep -iE "loaded|verts|error" || true
  "$PY" render_mesh.py "$PLY" "$OUT/body_res${R}.png" 2>&1 | grep -iE "loaded|verts|error" || true
done

echo "=== compose side-by-side (1024 | 1536) ==="
"$PY" - "$OUT" <<'PYEOF'
import sys, os
from PIL import Image, ImageDraw, ImageFont
OUT = sys.argv[1]
def grab(name):
    p = os.path.join(OUT, name)
    return Image.open(p).convert("RGB") if os.path.exists(p) else None
for kind in ("hands", "body"):
    a = grab(f"{kind}_res1024.png"); b = grab(f"{kind}_res1536.png")
    if a is None or b is None:
        print(f"[compose] skip {kind}: missing png"); continue
    h = max(a.height, b.height)
    a = a.resize((int(a.width*h/a.height), h)); b = b.resize((int(b.width*h/b.height), h))
    cv = Image.new("RGB", (a.width+b.width, h+28), (18,18,20))
    cv.paste(a,(0,28)); cv.paste(b,(a.width,28))
    d = ImageDraw.Draw(cv)
    d.text((6,6), "1024 (baseline)", fill=(235,235,235))
    d.text((a.width+6,6), "1536 (--resolution 1536)", fill=(235,235,235))
    op = os.path.join(OUT, f"CMP_{kind}_1024_vs_1536.png"); cv.save(op); print("[compose] wrote", op)
PYEOF
echo "=== DONE. Compare images: $OUT/CMP_hands_1024_vs_1536.png  $OUT/CMP_body_1024_vs_1536.png ==="
