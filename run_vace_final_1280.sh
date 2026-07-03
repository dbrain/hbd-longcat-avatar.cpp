#!/usr/bin/env bash
# FINAL fair-fight: VACE @ 1280x704 (LTX-2.3's res) with the full lever stack, ported music-video prompt.
# Config: 4-step DMD, maxv7.3, gray-cache, decode ov0.25. Runs TWICE (run1 warms gray cache; run2 = warm number).
# LTX-2.3 prod = 1280x704, ~50min/27s = ~111 render-sec / sec-of-video. We compute the same metric.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$PWD"; BUILDER="longcat-avatar-dev:builder"; OUT="$REPO/perf_out/final1280"; mkdir -p "$OUT/gcache"
M=/models; DR=/models/_drive; W=1280; H=704; FR="${FR:-21}"; MAXV=7.3; SEED=42; FPS=16
VL=$M/wan22-vace-fun-a14b-low-distill-q4_k.gguf; VH=$M/wan22-vace-fun-a14b-high-distill-q4_k.gguf
VAE=$M/longcat-wan-vae-f16.gguf; UMT5=$M/longcat-umt5-xxl-q8_0.gguf
# LTX music-video prompt ported to Wan2.2 structure (Subject -> Motion[concrete verbs] -> Camera -> Lighting -> Mood -> Quality)
P="A young man with tousled dark brown hair and light stubble in a faded blue denim jacket over a white t-shirt sings energetically into the camera, bobbing his head, mouthing the lyrics, shoulders swaying to an upbeat rock beat. Static locked-off medium shot. Warm amber neon glows behind him outside a corner bar at dusk, the damp street reflecting the lights in soft bokeh. Cinematic, volumetric lighting, shallow depth of field, high detail."
poll(){ local p=0; while kill -0 "$1" 2>/dev/null; do local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$u" ]&&[ "$u" -gt "$p" ]&&p=$u; sleep 0.3; done; echo "$p">"$2"; }
run(){ local tag=$1; local D="$OUT/$tag"; rm -rf "$D"; mkdir -p "$D"; local t0=$(date +%s.%N)
  docker run --rm --gpus all -v "$REPO:/src" -v "$REPO/models:/models" -w /src \
    -e VACE_GRAY_CACHE_DIR=/src/perf_out/final1280/gcache "$BUILDER" \
    /src/build/bin/sd-cli -M vid_gen --vae $VAE --t5xxl $UMT5 --cfg-scale 1 --high-noise-cfg-scale 1 \
    --sampling-method euler --high-noise-sampling-method euler --steps 2 --high-noise-steps 2 \
    --flow-shift 7 -W $W -H $H --video-frames $FR --diffusion-fa --offload-to-cpu --mmap --max-vram $MAXV \
    --vae-tiling --temporal-tiling --vae-relative-tile-size 0.25x0.25 --vae-tile-overlap 0.25 \
    -s $SEED --diffusion-model $VL --high-noise-diffusion-model $VH --init-img $DR/char.png -p "$P" \
    -o /src/perf_out/final1280/$tag/f%03d.png -v > "$D/run.log" 2>&1 &
  local pid=$!; poll $pid "$D/peak.txt" & local pp=$!; wait $pid; local rc=$?; wait $pp 2>/dev/null
  local enc=$(grep "encode_first_stage completed" "$D/run.log"|grep -oE "[0-9.]+ ms|[0-9.]+s"|tail -1)
  local hi=$(grep "sampling(high noise) completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local lo=$(grep -E "stable-diffusion.cpp:[0-9]+ +- sampling completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local dec=$(grep "decode_first_stage completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local gen=$(grep "generate_video completed" "$D/run.log"|grep -oE "[0-9.]+s"|tail -1)
  local wall=$(awk "BEGIN{print $(date +%s.%N)-$t0}")
  printf "  %-6s rc=%s enc=%-8s DiT_hi=%-7s DiT_lo=%-7s dec=%-7s gen=%-8s peak=%sMiB wall=%.1fs\n" \
    "$tag" "$rc" "${enc:-?}" "${hi:-?}" "${lo:-?}" "${dec:-?}" "${gen:-FAIL}" "$(cat $D/peak.txt 2>/dev/null)" "$wall"
}
echo "=== FINAL 1280x704 FR=$FR 4-step maxv7.3 ov0.25 (LTX-res fair fight) ==="
run warm1   # populates gray cache
run warm2   # the warm steady-state number
# mux warm2 to mp4
docker run --rm -v "$REPO:/src" -w /src "$BUILDER" bash -lc "ffmpeg -y -framerate $FPS -i /src/perf_out/final1280/warm2/f%03d.png -c:v libx264 -pix_fmt yuv420p /src/perf_out/final1280/final_1280x704.mp4 -loglevel error" 2>/dev/null
# throughput math
python3 - "$FR" "$FPS" <<'PY' 2>/dev/null || true
import sys,re,glob
fr,fps=int(sys.argv[1]),int(sys.argv[2])
log=open("perf_out/final1280/warm2/run.log").read()
m=re.search(r"generate_video completed in ([0-9.]+)s",log)
if m:
    gen=float(m.group(1)); secvid=fr/fps
    print(f"=== THROUGHPUT: gen={gen:.1f}s for {fr}f@{fps}fps = {secvid:.2f}s video -> {gen/secvid:.0f} render-sec/sec-of-video (LTX-2.3 = 111)")
PY
echo "=== mp4: perf_out/final1280/final_1280x704.mp4 ==="
