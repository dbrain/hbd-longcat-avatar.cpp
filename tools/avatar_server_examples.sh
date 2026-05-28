#!/usr/bin/env bash
# Example curls for the native LongCat-Avatar HTTP server
# (examples/server -> longcat-avatar-server). Same wire shape as the old
# tools/avatar_server.py supervisor (retired 2026-05-29).
#
# Start the server first (dev):
#   ITER_PORT=8095 kobbler/docker/longcat-avatar-dev/iter.sh restart
# or prod:
#   docker compose up -d longcat-avatar   # :8809
#
# Then:  HOST=localhost:8095 ./tools/avatar_server_examples.sh
set -euo pipefail
HOST="${HOST:-localhost:8095}"
IMG="${IMG:-models/_testinputs/girl_480x832.png}"
WAV="${WAV:-models/_testinputs/speech_16k.wav}"

echo "== /health =="
curl -s "$HOST/health"; echo

IMG_B64="$(base64 -w0 "$IMG")"
WAV_B64="$(base64 -w0 "$WAV")"

echo "== single-clip i2v (25f, 6 steps) -> /tmp/single.webm =="
curl -s "$HOST/generate" -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG_B64\",\"audio\":\"$WAV_B64\",
       \"prompt\":\"a person talking\",\"steps\":6,\"segment_frames\":25,
       \"offload\":false,\"return\":\"bytes\"}" -o /tmp/single.webm
echo "wrote /tmp/single.webm ($(stat -c%s /tmp/single.webm 2>/dev/null || echo ?) bytes)"

echo "== single-clip, milder mouth (audio_mouth_scale 0.7) -> file =="
curl -s "$HOST/generate" -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG_B64\",\"audio\":\"$WAV_B64\",
       \"steps\":6,\"segment_frames\":25,\"audio_mouth_scale\":0.7,\"offload\":false}"; echo

echo "== chained long-form by duration (auto segments, auto-offload) =="
curl -s "$HOST/generate" -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG_B64\",\"audio\":\"$WAV_B64\",
       \"duration_sec\":4,\"segment_frames\":49,\"cont_cond_frames\":13}"; echo

echo "== explicit unload (idle-clear; no-op when idle) =="
curl -s -X POST "$HOST/unload" -d '{}'; echo
