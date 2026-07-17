#!/usr/bin/env bash
# gen_flux.sh — generate a test character image via the local FLUX.2 service (sdcpp async API).
# Used to feed the image->rigged-asset pipe with realistic "wild" inputs (not just the miku asset).
#   gen_flux.sh "<prompt>" <out.png> [width=1024] [height=1024] [steps=20] [cfg=4.0] [seed=42]
# Env: FLUX2_URL=http://localhost:8095
set -euo pipefail
PROMPT="${1:?prompt}"; OUT="${2:?out.png}"
W="${3:-1024}"; H="${4:-1024}"; STEPS="${5:-20}"; CFG="${6:-4.0}"; SEED="${7:-42}"
FLUX2_URL="${FLUX2_URL:-http://localhost:8095}"

body=$(python3 -c "
import json,sys
print(json.dumps({'model':'base','prompt':sys.argv[1],'width':int(sys.argv[2]),'height':int(sys.argv[3]),
 'batch_count':1,'seed':int(sys.argv[4]),
 'sample_params':{'sample_steps':int(sys.argv[5]),'guidance':{'txt_cfg':float(sys.argv[6])}}}))
" "$PROMPT" "$W" "$H" "$SEED" "$STEPS" "$CFG")

id=$(curl -s -X POST "$FLUX2_URL/sdcpp/v1/img_gen" -H 'content-type: application/json' -d "$body" \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
echo "[flux2] job $id ($W x $H, $STEPS steps, cfg $CFG, seed $SEED)" >&2

while :; do
  resp=$(curl -s "$FLUX2_URL/sdcpp/v1/jobs/$id")
  st=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status','?'))")
  case "$st" in
    completed) echo "$resp" | python3 -c "
import json,sys,base64
d=json.load(sys.stdin); b=d['result']['images'][0]['b64_json']
open(sys.argv[1],'wb').write(base64.b64decode(b)); print('[flux2] ->',sys.argv[1])
" "$OUT"; break;;
    failed|cancelled) echo "[flux2] job $st: $resp" >&2; exit 1;;
    *) sleep 3;;
  esac
done
