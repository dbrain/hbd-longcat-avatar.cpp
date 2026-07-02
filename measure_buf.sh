#!/usr/bin/env bash
# Measure the DiT block compute_buf (the 8.1GB peak) at 65f@1280 under various env toggles.
# Each run loads + encodes then prints "compute buffer size: XXXX MB" at the first DiT block
# reserve, then OOMs on the cross-segment cache (that's fine — we only want the buffer number).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PROMPT="the same person keeps dancing energetically on the neon-lit city street at night, cinematic"
run() {
  local tag="$1"; shift
  echo "### $tag :: env: $*"
  env "$@" FR=65 W=1280 H=704 bash ./iter_seg2.sh "$PROMPT" mbuf_$tag >/dev/null 2>&1
  local bufs; bufs=$(grep -a "compute buffer size" "cont_mbuf_$tag/log" 2>/dev/null | grep -a "VACE" | sed -E 's/.*size: ([0-9.]+) MB.*/\1/' | sort -n | tail -1)
  local vace_reserve; vace_reserve=$(grep -a "\[VRAM\] Wan2.x-VACE.*reserve" "cont_mbuf_$tag/log" 2>/dev/null | tail -1)
  echo "    max DiT compute_buf = ${bufs:-<none>} MB"
  echo "    $vace_reserve"
  rm -rf "cont_mbuf_$tag"
}
run baseline
run f16out   GGML_CUDNN_ATTN_F16_OUT=1
run ffn2048  FFNTILE=2048
run ffn1024  FFNTILE=1024
run f16mod   WAN_DIT_F16_MOD=1
run combo    GGML_CUDNN_ATTN_F16_OUT=1 WAN_DIT_F16_MOD=1 FFNTILE=2048
