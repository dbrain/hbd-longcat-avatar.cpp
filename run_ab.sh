#!/usr/bin/env bash
# Sequential LTX-2.3 prod-recipe A/B over the cuDNN borrows. One GPU job at a time.
set -uo pipefail
cd /home/dbrain/dev/longcat-avatar.cpp
mkdir -p ltx_ab_out
export FRAMES="${FRAMES:-97}" W="${W:-1280}" H="${H:-704}" STEPS="${STEPS:-8}" SEED="${SEED:-42}" MAX_VRAM="${MAX_VRAM:-10.5}"

run() { # label  env
  echo "############################################################"
  echo "## RUN $1  env='$2'  $(date +%T)"
  LABEL="$1" EXTRA_ENV="$2" bash ltx_ab.sh > "ltx_ab_out/run_$1.log" 2>&1
  grep -E "RESULT\|" "ltx_ab_out/run_$1.log" | tail -1
}

run ab_baseline ""
run ab_attn     "GGML_CUDNN_ATTN=1"
run ab_conv3d   "GGML_CUDNN_CONV3D=1"
run ab_all      "GGML_CUDNN_ATTN=1 GGML_CUDNN_CONV3D=1"

echo "============== A/B SUMMARY =============="
grep -hE "RESULT\|" ltx_ab_out/run_ab_*.log | sed -E 's/.*RESULT/RESULT/'
