#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export FR=21
echo "### fr21 base maxv4.5"; LABEL=of21_base MAXV=4.5 PINNED=0 OFFLOAD=1 ./perf_a14b.sh 2>&1 | tail -1
echo "### fr21 maxv7";        LABEL=of21_max7 MAXV=7 PINNED=0 OFFLOAD=1 ./perf_a14b.sh 2>&1 | tail -1
echo "### fr21 maxv9";        LABEL=of21_max9 MAXV=9 PINNED=0 OFFLOAD=1 ./perf_a14b.sh 2>&1 | tail -1
echo "=== rows ==="; grep '^of21' perf_out/sweep.csv | column -t -s,
