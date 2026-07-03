#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export FR=13
echo "### base maxv4.5 pinned0"; LABEL=of_base MAXV=4.5 PINNED=0 OFFLOAD=1 ./perf_a14b.sh 2>&1 | tail -1
echo "### maxv9 pinned0";       LABEL=of_max9 MAXV=9   PINNED=0 OFFLOAD=1 ./perf_a14b.sh 2>&1 | tail -1
echo "### maxv9 pinned1";       LABEL=of_max9pin MAXV=9 PINNED=1 OFFLOAD=1 ./perf_a14b.sh 2>&1 | tail -1
echo "### maxv10.5 pinned1";    LABEL=of_max10pin MAXV=10.5 PINNED=1 OFFLOAD=1 ./perf_a14b.sh 2>&1 | tail -1
echo "=== sweep.csv ==="; column -t -s, perf_out/sweep.csv
