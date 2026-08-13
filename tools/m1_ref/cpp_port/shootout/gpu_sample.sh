#!/usr/bin/env bash
# Sample 3060 VRAM/util into a CSV while a run is in flight, then report peak.
#   gpu_sample.sh start <csv>   -> writes <csv>, prints sampler PID
#   gpu_sample.sh stop  <pid>   -> stops the sampler
#   gpu_sample.sh peak  <csv>   -> prints "peak_mib=<n> mean_util=<n>"
set -euo pipefail
GPU3060="${GPU_3060_UUID:-GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6}"
case "${1:?start|stop|peak}" in
  start)
    CSV="${2:?csv path}"
    echo "ts,mem_used_mib,util_pct" > "$CSV"
    (
      while :; do
        printf '%s,%s\n' "$(date +%s)" \
          "$(nvidia-smi --id="$GPU3060" --query-gpu=memory.used,utilization.gpu \
             --format=csv,noheader,nounits | tr -d ' ')"
        sleep 2
      done
    ) >> "$CSV" 2>/dev/null &
    echo $!
    ;;
  stop) kill "${2:?pid}" 2>/dev/null || true ;;
  peak)
    CSV="${2:?csv path}"
    awk -F, 'NR>1 && $2+0>m {m=$2+0} NR>1 {u+=$3+0; n++} END {printf "peak_mib=%d mean_util=%.0f samples=%d\n", m, (n?u/n:0), n}' "$CSV"
    ;;
esac
