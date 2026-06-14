#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# find the max-vram that stays sub-7.5GB while keeping most of the DiT residency win.
for mv in 6 6.5; do
  FR=21 MAXV=$mv LABEL=mv$mv ./run_maxv7.sh 2>&1 | grep -E ">> mv|sampling.*completed|generate_video completed"
done
echo "=== peaks ==="
for mv in 6 6.5; do echo -n "maxv$mv: "; grep ">> mv$mv" /dev/null 2>/dev/null; tail -3 perf_out/mv$mv.log | grep -oE "generate_video completed in [0-9.]+s"; done
