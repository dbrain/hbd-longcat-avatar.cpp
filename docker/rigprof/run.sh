#!/usr/bin/env bash
# docker run wrapper for the rigprof image. Mounts the worktree at /work and the hdd at its REAL path
# (so input/weight CLI paths are unchanged), GPU + SYS_ADMIN (for ncu perf counters). Pass any command.
#   ./run.sh bash docker/rigprof/build_in_container.sh        # build ggml+binary in-container
#   ./run.sh ./skintokens_e2e_docker <args...>                # run (cwd = cpp_port)
#   ./run.sh <ncu-path> --set basic ./skintokens_e2e_docker <args...>   # profile
set -euo pipefail
IMG=rigprof:12.4
exec docker run --rm --gpus all --cap-add SYS_ADMIN \
  -v /home/dbrain/dev/longcat-sparse-spike:/work \
  -v /mnt/hdd/3d/avatar-shootout:/mnt/hdd/3d/avatar-shootout \
  -e PIXAL3D_GGUF_DIR=/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf \
  -w /work/tools/m1_ref/cpp_port \
  "$IMG" "$@"
