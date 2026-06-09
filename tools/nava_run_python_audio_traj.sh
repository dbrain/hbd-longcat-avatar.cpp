#!/usr/bin/env bash
# Run the Python NAVA reference with C++ seed-42 init noise injected and dump
# per-step audio trajectory tensors for branch-level comparison.
#
# This intentionally uses the GPU through /mnt/hdd/nava/run_nava.sh. Use
# DRY_RUN=1 to print commands without starting the Python render.
set -euo pipefail

ROOT=${ROOT:-/home/dbrain/dev/longcat-avatar.cpp}
NAVA_ROOT=${NAVA_ROOT:-/home/dbrain/dev/NAVA}

NAME=${NAME:-py_inject42_audio_traj}
DATA_FILE=${DATA_FILE:-/mnt/hdd/nava/peter_noclone.jsonl}
CFG=${CFG:-/mnt/hdd/nava/nava_640.yaml}
W=${W:-896}
H=${H:-448}
FRAMES=${FRAMES:-13}
STEPS=${STEPS:-10}
DUR=${DUR:-2.0}
GENFLAGS=${GENFLAGS:---is_i2v}

INJECT_AUDIO=${INJECT_AUDIO:-/mnt/hdd/nava/cpp_traj/aud_noise.bin}
INJECT_VIDEO=${INJECT_VIDEO:-/mnt/hdd/nava/cpp_traj/vid_noise.bin}
PY_AUDIO_TRAJ=${PY_AUDIO_TRAJ:-/mnt/hdd/nava/py_inject42_audio_traj}
PY_FINAL_LATENT=${PY_FINAL_LATENT:-/mnt/hdd/nava/py_inject42_audio_traj_final_audio_latent.bin}
CPP_AUDIO_TRAJ=${CPP_AUDIO_TRAJ:-/mnt/hdd/nava/cpp_seed42_q8_intt_audio_traj}
CPP_FINAL_LATENT=${CPP_FINAL_LATENT:-/mnt/hdd/nava/cpp_seed42_q8_intt_final_audio_latent.bin}
TRAJ_COMPARE_LOG=${TRAJ_COMPARE_LOG:-/mnt/hdd/nava/py_vs_cpp_q8_intt_audio_traj_compare.txt}
LATENT_COMPARE_LOG=${LATENT_COMPARE_LOG:-/mnt/hdd/nava/py_vs_cpp_q8_intt_final_latent_compare.txt}

required=(
  /mnt/hdd/nava/run_nava.sh
  "$NAVA_ROOT/nava_src/pipeline_nava.py"
  "$DATA_FILE"
  "$CFG"
  "$INJECT_AUDIO"
  "$INJECT_VIDEO"
  "$ROOT/tools/nava_compare_audio_traj.py"
  "$ROOT/tools/nava_compare_audio_latents.py"
)
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "missing required input: $path" >&2
    exit 2
  fi
done

run_cmd=(
  bash /mnt/hdd/nava/run_nava.sh
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'cd %q\n' "$NAVA_ROOT"
  printf 'NAVA_INJECT_INIT=1\n'
  printf 'NAVA_INJECT_AUDIO=%q\n' "$INJECT_AUDIO"
  printf 'NAVA_INJECT_VIDEO=%q\n' "$INJECT_VIDEO"
  printf 'NAVA_DUMP_AUDIO_TRAJ=%q\n' "$PY_AUDIO_TRAJ"
  printf 'NAVA_DUMP_AUDIO_LATENT=%q\n' "$PY_FINAL_LATENT"
  printf 'NAME=%q DATA_FILE=%q W=%q H=%q FRAMES=%q STEPS=%q DUR=%q CFG=%q GENFLAGS=%q ' \
    "$NAME" "$DATA_FILE" "$W" "$H" "$FRAMES" "$STEPS" "$DUR" "$CFG" "$GENFLAGS"
  printf 'bash /mnt/hdd/nava/run_nava.sh\n'
  printf 'trajectory compare: python3 %q %q %q | tee %q\n' \
    "$ROOT/tools/nava_compare_audio_traj.py" "$PY_AUDIO_TRAJ" "$CPP_AUDIO_TRAJ" "$TRAJ_COMPARE_LOG"
  printf 'latent compare: python3 %q %q %q | tee %q\n' \
    "$ROOT/tools/nava_compare_audio_latents.py" "$PY_FINAL_LATENT" "$CPP_FINAL_LATENT" "$LATENT_COMPARE_LOG"
  exit 0
fi

mkdir -p "$PY_AUDIO_TRAJ" "$(dirname "$TRAJ_COMPARE_LOG")" "$(dirname "$LATENT_COMPARE_LOG")"

(
  cd "$NAVA_ROOT"
  NAVA_INJECT_INIT=1 \
  NAVA_INJECT_AUDIO="$INJECT_AUDIO" \
  NAVA_INJECT_VIDEO="$INJECT_VIDEO" \
  NAVA_DUMP_AUDIO_TRAJ="$PY_AUDIO_TRAJ" \
  NAVA_DUMP_AUDIO_LATENT="$PY_FINAL_LATENT" \
  NAME="$NAME" DATA_FILE="$DATA_FILE" W="$W" H="$H" FRAMES="$FRAMES" STEPS="$STEPS" DUR="$DUR" CFG="$CFG" GENFLAGS="$GENFLAGS" \
    "${run_cmd[@]}"
)

python3 "$ROOT/tools/nava_compare_audio_traj.py" "$PY_AUDIO_TRAJ" "$CPP_AUDIO_TRAJ" | tee "$TRAJ_COMPARE_LOG"
python3 "$ROOT/tools/nava_compare_audio_latents.py" "$PY_FINAL_LATENT" "$CPP_FINAL_LATENT" | tee "$LATENT_COMPARE_LOG"
