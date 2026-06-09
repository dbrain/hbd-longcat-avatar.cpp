#!/usr/bin/env bash
# Run the single GPU-slot test for the fp8-folded NAVA DiT GGUF, then compare
# the dumped C++ final audio latent to the Python injected final audio latent.
#
# This script intentionally uses CUDA. Do not run it unless the GPU slot is clear.
set -euo pipefail

ROOT=${ROOT:-/home/dbrain/dev/longcat-avatar.cpp}
cd "$ROOT"

export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

REF_LATENT=${REF_LATENT:-/mnt/hdd/nava/py_inject42_final_audio_latent.bin}
OUT_LATENT=${OUT_LATENT:-/mnt/hdd/nava/cpp_seed42_fp8fold_final_audio_latent.bin}
OUT_NAME=${OUT_NAME:-cpp_peter_fp8fold_latdump_seed42}
REF_WAV=${REF_WAV:-/mnt/hdd/nava/audio_demo/16_python_from_cpp42noise.wav}
OUT_WAV=${OUT_WAV:-/mnt/hdd/nava/audio_demo/22_cpp_fp8fold_render_seed42.wav}
DEMO_ROW_ID=${DEMO_ROW_ID:-22}
DEMO_ROW_LABEL=${DEMO_ROW_LABEL:-cpp fp8-fold q8 render seed42}
LATENT_COMPARE_LOG=${LATENT_COMPARE_LOG:-/mnt/hdd/nava/cpp_seed42_fp8fold_latent_compare.txt}
WAV_COMPARE_LOG=${WAV_COMPARE_LOG:-/mnt/hdd/nava/cpp_seed42_fp8fold_wav_compare.txt}
RENDER_LOG=${RENDER_LOG:-/mnt/hdd/nava/cpp_seed42_fp8fold_render.log}
AUDIO_TRAJ_DIR=${AUDIO_TRAJ_DIR:-/mnt/hdd/nava/cpp_seed42_fp8fold_audio_traj}
GGUF=${GGUF:-models/nava-dit-fp8fold-q8_0.gguf}

export NAVA_DUMP_AUDIO_LATENT="$OUT_LATENT"
export NAVA_DUMP_AUDIO_TRAJ="$AUDIO_TRAJ_DIR"

required=(
  ./build-nava/bin/nava
  "$GGUF"
  /mnt/hdd/nava/cpp-runs/_ref_peter_i2v/bin/context.bin
  /mnt/hdd/nava/vneg_now.bin
  /mnt/hdd/nava/aneg_now.bin
  /mnt/hdd/nava/peter_896x448.bin
  models/wan2.2-vae-48ch-f16.gguf
  models/nava-ltx-audio-vae-f16.gguf
  "$REF_LATENT"
  "$REF_WAV"
)
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "missing required input: $path" >&2
    exit 2
  fi
done

render_cmd=(
  ./build-nava/bin/nava render --cuda
  --gguf "$GGUF" \
  --context /mnt/hdd/nava/cpp-runs/_ref_peter_i2v/bin/context.bin \
  --neg-context /mnt/hdd/nava/vneg_now.bin --audio-neg-context /mnt/hdd/nava/aneg_now.bin \
  --image /mnt/hdd/nava/peter_896x448.bin \
  --vae models/wan2.2-vae-48ch-f16.gguf --audio-vae models/nava-ltx-audio-vae-f16.gguf \
  --steps 10 --frames 13 --width 896 --height 448 --fps 24 \
  --cfg 3.0 --cfg-align 3.0 --cfg-align-audio 2.0 --shift 5.0 --seed 42 \
  --runs-dir /mnt/hdd/nava/cpp-runs --out-name "$OUT_NAME"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'NAVA_DUMP_AUDIO_LATENT=%q\n' "$OUT_LATENT"
  printf 'NAVA_DUMP_AUDIO_TRAJ=%q\n' "$AUDIO_TRAJ_DIR"
  printf 'GGUF=%q\n' "$GGUF"
  printf 'skip render: %q\n' "${SKIP_RENDER:-0}"
  printf 'render command:'
  printf ' %q' "${render_cmd[@]}"
  printf '\n'
  printf 'latent compare: python3 tools/nava_compare_audio_latents.py %q %q\n' "$REF_LATENT" "$OUT_LATENT"
  printf 'extract audio: ffmpeg -y -i %q -vn -c:a pcm_s16le %q\n' "/mnt/hdd/nava/cpp-runs/$OUT_NAME/clip.webm" "$OUT_WAV"
  printf 'audio demo row: python3 tools/nava_audio_demo_add_row.py /mnt/hdd/nava/audio_demo/index.html %q %q --id %q\n' "$OUT_WAV" "$DEMO_ROW_LABEL" "$DEMO_ROW_ID"
  printf 'wav compare: python3 tools/nava_compare_audio_wavs.py %q %q\n' "$REF_WAV" "$OUT_WAV"
  printf 'render log: %q\n' "$RENDER_LOG"
  printf 'audio traj dir: %q\n' "$AUDIO_TRAJ_DIR"
  printf 'latent compare log: %q\n' "$LATENT_COMPARE_LOG"
  printf 'wav compare log: %q\n' "$WAV_COMPARE_LOG"
  exit 0
fi

mkdir -p "$(dirname "$RENDER_LOG")" "$(dirname "$LATENT_COMPARE_LOG")" "$(dirname "$WAV_COMPARE_LOG")" "$(dirname "$OUT_WAV")" "$AUDIO_TRAJ_DIR"

if [ "${SKIP_RENDER:-0}" != "1" ]; then
  "${render_cmd[@]}" 2>&1 | tee "$RENDER_LOG"
else
  if [ ! -e "$OUT_LATENT" ]; then
    echo "SKIP_RENDER=1 but missing dumped latent: $OUT_LATENT" >&2
    exit 2
  fi
  if [ ! -e "/mnt/hdd/nava/cpp-runs/$OUT_NAME/clip.webm" ]; then
    echo "SKIP_RENDER=1 but missing rendered clip: /mnt/hdd/nava/cpp-runs/$OUT_NAME/clip.webm" >&2
    exit 2
  fi
fi

python3 tools/nava_compare_audio_latents.py "$REF_LATENT" "$OUT_LATENT" | tee "$LATENT_COMPARE_LOG"

ffmpeg -y -i "/mnt/hdd/nava/cpp-runs/$OUT_NAME/clip.webm" -vn -c:a pcm_s16le "$OUT_WAV"
python3 tools/nava_audio_demo_add_row.py \
  /mnt/hdd/nava/audio_demo/index.html "$OUT_WAV" \
  "$DEMO_ROW_LABEL" --id "$DEMO_ROW_ID"
python3 tools/nava_compare_audio_wavs.py "$REF_WAV" "$OUT_WAV" | tee "$WAV_COMPARE_LOG"
