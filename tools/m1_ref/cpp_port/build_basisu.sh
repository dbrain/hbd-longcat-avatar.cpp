#!/usr/bin/env bash
# build_basisu.sh — compile the vendored basis_universal encoder into a static lib (libbasisu_enc.a)
# for in-process KTX2 encoding (ktx2_encode.hpp). Idempotent: skips if the .a is newer than all sources.
# Built with the TOOLCHAIN g++ (conda gcc 12.4) so its libstdc++ ABI matches the pixal3d CUDA link.
#   ./build_basisu.sh                 # build/refresh libbasisu_enc.a
#   CXX=/usr/bin/g++ ./build_basisu.sh   # override compiler (must match the final link's ABI)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BU="$HERE/../../../thirdparty/basis_universal"
OUTDIR="$BU/build"; OUT="$OUTDIR/libbasisu_enc.a"
CXX="${CXX:-/mnt/hdd/3d/avatar-shootout/toolchain/bin/g++}"
JOBS="${JOBS:-$(nproc)}"

# Bootstrap the vendored encoder if absent (the dir is gitignored — too big to commit wholesale).
BASISU_REPO="${BASISU_REPO:-https://github.com/BinomialLLC/basis_universal.git}"
BASISU_REF="${BASISU_REF:-1aab02b}"
if [ ! -f "$BU/encoder/basisu_comp.h" ]; then
  echo ">> basis_universal not found — cloning @ $BASISU_REF"
  git clone "$BASISU_REPO" "$BU"
  ( cd "$BU" && git checkout -q "$BASISU_REF" )
fi
mkdir -p "$OUTDIR"

DEFS="-DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0 \
      -DBASISU_SUPPORT_SSE=1 -DBASISU_DISABLE_ANDROID_ASTC_DECOMP=1"
CXXFLAGS="-O2 -std=c++17 -fPIC -msse4.1 -w $DEFS"
CFLAGS="-O2 -fPIC -w"

ENC_SRC=(basisu_backend basisu_basis_file basisu_comp basisu_enc basisu_etc basisu_frontend \
         basisu_gpu_texture basisu_pvrtc1_4 basisu_resampler basisu_resample_filters basisu_ssim \
         basisu_uastc_enc basisu_bc7enc jpgd basisu_kernels_sse basisu_opencl pvpngreader \
         basisu_uastc_hdr_4x4_enc basisu_astc_hdr_6x6_enc basisu_astc_hdr_common \
         basisu_astc_ldr_common basisu_astc_ldr_encode basisu_tinyexr)

# Collect (src -> obj) pairs.
declare -a SRCS OBJS
for f in "${ENC_SRC[@]}"; do SRCS+=("$BU/encoder/$f.cpp");      OBJS+=("$OUTDIR/$f.o"); done
SRCS+=("$BU/transcoder/basisu_transcoder.cpp"); OBJS+=("$OUTDIR/basisu_transcoder.o")
SRCS+=("$BU/zstd/zstd.c");                       OBJS+=("$OUTDIR/zstd.o")

# Up-to-date check: .a newer than every source + this script + the header.
if [ -f "$OUT" ]; then
  newest_src="$(ls -t "${SRCS[@]}" "$BASH_SOURCE" "$HERE/ktx2_encode.hpp" 2>/dev/null | head -1)"
  if [ "$OUT" -nt "$newest_src" ]; then echo ">> libbasisu_enc.a up to date"; exit 0; fi
fi

echo ">> compiling basis_universal encoder ($CXX, ${#SRCS[@]} TUs, -j$JOBS)"
pids=0
for i in "${!SRCS[@]}"; do
  src="${SRCS[$i]}"; obj="${OBJS[$i]}"
  if [ "$obj" -nt "$src" ] 2>/dev/null; then continue; fi
  case "$src" in
    *.c)   "$CXX" -x c   $CFLAGS   -c "$src" -o "$obj" & ;;
    *.cpp) "$CXX"        $CXXFLAGS -c "$src" -o "$obj" & ;;
  esac
  pids=$((pids+1)); if [ $((pids % JOBS)) -eq 0 ]; then wait; fi
done
wait
echo ">> archiving $OUT"
ar rcs "$OUT" "${OBJS[@]}"
echo ">> built $OUT ($(du -h "$OUT" | cut -f1))"
