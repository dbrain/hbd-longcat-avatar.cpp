#!/usr/bin/env bash
# build_visioncpp.sh — compile the vendored vision.cpp library plus matte_native.cpp into a single
# static lib (libvisioncpp.a) so the RMBG-2.0 matte runs IN-PROCESS: no vision-server, no
# `docker run vision-cli`, no runtime download.
#
# Vendored (thirdparty/visioncpp) rather than linked as an external CMake project, following the
# cumesh_native / xatlas / meshoptimizer precedent in this tree, because vision.cpp's own CMake
# would pull in a SECOND ggml. The vendored sources compile unmodified against THIS repo's ggml —
# vision.cpp's depend/ggml is the same dbrain/ggml fork, and its pin (d20b816d) is an ancestor of
# ours — which is what lets one process hold one CUDA context for both the matte and the rig.
#
# The result is a static ARCHIVE on purpose: the linker pulls a member only when it resolves an
# undefined symbol, so visp's stb_image / stb_image_write references bind to whatever the calling
# binary already instantiated (image_io.hpp does STB_IMAGE_IMPLEMENTATION), and the stb resize
# member is pulled in only if nothing else provides it. No duplicate-symbol link failures either
# way.
#
#   ./build_visioncpp.sh                    # build/refresh libvisioncpp.a
#   CXX=/usr/bin/g++-15 ./build_visioncpp.sh
#
# C++20 is required (visp uses <format>, <span>, designated init). Callers stay C++17: they only
# ever include matte_native.hpp / matte_native_imgio.hpp, which are C++17-clean.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
VISP="$REPO/thirdparty/visioncpp"
GGML="$REPO/ggml"
# VISP_BUILD_DIR / VISP_EXTRA_CXXFLAGS: the unified avatar_e2e binary links the STATIC ggml from
# build-hymo, which is compiled with -DGGML_MAX_NAME=160 (sd.cpp's CMakeLists adds it). That define
# changes sizeof(ggml_tensor), so a visioncpp built without it and linked against that ggml would
# corrupt memory rather than fail to link. Keep the two variants in SEPARATE build dirs so neither
# can silently pick up the other's objects.
OUTDIR="${VISP_BUILD_DIR:-$VISP/build}"; OUT="$OUTDIR/libvisioncpp.a"
# Must match the final link's compiler/ABI. build.sh's CUDA lane links with g++-15 (nvcc 13.3
# hard-errors above gcc 15 and this box ships gcc 16), and gcc 12 in the CUDA toolchain has no
# usable <format>, so g++-15 it is.
CXX="${CXX:-${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}}"
JOBS="${JOBS:-$(nproc)}"

test -d "$VISP/src/visp" || { echo "build_visioncpp: missing $VISP/src/visp" >&2; exit 1; }
mkdir -p "$OUTDIR"

INC="-I$VISP/include -I$VISP/src -I$REPO/thirdparty -I$GGML/include"
# -w: vendored third-party code, built with the same warning posture as xatlas/basisu here.
CXXFLAGS="-O2 -std=c++20 -fPIC -w ${VISP_EXTRA_CXXFLAGS:-}"

declare -a SRCS OBJS
for f in "$VISP"/src/visp/*.cpp "$VISP"/src/visp/arch/*.cpp "$VISP"/src/visp_stb_resize.cpp \
         "$HERE/matte_native.cpp"; do
  SRCS+=("$f"); OBJS+=("$OUTDIR/$(basename "${f%.cpp}").o")
done

if [ -f "$OUT" ]; then
  newest="$(ls -t "${SRCS[@]}" "$BASH_SOURCE" "$HERE/matte_native.hpp" 2>/dev/null | head -1)"
  if [ "$OUT" -nt "$newest" ]; then echo ">> libvisioncpp.a up to date"; exit 0; fi
fi

echo ">> compiling vendored vision.cpp + matte_native ($CXX, ${#SRCS[@]} TUs, -j$JOBS)"
pids=0
for i in "${!SRCS[@]}"; do
  src="${SRCS[$i]}"; obj="${OBJS[$i]}"
  if [ "$obj" -nt "$src" ] 2>/dev/null && [ "$obj" -nt "$HERE/matte_native.hpp" ]; then continue; fi
  "$CXX" $CXXFLAGS $INC -c "$src" -o "$obj" &
  pids=$((pids+1)); if [ $((pids % JOBS)) -eq 0 ]; then wait; fi
done
wait
echo ">> archiving $OUT"
ar rcs "$OUT" "${OBJS[@]}"
echo ">> built $OUT ($(du -h "$OUT" | cut -f1))"
