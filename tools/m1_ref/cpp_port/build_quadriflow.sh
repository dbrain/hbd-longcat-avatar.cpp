#!/usr/bin/env bash
# build_quadriflow.sh — bootstrap + build the QuadriFlow headless quad-retopo CLI (HANDOFF-C / R0).
# Idempotent: skips if the binary is newer than the sources. The vendored repo is gitignored (cloned
# on demand, pinned QF_REF) — same pattern as build_basisu.sh.
#
#   ./build_quadriflow.sh           # bootstrap deps + clone + build -> thirdparty/QuadriFlow/build/quadriflow
#   QF=$(./build_quadriflow.sh -p)  # print the binary path (build if needed)
#
# Deps via pixi (no root; conda-forge): eigen + libboost-devel into a `retopo-deps` global env.
# QuadriFlow MIT; Boost BSL-1.0; lemon BSL-1.0; Eigen MPL2 — all commercial-clean. Standalone CLI
# (reads OBJ, writes quad OBJ) so its compiler/ABI is independent of pixal3d — no toolchain pinning.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QF="$HERE/../../../thirdparty/QuadriFlow"
BIN="$QF/build/quadriflow"
QF_REPO="${QF_REPO:-https://github.com/hjwdzh/QuadriFlow.git}"
QF_REF="${QF_REF:-810b7a0}"
DEPS_ENV="${DEPS_ENV:-$HOME/.pixi/envs/retopo-deps}"

[ "${1:-}" = "-p" ] && { [ -x "$BIN" ] || "$0" >&2; echo "$BIN"; exit 0; }

if [ -x "$BIN" ] && [ "$BIN" -nt "$QF/CMakeLists.txt" ]; then echo ">> quadriflow up to date: $BIN"; exit 0; fi

# 1) deps (eigen headers + boost graph headers), no root
if [ ! -f "$DEPS_ENV/include/eigen3/signature_of_eigen3_matrix_library" ] \
   || [ ! -f "$DEPS_ENV/include/boost/graph/boykov_kolmogorov_max_flow.hpp" ]; then
  echo ">> installing eigen + libboost-devel into pixi env retopo-deps"
  pixi global install --environment retopo-deps eigen libboost-devel
fi

# 2) clone QuadriFlow if absent (gitignored; bootstrapped here)
if [ ! -f "$QF/CMakeLists.txt" ]; then
  echo ">> cloning QuadriFlow @ $QF_REF"
  git clone --recursive "$QF_REPO" "$QF"
  ( cd "$QF" && git checkout -q "$QF_REF" )
fi

# 3) configure + build (cmake 4.x needs the policy-min shim for QF's old VERSION 3.1)
mkdir -p "$QF/build"
( cd "$QF/build" && cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH="$DEPS_ENV" \
    -DEIGEN_INCLUDE_DIR_HINTS="$DEPS_ENV/include/eigen3" \
    -DBoost_INCLUDE_DIR="$DEPS_ENV/include" \
    -DBUILD_OPENMP=ON >/dev/null && ninja )
echo ">> built $BIN"
