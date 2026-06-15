#!/usr/bin/env bash
# build_instant_meshes.sh — bootstrap + build a HEADLESS Instant Meshes batch CLI (field-aligned quad
# retopo, NO GUI/OpenGL/nanogui). QuadriFlow OOM-blows on the dense AI mesh's non-manifold/holey
# topology; Instant Meshes ingests the dense MANIFOLD source directly and emits clean field-aligned
# topology that preserves fingers. Standalone (reads .ply/.obj, writes .ply/.obj) — compiler/ABI
# independent of pixal3d. Same bootstrap-on-demand pattern as build_quadriflow.sh.
#   ./build_instant_meshes.sh           # bootstrap classic TBB 2020 + clone + build
#   QF=$(./build_instant_meshes.sh -p)  # print the binary path (build if needed)
# Deps via pixi (no root, conda-forge): classic TBB 2020.2 (Instant Meshes uses the removed tbb::task /
# task_scheduler_init low-level API → oneTBB won't compile it) + eigen, into the `retopo-deps` env.
# Instant Meshes BSD-style (commercial OK, Modo ships it); TBB Apache-2.0; rply MIT; Eigen MPL2.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IM="$HERE/../../../thirdparty/instant-meshes"
BIN="$IM/build_headless/instant_meshes_batch"
IM_REPO="${IM_REPO:-https://github.com/wjakob/instant-meshes.git}"
DEPS_ENV="${DEPS_ENV:-$HOME/.pixi/envs/retopo-deps}"

[ "${1:-}" = "-p" ] && { [ -x "$BIN" ] || "$0" >&2; echo "$BIN"; exit 0; }

# 1) classic TBB 2020 + eigen (no root)
if [ ! -f "$DEPS_ENV/include/tbb/task_scheduler_init.h" ] \
   || [ ! -f "$DEPS_ENV/include/eigen3/Eigen/Core" ]; then
  echo ">> installing classic tbb 2020.2 + eigen into pixi env retopo-deps"
  pixi global install --environment retopo-deps "tbb-devel==2020.2" "tbb==2020.2" eigen
fi

# 2) clone Instant Meshes (recursive — needs ext/{pcg32,half,dset,pss,rply}) if absent
if [ ! -f "$IM/src/batch.cpp" ]; then
  echo ">> cloning Instant Meshes (recursive)"
  git clone --recursive "$IM_REPO" "$IM"
fi

# 3) compile the GUI-free core + headless main. g++14 promotes -Wchanges-meaning to an error inside
#    TBB's own task.h → -Wno-changes-meaning; -w silences the deprecation spam.
mkdir -p "$IM/build_headless"
CXX="${CXX:-g++}"
INC="-I $IM/src -I $IM/ext/pcg32 -I $IM/ext/half -I $IM/ext/dset -I $IM/ext/pss -I $IM/ext/rply \
     -I $DEPS_ENV/include -I $DEPS_ENV/include/eigen3"
CXXFLAGS="-O3 -std=gnu++17 -fopenmp -w -fpermissive -Wno-changes-meaning $INC"
CORE="meshio normal adjacency meshstats hierarchy extract field bvh subdivide reorder batch \
      smoothcurve cleanup dedge serializer"

echo ">> compiling rply.c"
gcc -O2 -c "$IM/ext/rply/rply.c" -o "$IM/build_headless/rply.o"
OBJS="$IM/build_headless/rply.o"
for f in $CORE; do
  echo ">> compiling $f.cpp"
  $CXX $CXXFLAGS -c "$IM/src/$f.cpp" -o "$IM/build_headless/$f.o"
  OBJS="$OBJS $IM/build_headless/$f.o"
done
echo ">> compiling im_batch_main.cpp + link"
$CXX $CXXFLAGS -c "$HERE/im_batch_main.cpp" -o "$IM/build_headless/im_batch_main.o"
OBJS="$OBJS $IM/build_headless/im_batch_main.o"
$CXX -fopenmp -o "$BIN" $OBJS -L "$DEPS_ENV/lib" -ltbb -Wl,-rpath,"$DEPS_ENV/lib"
echo ">> built $BIN"
