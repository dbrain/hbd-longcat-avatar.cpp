#!/usr/bin/env bash
# Build a Stage-1 port test against the locally-built ggml.
#   ./build.sh proj_grid_test            # CPU build (default)
#   ./build.sh proj_grid_test cuda       # CUDA build (links ggml-cuda; needs ggml/build-cuda)
# Binary is written next to the source (same basename, no extension).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GGML="$HERE/../../../ggml"
BASE="${1%.cpp}"; MODE="${2:-cpu}"
SRC="$BASE.cpp"
BIN="$BASE"

INC="-I$GGML/include -I$GGML/src"
COMMON="-O2 -std=c++17 -Wall -Wno-unused-variable"

# geometry_e2e / pixal3d are the FULL chain: ggml-cuda graphs (DINOv3/NAF/DiT/VAE) + the spike
# CUDA conv (M3a/M4). Links both ggml-cuda and sparse_subm_conv.o; defines M1_USE_CUDA + M3A_USE_CUDA.
if { [ "$BASE" = "geometry_e2e" ] || [ "$BASE" = "pixal3d" ]; } && [ "$MODE" = "cuda" ]; then
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  SPIKE="$HERE/../../sparse_spike"
  BUILD="$GGML/build-cuda"
  LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
  CUDALIBS="-L$TOOL/lib -lcudart -lcublas -L/usr/lib -lcuda"
  TP="$HERE/../../../thirdparty"
  echo ">> CUDA build geometry_e2e (ggml-cuda graphs + spike conv + GPU-resident decode + UV-atlas bake)"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$TOOL/bin/g++" \
    -c "$SPIKE/sparse_subm_conv.cu" -o "$HERE/sparse_subm_conv.o"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$TOOL/bin/g++" \
    -c "$HERE/svae_cuda.cu" -o "$HERE/svae_cuda.o"
  "$TOOL/bin/g++" $COMMON -fopenmp -DM1_USE_CUDA -DM3A_USE_CUDA $INC -I"$TOOL/include" \
    "$HERE/$SRC" "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" \
    "$HERE/sparse_subm_conv.o" "$HERE/svae_cuda.o" -o "$HERE/$BIN" $LIBS $CUDALIBS -lm -lpthread \
    -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
  echo ">> built $BIN"
  exit 0
fi

# GPU-resident decode tests: standalone host port + svae_cuda.cu (dense ops) + spike conv +
# cuBLAS (linear). No ggml. Keeps `feats` resident across the decode.
if { [ "$BASE" = "m4_gpu_test" ] || [ "$BASE" = "m6_gpu_test" ] || [ "$BASE" = "m3a_gpu_test" ] || [ "$BASE" = "decode_gpu_bench" ]; } && [ "$MODE" = "cuda" ]; then
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  SPIKE="$HERE/../../sparse_spike"
  echo ">> CUDA build $SRC (GPU-resident decode: nvcc spike+svae_cuda + g++ host + cublas)"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$TOOL/bin/g++" \
    -c "$SPIKE/sparse_subm_conv.cu" -o "$HERE/sparse_subm_conv.o"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$TOOL/bin/g++" \
    -c "$HERE/svae_cuda.cu" -o "$HERE/svae_cuda.o"
  "$TOOL/bin/g++" $COMMON -fopenmp -DM3A_USE_CUDA -I"$TOOL/include" "$HERE/$SRC" \
    "$HERE/sparse_subm_conv.o" "$HERE/svae_cuda.o" \
    -o "$HERE/$BIN" -L"$TOOL/lib" -lcudart -lcublas -L/usr/lib -lcuda -lm \
    -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
  echo ">> built $BIN"
  exit 0
fi

# tex_bake_test: UV-atlas bake (xatlas + CPU raster + grid_sample). No ggml, no CUDA — just
# xatlas.cpp + stb_image_write + OpenMP. (grid_sample_test uses the default ggml-linked path.)
if [ "$BASE" = "tex_bake_test" ] || [ "$BASE" = "remesh_test" ] || [ "$BASE" = "tex_bake_dump" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  TP="$HERE/../../../thirdparty"
  echo ">> build $BASE (xatlas + meshopt + raster + grid_sample, no ggml)"
  "$CXX" $COMMON -fopenmp "$HERE/$SRC" "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# m3a/m4 are standalone host ports (no ggml) that route the conv through the spike CUDA
# kernel: compile sparse_subm_conv.cu and link cudart only.
if { [ "$BASE" = "m3a_upsample" ] || [ "$BASE" = "m4_mesh" ] || [ "$BASE" = "m6_tex_decode_test" ] || [ "$BASE" = "m4_profile" ]; } && [ "$MODE" = "cuda" ]; then
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  SPIKE="$HERE/../../sparse_spike"
  echo ">> CUDA build $SRC (m3a: nvcc spike conv + g++ host)"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$TOOL/bin/g++" \
    -c "$SPIKE/sparse_subm_conv.cu" -o "$HERE/sparse_subm_conv.o"
  "$TOOL/bin/g++" $COMMON -fopenmp -DM3A_USE_CUDA -I"$TOOL/include" "$HERE/$SRC" "$HERE/sparse_subm_conv.o" \
    -o "$HERE/$BIN" -L"$TOOL/lib" -lcudart -L/usr/lib -lcuda -lm \
    -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
  echo ">> built $BIN"
  exit 0
fi

if [ "$MODE" = "cuda" ]; then
  BUILD="$GGML/build-cuda"
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
  CUDALIBS="-L$TOOL/lib -lcudart -lcublas -L/usr/lib -lcuda"
  echo ">> CUDA build $SRC"
  "$TOOL/bin/g++" $COMMON -fopenmp -DM1_USE_CUDA $INC "$HERE/$SRC" -o "$HERE/$BIN" \
    $LIBS $CUDALIBS -lm \
    -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
else
  BUILD="$GGML/build-cpu"
  LIBS="-L$BUILD/src -lggml -lggml-base -lggml-cpu"
  CXX="${CXX:-/usr/bin/g++}"   # must match the system gcc that built ggml (glibc 2.43)
  echo ">> CPU build $SRC (CXX=$CXX)"
  "$CXX" $COMMON -fopenmp $INC "$HERE/$SRC" -o "$HERE/$BIN" \
    $LIBS -lm -Wl,-rpath,"$BUILD/src"
fi
echo ">> built $BIN"
