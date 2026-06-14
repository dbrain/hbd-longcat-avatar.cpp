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
  CUMESH="$HERE/../../../thirdparty/cumesh_native"
  mkdir -p "$CUMESH/build"
  CUMESH_OBJS=""
  for f in cumesh shared geometry connectivity clean_up simplify atlas native_io; do
    "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$TOOL/bin/g++" \
      -I"$CUMESH/src" -c "$CUMESH/src/$f.cu" -o "$CUMESH/build/$f.o"
    CUMESH_OBJS="$CUMESH_OBJS $CUMESH/build/$f.o"
  done
  # --pack support: in-process compressed GLB (meshopt geo + KTX2 tex via basis_universal). Build the
  # basisu static lib on demand and link it + the meshopt codec TUs; define PIXAL3D_PACK to compile in
  # the glb_packed.hpp path. basisu KTX2 defines are needed for the pixal3d TU that includes ktx2_encode.hpp.
  BU="$HERE/../../../thirdparty/basis_universal"
  "$HERE/build_basisu.sh"
  PACK_DEFS="-DPIXAL3D_PACK -DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0 -DBASISU_SUPPORT_SSE=1 -msse4.1"
  "$TOOL/bin/g++" $COMMON -fopenmp -DM1_USE_CUDA -DM3A_USE_CUDA -DTEXATLAS_NATIVE_CUMESH $PACK_DEFS $INC -I"$TOOL/include" -I"$CUMESH/src" -I"$BU" \
    "$HERE/$SRC" "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" \
    "$TP/meshoptimizer/vertexcodec.cpp" "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" \
    "$HERE/native_cumesh_bridge.cpp" "$HERE/sparse_subm_conv.o" "$HERE/svae_cuda.o" $CUMESH_OBJS \
    "$BU/build/libbasisu_enc.a" -o "$HERE/$BIN" $LIBS $CUDALIBS -lm -lpthread \
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

# ktx2_test: standalone validation of the vendored basis_universal KTX2 encoder (ktx2_encode.hpp).
# Builds libbasisu_enc.a on demand and links it. Toolchain g++ (ABI-matches pixal3d's CUDA link).
if [ "$BASE" = "ktx2_test" ]; then
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  BU="$HERE/../../../thirdparty/basis_universal"
  "$HERE/build_basisu.sh"
  echo ">> build ktx2_test (basis_universal KTX2 encoder)"
  "$TOOL/bin/g++" $COMMON -I"$BU" "$HERE/$SRC" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# retopo_bake: bake the cached PBR volume onto a QuadriFlow retopo mesh + meshopt/KTX2 pack. CPU only
# (no cumesh/cuda: precluster=false → real xatlas). Links basisu + meshopt codec; toolchain g++ (ABI).
if [ "$BASE" = "retopo_bake" ]; then
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  BU="$HERE/../../../thirdparty/basis_universal"
  TP="$HERE/../../../thirdparty"
  "$HERE/build_basisu.sh"
  PACK_DEFS="-DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0 -DBASISU_SUPPORT_SSE=1 -msse4.1"
  echo ">> build retopo_bake (tex_atlas bake + meshopt + KTX2, CPU)"
  "$TOOL/bin/g++" $COMMON -fopenmp $PACK_DEFS -I"$BU" "$HERE/$SRC" \
    "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" "$TP/meshoptimizer/vertexcodec.cpp" \
    "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# glb_pack_test: validate the in-process compressed-GLB writer (glb_packed.hpp = meshopt + KTX2).
# Links libbasisu_enc.a + the meshopt codec TUs (encode/decode vertex+index). Toolchain g++.
if [ "$BASE" = "glb_pack_test" ]; then
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  BU="$HERE/../../../thirdparty/basis_universal"
  TP="$HERE/../../../thirdparty"
  "$HERE/build_basisu.sh"
  echo ">> build glb_pack_test (meshopt + KTX2 packed GLB writer)"
  "$TOOL/bin/g++" $COMMON -I"$BU" "$HERE/$SRC" \
    "$TP/meshoptimizer/vertexcodec.cpp" "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" \
    "$TP/meshoptimizer/quantization.cpp" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# glb_repack: real-scale CPU validation of the in-process packer — reads an uncompressed textured GLB
# and (REPACK_INPROC=hero|small) rewrites it via glb_packed.hpp (meshopt+KTX2). Toolchain g++ + basisu.
if [ "$BASE" = "glb_repack" ]; then
  TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
  BU="$HERE/../../../thirdparty/basis_universal"
  TP="$HERE/../../../thirdparty"
  "$HERE/build_basisu.sh"
  PACK_DEFS="-DREPACK_INPROC_BUILD -DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0 -DBASISU_SUPPORT_SSE=1 -msse4.1"
  echo ">> build glb_repack (+ in-process meshopt+KTX2 repack path)"
  "$TOOL/bin/g++" $COMMON -fopenmp $PACK_DEFS -I"$BU" "$HERE/$SRC" \
    "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" "$TP/meshoptimizer/vertexcodec.cpp" \
    "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# tex_bake_test: UV-atlas bake (xatlas + CPU raster + grid_sample). No ggml, no CUDA — just
# xatlas.cpp + stb_image_write + OpenMP. (grid_sample_test uses the default ggml-linked path.)
if [ "$BASE" = "tex_bake_test" ] || [ "$BASE" = "remesh_test" ] || [ "$BASE" = "tex_bake_dump" ] || [ "$BASE" = "tex_reproject" ]; then
  if [ "$MODE" = "cuda" ] && [ "$BASE" = "tex_reproject" ]; then
    TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
    TP="$HERE/../../../thirdparty"
    CUMESH="$HERE/../../../thirdparty/cumesh_native"
    mkdir -p "$CUMESH/build"
    CUMESH_OBJS=""
    for f in cumesh shared geometry connectivity clean_up simplify atlas native_io; do
      "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$TOOL/bin/g++" \
        -I"$CUMESH/src" -c "$CUMESH/src/$f.cu" -o "$CUMESH/build/$f.o"
      CUMESH_OBJS="$CUMESH_OBJS $CUMESH/build/$f.o"
    done
    echo ">> build $BASE cuda (xatlas + meshopt + raster + native CuMesh)"
    "$TOOL/bin/g++" $COMMON -fopenmp -DTEXATLAS_NATIVE_CUMESH -I"$TOOL/include" -I"$CUMESH/src" \
      "$HERE/$SRC" "$HERE/native_cumesh_bridge.cpp" "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" \
      $CUMESH_OBJS -o "$HERE/$BIN" -L"$TOOL/lib" -lcudart -L/usr/lib -lcuda -lm -lpthread \
      -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
    echo ">> built $BIN"
    exit 0
  fi
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
