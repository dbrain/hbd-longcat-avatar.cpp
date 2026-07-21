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
if { [ "$BASE" = "geometry_e2e" ] || [ "$BASE" = "pixal3d" ] || [ "$BASE" = "image_to_rig" ] || [ "$BASE" = "texture_mesh_native" ]; } && [ "$MODE" = "cuda" ]; then
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  SPIKE="$HERE/../../sparse_spike"
  BUILD="${PIXAL3D_GGML_BUILD:-$GGML/build-cuda13}"
  LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
  CUDALIBS="-L$TOOLLIB -lcudart -lcublas -L/usr/lib -lcuda"
  TP="$HERE/../../../thirdparty"
  echo ">> CUDA build geometry_e2e (ggml-cuda graphs + spike conv + GPU-resident decode + UV-atlas bake)"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
    -c "$SPIKE/sparse_subm_conv.cu" -o "$HERE/sparse_subm_conv.o"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
    -c "$HERE/svae_cuda.cu" -o "$HERE/svae_cuda.o"
  CUMESH="$HERE/../../../thirdparty/cumesh_native"
  mkdir -p "$CUMESH/build"
  CUMESH_OBJS=""
  for f in cumesh shared geometry connectivity clean_up simplify atlas native_io; do
    "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
      -I"$CUMESH/src" -c "$CUMESH/src/$f.cu" -o "$CUMESH/build/$f.o"
    CUMESH_OBJS="$CUMESH_OBJS $CUMESH/build/$f.o"
  done
  # --pack support: in-process compressed GLB (meshopt geo + KTX2 tex via basis_universal). Build the
  # basisu static lib on demand and link it + the meshopt codec TUs; define PIXAL3D_PACK to compile in
  # the glb_packed.hpp path. basisu KTX2 defines are needed for the pixal3d TU that includes ktx2_encode.hpp.
  BU="$HERE/../../../thirdparty/basis_universal"
  # The texture-only native parity command writes ordinary PNG-in-GLB assets and does not use the
  # optional KTX2 packer.  Avoid rebuilding Basis Universal here: its inherited nproc-wide build can
  # exhaust this shared host while the 3060 run lane is otherwise idle.
  PACK_DEFS=""
  BASISU_LIB=""
  if [ "$BASE" != "texture_mesh_native" ]; then
    "$HERE/build_basisu.sh"
    PACK_DEFS="-DPIXAL3D_PACK -DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0 -DBASISU_SUPPORT_SSE=1 -msse4.1"
    BASISU_LIB="$BU/build/libbasisu_enc.a"
  fi
  # image_to_rig --part-retopo: GPU P3-SAM seg/iou heads (cuBLAS). Compile the kernels + define
  # P3SAM_USE_CUDA so p3sam_postprocess.hpp routes the heads through the GPU (else they run ~30min CPU).
  P3SAM_OBJ=""; P3SAM_DEF=""
  if [ "$BASE" = "image_to_rig" ]; then
    "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
      -c "$HERE/p3sam_heads_cuda.cu" -o "$HERE/p3sam_heads_cuda.o"
    P3SAM_OBJ="$HERE/p3sam_heads_cuda.o"; P3SAM_DEF="-DP3SAM_USE_CUDA"
  fi
  "$HOSTCXX" $COMMON -fopenmp -DM1_USE_CUDA -DM3A_USE_CUDA -DTEXATLAS_NATIVE_CUMESH $P3SAM_DEF $PACK_DEFS $INC -I"$TOOL/include" -I"$CUMESH/src" -I"$BU" \
    "$HERE/$SRC" "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" \
    "$TP/meshoptimizer/vertexcodec.cpp" "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" \
    "$HERE/native_cumesh_bridge.cpp" "$HERE/sparse_subm_conv.o" "$HERE/svae_cuda.o" $CUMESH_OBJS $P3SAM_OBJ \
    $BASISU_LIB -o "$HERE/$BIN" $LIBS $CUDALIBS -lm -lpthread \
    -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
  echo ">> built $BIN"
  exit 0
fi

# GPU-resident decode tests: standalone host port + svae_cuda.cu (dense ops) + spike conv +
# cuBLAS (linear). No ggml. Keeps `feats` resident across the decode.
if { [ "$BASE" = "m4_gpu_test" ] || [ "$BASE" = "m6_gpu_test" ] || [ "$BASE" = "m3a_gpu_test" ] || [ "$BASE" = "decode_gpu_bench" ]; } && [ "$MODE" = "cuda" ]; then
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  SPIKE="$HERE/../../sparse_spike"
  echo ">> CUDA build $SRC (GPU-resident decode: nvcc spike+svae_cuda + g++ host + cublas)"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
    -c "$SPIKE/sparse_subm_conv.cu" -o "$HERE/sparse_subm_conv.o"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
    -c "$HERE/svae_cuda.cu" -o "$HERE/svae_cuda.o"
  "$HOSTCXX" $COMMON -fopenmp -DM3A_USE_CUDA -I"$TOOL/include" "$HERE/$SRC" \
    "$HERE/sparse_subm_conv.o" "$HERE/svae_cuda.o" \
    -o "$HERE/$BIN" -L"$TOOLLIB" -lcudart -lcublas -L/usr/lib -lcuda -lm \
    -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
  echo ">> built $BIN"
  exit 0
fi

# test_p3sam_segment / p3sam_retopo CUDA: GPU heads (cuBLAS + p3sam_heads_cuda.cu) for the seg/iou
# heads (the entire segmentation cost). Host preprocess/sonata/postprocess stay CPU (OpenMP).
if { [ "$BASE" = "test_p3sam_segment" ] || [ "$BASE" = "p3sam_retopo" ]; } && [ "$MODE" = "cuda" ]; then
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  echo ">> CUDA build $SRC (P3-SAM GPU heads: nvcc p3sam_heads_cuda + g++ host -DP3SAM_USE_CUDA + cublas)"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
    -c "$HERE/p3sam_heads_cuda.cu" -o "$HERE/p3sam_heads_cuda.o"
  "$HOSTCXX" -O3 -std=c++17 -march=native -ffast-math -fopenmp -DP3SAM_USE_CUDA -I"$TOOL/include" \
    "$HERE/$SRC" "$HERE/p3sam_heads_cuda.o" \
    -o "$HERE/$BIN" -L"$TOOLLIB" -lcudart -lcublas -L/usr/lib -lcuda -lm \
    -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
  echo ">> built $BIN"
  exit 0
fi

# ktx2_test: standalone validation of the vendored basis_universal KTX2 encoder (ktx2_encode.hpp).
# Builds libbasisu_enc.a on demand and links it. Toolchain g++ (ABI-matches pixal3d's CUDA link).
if [ "$BASE" = "ktx2_test" ]; then
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  BU="$HERE/../../../thirdparty/basis_universal"
  "$HERE/build_basisu.sh"
  echo ">> build ktx2_test (basis_universal KTX2 encoder)"
  "$HOSTCXX" $COMMON -I"$BU" "$HERE/$SRC" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# retopo_bake: bake the cached PBR volume onto a QuadriFlow retopo mesh + meshopt/KTX2 pack. CPU only
# (no cumesh/cuda: precluster=false → real xatlas). Links basisu + meshopt codec; toolchain g++ (ABI).
if [ "$BASE" = "retopo_bake" ]; then
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  BU="$HERE/../../../thirdparty/basis_universal"
  TP="$HERE/../../../thirdparty"
  "$HERE/build_basisu.sh"
  PACK_DEFS="-DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0 -DBASISU_SUPPORT_SSE=1 -msse4.1"
  echo ">> build retopo_bake (tex_atlas bake + meshopt + KTX2, CPU)"
  "$HOSTCXX" $COMMON -fopenmp $PACK_DEFS -I"$BU" "$HERE/$SRC" \
    "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" "$TP/meshoptimizer/vertexcodec.cpp" \
    "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# native_pbr_audit_export: exact native-preprocessed/decimated mesh + saved PBR volume for an
# offline Python-reference oracle. CPU only; never part of the production image-to-rig runbook.
if [ "$BASE" = "native_pbr_audit_export" ]; then
  TP="$HERE/../../../thirdparty"
  echo ">> build native_pbr_audit_export (native same-input reference-audit bridge, CPU only)"
  "${CXX:-/usr/bin/g++}" $COMMON -fopenmp "$HERE/$SRC" \
    "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" "$TP/meshoptimizer/vertexcodec.cpp" \
    "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# glb_pack_test: validate the in-process compressed-GLB writer (glb_packed.hpp = meshopt + KTX2).
# Links libbasisu_enc.a + the meshopt codec TUs (encode/decode vertex+index). Toolchain g++.
if [ "$BASE" = "glb_pack_test" ]; then
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  BU="$HERE/../../../thirdparty/basis_universal"
  TP="$HERE/../../../thirdparty"
  "$HERE/build_basisu.sh"
  echo ">> build glb_pack_test (meshopt + KTX2 packed GLB writer)"
  "$HOSTCXX" $COMMON -I"$BU" "$HERE/$SRC" \
    "$TP/meshoptimizer/vertexcodec.cpp" "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" \
    "$TP/meshoptimizer/quantization.cpp" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# glb_repack: real-scale CPU validation of the in-process packer — reads an uncompressed textured GLB
# and (REPACK_INPROC=hero|small) rewrites it via glb_packed.hpp (meshopt+KTX2). Toolchain g++ + basisu.
if [ "$BASE" = "glb_repack" ]; then
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  BU="$HERE/../../../thirdparty/basis_universal"
  TP="$HERE/../../../thirdparty"
  "$HERE/build_basisu.sh"
  PACK_DEFS="-DREPACK_INPROC_BUILD -DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0 -DBASISU_SUPPORT_SSE=1 -msse4.1"
  echo ">> build glb_repack (+ in-process meshopt+KTX2 repack path)"
  "$HOSTCXX" $COMMON -fopenmp $PACK_DEFS -I"$BU" "$HERE/$SRC" \
    "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" "$TP/meshoptimizer/vertexcodec.cpp" \
    "$TP/meshoptimizer/indexcodec.cpp" "$TP/meshoptimizer/vertexfilter.cpp" "$BU/build/libbasisu_enc.a" \
    -o "$HERE/$BIN" -lm -lpthread
  echo ">> built $BIN"
  exit 0
fi

# moge_fov_test: MoGe-2 recover_focal_shift -> FOV host math (moge_fov.hpp). Header-only + npy reader,
# no ggml/CUDA. Validates against the banked golden points/mask -> fov_result (41.252 deg).
if [ "$BASE" = "moge_fov_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build moge_fov_test (moge_fov.hpp recover_focal_shift LM, no ggml)"
  "$CXX" -O2 -std=c++17 -Wall -Wno-unused-variable "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# make_matte: square black-bg RGB matte from an RGBA source. Header-only stb (load+write), no ggml/CUDA.
if [ "$BASE" = "make_matte" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build make_matte (stb load+write, no ggml)"
  "$CXX" -O2 -std=c++17 "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# dump_to_glb: write a PIXAL3D_DUMP_BAKE binary mesh dump (float verts + int64 faces) out as a GLB. No deps.
if [ "$BASE" = "dump_to_glb" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build dump_to_glb (glb_writer, no ggml)"
  "$CXX" -O2 -std=c++17 "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# ultrashape_mc_test: native marching-cubes surface extractor (ultrashape_mc.hpp) validated vs the
# cubvh golden + GLB write. Pure CPU, no ggml/CUDA — npy.hpp + glb_writer.hpp only.
if [ "$BASE" = "ultrashape_mc_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build ultrashape_mc_test (ultrashape_mc + npy + glb_writer, no ggml)"
  "$CXX" $COMMON "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# shape_subs_test: validate senc::build_guide_subs (tex-decoder guide_subs from mesh occupancy) — grow
# grid-64 back to grid-1024 and check IoU vs the voxelizer. Pure CPU (shape_slat_encoder + sparse_vae + npy).
if [ "$BASE" = "shape_subs_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build shape_subs_test (build_guide_subs structural check, no ggml)"
  "$CXX" $COMMON -fopenmp "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# us_image_proc_test: native ImageProcessorV2 (us_image_proc.hpp) — raw PNG -> proc_image/proc_mask vs
# the cond_full goldens. Pure CPU (stb_image + npy.hpp), no ggml.
if [ "$BASE" = "us_image_proc_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build us_image_proc_test (ImageProcessorV2 cv2 resamplers, stb + npy, no ggml)"
  "$CXX" $COMMON "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# rig_grammar_test: CPU self-test for the SkinTokens/TokenRig "R3" constrained-decode grammar
# (rig_grammar.hpp). Replays the golden emitted sequence and validates the mask at every step.
# Pure CPU logic, header-only, no ggml. Tiny self-contained .npy reader in the .cpp.
if [ "$BASE" = "rig_grammar_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build rig_grammar_test (rig_grammar.hpp grammar state machine, no ggml)"
  "$CXX" -O2 -std=c++17 "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# rig_score: skeleton-quality scorer for a rigged GLB (joint/symmetry/coverage/depth metrics).
# Header-only (reuses glb_reader.hpp JSON parser + read_glb), no ggml. Used for best-of-N + A/B ranking.
if [ "$BASE" = "rig_score" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build rig_score (skeleton-quality scorer, glb_reader.hpp, no ggml)"
  "$CXX" -O2 -std=c++17 -Wall -Wno-unused-variable "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# rig_fsq_test: CPU self-test for the SkinTokens skin-VAE "R4" FSQ.indices_to_codes port
# (rig_fsq.hpp). Maps the golden skin tokens -> vae indices (offset 267) -> codes and compares
# to the Python FSQ reference. Header-only + the dependency-free gguf_reader.hpp (mmap, no ggml).
if [ "$BASE" = "rig_fsq_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build rig_fsq_test (rig_fsq.hpp FSQ index->code, gguf_reader mmap, no ggml)"
  "$CXX" -O2 -std=c++17 "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# voxelizer_test: validate vox::mesh_to_flexible_dual_grid (voxelizer.hpp, the FORWARD o_voxel
# flexible-dual-grid contouring) against the captured golden. Header-only + tiny npy reader in
# the .cpp, OpenMP. No ggml/CUDA — CPU g++.
if [ "$BASE" = "voxelizer_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build voxelizer_test (voxelizer.hpp QEF dual-grid, OpenMP, no ggml)"
  "$CXX" -O2 -std=c++17 -fopenmp "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# voxelizer_e2e: voxelizer -> shape_slat_encoder -> compare vs golden_69k. CPU or CUDA (spike conv).
if [ "$BASE" = "voxelizer_e2e" ]; then
  if [ "$MODE" = "cuda" ]; then
    TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
    SPIKE="$HERE/../../sparse_spike"
    echo ">> CUDA build voxelizer_e2e (nvcc spike+svae_cuda + g++ host -DM3A_USE_CUDA + cublas)"
    "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
      -c "$SPIKE/sparse_subm_conv.cu" -o "$HERE/sparse_subm_conv.o"
    "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
      -c "$HERE/svae_cuda.cu" -o "$HERE/svae_cuda.o"
    "$HOSTCXX" $COMMON -fopenmp -DM3A_USE_CUDA -I"$TOOL/include" "$HERE/$SRC" \
      "$HERE/sparse_subm_conv.o" "$HERE/svae_cuda.o" \
      -o "$HERE/$BIN" -L"$TOOLLIB" -lcudart -lcublas -L/usr/lib -lcuda -lm \
      -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
    echo ">> built $BIN"
    exit 0
  fi
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build voxelizer_e2e (voxelizer + shape_slat_encoder CPU, OpenMP, no ggml)"
  "$CXX" -O2 -std=c++17 -fopenmp "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# glb_reader_test: round-trip self-test for the header-only glTF .glb reader (glb_reader.hpp).
# Writes a cube via glb_writer.hpp, reads it back, asserts verts/faces. No deps.
if [ "$BASE" = "glb_reader_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build glb_reader_test (glb_reader, no ggml)"
  "$CXX" -O2 -std=c++17 "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# glb_rigged_test: self-test for the rigged/skinned .glb writer (glb_rigged.hpp). Loads the
# SkinTokens goldens, writes a skinned GLB (skeleton + JOINTS_0/WEIGHTS_0 + IBM), re-parses it
# (glb_reader.hpp) + independently verifies the skin encoding. Header-only, no ggml/CUDA.
if [ "$BASE" = "glb_rigged_test" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build glb_rigged_test (glb_rigged + glb_reader + glb_writer, no ggml)"
  "$CXX" -O2 -std=c++17 "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# rig_transfer_main: R7 — transfer the SAMPLED rig (joints/parents/skin weights) onto a high-res
# target mesh (glb_reader.hpp) via inverse-distance kNN (rig_transfer.hpp, voxel grid + OpenMP),
# then write a rigged full-mesh GLB (glb_rigged.hpp). Header-only, no ggml/CUDA — CPU g++ + -fopenmp.
if [ "$BASE" = "rig_transfer_main" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build rig_transfer_main (rig_transfer + glb_reader + glb_rigged, OpenMP, no ggml)"
  "$CXX" -O2 -std=c++17 -fopenmp "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# combine_rig_tex_main: FINAL-ASSET GLUE — merge a TEXTURED source mesh (glb_reader.hpp verts/uvs +
# read_glb_basecolor_png image bytes) with a SAMPLED rig (joints/parents/skin) by kNN skin-transfer
# (rig_transfer.hpp) onto the full mesh, then write a TEXTURED + SKINNED GLB (glb_rigged_textured.hpp,
# which reuses glb_rigged.hpp's IBM/top-4 logic). Header-only, no ggml/CUDA — CPU g++ + -fopenmp.
if [ "$BASE" = "combine_rig_tex_main" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build combine_rig_tex_main (combine_rig_tex + glb_reader + glb_rigged_textured + rig_transfer, OpenMP, no ggml)"
  "$CXX" -O2 -std=c++17 -fopenmp "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# per_part_decimate_main: rung-5 GLUE — split a P3-SAM-segmented mesh by per-FACE label, decimate
# each part to a region-adaptive budget by SHELLING OUT to ./obj_decimate (same meshopt+LockBorder
# binary the python ref uses), recombine -> out.glb. glb_reader/glb_writer + self-contained npy/obj
# I/O; header-only, no ggml/CUDA — CPU g++ + -fopenmp. (obj_decimate must be built separately.)
if [ "$BASE" = "per_part_decimate_main" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build per_part_decimate_main (per_part_decimate + glb_reader/writer + npy/obj, OpenMP, no ggml)"
  "$CXX" -O2 -std=c++17 -fopenmp "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# mesh_sample_main: auto-rig PREPROCESSING — load arbitrary GLB (glb_reader.hpp), normalize to
# [-1,1]^3, area-weighted surface-sample N pts+normals, derive R1 query sampled_pc[M] via
# choice N->4M then FPS 4M->M (mesh_sample.hpp). Header-only, no ggml/CUDA — CPU g++ + -fopenmp.
if [ "$BASE" = "mesh_sample_main" ]; then
  CXX="${CXX:-/usr/bin/g++}"
  echo ">> build mesh_sample_main (mesh_sample + glb_reader, OpenMP, no ggml)"
  "$CXX" -O2 -std=c++17 -fopenmp "$HERE/$SRC" -o "$HERE/$BIN" -lm
  echo ">> built $BIN"
  exit 0
fi

# tex_bake_test: UV-atlas bake (xatlas + CPU raster + grid_sample). No ggml, no CUDA — just
# xatlas.cpp + stb_image_write + OpenMP. (grid_sample_test uses the default ggml-linked path.)
if [ "$BASE" = "tex_bake_test" ] || [ "$BASE" = "tex_bake_trellis" ] || [ "$BASE" = "texture_rebake_native" ] || [ "$BASE" = "remesh_test" ] || [ "$BASE" = "tex_bake_dump" ] || [ "$BASE" = "tex_reproject" ] || [ "$BASE" = "texproj_probe" ] || [ "$BASE" = "native_atlas_project" ]; then
  if [ "$MODE" = "cuda" ] && [ "$BASE" = "tex_reproject" ]; then
    TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
    TP="$HERE/../../../thirdparty"
    CUMESH="$HERE/../../../thirdparty/cumesh_native"
    mkdir -p "$CUMESH/build"
    CUMESH_OBJS=""
    for f in cumesh shared geometry connectivity clean_up simplify atlas native_io; do
      "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
        -I"$CUMESH/src" -c "$CUMESH/src/$f.cu" -o "$CUMESH/build/$f.o"
      CUMESH_OBJS="$CUMESH_OBJS $CUMESH/build/$f.o"
    done
    echo ">> build $BASE cuda (xatlas + meshopt + raster + native CuMesh)"
    "$HOSTCXX" $COMMON -fopenmp -DTEXATLAS_NATIVE_CUMESH -I"$TOOL/include" -I"$CUMESH/src" \
      "$HERE/$SRC" "$HERE/native_cumesh_bridge.cpp" "$TP/xatlas.cpp" "$TP/meshoptimizer/simplifier.cpp" \
      $CUMESH_OBJS -o "$HERE/$BIN" -L"$TOOLLIB" -lcudart -L/usr/lib -lcuda -lm -lpthread \
      -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
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
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  SPIKE="$HERE/../../sparse_spike"
  echo ">> CUDA build $SRC (m3a: nvcc spike conv + g++ host)"
  "$TOOL/bin/nvcc" -O2 -std=c++17 -arch=sm_86 -ccbin "$HOSTCXX" \
    -c "$SPIKE/sparse_subm_conv.cu" -o "$HERE/sparse_subm_conv.o"
  "$HOSTCXX" $COMMON -fopenmp -DM3A_USE_CUDA -I"$TOOL/include" "$HERE/$SRC" "$HERE/sparse_subm_conv.o" \
    -o "$HERE/$BIN" -L"$TOOLLIB" -lcudart -L/usr/lib -lcuda -lm \
    -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
  echo ">> built $BIN"
  exit 0
fi

# rig_skin_decoder_test: the SkinTokens skin-VAE "R4" cond-encoder + skin-weight decoder ggml port
# (rig_skin_decoder.hpp). Links ggml (CPU default, or `cuda`); weights load from the packed GGUF via
# m1_ggml.hpp's GGUF mode (env PIXAL3D_GGUF_DIR=<dir-with-skin_vae.gguf>). Reuses rig_fsq.hpp +
# vecset_encoder.hpp (Tripo2 interleaved head-split) + gguf_reader.hpp (FSQ project_out). Same link
# recipe as the generic ggml tests below; this branch documents it explicitly.
if [ "$BASE" = "rig_skin_decoder_test" ]; then
  if [ "$MODE" = "cuda" ]; then
    BUILD="${PIXAL3D_GGML_BUILD:-$GGML/build-cuda13}"; TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
    LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
    CUDALIBS="-L$TOOLLIB -lcudart -lcublas -L/usr/lib -lcuda"
    echo ">> CUDA build rig_skin_decoder_test (ggml-cuda + GGUF skin-VAE decoder)"
    "$HOSTCXX" $COMMON -fopenmp -DM1_USE_CUDA $INC -I"$TOOL/include" "$HERE/$SRC" -o "$HERE/$BIN" \
      $LIBS $CUDALIBS -lm \
      -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
  else
    BUILD="$GGML/build-cpu"; LIBS="-L$BUILD/src -lggml -lggml-base -lggml-cpu"
    CXX="${CXX:-/usr/bin/g++}"
    echo ">> CPU build rig_skin_decoder_test (ggml CPU backend + GGUF skin-VAE decoder)"
    "$CXX" $COMMON -fopenmp $INC "$HERE/$SRC" -o "$HERE/$BIN" $LIBS -lm -Wl,-rpath,"$BUILD/src"
  fi
  echo ">> built $BIN"
  exit 0
fi

# skintokens_e2e: the FULL SkinTokens auto-rigging driver (R1->R3->R5->R4->R6). Single-TU: links
# ggml (CPU default, or `cuda`) for the qwen3 AR core (rig_generate.hpp/qwen3_forward.hpp) + the
# skin-VAE decoder (rig_skin_decoder.hpp); pure-host R5 detok (detok_r5.hpp), FSQ (rig_fsq.hpp),
# GLB writer (glb_rigged.hpp) and the dependency-free gguf_reader.hpp (FSQ project_out). Same link
# recipe as rig_skin_decoder_test / the generic ggml path; this branch documents it explicitly.
if [ "$BASE" = "skintokens_e2e" ]; then
  if [ "$MODE" = "cuda" ]; then
    BUILD="${PIXAL3D_GGML_BUILD:-$GGML/build-cuda13}"; TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
    LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
    CUDALIBS="-L$TOOLLIB -lcudart -lcublas -L/usr/lib -lcuda"
    echo ">> CUDA build skintokens_e2e (ggml-cuda + qwen3 AR core + skin-VAE decoder + GLB)"
    "$HOSTCXX" $COMMON -fopenmp -DM1_USE_CUDA $INC -I"$TOOL/include" "$HERE/$SRC" -o "$HERE/$BIN" \
      $LIBS $CUDALIBS -lm \
      -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
  else
    BUILD="$GGML/build-cpu"; LIBS="-L$BUILD/src -lggml -lggml-base -lggml-cpu"
    CXX="${CXX:-/usr/bin/g++}"
    echo ">> CPU build skintokens_e2e (ggml CPU backend + qwen3 AR core + skin-VAE decoder + GLB)"
    "$CXX" $COMMON -fopenmp $INC "$HERE/$SRC" -o "$HERE/$BIN" $LIBS -lm -Wl,-rpath,"$BUILD/src"
  fi
  echo ">> built $BIN"
  exit 0
fi

# qwen3_decode_test: KV-cache incremental-decode validation (qwen3_decode.hpp on top of the VALIDATED
# qwen3_forward.hpp). Single-TU; same ggml link recipe as the generic path / qwen3_r2_test. Explicit
# branch so the cache-backed decode build is self-documenting.
if [ "$BASE" = "qwen3_decode_test" ]; then
  if [ "$MODE" = "cuda" ]; then
    BUILD="${PIXAL3D_GGML_BUILD:-$GGML/build-cuda13}"; TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
    LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
    CUDALIBS="-L$TOOLLIB -lcudart -lcublas -L/usr/lib -lcuda"
    echo ">> CUDA build qwen3_decode_test (ggml-cuda + qwen3 KV-cache incremental decode)"
    "$HOSTCXX" $COMMON -fopenmp -DM1_USE_CUDA $INC "$HERE/$SRC" -o "$HERE/$BIN" \
      $LIBS $CUDALIBS -lm \
      -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
  else
    BUILD="$GGML/build-cpu"; LIBS="-L$BUILD/src -lggml -lggml-base -lggml-cpu"
    CXX="${CXX:-/usr/bin/g++}"
    echo ">> CPU build qwen3_decode_test (ggml CPU backend + qwen3 KV-cache incremental decode)"
    "$CXX" $COMMON -fopenmp $INC "$HERE/$SRC" -o "$HERE/$BIN" $LIBS -lm -Wl,-rpath,"$BUILD/src"
  fi
  echo ">> built $BIN"
  exit 0
fi

if [ "$MODE" = "cuda" ]; then
  BUILD="${PIXAL3D_GGML_BUILD:-$GGML/build-cuda13}"
  TOOL="${PIXAL3D_CUDA_ROOT:-/mnt/hdd/3d/avatar-shootout/toolchain-cuda13.3}"
  # Host compiler for nvcc. NOT the system gcc: nvcc 13.3 hard-errors above gcc 15 and this box
  # ships gcc 16. gcc15 is the stock Arch package (side-by-side, does not touch the default).
  HOSTCXX="${PIXAL3D_HOST_CXX:-/usr/bin/g++-15}"
  TOOLLIB="$TOOL/lib64"; [ -d "$TOOLLIB" ] || TOOLLIB="$TOOL/lib"
  LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
  CUDALIBS="-L$TOOLLIB -lcudart -lcublas -L/usr/lib -lcuda"
  echo ">> CUDA build $SRC"
  "$HOSTCXX" $COMMON -fopenmp -DM1_USE_CUDA $INC "$HERE/$SRC" -o "$HERE/$BIN" \
    $LIBS $CUDALIBS -lm \
    -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOLLIB" -Wl,-rpath,/usr/lib
else
  BUILD="$GGML/build-cpu"
  LIBS="-L$BUILD/src -lggml -lggml-base -lggml-cpu"
  CXX="${CXX:-/usr/bin/g++}"   # must match the system gcc that built ggml (glibc 2.43)
  echo ">> CPU build $SRC (CXX=$CXX)"
  "$CXX" $COMMON -fopenmp $INC "$HERE/$SRC" -o "$HERE/$BIN" \
    $LIBS -lm -Wl,-rpath,"$BUILD/src"
fi
echo ">> built $BIN"
