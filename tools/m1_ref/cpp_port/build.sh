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
    TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
    SPIKE="$HERE/../../sparse_spike"
    echo ">> CUDA build voxelizer_e2e (nvcc spike+svae_cuda + g++ host -DM3A_USE_CUDA + cublas)"
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

# rig_skin_decoder_test: the SkinTokens skin-VAE "R4" cond-encoder + skin-weight decoder ggml port
# (rig_skin_decoder.hpp). Links ggml (CPU default, or `cuda`); weights load from the packed GGUF via
# m1_ggml.hpp's GGUF mode (env PIXAL3D_GGUF_DIR=<dir-with-skin_vae.gguf>). Reuses rig_fsq.hpp +
# vecset_encoder.hpp (Tripo2 interleaved head-split) + gguf_reader.hpp (FSQ project_out). Same link
# recipe as the generic ggml tests below; this branch documents it explicitly.
if [ "$BASE" = "rig_skin_decoder_test" ]; then
  if [ "$MODE" = "cuda" ]; then
    BUILD="$GGML/build-cuda"; TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
    LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
    CUDALIBS="-L$TOOL/lib -lcudart -lcublas -L/usr/lib -lcuda"
    echo ">> CUDA build rig_skin_decoder_test (ggml-cuda + GGUF skin-VAE decoder)"
    "$TOOL/bin/g++" $COMMON -fopenmp -DM1_USE_CUDA $INC "$HERE/$SRC" -o "$HERE/$BIN" \
      $LIBS $CUDALIBS -lm \
      -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
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
    BUILD="$GGML/build-cuda"; TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
    LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
    CUDALIBS="-L$TOOL/lib -lcudart -lcublas -L/usr/lib -lcuda"
    echo ">> CUDA build skintokens_e2e (ggml-cuda + qwen3 AR core + skin-VAE decoder + GLB)"
    "$TOOL/bin/g++" $COMMON -fopenmp -DM1_USE_CUDA $INC "$HERE/$SRC" -o "$HERE/$BIN" \
      $LIBS $CUDALIBS -lm \
      -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
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
    BUILD="$GGML/build-cuda"; TOOL=/mnt/hdd/3d/avatar-shootout/toolchain
    LIBS="-L$BUILD/src -L$BUILD/src/ggml-cuda -lggml -lggml-base -lggml-cpu -lggml-cuda"
    CUDALIBS="-L$TOOL/lib -lcudart -lcublas -L/usr/lib -lcuda"
    echo ">> CUDA build qwen3_decode_test (ggml-cuda + qwen3 KV-cache incremental decode)"
    "$TOOL/bin/g++" $COMMON -fopenmp -DM1_USE_CUDA $INC "$HERE/$SRC" -o "$HERE/$BIN" \
      $LIBS $CUDALIBS -lm \
      -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$TOOL/lib" -Wl,-rpath,/usr/lib
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
