#!/usr/bin/env bash
# Runs INSIDE the rigprof container. Builds ggml (CUDA GRAPHS ON) into a docker-specific build dir
# (persisted in the mounted worktree) + skintokens_e2e_docker + rig_score_docker. Self-consistent with
# the container GLIBC. Re-run is incremental (cmake build dir cached in the worktree).
set -euo pipefail
WORK=/work
GGML=$WORK/ggml
BUILD=$GGML/build-cuda-docker
CP=$WORK/tools/m1_ref/cpp_port
CUDA=/usr/local/cuda

echo ">> [1/3] configure+build ggml (GGML_CUDA_GRAPHS=ON, sm_86, FA on)"
# The nvidia CUDA image prepends /usr/lib/ccache to PATH but ccache isn't installed (dangling wrappers),
# so pin the real /usr/bin compilers explicitly (C/CXX/ASM/CUDA host).
rm -rf "$BUILD"   # clean configure (a prior bad configure leaves the dir unusable)
cmake -S "$GGML" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_ASM_COMPILER=/usr/bin/gcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=ON
cmake --build "$BUILD" -j"$(nproc)"

INC="-I$GGML/include -I$GGML/src -I$CUDA/include"
COMMON="-O2 -std=c++17 -Wall -Wno-unused-variable"

echo ">> [2/3] build skintokens_e2e_docker (links docker ggml + container CUDA)"
cd "$CP"
g++ $COMMON -fopenmp -DM1_USE_CUDA $INC skintokens_e2e.cpp -o skintokens_e2e_docker \
  -L"$BUILD/src" -L"$BUILD/src/ggml-cuda" -lggml -lggml-base -lggml-cpu -lggml-cuda \
  -L"$CUDA/lib64" -lcudart -lcublas -L"$CUDA/lib64/stubs" -lcuda -lm \
  -Wl,-rpath,"$BUILD/src" -Wl,-rpath,"$BUILD/src/ggml-cuda" -Wl,-rpath,"$CUDA/lib64"

echo ">> [3/3] build rig_score_docker (CPU)"
g++ $COMMON $INC rig_score.cpp -o rig_score_docker -lm

echo ">> DONE: $CP/skintokens_e2e_docker + rig_score_docker"
