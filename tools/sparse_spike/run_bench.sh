#!/usr/bin/env bash
# GPU session, one command: run the Rung-1 CUDA submanifold-conv kernel against the
# real-layer goldens (correctness ~1e-7 expected vs the f64 golden) + per-layer
# timing next to the flex_gemm baseline (golden_model/flexgemm_timing.json).
#
# Uses the toolchain nvcc-built standalone binary (sm_86) — no docker needed for the
# spike bench. (Docker build applies later, when the sparse subsystem is integrated
# into longcat-avatar.cpp proper.)
#
# sm_86 ONLY, deliberately: $T is the CUDA 12.4 toolchain, whose nvcc has no compute_120.
# The pipeline lane (tools/m1_ref/cpp_port/build.sh) builds this same .cu for 86;120 with the
# CUDA 13.3 toolchain — this bench is a 3060-only fixture and is not part of that link.
set -e
T=/mnt/hdd/3d/avatar-shootout/toolchain
cd "$(dirname "$0")"

# rebuild if sources changed (compile is GPU-free; ~seconds)
if [ sparse_subm_conv.cu -nt test_subm_cuda ] || [ test_subm.cpp -nt test_subm_cuda ]; then
  echo "[build] recompiling test_subm_cuda (sm_86)..."
  "$T/bin/nvcc" -O3 -std=c++17 -arch=sm_86 -DUSE_CUDA -ccbin "$T/bin/g++" \
    test_subm.cpp sparse_subm_conv.cu -o test_subm_cuda
fi

echo "[run] CUDA submanifold conv vs goldens + flex_gemm baseline"
LD_LIBRARY_PATH="$T/lib:$LD_LIBRARY_PATH" ./test_subm_cuda "${1:-golden_model}"
