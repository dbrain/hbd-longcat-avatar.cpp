#!/usr/bin/env bash
set -uo pipefail
docker run --rm \
  -v /home/dbrain/dev/longcat-avatar-mergewt:/src \
  -v longcat-avatar-iter-ccache:/root/.ccache \
  -w /src \
  longcat-avatar-dev:builder bash -lc '
    set -o pipefail
    echo "=== RECONFIGURE (pick up restored layer_registry.cpp) ==="
    cmake -S /src -B /src/build-wtcuda -DCMAKE_BUILD_TYPE=Release \
      -DSD_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="86;120" \
      -DGGML_SAGEATTENTION3=ON 2>&1 | tail -4
    echo "=== BUILD sd-cli + sd-server ==="
    cmake --build /src/build-wtcuda --config Release -j"$(nproc)" \
      --target sd-cli sd-server 2>&1
  '
