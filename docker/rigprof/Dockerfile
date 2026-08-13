# Reusable build/run/profile image for the C++ SkinTokens rigger (skintokens_e2e).
# WHY: the host is CachyOS (GLIBC 2.43) so host-built binaries won't run in stock containers — and the
# host ggml is built GGML_CUDA_GRAPHS=OFF. This image builds ggml+binary IN-CONTAINER (self-consistent
# GLIBC) WITH CUDA graphs ON, and has --cap-add SYS_ADMIN + ncu work for profiling. CUDA 12.4 matches
# the host toolchain (cuda_12.4.r12.4) and the RTX 3060 (sm_86). Prod-representative.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake make build-essential libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
