#!/usr/bin/env bash
# STANDARD nsys profiling harness for the LTX engine (2-seg continuation memory profile).
# Solves the recurring CUDA-13 nsys importer hell: the BUILDER's bundled nsys (2026.2) captures
# richer data (full 2-seg) but its importer is broken; the CUDA-repo standalone (cuda-nsight-systems-13-0
# = 2025.3.2) has a WORKING importer + auto-exports .nsys-rep. Two modes:
#   MODE=capture (default) — run under the standalone 2025.3.2 nsys -> readable prof.nsys-rep directly.
#   MODE=capture2026       — run under the builder's 2026 nsys (richer) -> prof.qdstrm (needs a 2026
#                            host tool / nsys-ui to import; keep for when 2025.3.2 misses something).
#   MODE=analyze REP=x     — run the standard memory reports on an existing .nsys-rep.
# Usage: SEGMENTS=2 STEPS=2 REFSTEPS=1 bash nsys_profile.sh
set -u
WT=/home/dbrain/dev/longcat-avatar-ltxdenoise
LTX2=/home/dbrain/dev/longcat-avatar.cpp/models/ltx2
BUILDER=longcat-avatar-dev:builder-cudnn-ff
BIN=/src/build-cudnn/bin/sd-cli
DIT=${DIT:-nvfp4-CLEAN-dev050.gguf}
OUT=$WT/_ablation_out/nsys_prof; MODE=${MODE:-capture}
SEGMENTS=${SEGMENTS:-2}; STEPS=${STEPS:-2}; REFSTEPS=${REFSTEPS:-1}   # reduced steps: shapes built, small trace
mkdir -p "$OUT"

INSTALL_STD='apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq cuda-nsight-systems-13-0 >/dev/null 2>&1; NSYS=$(ls /opt/nvidia/nsight-systems/*/target-linux-x64/nsys | head -1)'

if [ "$MODE" = analyze ]; then
  REP=${REP:-$OUT/prof.nsys-rep}
  docker run --rm -v "$WT:/src" -w /src/_ablation_out/nsys_prof nvidia/cuda:13.3.0-devel-ubuntu24.04 bash -c "
    export DEBIAN_FRONTEND=noninteractive; $INSTALL_STD
    echo '=== GPU MemOps by size ==='; \$NSYS stats --report cuda_gpu_mem_size_sum --format table $(basename "$REP")
    echo '=== CUDA API (malloc/cuMem/cudnn) ==='; \$NSYS stats --report cuda_api_sum --format table $(basename "$REP") | grep -iE 'Malloc|MemCreate|MemAlloc|cudnn|Name'
    echo '=== export sqlite for unfreed-alloc query ==='; \$NSYS export --type sqlite --force-overwrite true -o prof.sqlite $(basename "$REP") 2>&1 | tail -1
    echo '  then query CUDA_GPU_MEMORY_USAGE_EVENTS: allocations (bytes>500MB) with no matching free = the reserve'
  "
  exit 0
fi

# capture modes: reduced-step SEGMENTS-way chain (shapes identical to prod, small trace)
NSYS_SETUP='NSYS=$(which nsys)'; [ "$MODE" = capture ] && NSYS_SETUP="$INSTALL_STD"
PRODENV=( -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1 -e LTX_DIT_F16=1
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks
  -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=0 -e LONGCAT_FFN_TILE_TOKENS=4096 -e LONGCAT_ENCODE_MAX_VRAM=6.5
  -e LTXAV_END_RENDER_RECLAIM=1 -e LTXAV_CHAIN_POOL_TRIM=1 -e LONGCAT_VRAM_BREAKDOWN=1
  -e LTXAV_VAE_LAZY=1 -e LTXAV_DIT_FREE_DURING_DECODE=1
  -e LTX_VAE_HEAD_F32=1 -e LTX_VAE_CONV3D_WTILES=16 -e LTX_VAE_CONV3D_HTILES=8 -e LTX_VAE_DECODE_F16=1
  -e LTX_VAE_SPATIAL_TILES=2x2 -e LTX_VAE_SPATIAL_OVERLAP=4 )
INNER="$NSYS_SETUP
\"\$NSYS\" profile --trace=cuda,cudnn --cuda-memory-usage=true --backtrace=dwarf --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /src/_ablation_out/nsys_prof/prof \
  stdbuf -oL -eL $BIN -M vid_gen --diffusion-model /ltx2/$DIT \
  --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
  --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  --lora-model-dir /ltx2/loras -p 'city street, woman in red raincoat crosses a zebra crossing, overcast' \
  --ltx-chain-segments $SEGMENTS -W 960 -H 544 --video-frames 121 --fps 24 --steps $STEPS --cfg-scale 1.0 --diffusion-fa \
  --hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-upscalers-dir /ltx2/latent_upscale_models --hires-steps $REFSTEPS --hires-sigmas '0.85,0.0' \
  --offload-to-cpu --mmap --max-vram 7 -s 42 -v -o /src/_ablation_out/nsys_prof/out.webm"
docker run --rm --gpus '"device=1"' --cap-add=SYS_ADMIN "${PRODENV[@]}" -e LTX_CUSTOM_SIGMAS='1.0,0.5,0.0' \
  -v "$WT:/src" -v "$LTX2:/ltx2" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src "$BUILDER" bash -c "$INNER"
echo "rc=$?  report: $OUT/prof.nsys-rep  (analyze: MODE=analyze REP=$OUT/prof.nsys-rep bash nsys_profile.sh)"

# ============================================================================
# THE IMPORTER FIX (the whole recurring pain): the builder's bundled nsys 2026.2
# (under /opt/nvidia/nsight-compute/2026.2.0/host/) captures RICH data (full 2-seg)
# but its QdstrmImporter fails with "unknown error" — because the minimal container
# is missing ONE lib: libdw.so.1. Install libdw1 and the SAME-VERSION importer works
# on the ORIGINAL .qdstrm (no regenerate, no version mismatch):
#   docker run --rm -v $WT:/src -w /src/_ablation_out/nsys_prof \
#     --entrypoint bash longcat-avatar-dev:builder-cudnn-ff -c '
#       apt-get update -qq && apt-get install -y libdw1 sqlite3
#       IMP=/opt/nvidia/nsight-compute/2026.2.0/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter
#       "$IMP" --input-file prof.qdstrm          # -> prof.nsys-rep
#       nsys export --type sqlite --force-overwrite true -o prof.sqlite prof.nsys-rep
#       # the analysis (memory table = CUDA_GPU_MEMORY_USAGE_EVENTS; oper 0=alloc 1=free):
#       sqlite3 prof.sqlite "SELECT (a.bytes/1048576) MB,count(*) n FROM CUDA_GPU_MEMORY_USAGE_EVENTS a
#         WHERE a.memoryOperationType=0 AND a.bytes>100*1048576 AND NOT EXISTS
#         (SELECT 1 FROM CUDA_GPU_MEMORY_USAGE_EVENTS f WHERE f.memoryOperationType=1
#          AND f.address=a.address AND f.start>a.start) GROUP BY a.bytes ORDER BY MB DESC;"
#     '
# RESULT (2026-07-07): unfreed >100MB = 216x2 + 151 + 108x2 = ~799MB, pc=0 (no ggml stack)
# => the +800 reserve is cuDNN-INTERNAL cached workspace. Not ggml, not the mempool, not a leak.
# ============================================================================
