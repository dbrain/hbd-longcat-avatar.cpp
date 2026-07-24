#!/usr/bin/env bash
# Serialize an ad-hoc CUDA command on the reserved physical RTX 3060.
#
# Production runners pin themselves already. Use this wrapper for native parity tests, diagnostics,
# and one-off probes so a bare CUDA process can never land on the owner's 5060.
# Usage: run_on_3060.sh <command> [args...]
set -euo pipefail

[[ $# -gt 0 ]] || { echo "usage: run_on_3060.sh <command> [args...]" >&2; exit 2; }
OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"
mkdir -p "$OUT_ROOT"
GPU_3060_UUID="${IMAGE_TO_RIG_GPU_3060_UUID:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {uuid=$1} END {print uuid}')}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_3060_UUID" 2>/dev/null | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_3060_UUID' is '$GPU_NAME', expected the reserved RTX 3060" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_3060_UUID"
exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
flock -n 9 || { echo "another image-to-rig job owns the 3060" >&2; exit 75; }
exec "$@"
