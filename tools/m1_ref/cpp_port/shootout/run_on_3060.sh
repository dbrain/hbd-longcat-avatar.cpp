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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: PCI GPU 0 is '$GPU_NAME', expected RTX 3060" >&2; exit 1; }
exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
flock -n 9 || { echo "another image-to-rig job owns the 3060" >&2; exit 75; }
exec "$@"
