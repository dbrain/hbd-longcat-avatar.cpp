#!/usr/bin/env bash
# P3-SAM (Hunyuan3D-Part) env build — native 3D part segmentation, "handles any input mesh".
# Goal: segment dense pixal3d Miku → isolate hands → feed part-importance into the patched-IM sizing
# field (semantic density, vs the curvature-only field we already have). Backbone = VENDORED facebook
# Sonata (XPart/partgen/models/sonata; deps = spconv + torch_scatter + flash_attn) + chamfer3D CUDA ext.
# Deps are AUTHORITATIVE from XPart/requirements.txt: cu124 (NOT the README's stale cu121), py3.10.
# Build is CPU/nvcc-compile (no GPU); the segmentation RUN needs GPU.
set -uo pipefail
PY310=/home/dbrain/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10  # uv's 3.10 (sys py=3.14, no wheels)
REPO=/mnt/hdd/3d/avatar-shootout/Hunyuan3D-Part/P3-SAM
cd "$REPO"
LOG="$REPO/_setup.log"; : > "$LOG"
say(){ echo "=== $* ===" | tee -a "$LOG"; }
pf(){ "$@" 2>>"$LOG" | tail -2; }

say "fresh venv on uv py3.10 ($("$PY310" --version 2>&1))"
rm -rf .venv; "$PY310" -m venv .venv
. .venv/bin/activate
pf pip install -U pip wheel setuptools

say "torch 2.4.0 + cu124 (XPart/requirements.txt)"
pf pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

say "pyg sparse ops (prebuilt pt24cu124 cp310 wheels)"
pf pip install https://data.pyg.org/whl/torch-2.4.0+cu124/torch_scatter-2.1.2+pt24cu124-cp310-cp310-linux_x86_64.whl
pf pip install https://data.pyg.org/whl/torch-2.4.0+cu124/torch_cluster-1.6.3+pt24cu124-cp310-cp310-linux_x86_64.whl

say "XPart/Sonata deps"
pf pip install "numpy==2.1.2" addict spconv-cu124 scikit-learn fpsample "pymeshlab==2023.12.post3" \
   easydict timm huggingface_hub safetensors scipy packaging viser trimesh numba gradio einops

say "flash_attn (Sonata attention path; ~30-45min build if no wheel)"
pf pip install flash-attn --no-build-isolation || say "flash-attn FAILED — Sonata may fall back to non-flash; check at run time"

say "chamfer3D CUDA ext"
( cd utils/chamfer3D && pf python setup.py install ) || say "chamfer3D build FAILED"

say "fetch P3-SAM + Sonata weights"
mkdir -p weights
pf huggingface-cli download tencent/Hunyuan3D-Part --local-dir weights || pf hf download tencent/Hunyuan3D-Part --local-dir weights || say "weight fetch FAILED"

say "import check"
PYTHONPATH="$REPO:$REPO/../XPart/partgen" python -c "import torch,trimesh,spconv.pytorch,torch_scatter; print('torch',torch.__version__,'cu',torch.version.cuda); from models import sonata; print('sonata vendored OK')" 2>>"$LOG" | tee -a "$LOG" || say "IMPORT CHECK FAILED — see $LOG"
echo "log: $LOG"
