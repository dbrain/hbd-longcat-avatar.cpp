# Geometry-shootout image: StdGEN + UltraShape in ONE image, two ISOLATED conda envs.
# Rationale: the two repos pin mutually-incompatible stacks (StdGEN = py3.9 / torch2.1+cu118;
# UltraShape = py3.10 / torch2.5+cu121 / flash_attn / cubvh). One image + two envs gives a single
# `docker rmi` cleanup with no host venv mess and no dep collision. Weights are NOT baked in —
# they mount from /mnt/hdd at run time (see fetch_weights.sh). Build context is this dir.
#
#   docker build -f Dockerfile -t geom-shootout .
#   # weights on /mnt/hdd, repos cloned under /mnt/hdd/3d/avatar-shootout
#   docker run --rm --gpus all --cap-add SYS_ADMIN \
#     -v /mnt/hdd/3d/avatar-shootout:/work -v /mnt/hdd/3d/avatar-shootout/_weights:/weights \
#     geom-shootout  conda run -n ultrashape  bash /work/UltraShape-1.0/scripts/run.sh   # etc.
#
# Base already local (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel, 20.9GB) → provides conda +
# cuda12.4 nvcc (builds flash_attn/cubvh/pytorch3d) + a torch2.5.1 base env reused by UltraShape.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_INPUT=1 \
    TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=6
RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs build-essential cmake ninja-build \
      libgl1 libglib2.0-0 libegl1 libgles2 libosmesa6 ffmpeg \
    && rm -rf /var/lib/apt/lists/* && git lfs install

# ───────────────────────── env 1: UltraShape (py3.10 / torch2.5 / cu12) ─────────────────────────
# Refine-only path needs just UltraShape (NOT Hunyuan3D-2.1 — that's only for its own coarse stage;
# we feed it our pixal3d/StdGEN coarse mesh via --mesh). flash_attn/cubvh/pytorch3d build from source
# against the cu124 torch (driver 610.x supports it); torch stays the base 2.5.1.
RUN conda create -y -n ultrashape python=3.10 && conda clean -ya
SHELL ["conda", "run", "--no-capture-output", "-n", "ultrashape", "/bin/bash", "-c"]
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
COPY ultrashape_requirements.txt /tmp/us_req.txt
RUN pip install -r /tmp/us_req.txt
RUN pip install --no-build-isolation flash_attn==2.8.3 || pip install flash_attn==2.8.3
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
RUN pip install --no-build-isolation "git+https://github.com/ashawkey/cubvh" \
 && pip install https://data.pyg.org/whl/torch-2.5.0+cu121/torch_cluster-1.6.3+pt25cu121-cp310-cp310-linux_x86_64.whl

# ───────────────────────── env 2: StdGEN (py3.9 / torch2.1 / cu118) ─────────────────────────
SHELL ["/bin/bash", "-c"]
RUN conda create -y -n stdgen python=3.9 && conda clean -ya
SHELL ["conda", "run", "--no-capture-output", "-n", "stdgen", "/bin/bash", "-c"]
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 \
 && pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118 \
 && pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
COPY stdgen_requirements.txt /tmp/sg_req.txt
RUN pip install -r /tmp/sg_req.txt \
 && pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

SHELL ["/bin/bash", "-c"]
WORKDIR /work
# Usage: conda run -n {ultrashape|stdgen} <command>.  EGL offscreen render needs --cap-add SYS_ADMIN.
CMD ["bash"]
