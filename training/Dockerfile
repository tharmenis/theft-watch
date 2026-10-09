# Training container. Works on Turing (e.g. T1000) and newer NVIDIA GPUs.
# Uses official PyTorch CUDA 12.8 wheels, which bundle their own CUDA runtime
# and include kernels from Turing (sm_75) to Blackwell (sm_120). Newer NGC
# PyTorch images have dropped Turing, so we do not use them here.
# Other CUDA build: docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 trainer
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace
